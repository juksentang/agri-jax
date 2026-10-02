"""Minimal reproduction of the G2 / G3 gradient gap of the DSSAT-CSM 4.8.6 maize day (UFGA8201 t4).

What is compared: on one weather-year x sowing scenario of ``exp.scenarios(treatment=4, ...)``
(published cultivar, nitrogen off, float64), the forward-mode derivative (``jax.jvp`` of the jitted
season scan) of every float leaf of the day's state on every day, against the central finite difference
of the same jitted function at a step ``frac * (MAXIMA - MINIMA)`` of the coefficient. Run in the
gradient modes ``ste`` (the facade's) and ``exact``.

    python scripts/diag/dssat_grad_gap.py --year 1979 --shift 0 --param G2   # first departing day
    python scripts/diag/dssat_grad_gap.py --sweep    # 9 scenarios: ste / exact / per-site AD vs FD steps
    python scripts/diag/dssat_grad_gap.py --facade   # the same scenarios through scen.sensitivity

Prints the end-of-season HWAM derivative at several steps, then the first day and state leaf where
the AD tangent departs from the finite difference (relative to the leaf's own scale).
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def build(year: int, shift: int):
    import agrijax as aj
    from agrijax.port.run_fortran import DSSAT_ENGINE

    exp = aj.dssat.experiment("UFGA8201", data_root=DSSAT_ENGINE)
    scen = exp.scenarios(treatment=4, years=[year], sowing_shift=[shift])
    return scen.runs[0], scen.published


def season_fn(run, n_pad: int = 60):
    """``f(theta [6]) -> (flat float state per day [T, N], names)`` of one season."""
    import equinox as eqx

    from agrijax.calib.dssat_day import _FIELDS, CUL_ORDER
    from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_processes

    model = day_dssat486(SLOT).compile(day_processes(SLOT, mesev=run.mesev), outputs=None, exact_lags=True)
    step = model.compile()
    n = run.n_days + n_pad
    params0 = run.params()
    forcing = run.forcing(n)
    state0 = run.state(params0)
    names = [_FIELDS[k] for k in CUL_ORDER]

    leaves, _ = jax.tree_util.tree_flatten_with_path(state0)
    float_idx = [i for i, (_, x) in enumerate(leaves) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)]
    labels = []
    for i in float_idx:
        p, x = leaves[i]
        labels += [f"{jax.tree_util.keystr(p)}[{j}]" for j in range(int(np.size(x)))]

    def flat(s):
        ls = jax.tree_util.tree_leaves(s)
        return jnp.concatenate([jnp.ravel(ls[i]).astype(jnp.float64) for i in float_idx])

    def f(theta):
        c = params0["crop"].cultivar
        vals = tuple(
            theta[i].astype(getattr(c, nm).dtype).reshape(getattr(c, nm).shape) for i, nm in enumerate(names)
        )
        cul = eqx.tree_at(lambda x: tuple(getattr(x, nm) for nm in names), c, vals)
        crop = eqx.tree_at(lambda x: x.cultivar, params0["crop"], cul)
        params = {**params0, "crop": crop}

        def body(s, t):
            s2, _ = step(s, params, jax.tree.map(lambda x: x[t], forcing))
            return s2, flat(s2)

        _, ys = jax.lax.scan(body, state0, jnp.arange(n))
        return ys

    return f, labels


#: the straight-through quantisation sites of the day: (module, helper name, what it quantises)
STE_SITES = {
    "rlv": ("agrijax.processes.crop.ceres_maize.roots", "trunc_st", "RLV to 1e-3 (MZ_ROOTS)"),
    "turfac": ("agrijax.processes.crop.ceres_maize.stress", "trunc_st", "TURFAC to 1e-3 (MZ_GROSUB)"),
    "sw": ("agrijax.processes.soil_water.bucket.kernels", "round_st", "SW to 1e-6 (WATBAL)"),
    "stalk": ("agrijax.processes.crop.ceres_maize.canopy", "round_st", "grain weight NINT (canopy)"),
}


def exact_at(sites):
    """Context: the helpers of ``sites`` traced in the ``exact`` mode, the others as they are."""
    import contextlib
    import functools
    import importlib

    @contextlib.contextmanager
    def ctx():
        saved = []
        for s in sites:
            mod, name, _ = STE_SITES[s]
            m = importlib.import_module(mod)
            saved.append((m, name, getattr(m, name)))
            setattr(m, name, functools.partial(getattr(m, name), mode="exact"))
        try:
            yield
        finally:
            for m, name, f in saved:
                setattr(m, name, f)

    return ctx()


def sweep(years, shifts, params) -> None:
    """End grain weight: AD (ste, exact, ste with each quantisation site exact) and central
    differences at several steps, on every scenario."""
    from agrijax.calib.ceres import CERES_SPECS
    from agrijax.calib.dssat_day import CUL_ORDER
    from agrijax.core.grad import bind_gradient_mode

    fracs = (1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 2e-2)
    for yr in years:
        for sh in shifts:
            run, pub = build(yr, sh)
            x = np.asarray([pub[k] for k in CUL_ORDER], float)
            f, labels = season_fn(run)
            gw = labels.index("['crops']['maize'].growth.grnwt[0]")
            fj = jax.jit(lambda th, f=f, gw=gw: f(th)[-1, gw])
            y0 = float(fj(jnp.asarray(x)))
            for p in params:
                k = CUL_ORDER.index(p)
                w = CERES_SPECS[p].upper - CERES_SPECS[p].lower
                e = jnp.asarray(np.eye(6)[k])
                ad = {}
                variants = [("ste", "ste", ()), ("exact", "exact", ())]
                variants += [(f"ste-{s}", "ste", (s,)) for s in STE_SITES]
                for lab, mode, sites in variants:
                    with exact_at(sites):
                        g = jax.jit(
                            bind_gradient_mode(
                                lambda th, f=f, gw=gw, e=e: jax.jvp(lambda z: f(z)[-1, gw], (th,), (e,))[1],
                                mode,
                            )
                        )
                        ad[lab] = float(g(jnp.asarray(x)))
                fd = {}
                for fr in fracs:
                    h = fr * w
                    fd[fr] = (
                        float(fj(jnp.asarray(x + h * np.eye(6)[k])))
                        - float(fj(jnp.asarray(x - h * np.eye(6)[k])))
                    ) / (2 * h)
                rel = lambda a, b: (a - b) / abs(b)  # noqa: E731
                print(
                    f"{yr} {sh:+3d} {p:3s} y={y0:9.4f} " + " ".join(f"{k_}={v:.7g}" for k_, v in ad.items())
                )
                print("      FD " + " ".join(f"{fr:.0e}:{v:.7g}" for fr, v in fd.items()))
                print(
                    f"      rel(ste vs FD1e-8)={rel(ad['ste'], fd[1e-8]):+.4f}"
                    f" rel(exact vs FD1e-8)={rel(ad['exact'], fd[1e-8]):+.2e}"
                    f" rel(ste vs FD1e-5)={rel(ad['ste'], fd[1e-5]):+.4f}"
                    f" rel(ste vs FD2e-2)={rel(ad['ste'], fd[2e-2]):+.4f}"
                    f" rel(exact vs FD2e-2)={rel(ad['exact'], fd[2e-2]):+.4f}",
                    flush=True,
                )


def facade(years, shifts) -> None:
    """The facade's view: ``scen.sensitivity`` of HWAM / CWAM to G2 / G3 (201 scan points) on every
    scenario, with the straight-through (``ad``) and exact-mode (``ad_exact``) derivatives."""
    import agrijax as aj
    from agrijax.port.run_fortran import DSSAT_ENGINE

    exp = aj.dssat.experiment("UFGA8201", data_root=DSSAT_ENGINE)
    scen = exp.scenarios(treatment=4, years=list(years), sowing_shift=list(shifts))
    sens = scen.sensitivity(outputs=["HWAM", "CWAM"], params=["G2", "G3"])
    t = sens.table.copy()
    t["ste/exact-1"] = t["ad"] / t["ad_exact"] - 1.0
    cols = ["year", "sowing_shift", "output", "param", "ad", "ad_exact", "ste/exact-1", "fd"]
    cols += ["err_small", "err_large", "class", "level", "jumps", "trust"]
    print(t[cols].to_string(float_format=lambda v: f"{v:.5g}"))
    print(sens.summary.to_string())
    print(sens.trust)


def main() -> None:
    from agrijax.calib.ceres import CERES_SPECS
    from agrijax.calib.dssat_day import CUL_ORDER
    from agrijax.core.grad import bind_gradient_mode

    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=1979)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--param", default="G2")
    ap.add_argument("--frac", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--facade", action="store_true")
    a = ap.parse_args()
    if a.facade:
        facade((1979, 1982, 1985), (-14, 0, 14))
        return
    if a.sweep:
        sweep((1979, 1982, 1985), (-14, 0, 14), ("G2", "G3"))
        return

    run, pub = build(a.year, a.shift)
    x = np.asarray([pub[k] for k in CUL_ORDER], float)
    k = CUL_ORDER.index(a.param)
    w = CERES_SPECS[a.param].upper - CERES_SPECS[a.param].lower
    e = np.eye(6)[k]
    f, labels = season_fn(run)
    print(
        f"scenario {a.year} {a.shift:+d} d, {a.param} = {x[k]}, width {w}, {len(labels)} float state values"
    )

    fj = jax.jit(f)
    jv = {
        m: jax.jit(bind_gradient_mode(lambda th: jax.jvp(f, (th,), (jnp.asarray(e),)), m))
        for m in ("ste", "exact")
    }
    y0 = np.asarray(fj(jnp.asarray(x)))
    tang = {m: np.asarray(jv[m](jnp.asarray(x))[1]) for m in jv}
    gw = labels.index(next(lab for lab in labels if "gwad" in lab.lower() or "grnwt" in lab.lower()))
    print("grain-weight leaf used:", labels[gw])

    def fd(frac):
        h = frac * w
        return (np.asarray(fj(jnp.asarray(x + h * e))) - np.asarray(fj(jnp.asarray(x - h * e)))) / (2 * h)

    print(f"{'step':>9} {'FD(end grain leaf)':>20}")
    for frac in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2e-2):
        print(f"{frac:9.0e} {fd(frac)[-1, gw]:20.10g}")
    for m in tang:
        print(f"AD {m:5s}  {tang[m][-1, gw]:20.10g}")

    d = fd(a.frac)
    scale = np.maximum(np.abs(y0).max(axis=0), 1e-12)
    for m in tang:
        err = np.abs(tang[m] - d) / (np.maximum(np.abs(d), np.abs(tang[m])) + 1e-9 * scale)
        bad = np.argwhere(err > a.rtol)
        if bad.size == 0:
            print(f"[{m}] AD equals FD on every day and leaf (rtol {a.rtol})")
            continue
        t0 = bad[:, 0].min()
        print(f"[{m}] first departing day index {t0}; leaves departing that day:")
        for j in bad[bad[:, 0] == t0][:, 1][:25]:
            print(f"   {labels[j]:60s} y={y0[t0, j]:.10g} AD={tang[m][t0, j]:.6g} FD={d[t0, j]:.6g}")
        if t0 > 0:
            print("   previous day (inputs of that day), nonzero tangents:")
            nz = np.nonzero(np.abs(tang[m][t0 - 1]) > 0)[0]
            for j in nz[:40]:
                print(
                    f"   {labels[j]:60s} y={y0[t0 - 1, j]:.10g} "
                    f"AD={tang[m][t0 - 1, j]:.6g} FD={d[t0 - 1, j]:.6g}"
                )


if __name__ == "__main__":
    main()
