"""Diagnostics (cluster): gradient mode vs jit, vmap-vs-single rounding, IUAF9901 G2/G3 jumps.

``python scripts/bench/calib/d3prep_diag.py [--out DIR]`` (needs ``prep.pkl`` of
``ceres_twin.py prep`` under ``--out``, default ``$AGRI_JAX_DATA/validation/aj_dcal``). Writes
``d3prep_diag.json`` there and prints a summary:

``modes``
    IUAF9901 treatment 3, ``d CWAD / d RUE`` at the INP cultivar: a plain ``jax.jit`` of a
    mode-unbound simulator first traced in ``ste`` and then differentiated in ``exact`` (the
    pitfall), a fresh exact jit, per-mode bound simulators, and the central finite difference.
``ulp``
    The synthetic seasons of ``tests/unit/test_calib_ceres.py``: batch group vs single runs, the
    largest difference of LAI / CWAD / GWAD in units in the last place.
``jumps``
    Line scans of G2 and G3 (the trust report's grid: truth +- 10 % of the bound width, 41 points)
    on the IUAF9901 yield and final biomass, with, at every grid point and treatment, the counts of
    the stage-5 switches of ``grain_fill_growth`` (stem at its floor SWMIN, leaf-to-stem transfer
    below 1.07 SWMIN, stem capped at SWMAX), the early-maturity events (EMAT / CMAT) and the stage
    dates, so that each jump interval can be matched to the switch that changed across it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "bench" / "calib"))
sys.path.insert(0, str(ROOT))

import ceres_twin as ct  # type: ignore[import-not-found]  # noqa: E402

from agrijax.calib import TrustConfig, line_scan  # noqa: E402
from agrijax.calib.ceres import calib_outputs, ceres_space, group_simulator, stack_treatments  # noqa: E402
from agrijax.core import run  # noqa: E402
from agrijax.core.grad import ModeMismatchError, gradient_mode  # noqa: E402
from agrijax.core.units import KG_HA_PER_G_M2  # noqa: E402
from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model  # noqa: E402

#: the stage-5 constants of MZ_GROSUB.for (as the port's GrosubCoefficients defaults)
SWMIN_MARGIN = 1.07
STEM_SPLIT = 0.5
CARBO_ACTIVE = 1e-4
ISTAGE_EFG = 5


def diag_outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
    g, ph = state.growth, state.phen
    return {
        "gwad": g.grnwt * ph.ears * KG_HA_PER_G_M2,
        "cwad": g.biomas * KG_HA_PER_G_M2,
        "istage": ph.istage,
        "sumdtt": ph.sumdtt,
        "stmwt": g.stmwt,
        "swmin": g.swmin,
        "swmax": g.swmax,
        "lfwt": g.leaf.mass[..., 0],
        "carbo": g.carbo,
        "grogrn": g.grogrn,
        "emat": g.emat,
        "cmat": g.cmat,
        "gpp": ph.gpp,
    }


def _unbound_simulator(space, params, forcing, state0):
    """``group_simulator`` without a gradient-mode binding (for the pitfall)."""
    model = ceres_maize_model(outputs=calib_outputs)

    def one(theta, p, f, s0):
        out = run(model, space.apply(p, theta), f, s0)
        return {k: v[..., 0] for k, v in out.items()}

    batched = jax.vmap(one, in_axes=(None, 0, 0, 0), out_axes=1)
    return lambda theta: batched(theta, params, forcing, state0)


def part_modes(prob) -> dict[str, Any]:
    t3 = [t for t in prob if t.name.endswith("_t3")]
    space = ceres_space(("RUE",))
    n = int(np.shape(t3[0].forcing.yrdoy)[0]) + ct.PAD_DAYS
    params, forcing, state0 = stack_treatments(t3, n)
    th = jnp.asarray(np.asarray(space.get(t3[0].params)))

    def d_cwad(sim):
        return float(jax.grad(lambda t: sim(t)["cwad"][-1, 0])(th)[0])

    plain = jax.jit(_unbound_simulator(space, params, forcing, state0))
    with gradient_mode("ste"):
        plain_ste = d_cwad(plain)
    with gradient_mode("exact"):
        plain_exact_cached = d_cwad(plain)
        fresh_exact = d_cwad(jax.jit(_unbound_simulator(space, params, forcing, state0)))
    bound = {m: group_simulator(space, params, forcing, state0, mode=m).jit() for m in ("ste", "exact")}
    try:
        with gradient_mode("exact"):
            d_cwad(bound["ste"])
        bound_ste_in_exact = "no error"
    except ModeMismatchError as e:
        bound_ste_in_exact = f"ModeMismatchError: {e}"
    h = 1e-5 * float(space.width[0])
    f = bound["ste"]
    fd = (float(f(th + h)["cwad"][-1, 0]) - float(f(th - h)["cwad"][-1, 0])) / (2 * h)
    return {
        "treatment": t3[0].name,
        "plain_jit_ste": plain_ste,
        "plain_jit_then_exact_context": plain_exact_cached,
        "fresh_jit_exact": fresh_exact,
        "bound_ste": d_cwad(bound["ste"]),
        "bound_exact": d_cwad(bound["exact"]),
        "bound_ste_called_in_exact_context": bound_ste_in_exact,
        "central_fd_1e-5": fd,
    }


def part_ulp() -> dict[str, Any]:
    from tests.unit.test_calib_ceres import SPACE, TREATMENTS  # type: ignore[import-not-found]

    params, forcing, state0 = stack_treatments(TREATMENTS, n_days=270)
    out = jax.jit(group_simulator(SPACE, params, forcing, state0, mode="ste"))(
        SPACE.get(TREATMENTS[0].params)
    )
    model = ceres_maize_model(outputs=calib_outputs)
    res: dict[str, Any] = {}
    for b, t in enumerate(TREATMENTS):
        ref = jax.jit(lambda p, f, s: run(model, p, f, s))(
            t.params, t.forcing, CeresMaizeState.initial(t.params, 1)
        )
        n = int(np.shape(t.forcing.yrdoy)[0])
        for k in ("lai", "cwad", "gwad"):
            a = np.asarray(out[k])[:n, b]
            r = np.asarray(ref[k])[:, 0]
            scale = np.spacing(np.maximum(np.abs(a), np.abs(r)))
            nz = np.maximum(np.abs(a), np.abs(r)) > 0
            ulp = np.where(nz, np.abs(a - r) / np.where(nz, scale, 1.0), 0.0)
            rel = np.where(nz, np.abs(a - r) / np.where(nz, np.maximum(np.abs(a), np.abs(r)), 1.0), 0.0)
            res[f"{t.name}/{k}"] = {
                "max_ulp": float(ulp.max()),
                "max_rel": float(rel.max()),
                "n_days_differ": int(np.sum(a != r)),
            }
    return res


def _flags(o: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per grid point and treatment: counts of the stage-5 switches and the stage dates."""
    st = o["istage"]  # [P, T, B]
    prev5 = np.zeros_like(st, dtype=bool)
    prev5[:, 1:] = st[:, :-1] == ISTAGE_EFG
    stm_prev = np.concatenate([o["stmwt"][:, :1], o["stmwt"][:, :-1]], axis=1)
    carbo, gro = o["carbo"], o["grogrn"]
    active = np.abs(carbo) > CARBO_ACTIVE
    floor = prev5 & active & (o["stmwt"] == o["swmin"])
    pos = carbo - gro >= 0.0
    expected = np.where(pos, stm_prev + (carbo - gro) * STEM_SPLIT, stm_prev + carbo - gro)
    thin = prev5 & active & (floor | (~pos & (expected <= o["swmin"] * SWMIN_MARGIN)))
    cap = prev5 & active & (o["stmwt"] == o["swmax"]) & (o["swmax"] > 0)

    def first(code: int) -> np.ndarray:
        hit = st == code
        return np.where(hit.any(axis=1), hit.argmax(axis=1), -1)

    end5 = first(6)
    sumdtt_end5 = np.take_along_axis(o["sumdtt"], np.maximum(end5, 0)[:, None, :], axis=1)[:, 0]
    return {
        "n_floor": floor.sum(axis=1),
        "n_thin": thin.sum(axis=1),
        "n_cap": cap.sum(axis=1),
        "n_deficit": (prev5 & active & ~pos).sum(axis=1),
        "stage5_days": prev5.sum(axis=1),
        "day_end5": end5,
        "day_mat": first(10),
        "sumdtt_end5": sumdtt_end5,
        "emat_max": o["emat"].max(axis=1),
        "cmat_max": o["cmat"].max(axis=1),
        "gpp": o["gpp"][:, -1],
    }


def part_jumps(prob, names=("G2", "G3")) -> dict[str, Any]:
    space, sim, truth = ct.build(prob, mode="ste")
    n = max(int(np.shape(t.forcing.yrdoy)[0]) for t in prob) + ct.PAD_DAYS
    params, forcing, state0 = stack_treatments(prob, n)
    dsim = jax.jit(
        jax.vmap(group_simulator(space, params, forcing, state0, outputs=diag_outputs, mode="ste"))
    )
    cfg = TrustConfig()
    b_names = [t.name for t in prob]

    def f(theta):
        o = sim(theta)
        return jnp.concatenate([o["gwad"][-1], o["cwad"][-1]])

    out: dict[str, Any] = {}
    with gradient_mode("ste"):
        for pn in names:
            i = list(space.names).index(pn)
            w = float(space.width[i])
            lo = max(float(space.lower[i]), truth[i] - cfg.scan_frac * w)
            hi = min(float(space.upper[i]), truth[i] + cfg.scan_frac * w)
            sc = line_scan(f, truth, i, lo, hi, cfg, width=w)
            grid = sc["grid"]
            pts = np.repeat(truth[None], grid.size, axis=0)
            pts[:, i] = grid
            o = {k: np.asarray(v) for k, v in dsim(jnp.asarray(pts)).items()}
            fl = _flags(o)
            y = sc["y"]
            dx = np.diff(grid)[:, None]
            unexpl = np.abs(np.diff(y, axis=0) - 0.5 * (sc["g"][1:] + sc["g"][:-1]) * dx)
            rng = y.max(axis=0) - y.min(axis=0)
            jump = unexpl > cfg.jump_frac * np.maximum(
                rng, cfg.abs_floor * np.maximum(np.abs(y).max(axis=0), 1.0)
            )
            per: dict[str, Any] = {}
            for b, tn in enumerate(b_names):
                nb = len(b_names)
                ks = sorted(set(np.nonzero(jump[:, b] | jump[:, nb + b])[0].tolist()))
                intervals = []
                for k in ks:
                    changed = {
                        key: [fl[key][k, b].item(), fl[key][k + 1, b].item()]
                        for key in fl
                        if key not in ("gpp", "sumdtt_end5") and fl[key][k, b] != fl[key][k + 1, b]
                    }
                    intervals.append(
                        {
                            "k": k,
                            "theta": [float(grid[k]), float(grid[k + 1])],
                            "yield": [float(y[k, b]), float(y[k + 1, b])],
                            "yield_unexplained": float(unexpl[k, b]),
                            "cwad_unexplained": float(unexpl[k, len(b_names) + b]),
                            "changed": changed,
                        }
                    )
                per[tn] = {
                    "n_jumps_yield": int(jump[:, b].sum()),
                    "n_jumps_cwad": int(jump[:, len(b_names) + b].sum()),
                    "intervals": intervals,
                    "scan": {key: fl[key][:, b].tolist() for key in fl},
                }
            out[pn] = {"grid": grid.tolist(), "treatments": per}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = ct.out_dir(args)
    problems = ct.load_problems(out)
    prob = problems["IUAF9901"]
    res = {"env": ct.env_info(), "modes": part_modes(prob), "ulp": part_ulp(), "jumps": part_jumps(prob)}
    print(json.dumps({"modes": res["modes"], "ulp": res["ulp"]}, indent=1))
    for pn, r in res["jumps"].items():
        for tn, t in r["treatments"].items():
            print(f"{pn} {tn}: {t['n_jumps_yield']} yield jumps, {t['n_jumps_cwad']} cwad jumps")
            for iv in t["intervals"]:
                print(
                    f"   {pn} {iv['theta'][0]:.4f}->{iv['theta'][1]:.4f} yield {iv['yield'][0]:.2f}->"
                    f"{iv['yield'][1]:.2f} unexplained {iv['yield_unexplained']:.2f}: {iv['changed']}"
                )
    (out / "d3prep_diag.json").write_text(json.dumps(res, default=float))
    print("wrote", out / "d3prep_diag.json")


if __name__ == "__main__":
    main()
