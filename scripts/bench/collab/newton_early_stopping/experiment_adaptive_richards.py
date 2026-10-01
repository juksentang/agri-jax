"""Bounded early stopping + checked implicit VJP, isolated from production code.

Run variants in separate processes. Only the process-local implicit solver is
replaced; equations, forcing, time grid, and production files are unchanged.
The optional --audit callbacks are excluded from timing runs.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import sys

from profile_richards_small import (Catpa2015, R, ROOT, THETA_INIT, jax, jnp,
                                    measure, np)
from agrijax.processes.soil_water import fixed_cn


def install_adaptive(tol, audit=None):
    """Install this experiment's Newton-only solver in the current process."""
    def iterate(cfg, h0, a):
        if cfg.jacobian != "newton":
            raise ValueError("This experiment supports Newton only")
        s = a.soil.hb
        lo = R.v_of_head(jnp.broadcast_to(a.h_min, h0.shape), s)
        hi = R.v_of_head(jnp.broadcast_to(a.h_hi, h0.shape), s)
        reg_h = a.tl * cfg.c_floor / a.dt

        def residual(v):
            h = R.head_of_v(v, s)
            return R.richards_residual(h, h, a)

        def bands(v):
            return R.tridiagonal_jacobian(residual, v)

        def error(r):
            return jnp.max(jnp.abs(r) * a.dt / a.tl)

        def body(carry):
            v, nclamp, i, (r, dl, d, du) = carry
            d = d + reg_h * R._dh_dv(v, s)
            dv = jnp.clip(R._tridiag_solve(dl, d, du, r), -cfg.dv_max, cfg.dv_max)
            v_raw = v - dv
            if cfg.chop:
                v_raw = jnp.where((v > 0.0) & (v_raw < 0.0), 0.0, v_raw)
            clamped = (v_raw < lo) | (v_raw > hi)
            v_new = jnp.clip(v_raw, lo, hi)
            return v_new, nclamp + jnp.sum(clamped).astype(h0.dtype), i + 1, bands(v_new)

        v0 = R.v_of_head(h0, s)
        # Reuse the residual returned by the colored JVPs for convergence.
        # One final Jacobian assembly is the cost of testing the updated state.
        init = (v0, jnp.zeros((), h0.dtype), jnp.int32(0), bands(v0))
        v, nclamp, count, final_bands = jax.lax.while_loop(
            lambda c: (c[2] < cfg.n_iter) & jnp.isfinite(error(c[3][0])) & (error(c[3][0]) > tol),
            body, init)
        err = error(final_bands[0])
        if audit is not None:
            jax.debug.callback(lambda i, e: audit["forward"].append(
                dict(iterations=int(i), residual=float(e))), count, err, ordered=True)
        return R.head_of_v(v, s), nclamp

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def solve(cfg, h0, a):
        return iterate(cfg, h0, a)

    def fwd(cfg, h0, a):
        h, nclamp = iterate(cfg, h0, a)
        return (h, nclamp), (h, a)

    def bwd(cfg, res, g):
        h, a = res
        gh = g[0]
        r, dl, d, du = R.tridiagonal_jacobian(lambda x: R.richards_residual(x, x, a), h)
        zero = jnp.zeros_like(d[:1])
        dl_t = jnp.concatenate([zero, du[:-1]])
        du_t = jnp.concatenate([dl[1:], zero])
        lam = R._tridiag_solve(dl_t, d, du_t, gh)
        # Componentwise scaled backward error; no regularization of the physical J.
        lower = jnp.concatenate([zero, lam[:-1]])
        upper = jnp.concatenate([lam[1:], zero])
        product = dl_t * lower + d * lam + du_t * upper
        scale = jnp.abs(dl_t * lower) + jnp.abs(d * lam) + jnp.abs(du_t * upper) + jnp.abs(gh)
        adj_err = jnp.max(jnp.abs(product - gh) / jnp.maximum(scale, jnp.finfo(h.dtype).tiny))
        root_err = jnp.max(jnp.abs(r) * a.dt / a.tl)
        _, pullback = jax.vjp(lambda aa: R.richards_residual(h, h, aa), a)
        (a_bar,) = pullback(-lam)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in
            [h, r, dl, d, du, gh, lam, *jax.tree.leaves(a_bar)]]))
        valid = finite & (root_err <= tol) & jnp.isfinite(adj_err) & (adj_err <= 1e-10)
        if audit is not None:
            jax.debug.callback(lambda e, ok: audit["backward"].append(
                dict(adjoint_backward_error=float(e), accepted=bool(ok))), adj_err, valid, ordered=True)
        # A rejected solve cannot silently provide a usable gradient to an optimizer.
        reject = lambda x: jnp.where(valid, x, jnp.full_like(x, jnp.nan))
        return reject(jnp.zeros_like(h)), jax.tree.map(reject, a_bar)

    solve.defvjp(fwd, bwd)
    # The fixed-step solver lives in fixed_cn; richards_step resolves
    # _solve_implicit there, so the patch must go on that module (R keeps a re-export).
    fixed_cn._solve_implicit = solve
    R._solve_implicit = solve


def make_objective(case, cfg, days, diagnostics=False):
    forcing = R.RichardsForcing(supply=jnp.asarray(case.supply[:days]),
        evaporation=jnp.asarray(case.evap_hourly[:days]), uptake=jnp.asarray(case.uptake[:days]))
    theta0 = jnp.full(case.grid.n_node, THETA_INIT)

    def objective(soil):
        params = R.RichardsParams(soil=soil, grid=case.grid, stepping=cfg)
        def body(w, f):
            w = R.richards_day(w, params, f.supply, f.evaporation, f.uptake)
            daily = 10 * (w.flux.drainage + w.flux.evaporation)
            return w, ((daily, w.flux.max_theta_residual, w.flux.balance_error)
                       if diagnostics else daily)
        w, outputs = jax.lax.scan(
            body, R.SoilWater.from_theta(theta0, soil), forcing)
        if not diagnostics:
            return w.storage(case.grid) + jnp.sum(outputs)
        daily, residuals, balances = outputs
        return w.storage(case.grid) + jnp.sum(daily), dict(
            max_theta_residual=jnp.max(residuals),
            max_abs_daily_balance_cm=jnp.max(jnp.abs(balances)),
            final_head=w.h, final_theta=w.theta)
    return objective


def accept(value_aux, gradient, tol):
    value, aux = value_aux
    finite = all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves((value_aux, gradient)))
    if not finite or float(aux["max_theta_residual"]) > tol:
        raise ValueError("Rejected: nonfinite result/gradient or unconverged forward solve")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["fixed", "adaptive", "unrolled"], default="adaptive")
    parser.add_argument("--tol", type=float, default=1e-14)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--reps", type=int, default=31)
    parser.add_argument("--tag", default="")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--expect-rejection", action="store_true")
    args = parser.parse_args()
    if args.tol <= 0 or args.max_iter < 1 or args.days < 1 or args.reps < 1:
        parser.error("tol, max-iter, days, and reps must be positive")
    if args.expect_rejection and args.variant != "adaptive":
        parser.error("--expect-rejection is for adaptive validation")
    out = Path(__file__).resolve().parent
    case = Catpa2015(out / "collab_inputs/data/agri_jax_data")
    audit = {"forward": [], "backward": []} if args.audit else None
    if args.variant == "adaptive":
        install_adaptive(args.tol, audit)
    cfg = R.FixedStepping(n_sub=24, n_iter=args.max_iter,
        grad="unrolled" if args.variant == "unrolled" else "implicit")
    objective = make_objective(case, cfg, args.days, diagnostics=True)
    result = dict(config=vars(args), environment=dict(jax=jax.__version__,
        devices=[str(d) for d in jax.devices()], x64=True, python=sys.version),
        case=dict(nodes=case.grid.n_node, days=args.days, n_sub=24, batch=1,
            dates=[str(case.days[0]), str(case.days[args.days-1])],
            source="CA-TPA 2015 reference infiltration, evaporation, uptake",
            objective="final storage + 10 * sum(drainage + evaporation)"), timings={})
    if args.audit or args.expect_rejection:
        forward = jax.jit(objective)
        value_aux, gradient = jax.block_until_ready(jax.jit(
            jax.value_and_grad(objective, has_aux=True))(case.soil))
        jax.effects_barrier()
    else:
        # Match the earlier benchmark's scalar objective. Diagnostic arrays and
        # callbacks are collected separately, not included in the timed outputs.
        scalar = make_objective(case, cfg, args.days)
        _, fwd_result = measure("forward", scalar, (case.soil,), result["timings"], args.reps)
        _, (value, gradient) = measure("value_and_grad", jax.value_and_grad(scalar),
            (case.soil,), result["timings"], args.reps)
        forward = jax.jit(objective)
        value_aux = jax.block_until_ready(forward(case.soil))
        np.testing.assert_allclose(fwd_result, value_aux[0], rtol=1e-13)
        np.testing.assert_allclose(value, value_aux[0], rtol=1e-13)
    try:
        accept(value_aux, gradient, args.tol)
        result["accepted"] = True
    except ValueError:
        result["accepted"] = False
        if not args.expect_rejection:
            raise
    if args.expect_rejection:
        assert not result["accepted"], "Failure test unexpectedly accepted a gradient"
        assert any(not np.all(np.isfinite(x)) for x in jax.tree.leaves(gradient))
    value, aux = value_aux
    result.update(loss=float(value), diagnostics={k: np.asarray(v).tolist() for k, v in aux.items()})
    if result["accepted"]:
        result["gradient_leaves"] = [np.asarray(x).tolist() for x in jax.tree.leaves(gradient)]
    if audit is not None:
        result["audit"] = audit
        counts = [x["iterations"] for x in audit["forward"]]
        result["iteration_summary"] = dict(steps=len(counts), total=sum(counts),
            minimum=min(counts), maximum=max(counts), mean=float(np.mean(counts)),
            histogram={str(n): counts.count(n) for n in sorted(set(counts))})
        print("AUDIT", json.dumps(result["iteration_summary"]), flush=True)
    elif result["accepted"]:
        checks = []
        for field, horizon in [("ksat", 0), ("lambda_", 2), ("hb", 4)]:
            x = getattr(case.soil, field)
            for relative_step in (1e-4, 1e-5):
                delta = relative_step * abs(float(x[horizon]))
                fp, ap = forward(case.soil.replace(**{field: x.at[horizon].add(delta)}))
                fm, am = forward(case.soil.replace(**{field: x.at[horizon].add(-delta)}))
                accept((fp, ap), (), args.tol)
                accept((fm, am), (), args.tol)
                fd = float((fp - fm) / (2 * delta))
                ad = float(getattr(gradient, field)[horizon])
                checks.append(dict(field=field, horizon=horizon, relative_step=relative_step,
                    ad=ad, fd=fd, relative_error=abs(ad-fd)/max(abs(fd), 1e-12),
                    passed=bool(np.isclose(ad, fd, rtol=1e-4, atol=1e-9))))
        result["finite_difference_checks"] = checks
    suffix = "_audit" if args.audit else "_rejection" if args.expect_rejection else ""
    dest = out / f"adaptive_{args.variant}_n{args.max_iter}_tol{args.tol:g}{suffix}{args.tag}.json"
    dest.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("RESULT", dest, "accepted=", result["accepted"], flush=True)
    assert all(c["passed"] for c in result.get("finite_difference_checks", [])), "Finite differences failed"


if __name__ == "__main__":
    main()
