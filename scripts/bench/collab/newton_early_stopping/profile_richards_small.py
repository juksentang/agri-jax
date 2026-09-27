"""Small CPU benchmark of the production Richards solver; run from any directory.

Uses the repository's existing 37-node fixtures, not the computational skeleton.
Independent microbenchmarks are diagnostic timings, not additive time shares.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "agri-jax"
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from agrijax.processes.soil_water import richards as R
from tests.unit.test_richards import nodes, step_args
from tests.integration.test_richards_catpa import Catpa2015, THETA_INIT


def measure(name, function, arguments, output, reps=15):
    start = time.perf_counter()
    executable = jax.jit(function).lower(*arguments).compile()
    compile_s = time.perf_counter() - start
    for _ in range(3):
        jax.block_until_ready(executable(*arguments))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        value = jax.block_until_ready(executable(*arguments))
        samples.append(time.perf_counter() - start)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(value)), name
    row = dict(compile_s=compile_s, median_s=statistics.median(samples),
               min_s=min(samples), max_s=max(samples), runs_s=samples)
    output[name] = row
    print(name, json.dumps({k:v for k,v in row.items() if k != "runs_s"}), flush=True)
    return executable, value


def capture(name, executable, arguments, destination):
    destination.mkdir(parents=True, exist_ok=True)
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 3
    options.python_tracer_level = 0
    # Profiling overhead is excluded from all timings above.
    with jax.profiler.trace(str(destination), profiler_options=options):
        with jax.profiler.TraceAnnotation(name):
            jax.block_until_ready(executable(*arguments))
    (destination / "optimized_hlo.txt").write_text(executable.as_text(), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--mode", choices=["unrolled", "implicit"], default="unrolled")
    parser.add_argument("--no-trace", action="store_true", help="Avoid detailed trace memory overhead")
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    case = Catpa2015(out / "collab_inputs/data/agri_jax_data")
    soil, grid = case.soil, case.grid
    theta0 = np.full(grid.n_node, THETA_INIT)
    supply, evaporation, uptake = case.supply[:args.days], case.evap_hourly[:args.days], case.uptake[:args.days]
    forcing = R.RichardsForcing(supply=jnp.asarray(supply), evaporation=jnp.asarray(evaporation),
                               uptake=jnp.asarray(uptake))
    result = dict(environment=dict(python=sys.version, jax=jax.__version__, platform=platform.platform(),
                                   processor=platform.processor(), logical_cpus=os.cpu_count(),
                                   devices=[str(x) for x in jax.devices()], x64=True),
                  case=dict(nodes=37, days=args.days, n_sub=24, batch=1,
                            dates=[str(case.days[0]), str(case.days[args.days-1])],
                            source="production richards_day; collab pack CA-TPA 2015 reference flux inputs",
                            loss="final storage + 10 * sum(drainage + evaporation)"), timings={})
    saved = {}
    result_file = out / f"results_iter{args.iterations}_{args.mode}.json"
    for iterations, mode in [(args.iterations, args.mode)]:
        name = f"iter{iterations}_{mode}"
        cfg = R.RichardsConfig(n_sub=24, n_iter=iterations, grad=mode)
        def loss(s, cfg=cfg):
            params = R.RichardsParams(soil=s, grid=grid, config=cfg)
            def body(w, f):
                w = R.richards_day(w, params, f.supply, f.evaporation, f.uptake)
                return w, 10.0 * (w.flux.drainage + w.flux.evaporation)
            w, daily = jax.lax.scan(body, R.SoilWater.from_theta(jnp.asarray(theta0), s), forcing)
            return w.storage(grid) + jnp.sum(daily)
        forward, value = measure(name + "_forward", loss, (soil,), result["timings"], args.reps)
        vg, pair = measure(name + "_value_grad", jax.value_and_grad(loss), (soil,), result["timings"], args.reps)
        np.testing.assert_allclose(value, pair[0], rtol=1e-12)
        checks = []
        for field, horizon in [("ksat", 0), ("lambda_", 2), ("hb", 4)]:
            x = getattr(soil, field)
            delta = 1e-5 * abs(float(x[horizon]))
            plus = soil.replace(**{field: x.at[horizon].add(delta)})
            minus = soil.replace(**{field: x.at[horizon].add(-delta)})
            fd = float((forward(plus) - forward(minus)) / (2 * delta))
            ad = float(getattr(pair[1], field)[horizon])
            checks.append(dict(field=field, horizon=horizon, ad=ad, fd=fd,
                               relative_error=abs(ad-fd)/max(abs(fd), 1e-12)))
        result[name] = dict(loss=float(value), finite_difference_checks=checks)
        saved[name] = (forward, vg, pair)
        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    gradient = saved[name][2][1]
    result["gradient_leaves"] = [np.asarray(x).tolist() for x in jax.tree.leaves(gradient)]
    # Check residuals independently; these diagnostic outputs are not added to the timed objective.
    for iterations in (args.iterations,):
        params = R.RichardsParams(soil=soil, grid=grid, config=R.RichardsConfig(n_sub=24, n_iter=iterations))
        def diagnostics(s):
            p = params.replace(soil=s)
            def body(w, f):
                w = R.richards_day(w, p, f.supply, f.evaporation, f.uptake)
                return w, (w.flux.max_theta_residual, w.flux.balance_error)
            return jax.lax.scan(body, R.SoilWater.from_theta(jnp.asarray(theta0), s), forcing)[1]
        try:
            residual, balance = jax.block_until_ready(jax.jit(diagnostics)(soil))
            result[f"diagnostics_iter{iterations}"] = dict(max_theta_residual=float(jnp.max(residual)),
                max_abs_daily_balance_cm=float(jnp.max(jnp.abs(balance))))
        except AttributeError as exc:
            result[f"diagnostics_iter{iterations}"] = {"error": str(exc)}

    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.no_trace:
        for name in saved:
            forward, vg, _ = saved[name]
            capture(name + "_forward", forward, (soil,), out / "traces" / (name + "_forward"))
            capture(name + "_value_grad", vg, (soil,), out / "traces" / (name + "_value_grad"))

    # These stand-alone measurements have dispatch/fusion differences: never sum them as percentages.
    sn = nodes(soil, 37)
    w = R.SoilWater.from_theta(jnp.asarray(theta0), sn)
    a = step_args(w.h, w.theta, sn, grid, q_demand=0.01)
    residual = lambda h, a: R.richards_residual(h, h, a)
    assemble = lambda h, a: R.tridiagonal_jacobian(lambda x: residual(x, a), h)
    micro = result["independent_microbenchmarks_not_additive"] = {}
    measure("residual", residual, (w.h, a), micro, 101)
    _, bands = measure("residual_and_jacobian", assemble, (w.h, a), micro, 101)
    r, dl, d, du = bands
    measure("tridiagonal_solve", R._tridiag_solve, (dl, d, du, r), micro, 101)
    result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULTS", str(result_file), flush=True)


if __name__ == "__main__":
    main()
