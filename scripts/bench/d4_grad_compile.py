"""Compile time and memory of the reverse-mode program of the free-run DSSAT day (``d4_jax_day.py``).

Measurement only. Task: the 65-run season mix at ``B`` seasons, split over the six (layer count,
MESEV) group programs as in ``d4_jax_scaling.py --group mix``, f64. Each group's program is
``jax.grad`` of the summed final grain weight with respect to the per-sample multipliers of five
cultivar parameters (P1, P5, G2, G3, PHINT; the soil multipliers are held fixed), through the
day loop with the day step under ``jax.checkpoint`` (as ``agrijax.core.runtime.run_and_grad`` does
by default). Reported per group: lowering, XLA compile, first call and the median of
``--repeats`` steady calls (``block_until_ready``), ``compiled.memory_analysis()`` (argument,
output, temporary bytes) and the device's ``peak_bytes_in_use`` after the group (cumulative over
the process). The layer-loop layout follows the execution setting ``depth_unroll``
(``agrijax.core.execution``; ``AGRI_JAX_DEPTH_UNROLL=0|1`` overrides the backend default); run one
process per setting.

Usage::

    python scripts/bench/d4_grad_compile.py --B 10000 --tag gpu_on
    AGRI_JAX_DEPTH_UNROLL=0 python scripts/bench/d4_grad_compile.py --B 10000 --tag gpu_off
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
OUT = DATA / "validation" / os.environ.get("AJ_DBENCH_DIR", "aj_dbench_grad")
INPUTS = DATA / "validation" / "aj_dbench_d41" / "inputs" / "d4_inputs.pkl"
#: number of leading multipliers of ``d4_jax_day.PERTURBED`` differentiated (the cultivar ones)
N_CULTIVAR = 5


def _groups(items: list[dict[str, Any]], B: int) -> dict[tuple[int, str], int]:
    """The ``--group mix`` allocation (largest remainder), as in ``d4_jax_scaling.py``."""
    groups: dict[tuple[int, str], int] = Counter((it["nl"], it["mesev"]) for it in items)
    n_all = len(items)
    exact = {k: B * groups[k] / n_all for k in sorted(groups)}
    alloc = {k: int(exact[k]) for k in exact}
    for k in sorted(exact, key=lambda k: -(exact[k] - alloc[k]))[: B - sum(alloc.values())]:
        alloc[k] += 1
    return {k: v for k, v in alloc.items() if v > 0}


def _loss(D: Any, model: Any, n_days: int) -> Any:
    """Summed final grain weight of the batch as a function of the cultivar multipliers."""
    import jax
    import jax.numpy as jnp

    step = jax.checkpoint(jax.vmap(model.compile(), in_axes=(0, 0, 0)))

    def loss(cul: Any, params_tr: Any, forcing_tr: Any, state_tr: Any, tid: Any, soil_mult: Any) -> Any:
        m = jnp.concatenate([cul, soil_mult], axis=1)
        params = D.perturb(jax.tree.map(lambda x: x[tid], params_tr), m)
        state0 = jax.tree.map(lambda x: x[tid], state_tr)

        def body(s: Any, t: Any) -> Any:
            f_t = D.perturb_day_soil(jax.tree.map(lambda x: x[tid, t], forcing_tr), m)
            return step(s, params, f_t)

        _, outs = jax.lax.scan(body, state0, jnp.arange(n_days))
        return jnp.sum(outs["gwad"][-1])

    return loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    import numpy as np

    from agrijax.core.execution import current_execution

    sys.path.insert(0, str(HERE))
    import d4_jax_day as D

    items = D.load(INPUTS)
    by: dict[tuple[int, str], list[Any]] = {}
    for it in items:
        by.setdefault((it["nl"], it["mesev"]), []).append(it)
    dev = jax.devices()[0]
    res: dict[str, Any] = {
        "device": dev.device_kind,
        "backend": jax.default_backend(),
        "jax": jax.__version__,
        "execution": str(current_execution()),
        "B": a.B,
        "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "groups": [],
    }
    for key, Bg in _groups(items, a.B).items():
        g = D.group_inputs(by[key], np.float64)
        n_tr, n_days = len(g["keys"]), g["n_days"]
        rng = np.random.default_rng(0)
        mult = rng.uniform(1 - D.SPREAD, 1 + D.SPREAD, size=(Bg, len(D.PERTURBED)))
        tid = (np.arange(Bg) % n_tr).astype(np.int32)
        model = D.bench_model(key[1]).model
        grad = jax.grad(_loss(D, model, n_days))
        dargs = jax.device_put(
            (mult[:, :N_CULTIVAR], g["params"], g["forcing"], g["state"], tid, mult[:, N_CULTIVAR:])
        )
        t0 = time.perf_counter()
        lowered = jax.jit(grad).lower(*dargs)
        t1 = time.perf_counter()
        compiled = lowered.compile()
        t2 = time.perf_counter()
        gr = jax.block_until_ready(compiled(*dargs))
        t3 = time.perf_counter()
        times = []
        for _ in range(a.repeats):
            s0 = time.perf_counter()
            jax.block_until_ready(compiled(*dargs))
            times.append(time.perf_counter() - s0)
        rec: dict[str, Any] = {
            "group": f"nl{key[0]}_{key[1]}",
            "B_group": Bg,
            "n_days": n_days,
            "lower_s": t1 - t0,
            "compile_s": t2 - t1,
            "first_call_s": t3 - t2,
            "steady_s": float(np.median(times)),
            "grad_sum": float(np.sum(np.asarray(gr))),
            "grad_finite": bool(np.all(np.isfinite(np.asarray(gr)))),
        }
        try:
            ma = compiled.memory_analysis()
            rec["memory_analysis"] = {
                k: int(getattr(ma, k))
                for k in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "generated_code_size_in_bytes",
                )
                if hasattr(ma, k)
            }
        except Exception as ex:
            rec["memory_analysis"] = str(ex)
        try:
            rec["peak_bytes_in_use"] = int((dev.memory_stats() or {}).get("peak_bytes_in_use", -1))
        except Exception as ex:
            rec["peak_bytes_in_use"] = str(ex)
        print(json.dumps(rec), flush=True)
        res["groups"].append(rec)
        del dargs, compiled, lowered, gr
    gs = res["groups"]
    res["total"] = {
        "lower_s": sum(r["lower_s"] for r in gs),
        "compile_s": sum(r["compile_s"] for r in gs),
        "steady_s": sum(r["steady_s"] for r in gs),
        "max_temp_bytes": max(r["memory_analysis"].get("temp_size_in_bytes", -1) for r in gs),
        "peak_bytes_in_use": max(r["peak_bytes_in_use"] for r in gs),
    }
    print(json.dumps({k: v for k, v in res.items() if k != "groups"}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"grad_{a.tag}.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
