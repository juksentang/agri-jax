"""GPU vs CPU check of the CA-TPA S-W PET pipeline at 10^5 parameter sets (one year).

What it does:

1. Inputs: the CA-TPA 2015 forcing and site parameters from the private data tree when
   ``<data>/narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario`` and ``<data>/validation/catpa_2015_ref``
   exist (``source = "CA-TPA 2015"``), otherwise a deterministic synthetic 365-day forcing
   (``source = "synthetic"``).
2. ``n`` parameter sets: the six :class:`~agrijax.processes.pet.PETParams` drawn by Latin
   hypercube (seeded), sample 0 is the unperturbed site parameter set.
3. GPU: :func:`agrijax.core.run_batch_chunked` over all ``n`` sets; wall time of the compile (AOT
   ``lower().compile()``), of the first run and of a repeat run; peak device memory from
   ``memory_stats``.
4. CPU on the same node, through different code paths: :func:`run_batch` (``vmap``, no ``lax.map``)
   over the first ``n_compare`` sets, and :func:`run` (plain ``lax.scan``) on sample 0. Max abs and
   max rel difference of every output against the GPU result.
5. Conservation on the full GPU batch: the running totals on the last day equal the sum of the
   daily fluxes (the sum is taken on the host in NumPy float64, independent of the scan).

Run: ``python tests/gpu/pipeline_check.py --n 100000 --out result.json``. Also imported
by ``tests/gpu/test_batch_pipeline.py`` (``check()``), so the test and the job run the same code.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core import run, run_batch, run_batch_chunked
from agrijax.models import catpa_pet_demo as demo
from agrijax.processes.pet import PETParams, PETSiteParams

SCENARIO = Path("narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario")
REF = Path("validation/catpa_2015_ref")
YEAR = ("2015-01-01", "2015-12-31")
#: LHS ranges of the six PET parameters (plausible physical ranges, not calibrated).
RANGES: dict[str, tuple[float, float]] = {
    "albedo_dry": (0.15, 0.35),
    "albedo_wet": (0.08, 0.20),
    "albedo_maturity": (0.18, 0.28),
    "albedo_residue": (0.20, 0.40),
    "soil_resistance": (20.0, 150.0),
    "stomatal_resistance": (100.0, 400.0),
}


def _synthetic() -> tuple[PETSiteParams, demo.DemoForcing, Any]:
    f = lambda x: jnp.asarray(x, dtype=float)  # noqa: E731
    params = PETSiteParams(
        pet=PETParams(f(0.25), f(0.15), f(0.23), f(0.30), f(54.0), f(224.0)),
        elevation=f(200.0),
        latitude=f(0.745163),
        wc13=f(0.255198),
        wc15=f(0.141628),
        wind_height=f(2.0),
        albedo_soil=f(0.13),
        trat=f(1.0),
        rainfall_zone=3,
    )
    t = np.arange(365, dtype=float)
    season = np.sin(2 * np.pi * (t - 100.0) / 365.0)
    lai = np.clip(4.0 * np.sin(np.pi * (t - 120.0) / 120.0), 0.0, None) * ((t > 120) & (t < 240))
    forcing = demo.DemoForcing(
        tmin=jnp.asarray(5.0 + 10.0 * season + 2.0 * np.sin(1.3 * t)),
        tmax=jnp.asarray(15.0 + 12.0 * season + 3.0 * np.cos(0.9 * t)),
        srad=jnp.asarray(np.clip(16.0 + 10.0 * season + 3.0 * np.sin(0.4 * t), 1.0, None)),
        rh=jnp.asarray(65.0 + 20.0 * np.sin(0.7 * t)),
        wind_run=jnp.asarray(150.0 + 60.0 * np.cos(0.5 * t)),
        doy=jnp.asarray(t + 1.0),
        lai=jnp.asarray(lai),
        height_cm=jnp.asarray(50.0 * lai),
        residue_mass=jnp.asarray(np.where(t < 250, 3000.0, 1500.0)),
        residue_age=jnp.asarray(demo.RESIDUE_AGE0 + t),
    )
    return params, forcing, demo.initial_state(0.2)


def load_inputs(data_dir: Path | None) -> tuple[PETSiteParams, demo.DemoForcing, Any, str]:
    """Real CA-TPA 2015 inputs when the data tree has them, else the synthetic year."""
    if data_dir is not None:
        scen, ref = data_dir / SCENARIO, data_dir / REF
        if (
            (scen / "rzwqm.dat").is_file()
            and (ref / "CA-TPA.ana").is_file()
            and (ref / "LAYER.PLT").is_file()
        ):
            from agrijax.io.rzwqm import read_ana, read_met, read_rzwqm_dat
            from agrijax.io.rzwqm.layers import read_layer_output

            met = read_met(scen / "CA-TPA.MET", prepare=True)
            ana = read_ana(ref / "CA-TPA.ana")
            theta = read_layer_output(ref / "LAYER.PLT", start=YEAR[0])["soil_water_content"]
            theta1 = float(np.asarray(theta.isel(depth=0).values, dtype=float)[0])
            dat = read_rzwqm_dat(scen / "rzwqm.dat")
            params = demo.site_params_from_dat(dat, scen / "rzwqm.dat")
            forcing = demo.build_forcing(met, ana, *YEAR)
            return params, forcing, demo.initial_state(theta1), "CA-TPA 2015"
    p, f, s = _synthetic()
    return p, f, s, "synthetic"


def lhs_params(base: PETSiteParams, n: int, seed: int = 20150101) -> PETSiteParams:
    """``n`` sets: the six PET parameters by Latin hypercube, sample 0 = ``base``; the rest broadcast."""
    rng = np.random.default_rng(seed)
    batched = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x))), base)
    for name, (lo, hi) in RANGES.items():
        u = (rng.permutation(n) + rng.random(n)) / n
        v = lo + (hi - lo) * u
        v[0] = float(getattr(base.pet, name))
        batched = batched.set(f"pet.{name}", jnp.asarray(v))
    return batched


def _take(tree: Any, sl: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x[sl], tree)


def _gpu_device() -> Any | None:
    try:
        return next(d for d in jax.devices() if d.platform != "cpu")
    except (StopIteration, RuntimeError):
        return None


def check(
    n: int = 100_000,
    *,
    n_compare: int = 256,
    chunk: int = 8192,
    data_dir: Path | None = None,
    seed: int = 20150101,
    repeat: bool = True,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    """Run the GPU/CPU comparison; returns a JSON-serialisable summary (see the module docstring).

    ``allow_cpu=True`` runs the "GPU" side on the CPU when no GPU is visible (a dry run of the
    script's plumbing before submitting, doc 05 section 6); the numbers are then CPU vs CPU.
    """
    gpu = _gpu_device()
    if gpu is None:
        if not allow_cpu:
            raise RuntimeError("no GPU device visible to JAX")
        gpu = jax.devices("cpu")[0]
    cpu = jax.devices("cpu")[0]
    base, forcing, state0, source = load_inputs(data_dir)
    params = lhs_params(base, n, seed)
    model = demo.catpa_pet_model()
    n_compare = min(n_compare, n)

    put = lambda tree, dev: jax.device_put(tree, dev)  # noqa: E731
    p_g, f_g, s_g = put(params, gpu), put(forcing, gpu), put(state0, gpu)

    def conservation_error(outs: dict[str, Any]) -> Any:  # on device; only one scalar leaves
        tt = jnp.max(
            jnp.abs(outs["totals.transpiration"][:, -1] - jnp.sum(outs["pet.transpiration"], axis=1))
        )
        ev = outs["pet.soil_evaporation"] + outs["pet.residue_evaporation"]
        te = jnp.max(jnp.abs(outs["totals.evaporation"][:, -1] - jnp.sum(ev, axis=1)))
        return jnp.maximum(tt, te)

    with jax.default_device(gpu):
        # run_batch_chunked builds a fresh jit per call; wrap it once so compile and run are timed apart
        fn = jax.jit(lambda p, f, s: run_batch_chunked(model, p, f, s, chunk=chunk))
        t0 = time.perf_counter()
        compiled = fn.lower(p_g, f_g, s_g).compile()
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        outs_g = jax.block_until_ready(compiled(p_g, f_g, s_g))
        t_first = time.perf_counter() - t0
        t_cached = None
        if repeat:
            del outs_g
            t0 = time.perf_counter()
            outs_g = jax.block_until_ready(compiled(p_g, f_g, s_g))
            t_cached = time.perf_counter() - t0
        stats = gpu.memory_stats() or {}
        head_g = jax.device_get(_take(outs_g, slice(0, n_compare)))
        cons_gpu = float(conservation_error(outs_g))
        annual_t = np.asarray(jax.device_get(outs_g["totals.transpiration"][:, -1]))
        # host-side float64 sums of the head (independent of the on-device reduction)
        cons_host = max(
            float(
                np.max(np.abs(head_g["totals.transpiration"][:, -1] - np.sum(head_g["pet.transpiration"], 1)))
            ),
            float(
                np.max(
                    np.abs(
                        head_g["totals.evaporation"][:, -1]
                        - np.sum(head_g["pet.soil_evaporation"] + head_g["pet.residue_evaporation"], 1)
                    )
                )
            ),
        )
        n_nonfinite = int(sum(int(jnp.sum(~jnp.isfinite(v))) for v in outs_g.values()))
        del outs_g

    with jax.default_device(cpu):
        p_c, f_c, s_c = put(_take(params, slice(0, n_compare)), cpu), put(forcing, cpu), put(state0, cpu)
        t0 = time.perf_counter()
        head_c = jax.device_get(jax.block_until_ready(run_batch(model, p_c, f_c, s_c)))
        t_cpu = time.perf_counter() - t0
        single_c = jax.device_get(jax.jit(lambda p: run(model, p, f_c, s_c))(put(base, cpu)))

    diffs: dict[str, dict[str, float]] = {}
    for k in head_c:
        a, b = np.asarray(head_g[k], dtype=float), np.asarray(head_c[k], dtype=float)
        d = np.abs(a - b)
        diffs[k] = {
            "max_abs": float(d.max()),
            "max_rel": float((d / np.maximum(np.abs(b), 1e-300)).max()),
            "max_abs_sample0_vs_run": float(np.max(np.abs(a[0] - np.asarray(single_c[k], dtype=float)))),
        }
    return {
        "source": source,
        "n": n,
        "n_days": int(forcing.tmin.shape[0]),
        "chunk": chunk,
        "n_compare": n_compare,
        "x64": bool(jax.config.jax_enable_x64),
        "jax": jax.__version__,
        "gpu": str(gpu.device_kind),
        "wall_gpu_compile_s": t_compile,
        "wall_gpu_run_s": t_first,
        "wall_gpu_run_repeat_s": t_cached,
        "wall_cpu_run_batch_head_s": t_cpu,
        "gpu_peak_bytes": int(stats.get("peak_bytes_in_use", -1)),
        "gpu_bytes_limit": int(stats.get("bytes_limit", -1)),
        "host_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "max_abs_diff": max(v["max_abs"] for v in diffs.values()),
        "max_rel_diff": max(v["max_rel"] for v in diffs.values()),
        "diffs": diffs,
        "conservation_max_abs_gpu": cons_gpu,
        "conservation_max_abs_host_head": cons_host,
        "n_nonfinite": n_nonfinite,
        "annual_transpiration_cm": {
            "min": float(annual_t.min()),
            "median": float(np.median(annual_t)),
            "max": float(annual_t.max()),
            "sample0": float(annual_t[0]),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--n-compare", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--data-dir", default=os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
    ap.add_argument("--out", default=None, help="write the summary JSON here")
    ap.add_argument("--allow-cpu", action="store_true", help="dry run on the CPU when no GPU is visible")
    a = ap.parse_args(argv)
    # x64 unless JAX_ENABLE_X64=0; set here, not at import, so pytest's own setting wins in the test
    jax.config.update("jax_enable_x64", os.environ.get("JAX_ENABLE_X64", "1").lower() not in {"0", "false"})
    print(jax.__version__, jax.devices(), flush=True)
    res = check(
        a.n,
        n_compare=a.n_compare,
        chunk=a.chunk,
        data_dir=Path(a.data_dir).expanduser(),
        allow_cpu=a.allow_cpu,
    )
    txt = json.dumps(res, indent=2)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
