"""The free-run acceptance on a GPU in float64 with XLA's default flags: the 65 runs (58 M2
maize treatments and the 7 CA-TPA seasons) give the CPU float64 result, and pass the acceptance
(harvest yield within 2 % of HWAM where HWAM > 0, emergence / silking / maturity dates as the
reference).

Why: the bucket stores its water content in REAL*4 (WATBAL.for:503-505), and the REAL*4 equality
tests of the day (``SW .LE. LL`` in ROOTWU and XTRACT) depend on it. Written as a float64 ->
float32 -> float64 convert pair, the store was removed by XLA's GPU pipeline (default
``--xla_allow_excess_precision=true``): on an H100 25 of 65 runs differed from CPU float64 by more
than 1e-6, CTPA1501 t02 by 20 % in yield, and one maturity date moved. The store
is now :func:`agrijax.core.grad.real4_store` (``reduce-precision``), which no backend removes.

The GPU pass runs with the GPU's execution settings (:mod:`agrijax.core.execution`: the bucket's
layer recurrences unrolled), the CPU pass with the CPU's (loops); a third pass on the GPU with the
loops (``execution(depth_unroll=False)``) must give the bucket's outputs and the stages bit for
bit and every other output to rounding (:func:`test_gpu_unrolled_matches_gpu_loops`).

Only with a GPU visible to JAX (skipped otherwise); run on rorqual with ``remote.sh --gpu``. Both
passes run in the same process, the CPU one on JAX's host backend. The inputs are those of the
``free`` configuration of :mod:`day_dssat486_free_harness`; the report goes to
``<data-dir>/validation/aj_dint/d2_1a_free_run_gpu.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import day_dssat486_free_harness as h
import jax
import numpy as np
import pytest

from agrijax.core.execution import Execution, execution
from agrijax.core.runtime import _RUNNERS_ATTR, run_sites
from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_outputs, day_processes

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.allow_skip(
        reason="needs a GPU visible to JAX, dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT), the DSSAT dump "
        "tables (WATBAL, SPAM, ROOTWU) and the CA-TPA DSSAT case"
    ),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the float64 acceptance"),
]

#: GPU float64 against CPU float64: the same program on the same inputs; what remains is the
#: order of floating-point operations the two backends choose (fused multiply-add, reductions).
#: Measured with the REAL*4 store surviving (flag off): at most 1.4e-15 relative in yield.
GPU_CPU_YIELD_REL = 1e-9
#: the same for the end-of-day layer water content [cm3 cm-3]
GPU_CPU_SW_ATOL = 1e-9
#: GPU unrolled against GPU loops (the execution setting ``depth_unroll``): the same model, the same
#: backend; only XLA's fusion of the code around the layer loops differs. Measured (H100,
#: whole GPU and MIG slice): <= 2.0e-16 relative in yield, <= 3.7e-12 in every series
#: (kg ha-1 for the masses), soil water, runoff and drainage bit for bit.
GPU_LOOPS_YIELD_REL = 1e-12
GPU_LOOPS_SERIES_ATOL = 1e-9
#: the bucket's own outputs, bit for bit under both layouts
BUCKET_SERIES = ("soil_sw", "runoff", "drain")
#: the daily series compared between the two backends
SERIES = ("gwad", "cwad", "lai", "soil_sw", "es", "ep", "runoff", "drain")


def _gpu() -> jax.Device | None:
    gpus = [d for d in jax.devices() if d.platform != "cpu"]
    return gpus[0] if gpus else None


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _first_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """The first day (0-based) on which any output differs between ``a`` and ``b``, and the outputs
    differing on it."""
    days = {
        k: int(np.argmax(np.any((a[k] != b[k]).reshape(a[k].shape[0], -1), axis=1)))
        for k in a
        if not np.array_equal(a[k], b[k], equal_nan=True)
    }
    if not days:
        return None
    d = min(days.values())
    return {"day": d, "outputs": sorted(k for k, v in days.items() if v == d)}


def _run(rs: list[Any], dev: jax.Device) -> tuple[dict[str, np.ndarray], Execution]:
    """``h.run_group`` for the ``free`` configuration on device ``dev`` (outputs checked there);
    also the execution settings the runtime traced the program with."""
    soil_values, soildyn, real4_sw = h.CONFIGS["free"]
    n = max(r.n_days for r in rs)
    with jax.default_device(dev):
        ps = [h.params_of(r, soil_values, real4_sw) for r in rs]
        params = h._stack(ps)
        state = h._stack([h.state_of(r, p, soil_values) for r, p in zip(rs, ps, strict=True)])
        forcing = h._stack([h.forcing_of(r, n, soil_values, soildyn) for r in rs])
        params, state, forcing = jax.device_put((params, state, forcing), dev)
        model = day_dssat486(SLOT).compile(
            day_processes(SLOT, mesev=rs[0].mesev), outputs=day_outputs(SLOT), exact_lags=True
        )
        out = run_sites(model, params, forcing, state)
    for k, v in out.items():
        assert v.devices() == {dev}, (k, v.devices(), dev)
    (key,) = model.__dict__[_RUNNERS_ATTR]
    ex = next(x for x in key if isinstance(x, Execution))
    return {k: np.asarray(v) for k, v in out.items()}, ex


@pytest.fixture(scope="module")
def table(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    gpu = _gpu()
    if gpu is None:
        pytest.skip("no GPU visible to JAX")
    if not h.m2.DSCSM.is_file() or not h.m2.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {h.m2.DSSAT_ENGINE}")
    for sub in (h.DSW, h.DET, h.A12, h.CATPA_CASE):
        if not (data_dir / sub).is_dir():
            pytest.skip(f"{data_dir / sub} not found")
    keys = h.a12_keys(data_dir) + h.catpa_keys()
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    refs = h.run_references(keys, tmp_path_factory.mktemp("dint_gpu"), data_dir, jobs)
    runs = [h.build(e, t, refs[h.key_of(e, t)], data_dir) for e, t in keys]
    cpu = jax.devices("cpu")[0]
    rows: dict[str, dict[str, Any]] = {}
    settings: dict[str, set[Execution]] = {"cpu": set(), "gpu": set(), "gpu_loops": set()}
    for grp in sorted({(r.nl, r.mesev) for r in runs}):
        rs = [r for r in runs if (r.nl, r.mesev) == grp]
        (oc, ec), (og, eg) = _run(rs, cpu), _run(rs, gpu)
        with execution(depth_unroll=False):
            ogl, egl = _run(rs, gpu)
        settings["cpu"].add(ec)
        settings["gpu"].add(eg)
        settings["gpu_loops"].add(egl)
        for i, r in enumerate(rs):
            n = r.n_days
            c = {k: v[i][:n] for k, v in oc.items()}
            g = {k: v[i][:n] for k, v in og.items()}
            gl = {k: v[i][:n] for k, v in ogl.items()}
            hwam = float(r.row["HWAM"])
            yc, yg = float(c["gwad"][n - 1, 0]), float(g["gwad"][n - 1, 0])
            rows[r.key] = {
                "run": r.key,
                "hwam": hwam,
                "yield_cpu": yc,
                "yield_gpu": yg,
                "yield_gpu_cpu_rel": abs(yg - yc) / max(abs(yc), 1.0),
                "yield_rel_hwam": abs(yg - hwam) / hwam if hwam > 0 else None,
                "yield_ok": (abs(yg - hwam) / hwam < h.YIELD_REL) if hwam > 0 else abs(yg - hwam) < 1.0,
                "dates_gpu": h._dates(r, g["istage"][:, 0]),
                "istage_equal": bool(np.array_equal(c["istage"], g["istage"])),
                "series_max_abs": {
                    k: float(np.max(np.abs(np.asarray(g[k], float) - np.asarray(c[k], float))))
                    for k in SERIES
                },
                "finite": bool(all(np.all(np.isfinite(g[k])) for k in SERIES)),
                "gpu_loops_bit_identical": bool(
                    all(np.array_equal(g[k], gl[k], equal_nan=True) for k in g) and set(g) == set(gl)
                ),
                "gpu_loops_yield_rel": abs(yg - float(gl["gwad"][n - 1, 0])) / max(abs(yg), 1.0),
                "gpu_loops_istage_equal": bool(np.array_equal(g["istage"], gl["istage"])),
                "gpu_loops_series_max_abs": {k: _max_abs(g[k], gl[k]) for k in SERIES},
                "gpu_loops_first_diff": _first_diff(g, gl),
            }
    out = {
        "step": "DSSAT day, free run, GPU float64 (XLA default flags) against CPU float64",
        "gpu": str(gpu.device_kind),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "execution": {k: sorted(map(str, v)) for k, v in settings.items()},
        "runs": [rows[h.key_of(e, t)] for e, t in keys],
    }
    path = data_dir / h.REPORT_DIR
    path.mkdir(parents=True, exist_ok=True)
    (path / "d2_1a_free_run_gpu.json").write_text(json.dumps(out, indent=1, default=float))
    assert len(out["runs"]) == 65
    out["_settings"] = settings
    return out


def test_gpu_float64_equals_cpu_float64(table: dict[str, Any]) -> None:
    rows = table["runs"]
    worst = max(rows, key=lambda r: r["yield_gpu_cpu_rel"])
    print(f"\nGPU vs CPU f64: worst yield rel {worst['yield_gpu_cpu_rel']:.3e} ({worst['run']})")
    bad_y = {
        r["run"]: r["yield_gpu_cpu_rel"] for r in rows if not r["yield_gpu_cpu_rel"] <= GPU_CPU_YIELD_REL
    }
    bad_sw = {
        r["run"]: r["series_max_abs"]["soil_sw"]
        for r in rows
        if not r["series_max_abs"]["soil_sw"] <= GPU_CPU_SW_ATOL
    }
    bad_stage = [r["run"] for r in rows if not r["istage_equal"]]
    assert all(r["finite"] for r in rows)
    assert not bad_y, bad_y
    assert not bad_sw, bad_sw
    assert not bad_stage, bad_stage


def test_gpu_acceptance_yield_and_dates(table: dict[str, Any]) -> None:
    rows = table["runs"]
    bad = {r["run"]: (r["yield_rel_hwam"], r["hwam"], r["yield_gpu"]) for r in rows if not r["yield_ok"]}
    assert set(bad) == set(h.KNOWN_YIELD_FAILURES), bad
    got = {
        r["run"]: {k: v["diff_days"] for k, v in r["dates_gpu"].items() if v["diff_days"] not in (0, None)}
        for r in rows
    }
    assert {k: v for k, v in got.items() if v} == h.KNOWN_DATE_DIFFS


def test_gpu_unrolled_matches_gpu_loops(table: dict[str, Any]) -> None:
    """The execution setting changes the program's layout, not the model: on the same GPU the
    unrolled layer recurrences (the GPU default) and the loops give the bucket's outputs (soil
    water, runoff, drainage) and every stage bit for bit; the other outputs may differ by the
    rounding of a different XLA fusion of the surrounding code (fused multiply-add), measured
    <= 2e-16 relative in yield, far inside :data:`GPU_LOOPS_YIELD_REL`."""
    st = table["_settings"]
    assert {e.depth_unroll for e in st["cpu"]} == {False}
    assert {e.depth_unroll for e in st["gpu"]} == {True}
    assert {e.depth_unroll for e in st["gpu_loops"]} == {False}
    assert {e.platform for e in st["gpu"] | st["gpu_loops"]} == {"gpu"}
    rows = table["runs"]
    n_bit = sum(r["gpu_loops_bit_identical"] for r in rows)
    print(f"\nGPU unrolled vs loops: {n_bit} of {len(rows)} runs bit for bit")
    assert not [r["run"] for r in rows if not r["gpu_loops_istage_equal"]]
    bad_bucket = {
        r["run"]: {k: r["gpu_loops_series_max_abs"][k] for k in BUCKET_SERIES}
        for r in rows
        if any(r["gpu_loops_series_max_abs"][k] != 0.0 for k in BUCKET_SERIES)
    }
    assert not bad_bucket, bad_bucket
    bad_y = {
        r["run"]: r["gpu_loops_yield_rel"]
        for r in rows
        if not r["gpu_loops_yield_rel"] <= GPU_LOOPS_YIELD_REL
    }
    assert not bad_y, bad_y
    bad_s = {
        r["run"]: r["gpu_loops_series_max_abs"]
        for r in rows
        if not all(v <= GPU_LOOPS_SERIES_ATOL for v in r["gpu_loops_series_max_abs"].values())
    }
    assert not bad_s, bad_s
