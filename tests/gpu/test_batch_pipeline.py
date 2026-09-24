"""GPU tier: the CA-TPA S-W PET pipeline, ``run_batch_chunked`` on the GPU vs the CPU.

Runs :func:`tests/gpu/pipeline_check.check` (also runnable as a script on a GPU node):
10^5 Latin-hypercube PET parameter sets over one year on the GPU, then

* the first 256 results against ``run_batch`` (vmap) on the CPU of the same node, and sample 0
  against a plain ``run`` on the CPU -- an independent backend (XLA:CPU vs XLA:GPU kernels,
  transcendental functions and reduction order) and an independent batching path;
* the running totals against the sum of the daily fluxes over the whole batch (conservation);
* every output finite.

Inputs are the real CA-TPA 2015 forcing when the data tree has it (``--data-dir`` /
``AGRI_JAX_DATA``), otherwise a deterministic synthetic year, so the test runs wherever a GPU is
visible. ``AGRI_JAX_GPU_N`` overrides the number of parameter sets (default 100,000).
Skipped by tests/gpu/conftest.py when JAX sees no GPU.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import jax
import pytest

SCRIPT = Path(__file__).resolve().parent / "pipeline_check.py"
N = int(os.environ.get("AGRI_JAX_GPU_N", "100000"))
X64 = bool(jax.config.jax_enable_x64)
# x64: GPU and CPU differ only by a few ulp of the transcendental functions, accumulated over 365 days
ATOL = 1e-9 if X64 else 5e-3  # [cm]; outputs are O(0.01-100) cm
CONS_ATOL = 1e-9 if X64 else 5e-3


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("gpu_pipeline_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def result(request: pytest.FixtureRequest) -> dict[str, Any]:
    data = Path(str(request.config.getoption("--data-dir"))).expanduser()
    res = _load().check(N, n_compare=256, data_dir=data if data.is_dir() else None, repeat=False)
    print(
        f"[gpu] {res['source']} n={res['n']} {res['gpu']}: compile {res['wall_gpu_compile_s']:.2f} s, "
        f"run {res['wall_gpu_run_s']:.2f} s, peak {res['gpu_peak_bytes'] / 2**30:.2f} GiB, "
        f"max|gpu-cpu| {res['max_abs_diff']:.3e}"
    )
    return res


def test_gpu_matches_cpu_run_batch(result: dict[str, Any]) -> None:
    assert result["n_compare"] == min(256, N)
    for key, d in result["diffs"].items():
        assert d["max_abs"] <= ATOL, (key, d)


def test_gpu_sample0_matches_cpu_run(result: dict[str, Any]) -> None:
    for key, d in result["diffs"].items():
        assert d["max_abs_sample0_vs_run"] <= ATOL, (key, d)


def test_day_counter_exact(result: dict[str, Any]) -> None:
    assert result["diffs"]["totals.days"]["max_abs"] == 0.0


def test_totals_conserve_daily_fluxes(result: dict[str, Any]) -> None:
    assert result["conservation_max_abs_gpu"] <= CONS_ATOL
    assert result["conservation_max_abs_host_head"] <= CONS_ATOL


def test_all_outputs_finite_and_plausible(result: dict[str, Any]) -> None:
    assert result["n_nonfinite"] == 0
    a = result["annual_transpiration_cm"]
    # one year of potential transpiration: positive and below 2 m of water
    assert 0.0 < a["min"] <= a["median"] <= a["max"] < 200.0
