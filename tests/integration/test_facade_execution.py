"""The facade's execution options on DSSAT's example experiment UFGA8201 (slow): float32 against
float64, the device, and the user's JAX settings put back.

* **float32 stays within the documented tolerance.** A season (treatments 4 and 6) and a batch of
  weather-year x sowing-date scenarios with perturbed cultivars, ``precision="float32"`` against the
  default float64 on the same inputs: yield, biomass, the silking and maturity dates and the daily
  series (:data:`SEASON_RTOL`, :data:`DAILY_RTOL`, :data:`BATCH_MEDIAN_RTOL`; the figures measured in the DSSAT free-run day are in
  :mod:`agrijax.facade_execution`). The float32 programs are kept (a second call compiles nothing) and
  the float64 programs are untouched by them (bit for bit the same as before).
* **The JAX settings are the user's.** With ``jax_enable_x64`` off before the call, a float32 season
  leaves it off; with it on, a float32 call leaves it on; the default device is the same after.
* **The device.** ``device="cpu"`` agrees with JAX's default device in float64 (the same program: bit for
  bit on a CPU, rounding on a GPU); with a GPU visible, ``device="gpu"`` agrees with the CPU to the GPU
  tolerance of the free-run acceptance (``tests/integration/test_day_dssat486_free_gpu.py``) in float64
  and to the float32 tolerance in float32; without one, it is a readable error.

The references of the float32 and GPU comparisons are run with ``device="cpu"`` explicitly, so that they
are the same on a machine with a GPU (where JAX's default device is the GPU) and on one without.

No ``dscsm048`` run is needed; DSSAT's example data (the mirror ``AGRI_JAX_DSSAT``) are.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

import agrijax as aj
from agrijax.port.run_fortran import DSSAT_ENGINE

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs DSSAT's example data (AGRI_JAX_DSSAT) and, for two tests, a GPU"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the float64 reference runs in float64"),
]

#: float32 against float64 on UFGA8201: relative difference of the yield and of the biomass at maturity
#: of one season. Measured (rorqual CPU node, JAX 0.10.2): at most 3.3e-7 over treatments 4 and 6 and a
#: changed cultivar. The bound is 30 times that; across the 65 free-run acceptance runs the
#: documented spread is wider (median 3e-7, 64 of 65 within 1e-3, largest 1.5e-2 in one treatment)
SEASON_RTOL = 1e-5
#: ... the daily LAI, tops, grain and soil-water series, relative to the series' largest value
#: (measured at most 2.0e-6)
DAILY_RTOL = 1e-4
#: a batch of 144 seasons (9 scenarios x 16 perturbed cultivars): the median yield and biomass
#: difference (measured 1.6e-7 and 1.5e-7) and the largest yield difference (measured 2.7e-6); no
#: stage date differs (0 of 288). On 10 000 perturbed samples of the free-run day the documented tail
#: is longer (99th percentile 3.6e-5, 13 samples above 1e-3), so this bound is for this fixed sample
BATCH_MEDIAN_RTOL = 1e-5
BATCH_MAX_RTOL = 1e-4
#: float64 on a GPU against float64 on the CPU (the free-run acceptance's GPU bound; measured 1.4e-15)
GPU_CPU_RTOL = 1e-9
#: float32 on a GPU against float32 on the CPU (free-run day: up to 2.2e-5 in yield on 65 runs;
#: measured here on an H100 MIG slice: at most 4.2e-7 in a season, 2.1e-6 in the batch)
GPU_CPU_FLOAT32_RTOL = 1e-4
#: the references and the float32 runs of the comparisons run on the CPU, wherever the test runs
CPU = {"device": "cpu"}
#: the cultivar samples of the batch tests (as the colab tutorial's): 16 samples of 5 coefficients
SAMPLES = 16


@pytest.fixture(scope="module")
def exp() -> aj.dssat.Experiment:
    mzx = DSSAT_ENGINE / "example_data" / "Maize" / "UFGA8201.MZX"
    if not mzx.is_file():
        pytest.skip(f"DSSAT's example data not found under {DSSAT_ENGINE}")
    return aj.dssat.experiment("UFGA8201", data_root=DSSAT_ENGINE)


def _gpus() -> list[Any]:
    try:
        return list(jax.devices("gpu"))
    except RuntimeError:
        return []


def _rel(a: Any, b: Any) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1.0)


def _season_report(s32: Any, s64: Any, label: str) -> dict[str, float]:
    """Differences of two seasons of one treatment (``s32`` against ``s64``); printed."""
    rep = {k: float(_rel(s32.summary[k], s64.summary[k])) for k in ("HWAM", "CWAM")}
    rep["dates_differ"] = float(sum(s32.summary[k] != s64.summary[k] for k in ("ADAT", "MDAT")))
    for k in ("lai", "cwad", "gwad", "swtd"):
        a, b = s32.daily[k].to_numpy(float), s64.daily[k].to_numpy(float)
        rep[k] = float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))
    print(f"{label}: " + ", ".join(f"{k} {v:.2e}" for k, v in rep.items()))
    return rep


@pytest.fixture(scope="module")
def seasons(exp: aj.dssat.Experiment) -> dict[Any, Any]:
    """float64 seasons of treatments 4 and 6, then the float32 ones (treatment 4 with ``jax_enable_x64``
    off beforehand), with the JAX settings seen before and after each float32 call."""
    out: dict[Any, Any] = {}
    for t in (4, 6):
        out[(t, "float64")] = exp.run(treatment=t, **CPU)
    settings = []
    for t, x64 in ((4, False), (6, True)):
        jax.config.update("jax_enable_x64", x64)
        dev = jax.config.jax_default_device
        try:
            out[(t, "float32")] = exp.run(treatment=t, precision="float32", **CPU)
            settings.append((x64, bool(jax.config.jax_enable_x64), dev is jax.config.jax_default_device))
        finally:
            jax.config.update("jax_enable_x64", True)
    out["settings"] = settings
    return out


def test_float32_season_agrees_with_float64_within_the_tolerance(seasons):
    for t in (4, 6):
        s64, s32 = seasons[(t, "float64")], seasons[(t, "float32")]
        assert s64.precision == "float64" and s32.precision == "float32"
        rep = _season_report(s32, s64, f"UFGA8201 t{t} float32 vs float64")
        assert rep["dates_differ"] == 0
        assert rep["HWAM"] < SEASON_RTOL and rep["CWAM"] < SEASON_RTOL, rep
        for k in ("lai", "cwad", "gwad", "swtd"):
            assert rep[k] < DAILY_RTOL, (k, rep)
        assert list(s32.daily["yrdoy"]) == list(s64.daily["yrdoy"])
        assert all(s32.daily[k].dtype == np.float64 for k in ("lai", "cwad", "gwad", "swtd"))
        assert s64.summary["HWAM"] > 1000.0  # a real crop, not two zeros agreeing


def test_float32_is_not_the_float64_run_relabelled(seasons):
    a, b = seasons[(4, "float64")], seasons[(4, "float32")]
    assert not np.array_equal(a.daily["cwad"].to_numpy(), b.daily["cwad"].to_numpy())


def test_a_float32_season_puts_the_jax_settings_back(seasons):
    # (x64 before, x64 after, default device unchanged): off -> off and on -> on
    assert seasons["settings"] == [(False, False, True), (True, True, True)]


def test_float32_cultivar_season_runs_past_the_reference_season(exp, seasons):
    pub = exp.inputs(4).published()
    c = {"P1": 0.95 * pub["P1"], "G2": 0.9 * pub["G2"]}
    s64 = exp.run(treatment=4, cultivar=c, **CPU)
    s32 = exp.run(treatment=4, cultivar=c, precision="float32", **CPU)
    rep = _season_report(s32, s64, "UFGA8201 t4 changed cultivar float32 vs float64")
    assert rep["dates_differ"] == 0 and rep["HWAM"] < SEASON_RTOL and len(s32.daily) == len(s64.daily)


@pytest.fixture(scope="module")
def batches(exp: aj.dssat.Experiment) -> dict[str, Any]:
    scen = exp.scenarios(treatment=4, years=[1979, 1982, 1985], sowing_shift=[-14, 0, 14])
    rng = np.random.default_rng(0)
    pub = scen.published
    cul = {n: pub[n] * rng.uniform(0.9, 1.1, SAMPLES) for n in ("P1", "P5", "G2", "G3", "PHINT")}
    for n in cul:
        cul[n][0] = pub[n]
    out: dict[str, Any] = {"scen": scen, "cul": cul}
    out["f64"] = scen.run(cul, **CPU)
    out["f32"] = scen.run(cul, precision="float32", **CPU)
    out["f32_again"] = scen.run({n: v[::-1] for n, v in cul.items()}, precision="float32", **CPU)
    out["f64_after"] = scen.run(cul, **CPU)
    return out


def test_float32_batch_agrees_with_float64_within_the_tolerance(batches):
    t64, t32 = batches["f64"].table, batches["f32"].table
    assert len(t64) == len(t32) == 9 * SAMPLES
    for c in ("scenario", "year", "sowing_shift", "sample", "P1", "G2"):
        assert (t32[c].to_numpy() == t64[c].to_numpy()).all(), c
    alive = t64["HWAM"].to_numpy() > 0
    y = _rel(t32["HWAM"], t64["HWAM"])[alive]
    b = _rel(t32["CWAM"], t64["CWAM"])[alive]
    dates = int(((t32["ADAT"] != t64["ADAT"]) & ~(t32["ADAT"].isna() & t64["ADAT"].isna())).sum()) + int(
        ((t32["MDAT"] != t64["MDAT"]) & ~(t32["MDAT"].isna() & t64["MDAT"].isna())).sum()
    )
    print(
        f"UFGA8201 t4 batch of {len(t64)} ({int(alive.sum())} with a yield) float32 vs float64: "
        f"yield median {np.median(y):.2e} p99 {np.quantile(y, 0.99):.2e} max {y.max():.2e}, "
        f"above 1e-3: {int((y > 1e-3).sum())}; biomass median {np.median(b):.2e} "
        f"max {b.max():.2e}; stage dates differing: {dates} of {2 * len(t64)}; "
        f"run_s float64 {batches['f64'].timing['run_s']:.3f} float32 {batches['f32'].timing['run_s']:.3f}; "
        f"compile_s float64 {batches['f64'].timing['compile_s']:.1f} float32 "
        f"{batches['f32'].timing['compile_s']:.1f}"
    )
    assert alive.sum() > len(t64) // 2
    assert dates == 0
    assert np.median(y) < BATCH_MEDIAN_RTOL and np.median(b) < BATCH_MEDIAN_RTOL
    assert y.max() < BATCH_MAX_RTOL and b.max() < BATCH_MAX_RTOL


def test_float32_programs_are_kept_and_the_float64_ones_are_untouched(batches):
    assert batches["f32"].timing["precision"] == "float32" and batches["f64"].timing["precision"] == "float64"
    assert batches["f32"].timing["compile_s"] > 0
    assert batches["f32_again"].timing["compile_s"] == 0.0  # the same float32 program
    assert batches["f64_after"].timing["compile_s"] == 0.0  # the float64 program of before
    a, b = batches["f64"].table, batches["f64_after"].table
    for c in ("HWAM", "CWAM", "ADAT", "MDAT"):
        np.testing.assert_array_equal(a[c].to_numpy(float), b[c].to_numpy(float), err_msg=c)
    assert {("float32", "cpu"), ("float64", "cpu")} <= set(batches["scen"].__dict__["_option_sims"])


def test_a_block_runs_a_batch_in_float32_and_run_batch_takes_the_keywords(exp, batches):
    cul = {n: v[:4] for n, v in batches["cul"].items()}
    with aj.options(precision="float32", device="cpu"):
        blk = batches["scen"].run(cul)
    assert blk.timing["precision"] == "float32"
    ref = batches["scen"].run(cul, precision="float32", **CPU)
    np.testing.assert_array_equal(blk.table["HWAM"].to_numpy(), ref.table["HWAM"].to_numpy())
    rb = exp.run_batch(4, cul, years=[1982], sowing_shift=[0], precision="float32", device="cpu")
    assert rb.timing["precision"] == "float32" and len(rb.table) == 4
    assert jax.config.jax_enable_x64


def test_device_cpu_float64_agrees_with_jax_default(exp, seasons, batches):
    s = exp.run(treatment=4)  # no options: JAX's default device (the GPU, when there is one)
    rep = _season_report(seasons[(4, "float64")], s, "UFGA8201 t4 device=cpu vs default (float64)")
    assert rep["dates_differ"] == 0 and rep["HWAM"] < GPU_CPU_RTOL and rep["gwad"] < GPU_CPU_RTOL
    assert s.precision == "float64"
    b = batches["scen"].run(batches["cul"])
    np.testing.assert_allclose(
        batches["f64"].table["HWAM"].to_numpy(), b.table["HWAM"].to_numpy(), rtol=GPU_CPU_RTOL
    )
    assert b.timing["precision"] == "float64" and b.timing["compile_s"] > 0
    assert ("float64", "auto") not in batches["scen"].__dict__.get("_option_sims", {})  # the plain path
    assert batches["scen"]._sim is not None


@pytest.mark.skipif(bool(_gpus()), reason="a GPU is visible: the error is for machines without one")
def test_gpu_that_is_absent_is_a_readable_error_before_anything_is_built(exp):
    fresh = aj.dssat.experiment("UFGA8201", data_root=DSSAT_ENGINE)
    for call in (
        lambda: fresh.run(treatment=4, device="gpu"),
        lambda: fresh.run_batch(4, device="gpu"),
        lambda: exp.scenarios(4, years=[1982]).run(device="gpu"),
    ):
        with pytest.raises(aj.DeviceNotFoundError, match=r"device='gpu'.*no GPU.*device='cpu'"):
            call()
    assert fresh._inputs == {}  # nothing was built


@pytest.mark.skipif(not _gpus(), reason="needs a GPU visible to JAX (run on rorqual with remote.sh --gpu)")
def test_gpu_agrees_with_the_cpu(exp, seasons, batches):
    s64 = exp.run(treatment=4, device="gpu")  # the references are the CPU runs of the fixtures
    rep = _season_report(s64, seasons[(4, "float64")], "UFGA8201 t4 gpu vs cpu (float64)")
    assert rep["dates_differ"] == 0 and rep["HWAM"] < GPU_CPU_RTOL and rep["gwad"] < GPU_CPU_RTOL
    s32 = exp.run(treatment=4, device="gpu", precision="float32")
    rep32 = _season_report(s32, seasons[(4, "float32")], "UFGA8201 t4 gpu vs cpu (float32)")
    assert rep32["dates_differ"] == 0 and rep32["HWAM"] < GPU_CPU_FLOAT32_RTOL
    g64 = batches["scen"].run(batches["cul"], device="gpu")
    np.testing.assert_allclose(
        g64.table["HWAM"].to_numpy(), batches["f64"].table["HWAM"].to_numpy(), rtol=GPU_CPU_RTOL
    )
    g32 = batches["scen"].run(batches["cul"], device="gpu", precision="float32")
    y = _rel(g32.table["HWAM"], batches["f32"].table["HWAM"])
    d = int((g32.table["MDAT"].to_numpy() != batches["f32"].table["MDAT"].to_numpy()).sum())
    print(f"batch gpu vs cpu float32: yield median {np.median(y):.2e} max {y.max():.2e}, MDAT differing {d}")
    assert y.max() <= GPU_CPU_FLOAT32_RTOL and d == 0
    assert "cpu" not in g32.timing["devices"].lower() and g32.timing["precision"] == "float32"
    print(
        f"batch run_s: gpu float64 {g64.timing['run_s']:.3f} float32 {g32.timing['run_s']:.3f}; "
        f"cpu float64 {batches['f64'].timing['run_s']:.3f} float32 {batches['f32'].timing['run_s']:.3f}"
    )


def test_calibrate_refuses_float32(exp):
    with aj.options(precision="float32"), pytest.raises(ValueError, match="float64"):
        aj.calibrate(exp, treatments=[4])
