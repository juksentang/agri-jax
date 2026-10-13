"""``agrijax.calib.calibrate`` end to end on a DSSAT experiment, against ``dscsm048`` (slow).

* **Inputs.** The free-run inputs built by :mod:`agrijax.sites.dssat_free_run` (the public builder
  ``calibrate`` uses) equal the free-run acceptance harness's (``day_dssat486_free_harness``,
  configuration ``free``) leaf for leaf on UFGA8201 treatment 4.
* **Calibration.** UFGA8201 (cultivar IB0035), calibrated on treatment 4, treatment 6 held out
  (treatment 2 is refused: DSSAT's yield changes by 14 % with nitrogen on), joint CMA-ES (the default), the new row written to a ``.CUL`` copy and run in DSSAT:
  the objective at the written coefficients is below the published cultivar's; ``dscsm048`` with the
  written row gives the same silking and maturity dates and a yield within ``DSSAT_YIELD_RTOL``
  (0.1 %) of Agri-JAX's prediction on every treatment, calibrated and held out.
* **Nothing dropped silently.** FLSC8101 treatment 1: its observed maturity (1981210) lies one day
  after the reference season's last simulated day, so ``calibrate`` raises ``ObservationError``
  naming MDAT; leaving MDAT out (``targets=``) is accepted up to the fit.
* **Scope.** A low-nitrogen treatment is refused before anything runs.

Runs are staged under ``AGRI_JAX_RUN_ROOT`` (a short node-local directory on the cluster). Set
``XLA_FLAGS=--xla_force_host_platform_device_count=<cores>`` to shard the candidates over the cores.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import numpy as np
import pytest

from agrijax.calib import calibrate
from agrijax.calib.workflow import DSSAT_YIELD_RTOL, ObservationError, ScopeError
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
from agrijax.sites.dssat_free_run import free_run_inputs, missing_tables, run_reference

pytestmark = pytest.mark.slow

MAIZE = DSSAT_ENGINE / "example_data" / "Maize"


@pytest.fixture(scope="module")
def engine_and_tables(data_dir: Path) -> Path:
    if not jax.config.jax_enable_x64:
        pytest.skip("the calibration runs in float64")
    exe, _ = dscsm_paths(DSSAT_ENGINE)
    if not exe.is_file():
        pytest.skip(f"dscsm048 not found at {exe}")
    miss = [p for t in (2, 4, 6) for p in missing_tables("UFGA8201", t, data_dir)]
    miss += missing_tables("FLSC8101", 1, data_dir)
    if miss:
        pytest.skip(f"free-run tables missing: {miss[:3]}")
    return data_dir


def _leaves_equal(a, b) -> list[str]:
    la, ta = jax.tree_util.tree_flatten_with_path(a)
    lb, tb = jax.tree_util.tree_flatten_with_path(b)
    assert ta == tb, (ta, tb)
    bad = []
    for (path, x), (_, y) in zip(la, lb, strict=True):
        xa, ya = np.asarray(x), np.asarray(y)
        if xa.shape != ya.shape or xa.dtype != ya.dtype or not np.array_equal(xa, ya, equal_nan=True):
            bad.append(jax.tree_util.keystr(path))
    return bad


def test_free_run_inputs_equal_the_harness(engine_and_tables: Path, tmp_path: Path) -> None:
    import day_dssat486_free_harness as h

    data = engine_and_tables
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tmp_path)
    out = run_reference(MAIZE / "UFGA8201.MZX", 4, root / "fr_UFGA8201_t04")
    ours = free_run_inputs(MAIZE / "UFGA8201.MZX", 4, out, data)
    ref = h.build("UFGA8201", 4, out, data)
    np.testing.assert_array_equal(ours.days, ref.days)
    assert ours.soildyn == ref.soildyn and ours.mesev == ref.mesev and ours.cultivar == "IB0035"
    soil_values, soildyn, real4_sw = h.CONFIGS["free"]
    n = ours.n_days + 60
    p_ref = h.params_of(ref, soil_values, real4_sw)
    p_ours = ours.params()
    assert _leaves_equal(p_ours, p_ref) == []
    assert _leaves_equal(ours.forcing(n), h.forcing_of(ref, n, soil_values, soildyn)) == []
    assert _leaves_equal(ours.state(p_ours), h.state_of(ref, p_ref, soil_values)) == []


def test_calibrate_ufga8201_round_trip_through_dssat(engine_and_tables: Path, tmp_path: Path) -> None:
    cul = tmp_path / "MZCER048.CUL"
    res = calibrate(
        "UFGA8201",
        treatments=[4],
        holdout=[6],
        write_cul=cul,
        dssat_check=True,
        data_dir=engine_and_tables,
        seed=0,
    )
    print(res)
    assert res.cultivar == "IB0035"
    assert res.treatments == ["UFGA8201_t04"] and res.improved
    # the objective at the written coefficients beats the published cultivar's
    assert res.loss["written"] < res.loss["published"]
    # the published cultivar on the free-run day reproduces the reference runs (acceptance: 2 %)
    for k, p in res.predictions["published"].items():
        ref = res.reference[k]
        assert abs(p["HWAM"] - ref["HWAM"]) <= 0.02 * ref["HWAM"], (k, p, ref)
        assert int(p["ADAT"]) == int(ref["ADAT"]) and int(p["MDAT"]) == int(ref["MDAT"]), (k, p, ref)
    # the written row is in the copy, with the written values
    text = cul.read_text(errors="replace")
    assert res.cul_line is not None and res.cul_line in text and res.cul_id is not None
    assert res.cul_line.startswith(res.cul_id)
    # DSSAT with the written row: same dates, yield within 0.1 %, calibrated and held-out treatments
    chk = res.dssat_check
    assert chk is not None
    assert [t["treatment"] for t in chk["treatments"]] == ["UFGA8201_t04", "UFGA8201_t06"]
    for t in chk["treatments"]:
        assert t["read_new_row"], t
        assert t["dates_equal"], t
        assert t["yield_rel_diff"] <= DSSAT_YIELD_RTOL, t
    assert chk["agree"]
    assert res.holdout is not None and np.isfinite(res.holdout["loss_written"])
    assert set(res.fit["set"]) == {"calibration"} and len(res.holdout["fit"]) > 0
    # the default targets: dates, yield, tops, grain number, LAI; the others are listed
    assert set(res.targets) == {"ADAT", "MDAT", "HWAM", "CWAM", "H#AM", "LAID"}
    assert "HWUM" in res.unsupported


def test_dropped_observed_target_is_an_error(engine_and_tables: Path) -> None:
    with pytest.raises(ObservationError, match=r"FLSC8101_t01: observed MDAT 1981210 lies outside"):
        calibrate("FLSC8101", treatments=[1], data_dir=engine_and_tables)


def test_out_of_scope_treatment_is_refused(engine_and_tables: Path) -> None:
    with pytest.raises(ScopeError, match="nitrogen matters"):
        calibrate("UFGA8201", treatments=[1, 2], data_dir=engine_and_tables)
    with pytest.raises(ScopeError, match="growth-chamber"):
        calibrate("GAGR0201", data_dir=engine_and_tables)
    with pytest.raises(ScopeError, match="one cultivar per calibration"):
        calibrate("IBWA8301", treatments=[3, 6], data_dir=engine_and_tables)


def test_calibrate_staged_and_adam_methods(engine_and_tables: Path) -> None:
    """The other two methods run on the real day (small budgets): staged improves on the published
    cultivar; adam calibrates only coefficients with a trusted gradient (G2 / G3 at most, the phenology
    coefficients are derivative-free by default), or refuses when none is trusted."""
    res = calibrate(
        "UFGA8201",
        treatments=[4, 6],
        method="staged",
        starts=4,
        budget=800,
        targets="all",
        data_dir=engine_and_tables,
    )
    assert res.loss["written"] < res.loss["published"]
    assert {"LWAD", "SWAD", "CWAD", "GWAD"} <= set(res.targets)
    assert res.calls["stages"]["stage1"]["params"] == ["P1", "P5", "PHINT"]
    try:
        res = calibrate(
            "UFGA8201", treatments=[4], method="adam", starts=2, budget=20, data_dir=engine_and_tables
        )
    except ValueError as e:
        assert "no free coefficient has a trusted gradient" in str(e)
        return
    assert set(res.free) <= {"G2", "G3"} and res.free
    assert "plan" in res.trust
    assert res.loss["written"] <= res.loss["published"]
