"""The DSSAT-CSM v4.8.6.0 day against ``dscsm048`` on the maize treatments of M2.

Harness and criteria: :mod:`day_dssat486_harness`. Two configurations of the same day
(:mod:`agrijax.models.day_dssat486`), both batched over the treatments:

* A (the M2 configuration: soil water, ``EOP`` and ``TRWUP`` replayed as M2 reads them) gives the
  CERES outputs of ``test_ceres_dssat.simulate`` bit for bit on every treatment;
* B (ROOTWU and CERES coupled; soil water, SPAM evaporation and partition replayed at full
  precision) meets the M2 criteria of the 58-treatment test on every treatment, and the water
  ledger of the replayed soil closes every day within the print half-steps of DSSAT's daily
  balance terms.

The default tier runs four treatments (UFGA8201 rainfed, irrigated and stressed; IUAF9901 with
waterlogging days); the slow tier runs all 58 and writes
``<data-dir>/validation/aj_dday/d2_0_day_dssat486.json``.
"""

from __future__ import annotations

import os
from pathlib import Path

import day_dssat486_harness as h
import jax
import pytest

pytestmark = [
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the SPAM dump tables"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

SMALL = [("UFGA8201", 1), ("UFGA8201", 3), ("UFGA8201", 5), ("IUAF9901", 1)]


@pytest.fixture(scope="module", params=["small", pytest.param("all", marks=pytest.mark.slow)])
def rows(request: pytest.FixtureRequest, data_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    if not h.m2.DSCSM.is_file() or not h.m2.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {h.m2.DSSAT_ENGINE}")
    tables = h.tables_dir(data_dir)
    if not (tables / "UFGA8201_t01_spam.npz").is_file():
        pytest.skip(f"SPAM dump tables not found under {tables}")
    keys = SMALL if request.param == "small" else h.treatments(tables)
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    out = h.evaluate(keys, data_dir, tmp_path_factory.mktemp("dday"), jobs)
    if request.param == "all":
        assert len(out) == 58
        h.write_report(out, data_dir)
    else:
        h.write_report(out, data_dir, name="d2_0_day_dssat486_small.json")
    return out


def test_detailed_output_switch_changes_no_output(rows) -> None:
    assert [r["treatment"] for r in rows if r["detailed_run_differs_in"]] == []


def test_m2_configuration_is_bit_identical_to_m2(rows) -> None:
    bad = {r["treatment"]: r["a_columns_not_identical"] for r in rows if not r["a_bit_identical_to_m2"]}
    assert not bad, bad


def test_coupled_day_meets_the_m2_criteria(rows) -> None:
    bad = {
        r["treatment"]: (r["b_max_units"], r["b_max_rel_big"], r["yield_b_rel"])
        for r in rows
        if not r["b_meets_m2_criteria"]
    }
    assert not bad, bad
    # ROOTWU on our record and the start-of-day soil: TRWUP at the reference's REAL*4 level
    for r in rows:
        assert r["trwup_zero_mismatch_days"] == 0, r["treatment"]


def test_replayed_soil_water_ledger_closes(rows) -> None:
    assert all(r["nonzero_zero_columns"] == [] for r in rows)
    bad = {
        r["treatment"]: (r["ledger_max_abs_residual_mm"], r["ledger_bound_mm"])
        for r in rows
        if not r["ledger_ok"]
    }
    assert not bad, bad
