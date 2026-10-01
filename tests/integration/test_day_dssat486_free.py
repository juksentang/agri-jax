"""The DSSAT-CSM v4.8.6.0 day run free against ``dscsm048``: yield, phenology, LAI, ET,
soil water and the daily water ledger on the 58 maize treatments of M2 and the 7 CA-TPA seasons.

Harness, inputs, configurations and criteria: :mod:`day_dssat486_free_harness`. The default tier
runs six runs (UFGA8201 rainfed / irrigated / stressed, IUAF9901 t1, GHWA0401 t1 with the SALUS
evaporation, mulch and the SOILDYN replay, CA-TPA 2015); the slow tier runs all 65 and writes
``<data-dir>/validation/aj_dint/d2_1a_free_run.json``.

Besides the yield and the dates, the secondary metrics (LAI and CWAD RMSE, ET.OUT, SoilWat.OUT, the
end-of-day SW against the dumps) are pinned at their measured bounds plus a margin
(``h.SECONDARY_BOUNDS``, separately for the 58 M2 treatments and the 7 CA-TPA seasons), and TRWUP is compared with the SPAM dump: CA-TPA's TRWUP deviates by up to
1.48 % of the season total (2020) and 11 % on single days, the 58 M2 treatments by at most
0.002 cm d-1 (``h.TRWUP_SUM_REL_CATPA``, ``h.TRWUP_ATOL_M2``).
"""

from __future__ import annotations

import os
from pathlib import Path

import day_dssat486_free_harness as h
import jax
import pytest

pytestmark = [
    pytest.mark.allow_skip(
        reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT), the DSSAT dump tables (WATBAL, SPAM, ROOTWU) and "
        "the CA-TPA DSSAT case"
    ),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

SMALL = [
    ("UFGA8201", 1),
    ("UFGA8201", 3),
    ("UFGA8201", 5),
    ("IUAF9901", 1),
    ("GHWA0401", 1),
    ("CTPA1501", 1),
]


@pytest.fixture(scope="module", params=["small", pytest.param("all", marks=pytest.mark.slow)])
def table(request: pytest.FixtureRequest, data_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    if not h.m2.DSCSM.is_file() or not h.m2.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {h.m2.DSSAT_ENGINE}")
    for sub in (h.DSW, h.DET, h.A12, h.CATPA_CASE):
        if not (data_dir / sub).is_dir():
            pytest.skip(f"{data_dir / sub} not found")
    keys = SMALL if request.param == "small" else h.a12_keys(data_dir) + h.catpa_keys()
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    out = h.evaluate(keys, data_dir, tmp_path_factory.mktemp("dint"), jobs)
    name = "d2_1a_free_run.json" if request.param == "all" else "d2_1a_free_run_small.json"
    h.write_report(out, data_dir, name)
    if request.param == "all":
        assert len(out["free"]) == 65
    return out


def test_free_run_is_finite_and_the_ledger_closes(table) -> None:
    for cfg in h.CONFIGS:
        rows = table[cfg]
        assert all(r["finite"] for r in rows), cfg
        bad = {r["run"]: r["ledger"] for r in rows if r["ledger"]["max_abs_residual_mm"] > h.LEDGER_ATOL_MM}
        assert not bad, (cfg, bad)


def test_yield_within_two_percent(table) -> None:
    bad = {r["run"]: (r["yield_rel"], r["hwam"], r["yield"]) for r in table["free"] if not r["yield_ok"]}
    assert set(bad) == set(h.KNOWN_YIELD_FAILURES), bad


def test_phenology_dates(table) -> None:
    got = {
        r["run"]: {k: v["diff_days"] for k, v in r["dates"].items() if v["diff_days"] not in (0, None)}
        for r in table["free"]
    }
    runs = {r["run"] for r in table["free"]}
    assert {k: v for k, v in got.items() if v} == {k: v for k, v in h.KNOWN_DATE_DIFFS.items() if k in runs}


def test_xtract_albedo_and_petpt_on_the_dumps(table) -> None:
    for x in table["xtract_one_day"]:
        assert x["msalb_ok_days"] == x["n_days"], (x["run"], x["msalb_ulp_max"])
        assert x["eo_ok_days"] == x["n_days"], (x["run"], x["eo_ulp_max"])
        if "uptake_ulp_max" in x:
            assert x["uptake_bad_layer_days"] == 0 and x["ep_bad_days"] == 0, x


def test_every_configuration_but_the_mixed_precision_meets_the_yield_criterion(table) -> None:
    """The REAL*4 convention (``free``), the decimal inputs (``input``) and the static soil
    (``static``: SOILDYN replay off) all keep every yield within 2 %; the mixed precision
    (``real4_limits``) does not, on the CA-TPA seasons (``MIXED_PRECISION_FAILURES``)."""
    runs = {r["run"] for r in table["free"]}
    for cfg in ("free", "input", "static"):
        assert all(r["yield_ok"] for r in table[cfg]), cfg
    bad = {r["run"] for r in table["real4_limits"] if not r["yield_ok"]}
    assert bad == h.MIXED_PRECISION_FAILURES & runs, bad


def test_secondary_metrics_within_their_measured_bounds(table) -> None:
    bad = {}
    for r in table["free"]:
        got = h.secondary(r)
        bounds = h.SECONDARY_BOUNDS[h.bound_group(r["run"])]
        over = {k: (v, bounds[k][1]) for k, v in got.items() if v > bounds[k][1]}
        if over:
            bad[r["run"]] = over
    assert not bad, bad


def test_trwup_against_the_spam_dump(table) -> None:
    """CA-TPA: the season total of TRWUP within ``TRWUP_SUM_REL_CATPA`` (measured at most 1.48 %,
    2020); the M2 treatments: every day within ``TRWUP_ATOL_M2`` cm d-1 (measured 0.002)."""
    catpa = {r["run"]: r["dump"]["trwup_sum_rel"] for r in table["free"] if r["run"].startswith("CTPA")}
    assert all(v <= h.TRWUP_SUM_REL_CATPA for v in catpa.values()), catpa
    m2 = {
        r["run"]: r["dump"]["trwup_max_abs"]
        for r in table["free"]
        if not r["run"].startswith("CTPA") and r["dump"]["trwup_max_abs"] > h.TRWUP_ATOL_M2
    }
    assert not m2, m2
