"""The daily radiation reconstruction (``agrijax.forcing.radiation``) against RZWQM2 4.6 itself.

References (see ``radiation_reference.py``): the full-precision dump tables of the binary rebuilt
from source (CA-TPA 2015-2023 at the PHYSCL entry: RTS, RTH, the 24 hourly HRTH and HRTS, CLOUDS;
40 POTEVPHR entry dumps of CA-TPA 2015; RTS and RTH at the POTEVPHR entry of 9 scenarios x 3
years) and the ``.ana`` column 88 of the shipped binary (CA-TPA 2015-2023, and the first year of
all 15 batch scenarios, run fresh; US_Rockford_Alfalfa has a 2-degree slope).

Tolerances are the reference's own limits, measured in the test from the same data:

* RTH is the ``.MET`` value times the modifier, bounded: bit-identical, no tolerance;
* the float32 DSSAT arithmetic is not reproducible to the bit (the reference's single-precision
  libm is not NumPy's): the limit is how far two builds of the reference itself disagree. The
  shipped binary's ``.ana`` differs from the rebuilt binary's dumped RTS by more than the print
  rounding on some days; that excess (relative to the value) is a lower bound of the
  build-to-build difference, :data:`build_bound`. RTS, HRTH and HRTS must agree with the dumps
  within it, CLOUDS within ``cloud_slope x`` it (the cloud fraction is linear in the daily
  transmissivity);
* against the printed ``.ana`` column (6 significant digits): the error beyond half a printed unit
  must not exceed the same build bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import radiation_reference as rr

from agrijax.forcing.radiation import RZWQM_RADIATION

pytestmark = pytest.mark.allow_skip(reason="needs the RZWQM2 dump tables and scenario files")


@pytest.fixture(scope="module")
def data(data_dir: Path) -> Path:
    for p in (
        rr.PHYSCL_2015_2023,
        rr.PET_SCEN,
        rr.POTEVPHR_DUMPS,
        rr.CATPA_ANA_2015_2023,
        rr.BATCH / "CA-TPA",
    ):
        if not (data_dir / p).exists():
            pytest.skip(f"{data_dir / p} not found")
    return data_dir


@pytest.fixture(scope="module")
def build_bound(data: Path) -> float:
    """Shipped-binary ``.ana`` vs rebuilt-binary dumps, CA-TPA 2015-2023: max relative excess over print."""
    b = rr.build_excess(rr.table_rts(data / rr.PHYSCL_2015_2023), rr.ana_rts(data / rr.CATPA_ANA_2015_2023))
    assert b["n"] == 3287
    # the reference itself is not reproducible to print precision across builds (measured 4.83e-7)
    assert b["n_days_beyond_print"] > 0
    return b["max_rel_excess"]


@pytest.fixture(scope="module")
def physcl(data: Path) -> dict[str, Any]:
    return rr.compare_physcl_catpa(data)[0]


def test_rth_is_bit_identical(physcl: dict[str, Any], data: Path) -> None:
    assert physcl["srad_horizontal (RTH)"]["n_bit_identical"] == 3287
    assert rr.compare_potevphr_dumps(data)["srad_horizontal (RTH)"]["n_bit_identical"] == 40
    for s in rr.pet_scen_sites(data):
        st = rr.compare_pet_scen(data, s)["srad_horizontal (RTH)"]
        assert st["n_bit_identical"] == st["n"], s


def test_catpa_2015_2023_dump_table(physcl: dict[str, Any], build_bound: float) -> None:
    for k in ("srad (RTS)", "hourly_horizontal (HRTH)", "hourly_slope (HRTS)"):
        assert physcl[k]["max_rel"] <= build_bound, (k, physcl[k], build_bound)
    cloud_bound = RZWQM_RADIATION.shaw.cloud_slope * build_bound
    assert physcl["clouds (CLOUDS)"]["max_abs"] <= cloud_bound, (physcl["clouds (CLOUDS)"], cloud_bound)
    # most days are bit-identical (measured 3117 of 3287 for RTS)
    assert physcl["srad (RTS)"]["n_bit_identical"] >= 3117


def test_catpa_2015_potevphr_dumps(data: Path, build_bound: float) -> None:
    r = rr.compare_potevphr_dumps(data)
    assert r["n_dumps"] == 40
    for k in ("srad (RTS)", "hourly_horizontal (HRTH)", "hourly_slope (HRTS)"):
        assert r[k]["max_rel"] <= build_bound, (k, r[k])
    assert r["clouds (CLOUDS)"]["max_abs"] <= RZWQM_RADIATION.shaw.cloud_slope * build_bound


def test_pet_scenario_tables(data: Path, build_bound: float) -> None:
    sites = rr.pet_scen_sites(data)
    assert len(sites) == 9
    for s in sites:
        st = rr.compare_pet_scen(data, s)["srad (RTS)"]
        assert st["n"] >= 1095, s
        assert st["max_rel"] <= build_bound, (s, st)


def test_ana_catpa_2015_2023(data: Path, build_bound: float) -> None:
    met, geo = rr.scenario_inputs(data, "CA-TPA")
    st, _ = rr.compare_ana(data / rr.CATPA_ANA_2015_2023, met, geo)
    assert st["n"] == 3287
    assert st["max_rel_excess_over_print"] <= build_bound, st


@pytest.mark.slow
def test_ana_first_year_all_scenarios(data: Path) -> None:
    """Fresh first-year runs of the 15 scenarios; the bound is measured on the 9 with dump tables."""
    batch = data / rr.BATCH
    tool = data / "narval_mirror/RZWQM_Tool/main_ryzen5_avx512"
    if not tool.is_file():
        pytest.skip("RZWQM binary not found")
    assert all((batch / s / "Scenario").is_dir() for s in rr.SITES)
    recs = rr.run_first_years(data, rr.SITES)
    failed = {s: r["error"] for s, r in recs.items() if not r["ok"]}
    assert not failed, failed
    bounds = [
        rr.build_excess(rr.table_rts(data / rr.PET_SCEN / f"{s}_entry.npz"), rr.ana_rts(Path(recs[s]["ana"])))
        for s in rr.pet_scen_sites(data)
    ]
    bound = max(b["max_rel_excess"] for b in bounds)
    for s in rr.SITES:
        met, geo = rr.scenario_inputs(data, s)
        st, _ = rr.compare_ana(Path(recs[s]["ana"]), met, geo)
        assert st["n"] >= 365, s
        assert st["max_rel_excess_over_print"] <= bound, (s, st, bound)
    # the sloped scenario exercises the SHAW direct/diffuse partition: treated as flat, it misses
    met, geo = rr.scenario_inputs(data, "US_Rockford_Alfalfa")
    assert geo["slope_rad"] > 0.0
    flat, _ = rr.compare_ana(Path(recs["US_Rockford_Alfalfa"]["ana"]), met, {**geo, "slope_rad": 0.0})
    assert flat["max_rel_excess_over_print"] > 100 * bound, flat
