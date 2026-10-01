"""The PRMS snowpack ``snow/prms@rzwqm2-4.6:faithful`` against RZWQM2 4.6 itself.

References and chains: see ``snow_reference.py``. Tolerances are the reference's own limits,
measured in the test from the same data:

* ``build_abs``: the rebuilt (instrumented) binary's full-double SNOWPK and SMELT against the shipped
  binary's printed ``.ana`` columns 92 and 105, beyond half a printed unit (CA-TPA 2015-2023): how
  far two builds of the reference disagree. The port driven by the reference's own inputs must
  agree with the full-double tables within it;
* ``rts_bound``: the same build-to-build limit of RTS (relative, ``radiation_reference``). Driven by
  the files chain (RTS rebuilt by ``agrijax.forcing.radiation``), the port may differ from the
  reference by as much as its output moves when RTS moves by that bound either way, plus
  ``build_abs``;
* against a printed column: half a printed unit plus those limits.

Event days (a pack at the end of the day, a melt) must agree exactly: 600 pack days and 195 melt
days at CA-TPA 2015-2023.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import snow_reference as sr

from agrijax.processes.snow import PrmsSnowParams, read_sno

pytestmark = pytest.mark.allow_skip(reason="needs the RZWQM2 dump tables and scenario files")


@pytest.fixture(scope="module")
def data(data_dir: Path) -> Path:
    for p in (sr.PHYSCL_ENTRY, sr.PHYSCL_EXIT, sr.CATPA_ANA, sr.PET_SCEN, sr.BATCH / "CA-TPA"):
        if not (data_dir / p).exists():
            pytest.skip(f"{data_dir / p} not found")
    return data_dir


@pytest.fixture(scope="module")
def catpa(data: Path) -> tuple[dict[str, Any], Any]:
    return sr.compare_catpa_dumps(data)


@pytest.fixture(scope="module")
def ana(data: Path, catpa: tuple[dict[str, Any], Any]) -> dict[str, Any]:
    return sr.compare_catpa_ana(data, catpa[1])


@pytest.fixture(scope="module")
def build_abs(ana: dict[str, Any]) -> dict[str, float]:
    b = {"swe": ana["col92"]["build_excess_abs"], "smelt": ana["col105"]["build_excess_abs"]}
    # two builds of the reference differ beyond print on some days (measured 25 and 21 days,
    # 1.45e-7 and 1.17e-7 cm)
    assert ana["col92"]["n_days_dump_beyond_print"] > 0 and ana["col105"]["n_days_dump_beyond_print"] > 0
    return b


def test_catpa_files_chain_inputs_are_the_reference_inputs(catpa: tuple[dict[str, Any], Any]) -> None:
    rep = catpa[0]
    inp = rep["inputs_files_vs_dump"]
    for k in ("tmin", "tmax", "precipitation (BRK storms vs DAYRAIN)"):
        assert inp[k]["n_bit_identical"] == rep["n_days"] == 3287, (k, inp[k])
    assert inp["srad (RTS)"]["max_rel"] <= rep["rts_build_bound"], inp["srad (RTS)"]


def test_catpa_2015_2023_against_the_full_double_tables(
    catpa: tuple[dict[str, Any], Any], build_abs: dict[str, float]
) -> None:
    rep = catpa[0]
    ref, ours = rep["reference"], rep["dump_chain"]
    assert (ref["pack_days"], ref["melt_days"]) == (600, 195)
    assert (ours["pack_days"], ours["melt_days"]) == (600, 195)
    assert ours["pack_day_mismatch"] == 0 and ours["melt_day_mismatch"] == 0
    assert ref["omsea92_equals_snowpk"]  # .ana column 92 is the pack water equivalent SNOWPK
    for k in ("swe", "smelt"):
        assert ours[k]["max_abs"] <= build_abs[k], (k, ours[k], build_abs[k])
    # the port's water balance closes to rounding every day
    assert ours["balance_max_abs"] <= 1e-12
    assert ours["swe_mm_is_10x"] and ours["melt_split_max_abs"] == 0.0


def test_catpa_yearly_melt_and_the_2015_melt_runoff(catpa: tuple[dict[str, Any], Any]) -> None:
    rep = catpa[0]
    ref, ours = rep["reference"]["yearly"], rep["dump_chain"]["yearly"]
    for y in ref:
        assert ref[y]["pack_days"] == ours[y]["pack_days"] and ref[y]["melt_days"] == ours[y]["melt_days"], y
        assert abs(ref[y]["melt_cm"] - ours[y]["melt_cm"]) <= 1e-9, y
    assert round(ref[2015]["melt_runoff_cm"], 2) == round(ours[2015]["melt_runoff_cm"], 2) == 2.13
    # 2022 is the year with almost no snow in the reference weather (9 pack days; 46-90 in the others)
    assert ours[2022]["pack_days"] == 9 and min(v["pack_days"] for y, v in ours.items() if y != 2022) >= 46


def test_catpa_files_chain_within_the_rts_build_bound(
    catpa: tuple[dict[str, Any], Any], build_abs: dict[str, float]
) -> None:
    rep = catpa[0]
    fc = rep["files_chain"]
    assert (fc["pack_days"], fc["melt_days"]) == (600, 195)
    assert fc["pack_day_mismatch"] == 0 and fc["melt_day_mismatch"] == 0
    for k in ("swe", "smelt"):
        assert fc[f"{k}_beyond_rts_spread_max"] <= build_abs[k], (k, fc)


def test_catpa_printed_columns(
    ana: dict[str, Any], catpa: tuple[dict[str, Any], Any], build_abs: dict[str, float]
) -> None:
    rep = catpa[0]
    for col, k in (("col92", "swe"), ("col105", "smelt")):
        a = ana[col]
        # the port is as close to the shipped binary's print as the rebuilt reference is
        assert a["port_excess_abs_max"] <= build_abs[k] + rep["dump_chain"][k]["max_abs"], (col, a)
        assert a["files_chain_excess_abs_max"] <= build_abs[k] + rep["files_chain"][f"{k}_rts_spread_max"], (
            col,
            a,
        )


def test_pack_temperature_against_the_pet_scenario_tables(
    data: Path, catpa: tuple[dict[str, Any], Any], ana: dict[str, Any]
) -> None:
    rep = sr.compare_pktemp(data, catpa[0]["rts_build_bound"], ana["col92"]["build_excess_rel"])
    assert len(rep) >= 9
    with_snow = [s for s, v in rep.items() if v["pktemp_nonzero_ref"] > 0]
    assert len(with_snow) >= 6, (
        rep
    )  # measured: CA-ER1, CA-MA1, CA-TPA, US-Mj1, US-manilacotton, US_OPE, US_Rockfish
    for s, v in rep.items():
        assert v["n_beyond_limit"] == 0, (s, v)


def test_every_batch_sno_file_is_inside_the_reproduced_scope(data: Path) -> None:
    for site in sr.SITES:
        scen = data / sr.BATCH / site / "Scenario"
        sno = read_sno(sr.ipnames_file(scen, 6))
        PrmsSnowParams.from_sno(sno, 0.7, ipet=sr.scenario_ipet(scen))  # raises outside the scope
        assert sno.cov_type == 1 and not any(sno.tstorm_mo), site


def test_catpa_winter_weather_2022_anomaly(data: Path) -> None:
    """The 2022 winter of the reference weather: almost no precipitation on freezing days (the
    input, not the pack routine, makes 2022 nearly snow-free), while ERA5 has as much as in other
    winters and the tower albedo shows snow cover."""
    if not (data / sr.ERA5_DD).is_file() or not (data / sr.BASE_HH).is_file():
        pytest.skip("AmeriFlux CA-TPA files not found")
    w = sr.catpa_winter_weather(data)["winters"]
    others = [w[y]["brk_precip_freezing_cm"] for y in w if y != "2022"]
    assert w["2022"]["brk_precip_freezing_cm"] < 0.25 * min(others), w["2022"]
    assert w["2022"]["era5_precip_on_met_freezing_days_cm"] > 5.0 * w["2022"]["brk_precip_freezing_cm"]
    assert w["2022"]["albedo_gt_0p5_days"] >= 30


@pytest.mark.slow
def test_fresh_full_period_runs_of_the_batch_scenarios(
    data: Path, catpa: tuple[dict[str, Any], Any], build_abs: dict[str, float]
) -> None:
    recs = sr.run_scenarios(data, sr.SITES)
    rep, _ = sr.compare_scenarios(data, recs, catpa[0]["rts_build_bound"], build_abs)
    assert set(rep) == set(sr.SITES)
    snowy = []
    for site, r in rep.items():
        assert "error" not in r, (site, r)
        assert r["metmod_identity"], site
        for k in ("precipitation", "tmin", "tmax"):  # the inputs equal the printed .ana columns
            assert r[k]["max_ratio_to_half_unit"] <= 1.0 + 1e-9, (site, k, r[k])
        p = r["port"]
        assert (
            p["pack_days"] == r["reference"]["pack_days"] and p["melt_days"] == r["reference"]["melt_days"]
        ), site
        assert p["pack_day_mismatch"] == 0 and p["melt_day_mismatch"] == 0, (site, p)
        assert p["swe_n_beyond_limit"] == 0 and p["smelt_n_beyond_limit"] == 0, (site, p)
        if r["reference"]["pack_days"]:
            snowy.append(site)
    # measured: 12 of the 15 scenarios have snow; two of them (US-S2, US_Rockford_Alfalfa) run a
    # reference-ET PET (IPET = 1), which never sets the sublimation potential: no sublimation
    assert len(snowy) >= 12, snowy
    assert {rep[s]["ipet"] for s in snowy} == {0, 1}
    assert np.isclose(sum(rep[s]["port"]["sublimation_cm"] for s in snowy if rep[s]["ipet"] == 1), 0.0)
