"""Shuttleworth-Wallace PET on 15 RZWQM2 scenarios x 3 years (H1 item 1), against the ``.ana`` output.

Every scenario of ``<data-dir>/narval_mirror/RZWQM_sw_batch`` is run with the RZWQM2 binary for the
first three calendar years of its period; the harness (:mod:`pet_scenarios`, conventions in its
docstring) drives :func:`agrijax.processes.pet.shuttleworth_wallace` with the run's own start-of-day
state and compares PE (col 8), PT (col 9) and PET (col 83). The table is written to
``<data-dir>/validation/pet_scenarios/pet_scenarios_3yr.csv`` (one row per scenario, year and weather
source) with ``<site>_daily.csv`` next to it.

Scenario classes are read from ``rzwqm.dat`` and pinned here: 9 scenarios use the daily S-W path
of the module; 3 use a crop coefficient on the tall reference ET (``pet_method = 1``) and 3 run the
S-W equations hourly on hourly weather (``hourly_weather = 1``). The latter two are other methods of
the reference model: they are reported, and shown to differ, but they are not a check of the module.

Tolerances. The ``.ana`` prints 6 significant digits (<= 5e-6 mm d-1 on a 1 mm d-1 flux); the
reference ET is reproduced at that level. The S-W kernel itself is exact against the reference
routine (``test_pet_dumps.py``: the ``POTEVPHR`` entry/exit dumps of the same 9 scenarios and
years, float64 rounding), and on the dumped run the error of this harness equals the error of the
reconstructed inputs to the ``.ana`` print precision
(``test_pet_dumps.py::test_harness_error_is_the_reconstruction``). The S-W bound of a scenario
here is therefore the reconstruction error measured on its dumped run
(:func:`pet_dumps.attribution`, largest over the 3 years), evaluated in this module from the dump
tables, plus the measured difference between that run and this one: the dumped run is the
instrumented ``-fp-model precise`` build, this one the shipped binary; the state they print
differs slightly, and so does the reconstruction error. Measured 2026-09-25, the per-scenario
maxima of the two runs differ by <= 1.2 % (:data:`CROSS_BUILD_REL` = 5 %) plus <= 6.2e-6 mm d-1
(US_Rockfish PT), within two ``.ana`` print steps (:data:`PRINT_MM` x 2); the prepared ``.MET``
chain stays inside the same margin.

What the reconstruction misses (the dumped ``POTEVPHR`` inputs name the input): the flat residue
mass the routine sees is not the printed end-of-day value of the previous day (up to 7.8e-3 mm d-1
of PE, US-manilacotton), and after a tillage the surface horizon's 1/3- and 15-bar water contents
are changed (``MATILL`` -> ``SOILPR``; 5.6e-3 mm d-1 of PE at US-Mj1). Neither is a kernel error.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pet_scenarios as ps
import pytest

from agrijax.io.rzwqm import read_rzwqm_dat
from agrijax.port.run_fortran import RZWQM_BINARY

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the RZWQM_sw_batch scenarios and the RZWQM2 binary"),
]

#: scenario -> class, pinned (read from rzwqm.dat by :func:`scenario_class`)
SW_DAILY = (
    "CA-ER1",
    "CA-MA1",
    "CA-TPA",
    "US-Mj1",
    "US-Tw2",
    "US-UA1_HartFarm",
    "US-manilacotton",
    "US_OPE",
    "US_Rockfish",
)
KC_TALL = ("US-S2", "US-TW3", "US_Rockford_Alfalfa")
HOURLY_SW = ("US-LYS_NW", "US-LYS_SE", "US-LYS_SW")

#: .ana print step of PE / PT (6 significant digits of a value below 1 cm) [mm d-1]
PRINT_MM = 5.0e-6
#: relative difference allowed between the reconstruction errors of the shipped and the precise run
CROSS_BUILD_REL = 0.05
#: reference ET, cols 81/82 print 6 significant digits of a value in cm: <= 5e-5 mm at 1 cm d-1
REFET_MAX_MM = 1.0e-4  # achieved 6.1e-5 (US-UA1_HartFarm)


def scenario_class(dat: Any) -> str:
    pet = dat.pet
    if int(pet["pet_method"]) != 0:
        return "kc_tall" if int(pet["pet_method"]) == 1 else "kc_short"
    if int(pet["use_shaw"]) or int(pet["use_penflux"]):
        return "shaw_penflux"
    if int(pet["hourly_weather"]):
        return "hourly_sw"
    return "sw_daily"


@pytest.fixture(scope="module")
def batch(data_dir: Path) -> Path:
    b = data_dir / ps.BATCH
    if not b.is_dir():
        pytest.skip(f"{b} not found")
    if not RZWQM_BINARY.is_file():
        pytest.skip(f"RZWQM binary {RZWQM_BINARY} not found")
    return b


@pytest.fixture(scope="module")
def results(
    batch: Path, data_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, dict[str, Any]]]:
    """Run all 15 scenarios (3 years each) and analyse them; write the CSV table."""
    runs = ps.run_all(data_dir, tmp_path_factory.mktemp("pet_runs"), tmp_path_factory.mktemp("pet_stage"))
    out_dir = data_dir / ps.OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for site in ps.SITES:
        rec = runs[site]
        entry: dict[str, Any] = {"rec": rec}
        if rec["ok"]:
            x, sims, st = ps.analyse(rec, data_dir)
            entry.update(x=x, sims=sims, rows=st)
            ps.daily_frame(x, sims).to_csv(out_dir / f"{site}_daily.csv", index=False)
            cls = scenario_class(read_rzwqm_dat(batch / site / "Scenario" / "rzwqm.dat"))
            for r in st:
                r["class"] = cls
            rows.extend(st)
        else:
            rows.append({"site": site, "error": rec["error"]})
        res[site] = entry
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "pet_scenarios_3yr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    yield res


@pytest.fixture(scope="module")
def bounds(data_dir: Path) -> dict[str, tuple[float, float]]:
    """Per S-W scenario ``(PE, PT)`` bound [mm d-1] from the reconstruction error on its dumped run."""
    import datetime as _dt

    import pet_dumps as pdm

    from agrijax.io.rzwqm.layers import simulation_start

    out = {}
    for site in SW_DAILY:
        pe, px = pdm.table_paths(data_dir, site)
        run = data_dir / pdm.RUNS / site
        if not (pe.is_file() and px.is_file() and (run / "LAYER.PLT").is_file()):
            pytest.skip(f"POTEVPHR dump tables / run of {site} not found")
        src = data_dir / pdm.RUNS / "_stage" / site / "Scenario"
        if not src.is_dir():
            src = data_dir / ps.BATCH / site / "Scenario"
        st = simulation_start(src / "IPNAMES.DAT").astype(object)
        rec = dict(site=site, ok=True, start=st, end=_dt.date(st.year + 2, 12, 31), source=src, out=run)
        x = ps.site_inputs(rec, data_dir)
        dat = read_rzwqm_dat(data_dir / ps.BATCH / site / "Scenario" / "rzwqm.dat")
        x.site_consts["params"] = ps.pet_params(dat)
        a = pdm.attribution(pdm.load_site(data_dir, site), x)["all reconstructed"]
        out[site] = (
            a["pe_max_mm"] * (1.0 + CROSS_BUILD_REL) + 2.0 * PRINT_MM,
            a["pt_max_mm"] * (1.0 + CROSS_BUILD_REL) + 2.0 * PRINT_MM,
        )
    return out


def _need(results: dict[str, dict[str, Any]], site: str) -> dict[str, Any]:
    e = results[site]
    if not e["rec"]["ok"]:
        pytest.fail(f"{site}: {e['rec']['error']}\n{e['rec'].get('log_tail', '')}")
    return e


@pytest.mark.parametrize("site", ps.SITES)
def test_run_completes_three_years(results: dict[str, dict[str, Any]], site: str) -> None:
    e = _need(results, site)
    x = e["x"]
    assert len(np.unique(x.days.year)) == ps.N_YEARS
    assert x.days[0] == np.datetime64(e["rec"]["start"]) and x.days[-1] == np.datetime64(e["rec"]["end"])
    for v in ("pe", "pt", "pet"):
        assert np.isfinite(x.ref[v]).all()
    # col 83 is PE + PT to print precision (three values of 6 significant digits each)
    np.testing.assert_allclose(x.ref["pet"], x.ref["pe"] + x.ref["pt"], rtol=1.5e-5, atol=2e-5)


def test_scenario_classes(batch: Path) -> None:
    """The class of each scenario, read from its rzwqm.dat, is the pinned one."""
    got = {s: scenario_class(read_rzwqm_dat(batch / s / "Scenario" / "rzwqm.dat")) for s in ps.SITES}
    want = (
        {s: "sw_daily" for s in SW_DAILY}
        | {s: "kc_tall" for s in KC_TALL}
        | {s: "hourly_sw" for s in HOURLY_SW}
    )
    assert got == want


@pytest.mark.parametrize("site", SW_DAILY)
@pytest.mark.parametrize("weather", ["ana", "met"])
def test_sw_daily_against_ana(
    results: dict[str, dict[str, Any]], bounds: dict[str, tuple[float, float]], site: str, weather: str
) -> None:
    """Per year: PE, PT max |error| within the reconstruction error measured on the dumped run."""
    rows = [r for r in _need(results, site)["rows"] if r["weather"] == weather]
    assert len(rows) == ps.N_YEARS
    pe_b, pt_b = bounds[site]
    for r in rows:
        tag = f"{site} {r['year']} {weather}"
        assert r["pe_max_abs_mm"] <= pe_b, (tag, r["pe_max_abs_mm"], pe_b, r["pe_worst_day"])
        assert r["pt_max_abs_mm"] <= pt_b, (tag, r["pt_max_abs_mm"], pt_b, r["pt_worst_day"])


def _pooled_max(results: dict[str, dict[str, Any]], fn) -> float:
    worst = 0.0
    for site in SW_DAILY:
        x = _need(results, site)["x"]
        sim = fn(x)
        for v in ("pe", "pt"):
            worst = max(worst, float(np.abs(sim[v] - x.ref[v])[1:].max()))
    return worst


def test_input_conventions_are_measured(results: dict[str, dict[str, Any]]) -> None:
    """Moving any start-of-day input by one step of its timing is at least 10x worse than the residual."""
    import dataclasses

    base = _pooled_max(results, lambda x: ps.simulate(x, "ana"))

    def nxt(a: np.ndarray) -> np.ndarray:
        return np.concatenate([a[1:], a[-1:]])

    ablations = {
        "LAI of the same row": lambda x: ps.simulate(dataclasses.replace(x, lai=x.extra["lai_row"]), "ana"),
        "height of the same row": lambda x: ps.simulate(dataclasses.replace(x, height=nxt(x.height)), "ana"),
        "end-of-day residue": lambda x: ps.simulate(
            dataclasses.replace(x, residue_mass=x.extra["residue_end_kg_ha"]), "ana"
        ),
        "end-of-day surface water": lambda x: ps.simulate(x, "ana", theta=nxt(x.theta)),
        "field radiation RTS also as RTH": lambda x: ps.simulate(
            x, "ana", srad_horizontal=x.weather["ana"]["srad"]
        ),
    }
    for label, fn in ablations.items():
        worst = _pooled_max(results, fn)
        assert worst > 10.0 * base, (label, worst, base)


def test_tillage_mixing(results: dict[str, dict[str, Any]], bounds: dict[str, tuple[float, float]]) -> None:
    """On tillage days the tillage-zone mean water content reproduces the reference, the node value not."""
    mixed, unmixed, n = 0.0, 0.0, 0
    for site in SW_DAILY:
        e = _need(results, site)
        x, sims = e["x"], e["sims"]
        k = np.flatnonzero(x.theta != x.extra["theta_unmixed"])
        n += len(k)
        for v in ("pe", "pt"):
            if len(k):
                mixed = max(mixed, float(np.abs(sims["ana"][v] - x.ref[v])[k].max()))
                unmixed = max(unmixed, float(np.abs(sims["ana_unmixed"][v] - x.ref[v])[k].max()))
    assert n >= 10
    assert mixed <= max(b[0] for b in bounds.values())
    assert unmixed > 3.0 * mixed, (mixed, unmixed)


@pytest.mark.parametrize("site", ps.SITES)
def test_reference_et(results: dict[str, dict[str, Any]], site: str) -> None:
    """Cols 81/82 (REF_ET): print precision with daily weather; explained elsewhere.

    * hourly weather (US-LYS_*): the reference sums 24 hourly REF_ET calls (Rzday.for 1197-1276);
    * US_Rockford_Alfalfa (CO2 407 ppm): the reference ET is scaled by DSSAT's CO2 transpiration
      ratio (Rzday.for line 1195), which differs from 1 above 330 ppm while the canopy has
      LAI >= 0.01; on bare days it is 1 and the module matches.
    """
    r = _need(results, site)["rows"][0]
    if site in HOURLY_SW:
        assert r["refet_tall_rmse_mm"] > 0.1
        return
    if site == "US_Rockford_Alfalfa":
        assert r["refet_tall_max_abs_bare_mm"] < REFET_MAX_MM
        assert r["refet_short_max_abs_bare_mm"] < REFET_MAX_MM
        assert r["refet_tall_max_abs_mm"] > 0.1
        return
    assert r["refet_tall_max_abs_mm"] < REFET_MAX_MM, r
    assert r["refet_short_max_abs_mm"] < REFET_MAX_MM, r


@pytest.mark.parametrize("site", KC_TALL + HOURLY_SW)
def test_other_methods_are_not_the_daily_sw(results: dict[str, dict[str, Any]], site: str) -> None:
    """Documentation, not validation: a run that used another PET method is far from the daily S-W.

    These scenarios are outside the module's scope (``pet_method = 1`` or hourly weather); the
    assertion only pins that they stay classified as such (RMSE > 0.1 mm d-1 in every year,
    measured >= 0.14).
    """
    for r in _need(results, site)["rows"]:
        if r["weather"] == "ana":
            assert max(r["pe_rmse_mm"], r["pt_rmse_mm"]) > 0.1, r["year"]


def test_summary_written(results: dict[str, dict[str, Any]], data_dir: Path) -> None:
    p = data_dir / ps.OUT_SUBDIR / "pet_scenarios_3yr.csv"
    with open(p, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2 * ps.N_YEARS * len(ps.SITES)
    assert os.path.getsize(p) > 0
