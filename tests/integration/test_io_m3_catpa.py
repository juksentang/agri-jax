"""The io of CA-TPA 2015-2023 against the RZWQM2 4.6 reference inputs and run.

Data (kept outside the repository, ``allow_skip``):

* ``<data>/narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario``: ``rzwqm.dat``, ``IPNAMES.DAT``,
  ``CA-TPA.MET``, ``CA-TPA.BRK``, ``MZCER040.{CUL,ECO,SPE}``;
* ``$AGRI_JAX_DSSAT/source/Data/Genotype/MZCER048.ECO`` (``TSEN``, ``CDAY``);
* ``<data>/dumps/tables/rzwqm46_catpa2015_2023``: the daily entry/exit tables (``PHYSCL``,
  ``DSSATDRV``, ``ROOTWU``, ``MZ_GROSUB``) and the per-event ``EVNTRO`` cases of the
  instrumented RZWQM2 4.6 run of CA-TPA 2015-2023 (1024 events);
* ``<data>/catpa/base_2015_2023/MANAGE.OUT`` of the 2015-2023 base run.

Checked:

1. write-back: ``CA-TPA.BRK`` and the embedded crop's 4.0 genotype files are written back byte
   for byte;
2. storms: every ``EVNTRO`` rain event (no snowmelt, no irrigation) has the breakpoints of our
   day arrays (count ``NBP``, interval lengths ``diff(BPWHEN)`` [h], depths ``diff(BPMUCH)``
   [cm]) and every other ``EVNTRO`` event is a snowmelt event (so the scenario has no
   irrigation, as its zero irrigation operations say); the breakpoint storms without an ``EVNTRO``
   rain event are the ones RZWQM2's snow routine takes;
3. seasons: the sow / harvest flags of the event table, the season table and the crop days of
   the reference (``DSSATDRV`` rows, ``MZ_GROSUB`` ``YRPLT``, ``HDATE``) agree, and so do
   ``MANAGE.OUT``'s planting and harvest dates and the per-season planting values;
4. CERES-Maize parameters: equal (in the crop's REAL precision) to what the embedded crop read
   (``DSSATDRV`` ``PLANTVAR``, ``MZ_GROSUB`` entry, ``ROOTWU`` entry soil) in every season;
5. weather: the prepared ``.MET`` against ``PHYSCL`` RTH (all 3287 days) and the crop's
   temperatures and CO2 against ``DSSATDRV`` (crop days);
6. the whole forcing pytree: shapes, finiteness and the contract's units.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pandas as pd
import pytest

from agrijax.core.state import field_metadata
from agrijax.io.catpa import read_manage_out
from agrijax.io.dssat.genotype import read_cul, read_eco, read_spe, write_cul, write_eco, write_spe
from agrijax.io.rzwqm.dat import read_rzwqm_dat
from agrijax.io.rzwqm.met import read_brk, write_brk
from agrijax.io.rzwqm.storms import irrigation_operations, read_met_modifiers, storm_arrays
from agrijax.port import dumps
from agrijax.sites.catpa_m3 import (
    CERES_SPECIES_KEYS,
    CatpaPaths,
    catpa_m3_inputs,
    ceres_maize_params,
    ceres_soil_from_dssatdrv,
    event_table,
    latitude_deg,
    season_table,
)

pytestmark = pytest.mark.allow_skip(reason="needs the CA-TPA scenario and RZWQM2 4.6 dumps")

TABLES = Path("dumps/tables/rzwqm46_catpa2015_2023")
EVNTRO = TABLES / "cases" / "EVNTRO"
MANAGE = Path("catpa/base_2015_2023/MANAGE.OUT")
START, END = np.datetime64("2015-01-01"), np.datetime64("2023-12-31")
DAYS = np.arange(START, END + np.timedelta64(1, "D"))
N_SEASON = 7
N_CROP_DAYS = 1088
#: EVNTRO events of the run: rain storms, snowmelt events
N_RAIN_EVENTS, N_SNOWMELT_EVENTS = 829, 195


def _paths(data_dir: Path) -> CatpaPaths:
    p = CatpaPaths.under(data_dir, os.environ.get("AGRI_JAX_DSSAT"))
    need = [p["rzwqm.dat"], p["CA-TPA.BRK"], p["CA-TPA.MET"], p["IPNAMES.DAT"], p["MZCER040.CUL"], p.eco48]
    for f in need:
        if not f.is_file():
            pytest.skip(f"{f} not found")
    return p


def _table(data_dir: Path, name: str) -> Any:
    f = data_dir / TABLES / f"{name}.npz"
    if not f.is_file():
        pytest.skip(f"{f} not found")
    return dumps.load_table(f)[0]


def _yrdoy_to_day(yrdoy: np.ndarray) -> np.ndarray:
    y = np.asarray(yrdoy, dtype=np.int64) // 1000
    d = np.asarray(yrdoy, dtype=np.int64) % 1000
    return (y - 1970).astype("datetime64[Y]").astype("datetime64[D]") + (d - 1).astype("timedelta64[D]")


def _f32(x: Any) -> np.ndarray:
    return np.asarray(np.asarray(x, dtype=np.float64), dtype=np.float32)


# ---------------------------------------------------------------------------- 1. write-back
def test_input_files_write_back_byte_identical(data_dir: Path, tmp_path: Path) -> None:
    p = _paths(data_dir)
    out = write_brk(read_brk(p["CA-TPA.BRK"]), tmp_path / "CA-TPA.BRK")
    assert out.read_bytes() == p["CA-TPA.BRK"].read_bytes()
    for name, rd, wr in (("MZCER040.CUL", read_cul, write_cul), ("MZCER040.ECO", read_eco, write_eco)):
        assert wr(rd(p[name]), tmp_path / name).read_bytes() == p[name].read_bytes(), name
    assert (
        write_spe(read_spe(p["MZCER040.SPE"]), tmp_path / "s.SPE").read_bytes()
        == p["MZCER040.SPE"].read_bytes()
    )


# ---------------------------------------------------------------------------- 2. storms
def _events(data_dir: Path) -> list[tuple[np.datetime64, dict[str, np.ndarray]]]:
    files = sorted((data_dir / EVNTRO).glob("catpa2015_2023_d*.npz"))
    if not files:
        pytest.skip(f"no EVNTRO cases under {data_dir / EVNTRO}")
    out = []
    for f in files:
        e = dumps.load_case(f).entry
        day = _yrdoy_to_day(np.array([int(e["IYYY"]) * 1000 + int(e["JDAY"])]))[0]
        out.append((day, e))
    return out


def test_storm_arrays_match_evntro_breakpoints(data_dir: Path) -> None:
    p = _paths(data_dir)
    s = storm_arrays(DAYS, read_brk(p["CA-TPA.BRK"]), met_modifiers=read_met_modifiers(p["IPNAMES.DAT"]))
    pos = {d: i for i, d in enumerate(DAYS)}
    rain, melt = [], 0
    d_dur = d_dep = 0.0
    for day, e in _events(data_dir):
        if float(e["SMELT"]) > 0.0 or float(e["AIRR"]) > 0.0:
            melt += 1
            assert float(e["SMELT"]) > 0.0, f"{day}: AIRR without snowmelt (irrigation)"
            continue
        i = pos[day]
        nbp = int(e["NBP"])
        when = np.diff(np.concatenate([[0.0], np.asarray(e["BPWHEN"], float)[:nbp]]))
        much = np.diff(np.concatenate([[0.0], np.asarray(e["BPMUCH"], float)[:nbp]]))
        assert s.event[i] >= 0, f"{day}: EVNTRO rain event, no breakpoint storm"
        assert int(np.count_nonzero(s.duration[i])) == nbp, day
        d_dur = max(d_dur, float(np.max(np.abs(s.duration[i, :nbp] - when))))
        d_dep = max(d_dep, float(np.max(np.abs(s.depth[i, :nbp] - much))))
        rain.append(i)
    print(
        f"EVNTRO rain {len(rain)}, snowmelt {melt}; max |d duration| {d_dur:.3e} h, |d depth| {d_dep:.3e} cm"
    )
    assert (len(rain), melt) == (N_RAIN_EVENTS, N_SNOWMELT_EVENTS)
    assert irrigation_operations(read_rzwqm_dat(p["rzwqm.dat"])) == 0
    assert d_dur <= TOL_DURATION_H and d_dep <= TOL_DEPTH_CM
    # the storms RZWQM2 did not run as an infiltration event: its snow routine took them
    pe = _table(data_dir, "physcl_entry")
    j = pe.index_of([int(pd.Timestamp(d).strftime("%Y%j")) for d in DAYS])
    assert np.all(j >= 0)
    snowpk = np.asarray(pe.values["SNOWPK"], float)[j]
    tmean = 0.5 * (np.asarray(pe.values["TMAX"], float)[j] + np.asarray(pe.values["TMIN"], float)[j])
    other = np.setdiff1d(np.nonzero(s.event >= 0)[0], rain)
    snow_branch = (snowpk[other] > 0.0) | (tmean[other] <= 0.0)
    print(
        f"breakpoint storm days {int(np.sum(s.event >= 0))}, without an EVNTRO rain event {len(other)}, "
        f"of them snowpack or TM <= 0: {int(snow_branch.sum())}"
    )
    assert len(other) == N_STORMS_TO_SNOW
    assert bool(np.all(snow_branch)), [str(DAYS[i]) for i in other[~snow_branch]][:10]


# ---------------------------------------------------------------------------- 3. seasons
def test_season_flags_match_reference(data_dir: Path) -> None:
    p = _paths(data_dir)
    dat = read_rzwqm_dat(p["rzwqm.dat"])
    ev = event_table(dat, DAYS)
    seasons = season_table(dat, START, END)
    sow_days = DAYS[np.asarray(ev.sow, bool)]
    harvest_days = DAYS[np.asarray(ev.harvest, bool)]
    assert seasons.n_season == N_SEASON
    np.testing.assert_array_equal(sow_days, seasons.sow)
    np.testing.assert_array_equal(harvest_days, seasons.harvest)
    assert float(np.abs(np.asarray(ev.irrig_cm)).sum()) == 0.0
    # MANAGE.OUT of the base run
    mf = data_dir / MANAGE
    if not mf.is_file():
        pytest.skip(f"{mf} not found")
    man = read_manage_out(mf)
    man = man[(man["date"] >= pd.Timestamp(START)) & (man["date"] <= pd.Timestamp(END))]
    np.testing.assert_array_equal(
        man.loc[man["event"] == "planting", "date"].to_numpy().astype("datetime64[D]"), seasons.sow
    )
    np.testing.assert_array_equal(
        man.loc[man["event"] == "harvest", "date"].to_numpy().astype("datetime64[D]"), seasons.harvest
    )
    np.testing.assert_array_equal(
        man.loc[man["event"] == "planting", "value"].to_numpy() / 1.0e4, seasons.pltpop
    )
    # the instrumented run: sowing days, harvest days, crop days, planting values
    dd = _table(data_dir, "dssatdrv_exit")
    mg = _table(data_dir, "mz_grosub_exit")
    np.testing.assert_array_equal(np.unique(np.asarray(mg.values["YRPLT"])), seasons.yrplt)
    hd = np.unique(np.asarray(dd.values["HDATE"]))
    np.testing.assert_array_equal(hd[hd > 0], seasons.harvest_yrdoy)
    crop_days = _yrdoy_to_day(np.asarray(dd.date))
    assert len(crop_days) == N_CROP_DAYS
    assert bool(np.all(seasons.in_crop(crop_days)))
    k = seasons.season_index(crop_days)
    per = np.bincount(k, minlength=N_SEASON)
    span = (seasons.harvest - seasons.sow).astype(int) + 1
    print(f"DSSATDRV crop days per season {per.tolist()}, sow..harvest days {span.tolist()}")
    np.testing.assert_array_equal(per, span)  # every day from sowing to harvest is a crop day
    for name, ours in (("PLTPOP", seasons.pltpop), ("SDEPTH", seasons.sdepth), ("ROWSPC", seasons.rowspc)):
        ref = np.asarray(dd.values[f"PLANTVAR%{name}"], float)
        np.testing.assert_array_equal(ref, ours[k], err_msg=name)


# ---------------------------------------------------------------------------- 4. CERES parameters
def test_ceres_params_match_embedded_crop(data_dir: Path) -> None:
    p = _paths(data_dir)
    dat = read_rzwqm_dat(p["rzwqm.dat"])
    seasons = season_table(dat, START, END)
    dd = _table(data_dir, "dssatdrv_exit")
    ge = _table(data_dir, "mz_grosub_entry")
    rw = _table(data_dir, "rootwu_entry")
    soil = ceres_soil_from_dssatdrv(dd.values)
    for key in ("dlayr", "ll", "sat"):  # the soil ROOTWU received
        nl = len(soil["dlayr"])
        np.testing.assert_array_equal(_f32(soil[key]), _f32(np.asarray(rw.values[key.upper()])[0, :nl]), key)
    vr = np.asarray(dd.values["VARNOR"])
    assert {str(x.decode() if isinstance(x, bytes) else x).strip() for x in vr} == {"IB0012"}
    kd = seasons.season_index(_yrdoy_to_day(np.asarray(dd.date)))
    kg = seasons.season_index(_yrdoy_to_day(np.asarray(ge.date)))
    checked = 0
    for s in range(seasons.n_season):
        prm = ceres_maize_params(p, soil, seasons, s)
        cv, sp = prm.cultivar, prm.species
        rows_d, rows_g = np.nonzero(kd == s)[0], np.nonzero(kg == s)[0]
        assert len(rows_d) and len(rows_g)
        ours_d = {f"PLANTVAR%{k.upper()}": getattr(cv, k) for k in ("p1", "p2", "p5", "g2", "g3", "phint")}
        ours_d |= {
            "PLANTVAR%PLTPOP": prm.pltpop,
            "PLANTVAR%SDEPTH": prm.sdepth,
            "PLANTVAR%ROWSPC": prm.rowspc,
        }
        ours_g = {k: getattr(cv, k.lower()) for k in ("P1", "P2", "P5", "G2", "G3", "PHINT", "RUE")}
        ours_g |= {k: getattr(sp, k.lower()) for k in CERES_SPECIES_KEYS if k in ge.values}
        ours_g |= {"PORMIN": sp.pormin, "RWUMX": sp.rwumx, "PLTPOP": prm.pltpop, "ROWSPC": prm.rowspc}
        ours_g |= {"SLPF": prm.soil.slpf}
        for tab, rows, ours in ((dd, rows_d, ours_d), (ge, rows_g, ours_g)):
            for name, v in ours.items():
                ref = np.asarray(tab.values[name])[rows]
                assert np.all(_f32(ref) == _f32(v)), (s, name, float(v), np.unique(ref)[:3])
                checked += 1
        assert int(prm.yrplt) == int(np.unique(np.asarray(ge.values["YRPLT"])[rows_g])[0])
    print(f"CERES parameters equal to the embedded crop's in REAL precision: {checked} (season, value) pairs")


# ---------------------------------------------------------------------------- 5. weather
def test_weather_matches_physcl_and_dssatdrv(data_dir: Path) -> None:
    p = _paths(data_dir)
    soil = ceres_soil_from_dssatdrv(_table(data_dir, "dssatdrv_exit").values)
    inp = catpa_m3_inputs(soil, paths=p)
    w = inp.forcing["weather"]
    assert "srad_horizontal" not in inp.forcing
    rth = np.asarray(w.srad_horizontal)
    pe = _table(data_dir, "physcl_exit")
    j = pe.index_of([int(pd.Timestamp(d).strftime("%Y%j")) for d in DAYS])
    assert np.all(j >= 0)
    diffs = {}
    for ours, ref in (
        ("tmin", "TMIN"),
        ("tmax", "TMAX"),
        ("rh", "RH"),
        ("wind_run", "U"),
    ):
        diffs[ref] = float(
            np.max(np.abs(np.asarray(getattr(w, ours)) - np.asarray(pe.values[ref], float)[j]))
        )
    diffs["RTH"] = float(np.max(np.abs(rth - np.asarray(pe.values["RTH"], float)[j])))
    dd = _table(data_dir, "dssatdrv_exit")
    dd_days = _yrdoy_to_day(np.asarray(dd.date))
    i = np.searchsorted(DAYS, dd_days)
    c = inp.forcing["crop"]
    for ours, ref in (("tmax", "TMAXR"), ("tmin", "TMINR"), ("co2", "CO2R")):
        diffs[ref] = float(
            np.max(np.abs(np.asarray(getattr(c, ours))[i] - np.asarray(dd.values[ref], float)))
        )
    diffs["XLATR"] = abs(
        latitude_deg(read_rzwqm_dat(p["rzwqm.dat"])) - float(np.asarray(dd.values["XLATR"])[0])
    )
    print("max |ours - reference|: " + ", ".join(f"{k} {v:.3e}" for k, v in diffs.items()))
    for k, v in diffs.items():
        assert v <= TOL_WEATHER.get(k, 0.0), (k, v)


# ---------------------------------------------------------------------------- 6. the pytree
def test_forcing_pytree_shapes_units(data_dir: Path) -> None:
    from agrijax.core.events import EventTable
    from agrijax.iface.surface import DailyWeather
    from agrijax.processes.crop.ceres_maize.state import CeresForcing

    p = _paths(data_dir)
    soil = ceres_soil_from_dssatdrv(_table(data_dir, "dssatdrv_exit").values)
    inp = catpa_m3_inputs(soil, paths=p)
    f = inp.forcing
    assert isinstance(f["weather"], DailyWeather) and "soil" not in f
    assert isinstance(f["events"], EventTable) and isinstance(f["crop"], CeresForcing)
    for leaf in jax.tree_util.tree_leaves(f):
        a = np.asarray(leaf)
        assert a.shape[0] == len(DAYS)
        if a.dtype.kind == "f":
            assert bool(np.all(np.isfinite(a)))
    # the contract's units (port P8 of the coupling contract)
    want = {
        (DailyWeather, "tmin"): "degC",
        (DailyWeather, "tmax"): "degC",
        (DailyWeather, "srad"): "MJ m-2 d-1",
        (DailyWeather, "rh"): "percent",
        (DailyWeather, "wind_run"): "km d-1",
        (EventTable, "irrig_cm"): "cm",
    }
    for (cls, name), unit in want.items():
        assert field_metadata(cls)[name]["unit"] == unit, (cls.__name__, name)
    # storm totals per day equal the breakpoint file's totals of the storms STMINP keeps [cm]
    brk = read_brk(p["CA-TPA.BRK"])
    dep = brk.events["depth_in"].to_numpy(float)
    when = np.asarray(brk.events["date"].to_numpy(), dtype="datetime64[D]")
    keep = (dep >= 0.01) & (when >= START) & (when <= END)
    np.testing.assert_allclose(
        float(np.asarray(f["soil"].depth).sum()), float(dep[keep].sum()) * 2.54, rtol=1e-12
    )
    assert len(inp.crop_params) == N_SEASON and inp.seasons.n_season == N_SEASON
    assert {s.name for s in inp.substitutions} >= {"cultivar.tsen", "cultivar.cday", "soil.*"}


# tolerances. Storm breakpoints: the reference's own arithmetic is reproduced, measured 0 (2026-09-26)
TOL_DURATION_H = 0.0
TOL_DEPTH_CM = 0.0
#: breakpoint storms of 2015-2023 that RZWQM2 4.6 gave to its snow routine (measured 2026-09-26)
N_STORMS_TO_SNOW = 239
#: every prepared weather value is the reference's to the bit
TOL_WEATHER: dict[str, float] = {}
