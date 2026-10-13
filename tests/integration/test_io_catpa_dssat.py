"""The CA-TPA DSSAT-CSM 4.8.6 reference case: conversion checks against the RZWQM2 4.6
reference and two dscsm048 runs of the maize seasons of the local scenario.

Data (kept outside the repository, ``allow_skip``): the CA-TPA scenario under
``<data>/narval_mirror/...`` (``rzwqm.dat``, ``MZDSSAT.RZX``, ``IPNAMES.DAT``, ``.MET``, ``.BRK``,
``MZCER040.*``); the ``DSSATDRV`` and ``MZ_GROSUB`` tables of the instrumented RZWQM2 4.6 run
(``<data>/dumps/tables/rzwqm46_catpa2015_2023``); ``LAYER.PLT`` of the base run
(``<data>/catpa/base_2015_2023``); the DSSAT-CSM v4.8.6.0 engine (``$AGRI_JAX_DSSAT``: ``build486``
dscsm048, ``source/Data/Genotype``).

The case is written to ``$AGRI_JAX_DATA/dssat_catpa/{catpa,stock}`` and run there (outputs in
``dssat_catpa/out_{catpa,stock}``); the per-season comparison is written to
``$AGRI_JAX_DATA/validation/aj_w6/catpa_dssat.json``. The values of the local scenario a test
pins (number of seasons, the control record's heads) are read from
``<data>/catpa/expected/test_io_catpa_dssat.json`` (:mod:`_scenario_expected`).

Checked:

1. soil: the Brooks-Corey LL / DUL at the scenario's heads on the crop's layers equal the embedded
   crop's ``SOILPROP%LL`` / ``DUL`` (float32); ``SAT`` below the tilled top layers, the root growth
   factors (``.RZX`` -> ``SOILPROP%WR``), ``SLPF`` and the organic C equal the reference's; the
   ``rzwqm.dat`` FC33 / WP columns agree with the curve to their 4 printed decimals;
2. weather: TMAX / TMIN of the case equal what RZWQM2 passed the embedded crop (``DSSATDRV``
   ``TMAXR`` / ``TMINR``) and the day's rain the crop saw (``RAINR``) on every crop day; SRAD
   equals the reference's daily radiation ``PHYSCL`` ``RTH`` (the prepared ``.MET`` value), while
   the crop's ``SRADR`` is ``RTS``, the re-sum of RZWQM2's hourly radiation (``HRTS``), which the
   case does not reproduce (strict xfail, measured difference in the test); the ``.WTH`` files
   read back (DSSAT field rule) within the written decimals;
3. genotype: the ``catpa`` files carry exactly the cultivar, ecotype and species values of our
   CERES-Maize parameters for CA-TPA (``ceres_maize_params``); every 4.0 / 4.8.6 difference is listed;
4. dscsm048 read what was written: 2 x (number of seasons) treatments per set finish, ``PDAT`` / ``HDAT`` are the sowing
   and harvest days, the ``OVERVIEW.OUT`` echo of the soil (LL, DUL, SAT, initial SW, NO3, OC) and
   of the cultivar equals the inputs to its printed decimals;
5. recorded (not asserted): per season stage dates, LAI max, biomass and yield of the four DSSAT
   runs against the RZWQM2 embedded 4.0.2 crop (``MZ_GROSUB``) and our CERES driven by the RZWQM2
   crop water (``validation/aj_w4/catpa_seasons.json``, from the CA-TPA seasons test).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _scenario_expected import expected

from agrijax.io.dssat.filex import read_filex
from agrijax.io.dssat.genotype import read_cul, read_eco, read_spe
from agrijax.io.dssat.sol import read_sol
from agrijax.io.dssat.wth import read_wth
from agrijax.io.rzwqm.dat import read_rzwqm_dat
from agrijax.io.rzwqm.events import read_management
from agrijax.io.rzwqm.rzx import read_rzx
from agrijax.port import dumps
from agrijax.sites.catpa_dssat import (
    CASE_SOIL_ID,
    DSSAT_ONLY_SETTINGS,
    GENOTYPE_SET_NAMES,
    N_OPTIONS,
    WTH_DECIMALS,
    OrganicProfile,
    brooks_corey_theta,
    build_case,
    catpa_weather,
    dssat_layer_bottoms_cm,
    genotype_differences,
    genotype_sets,
    layer_average,
    read_overview_cultivar,
    read_overview_soil,
    run_case,
    season_results,
    treatment_numbers,
)
from agrijax.sites.catpa_m3 import (
    CERES_SPECIES_KEYS,
    CatpaPaths,
    ceres_maize_params,
    ceres_soil_from_dssatdrv,
)

pytestmark = pytest.mark.allow_skip(reason="needs the CA-TPA scenario, RZWQM2 4.6 tables and dscsm048")

TABLES = Path("dumps/tables/rzwqm46_catpa2015_2023")
LAYER_PLT = Path("catpa/base_2015_2023/LAYER.PLT")
CASE_DIR = Path("dssat_catpa")
REPORT = Path("validation/aj_w6/catpa_dssat.json")
W4_REPORT = Path("validation/aj_w4/catpa_seasons.json")
START, END = "2015-01-01", "2021-12-31"
#: float32 (REAL) unit roundoff: the reference holds its soil and weather values in REAL
_EPS32 = 2.0**-23
#: LL / DUL / OC on the crop's layers: RZWQM2 rounds the double WC to REAL per node (1 rounding)
#: and averages two nodes at most per layer boundary in REAL (realMATCH: a product and a sum per
#: node, a division): 6 roundings of values < 1 bound the difference to our double evaluation
TOL_LAYER_REAL = 6 * _EPS32
#: the .SOL values are written with 4 decimals
TOL_SOL_FILE = 0.5e-4
#: the OVERVIEW.OUT echo prints LL / DUL / SAT / SW with 3 decimals, NO3 / OC with 2
TOL_ECHO_3, TOL_ECHO_2 = 0.5e-3, 0.5e-2
#: RZWQM2 hands the crop RTS as REAL: the re-sum of 24 REAL hourly values (tests/integration/
#: test_io_m3_catpa.py, _RTS_FLOAT32_BOUND)
TOL_SRAD = 24 * _EPS32 * 45.0
#: RTS [MJ m-2 d-1] = 0.0036 x the sum of the 24 hourly HRTS [W m-2] of PHYSCL, each held as REAL:
#: measured 4.7e-6 against a float32 re-sum (2026-10-13), the bound is the float32 rounding of
#: 24 terms of at most 45 MJ m-2 d-1
TOL_RTS_RESUM = 24 * _EPS32 * 45.0
#: RAINR [cm] is the REAL day total of the storm's breakpoint increments (16 bounds the increments
#: of a day; 10 cm bounds the day total): one rounding per increment
TOL_RAIN_CM = 16 * _EPS32 * 10.0


def _yrdoy_to_day(yrdoy: Any) -> np.ndarray:
    y = np.asarray(yrdoy, dtype=np.int64) // 1000
    d = np.asarray(yrdoy, dtype=np.int64) % 1000
    return (y - 1970).astype("datetime64[Y]").astype("datetime64[D]") + (d - 1).astype("timedelta64[D]")


def _yrdoy(d: Any) -> int:
    t = pd.Timestamp(d)
    return int(t.year * 1000 + t.dayofyear)


# ---------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def ref(data_dir: Path) -> dict[str, Any]:
    p = CatpaPaths.under(data_dir, os.environ.get("AGRI_JAX_DSSAT"))
    for n in ("rzwqm.dat", "MZDSSAT.RZX", "IPNAMES.DAT", "CA-TPA.MET", "CA-TPA.BRK", "MZCER040.CUL"):
        if not p[n].is_file():
            pytest.skip(f"{p[n]} not found")
    tabs = {}
    for name in ("dssatdrv_entry", "dssatdrv_exit", "mz_grosub_exit", "physcl_exit"):
        f = data_dir / TABLES / f"{name}.npz"
        if not f.is_file():
            pytest.skip(f"{f} not found")
        tabs[name] = dumps.load_table(f)[0]
    lp = data_dir / LAYER_PLT
    if not lp.is_file():
        pytest.skip(f"{lp} not found")
    from agrijax.io.rzwqm.layers import read_layer_output
    from agrijax.port.run_fortran import dscsm_paths

    exe, data48 = dscsm_paths()
    if not exe.is_file() or not (data48 / "Genotype" / "MZCER048.CUL").is_file():
        pytest.skip(f"DSSAT-CSM 4.8.6 engine not found ({exe})")
    dat = read_rzwqm_dat(p["rzwqm.dat"])
    ent = tabs["dssatdrv_entry"]
    from agrijax.sites.catpa_m3 import season_table

    seasons = season_table(dat, START, END)
    exp = expected(data_dir, __file__)
    assert seasons.n_season == exp["n_season"]
    r0 = int(ent.index_of([int(seasons.yrplt[0])])[0])
    nn = int(np.asarray(ent.values["NNR"])[r0])
    organic = OrganicProfile(
        node_bottoms_cm=np.asarray(ent.values["TLTR"], float)[r0, :nn],
        oc_pct=np.asarray(ent.values["RZOC"], float)[r0, :nn],
        on_pct=np.asarray(ent.values["RZON"], float)[r0, :nn],
        ph=np.asarray(ent.values["CGPH"], float)[r0, :nn],
    )
    return {
        "paths": p,
        "dat": dat,
        "seasons": seasons,
        "entry": ent,
        "exit": tabs["dssatdrv_exit"],
        "physcl": tabs["physcl_exit"],
        "mg": tabs["mz_grosub_exit"],
        "organic": organic,
        "row0": r0,
        "layer_plt": read_layer_output(lp),
        "genotype48": data48 / "Genotype",
        "data_dir": data_dir,
        "expected": exp,
        "genotype_sets": genotype_sets(dat),
    }


@pytest.fixture(scope="module")
def runs(ref: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    # the cases are built in this run's own directory (the node-local AGRI_JAX_RUN_ROOT, else a
    # pytest temporary directory), never in the shared data directory: concurrent runs would race
    run_root = os.environ.get("AGRI_JAX_RUN_ROOT")
    base = Path(run_root) if run_root else tmp_path_factory.mktemp("catpa_dssat")
    root = base / CASE_DIR
    out: dict[str, Any] = {}
    for gs in GENOTYPE_SET_NAMES:
        case = build_case(
            root / gs,
            paths=ref["paths"],
            organic=ref["organic"],
            layer_plt=ref["layer_plt"],
            genotype48=ref["genotype48"],
            genotype_set=gs,
            start=START,
            end=END,
        )
        res = run_case(case, root / f"out_{gs}", run_root=run_root)
        out[gs] = {"case": case, "result": res, "rows": season_results(res.out_dir, case.seasons)}
    return out


# ---------------------------------------------------------------------------- 1. soil
def test_soil_conversion_matches_embedded_crop(ref: dict[str, Any]) -> None:
    dat, ex = ref["dat"], ref["exit"]
    v, r = ex.values, 0
    nl = int(np.asarray(v["SOILPROP%NLAYR"])[r])
    ds = np.asarray(v["SOILPROP%DS"], float)[r, :nl]
    np.testing.assert_array_equal(ds, dssat_layer_bottoms_cm(dat.profile_depth_cm))
    h, heads = dat.hydraulics, dat.water_suction_heads_cm
    want_heads = ref["expected"]["water_suction_heads_cm"]
    assert (heads["hfc"], heads["hwp"]) == (want_heads["hfc"], want_heads["hwp"])
    hz = dat.horizon_depths_cm
    diffs = {}
    for key, head in (("LL", heads["hwp"]), ("DUL", heads["hfc"])):
        th = brooks_corey_theta(head, h["hb"], h["lam"], h["theta_r"], h["theta_s"])
        ours = layer_average(hz, th, ds)
        refv = np.asarray(v[f"SOILPROP%{key}"], float)[r, :nl]
        diffs[key] = float(np.max(np.abs(ours - refv)))
        # rzwqm.dat's own FC33 / WP columns: the same curve printed with 4 decimals
        col = h["theta_wp"] if key == "LL" else h["theta_fc33"]
        diffs[f"dat_{key}"] = float(np.max(np.abs(th - col)))
    sat = np.asarray(v["SOILPROP%SAT"], float)[r, :nl]
    sat_ours = layer_average(hz, h["theta_s"], ds)
    # the embedded crop sees the tilled theta_s in the layers within the scenario's tillage depth
    till = read_management(dat, START, END)
    till_depth = float(till.loc[till["event"] == "tillage", "value"].max())
    n_tilled = int(np.sum(ds <= till_depth))
    print(
        "soil |ours - SOILPROP|: "
        + ", ".join(f"{k} {x:.2e}" for k, x in diffs.items())
        + f"; SAT tilled layers reference {sat[:n_tilled].tolist()} (tilled theta_s) vs ours "
        + f"{sat_ours[:n_tilled].tolist()}"
    )
    assert diffs["LL"] <= TOL_LAYER_REAL and diffs["DUL"] <= TOL_LAYER_REAL
    assert diffs["dat_LL"] <= TOL_SOL_FILE and diffs["dat_DUL"] <= TOL_SOL_FILE
    assert float(np.max(np.abs(sat_ours[n_tilled:] - sat[n_tilled:]))) <= TOL_LAYER_REAL
    # root growth factors and SLPF: MZDSSAT.RZX -> SOILPROP
    rzx = read_rzx(ref["paths"]["MZDSSAT.RZX"])
    np.testing.assert_array_equal(
        np.float32(rzx.srgf), np.float32(np.asarray(v["SOILPROP%WR"], float)[r, :nl])
    )
    assert np.float32(rzx.slpf) == np.float32(np.asarray(v["SOILPROP%SLPF"])[r])
    # organic C on the crop's layers: RZOC of the DSSATDRV entry of the first sowing day, mapped
    # (SOILPROP%OC is set inside that call: the exit row of the same day)
    en, r0 = ref["exit"].values, int(ref["exit"].index_of([int(ref["seasons"].yrplt[0])])[0])
    org = ref["organic"]
    oc = layer_average(org.node_bottoms_cm, org.oc_pct, ds)
    d_oc = float(np.max(np.abs(oc - np.asarray(en["SOILPROP%OC"], float)[r0, :nl])))
    print(f"SLOC |ours - SOILPROP%OC| {d_oc:.2e}")
    assert d_oc <= TOL_LAYER_REAL * float(np.max(oc))


def test_sol_file_reads_back(ref: dict[str, Any], runs: dict[str, Any]) -> None:
    case = runs["catpa"]["case"]
    back = read_sol(case.sol, dssat_spans=True)[CASE_SOIL_ID]
    lay = case.soil.layers
    for c in ("SLB", "SLLL", "SDUL", "SSAT", "SRGF", "SSKS", "SBDM", "SLOC", "SLNI", "SLHW"):
        np.testing.assert_array_equal(back.layers[c].to_numpy(float), lay[c].to_numpy(float), err_msg=c)
    print("CT.SOL layers:\n" + lay.to_string())


# ---------------------------------------------------------------------------- 2. weather
def _crop_day_weather(ref: dict[str, Any]) -> tuple[Any, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """``(DSSATDRV exit table, case weather of [START, END], case weather on the crop days,
    PHYSCL rows of the crop days)``."""
    ex = ref["exit"]
    days = _yrdoy_to_day(np.asarray(ex.date))
    w = catpa_weather(ref["paths"], ref["dat"], START, END).set_index("date")
    m = w.loc[pd.DatetimeIndex(days)]
    pe = ref["physcl"]
    j = pe.index_of([int(d) for d in np.asarray(ex.date)])
    assert np.all(j >= 0)
    return ex, w, m, j


def test_weather_equals_crop_inputs(ref: dict[str, Any], runs: dict[str, Any]) -> None:
    ex, w, m, j = _crop_day_weather(ref)
    pv = ref["physcl"].values
    diffs = {
        "RTH": float(np.max(np.abs(m["srad"].to_numpy() - np.asarray(pv["RTH"], float)[j]))),
        "TMAXR": float(np.max(np.abs(m["tmax"].to_numpy() - np.asarray(ex.values["TMAXR"], float)))),
        "TMINR": float(np.max(np.abs(m["tmin"].to_numpy() - np.asarray(ex.values["TMINR"], float)))),
        "RAINR": float(np.max(np.abs(m["rain"].to_numpy() / 10.0 - np.asarray(ex.values["RAINR"], float)))),
    }
    print("crop days: max |case - reference|: " + ", ".join(f"{k} {x:.2e}" for k, x in diffs.items()))
    # the case's SRAD is the reference's daily radiation, exactly (as in test_io_m3_catpa.py)
    assert diffs["RTH"] == 0.0
    # what the crop received, SRADR, is RTS: the re-sum of the hourly radiation, not RTH
    hrts = np.asarray(pv["HRTS"], np.float32)[j]
    rts = (hrts.sum(axis=1, dtype=np.float32) * np.float32(0.0036)).astype(float)
    np.testing.assert_array_equal(np.asarray(pv["RTS"], float)[j], np.asarray(ex.values["SRADR"], float))
    assert float(np.max(np.abs(rts - np.asarray(ex.values["SRADR"], float)))) <= TOL_RTS_RESUM
    assert diffs["TMAXR"] == 0.0 and diffs["TMINR"] == 0.0
    assert diffs["RAINR"] <= TOL_RAIN_CM
    # the written files, read the way dscsm048 reads them
    case = runs["catpa"]["case"]
    back = pd.concat([read_wth(f, dssat_spans=True) for f in case.weather], ignore_index=True).set_index(
        "date"
    )
    wb = w.loc[back.index]
    for c, dec in WTH_DECIMALS.items():
        assert float(np.max(np.abs(back[c].to_numpy(float) - wb[c].to_numpy()))) <= 0.5 * 10.0**-dec + 1e-9, c
    assert back.attrs["site"]["CCO2"] == float(ref["dat"].physiography["co2_ppm"])


@pytest.mark.xfail(
    strict=True,
    reason="measured 2026-10-13: the case writes the daily .MET radiation (PHYSCL RTH); the crop "
    "received RTS = 0.0036 sum(HRTS), the re-sum of RZWQM2's hourly radiation, which differs from "
    "RTH beyond the REAL re-sum bound on nearly every crop day of the local scenario (by up to about 0.1 "
    "MJ m-2 d-1); the hourly radiation is not reproduced by the case",
)
def test_case_srad_equals_crop_sradr(ref: dict[str, Any]) -> None:
    ex, _, m, _ = _crop_day_weather(ref)
    d = float(np.max(np.abs(m["srad"].to_numpy() - np.asarray(ex.values["SRADR"], float))))
    print(f"crop days: max |case SRAD - DSSATDRV SRADR| {d:.3e} MJ m-2 d-1")
    assert d <= TOL_SRAD


# ---------------------------------------------------------------------------- 3. genotype
def test_genotype_equals_our_ceres_parameters(ref: dict[str, Any], runs: dict[str, Any]) -> None:
    case = runs["catpa"]["case"]
    soil = ceres_soil_from_dssatdrv(ref["exit"].values)
    prm = ceres_maize_params(ref["paths"], soil, ref["seasons"], 0)
    cul = read_cul(case.case_dir / "MZCER048.CUL").loc[ref["genotype_sets"]["catpa"][0]]
    for k in ("P1", "P2", "P5", "G2", "G3", "PHINT"):
        assert float(cul[k]) == float(getattr(prm.cultivar, k.lower())), k
    eco = read_eco(case.case_dir / "MZCER048.ECO").loc[str(cul["ECO#"])]
    for k in ("TBASE", "TOPT", "ROPT", "DJTI", "GDDE", "DSGFT", "RUE", "TSEN", "CDAY"):
        assert float(eco[k]) == float(getattr(prm.cultivar, k.lower())), k
    assert float(eco["P20"]) == float(prm.cultivar.p2o)
    spe = read_spe(case.case_dir / "MZCER048.SPE")
    for k in CERES_SPECIES_KEYS:
        np.testing.assert_array_equal(
            np.asarray(spe[k], float), np.asarray(getattr(prm.species, k.lower())), k
        )
    assert float(np.asarray(spe["PORM"])) == float(prm.species.pormin)
    assert float(np.asarray(spe["RWMX"])) == float(prm.species.rwumx)
    diffs = genotype_differences(ref["paths"], ref["genotype48"])
    print("genotype 4.0 (RZWQM2) vs 4.8.6 (MZCER048), value used by the catpa set:")
    for d in diffs:
        print(f"  {d.file} {d.name}: 4.0 {d.rzwqm_40} | 4.8.6 {d.dssat_486} | used {d.used}")
    names = {(d.file, d.name) for d in diffs}
    assert {
        ("CUL", "P1"),
        ("CUL", "P5"),
        ("CUL", "G2"),
        ("CUL", "G3"),
        ("ECO", "TSEN"),
        ("SPE", "RGFIL"),
    } <= names


# ---------------------------------------------------------------------------- 4. dscsm048 read what was written
def test_dssat_runs_and_echoes_inputs(ref: dict[str, Any], runs: dict[str, Any]) -> None:
    seasons = ref["seasons"]
    n_season = seasons.n_season
    years = [pd.Timestamp(d).year for d in seasons.sow]
    num = treatment_numbers(n_season)
    for gs, r in runs.items():
        rows = r["rows"]
        assert len(rows) == 2 * n_season
        for row in rows:
            k = years.index(row["season"])
            assert (row["pdat"], row["hdat"]) == (_yrdoy(seasons.sow[k]), _yrdoy(seasons.harvest[k])), row
        fx = read_filex(r["case"].filex)
        assert len(fx["TREATMENTS"]) == 2 * n_season
        soil: Any = read_overview_soil(Path(r["result"].out_dir) / "OVERVIEW.OUT")
        lay = r["case"].soil.layers
        ic_file = {lev: pd.DataFrame(d["rows"]) for lev, d in fx["INITIAL CONDITIONS"].items()}
        for (k, _nit), tr in num.items():
            s = soil[soil["RUN"] == tr].reset_index(drop=True)
            assert len(s) == len(lay), (gs, tr)
            np.testing.assert_array_equal(s["BOTTOM"].to_numpy(), lay["SLB"].to_numpy())
            icf = ic_file[k + 1]
            checks = [
                ("LL", lay["SLLL"], TOL_ECHO_3),
                ("DUL", lay["SDUL"], TOL_ECHO_3),
                ("SAT", lay["SSAT"], TOL_ECHO_3),
                ("INIT_SW", icf["SH2O"], TOL_ECHO_3),
                ("ORG_C", lay["SLOC"], TOL_ECHO_2),
            ]
            if _nit == "Y":
                checks.append(("NO3", icf["SNO3"], TOL_ECHO_2))
            for col, src, tol in checks:
                d = float(np.max(np.abs(s[col].to_numpy() - np.asarray(src, float))))
                assert d <= tol + 1e-9, (gs, tr, col, d)
            # the FileX holds the initial state with 4 decimals (SW) / to 5 columns (NO3)
            assert (
                float(np.max(np.abs(icf["SH2O"].to_numpy() - r["case"].ics[k]["SH2O"].to_numpy())))
                <= TOL_SOL_FILE
            )
        cul: Any = read_overview_cultivar(Path(r["result"].out_dir) / "OVERVIEW.OUT")
        want = read_cul((r["case"].case_dir if gs == "catpa" else ref["genotype48"]) / "MZCER048.CUL").loc[
            ref["genotype_sets"][gs][0]
        ]
        for k in ("P1", "P2", "P5", "G2", "G3", "PHINT"):
            assert np.allclose(cul[k].to_numpy(float), float(want[k]), rtol=0, atol=0.5e-2), (gs, k)
        print(f"dscsm048 {gs}: {len(rows)} treatments, {r['result'].elapsed_s:.1f} s")
    for k, ic in enumerate(runs["catpa"]["case"].ics):
        print(f"initial state {years[k]} (end of the day before sowing): SH2O {ic['SH2O'].round(4).tolist()}")
        print(
            f"   SNO3 {ic['SNO3'].round(3).tolist()} ug/g, LL {runs['catpa']['case'].soil.layers['SLLL'].tolist()}"
        )


# ---------------------------------------------------------------------------- 5. the comparison
def _rzwqm_season(mg: Any, sow: int, harvest: int) -> dict[str, Any]:
    d = np.asarray(mg.date, dtype=np.int64)
    sel = (d >= sow) & (d <= harvest)
    stg = np.asarray(mg.values["STGDOY"])[sel][-1]
    return {
        "stgdoy": {str(i + 1): int(stg[i]) for i in range(9) if 0 < int(stg[i]) < 9999999},
        "lai_max": float(np.max(np.asarray(mg.values["LAI"], float)[sel])),
        "biomass_harvest_g_m2": float(np.asarray(mg.values["BIOMAS"], float)[sel][-1]),
        "yield_kg_ha": float(np.asarray(mg.values["YIELD"], float)[sel][-1]),
        "mean_1_swfac": float(np.mean(1.0 - np.asarray(mg.values["SWFAC"], float)[sel])),
        "mean_1_turfac": float(np.mean(1.0 - np.asarray(mg.values["TURFAC"], float)[sel])),
        "mean_1_nstres": float(np.mean(1.0 - np.asarray(mg.values["NSTRES"], float)[sel])),
    }


def _days(a: int, b: int) -> int:
    return int((_yrdoy_to_day(a) - _yrdoy_to_day(b)).astype(int))


def _rel(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None or b == 0 else a / b - 1.0


def test_season_comparison_is_recorded(ref: dict[str, Any], runs: dict[str, Any]) -> None:
    """Measured, not asserted: DSSAT-CSM 4.8.6 standalone on CA-TPA (four runs) against the RZWQM2
    embedded DSSAT 4.0.2 CERES and our CERES driven by the RZWQM2 crop water."""
    seasons = ref["seasons"]
    w4p = Path(os.environ.get("AGRI_JAX_DATA", str(ref["data_dir"]))) / W4_REPORT
    w4 = {r["season"]: r for r in json.loads(w4p.read_text())["seasons"]} if w4p.is_file() else {}
    out = []
    for k in range(seasons.n_season):
        sow, har = _yrdoy(seasons.sow[k]), _yrdoy(seasons.harvest[k])
        rz = _rzwqm_season(ref["mg"], sow, har)
        year = sow // 1000
        row: dict[str, Any] = {"season": year, "sow": sow, "harvest": har, "rzwqm2_embedded_4_0_2": rz}
        if year in w4:
            o = w4[year]
            row["ours_ceres_4_8_6_rzwqm_water"] = {
                "lai_max": o["lai_max"][0],
                "biomass_harvest_g_m2": o["biomass_harvest_g_m2"][0],
                "yield_kg_ha": o["yield_harvest_kg_ha"][0],
                "stage_date_diff_days_vs_rzwqm2": o["stage_date_diff_days"],
            }
        ds: dict[str, Any] = {}
        for gs, r in runs.items():
            for nit, _label in N_OPTIONS:
                x = next(y for y in r["rows"] if y["season"] == year and y["nitro"] == nit)
                x = dict(x)
                x["stage_date_diff_days_vs_rzwqm2"] = {
                    s: _days(x["stgdoy"][s], rz["stgdoy"][s]) for s in x["stgdoy"] if s in rz["stgdoy"]
                }
                ds[f"{gs}_N{'on' if nit == 'Y' else 'off'}"] = x
        row["dssat_4_8_6"] = ds
        ours = row.get("ours_ceres_4_8_6_rzwqm_water", {}).get("yield_kg_ha")
        row["yield_ratios"] = {
            "version: ours / rzwqm2 - 1": _rel(ours, rz["yield_kg_ha"]),
            **{f"coupling: dssat {key} / ours - 1": _rel(v["yield_kg_ha"], ours) for key, v in ds.items()},
            **{
                f"total: dssat {key} / rzwqm2 - 1": _rel(v["yield_kg_ha"], rz["yield_kg_ha"])
                for key, v in ds.items()
            },
            "N effect: catpa N on / N off - 1": _rel(
                ds["catpa_Non"]["yield_kg_ha"], ds["catpa_Noff"]["yield_kg_ha"]
            ),
            "genotype: stock / catpa (N off) - 1": _rel(
                ds["stock_Noff"]["yield_kg_ha"], ds["catpa_Noff"]["yield_kg_ha"]
            ),
        }
        out.append(row)
    rep = {
        "what": f"CA-TPA {_yrdoy(seasons.sow[0]) // 1000}-{_yrdoy(seasons.sow[-1]) // 1000}: DSSAT-CSM 4.8.6 standalone (dscsm048 build486) vs RZWQM2 4.6 embedded DSSAT "
        "4.0.2 CERES vs our CERES (DSSAT-CSM 4.8.6) driven by the RZWQM2 crop water and NSTRES",
        "genotype_sets": {k: list(v) for k, v in ref["genotype_sets"].items()},
        "dssat_only_settings": [vars(s) for s in DSSAT_ONLY_SETTINGS],
        "genotype_differences": [vars(d) for d in genotype_differences(ref["paths"], ref["genotype48"])],
        "seasons": out,
    }
    path = Path(os.environ.get("AGRI_JAX_DATA", str(ref["data_dir"]))) / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    hdr = "season  yield: rzwqm2   ours | catpa Noff  Non | stock Noff  Non   (kg/ha)"
    print(hdr)
    for row in out:
        ds = row["dssat_4_8_6"]
        o = row.get("ours_ceres_4_8_6_rzwqm_water", {}).get("yield_kg_ha", float("nan"))
        print(
            f"{row['season']}  {row['rzwqm2_embedded_4_0_2']['yield_kg_ha']:8.0f} {o:7.0f} | "
            f"{ds['catpa_Noff']['yield_kg_ha']:6.0f} {ds['catpa_Non']['yield_kg_ha']:6.0f} | "
            f"{ds['stock_Noff']['yield_kg_ha']:6.0f} {ds['stock_Non']['yield_kg_ha']:6.0f}"
        )
    print("W6B " + json.dumps(out, default=str))
    assert len(out) == seasons.n_season
