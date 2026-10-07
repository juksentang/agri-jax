"""Unit tier (no data) for the CA-TPA DSSAT case converter (:mod:`agrijax.sites.catpa_dssat`) and the
``.RZX`` reader (:mod:`agrijax.io.rzwqm.rzx`): the Brooks-Corey conversion, the layer mapping, and
write -> read round trips of the generated FileX, SOL and WTH with the repository's DSSAT readers
(which split fields the way dscsm048 does). The CA-TPA checks against the RZWQM2 reference and the
dscsm048 runs are in ``tests/integration/test_io_catpa_dssat.py``."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from agrijax.io.dssat.filex import read_filex
from agrijax.io.dssat.sol import read_sol, write_sol
from agrijax.io.dssat.wth import read_wth
from agrijax.io.rzwqm.rzx import read_rzx
from agrijax.sites.catpa_dssat import (
    DSSAT_LAYER_BOTTOMS_CM,
    N_OPTIONS,
    brooks_corey_theta,
    catpa_soil_profile,
    initial_conditions,
    layer_average,
    read_overview_cultivar,
    read_overview_soil,
    read_overview_stages,
    treatment_numbers,
    weather_site,
    write_filex,
    write_weather_files,
)
from agrijax.sites.catpa_m3 import SeasonTable


# ---------------------------------------------------------------------------- Brooks-Corey
def test_brooks_corey_theta_closed_forms() -> None:
    hb, lam, tr, ts = 15.0, 0.25, 0.05, 0.45
    assert brooks_corey_theta(hb, hb, lam, tr, ts) == pytest.approx(ts, abs=1e-15)
    # Se = 1/2 at h = hb 2^(1/lam)
    h_half = hb * 2.0 ** (1.0 / lam)
    assert brooks_corey_theta(-h_half, hb, lam, tr, ts) == pytest.approx(tr + 0.5 * (ts - tr), abs=1e-14)
    h = np.array([333.0, 1000.0, 15000.0])
    th = brooks_corey_theta(h, hb, lam, tr, ts)
    assert np.all(np.diff(th) < 0) and np.all(th > tr)
    with pytest.raises(ValueError, match="bubbling"):
        brooks_corey_theta(10.0, hb, lam, tr, ts)


# ---------------------------------------------------------------------------- layer mapping
def test_layer_average_identity_mix_and_conservation() -> None:
    zb = np.array([15.0, 30.0, 70.0, 90.0, 150.0])
    v = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(layer_average(zb, v, zb), v)
    out = layer_average(zb, v, DSSAT_LAYER_BOTTOMS_CM)
    # 60-90 cm: 10 cm of horizon 3 and 20 cm of horizon 4
    assert out[5] == pytest.approx((10 * 0.3 + 20 * 0.4) / 30, abs=1e-15)
    dz_src = np.diff(np.r_[0.0, zb])
    dz_dst = np.diff(np.r_[0.0, DSSAT_LAYER_BOTTOMS_CM])
    assert float(out @ dz_dst) == pytest.approx(float(v @ dz_src), abs=1e-12)
    w = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
    assert layer_average(zb, v, [30.0], weights=w)[0] == pytest.approx((15 * 0.1 + 30 * 0.2) / 45, abs=1e-15)
    with pytest.raises(ValueError, match="deeper"):
        layer_average(zb, v, [160.0])


def test_dssat_layers_are_dssatdrv_ds() -> None:
    np.testing.assert_array_equal(DSSAT_LAYER_BOTTOMS_CM, [5, 15, 30, 45, 60, 90, 120, 150])


# ---------------------------------------------------------------------------- .RZX
_RZX = """====
= FACTORS
====
1  0.9  1  1
====
= SWITCHES
====
Y  N  Y  N  N  N  N  C
====
= OUTPUT
====
N  Y  N  10  Y  Y  Y  Y  N  N  N  1
====
= ROOTS
====
3  200  3
1
0.5
0.25
====
= DATABASE
====
C:\\X\\
1
MZCER040
"""


def test_read_rzx(tmp_path: Path) -> None:
    p = tmp_path / "MZDSSAT.RZX"
    p.write_text(_RZX)
    r = read_rzx(p)
    assert (r.slnf, r.slpf, r.efinoc, r.efnfix) == (1.0, 0.9, 1.0, 1.0)
    assert r.switches["ISWWAT"] == "Y" and r.switches["ISWNIT"] == "N" and r.switches["ISWPSN"] == "C"
    assert (r.n_root_layers, r.max_root_depth_cm, r.root_exponent) == (3, 200.0, 3.0)
    np.testing.assert_array_equal(r.srgf, [1.0, 0.5, 0.25])


# ---------------------------------------------------------------------------- SOL
def _fake_dat() -> Any:
    nh = 5
    return SimpleNamespace(
        horizon_depths_cm=np.array([15.0, 30.0, 70.0, 90.0, 150.0]),
        hydraulics={
            "hb": np.full(nh, 15.0),
            "lam": np.array([0.25, 0.30, 0.35, 0.20, 0.30]),
            "theta_r": np.array([0.05, 0.03, 0.04, 0.05, 0.04]),
            "theta_s": np.full(nh, 0.45),
            "ksat": np.array([5.0, 3.0, 3.5, 3.0, 2.5]),
        },
        water_suction_heads_cm={"hmin": -15000.0, "hfc": -333.0, "hwp": -15000.0},
        soil_physical={
            "bulk_density": np.full(nh, 1.45),
            "clay": np.full(nh, 0.1),
            "silt": np.full(nh, 0.25),
        },
        physiography={
            "latitude_rad": 0.745163,
            "longitude_rad": -1.40235,
            "elevation_m": 200.0,
            "co2_ppm": 330.0,
        },
        pet={"wind_height_m": 2.0},
    )


def _rzx8() -> Any:
    return SimpleNamespace(slnf=1.0, slpf=1.0, srgf=np.array([1, 1, 0.7, 0.54, 0.4, 0.24, 0.11, 0.03]))


def _organic() -> Any:
    from agrijax.sites.catpa_dssat import OrganicProfile

    nb = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 15.0, 30.0, 70.0, 90.0, 150.0])
    return OrganicProfile(nb, np.linspace(1.8, 0.2, 10), np.linspace(0.16, 0.02, 10), np.full(10, 6.91))


def test_soil_profile_values_and_sol_round_trip(tmp_path: Path) -> None:
    dat = _fake_dat()
    prof = catpa_soil_profile(cast(Any, dat), cast(Any, _rzx8()), _organic())
    lay = prof.layers
    h = dat.hydraulics
    ll_h = brooks_corey_theta(15000.0, h["hb"], h["lam"], h["theta_r"], h["theta_s"])
    dul_h = brooks_corey_theta(333.0, h["hb"], h["lam"], h["theta_r"], h["theta_s"])
    # layers 1-2 lie in horizon 1; layer 6 (60-90 cm) mixes horizons 3 and 4
    assert lay["SLLL"].iloc[0] == pytest.approx(round(float(ll_h[0]), 4), abs=0)
    assert lay["SDUL"].iloc[5] == pytest.approx(round((10 * dul_h[2] + 20 * dul_h[3]) / 30, 4), abs=0)
    assert np.all(lay["SSAT"] == 0.45) and np.all(lay["SLCL"] == 10.0) and np.all(lay["SLSI"] == 25.0)
    assert np.all(lay["SDUL"] > lay["SLLL"])
    p = write_sol([prof], tmp_path / "CT.SOL")
    back = read_sol(p, dssat_spans=True)[prof.id]
    for c in ("SLB", "SLLL", "SDUL", "SSAT", "SRGF", "SSKS", "SBDM", "SLOC", "SLNI", "SLHW", "SLCL", "SLSI"):
        np.testing.assert_array_equal(back.layers[c].to_numpy(float), lay[c].to_numpy(float), err_msg=c)
    assert back.surface["SLDR"] == 0.53 and back.surface["SLPF"] == 1.0


def test_soil_profile_needs_one_root_factor_per_layer() -> None:
    rzx = SimpleNamespace(slnf=1.0, slpf=1.0, srgf=np.ones(3))
    with pytest.raises(ValueError, match="root growth factors"):
        catpa_soil_profile(cast(Any, _fake_dat()), cast(Any, rzx), _organic())


# ---------------------------------------------------------------------------- WTH
def _weather() -> pd.DataFrame:
    d = pd.date_range("2015-01-01", "2016-12-31", freq="D")
    doy = np.asarray(cast(Any, d).dayofyear)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "date": d,
            "srad": 15 + 10 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.random(len(d)) * 1e-3,
            "tmax": 12 + 15 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.random(len(d)),
            "tmin": 2 + 12 * np.sin(2 * np.pi * (doy - 110) / 365) - rng.random(len(d)),
            "rain": np.where(rng.random(len(d)) < 0.3, rng.random(len(d)) * 25.4, 0.0),
            "rhum": 60 + 30 * rng.random(len(d)),
            "wind": 100 + 400 * rng.random(len(d)),
        }
    )


def test_weather_files_round_trip(tmp_path: Path) -> None:
    w = _weather()
    site = weather_site(cast(Any, _fake_dat()), w)
    assert site["INSI"] == "CTPA" and site["CCO2"] == 330.0 and site["REFHT"] == 2.0
    tav: Any = 0.5 * (w["tmax"] + w["tmin"])
    months = tav.groupby(w["date"].dt.month).mean()
    assert site["TAV"] == round(float(tav.mean()), 1)
    assert site["AMP"] == round(float(months.max() - months.min()), 1)
    files = write_weather_files(w, site, tmp_path)
    assert [f.name for f in files] == ["CTPA1501.WTH", "CTPA1601.WTH"]
    back = pd.concat([read_wth(f, dssat_spans=True) for f in files], ignore_index=True)
    assert len(back) == len(w)
    np.testing.assert_array_equal(pd.to_datetime(back["date"]).to_numpy(), w["date"].to_numpy())
    for c, dec in (("srad", 3), ("tmax", 3), ("tmin", 3), ("rain", 3), ("rhum", 2), ("wind", 1)):
        np.testing.assert_allclose(
            back[c].to_numpy(float), w[c].round(dec).to_numpy(), rtol=0, atol=1e-9, err_msg=c
        )
    assert back.attrs["site"]["CCO2"] == 330.0


# ---------------------------------------------------------------------------- initial conditions
def test_initial_conditions_from_layer_plt() -> None:
    depth = np.array([1.0, 5.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0])
    t = pd.date_range("2015-04-26", periods=3, freq="D")
    theta = np.tile(np.linspace(0.3, 0.2, len(depth)), (3, 1))
    no3 = np.tile(np.linspace(9.0, 1.0, len(depth)), (3, 1))
    bd = np.tile(np.r_[1.3, np.full(len(depth) - 1, 1.45)], (3, 1))
    ds = xr.Dataset(
        {
            "soil_water_content": (("time", "depth"), theta),
            "no3_n_conc": (("time", "depth"), no3),
            "soil_bulk_density": (("time", "depth"), bd),
        },
        coords={"time": t, "depth": depth},
    )
    ic = initial_conditions(ds, "2015-04-27")
    np.testing.assert_array_equal(ic["ICBL"], DSSAT_LAYER_BOTTOMS_CM)
    # layer 1 (0-5 cm): nodes of 1 cm and 4 cm; NO3 weighted by thickness x bulk density
    assert ic["SH2O"].iloc[0] == pytest.approx((theta[1, 0] + 4 * theta[1, 1]) / 5, abs=1e-15)
    assert ic["SNO3"].iloc[0] == pytest.approx(
        (1.3 * no3[1, 0] + 4 * 1.45 * no3[1, 1]) / (1.3 + 4 * 1.45), abs=1e-14
    )
    np.testing.assert_array_equal(ic["SH2O"].iloc[1:], theta[1, 2:])
    assert np.all(ic["SNH4"] == 0.0)


# ---------------------------------------------------------------------------- FileX
def _seasons() -> SeasonTable:
    sow = np.array(["2015-04-28", "2016-05-05"], dtype="datetime64[D]")
    harvest = np.array(["2015-09-25", "2016-10-20"], dtype="datetime64[D]")
    return SeasonTable(
        sow=sow,
        harvest=harvest,
        yrplt=np.array([2015118, 2016126]),
        harvest_yrdoy=np.array([2015268, 2016294]),
        pltpop=np.array([8.0, 8.0]),
        sdepth=np.array([3.0, 3.0]),
        rowspc=np.array([76.0, 76.0]),
    )


def _ics() -> list[pd.DataFrame]:
    zb = DSSAT_LAYER_BOTTOMS_CM
    return [
        pd.DataFrame(
            {
                "ICBL": zb,
                "SH2O": np.linspace(0.276741, 0.19, 8) + k * 0.01,
                "SNH4": 0.0,
                "SNO3": np.linspace(12.3456, 0.3, 8),
            }
        )
        for k in range(2)
    ]


def test_filex_round_trip(tmp_path: Path) -> None:
    s = _seasons()
    ferts = [[(s.sow[0], "FE008", "AP001", 180.0)], [(s.sow[1], "FE008", "AP001", 180.0)]]
    p = write_filex(tmp_path / "CTPA1501.MZX", s, _ics(), ferts, cultivar=("CT0012", "PIO 3382 RZWQM2"))
    fx = read_filex(p)
    trt = fx["TREATMENTS"]
    num = treatment_numbers(2)
    assert [r["N"] for r in trt] == [1, 2, 3, 4]
    assert {(k, nit): num[(k, nit)] for k in range(2) for nit, _ in N_OPTIONS} == {
        (0, "N"): 1,
        (1, "N"): 2,
        (0, "Y"): 3,
        (1, "Y"): 4,
    }
    for r in trt:
        k = (r["N"] - 1) % 2
        n_on = r["N"] > 2
        assert (r["IC"], r["MP"], r["MH"], r["SM"]) == (k + 1, k + 1, k + 1, r["N"])
        assert r["MF"] == (k + 1 if n_on else 0)
        assert r["TNAME"].startswith(str(2015 + k))
    assert fx["CULTIVARS"][1]["INGENO"] == "CT0012" and fx["CULTIVARS"][1]["CNAME"] == "PIO 3382 RZWQM2"
    f = fx["FIELDS"][1]
    assert (f["WSTA"], f["ID_SOIL"], f["SLDP"]) == ("CTPA", "CTPA150001", 150)
    for k in range(2):
        pl = fx["PLANTING DETAILS"][k + 1]
        assert (pl["PDATE"], pl["PPOP"], pl["PLRS"], pl["PLDP"], pl["PLME"]) == (
            [15118, 16126][k],
            8,
            76,
            3,
            "S",
        )
        assert fx["HARVEST DETAILS"][k + 1]["rows"][0]["HDATE"] == [15268, 16294][k]
        fe = fx["FERTILIZERS (INORGANIC)"][k + 1]["rows"]
        assert [(r["FDATE"], r["FMCD"], r["FACD"], r["FAMN"]) for r in fe] == [
            ([15118, 16126][k], "FE008", "AP001", 180)
        ]
        ic = fx["INITIAL CONDITIONS"][k + 1]
        assert ic["ICDAT"] == [15118, 16126][k]
        rows = ic["rows"]
        np.testing.assert_allclose([r["SH2O"] for r in rows], _ics()[k]["SH2O"].round(4), rtol=0, atol=1e-12)
        # NO3 keeps as many decimals as its 5 columns hold
        np.testing.assert_allclose([r["SNO3"] for r in rows], _ics()[k]["SNO3"], rtol=0, atol=0.5e-2)
        np.testing.assert_allclose(
            [r["SNO3"] for r in rows][2:], _ics()[k]["SNO3"].round(3)[2:], rtol=0, atol=1e-12
        )
    sim = fx["SIMULATION CONTROLS"]
    for tr in range(1, 5):
        assert sim[tr]["OPTIONS"]["WATER"] == "Y"
        assert sim[tr]["OPTIONS"]["NITRO"] == ("Y" if tr > 2 else "N")
        assert sim[tr]["OPTIONS"]["CO2"] == "W"
        assert sim[tr]["GENERAL"]["SDATE"] == [15118, 16126][(tr - 1) % 2]
        assert sim[tr]["MANAGEMENT"]["HARVS"] == "R" and sim[tr]["METHODS"]["EVAPO"] == "R"


def test_filex_values_end_at_their_header_token(tmp_path: Path) -> None:
    """Numbers and codes are right-aligned to their header token (DSSAT's fixed ``(1X,F5.0)``-style
    fields), so each data token ends where its header token ends."""
    s = _seasons()
    p = write_filex(tmp_path / "X.MZX", s, _ics(), [[], []])
    lines = p.read_text().splitlines()
    # header prefix -> indices of text tokens (left-aligned, skipped here)
    skip = {"@C  ICBL": (), "@P PDATE": (15,), "@N OPTIONS": (1,), "@N METHODS": (1,), "@H HDATE": (7,)}
    checked = 0
    for i, ln in enumerate(lines[:-1]):
        key = next((k for k in skip if ln.startswith(k)), None)
        if key is None:
            continue
        heads = [m.end() for m in re.finditer(r"\S+", ln)]
        vals = [m.end() for m in re.finditer(r"\S+", lines[i + 1])]
        assert len(vals) == len(heads), ln
        for j, (h, v) in enumerate(zip(heads, vals, strict=True)):
            if j not in skip[key]:
                assert h == v, (ln, j)
        checked += 1
    assert checked > 5
    assert "*FERTILIZERS" not in p.read_text()


# ---------------------------------------------------------------------------- OVERVIEW.OUT parsers
_OVERVIEW = """*SIMULATION OVERVIEW FILE

*RUN   1        : 2015 water on, N off      MZCER048 CTPA1501    1
 MODEL          : MZCER048 - Maize
 TREATMENT  1   : 2015 water on, N off      MZCER048
 STARTING DATE  : APR 28 2015

*SUMMARY OF SOIL AND GENETIC INPUT PARAMETERS

   SOIL LOWER UPPER   SAT  EXTR  INIT   ROOT   BULK     pH    NO3    NH4    ORG
-------------------------------------------------------------------------------
  0-  5 0.150 0.260 0.450 0.110 0.277   1.00   1.45   6.91   0.18   0.00   1.77
  5- 15 0.150 0.260 0.450 0.110 0.278   1.00   1.45   6.91   0.18   0.00   1.77

TOT-150   1.0   2.0   3.0   4.0   5.0  <--cm   -  kg/ha-->    0.0    0.0      0

 Maize            CULTIVAR: CT0012-PIO 3382 RZWQM2    ECOTYPE: IB0001
 P1     : 250.00  P2     : 0.7000  P5     : 890.00
 G2     : 825.00  G3     :  8.500  PHINT  : 38.900

*SIMULATED CROP AND SOIL STATUS AT MAIN DEVELOPMENT STAGES

   DATE  AGE STAGE        kg/ha    LAI   NUM  kg/ha  %   H2O  Nitr Phos1 Phos2  RSTG
 ------  --- ----------   -----  -----  ----  ---  ---  ----  ----  ----  ----  ----
 28 APR    0 Start Sim        0   0.00   0.0    0  0.0  0.00  0.00  0.00  0.00     7
 28 APR    0 Sowing           0   0.00   0.0    0  0.0  0.00  0.00  0.00  0.00     8
 29 APR    1 Germinate        0   0.00   0.0    0  0.0  0.00  0.00  0.00  0.00     9
  6 MAY    8 Emergence       29   0.00   1.7    0  0.0  0.00  0.00  0.00  0.00     1
 20 SEP  145 Maturity     22000   2.00  20.0    0  0.0  0.00  0.00  0.00  0.00    10

*MAIN GROWTH AND DEVELOPMENT VARIABLES
"""


def test_overview_parsers(tmp_path: Path) -> None:
    p = tmp_path / "OVERVIEW.OUT"
    p.write_text(
        _OVERVIEW + _OVERVIEW.replace("*RUN   1 ", "*RUN   2 ").replace("TREATMENT  1", "TREATMENT  8")
    )
    st: Any = read_overview_stages(p)
    one = st[st["RUN"] == 1]
    assert list(one["STAGE"]) == ["Start Sim", "Sowing", "Germinate", "Emergence", "Maturity"]
    assert list(one["STGDOY_INDEX"]) == [-1, 7, 8, 9, 6]
    assert one["DATE"].iloc[-1] == pd.Timestamp("2015-09-20").date()
    assert list(st[st["RUN"] == 2]["TRNO"].unique()) == [8]
    soil = read_overview_soil(p)
    assert len(soil) == 4 and soil["LL"].iloc[0] == 0.15 and soil["INIT_SW"].iloc[1] == 0.278
    assert soil["ORG_C"].iloc[0] == 1.77 and soil["BOTTOM"].iloc[1] == 15.0
    cul = read_overview_cultivar(p)
    assert cul.loc[0, ["P1", "P2", "P5", "G2", "G3", "PHINT"]].tolist() == [
        250.0,
        0.7,
        890.0,
        825.0,
        8.5,
        38.9,
    ]
