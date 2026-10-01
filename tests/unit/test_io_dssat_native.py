"""The native DSSAT input derivations (no data): Fortran formatted I/O, LYRSET2 / LMATCH, the soil and
initial-water chain, DAYLEN / HMET, the 2 m wind, CO2VAL, the reported irrigation, the residue
parameters and KEP, and the scope check."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agrijax.forcing.dssat_weather import DSSAT486_WEATHER, daylength486, hourly_mean_temperature, wind_at_2m
from agrijax.io.dssat._f77 import decimal_of, nint, read_f, round_trip, write_f
from agrijax.io.dssat.native_management import (
    extinction_kep,
    irrigation_amounts,
    residue_parameters,
    switches,
    unsupported_features,
)
from agrijax.io.dssat.native_soil import (
    find_soil_profile,
    initial_soil_water,
    lmatch,
    lyrset2,
    lyrset3,
    native_soil,
    swcn_width,
)
from agrijax.io.dssat.native_weather import CO2Series, co2_daily, read_co2_series
from agrijax.io.dssat.sol import SoilProfile

F = np.float32


# ------------------------------------------------------------------------ Fortran I/O
@pytest.mark.parametrize(
    ("x", "w", "d", "text"),
    [
        (12.25, 5, 1, " 12.2"),  # an exact tie rounds to even, as gfortran / printf
        (12.35, 5, 1, " 12.4"),
        (0.0265, 5, 3, "0.026"),  # the REAL*4 of 0.0265 is below the tie
        (0.125, 5, 2, " 0.12"),
        (2.5, 5, 0, "   2."),
        (3.5, 5, 0, "   4."),
        (-99.0, 5, 0, " -99."),
        (-99.0, 5, 3, "*****"),
        (5.0, 5, 0, "   5."),
    ],
)
def test_write_f_follows_gfortran(x: float, w: int, d: int, text: str) -> None:
    assert write_f(x, w, d) == text


def test_read_f_and_round_trip() -> None:
    assert read_f(" 0.110") == F(0.11)
    assert read_f("   12", 1) == F(1.2)  # digits alone are scaled by 10**-d
    assert read_f("     ") == F(0.0)
    with pytest.raises(ValueError, match="asterisks"):
        read_f("*****")
    assert round_trip(F(0.0865), 5, 3) == F(0.086)
    assert nint(2.5) == 3 and nint(-2.5) == -3 and nint(1.49) == 1
    np.testing.assert_array_equal(decimal_of(np.float32(0.1)), 0.1)


# ------------------------------------------------------------------------ layers
def test_lyrset2_keeps_5_and_15_cm_and_splits_thick_layers() -> None:
    # 15-30 one layer, 30-60 two (above 90 cm: at most 15 cm), 60-90 and 90-120 one (from 90 cm: 30 cm),
    # 120-180 two
    ds = lyrset2([5, 15, 30, 60, 90, 120, 180])
    np.testing.assert_array_equal(ds, [5, 15, 30, 45, 60, 90, 120, 150, 180])
    # a layer thinner than 2 cm joins the next; the profile bottom is kept
    np.testing.assert_array_equal(lyrset2([10, 16, 31, 32]), [5, 15, 23, 32])
    np.testing.assert_array_equal(lyrset2([20, 35]), [5, 15, 20, 35])
    # three and four equal parts (NINT of the fractions)
    np.testing.assert_array_equal(lyrset2([15, 60]), [5, 15, 30, 45, 60])
    np.testing.assert_array_equal(lyrset2([15, 80]), [5, 15, 31, 47, 64, 80])
    np.testing.assert_array_equal(lyrset3([7, 20]), [7, 20])


def test_lmatch_depth_weighted_mean_and_missing() -> None:
    v = lmatch([10, 30], [0.1, 0.3], [5, 15, 30])
    np.testing.assert_allclose(v, [0.1, (0.1 * 5 + 0.3 * 5) / 10, 0.3], rtol=1e-6)
    assert v.dtype == np.float32
    # a missing input layer makes every output layer that overlaps it missing
    v = lmatch([10, 30], [-99.0, 0.3], [5, 15, 30])
    np.testing.assert_array_equal(v, F([-99.0, -99.0, 0.3]))
    # input layers shallower than the output profile: the part below takes the last layer, deeper
    # output layers stay missing
    v = lmatch([10], [0.2], [5, 15, 30])
    np.testing.assert_allclose(v, [0.2, 0.2, -99.0])


def _profile(**cols: list[float]) -> SoilProfile:
    base = {
        "SLB": [5.0, 15.0, 30.0, 60.0],
        "SLLL": [0.1] * 4,
        "SDUL": [0.2] * 4,
        "SSAT": [0.3] * 4,
        "SSKS": [1.2] * 4,
    }
    base.update(cols)
    return SoilProfile(
        id="XXTEST0001",
        surface={"SCOM": "BN", "SALB": 0.13, "SLU1": 6.0, "SLDR": 0.6, "SLRO": 73.0},
        layers=pd.DataFrame(base),
    )


def test_native_soil_input_chain() -> None:
    p = _profile(
        SLLL=[0.2, 0.1, 0.1, 0.1],  # LL = DUL -> DUL - 0.01
        SSAT=[0.3, 0.2, 0.3, 0.3],  # SAT = DUL -> DUL + 0.01
        SSKS=[math.nan, 0.05, 1.2, 150.0],  # missing; printed F5.4, F5.2, F5.0
    )
    s = native_soil(p)
    np.testing.assert_array_equal(s.ds, [5, 15, 30, 45, 60])
    np.testing.assert_array_equal(s.dlayr, [5, 10, 15, 15, 15])
    assert s.ll[0] == 0.19 and s.sat[1] == 0.21
    np.testing.assert_array_equal(s.swcn, [-99.0, 0.05, 1.2, 150.0, 150.0])
    assert (s.salb, s.u, s.swcon, s.cn) == (0.13, 6.0, 0.6, 73.0)
    assert swcn_width(0.05) == (5, 4) and swcn_width(-99.0) == (5, 0) and swcn_width(5.0) == (5, 2)
    # the printed decimals: the 5-15 cm layer averages 0.2 and 0.2115 (0.20575), printed F5.3
    q = native_soil(
        _profile(
            SLB=[10.0, 20.0, 40.0], SLLL=[0.1] * 3, SDUL=[0.2, 0.2115, 0.24], SSAT=[0.3] * 3, SSKS=[1.2] * 3
        )
    )
    np.testing.assert_array_equal(q.ds, [5, 15, 20, 30, 40])
    assert q.dul[1] == 0.206 and abs(float(q.dul_input[1]) - 0.20575) < 1e-7
    with pytest.raises(NotImplementedError, match="MESOL = 1"):
        native_soil(p, "1")
    with pytest.raises(ValueError, match="missing"):
        native_soil(_profile(SDUL=[0.2, math.nan, 0.2, 0.2]))


def test_initial_soil_water_matching_and_bounds() -> None:
    s = native_soil(_profile())
    # no initial conditions: DUL everywhere
    np.testing.assert_array_equal(initial_soil_water(s, None), s.dul)
    rows = [{"ICBL": 15, "SH2O": 0.02}, {"ICBL": 30, "SH2O": 0.05}, {"ICBL": 60, "SH2O": 0.7}]
    sw = initial_soil_water(s, rows)
    # layer 1 below the air-dry 0.3 LL -> 0.3 LL; deeper layers below LL -> LL; above SAT -> SAT
    np.testing.assert_array_equal(sw, [float(decimal_of(F(0.3) * F(0.1))), 0.1, 0.1, 0.3, 0.3])
    with pytest.raises(ValueError, match=r"0\.75"):
        initial_soil_water(s, [{"ICBL": 60, "SH2O": 0.8}])


def test_find_soil_profile_search_order(tmp_path: Path) -> None:
    text = (
        "*SOILS: test\n\n*XXTEST0001  test        L       60 test\n"
        "@SITE        COUNTRY          LAT     LONG SCS FAMILY\n -99         -99              -99      -99 -99\n"
        "@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE\n    BN  0.13   6.0  0.60  73.0  1.00  1.00 IB001 IB001 IB001\n"
        "@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC  SLCL  SLSI  SLCF  SLNI  SLHW  SLHB  SCEC  SADC\n"
        "     5   -99 0.100 0.200 0.300 1.000  1.20  1.40  1.00   -99   -99   -99   -99   -99   -99   -99   -99\n"
        "    60   -99 0.100 0.200 0.300 1.000  1.20  1.40  1.00   -99   -99   -99   -99   -99   -99   -99   -99\n"
    )
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (b / "XX.SOL").write_text(text)
    (a / "SOIL.SOL").write_text(text.replace("XXTEST0001", "YYTEST0001"))
    p = find_soil_profile("XXTEST0001", [a, b])  # not in SOIL.SOL: the two-letter file
    assert p.id == "XXTEST0001" and p.n_layers == 2
    with pytest.raises(FileNotFoundError):
        find_soil_profile("ZZTEST0001", [a, b])


# ------------------------------------------------------------------------ weather
def test_daylength_and_hourly_mean_temperature() -> None:
    sun = daylength486(np.array([80, 172, 355]), 41.43)
    assert sun.dayl.dtype == np.float32
    assert abs(float(sun.dayl[0]) - 12.0) < 0.2 and float(sun.dayl[1]) > 15.0 > 9.5 > float(sun.dayl[2])
    np.testing.assert_allclose(sun.snup + sun.sndn, 24.0, rtol=1e-6)
    # polar day: clamped to 24 h (DSSAT 4.8.6)
    assert float(daylength486(np.array([172]), 80.0).dayl[0]) == 24.0
    tavg = hourly_mean_temperature(
        np.array([26.4], F), np.array([12.0], F), daylength486(np.array([178]), 41.43)
    )
    # SIAZ9601 1996-178: DSSAT-CSM 4.8.6 TAVG = 19.542358 (instrumented SPAM entry); within one REAL*4 unit
    assert abs(float(tavg[0]) - float(F(19.542358))) <= float(np.spacing(F(19.542358)))
    assert 12.0 < float(tavg[0]) < 26.4 and float(tavg[0]) != (26.4 + 12.0) / 2


def test_wind_at_two_metres() -> None:
    w = wind_at_2m(np.array([np.nan, 0.0, 100.0]), 3.0)
    assert w[0] == w[1] == F(DSSAT486_WEATHER.wind_default)
    np.testing.assert_allclose(w[2], 100.0 * (2.0 / 3.0) ** 0.2, rtol=1e-6)
    assert wind_at_2m(np.array([974.6]), 2.0)[0] == F(974.6)


def test_co2val_options(tmp_path: Path) -> None:
    f = tmp_path / "CO2048.WDA"
    f.write_text(
        "*CO2\n@CO2BAS\n   380.\n\n@ YEAR   DOY    CO2\n  1982    15 -99.99\n  1982    45 340.71\n  1982    74 340.80\n"
    )
    s = read_co2_series(f)
    assert s.base == 380.0 and s.co2.tolist() == [F(340.71), F(340.80)]
    days = [1982056, 1982073, 1982074, 1982075]
    np.testing.assert_array_equal(co2_daily(days, "M", s), F([340.71, 340.71, 340.80, 340.80]))
    np.testing.assert_array_equal(co2_daily(days, "D", s), F([380.0] * 4))
    np.testing.assert_array_equal(co2_daily(days[:2], "W", s, cco2=330.0), F([330.0, 330.0]))
    # a daily DCO2 wins over the site CCO2 and carries over
    got = co2_daily(days[:3], "W", s, cco2=330.0, dco2=np.array([np.nan, 410.0, np.nan]))
    np.testing.assert_array_equal(got, F([330.0, 410.0, 410.0]))
    np.testing.assert_array_equal(
        co2_daily(days[:2], "W", CO2Series(380.0, *(np.array([]),) * 3), dco2=np.array([np.nan, 400.0])),
        F([380.0, 400.0]),
    )


# ------------------------------------------------------------------------ management
def _filex(irrig: str = "R", mi: int = 1, **extra: object) -> dict[str, object]:
    tr = {
        "N": 1,
        "CU": 1,
        "FL": 1,
        "SA": 0,
        "IC": 1,
        "MP": 1,
        "MI": mi,
        "MF": 0,
        "MR": 0,
        "MC": 0,
        "MT": 0,
        "ME": 0,
        "MH": 0,
        "SM": 1,
    }
    tr.update(extra)
    return {
        "TREATMENTS": [tr],
        "FIELDS": {1: {"ID_SOIL": "XXTEST0001", "WSTA": "XXXX"}},
        "INITIAL CONDITIONS": {
            1: {"PCR": "MZ", "ICWD": -99, "ICRES": 1000, "ICRIP": 100, "ICRID": 15, "rows": []}
        },
        "IRRIGATION AND WATER MANAGEMENT": {
            1: {
                "EFIR": 0.75,
                "rows": [
                    {"IDATE": 82058, "IROP": "IR001", "IRVAL": 12.25},
                    {"IDATE": 82058, "IROP": "IR004", "IRVAL": 10},
                    {"IDATE": 82060, "IROP": "IR001", "IRVAL": -5},
                ],
            }
        },
        "SIMULATION CONTROLS": {
            1: {
                "OPTIONS": {"WATER": "Y", "CO2": "M"},
                "METHODS": {"MESEV": "S"},
                "MANAGEMENT": {"IRRIG": irrig},
            }
        },
    }


def test_reported_irrigation() -> None:
    days = [1982056, 1982057, 1982058, 1982059, 1982060]
    irr = irrigation_amounts(_filex(), 1, days)
    # 12.25 -> F5.1 "12.2" (tie to even), + 10.0, times the F5.3 efficiency; the negative amount skipped
    assert irr.dtype == np.float32
    np.testing.assert_array_equal(irr, F([0, 0, F(F(F(12.2) + F(10.0)) * F(0.75)), 0, 0]))
    np.testing.assert_array_equal(irrigation_amounts(_filex(irrig="N"), 1, days), np.zeros(5, F))
    np.testing.assert_array_equal(irrigation_amounts(_filex(mi=0), 1, days), np.zeros(5, F))
    with pytest.raises(NotImplementedError, match="IRRIG = A"):
        irrigation_amounts(_filex(irrig="A"), 1, days)


def test_scope_and_switches() -> None:
    assert unsupported_features(_filex(), 1) == []
    assert switches(_filex(), 1).mesev == "S" and switches(_filex(), 1).mesol == "2"
    got = unsupported_features(_filex(irrig="A", ME=1, MT=1), 1)
    assert any("IRRIG = A" in g for g in got) and any("WTHMOD" in g for g in got)
    x = _filex()
    x["INITIAL CONDITIONS"][1].update(ICRIP=0)  # type: ignore[index]
    assert any("surface residue" in g for g in unsupported_features(x, 1))


def test_residue_parameters_and_kep(tmp_path: Path) -> None:
    f = tmp_path / "RESCH048.SDA"
    f.write_text(
        "*X\n@RETYP CR      AM  WATFAC  EXTFAC   PSLIG   SCN   SCP   PRLIG   RCN   RCP Description\n"
        " RE001 --      37     3.8    0.80    0.10  1.10  0.20    0.15  0.90  0.09 Generic\n"
        " RE203 MZ      30     3.5    0.86    0.10  1.10  0.32    0.10  0.90  0.16 Maize residue\n"
    )
    r = residue_parameters("MZ", f)
    assert (r.am, r.watfac, r.extfac, r.restype) == (30.0, float(F(3.5)), float(F(0.86)), "RE203")
    assert residue_parameters("SB", f).restype == "RE001" and residue_parameters("", f).restype == "RE001"
    assert residue_parameters("MZ", None).am == 32.0
    assert extinction_kep(0.85) == float(F(0.6854839))


def test_switch_defaults_without_a_simulation_control_level() -> None:
    # IPSIM.for 110-160: no level (SM = 0, or a level not in the file) -> irrigation and residue
    # "as reported", tillage on; a blank column of a level -> off
    for x in (_filex(SM=0), _filex(SM=2)):
        sw = switches(x, 1)
        assert (sw.irrig, sw.resid, sw.till, sw.mesev) == ("R", "R", "Y", "R")
        days = [1982056, 1982057, 1982058]
        assert float(irrigation_amounts(x, 1, days)[2]) > 0
        assert any("tillage" in g for g in unsupported_features(_filex(SM=0, MT=1), 1))
        assert any("residue applications" in g for g in unsupported_features(_filex(SM=0, MR=1), 1))
    x = _filex()
    x["SIMULATION CONTROLS"][1]["MANAGEMENT"] = {}  # type: ignore[index]
    assert (switches(x, 1).irrig, switches(x, 1).till) == ("N", "N")


def test_irrigation_listed_after_a_later_event_is_not_applied() -> None:
    # IRRIG.for 704-713 stops its scan at the first event after the day
    x = _filex()
    x["IRRIGATION AND WATER MANAGEMENT"][1]["rows"] = [  # type: ignore[index]
        {"IDATE": 82058, "IROP": "IR001", "IRVAL": 10},
        {"IDATE": 82060, "IROP": "IR001", "IRVAL": 20},
        {"IDATE": 82059, "IROP": "IR001", "IRVAL": 30},
        {"IDATE": 82060, "IROP": "IR001", "IRVAL": 5},
    ]
    irr = irrigation_amounts(x, 1, [1982058, 1982059, 1982060])
    np.testing.assert_array_equal(irr, F([F(10) * F(0.75), 0.0, F(25) * F(0.75)]))


@pytest.mark.parametrize(
    ("edit", "feature"),
    [
        (lambda x: x["SIMULATION CONTROLS"][1]["OPTIONS"].update(WATER="N"), "WATER"),
        (lambda x: x["SIMULATION CONTROLS"][1]["METHODS"].update(WTHER="W"), "WTHER"),
        (lambda x: x["SIMULATION CONTROLS"][1]["METHODS"].update(EVAPO="F"), "EVAPO"),
        (lambda x: x["SIMULATION CONTROLS"][1]["METHODS"].update(MESOM="P"), "MESOM = P"),
        (lambda x: x["SIMULATION CONTROLS"][1]["METHODS"].update(MESOL="1"), "MESOL = 1"),
        (lambda x: x["SIMULATION CONTROLS"][1]["MANAGEMENT"].update(IRRIG="D"), "IRRIG = D"),
        (
            lambda x: x["IRRIGATION AND WATER MANAGEMENT"][1]["rows"].append(
                {"IDATE": 82059, "IROP": "IR011", "IRVAL": 50}
            ),
            "IR011",
        ),
        (lambda x: x["INITIAL CONDITIONS"][1].update(ICWD=80), "water table"),
        (lambda x: x["FIELDS"][1].update(PMALB=0.2, PMWD=30), "plastic mulch"),
        (lambda x: x["FIELDS"][1].update(FLDD=100), "tile drainage"),
        (
            lambda x: (
                x["TREATMENTS"][0].update(MR=1),
                x["SIMULATION CONTROLS"][1]["MANAGEMENT"].update(RESID="R"),
            ),
            "residue applications",
        ),
        (
            lambda x: (
                x["TREATMENTS"][0].update(MT=1),
                x["SIMULATION CONTROLS"][1]["OPTIONS"].update(TILL="Y"),
            ),
            "tillage",
        ),
    ],
)
def test_unported_features_are_refused(edit: Callable[[Any], object], feature: str) -> None:
    x = _filex()
    edit(x)
    got = unsupported_features(x, 1)
    assert any(feature in g for g in got), got
