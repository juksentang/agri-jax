"""``agrijax.io.dssat.native_crop`` on synthetic files: the ``DSSAT48.INP`` print / read round trip of
the cultivar and planting values, the defaults of the input module, the crop weather as
``Weather.OUT`` prints it, the weather file name and the simulation start (the equality with real
``DSSAT48.INP`` / ``Weather.OUT`` files is ``tests/integration/test_dssat_free_inputs.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from agrijax.io.dssat.native_crop import (
    crop_weather,
    native_cultivar,
    native_planting,
    season_start,
    weather_file_name,
)

_CUL = (
    "*MAIZE CULTIVAR COEFFICIENTS: MZCER048 MODEL\r\n"
    "@VAR#  VRNAME.......... EXPNO   ECO#    P1    P2    P5    G2    G3 PHINT\r\n"
    "IB0035 McCurdy 84aa         . IB0001 259.0 1.193 947.1 924.3  8.17 43.00\r\n"
    "XX0001 Long digits          . IB0002 259.04 1.1934 947.15 924.36 8.175 43.004\r\n"
)


def test_cultivar_as_dssat48_inp_holds_it(tmp_path):
    cul = tmp_path / "MZCER048.CUL"
    cul.write_bytes(_CUL.encode("latin-1"))
    c = native_cultivar(cul, "IB0035")
    assert c == {
        "P1": 259.0,
        "P2": 1.193,
        "P5": 947.1,
        "G2": 924.3,
        "G3": 8.17,
        "PHINT": 43.0,
        "ECO": "IB0001",
    }
    # values with more digits than the INP formats (F6.1, F6.3, F6.1, F6.1, F6.2, F6.2) are rounded there
    d = native_cultivar(cul, "XX0001")
    assert (d["P1"], d["P2"], d["P5"], d["G2"], d["G3"], d["PHINT"]) == (
        259.0,
        1.193,
        947.2,
        924.4,
        8.18,
        43.0,
    )
    with pytest.raises(ValueError, match="not in"):
        native_cultivar(cul, "NOPE01")


def _x(planting: dict, general: dict | None = None) -> dict:
    return {
        "TREATMENTS": [{"N": 1, "CU": 1, "FL": 1, "MP": 1, "SM": 1}],
        "PLANTING DETAILS": {1: planting},
        "SIMULATION CONTROLS": {1: {"GENERAL": general or {"START": "S", "SDATE": 82056}}},
    }


def test_planting_and_its_defaults():
    p = native_planting(_x({"PDATE": 82057, "PPOP": 7.2, "PPOE": 7.2, "PLRS": 61, "PLDP": 7}), 1)
    assert p == {"yrplt": 1982057, "pltpop": 7.2, "rowspc": 61.0, "sdepth": 7.0}
    # PPOE missing -> PPOP; row spacing missing -> 100 / sqrt(PLTPOP) (REAL*4), printed F5.0
    q = native_planting(_x({"PDATE": 82057, "PPOP": 6.25, "PPOE": -99, "PLRS": -99, "PLDP": 5.25}), 1)
    assert q["pltpop"] == 6.2 and q["rowspc"] == 40.0 and q["sdepth"] == 5.2
    with pytest.raises(ValueError, match="plant population"):
        native_planting(_x({"PDATE": 82057, "PPOP": -99, "PPOE": -99, "PLRS": 61, "PLDP": 7}), 1)
    with pytest.raises(ValueError, match="planting depth"):
        native_planting(_x({"PDATE": 82057, "PPOP": 7, "PPOE": 7, "PLRS": 61, "PLDP": -99}), 1)


def test_simulation_start():
    x = _x({"PDATE": 82057, "PPOP": 7, "PPOE": 7, "PLRS": 61, "PLDP": 7})
    assert season_start(x, 1) == 1982056
    x["SIMULATION CONTROLS"][1]["GENERAL"] = {"START": "P", "SDATE": 82001}
    assert season_start(x, 1) == 1982057
    x["SIMULATION CONTROLS"][1]["GENERAL"] = {"START": "E", "SDATE": 82001}
    with pytest.raises(NotImplementedError, match="emergence"):
        season_start(x, 1)


def test_crop_weather_is_printed_as_weather_out():
    w = crop_weather(
        np.float32([27.2, 30.05]),
        np.float32([10.6, -1.25]),
        np.float32([14.8, 0.1]),
        np.float32([340.73, 341.25]),
    )
    assert w["tmax"].tolist() == [27.2, 30.0]  # REAL*4 30.05 is 30.0499..., printed 30.0
    assert w["tmin"].tolist() == [10.6, -1.2] and w["srad"].tolist() == [14.8, 0.1]
    assert w["co2"].tolist() == [340.7, 341.2]  # F7.1 of the REAL*4 value (341.25 is exact: ties to even)
    assert w["tmax"].dtype == np.float64


def test_weather_file_name():
    assert weather_file_name("UFGA", 1982) == "UFGA8201.WTH"
    assert weather_file_name("UFGA", 2005) == "UFGA0501.WTH"
    assert weather_file_name("CTPA1501", 2015) == "CTPA1501.WTH"
