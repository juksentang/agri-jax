"""Readers / writers of agri_jax.io.rzwqm on inline snippets (no data).

The tests against the CA-TPA scenario files, the batch parameter map and GenerateDat.py are in
``tests/integration/test_io_rzwqm_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_jax.io.rzwqm import (
    KEY_COLUMNS,
    key_variables,
    read_ana,
    read_met,
    read_overview_yields,
)
from agri_jax.io.rzwqm._fortran import fortran_float
from agri_jax.io.rzwqm.ana import parse_ana_header, slugify
from agri_jax.io.rzwqm.dat import format_value
from agri_jax.io.rzwqm.met import WIND_FLOOR_KM_D, prepare_rzwqm_forcing
from agri_jax.io.rzwqm.params import canonical_name

# ----------------------------------------------------------------------------- fixtures


# ----------------------------------------------------------------------------- rzwqm.dat


def test_format_value_matches_generatedat() -> None:
    assert format_value(0.3) == "0.300"
    assert format_value(np.float64(2.1234567)) == "2.12346"
    assert format_value(54) == "54.00"
    assert format_value(1e-5) == "1e-05"
    assert format_value(0.123456, decimals=None) == "0.123456"
    assert format_value("abc") == "abc"


# ----------------------------------------------------------------------------- params
def test_canonical_name() -> None:
    assert canonical_name("Pore Size (c2)") == ("lam_1", "lam", 1)
    assert canonical_name("FC 1/10 WC (C5)") == ("theta_fc10_4", "theta_fc10", 4)
    assert canonical_name("Albedo Maturity") == ("albedo_crop", "albedo_crop", None)
    assert canonical_name("Stomatal Resistance corn") == ("rs_min_corn", "rs_min", None)
    with pytest.raises(KeyError):
        canonical_name("Unknown Thing")


# ----------------------------------------------------------------------------- MET / BRK


# ----------------------------------------------------------------------------- .ana
_ANA_HEADER = [
    "1) TIME (YEAR.DAY)                       3) LEAF AREA INDEX                  5) PEST #1 IN TILE DRAINAGE (KG/HA/DAY)6) SEED BIOMASS (G/PLANT)",
    "2) STORED SOIL WATER (CM)                4) PLANT GROWTH STAGE (0-1)        ",
    "",
    "   (1)         (2)            ( 3)            ( 4)            ( 5)       (6)",
    "=" * 80,
]


def test_parse_ana_header_tricky_names() -> None:
    entries, n = parse_ana_header([*_ANA_HEADER, "2015.000 1 2 3 4 5"])
    assert n == 5
    assert entries == {
        1: "TIME (YEAR.DAY)",
        2: "STORED SOIL WATER (CM)",
        3: "LEAF AREA INDEX",
        4: "PLANT GROWTH STAGE (0-1)",
        5: "PEST #1 IN TILE DRAINAGE (KG/HA/DAY)",
        6: "SEED BIOMASS (G/PLANT)",
    }
    assert slugify("PEST #1 IN TILE DRAINAGE") == "pest_1_in_tile_drainage"
    assert slugify("2ND THING") == "v_2nd_thing"


def test_read_ana_synthetic(tmp_path: Path) -> None:
    rows = [
        "2015.000     28.7550         0.00000   0.0 0.0 0.0",
        "2015.001     28.7217        0.273920E-01  1.5  0.5 0.1234-100",
        "2016.366     27.0        *******  2.0  1.0 0.0",
    ]
    p = tmp_path / "x.ana"
    p.write_text("\n".join(_ANA_HEADER + rows) + "\n")
    ds = read_ana(p)
    assert list(ds.data_vars) == [
        "stored_soil_water",
        "leaf_area_index",
        "plant_growth_stage",
        "pest_1_in_tile_drainage",
        "seed_biomass",
    ]
    assert ds["stored_soil_water"].attrs == {"long_name": "STORED SOIL WATER", "units": "CM", "column": 2}
    np.testing.assert_array_equal(
        ds.time.values,
        np.array(["2014-12-31", "2015-01-01", "2016-12-31"], dtype="datetime64[ns]"),
    )
    assert ds["leaf_area_index"].values[1] == pytest.approx(0.027392)
    assert np.isnan(ds["leaf_area_index"].values[2])
    assert ds["seed_biomass"].values[1] == pytest.approx(0.1234e-100)
    k = key_variables(ds, {"profile_water_cm": 2, "lai": 3})
    assert list(k.data_vars) == ["profile_water_cm", "lai"]


def test_fortran_float() -> None:
    assert fortran_float("1.0D-03") == 1e-3
    assert fortran_float("0.1234-100") == pytest.approx(0.1234e-100)
    assert fortran_float("-0.5+105") == pytest.approx(-0.5e105)
    assert np.isnan(fortran_float("********"))
    assert fortran_float("0.273920E-01") == 0.027392


def test_key_columns_are_the_documented_ones() -> None:
    assert KEY_COLUMNS["profile_water_cm"] == 2 and KEY_COLUMNS["aet_cm"] == 84
    assert KEY_COLUMNS["lai"] == 43 and KEY_COLUMNS["grain_kg_ha"] == 44


# ----------------------------------------------------------------------------- OVERVIEW.OUT
_OVERVIEW = """*SIMULATION OVERVIEW FILE

*RUN   1        : /x
 PLANTING DATE  : APR 28 2015    PLANTS/m2 :  8.0     ROW SPACING :  76.cm
 HARVEST DATE   : SEP 25 2015
\x00\x00\x00\x00
                     Maize YIELD :     9916 kg/ha    [Dry weight]
*RUN   2        : /x
 PLANTING DATE  : MAY  5 2016    PLANTS/m2 :  7.5     ROW SPACING :  76.cm
 HARVEST DATE   : OCT  2 2016
                     Maize YIELD :     9608 kg/ha    [Dry weight]
"""


def test_read_overview_yields_synthetic(tmp_path: Path) -> None:
    p = tmp_path / "OVERVIEW.OUT"
    p.write_bytes(_OVERVIEW.encode("latin-1") + b"\xff\xfe")
    df = read_overview_yields(p)
    assert df.season.tolist() == [1, 2]
    assert df.crop.tolist() == ["Maize", "Maize"]
    assert df.harvest_date.tolist() == [pd.Timestamp("2015-09-25"), pd.Timestamp("2016-10-02")]
    assert df.planting_date.tolist() == [pd.Timestamp("2015-04-28"), pd.Timestamp("2016-05-05")]
    assert df.plants_m2.tolist() == [8.0, 7.5]
    assert df.yield_kg_ha.tolist() == [9916.0, 9608.0]


_MET = """= daily meteorology (synthetic)
2015 1 1 2015 1 3 0
1 2015  -5.0   2.0   52.7  8.0  0.0  80.0  0.0  0.0
2 2015  -8.0   1.0  250.0  9.5  0.0  99.0  0.0  3.2
3 2015 -60.0  55.0 5000.0 50.0  0.0 120.0  0.0  0.0
"""


def test_prepare_rzwqm_forcing_wind_floor_and_bounds(tmp_path: Path) -> None:
    p = tmp_path / "X.MET"
    p.write_text(_MET)
    raw = read_met(p)
    assert list(raw["wind_run_km"]) == [52.7, 250.0, 5000.0]  # the reader returns the file values
    met = prepare_rzwqm_forcing(raw)
    assert list(met["wind_run_km"]) == [WIND_FLOOR_KM_D, 250.0, 4700.0]  # UBREEZ floor, TUX cap
    assert list(met["tmin"]) == [-5.0, -8.0, -50.0] and list(met["tmax"]) == [2.0, 1.0, 50.0]
    assert list(met["srad_mj"]) == [8.0, 9.5, 45.0] and list(met["rh"]) == [80.0, 99.0, 100.0]
    assert list(met["rain_mm"]) == list(raw["rain_mm"])
    assert met.attrs["prepared"] == "INPDAY" and "prepared" not in raw.attrs
    pd.testing.assert_frame_equal(read_met(p, prepare=True), met)
    assert prepare_rzwqm_forcing(raw, wind_floor_km_d=60.0)["wind_run_km"].iloc[0] == 60.0
