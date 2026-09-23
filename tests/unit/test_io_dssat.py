"""DSSAT readers/writers on inline snippets and the public BSD-3 fixtures in ``tests/fixtures/dssat``.

No private data: the tests that need the CA-TPA scenario, the DSSAT example tree or a DSSAT run
are in ``tests/integration/test_io_dssat_data.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_jax.io.dssat import (
    parse_dssat_date,
    read_cul,
    read_eco,
    read_plantgro,
    read_spe,
    read_summary,
    read_wth,
    write_wth,
)
from agri_jax.io.dssat._fixed import fmt_num, header_tokens, split_fixed

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dssat"


@pytest.fixture(scope="module")
def genotype48() -> Path:
    """MZCER048.* copied from dssat-csm-os (BSD-3); always present."""
    assert (FIXTURES / "MZCER048.CUL").is_file(), FIXTURES
    return FIXTURES


# --------------------------------------------------------------------------- fixed-width


def test_split_fixed_text_columns_and_overruns() -> None:
    hdr = "@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME"
    row = " 1 UFGA0002 UFGA       -99     0 DR000     0     0 00000 -99    180  IBMZ910014 Field section"
    toks = header_tokens(hdr)
    vals = dict(zip([t[0] for t in toks], split_fixed(toks, row), strict=True))
    assert vals["WSTA"] == "UFGA"
    assert vals["ID_SOIL"] == "IBMZ910014"  # overruns its 7-character header token
    assert vals["FLNAME"] == "Field section"
    assert vals["FLST"] == "00000"


def test_split_fixed_glued_values() -> None:
    toks = header_tokens("@DATE  SRAD  TMAX  TMIN  RAIN  RHUM  WIND")
    assert split_fixed(toks, "15317   4.8   8.4   2.9   1.0  77.41004.0")[-2:] == ["77.4", "1004.0"]
    toks = header_tokens("@  SLB  SLMH  SLLL")
    assert split_fixed(toks, "   210C2cacs 0.310") == ["210", "C2cacs", "0.310"]
    # a right-aligned value one column past its header token stays whole
    toks = header_tokens("@ECO#  ECONAME.........  TBASE TOPT  ROPT  P20   DJTI")
    row = "IB0001 GENERIC MIDWEST1    8.0 34.0  34.0  12.5   4.0"
    assert split_fixed(toks, row) == ["IB0001", "GENERIC MIDWEST1", "8.0", "34.0", "34.0", "12.5", "4.0"]


def test_fmt_num() -> None:
    assert fmt_num(0.03, 6) == " 0.030"
    assert fmt_num(127.75, 6) == "127.75"  # uses the separating blank only when needed
    assert fmt_num(math.nan, 6) == "   -99"
    assert fmt_num(0.0012, 6) == " .0012"
    assert fmt_num(180, 6) == "   180"


def test_parse_dssat_date() -> None:
    assert parse_dssat_date("15001").isoformat() == "2015-01-01"
    assert parse_dssat_date("82057").isoformat() == "1982-02-26"
    assert parse_dssat_date("40366").isoformat() == "2040-12-31"
    assert parse_dssat_date("41001").isoformat() == "1941-01-01"
    assert parse_dssat_date("2016060").isoformat() == "2016-02-29"


# --------------------------------------------------------------------------- outputs (inline)

_SUMMARY = """*SUMMARY : UFGA8201MZ NIT X IRR, GAINESVILLE 2N*3I

!IDENTIFIERS......................... EXPERIMENT AND TREATMENT.......... SITE INFORMATION
@   RUNNO   TRNO R# O# P# CR MODEL... EXNAME.. TNAM..................... FNAM.... WSTA.... WYEAR SOIL_ID...             XLAT            LONG      ELEV    SDAT    PDAT    HWAM    HWUM
        1      1  1  0  1 MZ MZCER048 UFGA8201 RAINFED LOW NITROGEN      UFGA0002 UFGA8201  1982 IBMZ910014         -82.3689          29.638        40 1982056 1982057    2293  0.3090
        4      4  1  0  1 MZ MZCER048 UFGA8201 IRRIGATED HIGH NITROGEN   UFGA0002 UFGA8201  1982 IBMZ910014         -82.3689          29.638        40 1982056 1982057   11854     -99
"""


def test_read_summary_inline(tmp_path: Path) -> None:
    p = tmp_path / "Summary.OUT"
    p.write_text(_SUMMARY)
    s = read_summary(p)
    assert list(s["TNAM"]) == ["RAINFED LOW NITROGEN", "IRRIGATED HIGH NITROGEN"]
    assert list(s["HWAM"]) == [2293, 11854]
    assert s["HWAM"].dtype.kind == "i"
    assert s["HWUM"].iloc[0] == pytest.approx(0.309) and np.isnan(s["HWUM"].iloc[1])
    assert list(s["SOIL_ID"]) == ["IBMZ910014"] * 2
    assert s["PDAT"].iloc[0] == 1982057


_PLANTGRO = """*GROWTH ASPECTS OUTPUT FILE

*RUN   2        : RAINFED HIGH NITROGEN     MZCER048 UFGA8201    2
 TREATMENT  2   : RAINFED HIGH NITROGEN     MZCER048

@YEAR DOY   DAS   DAP   LAID   CWAD
 1982 057     2     0   0.00      0
 1982 058     3     1   0.01     12
"""


def test_read_plantgro_inline(tmp_path: Path) -> None:
    p = tmp_path / "PlantGro.OUT"
    p.write_text(_PLANTGRO)
    g = read_plantgro(p)
    assert list(g["RUN"]) == [2, 2] and list(g["TRNO"]) == [2, 2]
    assert g["DATE"].iloc[0] == pd.Timestamp("1982-02-26")
    assert g["CWAD"].iloc[1] == 12


# --------------------------------------------------------------------------- weather


def test_write_wth_keeps_two_decimals(tmp_path: Path) -> None:
    """DSSAT allows two-decimal columns (TDEW 12.35, RHUM 77.45): write_wth must not round them."""
    w = pd.DataFrame(
        {
            "date": pd.to_datetime(["2015-01-01", "2015-01-02", "2015-01-03"]),
            "srad": [8.4, 12.25, 30.0],
            "tmax": [2.0, -10.3, 35.125],
            "tmin": [-5.1, -21.05, 20.0],
            "rain": [0.0, 12.7, 101.6],
            "tdew": [12.35, -3.05, 0.5],
            "rhum": [77.45, 77.4, 100.0],
            "wind": [1004.0, 52.7, 250.0],
        }
    )
    w.attrs["site"] = {"INSI": "XXXX", "LAT": 42.695, "LONG": -80.35, "ELEV": 200.0}
    for four in (False, True):
        out = write_wth(w, tmp_path / f"T{int(four)}.WTH", four_digit_year=four)
        w2 = read_wth(out)
        pd.testing.assert_frame_equal(w[list(w2.columns)], w2, check_exact=True, check_dtype=False)
    text = (tmp_path / "T0.WTH").read_text()
    assert " 12.35" in text and " 77.45" in text and "35.125" in text
    assert "  8.4" in text  # one-decimal values keep the DSSAT look


# --------------------------------------------------------------------------- genotype


def test_read_eco_048(genotype48: Path) -> None:
    e8 = read_eco(genotype48 / "MZCER048.ECO")
    assert {"TSEN", "CDAY"} <= set(e8.columns)
    assert e8.loc["IB0001", "TBASE"] == 8.0 and e8.loc["IB0001", "CDAY"] == 15.0


def test_read_spe_048(genotype48: Path) -> None:
    s8 = read_spe(genotype48 / "MZCER048.SPE")
    assert s8["RGFIL"] == (5.5, 16.0, 27.0, 35.0)
    assert s8["CTCNP1"] == 1.52 and s8["SRATPHOTO"] == 0.8 and s8["KEP"] == 0.68
    assert len(s8.rows["PHOSPHORUS CONTENT (g [P]/g [shoot])"]) >= 12


def test_read_cul_048(genotype48: Path) -> None:
    c = read_cul(genotype48 / "MZCER048.CUL")
    assert list(c.columns) == ["VRNAME", "EXPNO", "ECO#", "P1", "P2", "P5", "G2", "G3", "PHINT"]
    r = c.loc["IB0035"]  # the UFGA8201 cultivar
    assert r["VRNAME"] == "McCurdy 84aa" and r["ECO#"] == "IB0001"
    assert (r["P1"], r["P2"], r["P5"], r["G2"], r["G3"], r["PHINT"]) == (
        259.0, 1.193, 947.1, 924.3, 8.168, 43.0
    )  # fmt: skip
    assert c.loc["999991", "VRNAME"] == "MINIMA"
    assert bool(c[["P1", "P5", "G2", "G3", "PHINT"]].notna().to_numpy().all())
