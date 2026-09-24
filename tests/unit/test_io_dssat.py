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

from agrijax.io.dssat import (
    observed_date,
    parse_dssat_date,
    read_cul,
    read_eco,
    read_filex,
    read_plantgro,
    read_sol,
    read_spe,
    read_summary,
    read_wth,
    write_sol,
    write_wth,
)
from agrijax.io.dssat._fixed import fmt_num, header_tokens, split_fixed

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
    """DSSAT 4.8 ``Y4K_DOY`` (``DATES.for``, ``CROVER = 35``; the older ``Y2K_DOY`` rule with 40
    is no longer called anywhere in v4.8.6)."""
    assert parse_dssat_date("15001").isoformat() == "2015-01-01"
    assert parse_dssat_date("82057").isoformat() == "1982-02-26"
    assert parse_dssat_date("35365").isoformat() == "2035-12-31"
    assert parse_dssat_date("36001").isoformat() == "1936-01-01"
    assert parse_dssat_date("40366").isoformat() == "1940-12-31"
    assert parse_dssat_date("2016060").isoformat() == "2016-02-29"
    # with a YYYYDDD weather file DSSAT anchors YYDDD dates on its first date (FirstWeatherDate):
    # the first date on or after it with those two year digits
    assert parse_dssat_date("35120", first_weather="1930001").isoformat() == "1935-04-30"
    assert parse_dssat_date("29120", first_weather="1930001").isoformat() == "2029-04-30"
    assert parse_dssat_date("82057", first_weather=2079001).isoformat() == "2082-02-26"
    assert observed_date("35120", "1930001", first_weather="1930001").isoformat() == "1935-04-30"
    assert observed_date("35120", "1930001").isoformat() == "2035-04-30"


_WTH5 = """*WEATHER DATA : century test

@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT
  TEST   30.000  -80.000    10  20.0  10.0   -99   -99
@DATE  SRAD  TMAX  TMIN  RAIN
{rows}
"""


def test_read_wth_five_digit_years_follow_y2k_doyw(tmp_path: Path) -> None:
    """``Y2K_DOYW``: YYDDD records keep the century of the first one and move to the next century
    only after a year xx99 (so a 1930s file can be read with ``century=19``)."""
    rows = "\n".join(f"{c}  10.0  30.0  20.0   0.0" for c in ("99364", "99365", "00001", "00002"))
    f = tmp_path / "TEST9901.WTH"
    f.write_text(_WTH5.format(rows=rows))
    got = [d.date().isoformat() for d in read_wth(f)["date"]]
    assert got == ["1999-12-30", "1999-12-31", "2000-01-01", "2000-01-02"]
    rows = "\n".join(f"{c}  10.0  30.0  20.0   0.0" for c in ("35365", "36001"))
    f.write_text(_WTH5.format(rows=rows))
    assert [d.year for d in read_wth(f)["date"]] == [2035, 2036]  # no xx99: the century stays
    assert [d.year for d in read_wth(f, century=19)["date"]] == [1935, 1936]


# --------------------------------------------------------------------------- soil (inline)

_SOL = """*SOILS: test

*TEST000001  TEST        S       100 Test profile
@SITE        COUNTRY          LAT     LONG SCS FAMILY
 Somewhere   Nowhere       30.000  -80.000 Test family
@ SCOM  SALB  SLU1  SLDR  SLRO  SLNF  SLPF  SMHB  SMPX  SMKE
    BN  0.13   6.0  0.60  73.0  1.00  1.00 IB001 IB001 IB001
@  SLB  SLMH  SLLL  SDUL  SSAT  SRGF  SSKS  SBDM  SLOC   !  NOTE
    10    A1 0.050 0.150 0.400  1.00101.32 1.20 1.552    !  9.9
    30    A2 0.060 0.160 0.410  0.50  2.50  1.30  0.80   !  9.9

    60    B1 0.070 0.170 0.420  0.20  1.00  1.40  0.30   !  9.9
@  SLB  SLPX  SLKE
    10   -99   0.6
    30   -99   0.7
"""


def _sol(tmp_path: Path, text: str = _SOL) -> Path:
    f = tmp_path / "TE.SOL"
    f.write_text(text)
    return f


def test_read_sol_default_reads_values_as_written(tmp_path: Path) -> None:
    lay = read_sol(_sol(tmp_path))["TEST000001"].layers
    # a blank line does not end a layer table (DSSAT's IGNORE3 skips it)
    assert lay["SLB"].tolist() == [10.0, 30.0, 60.0]
    # values touching their left neighbour are read whole
    assert lay["SSKS"].tolist() == [101.32, 2.5, 1.0]
    assert lay["SLOC"].tolist() == [1.552, 0.8, 0.3]
    # PARSE_HEADERS ends the header at '!': the note column is not a variable
    assert "NOTE" not in lay.columns and "!" not in lay.columns
    assert lay["SLKE"].iloc[:2].tolist() == [0.6, 0.7] and math.isnan(lay["SLKE"].iloc[2])


def test_read_sol_dssat_spans_reads_what_dscsm048_reads(tmp_path: Path) -> None:
    """DSSAT's spans leave out the column after each header token: ``1.00101.32`` under
    ``SRGF  SSKS`` is SRGF = ``1.00``, SSKS = ``1.32``; `` 1.20 1.552 `` under ``  SBDM  SLOC`` is
    SLOC = ``.552`` when both sit one column left (worked out by hand from ``PARSE_HEADERS``)."""
    lay = read_sol(_sol(tmp_path), dssat_spans=True)["TEST000001"].layers
    assert lay["SLB"].tolist() == [10.0, 30.0, 60.0]
    assert lay["SRGF"].tolist() == [1.0, 0.5, 0.2]
    assert lay["SSKS"].tolist() == [1.32, 2.5, 1.0]
    assert lay["SLOC"].tolist() == [0.552, 0.8, 0.3]
    assert lay["SLMH"].tolist() == ["A1", "A2", "B1"]
    prof = read_sol(_sol(tmp_path), dssat_spans=True)["TEST000001"]
    assert prof.surface["SALB"] == 0.13 and prof.surface["SMKE"] == "IB001"


def test_read_sol_second_table_paired_by_row_in_dssat(tmp_path: Path) -> None:
    """DSSAT puts row i of every layer table into layer i (and re-reads the depth); the default
    reader joins on SLB. Both warn when the SLB columns differ."""
    text = _SOL.replace("    10   -99   0.6\n    30   -99   0.7", "    30   -99   0.6\n    60   -99   0.7")
    assert text != _SOL
    with pytest.warns(UserWarning, match="pairs the tables by row"):
        by_slb = read_sol(_sol(tmp_path, text))["TEST000001"].layers
    assert math.isnan(by_slb["SLKE"].iloc[0]) and by_slb["SLKE"].iloc[1:].tolist() == [0.6, 0.7]
    with pytest.warns(UserWarning, match="pairs the tables by row"):
        by_row = read_sol(_sol(tmp_path, text), dssat_spans=True)["TEST000001"].layers
    assert by_row["SLKE"].iloc[:2].tolist() == [0.6, 0.7] and math.isnan(by_row["SLKE"].iloc[2])
    assert by_row["SLB"].tolist() == [30.0, 60.0, 60.0]  # ZLYR(L) read again from the 2nd table


def test_write_sol_keeps_a_blank_before_every_value(tmp_path: Path) -> None:
    """A value filling all six columns (``101.32``) widens its column, so DSSAT's spans read it
    whole: the written file read the DSSAT way gives the values written."""
    a = read_sol(_sol(tmp_path))
    out = write_sol(a, tmp_path / "W.SOL")
    b = read_sol(out, dssat_spans=True)["TEST000001"].layers
    pd.testing.assert_frame_equal(a["TEST000001"].layers, b, check_exact=True)
    head = next(ln for ln in out.read_text().splitlines() if ln.startswith("@  SLB  SLMH"))
    assert "   SSKS" in head  # seven columns for SSKS


# --------------------------------------------------------------------------- FileX (inline)

_FILEX = """*EXP.DETAILS: TEST0001SQ TEST SEQUENCE

*TREATMENTS                        -------------FACTOR LEVELS------------
@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM
 1 1 1 0 Maize one                  1  1  0  1  1  0  0  0  0  0  0  0  1
 1 9 1 0 Maize nine                 1  1  0  1  1  0  0  0  0  0  0  0  1
 110 1 0 Maize ten                  1  1  0  1  1  0  0  0  0  0  0  0  1

*CULTIVARS
@C CR INGENO CNAME
 1 MZ IB0035 McCurdy 84aa

*SIMULATION CONTROLS
@N GENERAL     NYERS NREPS START SDATE RSEED SNAME.................... SMODEL
 1 GE              1     1     S 82056  2150 RAINFED LOW NITROGEN
@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2
 1 OP              Y     N     N     N     N     N     N     N     M
@N HARVEST     HFRST HLAST HPCNP HPCNR
 1 HA              0     1     0     0
@N HARVEST     HFRST HLAST HPCNP HPCNR
 1 HA              0  1001   100     0
"""


@pytest.mark.parametrize(("suffix", "keys"), [(".SQX", (1, 10, 1, 0)), (".MZX", (11, 0, 1, 0))])
def test_filex_treatment_keys_follow_ipexp_formats(
    tmp_path: Path, suffix: str, keys: tuple[int, ...]
) -> None:
    """``IPEXP``: ``(I3,I1,1X,I1,1X,I1,...)``, or ``(2I2,1X,I1,1X,I1,...)`` in sequence mode: `` 110 1 0``
    is treatment 1 / rotation 10 in a ``.SQX``, treatment 11 / rotation 0 otherwise."""
    f = tmp_path / f"TEST0001{suffix}"
    f.write_text(_FILEX)
    trt = read_filex(f)["TREATMENTS"]
    assert [(t["N"], t["R"]) for t in trt[:2]] == [(1, 1), (1, 9)]
    assert (trt[2]["N"], trt[2]["R"], trt[2]["O"], trt[2]["C"]) == keys
    assert trt[2]["TNAME"] == "Maize ten" and trt[2]["SM"] == 1


def test_filex_simulation_controls_first_line_of_a_group_wins(tmp_path: Path) -> None:
    """``IPSIM`` reads the lines of a level in order, so of two HARVEST lines it uses the first."""
    f = tmp_path / "TEST0001.MZX"
    f.write_text(_FILEX)
    sc = read_filex(f)["SIMULATION CONTROLS"][1]
    assert sc["OPTIONS"]["WATER"] == "Y" and sc["OPTIONS"]["NITRO"] == "N"  # ISWWAT, ISWNIT
    assert sc["GENERAL"]["SDATE"] == 82056 and sc["GENERAL"]["SNAME"] == "RAINFED LOW NITROGEN"
    assert sc["HARVEST"]["HLAST"] == 1 and sc["HARVEST"]["HPCNP"] == 0


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
