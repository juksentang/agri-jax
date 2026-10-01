"""DSSAT observed-data readers (FILEA / FILET) and their conversion to calibration targets.

The first part needs no data: small inline files written to ``tmp_path``. References are
independent of the code under test: values written by hand in the fixture, the FILEA rule of
DSSAT-CSM v4.8.6.0 CERES-Maize (``READA_Y4K``, ``Utilities/READS.for:1176-1323``, called from
``MZ_OPHARV.for:340-341``, with ``PARSE_HEADERS``, ``READS.for:997-1081``) restated here in its own
1-based Fortran column arithmetic without the package helpers, and a simulator that returns the
observed values exactly (zero loss).

The second part (``allow_skip``) reads every observed-data file of the DSSAT v4.8.6.0 example
tree under ``$AGRI_JAX_DSSAT/example_data``: the maize ``*.MZA`` / ``*.MZT`` with counts; every
FILEA of every crop against the ``READA_Y4K`` restatement (0 differences); the maize FILEA against
the header-position reader of the ``*.OUT`` files (a second implementation, 0 differences); the
engine's reading of ``GHNY9801.PNA`` ``SHAD`` (a misaligned header); every FILET read.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax.calib import observations as obs_mod
from agrijax.calib.observations import DSSAT_TARGETS, observation_targets
from agrijax.io.dssat._fixed import read_lines
from agrijax.io.dssat.observed import (
    find_observed_files,
    read_filea,
    read_filet,
    read_observed,
    reada_date,
    y4k_date,
)
from agrijax.io.dssat.outputs import observed_date, read_out
from agrijax.port.run_fortran import DEFAULT_DSSAT_ENGINE

DSSAT_ENGINE = Path(os.environ.get("AGRI_JAX_DSSAT", str(DEFAULT_DSSAT_ENGINE))).expanduser()

# FILEA: a HWAH column and a later lower-case hwam column (both HWAM for READA_Y4K, the last wins),
# a -99, a blank field (TRNO 2 CWAM), a text column, day-of-year and YYDDD dates, a '-99' header
# placeholder, a comment, a repeated TRNO (the first row wins), a second '@' table (read with the
# first table's columns) and a row after a '*' line (still read)
FILEA = """\
*EXP. DATA (A): TEST0101MZ INLINE FIXTURE

! a comment line
@TRNO   TNAM  HWAH  CWAM  ADAT  MDAT   -99  hwam
     1  LOWN 2929. 5532.   132   185   -99  3000
     2 HIGHN 3130.         133 82186   -99   -99
!    3 commented out
     3   -99   -99 14581   -99   -99   -99   -99
     1 AGAIN  9999  9999   999   999   -99  9999

@TRNO  L#SM
     4  20.0
*ANOTHER SECTION
     5     X  1234  5678   140   190   -99  4321
"""

# a header off the six-column grid: CWAM one column left, so SHAD's text starts with the tail of
# CWAM's value (the situation of GHNY9801.PNA)
FILEA_MISALIGNED = """\
*EXP. DATA (A): TEST0201MZ MISALIGNED
@TRNO   HWAM CWAM   SHAD  ADAT
     1  1491. 5091. 587.2   166
"""

FILET = """\
*EXP. DATA (T): TEST0101MZ INLINE FIXTURE
! comment
@TRNO   DATE  CWAD  LAID  SW1D
     1 82057     0  0.00  .200
     1 82089    88  0.17   -99
     1 82103   341  0.56  .180
     2 82089    96  0.19   -99
     2 82089   100   -99   -99

@TRNO    DATE  GWAD
     1 1982145    50
     2 1982145   -99
"""

# a FILEA date of 0.5 (TRNO 1): READA_Dates truncates it to 0, which is no date
FILEA_FRACTIONAL = """\
*EXP. DATA (A): TEST0501MZ FRACTIONAL DATE
@TRNO   HWAM  ADAT  MDAT
     1  3000   0.5   185
     2  3100   133   186
"""

_ALIAS = {"R5AT": "PDFT", "PDFD": "PDFT", "BWAH": "BWAM", "HWAH": "HWAM", "CWAH": "CWAM", "HDAT": "R8AT"}


def _write(tmp_path: Path, name: str, text: str, *, crlf: bool = False) -> Path:
    p = tmp_path / name
    p.write_bytes(text.replace("\n", "\r\n" if crlf else "\n").encode("latin-1"))
    return p


def _parse_headers(char: str) -> tuple[list[str], list[tuple[int, int]]]:
    """PARSE_HEADERS (READS.for:997-1081): names and 1-based inclusive columns."""
    length = len(char.rstrip())
    col: list[tuple[int, int]] = []
    c1, spaces = 1, True
    for i in range(2, length + 1):
        ch = char[i - 1]
        if ch == "!":
            length = i - 1
            break
        if ch == " ":
            if not spaces:
                col.append((c1, i - 1))
                c1, spaces = i + 1, True
        else:
            spaces = False
    col.append((c1, length))
    names = [char[a - 1 : b].strip() for a, b in col]
    names[0] = char[1 : col[0][1]].strip()
    return [n.rstrip(". ") for n in names], col


def _reada_y4k(lines: list[str], trtnum: int, olab: list[str]) -> dict[str, str]:
    """READA_Y4K (READS.for:1176-1323) for one treatment: ``X`` text per label of ``olab``."""
    it = iter(lines)
    c255 = next((ln for ln in it if ln.startswith("@")), None)
    if c255 is None:
        return {}
    names, col = _parse_headers(c255[:255])
    names, col = names[:40], col[:40]
    fa = [_ALIAS.get(n[:15].upper(), n[:15].upper()) for n in names]
    acn = len(fa)
    found, tr = None, None
    for c255 in it:  # IGNORE: skip blank, '!' and '@' lines
        if not c255.strip() or c255[0] in "!@":
            continue
        rec = c255[:255].ljust(255)
        for j in range(acn):
            if fa[j] == "TRNO":
                a, b = col[j]
                txt = rec[a - 1 : b + 1][2:6].replace(" ", "")  # C255(C1:C2+1), (2X,I4)
                if not txt:
                    tr = 0
                elif re.fullmatch(r"[+-]?\d+", txt):
                    tr = int(txt)
                if tr == trtnum:
                    found = rec
                    break
        if found is not None:
            break
    if found is None:
        return {}
    x: dict[str, str] = {}
    for lab in olab:
        for j in range(1, acn + 1):
            if j == 1:
                c1, c2 = col[0][0], col[1][0]
            elif j == acn:
                c1, c2 = col[j - 2][1], col[j - 1][1] + 1
            else:
                c1, c2 = col[j - 2][1], col[j][0]
            if fa[j - 1] == lab:
                first = c1 + 2 if (j > 1 and fa[j - 2] == "TRNO") else c1 + 1
                x[lab] = found[first - 1 : c2 - 1][:12].rstrip()  # C255(first:C2-1), A12, TRIM
    return x


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


# --------------------------------------------------------------------------- FILEA


@pytest.mark.parametrize("crlf", [False, True])
def test_filea_reada_y4k_rule(tmp_path: Path, crlf: bool) -> None:
    p = _write(tmp_path, "TEST0101.MZA", FILEA, crlf=crlf)
    a = read_filea(p)
    assert a.title == "TEST0101MZ INLINE FIXTURE"
    assert a.n_tables == 2 and a.aliases == {"HWAH": "HWAM"} and a.ignored_columns == ()
    assert list(a.table.columns) == ["TNAM", "HWAM", "CWAM", "ADAT", "MDAT", "-99"]
    assert a.codes == ("TNAM", "HWAM", "CWAM", "ADAT", "MDAT")
    assert list(a.table.index) == [1, 2, 3, 4, 5]
    # by hand: the last HWAM column wins (hwam upper-cased), even when it is missing (TRNO 2)
    assert a.value(1, "HWAM") == 3000.0 and math.isnan(a.value(2, "HWAM")) and a.value(5, "HWAM") == 4321.0
    assert a.value(1, "CWAM") == 5532.0 and math.isnan(a.value(2, "CWAM"))  # blank: missing
    assert a.value(3, "CWAM") == 14581.0 and math.isnan(a.value(3, "ADAT"))
    assert a.value(2, "MDAT") == 82186.0 and a.value(5, "ADAT") == 140.0  # row after '*' read
    assert a.text.at[1, "TNAM"] == "  LOWN" and math.isnan(a.value(1, "TNAM"))  # text, no number
    assert a.text.at[4, "TNAM"] == "  20.0" and a.value(4, "TNAM") == 20.0  # 2nd table: 1st columns
    assert a.text.at[1, "HWAM"] == "  3000"  # the first TRNO 1 row, not 'AGAIN'
    assert math.isnan(a.value(9, "HWAM")) and math.isnan(a.value(1, "XXXX"))
    # the READA_Y4K restatement gives the same text for every treatment and label
    lines = FILEA.splitlines()
    for trno in (1, 2, 3, 4, 5):
        x = _reada_y4k(lines, trno, list(a.text.columns))
        for code in a.text.columns:
            assert a.text.at[trno, code] == x.get(code, ""), (trno, code)


def test_filea_misaligned_header_as_the_engine(tmp_path: Path) -> None:
    a = read_filea(_write(tmp_path, "TEST0201.MZA", FILEA_MISALIGNED))
    x = _reada_y4k(FILEA_MISALIGNED.splitlines(), 1, ["HWAM", "CWAM", "SHAD", "ADAT"])
    assert x["SHAD"] == a.text.at[1, "SHAD"] == "1. 587.2"  # the tail of CWAM's '5091.' too
    assert math.isnan(a.value(1, "SHAD"))  # not a number for the engine either
    assert a.value(1, "HWAM") == 1491.0
    # ADAT's text takes the '2' of SHAD's '587.2': '2   16', which F12.0 (blanks ignored) reads 216
    assert x["ADAT"] == a.text.at[1, "ADAT"] == "2   16" and a.value(1, "ADAT") == 216.0


def _filea_table(ncols: int, rows: dict[int, list[int]], *, sep: str = " ") -> str:
    """A FILEA on the six-column grid: ``@TRNO`` then ``X01`` ... (``sep`` follows ``@TRNO``)."""
    names = [f"X{k:02d}" for k in range(1, ncols)]
    head = "@TRNO" + sep + f"{names[0]:>6}" + "".join(f"{n:>6}" for n in names[1:])
    body = [f"{t:>6}" + "".join(f"{v:>6}" for v in vals) for t, vals in rows.items()]
    return "\n".join(["*EXP. DATA (A): TEST0401MZ GRID", head, *body, ""])


def test_filea_pathological_layouts_keep_the_documented_rule(tmp_path: Path) -> None:
    """The cases of the module docstring ("Where READA_Y4K differs"): the reader keeps the rule."""
    # more than 1000 data lines before a treatment's row: the engine keeps the 1000th line, the
    # reader finds the row
    many = _filea_table(2, {t: [10 * t] for t in range(1, 1006)})
    a = read_filea(_write(tmp_path, "TEST0401.MZA", many))
    assert len(a.table) == 1005 and a.value(1005, "X01") == 10050.0 and a.value(1000, "X01") == 10000.0
    # more than 40 header columns: the first 40 are read, the rest listed
    wide = _filea_table(41, {1: list(range(1, 41)), 2: list(range(101, 141))})
    b = read_filea(_write(tmp_path, "TEST0402.MZA", wide))
    assert len(b.table.columns) == 39 and b.ignored_columns == ("X40",)
    assert b.value(2, "X39") == 139.0 and math.isnan(b.value(2, "X40"))
    _write(tmp_path, "TEST0402.MZX", "*EXP.DETAILS: TEST0402MZ\n")
    assert any("MAXCOLN = 40" in n for n in read_observed(tmp_path / "TEST0402.MZX").notes)
    # a tab in the header line: read as a blank (the engine would find no TRNO column)
    tab = _filea_table(2, {1: [3000]}, sep="\t")
    c = read_filea(_write(tmp_path, "TEST0403.MZA", tab))
    assert "\t" in tab and list(c.table.index) == [1] and c.value(1, "X01") == 3000.0
    # duplicate TRNO columns: the last one that reads as an integer decides (the engine matches a row
    # on any of them)
    dup = "@TRNO  HWAM  TRNO  CWAM\n     1  3000     7 5000\n"
    d = read_filea(_write(tmp_path, "TEST0404.MZA", dup))
    assert list(d.table.index) == [7] and d.value(7, "HWAM") == 3000.0 and d.value(7, "CWAM") == 5000.0


def test_filea_dates(tmp_path: Path) -> None:
    a = read_filea(_write(tmp_path, "TEST0101.MZA", FILEA))
    # day of year after the simulation start day: same year; YYDDD: the cross-over rule
    assert a.date(1, "ADAT", 1982056) == date(1982, 5, 12)
    assert a.date(2, "MDAT", 1982056) == date(1982, 7, 5)
    # a day of year not after the start day: the next year (READA_Dates)
    assert a.date(1, "ADAT", 1982200) == date(1983, 5, 12)
    assert a.date(3, "ADAT", 1982056) is None
    # FirstWeatherDate (Y4K_DOY, DATES.for:124-136): the first such date on or after it
    assert a.date(2, "MDAT", 1982056, first_weather=1981001) == date(1982, 7, 5)
    assert a.date(2, "MDAT", 1982056, first_weather=date(2080, 1, 1)) == date(2082, 7, 5)
    assert y4k_date(82186, 1990001) == date(2082, 7, 5)  # 1982186 is before 1990001
    with pytest.raises(ValueError, match="99 years"):
        y4k_date(82186, 1982187)  # 2082186 > 1982187 + 99000
    with pytest.raises(ValueError, match="99 years"):
        y4k_date(82186, 1983001)  # 2082186 > 1983001 + 99000
    assert y4k_date(1982186) == date(1982, 7, 5) and y4k_date(36001) == date(1936, 1, 1)


@pytest.mark.parametrize("fw", [-99, 0, "-99", "0"])
def test_first_weather_date_counts_only_when_positive(fw: int | str, tmp_path: Path) -> None:
    """Y4K_DOY uses FirstWeatherDate only when it is positive (``DATES.for:124``); DSSAT's value for
    "none" is -99 (``ModuleDefs.for:93``): the cross-over rule then, as without it."""
    assert y4k_date(82186, fw) == y4k_date(82186) == date(1982, 7, 5)
    assert y4k_date(36001, fw) == date(1936, 1, 1) and y4k_date(1982186, fw) == date(1982, 7, 5)
    assert y4k_date(35365, fw) == date(2035, 12, 31)
    assert reada_date(82186.0, 1982056, fw) == date(1982, 7, 5)
    assert reada_date(186.0, 1982056, fw) == date(1982, 7, 5)
    assert y4k_date(82186, 1990001) == date(2082, 7, 5)  # a positive one still anchors the date
    t = read_filet(_write(tmp_path, "TEST0101.MZT", FILET), first_weather=fw)
    assert t.first_weather is None  # the dates were converted by the cross-over rule
    assert list(t.series(1, "LAID")["YRDOY"]) == [1982057, 1982089, 1982103]


def test_filea_date_below_one_is_missing(tmp_path: Path) -> None:
    """``READA_Dates`` truncates the value: 0 < value < 1 is 0, no date, as a -99."""
    a = read_filea(_write(tmp_path, "TEST0501.MZA", FILEA_FRACTIONAL))
    assert a.value(1, "ADAT") == 0.5 and a.date(1, "ADAT", 1982056) is None
    assert a.date(2, "ADAT", 1982056) == date(1982, 5, 13)
    assert reada_date(0.5, 1982056) is None and reada_date(0.999, 1982056) is None
    assert reada_date(1.0, 1982056) == date(1983, 1, 1)  # day 1 is before the start day: next year
    _write(tmp_path, "TEST0501.MZX", "*EXP.DETAILS: TEST0501MZ\n")
    o = read_observed(tmp_path / "TEST0501.MZX")
    yr = np.stack([_days(1982056, 140)] * 2)
    # ADAT is observed for TRNO 2 only: a date target needs every treatment of the group, so it is
    # left out with a note (and no assertion fires on TRNO 1's 0.5); MDAT is observed for both
    ot = observation_targets(o, [1, 2], yr, codes={"ADAT", "MDAT"})
    assert {t.name for t in ot.targets} == {"MDAT"}
    assert any(s.startswith("ADAT: not observed") and "[1]" in s for s in ot.notes)
    # observed in no treatment of the group: absent without a note, as a -99
    alone = observation_targets(o, [1], yr[:1], codes={"ADAT"})
    assert alone.targets == () and alone.notes == ()


def test_observed_date_is_reada_date() -> None:
    """``outputs.observed_date`` is ``reada_date``: the same dates, and an invalid day of year raises
    instead of rolling over into the next year."""
    for code, start in [
        (132, 1982056),
        (76, 1984357),
        (357, 1984357),
        (85077, 1984357),
        ("1985077", 1984357),
        (185.0, date(1982, 2, 25)),
        ("132", "1982056"),
        (366, 1984056),
    ]:
        assert observed_date(code, start) == reada_date(float(code), start), (code, start)
    assert observed_date("35120", "1930001", first_weather="1930001") == date(1935, 4, 30)
    assert observed_date("35120", "1930001") == date(2035, 4, 30)
    assert observed_date(35120, "1930001", first_weather=-99) == date(2035, 4, 30)
    # a YYDDD simulation start is converted like any YYDDD code (Y4K_DOY), never read as a year 0082
    for start in (82056, "82056", 1982056, date(1982, 2, 25)):
        assert observed_date(132, start) == reada_date(132.0, start) == date(1982, 5, 12), start
        assert observed_date(40, start) == date(1983, 2, 9), start
    assert observed_date(132, 82056, first_weather=2079001) == date(2082, 5, 12)
    assert observed_date(132, 82056, first_weather=-99) == date(1982, 5, 12)
    assert reada_date(186.0, 36001) == date(1936, 7, 4)  # 36001 is 1936 by the cross-over rule
    assert observed_date(132, 82056, first_weather=2090001) == date(2182, 5, 12)  # the first 82 after it
    for code in (366, 400, 82366, 1982000):  # day 366 of 1982, day 400, 1982 day 366, day 0
        with pytest.raises(ValueError, match="day of year"):
            observed_date(code, 1982056)
    for code in (0, 0.5, -99, "-99", math.nan):
        with pytest.raises(ValueError, match="not an observed date"):
            observed_date(code, 1982056)


def test_day_366_of_a_non_leap_year_is_an_error() -> None:
    assert y4k_date(84366) == date(1984, 12, 31)
    with pytest.raises(ValueError, match="day of year 366"):
        y4k_date(82366)
    with pytest.raises(ValueError, match="day of year 366"):
        y4k_date(1982366)
    with pytest.raises(ValueError, match="day of year 366"):
        reada_date(366.0, 1982056)
    assert reada_date(366.0, 1984056) == date(1984, 12, 31)
    with pytest.raises(ValueError, match="day of year 0"):
        y4k_date(1982000)


def test_filet_invalid_dates_listed(tmp_path: Path) -> None:
    text = (
        "@TRNO   DATE  LAID\n     1 82365  1.00\n     1 82366  2.00\n     2 80000  3.00\n     2 84366  4.00\n"
    )
    _write(tmp_path, "TEST0301.MZX", "*EXP.DETAILS: TEST0301MZ\n")
    _write(tmp_path, "TEST0301.MZT", text)
    t = read_filet(tmp_path / "TEST0301.MZT")
    assert list(t.table["YRDOY"]) == [1982365, 1984366]
    assert [(tr, c) for tr, c, _ in t.invalid_dates] == [(1, 82366), (2, 80000)]
    o = read_observed(tmp_path / "TEST0301.MZX")
    assert any("invalid dates left out" in n for n in o.notes)


# --------------------------------------------------------------------------- FILET


def test_filet_tidy(tmp_path: Path) -> None:
    t = read_filet(_write(tmp_path, "TEST0101.MZT", FILET))
    assert t.n_tables == 2
    assert t.codes == ("CWAD", "LAID", "SW1D", "GWAD")
    assert list(t.table.columns) == ["TRNO", "DATE", "YRDOY", "DATECODE", "variable", "value"]
    assert t.first_weather is None
    assert np.isfinite(t.table["value"]).all()  # -99 never appears
    lai = t.series(1, "LAID")
    assert list(lai["YRDOY"]) == [1982057, 1982089, 1982103]
    np.testing.assert_array_equal(lai["value"], [0.0, 0.17, 0.56])
    assert lai["DATE"].iloc[1] == date(1982, 3, 30)
    assert list(t.series(1, "SW1D")["value"]) == [0.2, 0.18]
    assert list(t.series(2, "CWAD")["value"]) == [96.0, 100.0]  # a repeated date keeps both rows
    g = t.series(1, "GWAD")  # YYYYDDD dates
    assert list(g["YRDOY"]) == [1982145] and list(g["value"]) == [50.0]
    assert t.series(2, "GWAD").empty and t.series(7, "LAID").empty
    # FirstWeatherDate moves YYDDD dates, not YYYYDDD ones
    t2 = read_filet(_write(tmp_path, "TEST0102.MZT", FILET), first_weather=2080001)
    assert list(t2.series(1, "LAID")["YRDOY"]) == [2082057, 2082089, 2082103]
    assert list(t2.series(1, "GWAD")["YRDOY"]) == [1982145] and t2.first_weather == 2080001


def test_find_and_read_observed(tmp_path: Path) -> None:
    _write(tmp_path, "TEST0101.MZX", "*EXP.DETAILS: TEST0101MZ\n")
    _write(tmp_path, "test0101.mza", FILEA)
    _write(tmp_path, "TEST0101.MZT", FILET)
    fa, ft = find_observed_files(tmp_path / "TEST0101.MZX")
    assert fa is not None and fa.name == "test0101.mza" and ft is not None and ft.name == "TEST0101.MZT"
    o = read_observed(tmp_path / "TEST0101.MZX")
    assert o.experiment == "TEST0101" and o.trnos == (1, 2, 3, 4, 5)
    c = o.counts()
    assert c["A:HWAM"] == 2 and c["A:CWAM"] == 3 and c["T:CWAD"] == 5 and c["T:GWAD"] == 1
    assert any("READA_Y4K (READS.for:1176-1323) reads the rows of the later" in n for n in o.notes)
    assert any("aliases applied: {'HWAH': 'HWAM'}" in n for n in o.notes)
    assert any("not codes (not used): ['-99']" in n for n in o.notes)
    assert find_observed_files(tmp_path / "NONE0101.MZX") == (None, None)


# --------------------------------------------------------------------------- targets


def _days(start: int, n: int) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.int64)


def _observed(tmp_path: Path):
    _write(tmp_path, "TEST0101.MZX", "*EXP.DETAILS: TEST0101MZ\n")
    _write(tmp_path, "TEST0101.MZA", FILEA)
    _write(tmp_path, "TEST0101.MZT", FILET)
    return read_observed(tmp_path / "TEST0101.MZX")


def test_observation_targets_arrays(tmp_path: Path) -> None:
    o = _observed(tmp_path)
    n = 140  # 1982056 .. 1982195: every observation date inside
    yr = np.stack([_days(1982056, n)] * 3)
    ot = observation_targets(o, [1, 2, 5], yr)
    by = {t.name: t for t in ot.targets}
    assert set(by) == {"HWAM", "CWAM", "ADAT", "MDAT", "CWAD", "LAID", "GWAD"}
    # HWAM: READA_Y4K's last HWAM column; missing for TRNO 2 there (although its HWAH has 3130):
    # a last-day series target
    assert by["HWAM"].kind == "series" and by["HWAM"].output == "gwad"
    assert by["HWAM"].mask[-1].tolist() == [True, False, True]
    assert ot.observed["HWAM"][-1, 0] == 3000.0 and ot.observed["HWAM"][-1, 2] == 4321.0
    assert by["HWAM"].scale == pytest.approx((3000.0 + 4321.0) / 2)
    assert any(s.startswith("HWAM: missing for TRNO [2]") for s in ot.notes)
    assert "HWAH" not in ot.unsupported
    # dates: indices on the simulated days
    assert by["ADAT"].kind == "date" and by["ADAT"].code == DSSAT_TARGETS["ADAT"].code_stage
    np.testing.assert_array_equal(ot.observed["ADAT"], [132 - 56, 133 - 56, 140 - 56])
    np.testing.assert_array_equal(ot.observed["MDAT"], [185 - 56, 186 - 56, 190 - 56])
    assert by["ADAT"].scale == 1.0
    # series: mask and values on the observation days, repeated date averaged
    lai = by["LAID"]
    assert lai.mask.shape == (n, 3) and int(lai.mask.sum()) == 4
    assert ot.observed["LAID"][89 - 56, 0] == 0.17 and ot.observed["LAID"][89 - 56, 1] == 0.19
    assert ot.observed["CWAD"][89 - 56, 1] == 98.0
    assert any("CWAD: 1 dates observed more than once" in s for s in ot.notes)
    # soil water and the text column are observed but not mapped
    assert "SW1D" in ot.unsupported and "soil water" in ot.unsupported["SW1D"]
    assert "TNAM" in ot.unsupported and "SW1D" not in by


def test_observation_targets_first_weather(tmp_path: Path) -> None:
    o = _observed(tmp_path)
    n = 140
    yr = np.stack([_days(2082056, n)] * 2)  # the same season a century later
    ot = observation_targets(o, [1, 5], yr, first_weather=2080001, codes={"MDAT", "LAID"})
    np.testing.assert_array_equal(ot.observed["MDAT"], [185 - 56, 190 - 56])
    assert int(ot.targets[[t.name for t in ot.targets].index("LAID")].mask.sum()) == 3
    without = observation_targets(o, [1, 5], yr, codes={"MDAT", "LAID"})  # 1982 dates: all outside
    assert {t.name for t in without.targets} == {"MDAT"}  # MDAT: day of year, counted from the start
    with pytest.raises(ValueError, match="first_weather"):
        observation_targets(o, [1, 5], yr, first_weather=[2080001])


def _assert_same_targets(a, b) -> None:
    assert [t.name for t in a.targets] == [t.name for t in b.targets]
    for ta, tb in zip(a.targets, b.targets, strict=True):
        np.testing.assert_array_equal(a.observed[ta.name], b.observed[tb.name])
        assert (ta.mask is None) == (tb.mask is None)
        if ta.mask is not None:
            np.testing.assert_array_equal(ta.mask, tb.mask)


def test_observation_targets_first_weather_one_value_types(tmp_path: Path) -> None:
    """A single FirstWeatherDate may be an int, a numpy integer, a YYYYDDD text or a date (not a
    sequence to iterate); a sequence gives one per treatment; -99 / 0 are the cross-over rule."""
    o = _observed(tmp_path)
    codes = {"MDAT", "LAID"}
    yr = np.stack([_days(2082056, 140)] * 2)  # the same season a century later
    ref = observation_targets(o, [1, 5], yr, first_weather=2080001, codes=codes)
    assert {t.name for t in ref.targets} == {"MDAT", "LAID"}
    for fw in (
        np.int64(2080001),
        np.int32(2080001),
        "2080001",
        date(2080, 1, 1),
        [np.int64(2080001)] * 2,
        ["2080001", date(2080, 1, 1)],
        np.array([2080001, 2080001]),
        # floats (a pandas column with gaps is float64), 0-d arrays and numpy dates
        2080001.0,
        np.float64(2080001.0),
        np.float32(2080001.0),
        np.array(2080001),
        np.array(2080001.0),
        np.datetime64("2080-01-01"),
        np.datetime64("2080-01-01T00:00:00.000000000"),
        np.array(np.datetime64("2080-01-01", "ns")),
        [2080001.0, np.float64(2080001.0)],
        np.array([2080001.0, 2080001.0]),
        pd.Series([2080001.0, 2080001.0]),
        np.array(["2080-01-01", "2080-01-01"], dtype="datetime64[D]"),
        [pd.Timestamp("2080-01-01"), np.datetime64("2080-01-01")],
        (f for f in (2080001.0, np.int64(2080001))),
    ):
        _assert_same_targets(observation_targets(o, [1, 5], yr, first_weather=fw, codes=codes), ref)
    with pytest.raises(ValueError, match="first_weather: 1 values for 2 treatments"):
        observation_targets(o, [1, 5], yr, first_weather=[np.int64(2080001)], codes=codes)
    yr82 = np.stack([_days(1982056, 140)] * 2)
    none = observation_targets(o, [1, 5], yr82, codes=codes)
    for fw in (
        -99,
        0,
        np.int64(-99),
        "-99",
        -99.0,
        np.float64(-99.0),
        np.array(-99.0),
        # no date: NaN, NaT (a gap of a pandas column)
        math.nan,
        np.float64("nan"),
        np.array(math.nan),
        np.datetime64("NaT"),
        pd.NaT,
        [math.nan, np.float64("nan")],
        pd.Series([math.nan, math.nan]),
        [-99.0, pd.NaT],
    ):
        _assert_same_targets(observation_targets(o, [1, 5], yr82, first_weather=fw, codes=codes), none)
    # a YYDDD simulation start is the same day as its YYYYDDD
    _assert_same_targets(observation_targets(o, [1, 5], yr82, sim_start=[82056, 82056], codes=codes), none)
    with pytest.raises(ValueError, match="not a YYYYDDD date"):
        observation_targets(o, [1, 5], yr, first_weather=2080001.5, codes=codes)
    with pytest.raises(ValueError, match="a date is a scalar"):
        observation_targets(o, [1, 5], yr, first_weather=np.array([[2080001], [2080001]]), codes=codes)
    # a FILET read with the "none" value (-99) does not fix its dates for a later FirstWeatherDate
    o2 = read_observed(tmp_path / "TEST0101.MZX", first_weather=-99)
    assert o2.filet is not None and o2.filet.first_weather is None
    _assert_same_targets(observation_targets(o2, [1, 5], yr, first_weather=2080001, codes=codes), ref)


def test_observation_targets_drop_and_options(tmp_path: Path) -> None:
    o = _observed(tmp_path)
    yr = np.stack([_days(1982056, 40), _days(1982056, 40)])  # ends 1982095: no maturity, LAID 103 out
    ot = observation_targets(
        o,
        [1, 2],
        yr,
        codes={"MDAT", "LAID", "HWAM"},
        scales={"HWAM": 100.0},
        weights={"LAID": 2.0},
        available_outputs={"lai", "istage"},
    )
    by = {t.name: t for t in ot.targets}
    assert set(by) == {"LAID"} and by["LAID"].weight == 2.0
    assert any(s.startswith("MDAT: not observed (or outside the simulated days)") for s in ot.notes)
    assert any("HWAM: output 'gwad' not produced" in s for s in ot.notes)
    assert any(s.startswith("LAID: 1 observation dates outside") for s in ot.notes)
    with pytest.raises(ValueError, match="yrdoy must be"):
        observation_targets(o, [1, 2, 3], yr)
    # a final target: observed in every treatment of the group
    yr2 = np.stack([_days(1982056, 140)] * 2)
    ot2 = observation_targets(o, [1, 5], yr2, codes={"HWAM", "CWAM"}, scales={"HWAM": 100.0})
    by2 = {t.name: t for t in ot2.targets}
    assert by2["HWAM"].kind == "final" and by2["HWAM"].scale == 100.0
    np.testing.assert_array_equal(ot2.observed["HWAM"], [3000.0, 4321.0])
    np.testing.assert_array_equal(ot2.observed["CWAM"], [5532.0, 5678.0])


def test_group_loss_zero_on_observed_values(tmp_path: Path) -> None:
    """A simulator that reproduces the observations exactly gives a zero loss, a perturbed one not."""
    o = _observed(tmp_path)
    n = 140
    yr = np.stack([_days(1982056, n), _days(1982056, n)])
    ot = observation_targets(o, [1, 5], yr, codes={"HWAM", "ADAT", "LAID"})
    lai = jnp.asarray(ot.observed["LAID"])
    gw = jnp.zeros((n, 2)).at[-1].set(jnp.asarray(ot.observed["HWAM"]))
    t = jnp.arange(n)[:, None]
    stage = jnp.where(t >= jnp.asarray(ot.observed["ADAT"])[None, :], obs_mod.ISTAGE_SILKING, 3)

    def simulate(theta):
        return {"lai": lai * theta[0], "gwad": gw, "istage": stage}

    g = ot.group("test", simulate)
    assert float(g.target_losses(jnp.asarray([1.0]))["LAID"]) == 0.0
    total = sum(float(v) for v in g.target_losses(jnp.asarray([1.0])).values())
    assert total == 0.0
    assert float(g.target_losses(jnp.asarray([1.1]))["LAID"]) > 0.0


def test_stage_codes_match_the_model() -> None:
    """The stage codes of calib.observations are the crop model's (defined there to keep the module
    free of the crop-model import)."""
    from agrijax.calib.ceres import ISTAGE_MATURITY_OUT, ISTAGE_SILKING_OUT
    from agrijax.processes.crop.ceres_maize import constants as k

    assert obs_mod.ISTAGE_SILKING == ISTAGE_SILKING_OUT == k.ISTAGE_END_LEAF_GROWTH
    assert obs_mod.ISTAGE_AFTER_MATURITY == ISTAGE_MATURITY_OUT == k.ISTAGE_AFTER_MATURITY


def test_observations_module_does_not_load_the_crop_model() -> None:
    code = (
        "import sys, agrijax.calib.observations; "
        "bad = [m for m in sys.modules if m.startswith(('agrijax.processes', 'agrijax.calib.ceres'))]; "
        "assert not bad, bad"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0, r.stderr[-2000:]


# --------------------------------------------------------------------------- DSSAT example tree

_EXAMPLES = DSSAT_ENGINE / "example_data"
_OBS_RE = re.compile(r"^\.[A-Z]{2}[AT]$")


def _tree() -> Path:
    if not (_EXAMPLES / "Maize").is_dir():
        pytest.skip(f"{_EXAMPLES / 'Maize'} not found (DSSAT v4.8.6.0 example tree)")
    return _EXAMPLES


@pytest.mark.allow_skip(reason="needs the DSSAT example tree ($AGRI_JAX_DSSAT); CI has none")
def test_every_maize_observed_file(capsys: pytest.CaptureFixture[str]) -> None:
    root = _tree()
    fa = sorted(p for p in (root / "Maize").iterdir() if p.suffix.upper() == ".MZA")
    ft = sorted(p for p in (root / "Maize").iterdir() if p.suffix.upper() == ".MZT")
    assert len(fa) == 10 and len(ft) == 10, ([p.name for p in fa], [p.name for p in ft])
    lines = []
    n_a = n_t = 0
    mapped: dict[str, int] = {}
    for x in sorted((root / "Maize").glob("*.MZX")):
        o = read_observed(x)
        c = o.counts()
        n_a += sum(v for k, v in c.items() if k.startswith("A:"))
        n_t += sum(v for k, v in c.items() if k.startswith("T:"))
        for k, v in c.items():
            code = k[2:]
            if code in DSSAT_TARGETS and DSSAT_TARGETS[code].file == k[0]:
                mapped[code] = mapped.get(code, 0) + v
        lines.append(f"{o.experiment}: {len(o.trnos)} treatments, {c}")
    # UFGA8201 by hand from the file
    u = read_observed(root / "Maize" / "UFGA8201.MZX")
    assert u.filea is not None and u.filet is not None
    assert u.trnos == (1, 2, 3, 4, 5, 6)
    assert u.filea.value(4, "HWAM") == 11881.0 and u.filea.value(3, "CWAM") == 14581.0
    assert u.filea.value(1, "ADAT") == 132.0 and u.filea.value(6, "MDAT") == 185.0
    assert u.counts()["T:LAID"] == 78 and u.counts()["T:SW1D"] == 12
    assert u.filet.series(1, "LAID")["value"].iloc[-1] == 0.18
    with capsys.disabled():
        print(
            f"\nmaize observed files: {len(fa)} FILEA, {len(ft)} FILET; {n_a} season-end and {n_t} "
            f"time-series observations; mapped codes: {mapped}"
        )
        for ln in lines:
            print("  " + ln)


def _filea_files(root: Path, maize_only: bool = False) -> list[Path]:
    base = root / "Maize" if maize_only else root
    return sorted(
        p for p in base.rglob("*") if p.is_file() and re.fullmatch(r"\.[A-Z]{2}A", p.suffix.upper())
    )


@pytest.mark.allow_skip(reason="needs the DSSAT example tree ($AGRI_JAX_DSSAT); CI has none")
def test_every_example_filea_equals_reada_y4k(capsys: pytest.CaptureFixture[str]) -> None:
    """Every FILEA of every crop: the reader's text equals the READA_Y4K restatement for every
    treatment and label (0 differences), GHNY9801.PNA SHAD included."""
    root = _tree()
    files = _filea_files(root)
    n_val = 0
    diffs = []
    for f in files:
        a = read_filea(f)
        lines = read_lines(f)
        labels = list(a.text.columns)
        for trno in a.text.index:
            x = _reada_y4k(lines, int(trno), labels)
            for code in labels:
                n_val += 1
                if a.text.at[trno, code] != x.get(code, ""):
                    diffs.append((f.name, int(trno), code, a.text.at[trno, code], x.get(code)))
    assert diffs == []
    assert len(files) >= 200 and n_val > 10000
    # GHNY9801.PNA: '@TRNO   PWAM  HWAM CWAM   SHAD' (CWAM one column left): the engine's SHAD text
    # carries CWAM's trailing point and is no number
    g = read_filea(root / "Peanut" / "GHNY9801.PNA")
    assert g.text.at[1, "SHAD"] == ". 587.2" and math.isnan(g.value(1, "SHAD"))
    assert g.value(1, "CWAM") == 5091.0 and g.value(1, "HWAM") == 1491.0
    with capsys.disabled():
        print(f"\nREADA_Y4K restatement: {len(files)} FILEA, {n_val} values, {len(diffs)} differences")


@pytest.mark.allow_skip(reason="needs the DSSAT example tree ($AGRI_JAX_DSSAT); CI has none")
def test_filea_against_header_reader(capsys: pytest.CaptureFixture[str]) -> None:
    """The header-position reader of the *.OUT files (a second implementation) gives the same
    numbers on every maize FILEA (0 differences); on the other crops the differences are listed."""
    root = _tree()
    by_file: dict[str, list[tuple[str, int, float, float]]] = {}
    n_maize = 0
    for f in _filea_files(root):
        a = read_filea(f)
        ref = read_out(f)
        if ref.empty or "TRNO" not in ref.columns:
            continue
        first = ref.drop_duplicates("TRNO").set_index("TRNO")
        back = {v: k for k, v in a.aliases.items()}
        for code in a.codes:
            col = code if code in first.columns else back.get(code)
            for trno in a.table.index:
                got = a.value(int(trno), code)
                try:
                    want = float(first.at[trno, col]) if col in first.columns else math.nan
                except (KeyError, TypeError, ValueError):
                    want = math.nan
                want = math.nan if want == -99.0 else want
                if not _same(got, want):
                    by_file.setdefault(f.name, []).append((code, int(trno), got, want))
                elif f.suffix.upper() == ".MZA" and math.isfinite(got):
                    n_maize += 1
    maize = {k: v for k, v in by_file.items() if k.upper().endswith(".MZA")}
    assert maize == {} and n_maize > 400
    with capsys.disabled():
        print(f"\nFILEA vs the header reader: maize 0 differences ({n_maize} values); other crops:")
        for k, v in sorted(by_file.items()):
            print(f"  {k}: {len(v)} ({sorted({c for c, *_ in v})}) e.g. {v[0]}")


@pytest.mark.allow_skip(reason="needs the DSSAT example tree ($AGRI_JAX_DSSAT); CI has none")
def test_every_example_observed_file_reads(capsys: pytest.CaptureFixture[str]) -> None:
    """Every FILEA / FILET of every crop in the example tree reads without error."""
    root = _tree()
    files = sorted(p for p in root.rglob("*") if p.is_file() and _OBS_RE.match(p.suffix.upper()))
    n_a = n_t = n_obs = 0
    aliased = []
    invalid: list[tuple[str, int, int]] = []
    for f in files:
        if f.suffix.upper().endswith("A"):
            a = read_filea(f)
            n_a += 1
            n_obs += int(a.table[list(a.codes)].notna().to_numpy().sum())
            if a.aliases:
                aliased.append(f.name)
        else:
            t = read_filet(f)
            n_t += 1
            n_obs += len(t.table)
            invalid += [(f.name, tr, c) for tr, c, _ in t.invalid_dates]
    assert n_a >= 200 and n_t >= 230
    with capsys.disabled():
        print(
            f"\nexample tree: {n_a} FILEA + {n_t} FILET read, {n_obs} observations; "
            f"{len(aliased)} FILEA with READA_Y4K aliases; FILET rows with invalid dates: {invalid}"
        )
