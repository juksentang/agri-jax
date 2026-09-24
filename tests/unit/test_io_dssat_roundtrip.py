"""Round trips and alignment regressions of the DSSAT readers/writers.

The first part needs no data: crafted ``Summary.OUT`` / ``.WTH`` snippets, and the public BSD-3
``MZCER048.*`` fixtures of ``tests/fixtures/dssat``. The references are independent of the code
under test wherever possible: DSSAT's own reading rule (``PARSE_HEADERS`` spans + list-directed
read, re-implemented here from ``READS.for`` without using the package helper), byte identity with
the source file, or a hand edit of the source text.

The second part (``allow_skip``, each test checks for its own files) runs the same round trips over
every file of the DSSAT v4.8.6.0 example tree when it is present: all 1052 ``example_data/**/*.WTH``,
all 41 ``example_data/Soil/*.SOL`` (3826 profiles) and every ``source/Data/Genotype`` CUL/ECO/SPE.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrijax.io.dssat import (
    observed_date,
    read_cul,
    read_eco,
    read_sol,
    read_spe,
    read_summary,
    read_wth,
    write_cul,
    write_eco,
    write_sol,
    write_spe,
    write_wth,
)
from agrijax.io.dssat._fixed import dssat_header_spans, list_directed_float, read_lines

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dssat"
DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()


# --------------------------------------------------------------------------- DSSAT reading rule


def _dssat_rule(header: str, line: str) -> list[float]:
    """Values DSSAT 4.8 reads from ``line`` under ``header`` (fields 2..n): re-implemented from
    ``PARSE_HEADERS`` (``READS.for``) and a list-directed ``READ``, independently of ``_fixed``."""
    length = len(header.rstrip())
    ends, starts = [], [0]
    prev_blank = True
    for i in range(1, length):
        if header[i] == " " and not prev_blank:
            ends.append(i)
            starts.append(i + 1)
        prev_blank = header[i] == " "
    ends.append(length)
    out = []
    for a, b in list(zip(starts, ends, strict=True))[1:]:
        item = line[a:b].split()
        try:
            v = float(item[0]) if item else math.nan
        except ValueError:
            v = math.nan
        out.append(math.nan if -99.5 < v <= -99.0 else v)
    return out


def test_dssat_header_spans_match_the_reference_rule() -> None:
    hdr = "@DATE  SRAD  TMAX  TMIN  RAIN  DEWP  WIND   PAR"
    spans = dssat_header_spans(hdr)
    assert [s[0] for s in spans] == ["DATE", "SRAD", "TMAX", "TMIN", "RAIN", "DEWP", "WIND", "PAR"]
    # field i+1 starts two columns after token i ends: the column in between is in no field
    assert spans[1][1:] == (6, 11) and spans[2][1:] == (12, 17) and spans[-1][2] == len(hdr)
    line = "95170  16.0E 32.6  22.5   6.4 -99   345.6  12.1   trailing note"
    ref = _dssat_rule(hdr, line)
    mine = [list_directed_float(line[a:b]) for _, a, b in spans[1:]]
    mine = [math.nan if -99.5 < v <= -99.0 else v for v in mine]
    np.testing.assert_array_equal(mine, ref)
    assert ref[:4] == [16.0, 32.6, 22.5, 6.4] and math.isnan(ref[4])


def test_list_directed_float() -> None:
    assert list_directed_float("  12.5  ") == 12.5
    assert list_directed_float(" 1.5d2") == 150.0
    assert list_directed_float("3,4") == 3.0
    assert math.isnan(list_directed_float("   "))
    assert math.isnan(list_directed_float("E 32"))


def test_read_lines_stops_at_dos_eof(tmp_path: Path) -> None:
    p = tmp_path / "x.WTH"
    p.write_bytes(b"*WEATHER\r\n@DATE  SRAD\r\n82001   1.0\r\n\x1a")
    assert read_lines(p) == ["*WEATHER", "@DATE  SRAD", "82001   1.0"]


# --------------------------------------------------------------------------- Summary.OUT TNAM alignment

# real v4.8.6.0 Summary.OUT header (first 30 columns) and the UFGA8201 treatment-1 row
_SUM_HEAD = (
    "@   RUNNO   TRNO R# O# P# CR MODEL... EXNAME.. TNAM..................... FNAM.... WSTA.... "
    "WYEAR SOIL_ID...             XLAT            LONG      ELEV    SDAT    PDAT    EDAT    ADAT"
    "    MDAT    HDAT   HYEAR  DWAP    CWAM    HWAM    HWAH    BWAH  PWAM"
)
_SUM_ROW = (
    "        {run:>1}      {trno:>1}  1  0  1 MZ MZCER048 UFGA8201 {tnam:<25} UFGA0002 UFGA8201  1982 "
    "IBMZ910014         -82.3689          29.638        40 1982056 1982057 1982068 1982133 1982185"
    " 1982185    1982   -99    6556    {hwam:>4}    2293       0  2624"
)


def test_summary_tnam_with_spaces_keeps_alignment(tmp_path: Path) -> None:
    """TNAM (A25) holds blanks, digits, commas and may fill all 25 columns; the columns after it
    must not shift (a whitespace split would put 'LOW' into FNAM)."""
    tnams = [
        "RAINFED LOW NITROGEN",  # blanks
        "N 0 KG/HA, 1 2 3 -99 IRR",  # numbers and a missing-value look-alike inside the name
        "IRRIGATED,HIGH NITROGEN12",  # exactly 25 characters, touching the next column's blank
        "A",
    ]
    body = "\n".join(
        _SUM_ROW.format(run=i + 1, trno=i + 1, tnam=t, hwam=1000 + i) for i, t in enumerate(tnams)
    )
    p = tmp_path / "Summary.OUT"
    p.write_text("*SUMMARY : UFGA8201MZ NIT X IRR\n\n" + _SUM_HEAD + "\n" + body + "\n")
    s = read_summary(p)
    assert list(s["TNAM"]) == tnams
    assert list(s["FNAM"]) == ["UFGA0002"] * 4 and list(s["WSTA"]) == ["UFGA8201"] * 4
    assert list(s["SOIL_ID"]) == ["IBMZ910014"] * 4
    assert list(s["HWAM"]) == [1000, 1001, 1002, 1003] and s["HWAM"].dtype.kind == "i"
    assert list(s["MDAT"]) == [1982185] * 4 and s["XLAT"].iloc[0] == -82.3689
    assert s["DWAP"].isna().all() and list(s["PWAM"]) == [2624] * 4


# --------------------------------------------------------------------------- dates


def test_observed_date_follows_reada_dates() -> None:
    # DOY after the simulation start day -> same year, otherwise the next one (READA_Dates)
    assert observed_date(132, 1982056) == pd.Timestamp("1982-05-12").date()
    assert observed_date(76, 1984357) == pd.Timestamp("1985-03-17").date()
    assert observed_date(357, 1984357) == pd.Timestamp("1985-12-23").date()
    assert observed_date(85077, 1984357) == pd.Timestamp("1985-03-18").date()
    assert observed_date("1985077", 1984357) == pd.Timestamp("1985-03-18").date()


# --------------------------------------------------------------------------- weather


def _row(date: str, *cells: str) -> str:
    """A ``.WTH`` data line of six-column cells (a flag sits in a cell's first column, the
    column DSSAT reads as part of no field)."""
    assert all(len(c) == 6 for c in cells), cells
    return date + "".join(cells)


_WTH = "\n".join(
    [
        "*WEATHER DATA : test, with flags and notes",
        "",
        "@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT   CO2",
        "  XXXX  -22.7017 -47.642   547  21.5   7.1  2.00   2.0   380",
        "@DATE  SRAD  TMAX  TMIN  RAIN  DEWP  WIND   PAR  RHUM",
        _row("15001", "  16.0", "E 32.6", "  22.5", "   6.4", " 12.35", "  86.4", "  30.3", " 77.45"),
        _row("15002", "  23.8", "E 30.7", "  24.2", "  0.00", "      ", " 104.1", "  -99m", "  81.2")
        + "    NOTE: estimated",
        _row("15003", "  15.3", "  31.7", "  23.2", "  1.50", " -3.05", "  1004", "  33.1", "100.00"),
        "\x1a",
        "",
    ]
)


def test_read_wth_flags_notes_and_eof(tmp_path: Path) -> None:
    p = tmp_path / "XXXX1501.WTH"
    p.write_bytes(_WTH.encode("latin-1"))
    w = read_wth(p)
    assert list(w.columns) == ["date", "srad", "tmax", "tmin", "rain", "dewp", "wind", "par", "rhum"]
    assert len(w) == 3 and w["date"].iloc[-1] == pd.Timestamp("2015-01-03")
    np.testing.assert_array_equal(w["tmax"], [32.6, 30.7, 31.7])  # 'E' flag column skipped
    assert math.isnan(w["par"].iloc[1]) and w["rhum"].iloc[1] == 81.2  # '-99m'; note ignored
    assert math.isnan(w["dewp"].iloc[1]) and w["wind"].iloc[2] == 1004.0
    assert w.attrs["site"]["LAT"] == -22.7017 and w.attrs["site"]["CO2"] == 380
    assert w.attrs["decimals"]["rain"] == 2 and w.attrs["decimals"]["rhum"] == 2
    # DSSAT's own reading of every data line agrees (independent re-implementation)
    lines = _WTH.split("\x1a")[0].splitlines()
    hdr = next(ln for ln in lines if ln.startswith("@DATE"))
    data = [ln for ln in lines if ln[:5].isdigit()]
    ref = np.array([_dssat_rule(hdr, ln) for ln in data])
    np.testing.assert_array_equal(read_wth(p, dssat_spans=True).iloc[:, 1:].to_numpy(), ref)
    # the header-aligned reader agrees everywhere but on a value filling all six columns:
    # DSSAT drops its first character ('100.00' -> 0.0), the aligned reader keeps it
    mine = w.iloc[:, 1:].to_numpy()
    assert mine[2, 7] == 100.0 and ref[2, 7] == 0.0
    ref[2, 7] = 100.0
    np.testing.assert_array_equal(mine, ref)


def _assert_wth_roundtrip(w: pd.DataFrame, path: Path) -> str:
    out = write_wth(w, path)
    for dssat_spans in (False, True):  # ours, and exactly what dscsm048 reads
        w2 = read_wth(out, dssat_spans=dssat_spans)
        pd.testing.assert_frame_equal(w, w2, check_exact=True)
        assert w2.attrs["site"] == w.attrs["site"]
    return out.read_text()


def test_write_wth_keeps_source_decimals_and_site_text(tmp_path: Path) -> None:
    p = tmp_path / "XXXX1501.WTH"
    p.write_bytes(_WTH.encode("latin-1"))
    text = _assert_wth_roundtrip(read_wth(p), tmp_path / "out.WTH")
    assert "  0.00" in text and " 12.35" in text and "100.00" in text  # two-decimal columns
    assert "-22.7017" in text and "  2.00" in text  # site values as written
    assert " 104.1" in text and " 1004.0" in text  # 6-character values widen the column
    # every data row is readable by the reference DSSAT rule with the written header
    lines = text.splitlines()
    hdr = next(ln for ln in lines if ln.startswith("@DATE"))
    ref = np.array([_dssat_rule(hdr, ln) for ln in lines if ln[:5].isdigit()])
    np.testing.assert_array_equal(read_wth(p).iloc[:, 1:].to_numpy(), ref)


def test_write_wth_four_digit_years_need_dollar_weather(tmp_path: Path) -> None:
    """DSSAT 4.8 reads YYYYDDD dates only in a ``$WEATHER`` file (MAKEFILEW.f90) and the SRAD
    span starts two columns after the DATE token, so ``@  DATE`` must end with the date."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2039-12-25", periods=20, freq="D")  # crosses a year end
    w = pd.DataFrame(
        {
            "date": dates.astype("datetime64[s]"),  # the unit read_wth returns
            "srad": np.round(rng.uniform(0, 30, 20), 1),
            "tmax": np.round(rng.uniform(-10, 40, 20), 2),
            "tmin": np.round(rng.uniform(-30, 20, 20), 1),
            "rain": np.round(rng.exponential(5, 20), 1),
        }
    )
    w.attrs["site"] = {"INSI": "XXXX", "LAT": 10.5, "LONG": -3.25, "ELEV": 100}
    text = write_wth(w, tmp_path / "a.WTH", four_digit_year=True).read_text()
    assert text.startswith("$WEATHER") and "\n@  DATE" in text and "\n2039359" in text
    w.attrs["four_digit_year"] = True
    _assert_wth_roundtrip(w, tmp_path / "b.WTH")
    text5 = write_wth(w, tmp_path / "c.WTH", four_digit_year=False).read_text()
    assert text5.startswith("*WEATHER") and "\n@DATE" in text5 and "\n39359" in text5
    # a year YYDDD cannot hold switches to YYYYDDD on its own
    w1 = w.assign(date=pd.date_range("1930-01-01", periods=20, freq="D").astype("datetime64[s]"))
    w1.attrs = {"site": w.attrs["site"]}
    assert write_wth(w1, tmp_path / "d.WTH").read_text().startswith("$WEATHER")


def test_write_wth_random_values_roundtrip(tmp_path: Path) -> None:
    """Values with 0-3 decimals, negatives, NaN and 4-5 digit magnitudes survive both readers."""
    rng = np.random.default_rng(42)
    n = 200
    cols = {}
    for name, dec, lo, hi in (
        ("srad", 1, 0, 35),
        ("tmax", 2, -40, 50),
        ("tmin", 3, -60, 30),
        ("rain", 1, 0, 400),
        ("wind", 1, 0, 12000),
        ("rhum", 2, 0, 100),
    ):
        v = np.round(rng.uniform(lo, hi, n), dec)
        v[rng.random(n) < 0.05] = np.nan
        cols[name] = v
    w = pd.DataFrame(
        {"date": pd.date_range("1981-01-01", periods=n, freq="D").astype("datetime64[s]"), **cols}
    )
    w.attrs["site"] = {"INSI": "RAND", "LAT": -0.001, "LONG": 179.999, "ELEV": -5, "TAV": 20.25}
    _assert_wth_roundtrip(w, tmp_path / "r.WTH")


# --------------------------------------------------------------------------- genotype (fixtures)


@pytest.mark.parametrize("name", ["MZCER048.CUL", "MZCER048.ECO", "MZCER048.SPE"])
def test_genotype_fixture_written_back_byte_for_byte(name: str, tmp_path: Path) -> None:
    src = FIXTURES / name
    rd, wr = {".CUL": (read_cul, write_cul), ".ECO": (read_eco, write_eco), ".SPE": (read_spe, write_spe)}[
        src.suffix
    ]
    obj = rd(src)
    out = wr(obj, tmp_path / name)  # type: ignore[arg-type]
    assert out.read_bytes() == src.read_bytes()


def _hand_edit(src: Path, row_prefix: str, old: str, new: str) -> bytes:
    raw = src.read_bytes()
    nl = b"\r\n" if b"\r\n" in raw else b"\n"
    lines = raw.split(nl)
    k = next(i for i, ln in enumerate(lines) if ln.startswith(row_prefix.encode()))
    assert lines[k].count(old.encode()) == 1
    lines[k] = lines[k].replace(old.encode(), new.encode())
    return nl.join(lines)


def test_write_cul_edit_equals_hand_edit(tmp_path: Path) -> None:
    src = FIXTURES / "MZCER048.CUL"
    c = read_cul(src)
    c.loc["IB0035", "P1"] = 271.5
    c.loc["IB0035", "G3"] = 7.25
    out = write_cul(c, tmp_path / "a.CUL")
    by_hand = _hand_edit(src, "IB0035", " 259.0 1.193 947.1 924.3 8.168", " 271.5 1.193 947.1 924.3  7.25")
    assert out.read_bytes() == by_hand
    pd.testing.assert_frame_equal(read_cul(out), c)


def test_write_eco_keeps_a_decimal_point(tmp_path: Path) -> None:
    """The ecotype is read with F5.1 fields, where ``12`` would mean 1.2: integers get a point."""
    src = FIXTURES / "MZCER048.ECO"
    e = read_eco(src)
    e.loc["IB0001", "DSGFT"] = 180.0
    e.loc["IB0001", "P20"] = 12.0
    out = write_eco(e, tmp_path / "a.ECO")
    line = next(ln for ln in out.read_text().splitlines() if ln.startswith("IB0001"))
    assert "  12. " in line and " 180. " in line
    # numbers stay right-aligned to the end of their source token
    by_hand = _hand_edit(src, "IB0001", "  12.5   4.0   6.0   170.", "   12.   4.0   6.0   180.")
    assert out.read_bytes() == by_hand
    pd.testing.assert_frame_equal(read_eco(out), e)


def test_write_cul_rows_added_dropped_and_overflow(tmp_path: Path) -> None:
    c = read_cul(FIXTURES / "MZCER048.CUL")
    n = len(c)
    new = c.loc[["IB0035"]].rename(index={"IB0035": "ZZ0001"})
    new.loc["ZZ0001", "VRNAME"] = "My cultivar"
    new.loc["ZZ0001", "P1"] = 222.5
    c2 = pd.concat([c.drop(index="PC0001"), new])
    c2.attrs = c.attrs
    out = write_cul(c2, tmp_path / "b.CUL")
    back = read_cul(out)
    assert len(back) == n and "PC0001" not in back.index
    assert back.loc["ZZ0001", "VRNAME"] == "My cultivar" and back.loc["ZZ0001", "P1"] == 222.5
    row = next(ln for ln in out.read_text().splitlines() if ln.startswith("ZZ0001"))
    ref = next(ln for ln in out.read_text().splitlines() if ln.startswith("IB0035"))
    # the new row follows DSSAT's (A6,1X,A16,7X,A6,6F6.0) layout of the template row
    assert row[30:36] == ref[30:36] == "IB0001" and len(row) == len(ref.rstrip())
    c.loc["IB0035", "P1"] = 1.0 / 3.0  # needs more than its six columns: refuse to round
    with pytest.raises(ValueError, match="does not fit"):
        write_cul(c, tmp_path / "c.CUL")


def test_write_spe_edit_equals_hand_edit(tmp_path: Path) -> None:
    src = FIXTURES / "MZCER048.SPE"
    s = read_spe(src)
    s.params["PARSR"] = 0.45
    s.params["PRFTC"] = (6.2, 16.5, 33.0, 45.5)
    out = write_spe(s, tmp_path / "a.SPE")
    raw = _hand_edit(src, "  PARSR", "0.50", "0.45")
    tmp = tmp_path / "hand.SPE"
    tmp.write_bytes(raw)
    raw = _hand_edit(tmp, "  PRFTC", "44.0", "45.5")
    assert out.read_bytes() == raw
    back = read_spe(out)
    assert back["PARSR"] == 0.45 and back["PRFTC"] == (6.2, 16.5, 33.0, 45.5)
    s.params["PRFTC"] = (6.2, 16.5, 33.0)
    with pytest.raises(ValueError, match="3 values for 4 fields"):
        write_spe(s, tmp_path / "b.SPE")


# --------------------------------------------------------------------------- the whole example tree


def _tree(sub: str) -> Path:
    p = DSSAT_ENGINE / sub
    if not p.is_dir():
        pytest.skip(f"{p} not found (DSSAT v4.8.6.0 engine tree)")
    return p


_EXAMPLES = DSSAT_ENGINE / "example_data"
_WTH_FILES = sorted(_EXAMPLES.rglob("*.WTH")) if _EXAMPLES.is_dir() else []
_WTH_SHARDS = sorted({f.name[0].upper() for f in _WTH_FILES})
_SOL_FILES = sorted((_EXAMPLES / "Soil").glob("*.SOL")) if _EXAMPLES.is_dir() else []


@pytest.mark.allow_skip(reason="needs the DSSAT example tree; CI has none")
def test_example_tree_complete() -> None:
    _tree("example_data")
    assert len(_WTH_FILES) >= 1000 and len(_SOL_FILES) >= 40


@pytest.mark.slow
@pytest.mark.allow_skip(reason="needs the DSSAT example tree; CI has none")
@pytest.mark.parametrize("shard", _WTH_SHARDS)
def test_every_example_wth_roundtrips(shard: str, tmp_path: Path) -> None:
    """Every example weather file (sharded by first letter): read -> write -> read is identical
    for our reader and for DSSAT's reading rule."""
    for f in (f for f in _WTH_FILES if f.name[0].upper() == shard):
        w = read_wth(f)
        try:
            _assert_wth_roundtrip(w, tmp_path / "x.WTH")
        except AssertionError as e:
            raise AssertionError(f"{f.name}: {e}") from e


@pytest.mark.slow
@pytest.mark.allow_skip(reason="needs the DSSAT example tree; CI has none")
@pytest.mark.parametrize("sol", _SOL_FILES, ids=lambda p: p.name)
def test_every_example_sol_roundtrips(sol: Path, tmp_path: Path) -> None:
    a = read_sol(sol)
    b = read_sol(write_sol(a, tmp_path / "x.SOL"))
    assert list(a) == list(b) and len(a) >= 1
    for k, p in a.items():
        q = b[k]
        assert (p.source, p.texture, p.description) == (q.source, q.texture, q.description), k
        assert p.depth == q.depth or (math.isnan(p.depth) and math.isnan(q.depth)), k
        assert p.surface == q.surface, k
        for key, v in p.site.items():
            w = q.site.get(key)
            assert v == w or (isinstance(v, float) and math.isnan(v) and math.isnan(w)), (k, key)
        pd.testing.assert_frame_equal(p.layers, q.layers, check_exact=True, obj=f"{sol.name}:{k}")


#: sugarcane ecotype file of key/value lines, not a table (read_eco does not apply)
_NOT_TABLES = {"SCCSP048.ECO"}


@pytest.mark.allow_skip(reason="needs the DSSAT source tree; CI has none")
def test_every_genotype_file_written_back_byte_for_byte(tmp_path: Path) -> None:
    files = sorted(_tree("source/Data/Genotype").iterdir())
    readers = {".CUL": (read_cul, write_cul), ".ECO": (read_eco, write_eco), ".SPE": (read_spe, write_spe)}
    done = 0
    for f in files:
        if f.suffix not in readers or f.name in _NOT_TABLES:
            continue
        rd, wr = readers[f.suffix]
        out = wr(rd(f), tmp_path / f.name)  # type: ignore[arg-type]
        assert out.read_bytes() == f.read_bytes(), f.name
        done += 1
    assert done >= 170
