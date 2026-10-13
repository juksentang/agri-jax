"""Writing a calibrated cultivar into a copy of a DSSAT ``MZCER048.CUL``.

References independent of the writer: DSSAT's own read of a maize cultivar row, re-stated here
from ``IPVAR`` (``READ (C360,'(A6,1X,A16,7X,A6,6F6.0)')``: fixed columns, not whitespace), the
whitespace reader :func:`agrijax.io.dssat.genotype.read_cul`, and byte identity of the untouched
part of the file. Inline cultivar files plus the public BSD-3 ``MZCER048.CUL`` fixture of
``tests/fixtures/dssat`` (DSSAT-CSM v4.8.6.0).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from agrijax.io.dssat.cultivar_write import (
    CUL_COEFFICIENTS,
    cul_bounds,
    cul_line,
    cul_precision,
    write_cultivar,
)
from agrijax.io.dssat.genotype import read_cul

FIXTURE_CUL = Path(__file__).resolve().parents[1] / "fixtures" / "dssat" / "MZCER048.CUL"

CUL = """\
*MAIZE CULTIVAR COEFFICIENTS: MZCER048 MODEL
!
@VAR#  VRNAME.......... EXPNO   ECO#    P1    P2    P5    G2    G3 PHINT
!Coefficient #                           1     2     3     4     5     6

999991 MINIMA               . DFAULT   5.0 0.000 580.0 248.0  5.00 38.00
999992 MAXIMA               . DFAULT 450.0 2.000 999.0 990.0 16.50 75.00

TS0001 TEST ONE             . IB0001 160.0 0.750 780.0 750.0  8.50 49.00
TS0002 TEST TWO             . IB0001 259.0 1.193 947.1 924.3 8.168 43.00
! a trailing comment
"""


def _ipvar(line: str) -> dict[str, object]:
    """DSSAT's read of a CERES-Maize cultivar row: (A6,1X,A16,7X,A6,6F6.0); blanks inside a
    numeric field are ignored (internal READ, BLANK='NULL'), and ``F6.0`` scales nothing."""
    out: dict[str, object] = {"VAR#": line[0:6], "VRNAME": line[7:23], "ECO#": line[30:36]}
    for k, n in enumerate(CUL_COEFFICIENTS):
        out[n] = float(line[36 + 6 * k : 42 + 6 * k].replace(" ", ""))
    return out


def _points(line: str) -> bool:
    """Every coefficient field of a written row has a decimal point and a leading blank (true of
    every value that fits five characters with a point)."""
    fields = [line[36 + 6 * k : 42 + 6 * k] for k in range(len(CUL_COEFFICIENTS))]
    return all("." in f and f.startswith(" ") for f in fields)


@pytest.fixture
def cul(tmp_path: Path) -> Path:
    p = tmp_path / "src" / "MZCER048.CUL"
    p.parent.mkdir()
    p.write_bytes(CUL.replace("\n", "\r\n").encode("latin-1"))
    return p


def test_precision_and_bounds(cul: Path) -> None:
    assert cul_precision(cul) == {"P1": 1, "P2": 3, "P5": 1, "G2": 1, "G3": 2, "PHINT": 2}
    assert cul_bounds(cul)["P5"] == (580.0, 999.0) and cul_bounds(cul)["PHINT"] == (38.0, 75.0)


def test_write_new_row_dssat_layout(cul: Path, tmp_path: Path) -> None:
    before = cul.read_bytes()
    mtime = cul.stat().st_mtime_ns
    vals = {"P1": 172.3456, "P2": 0.51, "P5": 1003.2, "G2": 800.0, "G3": 8.16789, "PHINT": 43.123}
    r = write_cultivar(cul, tmp_path / "out" / "MZCER048.CUL", "AJ0001", "calibrated", vals, ecotype="IB0001")
    # the source is untouched
    assert cul.read_bytes() == before and cul.stat().st_mtime_ns == mtime
    # DSSAT's fixed-column read of the new row
    row = _ipvar(r.line)
    assert _points(r.line)
    assert row["VAR#"] == "AJ0001" and str(row["VRNAME"]).strip() == "calibrated" and row["ECO#"] == "IB0001"
    assert r.line == "AJ0001 calibrated           . IB0001 172.3 0.510 1003. 800.0 8.168 43.12"
    assert r.written == {"P1": 172.3, "P2": 0.51, "P5": 1003.0, "G2": 800.0, "G3": 8.168, "PHINT": 43.12}
    assert all(row[n] == r.written[n] for n in CUL_COEFFICIENTS)
    assert r.rounded == {
        "P1": (172.3456, 172.3),
        "P5": (1003.2, 1003.0),
        "G3": (8.16789, 8.168),
        "PHINT": (43.123, 43.12),
    }
    assert r.out_of_range == {"P5": (1003.0, 580.0, 999.0)}
    # the rest of the file is byte-identical, the row sits after the last cultivar, CRLF kept
    out = r.path.read_bytes()
    lines = out.split(b"\r\n")
    src_lines = before.split(b"\r\n")
    k = src_lines.index(b"TS0002 TEST TWO             . IB0001 259.0 1.193 947.1 924.3 8.168 43.00")
    assert lines[: k + 1] == src_lines[: k + 1] and lines[k + 2 :] == src_lines[k + 1 :]
    assert lines[k + 1] == r.line.encode()
    # the whitespace reader returns the written values too, and the other rows unchanged
    back = read_cul(r.path)
    assert list(back.index) == ["999991", "999992", "TS0001", "TS0002", "AJ0001"]
    for n in CUL_COEFFICIENTS:
        assert back.loc["AJ0001", n] == r.written[n]
    assert back.drop(index="AJ0001").equals(read_cul(cul))


def test_base_cultivar_values_come_back_exactly(cul: Path, tmp_path: Path) -> None:
    r = write_cultivar(cul, tmp_path / "a.CUL", "AJ0002", "copy of TS0002", {}, base="TS0002")
    ref = _ipvar(next(ln for ln in CUL.splitlines() if ln.startswith("TS0002")))
    assert r.rounded == {} and r.out_of_range == {}
    assert _ipvar(r.line)["ECO#"] == "IB0001"
    for n in CUL_COEFFICIENTS:
        assert r.written[n] == ref[n]
    assert r.line[23:] == next(ln for ln in CUL.splitlines() if ln.startswith("TS0002"))[23:]
    # a partial override keeps the base for the rest
    r2 = write_cultivar(cul, tmp_path / "b.CUL", "AJ0003", "", {"G2": 700.0}, base="TS0001")
    assert r2.written["G2"] == 700.0 and r2.written["P1"] == 160.0 and r2.written["P2"] == 0.75


def test_refusals(cul: Path, tmp_path: Path) -> None:
    ok = dict.fromkeys(CUL_COEFFICIENTS, 50.0)
    with pytest.raises(ValueError, match="destination is the source"):
        write_cultivar(cul, cul, "AJ0001", "x", ok, ecotype="IB0001", overwrite=True)
    with pytest.raises(ValueError, match="already in"):
        write_cultivar(cul, tmp_path / "a.CUL", "TS0001", "x", ok, ecotype="IB0001")
    with pytest.raises(ValueError, match="cultivar id"):
        write_cultivar(cul, tmp_path / "a.CUL", "!X0001", "x", ok, ecotype="IB0001")
    with pytest.raises(ValueError, match="cultivar id"):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ01", "x", ok, ecotype="IB0001")
    with pytest.raises(ValueError, match="longer than 16"):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x" * 17, ok, ecotype="IB0001")
    with pytest.raises(ValueError, match="missing coefficients"):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x", {"P1": 1.0}, ecotype="IB0001")
    with pytest.raises(ValueError, match="unknown coefficients"):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x", {**ok, "RUE": 4.0}, ecotype="IB0001")
    with pytest.raises(ValueError, match="give the ecotype"):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x", ok)
    with pytest.raises(ValueError, match="does not fit"):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x", {**ok, "G2": 123456.0}, ecotype="IB0001")
    with pytest.raises(ValueError, match="not a finite"):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x", {**ok, "G2": math.nan}, ecotype="IB0001")
    write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x", ok, ecotype="IB0001")
    with pytest.raises(FileExistsError):
        write_cultivar(cul, tmp_path / "a.CUL", "AJ0001", "x", ok, ecotype="IB0001")
    assert not (tmp_path / "b.CUL").exists()


def test_hard_link_to_the_source_is_refused(cul: Path, tmp_path: Path) -> None:
    link = tmp_path / "link.CUL"
    os.link(cul, link)
    ok = dict.fromkeys(CUL_COEFFICIENTS, 50.0)
    with pytest.raises(ValueError, match="destination is the source"):
        write_cultivar(cul, link, "AJ0001", "x", ok, ecotype="IB0001", overwrite=True)
    alias = cul.parent / ".." / "src" / "MZCER048.CUL"
    with pytest.raises(ValueError, match="destination is the source"):
        write_cultivar(cul, alias, "AJ0001", "x", ok, ecotype="IB0001", overwrite=True)


def test_short_and_integer_texts(cul: Path, tmp_path: Path) -> None:
    """|x| < 1 without the leading 0 when that keeps a decimal more; integers that need all five
    characters without a point (F6.0 reads them as written)."""
    vals = {"P1": 12345.0, "P2": 0.5001, "P5": -1235.0, "G2": 0.75, "G3": -0.5001, "PHINT": 12345.4}
    r = write_cultivar(cul, tmp_path / "a.CUL", "AJ0009", "texts", vals, ecotype="IB0001")
    fields = [r.line[36 + 6 * k : 42 + 6 * k] for k in range(6)]
    assert fields == [" 12345", " .5001", " -1235", "  0.75", " -0.50", " 12345"]
    assert r.written == {"P1": 12345.0, "P2": 0.5001, "P5": -1235.0, "G2": 0.75, "G3": -0.5, "PHINT": 12345.0}
    assert r.rounded == {"G3": (-0.5001, -0.5), "PHINT": (12345.4, 12345.0)}
    row = _ipvar(r.line)
    assert all(row[n] == r.written[n] for n in CUL_COEFFICIENTS)
    back = read_cul(r.path)
    assert all(back.loc["AJ0009", n] == r.written[n] for n in CUL_COEFFICIENTS)


def _integer_convention_cul(tmp_path: Path) -> Path:
    """A cultivar file whose MINIMA row is printed without decimals (convention: 0 decimals)."""

    def row(ident: str, name: str, eco: str, vals: list[int]) -> str:
        return f"{ident:<6} {name:<16}{'.':>6} {eco:<6}" + "".join(f"{v:>6}" for v in vals)

    text = "\n".join(
        [
            "*MAIZE CULTIVAR COEFFICIENTS: MZCER048 MODEL",
            "@VAR#  VRNAME.......... EXPNO   ECO#    P1    P2    P5    G2    G3 PHINT",
            row("999991", "MINIMA", "DFAULT", [5, 0, 580, 248, 5, 38]),
            row("999992", "MAXIMA", "DFAULT", [450, 2, 999, 990, 16, 75]),
            row("TS0001", "TEST ONE", "IB0001", [160, 1, 780, 750, 8, 49]),
            "",
        ]
    )
    p = tmp_path / "int" / "MZCER048.CUL"
    p.parent.mkdir()
    p.write_text(text, encoding="latin-1")
    return p


@pytest.mark.parametrize("x", [0.0, -0.0, 3e-5, -3e-5, 4.9e-5, -4.9e-5, 1e-12, -1e-12])
def test_value_rounding_to_zero_without_decimals_is_a_number(x: float) -> None:
    """With no decimals in the file's convention a value that rounds to zero is ``0.`` / ``-0.``
    (the digit-less ``.`` / ``-.`` was no number)."""
    dec = dict.fromkeys(CUL_COEFFICIENTS, 0)
    ln = cul_line("AB1234", "zero", dict.fromkeys(CUL_COEFFICIENTS, x), ecotype="IB0001", decimals=dec)
    fields = [ln[36 + 6 * k : 42 + 6 * k] for k in range(6)]
    assert fields == [f"{('-0.' if math.copysign(1.0, x) < 0 else '0.'):>6}"] * 6
    assert all(v == 0.0 for k, v in _ipvar(ln).items() if k in CUL_COEFFICIENTS)


def test_write_zero_value_into_a_file_of_integer_columns(tmp_path: Path) -> None:
    cul = _integer_convention_cul(tmp_path)
    assert cul_precision(cul) == dict.fromkeys(CUL_COEFFICIENTS, 0)
    vals = {"P1": 160.0, "P2": 0.0, "P5": 780.0, "G2": 750.0, "G3": -1e-5, "PHINT": 3e-5}
    r = write_cultivar(cul, tmp_path / "out.CUL", "AJ0001", "zero", vals, ecotype="IB0001")
    fields = [r.line[36 + 6 * k : 42 + 6 * k].strip() for k in range(6)]
    assert fields == ["160.", "0.", "780.", "750.", "-0.", "0."]
    assert r.written == {"P1": 160.0, "P2": 0.0, "P5": 780.0, "G2": 750.0, "G3": 0.0, "PHINT": 0.0}
    assert r.rounded == {"G3": (-1e-5, 0.0), "PHINT": (3e-5, 0.0)}
    assert _points(r.line) and all(_ipvar(r.line)[n] == r.written[n] for n in CUL_COEFFICIENTS)
    back = read_cul(r.path)
    assert all(back.loc["AJ0001", n] == r.written[n] for n in CUL_COEFFICIENTS)


def test_cul_line_layout() -> None:
    dec = dict.fromkeys(CUL_COEFFICIENTS, 2)
    ln = cul_line(
        "AB1234", "name", dict.fromkeys(CUL_COEFFICIENTS, 1.5), ecotype="IB0001", decimals=dec, expno="3"
    )
    assert ln == "AB1234 name                 3 IB0001  1.50  1.50  1.50  1.50  1.50  1.50"
    assert (
        ln[:6] == "AB1234"
        and ln[7:23] == "name".ljust(16)
        and ln[23:29] == "     3"
        and ln[30:36] == "IB0001"
    )
    assert len(ln) == 36 + 6 * 6


def test_every_fixture_cultivar_rewritten_under_a_new_id(tmp_path: Path) -> None:
    """Every cultivar of the DSSAT v4.8.6.0 file, written under a new id with its own values, reads
    back (DSSAT's columns and read_cul) with exactly the source row's values."""
    src = read_cul(FIXTURE_CUL)
    lines = FIXTURE_CUL.read_bytes().decode("latin-1").split("\r\n")
    n = 0
    dup = set(src.index[src.index.duplicated()])
    for k, ident in enumerate(src.index):
        if ident in {"999991", "999992"} or ident in dup:  # a repeated id: DSSAT uses the first
            continue
        vals = {c: float(src.loc[ident, c]) for c in CUL_COEFFICIENTS}
        if any(math.isnan(v) for v in vals.values()):
            continue
        r = write_cultivar(FIXTURE_CUL, tmp_path / f"{k}.CUL", "AJ9999", "x", vals, base=str(ident))
        assert r.rounded == {}, (ident, r.rounded)
        dssat = _ipvar(r.line)
        assert _points(r.line), r.line
        src_line = next(ln for ln in lines if ln[:6].strip() == ident)
        ref = _ipvar(src_line.ljust(72))
        for c in CUL_COEFFICIENTS:
            assert dssat[c] == ref[c] == vals[c], (ident, c)
        assert read_cul(r.path).loc["AJ9999", "P5"] == vals["P5"]
        n += 1
    assert n > 100
