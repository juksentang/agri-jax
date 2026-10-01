"""Write calibrated CERES-Maize cultivar coefficients into a copy of a DSSAT ``MZCER048.CUL``.

:func:`write_cultivar` adds **one new cultivar row** (a new ``VAR#`` and name, the coefficients
``P1 P2 P5 G2 G3 PHINT``) to a **copy** of a DSSAT v4.8 maize cultivar file; the source file is
never opened for writing. Every other byte of the copy is the source's (comments, line ends,
the existing cultivars), the new row goes right after the source's last cultivar row.

The row is written in the fixed format DSSAT-CSM v4.8.6.0 reads it with (``IPVAR``,
``InputModule/IPVAR.for``, case ``MZCER``: ``READ (C360,'(A6,1X,A16,7X,A6,6F6.0)') VARTY, VRNAME,
ECONO, P1, P2, P5, G2, G3, PHINT``): the id in columns 1-6, the name in 8-23, ``EXPNO`` right-aligned
in 24-29 (inside the ``7X`` DSSAT skips), the ecotype in 31-36, then one six-column field per
coefficient. The **printed precision** is that of the file's six-column fields. Every value is
right-aligned in its field and at most five characters long, so one blank always precedes it and
whitespace readers such as :func:`~agrijax.io.dssat.genotype.read_cul` read the same values as
DSSAT's fixed columns. The text is chosen in this order:

1. the most decimals (at most six, at least one) that fit in five characters, then trailing zeros
   removed down to the file's convention for that column; for ``|x| < 1`` the same without the
   leading ``0`` is used instead when it is closer to ``x`` (it keeps a decimal more:
   ``0.5001`` is written ``.5001``, ``0.75`` stays ``0.750``, ``-0.5001`` is ``-0.50``). The file's
   convention for a column is the decimals of its ``MINIMA`` row, else of most cultivar rows
   (P1 ``.1``, P2 ``.001``, P5 ``.1``, G2 ``.1``, G3 ``.01``, PHINT ``.01`` in MZCER048.CUL); with a
   convention of no decimals a value that rounds to zero is written ``0.`` (``-0.`` when negative),
   never the bare ``.`` that no number reader accepts;
2. else the value rounded to an integer with a decimal point (``1003.``);
3. else the integer without a point (``12345``, ``-1235``): ``F6.0`` reads it as that integer
   (a ``d`` of 0 scales nothing);
4. else (``123456``) ``ValueError``.

Published values therefore come back exactly (``IB0035`` G3 ``8.168``); a calibrated
``172.3456`` is written ``172.3``, and every value the text changes is reported in
:attr:`CultivarWrite.rounded`.

DSSAT looks a cultivar up by its id (first match in the file), and ``IGNORE`` skips lines that
start with ``!``, ``*``, ``$``, ``@`` or a blank, so the id must be six characters, start with a
letter or digit, and must not already be in the file. Values outside the file's ``MINIMA`` /
``MAXIMA`` rows (ids ``999991`` / ``999992``) are **reported** (:attr:`CultivarWrite.out_of_range`),
not refused: DSSAT does not check them either, the bounds are those of DSSAT's calibration tools.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ._fixed import read_lines

__all__ = [
    "CUL_COEFFICIENTS",
    "CultivarWrite",
    "cul_bounds",
    "cul_line",
    "cul_precision",
    "cul_written",
    "write_cultivar",
]

#: coefficients of a DSSAT 4.8 CERES-Maize cultivar row, in file order (``IPVAR``, case MZCER)
CUL_COEFFICIENTS = ("P1", "P2", "P5", "G2", "G3", "PHINT")
#: ids of the bound rows of a DSSAT cultivar file
MINIMA_ID, MAXIMA_ID = "999991", "999992"

_ID_WIDTH = 6
_NAME_WIDTH = 16
_EXPNO_WIDTH = 6  # right-aligned in the ``7X`` between the name and the ecotype
_ECO_WIDTH = 6
_FIELD = 6  # one ``F6.0`` coefficient field
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{5}$")
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)$")


@dataclass(frozen=True)
class CultivarWrite:
    """What :func:`write_cultivar` wrote."""

    path: Path
    source: Path
    cultivar_id: str
    line: str
    #: the values as written (rounded to the file's precision)
    written: dict[str, float]
    #: ``{name: (given, written)}`` for every value the rounding changed
    rounded: dict[str, tuple[float, float]]
    #: ``{name: (value, minimum, maximum)}`` for written values outside ``MINIMA`` / ``MAXIMA``
    out_of_range: dict[str, tuple[float, float, float]]


def _header_and_rows(lines: list[str]) -> tuple[int, list[int]]:
    """Index of the cultivar ``@`` header and of every cultivar row after it."""
    hdr = next((i for i, ln in enumerate(lines) if ln.startswith("@") and "VAR#" in ln), None)
    if hdr is None:
        raise ValueError("no '@VAR#' header line: not a DSSAT 4.8 cultivar file")
    rows = [i for i in range(hdr + 1, len(lines)) if lines[i][:1].strip() and lines[i][:1] not in "!*$@"]
    return hdr, rows


def _columns(header: str) -> list[str]:
    toks = header[1:].split()
    return [t.rstrip(".") for t in toks]


def _fields(line: str) -> dict[str, str]:
    """The coefficient fields of one row as text (``(A6,1X,A16,7X,A6,6F6.0)`` positions)."""
    start = _ID_WIDTH + 1 + _NAME_WIDTH + 7 + _ECO_WIDTH
    return {
        name: line[start + k * _FIELD : start + (k + 1) * _FIELD].strip()
        for k, name in enumerate(CUL_COEFFICIENTS)
    }


def _decimals(text: str) -> int | None:
    if not _NUM.match(text):
        return None
    return len(text.split(".", 1)[1]) if "." in text else 0


def cul_precision(path: str | Path) -> dict[str, int]:
    """The file's decimal convention per coefficient (the least decimals written): those of the
    ``MINIMA`` row, else the most common among the cultivar rows."""
    lines = read_lines(path)
    _, rows = _header_and_rows(lines)
    by_id = {lines[i][:_ID_WIDTH].strip(): lines[i] for i in rows}
    out: dict[str, int] = {}
    for name in CUL_COEFFICIENTS:
        d = _decimals(_fields(by_id[MINIMA_ID])[name]) if MINIMA_ID in by_id else None
        if d is None:
            counts = Counter(_decimals(_fields(lines[i])[name]) for i in rows)
            counts.pop(None, None)
            if not counts:
                raise ValueError(f"{path}: no printed value of {name}")
            d = int(counts.most_common(1)[0][0])  # type: ignore[arg-type]
        out[name] = d
    return out


def cul_bounds(path: str | Path) -> dict[str, tuple[float, float]]:
    """``{name: (MINIMA, MAXIMA)}`` from the file's bound rows (empty when it has none)."""
    lines = read_lines(path)
    _, rows = _header_and_rows(lines)
    by_id = {lines[i][:_ID_WIDTH].strip(): lines[i] for i in rows}
    if MINIMA_ID not in by_id or MAXIMA_ID not in by_id:
        return {}
    lo, hi = _fields(by_id[MINIMA_ID]), _fields(by_id[MAXIMA_ID])
    return {n: (float(lo[n]), float(hi[n])) for n in CUL_COEFFICIENTS}


#: most decimals tried for a coefficient (more than any six-column field can print)
_MAX_DECIMALS = 6


def _fmt(x: float, min_decimals: int, name: str) -> str:
    """Text of ``x`` in at most ``_FIELD - 1`` characters (see the module docstring): the most
    decimals that fit, with a decimal point (``0.`` written ``.`` for ``|x| < 1`` when that keeps a
    decimal more), trailing zeros removed down to ``min_decimals``; else the integer without a point
    (``12345``, ``-1235``), which ``F6.0`` reads as that integer."""
    if not math.isfinite(x):
        raise ValueError(f"{name}: {x!r} is not a finite value")
    room = _FIELD - 1

    def best(drop_zero: bool) -> str | None:
        for d in range(_MAX_DECIMALS, 0, -1):
            t = f"{x:.{d}f}"
            if drop_zero and t.lstrip("-").startswith("0."):
                t = t.replace("0.", ".", 1)
            if len(t) <= room:
                while d > max(min_decimals, 0) and t.endswith("0"):
                    t, d = t[:-1], d - 1
                if t.lstrip("-") == ".":  # every digit removed: "." is no number, "0." is
                    t = t.replace(".", "0.", 1)
                return t
        return None

    with_zero, no_zero = best(False), best(abs(x) < 1.0)
    if no_zero is not None and (with_zero is None or abs(float(no_zero) - x) < abs(float(with_zero) - x)):
        return no_zero
    if with_zero is not None:
        return with_zero
    t = f"{x:.0f}."
    if len(t) <= room:
        return t
    t = f"{x:.0f}"
    if len(t) <= room:
        return t
    raise ValueError(f"{name}: {x!r} does not fit a {_FIELD}-column field")


def cul_line(
    cultivar_id: str,
    name: str,
    values: Mapping[str, float],
    *,
    ecotype: str,
    decimals: Mapping[str, int],
    expno: str = ".",
) -> str:
    """One cultivar row in DSSAT's ``(A6,1X,A16,7X,A6,6F6.0)`` layout (see the module docstring)."""
    if not _ID.match(cultivar_id):
        raise ValueError(f"cultivar id {cultivar_id!r}: six letters/digits, not starting with a blank or '!'")
    if len(name) > _NAME_WIDTH:
        raise ValueError(f"cultivar name {name!r} longer than {_NAME_WIDTH} characters")
    if len(ecotype) > _ECO_WIDTH or not ecotype.strip():
        raise ValueError(f"ecotype {ecotype!r}: one to {_ECO_WIDTH} characters")
    if len(expno) > _EXPNO_WIDTH:
        raise ValueError(f"EXPNO {expno!r} longer than {_EXPNO_WIDTH} characters")
    missing = [n for n in CUL_COEFFICIENTS if n not in values]
    if missing:
        raise ValueError(f"missing coefficients {missing}")
    extra = sorted(set(values) - set(CUL_COEFFICIENTS))
    if extra:
        raise ValueError(f"unknown coefficients {extra} (a MZCER048 row holds {list(CUL_COEFFICIENTS)})")
    head = f"{cultivar_id:<{_ID_WIDTH}} {name:<{_NAME_WIDTH}}{expno:>{_EXPNO_WIDTH}} {ecotype:<{_ECO_WIDTH}}"
    body = "".join(f"{_fmt(float(values[n]), int(decimals[n]), n):>{_FIELD}}" for n in CUL_COEFFICIENTS)
    return head + body


def cul_written(values: Mapping[str, float], *, decimals: Mapping[str, int]) -> dict[str, float]:
    """The six coefficients as a row written by :func:`cul_line` holds them (what DSSAT reads back):
    each value through its printed text (``decimals``: the file's convention, :func:`cul_precision`).
    A calibration that is exported is selected and evaluated on these values."""
    line = cul_line("AJ9999", "written", values, ecotype="IB0001", decimals=decimals)
    return {n: float(t) for n, t in _fields(line).items()}


def write_cultivar(
    source: str | Path,
    dest: str | Path,
    cultivar_id: str,
    name: str,
    values: Mapping[str, float],
    *,
    base: str | None = None,
    ecotype: str | None = None,
    expno: str = ".",
    overwrite: bool = False,
) -> CultivarWrite:
    """Copy ``source`` (a DSSAT ``MZCER048.CUL``) to ``dest`` with one new cultivar row.

    ``values`` holds the six coefficients; with ``base`` (an existing cultivar id) any coefficient
    left out of ``values`` is taken from that row, and so is the ecotype unless ``ecotype`` is
    given. ``dest`` must not be the source file (``os.path.samefile``: also not a hard link or
    another path to it); an existing ``dest`` is replaced only with
    ``overwrite=True``. Returns what was written (rounding and ``MINIMA`` / ``MAXIMA`` report).
    """
    src, dst = Path(source), Path(dest)
    if dst.exists() and os.path.samefile(src, dst):  # also a hard link or another path to it
        raise ValueError(f"{dst}: the destination is the source file; write to a copy")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"{dst} exists (pass overwrite=True to replace it)")
    text = src.read_bytes().decode("latin-1")
    nl = "\r\n" if "\r\n" in text else "\n"
    raw = text.split(nl)
    lines = [ln.replace("\t", " ") for ln in raw]
    _, rows = _header_and_rows(lines)
    ids = {lines[i][:_ID_WIDTH].strip(): i for i in reversed(rows)}  # first occurrence wins
    if cultivar_id in ids:
        raise ValueError(f"cultivar id {cultivar_id!r} is already in {src}")
    vals = {k: float(v) for k, v in values.items()}
    eco = ecotype
    if base is not None:
        if base not in ids:
            raise ValueError(f"base cultivar {base!r} not in {src}")
        bl = lines[ids[base]]
        for n, t in _fields(bl).items():
            vals.setdefault(n, float(t))
        if eco is None:
            eco = bl[_ID_WIDTH + 1 + _NAME_WIDTH + 7 : _ID_WIDTH + 1 + _NAME_WIDTH + 7 + _ECO_WIDTH].strip()
    if eco is None:
        raise ValueError("give the ecotype (ecotype=...) or a base cultivar (base=...)")
    dec = cul_precision(src)
    line = cul_line(cultivar_id, name, vals, ecotype=eco, decimals=dec, expno=expno)
    fields = _fields(line)
    written = {n: float(fields[n]) for n in CUL_COEFFICIENTS}
    rounded = {n: (vals[n], written[n]) for n in CUL_COEFFICIENTS if written[n] != vals[n]}
    bounds = cul_bounds(src)
    out_of_range = {n: (written[n], lo, hi) for n, (lo, hi) in bounds.items() if not (lo <= written[n] <= hi)}
    last = max(rows) if rows else len(raw) - 1
    out = [*raw[: last + 1], line, *raw[last + 1 :]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(nl.join(out).encode("latin-1"))
    return CultivarWrite(dst, src, cultivar_id, line, written, rounded, out_of_range)
