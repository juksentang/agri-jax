"""Header-driven fixed-width parsing shared by the DSSAT readers.

DSSAT files describe each table by an ``@`` header line whose tokens sit roughly above
their values: numbers are right-aligned to the end of their header token, text such as
``TNAME....`` or ``ID_SOIL`` is left-aligned to its start and may overrun it
(``IBMZ910014`` under ``ID_SOIL``). Splitting on whitespace is therefore wrong whenever a
text value contains spaces (``RAINFED LOW NITROGEN``) or a column is blank.

The rule used here: field *i* spans from the end of header token *i-1* to the end of header
token *i* (the last field runs to the end of the line). When a boundary cuts through a
non-blank word of the data line, the word goes to the left field when it starts at or
before that field's header token (a left-aligned overrun), otherwise to the side holding
most of its characters (ties go left), which also absorbs right-aligned values that end
a character or two past their header token -- unless the two halves are themselves complete
numbers (``77.4`` + ``1004.0`` written with no separating blank), where the cut is kept.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_INT = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

MISSING = -99.0


def read_lines(path: str | Path) -> list[str]:
    """Read a DSSAT text file as a list of lines (latin-1: one byte == one column)."""
    text = Path(path).read_bytes().decode("latin-1")
    return [ln.rstrip("\r\n").replace("\t", " ") for ln in text.splitlines()]


def clean_name(tok: str) -> str:
    """``TNAME....`` -> ``TNAME``; ``...........XCRD`` -> ``XCRD``; keeps ``L#SD``."""
    t = tok.strip(".")
    return t if t else tok


def header_tokens(header: str, *, joins: Iterable[str] = ()) -> list[tuple[str, int, int]]:
    """Tokens of an ``@`` header line as ``(name, start, end)`` column spans.

    ``joins`` lists multi-word names (e.g. ``"SCS FAMILY"``) that form one column.
    """
    line = header.replace("@", " ", 1) if header.startswith("@") else header
    for j in joins:
        line = line.replace(j, j.replace(" ", "_"))
    return [(clean_name(m.group(0)), m.start(), m.end()) for m in re.finditer(r"\S+", line)]


def _is_num(s: str) -> bool:
    return bool(_INT.match(s) or _FLOAT.match(s))


def _glued_numbers(left: str, right: str) -> bool:
    """True when a word cut at a header boundary is two touching values (``77.41004.0``,
    ``0.086-0.12``, ``210C2cacs``) rather than one value overrunning its column
    (``29.630``, ``12.5``, ``IBMZ910014``)."""
    if _is_num(left) and not _is_num(right):
        return True  # number glued to a text code, e.g. SLB+SLMH ``210C2cacs``
    if not (_is_num(left) and _is_num(right)):
        return False
    if right.startswith("-"):
        return True
    return "." in left and not left.endswith(".") and "." in right


def _boundaries(tokens: Sequence[tuple[str, int, int]], line: str) -> list[int]:
    """Field cut points (len(tokens)+1 values) for one data line."""
    n = len(line)
    cuts = [0] + [end for _, _, end in tokens[:-1]] + [max(n, tokens[-1][2])]
    for i in range(1, len(cuts) - 1):
        b = cuts[i]
        if 0 < b < n and line[b - 1] != " " and line[b] != " ":
            ws = b
            while ws > 0 and line[ws - 1] != " ":
                ws -= 1
            we = b
            while we < n and line[we] != " ":
                we += 1
            if _glued_numbers(line[ws:b], line[b:we]):
                continue
            left_start = tokens[i - 1][1]
            cuts[i] = we if (ws <= left_start or (b - ws) >= (we - b)) else ws
    for i in range(1, len(cuts)):  # keep monotone
        cuts[i] = max(cuts[i], cuts[i - 1])
    # a value written one column right of its header token (``@TRNO`` / ``     1 2929.``)
    # leaves its own field blank and starts the next one: give it back
    for i in range(1, len(cuts) - 1):
        b = cuts[i]
        if b < n and line[b] != " " and not line[cuts[i - 1] : b].strip():
            we = b
            while we < n and line[we] != " ":
                we += 1
            if line[we : cuts[i + 1]].strip():
                cuts[i] = we
    return cuts


def split_fixed(tokens: Sequence[tuple[str, int, int]], line: str) -> list[str]:
    """Split one data line into stripped raw strings, one per header token."""
    cuts = _boundaries(tokens, line)
    return [line[cuts[i] : cuts[i + 1]].strip() for i in range(len(tokens))]


def convert(raw: str) -> Any:
    """Typed value: int, float or str (leading-zero codes like ``00000`` stay str)."""
    s = raw.strip()
    if _INT.match(s):
        digits = s.lstrip("+-")
        if len(digits) > 1 and digits.startswith("0"):
            return s
        return int(s)
    if _FLOAT.match(s):
        return float(s)
    return s


def to_float(raw: Any, *, missing_to_nan: bool = True) -> float:
    """Float value of a raw field; ``-99``, blanks and ``****`` overflows become NaN."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return math.nan
    if missing_to_nan and v <= -99.0 and v > -99.5:
        return math.nan
    return v


def frame_from_rows(
    names: Sequence[str], rows: Sequence[Sequence[str]], *, missing_to_nan: bool = True
) -> pd.DataFrame:
    """DataFrame from raw string rows: numeric columns to float/int, others kept as str."""
    data: dict[str, Any] = {}
    for j, name in enumerate(names):
        col = [r[j] if j < len(r) else "" for r in rows]
        nonblank = [c for c in col if c and "*" not in c]
        if all(_INT.match(c) or _FLOAT.match(c) for c in nonblank):
            arr = np.array([float(c) if c and "*" not in c else np.nan for c in col], dtype=float)
            if missing_to_nan:
                arr = np.where((arr <= -99.0) & (arr > -99.5), np.nan, arr)
            if nonblank and all(_INT.match(c) for c in nonblank) and not np.isnan(arr).any():
                data[name] = arr.astype(np.int64)
            else:
                data[name] = arr
        else:
            data[name] = list(col)
    return pd.DataFrame(data)


def fmt_num(v: Any, width: int, decimals: int | None = None) -> str:
    """Right-aligned value in ``width`` columns (leaving one leading blank when possible).

    NaN/None -> ``-99``. Strings are right-aligned. Floats use ``decimals`` when given,
    otherwise the first of 3, 2, 1, 0, 4, 5, 6 decimals that reproduces the value and fits
    ``width - 1`` columns (then ``width`` columns, touching the previous value the way a
    full Fortran ``F6.2`` field does).
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'-99':>{width}}"
    if isinstance(v, str):
        return f"{v:>{width}}"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return f"{int(v):>{width}d}"
    x = float(v)
    if x == MISSING:
        return f"{'-99':>{width}}"
    if decimals is not None:
        return f"{x:>{width}.{decimals}f}"
    for room in (width - 1, width):  # the separating blank is used only when necessary
        for d in (3, 2, 1, 0, 4, 5, 6):
            t = f"{x:.{d}f}"
            if len(t) > room and t.lstrip("-").startswith("0."):
                t = t.replace("0.", ".", 1)
            if len(t) <= room and abs(float(t) - x) <= 1e-9 * max(1.0, abs(x)):
                return f"{t:>{width}}"
    return f"{short_float(x, width - 1):>{width}}"


def short_float(x: float, room: int) -> str:
    """Shortest fixed-point text of ``x`` that reproduces it and fits ``room`` characters
    (``0.`` prefixes dropped if needed, e.g. ``.0012``); else the most decimals that fit."""
    best: str | None = None
    for d in range(0, 7):
        s = f"{x:.{d}f}"
        if len(s) > room and s.lstrip("-").startswith("0."):
            s = s.replace("0.", ".", 1)
        if len(s) > room:
            break
        best = s
        if abs(float(s) - x) <= 1e-9 * max(1.0, abs(x)):
            return s
    return best if best is not None else f"{x:.0f}"
