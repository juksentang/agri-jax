"""DSSAT weather files (``*.WTH``): read and write.

``read_wth`` returns a DataFrame with a ``date`` column (``datetime64``) and lower-cased
variables (``srad`` MJ m-2 d-1, ``tmax``/``tmin`` degC, ``rain`` mm, and whatever else the
file carries: ``rhum`` %, ``wind`` km d-1, ``par``, ``dewp`` ...). The site line
(``INSI LAT LONG ELEV TAV AMP REFHT WNDHT`` [``CO2`` ...]) is stored in ``df.attrs["site"]``
and the title in ``df.attrs["title"]`` (``df.attrs["site_text"]``: the site fields as written).
``df.attrs["decimals"]`` keeps, per variable, the most
decimals written in the file and ``df.attrs["four_digit_year"]`` the date style, so
:func:`write_wth` reproduces the original look (``0.00`` rain stays ``0.00``).

Dates may be ``YYDDD`` (DSSAT Y2K rule: ``YY <= 40`` -> 20YY, else 19YY, as ``Y2K_DOY``)
or ``YYYYDDD``.

Field splitting. Values are split by the header positions (:mod:`._fixed`), which also reads
values that overrun their header token by a column and six-column values that touch
(``77.41004.0``). A field that is not a number there (a quality flag in the column after a
value, ``16.0E 32.6``; a note after the last column; ``-99m``) is read the way DSSAT 4.8
``IPWTH`` reads every field: the ``PARSE_HEADERS`` span, first list-directed item, NaN when that
fails (:func:`agri_jax.io.dssat._fixed.dssat_header_spans`). ``read_wth(..., dssat_spans=True)``
uses the DSSAT rule for every field, i.e. returns exactly the numbers ``dscsm048`` reads (it
truncates values that overrun their header token: ``4.5`` written one column right reads
``4.``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._fixed import (
    convert,
    dssat_header_spans,
    header_tokens,
    list_directed_float,
    read_lines,
    split_fixed,
    to_float,
)

__all__ = ["parse_dssat_date", "read_wth", "write_wth"]

SITE_COLUMNS = ("INSI", "LAT", "LONG", "ELEV", "TAV", "AMP", "REFHT", "WNDHT")
#: width of each site column in the standard layout (other columns: 6)
_SITE_WIDTH = {"INSI": 6, "LAT": 9, "LONG": 9}
_DEC = re.compile(r"^[+-]?\d*\.(\d*)$")


def parse_dssat_date(code: str | int) -> date:
    """``YYDDD`` / ``YYYYDDD`` -> ``datetime.date`` (Y2K rule of DSSAT ``Y2K_DOY``)."""
    s = str(code).strip()
    if len(s) <= 5:
        v = int(s)
        yy, doy = divmod(v, 1000)
        year = 2000 + yy if yy <= 40 else 1900 + yy
    else:
        year, doy = int(s[:-3]), int(s[-3:])
    return date(year, 1, 1) + timedelta(days=doy - 1)


def _site_value(raw: str) -> Any:
    v = convert(raw) if raw else None
    if isinstance(v, (int, float)) and -99.5 < v <= -99.0:
        return None
    return v


def _decimals(raw: str) -> int:
    m = _DEC.match(raw.strip())
    return len(m.group(1)) if m else 0


def _field_values(
    toks: list[tuple[str, int, int]],
    spans: list[tuple[str, int, int]],
    line: str,
    *,
    dssat_spans: bool,
) -> tuple[list[float], list[str]]:
    """Float value and raw text of every field of one data line (see module docstring)."""
    aligned = split_fixed(toks, line)
    vals: list[float] = []
    raws: list[str] = []
    for j, raw in enumerate(aligned):
        _, a, b = spans[j]
        if dssat_spans:
            raw = line[a:b].strip()
            v = list_directed_float(raw)
        else:
            v = to_float(raw, missing_to_nan=False) if raw else math.nan
            if not raw or (math.isnan(v) and raw.lower() != "nan"):
                raw = line[a:b].strip()
                v = list_directed_float(raw)
        if -99.5 < v <= -99.0:
            v = math.nan
        vals.append(v)
        raws.append(raw.split()[0] if raw.split() else "")
    return vals, raws


def read_wth(path: str | Path, *, dssat_spans: bool = False) -> pd.DataFrame:
    """Read a ``.WTH`` file into a daily DataFrame (see module docstring).

    ``dssat_spans=True`` splits every field exactly as DSSAT 4.8 does (``PARSE_HEADERS`` spans,
    list-directed read), so the values are the ones ``dscsm048`` simulates with.
    """
    lines = read_lines(path)
    title = next((ln for ln in lines if ln.startswith(("*", "$"))), "")
    site: dict[str, Any] = {}
    site_raw: dict[str, str] = {}
    names: list[str] = []
    dates: list[str] = []
    rows: list[list[float]] = []
    decimals: dict[str, int] = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("@"):
            toks = header_tokens(ln)
            tnames = [t[0] for t in toks]
            if tnames and tnames[0] == "DATE":
                names = tnames
                spans = dssat_header_spans(ln)
                if len(spans) != len(toks):  # a '!' inside the header: DSSAT stops there
                    toks = toks[: len(spans)]
                    names = names[: len(spans)]
                for d in lines[i + 1 :]:
                    if not d.strip() or d.lstrip().startswith("!"):
                        continue
                    if d.startswith(("@", "*")):
                        break
                    vals, raws = _field_values(toks, spans, d, dssat_spans=dssat_spans)
                    dates.append(split_fixed(toks, d)[0])
                    rows.append(vals[1:])
                    for n, r in zip(names[1:], raws[1:], strict=True):
                        decimals[n] = max(decimals.get(n, 0), _decimals(r))
                break
            # a site table: first non-comment data line; several tables are merged
            for d in lines[i + 1 :]:
                if d.startswith(("@", "*")):
                    break
                if d.strip() and not d.lstrip().startswith("!"):
                    vals_s = split_fixed(toks, d)
                    for n, v in zip(tnames, vals_s, strict=True):
                        if n not in site:
                            site[n] = _site_value(v)
                            site_raw[n] = v
                    break
        i += 1
    if not names:
        raise ValueError(f"{path}: no '@DATE' table found")
    cols = [c.lower() for c in names[1:]]
    data = np.array(rows, dtype=float).reshape(len(rows), len(cols))
    df = pd.DataFrame(data, columns=pd.Index(cols))
    df.insert(0, "date", pd.to_datetime([parse_dssat_date(c) for c in dates]))
    df.attrs["site"] = site
    df.attrs["site_text"] = site_raw
    df.attrs["title"] = title
    df.attrs["decimals"] = {c.lower(): d for c, d in decimals.items()}
    df.attrs["four_digit_year"] = bool(dates) and len(dates[0]) > 5
    return df


def _wth_cell(v: float, min_decimals: int = 1) -> str:
    """Text of one ``.WTH`` value: the fewest decimals, at least ``max(min_decimals, 1)``, that
    reproduce it exactly (``8.4``, ``12.35``, ``0.00`` for a two-decimal column); NaN -> ``-99``.
    """
    if np.isnan(v):
        return "-99"
    for d in range(max(min_decimals, 1), 10):
        t = f"{v:.{d}f}"
        if abs(float(t) - v) <= 1e-9 * max(1.0, abs(v)):
            return t
    return repr(float(v))


def _site_text(key: str, v: Any) -> str:
    """Shortest exact text of a numeric site value (at least one decimal for floats)."""
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return str(int(v))
    x = float(v)
    for d in range(1, 7):
        t = f"{x:.{d}f}"
        if abs(float(t) - x) <= 1e-9 * max(1.0, abs(x)):
            return t
    return repr(x)


def write_wth(
    df: pd.DataFrame,
    path: str | Path,
    *,
    site: Mapping[str, Any] | None = None,
    title: str | None = None,
    four_digit_year: bool | None = None,
) -> Path:
    """Write a DataFrame (``date`` + variables, as returned by :func:`read_wth`) to ``.WTH``.

    ``site`` defaults to ``df.attrs["site"]`` (the standard eight columns when empty); its keys
    are written in order, each value exactly (``-22.7017`` stays ``-22.7017``), as the source
    text (``df.attrs["site_text"]``) when unchanged: DSSAT echoes ``LAT``/``LONG`` as text into
    ``Summary.OUT``, so ``-22.430`` must not become ``-22.43``. Each variable
    takes six columns, more when its name or a value needs them so that every value keeps a
    leading blank (DSSAT 4.8 reads the column right after a header token as part of no field,
    so a value touching its left neighbour would lose its first digit there). Values keep the
    decimals of the source file (``df.attrs["decimals"]``), or more when they need them, so
    ``12.35`` stays ``12.35``, ``8.4`` stays ``8.4`` and ``0.00`` stays ``0.00``
    (:func:`_wth_cell`); NaN -> ``-99``.
    Dates as ``YYDDD`` unless ``four_digit_year`` (default: ``df.attrs["four_digit_year"]``, or
    ``YYYYDDD`` when a year falls outside 1941-2040, which ``YYDDD`` cannot express). A
    ``YYYYDDD`` file gets a ``$WEATHER`` title line (``*WEATHER`` otherwise): DSSAT 4.8 reads
    seven-digit dates only when that keyword is present, else it takes ``2002303`` as ``20023``.
    """
    raw: dict[str, str] = {} if site is not None else dict(df.attrs.get("site_text", {}))
    site = dict(site if site is not None else df.attrs.get("site", {}))
    if not site:
        site = dict.fromkeys(SITE_COLUMNS)
    title = str(title if title is not None else df.attrs.get("title", "*WEATHER DATA :"))
    dates = pd.to_datetime(df["date"])
    if four_digit_year is None:
        four_digit_year = bool(df.attrs.get("four_digit_year", False)) or bool(
            len(dates) and ((dates.dt.year < 1941).any() or (dates.dt.year > 2040).any())
        )
    # DSSAT 4.8 reads YYYYDDD dates only from a file whose header holds "$WEATHER"
    # (MAKEFILEW.f90); otherwise it reads the first five columns as YYDDD
    if title[:1] in "*$" and title[1:].upper().startswith("WEATHER"):
        title = ("$" if four_digit_year else "*") + title[1:]
    else:
        title = ("$WEATHER DATA : " if four_digit_year else "*WEATHER DATA : ") + title.lstrip("*$ ")
    out = [title, ""]
    head, cells = "", ""
    for k, v in site.items():
        missing = v is None or (isinstance(v, float) and math.isnan(v))
        if k == "INSI":
            head += "  INSI"
            cells += f"  {'' if missing else v!s:<4}"
        elif isinstance(v, str):
            # text: the header token is padded with dots over the whole value (``SITE........``)
            name = k + "." * max(0, len(v) - len(k))
            head += " " + name
            cells += f" {v:<{len(name)}}"
        else:
            t = raw.get(k, "")
            if not t or _site_value(t) != (None if missing else v):  # keep the source text
                t = "-99" if missing else _site_text(k, v)
            w = max(_SITE_WIDTH.get(k, 6), len(k) + 1, len(t) + 1)
            head += f"{k:>{w}}"
            cells += f"{t:>{w}}"
    out.append("@" + head[1:])
    out.append(cells.rstrip())
    vars_ = [c for c in df.columns if c != "date"]
    width_date = 7 if four_digit_year else 5
    decs = {str(k): int(v) for k, v in dict(df.attrs.get("decimals", {})).items()}
    values = df[vars_].to_numpy(dtype=float)
    mins = [decs.get(c, 1) for c in vars_]
    # every value keeps a leading blank: DSSAT 4.8 reads the column after a header token as
    # part of no field, so a value touching its left neighbour would lose its first digit
    texts = [[_wth_cell(v, mins[j]) for v in values[:, j]] for j in range(len(vars_))]
    widths = [
        max(6, len(c) + 1, 1 + max((len(t) for t in col), default=3))
        for c, col in zip(vars_, texts, strict=True)
    ]
    out.append(
        "@"
        + "DATE".rjust(width_date - 1)  # DATE ends with the date: DSSAT's SRAD span starts after it
        + "".join(f"{c.upper():>{w}}" for c, w in zip(vars_, widths, strict=True))
    )
    for i, d in enumerate(dates):
        code = f"{d.year:04d}{d.dayofyear:03d}" if four_digit_year else f"{d.year % 100:02d}{d.dayofyear:03d}"
        out.append(code + "".join(f"{col[i]:>{w}}" for col, w in zip(texts, widths, strict=True)))
    p = Path(path)
    p.write_text("\n".join(out) + "\n", encoding="latin-1", errors="replace")
    return p
