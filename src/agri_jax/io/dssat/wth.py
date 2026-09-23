"""DSSAT weather files (``*.WTH``): read and write.

``read_wth`` returns a DataFrame with a ``date`` column (``datetime64``) and lower-cased
variables (``srad`` MJ m-2 d-1, ``tmax``/``tmin`` degC, ``rain`` mm, and whatever else the
file carries: ``rhum`` %, ``wind`` km d-1, ``par``, ``dewp`` ...). The site line
(``INSI LAT LONG ELEV TAV AMP REFHT WNDHT``) is stored in ``df.attrs["site"]`` and the title
in ``df.attrs["title"]``.

Dates may be ``YYDDD`` (DSSAT Y2K rule: ``YY <= 40`` -> 20YY, else 19YY, as ``Y2K_DOY``)
or ``YYYYDDD``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._fixed import convert, fmt_num, frame_from_rows, header_tokens, read_lines, split_fixed

__all__ = ["parse_dssat_date", "read_wth", "write_wth"]

SITE_COLUMNS = ("INSI", "LAT", "LONG", "ELEV", "TAV", "AMP", "REFHT", "WNDHT")


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


def read_wth(path: str | Path) -> pd.DataFrame:
    """Read a ``.WTH`` file into a daily DataFrame (see module docstring)."""
    lines = read_lines(path)
    title = next((ln for ln in lines if ln.startswith("*")), "")
    site: dict[str, Any] = {}
    names: list[str] = []
    rows: list[list[str]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("@"):
            toks = header_tokens(ln)
            tnames = [t[0] for t in toks]
            if "DATE" in tnames:
                names = tnames
                for d in lines[i + 1 :]:
                    if not d.strip() or d.lstrip().startswith("!"):
                        continue
                    if d.startswith(("@", "*")):
                        break
                    rows.append(split_fixed(toks, d))
                break
            # site header: first non-comment data line
            for d in lines[i + 1 :]:
                if d.strip() and not d.lstrip().startswith("!"):
                    vals = split_fixed(toks, d)
                    site = {n: _site_value(v) for n, v in zip(tnames, vals, strict=True)}
                    break
        i += 1
    if not names:
        raise ValueError(f"{path}: no '@DATE' table found")
    df = frame_from_rows(names, rows, missing_to_nan=True)
    dates = [parse_dssat_date(r[names.index("DATE")]) for r in rows]
    df = df.drop(columns=["DATE"])
    df.columns = [c.lower() for c in df.columns]
    df.insert(0, "date", pd.to_datetime(dates))
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    df.attrs["site"] = site
    df.attrs["title"] = title
    return df


def _wth_cell(v: float) -> str:
    """One six-column ``.WTH`` value: 1, 2 or 3 decimals, the fewest that round-trip exactly.

    A leading blank is kept when it fits; a value that needs all six columns touches the
    previous one (``77.41004.0``), as DSSAT itself writes and :func:`read_wth` reads. Values that
    fit none of these fall back to :func:`fmt_num` (shortest exact text in six columns).
    """
    if np.isnan(v):
        return "   -99"
    for room in (5, 6):
        for d in (1, 2, 3):
            t = f"{v:.{d}f}"
            if len(t) <= room and abs(float(t) - v) <= 1e-9 * max(1.0, abs(v)):
                return f"{t:>6}"
    return fmt_num(float(v), 6)


def write_wth(
    df: pd.DataFrame,
    path: str | Path,
    *,
    site: Mapping[str, Any] | None = None,
    title: str | None = None,
    four_digit_year: bool = False,
) -> Path:
    """Write a DataFrame (``date`` + variables, as returned by :func:`read_wth`) to ``.WTH``.

    ``site`` defaults to ``df.attrs["site"]``. Each variable takes six columns with the fewest
    decimals (at least one) that reproduce the value exactly, so ``12.35`` stays ``12.35`` and
    ``8.4`` stays ``8.4`` (:func:`_wth_cell`); NaN -> ``-99``. Dates as ``YYDDD`` unless
    ``four_digit_year``.
    """
    site = dict(site if site is not None else df.attrs.get("site", {}))
    title = str(title if title is not None else df.attrs.get("title", "*WEATHER DATA :"))
    out = [title if title.startswith("*") else "*WEATHER DATA : " + title, ""]
    out.append("@ INSI      LAT     LONG  ELEV   TAV   AMP REFHT WNDHT")
    insi = site.get("INSI") or ""

    def g(key: str) -> float:
        v = site.get(key)
        return math.nan if v is None or v == "" else float(v)

    out.append(
        f"  {insi!s:<4}"
        + f"{fmt_num(g('LAT'), 9, 3)}{fmt_num(g('LONG'), 9, 3)}{fmt_num(g('ELEV'), 6, 0)}"
        + f"{fmt_num(g('TAV'), 6, 1)}{fmt_num(g('AMP'), 6, 1)}"
        + f"{fmt_num(g('REFHT'), 6, 2)}{fmt_num(g('WNDHT'), 6, 2)}"
    )
    vars_ = [c for c in df.columns if c != "date"]
    width_date = 7 if four_digit_year else 5
    out.append("@" + "DATE".ljust(width_date - 1) + "".join(f"{c.upper():>6}" for c in vars_))
    dates = pd.to_datetime(df["date"])
    values = df[vars_].to_numpy(dtype=float)
    for d, row in zip(dates, values, strict=True):
        code = f"{d.year:04d}{d.dayofyear:03d}" if four_digit_year else f"{d.year % 100:02d}{d.dayofyear:03d}"
        out.append(code + "".join(_wth_cell(v) for v in row))
    p = Path(path)
    p.write_text("\n".join(out) + "\n", encoding="latin-1", errors="replace")
    return p
