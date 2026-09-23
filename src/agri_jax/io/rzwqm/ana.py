"""Reader for the RZWQM2 daily analysis output (``.ana``).

Layout (RZWQM2 4.5 binary, CA-TPA): a header of 20 lines listing ``N) NAME (UNITS)`` in
seven fixed-width columns (N = 1..139, where 1 is ``TIME (YEAR.DAY)``), a blank line, a
``(1) (2) ...`` line and a ``=====`` rule (23 header lines in total), then one whitespace-
separated row per day: ``YYYY.DDD`` followed by the other 138 variables. The first row of a
run is ``YYYY.000``, the initial state before day 1. Rows are converted to dates as
``Jan 1 of YYYY + (DDD - 1) days`` (so ``2015.000`` is 2014-12-31).

Variable names are slugified from the header (``STORED SOIL WATER (CM)`` ->
``stored_soil_water``, units ``CM`` in ``attrs["units"]``; the 1-based header number is in
``attrs["column"]``). :data:`KEY_COLUMNS` names the columns the PoC compares against.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ._fortran import fortran_float, year_doy_to_datetime64

__all__ = ["KEY_COLUMNS", "key_variables", "parse_ana_header", "read_ana", "slugify"]

KEY_COLUMNS: dict[str, int] = {
    "profile_water_cm": 2,
    "precip_cm": 3,
    "evap_cm": 6,
    "transp_cm": 7,
    "pot_evap_cm": 8,
    "pot_transp_cm": 9,
    "lai": 43,
    "grain_kg_ha": 44,
    "pet_cm": 83,
    "aet_cm": 84,
}
"""Short name -> 1-based ``.ana`` column (column 1 is the ``YYYY.DDD`` time)."""

# "N)" starting a header entry: at line start, after whitespace or right after a ")" that
# closed the previous entry's units (e.g. "(KG/HA/DAY)68) SEED ..."); never "(0-1)".
_ENTRY_RE = re.compile(r"(?:(?<=^)|(?<=\s)|(?<=\)))(\d{1,3})\)")
_UNITS_RE = re.compile(r"^(?P<name>.*?)\s*\((?P<units>[^()]*)\)\s*$")


def slugify(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")
    if not s:
        s = "var"
    if s[0].isdigit():
        s = "v_" + s
    return s


def parse_ana_header(lines: list[str]) -> tuple[dict[int, str], int]:
    """Header entries ``{column: "NAME (UNITS)"}`` and the number of header lines."""
    n_header = next((i + 1 for i, ln in enumerate(lines) if ln.lstrip().startswith("====")), None)
    if n_header is None:
        raise ValueError("no '=====' rule ending the .ana header")
    entries: dict[int, str] = {}
    for ln in lines[: n_header - 1]:
        if ln.lstrip().startswith("("):  # the "(1) (2) ( 3) ..." line
            continue
        ms = list(_ENTRY_RE.finditer(ln))
        for k, m in enumerate(ms):
            end = ms[k + 1].start() if k + 1 < len(ms) else len(ln)
            col = int(m.group(1))
            if col in entries:
                raise ValueError(f"column {col} appears twice in the .ana header")
            entries[col] = ln[m.end() : end].strip()
    cols = sorted(entries)
    if cols != list(range(1, len(cols) + 1)):
        missing = sorted(set(range(1, max(cols) + 1)) - set(cols))
        raise ValueError(f".ana header columns are not 1..N (missing {missing})")
    return entries, n_header


def _split_units(label: str) -> tuple[str, str]:
    m = _UNITS_RE.match(label)
    if m and m.group("name"):
        return m.group("name"), m.group("units").strip()
    return label, ""


def _time_to_dates(tokens: list[str]) -> pd.DatetimeIndex:
    years, doys = [], []
    for t in tokens:
        y, _, d = t.partition(".")
        years.append(int(y))
        doys.append(int(d) if d else 0)
    return pd.DatetimeIndex(year_doy_to_datetime64(years, doys).astype("datetime64[ns]"))


def read_ana(path: str | Path) -> xr.Dataset:
    """``.ana`` -> Dataset with a ``time`` coordinate and one variable per header column.

    Also carries ``yyyyddd`` (the raw time token) as a coordinate; ``ds.attrs["columns"]`` maps
    1-based column numbers to variable names.
    """
    text = Path(path).read_bytes().decode("latin-1")
    lines = text.splitlines()
    entries, n_header = parse_ana_header(lines)
    ncol = len(entries)
    rows = [ln.split() for ln in lines[n_header:] if ln.strip()]
    bad = [i for i, r in enumerate(rows) if len(r) != ncol]
    if bad:
        raise ValueError(
            f"{path}: data row {bad[0] + 1} has {len(rows[bad[0]])} fields, header declares {ncol}"
        )
    time_tok = [r[0] for r in rows]
    try:
        data = np.array([r[1:] for r in rows], dtype=np.float64)
    except ValueError:
        data = np.array([[fortran_float(t) for t in r[1:]] for r in rows], dtype=np.float64)
    time = _time_to_dates(time_tok)

    names: dict[int, str] = {}
    used: set[str] = {"time", "yyyyddd"}
    data_vars = {}
    for col in range(2, ncol + 1):
        label, units = _split_units(entries[col])
        name = slugify(label)
        if name in used:
            name = f"{name}_c{col}"
        used.add(name)
        names[col] = name
        data_vars[name] = (
            "time",
            data[:, col - 2],
            {"long_name": label, "units": units, "column": col},
        )
    ds = xr.Dataset(
        data_vars,
        coords={"time": time, "yyyyddd": ("time", np.array(time_tok))},
        attrs={"source": str(path), "columns": {str(k): v for k, v in names.items()}},
    )
    return ds


def key_variables(ds: xr.Dataset, columns: dict[str, int] | None = None) -> xr.Dataset:
    """Select the PoC comparison columns and rename them to the short names of :data:`KEY_COLUMNS`."""
    cols = KEY_COLUMNS if columns is None else columns
    by_col = {int(k): v for k, v in ds.attrs["columns"].items()}
    missing = [c for c in cols.values() if c not in by_col]
    if missing:
        raise KeyError(f"columns {missing} not in this .ana file")
    return ds[[by_col[c] for c in cols.values()]].rename({by_col[c]: short for short, c in cols.items()})
