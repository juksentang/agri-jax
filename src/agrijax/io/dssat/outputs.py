"""Readers for DSSAT ``*.OUT`` files (PlantGro, SoilWat, ET, Summary, ...).

Every table is parsed by the positions of its ``@`` header line (see
:mod:`agrijax.io.dssat._fixed`), never by splitting on whitespace, so text columns such as
``TNAM`` (``RAINFED LOW NITROGEN``) and blank columns stay aligned.

Daily files hold one block per run (``*RUN   n`` ... ``TREATMENT  n`` ... ``@YEAR DOY``).
All blocks are concatenated into one DataFrame with ``RUN`` and ``TRNO`` columns and, when
``YEAR``/``DOY`` exist, a ``DATE`` column (``datetime64``).
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from ._fixed import frame_from_rows, header_tokens, read_lines, split_fixed
from .observed import reada_date

__all__ = [
    "observed_date",
    "read_et",
    "read_evaluate",
    "read_out",
    "read_plantgro",
    "read_soilwat",
    "read_summary",
]

_RUN = re.compile(r"^\*RUN\s+(\d+)")
_TRT = re.compile(r"^\s*TREATMENT\s+(\d+)")


def read_out(path: str | Path, *, missing_to_nan: bool = True) -> pd.DataFrame:
    """Parse every ``@``-headed table of a DSSAT OUT file into one DataFrame.

    Blocks with differing headers are concatenated (union of columns). ``-99`` becomes NaN
    unless ``missing_to_nan=False``.
    """
    lines = read_lines(path)
    frames: list[pd.DataFrame] = []
    run: int | None = None
    trno: int | None = None
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        m = _RUN.match(ln)
        if m:
            run = int(m.group(1))
            trno = None
        m = _TRT.match(ln)
        if m:
            trno = int(m.group(1))
        if ln.startswith("@"):
            toks = header_tokens(ln)
            names = [t[0] for t in toks]
            rows: list[list[str]] = []
            i += 1
            while i < n:
                d = lines[i]
                if not d.strip() or d.startswith(("@", "*")):
                    break
                if not d.lstrip().startswith("!"):
                    rows.append(split_fixed(toks, d))
                i += 1
            if rows:
                df = frame_from_rows(names, rows, missing_to_nan=missing_to_nan)
                if "RUN" not in df.columns and "RUNNO" not in df.columns and run is not None:
                    df.insert(0, "RUN", run)
                if "TRNO" not in df.columns and trno is not None:
                    df.insert(1 if "RUN" in df.columns else 0, "TRNO", trno)
                frames.append(df)
            continue
        i += 1
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "YEAR" in out.columns and "DOY" in out.columns:
        out["DATE"] = pd.to_datetime(
            out["YEAR"].astype(int).astype(str) + out["DOY"].astype(int).astype(str).str.zfill(3),
            format="%Y%j",
        )
    return out


def read_plantgro(path: str | Path, **kw: bool) -> pd.DataFrame:
    """``PlantGro.OUT``: daily crop growth (LAID, CWAD, GWAD, ...)."""
    return read_out(path, **kw)


def read_soilwat(path: str | Path, **kw: bool) -> pd.DataFrame:
    """``SoilWat.OUT``: daily soil water (SWTD, PREC, SW1D..SWnD, ...)."""
    return read_out(path, **kw)


def read_et(path: str | Path, **kw: bool) -> pd.DataFrame:
    """``ET.OUT``: daily evapotranspiration (EOAA, ETAA, EPAA, ESAA, ...)."""
    return read_out(path, **kw)


def read_summary(path: str | Path, **kw: bool) -> pd.DataFrame:
    """``Summary.OUT``: one row per run (RUNNO, TRNO, TNAM, HWAM, MDAT, ...).

    Date columns (``SDAT``, ``PDAT``, ``ADAT``, ``MDAT``, ``HDAT``...) stay ``YYYYDDD``
    integers (NaN when missing).
    """
    return read_out(path, **kw)


def read_evaluate(path: str | Path, **kw: bool) -> pd.DataFrame:
    """``Evaluate.OUT``: one row per run, simulated/measured pairs (``HWAMS``/``HWAMM``,
    ``ADAPS``/``ADAPM`` days after planting, ``CWAMS``/``CWAMM`` ...) keyed by ``RUN`` and ``TN``."""
    return read_out(path, **kw)


def observed_date(
    code: int | str, sim_start: int | str | date, *, first_weather: date | str | int | None = None
) -> date:
    """Calendar date of an observed-data date (``ADAT``, ``MDAT`` ... of a ``.MZA`` file).

    The one implementation of DSSAT's ``READA_Dates`` (``READS.for``) is
    :func:`agrijax.io.dssat.observed.reada_date`, which this calls: a value below 1000 is a day of
    year, in the year of the simulation start when it is later than the start day of year, else in
    the next year; ``YYDDD`` / ``YYYYDDD`` values are full dates through ``Y4K_DOY``
    (:func:`~agrijax.io.dssat.observed.y4k_date`, with ``first_weather`` when the weather file has
    four-digit years). ``sim_start`` is ``SDAT`` (``YYYYDDD``, or ``YYDDD`` converted by ``Y4K_DOY``
    like any other ``YYDDD`` code) or a date. A value that is not a date
    (not positive, or not finite) and a day of year its year does not have (366 in a non-leap year)
    raise ``ValueError``; a day of year is never rolled over into the next year.
    """
    d = reada_date(float(code), sim_start, first_weather)
    if d is None:
        raise ValueError(f"not an observed date: {code!r}")
    return d
