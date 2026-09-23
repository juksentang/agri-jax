"""Readers for DSSAT ``*.OUT`` files (PlantGro, SoilWat, ET, Summary, ...).

Every table is parsed by the positions of its ``@`` header line (see
:mod:`agri_jax.io.dssat._fixed`), never by splitting on whitespace, so text columns such as
``TNAM`` (``RAINFED LOW NITROGEN``) and blank columns stay aligned.

Daily files hold one block per run (``*RUN   n`` ... ``TREATMENT  n`` ... ``@YEAR DOY``).
All blocks are concatenated into one DataFrame with ``RUN`` and ``TRNO`` columns and, when
``YEAR``/``DOY`` exist, a ``DATE`` column (``datetime64``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ._fixed import frame_from_rows, header_tokens, read_lines, split_fixed

__all__ = [
    "read_et",
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
