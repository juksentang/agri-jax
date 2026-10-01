"""Tolerant parsing of Fortran-formatted numbers (``1.0D-03``, ``0.1234-100``, ``*****``) and dates."""

from __future__ import annotations

import re

import numpy as np

_EXP_NO_E = re.compile(r"^([+-]?\d*\.?\d+)([+-]\d{2,3})$")


def fortran_float(tok: str) -> float:
    """Parse one Fortran list-directed / formatted real; overflow fields become NaN."""
    try:
        return float(tok)
    except ValueError:
        pass
    t = tok.strip().replace("D", "E").replace("d", "e")
    if not t or set(t) <= {"*"} or t.upper() in {"NAN", "-NAN"}:
        return float("nan")
    m = _EXP_NO_E.match(t)
    if m:
        return float(f"{m.group(1)}e{m.group(2)}")
    return float(t)


def year_doy_to_datetime64(year: object, doy: object) -> np.ndarray:
    """``datetime64[D]`` of ``Jan 1 of year + (doy - 1)`` days (doy 0 -> Dec 31 of year - 1)."""
    y = np.asarray(year, dtype=np.int64)
    d = np.asarray(doy, dtype=np.int64)
    jan1 = (y - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    return jan1 + (d - 1).astype("timedelta64[D]")
