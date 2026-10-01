"""Host-side: the Fortran formatted I/O and single-precision arithmetic of ``dscsm048`` (NumPy).

DSSAT-CSM passes most inputs from its input module to the simulation modules through the text file
``DSSAT48.INP``: a value is written with an edit descriptor ``Fw.d`` and read back as ``REAL``. The
number a module holds is therefore the ``REAL*4`` of the *printed* decimal, not of the value the
input module computed. These helpers reproduce that round trip and the ``REAL`` (float32)
arithmetic of the routines ported natively (:mod:`agrijax.io.dssat.native_soil`,
:mod:`agrijax.io.dssat.native_weather`):

* :func:`write_f` -- the ``Fw.d`` output of a ``REAL`` (gfortran: the exact binary value rounded to
  ``d`` decimals, round-half-even, as ``printf``; a field too narrow for the value is ``w``
  asterisks, as in Fortran);
* :func:`read_f` -- the ``Fw.d`` / list-directed input of a field (an explicit decimal point wins
  over ``d``; a blank field is 0);
* :func:`f32` -- a value as ``REAL*4`` (the correctly rounded float32 of a decimal);
* :func:`round_trip` -- ``read_f(write_f(x, w, d))``;
* :func:`nint` -- Fortran ``NINT`` (half away from zero).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["decimal_of", "f32", "nint", "read_f", "round_trip", "write_f"]

F32 = np.float32


def f32(x: Any) -> np.float32:
    """Host-side: ``x`` as a ``REAL*4`` scalar (a string is parsed as a decimal first)."""
    if isinstance(x, str):
        return F32(float(x))
    return F32(x)


def write_f(x: Any, w: int, d: int) -> str:
    """Host-side: Fortran ``Fw.d`` output of the ``REAL`` ``x`` (right-aligned in ``w`` columns).

    gfortran prints the exact binary value rounded to ``d`` decimals (ties to even, what
    ``printf`` does); ``d = 0`` keeps the decimal point (``5.`` for 5). A value that does not fit
    prints ``w`` asterisks. A leading zero is dropped when the field has no room for it
    (``.123`` in ``F4.3``)."""
    v = float(np.float32(x))
    text = f"{v:.{d}f}"
    if d == 0:
        text += "."
    if len(text) > w:
        if text.startswith("0.") and len(text) - 1 <= w:
            text = text[1:]
        elif text.startswith("-0.") and len(text) - 1 <= w:
            text = "-" + text[2:]
        else:
            return "*" * w
    return text.rjust(w)


def read_f(field: str, d: int = 0) -> np.float32:
    """Host-side: Fortran ``Fw.d`` input of one field as ``REAL*4``: a number with a decimal point
    (or an exponent) is read as written; digits alone are scaled by ``10**-d``; blanks are 0."""
    t = field.strip()
    if not t:
        return F32(0.0)
    if "*" in t:
        raise ValueError(f"field {field!r} is an overflowed Fortran output (asterisks)")
    if "." in t or "e" in t.lower():
        return F32(float(t))
    return F32(int(t) / 10**d)


def round_trip(x: Any, w: int, d: int) -> np.float32:
    """Host-side: the ``REAL*4`` a module reads back after ``x`` was written with ``Fw.d``."""
    return read_f(write_f(x, w, d), d)


def nint(x: Any) -> int:
    """Host-side: Fortran ``NINT``: the nearest integer, halves away from zero."""
    v = float(x)
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def decimal_of(x: Any) -> np.ndarray:
    """Host-side: the shortest decimal that rounds to each ``REAL*4`` value (the number as written
    in an input file), as float64."""
    a = np.asarray(x, dtype=np.float32)
    return np.asarray([float(str(v)) for v in a.reshape(-1)], dtype=np.float64).reshape(a.shape)
