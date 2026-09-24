"""Small array kernels shared by the CERES-Maize processes.

``curv_lin`` and ``tabex`` follow DSSAT-CSM ``Utilities/UTILS.for`` (``CURV`` with type ``LIN``,
``TABEX``); ``daylength`` and ``twilight_daylength`` follow ``Weather/SOLAR.for`` (``DAYLEN``,
``TWILIGHT``). DSSAT-CSM is BSD-3 (DSSAT Foundation, University of Florida, IFDC).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from jaxtyping import Array

__all__ = [
    "curv_lin",
    "daylength",
    "safe_div",
    "tabex",
    "trunc_st",
    "twilight_daylength",
]


def safe_div(num: ArrayLike, den: ArrayLike, fill: ArrayLike = 0.0) -> Array:
    """``num / den`` where ``den != 0``, else ``fill``; both branches finite for any input.

    Source: guard for the divisions DSSAT-CSM protects with ``IF (x .GT. 0.0)``.
    """
    den = jnp.asarray(den)
    ok = den != 0.0
    return jnp.where(ok, jnp.asarray(num) / jnp.where(ok, den, 1.0), fill)


def trunc_st(x: ArrayLike) -> Array:
    """``INT``-style truncation toward zero with an identity (straight-through) derivative.

    The value is exactly ``trunc(x)`` (``x + (trunc(x) - x)`` rounds back to ``trunc(x)``: the
    difference is exact by Sterbenz' lemma for ``|x| >= 1`` and ``-x`` below), so outputs agree
    with DSSAT's ``REAL(INT(x))``; the derivative is 1 instead of 0 so that gradients flow through
    the 1e-3 quantisation DSSAT applies to TURFAC and RLV.

    Source: DSSAT-CSM MZ_GROSUB.for (TURFAC), MZ_ROOTS.for (RLV) ``REAL(INT(x*1000))/1000``.
    """
    x = jnp.asarray(x)
    return x + lax.stop_gradient(jnp.trunc(x) - x)


def curv_lin(xb: ArrayLike, x1: ArrayLike, x2: ArrayLike, xm: ArrayLike, x: ArrayLike) -> Array:
    """Trapezoidal response ``CURV('LIN', XB, X1, X2, XM, X)``: 0 below ``xb`` and above ``xm``,
    1 between ``x1`` and ``x2``, linear in between, clipped to ``[0, 1]``.

    Source: DSSAT-CSM Utilities/UTILS.for, FUNCTION CURV (CTYPE 'LIN').
    """
    x = jnp.asarray(x)
    up = (x - xb) / jnp.maximum(jnp.asarray(x1) - xb, 1e-6)
    down = 1.0 - (x - x2) / jnp.maximum(jnp.asarray(xm) - x2, 1e-6)
    out = jnp.where(
        (x > xb) & (x < x1),
        up,
        jnp.where((x >= x1) & (x <= x2), 1.0, jnp.where((x > x2) & (x < xm), down, 0.0)),
    )
    return jnp.clip(out, 0.0, 1.0)


def tabex(val: ArrayLike, arg: ArrayLike, x: ArrayLike) -> Array:
    """Piecewise-linear table lookup with linear extrapolation beyond the ends (``TABEX``).

    ``arg`` and ``val`` are ``[..., K]`` with increasing ``arg``; the segment is the first
    ``J >= 2`` with ``x <= ARG(J)`` (``J = K`` beyond the last point).

    Source: DSSAT-CSM Utilities/UTILS.for, FUNCTION TABEX.
    """
    val = jnp.asarray(val)
    arg = jnp.asarray(arg)
    x = jnp.asarray(x)
    k = arg.shape[-1]
    xe = x[..., None]
    # 0-based upper index of the segment: 1 + count(arg[1:K-1] < x), in 1..K-1
    j = 1 + jnp.sum(xe > arg[..., 1 : k - 1], axis=-1)
    hi = jnp.arange(k) == j[..., None]
    lo = jnp.arange(k) == (j - 1)[..., None]
    a_hi = jnp.sum(jnp.where(hi, arg, 0.0), axis=-1)
    a_lo = jnp.sum(jnp.where(lo, arg, 0.0), axis=-1)
    v_hi = jnp.sum(jnp.where(hi, val, 0.0), axis=-1)
    v_lo = jnp.sum(jnp.where(lo, val, 0.0), axis=-1)
    return (x - a_lo) * (v_hi - v_lo) / jnp.maximum(a_hi - a_lo, 1e-6) + v_lo


def daylength(doy: ArrayLike, lat_deg: ArrayLike) -> Array:
    """Astronomical daylength [h] (``DAYLEN``), clipped to ``[0, 24]``.

    Source: DSSAT-CSM Weather/SOLAR.for, SUBROUTINE DAYLEN (Spitters 1986 eqs. 16-17, PI = 3.14159).
    """
    pi = 3.14159
    rad = pi / 180.0
    dec = -23.45 * jnp.cos(2.0 * pi * (jnp.asarray(doy) + 10.0) / 365.0)
    soc = jnp.clip(jnp.tan(rad * dec) * jnp.tan(rad * jnp.asarray(lat_deg)), -1.0, 1.0)
    return jnp.clip(12.0 + 24.0 * jnp.arcsin(soc) / pi, 0.0, 24.0)


def twilight_daylength(doy: ArrayLike, lat_deg: ArrayLike) -> Array:
    """Twilight-to-twilight daylength [h] (``TWILIGHT``) used by the maize photoperiod response.

    Source: DSSAT-CSM Weather/SOLAR.for, SUBROUTINE TWILIGHT.
    """
    lat = jnp.asarray(lat_deg)
    s1 = jnp.sin(lat * 0.01745)
    c1 = jnp.cos(lat * 0.01745)
    dec = 0.4093 * jnp.sin(0.0172 * (jnp.asarray(doy) - 82.2))
    dlv = (-s1 * jnp.sin(dec) - 0.1047) / (c1 * jnp.cos(dec))
    dlv = jnp.clip(dlv, -0.87, 1.0)
    return 7.639 * jnp.arccos(dlv)
