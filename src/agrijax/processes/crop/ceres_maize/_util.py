"""Small array kernels shared by the CERES-Maize processes.

``daylength`` and ``twilight_daylength`` follow DSSAT-CSM ``Weather/SOLAR.for`` (``DAYLEN``,
``TWILIGHT``). ``safe_div``, ``trunc_st``, ``curv_lin`` and ``tabex`` live in
:mod:`agrijax.core.grad` and are re-exported here unchanged. DSSAT-CSM is BSD-3 (DSSAT
Foundation, University of Florida, IFDC).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.grad import curv_lin, safe_div, tabex, trunc_st

__all__ = [
    "curv_lin",
    "daylength",
    "safe_div",
    "tabex",
    "trunc_st",
    "twilight_daylength",
]


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
