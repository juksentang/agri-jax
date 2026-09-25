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
from agrijax.core.units import HOURS_PER_DAY

from .constants import (
    DEG_PER_HALF_TURN,
    FULL_TURN_PER_PI,
    HOURS_PER_HALF_DAY,
    SOLAR_COEFFICIENTS,
    SolarCoefficients,
)

__all__ = [
    "curv_lin",
    "daylength",
    "safe_div",
    "tabex",
    "trunc_st",
    "twilight_daylength",
]


def daylength(doy: ArrayLike, lat_deg: ArrayLike, c: SolarCoefficients = SOLAR_COEFFICIENTS) -> Array:
    """Astronomical daylength [h] (``DAYLEN``), clipped to ``[0, 24]``.

    ``c`` holds the sun geometry (:class:`~.constants.SolarCoefficients`, default the DSSAT
    values, ``PI = 3.14159``).

    Source: DSSAT-CSM Weather/SOLAR.for, SUBROUTINE DAYLEN (Spitters 1986 eqs. 16-17, PI = 3.14159).
    """
    pi = c.pi
    rad = pi / DEG_PER_HALF_TURN
    dec = -c.dec_amplitude * jnp.cos(
        FULL_TURN_PER_PI * pi * (jnp.asarray(doy) + c.dec_doy_offset) / c.year_days
    )
    soc = jnp.clip(jnp.tan(rad * dec) * jnp.tan(rad * jnp.asarray(lat_deg)), -1.0, 1.0)
    return jnp.clip(HOURS_PER_HALF_DAY + HOURS_PER_DAY * jnp.arcsin(soc) / pi, 0.0, HOURS_PER_DAY)


def twilight_daylength(
    doy: ArrayLike, lat_deg: ArrayLike, c: SolarCoefficients = SOLAR_COEFFICIENTS
) -> Array:
    """Twilight-to-twilight daylength [h] (``TWILIGHT``) used by the maize photoperiod response.

    ``c`` holds the sun geometry (:class:`~.constants.SolarCoefficients`, default the DSSAT values).

    Source: DSSAT-CSM Weather/SOLAR.for, SUBROUTINE TWILIGHT.
    """
    lat = jnp.asarray(lat_deg)
    s1 = jnp.sin(lat * c.tw_deg_to_rad)
    c1 = jnp.cos(lat * c.tw_deg_to_rad)
    dec = c.tw_dec_amplitude * jnp.sin(c.tw_dec_rate * (jnp.asarray(doy) - c.tw_dec_doy0))
    dlv = (-s1 * jnp.sin(dec) - c.tw_sin_depression) / (c1 * jnp.cos(dec))
    dlv = jnp.clip(dlv, c.tw_dlv_min, 1.0)
    return c.tw_hours_per_rad * jnp.arccos(dlv)
