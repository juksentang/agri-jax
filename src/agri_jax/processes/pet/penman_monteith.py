"""ASCE standardized reference evapotranspiration (short and tall reference), daily.

Independent JAX implementation of

* Allen, R.G., Walter, I.A., Elliott, R.L., Howell, T.A., Itenfisu, D., Jensen, M.E. and
  Snyder, R.L. (eds.) (2005). The ASCE Standardized Reference Evapotranspiration Equation.
  ASCE, Reston. Equations are cited by their number in that report.

RZWQM2 evaluates the same equation in ``REF_ET.FOR`` (subroutine ``REF_ET``, daily branch
``ihourly = 0``) and writes the results to ``.ana`` columns 81 (tall) and 82 (short) in cm.
That code was read to identify its small departures from the ASCE text; the ``variant``
switch reproduces them so that the function can be compared against the reference model
(``"rzwqm"``) or against textbook / ``pyet`` values (``"asce"``, the default):

=================  ==============================  =================================
quantity           ``"asce"``                      ``"rzwqm"`` (REF_ET.FOR)
=================  ==============================  =================================
``dr``, ``delta``  ``2 pi J / 365`` (eqs. 50-51)   ``0.0172 J``
``e_a``            ``RH/100 (e_s(Tmax)+e_s(Tmin))/2``  ``RH/100 e_s(Tmean)``
``sigma``          4.903e-9 MJ m-2 K-4 d-1         4.901e-9
``T_K`` in eq. 1   ``T + 273``                     ``T + 273.16``
=================  ==============================  =================================

Both variants clip ``Rs/Rso`` to [0.3, 1] (eq. 45), use ``Rns = 0.77 Rs`` (eq. 43),
``Rso = (0.75 + 2e-5 z) Ra`` (eq. 47), ``G = 0`` (eq. 30), ``P = 101.3 ((293 - 0.0065 z)/293)^5.26``
(eq. 34), ``gamma = 0.000665 P`` (eq. 35), ``Delta = 2503 exp(17.27 T/(T+237.3))/(T+237.3)^2``
(eq. 36) and the wind log-law ``u2 = u_z 4.87 / ln(67.8 z_w - 5.42)`` (eq. 33).

Units: temperatures degC, ``srad`` MJ m-2 d-1, ``rh`` percent, ``wind`` m s-1 at ``wind_height`` m,
elevation m, latitude radians; the result is in **mm d-1** (RZWQM stores it /10 in cm).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

__all__ = ["ASCE_SHORT", "ASCE_TALL", "ReferenceET", "asce_reference_et", "wind_to_2m"]

#: (Cn [K mm s3 Mg-1 d-1], Cd [s m-1]) for the daily time step, ASCE Table 1.
ASCE_SHORT = (900.0, 0.34)
ASCE_TALL = (1600.0, 0.38)

_SIGMA = {"asce": 4.903e-9, "rzwqm": 4.901e-9}
_DAY_ANGLE = {"asce": 2.0 * jnp.pi / 365.0, "rzwqm": 0.0172}
_T_KELVIN_ET = {"asce": 273.0, "rzwqm": 273.16}


class ReferenceET(NamedTuple):
    """Outputs of :func:`asce_reference_et` (mm d-1 and MJ m-2 d-1)."""

    et_short: Array  # ETos: 12-cm clipped grass [mm d-1]
    et_tall: Array  # ETrs: 50-cm alfalfa [mm d-1]
    rn: Array  # net radiation [MJ m-2 d-1]
    ra: Array  # extraterrestrial radiation [MJ m-2 d-1]
    rso: Array  # clear-sky radiation [MJ m-2 d-1]
    es: Array  # saturation vapour pressure [kPa]
    ea: Array  # actual vapour pressure [kPa]
    u2: Array  # wind at 2 m [m s-1]


def _es(t: Array) -> Array:
    """Tetens saturation vapour pressure [kPa], ASCE eq. 7."""
    return 0.6108 * jnp.exp(17.27 * t / (t + 237.3))


def wind_to_2m(wind: ArrayLike, wind_height: ArrayLike = 2.0) -> Array:
    """Wind speed at 2 m from the speed at ``wind_height`` m over clipped grass (ASCE eq. 33).

    Source: REF_ET.FOR ``U2_hr = hru1*4.87/log(67.8*xw-5.42)``.
    """
    return jnp.asarray(wind) * 4.87 / jnp.log(67.8 * jnp.asarray(wind_height) - 5.42)


def asce_reference_et(
    tmin: ArrayLike,
    tmax: ArrayLike,
    srad: ArrayLike,
    rh: ArrayLike,
    wind: ArrayLike,
    *,
    elevation: ArrayLike,
    latitude: ArrayLike,
    doy: ArrayLike,
    wind_height: ArrayLike = 2.0,
    trat: ArrayLike = 1.0,
    variant: Literal["asce", "rzwqm"] = "asce",
) -> ReferenceET:
    """Daily ASCE standardized reference ET for the short and tall reference surfaces [mm d-1].

    ``ET = (0.408 Delta (Rn - G) + gamma Cn/(T + 273) u2 (es - ea)) / (Delta + gamma (1 + Cd u2 trat))``
    (ASCE eq. 1) with ``T`` the mean of ``tmin`` and ``tmax``. ``trat`` multiplies ``Cd u2`` as in
    the reference model (DSSAT ``TRATIO`` CO2 factor, 1.0 at 330 ppm); it is not part of the ASCE
    equation. Radiation: ``Ra`` from eqs. 21-24 with ``dr`` and declination per ``variant``;
    ``Rnl = sigma fcd (0.34 - 0.14 sqrt(ea)) (Tmax_K^4 + Tmin_K^4)/2`` (eqs. 17, 19) with
    ``T_K = T + 273.16``, ``fcd = 1.35 Rs/Rso - 0.35``.

    Known deviations from ASCE (2005): none in ``variant="asce"`` beyond the ``trat`` hook and the
    mean-RH vapour pressure (eq. 17 of FAO-56 rather than eq. 11 of ASCE, which needs RHmax/RHmin);
    ``variant="rzwqm"`` reproduces REF_ET.FOR (table in the module docstring). The hourly branch
    of REF_ET.FOR (``fcd`` carried over night) is not implemented.

    Source: REF_ET.FOR (RZWQM2 4.5) lines 15-357, daily branch; ASCE-EWRI (2005).
    """
    tmin = jnp.asarray(tmin, dtype=float)
    tmax = jnp.asarray(tmax, dtype=float)
    rs = jnp.asarray(srad, dtype=float)
    rh_frac = jnp.asarray(rh, dtype=float) / 100.0
    z = jnp.asarray(elevation, dtype=float)
    lat = jnp.asarray(latitude, dtype=float)
    j = jnp.asarray(doy, dtype=float)
    k_day = _DAY_ANGLE[variant]

    u2 = wind_to_2m(wind, wind_height)
    tmean = 0.5 * (tmin + tmax)
    pressure = 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26
    gamma = 0.000665 * pressure
    delta = 2503.0 * jnp.exp(17.27 * tmean / (tmean + 237.3)) / (tmean + 237.3) ** 2
    es = 0.5 * (_es(tmax) + _es(tmin))
    ea = rh_frac * (_es(tmean) if variant == "rzwqm" else es)

    dr = 1.0 + 0.033 * jnp.cos(k_day * j)
    sd = 0.409 * jnp.sin(k_day * j - 1.39)
    ws = jnp.arccos(jnp.clip(-jnp.tan(lat) * jnp.tan(sd), -1.0, 1.0))
    ra = (
        (24.0 / jnp.pi)
        * 4.92
        * dr
        * (ws * jnp.sin(lat) * jnp.sin(sd) + jnp.cos(lat) * jnp.cos(sd) * jnp.sin(ws))
    )
    rso = (0.75 + 2.0e-5 * z) * ra
    relsol = jnp.where(rso > 1.0e-10, rs / jnp.maximum(rso, 1.0e-10), 0.0)
    relsol = jnp.clip(relsol, 0.3, 1.0)
    fcd = 1.35 * relsol - 0.35
    rns = 0.77 * rs
    tk4 = 0.5 * ((tmax + 273.16) ** 4 + (tmin + 273.16) ** 4)
    rnl = _SIGMA[variant] * fcd * (0.34 - 0.14 * jnp.sqrt(jnp.maximum(ea, 0.0))) * tk4
    rn = rns - rnl
    g = 0.0

    tk = tmean + _T_KELVIN_ET[variant]
    aero = gamma * u2 * (es - ea) / tk
    rad = 0.408 * delta * (rn - g)

    def _et(cn: float, cd: float) -> Array:
        return (rad + cn * aero) / (delta + gamma * (1.0 + cd * u2 * jnp.asarray(trat)))

    et_short = _et(*ASCE_SHORT)
    et_tall = _et(*ASCE_TALL)
    return ReferenceET(et_short, et_tall, rn, ra, rso, es, ea, u2)
