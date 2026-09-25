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

=================  ==================================  ========================
quantity           ``"asce"``                          ``"rzwqm"`` (REF_ET.FOR)
=================  ==================================  ========================
``dr``, ``delta``  ``2 pi J / 365`` (eqs. 23-24)       ``0.0172 J``
``e_a``            ``RH/100 (e_s(Tmax)+e_s(Tmin))/2``  ``RH/100 e_s(Tmean)``
``sigma``          4.903e-9 MJ m-2 K-4 d-1 (FAO-56)    4.901e-9 (ASCE eq. 17)
``T_K`` in eq. 1   ``T + 273``                         ``T + 273.16``
=================  ==================================  ========================

The ``"asce"`` sigma is the FAO-56 value (Allen et al. 1998, eq. 39); ASCE-EWRI (2005) eq. 17
states 4.901e-9, which only the ``"rzwqm"`` variant uses. The ``"asce"`` default keeps 4.903e-9
and records the difference as a deviation.

Both variants clip ``Rs/Rso`` to [0.3, 1] (eq. 18), use ``Rns = 0.77 Rs`` (eq. 16),
``Rso = (0.75 + 2e-5 z) Ra`` (eq. 19), ``G = 0`` (eq. 30), ``P = 101.3 ((293 - 0.0065 z)/293)^5.26``
(eq. 3), ``gamma = 0.000665 P`` (eq. 4), ``Delta = 2503 exp(17.27 T/(T+237.3))/(T+237.3)^2``
(eq. 5) and the wind log-law ``u2 = u_z 4.87 / ln(67.8 z_w - 5.42)`` (eq. 33).

Units: temperatures degC, ``srad`` MJ m-2 d-1, ``rh`` percent, ``wind`` m s-1 at ``wind_height`` m,
elevation m, latitude radians; the result is in **mm d-1** (RZWQM stores it /10 in cm).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.units import HOURS_PER_DAY

from .coefficients import ASCE_2005, ASCECoefficients

__all__ = [
    "ASCE_2005",
    "ASCE_SHORT",
    "ASCE_TALL",
    "ASCECoefficients",
    "ReferenceET",
    "asce_reference_et",
    "wind_to_2m",
]

#: (Cn [K mm s3 Mg-1 d-1], Cd [s m-1]) for the daily time step, ASCE Table 1 (read-only aliases of the
#: defaults declared in :class:`~.coefficients.ASCECoefficients`; the kernel reads the coefficient set)
ASCE_SHORT = (ASCE_2005.cn_short, ASCE_2005.cd_short)
ASCE_TALL = (ASCE_2005.cn_tall, ASCE_2005.cd_tall)

#: default anemometer height [m] (an input default, not a coefficient)
DEFAULT_WIND_HEIGHT_M = 2.0
_TWO_PI = 2.0 * jnp.pi  # full circle of the day angle 2 pi J / 365
_PERCENT_IN_ONE = 100.0  # relative humidity: percent -> fraction (divisor)
_HALF = 0.5  # arithmetic means of two values
_ACOS_MARGIN = numerical_guard(
    "pet.asce.acos_margin", 1.0e-12, "keeps the arccos argument strictly inside (-1, 1) (infinite derivative)"
)
_RSO_FLOOR = numerical_guard(
    "pet.asce.rso_floor", 1.0e-10, "clear-sky radiation below which Rs/Rso is 0 (REF_ET.FOR)"
)
_SQRT_FLOOR = numerical_guard(
    "pet.asce.sqrt_floor", 1.0e-12, "floor of ea under the sqrt (infinite derivative at 0)"
)


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


def _es(t: Array, c: ASCECoefficients) -> Array:
    """Tetens saturation vapour pressure [kPa], ASCE eq. 7."""
    return c.tetens_a * jnp.exp(c.tetens_b * t / (t + c.tetens_c))


def wind_to_2m(
    wind: ArrayLike, wind_height: ArrayLike = DEFAULT_WIND_HEIGHT_M, c: ASCECoefficients = ASCE_2005
) -> Array:
    """Wind speed at 2 m from the speed at ``wind_height`` m over clipped grass (ASCE eq. 33).

    Source: ASCE-EWRI (2005) eq. 33, as REF_ET.FOR (RZWQM2) evaluates it.
    """
    return jnp.asarray(wind) * c.wind_a / jnp.log(c.wind_b * jnp.asarray(wind_height) - c.wind_c)


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
    wind_height: ArrayLike = DEFAULT_WIND_HEIGHT_M,
    trat: ArrayLike = 1.0,
    variant: Literal["asce", "rzwqm"] = "asce",
    coefficients: ASCECoefficients = ASCE_2005,
) -> ReferenceET:
    """Daily ASCE standardized reference ET for the short and tall reference surfaces [mm d-1].

    ``ET = (0.408 Delta (Rn - G) + gamma Cn/(T + 273) u2 (es - ea)) / (Delta + gamma (1 + Cd u2 trat))``
    (ASCE eq. 1) with ``T`` the mean of ``tmin`` and ``tmax``. ``trat`` multiplies ``Cd u2`` as in
    the reference model (DSSAT ``TRATIO`` CO2 factor, 1.0 at 330 ppm); it is not part of the ASCE
    equation. Radiation: ``Ra`` from eqs. 21-24 with ``dr`` and declination per ``variant``;
    ``Rnl = sigma fcd (0.34 - 0.14 sqrt(ea)) (Tmax_K^4 + Tmin_K^4)/2`` (eqs. 17-18) with
    ``T_K = T + 273.16``, ``fcd = 1.35 Rs/Rso - 0.35``.

    Known deviations from ASCE (2005) in ``variant="asce"``: the ``trat`` hook, the mean-RH vapour
    pressure (eq. 19 of FAO-56 rather than eq. 11 of ASCE, which needs RHmax/RHmin) and the
    FAO-56 Stefan-Boltzmann value 4.903e-9 (FAO-56 eq. 39; ASCE eq. 17 states 4.901e-9);
    ``variant="rzwqm"`` reproduces REF_ET.FOR (table in the module docstring). The hourly branch
    of REF_ET.FOR (``fcd`` carried over night) is not implemented.

    ``coefficients`` (:class:`~.coefficients.ASCECoefficients`, default :data:`~.coefficients.ASCE_2005`)
    holds every coefficient; an instance with array leaves makes them differentiable.

    Source: REF_ET.FOR (RZWQM2 4.5) lines 15-357, daily branch; ASCE-EWRI (2005).
    """
    if variant not in ("asce", "rzwqm"):
        raise KeyError(f"variant must be 'asce' or 'rzwqm', got {variant!r}")
    c = coefficients
    rzwqm = variant == "rzwqm"
    tmin = jnp.asarray(tmin, dtype=float)
    tmax = jnp.asarray(tmax, dtype=float)
    rs = jnp.asarray(srad, dtype=float)
    rh_frac = jnp.asarray(rh, dtype=float) / _PERCENT_IN_ONE
    z = jnp.asarray(elevation, dtype=float)
    lat = jnp.asarray(latitude, dtype=float)
    j = jnp.asarray(doy, dtype=float)
    k_day = c.day_angle_rzwqm if rzwqm else _TWO_PI / c.days_per_year

    u2 = wind_to_2m(wind, wind_height, c)
    tmean = _HALF * (tmin + tmax)
    pressure = c.p0 * ((c.t0 - c.lapse_rate * z) / c.t0) ** c.pressure_exp
    gamma = c.psychrometric * pressure
    delta = c.slope_a * jnp.exp(c.tetens_b * tmean / (tmean + c.tetens_c)) / (tmean + c.tetens_c) ** 2
    es = _HALF * (_es(tmax, c) + _es(tmin, c))
    ea = rh_frac * (_es(tmean, c) if rzwqm else es)

    dr = 1.0 + c.eccentricity * jnp.cos(k_day * j)
    sd = c.decl_amp * jnp.sin(k_day * j - c.decl_phase)
    # clipped strictly inside (-1, 1): d/dx arccos is infinite at +-1 (polar day / night)
    ws = jnp.arccos(jnp.clip(-jnp.tan(lat) * jnp.tan(sd), -1.0 + _ACOS_MARGIN, 1.0 - _ACOS_MARGIN))
    ra = (
        (HOURS_PER_DAY / jnp.pi)
        * c.solar_const_hourly
        * dr
        * (ws * jnp.sin(lat) * jnp.sin(sd) + jnp.cos(lat) * jnp.cos(sd) * jnp.sin(ws))
    )
    rso = (c.rso_a + c.rso_b * z) * ra
    relsol = jnp.where(rso > _RSO_FLOOR, rs / jnp.maximum(rso, _RSO_FLOOR), 0.0)
    relsol = jnp.clip(relsol, c.relsol_min, 1.0)
    fcd = c.fcd_a * relsol - c.fcd_b
    rns = c.net_shortwave * rs
    tk4 = _HALF * ((tmax + c.t_kelvin_longwave) ** 4 + (tmin + c.t_kelvin_longwave) ** 4)
    # sqrt floored at 1e-12: d sqrt(x)/dx is infinite at x = 0 (rh = 0), the floor makes it zero
    sigma = c.stefan_boltzmann_rzwqm if rzwqm else c.stefan_boltzmann
    rnl = sigma * fcd * (c.emissivity_a - c.emissivity_b * jnp.sqrt(jnp.maximum(ea, _SQRT_FLOOR))) * tk4
    rn = rns - rnl
    g = 0.0

    tk = tmean + (c.t_kelvin_rzwqm if rzwqm else c.t_kelvin)
    aero = gamma * u2 * (es - ea) / tk
    rad = c.radiation_to_et * delta * (rn - g)

    def _et(cn: ArrayLike, cd: ArrayLike) -> Array:
        return (rad + cn * aero) / (delta + gamma * (1.0 + cd * u2 * jnp.asarray(trat)))

    et_short = _et(c.cn_short, c.cd_short)
    et_tall = _et(c.cn_tall, c.cd_tall)
    return ReferenceET(et_short, et_tall, rn, ra, rso, es, ea, u2)
