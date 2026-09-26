"""Shuttleworth-Wallace potential evapotranspiration, daily, as RZWQM2 computes it.

Independent JAX implementation of the published equations:

* Shuttleworth, W.J. and Wallace, J.S. (1985). Evaporation from sparse crops: an
  energy combination theory. Q. J. R. Meteorol. Soc. 111, 839-855.
* Farahani, H.J. and Ahuja, L.R. (1996). Evapotranspiration modeling of partial
  canopy/residue-covered fields. Trans. ASAE 39(6), 2051-2064 (three-source extension:
  canopy, bare soil, residue-covered soil).
* Ahuja, L.R. et al. (eds.) (2000). RZWQM: Modeling management effects on water
  quality and crop production. Water Resources Publications, chapter 3 (PET).

The reference model (RZWQM2 4.5, ``Rzpet.for``) was read to extract the exact
constants and branch structure; the code here is written from scratch. Subroutine
names below refer to that file: ``POTEVPHR`` (daily driver; ``POTEVP`` is the retired
version with the same daily branch), ``RESISThr`` (resistances), ``NETRAD``,
``MAXSW``/``CSRAD``/``GAUSS`` (clear-sky radiation), ``ECONST`` (psychrometrics),
``ALBSWS`` (soil albedo).

Unit conventions (RZWQM ``.ana`` / ``.MET`` conventions, kept internally):

* temperatures degC, ``srad`` MJ m-2 d-1 (measured global shortwave on the horizontal),
  ``rh`` in percent, ``wind_run`` km d-1 at ``wind_height`` m, ``lai`` m2 m-2,
  ``height`` cm, residue mass kg ha-1, latitude radians, elevation m.
* energy fluxes MJ m-2 d-1; resistances s m-1; the combination equation is evaluated
  with ``K = 86400`` s d-1 so that ``rho_a c_p (e_a - e_d) / r_a`` is in MJ m-2 d-1.
* outputs (potential transpiration, soil evaporation, residue evaporation) in **cm d-1**:
  the latent-heat flux is divided by ``10 * lambda`` with lambda in MJ kg-1.
* PAR is not used here; RZWQM takes PAR = 0.5 * srad elsewhere (``units.PAR_FRACTION``).

Every function is a pure function of arrays (scalars broadcast), ``jax.vmap``-able and
differentiable; branches use ``jnp.where`` and every division inside a masked branch
is guarded so that the unselected branch never produces NaN gradients.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.state import Params, field
from agrijax.core.units import CM_PER_M, KELVIN_OFFSET, MM_PER_CM, SECONDS_PER_HOUR

from .coefficients import (
    RZWQM_SW,
    AlbedoCoefficients,
    EconstCoefficients,
    MaxswCoefficients,
    NetradCoefficients,
    ResistCoefficients,
    SWCoefficients,
    WindCoefficients,
)

__all__ = [
    "RZWQM_SW",
    "AerodynamicResistances",
    "ClearSkyRadiation",
    "EnergyConstants",
    "NetRadiation",
    "PETParams",
    "SWCoefficients",
    "SWResult",
    "WindAdjustment",
    "clear_sky_radiation",
    "energy_constants",
    "net_radiation",
    "residue_albedo",
    "resistances",
    "saturation_vapour_pressure",
    "shuttleworth_wallace",
    "soil_albedo",
    "wind_adjustment",
]

# --------------------------------------------------------------------------------------
# coefficients: declared once in :mod:`.coefficients` (RZWQM_SW); the names below are
# read-only aliases of the defaults, kept for callers. The kernels read the coefficient set
# passed to them (``coefficients=`` / ``c=``), never these aliases.
# --------------------------------------------------------------------------------------
_RES = RZWQM_SW.resist
_POT = RZWQM_SW.potevp
VON_KARMAN = _RES.von_karman  # K in RESISThr
Z0_BARE_SOIL = _RES.z0_soil  # Z0P: effective roughness length of bare soil [m]
EDDY_DECAY = _RES.eddy_decay  # N: eddy diffusivity decay constant
DRAG_COEFF = _RES.drag_coeff  # CD
LEAF_BOUNDARY_RESISTANCE = _RES.leaf_boundary_resistance  # RB [s m-1]
STEFAN_BOLTZMANN = _POT.stefan_boltzmann  # SIGMA [MJ m-2 K-4 d-1]
CP_AIR = RZWQM_SW.econst.cp_air  # CP [MJ kg-1 degC-1]
TWO_THIRDS = RZWQM_SW.wind.canopy_height_factor
PEN123 = RZWQM_SW.wind.canopy_roughness_factor
CANOPY_EXTINCTION = _POT.canopy_extinction  # CO = exp(-0.594 TLAI)
SOLAR_CONST_HOURLY = RZWQM_SW.maxsw.solar_const_hourly  # W in MAXSW [MJ m-2 h-1]
TURBIDITY = RZWQM_SW.maxsw.turbidity  # B in MAXSW
_HOURS_PER_RADIAN = RZWQM_SW.maxsw.hours_per_radian  # PPCNST = 12/pi
_MIN_SIN_ALTITUDE = RZWQM_SW.maxsw.min_sin_altitude  # CSRAD cutoff
# residue characteristics in crop order corn / soybean / wheat (RESISThr DATA statements)
RESIDUE_DIAMETER_CM = {
    "corn": _RES.residue_diameter_corn,
    "soybean": _RES.residue_diameter_soybean,
    "wheat": _RES.residue_diameter_wheat,
}
RESIDUE_DENSITY_G_CM3 = {
    "corn": _RES.residue_density_corn,
    "soybean": _RES.residue_density_soybean,
    "wheat": _RES.residue_density_wheat,
}
RESIDUE_RANDOMNESS = _RES.residue_cover_default  # CRES default
# net long-wave coefficients per rainfall zone (POTEVPHR DATA al/bl): 1 arid, 2 semi-arid, 3 humid
LONGWAVE_COEFFS = {
    1: (_POT.lw_a_arid, _POT.lw_b_arid),
    2: (_POT.lw_a_semiarid, _POT.lw_b_semiarid),
    3: (_POT.lw_a_humid, _POT.lw_b_humid),
}
#: the rainfall zones of ``rainfall_zone`` (``rzwqm.dat`` IRAIN)
RAINFALL_ZONES = (1, 2, 3)
#: default inputs of the driver: anemometer height [m] and rainfall zone (semi-arid)
DEFAULT_WIND_HEIGHT_M = 2.0
DEFAULT_RAINFALL_ZONE = 2

# ---- unit conversions (not coefficients) ---------------------------------------------
KM_DAY_TO_M_S = 1.0e3 / 86400.0  # CONVW
SECONDS_PER_DAY = 86400.0  # K in POTEVPHR (daily)
_ONE_PERCENT = 1.0e-2  # relative humidity: percent -> fraction
_KILO = 1.0e3  # kPa -> Pa
_MILLI = 1.0e-3  # kg ha-1 -> t ha-1
_CENTI = 1.0e-2  # t ha-1 -> g cm-2
_MEGA = 1.0e6  # MJ -> J
#: arithmetic halves (means of two values, half-interval and midpoint of the quadrature)
_HALF = 0.5

# 10-point Gauss-Legendre nodes and weights (GAUSS, NGP=10; KEY(6)..KEY(7)-1): constants of the
# quadrature rule as the reference tabulates them, not model coefficients
_GAUSS_X = jnp.asarray([0.148874339, 0.433395394, 0.679409568, 0.865063367, 0.973906529])
_GAUSS_W = jnp.asarray([0.295524225, 0.269266719, 0.219086363, 0.149451349, 0.066671344])

_EPS = numerical_guard(
    "pet.sw.tiny",
    1e-12,
    "floor of divisors and logs, and the margin of the arccos argument, so that masked branches stay finite",
)
_WIND_FLOOR = numerical_guard("pet.sw.wind_floor", 1e-6, "floor of the wind speed [m s-1] in 1/(k^2 u)")
_HEIGHT_FLOOR = numerical_guard(
    "pet.sw.height_floor",
    1e-6,
    "floor of the canopy height [m] inside the log of the wind factor (masked at h = 0)",
)


class PETParams(Params):
    """Parameters of the Shuttleworth-Wallace PET process (RZWQM ``rzwqm.dat`` PET block, ``/IPOTEV/``)."""

    albedo_dry: Array = field(dims=(), unit="-", description="albedo of dry soil", fortran_name="A0")
    albedo_wet: Array = field(dims=(), unit="-", description="albedo of wet soil", fortran_name="AW")
    albedo_maturity: Array = field(
        dims=(), unit="-", description="albedo of the crop canopy at maturity", fortran_name="AC"
    )
    albedo_residue: Array = field(
        dims=(), unit="-", description="albedo of fresh residue", fortran_name="ARI"
    )
    soil_resistance: Array = field(
        dims=(), unit="s m-1", description="soil surface resistance to evaporation", fortran_name="RSS"
    )
    stomatal_resistance: Array = field(
        dims=(), unit="s m-1", description="minimum leaf stomatal resistance of the crop", fortran_name="RST"
    )


class EnergyConstants(NamedTuple):
    """Outputs of :func:`energy_constants` (``ECONST``)."""

    ea: Array  # mean saturation vapour pressure [kPa]
    ed: Array  # actual vapour pressure [kPa]
    ta: Array  # temperature at which e_sat = ea [degC]
    delta: Array  # slope of the saturation curve at ta [kPa degC-1]
    pressure: Array  # atmospheric pressure [kPa]
    rho_air: Array  # air density [kg m-3]
    latent_heat: Array  # [MJ kg-1]
    gamma: Array  # psychrometric constant [kPa degC-1]


class ClearSkyRadiation(NamedTuple):
    """Outputs of :func:`clear_sky_radiation` (``MAXSW``, horizontal surface)."""

    extraterrestrial: Array  # RP: top-of-atmosphere daily shortwave [MJ m-2 d-1]
    direct: Array  # RCHD: clear-sky direct on the horizontal [MJ m-2 d-1]
    diffuse: Array  # RDIF [MJ m-2 d-1]
    total: Array  # RCH = direct + diffuse [MJ m-2 d-1]
    sunrise_hour: Array  # PP(1) [h]
    sunset_hour: Array  # PP(2) [h]


class WindAdjustment(NamedTuple):
    """Outputs of :func:`wind_adjustment` (``POTEVPHR`` wind block)."""

    wind_run: Array  # UNEW [km d-1]
    reference_height: Array  # XWNEW [m]
    factor: Array  # WNDADJ [-]


class AerodynamicResistances(NamedTuple):
    """Outputs of :func:`resistances` (``RESISThr``), all in s m-1 unless noted."""

    raa: Array  # canopy source height -> reference height
    rac: Array  # bulk canopy boundary layer
    ras: Array  # substrate -> canopy source height
    rsc: Array  # bulk stomatal
    rss: Array  # soil surface
    rsr: Array  # residue layer
    soil_fraction: Array  # CS: fraction of soil not covered by residue [-]
    residue_thickness: Array  # HR [cm]


class NetRadiation(NamedTuple):
    """Outputs of :func:`net_radiation` (``NETRAD``), MJ m-2 d-1."""

    rn: Array  # net radiation over the field
    rns: Array  # net radiation at the bare-soil surface
    rnr: Array  # net radiation at the residue surface


class SWResult(eqx.Module):
    """Outputs of :func:`shuttleworth_wallace`; fluxes in cm d-1, energy in MJ m-2 d-1, resistances s m-1."""

    transpiration: Array  # PET: potential transpiration [cm d-1]
    soil_evaporation: Array  # PES [cm d-1]
    residue_evaporation: Array  # PER [cm d-1]
    latent_total: Array  # SWET: total S-W latent heat flux [MJ m-2 d-1]
    rn: Array
    rns: Array
    rnr: Array
    rnl: Array
    albedo_soil: Array
    albedo_residue: Array
    soil_fraction: Array
    canopy_fraction: Array  # CCL = 1 - exp(-0.594 TLAI)
    raa: Array
    rac: Array
    ras: Array
    rsc: Array
    rss: Array
    rsr: Array
    wind_run_adjusted: Array  # UNEW [km d-1]
    latent_heat: Array  # [MJ kg-1]
    vpd_source: Array  # DOP: vapour pressure deficit at the canopy source height [kPa]


# --------------------------------------------------------------------------------------
# psychrometrics (ECONST)
# --------------------------------------------------------------------------------------
def saturation_vapour_pressure(t: ArrayLike, c: EconstCoefficients = RZWQM_SW.econst) -> Array:
    """Saturation vapour pressure [kPa] at ``t`` degC.

    Source: Bosen (1960) form used in ``ECONST`` (Rzpet.for): ``exp((16.78 T - 116.9)/(T + 237.3))``,
    valid for -51 < T < 51 degC. Differs from the FAO-56 Tetens form by < 0.3 %. Coefficients:
    :class:`~.coefficients.EconstCoefficients`.
    """
    t = jnp.asarray(t)
    return jnp.exp((c.svp_a * t - c.svp_b) / (t + c.svp_c))


def energy_constants(
    tmin: ArrayLike,
    tmax: ArrayLike,
    rh: ArrayLike,
    elevation: ArrayLike,
    c: EconstCoefficients = RZWQM_SW.econst,
) -> EnergyConstants:
    """Psychrometric constants of the day (``ECONST``, Rzpet.for; van Bavel 1966).

    ``ea`` is the mean of the saturation pressures at ``tmax`` and ``tmin``; ``ta`` is the
    temperature whose saturation pressure is ``ea`` (inverted Bosen curve), and the slope
    ``delta`` is evaluated at ``ta``. ``ed = rh/100 * ea``. Pressure follows a polytropic
    atmosphere (T0 = 288 K, lapse 0.01 K/m, R = 286.9 J/kg/K); the virtual temperature uses
    273.2 K (as the reference model does, not 273.15). ``gamma = cp p / (0.622 lambda)``.

    Source: ECONST, Rzpet.for lines 136-241.
    """
    tmin = jnp.asarray(tmin)
    tmax = jnp.asarray(tmax)
    ea = _HALF * (saturation_vapour_pressure(tmax, c) + saturation_vapour_pressure(tmin, c))
    log_ea = jnp.log(ea)
    ta = (c.svp_c * log_ea + c.svp_b) / (c.svp_a - log_ea)
    ed = jnp.asarray(rh) * ea * _ONE_PERCENT
    delta = c.slope_a * saturation_vapour_pressure(ta, c) / (ta + c.svp_c) ** 2
    pressure = c.p0 * ((c.t0 - c.lapse_rate * jnp.asarray(elevation)) / c.t0) ** (
        c.gravity / c.lapse_rate / c.gas_constant
    )
    tv = (ta + c.virtual_t_offset) / (1.0 - c.virtual_vapour * ed / pressure)
    rho_air = _KILO * pressure / (tv * c.gas_constant)
    latent_heat = c.latent_heat_0 - c.latent_heat_slope * ta
    gamma = c.cp_air * pressure / (c.mw_ratio * latent_heat)
    return EnergyConstants(ea, ed, ta, delta, pressure, rho_air, latent_heat, gamma)


# --------------------------------------------------------------------------------------
# clear-sky radiation (MAXSW / CSRAD / GAUSS), horizontal surface only
# --------------------------------------------------------------------------------------
def _clear_sky_direct_instant(hour_angle: Array, c1: Array, c2: Array, c: MaxswCoefficients) -> Array:
    """``CSRAD``: instantaneous clear-sky direct beam relative to the solar constant (Eagleson 1970)."""
    sin_alt = c1 + c2 * jnp.cos(hour_angle)
    sin_safe = jnp.maximum(sin_alt, c.min_sin_altitude)
    h0 = 1.0 / sin_safe
    a1 = c.scatter_a - c.scatter_b * jnp.log10(h0)
    return jnp.where(sin_alt > c.min_sin_altitude, sin_safe * jnp.exp(-c.turbidity * a1 * h0), 0.0)


def clear_sky_radiation(
    doy: ArrayLike, latitude: ArrayLike, c: MaxswCoefficients = RZWQM_SW.maxsw
) -> ClearSkyRadiation:
    """Daily clear-sky shortwave on a horizontal surface (``MAXSW`` with slope = 0).

    Steps (Swift 1976; Shaffer & Larson 1982):
    orbital eccentricity ``E = 1 - 0.0167 cos(0.0172 (J - 3))`` scales the hourly solar constant
    ``W = 4.9212 MJ m-2 h-1`` as ``W/E^2``; declination from the RZ-SHAW formula
    ``asin(0.39785 sin(4.868961 + 0.017203 J + 0.033446 sin(6.224111 + 0.017202 J)))``;
    sunset hour angle ``acos(-tan(lat) tan(dec))`` (argument clipped to [-1, 1] here, the
    reference model would fail); extraterrestrial ``RP = W_e (12/pi)(2 c1 w_s + 2 c2 sin w_s)``;
    direct beam integrated with 10-point Gauss-Legendre quadrature of ``CSRAD`` over daylight;
    diffuse ``max(0, (0.91 RP - RCHD)/2)``; total = direct + diffuse.

    Source: MAXSW / CSRAD / GAUSS, Rzpet.for lines 70-134, 244-380, 543-907. Slope and aspect
    terms are omitted (the CA-TPA scenario has slope 0); the hourly arrays are not produced.
    Coefficients: :class:`~.coefficients.MaxswCoefficients`.
    """
    xj = jnp.asarray(doy, dtype=float)
    lat = jnp.asarray(latitude, dtype=float)
    ecc = 1.0 - c.eccentricity * jnp.cos(c.orbit_freq * (xj - c.perihelion_day))
    we = c.solar_const_hourly / (ecc * ecc)
    dec = jnp.arcsin(
        c.decl_amp
        * jnp.sin(
            c.decl_phase
            + c.decl_freq * xj
            + c.decl_ecc_amp * jnp.sin(c.decl_ecc_phase + c.decl_ecc_freq * xj)
        )
    )
    # clipped strictly inside (-1, 1): d/dx arccos is infinite at +-1, and the clip keeps the
    # gradient finite (zero) in the polar day / night regime the reference model cannot handle
    tssh = jnp.arccos(jnp.clip(-jnp.tan(lat) * jnp.tan(dec), -1.0 + _EPS, 1.0 - _EPS))
    tsrh = -tssh
    c1 = jnp.sin(dec) * jnp.sin(lat)
    c2 = jnp.cos(dec) * jnp.cos(lat)
    rp = we * c.hours_per_radian * (c1 * (tssh - tsrh) + c2 * (jnp.sin(tssh) - jnp.sin(tsrh)))
    # 10-point Gauss-Legendre over [tsrh, tssh]
    half = _HALF * (tssh - tsrh)
    mid = _HALF * (tssh + tsrh)
    xc = _GAUSS_X * half[..., None]
    f_plus = _clear_sky_direct_instant(mid[..., None] + xc, c1[..., None], c2[..., None], c)
    f_minus = _clear_sky_direct_instant(mid[..., None] - xc, c1[..., None], c2[..., None], c)
    rchd = we * c.hours_per_radian * half * jnp.sum(_GAUSS_W * (f_plus + f_minus), axis=-1)
    rdif = jnp.maximum(0.0, c.diffuse_share * (c.diffuse_fraction * rp - rchd))
    rch = rdif + rchd
    sunrise = c.solar_noon_hour + tsrh / c.radians_per_hour
    sunset = c.solar_noon_hour + tssh / c.radians_per_hour
    return ClearSkyRadiation(rp, rchd, rdif, rch, sunrise, sunset)


# --------------------------------------------------------------------------------------
# albedos (ALBSWS and the residue albedo block of POTEVPHR)
# --------------------------------------------------------------------------------------
def soil_albedo(
    theta: ArrayLike,
    wc13: ArrayLike,
    wc15: ArrayLike,
    albedo_dry: ArrayLike,
    albedo_wet: ArrayLike,
    *,
    crust: ArrayLike = 0.0,
    roughness_cm: ArrayLike = 0.0,
    c: AlbedoCoefficients = RZWQM_SW.albedo,
) -> Array:
    """Soil albedo weighted by surface water content (``ALBSWS``, DeCoursey).

    Linear between the dry value at the 15-bar content ``wc15`` and the wet value at the
    1/3-bar content ``wc13``, clipped outside. A crust (``crust = 1``) doubles the dry albedo
    (Shaffer & Larson, NTRM); random roughness ``RR`` (cm) reduces both by 8 % per cm.

    Source: ALBSWS, Rzpet.for lines 3-68.
    """
    theta = jnp.asarray(theta)
    rr_mod = jnp.asarray(roughness_cm) * c.roughness_reduction
    scale = jnp.where(rr_mod > 0.0, 1.0 - rr_mod, 1.0)
    a_dry = jnp.asarray(albedo_dry) * (1.0 + jnp.asarray(crust)) * scale
    a_wet = jnp.asarray(albedo_wet) * scale
    span = jnp.maximum(jnp.asarray(wc13) - jnp.asarray(wc15), _EPS)
    frac = jnp.clip((theta - wc15) / span, 0.0, 1.0)
    return a_dry + (a_wet - a_dry) * frac


def residue_albedo(
    albedo_dry: ArrayLike,
    albedo_residue: ArrayLike,
    residue_age: ArrayLike = 0.0,
    residue_wet: ArrayLike = 0.0,
    c: AlbedoCoefficients = RZWQM_SW.albedo,
) -> Array:
    """Residue albedo decaying with age towards 1.06 x the dry-soil albedo; 0.75 x when wet.

    ``AR = A0 (1.06 + (ARI/A0 - 1.06) exp(-0.0255 RESAGE))``, floored at 0, times 0.75 when the
    residue holds water (``WRES > 0``).

    Source: POTEVPHR, Rzpet.for (residue albedo block after ALBSWS).
    """
    a0 = jnp.asarray(albedo_dry)
    ari = jnp.asarray(albedo_residue)
    r = c.residue_aged_ratio
    ar = a0 * (
        r + (ari / jnp.maximum(a0, _EPS) - r) * jnp.exp(-c.residue_ageing_rate * jnp.asarray(residue_age))
    )
    ar = jnp.maximum(ar, 0.0)
    return jnp.where(jnp.asarray(residue_wet) > 0.0, c.wet_residue_factor * ar, ar)


# --------------------------------------------------------------------------------------
# wind adjustment (POTEVPHR wind block, "HAMID, 9/10/93")
# --------------------------------------------------------------------------------------
def wind_adjustment(
    wind_run: ArrayLike,
    height_cm: ArrayLike,
    wind_height: ArrayLike = DEFAULT_WIND_HEIGHT_M,
    c: WindCoefficients = RZWQM_SW.wind,
) -> WindAdjustment:
    """Move the wind run from the measurement height to a reference height above the canopy.

    With the anemometer below 10 m (a micro-tower with 10 cm of cover) the reference height is
    ``XWNEW = 1.93 + (2/3) h`` for a canopy of height ``h`` (m) and 2.0 m without a canopy; the
    factor is ``ln(1.93/(0.123 h)) / ln((XW - 0.07)/0.0123)`` (bare: ``ln(200)`` in the numerator).
    An anemometer at >= 10 m is assumed at an airport with 40 cm of cover: ``XWNEW = 1.33 + (2/3) h``
    (1.6 bare), factor ``ln(1.33/(0.123 h)) / ln((XW - 0.27)/0.05)`` (bare: ``ln(160)``).
    Known deviation: none; note the reference model is discontinuous at ``h -> 0+``.

    Source: POTEVPHR, Rzpet.for (wind height block, "HAMID, 9/10/93").
    """
    h = jnp.asarray(height_cm) / CM_PER_M
    xw = jnp.asarray(wind_height)
    h_safe = jnp.maximum(h, _HEIGHT_FLOOR)
    airport = xw >= c.airport_height
    xw_new_crop = jnp.where(airport, c.ref_offset_airport, c.ref_offset_micro) + c.canopy_height_factor * h
    w1_crop = jnp.log(
        jnp.where(airport, c.ref_offset_airport, c.ref_offset_micro) / (c.canopy_roughness_factor * h_safe)
    )
    xw_new_bare = jnp.where(airport, c.ref_bare_airport, c.ref_bare_micro)
    w1_bare = jnp.log(xw_new_bare / c.bare_roughness)
    # both branches are evaluated (as inside the where); the roughness lengths are positive coefficients
    w2_airport = jnp.log(jnp.maximum(xw - c.airport_displacement, _EPS) / c.airport_roughness)
    w2_micro = jnp.log(jnp.maximum(xw - c.micro_displacement, _EPS) / c.micro_roughness)
    w2 = jnp.where(airport, w2_airport, w2_micro)
    has_canopy = h > 0.0
    xw_new = jnp.where(has_canopy, xw_new_crop, xw_new_bare)
    w1 = jnp.where(has_canopy, w1_crop, w1_bare)
    factor = w1 / w2
    return WindAdjustment(jnp.asarray(wind_run) * factor, xw_new, factor)


# --------------------------------------------------------------------------------------
# resistances (RESISThr, daily, no standing stubble, no plastic mulch)
# --------------------------------------------------------------------------------------
def resistances(
    wind: WindAdjustment,
    lai: ArrayLike,
    tlai: ArrayLike,
    height_cm: ArrayLike,
    ta: ArrayLike,
    residue_mass: ArrayLike,
    soil_resistance: ArrayLike,
    stomatal_resistance: ArrayLike,
    *,
    trat: ArrayLike = 1.0,
    residue_diameter_cm: ArrayLike | None = None,
    residue_density: ArrayLike | None = None,
    residue_randomness: ArrayLike = RESIDUE_RANDOMNESS,
    c: ResistCoefficients = RZWQM_SW.resist,
) -> AerodynamicResistances:
    """Aerodynamic and surface resistances of the three-source Shuttleworth-Wallace scheme.

    Canopy (``lai > 0``; plant height floored at 0.05 m, ``XLAI = max((LAI+TLAI)/2, 0.05)``):
    displacement ``d = 1.1 h ln(1 + (0.07 XLAI)^0.25)``; roughness ``z0 = 0.01 + 0.3 h sqrt(X)``
    for ``X = 0.07 XLAI <= 0.2`` else ``0.3 h (1 - d/h)``; bulk boundary-layer ``rac = 10/(2 XLAI)``;
    bulk stomatal ``rsc = RST/(2 LAI)`` (XLAI < 2), ``RST/LAI`` (2..3), ``RST/3`` (XLAI > 3), times the
    DSSAT CO2 transpiration ratio ``trat`` (Farahani & Bausch 1994; Jagtap & Jones); ``ras`` and
    ``raa`` from the Shuttleworth & Gurney (1990) exponential eddy-diffusivity profile with decay
    constant 2.5 and the preferred values ``dp = 0.63 h``, ``zp = 0.13 h`` (S-W 1985 eqs. 41-43).
    Bare soil (``lai == 0``): ``z0 = max(0.01, z0r)``, ``ras = ln(z/z0) ln(z/2/z0) / (k^2 u)``,
    ``raa = ln(z/z0)^2/(k^2 u) - ras``.
    Residue (mass > 1e-6 kg/ha): exposed soil fraction ``CS = exp(-CRES 0.0127 M/(d rho))`` with M in
    t/ha, diameter and specific density of the residue type (corn 1.0 cm / 0.15 g cm-3); bulk density
    ``0.2 rho``; thickness ``HR = 0.01 M / ((1 - CS) rhob)`` cm; ``z0r = 0.197 HR``; residue resistance
    ``rsr = 1.1 HR / (2.12e-5 (1 + 0.007 max(0, Ta - 20)) (1 + 1.25e-3 rhob^-1.79 u2) 0.8)`` with ``u2``
    the wind 2 m above the residue.
    Soil surface resistance is the constant parameter (``rzwqm.dat`` item 11 >= 0).
    ``residue_diameter_cm``, ``residue_density`` and ``residue_randomness`` may be traced arrays
    (one value per day under ``vmap``), so that the reference model's switch of the residue-type
    constants at harvest can be expressed without splitting the batch; ``None`` takes the corn values
    of ``c``.

    Known deviations: the wetness-dependent ``rss`` options (item 11 = -1 Sakaguchi & Zeng 2009,
    -2 Farahani & Bausch 1995), standing stubble (``sai``), plastic mulch and the PENFLUX residue
    conductances are not implemented. The reference model uses 1e30 for the missing canopy
    resistances; here ``rac``/``rsc`` are still computed (masked downstream) so gradients stay finite.

    Coefficients: :class:`~.coefficients.ResistCoefficients`.

    Source: RESISThr, Rzpet.for lines 2433-2838.
    """
    lai = jnp.asarray(lai)
    tlai = jnp.asarray(tlai)
    us = jnp.maximum(wind.wind_run * KM_DAY_TO_M_S, _WIND_FLOOR)
    dk = 1.0 / (c.von_karman**2 * us)
    xw_new = wind.reference_height
    plht = jnp.maximum(jnp.asarray(height_cm) / CM_PER_M, c.min_plant_height)
    xlai = jnp.maximum(_HALF * (lai + tlai), c.min_lai)
    has_canopy = lai > 0.0
    lai_safe = jnp.maximum(lai, _EPS)

    # ---- canopy ----
    n = c.eddy_decay
    dp = c.displacement_pref * plht
    zp = c.roughness_pref * plht
    x = c.drag_coeff * xlai
    d = c.displacement_factor * plht * jnp.log(1.0 + x**c.displacement_exp)
    # X = cd XLAI >= 0.05 cd > 0 for a positive drag coefficient: the sqrt is finite on both branches
    sparse = x <= c.sparse_x_limit
    z0_sparse = c.z0_soil + c.roughness_factor * plht * jnp.sqrt(x)
    z0_can = jnp.where(sparse, z0_sparse, c.roughness_factor * plht * (1.0 - d / plht))
    rac = c.leaf_boundary_resistance / (c.rac_lai_factor * xlai)
    rst = jnp.asarray(stomatal_resistance)
    # the divisors are positive coefficients (and the floored LAI): every branch is finite
    low_lai = xlai < c.rsc_low_lai
    rsc_low = rst / (c.rsc_low_divisor * lai_safe)
    high_lai = xlai > c.rsc_high_lai
    rsc_high = rst / c.rsc_high_divisor
    rsc = jnp.where(low_lai, rsc_low, jnp.where(high_lai, rsc_high, rst / lai_safe))
    rsc = rsc * jnp.asarray(trat)
    gap = jnp.maximum(xw_new - d, _EPS)
    c1 = jnp.log(gap / z0_can) * dk
    shape = plht / (n * jnp.maximum(plht - d, _EPS)) * jnp.exp(n)
    c2 = shape * jnp.exp(-n * c.z0_soil / plht)
    c3 = shape * jnp.exp(-n * (zp + dp) / plht)
    ras_neutral = jnp.log(xw_new / z0_can) * jnp.log(c.ras_height_fraction * xw_new / z0_can) * dk
    ras_can = jnp.where(c2 <= c3, ras_neutral, c1 * (c2 - c3))
    raa_can = c1 * (
        jnp.log(gap / jnp.maximum(plht - d, _EPS))
        + plht / (n * jnp.maximum(plht - d, _EPS)) * (jnp.exp(n * (1.0 - (dp + zp) / plht)) - 1.0)
    )

    # ---- residue ----
    rm = jnp.asarray(residue_mass)
    rdia = jnp.asarray(c.residue_diameter_corn if residue_diameter_cm is None else residue_diameter_cm)
    rhors = jnp.asarray(c.residue_density_corn if residue_density is None else residue_density)
    cres = jnp.asarray(residue_randomness)
    has_residue = rm > c.residue_mass_threshold
    trm = rm * _MILLI
    cs = jnp.where(
        has_residue,
        jnp.exp(-cres * c.residue_cover_coeff * trm / jnp.maximum(rdia * rhors, _EPS)),
        1.0,
    )
    rhorb = c.residue_bulk_ratio * rhors
    hr = jnp.where(has_residue, trm * _CENTI / (jnp.maximum(1.0 - cs, _EPS) * jnp.maximum(rhorb, _EPS)), 0.0)
    hrm = hr / CM_PER_M
    z0r = c.residue_roughness_ratio * hrm
    z0r_safe = jnp.maximum(z0r, _EPS)
    resp = 1.0 - rhorb / jnp.maximum(rhors, _EPS)
    resp = jnp.where((resp <= c.porosity_low) | (resp > c.porosity_high), c.porosity_default, resp)
    u2 = us * jnp.log(c.u2_height / z0r_safe) / jnp.log(xw_new / z0r_safe)
    # the vapour diffusivity and the bracketed factors are positive: finite with or without residue
    rsr_layer = (
        c.rsr_tortuosity
        * hrm
        / (
            c.vapour_diffusivity
            * (1.0 + c.diffusivity_temp_coeff * jnp.maximum(0.0, jnp.asarray(ta) - c.diffusivity_ref_temp))
            * (
                1.0
                + c.residue_wind_coeff
                * jnp.maximum(rhorb, _EPS) ** (-c.residue_wind_exp)
                * jnp.maximum(u2, 0.0)
            )
            * resp
        )
    )
    rsr = jnp.where(has_residue, rsr_layer, 0.0)

    # ---- bare soil / no canopy ----
    z0_bare = jnp.maximum(c.z0_soil, z0r)
    log_bare = jnp.log(xw_new / z0_bare)
    ras_bare = log_bare * jnp.log(c.ras_height_fraction * xw_new / z0_bare) * dk
    raa_bare = log_bare**2 * dk - ras_bare

    ras = jnp.where(has_canopy, ras_can, ras_bare)
    raa = jnp.where(has_canopy, raa_can, raa_bare)
    rss = jnp.asarray(soil_resistance) * jnp.ones_like(ras)
    return AerodynamicResistances(raa, rac, ras, rsc, rss, rsr, cs, hr)


# --------------------------------------------------------------------------------------
# net radiation (NETRAD, no plastic mulch)
# --------------------------------------------------------------------------------------
def net_radiation(
    srad: ArrayLike,
    rnl: ArrayLike,
    albedo_canopy: ArrayLike,
    albedo_residue: ArrayLike,
    albedo_soil: ArrayLike,
    canopy_fraction: ArrayLike,
    soil_fraction: ArrayLike,
    c: NetradCoefficients = RZWQM_SW.netrad,
) -> NetRadiation:
    """Net radiation over the field and at the soil and residue surfaces (``NETRAD``).

    ``TCAN = CCL (1-AC) + CO CR (1-AR) + CO CS (1-AS)``; ``RN = TCAN RTS + RNL`` (replaced by
    ``TCAN RTS/3`` when negative). The substrate shares use the canopy transmission
    ``TAC = 1 - (0.5 + 0.44 CCL)``: ``RNS = CS (1-AS)(CO + TAC CCL (1-AC)) RTS + CO CS RNL`` and the same
    for residue with ``CR``; when either is negative both are reset to ``RN CO CS`` / ``RN CO CR``.

    Source: NETRAD, Rzpet.for lines 914-1019 (plastic-mulch terms set to zero).
    """
    rts = jnp.asarray(srad)
    rnl = jnp.asarray(rnl)
    ccl = jnp.asarray(canopy_fraction)
    co = 1.0 - ccl
    cs = jnp.asarray(soil_fraction)
    cr = 1.0 - cs
    tc = 1.0 - jnp.asarray(albedo_canopy)
    tr = 1.0 - jnp.asarray(albedo_residue)
    ts = 1.0 - jnp.asarray(albedo_soil)
    tcan = ccl * tc + co * cr * tr + co * cs * ts
    rn = tcan * rts + rnl
    negative_rn = rn < 0.0
    rn_floor = tcan * rts / c.negative_rn_divisor  # positive coefficient divisor
    rn = jnp.where(negative_rn, rn_floor, rn)
    tac = 1.0 - (c.canopy_absorb_base + c.canopy_absorb_slope * ccl)
    through = co + tac * ccl * tc
    rnr = cr * tr * through * rts + co * cr * rnl
    rns = cs * ts * through * rts + co * cs * rnl
    negative = (rns < 0.0) | (rnr < 0.0)
    rns = jnp.where(negative, rn * co * cs, rns)
    rnr = jnp.where(negative, rn * co * cr, rnr)
    return NetRadiation(rn, rns, rnr)


# --------------------------------------------------------------------------------------
# driver (POTEVPHR daily branch, ipet = 0, no pan, no SHAW/PENFLUX)
# --------------------------------------------------------------------------------------
def shuttleworth_wallace(
    tmin: ArrayLike,
    tmax: ArrayLike,
    srad: ArrayLike,
    rh: ArrayLike,
    wind_run: ArrayLike,
    lai: ArrayLike,
    height_cm: ArrayLike,
    params: PETParams,
    *,
    theta_surface: ArrayLike,
    wc13: ArrayLike,
    wc15: ArrayLike,
    elevation: ArrayLike,
    latitude: ArrayLike,
    doy: ArrayLike,
    tlai: ArrayLike | None = None,
    srad_horizontal: ArrayLike | None = None,
    residue_mass: ArrayLike = 0.0,
    residue_age: ArrayLike = 0.0,
    residue_wet: ArrayLike = 0.0,
    crust: ArrayLike = 0.0,
    roughness_cm: ArrayLike = 0.0,
    wind_height: ArrayLike = DEFAULT_WIND_HEIGHT_M,
    trat: ArrayLike = 1.0,
    soil_heat_flux: ArrayLike = 0.0,
    rainfall_zone: int = DEFAULT_RAINFALL_ZONE,
    residue_type: str = "corn",
    residue_cover_factor: ArrayLike = RESIDUE_RANDOMNESS,
    residue_diameter_cm: ArrayLike | None = None,
    residue_density: ArrayLike | None = None,
    coefficients: SWCoefficients = RZWQM_SW,
) -> SWResult:
    """Daily Shuttleworth-Wallace potential transpiration, soil and residue evaporation [cm d-1].

    Parameters
    ----------
    tmin, tmax : degC. srad : daily solar radiation of the field [MJ m-2 d-1] (``RTS``; in the
        reference model the daily sum of the hourly radiation after the direct/diffuse split,
        Rzmain.for line 1389, printed in ``.ana`` col 88); it drives the net shortwave radiation.
        rh : percent.
    srad_horizontal : measured horizontal daily solar radiation of the weather file
        [MJ m-2 d-1] (``RTH``, read by INPDAY, Rzmain.for line 1312). It enters only the
        cloudiness ratio ``RTH / RCH`` of the net long-wave radiation and the floor
        ``RCH >= RTH`` of the clear-sky radiation (POTEVPHR, Rzpet.for lines 1983, 1988, 1998;
        the long-wave form is FAO-24 / Wright & Jensen 1972). Default: ``srad``. Measured on the
        POTEVPHR dumps of 9 scenarios x 3 years: RTS and RTH differ by up to 0.11 MJ m-2 d-1,
        and using RTS for both changes PE by up to 0.1 mm d-1 on days near the ``RN < 0`` switch
        of :func:`net_radiation` (about 5e-3 mm d-1 on other days).
    wind_run : km d-1 at ``wind_height`` m. lai : green LAI; ``tlai`` total LAI (default = lai).
    height_cm : canopy height [cm]. params : :class:`PETParams`.
    theta_surface, wc13, wc15 : surface-layer water content and its 1/3-bar and 15-bar values.
    elevation : m. latitude : radians. doy : day of year.
    residue_mass : flat residue [kg ha-1]; ``residue_age`` days, ``residue_wet`` > 0 when wet.
    crust, roughness_cm : crust flag and random roughness for the soil albedo.
    trat : DSSAT ``TRATIO`` CO2 factor on stomatal resistance (1.0 at 330 ppm).
    soil_heat_flux : G [MJ m-2 d-1]; the daily reference model uses 0.
    rainfall_zone : 1 arid, 2 semi-arid, 3 humid (net long-wave coefficients; static).
    residue_type : "corn" | "soybean" | "wheat" (static; RZWQM ``IPR`` residue-type index 1..3).
        Gives the residue diameter and specific density unless ``residue_diameter_cm`` and
        ``residue_density`` are passed explicitly (traced values, one per day under ``vmap``).
    residue_cover_factor : ``CRES`` of the ``rzwqm.dat`` residue block (corn 2.0, soybean 2.5,
        wheat 4.0; the reference model falls back to 1.32 when the file gives <= 0).
    coefficients : :class:`~.coefficients.SWCoefficients`, every coefficient of the equations below
        (default :data:`~.coefficients.RZWQM_SW`, the RZWQM2 values; an instance with array leaves,
        ``RZWQM_SW.as_arrays()``, makes them differentiable and calibratable). The residue diameter
        and density of ``residue_type`` are read from it unless passed explicitly.

    Input timing (what the reference model passes at its daily PET call, measured on CA-TPA 2015
    against every ``.ana`` row, see ``poc/coarse_compare.py``): all state inputs are
    **start-of-day** values. In the reference day loop the management events (tillage, which
    incorporates residue) precede the PET call, whereas crop growth, harvest (canopy removal and
    the addition of harvest residue) and the residue-type switch to the harvested crop's
    constants follow it. Hence LAI, height and residue mass are the previous day's end state,
    except that a tillage day uses the post-tillage residue mass, and the day after a harvest
    uses LAI = height = 0 with the harvest residue and residue age 1. Residue age counts from
    the file's initial age (+1 on the first day) and restarts at harvest. Wind is the forcing
    after :func:`agrijax.io.rzwqm.prepare_rzwqm_forcing` (100 km/d floor).

    Algorithm (POTEVPHR, ipet = 0, ihourly = 0):

    1. wind adjusted to the reference height (:func:`wind_adjustment`);
    2. psychrometrics (:func:`energy_constants`);
    3. clear-sky ``RCH`` (:func:`clear_sky_radiation`), ``RCH = max(RCH, RTH)``, RTH = ``srad_horizontal``;
    4. soil albedo (:func:`soil_albedo`), residue albedo (:func:`residue_albedo`);
    5. resistances (:func:`resistances`); ``rsc`` x 10 on the "night" test ``srad * 1e6 / 3.6e3 < 10``.
       Reference-model quirk: that is the hourly W m-2 test (MJ m-2 h-1 -> W m-2) applied to the
       *daily* total, so the effective threshold is ``srad < 0.036 MJ m-2 d-1``, never true for
       daily input. Kept as is to match the reference model (Rzpet.for line 1957); do not "fix"
       the conversion to the daily one (``/ 0.0864``) without changing the oracle comparison;
    6. canopy cover ``CCL = 1 - exp(-0.594 TLAI)``;
    7. net long-wave ``RNL = -(a RTH/RCH + b) (0.39 - 0.158 sqrt(ed)) sigma (Tmax_K^4 + Tmin_K^4)/2``
       with ``(a, b)`` by rainfall zone and the ratio clipped to [0, 1];
    8. net radiation (:func:`net_radiation`);
    9. combination: with ``C1 = delta (Rn - G)``, ``C2 = 86400 rho_a c_p``, the Penman-Monteith
       "closed" fluxes ``PMC`` (canopy), ``PMS`` (soil), ``PMR`` (residue) of S-W 1985 eqs. 12-13
       extended to three sources; coefficients ``CCC, CCS, CCR`` (S-W eqs. 14-15, Farahani & Ahuja
       1996 eqs. 5-8) built from ``R_a = (delta+gamma) raa``, ``R_s = (delta+gamma) ras + gamma rss``,
       ``R_c = (delta+gamma) rac + gamma rsc``, ``R_r = (delta+gamma) ras + gamma (rss + rsr)``;
       total ``SWET = CCC PMC + CCS PMS + CCR PMR``; source-height deficit
       ``D0 = (ea - ed) + (C1 - (delta+gamma) SWET) raa / C2`` (S-W eq. 16); component fluxes
       (S-W eqs. 17-18) ``lambda T = (delta (Rn - Rn_sub) + C2 D0/rac)/(delta + gamma (1 + rsc/rac))``,
       ``lambda E_s = CS (delta (Rn_sub - G) + C2 D0/ras)/(delta + gamma (1 + rss/ras))``,
       ``lambda E_r = CR (delta (Rn_sub - G) + C2 D0/ras)/(delta + gamma (1 + (rss + rsr)/ras))``
       where ``Rn_sub = Rns + Rnr``; each divided by ``10 lambda`` and floored at 0 (cm d-1).

    Known deviations from the reference model: snow-pack sublimation (``SNOWQE``, whose result
    the reference model overwrites with a constant anyway), the snow long-wave variant (its
    switch ``NSP`` is only set by the SHAW path, never in the daily branch), pan-evaporation
    input, hourly / SHAW / PENFLUX paths, slope and aspect, standing stubble (``SAI``,
    ``SDEAD_HEIGHT`` in the wind-height block) and plastic mulch are not implemented; infinite
    resistances are handled by masking rather than 1e30; the sunset-angle argument and the
    square root of the vapour pressure are guarded so that gradients stay finite.

    Source: POTEVPHR, Rzpet.for lines 1673-2428; Shuttleworth & Wallace (1985); Farahani & Ahuja (1996).
    """
    cf = coefficients
    pc = cf.potevp
    tlai_arr = jnp.asarray(lai) if tlai is None else jnp.asarray(tlai)
    lai = jnp.asarray(lai)
    srad = jnp.asarray(srad)
    tmin = jnp.asarray(tmin)
    tmax = jnp.asarray(tmax)
    g = jnp.asarray(soil_heat_flux)
    zone = int(rainfall_zone)
    if zone not in RAINFALL_ZONES:
        raise KeyError(f"rainfall_zone must be one of {RAINFALL_ZONES}, got {rainfall_zone!r}")
    if residue_type not in RESIDUE_DIAMETER_CM:
        raise KeyError(f"residue_type must be one of {tuple(RESIDUE_DIAMETER_CM)}, got {residue_type!r}")
    a_lw = (pc.lw_a_arid, pc.lw_a_semiarid, pc.lw_a_humid)[zone - 1]
    b_lw = (pc.lw_b_arid, pc.lw_b_semiarid, pc.lw_b_humid)[zone - 1]

    wind = wind_adjustment(wind_run, height_cm, wind_height, cf.wind)
    ec = energy_constants(tmin, tmax, rh, elevation, cf.econst)
    csr = clear_sky_radiation(doy, latitude, cf.maxsw)
    srad_h = srad if srad_horizontal is None else jnp.asarray(srad_horizontal)
    rch = jnp.maximum(csr.total, srad_h)

    a_soil = soil_albedo(
        theta_surface,
        wc13,
        wc15,
        params.albedo_dry,
        params.albedo_wet,
        crust=crust,
        roughness_cm=roughness_cm,
        c=cf.albedo,
    )
    a_res = residue_albedo(params.albedo_dry, params.albedo_residue, residue_age, residue_wet, cf.albedo)

    rdia = (
        getattr(cf.resist, f"residue_diameter_{residue_type}")
        if residue_diameter_cm is None
        else residue_diameter_cm
    )
    rhors = (
        getattr(cf.resist, f"residue_density_{residue_type}") if residue_density is None else residue_density
    )
    res = resistances(
        wind,
        lai,
        tlai_arr,
        height_cm,
        ec.ta,
        residue_mass,
        params.soil_resistance,
        params.stomatal_resistance,
        trat=trat,
        residue_diameter_cm=rdia,
        residue_density=rhors,
        residue_randomness=residue_cover_factor,
        c=cf.resist,
    )
    # Rzpet.for line 1957: the hourly W m-2 night test on the daily total (effective threshold
    # 0.036 MJ m-2 d-1); intentionally reproduced, see step 5 of the docstring
    night = srad * _MEGA / SECONDS_PER_HOUR < pc.night_radiation
    rsc = jnp.where(night, res.rsc * pc.night_rsc_factor, res.rsc)
    cs = res.soil_fraction
    cr = 1.0 - cs
    # RESISThr takes the canopy branch for LAI > 0 (height floored at 5 cm); the transpiration
    # flux in POTEVPHR additionally requires the raw HEIGHT > 0.
    has_canopy_res = lai > 0.0
    has_canopy = has_canopy_res & (jnp.asarray(height_cm) > 0.0)
    has_residue = jnp.asarray(residue_mass) > 0.0

    ccl = 1.0 - jnp.exp(-pc.canopy_extinction * tlai_arr)

    tl4 = _HALF * ((tmax + KELVIN_OFFSET) ** 4 + (tmin + KELVIN_OFFSET) ** 4)
    # sqrt floored at _EPS: d sqrt(x)/dx is infinite at x = 0 (rh = 0), the floor makes it zero
    rb0 = (pc.emissivity_a - pc.emissivity_b * jnp.sqrt(jnp.maximum(ec.ed, _EPS))) * pc.stefan_boltzmann * tl4
    rsratio = jnp.clip(jnp.where(rch > 0.0, srad_h / jnp.maximum(rch, _EPS), 0.0), 0.0, 1.0)
    rnl = -(a_lw * rsratio + b_lw) * rb0

    nr = net_radiation(srad, rnl, params.albedo_maturity, a_res, a_soil, ccl, cs, cf.netrad)
    rn, rns, rnr = nr
    rnsub = rns + rnr

    delta, gamma = ec.delta, ec.gamma
    vpd = ec.ea - ec.ed
    c1 = delta * (rn - g)
    c2 = SECONDS_PER_DAY * ec.rho_air * cf.econst.cp_air
    raa, rac, ras, rss, rsr = res.raa, res.rac, res.ras, res.rss, res.rsr

    pmc = (c1 + (c2 * vpd - delta * rac * (rnsub - g)) / (raa + rac)) / (
        delta + gamma * (1.0 + rsc / (raa + rac))
    )
    pmc = jnp.where(has_canopy, jnp.maximum(pmc, 0.0), 0.0)
    pm_sub = c1 + (c2 * vpd - delta * ras * (rn - rnsub)) / (raa + ras)
    pms = pm_sub / (delta + gamma * (1.0 + rss / (raa + ras)))
    pms = jnp.where(cs > 0.0, jnp.maximum(pms, 0.0), 0.0)
    pmr = pm_sub / (delta + gamma * (1.0 + (rss + rsr) / (raa + ras)))
    pmr = jnp.where(has_residue, jnp.maximum(pmr, 0.0), 0.0)

    cra = (delta + gamma) * raa
    crs = (delta + gamma) * ras + gamma * rss
    crc = (delta + gamma) * rac + gamma * rsc
    crr = (delta + gamma) * ras + gamma * (rss + rsr)
    # infinite canopy resistance when no canopy; crc > 0 on the selected branch, the floor keeps
    # the unselected branch finite (no NaN cotangent through where)
    inv_crc = jnp.where(has_canopy_res, 1.0 / jnp.maximum(crc, _EPS), 0.0)
    denom = crs * crr + cra * crs * crr * inv_crc + cra * crr * cs + cra * crs * cr
    ccc = crs * crr * (1.0 + cra * inv_crc) / denom
    ccs = crr * (crs + cra) * cs / denom
    ccr = crs * (crr + cra) * cr / denom
    swet = ccc * pmc + ccs * pms + ccr * pmr

    d0 = vpd + (c1 - (delta + gamma) * swet) * raa / c2

    lam_t = (delta * (rn - rnsub) + c2 * d0 / rac) / (delta + gamma * (1.0 + rsc / rac))
    lam_t = jnp.where(has_canopy, lam_t, 0.0)
    lam_sub = delta * (rnsub - g) + c2 * d0 / ras
    lam_es = lam_sub / (delta + gamma * (1.0 + rss / ras)) * cs
    lam_er = lam_sub / (delta + gamma * (1.0 + (rss + rsr) / ras)) * cr

    to_cm = 1.0 / (ec.latent_heat * MM_PER_CM)
    transp = jnp.maximum(lam_t * to_cm, 0.0)
    soil_evap = jnp.maximum(lam_es * to_cm, 0.0)
    res_evap = jnp.maximum(lam_er * to_cm, 0.0)

    return SWResult(
        transpiration=transp,
        soil_evaporation=soil_evap,
        residue_evaporation=res_evap,
        latent_total=swet,
        rn=rn,
        rns=rns,
        rnr=rnr,
        rnl=rnl,
        albedo_soil=a_soil,
        albedo_residue=a_res,
        soil_fraction=cs,
        canopy_fraction=ccl,
        raa=raa,
        rac=rac,
        ras=ras,
        rsc=rsc,
        rss=rss,
        rsr=rsr,
        wind_run_adjusted=wind.wind_run,
        latent_heat=ec.latent_heat,
        vpd_source=d0,
    )
