"""DSSAT-CSM SPAM partition of the potential evapotranspiration ``EO`` into the potential soil
evaporation ``EOS`` and the potential transpiration ``EOP`` (the pet slot's DSSAT day).

In DSSAT-CSM v4.8.6.0 ``SPAM`` (``SPAM/SPAM.for`` RATE, lines 300-387) splits the day's potential
evapotranspiration in two steps around the actual evaporation of the soil water module:

1. ``PET`` (``SPAM.for:300``) gives ``EO`` [mm d-1]; with ``EVAPO = R`` this is ``PETPT``
   (:func:`agrijax.processes.pet.priestley_taylor`, registered as
   ``pet/priestley_taylor@dssat-4.8.6.0:faithful``);
2. ``PSE`` (``SPAM.for:314``; ``SPAM/PET.for:1448-1496``) gives the potential soil evaporation
   ``EOS`` from ``EO`` and the leaf area ``XLAI`` with the extinction coefficient ``KSEVAP``
   (:func:`potential_soil_evaporation`, process :func:`spam_potential_soil_evaporation`);
3. the soil water module takes the mulch and soil evaporation out of ``EOS`` (``MULCH_EVAP``,
   ``SOILEV`` or ``ESR_SoilEvap``; :mod:`agrijax.processes.soil_water.bucket_evap`), which gives
   the actual evaporation ``EVAP = ES + EM + EF`` (``SPAM.for:373``);
4. ``TRANS`` (``SPAM.for:378-387``; ``SPAM/TRANS.for:29-151``) gives the potential transpiration
   ``EOP`` from ``EO``, the healthy leaf area ``XHLAI``, the extinction coefficient ``KTRANS``,
   the CO2 factor ``TRATIO`` (``TRANS.for:195-288``) and ``EVAP``, which caps ``EOP`` at
   ``EO - EOP_reduc - EVAP`` (:func:`potential_transpiration`, process
   :func:`spam_potential_transpiration`).

Supported configuration (the 58 DSSAT maize example treatments and the CA-TPA DSSAT case, every one
``EVAPO = R``, no flood, no ASCE dual-Kc): ``KE < 0`` in ``PSE`` and ``KCB < 0``, ``MEEVP /= 'H'`` in
``TRANS``. Not supported (declared deviations): the ASCE dual-Kc branches (``EOS = KE REFET``,
``EOP = KCB REFET``), the hourly VPD branch of ``TRANS`` (``MEEVP = 'H'``), the flood evaporation
``FLOOD_EVAP`` (``SPAM.for:324-332``), and the zonal energy balance ``ETPHOT`` (``MEEVP = 'Z'`` or
``MEPHO = 'L'``).

Conservation: both processes write potential rates only (port P5, ``iface.pet``); no stock
changes here. The actual soil evaporation is booked by the soil water module, the actual
transpiration by the root water uptake (``EP = MIN(EOP, 10 TRWUP)``, ``SPAM.for:392-398``, part
of the extraction ``XTRACT``, not of this module).

Units: DSSAT computes in mm d-1; the port P5 carries cm d-1 (``transpiration = EOP / 10``,
``soil_evaporation = EOS / 10``, as the coupling contract fixes). ``EO`` is read from
``iface.pet.eo_priestley_taylor`` in mm d-1.

Source: DSSAT-CSM v4.8.6.0 ``SPAM/SPAM.for``, ``SPAM/PET.for`` (``PSE``), ``SPAM/TRANS.for``
(``TRANS``, ``TRATIO``), ``Weather/HMET.for`` (``VPSAT``, ``VPSLOP``). DSSAT-CSM is distributed
under the BSD 3-Clause licence, Copyright 1998-2026 DSSAT Foundation, University of Florida,
Gainesville, Florida, and International Fertilizer Development Center; this module is a
translation of those routines. Published equations: Ritchie, J.T. (1972), Water Resour. Res. 8,
1204-1213; Allen, L.H. Jr. (1986), in Enoch & Kimball (eds) Carbon Dioxide Enrichment of
Greenhouse Crops, CRC Press (the C4 stomatal resistance); Allen, R.G. et al. (1998), FAO-56.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef, numerical_guard
from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field
from agrijax.core.units import MM_PER_CM
from agrijax.iface.crop import CanopyRecord
from agrijax.iface.surface import PETFluxes, SoilAlbedo

from ._precision import as_float
from .coefficients import DSSAT_PT, PTCoefficients
from .priestley_taylor import priestley_taylor

__all__ = [
    "SPAM_COEFFICIENTS",
    "PSECoefficients",
    "PTAlbedoParams",
    "PTAlbedoState",
    "SpamCoefficients",
    "SpamPSEState",
    "SpamParams",
    "SpamTransState",
    "SpamWeather",
    "TransCoefficients",
    "potential_soil_evaporation",
    "potential_transpiration",
    "spam_potential_soil_evaporation",
    "spam_potential_transpiration",
    "spam_priestley_taylor",
    "transpiration_ratio",
    "vpsat",
    "vpslop",
]

_REF = "dssat-4.8.6.0"
_RITCHIE = "Ritchie (1972)"
_ALLEN86 = "Allen (1986)"
_FAO56 = "Allen et al. (1998), FAO-56"


def _at(file_line: str, routine: str, statement: str, paper: str = "", note: str = "") -> Provenance:
    return Provenance.at(_REF, file_line, routine=routine, statement=statement, paper=paper, note=note)


# ------------------------------------------------------------------------ coefficients
class PSECoefficients(Coefficients):
    """``PSE``: potential soil evaporation from ``EO`` and ``XLAI`` (``SPAM/PET.for:1448-1496``)."""

    old_lai_split: float = coef(
        1.0,
        "m2 m-2",
        "XLAI up to which the old linear form applies (KSEVAP <= 0)",
        _at("SPAM/PET.for:1473", "PSE", "IF (XLAI .LE. 1.0) THEN", _RITCHIE),
    )
    old_linear_slope: float = coef(
        0.39,
        "m-2 m2",
        "decrease of EOS / EO per unit XLAI below the split (KSEVAP <= 0)",
        _at("SPAM/PET.for:1476", "PSE", "EOS = EO*(1.0 - 0.39*XLAI)", _RITCHIE),
    )
    old_exp_divisor: float = coef(
        1.1,
        "-",
        "divisor of EO in the exponential form above the split (KSEVAP <= 0)",
        _at("SPAM/PET.for:1478", "PSE", "EOS = EO/1.1*EXP(-0.4*XLAI)", _RITCHIE),
    )
    old_exp_k: float = coef(
        0.4,
        "m-2 m2",
        "extinction coefficient of the exponential form above the split (KSEVAP <= 0)",
        _at("SPAM/PET.for:1478", "PSE", "EOS = EO/1.1*EXP(-0.4*XLAI)", _RITCHIE),
    )


class TransCoefficients(Coefficients):
    """``TRANS`` and ``TRATIO`` (``SPAM/TRANS.for``), ``VPSAT`` / ``VPSLOP`` (``Weather/HMET.for``)."""

    lai_min_trans: float = coef(
        1e-6,
        "m2 m-2",
        "XHLAI above which SPAM calls TRANS (else EOP = 0)",
        _at("SPAM/SPAM.for:378", "SPAM", "IF (XHLAI > 1.E-6) THEN"),
        calibrate=False,
    )
    lai_min_tratio: float = coef(
        0.01,
        "m2 m-2",
        "XHLAI below which the CO2 factor TRATIO is 1",
        _at("SPAM/TRANS.for:211", "TRATIO", "IF (XHLAI .LT. 0.01) THEN"),
        calibrate=False,
    )
    km_d_per_m_s: float = coef(
        86.4,
        "km d-1 m-1 s",
        "wind run per unit wind speed (86400 s d-1 / 1000 m km-1), the reference's own conversion",
        _at("SPAM/TRANS.for:218", "TRATIO", "UAVG = WINDSP / 86.4", note="unit conversion"),
        calibrate=False,
    )
    rb: float = coef(
        10.0,
        "s m-1",
        "leaf boundary-layer resistance added to the C4 stomatal resistance",
        _at("SPAM/TRANS.for:219", "TRATIO", "RB     = 10."),
    )
    co2_ref: float = coef(
        330.0,
        "ppm",
        "CO2 concentration of the reference resistance (TRATIO = 1 at this CO2)",
        _at(
            "SPAM/TRANS.for:244",
            "TRATIO",
            "RLF  = (1.0/(0.0328 - 5.49E-5*330.0 + 2.96E-8 * 330.0**2)) + RB",
            _ALLEN86,
        ),
        calibrate=False,
    )
    c4_a: float = coef(
        0.0328,
        "m s-1",
        "intercept of the C4 leaf conductance quadratic in CO2 (Allen 1986 eq. 7, corn)",
        _at(
            "SPAM/TRANS.for:245",
            "TRATIO",
            "RLFC = (1.0/(0.0328 - 5.49E-5* CO2  + 2.96E-8 * CO2  **2)) + RB",
            _ALLEN86,
        ),
    )
    c4_b: float = coef(
        5.49e-5,
        "m s-1 ppm-1",
        "linear decrease of the C4 leaf conductance with CO2 (Allen 1986 eq. 7)",
        _at(
            "SPAM/TRANS.for:245",
            "TRATIO",
            "RLFC = (1.0/(0.0328 - 5.49E-5* CO2  + 2.96E-8 * CO2  **2)) + RB",
            _ALLEN86,
        ),
    )
    c4_c: float = coef(
        2.96e-8,
        "m s-1 ppm-2",
        "quadratic term of the C4 leaf conductance in CO2 (Allen 1986 eq. 7)",
        _at(
            "SPAM/TRANS.for:245",
            "TRATIO",
            "RLFC = (1.0/(0.0328 - 5.49E-5* CO2  + 2.96E-8 * CO2  **2)) + RB",
            _ALLEN86,
        ),
    )
    c3_a: float = coef(
        9.72,
        "s m-1",
        "intercept of the C3 leaf stomatal resistance in CO2",
        _at("SPAM/TRANS.for:249", "TRATIO", "RLFC = 9.72 + 0.0757 *  CO2  + 10.0", _ALLEN86),
    )
    c3_b: float = coef(
        0.0757,
        "s m-1 ppm-1",
        "increase of the C3 leaf stomatal resistance per ppm CO2",
        _at("SPAM/TRANS.for:249", "TRATIO", "RLFC = 9.72 + 0.0757 *  CO2  + 10.0", _ALLEN86),
    )
    c3_rb: float = coef(
        10.0,
        "s m-1",
        "boundary-layer resistance added to the C3 stomatal resistance",
        _at("SPAM/TRANS.for:249", "TRATIO", "RLFC = 9.72 + 0.0757 *  CO2  + 10.0"),
    )
    active_lai_fraction: float = coef(
        0.5,
        "-",
        "fraction of the reference LAI that is active (bulk canopy resistance RL = RLF / (0.5 LAI_ref))",
        _at("SPAM/TRANS.for:269", "TRATIO", "RL = RLF / (0.5 * 2.88)", _FAO56),
    )
    lai_ref: float = coef(
        2.88,
        "m2 m-2",
        "LAI of the FAO-56 reference crop used for the canopy resistances",
        _at("SPAM/TRANS.for:269", "TRATIO", "RL = RLF / (0.5 * 2.88)", _FAO56),
    )
    ra_numerator: float = coef(
        208.0,
        "m",
        "aerodynamic resistance times wind speed of the FAO-56 reference crop (RA = 208 / u2)",
        _at("SPAM/TRANS.for:273", "TRATIO", "RA = 208. / UAVG", _FAO56),
    )
    pa_per_hpa: float = coef(
        100.0,
        "-",
        "Pa per hPa: VPSLOP (Pa K-1) to the unit of GAMMA (hPa K-1; the unit grammar has no hPa)",
        _at("SPAM/TRANS.for:280", "TRATIO", "DELTA = VPSLOP(TAVG) / 100.0", note="unit conversion"),
        calibrate=False,
    )
    lhv_0: float = coef(
        2500.9,
        "J g-1",
        "latent heat of vaporisation at 0 degC",
        _at("SPAM/TRANS.for:281", "TRATIO", "LHV    = 2500.9 - 2.345*TAVG"),
        calibrate=False,
    )
    lhv_slope: float = coef(
        2.345,
        "J g-1 degC-1",
        "decrease of the latent heat of vaporisation per degC",
        _at("SPAM/TRANS.for:281", "TRATIO", "LHV    = 2500.9 - 2.345*TAVG"),
        calibrate=False,
    )
    pressure: float = coef(
        1013.0,
        "-",
        "air pressure of the psychrometric constant, in hPa (the unit grammar has no hPa)",
        _at("SPAM/TRANS.for:282", "TRATIO", "GAMMA  = 1013.0*1.005/(LHV*0.622)"),
        calibrate=False,
    )
    cp_air: float = coef(
        1.005,
        "J g-1",
        "specific heat of air per degC (J g-1 K-1)",
        _at("SPAM/TRANS.for:282", "TRATIO", "GAMMA  = 1013.0*1.005/(LHV*0.622)"),
        calibrate=False,
    )
    mw_ratio: float = coef(
        0.622,
        "-",
        "ratio of the molecular weights of water vapour and dry air",
        _at("SPAM/TRANS.for:282", "TRATIO", "GAMMA  = 1013.0*1.005/(LHV*0.622)"),
        calibrate=False,
    )
    vpsat_e0: float = coef(
        610.78,
        "Pa",
        "saturation vapour pressure at 0 degC (Tetens 1930)",
        _at("Weather/HMET.for:643", "VPSAT", "VPSAT = 610.78 * EXP(17.269*T/(T+237.30))", "Tetens (1930)"),
        calibrate=False,
    )
    vpsat_a: float = coef(
        17.269,
        "-",
        "Tetens coefficient a of the saturation vapour pressure",
        _at("Weather/HMET.for:643", "VPSAT", "VPSAT = 610.78 * EXP(17.269*T/(T+237.30))", "Tetens (1930)"),
        calibrate=False,
    )
    vpsat_b: float = coef(
        237.30,
        "degC",
        "Tetens coefficient b of the saturation vapour pressure",
        _at("Weather/HMET.for:643", "VPSAT", "VPSAT = 610.78 * EXP(17.269*T/(T+237.30))", "Tetens (1930)"),
        calibrate=False,
    )
    mw_water: float = coef(
        18.0,
        "-",
        "molecular weight of water in g mol-1 (Clausius-Clapeyron slope; the unit grammar has no mol)",
        _at(
            "Weather/HMET.for:677",
            "VPSLOP",
            "VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)",
            "Brutsaert (1982)",
        ),
        calibrate=False,
    )
    lhv_vpslop_0: float = coef(
        2501.0,
        "J g-1",
        "latent heat of vaporisation at 0 degC in VPSLOP",
        _at(
            "Weather/HMET.for:677",
            "VPSLOP",
            "VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)",
            "Brutsaert (1982)",
        ),
        calibrate=False,
    )
    lhv_vpslop_slope: float = coef(
        2.373,
        "J g-1 degC-1",
        "decrease of the latent heat per degC in VPSLOP",
        _at(
            "Weather/HMET.for:677",
            "VPSLOP",
            "VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)",
            "Brutsaert (1982)",
        ),
        calibrate=False,
    )
    r_gas: float = coef(
        8.314,
        "-",
        "universal gas constant in J mol-1 K-1 (the unit grammar has no mol)",
        _at(
            "Weather/HMET.for:677",
            "VPSLOP",
            "VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)",
            "Brutsaert (1982)",
        ),
        calibrate=False,
    )
    kelvin_offset: float = coef(
        273.0,
        "degC",
        "offset of the kelvin temperature as the reference writes it (273.0, not 273.15)",
        _at(
            "Weather/HMET.for:677",
            "VPSLOP",
            "VPSLOP = 18.0 * (2501.0-2.373*T) * VPSAT(T) / (8.314*(T+273.0)**2)",
            "Brutsaert (1982)",
        ),
        calibrate=False,
    )


class SpamCoefficients(Coefficients):
    """The numbers of the SPAM partition: ``pse`` (``PSE``) and ``trans`` (``TRANS``, ``TRATIO``)."""

    pse: PSECoefficients = eqx.field(default_factory=PSECoefficients)
    trans: TransCoefficients = eqx.field(default_factory=TransCoefficients)


#: the DSSAT-CSM v4.8.6.0 values
SPAM_COEFFICIENTS = SpamCoefficients()

# RA = 208 / UAVG is infinite at zero wind (RL / RA = 0 in the reference); the floor keeps it finite
_UAVG_MIN = numerical_guard(
    "spam.tratio.uavg_min", 1e-12, "floor of the wind speed in RA = 208 / UAVG (zero wind: RL / RA -> 0)"
)


# ------------------------------------------------------------------------ kernels
def potential_soil_evaporation(
    eo: ArrayLike, xlai: ArrayLike, ksevap: ArrayLike, c: PSECoefficients = SPAM_COEFFICIENTS.pse
) -> Array:
    """``PSE`` (``KE < 0``): potential soil evaporation ``EOS`` [mm d-1] from ``EO`` [mm d-1].

    ``KSEVAP > 0``: ``EOS = EO exp(-KSEVAP XLAI)``; ``KSEVAP <= 0`` (no crop value, e.g. -99 before
    a crop module sets it): ``EO (1 - 0.39 XLAI)`` for ``XLAI <= 1``, else ``EO / 1.1 exp(-0.4
    XLAI)``; floored at 0. Both branches are finite everywhere.

    Source: DSSAT-CSM v4.8.6.0 SPAM/PET.for PSE, lines 1468-1493 (BSD-3).
    """
    eo, xlai, ks = as_float(eo), as_float(xlai), as_float(ksevap)
    new = eo * jnp.exp(-ks * xlai)
    lin = eo * (c.old_lai_split - c.old_linear_slope * xlai)
    ex = eo / c.old_exp_divisor * jnp.exp(-c.old_exp_k * xlai)
    old = jnp.where(xlai <= c.old_lai_split, lin, ex)
    return jnp.maximum(jnp.where(ks <= 0.0, old, new), 0.0)


def vpsat(t: ArrayLike, c: TransCoefficients = SPAM_COEFFICIENTS.trans) -> Array:
    """Saturation vapour pressure [Pa] at ``t`` [degC] (Tetens).

    Source: DSSAT-CSM v4.8.6.0 Weather/HMET.for VPSAT, line 643 (BSD-3).
    """
    t = as_float(t)
    return c.vpsat_e0 * jnp.exp(c.vpsat_a * t / (t + c.vpsat_b))


def vpslop(t: ArrayLike, c: TransCoefficients = SPAM_COEFFICIENTS.trans) -> Array:
    """Slope of the saturation vapour pressure curve [Pa K-1] at ``t`` [degC] (Clausius-Clapeyron).

    Source: DSSAT-CSM v4.8.6.0 Weather/HMET.for VPSLOP, line 677 (BSD-3).
    """
    t = as_float(t)
    return (
        c.mw_water
        * (c.lhv_vpslop_0 - c.lhv_vpslop_slope * t)
        * vpsat(t, c)
        / (c.r_gas * (t + c.kelvin_offset) ** 2)
    )


def transpiration_ratio(
    co2: ArrayLike,
    tavg: ArrayLike,
    windsp: ArrayLike,
    xhlai: ArrayLike,
    *,
    c4: bool,
    c: TransCoefficients = SPAM_COEFFICIENTS.trans,
) -> Array:
    """``TRATIO``: relative transpiration at ``co2`` [ppm] against 330 ppm (1 for ``XHLAI < 0.01``).

    Bulk canopy resistances ``RL = RLF / (0.5 * 2.88)`` at 330 ppm and ``RLC`` at ``co2`` from the
    leaf resistances (C4: ``1 / (0.0328 - 5.49e-5 CO2 + 2.96e-8 CO2^2) + RB``; C3: ``9.72 + 0.0757
    CO2 + 10``), aerodynamic resistance ``RA = 208 / u`` (``u = WINDSP / 86.4`` m s-1), ``DELTA =
    VPSLOP(TAVG) / 100``, ``GAMMA = 1013 * 1.005 / (LHV * 0.622)``, ``LHV = 2500.9 - 2.345 TAVG``;
    ``TRATIO = (DELTA + GAMMA (1 + RL / RA)) / (DELTA + GAMMA (1 + RLC / RA))``. ``c4`` is the
    reference's crop list test (``MZ, ML, SG, SC, SW, BM, BH, BR, NP, SI``: C4).

    Source: DSSAT-CSM v4.8.6.0 SPAM/TRANS.for TRATIO, lines 211-286 (BSD-3); Allen (1986) eq. 7.
    """
    co2, tavg, windsp, xhlai = as_float(co2), as_float(tavg), as_float(windsp), as_float(xhlai)
    uavg = windsp / c.km_d_per_m_s
    if c4:
        rlf = 1.0 / (c.c4_a - c.c4_b * c.co2_ref + c.c4_c * c.co2_ref**2) + c.rb
        rlfc = 1.0 / (c.c4_a - c.c4_b * co2 + c.c4_c * co2**2) + c.rb
    else:
        rlf = c.c3_a + c.c3_b * c.co2_ref + c.c3_rb
        rlfc = c.c3_a + c.c3_b * co2 + c.c3_rb
    rl = rlf / (c.active_lai_fraction * c.lai_ref)
    rlc = rlfc / (c.active_lai_fraction * c.lai_ref)
    ra = c.ra_numerator / jnp.maximum(uavg, _UAVG_MIN)
    delta = vpslop(tavg, c) / c.pa_per_hpa
    lhv = c.lhv_0 - c.lhv_slope * tavg
    gamma = c.pressure * c.cp_air / (lhv * c.mw_ratio)
    ratio = (delta + gamma * (1.0 + rl / ra)) / (delta + gamma * (1.0 + rlc / ra))
    return jnp.where(xhlai < c.lai_min_tratio, 1.0, ratio)


def potential_transpiration(
    eo: ArrayLike,
    xhlai: ArrayLike,
    ktrans: ArrayLike,
    trat: ArrayLike,
    evap: ArrayLike,
    c: TransCoefficients = SPAM_COEFFICIENTS.trans,
) -> Array:
    """``TRANS`` (``KCB < 0``, ``MEEVP /= 'H'``) as SPAM calls it: potential transpiration ``EOP``.

    ``FDINT = 1 - exp(-KTRANS XHLAI)``; ``EOP = EO FDINT TRAT``, capped at ``EO - EO FDINT (1 -
    TRAT) - EVAP`` (``EVAP`` the day's actual soil, mulch and flood evaporation, mm d-1) and floored
    at 0; ``EOP = 0`` when ``XHLAI <= 1e-6`` (SPAM does not call ``TRANS``). mm d-1 throughout.

    Source: DSSAT-CSM v4.8.6.0 SPAM/TRANS.for TRANS, lines 82-142, and SPAM/SPAM.for lines 378-387 (BSD-3).
    """
    eo, xhlai, kt, trat, evap = (as_float(x) for x in (eo, xhlai, ktrans, trat, evap))
    fdint = 1.0 - jnp.exp(-kt * xhlai)
    eop0 = eo * fdint
    eop_reduc = eop0 * (1.0 - trat)
    eop = eop0 * trat
    eop_max = eo - eop_reduc - evap
    eop = jnp.maximum(jnp.minimum(eop, eop_max), 0.0)
    return jnp.where(xhlai > c.lai_min_trans, eop, 0.0)


# ------------------------------------------------------------------------ processes
class SpamWeather(Forcing):
    """The weather ``TRANS`` reads (DSSAT ``WEATHER`` record), time axis first when stacked.

    ``tavg`` is DSSAT's daily mean of the hourly air temperatures (``Weather/HMET.for``, lines 68-116),
    not ``(TMAX + TMIN) / 2``; the day assembly computes it with the weather module.
    """

    tavg: Array = field(
        unit="degC", dims="T", fortran_name="TAVG", description="mean of the hourly air temperature"
    )
    wind_run: Array = field(unit="km d-1", dims="T", fortran_name="WINDSP", description="wind run at 2 m")
    co2: Array = field(unit="ppm", dims="T", fortran_name="CO2", description="atmospheric CO2 concentration")
    srad: Array = field(
        unit="MJ m-2 d-1",
        dims="T",
        fortran_name="SRAD",
        description="solar radiation (PETPT; None where only TRANS runs)",
        default=None,
    )
    tmax: Array = field(
        unit="degC",
        dims="T",
        fortran_name="TMAX",
        description="maximum air temperature (PETPT)",
        default=None,
    )
    tmin: Array = field(
        unit="degC",
        dims="T",
        fortran_name="TMIN",
        description="minimum air temperature (PETPT)",
        default=None,
    )


class SpamParams(Params):
    """Crop constants of the partition: the extinction coefficients the crop module passes to SPAM
    (``plant.for``: ``KSEVAP = KTRANS = KEP`` for CERES-Maize, ``KEP`` from ``MZ_PHENOL.for:338``) and
    the C4 switch of ``TRATIO``."""

    ksevap: Array = field(
        dims=(),
        unit="m-2 m2",
        description="extinction coefficient of the potential soil evaporation (<= 0: old form)",
        fortran_name="KSEVAP",
    )
    ktrans: Array = field(
        dims=(),
        unit="m-2 m2",
        description="extinction coefficient of the potential transpiration",
        fortran_name="KTRANS",
    )
    c4: bool = field(
        description="C4 crop in TRATIO (CROP in MZ, ML, SG, SC, SW, BM, BH, BR, NP, SI)",
        static=True,
        default=True,
    )
    coefficients: SpamCoefficients | None = field(
        description="the coefficients of PSE / TRANS / TRATIO (None: the DSSAT-CSM v4.8.6.0 values)",
        default=None,
    )

    @property
    def coeffs(self) -> SpamCoefficients:
        """The coefficients in force (:attr:`coefficients` or :data:`SPAM_COEFFICIENTS`)."""
        return SPAM_COEFFICIENTS if self.coefficients is None else self.coefficients


class SpamPSEState(State):
    """Module state of the potential soil evaporation: no fields of its own, two ports::

    canopy  CanopyRecord  P6  iface.canopy.<slot>  read (lai = XLAI, yesterday's)
    pet     PETFluxes     P5  iface.pet            eo_priestley_taylor read (mm d-1), soil_evaporation written
    """

    canopy: CanopyRecord = port(description="the crop's canopy (P6), yesterday's record")
    pet: PETFluxes = port(description="the potential fluxes (P5): EO read, EOS written")


class SpamTransState(State):
    """Module state of the potential transpiration: the day's actual evaporation (bound to the soil
    water module's ``EVAP``, a same-day read after the soil evaporation) and two ports::

        canopy       CanopyRecord  P6  iface.canopy.<slot>  read (lai = XHLAI)
        pet          PETFluxes     P5  iface.pet            eo_priestley_taylor read, transpiration written
        evaporation  Array             PD1  iface.evaporation  EVAP = ES + EM + EF [mm d-1], read
    """

    canopy: CanopyRecord = port(description="the crop's canopy (P6), yesterday's record")
    pet: PETFluxes = port(description="the potential fluxes (P5): EO read, EOP written")
    evaporation: Array = port(
        unit="mm d-1",
        dims=(),
        fortran_name="EVAP",
        description="today's actual soil + mulch + flood evaporation, written by the soil water module",
    )


_PSE_DEVIATES = (
    (
        "the ASCE dual-Kc branch EOS = KE REFET (KE >= 0) is not implemented",
        "EVAPO = R in every supported run: KE = -99 (SPAM.for:148-151)",
        "tests/integration/test_spam_evap_dssat.py (the dumped EOS equals the KE < 0 form)",
    ),
    (
        "DSSAT single precision (REAL*4) is not reproduced",
        "the kernel runs in the precision of its inputs",
        "tests/integration/test_spam_evap_dssat.py tolerances (measured REAL*4 rounding)",
    ),
    (
        "XLAI is the canopy port's green LAI; with several crops in the slot, their sum",
        "CERES-Maize sets XLAI = XHLAI = LAI (MZ_GROSUB.for:1818-1819); one canopy per field "
        "(the PET split between crops is not defined)",
        "spam_dssat.py spam_potential_soil_evaporation docstring",
    ),
)


@process(
    reads=("pet.eo_priestley_taylor", "canopy.lai"),
    writes=("pet.soil_evaporation",),
    source="DSSAT-CSM v4.8.6.0 SPAM/PET.for PSE, SPAM/SPAM.for line 314 (BSD-3)",
    fortran_name="PSE",
    key="pet/spam_pse@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (gfortran 13, instrumented build for the day-by-day comparison)",
    sources=(
        (
            "potential soil evaporation EOS from EO and XLAI (KSEVAP > 0 and <= 0 forms)",
            "SPAM/PET.for:1468-1493",
        ),
        ("called with XLAI after PET in the RATE step", "SPAM/SPAM.for:310-314"),
    ),
    deviates=_PSE_DEVIATES,
)
def spam_potential_soil_evaporation(state: SpamPSEState, params: SpamParams, forcing_t: Any) -> SpamPSEState:
    """``PSE``: today's potential soil evaporation into P5 ``soil_evaporation`` [cm d-1] from
    ``EO`` (P5 ``eo_priestley_taylor``, mm d-1, written earlier the same day by the PT entry) and
    yesterday's LAI (P6, summed over the slot's crops). Reads no forcing.

    Source: DSSAT-CSM v4.8.6.0 SPAM/PET.for PSE lines 1468-1493, SPAM/SPAM.for line 314 (BSD-3).
    """
    pet = state.pet
    xlai = jnp.sum(state.canopy.lai, axis=-1)
    eos = potential_soil_evaporation(pet.eo_priestley_taylor, xlai, params.ksevap, params.coeffs.pse)
    old = pet.soil_evaporation
    return eqx.tree_at(
        lambda s: s.pet.soil_evaporation, state, (eos / MM_PER_CM).astype(jnp.result_type(old))
    )


@process(
    reads=("pet.eo_priestley_taylor", "canopy.lai", "evaporation"),
    writes=("pet.transpiration",),
    source="DSSAT-CSM v4.8.6.0 SPAM/TRANS.for TRANS, TRATIO; Weather/HMET.for VPSLOP (BSD-3)",
    fortran_name="TRANS",
    key="pet/spam_trans@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (gfortran 13, instrumented build for the day-by-day comparison)",
    sources=(
        ("EOP = EO FDINT TRAT, capped by EO - EOP_reduc - EVAP", "SPAM/TRANS.for:82-142"),
        (
            "relative transpiration TRATIO (CO2, FAO-56 resistances)",
            "SPAM/TRANS.for:195-288; Allen (1986) eq. 7",
        ),
        ("VPSAT, VPSLOP", "Weather/HMET.for:636-679"),
        ("TRANS called when XHLAI > 1e-6, after the soil evaporation", "SPAM/SPAM.for:372-387"),
    ),
    deviates=(
        (
            "the dual-Kc branch EOP = KCB REFET and the hourly VPD branch (MEEVP = H) are not implemented",
            "EVAPO = R in every supported run (KCB = -99)",
            "tests/integration/test_spam_evap_dssat.py",
        ),
        (
            "DSSAT single precision (REAL*4) is not reproduced; the zero-wind RA = 208 / 0 is floored",
            "the kernel runs in the precision of its inputs; finite values and gradients at zero wind",
            "tests/integration/test_spam_evap_dssat.py tolerances",
        ),
        (
            "EVAP enters TRANS as an argument of the SPAM driver (same-day cycle PSE -> soil evaporation "
            "-> TRANS); here it is read from the day's evaporation record",
            "the soil evaporation is a process of its own and writes the record PD1 before TRANS runs",
            "spam_dssat.py SpamTransState docstring",
        ),
        (
            "XHLAI is the canopy port's green LAI; with several crops in the slot, their sum",
            "one canopy per field (the PET split between crops is not defined)",
            "spam_dssat.py spam_potential_transpiration docstring",
        ),
    ),
)
def spam_potential_transpiration(
    state: SpamTransState, params: SpamParams, forcing_t: SpamWeather
) -> SpamTransState:
    """``TRANS``: today's potential transpiration into P5 ``transpiration`` [cm d-1] from ``EO``
    (P5, mm d-1), yesterday's LAI (P6), today's actual evaporation (port ``evaporation``, mm d-1)
    and the weather's ``tavg``, ``wind_run``, ``co2`` (``TRATIO``).

    Forcing fields read: tavg, wind_run, co2.

    Source: DSSAT-CSM v4.8.6.0 SPAM/TRANS.for TRANS and TRATIO, SPAM/SPAM.for lines 378-387 (BSD-3).
    """
    c = params.coeffs.trans
    pet = state.pet
    xhlai = jnp.sum(state.canopy.lai, axis=-1)
    trat = transpiration_ratio(forcing_t.co2, forcing_t.tavg, forcing_t.wind_run, xhlai, c4=params.c4, c=c)
    eop = potential_transpiration(pet.eo_priestley_taylor, xhlai, params.ktrans, trat, state.evaporation, c)
    old = pet.transpiration
    return eqx.tree_at(lambda s: s.pet.transpiration, state, (eop / MM_PER_CM).astype(jnp.result_type(old)))


# ------------------------------------------------------------------------ PETPT with the daily MSALB
class PTAlbedoParams(Params):
    """The ``PETPT`` coefficients (``None``: :data:`~agrijax.processes.pet.coefficients.DSSAT_PT`)."""

    coefficients: PTCoefficients | None = field(
        description="the PETPT coefficients (None: the DSSAT-CSM v4.8.6.0 values)", default=None
    )

    @property
    def coeffs(self) -> PTCoefficients:
        """The coefficients in force."""
        return DSSAT_PT if self.coefficients is None else self.coefficients


class PTAlbedoState(State):
    """Module state of ``PETPT`` with the soil albedo from a port: no fields of its own, three ports::

    canopy  CanopyRecord  P6   iface.canopy.<slot>  read (lai = XHLAI, yesterday's)
    pet     PETFluxes     P5   iface.pet            eo_priestley_taylor written (mm d-1)
    albedo  SoilAlbedo    PD2  iface.soil_albedo    msalb read (SOILDYN of the same day)
    """

    canopy: CanopyRecord = port(description="the crop's canopy (P6), yesterday's record")
    pet: PETFluxes = port(description="the potential fluxes (P5): EO written")
    albedo: SoilAlbedo = port(description="the day's soil albedo (PD2): MSALB read")


@process(
    reads=("canopy.lai", "albedo.msalb"),
    writes=("pet.eo_priestley_taylor",),
    source="Priestley & Taylor (1972); Ritchie (1972); DSSAT-CSM v4.8.6.0 SPAM/PET.for PETPT (BSD-3)",
    fortran_name="PETPT",
    key="pet/priestley_taylor@dssat-4.8.6.0:port_soil_albedo",
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (gfortran 13, instrumented build for the day-by-day comparison)",
    sources=(
        ("equilibrium evaporation, LAI-dependent albedo", "Priestley & Taylor (1972); Ritchie (1972)"),
        ("TD weighting, Tmax > 35 / Tmax < 5 branches, 1e-4 mm floor", "SPAM/PET.for:871-918 (PETPT)"),
        ("ET_ALB = MSALB (no flood), the soil albedo SOILDYN computed the same day", "SPAM/SPAM.for:292-305"),
    ),
    deviates=(
        (
            "DSSAT single precision is not reproduced",
            "the kernel runs in the precision of its inputs",
            "tests/integration/test_spam_evap_dssat.py (EO within the REAL*4 limit)",
        ),
        (
            "with several crops in the slot, the LAI is their sum; under flood the reference uses 0.05",
            "one canopy per field (the PET split between crops is not defined); FLOOD = 0 in every "
            "supported run",
            "spam_dssat.py spam_priestley_taylor docstring",
        ),
    ),
)
def spam_priestley_taylor(
    state: PTAlbedoState, params: PTAlbedoParams, forcing_t: SpamWeather
) -> PTAlbedoState:
    """``PETPT`` as ``SPAM`` calls it: ``EO`` [mm d-1] into P5 ``eo_priestley_taylor`` from the weather
    (``srad``, ``tmax``, ``tmin``), yesterday's LAI (P6) and the day's soil albedo ``MSALB`` (PD2). The
    faithful computation of :func:`~agrijax.processes.pet.priestley_taylor` with the albedo read
    from a port instead of a constant parameter.

    Forcing fields read: srad, tmax, tmin.

    Source: DSSAT-CSM v4.8.6.0 SPAM/PET.for PETPT lines 871-918, SPAM/SPAM.for lines 292-305 (BSD-3).
    """
    eo = priestley_taylor(
        forcing_t.srad,
        forcing_t.tmax,
        forcing_t.tmin,
        jnp.sum(state.canopy.lai, axis=-1),
        state.albedo.msalb,
        params.coeffs,
    )
    old = state.pet.eo_priestley_taylor
    return eqx.tree_at(lambda s: s.pet.eo_priestley_taylor, state, eo.astype(jnp.result_type(old)))
