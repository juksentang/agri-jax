"""Named codes, conversions and small coefficient groups of the CERES-Maize port.

Three kinds of numbers live here so that the process and kernel code carries no bare literal
(lint rule AJ007, :mod:`agrijax.core.lint`):

* **codes** of DSSAT-CSM v4.8.6.0: the growth-stage codes ``ISTAGE`` (``MZ_PHENOL``), the
  ``CropStatus`` codes of a failed or matured crop, and the ``SEASINIT`` values of ``XSTAGE`` and
  ``MDATE``. They are labels, not responses: plain module constants, each with the
  ``MZ_PHENOL.for`` / ``MZ_GROSUB.for`` line that sets it;
* **conversions and arithmetic** the Fortran spells as literals (``1000`` g -> mg, ``0.001``
  mg -> g, the ``1e-3`` storage precision of ``RLV``, the ``/ 2`` of a two-value mean, the
  ``12`` hours of half a day, the ``YYYYDDD`` packing of dates), and the numerical guard of the
  divisors the port adds for finite gradients (:func:`agrijax.core.coefficients.numerical_guard`);
  the conversions that :mod:`agrijax.core.units` already names (``CM_PER_MM``, ``M2_PER_CM2``,
  ``M_PER_CM``, ``KG_HA_PER_G_M2``, ``HOURS_PER_DAY``) are imported from there;
* **coefficients** with unit, meaning and provenance (:func:`agrijax.core.coefficients.coef`):
  :class:`SolarCoefficients` (the sun geometry of ``DAYLEN`` and ``TWILIGHT``,
  ``Weather/SOLAR.for``) and :class:`XstageCoefficients` (the noninteger growth stage
  ``XSTAGE`` of ``MZ_PHENOL``). Both are leaves kept out of the calibration vector
  (``calibrate=False``): the first is astronomy, the second is read only by a nitrogen module
  that is not built.

Every default is the DSSAT-CSM v4.8.6.0 literal, as a Python number, so naming it does not change
a result (same value, same operation order). DSSAT-CSM is distributed under the BSD 3-clause
licence (Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center); this is an independent re-implementation of its published equations.
"""

from __future__ import annotations

from typing import Any

from agrijax.core.coefficients import Coefficients, Provenance, coef, numerical_guard

__all__ = [
    "CROP_STATUS_COLD",
    "CROP_STATUS_DROUGHT",
    "CROP_STATUS_MATURE",
    "CROP_STATUS_NO_EMERGENCE",
    "CROP_STATUS_NO_GERMINATION",
    "DEG_PER_HALF_TURN",
    "DEN_MIN",
    "FULL_TURN_PER_PI",
    "G_PER_MG",
    "HOURS_PER_HALF_DAY",
    "ISTAGE_AFTER_MATURITY",
    "ISTAGE_EFG",
    "ISTAGE_EMERGENCE",
    "ISTAGE_END_JUVENILE",
    "ISTAGE_END_LEAF_GROWTH",
    "ISTAGE_GERMINATION",
    "ISTAGE_JUVENILE",
    "ISTAGE_MATURITY",
    "ISTAGE_SOWING",
    "ISTAGE_TASSEL_INIT",
    "MDATE_NONE",
    "MG_PER_G",
    "PAIR_MEAN_DIVISOR",
    "PAIR_MEAN_WEIGHT",
    "RLV_PRECISION",
    "SOLAR_COEFFICIENTS",
    "XSTAGE_COEFFICIENTS",
    "XSTAGE_SEASINIT",
    "YRDOY_SCALE",
    "SolarCoefficients",
    "XstageCoefficients",
]

_REF = "dssat-4.8.6.0"

# ---- growth-stage codes ISTAGE (MZ_PHENOL.for, INTEGR; the comment line of each stage block) ----
#: ``ISTAGE = 1``: emergence to end of juvenile stage (MZ_PHENOL.for:639)
ISTAGE_JUVENILE = 1
#: ``ISTAGE = 2``: end of juvenile stage to tassel initiation (MZ_PHENOL.for:681)
ISTAGE_END_JUVENILE = 2
#: ``ISTAGE = 3``: tassel initiation to end of leaf growth (silking) (MZ_PHENOL.for:748)
ISTAGE_TASSEL_INIT = 3
#: ``ISTAGE = 4``: end of leaf growth to beginning of effective grain filling (MZ_PHENOL.for:783)
ISTAGE_END_LEAF_GROWTH = 4
#: ``ISTAGE = 5``: beginning to end of effective grain filling (MZ_PHENOL.for:855)
ISTAGE_EFG = 5
#: ``ISTAGE = 6``: end of effective grain filling to physiological maturity; also the stage of a
#: failed crop (MZ_PHENOL.for:878, MZ_GROSUB.for:1661)
ISTAGE_MATURITY = 6
#: ``ISTAGE = 7``: before sowing / sowing day (MZ_PHENOL.for:351 ``ISTAGE = 7`` in SEASINIT, :529)
ISTAGE_SOWING = 7
#: ``ISTAGE = 8``: sowing to germination (MZ_PHENOL.for:553)
ISTAGE_GERMINATION = 8
#: ``ISTAGE = 9``: germination to emergence (MZ_PHENOL.for:596)
ISTAGE_EMERGENCE = 9
#: ``ISTAGE = 10``: after physiological maturity (MZ_PHENOL.for:899 ``ISTAGE = 10``)
ISTAGE_AFTER_MATURITY = 10

# ---- CropStatus codes ---------------------------------------------------------------------------
#: crop matured normally (MZ_PHENOL.for:897 ``CropStatus = 1``)
CROP_STATUS_MATURE = 1
#: failure to germinate (MZ_PHENOL.for:576 ``CropStatus = 12``)
CROP_STATUS_NO_GERMINATION = 12
#: failure to emerge (MZ_PHENOL.for:624 ``CropStatus = 13``)
CROP_STATUS_NO_EMERGENCE = 13
#: crop killed by cold stress (MZ_GROSUB.for:1663, :1679 ``CropStatus = 32``)
CROP_STATUS_COLD = 32
#: crop killed by water stress (MZ_GROSUB.for:1706 ``CropStatus = 33``)
CROP_STATUS_DROUGHT = 33

# ---- SEASINIT values (MZ_PHENOL.for, SEASINIT) --------------------------------------------------
#: initial noninteger growth stage (MZ_PHENOL.for:352 ``XSTAGE = 0.1``)
XSTAGE_SEASINIT = 0.1
#: maturity date not yet set (MZ_PHENOL.for:353 ``MDATE = -99``)
MDATE_NONE = -99

# ---- conversions and arithmetic ------------------------------------------------------------------
#: g -> mg (MZ_PHENOL.for:810 ``PSKER = SUMP*1000.0/IDURP*3.4/5.0``: g plant-1 to mg plant-1)
MG_PER_G = 1000.0
#: mg -> g (MZ_GROSUB.for:1468 ``GROGRN = RGFILL*GPP*G3*0.001*(...)``: G3 in mg kernel-1 d-1)
G_PER_MG = 0.001
#: ``RLV`` is stored to 1 / RLV_PRECISION cm cm-3 (MZ_ROOTS.for:202
#: ``RLV(L) = REAL(INT(RLV(L)*1000.))/1000.``): a storage precision, not a response
RLV_PRECISION = 1000.0
#: the ``/ 2`` of the mean of two values, ``(a + b) / 2`` (MZ_PHENOL.for:457, 479, 481, 495, 505)
PAIR_MEAN_DIVISOR = 2.0
#: the ``* 0.5`` of the mean of two values (MZ_GROSUB.for:1022 ``TEMPM = (TMAX + TMIN)*0.5``)
PAIR_MEAN_WEIGHT = 0.5
#: hours of half a day: the hourly sine of MZ_PHENOL.for:495 (``SIN(3.14/12.*I)``) and the solar
#: noon of SOLAR.for:133 (``DAYL = 12.0 + 24.0*ASIN(SOC)/PI``)
HOURS_PER_HALF_DAY = 12.0
#: degrees in PI radians (SOLAR.for:121 ``RAD=PI/180.0``)
DEG_PER_HALF_TURN = 180.0
#: a full turn is 2 PI (SOLAR.for:126 ``COS(2.0*PI*(DOY+10.0)/365.0)``)
FULL_TURN_PER_PI = 2.0
#: ``YYYYDDD`` date packing of DSSAT (``YRDOY = YEAR*1000 + DOY``)
YRDOY_SCALE = 1000

#: floor of divisors that are positive for any real crop (phyllochron, kernel numbers, the
#: photoperiod denominator, layer thicknesses, PORMIN); keeps quotients and their gradients finite
DEN_MIN = numerical_guard(
    "ceres_maize.den_min",
    1e-6,
    "floor of divisors that are positive for any real crop (PHINT, G2, DJTI, DLAYR, PORMIN)",
)


# ---- DAYLEN / TWILIGHT (Weather/SOLAR.for) -------------------------------------------------------
def _solar(
    value: float, unit: str, description: str, line: int, routine: str, statement: str, **kw: Any
) -> Any:
    """A sun-geometry constant of DSSAT-CSM v4.8.6.0 ``Weather/SOLAR.for`` (not calibrated)."""
    prov = Provenance.at(
        _REF,
        f"Weather/SOLAR.for:{line}",
        routine=routine,
        statement=statement,
        paper=kw.pop("paper", ""),
        equation=kw.pop("equation", ""),
    )
    return coef(value, unit, description, prov, calibrate=False, **kw)


_SPITTERS = "Spitters et al. (1986)"


class SolarCoefficients(Coefficients):
    """The sun geometry of DSSAT ``DAYLEN`` (astronomical daylength, Spitters et al. 1986
    eqs. 16-17) and ``TWILIGHT`` (twilight-to-twilight daylength of the maize photoperiod).

    Astronomy, not a crop response: leaves kept out of the calibration vector. The values are the
    truncated ones the reference uses (``PI = 3.14159``, ``0.01745`` rad per degree).
    """

    pi: float = _solar(
        3.14159, "-", "value of pi used by DAYLEN", 121, "DAYLEN", "PARAMETER (PI=3.14159, RAD=PI/180.0)"
    )
    dec_amplitude: float = _solar(
        23.45,
        "-",
        "amplitude of the solar declination, in degrees (DEC = -23.45 cos(2 pi (DOY + 10) / 365))",
        126,
        "DAYLEN",
        "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)",
        paper=_SPITTERS,
        equation="16",
    )
    dec_doy_offset: float = _solar(
        10.0,
        "d",
        "days from the December solstice to the start of the year in the declination",
        126,
        "DAYLEN",
        "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)",
        paper=_SPITTERS,
        equation="16",
    )
    year_days: float = _solar(
        365.0,
        "d",
        "length of the year in the declination",
        126,
        "DAYLEN",
        "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)",
        paper=_SPITTERS,
        equation="16",
    )
    tw_deg_to_rad: float = _solar(
        0.01745, "-", "radians per degree in TWILIGHT", 235, "TWILIGHT", "S1 = SIN(XLAT*0.01745)"
    )
    tw_dec_amplitude: float = _solar(
        0.4093,
        "rad",
        "amplitude of the solar declination in TWILIGHT",
        237,
        "TWILIGHT",
        "DEC    = 0.4093*SIN(0.0172*(DOY-82.2))",
    )
    tw_dec_rate: float = _solar(
        0.0172,
        "rad d-1",
        "angular speed of the declination cycle in TWILIGHT (2 pi / 365)",
        237,
        "TWILIGHT",
        "DEC    = 0.4093*SIN(0.0172*(DOY-82.2))",
    )
    tw_dec_doy0: float = _solar(
        82.2,
        "d",
        "day of the year of zero declination (March equinox) in TWILIGHT",
        237,
        "TWILIGHT",
        "DEC    = 0.4093*SIN(0.0172*(DOY-82.2))",
    )
    tw_sin_depression: float = _solar(
        0.1047,
        "-",
        "sine of the sun depression that ends twilight (about 6 degrees below the horizon)",
        238,
        "TWILIGHT",
        "DLV    = ((-S1*SIN(DEC)-0.1047)/(C1*COS(DEC)))",
    )
    tw_dlv_min: float = _solar(
        -0.87,
        "-",
        "lower bound of the cosine of the twilight hour angle",
        239,
        "TWILIGHT",
        "DLV    = MIN(MAX(DLV,-0.87), 1.0)",
    )
    tw_hours_per_rad: float = _solar(
        7.639,
        "h",
        "twilight daylength per radian of hour angle (24 / pi)",
        240,
        "TWILIGHT",
        "TWILEN = 7.639*ACOS(DLV)",
    )


#: the DSSAT-CSM v4.8.6.0 values of :class:`SolarCoefficients`
SOLAR_COEFFICIENTS = SolarCoefficients()


# ---- XSTAGE (MZ_PHENOL.for) ----------------------------------------------------------------------
def _xstage(value: float, description: str, line: int, statement: str) -> Any:
    """An ``XSTAGE`` coefficient of DSSAT-CSM v4.8.6.0 ``MZ_PHENOL`` (not calibrated)."""
    prov = Provenance.at(
        _REF,
        f"Plant/CERES-Maize/MZ_PHENOL.for:{line}",
        routine="MZ_PHENOL",
        statement=statement,
        note="XSTAGE is read only by the nitrogen module, which is not built",
    )
    return coef(value, "-", description, prov, calibrate=False)


class XstageCoefficients(Coefficients):
    """The scale of the noninteger growth stage ``XSTAGE`` (``1 + 0.5 SIND`` in stage 2,
    ``1.5 + 3 SUMDTT / P3`` in stage 3, ``4.5 + 5.5 SUMDTT / P5`` in stages 4-5).

    Only the nitrogen module reads ``XSTAGE`` (it computes the nitrogen demand); with nitrogen off
    nothing does, so these are leaves kept out of the calibration vector.
    """

    ti_base: float = _xstage(1.0, "XSTAGE at the end of the juvenile stage", 687, "XSTAGE = 1.0 + 0.5*SIND")
    ti_slope: float = _xstage(
        0.5, "XSTAGE per unit photoperiod induction SIND", 687, "XSTAGE = 1.0 + 0.5*SIND"
    )
    silk_base: float = _xstage(1.5, "XSTAGE at tassel initiation", 755, "XSTAGE = 1.5 + 3.0*SUMDTT/P3")
    silk_slope: float = _xstage(
        3.0, "XSTAGE per unit SUMDTT / P3 in stage 3", 755, "XSTAGE = 1.5 + 3.0*SUMDTT/P3"
    )
    efg_base: float = _xstage(4.5, "XSTAGE at silking", 790, "XSTAGE = 4.5+5.5*SUMDTT/(P5*0.95)")
    efg_slope: float = _xstage(
        5.5, "XSTAGE per unit SUMDTT / P5 in stages 4-5", 790, "XSTAGE = 4.5+5.5*SUMDTT/(P5*0.95)"
    )


#: the DSSAT-CSM v4.8.6.0 values of :class:`XstageCoefficients`
XSTAGE_COEFFICIENTS = XstageCoefficients()
