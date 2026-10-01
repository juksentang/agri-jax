"""Daily radiation of RZWQM2 4.6: the horizontal value RTH and the hourly re-sum RTS.

RZWQM2 reads a daily solar radiation from the ``.MET`` file and then, every day and whatever the
energy options (also with ``ISHAW = 0``), rebuilds the radiation its processes use:

1. ``INPDAY`` (``RZWQM/Rzmain.for`` 3458-3600) applies the monthly weather modifier and the input
   bounds to the file value: **RTH**, the measured radiation on a horizontal surface
   [MJ m-2 d-1] (``srad_horizontal`` of port P8). It is what the Shuttleworth-Wallace PET uses
   for the net long-wave cloudiness ratio of RTH to the clear-sky radiation (``POTEVPHR``,
   ``Rzpet.for`` 1983-1988; the hourly ``HRTH`` of step 2 instead when the PET runs hourly or
   with PENFLUX, 1984-1986).
2. DSSAT 4.0 ``DAYLEN`` / ``SOLAR`` / ``HMET`` (called at ``Rzmain.for`` 1316-1329) spread RTH
   over 24 whole hours with Spitters' sine-of-elevation shape: **HRTH** [W m-2], in single
   precision (the DSSAT routines declare ``REAL``).
3. SHAW's ``CLOUDY`` (``Rzmain.for`` 1379) clips negative hours, returns the declination and the
   half-day length and a cloud fraction **CLOUDS** (used only by the PENFLUX energy balance), and
   24 calls of ``SOLAR_SHAW`` (``Rzmain.for`` 1380-1391) split each hour into direct and diffuse
   radiation on the local slope: **HRTS** = direct + diffuse [W m-2].
4. **RTS** = the sum of the 24 HRTS [MJ m-2 d-1] (``srad`` of port P8, ``.ana`` column 88): the
   radiation of the PET, of the PRMS snow model and of the crop. The whole-hour sum is not the
   daily input: 0.993 .. 1.005 of it at CA-TPA.

This module reconstructs all of them from the daily input as forcing preprocessing (in NumPy, not
differentiable; a differentiable port is deferred to the weather-forcing module). The arithmetic
follows the reference's precision, so the result is meant to equal the binary's to rounding: the
DSSAT part in float32, the SHAW part in float64, the hourly sums accumulated hour by hour in the
reference's order.

Written from the published equations and the DSSAT-CSM v4.8.6.0 source (BSD-3; its statements are
quoted in the coefficients): Spitters, C.J.T., Toussaint, H.A.J.M. and Goudriaan, J. (1986),
Agric. For. Meteorol. 38, 217-229 (eq. 6: the hourly shape and its daily integral); Flerchinger,
G.N. (2000), The Simultaneous Heat and Water (SHAW) Model: Technical Documentation, NWRC 2000-09
(solar geometry and the direct/diffuse partition of Bristow, Campbell and Saxton 1985); Flerchinger
and Yu (2007) as cited by SHAW's ``CLOUDY`` for the cloud fraction. RZWQM2 source statements are
not reproduced (file, line and routine only).

RZWQM2 runs the DSSAT 4.0 copy of ``DAYLEN``/``SOLAR``/``HMET`` (``DSSAT40/Weather``); the
statements used here are the same in DSSAT-CSM v4.8.6.0 except that 4.8.6 clamps the day length to
[0, 24] h (``Weather/SOLAR.for`` 136) and guards two divisions of ``SOLAR`` that ``ISINB`` does not
use. The clamp changes the day length only where ``|tan(dec) tan(lat)| >= 1`` (polar day or
night), where 4.0 gives 24.00003 h; this module follows 4.0, what the reference binary runs.

Not reproduced (outside the daily ``.MET`` path of the 15 reference scenarios, all flat or with a
2-degree slope, latitude 33-50 N):

* hourly weather input (``Iweather = 1``, ``INPHOUR``): HRTH is read, not generated;
* a day with RTS = 0: ``POTEVPHR`` then replaces RTS and RTH by a sunshine fraction of the
  clear-sky radiation (``Rzpet.for`` 1917-1930). :func:`rzwqm_radiation` raises on such a day
  unless ``allow_zero=True``;
* the PAR fall-backs of hourly and daily PAR derived from HRTS and RTS (``Rzmain.for`` 1390-1392).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.coefficients import coefficient_table as _coefficient_table
from agrijax.core.units import SECONDS_PER_HOUR, conversion_factor

__all__ = [
    "N_HOURS",
    "RZWQM_RADIATION",
    "DailyRadiation",
    "HmetCoefficients",
    "InpdayCoefficients",
    "RadiationCoefficients",
    "ShawCoefficients",
    "coefficient_table",
    "horizontal_radiation",
    "hourly_horizontal_radiation",
    "radiation_from_met",
    "rzwqm_radiation",
    "shaw_slope_partition",
]

REF_DSSAT = "dssat-4.8.6.0"
REF_RZWQM = "rzwqm2-4.6"
SOLAR_FOR = "Weather/SOLAR.for"
HMET_FOR = "Weather/HMET.for"
SHAW_SOLAR = "SHAW24b/SHAW_Solar.for"
SHAW_CLOUDY = "SHAW24b/SHAW_Cloudy.for"
RZMAIN = "RZWQM/Rzmain.for"

SPITTERS = "Spitters, Toussaint and Goudriaan (1986)"
SHAW_DOC = "Flerchinger (2000), SHAW technical documentation"
RZWQM_BOOK = "Ahuja et al. (2000)"

#: hours of the reference's hourly disaggregation (the hourly array size of ``Rzmain.for`` 241;
#: the ``TS`` of DSSAT HMET)
N_HOURS = 24
#: months of the monthly weather-modifier table (``Rzmain.for`` 9187)
N_MONTHS = 12
#: J in one MJ (the MJ/J conversions of DSSAT HRAD and of the RTS sum, ``Rzmain.for`` 1389) and
#: the fraction in one percent (the percent modifiers of INPDAY, ``Rzmain.for`` 3556)
_J_IN_MJ = conversion_factor("MJ", "J")
_FRACTION_OF_PERCENT = conversion_factor("%", "-")
#: the identity modifier, 100 % (one in percent): the modifier of all 15 reference scenarios
_IDENTITY_PERCENT = conversion_factor("-", "%")
#: sun positions per call of SOLAR_SHAW (``SHAW_Solar.for`` 41 and ``Rzmain.for`` 1383 give a
#: one-hour step): the sun position is averaged over the two ends of the hour
_ENDPOINTS = 2.0

_F32 = np.float32


def _rn(fn: Any, x: np.ndarray) -> np.ndarray:
    """A float32 transcendental rounded to nearest: evaluated in float64, rounded once to float32.

    The reference's single-precision ``SIN``/``COS``/``TAN``/``ASIN`` (Intel libm) are accurate to
    well under one float32 unit; NumPy's own float32 kernels are not always, and near sunrise and
    sunset (``BETAS = SSIN + CCOS COS(HANGL)`` cancels) one unit of ``COS`` moves the hourly value
    by several units.
    """
    return fn(np.asarray(x, dtype=np.float64)).astype(_F32)


def _dssat(file_line: str, routine: str, statement: str, **kw: Any) -> Provenance:
    """DSSAT-CSM v4.8.6.0 provenance with the quoted statement (BSD-3)."""
    return Provenance.at(REF_DSSAT, file_line, routine=routine, statement=statement, **kw)


def _rz(file: str, line: int, routine: str, paper: str = SHAW_DOC, **kw: Any) -> Provenance:
    """RZWQM2 4.6 provenance: file, line, routine and the published equation (no statement)."""
    return Provenance(REF_RZWQM, file=file, line=line, routine=routine, paper=paper, **kw)


_CONST = "constant of the reference arithmetic, not calibrated"


class HmetCoefficients(Coefficients):
    """DSSAT ``DAYLEN``, ``SOLAR`` (``ISINB``), ``HANG`` and ``HRAD``: daily -> hourly radiation (float32)."""

    pi: float = coef(
        3.14159,
        "-",
        "value of pi of the DSSAT weather routines",
        _dssat(f"{SOLAR_FOR}:121", "DAYLEN", "PARAMETER (PI=3.14159, RAD=PI/180.0)", note=_CONST),
        calibrate=False,
    )
    deg_per_half_turn: float = coef(
        180.0,
        "deg",
        "degrees in pi radians (RAD = PI/180)",
        _dssat(f"{HMET_FOR}:206", "HANG", "PARAMETER (PI=3.14159, RAD=PI/180.)", note=_CONST),
        calibrate=False,
    )
    declination_amplitude: float = coef(
        -23.45,
        "deg",
        "amplitude of the solar declination (negative: minimum at the December solstice)",
        _dssat(
            f"{SOLAR_FOR}:126",
            "DAYLEN",
            "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)",
            paper=SPITTERS,
            equation="16",
        ),
    )
    declination_day_offset: float = coef(
        10.0,
        "d",
        "days from the December solstice to 1 January in the declination cosine",
        _dssat(f"{SOLAR_FOR}:126", "DAYLEN", "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)"),
    )
    full_turn: float = coef(
        2.0,
        "-",
        "multiple of pi of one declination cycle (2 pi per year)",
        _dssat(f"{SOLAR_FOR}:126", "DAYLEN", "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)", note=_CONST),
        calibrate=False,
    )
    year_length: float = coef(
        365.0,
        "d",
        "days of the declination cycle",
        _dssat(f"{SOLAR_FOR}:126", "DAYLEN", "DEC = -23.45 * COS(2.0*PI*(DOY+10.0)/365.0)", note=_CONST),
        calibrate=False,
    )
    noon: float = coef(
        12.0,
        "h",
        "solar noon, and half the day length of an equinox day",
        _dssat(
            f"{SOLAR_FOR}:133", "DAYLEN", "DAYL = 12.0 + 24.0*ASIN(SOC)/PI", paper=SPITTERS, equation="17"
        ),
        calibrate=False,
    )
    day_hours: float = coef(
        24.0,
        "h",
        "hours of one Earth rotation (DAYL = 12 + 24 asin(SOC) / pi; ISINB's 24/pi)",
        _dssat(f"{SOLAR_FOR}:133", "DAYLEN", "DAYL = 12.0 + 24.0*ASIN(SOC)/PI", note=_CONST),
        calibrate=False,
    )
    halves: float = coef(
        2.0,
        "-",
        "sunrise and sunset are half the day length before and after noon",
        _dssat(f"{SOLAR_FOR}:137", "DAYLEN", "SNUP = 12.0 - DAYL/2.0", note=_CONST),
        calibrate=False,
    )
    hours_per_half_turn: float = coef(
        12.0,
        "h",
        "hours per pi radians of hour angle (HANGL = (HS - 12) pi / 12)",
        _dssat(f"{HMET_FOR}:213", "HANG", "HANGL = (HS-12.0)*PI/12.0", note=_CONST),
        calibrate=False,
    )
    elevation_trigger: float = coef(
        1.0e-4,
        "deg",
        "solar elevation above which the elevation is raised to elevation_min",
        _dssat(f"{HMET_FOR}:224", "HANG", "IF (BETA .GT. 1.E-4) THEN"),
    )
    elevation_min: float = coef(
        1.0,
        "deg",
        "minimum solar elevation of an hour with the sun above the horizon",
        _dssat(f"{HMET_FOR}:225", "HANG", "BETA = MAX(BETA, 1.0)", note="HMET change log 11/23/1993 NBP"),
    )
    spitters_b: float = coef(
        0.4,
        "-",
        "coefficient of the second sine term of the hourly global radiation shape sinB (1 + b sinB)",
        _dssat(
            f"{HMET_FOR}:358",
            "HRAD",
            "RADHR = SINB * (1.0+0.4*SINB) * SRAD*1.0E6 / ISINB",
            paper=SPITTERS,
            equation="6",
            note="the same b appears in ISINB (SOLAR.for 72-73), the shape's daily integral",
        ),
    )
    cos2_mean: float = coef(
        0.5,
        "-",
        "mean of cos^2 of the hour angle over a day in the closed-form integral ISINB",
        _dssat(
            f"{SOLAR_FOR}:72",
            "SOLAR",
            "ISINB = 3600.0 * (DAYL*(SSIN+0.4*(SSIN**2+0.5*CCOS**2)) +",
            note="exact integral of the shape; " + _CONST,
        ),
        calibrate=False,
    )
    cross_term: float = coef(
        1.5,
        "-",
        "factor of the sin x cos cross term of the integral ISINB",
        _dssat(
            f"{SOLAR_FOR}:73",
            "SOLAR",
            "&  24.0/PI*CCOS*(1.0+1.5*0.4*SSIN)*SQRT(1.0-SOC**2))",
            note="exact integral of the shape; " + _CONST,
        ),
        calibrate=False,
    )


class ShawCoefficients(Coefficients):
    """SHAW ``CLOUDY`` and ``SOLAR_SHAW``: direct/diffuse on the slope and the cloud fraction (float64)."""

    pi: float = coef(
        3.14159,
        "-",
        "value of pi of the SHAW solar routines",
        _rz(SHAW_CLOUDY, 23, "CLOUDY", note=_CONST),
        calibrate=False,
    )
    solar_constant: float = coef(
        1360.0,
        "W m-2",
        "solar constant",
        _rz(SHAW_SOLAR, 12, "SOLAR_SHAW", note="also SHAW_Cloudy.for 13"),
    )
    max_transmissivity: float = coef(
        0.76,
        "-",
        "maximum clear-sky transmissivity of the atmosphere (no diffuse radiation at or above it)",
        _rz(
            SHAW_SOLAR, 12, "SOLAR_SHAW", note="Bristow, Campbell and Saxton (1985); also SHAW_Cloudy.for 13"
        ),
    )
    noon: float = coef(
        12.0,
        "h",
        "solar noon (HRNOON)",
        _rz(SHAW_SOLAR, 12, "SOLAR_SHAW", note=_CONST),
        calibrate=False,
    )
    rad_per_hour: float = coef(
        0.261799,
        "rad h-1",
        "hour angle per hour (pi / 12)",
        _rz(SHAW_SOLAR, 21, "SOLAR_SHAW", note="also lines 22 and 51; " + _CONST),
        calibrate=False,
    )
    declination_amplitude: float = coef(
        0.4102,
        "rad",
        "amplitude of the solar declination",
        _rz(SHAW_CLOUDY, 23, "CLOUDY"),
    )
    equinox_day: float = coef(
        80.0,
        "d",
        "day of year of the March equinox in the declination sine",
        _rz(SHAW_CLOUDY, 23, "CLOUDY"),
    )
    full_turn: float = coef(
        2.0,
        "-",
        "multiple of pi of one declination cycle",
        _rz(SHAW_CLOUDY, 23, "CLOUDY", note=_CONST),
        calibrate=False,
    )
    year_length: float = coef(
        365.0,
        "d",
        "days of the declination cycle",
        _rz(SHAW_CLOUDY, 23, "CLOUDY", note=_CONST),
        calibrate=False,
    )
    day_hours: float = coef(
        24.0,
        "h",
        "hours per day of the daily extraterrestrial radiation SUNMAX of CLOUDY",
        _rz(SHAW_CLOUDY, 36, "CLOUDY", note=_CONST),
        calibrate=False,
    )
    diffuse_exponent: float = coef(
        0.6,
        "-",
        "factor of the exponent of the diffuse transmissivity tau_d = tau (1 - exp(a (1 - B/tau) / (B - c)))",
        _rz(SHAW_SOLAR, 103, "SOLAR_SHAW", note="Bristow, Campbell and Saxton (1985)"),
    )
    diffuse_offset: float = coef(
        0.4,
        "-",
        "transmissivity offset c of the diffuse transmissivity",
        _rz(SHAW_SOLAR, 104, "SOLAR_SHAW", note="Bristow, Campbell and Saxton (1985)"),
    )
    direct_cap: float = coef(
        5.0,
        "-",
        "cap of the direct radiation on the slope as a multiple of the direct horizontal radiation",
        _rz(SHAW_SOLAR, 119, "SOLAR_SHAW", note="about a sun 10 degrees high on a perpendicular slope"),
    )
    cloud_intercept: float = coef(
        1.333,
        "-",
        "cloud fraction at zero daily transmissivity, clouds = a - b tau",
        _rz(
            SHAW_CLOUDY,
            40,
            "CLOUDY",
            paper="Flerchinger and Yu (2007), as cited in SHAW_Cloudy.for 39",
            note="a single-precision literal in a double expression: used as float64(float32(1.333))",
        ),
    )
    cloud_slope: float = coef(
        1.666,
        "-",
        "decrease of the cloud fraction per unit daily transmissivity",
        _rz(
            SHAW_CLOUDY,
            40,
            "CLOUDY",
            paper="Flerchinger and Yu (2007), as cited in SHAW_Cloudy.for 39",
            note="a single-precision literal in a double expression: used as float64(float32(1.666))",
        ),
    )


class InpdayCoefficients(Coefficients):
    """``INPDAY`` bounds of the daily radiation and the degrees-per-radian of the RZWQM2 main program."""

    srad_min: float = coef(
        0.0,
        "MJ m-2 d-1",
        "lower bound of the daily solar radiation read from the .MET file (TRN)",
        _rz(RZMAIN, 3522, "INPDAY", paper=RZWQM_BOOK, note="applied at line 3579"),
    )
    srad_max: float = coef(
        45.0,
        "MJ m-2 d-1",
        "upper bound of the daily solar radiation read from the .MET file (TRX)",
        _rz(RZMAIN, 3522, "INPDAY", paper=RZWQM_BOOK, note="applied at line 3580"),
    )
    pi: float = coef(
        3.141592654,
        "-",
        "value of pi of the degrees-per-radian factor 180 / pi of the latitude passed to DSSAT",
        _rz(RZMAIN, 244, "RZWQM2", paper=RZWQM_BOOK, note=_CONST),
        calibrate=False,
    )
    deg_per_half_turn: float = coef(
        180.0,
        "deg",
        "degrees in pi radians (the degrees-per-radian factor 180 / pi)",
        _rz(RZMAIN, 244, "RZWQM2", paper=RZWQM_BOOK, note=_CONST),
        calibrate=False,
    )


class RadiationCoefficients(Coefficients):
    """All coefficients of the RZWQM2 4.6 daily radiation reconstruction."""

    hmet: HmetCoefficients = HmetCoefficients()
    shaw: ShawCoefficients = ShawCoefficients()
    inpday: InpdayCoefficients = InpdayCoefficients()


#: the default set (the reference values)
RZWQM_RADIATION = RadiationCoefficients()


def coefficient_table(coefficients: RadiationCoefficients = RZWQM_RADIATION) -> list[dict[str, Any]]:
    """One row per coefficient (path, value, unit, description, provenance)."""
    return _coefficient_table(coefficients)


# ================================================================================================
# results
# ================================================================================================
@dataclass(frozen=True)
class DailyRadiation:
    """The radiation RZWQM2 4.6 rebuilds for each day; leading shape ``(...,)`` of the inputs.

    ``srad`` (RTS) and ``srad_horizontal`` (RTH) are the two daily fields of port P8.
    """

    srad: np.ndarray  #: RTS [MJ m-2 d-1]: sum of the hourly radiation on the slope
    srad_horizontal: np.ndarray  #: RTH [MJ m-2 d-1]: bounded, modified .MET value
    hourly_horizontal: np.ndarray  #: HRTH [W m-2], (..., 24): hours 1..24, negatives clipped
    hourly_slope: np.ndarray  #: HRTS [W m-2], (..., 24): direct + diffuse on the slope
    clouds: np.ndarray  #: CLOUDS [-]: SHAW cloud fraction (PENFLUX long-wave only)


# ================================================================================================
# INPDAY: RTH
# ================================================================================================
def horizontal_radiation(
    srad_met: Any,
    month: Any = None,
    metmod_srad_pct: Any = None,
    *,
    coefficients: InpdayCoefficients = RZWQM_RADIATION.inpday,
) -> np.ndarray:
    """RTH [MJ m-2 d-1]: the ``.MET`` daily radiation as ``INPDAY`` prepares it.

    The file value times the month's radiation modifier (a percentage, converted to a
    fraction; ``Rzmain.for`` 3556), then bounded to
    ``[srad_min, srad_max]`` (3579-3580). ``metmod_srad_pct`` is the 12 monthly radiation
    modifiers of ``IPNAMES.DAT`` in percent (row 4 of the modifier block, read at ``Rzmain.for``
    9187; default 100 for every month, the value of all 15 reference scenarios) and ``month``
    (1..12, the shape of ``srad_met``) selects them. The multiplication is applied also for 100 %,
    as in the reference (``x * 100 * 0.01`` need not be ``x`` in floating point).
    """
    srad = np.asarray(srad_met, dtype=np.float64)
    if metmod_srad_pct is None:
        pct = np.full(srad.shape, _IDENTITY_PERCENT)
    else:
        table = np.asarray(metmod_srad_pct, dtype=np.float64)
        if table.shape != (N_MONTHS,):
            raise ValueError(f"metmod_srad_pct must hold {N_MONTHS} monthly values, got shape {table.shape}")
        if month is None:
            raise ValueError("month is needed to apply the monthly modifiers")
        m = np.asarray(month).astype(int)
        if np.any((m < 1) | (m > N_MONTHS)):
            raise ValueError("month must be 1..12")
        pct = table[m - 1]
    rth = srad * pct * _FRACTION_OF_PERCENT
    return np.minimum(np.maximum(rth, coefficients.srad_min), coefficients.srad_max)


# ================================================================================================
# DSSAT DAYLEN / SOLAR / HANG / HRAD: HRTH (float32)
# ================================================================================================
def _latitude_deg32(latitude_rad: Any, c: InpdayCoefficients) -> np.ndarray:
    """Latitude [deg] as DSSAT receives it: converted in double, rounded to float32 (``Rzmain.for`` 1316)."""
    r2d = c.deg_per_half_turn / c.pi
    return (np.asarray(latitude_rad, dtype=np.float64) * r2d).astype(_F32)


def hourly_horizontal_radiation(
    srad_horizontal: Any,
    doy: Any,
    latitude_rad: Any,
    *,
    coefficients: RadiationCoefficients = RZWQM_RADIATION,
) -> np.ndarray:
    """HRTH [W m-2], ``(..., 24)`` float32: hours 1..24 of the DSSAT ``HMET`` disaggregation of RTH.

    ``DAYLEN`` (declination, day length, sunrise and sunset), ``SOLAR`` (``ISINB``, the daily
    integral of Spitters' eq. 6), ``HANG`` (solar elevation at each whole hour ``HS = 1..24``,
    raised to 1 degree once above 1e-4 degree) and ``HRAD`` (``sinB (1 + 0.4 sinB) SRAD / ISINB``
    between sunrise and sunset), in float32 with the reference's operation order. Negative hours
    are *not* clipped here (``CLOUDY`` does that, :func:`rzwqm_radiation`).
    """
    c = coefficients.hmet
    f = _F32
    pi = f(c.pi)
    rad = f(pi / f(c.deg_per_half_turn))
    srad = np.asarray(srad_horizontal, dtype=np.float64).astype(f)[..., None]
    d = np.asarray(doy).astype(f)[..., None]
    lat = _latitude_deg32(latitude_rad, coefficients.inpday)
    lat = np.broadcast_to(lat, np.broadcast_shapes(lat.shape, srad.shape[:-1]))[..., None]
    # DAYLEN (SOLAR.for 126-138; DSSAT 4.0 has no day-length clamp)
    dec = f(c.declination_amplitude) * _rn(
        np.cos, f(c.full_turn) * pi * (d + f(c.declination_day_offset)) / f(c.year_length)
    )
    soc = np.clip(_rn(np.tan, rad * dec) * _rn(np.tan, rad * lat), f(-1.0), f(1.0))
    dayl = f(c.noon) + f(c.day_hours) * _rn(np.arcsin, soc) / pi
    snup = f(c.noon) - dayl / f(c.halves)
    sndn = f(c.noon) + dayl / f(c.halves)
    # SOLAR: ISINB (SOLAR.for 38-41, 72-73)
    ssin = _rn(np.sin, rad * dec) * _rn(np.sin, rad * lat)
    ccos = _rn(np.cos, rad * dec) * _rn(np.cos, rad * lat)
    soc2 = np.clip(ssin / ccos, f(-1.0), f(1.0))
    b = f(c.spitters_b)
    isinb = f(SECONDS_PER_HOUR) * (
        dayl * (ssin + b * (ssin**2 + f(c.cos2_mean) * ccos**2))
        + f(c.day_hours) / pi * ccos * (f(1.0) + (f(c.cross_term) * b) * ssin) * np.sqrt(f(1.0) - soc2**2)
    )
    # HANG (HMET.for 211-226) at HS = 1..24 h
    hs = np.arange(1, N_HOURS + 1, dtype=f)
    hangl = (hs - f(c.noon)) * pi / f(c.hours_per_half_turn)
    beta = _rn(np.arcsin, np.clip(ssin + ccos * _rn(np.cos, hangl), f(-1.0), f(1.0))) / rad
    beta = np.where(beta > f(c.elevation_trigger), np.maximum(beta, f(c.elevation_min)), beta)
    # HRAD (HMET.for 352-362)
    sinb = _rn(np.sin, rad * beta)
    day = (hs > snup) & (hs < sndn)
    with np.errstate(divide="ignore", invalid="ignore"):
        radhr = sinb * (f(1.0) + b * sinb) * srad * f(_J_IN_MJ) / isinb
    return np.where(day, radhr, f(0.0)).astype(f)


# ================================================================================================
# SHAW CLOUDY / SOLAR_SHAW: HRTS and CLOUDS (float64)
# ================================================================================================
def _shaw_sun(doy: np.ndarray, lat: np.ndarray, c: ShawCoefficients) -> tuple[np.ndarray, np.ndarray]:
    """``CLOUDY``'s declination and half-day length [rad] (SHAW_Cloudy.for 23-35)."""
    declin = c.declination_amplitude * np.sin(c.full_turn * c.pi * (doy - c.equinox_day) / c.year_length)
    coshaf = -(np.tan(lat) * np.tan(declin))
    polar = np.where(coshaf >= 1.0, 0.0, c.pi)
    hafday = np.where(np.abs(coshaf) >= 1.0, polar, np.arccos(np.clip(coshaf, -1.0, 1.0)))
    return declin, hafday


def shaw_slope_partition(
    sunhor: Any,
    doy: Any,
    latitude_rad: Any,
    slope_rad: Any = 0.0,
    aspect_rad: Any = 0.0,
    *,
    coefficients: ShawCoefficients = RZWQM_RADIATION.shaw,
) -> tuple[np.ndarray, np.ndarray]:
    """``(direct, diffuse)`` [W m-2], ``(..., 24)``: SHAW's partition of the hourly horizontal radiation.

    ``SOLAR_SHAW`` (SHAW_Solar.for 2-123) with one-hour steps: the sun position is
    the radiation-weighted vector mean of the two ends of the hour; the diffuse part follows the
    transmissivity (Bristow et al. 1985, capped at ``max_transmissivity``); the direct part is
    projected on the slope and capped at ``direct_cap`` times the direct horizontal part. On a
    flat surface ``direct + diffuse`` is the horizontal value (to rounding). ``sunhor`` must be
    already clipped at zero (as ``CLOUDY`` leaves it). ``aspect_rad`` is clockwise from north.
    """
    c = coefficients
    sh = np.asarray(sunhor, dtype=np.float64)
    lead = sh.shape[:-1]
    d = np.broadcast_to(np.asarray(doy).astype(np.float64), lead)[..., None]
    lat = np.broadcast_to(np.asarray(latitude_rad, dtype=np.float64), lead)[..., None]
    slope = np.broadcast_to(np.asarray(slope_rad, dtype=np.float64), lead)[..., None]
    aspect = np.broadcast_to(np.asarray(aspect_rad, dtype=np.float64), lead)[..., None]
    declin, hafday = _shaw_sun(d, lat, c)
    sunris = c.noon - hafday / c.rad_per_hour
    sunset = c.noon + hafday / c.rad_per_hour
    tropics = np.abs(declin) >= np.abs(lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        west_arg = np.tan(declin) / np.tan(lat)
    hrwest = np.where(tropics, c.pi, np.arccos(np.clip(np.where(tropics, 0.0, west_arg), -1.0, 1.0)))
    north = (lat - declin) > 0.0
    hour = np.arange(1, N_HOURS + 1, dtype=np.float64)
    zero = np.zeros(np.broadcast_shapes(sh.shape, d.shape))
    sinazm, cosazm, sumalt, cosalt, sunmax = zero, zero, zero, zero, zero
    for back in (1.0, 0.0):  # the start and the end of the hour (SHAW_Solar.for 47-77)
        thour = hour - back
        hrangl = c.rad_per_hour * (thour - c.noon)
        up = (thour > sunris) & (thour < sunset)
        sinalt = np.sin(lat) * np.sin(declin) + np.cos(lat) * np.cos(declin) * np.cos(hrangl)
        alt = np.arcsin(np.clip(sinalt, -1.0, 1.0))
        cos_alt = np.cos(alt)
        safe_cos = np.where(up, cos_alt, 1.0)
        azm = np.arcsin(np.clip(-(np.cos(declin) * np.sin(hrangl) / safe_cos), -1.0, 1.0))
        flip = np.where(north, np.abs(hrangl) < hrwest, np.abs(hrangl) >= hrwest)
        azm = np.where(flip, c.pi - azm, azm)
        sun = c.solar_constant * sinalt
        sumalt = sumalt + np.where(up, sun * sinalt, 0.0)
        cosalt = cosalt + np.where(up, sun * cos_alt, 0.0)
        sinazm = sinazm + np.where(up, sun * np.sin(azm), 0.0)
        cosazm = cosazm + np.where(up, sun * np.cos(azm), 0.0)
        sunmax = sunmax + np.where(up, sun, 0.0)
    risen = sunmax != 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        altitu = np.where(risen, np.arctan(sumalt / np.where(risen, cosalt, 1.0)), 0.0)
    azmuth = np.arctan2(sinazm, cosazm)
    sunmax = sunmax / _ENDPOINTS
    sunslp = np.where(
        risen,
        np.arcsin(
            np.clip(
                np.sin(altitu) * np.cos(slope) + np.cos(altitu) * np.sin(slope) * np.cos(azmuth - aspect),
                -1.0,
                1.0,
            )
        ),
        0.0,
    )
    lit = altitu > 0.0
    safe_max = np.where(lit, sunmax, 1.0)
    ttotal = np.minimum(np.where(lit, sh / safe_max, c.max_transmissivity), c.max_transmissivity)
    ttotal_safe = np.where(ttotal > 0.0, ttotal, c.max_transmissivity)
    tdiffu = ttotal * (
        1.0
        - np.exp(
            c.diffuse_exponent
            * (1.0 - c.max_transmissivity / ttotal_safe)
            / (c.max_transmissivity - c.diffuse_offset)
        )
    )
    diffus_lit = tdiffu * sunmax
    dirhor = sh - diffus_lit
    safe_sin_alt = np.where(lit, np.sin(altitu), 1.0)
    direct_lit = np.where(
        sunslp > 0.0, np.minimum(dirhor * np.sin(sunslp) / safe_sin_alt, c.direct_cap * dirhor), 0.0
    )
    positive = sh > 0.0
    direct = np.where(positive & lit, direct_lit, 0.0)
    diffus = np.where(positive, np.where(lit, diffus_lit, sh), 0.0)
    return direct, diffus


def _cloudy(sunhor: np.ndarray, doy: np.ndarray, lat: np.ndarray, c: ShawCoefficients) -> np.ndarray:
    """``CLOUDS`` of SHAW ``CLOUDY`` (SHAW_Cloudy.for 18-43) from the clipped hourly radiation."""
    declin, hafday = _shaw_sun(doy.astype(np.float64), lat, c)
    totsun = np.zeros(sunhor.shape[:-1])
    for h in range(N_HOURS):  # daily total accumulated in hour order (SHAW_Cloudy.for 21)
        totsun = totsun + sunhor[..., h]
    sunmax = (
        c.day_hours
        * c.solar_constant
        * (hafday * np.sin(lat) * np.sin(declin) + np.cos(lat) * np.cos(declin) * np.sin(hafday))
        / c.pi
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ttotal = np.where(sunmax > 0.0, totsun / np.where(sunmax > 0.0, sunmax, 1.0), 0.0)
    a = float(_F32(c.cloud_intercept))
    b = float(_F32(c.cloud_slope))
    return np.clip(a - b * ttotal, 0.0, 1.0)


# ================================================================================================
# the whole daily chain
# ================================================================================================
def rzwqm_radiation(
    srad_met: Any,
    doy: Any,
    latitude_rad: Any,
    slope_rad: Any = 0.0,
    aspect_rad: Any = 0.0,
    *,
    month: Any = None,
    metmod_srad_pct: Any = None,
    allow_zero: bool = False,
    coefficients: RadiationCoefficients = RZWQM_RADIATION,
) -> DailyRadiation:
    """RTH, HRTH, HRTS, RTS and CLOUDS of RZWQM2 4.6 from the daily ``.MET`` radiation.

    ``srad_met`` [MJ m-2 d-1] and ``doy`` (1..366) have any shape ``(...)`` (days, or scenarios x
    days); ``latitude_rad``, ``slope_rad`` and ``aspect_rad`` (the ``rzwqm.dat`` physiography,
    radians, aspect clockwise from north) broadcast against it, so many sites and years go in one
    call. ``month`` and ``metmod_srad_pct`` are for :func:`horizontal_radiation`.

    Raises ``ValueError`` on a day with RTS = 0 (the reference then substitutes a clear-sky
    estimate inside the PET routine, which is not reproduced) unless ``allow_zero``.
    """
    srad = np.asarray(srad_met, dtype=np.float64)
    days = np.broadcast_to(np.asarray(doy).astype(int), srad.shape)
    lat = np.broadcast_to(np.asarray(latitude_rad, dtype=np.float64), srad.shape)
    rth = horizontal_radiation(srad, month, metmod_srad_pct, coefficients=coefficients.inpday)
    hrth32 = hourly_horizontal_radiation(rth, days, lat, coefficients=coefficients)
    hrth = np.maximum(hrth32.astype(np.float64), 0.0)  # CLOUDY clips negative hours in place
    clouds = _cloudy(hrth, days, lat, coefficients.shaw)
    direct, diffus = shaw_slope_partition(
        hrth, days, lat, slope_rad, aspect_rad, coefficients=coefficients.shaw
    )
    hrts = direct + diffus
    rts = np.zeros(srad.shape)
    for h in range(N_HOURS):  # hourly W m-2 to MJ m-2, summed in hour order (Rzmain.for 1389)
        rts = rts + hrts[..., h] * SECONDS_PER_HOUR / _J_IN_MJ
    if not allow_zero and np.any(rts == 0.0):
        n = int(np.count_nonzero(rts == 0.0))
        raise ValueError(
            f"{n} day(s) with RTS = 0: RZWQM2 replaces them by a clear-sky estimate in POTEVPHR, "
            "which this reconstruction does not reproduce (pass allow_zero=True to keep the zeros)"
        )
    return DailyRadiation(
        srad=rts, srad_horizontal=rth, hourly_horizontal=hrth, hourly_slope=hrts, clouds=clouds
    )


def radiation_from_met(
    met: pd.DataFrame,
    latitude_rad: float,
    slope_rad: float = 0.0,
    aspect_rad: float = 0.0,
    *,
    metmod_srad_pct: Any = None,
    column: str = "srad_mj",
    coefficients: RadiationCoefficients = RZWQM_RADIATION,
) -> pd.DataFrame:
    """Daily ``srad`` (RTS), ``srad_horizontal`` (RTH) and ``clouds`` for a date-indexed ``.MET`` frame.

    ``met`` is :func:`agrijax.io.rzwqm.read_met` output (the raw file values; the bounds and
    modifiers are applied here); the index gives the day of year and the month.
    """
    dates = pd.Series(pd.DatetimeIndex(met.index)).dt
    out = rzwqm_radiation(
        met[column].to_numpy(dtype=np.float64),
        dates.dayofyear.to_numpy(),
        latitude_rad,
        slope_rad,
        aspect_rad,
        month=dates.month.to_numpy(),
        metmod_srad_pct=metmod_srad_pct,
        coefficients=coefficients,
    )
    return pd.DataFrame(
        {"srad": out.srad, "srad_horizontal": out.srad_horizontal, "clouds": out.clouds}, index=met.index
    )
