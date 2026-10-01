"""Host-side: the daily weather record DSSAT-CSM v4.8.6.0 derives from a ``.WTH`` day (NumPy, float32).

What ``WEATHR`` (``Weather/weathr.for``) hands to ``SPAM`` and ``WATBAL`` besides the file's own
values, reproduced in the reference's single precision:

* **day length, sunrise, sunset** -- ``DAYLEN`` (``Weather/SOLAR.for`` 113-141): the declination
  cosine (Spitters et al. 1986, eq. 16), the sunrise equation (eq. 17), the day length clamped to
  [0, 24] h (4.8.6, line 136);
* **the hourly mean air temperature TAVG** -- ``HMET`` (``Weather/HMET.for`` 39-130) averages the
  24 whole-hour temperatures of ``HTEMP`` (261-304): Parton and Logan's (1981) sine curve from the
  minimum at sunrise + ``C`` to the maximum at noon + ``A`` + ``C``, and an exponential decay with
  the rate ``B`` from the sunset temperature through the night. ``TAVG`` is not
  ``(TMAX + TMIN) / 2``;
* **the wind at 2 m** -- ``weathr.for`` 412-420: the file's wind run scaled by ``(2 / WNDHT)^0.2``,
  or 86.4 km d-1 (1 m s-1) when the file has none.

The arithmetic is float32 in the reference's operation order; the transcendentals are evaluated in
float64 and rounded once (the correctly rounded single-precision result, as the glibc ``sinf`` /
``expf`` of the static reference binary). Written from the published equations (Parton, W.J. and
Logan, J.A. 1981, Agric. Meteorol. 23, 205-216; Spitters, Toussaint and Goudriaan 1986) and the
DSSAT-CSM v4.8.6.0 source (BSD-3), cited by file, line and routine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agrijax.core.coefficients import Coefficients, Provenance, coef

__all__ = [
    "DSSAT486_WEATHER",
    "DailyWeatherCoefficients",
    "SunTimes",
    "daylength486",
    "hourly_mean_temperature",
    "wind_at_2m",
]

REF = "dssat-4.8.6.0"
SOLAR_FOR = "Weather/SOLAR.for"
HMET_FOR = "Weather/HMET.for"
WEATHR_FOR = "Weather/weathr.for"
PARTON = "Parton and Logan (1981)"
SPITTERS = "Spitters, Toussaint and Goudriaan (1986)"
_CONST = "constant of the reference arithmetic, not calibrated"
_F = np.float32


def _p(file_line: str, routine: str, **kw: Any) -> Provenance:
    return Provenance.at(REF, file_line, routine=routine, **kw)


class DailyWeatherCoefficients(Coefficients):
    """``DAYLEN``, ``HTEMP`` / ``HMET`` and the 2 m wind of ``WEATHR`` (float32)."""

    pi: float = coef(
        3.14159,
        "-",
        "value of pi of the DSSAT weather routines",
        _p(f"{SOLAR_FOR}:121", "DAYLEN", note=_CONST),
        calibrate=False,
    )
    deg_per_half_turn: float = coef(
        180.0,
        "deg",
        "degrees in pi radians (RAD = PI/180)",
        _p(f"{SOLAR_FOR}:121", "DAYLEN", note=_CONST),
        calibrate=False,
    )
    declination_amplitude: float = coef(
        -23.45,
        "deg",
        "amplitude of the solar declination (negative: minimum at the December solstice)",
        _p(f"{SOLAR_FOR}:126", "DAYLEN", paper=SPITTERS, equation="16"),
    )
    declination_day_offset: float = coef(
        10.0,
        "d",
        "days from the December solstice to 1 January in the declination cosine",
        _p(f"{SOLAR_FOR}:126", "DAYLEN", paper=SPITTERS, equation="16"),
    )
    full_turn: float = coef(
        2.0,
        "-",
        "multiple of pi of one declination cycle",
        _p(f"{SOLAR_FOR}:126", "DAYLEN", note=_CONST),
        calibrate=False,
    )
    year_length: float = coef(
        365.0,
        "d",
        "days of the declination cycle",
        _p(f"{SOLAR_FOR}:126", "DAYLEN", note=_CONST),
        calibrate=False,
    )
    noon: float = coef(
        12.0,
        "h",
        "solar noon, and half the day length of an equinox day",
        _p(f"{SOLAR_FOR}:133", "DAYLEN", paper=SPITTERS, equation="17"),
        calibrate=False,
    )
    day_hours: float = coef(
        24.0,
        "h",
        "hours of one day (DAYL = 12 + 24 asin(SOC) / pi; the day-length clamp; the night decay)",
        _p(f"{SOLAR_FOR}:133", "DAYLEN", note="also the upper day-length bound, line 136"),
        calibrate=False,
    )
    halves: float = coef(
        2.0,
        "-",
        "sunrise and sunset are half the day length before and after noon",
        _p(f"{SOLAR_FOR}:137", "DAYLEN", note=_CONST),
        calibrate=False,
    )
    tmax_lag: float = coef(
        2.0,
        "h",
        "time of the daily maximum temperature after solar noon (A)",
        _p(f"{HMET_FOR}:269", "HTEMP", paper=PARTON),
    )
    night_decay: float = coef(
        2.2,
        "-",
        "nocturnal exponential decay coefficient of the air temperature (B)",
        _p(f"{HMET_FOR}:269", "HTEMP", paper=PARTON),
    )
    tmin_lag: float = coef(
        1.0,
        "h",
        "time of the daily minimum temperature after sunrise (C)",
        _p(f"{HMET_FOR}:269", "HTEMP", paper=PARTON),
    )
    quarter_turn: float = coef(
        0.5,
        "-",
        "multiple of pi of the rising half of the daytime sine (the maximum at pi/2)",
        _p(f"{HMET_FOR}:278", "HTEMP", note=_CONST),
        calibrate=False,
    )
    hours: int = coef(
        24,
        "-",
        "hourly steps per day (TS of ModuleDefs)",
        _p("Utilities/ModuleDefs.for:50", "ModuleDefs", note=_CONST),
        static=True,
    )
    wind_height: float = coef(
        2.0,
        "m",
        "height the SPAM wind run refers to",
        _p(f"{WEATHR_FOR}:416", "WEATHR"),
        calibrate=False,
    )
    wind_exponent: float = coef(
        0.2,
        "-",
        "exponent of the power-law wind profile from the measurement height to 2 m",
        _p(f"{WEATHR_FOR}:416", "WEATHR", note="changed from 2.0 on 2013-08-28 (chp)"),
    )
    wind_default: float = coef(
        86.4,
        "km d-1",
        "wind run used when the weather file has none (1 m s-1)",
        _p(f"{WEATHR_FOR}:419", "WEATHR"),
    )


DSSAT486_WEATHER = DailyWeatherCoefficients()


def _rn(fn: Any, x: Any) -> np.ndarray:
    """A float32 transcendental rounded to nearest: evaluated in float64, rounded once."""
    return np.asarray(fn(np.asarray(x, dtype=np.float64))).astype(_F)


@dataclass(frozen=True)
class SunTimes:
    """``DAYLEN`` outputs (float32): day length, declination, sunrise and sunset."""

    dayl: np.ndarray
    dec: np.ndarray
    snup: np.ndarray
    sndn: np.ndarray


def daylength486(doy: Any, xlat: Any, c: DailyWeatherCoefficients = DSSAT486_WEATHER) -> SunTimes:
    """Host-side: DSSAT-CSM v4.8.6.0 ``DAYLEN`` (``SOLAR.for`` 113-141) in float32 for day of year
    ``doy`` and latitude ``xlat`` [deg] (the weather file's ``LAT`` as ``REAL``)."""
    f = _F
    pi = f(c.pi)
    rad = f(pi / f(c.deg_per_half_turn))
    d = np.asarray(doy).astype(f)
    lat = np.asarray(xlat, dtype=np.float64).astype(f)
    dec = f(c.declination_amplitude) * _rn(
        np.cos, f(c.full_turn) * pi * (d + f(c.declination_day_offset)) / f(c.year_length)
    )
    soc = np.clip(_rn(np.tan, rad * dec) * _rn(np.tan, rad * lat), f(-1.0), f(1.0))
    dayl = f(c.noon) + f(c.day_hours) * _rn(np.arcsin, soc) / pi
    dayl = np.clip(dayl, f(0.0), f(c.day_hours)).astype(f)
    snup = (f(c.noon) - dayl / f(c.halves)).astype(f)
    sndn = (f(c.noon) + dayl / f(c.halves)).astype(f)
    return SunTimes(dayl=dayl.astype(f), dec=np.asarray(dec, dtype=f), snup=snup, sndn=sndn)


def hourly_mean_temperature(
    tmax: Any, tmin: Any, sun: SunTimes, c: DailyWeatherCoefficients = DSSAT486_WEATHER
) -> np.ndarray:
    """Host-side: ``TAVG`` of ``HMET`` (``HMET.for`` 68-116): the mean of the 24 whole-hour ``HTEMP``
    temperatures (261-304) [degC, float32] of each day (``tmax``, ``tmin`` [degC], ``sun`` from
    :func:`daylength486`), summed hour by hour as the reference does."""
    f = _F
    tx = np.asarray(tmax, dtype=np.float64).astype(f)
    tn = np.asarray(tmin, dtype=np.float64).astype(f)
    dayl, snup, sndn = sun.dayl, sun.snup, sun.sndn
    a, b, cc = f(c.tmax_lag), f(c.night_decay), f(c.tmin_lag)
    pi = f(c.pi)
    half_pi = f(f(c.quarter_turn) * pi)
    ts = int(c.hours)
    tincr = f(f(c.day_hours) / f(ts))
    # HTEMP, day-level terms (HMET.for 276-281)
    tmn = (snup + cc).astype(f)
    tmx = (tmn + dayl / f(c.halves) + a).astype(f)
    t = (half_pi * (sndn - tmn) / (tmx - tmn)).astype(f)
    tsndn = (tn + (tx - tn) * _rn(np.sin, t)).astype(f)
    eb = _rn(np.exp, -b)
    tmini = ((tn - tsndn * eb) / (f(1.0) - eb)).astype(f)
    hdecay = (f(c.day_hours) + cc - dayl).astype(f)
    tavg = np.zeros_like(tx)
    for h in range(1, ts + 1):
        hs = f(f(h) * tincr)
        day = (hs >= snup + cc) & (hs <= sndn)
        td = (tn + (tx - tn) * _rn(np.sin, (half_pi * (hs - tmn) / (tmx - tmn)).astype(f))).astype(f)
        tnight = np.where(hs < snup + cc, (f(c.day_hours) + hs - sndn).astype(f), (hs - sndn).astype(f))
        arg = (-b * tnight / hdecay).astype(f)
        tn_h = (tmini + (tsndn - tmini) * _rn(np.exp, arg)).astype(f)
        tavg = (tavg + np.where(day, td, tn_h)).astype(f)
    return (tavg / f(ts)).astype(f)


def wind_at_2m(wind: Any, windht: Any, c: DailyWeatherCoefficients = DSSAT486_WEATHER) -> np.ndarray:
    """Host-side: the wind run SPAM reads [km d-1, float32] (``weathr.for`` 412-420): the file's
    ``wind`` (NaN or <= 0: none) at the height ``windht`` [m] scaled to 2 m, else the default."""
    f = _F
    w = np.asarray(wind, dtype=np.float64)
    has = np.isfinite(w) & (w > 0)
    w32 = np.where(has, w, 1.0).astype(f)
    ratio = f(f(c.wind_height) / f(windht))
    scaled = (w32 * _rn(lambda x: np.power(x, np.float64(f(c.wind_exponent))), ratio)).astype(f)
    return np.where(has, scaled, f(c.wind_default)).astype(f)
