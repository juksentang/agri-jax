"""Host-side: the daily weather record ``dscsm048`` builds from the public ``.WTH`` and CO2 files (NumPy).

For the simulated days of a run this returns what ``WEATHR`` (DSSAT-CSM v4.8.6.0,
``Weather/weathr.for`` 328-459, the RATE block) puts in ``WEATHER`` for ``SPAM`` and ``WATBAL`` (``REAL*4``):

* ``SRAD``, ``TMAX``, ``TMIN``, ``RAIN`` as ``IPWTH`` reads them (``Weather/IPWTH_alt.for``: the
  ``PARSE_HEADERS`` span of each ``@DATE`` column, list-directed; ``SRAD < 0.1`` raised to 0.1,
  lines 1337-1345; a day with ``TMAX < TMIN``, negative rain or missing radiation stops DSSAT,
  1283-1292: raised here);
* ``TAVG`` of ``HMET`` and the 2 m wind run (:mod:`agrijax.forcing.dssat_weather`), with the file's
  ``LAT`` and ``WNDHT`` (``WNDHT <= 0`` or missing -> 2 m, line 450);
* ``CO2`` of ``CO2VAL`` (``Weather/CO2VAL.for``): option ``M`` the monthly series of the engine's
  ``StandardData/CO2048.WDA`` (the value of the last record on or before the simulation start,
  then the value of a record on its own day; records below 1 ppm or past the year's last day are
  skipped: SEASINIT 61-212, RATE 215-246), ``D`` the file's ``@CO2BAS``, ``W`` the weather file's
  site ``CCO2`` (or ``CO2``) when above 1 ppm (67-70), else the daily ``DCO2`` column (238-242).

A season crossing a year reads the next year's file ``<INSI><YY+1>01.WTH`` (the ``IPWTH`` naming).
Not supported (raised): generated weather (``WTHER`` other than ``M``) and FileX environment
modifications (``WTHMOD``, treatment ``ME`` level): their daily values are not ported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.forcing.dssat_weather import (
    DSSAT486_WEATHER,
    DailyWeatherCoefficients,
    daylength486,
    hourly_mean_temperature,
    wind_at_2m,
)

from ._fixed import read_lines
from .wth import read_wth

__all__ = [
    "DSSAT_WEATHER_INPUT",
    "CO2Series",
    "DailyWeatherRecord",
    "WeatherInputCoefficients",
    "co2_daily",
    "native_weather",
    "read_co2_series",
    "yrdoy_dates",
]

REF = "dssat-4.8.6.0"
IPWTH = "Weather/IPWTH_alt.for"
CO2VAL = "Weather/CO2VAL.for"
_F = np.float32


def _p(file_line: str, routine: str, note: str = "") -> Provenance:
    return Provenance.at(REF, file_line, routine=routine, note=note)


class WeatherInputCoefficients(Coefficients):
    """Bounds and defaults of ``IPWTH`` and ``CO2VAL``."""

    srad_min: float = coef(
        0.1,
        "MJ m-2 d-1",
        "a daily radiation below this is raised to it",
        _p(f"{IPWTH}:1344", "DailyWeatherCheck"),
        calibrate=False,
    )
    windht_default: float = coef(
        2.0,
        "m",
        "wind measurement height used when WNDHT is missing or not positive",
        _p(f"{IPWTH}:450", "IPWTH"),
        calibrate=False,
    )
    co2_min: float = coef(
        1.0,
        "ppm",
        "a CO2 value at or below this is missing",
        _p(f"{CO2VAL}:164", "CO2VAL"),
        calibrate=False,
    )
    co2_default: float = coef(
        380.0,
        "ppm",
        "CO2 used when the CO2 file gives no base value",
        _p(f"{CO2VAL}:72", "CO2VAL"),
        calibrate=False,
    )


DSSAT_WEATHER_INPUT = WeatherInputCoefficients()


@dataclass(frozen=True)
class CO2Series:
    """The ``@CO2BAS`` value and the monthly records of a DSSAT CO2 file (``CO2048.WDA``)."""

    base: float
    year: np.ndarray
    doy: np.ndarray
    co2: np.ndarray


def read_co2_series(path: str | Path, c: WeatherInputCoefficients = DSSAT_WEATHER_INPUT) -> CO2Series:
    """Host-side: the base value and the valid records of a DSSAT CO2 file, as ``CO2VAL`` keeps them."""
    lines = read_lines(path)
    base = c.co2_default
    years: list[int] = []
    doys: list[int] = []
    vals: list[float] = []
    mode = ""
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("!"):
            continue
        if ln.startswith("@"):
            mode = "base" if "CO2BAS" in ln else ("series" if "YEAR" in ln else "")
            continue
        if ln.startswith("*"):
            mode = ""
            continue
        tok = s.split()
        if mode == "base":
            v = float(tok[0])
            base = v if v > c.co2_min else c.co2_default
            mode = ""
        elif mode == "series" and len(tok) >= 3:
            y, d, v = int(tok[0]), int(tok[1]), float(_F(tok[2]))
            leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
            if v < c.co2_min or d > (366 if leap else 365):
                continue
            years.append(y)
            doys.append(d)
            vals.append(v)
    return CO2Series(
        float(_F(base)),
        np.asarray(years, dtype=np.int64),
        np.asarray(doys, dtype=np.int64),
        np.asarray(vals, _F),
    )


def yrdoy_dates(yrdoy: Sequence[int]) -> list[date]:
    """``YYYYDDD`` codes as dates."""
    return [date(int(v) // 1000, 1, 1) + timedelta(days=int(v) % 1000 - 1) for v in yrdoy]


def co2_daily(
    days: Sequence[int],
    option: str,
    series: CO2Series | None,
    cco2: float | None = None,
    dco2: np.ndarray | None = None,
    c: WeatherInputCoefficients = DSSAT_WEATHER_INPUT,
) -> np.ndarray:
    """Host-side: ``CO2VAL``'s daily CO2 [ppm, float32] on ``days`` (``YYYYDDD``, the first the
    simulation start) for the FileX ``CO2`` option (module docstring)."""
    n = len(days)
    opt = (option or "M").strip().upper() or "M"
    if opt == "W":
        # SEASINIT: the site CCO2 above 1 ppm, else the CO2 file's base; RATE: a daily DCO2 above
        # 1 ppm replaces it and carries over to the next days
        if cco2 is not None and cco2 > c.co2_min:
            base = cco2
        else:
            base = series.base if series is not None else c.co2_default
        out = np.empty(n, dtype=_F)
        prev = _F(base)
        for i in range(n):
            v = None if dco2 is None else float(dco2[i])
            if v is not None and np.isfinite(v) and v > c.co2_min:
                prev = _F(v)
            out[i] = prev
        return out
    if series is None:
        raise ValueError("CO2 option M or D needs the CO2 file of the engine (StandardData/CO2048.WDA)")
    if opt == "D":
        return np.full(n, _F(series.base), dtype=_F)
    if opt != "M":
        raise NotImplementedError(f"CO2 option {option!r} is not ported (M, D, W are)")
    key = series.year * 1000 + series.doy
    first = int(days[0])
    before = np.nonzero(key <= first)[0]
    co2 = _F(series.co2[before[-1]]) if before.size else _F(series.co2[0]) if key.size else _F(series.base)
    exact = {int(k): _F(v) for k, v in zip(key.tolist(), series.co2.tolist(), strict=True)}
    out = np.empty(n, dtype=_F)
    for i, d in enumerate(days):
        co2 = exact.get(int(d), co2)
        out[i] = co2
    return out


@dataclass(frozen=True)
class DailyWeatherRecord:
    """The daily ``WEATHER`` values of a run's days (float32; ``days`` as ``YYYYDDD``)."""

    days: np.ndarray
    srad: np.ndarray
    tmax: np.ndarray
    tmin: np.ndarray
    rain: np.ndarray
    tavg: np.ndarray
    windsp: np.ndarray
    co2: np.ndarray
    dayl: np.ndarray
    xlat: float
    windht: float
    files: tuple[str, ...]


def _next_year_file(p: Path, year: int) -> Path:
    name = p.name
    if len(name) >= 12 and name[4:6].isdigit():
        return p.with_name(f"{name[:4]}{year % 100:02d}{name[6:]}")
    raise FileNotFoundError(f"{p.name}: cannot name the weather file of {year}")


def native_weather(
    wth: str | Path,
    days: Sequence[int],
    *,
    co2_option: str = "M",
    co2_file: str | Path | None = None,
    c: WeatherInputCoefficients = DSSAT_WEATHER_INPUT,
    w: DailyWeatherCoefficients = DSSAT486_WEATHER,
) -> DailyWeatherRecord:
    """Host-side: the weather record of ``days`` (``YYYYDDD``, consecutive, the first the simulation
    start) from the weather file ``wth`` of the first day (the ``WEATHERW`` file of ``DSSAT48.INP``)
    and, for option ``M`` / ``D``, the engine's ``co2_file`` (module docstring)."""
    want = yrdoy_dates(days)
    path = Path(wth)
    frames = []
    site: dict[str, object] = {}
    files: list[str] = []
    have: dict[date, tuple[int, int]] = {}
    need = set(want)
    while True:
        df = read_wth(path, dssat_spans=True, century=want[0].year // 100)
        if not site:
            site = dict(df.attrs["site"])
        frames.append(df)
        files.append(path.name)
        for i, d in enumerate(df["date"].dt.date):
            have.setdefault(d, (len(frames) - 1, i))
        missing = sorted(need - set(have))
        if not missing:
            break
        nxt = _next_year_file(path, missing[0].year)
        if nxt == path or not nxt.is_file():
            raise FileNotFoundError(f"weather for {missing[0]} not in {files} (next file {nxt.name} missing)")
        path = nxt

    def col(name: str) -> np.ndarray:
        out = np.full(len(want), np.nan)
        for j, d in enumerate(want):
            fi, ri = have[d]
            df = frames[fi]
            if name in df.columns:
                out[j] = float(df[name].iloc[ri])
        return out

    srad, tmax, tmin, rain = (col(k) for k in ("srad", "tmax", "tmin", "rain"))
    for name, v in (("SRAD", srad), ("TMAX", tmax), ("TMIN", tmin), ("RAIN", rain)):
        if np.any(~np.isfinite(v)):
            raise ValueError(f"{files}: {name} missing on {int(np.sum(~np.isfinite(v)))} simulated days")
    if np.any(rain < 0) or np.any(srad < 0) or np.any(tmax < tmin):
        raise ValueError(f"{files}: negative rain / radiation or TMAX < TMIN (DSSAT stops, IPWTH 1283-1292)")
    srad32 = np.maximum(srad.astype(_F), _F(c.srad_min)).astype(_F)
    tmax32, tmin32, rain32 = tmax.astype(_F), tmin.astype(_F), rain.astype(_F)
    lat = site.get("LAT")
    xlat = _F(lat) if isinstance(lat, (int, float)) else _F(0.0)
    wh = site.get("WNDHT")
    windht = _F(wh) if isinstance(wh, (int, float)) and wh > 0 else _F(c.windht_default)
    doy = np.asarray([d.timetuple().tm_yday for d in want])
    sun = daylength486(doy, xlat, w)
    tavg = hourly_mean_temperature(tmax32, tmin32, sun, w)
    wind = col("wind")
    windsp = wind_at_2m(wind, windht, w)
    cc = site.get("CCO2", site.get("CO2"))
    cco2 = float(cc) if isinstance(cc, (int, float)) else None
    series = read_co2_series(co2_file, c) if co2_file is not None else None
    dco2 = col("dco2") if any("dco2" in f.columns for f in frames) else col("co2")
    co2 = co2_daily(days, co2_option, series, cco2, dco2, c)
    return DailyWeatherRecord(
        days=np.asarray(days, dtype=np.int64),
        srad=srad32,
        tmax=tmax32,
        tmin=tmin32,
        rain=rain32,
        tavg=tavg,
        windsp=windsp,
        co2=co2,
        dayl=sun.dayl,
        xlat=float(xlat),
        windht=float(windht),
        files=tuple(files),
    )
