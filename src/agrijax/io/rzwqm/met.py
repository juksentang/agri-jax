"""Readers for the RZWQM2 daily meteorology (``.MET``) and breakpoint rainfall (``.BRK``) files.

``.MET`` (daily, MetFlag = 0), from the header comments of ``CA-TPA.MET``::

    record 1:    begin date [yyyy mm dd]  end date [yyyy mm dd]  flag (0 daily .met, 1 hourly .hmt)
    record 2..N: doy  year  tmin[C]  tmax[C]  wind run[km/day]  sw rad[MJ/m2/day]
                 pan evap[cm/day]  RH[0..100]  PAR[mol/m2/day]  rain[mm]

``.BRK``::

    record 1:  calendar code (1 = day counted as Julian day within the year)
    record 2:  year  doy  n_breakpoints  midnight-spanning flag (1 = yes)  total storm depth [in]
    record 3:  pairs (cumulative storm depth [in], clock time [min]), 5 pairs per record,
               repeated until n_breakpoints pairs are read; then the next record 2.

The depth unit of the BRK file is inches (header says so, and CA-TPA's events equal the
MET daily rain / 25.4); ``depth_mm`` columns are added for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ._fortran import fortran_float, year_doy_to_datetime64

__all__ = [
    "INPDAY_BOUNDS",
    "MET_COLUMNS",
    "MET_UNITS",
    "WIND_FLOOR_KM_D",
    "BrkData",
    "prepare_rzwqm_forcing",
    "read_brk",
    "read_met",
    "rzwqm_daily_srad",
]

MET_COLUMNS: tuple[str, ...] = (
    "tmin",
    "tmax",
    "wind_run_km",
    "srad_mj",
    "epan",
    "rh",
    "par",
    "rain_mm",
)
MET_UNITS: dict[str, str] = {
    "tmin": "degC",
    "tmax": "degC",
    "wind_run_km": "km/day",
    "srad_mj": "MJ/m2/day",
    "epan": "cm/day",
    "rh": "percent",
    "par": "mol/m2/day",
    "rain_mm": "mm/day",
}
_INCH_MM = 25.4
#: ``R2D`` of ``Rzmain.for`` (PARAMETER, line 244).
_R2D = 180.0 / 3.141592654

#: RZWQM2 floors the daily wind run at ``UBREEZ = 100`` km d-1 when it reads a ``.MET`` record
#: (``INPDAY``, ``Rzmain.for`` lines 3521-3522 PARAMETER and line 3543 ``U = MAX(U, UBREEZ)``).
WIND_FLOOR_KM_D: float = 100.0
#: bounds ``INPDAY`` then applies (``Rzmain.for`` lines 3521-3522 and 3575-3582): column -> (low, high).
INPDAY_BOUNDS: dict[str, tuple[float, float]] = {
    "tmin": (-50.0, np.inf),  # TTN
    "tmax": (-np.inf, 50.0),  # TTX
    "wind_run_km": (45.0, 4700.0),  # TUN, TUX
    "srad_mj": (0.0, 45.0),  # TRN, TRX
    "rh": (0.0, 100.0),  # TRHN, TRHX
}


def _data_lines(path: str | Path) -> list[list[str]]:
    text = Path(path).read_bytes().decode("latin-1")
    return [ln.split() for ln in text.splitlines() if ln.strip() and not ln.startswith("=")]


def _doy_to_date(year: np.ndarray, doy: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(year_doy_to_datetime64(year, doy).astype("datetime64[ns]"))


def read_met(path: str | Path, *, prepare: bool = False) -> pd.DataFrame:
    """Daily ``.MET`` file -> DataFrame indexed by ``date`` with :data:`MET_COLUMNS`.

    ``df.attrs`` holds ``begin``, ``end`` (Timestamps from record 1), ``met_flag`` and ``units``.
    The values are returned as written in the file; with ``prepare=True`` the result is passed
    through :func:`prepare_rzwqm_forcing` (what the reference model actually uses).
    """
    rows = _data_lines(path)
    if not rows:
        raise ValueError(f"{path}: no data records")
    head, body = rows[0], rows[1:]
    if len(head) < 7:
        raise ValueError(f"{path}: record 1 should hold begin/end dates and the MetFlag, got {head}")
    flag = int(head[6])
    if flag != 0:
        raise NotImplementedError(f"{path}: MetFlag={flag} (hourly .hmt) is not supported")
    ncol = 2 + len(MET_COLUMNS)
    bad = [i for i, r in enumerate(body) if len(r) < ncol]
    if bad:
        raise ValueError(f"{path}: data record {bad[0] + 2} has fewer than {ncol} fields: {body[bad[0]]}")
    arr = np.array([[fortran_float(t) for t in r[:ncol]] for r in body], dtype=np.float64)
    index = _doy_to_date(arr[:, 1].astype(int), arr[:, 0].astype(int))
    index.name = "date"
    df = pd.DataFrame(arr[:, 2:], index=index, columns=list(MET_COLUMNS))
    df.attrs.update(
        begin=pd.Timestamp(int(head[0]), int(head[1]), int(head[2])),
        end=pd.Timestamp(int(head[3]), int(head[4]), int(head[5])),
        met_flag=flag,
        units=dict(MET_UNITS),
    )
    return prepare_rzwqm_forcing(df) if prepare else df


def prepare_rzwqm_forcing(
    met: pd.DataFrame,
    *,
    wind_floor_km_d: float = WIND_FLOOR_KM_D,
    latitude_rad: float | None = None,
    slope_rad: float = 0.0,
    aspect_rad: float = 0.0,
) -> pd.DataFrame:
    """Apply RZWQM2's daily forcing preparation (``INPDAY``) to a :func:`read_met` frame.

    1. wind run floored at ``wind_floor_km_d`` (``UBREEZ = 100`` km d-1, ``Rzmain.for`` line 3543);
    2. the bounds of :data:`INPDAY_BOUNDS` (``Rzmain.for`` lines 3575-3582);
    3. only when ``latitude_rad`` is given (the ``rzwqm.dat`` physiography, radians): the daily
       solar radiation the model actually uses, i.e. the re-sum of its hourly disaggregation
       (:func:`rzwqm_daily_srad`); the ``.MET`` value is kept in ``srad_mj_met``. This is what
       ``.ana`` column 88 prints (-0.7 % .. +0.5 % from the ``.MET`` value day to day).

    With these two steps the CA-TPA ``.MET`` wind equals the ``.ana`` wind column (90) on every
    day; without them the floor is the whole MET-vs-ana difference in reference ET (max 0.535
    mm d-1 on 2015-07-28). Not reproduced: the monthly ``METMOD`` multipliers (the identity for
    CA-TPA, whose ``.ana`` tmin/tmax/wind/RH equal the prepared ``.MET`` exactly) and the
    ``RH = 0`` replacement by a dew-point estimate (no CA-TPA record has RH = 0).
    Returns a copy; ``attrs`` gain ``prepared = "INPDAY"``.
    """
    out = met.copy()
    if "wind_run_km" in out:
        out["wind_run_km"] = np.maximum(out["wind_run_km"].to_numpy(dtype=float), wind_floor_km_d)
    for col, (lo, hi) in INPDAY_BOUNDS.items():
        if col in out:
            out[col] = np.clip(out[col].to_numpy(dtype=float), lo, hi)
    out.attrs.update(prepared="INPDAY", wind_floor_km_d=wind_floor_km_d)
    if latitude_rad is not None and "srad_mj" in out:
        idx = pd.DatetimeIndex(out.index)
        out["srad_mj_met"] = out["srad_mj"]
        out["srad_mj"] = rzwqm_daily_srad(
            out["srad_mj"].to_numpy(dtype=float),
            np.array([d.timetuple().tm_yday for d in idx], dtype=int),
            latitude_rad,
            slope_rad=slope_rad,
            aspect_rad=aspect_rad,
        )
        out.attrs.update(srad="hourly re-sum (rzwqm_daily_srad)")
    return out


def _hourly_radiation_dssat40(srad_mj: np.ndarray, doy: np.ndarray, lat_deg: np.float32) -> np.ndarray:
    """Hourly global radiation [W m-2], hours 1..24, as DSSAT 4.0 ``HMET`` computes it (``[n, 24]``).

    ``DAYLEN`` (declination, day length), ``SOLAR`` (the daily integral ``ISINB`` of Spitters'
    eq. 6), ``HANG`` (solar elevation at each whole hour, raised to 1 degree once above 1e-4)
    and ``HRAD`` (``sinB (1 + 0.4 sinB) SRAD / ISINB`` between sunrise and sunset). DSSAT declares
    these ``REAL``, so the arithmetic is float32 with ``PI = 3.14159``.
    """
    f = np.float32
    pi = f(3.14159)
    rad = f(pi / f(180.0))
    srad = srad_mj.astype(f)[:, None]
    d = doy.astype(f)[:, None]
    dec = f(-23.45) * np.cos(f(2.0) * pi * (d + f(10.0)) / f(365.0))
    soc = np.clip(np.tan(rad * dec) * np.tan(rad * lat_deg), f(-1.0), f(1.0))
    dayl = f(12.0) + f(24.0) * np.arcsin(soc) / pi
    snup, sndn = f(12.0) - dayl / f(2.0), f(12.0) + dayl / f(2.0)
    ssin = np.sin(rad * dec) * np.sin(rad * lat_deg)
    ccos = np.cos(rad * dec) * np.cos(rad * lat_deg)
    soc2 = np.clip(ssin / ccos, f(-1.0), f(1.0))
    isinb = f(3600.0) * (
        dayl * (ssin + f(0.4) * (ssin**2 + f(0.5) * ccos**2))
        + f(24.0) / pi * ccos * (f(1.0) + f(1.5) * f(0.4) * ssin) * np.sqrt(f(1.0) - soc2**2)
    )
    hs = np.arange(1, 25, dtype=f)[None, :]
    hangl = (hs - f(12.0)) * pi / f(12.0)
    beta = np.arcsin(np.clip(ssin + ccos * np.cos(hangl), f(-1.0), f(1.0))) / rad
    beta = np.where(beta > f(1e-4), np.maximum(beta, f(1.0)), beta)
    sinb = np.sin(rad * beta)
    day = (hs > snup) & (hs < sndn)
    radhr = np.where(day, sinb * (f(1.0) + f(0.4) * sinb) * srad * f(1.0e6) / isinb, f(0.0))
    return radhr.astype(f)


def _shaw_slope_sum(
    sunhor: np.ndarray, doy: np.ndarray, lat: float, slope: float, aspect: float
) -> np.ndarray:
    """SHAW's ``CLOUDY`` + ``SOLAR_SHAW`` (double): direct + diffuse on the slope, summed [MJ m-2]."""
    pi = 3.14159
    solcon, difatm = 1360.0, 0.76
    n = sunhor.shape[0]
    out = np.zeros(n)
    for k in range(n):
        declin = 0.4102 * np.sin(2 * pi * (int(doy[k]) - 80) / 365.0)
        coshaf = -np.tan(lat) * np.tan(declin)
        hafday = (0.0 if coshaf >= 1.0 else pi) if abs(coshaf) >= 1.0 else float(np.arccos(coshaf))
        sunris, sunset = 12.0 - hafday / 0.261799, 12.0 + hafday / 0.261799
        hrwest = pi if abs(declin) >= abs(lat) else float(np.arccos(np.tan(declin) / np.tan(lat)))
        rts = 0.0
        for hour in range(1, 25):
            sh = max(float(sunhor[k, hour - 1]), 0.0)  # CLOUDY clips negative hourly values
            if sh <= 0.0:
                continue
            sinazm = cosazm = sumalt = cosalt = sunmax = 0.0
            for thour in (hour - 1.0, float(hour)):  # NHRPDT = 1: both ends of the hour
                hrangl = 0.261799 * (thour - 12.0)
                if sunris < thour < sunset:
                    sinalt = np.sin(lat) * np.sin(declin) + np.cos(lat) * np.cos(declin) * np.cos(hrangl)
                    altitu = np.arcsin(sinalt)
                    azm = np.arcsin(-np.cos(declin) * np.sin(hrangl) / np.cos(altitu))
                    if lat - declin > 0.0:
                        if abs(hrangl) < hrwest:
                            azm = pi - azm
                    elif abs(hrangl) >= hrwest:
                        azm = pi - azm
                    sun = solcon * sinalt
                    sumalt += sun * sinalt
                    cosalt += sun * np.cos(altitu)
                    sinazm += sun * np.sin(azm)
                    cosazm += sun * np.cos(azm)
                    sunmax += sun
            if sunmax == 0.0:
                altitu = sunslp = 0.0
            else:
                altitu = float(np.arctan(sumalt / cosalt))
                azmuth = float(np.arctan2(sinazm, cosazm))
                sunmax /= 2.0
                sunslp = float(
                    np.arcsin(
                        np.sin(altitu) * np.cos(slope)
                        + np.cos(altitu) * np.sin(slope) * np.cos(azmuth - aspect)
                    )
                )
            if altitu <= 0.0:
                total = sh
            else:
                tt = min(sh / sunmax, difatm)
                tdiffu = tt * (1.0 - np.exp(0.6 * (1.0 - difatm / tt) / (difatm - 0.4)))
                diffus = tdiffu * sunmax
                dirhor = sh - diffus
                direct = 0.0 if sunslp <= 0.0 else min(dirhor * np.sin(sunslp) / np.sin(altitu), 5.0 * dirhor)
                total = direct + diffus
            rts += total * 3.6e3 / 1.0e6
        out[k] = rts
    return out


def rzwqm_daily_srad(
    srad_mj: np.ndarray,
    doy: np.ndarray,
    latitude_rad: float,
    *,
    slope_rad: float = 0.0,
    aspect_rad: float = 0.0,
) -> np.ndarray:
    """Daily solar radiation [MJ m-2 d-1] that RZWQM2 uses and prints in ``.ana`` column 88.

    RZWQM2 (daily ``.MET`` input, ``Iweather = 0``) disaggregates the daily value into 24 hourly
    values with DSSAT 4.0 ``HMET`` (:func:`_hourly_radiation_dssat40`), partitions each hour
    into direct and diffuse radiation on the local slope with SHAW's ``SOLAR_SHAW`` and sums the
    24 hours again (``RTS``, ``Rzmain.for`` 1376-1389). The hourly values are the sine-of-elevation
    shape sampled at whole hours, so their sum is not the daily input: the ratio is 0.993..1.005
    at CA-TPA. On a flat surface (``slope_rad == 0``) direct + diffuse is the hourly value itself
    and the result is the float32 hourly sum; otherwise the SHAW partition is applied.

    Written from the equations (Spitters et al. 1986; Flerchinger 2000), validated against the
    ``.ana`` column 88 of the RZWQM2 binary on the 15 ``RZWQM_sw_batch`` scenarios.
    """
    srad = np.asarray(srad_mj, dtype=float)
    days = np.asarray(doy).astype(int)
    lat_deg = np.float32(float(latitude_rad) * _R2D)  # REAL(XLAT*R2D), product in double
    hourly = _hourly_radiation_dssat40(srad, days, lat_deg).astype(np.float64)
    if slope_rad == 0.0:
        return np.maximum(hourly, 0.0).sum(axis=1) * 3.6e3 / 1.0e6
    return _shaw_slope_sum(hourly, days, float(latitude_rad), float(slope_rad), float(aspect_rad))


@dataclass(frozen=True)
class BrkData:
    """Breakpoint rainfall: one row per event plus one row per breakpoint."""

    calendar_code: int
    events: pd.DataFrame  # event, year, doy, date, n_breakpoints, midnight, depth_in, depth_mm
    breakpoints: pd.DataFrame  # event, cum_depth_in, cum_depth_mm, time_min

    def daily_rain_mm(self) -> pd.Series:
        """Total storm depth summed per calendar day [mm]."""
        s = cast(pd.Series, self.events.groupby("date")["depth_mm"].sum())
        s.name = "rain_mm"
        return s


def read_brk(path: str | Path) -> BrkData:
    """Parse a ``.BRK`` breakpoint rainfall file (see module docstring for the layout)."""
    rows = _data_lines(path)
    if not rows:
        raise ValueError(f"{path}: no data records")
    code = int(rows[0][0])
    ev: list[tuple[int, int, int, int, float]] = []
    bp: list[tuple[int, float, float]] = []
    i = 1
    while i < len(rows):
        r = rows[i]
        if len(r) < 5:
            raise ValueError(f"{path}: expected an event record (5 fields), got {r}")
        year, doy, nbp, midnight = (int(float(x)) for x in r[:4])
        depth = fortran_float(r[4])
        k = len(ev)
        ev.append((year, doy, nbp, midnight, depth))
        i += 1
        vals: list[float] = []
        while len(vals) < 2 * nbp:
            if i >= len(rows):
                raise ValueError(f"{path}: event {year}/{doy} ends before its {nbp} breakpoints")
            vals.extend(fortran_float(t) for t in rows[i])
            i += 1
        for j in range(nbp):
            bp.append((k, vals[2 * j], vals[2 * j + 1]))
    events = pd.DataFrame(ev, columns=["year", "doy", "n_breakpoints", "midnight", "depth_in"])
    events.index.name = "event"
    events.insert(2, "date", _doy_to_date(events["year"].to_numpy(), events["doy"].to_numpy()))
    events["depth_mm"] = events["depth_in"] * _INCH_MM
    breakpoints = pd.DataFrame(bp, columns=["event", "cum_depth_in", "time_min"])
    breakpoints.insert(2, "cum_depth_mm", breakpoints["cum_depth_in"] * _INCH_MM)
    return BrkData(calendar_code=code, events=events, breakpoints=breakpoints)
