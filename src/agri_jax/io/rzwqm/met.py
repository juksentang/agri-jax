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


def prepare_rzwqm_forcing(met: pd.DataFrame, *, wind_floor_km_d: float = WIND_FLOOR_KM_D) -> pd.DataFrame:
    """Apply RZWQM2's daily forcing preparation (``INPDAY``) to a :func:`read_met` frame.

    1. wind run floored at ``wind_floor_km_d`` (``UBREEZ = 100`` km d-1, ``Rzmain.for`` line 3543);
    2. the bounds of :data:`INPDAY_BOUNDS` (``Rzmain.for`` lines 3575-3582).

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
    return out


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
