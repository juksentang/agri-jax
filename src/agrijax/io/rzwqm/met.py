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

:func:`write_brk` writes a :class:`BrkData` back in the layout of the RZWQM2 interface's
"breakpoint rainfall file from daily met file" (the ``=`` comment header kept verbatim, event
records ``I9 I9 I9 I9 F10.3``, breakpoint records of up to 5 pairs, the first pair ``F9.3 I9``
and the next ones ``F10.3 I10``, the file's line terminator); a file in that layout is
reproduced byte for byte (``tests/integration/test_io_m3_catpa.py`` on ``CA-TPA.BRK``).
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
    "format_brk",
    "prepare_rzwqm_forcing",
    "read_brk",
    "read_met",
    "write_brk",
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
#: radians to degrees, as the reference model sets it (a PARAMETER with pi to ten digits)
_R2D = 180.0 / 3.141592654

#: floor of the daily wind run [km d-1] the reference model applies when it reads a ``.MET``
#: record (a user parameter here)
WIND_FLOOR_KM_D: float = 100.0
#: bounds applied after the wind floor (user parameters here): column -> (low, high)
INPDAY_BOUNDS: dict[str, tuple[float, float]] = {
    "tmin": (-50.0, np.inf),  # TTN
    "tmax": (-np.inf, 50.0),  # TTX
    "wind_run_km": (45.0, 4700.0),  # TUN, TUX
    "srad_mj": (0.0, 45.0),  # TRN, TRX
    "rh": (0.0, 100.0),  # TRHN, TRHX
}


#: ``METMOD`` rows (0-based) of ``IPNAMES.DAT`` added to / multiplied (percent) into the ``.MET``
#: columns; the rainfall row (6) is applied by :func:`agrijax.io.rzwqm.storms.storm_arrays` only
_MET_MODIFIER_ADD: dict[str, int] = {"tmin": 0, "tmax": 1}
_MET_MODIFIER_PCT: dict[str, int] = {"wind_run_km": 2, "srad_mj": 3, "epan": 4, "rh": 5}
#: percent -> fraction
_PERCENT_TO_FRACTION = 1.0e-2


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
    met_modifiers: np.ndarray | None = None,
) -> pd.DataFrame:
    """Apply a daily forcing preparation to a :func:`read_met` frame: the wind run floored at
    ``wind_floor_km_d``, the optional monthly modifiers ``met_modifiers`` (the ``[8, 12]``
    ``METMOD`` block of ``IPNAMES.DAT``, :func:`agrijax.io.rzwqm.storms.read_met_modifiers`:
    temperatures added, wind, radiation, pan evaporation and RH multiplied by a percentage), then the
    bounds of :data:`INPDAY_BOUNDS`. The solar radiation is the ``.MET`` value (after its modifier);
    no hourly disaggregation is applied.

    Returns a copy; ``attrs`` gain ``prepared = "INPDAY"``.
    """
    out = met.copy()
    if "wind_run_km" in out:
        out["wind_run_km"] = np.maximum(out["wind_run_km"].to_numpy(dtype=float), wind_floor_km_d)
    if met_modifiers is not None:
        mod = np.asarray(met_modifiers, dtype=float)
        month = np.asarray(out.index.to_numpy(), dtype="datetime64[M]").astype(np.int64) % 12
        for col, row in _MET_MODIFIER_ADD.items():
            if col in out:
                out[col] = out[col].to_numpy(dtype=float) + mod[row, month]
        for col, row in _MET_MODIFIER_PCT.items():
            if col in out:
                out[col] = out[col].to_numpy(dtype=float) * mod[row, month] * _PERCENT_TO_FRACTION
        out.attrs.update(met_modifiers="METMOD")
    for col, (lo, hi) in INPDAY_BOUNDS.items():
        if col in out:
            out[col] = np.clip(out[col].to_numpy(dtype=float), lo, hi)
    out.attrs.update(prepared="INPDAY", wind_floor_km_d=wind_floor_km_d)
    return out


@dataclass(frozen=True)
class BrkData:
    """Breakpoint rainfall: one row per event plus one row per breakpoint.

    ``header`` holds the ``=`` comment lines before the calendar code (without terminators) and
    ``newline`` the file's line terminator; both only serve :func:`write_brk`.
    """

    calendar_code: int
    events: pd.DataFrame  # event, year, doy, date, n_breakpoints, midnight, depth_in, depth_mm
    breakpoints: pd.DataFrame  # event, cum_depth_in, cum_depth_mm, time_min
    header: tuple[str, ...] = ()
    newline: str = "\n"

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
    text = Path(path).read_bytes().decode("latin-1")
    newline = "\r\n" if "\r\n" in text else "\n"
    header: list[str] = []
    for ln in text.splitlines():
        if ln.strip() and not ln.startswith("="):
            break
        header.append(ln)
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
    return BrkData(
        calendar_code=code, events=events, breakpoints=breakpoints, header=tuple(header), newline=newline
    )


#: breakpoint pairs per record 3 of a ``.BRK`` file (module docstring)
_BRK_PAIRS_PER_RECORD = 5


def _brk_int(x: float, what: str) -> int:
    if float(x) != round(float(x)):
        raise ValueError(f"{what} {x} is not an integer; the .BRK layout writes it as I9/I10")
    return round(float(x))


def format_brk(brk: BrkData) -> str:
    """The text of :func:`write_brk` (see the module docstring for the layout)."""
    nl = brk.newline
    lines = [*brk.header, str(int(brk.calendar_code))]
    ev = brk.events
    bp = brk.breakpoints
    owner = bp["event"].to_numpy(dtype=np.int64)
    cum = bp["cum_depth_in"].to_numpy(dtype=float)
    tim = bp["time_min"].to_numpy(dtype=float)
    cols = [ev[c].to_numpy() for c in ("year", "doy", "n_breakpoints", "midnight", "depth_in")]
    rows = zip(ev.index.to_numpy(), zip(*cols, strict=True), strict=True)
    for k, (year, doy, nbp, midnight, depth) in rows:
        sel = owner == int(k)
        if int(sel.sum()) != int(nbp):
            raise ValueError(f"event {k}: {int(sel.sum())} breakpoints, header says {int(nbp)}")
        lines.append(f"{int(year):9d}{int(doy):9d}{int(nbp):9d}{int(midnight):9d}{float(depth):10.3f}")
        pairs = list(zip(cum[sel], tim[sel], strict=True))
        for i in range(0, len(pairs), _BRK_PAIRS_PER_RECORD):
            rec = ""
            for j, (c, t) in enumerate(pairs[i : i + _BRK_PAIRS_PER_RECORD]):
                w = 9 if j == 0 else 10
                rec += f"{c:{w}.3f}{_brk_int(t, 'breakpoint time [min]'):{w}d}"
            lines.append(rec)
    return nl.join(lines) + nl


def write_brk(brk: BrkData, path: str | Path) -> Path:
    """Write ``brk`` as a ``.BRK`` file (latin-1); the inverse of :func:`read_brk` for files in
    the layout of the module docstring."""
    p = Path(path)
    p.write_bytes(format_brk(brk).encode("latin-1"))
    return p
