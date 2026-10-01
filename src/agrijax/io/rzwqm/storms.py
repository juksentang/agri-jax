"""Per-day water-event arrays of an RZWQM2 scenario: breakpoint storms and irrigation.

The soil-water day (``soil_water/day@rzwqm2-4.6:faithful``) takes one storm segment per day
(:class:`~agrijax.processes.soil_water.infiltration.StormForcing`: start clock time ``ts0`` [h],
breakpoint interval lengths ``duration`` [h] and depths ``depth`` [cm], padded to a common
``n_bp``). :func:`storm_arrays` builds those arrays from a ``.BRK`` file (:func:`read_brk`) the way
RZWQM2 4.6 reads it (``STMINP``, Rzday.for:3758-3962, RZWQM2 4.5 source, read only):

* a storm whose total in the event record is below :data:`STMINP_MIN_DEPTH_IN` is skipped
  (Rzday.for:3842, 3847);
* the cumulative breakpoints (clock time [min], depth [in]) become increments, converted to
  hours and centimetres (Rzday.for:3915-3916) and multiplied by the rainfall modifier of
  ``IPNAMES.DAT`` (``METMOD`` row 7, percent; Rzday.for:3919, read at Rzmain.for:9185-9188);
* the start clock time is the first breakpoint's time (Rzday.for:3868);
* a storm that crosses midnight continues on the next day at ``ts0 = 0`` with each interval
  split in proportion to time (``CHSPAN``, Rzday.for:126). CA-TPA has no such storm, so this
  split is checked only by the synthetic unit test, not against the reference.

RZWQM2 applies the modifier of the month of the day on which it reads the storm, which is the
day the previous storm ended (``CDATE(JDAY, ...)`` at Rzday.for:3913); :func:`storm_arrays`
therefore accepts only a modifier that is the same in every month (CA-TPA: 100 %) and raises
otherwise. A day's precipitation that falls while RZWQM2's snow routine is active is not an
infiltration event there (the snow branch accumulates it, Rzday.for:1528-1597); that partition
belongs to the snow module, so these arrays carry every breakpoint storm and the consumer
decides.

Irrigation (:func:`irrigation_cm`): the ``IRRIGATION MANAGEMENT`` block of ``rzwqm.dat`` gives
the number of irrigation operations in its first record; with none, the per-day irrigation is
zero on every day. Operations are not parsed yet (their timings depend on the simulation, as in
:func:`agrijax.io.rzwqm.events.read_management`) and raise ``NotImplementedError``.

Nothing here imports JAX. Validated on CA-TPA 2015-2023 against the ``EVNTRO`` entry breakpoints
(``BPWHEN``, ``BPMUCH``, ``NBP``) of the instrumented RZWQM2 4.6 run
(``tests/integration/test_io_m3_catpa.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .dat import RzwqmDat
from .met import BrkData

__all__ = [
    "CM_PER_INCH",
    "MET_MODIFIER_ROWS",
    "MINUTES_PER_HOUR",
    "RAIN_MODIFIER_ROW",
    "STMINP_MIN_DEPTH_IN",
    "StormArrays",
    "irrigation_cm",
    "irrigation_operations",
    "read_met_modifiers",
    "storm_arrays",
    "storm_depths",
]

#: RZWQM2 ``STMINP`` skips a breakpoint storm whose event-record total is below this depth [in]
#: (RZWQM2 4.5 Rzday.for:3842 and 3847; the 4.6 binary behaves the same: CA-TPA 2015-2023 has
#: no ``EVNTRO`` event for a storm below it, tests/integration/test_io_m3_catpa.py).
STMINP_MIN_DEPTH_IN: float = 0.01
#: unit conversion inch -> cm (exact; Rzday.for:3916 uses the same factor)
CM_PER_INCH: float = 2.54
#: unit conversion hour -> minutes (Rzday.for:3915)
MINUTES_PER_HOUR: float = 60.0
#: percent -> fraction of the ``METMOD`` modifiers, as the reference writes it (``* 1.0D-2``,
#: Rzday.for:3919)
_PERCENT_TO_FRACTION: float = 1.0e-2
#: the identity modifier, 100 % (every month of the 15 reference scenarios)
_IDENTITY_PERCENT: float = 100.0
#: hours of a day: the clock at which a storm segment is cut (CHSPAN)
_HOURS_PER_DAY: float = 24.0
#: rows of the meteorology modifier matrix of ``IPNAMES.DAT`` (Rzmain.for:9185-9188; the file's
#: own comment lines name them): additive for the temperatures [degC], percent for the others
MET_MODIFIER_ROWS: tuple[str, ...] = (
    "tmin_add_degC",
    "tmax_add_degC",
    "wind_pct",
    "srad_pct",
    "epan_pct",
    "rh_pct",
    "rain_pct",
    "co2_pct",
)
#: 0-based row of the rainfall modifier (``METMOD(7, month)``)
RAIN_MODIFIER_ROW: int = 6
_N_MONTHS = 12
#: 0-based line of ``IPNAMES.DAT`` holding the simulation period; the 8 modifier rows follow it
_IPNAMES_PERIOD_LINE = 8


def read_met_modifiers(ipnames: str | Path) -> np.ndarray:
    """The ``METMOD(8, 12)`` meteorology modifiers of an ``IPNAMES.DAT`` as ``[8, 12]`` (rows
    :data:`MET_MODIFIER_ROWS`, columns January..December): the 8 records after the simulation
    period (Rzmain.for:9174 and 9185-9188)."""
    lines = Path(ipnames).read_bytes().decode("latin-1").splitlines()
    rows = lines[_IPNAMES_PERIOD_LINE + 1 : _IPNAMES_PERIOD_LINE + 1 + len(MET_MODIFIER_ROWS)]
    out = np.array([[float(t) for t in ln.split()[:_N_MONTHS]] for ln in rows], dtype=float)
    if out.shape != (len(MET_MODIFIER_ROWS), _N_MONTHS):
        raise ValueError(f"{ipnames}: expected 8 modifier records of 12 values, got shape {out.shape}")
    return out


@dataclass(frozen=True)
class StormArrays:
    """One storm segment per day (numpy; :class:`StormForcing` fields plus bookkeeping).

    ``ts0`` [h] is the clock time of the segment start (24 on days without a storm), ``duration``
    [h] and ``depth`` [cm] the breakpoint intervals ``[T, n_bp]`` (zero padded), ``event`` the
    row of :attr:`BrkData.events` that owns the day's segment (-1: none).
    """

    days: np.ndarray
    ts0: np.ndarray
    duration: np.ndarray
    depth: np.ndarray
    event: np.ndarray

    @property
    def n_bp(self) -> int:
        """Padded number of breakpoint intervals."""
        return int(self.depth.shape[1])

    @property
    def total_cm(self) -> np.ndarray:
        """Storm depth of each day [cm]."""
        return self.depth.sum(axis=1)


def _rain_percent(met_modifiers: np.ndarray | None) -> float | None:
    if met_modifiers is None:
        return None
    row = np.asarray(met_modifiers, dtype=float)[RAIN_MODIFIER_ROW]
    if not np.all(row == row[0]):
        raise NotImplementedError(
            "a rainfall modifier that changes with the month is applied by RZWQM2 with the month of "
            "the day the storm is read (the previous storm's end), which is not reproduced"
        )
    return float(row[0])


def storm_depths(
    brk: BrkData,
    *,
    met_modifiers: np.ndarray | None = None,
    min_storm_in: float = STMINP_MIN_DEPTH_IN,
) -> pd.DataFrame:
    """The storms ``STMINP`` reads and their depth ``RFDNEW`` [cm], one row per storm kept.

    A storm whose event-record total is below ``min_storm_in`` is skipped (Rzday.for:3842), as is
    one without breakpoints; the depth is the last cumulative breakpoint [in] converted to cm and
    multiplied by the rainfall modifier (percent, then the fraction; Rzday.for:3916-3924). The
    modifier is the one of :func:`storm_arrays` (a month-dependent modifier raises; ``None``:
    100 %). Index: the event (row of :attr:`BrkData.events`); columns ``date`` (the day the
    storm starts, ``datetime64[D]``) and ``depth_cm``. This is the parsed storm list that
    :func:`agrijax.forcing.precipitation.daily_storm_precipitation` sums per day.
    """
    pct = _rain_percent(met_modifiers)
    if pct is None:
        pct = _IDENTITY_PERCENT
    last = brk.breakpoints.groupby("event", sort=False)["cum_depth_in"].last()
    ev = brk.events.loc[brk.events["depth_in"] >= min_storm_in]
    ev = ev.loc[ev.index.isin(last.index)]
    depth = last.loc[ev.index].to_numpy(dtype=np.float64) * CM_PER_INCH * pct * _PERCENT_TO_FRACTION
    days = np.asarray(pd.to_datetime(ev["date"]).to_numpy(), dtype="datetime64[D]")
    return pd.DataFrame({"date": days, "depth_cm": depth}, index=ev.index)


def storm_arrays(
    days: Sequence[np.datetime64] | np.ndarray,
    brk: BrkData,
    *,
    met_modifiers: np.ndarray | None = None,
    min_storm_in: float = STMINP_MIN_DEPTH_IN,
) -> StormArrays:
    """Per-day storm segments of ``brk`` on the forcing ``days`` (module docstring).

    ``met_modifiers`` is :func:`read_met_modifiers` of the scenario (``None``: no modifier). A day
    that two storms would share raises ``ValueError`` (RZWQM2 runs one storm event at a time and
    the day's arrays hold one segment).
    """
    d = np.asarray(days, dtype="datetime64[D]")
    index = {day: k for k, day in enumerate(d)}
    pct = _rain_percent(met_modifiers)
    segs: dict[int, list[tuple[float, float]]] = {}
    start: dict[int, float] = {}
    owner: dict[int, int] = {}
    bp = brk.breakpoints
    owner_bp = bp["event"].to_numpy(dtype=np.int64)
    t_bp = bp["time_min"].to_numpy(dtype=float) / MINUTES_PER_HOUR
    c_bp = bp["cum_depth_in"].to_numpy(dtype=float)
    ev_ids = brk.events.index.to_numpy(dtype=np.int64)
    ev_depth = brk.events["depth_in"].to_numpy(dtype=float)
    ev_day = np.asarray(brk.events["date"].to_numpy(), dtype="datetime64[D]")
    for ev_id, depth_in, day0 in zip(ev_ids, ev_depth, ev_day, strict=True):
        if depth_in < min_storm_in:
            continue
        sel = owner_bp == ev_id
        t, c = t_bp[sel], c_bp[sel]
        if len(t) < 2:
            continue
        eid = int(ev_id)
        for k in range(len(t) - 1):
            t0, t1 = float(t[k]), float(t[k + 1])
            dd = float(c[k + 1] - c[k]) * CM_PER_INCH  # Rzday.for:3916: the increment, then inches -> cm
            if pct is not None:
                dd = dd * pct * _PERCENT_TO_FRACTION  # Rzday.for:3919
            while t1 > t0:
                off = int(np.floor(t0 / _HOURS_PER_DAY))
                cut = min(t1, _HOURS_PER_DAY * (off + 1))
                share = (cut - t0) / (t1 - t0)
                i = index.get(day0 + np.timedelta64(off, "D"))
                if i is not None:
                    if owner.setdefault(i, eid) != eid:
                        raise ValueError(f"two storms on {d[i]}: the day's arrays hold one storm segment")
                    start.setdefault(i, t0 - _HOURS_PER_DAY * off)
                    segs.setdefault(i, []).append((cut - t0, dd * share))
                dd *= 1.0 - share
                t0 = cut
    width = max([len(v) for v in segs.values()] + [1])
    n = len(d)
    ts0 = np.full(n, _HOURS_PER_DAY)
    dur = np.zeros((n, width))
    dep = np.zeros((n, width))
    event = np.full(n, -1, dtype=np.int64)
    for i, v in segs.items():
        ts0[i] = start[i]
        dur[i, : len(v)] = [a for a, _ in v]
        dep[i, : len(v)] = [b for _, b in v]
        event[i] = owner[i]
    return StormArrays(days=d, ts0=ts0, duration=dur, depth=dep, event=event)


def irrigation_operations(dat: RzwqmDat) -> int:
    """Number of irrigation operations of the ``IRRIGATION MANAGEMENT`` block (its record 1.1)."""
    b = next((b for b in dat.blocks if "I R R I G A T I O N   M A N A G" in b.header.upper()), None)
    if b is None:
        raise ValueError(f"{dat.path}: no IRRIGATION MANAGEMENT block")
    return int(dat.tokens(b.start)[0])


def irrigation_cm(dat: RzwqmDat, days: Sequence[np.datetime64] | np.ndarray) -> np.ndarray:
    """Irrigation depth of each day [cm] (``[T]``): zeros when the scenario has no irrigation
    operation; operations are not parsed yet and raise ``NotImplementedError``."""
    n_ops = irrigation_operations(dat)
    if n_ops != 0:
        raise NotImplementedError(f"{dat.path}: {n_ops} irrigation operations; irrigation is not parsed yet")
    return np.zeros(len(np.asarray(days)), dtype=float)
