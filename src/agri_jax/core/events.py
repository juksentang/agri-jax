"""Management events expanded to per-day arrays (docs/en/02_architecture.md sections 3.5, 3.7).

An :class:`EventTable` is a :class:`~agri_jax.core.state.Forcing` whose leaves all have shape
``[T]``: it goes into the model forcing (``Forcing.events``) and ``lax.scan`` slices it with the
rest of the forcing, so a process sees one day of it (the same class with scalar leaves, also
obtainable with :meth:`EventTable.day`). State-dependent decisions (automatic irrigation) are
processes, not events.

Event kinds accepted by :meth:`EventTable.from_records` (value in brackets):

==============  ==========================================  ==================================
kind            aliases                                     effect
==============  ==========================================  ==================================
``sow``         ``planting``                                ``sow[t] = True`` (value ignored)
``harvest``                                                 ``harvest[t] = True``
``irrigation``  ``irrig``                                   ``irrig_cm[t] +=`` value [cm]
``fertilizer``  ``fert``, ``fertilizer_no3/_nh4/_urea``     ``fert_kg_ha[t] +=`` value [kg N/ha]
``topping``                                                 ``topping[t] = True``
``priming``                                                 ``priming_lo/hi[t] =`` value ``(lo, hi)``
==============  ==========================================  ==================================

Priming harvests the organ ranks ``lo <= rank < hi`` (0-based, see
:func:`agri_jax.core.organs.prime`); days without priming carry ``-1, -1``, an empty range, so
``prime(q, ev.priming_lo, ev.priming_hi)`` can be called unconditionally. The table carries no
``n_crop`` axis: rotations reuse the crop slot (section 3.5).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd
from jax.typing import ArrayLike
from jaxtyping import Array

from agri_jax.core.state import Forcing, field

__all__ = ["CSV_IGNORED", "EVENT_ALIASES", "EVENT_KINDS", "NO_PRIMING", "EventTable"]

EVENT_KINDS = ("sow", "harvest", "irrigation", "fertilizer", "topping", "priming")
"""Canonical event kinds, in the order :meth:`EventTable.to_records` emits same-day events."""

EVENT_ALIASES: dict[str, str] = {
    "planting": "sow",
    "irrig": "irrigation",
    "fert": "fertilizer",
    "fertilizer_no3": "fertilizer",
    "fertilizer_nh4": "fertilizer",
    "fertilizer_urea": "fertilizer",
}
"""Alternative spellings (``io.catpa`` event names) mapped to canonical kinds."""

CSV_IGNORED = ("pesticide", "tillage")
"""``events.csv`` kinds with no model effect yet; :meth:`EventTable.from_csv` drops them by default."""

NO_PRIMING = -1
"""Rank bound stored on days without priming (``[-1, -1)`` is empty)."""

_IRRIG_UNIT_TO_CM = {"": 1.0, "cm": 1.0, "mm": 0.1}


def _days(dates: pd.DatetimeIndex | Sequence[Any]) -> np.ndarray:
    """Forcing dates as ``datetime64[D]`` (times of day dropped)."""
    return np.asarray([_day(d) for d in dates], dtype="datetime64[D]")


def _day(date: Any) -> np.datetime64:
    """One date (string, Timestamp, datetime64, date) as ``datetime64[D]``."""
    return np.datetime64(pd.Timestamp(date).to_datetime64(), "D")


def _canonical(kind: str) -> str:
    k = str(kind).strip().lower()
    k = EVENT_ALIASES.get(k, k)
    if k not in EVENT_KINDS:
        raise ValueError(
            f"unknown event kind {kind!r}; known {list(EVENT_KINDS)} and aliases {list(EVENT_ALIASES)}"
        )
    return k


class EventTable(Forcing):
    """Per-day management arrays, every leaf of shape ``[T]`` (time axis of the forcing)."""

    sow: Array = field(unit="-", description="sowing day flag", dims="T")
    harvest: Array = field(unit="-", description="harvest day flag", dims="T")
    irrig_cm: Array = field(unit="cm", description="irrigation amount", dims="T")
    fert_kg_ha: Array = field(unit="kg ha-1", description="fertilizer N applied", dims="T")
    topping: Array = field(unit="-", description="topping day flag (stops organ appearance)", dims="T")
    priming_lo: Array = field(unit="-", description="first organ rank harvested (-1: no priming)", dims="T")
    priming_hi: Array = field(
        unit="-", description="one past the last organ rank harvested (-1: none)", dims="T"
    )

    @classmethod
    def empty(cls, n_days: int) -> EventTable:
        """A table of ``n_days`` days with no event."""
        z = jnp.zeros((n_days,))
        f = jnp.zeros((n_days,), dtype=bool)
        none = jnp.full((n_days,), NO_PRIMING, dtype=jnp.int32)
        return cls(sow=f, harvest=f, irrig_cm=z, fert_kg_ha=z, topping=f, priming_lo=none, priming_hi=none)

    @classmethod
    def from_records(
        cls,
        records: Iterable[tuple[Any, str, Any]],
        dates: pd.DatetimeIndex | Sequence[Any],
        *,
        outside: Literal["raise", "drop"] = "raise",
        ignore: Iterable[str] = (),
    ) -> EventTable:
        """Expand ``(date, kind, value)`` records onto the forcing days ``dates``.

        ``kind`` is a canonical kind or alias (module docstring); an unknown kind raises
        ``ValueError`` unless listed in ``ignore``. Same-day irrigation and fertilizer amounts
        add up; two primings on one day raise. A record dated outside ``dates`` raises, or is
        dropped with ``outside="drop"``. ``dates`` is the forcing's daily ``DatetimeIndex``
        (times of day are ignored).
        """
        idx = _days(dates)
        pos = {d: t for t, d in enumerate(idx)}
        if len(pos) != len(idx):
            raise ValueError("dates must be unique")
        n = len(idx)
        sow = np.zeros(n, dtype=bool)
        harvest = np.zeros(n, dtype=bool)
        topping = np.zeros(n, dtype=bool)
        irrig = np.zeros(n)
        fert = np.zeros(n)
        lo = np.full(n, NO_PRIMING, dtype=np.int32)
        hi = np.full(n, NO_PRIMING, dtype=np.int32)
        skip = {str(k).strip().lower() for k in ignore}

        for date, kind, value in records:
            if str(kind).strip().lower() in skip:
                continue
            k = _canonical(kind)
            t = pos.get(_day(date), -1)
            if t < 0:
                if outside == "drop":
                    continue
                raise ValueError(f"{kind} event on {pd.Timestamp(date).date()} is outside the forcing dates")
            if k == "sow":
                sow[t] = True
            elif k == "harvest":
                harvest[t] = True
            elif k == "topping":
                topping[t] = True
            elif k == "irrigation":
                irrig[t] += float(value)
            elif k == "fertilizer":
                fert[t] += float(value)
            else:  # priming
                r_lo, r_hi = (int(v) for v in value)
                if not 0 <= r_lo <= r_hi:
                    raise ValueError(f"priming ranks must satisfy 0 <= lo <= hi, got ({r_lo}, {r_hi})")
                if lo[t] != NO_PRIMING:
                    raise ValueError(f"two primings on {idx[t]}")
                lo[t], hi[t] = r_lo, r_hi

        return cls(
            sow=jnp.asarray(sow),
            harvest=jnp.asarray(harvest),
            irrig_cm=jnp.asarray(irrig),
            fert_kg_ha=jnp.asarray(fert),
            topping=jnp.asarray(topping),
            priming_lo=jnp.asarray(lo),
            priming_hi=jnp.asarray(hi),
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        dates: pd.DatetimeIndex | Sequence[Any],
        *,
        ignore: Iterable[str] = CSV_IGNORED,
        outside: Literal["raise", "drop"] = "drop",
    ) -> EventTable:
        """Load an ``events.csv`` written by :func:`agri_jax.io.catpa.write_events`.

        Uses the columns ``date``, ``event``, ``value``, ``unit`` and ``detail``
        (:data:`agri_jax.io.catpa.EVENT_COLUMNS`). Kinds in ``ignore`` (pesticide and tillage
        by default: no process consumes them yet) are dropped, as are events outside ``dates``
        (the csv covers the whole scenario, a run may cover a sub-window). Irrigation in
        ``mm`` is converted to cm. A priming row gives its ranks in ``detail`` as
        ``rank_lo=<int>;rank_hi=<int>``.
        """
        from agri_jax.io.catpa import load_events

        df = load_events(path)
        records: list[tuple[Any, str, Any]] = []
        for date, kind, value, unit, detail in zip(
            df["date"], df["event"], df["value"], df["unit"], df["detail"]
        ):
            k = str(kind).strip().lower()
            v: Any = value
            if k in {"irrigation", "irrig"}:
                u = str(unit).strip().lower()
                if u not in _IRRIG_UNIT_TO_CM:
                    raise ValueError(f"{path}: irrigation unit {unit!r} not in {list(_IRRIG_UNIT_TO_CM)}")
                v = float(value) * _IRRIG_UNIT_TO_CM[u]
            elif k == "priming":
                kv = dict(p.split("=", 1) for p in str(detail).split(";") if "=" in p)
                v = (int(kv["rank_lo"]), int(kv["rank_hi"]))
            records.append((date, str(kind), v))
        return cls.from_records(records, dates, outside=outside, ignore=ignore)

    def to_records(self, dates: pd.DatetimeIndex | Sequence[Any]) -> list[tuple[Any, str, Any]]:
        """Inverse of :meth:`from_records` in canonical kinds (flags get value ``1.0``).

        ``from_records(table.to_records(dates), dates)`` reproduces the table.
        """
        idx = _days(dates)
        if len(idx) != self.n_days:
            raise ValueError(f"{len(idx)} dates for a {self.n_days}-day table")
        a = {k: np.asarray(getattr(self, k)) for k in ("sow", "harvest", "irrig_cm", "fert_kg_ha", "topping")}
        lo, hi = np.asarray(self.priming_lo), np.asarray(self.priming_hi)
        out: list[tuple[Any, str, Any]] = []
        for t in range(self.n_days):
            day = pd.Timestamp(idx[t])
            if a["sow"][t]:
                out.append((day, "sow", 1.0))
            if a["harvest"][t]:
                out.append((day, "harvest", 1.0))
            if a["irrig_cm"][t] != 0.0:
                out.append((day, "irrigation", float(a["irrig_cm"][t])))
            if a["fert_kg_ha"][t] != 0.0:
                out.append((day, "fertilizer", float(a["fert_kg_ha"][t])))
            if a["topping"][t]:
                out.append((day, "topping", 1.0))
            if lo[t] != NO_PRIMING:
                out.append((day, "priming", (int(lo[t]), int(hi[t]))))
        return out

    def day(self, t: ArrayLike) -> EventTable:
        """The events of day ``t`` (a Python int or a traced index): the same class with scalar leaves.

        Equivalent to the slice ``lax.scan`` hands a process through ``forcing_t.events``.
        """
        return jtu.tree_map(lambda x: x[t], self)
