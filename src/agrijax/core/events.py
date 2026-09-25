"""Management events expanded to per-day arrays.

An :class:`EventTable` is a :class:`~agrijax.core.state.Forcing` whose leaves all have shape
``[T]``: it goes into the model forcing (``Forcing.events``) and ``lax.scan`` slices it with the
rest of the forcing, so a process sees one day of it (the same class with scalar leaves, also
obtainable with :meth:`EventTable.day`). State-dependent decisions (automatic irrigation) are
processes, not events.

Event kinds
-----------
Every kind is an :class:`EventKind` of :data:`EVENT_REGISTRY`: a name, aliases, and the payload
keys a record may carry, each bound to one ``[T]`` leaf with a unit and a same-day rule
(``add``: amounts of one day add up; ``same``: an attribute that same-day records must agree on,
unset allowed; ``flag``: the day is marked). A record is ``(date, kind, value)``; ``value`` is a
number (bound to the kind's scalar key, or to the key an alias names), a ``(lo, hi)`` pair for
priming, or a ``{key: value}`` payload dict.

==============  ===========================  ==========================================================
kind            aliases                      payload keys -> leaves (units)
==============  ===========================  ==========================================================
``sow``         ``planting``                 flag ``sow`` (value ignored)
``harvest``                                  flag ``harvest``
``irrigation``  ``irrig``                    ``amount_cm`` -> ``irrig_cm`` [cm] (scalar)
``fertilizer``  ``fert``,                    ``no3`` / ``nh4`` / ``urea`` / ``org`` / ``unspecified``
                ``fertilizer_no3/_nh4/       -> ``fert_<key>_kg_ha`` [kg N ha-1] (add); ``depth_cm`` ->
                _urea/_org``                 ``fert_depth_cm`` [cm], ``method`` -> ``fert_method`` [code]
                                             (same). A bare number is ``unspecified`` (``fert``,
                                             ``fertilizer``) or the alias's species.
``topping``                                  flag ``topping``
``priming``                                  ``(lo, hi)`` -> ``priming_lo/hi`` (once per day)
``tillage``     ``till``                     flag ``tillage``; ``depth_cm`` -> ``till_depth_cm`` [cm]
                                             (scalar), ``implement`` -> ``till_implement`` [code],
                                             ``operation`` -> ``till_operation`` (1 primary,
                                             2 secondary, 3 tertiary; 0 not given) (same)
``residue``                                  ``amount_kg_ha`` -> ``residue_kg_ha`` [kg dry matter
                                             ha-1] (scalar), ``n_kg_ha`` -> ``residue_n_kg_ha``
                                             [kg N ha-1] (add); ``depth_cm`` -> ``residue_depth_cm``,
                                             ``incorp_pct`` -> ``residue_incorp_pct`` [%] (same)
==============  ===========================  ==========================================================

Fertilizer N keeps its species (NO3-N, NH4-N, urea-N, organic N, and N whose form the source did
not give), its application depth and its method; :attr:`EventTable.fert_kg_ha` is the total N of
the day (the combined field of the first event table, kept as a read-only accessor). Depth 0 with
method 0 means "not given"; a consumer that needs a placement checks ``fert_method != 0``.

Categorical payloads (fertilizer method, tillage implement) are *source-qualified codes*:
``1000 * vocabulary + native code`` with the vocabularies of :data:`CODE_VOCABULARIES`
(``rzwqm2``: the integer codes of ``rzwqm.dat``; ``dssat``: the number of a DSSAT ``APnnn`` /
``TInnn`` code). No cross-model mapping is implied: a process that interprets them states which
vocabulary it reads. :func:`source_code` and :func:`decode_code` convert; a payload may give the
code as an int or as ``"<vocabulary>:<code>"`` (``"dssat:AP002"``, ``"rzwqm2:1"``).

One day holds at most one placement per kind (``same`` rule): two fertilizer records of one day
with different depths or methods raise, as two tillage operations with different implements do
(a per-day event axis is plan item L1). Priming harvests the organ ranks ``lo <= rank < hi``
(0-based, see :func:`agrijax.core.organs.prime`); days without priming carry ``-1, -1``, an
empty range, so ``prime(q, ev.priming_lo, ev.priming_hi)`` can be called unconditionally. The
table carries no ``n_crop`` axis: rotations reuse the crop slot (section 3.5).
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pandas as pd
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.state import Forcing, field

__all__ = [
    "CODE_VOCABULARIES",
    "CSV_IGNORED",
    "EVENT_ALIASES",
    "EVENT_KINDS",
    "EVENT_REGISTRY",
    "FERT_SPECIES",
    "NO_PRIMING",
    "EventKind",
    "EventTable",
    "Payload",
    "decode_code",
    "source_code",
]

NO_PRIMING = -1
"""Rank bound stored on days without priming (``[-1, -1)`` is empty)."""

FERT_SPECIES = ("no3", "nh4", "urea", "org", "unspecified")
"""Fertilizer N forms, each kept in its own ``fert_<species>_kg_ha`` leaf [kg N ha-1]."""

CODE_VOCABULARIES: dict[str, int] = {"rzwqm2": 1, "dssat": 2}
"""Vocabulary of a categorical payload code -> its thousands digit (see module docstring)."""

_CODE_BASE = 1000
_CODE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*([A-Za-z]*)\s*(\d+)\s*$")


def source_code(vocabulary: str, native: int | str) -> int:
    """Source-qualified code of ``native`` (an int or a ``APnnn``/``TInnn``-style string)."""
    if vocabulary not in CODE_VOCABULARIES:
        raise ValueError(f"unknown code vocabulary {vocabulary!r}; known {list(CODE_VOCABULARIES)}")
    n = int(re.sub(r"^[A-Za-z]+", "", native)) if isinstance(native, str) else int(native)
    if not 0 < n < _CODE_BASE:
        raise ValueError(f"native code {native!r} outside 1..{_CODE_BASE - 1}")
    return CODE_VOCABULARIES[vocabulary] * _CODE_BASE + n


def decode_code(code: int) -> tuple[str, int] | None:
    """``(vocabulary, native)`` of a source-qualified code; ``None`` for 0 (not given)."""
    c = int(code)
    if c == 0:
        return None
    voc, n = divmod(c, _CODE_BASE)
    names = {v: k for k, v in CODE_VOCABULARIES.items()}
    if voc not in names or n == 0:
        raise ValueError(f"{code} is not a source-qualified code")
    return names[voc], n


def _as_code(v: Any) -> int:
    if isinstance(v, str):
        m = _CODE_RE.match(v)
        if m is None:
            raise ValueError(f"code {v!r} is not '<vocabulary>:<code>'")
        return source_code(m.group(1).lower(), int(m.group(3)))
    c = int(v)
    decode_code(c)  # validates
    return c


# ------------------------------------------------------------------------------ kinds registry

Rule = Literal["add", "same", "flag", "once"]


@dataclasses.dataclass(frozen=True)
class Payload:
    """One payload key of an event kind: the ``[T]`` leaf it fills, its unit and same-day rule."""

    key: str
    leaf: str
    unit: str
    rule: Rule
    description: str
    dtype: Literal["float", "int", "code", "bool"] = "float"
    """``code``: a source-qualified code (int leaf; payload given as int or ``"voc:code"``)."""
    fill: float = 0.0
    """Value on days without the event."""


@dataclasses.dataclass(frozen=True)
class EventKind:
    """An event kind: name, aliases (each may bind a bare number to a payload key) and payload."""

    name: str
    payload: tuple[Payload, ...]
    scalar_key: str | None = None
    """Payload key a bare number goes to (``None``: a bare number is ignored, e.g. flags)."""
    aliases: Mapping[str, str | None] = dataclasses.field(default_factory=dict)
    """Alias -> payload key its bare number goes to (``None``: the kind's ``scalar_key``)."""

    def key(self, k: str) -> Payload:
        for p in self.payload:
            if p.key == k:
                return p
        raise ValueError(f"{self.name} event has no payload key {k!r}; known {[p.key for p in self.payload]}")


def _fert_payload() -> tuple[Payload, ...]:
    names = {
        "no3": "nitrate N",
        "nh4": "ammonium N",
        "urea": "urea N",
        "org": "organic N",
        "unspecified": "N of a form the source does not give",
    }
    amounts = tuple(
        Payload(s, f"fert_{s}_kg_ha", "kg ha-1", "add", f"fertilizer {names[s]} applied [kg N ha-1]")
        for s in FERT_SPECIES
    )
    return (
        *amounts,
        Payload(
            "depth_cm",
            "fert_depth_cm",
            "cm",
            "same",
            "fertilizer application depth (0: surface or not given)",
        ),
        Payload(
            "method",
            "fert_method",
            "-",
            "same",
            "fertilizer method, source-qualified code (0: not given)",
            "code",
        ),
    )


EVENT_REGISTRY: dict[str, EventKind] = {
    k.name: k
    for k in (
        EventKind(
            "sow",
            (Payload("flag", "sow", "-", "flag", "sowing day flag", "bool"),),
            aliases={"planting": None},
        ),
        EventKind("harvest", (Payload("flag", "harvest", "-", "flag", "harvest day flag", "bool"),)),
        EventKind(
            "irrigation",
            (Payload("amount_cm", "irrig_cm", "cm", "add", "irrigation amount"),),
            scalar_key="amount_cm",
            aliases={"irrig": None},
        ),
        EventKind(
            "fertilizer",
            _fert_payload(),
            scalar_key="unspecified",
            aliases={"fert": None, **{f"fertilizer_{s}": s for s in ("no3", "nh4", "urea", "org")}},
        ),
        EventKind(
            "topping",
            (Payload("flag", "topping", "-", "flag", "topping day flag (stops organ appearance)", "bool"),),
        ),
        EventKind(
            "priming",
            (
                Payload(
                    "lo",
                    "priming_lo",
                    "-",
                    "once",
                    "first organ rank harvested (-1: no priming)",
                    "int",
                    NO_PRIMING,
                ),
                Payload(
                    "hi",
                    "priming_hi",
                    "-",
                    "once",
                    "one past the last organ rank harvested (-1: none)",
                    "int",
                    NO_PRIMING,
                ),
            ),
        ),
        EventKind(
            "tillage",
            (
                Payload("flag", "tillage", "-", "flag", "tillage day flag", "bool"),
                Payload("depth_cm", "till_depth_cm", "cm", "same", "tillage depth"),
                Payload(
                    "implement",
                    "till_implement",
                    "-",
                    "same",
                    "tillage implement, source-qualified code",
                    "code",
                ),
                Payload(
                    "operation",
                    "till_operation",
                    "-",
                    "same",
                    "1 primary, 2 secondary, 3 tertiary (0: not given)",
                    "int",
                ),
            ),
            scalar_key="depth_cm",
            aliases={"till": None},
        ),
        EventKind(
            "residue",
            (
                Payload(
                    "amount_kg_ha",
                    "residue_kg_ha",
                    "kg ha-1",
                    "add",
                    "residue / organic amendment applied [kg dry matter ha-1]",
                ),
                Payload(
                    "n_kg_ha", "residue_n_kg_ha", "kg ha-1", "add", "N in the applied residue [kg N ha-1]"
                ),
                Payload("depth_cm", "residue_depth_cm", "cm", "same", "residue incorporation depth"),
                Payload(
                    "incorp_pct", "residue_incorp_pct", "%", "same", "percentage of the residue incorporated"
                ),
            ),
            scalar_key="amount_kg_ha",
        ),
    )
}
"""Event kinds by canonical name (module docstring)."""

EVENT_KINDS = tuple(EVENT_REGISTRY)
"""Canonical event kinds, in the order :meth:`EventTable.to_records` emits same-day events."""

EVENT_ALIASES: dict[str, str] = {a: k.name for k in EVENT_REGISTRY.values() for a in k.aliases}
"""Alternative spellings (``io.catpa`` event names) mapped to canonical kinds."""

CSV_IGNORED = ("pesticide",)
"""``events.csv`` kinds with no model effect yet; :meth:`EventTable.from_csv` drops them by default."""

_LEAVES: dict[str, Payload] = {p.leaf: p for k in EVENT_REGISTRY.values() for p in k.payload}


def _days(dates: pd.DatetimeIndex | Sequence[Any]) -> np.ndarray:
    """Forcing dates as ``datetime64[D]`` (times of day dropped)."""
    return np.asarray([_day(d) for d in dates], dtype="datetime64[D]")


def _day(date: Any) -> np.datetime64:
    """One date (string, Timestamp, datetime64, date) as ``datetime64[D]``."""
    return np.datetime64(pd.Timestamp(date).to_datetime64(), "D")


def _canonical(kind: str) -> tuple[EventKind, str | None]:
    """Kind of a record name and the payload key a bare number of that name goes to."""
    k = str(kind).strip().lower()
    if k in EVENT_REGISTRY:
        ek = EVENT_REGISTRY[k]
        return ek, ek.scalar_key
    if k in EVENT_ALIASES:
        ek = EVENT_REGISTRY[EVENT_ALIASES[k]]
        key = ek.aliases[k]
        return ek, key if key is not None else ek.scalar_key
    raise ValueError(
        f"unknown event kind {kind!r}; known {list(EVENT_KINDS)} and aliases {list(EVENT_ALIASES)}"
    )


def _payload(ek: EventKind, key: str | None, value: Any) -> dict[str, Any]:
    """Record value -> ``{payload key: value}``."""
    if ek.name == "priming":
        r_lo, r_hi = (int(v) for v in value)
        if not 0 <= r_lo <= r_hi:
            raise ValueError(f"priming ranks must satisfy 0 <= lo <= hi, got ({r_lo}, {r_hi})")
        return {"lo": r_lo, "hi": r_hi}
    if isinstance(value, Mapping):
        out = dict(value)
        for k in out:
            ek.key(k)
        return out
    return {key: float(value)} if key is not None else {}


def _np_dtype(p: Payload) -> Any:
    return {"float": np.float64, "int": np.int32, "code": np.int32, "bool": bool}[p.dtype]


class EventTable(Forcing):
    """Per-day management arrays, every leaf of shape ``[T]`` (time axis of the forcing)."""

    sow: Array = field(unit="-", description="sowing day flag", dims="T")
    harvest: Array = field(unit="-", description="harvest day flag", dims="T")
    irrig_cm: Array = field(unit="cm", description="irrigation amount", dims="T")
    topping: Array = field(unit="-", description="topping day flag (stops organ appearance)", dims="T")
    priming_lo: Array = field(unit="-", description="first organ rank harvested (-1: no priming)", dims="T")
    priming_hi: Array = field(
        unit="-", description="one past the last organ rank harvested (-1: none)", dims="T"
    )
    fert_no3_kg_ha: Array = field(
        unit="kg ha-1", description="fertilizer nitrate N applied [kg N ha-1]", dims="T"
    )
    fert_nh4_kg_ha: Array = field(
        unit="kg ha-1", description="fertilizer ammonium N applied [kg N ha-1]", dims="T"
    )
    fert_urea_kg_ha: Array = field(
        unit="kg ha-1", description="fertilizer urea N applied [kg N ha-1]", dims="T"
    )
    fert_org_kg_ha: Array = field(
        unit="kg ha-1", description="fertilizer organic N applied [kg N ha-1]", dims="T"
    )
    fert_unspecified_kg_ha: Array = field(
        unit="kg ha-1", description="fertilizer N of a form the source does not give [kg N ha-1]", dims="T"
    )
    fert_depth_cm: Array = field(
        unit="cm", description="fertilizer application depth (0: surface or not given)", dims="T"
    )
    fert_method: Array = field(
        unit="-", description="fertilizer method, source-qualified code (0: not given)", dims="T"
    )
    tillage: Array = field(unit="-", description="tillage day flag", dims="T")
    till_depth_cm: Array = field(unit="cm", description="tillage depth", dims="T")
    till_implement: Array = field(unit="-", description="tillage implement, source-qualified code", dims="T")
    till_operation: Array = field(
        unit="-", description="tillage operation: 1 primary, 2 secondary, 3 tertiary (0: not given)", dims="T"
    )
    residue_kg_ha: Array = field(
        unit="kg ha-1", description="residue / organic amendment applied [kg dry matter ha-1]", dims="T"
    )
    residue_n_kg_ha: Array = field(
        unit="kg ha-1", description="N in the applied residue [kg N ha-1]", dims="T"
    )
    residue_depth_cm: Array = field(unit="cm", description="residue incorporation depth", dims="T")
    residue_incorp_pct: Array = field(
        unit="%", description="percentage of the residue incorporated", dims="T"
    )

    @property
    def fert_kg_ha(self) -> Array:
        """Total fertilizer N of the day [kg N ha-1]: the sum of the ``fert_<species>_kg_ha`` leaves.

        The combined field of the first event table, kept as a read-only accessor.
        """
        return (
            self.fert_no3_kg_ha
            + self.fert_nh4_kg_ha
            + self.fert_urea_kg_ha
            + self.fert_org_kg_ha
            + self.fert_unspecified_kg_ha
        )

    @classmethod
    def empty(cls, n_days: int) -> EventTable:
        """A table of ``n_days`` days with no event."""
        return cls._from_numpy(
            {leaf: np.full(n_days, p.fill, dtype=_np_dtype(p)) for leaf, p in _LEAVES.items()}
        )

    @classmethod
    def _from_numpy(cls, arrays: Mapping[str, np.ndarray]) -> EventTable:
        return cls(**{k: jnp.asarray(v) for k, v in arrays.items()})

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

        ``kind`` is a canonical kind or alias and ``value`` a number, a priming ``(lo, hi)`` or
        a payload dict (module docstring); an unknown kind raises ``ValueError`` unless listed in
        ``ignore``. Same-day amounts add up; same-day records that disagree on a ``same``
        attribute (depth, method, implement) raise, as do two primings on one day. A record
        dated outside ``dates`` raises, or is dropped with ``outside="drop"``. ``dates`` is the
        forcing's daily ``DatetimeIndex`` (times of day are ignored).
        """
        idx = _days(dates)
        pos = {d: t for t, d in enumerate(idx)}
        if len(pos) != len(idx):
            raise ValueError("dates must be unique")
        n = len(idx)
        arr = {leaf: np.full(n, p.fill, dtype=_np_dtype(p)) for leaf, p in _LEAVES.items()}
        is_set = {leaf: np.zeros(n, dtype=bool) for leaf, p in _LEAVES.items() if p.rule in ("same", "once")}
        skip = {str(k).strip().lower() for k in ignore}

        for date, kind, value in records:
            if str(kind).strip().lower() in skip:
                continue
            ek, key = _canonical(kind)
            t = pos.get(_day(date), -1)
            if t < 0:
                if outside == "drop":
                    continue
                raise ValueError(f"{kind} event on {pd.Timestamp(date).date()} is outside the forcing dates")
            payload = _payload(ek, key, value)
            if ek.name == "priming" and is_set["priming_lo"][t]:
                raise ValueError(f"two primings on {idx[t]}")
            for p in ek.payload:
                if p.rule == "flag":
                    arr[p.leaf][t] = True
                    continue
                if p.key not in payload or payload[p.key] is None:
                    continue
                v = _as_code(payload[p.key]) if p.dtype == "code" else payload[p.key]
                if p.rule == "add":
                    arr[p.leaf][t] += float(v)
                elif p.rule == "once":
                    arr[p.leaf][t] = v
                    is_set[p.leaf][t] = True
                else:  # same
                    v = int(v) if p.dtype in ("int", "code") else float(v)
                    if is_set[p.leaf][t] and arr[p.leaf][t] != v:
                        raise ValueError(
                            f"two {ek.name} events on {idx[t]} with different {p.key} "
                            f"({arr[p.leaf][t]} and {v}); one placement per kind and day"
                        )
                    arr[p.leaf][t] = v
                    is_set[p.leaf][t] = True
        return cls._from_numpy(arr)

    @classmethod
    def from_frame(
        cls,
        df: pd.DataFrame,
        dates: pd.DatetimeIndex | Sequence[Any],
        *,
        ignore: Iterable[str] = CSV_IGNORED,
        outside: Literal["raise", "drop"] = "drop",
        source: str = "events table",
    ) -> EventTable:
        """Expand an event frame with columns :data:`agrijax.io.catpa.EVENT_COLUMNS`.

        The frame is what :func:`agrijax.io.catpa.build_events` returns and ``events.csv``
        holds; rows are converted by :func:`agrijax.io.rzwqm.events.frame_records` (units,
        ``detail`` keys, RZWQM2 method / implement names -> codes) and expanded with
        :meth:`from_records`.
        """
        from agrijax.io.rzwqm.events import frame_records

        return cls.from_records(frame_records(df, source=source), dates, outside=outside, ignore=ignore)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        dates: pd.DatetimeIndex | Sequence[Any],
        *,
        ignore: Iterable[str] = CSV_IGNORED,
        outside: Literal["raise", "drop"] = "drop",
    ) -> EventTable:
        """Load an ``events.csv`` written by :func:`agrijax.io.catpa.write_events`.

        Uses the columns ``date``, ``event``, ``value``, ``unit`` and ``detail``
        (:data:`agrijax.io.catpa.EVENT_COLUMNS`). Kinds in ``ignore`` (pesticide by default: no
        process consumes it yet) are dropped, as are events outside ``dates`` (the csv covers
        the whole scenario, a run may cover a sub-window). Irrigation in ``mm`` is converted to
        cm. Fertilizer rows keep their species (``fertilizer_no3`` ...), method and depth;
        tillage rows their depth, implement and operation. A priming row gives its ranks in
        ``detail`` as ``rank_lo=<int>;rank_hi=<int>``.
        """
        from agrijax.io.catpa import load_events

        return cls.from_frame(load_events(path), dates, ignore=ignore, outside=outside, source=str(path))

    def to_records(self, dates: pd.DatetimeIndex | Sequence[Any]) -> list[tuple[Any, str, Any]]:
        """Inverse of :meth:`from_records` in canonical kinds.

        Flags get value ``1.0``, irrigation its amount in cm, priming ``(lo, hi)``; fertilizer,
        tillage and residue a payload dict of their non-default keys.
        ``from_records(table.to_records(dates), dates)`` reproduces the table.
        """
        idx = _days(dates)
        if len(idx) != self.n_days:
            raise ValueError(f"{len(idx)} dates for a {self.n_days}-day table")
        a = {leaf: np.asarray(getattr(self, leaf)) for leaf in _LEAVES}
        out: list[tuple[Any, str, Any]] = []
        for t in range(self.n_days):
            day = pd.Timestamp(idx[t])
            for ek in EVENT_REGISTRY.values():
                flags = [p for p in ek.payload if p.rule == "flag"]
                vals = {
                    p.key: a[p.leaf][t] for p in ek.payload if p.rule != "flag" and a[p.leaf][t] != p.fill
                }
                if flags and not any(a[p.leaf][t] for p in flags):
                    continue
                if not flags and not vals:
                    continue
                if ek.name == "priming":
                    out.append((day, "priming", (int(a["priming_lo"][t]), int(a["priming_hi"][t]))))
                elif ek.name == "irrigation":
                    out.append((day, "irrigation", float(a["irrig_cm"][t])))
                elif all(p.rule == "flag" for p in ek.payload):
                    out.append((day, ek.name, 1.0))
                else:
                    out.append(
                        (
                            day,
                            ek.name,
                            {k: (int(v) if v.dtype.kind in "iu" else float(v)) for k, v in vals.items()},
                        )
                    )
        return out

    def day(self, t: ArrayLike) -> EventTable:
        """The events of day ``t`` (a Python int or a traced index): the same class with scalar leaves.

        Equivalent to the slice ``lax.scan`` hands a process through ``forcing_t.events``.
        """
        return jtu.tree_map(lambda x: x[t], self)
