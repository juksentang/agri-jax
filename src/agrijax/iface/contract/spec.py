"""The record types of the coupling contract's port table: :class:`FieldSpec`, :class:`AllowedLag`
and :class:`PortSpec`, the time semantics and the stages.

Part of :mod:`agrijax.iface.contract` (its docstring explains the tables); import the names from
there.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agrijax.core.dims import parse_dims
from agrijax.core.units import parse_unit

Time = Literal["same_day", "lag1", "mixed", "forcing", "cumulative"]
TIME_SEMANTICS: tuple[str, ...] = ("same_day", "lag1", "mixed", "forcing", "cumulative")
Stage = Literal["m3", "post_m3"]
#: contract stages: ``m3`` is the RZWQM2 4.6 day, ``post_m3`` the planned modules (module docstring)
STAGES: tuple[str, ...] = ("m3", "post_m3")
_M3: tuple[str, ...] = ("m3",)


@dataclass(frozen=True)
class FieldSpec:
    """One field of a port: unit string, trailing dims and (for layered fields) grid."""

    unit: str
    dims: tuple[str, ...]
    grid: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        parse_unit(self.unit)  # raises UnitError
        parse_dims(self.dims)  # raises DimsError


@dataclass(frozen=True)
class AllowedLag:
    """A one-day lag the contract allows on a port: entry ``reader`` (``{slot}`` is replaced by
    the crop slot) reads the port, or its field ``field``, before the writer runs."""

    reader: str
    field: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class PortSpec:
    """One port of the coupling contract (see the module docstring)."""

    id: str
    path: str
    record: type | None
    fields: tuple[tuple[str, FieldSpec], ...]
    producers: tuple[str, ...]
    consumers: tuple[str, ...]
    time: Time
    default: str
    lags: tuple[AllowedLag, ...] = ()
    pending: tuple[tuple[str, FieldSpec, str], ...] = ()
    kind: Literal["state", "forcing"] = "state"
    field_of: str = ""
    #: ``m3`` (in the RZWQM2 4.6 day) or ``post_m3`` (planned; see the module docstring)
    stage: Stage = "m3"
    #: the "faithful off" behaviour: what the consumers see before the producer module exists
    #: (a record constructor, a replay key, a parameter path, or "unbound: ..."); required for a
    #: planned port
    off: str = ""
    #: ``((entry, fields), ...)``: which entry writes which fields (``()`` fields: all). Empty: every
    #: producer (first word of :attr:`producers`) writes the fields named in its parentheses, or all
    writers: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.field_of and self.record is not None:
            raise ValueError(f"{self.id}: give either a record class or field_of, not both")
        if self.stage not in STAGES:
            raise ValueError(f"{self.id}: unknown stage {self.stage!r}")
        if self.stage != "m3" and not self.off.strip():
            raise ValueError(f"{self.id}: a planned port says what runs before its module exists (off=)")
        if self.time not in TIME_SEMANTICS:
            raise ValueError(f"{self.id}: unknown time semantics {self.time!r}")
        if self.time in ("lag1", "mixed") and not self.lags:
            raise ValueError(f"{self.id}: time {self.time!r} needs the allowed lags")
        names = {n for n, _ in self.fields}
        for lag in self.lags:
            if lag.field and lag.field not in names:
                raise ValueError(f"{self.id}: lag on unknown field {lag.field!r}")

    @property
    def field_map(self) -> dict[str, FieldSpec]:
        return dict(self.fields)

    @property
    def field_names(self) -> tuple[str, ...]:
        """The fields in the record and the pending ones."""
        return (*(n for n, _ in self.fields), *(n for n, _, _ in self.pending))

    def field_writers(self) -> dict[str, tuple[str, ...]]:
        """``{field: entries that write it}`` (from :attr:`writers`, else from :attr:`producers`)."""
        pairs: list[tuple[str, tuple[str, ...]]] = list(self.writers)
        if not pairs:
            for text in self.producers:
                entry, _, rest = text.partition(" ")
                named = tuple(
                    n for n in re.findall(r"[a-z_0-9]+", rest.split(")", 1)[0]) if n in self.field_names
                )
                pairs.append((entry, named if rest.startswith("(") and named else ()))
        out: dict[str, list[str]] = {n: [] for n in self.field_names}
        for entry, names in pairs:
            for n in names or self.field_names:
                out.setdefault(n, []).append(entry)
        return {n: tuple(dict.fromkeys(e)) for n, e in out.items()}

    def field_consumers(self, name: str) -> tuple[str, ...]:
        """The consumer entries that read field ``name``: a consumer lists the fields it reads in
        parentheses (``"crops.{slot}.stress (sw, eop, trwup)"``); without a list it reads them all."""
        out: list[str] = []
        for text in self.consumers:
            entry, _, rest = text.partition(" ")
            named = re.findall(r"[a-z_0-9]+", rest.split(")", 1)[0]) if rest.startswith("(") else []
            listed = [n for n in named if n in self.field_names]
            if not listed or name in listed:
                out.append(entry)
        return tuple(out)

    def global_path(self, slot: str | None = None) -> str:
        """The port's path with ``{slot}`` filled (a per-crop port needs ``slot``)."""
        if "{slot}" in self.path:
            if not slot:
                raise ValueError(f"{self.id} ({self.path}) is per crop slot: give slot=")
            return self.path.format(slot=slot)
        return self.path


def _f(unit: str, *dims: str, grid: str | None = None) -> FieldSpec:
    return FieldSpec(unit, tuple(dims), grid)
