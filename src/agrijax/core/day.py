"""The day as reference-declared phases with process-level order (plan 19 A1; decision Q1).

An assembly declares its day as data::

    DAY = Day(
        ref="rzwqm2-4.6",
        phases=(
            Phase("management", ("events.apply",)),
            Phase("physcl", ("pet.sw_daily", "soil_water.day")),
            Phase("plant", ("crops.maize.rootwu", "crops.maize.phenology", "crops.maize.growth",
                            "crops.maize.publish_uptake")),
            Phase("ledger", ("ledger.close",)),
        ),
        lags=(Lag("soil_water.day", "iface.root_uptake.maize", evidence="..."),),
    )
    model = DAY.compile({"events.apply": apply_events, "soil_water.day": bound_richards, ...})

* Entries are ``slot.process`` names. One module contributes as many entries as the reference
  calls it (DSSAT calls SOIL and SPAM once in RATE and once in INTEGR); the same function may
  appear under several entries.
* Within a phase, an entry may read what an **earlier** entry of the same phase wrote. That is
  the reference behaviour (DSSAT SPAM reads WATBAL's same-pass ``SWDELTS``), so no purity rule
  applies by default. ``Phase(..., discipline="start_of_day")`` is an opt-in check for
  PCSE-style assemblies: no entry of that phase may read a path written by an earlier entry of
  the same phase.
* **Lags are declared, never implicit.** A read that comes before another module's write of the
  same path sees yesterday's value; :meth:`Day.compile` requires the set of such reads
  (:meth:`Day.lagged_reads`) to equal the declared :class:`Lag` table exactly, in both
  directions. The module of an entry is its name without the last component, so a module's
  later entry updating the state its earlier entry read (DSSAT ``soil.watbal_rate`` reading the
  ``SW`` that ``soil.watbal_integr`` writes) is that module's own carried state, not a lag. The
  remaining stale reads are true state carried across days or initial conditions
  (:meth:`Day.carried_reads`); ``Model.stale_reads()`` is exactly the union of the two.
* A read that must see the start-of-day value although an earlier entry overwrote it goes
  through an explicit ``prev.*`` copy written by a framework :func:`snapshot` entry.

Compiling a ``Day`` gives the existing :class:`~agrijax.core.model.Model` (a flat, unrolled list
of processes, renamed to their entry names). Phases add names, checks and tests; they add no
runtime structure and no cost.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agrijax.core.model import Model, OutputFn, _overlap
from agrijax.core.process import Process, process
from agrijax.core.state import get_path, set_path

__all__ = [
    "DISCIPLINES",
    "SNAPSHOT_SOURCE",
    "Day",
    "DayError",
    "DayLagError",
    "DayOrderError",
    "Lag",
    "Phase",
    "snapshot",
]

Discipline = Literal["start_of_day"]
#: ``Process.source`` of the entries built by :func:`snapshot`
SNAPSHOT_SOURCE = "plan 19 A1 snapshot"
DISCIPLINES: tuple[str, ...] = ("start_of_day",)


class DayError(ValueError):
    """A :class:`Day` declaration that is inconsistent or does not match its processes."""


class DayOrderError(DayError):
    """An entry reads a same-phase write inside a ``discipline="start_of_day"`` phase."""


class DayLagError(DayError):
    """The lagged reads of the compiled day differ from the declared :class:`Lag` table."""


@dataclass(frozen=True)
class Lag:
    """A declared one-day lag: entry ``reader`` reads ``path`` before the entry that writes it.

    ``path`` is spelled as the reader declares it in its ``reads``. ``evidence`` cites where the
    reference model has the same lag (file and line, or dump). Only ``days = 1`` is expressible
    by order alone; longer lags need explicit history state.
    """

    reader: str
    path: str
    days: int = 1
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.days != 1:
            raise DayError(f"lag {self.reader} <- {self.path}: only days=1 is supported, got {self.days}")
        if not self.evidence.strip():
            raise DayError(
                f"lag {self.reader} <- {self.path}: give the evidence (reference file:line or dump)"
            )

    @property
    def pair(self) -> tuple[str, str]:
        return (self.reader, self.path)


@dataclass(frozen=True)
class Phase:
    """A named, ordered group of entries (``slot.process`` names).

    ``discipline=None`` (the default, and what faithful reference assemblies use) lets an entry
    read what earlier entries of the phase wrote. ``discipline="start_of_day"`` forbids it.
    """

    name: str
    entries: tuple[str, ...]
    discipline: Discipline | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.name or not self.name.replace("_", "").isalnum():
            raise DayError(f"invalid phase name {self.name!r}")
        if self.discipline is not None and self.discipline not in DISCIPLINES:
            raise DayError(
                f"phase {self.name!r}: unknown discipline {self.discipline!r} (known: {DISCIPLINES})"
            )
        for e in self.entries:
            if not e or e.startswith(".") or e.endswith(".") or ".." in e or " " in e:
                raise DayError(f"phase {self.name!r}: invalid entry name {e!r}")


@dataclass(frozen=True)
class Day:
    """A reference model's day: ordered :class:`Phase` objects and the declared :class:`Lag` table."""

    ref: str
    phases: tuple[Phase, ...]
    lags: tuple[Lag, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "lags", tuple(self.lags))
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise DayError(f"duplicate phase names: {sorted({n for n in names if names.count(n) > 1})}")
        entries = self.entries
        if len(set(entries)) != len(entries):
            raise DayError(f"entries listed twice: {sorted({e for e in entries if entries.count(e) > 1})}")
        pairs = [lag.pair for lag in self.lags]
        if len(set(pairs)) != len(pairs):
            raise DayError(f"lags declared twice: {sorted({p for p in pairs if pairs.count(p) > 1})}")
        unknown = sorted({lag.reader for lag in self.lags} - set(entries))
        if unknown:
            raise DayError(f"lags name readers that are not entries of the day: {unknown}")

    # ------------------------------------------------------------------ introspection
    @property
    def entries(self) -> tuple[str, ...]:
        """Every entry in day order."""
        return tuple(e for p in self.phases for e in p.entries)

    def phase_of(self, entry: str) -> str:
        """Name of the phase that holds ``entry``."""
        for p in self.phases:
            if entry in p.entries:
                return p.name
        raise KeyError(entry)

    @staticmethod
    def module_of(entry: str) -> str:
        """The module of an entry: its name without the last component (``crops.maize.growth`` ->
        ``crops.maize``); an undotted name is its own module."""
        return entry.rsplit(".", 1)[0] if "." in entry else entry

    def lagged_reads(self, model: Model) -> list[tuple[str, str]]:
        """Lagged reads of ``model`` with this day's module grouping. The reads of a
        :func:`snapshot` entry are start-of-day copies by construction and never lags."""
        snaps = {p.name for p in model.processes if p.source == SNAPSHOT_SOURCE}
        return [(r, p) for r, p in model.lagged_reads(self.module_of) if r not in snaps]

    def carried_reads(self, model: Model) -> list[tuple[str, str]]:
        """Carried-state reads of ``model`` (with this day's module grouping, and the reads of
        :func:`snapshot` entries): every stale read that is not a lag."""
        lagged = set(self.lagged_reads(model))
        return [pr for pr in model.stale_reads() if pr not in lagged]

    def declared_lags(self) -> set[tuple[str, str]]:
        """``{(reader, path)}`` of the declared lags."""
        return {lag.pair for lag in self.lags}

    # ------------------------------------------------------------------ checks
    def order_violations(self, model: Model) -> list[tuple[str, str, str, str]]:
        """``(phase, writer, reader, path)`` same-phase reads inside ``start_of_day`` phases."""
        by_name = {p.name: p for p in model.processes}
        out: list[tuple[str, str, str, str]] = []
        for ph in self.phases:
            if ph.discipline != "start_of_day":
                continue
            for i, reader in enumerate(ph.entries):
                for r in by_name[reader].reads:
                    for writer in ph.entries[:i]:
                        if any(_overlap(r, w) for w in by_name[writer].writes):
                            out.append((ph.name, writer, reader, r))
        return out

    def check(self, model: Model) -> None:
        """Raise unless ``model`` follows this day: same entries in the same order, no forbidden
        same-phase read, and lagged reads equal to the declared lags."""
        if model.names != self.entries:
            raise DayError(
                f"model order {list(model.names)} differs from the day's entries {list(self.entries)}"
            )
        bad = self.order_violations(model)
        if bad:
            lines = "; ".join(f"[{ph}] {rd} reads {path} written earlier by {wr}" for ph, wr, rd, path in bad)
            raise DayOrderError(f"same-phase reads in a start_of_day phase: {lines}")
        found = set(self.lagged_reads(model))
        declared = self.declared_lags()
        undeclared = sorted(found - declared)
        unused = sorted(declared - found)
        if undeclared or unused:
            msg = []
            if undeclared:
                msg.append(
                    "undeclared lags (reader runs before another entry that writes the path): "
                    + ", ".join(f"{r} <- {p} (written by {list(model.writers(p))})" for r, p in undeclared)
                )
            if unused:
                msg.append(
                    "declared lags that the order does not produce: "
                    + ", ".join(f"{r} <- {p}" for r, p in unused)
                )
            raise DayLagError("; ".join(msg))

    # ------------------------------------------------------------------ compile
    def compile(
        self,
        processes: Mapping[str, Process | Callable[..., Any]] | Iterable[Process],
        *,
        state_spec: type | None = None,
        outputs: Sequence[str] | OutputFn | None = None,
        name: str = "",
        check: bool = True,
    ) -> Model:
        """Build the :class:`~agrijax.core.model.Model` of this day.

        ``processes`` maps every entry name to its process (or is an iterable of processes whose
        names are the entry names, e.g. the output of :func:`agrijax.core.ports.bind` with
        ``name=``). Each process is renamed to its entry, so one function may serve several
        entries. Missing or surplus entries raise :class:`DayError`; with ``check=True`` (the
        default) :meth:`check` runs on the result.
        """
        if isinstance(processes, Mapping):
            table: dict[str, Process] = {str(k): Model._as_process(v) for k, v in processes.items()}
        else:
            table = {}
            for p in processes:
                if not isinstance(p, Process):
                    raise TypeError(f"an iterable of processes must hold Process objects, got {p!r}")
                if p.name in table:
                    raise DayError(f"process name {p.name!r} given twice")
                table[p.name] = p
        missing = [e for e in self.entries if e not in table]
        surplus = sorted(set(table) - set(self.entries))
        if missing or surplus:
            raise DayError(f"entries without a process: {missing}; processes not in the day: {surplus}")
        procs = [
            table[e] if table[e].name == e else dataclasses.replace(table[e], name=e) for e in self.entries
        ]
        model = Model(state_spec, procs, outputs=outputs, name=name or f"day@{self.ref}", day=self)
        if check:
            self.check(model)
        return model


def snapshot(name: str, copies: Mapping[str, str]) -> Process:
    """A framework entry that copies ``{src: dst}`` state paths, e.g. ``{"soil_water.theta":
    "prev.soil_water.theta"}``, so that a later entry can read a start-of-day value after an
    earlier entry overwrote ``src``. Reads the sources, writes the destinations; not registered.
    """
    if not copies:
        raise DayError("snapshot needs at least one (src, dst) pair")
    pairs = tuple((str(s), str(d)) for s, d in copies.items())
    for s, d in pairs:
        if _overlap(s, d):
            raise DayError(f"snapshot source {s!r} and destination {d!r} overlap")

    def _snapshot(state: Any, params: Any, forcing_t: Any) -> Any:
        """Copy the start-of-day values of the source paths into their ``prev.*`` destinations.

        Source: plan 19 A1 (framework entry, no reference equation).
        """
        for s, d in pairs:  # static loop over the declared pairs, unrolled at trace time
            state = set_path(state, d, get_path(state, s))
        return state

    return process(
        _snapshot,
        reads=tuple(s for s, _ in pairs),
        writes=tuple(d for _, d in pairs),
        name=name,
        register=False,
        source=SNAPSHOT_SOURCE,
    )
