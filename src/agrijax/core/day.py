"""The day as reference-declared phases with process-level order.

An assembly declares its day as data::

    DAY = Day(
        ref="rzwqm2-4.6",
        contract_slot="maize",  # the contract's owners and phased writes (a registered reference)
        phases=(
            Phase("management", ("events.apply",)),
            Phase("physcl", ("pet.sw_daily", "soil_water.day")),
            Phase("plant", ("water_supply.maize.rootwu", "crops.maize.phenology", "crops.maize.growth",
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
  same path sees yesterday's value. The :class:`Lag` table of a ``Day`` is the set of lags the
  coupling contract **allows** (lags hang on the contract's ports, so swapping one
  implementation for another with a smaller read set changes no ``Day``).
  :meth:`Day.check` requires every lagged read of the compiled model (:meth:`Day.lagged_reads`)
  to be covered by an allowed lag (same reader; the allowed path equal to the read or a dotted
  prefix of it) and **reports** the allowed lags the implementation does not use
  (:class:`LagReport`) instead of rejecting them; ``exact_lags=True`` restores the two-way
  equality for a day that must reproduce a reference order exactly. An allowed lag must stay a
  lag: when an entry of another module writes the lag's path **before** the reader while the
  path's producer writes it after the reader (an event reset at the start of the day, say), the
  reader would see today's value on the days the early entry writes and yesterday's otherwise,
  and the order analysis would no longer see the lag. :meth:`Day.hidden_lags` lists such reads and
  :meth:`Day.check` rejects them (a season initialisation that wrote the root and canopy
  records at the start of the day would hide the P2 and P6 lags). The module of
  an entry is its name without the last component, so a module's later entry updating the state
  its earlier entry read (DSSAT ``soil.watbal_rate`` reading the ``SW`` that
  ``soil.watbal_integr`` writes) is that module's own carried state, not a lag. The
  remaining stale reads are true state carried across days or initial conditions
  (:meth:`Day.carried_reads`). A module resets its own carried state on an event day itself
  (ROOTWU's ``TSS`` at a harvest is ROOTWU's own entry's write): no entry writes another
  module's state, so there is no cross-module reset to declare. ``Model.stale_reads()``
  is exactly the union of the lagged and the carried reads.
* **One owning module per path, extra writes by declared execution phase.** No two modules write
  overlapping paths, and no entry writes a path under another
  module of the day (the module's own state); for the shared ``iface.*`` records the contract's
  port writer table (``Day(owners=...)``, :func:`agrijax.iface.contract.port_owners`) names the
  owner of each field, and a write of another module there is rejected too
  (:meth:`Day.owner_problems`). When the owner writes a path that other modules read from more
  than one entry (ROOTWU's uptake and its end of season both write ``trwup``), the **first**
  writing entry of the day is the producer (it computes the day's value, undeclared) and every
  later write is declared as a :class:`PhasedWrite`: entry, path, execution phase
  (:data:`WRITE_PHASES`) and a one-line meaning. A ``season_end`` or ``reset`` write runs after
  every same-day consumer of the path (an entry of another module that reads it after the
  producer: the crop, a daily output or diagnostic entry, the water ledger); a consumer before the
  producer (an allowed lag) sees yesterday's end-of-day value, i.e. the reset value the morning
  after. Readers in the owner's own module are not consumers: the module orders its own entries
  (a report entry of ROOTWU after its season end would see the reset value by the module's own
  choice). A ``daily`` extra write of a path other modules read is rejected until a reference case
  needs it (:meth:`Day.write_phase_problems`). A declaration covers its path and the paths below
  it only (a declared ``trwup`` reset does not excuse a write of the whole record).
* **Outputs see the day's values.** A model's per-day outputs are read at the end of the day,
  after every reset. When the model performs a declared ``season_end``/``reset`` write, its outputs
  must be declared (a sequence of paths, or a callable with ``output_reads``; the full state is
  rejected), and no declared output read may overlap a reset path unless it is listed in
  ``end_of_day_reads`` (a deliberate end-of-day value, such as a reference model's exit state).
  The day's value of a reset path is recorded by an entry before the reset (a :func:`snapshot`).
* **What the checks trust** (declarations, not verified): a phased write's ``when`` and
  ``meaning`` (a reset mislabelled ``daily`` and placed before the first consumer passes the order
  rule; in a contract day the contract's table fixes the labels); ``Process.source ==
  SNAPSHOT_SOURCE`` marks a :func:`snapshot` entry (the ``prev.*`` reservation and the lag analysis
  rely on it); ``output_reads`` of a callable output; ``end_of_day_reads`` may name a whole declared
  output record (every path below it is then exempt). The static contract check
  (``agrijax.iface.contract.dssat_aj013_problems``, AJ013) sees port fields and the shared-state
  rows only: a module-state path that another module reads but that has no shared-state row and no
  declaration is found by :meth:`Day.check` on the compiled reads, which is the stricter gate. A
  day of a reference whose contract is registered (:func:`register_contract`, once per reference)
  is a contract day (``contract_slot``) unless it is a test fixture marked ``bare=True``.
* A read that must see the start-of-day value although an earlier entry overwrote it goes
  through an explicit ``prev.*`` copy written by a framework :func:`snapshot` entry.

Compiling a ``Day`` gives the existing :class:`~agrijax.core.model.Model` (a flat, unrolled list
of processes, renamed to their entry names). Phases add names, checks and tests; they add no
runtime structure and no cost.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from agrijax.core.model import Model, OutputFn, _overlap
from agrijax.core.process import Process, process
from agrijax.core.state import get_path, set_path

__all__ = [
    "DISCIPLINES",
    "RESET_PHASES",
    "SHARED_NAMESPACES",
    "SNAPSHOT_SOURCE",
    "WRITE_PHASES",
    "Day",
    "DayError",
    "DayLagError",
    "DayOrderError",
    "DayWriteError",
    "Lag",
    "LagReport",
    "Phase",
    "PhasedWrite",
    "register_contract",
    "snapshot",
]

Discipline = Literal["start_of_day"]
#: ``Process.source`` of the entries built by :func:`snapshot`
SNAPSHOT_SOURCE = "snapshot entry (framework primitive, no reference equation)"
DISCIPLINES: tuple[str, ...] = ("start_of_day",)
#: execution phases of a declared extra write (:class:`PhasedWrite`): ``daily``, a second write of
#: the day's value (a RATE and an INTEGR call); ``season_end``, the owner ends its season on an event
#: day; ``reset``, the owner re-initialises for the next day
WRITE_PHASES: tuple[str, ...] = ("daily", "season_end", "reset")
#: the execution phases that must run after the producer and after every same-day consumer
RESET_PHASES: tuple[str, ...] = ("season_end", "reset")
#: namespaces that hold shared records (``iface``) and start-of-day copies (``prev``): an entry
#: named under them is not a module whose state other entries must leave alone
SHARED_NAMESPACES: tuple[str, ...] = ("iface", "prev")

_log = logging.getLogger(__name__)


class DayError(ValueError):
    """A :class:`Day` declaration that is inconsistent or does not match its processes."""


class DayOrderError(DayError):
    """An entry reads a same-phase write inside a ``discipline="start_of_day"`` phase."""


class DayLagError(DayError):
    """A lagged read of the compiled day that the :class:`Lag` table does not allow (or, with
    ``exact_lags=True``, an allowed lag the order does not produce)."""


class DayWriteError(DayError):
    """An owner's extra write of a path that other modules read which is not declared as a
    :class:`PhasedWrite`, or a declared ``season_end``/``reset`` write scheduled before the path's
    producer, before a same-day consumer or before an end-of-day output of the path."""


#: ``{ref: slot -> (owners, phased writes)}``: the contract tables of a reference day
#: (:func:`register_contract`; ``agrijax.iface.contract`` registers its days when imported)
_CONTRACTS: dict[str, Callable[[str], tuple[tuple[tuple[str, str], ...], tuple[PhasedWrite, ...]]]] = {}


def register_contract(
    ref: str, tables: Callable[[str], tuple[tuple[tuple[str, str], ...], tuple[PhasedWrite, ...]]]
) -> None:
    """Register the contract tables of the reference day ``ref``: ``tables(slot)`` gives the port
    owners and the phased writes a :class:`Day` with ``contract_slot=slot`` must use. A reference
    is registered once: a second registration of ``ref`` (a replaced table after import) raises
    :class:`DayError`."""
    if ref in _CONTRACTS:
        raise DayError(f"contract of {ref!r} is already registered; a contract table is not replaced")
    _CONTRACTS[ref] = tables


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

    def covers(self, reader: str, path: str) -> bool:
        """``True`` when this lag allows ``reader``'s lagged read of ``path``: the same reader, and
        ``path`` equal to the lag's path or below it (``iface.root.maize`` allows
        ``iface.root.maize.rlv``)."""
        return reader == self.reader and (path == self.path or path.startswith(self.path + "."))


@dataclass(frozen=True)
class PhasedWrite:
    """A declared extra write of an owner module: entry ``entry`` writes ``path`` (or below it) in
    the execution phase ``when`` (:data:`WRITE_PHASES`), besides the module's producer entry of
    that path. ``meaning`` says in one line what the write does and why at that point of the day
    (``"TRWUP = 0 at the end of a harvest day; the next morning's WUF reads the 0"``). ``path``
    may carry ``{slot}`` in a contract table; a :class:`Day` takes it filled in.
    """

    entry: str
    path: str
    when: str
    meaning: str

    def __post_init__(self) -> None:
        if self.when not in WRITE_PHASES:
            raise DayError(
                f"phased write {self.entry} -> {self.path}: unknown execution phase {self.when!r} "
                f"(known: {WRITE_PHASES})"
            )
        if not self.meaning.strip():
            raise DayError(f"phased write {self.entry} -> {self.path}: give its meaning (one line)")

    @property
    def is_reset(self) -> bool:
        """``True`` for a ``season_end`` or ``reset`` write (it must follow every same-day consumer)."""
        return self.when in RESET_PHASES

    def covers(self, entry: str, path: str) -> bool:
        """``True`` when this declaration is ``entry``'s and ``path`` (a write) is its path or below
        it: a declared ``...trwup`` does not cover a write of the whole record."""
        return entry == self.entry and (path == self.path or path.startswith(self.path + "."))


@dataclass(frozen=True)
class LagReport:
    """What :meth:`Day.check` found: the lagged reads of the model (each covered by an allowed
    lag) and the allowed lags that no read of this implementation uses."""

    used: tuple[tuple[str, str], ...]
    unused: tuple[Lag, ...]

    @property
    def unused_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(lag.pair for lag in self.unused)


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
    """A reference model's day: ordered :class:`Phase` objects, the declared :class:`Lag` table and
    the owners' declared extra writes (:class:`PhasedWrite`)."""

    ref: str
    phases: tuple[Phase, ...]
    lags: tuple[Lag, ...] = ()
    phased_writes: tuple[PhasedWrite, ...] = ()
    #: ``((path, module), ...)``: the owning module of shared paths (the contract's port writers)
    owners: tuple[tuple[str, str], ...] = ()
    #: the crop slot of a contract day: its owners and phased writes are the contract's
    #: (:func:`register_contract` for :attr:`ref`), filled in when empty and checked otherwise
    contract_slot: str = ""
    #: a day of a registered reference that is not the contract's day (a test fixture: part of the
    #: day, stand-ins): without it such a day must name its ``contract_slot``
    bare: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        object.__setattr__(self, "lags", tuple(self.lags))
        object.__setattr__(self, "phased_writes", tuple(self.phased_writes))
        object.__setattr__(self, "owners", tuple((str(a), str(b)) for a, b in self.owners))
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
        keys = [(w.entry, w.path) for w in self.phased_writes]
        if len(set(keys)) != len(keys):
            raise DayError(f"phased writes declared twice: {sorted({k for k in keys if keys.count(k) > 1})}")
        unknown = sorted({w.entry for w in self.phased_writes} - set(entries))
        if unknown:
            raise DayError(f"phased writes name entries that are not entries of the day: {unknown}")
        modules = {self.module_of(e) for e in entries}
        for path, owner in self.owners:
            if not path or not owner:
                raise DayError(f"owner ({path!r}, {owner!r}): give a path and a module")

        def own_port(w: PhasedWrite) -> bool:
            m = self.module_of(w.entry)
            return any(o == m and (w.path == q or w.path.startswith(q + ".")) for q, o in self.owners)

        for w in self.phased_writes:
            other = sorted(
                m
                for m in modules - set(SHARED_NAMESPACES)
                if m != self.module_of(w.entry) and _overlap(m, w.path) and not own_port(w)
            )
            if other:
                raise DayError(
                    f"phased write {w.entry} -> {w.path}: the path is the state of another module {other}; "
                    "a module declares writes of its own state and its out ports only"
                )
        if self.bare and self.contract_slot:
            raise DayError(f"day {self.ref!r}: a bare day has no contract_slot")
        if self.ref in _CONTRACTS and not self.contract_slot and not self.bare:
            raise DayError(
                f"day {self.ref!r}: the reference has a registered contract, so its day is a contract "
                "day (contract_slot=...); a test fixture that is part of the day passes bare=True"
            )
        if self.contract_slot:
            if self.ref not in _CONTRACTS:
                raise DayError(
                    f"contract day {self.ref!r}: no contract registered (import agrijax.iface.contract)"
                )
            owners, phased = _CONTRACTS[self.ref](self.contract_slot)
            rows = set(entries)
            phased = tuple(w for w in phased if w.entry in rows)
            if not self.owners:
                object.__setattr__(self, "owners", tuple(owners))
            if not self.phased_writes:
                object.__setattr__(self, "phased_writes", phased)
            if set(self.owners) != set(owners):
                raise DayError(
                    f"contract day {self.ref!r}: the owners differ from the contract's port writers"
                )
            if set(self.phased_writes) != set(phased):
                raise DayError(f"contract day {self.ref!r}: the phased writes differ from the contract's")

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
        """``{(reader, path)}`` of the declared (allowed) lags."""
        return {lag.pair for lag in self.lags}

    def lag_report(self, model: Model) -> LagReport:
        """Split the lags of ``model``: its lagged reads, and the allowed lags none of them uses.

        Raises :class:`DayLagError` for a lagged read no allowed lag covers.
        """
        found = self.lagged_reads(model)
        undeclared = sorted({(r, p) for r, p in found if not any(lag.covers(r, p) for lag in self.lags)})
        if undeclared:
            raise DayLagError(
                "undeclared lags (reader runs before another entry that writes the path, and the day "
                "does not allow the lag): "
                + ", ".join(f"{r} <- {p} (written by {list(model.writers(p))})" for r, p in undeclared)
            )
        unused = tuple(lag for lag in self.lags if not any(lag.covers(r, p) for r, p in found))
        return LagReport(used=tuple(dict.fromkeys(found)), unused=unused)

    def hidden_lags(self, model: Model) -> list[tuple[str, str, str, str]]:
        """``(reader, path, early writer, late writer)`` for each read that an allowed lag covers
        but that an entry of another module writes before the reader, while another module's entry
        writes it after the reader. On the early writer's days the reader sees today's value, on
        the other days yesterday's: the lag the contract allows is hidden from the order analysis.
        Only reads covered by a :class:`Lag` of this day are examined."""
        mod = self.module_of
        procs = model.processes
        out: list[tuple[str, str, str, str]] = []
        for i, reader in enumerate(procs):
            for r in reader.reads:
                if not any(lag.covers(reader.name, r) for lag in self.lags):
                    continue
                before = [
                    p.name
                    for p in procs[:i]
                    if mod(p.name) != mod(reader.name) and any(_overlap(r, w) for w in p.writes)
                ]
                after = [
                    p.name
                    for p in procs[i + 1 :]
                    if mod(p.name) != mod(reader.name) and any(_overlap(r, w) for w in p.writes)
                ]
                out.extend((reader.name, r, b, a) for b in before for a in after[:1])
        return out

    def owner_problems(self, model: Model) -> list[str]:
        """Writes of a path owned by another module (``[]``: none): two modules writing
        overlapping paths; a write of a path the :attr:`owners` table (the contract's port writers)
        gives to another module; an entry writing under another module of the day (its state),
        unless the owners table gives that path to the writer (a port that lives under a module's
        namespace, such as DSSAT's P4 ``soil_water.sink_in`` written by ``spam.xtract``); a write
        under ``prev.*`` by an entry that is not a :func:`snapshot`. With declared phased writes or
        owners, an entry with unchecked writes (``*``) is rejected (its writes cannot be
        attributed); in a contract day (:attr:`contract_slot`) a write of an ``iface.*`` path that
        the owners table does not cover is rejected too."""
        mod = self.module_of
        procs = model.processes
        out: list[str] = []
        if self.phased_writes or self.owners:
            out.extend(
                f"{p.name} declares no writes ('*'): a day with declared owner writes needs every "
                "entry's writes"
                for p in procs
                if "*" in p.writes
            )
        # the shared namespaces of records and start-of-day copies are nobody's state
        modules = {mod(p.name) for p in procs} - set(SHARED_NAMESPACES)
        writes = [(p.name, w) for p in procs for w in p.writes if w != "*"]
        snaps = {p.name for p in procs if p.source == SNAPSHOT_SOURCE}

        def owned_by(w: str, m: str) -> bool:
            """``w`` is at or below an owners path of module ``m``."""
            return any(owner == m and (w == path or w.startswith(path + ".")) for path, owner in self.owners)

        for e, w in writes:
            if _overlap("prev", w) and e not in snaps:
                out.append(
                    f"{e} writes {w}: prev.* holds start-of-day copies written by snapshot entries only"
                )
            if not owned_by(w, mod(e)):
                for m in sorted(modules - {mod(e)}):
                    if _overlap(m, w):
                        out.append(
                            f"{e} writes {w}: the state of module {m} (a module writes its own state only)"
                        )
            for path, owner in self.owners:
                if owner != mod(e) and _overlap(path, w):
                    out.append(
                        f"{e} writes {w}: {path} is owned by module {owner} (the contract's port writers)"
                    )
            if (
                self.contract_slot
                and _overlap("iface", w)
                and not any(w == path or w.startswith(path + ".") for path, _ in self.owners)
                and not any(path.startswith(w + ".") for path, _ in self.owners)
            ):
                out.append(f"{e} writes {w}: an iface path that no port of the contract owns")
        for i, (e1, w1) in enumerate(writes):
            for e2, w2 in writes[i + 1 :]:
                if mod(e1) != mod(e2) and _overlap(w1, w2):
                    out.append(f"two modules write one path: {e1} writes {w1}, {e2} writes {w2}")
        return list(dict.fromkeys(out))

    def write_phase_problems(self, model: Model) -> list[str]:
        """Problems of the owners' writes of the paths other modules read (``[]``: none).

        For each read of an entry ``R`` and each other module ``M`` that writes the read path, the
        entries of ``M`` whose writes overlap there, in day order: the first is the producer and
        must be undeclared; every later one must be declared (:class:`PhasedWrite`, covering its
        write); a ``daily`` extra write is rejected; a ``season_end``/``reset`` write must run
        after ``R`` when ``R`` reads after the producer (a same-day consumer; a reader before the
        producer is a lag and sees yesterday's end-of-day value). A declaration whose entry does not
        write its path is unused, not an error (a stand-in entry that writes nothing).

        Outputs: when a declared ``season_end``/``reset`` write is performed (its entry writes its
        path), the model's outputs must declare their reads (:attr:`Model.output_paths`: a path
        list, or ``output_reads`` of a callable) and none may overlap a reset path unless listed in
        :attr:`Model.end_of_day_reads`."""
        mod = self.module_of
        procs = model.processes
        pos = {p.name: i for i, p in enumerate(procs)}
        decl = self.phased_writes
        out: list[str] = []
        for i, reader in enumerate(procs):
            for r in reader.reads:
                by_mod: dict[str, list[tuple[str, str]]] = {}
                for p in procs:
                    if mod(p.name) == mod(reader.name):
                        continue
                    for w in p.writes:
                        if w != "*" and _overlap(r, w):
                            by_mod.setdefault(mod(p.name), []).append((p.name, w))
                for m, pairs in by_mod.items():
                    for _, w in pairs:
                        group = [(n, w2) for n, w2 in pairs if _overlap(w, w2)]
                        names = list(dict.fromkeys(n for n, _ in group))
                        if len(names) < 2:
                            continue
                        producer = names[0]
                        if any(d.covers(n, w2) for d in decl for n, w2 in group if n == producer):
                            out.append(
                                f"{reader.name} reads {r}; the first writer {producer} of module {m} is "
                                "declared as a phased write: the first writer is the undeclared producer"
                            )
                        for n, w2 in group:
                            if n == producer:
                                continue
                            ds = [d for d in decl if d.covers(n, w2)]
                            if not ds:
                                out.append(
                                    f"{reader.name} reads {r}; module {m} writes it from {names}: the "
                                    f"write of {w2} by {n} after the producer {producer} is undeclared "
                                    "(declare its execution phase)"
                                )
                            for d in ds:
                                if d.when == "daily":
                                    # a second write of the day's value (DSSAT RATE, then INTEGR): it
                                    # completes the value before any same-day consumer reads it
                                    if pos[producer] < i < pos[n]:
                                        out.append(
                                            f"the daily write of {w2} by {n} ({d.meaning}) runs after its "
                                            f"same-day consumer {reader.name} (a daily extra write completes "
                                            "the day's value before the first consumer)"
                                        )
                                elif pos[producer] < i and pos[n] < i:
                                    out.append(
                                        f"the {d.when} write of {w2} by {n} ({d.meaning}) runs before its "
                                        f"same-day consumer {reader.name}"
                                    )
        resets = [
            (d, w)
            for d in decl
            if d.is_reset and d.entry in pos
            for w in procs[pos[d.entry]].writes
            if d.covers(d.entry, w)
        ]
        if resets:
            paths = model.output_paths
            if paths is None:
                kind = "the full state" if model.output_kind == "state" else "a callable without output_reads"
                out.append(
                    f"the outputs are {kind} while {resets[0][0].entry} performs a {resets[0][0].when} "
                    "write: declare the paths the outputs read (output_reads=, end_of_day_reads=)"
                )
            else:
                for o in paths:
                    if any(_overlap(o, x) for x in model.end_of_day_reads):
                        continue
                    for d, w in resets:
                        if _overlap(o, w):
                            out.append(
                                f"the end-of-day output {o} shows the {d.when} value of {d.entry} "
                                f"({d.meaning}), not the day's: record the day's value with an entry before "
                                "the reset, or list the path in end_of_day_reads"
                            )
        return list(dict.fromkeys(out))

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

    def check(self, model: Model, *, exact_lags: bool = False) -> LagReport:
        """Raise unless ``model`` follows this day: same entries in the same order, no forbidden
        same-phase read, no allowed lag hidden by an earlier writer (:meth:`hidden_lags`), every
        lagged read allowed by the :class:`Lag` table, one owning module per path
        (:meth:`owner_problems`) and the owners' extra writes declared, with a reset after the
        day's consumers and outputs that show the day's values (:meth:`write_phase_problems`;
        both :class:`DayWriteError`). Returns the
        :class:`LagReport` (the allowed lags this implementation does not use are reported, not
        rejected); with ``exact_lags=True`` an unused allowed lag raises :class:`DayLagError`."""
        if model.names != self.entries:
            raise DayError(
                f"model order {list(model.names)} differs from the day's entries {list(self.entries)}"
            )
        bad = self.order_violations(model)
        if bad:
            lines = "; ".join(f"[{ph}] {rd} reads {path} written earlier by {wr}" for ph, wr, rd, path in bad)
            raise DayOrderError(f"same-phase reads in a start_of_day phase: {lines}")
        hidden = self.hidden_lags(model)
        if hidden:
            raise DayLagError(
                "allowed lags hidden by an earlier writer (the reader sees today's value on the days "
                "the earlier entry writes, yesterday's otherwise; write the path after the reader): "
                + ", ".join(f"{r} <- {p} (written before by {b}, after by {a})" for r, p, b, a in hidden)
            )
        report = self.lag_report(model)
        writes = self.owner_problems(model) + self.write_phase_problems(model)
        if writes:
            raise DayWriteError("owner writes: " + "; ".join(writes))
        if exact_lags and report.unused:
            raise DayLagError(
                "declared lags that the order does not produce: "
                + ", ".join(f"{r} <- {p}" for r, p in report.unused_pairs)
            )
        if report.unused:
            _log.info(
                "day %s: allowed lags not used by this implementation: %s",
                self.ref,
                ", ".join(f"{r} <- {p}" for r, p in report.unused_pairs),
            )
        return report

    # ------------------------------------------------------------------ compile
    def compile(
        self,
        processes: Mapping[str, Process | Callable[..., Any]] | Iterable[Process],
        *,
        state_spec: type | None = None,
        outputs: Sequence[str] | OutputFn | None = None,
        name: str = "",
        check: bool = True,
        exact_lags: bool = False,
        output_reads: Sequence[str] | None = None,
        end_of_day_reads: Sequence[str] = (),
    ) -> Model:
        """Build the :class:`~agrijax.core.model.Model` of this day.

        ``processes`` maps every entry name to its process (or is an iterable of processes whose
        names are the entry names, e.g. the output of :func:`agrijax.core.ports.bind` with
        ``name=``). Each process is renamed to its entry, so one function may serve several
        entries. Missing or surplus entries raise :class:`DayError`; with ``check=True`` (the
        default) :meth:`check` runs on the result (``exact_lags`` is passed on). The allowed lags
        the result does not use are in :meth:`lag_report`. ``output_reads`` declares the state paths
        a callable ``outputs`` reads and ``end_of_day_reads`` the output paths that deliberately show
        the end-of-day value (:class:`~agrijax.core.model.Model`; checked by
        :meth:`write_phase_problems`).
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
        model = Model(
            state_spec,
            procs,
            outputs=outputs,
            name=name or f"day@{self.ref}",
            day=self,
            output_reads=output_reads,
            end_of_day_reads=end_of_day_reads,
        )
        if check:
            self.check(model, exact_lags=exact_lags)
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

        Source: framework entry (no reference equation).
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
