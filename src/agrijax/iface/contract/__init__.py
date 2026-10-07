"""The port table of the coupling contract (P1-P11) in machine-readable form.

Each :class:`PortSpec` names a port's global path, its record class, the fields the contract
fixes (unit string in :func:`agrijax.core.units.parse_unit` syntax, trailing dims from
:data:`agrijax.core.dims.DIMS`, grid), who writes and who reads it, its time semantics and its
default. :func:`record_problems` compares a spec with the field metadata of its record class
(unit strings identical, not just the same dimension; dims identical; grid identical);
``tests/unit/test_iface.py`` runs it on every port.

Time semantics (:data:`TIME_SEMANTICS`)
---------------------------------------
* ``same_day``: every reader runs after the writer in the day's order and sees today's value.
* ``lag1``: the reader runs before the writer and sees yesterday's value; the reader must be
  covered by a :class:`~agrijax.core.day.Lag` of the day. The allowed lags hang on the ports
  (:attr:`PortSpec.lags`); :func:`allowed_lags` turns them into the ``Lag`` table of a
  :class:`~agrijax.core.day.Day`, and :meth:`~agrijax.core.day.Day.check` accepts an
  implementation that uses a subset of them, so an implementation that reads less can be swapped
  in without changing the day.
* ``mixed``: same-day for most readers, one-day lag for the readers listed in ``lags``.
* ``forcing``: an input series sliced per day by the runtime, not state.
* ``cumulative``: a running total over the run (the ledger).

Fields listed in :attr:`PortSpec.pending` belong to the contract but are not yet in the record
class; each says what will add it.

A port that is a field of a slot's own state (P7, ``soil_water.theta``) has no record class of
its own: :attr:`PortSpec.field_of` names the holder class by dotted path, and this module does
not import it (``agrijax.iface`` imports nothing from ``agrijax.processes``);
``tests/unit/test_iface.py`` resolves the path and checks the fields against it.

The day (:data:`DAY_TABLE`)
---------------------------
The contract's day in the RZWQM2 4.6 order: every entry, its phase, the registry key of its
default implementation, the ports it produces and whether that implementation exists. Every
producer named in :data:`PORTS` is an entry of this table (``tests/unit/test_iface.py``).
Per-crop entries carry ``{slot}``. An entry's module (for the lag check) is its name
without the last component: ROOTWU is ``water_supply.<slot>.rootwu`` and the P10 producer
``n_supply.<slot>.replay``, each a module of its own outside ``crops.<slot>``.

Stages: the planned ports and rows (``post_m3``)
-------------------------------------------------
The first stage, ``m3``, is the day in the RZWQM2 4.6 order. The ports P12 and up, and the rows of
:data:`POST_M3_DAY_TABLE`, belong to the modules planned after it (soil nitrogen, residue and
organic matter, soil heat and ice, tile drainage, phosphorus). They carry ``stage="post_m3"``;
everything that builds the M3 day (:func:`day_entries`, :func:`allowed_lags`) takes the ``m3``
stage only by default, so the
planned contract changes no M3 behaviour. A planned port says what runs before its module exists
(:attr:`PortSpec.off`, the "faithful off" behaviour). A planned row names the entry it follows
(:attr:`DayEntry.after`) and the M3 rows it replaces when its module is on
(:attr:`DayEntry.supersedes`); :func:`day_table` merges the stages into one order.
:data:`POST_M3_ENTRY_PORTS` lists the planned ports that an M3 entry's post-M3 variant writes.

Owners and execution phases (:data:`PHASED_WRITES`)
---------------------------------------------------
A field has exactly one owning module (no module writes another module's state; a module resets its
own state from the event table). The owner's producer entry writes the day's value; when the owner
also writes the field from another entry of the day, that write is declared in :data:`PHASED_WRITES`
with its execution phase (:data:`agrijax.core.day.WRITE_PHASES`) and a one-line meaning. A
``season_end`` or ``reset`` write runs after the producer and after every consumer of the field in
the day's order, so on a harvest day the crop, the daily output and diagnostics and the ledger see
the day's value and the next morning starts from the reset value. :func:`phased_writes` gives the
declarations of a slot for :class:`~agrijax.core.day.Day`, whose check enforces the same rule on the
compiled processes' reads.

:func:`contract_problems` checks the tables mechanically (rules AJ012-AJ016: every port has a
producer and an off behaviour, one writer module per field with its extra writes declared by
execution phase, bounded records, lags consistent with the order, stages and namespaces
consistent); :func:`day_status_problems` (AJ017) compares the rows'
status with the process registry both ways, :func:`registry_slot_problems` (AJ018) the registry
slot of a key with the directory of its code, and :func:`variant_problems` (AJ019) the variant
label with the closed vocabulary :data:`VARIANTS`. :mod:`agrijax.iface.render` renders the tables
as markdown, so documents quote the code instead of copying it.

Layout
------
The tables live in submodules and are re-exported here, so every name is imported from
``agrijax.iface.contract``: :mod:`.spec` (the record types :class:`FieldSpec`, :class:`AllowedLag`,
:class:`PortSpec`), :mod:`.ports` (:data:`PORTS`), :mod:`.rzwqm_day` (:data:`DAY_TABLE`,
:data:`POST_M3_DAY_TABLE`, :data:`PHASED_WRITES`, :data:`SHARED_STATE`), :mod:`.dssat_day` (the
DSSAT-CSM v4.8.6.0 day and its ports) and :mod:`.variants` (AJ018, AJ019). The functions that read
the day tables and the checks (AJ012-AJ017) stay in this module, so they read the tables through
this module's namespace.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

from agrijax.core.day import Day, Lag, PhasedWrite, register_contract
from agrijax.core.ports import NAMESPACES, RESERVED_NAMESPACES

from .dssat_day import (
    DSSAT_DAY_LAGS,
    DSSAT_DAY_PHASES,
    DSSAT_DAY_REF,
    DSSAT_DAY_TABLE,
    DSSAT_ENTRY_STATUS,
    DSSAT_PHASED_WRITES,
    DSSAT_PORTS,
    DSSAT_SHARED_STATE,
    RefDayEntry,
    RefLag,
)
from .ports import PORTS
from .rzwqm_day import (
    DAY_PHASES,
    DAY_TABLE,
    ENTRY_STATUS,
    EXTENDED_PHASES,
    PHASED_WRITES,
    POST_M3_DAY_TABLE,
    POST_M3_ENTRY_PORTS,
    SHARED_STATE,
    DayEntry,
)
from .rzwqm_day import Status as Status
from .spec import _M3, STAGES, TIME_SEMANTICS, AllowedLag, FieldSpec, PortSpec
from .spec import Stage as Stage
from .spec import Time as Time
from .variants import ADAPTER_SLOTS as ADAPTER_SLOTS
from .variants import LEGACY_VARIANTS as LEGACY_VARIANTS
from .variants import VARIANT_MEANING as VARIANT_MEANING
from .variants import VARIANTS, registry_slot_problems, variant_problems
from .variants import variant_kind as variant_kind

__all__ = [
    "DAY_PHASES",
    "DAY_TABLE",
    "DSSAT_DAY_LAGS",
    "DSSAT_DAY_PHASES",
    "DSSAT_DAY_REF",
    "DSSAT_DAY_TABLE",
    "DSSAT_ENTRY_STATUS",
    "DSSAT_PHASED_WRITES",
    "DSSAT_PORTS",
    "DSSAT_SHARED_STATE",
    "ENTRY_STATUS",
    "EXTENDED_PHASES",
    "MAX_PORT_CONSUMERS",
    "MAX_PORT_FIELDS",
    "PHASED_WRITES",
    "PORTS",
    "POST_M3_DAY_TABLE",
    "POST_M3_ENTRY_PORTS",
    "SHARED_STATE",
    "STAGES",
    "TIME_SEMANTICS",
    "VARIANTS",
    "AllowedLag",
    "DayEntry",
    "FieldSpec",
    "PortSpec",
    "RefDayEntry",
    "RefLag",
    "allowed_lags",
    "contract_problems",
    "day_entries",
    "day_entry",
    "day_status_problems",
    "day_table",
    "dssat_aj013_problems",
    "dssat_allowed_lags",
    "dssat_day_entries",
    "dssat_day_problems",
    "dssat_phased_writes",
    "dssat_port_owners",
    "dssat_port_spec",
    "entry_ports_out",
    "path_consumers",
    "phased_writes",
    "port_owners",
    "port_spec",
    "record_problems",
    "registry_slot_problems",
    "variant_problems",
]


def phased_writes(slot: str, stages: Iterable[str] = _M3) -> tuple[PhasedWrite, ...]:
    """The declarations of :data:`PHASED_WRITES` for crop slot ``slot`` whose entry is a row of the
    day of ``stages``. Pass it as ``Day(phased_writes=...)``."""
    rows = {r.entry for r in day_table(stages)}
    return tuple(
        PhasedWrite(w.entry.format(slot=slot), w.path.format(slot=slot), w.when, w.meaning)
        for w in PHASED_WRITES
        if w.entry in rows
    )


def path_consumers(path: str) -> tuple[str, ...]:
    """The consumer entries of a port field or module-state path (``{slot}`` form): the port's
    :meth:`PortSpec.field_consumers` for a port field (every field for a whole port path),
    :data:`SHARED_STATE` for module state."""
    out: list[str] = []
    for spec in PORTS.values():
        if spec.kind != "state":
            continue
        for n in spec.field_names:
            f = f"{spec.path}.{n}"
            if f == path or f.startswith(path + ".") or path.startswith(f + "."):
                out.extend(spec.field_consumers(n))
    for sp, (_, readers) in SHARED_STATE.items():
        if sp == path or sp.startswith(path + ".") or path.startswith(sp + "."):
            out.extend(readers)
    return tuple(dict.fromkeys(out))


def _field_path(spec: PortSpec, name: str) -> str:
    """The global path of field ``name`` of ``spec`` (``{slot}`` form): a port that is one field of a
    module's state (``field_of``, the field named as the path's last component) is its path."""
    if spec.field_of and len(spec.field_names) == 1 and spec.path.rsplit(".", 1)[-1] == name:
        return spec.path
    return f"{spec.path}.{name}"


def _owners(
    specs: Iterable[PortSpec], slot: str, rows: set[str] | None = None
) -> tuple[tuple[str, str], ...]:
    """Owners of the fields of ``specs``: the module of the field's writers (those that are
    ``rows`` of the day when any is, else all of them)."""
    out: list[tuple[str, str]] = []
    for spec in specs:
        if spec.kind != "state":
            continue
        for n, all_ws in spec.field_writers().items():
            ws = [e for e in all_ws if rows is not None and e in rows] or list(all_ws)
            mods = {Day.module_of(e) for e in ws}
            if len(mods) == 1:  # a field of two modules is an AJ013 problem, not an owner
                out.append((_field_path(spec, n).format(slot=slot), next(iter(mods)).format(slot=slot)))
    return tuple(dict.fromkeys(out))


def port_owners(slot: str) -> tuple[tuple[str, str], ...]:
    """``((field path, owning module), ...)`` of every state port of :data:`PORTS` for crop slot
    ``slot`` (the M3 ports, the planned ones and the cumulative ledger): the module of each field's
    writer entries (:meth:`PortSpec.field_writers`). The M3 day's ``Day(contract_slot=...)`` uses
    it, so the compiled day and AJ013 check ownership against the same writer table."""
    return _owners(PORTS.values(), slot)


def day_table(stages: Iterable[str] = _M3) -> tuple[DayEntry, ...]:
    """The contract's rows of ``stages`` in day order. With ``post_m3``, each planned row goes after
    its :attr:`DayEntry.after` entry (or first in its phase for ``"^<phase>"``), and the rows it
    supersedes are left out. ``day_table()`` is :data:`DAY_TABLE`."""
    st = tuple(stages)
    unknown = sorted(set(st) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stages {unknown} (stages: {STAGES})")
    rows = list(DAY_TABLE) if "m3" in st else []
    if "post_m3" in st:
        for e in POST_M3_DAY_TABLE:
            if e.after.startswith("^"):
                phase = e.after[1:]
                order = EXTENDED_PHASES.index(phase)
                i = next(
                    (k for k, r in enumerate(rows) if EXTENDED_PHASES.index(r.phase) >= order), len(rows)
                )
            else:
                at = [k for k, r in enumerate(rows) if r.entry == e.after]
                if not at:
                    raise ValueError(f"planned row {e.entry!r}: no entry {e.after!r} to follow")
                i = at[0] + 1
            rows.insert(i, e)
        gone = {name for e in rows for name in e.supersedes}
        rows = [r for r in rows if r.entry not in gone]
    ranks = [EXTENDED_PHASES.index(r.phase) for r in rows]
    if ranks != sorted(ranks):
        raise ValueError(
            "the merged day is not in phase order: " + ", ".join(f"{r.entry}[{r.phase}]" for r in rows)
        )
    return tuple(rows)


def day_entries(slot: str, stages: Iterable[str] = _M3) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((phase, entry names), ...)`` of the contract's day for crop slot ``slot``, in order
    (the M3 day by default; ``stages=STAGES`` with the planned rows)."""
    rows = day_table(stages)
    phases = DAY_PHASES if tuple(stages) == _M3 else EXTENDED_PHASES
    return tuple(
        (ph, tuple(e.name(slot) for e in rows if e.phase == ph))
        for ph in phases
        if any(e.phase == ph for e in rows)
    )


def day_entry(name: str, slot: str, stages: Iterable[str] = _M3) -> DayEntry:
    """The :class:`DayEntry` whose name for crop slot ``slot`` is ``name``."""
    for e in day_table(stages):
        if e.name(slot) == name:
            return e
    raise KeyError(f"no entry {name!r} in the contract's day (slot {slot!r})")


def entry_ports_out(entry: DayEntry) -> tuple[str, ...]:
    """The ports ``entry`` writes: its row's, plus :data:`POST_M3_ENTRY_PORTS` of a post-M3 variant."""
    return (*entry.ports_out, *POST_M3_ENTRY_PORTS.get(entry.entry, ()))


def port_spec(port_id: str) -> PortSpec:
    """The :class:`PortSpec` ``P1`` .. ``P11``."""
    try:
        return PORTS[port_id]
    except KeyError:
        raise KeyError(f"no port {port_id!r} (ports: {list(PORTS)})") from None


def record_problems(spec: PortSpec, record: type | None = None) -> list[str]:
    """Differences between ``spec`` and its record class's field metadata: a missing field, a
    unit string that differs (even with the same dimension), different dims or grid; and a
    ``pending`` field that is already in the record (the spec is out of date).

    ``record`` replaces :attr:`PortSpec.record`: a test passes the class that
    :attr:`PortSpec.field_of` names (P7's holder is the soil-water slot's own state, which this
    module does not import)."""
    rec = spec.record if record is None else record
    if rec is None:
        return []
    meta = {f.name: f.metadata for f in dataclasses.fields(rec)}
    out: list[str] = []
    for name, fs in spec.fields:
        m = meta.get(name)
        if m is None:
            out.append(f"{spec.id}.{name}: not a field of {rec.__name__}")
            continue
        if m.get("unit") != fs.unit:
            out.append(f"{spec.id}.{name}: unit {m.get('unit')!r} in {rec.__name__}, contract {fs.unit!r}")
        if m.get("dims") != fs.dims:
            out.append(f"{spec.id}.{name}: dims {m.get('dims')!r} in {rec.__name__}, contract {fs.dims!r}")
        if fs.grid is not None and m.get("grid") != fs.grid:
            out.append(f"{spec.id}.{name}: grid {m.get('grid')!r} in {rec.__name__}, contract {fs.grid!r}")
    for name, _, _ in spec.pending:
        if name in meta:
            out.append(f"{spec.id}.{name}: listed as pending but already a field of {rec.__name__}")
    return out


def allowed_lags(
    slot: str, ports: tuple[str, ...] | None = None, stages: Iterable[str] = _M3
) -> tuple[Lag, ...]:
    """The :class:`~agrijax.core.day.Lag` table the contract allows for crop slot ``slot``: one lag
    per :class:`AllowedLag` of the given ports (default: every port of ``stages``, the M3 ports by
    default), with ``{slot}`` filled, for the readers that are entries of the ``stages`` day.
    Pass it as ``Day(lags=...)``; an implementation may use any subset of it."""
    st = tuple(stages)
    readers = {e.entry for e in day_table(st)}
    out: list[Lag] = []
    for pid in ports if ports is not None else tuple(p for p, sp in PORTS.items() if sp.stage in st):
        spec = port_spec(pid)
        base = spec.global_path(slot)
        for lag in spec.lags:
            if lag.reader not in readers:
                continue
            path = f"{base}.{lag.field}" if lag.field else base
            out.append(Lag(lag.reader.format(slot=slot), path, evidence=lag.evidence or spec.id))
    return tuple(out)


# ------------------------------------------------------------------------ mechanical checks
#: most fields (with the pending ones) a port record may carry before it must be split (AJ014)
MAX_PORT_FIELDS: int = 12
#: most consumer entries a port may have before it must be split (AJ014)
MAX_PORT_CONSUMERS: int = 6
#: first components of entry names that are not state namespaces: assembly entries without state
#: (``events``, ``weather``) and the M3 modules whose state sits under ``surface`` (``pet``, ``snow``;
#: the entry name is the module, the state path is its binding)
_ENTRY_HEADS: frozenset[str] = frozenset({"events", "weather", "pet", "snow"})


def _first(text: str) -> str:
    return text.split(" ", 1)[0]


def _all_rows() -> tuple[DayEntry, ...]:
    return (*DAY_TABLE, *POST_M3_DAY_TABLE)


def _stage_of(entry: str) -> str | None:
    return next((r.stage for r in _all_rows() if r.entry == entry), None)


def contract_problems() -> list[str]:
    """Problems of the contract tables, each prefixed with its rule id (``[]``: none).

    * AJ012: every state port has a producer that is a row of the day and lists the port in its
      ports (:func:`entry_ports_out`); a planned port says what runs before its module (``off``).
    * AJ013: one writer module per field (fan-in 1; the module of an entry is its name without the
      last component, :meth:`agrijax.core.day.Day.module_of`): two modules never write the same
      field. The owner may write a field from more than one entry (ROOTWU's ``trwup`` and its end of
      season at a harvest) only when every write besides the producer is declared in
      :data:`PHASED_WRITES` with its execution phase; a ``season_end``/``reset`` write runs after
      the producer and after every consumer of the field, so the day's value is still readable on a
      harvest day and the next morning starts from the reset value. A declaration names a row,
      writes a field the port lists for its entry or the entry's own module state, never another
      module's; a row that resets another module's state is a second writer (a module re-initialises
      its own state from the event table instead).
    * AJ014: at most :data:`MAX_PORT_FIELDS` fields and :data:`MAX_PORT_CONSUMERS` consumers.
    * AJ015: in the merged day order, a consumer that runs before a writer of the port is covered
      by an allowed lag, an allowed lag's reader runs before a writer, and the time semantics say
      so (``lag1``: every consumer lagged, ``same_day``: none, ``mixed``: some).
    * AJ016: paths and entries start with a namespace; an M3 row writes and an M3 port is read
      with a lag only within the M3 stage, so the planned contract cannot change the M3 day.
    """
    out: list[str] = []
    rows = _all_rows()
    names = {r.entry for r in rows}
    spaces = {*NAMESPACES, *RESERVED_NAMESPACES}
    for r in rows:
        head = r.entry.split(".", 1)[0]
        if head not in spaces | _ENTRY_HEADS:
            out.append(f"AJ016 row {r.row} {r.entry}: {head!r} is not a namespace")
        for pid in r.ports_out:
            if pid not in PORTS:
                out.append(f"AJ016 row {r.row} {r.entry}: unknown port {pid}")
            elif r.stage == "m3" and PORTS[pid].stage != "m3":
                out.append(
                    f"AJ016 row {r.row} {r.entry}: an m3-stage row writes the planned port {pid} "
                    "(use POST_M3_ENTRY_PORTS)"
                )
        resets = getattr(r, "resets", ())  # a cross-module reset field, if a row carries one
        if resets:
            out.append(
                f"AJ013 row {r.row} {r.entry}: resets another module's state {resets}; the owner "
                "re-initialises its own state from the event table"
            )
    out.extend(_declaration_problems(names))
    for entry, pids in POST_M3_ENTRY_PORTS.items():
        if entry not in names:
            out.append(f"AJ016 POST_M3_ENTRY_PORTS: {entry!r} is not a row")
        out.extend(
            f"AJ016 POST_M3_ENTRY_PORTS {entry}: {p} is not a planned port"
            for p in pids
            if PORTS.get(p) is None or PORTS[p].stage == "m3"
        )
    for pid, spec in PORTS.items():
        head = spec.path.split(".", 1)[0]
        if head not in spaces | {"forcing"}:
            out.append(f"AJ016 {pid} {spec.path}: {head!r} is not a namespace")
        if spec.stage != "m3" and not spec.off.strip():
            out.append(f"AJ012 {pid}: a planned port says what runs before its module exists (off)")
        n_fields = len(spec.field_names)
        if n_fields > MAX_PORT_FIELDS:
            out.append(f"AJ014 {pid}: {n_fields} fields > {MAX_PORT_FIELDS}: split the record")
        if len(spec.consumers) > MAX_PORT_CONSUMERS:
            out.append(
                f"AJ014 {pid}: {len(spec.consumers)} consumers > {MAX_PORT_CONSUMERS}: split the record"
            )
        if spec.kind != "state" or spec.time == "cumulative":
            continue
        if not spec.producers:
            out.append(f"AJ012 {pid}: no producer")
        stray = sorted({n for _, names in spec.writers for n in names} - set(spec.field_names))
        if stray:
            out.append(f"AJ013 {pid}: writers of fields that are not in the port {stray}")
        single: list[str] = []
        for fname, w_entries in spec.field_writers().items():
            w_modules = tuple(dict.fromkeys(Day.module_of(e) for e in w_entries))
            if len(w_modules) > 1:
                out.append(
                    f"AJ013 {pid} {fname}: written by {len(w_entries)} entries {w_entries} of "
                    f"{len(w_modules)} modules {w_modules}"
                )
            else:
                single.append(fname)
        out.extend(_phase_problems(pid, spec, single))
        writers = {e for ws in spec.field_writers().values() for e in ws}
        for w in sorted(writers):
            row = next((r for r in rows if r.entry == w), None)
            if row is None:
                out.append(f"AJ012 {pid}: producer {w!r} is not a row of the day")
            elif pid not in entry_ports_out(row):
                out.append(f"AJ012 {pid}: producer row {row.row} {w} does not list {pid} in its ports")
            elif spec.stage == "m3" and row.stage != "m3" and w not in {e for e, _ in spec.writers}:
                out.append(f"AJ016 {pid}: the m3-stage port's producer {w} is a planned row")
        for lag in spec.lags:
            if spec.stage == "m3" and _stage_of(lag.reader) != "m3":
                out.append(
                    f"AJ016 {pid}: the m3-stage port's allowed lag has the planned reader {lag.reader}"
                )
        out.extend(_order_problems(pid, spec, writers))
    return out


def _declared(entry: str, path: str, decls: Iterable[PhasedWrite] | None = None) -> list[PhasedWrite]:
    """The declarations of ``entry`` that cover the write ``path`` (at or below the declared path)."""
    return [d for d in (PHASED_WRITES if decls is None else decls) if d.covers(entry, path)]


def _m3_orders() -> tuple[list[str], ...]:
    return tuple([r.entry for r in day_table(st)] for st in (_M3, STAGES))


def _reset_order(
    tag: str,
    path: str,
    writers: Iterable[str],
    consumers: Iterable[str],
    decls: Iterable[PhasedWrite] | None = None,
    orders: Iterable[list[str]] | None = None,
) -> list[str]:
    """Order problems of the declared writes of ``path`` in each day order (default: the M3 day and
    the merged day): the first writer is the producer and undeclared; a ``daily`` extra write runs
    before every consumer that reads after the producer; a ``season_end``/``reset`` write runs after
    every consumer."""
    out: list[str] = []
    ds = tuple(PHASED_WRITES if decls is None else decls)
    ws, cs = tuple(writers), tuple(consumers)
    for order in _m3_orders() if orders is None else orders:
        pos = {e: i for i, e in enumerate(order)}
        present = sorted((w for w in ws if w in pos), key=pos.__getitem__)
        if not present:
            continue
        producer = present[0]
        if _declared(producer, path, ds):
            out.append(
                f"{tag}: the first writer {producer} is declared; the first writer is the undeclared producer"
            )
        for e in present[1:]:
            for d in _declared(e, path, ds):
                if d.when == "daily":
                    early = [c for c in cs if c in pos and pos[producer] < pos[c] < pos[e]]
                    if early:
                        out.append(
                            f"{tag}: the daily write of {e} runs after the consumer(s) {early} (a daily "
                            "extra write completes the day's value before the first consumer)"
                        )
                elif d.is_reset:
                    late = [c for c in cs if c in pos and pos[c] > pos[e]]
                    if late:
                        out.append(
                            f"{tag}: the {d.when} write of {e} runs before the consumer(s) {late} "
                            "(a reset follows every consumer of the day)"
                        )
    return list(dict.fromkeys(out))


def _field_phase_problems(
    pid: str,
    spec: PortSpec,
    fields: Iterable[str],
    decls: Iterable[PhasedWrite] | None = None,
    orders: Iterable[list[str]] | None = None,
    table: str = "PHASED_WRITES",
    rows: set[str] | None = None,
) -> list[str]:
    """AJ013 for the fields of a port that have one writer module: one producer per field, the
    other writes declared, and the declared writes ordered (:func:`_reset_order`) against the
    field's consumers (:meth:`PortSpec.field_consumers`). ``rows`` limits the writers to the rows of
    a day (a port shared by two reference days)."""
    out: list[str] = []
    ds = tuple(PHASED_WRITES if decls is None else decls)
    fw = spec.field_writers()
    for fname in fields:
        ws = tuple(e for e in fw.get(fname, ()) if rows is None or e in rows)
        if len(ws) < 2:
            continue
        path = _field_path(spec, fname)
        plain = [e for e in ws if not _declared(e, path, ds)]
        if len(plain) != 1:
            out.append(
                f"AJ013 {pid} {fname}: written by {list(ws)} of one module; one producer and a declared "
                f"execution phase ({table}) for every other write (undeclared: {plain})"
            )
        out.extend(_reset_order(f"AJ013 {pid} {fname}", path, ws, spec.field_consumers(fname), ds, orders))
    return list(dict.fromkeys(out))


def _phase_problems(pid: str, spec: PortSpec, fields: Iterable[str]) -> list[str]:
    """:func:`_field_phase_problems` of the contract's :data:`PHASED_WRITES` (the ``m3`` and merged days)."""
    return _field_phase_problems(pid, spec, fields)


def _declaration_problems(
    rows: set[str],
    decls: Iterable[PhasedWrite] | None = None,
    specs: Iterable[PortSpec] | None = None,
    shared: dict[str, tuple[str, tuple[str, ...]]] | None = None,
    orders: Iterable[list[str]] | None = None,
    table: str = "PHASED_WRITES",
) -> list[str]:
    """AJ013 for a phased-write table (default the contract's :data:`PHASED_WRITES`): each
    declaration names a row and a field that its entry writes, as a port lists it, or a path of the
    entry's own module state with its producer and readers in the shared-state table; the declared
    writes of module state are ordered like those of port fields."""
    out: list[str] = []
    ds = tuple(PHASED_WRITES if decls is None else decls)
    sp = tuple(PORTS.values() if specs is None else specs)
    sh = SHARED_STATE if shared is None else shared
    orders = None if orders is None else tuple(orders)
    seen: set[tuple[str, str]] = set()
    for d in ds:
        tag = f"AJ013 {table} {d.entry} -> {d.path}"
        if (d.entry, d.path) in seen:
            out.append(f"{tag}: declared twice")
        seen.add((d.entry, d.path))
        if d.entry not in rows:
            out.append(f"{tag}: {d.entry!r} is not a row")
            continue
        fields = [
            (spec, n)
            for spec in sp
            if spec.kind == "state"
            for n in spec.field_names
            if _overlaps(_field_path(spec, n), d.path)
        ]
        if fields:
            for spec, n in fields:
                if d.entry not in spec.field_writers().get(n, ()):
                    out.append(f"{tag}: {spec.id} does not list {d.entry} as a writer of {n}")
        elif not d.path.startswith(Day.module_of(d.entry) + "."):
            out.append(
                f"{tag}: neither a port field nor the state of the entry's module "
                f"{Day.module_of(d.entry)!r} (a module declares its own writes only)"
            )
        elif d.path not in sh:
            out.append(f"{tag}: module state without a shared-state row (its producer and readers)")
        else:
            producer, readers = sh[d.path]
            if Day.module_of(producer) != Day.module_of(d.entry):
                out.append(f"{tag}: the producer {producer} is another module's entry")
            out.extend(_reset_order(tag, d.path, (producer, d.entry), readers, ds, orders))
    return out


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def _order_problems(pid: str, spec: PortSpec, writers: set[str]) -> list[str]:
    out: list[str] = []
    for stages in ((_M3,) if spec.stage == "m3" else ()) + (STAGES,):
        order = [r.entry for r in day_table(stages)]
        pos = {e: i for i, e in enumerate(order)}
        present = [w for w in writers if w in pos]
        if not present:
            continue  # every writer superseded in this day: the port is retired there
        first_writer = min(pos[w] for w in present)
        lagged: list[str] = []
        for c in (_first(c) for c in spec.consumers):
            if c not in pos:
                if c not in {r.entry for r in _all_rows()}:
                    out.append(f"AJ015 {pid}: consumer {c!r} is not a row of the day")
                continue
            if pos[c] <= max(pos[w] for w in present) and pos[c] <= first_writer:
                lagged.append(c)
                if not any(lag.reader == c for lag in spec.lags):
                    out.append(f"AJ015 {pid}: {c} reads before the writer {present} without an allowed lag")
        for lag in spec.lags:
            if lag.reader in pos and pos[lag.reader] > first_writer:
                out.append(
                    f"AJ015 {pid}: allowed lag of {lag.reader}, which runs after the writer (a hidden lag)"
                )
        if stages == STAGES or spec.stage == "m3":
            if spec.time == "lag1" and len(lagged) != len([c for c in spec.consumers if _first(c) in pos]):
                out.append(f"AJ015 {pid}: time lag1 but not every consumer reads before the writer")
            if spec.time == "same_day" and lagged:
                out.append(f"AJ015 {pid}: time same_day but {lagged} read before the writer")
    return list(dict.fromkeys(out))


def day_status_problems(is_registered: Callable[[str], bool], stages: Iterable[str] = STAGES) -> list[str]:
    """AJ017: a row's status against the process registry, both ways: a row whose key is registered
    says ``registered``, and a ``registered`` row's key is registered (``kernel``, ``adapter`` and
    ``none`` rows name keys that are not). ``is_registered`` is e.g. ``lambda k: lookup(k) is not
    None`` after importing the process packages."""
    out: list[str] = []
    seen: set[str] = set()
    for r in (*day_table(tuple(stages)), *(POST_M3_DAY_TABLE if "post_m3" in tuple(stages) else ())):
        if not r.key or r.entry in seen:
            continue
        seen.add(r.entry)
        reg = bool(is_registered(r.key))
        if reg and r.status != "registered":
            out.append(f"AJ017 row {r.row} {r.entry}: {r.key} is registered, the row says {r.status!r}")
        if not reg and r.status == "registered":
            out.append(f"AJ017 row {r.row} {r.entry}: the row says registered, {r.key} is not")
    return out


def dssat_day_entries(slot: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((phase, entry names), ...)`` of :data:`DSSAT_DAY_TABLE` for crop slot ``slot``, in order."""
    return tuple(
        (ph, tuple(e.name(slot) for e in DSSAT_DAY_TABLE if e.phase == ph))
        for ph in DSSAT_DAY_PHASES
        if any(e.phase == ph for e in DSSAT_DAY_TABLE)
    )


def dssat_allowed_lags(slot: str) -> tuple[Lag, ...]:
    """The :class:`~agrijax.core.day.Lag` table of the DSSAT-CSM day for crop slot ``slot``."""
    return tuple(
        Lag(lag.reader.format(slot=slot), lag.path.format(slot=slot), evidence=lag.evidence)
        for lag in DSSAT_DAY_LAGS
    )


def _dssat_specs() -> tuple[PortSpec, ...]:
    """The port table of the DSSAT-CSM day: :data:`DSSAT_PORTS` over :data:`PORTS` by id."""
    return tuple({**PORTS, **DSSAT_PORTS}.values())


def _dssat_rows() -> set[str]:
    return {e.entry for e in DSSAT_DAY_TABLE}


def dssat_port_owners(slot: str) -> tuple[tuple[str, str], ...]:
    """:func:`port_owners` of the DSSAT-CSM day: every state port of :data:`DSSAT_PORTS` over
    :data:`PORTS`, a field owned by the module of its writers that are rows of
    :data:`DSSAT_DAY_TABLE` (else of all its writers)."""
    return _owners(_dssat_specs(), slot, _dssat_rows())


def dssat_phased_writes(slot: str) -> tuple[PhasedWrite, ...]:
    """The declarations of :data:`DSSAT_PHASED_WRITES` for crop slot ``slot``."""
    return tuple(
        PhasedWrite(w.entry.format(slot=slot), w.path.format(slot=slot), w.when, w.meaning)
        for w in DSSAT_PHASED_WRITES
    )


def dssat_aj013_problems() -> list[str]:
    """AJ013 on the DSSAT-CSM day: one writer module per field of its port table (writers that are
    rows of the day), one producer and declared extra writes (:data:`DSSAT_PHASED_WRITES`) ordered
    in the DSSAT day, and the declarations themselves (rows, port writers or own module state with
    :data:`DSSAT_SHARED_STATE`)."""
    out: list[str] = []
    rows = _dssat_rows()
    order = [e.entry for e in DSSAT_DAY_TABLE]
    for spec in _dssat_specs():
        if spec.kind != "state" or spec.time == "cumulative":
            continue
        single: list[str] = []
        for fname, all_ws in spec.field_writers().items():
            ws = [e for e in all_ws if e in rows]
            mods = tuple(dict.fromkeys(Day.module_of(e) for e in ws))
            if len(mods) > 1:
                out.append(f"AJ013 DSSAT {spec.id} {fname}: written by {ws} of {len(mods)} modules {mods}")
            else:
                single.append(fname)
        out.extend(
            _field_phase_problems(
                f"DSSAT {spec.id}", spec, single, DSSAT_PHASED_WRITES, (order,), "DSSAT_PHASED_WRITES", rows
            )
        )
    out.extend(
        _declaration_problems(
            rows, DSSAT_PHASED_WRITES, _dssat_specs(), DSSAT_SHARED_STATE, (order,), "DSSAT_PHASED_WRITES"
        )
    )
    return out


def dssat_port_spec(port_id: str) -> PortSpec:
    """The port ``port_id`` as the DSSAT-CSM day uses it: :data:`DSSAT_PORTS`, else :data:`PORTS`."""
    if port_id in DSSAT_PORTS:
        return DSSAT_PORTS[port_id]
    return port_spec(port_id)


def dssat_day_problems(slot: str = "maize") -> list[str]:
    """Consistency of :data:`DSSAT_DAY_TABLE`, :data:`DSSAT_DAY_LAGS` and :data:`DSSAT_PORTS`: known
    phases in day order, unique entries and rows, a known status, an owner and a reference call site
    on every row, a replay row that names who replaces it, output ports that exist in
    :data:`DSSAT_PORTS` or :data:`PORTS`, lag readers that are entries of the day, lag paths inside
    the state namespaces; every DSSAT port's producer and consumer an entry of the day, every
    producer listing the port among its row's outputs, and the record of every DSSAT port matching
    its field specs (:func:`record_problems`)."""
    out: list[str] = []
    ranks = []
    for e in DSSAT_DAY_TABLE:
        if e.phase not in DSSAT_DAY_PHASES:
            out.append(f"{e.row} {e.entry}: unknown phase {e.phase!r}")
            continue
        ranks.append(DSSAT_DAY_PHASES.index(e.phase))
        if e.status not in DSSAT_ENTRY_STATUS:
            out.append(f"{e.row} {e.entry}: unknown status {e.status!r}")
        if not e.owner.strip() or not e.source.strip():
            out.append(f"{e.row} {e.entry}: a row names its owner and its reference call site")
        if e.status == "replay" and e.owner in ("", "day_adapters"):
            out.append(f"{e.row} {e.entry}: a replay row names the module whose process replaces it")
        for pid in e.ports_out:
            if pid not in PORTS and pid not in DSSAT_PORTS:
                out.append(f"{e.row} {e.entry}: unknown port {pid!r}")
    if ranks != sorted(ranks):
        out.append("DSSAT_DAY_TABLE is not in phase order")
    names = [e.name(slot) for e in DSSAT_DAY_TABLE]
    rows = [e.row for e in DSSAT_DAY_TABLE]
    for what, xs in (("entries", names), ("rows", rows)):
        dup = sorted({x for x in xs if xs.count(x) > 1})
        if dup:
            out.append(f"DSSAT_DAY_TABLE: {what} listed twice: {dup}")
    ns = set(NAMESPACES) | set(RESERVED_NAMESPACES)
    for lag in DSSAT_DAY_LAGS:
        if lag.reader.format(slot=slot) not in names:
            out.append(f"lag {lag.reader} <- {lag.path}: the reader is not an entry of the day")
        if lag.path.split(".", 1)[0] not in ns:
            out.append(f"lag {lag.reader} <- {lag.path}: path outside the state namespaces")
        if not lag.evidence.strip():
            out.append(f"lag {lag.reader} <- {lag.path}: no evidence")
    by_name = {e.name(slot): e for e in DSSAT_DAY_TABLE}
    for pid, spec in DSSAT_PORTS.items():
        for text in spec.producers:
            entry = text.split(" ", 1)[0].format(slot=slot)
            row = by_name.get(entry)
            if row is None:
                out.append(f"{pid}: producer {entry!r} is not an entry of the DSSAT day")
            elif pid not in row.ports_out:
                out.append(f"{pid}: producer {entry!r} does not list {pid} among its outputs")
        for c in spec.consumers:
            if c.split(" ", 1)[0].format(slot=slot) not in by_name:
                out.append(f"{pid}: consumer {c!r} is not an entry of the DSSAT day")
        for lag in spec.lags:
            if lag.reader.format(slot=slot) not in by_name:
                out.append(f"{pid}: lag reader {lag.reader!r} is not an entry of the DSSAT day")
        out.extend(record_problems(spec))
    out.extend(dssat_aj013_problems())
    return out


# the contract tables of the two reference days, for Day(contract_slot=...)
register_contract("rzwqm2-4.6", lambda slot: (port_owners(slot), phased_writes(slot)))
register_contract(DSSAT_DAY_REF, lambda slot: (dssat_port_owners(slot), dssat_phased_writes(slot)))
