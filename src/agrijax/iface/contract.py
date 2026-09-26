"""The port table of the M3 coupling contract (P1-P11) in machine-readable form.

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
  implementation that uses a subset of them (M3 contract, decision 3).
* ``mixed``: same-day for most readers, one-day lag for the readers listed in ``lags``.
* ``forcing``: an input series sliced per day by the runtime, not state.
* ``cumulative``: a running total over the run (the ledger).

Fields listed in :attr:`PortSpec.pending` belong to the contract but are not yet in the record
class; each names the gap that adds it.

A port that is a field of a slot's own state (P7, ``soil_water.theta``) has no record class of
its own: :attr:`PortSpec.field_of` names the holder class by dotted path, and this module does
not import it (``agrijax.iface`` imports nothing from ``agrijax.processes``);
``tests/unit/test_iface.py`` resolves the path and checks the fields against it.

The day (:data:`DAY_TABLE`)
---------------------------
The contract's day in the RZWQM2 4.6 order: every entry, its phase, the registry key of its
default implementation, the ports it produces and whether that implementation exists. Every
producer named in :data:`PORTS` is an entry of this table (``tests/unit/test_iface.py``), and the
skeleton day :func:`agrijax.models.day_rzwqm46.day_rzwqm46` is built from it, so the whole order is
exercised. Per-crop entries carry ``{slot}``. An entry's module (for the lag check) is its name
without the last component: ROOTWU is ``water_supply.<slot>.rootwu`` and the P10 producer
``n_supply.<slot>.replay``, each a module of its own outside ``crops.<slot>``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

from agrijax.core.day import Lag
from agrijax.core.dims import parse_dims
from agrijax.core.ledger import WaterLedger
from agrijax.core.units import parse_unit

from .crop import CanopyRecord, CropNIn, CropWaterIn, RootRecord
from .soil import NodeUptake, SinkInputs
from .surface import DailyWeather, PETFluxes, SnowOut

__all__ = [
    "DAY_PHASES",
    "DAY_TABLE",
    "ENTRY_STATUS",
    "PORTS",
    "TIME_SEMANTICS",
    "AllowedLag",
    "DayEntry",
    "FieldSpec",
    "PortSpec",
    "allowed_lags",
    "day_entries",
    "day_entry",
    "port_spec",
    "record_problems",
]

Time = Literal["same_day", "lag1", "mixed", "forcing", "cumulative"]
TIME_SEMANTICS: tuple[str, ...] = ("same_day", "lag1", "mixed", "forcing", "cumulative")


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

    def __post_init__(self) -> None:
        if self.field_of and self.record is not None:
            raise ValueError(f"{self.id}: give either a record class or field_of, not both")
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

    def global_path(self, slot: str | None = None) -> str:
        """The port's path with ``{slot}`` filled (a per-crop port needs ``slot``)."""
        if "{slot}" in self.path:
            if not slot:
                raise ValueError(f"{self.id} ({self.path}) is per crop slot: give slot=")
            return self.path.format(slot=slot)
        return self.path


def _f(unit: str, *dims: str, grid: str | None = None) -> FieldSpec:
    return FieldSpec(unit, tuple(dims), grid)


_RZ_ORDER = "RZWQM2 4.6 daily order (PHYSCL before the embedded crop), verified on the reference binary"

#: the port table of the M3 coupling contract, section 2.2
PORTS: dict[str, PortSpec] = {
    p.id: p
    for p in (
        PortSpec(
            id="P1",
            path="iface.crop_water.{slot}",
            record=CropWaterIn,
            fields=(
                ("sw", _f("cm3 cm-3", "n_layer", grid="dssat_layers")),
                ("eop", _f("mm d-1", "n_crop")),
                ("trwup", _f("cm d-1", "n_crop")),
            ),
            producers=(
                "crops.{slot}.remap_in (sw)",
                "crops.{slot}.eop (eop)",
                "water_supply.{slot}.rootwu (trwup)",
            ),
            consumers=(
                "crops.{slot}.phenology",
                "crops.{slot}.stress",
                "crops.{slot}.roots",
                "soil_water.uptake_limit",
            ),
            time="mixed",
            default=(
                "zeros: eop = 0 gives no water stress; trwup = 0 with eop > 0 full stress; "
                "sw = 0 no root growth"
            ),
            lags=(
                AllowedLag(
                    "soil_water.uptake_limit",
                    "trwup",
                    "RZWQM2 4.6 PHYSCL: WUF = min(1, PET / TRWUP) with the previous day's TRWUP ("
                    + _RZ_ORDER
                    + ")",
                ),
            ),
        ),
        PortSpec(
            id="P2",
            path="iface.root.{slot}",
            record=RootRecord,
            fields=(
                ("rlv", _f("cm cm-3", "n_crop", "n_layer", grid="dssat_layers")),
                ("rtdep", _f("cm", "n_crop")),
                ("rwumx", _f("cm3 cm-1 d-1", "n_crop")),
                ("pormin", _f("cm3 cm-3", "n_crop")),
                ("xhlai", _f("m2 m-2", "n_crop")),
            ),
            producers=("crops.{slot}.publish",),
            consumers=("water_supply.{slot}.rootwu",),
            time="lag1",
            default="zeros: xhlai = 0, ROOTWU not called, rwu = trwup = 0, TSS unchanged",
            lags=(
                AllowedLag(
                    "water_supply.{slot}.rootwu",
                    "",
                    "DSSAT-CSM LAND.for: SPAM reads the RLV the crop published on the previous call; "
                    "RZWQM2 4.6 ROOTWU reads the previous day's DSSATDRV exit RLV",
                ),
            ),
        ),
        PortSpec(
            id="P3",
            path="iface.root_uptake.{slot}",
            record=NodeUptake,
            fields=(("uptake", _f("cm d-1", "n_crop", "n_node", grid="rzwqm2_nodes")),),
            producers=("crops.{slot}.publish_uptake",),
            consumers=("soil_water.uptake_limit",),
            time="lag1",
            default="zeros: no sink",
            lags=(
                AllowedLag(
                    "soil_water.uptake_limit",
                    "",
                    "RZWQM2 4.6: the PHYSCL entry QSR is the previous day's DSSATDRV exit QSR ("
                    + _RZ_ORDER
                    + ")",
                ),
            ),
        ),
        PortSpec(
            id="P4",
            path="soil_water.sink_in",
            record=SinkInputs,
            fields=tuple(
                (n, _f("cm d-1", "n_node", grid="rzwqm2_nodes"))
                for n in ("uptake", "tile", "lateral", "subirrigation", "macropore_to_drain")
            ),
            producers=("soil_water.uptake_limit (uptake)",),
            consumers=("soil_water.day",),
            time="same_day",
            default="zeros with only uptake active: bit-identical to M1",
        ),
        PortSpec(
            id="P5",
            path="iface.pet",
            record=PETFluxes,
            fields=(
                ("transpiration", _f("cm d-1")),
                ("soil_evaporation", _f("cm d-1")),
                ("residue_evaporation", _f("cm d-1")),
                ("eo_priestley_taylor", _f("mm d-1")),
            ),
            producers=("pet.sw_daily",),
            consumers=("snow.prms", "soil_water.uptake_limit", "crops.{slot}.eop"),
            time="same_day",
            default="zeros: no demand",
        ),
        PortSpec(
            id="P6",
            path="iface.canopy.{slot}",
            record=CanopyRecord,
            fields=(
                ("lai", _f("m2 m-2", "n_crop")),
                ("tlai", _f("m2 m-2", "n_crop")),
                ("height", _f("cm", "n_crop")),
            ),
            producers=("crops.{slot}.canopy",),
            consumers=("pet.sw_daily",),
            time="lag1",
            default="zeros: bare soil (soil evaporation only)",
            lags=(
                AllowedLag(
                    "pet.sw_daily", "", "RZWQM2 4.6: PET in PHYSCL runs before the crop (" + _RZ_ORDER + ")"
                ),
            ),
        ),
        PortSpec(
            id="P7",
            path="soil_water.theta",
            record=None,
            field_of="agrijax.processes.soil_water.richards.SoilWater",
            fields=(("theta", _f("cm3 cm-3", "n_node")),),
            producers=("soil_water.day",),
            consumers=("pet.sw_daily",),
            time="lag1",
            default="the initial soil water",
            lags=(
                AllowedLag(
                    "pet.sw_daily",
                    "",
                    "RZWQM2 4.6: PET is computed before the PHYSCL time loop (" + _RZ_ORDER + ")",
                ),
            ),
        ),
        PortSpec(
            id="P8",
            path="forcing.weather",
            record=DailyWeather,
            fields=(
                ("tmin", _f("degC", "T")),
                ("tmax", _f("degC", "T")),
                ("srad", _f("MJ m-2 d-1", "T")),
                ("rh", _f("percent", "T")),
                ("wind_run", _f("km d-1", "T")),
                ("doy", _f("d", "T")),
            ),
            producers=("io (reference weather files) and the radiation reconstruction",),
            consumers=("pet.sw_daily", "snow.prms", "crops.{slot}"),
            time="forcing",
            default="-",
            pending=(
                (
                    "srad_horizontal",
                    _f("MJ m-2 d-1", "T"),
                    "G3: horizontal-surface radiation RTH for the Shuttleworth-Wallace kernel (srad is RTS)",
                ),
            ),
            kind="forcing",
        ),
        PortSpec(
            id="P9",
            path="iface.snow",
            record=SnowOut,
            fields=(
                ("melt", _f("cm d-1")),
                ("melt_runoff", _f("cm d-1")),
                ("swe", _f("mm")),
                ("sublimation", _f("cm d-1")),
            ),
            producers=("snow.prms",),
            consumers=("soil_water.day", "crops.{slot}.phenology", "ledger.close"),
            time="same_day",
            default="zeros: no snow",
        ),
        PortSpec(
            id="P10",
            path="iface.crop_n.{slot}",
            record=CropNIn,
            fields=(("nstres", _f("-", "n_crop")),),
            # The producer entry is n_supply.{slot}.replay with the registry key
            # n_supply/forcing_replay@none:replay (DAY_TABLE). docs/25 section 2.2 writes the key as
            # "@rzwqm2-4.6:replay"; the code key (ref_version none: a replay of data, exempt from the
            # faithful-sibling rule) is the one in force, and docs/25 section 2.2 is to be aligned.
            producers=(
                "n_supply.{slot}.replay (M3: n_supply/forcing_replay@none:replay); a nitrogen module later",
            ),
            consumers=("crops.{slot}.growth (nstress_replay variant)",),
            time="same_day",
            default="nstres = 1: no nitrogen limitation (the faithful nitrogen-off growth)",
        ),
        PortSpec(
            id="P11",
            path="ledger.water",
            record=WaterLedger,
            fields=(("storage0", _f("cm")), ("storage", _f("cm")), ("residual", _f("cm"))),
            producers=("ledger.close",),
            consumers=("outputs and checks",),
            time="cumulative",
            default="WaterLedger.init(storage0, inflows=..., outflows=...)",
        ),
    )
}


#: implementation status of a day entry: ``registered`` (a process under the entry's key),
#: ``kernel`` (a kernel or operator, no process yet), ``adapter`` (an unregistered assembly adapter
#: in :mod:`agrijax.models.day_rzwqm46`), ``none`` (no code)
ENTRY_STATUS: tuple[str, ...] = ("registered", "kernel", "adapter", "none")
Status = Literal["registered", "kernel", "adapter", "none"]

#: the phases of the contract's day, in order
DAY_PHASES: tuple[str, ...] = ("weather", "management", "physcl", "plant", "ledger")


@dataclass(frozen=True)
class DayEntry:
    """One entry of the contract's day (M3 contract, section 1.2).

    ``row`` is the row of the contract's day table (``"15a"``: the canopy producer, ``"P10"``: the
    crop nitrogen producer, both added by the W0 revision). ``key`` is the registry key of the
    default implementation (``""``: the assembly's own entry). ``ports_out`` are the ids of the
    ports the entry produces.
    """

    row: str
    phase: str
    entry: str
    key: str
    ports_out: tuple[str, ...]
    status: Status
    note: str = ""

    def __post_init__(self) -> None:
        if self.phase not in DAY_PHASES:
            raise ValueError(f"day entry {self.entry!r}: unknown phase {self.phase!r}")
        if self.status not in ENTRY_STATUS:
            raise ValueError(f"day entry {self.entry!r}: unknown status {self.status!r}")

    def name(self, slot: str) -> str:
        """The entry name for crop slot ``slot``."""
        return self.entry.format(slot=slot)


_G14 = "gap G14: no season initialisation process yet (sowing / harvest from the events)"

#: the contract's day in the RZWQM2 4.6 order (M3 contract section 1.2, with the W0 revision)
DAY_TABLE: tuple[DayEntry, ...] = (
    DayEntry(
        "1",
        "weather",
        "weather.radiation",
        "weather/rzwqm_radiation@rzwqm2-4.6:faithful",
        (),
        "kernel",
        "RTS and RTH are rebuilt in the forcing preprocessing (io, decision 11); the entry writes no state",
    ),
    DayEntry("2", "management", "events.apply", "", (), "none", "gap G15: irrigation and tillage events"),
    DayEntry(
        "3",
        "management",
        "crops.{slot}.season_init",
        "crop/ceres_maize.season_init@dssat-4.8.6.0:faithful",
        ("P2", "P6"),
        "none",
        _G14,
    ),
    DayEntry(
        "4",
        "physcl",
        "pet.sw_daily",
        "pet/shuttleworth_wallace@rzwqm2-4.6:faithful",
        ("P5",),
        "registered",
        "reads its own surface subtree, not the P6/P7 ports yet (gap G5)",
    ),
    DayEntry("5", "physcl", "snow.prms", "snow/prms@rzwqm2-4.6:faithful", ("P9",), "none", "gap G16"),
    DayEntry(
        "6",
        "physcl",
        "soil_water.uptake_limit",
        "soil_water/wuf@rzwqm2-4.6:faithful",
        ("P4",),
        "none",
        "gap G8",
    ),
    DayEntry(
        "7",
        "physcl",
        "soil_water.day",
        "soil_water/day@rzwqm2-4.6:faithful",
        ("P7",),
        "registered",
        "the sink comes from the forcing, not P4 (gap G1)",
    ),
    DayEntry(
        "P10",
        "plant",
        "n_supply.{slot}.replay",
        # docs/25 section 2.2 writes "@rzwqm2-4.6:replay"; this code key is the one in force and
        # docs/25 section 2.2 is to be aligned (M3 contract section 11, item 9)
        "n_supply/forcing_replay@none:replay",
        ("P10",),
        "registered",
        "replay of the reference run's NSTRES (RZWQM2 NUTRI runs before MAPLNT); a nitrogen module later",
    ),
    DayEntry(
        "8",
        "plant",
        "crops.{slot}.remap_in",
        "crop_iface/remap_in@rzwqm2-4.6:faithful",
        ("P1",),
        "adapter",
        "REALMATCH kernel agrijax.core.grids.remap_intensive",
    ),
    DayEntry(
        "9",
        "plant",
        "crops.{slot}.eop",
        "crop_iface/eop_from_pet@rzwqm2-4.6:istress0",
        ("P1",),
        "adapter",
        "gap G7",
    ),
    DayEntry(
        "10",
        "plant",
        "water_supply.{slot}.rootwu",
        "water_supply/rootwu@dssat-4.8.6.0:faithful",
        ("P1",),
        "registered",
        "state at water_supply.<slot>, outside the crop subtree (section 11, items 1-2)",
    ),
    DayEntry(
        "11",
        "plant",
        "crops.{slot}.phenology",
        "crop/ceres_maize.phenology@dssat-4.8.6.0:faithful",
        (),
        "registered",
    ),
    DayEntry(
        "12",
        "plant",
        "crops.{slot}.stress",
        "crop/ceres_maize.stress@dssat-4.8.6.0:faithful",
        (),
        "registered",
    ),
    DayEntry(
        "13",
        "plant",
        "crops.{slot}.growth",
        "crop/ceres_maize.growth@dssat-4.8.6.0:nstress_replay",
        (),
        "registered",
        "reads P10",
    ),
    DayEntry(
        "14", "plant", "crops.{slot}.roots", "crop/ceres_maize.roots@dssat-4.8.6.0:faithful", (), "registered"
    ),
    DayEntry(
        "15",
        "plant",
        "crops.{slot}.publish",
        "crop/ceres_maize.publish@dssat-4.8.6.0:faithful",
        ("P2",),
        "registered",
    ),
    DayEntry(
        "15a",
        "plant",
        "crops.{slot}.canopy",
        "crop/ceres_maize.canopy@dssat-4.8.6.0:faithful",
        ("P6",),
        "none",
        "the canopy record read by PET the next day (section 11, item 10); its source (MAPLNT exit, harvest "
        "day, height) is open (section 11, item 6)",
    ),
    DayEntry(
        "16",
        "plant",
        "crops.{slot}.publish_uptake",
        "crop_iface/publish_uptake@rzwqm2-4.6:faithful",
        ("P3",),
        "adapter",
        "gaps G8, G21",
    ),
    DayEntry("17", "ledger", "ledger.close", "", ("P11",), "adapter", "soil column and pond only (gap G17)"),
)


def day_entries(slot: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((phase, entry names), ...)`` of the contract's day for crop slot ``slot``, in order."""
    return tuple(
        (ph, tuple(e.name(slot) for e in DAY_TABLE if e.phase == ph))
        for ph in DAY_PHASES
        if any(e.phase == ph for e in DAY_TABLE)
    )


def day_entry(name: str, slot: str) -> DayEntry:
    """The :class:`DayEntry` whose name for crop slot ``slot`` is ``name``."""
    for e in DAY_TABLE:
        if e.name(slot) == name:
            return e
    raise KeyError(f"no entry {name!r} in the contract's day (slot {slot!r})")


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


def allowed_lags(slot: str, ports: tuple[str, ...] | None = None) -> tuple[Lag, ...]:
    """The :class:`~agrijax.core.day.Lag` table the contract allows for crop slot ``slot``: one lag
    per :class:`AllowedLag` of the given ports (default: every port), with ``{slot}`` filled.
    Pass it as ``Day(lags=...)``; an implementation may use any subset of it."""
    out: list[Lag] = []
    for pid in ports if ports is not None else tuple(PORTS):
        spec = port_spec(pid)
        base = spec.global_path(slot)
        for lag in spec.lags:
            path = f"{base}.{lag.field}" if lag.field else base
            out.append(Lag(lag.reader.format(slot=slot), path, evidence=lag.evidence or spec.id))
    return tuple(out)
