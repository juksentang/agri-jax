"""A one-day skeleton of the M3 day in the RZWQM2 4.6 order (``DAY_RZWQM46``), for the contract smoke test.

This is not the M3 assembly. It checks that the coupling contract (the port records of
:mod:`agrijax.iface`, their global paths, units and allowed lags) fits the modules that exist
today before the slot modules are developed in parallel. The day is the contract's whole day
(:data:`agrijax.iface.contract.DAY_TABLE`, every phase and entry in order). Every entry that has
no process yet is stood in for by a **replay** entry that writes the entry's output port from
data (a reference run), so the ports are exercised end to end, or by a **no-op** entry where
nothing changes on the smoke days:

======================================  =============================================  =========
entry                                   process                                        port out
======================================  =============================================  =========
``weather``
``weather.radiation``                   no-op (RTS, RTH preprocessed in io)            --
``management``
``events.apply``                        no-op (no event process yet, gap G15)          --
``crops.<slot>.season_init``            no-op (no season init yet, gap G14)            --
``physcl``
``pet.sw_daily``                        replay (PET fluxes to the port)                P5
``snow.prms``                           replay (no snow module yet)                    P9
``soil_water.uptake_limit``             replay (no WUF yet)                            P4
``soil_water.day``                      :func:`soil_water_day_entry` (the registered
                                        Green-Ampt day on the ports)                   --
``plant``
``n_supply.<slot>.replay``              ``n_supply/forcing_replay@none:replay``        P10
``crops.<slot>.remap_in``               :func:`remap_in_entry` (``REALMATCH`` kernel)  P1 ``sw``
``crops.<slot>.eop``                    :func:`eop_entry` (``EOP = 10 PET``)           P1 ``eop``
``water_supply.<slot>.rootwu``          ``water_supply/rootwu@dssat-4.8.6.0:faithful`` P1 ``trwup``
``crops.<slot>.phenology`` .. ``roots`` CERES-Maize, growth ``:nstress_replay``        --
``crops.<slot>.publish``                ``crop/ceres_maize.publish@...:faithful``      P2
``crops.<slot>.canopy``                 replay (the crop publishes no canopy yet)      P6
``crops.<slot>.publish_uptake``         :func:`publish_uptake_entry`                   P3
``ledger``
``ledger.close``                        :func:`ledger_entry` (soil column + pond)      P11
======================================  =============================================  =========

A replay entry declares, as its reads, the read set of the producer it stands in for (the
contract's day table), so :meth:`~agrijax.core.day.Day.check` sees the lagged reads the coupled
day will have; the replay itself uses none of them (it copies a forcing record). A no-op entry
declares no reads and no writes: a season initialisation that declared its writes of ``P2`` and
``P6`` would run before ROOTWU and PET and so hide their lags from the static check (the
reference's own exception on a season's first day, O-RZ3), which the skeleton does not model.

Global state (paths are the contract's; a nested dict of records)::

    soil_water        {h, theta, pond, flux, sink_in}   the soil-water state plus P4 (see below)
    iface.pet         PETFluxes          P5      iface.snow            SnowOut         P9
    iface.canopy.<s>  CanopyRecord       P6      iface.crop_water.<s>  CropWaterIn     P1
    iface.root.<s>    RootRecord         P2      iface.root_uptake.<s> NodeUptake      P3
    iface.crop_n.<s>  CropNIn            P10     ledger.water          WaterLedger     P11
    crops.<s>         CERES-Maize state (ports detached)
    water_supply.<s>  ROOTWU producer state (TSS, RWU)
    n_supply.<s>      the P10 replay producer (its only field is its port)

``soil_water`` is a dict of the :class:`~agrijax.processes.soil_water.richards.SoilWater` fields
plus ``sink_in`` because the soil-water state class has no ``sink_in`` field yet (contract gap
G1); :func:`soil_water_day_entry` rebuilds the ``SoilWater`` record, calls the registered
``soil_water/day@rzwqm2-4.6:faithful`` process on it and writes the result back. The ROOTWU
state sits at ``water_supply.<s>`` and its entry is ``water_supply.<s>.rootwu``: a module of its
own, outside the CERES subtree ``crops.<s>``, so its read of yesterday's root record (P2) is a
lag :meth:`~agrijax.core.day.Day.check` sees (M3 contract section 11, items 1-2).

Global params: ``{"soil": SoilWaterDayParams, "crop": CeresMaizeParams, "rootwu": RootwuParams,
"crop_iface": CropIfaceParams}``. Global forcing: ``{"soil": StormForcing, "crop": CeresForcing,
"n": CropNReplayForcing, "replay": {...}}`` with the replay records under ``replay``.

Source: M3 coupling contract (docs 25, private), sections 1.2 and 2.2; decision 3 (allowed lags).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.day import Day, Phase
from agrijax.core.grids import SoilGrid, remap_intensive
from agrijax.core.ledger import WaterLedger, water_ledger
from agrijax.core.ports import Binding, bind, compose
from agrijax.core.process import Process, process
from agrijax.core.state import Params, field, get_path, set_path
from agrijax.core.units import HOURS_PER_DAY, MM_PER_CM
from agrijax.iface.contract import allowed_lags, day_entries
from agrijax.processes.crop.ceres_maize import CROP_PROCESSES_NSTRESS_REPLAY, CeresMaizeState
from agrijax.processes.n_supply import CropNReplayState, crop_n_replay
from agrijax.processes.soil_water.day import SoilWaterDayForcing, soil_water_day
from agrijax.processes.soil_water.richards import RichardsState, SoilWater
from agrijax.processes.soil_water.sinks import SINK_LEDGER_OUTFLOWS
from agrijax.processes.soil_water.uptake import RootwuState, rootwu_supply

__all__ = [
    "CROP_ENTRIES",
    "DEFAULT_EVAPORATION_DEMAND",
    "LEDGER_INFLOWS",
    "LEDGER_OUTFLOWS",
    "SLOT",
    "CropIfaceParams",
    "day_processes",
    "day_rzwqm46",
    "eop_entry",
    "initial_state",
    "ledger_entry",
    "ledger_initial",
    "noop_entry",
    "publish_uptake_entry",
    "remap_in_entry",
    "replay_entry",
    "soil_water_day_entry",
]

#: the crop slot of the M3 assembly
SLOT = "maize"
#: hours of the day's hourly forcing arrays
_N_HOURS: int = int(HOURS_PER_DAY)
#: the CERES-Maize entries after the crop-water producers, in the ``MZ_CERES`` order
CROP_ENTRIES: tuple[str, ...] = ("phenology", "stress", "growth", "roots", "publish")
#: the P5 fields whose sum is the soil-water day's evaporation demand. At CA-TPA 2015-2023 the
#: reference's actual evaporation (``.ana`` column 6) equals ``PES + PER`` on every day it is not
#: supply-limited (measured in ``tests/integration/test_day_rzwqm46_smoke.py``), so the residue
#: evaporation is drawn from the soil water too (contract gap G2 names ``PES`` only).
DEFAULT_EVAPORATION_DEMAND: tuple[str, ...] = ("soil_evaporation", "residue_evaporation")

#: ledger channels of the skeleton: rain events and the snowmelt supply in; the soil column's outflows
LEDGER_INFLOWS: dict[str, str] = {"rain": "soil_water.flux.rain", "snowmelt": "iface.snow.melt"}
LEDGER_OUTFLOWS: dict[str, str] = {
    "evaporation": "soil_water.flux.evaporation",
    "drainage": "soil_water.flux.drainage",
    "runoff": "soil_water.flux.runoff",
    **SINK_LEDGER_OUTFLOWS,
}


def _p(slot: str) -> dict[str, str]:
    """Global paths of the per-crop ports of ``slot``."""
    return {
        "crop_water": f"iface.crop_water.{slot}",
        "root": f"iface.root.{slot}",
        "canopy": f"iface.canopy.{slot}",
        "root_uptake": f"iface.root_uptake.{slot}",
        "crop_n": f"iface.crop_n.{slot}",
        "crop": f"crops.{slot}",
        "rootwu": f"water_supply.{slot}",
        "n_supply": f"n_supply.{slot}",
    }


def day_rzwqm46(slot: str = SLOT) -> Day:
    """The contract's day in the RZWQM2 4.6 order for crop ``slot`` (every phase and entry of
    :data:`agrijax.iface.contract.DAY_TABLE`), with the contract's allowed lags."""
    return Day(
        ref="rzwqm2-4.6",
        phases=tuple(Phase(ph, entries) for ph, entries in day_entries(slot)),
        lags=allowed_lags(slot),
    )


# ------------------------------------------------------------------------ params of the adapters
class CropIfaceParams(Params):
    """The two grids of the crop interface (soil-water nodes, crop layers) and the crop layers'
    lower limit, read by :func:`remap_in_entry` and :func:`publish_uptake_entry`."""

    nodes: SoilGrid = field(description="soil-water node grid (RZWQM2 TLT)", static=True)
    layers: SoilGrid = field(description="crop layers (LYRSET on the node grid)", static=True)
    ll: Array = field(
        unit="cm3 cm-3", description="lower limit of the crop layers", fortran_name="LL", dims=("n_layer",)
    )


# ------------------------------------------------------------------------ framework entries
def replay_entry(name: str, copies: Mapping[str, str], *, stands_in_reads: Sequence[str] = ()) -> Process:
    """A replay entry: copy ``{state path: forcing path}`` records from the day's forcing into the state.

    ``stands_in_reads`` are declared as its reads: the read set of the producer the replay stands
    in for, so the day's lag check sees the coupled dataflow. The copy itself reads no state.
    """
    pairs = tuple((str(t), str(s)) for t, s in copies.items())

    def _replay(state: Any, params: Any, forcing_t: Any) -> Any:
        """Write each target record from its forcing record (a reference run's values).

        Source: M3 coupling contract, section 2.1 (replay and coupled bindings of the same ports).
        """
        for target, src in pairs:  # static loop over the declared copies
            state = set_path(state, target, get_path(forcing_t, src))
        return state

    return process(
        _replay,
        reads=tuple(stands_in_reads),
        writes=tuple(t for t, _ in pairs),
        name=name,
        register=False,
        source="replay of a reference run (M3 coupling contract, section 2.1)",
    )


def noop_entry(name: str, *, why: str) -> Process:
    """An entry of the contract's day whose producer does not exist yet and that changes nothing
    on the days it is run (``why`` says why): no reads, no writes, the state comes back as is."""

    def _noop(state: Any, params: Any, forcing_t: Any) -> Any:
        """Leave the state unchanged (a placeholder of the contract's day order).

        Source: M3 coupling contract, section 1.2 (the entry's producer is not implemented yet).
        """
        return state

    return process(
        _noop,
        reads=(),
        writes=(),
        name=name,
        register=False,
        source=f"placeholder of the contract's day: {why}",
    )


def soil_water_day_entry(
    *,
    name: str = "soil_water.day",
    params_key: str = "soil",
    forcing_key: str = "soil",
    evaporation_demand: Sequence[str] = DEFAULT_EVAPORATION_DEMAND,
) -> Process:
    """The registered soil-water day (``soil_water/day@rzwqm2-4.6:faithful``) on the contract's ports.

    Reads the sink ``soil_water.sink_in.uptake`` (P4), the evaporation demand from P5 (the sum of
    ``evaporation_demand``, spread uniformly over the 24 h: the contract has no evaporation-demand
    adapter yet, gap G2), the snowmelt ``iface.snow.melt`` (P9) as a surface supply spread over the
    day (the Green-Ampt event has no entry for an event without breakpoints, gap G16), and the
    day's storm from the forcing. Writes ``soil_water.{h, theta, pond, flux}``.
    """
    demand = tuple(evaporation_demand)

    def _soil_water_day(state: Any, params: Any, forcing_t: Any) -> Any:
        """One soil-water day with its inputs taken from the ports.

        Source: M3 coupling contract, day table entry 7 and ports P4, P5, P9; the day itself is
        ``soil_water/day@rzwqm2-4.6:faithful``.
        """
        sw = get_path(state, "soil_water")
        water = SoilWater(flux=sw["flux"], h=sw["h"], theta=sw["theta"], pond=sw["pond"])
        pet = get_path(state, "iface.pet")
        hours = jnp.ones((_N_HOURS,), dtype=water.theta.dtype)
        daily_demand = sum(getattr(pet, d) for d in demand)
        f = SoilWaterDayForcing(
            storm=get_path(forcing_t, forcing_key),
            supply=get_path(state, "iface.snow.melt") / HOURS_PER_DAY * hours,
            evaporation=daily_demand / HOURS_PER_DAY * hours,
            uptake=sw["sink_in"].uptake,
        )
        new = soil_water_day(RichardsState(soil_water=water), get_path(params, params_key), f).soil_water
        for k in ("h", "theta", "pond", "flux"):  # static loop over the written fields
            state = set_path(state, f"soil_water.{k}", getattr(new, k))
        return state

    return process(
        _soil_water_day,
        reads=(
            "soil_water.h",
            "soil_water.theta",
            "soil_water.pond",
            "soil_water.sink_in.uptake",
            *(f"iface.pet.{d}" for d in demand),
            "iface.snow.melt",
        ),
        writes=("soil_water.h", "soil_water.theta", "soil_water.pond", "soil_water.flux"),
        name=name,
        register=False,
        source="soil_water/day@rzwqm2-4.6:faithful bound to the M3 ports (P4, P5, P9)",
    )


def remap_in_entry(slot: str = SLOT, *, params_key: str = "crop_iface") -> Process:
    """``crops.<slot>.remap_in``: today's node water content mapped to the crop layers (P1 ``sw``),
    the thickness-weighted mean of RZWQM2 ``REALMATCH`` (:func:`agrijax.core.grids.remap_intensive`)."""
    target = f"iface.crop_water.{slot}.sw"

    def _remap_in(state: Any, params: Any, forcing_t: Any) -> Any:
        """Map ``soil_water.theta`` (nodes) to the crop layers.

        Source: DSSAT-CSM v4.8.6.0 LMATCH.for (thickness-weighted mean), the node -> layer map of
        RZWQM2 DSSATDRV validated in tests/integration/test_rootwu_dssat.py.
        """
        p: CropIfaceParams = get_path(params, params_key)
        old = get_path(state, target)
        sw = remap_intensive(get_path(state, "soil_water.theta"), p.nodes, p.layers)
        return set_path(state, target, sw.astype(old.dtype))

    return process(
        _remap_in,
        reads=("soil_water.theta",),
        writes=(target,),
        name=f"crops.{slot}.remap_in",
        register=False,
        source="DSSAT-CSM v4.8.6.0 LMATCH.for; RZWQM2 DSSATDRV node -> layer map",
    )


def eop_entry(slot: str = SLOT) -> Process:
    """``crops.<slot>.eop``: ``EOP = 10 PET`` (ISTRESS = 0), P5 ``transpiration`` [cm d-1] to
    P1 ``eop`` [mm d-1] for every crop of the slot."""
    target = f"iface.crop_water.{slot}.eop"

    def _eop(state: Any, params: Any, forcing_t: Any) -> Any:
        """The crop's potential transpiration from the PET port.

        Source: RZWQM2 4.6 DSSATDRV with ISTRESS = 0 (EOP = PET in mm d-1; the reference's
        DSSATDRV exit EOP equals 10 PET, tests/integration/test_day_rzwqm46_smoke.py).
        """
        old = get_path(state, target)
        eop = get_path(state, "iface.pet.transpiration") * MM_PER_CM * jnp.ones_like(old)
        return set_path(state, target, eop.astype(old.dtype))

    return process(
        _eop,
        reads=("iface.pet.transpiration",),
        writes=(target,),
        name=f"crops.{slot}.eop",
        register=False,
        source="RZWQM2 4.6 DSSATDRV (ISTRESS = 0): EOP = PET",
    )


def publish_uptake_entry(slot: str = SLOT, *, params_key: str = "crop_iface") -> Process:
    """``crops.<slot>.publish_uptake``: the day's layer uptake as node uptake (P3, extensive).

    The layer rate ``rwu / dlayr`` [d-1] of the layers with ``SW > LL`` (0 elsewhere) is mapped to
    the nodes as an intensive quantity (thickness-weighted mean, RZWQM2 ``DSSATDRV``) and times the
    node thickness gives the node amount [cm d-1] of :class:`~agrijax.iface.soil.NodeUptake`. The
    layer with ``SW == LL`` exactly keeps the reference's previous node value in RZWQM2 (gap G21);
    this entry gives it 0.
    """
    rwu_path = f"{_p(slot)['rootwu']}.rwu"
    sw_path = f"iface.crop_water.{slot}.sw"
    target = f"iface.root_uptake.{slot}.uptake"

    def _publish_uptake(state: Any, params: Any, forcing_t: Any) -> Any:
        """Layer uptake -> node uptake (the soil side reads it on the next day).

        Source: RZWQM2 4.6 DSSATDRV layer -> node map of qsr (thickness-weighted mean of the layer
        rates), validated in tests/integration/test_rootwu_dssat.py; DSSAT-CSM LMATCH.for.
        """
        p: CropIfaceParams = get_path(params, params_key)
        old = get_path(state, target)
        rwu = get_path(state, rwu_path)
        dlayr = jnp.asarray(p.layers.thickness, dtype=rwu.dtype)
        tl = jnp.asarray(p.nodes.thickness, dtype=rwu.dtype)
        rate = rwu / dlayr
        rate = jnp.where(get_path(state, sw_path) > p.ll, rate, 0.0)
        node = remap_intensive(rate, p.layers, p.nodes) * tl
        return set_path(state, target, node.astype(old.dtype))

    return process(
        _publish_uptake,
        reads=(rwu_path, sw_path),
        writes=(target,),
        name=f"crops.{slot}.publish_uptake",
        register=False,
        source="RZWQM2 4.6 DSSATDRV layer -> node map of the uptake; DSSAT-CSM LMATCH.for",
    )


def _storage(state: Any, params: Any, forcing_t: Any) -> Array:
    """Soil profile storage plus pond [cm]."""
    tl = get_path(params, "soil").richards.grid.tl
    return jnp.sum(get_path(state, "soil_water.theta") * tl, axis=-1) + get_path(state, "soil_water.pond")


def ledger_entry(*, atol: float | None = None, rtol: float | None = None) -> Process:
    """``ledger.close`` of the skeleton: soil column plus pond, inflows :data:`LEDGER_INFLOWS`,
    outflows :data:`LEDGER_OUTFLOWS` (no snow storage, sublimation or irrigation yet: gap G17)."""
    return water_ledger(
        storage=_storage,
        inflows=LEDGER_INFLOWS,
        outflows=LEDGER_OUTFLOWS,
        reads=("soil_water.theta", "soil_water.pond"),
        atol=atol,
        rtol=rtol,
    )


def ledger_initial(storage0: Any) -> WaterLedger:
    """A zero ledger with the skeleton's channels starting at ``storage0`` [cm]."""
    return WaterLedger.init(storage0, inflows=tuple(LEDGER_INFLOWS), outflows=tuple(LEDGER_OUTFLOWS))


# ------------------------------------------------------------------------ assembly
def _stand_in_reads(slot: str) -> dict[str, tuple[str, ...]]:
    """Read sets of the producers the replay entries stand in for (contract day table rows 4-6, 15a)."""
    p = _p(slot)
    return {
        "pet.sw_daily": (p["canopy"], "soil_water.theta"),
        "snow.prms": ("iface.pet.soil_evaporation",),
        "soil_water.uptake_limit": (
            p["root_uptake"],
            f"{p['crop_water']}.trwup",
            "iface.pet.transpiration",
            "soil_water.theta",
        ),
        f"crops.{slot}.canopy": (f"{p['crop']}.growth",),
    }


def day_processes(
    slot: str = SLOT,
    *,
    evaporation_demand: Sequence[str] = DEFAULT_EVAPORATION_DEMAND,
    replace: Mapping[str, Process] | None = None,
) -> dict[str, Process]:
    """``{entry: process}`` of the skeleton for crop ``slot`` (feed to ``day_rzwqm46(slot).compile``).

    ``replace`` swaps entries (a replay binding of a port that has a producer, for the replay vs
    coupled comparison); the day and its lag check stay the same.
    """
    p = _p(slot)
    si = _stand_in_reads(slot)
    ports = {"water_in": p["crop_water"], "root_out": p["root"], "n_in": p["crop_n"]}
    procs: dict[str, Process] = {
        "weather.radiation": noop_entry(
            "weather.radiation",
            why="RTS and RTH are rebuilt in the forcing preprocessing (io; M3 contract decision 11)",
        ),
        "events.apply": noop_entry(
            "events.apply", why="no management event process yet (gap G15); the smoke days have no event"
        ),
        f"crops.{slot}.season_init": noop_entry(
            f"crops.{slot}.season_init",
            why="no season initialisation process yet (gap G14); the smoke days are no sowing or harvest day",
        ),
        "pet.sw_daily": replay_entry(
            "pet.sw_daily", {"iface.pet": "replay.pet"}, stands_in_reads=si["pet.sw_daily"]
        ),
        "snow.prms": replay_entry(
            "snow.prms", {"iface.snow": "replay.snow"}, stands_in_reads=si["snow.prms"]
        ),
        "soil_water.uptake_limit": replay_entry(
            "soil_water.uptake_limit",
            {"soil_water.sink_in": "replay.sink_in"},
            stands_in_reads=si["soil_water.uptake_limit"],
        ),
        "soil_water.day": soil_water_day_entry(evaporation_demand=evaporation_demand),
        f"n_supply.{slot}.replay": bind(
            crop_n_replay,
            own=p["n_supply"],
            ports={"n_out": p["crop_n"]},
            forcing="n",
            name=f"n_supply.{slot}.replay",
        ),
        f"crops.{slot}.remap_in": remap_in_entry(slot),
        f"crops.{slot}.eop": eop_entry(slot),
        f"water_supply.{slot}.rootwu": bind(
            rootwu_supply,
            own=p["rootwu"],
            ports={"root": p["root"], "water": p["crop_water"]},
            params="rootwu",
            name=f"water_supply.{slot}.rootwu",
        ),
        **{
            f"crops.{slot}.{n}": bind(
                proc, own=p["crop"], ports=ports, params="crop", forcing="crop", name=f"crops.{slot}.{n}"
            )
            for n, proc in zip(CROP_ENTRIES, CROP_PROCESSES_NSTRESS_REPLAY, strict=True)
        },
        f"crops.{slot}.canopy": replay_entry(
            f"crops.{slot}.canopy", {p["canopy"]: "replay.canopy"}, stands_in_reads=si[f"crops.{slot}.canopy"]
        ),
        f"crops.{slot}.publish_uptake": publish_uptake_entry(slot),
        "ledger.close": ledger_entry(),
    }
    for k, v in (replace or {}).items():
        if k not in procs:
            raise KeyError(f"no entry {k!r} in the skeleton day")
        procs[k] = v
    return procs


def initial_state(
    *,
    water: SoilWater,
    sink_in: Any,
    crop: CeresMaizeState,
    rootwu: RootwuState,
    records: Mapping[str, Any],
    storage0: Any,
    slot: str = SLOT,
) -> dict[str, Any]:
    """The global morning state: the soil water with its sink record, the crop and ROOTWU states
    (their own subtrees; ``crop``'s ports are replaced), the P10 producer, and the interface
    ``records`` ``{"pet", "snow", "canopy", "crop_water", "root", "root_uptake", "crop_n"}``
    (yesterday's values of the lagged ports). ``storage0`` starts the ledger."""
    p = _p(slot)
    crop_binding = Binding(
        p["crop"], (("water_in", p["crop_water"]), ("root_out", p["root"]), ("n_in", p["crop_n"]))
    )
    crop_full = crop.replace(water_in=records["crop_water"], root_out=records["root"], n_in=records["crop_n"])
    soil = {"flux": water.flux, "h": water.h, "theta": water.theta, "pond": water.pond, "sink_in": sink_in}
    return compose(
        {
            "soil_water": soil,
            **crop_binding.entries(crop_full),
            p["rootwu"]: rootwu,
            p["n_supply"]: CropNReplayState(),
            "iface.pet": records["pet"],
            "iface.snow": records["snow"],
            p["canopy"]: records["canopy"],
            p["root_uptake"]: records["root_uptake"],
            "ledger.water": ledger_initial(storage0),
        }
    )
