"""The contract's day in the RZWQM2 4.6 order: :class:`DayEntry`, :data:`DAY_TABLE`, the planned
rows :data:`POST_M3_DAY_TABLE`, and the owners' declared extra writes (:data:`PHASED_WRITES`,
:data:`SHARED_STATE`).

Part of :mod:`agrijax.iface.contract` (its docstring explains the tables); import the names from
there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agrijax.core.day import PhasedWrite

from .spec import STAGES, Stage

#: implementation status of a day entry: ``registered`` (a process under the entry's key),
#: ``kernel`` (a kernel or operator, no process yet), ``adapter`` (an unregistered assembly adapter
#: in :mod:`agrijax.models.day_rzwqm46`), ``none`` (no code)
ENTRY_STATUS: tuple[str, ...] = ("registered", "kernel", "adapter", "none")
Status = Literal["registered", "kernel", "adapter", "none"]

#: the phases of the contract's day, in order
DAY_PHASES: tuple[str, ...] = ("weather", "management", "physcl", "plant", "ledger")
#: the phases with the planned modules: ``chem`` (the reference's chemistry and nutrient call after
#: the soil physics and before the crop) and ``post_plant`` (the phosphorus block after the crop)
EXTENDED_PHASES: tuple[str, ...] = (
    "weather",
    "management",
    "physcl",
    "chem",
    "plant",
    "post_plant",
    "ledger",
)


@dataclass(frozen=True)
class DayEntry:
    """One entry of the contract's day (the RZWQM2 4.6 order).

    ``row`` is the row of the contract's day table (``"15a"``: the canopy producer, ``"P10"``: the
    crop nitrogen producer, both inserted between the numbered rows; ``"16a"``: the crop's harvest
    reset; ``"16b"``: ROOTWU's own end of season, written by ROOTWU itself, not by the crop).
    ``key`` is the registry key of the default implementation (``""``: the assembly's own entry).
    ``ports_out`` are the ids of the ports the entry produces (or, for a season end, resets on event
    days).
    """

    row: str
    phase: str
    entry: str
    key: str
    ports_out: tuple[str, ...]
    status: Status
    note: str = ""
    #: ``m3`` or ``post_m3`` (a planned row; module docstring)
    stage: Stage = "m3"
    #: a planned row: the entry it follows (``{slot}`` form), or ``"^<phase>"`` for the start of a phase
    after: str = ""
    #: M3 rows (``{slot}`` form) that this planned row replaces when its module is on
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        phases = DAY_PHASES if self.stage == "m3" else EXTENDED_PHASES
        if self.phase not in phases:
            raise ValueError(f"day entry {self.entry!r}: unknown phase {self.phase!r}")
        if self.status not in ENTRY_STATUS:
            raise ValueError(f"day entry {self.entry!r}: unknown status {self.status!r}")
        if self.stage not in STAGES:
            raise ValueError(f"day entry {self.entry!r}: unknown stage {self.stage!r}")
        if (self.stage == "m3") == bool(self.after):
            raise ValueError(
                f"day entry {self.entry!r}: a planned row (and only a planned row) names 'after'"
            )

    def name(self, slot: str) -> str:
        """The entry name for crop slot ``slot``."""
        return self.entry.format(slot=slot)


_G14 = (
    "SEASINIT of the crop's own subtree on the event table's sow flag; it writes no port, "
    "so the lagged reads of P2 and P6 by ROOTWU and PET stay lags (the no-crop records of a sowing "
    "morning come from the initial state or the previous harvest, entry 16a)"
)

#: the contract's day in the RZWQM2 4.6 order
DAY_TABLE: tuple[DayEntry, ...] = (
    DayEntry(
        "1",
        "weather",
        "weather.radiation",
        "weather/rzwqm_radiation@rzwqm2-4.6:faithful",
        (),
        "kernel",
        "RTS and RTH are rebuilt in the forcing preprocessing (io, outside the differentiable day); "
        "the entry writes no state",
    ),
    DayEntry(
        "2", "management", "events.apply", "", (), "none", "irrigation and tillage events; no process yet"
    ),
    DayEntry(
        "3",
        "management",
        "crops.{slot}.season_init",
        "crop/ceres_maize.season_init@dssat-4.8.6.0:faithful",
        (),
        "registered",
        _G14,
    ),
    DayEntry(
        "4",
        "physcl",
        "pet.sw_daily",
        "pet/shuttleworth_wallace@rzwqm2-4.6:faithful",
        ("P5",),
        "registered",
        "reads P6 (lag 1), P7 (lag 1) and the weather with RTH; writes P5",
    ),
    DayEntry(
        "5",
        "physcl",
        "snow.prms",
        "snow/prms@rzwqm2-4.6:faithful",
        ("P9",),
        "registered",
        "PRMS snowpack, state at surface.snow",
    ),
    DayEntry(
        "6",
        "physcl",
        "soil_water.uptake_limit",
        "soil_water/wuf@rzwqm2-4.6:faithful",
        ("P4",),
        "registered",
        "WUF = min(1, PET / TRWUP) on yesterday's P3 and P1 trwup",
    ),
    DayEntry(
        "7",
        "physcl",
        "soil_water.day",
        "soil_water/day@rzwqm2-4.6:faithful",
        ("P7",),
        "registered",
        "the registered process reads its sink from the forcing, not from P4; models.day_rzwqm46 binds it "
        "to P4",
    ),
    DayEntry(
        "P10",
        "plant",
        "n_supply.{slot}.replay",
        # a replay of data: ref_version none, exempt from the faithful-sibling rule
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
        "crop_iface/eop_from_pet@rzwqm2-4.6:faithful",
        ("P1",),
        "registered",
        "reads P5 transpiration the same day, writes P1 eop",
    ),
    DayEntry(
        "10",
        "plant",
        "water_supply.{slot}.rootwu",
        "water_supply/rootwu@dssat-4.8.6.0:faithful",
        ("P1",),
        "registered",
        "state at water_supply.<slot>, outside the crop subtree",
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
        "crop/ceres_maize.canopy@rzwqm2-4.6:faithful",
        ("P6",),
        "registered",
        "the canopy record read by PET the next day is the MAPLNT exit: "
        "the RZWQM2 driver's LAI (XHLAI, declining to 0 after maturity when HDATE > MDATE), TLAI = LAI, "
        "and the stalk-mass height in cm (running maximum over the season); the harvest day's zeros are "
        "the crop's harvest reset's (16a)",
    ),
    DayEntry(
        "16",
        "plant",
        "crops.{slot}.publish_uptake",
        "crop_iface/publish_uptake@rzwqm2-4.6:faithful",
        ("P3",),
        "registered",
        "registered in processes/water_supply/publish.py; it reproduces the reference's SW == LL quirk of "
        "the layer publish",
    ),
    DayEntry(
        "16a",
        "plant",
        "crops.{slot}.harvest",
        "crop/ceres_maize.harvest@rzwqm2-4.6:faithful",
        ("P2", "P6"),
        "registered",
        "at the end of a harvest day (event table's harvest flag) the crop becomes the SEASINIT state "
        "of the next season, P2 the no-crop root record and P6 bare soil; masked resets after the day's "
        "readers, so P2 and P6 keep their lags (the same module as their producers publish, canopy: one "
        "writer module, AJ013; the writes are declared season_end in PHASED_WRITES). It writes nothing "
        "outside the crop slot",
    ),
    DayEntry(
        "16b",
        "plant",
        "water_supply.{slot}.season_end",
        "water_supply/rootwu_season_end@rzwqm2-4.6:faithful",
        ("P1",),
        "registered",
        "ROOTWU ends its own season on the event table's harvest flag, after the day's "
        "readers of P1 trwup and of its rwu: TSS = RWU = 0 (ROOTWU SEASINIT; between harvest and sowing "
        "XHLAI = 0 and ROOTWU is not called, so the reset at harvest gives the sowing-day state) and "
        "P1 trwup = 0, which the next morning's uptake limit reads (DSSATDRV exit TRWUP = 0 on the 7 "
        "CA-TPA harvest days); declared season_end in PHASED_WRITES, after every consumer of P1 and rwu",
    ),
    DayEntry(
        "17",
        "ledger",
        "ledger.close",
        "",
        ("P11",),
        "adapter",
        "soil column and pond only (no snow storage, sublimation or irrigation yet)",
    ),
)


# ------------------------------------------------------------------------ the planned rows
def _post(
    row: str,
    phase: str,
    entry: str,
    key: str,
    ports_out: tuple[str, ...],
    after: str,
    note: str,
    supersedes: tuple[str, ...] = (),
) -> DayEntry:
    return DayEntry(row, phase, entry, key, ports_out, "none", note, "post_m3", after, supersedes)


#: the planned rows in the RZWQM2 4.6 order (RZWQM2 file:line references are to the 4.5 source tree,
#: which is the tree the 4.6 binary is built from: it prints the 4.6 version banner)
POST_M3_DAY_TABLE: tuple[DayEntry, ...] = (
    _post(
        "D1",
        "management",
        "drainage.control",
        "drainage/control@rzwqm2-4.6:faithful",
        ("P23",),
        "events.apply",
        "drain geometry and the dated headgate depth "
        "(RZWQM2 Rzman.for:2218 THGDEP; Rzmain.for:5625 schedule)",
    ),
    _post(
        "D2",
        "physcl",
        "drainage.tile",
        "drainage/hooghoudt@rzwqm2-4.6:alt_daily",
        ("P4",),
        "soil_water.uptake_limit",
        "daily tile sink into P4.tile from yesterday's water table; the faithful tile is a sub-step sink "
        "kernel inside the soil-water day (RZWQM2 Rzday.for:4138 TILEFLO, 2044-2062), "
        "where this row is a no-op",
    ),
    _post(
        "H1",
        "physcl",
        "soil_heat.heatfx",
        "soil_heat/heatfx@rzwqm2-4.6:faithful",
        ("P14", "P15"),
        "soil_water.day",
        "soil heat after the water day (reference: inside the water loop after REDIST, Rzday.for:2101, "
        "2110-2121; faithful form reads P13); writes P15 zeros (ISHAW = 0 has no ice)",
    ),
    _post(
        "E1",
        "physcl",
        "erosion.gleams",
        "erosion/gleams@rzwqm2-4.6:faithful",
        ("P24",),
        "soil_heat.heatfx",
        "sediment yield of the day (reference OMSEA(113), line to confirm); no module yet: P24 zeros",
    ),
    _post(
        "N1",
        "physcl",
        "soil_om.transport",
        "soil_om/transport@rzwqm2-4.6:faithful",
        (),
        "erosion.gleams",
        "solute (NO3) transport with the day's water (P12 daily, P13 per step); the reference moves "
        "chemicals inside the PHYSCL water loop (inferred from the source, Rzrich.for:899-1115; not yet "
        "confirmed on a reference dump)",
    ),
    _post(
        "N2",
        "chem",
        "soil_om.day",
        "soil_om/apsim_nutrient@apsimng-05c84040:faithful",
        ("P17", "P19"),
        "^chem",
        "surface residue, organic matter and mineral nitrogen of the day, one entry (the module's own "
        "order inside); reference position: the nutrient call after PHYSCL and before MAPLNT "
        "(Rzmain.for:1546, "
        "1887; the call line to confirm)",
    ),
    _post(
        "N3",
        "plant",
        "crops.{slot}.n_remap_in",
        "crop_iface/n_remap_in@rzwqm2-4.6:faithful",
        ("P20",),
        "crops.{slot}.remap_in",
        "node mineral N onto the crop layers, conserving mass (as remap_in for the water)",
        supersedes=("n_supply.{slot}.replay",),
    ),
    _post(
        "N4",
        "plant",
        "crops.{slot}.publish_n_uptake",
        "crop_iface/publish_n_uptake@rzwqm2-4.6:faithful",
        ("P22",),
        "crops.{slot}.publish_uptake",
        "the crop layers' N uptake (P21) onto the soil nodes, read by the N module the next day",
    ),
    _post(
        "R1",
        "plant",
        "crops.{slot}.residue_out",
        "crop/ceres_maize.residue_out@dssat-4.8.6.0:faithful",
        ("P18",),
        "crops.{slot}.publish_n_uptake",
        "senesced matter and harvest residue to the soil (DSSAT-CSM SENESCE, HARVRES); "
        "before any harvest reset",
    ),
    _post(
        "P0",
        "plant",
        "crops.{slot}.status",
        "crop/ceres_maize.status@rzwqm2-4.6:faithful",
        ("P25",),
        "crops.{slot}.residue_out",
        "crop quantities for the phosphorus module (reference OMSEA(13, 40-45, 62-64))",
    ),
    _post(
        "P1",
        "post_plant",
        "phosphorus.day",
        "phosphorus/rzwqm2p@rzwqm2-4.6:faithful",
        (),
        "^post_plant",
        "the RZWQM2-P day, one entry (FERTILIZER ... PBALANCE inside); reference position: after MAPLNT "
        "(Rzmain.for about 2060-2275); no write-back into water, nitrogen or crop state",
    ),
)

#: planned ports that an M3 entry writes once its post-M3 variant is used (not in the M3 row)
POST_M3_ENTRY_PORTS: dict[str, tuple[str, ...]] = {
    "soil_water.day": ("P12", "P13"),
    "crops.{slot}.growth": ("P21",),
}


_HARVEST_END = "at the end of a harvest day (the event table's harvest flag), after the day's readers"

#: the owners' declared extra writes (``{slot}`` form): an entry of the owning module that writes a
#: field besides the module's producer entry of it, with the write's execution phase and meaning
#: (module docstring; AJ013 and :meth:`agrijax.core.day.Day.check`)
PHASED_WRITES: tuple[PhasedWrite, ...] = (
    PhasedWrite(
        "water_supply.{slot}.season_end",
        "iface.crop_water.{slot}.trwup",
        "season_end",
        "TRWUP = 0 " + _HARVEST_END + " (the crop read the day's non-zero TRWUP); the next morning's "
        "uptake limit (WUF, lag 1) reads the 0 (RZWQM2 4.5 DSSATDRV.for:1922-1931, 2020-2027)",
    ),
    PhasedWrite(
        "water_supply.{slot}.season_end",
        "water_supply.{slot}.rwu",
        "season_end",
        "RWU = 0 (ROOTWU SEASINIT) " + _HARVEST_END + " (the layer -> node publish read the day's uptake)",
    ),
    PhasedWrite(
        "water_supply.{slot}.season_end",
        "water_supply.{slot}.tss",
        "season_end",
        "TSS = 0 (ROOTWU SEASINIT) " + _HARVEST_END + "; ROOTWU is not called until the next sowing",
    ),
    PhasedWrite(
        "crops.{slot}.harvest",
        "iface.root.{slot}",
        "season_end",
        "the no-crop root record (XHLAI = RLV = RTDEP = 0) " + _HARVEST_END + "; ROOTWU reads it "
        "the next morning (lag 1)",
    ),
    PhasedWrite(
        "crops.{slot}.harvest",
        "iface.canopy.{slot}",
        "season_end",
        "bare soil (LAI = TLAI = HEIGHT = 0) " + _HARVEST_END + "; PET reads it the next morning (lag 1; "
        "the MAPLNT exit canopy is 0 on the 7 CA-TPA harvest days in the reference run's dump tables)",
    ),
)


#: module state (not a port) with a declared extra write: ``{path: (producer entry, readers of other
#: modules)}`` (``{slot}`` form). AJ013 orders the declared reset after the producer and the readers,
#: as for a port field
SHARED_STATE: dict[str, tuple[str, tuple[str, ...]]] = {
    # the layer -> node publish maps ROOTWU's layer uptake to the nodes (P3), row 16
    "water_supply.{slot}.rwu": ("water_supply.{slot}.rootwu", ("crops.{slot}.publish_uptake",)),
    # ROOTWU's own carried counter: no reader of another module
    "water_supply.{slot}.tss": ("water_supply.{slot}.rootwu", ()),
}
