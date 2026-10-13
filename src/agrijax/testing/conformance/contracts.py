"""Slot contracts: which ports of the coupling contract each slot reads and writes.

The ports themselves (global path, record class, fields with unit, dims and grid, time
semantics) are :data:`agrijax.iface.contract.PORTS`, the single machine-readable form of the
port table. A :class:`SlotContract` says, for one slot, which of those ports its processes may
read (``in``), write (``out``) or both (``inout``), and which fields of a written port they may
write. :func:`agrijax.testing.conformance.checks.check_slot_contract` holds a bound process to it:
its reads and writes stay inside its own subtree and the ports of its slot, it never writes an
``in`` port, and every bound port uses the contract's record class at the contract's path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

from agrijax.iface.contract import PORTS, PortSpec, port_spec

__all__ = [
    "POST_M3_SLOT_CONTRACTS",
    "POST_M3_SLOT_PORTS",
    "SLOT_CONTRACTS",
    "SlotContract",
    "SlotPort",
    "slot_contract",
]

Direction = Literal["in", "out", "inout"]


@dataclass(frozen=True)
class SlotPort:
    """One port of a slot: its id (``P1`` ..), the direction seen from the slot's processes, and
    the fields they may write (empty: every field of an ``out`` / ``inout`` port)."""

    port: str
    direction: Direction
    writes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        spec = port_spec(self.port)  # raises for an unknown port id
        if self.direction == "in" and self.writes:
            raise ValueError(f"{self.port}: an 'in' port has no writable fields")
        unknown = sorted(set(self.writes) - set(spec.field_map))
        if unknown:
            raise ValueError(f"{self.port}: unknown fields {unknown}")

    @property
    def spec(self) -> PortSpec:
        return PORTS[self.port]

    def writable(self, field_name: str | None) -> bool:
        """May the slot write field ``field_name`` of the port (``None``: the whole record)?"""
        if self.direction == "in":
            return False
        if not self.writes:
            return True
        return field_name is not None and field_name in self.writes


@dataclass(frozen=True)
class SlotContract:
    """The ports of one slot and the day phase its entries run in."""

    slot: str
    phase: str
    ports: tuple[SlotPort, ...]
    ledger_channels: tuple[str, ...] = ()
    note: str = ""

    def port(self, port_id: str) -> SlotPort | None:
        for p in self.ports:
            if p.port == port_id:
                return p
        return None


#: the slot contracts of the contract's day (:data:`agrijax.iface.contract.DAY_TABLE`)
SLOT_CONTRACTS: dict[str, SlotContract] = {
    c.slot: c
    for c in (
        SlotContract(
            "crop",
            "plant",
            (
                SlotPort("P1", "in"),
                SlotPort("P2", "out"),
                SlotPort("P6", "out"),
                SlotPort("P9", "in"),
                SlotPort("P10", "in"),
            ),
            note=(
                "entries 11-15: phenology, stress, growth, roots read P1 (and P10), phenology reads P9 swe; "
                "publish writes P2, canopy (15a) P6"
            ),
        ),
        SlotContract(
            "water_supply",
            "plant",
            (SlotPort("P1", "inout"), SlotPort("P2", "in"), SlotPort("P3", "out"), SlotPort("P4", "in")),
            note=(
                "entry 10 (ROOTWU) reads P1.sw and writes P1.trwup; the replay producer "
                "water_supply/forcing_replay stands in for entries 8-10 and writes the whole P1 record; "
                "entry 16 (publish_uptake, bound on the ROOTWU state water_supply.<slot>) reads rwu and "
                "the day's sink P4 and writes P3"
            ),
        ),
        SlotContract("n_supply", "plant", (SlotPort("P10", "out"),), note="replay producer of P10"),
        SlotContract(
            "crop_iface",
            "plant",
            (
                SlotPort("P1", "out", ("sw", "eop")),
                SlotPort("P3", "out"),
                SlotPort("P5", "in"),
                SlotPort("P7", "in"),
            ),
            note="entries 8, 9, 16: remap_in (P1.sw), eop (P1.eop from P5), publish_uptake (P3)",
        ),
        SlotContract(
            "pet",
            "physcl",
            (SlotPort("P5", "out"), SlotPort("P6", "in"), SlotPort("P7", "in")),
            note="entry 4: the PET processes read P6 (canopy) and P7 (theta), both lag 1, and write P5",
        ),
        SlotContract("snow", "physcl", (SlotPort("P5", "in"), SlotPort("P9", "out")), note="entry 5"),
        SlotContract(
            "soil_water",
            "physcl",
            (
                SlotPort("P3", "in"),
                SlotPort("P1", "in"),
                SlotPort("P4", "inout"),
                SlotPort("P5", "in"),
                SlotPort("P9", "in"),
                SlotPort("P7", "in"),
            ),
            ledger_channels=("infiltration", "runoff", "soil_evaporation", "transpiration", "drainage"),
            note=("entries 6-7: the uptake limit writes P4.uptake, the soil-water day reads P4"),
        ),
    )
}


#: the planned slots (stage ``post_m3`` of the coupling contract): their ports and day phases
POST_M3_SLOT_CONTRACTS: dict[str, SlotContract] = {
    c.slot: c
    for c in (
        SlotContract(
            "soil_heat",
            "physcl",
            (
                SlotPort("P12", "in"),
                SlotPort("P13", "in"),
                SlotPort("P9", "in"),
                SlotPort("P16", "in"),
                SlotPort("P14", "out"),
                SlotPort("P15", "out"),
            ),
            note="soil heat after the water day: HEATFX (ISHAW = 0, P15 zeros), SHAW later",
        ),
        SlotContract(
            "soil_om",
            "chem",
            (
                SlotPort("P7", "in"),
                SlotPort("P12", "in"),
                SlotPort("P13", "in"),
                SlotPort("P14", "in"),
                SlotPort("P18", "in"),
                SlotPort("P22", "in"),
                SlotPort("P17", "out"),
                SlotPort("P19", "out"),
            ),
            note="residue, organic matter and mineral N (soil_om.day), their transport (soil_om.transport)",
        ),
        SlotContract(
            "drainage",
            "physcl",
            (SlotPort("P12", "in"), SlotPort("P23", "inout"), SlotPort("P4", "out", ("tile",))),
            note="drainage.control writes P23; the daily variant writes P4.tile; the faithful tile sink is a "
            "sub-step kernel the RZWQM2 day passes to the soil-water day",
        ),
        SlotContract("erosion", "physcl", (SlotPort("P24", "out"),), note="GLEAMS-type sediment yield"),
        SlotContract(
            "phosphorus",
            "post_plant",
            (
                SlotPort("P7", "in"),
                SlotPort("P12", "in"),
                SlotPort("P14", "in"),
                SlotPort("P17", "in"),
                SlotPort("P18", "in"),
                SlotPort("P19", "in"),
                SlotPort("P24", "in"),
                SlotPort("P25", "in"),
            ),
            note="RZWQM2-P day: reads only, no write-back into water, nitrogen or crop state",
        ),
    )
}
#: planned ports added to the slots above (their processes may bind them once their ``post_m3``
#: variants exist)
POST_M3_SLOT_PORTS: dict[str, tuple[SlotPort, ...]] = {
    "crop": (SlotPort("P18", "out"), SlotPort("P20", "in"), SlotPort("P21", "out"), SlotPort("P25", "out")),
    "crop_iface": (
        SlotPort("P19", "in"),
        SlotPort("P20", "out"),
        SlotPort("P21", "in"),
        SlotPort("P22", "out"),
    ),
    "soil_water": (
        SlotPort("P12", "out"),
        SlotPort("P13", "out"),
        SlotPort("P15", "in"),
        SlotPort("P23", "in"),
    ),
    "pet": (SlotPort("P17", "in"),),
}


def slot_contract(slot: str) -> SlotContract:
    """The :class:`SlotContract` of ``slot``: a slot of the RZWQM2 4.6 day with its planned ports added
    (:data:`POST_M3_SLOT_PORTS`), or a planned slot (:data:`POST_M3_SLOT_CONTRACTS`)."""
    if slot in SLOT_CONTRACTS:
        base = SLOT_CONTRACTS[slot]
        extra = tuple(p for p in POST_M3_SLOT_PORTS.get(slot, ()) if base.port(p.port) is None)
        return dataclasses.replace(base, ports=(*base.ports, *extra)) if extra else base
    if slot in POST_M3_SLOT_CONTRACTS:
        return POST_M3_SLOT_CONTRACTS[slot]
    known = sorted({*SLOT_CONTRACTS, *POST_M3_SLOT_CONTRACTS})
    raise KeyError(f"no slot contract {slot!r} (slots: {known})")
