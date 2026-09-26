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

from dataclasses import dataclass
from typing import Literal

from agrijax.iface.contract import PORTS, PortSpec, port_spec

__all__ = ["SLOT_CONTRACTS", "SlotContract", "SlotPort", "slot_contract"]

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
    """The ports of one slot and the day phase its entries run in (M3 contract, section 1.2)."""

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


#: the slot contracts of the M3 assembly (M3 coupling contract, sections 1.2 and 2.2)
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
                SlotPort("P10", "in"),
            ),
            note="entries 11-15: phenology, stress, growth, roots read P1 (and P10); publish writes P2, P6",
        ),
        SlotContract(
            "water_supply",
            "plant",
            (SlotPort("P1", "inout"), SlotPort("P2", "in")),
            note=(
                "entry 10 (ROOTWU) reads P1.sw and writes P1.trwup; the replay producer "
                "water_supply/forcing_replay stands in for entries 8-10 and writes the whole P1 record"
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
            note="entry 4; the M3 PET processes still read their own surface subtree (gap G5)",
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
            ),
            ledger_channels=("infiltration", "runoff", "soil_evaporation", "transpiration", "drainage"),
            note="entries 6-7: the uptake limit writes P4.uptake, the soil-water day reads P4",
        ),
    )
}


def slot_contract(slot: str) -> SlotContract:
    """The :class:`SlotContract` of ``slot``."""
    try:
        return SLOT_CONTRACTS[slot]
    except KeyError:
        raise KeyError(f"no slot contract {slot!r} (slots: {sorted(SLOT_CONTRACTS)})") from None
