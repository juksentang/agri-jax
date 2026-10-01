"""Module declarations: what a slot implementation supports, needs and books (swap contract).

Two implementations of one slot (the RZWQM2 Richards day and the DSSAT tipping bucket in the
``soil_water`` slot, say) are interchangeable in an assembly when both talk to the rest of the
model only through the contract ports (:data:`agrijax.iface.contract.PORTS`) and each states,
machine-readably,

* which day entries it provides (registry keys, in the order they run) and where in the day
  another module must run between them;
* which processes of the reference it covers and which it does not (the scope);
* the parameters it needs, with units and the reference-model input names (they are **not**
  interchangeable between implementations: the bucket's ``SDUL`` is not the Brooks-Corey curve);
* the ports it reads and writes, its own input fields written by other modules, and the grid it
  runs on;
* its conservation responsibility: the ledger channels it books, what its storage is, and which
  amounts it only relays (booked by another module).

A :class:`ModuleDeclaration` is plain data (no JAX, no import of ``agrijax.processes``); the slot
package exposes one as ``MODULE``, and ``tests/unit/test_bucket.py`` checks the bucket's
declaration against its registry entries, field metadata and ledger channels.

Design rule: both soil-water modules are first-class, and swapping one for the other must not
require rewriting the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["LedgerDuty", "ModuleDeclaration", "ParameterDecl"]


@dataclass(frozen=True)
class ParameterDecl:
    """One parameter a module needs: its path in the module's params, unit, reference input name
    and meaning."""

    path: str
    unit: str
    reference_name: str
    description: str


@dataclass(frozen=True)
class LedgerDuty:
    """The module's conservation responsibility for one conserved quantity."""

    quantity: str
    storage: str
    inflows: tuple[str, ...]
    outflows: tuple[str, ...]
    relayed: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ModuleDeclaration:
    """What one implementation of a slot supports, needs and books (module docstring)."""

    slot: str
    name: str
    reference: str
    entries: tuple[tuple[str, str], ...]
    order: str
    grid: str
    supports: tuple[str, ...]
    not_supported: tuple[str, ...]
    parameters: tuple[ParameterDecl, ...]
    ports_in: tuple[tuple[str, str], ...]
    ports_out: tuple[tuple[str, str], ...]
    inputs_own: tuple[tuple[str, str], ...] = ()
    forcing: tuple[tuple[str, str], ...] = ()
    ledger: tuple[LedgerDuty, ...] = ()
    accuracy: str = ""
    notes: tuple[str, ...] = field(default=())

    def keys(self) -> tuple[str, ...]:
        """The registry keys of the module's day entries, in order."""
        return tuple(k for _, k in self.entries)
