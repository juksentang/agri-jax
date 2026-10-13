"""The water-supply slot: root water uptake between the soil and the crop (entry 10 of the coupling
contract's day, :data:`agrijax.iface.contract.DAY_TABLE`).

* :mod:`.rootwu` - DSSAT ``ROOTWU`` potential root water uptake on the crop layers
  (``water_supply/rootwu@dssat-4.8.6.0:faithful``, entry ``water_supply.<slot>.rootwu``; P2 in,
  P1 ``sw`` in, P1 ``trwup`` out; state ``water_supply.<slot>``).

The slot imports only :mod:`agrijax.core` and the port records :mod:`agrijax.iface` (lint AJ008).
"""

from .rootwu import (
    ROOTWU_COEFFICIENTS,
    RWU_SWCON1,
    RWU_SWCON3,
    RootwuCoefficients,
    RootwuParams,
    RootwuResult,
    RootwuState,
    SoilView,
    rootwu_estimate,
    rootwu_supply,
    rootwu_trwup_replay,
)

__all__ = [
    "ROOTWU_COEFFICIENTS",
    "RWU_SWCON1",
    "RWU_SWCON3",
    "RootwuCoefficients",
    "RootwuParams",
    "RootwuResult",
    "RootwuState",
    "SoilView",
    "rootwu_estimate",
    "rootwu_supply",
    "rootwu_trwup_replay",
]
