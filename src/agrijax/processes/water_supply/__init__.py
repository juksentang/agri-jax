"""The water-supply slot: root water uptake between the soil and the crop (entries 6, 10, 16 and 16b
of the coupling contract's day, :data:`agrijax.iface.contract.DAY_TABLE`).

* :mod:`.rootwu` - DSSAT ``ROOTWU`` potential root water uptake on the crop layers
  (``water_supply/rootwu@dssat-4.8.6.0:faithful``, entry ``water_supply.<slot>.rootwu``; P2 in,
  P1 ``sw`` in, P1 ``trwup`` out; state ``water_supply.<slot>``).
* :mod:`.publish` - the day's layer uptake mapped to the soil nodes as the RZWQM2 crop driver does
  it, with its ``SW == LL`` quirk (``crop_iface/publish_uptake@rzwqm2-4.6:faithful``, entry
  ``crops.<slot>.publish_uptake`` of the contract's day, bound on the producer's state
  ``water_supply.<slot>``; P3 out, P1 and P4 in).
* :mod:`.season` - the producer's own end of season on a harvest day (``TSS = RWU = 0`` and P1
  ``trwup = 0`` at the end of the plant phase; ``water_supply/rootwu_season_end@rzwqm2-4.6:faithful``,
  entry ``water_supply.<slot>.season_end``; a module resets only its own state).
* :mod:`.uptake_limit` - the RZWQM2 morning limit ``WUF = min(1, PET / TRWUP)`` of yesterday's
  node uptake (``soil_water/wuf@rzwqm2-4.6:faithful``, entry ``soil_water.uptake_limit``; P3, P1
  ``trwup`` (both one day old), P5, P7 in, P4 ``uptake`` out). The module holds no state of its
  own; every field of :class:`~.uptake_limit.UptakeLimitState` is a port.

The slot imports only :mod:`agrijax.core` and the port records :mod:`agrijax.iface` (lint AJ008).
"""

from .publish import PublishUptakeParams, rzwqm_publish_uptake
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
from .season import rootwu_season_end
from .uptake_limit import UptakeLimitParams, UptakeLimitState, rzwqm_uptake_limit, uptake_limit_factor

__all__ = [
    "ROOTWU_COEFFICIENTS",
    "RWU_SWCON1",
    "RWU_SWCON3",
    "PublishUptakeParams",
    "RootwuCoefficients",
    "RootwuParams",
    "RootwuResult",
    "RootwuState",
    "SoilView",
    "UptakeLimitParams",
    "UptakeLimitState",
    "rootwu_estimate",
    "rootwu_season_end",
    "rootwu_supply",
    "rootwu_trwup_replay",
    "rzwqm_publish_uptake",
    "rzwqm_uptake_limit",
    "uptake_limit_factor",
]
