"""End of season of the ``ROOTWU`` producer on a harvest day (day entry 16b of the coupling contract,
:data:`agrijax.iface.contract.DAY_TABLE`).

The uptake producer re-initialises its **own** state from the event table: on a harvest day, at
the end of the plant phase (after the crop and the layer -> node publish have read that day's
uptake), :func:`rootwu_season_end` sets the producer's carried state to its ``SEASINIT`` values
(``TSS = RWU = 0``) and its output ``TRWUP`` of the crop water record (P1) to 0. No other module
writes this state or this field on a harvest day; the crop's own season end
(:func:`agrijax.processes.crop.ceres_maize.season.ceres_harvest`, entry 16a) writes only the crop
slot's subtree and out ports.

Why at the harvest and at the end of the day:

* ``TRWUP``: on the harvest day the embedded crop runs with ROOTWU's ``TRWUP`` (its ROOTWU exit
  ``TRWUP`` is not 0), and at the end of that day's call the RZWQM2 crop driver sets ``TRWUP = 0``
  [RZWQM2 4.5 DSSATDRV.for:1922-1931, 2020-2027, conventions]; the next morning's uptake limit
  (entry 6, ``WUF = min(1, PET / TRWUP)``) reads that 0 (DSSATDRV exit ``TRWUP = 0`` on the 7
  harvest days of CA-TPA 2015-2021 while the ROOTWU exit ``TRWUP > 0``, dump tables of the
  instrumented RZWQM2 4.6 build).
* ``TSS``, ``RWU``: DSSAT re-initialises ROOTWU with the crop's season (``SEASINIT``,
  ``TSS = RWU = 0``) [DSSAT-CSM v4.8.6.0 SPAM/ROOTWU.for, SEASINIT]; RZWQM2 does so on the next
  sowing day [RZWQM2 4.5 DSSATDRV.for:1152-1255]. Between the harvest and the next sowing the root
  record has ``XHLAI = 0``, so ROOTWU is not called and its state is not touched: the reset at the
  harvest gives the same state on the sowing day (the same dump tables: ROOTWU entry ``TSS = 0`` on
  every sowing day).

The reset is the producer's own entry, not a write of the crop's harvest entry into the producer's
state: a field has one owning module, and the crop's entries write only the crop slot's subtree and
its out ports.

DSSAT-CSM is distributed under the BSD 3-clause licence (Copyright 1998-2026 DSSAT Foundation,
University of Florida, International Fertilizer Development Center). RZWQM2 is read for its
conventions only (no statement is reproduced).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from agrijax.core.events import EventTable
from agrijax.core.process import process

from .rootwu import RootwuState

__all__ = ["rootwu_season_end"]


@process(
    reads=("tss", "rwu", "water.trwup"),
    writes=("tss", "rwu", "water.trwup"),
    source=(
        "RZWQM2 4.6 driver of the embedded DSSAT crop, end of the harvest day (TRWUP = 0), with "
        "DSSAT-CSM v4.8.6.0 SPAM/ROOTWU.for SEASINIT (TSS = RWU = 0)"
    ),
    fortran_name="ROOTWU",
    key="water_supply/rootwu_season_end@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="dssat_layers",
    ref_build=(
        "RZWQM2 4.6 main_ryzen5_avx512 (instrumented build that dumps the daily entry and exit tables, "
        "CA-TPA 2015-2023)"
    ),
    sources=(
        (
            "the crop water uptake TRWUP is zeroed at the end of the harvest day",
            "RZWQM2 4.5 source DSSATDRV.for:1922-1931, 2020-2027 (read for conventions)",
        ),
        (
            "DSSATDRV exit TRWUP = 0 on the 7 harvest days of CA-TPA 2015-2021 while ROOTWU exit TRWUP > 0",
            "dump tables rzwqm46_catpa2015_2023 of the instrumented RZWQM2 4.6 build; "
            "tests/integration/test_day_rzwqm46_smoke.py",
        ),
        (
            "ROOTWU SEASINIT TSS = RWU = 0 with the crop's season",
            "DSSAT-CSM v4.8.6.0 SPAM/ROOTWU.for SEASINIT (BSD-3); RZWQM2 4.5 DSSATDRV.for:1152-1255",
        ),
    ),
    deviates=(
        (
            "TSS and RWU are reset at the harvest instead of at the next sowing",
            "between the two the root record has XHLAI = 0, so ROOTWU does not run and its state is "
            "not read; the reset at the harvest needs no start-of-day write before the producer runs",
            "tests/integration/test_ceres_catpa_seasons.py (ROOTWU TSS = 0 at each harvest's end, "
            "as the reference ROOTWU entry TSS of each sowing day, in the dump tables of the instrumented "
            "RZWQM2 4.6 build)",
        ),
    ),
)
def rootwu_season_end(state: RootwuState, params: Any, forcing_t: EventTable) -> RootwuState:
    """On a harvest day (the event table's ``harvest`` flag): ``tss = rwu = 0`` (``SEASINIT``) and
    ``water.trwup = 0`` (P1); other days leave the state unchanged. ``params`` is not read.

    Source: RZWQM2 4.5 DSSATDRV.for:1922-1931, 2020-2027 (conventions); DSSAT-CSM v4.8.6.0
    SPAM/ROOTWU.for SEASINIT (BSD-3).
    """
    h = jnp.asarray(forcing_t.harvest)

    def zero(x: Any) -> Any:
        return jnp.where(h, jnp.zeros_like(x), x)

    trwup = state.water.trwup
    return eqx.tree_at(
        lambda s: (s.tss, s.rwu, s.water.trwup), state, (zero(state.tss), zero(state.rwu), zero(trwup))
    )
