"""Season start of CERES-Maize in a multi-season run (an entry of the coupling contract).

:func:`ceres_season_init` (``crops.<slot>.season_init``, management phase, before the day's soil
physics), driven by the event table's ``sow`` flag (:class:`agrijax.core.events.EventTable`, bound
as the process's forcing): it writes the day's ``sow`` flag to the crop's ``sowing`` leaf, which is
what starts the crop that day (phenology leaves stage 7 and the roots start on the flag's day, not
on ``YRPLT``; :func:`~.phenology.sowing_day`), and on a sowing day the crop's own subtree becomes the
``SEASINIT`` state of its current season (:meth:`~.state.CeresMaizeState.initial`), leaf by leaf
through a mask. It writes nothing outside the crop's subtree. The crop state must be event-driven
(``CeresMaizeState.initial(..., events=True)``); a state without the ``sowing`` leaf is rejected when
the process is traced, so an event table cannot be silently ignored.

The season index: ``season`` counts the seasons that have ended and picks the row of the
per-season parameters (:class:`~.state.CeresSeasons`); the ``SEASINIT`` state carries the management
of that season (population, sowing depth, row spacing), and the crop processes leave it alone until
the next ``sow`` flag. In an event-driven run ``YRPLT`` of the table is not read;
:func:`check_sowing_dates` compares it with the flags for runs that want both to agree. Without a
table (``season`` is ``None``) the scalar parameters are the only season.

DSSAT-CSM is distributed under the BSD 3-clause licence (Copyright 1998-2026 DSSAT Foundation,
University of Florida, International Fertilizer Development Center); the ``SEASINIT`` state here
is an independent implementation of its published initialisation.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.events import EventTable
from agrijax.core.process import process

from .state import NOT_REACHED, CeresMaizeParams, CeresMaizeState

__all__ = ["OWN_FIELDS", "ceres_season_init", "check_sowing_dates", "fresh_state"]

#: the crop's own subtree (every non-port field that a season boundary resets)
OWN_FIELDS: tuple[str, ...] = ("phen", "stress", "growth", "roots")


def fresh_state(state: CeresMaizeState, params: CeresMaizeParams, season: Any) -> CeresMaizeState:
    """The ``SEASINIT`` state of season ``season`` for the crops of ``state`` (same number of crops,
    layers and dtype); the ports come from :meth:`~.state.CeresMaizeState.initial`.

    Source: DSSAT-CSM v4.8.6.0 MZ_CERES.for:452-597 (DYNAMIC = SEASINIT), MZ_PHENOL.for:171,
    MZ_GROSUB.for:376, MZ_ROOTS.for:81 (BSD-3).
    """
    lai = state.growth.lai
    return CeresMaizeState.initial(
        params, int(lai.shape[-1]), dtype=lai.dtype, season=season, events=state.sowing is not None
    )


def _own(state: CeresMaizeState) -> dict[str, Any]:
    """``{field: subtree}`` of the crop's own subtree (:data:`OWN_FIELDS`).

    Source: this implementation: the non-port fields that a season boundary resets; the ports
    ``root_out`` and ``canopy_out`` are left alone by :func:`ceres_season_init`.
    """
    return {n: getattr(state, n) for n in OWN_FIELDS}


def _masked(mask: Array, new: Any, old: Any) -> Any:
    """``new`` where ``mask`` else ``old``, leaf by leaf (both branches finite).

    Source: this implementation: both boundary entries select the ``SEASINIT`` state with a leaf-by-leaf
    mask on the event flag, not with a Python branch.
    """
    return jax.tree_util.tree_map(lambda a, b: jnp.where(mask, a, b), new, old)


@process(
    reads=(*OWN_FIELDS, "season"),
    writes=(*OWN_FIELDS, "sowing"),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_CERES.for, DYNAMIC = SEASINIT (BSD-3)",
    fortran_name="MZ_CERES",
    key="crop/ceres_maize.season_init@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        (
            "SEASINIT state of the crop (phenology, stress, growth, roots)",
            "MZ_CERES.for:452-597; MZ_PHENOL.for:171; MZ_GROSUB.for:376; MZ_ROOTS.for:81",
        ),
        (
            "the season starts on the sowing day, before that day's crop rates",
            "this implementation: the SEASINIT state is selected on the event table's sow flag at the "
            "start of the sowing day",
        ),
        (
            "per-season management: the row of the crop's season",
            "this implementation: a multi-season run carries one row of planting parameters per season "
            "(CeresSeasons), picked by the crop's season index",
        ),
    ),
    deviates=(
        (
            "the sowing day is the event table's sow flag and the SEASINIT state is selected leaf by "
            "leaf with a mask instead of a separate SEASINIT call",
            "the three process rules (no Python branch on a traced flag; the runtime owns the scan)",
            "tests/unit/test_ceres_seasons.py (on a sowing day the result is CeresMaizeState.initial "
            "bit for bit; on other days the state is unchanged)",
        ),
    ),
)
def ceres_season_init(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: EventTable
) -> CeresMaizeState:
    """Start a season on a sowing day: ``sowing`` is the day's ``sow`` flag (the crop starts on
    that day, :func:`~.phenology.sowing_day`) and the crop's own subtree becomes the ``SEASINIT``
    state of its current season (``season``); other days leave the own
    subtree unchanged and set ``sowing`` to false.

    Forcing read: ``sow`` (the event table's sowing flag). Ports are not touched: the root and
    canopy records of a sowing morning are the no-crop records, so the lagged reads of ROOTWU and
    PET stay lags.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_CERES.for, DYNAMIC = SEASINIT (BSD-3).
    """
    if state.sowing is None:  # static: the structure of the state
        raise ValueError(
            "ceres_season_init needs an event-driven crop state (CeresMaizeState.initial(..., "
            "events=True)): without the sowing leaf the crop would start on YRPLT and ignore the "
            "event table's sow flags"
        )
    sow = jnp.asarray(forcing_t.sow).astype(bool)
    fresh = fresh_state(state, params, state.season)
    return state.replace(**_masked(sow, _own(fresh), _own(state)), sowing=sow)


def check_sowing_dates(sow: Any, yrdoy: Any, seasons: Any) -> None:
    """Host-side: raise ``ValueError`` unless the event sowing days are the season table's dates.

    The ``k``-th ``sow`` flag falls on ``seasons.yrplt[k]`` and the rows after the last flag are
    :data:`~.state.NOT_REACHED`. NumPy, for the inputs of a run; an event-driven crop starts on
    the flags whatever the table says, so this is the check for runs that carry both (DSSAT input
    compatibility).

    Source: this implementation: the sowing decision is the event table's; ``YRPLT`` stays only for DSSAT
    input compatibility.
    """
    flags = np.asarray(sow).astype(bool)
    days = np.asarray(yrdoy).astype(np.int64)
    _require(flags.shape == days.shape, f"sow flags {flags.shape} and dates {days.shape} differ in shape")
    sown = [int(d) for d in days[flags]]
    table = [int(y) for y in np.asarray(seasons.yrplt).reshape(-1)]
    want = sown + [int(NOT_REACHED)] * (len(table) - len(sown))
    _require(
        len(sown) <= len(table) and table == want,
        f"event sowing days {sown} are not the season table's YRPLT {table}",
    )


def _require(ok: bool, message: str) -> None:
    """Raise ``ValueError(message)`` unless ``ok`` (a host-side input check).

    Source: this implementation: input consistency of an event-driven run (the flags and the season table
    must agree).
    """
    if not ok:
        raise ValueError(message)
