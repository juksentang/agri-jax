"""Conformance cases of the water-supply slot: the ``ROOTWU`` producer
``water_supply/rootwu@dssat-4.8.6.0:faithful``, its end of season
``water_supply/rootwu_season_end@rzwqm2-4.6:faithful``, the RZWQM2 uptake limit
``soil_water/wuf@rzwqm2-4.6:faithful`` and the layer -> node publish
``crop_iface/publish_uptake@rzwqm2-4.6:faithful`` (all in ``processes/water_supply``).

ROOTWU:

Synthetic soil on five crop layers and two crops: a nominal soil between the lower limit and
saturation, a waterlogged one (air-filled porosity below ``PORMIN`` with the saturation-day
counter past its delay, so the excess-water factor ``SWEXF`` is active) and, for gradients only,
a bare-canopy one (``XHLAI = 0``: ``ROOTWU`` is not called).
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.events import EventTable
from agrijax.core.grids import SoilGrid
from agrijax.iface.crop import CropWaterIn, RootRecord
from agrijax.iface.soil import NodeUptake, SinkInputs
from agrijax.iface.surface import PETFluxes
from agrijax.processes.water_supply import (
    ROOTWU_COEFFICIENTS,
    PublishUptakeParams,
    RootwuParams,
    RootwuState,
    UptakeLimitParams,
    UptakeLimitState,
)

from ..case import ConformanceCase, GradSpec

N_CROP, N_LAYER = 2, 5
DLAYR = (5.0, 10.0, 15.0, 20.0, 30.0)


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Soil layers, root record and crop water record for ``variant`` (nominal, wet, no_canopy)."""
    ll = rng.uniform(0.06, 0.14, N_LAYER)
    sat = ll + rng.uniform(0.25, 0.33, N_LAYER)
    pormin = np.full(N_CROP, 0.05)
    if variant == "wet":
        sw = sat - rng.uniform(0.005, 0.03, N_LAYER)  # air-filled porosity below PORMIN
        tss = rng.integers(3, 6, (N_CROP, N_LAYER)).astype(float)
    else:
        sw = ll + rng.uniform(0.1, 0.8, N_LAYER) * (sat - ll - 0.06)
        tss = rng.integers(0, 3, (N_CROP, N_LAYER)).astype(float)
    xhlai = np.zeros(N_CROP) if variant == "no_canopy" else rng.uniform(1.0, 4.0, N_CROP)
    root = RootRecord(
        rlv=jnp.asarray(rng.uniform(0.2, 3.0, (N_CROP, N_LAYER)), dtype),
        rtdep=jnp.asarray(rng.uniform(40.0, 80.0, N_CROP), dtype),
        rwumx=jnp.asarray(np.full(N_CROP, 0.03), dtype),
        pormin=jnp.asarray(pormin, dtype),
        xhlai=jnp.asarray(xhlai, dtype),
    )
    water = CropWaterIn(
        sw=jnp.asarray(sw, dtype),
        eop=jnp.asarray(rng.uniform(2.0, 6.0, N_CROP), dtype),
        trwup=jnp.zeros(N_CROP, dtype),
    )
    state = RootwuState(
        tss=jnp.asarray(tss, dtype), rwu=jnp.zeros((N_CROP, N_LAYER), dtype), root=root, water=water
    )
    params = RootwuParams(
        dlayr=jnp.asarray(DLAYR, dtype),
        ll=jnp.asarray(ll, dtype),
        sat=jnp.asarray(sat, dtype),
        coefficients=ROOTWU_COEFFICIENTS.as_arrays(dtype),
    )
    return state, params, None


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="water_supply/rootwu@dssat-4.8.6.0:faithful",
            make=make,
            variants=("nominal", "wet"),
            ports={"root": "iface.root.{slot}", "water": "iface.crop_water.{slot}"},
            no_balance=(
                "a potential uptake estimate: no stock changes here; the soil-water day removes the "
                "water and the water ledger closes it"
            ),
            grad=GradSpec(edge_variants=("no_canopy",)),
            coefficient_sets=("coefficients",),
            # replaying P1.sw from data: rwu within 4 ulp (5.6e-17) of the coupled run, measured on a
            # cluster CPU node (the two programs are compiled separately)
            binding_exact=False,
        ),
        ConformanceCase(
            key="water_supply/rootwu_season_end@rzwqm2-4.6:faithful",
            make=make_season_end,
            variants=SEASON_END_VARIANTS,
            own="water_supply.{slot}",
            ports={"water": "iface.crop_water.{slot}"},
            no_balance=(
                "a season end selects the SEASINIT values (TSS = RWU = 0) and zeroes the potential "
                "uptake TRWUP: no stock with daily flows"
            ),
            grad=None,
            no_grad=(
                "no parameter: the harvest flag only selects between the state and zeros (a gradient "
                "of 1 or 0 with respect to the state)"
            ),
            forcing_fields=("harvest",),
            coefficient_sets=(),
        ),
        ConformanceCase(
            key="soil_water/wuf@rzwqm2-4.6:faithful",
            make=make_limit,
            variants=("nominal", "no_trwup"),
            # the module has no state of its own; every field is a port (P7 and P4 sit in soil_water)
            own="water_supply.uptake_limit",
            ports={
                "theta": "soil_water.theta",
                "root_uptake": "iface.root_uptake.{slot}",
                "water": "iface.crop_water.{slot}",
                "pet": "iface.pet",
                "sink_in": "soil_water.sink_in",
            },
            no_balance=(
                "a limit on the next soil-water day's sink amount: no stock changes here; the soil-water "
                "day removes the water and the water ledger closes it"
            ),
            grad=None,
            no_grad=(
                "no calibratable parameter: the wilting point theta_wp only selects branches (zero "
                "gradient); gradients with respect to the input records (P3, P1 trwup, P5) are checked "
                "finite and against finite differences in tests/unit/test_water_supply.py"
            ),
            coefficient_sets=(),
        ),
        ConformanceCase(
            key="crop_iface/publish_uptake@rzwqm2-4.6:faithful",
            make=make_publish,
            variants=("nominal", "no_pet"),
            # bound on the ROOTWU producer's state, whose layer uptake rwu it reads
            own="water_supply.{slot}",
            slot_contract="water_supply",
            ports={
                "water": "iface.crop_water.{slot}",
                "root_uptake": "iface.root_uptake.{slot}",
                "sink_in": "soil_water.sink_in",
            },
            no_balance=(
                "a map of the day's layer uptake onto the nodes: the node amounts of the SW > LL layers "
                "sum to the layer amounts (tests/unit/test_water_supply.py); the soil-water day of the "
                "next day removes the water and the water ledger closes it"
            ),
            grad=None,
            no_grad=(
                "no calibratable parameter: the lower limit ll only selects branches (zero gradient) and "
                "the grids are static; gradients with respect to rwu are checked finite and against "
                "finite differences in tests/unit/test_water_supply.py"
            ),
            coefficient_sets=(),
        ),
    ]


# ------------------------------------------------------------------ the uptake limit (WUF)
N_NODE = 9
TL = (2.0, 3.0, 5.0, 5.0, 10.0, 10.0, 15.0, 20.0, 30.0)
#: crop layers on the node grid (bottoms 5, 15, 30, 60, 100 cm)
LAYER_DLAYR = (5.0, 10.0, 15.0, 30.0, 40.0)


def make_limit(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Yesterday's node uptake of two crops, their TRWUP, today's PET and the node water: crop 0
    limited (PET < TRWUP), crop 1 not (PET > TRWUP); two nodes below the wilting point. With
    ``no_trwup`` crop 0 had no potential uptake yesterday (the limit does not apply to it)."""
    theta_wp = rng.uniform(0.08, 0.12, N_NODE)
    theta = theta_wp + rng.uniform(0.02, 0.2, N_NODE)
    theta[[3, 7]] = theta_wp[[3, 7]] - rng.uniform(0.005, 0.02, 2)
    q = rng.uniform(0.0, 0.05, (N_CROP, N_NODE))
    pet = rng.uniform(0.2, 0.4)
    trwup = np.asarray([pet * rng.uniform(1.5, 3.0), pet * rng.uniform(0.3, 0.8)])
    if variant == "no_trwup":
        trwup[0] = 0.0
    z = jnp.zeros((), dtype)
    state = UptakeLimitState(
        theta=jnp.asarray(theta, dtype),
        root_uptake=NodeUptake(uptake=jnp.asarray(q, dtype)),
        water=CropWaterIn(
            sw=jnp.zeros(N_LAYER, dtype),
            eop=jnp.full(N_CROP, 10.0 * pet, dtype),
            trwup=jnp.asarray(trwup, dtype),
        ),
        pet=PETFluxes(
            transpiration=jnp.asarray(pet, dtype),
            soil_evaporation=jnp.asarray(rng.uniform(0.05, 0.2), dtype),
            residue_evaporation=z,
            reference_short=z,
            reference_tall=z,
            eo_priestley_taylor=z,
        ),
        sink_in=SinkInputs.zeros(N_NODE, dtype),
    )
    return state, UptakeLimitParams(theta_wp=jnp.asarray(theta_wp, dtype)), None


# ------------------------------------------------------------------ the layer -> node publish
def make_publish(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """The producer's layer uptake of two crops on five layers over nine nodes: layers above,
    below and exactly at the lower limit (the ``SW == LL`` quirk keeps the node value of the
    day's sink). With ``no_pet`` crop 1 has no potential transpiration today (no uptake)."""
    nodes = SoilGrid.from_thickness("rzwqm2_nodes", TL)
    layers = SoilGrid.from_thickness("rzwqm2_lyrset", LAYER_DLAYR)
    ll = np.round(rng.uniform(0.08, 0.14, N_LAYER), 3)
    sw = ll + rng.uniform(0.02, 0.15, N_LAYER)
    sw[1] = ll[1] - 0.01  # below LL: no uptake
    sw[3] = ll[3]  # SW == LL: the quirk
    eop = rng.uniform(2.0, 6.0, N_CROP)
    if variant == "no_pet":
        eop[1] = 0.0
    state = RootwuState(
        tss=jnp.zeros((N_CROP, N_LAYER), dtype),
        rwu=jnp.asarray(rng.uniform(0.0, 0.1, (N_CROP, N_LAYER)), dtype),
        root=cast(Any, None),  # the publish entry does not read the root record: left unbound
        water=CropWaterIn(
            sw=jnp.asarray(sw, dtype), eop=jnp.asarray(eop, dtype), trwup=jnp.zeros(N_CROP, dtype)
        ),
        root_uptake=NodeUptake.zeros(N_CROP, N_NODE, dtype),
        sink_in=SinkInputs.zeros(N_NODE, dtype).replace(
            uptake=jnp.asarray(rng.uniform(0.0, 0.05, N_NODE), dtype)
        ),
    )
    params = PublishUptakeParams(nodes=nodes, layers=layers, ll=jnp.asarray(ll, dtype))
    return state, params, None


# ------------------------------------------------------------------ ROOTWU's end of season
#: the harvest flag on the first day (``harvest``) or on no day (``ordinary``)
SEASON_END_VARIANTS = ("harvest", "ordinary")
#: days of the season-end case (the ConformanceCase default)
SEASON_END_DAYS = 3


def make_season_end(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """The ROOTWU state of :func:`make` (``nominal``) with a non-zero layer uptake and TRWUP, and an
    event table of ``SEASON_END_DAYS`` days whose harvest flag is set on day 0 (``harvest``) or never."""
    state, params, _ = make(rng, dtype, "nominal")
    rwu = jnp.asarray(rng.uniform(0.01, 0.1, (N_CROP, N_LAYER)), dtype)
    state = state.replace(rwu=rwu, water=state.water.replace(trwup=jnp.sum(rwu, axis=-1)))
    n_days = SEASON_END_DAYS
    on = np.arange(n_days) == 0 if variant == "harvest" else np.zeros(n_days, bool)
    events = EventTable.empty(n_days).replace(harvest=jnp.asarray(on))
    events = jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, events
    )
    return state, params, events
