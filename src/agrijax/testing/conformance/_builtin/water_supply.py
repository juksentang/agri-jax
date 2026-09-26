"""Conformance case of the ``ROOTWU`` producer ``water_supply/rootwu@dssat-4.8.6.0:faithful``.

Synthetic soil on five crop layers and two crops: a nominal soil between the lower limit and
saturation, a waterlogged one (air-filled porosity below ``PORMIN`` with the saturation-day
counter past its delay, so the excess-water factor ``SWEXF`` is active) and, for gradients only,
a bare-canopy one (``XHLAI = 0``: ``ROOTWU`` is not called).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from agrijax.iface.crop import CropWaterIn, RootRecord
from agrijax.processes.soil_water.uptake import ROOTWU_COEFFICIENTS, RootwuParams, RootwuState

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
            # replaying P1.sw from data: rwu within 4 ulp (5.6e-17) of the coupled run, measured on
            # rorqual (the two programs are compiled separately)
            binding_exact=False,
        )
    ]
