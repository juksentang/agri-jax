"""Conformance case of the soil-grid remapping (:mod:`agrijax.core.grids`) through a kit fixture.

A synthetic node grid to 150 cm (thin cells at the top, as RZWQM2 builds them), its ``LYRSET``
crop layers, three days of node water contents between wilting and saturation. The balance is
the thickness-weighted column total: the water on the crop layers changes by exactly the change
on the nodes (the intensive map preserves ``sum dz y`` where the layers cover the nodes).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from agrijax.core.grids import SoilGrid, remap_weights, rzwqm_lyrset
from agrijax.iface.crop import CropWaterIn

from ..case import Balance, ConformanceCase, Tolerance
from .grid_remap import KEY, RemapForcing, RemapParams, RemapState, remap_in_fixture

N_DAYS = 3
N_CROP = 1
NODE_THICKNESS = (1.0, 2.0, 2.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 15.0, 15.0, 20.0, 20.0, 30.0)
NODES = SoilGrid.from_thickness("rzwqm2_nodes", NODE_THICKNESS)
LAYERS = rzwqm_lyrset(NODES)
_TOL64 = Tolerance(1e-12, 1e-12)
_TOL32 = Tolerance(1e-6, 1e-5)


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """Node water contents for three days; the record starts consistent with the first day's
    predecessor (``sw0 = W theta0``), so the balance holds from day 0."""
    n = NODES.n
    theta0 = rng.uniform(0.08, 0.4, n)
    days = np.clip(theta0 + rng.normal(0.0, 0.03, (N_DAYS, n)).cumsum(axis=0), 0.05, 0.45)
    w = remap_weights(NODES, LAYERS)
    state = RemapState(
        theta=jnp.asarray(theta0, dtype),
        water_out=CropWaterIn(
            sw=jnp.asarray(w @ theta0, dtype), eop=jnp.zeros(N_CROP, dtype), trwup=jnp.zeros(N_CROP, dtype)
        ),
    )
    return state, RemapParams(nodes=NODES, layers=LAYERS), RemapForcing(theta=jnp.asarray(days, dtype))


def _layer_water(state: Any, params: Any) -> Any:
    return np.sum(np.asarray(state.water_out.sw, np.float64) * params.layers.thickness)


def _node_change(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    dz = params.nodes.thickness
    new = np.sum(np.asarray(after.theta, np.float64) * dz)
    return new - np.sum(np.asarray(before.theta, np.float64) * dz)


def _zero(before: Any, after: Any, params: Any, forcing_t: Any) -> Any:
    return 0.0


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key=KEY,
            make=make,
            process=remap_in_fixture,
            n_days=N_DAYS,
            own="crops.{slot}.remap_in",
            slot_contract="crop_iface",
            ports={"water_out": "iface.crop_water.{slot}"},
            balances=(
                Balance(
                    "thickness-weighted column water (layers against nodes)",
                    "cm",
                    storage=_layer_water,
                    inflow=_node_change,
                    outflow=_zero,
                    tol64=_TOL64,
                    tol32=_TOL32,
                ),
            ),
            grad=None,
            no_grad=(
                "a linear map with constant weights: its gradient is the weight matrix, tested in "
                "tests/unit/test_grids.py"
            ),
            forcing_fields=("theta",),
            # remap is an einsum: batched, XLA may sum the overlaps in another order (4 ulp)
            transforms_exact=False,
        )
    ]
