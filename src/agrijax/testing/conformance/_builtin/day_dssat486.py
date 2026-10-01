"""Conformance case of the DSSAT day's crop-interface adapter ``crop_iface/layers_in@dssat-4.8.6.0:faithful``.

``crops.<slot>.layers_in`` of :mod:`agrijax.models.day_dssat486` hands the
soil's end-of-day layer water content (P7 ``soil_water.theta``, on the DSSAT layers) to the crop
water record (P1 ``sw``) unchanged: in DSSAT-CSM the crop layers are the soil layers; and the
soil's snow pack (``soil_water.snow``) to the crop's snow record (P9 ``swe``). The case gives a
nine-layer profile between the lower limit and saturation and a snow pack; the adapter writes
nothing else, so P1 ``eop`` and ``trwup`` and the other P9 fields pass through.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.state import Forcing, field
from agrijax.iface.crop import CropWaterIn
from agrijax.iface.surface import SnowOut
from agrijax.models.day_dssat486_adapters import LAYERS_IN_KEY, LayersInState, layers_in

from ..case import ConformanceCase

N_DAYS = 3
N_CROP = 1
N_LAYER = 9


class LayersInForcing(Forcing):
    """The adapter reads no forcing; the day index gives the runs their time axis."""

    day: Array = field(unit="d", description="day index of the run", dims=("T",))


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """A layer water content between 0.05 and 0.40, a snow pack, and crop records with another
    ``sw`` and ``swe``."""
    theta = rng.uniform(0.05, 0.40, N_LAYER)
    state = LayersInState(
        theta=jnp.asarray(theta, dtype),
        snow=jnp.asarray(rng.uniform(0.0, 20.0), dtype),
        snow_out=SnowOut.zeros(dtype).replace(swe=jnp.asarray(rng.uniform(0.0, 20.0), dtype)),
        water_out=CropWaterIn(
            sw=jnp.asarray(rng.uniform(0.05, 0.40, N_LAYER), dtype),
            eop=jnp.asarray(rng.uniform(1.0, 6.0, N_CROP), dtype),
            trwup=jnp.asarray(rng.uniform(0.1, 0.8, N_CROP), dtype),
        ),
    )
    return state, None, LayersInForcing(day=jnp.arange(N_DAYS, dtype=dtype))


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key=LAYERS_IN_KEY,
            make=make,
            process=layers_in,
            n_days=N_DAYS,
            own="crops.{slot}.layers_in",
            slot_contract=None,
            no_slot_contract=(
                "an assembly adapter of the DSSAT-CSM day (P7 theta and PD7 snow to P1 sw and P9 swe): "
                "its ports are those of agrijax.iface.contract.DSSAT_PORTS, not of the slot contracts of the "
                "RZWQM2 day (the crop_iface slot has no P9 output: in the RZWQM2 day the snow module "
                "writes P9)"
            ),
            ports={
                "theta": "soil_water.theta",
                "snow": "soil_water.snow",
                "water_out": "iface.crop_water.{slot}",
                "snow_out": "iface.snow",
            },
            no_balance=(
                "an identity map (P1 sw = P7 theta, P9 swe = SNOW): the layer water and the snow pack are "
                "handed over unchanged, checked "
                "bit for bit in tests/unit/test_day_dssat486.py"
            ),
            grad=None,
            no_grad="an identity map without parameters: d sw / d theta = I",
        )
    ]
