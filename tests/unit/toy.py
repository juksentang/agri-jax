"""A two-field toy model shared by the unit tests: water bucket + biomass.

Not a crop model; just enough structure to exercise @process, Model and the runtime.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core import Forcing, Model, Params, State, field, process


class ToyState(State):
    water: Array = field(unit="cm", description="bucket water", fortran_name="SW")
    biomass: Array = field(unit="kg ha-1", description="above-ground biomass")


class ToyParams(Params):
    k: Array = field(unit="d-1", description="fraction of water lost per day")
    rue: Array = field(unit="kg ha-1 per MJ m-2", description="radiation use efficiency")


class ToyForcing(Forcing):
    rain: Array = field(unit="cm", dims="T")
    srad: Array = field(unit="MJ m-2 d-1", dims="T")


@process(reads=("water",), writes=("water",), register=False, source="toy")
def infiltrate(state: ToyState, params: ToyParams, forcing_t: ToyForcing) -> ToyState:
    """Add today's rain to the bucket.

    Source: toy model, no literature.
    """
    return eqx.tree_at(lambda s: s.water, state, state.water + forcing_t.rain)


@process(reads=("water", "biomass"), writes=("water", "biomass"), register=False, source="toy")
def grow(state: ToyState, params: ToyParams, forcing_t: ToyForcing) -> ToyState:
    """Lose a fraction of the water and turn the loss into biomass when water is positive.

    Source: toy model, no literature.
    """
    uptake = params.k * state.water
    gain = jnp.where(state.water > 0.0, params.rue * forcing_t.srad * uptake, 0.0)
    return eqx.tree_at(lambda s: (s.water, s.biomass), state, (state.water - uptake, state.biomass + gain))


def toy_model(outputs=("water", "biomass")) -> Model:
    return Model(ToyState, [infiltrate, grow], outputs=outputs, name="toy")


def toy_inputs(n_days: int = 10) -> tuple[ToyParams, ToyForcing, ToyState]:
    params = ToyParams(k=jnp.asarray(0.1), rue=jnp.asarray(2.0))
    t = jnp.arange(n_days, dtype=jnp.float64)
    forcing = ToyForcing(rain=jnp.where(t % 3 == 0, 1.0, 0.0), srad=15.0 + 5.0 * jnp.sin(t))
    state0 = ToyState(water=jnp.asarray(5.0), biomass=jnp.asarray(0.0))
    return params, forcing, state0
