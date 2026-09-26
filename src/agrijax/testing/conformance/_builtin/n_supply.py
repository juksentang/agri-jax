"""Conformance case of the crop nitrogen replay ``n_supply/forcing_replay@none:replay`` (port P10)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from agrijax.iface.crop import CropNIn
from agrijax.processes.n_supply import CropNReplayForcing, CropNReplayState

from ..case import ConformanceCase

N_CROP, N_DAYS = 2, 3


def make(rng: np.random.Generator, dtype: Any, variant: str) -> tuple[Any, Any, Any]:
    """The replay's record (``NSTRES = 1``) and a stressed daily series."""
    state = CropNReplayState(n_out=CropNIn(nstres=jnp.ones(N_CROP, dtype)))
    forcing = CropNReplayForcing(nstres=jnp.asarray(rng.uniform(0.3, 1.0, N_DAYS), dtype))
    return state, None, forcing


def cases() -> list[ConformanceCase]:
    return [
        ConformanceCase(
            key="n_supply/forcing_replay@none:replay",
            make=make,
            n_days=N_DAYS,
            ports={"n_out": "iface.crop_n.{slot}"},
            no_balance="a replay of the reference run's stress factor: no conserved quantity",
            grad=None,
            no_grad="a replay copies the forcing: it has no parameter to differentiate",
            forcing_fields=("nstres",),
        )
    ]
