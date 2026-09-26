"""Surface-side port records: P5 PET fluxes, P8 daily weather, P9 snow outputs.

:class:`PETFluxes` and :class:`DailyWeather` are defined in :mod:`agrijax.processes.pet.daily`
and re-exported here unchanged. :class:`SnowOut` is new (the M3 snow module will write it).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.state import State, field
from agrijax.processes.pet.daily import DailyWeather, PETFluxes

__all__ = ["DailyWeather", "PETFluxes", "SnowOut"]


class SnowOut(State):
    """What the snow module passes on each day (P9, ``iface.snow``), per site.

    Written by the snow entry in the soil-physics phase and read the same day by the soil-water
    day (``melt`` enters infiltration as an event without breakpoints), by the crop (``swe`` as
    DSSAT's ``SNOW``, in ``mm`` as the crop reads it) and by the ledger. The all-zero record is
    no snow.
    """

    melt: Array = field(
        unit="cm d-1", description="snowmelt entering infiltration", fortran_name="AIRR", dims=()
    )
    melt_runoff: Array = field(unit="cm d-1", description="snowmelt runoff", fortran_name="SNRO", dims=())
    swe: Array = field(
        unit="mm", description="snow water equivalent (the crop's SNOW)", fortran_name="SNOW", dims=()
    )
    sublimation: Array = field(
        unit="cm d-1", description="sublimation (capped by the potential soil evaporation)", dims=()
    )

    @classmethod
    def zeros(cls, dtype: Any = None) -> SnowOut:
        """No snow."""
        z = jnp.zeros((), dtype=dtype if dtype is not None else jnp.result_type(float))
        return cls(melt=z, melt_runoff=z, swe=z, sublimation=z)
