"""Surface-side port records: P5 PET fluxes, P8 daily weather, P9 snow outputs.

:class:`PETFluxes` and :class:`DailyWeather` are defined here and re-exported unchanged from
:mod:`agrijax.processes.pet.daily` (their first home; both paths name the same classes).
:class:`SnowOut` is new (the M3 snow module will write it).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.state import Forcing, State, field

__all__ = ["DailyWeather", "PETFluxes", "SnowOut"]


class PETFluxes(State):
    """Potential fluxes written by the PET processes."""

    transpiration: Array = field(
        dims=(), unit="cm d-1", description="S-W potential transpiration", fortran_name="PET"
    )
    soil_evaporation: Array = field(
        dims=(), unit="cm d-1", description="S-W potential soil evaporation", fortran_name="PES"
    )
    residue_evaporation: Array = field(
        dims=(), unit="cm d-1", description="S-W potential residue evaporation", fortran_name="PER"
    )
    reference_short: Array = field(
        dims=(), unit="mm d-1", description="ASCE short-reference ET (grass)", fortran_name="ETO"
    )
    reference_tall: Array = field(
        dims=(), unit="mm d-1", description="ASCE tall-reference ET (alfalfa)", fortran_name="ETR"
    )
    eo_priestley_taylor: Array = field(
        dims=(), unit="mm d-1", description="DSSAT PETPT potential ET", fortran_name="EO"
    )


class DailyWeather(Forcing):
    """Daily weather forcing of the PET processes (time axis first when stacked)."""

    tmin: Array = field(unit="degC", dims="T", fortran_name="TMIN")
    tmax: Array = field(unit="degC", dims="T", fortran_name="TMAX")
    srad: Array = field(unit="MJ m-2 d-1", dims="T", fortran_name="RTS")
    rh: Array = field(unit="percent", dims="T", fortran_name="RH")
    wind_run: Array = field(unit="km d-1", dims="T", fortran_name="U")
    doy: Array = field(unit="d", dims="T", fortran_name="JDAY")


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
