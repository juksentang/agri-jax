"""Surface-side port records: P5 PET fluxes, P8 daily weather, P9 snow outputs; and the post-M3
records P14 soil temperature, P16 hourly weather, P17 surface residue, P24 sediment (planned ports,
stage ``post_m3`` in :data:`.contract.PORTS`).

:class:`PETFluxes` and :class:`DailyWeather` are defined here and re-exported unchanged from
:mod:`agrijax.processes.pet.daily` (their first home; both paths name the same classes).
:class:`SnowOut` is new (the snow module writes it).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.state import Forcing, State, field

__all__ = [
    "EVAPORATION_DEMAND_FIELDS",
    "DailyWeather",
    "EvaporationRecord",
    "HourlyWeather",
    "PETFluxes",
    "ResidueRecord",
    "SedimentOut",
    "SnowOut",
    "SoilAlbedo",
    "SoilTemperature",
    "evaporation_demand",
]

#: the P5 fields whose sum is the soil-water day's evaporation demand. The reference's actual
#: evaporation (``.ana`` column 6) never exceeds ``PES + PER`` and equals it on the days that are
#: not supply-limited (compared with RZWQM2 4.6 outputs), so the residue evaporation is drawn from
#: the soil water too.
EVAPORATION_DEMAND_FIELDS: tuple[str, ...] = ("soil_evaporation", "residue_evaporation")


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

    @classmethod
    def zeros(cls, dtype: Any = None) -> PETFluxes:
        """No demand (the contract's default of P5)."""
        z = jnp.zeros((), dtype=dtype if dtype is not None else jnp.result_type(float))
        return cls(z, z, z, z, z, z)


def evaporation_demand(pet: PETFluxes) -> Array:
    """The soil-water day's daily evaporation demand from P5 [cm d-1]: ``PES + PER``
    (:data:`EVAPORATION_DEMAND_FIELDS`)."""
    return pet.soil_evaporation + pet.residue_evaporation


class DailyWeather(Forcing):
    """Daily weather forcing of the PET processes (time axis first when stacked).

    ``srad`` is the field radiation ``RTS`` (in RZWQM2 the daily re-sum of the hourly radiation
    after the direct / diffuse split); ``srad_horizontal`` is the measured horizontal radiation
    ``RTH`` of the weather file, which the Shuttleworth-Wallace PET uses in the cloudiness ratio
    of the net long-wave radiation and the clear-sky floor. Left out, it is ``srad``
    (the record then behaves as before the field existed).
    """

    tmin: Array = field(unit="degC", dims="T", fortran_name="TMIN")
    tmax: Array = field(unit="degC", dims="T", fortran_name="TMAX")
    srad: Array = field(unit="MJ m-2 d-1", dims="T", fortran_name="RTS")
    rh: Array = field(unit="percent", dims="T", fortran_name="RH")
    wind_run: Array = field(unit="km d-1", dims="T", fortran_name="U")
    doy: Array = field(unit="d", dims="T", fortran_name="JDAY")
    srad_horizontal: Array = field(
        unit="MJ m-2 d-1",
        dims="T",
        fortran_name="RTH",
        description="measured horizontal solar radiation (default: srad)",
        default=None,
    )

    def __post_init__(self) -> None:
        if self.srad_horizontal is None:  # construction-time default, not a traced branch
            object.__setattr__(self, "srad_horizontal", self.srad)


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


# ------------------------------------------------------------------------ post-M3 records (planned)
def _z(dtype: Any) -> Array:
    return jnp.zeros((), dtype=dtype if dtype is not None else jnp.result_type(float))


class SoilTemperature(State):
    """Soil temperature on the soil-water nodes (P14, ``iface.soil_temp``), written by the energy
    slot after the soil-water day and read the same day by the nutrient modules (nitrogen,
    phosphorus, organic matter kinetics)."""

    t: Array = field(
        unit="degC",
        description="soil temperature of each node",
        fortran_name="T",
        dims=("n_node",),
        grid="rzwqm2_nodes",
    )
    t_surface: Array = field(
        unit="degC", description="soil surface temperature", fortran_name="SRFTEMP", dims=()
    )

    @classmethod
    def constant(cls, n_node: int, value: Any, dtype: Any = None) -> SoilTemperature:
        """A uniform profile at ``value`` degC (test and replay fixtures)."""
        v = jnp.asarray(value, dtype=dtype if dtype is not None else jnp.result_type(float))
        return cls(t=jnp.broadcast_to(v, (n_node,)), t_surface=v)


class HourlyWeather(Forcing):
    """Hourly forcing derived from the daily weather (P16, ``forcing.hourly``): the reference's
    daily-to-hourly generator runs in the forcing preprocessing (NumPy, not differentiable, not a
    process), like the radiation reconstruction."""

    tair: Array = field(
        unit="degC", dims=("T", "hour"), fortran_name="HRT", description="hourly air temperature"
    )


class ResidueRecord(State):
    """The flat surface residue each day (P17, ``iface.residue``): written by the organic-matter
    slot after its day and read by the PET on the **next** day (the reference decomposes residue
    after the soil physics). The fields are those the Shuttleworth-Wallace PET reads (RZWQM2
    ``RM``, ``RESAGE``, ``WRES``, ``IPR``); ``kind`` codes 0 none, 1 corn, 2 soybean, 3 wheat.
    Without an organic-matter module the PET keeps its residue parameters (the port is unbound)."""

    mass: Array = field(unit="kg ha-1", description="flat residue mass", fortran_name="RM", dims=())
    age: Array = field(
        unit="d", description="days since the last residue addition", fortran_name="RESAGE", dims=()
    )
    wet: Array = field(unit="-", description="> 0 when the residue is wet", fortran_name="WRES", dims=())
    kind: Array = field(
        unit="-", description="residue type 0 none, 1 corn, 2 soybean, 3 wheat", fortran_name="IPR", dims=()
    )

    @classmethod
    def none(cls, dtype: Any = None) -> ResidueRecord:
        """No residue (``kind = 0``, mass 0)."""
        z = _z(dtype)
        return cls(mass=z, age=z, wet=z, kind=z)


class SedimentOut(State):
    """The day's eroded sediment (P24, ``iface.sediment``). The all-zero record is the reference's
    own behaviour with erosion off (runoff particulate phosphorus 0)."""

    sediment: Array = field(unit="kg ha-1 d-1", description="sediment yield of the day", dims=())

    @classmethod
    def zeros(cls, dtype: Any = None) -> SedimentOut:
        """No erosion."""
        return cls(sediment=_z(dtype))


# ------------------------------------------------------------------------ DSSAT-CSM day records
# The DSSAT-CSM v4.8.6.0 day (agrijax.iface.contract.DSSAT_DAY_TABLE) keeps the potential rates in
# P5 (iface.pet: EO, EOS, EOP) and the actual evaporation of SPAM in a record of its own, so that
# no field of P5 changes meaning between the two reference days (DSSAT_PORTS "PD1", "PD2").


class EvaporationRecord(State):
    """The day's **actual** evaporation and transpiration of the DSSAT-CSM ``SPAM`` module (PD1,
    ``iface.evaporation``): written by the soil and mulch evaporation (``ES``, ``EM``, ``EVAP``,
    ``ES_LYR``) and by the root extraction ``XTRACT`` (``EP``) in the RATE phase, read the same day
    by ``TRANS`` (``EVAP``), ``XTRACT`` (``ES``, ``ES_LYR``) and ``WATBAL`` INTEGR (``ES``, ``EM``).
    The potential rates stay in P5 (:class:`PETFluxes`). The field names of the soil and residue
    evaporation are those of P5, so a soil-water module that reads the actual amounts from a
    P5-shaped port (the DSSAT tipping bucket) binds to this record unchanged.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for lines 319-398 (ES, EM, EF, EVAP, EP), BSD-3.
    """

    soil_evaporation: Array = field(
        dims=(), unit="cm d-1", description="actual soil evaporation ES / 10", fortran_name="ES"
    )
    residue_evaporation: Array = field(
        dims=(), unit="cm d-1", description="actual mulch (residue) evaporation EM / 10", fortran_name="EM"
    )
    flood_evaporation: Array = field(
        dims=(), unit="cm d-1", description="actual floodwater evaporation EF / 10", fortran_name="EF"
    )
    evaporation: Array = field(
        dims=(),
        unit="mm d-1",
        description="total actual evaporation EVAP = ES + EM + EF (SPAM.for:373; TRANS reads it)",
        fortran_name="EVAP",
    )
    soil_evaporation_layers: Array = field(
        dims=("n_layer",),
        unit="cm d-1",
        description="actual soil evaporation of each layer ES_LYR / 10 (MESEV = S; 0 with MESEV = R)",
        fortran_name="ES_LYR",
        grid="dssat_layers",
    )
    transpiration: Array = field(
        dims=(), unit="mm d-1", description="actual transpiration EP = 10 TRWU (XTRACT)", fortran_name="EP"
    )

    @classmethod
    def zeros(cls, n_layer: int, dtype: Any = None) -> EvaporationRecord:
        """No evaporation, no transpiration."""
        z = _z(dtype)
        return cls(z, z, z, z, jnp.zeros((int(n_layer),), dtype=z.dtype), z)


class SoilAlbedo(State):
    """The day's soil surface albedo of DSSAT-CSM ``SOILDYN`` (PD2, ``iface.soil_albedo``): written in
    the RATE phase before the soil water rate, read the same day by the potential
    evapotranspiration (``PETPT`` reads ``MSALB``).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for ALBEDO_avg (lines 1505-1577), BSD-3.
    """

    msalb: Array = field(
        dims=(), unit="-", description="albedo of the soil with its mulch cover", fortran_name="MSALB"
    )
    swalb: Array = field(
        dims=(),
        unit="-",
        description="albedo of the bare soil at its top-layer water content",
        fortran_name="SWALB",
    )

    @classmethod
    def constant(cls, salb: Any, dtype: Any = None) -> SoilAlbedo:
        """``SEASINIT``: both albedos at the soil file's ``SALB`` (SOILDYN.for:870)."""
        v = jnp.asarray(salb, dtype=dtype if dtype is not None else jnp.result_type(float))
        return cls(msalb=v, swalb=v)
