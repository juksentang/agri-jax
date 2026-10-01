"""CA-TPA PET demo: the first real :class:`~agrijax.core.Model` assembled from existing processes.

A pipeline model, not a science model. It exists so that the core runtime (``run``, ``run_batch``,
``run_batch_chunked``, checkpointing, ``run_and_grad``, the writes check and the three-rules lint)
is exercised on a real process, real forcing and a multi-year horizon instead of a toy.

State::

    theta_surface                         surface-node water content [cm3 cm-3] (held constant:
                                          no soil water process yet; initial value from LAYER.PLT)
    pet.{transpiration, soil_evaporation, residue_evaporation}   today's S-W fluxes [cm d-1]
    totals.{transpiration, evaporation, days}                    running sums [cm], day counter

Forcing (one day, time axis first when stacked): the prepared ``.MET`` weather (``tmin``,
``tmax`` degC, ``srad`` MJ m-2 d-1, ``rh`` percent, ``wind_run`` km d-1, ``doy``) plus the canopy
and residue the S-W process would otherwise read from a crop model (``lai``, ``height_cm``,
``residue_mass`` kg ha-1, ``residue_age`` d). :func:`build_forcing` takes the canopy from the
columns 43 / 62 / 72 of a one-year reference ``.ana`` and repeats that year for the other years.

Processes (in order):

1. :func:`sw_pet_from_forcing` -- thin wrapper of
   :func:`agrijax.processes.pet.shuttleworth_wallace` that reads the canopy from the forcing;
2. :func:`accumulate_totals` -- running sums of the fluxes and a day counter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxtyping import Array

from agrijax.core.model import Model
from agrijax.core.process import process
from agrijax.core.state import Forcing, State, field
from agrijax.processes.pet import PETParams, PETSiteParams, shuttleworth_wallace
from agrijax.processes.pet.daily import SW_DEVIATES, SW_SOURCES

__all__ = [
    "OUTPUTS",
    "DemoFluxes",
    "DemoForcing",
    "DemoState",
    "DemoTotals",
    "accumulate_totals",
    "build_forcing",
    "catpa_pet_model",
    "initial_state",
    "site_params_from_dat",
    "sw_pet_from_forcing",
]

#: .ana columns used for the canopy: LAI, plant height [cm], surface residue mass [kg ha-1].
ANA_LAI, ANA_HEIGHT, ANA_RESIDUE = 43, 62, 72
#: residue age used for the residue albedo on 1 January (only a weak effect, see test_pet_oracle).
RESIDUE_AGE0 = 50.0

OUTPUTS: tuple[str, ...] = (
    "pet.transpiration",
    "pet.soil_evaporation",
    "pet.residue_evaporation",
    "totals.transpiration",
    "totals.evaporation",
    "totals.days",
)


class DemoFluxes(State):
    """Today's Shuttleworth-Wallace potential fluxes."""

    transpiration: Array = field(
        dims=(), unit="cm d-1", description="potential transpiration", fortran_name="PET"
    )
    soil_evaporation: Array = field(
        dims=(), unit="cm d-1", description="potential soil evaporation", fortran_name="PES"
    )
    residue_evaporation: Array = field(
        dims=(), unit="cm d-1", description="potential residue evaporation", fortran_name="PER"
    )


class DemoTotals(State):
    """Running sums since the start of the run."""

    transpiration: Array = field(dims=(), unit="cm", description="cumulative potential transpiration")
    evaporation: Array = field(
        dims=(), unit="cm", description="cumulative potential soil + residue evaporation"
    )
    days: Array = field(dims=(), unit="d", description="number of days simulated")


class DemoState(State):
    """State of the CA-TPA PET demo model."""

    theta_surface: Array = field(
        dims=(), unit="cm3 cm-3", description="water content of the surface node", fortran_name="THETA(1)"
    )
    pet: DemoFluxes = field(description="today's S-W potential fluxes")
    totals: DemoTotals = field(description="running sums since the start of the run")


class DemoForcing(Forcing):
    """Daily weather plus the prescribed canopy and residue (time axis first when stacked)."""

    tmin: Array = field(unit="degC", dims="T", fortran_name="TMIN")
    tmax: Array = field(unit="degC", dims="T", fortran_name="TMAX")
    srad: Array = field(unit="MJ m-2 d-1", dims="T", fortran_name="RTS")
    rh: Array = field(unit="percent", dims="T", fortran_name="RH")
    wind_run: Array = field(unit="km d-1", dims="T", fortran_name="U")
    doy: Array = field(unit="d", dims="T", fortran_name="JDAY")
    lai: Array = field(unit="m2 m-2", dims="T", fortran_name="LAI")
    height_cm: Array = field(unit="cm", dims="T", fortran_name="HEIGHT")
    residue_mass: Array = field(unit="kg ha-1", dims="T", fortran_name="RM")
    residue_age: Array = field(unit="d", dims="T", fortran_name="RESAGE")


# ---------------------------------------------------------------------------------- processes


@process(
    reads=("theta_surface",),
    writes=("pet",),
    source="Shuttleworth & Wallace (1985); Farahani & Ahuja (1996); RZWQM2 Rzpet.for POTEVPHR",
    fortran_name="POTEVPHR",
    key="pet/shuttleworth_wallace@rzwqm2-4.6:prescribed_canopy",
    provenance="reference_only_conventions",
    grid="point",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=SW_SOURCES,
    deviates=(
        *SW_DEVIATES,
        (
            "canopy LAI and height come from the forcing, green LAI = total LAI, dry residue, no crust, "
            "no roughness",
            "demonstration of S-W PET on a prescribed canopy (no crop module)",
            "catpa_pet_demo.py sw_pet_from_forcing docstring",
        ),
    ),
)
def sw_pet_from_forcing(state: DemoState, params: PETSiteParams, forcing_t: DemoForcing) -> DemoState:
    """Daily three-source Shuttleworth-Wallace PET [cm d-1] with the canopy taken from the forcing.

    Thin wrapper of :func:`agrijax.processes.pet.shuttleworth_wallace`: green = total LAI,
    dry residue, no crust, no roughness.

    Source: POTEVPHR, Rzpet.for lines 1673-2428; Shuttleworth & Wallace (1985); Farahani & Ahuja (1996).
    """
    r = shuttleworth_wallace(
        forcing_t.tmin,
        forcing_t.tmax,
        forcing_t.srad,
        forcing_t.rh,
        forcing_t.wind_run,
        forcing_t.lai,
        forcing_t.height_cm,
        params.pet,
        theta_surface=state.theta_surface,
        wc13=params.wc13,
        wc15=params.wc15,
        elevation=params.elevation,
        latitude=params.latitude,
        doy=forcing_t.doy,
        tlai=forcing_t.lai,
        residue_mass=forcing_t.residue_mass,
        residue_age=forcing_t.residue_age,
        wind_height=params.wind_height,
        trat=params.trat,
        rainfall_zone=params.rainfall_zone,
        residue_type=params.residue_type,
        residue_cover_factor=params.residue_cover_factor,
    )
    return eqx.tree_at(
        lambda s: (s.pet.transpiration, s.pet.soil_evaporation, s.pet.residue_evaporation),
        state,
        (r.transpiration, r.soil_evaporation, r.residue_evaporation),
    )


@process(
    reads=("pet", "totals"),
    writes=("totals",),
    source="bookkeeping; no literature",
    key="diagnostic/catpa_pet_totals@none:demo",
    provenance="equations_only",
    grid="point",
    sources=(("running sums of transpiration and soil + residue evaporation, day count", "bookkeeping"),),
    deviates=(),
)
def accumulate_totals(state: DemoState, params: PETSiteParams, forcing_t: DemoForcing) -> DemoState:
    """Add today's fluxes to the running sums [cm] and count the day.

    Source: bookkeeping, no literature (evaporation = soil + residue evaporation).
    """
    p, t = state.pet, state.totals
    return eqx.tree_at(
        lambda s: (s.totals.transpiration, s.totals.evaporation, s.totals.days),
        state,
        (
            t.transpiration + p.transpiration,
            t.evaporation + p.soil_evaporation + p.residue_evaporation,
            t.days + 1.0,
        ),
    )


def catpa_pet_model(outputs: tuple[str, ...] | None = OUTPUTS) -> Model:
    """The demo model: S-W PET from the prescribed canopy, then the running sums."""
    return Model(DemoState, [sw_pet_from_forcing, accumulate_totals], outputs=outputs, name="catpa_pet_demo")


# ---------------------------------------------------------------------------------- inputs


def initial_state(theta_surface: float) -> DemoState:
    """Zero fluxes and sums, the given surface water content."""
    z = jnp.asarray(0.0, dtype=float)
    return DemoState(
        theta_surface=jnp.asarray(float(theta_surface), dtype=float),
        pet=DemoFluxes(z, z, z),
        totals=DemoTotals(z, z, z),
    )


def _ana_column(ana: Any, col: int) -> np.ndarray:
    cols = {int(k): v for k, v in ana.attrs["columns"].items()}
    return np.asarray(ana[cols[col]].values, dtype=np.float64)


def build_forcing(met: pd.DataFrame, ana: Any, start: str, end: str) -> DemoForcing:
    """Stack the prepared ``.MET`` weather of ``[start, end]`` and the canopy of a one-year ``.ana``.

    ``met`` is :func:`agrijax.io.rzwqm.read_met` with ``prepare=True``. ``ana`` is
    :func:`agrijax.io.rzwqm.read_ana` of a one-year run whose row 0 is the ``YYYY.000``
    initial state. LAI and height are the ``.ana`` values of the same day of year; the residue
    mass is the previous row's (start-of-day) value. Days of year past the ``.ana`` year (31
    December of a leap year) reuse its last day. The residue age restarts at
    :data:`RESIDUE_AGE0` every 1 January.
    """
    days = pd.date_range(start, end, freq="D")
    missing = days.difference(met.index)
    if len(missing):
        raise ValueError(f".MET has no record for {len(missing)} days, first {str(missing[0])[:10]}")
    w = met.loc[days]
    doy = np.array([d.timetuple().tm_yday for d in days], dtype=np.int64)
    lai = _ana_column(ana, ANA_LAI)
    height = _ana_column(ana, ANA_HEIGHT)
    residue = _ana_column(ana, ANA_RESIDUE)
    n_year = len(lai) - 1  # row 0 is the initial state
    k = np.minimum(doy, n_year)  # .ana row of that day of year
    return DemoForcing(
        tmin=jnp.asarray(w["tmin"].to_numpy(dtype=float)),
        tmax=jnp.asarray(w["tmax"].to_numpy(dtype=float)),
        srad=jnp.asarray(w["srad_mj"].to_numpy(dtype=float)),
        rh=jnp.asarray(w["rh"].to_numpy(dtype=float)),
        wind_run=jnp.asarray(w["wind_run_km"].to_numpy(dtype=float)),
        doy=jnp.asarray(doy.astype(float)),
        lai=jnp.asarray(lai[k]),
        height_cm=jnp.asarray(height[k]),
        residue_mass=jnp.asarray(residue[k - 1]),
        residue_age=jnp.asarray(RESIDUE_AGE0 + (doy - 1).astype(float)),
    )


def _cres(dat_path: Path) -> float:
    """CRES (third value) of the line after the ``C:P ratio of dominate residue material`` header."""
    lines = dat_path.read_bytes().decode("latin-1").splitlines()
    for i, ln in enumerate(lines):
        if "C:P ratio of dominate residue material" in ln:
            for nxt in lines[i + 1 :]:
                s = nxt.strip()
                if s and not s.startswith("="):
                    return float(s.split()[2])
    raise KeyError("residue block not found")


#: residue type of each value of ``CRES`` in the residue block of ``rzwqm.dat`` (the mapping this demo
#: assumes; any other value falls back to corn). A decoding table, not a model coefficient.
_RESIDUE_TYPE_BY_CRES: dict[float, str] = {2.0: "corn", 2.5: "soybean", 4.0: "wheat"}
#: DSSAT soil albedo ``MSALB`` of :class:`PETSiteParams`. Only the Priestley-Taylor path reads it;
#: this Shuttleworth-Wallace demo never does, so it is a placeholder site input, not a coefficient.
_MSALB_PLACEHOLDER: float = 0.13


def site_params_from_dat(dat: Any, dat_path: str | Path) -> PETSiteParams:
    """:class:`PETSiteParams` of a scenario, from its parsed ``rzwqm.dat`` (``read_rzwqm_dat``)."""
    phys, pet, plant, hyd = dat.physiography, dat.pet, dat.plant_site_params[0], dat.hydraulics
    cres = _cres(Path(dat_path))
    residue_type = _RESIDUE_TYPE_BY_CRES.get(cres, "corn")

    def f(x: Any) -> Array:
        return jnp.asarray(float(x), dtype=float)

    return PETSiteParams(
        pet=PETParams(
            albedo_dry=f(pet["albedo_dry"]),
            albedo_wet=f(pet["albedo_wet"]),
            albedo_maturity=f(pet["albedo_crop"]),
            albedo_residue=f(pet["albedo_residue"]),
            soil_resistance=f(pet["soil_resistance"]),
            stomatal_resistance=f(plant["rs_min"]),
        ),
        elevation=f(phys["elevation_m"]),
        latitude=f(phys["latitude_rad"]),
        wc13=f(hyd["theta_fc33"][0]),
        wc15=f(hyd["theta_wp"][0]),
        wind_height=f(pet["wind_height_m"]),
        albedo_soil=f(_MSALB_PLACEHOLDER),
        trat=f(1.0),
        rainfall_zone=int(phys["rainfall_zone"]),
        residue_type=residue_type,
        residue_cover_factor=cres,
    )
