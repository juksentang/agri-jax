"""Day-level PET processes: ``(state, params, forcing_t) -> state`` wrappers of the PET kernels.

The kernels in :mod:`.shuttleworth_wallace`, :mod:`.penman_monteith` and
:mod:`.priestley_taylor` take plain arrays. The processes here give them the process
signature so that they are registered (``agrijax.core.registry``), linted with all five
rules, write-checked under ``AGRI_JAX_CHECK=1`` and can be composed in a
:class:`agrijax.core.Model`.

State layout (the minimal slice of a crop-soil state the PET needs)::

    state.surface.{lai, tlai, height_cm, theta_surface, residue_mass, residue_age, residue_wet}
    state.pet.{transpiration, soil_evaporation, residue_evaporation}   cm d-1 (S-W)
    state.pet.{reference_short, reference_tall}                          mm d-1 (ASCE)
    state.pet.eo_priestley_taylor                                        mm d-1 (DSSAT PETPT)

Forcing (one day): ``tmin, tmax`` degC, ``srad`` MJ m-2 d-1, ``rh`` percent, ``wind_run``
km d-1 at ``params.wind_height`` m, ``doy``. The wind run is used as given; apply the
reference model's forcing preparation (:func:`agrijax.io.rzwqm.met.prepare_rzwqm_forcing`)
before building the forcing when the result is compared with RZWQM2.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field

from .penman_monteith import asce_reference_et
from .priestley_taylor import priestley_taylor
from .shuttleworth_wallace import KM_DAY_TO_M_S, RESIDUE_RANDOMNESS, PETParams, shuttleworth_wallace

__all__ = [
    "DailyWeather",
    "PETFluxes",
    "PETSiteParams",
    "PETState",
    "SurfaceState",
    "pet_asce_reference",
    "pet_priestley_taylor",
    "pet_shuttleworth_wallace",
]


class SurfaceState(State):
    """Canopy, residue and surface-soil state read by the PET processes."""

    lai: Array = field(unit="m2 m-2", description="green leaf area index", fortran_name="LAI")
    tlai: Array = field(
        unit="m2 m-2", description="total (green + senesced) leaf area index", fortran_name="TLAI"
    )
    height_cm: Array = field(unit="cm", description="canopy height", fortran_name="HEIGHT")
    theta_surface: Array = field(
        unit="cm3 cm-3", description="water content of the surface node", fortran_name="THETA(1)"
    )
    residue_mass: Array = field(unit="kg ha-1", description="flat residue mass", fortran_name="RM")
    residue_age: Array = field(
        unit="d", description="days since the last residue addition", fortran_name="RESAGE"
    )
    residue_wet: Array = field(unit="-", description="> 0 when the residue is wet", fortran_name="IRESW")


class PETFluxes(State):
    """Potential fluxes written by the PET processes."""

    transpiration: Array = field(unit="cm d-1", description="S-W potential transpiration", fortran_name="PET")
    soil_evaporation: Array = field(
        unit="cm d-1", description="S-W potential soil evaporation", fortran_name="PES"
    )
    residue_evaporation: Array = field(
        unit="cm d-1", description="S-W potential residue evaporation", fortran_name="PER"
    )
    reference_short: Array = field(
        unit="mm d-1", description="ASCE short-reference ET (grass)", fortran_name="ETO"
    )
    reference_tall: Array = field(
        unit="mm d-1", description="ASCE tall-reference ET (alfalfa)", fortran_name="ETR"
    )
    eo_priestley_taylor: Array = field(
        unit="mm d-1", description="DSSAT PETPT potential ET", fortran_name="EO"
    )


class PETState(State):
    """State slice of the PET processes."""

    surface: SurfaceState
    pet: PETFluxes

    @classmethod
    def zeros_like_surface(cls, surface: SurfaceState) -> PETState:
        """A state with ``surface`` and all fluxes set to zero (same shape as ``surface.lai``)."""
        z = jnp.zeros_like(jnp.asarray(surface.lai, dtype=float))
        return cls(surface=surface, pet=PETFluxes(z, z, z, z, z, z))


class PETSiteParams(Params):
    """Site and surface parameters of the PET processes."""

    pet: PETParams
    elevation: Array = field(unit="m", description="site elevation", fortran_name="ELEV")
    latitude: Array = field(unit="rad", description="site latitude", fortran_name="ALAT")
    wc13: Array = field(
        unit="cm3 cm-3", description="1/3-bar water content of the surface layer", fortran_name="SOILHP(7)"
    )
    wc15: Array = field(
        unit="cm3 cm-3", description="15-bar water content of the surface layer", fortran_name="SOILHP(9)"
    )
    wind_height: Array = field(unit="m", description="wind measurement height", fortran_name="XW")
    albedo_soil: Array = field(
        unit="-", description="DSSAT soil albedo MSALB (Priestley-Taylor)", fortran_name="MSALB"
    )
    trat: Array = field(
        unit="-", description="CO2 factor on stomatal resistance (1 at 330 ppm)", fortran_name="TRATIO"
    )
    rainfall_zone: int = field(
        description="1 arid, 2 semi-arid, 3 humid", fortran_name="IRAIN", static=True, default=2
    )
    residue_type: str = field(
        description="corn | soybean | wheat", fortran_name="IPR", static=True, default="corn"
    )
    residue_cover_factor: float = field(
        description="CRES of the residue block", fortran_name="CRES", static=True, default=RESIDUE_RANDOMNESS
    )
    asce_variant: str = field(
        description="'asce' or 'rzwqm' (REF_ET.FOR constants)", static=True, default="asce"
    )


class DailyWeather(Forcing):
    """Daily weather forcing of the PET processes (time axis first when stacked)."""

    tmin: Array = field(unit="degC", dims="T", fortran_name="TMIN")
    tmax: Array = field(unit="degC", dims="T", fortran_name="TMAX")
    srad: Array = field(unit="MJ m-2 d-1", dims="T", fortran_name="RTS")
    rh: Array = field(unit="percent", dims="T", fortran_name="RH")
    wind_run: Array = field(unit="km d-1", dims="T", fortran_name="U")
    doy: Array = field(unit="d", dims="T", fortran_name="JDAY")


@process(
    reads=("surface",),
    writes=("pet.transpiration", "pet.soil_evaporation", "pet.residue_evaporation"),
    source="Shuttleworth & Wallace (1985); Farahani & Ahuja (1996); RZWQM2 Rzpet.for POTEVPHR",
    fortran_name="POTEVPHR",
)
def pet_shuttleworth_wallace(state: PETState, params: PETSiteParams, forcing_t: DailyWeather) -> PETState:
    """Daily three-source Shuttleworth-Wallace potential transpiration and evaporation [cm d-1].

    Thin wrapper of :func:`agrijax.processes.pet.shuttleworth_wallace`; see that function for
    the algorithm and the known deviations from the reference model.

    Source: POTEVPHR, Rzpet.for lines 1673-2428; Shuttleworth & Wallace (1985); Farahani & Ahuja (1996).
    """
    s = state.surface
    r = shuttleworth_wallace(
        forcing_t.tmin,
        forcing_t.tmax,
        forcing_t.srad,
        forcing_t.rh,
        forcing_t.wind_run,
        s.lai,
        s.height_cm,
        params.pet,
        theta_surface=s.theta_surface,
        wc13=params.wc13,
        wc15=params.wc15,
        elevation=params.elevation,
        latitude=params.latitude,
        doy=forcing_t.doy,
        tlai=s.tlai,
        residue_mass=s.residue_mass,
        residue_age=s.residue_age,
        residue_wet=s.residue_wet,
        wind_height=params.wind_height,
        trat=params.trat,
        rainfall_zone=params.rainfall_zone,
        residue_type=params.residue_type,
        residue_cover_factor=params.residue_cover_factor,
    )
    return eqx.tree_at(
        lambda st: (st.pet.transpiration, st.pet.soil_evaporation, st.pet.residue_evaporation),
        state,
        (r.transpiration, r.soil_evaporation, r.residue_evaporation),
    )


@process(
    reads=(),
    writes=("pet.reference_short", "pet.reference_tall"),
    source="ASCE-EWRI (2005); RZWQM2 REF_ET.FOR",
    fortran_name="REF_ET",
)
def pet_asce_reference(state: PETState, params: PETSiteParams, forcing_t: DailyWeather) -> PETState:
    """Daily ASCE standardized reference ET, short and tall surfaces [mm d-1].

    Thin wrapper of :func:`agrijax.processes.pet.asce_reference_et` with the wind run converted
    from km d-1 to m s-1.

    Source: REF_ET.FOR (RZWQM2 4.5) lines 15-357, daily branch; ASCE-EWRI (2005).
    """
    r = asce_reference_et(
        forcing_t.tmin,
        forcing_t.tmax,
        forcing_t.srad,
        forcing_t.rh,
        jnp.asarray(forcing_t.wind_run) * KM_DAY_TO_M_S,
        elevation=params.elevation,
        latitude=params.latitude,
        doy=forcing_t.doy,
        wind_height=params.wind_height,
        trat=params.trat,
        variant="rzwqm" if params.asce_variant == "rzwqm" else "asce",
    )
    return eqx.tree_at(
        lambda st: (st.pet.reference_short, st.pet.reference_tall), state, (r.et_short, r.et_tall)
    )


@process(
    reads=("surface.lai",),
    writes=("pet.eo_priestley_taylor",),
    source="Priestley & Taylor (1972); Ritchie (1972); DSSAT-CSM SPAM/PET.for PETPT",
    fortran_name="PETPT",
)
def pet_priestley_taylor(state: PETState, params: PETSiteParams, forcing_t: DailyWeather) -> PETState:
    """DSSAT-CSM Priestley-Taylor potential ET [mm d-1].

    Thin wrapper of :func:`agrijax.processes.pet.priestley_taylor`.

    Source: PETPT, dssat-csm-os SPAM/PET.for; Priestley & Taylor (1972); Ritchie (1972).
    """
    eo = priestley_taylor(
        forcing_t.srad, forcing_t.tmax, forcing_t.tmin, state.surface.lai, params.albedo_soil
    )
    return eqx.tree_at(lambda st: st.pet.eo_priestley_taylor, state, eo)
