"""Day-level PET processes: ``(state, params, forcing_t) -> state`` wrappers of the PET kernels.

The kernels in :mod:`.penman_monteith` and :mod:`.priestley_taylor` take plain arrays. The
processes here give them the process signature so that they are registered
(``agrijax.core.registry``), linted with all rules, write-checked under ``AGRI_JAX_CHECK=1`` and
bound to the ports of the coupling contract (:mod:`agrijax.iface`) in an assembled day.

Module state (:class:`PETState`, own subtree ``surface.pet``): the PET module keeps no state of its
own; it has three ports, bound at assembly (:func:`agrijax.core.ports.bind`)::

    state.canopy  CanopyRecord  P6  iface.canopy.<slot>  read, one-day lag (the crop publishes after PET)
    state.theta   Array         P7  soil_water.theta     soil water content (not read by these processes)
    state.pet     PETFluxes     P5  iface.pet            written, fields:

    state.pet.{transpiration, soil_evaporation, residue_evaporation}   cm d-1 (a three-source PET; not here)
    state.pet.{reference_short, reference_tall}                          mm d-1 (ASCE)
    state.pet.eo_priestley_taylor                                        mm d-1 (DSSAT PETPT)

Forcing (one day, :class:`~agrijax.iface.surface.DailyWeather`, P8): ``tmin, tmax`` degC,
``srad`` MJ m-2 d-1, ``rh`` percent, ``wind_run``
km d-1 at ``params.wind_height`` m, ``doy``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Params, State, field
from agrijax.core.units import conversion_factor
from agrijax.iface.crop import CanopyRecord
from agrijax.iface.surface import DailyWeather, PETFluxes

from .coefficients import PET_COEFFICIENTS, PETCoefficients
from .penman_monteith import asce_reference_et
from .priestley_taylor import priestley_taylor

#: wind run [km d-1] to wind speed [m s-1]
KM_DAY_TO_M_S: float = conversion_factor("km d-1", "m s-1")

__all__ = [
    "DailyWeather",
    "PETFluxes",
    "PETSiteParams",
    "PETState",
    "pet_asce_reference",
    "pet_priestley_taylor",
]


class PETState(State):
    """The PET module's state: no fields of its own, three ports (see the module docstring).

    In the global state the module sits at ``surface.pet`` with its ports set to ``None``; each
    port's record lives at its bound global path. Build the module state for a standalone call
    with :meth:`module`.
    """

    canopy: CanopyRecord = port(
        description="the crops' canopy (P6), yesterday's record: the crop publishes after PET"
    )
    theta: Array = port(
        unit="cm3 cm-3",
        dims=("n_node",),
        grid="rzwqm2_nodes",
        fortran_name="THETA",
        description="node water content (P7), yesterday's end state (not read by the ASCE and PT processes)",
    )
    pet: PETFluxes = port(description="the potential fluxes this module writes (P5)")

    @classmethod
    def module(cls, canopy: CanopyRecord, theta: Any, pet: PETFluxes | None = None) -> PETState:
        """The module state with its ports filled (``pet`` defaults to :meth:`PETFluxes.zeros`)."""
        theta = jnp.asarray(theta)
        return cls(canopy=canopy, theta=theta, pet=PETFluxes.zeros(theta.dtype) if pet is None else pet)


class PETSiteParams(Params):
    """Site and surface parameters of the PET processes."""

    elevation: Array = field(dims=(), unit="m", description="site elevation", fortran_name="ELEV")
    latitude: Array = field(dims=(), unit="rad", description="site latitude", fortran_name="ALAT")
    wind_height: Array = field(dims=(), unit="m", description="wind measurement height", fortran_name="XW")
    albedo_soil: Array = field(
        dims=(), unit="-", description="DSSAT soil albedo MSALB (Priestley-Taylor)", fortran_name="MSALB"
    )
    trat: Array = field(
        dims=(),
        unit="-",
        description="CO2 factor on stomatal resistance (1 at 330 ppm)",
        fortran_name="TRATIO",
    )
    asce_variant: str = field(
        description="'asce' or 'rzwqm' (REF_ET.FOR constants)", static=True, default="asce"
    )
    coefficients: PETCoefficients | None = field(
        description="the coefficients of the PET equations (None: ASCE-EWRI 2005 / DSSAT values)",
        default=None,
    )

    @property
    def coeffs(self) -> PETCoefficients:
        """The coefficients in force: :attr:`coefficients`, or :data:`~.coefficients.PET_COEFFICIENTS`."""
        return PET_COEFFICIENTS if self.coefficients is None else self.coefficients


@process(
    reads=(),
    writes=("pet.reference_short", "pet.reference_tall"),
    source="ASCE-EWRI (2005); RZWQM2 REF_ET.FOR",
    fortran_name="REF_ET",
    key="pet/asce_reference@asce-ewri-2005:faithful",
    provenance="equations_only",
    grid="point",
    ref_build="ASCE-EWRI (2005) final report, eq. 1 and Table 1; RZWQM2 4.6 .ana for variant rzwqm",
    sources=(
        ("standardized reference ET, daily Cn / Cd (short, tall)", "ASCE-EWRI (2005) eq. 1, Table 1"),
        ("extraterrestrial, clear-sky and net radiation", "ASCE-EWRI (2005) eqs. 17-24"),
        ("wind speed at 2 m from the measurement height", "ASCE-EWRI (2005) log profile, as REF_ET.FOR"),
    ),
    deviates=(
        (
            "actual vapour pressure from the daily mean RH (FAO-56 eq. 19 form), not RHmax / RHmin",
            "the daily forcing carries mean RH only",
            "penman_monteith.py asce_reference_et docstring",
        ),
        (
            "variant 'asce' uses the FAO-56 Stefan-Boltzmann constant 4.903e-9 MJ m-2 K-4 d-1 "
            "(FAO-56 eq. 39); ASCE-EWRI (2005) eq. 17 states 4.901e-9",
            "kept for comparison against textbook / pyet values; variant 'rzwqm' uses 4.901e-9",
            "penman_monteith.py asce_reference_et docstring; coefficients.ASCECoefficients.stefan_boltzmann",
        ),
        (
            "trat multiplies Cd u2 (DSSAT TRATIO CO2 hook, 1.0 at 330 ppm)",
            "kept for the reference-model coupling; not part of the ASCE equation",
            "penman_monteith.py asce_reference_et docstring",
        ),
        (
            "params.asce_variant='rzwqm' switches to the RZWQM2 REF_ET.FOR constants (sigma, day angle, "
            "Kelvin offset); a static parameter inside one process, not a sibling key",
            "the same kernel serves both references; it could become a sibling @rzwqm2-4.6 variant",
            "tests/integration/test_pet_oracle.py::test_asce_reference_et_against_ana (1e-3 mm/d)",
        ),
        (
            "the hourly branch of REF_ET.FOR is not implemented",
            "the model is daily",
            "penman_monteith.py asce_reference_et docstring",
        ),
    ),
)
def pet_asce_reference(state: PETState, params: PETSiteParams, forcing_t: DailyWeather) -> PETState:
    """Daily ASCE standardized reference ET, short and tall surfaces [mm d-1], written to P5.

    Thin wrapper of :func:`agrijax.processes.pet.asce_reference_et` with the wind run converted
    from km d-1 to m s-1. Reads no port.

    Forcing fields read: tmin, tmax, srad, rh, wind_run, doy.

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
        coefficients=params.coeffs.asce,
    )
    return eqx.tree_at(
        lambda st: (st.pet.reference_short, st.pet.reference_tall), state, (r.et_short, r.et_tall)
    )


@process(
    reads=("canopy.lai",),
    writes=("pet.eo_priestley_taylor",),
    source="Priestley & Taylor (1972); Ritchie (1972); DSSAT-CSM SPAM/PET.for PETPT",
    fortran_name="PETPT",
    key="pet/priestley_taylor@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="point",
    ref_build="DSSAT-CSM v4.8.6.0 source, SPAM/PET.for",
    sources=(
        ("equilibrium evaporation, LAI-dependent albedo", "Priestley & Taylor (1972); Ritchie (1972)"),
        (
            "TD weighting, Tmax > 35 / Tmax < 5 branches, 1e-4 mm floor",
            "DSSAT-CSM SPAM/PET.for PETPT, lines 871-918 (BSD-3)",
        ),
    ),
    deviates=(
        (
            "DSSAT single precision is not reproduced",
            "the kernels run in the precision of their inputs (float64 with x64, float32 otherwise)",
            "priestley_taylor.py docstring",
        ),
        (
            "with several crops in the slot, the LAI is their sum",
            "one canopy per field (how to split the PET between several crops is not defined)",
            "daily.py pet_priestley_taylor docstring",
        ),
    ),
)
def pet_priestley_taylor(state: PETState, params: PETSiteParams, forcing_t: DailyWeather) -> PETState:
    """DSSAT-CSM Priestley-Taylor potential ET [mm d-1], written to P5 ``eo_priestley_taylor``.

    Reads yesterday's green LAI from the canopy port (P6, summed over the slot's crops). Thin
    wrapper of :func:`agrijax.processes.pet.priestley_taylor`.

    Forcing fields read: tmin, tmax, srad.

    Source: PETPT, dssat-csm-os SPAM/PET.for; Priestley & Taylor (1972); Ritchie (1972).
    """
    eo = priestley_taylor(
        forcing_t.srad,
        forcing_t.tmax,
        forcing_t.tmin,
        jnp.sum(state.canopy.lai, axis=-1),
        params.albedo_soil,
        params.coeffs.pt,
    )
    return eqx.tree_at(lambda st: st.pet.eo_priestley_taylor, state, eo)
