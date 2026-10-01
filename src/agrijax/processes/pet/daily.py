"""Day-level PET processes: ``(state, params, forcing_t) -> state`` wrappers of the PET kernels.

The kernels in :mod:`.shuttleworth_wallace`, :mod:`.penman_monteith` and
:mod:`.priestley_taylor` take plain arrays. The processes here give them the process
signature so that they are registered (``agrijax.core.registry``), linted with all rules,
write-checked under ``AGRI_JAX_CHECK=1`` and bound to the ports of the coupling contract
(:mod:`agrijax.iface`) in an assembled day.

Module state (:class:`PETState`, own subtree ``surface.pet``): the PET module keeps no state of its
own; it has three ports, bound at assembly (:func:`agrijax.core.ports.bind`)::

    state.canopy  CanopyRecord  P6  iface.canopy.<slot>  read, one-day lag (the crop publishes after PET)
    state.theta   Array         P7  soil_water.theta     read, one-day lag (surface node, index 0)
    state.pet     PETFluxes     P5  iface.pet            written, fields:

    state.pet.{transpiration, soil_evaporation, residue_evaporation}   cm d-1 (S-W)
    state.pet.{reference_short, reference_tall}                          mm d-1 (ASCE)
    state.pet.eo_priestley_taylor                                        mm d-1 (DSSAT PETPT)

The flat residue (mass, age, wetness, type) is a parameter record, :class:`SurfaceResidue` in
``PETSiteParams.residue`` (there is no residue module yet to publish it as a port).

Forcing (one day, :class:`~agrijax.iface.surface.DailyWeather`, P8): ``tmin, tmax`` degC,
``srad`` MJ m-2 d-1, ``rh`` percent, ``wind_run``
km d-1 at ``params.wind_height`` m, ``doy``; ``srad`` is the field radiation RTS and
``srad_horizontal`` (MJ m-2 d-1) the measured horizontal radiation RTH. The wind run is used as
given; apply the reference model's forcing preparation
(:func:`agrijax.io.rzwqm.met.prepare_rzwqm_forcing`) before building the forcing when the result
is compared with RZWQM2.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Params, State, field
from agrijax.iface.crop import CanopyRecord
from agrijax.iface.surface import DailyWeather, PETFluxes

from .coefficients import PET_COEFFICIENTS, PETCoefficients
from .penman_monteith import asce_reference_et
from .priestley_taylor import priestley_taylor
from .shuttleworth_wallace import KM_DAY_TO_M_S, RESIDUE_RANDOMNESS, PETParams, shuttleworth_wallace

__all__ = [
    "RESIDUE_KINDS",
    "DailyWeather",
    "PETFluxes",
    "PETSiteParams",
    "PETState",
    "SurfaceResidue",
    "pet_asce_reference",
    "pet_priestley_taylor",
    "pet_shuttleworth_wallace",
]

#: residue-type codes of :attr:`SurfaceResidue.kind` (RZWQM2 ``IPR``: 0 none, 1 corn, 2 soybean,
#: 3 wheat; the order of the ``RESISThr`` residue diameter and density tables)
RESIDUE_KINDS: dict[str, float] = {"none": 0.0, "corn": 1.0, "soybean": 2.0, "wheat": 3.0}
_SOYBEAN = RESIDUE_KINDS["soybean"]
_WHEAT = RESIDUE_KINDS["wheat"]

#: registry metadata of the Shuttleworth-Wallace kernel, shared by the processes that wrap it
SW_SOURCES = (
    (
        "three-source (canopy, soil, residue) combination equations",
        "Shuttleworth & Wallace (1985); Farahani & Ahuja (1996) eqs. 5-8, 16-18",
    ),
    ("saturation vapour pressure", "Bosen (1960) form, as RZWQM2 ECONST (Rzpet.for)"),
    ("clear-sky radiation, albedo, net radiation", "RZWQM2 MAXSW / CSRAD / ALBSWS / NETRAD (Rzpet.for)"),
    ("aerodynamic, surface and residue resistances", "Farahani & Ahuja (1996); RZWQM2 RESISThr (Rzpet.for)"),
)
SW_DEVIATES = (
    (
        "snow sublimation (SNOWQE), snow long-wave, pan evaporation, hourly / SHAW / PENFLUX paths, slope "
        "and aspect, standing stubble and plastic mulch are not implemented",
        "outside the daily configuration that is validated",
        "shuttleworth_wallace.py shuttleworth_wallace docstring",
    ),
    (
        "wetness-dependent soil surface resistance (rzwqm.dat item 11 = -1, -2) is not implemented",
        "only the constant rss is used in the validated scenarios",
        "shuttleworth_wallace.py resistances docstring",
    ),
    (
        "infinite resistances are masked instead of 1e30; the sunset angle and vapour-pressure square root "
        "are guarded",
        "finite values and gradients on every branch",
        "tests/unit/test_pet_grad_finite.py",
    ),
)
#: deviations of the port-bound process (on top of the kernel's)
SW_PORT_DEVIATES = (
    (
        "the flat residue (mass, age, wetness, type) is a parameter, constant over a run",
        "there is no residue module yet; one would publish the residue as a port",
        "daily.py SurfaceResidue docstring",
    ),
    (
        "with several crops in the slot, LAI and TLAI are summed and the height is the tallest crop's",
        "S-W has one canopy source; RZWQM2 passes one plant's canopy "
        "(how to split the PET between several crops is not defined)",
        "daily.py pet_shuttleworth_wallace docstring",
    ),
    (
        "soil crust and random roughness of the soil albedo are 0",
        "ICRUST = 0 and RR = 0 on every day of the 9 validated scenarios",
        "tests/integration/test_pet_process_dumps.py",
    ),
)


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
        description="node water content (P7), yesterday's end state; the PET reads the surface node",
    )
    pet: PETFluxes = port(description="the potential fluxes this module writes (P5)")

    @classmethod
    def module(cls, canopy: CanopyRecord, theta: Any, pet: PETFluxes | None = None) -> PETState:
        """The module state with its ports filled (``pet`` defaults to :meth:`PETFluxes.zeros`)."""
        theta = jnp.asarray(theta)
        return cls(canopy=canopy, theta=theta, pet=PETFluxes.zeros(theta.dtype) if pet is None else pet)


class SurfaceResidue(Params):
    """Flat residue on the soil surface, read by the Shuttleworth-Wallace PET (a parameter for now,
    as there is no residue module yet). ``kind`` is RZWQM2's ``IPR`` (:data:`RESIDUE_KINDS`): it
    selects the residue diameter and density of :class:`~.coefficients.ResistCoefficients`;
    ``kind = 0`` (no residue type) switches the residue branch off whatever the mass (``RESISThr``,
    Rzpet.for line 2675)."""

    mass: Array = field(dims=(), unit="kg ha-1", description="flat residue mass", fortran_name="RM")
    age: Array = field(
        dims=(), unit="d", description="days since the last residue addition", fortran_name="RESAGE"
    )
    wet: Array = field(dims=(), unit="-", description="> 0 when the residue is wet", fortran_name="WRES")
    kind: Array = field(
        dims=(), unit="-", description="residue type 0 none, 1 corn, 2 soybean, 3 wheat", fortran_name="IPR"
    )

    @classmethod
    def none(cls) -> SurfaceResidue:
        """No residue (``kind = 0``, mass 0); weakly typed zeros, so they take the precision of the
        other inputs (no float32 upcast)."""
        z = jnp.asarray(RESIDUE_KINDS["none"])
        return cls(mass=z, age=z, wet=z, kind=z)


class PETSiteParams(Params):
    """Site and surface parameters of the PET processes."""

    pet: PETParams
    elevation: Array = field(dims=(), unit="m", description="site elevation", fortran_name="ELEV")
    latitude: Array = field(dims=(), unit="rad", description="site latitude", fortran_name="ALAT")
    wc13: Array = field(
        dims=(),
        unit="cm3 cm-3",
        description="1/3-bar water content of the surface layer",
        fortran_name="SOILHP(7)",
    )
    wc15: Array = field(
        dims=(),
        unit="cm3 cm-3",
        description="15-bar water content of the surface layer",
        fortran_name="SOILHP(9)",
    )
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
    rainfall_zone: int = field(
        description="1 arid, 2 semi-arid, 3 humid", fortran_name="IRAIN", static=True, default=2
    )
    residue_type: str = field(
        description="corn | soybean | wheat: residue type of the prescribed-canopy demo "
        "(the port-bound S-W process reads residue.kind)",
        fortran_name="IPR",
        static=True,
        default="corn",
    )
    residue_cover_factor: float = field(
        description="CRES of the residue block", fortran_name="CRES", static=True, default=RESIDUE_RANDOMNESS
    )
    asce_variant: str = field(
        description="'asce' or 'rzwqm' (REF_ET.FOR constants)", static=True, default="asce"
    )
    coefficients: PETCoefficients | None = field(
        description="the coefficients of the PET equations (None: RZWQM2 / ASCE-EWRI 2005 / DSSAT values)",
        default=None,
    )
    residue: SurfaceResidue = field(
        description="flat residue read by the S-W process (default: none)",
        default_factory=SurfaceResidue.none,
    )

    @property
    def coeffs(self) -> PETCoefficients:
        """The coefficients in force: :attr:`coefficients`, or :data:`~.coefficients.PET_COEFFICIENTS`."""
        return PET_COEFFICIENTS if self.coefficients is None else self.coefficients


def _canopy_sums(canopy: CanopyRecord) -> tuple[Array, Array, Array]:
    """``(lai, tlai, height)`` of the slot's crops as the one canopy source of S-W: LAI and TLAI
    summed over the crops, the tallest height (one crop: its own values, bit for bit).

    Source: coupling contract port P6; S-W has one canopy source per field, so the crops of a slot are
    merged (how the PET would be split between several crops is not defined).
    """
    return (
        jnp.sum(canopy.lai, axis=-1),
        jnp.sum(canopy.tlai, axis=-1),
        jnp.max(canopy.height, axis=-1),
    )


def _residue_shape(residue: SurfaceResidue, c: Any) -> tuple[Array, Array, Array]:
    """``(mass, diameter, density)`` of the flat residue: the mass is 0 for ``kind = 0`` (no residue
    type: ``RESISThr`` skips the residue branch), diameter and density follow ``kind``.

    Source: RESISThr, Rzpet.for line 2675 (IPR test) and its residue DATA tables (IPR order).
    """
    kind = jnp.asarray(residue.kind)
    mass = jnp.where(kind > 0.0, residue.mass, 0.0)
    soy, wheat = kind == _SOYBEAN, kind == _WHEAT
    rdia = jnp.where(
        soy, c.residue_diameter_soybean, jnp.where(wheat, c.residue_diameter_wheat, c.residue_diameter_corn)
    )
    rho = jnp.where(
        soy, c.residue_density_soybean, jnp.where(wheat, c.residue_density_wheat, c.residue_density_corn)
    )
    return mass, rdia, rho


@process(
    reads=("canopy", "theta"),
    writes=("pet.transpiration", "pet.soil_evaporation", "pet.residue_evaporation"),
    source="Shuttleworth & Wallace (1985); Farahani & Ahuja (1996); RZWQM2 Rzpet.for POTEVPHR",
    fortran_name="POTEVPHR",
    key="pet/shuttleworth_wallace@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="point",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=SW_SOURCES,
    deviates=SW_DEVIATES + SW_PORT_DEVIATES,
)
def pet_shuttleworth_wallace(state: PETState, params: PETSiteParams, forcing_t: DailyWeather) -> PETState:
    """Daily three-source Shuttleworth-Wallace potential transpiration and evaporation [cm d-1].

    Reads yesterday's canopy (port ``canopy``, P6: LAI, TLAI, height), yesterday's water content
    of the surface node (port ``theta``, P7, node 0), the day's weather including the measured
    horizontal radiation ``srad_horizontal`` (RTH) and the residue parameters; writes P5
    (``transpiration``, ``soil_evaporation``, ``residue_evaporation``). The reference computes
    PET at the start of PHYSCL, before the soil-water time loop and before the crop, so every
    state input is a start-of-day value (the one-day lags of the contract's ports P6 and P7).

    Forcing fields read: tmin, tmax, srad, srad_horizontal, rh, wind_run, doy.

    Thin wrapper of :func:`agrijax.processes.pet.shuttleworth_wallace`; see that function for
    the algorithm and the known deviations from the reference model.

    Source: POTEVPHR, Rzpet.for lines 1673-2428; Shuttleworth & Wallace (1985); Farahani & Ahuja (1996).
    """
    cf = params.coeffs.sw
    lai, tlai, height = _canopy_sums(state.canopy)
    mass, rdia, rho = _residue_shape(params.residue, cf.resist)
    r = shuttleworth_wallace(
        forcing_t.tmin,
        forcing_t.tmax,
        forcing_t.srad,
        forcing_t.rh,
        forcing_t.wind_run,
        lai,
        height,
        params.pet,
        theta_surface=state.theta[..., 0],
        wc13=params.wc13,
        wc15=params.wc15,
        elevation=params.elevation,
        latitude=params.latitude,
        doy=forcing_t.doy,
        tlai=tlai,
        srad_horizontal=forcing_t.srad_horizontal,
        residue_mass=mass,
        residue_age=params.residue.age,
        residue_wet=params.residue.wet,
        wind_height=params.wind_height,
        trat=params.trat,
        rainfall_zone=params.rainfall_zone,
        residue_cover_factor=params.residue_cover_factor,
        residue_diameter_cm=rdia,
        residue_density=rho,
        coefficients=cf,
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
