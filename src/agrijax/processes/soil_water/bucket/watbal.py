"""The DSSAT-CSM v4.8.6.0 tipping bucket as two soil-water processes: ``WATBAL`` RATE and INTEGR.

DSSAT splits the soil-water day in two around the evapotranspiration module (``SPAM``):
``WATBAL`` RATE (snow, mulch interception, SCS runoff, infiltration or drainage, upward flow)
runs first, ``SPAM`` then computes the soil evaporation and the root water extraction from the
water content left by it, and ``WATBAL`` INTEGR removes both and updates the water content
[DSSAT-CSM ``CSM_Main/LAND.for`` lines 311-386: ``SOIL`` RATE, ``SPAM`` RATE, ``PLANT`` RATE,
``SOIL`` INTEGR]. The two processes keep that split so that a DSSAT day places the
evapotranspiration module between them:

* :func:`bucket_rate` (``soil_water/tipping_bucket.rate@dssat-4.8.6.0:faithful``) writes the
  day's flux record :class:`BucketFluxes` (``SWDELTS``, ``SWDELTU``, runoff, drainage ...), the
  snow pack and the mulch interception;
* :func:`bucket_integrate` (``soil_water/tipping_bucket.integrate@dssat-4.8.6.0:faithful``) reads
  the root water uptake from the sink record ``soil_water.sink_in`` (P4, field ``uptake``: per
  layer, cm d-1, positive removes water; DSSAT ``SWDELTX = -uptake / DLAYR``), the soil
  evaporation ``ES`` and the mulch evaporation from the PET record (P5 ``soil_evaporation``,
  ``residue_evaporation``; see below) and integrates the water content, the mulch water and the
  truncations; it publishes ``theta`` (P7) on the core grid.

**P5 semantics.** DSSAT's ``SPAM`` hands ``WATBAL`` the *actual* soil evaporation ``ES`` (Ritchie's
two-stage ``SOILEV`` limited by the soil water) and the actual mulch evaporation ``EM``. Where the
evaporation processes of :mod:`agrijax.processes.soil_water.bucket_evap` do not compute them, the
bucket takes P5's ``soil_evaporation`` and ``residue_evaporation`` as those actual amounts (the
day-by-day validation against the reference replays them from the reference run; the assembled
DSSAT day binds the actual-evaporation record of ``bucket_evap`` instead). With the SALUS
evaporation (``MESEV = 'S'``) SPAM removes the evaporation layer by layer (``ESR_SoilEvap``, through
``SWDELTU``): the per-layer amounts come in the own input field :attr:`BucketState.evap_layers`
(cm d-1, positive removes), whose layer sum is ``ES``.

**Soil properties.** DSSAT's ``SOILDYN`` changes ``DLAYR``, ``DS``, ``LL`` and ``DUL`` daily when
the soil organic matter changes (``SOILDYN.for`` lines 1057-1152) and the tillage properties on a
tillage day; ``WATBAL`` reads ``SOILPROP`` at RATE and ``DLAYR`` again at INTEGR. The static soil
of :class:`BucketParams` is used unless the forcing carries the day's properties
(:attr:`BucketForcing.soil`, :attr:`BucketForcing.dlayr_end`: a replay of ``SOILDYN``).

**Conservation.** The bucket books every water movement it makes: rain, irrigation and the water of
newly added residue in; runoff, drainage below the profile, soil evaporation, root uptake and mulch
evaporation out; and the reference's truncations (the rounding of the water content to 1e-6, the
cuts of snow < 0.001 mm, mulch water < 1e-4 mm, water content < 1e-4, and the infiltration excess
< 1e-4 cm that ``INFIL`` does not place; ``PINF <= 1e-4`` cm that is not infiltrated) as one signed
outflow ``truncation``, so the ledger (:func:`bucket_ledger`) closes to floating-point rounding.
Storage is the profile water ``sum(SW DLAYR)`` plus the snow pack and the mulch water.

Not supported (the faithful keys raise nothing but the treatments that need them are out of scope,
and the DSSAT reference runs used for the validation never activate them:
``tests/integration/test_bucket_dssat.py`` checks the switches in every dump): puddled and bunded
(flooded) fields, tile drainage (``TDLNO > 0``), a managed water table (``ICWD``), tillage mixing of
the water content (``SoilMixing``), and the plastic-mulch runoff share only as a parameter (0 in the
reference set).

Source: DSSAT-CSM v4.8.6.0 ``Soil/SoilWater/WATBAL.for`` and callees (BSD-3, Copyright 1998-2026
DSSAT Foundation, University of Florida, International Fertilizer Development Center).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.grids import SoilGrid, remap_extensive, remap_intensive
from agrijax.core.ledger import Channel
from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field, get_path
from agrijax.core.units import CM_PER_MM, MM_PER_CM
from agrijax.iface.soil import SinkInputs
from agrijax.iface.surface import EvaporationRecord, PETFluxes

from ..coefficients import setting_field
from .coefficients import WATBAL_COEFFICIENTS, WatbalCoefficients
from .kernels import infil, integrate_sw, mulch_integrate, mulch_rate, rnoff, satflo, snowfall, up_flow

__all__ = [
    "BUCKET_LEDGER_INFLOWS",
    "BUCKET_LEDGER_OUTFLOWS",
    "BucketFluxes",
    "BucketForcing",
    "BucketParams",
    "BucketSoil",
    "BucketState",
    "MulchForcing",
    "bucket_integrate",
    "bucket_ledger_channels",
    "bucket_rate",
    "bucket_storage",
]

_L = ("n_layer",)
_TL = ("T", "n_layer")
_GRID = "dssat_layers"
_REF_BUILD = (
    "dscsm048 v4.8.6.0 build486 (gfortran 13.3, static); the instrumented build of the same source "
    "used for the day-by-day comparison"
)
_REAL = (
    "DSSAT single precision (REAL*4) is not reproduced",
    "the kernels run in float64 (float32 with AGRI_JAX_X64=0); the rounding to 1e-6 and the thresholds "
    "may land one quantum apart",
    "tests/integration/test_bucket_dssat.py tolerances",
)

#: the derivative through the 1e-6 rounding of SW (WATBAL.for:503-505)
_ROUND_STE = (
    "derivative (ste / implicit gradient modes) through the rounding SW = ANINT(SW*1E6)/1E6: identity "
    "(straight-through, round_st), not the derivative of the rounded value (0 between quanta); the "
    "forward value is DSSAT's in every mode",
    "a perturbation smaller than the 1e-6 quantum leaves SW unchanged, so the exact derivative cuts every "
    "path through the soil water; the straight-through one keeps it (the slope of the model without "
    "the quantum). With the root length density truncation of MZ_ROOTS it carries the difference "
    "between the ste and exact G2 / G3 derivatives of the DSSAT maize day",
    "scripts/diag/dssat_grad_gap.py; tests/integration/test_facade_grad.py::test_scenario_batch",
)


def _layers(unit: str, description: str, fortran_name: str = "", dims: tuple[str, ...] = _L) -> Any:
    return field(unit=unit, description=description, fortran_name=fortran_name, dims=dims, grid=_GRID)


def _scalar(unit: str, description: str, fortran_name: str = "", dims: tuple[str, ...] = ()) -> Any:
    return field(unit=unit, description=description, fortran_name=fortran_name, dims=dims)


# --------------------------------------------------------------------------- records
class BucketSoil(Params):
    """The soil properties ``WATBAL`` reads from ``SOILPROP`` (DSSAT layers; padded layers have
    ``dlayr = 0`` and come after the ``NLAYR`` real ones)."""

    dlayr: Array = _layers("cm", "layer thickness", "DLAYR")
    ds: Array = _layers("cm", "depth of the layer bottom", "DS")
    ll: Array = _layers("cm3 cm-3", "lower limit of plant-extractable water", "LL")
    dul: Array = _layers("cm3 cm-3", "drained upper limit", "DUL")
    sat: Array = _layers("cm3 cm-3", "saturated water content", "SAT")
    swcn: Array = _layers("cm h-1", "saturated hydraulic conductivity (<= 0: not given, no cap)", "SWCN")
    cn: Array = _scalar("-", "runoff curve number", "CN")
    swcon: Array = _scalar("d-1", "whole-profile drainage coefficient", "SWCON")


class BucketParams(Params):
    """Parameters of the tipping bucket.

    ``soil`` holds the ``.SOL`` values (``SLLL``, ``SDUL``, ``SSAT``, ``SSKS``, ``SLRO``, ``SLDR``)
    as DSSAT's ``SOILPROP`` carries them; ``mulch_on`` is ``MEINF`` in ``'RSM'`` (mulch effects on
    runoff and interception), ``salus_es`` is ``MESEV = 'S'`` (the SALUS evaporation removes water
    layer by layer through :attr:`BucketState.evap_layers`, and ``UP_FLOW`` is not called);
    ``actwtd`` is the water table depth ``INFIL`` compares with (1000 cm: none). ``core`` is the
    grid of the ports P4 and P7 (``None``: the DSSAT layers themselves); the layer grid ``layers``
    must then be given as well.
    """

    soil: BucketSoil
    mulch_on: Array = _scalar("-", "mulch effects on runoff and interception (MEINF in 'RSM')", "MEINF")
    salus_es: Array = _scalar("-", "SALUS layer-by-layer soil evaporation (MESEV = 'S')", "MESEV")
    actwtd: Array = _scalar("cm", "water table depth seen by INFIL (1000: no water table)", "ActWTD")
    pm_fraction: Array = _scalar("-", "plastic mulch cover fraction (PMFRACTION)", "PMFRACTION")
    coefficients: WatbalCoefficients | None = field(
        description="the numbers WATBAL and its callees hard-code (None: DSSAT-CSM v4.8.6.0)", default=None
    )
    real4_sw: bool = setting_field(
        "tipping_bucket.real4_sw",
        False,
        "-",
        "store the day's rounded water content in REAL*4 as DSSAT does (True: the REAL*4 soil-value "
        "convention of the DSSAT day, with REAL*4 soil limits; False: float64, as in the day-by-day "
        "validation of the bucket)",
        origin="agrijax",
        basis="DSSAT-CSM v4.8.6.0 holds SW in a REAL (Soil/SoilWater/WATBAL.for:503-505 ANINT(SW*1E6)/1E6); "
        "measured effect in tests/integration/test_day_dssat486_free.py",
    )
    layers: SoilGrid | None = eqx.field(static=True, default=None)
    core: SoilGrid | None = eqx.field(static=True, default=None)

    def coef(self) -> WatbalCoefficients:
        """The coefficients in force: :attr:`coefficients`, or the DSSAT values."""
        return WATBAL_COEFFICIENTS if self.coefficients is None else self.coefficients


class MulchForcing(Forcing):
    """The surface residue the soil organic matter module leaves for ``MULCHWATER`` (per day)."""

    mass: Array = _scalar("kg ha-1", "surface residue mass", "MULCHMASS", ("T",))
    cover: Array = _scalar("-", "fraction of the soil surface covered by residue", "MULCHCOVER", ("T",))
    new_mass: Array = _scalar("kg ha-1", "residue added to the surface today", "NEWMULCH", ("T",))
    watfac: Array = _scalar(
        "-", "water holding factor of the residue (mm per 1e-4 kg ha-1)", "MUL_WATFAC", ("T",)
    )


class BucketForcing(Forcing):
    """Daily inputs of the tipping bucket (time axis first).

    ``rain`` and ``tmax`` are the weather (``WEATHER%RAIN``, ``%TMAX``), ``irrigation`` the day's
    irrigation depth (``IRRAMT`` of ``MGMTOPS``). ``mulch`` is the residue record (``None``: no
    residue). ``soil`` and ``dlayr_end`` optionally replay ``SOILDYN``'s daily ``SOILPROP`` at RATE
    and its ``DLAYR`` at INTEGR (``None``: :attr:`BucketParams.soil` every day).
    """

    rain: Array = _scalar("mm d-1", "precipitation", "RAIN", ("T",))
    tmax: Array = _scalar("degC", "maximum air temperature", "TMAX", ("T",))
    irrigation: Array = _scalar("mm d-1", "irrigation", "IRRAMT", ("T",))
    mulch: MulchForcing | None = None
    soil: BucketSoil | None = None
    dlayr_end: Array | None = field(
        unit="cm",
        description="layer thickness at INTEGR (SOILDYN replay)",
        fortran_name="DLAYR",
        dims=_TL,
        grid=_GRID,
        default=None,
    )


class BucketFluxes(State):
    """The day's water movement of the bucket (DSSAT units: surface water mm, layer flows cm)."""

    watavl: Array = _scalar(
        "mm d-1", "water available for runoff and infiltration after snow and mulch", "WATAVL"
    )
    runoff: Array = _scalar("mm d-1", "surface runoff (SCS curve number plus infiltration excess)", "RUNOFF")
    infiltration: Array = _scalar("mm d-1", "water that entered the profile", "INFILT")
    drain: Array = _scalar("mm d-1", "drainage out of the bottom of the profile", "DRAIN")
    excs: Array = _scalar("cm d-1", "infiltration excess added to runoff", "EXCS")
    drn: Array = _layers("cm d-1", "drainage through the bottom of each layer", "DRN")
    upflow: Array = _layers("cm d-1", "flow from layer L+1 into layer L (negative: downward)", "UPFLOW")
    swdelts: Array = _layers("cm3 cm-3", "change of water content by infiltration or drainage", "SWDELTS")
    swdeltu: Array = _layers("cm3 cm-3", "change of water content by the unsaturated flow", "SWDELTU")
    mulch_intercept: Array = _scalar("mm d-1", "rain intercepted by the mulch", "MULWATADD")
    residue_water: Array = _scalar("mm d-1", "water brought in by new residue", "RESWATADD")
    truncation: Array = _scalar(
        "cm d-1", "water removed by the reference's cuts and rounding (negative: added)", ""
    )
    winf: Array = _scalar(
        "mm d-1",
        "water available for infiltration WATAVL - RUNOFF + IRRAMT (SOILEV reads it; before INFIL's excess)",
        "WINF",
    )

    @classmethod
    def zeros(cls, n_layer: int, dtype: Any = None) -> BucketFluxes:
        """No movement."""
        dt = dtype if dtype is not None else jnp.result_type(float)
        z = jnp.zeros((), dt)
        zl = jnp.zeros((n_layer,), dt)
        return cls(z, z, z, z, z, zl, zl, zl, zl, z, z, z, z)


class BucketState(State):
    """State of the tipping bucket (at ``soil_water``).

    ``sw`` is the water content of the DSSAT layers; ``theta`` its copy on the core grid (P7, the
    DSSAT layers when :attr:`BucketParams.core` is ``None``). ``sink_in`` (P4, on the core grid) and
    ``evap_layers`` (DSSAT layers) are inputs written by other modules before the integration; ``pet``
    is the P5 port. ``mulch_wat`` is ``MULCHWATER``'s saved mulch water and ``mulch_evap_prev`` the
    mulch evaporation ``MULCHWATER`` RATE sees (the previous SPAM call's).
    """

    sw: Array = _layers("cm3 cm-3", "volumetric water content", "SW")
    theta: Array = field(unit="cm3 cm-3", description="water content on the core grid (P7)", dims=("n_node",))
    snow: Array = _scalar("mm", "snow pack", "SNOW")
    mulch_wat: Array = _scalar("mm", "water held by the surface residue", "MULCHWAT")
    mulch_evap_prev: Array = _scalar("mm d-1", "mulch evaporation of the previous SPAM call", "MULCHEVAP")
    flux: BucketFluxes = field(description="the day's water movement")
    sink_in: SinkInputs = field(description="sinks of the day on the core grid (P4; uptake read)")
    evap_layers: Array = _layers(
        "cm d-1", "per-layer soil evaporation of the SALUS method (MESEV = 'S')", "SWDELTU"
    )
    pet: PETFluxes | EvaporationRecord = port(
        description="evaporation record: the actual soil and residue evaporation read (the RZWQM2-order "
        "day binds P5, the DSSAT day the actual-evaporation record PD1 iface.evaporation)"
    )

    @classmethod
    def initial(
        cls,
        sw: Any,
        *,
        n_core: int | None = None,
        snow: Any = 0.0,
        mulch_wat: Any = 0.0,
        mulch_evap_prev: Any = 0.0,
        dtype: Any = None,
    ) -> BucketState:
        """``SEASINIT``: the initial water content ``sw`` (``IPWBAL``), no fluxes, the P5 port empty.
        ``n_core``: the core-grid size (default: the layer count)."""
        dt = dtype if dtype is not None else jnp.result_type(float)
        sw = jnp.asarray(sw, dt)
        n = int(sw.shape[-1])
        theta = sw if n_core is None else jnp.zeros((int(n_core),), dt)
        return cls(
            sw=sw,
            theta=theta,
            snow=jnp.asarray(snow, dt),
            mulch_wat=jnp.asarray(mulch_wat, dt),
            mulch_evap_prev=jnp.asarray(mulch_evap_prev, dt),
            flux=BucketFluxes.zeros(n, dt),
            sink_in=SinkInputs.zeros(int(theta.shape[-1]), dt),
            evap_layers=jnp.zeros((n,), dt),
        )


# --------------------------------------------------------------------------- helpers
def _rate_soil(params: BucketParams, forcing_t: BucketForcing) -> BucketSoil:
    """The day's ``SOILPROP`` at RATE: the replayed one when the forcing has it, else the static soil.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:139-148 (SOILPROP read at every call).
    """
    return params.soil if forcing_t.soil is None else forcing_t.soil


def _end_dlayr(params: BucketParams, forcing_t: BucketForcing, soil: BucketSoil) -> Array:
    """``DLAYR`` at INTEGR: the replayed one, else the RATE thickness (no change in the day).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:140, 500 (SOILPROP DLAYR at INTEGR).
    """
    return soil.dlayr if forcing_t.dlayr_end is None else forcing_t.dlayr_end


def _to_layers(x: Array, params: BucketParams) -> Array:
    """An extensive per-cell amount on the core grid [cm] onto the DSSAT layers.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/LMATCH.for (overlap weights; core/grids.py).
    """
    if params.core is None or params.layers is None:
        return x
    return remap_extensive(x, params.core, params.layers)


def _to_core(sw: Array, params: BucketParams) -> Array:
    """The layer water content on the core grid (thickness-weighted mean; identity on the layers).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/LMATCH.for (LMATCH; core/grids.py).
    """
    if params.core is None or params.layers is None:
        return sw
    return remap_intensive(sw, params.layers, params.core)


# --------------------------------------------------------------------------- RATE
@process(
    reads=("sw", "snow", "mulch_wat", "mulch_evap_prev"),
    writes=("flux", "snow"),
    source="DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for RATE (BSD-3)",
    fortran_name="WATBAL",
    key="soil_water/tipping_bucket.rate@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid=_GRID,
    ref_build=_REF_BUILD,
    sources=(
        ("snow accumulation and melt", "Soil/SoilWater/WATBAL.for:278-288; WBSUBS.for SNOWFALL"),
        ("mulch interception", "Soil/Mulch/MULCHWAT.for RATE"),
        ("SCS curve-number runoff", "Soil/SoilWater/RNOFF.for"),
        ("infiltration and saturated drainage", "Soil/SoilWater/INFIL.for, SATFLO.for; WATBAL.for:371-402"),
        ("unsaturated upward flow", "Soil/SoilWater/WBSUBS.for UP_FLOW; WATBAL.for:413-425"),
    ),
    deviates=(_REAL,),
)
def bucket_rate(state: BucketState, params: BucketParams, forcing_t: BucketForcing) -> BucketState:
    """``WATBAL`` RATE of a non-flooded upland field: snow, mulch interception, runoff,
    infiltration (``PINF > 1e-4`` cm) or saturated flow, and the unsaturated flow (not with the
    SALUS evaporation). Reads the forcing's ``rain``, ``tmax``, ``irrigation``, ``mulch`` and, when
    given, ``soil``; writes the flux record and the snow pack.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:276-459 (BSD-3).
    """
    c = params.coef()
    soil = _rate_soil(params, forcing_t)
    sw = state.sw
    dt = sw.dtype
    sn = snowfall(forcing_t.tmax, forcing_t.rain, state.snow, c)
    mf = forcing_t.mulch
    zero = jnp.zeros((), dt)
    mass = zero if mf is None else mf.mass
    cover = zero if mf is None else mf.cover
    new_mass = zero if mf is None else mf.new_mass
    watfac = zero if mf is None else mf.watfac
    mr = mulch_rate(
        sn.watavl, state.mulch_wat, state.mulch_evap_prev, mass, cover, new_mass, watfac, params.mulch_on, c
    )
    ro = rnoff(soil.cn, soil.ll, soil.sat, sw, mr.watavl, cover, params.mulch_on, params.pm_fraction, c)
    winf = mr.watavl - ro.runoff + forcing_t.irrigation
    pinf = winf * CM_PER_MM
    wet = pinf > c.pinf_min
    inf = infil(soil.dlayr, soil.ds, soil.dul, soil.sat, sw, soil.swcn, soil.swcon, pinf, params.actwtd, c)
    sat = satflo(soil.dlayr, soil.dul, soil.sat, sw, soil.swcn, soil.swcon, c)
    swdelts = jnp.where(wet, inf.swdelts, sat.swdelts)
    drn = jnp.where(wet, inf.drn, sat.drn)
    drain = jnp.where(wet, inf.drain, sat.drain)
    excs = jnp.where(wet, inf.excs, zero)
    infilt = jnp.where(wet, jnp.sum(swdelts * soil.dlayr, axis=-1) * MM_PER_CM + drain, zero)
    runoff = ro.runoff + jnp.where(wet & (excs > 0.0), excs * MM_PER_CM, zero)
    uf = up_flow(soil.dlayr, soil.dul, soil.ll, soil.sat, sw, jnp.maximum(0.0, sw + swdelts), c)
    salus = jnp.asarray(params.salus_es, dtype=bool)
    swdeltu = jnp.where(salus, 0.0, uf.swdeltu)
    upflow = jnp.where(salus, 0.0, uf.upflow)
    # the reference's unaccounted water of the RATE step [cm]: PINF <= 1e-4 is not infiltrated,
    # INFIL drops an excess below 1e-4 cm, SNOWFALL cuts a pack below 0.001 mm
    trunc = jnp.where(wet, inf.lost, jnp.maximum(pinf, 0.0)) + sn.dropped * CM_PER_MM
    flux = BucketFluxes(
        watavl=mr.watavl,
        runoff=runoff,
        infiltration=infilt,
        drain=drain,
        excs=excs,
        drn=drn,
        upflow=upflow,
        swdelts=swdelts,
        swdeltu=swdeltu,
        mulch_intercept=mr.mulwatadd,
        residue_water=mr.reswatadd,
        truncation=trunc,
        winf=winf,
    )
    return eqx.tree_at(lambda s: (s.flux, s.snow), state, (flux, sn.snow))


# --------------------------------------------------------------------------- INTEGR
@process(
    reads=(
        "sw",
        "flux",
        "mulch_wat",
        "sink_in.uptake",
        "evap_layers",
        "pet.soil_evaporation",
        "pet.residue_evaporation",
    ),
    writes=("sw", "theta", "mulch_wat", "mulch_evap_prev", "flux.truncation"),
    source="DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for INTEGR (BSD-3)",
    fortran_name="WATBAL",
    key="soil_water/tipping_bucket.integrate@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid=_GRID,
    ref_build=_REF_BUILD,
    sources=(
        (
            "integration of the day's changes on yesterday's layer thickness",
            "Soil/SoilWater/WATBAL.for:467-506",
        ),
        ("soil evaporation from layer 1 (Ritchie)", "Soil/SoilWater/WATBAL.for:471-475"),
        ("mulch water update", "Soil/Mulch/MULCHWAT.for INTEGR"),
    ),
    deviates=(_REAL, _ROUND_STE),
)
def bucket_integrate(state: BucketState, params: BucketParams, forcing_t: BucketForcing) -> BucketState:
    """``WATBAL`` INTEGR: the day's water content from the RATE changes, the root uptake (P4
    ``sink_in.uptake``) and the soil evaporation (P5 ``soil_evaporation`` from layer 1, or
    :attr:`BucketState.evap_layers` with the SALUS method), rounded to 1e-6; the mulch water from
    the interception and the mulch evaporation (P5 ``residue_evaporation``). Reads the forcing's
    ``soil`` and ``dlayr_end`` when given. Publishes ``theta`` on the core grid.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:465-533, Soil/Mulch/MULCHWAT.for:157-167 (BSD-3).
    """
    c = params.coef()
    soil = _rate_soil(params, forcing_t)
    dy = soil.dlayr  # DLAYR_YEST: the thickness the day's rates were computed on
    dl = _end_dlayr(params, forcing_t, soil)
    f = state.flux
    dt = state.sw.dtype
    salus = jnp.asarray(params.salus_es, dtype=bool)
    uptake = _to_layers(state.sink_in.uptake, params)  # cm d-1 per layer
    es_mm = state.pet.soil_evaporation * MM_PER_CM
    em_mm = state.pet.residue_evaporation * MM_PER_CM
    safe = jnp.where(dy > 0.0, dy, 1.0)  # padded layers (dlayr = 0) take no water
    swdeltx = -uptake / safe
    swdeltu = jnp.where(salus[..., None], -state.evap_layers / safe, f.swdeltu)
    deltas = f.swdelts + swdeltu + swdeltx
    sw_new, rounding = integrate_sw(state.sw, dl, dy, es_mm, deltas, ~salus, c, real4=params.real4_sw)
    mw, m_drop = mulch_integrate(
        state.mulch_wat, f.mulch_intercept, f.residue_water, em_mm, params.mulch_on, c
    )
    trunc = f.truncation - jnp.sum(rounding, axis=-1) + m_drop * CM_PER_MM
    return eqx.tree_at(
        lambda s: (s.sw, s.theta, s.mulch_wat, s.mulch_evap_prev, s.flux.truncation),
        state,
        (sw_new, _to_core(sw_new, params).astype(dt), mw, em_mm.astype(dt), trunc),
    )


# --------------------------------------------------------------------------- ledger
#: ledger channels of the bucket (cm d-1 each; the soil evaporation of the SALUS method is the
#: layer sum of ``evap_layers``)
BUCKET_LEDGER_INFLOWS: tuple[str, ...] = ("rain", "irrigation", "residue_water")
BUCKET_LEDGER_OUTFLOWS: tuple[str, ...] = (
    "runoff",
    "drainage",
    "soil_evaporation",
    "transpiration",
    "residue_evaporation",
    "truncation",
)


def bucket_storage(state: BucketState, params: BucketParams, forcing_t: BucketForcing | None = None) -> Array:
    """Water stored by the bucket [cm]: ``sum(SW DLAYR)`` (today's thickness) + snow + mulch water.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:605-620 (SUMSW); Soil/SoilWater/WBAL.for.
    """
    dl = params.soil.dlayr
    if forcing_t is not None:
        soil = _rate_soil(params, forcing_t)
        dl = _end_dlayr(params, forcing_t, soil)
    return jnp.sum(state.sw * dl, axis=-1) + (state.snow + state.mulch_wat) * CM_PER_MM


def bucket_ledger_channels(at: str = "soil_water") -> tuple[dict[str, Channel], dict[str, Channel]]:
    """``(inflows, outflows)`` for :func:`agrijax.core.ledger.water_ledger`, with the bucket state at
    ``at``, its P5 record at ``iface.pet`` and the forcing of the bucket at the forcing root.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WBAL.for (the daily water balance it prints).
    """

    def own(s: Any) -> BucketState:
        return get_path(s, at)

    def evap(s: Any, p: Any, f: Any) -> Any:
        b = own(s)
        pet = get_path(s, "iface.pet")
        return jnp.where(
            jnp.asarray(p.salus_es, dtype=bool), jnp.sum(b.evap_layers, axis=-1), pet.soil_evaporation
        )

    inflows: dict[str, Channel] = {
        "rain": lambda s, p, f: f.rain * CM_PER_MM,
        "irrigation": lambda s, p, f: f.irrigation * CM_PER_MM,
        "residue_water": lambda s, p, f: own(s).flux.residue_water * CM_PER_MM,
    }
    outflows: dict[str, Channel] = {
        "runoff": lambda s, p, f: own(s).flux.runoff * CM_PER_MM,
        "drainage": lambda s, p, f: own(s).flux.drain * CM_PER_MM,
        "soil_evaporation": evap,
        "transpiration": lambda s, p, f: jnp.sum(own(s).sink_in.uptake, axis=-1),
        "residue_evaporation": lambda s, p, f: jnp.where(
            jnp.asarray(p.mulch_on, dtype=bool), get_path(s, "iface.pet").residue_evaporation, 0.0
        ),
        "truncation": lambda s, p, f: own(s).flux.truncation,
    }
    return inflows, outflows
