"""The DSSAT-CSM v4.8.6.0 day, run free: SOILDYN albedo, tipping bucket, SPAM, ROOTWU and CERES-Maize.

The daily assembly of the DSSAT-based model: the day of ``dscsm048`` for the maize treatments
(Priestley-Taylor PET, Ritchie or SALUS soil evaporation, nitrogen off), in the order of
``CSM_Main/LAND.for``: the weather, the management, every module's RATE call (SOIL, SPAM, PLANT),
every module's INTEGR call, then the water ledger. Every row of
:data:`agrijax.iface.contract.DSSAT_DAY_TABLE` runs its own process; nothing of the soil water, the
evapotranspiration or the uptake is replayed:

==========================  ================================================  ===============
entry                       process                                           ports out
==========================  ================================================  ===============
``weather.daily``           no-op (the weather record is forcing)             --
``events.apply``            no-op (irrigation depth is forcing)               --
``soil_water.albedo``       ``soil_water/soil_albedo`` (SOILDYN ALBEDO_avg)   PD2 MSALB
``soil_water.rate``         ``soil_water/tipping_bucket.rate`` (WATBAL RATE)  soil flux, snow
``pet.priestley_taylor``    ``pet/priestley_taylor:port_soil_albedo``         P5 EO
``spam.pse``                ``pet/spam_pse`` (PSE)                            P5 EOS
``spam.mulch_evap``         ``soil_water/mulch_evap`` (MULCH_EVAP)            SPAM store (EM)
``spam.soil_evaporation``   ``soil_water/soilev`` or ``/esr_soilevap``        PD1 ES, EM, EVAP
``spam.transpiration``      ``pet/spam_trans`` (TRANS)                        P5 EOP, P1 eop
``water_supply.<s>.rootwu`` ``water_supply/rootwu`` (ROOTWU)                  P1 trwup
``spam.xtract``             ``soil_water/xtract`` (EP, XTRACT)                P4 uptake, PD1 EP
``soil_water.integrate``    ``soil_water/tipping_bucket.integrate``           P7
``crops.<s>.layers_in``     :func:`layers_in_entry` (identity)                P1 sw, P9 swe
``crops.<s>.phenology``..   CERES-Maize, faithful growth (N off)              P2
``crops.<s>.canopy``        :func:`canopy_entry` (XHLAI = XLAI = LAI)         P6
``ledger.close``            :func:`ledger_entry`                              P11
==========================  ================================================  ===============

**Potential vs actual.** P5 (``iface.pet``) carries only the potential rates (``EO`` mm d-1,
``EOS / 10`` and ``EOP / 10`` cm d-1); the actual evaporation of SPAM (``ES``, ``EM``, ``EF``,
``EVAP = ES + EM + EF``, ``ES_LYR``) and the actual transpiration ``EP`` are the record PD1
(``iface.evaporation``, :class:`~agrijax.iface.surface.EvaporationRecord`), which ``TRANS`` reads for
``EVAP``, ``XTRACT`` for ``ES`` and the bucket's INTEGR for ``ES`` and ``EM``
(:data:`agrijax.iface.contract.DSSAT_PORTS`).

**Swapping the soil evaporation.** ``spam.soil_evaporation`` runs a registered implementation chosen
by key, not a switch inside the model: ``day_processes(soil_evaporation=key)`` and
``day_params(..., soil_evaporation=key)`` (:data:`SOIL_EVAPORATION_KEYS`: ``MESEV`` R is
``soil_water/soilev``, S ``soil_water/esr_soilevap``; the site's ``MESEV`` is the default).
:func:`resolve_soil_evaporation` checks the implementation against the entry's interface from its
declarations (:func:`soil_evaporation_problems`: units and dims of the SPAM store, reads, writes,
slot, grid) and derives how it removes the water (:class:`SoilEvaporation`), which sets the bucket's
and ``XTRACT``'s coupling in :func:`day_params`; the entry rejects params built for the other kind at
trace time (:class:`SoilEvaporationError`). The entry's reads and writes do not change, so the day,
its lags and the contract checks are the same for every implementation.

**What is still a replay (labelled).** Inputs that no module of the day produces, all of them
forcing, never state: the residue record of the soil organic matter module (``MULCHMASS``,
``MULCHCOVER``, ``NEWMULCH``, ``MUL_WATFAC``, ``MULCH_AM``, ``MUL_EXTFAC``: non-zero in 9 of the
72 reference runs), the daily soil properties of ``SOILDYN``'s organic-matter update
(:attr:`BucketForcing.soil <agrijax.processes.soil_water.bucket.BucketForcing.soil>`, only for the
runs whose ``DLAYR``, ``DS``, ``DUL`` or ``LL`` change; the bucket, the soil evaporation, the albedo
and ``XTRACT`` read them, ROOTWU and CERES read the static soil) and the weather's hourly mean
``TAVG`` of ``HMET``. The harness of ``tests/integration/test_day_dssat486_free.py`` reports the
effect of each.

**Soil values (a labelled convention).** DSSAT holds every soil property and the water content in
REAL*4, and its threshold tests are exact comparisons in REAL*4: a layer that ``XTRACT`` dried to
the lower limit has ``SW .LE. LL`` exactly and gives no more water. :func:`dssat_soil_values` with
``"real4"`` (the faithful default of :func:`day_params`) gives every module the REAL*4 value of each
soil property and stores the bucket's rounded water content in REAL*4 (``BucketParams.real4_sw``);
``"input"`` gives the decimal input values and a float64 water content. Mixing the two (REAL*4
water content against float64 limits, or the reverse) breaks the equalities: a float64 lower
limit was measured to make UFGA8201 t1's yield 62 % high with the reference's REAL*4 soil water,
and the free run measures the reverse mix (``tests/integration/test_day_dssat486_free.py``).

Global state (paths from the contract)::

    soil_water          BucketState (sw, theta P7, snow, mulch_wat, flux, sink_in P4, evap_layers)
    surface.soil_evap   SoilEvapState (the SOILEV store and SPAM's EOS_SOIL, EM, ES ...)
    surface.xtract      XtractState (TRWU)            surface.{albedo, pt, pse, trans}  port-only
    iface.pet           PETFluxes P5                  iface.evaporation   EvaporationRecord PD1
    iface.soil_albedo   SoilAlbedo PD2                iface.crop_water.<s> CropWaterIn P1
    iface.root.<s>      RootRecord P2                 iface.canopy.<s>    CanopyRecord P6
    iface.snow          SnowOut P9 (the crop's SNOW: swe = the bucket's snow pack)
    crops.<s>           CERES-Maize state             water_supply.<s>    ROOTWU state
    ledger.water        WaterLedger P11

Global params (:func:`day_params`): ``crop``, ``rootwu``, ``soil`` (BucketParams), ``evap``
(SoilEvapParams), ``spam`` (SpamParams), ``pt`` (PTAlbedoParams), ``albedo``, ``xtract``. Global
forcing: ``crop`` (CeresForcing, the weather; the ``snow``, ``sw``, ``eop``, ``trwup`` of a
:class:`~agrijax.processes.crop.ceres_maize.CeresReplayForcing` are not read), ``soil``
(BucketForcing), ``spam`` (``{"weather": SpamWeather, "mulch_am", "mulch_extfac"}``) and, for the
replay configuration only (:func:`replay_processes`), ``replay``.

Source: DSSAT-CSM v4.8.6.0 CSM_Main/LAND.for (RATE lines 288-352, INTEGR 355-410), Soil/SOIL.for,
SPAM/SPAM.for (RATE 248-455), Soil/SoilWater/WATBAL.for (RATE 276-459, INTEGR 465-533),
Plant/CERES-Maize/MZ_CERES.for (INTEGR 619-745), BSD-3.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.day import Day, Phase
from agrijax.core.ledger import WaterLedger, water_ledger
from agrijax.core.ports import Binding, bind, compose, detach
from agrijax.core.process import Process, lookup, process
from agrijax.core.state import field_metadata, get_path, set_path
from agrijax.core.units import CM_PER_MM, MM_PER_CM, UnitError, conversion_factor
from agrijax.iface.contract import DSSAT_DAY_REF, dssat_allowed_lags, dssat_day_entries
from agrijax.iface.crop import CanopyRecord, CropWaterIn, RootRecord
from agrijax.iface.surface import EvaporationRecord, PETFluxes, SnowOut, SoilAlbedo
from agrijax.models.day_dssat486_adapters import (
    CANOPY_KEY,
    LAYERS_IN_KEY,
    CanopyInState,
    LayersInState,
    canopy_from_ceres,
    layers_in,
)
from agrijax.models.day_rzwqm46 import noop_entry, replay_entry
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    CeresMaizeParams,
    CeresMaizeState,
    plantgro_outputs,
)
from agrijax.processes.pet.spam_dssat import (
    PTAlbedoParams,
    PTAlbedoState,
    SpamParams,
    SpamPSEState,
    SpamTransState,
    spam_potential_soil_evaporation,
    spam_potential_transpiration,
    spam_priestley_taylor,
)
from agrijax.processes.soil_water.bucket import (
    BucketParams,
    BucketSoil,
    BucketState,
    bucket_integrate,
    bucket_rate,
)
from agrijax.processes.soil_water.bucket_evap import (
    SoilAlbedoParams,
    SoilAlbedoState,
    SoilEvapParams,
    SoilEvapState,
    XtractParams,
    XtractState,
    soil_albedo_rate,
    soil_evaporation_esr,
    soil_evaporation_mulch,
    soil_evaporation_soilev,
    spam_xtract,
)
from agrijax.processes.water_supply import RootwuParams, RootwuState, rootwu_supply

__all__ = [
    "CANOPY_KEY",
    "CROP_ENTRIES",
    "LAYERS_IN_KEY",
    "LEDGER_INFLOWS",
    "LEDGER_OUTFLOWS",
    "MESEV",
    "SLOT",
    "SOIL_EVAPORATION_GRIDS",
    "SOIL_EVAPORATION_INPUTS",
    "SOIL_EVAPORATION_KEYS",
    "SOIL_EVAPORATION_LAYERED",
    "SOIL_EVAPORATION_OPTIONAL",
    "SOIL_EVAPORATION_REQUIRED",
    "SOIL_EVAPORATION_SLOT",
    "SOIL_EVAPORATION_STORE",
    "SOIL_EVAPORATION_UPSTREAM",
    "SOIL_VALUES",
    "CanopyInState",
    "DssatSite",
    "LayersInState",
    "SoilEvaporation",
    "SoilEvaporationError",
    "canopy_entry",
    "day_dssat486",
    "day_outputs",
    "day_params",
    "day_processes",
    "dssat_soil_values",
    "initial_state",
    "layers_in",
    "layers_in_entry",
    "ledger_entry",
    "ledger_initial",
    "mulch_evap_entry",
    "replay_processes",
    "resolve_soil_evaporation",
    "soil_evaporation_entry",
    "soil_evaporation_gate_problems",
    "soil_evaporation_params_problems",
    "soil_evaporation_problems",
    "soil_storage",
    "transpiration_entry",
    "trwup_replay_entry",
]

#: the crop slot of the assembly
SLOT = "maize"
#: the CERES-Maize entries of the INTEGR phase, in the ``MZ_CERES`` order
CROP_ENTRIES: tuple[str, ...] = ("phenology", "stress", "growth", "roots", "publish")
#: the soil evaporation methods (``MESEV``): Ritchie two-stage ``SOILEV`` and SALUS ``ESR_SoilEvap``
MESEV: tuple[str, ...] = ("R", "S")
#: the soil-value conventions of :func:`dssat_soil_values`
SOIL_VALUES: tuple[str, ...] = ("real4", "input")

#: ledger channels [cm d-1]: the day's water in, and out of the profile, snow and mulch
LEDGER_INFLOWS: tuple[str, ...] = ("rain", "irrigation", "residue_water")
LEDGER_OUTFLOWS: tuple[str, ...] = (
    "runoff",
    "drainage",
    "soil_evaporation",
    "residue_evaporation",
    "transpiration",
    "truncation",
)

_EVAP = "surface.soil_evap"
_PET = "iface.pet"
_EV = "iface.evaporation"
_ALB = "iface.soil_albedo"
_SNOW = "iface.snow"


def _p(slot: str) -> dict[str, str]:
    """Global paths of the per-crop ports and states of ``slot``."""
    return {
        "crop_water": f"iface.crop_water.{slot}",
        "root": f"iface.root.{slot}",
        "canopy": f"iface.canopy.{slot}",
        "crop": f"crops.{slot}",
        "rootwu": f"water_supply.{slot}",
    }


def day_dssat486(slot: str = SLOT) -> Day:
    """The DSSAT-CSM v4.8.6.0 day for crop ``slot`` (:data:`~agrijax.iface.contract.DSSAT_DAY_TABLE`)
    with its allowed lags (:data:`~agrijax.iface.contract.DSSAT_DAY_LAGS`), and the contract's port
    owners and declared phased writes (``contract_slot``:
    :func:`~agrijax.iface.contract.dssat_port_owners`,
    :data:`~agrijax.iface.contract.DSSAT_PHASED_WRITES`)."""
    return Day(
        ref=DSSAT_DAY_REF,
        phases=tuple(Phase(ph, entries) for ph, entries in dssat_day_entries(slot)),
        lags=dssat_allowed_lags(slot),
        contract_slot=slot,
    )


# ------------------------------------------------------------------------ soil values and params
def dssat_soil_values(x: Any, convention: str = "real4") -> np.ndarray:
    """A soil property as the day's modules receive it (a labelled convention, see the module
    docstring): ``"real4"`` is the value DSSAT holds after reading the number into a REAL (the
    nearest ``float32``, promoted exactly to ``float64``); with it :func:`day_params` also stores
    the bucket's rounded water content in REAL*4 (``BucketParams.real4_sw``), so the REAL*4 equality
    tests of the reference (``SW .LE. LL`` in ``ROOTWU`` and ``XTRACT`` on a layer dried to its lower
    limit, ``SW`` at ``DUL`` or ``SAT``) hold as they do in the reference. ``"input"`` is the input
    number as given, with the water content in float64 (its rounding to 1e-6 then lands on the
    decimal soil limits of the input files).

    Source: DSSAT-CSM v4.8.6.0 ModuleDefs.for (SoilType: REAL DLAYR, DS, DUL, LL, SAT, ...),
    SPAM/ROOTWU.for and SPAM/SPSUBS.for XTRACT (SW .GT. LL tests), BSD-3.
    """
    if convention not in SOIL_VALUES:
        raise ValueError(f"unknown soil-value convention {convention!r} (known: {SOIL_VALUES})")
    a = np.asarray(x, dtype=np.float64)
    return a.astype(np.float32).astype(np.float64) if convention == "real4" else a


@dataclass(frozen=True)
class DssatSite:
    """The static soil and switches of one DSSAT run (host-side NumPy, before :func:`day_params`):
    the ``SOILPROP`` layers (``NLAYR`` real layers, then padding with ``dlayr = 0``), the surface
    values ``SALB``, ``U`` (``SLU1``), ``CN``, ``SWCON``, the simulation options ``MESEV`` (``R`` or
    ``S``) and ``MEINF`` (mulch on for ``R``, ``S``, ``M``), the water table depth ``ActWTD`` and the
    plastic-mulch fraction."""

    dlayr: np.ndarray
    ds: np.ndarray
    ll: np.ndarray
    dul: np.ndarray
    sat: np.ndarray
    swcn: np.ndarray
    cn: float
    swcon: float
    salb: float
    u: float
    mesev: str = "R"
    meinf: str = "S"
    actwtd: float = 1000.0
    pmfraction: float = 0.0

    @property
    def mulch_on(self) -> bool:
        """``INDEX('RSM', MEINF) > 0``."""
        return bool(self.meinf.strip()) and self.meinf in "RSM"


def _soil(site: DssatSite, convention: str) -> BucketSoil:
    v = lambda x: jnp.asarray(dssat_soil_values(x, convention))  # noqa: E731
    return BucketSoil(
        dlayr=v(site.dlayr),
        ds=v(site.ds),
        ll=v(site.ll),
        dul=v(site.dul),
        sat=v(site.sat),
        swcn=v(site.swcn),
        cn=v(site.cn),
        swcon=v(site.swcon),
    )


def day_params(
    site: DssatSite,
    crop: CeresMaizeParams,
    *,
    ksevap: Any,
    ktrans: Any,
    soil_values: str = "real4",
    rootwu: RootwuParams | None = None,
    soil_evaporation: str | Process | SoilEvaporation | None = None,
) -> dict[str, Any]:
    """The global params of the day for one run: every module's soil from ``site`` under the
    soil-value convention ``soil_values`` (:func:`dssat_soil_values`; with ``"real4"`` the crop's
    ``ll``, ``dul``, ``sat`` and ``dlayr`` too, so every module compares the same numbers), SPAM's
    extinction coefficients ``ksevap`` and ``ktrans`` (``KSEVAP = KTRANS = KEP`` of CERES-Maize),
    and the DSSAT-CSM v4.8.6.0 coefficients everywhere. ``rootwu`` overrides ROOTWU's parameters
    (its coefficients); its soil is replaced by the site's. ``soil_evaporation`` is the soil
    evaporation the day runs (:func:`resolve_soil_evaporation`; default: the site's ``MESEV``): how it
    removes the water sets the bucket's and ``XTRACT``'s ``salus_es`` and the static
    ``SoilEvapParams.layer_removal`` the ``spam.soil_evaporation`` entry checks, so params built for
    one implementation cannot run another silently."""
    se = resolve_soil_evaporation(site.mesev if soil_evaporation is None else soil_evaporation)
    soil = _soil(site, soil_values)
    real = lambda x: jnp.asarray(dssat_soil_values(x, soil_values))  # noqa: E731
    one = lambda b: jnp.asarray(1.0 if b else 0.0)  # noqa: E731
    salus = se.layer_removal
    bucket = BucketParams(
        soil=soil,
        mulch_on=one(site.mulch_on),
        salus_es=one(salus),
        actwtd=jnp.asarray(float(site.actwtd)),
        pm_fraction=jnp.asarray(float(site.pmfraction)),
        real4_sw=soil_values == "real4",
    )
    if soil_values == "real4":
        cs = crop.soil
        crop = crop.replace(soil=cs.replace(dlayr=soil.dlayr, ll=soil.ll, dul=soil.dul, sat=soil.sat))
    rw = RootwuParams(dlayr=soil.dlayr, ll=soil.ll, sat=soil.sat)
    if rootwu is not None:
        rw = rootwu.replace(dlayr=soil.dlayr, ll=soil.ll, sat=soil.sat)
    return {
        "crop": crop,
        "rootwu": rw,
        "soil": bucket,
        "evap": SoilEvapParams(
            u=real(site.u),
            pmfraction=jnp.asarray(float(site.pmfraction)),
            mulch_active=site.mulch_on,
            layer_removal=salus,
        ),
        "spam": SpamParams(ksevap=jnp.asarray(ksevap), ktrans=jnp.asarray(ktrans)),
        "pt": PTAlbedoParams(),
        "albedo": SoilAlbedoParams(salb=real(site.salb), dul1=soil.dul[..., 0], mulch_on=one(site.mulch_on)),
        "xtract": XtractParams(dlayr=soil.dlayr, ll=soil.ll, salus_es=one(salus)),
    }


def _day_soil(params: Any, forcing_t: Any) -> BucketSoil:
    """The day's ``SOILPROP``: the ``SOILDYN`` replay of the soil forcing, else the static soil.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for RATE (SOILPROP of the day), BSD-3.
    """
    f = forcing_t["soil"]
    return get_path(params, "soil").soil if f.soil is None else f.soil


# ------------------------------------------------------------------------ SPAM entries
def mulch_evap_entry() -> Process:
    """``spam.mulch_evap``: ``soil_water/mulch_evap`` on the SPAM store, with the start-of-day mulch
    water of the bucket (``soil_water.mulch_wat``, a one-day lag: ``MULCHWATER`` INTEGR runs after
    SPAM), the potential soil evaporation of P5 and the day's residue record (forcing)."""

    def _mulch_evap(state: Any, params: Any, forcing_t: Any) -> Any:
        """SPAM's mulch step on the store (EM, EOS_SOIL) from EOS, MULCHWAT and the residue record.

        Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for:338-346, Soil/Mulch/MULCHEVAP.for (BSD-3).
        """
        se: SoilEvapState = get_path(state, _EVAP)
        mf = forcing_t["soil"].mulch
        sf = forcing_t["spam"]
        dt = se.em.dtype
        z = jnp.zeros((), dt)
        view = se.replace(
            mulch_mass=z if mf is None else jnp.asarray(mf.mass, dt),
            mulch_cover=z if mf is None else jnp.asarray(mf.cover, dt),
            mulch_water=get_path(state, "soil_water.mulch_wat"),
            mulch_am=jnp.asarray(sf["mulch_am"], dt),
            mulch_extfac=jnp.asarray(sf["mulch_extfac"], dt),
            pet=get_path(state, _PET),
        )
        new = soil_evaporation_mulch(view, get_path(params, "evap"), None)
        return set_path(state, _EVAP, detach(new))

    return process(
        _mulch_evap,
        reads=(_EVAP, "soil_water.mulch_wat", f"{_PET}.soil_evaporation"),
        writes=(_EVAP,),
        name="spam.mulch_evap",
        register=False,
        source="DSSAT-CSM v4.8.6.0 SPAM/SPAM.for MULCH_EVAP call "
        "(soil_water/mulch_evap@dssat-4.8.6.0:faithful)",
    )


_EV_FIELDS = (
    "soil_evaporation",
    "residue_evaporation",
    "flood_evaporation",
    "evaporation",
    "soil_evaporation_layers",
)


# ------------------------------------------------------------------------ soil evaporation implementations
def _registered_key(proc: Process) -> str:
    """The registry key of a keyed process (the built-in implementations are all keyed)."""
    if proc.key is None:
        raise SoilEvaporationError(f"process {proc.name!r} has no registry key")
    return proc.key


class SoilEvaporationError(ValueError):
    """A soil evaporation does not fit the ``spam.soil_evaporation`` entry, or the day's params were
    built for an implementation that removes the water differently."""


#: the registered soil evaporation each value of DSSAT's ``MESEV`` names: the reference's static
#: switch, written as registry keys (:func:`day_processes` and :func:`day_params` take either)
SOIL_EVAPORATION_KEYS: dict[str, str] = {
    "R": _registered_key(soil_evaporation_soilev),
    "S": _registered_key(soil_evaporation_esr),
}
#: the registry slot of a soil evaporation implementation
SOIL_EVAPORATION_SLOT = "soil_water"
#: what ``spam.soil_evaporation`` gives an implementation, as fields of the SPAM store
#: (:class:`SoilEvapState`, whose units and dims are the interface): the day's inputs the entry sets
#: before the call (the start-of-day water P7, WATBAL RATE's SWDELTS / SWDELTU / WINF, the day's soil)
SOIL_EVAPORATION_INPUTS: tuple[str, ...] = ("sw", "swdelts", "swdeltu", "winf", "dlayr", "ds", "dul", "ll")
#: the store's fields the mulch entry wrote earlier in the day (read-only for the implementation)
SOIL_EVAPORATION_UPSTREAM: tuple[str, ...] = (
    "eos_soil",
    "em",
    "mulch_mass",
    "mulch_cover",
    "mulch_water",
    "mulch_am",
    "mulch_extfac",
)
#: the implementation's own store carried across days (read and written)
SOIL_EVAPORATION_STORE: tuple[str, ...] = ("sumes1", "sumes2", "t", "swef")
#: outputs every implementation writes (PD1 ``soil_evaporation`` = ES / 10, ``evaporation`` = EVAP)
SOIL_EVAPORATION_REQUIRED: tuple[str, ...] = ("es", "evap")
#: outputs of an implementation that removes water layer by layer (PD1 ``soil_evaporation_layers``,
#: PD6 ``evap_layers = -SWDELTU DLAYR``): all of them or none
SOIL_EVAPORATION_LAYERED: tuple[str, ...] = ("es_lyr", "swdeltu")
#: outputs the entry does not publish (diagnostics of the store)
SOIL_EVAPORATION_OPTIONAL: tuple[str, ...] = ("upflow",)
#: the grids an implementation may run on
SOIL_EVAPORATION_GRIDS: tuple[str, ...] = ("point", "dssat_layers")


def _state_class(proc: Process) -> type | None:
    """The dataclass annotated on the process's state argument (None when there is none)."""
    try:
        hints = typing.get_type_hints(proc.fn)
        first = next(iter(inspect.signature(proc.fn).parameters))
    except (NameError, TypeError, ValueError, StopIteration):
        return None
    cls = hints.get(first)
    return cls if isinstance(cls, type) and dataclasses.is_dataclass(cls) else None


def _unit_problem(name: str, got: str, want: str) -> str | None:
    """A unit mismatch of field ``name`` (the implementation's ``got`` against the store's ``want``)."""
    try:
        if conversion_factor(got, want) == 1.0:
            return None
    except UnitError:
        pass
    return f"field {name!r} is in {got!r} in the implementation, the SPAM store holds {want!r}"


def soil_evaporation_problems(proc: Process) -> list[str]:
    """Why ``proc`` cannot fill ``spam.soil_evaporation`` (empty: it can).

    Checked from the process's declarations, never by running it: a registry key in the
    ``soil_water`` slot and a grid of :data:`SOIL_EVAPORATION_GRIDS`; a state argument annotated
    with a state class whose read and written fields have the units and dims of the SPAM store
    (:class:`SoilEvapState`); reads inside what the entry gives (:data:`SOIL_EVAPORATION_INPUTS`,
    :data:`SOIL_EVAPORATION_UPSTREAM`, :data:`SOIL_EVAPORATION_STORE`: no port, P5 is the mulch
    entry's); writes covering :data:`SOIL_EVAPORATION_REQUIRED` and inside the outputs and the store
    (the bucket owns the soil water and the soil properties); :data:`SOIL_EVAPORATION_LAYERED` all
    written or none.
    """
    out: list[str] = []
    info = proc.info
    if info is None:
        out.append(
            f"process {proc.name!r} has no registry key (slot/impl@ref_version:variant) and no "
            "provenance: register it with @process(key=...)"
        )
    else:
        if info.slot != SOIL_EVAPORATION_SLOT:
            out.append(f"{info.key}: slot {info.slot!r}, the entry takes a {SOIL_EVAPORATION_SLOT!r} process")
        if info.grid not in SOIL_EVAPORATION_GRIDS:
            out.append(f"{info.key}: grid {info.grid!r}, the entry runs on {SOIL_EVAPORATION_GRIDS}")
    store = field_metadata(SoilEvapState)
    cls = _state_class(proc)
    if cls is None:
        out.append(
            f"{proc.name}: the state argument is not annotated with a state class, so the units and dims "
            "of its fields cannot be checked against the SPAM store (SoilEvapState)"
        )
    readable = (*SOIL_EVAPORATION_INPUTS, *SOIL_EVAPORATION_UPSTREAM, *SOIL_EVAPORATION_STORE)
    writable = (
        *SOIL_EVAPORATION_REQUIRED,
        *SOIL_EVAPORATION_LAYERED,
        *SOIL_EVAPORATION_OPTIONAL,
        *SOIL_EVAPORATION_STORE,
    )
    heads = {p.split(".", 1)[0] for p in (*proc.reads, *proc.writes)}
    for path in proc.reads:
        if path.split(".", 1)[0] not in readable:
            out.append(f"{proc.name} reads {path!r}: the entry gives only {readable}")
    for path in proc.writes:
        if "." in path and path.split(".", 1)[0] in writable:
            out.append(
                f"{proc.name} writes {path!r}: a sub-path of an array field; declare the whole field "
                f"{path.split('.', 1)[0]!r}"
            )
        elif path.split(".", 1)[0] not in writable:
            out.append(
                f"{proc.name} writes {path!r}: not an output of the soil evaporation {writable} (the bucket "
                "owns the soil water and the soil properties, the mulch entry EOS_SOIL and EM)"
            )
    written = _whole_writes(proc)
    missing = [n for n in SOIL_EVAPORATION_REQUIRED if n not in written]
    if missing:
        out.append(f"{proc.name} does not write {missing} (PD1: ES / 10 and EVAP)")
    layered = [n for n in SOIL_EVAPORATION_LAYERED if n in written]
    if layered and len(layered) != len(SOIL_EVAPORATION_LAYERED):
        out.append(
            f"{proc.name} writes {layered} of {SOIL_EVAPORATION_LAYERED}: a layer-by-layer removal "
            "writes all of them (the bucket applies SWDELTU, XTRACT reads ES_LYR)"
        )
    if cls is not None:
        mine = field_metadata(cls)
        for name in sorted(heads & set(store)):
            if name not in mine:
                out.append(f"{proc.name} declares {name!r}, which its state class {cls.__name__} lacks")
                continue
            got, want = mine[name]["unit"], store[name]["unit"]
            problem = _unit_problem(name, got, want)
            if problem is not None:
                out.append(f"{proc.name}: {problem}")
            if mine[name]["dims"] != store[name]["dims"]:
                out.append(
                    f"{proc.name}: field {name!r} has dims {mine[name]['dims']}, the store "
                    f"{store[name]['dims']}"
                )
    return out


def _whole_writes(proc: Process) -> set[str]:
    """The fields ``proc`` declares as whole writes (a sub-path write does not count)."""
    return {p for p in proc.writes if "." not in p}


def _layer_removal(proc: Process) -> bool:
    """Does ``proc`` remove water layer by layer (writes the whole ``ES_LYR`` and ``SWDELTU``)?"""
    return set(SOIL_EVAPORATION_LAYERED) <= _whole_writes(proc)


# the conformance gate: synthetic profiles on five DSSAT layers (the thicknesses of the maize
# examples' top layers, the limits of testing/conformance/_builtin/dssat_evap.py), run once when an
# implementation is resolved. They are test inputs, not model coefficients.
#: layer thickness [cm], lower limit and drained upper limit [cm3 cm-3] of the gate's soil
_GATE_DLAYR = (5.0, 10.0, 15.0, 15.0, 15.0)
_GATE_LL = (0.08, 0.09, 0.10, 0.11, 0.12)
_GATE_DUL = (0.20, 0.21, 0.22, 0.23, 0.24)
#: the water content of each case, as the fraction of the way from LL to DUL (above 1: wet;
#: about 0.5: intermediate; low: dry), and its potential soil evaporation EOS_SOIL [mm d-1]
#: (0: the call gate; 12: above what the profile can give)
_GATE_WETNESS = (1.1, 0.8, 0.5, 0.2, 1.1, 0.5)
_GATE_EOS_SOIL = (4.0, 4.0, 4.0, 4.0, 0.0, 12.0)
#: the mulch evaporation EM [mm d-1] and the stage-1 limit U [mm] of every case
_GATE_EM = 0.3
_GATE_U = 6.0
#: the gate's tolerance in units of the dtype's epsilon, relative to max(EOS_SOIL, 1 mm d-1)
_GATE_EPS = 1024.0


def soil_evaporation_gate_problems(proc: Process) -> list[str]:
    """Run ``proc`` once on the gate's synthetic profiles (:data:`_GATE_WETNESS`: wet, intermediate,
    dry, the call gate ``EOS_SOIL = 0`` and an ``EOS_SOIL`` above the supply) and check what the
    bucket and the ledger rely on: finite ``ES`` and ``EVAP``, ``0 <= ES <= EOS_SOIL``, ``EVAP = ES +
    EM``; for a layer-by-layer removal also ``ES_LYR >= 0``, ``ES = sum(ES_LYR)`` and ``ES_LYR = -SWDELTU
    DLAYR 10`` (the water the bucket removes is the water booked). Tolerance: :data:`_GATE_EPS`
    epsilons of the run's dtype. Empty when every check holds. These are necessary conditions, not a
    validation: an implementation is validated against its reference in the integration tier
    (``testing/conformance/_builtin/dssat_evap.py`` has the conformance cases of the built-in ones).
    Runs eagerly, also when the day is assembled inside a ``jit`` trace."""
    with jax.ensure_compile_time_eval():
        return _gate(proc)


def _gate(proc: Process) -> list[str]:
    """:func:`soil_evaporation_gate_problems` on concrete values."""
    n = len(_GATE_WETNESS)
    dl = jnp.broadcast_to(jnp.asarray(_GATE_DLAYR), (n, len(_GATE_DLAYR)))
    ll = jnp.broadcast_to(jnp.asarray(_GATE_LL), dl.shape)
    dul = jnp.broadcast_to(jnp.asarray(_GATE_DUL), dl.shape)
    sw = ll + jnp.asarray(_GATE_WETNESS)[:, None] * (dul - ll)
    ds = jnp.cumsum(dl, axis=-1)
    dt = sw.dtype
    u = jnp.asarray(_GATE_U, dt)
    eos = jnp.asarray(_GATE_EOS_SOIL, dt)
    em = jnp.full((n,), _GATE_EM, dt)
    zl = jnp.zeros_like(sw)
    state = SoilEvapState.initial(sw, dl, ds, dul, ll, u).replace(
        eos_soil=eos, em=em, swdelts=zl, swdeltu=zl, winf=jnp.zeros((n,), dt), pet=None
    )
    params = SoilEvapParams(u=u, pmfraction=jnp.asarray(0.0, dt))
    try:
        new = proc(state, params, None)
        es, evap = np.asarray(new.es), np.asarray(new.evap)
        es_lyr, swdeltu = np.asarray(new.es_lyr), np.asarray(new.swdeltu)
    except Exception as e:  # any failure of a foreign implementation is a gate failure
        return [f"{proc.name} does not run on the gate's profiles: {type(e).__name__}: {e}"]
    tol = _GATE_EPS * float(jnp.finfo(dt).eps) * np.maximum(np.asarray(eos), 1.0)
    out: list[str] = []

    def bad(what: str, ok: np.ndarray) -> None:
        if not bool(np.all(ok)):
            out.append(f"{proc.name}: {what} fails on gate cases {np.nonzero(~ok)[0].tolist()}")

    bad("finite ES and EVAP", np.isfinite(es) & np.isfinite(evap))
    bad("ES >= 0", es >= -tol)
    bad("ES <= EOS_SOIL", es <= np.asarray(eos) + tol)
    bad("EVAP = ES + EM", np.abs(evap - (es + np.asarray(em))) <= tol)
    if _layer_removal(proc):
        tl = tol[:, None]
        bad("ES_LYR >= 0", np.all(es_lyr >= -tl, axis=-1))
        bad("ES = sum(ES_LYR)", np.abs(es - es_lyr.sum(-1)) <= tol)
        booked = -swdeltu * np.asarray(dl) * MM_PER_CM
        bad("ES_LYR = -SWDELTU DLAYR 10", np.all(np.abs(es_lyr - booked) <= tl, axis=-1))
    return out


@dataclass(frozen=True)
class SoilEvaporation:
    """A checked soil evaporation for ``spam.soil_evaporation``: constructing one runs
    :func:`soil_evaporation_problems` and :func:`soil_evaporation_gate_problems` and raises
    :class:`SoilEvaporationError` with every problem, so an instance is always a checked one.

    ``layer_removal`` is read from the declared writes: an implementation that writes ``ES_LYR`` and
    ``SWDELTU`` (as ``ESR_SoilEvap``) removes water layer by layer, so the bucket applies ``SWDELTU``
    to every layer and runs no ``UP_FLOW``, and ``XTRACT`` lowers each layer's available water by
    ``ES_LYR``; otherwise (as ``SOILEV``) the bucket removes ``ES`` from the top layer. These are the
    ``salus_es`` flags :func:`day_params` sets (``WATBAL.for:471-495``, ``SPAM.for:422-432``)."""

    process: Process

    def __post_init__(self) -> None:
        proc = self.process
        problems = soil_evaporation_problems(proc)
        if not problems:
            problems = soil_evaporation_gate_problems(proc)
        if problems:
            raise SoilEvaporationError(
                f"{proc.key or proc.name} cannot fill spam.soil_evaporation:\n  " + "\n  ".join(problems)
            )

    @property
    def layer_removal(self) -> bool:
        """Does the implementation remove water layer by layer (from its declared writes)?"""
        return _layer_removal(self.process)

    @property
    def key(self) -> str:
        """The registry key of the implementation."""
        return _registered_key(self.process)

    @property
    def mesev(self) -> str | None:
        """The DSSAT ``MESEV`` that names this implementation, None for another one."""
        return next((m for m, k in SOIL_EVAPORATION_KEYS.items() if k == self.key), None)


def resolve_soil_evaporation(impl: str | Process | SoilEvaporation) -> SoilEvaporation:
    """The checked implementation ``impl``: a DSSAT ``MESEV`` code (``"R"``, ``"S"``,
    :data:`SOIL_EVAPORATION_KEYS`), a registry key (looked up in
    :data:`agrijax.core.process.registry`) or a process; raises :class:`SoilEvaporationError` with
    every problem (:class:`SoilEvaporation`). A given :class:`SoilEvaporation` (or subclass, or an
    instance made without its constructor) is checked again: only its process is kept."""
    if isinstance(impl, SoilEvaporation):
        proc = getattr(impl, "process", None)
        if not isinstance(proc, Process):
            raise SoilEvaporationError(f"{type(impl).__name__} instance without a process")
    elif isinstance(impl, str):
        key = SOIL_EVAPORATION_KEYS.get(impl, impl)
        if "/" not in key:
            raise SoilEvaporationError(
                f"MESEV {impl!r} not in {MESEV} and not a registry key (slot/impl@ref_version:variant)"
            )
        try:
            proc = lookup(key)
        except (KeyError, ValueError) as e:
            raise SoilEvaporationError(f"no registered process {key!r} ({e})") from e
    else:
        proc = impl
    return SoilEvaporation(proc)


#: the params entries whose ``salus_es`` flag couples a module to the soil evaporation's removal
_COUPLED_FLAGS: tuple[str, ...] = ("soil", "xtract")
_HOW = {True: "layer by layer", False: "from the top layer"}


def soil_evaporation_params_problems(
    params: Mapping[str, Any], impl: str | Process | SoilEvaporation
) -> list[str]:
    """Why the day's global ``params`` do not fit the soil evaporation ``impl`` (empty: they do).

    The coupling itself is checked: the bucket's and ``XTRACT``'s ``salus_es`` flags (``params["soil"]``,
    ``params["xtract"]``, every sample of a batch) must say the removal of ``impl``, and so must the
    static ``SoilEvapParams.layer_removal`` that :func:`day_params` records. Inside ``jit`` the flags
    are traced and only the static record can be read: then it must be stated (not None). Call this on
    the host before a run to check the flags themselves; ``spam.soil_evaporation`` calls it at trace
    time."""
    se = resolve_soil_evaporation(impl)
    want = se.layer_removal
    out: list[str] = []
    fix = f"build the params with day_params(..., soil_evaporation={se.key!r})"
    stated = get_path(params, "evap").layer_removal
    if stated is not None and stated != want:
        out.append(
            f"{se.key} removes water {_HOW[want]}, but the params were built for an implementation removing "
            f"it {_HOW[stated]} (SoilEvapParams.layer_removal); {fix}"
        )
    traced = []
    for name in _COUPLED_FLAGS:
        try:
            flags = np.asarray(get_path(params, name).salus_es)
        except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
            traced.append(name)
            continue
        if not bool(np.all(flags.astype(bool) == want)):
            out.append(
                f"params[{name!r}].salus_es does not say {_HOW[want]} on every sample, the removal of "
                f"{se.key}: the module would book the evaporation the other way; {fix}"
            )
    if traced and stated is None:
        out.append(
            f"the coupling flags of {traced} are traced and SoilEvapParams.layer_removal is None, so the "
            f"params cannot be checked against {se.key}; {fix} or check them on the host first"
        )
    return out


def _check_params(params: Mapping[str, Any], impl: SoilEvaporation) -> None:
    """Trace-time check of the day's params against the implementation
    (:func:`soil_evaporation_params_problems`)."""
    problems = soil_evaporation_params_problems(params, impl)
    if problems:
        raise SoilEvaporationError("spam.soil_evaporation: " + "\n  ".join(problems))


def soil_evaporation_entry(impl: str | Process | SoilEvaporation = "R") -> Process:
    """``spam.soil_evaporation``: the soil evaporation ``impl`` (:func:`resolve_soil_evaporation`: a
    ``MESEV`` code, a registry key or a process; ``R`` is ``soil_water/soilev``, ``S``
    ``soil_water/esr_soilevap``) on the SPAM store, with the start-of-day water content (P7 of
    yesterday, lag 1), the day's rates of ``WATBAL`` RATE (``SWDELTS``, ``SWDELTU``, ``WINF``) and the
    day's soil; then PD1 (the actual ``ES``, ``EM``, ``EF = 0``, ``EVAP``, ``ES_LYR``, cm d-1 except
    ``EVAP`` in mm d-1) and, for a layer-by-layer removal, the bucket's per-layer evaporation
    ``evap_layers = -SWDELTU DLAYR`` (``WATBAL.for:490-495``). The choice is static (by key); the
    entry's reads and writes are the same for every implementation, so the day and its checks do not
    change."""
    se = resolve_soil_evaporation(impl)
    proc = se.process
    layered = se.layer_removal

    def _soil_evaporation(state: Any, params: Any, forcing_t: Any) -> Any:
        """The day's actual soil evaporation of SPAM into the store, PD1 and the bucket's layer input.

        Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for:349-373, Soil/SoilWater/WATBAL.for:490-495 (BSD-3).
        """
        _check_params(params, se)
        se_state: SoilEvapState = get_path(state, _EVAP)
        soil = _day_soil(params, forcing_t)
        view = se_state.replace(
            sw=get_path(state, "soil_water.theta"),
            swdelts=get_path(state, "soil_water.flux.swdelts"),
            swdeltu=get_path(state, "soil_water.flux.swdeltu"),
            winf=get_path(state, "soil_water.flux.winf"),
            dlayr=soil.dlayr,
            ds=soil.ds,
            dul=soil.dul,
            ll=soil.ll,
            pet=None,
        )
        new = proc(view, get_path(params, "evap"), None)
        state = set_path(state, _EVAP, detach(new))
        ev = get_path(state, _EV)
        dt = ev.soil_evaporation.dtype
        values = (
            (new.es * CM_PER_MM).astype(dt),
            (view.em * CM_PER_MM).astype(dt),  # EM of the mulch entry, not of the implementation
            jnp.zeros_like(ev.flood_evaporation),
            new.evap.astype(dt),
            (new.es_lyr * CM_PER_MM).astype(dt) if layered else jnp.zeros_like(ev.soil_evaporation_layers),
        )
        for name, value in zip(_EV_FIELDS, values, strict=True):  # static loop over the record's fields
            state = set_path(state, f"{_EV}.{name}", value)
        layers = -new.swdeltu * soil.dlayr if layered else jnp.zeros_like(new.swdeltu)
        return set_path(state, "soil_water.evap_layers", layers.astype(dt))

    return process(
        _soil_evaporation,
        reads=(
            _EVAP,
            "soil_water.theta",
            "soil_water.flux.swdelts",
            "soil_water.flux.swdeltu",
            "soil_water.flux.winf",
        ),
        writes=(_EVAP, *(f"{_EV}.{n}" for n in _EV_FIELDS), "soil_water.evap_layers"),
        name="spam.soil_evaporation",
        register=False,
        source=f"DSSAT-CSM v4.8.6.0 SPAM/SPAM.for soil evaporation ({se.key})",
    )


def transpiration_entry(slot: str = SLOT) -> Process:
    """``spam.transpiration``: ``pet/spam_trans`` (P5 ``transpiration`` = EOP / 10) from P5 EO, the
    day's actual evaporation (PD1 ``evaporation``) and yesterday's canopy, then P1 ``eop`` = 10 P5
    ``transpiration`` [mm d-1] for every crop of the slot (``SPAM`` hands ``EOP`` to ``PLANT``; the
    cm-to-mm round trip may move ``EOP`` by one unit in the last place)."""
    p = _p(slot)
    bt = bind(
        spam_potential_transpiration,
        own="surface.trans",
        ports={"canopy": p["canopy"], "pet": _PET, "evaporation": f"{_EV}.evaporation"},
        params="spam",
        forcing="spam.weather",
        name="spam.transpiration",
    )
    target = f"{p['crop_water']}.eop"

    def _transpiration(state: Any, params: Any, forcing_t: Any) -> Any:
        """TRANS, then P1 eop from P5 transpiration (mm d-1).

        Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for:378-387; CSM_Main/LAND.for:340 (PLANT reads EOP), BSD-3.
        """
        state = bt(state, params, forcing_t)
        old = get_path(state, target)
        t = get_path(state, f"{_PET}.transpiration") * MM_PER_CM
        return set_path(state, target, jnp.broadcast_to(t[..., None], old.shape).astype(old.dtype))

    return process(
        _transpiration,
        reads=bt.reads,
        writes=(*bt.writes, target),
        name="spam.transpiration",
        register=False,
        source=bt.source,
    )


# ------------------------------------------------------------------------ crop adapters
def layers_in_entry(slot: str = SLOT) -> Process:
    """``crops.<slot>.layers_in``: :func:`layers_in` on the global state (P7 -> P1 ``sw``, the soil's
    snow pack -> P9 ``swe``)."""
    target = f"{_p(slot)['crop_water']}.sw"
    snow_target = f"{_SNOW}.swe"

    def _layers_in(state: Any, params: Any, forcing_t: Any) -> Any:
        """The crop water record's layer water content from the soil's (identity map) and the
        crop's snow from the soil's snow pack.

        Source: DSSAT-CSM v4.8.6.0 CSM_Main/LAND.for:386 (PLANT receives SW and SNOW), BSD-3.
        """
        view = LayersInState(
            theta=get_path(state, "soil_water.theta"),
            snow=get_path(state, "soil_water.snow"),
            water_out=get_path(state, _p(slot)["crop_water"]),
            snow_out=get_path(state, _SNOW),
        )
        new = layers_in(view, params, forcing_t)
        state = set_path(state, target, new.water_out.sw)
        return set_path(state, snow_target, new.snow_out.swe)

    return process(
        _layers_in,
        reads=("soil_water.theta", "soil_water.snow"),
        writes=(target, snow_target),
        name=f"crops.{slot}.layers_in",
        register=False,
        source="DSSAT-CSM v4.8.6.0 CSM_Main/LAND.for (PLANT receives SOIL's SW and SNOW)",
    )


def canopy_entry(slot: str = SLOT) -> Process:
    """``crops.<slot>.canopy``: :func:`~agrijax.models.day_dssat486_adapters.canopy_from_ceres` on the
    global state (the crop's end-of-day LAI and CANHT -> P6, read by SPAM the next day)."""
    p = _p(slot)

    def _canopy(state: Any, params: Any, forcing_t: Any) -> Any:
        """P6 from the crop's end-of-day LAI and height.

        Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for:1818-1819, BSD-3.
        """
        g = get_path(state, f"{p['crop']}.growth")
        view = CanopyInState(lai=g.lai, canht=g.canht, canopy_out=get_path(state, p["canopy"]))
        return set_path(state, p["canopy"], canopy_from_ceres(view, params, forcing_t).canopy_out)

    return process(
        _canopy,
        reads=(f"{p['crop']}.growth.lai", f"{p['crop']}.growth.canht"),
        writes=(p["canopy"],),
        name=f"crops.{slot}.canopy",
        register=False,
        source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for (XLAI = XHLAI = LAI)",
    )


def trwup_replay_entry(slot: str = SLOT) -> Process:
    """A replay of P1 ``trwup`` in place of ROOTWU (``forcing.replay.trwup``, [T, n_crop] cm d-1):
    with the soil water and ``EOP`` also replayed, the day is the crop-only configuration (the crop
    isolated, all its water inputs from a reference run)."""
    p = _p(slot)
    return replay_entry(
        f"water_supply.{slot}.rootwu",
        {f"{p['crop_water']}.trwup": "replay.trwup"},
        stands_in_reads=(p["root"], f"{p['crop_water']}.sw"),
    )


# ------------------------------------------------------------------------ ledger
def soil_storage(state: Any, params: Any, forcing_t: Any) -> Array:
    """Water held by the soil [cm]: ``sum(SW DLAYR)`` on the day's end thickness + (snow + mulch
    water) / 10.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WBAL.for (daily balance: TSW, SNOW, MULCHWAT).
    """
    f = forcing_t["soil"]
    dl = _day_soil(params, forcing_t).dlayr if f.dlayr_end is None else f.dlayr_end
    b = get_path(state, "soil_water")
    return jnp.sum(b.sw * dl, axis=-1) + (b.snow + b.mulch_wat) * CM_PER_MM


def _soil_mm(name: str) -> Any:
    def channel(state: Any, params: Any, forcing_t: Any) -> Any:
        return getattr(forcing_t["soil"], name) * CM_PER_MM

    return channel


def _flux_mm(name: str) -> Any:
    def channel(state: Any, params: Any, forcing_t: Any) -> Any:
        return get_path(state, f"soil_water.flux.{name}") * CM_PER_MM

    return channel


def _es_out(state: Any, params: Any, forcing_t: Any) -> Any:
    """ES [cm d-1] as the bucket removes it: PD1 ``soil_evaporation``, or the layer sum of the SALUS
    evaporation."""
    salus = jnp.asarray(get_path(params, "soil").salus_es, dtype=bool)
    return jnp.where(
        salus,
        jnp.sum(get_path(state, "soil_water.evap_layers"), axis=-1),
        get_path(state, f"{_EV}.soil_evaporation"),
    )


def _residue_evaporation(state: Any, params: Any, forcing_t: Any) -> Any:
    """EM [cm d-1] as the mulch water loses it (the mulch switch on)."""
    on = jnp.asarray(get_path(params, "soil").mulch_on, dtype=bool)
    return jnp.where(on, get_path(state, f"{_EV}.residue_evaporation"), 0.0)


def _uptake(state: Any, params: Any, forcing_t: Any) -> Any:
    """The day's root water uptake from the profile [cm d-1] (sum of P4 ``uptake``)."""
    return jnp.sum(get_path(state, "soil_water.sink_in.uptake"), axis=-1)


def ledger_entry(*, atol: float | None = None, rtol: float | None = None) -> Process:
    """``ledger.close``: soil profile + snow + mulch water (:func:`soil_storage`); in: rain,
    irrigation, residue water; out: runoff, drainage, soil and mulch evaporation (PD1), uptake (P4)
    and the reference's truncations (the bucket's own booking of its cuts and rounding). DSSAT's
    ``WBAL`` has the same terms (flood and lateral flow are zero for these treatments)."""
    inflows = {
        "rain": _soil_mm("rain"),
        "irrigation": _soil_mm("irrigation"),
        "residue_water": _flux_mm("residue_water"),
    }
    outflows = {
        "runoff": _flux_mm("runoff"),
        "drainage": _flux_mm("drain"),
        "soil_evaporation": _es_out,
        "residue_evaporation": _residue_evaporation,
        "transpiration": _uptake,
        "truncation": "soil_water.flux.truncation",
    }
    return water_ledger(
        storage=soil_storage,
        inflows=inflows,
        outflows=outflows,
        reads=(
            "soil_water.sw",
            "soil_water.snow",
            "soil_water.mulch_wat",
            "soil_water.flux",
            "soil_water.evap_layers",
            "soil_water.sink_in.uptake",
            _EV,
        ),
        atol=atol,
        rtol=rtol,
    )


def ledger_initial(storage0: Any) -> WaterLedger:
    """A zero ledger with the day's channels starting at ``storage0`` [cm]."""
    return WaterLedger.init(storage0, inflows=LEDGER_INFLOWS, outflows=LEDGER_OUTFLOWS)


# ------------------------------------------------------------------------ assembly
def day_processes(
    slot: str = SLOT,
    *,
    mesev: str | None = None,
    soil_evaporation: str | Process | SoilEvaporation | None = None,
    replace: Mapping[str, Process] | None = None,
) -> dict[str, Process]:
    """``{entry: process}`` of the DSSAT day for crop ``slot`` (feed to ``day_dssat486(slot).compile``):
    every row's own process. ``soil_evaporation`` is the implementation of ``spam.soil_evaporation``
    (:func:`resolve_soil_evaporation`: a registry key such as ``SOIL_EVAPORATION_KEYS["S"]``, a
    process, or a ``MESEV`` code), checked against the entry's interface; ``mesev`` (``R``: SOILEV,
    ``S``: ESR) is the same choice by the reference's name; give one or neither (default ``R``). Build
    the params with the same choice (``day_params(..., soil_evaporation=...)``).

    ``replace`` swaps entries (a process for a replay: :func:`replay_processes` gives the replay
    configuration, ``{"water_supply.maize.rootwu": trwup_replay_entry()}`` a replayed ``TRWUP``); the
    day and its lag check stay the same.
    """
    if mesev is not None and soil_evaporation is not None:
        raise TypeError("give mesev or soil_evaporation, not both")
    impl = soil_evaporation if soil_evaporation is not None else (mesev or "R")
    p = _p(slot)
    ports = {"water_in": p["crop_water"], "root_out": p["root"], "snow_in": _SNOW}
    procs: dict[str, Process] = {
        "weather.daily": noop_entry(
            "weather.daily", why="WEATHR: the day's weather record is forcing (io/dssat readers)"
        ),
        "events.apply": noop_entry(
            "events.apply",
            why="MGMTOPS RATE: the day's irrigation depth is forcing of the soil rate, the planting "
            "date a crop parameter",
        ),
        "soil_water.albedo": bind(
            soil_albedo_rate,
            own="surface.albedo",
            ports={"sw": "soil_water.theta", "albedo": _ALB},
            params="albedo",
            forcing="soil",
            name="soil_water.albedo",
        ),
        "soil_water.rate": bind(
            bucket_rate, own="soil_water", params="soil", forcing="soil", name="soil_water.rate"
        ),
        "pet.priestley_taylor": bind(
            spam_priestley_taylor,
            own="surface.pt",
            ports={"canopy": p["canopy"], "pet": _PET, "albedo": _ALB},
            params="pt",
            forcing="spam.weather",
            name="pet.priestley_taylor",
        ),
        "spam.pse": bind(
            spam_potential_soil_evaporation,
            own="surface.pse",
            ports={"canopy": p["canopy"], "pet": _PET},
            params="spam",
            name="spam.pse",
        ),
        "spam.mulch_evap": mulch_evap_entry(),
        "spam.soil_evaporation": soil_evaporation_entry(impl),
        "spam.transpiration": transpiration_entry(slot),
        f"water_supply.{slot}.rootwu": bind(
            rootwu_supply,
            own=p["rootwu"],
            ports={"root": p["root"], "water": p["crop_water"]},
            params="rootwu",
            name=f"water_supply.{slot}.rootwu",
        ),
        "spam.xtract": bind(
            spam_xtract,
            own="surface.xtract",
            ports={
                "sw": "soil_water.theta",
                "flux": "soil_water.flux",
                "evaporation": _EV,
                "water": p["crop_water"],
                "rwu": f"{p['rootwu']}.rwu",
                "canopy": p["canopy"],
                "sink": "soil_water.sink_in",
            },
            params="xtract",
            forcing="soil",
            name="spam.xtract",
        ),
        "soil_water.integrate": bind(
            bucket_integrate,
            own="soil_water",
            ports={"pet": _EV},
            params="soil",
            forcing="soil",
            name="soil_water.integrate",
        ),
        f"crops.{slot}.layers_in": layers_in_entry(slot),
        **{
            f"crops.{slot}.{n}": bind(
                proc, own=p["crop"], ports=ports, params="crop", forcing="crop", name=f"crops.{slot}.{n}"
            )
            for n, proc in zip(CROP_ENTRIES, CROP_PROCESSES, strict=True)
        },
        f"crops.{slot}.canopy": canopy_entry(slot),
        "ledger.close": ledger_entry(),
    }
    for k, v in (replace or {}).items():
        if k not in procs:
            raise KeyError(f"no entry {k!r} in the DSSAT day")
        procs[k] = v
    return procs


def replay_processes(slot: str = SLOT) -> dict[str, Process]:
    """The replay configuration as ``day_processes(replace=...)``: every soil-water, SPAM and canopy
    entry replaced by a replay of a reference run (``forcing.replay``: ``soil.{sw, snow, mulch_wat,
    runoff, drain, residue_water}``, ``spam.{eo, es, em, eop, uptake}``, ``canopy``), each declaring
    the read set of the process it stands in for, so the lag check is the free day's. ROOTWU and
    CERES stay coupled (add :func:`trwup_replay_entry` to replay ``TRWUP`` as well)."""
    p = _p(slot)
    canopy = p["canopy"]
    flux = {f"soil_water.flux.{k}": f"replay.soil.{k}" for k in ("runoff", "drain", "residue_water")}
    ev = {f"{_EV}.soil_evaporation": "replay.spam.es", f"{_EV}.residue_evaporation": "replay.spam.em"}
    return {
        "soil_water.albedo": replay_entry("soil_water.albedo", {}, stands_in_reads=("soil_water.theta",)),
        "soil_water.rate": replay_entry(
            "soil_water.rate",
            {**flux, "soil_water.snow": "replay.soil.snow"},
            stands_in_reads=("soil_water.sw", "soil_water.snow", "soil_water.mulch_wat"),
        ),
        "pet.priestley_taylor": replay_entry(
            "pet.priestley_taylor",
            {f"{_PET}.eo_priestley_taylor": "replay.spam.eo"},
            stands_in_reads=(f"{canopy}.lai", f"{_ALB}.msalb"),
        ),
        "spam.pse": replay_entry(
            "spam.pse", {}, stands_in_reads=(f"{_PET}.eo_priestley_taylor", f"{canopy}.lai")
        ),
        "spam.mulch_evap": replay_entry(
            "spam.mulch_evap", {}, stands_in_reads=("soil_water.mulch_wat", f"{_PET}.soil_evaporation")
        ),
        "spam.soil_evaporation": replay_entry(
            "spam.soil_evaporation", ev, stands_in_reads=("soil_water.theta", "soil_water.flux.swdelts")
        ),
        "spam.transpiration": replay_entry(
            "spam.transpiration",
            {f"{p['crop_water']}.eop": "replay.spam.eop"},
            stands_in_reads=(f"{_PET}.eo_priestley_taylor", f"{canopy}.lai", f"{_EV}.evaporation"),
        ),
        "spam.xtract": replay_entry(
            "spam.xtract",
            {"soil_water.sink_in.uptake": "replay.spam.uptake"},
            stands_in_reads=(
                "soil_water.theta",
                "soil_water.flux.swdelts",
                f"{_EV}.soil_evaporation",
                f"{p['crop_water']}.eop",
                f"{p['crop_water']}.trwup",
                f"{p['rootwu']}.rwu",
                f"{canopy}.lai",
            ),
        ),
        "soil_water.integrate": replay_entry(
            "soil_water.integrate",
            {
                "soil_water.sw": "replay.soil.sw",
                "soil_water.theta": "replay.soil.sw",
                "soil_water.mulch_wat": "replay.soil.mulch_wat",
            },
            stands_in_reads=("soil_water.sw", "soil_water.flux", "soil_water.sink_in.uptake", _EV),
        ),
        f"crops.{slot}.canopy": replay_entry(
            f"crops.{slot}.canopy", {canopy: "replay.canopy"}, stands_in_reads=(f"{p['crop']}.growth",)
        ),
    }


def initial_state(
    *,
    bucket: BucketState,
    soil_evap: SoilEvapState,
    crop: CeresMaizeState,
    rootwu: RootwuState,
    salb: Any,
    storage0: Any,
    records: Mapping[str, Any] | None = None,
    slot: str = SLOT,
) -> dict[str, Any]:
    """The global morning state: the bucket (``BucketState.initial``), the SPAM store
    (``SoilEvapState.initial``: the ``SOILEV`` store from the initial top layer), the crop and ROOTWU
    states (their own subtrees; ``crop``'s bound ports are replaced), the interface records
    (``records``: ``{"crop_water", "root", "canopy", "pet", "evaporation", "soil_albedo", "snow"}``,
    default the empty ones with P1 ``sw`` = the soil's ``sw``, P9 ``swe`` = the bucket's snow pack and
    the albedo at ``salb``) and the ledger
    starting at ``storage0`` [cm]."""
    p = _p(slot)
    sw = jnp.asarray(bucket.sw)
    dt = sw.dtype
    n_crop = int(crop.growth.lai.shape[-1])
    n_layer = int(sw.shape[-1])
    rec = dict(records or {})
    defaults = {
        "crop_water": CropWaterIn.zeros(n_crop, n_layer, dt).replace(sw=jnp.array(sw)),
        "root": RootRecord.zeros(n_crop, n_layer, dt),
        "pet": PETFluxes.zeros(dt),
        "canopy": CanopyRecord.zeros(n_crop, dt),
        "evaporation": EvaporationRecord.zeros(n_layer, dt),
        "soil_albedo": SoilAlbedo.constant(salb, dt),
        "snow": SnowOut.zeros(dt).replace(swe=jnp.asarray(bucket.snow, dt)),
    }
    unknown = sorted(set(rec) - set(defaults))
    if unknown:
        raise KeyError(f"unknown interface records {unknown} (known: {sorted(defaults)})")
    rec = {**defaults, **rec}
    water, root = rec["crop_water"], rec["root"]
    crop_binding = Binding(
        p["crop"], (("water_in", p["crop_water"]), ("root_out", p["root"]), ("snow_in", _SNOW))
    )
    crop_full = crop.replace(water_in=water, root_out=root, snow_in=rec["snow"])
    return compose(
        {
            "soil_water": detach(bucket),
            _EVAP: detach(soil_evap),
            "surface.xtract": XtractState(trwu=jnp.zeros((), dt)),
            "surface.albedo": SoilAlbedoState(),
            "surface.pt": PTAlbedoState(),
            "surface.pse": SpamPSEState(),
            "surface.trans": SpamTransState(),
            **crop_binding.entries(crop_full),
            p["rootwu"]: rootwu,
            _PET: rec["pet"],
            _EV: rec["evaporation"],
            _ALB: rec["soil_albedo"],
            p["canopy"]: rec["canopy"],
            "ledger.water": ledger_initial(storage0),
        }
    )


def day_outputs(slot: str = SLOT) -> Any:
    """The daily outputs of the assembly: CERES-Maize's ``PlantGro.OUT`` quantities
    (:func:`~agrijax.processes.crop.ceres_maize.plantgro_outputs`), the crop water record (``sw``,
    ``eop``, ``trwup``), the canopy SPAM reads next (P6), the crop's snow (P9 ``swe``), the soil water
    and its day's fluxes, the potential (P5) and actual (PD1) evaporation and transpiration [mm d-1],
    ``MSALB`` and the ledger storage and daily residual."""
    p = _p(slot)

    def outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
        out = dict(plantgro_outputs(get_path(state, p["crop"]), get_path(params, "crop"), forcing_t))
        w = get_path(state, p["crop_water"])
        led = get_path(state, "ledger.water")
        pet = get_path(state, _PET)
        ev = get_path(state, _EV)
        b = get_path(state, "soil_water")
        out.update(
            p1_sw=w.sw,
            p1_eop=w.eop,
            p1_trwup=w.trwup,
            p6_lai=get_path(state, p["canopy"]).lai,
            p9_swe=get_path(state, _SNOW).swe,
            soil_sw=b.sw,
            snow=b.snow,
            mulch_wat=b.mulch_wat,
            runoff=b.flux.runoff,
            drain=b.flux.drain,
            uptake=b.sink_in.uptake,
            eo=pet.eo_priestley_taylor,
            eos=pet.soil_evaporation * MM_PER_CM,
            eop=pet.transpiration * MM_PER_CM,
            es=ev.soil_evaporation * MM_PER_CM,
            em=ev.residue_evaporation * MM_PER_CM,
            evap=ev.evaporation,
            ep=ev.transpiration,
            msalb=get_path(state, _ALB).msalb,
            storage=led.storage,
            residual=led.residual,
        )
        return out

    return outputs
