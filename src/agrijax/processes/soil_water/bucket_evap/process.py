"""The registered soil-side evaporation processes of the DSSAT day (soil_water slot).

SPAM's actual evaporation (``SPAM/SPAM.for`` RATE, lines 319-373), after the potential soil
evaporation ``EOS`` of the pet slot (``pet/spam_pse@dssat-4.8.6.0:faithful``, port P5)::

    soil_water/mulch_evap@dssat-4.8.6.0:faithful    EOS -> EM (mulch), EOS_SOIL
    soil_water/soilev@dssat-4.8.6.0:faithful        MESEV = 'R': EOS_SOIL -> ES (Ritchie two-stage)
    soil_water/esr_soilevap@dssat-4.8.6.0:faithful  MESEV = 'S': EOS_SOIL -> ES, ES_LYR, SWDELTU, UPFLOW

and ``EVAP = ES + EM`` (no flood), which ``pet/spam_trans@dssat-4.8.6.0:faithful`` reads next.

Module state (:class:`SoilEvapState`, own subtree of the soil_water slot). Inputs the bucket
module writes (``WATBAL``; the assembled DSSAT day binds them, here they are replayed from the
reference's SPAM entry): today's start-of-day water ``sw``, drainage change ``swdelts``,
upward-flow change ``swdeltu`` (``UP_FLOW``, ``MESEV = 'R'`` only), the water available for
infiltration ``winf``, the soil layer geometry and limits (``dlayr``, ``ds``, ``dul``, ``ll``: the
reference's ``SOILPROP``, which DSSAT may change during a run) and the mulch record (the residue
module's ``MULCH``). State of its own: the ``SOILEV`` store (``sumes1``, ``sumes2``, ``t``,
``swef``). Outputs: ``eos_soil``, ``em``, ``es``, ``es_lyr``, ``swdeltu`` (ESR), ``upflow`` (ESR),
``evap``. Port: ``pet`` (P5, read: ``soil_evaporation`` = EOS / 10).

Supported processes and switches: ``MESEV`` R and S, ``INFIL`` R / S / M (mulch evaporation on,
static ``mulch_active``) or other (off); plastic mulch (``pmfraction``). Not supported: flood
evaporation (``FLOOD > 0``; ``FLOOD_EVAP``), the zonal ``ETPHOT`` path (``MEEVP = 'Z'``). Units: mm
d-1 for the evaporation (DSSAT), cm3 cm-3 for the water contents, cm for the geometry.

Conservation responsibility: these processes compute the evaporation fluxes and book them in
``es`` (soil, per layer ``es_lyr`` for ESR) and ``em`` (mulch); they change no water store. The
bucket removes ``ES`` from the top layer (``MESEV = 'R'``, ``WATBAL.for:471-474``) or adds
``SWDELTU`` to every layer (``MESEV = 'S'``, ``WATBAL.for:490-495``) and closes the soil water
balance; the mulch module removes ``EM`` from the mulch water. The ledger channel
``soil_evaporation`` (cm) takes ``ES / 10``; the mulch evaporation is not soil water.

Source: DSSAT-CSM v4.8.6.0 ``SPAM/SPAM.for``, ``SPAM/SOILEV.for``, ``SPAM/ESR_SoilEvap.for``,
``Soil/Mulch/MULCHEVAP.for``, BSD-3 (Copyright 1998-2026 DSSAT Foundation, University of
Florida, International Fertilizer Development Center).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Params, State, field
from agrijax.core.units import MM_PER_CM
from agrijax.iface.surface import PETFluxes

from .coefficients import EVAP_COEFFICIENTS, EvapCoefficients
from .mulch import spam_mulch_step
from .ritchie import SoilevStore, soilev_init, soilev_rate
from .suleiman_ritchie import esr_soil_evaporation

__all__ = [
    "SoilEvapParams",
    "SoilEvapState",
    "soil_evaporation_esr",
    "soil_evaporation_mulch",
    "soil_evaporation_soilev",
]

_L = ("n_layer",)
_REF_BUILD = "dscsm048 v4.8.6.0 (gfortran 13, instrumented build for the day-by-day comparison)"
_F32 = (
    "DSSAT single precision (REAL*4) is not reproduced",
    "the kernels run in the precision of their inputs (float64 with x64)",
    "tests/integration/test_spam_evap_dssat.py tolerances (measured REAL*4 rounding)",
)
_NO_FLOOD = (
    "flood evaporation (FLOOD > 0, FLOOD_EVAP) is not implemented: EOS_SOIL starts from EOS",
    "no supported run has floodwater (FLOOD = 0 on every day of the 65 reference runs)",
    "tests/integration/test_spam_evap_dssat.py",
)


class SoilEvapParams(Params):
    """Constants of the soil-side evaporation: the stage-1 limit ``U`` (``SLU1``), the plastic-mulch
    fraction and the mulch switch, and the coefficients."""

    u: Array = field(
        dims=(), unit="mm", description="stage-1 soil evaporation limit (SOILPROP%U, SLU1)", fortran_name="U"
    )
    pmfraction: Array = field(
        dims=(),
        unit="-",
        description="plastic-mulch cover fraction (GET 'PM' 'PMFRACTION')",
        fortran_name="PMFRACTION",
    )
    mulch_active: bool = field(
        description="INFIL (MEINF) is R, S or M: SPAM lets the mulch evaporate", static=True, default=True
    )
    coefficients: EvapCoefficients | None = field(
        description="coefficients of SOILEV, ESR_SoilEvap, MULCH_EVAP (None: the DSSAT-CSM 4.8.6.0 values)",
        default=None,
    )
    layer_removal: bool | None = field(
        description="the assembly built the other modules' params for a soil evaporation that removes "
        "water layer by layer (True: ES_LYR and SWDELTU, as ESR_SoilEvap) or from the top layer through "
        "ES (False: as SOILEV); None: not stated. Read by the assembly's static check only, never by a "
        "kernel",
        static=True,
        default=None,
    )

    @property
    def coeffs(self) -> EvapCoefficients:
        """The coefficients in force (:attr:`coefficients` or :data:`EVAP_COEFFICIENTS`)."""
        return EVAP_COEFFICIENTS if self.coefficients is None else self.coefficients


class SoilEvapState(State):
    """State of the soil-side evaporation (see the module docstring)."""

    # inputs written by the bucket (WATBAL) and the soil properties
    sw: Array = field(
        unit="cm3 cm-3",
        dims=_L,
        grid="dssat_layers",
        fortran_name="SW",
        description="start-of-day layer water content",
    )
    swdelts: Array = field(
        unit="cm3 cm-3",
        dims=_L,
        grid="dssat_layers",
        fortran_name="SWDELTS",
        description="today's drainage change of the layer water content",
    )
    swdeltu: Array = field(
        unit="cm3 cm-3",
        dims=_L,
        grid="dssat_layers",
        fortran_name="SWDELTU",
        description="today's upward-flow / evaporation change (UP_FLOW in; ESR out)",
    )
    winf: Array = field(
        unit="mm", dims=(), fortran_name="WINF", description="water available for infiltration"
    )
    dlayr: Array = field(
        unit="cm", dims=_L, grid="dssat_layers", fortran_name="DLAYR", description="layer thickness"
    )
    ds: Array = field(
        unit="cm", dims=_L, grid="dssat_layers", fortran_name="DS", description="layer bottom depth"
    )
    dul: Array = field(
        unit="cm3 cm-3", dims=_L, grid="dssat_layers", fortran_name="DUL", description="drained upper limit"
    )
    ll: Array = field(
        unit="cm3 cm-3", dims=_L, grid="dssat_layers", fortran_name="LL", description="lower limit"
    )
    # the mulch record (residue module)
    mulch_mass: Array = field(
        unit="kg ha-1", dims=(), fortran_name="MULCHMASS", description="surface mulch mass"
    )
    mulch_cover: Array = field(
        unit="-", dims=(), fortran_name="MULCHCOVER", description="mulch cover fraction"
    )
    mulch_water: Array = field(
        unit="mm", dims=(), fortran_name="MULCHWAT", description="water held by the mulch"
    )
    mulch_am: Array = field(
        unit="ha kg-1",
        dims=(),
        fortran_name="MULCH_AM",
        description="mulch area per unit mass (the reference scales it by 1e-5)",
    )
    mulch_extfac: Array = field(unit="-", dims=(), fortran_name="MUL_EXTFAC", description="mulch extinction")
    # SOILEV store (SAVE)
    sumes1: Array = field(
        unit="mm", dims=(), fortran_name="SUMES1", description="cumulative stage-1 evaporation"
    )
    sumes2: Array = field(
        unit="mm", dims=(), fortran_name="SUMES2", description="cumulative stage-2 evaporation"
    )
    t: Array = field(unit="d", dims=(), fortran_name="T", description="days into stage-2 evaporation")
    swef: Array = field(unit="-", dims=(), fortran_name="SWEF", description="air-dry fraction of LL(1)")
    # outputs
    eos_soil: Array = field(
        unit="mm d-1",
        dims=(),
        fortran_name="EOS_SOIL",
        description="potential soil evaporation after the mulch",
    )
    em: Array = field(unit="mm d-1", dims=(), fortran_name="EM", description="mulch evaporation")
    es: Array = field(unit="mm d-1", dims=(), fortran_name="ES", description="actual soil evaporation")
    es_lyr: Array = field(
        unit="mm d-1",
        dims=_L,
        grid="dssat_layers",
        fortran_name="ES_LYR",
        description="soil evaporation of each layer (ESR)",
    )
    upflow: Array = field(
        unit="cm d-1",
        dims=_L,
        grid="dssat_layers",
        fortran_name="UPFLOW",
        description="evaporative flux through the top of each layer (ESR)",
    )
    evap: Array = field(unit="mm d-1", dims=(), fortran_name="EVAP", description="soil + mulch evaporation")
    pet: PETFluxes = port(description="potential fluxes (P5): soil_evaporation = EOS / 10 read")

    @classmethod
    def initial(
        cls, sw: Any, dlayr: Any, ds: Any, dul: Any, ll: Any, u: Any, *, pet: PETFluxes | None = None
    ) -> SoilEvapState:
        """``SEASINIT``: the ``SOILEV`` store from the initial top layer (:func:`soilev_init`), all
        rates zero, no mulch; the daily inputs start at the initial soil."""
        sw = jnp.asarray(sw)
        z = jnp.zeros((), dtype=sw.dtype)
        zl = jnp.zeros_like(sw)
        st = soilev_init(
            sw[..., 0], jnp.asarray(ll)[..., 0], jnp.asarray(dul)[..., 0], jnp.asarray(dlayr)[..., 0], u
        )
        return cls(
            sw=sw,
            swdelts=zl,
            swdeltu=zl,
            winf=z,
            dlayr=jnp.asarray(dlayr),
            ds=jnp.asarray(ds),
            dul=jnp.asarray(dul),
            ll=jnp.asarray(ll),
            mulch_mass=z,
            mulch_cover=z,
            mulch_water=z,
            mulch_am=z,
            mulch_extfac=z,
            sumes1=jnp.asarray(st.sumes1, sw.dtype),
            sumes2=jnp.asarray(st.sumes2, sw.dtype),
            t=jnp.asarray(st.t, sw.dtype),
            swef=jnp.asarray(st.swef, sw.dtype),
            eos_soil=z,
            em=z,
            es=z,
            es_lyr=zl,
            upflow=zl,
            evap=z,
            pet=PETFluxes.zeros(sw.dtype) if pet is None else pet,
        )


def _eos_mm(state: SoilEvapState) -> Array:
    return jnp.asarray(state.pet.soil_evaporation) * MM_PER_CM


@process(
    reads=("pet.soil_evaporation", "mulch_mass", "mulch_cover", "mulch_water", "mulch_am", "mulch_extfac"),
    writes=("eos_soil", "em"),
    source="DSSAT-CSM v4.8.6.0 Soil/Mulch/MULCHEVAP.for, SPAM/SPAM.for lines 334-346 (BSD-3)",
    fortran_name="MULCH_EVAP",
    key="soil_water/mulch_evap@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="point",
    ref_build=_REF_BUILD,
    sources=(
        (
            "mulch area index, potential and actual mulch evaporation, reduced EOS",
            "MULCHEVAP.for:39-96; Scopel et al. (2004)",
        ),
        (
            "mulch step of SPAM (EOS_SOIL = EOS3 - EM), called for EOS > 1e-6 and MEINF in R, S, M",
            "SPAM.for:334-346",
        ),
    ),
    deviates=(_F32, _NO_FLOOD),
)
def soil_evaporation_mulch(state: SoilEvapState, params: SoilEvapParams, forcing_t: Any) -> SoilEvapState:
    """SPAM's mulch step: ``EM`` and the potential soil evaporation left for the soil,
    ``EOS_SOIL`` [mm d-1], from P5 ``soil_evaporation`` (EOS / 10, cm d-1) and the mulch record.
    Reads no forcing.

    Source: DSSAT-CSM v4.8.6.0 Soil/Mulch/MULCHEVAP.for lines 39-96, SPAM/SPAM.for lines 334-346 (BSD-3).
    """
    c = params.coeffs
    r = spam_mulch_step(
        _eos_mm(state),
        state.mulch_mass,
        state.mulch_cover,
        state.mulch_water,
        state.mulch_am,
        state.mulch_extfac,
        mulch_active=params.mulch_active,
        c=c.mulch,
        gate=c.gate,
    )
    dt = jnp.result_type(state.es)
    return eqx.tree_at(lambda s: (s.eos_soil, s.em), state, (r.eos.astype(dt), r.em.astype(dt)))


@process(
    reads=(
        "eos_soil",
        "em",
        "winf",
        "sw",
        "swdelts",
        "swdeltu",
        "ll",
        "dlayr",
        "sumes1",
        "sumes2",
        "t",
        "swef",
    ),
    writes=("es", "sumes1", "sumes2", "t", "evap"),
    source="DSSAT-CSM v4.8.6.0 SPAM/SOILEV.for SOILEV, ESUP; SPAM/SPAM.for lines 349-373 (BSD-3)",
    fortran_name="SOILEV",
    key="soil_water/soilev@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build=_REF_BUILD,
    sources=(
        (
            "two-stage soil evaporation (stage sums, infiltration reset, AWEV1 / SWMIN limits)",
            "SOILEV.for:96-174; Ritchie (1972)",
        ),
        ("stage-1 step ESUP", "SOILEV.for:218-226"),
        ("SW_AVAIL(1), call gate EOS_SOIL > 1e-6, EVAP = ES + EM + EF", "SPAM.for:349-373"),
    ),
    deviates=(_F32, _NO_FLOOD),
)
def soil_evaporation_soilev(state: SoilEvapState, params: SoilEvapParams, forcing_t: Any) -> SoilEvapState:
    """``MESEV = 'R'``: the day's actual soil evaporation ``ES`` (Ritchie two-stage) and ``EVAP = ES
    + EM``; the ``SOILEV`` store advances only on the days SPAM calls it (``EOS_SOIL > 1e-6``,
    else ``ES = 0``). Reads no forcing.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SOILEV.for lines 96-174, SPAM/SPAM.for lines 349-373 (BSD-3).
    """
    c = params.coeffs
    sw_avail1 = jnp.maximum(0.0, state.sw[..., 0] + state.swdelts[..., 0] + state.swdeltu[..., 0])
    store = SoilevStore(state.sumes1, state.sumes2, state.t, state.swef)
    es, new = soilev_rate(
        store,
        state.eos_soil,
        state.winf,
        state.sw[..., 0],
        state.ll[..., 0],
        state.dlayr[..., 0],
        sw_avail1,
        params.u,
        params.pmfraction,
        c.soilev,
    )
    called = state.eos_soil > c.gate.eos_min_soil
    es = jnp.where(called, es, 0.0)
    s1, s2, t = (jnp.where(called, a, b) for a, b in zip(new[:3], store[:3], strict=True))
    dt = jnp.result_type(state.es)
    return eqx.tree_at(
        lambda s: (s.es, s.sumes1, s.sumes2, s.t, s.evap),
        state,
        (es.astype(dt), s1.astype(dt), s2.astype(dt), t.astype(dt), (es + state.em).astype(dt)),
    )


@process(
    reads=("eos_soil", "em", "sw", "swdelts", "swdeltu", "dlayr", "ds", "dul", "ll"),
    writes=("es", "es_lyr", "swdeltu", "upflow", "evap"),
    source="DSSAT-CSM v4.8.6.0 SPAM/ESR_SoilEvap.for; SPAM/SPAM.for lines 349-373 (BSD-3)",
    fortran_name="ESR_SoilEvap",
    key="soil_water/esr_soilevap@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build=_REF_BUILD,
    sources=(
        (
            "layered evaporation by profile type (wet, intermediate, dry)",
            "ESR_SoilEvap.for:79-164; Ritchie et al. (2009)",
        ),
        ("limit to EOS, UPFLOW", "ESR_SoilEvap.for:166-179; Suleiman and Ritchie (2003)"),
        ("call gate EOS_SOIL > 1e-6, EVAP = ES + EM + EF", "SPAM.for:349-373"),
    ),
    deviates=(_F32, _NO_FLOOD),
)
def soil_evaporation_esr(state: SoilEvapState, params: SoilEvapParams, forcing_t: Any) -> SoilEvapState:
    """``MESEV = 'S'``: the day's layered soil evaporation (``ES``, ``ES_LYR``, ``SWDELTU``,
    ``UPFLOW``) and ``EVAP = ES + EM``; not called for ``EOS_SOIL <= 1e-6`` (``ES = ES_LYR =
    UPFLOW = 0``, ``SWDELTU`` unchanged, as SPAM leaves them). Reads no forcing.

    Source: DSSAT-CSM v4.8.6.0 SPAM/ESR_SoilEvap.for lines 79-179, SPAM/SPAM.for lines 319-373 (BSD-3).
    """
    c = params.coeffs
    r = esr_soil_evaporation(
        state.eos_soil,
        state.sw,
        state.swdelts,
        state.dlayr,
        state.ds,
        state.dul,
        state.ll,
        params.pmfraction,
        c.esr,
    )
    called = state.eos_soil > c.gate.eos_min_soil
    cl = called[..., None]
    es = jnp.where(called, r.es, 0.0)
    dt = jnp.result_type(state.es)
    return eqx.tree_at(
        lambda s: (s.es, s.es_lyr, s.swdeltu, s.upflow, s.evap),
        state,
        (
            es.astype(dt),
            jnp.where(cl, r.es_lyr, 0.0).astype(dt),
            jnp.where(cl, r.swdeltu, state.swdeltu).astype(dt),
            jnp.where(cl, r.upflow, 0.0).astype(dt),
            (es + state.em).astype(dt),
        ),
    )
