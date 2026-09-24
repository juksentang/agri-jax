"""State, parameter and forcing pytrees of CERES-Maize (nitrogen, phosphorus and potassium off).

Every crop field carries a leading ``n_crop`` axis (handover section 4); soil-layer fields of the
crop (root length density, days saturated) are ``[n_crop, n_layer]``; the soil itself (layer
thickness and limits in :class:`CeresSoil`, water content in the forcing) is shared by the crops,
``[n_layer]``. The single lumped leaf pool of CERES-Maize lives in slot 0 of an
:class:`~agrijax.core.organs.OrganQueue` with ``n_cohort = 1``: ``leaf.area[:, 0]`` is the plant
leaf area ``PLA`` [cm2 plant-1] and ``leaf.mass[:, 0]`` the leaf weight ``LFWT`` [g plant-1].

Names follow DSSAT-CSM v4.8 ``Plant/CERES-Maize`` (``MZ_PHENOL``, ``MZ_GROSUB``, ``MZ_ROOTGR``);
each field records its Fortran name. Stage numbers are the DSSAT ``ISTAGE`` codes: 7 sowing,
8 germination, 9 emergence, 1 end of juvenile phase, 2 floral (tassel) initiation, 3 end of leaf
growth / 75 % silking, 4 beginning of effective grain filling, 5 end of effective grain filling,
6 physiological maturity (the stage *entered* is ``ISTAGE + 1`` except 9 -> 1 and 6 -> 10,
the post-maturity code). ``stgdoy[:, k]`` holds the ``YYYYDDD`` date on which stage ``k + 1``
ended (DSSAT ``STGDOY(k + 1)``), ``9999999`` before.

DSSAT-CSM is distributed under the BSD 3-clause licence (Copyright 1998-2026 DSSAT Foundation,
University of Florida, International Fertilizer Development Center); this is an independent
re-implementation of its published equations.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.organs import OrganQueue
from agrijax.core.state import Forcing, Params, State, field

__all__ = [
    "NOT_REACHED",
    "N_STAGE_DATES",
    "CeresCultivar",
    "CeresForcing",
    "CeresGrowthState",
    "CeresMaizeParams",
    "CeresMaizeState",
    "CeresPhenologyState",
    "CeresRootState",
    "CeresSoil",
    "CeresSpecies",
    "CeresStressState",
]

NOT_REACHED = 9999999
"""Value of a stage date that has not happened yet (DSSAT initialises ``STGDOY`` to 9999999)."""
N_STAGE_DATES = 10
"""Stage dates kept: DSSAT ``STGDOY(1..10)``; ``stgdoy[:, k]`` is ``STGDOY(k + 1)``."""

_C = ("n_crop",)
_CL = ("n_crop", "n_layer")


# ---------------------------------------------------------------------------------------- params
class CeresCultivar(Params):
    """Cultivar (``*.CUL``) and ecotype (``*.ECO``) coefficients."""

    p1: Array = field(
        unit="degC d", description="thermal time emergence -> end of juvenile", fortran_name="P1"
    )
    p2: Array = field(unit="d h-1", description="photoperiod sensitivity", fortran_name="P2")
    p5: Array = field(unit="degC d", description="thermal time silking -> maturity", fortran_name="P5")
    g2: Array = field(unit="kernel plant-1", description="potential kernel number", fortran_name="G2")
    g3: Array = field(unit="mg kernel-1 d-1", description="potential kernel growth rate", fortran_name="G3")
    phint: Array = field(unit="degC d", description="phyllochron interval", fortran_name="PHINT")
    tbase: Array = field(unit="degC", description="base temperature for development", fortran_name="TBASE")
    topt: Array = field(unit="degC", description="optimum temperature, vegetative", fortran_name="TOPT")
    ropt: Array = field(unit="degC", description="optimum temperature, reproductive", fortran_name="ROPT")
    p2o: Array = field(unit="h", description="critical (twilight) daylength", fortran_name="P2O")
    djti: Array = field(
        unit="d", description="minimum days end of juvenile -> tassel initiation", fortran_name="DJTI"
    )
    gdde: Array = field(
        unit="degC d cm-1", description="thermal time per cm of sowing depth", fortran_name="GDDE"
    )
    dsgft: Array = field(
        unit="degC d", description="thermal time silking -> effective grain filling", fortran_name="DSGFT"
    )
    rue: Array = field(unit="g MJ-1", description="radiation use efficiency (PAR)", fortran_name="RUE")
    tsen: Array = field(unit="degC", description="cold-sensitivity temperature", fortran_name="TSEN")
    cday: Array = field(unit="d", description="cold days that end the crop", fortran_name="CDAY")


class CeresSpecies(Params):
    """Species coefficients (``MZCER048.SPE``) and the constants ``MZ_GROSUB`` sets in its code."""

    prftc: Array = field(
        unit="degC",
        description="photosynthesis temperature curve (4 cardinal points)",
        fortran_name="PRFTC",
        dims=("4",),
    )
    rgfil: Array = field(
        unit="degC",
        description="grain-fill temperature curve (4 cardinal points)",
        fortran_name="RGFIL",
        dims=("4",),
    )
    parsr: Array = field(unit="-", description="PAR fraction of solar radiation", fortran_name="PARSR")
    co2x: Array = field(
        unit="ppm", description="CO2 abscissae of the photosynthesis table", fortran_name="CO2X", dims=("10",)
    )
    co2y: Array = field(
        unit="-", description="relative photosynthesis at co2x", fortran_name="CO2Y", dims=("10",)
    )
    fslfw: Array = field(unit="d-1", description="leaf senescence at full water stress", fortran_name="FSLFW")
    rsgr: Array = field(
        unit="-",
        description="relative grain-fill rate below which the crop may mature early",
        fortran_name="RSGR",
    )
    rsgrt: Array = field(
        unit="d", description="days below rsgr that trigger early maturity", fortran_name="RSGRT"
    )
    carbot: Array = field(
        unit="d", description="days with no assimilation that trigger early maturity", fortran_name="CARBOT"
    )
    dsgt: Array = field(unit="d", description="maximum days sowing -> germination", fortran_name="DSGT")
    dget: Array = field(
        unit="degC d", description="maximum thermal time germination -> emergence", fortran_name="DGET"
    )
    swcg: Array = field(
        unit="cm3 cm-3", description="available soil water needed to germinate", fortran_name="SWCG"
    )
    stmwte: Array = field(unit="g plant-1", description="stem weight at emergence", fortran_name="STMWTE")
    rtwte: Array = field(unit="g plant-1", description="root weight at emergence", fortran_name="RTWTE")
    lfwte: Array = field(unit="g plant-1", description="leaf weight at emergence", fortran_name="LFWTE")
    seedrve: Array = field(unit="g plant-1", description="seed reserve at emergence", fortran_name="SEEDRVE")
    leafnoe: Array = field(unit="-", description="leaf number at emergence", fortran_name="LEAFNOE")
    plae: Array = field(unit="cm2 plant-1", description="leaf area at emergence", fortran_name="PLAE")
    pormin: Array = field(
        unit="cm3 cm-3", description="minimum air-filled porosity for roots", fortran_name="PORMIN"
    )
    rlwr: Array = field(unit="cm g-1 x 1e4", description="root length to weight ratio", fortran_name="RLWR")
    rwuep1: Array = field(
        unit="-", description="uptake/demand ratio below which expansion is stressed", fortran_name="RWUEP1"
    )
    canht_pot: Array = field(
        unit="m", description="potential canopy height (1.6 m in MZ_GROSUB)", fortran_name="CANHT_POT"
    )
    bsgdd: Array = field(
        unit="degC d",
        description="start of ear growth before silking (250 in MZ_GROSUB)",
        fortran_name="BSGDD",
    )


class CeresSoil(Params):
    """Soil layers seen by the crop (shared by all crops), ``[n_layer]``."""

    dlayr: Array = field(unit="cm", description="layer thickness", fortran_name="DLAYR", dims=("n_layer",))
    ll: Array = field(unit="cm3 cm-3", description="lower limit", fortran_name="LL", dims=("n_layer",))
    dul: Array = field(
        unit="cm3 cm-3", description="drained upper limit", fortran_name="DUL", dims=("n_layer",)
    )
    sat: Array = field(unit="cm3 cm-3", description="saturation", fortran_name="SAT", dims=("n_layer",))
    shf: Array = field(unit="-", description="root growth factor", fortran_name="SHF", dims=("n_layer",))
    slpf: Array = field(unit="-", description="soil photosynthesis factor", fortran_name="SLPF")


class CeresMaizeParams(Params):
    """Everything constant over a CERES-Maize run: cultivar, species, management and soil."""

    cultivar: CeresCultivar
    species: CeresSpecies
    soil: CeresSoil
    pltpop: Array = field(unit="plant m-2", description="plant population at sowing", fortran_name="PLTPOP")
    sdepth: Array = field(unit="cm", description="sowing depth", fortran_name="SDEPTH")
    rowspc: Array = field(unit="cm", description="row spacing", fortran_name="ROWSPC")
    yrplt: Array = field(unit="YYYYDDD", description="sowing date", fortran_name="YRPLT")
    iswwat: bool = field(
        description="water balance on (DSSAT ISWWAT = Y)", fortran_name="ISWWAT", static=True, default=True
    )


# --------------------------------------------------------------------------------------- forcing
class CeresForcing(Forcing):
    """Daily inputs of the crop (time axis first). Soil water and the water-stress factors come
    from the soil-water / transpiration modules (in isolation: from the reference run)."""

    yrdoy: Array = field(unit="YYYYDDD", description="date", fortran_name="YRDOY", dims="T")
    tmax: Array = field(unit="degC", description="maximum air temperature", fortran_name="TMAX", dims="T")
    tmin: Array = field(unit="degC", description="minimum air temperature", fortran_name="TMIN", dims="T")
    srad: Array = field(unit="MJ m-2 d-1", description="solar radiation", fortran_name="SRAD", dims="T")
    dayl: Array = field(unit="h", description="daylength (DAYLEN)", fortran_name="DAYL", dims="T")
    twilen: Array = field(
        unit="h", description="twilight-to-twilight daylength (TWILIGHT)", fortran_name="TWILEN", dims="T"
    )
    co2: Array = field(unit="ppm", description="atmospheric CO2", fortran_name="CO2", dims="T")
    snow: Array = field(unit="mm", description="snow depth", fortran_name="SNOW", dims="T")
    sw: Array = field(
        unit="cm3 cm-3",
        description="soil water content after today's soil update",
        fortran_name="SW",
        dims=("T", "n_layer"),
    )
    swfac: Array = field(
        unit="-",
        description="water stress on photosynthesis (TRWUP / EP1, capped at 1)",
        fortran_name="SWFAC",
        dims="T",
    )
    turfac: Array = field(
        unit="-", description="water stress on expansion (truncated to 1e-3)", fortran_name="TURFAC", dims="T"
    )


# ----------------------------------------------------------------------------------------- state
class CeresPhenologyState(State):
    """Development state (``MZ_PHENOL``)."""

    istage: Array = field(
        unit="-", description="DSSAT stage code (7 before sowing)", fortran_name="ISTAGE", dims=_C
    )
    sumdtt: Array = field(
        unit="degC d", description="thermal time in the current stage", fortran_name="SUMDTT", dims=_C
    )
    cumdtt: Array = field(
        unit="degC d", description="thermal time since germination", fortran_name="CUMDTT", dims=_C
    )
    dtt: Array = field(unit="degC d", description="thermal time today", fortran_name="DTT", dims=_C)
    ndas: Array = field(
        unit="d", description="days after sowing (stage-block counter)", fortran_name="NDAS", dims=_C
    )
    xstage: Array = field(unit="-", description="non-integer growth stage", fortran_name="XSTAGE", dims=_C)
    sind: Array = field(unit="-", description="summed photoperiod induction", fortran_name="SIND", dims=_C)
    p3: Array = field(
        unit="degC d", description="thermal time tassel initiation -> silking", fortran_name="P3", dims=_C
    )
    p9: Array = field(
        unit="degC d", description="thermal time germination -> emergence", fortran_name="P9", dims=_C
    )
    tlno: Array = field(unit="-", description="total leaf number", fortran_name="TLNO", dims=_C)
    xnti: Array = field(
        unit="-", description="leaf number at tassel initiation", fortran_name="XNTI", dims=_C
    )
    gpp: Array = field(unit="kernel plant-1", description="grains per plant", fortran_name="GPP", dims=_C)
    ears: Array = field(unit="ear m-2", description="ears per square metre", fortran_name="EARS", dims=_C)
    idurp: Array = field(unit="d", description="days in stage 4", fortran_name="IDURP", dims=_C)
    seed_layer: Array = field(
        unit="-", description="0-based soil layer holding the seed", fortran_name="L0", dims=_C
    )
    stgdoy: Array = field(
        unit="YYYYDDD", description="end date of stage k + 1", fortran_name="STGDOY", dims=("n_crop", "10")
    )
    mdate: Array = field(
        unit="YYYYDDD", description="maturity (or failure) date, -99 before", fortran_name="MDATE", dims=_C
    )
    crop_status: Array = field(
        unit="-", description="0 growing, 1 mature, 12/13/32/33 failures", fortran_name="CropStatus", dims=_C
    )


class CeresStressState(State):
    """Stress factors of the day and the saturation counters (``MZ_GROSUB`` stress block)."""

    swfac: Array = field(
        unit="-", description="water stress on photosynthesis", fortran_name="SWFAC", dims=_C
    )
    turfac: Array = field(unit="-", description="water stress on expansion", fortran_name="TURFAC", dims=_C)
    satfac: Array = field(
        unit="-", description="excess-water (saturation) stress", fortran_name="SATFAC", dims=_C
    )
    tss: Array = field(
        unit="d", description="days each layer has been near saturation", fortran_name="TSS", dims=_CL
    )


class CeresGrowthState(State):
    """Biomass, leaf area and partitioning state (``MZ_GROSUB``), per plant unless noted."""

    leaf: OrganQueue
    pltpop: Array = field(
        unit="plant m-2",
        description="plant population (0 after a crop failure)",
        fortran_name="PLTPOP",
        dims=_C,
    )
    senla: Array = field(unit="cm2 plant-1", description="senesced leaf area", fortran_name="SENLA", dims=_C)
    lai: Array = field(unit="m2 m-2", description="green leaf area index", fortran_name="LAI", dims=_C)
    stmwt: Array = field(unit="g plant-1", description="stem weight", fortran_name="STMWT", dims=_C)
    earwt: Array = field(unit="g plant-1", description="ear weight", fortran_name="EARWT", dims=_C)
    grnwt: Array = field(unit="g plant-1", description="grain weight", fortran_name="GRNWT", dims=_C)
    rtwt: Array = field(unit="g plant-1", description="root weight", fortran_name="RTWT", dims=_C)
    seedrv: Array = field(unit="g plant-1", description="seed reserve", fortran_name="SEEDRV", dims=_C)
    cumph: Array = field(unit="-", description="cumulative phyllochrons", fortran_name="CUMPH", dims=_C)
    xn: Array = field(unit="-", description="number of the oldest expanding leaf", fortran_name="XN", dims=_C)
    leafno: Array = field(unit="-", description="leaf number INT(XN)", fortran_name="LEAFNO", dims=_C)
    slan: Array = field(
        unit="cm2 plant-1", description="normal leaf senescence", fortran_name="SLAN", dims=_C
    )
    stg2cls: Array = field(
        unit="kg ha-1",
        description="leaf senescence up to the end of stage 2",
        fortran_name="Stg2CLS",
        dims=_C,
    )
    cum_leaf_senes: Array = field(
        unit="kg ha-1", description="cumulative leaf senescence", fortran_name="CumLeafSenes", dims=_C
    )
    swmin: Array = field(
        unit="g plant-1", description="minimum stem weight in grain fill", fortran_name="SWMIN", dims=_C
    )
    swmax: Array = field(
        unit="g plant-1", description="maximum stem weight in grain fill", fortran_name="SWMAX", dims=_C
    )
    sump: Array = field(
        unit="g plant-1", description="assimilation summed over stage 4", fortran_name="SUMP", dims=_C
    )
    cumdtteg: Array = field(
        unit="degC d", description="thermal time of ear growth", fortran_name="CUMDTTEG", dims=_C
    )
    carbo: Array = field(
        unit="g plant-1 d-1", description="assimilation today", fortran_name="CARBO", dims=_C
    )
    grort: Array = field(unit="g plant-1 d-1", description="root growth today", fortran_name="GRORT", dims=_C)
    grogrn: Array = field(
        unit="g plant-1 d-1", description="grain growth today", fortran_name="GROGRN", dims=_C
    )
    emat: Array = field(unit="d", description="days of slow grain filling", fortran_name="EMAT", dims=_C)
    cmat: Array = field(
        unit="d", description="days without assimilation in grain fill", fortran_name="CMAT", dims=_C
    )
    icold: Array = field(unit="d", description="consecutive cold days", fortran_name="ICOLD", dims=_C)
    nwsd: Array = field(
        unit="d", description="consecutive severe water-stress days", fortran_name="NWSD", dims=_C
    )
    maxlai: Array = field(unit="m2 m-2", description="maximum LAI so far", fortran_name="MAXLAI", dims=_C)
    canht: Array = field(unit="m", description="canopy height", fortran_name="CANHT", dims=_C)
    biomas: Array = field(unit="g m-2", description="above-ground biomass", fortran_name="BIOMAS", dims=_C)
    rstage: Array = field(
        unit="-", description="reproductive stage for output", fortran_name="RSTAGE", dims=_C
    )


class CeresRootState(State):
    """Root depth and length density (``MZ_ROOTGR``)."""

    rtdep: Array = field(unit="cm", description="rooting depth", fortran_name="RTDEP", dims=_C)
    rlv: Array = field(unit="cm cm-3", description="root length density", fortran_name="RLV", dims=_CL)


class CeresMaizeState(State):
    """The whole CERES-Maize state."""

    phen: CeresPhenologyState
    stress: CeresStressState
    growth: CeresGrowthState
    roots: CeresRootState

    @classmethod
    def initial(cls, params: CeresMaizeParams, n_crop: int = 1, dtype: Any = None) -> CeresMaizeState:
        """The state of DSSAT's seasonal initialisation (``SEASINIT``) for ``n_crop`` crops."""
        n_layer = int(jnp.shape(params.soil.dlayr)[-1])
        f = jnp.zeros((n_crop,), dtype=dtype if dtype is not None else jnp.result_type(float))
        i = jnp.zeros((n_crop,), dtype=jnp.int32)
        fl = jnp.zeros((n_crop, n_layer), dtype=f.dtype)
        phen = CeresPhenologyState(
            istage=i + 7,
            sumdtt=f,
            cumdtt=f,
            dtt=f,
            ndas=f,
            xstage=f + 0.1,
            sind=f,
            p3=f,
            p9=f,
            tlno=f,
            xnti=f,
            gpp=f,
            ears=f,
            idurp=i,
            seed_layer=i,
            stgdoy=jnp.full((n_crop, N_STAGE_DATES), NOT_REACHED, dtype=jnp.int32),
            mdate=i - 99,
            crop_status=i,
        )
        stress = CeresStressState(swfac=f + 1.0, turfac=f + 1.0, satfac=f, tss=fl)
        growth = CeresGrowthState(
            leaf=OrganQueue.empty(n_crop, 1, dtype=f.dtype),
            pltpop=f + jnp.broadcast_to(jnp.asarray(params.pltpop, dtype=f.dtype), (n_crop,)),
            senla=f,
            lai=f,
            stmwt=f,
            earwt=f,
            grnwt=f,
            rtwt=f,
            seedrv=f,
            cumph=f,
            xn=f,
            leafno=i,
            slan=f,
            stg2cls=f,
            cum_leaf_senes=f,
            swmin=f,
            swmax=f,
            sump=f,
            cumdtteg=f,
            carbo=f,
            grort=f,
            grogrn=f,
            emat=i,
            cmat=i,
            icold=i,
            nwsd=i,
            maxlai=f,
            canht=f,
            biomas=f,
            rstage=i,
        )
        roots = CeresRootState(rtdep=f, rlv=fl)
        return cls(phen=phen, stress=stress, growth=growth, roots=roots)
