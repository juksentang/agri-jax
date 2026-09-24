"""CERES-Maize growth: stress factors, leaf area, assimilation, partitioning and senescence.

Two processes port ``MZ_GROSUB`` (``DYNAMIC = INTEGR``) with nitrogen, phosphorus, potassium and
pests off (``ISWNIT = ISWPHO = ISWPOT = ISWDIS = N``: ``AGEFAC = NSTRES = NDEF3 = PSTRES1 =
PSTRES2 = KSTRES = 1``, no pest damage):

* :func:`ceres_stress` - the water-stress block (``SWFAC``, ``TURFAC``) and the excess-water
  factor ``SATFAC`` with its per-layer saturation-day counters ``TSS``. The water-stress factors
  are read from the forcing (the transpiration module's ``TRWUP`` / ``EP1`` ratio, see
  :func:`water_stress_factors`), which isolates the crop from the soil-water model.
* :func:`ceres_growth` - stage-date initialisations, daily assimilation ``CARBO`` (intercepted
  PAR x RUE x CO2 x temperature / water stress x ``SLPF``), leaf appearance, the per-stage leaf,
  stem, ear, grain and root growth, leaf senescence, cold / drought crop failure and the state
  totals.

``ceres_growth`` is a thin process over kernels with one responsibility each, in the order of
the Fortran: :func:`stage_date_init`, :func:`emergence_init`, :func:`assimilation`,
:func:`leaf_appearance`, one block per stage (:func:`juvenile_growth`,
:func:`floral_induction_growth`, :func:`tassel_silk_growth` with :func:`ear_growth_fraction`,
:func:`stage3_demand`, :func:`stage3_partition`; :func:`silk_efg_growth`;
:func:`grain_fill_growth` with :func:`grain_fill_rate`, :func:`early_maturity`), the stage
selection (:func:`select_stage_block`, :func:`grosub_blocks`), :func:`leaf_senescence`,
:func:`crop_failure`, :func:`canopy_height` and :func:`growth_totals`. Every number
``MZ_GROSUB`` hard-codes is a field of
:class:`~agrijax.processes.crop.ceres_maize.coefficients.GrosubCoefficients`
(``params.coef().grosub``), with its source line.

As in ``MZ_CERES`` the growth routine runs only in stages 1-6; on the maturity (or failure) day
and in stage 6 it stops after the stress block, so growth ends when effective grain filling
ends. Each stage block is computed for every crop and selected with ``jnp.where`` on the stage
mask; every power, square root and division has a clamped argument so both branches of every
``where`` stay finite.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_GROSUB.for`` and ``MZ_CERES.for`` (BSD-3,
Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center); Jones & Kiniry (1986) CERES-Maize; ear growth after J. I. Lizaso (2006).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.organs import OrganQueue, appear
from agrijax.core.process import process

from ._util import curv_lin, safe_div, tabex, trunc_st
from .coefficients import DSSAT_COEFFICIENTS, GrosubCoefficients
from .state import CeresForcing, CeresGrowthState, CeresMaizeParams, CeresMaizeState, CeresSpecies

__all__ = [
    "CropFailure",
    "EarlyMaturity",
    "EmergenceInit",
    "GrainFill",
    "GrosubDay",
    "LeafAppearance",
    "OrganGrowth",
    "StageDateInit",
    "assimilation",
    "canopy_height",
    "ceres_growth",
    "ceres_stress",
    "crop_failure",
    "ear_growth_fraction",
    "early_maturity",
    "emergence_init",
    "floral_induction_growth",
    "grain_fill_growth",
    "grain_fill_rate",
    "grosub_blocks",
    "growth_totals",
    "juvenile_growth",
    "leaf_appearance",
    "leaf_senescence",
    "saturation_factor",
    "select_stage_block",
    "silk_efg_growth",
    "stage3_demand",
    "stage3_partition",
    "stage_date_init",
    "tassel_silk_growth",
    "water_stress_factors",
]

_EPS = 1e-12


def water_stress_factors(eop: ArrayLike, trwup: ArrayLike, rwuep1: ArrayLike) -> tuple[Array, Array]:
    """``(SWFAC, TURFAC)`` from potential transpiration ``EOP`` [mm d-1] and potential root water
    uptake ``TRWUP`` [cm d-1].

    ``EP1 = 0.1 EOP``; ``TURFAC = TRWUP / (RWUEP1 EP1)`` when that ratio is below 1 and
    ``SWFAC = TRWUP / EP1`` when ``EP1 >= TRWUP``, both 1 without demand; ``TURFAC`` is then
    truncated to 1e-3 as in the Fortran (value exact, identity derivative, :func:`trunc_st`).

    Source: DSSAT-CSM MZ_GROSUB.for, "Compute Water Stress Factors".
    """
    eop = jnp.asarray(eop)
    trwup = jnp.asarray(trwup)
    ep1 = eop * 0.1
    demand = eop > 0.0
    ratio = safe_div(trwup, ep1, 1.0)
    turfac = jnp.where(demand & (ratio < rwuep1), ratio / jnp.maximum(jnp.asarray(rwuep1), _EPS), 1.0)
    swfac = jnp.where(demand & (ep1 >= trwup), ratio, 1.0)
    return swfac, trunc_st(turfac * 1000.0) / 1000.0


def saturation_factor(
    sw: ArrayLike,
    sat: ArrayLike,
    dlayr: ArrayLike,
    rlv: ArrayLike,
    tss: ArrayLike,
    pormin: ArrayLike,
    tss_days: ArrayLike = DSSAT_COEFFICIENTS.grosub.tss_days,
) -> tuple[Array, Array]:
    """``(SATFAC, TSS)``: root-length-weighted excess-water stress and the updated saturation days.

    A layer with air-filled porosity ``SAT - SW`` below ``PORMIN`` counts one more saturated day;
    after more than ``tss_days`` (2) days its root activity drops to ``(SAT - SW) / PORMIN``.
    ``sw``, ``sat``, ``dlayr`` are ``[n_layer]``; ``rlv`` and ``tss`` ``[n_crop, n_layer]``.

    Source: DSSAT-CSM MZ_GROSUB.for, "Compute Water Saturation Factors".
    """
    air = jnp.asarray(sat) - jnp.asarray(sw)
    tss_new = jnp.where(air >= pormin, 0.0, jnp.asarray(tss) + 1.0)
    swexf = jnp.where(tss_new > tss_days, jnp.maximum(air / jnp.maximum(jnp.asarray(pormin), _EPS), 0.0), 1.0)
    swexf = jnp.minimum(swexf, 1.0)
    w = jnp.asarray(dlayr) * jnp.asarray(rlv)
    sumex = jnp.sum(w * (1.0 - swexf), axis=-1)
    sumrl = jnp.sum(w, axis=-1)
    satfac = jnp.clip(safe_div(sumex, sumrl), 0.0, 1.0)
    return satfac, tss_new


@process(
    reads=("phen.istage", "phen.mdate", "stress", "roots.rlv"),
    writes=("stress",),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, MZ_CERES.for (BSD-3)",
    fortran_name="MZ_GROSUB",
)
def ceres_stress(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """Water and excess-water stress factors of the day (the ``MZ_GROSUB`` stress block).

    Runs when ``MZ_GROSUB`` runs (stages 1-6) and the day is not the maturity / failure day.
    Outside stages 1-6 ``MZ_CERES`` resets ``SWFAC`` to 1. Without a water balance both
    factors are 1. ``SATFAC`` uses yesterday's root length density (roots grow after growth).

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for (INTEGR stress block) and MZ_CERES.for (BSD-3).
    """
    s = state.phen.istage
    st = state.stress
    yrdoy = jnp.asarray(forcing_t.yrdoy)
    called = (s >= 1) & (s <= 6)
    run = called & (state.phen.mdate != yrdoy)
    wat = params.iswwat
    one = jnp.ones_like(st.swfac)
    sw_in = jnp.where(wat, jnp.asarray(forcing_t.swfac) * one, one)
    tu_in = jnp.where(wat, jnp.asarray(forcing_t.turfac) * one, one)
    swfac = jnp.where(run, sw_in, jnp.where(called, st.swfac, 1.0))
    turfac = jnp.where(run, tu_in, st.turfac)
    satfac_new, tss_new = saturation_factor(
        forcing_t.sw,
        params.soil.sat,
        params.soil.dlayr,
        state.roots.rlv,
        st.tss,
        params.species.pormin,
        params.coef().grosub.tss_days,
    )
    satfac = jnp.where(run, satfac_new, st.satfac)
    tss = jnp.where(run[..., None], tss_new, st.tss)
    new = eqx.tree_at(lambda x: (x.swfac, x.turfac, x.satfac, x.tss), st, (swfac, turfac, satfac, tss))
    return eqx.tree_at(lambda x: x.stress, state, new)


def _pow(x: ArrayLike, p: ArrayLike) -> Array:
    """``x ** p`` on a clamped base (finite value and derivative everywhere)."""
    return jnp.maximum(x, _EPS) ** p


class StageDateInit(NamedTuple):
    """Values ``MZ_GROSUB`` resets on the day a stage ends (before its growth of the day)."""

    ears: Array  # EARS = PLTPOP at tassel initiation (end of stage 2)
    swmin: Array  # SWMIN = 0.85 STMWT at silking (end of stage 3)
    sump: Array  # SUMP = 0 at silking
    swmax: Array  # SWMAX = STMWT at the beginning of effective grain filling (end of stage 4)
    emat: Array  # EMAT = 0 at the beginning of effective grain filling


def stage_date_init(
    yrdoy: ArrayLike,
    called: Array,
    stgdoy: Array,
    pltpop: Array,
    ears: Array,
    g: CeresGrowthState,
    c: GrosubCoefficients,
) -> StageDateInit:
    """The ``IF (YRDOY .EQ. STGDOY(k))`` resets of stages 2, 3 and 4.

    ``called`` is the ``MZ_GROSUB`` call mask (stages 1-6); ``stgdoy[:, k]`` is ``STGDOY(k + 1)``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, stage-date initialisations (EARS, SWMIN,
    SUMP, SWMAX, EMAT).
    """
    at_ti = called & (yrdoy == stgdoy[..., 1])
    at_silk = called & (yrdoy == stgdoy[..., 2])
    at_efg = called & (yrdoy == stgdoy[..., 3])
    return StageDateInit(
        ears=jnp.where(at_ti, pltpop, ears),
        swmin=jnp.where(at_silk, g.stmwt * c.swmin_frac, g.swmin),
        sump=jnp.where(at_silk, 0.0, g.sump),
        swmax=jnp.where(at_efg, g.stmwt, g.swmax),
        emat=jnp.where(at_efg, 0, g.emat),
    )


class EmergenceInit(NamedTuple):
    """The plant on the emergence day (``STGDOY(9)``), otherwise yesterday's values."""

    leaf: OrganQueue
    pla: Array
    lfwt: Array
    stmwt: Array
    rtwt: Array
    seedrv: Array
    leafno: Array
    senla: Array
    lai: Array
    cumph: Array


def emergence_init(
    at_em: Array, pltpop: Array, g: CeresGrowthState, spe: CeresSpecies, c: GrosubCoefficients
) -> EmergenceInit:
    """Seedling weights, leaf area and ``CUMPH = 0.514`` set on the emergence day.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``IF (YRDOY.EQ.STGDOY(9))`` block
    (``STMWTE``, ``RTWTE``, ``LFWTE``, ``SEEDRVE``, ``LEAFNOE``, ``PLAE`` from MZCER048.SPE).
    """
    one = jnp.ones_like(pltpop)
    pla = jnp.where(at_em, spe.plae, g.leaf.area[..., 0])
    return EmergenceInit(
        leaf=appear(g.leaf, at_em, spe.plae * one, spe.lfwte * one),
        pla=pla,
        lfwt=jnp.where(at_em, spe.lfwte, g.leaf.mass[..., 0]),
        stmwt=jnp.where(at_em, spe.stmwte, g.stmwt),
        rtwt=jnp.where(at_em, spe.rtwte, g.rtwt),
        seedrv=jnp.where(at_em, spe.seedrve, g.seedrv),
        leafno=jnp.where(at_em, jnp.floor(spe.leafnoe).astype(g.leafno.dtype), g.leafno),
        senla=jnp.where(at_em, 0.0, g.senla),
        lai=jnp.where(at_em, pltpop * pla * 0.0001, g.lai),
        cumph=jnp.where(at_em, c.cumph_emergence, g.cumph),
    )


def assimilation(
    srad: ArrayLike,
    tmax: ArrayLike,
    tmin: ArrayLike,
    co2: ArrayLike,
    lai: Array,
    pltpop: Array,
    rowspc: ArrayLike,
    swfac: Array,
    slpf: ArrayLike,
    rue: ArrayLike,
    spe: CeresSpecies,
    c: GrosubCoefficients,
) -> Array:
    """Daily assimilation ``CARBO`` [g plant-1 d-1] (nitrogen, phosphorus, potassium off).

    ``LIFAC = 1.5 - 0.768 ((0.01 ROWSPC)^2 PLTPOP)^0.1``; intercepted PAR per plant
    ``IPAR = PARSR SRAD / PLTPOP (1 - exp(-LIFAC LAI))``; ``PCARB = IPAR RUE PCO2`` with the
    CO2 table ``PCO2 = TABEX(CO2Y, CO2X, CO2)``; ``CARBO = max(PCARB min(PRFT, SWFAC) SLPF, 0)``
    with ``PRFT = CURV('LIN', PRFTC, 0.25 TMIN + 0.75 TMAX)`` clipped to ``[0, 1]``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, "Calculate Potential Photosynthesis" to
    ``CARBO = MAX(CARBO, 0.0)``.
    """
    tmax = jnp.asarray(tmax)
    tmin = jnp.asarray(tmin)
    par = jnp.asarray(srad) * spe.parsr
    lifac = c.lifac_max - c.lifac_slope * _pow((rowspc * 0.01) ** 2 * pltpop, c.lifac_exp)
    pco2 = tabex(spe.co2y, spe.co2x, co2)
    ipar = jnp.where(pltpop > 0.0, safe_div(par, pltpop) * (1.0 - jnp.exp(-lifac * lai)), 0.0)
    pcarb = ipar * rue * pco2
    tavgd = c.tavgd_tmin_w * tmin + c.tavgd_tmax_w * tmax
    pr = spe.prftc
    prft = jnp.clip(curv_lin(pr[..., 0], pr[..., 1], pr[..., 2], pr[..., 3], tavgd), 0.0, 1.0)
    return jnp.maximum(pcarb * jnp.minimum(prft, swfac) * slpf, 0.0)


class LeafAppearance(NamedTuple):
    """Leaf tip appearance of the day (stages 1-3)."""

    ti: Array  # TI, phyllochrons today
    cumph: Array  # CUMPH after today's appearance (stages 1-2)
    xn: Array  # XN = CUMPH + 1 (stages 1-2)
    cumph3: Array  # CUMPH in stage 3 (appearance stops 2 PHINT before silking)
    xn3: Array  # XN in stage 3


def leaf_appearance(
    cumph: Array, dtt: Array, phint: ArrayLike, sumdtt: Array, p3: Array, c: GrosubCoefficients
) -> LeafAppearance:
    """``TI = DTT / (PHINT PC)`` with ``PC = 0.66 + 0.068 CUMPH`` below 5 phyllochrons, else 1.

    In stage 3 no new tip appears once ``SUMDTT > P3 - 2 PHINT`` (``CUMPH = CUMPH - TI``).

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, "Calculate leaf emergence" and the start
    of the ``ISTAGE .EQ. 3`` block.
    """
    pc = jnp.where(cumph < c.pc_cumph, c.pc_base + c.pc_slope * cumph, 1.0)
    ti = dtt / jnp.maximum(phint * pc, _EPS)
    cumph1 = cumph + ti
    cumph3 = jnp.where(sumdtt > p3 - c.cumph3_phint * phint, cumph1 - ti, cumph1)
    return LeafAppearance(ti=ti, cumph=cumph1, xn=cumph1 + 1.0, cumph3=cumph3, xn3=cumph3 + 1.0)


class OrganGrowth(NamedTuple):
    """Per-plant organ state after one stage block of ``MZ_GROSUB`` (before senescence).

    A stage block returns every field; one that the block leaves alone keeps its day-start
    value, which is also what a crop outside stages 1-5 keeps."""

    pla: Array  # PLA, plant leaf area [cm2 plant-1]
    lfwt: Array  # LFWT, leaf weight [g plant-1]
    stmwt: Array  # STMWT, stem weight [g plant-1]
    earwt: Array  # EARWT, ear weight [g plant-1]
    grort: Array  # GRORT, root growth today [g plant-1 d-1]
    slan: Array  # SLAN, normal leaf senescence [cm2 plant-1]
    cls: Array  # CumLeafSenes [kg ha-1]


def _leaf_respiration(lfwt: Array, slan: Array, pltpop: Array, c: GrosubCoefficients) -> tuple[Array, Array]:
    """``(LFWT - SLAN / 600, SLAN / 600 PLTPOP 10)``: leaf weight after the respiration of the
    senesced area and the day's ``CumLeafSenes`` term [kg ha-1] (stages 1-4).

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``LFWT = LFWT - SLAN/600.0`` and
    ``CumLeafSenes = SLAN / 600. * PLTPOP * 10.`` of the stage 1-4 blocks.
    """
    return lfwt - slan / c.sla_senes, slan / c.sla_senes * pltpop * 10.0


def juvenile_growth(
    org: OrganGrowth,
    carbo: Array,
    la: LeafAppearance,
    fexp: Array,
    seedrv: Array,
    sumdtt: Array,
    pltpop: Array,
    c: GrosubCoefficients,
) -> tuple[OrganGrowth, Array]:
    """Stage 1 (emergence -> end of juvenile): leaf area from leaf appearance, leaf weight from
    ``(PLA / 250)^1.25``, the rest to roots with at least 25 % of ``CARBO``, drawing on the seed
    reserve; returns the organs and ``SEEDRV``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 1`` block.
    """
    xn1, ti = la.xn, la.ti
    lfwt = org.lfwt
    plag = jnp.where(xn1 < c.plag1_xn, c.plag1_lin * xn1 * ti * fexp, c.plag1_quad * xn1 * xn1 * ti * fexp)
    pla = org.pla + plag
    xlfwt = jnp.maximum(_pow(pla / c.sla_juvenile, c.lfwt_exp), lfwt)
    grolf = xlfwt - lfwt
    grort = carbo - grolf
    low = grort <= c.grort1_min_frac * carbo
    grort = jnp.where(low, carbo * c.grort1_min_frac, grort)
    seedrv_1 = jnp.where(low, seedrv + carbo - grolf - grort, seedrv)
    spent = low & (seedrv_1 <= 0.0)
    seedrv_1 = jnp.where(spent, 0.0, seedrv_1)
    grolf = jnp.where(spent, carbo * c.grolf_max_frac, grolf)
    pla = jnp.where(spent, _pow(lfwt + grolf, c.pla_exp) * c.sla_leaf, pla)
    slan_new = sumdtt * pla / c.slan12_tt  # slan12_tt > 0 (a coefficient, not traced state)
    slan = jnp.where(grolf > 0.0, slan_new, org.slan)
    lfwt_1, cls = _leaf_respiration(lfwt + grolf, slan, pltpop, c)
    return org._replace(pla=pla, lfwt=lfwt_1, grort=grort, slan=slan, cls=cls), seedrv_1


def floral_induction_growth(
    org: OrganGrowth,
    carbo: Array,
    la: LeafAppearance,
    fexp: Array,
    sumdtt: Array,
    pltpop: Array,
    c: GrosubCoefficients,
) -> OrganGrowth:
    """Stage 2 (end of juvenile -> tassel initiation): leaf growth capped at 75 % of ``CARBO``,
    the rest to roots; its ``CumLeafSenes`` is also ``Stg2CLS``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 2`` block.
    """
    xn1, ti = la.xn, la.ti
    lfwt = org.lfwt
    pla = org.pla + c.plag23_quad * xn1 * xn1 * ti * fexp
    grolf = _pow(pla / c.sla_leaf, c.lfwt_exp) - lfwt
    cap = grolf >= carbo * c.grolf_max_frac
    grolf = jnp.where(cap, carbo * c.grolf_max_frac, grolf)
    pla = jnp.where(cap, _pow(lfwt + grolf, c.pla_exp) * c.sla_leaf, pla)
    grort = carbo - grolf
    slan = sumdtt * pla / c.slan12_tt
    lfwt_2, cls = _leaf_respiration(lfwt + grolf, slan, pltpop, c)
    return org._replace(pla=pla, lfwt=lfwt_2, grort=grort, slan=slan, cls=cls)


def ear_growth_fraction(cumdtteg: Array, c: GrosubCoefficients) -> Array:
    """Fraction of ``CARBO`` the ear demands: ``0.81 / (1 + exp(-0.02 (CUMDTTEG - 210)))``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``GROEAR`` of the stage 3 and 4 blocks
    (ear growth after J. I. Lizaso 2006).
    """
    return c.groear_max / (1.0 + jnp.exp(-c.groear_slope * (cumdtteg - c.groear_mid)))


def stage3_demand(
    la: LeafAppearance, fexp: Array, tlno: Array, xnti: Array, pla: Array, c: GrosubCoefficients
) -> tuple[Array, Array]:
    """``(GROLF, GROSTM)`` demanded in stage 3 before the ear and the root floor.

    Leaf expansion ``PLAG`` grows with ``XN^2`` up to leaf 12, stays at ``3.5 x 170`` until
    ``TLNO - 3`` and then declines as ``(XN + 3 - TLNO)^-0.5``; ``GROLF = 0.00116 PLAG PLA^0.25``;
    ``GROSTM = 0.0182 GROLF (XN - XNTI)^2``, or ``3 x 3.1 TI`` for the last leaves.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 3`` block (PLAG, GROLF, GROSTM).
    """
    xn3, ti = la.xn3, la.ti
    early = xn3 < c.xn3_quad
    plateau = xn3 < tlno - c.xn3_decline
    plag3 = jnp.select(
        [early, plateau],
        [c.plag23_quad * xn3 * xn3 * ti * fexp, c.plag23_quad * c.plag3_xn2_max * ti * fexp],
        c.plag3_xn2_max
        * c.plag23_quad
        * _pow(xn3 + c.plag3_decline_offset - tlno, -c.plag3_decline_exp)
        * ti
        * fexp,
    )
    grolf = c.grolf3_coef * plag3 * _pow(pla, c.grolf3_exp)
    grostm = jnp.where(
        early | plateau,
        grolf * c.grostm3_coef * (xn3 - xnti) ** 2,
        c.grostm3_late_a * c.grostm3_late_b * ti * fexp,
    )
    return grolf, grostm


def stage3_partition(
    carbo: Array, grolf: Array, grostm: Array, groear: Array, turfac: Array, c: GrosubCoefficients
) -> tuple[Array, Array, Array, Array]:
    """``(GROLF, GROSTM, GROEAR, GRORT)`` of stage 3: the ear takes its demand out of the stem
    growth (or half of it when the stem growth is smaller), roots get the rest, and when that is
    at most 10 % of ``CARBO`` the shoot organs are scaled to 90 % of it.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 3`` block (GROEAR / GROSTM
    split and the ``GRF`` scaling).
    """
    gt = grostm > groear
    groear = jnp.where(gt, groear, grostm * c.ear_stem_split)
    grostm = jnp.where(gt, grostm - groear, grostm * c.ear_stem_split)
    grort = carbo - grolf - grostm - groear
    scale = (grort <= c.grort3_min_frac * carbo) & (turfac > 0.0)
    any_g = (grolf > 0.0) | (grostm > 0.0)
    grf = jnp.where(any_g, carbo * c.grf3_frac / jnp.maximum(grostm + grolf + groear, _EPS), 1.0)
    grort = jnp.where(scale & any_g, carbo * c.grort3_min_frac, grort)
    grf = jnp.where(scale, grf, 1.0)
    return grolf * grf, grostm * grf, groear * grf, grort


def tassel_silk_growth(
    org: OrganGrowth,
    carbo: Array,
    la: LeafAppearance,
    fexp: Array,
    sumdtt: Array,
    p3: Array,
    tlno: Array,
    xnti: Array,
    bsgdd: ArrayLike,
    turfac: Array,
    pltpop: Array,
    stg2cls: Array,
    c: GrosubCoefficients,
) -> tuple[OrganGrowth, Array]:
    """Stage 3 (tassel initiation -> silking): leaf, stem, ear (from ``BSGDD`` before silking)
    and root growth; returns the organs and the ear thermal time ``CUMDTTEG``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 3`` block.
    """
    eg = sumdtt >= p3 - bsgdd
    cumdtteg = jnp.where(eg, sumdtt - (p3 - bsgdd), 0.0)
    groear = jnp.where(eg, ear_growth_fraction(cumdtteg, c) * carbo, 0.0)
    grolf, grostm = stage3_demand(la, fexp, tlno, xnti, org.pla, c)
    grolf, grostm, groear, grort = stage3_partition(carbo, grolf, grostm, groear, turfac, c)
    pla = _pow(org.lfwt + grolf, c.pla_exp) * c.sla_leaf
    slan = pla / c.slan3_div
    lfwt, cls = _leaf_respiration(org.lfwt + grolf, slan, pltpop, c)
    new = OrganGrowth(
        pla=pla,
        lfwt=lfwt,
        stmwt=org.stmwt + grostm,
        earwt=org.earwt + groear,
        grort=grort,
        slan=slan,
        cls=cls + stg2cls,
    )
    return new, cumdtteg


def silk_efg_growth(
    org: OrganGrowth,
    carbo: Array,
    fexp: Array,
    dtt: Array,
    sumdtt: Array,
    cumdtteg: Array,
    sump: Array,
    pltpop: Array,
    stg2cls: Array,
    c: GrosubCoefficients,
) -> tuple[OrganGrowth, Array, Array]:
    """Stage 4 (silking -> beginning of effective grain filling): ear growth, 8 % of ``CARBO``
    to roots and the rest to the stem (or half each when short), leaf senescence
    ``SLAN = PLA (0.05 + 0.05 SUMDTT / 200)``; returns the organs, ``CUMDTTEG`` and ``SUMP``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 4`` block.
    """
    cumdtteg = cumdtteg + dtt
    groear = ear_growth_fraction(cumdtteg, c) * carbo * fexp
    big = carbo > groear + carbo * c.grort4_frac
    grostm = jnp.where(big, carbo - groear - carbo * c.grort4_frac, (carbo - groear) * c.stem_root_split4)
    grort = jnp.where(big, carbo * c.grort4_frac, (carbo - groear) * c.stem_root_split4)
    slan = org.pla * (c.slan4_base + sumdtt / c.slan4_tt * c.slan4_slope)
    lfwt, cls = _leaf_respiration(org.lfwt, slan, pltpop, c)
    new = org._replace(
        lfwt=lfwt,
        stmwt=org.stmwt + grostm,
        earwt=org.earwt + groear,
        grort=grort,
        slan=slan,
        cls=cls + stg2cls,
    )
    return new, cumdtteg, sump + carbo


def grain_fill_rate(
    tempm: Array, swfac: Array, gpp: Array, g3: ArrayLike, rgfil: Array, c: GrosubCoefficients
) -> tuple[Array, Array]:
    """``(RGFILL, GROGRN)``: relative grain-fill rate ``CURV('LIN', RGFIL, TEMPM)`` clipped to
    ``[0, 1]`` and potential grain growth ``RGFILL GPP G3 0.001 (0.45 + 0.55 SWFAC)`` [g plant-1 d-1].

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 5`` block (RGFILL, GROGRN).
    """
    rg = rgfil
    rgfill = jnp.clip(curv_lin(rg[..., 0], rg[..., 1], rg[..., 2], rg[..., 3], tempm), 0.0, 1.0)
    return rgfill, rgfill * gpp * g3 * 0.001 * (c.grogrn_sw_base + c.grogrn_sw_slope * swfac)


class EarlyMaturity(NamedTuple):
    """Counters of slow grain filling and of days without assimilation in stage 5."""

    emat: Array  # EMAT, consecutive days with RGFILL <= RSGR
    cmat: Array  # CMAT, consecutive days with |CARBO| <= 0.0001
    early: Array  # the crop matures today (SUMDTT = P5)
    early_idle: Array  # ... because of CARBOT days without assimilation


def early_maturity(
    active: Array, rgfill: Array, emat: Array, cmat: Array, spe: CeresSpecies
) -> EarlyMaturity:
    """Early maturity after ``RSGRT`` slow grain-fill days or ``CARBOT`` idle days (``SUMDTT = P5``).

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 5`` block (EMAT, CMAT; RSGR,
    RSGRT, CARBOT from MZCER048.SPE).
    """
    fast = rgfill > spe.rsgr
    emat_a = jnp.where(fast, 0, emat + 1)
    early_a = ~fast & (emat_a.astype(rgfill.dtype) > spe.rsgrt)
    emat_a = jnp.where(early_a, 0, emat_a)
    cmat_b = cmat + 1
    early_b = cmat_b.astype(rgfill.dtype) >= spe.carbot
    return EarlyMaturity(
        emat=jnp.where(active, emat_a, jnp.where(early_b, 0, emat)),
        cmat=jnp.where(active, 0, cmat_b),
        early=jnp.where(active, early_a, early_b),
        early_idle=early_b,
    )


class GrainFill(NamedTuple):
    """Stage-5 results besides the organs."""

    grnwt: Array  # GRNWT, grain weight [g plant-1]
    grogrn: Array  # GROGRN, grain growth today [g plant-1 d-1]
    maturity: EarlyMaturity


def grain_fill_growth(
    org: OrganGrowth,
    carbo: Array,
    tempm: Array,
    swfac: Array,
    sumdtt: Array,
    gpp: Array,
    grnwt: Array,
    grogrn_prev: Array,
    emat: Array,
    cmat: Array,
    swmin: Array,
    swmax: Array,
    cul_g3: ArrayLike,
    cul_p5: ArrayLike,
    spe: CeresSpecies,
    c: GrosubCoefficients,
) -> tuple[OrganGrowth, GrainFill]:
    """Stage 5 (effective grain filling): grain growth, then the stem balance: a surplus goes
    half to the stem and half to roots, a deficit is drawn from the stem down to ``SWMIN`` (with
    0.5 % of the leaf weight moved to a depleted stem); the stem stays below ``SWMAX``.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, ``ISTAGE .EQ. 5`` block.
    """
    active = jnp.abs(carbo) > c.carbo_active
    rgfill, grogrn_a = grain_fill_rate(tempm, swfac, gpp, cul_g3, spe.rgfil, c)
    mat = early_maturity(active, rgfill, emat, cmat, spe)
    pla, lfwt, stmwt = org.pla, org.lfwt, org.stmwt
    slan = jnp.where(active, pla * (c.slan5_base + c.slan5_slope * (safe_div(sumdtt, cul_p5)) ** 3), org.slan)
    grogrn = jnp.where(active, grogrn_a, grogrn_prev)
    grort = jnp.where(active | mat.early_idle, 0.0, org.grort)
    grostm = carbo - grogrn
    pos = grostm >= 0.0
    stmwt_5 = jnp.where(pos, stmwt + grostm * c.stem_root_split5, stmwt + carbo - grogrn)
    grort = jnp.where(pos, grostm * c.stem_root_split5, grort)
    thin = ~pos & (stmwt_5 <= swmin * c.swmin_margin)
    stmwt_5 = jnp.where(thin, stmwt_5 + lfwt * c.leaf_to_stem5, stmwt_5)
    floor_ = thin & (stmwt_5 < swmin)
    stmwt_5 = jnp.where(floor_, swmin, stmwt_5)
    grogrn = jnp.where(floor_, carbo, grogrn)
    new = org._replace(stmwt=jnp.minimum(stmwt_5, swmax), earwt=org.earwt + grogrn, grort=grort, slan=slan)
    return new, GrainFill(grnwt=grnwt + grogrn, grogrn=grogrn, maturity=mat)


def select_stage_block(masks: list[Array], blocks: tuple[OrganGrowth, ...], keep: OrganGrowth) -> OrganGrowth:
    """Field by field, the block of the stage each crop is in (masks disjoint), else ``keep``.

    Source: the ``IF (ISTAGE .EQ. k) ... ELSEIF`` chain of DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR,
    evaluated for every crop and selected (no Python branch on the stage).
    """
    return OrganGrowth(*(jnp.select(masks, list(vals), k) for *vals, k in zip(*blocks, keep, strict=True)))


def leaf_senescence(
    pla: Array,
    senla: Array,
    slan: Array,
    lai: Array,
    swfac: Array,
    tmin: Array,
    pltpop: Array,
    fslfw: ArrayLike,
    c: GrosubCoefficients,
) -> tuple[Array, Array]:
    """``(SENLA, LAI)`` after the day's senescence (water, competition and cold; N, P off).

    ``SLFW = (1 - FSLFW) + FSLFW SWFAC``, ``SLFC = 1 - 0.008 (LAI - 4)`` above LAI 4,
    ``SLFT = max(0, 1 - 0.01 (TMIN - 6)^2)`` at or below 6 degC; ``PLAS = (PLA - SENLA)
    (1 - min(SLFW, SLFC, SLFT))``; ``SENLA`` is at least ``SLAN`` and at most ``PLA``.
    ``lai`` is the day-start LAI.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, "Leaf senescence" (SLFW, SLFC, SLFT, PLAS).
    """
    slfw = (1.0 - fslfw) + fslfw * swfac
    slfc = jnp.where(lai > c.slfc_lai, 1.0 - c.slfc_slope * (lai - c.slfc_lai), 1.0)
    slft = jnp.where(
        tmin <= c.slft_tmin, jnp.maximum(0.0, 1.0 - c.slft_coef * (tmin - c.slft_tmin) ** 2), 1.0
    )
    plas = (pla - senla) * (1.0 - jnp.minimum(jnp.minimum(slfw, slfc), jnp.minimum(slft, 1.0)))
    senla_g = jnp.minimum(jnp.maximum(senla + plas, slan), pla)
    return senla_g, (pla - senla_g) * pltpop * 0.0001


class CropFailure(NamedTuple):
    """Stage, maturity date and status after the cold and drought checks, and their counters."""

    istage: Array
    mdate: Array
    status: Array  # CropStatus: 32 cold, 33 drought
    icold: Array  # ICOLD, consecutive days with TMIN <= TSEN
    nwsd: Array  # NWSD, consecutive days with SWFAC <= 0.1


def crop_failure(
    grow: Array,
    istage: Array,
    mdate: Array,
    status: Array,
    yrdoy: Array,
    leafno: Array,
    lai: Array,
    tmin: Array,
    swfac: Array,
    icold: Array,
    nwsd: Array,
    tsen: ArrayLike,
    cday: ArrayLike,
    c: GrosubCoefficients,
) -> CropFailure:
    """The crop ends (stage 6, ``MDATE = YRDOY``) from cold (a leafless crop with more than 4
    leaves after more than 6 cold days, or ``CDAY`` cold days) or from drought (LAI <= 0.1
    before stage 4 after more than 10 days with ``SWFAC <= 0.1``). ``lai`` is after senescence.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, cold (``ICOLD``, TSEN, CDAY from the ECO
    file) and drought (``NWSD``) crop failure.
    """
    icold_n = jnp.where(grow, jnp.where(tmin <= tsen, icold + 1, 0), icold)
    cold = grow & (
        ((leafno > c.cold_leafno) & (lai <= 0.0) & (istage <= 4) & (icold_n > c.cold_days))
        | (icold_n.astype(lai.dtype) >= cday)
    )
    istage = jnp.where(cold, 6, istage)
    mdate = jnp.where(cold, yrdoy, mdate)
    status = jnp.where(cold, 32, status)
    nwsd_n = jnp.where(grow, jnp.where(swfac > c.drought_swfac, 0, nwsd + 1), nwsd)
    drought = grow & (lai <= c.drought_lai) & (istage < 4) & (nwsd_n > c.drought_days)
    return CropFailure(
        istage=jnp.where(drought, 6, istage),
        mdate=jnp.where(drought, yrdoy, mdate),
        status=jnp.where(drought, 33, status),
        icold=icold_n,
        nwsd=nwsd_n,
    )


def canopy_height(lai: Array, pltpop: Array, canht_pot: ArrayLike, c: GrosubCoefficients) -> Array:
    """``CANHT = min(LAI / (0.4238 PLTPOP + 0.3424) CANHT_POT, CANHT_POT)`` [m].

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, canopy height (updated while LAI >= MAXLAI).
    """
    return jnp.minimum(
        lai / jnp.maximum(c.canht_pop * pltpop + c.canht_base, _EPS) * canht_pot,
        canht_pot * jnp.ones_like(lai),
    )


def _nonneg(grow: Array, x: Array) -> Array:
    """``MAX(0.0, x)`` on the days ``MZ_GROSUB`` grows the crop."""
    return jnp.where(grow, jnp.maximum(x, 0.0), x)


class GrosubDay(NamedTuple):
    """The day's ``MZ_GROSUB`` results before senescence, with each crop's stage block selected."""

    grow: Array  # the crop grows today (stages 1-5, not the maturity / failure day)
    sd: StageDateInit
    em: EmergenceInit
    carbo: Array
    organs: OrganGrowth
    cumph: Array
    xn: Array
    leafno: Array
    cumdtteg: Array
    seedrv: Array
    stg2cls: Array
    sump: Array
    grnwt: Array
    grogrn: Array
    emat: Array
    cmat: Array
    sumdtt: Array  # SUMDTT = P5 on an early-maturity day


def grosub_blocks(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing, c: GrosubCoefficients
) -> GrosubDay:
    """Stage-date resets, assimilation, leaf appearance and the five stage blocks of
    ``MZ_GROSUB``, every block evaluated from the day-start organs and selected by stage.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, from the stage-date initialisations to the
    end of the ``ISTAGE`` ``IF / ELSEIF`` chain.
    """
    ph, g, stq = state.phen, state.growth, state.stress
    cul, spe, f = params.cultivar, params.species, forcing_t
    yrdoy = jnp.asarray(f.yrdoy)
    s = ph.istage
    called = (s >= 1) & (s <= 6)
    pltpop = g.pltpop

    # stage-date initialisations (before any return), then the MZ_CERES / MZ_GROSUB returns
    sd = stage_date_init(yrdoy, called, ph.stgdoy, pltpop, ph.ears, g, c)
    em = emergence_init(called & (yrdoy == ph.stgdoy[..., 8]), pltpop, g, spe, c)
    grow = called & (ph.mdate != yrdoy) & (s <= 5) & ~((s == 5) & (pltpop <= c.pltpop_min5))

    tmax = jnp.asarray(f.tmax)
    tmin = jnp.asarray(f.tmin)
    swfac, turfac = stq.swfac, stq.turfac
    carbo = assimilation(
        f.srad, tmax, tmin, f.co2, em.lai, pltpop, params.rowspc, swfac, params.soil.slpf, cul.rue, spe, c
    )
    fexp = jnp.minimum(turfac, 1.0 - stq.satfac)  # min(AGEFAC, TURFAC, 1 - SATFAC, PSTRES2, KSTRES)
    dtt, sumdtt = ph.dtt, ph.sumdtt
    la = leaf_appearance(em.cumph, dtt, cul.phint, sumdtt, ph.p3, c)

    org = OrganGrowth(
        pla=em.pla,
        lfwt=em.lfwt,
        stmwt=em.stmwt,
        earwt=g.earwt,
        grort=g.grort,
        slan=g.slan,
        cls=g.cum_leaf_senes,
    )
    b1, seedrv_1 = juvenile_growth(org, carbo, la, fexp, em.seedrv, sumdtt, pltpop, c)
    b2 = floral_induction_growth(org, carbo, la, fexp, sumdtt, pltpop, c)
    b3, cumdtteg_3 = tassel_silk_growth(
        org, carbo, la, fexp, sumdtt, ph.p3, ph.tlno, ph.xnti, spe.bsgdd, turfac, pltpop, g.stg2cls, c
    )
    b4, cumdtteg_4, sump_4 = silk_efg_growth(
        org, carbo, fexp, dtt, sumdtt, g.cumdtteg, sd.sump, pltpop, g.stg2cls, c
    )
    b5, gf = grain_fill_growth(
        org,
        carbo,
        (tmax + tmin) * 0.5,
        swfac,
        sumdtt,
        ph.gpp,
        g.grnwt,
        g.grogrn,
        sd.emat,
        g.cmat,
        sd.swmin,
        sd.swmax,
        cul.g3,
        cul.p5,
        spe,
        c,
    )

    masks = [grow & (s == k) for k in range(1, 6)]
    m1, m2, m3, m4, m5 = masks
    xn = jnp.select(masks, [la.xn, la.xn, la.xn3, g.xn, g.xn], g.xn)
    mat = gf.maturity
    return GrosubDay(
        grow=grow,
        sd=sd,
        em=em,
        carbo=carbo,
        organs=select_stage_block(masks, (b1, b2, b3, b4, b5), org),
        cumph=jnp.select(masks, [la.cumph, la.cumph, la.cumph3, em.cumph, em.cumph], em.cumph),
        xn=xn,
        leafno=jnp.where(m1 | m2 | m3, jnp.floor(xn).astype(em.leafno.dtype), em.leafno),
        cumdtteg=jnp.select(masks, [g.cumdtteg, g.cumdtteg, cumdtteg_3, cumdtteg_4, g.cumdtteg], g.cumdtteg),
        seedrv=jnp.where(m1, seedrv_1, em.seedrv),
        stg2cls=jnp.where(m2, b2.cls, g.stg2cls),
        sump=jnp.where(m4, sump_4, sd.sump),
        grnwt=jnp.where(m5, gf.grnwt, g.grnwt),
        grogrn=jnp.where(m5, gf.grogrn, g.grogrn),
        emat=jnp.where(m5, mat.emat, sd.emat),
        cmat=jnp.where(m5, mat.cmat, g.cmat),
        sumdtt=jnp.where(m5 & mat.early, cul.p5 * jnp.ones_like(pltpop), ph.sumdtt),
    )


def growth_totals(
    g: CeresGrowthState,
    d: GrosubDay,
    senla: Array,
    lai: Array,
    fail: CropFailure,
    canht_pot: ArrayLike,
    c: GrosubCoefficients,
) -> CeresGrowthState:
    """The growth state at the end of the day: non-negative weights and areas, root weight
    ``RTWT + 0.5 GRORT - 0.005 RTWT``, ``BIOMAS``, ``RSTAGE``, ``MAXLAI`` and ``CANHT``.

    ``lai`` is the LAI after senescence (before the ``MAX(0, LAI)``).

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, from ``IF (CARBO .LE. 0.0)`` to ``MAXLAI``.
    """
    grow, em, new = d.grow, d.em, d.organs
    lfwt = _nonneg(grow, new.lfwt)
    pla = _nonneg(grow, new.pla)
    lai = _nonneg(grow, lai)
    stmwt = _nonneg(grow, new.stmwt)
    earwt = _nonneg(grow, new.earwt)
    pltpop = _nonneg(grow, g.pltpop)
    rtwt = em.rtwt
    leaf = eqx.tree_at(
        lambda q: (q.area, q.mass),
        em.leaf,
        (em.leaf.area.at[..., 0].set(pla), em.leaf.mass.at[..., 0].set(lfwt)),
    )
    return g.replace(
        leaf=leaf,
        pltpop=pltpop,
        senla=senla,
        lai=lai,
        stmwt=stmwt,
        earwt=earwt,
        grnwt=_nonneg(grow, d.grnwt),
        rtwt=jnp.where(
            grow, jnp.maximum(rtwt + c.root_growth_frac * new.grort - c.root_senes * rtwt, 0.0), rtwt
        ),
        seedrv=d.seedrv,
        cumph=d.cumph,
        xn=d.xn,
        leafno=d.leafno.astype(g.leafno.dtype),
        slan=new.slan,
        stg2cls=d.stg2cls,
        cum_leaf_senes=new.cls,
        swmin=d.sd.swmin,
        swmax=d.sd.swmax,
        sump=d.sump,
        cumdtteg=d.cumdtteg,
        carbo=jnp.where(grow, jnp.where(d.carbo <= 0.0, c.carbo_floor, d.carbo), g.carbo),
        grort=new.grort,
        grogrn=d.grogrn,
        emat=d.emat.astype(g.emat.dtype),
        cmat=d.cmat.astype(g.cmat.dtype),
        icold=fail.icold.astype(g.icold.dtype),
        nwsd=fail.nwsd.astype(g.nwsd.dtype),
        maxlai=jnp.where(grow, jnp.maximum(g.maxlai, lai), g.maxlai),
        canht=jnp.where(grow & (lai >= g.maxlai), canopy_height(lai, pltpop, canht_pot, c), g.canht),
        biomas=jnp.where(grow, (lfwt + stmwt + earwt) * pltpop, g.biomas),
        rstage=jnp.where(grow, jnp.where(d.leafno > 0, fail.istage, 0), g.rstage).astype(g.rstage.dtype),
    )


@process(
    reads=("phen", "stress", "growth"),
    writes=(
        "growth",
        "phen.istage",
        "phen.mdate",
        "phen.crop_status",
        "phen.sumdtt",
        "phen.ears",
        "phen.gpp",
    ),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for (BSD-3)",
    fortran_name="MZ_GROSUB",
)
def ceres_growth(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """One day of ``MZ_GROSUB`` (``DYNAMIC = INTEGR``) after the stress block, for every crop.

    In the Fortran order: stage-date resets, assimilation, leaf appearance and the block of the
    crop's stage (:func:`grosub_blocks`: :func:`juvenile_growth`, :func:`floral_induction_growth`,
    :func:`tassel_silk_growth`, :func:`silk_efg_growth`, :func:`grain_fill_growth`, all
    evaluated and selected by stage), leaf senescence, cold / drought failure and the state
    totals (:func:`growth_totals`). The hard-coded coefficients come from
    ``params.coef().grosub``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, DYNAMIC = INTEGR (BSD-3).
    """
    ph = state.phen
    g = state.growth
    cul = params.cultivar
    c = params.coef().grosub
    d = grosub_blocks(state, params, forcing_t, c)
    grow, em = d.grow, d.em
    tmin = jnp.asarray(forcing_t.tmin)
    swfac = state.stress.swfac

    senla, lai = leaf_senescence(
        d.organs.pla, em.senla, d.organs.slan, em.lai, swfac, tmin, g.pltpop, params.species.fslfw, c
    )
    senla = jnp.where(grow, senla, em.senla)
    lai = jnp.where(grow, lai, em.lai)
    fail = crop_failure(
        grow,
        ph.istage,
        ph.mdate,
        ph.crop_status,
        jnp.asarray(forcing_t.yrdoy),
        d.leafno,
        lai,
        tmin,
        swfac,
        g.icold,
        g.nwsd,
        cul.tsen,
        cul.cday,
        c,
    )
    new_growth = growth_totals(g, d, senla, lai, fail, params.species.canht_pot, c)
    new_phen = eqx.tree_at(
        lambda p: (p.istage, p.mdate, p.crop_status, p.sumdtt, p.ears, p.gpp),
        ph,
        (
            fail.istage.astype(ph.istage.dtype),
            fail.mdate.astype(ph.mdate.dtype),
            fail.status.astype(ph.crop_status.dtype),
            d.sumdtt,
            _nonneg(grow, d.sd.ears),
            _nonneg(grow, ph.gpp),
        ),
    )
    return eqx.tree_at(lambda x: (x.phen, x.growth), state, (new_phen, new_growth))
