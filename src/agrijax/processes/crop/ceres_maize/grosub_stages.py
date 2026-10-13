"""CERES-Maize ``MZ_GROSUB`` kernels up to the stage blocks.

Stage-date and emergence initialisations, daily assimilation ``CARBO``, leaf appearance and
the five per-stage organ-growth blocks (juvenile, floral induction, tassel initiation to
silking, silking to effective grain filling, effective grain filling) with their helpers, and
the per-crop selection of the stage block. Split out of
:mod:`agrijax.processes.crop.ceres_maize.growth`, which re-exports every public name.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_GROSUB.for`` (BSD-3, Copyright 1998-2026
DSSAT Foundation, University of Florida, International Fertilizer Development Center); Jones &
Kiniry (1986) CERES-Maize; ear growth after J. I. Lizaso (2006).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.organs import OrganQueue, appear
from agrijax.core.units import KG_HA_PER_G_M2, M2_PER_CM2, M_PER_CM

from ._util import coef_div, curv_lin, safe_div, tabex
from .coefficients import GrosubCoefficients
from .constants import G_PER_MG
from .state import CeresGrowthState, CeresSpecies
from .stress import _EPS

__all__ = [
    "EarlyMaturity",
    "EmergenceInit",
    "GrainFill",
    "LeafAppearance",
    "OrganGrowth",
    "StageDateInit",
    "assimilation",
    "ear_growth_fraction",
    "early_maturity",
    "emergence_init",
    "floral_induction_growth",
    "grain_fill_growth",
    "grain_fill_rate",
    "juvenile_growth",
    "leaf_appearance",
    "select_stage_block",
    "silk_efg_growth",
    "stage3_demand",
    "stage3_partition",
    "stage_date_init",
    "tassel_silk_growth",
]


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
        lai=jnp.where(at_em, pltpop * pla * M2_PER_CM2, g.lai),
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
    lifac = c.lifac_max - c.lifac_slope * _pow((rowspc * M_PER_CM) ** 2 * pltpop, c.lifac_exp)
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

    ``SLAN / 600`` is one guarded quotient used twice: with traced coefficients, two separate
    :func:`coef_div` calls compiled differently from the unguarded ``slan / c`` and moved the
    stage-3 ``CumLeafSenes`` by 1 ulp in the calibration runs (:mod:`agrijax.calib.ceres`); one
    quotient keeps the forward bit-identical.
    """
    senes = coef_div(slan, c.sla_senes)
    return lfwt - senes, senes * pltpop * KG_HA_PER_G_M2


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
    xlfwt = jnp.maximum(_pow(coef_div(pla, c.sla_juvenile), c.lfwt_exp), lfwt)
    grolf = xlfwt - lfwt
    grort = carbo - grolf
    low = grort <= c.grort1_min_frac * carbo
    grort = jnp.where(low, carbo * c.grort1_min_frac, grort)
    seedrv_1 = jnp.where(low, seedrv + carbo - grolf - grort, seedrv)
    spent = low & (seedrv_1 <= 0.0)
    seedrv_1 = jnp.where(spent, 0.0, seedrv_1)
    grolf = jnp.where(spent, carbo * c.grolf_max_frac, grolf)
    pla = jnp.where(spent, _pow(lfwt + grolf, c.pla_exp) * c.sla_leaf, pla)
    slan_new = coef_div(sumdtt * pla, c.slan12_tt)
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
    grolf = _pow(coef_div(pla, c.sla_leaf), c.lfwt_exp) - lfwt
    cap = grolf >= carbo * c.grolf_max_frac
    grolf = jnp.where(cap, carbo * c.grolf_max_frac, grolf)
    pla = jnp.where(cap, _pow(lfwt + grolf, c.pla_exp) * c.sla_leaf, pla)
    grort = carbo - grolf
    slan = coef_div(sumdtt * pla, c.slan12_tt)
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
    slan = coef_div(pla, c.slan3_div)
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
    slan = org.pla * (c.slan4_base + coef_div(sumdtt, c.slan4_tt) * c.slan4_slope)
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
    return rgfill, rgfill * gpp * g3 * G_PER_MG * (c.grogrn_sw_base + c.grogrn_sw_slope * swfac)


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
