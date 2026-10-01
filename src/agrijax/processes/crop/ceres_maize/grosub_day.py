"""CERES-Maize ``MZ_GROSUB`` day assembly after the stage blocks.

:func:`grosub_blocks` runs the stage-date resets, assimilation, leaf appearance and the stage
blocks of :mod:`~agrijax.processes.crop.ceres_maize.grosub_stages` and selects each crop's
block; then leaf senescence (with the nitrogen factor :func:`nitrogen_senescence_factor`), the
grain-number cap :func:`grain_number_cap`, cold / drought crop failure, canopy height and the
end-of-day state totals (:func:`growth_totals`). Split out of
:mod:`agrijax.processes.crop.ceres_maize.growth`, which re-exports every public name.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_GROSUB.for`` (BSD-3, Copyright 1998-2026
DSSAT Foundation, University of Florida, International Fertilizer Development Center).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.units import M2_PER_CM2

from ._util import coef_div
from .coefficients import GrosubCoefficients
from .constants import (
    CROP_STATUS_COLD,
    CROP_STATUS_DROUGHT,
    ISTAGE_EFG,
    ISTAGE_END_LEAF_GROWTH,
    ISTAGE_MATURITY,
    PAIR_MEAN_WEIGHT,
)
from .grosub_stages import (
    EmergenceInit,
    OrganGrowth,
    StageDateInit,
    assimilation,
    emergence_init,
    floral_induction_growth,
    grain_fill_growth,
    juvenile_growth,
    leaf_appearance,
    select_stage_block,
    silk_efg_growth,
    stage_date_init,
    tassel_silk_growth,
)
from .state import CeresForcing, CeresGrowthState, CeresMaizeParams, CeresMaizeState
from .stress import _EPS, WATER_STRESS_COEFFICIENTS

__all__ = [
    "CropFailure",
    "GrosubDay",
    "canopy_height",
    "crop_failure",
    "grain_number_cap",
    "grosub_blocks",
    "growth_totals",
    "leaf_senescence",
    "nitrogen_senescence_factor",
]


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
    slfn: ArrayLike = 1.0,
) -> tuple[Array, Array]:
    """``(SENLA, LAI)`` after the day's senescence (water, nitrogen, competition and cold; P off).

    ``SLFW = (1 - FSLFW) + FSLFW SWFAC``, ``SLFN`` the nitrogen factor
    (:func:`nitrogen_senescence_factor`; ``1`` with nitrogen off), ``SLFC = 1 - 0.008 (LAI - 4)``
    above LAI 4, ``SLFT = max(0, 1 - 0.01 (TMIN - 6)^2)`` at or below 6 degC; ``PLAS = (PLA -
    SENLA) (1 - min(SLFW, SLFC, SLFT, SLFN))`` (``SLFP = 1``); ``SENLA`` is at least ``SLAN`` and
    at most ``PLA``. ``lai`` is the day-start LAI. ``min`` is exact, so ``SLFN = 1`` gives the
    nitrogen-off value bit for bit.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, "Leaf senescence" (SLFW, SLFN, SLFC, SLFT,
    PLAS: ``PLAS = (PLA-SENLA)*(1.0-AMIN1(SLFW,SLFC,SLFT,SLFN,SLFP))``, line 1632).
    """
    slfw = (1.0 - fslfw) + fslfw * swfac
    slfc = jnp.where(lai > c.slfc_lai, 1.0 - c.slfc_slope * (lai - c.slfc_lai), 1.0)
    slft = jnp.where(
        tmin <= c.slft_tmin, jnp.maximum(0.0, 1.0 - c.slft_coef * (tmin - c.slft_tmin) ** 2), 1.0
    )
    fac = jnp.minimum(jnp.minimum(slfw, slfc), jnp.minimum(jnp.minimum(slft, 1.0), slfn))
    plas = (pla - senla) * (1.0 - fac)
    senla_g = jnp.minimum(jnp.maximum(senla + plas, slan), pla)
    return senla_g, (pla - senla_g) * pltpop * M2_PER_CM2


def nitrogen_senescence_factor(agefac: ArrayLike, fslfn: ArrayLike) -> Array:
    """``SLFN = (1 - FSLFN) + FSLFN AGEFAC``: the leaf senescence factor of nitrogen stress
    (``FSLFN`` from the species file), in every stage.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for:1612, ``SLFN   = (1-FSLFN) + FSLFN*AGEFAC``.
    """
    return (1.0 - jnp.asarray(fslfn)) + fslfn * jnp.asarray(agefac)


def grain_number_cap(gpp: Array, ndef3: ArrayLike, npool: ArrayLike, c: GrosubCoefficients) -> Array:
    """``GPP = min(GPP NDEF3, NPOOL / (0.062 x 0.0095))``: the grain number per plant limited by
    the nitrogen factor ``NDEF3`` and by the plant's translocatable nitrogen ``NPOOL`` [g plant-1].

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for:1556-1557 (``IF (ICSDUR .EQ. 1)``, ``GPP  = AMIN1
    (GPP*NDEF3,(NPOOL/(0.062*0.0095)))``), inside the ``ISTAGE .EQ. 5`` nitrogen block after the
    day's grain growth.
    """
    return jnp.minimum(gpp * ndef3, coef_div(npool, c.gpp_npool_a * c.gpp_npool_b))


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
        (
            (leafno > c.cold_leafno)
            & (lai <= 0.0)
            & (istage <= ISTAGE_END_LEAF_GROWTH)
            & (icold_n > c.cold_days)
        )
        | (icold_n.astype(lai.dtype) >= cday)
    )
    istage = jnp.where(cold, ISTAGE_MATURITY, istage)
    mdate = jnp.where(cold, yrdoy, mdate)
    status = jnp.where(cold, CROP_STATUS_COLD, status)
    nwsd_n = jnp.where(grow, jnp.where(swfac > c.drought_swfac, 0, nwsd + 1), nwsd)
    drought = grow & (lai <= c.drought_lai) & (istage < ISTAGE_END_LEAF_GROWTH) & (nwsd_n > c.drought_days)
    return CropFailure(
        istage=jnp.where(drought, ISTAGE_MATURITY, istage),
        mdate=jnp.where(drought, yrdoy, mdate),
        status=jnp.where(drought, CROP_STATUS_DROUGHT, status),
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
    state: CeresMaizeState,
    params: CeresMaizeParams,
    forcing_t: CeresForcing,
    c: GrosubCoefficients,
    photo_stress: Array,
    agefac: ArrayLike = 1.0,
) -> GrosubDay:
    """Stage-date resets, assimilation, leaf appearance and the five stage blocks of
    ``MZ_GROSUB``, every block evaluated from the day-start organs and selected by stage.

    ``photo_stress`` is the stress factor that multiplies ``PCARB`` next to ``PRFT`` in
    ``CARBO``: ``SWFAC`` with nitrogen off (:func:`ceres_growth`), ``min(SWFAC, NSTRES)`` with a
    nitrogen stress (:func:`ceres_growth_nstress_replay`); ``min`` is exact, so
    ``min(PRFT, min(SWFAC, NSTRES)) = AMIN1(PRFT, SWFAC, NSTRES)``.

    ``agefac`` is the nitrogen factor ``AGEFAC`` of leaf expansion (``1`` with nitrogen off): the
    expansion factor is ``min(TURFAC, 1 - SATFAC)`` in stage 1 (4.8.6 removed ``AGEFAC`` there)
    and ``min(AGEFAC, TURFAC, 1 - SATFAC)`` in stages 2-4 (``PLAG``, the late stage-3 ``GROSTM``,
    the stage-4 ``GROEAR``), taken as a select (``AGEFAC`` where it is smaller): exact, and at the
    tie ``AGEFAC = 1`` both the value and the gradient are the nitrogen-off ones bit for bit.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR, from the stage-date initialisations to the
    end of the ``ISTAGE`` ``IF / ELSEIF`` chain; AGEFAC: lines 1210-1215 (stage 1, removed),
    1282-1283 (stage 2 PLAG), 1337-1355 (stage 3 PLAG, GROSTM), 1404-1405 (stage 4 GROEAR).
    """
    ph, g, stq = state.phen, state.growth, state.stress
    cul, spe, f = params.cultivar, params.species, forcing_t
    yrdoy = jnp.asarray(f.yrdoy)
    s = ph.istage
    called = (s >= 1) & (s <= WATER_STRESS_COEFFICIENTS.grosub_last_stage)
    pltpop = g.pltpop

    # stage-date initialisations (before any return), then the MZ_CERES / MZ_GROSUB returns
    sd = stage_date_init(yrdoy, called, ph.stgdoy, pltpop, ph.ears, g, c)
    em = emergence_init(called & (yrdoy == ph.stgdoy[..., 8]), pltpop, g, spe, c)
    grow = called & (ph.mdate != yrdoy) & (s <= ISTAGE_EFG) & ~((s == ISTAGE_EFG) & (pltpop <= c.pltpop_min5))

    tmax = jnp.asarray(f.tmax)
    tmin = jnp.asarray(f.tmin)
    swfac, turfac = stq.swfac, stq.turfac
    carbo = assimilation(
        f.srad,
        tmax,
        tmin,
        f.co2,
        em.lai,
        pltpop,
        params.rowspc,
        photo_stress,
        params.soil.slpf,
        cul.rue,
        spe,
        c,
    )
    fexp1 = jnp.minimum(turfac, 1.0 - stq.satfac)  # stage 1: min(TURFAC, 1 - SATFAC, PSTRES2, KSTRES)
    # stages 2-4: min(AGEFAC, TURFAC, 1 - SATFAC, PSTRES2, KSTRES), as a select so that at a tie
    # (AGEFAC = 1 = min(TURFAC, 1 - SATFAC), the nitrogen-off case) the gradient flows to fexp1 whole,
    # as without AGEFAC (jnp.minimum would split it between the tied arguments)
    fexp = jnp.where(agefac < fexp1, agefac, fexp1)
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
    b1, seedrv_1 = juvenile_growth(org, carbo, la, fexp1, em.seedrv, sumdtt, pltpop, c)
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
        (tmax + tmin) * PAIR_MEAN_WEIGHT,
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
