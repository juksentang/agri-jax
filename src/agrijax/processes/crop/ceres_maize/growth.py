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

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.organs import appear
from agrijax.core.process import process

from ._util import curv_lin, safe_div, tabex, trunc_st
from .state import CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = ["ceres_growth", "ceres_stress", "saturation_factor", "water_stress_factors"]

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
    sw: ArrayLike, sat: ArrayLike, dlayr: ArrayLike, rlv: ArrayLike, tss: ArrayLike, pormin: ArrayLike
) -> tuple[Array, Array]:
    """``(SATFAC, TSS)``: root-length-weighted excess-water stress and the updated saturation days.

    A layer with air-filled porosity ``SAT - SW`` below ``PORMIN`` counts one more saturated day;
    after more than 2 days its root activity drops to ``(SAT - SW) / PORMIN``. ``sw``, ``sat``,
    ``dlayr`` are ``[n_layer]``; ``rlv`` and ``tss`` ``[n_crop, n_layer]``.

    Source: DSSAT-CSM MZ_GROSUB.for, "Compute Water Saturation Factors".
    """
    air = jnp.asarray(sat) - jnp.asarray(sw)
    tss_new = jnp.where(air >= pormin, 0.0, jnp.asarray(tss) + 1.0)
    swexf = jnp.where(tss_new > 2.0, jnp.maximum(air / jnp.maximum(jnp.asarray(pormin), _EPS), 0.0), 1.0)
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
        forcing_t.sw, params.soil.sat, params.soil.dlayr, state.roots.rlv, st.tss, params.species.pormin
    )
    satfac = jnp.where(run, satfac_new, st.satfac)
    tss = jnp.where(run[..., None], tss_new, st.tss)
    new = eqx.tree_at(lambda x: (x.swfac, x.turfac, x.satfac, x.tss), st, (swfac, turfac, satfac, tss))
    return eqx.tree_at(lambda x: x.stress, state, new)


def _pow(x: Array, p: float) -> Array:
    """``x ** p`` on a clamped base (finite value and derivative everywhere)."""
    return jnp.maximum(x, _EPS) ** p


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

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, DYNAMIC = INTEGR (BSD-3).
    """
    ph = state.phen
    g = state.growth
    stq = state.stress
    cul = params.cultivar
    spe = params.species
    f = forcing_t
    yrdoy = jnp.asarray(f.yrdoy)
    s = ph.istage
    called = (s >= 1) & (s <= 6)
    pltpop = g.pltpop
    one = jnp.ones_like(pltpop)

    # ------------------------------------------------ stage-date initialisations (before any return)
    at_ti = called & (yrdoy == ph.stgdoy[..., 1])
    at_silk = called & (yrdoy == ph.stgdoy[..., 2])
    at_efg = called & (yrdoy == ph.stgdoy[..., 3])
    at_em = called & (yrdoy == ph.stgdoy[..., 8])
    ears = jnp.where(at_ti, pltpop, ph.ears)
    swmin = jnp.where(at_silk, g.stmwt * 0.85, g.swmin)
    sump = jnp.where(at_silk, 0.0, g.sump)
    swmax = jnp.where(at_efg, g.stmwt, g.swmax)
    emat = jnp.where(at_efg, 0, g.emat)

    leafq = appear(g.leaf, at_em, spe.plae * one, spe.lfwte * one)
    pla = jnp.where(at_em, spe.plae, g.leaf.area[..., 0])
    lfwt = jnp.where(at_em, spe.lfwte, g.leaf.mass[..., 0])
    stmwt = jnp.where(at_em, spe.stmwte, g.stmwt)
    rtwt = jnp.where(at_em, spe.rtwte, g.rtwt)
    seedrv = jnp.where(at_em, spe.seedrve, g.seedrv)
    leafno = jnp.where(at_em, jnp.floor(spe.leafnoe).astype(g.leafno.dtype), g.leafno)
    senla = jnp.where(at_em, 0.0, g.senla)
    lai = jnp.where(at_em, pltpop * pla * 0.0001, g.lai)
    cumph = jnp.where(at_em, 0.514, g.cumph)

    run = called & (ph.mdate != yrdoy)
    grow = run & (s <= 5) & ~((s == 5) & (pltpop <= 0.01))

    # ------------------------------------------------ assimilation
    tmax = jnp.asarray(f.tmax)
    tmin = jnp.asarray(f.tmin)
    swfac = stq.swfac
    turfac = stq.turfac
    satfac = stq.satfac
    tempm = (tmax + tmin) * 0.5
    par = jnp.asarray(f.srad) * spe.parsr
    lifac = 1.5 - 0.768 * _pow((params.rowspc * 0.01) ** 2 * pltpop, 0.1)
    pco2 = tabex(spe.co2y, spe.co2x, f.co2)
    ipar = jnp.where(pltpop > 0.0, safe_div(par, pltpop) * (1.0 - jnp.exp(-lifac * lai)), 0.0)
    pcarb = ipar * cul.rue * pco2
    tavgd = 0.25 * tmin + 0.75 * tmax
    pr = spe.prftc
    prft = jnp.clip(curv_lin(pr[..., 0], pr[..., 1], pr[..., 2], pr[..., 3], tavgd), 0.0, 1.0)
    carbo = jnp.maximum(pcarb * jnp.minimum(prft, swfac) * params.soil.slpf, 0.0)
    fexp = jnp.minimum(turfac, 1.0 - satfac)  # min(AGEFAC, TURFAC, 1 - SATFAC, PSTRES2, KSTRES)

    # ------------------------------------------------ leaf appearance (stages 1-3)
    dtt = ph.dtt
    sumdtt = ph.sumdtt
    pc = jnp.where(cumph < 5.0, 0.66 + 0.068 * cumph, 1.0)
    ti = dtt / jnp.maximum(cul.phint * pc, _EPS)
    cumph1 = cumph + ti
    xn1 = cumph1 + 1.0

    # ------------------------------------------------ stage 1: emergence -> end of juvenile
    plag1 = jnp.where(xn1 < 4.0, 4.0 * xn1 * ti * fexp, 3.0 * xn1 * xn1 * ti * fexp)
    pla_1 = pla + plag1
    xlfwt = jnp.maximum(_pow(pla_1 / 250.0, 1.25), lfwt)
    grolf_1 = xlfwt - lfwt
    grort_1 = carbo - grolf_1
    low = grort_1 <= 0.25 * carbo
    grort_1 = jnp.where(low, carbo * 0.25, grort_1)
    seedrv_1 = jnp.where(low, seedrv + carbo - grolf_1 - grort_1, seedrv)
    spent = low & (seedrv_1 <= 0.0)
    seedrv_1 = jnp.where(spent, 0.0, seedrv_1)
    grolf_1 = jnp.where(spent, carbo * 0.75, grolf_1)
    pla_1 = jnp.where(spent, _pow(lfwt + grolf_1, 0.8) * 267.0, pla_1)
    lfwt_1 = lfwt + grolf_1
    slan_1 = jnp.where(grolf_1 > 0.0, sumdtt * pla_1 / 10000.0, g.slan)
    lfwt_1 = lfwt_1 - slan_1 / 600.0
    cls_1 = slan_1 / 600.0 * pltpop * 10.0

    # ------------------------------------------------ stage 2: end of juvenile -> tassel initiation
    plag2 = 3.5 * xn1 * xn1 * ti * fexp
    pla_2 = pla + plag2
    grolf_2 = _pow(pla_2 / 267.0, 1.25) - lfwt
    cap = grolf_2 >= carbo * 0.75
    grolf_2 = jnp.where(cap, carbo * 0.75, grolf_2)
    pla_2 = jnp.where(cap, _pow(lfwt + grolf_2, 0.8) * 267.0, pla_2)
    grort_2 = carbo - grolf_2
    lfwt_2 = lfwt + grolf_2
    slan_2 = sumdtt * pla_2 / 10000.0
    lfwt_2 = lfwt_2 - slan_2 / 600.0
    cls_2 = slan_2 / 600.0 * pltpop * 10.0

    # ------------------------------------------------ stage 3: tassel initiation -> silking
    p3 = ph.p3
    cumph3 = jnp.where(sumdtt > p3 - 2.0 * cul.phint, cumph1 - ti, cumph1)
    xn3 = cumph3 + 1.0
    eg = sumdtt >= p3 - spe.bsgdd
    cumdtteg_3 = jnp.where(eg, sumdtt - (p3 - spe.bsgdd), 0.0)
    groear_3 = jnp.where(eg, 0.81 / (1.0 + jnp.exp(-0.02 * (cumdtteg_3 - 210.0))) * carbo, 0.0)
    tlno = ph.tlno
    plag3 = jnp.select(
        [xn3 < 12.0, xn3 < tlno - 2.9999],
        [3.5 * xn3 * xn3 * ti * fexp, 3.5 * 170.0 * ti * fexp],
        170.0 * 3.5 * _pow(xn3 + 3.0 - tlno, -0.5) * ti * fexp,
    )
    grolf_3 = 0.00116 * plag3 * _pow(pla, 0.25)
    grostm_3 = jnp.where(
        (xn3 < 12.0) | (xn3 < tlno - 2.9999), grolf_3 * 0.0182 * (xn3 - ph.xnti) ** 2, 3.0 * 3.1 * ti * fexp
    )
    gt = grostm_3 > groear_3
    groear_3 = jnp.where(gt, groear_3, grostm_3 * 0.5)
    grostm_3 = jnp.where(gt, grostm_3 - groear_3, grostm_3 * 0.5)
    grort_3 = carbo - grolf_3 - grostm_3 - groear_3
    scale = (grort_3 <= 0.10 * carbo) & (turfac > 0.0)
    any_g = (grolf_3 > 0.0) | (grostm_3 > 0.0)
    grf = jnp.where(any_g, carbo * 0.90 / jnp.maximum(grostm_3 + grolf_3 + groear_3, _EPS), 1.0)
    grort_3 = jnp.where(scale & any_g, carbo * 0.10, grort_3)
    grf = jnp.where(scale, grf, 1.0)
    grolf_3 = grolf_3 * grf
    grostm_3 = grostm_3 * grf
    groear_3 = groear_3 * grf
    pla_3 = _pow(lfwt + grolf_3, 0.8) * 267.0
    slan_3 = pla_3 / 1000.0
    lfwt_3 = lfwt + grolf_3 - slan_3 / 600.0
    stmwt_3 = stmwt + grostm_3
    earwt_3 = g.earwt + groear_3
    cls_3 = slan_3 / 600.0 * pltpop * 10.0 + g.stg2cls

    # ------------------------------------------------ stage 4: silking -> effective grain filling
    cumdtteg_4 = g.cumdtteg + dtt
    groear_4 = 0.81 / (1.0 + jnp.exp(-0.02 * (cumdtteg_4 - 210.0))) * carbo * fexp
    big = carbo > groear_4 + carbo * 0.08
    grostm_4 = jnp.where(big, carbo - groear_4 - carbo * 0.08, (carbo - groear_4) * 0.5)
    grort_4 = jnp.where(big, carbo * 0.08, (carbo - groear_4) * 0.5)
    slan_4 = pla * (0.05 + sumdtt / 200.0 * 0.05)
    lfwt_4 = lfwt - slan_4 / 600.0
    earwt_4 = g.earwt + groear_4
    stmwt_4 = stmwt + grostm_4
    sump_4 = sump + carbo
    cls_4 = slan_4 / 600.0 * pltpop * 10.0 + g.stg2cls

    # ------------------------------------------------ stage 5: effective grain filling
    act = jnp.abs(carbo) > 0.0001
    rg = spe.rgfil
    rgfill = jnp.clip(curv_lin(rg[..., 0], rg[..., 1], rg[..., 2], rg[..., 3], tempm), 0.0, 1.0)
    grogrn_a = rgfill * ph.gpp * cul.g3 * 0.001 * (0.45 + 0.55 * swfac)
    fast = rgfill > spe.rsgr
    emat_a = jnp.where(fast, 0, emat + 1)
    early_a = ~fast & (emat_a.astype(carbo.dtype) > spe.rsgrt)
    emat_a = jnp.where(early_a, 0, emat_a)
    cmat_b = g.cmat + 1
    early_b = cmat_b.astype(carbo.dtype) >= spe.carbot
    early = jnp.where(act, early_a, early_b)
    cmat_5 = jnp.where(act, 0, cmat_b)
    emat_5 = jnp.where(act, emat_a, jnp.where(early_b, 0, emat))
    slan_5 = jnp.where(act, pla * (0.1 + 0.60 * (safe_div(sumdtt, cul.p5)) ** 3), g.slan)
    grogrn_5 = jnp.where(act, grogrn_a, g.grogrn)
    grort_5 = jnp.where(act | early_b, 0.0, g.grort)
    grostm_5 = carbo - grogrn_5
    pos = grostm_5 >= 0.0
    stmwt_5 = jnp.where(pos, stmwt + grostm_5 * 0.50, stmwt + carbo - grogrn_5)
    grort_5 = jnp.where(pos, grostm_5 * 0.50, grort_5)
    thin = ~pos & (stmwt_5 <= swmin * 1.07)
    stmwt_5 = jnp.where(thin, stmwt_5 + lfwt * 0.0050, stmwt_5)
    floor_ = thin & (stmwt_5 < swmin)
    stmwt_5 = jnp.where(floor_, swmin, stmwt_5)
    grogrn_5 = jnp.where(floor_, carbo, grogrn_5)
    grnwt_5 = g.grnwt + grogrn_5
    earwt_5 = g.earwt + grogrn_5
    stmwt_5 = jnp.minimum(stmwt_5, swmax)

    # ------------------------------------------------ select the block of each crop's stage
    m1 = grow & (s == 1)
    m2 = grow & (s == 2)
    m3 = grow & (s == 3)
    m4 = grow & (s == 4)
    m5 = grow & (s == 5)

    def sel(
        v1: ArrayLike, v2: ArrayLike, v3: ArrayLike, v4: ArrayLike, v5: ArrayLike, keep: ArrayLike
    ) -> Array:
        return jnp.select([m1, m2, m3, m4, m5], [v1, v2, v3, v4, v5], keep)

    cumph_n = sel(cumph1, cumph1, cumph3, cumph, cumph, cumph)
    xn_n = sel(xn1, xn1, xn3, g.xn, g.xn, g.xn)
    leafno_n = jnp.where(m1 | m2 | m3, jnp.floor(xn_n).astype(leafno.dtype), leafno)
    pla_n = sel(pla_1, pla_2, pla_3, pla, pla, pla)
    lfwt_n = sel(lfwt_1, lfwt_2, lfwt_3, lfwt_4, lfwt, lfwt)
    stmwt_n = sel(stmwt, stmwt, stmwt_3, stmwt_4, stmwt_5, stmwt)
    earwt_n = sel(g.earwt, g.earwt, earwt_3, earwt_4, earwt_5, g.earwt)
    grnwt_n = jnp.where(m5, grnwt_5, g.grnwt)
    seedrv_n = jnp.where(m1, seedrv_1, seedrv)
    slan_n = sel(slan_1, slan_2, slan_3, slan_4, slan_5, g.slan)
    grort_n = sel(grort_1, grort_2, grort_3, grort_4, grort_5, g.grort)
    grogrn_n = jnp.where(m5, grogrn_5, g.grogrn)
    cls_n = sel(cls_1, cls_2, cls_3, cls_4, g.cum_leaf_senes, g.cum_leaf_senes)
    stg2cls_n = jnp.where(m2, cls_2, g.stg2cls)
    cumdtteg_n = sel(g.cumdtteg, g.cumdtteg, cumdtteg_3, cumdtteg_4, g.cumdtteg, g.cumdtteg)
    sump_n = jnp.where(m4, sump_4, sump)
    emat_n = jnp.where(m5, emat_5, emat)
    cmat_n = jnp.where(m5, cmat_5, g.cmat)
    sumdtt_n = jnp.where(m5 & early, cul.p5 * one, ph.sumdtt)

    # ------------------------------------------------ leaf senescence (every growing day)
    slfw = (1.0 - spe.fslfw) + spe.fslfw * swfac
    slfc = jnp.where(lai > 4.0, 1.0 - 0.008 * (lai - 4.0), 1.0)
    slft = jnp.where(tmin <= 6.0, jnp.maximum(0.0, 1.0 - 0.01 * (tmin - 6.0) ** 2), 1.0)
    plas = (pla_n - senla) * (1.0 - jnp.minimum(jnp.minimum(slfw, slfc), jnp.minimum(slft, 1.0)))
    senla_g = jnp.minimum(jnp.maximum(senla + plas, slan_n), pla_n)
    lai_g = (pla_n - senla_g) * pltpop * 0.0001
    senla_n = jnp.where(grow, senla_g, senla)
    lai_n = jnp.where(grow, lai_g, lai)

    # ------------------------------------------------ crop failure from cold or drought
    istage = s
    mdate = ph.mdate
    status = ph.crop_status
    icold_n = jnp.where(grow, jnp.where(tmin <= cul.tsen, g.icold + 1, 0), g.icold)
    cold = grow & (
        ((leafno_n > 4) & (lai_n <= 0.0) & (istage <= 4) & (icold_n > 6))
        | (icold_n.astype(carbo.dtype) >= cul.cday)
    )
    istage = jnp.where(cold, 6, istage)
    mdate = jnp.where(cold, yrdoy, mdate)
    status = jnp.where(cold, 32, status)
    nwsd_n = jnp.where(grow, jnp.where(swfac > 0.1, 0, g.nwsd + 1), g.nwsd)
    drought = grow & (lai_n <= 0.1) & (istage < 4) & (nwsd_n > 10)
    istage = jnp.where(drought, 6, istage)
    mdate = jnp.where(drought, yrdoy, mdate)
    status = jnp.where(drought, 33, status)

    # ------------------------------------------------ state totals
    carbo_n = jnp.where(grow, jnp.where(carbo <= 0.0, 0.001, carbo), g.carbo)
    rtwt_n = jnp.where(grow, jnp.maximum(rtwt + 0.5 * grort_n - 0.005 * rtwt, 0.0), rtwt)
    lfwt_n = jnp.where(grow, jnp.maximum(lfwt_n, 0.0), lfwt_n)
    pla_n = jnp.where(grow, jnp.maximum(pla_n, 0.0), pla_n)
    lai_n = jnp.where(grow, jnp.maximum(lai_n, 0.0), lai_n)
    stmwt_n = jnp.where(grow, jnp.maximum(stmwt_n, 0.0), stmwt_n)
    grnwt_n = jnp.where(grow, jnp.maximum(grnwt_n, 0.0), grnwt_n)
    earwt_n = jnp.where(grow, jnp.maximum(earwt_n, 0.0), earwt_n)
    gpp_n = jnp.where(grow, jnp.maximum(ph.gpp, 0.0), ph.gpp)
    pltpop_n = jnp.where(grow, jnp.maximum(pltpop, 0.0), pltpop)
    ears_n = jnp.where(grow, jnp.maximum(ears, 0.0), ears)
    biomas_n = jnp.where(grow, (lfwt_n + stmwt_n + earwt_n) * pltpop_n, g.biomas)
    rstage_n = jnp.where(grow, jnp.where(leafno_n > 0, istage, 0), g.rstage)
    tall = grow & (lai_n >= g.maxlai)
    canht_new = jnp.minimum(
        lai_n / jnp.maximum(0.4238 * pltpop_n + 0.3424, _EPS) * spe.canht_pot, spe.canht_pot * one
    )
    canht_n = jnp.where(tall, canht_new, g.canht)
    maxlai_n = jnp.where(grow, jnp.maximum(g.maxlai, lai_n), g.maxlai)

    leafq = eqx.tree_at(
        lambda q: (q.area, q.mass),
        leafq,
        (leafq.area.at[..., 0].set(pla_n), leafq.mass.at[..., 0].set(lfwt_n)),
    )
    new_growth = eqx.tree_at(
        lambda x: (
            x.leaf,
            x.pltpop,
            x.senla,
            x.lai,
            x.stmwt,
            x.earwt,
            x.grnwt,
            x.rtwt,
            x.seedrv,
            x.cumph,
            x.xn,
            x.leafno,
            x.slan,
            x.stg2cls,
            x.cum_leaf_senes,
            x.swmin,
            x.swmax,
            x.sump,
            x.cumdtteg,
            x.carbo,
            x.grort,
            x.grogrn,
            x.emat,
            x.cmat,
            x.icold,
            x.nwsd,
            x.maxlai,
            x.canht,
            x.biomas,
            x.rstage,
        ),
        g,
        (
            leafq,
            pltpop_n,
            senla_n,
            lai_n,
            stmwt_n,
            earwt_n,
            grnwt_n,
            rtwt_n,
            seedrv_n,
            cumph_n,
            xn_n,
            leafno_n.astype(g.leafno.dtype),
            slan_n,
            stg2cls_n,
            cls_n,
            swmin,
            swmax,
            sump_n,
            cumdtteg_n,
            carbo_n,
            grort_n,
            grogrn_n,
            emat_n.astype(g.emat.dtype),
            cmat_n.astype(g.cmat.dtype),
            icold_n.astype(g.icold.dtype),
            nwsd_n.astype(g.nwsd.dtype),
            maxlai_n,
            canht_n,
            biomas_n,
            rstage_n.astype(g.rstage.dtype),
        ),
    )
    new_phen = eqx.tree_at(
        lambda p: (p.istage, p.mdate, p.crop_status, p.sumdtt, p.ears, p.gpp),
        ph,
        (
            istage.astype(ph.istage.dtype),
            mdate.astype(ph.mdate.dtype),
            status.astype(ph.crop_status.dtype),
            sumdtt_n,
            ears_n,
            gpp_n,
        ),
    )
    return eqx.tree_at(lambda x: (x.phen, x.growth), state, (new_phen, new_growth))
