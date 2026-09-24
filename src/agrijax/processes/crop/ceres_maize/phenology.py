"""CERES-Maize phenology: daily thermal time and the stage machine of ``MZ_PHENOL``.

Every crop advances through the DSSAT ``ISTAGE`` codes 7 (sowing) -> 8 (germination) -> 9
(emergence) -> 1 (end of juvenile) -> 2 (tassel initiation) -> 3 (silking) -> 4 (beginning of
effective grain filling) -> 5 (end of effective grain filling) -> 6 (physiological maturity) ->
10 (after maturity). As in the Fortran ``IF / ELSEIF`` chain, at most one stage ends per day and
only the block of the stage the crop is in on entering the day runs; here every block is
evaluated for every crop and selected with ``jnp.where`` on the stage mask (the masks are
disjoint), so the process has no Python branch on state and no loop.

Nitrogen is off (``ISWNIT = N``): ``XSTAGE`` is kept for a later nitrogen module but nothing
reads it. The phosphorus bookkeeping fractions ``VegFrac`` / ``SeedFrac`` are not ported.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_PHENOL.for`` (BSD-3, Copyright 1998-2026 DSSAT
Foundation, University of Florida, International Fertilizer Development Center); thermal time
after J. T. Ritchie (CIMMYT 1998) as coded there; Jones & Kiniry (1986) CERES-Maize.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.process import process

from ._util import safe_div
from .state import CeresCultivar, CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = ["ceres_phenology", "thermal_time"]

_HOURS = jnp.arange(1, 25)


def thermal_time(
    tmax: ArrayLike,
    tmin: ArrayLike,
    srad: ArrayLike,
    dayl: ArrayLike,
    snow: ArrayLike,
    leafno: ArrayLike,
    istage: ArrayLike,
    cul: CeresCultivar,
) -> Array:
    """Daily thermal time ``DTT`` [degC d] of ``MZ_PHENOL``.

    Branches, in the Fortran order: ``TMAX < TBASE`` -> 0; ``TMIN > DOPT`` -> ``DOPT - TBASE``;
    up to 10 leaves the growing point is below ground and the soil (or snow-covered crown)
    temperature is used; otherwise the 24-point sine interpolation when a limit is crossed, or the
    daily mean minus ``TBASE``. ``DOPT`` is ``ROPT`` in stages 4-6, else ``TOPT``.

    Source: DSSAT-CSM MZ_PHENOL.for (DYNAMIC = INTEGR, thermal time block).
    """
    tmax = jnp.asarray(tmax)
    tmin = jnp.asarray(tmin)
    istage = jnp.asarray(istage)
    tbase = cul.tbase
    dopt = jnp.where((istage > 3) & (istage <= 6), cul.ropt, cul.topt)

    xs = jnp.minimum(jnp.asarray(snow), 15.0)
    snowfac = 0.4 + 0.0018 * (xs - 15.0) ** 2
    tempcn = jnp.where(tmin < 0.0, 2.0 + tmin * snowfac, tmin)
    tempcx = jnp.where(tmax < 0.0, 2.0 + tmax * snowfac, tmax)

    # soil temperature near the growing point (no snow)
    acoef = 0.01061 * jnp.asarray(srad) + 0.5902
    tdsoil = acoef * tmax + (1.0 - acoef) * tmin
    tnsoil = jnp.maximum(0.36354 * tmax + 0.63646 * tmin, tbase)
    tdsoil_c = jnp.minimum(tdsoil, dopt)
    dl = jnp.asarray(dayl)
    tmsoil = tdsoil_c * (dl / 24.0) + tnsoil * ((24.0 - dl) / 24.0)
    dtt_soil = jnp.where(tmsoil < tbase, (tbase + tdsoil_c) / 2.0 - tbase, (tnsoil + tdsoil_c) / 2.0 - tbase)
    dtt_soil = jnp.where(tdsoil < tbase, 0.0, jnp.minimum(dtt_soil, dopt - tbase))
    dtt_snow = (tempcn + tempcx) / 2.0 - tbase
    dtt_ground = jnp.where(xs > 0.0, dtt_snow, dtt_soil)

    # 24-point sine interpolation (SIN(3.14/12*I), I = 1..24)
    th = (tmax + tmin)[..., None] / 2.0 + (tmax - tmin)[..., None] / 2.0 * jnp.sin(3.14 / 12.0 * _HOURS)
    th = jnp.minimum(jnp.maximum(th, jnp.asarray(tbase)[..., None]), jnp.asarray(dopt)[..., None])
    dtt_hourly = jnp.sum((th - jnp.asarray(tbase)[..., None]) / 24.0, axis=-1)

    dtt = jnp.select(
        [
            tmax < tbase,
            tmin > dopt,
            jnp.asarray(leafno) <= 10,
            (tmin < tbase) | (tmax > dopt),
        ],
        [jnp.zeros_like(dtt_hourly), dopt - tbase, dtt_ground, dtt_hourly],
        (tmax + tmin) / 2.0 - tbase,
    )
    return jnp.maximum(dtt, 0.0)


def _layer_of_depth(depth: Array, dlayr: Array) -> Array:
    """0-based index of the first layer whose bottom is below ``depth`` (last layer if none)."""
    bottom = jnp.cumsum(dlayr)
    n = dlayr.shape[-1]
    inside = jnp.asarray(depth)[..., None] < bottom
    first = jnp.argmax(inside, axis=-1)
    return jnp.where(jnp.any(inside, axis=-1), first, n - 1).astype(jnp.int32)


def _pick(x: Array, idx: Array) -> Array:
    """``x[..., idx]`` for a ``[n_layer]`` array and a ``[n_crop]`` index, by one-hot sum."""
    hot = jnp.arange(x.shape[-1]) == idx[..., None]
    return jnp.sum(jnp.where(hot, x, 0.0), axis=-1)


@process(
    reads=("phen", "growth.leafno", "growth.xn", "growth.sump", "growth.pltpop"),
    writes=("phen", "growth.pltpop"),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_PHENOL.for (BSD-3)",
    fortran_name="MZ_PHENOL",
)
def ceres_phenology(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """One day of ``MZ_PHENOL`` (``DYNAMIC = INTEGR``) for every crop.

    ``MZ_CERES`` calls it on the sowing day and on every later day (``YRDOY == YRPLT`` or
    ``ISTAGE != 7``). Germination waits for moist soil in the seed layer when the water balance
    is on; the crop fails (stage 6, population 0) after ``DSGT`` dry days or when the emergence
    requirement ``P9`` exceeds ``DGET``. Grains per plant ``GPP`` and ears ``EARS`` are set at the
    beginning of effective grain filling from the stage-4 assimilation ``SUMP``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_PHENOL.for, DYNAMIC = INTEGR (BSD-3).
    """
    ph = state.phen
    g = state.growth
    cul = params.cultivar
    spe = params.species
    soil = params.soil
    f = forcing_t
    yrdoy = jnp.asarray(f.yrdoy)
    s = ph.istage
    active = (yrdoy == params.yrplt) | (s != 7)

    dtt_raw = thermal_time(f.tmax, f.tmin, f.srad, f.dayl, f.snow, g.leafno, s, cul)
    dtt0 = jnp.where(active, dtt_raw, ph.dtt)
    sumdtt = jnp.where(active, ph.sumdtt + dtt_raw, ph.sumdtt)
    cumdtt = jnp.where(active, ph.cumdtt + dtt_raw, ph.cumdtt)

    st = [active & (s == k) for k in range(11)]  # st[k]: crop in stage k at the start of the day
    one = jnp.ones_like(sumdtt)

    # start from "nothing changes" and overwrite the fields of each stage block (masks disjoint)
    istage = s
    ndas = ph.ndas
    xstage = ph.xstage
    sind = ph.sind
    p3 = ph.p3
    p9 = ph.p9
    tlno = ph.tlno
    xnti = ph.xnti
    gpp = ph.gpp
    ears = ph.ears
    idurp = ph.idurp
    seed_layer = ph.seed_layer
    mdate = ph.mdate
    status = ph.crop_status
    pltpop = g.pltpop
    dtt = dtt0
    ended = jnp.zeros_like(active)  # the block's stage ended today (STGDOY(old stage) = YRDOY)

    # ---- stage 7: sowing ------------------------------------------------------------------
    m = st[7]
    istage = jnp.where(m, 8, istage)
    ndas = jnp.where(m, 0.0, ndas)
    sumdtt = jnp.where(m, 0.0, sumdtt)
    ended = ended | m
    wat = params.iswwat  # static switch: without a water balance the seed layer is never needed
    seed_layer = jnp.where(m & wat, _layer_of_depth(params.sdepth * one, soil.dlayr), seed_layer)

    # ---- stage 8: germination -------------------------------------------------------------
    m = st[8]
    l0 = seed_layer
    l1 = jnp.minimum(l0 + 1, soil.dlayr.shape[-1] - 1)
    sw0 = _pick(f.sw, l0)
    ll0 = _pick(soil.ll, l0)
    dry = (sw0 <= ll0) & wat  # the seed-layer check only runs with the water balance on
    swsd = (sw0 - ll0) * 0.65 + (_pick(f.sw, l1) - _pick(soil.ll, l1)) * 0.35
    ndas = jnp.where(m & dry, ndas + 1.0, ndas)
    fail = m & dry & (ndas >= spe.dsgt)
    istage = jnp.where(fail, 6, istage)
    pltpop = jnp.where(fail, 0.0, pltpop)
    gpp = jnp.where(fail, 1.0, gpp)
    mdate = jnp.where(fail, yrdoy, mdate)
    status = jnp.where(fail, 12, status)
    germinate = m & ~fail & ~(dry & (swsd < spe.swcg))
    istage = jnp.where(germinate, 9, istage)
    cumdtt = jnp.where(germinate, 0.0, cumdtt)
    sumdtt = jnp.where(germinate, 0.0, sumdtt)
    p9 = jnp.where(germinate, 45.0 + cul.gdde * params.sdepth, p9)
    ended = ended | germinate

    # ---- stage 9: emergence ---------------------------------------------------------------
    m = st[9]
    ndas = jnp.where(m, ndas + 1.0, ndas)
    reached = m & (sumdtt >= p9)
    fail = reached & (p9 > spe.dget)
    istage = jnp.where(fail, 6, istage)
    pltpop = jnp.where(fail, 0.0, pltpop)
    gpp = jnp.where(fail, 1.0, gpp)
    mdate = jnp.where(fail, yrdoy, mdate)
    status = jnp.where(fail, 13, status)
    emerge = reached & ~fail
    istage = jnp.where(emerge, 1, istage)
    sumdtt = jnp.where(emerge, sumdtt - p9, sumdtt)
    tlno = jnp.where(emerge, 30.0, tlno)
    ended = ended | emerge

    # ---- stage 1: emergence -> end of juvenile phase ----------------------------------------
    m = st[1]
    ndas = jnp.where(m, ndas + 1.0, ndas)
    xstage = jnp.where(m, safe_div(sumdtt, cul.p1), xstage)
    done = m & (sumdtt >= cul.p1)
    istage = jnp.where(done, 2, istage)
    sind_in = jnp.where(done, 0.0, sind)
    ended = ended | done

    # ---- stage 2: end of juvenile -> tassel initiation (photoperiod) -------------------------
    m = st[2]
    ndas = jnp.where(m, ndas + 1.0, ndas)
    xstage = jnp.where(m, 1.0 + 0.5 * sind, xstage)
    twilen = jnp.asarray(f.twilen)
    ratein = jnp.where(
        twilen > cul.p2o,
        1.0 / jnp.maximum(cul.djti + cul.p2 * (twilen - cul.p2o), 1e-6),
        safe_div(1.0, cul.djti),
    )
    sind_new = jnp.where(m, sind + ratein, sind_in)
    done = m & (sind_new >= 1.0)
    tlno_ti = sumdtt / jnp.maximum(cul.phint * 0.5, 1e-6) + 5.0
    istage = jnp.where(done, 3, istage)
    tlno = jnp.where(done, tlno_ti, tlno)
    p3 = jnp.where(done, (tlno_ti + 0.5) * cul.phint - sumdtt, p3)
    xnti = jnp.where(done, g.xn, xnti)
    sumdtt = jnp.where(done, 0.0, sumdtt)
    sind = sind_new
    ended = ended | done

    # ---- stage 3: tassel initiation -> silking (end of leaf growth) --------------------------
    m = st[3]
    ndas = jnp.where(m, ndas + 1.0, ndas)
    xstage = jnp.where(m, 1.5 + 3.0 * safe_div(sumdtt, p3), xstage)
    done = m & (sumdtt >= p3)
    istage = jnp.where(done, 4, istage)
    sumdtt = jnp.where(done, sumdtt - p3, sumdtt)
    idurp = jnp.where(done, 0, idurp)
    ended = ended | done

    # ---- stage 4: silking -> beginning of effective grain filling ----------------------------
    m = st[4]
    ndas = jnp.where(m, ndas + 1.0, ndas)
    idurp = jnp.where(m, idurp + 1, idurp)
    xstage = jnp.where(m, 4.5 + 5.5 * safe_div(sumdtt, cul.p5 * 0.95), xstage)
    done = m & (sumdtt >= cul.dsgft)
    psker = safe_div(g.sump * 1000.0, idurp.astype(sumdtt.dtype)) * 3.4 / 5.0
    gpp_new = jnp.maximum(jnp.clip(cul.g2 * psker / 7200.0 + 50.0, 0.0, cul.g2), 51.0)
    g2_15 = jnp.maximum(cul.g2 * 0.15, 1e-6)
    g2_50 = jnp.maximum(cul.g2 * 0.50, 1e-6)
    ratio15 = jnp.maximum(gpp_new / g2_15, 1e-6)
    ratio50 = jnp.maximum(gpp_new / g2_50, 1e-6)
    barfac = 0.0085 * (1.0 - safe_div(gpp_new, cul.g2)) * jnp.maximum(pltpop, 0.0) ** 1.5
    ears_new = jnp.where(
        gpp_new < cul.g2 * 0.15,
        pltpop * ratio15**0.33,
        jnp.where((pltpop > 12.0) & (gpp_new < cul.g2 * 0.5), pltpop * ratio50**barfac, pltpop),
    )
    gpp = jnp.where(done, gpp_new, gpp)
    ears = jnp.where(done, jnp.maximum(ears_new, 0.0), ears)
    istage = jnp.where(done, 5, istage)
    ended = ended | done

    # ---- stage 5: effective grain filling -----------------------------------------------------
    m = st[5]
    ndas = jnp.where(m, ndas + 1.0, ndas)
    xstage = jnp.where(m, 4.5 + 5.5 * safe_div(sumdtt, cul.p5), xstage)
    done = m & (sumdtt >= cul.p5 * 0.95)
    istage = jnp.where(done, 6, istage)
    ended = ended | done

    # ---- stage 6: end of effective grain filling -> physiological maturity --------------------
    m = st[6]
    sumdtt = jnp.where(m & (dtt < 2.0), cul.p5 * one, sumdtt)
    done = m & (sumdtt >= cul.p5)
    istage = jnp.where(done, 10, istage)
    mdate = jnp.where(done, yrdoy, mdate)
    status = jnp.where(done, 1, status)
    cumdtt = jnp.where(done, 0.0, cumdtt)
    dtt = jnp.where(done, 0.0, dtt)
    gpp = jnp.where(done & (pltpop != 0.0) & (gpp <= 0.0), 1.0, gpp)
    ended = ended | done

    # STGDOY(old stage) = YRDOY for the stage that ended (failures set no stage date)
    col = (jnp.arange(ph.stgdoy.shape[-1]) + 1) == s[..., None]
    stgdoy = jnp.where(col & ended[..., None], yrdoy, ph.stgdoy)

    new_phen = eqx.tree_at(
        lambda p: (
            p.istage,
            p.sumdtt,
            p.cumdtt,
            p.dtt,
            p.ndas,
            p.xstage,
            p.sind,
            p.p3,
            p.p9,
            p.tlno,
            p.xnti,
            p.gpp,
            p.ears,
            p.idurp,
            p.seed_layer,
            p.stgdoy,
            p.mdate,
            p.crop_status,
        ),
        ph,
        (
            istage.astype(ph.istage.dtype),
            sumdtt,
            cumdtt,
            dtt,
            ndas,
            xstage,
            sind,
            p3,
            p9,
            tlno,
            xnti,
            gpp,
            ears,
            idurp.astype(ph.idurp.dtype),
            seed_layer.astype(ph.seed_layer.dtype),
            stgdoy.astype(ph.stgdoy.dtype),
            mdate.astype(ph.mdate.dtype),
            status.astype(ph.crop_status.dtype),
        ),
    )
    return eqx.tree_at(lambda st_: (st_.phen, st_.growth.pltpop), state, (new_phen, pltpop))
