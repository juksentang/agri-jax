"""CERES-Maize root depth and root length density (``MZ_ROOTGR``, nitrogen off).

Before emergence the root front sits at the sowing depth (stages 7, 8) and moves down with
``0.15 DTT`` in stage 9; on the emergence day ``0.2 PLTPOP`` cm of root per cm2 of soil is spread
over the layers down to the front. On every later day with root growth ``GRORT > 1e-4`` the new
length ``GRORT x RLWR x PLTPOP`` is distributed over the rooted layers in proportion to
``min(SWDF, RNFAC) x SHF x DLAYR`` (``RNFAC = 1`` with nitrogen off), the front deepens with
``DTT x (0.1 | 0.2) x sqrt(SHF x min(2 SWFAC, SWDF))`` of the deepest rooted layer, and the
existing density decays by 0.5 % per day, is reduced by waterlogging of the deepest rooted layer
and truncated to 1e-3 (value exact, identity derivative) within ``[0, 4]`` cm cm-3.

The Fortran loops over layers become masks over the layer axis: the rooted layers of the growth
loop are those whose top lies above the root front, the emergence loop fills layers down to the
first one whose bottom lies below the front.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_ROOTS.for`` (``MZ_ROOTGR``; BSD-3, Copyright
1998-2026 DSSAT Foundation, University of Florida, International Fertilizer Development Center);
Jones & Kiniry (1986) CERES-Maize, root growth after J. T. Ritchie (1994).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.process import process

from ._util import safe_div, trunc_st
from .state import CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = ["ceres_roots"]


def _last_true(mask: Array) -> Array:
    """0-based index of the last True along the last axis (0 when none)."""
    n = mask.shape[-1]
    idx = jnp.arange(n)
    return jnp.max(jnp.where(mask, idx, 0), axis=-1)


@process(
    reads=(
        "phen.istage",
        "phen.dtt",
        "phen.cumdtt",
        "phen.stgdoy",
        "growth.grort",
        "growth.pltpop",
        "stress.swfac",
        "roots",
    ),
    writes=("roots",),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_ROOTS.for (BSD-3)",
    fortran_name="MZ_ROOTGR",
)
def ceres_roots(state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing) -> CeresMaizeState:
    """One day of ``MZ_ROOTGR`` (``DYNAMIC = INTEGR``); runs from the sowing day on, only with
    the water balance on (``MZ_CERES`` skips it when ``ISWWAT = N``).

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_ROOTS.for, DYNAMIC = INTEGR (BSD-3).
    """
    ph = state.phen
    r = state.roots
    soil = params.soil
    spe = params.species
    yrdoy = jnp.asarray(forcing_t.yrdoy)
    on = (yrdoy >= params.yrplt) & params.iswwat
    s = ph.istage
    dtt = ph.dtt
    pltpop = state.growth.pltpop
    dlayr = soil.dlayr
    n = dlayr.shape[-1]
    bottom = jnp.cumsum(dlayr)
    top = bottom - dlayr
    depmax = bottom[..., -1]

    rtdep = jnp.where((s == 7) | (s == 8), params.sdepth, r.rtdep)
    rtdep = jnp.where(s == 9, rtdep + 0.15 * dtt, rtdep)

    # ---- emergence: initial root length density down to the front
    at_em = yrdoy == ph.stgdoy[..., 8]
    below = bottom > rtdep[..., None]
    l_em = jnp.where(jnp.any(below, axis=-1), jnp.argmax(below, axis=-1), n - 1)
    idx = jnp.arange(n)
    rlv0 = jnp.where(idx <= l_em[..., None], 0.20 * pltpop[..., None] / jnp.maximum(dlayr, 1e-6), 0.0)
    cum_em = jnp.sum(jnp.where(idx == l_em[..., None], bottom, 0.0), axis=-1)
    dl_em = jnp.sum(jnp.where(idx == l_em[..., None], dlayr, 0.0), axis=-1)
    frac_em = 1.0 - (cum_em - rtdep) / jnp.maximum(dl_em, 1e-6)
    rlv0 = jnp.where(idx == l_em[..., None], rlv0 * frac_em[..., None], rlv0)
    rlv = jnp.where(at_em[..., None], rlv0, r.rlv)

    # ---- daily growth
    grort = state.growth.grort
    rlnew = grort * spe.rlwr * pltpop
    rooted = top < rtdep[..., None]  # layers the DO WHILE visits (CUMDEP < RTDEP before adding)
    l1 = _last_true(rooted)
    at_l1 = idx == l1[..., None]
    sw = jnp.asarray(forcing_t.sw)
    esw = soil.dul - soil.ll
    avail = sw - soil.ll
    swdf = jnp.where(avail < 0.25 * esw, jnp.maximum(4.0 * safe_div(avail, esw), 0.0), 1.0)
    rldf = jnp.where(rooted, swdf * soil.shf * dlayr, 0.0)  # min(SWDF, RNFAC = 1) * SHF * DLAYR

    def at_last(x: Array) -> Array:
        return jnp.sum(jnp.where(at_l1, x, 0.0), axis=-1)

    sat_l1 = at_last(soil.sat * jnp.ones_like(rldf))
    sw_l1 = at_last(sw * jnp.ones_like(rldf))
    air = sat_l1 - sw_l1
    swexf = jnp.where(air < spe.pormin, jnp.minimum(air / jnp.maximum(spe.pormin, 1e-6), 1.0), 1.0)
    rtsurv = jnp.minimum(1.0, 1.0 - 0.1 * (1.0 - swexf))
    swdf_l1 = at_last(swdf * jnp.ones_like(rldf))
    shf_l1 = at_last(soil.shf * jnp.ones_like(rldf))
    x = shf_l1 * jnp.minimum(state.stress.swfac * 2.0, swdf_l1)
    root_x = jnp.where(x > 0.0, jnp.sqrt(jnp.where(x > 0.0, x, 1.0)), 0.0)
    rate = jnp.where(ph.cumdtt < 275.0, 0.1, 0.2)
    rtdep_g = jnp.minimum(rtdep + dtt * rate * root_x, depmax)
    cum_l1 = at_last(bottom * jnp.ones_like(rldf))
    dl_l1 = at_last(dlayr * jnp.ones_like(rldf))
    rldf = jnp.where(at_l1, rldf * (1.0 - (cum_l1 - rtdep_g) / jnp.maximum(dl_l1, 1e-6))[..., None], rldf)
    trldf = jnp.sum(rldf, axis=-1)
    grows = grort > 0.0001
    spread = grows & (trldf >= rlnew * 0.00001)
    rnlf = safe_div(rlnew, trldf)
    rlv_g = rlv + rldf * rnlf[..., None] / dlayr - 0.005 * rlv
    rlv_g = rlv_g * rtsurv[..., None]
    rlv_g = jnp.clip(trunc_st(rlv_g * 1000.0) / 1000.0, 0.0, 4.0)
    rlv = jnp.where((spread[..., None] & rooted), rlv_g, rlv)
    rtdep = jnp.where(grows, rtdep_g, rtdep)

    rtdep = jnp.where(on, rtdep, r.rtdep)
    rlv = jnp.where(on[..., None], rlv, r.rlv)
    new = eqx.tree_at(lambda x_: (x_.rtdep, x_.rlv), r, (rtdep, rlv))
    return eqx.tree_at(lambda x_: x_.roots, state, new)
