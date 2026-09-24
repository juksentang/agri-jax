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
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.process import process

from ._util import safe_div, trunc_st
from .coefficients import RootgrCoefficients
from .state import CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = [
    "ceres_roots",
    "emergence_rlv",
    "root_front_advance",
    "root_length_growth",
    "root_water_deficit",
    "waterlogging_survival",
]


def _last_true(mask: Array) -> Array:
    """0-based index of the last True along the last axis (0 when none)."""
    n = mask.shape[-1]
    idx = jnp.arange(n)
    return jnp.max(jnp.where(mask, idx, 0), axis=-1)


def _at(mask: Array, x: Array) -> Array:
    """``x`` at the (one) layer where ``mask`` is True, by masked sum over the layer axis."""
    return jnp.sum(jnp.where(mask, x, 0.0), axis=-1)


def emergence_rlv(rtdep: Array, pltpop: Array, dlayr: Array, c: RootgrCoefficients) -> Array:
    """Root length density [cm cm-3] on the emergence day: ``0.20 PLTPOP / DLAYR`` down to the
    first layer whose bottom is below the root front, that layer weighted by the rooted fraction
    ``1 - (CUMDEP - RTDEP) / DLAYR``, 0 below.

    Source: DSSAT-CSM v4.8.6.0 MZ_ROOTS.for (MZ_ROOTGR), INTEGR, ``IF(YRDOY.EQ.STGDOY(9))`` block.
    """
    n = dlayr.shape[-1]
    bottom = jnp.cumsum(dlayr)
    below = bottom > rtdep[..., None]
    l_em = jnp.where(jnp.any(below, axis=-1), jnp.argmax(below, axis=-1), n - 1)
    idx = jnp.arange(n)
    at_l = idx == l_em[..., None]
    rlv0 = jnp.where(
        idx <= l_em[..., None], c.rlv_emergence * pltpop[..., None] / jnp.maximum(dlayr, 1e-6), 0.0
    )
    frac = 1.0 - (_at(at_l, bottom) - rtdep) / jnp.maximum(_at(at_l, dlayr), 1e-6)
    return jnp.where(at_l, rlv0 * frac[..., None], rlv0)


def root_water_deficit(sw: Array, ll: Array, dul: Array, c: RootgrCoefficients) -> Array:
    """``SWDF``: 1, or ``4 (SW - LL) / (DUL - LL)`` (at least 0) when less than a quarter of the
    extractable water is left.

    Source: DSSAT-CSM v4.8.6.0 MZ_ROOTS.for (MZ_ROOTGR), INTEGR, SWDF in the layer loop.
    """
    esw = dul - ll
    avail = sw - ll
    return jnp.where(
        avail < c.swdf_esw_frac * esw, jnp.maximum(c.swdf_slope * safe_div(avail, esw), 0.0), 1.0
    )


def waterlogging_survival(sat: Array, sw: Array, pormin: ArrayLike, c: RootgrCoefficients) -> Array:
    """``RTSURV = min(1, 1 - 0.1 (1 - SWEXF))`` with ``SWEXF = min((SAT - SW) / PORMIN, 1)`` of the
    deepest rooted layer when its air-filled porosity is below ``PORMIN``.

    Source: DSSAT-CSM v4.8.6.0 MZ_ROOTS.for (MZ_ROOTGR), INTEGR, RTEXF / SWEXF / RTSURV.
    """
    air = sat - sw
    swexf = jnp.where(air < pormin, jnp.minimum(air / jnp.maximum(pormin, 1e-6), 1.0), 1.0)
    return jnp.minimum(1.0, 1.0 - c.rtexf * (1.0 - swexf))


def root_front_advance(
    rtdep: Array,
    dtt: Array,
    cumdtt: Array,
    shf: Array,
    swfac: Array,
    swdf: Array,
    depmax: Array,
    c: RootgrCoefficients,
) -> Array:
    """New rooting depth ``min(RTDEP + DTT r sqrt(SHF min(2 SWFAC, SWDF)), DEPMAX)`` [cm], with
    ``r = 0.1`` before 275 degC d since germination and 0.2 after; ``shf`` and ``swdf`` are those of
    the deepest rooted layer.

    Source: DSSAT-CSM v4.8.6.0 MZ_ROOTS.for (MZ_ROOTGR), INTEGR, RTDEP (JTR 6/17/94).
    """
    x = shf * jnp.minimum(swfac * c.rtdep_swfac_mult, swdf)
    root_x = jnp.where(x > 0.0, jnp.sqrt(jnp.where(x > 0.0, x, 1.0)), 0.0)
    rate = jnp.where(cumdtt < c.rtdep_cumdtt, c.rtdep_rate_early, c.rtdep_rate_late)
    return jnp.minimum(rtdep + dtt * rate * root_x, depmax)


def root_length_growth(
    rlv: Array, rldf: Array, rlnew: Array, dlayr: Array, rtsurv: Array, c: RootgrCoefficients
) -> tuple[Array, Array]:
    """``(RLV, spread)``: the new root length ``RLNEW`` spread over the layers in proportion to
    ``RLDF``, 0.5 % daily loss, waterlogging survival, truncation to 1e-3 (identity derivative)
    within ``[0, 4]``; ``spread`` is False when ``sum(RLDF) < 1e-5 RLNEW`` (no update).

    Source: DSSAT-CSM v4.8.6.0 MZ_ROOTS.for (MZ_ROOTGR), INTEGR, the ``TRLDF`` / ``RNLF`` loop.
    """
    trldf = jnp.sum(rldf, axis=-1)
    spread = trldf >= rlnew * c.trldf_min_frac
    rnlf = safe_div(rlnew, trldf)
    rlv_g = rlv + rldf * rnlf[..., None] / dlayr - c.rlv_decay * rlv
    rlv_g = rlv_g * rtsurv[..., None]
    return jnp.clip(trunc_st(rlv_g * 1000.0) / 1000.0, 0.0, c.rlv_max), spread


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
    key="crop/ceres_maize.roots@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        ("root length at emergence", "MZ_ROOTS.for (MZ_ROOTGR) INTEGR, IF (YRDOY.EQ.STGDOY(9)) block"),
        ("layer water deficit SWDF, waterlogging survival", "MZ_ROOTS.for (MZ_ROOTGR) INTEGR layer loop"),
        ("root front advance RTDEP", "MZ_ROOTS.for (MZ_ROOTGR) INTEGR; J. T. Ritchie (1994)"),
        ("new root length distribution TRLDF / RNLF", "MZ_ROOTS.for (MZ_ROOTGR) INTEGR (BSD-3)"),
    ),
    deviates=(
        (
            "nitrogen off: RNFAC = 1",
            "M2 scope; the nitrogen module is not built",
            "roots.py module docstring",
        ),
        (
            "DSSAT single precision (REAL*4) is not reproduced",
            "the kernels run in float64 (float32 with AGRI_JAX_X64=0)",
            "tests/integration/test_ceres_dssat.py tolerances",
        ),
    ),
)
def ceres_roots(state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing) -> CeresMaizeState:
    """One day of ``MZ_ROOTGR`` (``DYNAMIC = INTEGR``); runs from the sowing day on, only with
    the water balance on (``MZ_CERES`` skips it when ``ISWWAT = N``). The hard-coded
    coefficients come from ``params.coef().roots``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_ROOTS.for, DYNAMIC = INTEGR (BSD-3).
    """
    ph = state.phen
    r = state.roots
    soil = params.soil
    c = params.coef().roots
    yrdoy = jnp.asarray(forcing_t.yrdoy)
    on = (yrdoy >= params.yrplt) & params.iswwat
    s = ph.istage
    dtt = ph.dtt
    pltpop = state.growth.pltpop
    dlayr = soil.dlayr
    bottom = jnp.cumsum(dlayr)
    top = bottom - dlayr

    rtdep = jnp.where((s == 7) | (s == 8), params.sdepth, r.rtdep)
    rtdep = jnp.where(s == 9, rtdep + c.rtdep_emerg * dtt, rtdep)
    rlv = jnp.where((yrdoy == ph.stgdoy[..., 8])[..., None], emergence_rlv(rtdep, pltpop, dlayr, c), r.rlv)

    # ---- daily growth over the rooted layers (those the DO WHILE visits: CUMDEP < RTDEP before adding)
    grort = state.growth.grort
    rlnew = grort * params.species.rlwr * pltpop
    rooted = top < rtdep[..., None]
    at_l1 = jnp.arange(dlayr.shape[-1]) == _last_true(rooted)[..., None]
    sw = jnp.asarray(forcing_t.sw)
    swdf = root_water_deficit(sw, soil.ll, soil.dul, c)
    rldf = jnp.where(rooted, swdf * soil.shf * dlayr, 0.0)  # min(SWDF, RNFAC = 1) * SHF * DLAYR
    ones = jnp.ones_like(rldf)

    rtsurv = waterlogging_survival(
        _at(at_l1, soil.sat * ones), _at(at_l1, sw * ones), params.species.pormin, c
    )
    rtdep_g = root_front_advance(
        rtdep,
        dtt,
        ph.cumdtt,
        _at(at_l1, soil.shf * ones),
        state.stress.swfac,
        _at(at_l1, swdf * ones),
        bottom[..., -1],
        c,
    )
    frac_l1 = 1.0 - (_at(at_l1, bottom * ones) - rtdep_g) / jnp.maximum(_at(at_l1, dlayr * ones), 1e-6)
    rldf = jnp.where(at_l1, rldf * frac_l1[..., None], rldf)
    rlv_g, spread = root_length_growth(rlv, rldf, rlnew, dlayr, rtsurv, c)
    grows = grort > c.grort_min
    rlv = jnp.where(((grows & spread)[..., None] & rooted), rlv_g, rlv)
    rtdep = jnp.where(grows, rtdep_g, rtdep)

    rtdep = jnp.where(on, rtdep, r.rtdep)
    rlv = jnp.where(on[..., None], rlv, r.rlv)
    new = eqx.tree_at(lambda x_: (x_.rtdep, x_.rlv), r, (rtdep, rlv))
    return eqx.tree_at(lambda x_: x_.roots, state, new)
