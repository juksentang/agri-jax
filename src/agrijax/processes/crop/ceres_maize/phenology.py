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

Structure: :func:`thermal_time` (with :func:`growing_point_thermal_time` and
:func:`hourly_thermal_time`), then one kernel per stage block (:func:`sowing_block`,
:func:`germination_block`, :func:`emergence_block`, :func:`juvenile_block`,
:func:`floral_induction_block` with :func:`photoperiod_rate` and :func:`leaf_number_at_ti`,
:func:`tassel_silk_block`, :func:`silk_efg_block` with :func:`grain_number`,
:func:`grain_fill_block`, :func:`maturity_block`) applied in the Fortran order to a carried
:class:`PhenDay`. The hard-coded numbers are the fields of
:class:`~agrijax.processes.crop.ceres_maize.coefficients.PhenolCoefficients`
(``params.coef().phenol``); the ``XSTAGE`` scale (``1 + 0.5 SIND``, ``1.5 + 3 SUMDTT / P3``,
``4.5 + 5.5 SUMDTT / P5``) stays literal since only the nitrogen module reads it.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_PHENOL.for`` (BSD-3, Copyright 1998-2026 DSSAT
Foundation, University of Florida, International Fertilizer Development Center); thermal time
after J. T. Ritchie (CIMMYT 1998) as coded there; Jones & Kiniry (1986) CERES-Maize.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.process import process

from ._util import safe_div
from .coefficients import DSSAT_COEFFICIENTS, PhenolCoefficients
from .state import CeresCultivar, CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = [
    "PhenDay",
    "ceres_phenology",
    "emergence_block",
    "floral_induction_block",
    "germination_block",
    "grain_fill_block",
    "grain_number",
    "growing_point_thermal_time",
    "hourly_thermal_time",
    "juvenile_block",
    "leaf_number_at_ti",
    "maturity_block",
    "photoperiod_rate",
    "silk_efg_block",
    "sowing_block",
    "tassel_silk_block",
    "thermal_time",
]

_HOURS = jnp.arange(1, 25)


def growing_point_thermal_time(
    tmax: Array,
    tmin: Array,
    srad: ArrayLike,
    dayl: ArrayLike,
    snow: ArrayLike,
    tbase: ArrayLike,
    dopt: Array,
    c: PhenolCoefficients,
) -> Array:
    """Thermal time [degC d] of a growing point below ground (up to 10 leaves).

    Under snow the crown temperature ``2 + T (0.4 + 0.0018 (min(SNOW, 15) - 15)^2)`` of a
    sub-zero ``T`` is averaged; without snow the day soil temperature
    ``TDSOIL = ACOEF TMAX + (1 - ACOEF) TMIN`` (``ACOEF = 0.01061 SRAD + 0.5902``, capped at
    ``DOPT``) and the night one ``TNSOIL = 0.36354 TMAX + 0.63646 TMIN`` (at least ``TBASE``) give
    ``DTT``, 0 when ``TDSOIL < TBASE`` and at most ``DOPT - TBASE``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ELSEIF (LEAFNO.LE.10)`` branch and the
    snow crown temperature (TEMPCN, TEMPCX).
    """
    xs = jnp.minimum(jnp.asarray(snow), c.snow_max)
    snowfac = c.snowfac_base + c.snowfac_coef * (xs - c.snow_max) ** 2
    tempcn = jnp.where(tmin < 0.0, c.crown_offset + tmin * snowfac, tmin)
    tempcx = jnp.where(tmax < 0.0, c.crown_offset + tmax * snowfac, tmax)

    acoef = c.acoef_srad * jnp.asarray(srad) + c.acoef_base
    tdsoil = acoef * tmax + (1.0 - acoef) * tmin
    tnsoil = jnp.maximum(c.tnsoil_tmax_w * tmax + c.tnsoil_tmin_w * tmin, tbase)
    tdsoil_c = jnp.minimum(tdsoil, dopt)
    dl = jnp.asarray(dayl)
    tmsoil = tdsoil_c * (dl / 24.0) + tnsoil * ((24.0 - dl) / 24.0)
    dtt_soil = jnp.where(tmsoil < tbase, (tbase + tdsoil_c) / 2.0 - tbase, (tnsoil + tdsoil_c) / 2.0 - tbase)
    dtt_soil = jnp.where(tdsoil < tbase, 0.0, jnp.minimum(dtt_soil, dopt - tbase))
    dtt_snow = (tempcn + tempcx) / 2.0 - tbase
    return jnp.where(xs > 0.0, dtt_snow, dtt_soil)


def hourly_thermal_time(
    tmax: Array, tmin: Array, tbase: ArrayLike, dopt: Array, c: PhenolCoefficients
) -> Array:
    """Thermal time [degC d] from 24 hourly temperatures ``(TMAX + TMIN) / 2 + (TMAX - TMIN) / 2
    sin(3.14 / 12 I)``, ``I = 1..24``, each clipped to ``[TBASE, DOPT]``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ELSEIF (TMIN .LT. TBASE .OR. TMAX .GT. DOPT)``
    branch.
    """
    th = (tmax + tmin)[..., None] / 2.0 + (tmax - tmin)[..., None] / 2.0 * jnp.sin(
        c.hourly_pi / 12.0 * _HOURS
    )
    th = jnp.minimum(jnp.maximum(th, jnp.asarray(tbase)[..., None]), jnp.asarray(dopt)[..., None])
    return jnp.sum((th - jnp.asarray(tbase)[..., None]) / 24.0, axis=-1)


def thermal_time(
    tmax: ArrayLike,
    tmin: ArrayLike,
    srad: ArrayLike,
    dayl: ArrayLike,
    snow: ArrayLike,
    leafno: ArrayLike,
    istage: ArrayLike,
    cul: CeresCultivar,
    c: PhenolCoefficients = DSSAT_COEFFICIENTS.phenol,
) -> Array:
    """Daily thermal time ``DTT`` [degC d] of ``MZ_PHENOL``.

    Branches, in the Fortran order: ``TMAX < TBASE`` -> 0; ``TMIN > DOPT`` -> ``DOPT - TBASE``;
    up to 10 leaves the growing point is below ground and the soil (or snow-covered crown)
    temperature is used; otherwise the 24-point sine interpolation when a limit is crossed, or the
    daily mean minus ``TBASE``. ``DOPT`` is ``ROPT`` in stages 4-6, else ``TOPT``. ``c`` holds the
    hard-coded coefficients (default: the DSSAT values).

    Source: DSSAT-CSM MZ_PHENOL.for (DYNAMIC = INTEGR, thermal time block).
    """
    tmax = jnp.asarray(tmax)
    tmin = jnp.asarray(tmin)
    istage = jnp.asarray(istage)
    tbase = cul.tbase
    dopt = jnp.where((istage > 3) & (istage <= 6), cul.ropt, cul.topt)
    dtt_ground = growing_point_thermal_time(tmax, tmin, srad, dayl, snow, tbase, dopt, c)
    dtt_hourly = hourly_thermal_time(tmax, tmin, tbase, dopt, c)
    dtt = jnp.select(
        [
            tmax < tbase,
            tmin > dopt,
            jnp.asarray(leafno) <= c.leafno_soil,
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


class PhenDay(NamedTuple):
    """The phenology fields a stage block may change, carried through the blocks in the Fortran
    order (a block changes only the crops whose day-start stage is its own)."""

    istage: Array
    sumdtt: Array
    cumdtt: Array
    dtt: Array
    ndas: Array
    xstage: Array
    sind: Array
    p3: Array
    p9: Array
    tlno: Array
    xnti: Array
    gpp: Array
    ears: Array
    idurp: Array
    seed_layer: Array
    mdate: Array
    status: Array  # CropStatus
    pltpop: Array  # growth.pltpop (0 after a germination / emergence failure)
    ended: Array  # the crop's day-start stage ended today (STGDOY(old stage) = YRDOY)


def _count_day(v: PhenDay, m: Array) -> PhenDay:
    """``NDAS = NDAS + 1`` in the stage block ``m``."""
    return v._replace(ndas=jnp.where(m, v.ndas + 1.0, v.ndas))


def _advance(v: PhenDay, done: Array, new_stage: int) -> PhenDay:
    """``STGDOY(ISTAGE) = YRDOY; ISTAGE = new_stage`` where ``done``."""
    return v._replace(istage=jnp.where(done, new_stage, v.istage), ended=v.ended | done)


def _seed_failure(v: PhenDay, fail: Array, yrdoy: Array, status: int) -> PhenDay:
    """Germination / emergence failure: stage 6, no plants, ``GPP = 1``, ``MDATE = YRDOY``."""
    return v._replace(
        istage=jnp.where(fail, 6, v.istage),
        pltpop=jnp.where(fail, 0.0, v.pltpop),
        gpp=jnp.where(fail, 1.0, v.gpp),
        mdate=jnp.where(fail, yrdoy, v.mdate),
        status=jnp.where(fail, status, v.status),
    )


def sowing_block(v: PhenDay, m: Array, params: CeresMaizeParams) -> PhenDay:
    """Stage 7 (sowing day): to stage 8; with the water balance on, find the seed layer ``L0``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 7`` block.
    """
    wat = params.iswwat  # static switch: without a water balance the seed layer is never needed
    one = jnp.ones_like(v.sumdtt)
    v = _advance(v, m, 8)._replace(ndas=jnp.where(m, 0.0, v.ndas), sumdtt=jnp.where(m, 0.0, v.sumdtt))
    layer = _layer_of_depth(params.sdepth * one, params.soil.dlayr)
    return v._replace(seed_layer=jnp.where(m & wat, layer, v.seed_layer))


def germination_block(
    v: PhenDay, m: Array, params: CeresMaizeParams, sw: ArrayLike, yrdoy: Array, c: PhenolCoefficients
) -> PhenDay:
    """Stage 8: germinate unless the seed layer is dry (``SW <= LL``) and the weighted available
    water ``0.65 (SW - LL)(L0) + 0.35 (SW - LL)(L0 + 1)`` is below ``SWCG``; fail after ``DSGT``
    dry days. On germination ``P9 = 45 + GDDE SDEPTH``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 8`` block.
    """
    soil, spe = params.soil, params.species
    sw = jnp.asarray(sw)
    l0 = v.seed_layer
    l1 = jnp.minimum(l0 + 1, soil.dlayr.shape[-1] - 1)
    sw0 = _pick(sw, l0)
    ll0 = _pick(soil.ll, l0)
    dry = (sw0 <= ll0) & params.iswwat  # the seed-layer check only runs with the water balance on
    swsd = (sw0 - ll0) * c.swsd_w_seed + (_pick(sw, l1) - _pick(soil.ll, l1)) * c.swsd_w_below
    v = v._replace(ndas=jnp.where(m & dry, v.ndas + 1.0, v.ndas))
    fail = m & dry & (v.ndas >= spe.dsgt)
    v = _seed_failure(v, fail, yrdoy, 12)
    germinate = m & ~fail & ~(dry & (swsd < spe.swcg))
    v = _advance(v, germinate, 9)
    return v._replace(
        cumdtt=jnp.where(germinate, 0.0, v.cumdtt),
        sumdtt=jnp.where(germinate, 0.0, v.sumdtt),
        p9=jnp.where(germinate, c.p9_base + params.cultivar.gdde * params.sdepth, v.p9),
    )


def emergence_block(
    v: PhenDay, m: Array, params: CeresMaizeParams, yrdoy: Array, c: PhenolCoefficients
) -> PhenDay:
    """Stage 9: emerge when ``SUMDTT >= P9`` (fail if ``P9 > DGET``); ``TLNO = 30`` provisionally.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 9`` block.
    """
    v = _count_day(v, m)
    reached = m & (v.sumdtt >= v.p9)
    fail = reached & (v.p9 > params.species.dget)
    v = _seed_failure(v, fail, yrdoy, 13)
    emerge = reached & ~fail
    v = _advance(v, emerge, 1)
    return v._replace(
        sumdtt=jnp.where(emerge, v.sumdtt - v.p9, v.sumdtt), tlno=jnp.where(emerge, c.tlno_emergence, v.tlno)
    )


def juvenile_block(v: PhenDay, m: Array, cul: CeresCultivar) -> PhenDay:
    """Stage 1 (emergence -> end of juvenile): ends when ``SUMDTT >= P1``; ``SIND = 0``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 1`` block.
    """
    v = _count_day(v, m)
    v = v._replace(xstage=jnp.where(m, safe_div(v.sumdtt, cul.p1), v.xstage))
    done = m & (v.sumdtt >= cul.p1)
    return _advance(v, done, 2)._replace(sind=jnp.where(done, 0.0, v.sind))


def photoperiod_rate(twilen: ArrayLike, cul: CeresCultivar) -> Array:
    """Daily photoperiod induction ``RATEIN = 1 / (DJTI + P2 (TWILEN - P2O))`` above the
    critical twilight daylength ``P2O``, else ``1 / DJTI`` [d-1].

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 2`` block (RATEIN).
    """
    twilen = jnp.asarray(twilen)
    return jnp.where(
        twilen > cul.p2o,
        1.0 / jnp.maximum(cul.djti + cul.p2 * (twilen - cul.p2o), 1e-6),
        safe_div(1.0, cul.djti),
    )


def leaf_number_at_ti(sumdtt: Array, phint: ArrayLike, c: PhenolCoefficients) -> tuple[Array, Array]:
    """``(TLNO, P3)`` at tassel initiation: ``TLNO = SUMDTT / (0.5 PHINT) + 5`` and
    ``P3 = (TLNO + 0.5) PHINT - SUMDTT`` [degC d].

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, end of the ``ISTAGE .EQ. 2`` block.
    """
    tlno = sumdtt / jnp.maximum(phint * c.tlno_phint_frac, 1e-6) + c.tlno_offset
    return tlno, (tlno + c.p3_leaf_offset) * phint - sumdtt


def floral_induction_block(
    v: PhenDay, m: Array, cul: CeresCultivar, twilen: ArrayLike, xn: Array, c: PhenolCoefficients
) -> PhenDay:
    """Stage 2 (end of juvenile -> tassel initiation): sum ``RATEIN`` until ``SIND >= 1``; then
    total leaf number, ``P3`` and ``XNTI = XN``, and ``SUMDTT = 0``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 2`` block.
    """
    v = _count_day(v, m)
    v = v._replace(xstage=jnp.where(m, 1.0 + 0.5 * v.sind, v.xstage))
    sind = jnp.where(m, v.sind + photoperiod_rate(twilen, cul), v.sind)
    done = m & (sind >= 1.0)
    tlno_ti, p3_ti = leaf_number_at_ti(v.sumdtt, cul.phint, c)
    return _advance(v, done, 3)._replace(
        tlno=jnp.where(done, tlno_ti, v.tlno),
        p3=jnp.where(done, p3_ti, v.p3),
        xnti=jnp.where(done, xn, v.xnti),
        sumdtt=jnp.where(done, 0.0, v.sumdtt),
        sind=sind,
    )


def tassel_silk_block(v: PhenDay, m: Array) -> PhenDay:
    """Stage 3 (tassel initiation -> silking): ends when ``SUMDTT >= P3``; ``IDURP = 0``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 3`` block.
    """
    v = _count_day(v, m)
    v = v._replace(xstage=jnp.where(m, 1.5 + 3.0 * safe_div(v.sumdtt, v.p3), v.xstage))
    done = m & (v.sumdtt >= v.p3)
    return _advance(v, done, 4)._replace(
        sumdtt=jnp.where(done, v.sumdtt - v.p3, v.sumdtt), idurp=jnp.where(done, 0, v.idurp)
    )


def grain_number(
    sump: Array, idurp: Array, g2: ArrayLike, pltpop: Array, c: PhenolCoefficients
) -> tuple[Array, Array]:
    """``(GPP, EARS)`` at the beginning of effective grain filling.

    ``PSKER = SUMP 1000 / IDURP x 3.4 / 5`` (mean stage-4 assimilation, mg plant-1 d-1);
    ``GPP = max(min(G2 PSKER / 7200 + 50, G2), 0, 51)``; ears drop to
    ``PLTPOP (GPP / 0.15 G2)^0.33`` below ``0.15 G2`` kernels and, above 12 plants m-2 and below
    ``0.5 G2``, to ``PLTPOP (GPP / 0.5 G2)^BARFAC`` with ``BARFAC = 0.0085 (1 - GPP / G2) PLTPOP^1.5``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, end of the ``ISTAGE .EQ. 4`` block.
    """
    psker = safe_div(sump * 1000.0, idurp.astype(sump.dtype)) * c.psker_a / c.psker_b
    gpp = jnp.maximum(jnp.clip(g2 * psker / c.gpp_psker + c.gpp_offset, 0.0, g2), c.gpp_min)
    g2_low = jnp.maximum(g2 * c.ears_low_frac, 1e-6)
    g2_barren = jnp.maximum(g2 * c.barren_gpp_frac, 1e-6)
    ratio_low = jnp.maximum(gpp / g2_low, 1e-6)
    ratio_barren = jnp.maximum(gpp / g2_barren, 1e-6)
    barfac = c.barfac_coef * (1.0 - safe_div(gpp, g2)) * jnp.maximum(pltpop, 0.0) ** c.barfac_exp
    ears = jnp.where(
        gpp < g2 * c.ears_low_frac,
        pltpop * ratio_low**c.ears_low_exp,
        jnp.where(
            (pltpop > c.barren_pltpop) & (gpp < g2 * c.barren_gpp_frac), pltpop * ratio_barren**barfac, pltpop
        ),
    )
    return gpp, jnp.maximum(ears, 0.0)


def silk_efg_block(v: PhenDay, m: Array, cul: CeresCultivar, sump: Array, c: PhenolCoefficients) -> PhenDay:
    """Stage 4 (silking -> beginning of effective grain filling): count ``IDURP``; after
    ``DSGFT`` set kernel and ear numbers (:func:`grain_number`).

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 4`` block.
    """
    v = _count_day(v, m)
    v = v._replace(idurp=jnp.where(m, v.idurp + 1, v.idurp))
    v = v._replace(xstage=jnp.where(m, 4.5 + 5.5 * safe_div(v.sumdtt, cul.p5 * c.efg_end_frac), v.xstage))
    done = m & (v.sumdtt >= cul.dsgft)
    gpp, ears = grain_number(sump, v.idurp, cul.g2, v.pltpop, c)
    v = v._replace(gpp=jnp.where(done, gpp, v.gpp), ears=jnp.where(done, ears, v.ears))
    return _advance(v, done, 5)


def grain_fill_block(v: PhenDay, m: Array, cul: CeresCultivar, c: PhenolCoefficients) -> PhenDay:
    """Stage 5 (effective grain filling): ends at ``SUMDTT >= 0.95 P5``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 5`` block.
    """
    v = _count_day(v, m)
    v = v._replace(xstage=jnp.where(m, 4.5 + 5.5 * safe_div(v.sumdtt, cul.p5), v.xstage))
    return _advance(v, m & (v.sumdtt >= cul.p5 * c.efg_end_frac), 6)


def maturity_block(v: PhenDay, m: Array, cul: CeresCultivar, yrdoy: Array, c: PhenolCoefficients) -> PhenDay:
    """Stage 6 (end of effective grain filling -> physiological maturity) at ``SUMDTT >= P5``
    (at once when ``DTT < 2``): stage 10, ``MDATE``, status 1, ``GPP >= 1`` for a live crop.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 6`` block.
    """
    sumdtt = jnp.where(m & (v.dtt < c.dtt_maturity), cul.p5 * jnp.ones_like(v.sumdtt), v.sumdtt)
    done = m & (sumdtt >= cul.p5)
    v = _advance(v, done, 10)
    return v._replace(
        sumdtt=sumdtt,
        mdate=jnp.where(done, yrdoy, v.mdate),
        status=jnp.where(done, 1, v.status),
        cumdtt=jnp.where(done, 0.0, v.cumdtt),
        dtt=jnp.where(done, 0.0, v.dtt),
        gpp=jnp.where(done & (v.pltpop != 0.0) & (v.gpp <= 0.0), 1.0, v.gpp),
    )


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
    beginning of effective grain filling from the stage-4 assimilation ``SUMP``. Each stage
    block is a kernel (:func:`sowing_block` ... :func:`maturity_block`) applied in the Fortran
    order to the crops in that stage; the hard-coded coefficients come from
    ``params.coef().phenol``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_PHENOL.for, DYNAMIC = INTEGR (BSD-3).
    """
    ph = state.phen
    g = state.growth
    cul = params.cultivar
    c = params.coef().phenol
    f = forcing_t
    yrdoy = jnp.asarray(f.yrdoy)
    s = ph.istage
    active = (yrdoy == params.yrplt) | (s != 7)

    dtt_raw = thermal_time(f.tmax, f.tmin, f.srad, f.dayl, f.snow, g.leafno, s, cul, c)
    # start from "nothing changes" and let each stage block change its crops (masks disjoint)
    v = PhenDay(
        istage=s,
        sumdtt=jnp.where(active, ph.sumdtt + dtt_raw, ph.sumdtt),
        cumdtt=jnp.where(active, ph.cumdtt + dtt_raw, ph.cumdtt),
        dtt=jnp.where(active, dtt_raw, ph.dtt),
        ndas=ph.ndas,
        xstage=ph.xstage,
        sind=ph.sind,
        p3=ph.p3,
        p9=ph.p9,
        tlno=ph.tlno,
        xnti=ph.xnti,
        gpp=ph.gpp,
        ears=ph.ears,
        idurp=ph.idurp,
        seed_layer=ph.seed_layer,
        mdate=ph.mdate,
        status=ph.crop_status,
        pltpop=g.pltpop,
        ended=jnp.zeros_like(active),
    )
    st = [active & (s == k) for k in range(11)]  # st[k]: crop in stage k at the start of the day
    v = sowing_block(v, st[7], params)
    v = germination_block(v, st[8], params, f.sw, yrdoy, c)
    v = emergence_block(v, st[9], params, yrdoy, c)
    v = juvenile_block(v, st[1], cul)
    v = floral_induction_block(v, st[2], cul, f.twilen, g.xn, c)
    v = tassel_silk_block(v, st[3])
    v = silk_efg_block(v, st[4], cul, g.sump, c)
    v = grain_fill_block(v, st[5], cul, c)
    v = maturity_block(v, st[6], cul, yrdoy, c)

    # STGDOY(old stage) = YRDOY for the stage that ended (failures set no stage date)
    col = (jnp.arange(ph.stgdoy.shape[-1]) + 1) == s[..., None]
    stgdoy = jnp.where(col & v.ended[..., None], yrdoy, ph.stgdoy)
    new_phen = ph.replace(
        istage=v.istage.astype(ph.istage.dtype),
        sumdtt=v.sumdtt,
        cumdtt=v.cumdtt,
        dtt=v.dtt,
        ndas=v.ndas,
        xstage=v.xstage,
        sind=v.sind,
        p3=v.p3,
        p9=v.p9,
        tlno=v.tlno,
        xnti=v.xnti,
        gpp=v.gpp,
        ears=v.ears,
        idurp=v.idurp.astype(ph.idurp.dtype),
        seed_layer=v.seed_layer.astype(ph.seed_layer.dtype),
        stgdoy=stgdoy.astype(ph.stgdoy.dtype),
        mdate=v.mdate.astype(ph.mdate.dtype),
        crop_status=v.status.astype(ph.crop_status.dtype),
    )
    return eqx.tree_at(lambda st_: (st_.phen, st_.growth.pltpop), state, (new_phen, v.pltpop))
