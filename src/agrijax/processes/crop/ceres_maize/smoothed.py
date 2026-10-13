"""CERES-Maize with smoothed phenology: logistic stage gates in place of the stage thresholds.

A non-faithful variant for the gradient experiments, registered next to the faithful processes as
``crop/ceres_maize.phenology@dssat-4.8.6.0:alt_smoothed`` and
``crop/ceres_maize.growth@dssat-4.8.6.0:alt_smoothed`` (an ``alt_`` variant: an alternative
formulation with declared deviations). The growth half consumes what the phenology half writes:
run them together, with :class:`~.smoothed_params.SoftPhenologyState` in ``CeresMaizeState.soft``.
The variant changes the forward model on purpose, so that each phenology coefficient (``P1``,
``P2``, ``P5``, ``PHINT``) has a continuous derivative; how far it departs from the reference is a
function of the gate scale
``CeresMaizeParams.smoothing.width`` (:class:`~.smoothed_params.SmoothingCoefficients`).

Construction (``s`` the gate scale, ``sig(x) = 1 / (1 + exp(-x / s))``, ``sp(x) = s log(1 + exp(x / s))``,
which tends to ``max(x, 0)``):

* **soft clocks** - ``tt_germ`` is the thermal time since germination (it is ``SUMDTT`` of stage 9);
  the thermal time since emergence is ``sp(tt_germ - P9)`` (``SUMDTT`` of stages 1-2), the photoperiod
  induction ``SIND`` sums ``RATEIN`` weighted by the soft end-of-juvenile passage, ``tt_pre_ti``
  sums the emergence clock's increments weighted by the soft probability of not yet having passed
  tassel initiation (it sets ``TLNO`` and ``P3``), ``tt_ti`` sums ``DTT`` weighted by that passage
  (``SUMDTT`` of stage 3) and the thermal time since silking is ``sp(tt_ti - P3)`` (``SUMDTT`` of
  stages 4-6);
* **passage chain** - the soft probability of having passed boundary ``k`` is the product of the
  gates up to ``k``: emergence ``sig(tt_germ - P9)``; end of juvenile ``sig(tt_em - P1)``; tassel
  initiation ``sig((SIND - 1) sind_tt)``; silking ``sig(tt_ti - P3)``; ``DSGFT``, ``0.95 P5`` and
  ``P5`` on the silking clock (the emergence factor is kept out of the product of the later ones,
  which start from the hard emergence of growth). The expected passage day of a boundary is the
  sum over days of ``1 - passage``;
* **hard stage machine** - the integer ``ISTAGE`` (stage dates, the stage-date events of growth)
  advances when a soft clock crosses its threshold (gate 0.5), with the faithful stage blocks
  (:mod:`.phenology`) run on the soft clocks; the grain number at the beginning of effective grain
  filling comes from the soft stage-4 assimilation and duration (``sump``, ``dur4``);
* **growth blend** - growth (:func:`ceres_growth_smoothed`) blends the five stage blocks of
  ``MZ_GROSUB`` with the soft stage occupancies (differences of successive passages) instead of
  selecting the block of the hard stage, each block on its own soft clock; the occupancy of stage 5
  that remains after the hard end of effective grain filling still fills grain on the stage-6 days;
* **hard ends** - an early maturity of slow grain filling (growth's ``SUMDTT = P5``, or the stage-6
  ``DTT < 2`` rule) and a crop failure (no germination, no emergence, cold in stages 1-5 or drought
  before silking: stage 6, ``MDATE = YRDOY``) force the soft passages: a failure passes every boundary up
  to the end of effective grain filling on the failure day (no stage occupancy is left to grow), the
  soft clocks of a failed crop stop, its stage-6 clock is the ``SUMDTT`` it failed with carried on (as
  in the reference), and maturity is passed on the hard maturity day; a forced passage stays at 1.

As ``s -> 0`` every gate becomes the reference's step and the variant reproduces the faithful
model to rounding (tested in ``tests/unit/test_ceres_smoothed.py``, including an early maturity, a
drought failure before silking and cold failures in stages 4 and 5) as long as ``YRDOY`` strictly
increases from day to day.
On days that repeat a date (a driver padding a season with its last day) the
two differ even in the limit: the faithful growth resets ``SUMP`` on every day whose ``YRDOY`` equals
the silking date ``STGDOY(3)``, while the soft stage-4 sums (``sump``, ``dur4``) are reset only at the
hard tassel initiation, so a crop that silks on a repeated date gets a different grain number (DSSAT
example GAGR0201, padded growth-chamber seasons). Germination (soil water), the germination and
emergence failures, the cold and drought failures and the early maturity of slow grain filling stay
hard events. (In an earlier version a failure did not force the passages: after a cold failure
in stage 4 or 5 the soft occupancies of stages 4-5 kept filling grain on the stage-6 days, +79 to +109 %
yield at the vanishing scale on the synthetic season, and after a failure before silking stage 6 ran on
the soft silking clock.)

Source: this implementation on the DSSAT-CSM v4.8.6.0 ``MZ_PHENOL`` / ``MZ_GROSUB``
kernels of :mod:`.phenology` and :mod:`.grosub_stages` (BSD-3, Copyright 1998-2026 DSSAT Foundation,
University of Florida, International Fertilizer Development Center); the logistic gate after the
forward-smoothing mode of torchcrop and van Bree et al. (2025, arXiv:2501.16848, Eq. 4-6).
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.process import process

from ._util import safe_div
from .constants import (
    CROP_STATUS_COLD,
    CROP_STATUS_DROUGHT,
    CROP_STATUS_NO_EMERGENCE,
    CROP_STATUS_NO_GERMINATION,
    ISTAGE_AFTER_MATURITY,
    ISTAGE_EFG,
    ISTAGE_EMERGENCE,
    ISTAGE_END_JUVENILE,
    ISTAGE_END_LEAF_GROWTH,
    ISTAGE_GERMINATION,
    ISTAGE_JUVENILE,
    ISTAGE_MATURITY,
    ISTAGE_SOWING,
    ISTAGE_TASSEL_INIT,
    PAIR_MEAN_WEIGHT,
    XSTAGE_COEFFICIENTS,
)
from .grosub_day import (
    GrosubDay,
    _nonneg,
    crop_failure,
    growth_totals,
    leaf_senescence,
)
from .grosub_stages import (
    OrganGrowth,
    assimilation,
    emergence_init,
    floral_induction_growth,
    grain_fill_growth,
    juvenile_growth,
    leaf_appearance,
    silk_efg_growth,
    stage_date_init,
    tassel_silk_growth,
)
from .phenology import (
    PhenDay,
    _advance,
    _count_day,
    emergence_block,
    germination_block,
    grain_fill_block,
    grain_number,
    juvenile_block,
    leaf_number_at_ti,
    maturity_block,
    photoperiod_rate,
    sowing_block,
    sowing_day,
    tassel_silk_block,
    thermal_time,
)
from .smoothed_params import N_PASSAGE, SmoothingCoefficients, SoftPhenologyState
from .state import CeresCultivar, CeresForcing, CeresMaizeParams, CeresMaizeState
from .stress import WATER_STRESS_COEFFICIENTS

__all__ = [
    "DUR4_MIN",
    "TIE",
    "SoftClocks",
    "ceres_growth_smoothed",
    "ceres_phenology_smoothed",
    "gate",
    "soft_clock",
    "soft_clocks",
    "stage_occupancy",
    "with_soft_state",
]

#: offset of every gate towards the passed side: an exact tie of a clock with its threshold counts as
#: passed, as the reference's ``>=`` (in the units of the gated quantity: degC d, or SIND)
TIE = numerical_guard(
    "ceres.smoothed_tie",
    1e-9,
    "an exact tie of a clock with its threshold passes the logistic gate as the reference's >= once the "
    "scale is below it",
)
# Precision: the offset acts on exact ties only (a clock difference that is exactly 0, as the SIND sums of
# 0.25 give in float32 and float64 alike); a near-tie within rounding of the clocks is not a tie of the
# reference either, and in float32 the hard limit (scale -> 0) is reproduced only to float32 rounding.

#: floor of the soft stage-4 duration [d] in the grain number: the soft duration of a crop that has not
#: reached stage 4 is a logistic tail (down to 1e-300 at small scales), and SUMP / DUR4 of two such
#: tails overflows and makes every forward-mode derivative NaN; at and above the floor the quotient is
#: unchanged (the reference's IDURP is an integer >= 1 when it is used, so the scale -> 0 limit is too)
DUR4_MIN = numerical_guard(
    "ceres.smoothed_dur4_min",
    1e-6,
    "floor of the soft stage-4 duration in SUMP / DUR4 (finite value and derivatives for logistic tails)",
)


def _soft_grain_number(
    sump: Array, dur4: Array, g2: ArrayLike, pltpop: Array, c: object
) -> tuple[Array, Array]:
    """:func:`~.phenology.grain_number` from the soft stage-4 sums, the duration floored at
    :data:`DUR4_MIN` (with ``dur4 = 0`` and ``sump = 0`` before stage 4 the quotient is 0 either way).

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, end of the ``ISTAGE .EQ. 4`` block, with this
    implementation's soft duration.
    """
    return grain_number(sump, jnp.maximum(dur4, DUR4_MIN), g2, pltpop, c)  # type: ignore[arg-type]


# passage indices (PASSAGE_NAMES)
_EM, _JUV, _TI, _SILK, _EFG, _END, _MAT = range(N_PASSAGE)


def _crop_failed(status: Array) -> Array:
    """The crop status of a failed crop: no germination, no emergence (phenology), cold or drought
    (growth); each sends the crop to stage 6 with ``MDATE = YRDOY``.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for / MZ_GROSUB.for, INTEGR (the CropStatus codes 12, 13, 32, 33).
    """
    return (
        (status == CROP_STATUS_NO_GERMINATION)
        | (status == CROP_STATUS_NO_EMERGENCE)
        | (status == CROP_STATUS_COLD)
        | (status == CROP_STATUS_DROUGHT)
    )


def gate(x: ArrayLike, scale: ArrayLike) -> Array:
    """The logistic stage gate ``1 / (1 + exp(-(x + TIE) / scale))`` of a threshold ``x >= 0`` (finite
    value and derivative for every ``x``; ``scale > 0``). The offset :data:`TIE` sends an exact tie
    ``x = 0`` to the passed side as the reference's ``>=`` does once the scale is far below it (the
    induction sum ``SIND`` of a short-day site adds ``1 / DJTI = 0.25`` a day and reaches 1 exactly);
    at the scales of the experiments it shifts nothing measurable.

    Source: this implementation, after the logistic gates of torchcrop and van Bree
    et al. (2025, Eq. 4).
    """
    return jax.nn.sigmoid((jnp.asarray(x) + TIE) / scale)


def soft_clock(x: ArrayLike, scale: ArrayLike) -> Array:
    """``scale log(1 + exp(x / scale))``: the integral of :func:`gate`, a clock that tends to
    ``max(x, 0)`` (the reference's ``SUMDTT - P`` carried over a stage end) as ``scale -> 0``.

    Source: this implementation.
    """
    return jnp.asarray(scale) * jnp.logaddexp(0.0, jnp.asarray(x) / scale)


class SoftClocks(NamedTuple):
    """The soft clocks of one day, after the day's thermal time (module docstring)."""

    tt_germ: Array  # thermal time since germination (SUMDTT of stage 9)
    tt_em: Array  # thermal time since emergence (SUMDTT of stages 1-2)
    tt_ti: Array  # thermal time since tassel initiation (SUMDTT of stage 3)
    tt_silk: Array  # thermal time since silking (SUMDTT of stages 4-6)
    p3: Array  # P3 from the soft thermal time before tassel initiation
    tlno: Array  # TLNO from the same


def soft_clocks(
    tt_germ: Array, tt_ti: Array, tt_pre_ti: Array, p9: Array, cul: CeresCultivar, scale: ArrayLike, c: object
) -> SoftClocks:
    """The derived clocks from the summed ones: ``tt_em = sp(tt_germ - P9)``, ``(TLNO, P3)`` from
    ``tt_pre_ti`` (:func:`~.phenology.leaf_number_at_ti`), ``tt_silk = sp(tt_ti - P3)``.

    Source: this implementation on MZ_PHENOL.for (DSSAT-CSM v4.8.6.0) ``TLNO`` /
    ``P3`` at tassel initiation.
    """
    tlno, p3 = leaf_number_at_ti(tt_pre_ti, cul.phint, c)  # type: ignore[arg-type]
    return SoftClocks(
        tt_germ=tt_germ,
        tt_em=soft_clock(tt_germ - p9, scale),
        tt_ti=tt_ti,
        tt_silk=soft_clock(tt_ti - p3, scale),
        p3=p3,
        tlno=tlno,
    )


def stage_occupancy(passage: Array) -> tuple[Array, Array, Array, Array, Array]:
    """Soft occupancies of the growth stages 1-5 from the passage chain: ``o1 = 1 - P_juv``,
    ``o2 = P_juv - P_ti``, ``o3 = P_ti - P_silk``, ``o4 = P_silk - P_efg``, ``o5 = P_efg - P_end``
    (each in ``[0, 1]``: the chain is a product of gates, so it never increases along the stages).

    Source: this implementation.
    """
    p = passage
    return (
        1.0 - p[..., _JUV],
        p[..., _JUV] - p[..., _TI],
        p[..., _TI] - p[..., _SILK],
        p[..., _SILK] - p[..., _EFG],
        p[..., _EFG] - p[..., _END],
    )


def with_soft_state(state: CeresMaizeState) -> CeresMaizeState:
    """``state`` with zero soft clocks in ``soft`` (what the smoothed processes need), same crop count
    and dtype.

    Source: this implementation.
    """
    lai = state.growth.lai
    return state.replace(soft=SoftPhenologyState.initial(int(lai.shape[-1]), dtype=lai.dtype))


def _floral_induction_soft(v: PhenDay, m: Array, sind: Array, clk: SoftClocks, xn: Array) -> PhenDay:
    """Stage 2 on the soft induction sum: tassel initiation when ``SIND >= 1`` (gate 0.5), with the
    soft ``TLNO`` and ``P3``, ``XNTI = XN`` and ``SUMDTT = 0`` (the faithful block's bookkeeping).

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 2`` block, with this
    implementation's soft ``SIND``.
    """
    v = _count_day(v, m)
    xs = XSTAGE_COEFFICIENTS
    v = v._replace(xstage=jnp.where(m, xs.ti_base + xs.ti_slope * sind, v.xstage))
    done = m & (sind >= 1.0)
    return _advance(v, done, ISTAGE_TASSEL_INIT)._replace(
        tlno=jnp.where(done, clk.tlno, v.tlno),
        p3=jnp.where(done, clk.p3, v.p3),
        xnti=jnp.where(done, xn, v.xnti),
        sumdtt=jnp.where(done, 0.0, v.sumdtt),
    )


def _silk_efg_soft(v: PhenDay, m: Array, cul: CeresCultivar, sump: Array, dur4: Array, c: object) -> PhenDay:
    """Stage 4 with the grain number from the soft stage-4 assimilation ``sump`` and duration ``dur4``
    (:func:`~.phenology.grain_number`); ``IDURP`` is still counted.

    Source: DSSAT-CSM v4.8.6.0 MZ_PHENOL.for, INTEGR, ``ISTAGE .EQ. 4`` block, with this
    implementation's soft duration.
    """
    v = _count_day(v, m)
    v = v._replace(idurp=jnp.where(m, v.idurp + 1, v.idurp))
    xs = XSTAGE_COEFFICIENTS
    v = v._replace(
        xstage=jnp.where(
            m,
            xs.efg_base + xs.efg_slope * safe_div(v.sumdtt, cul.p5 * c.efg_end_frac),  # type: ignore[attr-defined]
            v.xstage,
        )
    )
    done = m & (v.sumdtt >= cul.dsgft)
    gpp, ears = _soft_grain_number(sump, dur4, cul.g2, v.pltpop, c)
    v = v._replace(gpp=jnp.where(done, gpp, v.gpp), ears=jnp.where(done, ears, v.ears))
    return _advance(v, done, ISTAGE_EFG)


_SMOOTHED_DEVIATION = (
    "every thermal-time stage threshold of MZ_PHENOL (P9, P1, P3, DSGFT, 0.95 P5, P5) and the induction "
    "threshold SIND >= 1 is replaced by a logistic gate of scale CeresMaizeParams.smoothing.width; the stage "
    "clocks are soft (thermal time weighted by the soft passage of the previous boundary), TLNO and P3 come "
    "from the soft thermal time before tassel initiation, the grain number from the soft stage-4 "
    "assimilation and duration; the integer ISTAGE advances at gate 0.5; an early maturity or a crop "
    "failure forces the passages it crosses to 1",
    "a forward-smoothing comparator for the gradient experiments: every phenology coefficient gets a "
    "continuous derivative, at the price of a forward model that is no longer the reference",
    "tests/unit/test_ceres_smoothed.py (the faithful model as the scale tends to 0, monotone passages, "
    "finite derivatives); the departure from dscsm048 per scale is measured by "
    "scripts/bench/calib/smoothed_phenology.py on 52 of the 58 DSSAT maize example treatments (GAGR0201 "
    "excluded: its seasons end on padded days that repeat the last date, where the limit does not hold); "
    "the limit is tested with an early maturity, a drought failure before silking and cold failures in "
    "stages 4 and 5 (a crop failure forces the passages)",
)
_PRECISION_DEVIATION = (
    "DSSAT single precision (REAL*4) is not reproduced",
    "the kernels run in float64 (float32 with AGRI_JAX_X64=0)",
    "tests/integration/test_ceres_dssat.py tolerances",
)


@process(
    reads=(
        "phen",
        "soft",
        "growth.leafno",
        "growth.xn",
        "growth.pltpop",
        "season",
        "sowing",
        "water_in.sw",
        "snow_in.swe",
    ),
    writes=("phen", "growth.pltpop", "soft"),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_PHENOL.for (BSD-3), stage thresholds smoothed",
    fortran_name="MZ_PHENOL",
    key="crop/ceres_maize.phenology@dssat-4.8.6.0:alt_smoothed",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        (
            "thermal time, stage blocks ISTAGE 7, 8, 9, 1-6, RATEIN, TLNO / P3, GPP / EARS",
            "MZ_PHENOL.for INTEGR (BSD-3)",
        ),
        ("logistic stage gate", "torchcrop smooth mode; van Bree et al. (2025), arXiv:2501.16848, Eq. 4-6"),
        ("soft clocks, passage chain and stage occupancies", "this implementation"),
    ),
    deviates=(
        _SMOOTHED_DEVIATION,
        (
            "nitrogen off (ISWNIT = N): XSTAGE is kept but nothing reads it; VegFrac / SeedFrac not ported",
            "the nitrogen and phosphorus modules are not built in this implementation",
            "phenology.py module docstring",
        ),
        _PRECISION_DEVIATION,
    ),
)
def ceres_phenology_smoothed(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """One day of ``MZ_PHENOL`` with logistic stage gates (module docstring).

    The soft clocks start from zero on the sowing stage (``ISTAGE = 7``) and run from the day after
    germination on, also after maturity (so every passage tends to 1); the faithful stage blocks of
    :mod:`.phenology` run on the soft clocks (germination and sowing unchanged), ``SUMDTT`` is written
    as the soft clock of the new stage, ``TLNO`` / ``P3`` as their soft values from emergence on, and
    ``SIND`` as the soft induction sum. An early maturity set by growth (``SUMDTT = P5`` in stage 5 or
    6) is kept as in the reference and sets the passages of the boundaries it crosses (and of those
    before them) to 1, where they stay on the following days (a passage at exactly 1 is not recomputed
    from the clocks). A failed crop (stage 6 with a failure status) stops its soft clocks, runs stage 6
    on its carried ``SUMDTT`` and passes every boundary up to the end of effective grain filling (a seed
    failure here, a cold or drought failure in growth) and maturity on its hard maturity day. Forcing
    read: ``yrdoy``, ``tmax``, ``tmin``, ``srad``, ``dayl``, ``twilen``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_PHENOL.for, DYNAMIC = INTEGR (BSD-3), with the
    logistic gates of this implementation.
    """
    ph, g = state.phen, state.growth
    q = state.soft
    if q is None:  # static: the structure of the state
        raise ValueError("the smoothed phenology needs CeresMaizeState.soft (smoothed.with_soft_state)")
    params = params.in_season(state.season)
    cul = params.cultivar
    c = params.coef().phenol
    sm: SmoothingCoefficients = params.smoothing_coef()
    scale = jnp.asarray(sm.width)
    scale_sind = scale / sm.sind_tt
    f = forcing_t
    yrdoy = jnp.asarray(f.yrdoy)
    s = ph.istage
    active = sowing_day(state, params, yrdoy) | (s != ISTAGE_SOWING)
    dtt_raw = thermal_time(f.tmax, f.tmin, f.srad, f.dayl, state.snow_in.swe, g.leafno, s, cul, c)

    # ---- soft clocks: zero up to the sowing stage, running from the day after germination
    fresh = s == ISTAGE_SOWING
    z = jnp.zeros_like(q.tt_germ)
    q0 = SoftPhenologyState(
        tt_germ=jnp.where(fresh, z, q.tt_germ),
        sind=jnp.where(fresh, z, q.sind),
        tt_pre_ti=jnp.where(fresh, z, q.tt_pre_ti),
        tt_ti=jnp.where(fresh, z, q.tt_ti),
        passage=jnp.where(fresh[..., None], 0.0, q.passage),
        dur4=jnp.where(fresh, z, q.dur4),
        sump=jnp.where(fresh, z, q.sump),
    )
    # a failed crop (stage 6 after a germination, emergence, cold or drought failure): its soft clocks
    # stop (the passages were forced to 1 on the failure day, below and in the growth process)
    failed = (s == ISTAGE_MATURITY) & _crop_failed(ph.crop_status)
    running = active & (s != ISTAGE_SOWING) & (s != ISTAGE_GERMINATION) & ~failed
    dtt_r = jnp.where(running, dtt_raw, 0.0)
    prev = q0.passage
    p9 = ph.p9
    tt_em_prev = soft_clock(q0.tt_germ - p9, scale)
    tt_germ = q0.tt_germ + dtt_r
    tt_em = soft_clock(tt_germ - p9, scale)
    sind = q0.sind + jnp.where(running, photoperiod_rate(f.twilen, cul) * prev[..., _JUV], 0.0)
    tt_pre_ti = q0.tt_pre_ti + jnp.where(running, (tt_em - tt_em_prev) * (1.0 - prev[..., _TI]), 0.0)
    tt_ti = q0.tt_ti + dtt_r * prev[..., _TI]
    clk = soft_clocks(tt_germ, tt_ti, tt_pre_ti, p9, cul, scale, c)
    g_juv = gate(clk.tt_em - cul.p1, scale)
    g_ti = gate(sind - 1.0, scale_sind)
    g_silk = gate(clk.tt_ti - clk.p3, scale)
    g_efg = gate(clk.tt_silk - cul.dsgft, scale)
    g_end = gate(clk.tt_silk - c.efg_end_frac * cul.p5, scale)
    g_mat = gate(clk.tt_silk - cul.p5, scale)
    q_juv = g_juv
    q_ti = q_juv * g_ti
    q_silk = q_ti * g_silk
    q_efg = q_silk * g_efg
    q_end = q_efg * g_end
    q_mat = q_end * g_mat
    passage_new = jnp.stack([gate(tt_germ - p9, scale), q_juv, q_ti, q_silk, q_efg, q_end, q_mat], axis=-1)
    passage = jnp.where(running[..., None], passage_new, prev)
    # soft stage-4 duration: the day-start occupancy of stage 4 (the reference counts IDURP in the
    # stage-4 block, on the days the crop starts in stage 4)
    dur4 = q0.dur4 + jnp.where(running, prev[..., _SILK] - prev[..., _EFG], 0.0)

    # ---- the hard stage machine on the soft clocks (faithful blocks, masks of the day-start stage)
    early = ((s == ISTAGE_EFG) | (s == ISTAGE_MATURITY)) & (ph.sumdtt >= cul.p5)  # growth's early maturity
    # after an early maturity or a crop failure the stage-5 / 6 clock is the reference's SUMDTT carried
    # on (a failed crop keeps the SUMDTT of the stage it failed in)
    hard_56 = early | failed
    clock_56 = jnp.where(hard_56, ph.sumdtt + dtt_raw, clk.tt_silk)
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
    st = [active & (s == k) for k in range(ISTAGE_AFTER_MATURITY + 1)]

    def on(v_: PhenDay, k: int, clock: Array) -> PhenDay:
        return v_._replace(sumdtt=jnp.where(st[k], clock, v_.sumdtt))

    v = sowing_block(v, st[ISTAGE_SOWING], params)
    v = germination_block(v, st[ISTAGE_GERMINATION], params, state.water_in.sw, yrdoy, c)
    v = emergence_block(on(v, ISTAGE_EMERGENCE, clk.tt_germ), st[ISTAGE_EMERGENCE], params, yrdoy, c)
    v = juvenile_block(on(v, ISTAGE_JUVENILE, clk.tt_em), st[ISTAGE_JUVENILE], cul)
    v = _floral_induction_soft(
        on(v, ISTAGE_END_JUVENILE, clk.tt_em), st[ISTAGE_END_JUVENILE], sind, clk, g.xn
    )
    v = tassel_silk_block(
        on(v, ISTAGE_TASSEL_INIT, clk.tt_ti)._replace(p3=jnp.where(st[ISTAGE_TASSEL_INIT], clk.p3, v.p3)),
        st[ISTAGE_TASSEL_INIT],
    )
    v = _silk_efg_soft(
        on(v, ISTAGE_END_LEAF_GROWTH, clk.tt_silk), st[ISTAGE_END_LEAF_GROWTH], cul, q0.sump, dur4, c
    )
    v = grain_fill_block(on(v, ISTAGE_EFG, clock_56), st[ISTAGE_EFG], cul, c)
    v = maturity_block(on(v, ISTAGE_MATURITY, clock_56), st[ISTAGE_MATURITY], cul, yrdoy, c)

    # ---- outputs: SUMDTT as the soft clock of the new stage, soft TLNO / P3 from emergence on
    n = v.istage
    emerged = (n != ISTAGE_SOWING) & (n != ISTAGE_GERMINATION) & (n != ISTAGE_EMERGENCE)
    late = (n == ISTAGE_EFG) | (n == ISTAGE_MATURITY)
    sumdtt = jnp.select(
        [
            n == ISTAGE_EMERGENCE,
            (n == ISTAGE_JUVENILE) | (n == ISTAGE_END_JUVENILE),
            n == ISTAGE_TASSEL_INIT,
            (n == ISTAGE_END_LEAF_GROWTH) | (late & ~hard_56),
        ],
        [clk.tt_germ, clk.tt_em, clk.tt_ti, clk.tt_silk],
        v.sumdtt,
    )
    sumdtt = jnp.where(active & (s != ISTAGE_SOWING) & (s != ISTAGE_GERMINATION), sumdtt, v.sumdtt)
    # an early maturity (growth's SUMDTT = P5) passes the end of effective grain filling when the hard
    # stage does, and maturity with it: no stage-5 occupancy is left to fill grain on the stage-6 days
    # (likewise the reference's immediate maturity of a stage-6 crop on a day with DTT < 2). A crop
    # failure passes every boundary up to the end of effective grain filling on the failure day (no stage
    # occupancy is left to grow), and maturity on the hard maturity day of the failed crop
    k = jnp.arange(N_PASSAGE)
    quick = (s == ISTAGE_MATURITY) & (dtt_raw < c.dtt_maturity) & (n == ISTAGE_AFTER_MATURITY)
    fail_now = failed | ((n == ISTAGE_MATURITY) & _crop_failed(v.status))  # phenology's seed failures
    end_now = (early & ((n == ISTAGE_MATURITY) | (n == ISTAGE_AFTER_MATURITY))) | quick | fail_now
    mat_now = (hard_56 & (n == ISTAGE_AFTER_MATURITY)) | quick
    # a forced passage stays at 1 on the following days: from the first stage-10 day on `early` is False
    # and the soft clock of an early-matured crop is still below P5, so a passage recomputed from the
    # clocks would drop back towards 0 and the soft date would keep counting after the hard maturity (a
    # passage that reaches 1.0 from its gate is kept at 1 too: its value and derivative are already 1, 0).
    # The boundaries before a forced one are passed with it, so the chain stays ordered and every stage
    # occupancy non-negative (at a finite scale the soft DSGFT passage can still be below 1 there).
    end_now = end_now | (prev[..., _END] >= 1.0)
    mat_now = mat_now | (prev[..., _MAT] >= 1.0)
    passage = jnp.where(end_now[..., None] & (k <= _END), 1.0, passage)
    passage = jnp.where(mat_now[..., None] & (k <= _MAT), 1.0, passage)
    at_ti = st[ISTAGE_END_JUVENILE] & (n == ISTAGE_TASSEL_INIT)  # the soft stage-4 sums start here
    soft = SoftPhenologyState(
        tt_germ=tt_germ,
        sind=sind,
        tt_pre_ti=tt_pre_ti,
        tt_ti=tt_ti,
        passage=passage,
        dur4=jnp.where(at_ti, 0.0, dur4),
        sump=jnp.where(at_ti, 0.0, q0.sump),
    )

    col = (jnp.arange(ph.stgdoy.shape[-1]) + 1) == s[..., None]
    stgdoy = jnp.where(col & v.ended[..., None], yrdoy, ph.stgdoy)
    new_phen = ph.replace(
        istage=n.astype(ph.istage.dtype),
        sumdtt=sumdtt,
        cumdtt=v.cumdtt,
        dtt=v.dtt,
        ndas=v.ndas,
        xstage=v.xstage,
        sind=jnp.where(emerged, sind, v.sind),
        p3=jnp.where(emerged, clk.p3, v.p3),
        p9=v.p9,
        tlno=jnp.where(emerged, clk.tlno, v.tlno),
        xnti=v.xnti,
        gpp=v.gpp,
        ears=v.ears,
        idurp=v.idurp.astype(ph.idurp.dtype),
        seed_layer=v.seed_layer.astype(ph.seed_layer.dtype),
        stgdoy=stgdoy.astype(ph.stgdoy.dtype),
        mdate=v.mdate.astype(ph.mdate.dtype),
        crop_status=v.status.astype(ph.crop_status.dtype),
    )
    return eqx.tree_at(lambda x: (x.phen, x.growth.pltpop, x.soft), state, (new_phen, v.pltpop, soft))


# ------------------------------------------------------------------------------------------- growth
def _mix(weights: tuple[Array, ...], vals: tuple[Array, ...], keep: Array) -> Array:
    """``sum_k w_k vals_k + (1 - sum_k w_k) keep`` for one field (the soft counterpart of a stage
    ``jnp.select``; with one-hot weights it is the selected value bit for bit).

    Source: this implementation.
    """
    rest = jnp.ones_like(keep)
    for w in weights:
        rest = rest - w
    out = rest * keep
    for w, x in zip(weights, vals, strict=True):
        out = out + w * x
    return out


def _blend(weights: tuple[Array, ...], blocks: tuple[OrganGrowth, ...], keep: OrganGrowth) -> OrganGrowth:
    """:func:`_mix` field by field over the stage blocks (the soft counterpart of
    :func:`~.grosub_stages.select_stage_block`).

    Source: this implementation.
    """
    return OrganGrowth(*(_mix(weights, tuple(vals), k) for *vals, k in zip(*blocks, keep, strict=True)))


class SmoothedGrosubDay(NamedTuple):
    """The blended growth day and its masks."""

    day: GrosubDay  # ``grow`` is the soft support (stages 1-6, not the maturity / failure day)
    grow_hard: Array  # the faithful growth mask (stages 1-5)
    tail: Array  # stage-6 days of the support: only the stage-5 occupancy left over grows (none after a
    # crop failure: its passages are all 1)
    o5: Array  # stage-5 occupancy
    sump_soft: Array  # soft stage-4 assimilation after today


def grosub_blocks_smoothed(
    state: CeresMaizeState,
    params: CeresMaizeParams,
    forcing_t: CeresForcing,
    photo_stress: Array,
) -> SmoothedGrosubDay:
    """The ``MZ_GROSUB`` day of :func:`~.grosub_day.grosub_blocks` with the five stage blocks blended by
    the soft stage occupancies (:func:`stage_occupancy`) over the support stages 1-6, each block on its
    own soft clock (stages 1-2: thermal time since emergence, 3: since tassel initiation, 4-5: since
    silking). The stage-date resets, the emergence initialisation and the early-maturity counters stay
    on the hard stage.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for, INTEGR (BSD-3), with this implementation's blend.
    """
    ph, g, stq = state.phen, state.growth, state.stress
    q = state.soft
    assert q is not None  # checked by the process
    cul, spe, f = params.cultivar, params.species, forcing_t
    c = params.coef().grosub
    scale = jnp.asarray(params.smoothing_coef().width)
    yrdoy = jnp.asarray(f.yrdoy)
    s = ph.istage
    called = (s >= 1) & (s <= WATER_STRESS_COEFFICIENTS.grosub_last_stage)
    pltpop = g.pltpop
    sd = stage_date_init(yrdoy, called, ph.stgdoy, pltpop, ph.ears, g, c)
    em = emergence_init(called & (yrdoy == ph.stgdoy[..., 8]), pltpop, g, spe, c)
    base = called & (ph.mdate != yrdoy)
    grow_hard = base & (s <= ISTAGE_EFG) & ~((s == ISTAGE_EFG) & (pltpop <= c.pltpop_min5))
    support = base & ~((s >= ISTAGE_EFG) & (pltpop <= c.pltpop_min5))
    tail = support & ~grow_hard

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
    fexp = jnp.minimum(turfac, 1.0 - stq.satfac)
    dtt = ph.dtt
    tt_em = soft_clock(q.tt_germ - ph.p9, scale)
    tt_ti = q.tt_ti
    tt_silk = soft_clock(tt_ti - ph.p3, scale)
    la = leaf_appearance(em.cumph, dtt, cul.phint, tt_ti, ph.p3, c)
    org = OrganGrowth(
        pla=em.pla,
        lfwt=em.lfwt,
        stmwt=em.stmwt,
        earwt=g.earwt,
        grort=g.grort,
        slan=g.slan,
        cls=g.cum_leaf_senes,
    )
    b1, seedrv_1 = juvenile_growth(org, carbo, la, fexp, em.seedrv, tt_em, pltpop, c)
    b2 = floral_induction_growth(org, carbo, la, fexp, tt_em, pltpop, c)
    # a block weighted before the hard event that sets one of its inputs gets the value the event will
    # set: XNTI = XN at tassel initiation, SWMIN = 0.85 STMWT at silking, SWMAX = STMWT and GPP (from
    # the soft stage-4 sums) at the beginning of effective grain filling (with zero weight, as in the
    # scale -> 0 limit, nothing changes)
    xnti = jnp.where(s <= ISTAGE_END_JUVENILE, g.xn, ph.xnti)
    swmin = jnp.where(s <= ISTAGE_TASSEL_INIT, em.stmwt * c.swmin_frac, sd.swmin)
    swmax = jnp.where(s <= ISTAGE_END_LEAF_GROWTH, em.stmwt, sd.swmax)
    gpp_soft, _ = _soft_grain_number(q.sump, q.dur4, cul.g2, pltpop, params.coef().phenol)
    gpp = jnp.where(s <= ISTAGE_END_LEAF_GROWTH, gpp_soft, ph.gpp)
    b3, cumdtteg_3 = tassel_silk_growth(
        org, carbo, la, fexp, tt_ti, ph.p3, ph.tlno, xnti, spe.bsgdd, turfac, pltpop, g.stg2cls, c
    )
    b4, cumdtteg_4, _ = silk_efg_growth(
        org, carbo, fexp, dtt, tt_silk, g.cumdtteg, sd.sump, pltpop, g.stg2cls, c
    )
    b5, gf = grain_fill_growth(
        org,
        carbo,
        (tmax + tmin) * PAIR_MEAN_WEIGHT,
        swfac,
        tt_silk,
        gpp,
        g.grnwt,
        g.grogrn,
        sd.emat,
        g.cmat,
        swmin,
        swmax,
        cul.g3,
        cul.p5,
        spe,
        c,
    )
    occ = stage_occupancy(q.passage)
    w = tuple(jnp.where(support, o, 0.0) for o in occ)
    w1, w2, w3, w4, w5 = w
    hard = [grow_hard & (s == k) for k in range(1, 6)]
    m5 = hard[4]
    xn = _mix((w1, w2, w3), (la.xn, la.xn, la.xn3), g.xn)
    mat = gf.maturity
    day = GrosubDay(
        grow=support,
        sd=sd,
        em=em,
        carbo=carbo,
        organs=_blend(w, (b1, b2, b3, b4, b5), org),
        cumph=_mix((w1, w2, w3), (la.cumph, la.cumph, la.cumph3), em.cumph),
        xn=xn,
        leafno=jnp.where(hard[0] | hard[1] | hard[2], jnp.floor(xn).astype(em.leafno.dtype), em.leafno),
        cumdtteg=_mix((w3, w4), (cumdtteg_3, cumdtteg_4), g.cumdtteg),
        seedrv=_mix((w1,), (seedrv_1,), em.seedrv),
        stg2cls=_mix((w2,), (b2.cls,), g.stg2cls),
        sump=jnp.where(hard[3], sd.sump + carbo, sd.sump),
        grnwt=_mix((w5,), (gf.grnwt,), g.grnwt),
        grogrn=_mix((w5,), (gf.grogrn,), g.grogrn),
        emat=jnp.where(m5, mat.emat, sd.emat),
        cmat=jnp.where(m5, mat.cmat, g.cmat),
        sumdtt=jnp.where(m5 & mat.early, cul.p5 * jnp.ones_like(pltpop), ph.sumdtt),
    )
    return SmoothedGrosubDay(day=day, grow_hard=grow_hard, tail=tail, o5=w5, sump_soft=q.sump + w4 * carbo)


@process(
    reads=("phen", "stress", "growth", "season", "soft"),
    writes=(
        "growth",
        "phen.istage",
        "phen.mdate",
        "phen.crop_status",
        "phen.sumdtt",
        "phen.ears",
        "phen.gpp",
        "soft.sump",
        "soft.passage",
    ),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for (BSD-3), stage blocks blended",
    fortran_name="MZ_GROSUB",
    key="crop/ceres_maize.growth@dssat-4.8.6.0:alt_smoothed",
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        ("everything of the faithful growth process", "MZ_GROSUB.for INTEGR (BSD-3), as ceres_growth"),
        ("blend of the stage blocks by soft stage occupancies", "this implementation"),
    ),
    deviates=(
        (
            "the five stage blocks of MZ_GROSUB are blended with the soft stage occupancies of the "
            "smoothed phenology (differences of successive soft passages) instead of selected by ISTAGE, "
            "each block on its own soft thermal-time clock; on the stage-6 days the occupancy of stage 5 "
            "that is left still fills grain (with its share of the senescence and root turnover), none after "
            "a crop failure (its passages up to the end of effective grain filling are set to 1 here on the "
            "failure day); the soft stage-4 assimilation for the grain number is summed here",
            "the growth half of the forward-smoothing comparator: without it the phenology "
            "coefficients would move the yield only through the integer stage dates",
            "tests/unit/test_ceres_smoothed.py (the faithful growth as the scale tends to 0 while YRDOY "
            "strictly increases, also after an early maturity, a drought failure before silking and cold "
            "failures in stages 4 and 5; on days that repeat a date the soft stage-4 sums miss the "
            "faithful SUMP resets, DSSAT example GAGR0201); "
            "scripts/bench/calib/smoothed_phenology.py (departure from dscsm048 per scale)",
        ),
        (
            "nitrogen, phosphorus, potassium and pests off: AGEFAC = NSTRES = NDEF3 = PSTRES1 = PSTRES2 = "
            "KSTRES = 1, no pest damage",
            "the nutrient and pest modules are not built in this implementation",
            "growth.py module docstring",
        ),
        _PRECISION_DEVIATION,
    ),
)
def ceres_growth_smoothed(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """One ``MZ_GROSUB`` day with the stage blocks blended by the soft occupancies of the smoothed
    phenology (:func:`grosub_blocks_smoothed`), then the faithful senescence, crop failure and totals.

    On a stage-6 day of the support (``tail``) the day is the faithful no-growth day except the organ
    weights (already weighted by the stage-5 occupancy ``o5``) and the senescence and root turnover,
    taken as ``o5`` of the full day's change. On the day of a cold or drought failure the passages up to
    the end of effective grain filling are set to 1 (no occupancy is left for the tail). Forcing read:
    ``yrdoy``, ``tmax``, ``tmin``, ``srad``, ``co2``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, DYNAMIC = INTEGR (BSD-3), with this
    implementation's blend.
    """
    if state.soft is None:  # static: the structure of the state
        raise ValueError("the smoothed growth needs CeresMaizeState.soft (smoothed.with_soft_state)")
    p = params.in_season(state.season)
    ph, g = state.phen, state.growth
    c = p.coef().grosub
    swfac = state.stress.swfac
    sg = grosub_blocks_smoothed(state, p, forcing_t, swfac)
    d, em = sg.day, sg.day.em
    tmin = jnp.asarray(forcing_t.tmin)
    senla_f, lai_f = leaf_senescence(
        d.organs.pla, em.senla, d.organs.slan, em.lai, swfac, tmin, g.pltpop, p.species.fslfw, c
    )
    senla = jnp.where(sg.tail, em.senla + sg.o5 * (senla_f - em.senla), jnp.where(d.grow, senla_f, em.senla))
    lai = jnp.where(sg.tail, em.lai + sg.o5 * (lai_f - em.lai), jnp.where(d.grow, lai_f, em.lai))
    fail = crop_failure(
        sg.grow_hard,
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
        p.cultivar.tsen,
        p.cultivar.cday,
        c,
    )
    canht_pot = p.species.canht_pot
    # ``full`` grows on the whole support, ``none`` only on the faithful growth days (the two agree
    # off the stage-6 tail); on the tail the day is ``none`` but for the organs and the turnover
    full = growth_totals(g, d, senla, lai, fail, canht_pot, c)
    none = growth_totals(g, d._replace(grow=sg.grow_hard), senla, lai, fail, canht_pot, c)
    tl = sg.tail
    new_growth = none.replace(
        leaf=jax.tree_util.tree_map(
            lambda a, b: jnp.where(jnp.reshape(tl, tl.shape + (1,) * (a.ndim - tl.ndim)), a, b),
            full.leaf,
            none.leaf,
        ),
        senla=jnp.where(tl, full.senla, none.senla),
        lai=jnp.where(tl, full.lai, none.lai),
        stmwt=jnp.where(tl, full.stmwt, none.stmwt),
        earwt=jnp.where(tl, full.earwt, none.earwt),
        grnwt=jnp.where(tl, full.grnwt, none.grnwt),
        grogrn=jnp.where(tl, full.grogrn, none.grogrn),
        biomas=jnp.where(tl, full.biomas, none.biomas),
        rtwt=jnp.where(tl, g.rtwt + sg.o5 * (full.rtwt - g.rtwt), none.rtwt),
    )
    new_phen = eqx.tree_at(
        lambda x: (x.istage, x.mdate, x.crop_status, x.sumdtt, x.ears, x.gpp),
        ph,
        (
            fail.istage.astype(ph.istage.dtype),
            fail.mdate.astype(ph.mdate.dtype),
            fail.status.astype(ph.crop_status.dtype),
            d.sumdtt,
            _nonneg(sg.grow_hard, d.sd.ears),
            _nonneg(sg.grow_hard, ph.gpp),
        ),
    )
    sump = jnp.where(d.grow, sg.sump_soft, state.soft.sump)
    # a cold or drought failure passes every boundary up to the end of effective grain filling today, as
    # the hard stage 6 does (no soft stage occupancy is left to grow on the following days; maturity is
    # passed with the hard maturity of the failed crop, in the phenology)
    failed_now = sg.grow_hard & (fail.istage == ISTAGE_MATURITY)
    k = jnp.arange(N_PASSAGE)
    passage = jnp.where(failed_now[..., None] & (k <= _END), 1.0, state.soft.passage)
    return eqx.tree_at(
        lambda x: (x.phen, x.growth, x.soft.sump, x.soft.passage),  # type: ignore[union-attr]
        state,
        (new_phen, new_growth, sump, passage),
    )
