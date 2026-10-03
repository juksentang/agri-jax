"""Ritchie's two-stage soil evaporation ``SOILEV`` / ``ESUP`` (DSSAT-CSM v4.8.6.0, ``MESEV = 'R'``).

Stage 1 (energy limited): the day's evaporation equals the potential ``EOS`` until the cumulative
stage-1 evaporation ``SUMES1`` reaches the upper limit ``U`` (the soil's ``SLU1``, mm). Stage 2
(supply limited): the cumulative stage-2 evaporation follows ``SUMES2 = 3.5 sqrt(T)``, ``T`` the
days into stage 2. Infiltration ``WINF`` (mm) first refills ``SUMES2``, then ``SUMES1``. The day's
``ES`` is then limited by the water of the top layer above the air-dry content ``SWEF LL(1)``
(``AWEV1`` from ``SW(1)``, and ``SWMIN`` from the day's available ``SW_AVAIL(1) = SW(1) + SWDELTS(1)
+ SWDELTU(1)``, both in mm over ``DLAYR(1)``), and reduced by a plastic-mulch fraction.

The routine only computes ``ES``; the bucket (``WATBAL``, ``Soil/SoilWater/WATBAL.for:471-474``)
removes it from the top layer. ``SUMES1``, ``SUMES2``, ``T`` and ``SWEF`` are the routine's
``SAVE``d state (:class:`SoilevStore`): set at ``SEASINIT`` from the initial top-layer water
(:func:`soilev_init`), advanced by every ``RATE`` call (:func:`soilev_rate`). SPAM calls ``RATE``
only when the day's potential soil evaporation after flood and mulch exceeds 1e-6 mm; the
process gates that (:mod:`.process`).

Both kernels are elementwise over any leading batch axes; every ``where`` has finite branches. At
``WINF = 0`` (no rain, a domain boundary) the stage-2 derivative with respect to ``WINF`` is the
right-hand one (:func:`agrijax.core.grad.one_sided_tangent`; values unchanged).

Source: DSSAT-CSM v4.8.6.0 ``SPAM/SOILEV.for`` (``SOILEV`` lines 30-183, ``ESUP`` lines 200-228) and
``SPAM/SPAM.for`` lines 358-367, BSD-3 (Copyright 1998-2026 DSSAT Foundation, University of Florida,
International Fertilizer Development Center); Ritchie, J.T. (1972), Water Resour. Res. 8, 1204-1213.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.grad import one_sided_tangent
from agrijax.core.units import MM_PER_CM

from .coefficients import EVAP_COEFFICIENTS, SoilevCoefficients

__all__ = ["SoilevStore", "soilev_init", "soilev_rate"]

# DUL(1) - LL(1) > 0 for a valid soil; the floor keeps SWR finite for a degenerate one
_RANGE_MIN = numerical_guard("soilev.range_min", 1e-12, "floor of DUL(1) - LL(1) in SWR (SOILEV SEASINIT)")


class SoilevStore(NamedTuple):
    """The ``SAVE``d state of ``SOILEV``: stage sums [mm], stage-2 days [d], air-dry fraction [-]."""

    sumes1: ArrayLike
    sumes2: ArrayLike
    t: ArrayLike
    swef: ArrayLike


def soilev_init(
    sw1: ArrayLike,
    ll1: ArrayLike,
    dul1: ArrayLike,
    dlayr1: ArrayLike,
    u: ArrayLike,
    c: SoilevCoefficients = EVAP_COEFFICIENTS.soilev,
) -> SoilevStore:
    """``SOILEV`` ``SEASINIT``: the stage sums from the top layer's deficit below DUL, and SWEF.

    ``SWR = max(0, (SW - LL) / (DUL - LL))``, ``USOIL = (DUL - SW) DLAYR`` [mm]: wet (``SWR >= 1``)
    all zero; ``USOIL <= U``: stage 1 with ``SUMES1 = USOIL``; else stage 2 with ``SUMES1 = U``,
    ``SUMES2 = USOIL - U``, ``T = (SUMES2 / 3.5)^2``. ``SWEF = 0.9 - 0.00038 (DLAYR - 30)^2``.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SOILEV.for lines 64-85 (BSD-3).
    """
    sw1, ll1, dul1, dlayr1, u = (jnp.asarray(x) for x in (sw1, ll1, dul1, dlayr1, u))
    swr = jnp.maximum(0.0, (sw1 - ll1) / jnp.maximum(dul1 - ll1, _RANGE_MIN))
    usoil = (dul1 - sw1) * dlayr1 * MM_PER_CM
    zero = jnp.zeros_like(usoil)
    s2_st2 = usoil - u
    wet = swr >= 1.0
    st1 = usoil <= u
    sumes1 = jnp.where(wet, zero, jnp.where(st1, usoil, u + zero))
    sumes2 = jnp.where(wet | st1, zero, s2_st2)
    t_st2 = (s2_st2 / c.stage2_rate) ** 2
    t = jnp.where(wet | st1, zero, t_st2)
    swef = c.swef_max - c.swef_curvature * (dlayr1 - c.swef_dlayr_ref) ** 2
    return SoilevStore(sumes1, sumes2, t, swef + zero)


def _esup(eos: Array, s1: Array, s2: Array, t: Array, u: Array, c: SoilevCoefficients) -> tuple[Array, ...]:
    """``ESUP``: stage-1 evaporation; on passing ``U`` the excess starts stage 2.

    Returns ``(es, sumes1, sumes2, t)``; ``sumes2`` and ``t`` keep their incoming values when
    ``SUMES1 + EOS <= U`` (the Fortran does not assign them then).

    Source: DSSAT-CSM v4.8.6.0 SPAM/SOILEV.for ESUP, lines 218-226 (BSD-3).
    """
    s1n = s1 + eos
    over = s1n > u
    excess = s1n - u
    es = jnp.where(over, eos - c.esup_es_frac * excess, eos)
    s2n = jnp.where(over, c.esup_s2_frac * excess, s2)
    t_over = (c.esup_s2_frac * excess / c.stage2_rate) ** 2
    tn = jnp.where(over, t_over, t)
    return es, jnp.where(over, u, s1n), s2n, tn


def soilev_rate(
    store: SoilevStore,
    eos: ArrayLike,
    winf: ArrayLike,
    sw1: ArrayLike,
    ll1: ArrayLike,
    dlayr1: ArrayLike,
    sw_avail1: ArrayLike,
    u: ArrayLike,
    pmfraction: ArrayLike = 0.0,
    c: SoilevCoefficients = EVAP_COEFFICIENTS.soilev,
) -> tuple[Array, SoilevStore]:
    """``SOILEV`` ``RATE``: the day's actual soil evaporation ``ES`` [mm d-1] and the new store.

    ``eos`` is the potential soil evaporation left after flood and mulch (``EOS_SOIL``, mm d-1),
    ``winf`` the water available for infiltration (mm), ``sw1`` / ``ll1`` / ``dlayr1`` the top
    layer's water content, lower limit and thickness (cm), ``sw_avail1`` its available water
    ``max(0, SW + SWDELTS + SWDELTU)`` (SPAM.for:361-363), ``u`` the stage-1 limit (mm),
    ``pmfraction`` the plastic-mulch cover (``GET('PM', 'PMFRACTION')``).

    Source: DSSAT-CSM v4.8.6.0 SPAM/SOILEV.for lines 96-174 (BSD-3).
    """
    eos, winf, sw1, ll1, dlayr1, sw_avail1, u, pm = (
        jnp.asarray(x) for x in (eos, winf, sw1, ll1, dlayr1, sw_avail1, u, pmfraction)
    )
    s1, s2, t, swef = (jnp.asarray(x) for x in store)
    zero = jnp.zeros_like(s1)
    in2 = s1 >= u
    # branch 1: stage 1 after rain that refilled stage 2 (lines 96-104)
    winfmod = winf - s2
    s1_b1 = jnp.where(winfmod > u, zero, u - winfmod)
    es1, s1_1, s2_1, t_1 = _esup(eos, s1_b1, zero, zero, u, c)
    # branch 2: stage 2 (lines 106-119)
    t_b2 = t + 1.0
    es_b2 = c.stage2_rate * jnp.sqrt(t_b2) - s2
    esx = c.stage2_infil_frac * winf
    esx = jnp.where(esx <= es_b2, es_b2 + winf, esx)
    esx = jnp.minimum(esx, eos)
    es2 = jnp.where(winf > 0.0, esx, jnp.minimum(es_b2, eos))
    # infiltration cannot go below 0: at WINF = 0 the dry branch MIN(ES, EOS) is selected (d ES / d WINF
    # = 0, and SUMES2 = SUMES2 + ES - WINF falls with WINF), but just above it ESX = ES + WINF while
    # 0.6 WINF <= ES (else 0.6 WINF), cut at EOS. The derivative there is the right-hand one (values
    # unchanged, core.grad.one_sided_tangent): d ES = d WINF while 0 < ES < EOS (so d SUMES2 = 0: the
    # rain evaporates), 0.6 d WINF for ES <= 0, 0 once EOS binds
    slope2 = jnp.where(
        es_b2 > 0.0, jnp.where(es_b2 < eos, 1.0, 0.0), jnp.where(eos > 0.0, c.stage2_infil_frac, 0.0)
    )
    es2 = es2 + one_sided_tangent(winf, slope2, winf == 0.0)
    s2_2 = s2 + es2 - winf
    t_2 = (s2_2 / c.stage2_rate) ** 2
    # branch 3: stage 1 reset by rain (lines 121-124)
    es3, s1_3, s2_3, t_3 = _esup(eos, zero, s2, t, u, c)
    # branch 4: stage 1 (lines 126-129)
    es4, s1_4, s2_4, t_4 = _esup(eos, s1 - winf, s2, t, u, c)
    b1 = in2 & (winf >= s2)
    b2 = in2 & (winf < s2)
    b3 = ~in2 & (winf >= s1)
    es = jnp.select([b1, b2, b3], [es1, es2, es3], es4)
    s1 = jnp.select([b1, b2, b3], [s1_1, s1, s1_3], s1_4)
    s2 = jnp.select([b1, b2, b3], [s2_1, s2_2, s2_3], s2_4)
    t = jnp.select([b1, b2, b3], [t_1, t_2, t_3], t_4)
    # limit by the top layer's extractable water (lines 139-159)
    awev1 = jnp.maximum(0.0, (sw1 - ll1 * swef) * dlayr1 * MM_PER_CM)
    short = awev1 < es
    in2 = s1 >= u
    la = in2 & (s2 > es)
    lb = in2 & (s2 < es) & (s2 > 0.0)
    s2_a = s2 - es + awev1
    s1_b = s1 - (es - s2)
    s2_b = jnp.maximum(s1_b + awev1 - u, 0.0)
    s1_b = jnp.minimum(s1_b + awev1, u)
    s1_c = s1 - es + awev1
    s1n = jnp.select([la, lb], [s1, s1_b], s1_c)
    s2n = jnp.select([la, lb], [s2_a, s2_b], s2)
    t_a = (s2_a / c.stage2_rate) ** 2
    t_b = (s2_b / c.stage2_rate) ** 2
    tn = jnp.select([la, lb], [t_a, t_b], t)
    s1 = jnp.where(short, s1n, s1)
    s2 = jnp.where(short, s2n, s2)
    t = jnp.where(short, tn, t)
    es = jnp.where(short, awev1, es)
    # plastic mulch (lines 162-164), available water of layer 1 (lines 168-174)
    es = jnp.where(pm > c.pm_min, es * (1.0 - pm), es)
    swmin = jnp.maximum(0.0, sw_avail1 - swef * ll1)
    es = jnp.maximum(jnp.minimum(es, swmin * dlayr1 * MM_PER_CM), 0.0)
    return es, SoilevStore(s1, s2, t, swef)
