"""CERES-Maize water and excess-water stress: the ``MZ_GROSUB`` stress block.

:func:`ceres_stress` computes ``SWFAC`` and ``TURFAC`` from the ``water_in`` port
(:func:`water_stress_factors`) and the excess-water factor ``SATFAC`` with its saturation-day
counters (:func:`saturation_factor`); :class:`WaterStressCoefficients` holds the numbers of the
crop's water-stress interface. Split out of :mod:`agrijax.processes.crop.ceres_maize.growth`,
which re-exports every public name.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_GROSUB.for`` and ``MZ_CERES.for`` (BSD-3,
Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef, numerical_guard
from agrijax.core.process import process
from agrijax.core.units import mm_to_cm

from ._util import coef_div, safe_div, trunc_st
from .coefficients import DSSAT_COEFFICIENTS
from .state import CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = [
    "WATER_STRESS_COEFFICIENTS",
    "WaterStressCoefficients",
    "ceres_stress",
    "saturation_factor",
    "water_stress_factors",
]

_EPS = numerical_guard(
    "ceres_maize.eps",
    1e-12,
    "floor of the divisors and power bases of the growth and stress kernels (RWUEP1, PORMIN, PHINT, "
    "the organ demand, the canopy-height denominator)",
)


def _ws(value: float, description: str, file_line: str, routine: str, statement: str, **kw: Any) -> Any:
    """A coefficient of the crop's water-stress interface (DSSAT-CSM v4.8.6.0, not calibrated)."""
    prov = Provenance.at(
        "dssat-4.8.6.0", f"Plant/CERES-Maize/{file_line}", routine=routine, statement=statement
    )
    return coef(value, "-", description, prov, **kw)


class WaterStressCoefficients(Coefficients):
    """The numbers of the crop's water-stress interface (``SWFAC``, ``TURFAC`` from ``EOP`` and
    ``TRWUP``; the stages in which ``MZ_CERES`` calls the stress block).

    Neither is a response to calibrate: ``turfac_scale`` is the storage precision DSSAT gives
    ``TURFAC`` (a leaf kept out of the calibration vector), ``grosub_last_stage`` a stage code
    (static). The ``0.1`` of ``EP1 = EOP * 0.1`` is the mm -> cm conversion
    (:func:`agrijax.core.units.mm_to_cm`).
    """

    turfac_scale: float = _ws(
        1000.0,
        "TURFAC is truncated to 1 / turfac_scale (REAL(INT(TURFAC*1000))/1000)",
        "MZ_GROSUB.for:1066",
        "MZ_GROSUB",
        "TURFAC = REAL(INT(TURFAC*1000))/1000",
        calibrate=False,
    )
    grosub_last_stage: int = _ws(
        6,
        "last stage (ISTAGE) in which MZ_CERES runs MZ_GROSUB and its stress block",
        "MZ_CERES.for:658",
        "MZ_CERES",
        "IF (ISTAGE .GT. 0 .AND. ISTAGE .LE. 6) THEN",
        static=True,
    )


#: the DSSAT-CSM v4.8.6.0 values of :class:`WaterStressCoefficients`
WATER_STRESS_COEFFICIENTS = WaterStressCoefficients()


def water_stress_factors(
    eop: ArrayLike,
    trwup: ArrayLike,
    rwuep1: ArrayLike,
    c: WaterStressCoefficients = WATER_STRESS_COEFFICIENTS,
) -> tuple[Array, Array]:
    """``(SWFAC, TURFAC)`` from potential transpiration ``EOP`` [mm d-1] and potential root water
    uptake ``TRWUP`` [cm d-1].

    ``EP1 = 0.1 EOP``; ``TURFAC = TRWUP / (RWUEP1 EP1)`` when that ratio is below 1 and
    ``SWFAC = TRWUP / EP1`` when ``EP1 >= TRWUP``, both 1 without demand; ``TURFAC`` is then
    truncated to 1e-3 (``1 / c.turfac_scale``) as in the Fortran (value exact, identity
    derivative, :func:`trunc_st`).

    Source: DSSAT-CSM MZ_GROSUB.for, "Compute Water Stress Factors".
    """
    eop = jnp.asarray(eop)
    trwup = jnp.asarray(trwup)
    ep1 = mm_to_cm(eop)  # EP1 = EOP * 0.1
    demand = eop > 0.0
    ratio = safe_div(trwup, ep1, 1.0)
    turfac = jnp.where(demand & (ratio < rwuep1), ratio / jnp.maximum(jnp.asarray(rwuep1), _EPS), 1.0)
    swfac = jnp.where(demand & (ep1 >= trwup), ratio, 1.0)
    return swfac, coef_div(trunc_st(turfac * c.turfac_scale), c.turfac_scale)


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
    reads=("phen.istage", "phen.mdate", "stress", "roots.rlv", "water_in"),
    writes=("stress",),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, MZ_CERES.for (BSD-3)",
    fortran_name="MZ_GROSUB",
    key="crop/ceres_maize.stress@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        (
            "water-stress factors SWFAC, TURFAC from EOP, TRWUP and RWUEP1",
            "MZ_GROSUB.for, Compute Water Stress Factors; MZ_CERES.for reset",
        ),
        (
            "excess-water factor SATFAC and saturation-day counters TSS",
            "MZ_GROSUB.for, Compute Water Saturation Factors",
        ),
    ),
    deviates=(
        (
            "DSSAT single precision (REAL*4) is not reproduced",
            "the kernels run in float64 (float32 with AGRI_JAX_X64=0)",
            "tests/integration/test_ceres_dssat.py tolerances",
        ),
    ),
)
def ceres_stress(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """Water and excess-water stress factors of the day (the ``MZ_GROSUB`` stress block).

    ``SWFAC`` and ``TURFAC`` come from today's ``EOP`` and ``TRWUP`` in the ``water_in`` port
    (:func:`water_stress_factors` with ``RWUEP1``). Runs when ``MZ_GROSUB`` runs (stages 1-6) and
    the day is not the maturity / failure day. Outside stages 1-6 ``MZ_CERES`` resets ``SWFAC`` to
    1. Without a water balance both factors are 1. ``SATFAC`` uses yesterday's root length
    density (roots grow after growth) and the port's soil water.

    Source: DSSAT-CSM v4.8.6.0 MZ_GROSUB.for (INTEGR stress block) and MZ_CERES.for (BSD-3).
    """
    s = state.phen.istage
    st = state.stress
    w = state.water_in
    yrdoy = jnp.asarray(forcing_t.yrdoy)
    called = (s >= 1) & (s <= WATER_STRESS_COEFFICIENTS.grosub_last_stage)
    run = called & (state.phen.mdate != yrdoy)
    wat = params.iswwat
    one = jnp.ones_like(st.swfac)
    sw_calc, tu_calc = water_stress_factors(w.eop, w.trwup, params.species.rwuep1)
    sw_in = jnp.where(wat, sw_calc * one, one)
    tu_in = jnp.where(wat, tu_calc * one, one)
    swfac = jnp.where(run, sw_in, jnp.where(called, st.swfac, 1.0))
    turfac = jnp.where(run, tu_in, st.turfac)
    satfac_new, tss_new = saturation_factor(
        w.sw,
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
