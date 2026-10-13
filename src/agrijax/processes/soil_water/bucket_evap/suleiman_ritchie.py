"""The Suleiman-Ritchie layered soil evaporation ``ESR_SoilEvap`` (DSSAT-CSM v4.8.6.0, ``MESEV = 'S'``).

Every layer loses a fraction of its water above the air-dry content ``SWAD = 0.30 LL`` each day,
``SWDELTU = -(SWTEMP - SWAD) ES_Coef``, with ``SWTEMP`` the day's content (``SW`` plus half of a
positive, or all of a negative, drainage change ``SWDELTS``). The coefficient depends on the
profile: *wet* (a layer with mean depth < 100 cm is above DUL) ``0.26 MEANDEP^-0.70``;
*intermediate* (wet, but the top layer below the threshold ``0.275 DUL + 1.165 DUL^2 + 1.2
DUL^3.75 MEANDEP``) 0.011 everywhere; *dry* ``(0.5 + 0.24 DUL) MEANDEP^(-2.04 + 0.20 DUL)``. The
loss is limited to the layer's available water and the profile sum to the potential soil
evaporation ``EOS`` (proportional reduction). ``UPFLOW(L)`` is the evaporative flux through the
top of layer ``L`` [cm d-1], the sum of the layer losses from ``L`` down.

This routine replaces ``SOILEV`` and the bucket's ``UP_FLOW`` (WATBAL does not call ``UP_FLOW``
with ``MESEV = 'S'``, ``Soil/SoilWater/WATBAL.for:413-424``); the bucket adds ``SWDELTU`` to the
layers (``WATBAL.for:490-495``). The routine has no state.

The layer sums are vectorised over the trailing layer axis (no Python loop over layers); every
power has a clamped base, so every branch is finite.

Source: DSSAT-CSM v4.8.6.0 ``SPAM/ESR_SoilEvap.for`` lines 33-183 and ``SPAM/SPAM.for`` lines 352-356,
BSD-3 (Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center); Suleiman, A.A. and Ritchie, J.T. (2003), Soil Sci. Soc. Am. J. 67, 377-386;
Ritchie, J.T. et al. (2009), Soil Sci. Soc. Am. J. 73, 792-801.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard
from agrijax.core.units import MM_PER_CM

from .coefficients import EVAP_COEFFICIENTS, EsrCoefficients

__all__ = ["EsrResult", "esr_soil_evaporation"]

#: mean depth of a layer: its bottom less half its thickness (a geometric constant)
_HALF = 0.5
# layer mean depths are > 0 cm and DUL > 0 in a valid soil; the floors keep the powers finite
_DEPTH_MIN = numerical_guard(
    "esr.depth_min", 1e-6, "floor of the layer mean depth [cm] under a negative power"
)
_DUL_MIN = numerical_guard("esr.dul_min", 1e-12, "floor of DUL(1) under the power 3.75 of the threshold")
_ES_MIN = numerical_guard("esr.es_min", 1e-30, "floor of the profile evaporation in the reduction EOS / ES")

#: profile types of the reference (``ProfileType``)
WET, INTERMEDIATE, DRY = 1, 2, 3


class EsrResult(NamedTuple):
    """Result of :func:`esr_soil_evaporation` (layer arrays ``[..., n_layer]``)."""

    es: Array  # mm d-1, actual soil evaporation
    es_lyr: Array  # mm d-1, evaporation from each layer
    swdeltu: Array  # cm3 cm-3 d-1, change of the layer water content (<= 0)
    upflow: Array  # cm d-1, evaporative flux through the top of each layer
    profile: Array  # 1 wet, 2 intermediate, 3 dry


def esr_soil_evaporation(
    eos: ArrayLike,
    sw: ArrayLike,
    swdelts: ArrayLike,
    dlayr: ArrayLike,
    ds: ArrayLike,
    dul: ArrayLike,
    ll: ArrayLike,
    pmfraction: ArrayLike = 0.0,
    c: EsrCoefficients = EVAP_COEFFICIENTS.esr,
) -> EsrResult:
    """``ESR_SoilEvap``: layered actual soil evaporation for potential ``eos`` [mm d-1].

    Layer arrays (``[..., n_layer]``, the soil's ``NLAYR`` layers): ``sw`` water content,
    ``swdelts`` today's drainage change (cm3 cm-3), ``dlayr`` thickness and ``ds`` bottom depth
    (cm), ``dul`` / ``ll`` drained upper and lower limits; ``pmfraction`` plastic-mulch cover.

    Source: DSSAT-CSM v4.8.6.0 SPAM/ESR_SoilEvap.for lines 79-179 (BSD-3).
    """
    eos, sw, swdelts, dlayr, ds, dul, ll, pm = (
        jnp.asarray(x) for x in (eos, sw, swdelts, dlayr, ds, dul, ll, pmfraction)
    )
    swad = c.air_dry_frac * ll
    meandep = ds - dlayr * _HALF
    swtemp = jnp.where(swdelts > 0.0, sw + c.pseudo_integration * swdelts, sw + swdelts)
    wet = jnp.any((meandep < c.wet_depth) & (swtemp > dul), axis=-1)
    dul1, md1 = dul[..., 0], meandep[..., 0]
    thr = c.thr_a * dul1 + c.thr_b * dul1 * dul1 + (c.thr_c * jnp.maximum(dul1, _DUL_MIN) ** c.thr_exp) * md1
    inter = wet & (swtemp[..., 0] < thr)
    profile = jnp.where(wet, jnp.where(inter, INTERMEDIATE, WET), DRY)
    md = jnp.maximum(meandep, _DEPTH_MIN)
    a_dry = c.dry_a0 + c.dry_a1 * dul
    b_dry = c.dry_b0 + c.dry_b1 * dul
    coef_dry = a_dry * md**b_dry
    coef_wet = c.wet_a * md**c.wet_b
    pw = profile[..., None]
    es_coef = jnp.where(pw == DRY, coef_dry, jnp.where(pw == INTERMEDIATE, c.equilibrium_coef, coef_wet))
    swdeltu = -(swtemp - swad) * es_coef
    swdeltu = jnp.where(pm[..., None] > c.pm_min, swdeltu * (1.0 - pm[..., None]), swdeltu)
    sw_avail = sw + swdelts - swad
    swdeltu = jnp.where(-swdeltu > sw_avail, -sw_avail, swdeltu)
    swdeltu = jnp.minimum(0.0, swdeltu)
    es_lyr = -swdeltu * dlayr * MM_PER_CM
    es = jnp.sum(es_lyr, axis=-1)
    over = es > eos
    red = jnp.where(over, eos / jnp.maximum(es, _ES_MIN), 1.0)
    es_lyr = jnp.where(over[..., None], es_lyr * red[..., None], es_lyr)
    swdeltu = jnp.where(over[..., None], swdeltu * red[..., None], swdeltu)
    es = jnp.where(over, eos, es)
    upflow = jnp.flip(jnp.cumsum(jnp.flip(es_lyr / MM_PER_CM, axis=-1), axis=-1), axis=-1)
    return EsrResult(es=es, es_lyr=es_lyr, swdeltu=swdeltu, upflow=upflow, profile=profile)
