"""Evaporation of the surface mulch ``MULCH_EVAP`` and SPAM's mulch step (DSSAT-CSM v4.8.6.0).

With ``INFIL`` (``MEINF``) one of ``R``, ``S``, ``M`` and a potential soil evaporation above 1e-6 mm,
SPAM lets the mulch evaporate first (``SPAM.for:337-346``). Over the mulched fraction ``COVER``
the mulch area index ``MAI = AM 1e-5 MASS / COVER`` intercepts ``EOM = EOS (1 - exp(-EXTFAC
MAI))``, of which at most 85 % of the mulch water evaporates; scaled to the field, that is ``EM``.
``MULCH_EVAP`` returns ``EOS3 = min(EOS - EM, EOS exp(-EXTFAC MAI) COVER + EOS (1 - COVER))``
in place of its ``EOS`` argument, and SPAM then subtracts ``EM`` once more from what is left for
the soil (``EOS_SOIL = EOS3 - EM`` when positive, else 0): the reference's arithmetic, kept.

The mulch state (mass, cover, water, ``AM``, ``EXTFAC``) belongs to the residue / mulch module
(``Soil/Mulch/MULCHLAYER.for``); here it is an input.

Source: DSSAT-CSM v4.8.6.0 ``Soil/Mulch/MULCHEVAP.for`` lines 39-96 and ``SPAM/SPAM.for`` lines 334-346,
BSD-3 (Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center); Scopel, E. et al. (2004), Agronomie 24, 383-395.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from .coefficients import EVAP_COEFFICIENTS, EvapGateCoefficients, MulchEvapCoefficients

__all__ = ["MulchEvapResult", "mulch_evaporation", "spam_mulch_step"]


class MulchEvapResult(NamedTuple):
    """``MULCH_EVAP`` outputs: the mulch evaporation and its ``EOS`` argument on return [mm d-1]."""

    em: Array
    eos: Array


def mulch_evaporation(
    eos: ArrayLike,
    mass: ArrayLike,
    cover: ArrayLike,
    water: ArrayLike,
    am: ArrayLike,
    extfac: ArrayLike,
    c: MulchEvapCoefficients = EVAP_COEFFICIENTS.mulch,
) -> MulchEvapResult:
    """``MULCH_EVAP`` ``RATE``: ``(EM, EOS3)``; no mulch (``MASS <= 0.1`` kg ha-1): ``(0, EOS)``.

    ``eos`` [mm d-1], ``mass`` [kg ha-1], ``cover`` [-], ``water`` the mulch water [mm], ``am``
    (``MULCH_AM``) and ``extfac`` (``MUL_EXTFAC``) the mulch area coefficient and extinction.

    Source: DSSAT-CSM v4.8.6.0 Soil/Mulch/MULCHEVAP.for lines 39-96 (BSD-3).
    """
    eos, mass, cover, water, am, extfac = (jnp.asarray(x) for x in (eos, mass, cover, water, am, extfac))
    has_cover = cover > c.cover_min
    mai = jnp.where(has_cover, am * c.am_unit * mass / jnp.where(has_cover, cover, 1.0), 0.0)
    shade = jnp.exp(-extfac * mai)
    eom = eos * (1.0 - shade)
    em2 = jnp.maximum(jnp.minimum(eom, water * c.water_frac), 0.0) * cover
    eos3 = jnp.minimum(eos - em2, eos * shade * cover + eos * (1.0 - cover))
    on = mass > c.mass_min
    return MulchEvapResult(em=jnp.where(on, em2, 0.0), eos=jnp.where(on, eos3, eos))


def spam_mulch_step(
    eos: ArrayLike,
    mass: ArrayLike,
    cover: ArrayLike,
    water: ArrayLike,
    am: ArrayLike,
    extfac: ArrayLike,
    *,
    mulch_active: bool,
    c: MulchEvapCoefficients = EVAP_COEFFICIENTS.mulch,
    gate: EvapGateCoefficients = EVAP_COEFFICIENTS.gate,
) -> MulchEvapResult:
    """SPAM's mulch step: ``(EM, EOS_SOIL)`` from the potential soil evaporation ``EOS`` (no flood).

    ``mulch_active`` is the reference's ``INDEX('RSM', MEINF) > 0``. When the mulch evaporates,
    ``EOS_SOIL = EOS3 - EM`` if ``EOS3 > EM``, else 0; otherwise ``EM = 0``, ``EOS_SOIL = EOS``.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for lines 334-346 (BSD-3).
    """
    eos = jnp.asarray(eos)
    if not mulch_active:
        return MulchEvapResult(em=jnp.zeros_like(eos), eos=eos)
    r = mulch_evaporation(eos, mass, cover, water, am, extfac, c)
    left = jnp.where(r.eos > r.em, r.eos - r.em, 0.0)
    called = eos > gate.eos_min_mulch
    return MulchEvapResult(em=jnp.where(called, r.em, 0.0), eos=jnp.where(called, left, eos))
