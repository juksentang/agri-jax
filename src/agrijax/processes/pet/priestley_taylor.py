"""Priestley-Taylor potential evapotranspiration as in DSSAT-CSM (``PETPT``, Ritchie).

Independent JAX implementation of the DSSAT-CSM default PET method (``SPAM/PET.for``,
subroutine ``PETPT``, J.T. Ritchie; MEEVP = 'R'), following

* Priestley, C.H.B. and Taylor, R.J. (1972). On the assessment of surface heat flux and
  evaporation using large-scale parameters. Mon. Weather Rev. 100, 81-92.
* Ritchie, J.T. (1972). Model for predicting evaporation from a row crop with incomplete
  cover. Water Resour. Res. 8, 1204-1213 (the CERES form with the empirical
  ``(TD + 29)`` temperature factor and the hot / cold corrections).

Units: ``srad`` MJ m-2 d-1, temperatures degC, ``lai`` m2 m-2, ``albedo_soil`` (DSSAT ``MSALB``,
the soil albedo including mulch and soil-water effects); the result ``EO`` is in **mm d-1**
(DSSAT's unit; RZWQM's S-W outputs are in cm d-1).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from .coefficients import DSSAT_PT, PTCoefficients

__all__ = ["DSSAT_PT", "PTCoefficients", "priestley_taylor"]

LANGLEY_PER_MJ_M2 = 23.923  # SLANG = SRAD * 23.923 (unit conversion, as DSSAT writes it)
#: read-only alias of the default EO floor (:class:`~.coefficients.PTCoefficients`)
EO_FLOOR_MM = DSSAT_PT.eo_floor


def priestley_taylor(
    srad: ArrayLike,
    tmax: ArrayLike,
    tmin: ArrayLike,
    lai: ArrayLike,
    albedo_soil: ArrayLike,
    coefficients: PTCoefficients = DSSAT_PT,
) -> Array:
    """DSSAT-CSM Priestley-Taylor potential ET [mm d-1] (``PETPT``).

    ``TD = 0.6 Tmax + 0.4 Tmin``; ``albedo = 0.23 - (0.23 - MSALB) exp(-0.75 LAI)`` (``MSALB`` when
    ``LAI <= 0``); equilibrium evaporation ``EEQ = 23.923 SRAD (2.04e-4 - 1.83e-4 albedo)(TD + 29)``
    (mm d-1, SRAD in langleys); ``EO = 1.1 EEQ``, replaced by ``EEQ ((Tmax - 35) 0.05 + 1.1)`` when
    ``Tmax > 35`` and by ``EEQ 0.01 exp(0.18 (Tmax + 20))`` when ``Tmax < 5``; floored at 1e-4 mm d-1
    (the reference model's ``MAX(EO, 0.0001)``).

    Known deviations: none (the branches are the reference model's; DSSAT single precision is
    not reproduced). ``coefficients`` (:class:`~.coefficients.PTCoefficients`, default
    :data:`~.coefficients.DSSAT_PT`) holds every coefficient; array leaves make them differentiable.

    Source: PETPT, dssat-csm-os ``SPAM/PET.for`` (subroutine at the ``SUBROUTINE PETPT`` block,
    lines 871-918 of the 4.8 source used here).
    """
    srad = jnp.asarray(srad, dtype=float)
    tmax = jnp.asarray(tmax, dtype=float)
    tmin = jnp.asarray(tmin, dtype=float)
    lai = jnp.asarray(lai, dtype=float)
    msalb = jnp.asarray(albedo_soil, dtype=float)

    c = coefficients
    td = c.td_tmax_weight * tmax + c.td_tmin_weight * tmin
    albedo = jnp.where(
        lai <= 0.0, msalb, c.canopy_albedo - (c.canopy_albedo - msalb) * jnp.exp(-c.albedo_lai_decay * lai)
    )
    slang = srad * LANGLEY_PER_MJ_M2
    eeq = slang * (c.eeq_a - c.eeq_b * albedo) * (td + c.eeq_t_offset)
    eo = eeq * c.alpha
    eo = jnp.where(tmax > c.hot_threshold, eeq * ((tmax - c.hot_threshold) * c.hot_slope + c.alpha), eo)
    eo = jnp.where(
        tmax < c.cold_threshold, eeq * c.cold_factor * jnp.exp(c.cold_exp * (tmax + c.cold_offset)), eo
    )
    return jnp.maximum(eo, c.eo_floor)
