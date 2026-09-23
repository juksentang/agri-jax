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

__all__ = ["priestley_taylor"]

LANGLEY_PER_MJ_M2 = 23.923  # SLANG = SRAD * 23.923
EO_FLOOR_MM = 0.0001


def priestley_taylor(
    srad: ArrayLike, tmax: ArrayLike, tmin: ArrayLike, lai: ArrayLike, albedo_soil: ArrayLike
) -> Array:
    """DSSAT-CSM Priestley-Taylor potential ET [mm d-1] (``PETPT``).

    ``TD = 0.6 Tmax + 0.4 Tmin``; ``albedo = 0.23 - (0.23 - MSALB) exp(-0.75 LAI)`` (``MSALB`` when
    ``LAI <= 0``); equilibrium evaporation ``EEQ = 23.923 SRAD (2.04e-4 - 1.83e-4 albedo)(TD + 29)``
    (mm d-1, SRAD in langleys); ``EO = 1.1 EEQ``, replaced by ``EEQ ((Tmax - 35) 0.05 + 1.1)`` when
    ``Tmax > 35`` and by ``EEQ 0.01 exp(0.18 (Tmax + 20))`` when ``Tmax < 5``; floored at 1e-4 mm d-1
    (the reference model's ``MAX(EO, 0.0001)``).

    Known deviations: none (the branches are the reference model's; DSSAT single precision is
    not reproduced).

    Source: PETPT, dssat-csm-os ``SPAM/PET.for`` (subroutine at the ``SUBROUTINE PETPT`` block,
    lines 871-918 of the 4.8 source used here).
    """
    srad = jnp.asarray(srad, dtype=float)
    tmax = jnp.asarray(tmax, dtype=float)
    tmin = jnp.asarray(tmin, dtype=float)
    lai = jnp.asarray(lai, dtype=float)
    msalb = jnp.asarray(albedo_soil, dtype=float)

    td = 0.6 * tmax + 0.4 * tmin
    albedo = jnp.where(lai <= 0.0, msalb, 0.23 - (0.23 - msalb) * jnp.exp(-0.75 * lai))
    slang = srad * LANGLEY_PER_MJ_M2
    eeq = slang * (2.04e-4 - 1.83e-4 * albedo) * (td + 29.0)
    eo = eeq * 1.1
    eo = jnp.where(tmax > 35.0, eeq * ((tmax - 35.0) * 0.05 + 1.1), eo)
    eo = jnp.where(tmax < 5.0, eeq * 0.01 * jnp.exp(0.18 * (tmax + 20.0)), eo)
    return jnp.maximum(eo, EO_FLOOR_MM)
