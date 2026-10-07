"""Potential evapotranspiration: ASCE Penman-Monteith reference ET, Priestley-Taylor, DSSAT SPAM.

All functions are pure, ``vmap``-able and differentiable; see each module for units and sources.
Every coefficient of the equations is declared once, with unit, meaning and provenance, in
:mod:`.coefficients`.
"""

from ._precision import as_float
from .coefficients import (
    ASCE_2005,
    DSSAT_PT,
    PET_COEFFICIENTS,
    ASCECoefficients,
    PETCoefficients,
    PTCoefficients,
)
from .daily import (
    DailyWeather,
    PETFluxes,
    PETSiteParams,
    PETState,
    pet_asce_reference,
    pet_priestley_taylor,
)
from .eop import EOPState, eop_from_pet
from .penman_monteith import ASCE_SHORT, ASCE_TALL, ReferenceET, asce_reference_et, wind_to_2m
from .priestley_taylor import priestley_taylor

__all__ = [
    "ASCE_2005",
    "ASCE_SHORT",
    "ASCE_TALL",
    "DSSAT_PT",
    "PET_COEFFICIENTS",
    "ASCECoefficients",
    "DailyWeather",
    "EOPState",
    "PETCoefficients",
    "PETFluxes",
    "PETSiteParams",
    "PETState",
    "PTCoefficients",
    "ReferenceET",
    "as_float",
    "asce_reference_et",
    "eop_from_pet",
    "pet_asce_reference",
    "pet_priestley_taylor",
    "priestley_taylor",
    "wind_to_2m",
]
