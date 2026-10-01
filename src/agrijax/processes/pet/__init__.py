"""Potential evapotranspiration: Shuttleworth-Wallace (RZWQM2), ASCE Penman-Monteith, Priestley-Taylor.

All functions are pure, ``vmap``-able and differentiable; see each module for units and sources.
Every coefficient of the equations is declared once, with unit, meaning and provenance, in
:mod:`.coefficients`.
"""

from ._precision import as_float
from .coefficients import (
    ASCE_2005,
    DSSAT_PT,
    PET_COEFFICIENTS,
    RZWQM_SW,
    ASCECoefficients,
    PETCoefficients,
    PTCoefficients,
    ***REMOVED***,
)
from .daily import (
    RESIDUE_KINDS,
    DailyWeather,
    PETFluxes,
    PETSiteParams,
    PETState,
    SurfaceResidue,
    pet_asce_reference,
    pet_priestley_taylor,
    pet_shuttleworth_wallace,
)
from .eop import EOPState, eop_from_pet
from .penman_monteith import ASCE_SHORT, ASCE_TALL, ReferenceET, asce_reference_et, wind_to_2m
from .priestley_taylor import priestley_taylor
from .shuttleworth_wallace import (
    AerodynamicResistances,
    ClearSkyRadiation,
    EnergyConstants,
    NetRadiation,
    PETParams,
    SWResult,
    WindAdjustment,
    clear_sky_radiation,
    energy_constants,
    net_radiation,
    residue_albedo,
    resistances,
    saturation_vapour_pressure,
    shuttleworth_wallace,
    soil_albedo,
    wind_adjustment,
)

__all__ = [
    "ASCE_2005",
    "ASCE_SHORT",
    "ASCE_TALL",
    "DSSAT_PT",
    "PET_COEFFICIENTS",
    "RESIDUE_KINDS",
    "RZWQM_SW",
    "ASCECoefficients",
    "AerodynamicResistances",
    "ClearSkyRadiation",
    "DailyWeather",
    "EOPState",
    "EnergyConstants",
    "NetRadiation",
    "PETCoefficients",
    "PETFluxes",
    "PETParams",
    "PETSiteParams",
    "PETState",
    "PTCoefficients",
    "ReferenceET",
    "***REMOVED***",
    "SWResult",
    "SurfaceResidue",
    "WindAdjustment",
    "as_float",
    "asce_reference_et",
    "clear_sky_radiation",
    "energy_constants",
    "eop_from_pet",
    "net_radiation",
    "pet_asce_reference",
    "pet_priestley_taylor",
    "pet_shuttleworth_wallace",
    "priestley_taylor",
    "residue_albedo",
    "resistances",
    "saturation_vapour_pressure",
    "shuttleworth_wallace",
    "soil_albedo",
    "wind_adjustment",
    "wind_to_2m",
]
