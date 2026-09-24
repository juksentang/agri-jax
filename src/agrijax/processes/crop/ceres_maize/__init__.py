"""CERES-Maize ported from DSSAT-CSM v4.8.6.0 (MZ_PHENOL, MZ_GROSUB, MZ_ROOTGR), nitrogen off.

An independent implementation from the published equations of DSSAT-CSM ``Plant/CERES-Maize``
(BSD-3, Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center), validated against the ``dscsm048`` outputs.

The numbers DSSAT hard-codes in these routines are named, documented parameters
(:mod:`.coefficients`, ``CeresMaizeParams.coefficients``; ``None`` means the DSSAT values).
"""

from .coefficients import (
    DSSAT_COEFFICIENTS,
    CeresCoefficients,
    GrosubCoefficients,
    PhenolCoefficients,
    RootgrCoefficients,
    coefficient_table,
)
from .growth import ceres_growth, ceres_stress, saturation_factor, water_stress_factors
from .model import OUTPUT_UNITS, ceres_maize_model, plantgro_outputs, yield_kg_ha
from .phenology import ceres_phenology, thermal_time
from .roots import ceres_roots
from .state import (
    N_STAGE_DATES,
    NOT_REACHED,
    CeresCultivar,
    CeresForcing,
    CeresGrowthState,
    CeresMaizeParams,
    CeresMaizeState,
    CeresPhenologyState,
    CeresRootState,
    CeresSoil,
    CeresSpecies,
    CeresStressState,
)

__all__ = [
    "DSSAT_COEFFICIENTS",
    "NOT_REACHED",
    "N_STAGE_DATES",
    "OUTPUT_UNITS",
    "CeresCoefficients",
    "CeresCultivar",
    "CeresForcing",
    "CeresGrowthState",
    "CeresMaizeParams",
    "CeresMaizeState",
    "CeresPhenologyState",
    "CeresRootState",
    "CeresSoil",
    "CeresSpecies",
    "CeresStressState",
    "GrosubCoefficients",
    "PhenolCoefficients",
    "RootgrCoefficients",
    "ceres_growth",
    "ceres_maize_model",
    "ceres_phenology",
    "ceres_roots",
    "ceres_stress",
    "coefficient_table",
    "plantgro_outputs",
    "saturation_factor",
    "thermal_time",
    "water_stress_factors",
    "yield_kg_ha",
]
