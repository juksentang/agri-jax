"""CERES-Maize ported from DSSAT-CSM v4.8.6.0 (MZ_PHENOL, MZ_GROSUB, MZ_ROOTGR), nitrogen off.

The crop reads its water through the ``water_in`` port (soil water, ``EOP``, ``TRWUP``) and computes
its water-stress factors itself; it publishes a root record (``root_out``) for the uptake
producers. The port records are those of :mod:`agrijax.iface.crop`; the ``nstress_replay`` growth
variant also reads the nitrogen stress ``NSTRES`` from the ``n_in`` port.

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
from .growth import (
    ceres_growth,
    ceres_growth_nstress_replay,
    ceres_stress,
    saturation_factor,
    water_stress_factors,
)
from .model import (
    CROP_PROCESSES,
    CROP_PROCESSES_NSTRESS_REPLAY,
    OUTPUT_UNITS,
    ceres_maize_model,
    ceres_publish,
    ceres_water_replay,
    plantgro_outputs,
    yield_kg_ha,
)
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
    "CROP_PROCESSES",
    "CROP_PROCESSES_NSTRESS_REPLAY",
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
    "ceres_growth_nstress_replay",
    "ceres_maize_model",
    "ceres_phenology",
    "ceres_publish",
    "ceres_roots",
    "ceres_stress",
    "ceres_water_replay",
    "coefficient_table",
    "plantgro_outputs",
    "saturation_factor",
    "thermal_time",
    "water_stress_factors",
    "yield_kg_ha",
]
