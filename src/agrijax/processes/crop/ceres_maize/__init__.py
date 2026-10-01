"""CERES-Maize ported from DSSAT-CSM v4.8.6.0 (MZ_PHENOL, MZ_GROSUB, MZ_ROOTGR), nitrogen off.

The crop reads its water through the ``water_in`` port (soil water, ``EOP``, ``TRWUP``) and computes
its water-stress factors itself; it reads its snow through the ``snow_in`` port and publishes a
root record (``root_out``) for the uptake producers. The port records are those of
:mod:`agrijax.iface`; the ``nstress_replay`` growth variant also reads the nitrogen factors
``NSTRES``, ``AGEFAC``, ``NDEF3`` and ``NPOOL`` from the ``n_in`` port. Season boundaries of a
multi-season run (sowing, harvest, the per-season parameter table) are in :mod:`.season`.

An independent implementation from the published equations of DSSAT-CSM ``Plant/CERES-Maize``
(BSD-3, Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center), validated against the ``dscsm048`` outputs.

The numbers DSSAT hard-codes in these routines are named, documented parameters
(:mod:`.coefficients`, ``CeresMaizeParams.coefficients``; ``None`` means the DSSAT values).
"""

from .canopy import (
    RZWQM2_CANOPY,
    CanopyCoefficients,
    CeresCanopyParams,
    ceres_canopy,
    published_lai,
    stalk_height,
)
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
    REPLAY_PROCESSES,
    ceres_crop_water_replay,
    ceres_maize_model,
    ceres_publish,
    ceres_snow_replay,
    ceres_water_replay,
    plantgro_outputs,
    root_record,
    yield_kg_ha,
)
from .phenology import ceres_phenology, thermal_time
from .roots import ceres_roots
from .season import ceres_harvest, ceres_season_init, check_sowing_dates, fresh_state
from .state import (
    N_STAGE_DATES,
    NOT_REACHED,
    CeresCultivar,
    CeresForcing,
    CeresGrowthState,
    CeresMaizeParams,
    CeresMaizeState,
    CeresPhenologyState,
    CeresReplayForcing,
    CeresRootState,
    CeresSeasons,
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
    "REPLAY_PROCESSES",
    "RZWQM2_CANOPY",
    "CanopyCoefficients",
    "CeresCanopyParams",
    "CeresCoefficients",
    "CeresCultivar",
    "CeresForcing",
    "CeresGrowthState",
    "CeresMaizeParams",
    "CeresMaizeState",
    "CeresPhenologyState",
    "CeresReplayForcing",
    "CeresRootState",
    "CeresSeasons",
    "CeresSoil",
    "CeresSpecies",
    "CeresStressState",
    "GrosubCoefficients",
    "PhenolCoefficients",
    "RootgrCoefficients",
    "ceres_canopy",
    "ceres_crop_water_replay",
    "ceres_growth",
    "ceres_growth_nstress_replay",
    "ceres_harvest",
    "ceres_maize_model",
    "ceres_phenology",
    "ceres_publish",
    "ceres_roots",
    "ceres_season_init",
    "ceres_snow_replay",
    "ceres_stress",
    "ceres_water_replay",
    "check_sowing_dates",
    "coefficient_table",
    "fresh_state",
    "plantgro_outputs",
    "published_lai",
    "root_record",
    "saturation_factor",
    "stalk_height",
    "thermal_time",
    "water_stress_factors",
    "yield_kg_ha",
]
