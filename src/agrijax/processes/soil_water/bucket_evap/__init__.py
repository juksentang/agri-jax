"""DSSAT-CSM v4.8.6.0 soil and mulch evaporation of the soil_water slot (evaporation of the DSSAT day).

* :mod:`.ritchie` - Ritchie's two-stage ``SOILEV`` / ``ESUP`` (``MESEV = 'R'``);
* :mod:`.suleiman_ritchie` - the layered ``ESR_SoilEvap`` (``MESEV = 'S'``);
* :mod:`.mulch` - ``MULCH_EVAP`` and SPAM's mulch step;
* :mod:`.process` - the registered processes ``soil_water/{mulch_evap, soilev,
  esr_soilevap}@dssat-4.8.6.0:faithful`` on :class:`~.process.SoilEvapState`;
* :mod:`.coefficients` - every coefficient with its source line and statement;
* :mod:`.xtract` - the actual transpiration ``EP`` and the root extraction ``XTRACT``
  (``soil_water/xtract@dssat-4.8.6.0:faithful``, writes the sink record P4);
* :mod:`.albedo` - the daily soil albedo ``MSALB`` of ``SOILDYN`` (``ALBEDO_avg``;
  ``soil_water/soil_albedo@dssat-4.8.6.0:faithful``).

The potential soil evaporation they consume comes from the pet slot's SPAM partition
(:mod:`agrijax.processes.pet.spam_dssat`) through port P5; the vertical water balance that
applies their fluxes is the bucket module (:mod:`agrijax.processes.soil_water.bucket`). Both are joined
in the assembled DSSAT day.

Source: DSSAT-CSM v4.8.6.0 (BSD-3, Copyright 1998-2026 DSSAT Foundation, University of Florida,
International Fertilizer Development Center).
"""

from .albedo import (
    ALBEDO_COEFFICIENTS,
    AlbedoCoefficients,
    SoilAlbedoParams,
    SoilAlbedoState,
    soil_albedo,
    soil_albedo_rate,
)
from .coefficients import (
    EVAP_COEFFICIENTS,
    EsrCoefficients,
    EvapCoefficients,
    EvapGateCoefficients,
    MulchEvapCoefficients,
    SoilevCoefficients,
    evap_coefficient_table,
)
from .mulch import MulchEvapResult, mulch_evaporation, spam_mulch_step
from .process import (
    SoilEvapParams,
    SoilEvapState,
    soil_evaporation_esr,
    soil_evaporation_mulch,
    soil_evaporation_soilev,
)
from .ritchie import SoilevStore, soilev_init, soilev_rate
from .suleiman_ritchie import EsrResult, esr_soil_evaporation
from .xtract import (
    XTRACT_COEFFICIENTS,
    XtractCoefficients,
    XtractParams,
    XtractResult,
    XtractState,
    actual_transpiration,
    extraction_supply,
    spam_xtract,
    xtract,
)

__all__ = [
    "ALBEDO_COEFFICIENTS",
    "EVAP_COEFFICIENTS",
    "XTRACT_COEFFICIENTS",
    "AlbedoCoefficients",
    "EsrCoefficients",
    "EsrResult",
    "EvapCoefficients",
    "EvapGateCoefficients",
    "MulchEvapCoefficients",
    "MulchEvapResult",
    "SoilAlbedoParams",
    "SoilAlbedoState",
    "SoilEvapParams",
    "SoilEvapState",
    "SoilevCoefficients",
    "SoilevStore",
    "XtractCoefficients",
    "XtractParams",
    "XtractResult",
    "XtractState",
    "actual_transpiration",
    "esr_soil_evaporation",
    "evap_coefficient_table",
    "extraction_supply",
    "mulch_evaporation",
    "soil_albedo",
    "soil_albedo_rate",
    "soil_evaporation_esr",
    "soil_evaporation_mulch",
    "soil_evaporation_soilev",
    "soilev_init",
    "soilev_rate",
    "spam_mulch_step",
    "spam_xtract",
    "xtract",
]
