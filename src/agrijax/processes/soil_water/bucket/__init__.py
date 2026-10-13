"""The DSSAT-CSM v4.8.6.0 multi-layer tipping bucket: a soil-water module of its own.

A first-class alternative to the RZWQM2 Richards day of :mod:`agrijax.processes.soil_water`, for
daily crop simulation, large scenario batches and fast calibration. Both sit in the ``soil_water``
slot and exchange water with the other modules through the same contract ports; what each module
supports, needs and books is declared in :data:`MODULE` (:mod:`.declaration`), so an assembly can
swap one for the other without rewriting the rest of the model. Parameters are **not**
interchangeable: the bucket takes DSSAT's ``SLLL``/``SDUL``/``SSAT``/``SSKS``/``SLDR``/``SLRO``,
Richards the Brooks-Corey curve of ``rzwqm.dat``.

* :mod:`.coefficients` - every number ``WATBAL`` and its callees hard-code, with file:line;
* :mod:`.kernels` - one pure kernel per Fortran routine (``SNOWFALL``, ``MULCHWATER``, ``RNOFF``,
  ``INFIL``, ``SATFLO``, ``UP_FLOW``, the integration);
* :mod:`.watbal` - the two registered processes (RATE and INTEGR), their records and the ledger;
* :mod:`.declaration` - the module declaration.

Translated from the DSSAT-CSM v4.8.6.0 Fortran (``Soil/SoilWater``, ``Soil/Mulch``) under its
BSD-3 licence: Copyright 1998-2026 DSSAT Foundation, University of Florida, International
Fertilizer Development Center; redistribution and use in source and binary forms, with or without
modification, are permitted provided the conditions of the licence are met (see
``THIRD_PARTY_NOTICES.md``). Validated day by day against the instrumented ``dscsm048``
(``tests/integration/test_bucket_dssat.py``).
"""

from __future__ import annotations

from .coefficients import WATBAL_COEFFICIENTS, WatbalCoefficients
from .declaration import MODULE
from .watbal import (
    BUCKET_LEDGER_INFLOWS,
    BUCKET_LEDGER_OUTFLOWS,
    BucketFluxes,
    BucketForcing,
    BucketParams,
    BucketSoil,
    BucketState,
    MulchForcing,
    bucket_integrate,
    bucket_ledger_channels,
    bucket_rate,
    bucket_storage,
)

__all__ = [
    "BUCKET_LEDGER_INFLOWS",
    "BUCKET_LEDGER_OUTFLOWS",
    "MODULE",
    "WATBAL_COEFFICIENTS",
    "BucketFluxes",
    "BucketForcing",
    "BucketParams",
    "BucketSoil",
    "BucketState",
    "MulchForcing",
    "WatbalCoefficients",
    "bucket_integrate",
    "bucket_ledger_channels",
    "bucket_rate",
    "bucket_storage",
]
