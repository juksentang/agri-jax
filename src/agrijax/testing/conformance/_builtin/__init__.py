"""Conformance cases of the processes in this repository (synthetic inputs, no data).

One module per slot; each exposes ``cases() -> list[ConformanceCase]``. They live here, not in
``processes/<slot>/_conformance.py``, for two reasons: the source lint checks every function under
``processes/`` as a numerical kernel, so the literals of synthetic inputs would be AJ007 findings
there; and keeping them out of ``processes/`` keeps test inputs out of the slot packages and lets
the slot packages change without touching them. Discovery (:mod:`..discover`)
reads this list and the ``agrijax.conformance`` entry points through the same path.
"""

from __future__ import annotations

#: the case modules of this repository, in the order of the day (plant phase last)
MODULES: tuple[str, ...] = (
    "agrijax.testing.conformance._builtin.pet",
    "agrijax.testing.conformance._builtin.snow",
    "agrijax.testing.conformance._builtin.soil_water",
    "agrijax.testing.conformance._builtin.soil_water_conventions",
    "agrijax.testing.conformance._builtin.soil_water_bucket",
    "agrijax.testing.conformance._builtin.water_supply",
    "agrijax.testing.conformance._builtin.n_supply",
    "agrijax.testing.conformance._builtin.crop",
    "agrijax.testing.conformance._builtin.grids",
    "agrijax.testing.conformance._builtin.dssat_evap",
    "agrijax.testing.conformance._builtin.day_dssat486",
    "agrijax.testing.conformance._builtin.dssat_day_free",
)
