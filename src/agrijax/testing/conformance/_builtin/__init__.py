"""Conformance cases of the processes in this repository (synthetic inputs, no data).

One module per slot; each exposes ``cases() -> list[ConformanceCase]``. They live here, not in
``processes/<slot>/_conformance.py`` (M3 coupling contract, section 4.2), for two reasons: the
source lint checks every function under ``processes/`` as a numerical kernel, so the literals of
synthetic inputs would be AJ007 findings there; and keeping them out of ``processes/`` leaves the
slot packages free for the parallel work lines that change them. Discovery (:mod:`..discover`)
reads this list and the ``agrijax.conformance`` entry points through the same path.
"""

from __future__ import annotations

#: the case modules of this repository, in the order of the day (plant phase last)
MODULES: tuple[str, ...] = (
    "agrijax.testing.conformance._builtin.pet",
    "agrijax.testing.conformance._builtin.soil_water",
    "agrijax.testing.conformance._builtin.water_supply",
    "agrijax.testing.conformance._builtin.n_supply",
    "agrijax.testing.conformance._builtin.crop",
    "agrijax.testing.conformance._builtin.grids",
)
