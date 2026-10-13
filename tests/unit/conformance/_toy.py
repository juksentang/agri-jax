"""A clean toy slot implementation for the kit's self-test: a crop-slot water bucket.

The bucket gains the day's rain and drains a fraction ``k * nstres`` of it, with ``nstres`` read
from the crop nitrogen port (P10, ``iface.crop_n.<slot>``). It follows every rule the kit checks,
so the positive control must pass every check; ``_toy_bad`` breaks one rule per fixture. The
module holds only declarations and the process (the kit lints it as kernel code).
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field
from agrijax.iface.crop import CropNIn

KEY = "crop/toy_bucket@none:kit_self"
_C = ("n_crop",)


class BucketCoefficients(Coefficients):
    """The drainage fraction of the toy bucket."""

    k: float = coef(
        0.1, "d-1", "fraction of the bucket drained per day", Provenance("none", paper="kit fixture")
    )


class BucketState(State):
    water: Array = field(unit="cm", description="bucket water", dims=_C)
    drained: Array = field(unit="cm d-1", description="water drained today", dims=_C)
    n_in: CropNIn = port(description="nitrogen stress scaling the drainage (read)")


class BucketParams(Params):
    coefficients: BucketCoefficients = field(description="drainage coefficients")


class BucketForcing(Forcing):
    rain: Array = field(unit="cm d-1", description="rain", dims=("T", "n_crop"))


def _bucket(state: BucketState, params: BucketParams, forcing_t: BucketForcing) -> BucketState:
    """``w = water + rain``; ``drained = k nstres w``; ``water = w - drained``.

    Source: kit self-test fixture.
    """
    w = state.water + forcing_t.rain
    d = params.coefficients.k * state.n_in.nstres * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


META = dict(provenance="equations_only", sources=(("bucket", "kit fixture"),), grid="point", deviates=())

bucket = process(
    _bucket,
    reads=("water", "n_in"),
    writes=("water", "drained"),
    register=False,
    name="toy_bucket",
    key=KEY,
    **META,
)
