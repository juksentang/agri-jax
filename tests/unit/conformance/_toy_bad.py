"""Known-bad variants of the toy bucket: each breaks exactly one rule a kit check guards.

Each process is registered with ``register=False`` and a key of its own, so nothing leaks into the
registry. Every variant declares its writes correctly unless the broken rule is the writes (the CI
unit tier runs with ``AGRI_JAX_CHECK=1``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Params, State, field
from agrijax.iface.crop import CropWaterIn

from ._toy import META, BucketCoefficients

_RNG = np.random.default_rng(0)
#: the drainage fraction at which the ``jump`` fixture's step sits (just above the default 0.1)
K_JUMP = 0.1 * (1.0 + 1e-11)


class OtherRefCoefficients(Coefficients):
    """A coefficient citing a reference the key does not follow, without a note."""

    k: float = coef(0.1, "d-1", "drainage fraction", Provenance("fao56-1998", paper="FAO-56"))


class BoundedCoefficients(Coefficients):
    """A drainage fraction with physical bounds (a case can still pass a value outside them)."""

    k: float = coef(
        0.1, "d-1", "drainage fraction", Provenance("none", paper="kit fixture"), bounds=(0.0, 0.5)
    )


#: offset of the ``cancel`` fixture per unit of ``k`` (1e5 cm at k = 0.1): float32 keeps ~2
#: decimals of water around it. Built from a parameter so XLA cannot fold ``(w + c) - c``.
BIG_PER_K = 1.0e6


def _key(name: str) -> str:
    return f"crop/toy_{name}@none:kit_self"


def _proc(fn, name, reads=("water", "n_in"), writes=("water", "drained"), key=None, **kw):
    meta = {**META, **kw}
    return process(
        fn, reads=reads, writes=writes, register=False, name=f"toy_{name}", key=key or _key(name), **meta
    )


def _drain(state, params):
    return params.coefficients.k * state.n_in.nstres


def _python_if(state, params, forcing_t):
    """AJ001: a Python branch on the state.

    Source: kit self-test fixture.
    """
    w = state.water + forcing_t.rain
    if w.sum() > 0:
        w = w * 1.0
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _good_body(state, params, forcing_t):
    """The toy bucket.

    Source: kit self-test fixture.
    """
    w = state.water + forcing_t.rain
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _no_source(state, params, forcing_t):
    """The toy bucket, but this docstring cites nothing."""
    return _good_body(state, params, forcing_t)


def _undeclared_write(state, params, forcing_t):
    """Writes ``drained`` without declaring it.

    Source: kit self-test fixture.
    """
    return _good_body(state, params, forcing_t)


def _undeclared_read(state, params, forcing_t):
    """Reads ``n_in`` without declaring it.

    Source: kit self-test fixture.
    """
    return _good_body(state, params, forcing_t)


def _writes_in_port(state, params, forcing_t):
    """Writes the nitrogen record it only may read.

    Source: kit self-test fixture.
    """
    s = _good_body(state, params, forcing_t)
    return eqx.tree_at(lambda x: x.n_in.nstres, s, s.n_in.nstres * 0.5)


def _leak(state, params, forcing_t):
    """Loses the drained water twice (not conservative).

    Source: kit self-test fixture.
    """
    w = state.water + forcing_t.rain
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - 2.0 * d, d))


def _impure(state, params, forcing_t):
    """Draws a Python-side random number at trace time (jit freezes it, eager does not).

    Source: kit self-test fixture.
    """
    w = state.water + forcing_t.rain + float(_RNG.uniform())
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _nonfinite(state, params, forcing_t):
    """Returns a NaN.

    Source: kit self-test fixture.
    """
    w = state.water + forcing_t.rain
    d = _drain(state, params) * w + jnp.log(-jnp.abs(w) - 1.0)
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _upcast(state, params, forcing_t):
    """Returns float64 from float32 inputs.

    Source: kit self-test fixture.
    """
    w = (state.water + forcing_t.rain).astype(jnp.float64)
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _nan_grad(state, params, forcing_t):
    """A where whose unselected branch has a NaN derivative.

    Source: kit self-test fixture.
    """
    k = params.coefficients.k
    w = state.water + forcing_t.rain
    frac = jnp.where(k > 0.0, k, jnp.sqrt(-k)) * state.n_in.nstres
    d = frac * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


@jax.custom_jvp
def _wrong(k):
    return k


@_wrong.defjvp
def _wrong_jvp(primals, tangents):
    (k,), (t,) = primals, tangents
    return k, 2.0 * t


def _wrong_grad(state, params, forcing_t):
    """A custom derivative twice the true one.

    Source: kit self-test fixture.
    """
    w = state.water + forcing_t.rain
    d = _wrong(params.coefficients.k) * state.n_in.nstres * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _jump(state, params, forcing_t):
    """A step in ``k`` right next to the default value (a kink for finite differences).

    Source: kit self-test fixture.
    """
    k = params.coefficients.k
    w = state.water + forcing_t.rain
    d = (k + jnp.where(k > K_JUMP, 0.05, 0.0)) * state.n_in.nstres * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _cancel(state, params, forcing_t):
    """Adds and removes a large offset: float32 loses the water's low digits (catastrophic cancellation).

    Source: kit self-test fixture.
    """
    big = params.coefficients.k * BIG_PER_K
    w = (state.water + forcing_t.rain + big) - big
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _batch_mean_host(w):
    w = np.asarray(w)
    if w.ndim == 1:  # one sample
        return w
    return np.broadcast_to(w.mean(axis=0, keepdims=True), w.shape).astype(w.dtype)  # a batch


def _batch_mixing(state, params, forcing_t):
    """A host callback that, under ``vmap``, replaces each sample's water by the batch mean.

    Source: kit self-test fixture.
    """
    w0 = state.water + forcing_t.rain
    w = jax.pure_callback(
        _batch_mean_host, jax.ShapeDtypeStruct(w0.shape, w0.dtype), w0, vmap_method="expand_dims"
    )
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _scan_upcast(state, params, forcing_t):
    """An inner loop whose carry turns float64 on float32 inputs (under x64): it does not trace.

    Source: kit self-test fixture.
    """
    w0 = state.water + forcing_t.rain

    def body(w, _):
        return w * jnp.ones((), jnp.float64), None

    w, _ = jax.lax.scan(body, w0, None, length=2)
    d = _drain(state, params) * w
    return eqx.tree_at(lambda s: (s.water, s.drained), state, (w - d, d))


def _type_error(state, params, forcing_t):
    """Raises a TypeError that is not a loop-carry type mismatch (a bug, not an upcast).

    Source: kit self-test fixture.
    """
    raise TypeError("toy_type_error: unrelated fault")


# ------------------------------------------------------------------ a water-supply toy with an inout port
class SupplyState(State):
    """A potential-uptake producer: reads ``sw``, ``eop`` of the crop water record, writes ``trwup``."""

    uptake: Array = field(unit="cm d-1", description="potential uptake kept in the module", dims=("n_crop",))
    water: CropWaterIn = port(description="crop water record: sw, eop read, trwup written")


class SupplyParams(Params):
    coefficients: BucketCoefficients = field(description="uptake fraction")


#: mm d-1 -> cm d-1 (EOP is in mm d-1, TRWUP in cm d-1)
MM_TO_CM = 0.1
#: weight of yesterday's trwup in the self-reading fixture
CARRY = 0.5


def _supply(state, params, forcing_t):
    """``trwup = k eop mean(sw)`` (cm d-1).

    Source: kit self-test fixture.
    """
    t = params.coefficients.k * MM_TO_CM * state.water.eop * jnp.mean(state.water.sw)
    return eqx.tree_at(lambda s: (s.uptake, s.water.trwup), state, (t, t))


def _self_read(state, params, forcing_t):
    """As ``_supply`` plus half of the ``trwup`` already in the port: the module keeps its memory in
    a shared port (declared: it reads the whole record), so a replay of recorded records differs.

    Source: kit self-test fixture.
    """
    w = state.water
    t = params.coefficients.k * MM_TO_CM * w.eop * jnp.mean(w.sw) + CARRY * w.trwup
    return eqx.tree_at(lambda s: (s.uptake, s.water.trwup), state, (t, t))


python_if = _proc(_python_if, "python_if")
no_source = _proc(_no_source, "no_source")
#: a faithful DSSAT key (BSD-3, statements allowed) claiming reference_only_conventions, no ref_build
WRONG_PROVENANCE_KEY = "crop/toy_wrong_provenance@dssat-4.8.6.0:faithful"
wrong_provenance = _proc(
    _good_body, "wrong_provenance", key=WRONG_PROVENANCE_KEY, provenance="reference_only_conventions"
)
undeclared_write = _proc(_undeclared_write, "undeclared_write", writes=("water",))
undeclared_read = _proc(_undeclared_read, "undeclared_read", reads=("water",))
writes_in_port = _proc(_writes_in_port, "writes_in_port", writes=("water", "drained", "n_in"))
leak = _proc(_leak, "leak")
impure = _proc(_impure, "impure")
nonfinite = _proc(_nonfinite, "nonfinite")
upcast = _proc(_upcast, "upcast")
nan_grad = _proc(_nan_grad, "nan_grad")
wrong_grad = _proc(_wrong_grad, "wrong_grad")
jump = _proc(_jump, "jump")
cancel = _proc(_cancel, "cancel")
scan_upcast = _proc(_scan_upcast, "scan_upcast")
type_error = _proc(_type_error, "type_error")
batch_mixing = _proc(_batch_mixing, "batch_mixing")
#: a variant of a reference without its faithful sibling in the registry (``process`` itself
#: refuses a variant without deviations, so the fixture lists one)
orphan = _proc(
    _good_body,
    "orphan",
    key="crop/toy_orphan@dssat-4.8.6.0:tuned",
    deviates=(("drainage fraction tuned", "kit fixture", "test_kit_self.py"),),
)
SUPPLY_KEY = "water_supply/toy_supply@none:kit_self"
supply = _proc(
    _supply, "supply", reads=("water.sw", "water.eop"), writes=("uptake", "water.trwup"), key=SUPPLY_KEY
)
self_read = _proc(
    _self_read,
    "self_read",
    reads=("water",),
    writes=("uptake", "water.trwup"),
    key="water_supply/toy_self_read@none:kit_self",
)

__all__ = ["BoundedCoefficients", "BucketCoefficients", "OtherRefCoefficients", "SupplyParams", "SupplyState"]
