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

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.process import process

from ._toy import META, BucketCoefficients

_RNG = np.random.default_rng(0)
#: the drainage fraction at which the ``jump`` fixture's step sits (just above the default 0.1)
K_JUMP = 0.1 * (1.0 + 1e-11)


class OtherRefCoefficients(Coefficients):
    """A coefficient citing a reference the key does not follow, without a note."""

    k: float = coef(0.1, "d-1", "drainage fraction", Provenance("fao56-1998", paper="FAO-56"))


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

__all__ = ["BucketCoefficients", "OtherRefCoefficients"]
