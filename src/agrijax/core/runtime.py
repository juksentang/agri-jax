"""Runtime: ``run`` (scan over days), ``run_batch`` (vmap + jit), ``run_batch_chunked`` (bounded memory).

Process authors never see these; users call them with a :class:`~agrijax.core.model.Model`,
a ``Params`` pytree (optionally with a leading batch axis), a ``Forcing`` pytree with the
time axis first, and an initial state.

Gradients: reverse mode through a 3,300-day scan needs the per-day residuals in memory.
``checkpoint=True`` wraps the day step in :func:`jax.checkpoint` so that the forward
values are recomputed during the backward pass; this is what lets 10^5 samples fit on
one GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from agrijax.core.model import Model
from agrijax.core.process import check_enabled

__all__ = ["run", "run_and_grad", "run_batch", "run_batch_chunked", "run_sites"]


def _day_step(model: Model, checkpoint: bool) -> Callable[..., Any]:
    step = model.compile()
    if checkpoint:
        step = jax.checkpoint(step)  # type: ignore[assignment]
    return step


def run(
    model: Model,
    params: Any,
    forcing: Any,
    state0: Any,
    *,
    checkpoint: bool = False,
    return_final: bool = False,
) -> Any:
    """Simulate every day of ``forcing`` for one parameter set.

    Returns the per-day outputs (each leaf ``[T, ...]``); with ``return_final=True``
    returns ``(final_state, outputs)``. Not jitted itself: wrap in ``jax.jit`` or use
    :func:`run_batch`.
    """
    step = _day_step(model, checkpoint)

    def body(state: Any, forcing_t: Any) -> tuple[Any, Any]:
        return step(state, params, forcing_t)

    final_state, outs = lax.scan(body, state0, forcing)
    if return_final:
        return final_state, outs
    return outs


def run_batch(
    model: Model,
    params: Any,
    forcing: Any,
    state0: Any,
    *,
    checkpoint: bool = False,
    in_axes: tuple[Any, Any, Any] = (0, None, None),
    jit: bool = True,
) -> Any:
    """``vmap`` of :func:`run` over a leading batch axis, jitted.

    ``in_axes`` refers to ``(params, forcing, state0)``; the default batches parameters
    only. Every leaf of a batched argument must carry the batch axis.
    """

    return _batch_fn(model, checkpoint, in_axes, jit)(params, forcing, state0)


#: attribute of a :class:`Model` holding its jitted batch runners, so repeated ``run_batch`` calls
#: with the same shapes reuse one compilation (a fresh ``jax.jit`` per call recompiles every time).
#: Stored on the model (not in a global table) so the runners die with it.
_RUNNERS_ATTR = "_agrijax_batch_runners"


def _batch_fn(model: Model, checkpoint: bool, in_axes: Any, jit: bool) -> Callable[..., Any]:
    def single(p: Any, f: Any, s: Any) -> Any:
        return run(model, p, f, s, checkpoint=checkpoint)

    def build() -> Callable[..., Any]:
        fn = jax.vmap(single, in_axes=in_axes)
        return jax.jit(fn) if jit else fn

    if not jit:
        return build()
    # the processes and outputs are part of the key in case the model's attributes are reassigned;
    # the write check is decided at trace time, so its switch is part of the key too
    key = (tuple(id(p) for p in model.processes), id(model._outputs), checkpoint, in_axes, check_enabled())
    try:
        hash(key)
    except TypeError:  # an unhashable in_axes pytree: no caching
        return build()
    per_model: dict[Hashable, Callable[..., Any]] = model.__dict__.setdefault(_RUNNERS_ATTR, {})
    fn = per_model.get(key)
    if fn is None:
        fn = per_model[key] = build()
    return fn


def run_sites(
    model: Model,
    params: Any,
    forcing: Any,
    state0: Any,
    *,
    checkpoint: bool = False,
) -> Any:
    """Batch over sites: params, forcing and state0 all carry the batch axis (same static shapes)."""
    return run_batch(model, params, forcing, state0, checkpoint=checkpoint, in_axes=(0, 0, 0))


def run_batch_chunked(
    model: Model,
    params: Any,
    forcing: Any,
    state0: Any,
    *,
    chunk: int = 8192,
    checkpoint: bool = False,
) -> Any:
    """Like :func:`run_batch` but processes the parameter batch ``chunk`` samples at a time.

    Implemented with ``lax.map(..., batch_size=chunk)`` inside one ``jit``: a sequential
    loop over chunks, each chunk vmapped, so peak memory is that of ``chunk`` samples.
    The last partial chunk is handled by ``lax.map``. This is what lets an 8 GB GPU run
    10^5 samples.
    """
    if chunk <= 0:
        raise ValueError("chunk must be positive")

    def single(p: Any) -> Any:
        return run(model, p, forcing, state0, checkpoint=checkpoint)

    n = jax.tree_util.tree_leaves(params)[0].shape[0]
    batch_size = min(int(chunk), int(n))

    @jax.jit
    def mapped(p: Any) -> Any:
        return lax.map(single, p, batch_size=batch_size)

    return mapped(params)


def run_and_grad(
    model: Model,
    loss: Callable[[Any], Any],
    params: Any,
    forcing: Any,
    state0: Any,
    *,
    checkpoint: bool = True,
) -> tuple[Any, Any]:
    """``(loss_value, d loss / d params)``, with ``loss(outputs) -> scalar`` applied to :func:`run`'s outputs.

    Reverse mode with per-day checkpointing by default.
    """

    def objective(p: Any) -> Any:
        return loss(run(model, p, forcing, state0, checkpoint=checkpoint))

    value, grad = jax.value_and_grad(objective)(params)
    return value, grad


def stack_days(*days: Any) -> Any:
    """Utility: stack per-day forcing pytrees into one ``[T, ...]`` forcing."""
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *days)
