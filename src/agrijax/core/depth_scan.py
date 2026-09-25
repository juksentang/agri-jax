"""``depth_scan``: the one sanctioned recurrence along the depth axis inside a process kernel.

The three process rules forbid loops over layers, days and samples: the runtime owns ``scan``
and ``vmap`` over days and samples, and a layer loop is normally a vectorisable expression
(prefix sums, running minima, gathers). A few reference algorithms are true recurrences along
depth, where the value at depth ``k`` depends on a quantity accumulated above it that no
associative operator carries:

* the Green-Ampt wetting front of RZWQM2 ``INFIL``: the rain intensity at slice ``k`` is the
  breakpoint intensity at the cumulative rain consumed by the slices above it;
* DSSAT nitrogen carry-over down the profile (``NNOM``, ``PMINERAL``), later.

Those go through :func:`depth_scan`, a static-length :func:`jax.lax.scan` over the leading axis
of its inputs (the layers or slices of a static grid), with an optional mask for padded or
skipped entries. Keeping the recurrence in one primitive makes it reviewable, testable and
measurable, and it lets the lint (rule AJ006, :mod:`agrijax.core.lint`) tell a sanctioned
recurrence from a Python ``for`` loop unrolled over a shape-derived range.

Contract
--------
* ``step(carry, x) -> (carry, y)`` is pure; ``x`` is one depth entry of ``xs``.
* The length is the leading dimension of every leaf of ``xs`` and is static (it is part of the
  compiled program). A recurrence whose number of steps depends on the state and has no static
  bound is not allowed.
* ``mask[k] = False`` skips entry ``k``: the carry passes through unchanged and ``y`` is zero.
  The step is still evaluated there (``jnp.where`` selects afterwards), so it must stay finite on
  masked inputs, as every ``jnp.where`` branch must.

Source: user decision Q6 (private design note 20); Blelloch (1990) *Prefix sums and their
applications* for the associative cases that do not need this primitive.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = ["depth_scan"]

C = TypeVar("C")
Y = TypeVar("Y")


def _depth_length(xs: Any, length: int | None) -> int:
    leaves = jax.tree_util.tree_leaves(xs)
    sizes = {int(np.shape(x)[0]) for x in leaves if np.ndim(x) >= 1}
    if any(np.ndim(x) == 0 for x in leaves):
        raise ValueError("depth_scan: every leaf of xs needs a leading depth axis")
    if length is not None:
        sizes.add(int(length))
    if len(sizes) != 1:
        raise ValueError(f"depth_scan: inconsistent depth lengths {sorted(sizes)} (xs leaves / length)")
    (n,) = sizes
    return n


def depth_scan(
    step: Callable[[C, Any], tuple[C, Y]],
    carry: C,
    xs: Any,
    *,
    mask: Any = None,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
) -> tuple[C, Y]:
    """Scan ``step`` down (or, with ``reverse=True``, up) the leading depth axis of ``xs``.

    Returns ``(final_carry, ys)`` with ``ys`` stacked along the depth axis, like
    :func:`jax.lax.scan`. ``mask`` (bool, shape ``[n]``) skips entries: the carry is kept and the
    output is zero where it is ``False``. ``length`` is only needed when ``xs`` has no leaves.

    Source: user decision Q6 (private design note 20); :func:`jax.lax.scan`.
    """
    n = _depth_length(xs, length)
    body: Callable[[Any, Any], tuple[Any, Any]]
    if mask is None:
        body = step
        scan_xs = xs
    else:
        m = jnp.asarray(mask, dtype=bool)
        if m.shape != (n,):
            raise ValueError(f"depth_scan: mask shape {m.shape} != ({n},)")

        def masked(c: Any, xm: tuple[Any, Any]) -> tuple[Any, Any]:
            x, keep = xm
            c_new, y = step(c, x)
            c_out = jax.tree_util.tree_map(lambda a, b: jnp.where(keep, a, b), c_new, c)
            y_out = jax.tree_util.tree_map(lambda a: jnp.where(keep, a, jnp.zeros_like(a)), y)
            return c_out, y_out

        body = masked
        scan_xs = (xs, m)
    with jax.named_scope("depth_scan"):
        return lax.scan(body, carry, scan_xs, length=n, reverse=reverse, unroll=unroll)
