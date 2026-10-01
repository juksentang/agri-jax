"""Precision helper of the PET kernels: floating inputs keep their dtype."""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

__all__ = ["as_float"]


def as_float(x: ArrayLike) -> Array:
    """``x`` as a floating array that keeps its precision.

    A float32 input stays float32 with x64 enabled (``jnp.asarray(x, dtype=float)``, used before,
    upcast it to float64, so float32 inputs gave float64 fluxes); an integer array or a Python
    number becomes the default float (float64 under x64), as before.
    """
    a = jnp.asarray(x)
    return a if jnp.issubdtype(a.dtype, jnp.floating) else a.astype(jnp.result_type(float))
