"""Gradient helpers shared by every process: guarded arithmetic, table kernels and straight-through rules.

Every helper here computes exactly the reference model's forward value; only the derivative
depends on the **gradient mode** (plan 19 A11; doc 17 section 5):

``"exact"``
    The derivative JAX gives for the forward expression: 0 through a truncation, a rounding
    or a threshold, the taken branch through a selection.
``"ste"`` (the default)
    Straight-through rules: identity through a truncation or rounding (bias bounded by the
    quantum), a ramp of one step of the driving accumulator through a threshold
    (:func:`event_ste`), and the jump of the selected quantity through a selection on such a
    threshold (:func:`select_ste`).
``"implicit"``
    As ``"ste"`` for the helpers of this module (on a daily clock the implicit-function-theorem
    derivative of an event time is the one-step ramp, doc 17 section 5.2); in addition, iterative
    solvers that offer an implicit-function-theorem backward pass use it
    (:func:`solver_adjoint` returns ``"implicit"``).

The mode is resolved **at trace time**, never traced: the forward program is the same in every
mode (tested bit for bit), and a mode is orthogonal to the process variants of the registry.
Resolution order: the ``mode=`` argument of a helper, the innermost :func:`gradient_mode`
context, the environment variable ``AGRI_JAX_GRADIENT_MODE``, then ``"ste"``. Because it is a
trace-time setting, a function jitted under one mode keeps that mode; the runtime's cached batch
runners include the mode in their cache key, and a user-jitted function must be re-jitted (or
the mode passed explicitly) to change it.

``safe_div``, ``trunc_st``, ``curv_lin`` and ``tabex`` moved here from
``agrijax.processes.crop.ceres_maize._util`` (re-exported there unchanged). ``curv_lin`` and
``tabex`` follow DSSAT-CSM ``Utilities/UTILS.for`` (``CURV`` with type ``LIN``, ``TABEX``); the
truncation and rounding rules reproduce Fortran ``INT`` and ``NINT``. DSSAT-CSM is BSD-3 (DSSAT
Foundation, University of Florida, IFDC); see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard

__all__ = [
    "DEFAULT_GRADIENT_MODE",
    "GRADIENT_MODES",
    "GRADIENT_MODE_ENV",
    "GradientMode",
    "current_gradient_mode",
    "curv_lin",
    "event_ste",
    "gradient_mode",
    "resolve_gradient_mode",
    "round_st",
    "safe_div",
    "select_ste",
    "solver_adjoint",
    "tabex",
    "trunc_st",
]

#: floor of the segment widths ``x1 - xb``, ``xm - x2`` (``curv_lin``) and ``arg[j] - arg[j-1]``
#: (``tabex``): keeps a degenerate (zero-width) segment finite
_SEGMENT_WIDTH_MIN: float = numerical_guard(
    "grad.segment_width_min", 1e-6, "floor of a piecewise-linear segment width (curv_lin, tabex)"
)
#: Fortran ``NINT`` rounds a fractional part of magnitude >= 1/2 away from zero
_NINT_HALF: float = 0.5
#: base of the ``10**decimals`` scale of :func:`round_st`
_DECIMAL_BASE: float = 10.0

GradientMode = Literal["exact", "ste", "implicit"]
GRADIENT_MODES: tuple[str, ...] = ("exact", "ste", "implicit")
DEFAULT_GRADIENT_MODE: GradientMode = "ste"
GRADIENT_MODE_ENV = "AGRI_JAX_GRADIENT_MODE"

_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("agrijax_gradient_mode", default=None)

#: smallest accumulator step used by :func:`event_ste` (the ramp width is ``max(rate, _RATE_FLOOR)``)
_RATE_FLOOR: float = numerical_guard(
    "grad.rate_floor", 1e-6, "smallest accumulator step of the event_ste ramp (keeps the ramp finite)"
)


# ------------------------------------------------------------------------------------- the mode
def _check_mode(mode: str) -> GradientMode:
    m = str(mode).strip().lower()
    if m not in GRADIENT_MODES:
        raise ValueError(f"unknown gradient mode {mode!r}; expected one of {GRADIENT_MODES}")
    return cast(GradientMode, m)


def current_gradient_mode() -> GradientMode:
    """The mode in force: innermost :func:`gradient_mode` context, else ``AGRI_JAX_GRADIENT_MODE``,
    else ``"ste"``. Read at trace time."""
    m = _MODE.get()
    if m is not None:
        return _check_mode(m)
    env = os.environ.get(GRADIENT_MODE_ENV, "").strip()
    return _check_mode(env) if env else DEFAULT_GRADIENT_MODE


def resolve_gradient_mode(mode: str | None = None) -> GradientMode:
    """``mode`` if given (validated), else :func:`current_gradient_mode`."""
    return current_gradient_mode() if mode is None else _check_mode(mode)


@contextlib.contextmanager
def gradient_mode(mode: str) -> Iterator[GradientMode]:
    """Context manager setting the gradient mode for everything traced inside it.

    ::

        with gradient_mode("exact"):
            g = jax.grad(loss)(params)     # traced here: exact derivatives
    """
    m = _check_mode(mode)
    token = _MODE.set(m)
    try:
        yield m
    finally:
        _MODE.reset(token)


def solver_adjoint(mode: str | None = None) -> Literal["unrolled", "implicit"]:
    """Backward pass an iterative solver should use under ``mode``: ``"implicit"`` (IFT) in the
    ``"implicit"`` mode, ``"unrolled"`` otherwise. Solvers keep their own explicit override."""
    return "implicit" if resolve_gradient_mode(mode) == "implicit" else "unrolled"


# ------------------------------------------------------------------------------ guarded kernels
def safe_div(num: ArrayLike, den: ArrayLike, fill: ArrayLike = 0.0) -> Array:
    """``num / den`` where ``den != 0``, else ``fill``; both branches finite for any input.

    Source: guard for the divisions DSSAT-CSM protects with ``IF (x .GT. 0.0)``.
    """
    den = jnp.asarray(den)
    ok = den != 0.0
    return jnp.where(ok, jnp.asarray(num) / jnp.where(ok, den, 1.0), fill)


def curv_lin(xb: ArrayLike, x1: ArrayLike, x2: ArrayLike, xm: ArrayLike, x: ArrayLike) -> Array:
    """Trapezoidal response ``CURV('LIN', XB, X1, X2, XM, X)``: 0 below ``xb`` and above ``xm``,
    1 between ``x1`` and ``x2``, linear in between, clipped to ``[0, 1]``.

    Source: DSSAT-CSM Utilities/UTILS.for, FUNCTION CURV (CTYPE 'LIN').
    """
    x = jnp.asarray(x)
    up = (x - xb) / jnp.maximum(jnp.asarray(x1) - xb, _SEGMENT_WIDTH_MIN)
    down = 1.0 - (x - x2) / jnp.maximum(jnp.asarray(xm) - x2, _SEGMENT_WIDTH_MIN)
    out = jnp.where(
        (x > xb) & (x < x1),
        up,
        jnp.where((x >= x1) & (x <= x2), 1.0, jnp.where((x > x2) & (x < xm), down, 0.0)),
    )
    return jnp.clip(out, 0.0, 1.0)


def tabex(val: ArrayLike, arg: ArrayLike, x: ArrayLike) -> Array:
    """Piecewise-linear table lookup with linear extrapolation beyond the ends (``TABEX``).

    ``arg`` and ``val`` are ``[..., K]`` with increasing ``arg``; the segment is the first
    ``J >= 2`` with ``x <= ARG(J)`` (``J = K`` beyond the last point).

    Source: DSSAT-CSM Utilities/UTILS.for, FUNCTION TABEX.
    """
    val = jnp.asarray(val)
    arg = jnp.asarray(arg)
    x = jnp.asarray(x)
    k = arg.shape[-1]
    xe = x[..., None]
    # 0-based upper index of the segment: 1 + count(arg[1:K-1] < x), in 1..K-1
    j = 1 + jnp.sum(xe > arg[..., 1 : k - 1], axis=-1)
    hi = jnp.arange(k) == j[..., None]
    lo = jnp.arange(k) == (j - 1)[..., None]
    a_hi = jnp.sum(jnp.where(hi, arg, 0.0), axis=-1)
    a_lo = jnp.sum(jnp.where(lo, arg, 0.0), axis=-1)
    v_hi = jnp.sum(jnp.where(hi, val, 0.0), axis=-1)
    v_lo = jnp.sum(jnp.where(lo, val, 0.0), axis=-1)
    return (x - a_lo) * (v_hi - v_lo) / jnp.maximum(a_hi - a_lo, _SEGMENT_WIDTH_MIN) + v_lo


# ------------------------------------------------------------------- truncation and rounding
def trunc_st(x: ArrayLike, *, mode: str | None = None) -> Array:
    """``INT``-style truncation toward zero; identity (straight-through) derivative in ``ste`` /
    ``implicit`` mode, 0 in ``exact`` mode.

    The value is exactly ``trunc(x)`` (``x + (trunc(x) - x)`` rounds back to ``trunc(x)``: the
    difference is exact by Sterbenz' lemma for ``|x| >= 1`` and ``-x`` below), so outputs agree
    with DSSAT's ``REAL(INT(x))``; the straight-through derivative is 1 instead of 0 so that
    gradients flow through the 1e-3 quantisation DSSAT applies to TURFAC and RLV. The ``exact``
    mode evaluates the same expression and stops its gradient, so the forward value is
    bit-identical in every mode.

    Source: DSSAT-CSM MZ_GROSUB.for (TURFAC), MZ_ROOTS.for (RLV) ``REAL(INT(x*1000))/1000``.
    """
    x = jnp.asarray(x)
    y = x + lax.stop_gradient(jnp.trunc(x) - x)
    return lax.stop_gradient(y) if resolve_gradient_mode(mode) == "exact" else y


def _nint(x: Array) -> Array:
    """Fortran ``NINT``: round half away from zero, from the exact fractional part."""
    t = jnp.trunc(x)
    frac = x - t  # exact for every finite x
    return t + jnp.where(jnp.abs(frac) >= _NINT_HALF, jnp.sign(x), 0.0)


def round_st(x: ArrayLike, decimals: int = 0, *, mode: str | None = None) -> Array:
    """``REAL(NINT(x * 10**decimals)) / 10**decimals``: Fortran ``NINT`` (half away from zero, not
    ``jnp.round``'s half to even) with an identity derivative in ``ste`` / ``implicit`` mode and
    0 in ``exact`` mode. The forward value is bit-identical in every mode.

    ``decimals`` is static. With ``decimals = 0`` the value is ``NINT(x)``. The final division
    is a true division (see :func:`_divide`), not XLA's multiplication by the reciprocal.

    Source: DSSAT-CSM SPAM/STEMP.for ``ST(L) = NINT(ST(L)*1000.)/1000.``,
    ``TMA(1) = NINT(TMA(1)*10000.)/10000.``.
    """
    if int(decimals) != decimals or decimals < 0:
        raise ValueError(f"decimals must be a non-negative integer, got {decimals!r}")
    x = jnp.asarray(x)
    scale = _DECIMAL_BASE ** int(decimals)
    xs = x * scale if decimals else x
    q = _nint(lax.stop_gradient(xs))
    y = xs + lax.stop_gradient(q - xs)  # value q (exact, as in trunc_st), derivative 1 w.r.t. xs
    y = _divide(y, scale) if decimals else y
    return lax.stop_gradient(y) if resolve_gradient_mode(mode) == "exact" else y


def _divide(y: Array, scale: float) -> Array:
    """``y / scale`` as a true IEEE division, like the Fortran ``NINT(...)/1000.``.

    XLA rewrites a division by a compile-time constant into a multiplication by its rounded
    reciprocal, which differs from the quotient in the last bit for about 1 value in 8 (float64)
    or 2 in 3 (float32). ``y * 0 + scale`` is not constant-folded (IEEE ``0 * inf`` is NaN), so
    the division stays a division; for finite ``y`` its value is exactly ``scale``.
    """
    return y / (y * 0.0 + scale)


# ------------------------------------------------------------------------ thresholds and events
@jax.custom_jvp
def _event_ste(margin: Array, rate: Array) -> Array:
    return (margin >= 0.0).astype(margin.dtype)


def _ramp(margin: Array, rate: Array) -> Array:
    return jnp.clip(margin / jnp.maximum(rate, _RATE_FLOOR), 0.0, 1.0)


@_event_ste.defjvp
def _event_ste_jvp(primals: tuple[Array, Array], tangents: tuple[Array, Array]) -> tuple[Array, Array]:
    margin, rate = primals
    _, t_out = jax.jvp(_ramp, primals, tangents)
    return _event_ste(margin, rate), t_out


def event_ste(margin: ArrayLike, rate: ArrayLike, *, mode: str | None = None) -> Array:
    """Indicator ``margin >= 0`` of a threshold on an accumulator, as a float 0/1.

    ``margin = S_d - P`` is the accumulator past its threshold on day ``d`` and ``rate = r_d`` the
    accumulator's step that day (``S_d = S_{d-1} + r_d``). The value is the hard indicator in
    every mode (bit-identical). In ``ste`` / ``implicit`` mode the derivative is that of the ramp
    ``clip(margin / max(rate, 1e-6), 0, 1)``, the fraction of the crossing day spent past the
    threshold: on the crossing day ``d(ind)/dP = -1/r_d``, the discrete event-time derivative, and
    0 on every other day. In ``exact`` mode the derivative is 0. Every quotient is on a clamped
    denominator, so the derivative is finite for any input.

    Let the switch act through :func:`select_ste` (or an arithmetic blend on the indicator), not
    ``jnp.where`` on a boolean, for the jump to reach the loss.

    Source: doc 17 section 5.2 (straight-through estimator, Bengio, Leonard & Courville 2013).
    """
    margin = jnp.asarray(margin)
    dtype = jnp.result_type(margin, rate, float)
    margin = margin.astype(dtype)
    rate = jnp.asarray(rate, dtype)
    margin, rate = jnp.broadcast_arrays(margin, rate)
    if resolve_gradient_mode(mode) == "exact":
        return lax.stop_gradient((margin >= 0.0).astype(dtype))
    return _event_ste(margin, rate)


@jax.custom_jvp
def _select_ste(ind: Array, a: Array, b: Array) -> Array:
    return jnp.where(ind != 0.0, a, b)


@_select_ste.defjvp
def _select_ste_jvp(
    primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]
) -> tuple[Array, Array]:
    ind, a, b = primals
    d_ind, d_a, d_b = tangents
    # blend ind*a + (1 - ind)*b: the tangent of the selected branch plus the jump times d(ind)
    t_out = ind * d_a + (1.0 - ind) * d_b + (a - b) * d_ind
    return _select_ste(ind, a, b), t_out


def select_ste(ind: ArrayLike, a: ArrayLike, b: ArrayLike, *, mode: str | None = None) -> Array:
    """``a`` where the 0/1 indicator ``ind`` is set, else ``b``.

    The value is ``jnp.where(ind != 0, a, b)`` in every mode (bit-identical; the arithmetic blend
    would turn ``-0.0`` into ``+0.0`` and an infinite unselected branch into NaN). In ``ste`` /
    ``implicit`` mode the derivative is that of the blend ``ind * a + (1 - ind) * b``, so the
    surrogate derivative of an :func:`event_ste` indicator carries the jump ``a - b``; in
    ``exact`` mode it is ``jnp.where``'s (the taken branch only). ``a`` and ``b`` must be finite
    (rule: both ``jnp.where`` branches finite).

    Source: doc 17 section 5.2 (the switch acts through the indicator arithmetically).
    """
    ind = jnp.asarray(ind)
    dtype = jnp.result_type(ind, a, b, float)
    ind, a_, b_ = jnp.broadcast_arrays(ind.astype(dtype), jnp.asarray(a, dtype), jnp.asarray(b, dtype))
    if resolve_gradient_mode(mode) == "exact":
        return jnp.where(ind != 0.0, a_, b_)
    return _select_ste(ind, a_, b_)
