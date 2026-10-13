"""Gradient helpers shared by every process: guarded arithmetic, table kernels and straight-through rules.

Every helper here computes exactly the reference model's forward value; only the derivative
depends on the **gradient mode**:

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
    derivative of an event time is the one-step ramp); in addition, iterative
    solvers that offer an implicit-function-theorem backward pass use it
    (:func:`solver_adjoint` returns ``"implicit"``).

**The straight-through derivative, defined.** For a model whose only non-smooth steps are the
quantisers of this module (:func:`trunc_st`, :func:`round_st`, :func:`real4_store`), the ``ste``
derivative is the derivative of the model with the quantisation removed, taken along the quantised
trajectory: every quantiser contributes the identity (``trunc_st`` / ``round_st``; ``real4_store``
contributes the identity with the tangent rounded to binary32, in the ``exact`` mode too: unlike
``trunc_st`` / ``round_st``, whose exact-mode derivative is 0, its exact derivative is that of the
cast). :func:`unrounded` (a trace-time switch, orthogonal to the gradient mode; :func:`bind_unrounded`
fixes it on a function) builds that model itself: every quantiser returns its argument unchanged
(value and derivative); :mod:`agrijax.calib.trust` reports it as a diagnostic. It is the one setting
here that changes forward values; it is never on unless asked for. Like the mode it is read at trace
time: a function already jitted keeps its earlier setting (bind it with :func:`bind_unrounded`). The
events (:func:`event_ste`, :func:`select_ste`) are thresholds, not quantisers: :func:`unrounded`
leaves them as they are.

The mode is resolved **at trace time**, never traced: the forward program is the same in every
mode (tested bit for bit), and a mode is orthogonal to the process variants of the registry.
Resolution order: the ``mode=`` argument of a helper, the innermost :func:`gradient_mode`
context, the environment variable ``AGRI_JAX_GRADIENT_MODE``, then ``"ste"``. Because it is a
trace-time setting, a function jitted under one mode keeps that mode: ``jax.jit`` caches the
traced program by the function and its argument shapes, not by the context, so a jitted function
first traced in ``ste`` still differentiates in ``ste`` when it is called later inside
``gradient_mode("exact")`` (also when it is called from an outer function that is traced anew).
Two ways keep a jit and its mode together: the runtime's cached batch runners include the mode in
their cache key, and :func:`bind_gradient_mode` fixes the mode of a function when it is built
(the function enters its own :func:`gradient_mode` context whenever it is traced, so every jit
of it, cached or not, has that mode). Build one bound function, and one jit, per mode.

``safe_div``, ``trunc_st``, ``curv_lin`` and ``tabex`` moved here from
``agrijax.processes.crop.ceres_maize._util`` (re-exported there unchanged). ``curv_lin`` and
``tabex`` follow DSSAT-CSM ``Utilities/UTILS.for`` (``CURV`` with type ``LIN``, ``TABEX``); the
truncation and rounding rules reproduce Fortran ``INT`` and ``NINT``. DSSAT-CSM is BSD-3 (DSSAT
Foundation, University of Florida, IFDC); see ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
from collections.abc import Callable, Iterator
from typing import Any, Generic, Literal, ParamSpec, TypeVar, cast

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
    "ModeBound",
    "ModeMismatchError",
    "Unrounded",
    "bind_gradient_mode",
    "bind_unrounded",
    "coef_div",
    "current_gradient_mode",
    "curv_lin",
    "event_ste",
    "gradient_mode",
    "one_sided_tangent",
    "real4_store",
    "resolve_gradient_mode",
    "round_st",
    "safe_div",
    "select_ste",
    "solver_adjoint",
    "tabex",
    "trunc_st",
    "unrounded",
    "unrounded_active",
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
#: exponent and explicit mantissa bits of a Fortran ``REAL`` (``REAL*4``, IEEE 754 binary32; the
#: DSSAT-CSM and RZWQM2 builds use gfortran / ifort defaults), used by :func:`real4_store`
_REAL4_EXPONENT_BITS: int = 8
_REAL4_MANTISSA_BITS: int = 23

GradientMode = Literal["exact", "ste", "implicit"]
GRADIENT_MODES: tuple[str, ...] = ("exact", "ste", "implicit")
DEFAULT_GRADIENT_MODE: GradientMode = "ste"
GRADIENT_MODE_ENV = "AGRI_JAX_GRADIENT_MODE"

_MODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("agrijax_gradient_mode", default=None)
_UNROUNDED: contextvars.ContextVar[bool] = contextvars.ContextVar("agrijax_unrounded", default=False)

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


_P = ParamSpec("_P")
_R = TypeVar("_R")


class ModeMismatchError(ValueError):
    """A mode-bound function was called inside a :func:`gradient_mode` context of another mode."""


class ModeBound(Generic[_P, _R]):
    """``fn`` with its gradient mode fixed: every call (and so every trace, by ``jax.jit``,
    ``jax.grad``, ``jax.vmap`` or an enclosing trace) runs inside ``gradient_mode(self.mode)``.
    Called inside an explicit :func:`gradient_mode` context of a **different** mode it raises
    :class:`ModeMismatchError` instead of silently running its own mode (the caller meant the
    other mode: build a function bound to it). The environment variable does not count as an
    explicit context. Built by :func:`bind_gradient_mode`.

    When the check runs: whenever this Python ``__call__`` runs, i.e. at every trace and at every
    call of :meth:`jit`'s result (the check is Python before the compiled program). Once compiled,
    a bare ``jax.jit(bound)`` or an outer ``jax.jit`` around ``bound.jit()`` skips the check on a
    cache hit (no Python runs); the program it runs is still the one traced in the bound mode, never
    another mode's, so the check only ever turns a silent mode mix-up into an error.

    The check is on the call, not only on derivatives: a forward-only call inside a context of
    another mode raises too, by design (the forward is the same program in every mode, but the
    caller asked for the other mode). To evaluate the forward under such a context, call it
    outside the ``with gradient_mode(...)`` block, or use the simulator bound to that mode."""

    def __init__(self, fn: Callable[_P, _R], mode: GradientMode) -> None:
        # name and docstring of fn, but not its __dict__: a jax.jit of a ModeBound carries the
        # inner ModeBound's attributes (fn, mode), which must not overwrite this object's own
        functools.update_wrapper(self, fn, updated=())
        self.fn = fn
        self.mode: GradientMode = mode

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        outer = _MODE.get()
        if outer is not None and _check_mode(outer) != self.mode:
            raise ModeMismatchError(
                f"{self!r} is bound to gradient mode {self.mode!r} but was called inside "
                f"gradient_mode({outer!r}); build one function per mode (bind_gradient_mode(fn, {outer!r}))"
            )
        with gradient_mode(self.mode):
            return self.fn(*args, **kwargs)

    def jit(self, **jit_kwargs: Any) -> ModeBound[_P, _R]:
        """``jax.jit`` of this function, still bound: the mode check runs on every call (in
        Python, before the compiled program), and the program is traced in :attr:`mode`."""
        return ModeBound(cast(Callable[_P, _R], jax.jit(self, **jit_kwargs)), self.mode)

    def __repr__(self) -> str:
        return f"ModeBound({getattr(self.fn, '__name__', self.fn)!r}, mode={self.mode!r})"


@contextlib.contextmanager
def unrounded() -> Iterator[None]:
    """Trace the quantisers of this module as the identity (module docstring: the model whose exact
    derivative the ``ste`` derivative is). Read at trace time, like the gradient mode, so it applies
    to what is traced inside the block; a function already jitted keeps its earlier setting (``jax.jit``
    caches the traced program by the function and its argument shapes, not by this switch). Build the
    unrounded function once with :func:`bind_unrounded`, which enters the switch whenever it is
    traced::

        f_unrounded = jax.jit(bind_unrounded(f))
        y = f_unrounded(theta)      # no truncation, rounding or REAL*4 store, whatever f was before
    """
    token = _UNROUNDED.set(True)
    try:
        yield
    finally:
        _UNROUNDED.reset(token)


def unrounded_active() -> bool:
    """Whether :func:`unrounded` is in force (at trace time)."""
    return _UNROUNDED.get()


class Unrounded(Generic[_P, _R]):
    """``fn`` traced with the quantisers removed (:func:`unrounded`) whenever it is called (so at every
    trace by ``jax.jit``, ``jax.jvp`` or an enclosing trace). Built by :func:`bind_unrounded`; combine
    with a gradient mode as ``bind_gradient_mode(bind_unrounded(fn), mode)``."""

    def __init__(self, fn: Callable[_P, _R]) -> None:
        functools.update_wrapper(self, fn, updated=())
        self.fn = fn

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with unrounded():
            return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Unrounded({getattr(self.fn, '__name__', self.fn)!r})"


def bind_unrounded(fn: Callable[_P, _R]) -> Unrounded[_P, _R]:
    """``fn`` with the quantisers removed at every trace (:class:`Unrounded`)."""
    return Unrounded(fn)


def bind_gradient_mode(fn: Callable[_P, _R], mode: str | None = None) -> ModeBound[_P, _R]:
    """``fn`` bound to ``mode`` (``None``: the mode in force **now**, when it is bound).

    The mode of a bound function travels with it, so ``jax.jit(bind_gradient_mode(f, "exact"))``
    is an exact-mode program however often it is called and under whatever outer context; a
    ``jax.jit(f)`` of the unbound ``f`` would keep the mode of its first trace instead (see the
    module docstring). A bound function called inside an explicit context of another mode raises
    :class:`ModeMismatchError`."""
    return ModeBound(fn, resolve_gradient_mode(mode))


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


def coef_div(num: ArrayLike, den: Any) -> Array | Any:
    """``num / den`` for a divisor ``den`` that is a **model coefficient** (or a calibratable
    parameter).

    A Python-number divisor (a coefficient at its default, not calibrated) is divided directly,
    so the traced program is exactly the one before the coefficient was named, and a zero divisor
    is a declaration error raised at trace time. An array divisor (the coefficient is a pytree
    leaf being calibrated or differentiated) is guarded **before** the division:
    ``num / where(den != 0, den, 1)``, so value and derivative are finite everywhere (the quotient
    by 1, and a zero derivative with respect to ``den``, where ``den == 0``). Nothing is placed
    between the quotient and its consumer: a select on the quotient (as in :func:`safe_div`)
    changed the compiled program of ``lfwt - slan / c`` and moved the 58-treatment CERES-Maize
    comparison report against DSSAT-CSM in its last bit; guarding the divisor alone keeps the
    forward bit-identical to the unprotected ``num / den`` wherever ``den != 0`` (checked on that
    report and the CERES runs).
    Derivatives are not always bit-identical: the guard changes the compiled backward program, and
    on the DSSAT problems the cultivar Jacobian of final GWAD / CWAD moves by up to 4 ulp
    (9.1e-13 absolute, through ``grain_number``'s ``PSKER`` and ``GPP`` divisors).
    The valid domain (:class:`agrijax.calib.ParamSpec`) keeps a calibrated divisor away from 0 in
    the first place.
    """
    if isinstance(den, (int, float)) and not isinstance(den, bool):
        if den == 0:
            raise ValueError("coef_div: a coefficient divisor declared as 0")
        return cast(Any, num) / den
    den_a = jnp.asarray(den)
    return cast(Any, num) / jnp.where(den_a != 0.0, den_a, 1.0)


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
    bit-identical in every mode. Under :func:`unrounded` it returns ``x``.

    Source: DSSAT-CSM MZ_GROSUB.for (TURFAC), MZ_ROOTS.for (RLV) ``REAL(INT(x*1000))/1000``.
    """
    x = jnp.asarray(x)
    if _UNROUNDED.get():
        return x
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
    0 in ``exact`` mode. The forward value is bit-identical in every mode. Under :func:`unrounded` it
    returns ``x``.

    ``decimals`` is static. With ``decimals = 0`` the value is ``NINT(x)``. The final division
    is a true division (see :func:`_divide`), not XLA's multiplication by the reciprocal.

    Source: DSSAT-CSM SPAM/STEMP.for ``ST(L) = NINT(ST(L)*1000.)/1000.``,
    ``TMA(1) = NINT(TMA(1)*10000.)/10000.``.
    """
    if int(decimals) != decimals or decimals < 0:
        raise ValueError(f"decimals must be a non-negative integer, got {decimals!r}")
    x = jnp.asarray(x)
    if _UNROUNDED.get():
        return x
    scale = _DECIMAL_BASE ** int(decimals)
    xs = x * scale if decimals else x
    q = _nint(lax.stop_gradient(xs))
    y = xs + lax.stop_gradient(q - xs)  # value q (exact, as in trunc_st), derivative 1 w.r.t. xs
    y = _divide(y, scale) if decimals else y
    return lax.stop_gradient(y) if resolve_gradient_mode(mode) == "exact" else y


def real4_store(x: ArrayLike) -> Array:
    """``x`` stored in a Fortran ``REAL*4`` and read back: the nearest binary32 value (round to
    nearest even), kept in the dtype of ``x``.

    This is ``lax.reduce_precision`` to 8 exponent and 23 mantissa bits, not the convert pair
    ``x.astype(float32).astype(x.dtype)``: with XLA's default ``--xla_allow_excess_precision=true``
    the GPU pipeline removes a float64 -> float32 -> float64 convert pair as a no-op, so the store
    silently vanished on GPU (measured on an H100, 2026-09-28: every value kept its float64 bits,
    and the REAL*4 equality tests ``SW .LE. LL`` of the DSSAT day failed). ``reduce-precision`` is
    kept by every backend. For normal binary32 values (``|x| >= 2**-126``) the result equals the
    convert pair bit for bit; a value in the binary32 subnormal range becomes a signed zero
    instead of a subnormal (XLA flushes it), overflow gives infinity as the cast does. On a
    float32 input it is the identity.

    Derivative: that of the convert pair, kept on purpose. ``reduce_precision`` is linear, so its
    JVP applies the same rounding to the tangent and its transpose to the cotangent: the
    derivative is the identity with the tangent / cotangent rounded to binary32 (relative change
    at most 2**-24; a cotangent below the binary32 normal range becomes 0). An exact identity
    derivative would be ``x + lax.stop_gradient(real4_store(x) - x)`` (straight-through, as in
    :func:`round_st`); not used, so that the gradients stay those of the convert pair. Under
    :func:`unrounded` it returns ``x``.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilWater/WATBAL.for:503-505 (``SW(L) = ANINT(SW(L)*1.E6)/1.E6``
    into ``REAL SW(NL)``), ModuleDefs.for (``REAL`` soil variables), BSD-3.
    """
    if _UNROUNDED.get():
        return jnp.asarray(x)
    return lax.reduce_precision(
        jnp.asarray(x), exponent_bits=_REAL4_EXPONENT_BITS, mantissa_bits=_REAL4_MANTISSA_BITS
    )


def _divide(y: Array, scale: float) -> Array:
    """``y / scale`` as a true IEEE division, like the Fortran ``NINT(...)/1000.``.

    XLA rewrites a division by a compile-time constant into a multiplication by its rounded
    reciprocal, which differs from the quotient in the last bit for about 1 value in 8 (float64)
    or 2 in 3 (float32). ``y * 0 + scale`` is not constant-folded (IEEE ``0 * inf`` is NaN), so
    the division stays a division; for finite ``y`` its value is exactly ``scale``.

    Limitation (GPU, float32): XLA's GPU float32 division is not correctly rounded (it lowers to
    an approximate reciprocal-based division), so in float32 on GPU the quotient, and with it
    :func:`round_st` with ``decimals > 0``, can differ from IEEE by 1 ulp (measured on an H100
    MIG slice, 2026-09-28: 31,905 of 1e6 quotients, ``round_st(x, 6)`` in 24,831 of 1e6; the
    free-run DSSAT day in float32 over its 65 validation runs, GPU against CPU: 39 of 65 runs differ
    by more than 1e-9, at most 2.2e-5 in yield, growth stages identical). Divisions on CPU are
    correctly rounded; in float64 the GPU run of the same validation equals CPU to 1.4e-15.
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


def one_sided_tangent(x: ArrayLike, slope: ArrayLike, where: ArrayLike) -> Array:
    """A term of value 0 whose derivative is ``slope * dx`` where ``where`` holds (0 elsewhere): added
    to a quantity at a **domain boundary** of ``x`` (an input that cannot go below it, rain at 0 mm),
    it turns the derivative of the branch the forward expression selects *at* the boundary into the
    one-sided derivative of the branch that holds just inside the domain. Forward values are unchanged
    (the term is ``slope * (x - x)`` with the second ``x`` constant: +-0.0), in every gradient mode: it
    is the derivative of the function on its domain, not a convention, so it is not a quantiser call
    site (:class:`agrijax.core.process.GradientConvention`)."""
    x = jnp.asarray(x)
    return jnp.where(jnp.asarray(where), jnp.asarray(slope), 0.0) * (x - lax.stop_gradient(x))


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

    Source: straight-through estimator (Bengio, Leonard & Courville 2013), surrogate scaled to one
    step of the driving accumulator.
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

    Source: straight-through estimator (Bengio, Leonard & Courville 2013); the switch acts through
    the indicator arithmetically.
    """
    ind = jnp.asarray(ind)
    dtype = jnp.result_type(ind, a, b, float)
    ind, a_, b_ = jnp.broadcast_arrays(ind.astype(dtype), jnp.asarray(a, dtype), jnp.asarray(b, dtype))
    if resolve_gradient_mode(mode) == "exact":
        return jnp.where(ind != 0.0, a_, b_)
    return _select_ste(ind, a_, b_)
