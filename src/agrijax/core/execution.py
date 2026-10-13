"""Execution settings: how a program is laid out for a backend, never what it computes.

An execution setting changes the compiled program's structure (loop unrolling, and later other
backend-specific choices) and nothing in the model: the process equations, the coefficients and
the order of the arithmetic inside one step are the same under every setting. Process authors do
not see these settings; the sanctioned primitives read them (``depth_scan`` reads
``depth_unroll``) and the runtime decides the target backend.

``depth_unroll``
    Whether :func:`agrijax.core.depth_scan.depth_scan` unrolls its static-length layer recurrence
    (``lax.scan(..., unroll=True)``, one straight-line block per layer) or keeps it a loop
    (``unroll=1``, an XLA while loop). On a GPU a layer loop is a while loop of a few small kernels
    per layer, launched once per layer and per day. The DSSAT tipping bucket's nested layer loops
    made the free-run day launch-bound on an H100 (measured: ~489 kernels per simulated day, the
    device idle 34-45 %); unrolling them cut this to ~63 kernels per day and made the steady call
    about 3x faster (f64, 1e4 and 1e5 seasons). The layer recurrences give the same bits either
    way; on a GPU the code XLA fuses around them can round differently (measured <= 2e-16 relative
    in yield, the bucket's outputs and all crop stages bit for bit; on a CPU everything is bit for
    bit). The cost is compile time: the unrolled programs compile about 3.5x longer on the GPU
    (~+45 s for the six DSSAT group programs on a whole H100, so a one-off run at 1e5 seasons on a
    cold cache is slower unrolled; break-even at about 50 repeated calls; with a warm persistent
    compilation cache the extra cost is ~+2.5 s). For one-off GPU runs, opt out with
    ``AGRI_JAX_DEPTH_UNROLL=0`` or ``execution(depth_unroll=False)``. On a CPU the loop is cheap
    and the unrolled program is larger: 20-35 % slower and twice the compile time. Hence the
    per-backend defaults :data:`DEPTH_UNROLL_DEFAULTS`: unrolled on
    GPUs, a loop everywhere else (the behaviour before this setting existed). A ``depth_scan`` call
    that passes its own ``unroll=`` keeps it.

Resolution, at trace time (like the gradient mode, :mod:`agrijax.core.grad`): the innermost
:func:`execution` context that sets the field, else the environment variable
``AGRI_JAX_DEPTH_UNROLL`` (``1``/``on``/``true``, ``0``/``off``/``false``, ``auto`` or empty = the
backend default), else ``DEPTH_UNROLL_DEFAULTS[platform]`` for the **target platform**. The target
platform is the innermost :func:`execution` context's ``platform`` (the runtime's entry points
:func:`~agrijax.core.runtime.run`, ``run_batch``, ``run_sites``, ``run_batch_chunked`` and
``run_and_grad`` set it from the device of their committed input arrays), else the platform of
``jax.default_device`` if one is set, else ``jax.default_backend()``.

Only **committed** arrays (placed with ``jax.device_put(x, device)`` or produced on a device)
decide the platform of the runtime's inputs: JAX runs uncommitted inputs on the default device,
so for them the target platform is the default one (:func:`platform_of`).

Pitfall: a setting is read at trace time, and JAX reuses a trace of any function whose identity is
stable across traces: a ``jax.jit`` of the same function object, a ``jax.checkpoint``-wrapped
function, a ``lax.scan`` body or any other inner function defined once and reused. Such a
function keeps the setting of its first trace, whatever context it is called in later. The
runtime's cached batch runners include the resolved :class:`Execution` in their cache key and
build fresh closures per key. Code that changes a setting itself should build fresh closures (a
new model step, a new ``jax.jit``) inside ``execution(...)``, and code that jits a model by hand
for a device other than the default one should trace it inside ``execution(platform=...)``.

Source: measurements of the DSSAT-CSM v4.8.6.0 free-run day on an NVIDIA H100 and CPU nodes of an
HPC cluster (JAX 0.10.2), reproducible with ``scripts/bench/d4_jax_scaling.py`` and
``scripts/bench/d4_2_gpu_diag.py``.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
from jax.core import Tracer

__all__ = [
    "DEPTH_UNROLL_DEFAULTS",
    "DEPTH_UNROLL_ENV",
    "Execution",
    "current_execution",
    "execution",
    "on_platform_of",
    "platform_of",
    "resolve_depth_unroll",
    "target_platform",
]

#: environment variable overriding the per-backend default of ``depth_unroll``
DEPTH_UNROLL_ENV = "AGRI_JAX_DEPTH_UNROLL"

#: ``depth_unroll`` per target platform (normalised by :func:`_normalise`: ``cuda`` and ``rocm``
#: are ``gpu``); a platform not listed keeps the loop (measurements in the module docstring).
DEPTH_UNROLL_DEFAULTS: Mapping[str, bool] = MappingProxyType({"cpu": False, "gpu": True})

_TRUE = frozenset({"1", "on", "true", "yes"})
_FALSE = frozenset({"0", "off", "false", "no"})
_AUTO = frozenset({"", "auto"})
_GPU_PLATFORMS = frozenset({"gpu", "cuda", "rocm"})

#: the fields set by the enclosing :func:`execution` contexts (innermost wins field by field)
_CTX: contextvars.ContextVar[Mapping[str, Any]] = contextvars.ContextVar(
    "agrijax_execution", default=MappingProxyType({})
)


@dataclass(frozen=True)
class Execution:
    """The resolved execution settings of one trace (hashable: part of the runtime's cache keys)."""

    #: target platform, normalised (``cpu``, ``gpu``, ``tpu``, ...)
    platform: str
    #: :func:`~agrijax.core.depth_scan.depth_scan` unrolls its layer recurrence
    depth_unroll: bool


def _normalise(platform: str) -> str:
    p = str(platform).strip().lower()
    return "gpu" if p in _GPU_PLATFORMS else p


def _parse_bool(value: str, what: str) -> bool | None:
    v = value.strip().lower()
    if v in _AUTO:
        return None
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{what}={value!r}: expected one of 1/on/true, 0/off/false, auto")


def target_platform() -> str:
    """The platform the program being traced is meant for (module docstring, resolution)."""
    p = _CTX.get().get("platform")
    if p is not None:
        return _normalise(p)
    dev = jax.config.jax_default_device
    if dev is not None:
        return _normalise(dev if isinstance(dev, str) else dev.platform)
    return _normalise(jax.default_backend())


def resolve_depth_unroll(platform: str | None = None) -> bool:
    """``depth_unroll`` in force: the :func:`execution` context, else ``AGRI_JAX_DEPTH_UNROLL``,
    else the default of ``platform`` (``None``: :func:`target_platform`)."""
    v = _CTX.get().get("depth_unroll")
    if v is not None:
        return bool(v)
    env = _parse_bool(os.environ.get(DEPTH_UNROLL_ENV, ""), DEPTH_UNROLL_ENV)
    if env is not None:
        return env
    plat = target_platform() if platform is None else _normalise(platform)
    return DEPTH_UNROLL_DEFAULTS.get(plat, False)


def current_execution() -> Execution:
    """The settings a trace started now would use."""
    plat = target_platform()
    return Execution(platform=plat, depth_unroll=resolve_depth_unroll(plat))


@contextlib.contextmanager
def execution(*, platform: str | None = None, depth_unroll: bool | None = None) -> Iterator[Execution]:
    """Set execution settings for everything traced inside the block (``None`` = inherit).

    ::

        with execution(depth_unroll=False):      # force the layer loops, e.g. on a GPU
            out = run_batch(model, params, forcing, state0)

    ``platform`` names the target backend (the runtime sets it; a user sets it when jitting a model
    by hand for a non-default device). Settings are read at trace time (module docstring).
    """
    fields = dict(_CTX.get())
    if platform is not None:
        fields["platform"] = _normalise(platform)
    if depth_unroll is not None:
        fields["depth_unroll"] = bool(depth_unroll)
    token = _CTX.set(MappingProxyType(fields))
    try:
        yield current_execution()
    finally:
        _CTX.reset(token)


def platform_of(*trees: Any) -> str | None:
    """Platform of the first **committed** JAX array among ``trees``, or ``None`` when there is
    none (NumPy inputs, uncommitted arrays, or tracers under an outer transformation). An
    uncommitted array does not decide: JAX runs it on the default device, wherever it lives, so
    the target platform falls back to :func:`target_platform`."""
    for leaf in jax.tree_util.tree_leaves(trees):
        if not isinstance(leaf, jax.Array) or isinstance(leaf, Tracer):
            continue
        if not getattr(leaf, "committed", False):
            continue
        devs = leaf.devices()
        if devs:
            return _normalise(next(iter(devs)).platform)
    return None


@contextlib.contextmanager
def on_platform_of(*trees: Any) -> Iterator[Execution]:
    """:func:`execution` with the platform of ``trees`` (:func:`platform_of`); inherits the
    enclosing target platform when ``trees`` hold no committed array. Used by the runtime."""
    with execution(platform=platform_of(*trees)) as ex:
        yield ex
