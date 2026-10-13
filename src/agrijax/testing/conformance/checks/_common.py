"""Shared helpers of the conformance checks: dtype parts, input casting, leaf comparison, the
scanned and eager day loops."""

from __future__ import annotations

import contextlib
import contextvars
import functools
import os
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import lax

from agrijax.core.process import (
    Process,
    _covered,
)
from agrijax.core.state import leaf_paths

from ..case import ConformanceCase, ConformanceError, Inputs

#: eager against jit in float64 / float32: XLA rewrites a division by a constant as a multiplication
#: by its reciprocal, so the two are not bit for bit
_EAGER_JIT_RTOL = {np.dtype(np.float64): 1e-12, np.dtype(np.float32): 1e-5}
#: ``vmap(jit)`` against per-sample ``jit`` when a case does not ask for bit identity
_ULPS = 4


# ------------------------------------------------------------------------------------ helpers
def _fail(case: ConformanceCase, check: str, msg: str) -> ConformanceError:
    return ConformanceError(f"[{case.key}] {check}: {msg}")


def _x64() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


#: the dtype part a split check runs (:func:`dtype_part`); ``None``: every dtype
_PART: contextvars.ContextVar[tuple[Any, ...] | None] = contextvars.ContextVar("_PART", default=None)


def _dtypes() -> tuple[Any, ...]:
    every = (jnp.float64, jnp.float32) if _x64() else (jnp.float32,)
    only = _PART.get()
    return every if only is None else tuple(d for d in every if d in only)


def _main_dtype() -> Any:
    """The dtype of the single-dtype checks: float64 with x64, else float32 (not restricted by
    :func:`dtype_part`)."""
    return jnp.float64 if _x64() else jnp.float32


@contextlib.contextmanager
def dtype_part(*dtypes: Any) -> Iterator[None]:
    """Run the per-dtype checks (:data:`~.case.DTYPE_SPLIT_CHECKS`) on ``dtypes`` only."""
    token = _PART.set(tuple(dtypes))
    try:
        yield
    finally:
        _PART.reset(token)


def check_parts(
    case: ConformanceCase, name: str
) -> list[tuple[str, str | None, contextlib.AbstractContextManager[None]]]:
    """``[(label, exemption reason or None, context)]``: how :func:`run_checks` and the pytest
    plugin run check ``name`` on ``case``. A float32 exemption under x64
    (``case.exempt_float32_x64``) splits the check into its float64 part (must pass) and its
    float32 part (must fail)."""
    why32 = case.exempt_float32_x64.get(name) if _x64() else None
    if why32 is None:
        return [("", case.exempt_checks.get(name), contextlib.nullcontext())]
    return [
        ("float64 part", None, dtype_part(jnp.float64)),
        ("float32 part", f"float32 under x64: {why32}", dtype_part(jnp.float32)),
    ]


def _is_float(x: Any) -> bool:
    return hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)


def _is_array_leaf(x: Any) -> bool:
    return isinstance(x, (jax.Array, np.ndarray, np.generic, float, int)) and not isinstance(x, bool)


def _to_jax(tree: Any) -> Any:
    return jtu.tree_map(lambda x: jnp.asarray(x) if isinstance(x, (np.ndarray, np.generic)) else x, tree)


def _cast(tree: Any, dtype: Any) -> Any:
    """Every floating array leaf of ``tree`` as ``dtype`` (integers, bools and Python floats unchanged)."""
    return jtu.tree_map(lambda x: jnp.asarray(x, dtype) if _is_float(x) else x, tree)


def _inputs(case: ConformanceCase, dtype: Any, variant: str | None = None, sample: int = 0) -> Inputs:
    s, p, f = case.inputs(dtype, variant, sample)
    return _to_jax(s), _to_jax(p), _to_jax(f)


def _day(forcing: Any, t: int) -> Any:
    return jtu.tree_map(lambda x: x[t], forcing)


def _paths_leaves(tree: Any) -> list[tuple[str, Any]]:
    return list(zip(leaf_paths(tree), jtu.tree_leaves(tree), strict=True))


def _same_bits(a: Any, b: Any) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _diff_leaves(a: Any, b: Any, close: Callable[[Any, Any], bool] | None = None) -> list[str]:
    """Leaf paths where ``a`` and ``b`` differ (bitwise unless ``close`` is given)."""
    if jtu.tree_structure(a) != jtu.tree_structure(b):
        return ["<tree structure>"]
    out = []
    for (path, x), y in zip(_paths_leaves(a), jtu.tree_leaves(b), strict=True):
        if close is None or not _is_float(np.asarray(x)):
            if not _same_bits(x, y):
                out.append(path)
        elif not close(x, y):
            out.append(path)
    return out


def _ulps(a: Any, b: Any) -> float:
    """Largest distance between ``a`` and ``b`` in units in the last place of the larger."""
    a, b = np.asarray(a), np.asarray(b)
    if not np.issubdtype(a.dtype, np.floating) or a.shape != b.shape:
        return float("nan")
    spacing = np.spacing(np.maximum(np.abs(a), np.abs(b)).astype(a.dtype))
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)) / spacing.astype(np.float64)
    return float(np.nanmax(d, initial=0.0))


def _describe(paths: Sequence[str], a: Any, b: Any) -> str:
    """``path (n ulp)`` for each differing leaf of ``a`` against ``b``."""
    table_a = dict(_paths_leaves(a))
    table_b = dict(_paths_leaves(b))
    out = []
    for p in paths:
        if p in table_a and p in table_b:
            out.append(f"{p} ({_ulps(table_a[p], table_b[p]):.0f} ulp)")
        else:
            out.append(p)
    return ", ".join(out)


def _within_ulps(a: Any, b: Any, n: int = _ULPS) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    tol = n * np.spacing(np.maximum(np.abs(a), np.abs(b)).astype(a.dtype))
    return bool(np.all(both_nan | (np.abs(a - b) <= tol)))


def _rel_close(rtol: float) -> Callable[[Any, Any], bool]:
    """Close relative to the largest magnitude of the leaf (a sum near zero rounds absolutely)."""

    def close(a: Any, b: Any) -> bool:
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        if a.shape != b.shape:
            return False
        scale = max(float(np.max(np.abs(b), initial=0.0)), np.finfo(np.float64).tiny)
        return bool(np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= rtol * scale)))

    return close


@contextlib.contextmanager
def _env(name: str, value: str) -> Iterator[None]:
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


#: the TypeError of a loop whose carry changes type (what :func:`_traces` reports)
_CARRY_TYPES = re.compile(r"carry|must have (equal|identical) types")


@contextlib.contextmanager
def _traces(case: ConformanceCase, check: str, dtype: Any, variant: str) -> Iterator[None]:
    """A run that does not trace (a scan whose carry changes dtype: an implicit upcast of float32
    inputs under x64) is a failure of ``check``, reported as :class:`ConformanceError`."""
    try:
        yield
    except TypeError as e:
        if not _CARRY_TYPES.search(str(e)):
            raise
        first = " ".join(line.strip() for line in str(e).splitlines()[:4] if line.strip())
        raise _fail(
            case,
            check,
            f"{jnp.dtype(dtype).name} inputs, variant {variant!r}: the run does not trace ({first})",
        ) from None


def _step_all(procs: Sequence[Process]) -> Callable[[Any, Any, Any], Any]:
    def step(s: Any, p: Any, f: Any) -> Any:
        for q in procs:
            s = q(s, p, f)
        return s

    return step


def _keep_dtypes(new: Any, old: Any) -> Any:
    """``new`` with every leaf cast back to the dtype of ``old`` where they differ (a trace-time
    choice: no operation is added when the dtypes agree). An upcast is reported by
    :func:`check_precision`; the other checks go on with the state's dtypes."""
    if jtu.tree_structure(new) != jtu.tree_structure(old):
        return new
    return jtu.tree_map(
        lambda a, b: (
            a.astype(b.dtype) if hasattr(a, "dtype") and hasattr(b, "dtype") and a.dtype != b.dtype else a
        ),
        new,
        old,
    )


def _scan(step: Callable[[Any, Any, Any], Any], params: Any, forcing: Any, state0: Any, n: int) -> Any:
    """``(final state, per-day states)`` of ``n`` days of ``step``."""

    def body(s: Any, f_t: Any) -> tuple[Any, Any]:
        s1 = _keep_dtypes(step(s, params, f_t), s)
        return s1, s1

    return lax.scan(body, state0, forcing, length=n)


@functools.cache
def _jit_run(proc: Process, n: int) -> Callable[..., Any]:
    """Jitted ``(params, forcing, state0) -> (final, trajectory)`` of ``proc`` (one per process)."""
    return jax.jit(lambda p, f, s: _scan(proc, p, f, s, n))


def _eager_run(proc: Process, params: Any, forcing: Any, state0: Any, n: int) -> Any:
    s = state0
    for t in range(n):
        s = proc(s, params, _day(forcing, t))
    return s


def _written(proc: Process, tree: Any) -> list[str]:
    return [p for p in leaf_paths(tree) if _covered(p, proc.writes)]


def _pick(tree: Any, paths: Sequence[str]) -> dict[str, Any]:
    names = leaf_paths(tree)
    leaves = jtu.tree_leaves(tree)
    table = dict(zip(names, leaves, strict=True))
    return {p: table[p] for p in paths}


def _under(path: str, base: str) -> bool:
    return path == base or path.startswith(base + ".")
