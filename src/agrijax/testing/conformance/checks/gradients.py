"""Gradient checks: finite gradients and AD against central finite differences."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.grad import gradient_mode
from agrijax.core.process import (
    Process,
    _covered,
)
from agrijax.core.state import get_path, set_path

from ..case import CALIBRATABLE, ConformanceCase
from ._common import (
    _cast,
    _dtypes,
    _fail,
    _inputs,
    _is_float,
    _main_dtype,
    _paths_leaves,
    _scan,
    _traces,
    _x64,
)
from .metadata import _coefficient_sets


# ------------------------------------------------------------------------------------ gradients
def _wrt_paths(case: ConformanceCase, params: Any) -> list[str]:
    assert case.grad is not None
    out: list[str] = []
    for w in case.grad.wrt:
        if w == CALIBRATABLE:
            for sp, cset in _coefficient_sets(case, params):
                out.extend(f"{sp}.{c}" if sp else c for c in cset.calibratable_paths())
        else:
            out.append(w)
    return list(dict.fromkeys(out))


def _vector(params: Any, paths: Sequence[str], dtype: Any) -> tuple[Any, list[tuple[int, ...]]]:
    parts = [jnp.asarray(get_path(params, q), dtype) for q in paths]
    shapes = [tuple(x.shape) for x in parts]
    return jnp.concatenate([x.ravel() for x in parts]), shapes


def _with_vector(params: Any, paths: Sequence[str], shapes: Sequence[tuple[int, ...]], vec: Any) -> Any:
    i = 0
    for q, shape in zip(paths, shapes, strict=True):
        n = int(np.prod(shape)) if shape else 1
        params = set_path(params, q, jnp.reshape(vec[i : i + n], shape))
        i += n
    return params


def _default_loss(proc: Process) -> Callable[[Any], Any]:
    def loss(final: Any) -> Any:
        leaves = [x for p, x in _paths_leaves(final) if _covered(p, proc.writes) and _is_float(x)]
        if not leaves:
            raise ValueError(f"{proc.name} writes no floating leaf; give GradSpec.loss")
        return sum((jnp.sum(x) for x in leaves), jnp.zeros((), leaves[0].dtype))

    return loss


def _loss_fn(case: ConformanceCase, s0: Any, p: Any, f: Any, paths: Sequence[str], shapes: Any) -> Any:
    assert case.grad is not None
    proc = case.proc
    loss = case.grad.loss or _default_loss(proc)

    def fn(vec: Any) -> Any:
        pp = _with_vector(p, paths, shapes, vec)
        final, _ = _scan(proc, pp, f, s0, case.n_days)
        return loss(final)

    return fn


def check_grad_finite(case: ConformanceCase) -> None:
    """``jax.grad`` with respect to ``GradSpec.wrt`` is finite in the case's gradient mode and in
    the default ``ste`` mode, for every variant and edge variant, in float64 and float32."""
    name = "grad_finite"
    spec = case.grad
    if spec is None:
        return  # the case gives the reason (no_grad), enforced by ConformanceCase
    modes = tuple(dict.fromkeys((spec.mode, "ste")))
    for dtype in _dtypes():
        for variant in (*case.variants, *spec.edge_variants):
            s0, p, f = _inputs(case, _main_dtype(), variant)
            s0, p, f = _cast(s0, dtype), _cast(p, dtype), _cast(f, dtype)
            paths = _wrt_paths(case, p)
            if not paths:
                raise _fail(case, name, "nothing to differentiate; set grad=None with a reason (no_grad)")
            vec, shapes = _vector(p, paths, dtype)
            fn = _loss_fn(case, s0, p, f, paths, shapes)
            for mode in modes:
                with gradient_mode(mode), _traces(case, name, dtype, variant):
                    g = np.asarray(jax.jit(jax.grad(fn))(vec))
                bad = [paths[i] for i in _owner(np.flatnonzero(~np.isfinite(g)), shapes)]
                if bad:
                    raise _fail(
                        case,
                        name,
                        f"non-finite gradient ({jnp.dtype(dtype).name}, mode {mode}, variant {variant!r}) "
                        f"with respect to {bad[:8]}",
                    )


def _owner(flat_idx: Any, shapes: Sequence[tuple[int, ...]]) -> list[int]:
    """Index of the path that owns each flat index of the parameter vector."""
    bounds = np.cumsum([int(np.prod(s)) if s else 1 for s in shapes])
    return sorted({int(np.searchsorted(bounds, i, side="right")) for i in flat_idx})


def check_grad_fd(case: ConformanceCase) -> None:
    """Float64, gradient mode ``GradSpec.mode``: ``grad . d`` against central differences along
    random directions ``d`` (NumPy), with steps ``h`` and ``h / 10``. A disagreement where the two
    differences also disagree is a kink next to the point: the case moves the point or exempts the
    parameter (``fd_exempt``, with the reason)."""
    name = "grad_fd"
    spec = case.grad
    if spec is None or not _x64():
        return  # float64 only; the float32 pass leaves it to the float64 pass
    exempt = {p for p, _ in spec.fd_exempt}
    rng = np.random.default_rng([case.seed, 11])
    for variant in case.variants:
        s0, p, f = _inputs(case, jnp.float64, variant)
        paths = _wrt_paths(case, p)
        if not paths:
            raise _fail(case, name, "nothing to differentiate; set grad=None with a reason (no_grad)")
        vec, shapes = _vector(p, paths, jnp.float64)
        fn = _loss_fn(case, s0, p, f, paths, shapes)
        x = np.asarray(vec)
        keep = np.ones_like(x)
        for i, (q, shape) in enumerate(zip(paths, shapes, strict=True)):
            if q in exempt:
                lo = int(sum(int(np.prod(s)) if s else 1 for s in shapes[:i]))
                keep[lo : lo + (int(np.prod(shape)) if shape else 1)] = 0.0
        scale = np.where(np.abs(x) > 0.0, np.abs(x), 1.0)
        with gradient_mode(spec.mode):  # every trace below happens in the case's mode
            loss = jax.jit(fn)
            g = np.asarray(jax.jit(jax.grad(fn))(vec))
            evals = []
            for _ in range(spec.fd_directions):
                d = rng.standard_normal(x.shape) * scale * keep
                fds = []
                for h in (spec.fd_rel_step, spec.fd_rel_step / 10.0):
                    lp = float(loss(jnp.asarray(x + h * d)))
                    lm = float(loss(jnp.asarray(x - h * d)))
                    fds.append((lp - lm) / (2.0 * h))
                evals.append((d, fds))
        for k, (d, (fd1, fd2)) in enumerate(evals):
            ad = float(g @ d)
            tol = spec.fd_tol
            ref = max(abs(fd1), abs(ad))
            if abs(ad - fd1) <= tol.atol + tol.rtol * ref:
                continue
            kink = abs(fd1 - fd2) > tol.atol + tol.rtol * max(abs(fd1), abs(fd2))
            what = (
                "the two finite differences disagree too: a kink next to the point (move the point or "
                "exempt the parameter with fd_exempt)"
                if kink
                else "AD and finite differences disagree at a smooth point"
            )
            raise _fail(
                case,
                name,
                f"variant {variant!r}, direction {k}: grad.d = {ad:.9e}, FD(h) = {fd1:.9e}, "
                f"FD(h/10) = {fd2:.9e}; {what}",
            )
