"""Run checks: declared writes and reads, balances, transforms (jit, vmap), float32 precision."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from agrijax.core.process import (
    CHECK_ENV,
    ProcessWriteError,
    _covered,
)
from agrijax.core.state import leaf_paths

from ..case import ConformanceCase
from ._common import (
    _EAGER_JIT_RTOL,
    _ULPS,
    _cast,
    _day,
    _describe,
    _diff_leaves,
    _dtypes,
    _eager_run,
    _env,
    _fail,
    _inputs,
    _is_float,
    _jit_run,
    _main_dtype,
    _paths_leaves,
    _pick,
    _rel_close,
    _same_bits,
    _scan,
    _traces,
    _within_ulps,
    _written,
    _x64,
)
from .structure import _step_problems


# ------------------------------------------------------------------------------------ writes and reads
def check_writes(case: ConformanceCase) -> None:
    """With ``AGRI_JAX_CHECK=1``, an eager and a jitted run change only the declared writes."""
    proc = case.proc
    dtype = _main_dtype()
    with _env(CHECK_ENV, "1"):
        for variant in case.variants:
            s0, p, f = _inputs(case, dtype, variant)
            try:
                _eager_run(proc, p, f, s0, case.n_days)
                step = jax.jit(lambda s, pp, ff: proc(s, pp, ff))  # a fresh trace under the check
                s = s0
                for t in range(case.n_days):
                    s = step(s, p, _day(f, t))
                jax.block_until_ready(s)
            except ProcessWriteError as e:
                raise _fail(case, "writes", f"variant {variant!r}: {e}") from None


def _perturb(x: Any, rng: np.random.Generator, kind: str) -> Any:
    a = np.asarray(x)
    if a.dtype == np.bool_:
        return jnp.asarray(~a)
    if np.issubdtype(a.dtype, np.integer):
        return jnp.asarray(a + rng.integers(1, 5, size=a.shape).astype(a.dtype))
    if kind == "nan":
        return jnp.full(a.shape, np.nan, dtype=a.dtype)
    return jnp.asarray(a + (1.0 + np.abs(a)) * rng.uniform(0.5, 2.0, size=a.shape).astype(a.dtype))


def _replace_leaf(tree: Any, index: int, value: Any) -> Any:
    leaves, treedef = jtu.tree_flatten(tree)
    leaves = list(leaves)
    leaves[index] = value
    return jtu.tree_unflatten(treedef, leaves)


def check_reads(case: ConformanceCase) -> None:
    """Perturbation: every state leaf outside the declared reads is replaced by NaN and by a random
    value; the written outputs must stay bit for bit the same (the reads are not under-reported).
    With ``forcing_fields`` declared, the other forcing fields are perturbed too."""
    name = "reads"
    proc = case.proc
    rng = np.random.default_rng([case.seed, 7])
    for variant in case.variants:
        s0, p, f = _inputs(case, _main_dtype(), variant)
        f0 = _day(f, 0)
        step = jax.jit(lambda s, ff, pp=p: proc(s, pp, ff))
        base = step(s0, f0)
        written = _written(proc, base)
        want = _pick(base, written)
        paths = leaf_paths(s0)
        leaks: list[str] = []
        for i, path in enumerate(paths):
            if _covered(path, proc.reads):
                continue
            leaf = jtu.tree_leaves(s0)[i]
            for kind in ("nan", "random"):
                out = _pick(step(_replace_leaf(s0, i, _perturb(leaf, rng, kind)), f0), written)
                changed = [w for w in written if not _same_bits(out[w], want[w])]
                if changed:
                    leaks.append(f"{path} ({kind}) changes {changed[:4]}")
                    break
        if case.forcing_fields and dataclasses.is_dataclass(f0) and not isinstance(f0, type):
            for fld in dataclasses.fields(f0):
                if fld.name in case.forcing_fields or getattr(f0, fld.name, None) is None:
                    continue
                pert = jtu.tree_map(lambda x: _perturb(x, rng, "random"), getattr(f0, fld.name))
                out = _pick(step(s0, dataclasses.replace(f0, **{fld.name: pert})), written)
                changed = [w for w in written if not _same_bits(out[w], want[w])]
                if changed:
                    leaks.append(f"forcing.{fld.name} (not in forcing_fields) changes {changed[:4]}")
        if leaks:
            raise _fail(
                case,
                name,
                f"variant {variant!r}: outputs depend on reads outside {list(proc.reads)}: "
                + "; ".join(leaks),
            )


# ------------------------------------------------------------------------------------ balance
def check_balance(case: ConformanceCase) -> None:
    """Every :class:`~.case.Balance` closes each day, in float64 and float32."""
    name = "balance"
    if not case.balances:
        return  # the case gives the reason (no_balance), enforced by ConformanceCase
    for dtype in _dtypes():
        for variant in case.variants:
            s0, p, f = _inputs(case, dtype, variant)
            with _traces(case, name, dtype, variant):
                _, traj = _jit_run(case.proc, case.n_days)(p, f, s0)
            for b in case.balances:
                tol = b.tol64 if dtype == jnp.float64 else b.tol32
                before = s0
                for t in range(case.n_days):
                    after = _day(traj, t)
                    ft = _day(f, t)
                    s_after = np.asarray(b.storage(after, p), np.float64)
                    ds = s_after - np.asarray(b.storage(before, p), np.float64)
                    fin = np.asarray(b.inflow(before, after, p, ft), np.float64)
                    fout = np.asarray(b.outflow(before, after, p, ft), np.float64)
                    res = ds - (fin - fout)
                    scale = np.abs(s_after) + np.abs(fin) + np.abs(fout)
                    if not bool(np.all(np.abs(res) <= tol.atol + tol.rtol * scale)):
                        raise _fail(
                            case,
                            name,
                            f"{b.quantity} [{b.unit}] does not close on day {t} ({jnp.dtype(dtype).name}, "
                            f"variant {variant!r}): max |dS - (in - out)| = {float(np.max(np.abs(res))):.3e}",
                        )
                    before = after


# ------------------------------------------------------------------------------------ transforms
def _stack(trees: Sequence[Any]) -> Any:
    structs = {jtu.tree_structure(t) for t in trees}
    if len(structs) != 1:
        raise ValueError("samples differ in tree structure (static fields); make() must keep them equal")
    return jtu.tree_map(lambda *xs: jnp.stack(xs), *trees)


def check_transforms(case: ConformanceCase) -> None:
    """Eager against ``jit`` (float64 rtol 1e-12), ``vmap(jit)`` against per-sample ``jit`` (bit
    for bit, or 4 ulp), and batch independence (changing sample k changes no other sample)."""
    name = "transforms"
    proc = case.proc
    dtype = _main_dtype()
    run = _jit_run(proc, case.n_days)
    rtol = _EAGER_JIT_RTOL[np.dtype(dtype)]
    tol = None
    if case.transforms_tol is not None:
        tol = case.transforms_tol[0 if np.dtype(dtype) == np.dtype(np.float64) else 1]
    for variant in case.variants:
        s0, p, f = _inputs(case, dtype, variant)
        eager = _eager_run(proc, p, f, s0, case.n_days)
        jitted, _ = run(p, f, s0)
        bad = _diff_leaves(eager, jitted, _rel_close(rtol) if tol is None else tol.close)
        if bad:
            what = f"rtol {rtol}" if tol is None else f"the case's {tol}"
            raise _fail(
                case,
                name,
                f"variant {variant!r}: eager and jit differ beyond {what} at {_describe(bad, eager, jitted)}",
            )
        samples = [_inputs(case, dtype, variant, sample=k) for k in range(case.batch)]
        try:
            batched = _stack(samples)
        except ValueError as e:
            raise _fail(case, name, str(e)) from None
        vrun = jax.jit(jax.vmap(lambda s, pp, ff: _scan(proc, pp, ff, s, case.n_days)[0]))
        vout = vrun(batched[0], batched[1], batched[2])
        close = tol.close if tol is not None else (None if case.transforms_exact else _within_ulps)
        for k, (sk, pk, fk) in enumerate(samples):
            single, _ = run(pk, fk, sk)
            vk = jtu.tree_map(lambda x, k=k: x[k], vout)
            bad = _diff_leaves(vk, single, close)
            if bad:
                how = (
                    f"within the case's {tol}"
                    if tol is not None
                    else ("bit for bit" if case.transforms_exact else f"within {_ULPS} ulp")
                )
                raise _fail(
                    case,
                    name,
                    f"variant {variant!r}: vmap(jit) sample {k} != jit {how} at {_describe(bad, vk, single)}",
                )
        other = _inputs(case, dtype, variant, sample=case.batch)
        mixed = list(samples)
        mixed[1] = other
        vmix = vrun(*_stack(mixed))
        for k in range(case.batch):
            if k == 1:
                continue
            bad = _diff_leaves(
                jtu.tree_map(lambda x, k=k: x[k], vmix), jtu.tree_map(lambda x, k=k: x[k], vout)
            )
            if bad:
                raise _fail(case, name, f"variant {variant!r}: changing sample 1 changed sample {k} at {bad}")


# ------------------------------------------------------------------------------------ precision
def check_precision(case: ConformanceCase) -> None:
    """The same inputs in float32 and float64 agree within ``case.f32``; float32 in gives float32
    out (no implicit upcast); every output is finite."""
    name = "precision"
    run = _jit_run(case.proc, case.n_days)
    for variant in case.variants:
        if _x64():
            s64, p64, f64 = _inputs(case, jnp.float64, variant)
            s32, p32, f32 = _cast(s64, jnp.float32), _cast(p64, jnp.float32), _cast(f64, jnp.float32)
            out64, _ = run(p64, f64, s64)
        else:
            s32, p32, f32 = _inputs(case, jnp.float32, variant)
            out64 = None
        if jnp.float32 not in _dtypes():  # the float64 part alone (a float32 exemption under x64)
            assert out64 is not None
            bad = [
                p for p, a in _paths_leaves(out64) if _is_float(a) and not np.all(np.isfinite(np.asarray(a)))
            ]
            if bad:
                raise _fail(case, name, f"variant {variant!r}: not finite in float64: {bad[:8]}")
            continue
        with _traces(case, name, jnp.float32, variant):
            problems = _step_problems(case.proc, s32, p32, f32)  # an upcast would break the scan's carry
        if problems:
            raise _fail(case, name, f"variant {variant!r}, float32 inputs: " + "; ".join(problems))
        out32, _ = run(p32, f32, s32)
        for path, b in _paths_leaves(out32):
            if _is_float(b) and not bool(np.all(np.isfinite(np.asarray(b)))):
                problems.append(f"{path}: not finite in float32")
        if out64 is not None:
            for (path, a), b in zip(_paths_leaves(out64), jtu.tree_leaves(out32), strict=True):
                if _is_float(a):
                    if not bool(np.all(np.isfinite(np.asarray(a)))):
                        problems.append(f"{path}: not finite in float64")
                    elif not case.f32.close(b, a):
                        err = float(np.max(np.abs(np.asarray(b, np.float64) - np.asarray(a, np.float64))))
                        problems.append(f"{path}: float32 differs from float64 by {err:.3e}")
                elif not _same_bits(np.asarray(a), np.asarray(b)):
                    problems.append(f"{path}: integer result differs between float32 and float64")
        if problems:
            raise _fail(case, name, f"variant {variant!r}: " + "; ".join(problems))
