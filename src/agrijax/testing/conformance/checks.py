"""The generic checks of the conformance kit (M3 coupling contract, section 4.4).

Each ``check_x(case)`` is a plain function of a :class:`~.case.ConformanceCase`: it returns
``None`` or raises :class:`~.case.ConformanceError` naming what differs. No check needs pytest or
data; every input comes from the case's ``make`` with a NumPy generator. :data:`CHECKS` lists them
in the order of the contract's table; :func:`run_checks` runs them outside pytest.

The checks run the process over the case's ``n_days`` with ``lax.scan`` (the runtime's day loop),
except where a check needs concrete values (the eager runs of :func:`check_writes` and
:func:`check_transforms`). Float64 parts run when JAX has x64 enabled; with ``AGRI_JAX_X64=0``
the float32 parts still run and the float64 comparisons are left to the float64 pass.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib.util
import inspect
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import lax

from agrijax.core import lint as _lint
from agrijax.core.coefficients import Coefficients, _split_ref, coefficient_table
from agrijax.core.dims import DimsError, check_tree_dims
from agrijax.core.grad import gradient_mode
from agrijax.core.ports import Binding, BindingError, bind, compose, detach, port_names
from agrijax.core.process import (
    CHECK_ENV,
    FAITHFUL,
    NO_REFERENCE,
    Process,
    ProcessWriteError,
    _covered,
    metadata_problems,
    process,
    registry,
)
from agrijax.core.state import get_path, leaf_paths, set_path
from agrijax.core.units import UnitError, parse_unit
from agrijax.iface.contract import PORTS, PortSpec

from .case import CALIBRATABLE, ConformanceCase, ConformanceError, Inputs
from .contracts import slot_contract

__all__ = [
    "CHECKS",
    "check_balance",
    "check_binding",
    "check_coefficients",
    "check_grad_fd",
    "check_grad_finite",
    "check_lint",
    "check_precision",
    "check_reads",
    "check_registry",
    "check_shapes_dims",
    "check_slot_contract",
    "check_transforms",
    "check_units",
    "check_writes",
    "run_checks",
]

#: eager against jit in float64 / float32: XLA rewrites a division by a constant as a multiplication
#: by its reciprocal, so the two are not bit for bit (M3 minimal set, core item)
_EAGER_JIT_RTOL = {np.dtype(np.float64): 1e-12, np.dtype(np.float32): 1e-5}
#: ``vmap(jit)`` against per-sample ``jit`` when a case does not ask for bit identity
_ULPS = 4


# ------------------------------------------------------------------------------------ helpers
def _fail(case: ConformanceCase, check: str, msg: str) -> ConformanceError:
    return ConformanceError(f"[{case.key}] {check}: {msg}")


def _x64() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def _dtypes() -> tuple[Any, ...]:
    return (jnp.float64, jnp.float32) if _x64() else (jnp.float32,)


def _main_dtype() -> Any:
    return _dtypes()[0]


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


# ------------------------------------------------------------------------------------ registry
def check_registry(case: ConformanceCase) -> None:
    """Registry metadata complete and consistent with the reference and its licence."""
    name = "registry"
    proc = case.proc
    problems = list(metadata_problems(proc))
    info = proc.info
    if info is None:
        raise _fail(case, name, "; ".join(problems))
    if str(info.key) != case.key:
        problems.append(f"the process is registered as {info.key}, the case says {case.key}")
    key = info.key
    if key.needs_faithful_sibling:
        sibling = str(key.faithful)
        if sibling not in registry.keyed():
            problems.append(f"variant without its faithful sibling {sibling} in the registry")
        if not info.deviates:
            problems.append("a non-faithful variant lists no deviations")
    if key.ref_version != NO_REFERENCE:
        try:
            ref, _ = _split_ref(key.ref_version)
        except ValueError as e:
            problems.append(str(e))
            ref = None
        if ref is not None:
            if info.provenance == "translated_bsd3" and "BSD-3" not in ref.licence:
                problems.append(f"translated_bsd3 but the reference {ref.name} is under {ref.licence!r}")
            if info.provenance == "reference_only_conventions" and ref.statement_allowed:
                problems.append(
                    f"reference_only_conventions but {ref.name} allows quoting its source ({ref.licence})"
                )
        if key.variant == FAITHFUL and not info.ref_build.strip():
            problems.append("a faithful process of a reference names no ref_build")
    doc = proc.doc or inspect.getdoc(proc.fn) or ""
    if not any(line.strip().lower().startswith("source:") for line in doc.splitlines()):
        problems.append("docstring has no 'Source:' line")
    if not case.origin.strip():
        problems.append("the case does not record its origin (distribution and version)")
    if problems:
        raise _fail(case, name, "; ".join(problems))


# ------------------------------------------------------------------------------------ lint
def _module_file(module: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin)


def lint_scope(proc: Process) -> list[Path]:
    """The files :func:`check_lint` lints for ``proc``: its module and, transitively, every module
    of the same package that module imports (M3 contract, section 3 item 5)."""
    import ast

    module = getattr(proc.fn, "__module__", None)
    if not module:
        return []
    package = module.rpartition(".")[0]
    seen: dict[str, Path] = {}
    todo = [module]
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        path = _module_file(mod)
        if path is None:
            continue
        seen[mod] = path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        here = mod if path.name == "__init__.py" else mod.rpartition(".")[0]
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    try:
                        base = importlib.util.resolve_name("." * node.level + base, here)
                    except ImportError:
                        continue
                names = [base] + [f"{base}.{a.name}" for a in node.names]
            for n in names:
                if package and (n == package or n.startswith(package + ".")) and _module_file(n):
                    todo.append(n)
    return list(seen.values())


def check_lint(case: ConformanceCase) -> None:
    """``agrijax.core.lint --strict`` on the process module and the same-package modules it
    imports, every function checked as a kernel (so a plugin outside ``processes/`` is covered) and
    the process function with every rule, decorated or not."""
    files = lint_scope(case.proc)
    if not files:
        raise _fail(case, "lint", f"no source file found for {case.proc.name}")
    own = Path(inspect.getsourcefile(case.proc.fn) or "").resolve()
    fn_name = getattr(case.proc.fn, "__name__", "")
    findings = [
        f
        for path in files
        for f in _lint.lint_file(path, kernel=True, processes=(fn_name,) if path.resolve() == own else ())
    ]
    if findings:
        raise _fail(case, "lint", "\n" + "\n".join(f.format() for f in findings))


# ------------------------------------------------------------------------------------ coefficients
def _coefficient_sets(case: ConformanceCase, params: Any) -> list[tuple[str, Coefficients]]:
    if case.coefficient_sets is not None:
        out = []
        for path in case.coefficient_sets:
            value = get_path(params, path) if path else params
            if not isinstance(value, Coefficients):
                raise _fail(
                    case,
                    "coefficients",
                    f"params.{path} is {type(value).__name__}, not a Coefficients set (pass the set in "
                    "the case's params so it can be checked and differentiated)",
                )
            out.append((path, value))
        return out
    found: list[tuple[str, Coefficients]] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, Coefficients):
            found.append((prefix, node))
            return
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for f in dataclasses.fields(node):
                walk(getattr(node, f.name, None), f"{prefix}.{f.name}" if prefix else f.name)
        elif isinstance(node, Mapping):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))

    walk(params, "")
    return found


def check_coefficients(case: ConformanceCase) -> None:
    """Every coefficient: unit parses, provenance given, same reference as the key (or ``none``,
    or a note), default inside its bounds."""
    _, params, _ = _inputs(case, _main_dtype())
    key_ref = case.parsed_key.ref_version
    problems: list[str] = []
    for set_path_, cset in _coefficient_sets(case, params):
        for row in coefficient_table(cset):
            where = f"{set_path_}.{row['path']}" if set_path_ else row["path"]
            try:
                parse_unit(row["unit"])
            except UnitError as e:
                problems.append(f"{where}: {e}")
            if not str(row["source"]).strip():
                problems.append(f"{where}: no provenance")
            ref = row["ref_version"]
            if ref not in (key_ref, NO_REFERENCE) and not str(row["note"]).strip():
                problems.append(f"{where}: reference {ref} differs from the key's {key_ref} without a note")
            if row["bounds"] is not None:
                lo, hi = row["bounds"]
                v = np.asarray(row["value"], np.float64)
                if not bool(np.all((v >= lo) & (v <= hi))):
                    problems.append(f"{where}: value {row['value']} outside bounds {row['bounds']}")
    if problems:
        raise _fail(case, "coefficients", "; ".join(problems))


# ------------------------------------------------------------------------------------ shapes and units
def _walk_fields(node: Any, prefix: str, visit: Callable[[str, dataclasses.Field[Any], Any], None]) -> None:
    """Call ``visit(path, field, value)`` for every non-static field of every dataclass in ``node``."""
    if node is None:
        return
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        for f in dataclasses.fields(node):
            if f.metadata.get("static", False):
                continue
            value = getattr(node, f.name, None)
            path = f"{prefix}.{f.name}" if prefix else f.name
            if value is None:
                continue
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                _walk_fields(value, path, visit)
            elif isinstance(value, Mapping):
                for k, v in value.items():
                    _walk_fields(v, f"{path}.{k}", visit)
            elif isinstance(value, (list, tuple)) and not _is_array_leaf(value):
                for i, v in enumerate(value):
                    if dataclasses.is_dataclass(v):
                        _walk_fields(v, f"{path}.{i}", visit)
                    else:
                        visit(f"{path}.{i}", f, v)
            else:
                visit(path, f, value)
    elif isinstance(node, Mapping):
        for k, v in node.items():
            _walk_fields(v, f"{prefix}.{k}" if prefix else str(k), visit)


def _is_static_field(f: dataclasses.Field[Any]) -> bool:
    return bool(f.metadata.get("static", False))


def _step_problems(proc: Process, s0: Any, p: Any, f: Any, after: Any = None) -> list[str]:
    """Tree structure, shape or dtype changes between ``s0`` and one jitted step (or ``after``)."""
    if after is None:
        after = jax.jit(lambda s, pp, ff: proc(s, pp, ff))(s0, p, _day(f, 0))
    if jtu.tree_structure(after) != jtu.tree_structure(s0):
        return ["the state's tree structure changed"]
    out = []
    for (path, a), b in zip(_paths_leaves(s0), jtu.tree_leaves(after), strict=True):
        if np.shape(a) != np.shape(b):
            out.append(f"{path}: shape {np.shape(a)} -> {np.shape(b)}")
        elif jnp.result_type(a) != jnp.result_type(b):
            out.append(f"{path}: {jnp.result_type(a)} in, {jnp.result_type(b)} out")
    return out


def check_shapes_dims(case: ConformanceCase) -> None:
    """Declared dims match the shapes, every array leaf declares its dims, and ``n_days`` days keep
    the state's tree structure, shapes and dtypes."""
    name = "shapes_dims"
    dtype = _main_dtype()
    for variant in case.variants:
        s0, p, f = _inputs(case, dtype, variant)
        try:
            check_tree_dims({"params": p, "forcing": f, "state": s0})
        except DimsError as e:
            raise _fail(case, name, f"variant {variant!r}: {e}") from None
        undeclared: list[str] = []

        def visit(path: str, fld: dataclasses.Field[Any], value: Any, _u: list[str] = undeclared) -> None:
            if _is_array_leaf(value) and fld.metadata.get("dims") is None:
                _u.append(path)

        for root, tree in (("state", s0), ("params", p), ("forcing", f)):
            _walk_fields(tree, root, visit)
        if undeclared:
            raise _fail(case, name, f"variant {variant!r}: array leaves without declared dims: {undeclared}")
        leaves = jtu.tree_leaves(f)
        if leaves and any(np.shape(x)[:1] != (case.n_days,) for x in leaves):
            raise _fail(case, name, f"variant {variant!r}: forcing leaves need a time axis of {case.n_days}")
        problems = _step_problems(case.proc, s0, p, f)
        if problems:
            raise _fail(case, name, f"variant {variant!r}: " + "; ".join(problems))
        final, _ = _jit_run(case.proc, case.n_days)(p, f, s0)
        problems = _step_problems(case.proc, s0, p, f, after=final)
        if problems:
            raise _fail(case, name, f"variant {variant!r}, after {case.n_days} days: " + "; ".join(problems))


def _contract_port(target: str, slot: str) -> PortSpec | None:
    for spec in PORTS.values():
        try:
            path = spec.global_path(slot)
        except ValueError:
            continue
        if spec.kind == "state" and path == target:
            return spec
    return None


def _record_vs_spec(cls: type, spec: PortSpec) -> list[str]:
    """Field metadata of record class ``cls`` against the contract's fields (unit strings equal,
    dims equal, grid equal when the contract fixes one)."""
    meta = {f.name: f.metadata for f in dataclasses.fields(cls)}
    out = []
    for fname, fs in spec.fields:
        m = meta.get(fname)
        if m is None:
            out.append(f"{spec.id}.{fname}: missing in {cls.__name__}")
            continue
        if m.get("unit") != fs.unit:
            out.append(f"{spec.id}.{fname}: unit {m.get('unit')!r}, contract {fs.unit!r}")
        else:
            try:
                if not parse_unit(m["unit"]).same_dimension(parse_unit(fs.unit)):
                    out.append(f"{spec.id}.{fname}: dimension differs from {fs.unit!r}")
            except UnitError as e:
                out.append(f"{spec.id}.{fname}: {e}")
        if m.get("dims") != fs.dims:
            out.append(f"{spec.id}.{fname}: dims {m.get('dims')!r}, contract {fs.dims!r}")
        if fs.grid is not None and m.get("grid") != fs.grid:
            out.append(f"{spec.id}.{fname}: grid {m.get('grid')!r}, contract {fs.grid!r}")
    return out


def check_units(case: ConformanceCase) -> None:
    """Every array field declares a unit that parses; every bound port's record matches the
    contract field by field (unit string, dimension, dims, grid)."""
    name = "units"
    s0, p, f = _inputs(case, _main_dtype())
    problems: list[str] = []

    def visit(path: str, fld: dataclasses.Field[Any], value: Any) -> None:
        if not _is_array_leaf(value):
            return
        unit = fld.metadata.get("unit", "")
        if not str(unit).strip():
            problems.append(f"{path}: no unit")
            return
        try:
            parse_unit(unit)
        except UnitError as e:
            problems.append(f"{path}: {e}")

    for root, tree in (("state", s0), ("params", p), ("forcing", f)):
        _walk_fields(tree, root, visit)
    for port_field, target in case.port_map.items():
        spec = _contract_port(target, case.crop_slot)
        if spec is None:
            problems.append(f"port {port_field} -> {target}: not a port of the coupling contract")
            continue
        value = getattr(s0, port_field, None)
        if value is None:
            problems.append(f"port {port_field}: the case's state0 leaves it empty")
            continue
        if spec.record is not None:
            problems.extend(_record_vs_spec(type(value), spec))
    if problems:
        raise _fail(case, name, "; ".join(problems))


# ------------------------------------------------------------------------------------ slot contract
def _under(path: str, base: str) -> bool:
    return path == base or path.startswith(base + ".")


def check_slot_contract(case: ConformanceCase) -> None:
    """Bound reads and writes stay inside the own subtree and the slot's contract ports; ``in``
    ports are never written; each bound port has the contract's path and record class."""
    name = "slot_contract"
    cname = case.contract_name
    if cname is None:
        return
    contract = slot_contract(cname)
    proc = case.proc
    s0, _, _ = _inputs(case, _main_dtype())
    cls = type(s0)
    ports = port_names(cls)
    pmap = case.port_map
    problems: list[str] = []
    unknown = sorted(set(pmap) - set(ports))
    if unknown:
        raise _fail(case, name, f"{cls.__name__} has no port fields {unknown}")
    for path in (*proc.reads, *proc.writes):
        head = path.split(".", 1)[0]
        if head in ports and head not in pmap:
            problems.append(f"{proc.name} uses port {head!r} ({path}) but the case binds no global path")
        if path == "*":
            problems.append(f"{proc.name} declares the wildcard {path!r}; a slot process names its paths")
    allowed: dict[str, Any] = {}
    for sp in contract.ports:
        try:
            allowed[sp.spec.global_path(case.crop_slot)] = sp
        except ValueError:
            continue
    for port_field, target in pmap.items():
        sp = allowed.get(target)
        if sp is None:
            problems.append(f"port {port_field} -> {target}: not a port of slot {cname!r} {sorted(allowed)}")
            continue
        value = getattr(s0, port_field, None)
        if sp.spec.record is not None and value is not None and type(value) is not sp.spec.record:
            problems.append(
                f"port {port_field}: record {type(value).__name__}, the contract's {sp.port} is "
                f"{sp.spec.record.__name__}"
            )
    if problems:
        raise _fail(case, name, "; ".join(problems))
    bound = bind(proc, own=case.own_path, ports=pmap)
    own = case.own_path
    for r in bound.reads:
        if not (_under(r, own) or any(_under(r, t) for t in allowed)):
            problems.append(f"reads {r}: outside {own} and the ports of slot {cname!r}")
    for w in bound.writes:
        if _under(w, own):
            continue
        hit = [(t, sp) for t, sp in allowed.items() if _under(w, t) or _under(t, w)]
        if not hit:
            problems.append(f"writes {w}: outside {own} and the ports of slot {cname!r}")
            continue
        for t, sp in hit:
            rest = w[len(t) + 1 :] if w.startswith(t + ".") else None
            fname = rest.split(".", 1)[0] if rest else None
            if not sp.writable(fname):
                what = "an 'in' port" if sp.direction == "in" else f"field {fname or '<record>'} of {sp.port}"
                problems.append(f"writes {w}: {what} (slot {cname!r} may write {sp.writes or 'all fields'})")
    if problems:
        raise _fail(case, name, "; ".join(problems))


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
    for variant in case.variants:
        s0, p, f = _inputs(case, dtype, variant)
        eager = _eager_run(proc, p, f, s0, case.n_days)
        jitted, _ = run(p, f, s0)
        bad = _diff_leaves(eager, jitted, _rel_close(rtol))
        if bad:
            raise _fail(case, name, f"variant {variant!r}: eager and jit differ beyond rtol {rtol} at {bad}")
        samples = [_inputs(case, dtype, variant, sample=k) for k in range(case.batch)]
        try:
            batched = _stack(samples)
        except ValueError as e:
            raise _fail(case, name, str(e)) from None
        vrun = jax.jit(jax.vmap(lambda s, pp, ff: _scan(proc, pp, ff, s, case.n_days)[0]))
        vout = vrun(batched[0], batched[1], batched[2])
        close = None if case.transforms_exact else _within_ulps
        for k, (sk, pk, fk) in enumerate(samples):
            single, _ = run(pk, fk, sk)
            vk = jtu.tree_map(lambda x, k=k: x[k], vout)
            bad = _diff_leaves(vk, single, close)
            if bad:
                how = "bit for bit" if case.transforms_exact else f"within {_ULPS} ulp"
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
                with gradient_mode(mode):
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


# ------------------------------------------------------------------------------------ binding
def _producer(port_field: str, target: str) -> Process:
    def produce(state: Any, params: Any, forcing_t: Any) -> Any:
        """Kit fixture: the port record of today, computed in the same step (coupled binding).

        Source: conformance kit fixture.
        """
        rec0 = params["rec0"][port_field]
        factor = forcing_t["factor"]
        rec = jtu.tree_map(lambda x: x * factor.astype(x.dtype) if _is_float(x) else x, rec0)
        return set_path(state, target, rec)

    return process(produce, reads=(), writes=(target,), register=False, name=f"kit.produce.{port_field}")


def _replayer(port_field: str, target: str) -> Process:
    def replay(state: Any, params: Any, forcing_t: Any) -> Any:
        """Kit fixture: the port record of today, read from data (replay binding).

        Source: conformance kit fixture.
        """
        return set_path(state, target, forcing_t["replay"][port_field])

    return process(replay, reads=(), writes=(target,), register=False, name=f"kit.replay.{port_field}")


def check_binding(case: ConformanceCase) -> None:
    """Ports: the bound process equals the process on its own; a replay binding (port records
    from data) equals a coupled binding (records produced in the same step) bit for bit; a port
    the process uses but the binding leaves out raises :class:`~agrijax.core.ports.BindingError`."""
    name = "binding"
    proc = case.proc
    s0, p, f = _inputs(case, _main_dtype())
    ports = port_names(type(s0))
    pmap = case.port_map
    if not ports:
        return
    used = sorted({q.split(".", 1)[0] for q in (*proc.reads, *proc.writes)} & set(ports))
    if not pmap:
        if used:
            raise _fail(case, name, f"{proc.name} uses ports {used} and the case binds none")
        return
    own = case.own_path
    close = None if case.transforms_exact else _within_ulps
    binding = Binding(own, tuple(pmap.items()))
    g0 = compose(binding.entries(s0))
    bound = bind(proc, own=own, ports=pmap)

    # 1. bound == alone
    final_b, _ = jax.jit(lambda pp, ff, gg: _scan(bound, pp, ff, gg, case.n_days))(p, f, g0)
    final_a, _ = _jit_run(proc, case.n_days)(p, f, s0)
    bad = _diff_leaves(get_path(final_b, own), detach(final_a), close)
    for port_field, target in pmap.items():
        diff = _diff_leaves(get_path(final_b, target), getattr(final_a, port_field), close)
        bad += [f"{port_field}.{x}" for x in diff]
    if bad:
        raise _fail(case, name, f"bound and unbound runs differ at {bad}")

    # 2. replay == coupled, for every port the process reads and does not write
    reads_in = [
        q
        for q in used
        if any(_under(r, q) for r in proc.reads)
        and not any(_under(w, q) or _under(q, w) for w in proc.writes)
    ]
    if reads_in:
        rng = np.random.default_rng([case.seed, 13])
        dt = _main_dtype()
        factor = jnp.asarray(1.0 + 0.01 * rng.uniform(-1.0, 1.0, case.n_days), dt)
        rec0 = {q: getattr(s0, q) for q in reads_in}
        gp = {"module": p, "rec0": rec0}
        inner = bind(proc, own=own, ports=pmap, params="module", forcing="module", name="kit.module")
        producers = [_producer(q, pmap[q]) for q in reads_in]
        coupled = _step_all([*producers, inner])

        def run_coupled(pp: Any, ff: Any, gg: Any) -> Any:
            def body(s: Any, f_t: Any) -> tuple[Any, Any]:
                s1 = coupled(s, pp, f_t)
                return s1, {q: get_path(s1, pmap[q]) for q in reads_in}

            return lax.scan(body, gg, ff, length=case.n_days)

        final_c, recs = jax.jit(run_coupled)(gp, {"module": f, "factor": factor}, g0)
        replay = _step_all([*(_replayer(q, pmap[q]) for q in reads_in), inner])
        final_r, _ = jax.jit(lambda pp, ff, gg: _scan(replay, pp, ff, gg, case.n_days))(
            gp, {"module": f, "replay": recs}, g0
        )
        bad = _diff_leaves(get_path(final_c, own), get_path(final_r, own))
        for port_field, target in pmap.items():
            if port_field not in reads_in:
                diff = _diff_leaves(get_path(final_c, target), get_path(final_r, target))
                bad += [f"{port_field}.{x}" for x in diff]
        if bad:
            raise _fail(case, name, f"replay and coupled bindings differ at {bad}")

    # 3. a used port left unbound fails at trace time
    f0 = _day(f, 0)
    for q in used:
        rest = {k: v for k, v in pmap.items() if k != q}
        partial = bind(proc, own=own, ports=rest)
        g_partial = compose(Binding(own, tuple(rest.items())).entries(s0))
        try:
            partial(g_partial, p, f0)
        except BindingError:
            continue
        raise _fail(case, name, f"port {q!r} left unbound did not raise BindingError")


#: every check, in the order of the contract's table (M3 coupling contract, section 4.4)
CHECKS: tuple[Callable[[ConformanceCase], None], ...] = (
    check_registry,
    check_lint,
    check_coefficients,
    check_shapes_dims,
    check_units,
    check_slot_contract,
    check_writes,
    check_reads,
    check_balance,
    check_transforms,
    check_precision,
    check_grad_finite,
    check_grad_fd,
    check_binding,
)


def check_name(check: Callable[..., Any]) -> str:
    """``check_reads`` -> ``reads`` (the name used by ``exempt_checks`` and the pytest ids)."""
    return check.__name__.removeprefix("check_")


def run_checks(
    case: ConformanceCase, checks: Sequence[Callable[[ConformanceCase], None]] = CHECKS
) -> dict[str, str | None]:
    """Run ``checks`` on ``case`` outside pytest: ``{check name: None if it passed, else the
    message}``. An exempted check (``case.exempt_checks``) that fails reports ``"exempt: ..."``;
    one that passes is reported as a failure (the exemption is stale)."""
    out: dict[str, str | None] = {}
    for check in checks:
        n = check_name(check)
        try:
            check(case)
        except ConformanceError as e:
            out[n] = f"exempt: {case.exempt_checks[n]}: {e}" if n in case.exempt_checks else str(e)
        else:
            stale = n in case.exempt_checks
            out[n] = f"exempt check passes; remove the exemption ({case.exempt_checks[n]})" if stale else None
    return out
