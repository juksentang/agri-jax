"""Metadata checks: registry entry, strict lint of the process modules, coefficient labels."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from agrijax.core import lint as _lint
from agrijax.core.coefficients import Coefficients, _split_ref, coefficient_table
from agrijax.core.process import (
    FAITHFUL,
    NO_REFERENCE,
    Process,
    metadata_problems,
    registry,
)
from agrijax.core.state import get_path
from agrijax.core.units import UnitError, parse_unit

from ..case import ConformanceCase
from ._common import (
    _fail,
    _inputs,
    _main_dtype,
)


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
    of the same package that module imports."""
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
    the process function with every rule, decorated or not. Like ``--strict``, the rules of
    :data:`agrijax.core.lint.NOT_STRICT_RULES` (AJ009, AJ011: still open in the soil-water slot) are
    reported by the lint but do not fail the case."""
    files = lint_scope(case.proc)
    if not files:
        raise _fail(case, "lint", f"no source file found for {case.proc.name}")
    own = Path(inspect.getsourcefile(case.proc.fn) or "").resolve()
    fn_name = getattr(case.proc.fn, "__name__", "")
    findings = [
        f
        for path in files
        for f in _lint.lint_file(path, kernel=True, processes=(fn_name,) if path.resolve() == own else ())
        if f.rule not in _lint.NOT_STRICT_RULES
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
