"""Standalone AST lint for the three process rules.

Scope
-----
* every function decorated with ``@process`` gets every rule;
* every other function defined in a file under a ``processes/`` directory (the numerical
  kernels the processes call: ``shuttleworth_wallace``, ``theta_of_h``, private helpers, ...)
  gets the numerical rules AJ001-AJ003 and AJ006; AJ004/AJ005 are about the process contract
  and do not apply to kernels that return NamedTuples or arrays;
* ``--all`` applies every rule to every function in every file.

Arguments that are static by convention are not traced: ``self``/``cls``, and arguments
annotated ``bool``, ``int``, ``str``, ``float`` or ``Literal[...]`` (optionally ``| None``);
traced inputs are annotated ``ArrayLike`` / ``Array`` / a pytree class. Python-level tests on
shapes (``x.shape``, ``x.ndim``, ``np.ndim(x)``, ``len(x)``) and comparisons with a string
constant are also static and never reported by AJ001.

This module deliberately imports nothing from JAX so that pre-commit can run it
in a bare interpreter.

Rules
-----
AJ001  error    ``if``/``while`` (and ``x if c else y``) whose condition references a
                name derived from the function arguments (state / params / forcing).
AJ002  error    Subscript assignment (``a[i] = ...``, ``a[i] += ...``) inside a ``for`` body.
AJ003  warning  In ``jnp.where(c, a, b)`` / ``jnp.select``, a branch contains ``log``,
                ``sqrt`` or ``/`` whose operand is not guarded by ``maximum``/``clip``.
AJ004  warning  A ``return`` value that is not obtained via ``eqx.tree_at`` / ``replace``.
AJ005  warning  Missing docstring, or docstring without a ``Source:`` line.
AJ006  error    A Python ``for`` loop or comprehension over ``range(...)`` / ``arange(...)`` whose
                bound is shape-derived: ``x.shape``, ``x.size``, ``np.shape(x)``, a grid size
                attribute (``n_node``, ``n_layer``, ``n_slice``, ...), ``len(<traced name>)``, or a
                name assigned from one of these; or directly over an array of traced values
                (``for t in state.theta``, ``enumerate(grid.tl)``, a local from array arithmetic
                or a ``jnp`` call). Such a loop unrolls over layers at trace time;
                vectorise it, or write a true recurrence with
                :func:`agrijax.core.depth_scan.depth_scan`. Loops over literal or configuration
                counts (``range(3)``, ``range(cfg.n_iter)``) are not reported.

Usage::

    python -m agrijax.core.lint src/agrijax/processes [--all] [--strict]
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "ALL_RULES",
    "KERNEL_RULES",
    "RULES",
    "CheckedFunction",
    "Finding",
    "checked_functions",
    "lint_file",
    "lint_paths",
    "lint_source",
    "main",
]

RULES: dict[str, tuple[str, str]] = {
    "AJ001": ("error", "Python branch on a value derived from the process arguments"),
    "AJ002": ("error", "subscript assignment inside a for loop"),
    "AJ003": ("warning", "unguarded log/sqrt/division inside a where/select branch"),
    "AJ004": ("warning", "return value not built with eqx.tree_at / replace"),
    "AJ005": ("warning", "missing docstring or no 'Source:' line"),
    "AJ006": ("error", "Python loop over a shape-derived range or an array (unrolled layer loop)"),
}

_RISKY_CALLS = {"log", "log2", "log10", "sqrt", "rsqrt", "power", "pow", "arccos", "arcsin", "arctanh"}
_GUARD_CALLS = {"maximum", "clip", "clamp", "minimum", "abs", "exp", "where", "select", "square", "softplus"}
_UPDATE_CALLS = {"tree_at", "replace", "set"}
_WHERE_CALLS = {"where", "select"}
_PROCESS_DECORATOR = "process"
#: rules applied to non-``@process`` functions of ``processes/`` modules (numerical kernels)
KERNEL_RULES: frozenset[str] = frozenset({"AJ001", "AJ002", "AJ003", "AJ006"})
ALL_RULES: frozenset[str] = frozenset(RULES)
_KERNEL_DIR = "processes"
_STATIC_ANNOTATIONS = {"bool", "int", "str", "float", "None", "Literal", "type"}
_STATIC_ATTRS = {"shape", "ndim", "dtype", "size"}
_STATIC_CALLS = {"ndim", "shape", "len", "isinstance", "hasattr", "callable", "type", "issubclass"}
_RANGE_CALLS = {"range", "arange"}
#: attributes that hold the size of a grid axis (AJ006)
_GRID_SIZE_ATTRS = {"n_node", "n_layer", "n_slice", "n_horizon", "n_lyr", "nlayr", "n_cell", "n_depth"}


@dataclass(frozen=True)
class Finding:
    rule: str
    level: str
    path: str
    line: int
    col: int
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.rule} [{self.level}] {self.message}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _call_name(node: ast.AST) -> str | None:
    """Terminal name of a call target: ``jnp.where`` -> ``where``, ``eqx.tree_at`` -> ``tree_at``."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _names_in(node: ast.AST) -> set[str]:
    """Root names referenced in an expression (``state.crop.lai`` -> ``state``)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _target_names(target: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(target):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _is_process_decorated(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for d in fn.decorator_list:
        if _call_name(d) == _PROCESS_DECORATOR:
            return True
    return False


def _is_static_test(test: ast.AST) -> bool:
    """Tests that are legitimately Python-level: ``x is None``, ``isinstance(...)``, ``TYPE_CHECKING``."""
    if isinstance(test, ast.Compare) and all(isinstance(op, (ast.Is, ast.IsNot)) for op in test.ops):
        return True
    if isinstance(test, ast.Call) and _call_name(test) in {"isinstance", "hasattr", "callable"}:
        return True
    if isinstance(test, ast.Name) and test.id in {"TYPE_CHECKING", "__debug__"}:
        return True
    return False


def _is_str_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _dynamic_names(node: ast.AST) -> set[str]:
    """Root names of ``node`` outside static sub-expressions (shape queries, string comparisons)."""
    if isinstance(node, ast.Attribute) and node.attr in _STATIC_ATTRS:
        return set()
    if isinstance(node, ast.Call) and _call_name(node) in _STATIC_CALLS:
        return set()
    if isinstance(node, ast.Compare) and any(_is_str_constant(c) for c in (node.left, *node.comparators)):
        return set()
    if isinstance(node, ast.Compare) and all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
        return set()
    if isinstance(node, ast.Name):
        return {node.id}
    out: set[str] = set()
    for child in ast.iter_child_nodes(node):
        out |= _dynamic_names(child)
    return out


def _annotation_is_static(ann: ast.AST | None) -> bool:
    """``bool``, ``int``, ``str``, ``float``, ``Literal[...]`` and unions of them (``int | None``)."""
    if ann is None:
        return False
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):  # string annotation
        try:
            ann = ast.parse(ann.value, mode="eval").body
        except SyntaxError:
            return False
    if isinstance(ann, ast.Constant) and ann.value is None:
        return True
    if isinstance(ann, ast.Name):
        return ann.id in _STATIC_ANNOTATIONS
    if isinstance(ann, ast.Attribute):
        return ann.attr in _STATIC_ANNOTATIONS
    if isinstance(ann, ast.Subscript):
        return _call_name(ann.value) == "Literal"
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        return _annotation_is_static(ann.left) and _annotation_is_static(ann.right)
    return False


def _is_positive_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value > 0
    )


def _is_nonzero_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return _is_positive_constant(node)


def _iter_own_nodes(fn: ast.AST) -> Iterator[ast.AST]:
    """Walk a function body without descending into nested function/class definitions."""
    stack: list[ast.AST] = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


# ---------------------------------------------------------------------------
# per-function checker
# ---------------------------------------------------------------------------


class _FunctionChecker:
    def __init__(
        self,
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        path: str,
        rules: frozenset[str] = ALL_RULES,
    ) -> None:
        self.fn = fn
        self.path = path
        self.rules = rules
        self.findings: list[Finding] = []
        args = fn.args
        all_args = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        self.arg_names: set[str] = {a.arg for a in all_args}
        if args.vararg:
            self.arg_names.add(args.vararg.arg)
        if args.kwarg:
            self.arg_names.add(args.kwarg.arg)
        # self / cls and static-annotated configuration arguments are not traced values
        static = {a.arg for a in all_args if a.arg in {"self", "cls"} or _annotation_is_static(a.annotation)}
        self.tainted: set[str] = set(self.arg_names) - static
        self.assigned: dict[str, ast.AST] = {}
        self._compute_taint()

    # ---- dataflow ------------------------------------------------------------
    def _compute_taint(self) -> None:
        """Forward taint: names assigned from tainted expressions are tainted. Iterated to a fixed point."""
        changed = True
        statements = list(self._statements(self.fn.body))
        while changed:
            changed = False
            for stmt in statements:
                targets: list[ast.AST] = []
                value: ast.AST | None = None
                if isinstance(stmt, ast.Assign):
                    targets, value = list(stmt.targets), stmt.value
                elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    targets, value = [stmt.target], stmt.value
                elif isinstance(stmt, ast.AugAssign):
                    targets, value = [stmt.target], stmt.value
                elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                    targets, value = [stmt.target], stmt.iter
                elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                    for item in stmt.items:
                        if item.optional_vars is not None:
                            t = _target_names(item.optional_vars)
                            if _names_in(item.context_expr) & self.tainted and not t <= self.tainted:
                                self.tainted |= t
                                changed = True
                    continue
                elif isinstance(stmt, ast.NamedExpr):
                    targets, value = [stmt.target], stmt.value
                if value is None:
                    continue
                tnames: set[str] = set()
                for t in targets:
                    tnames |= _target_names(t)
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.assigned[t.id] = value
                if _names_in(value) & self.tainted and not tnames <= self.tainted:
                    self.tainted |= tnames
                    changed = True

    def _statements(self, body: Iterable[ast.AST]) -> Iterator[ast.AST]:
        for node in body:
            for n in _iter_own_nodes(node):
                yield n
            yield node

    # ---- rules -----------------------------------------------------------------
    def _add(self, rule: str, node: ast.AST, detail: str = "") -> None:
        level, msg = RULES[rule]
        if detail:
            msg = f"{msg}: {detail}"
        self.findings.append(
            Finding(rule, level, self.path, getattr(node, "lineno", 0), getattr(node, "col_offset", 0), msg)
        )

    def run(self) -> list[Finding]:
        for rule, check in (
            ("AJ001", self.check_aj001),
            ("AJ002", self.check_aj002),
            ("AJ003", self.check_aj003),
            ("AJ004", self.check_aj004),
            ("AJ005", self.check_aj005),
            ("AJ006", self.check_aj006),
        ):
            if rule in self.rules:
                check()
        return self.findings

    def check_aj001(self) -> None:
        for node in _iter_own_nodes(self.fn):
            if isinstance(node, (ast.If, ast.While, ast.IfExp)):
                test = node.test
                if _is_static_test(test):
                    continue
                hit = sorted(_dynamic_names(test) & self.tainted)
                if hit:
                    kind = {ast.If: "if", ast.While: "while", ast.IfExp: "conditional expression"}[type(node)]
                    self._add("AJ001", node, f"{kind} on {', '.join(hit)}; use jnp.where / jnp.select")
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for gen in node.generators:
                    for cond in gen.ifs:
                        hit = sorted(_dynamic_names(cond) & self.tainted)
                        if hit:
                            self._add("AJ001", cond, f"comprehension filter on {', '.join(hit)}")

    def check_aj002(self) -> None:
        for node in _iter_own_nodes(self.fn):
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                for inner in ast.walk(node):
                    if inner is node:
                        continue
                    if isinstance(inner, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                        targets = inner.targets if isinstance(inner, ast.Assign) else [inner.target]
                        for t in targets:
                            if isinstance(t, ast.Subscript):
                                self._add(
                                    "AJ002",
                                    inner,
                                    f"{ast.unparse(t)} assigned in a loop; vectorise over the axis instead",
                                )
                            elif isinstance(t, (ast.Tuple, ast.List)) and any(
                                isinstance(e, ast.Subscript) for e in t.elts
                            ):
                                self._add("AJ002", inner, f"{ast.unparse(t)} assigned in a loop")
                    # jnp .at[i].set(...) inside a loop is the functional spelling of the same thing
                    if isinstance(inner, ast.Call) and _call_name(inner) in {"set", "add", "multiply"}:
                        f = inner.func
                        if (
                            isinstance(f, ast.Attribute)
                            and isinstance(f.value, ast.Subscript)
                            and isinstance(f.value.value, ast.Attribute)
                            and f.value.value.attr == "at"
                        ):
                            self._add("AJ002", inner, f"{ast.unparse(inner)} inside a loop")

    def _is_guarded(self, operand: ast.AST, depth: int = 0) -> bool:
        """True when ``operand`` is provably positive (log/sqrt argument, divisor).

        Positive constants, guard calls (``maximum``, ``clip``, ``exp``, ...), untainted names and
        attributes count as guarded. ``a * b``, ``a + b`` and ``a / b`` are guarded only when both
        sides are; ``-a`` and ``a - b`` of traced values never are (``log(x + -5.0)`` and
        ``sqrt(-2.0 * x)`` produce NaN).
        """
        if depth > 12:
            return False
        if isinstance(operand, ast.Constant):
            return _is_positive_constant(operand)
        if isinstance(operand, ast.Call):
            name = _call_name(operand)
            if name in _GUARD_CALLS:
                return True
            return False
        if isinstance(operand, ast.Name):
            src = self.assigned.get(operand.id)
            if src is None:
                return operand.id not in self.tainted
            return self._is_guarded(src, depth + 1)
        if isinstance(operand, ast.UnaryOp):
            if isinstance(operand.op, ast.UAdd):
                return self._is_guarded(operand.operand, depth + 1)
            return False
        if isinstance(operand, ast.BinOp):
            if isinstance(operand.op, (ast.Add, ast.Mult, ast.Div)):
                return self._is_guarded(operand.left, depth + 1) and self._is_guarded(
                    operand.right, depth + 1
                )
            if isinstance(operand.op, ast.Pow):
                return self._is_guarded(operand.left, depth + 1)
            # a - b and the rest: guarded only when neither side is traced
            return _names_in(operand).isdisjoint(self.tainted) and not any(
                isinstance(n, ast.Constant) and not _is_positive_constant(n) for n in ast.walk(operand)
            )
        if isinstance(operand, ast.Attribute):
            return _names_in(operand).isdisjoint(self.tainted)
        return False

    def _risky_in(self, expr: ast.AST) -> Iterator[tuple[ast.AST, str]]:
        for n in ast.walk(expr):
            if isinstance(n, ast.Call) and _call_name(n) in _RISKY_CALLS and n.args:
                if not self._is_guarded(n.args[0]):
                    yield n, f"{_call_name(n)}({ast.unparse(n.args[0])})"
            elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
                # any non-zero constant divisor is safe, whatever its sign
                if not (_is_nonzero_constant(n.right) or self._is_guarded(n.right)):
                    yield n, f"division by {ast.unparse(n.right)}"
            elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
                if isinstance(n.right, ast.Constant) and isinstance(n.right.value, (int, float)):
                    if n.right.value >= 1 and float(n.right.value).is_integer():
                        continue
                if not self._is_guarded(n.left):
                    yield n, f"fractional/negative power of {ast.unparse(n.left)}"

    def check_aj003(self) -> None:
        for node in _iter_own_nodes(self.fn):
            if not (isinstance(node, ast.Call) and _call_name(node) in _WHERE_CALLS):
                continue
            branches: list[ast.AST] = []
            name = _call_name(node)
            if name == "where":
                branches = list(node.args[1:3])
                for kw in node.keywords:
                    if kw.arg in {"x", "y"}:
                        branches.append(kw.value)
            elif name == "select":
                if len(node.args) >= 2 and isinstance(node.args[1], (ast.List, ast.Tuple)):
                    branches = list(node.args[1].elts)
                if len(node.args) >= 3:
                    branches.append(node.args[2])
            seen: set[int] = set()
            for br in branches:
                for risky, detail in self._risky_in(br):
                    if id(risky) in seen:
                        continue
                    seen.add(id(risky))
                    self._add("AJ003", risky, f"{detail} in a {name} branch; both branches must stay finite")

    def _is_update(self, value: ast.AST, depth: int = 0) -> bool:
        if depth > 6:
            return False
        if isinstance(value, ast.Call):
            name = _call_name(value)
            if name in _UPDATE_CALLS:
                return True
            return False
        if isinstance(value, ast.Name):
            # returning an argument unchanged is a legal no-op
            if value.id in self.arg_names:
                return True
            src = self.assigned.get(value.id)
            if src is None:
                return False
            return self._is_update(src, depth + 1)
        return False

    def check_aj004(self) -> None:
        for node in _iter_own_nodes(self.fn):
            if isinstance(node, ast.Return):
                if node.value is None:
                    self._add("AJ004", node, "process returns None")
                elif not self._is_update(node.value):
                    self._add("AJ004", node, f"returns {ast.unparse(node.value)[:60]}")

    def check_aj005(self) -> None:
        doc = ast.get_docstring(self.fn)
        if not doc:
            self._add("AJ005", self.fn, f"{self.fn.name} has no docstring")
            return
        if not any(line.strip().lower().startswith("source:") for line in doc.splitlines()):
            self._add("AJ005", self.fn, f"{self.fn.name} docstring has no 'Source:' line")

    def _shape_derived(self, expr: ast.AST, depth: int = 0) -> bool:
        """True when ``expr`` contains a shape query, a grid size or ``len`` of an argument (AJ006)."""
        if depth > 8:
            return False
        for n in ast.walk(expr):
            if isinstance(n, ast.Attribute) and n.attr in ({"shape", "size"} | _GRID_SIZE_ATTRS):
                return True
            if isinstance(n, ast.Call):
                name = _call_name(n)
                if name in {"shape", "size"}:
                    return True
                if name == "len" and n.args and _names_in(n.args[0]) & self.tainted:
                    return True
            if isinstance(n, ast.Name) and n.id in self.assigned:
                src = self.assigned[n.id]
                if src is not expr and self._shape_derived(src, depth + 1):
                    return True
        return False

    def _range_over_shape(self, it: ast.AST) -> bool:
        return (
            isinstance(it, ast.Call)
            and _call_name(it) in _RANGE_CALLS
            and any(self._shape_derived(a) for a in it.args)
        )

    def _array_valued(self, expr: ast.AST, depth: int = 0) -> bool:
        """True when ``expr`` is visibly an array of traced values (AJ006, direct iteration).

        An attribute of a traced argument (``state.theta``, ``grid.tl``), array arithmetic on
        traced names, a ``jnp``/``np`` call on traced names, or a name assigned from one of these.
        A bare argument is not enough (it may be a tuple of names or pytrees).
        """
        if depth > 8:
            return False
        if isinstance(expr, ast.Attribute):
            return expr.attr not in _STATIC_ATTRS and bool(_names_in(expr.value) & self.tainted)
        if isinstance(expr, (ast.BinOp, ast.UnaryOp)):
            return bool(_names_in(expr) & self.tainted)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            mod = expr.func.value
            if isinstance(mod, ast.Name) and mod.id in {"jnp", "np", "numpy", "lax"}:
                return bool(_names_in(expr) & self.tainted)
            return False
        if isinstance(expr, ast.Name) and expr.id in self.assigned and expr.id in self.tainted:
            src = self.assigned[expr.id]
            return src is not expr and self._array_valued(src, depth + 1)
        return False

    def _loops_over_array(self, it: ast.AST) -> bool:
        if isinstance(it, ast.Call) and _call_name(it) in {"enumerate", "zip", "reversed"}:
            return any(self._array_valued(a) or self._range_over_shape(a) for a in it.args)
        return self._array_valued(it)

    def check_aj006(self) -> None:
        for node in _iter_own_nodes(self.fn):
            iters: list[ast.AST] = []
            if isinstance(node, (ast.For, ast.AsyncFor)):
                iters.append(node.iter)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                iters.extend(g.iter for g in node.generators)
            for it in iters:
                if self._range_over_shape(it) or self._loops_over_array(it):
                    self._add(
                        "AJ006",
                        it,
                        f"loop over {ast.unparse(it)}; vectorise over the axis or use core.depth_scan",
                    )


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


_FnNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class CheckedFunction:
    """A function visited by the lint and the rules applied to it."""

    path: str
    name: str
    line: int
    is_process: bool
    rules: frozenset[str]


def _in_kernel_dir(path: str) -> bool:
    return _KERNEL_DIR in Path(path).parts[:-1]


def _select(tree: ast.AST, path: str, all_functions: bool) -> Iterator[tuple[_FnNode, bool, frozenset[str]]]:
    kernel_file = _in_kernel_dir(path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_proc = _is_process_decorated(node)
            if all_functions or is_proc:
                yield node, is_proc, ALL_RULES
            elif kernel_file:
                yield node, is_proc, KERNEL_RULES


def lint_source(source: str, path: str = "<string>", *, all_functions: bool = False) -> list[Finding]:
    """Lint Python source text.

    ``@process`` functions get every rule; other functions in a ``processes/`` directory get
    :data:`KERNEL_RULES`; ``all_functions=True`` applies every rule to every function.
    """
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:  # report as a finding instead of crashing pre-commit
        return [Finding("AJ000", "error", path, e.lineno or 0, e.offset or 0, f"syntax error: {e.msg}")]
    findings: list[Finding] = []
    for node, _, rules in _select(tree, path, all_functions):
        findings.extend(_FunctionChecker(node, path, rules).run())
    findings.sort(key=lambda f: (f.path, f.line, f.col, f.rule))
    return findings


def checked_functions(paths: Iterable[str | Path], *, all_functions: bool = False) -> list[CheckedFunction]:
    """Every function the lint visits under ``paths`` (to assert that a run is not vacuous)."""
    out: list[CheckedFunction] = []
    for f in _iter_py_files(paths):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except SyntaxError:
            continue
        for node, is_proc, rules in _select(tree, str(f), all_functions):
            out.append(CheckedFunction(str(f), node.name, node.lineno, is_proc, rules))
    return out


def lint_file(path: str | Path, *, all_functions: bool = False) -> list[Finding]:
    p = Path(path)
    return lint_source(p.read_text(encoding="utf-8"), str(p), all_functions=all_functions)


def _iter_py_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if "__pycache__" not in f.parts:
                    yield f
        elif p.suffix == ".py":
            yield p


def lint_paths(paths: Iterable[str | Path], *, all_functions: bool = False) -> list[Finding]:
    out: list[Finding] = []
    for f in _iter_py_files(paths):
        out.extend(lint_file(f, all_functions=all_functions))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrijax.core.lint", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="files or directories")
    ap.add_argument("--all", action="store_true", help="check every function, not only @process ones")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    ns = ap.parse_args(argv)
    findings = lint_paths(ns.paths, all_functions=ns.all)
    visited = checked_functions(ns.paths, all_functions=ns.all)
    if not ns.quiet:
        for f in findings:
            print(f.format())
    n_err = sum(1 for f in findings if f.level == "error")
    n_warn = sum(1 for f in findings if f.level == "warning")
    n_proc = sum(1 for c in visited if c.is_process)
    print(
        f"agrijax lint: {n_err} error(s), {n_warn} warning(s) in {len(visited)} function(s) "
        f"({n_proc} @process, {len(visited) - n_proc} kernel)"
    )
    if n_err or (ns.strict and n_warn):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
