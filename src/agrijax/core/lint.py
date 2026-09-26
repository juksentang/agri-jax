"""Standalone AST lint for the three process rules.

Scope
-----
* every function decorated with ``@process`` gets every rule;
* every other function defined in a file under a ``processes/`` directory (the numerical
  kernels the processes call: ``shuttleworth_wallace``, ``theta_of_h``, private helpers, ...)
  gets the numerical rules AJ001-AJ003, AJ006 and AJ007; AJ004/AJ005 are about the process contract
  and do not apply to kernels that return NamedTuples or arrays;
* ``--all`` applies every rule to every function in every file;
* AJ008 is a file-level rule: it checks the import statements of every file under a
  ``processes/`` directory, and of every file of ``agrijax/core/`` and ``agrijax/iface/``,
  whatever the functions in it.

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
AJ007  warning  A bare numeric literal in a ``@process`` function or a numerical kernel. Every
                model coefficient is declared once, with unit, meaning and provenance, by
                :func:`agrijax.core.coefficients.coef`; a unit conversion goes through a named
                adapter of :mod:`agrijax.core.units` (``mm_to_cm``, ``KG_HA_PER_G_M2``), and a
                numerical guard is a named module constant
                (:func:`agrijax.core.coefficients.numerical_guard`). The central whitelist
                (:data:`AJ007_TRIVIAL`, :data:`AJ007_MAX_INDEX`, :data:`AJ007_MAX_EXPONENT`,
                :data:`AJ007_STRUCTURAL_CALLS`, :data:`AJ007_STRUCTURAL_KEYWORDS`) allows ``0``,
                ``1`` and ``-1`` anywhere (``0.0``, ``1.0``, ``-1.0`` too); small integers used as
                indices, slice bounds, axes, shapes, counts (a subscript, ``axis=-1``,
                ``range(3)``, ``reshape(x, (-1, 2))``) or in a comparison with a shape query
                (``x.ndim == 2``); and small integer-valued exponents (``x**2``,
                ``jnp.power(x, 3)``). Anything else, including powers of ten, ``0.5`` and ``2.0``,
                is reported. Decorators and annotations are not checked; defaults of arguments
                are. AJ007 is a warning; ``--strict`` (what CI and pre-commit run) makes it fail
                like every other warning, and ``--strict-aj007`` fails on AJ007 alone.
AJ008  warning  An import of another slot's package from code under ``processes/<a>/``: modules of
                different slots talk only through the port records (M3 coupling contract; parallel
                development rule 2). A file of slot ``a`` (the directory right below the last
                ``processes/`` of its path, or the file itself when it sits directly in
                ``processes/``) may import ``agrijax.core``, the port records ``agrijax.iface``,
                its own slot ``processes/a`` and third-party packages; ``import
                <pkg>.processes.<b>``, ``from <pkg>.processes.<b> import ...``, ``from
                <pkg>.processes import <b>`` and relative imports that climb into ``processes/<b>``
                (``from ...pet import x``) are reported for every ``b != a``, also inside
                functions. The same rule keeps the port records below the slots: a file of
                ``agrijax/core/`` or ``agrijax/iface/`` that imports any ``agrijax.processes``
                module is reported, and a file of ``agrijax/iface/`` may import only
                ``agrijax.core``, ``agrijax.iface`` and third-party packages (so that the
                records' import closure is ``iface`` plus ``core``; :func:`import_closure` checks
                it transitively). ``--strict`` enforces it (the tree has no AJ008 finding).

Usage::

    python -m agrijax.core.lint src/agrijax/processes [--all] [--strict] [--strict-aj007]
                                                      [--ignore AJ007] [--aj007-report]
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

__all__ = [
    "AJ007_MAX_EXPONENT",
    "AJ007_MAX_INDEX",
    "AJ007_STRUCTURAL_CALLS",
    "AJ007_STRUCTURAL_KEYWORDS",
    "AJ007_TRIVIAL",
    "ALL_RULES",
    "KERNEL_RULES",
    "NOT_STRICT_RULES",
    "RULES",
    "CheckedFunction",
    "Finding",
    "checked_functions",
    "count_by_file",
    "import_closure",
    "import_slot",
    "lint_file",
    "lint_paths",
    "lint_source",
    "main",
    "module_imports",
    "slot_of_path",
]

RULES: dict[str, tuple[str, str]] = {
    "AJ001": ("error", "Python branch on a value derived from the process arguments"),
    "AJ002": ("error", "subscript assignment inside a for loop"),
    "AJ003": ("warning", "unguarded log/sqrt/division inside a where/select branch"),
    "AJ004": ("warning", "return value not built with eqx.tree_at / replace"),
    "AJ005": ("warning", "missing docstring or no 'Source:' line"),
    "AJ006": ("error", "Python loop over a shape-derived range or an array (unrolled layer loop)"),
    "AJ007": ("warning", "bare numeric literal in process or kernel code"),
    "AJ008": (
        "warning",
        "import of another slot's package from code under processes/ (or of processes from core/, iface/)",
    ),
}
#: warnings that ``--strict`` does not turn into failures (each has its own ``--strict-<rule>``);
#: empty since every module's coefficients are labelled and AJ007 is enforced
NOT_STRICT_RULES: frozenset[str] = frozenset()

# ---- AJ007 whitelist (the one place that says which bare numbers are allowed) ----------------
#: values allowed as a bare literal anywhere (int or float, either sign where listed)
AJ007_TRIVIAL: frozenset[float] = frozenset({0.0, 1.0, -1.0})
#: largest ``|n|`` of an integer literal used as an index, slice bound, axis, shape entry or count
AJ007_MAX_INDEX: int = 16
#: largest ``|n|`` of an integer-valued exponent (``x**2``, ``x**-2``, ``jnp.power(x, 3)``)
AJ007_MAX_EXPONENT: int = 4
#: calls whose integer arguments are structure (shapes, axes, counts), not model numbers
AJ007_STRUCTURAL_CALLS: frozenset[str] = frozenset(
    {
        "range", "arange", "reshape", "zeros", "ones", "empty", "full", "eye", "identity",
        "expand_dims", "squeeze", "moveaxis", "swapaxes", "transpose", "broadcast_to",
        "concatenate", "stack", "hstack", "vstack", "split", "take", "take_along_axis", "roll",
        "flip", "tile", "repeat", "pad", "linspace", "index_in_dim", "slice_in_dim",
        "dynamic_slice", "dynamic_slice_in_dim", "dynamic_update_slice", "tril", "triu", "diag",
        "diagonal", "enumerate", "ndim", "shape",
    }
)  # fmt: skip
#: keyword arguments whose integer values are structure
AJ007_STRUCTURAL_KEYWORDS: frozenset[str] = frozenset(
    {"axis", "axes", "ndim", "shape", "n", "k", "num", "offset", "keepdims", "unroll", "length",
     "size", "start", "stop", "step", "indices_or_sections", "static_argnums", "in_axes",
     "out_axes", "decimals", "ord"}
)  # fmt: skip

_RISKY_CALLS = {"log", "log2", "log10", "sqrt", "rsqrt", "power", "pow", "arccos", "arcsin", "arctanh"}
_GUARD_CALLS = {"maximum", "clip", "clamp", "minimum", "abs", "exp", "where", "select", "square", "softplus"}
_UPDATE_CALLS = {"tree_at", "replace", "set"}
_WHERE_CALLS = {"where", "select"}
_PROCESS_DECORATOR = "process"
#: rules applied to non-``@process`` functions of ``processes/`` modules (numerical kernels)
KERNEL_RULES: frozenset[str] = frozenset({"AJ001", "AJ002", "AJ003", "AJ006", "AJ007"})
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
            ("AJ007", self.check_aj007),
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

    # ---- AJ007 --------------------------------------------------------------------
    def _aj007_nodes(self) -> Iterator[tuple[ast.AST, ast.AST | None]]:
        """``(node, parent)`` of the function body and argument defaults, lambdas included,
        nested function / class definitions, decorators and annotations excluded."""
        args = self.fn.args
        roots: list[ast.AST] = [
            *self.fn.body,
            *args.defaults,
            *(d for d in args.kw_defaults if d is not None),
        ]
        stack: list[tuple[ast.AST, ast.AST | None]] = [(r, None) for r in roots]
        while stack:
            node, parent = stack.pop()
            yield node, parent
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for field, child in ast.iter_fields(node):
                if field in {"annotation", "returns"}:
                    continue
                children = child if isinstance(child, list) else [child]
                for c in children:
                    if isinstance(c, ast.AST):
                        stack.append((c, node))

    def check_aj007(self) -> None:
        parents: dict[int, ast.AST | None] = {}
        literals: list[tuple[ast.Constant, float, bool]] = []
        for n, parent in self._aj007_nodes():
            parents[id(n)] = parent
            v = n.value if isinstance(n, ast.Constant) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                assert isinstance(n, ast.Constant)
                literals.append((n, float(v), isinstance(v, int)))
        for lit, value, is_int in literals:
            node: ast.AST = lit
            parent = parents.get(id(lit))
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, (ast.USub, ast.UAdd)):
                node = parent
                value = -value if isinstance(parent.op, ast.USub) else value
            if _aj007_allowed(node, value, is_int, parents):
                continue
            text = ast.unparse(node)
            hint = "declare it with core.coefficients.coef (or as a named constant)"
            if _is_power_of_ten(value):
                hint += "; a unit conversion goes through a named adapter of core.units"
            self._add("AJ007", node, f"{text}; {hint}")


def _is_power_of_ten(value: float) -> bool:
    import math

    if value == 0.0 or not math.isfinite(value):
        return False
    e = math.log10(abs(value))
    return abs(e - round(e)) < 1e-9 and round(e) != 0


def _is_shape_query(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr in _STATIC_ATTRS:
            return True
        if isinstance(n, ast.Call) and _call_name(n) in {"len", "ndim", "shape"}:
            return True
    return False


def _aj007_allowed(node: ast.AST, value: float, is_int: bool, parents: dict[int, ast.AST | None]) -> bool:
    """The AJ007 whitelist (see the module docstring)."""
    if value in AJ007_TRIVIAL:
        return True
    integral = float(value).is_integer()
    parent = parents.get(id(node))
    # exponent: x ** 2, jnp.power(x, 3)
    if integral and abs(value) <= AJ007_MAX_EXPONENT:
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Pow) and parent.right is node:
            return True
        if (
            isinstance(parent, ast.Call)
            and _call_name(parent) in {"power", "pow"}
            and len(parent.args) > 1
            and parent.args[1] is node
        ):
            return True
    if not (is_int and abs(value) <= AJ007_MAX_INDEX):
        return False
    # climb through tuples, lists, slices and unary signs to the structural context; integer
    # arithmetic (``n - 2``) is climbed through only up to a subscript (``x[..., n - 2]``)
    child, cur = node, parent
    arithmetic = False
    while cur is not None:
        if isinstance(cur, ast.Subscript):
            return child is cur.slice
        if isinstance(cur, ast.keyword):
            return not arithmetic and cur.arg in AJ007_STRUCTURAL_KEYWORDS
        if isinstance(cur, ast.Call):
            return (
                not arithmetic
                and any(a is child for a in cur.args)
                and _call_name(cur) in AJ007_STRUCTURAL_CALLS
            )
        if isinstance(cur, ast.Compare):
            return not arithmetic and any(
                _is_shape_query(o) for o in (cur.left, *cur.comparators) if o is not child
            )
        if isinstance(cur, ast.BinOp):
            arithmetic = True
        elif not isinstance(cur, (ast.Tuple, ast.List, ast.Slice, ast.UnaryOp)):
            return False
        child, cur = cur, parents.get(id(cur))
    return False


# ---------------------------------------------------------------------------
# AJ008: slots talk only through the port records
# ---------------------------------------------------------------------------


def slot_of_path(path: str | Path) -> tuple[str, tuple[str, ...]] | None:
    """``(slot, package parts below processes/)`` of a file under a ``processes/`` directory, else
    ``None``. The slot is the directory right below the last ``processes/`` of the path
    (``processes/crop/ceres_maize/growth.py`` -> ``("crop", ("crop", "ceres_maize"))``), or the
    module itself for a file directly in ``processes/`` (``processes/bucket.py`` -> ``("bucket",
    ())``); ``processes/__init__.py`` has no slot."""
    parts = Path(path).parts
    dirs = parts[:-1]
    idx = [i for i, d in enumerate(dirs) if d == _KERNEL_DIR]
    if not idx:
        return None
    below = tuple(dirs[idx[-1] + 1 :])
    if below:
        return below[0], below
    stem = Path(parts[-1]).stem
    if stem == "__init__":
        return None
    return stem, ()


def _slot_in_dotted(parts: Sequence[str]) -> str | None:
    """The slot a dotted module path names: the part after its last ``processes`` component."""
    idx = [i for i, p in enumerate(parts) if p == _KERNEL_DIR]
    if not idx or idx[-1] + 1 >= len(parts):
        return None
    return parts[idx[-1] + 1]


def import_slot(node: ast.Import | ast.ImportFrom, package: Sequence[str]) -> list[tuple[str, str]]:
    """``[(slot, imported name)]`` of the ``processes/<slot>`` packages an import statement reaches,
    for a file whose package parts below ``processes/`` are ``package`` (see :func:`slot_of_path`).

    Absolute imports are matched on a ``processes`` component of the dotted name; a relative import
    is resolved against ``package`` first. ``from <pkg>.processes import <b>`` reaches ``b``; a bare
    ``import <pkg>.processes`` reaches no slot."""
    out: list[tuple[str, str]] = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            slot = _slot_in_dotted(alias.name.split("."))
            if slot is not None:
                out.append((slot, alias.name))
        return out
    module = node.module.split(".") if node.module else []
    names = [a.name for a in node.names]
    if node.level == 0:
        slot = _slot_in_dotted(module)
        if slot is not None:
            return [(slot, ".".join(module))]
        if module and module[-1] == _KERNEL_DIR:
            return [(n, ".".join([*module, n])) for n in names if n != "*"]
        return []
    up = node.level - 1  # ``from .x`` stays in the file's package
    text = "." * node.level + ".".join(module)
    if up < len(package):
        base = list(package[: len(package) - up])
        return [(base[0], text)]
    if up == len(package):  # resolved at the processes/ directory itself
        if module:
            return [(module[0], text)]
        return [(n, f"{text}{n}") for n in names if n != "*"]
    # climbed above processes/: it re-enters a slot only through a ``processes`` component
    slot = _slot_in_dotted(module)
    if slot is not None:
        return [(slot, text)]
    if module and module[-1] == _KERNEL_DIR:
        return [(n, f"{text}.{n}") for n in names if n != "*"]
    return []


#: packages of ``agrijax`` below the slots: they never import ``agrijax.processes``
_BELOW_SLOTS: tuple[str, ...] = ("core", "iface")
#: what a file of ``agrijax/iface/`` may import from ``agrijax`` (its closure stays iface + core)
_IFACE_MAY_IMPORT: tuple[str, ...] = ("agrijax.core", "agrijax.iface")


def _below_slots_package(path: str | Path) -> tuple[str, ...] | None:
    """The package parts (``("agrijax", "iface")``) of a file of ``agrijax/core/`` or
    ``agrijax/iface/`` (or a subpackage of them), else ``None``."""
    dirs = Path(path).parts[:-1]
    for i in range(len(dirs) - 1, 0, -1):
        if dirs[i - 1] == "agrijax" and dirs[i] in _BELOW_SLOTS:
            return tuple(dirs[i - 1 :])
    return None


def module_imports(node: ast.Import | ast.ImportFrom, package: Sequence[str]) -> list[str]:
    """The dotted modules an import statement names, resolved against the importing file's
    package ``package`` (``("agrijax", "iface")``): ``import a.b`` -> ``a.b``; ``from a import
    b`` -> ``a`` and ``a.b`` (``b`` may be a submodule); ``from .. import x`` relative to
    ``package``. A relative import that climbs above the top package names nothing."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    module = node.module.split(".") if node.module else []
    if node.level == 0:
        base = module
    else:
        up = node.level - 1
        if up >= len(package):
            return []
        base = [*package[: len(package) - up], *module]
    if not base:
        return []
    head = ".".join(base)
    return [head, *(f"{head}.{a.name}" for a in node.names if a.name != "*")]


def _is_processes_module(dotted: str) -> bool:
    parts = dotted.split(".")
    return len(parts) >= 2 and parts[0] == "agrijax" and parts[1] == _KERNEL_DIR


def _check_below_slots(tree: ast.AST, path: str, package: tuple[str, ...]) -> list[Finding]:
    level, msg = RULES["AJ008"]
    where = "/".join(package)
    iface = package[1] == "iface"
    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        names = module_imports(node, package)
        bad = [m for m in names if _is_processes_module(m)]
        if iface:
            bad += [
                m
                for m in names
                if m.split(".")[0] == "agrijax"
                and m != "agrijax"
                and not _is_processes_module(m)
                and not any(m == p or m.startswith(p + ".") for p in _IFACE_MAY_IMPORT)
            ]
        if bad:
            out.append(
                Finding(
                    "AJ008",
                    level,
                    path,
                    node.lineno,
                    node.col_offset,
                    f"{msg}: {where} imports {bad[0]}; "
                    + (
                        "the port records import only agrijax.core, agrijax.iface and third-party packages"
                        if iface
                        else "core imports no agrijax.processes module"
                    ),
                )
            )
    return out


def _module_file(dotted: str, src: Path) -> Path | None:
    base = src.joinpath(*dotted.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def import_closure(roots: Iterable[str], src: str | Path, *, depth: int | None = None) -> dict[str, str]:
    """The modules of the roots' top package that importing ``roots`` can load, transitively over
    every import statement of their source (also inside functions and ``if TYPE_CHECKING:``),
    with the parent packages each import runs: ``{module: the module that imports it}``. Modules
    are resolved as files under ``src`` (the directory that holds the top package); third-party
    and unresolvable names are skipped. ``depth=1`` gives the direct imports only."""
    src = Path(src)
    tops = {r.split(".")[0] for r in roots}
    seen: dict[str, str] = {}
    todo: list[tuple[str, str, int]] = [(r, "", 0) for r in roots]

    def with_parents(dotted: str) -> list[str]:
        parts = dotted.split(".")
        return [".".join(parts[: i + 1]) for i in range(len(parts))]

    while todo:
        mod, via, d = todo.pop()
        for m in with_parents(mod):
            if m in seen or m.split(".")[0] not in tops:
                continue
            file = _module_file(m, src)
            if file is None:
                continue
            seen[m] = via
            if depth is not None and d >= depth:
                continue
            package = tuple(m.split(".")) if file.name == "__init__.py" else tuple(m.split(".")[:-1])
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    todo.extend((n, m, d + 1) for n in module_imports(node, package))
    return seen


def _check_aj008(tree: ast.AST, path: str) -> list[Finding]:
    below = _below_slots_package(path)
    if below is not None:
        return _check_below_slots(tree, path, below)
    where = slot_of_path(path)
    if where is None:
        return []
    slot, package = where
    level, msg = RULES["AJ008"]
    out: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for other, name in import_slot(node, package):
            if other != slot:
                out.append(
                    Finding(
                        "AJ008",
                        level,
                        path,
                        node.lineno,
                        node.col_offset,
                        f"{msg}: processes/{slot} imports processes/{other} ({name}); import "
                        "agrijax.core and the port records agrijax.iface instead",
                    )
                )
    return out


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


def _select(
    tree: ast.AST,
    path: str,
    all_functions: bool,
    kernel: bool | None = None,
    processes: frozenset[str] = frozenset(),
) -> Iterator[tuple[_FnNode, bool, frozenset[str]]]:
    kernel_file = _in_kernel_dir(path) if kernel is None else kernel
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_proc = _is_process_decorated(node) or node.name in processes
            if all_functions or is_proc:
                yield node, is_proc, ALL_RULES
            elif kernel_file:
                yield node, is_proc, KERNEL_RULES


def lint_source(
    source: str,
    path: str = "<string>",
    *,
    all_functions: bool = False,
    ignore: Iterable[str] = (),
    kernel: bool | None = None,
    processes: Iterable[str] = (),
) -> list[Finding]:
    """Lint Python source text.

    ``@process`` functions get every rule; other functions in a ``processes/`` directory get
    :data:`KERNEL_RULES`; ``all_functions=True`` applies every rule to every function. Rules in
    ``ignore`` (e.g. ``{"AJ007"}``) are not run. ``kernel=True`` checks every non-process function
    as a kernel wherever the file is (the conformance kit uses it for a plugin whose kernels are
    not under a ``processes/`` directory); ``None`` decides by the path. ``processes`` names
    functions that are processes although not decorated (``p = process(fn, ...)``): they get every
    rule. AJ008 runs on the imports of every file under ``processes/``, ``agrijax/core/`` and
    ``agrijax/iface/``.
    """
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:  # report as a finding instead of crashing pre-commit
        return [Finding("AJ000", "error", path, e.lineno or 0, e.offset or 0, f"syntax error: {e.msg}")]
    skip = frozenset(ignore)
    findings: list[Finding] = []
    for node, _, rules in _select(tree, path, all_functions, kernel, frozenset(processes)):
        findings.extend(_FunctionChecker(node, path, rules - skip).run())
    if "AJ008" not in skip:
        findings.extend(_check_aj008(tree, path))
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


def lint_file(
    path: str | Path,
    *,
    all_functions: bool = False,
    ignore: Iterable[str] = (),
    kernel: bool | None = None,
    processes: Iterable[str] = (),
) -> list[Finding]:
    p = Path(path)
    return lint_source(
        p.read_text(encoding="utf-8"),
        str(p),
        all_functions=all_functions,
        ignore=ignore,
        kernel=kernel,
        processes=processes,
    )


def _iter_py_files(paths: Iterable[str | Path]) -> Iterator[Path]:
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if "__pycache__" not in f.parts:
                    yield f
        elif p.suffix == ".py":
            yield p


def lint_paths(
    paths: Iterable[str | Path], *, all_functions: bool = False, ignore: Iterable[str] = ()
) -> list[Finding]:
    skip = frozenset(ignore)
    out: list[Finding] = []
    for f in _iter_py_files(paths):
        out.extend(lint_file(f, all_functions=all_functions, ignore=skip))
    return out


def count_by_file(findings: Iterable[Finding], rule: str) -> dict[str, int]:
    """``{path: number of findings of rule}``, most findings first (the ``--aj007-report`` table)."""
    counts: dict[str, int] = {}
    for f in findings:
        if f.rule == rule:
            counts[f.path] = counts.get(f.path, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrijax.core.lint", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="files or directories")
    ap.add_argument("--all", action="store_true", help="check every function, not only @process ones")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="treat warnings as errors (AJ007 included)",
    )
    ap.add_argument(
        "--strict-aj007", action="store_true", help="treat AJ007 (bare numeric literals) as errors"
    )
    ap.add_argument(
        "--ignore", action="append", default=[], metavar="RULE", help="do not run RULE (repeatable)"
    )
    ap.add_argument("--aj007-report", action="store_true", help="print the AJ007 count per file")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    ns = ap.parse_args(argv)
    unknown = set(ns.ignore) - set(RULES)
    if unknown:
        ap.error(f"unknown rule(s) {sorted(unknown)}")
    findings = lint_paths(ns.paths, all_functions=ns.all, ignore=ns.ignore)
    visited = checked_functions(ns.paths, all_functions=ns.all)
    if not ns.quiet:
        for f in findings:
            print(f.format())
    n_err = sum(1 for f in findings if f.level == "error")
    n_warn = sum(1 for f in findings if f.level == "warning")
    n_aj007 = sum(1 for f in findings if f.rule == "AJ007")
    n_strict = sum(1 for f in findings if f.level == "warning" and f.rule not in NOT_STRICT_RULES)
    n_proc = sum(1 for c in visited if c.is_process)
    if ns.aj007_report:
        for path, n in count_by_file(findings, "AJ007").items():
            print(f"AJ007 {n:5d}  {path}")
    print(
        f"agrijax lint: {n_err} error(s), {n_warn} warning(s) ({n_aj007} AJ007) in {len(visited)} "
        f"function(s) ({n_proc} @process, {len(visited) - n_proc} kernel)"
    )
    if n_err or (ns.strict and n_strict) or (ns.strict_aj007 and n_aj007):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
