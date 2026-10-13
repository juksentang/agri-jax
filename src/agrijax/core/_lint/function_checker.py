"""The per-function checker of :mod:`agrijax.core.lint` (rules AJ001-AJ007, AJ020 and AJ021) and
the AJ007 whitelist test. :class:`_FunctionChecker` is one class of about 1040 lines: its
visitors share the state of one function walk, so it stays in one module."""

from __future__ import annotations

import ast
from typing import Callable, Collection, Iterable, Iterator, Sequence

from agrijax.core._lint.ast_facts import (
    _AJ020_STATIC_ATTRS,
    _annotation_is_static,
    _bind_module_names,
    _call_name,
    _callback_target,
    _chain_root,
    _dynamic_names,
    _Env,
    _FileFacts,
    _flat_targets,
    _has_at_link,
    _is_callback_call,
    _is_nonzero_constant,
    _is_positive_constant,
    _is_static_test,
    _is_str_constant,
    _iter_own_nodes,
    _lambda_params,
    _leaves,
    _names_in,
    _numpy_dotted,
    _pattern_names,
    _ref_in,
    _says_host_side,
    _static_fields,
    _target_names,
    _tracer_guard,
)
from agrijax.core._lint.rules import (
    _ALIAS_METHODS,
    _ARRAY_METHODS,
    _CONSTRUCTOR_METHODS,
    _DUNDER_MUTATORS,
    _GRID_SIZE_ATTRS,
    _GUARD_CALLS,
    _INPLACE_ANYWHERE,
    _INPLACE_IF_DISCARDED,
    _NP_ALIAS_FUNCS,
    _NP_FACTORIES,
    _NP_INPLACE_FUNCS,
    _NP_NON_FUNCTIONS,
    _OPERATOR_MUTATORS,
    _RANGE_CALLS,
    _RISKY_CALLS,
    _SHALLOW_CALLS,
    _SHALLOW_METHODS,
    _STATIC_ATTRS,
    _TYPE_RECEIVERS,
    _UPDATE_CALLS,
    _WHERE_CALLS,
    AJ007_MAX_EXPONENT,
    AJ007_MAX_INDEX,
    AJ007_STRUCTURAL_CALLS,
    AJ007_STRUCTURAL_KEYWORDS,
    AJ007_TRIVIAL,
    AJ020_HOST_MARKER,
    AJ020_STATIC_CALLS,
    ALL_RULES,
    RULES,
    Finding,
)

# ---------------------------------------------------------------------------
# per-function checker
# ---------------------------------------------------------------------------


class _FunctionChecker:
    def __init__(
        self,
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        path: str,
        rules: frozenset[str] = ALL_RULES,
        *,
        is_process: bool = True,
        host: bool = False,
        facts: _FileFacts | None = None,
    ) -> None:
        self.fn = fn
        self.path = path
        self.rules = rules
        self.findings: list[Finding] = []
        self._seen: set[tuple[str, int, int, str]] = set()
        #: a @process always runs traced; a kernel may be host-side (AJ020 does not apply to it)
        self.is_process = is_process
        self.host = host and not is_process
        self.facts = facts if facts is not None else _FileFacts.of(ast.Module(body=[], type_ignores=[]))
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
        # a method's ``self`` is a pytree like any argument (AJ020 / AJ021), its static fields aside
        cls = self.facts.class_of.get(id(fn))
        decorators = {_call_name(d) for d in fn.decorator_list}
        self.is_method = (
            cls is not None
            and id(cls) in self.facts.pytree_classes
            and bool(all_args)
            and all_args[0].arg == "self"
            and not decorators & {"staticmethod", "classmethod"}
        )
        self.self_static: frozenset[str] = _static_fields(cls) if self.is_method else frozenset()
        #: the objects the caller owns (AJ021); ``*args`` / ``**kwargs`` are fresh containers of them
        owned = {a.arg for a in all_args} - static
        if self.is_method and fn.name not in _CONSTRUCTOR_METHODS:
            owned.add("self")
        self.owned: frozenset[str] = frozenset(owned)
        self.fresh: frozenset[str] = frozenset(a.arg for a in (args.vararg, args.kwarg) if a is not None)
        self.numpy: dict[str, str] = self._local_module_names(self.facts.numpy, "numpy")
        self.operator: dict[str, str] = self._local_module_names(self.facts.operator, "operator")
        #: host-side functions of this function: nested defs marked host-side, and local names
        #: bound to a host function (``g = h``)
        self.local_host: set[str] = {
            n.name
            for n in ast.iter_child_nodes(fn)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _says_host_side(n)
        }
        for n in sorted(
            (n for n in _iter_own_nodes(fn) if isinstance(n, ast.Assign)),
            key=lambda n: (n.lineno, n.col_offset),
        ):
            if isinstance(n.value, ast.Name) and (
                n.value.id in self.facts.host_functions or n.value.id in self.local_host
            ):
                self.local_host.update(t.id for t in n.targets if isinstance(t, ast.Name))
        self.parents: dict[int, ast.AST] = {id(c): n for n in ast.walk(fn) for c in ast.iter_child_nodes(n)}
        self.assigned: dict[str, ast.AST] = {}
        #: AJ020: the taint without static sub-expressions (``n = x.shape[-1]`` is a Python int),
        #: through comprehension targets, without the names a ``Tracer`` guard makes concrete
        traced = set(self.tainted) - self._tracer_guarded()
        if self.is_method:
            traced.add("self")
        self.traced: set[str] = self._compute_taint(traced, self._tnames, comprehensions=True)
        self.tainted = self._compute_taint(self.tainted, _names_in)

    def _tracer_guarded(self) -> set[str]:
        """Arguments made concrete by a top-level ``if isinstance(x, Tracer): return`` before any
        other use of them (:func:`_tracer_guard`)."""
        out: set[str] = set()
        used: set[str] = set()
        for stmt in self.fn.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            guarded = _tracer_guard(stmt)
            if guarded:
                out |= guarded - used
                continue
            used |= _names_in(stmt)
        return out

    # ---- dataflow ------------------------------------------------------------
    def _compute_taint(
        self, tainted: set[str], names_of: Callable[[ast.AST], set[str]], *, comprehensions: bool = False
    ) -> set[str]:
        """Forward taint: names assigned from tainted expressions are tainted. Iterated to a fixed
        point. ``names_of`` gives the names an expression depends on (:func:`_names_in` for
        AJ001-AJ006, :meth:`_tnames` for AJ020, which also taints comprehension targets)."""
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
                            if names_of(item.context_expr) & tainted and not t <= tainted:
                                tainted |= t
                                changed = True
                    continue
                elif isinstance(stmt, ast.NamedExpr):
                    targets, value = [stmt.target], stmt.value
                elif comprehensions and isinstance(stmt, ast.comprehension):
                    targets, value = [stmt.target], stmt.iter
                if value is None:
                    continue
                tnames: set[str] = set()
                for t in targets:
                    tnames |= _target_names(t)
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.assigned[t.id] = value
                if names_of(value) & tainted and not tnames <= tainted:
                    tainted |= tnames
                    changed = True
        return tainted

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
        line, col = getattr(node, "lineno", 0), getattr(node, "col_offset", 0)
        if (rule, line, col, msg) in self._seen:  # a loop body or a lambda is walked more than once
            return
        self._seen.add((rule, line, col, msg))
        self.findings.append(Finding(rule, level, self.path, line, col, msg))

    def run(self) -> list[Finding]:
        for rule, check in (
            ("AJ001", self.check_aj001),
            ("AJ002", self.check_aj002),
            ("AJ003", self.check_aj003),
            ("AJ004", self.check_aj004),
            ("AJ005", self.check_aj005),
            ("AJ006", self.check_aj006),
            ("AJ007", self.check_aj007),
            ("AJ020", self.check_aj020),
            ("AJ021", self.check_aj021),
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

    # ---- AJ020: NumPy on traced values -----------------------------------------------------
    def _tnames(self, node: ast.AST) -> set[str]:
        """Names whose values reach ``node`` as arrays: :func:`_dynamic_names` with the grid sizes
        and the static fields of ``self`` static too (AJ020)."""
        return _dynamic_names(node, _AJ020_STATIC_ATTRS, self.self_static)

    def _local_module_names(self, base: dict[str, str], module: str) -> dict[str, str]:
        """``base`` (the module-level names) updated by the function's own imports and
        assignments, in source order; a parameter of the same name hides a module name."""
        out = dict(base)
        a = self.fn.args
        for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg):
            if arg is not None:
                out.pop(arg.arg, None)
        stmts = [
            n
            for n in _iter_own_nodes(self.fn)
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
        ]
        for s in sorted(stmts, key=lambda n: (n.lineno, n.col_offset)):
            _bind_module_names(s, out, module)
        return out

    def _is_callback_lambda(self, lam: ast.Lambda) -> bool:
        parent = self.parents.get(id(lam))
        if isinstance(parent, ast.keyword):  # callback=lambda ...: the keyword sits between
            parent = self.parents.get(id(parent))
        if (
            isinstance(parent, ast.Call)
            and _call_name(parent) == "partial"
            and parent.args
            and parent.args[0] is lam
        ):
            lam_or_partial: ast.AST = parent
            parent = self.parents.get(id(parent))
        else:
            lam_or_partial = lam
        if not (isinstance(parent, ast.Call) and _is_callback_call(parent, self.facts.callback_aliases)):
            return False
        first = bool(parent.args) and parent.args[0] is lam_or_partial
        return first or any(k.arg == "callback" and k.value is lam_or_partial for k in parent.keywords)

    def _walk20(self) -> Iterator[tuple[ast.AST, frozenset[str]]]:
        """The function's nodes with the parameters of the lambdas around each one (traced:
        a lambda gets traced values from ``tree_map``, ``vmap``, ``scan``); nested definitions
        and host callbacks are skipped."""
        stack: list[tuple[ast.AST, frozenset[str]]] = [
            (c, frozenset()) for c in ast.iter_child_nodes(self.fn)
        ]
        while stack:
            n, extra = stack.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(n, ast.Lambda):
                if self._is_callback_lambda(n):
                    continue
                extra = extra | frozenset(_lambda_params(n))
            yield n, extra
            stack.extend((c, extra) for c in ast.iter_child_nodes(n))

    def _traced_args(self, call: ast.Call, traced: Collection[str], skip: ast.AST | None = None) -> list[str]:
        names: set[str] = set()
        for a in (*call.args, *(k.value for k in call.keywords)):
            if a is not skip:
                names |= self._tnames(a)
        return sorted(names & set(traced))

    def _numpy_callee(self, func: ast.AST) -> str | None:
        """The NumPy function a call target is: a NumPy reference or alias (:func:`_ref_in`), or
        the callable a NumPy factory returns (``np.vectorize(f)``)."""
        ref = _numpy_dotted(func, self.numpy)
        if ref is not None:
            return ref
        if isinstance(func, ast.Call):
            inner = _numpy_dotted(func.func, self.numpy)
            if inner in _NP_FACTORIES:
                return inner
        return None

    def _host_callee(self, func: ast.AST) -> str | None:
        """The host-side function or method a call target names, if it is one (same file, or
        imported with ``from <module> import <name>``)."""
        if isinstance(func, ast.Name) and (
            func.id in self.facts.host_functions or func.id in self.local_host
        ):
            return func.id
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = func.value.id
            if owner in {"self", "cls"}:
                cls = self.facts.class_of.get(id(self.fn))
                if cls is not None and func.attr in self.facts.host_methods.get(cls.name, frozenset()):
                    return f"{owner}.{func.attr}"
            elif func.attr in self.facts.host_methods.get(owner, frozenset()):
                return f"{owner}.{func.attr}"
        return None

    def _writes_into_argument(self, call: ast.Call, dotted: str) -> bool:
        """An in-place NumPy writer whose target is an argument: AJ021 reports it."""
        method = dotted.rsplit(".", 1)[-1]
        unbound = dotted.startswith("ndarray.") and (
            method in _INPLACE_ANYWHERE or method in _INPLACE_IF_DISCARDED
        )
        writer = dotted in _NP_INPLACE_FUNCS or dotted.endswith(".at") or unbound
        return (
            writer and "AJ021" in self.rules and bool(call.args) and _chain_root(call.args[0]) in self.owned
        )

    def check_aj020(self) -> None:
        if self.host:
            return
        why = (
            "a @process always runs traced"
            if self.is_process
            else f"a kernel that runs only on the host starts its docstring with {AJ020_HOST_MARKER!r}"
        )
        hits: list[tuple[ast.Call, str]] = []
        for node, extra in self._walk20():
            if not isinstance(node, ast.Call):
                continue
            traced = self.traced | extra
            dotted = self._numpy_callee(node.func)
            if dotted is not None:
                if dotted.rsplit(".", 1)[-1] in AJ020_STATIC_CALLS:
                    continue
                text = ast.unparse(node.func)
                if dotted.startswith("random."):
                    msg = f"{text}(...) in traced code draws once at trace time (a constant under jit)"
                    hits.append((node, f"{msg}; use jax.random with an explicit key"))
                    continue
                names = self._traced_args(node, traced)
                if names and not self._writes_into_argument(node, dotted):
                    if dotted in _NP_INPLACE_FUNCS or dotted.endswith(".at"):
                        fix = "x.at[i].set(v) / .add(v) or jnp.where(mask, v, x)"
                    elif dotted in _NP_FACTORIES:
                        fix = "jax.vmap over a jnp function"
                    else:
                        fix = f"jnp.{dotted}" if "?" not in dotted else "the jnp function"
                    msg = f"{text}(...) on {', '.join(names)}; use {fix}"
                    hits.append((node, f"{msg} (NumPy cannot take a tracer and fails under jit; {why})"))
                continue
            host = self._host_callee(node.func)
            if host is not None:
                names = self._traced_args(node, traced)
                if not names and host.startswith("self.") and "self" in traced:
                    names = ["self"]  # self.build(): the receiver is the traced argument
                if names:
                    msg = f"{host}(...) is host-side (NumPy on its arguments) and gets the traced values"
                    fix = "call it on concrete values before the trace, or write it with jnp"
                    hits.append((node, f"{msg} {', '.join(names)}; {fix}"))
                    continue
            self._aj020_passed(node, traced, hits)
        # one finding per outermost call: np.cumsum(np.asarray(x)) is one mistake
        inner = {id(n) for call, _ in hits for n in ast.walk(call) if n is not call}
        seen: set[int] = set()
        for call, msg in hits:
            if id(call) in inner or id(call) in seen:
                continue
            seen.add(id(call))
            self._add("AJ020", call, msg)

    def _aj020_passed(
        self, call: ast.Call, traced: Collection[str], hits: list[tuple[ast.Call, str]]
    ) -> None:
        """A NumPy function passed as a value (``tree_map(np.asarray, state)``, ``map(np.exp, xs)``,
        ``jax.vmap(np.sum)(x)``): reported when the call, or the call of its result, gets traced
        values. dtypes, types and constants (``dtype=np.float32``) are not functions."""
        callback = _is_callback_call(call, self.facts.callback_aliases)
        target = _callback_target(call) if callback else None
        for a in (*call.args, *(k.value for k in call.keywords if k.arg != "dtype")):
            v = a.value if isinstance(a, ast.Starred) else a
            if v is target:
                continue
            ref = _numpy_dotted(v, self.numpy)
            host = self._host_callee(v) if not ref else None
            if host is None and (
                not ref or ref.rsplit(".", 1)[-1] in (_NP_NON_FUNCTIONS | AJ020_STATIC_CALLS)
            ):
                continue
            names = self._traced_args(call, traced, skip=a)
            outer = self.parents.get(id(call))
            if not names and isinstance(outer, ast.Call) and outer.func is call:
                names = self._traced_args(outer, traced)
            if names:
                where = f"{ast.unparse(v)} passed to {ast.unparse(call.func)}(...)"
                msg = f"{where} runs NumPy on {', '.join(names)}"
                if host is not None:
                    fix = f"{host} is host-side: call it on concrete values before the trace, or use jnp"
                else:
                    fix = f"use jnp.{ref} (NumPy cannot take a tracer)"
                hits.append((call, f"{msg}; {fix}"))
                return

    # ---- AJ021: in-place mutation of an argument ------------------------------------------
    def _alias_of(self, v: ast.AST, env: _Env) -> str | None:
        """The argument whose object ``v`` is, or a part or view of (no copy); ``None`` for a new
        object. An element of a shallow container (``dict(state)["crop"]``) is a part too."""
        if isinstance(v, ast.Name):
            return env.full.get(v.id)
        if isinstance(v, ast.Attribute):
            return None if v.attr in _STATIC_ATTRS else self._alias_of(v.value, env)
        if isinstance(v, ast.Subscript):
            base = self._alias_of(v.value, env)
            if base is None and not isinstance(v.slice, ast.Slice):
                base = self._shallow_of(v.value, env)
            return base
        if isinstance(v, (ast.Starred, ast.NamedExpr)):
            return self._alias_of(v.value, env)
        if isinstance(v, ast.IfExp):
            return self._alias_of(v.body, env) or self._alias_of(v.orelse, env)
        if isinstance(v, ast.BoolOp):
            return next((o for o in (self._alias_of(x, env) for x in v.values) if o), None)
        if isinstance(v, ast.Call):
            f, args = v.func, v.args
            if isinstance(f, ast.Name):
                if f.id in {"getattr", "vars"} and args:
                    return self._alias_of(args[0], env)
                if f.id == "next" and args:
                    return self._elements_of(args[0], env)
                if f.id == "super":
                    if len(args) >= 2:
                        return self._alias_of(args[1], env)
                    return env.full.get("self") if self.is_method else None
            dotted = _numpy_dotted(f, self.numpy)
            if dotted is not None:
                no_copy = dotted == "array" and any(
                    k.arg == "copy" and isinstance(k.value, ast.Constant) and k.value.value is False
                    for k in v.keywords
                )
                if (dotted in _NP_ALIAS_FUNCS or no_copy) and args:
                    return self._alias_of(args[0], env)
                return None
            if isinstance(f, ast.Attribute):
                if f.attr == "get":
                    return self._elements_of(f.value, env)
                if f.attr in _ALIAS_METHODS:
                    return self._alias_of(f.value, env)
        return None

    def _shallow_of(self, v: ast.AST, env: _Env) -> str | None:
        """The argument whose elements a new container ``v`` holds: ``dict(state)``, ``{**state}``,
        ``copy.copy(x)``, ``x.copy()``, ``list(x)``, ``[state.x, 1.0]``, ``[*x]``, ``*args``."""
        if isinstance(v, ast.Name):
            return env.shallow.get(v.id)
        if isinstance(v, (ast.List, ast.Tuple, ast.Set)):
            for e in v.elts:
                o = self._elements_of(e.value, env) if isinstance(e, ast.Starred) else self._alias_of(e, env)
                if o:
                    return o
            return None
        if isinstance(v, ast.Dict):
            for k, val in zip(v.keys, v.values):
                o = self._elements_of(val, env) if k is None else self._alias_of(val, env)
                if o:
                    return o
            return None
        if isinstance(v, ast.Subscript) and isinstance(v.slice, ast.Slice):
            return self._shallow_of(v.value, env)
        if isinstance(v, ast.IfExp):
            return self._shallow_of(v.body, env) or self._shallow_of(v.orelse, env)
        if isinstance(v, ast.Call) and _numpy_dotted(v.func, self.numpy) is None:
            f = v.func
            shallow = (isinstance(f, ast.Name) and f.id in _SHALLOW_CALLS) or (
                isinstance(f, ast.Attribute)
                and f.attr == "copy"
                and isinstance(f.value, ast.Name)
                and f.value.id == "copy"
            )
            if shallow:
                return next((o for o in (self._elements_of(a, env) for a in v.args) if o), None)
            if isinstance(f, ast.Attribute) and f.attr in _SHALLOW_METHODS:
                return self._elements_of(f.value, env)
        return None

    def _elements_of(self, v: ast.AST, env: _Env) -> str | None:
        """The argument that the elements of ``v`` belong to (iterating it, unpacking it)."""
        return self._alias_of(v, env) or self._shallow_of(v, env)

    def _bind_origin(self, t: ast.AST, full: str | None, shallow: str | None, env: _Env) -> None:
        if isinstance(t, ast.Starred):
            self._bind_origin(t.value, None, full or shallow, env)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._bind_origin(e, full or shallow, None, env)
        elif isinstance(t, ast.Name):
            env.forget(t.id)
            if full:
                env.full[t.id] = full
            elif shallow:
                env.shallow[t.id] = shallow

    def _bind(self, targets: Sequence[ast.AST], value: ast.AST, env: _Env) -> None:
        """``targets = value``: a name becomes an alias, a shallow container, a bound lambda, or
        a new object (a copy, a computed array, ``state.replace(...)``) that is no alias any more."""
        for t in targets:
            if isinstance(t, (ast.Tuple, ast.List)):
                stars = any(isinstance(e, ast.Starred) for e in t.elts)
                if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(t.elts) and not stars:
                    for e, v in zip(t.elts, value.elts):
                        self._bind([e], v, env)
                else:
                    origin = self._elements_of(value, env)
                    for e in t.elts:
                        self._bind_origin(e, None if isinstance(e, ast.Starred) else origin, origin, env)
                continue
            if isinstance(t, ast.Starred):
                t = t.value
            if not isinstance(t, ast.Name):
                continue
            if isinstance(value, ast.Lambda):
                env.forget(t.id)
                env.lambdas[t.id] = value
                continue
            self._bind_origin(t, self._alias_of(value, env), self._shallow_of(value, env), env)

    def _bind_elements(self, target: ast.AST, iterable: ast.AST, env: _Env) -> None:
        origin = self._elements_of(iterable, env)
        self._bind_origin(target, origin, None, env)

    def _aj021(self, node: ast.AST, what: str, via: ast.AST, origin: str, hint: str) -> None:
        root = _chain_root(via)
        through = "" if root in {None, origin} else f" (through {root})"
        self._add("AJ021", node, f"{what} mutates the argument {origin} in place{through}; {hint}")

    def _check_targets(self, stmt: ast.AST, targets: Sequence[ast.AST], env: _Env, verb: str) -> None:
        for t in targets:
            for tgt in _flat_targets(t):
                if not isinstance(tgt, (ast.Attribute, ast.Subscript)):
                    continue
                origin = self._alias_of(tgt.value, env)
                if origin is None:
                    continue
                text = ast.unparse(tgt)
                if isinstance(tgt, ast.Subscript) and not _is_str_constant(tgt.slice):
                    hint = (
                        f"write {ast.unparse(tgt.value)}.at[{ast.unparse(tgt.slice)}].set(...) and "
                        "return the new array in the new state"
                    )
                else:
                    root = _chain_root(tgt) or origin
                    path = "s" + text[len(root) :] if text.startswith(root) else "s.<path>"
                    hint = (
                        f"return eqx.tree_at(lambda s: {path}, {root}, value) / {root}.replace(...) instead"
                    )
                self._aj021(stmt, f"{verb} {text}", tgt, origin, hint)

    def _is_type_receiver(self, recv: ast.AST) -> bool:
        return (
            (isinstance(recv, ast.Name) and recv.id in _TYPE_RECEIVERS)
            or _numpy_dotted(recv, self.numpy) == "ndarray"
            or (isinstance(recv, ast.Call) and _call_name(recv) == "type")
        )

    def _mutation_kind(self, method: str, discarded: bool) -> str | None:
        record = "return eqx.tree_at(...) / .replace(...) instead"
        if method in {"__setattr__", "__delattr__", "__setitem__", "__delitem__"}:
            return record
        if method in _DUNDER_MUTATORS:
            return "compute x = x + y (a new value) and return it in the new state"
        if method in _INPLACE_ANYWHERE or (method in _INPLACE_IF_DISCARDED and discarded):
            if method in _ARRAY_METHODS:
                return "use the functional form (jnp.sort(x), jnp.full_like(x, v), x.at[i].set(v))"
            return "build a new container ({**d, k: v}, [*xs, x]) and return it in the new state"
        return None

    def _mutating_call(self, call: ast.Call, env: _Env, discarded: set[int]) -> None:
        """``setattr`` / ``object.__setattr__`` / ``dict.update(x, ...)``, in-place methods, the
        in-place functions of NumPy and :mod:`operator`, and ``out=`` into an argument."""
        f, args = call.func, call.args
        first = args[0] if args else None
        hit: ast.AST | None = None
        hint = ""
        if isinstance(f, ast.Name) and f.id in {"setattr", "delattr"}:
            hit, hint = first, "return eqx.tree_at(...) / .replace(...) instead"
        elif isinstance(f, ast.Attribute):
            kind = self._mutation_kind(f.attr, id(call) in discarded)
            recv = f.value
            if kind is not None:
                if self._is_type_receiver(recv):
                    hit, hint = first, kind
                elif (
                    isinstance(recv, ast.Call)
                    and _call_name(recv) == "super"
                    and isinstance(recv.func, ast.Name)
                ):
                    hit = recv.args[1] if len(recv.args) >= 2 else ast.Name(id="self", ctx=ast.Load())
                    hint = kind
                elif not _has_at_link(recv):
                    hit, hint = recv, kind
        op = _ref_in(f, self.operator, "operator")
        if op and op.rsplit(".", 1)[-1] in _OPERATOR_MUTATORS:
            hit, hint = first, "compute the new value (x + y, {**d, k: v}) and return it in the new state"
        dotted = _numpy_dotted(f, self.numpy)
        if dotted is not None and (dotted in _NP_INPLACE_FUNCS or dotted.endswith(".at")):
            hit, hint = first, "use x.at[i].set(v) / jnp.where(mask, v, x) and return it in the new state"
        origin = self._alias_of(hit, env) if hit is not None else None
        if hit is not None and origin is not None:
            self._aj021(call, ast.unparse(call)[:70], hit, origin, hint)
        for kw in call.keywords:
            origin = self._alias_of(kw.value, env) if kw.arg == "out" else None
            if origin is not None:
                self._aj021(
                    call,
                    f"out={ast.unparse(kw.value)}",
                    kw.value,
                    origin,
                    "use the returned value and put it in the new state",
                )

    def _arg_kinds(self, args: Sequence[ast.AST], env: _Env) -> list[tuple[str | None, str | None]]:
        return [(self._alias_of(a, env), self._shallow_of(a, env)) for a in args]

    def _lambda(
        self, lam: ast.Lambda, env: _Env, kinds: Sequence[tuple[str | None, str | None]] | None
    ) -> None:
        """Check a lambda body: its parameters are bound to ``kinds`` (the arguments it is called
        with, or the leaves of the tree a ``tree_map`` walks), the enclosing aliases are closures."""
        e = env.copy()
        params = _lambda_params(lam)
        for p in params:
            e.forget(p)
        for p, (full, shallow) in zip(params, kinds or ()):
            self._bind_origin(ast.Name(id=p, ctx=ast.Store()), full, shallow, e)
        for d in (*lam.args.defaults, *(x for x in lam.args.kw_defaults if x is not None)):
            self._scan(d, env, set())
        self._scan(lam.body, e, {id(lam.body)})

    def _scan(self, node: ast.AST, env: _Env, discarded: set[int]) -> None:
        """Walk an expression for AJ021; ``discarded`` holds the calls whose result is not used."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.Lambda):
            self._lambda(node, env, None)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            e = env.copy()
            for gen in node.generators:
                self._scan(gen.iter, e, discarded)
                self._bind_elements(gen.target, gen.iter, e)
                for c in gen.ifs:
                    self._scan(c, e, discarded)
            if isinstance(node, ast.DictComp):
                self._scan(node.key, e, set())
                self._scan(node.value, e, set())
            else:
                self._scan(node.elt, e, {id(node.elt)} if id(node) in discarded else set())
            return
        if isinstance(node, ast.NamedExpr):
            self._scan(node.value, env, discarded)
            self._bind([node.target], node.value, env)
            return
        if not isinstance(node, ast.Call):
            for child in ast.iter_child_nodes(node):
                self._scan(child, env, discarded)
            return
        self._mutating_call(node, env, discarded)
        values = [*node.args, *(k.value for k in node.keywords)]
        lambdas = [a for a in values if isinstance(a, ast.Lambda)]
        if lambdas:
            # tree_map(lambda a: ..., state): the lambda gets the leaves of the other arguments
            others = [a for a in values if not isinstance(a, ast.Lambda)]
            origin = next((o for o in (self._elements_of(a, env) for a in others) if o), None)
            if _is_callback_call(node, self.facts.callback_aliases):
                origin = None
            for lam in lambdas:
                self._lambda(lam, env, [(origin, None)] * len(_lambda_params(lam)))
        if isinstance(node.func, ast.Lambda):
            self._lambda(node.func, env, self._arg_kinds(node.args, env))
        else:
            if isinstance(node.func, ast.Name) and node.func.id in env.lambdas:
                self._lambda(env.lambdas[node.func.id], env, self._arg_kinds(node.args, env))
            self._scan(node.func, env, discarded)
        for a in values:
            if not isinstance(a, ast.Lambda):
                self._scan(a, env, discarded)

    def _loop(
        self, body: Sequence[ast.stmt], env: _Env, target: ast.AST | None, iterable: ast.AST | None
    ) -> None:
        """A loop body, walked twice: an alias made in one iteration is seen by the next one; after
        the loop a name is an alias when it is one before it or after an iteration."""
        first = env.copy()
        if target is not None and iterable is not None:
            self._bind_elements(target, iterable, first)
        self._aj021_block(body, first)
        again = env.copy()
        again.update_from(env, first)
        after_first = again.copy()
        if target is not None and iterable is not None:
            self._bind_elements(target, iterable, again)
        self._aj021_block(body, again)
        env.update_from(after_first, again)

    def _aj021_block(self, body: Sequence[ast.stmt], env: _Env) -> None:
        """Walk ``body`` in order: a rebinding ends an alias for the rest of its block, and after a
        branch a name is an alias when it is one on any path."""
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                env.forget(stmt.name)
            elif isinstance(stmt, ast.If):
                self._scan(stmt.test, env, set())
                a, b = env.copy(), env.copy()
                self._aj021_block(stmt.body, a)
                self._aj021_block(stmt.orelse, b)
                live = [e for e, part in ((a, stmt.body), (b, stmt.orelse)) if not _leaves(part)]
                env.update_from(*(live or [a, b]))
            elif isinstance(stmt, ast.While):
                self._scan(stmt.test, env, set())
                self._loop(stmt.body, env, None, None)
                self._aj021_block(stmt.orelse, env)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                self._scan(stmt.iter, env, set())
                self._loop(stmt.body, env, stmt.target, stmt.iter)
                self._aj021_block(stmt.orelse, env)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    self._scan(item.context_expr, env, set())
                    if item.optional_vars is not None:
                        for n in _target_names(item.optional_vars):
                            env.forget(n)
                self._aj021_block(stmt.body, env)
            elif isinstance(stmt, (ast.Try, ast.TryStar)):
                done = env.copy()
                self._aj021_block(stmt.body, done)
                ends = []
                for h in stmt.handlers:
                    e = env.copy()
                    e.update_from(env, done)  # the exception can come from anywhere in the body
                    if h.name:
                        e.forget(h.name)
                    self._aj021_block(h.body, e)
                    ends.append(e)
                self._aj021_block(stmt.orelse, done)
                env.update_from(done, *ends)
                self._aj021_block(stmt.finalbody, env)
            elif isinstance(stmt, ast.Match):
                self._scan(stmt.subject, env, set())
                origin = self._elements_of(stmt.subject, env)
                ends = [env.copy()]
                for case in stmt.cases:
                    e = env.copy()
                    for n in _pattern_names(case.pattern):
                        self._bind_origin(ast.Name(id=n, ctx=ast.Store()), origin, None, e)
                    if case.guard is not None:
                        self._scan(case.guard, e, set())
                    self._aj021_block(case.body, e)
                    ends.append(e)
                env.update_from(*ends)
            else:
                self._aj021_simple(stmt, env)

    def _aj021_simple(self, stmt: ast.stmt, env: _Env) -> None:
        discarded = {id(stmt.value)} if isinstance(stmt, ast.Expr) else set()
        for child in ast.iter_child_nodes(stmt):
            self._scan(child, env, discarded)
        if isinstance(stmt, ast.Assign):
            self._check_targets(stmt, stmt.targets, env, "assignment to")
            self._bind(stmt.targets, stmt.value, env)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            self._check_targets(stmt, [stmt.target], env, "assignment to")
            self._bind([stmt.target], stmt.value, env)
        elif isinstance(stmt, ast.AugAssign):
            t = stmt.target
            container = isinstance(
                stmt.value, (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.ListComp, ast.DictComp, ast.SetComp)
            ) or (self.is_process and t.id in self.owned if isinstance(t, ast.Name) else False)
            if isinstance(t, ast.Name) and t.id in env.full and not container:
                pass  # sw = state.sw; sw += rain: a JAX array leaf, += rebinds (arrays are immutable)
            elif isinstance(t, ast.Name) and t.id in env.full:
                op = ast.unparse(ast.BinOp(left=ast.Name(id="a"), op=stmt.op, right=ast.Name(id="b")))[2:-2]
                self._aj021(
                    stmt,
                    f"augmented assignment {t.id} {op}= ... (in place on a NumPy array, list or dict)",
                    t,
                    env.full[t.id],
                    f"write {t.id} = {t.id} {op} ... (a new value) instead",
                )
            else:
                self._check_targets(stmt, [t], env, "augmented assignment to")
        elif isinstance(stmt, ast.Delete):
            self._check_targets(stmt, stmt.targets, env, "del")
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    env.forget(t.id)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                env.forget((a.asname or a.name).split(".")[0])

    def check_aj021(self) -> None:
        env = _Env(full={a: a for a in self.owned}, shallow={a: a for a in self.fresh})
        self._aj021_block(self.fn.body, env)


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
