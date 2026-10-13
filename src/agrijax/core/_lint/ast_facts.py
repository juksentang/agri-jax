"""What the per-function rules of :mod:`agrijax.core.lint` need to know beyond one function: the
names bound to NumPy and ``operator`` in a module, the host-side API (AJ020), the AJ021 alias
environment, and the small AST helpers they share."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterator, Sequence

from agrijax.core._lint.imports import _DYNAMIC_IMPORTS, _module_file
from agrijax.core._lint.rules import (
    _CALLBACK_CALLS,
    _GRID_SIZE_ATTRS,
    _NP_FACTORIES,
    _NUMPY_DEFAULT,
    _OPERATOR_DEFAULT,
    _PROCESS_DECORATOR,
    _PYTREE_BASES,
    _STATIC_ANNOTATIONS,
    _STATIC_ATTRS,
    _STATIC_CALLS,
    AJ020_HOST_MARKER,
)


def _is_type_checking(test: ast.AST) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_statements(body: Sequence[ast.stmt]) -> Iterator[ast.stmt]:
    """The module-level statements in order, into ``if`` / ``try`` / ``with`` blocks, without
    ``if TYPE_CHECKING:`` blocks (they do not run) and without function and class bodies."""
    for s in body:
        if isinstance(s, ast.If):
            if not _is_type_checking(s.test):
                yield from _module_statements(s.body)
            yield from _module_statements(s.orelse)
        elif isinstance(s, (ast.Try, ast.TryStar)):
            yield from _module_statements(s.body)
            for h in s.handlers:
                yield from _module_statements(h.body)
            yield from _module_statements(s.orelse)
            yield from _module_statements(s.finalbody)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            yield from _module_statements(s.body)
        else:
            yield s


def _ref_in(node: ast.AST, names: dict[str, str], module: str) -> str | None:
    """The dotted path inside ``module`` that a reference expression names, ``""`` for the module
    itself, ``None`` when it is not one. ``names`` maps the local names bound to the module or to
    its members (:func:`numpy_names`): ``np.linalg.norm`` -> ``linalg.norm``; ``e`` after ``from
    numpy import exp as e`` -> ``exp``; ``getattr(np, "exp")`` -> ``exp`` (``?`` for a computed
    name); ``importlib.import_module("numpy")`` -> ``""``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    base: str | None = None
    if isinstance(node, ast.Name):
        base = names.get(node.id)
    elif isinstance(node, ast.Call):
        name = _call_name(node)
        if name == "getattr" and isinstance(node.func, ast.Name) and len(node.args) >= 2:
            inner = _ref_in(node.args[0], names, module)
            if inner is not None:
                a = node.args[1]
                attr = a.value if isinstance(a, ast.Constant) and isinstance(a.value, str) else "?"
                base = ".".join(p for p in (inner, attr) if p)
        elif name in _DYNAMIC_IMPORTS and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                mod = a.value.split(".")
                if mod[0] == module:
                    base = ".".join(mod[1:])
    if base is None:
        return None
    return ".".join(p for p in (base, *reversed(parts)) if p)


def _bind_module_names(stmt: ast.AST, names: dict[str, str], module: str) -> None:
    """Update ``names`` (local name -> dotted path inside ``module``) for one statement: an
    import, or an assignment of a reference (``N = np``, ``_EXP = np.exp``); any other binding of
    a name removes it."""
    if isinstance(stmt, ast.Import):
        for a in stmt.names:
            parts = a.name.split(".")
            local = a.asname or parts[0]
            if parts[0] == module:
                names[local] = ".".join(parts[1:]) if a.asname else ""
            else:
                names.pop(local, None)
    elif isinstance(stmt, ast.ImportFrom):
        mod = stmt.module.split(".") if stmt.module and stmt.level == 0 else []
        for a in stmt.names:
            local = a.asname or a.name
            if mod and mod[0] == module:
                names[local] = ".".join([*mod[1:], a.name])
            else:
                names.pop(local, None)
    elif isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
        ref = _ref_in(stmt.value, names, module)
        if ref is None and module == "numpy" and isinstance(stmt.value, ast.Call):
            factory = _ref_in(stmt.value.func, names, module)
            ref = factory if factory in _NP_FACTORIES else None  # f = np.vectorize(g): f is NumPy
        for t in targets:
            if isinstance(t, ast.Name) and ref is not None:
                names[t.id] = ref
            else:
                for n in _target_names(t):
                    names.pop(n, None)
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.pop(stmt.name, None)


def _module_names(tree: ast.AST, module: str, default: dict[str, str]) -> dict[str, str]:
    out = dict(default)
    for s in _module_statements(getattr(tree, "body", [])):
        _bind_module_names(s, out, module)
    return out


def numpy_names(tree: ast.AST) -> dict[str, str]:
    """``{local name: dotted path inside numpy}`` of the names a module binds to NumPy at module
    level: ``import numpy as np`` -> ``{"np": ""}``, ``import numpy.linalg as la`` -> ``{"la":
    "linalg"}``, ``from numpy import exp as e`` -> ``{"e": "exp"}``, and assignments ``N = np``,
    ``_EXP = np.exp``, ``_np = importlib.import_module("numpy")``. ``np`` and ``numpy`` count by
    default, unless the module binds them to something else (``import jax.numpy as np``). Imports
    inside functions count for that function only; ``if TYPE_CHECKING:`` blocks do not run."""
    return _module_names(tree, "numpy", _NUMPY_DEFAULT)


def _numpy_dotted(func: ast.AST, names: dict[str, str]) -> str | None:
    """The dotted path inside numpy a reference names, ``None`` for the module itself or a
    non-NumPy name (see :func:`_ref_in`)."""
    return _ref_in(func, names, "numpy") or None


def _says_host_side(node: ast.AST) -> bool:
    """True when the docstring of a function or module starts with :data:`AJ020_HOST_MARKER`."""
    if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return (ast.get_docstring(node) or "").startswith(AJ020_HOST_MARKER)


def _callback_aliases(tree: ast.AST) -> frozenset[str]:
    """Local names bound to a host callback at module level: ``from jax.debug import callback``."""
    out: set[str] = set()
    for s in _module_statements(getattr(tree, "body", [])):
        if isinstance(s, ast.ImportFrom) and s.module in {"jax.debug", "jax.experimental"}:
            out.update(a.asname or a.name for a in s.names if a.name in {"callback", "io_callback"})
    return frozenset(out)


def _is_callback_call(call: ast.AST, aliases: Collection[str]) -> bool:
    if not isinstance(call, ast.Call):
        return False
    f = call.func
    if _call_name(call) in _CALLBACK_CALLS:
        return True
    if isinstance(f, ast.Attribute) and f.attr == "callback" and _call_name(f.value) == "debug":
        return True
    return isinstance(f, ast.Name) and f.id in aliases


def _callback_target(call: ast.Call) -> ast.AST | None:
    """The function a callback call runs (first argument or ``callback=``), through ``partial``."""
    cand = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "callback"), None)
    if isinstance(cand, ast.Call) and _call_name(cand) == "partial" and cand.args:
        cand = cand.args[0]
    return cand


def _callback_functions(tree: ast.AST, aliases: Collection[str] = ()) -> frozenset[str]:
    """Names passed as the callback of ``jax.pure_callback`` / ``io_callback`` / ``jax.debug.callback``
    (first argument or ``callback=``, also through ``functools.partial``): JAX runs them on the
    host with NumPy arrays."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if _is_callback_call(n, aliases):
            assert isinstance(n, ast.Call)
            t = _callback_target(n)
            if isinstance(t, ast.Name):
                out.add(t.id)
    return frozenset(out)


def _static_fields(cls: ast.ClassDef | None) -> frozenset[str]:
    """Attributes of a class that hold Python values, not traced leaves: fields declared with
    ``static=True`` (``eqx.field(static=True)``, ``field(..., static=True)``), ``ClassVar`` and
    plain class attributes (``_KEYS = (...)``)."""
    if cls is None:
        return frozenset()
    out: set[str] = set()
    for s in cls.body:
        if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
            v = s.value
            static = isinstance(v, ast.Call) and any(
                k.arg == "static" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in v.keywords
            )
            if static or "ClassVar" in ast.unparse(s.annotation):
                out.add(s.target.id)
        elif isinstance(s, ast.Assign):
            out.update(t.id for t in s.targets if isinstance(t, ast.Name))
    return frozenset(out)


@dataclass(frozen=True)
class _HostApi:
    """The host-side functions a module offers: its module-level host functions and, per class,
    its host methods (:data:`AJ020_HOST_MARKER`, a host module, a callback)."""

    functions: frozenset[str]
    methods: dict[str, frozenset[str]]


def _host_api(tree: ast.AST) -> _HostApi:
    body = getattr(tree, "body", [])
    host_module = _says_host_side(tree)
    callbacks = _callback_functions(tree, _callback_aliases(tree))
    functions = frozenset(
        n.name
        for n in body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not _is_process_decorated(n)
        and (host_module or n.name in callbacks or _says_host_side(n))
    )
    methods: dict[str, frozenset[str]] = {}
    for c in body:
        if isinstance(c, ast.ClassDef):
            ms = frozenset(
                m.name
                for m in c.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                and (host_module or _says_host_side(m))
            )
            if ms:
                methods[c.name] = ms
    return _HostApi(functions, methods)


_HOST_API_CACHE: dict[str, _HostApi] = {}


def _package_parts(path: str) -> tuple[Path, tuple[str, ...]] | None:
    """``(source root, package parts)`` of a file inside a package (directories with
    ``__init__.py``), ``None`` for a file outside one or a path that does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    d = p.resolve().parent
    parts: list[str] = []
    while (d / "__init__.py").is_file():
        parts.insert(0, d.name)
        d = d.parent
    return (d, tuple(parts)) if parts else None


def _imported_host_api(tree: ast.AST, path: str) -> _HostApi:
    """The host-side functions and methods that module-level ``from <module> import <name>``
    statements of ``tree`` bring in, resolved to the source files of the same source root."""
    where = _package_parts(path)
    if where is None:
        return _HostApi(frozenset(), {})
    root, package = where
    functions: set[str] = set()
    methods: dict[str, frozenset[str]] = {}
    for s in _module_statements(getattr(tree, "body", [])):
        if not isinstance(s, ast.ImportFrom):
            continue
        mod = s.module.split(".") if s.module else []
        if s.level:
            up = s.level - 1
            if up >= len(package):
                continue
            mod = [*package[: len(package) - up], *mod]
        file = _module_file(".".join(mod), root) if mod else None
        if file is None:
            continue
        key = str(file)
        if key not in _HOST_API_CACHE:
            _HOST_API_CACHE[key] = _HostApi(frozenset(), {})  # guards import cycles
            try:
                mod_tree = ast.parse(file.read_text(encoding="utf-8"))
            except SyntaxError:
                mod_tree = None
            if mod_tree is not None:
                own = _host_api(mod_tree)
                # names the module re-exports (``from .records import read`` in a package __init__)
                again = _imported_host_api(mod_tree, key)
                _HOST_API_CACHE[key] = _HostApi(
                    own.functions | again.functions, {**again.methods, **own.methods}
                )
        api = _HOST_API_CACHE[key]
        for a in s.names:
            local = a.asname or a.name
            if a.name in api.functions:
                functions.add(local)
            if a.name in api.methods:
                methods[local] = api.methods[a.name]
    return _HostApi(frozenset(functions), methods)


@dataclass(frozen=True)
class _FileFacts:
    """What AJ020 / AJ021 need to know about a file beyond one function."""

    numpy: dict[str, str]
    operator: dict[str, str]
    callback_aliases: frozenset[str]
    callbacks: frozenset[str]
    host_module: bool
    #: host-side functions callable by name here (defined in the file or imported)
    host_functions: frozenset[str]
    #: host-side methods per class name (defined in the file or imported)
    host_methods: dict[str, frozenset[str]]
    #: method -> its class; nested function -> its enclosing function
    class_of: dict[int, ast.ClassDef]
    #: the classes of the file that are pytree records (an ``eqx.Module``, ``State``, ``Params``,
    #: ``Coefficients``, or a subclass of one defined in the file)
    pytree_classes: frozenset[int]
    enclosing: dict[int, ast.FunctionDef | ast.AsyncFunctionDef]

    @staticmethod
    def of(tree: ast.AST, path: str = "<string>") -> _FileFacts:
        aliases = _callback_aliases(tree)
        own = _host_api(tree)
        imported = _imported_host_api(tree, path)
        class_of: dict[int, ast.ClassDef] = {}
        enclosing: dict[int, ast.FunctionDef | ast.AsyncFunctionDef] = {}

        def visit(node: ast.AST, fn: ast.FunctionDef | ast.AsyncFunctionDef | None) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if fn is not None:
                        enclosing[id(child)] = fn
                    if isinstance(node, ast.ClassDef):
                        class_of[id(child)] = node
                    visit(child, child)
                else:
                    visit(child, fn)

        visit(tree, None)
        classes = [c for c in ast.walk(tree) if isinstance(c, ast.ClassDef)]
        pytree_names: set[str] = set()
        changed = True
        while changed:
            changed = False
            for c in classes:
                if c.name not in pytree_names and any(
                    _call_name(b) in _PYTREE_BASES or _call_name(b) in pytree_names for b in c.bases
                ):
                    pytree_names.add(c.name)
                    changed = True
        return _FileFacts(
            numpy=numpy_names(tree),
            operator=_module_names(tree, "operator", _OPERATOR_DEFAULT),
            callback_aliases=aliases,
            callbacks=_callback_functions(tree, aliases),
            host_module=_says_host_side(tree),
            host_functions=own.functions | imported.functions,
            host_methods={**imported.methods, **own.methods},
            class_of=class_of,
            pytree_classes=frozenset(id(c) for c in classes if c.name in pytree_names),
            enclosing=enclosing,
        )


def _chain_root(node: ast.AST) -> str | None:
    """The name an attribute / subscript / method chain starts from (``state["crop"].lai`` ->
    ``state``; ``getattr(state, "crop")`` -> ``state``)."""
    while True:
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
            node = node.value
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                node = node.func.value
            elif node.args:
                node = node.args[-1] if _call_name(node) == "super" else node.args[0]
            else:
                return None
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return None


def _has_at_link(node: ast.AST) -> bool:
    """True when a receiver chain goes through ``.at`` (``x.at[i].add(v)`` is functional)."""
    while isinstance(node, (ast.Attribute, ast.Subscript, ast.Call)):
        if isinstance(node, ast.Attribute) and node.attr == "at":
            return True
        node = node.func if isinstance(node, ast.Call) else node.value
    return False


def _flat_targets(target: ast.AST) -> Iterator[ast.AST]:
    if isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            yield from _flat_targets(e)
    elif isinstance(target, ast.Starred):
        yield from _flat_targets(target.value)
    else:
        yield target


def _leaves(body: Sequence[ast.stmt]) -> bool:
    """True when a block ends by leaving the function (``return`` / ``raise``)."""
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))


def _lambda_params(lam: ast.Lambda) -> list[str]:
    a = lam.args
    out = [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    out += [x.arg for x in (a.vararg, a.kwarg) if x is not None]
    return out


def _pattern_names(p: ast.AST) -> set[str]:
    """Names a ``match`` pattern captures."""
    out: set[str] = set()
    for n in ast.walk(p):
        if isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
    return out


def _tracer_guard(stmt: ast.AST) -> set[str]:
    """Names ``x`` of a guard ``if isinstance(x, Tracer): return`` (or ``raise``; ``jax.core.Tracer``,
    a tuple of types, several tests joined by ``or``): below it ``x`` is concrete."""
    if not (isinstance(stmt, ast.If) and not stmt.orelse and stmt.body):
        return set()
    if not isinstance(stmt.body[-1], (ast.Return, ast.Raise)):
        return set()
    tests = (
        stmt.test.values
        if isinstance(stmt.test, ast.BoolOp) and isinstance(stmt.test.op, ast.Or)
        else [stmt.test]
    )
    out: set[str] = set()
    for t in tests:
        if not (isinstance(t, ast.Call) and _call_name(t) == "isinstance" and len(t.args) == 2):
            return set()
        obj, typ = t.args
        types = typ.elts if isinstance(typ, ast.Tuple) else [typ]
        if not (isinstance(obj, ast.Name) and any(_call_name(x) == "Tracer" for x in types)):
            return set()
        out.add(obj.id)
    return out


class _Env:
    """AJ021 aliases at one point of a function: ``full`` names are (a part of) an argument,
    ``shallow`` names a new container whose elements are (``dict(state)``, ``[state.x, 1.0]``,
    ``*args``), ``lambdas`` the lambdas bound to names."""

    __slots__ = ("full", "lambdas", "shallow")

    def __init__(
        self,
        full: dict[str, str] | None = None,
        shallow: dict[str, str] | None = None,
        lambdas: dict[str, ast.Lambda] | None = None,
    ) -> None:
        self.full: dict[str, str] = dict(full or {})
        self.shallow: dict[str, str] = dict(shallow or {})
        self.lambdas: dict[str, ast.Lambda] = dict(lambdas or {})

    def copy(self) -> _Env:
        return _Env(self.full, self.shallow, self.lambdas)

    def forget(self, name: str) -> None:
        self.full.pop(name, None)
        self.shallow.pop(name, None)
        self.lambdas.pop(name, None)

    def update_from(self, *envs: _Env) -> None:
        """Become the join of ``envs``: a name is an alias after a branch when it is one on any path."""
        full: dict[str, str] = {}
        shallow: dict[str, str] = {}
        lambdas: dict[str, ast.Lambda] = {}
        for e in envs:
            full.update(e.full)
            shallow.update(e.shallow)
            lambdas.update(e.lambdas)
        self.full, self.shallow, self.lambdas = full, shallow, lambdas


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


def _dynamic_names(
    node: ast.AST, static_attrs: Collection[str] = _STATIC_ATTRS, static_self: Collection[str] = ()
) -> set[str]:
    """Root names of ``node`` outside static sub-expressions (shape queries, string comparisons;
    ``self.<f>`` for ``f`` in ``static_self``)."""
    if isinstance(node, ast.Attribute) and node.attr in static_attrs:
        return set()
    if (
        isinstance(node, ast.Attribute)
        and node.attr in static_self
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
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
        out |= _dynamic_names(child, static_attrs, static_self)
    return out


#: attributes that hold Python values for AJ020: the shape queries and the grid sizes
_AJ020_STATIC_ATTRS: frozenset[str] = frozenset(_STATIC_ATTRS | _GRID_SIZE_ATTRS)


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
