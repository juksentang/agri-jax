"""The file-level rules of :mod:`agrijax.core.lint`: AJ009-AJ011 (no hidden state, no global
paths, no forcing that shadows a port) and AJ012 (the io / processes layers)."""

from __future__ import annotations

import ast
from pathlib import Path

from agrijax.core._lint.ast_facts import _call_name
from agrijax.core._lint.imports import (
    _below_slots_package,
    _check_below_slots,
    _import_nodes,
    dynamic_imports,
    import_slot,
    module_imports,
    slot_of_path,
)
from agrijax.core._lint.rules import _KERNEL_DIR, _PROCESS_DECORATOR, RULES, Finding

# ---------------------------------------------------------------------------
# AJ009-AJ011: no hidden state, no global paths, no forcing that shadows a port
# ---------------------------------------------------------------------------

#: directories whose files keep no module state (AJ009)
_STATELESS_DIRS: frozenset[str] = frozenset({"processes", "iface", "forcing"})
#: methods that mutate a container in place
_MUTATORS: frozenset[str] = frozenset(
    {
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "popitem",
        "clear",
        "update",
        "setdefault",
        "add",
        "discard",
    }
)
#: constructors of mutable containers
_MUTABLE_CALLS: frozenset[str] = frozenset(
    {"dict", "list", "set", "defaultdict", "OrderedDict", "Counter", "deque"}
)
#: global namespaces a slot process never names in its reads/writes (AJ010)
_GLOBAL_HEADS: frozenset[str] = frozenset({"iface", "prev", "ledger", "forcing"})


def _in_dirs(path: str, dirs: frozenset[str]) -> bool:
    return any(d in dirs for d in Path(path).parts[:-1])


def _mutable_module_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if value is None:
            continue
        mutable = isinstance(
            value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)
        ) or (isinstance(value, ast.Call) and _call_name(value) in _MUTABLE_CALLS)
        if mutable:
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _check_aj009(tree: ast.AST, path: str) -> list[Finding]:
    if not isinstance(tree, ast.Module) or not _in_dirs(path, _STATELESS_DIRS):
        return []
    level, msg = RULES["AJ009"]
    mutable = _mutable_module_names(tree)
    out: list[Finding] = []

    def hit(node: ast.AST, what: str) -> None:
        out.append(
            Finding(
                "AJ009",
                level,
                path,
                getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0),
                f"{msg}: {what}",
            )
        )

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {a.arg for a in (*fn.args.args, *fn.args.kwonlyargs, *fn.args.posonlyargs)}
        for node in ast.walk(fn):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                hit(node, f"{type(node).__name__.lower()} {', '.join(node.names)} in {fn.name}")
            elif isinstance(node, (ast.Assign, ast.Delete, ast.AugAssign)):
                tgts = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                for t in tgts:
                    if isinstance(t, ast.Name):
                        local.add(t.id)
                    elif (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and t.value.id in mutable - local
                    ):
                        hit(node, f"{fn.name} writes into module-level {t.value.id!r}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _MUTATORS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in mutable - local
            ):
                hit(node, f"{fn.name} calls {node.func.value.id}.{node.func.attr}() on module-level state")
    return out


def _process_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.Call | None:
    for d in fn.decorator_list:
        if isinstance(d, ast.Call) and _call_name(d) == _PROCESS_DECORATOR:
            return d
    return None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _check_aj010(tree: ast.AST, path: str) -> list[Finding]:
    if slot_of_path(path) is None:
        return []
    level, msg = RULES["AJ010"]
    out: list[Finding] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dec = _process_decorator(fn)
        if dec is None:
            continue
        for kw in ("reads", "writes"):
            val = _keyword(dec, kw)
            if not isinstance(val, (ast.Tuple, ast.List)):
                continue
            for el in val.elts:
                if isinstance(el, ast.Constant) and isinstance(el.value, str):
                    head = el.value.split(".", 1)[0]
                    if head in _GLOBAL_HEADS:
                        out.append(
                            Finding(
                                "AJ010",
                                level,
                                path,
                                el.lineno,
                                el.col_offset,
                                f"{msg}: {fn.name} {kw} {el.value!r}; declare a port() field and bind it",
                            )
                        )
    return out


def _agrijax_dir() -> Path | None:
    import importlib.util

    try:
        spec = importlib.util.find_spec("agrijax")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


_PORT_FIELDS: dict[str, frozenset[str]] = {}


def port_field_names(iface_dir: str | Path | None = None) -> frozenset[str]:
    """Field names of the port records (``State`` subclasses) defined in ``agrijax/iface/*.py``,
    read from the source (no import). ``iface_dir`` defaults to the installed ``agrijax.iface``."""
    d = Path(iface_dir) if iface_dir is not None else (_agrijax_dir() or Path(".")) / "iface"
    key = str(d.resolve()) if d.exists() else str(d)
    if key in _PORT_FIELDS:
        return _PORT_FIELDS[key]
    names: set[str] = set()
    for f in sorted(d.glob("*.py")) if d.is_dir() else ():
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            if not any(isinstance(b, ast.Name) and b.id == "State" for b in cls.bases):
                continue
            names.update(
                n.target.id
                for n in cls.body
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
            )
    out = frozenset(names)
    _PORT_FIELDS[key] = out
    return out


def _check_aj011(tree: ast.AST, path: str, iface_dir: str | Path | None = None) -> list[Finding]:
    if slot_of_path(path) is None:
        return []
    level, msg = RULES["AJ011"]
    fields = port_field_names(iface_dir)
    out: list[Finding] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dec = _process_decorator(fn)
        args = fn.args.posonlyargs + fn.args.args
        if dec is None or len(args) < 3:
            continue
        key = _keyword(dec, "key")
        if (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and "replay" in key.value.rsplit(":", 1)[-1]
        ):
            continue
        forcing = args[2].arg
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == forcing
                and node.attr in fields
            ):
                out.append(
                    Finding(
                        "AJ011",
                        level,
                        path,
                        node.lineno,
                        node.col_offset,
                        f"{msg}: {fn.name} reads {forcing}.{node.attr}; read it from the port",
                    )
                )
    return out


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
    for call, stmt in dynamic_imports(tree):
        if stmt is None:
            out.append(
                Finding(
                    "AJ008",
                    level,
                    path,
                    call.lineno,
                    call.col_offset,
                    f"{msg}: processes/{slot} imports a module the lint cannot resolve (a dynamic "
                    "import of a non-constant name); import agrijax.core and the port records "
                    "agrijax.iface with import statements",
                )
            )
    for node in _import_nodes(tree):
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
# AJ012: io below the processes, site assembly above both
# ---------------------------------------------------------------------------

#: what a file of ``agrijax/io/`` may import from ``agrijax``
_IO_MAY_IMPORT: tuple[str, ...] = ("agrijax.core", "agrijax.forcing", "agrijax.port", "agrijax.io")
#: package layers that import neither the readers nor the site assembly
_NO_IO_LAYERS: frozenset[str] = frozenset({"core", "iface", "forcing", "processes", "models"})
#: the layers a file of :data:`_NO_IO_LAYERS` may not import
_IO_LAYERS: frozenset[str] = frozenset({"io", "sites"})


def _layer_package(path: str | Path) -> tuple[str, tuple[str, ...]] | None:
    """``(layer, package parts)`` of a file for AJ012: the layer is the directory right below the
    last ``agrijax/`` of the path (``agrijax/io/dssat/sol.py`` -> ``("io", ("agrijax", "io",
    "dssat"))``); a file under a ``processes/`` directory outside ``agrijax/`` is of the
    ``processes`` layer (the slot convention of :func:`slot_of_path`). ``None`` otherwise."""
    dirs = Path(path).parts[:-1]
    for i in range(len(dirs) - 1, 0, -1):
        if dirs[i - 1] == "agrijax":
            return dirs[i], tuple(dirs[i - 1 :])
    where = slot_of_path(path)
    if where is not None:
        return _KERNEL_DIR, ("agrijax", _KERNEL_DIR, *where[1])
    return None


def _agrijax_layer(dotted: str) -> str | None:
    parts = dotted.split(".")
    return parts[1] if len(parts) >= 2 and parts[0] == "agrijax" else None


def _check_aj012(tree: ast.AST, path: str) -> list[Finding]:
    where = _layer_package(path)
    if where is None:
        return []
    layer, package = where
    if layer != "io" and layer not in _NO_IO_LAYERS:
        return []
    level, msg = RULES["AJ012"]
    out: list[Finding] = []
    for node in _import_nodes(tree):
        names = module_imports(node, package)
        if layer == "io":
            bad = [
                m
                for m in names
                if _agrijax_layer(m) is not None
                and not any(m == p or m.startswith(p + ".") for p in _IO_MAY_IMPORT)
            ]
            hint = (
                "io imports only agrijax.core, agrijax.forcing, agrijax.port and agrijax.io; "
                "move the assembly to agrijax.sites"
            )
        else:
            bad = [m for m in names if _agrijax_layer(m) in _IO_LAYERS]
            hint = f"{layer} imports no agrijax.io or agrijax.sites module; the site layer builds its inputs"
        if bad:
            out.append(
                Finding(
                    "AJ012",
                    level,
                    path,
                    node.lineno,
                    node.col_offset,
                    f"{msg}: {layer} imports {bad[0]}; {hint}",
                )
            )
    return out
