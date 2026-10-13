"""Import analysis of :mod:`agrijax.core.lint`: the slots of AJ008, the import statements of a
module (static and dynamic) and the import closure of a set of modules."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from agrijax.core._lint.rules import _KERNEL_DIR, RULES, Finding

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
_BELOW_SLOTS: tuple[str, ...] = ("core", "iface", "forcing")
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


#: calls that import a module by name: ``importlib.import_module(name)``, ``import_module(name)``,
#: ``__import__(name)`` and ``importlib.__import__(name)``
_DYNAMIC_IMPORTS: frozenset[str] = frozenset({"import_module", "__import__"})


def _module_strings(tree: ast.AST) -> dict[str, str]:
    """Module-level names bound once to a string constant (``TARGET = "a.b"``), for resolving the
    argument of a dynamic import."""
    out: dict[str, str] = {}
    seen: set[str] = set()
    for node in getattr(tree, "body", []):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = list(node.targets), node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            if t.id in seen:
                out.pop(t.id, None)
                continue
            seen.add(t.id)
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                out[t.id] = value.value
    return out


def dynamic_imports(tree: ast.AST) -> list[tuple[ast.Call, ast.Import | ast.ImportFrom | None]]:
    """Every dynamic import call of ``tree`` with the import statement it amounts to: a call of
    ``import_module``/``__import__`` (bare or as an attribute, e.g. ``importlib.import_module``)
    whose first argument is a string constant, or a module-level name bound once to one, maps to
    ``import <name>`` (``from .<name> import`` for a relative ``.name``); any other argument maps
    to ``None`` (the lint cannot tell which module it loads)."""
    consts = _module_strings(tree)
    out: list[tuple[ast.Call, ast.Import | ast.ImportFrom | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else None
        if name not in _DYNAMIC_IMPORTS or not node.args:
            continue
        arg = node.args[0]
        target: str | None = None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            target = arg.value
        elif isinstance(arg, ast.Name):
            target = consts.get(arg.id)
        stmt: ast.Import | ast.ImportFrom | None = None
        if target:
            level = len(target) - len(target.lstrip("."))
            if level:
                stmt = ast.ImportFrom(module=target[level:] or None, names=[], level=level)
            else:
                stmt = ast.Import(names=[ast.alias(name=target)])
            ast.copy_location(stmt, node)
        out.append((node, stmt))
    return out


def _is_processes_module(dotted: str) -> bool:
    parts = dotted.split(".")
    return len(parts) >= 2 and parts[0] == "agrijax" and parts[1] == _KERNEL_DIR


def _check_below_slots(tree: ast.AST, path: str, package: tuple[str, ...]) -> list[Finding]:
    level, msg = RULES["AJ008"]
    where = "/".join(package)
    iface = package[1] == "iface"
    out: list[Finding] = []
    for node in _import_nodes(tree):
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
                        else f"{package[1]} imports no agrijax.processes module"
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


def _import_nodes(tree: ast.AST) -> Iterator[ast.Import | ast.ImportFrom]:
    """Every import statement of ``tree`` (also inside functions) and the statement every
    resolvable dynamic import amounts to (:func:`dynamic_imports`)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
    for _, stmt in dynamic_imports(tree):
        if stmt is not None:
            yield stmt
