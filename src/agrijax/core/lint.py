"""Standalone AST lint for the three process rules.

Scope
-----
* every function decorated with ``@process`` gets every rule;
* every other function defined in a file under a ``processes/`` directory (the numerical
  kernels the processes call: ``shuttleworth_wallace``, ``theta_of_h``, private helpers, ...)
  gets the numerical rules AJ001-AJ003, AJ006, AJ007, AJ020 and AJ021; AJ004/AJ005 are about the
  process contract and do not apply to kernels that return NamedTuples or arrays;
* every other function in a file under ``forcing/`` (host-side NumPy preprocessing, not traced)
  gets AJ007 only: its coefficients are labelled like a kernel's;
* a function passed as the first argument of a ``process(fn, ...)`` call in a file under
  ``processes/``, ``forcing/`` or ``models/`` (an assembly entry built without the decorator) is a
  process and gets every rule;
* ``--all`` applies every rule to every function in every file;
* AJ008 is a file-level rule: it checks the import statements of every file under a
  ``processes/`` directory, and of every file of ``agrijax/core/`` and ``agrijax/iface/``,
  whatever the functions in it.
* AJ012 is a file-level rule on the layers of the package: it checks the imports of every file
  of ``agrijax/io/``, ``agrijax/core/``, ``agrijax/iface/``, ``agrijax/forcing/``,
  ``agrijax/models/`` and of every file under a ``processes/`` directory.
* a nested function (a ``lax.scan`` body) is checked as a function of its own, with its own
  arguments: the names of the enclosing function it closes over are not followed.

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
                functions. Dynamic imports count as imports: ``importlib.import_module(name)``
                and ``__import__(name)`` with a string constant (or a module-level name bound to
                one) are checked like the statement they amount to, and one whose target the
                lint cannot resolve is reported under ``processes/``. The same rule keeps the
                port records below the slots: a file of ``agrijax/core/``, ``agrijax/iface/`` or
                ``agrijax/forcing/`` that imports any ``agrijax.processes`` module is reported, and a file of
                ``agrijax/iface/`` may import only ``agrijax.core``, ``agrijax.iface`` and
                third-party packages (so that the records' import closure is ``iface`` plus
                ``core``; :func:`import_closure` checks it transitively). ``--strict`` enforces
                it (the tree has no AJ008 finding).
AJ009  warning  Module state mutated from a function, in a file under ``processes/``, ``iface/`` or
                ``forcing/``: a ``global``/``nonlocal`` statement, or a call of a mutating method
                (``append``, ``update``, ``setdefault``, ...), a subscript assignment or a ``del``
                on a module-level name bound to a mutable container (``{}``, ``[]``, ``set()``,
                ``dict()``, ``defaultdict()``, ...). Slots keep no hidden state; everything a day
                changes is in the returned state (three rules). Registries live in ``agrijax.core``.
                Not enforced by ``--strict`` until the numerical-settings registry moves from
                ``processes/soil_water/coefficients.py`` to core (gap G24); ``--enforce AJ009``.
AJ010  warning  A ``@process`` under ``processes/`` whose ``reads``/``writes`` literal names a global
                path (``iface.``, ``prev.``, ``ledger.``, ``forcing.``): a slot process declares
                paths relative to its module state, and reaches another slot's record only through
                a ``port()`` field bound at assembly (:func:`agrijax.core.ports.bind`).
AJ011  warning  A ``@process`` under ``processes/`` that reads ``<forcing>.<name>`` where ``<name>`` is
                a field of a port record of ``agrijax.iface`` (a ``State`` class; the names are read
                from the iface source, not imported): the process takes from the forcing a quantity
                the contract delivers through a port, so the coupled binding would silently ignore
                the port (M3 contract section 11, open item 2). Processes whose ``key=`` variant
                contains ``replay`` are exempt. Not enforced by ``--strict`` until the known cases
                (gap G1) are closed; ``--enforce AJ011`` (or ``--strict-aj011``) enforces it.
AJ012  warning  An import across the io / processes layers. The file readers ``agrijax.io`` sit
                below the processes: a file of ``agrijax/io/`` may import from ``agrijax`` only
                ``agrijax.core``, ``agrijax.forcing``, ``agrijax.port`` and ``agrijax.io``; a file
                of ``agrijax/core/``, ``agrijax/iface/``, ``agrijax/forcing/``,
                ``agrijax/models/`` or under a ``processes/`` directory imports no
                ``agrijax.io`` and no ``agrijax.sites`` module. Site assembly (reading a site's
                or a reference run's files into the records of processes) lives in
                ``agrijax.sites``, the only package layer that imports both (with calibration,
                tests and scripts above it). Statements inside functions and resolvable dynamic
                imports count. ``--strict`` enforces it (the tree has no AJ012 finding).
AJ020  error    NumPy on a traced value. The value dependence is the AJ001 taint with static
                sub-expressions left out (``n = x.shape[-1]``, ``len(x)``, ``x.dtype``,
                ``np.ndim(x)``, a grid size such as ``params.n_layer``, string comparisons), through
                comprehension targets and lambda parameters (a lambda gets traced values from
                ``tree_map``, ``vmap``, ``scan``); in a method of a pytree class (``eqx.Module``,
                ``State``, ``Params``, ``Coefficients`` or a subclass in the same file) ``self`` is
                traced too, except its static fields (``field(static=True)``, ``ClassVar``, plain
                class attributes). An argument checked by a top-level ``if isinstance(x, Tracer):
                return`` (or ``raise``) before any other use is concrete below it. Reported:
                a call of a NumPy function with a traced argument: ``np.f(...)``, ``numpy.f``,
                ``np.linalg.f``, ``from numpy import f``, the aliases a file or function binds
                (``N = np``, ``_EXP = np.exp``, ``f = getattr(np, "exp")``,
                ``importlib.import_module("numpy")``, ``f = np.vectorize(g)``), and
                ``np.vectorize(g)(x)`` / ``np.frompyfunc(...)(x)``; a NumPy function passed as a value
                to a call that gets traced values, or whose result does (``tree_map(np.asarray,
                state)``, ``map(np.exp, xs)``, ``jax.vmap(np.sum)(x)``); every ``np.random`` call
                in traced code, whatever its arguments (drawn once at trace time: a constant under
                ``jit``); a call with traced arguments into a host-side function (see below) of the
                same file or imported with ``from <module> import <name>`` from the same source root
                (``helper(...)``, ``Class.method(...)``, ``self.method(...)``, followed through the
                re-exports of a package ``__init__``), or in a function nested in the traced one; a
                host-side function passed as a value or bound to a name (``jax.vmap(h)(x)``,
                ``tree_map(h, state)``, ``functools.partial(h, ...)(x)``, ``g = h; g(x)``) counts the
                same way. Imports count at
                module level (``if TYPE_CHECKING:`` blocks do not run) and, inside a function, for
                that function only; ``np`` bound to ``jax.numpy`` is not NumPy. Allowed: NumPy on
                literals, module constants, static arguments and shape queries (``np.zeros(3)``,
                ``np.asarray(_TABLE)``, ``np.arange(x.shape[-1])``: trace-time constants, which the
                tree relies on); the structure and dtype queries of :data:`AJ020_STATIC_CALLS` on
                anything; dtypes, types and constants as values (``dtype=np.float32``,
                ``isinstance(x, np.ndarray)``, ``np.pi``); a NumPy function passed as the callback of
                ``jax.pure_callback`` / ``io_callback`` / ``jax.debug.callback``. An in-place NumPy
                writer into an argument (``np.copyto(state.x, v)``) is left to AJ021. Host-side
                code is exempt: a non-``@process`` function whose docstring's first line starts
                with :data:`AJ020_HOST_MARKER` (``Host-side: build the grid from the file
                records.``), every non-``@process`` function of a module whose docstring's first
                line does, a function nested in a host-side one, and a function passed (also through
                ``functools.partial``) as a host callback. A mention of "host-side" anywhere else
                does not count, and a ``@process`` is never exempt. One finding per outermost call
                (``np.cumsum(np.asarray(x))``).
AJ021  error    In-place mutation of an argument or of an alias of one. The arguments are the traced
                ones and, in a method of a pytree class other than ``__init__``-like constructors,
                ``self`` (static-annotated arguments and ``cls`` are not); ``*args`` / ``**kwargs``
                are fresh containers whose elements are the caller's. Reported: an assignment, an
                augmented assignment or ``del`` whose target is an attribute or subscript of it
                (``state["crop"]["lai"] = lai``, ``state.crop.lai += x``, ``del params.k``); an
                augmented assignment to an alias name when it works on a container: an argument
                of a ``@process`` itself (``forcing += 1``) or a container right-hand side
                (``lst += [...]``, ``d |= {...}``); ``sw = state.sw; sw += rain`` and ``dt *= 0.5`` on
                a kernel argument are array leaves, where ``+=`` rebinds, and are not reported;
                ``setattr`` / ``delattr``; in-place methods on it: ``.fill``, ``.put``,
                ``.itemset``, ``.pop``, ``.setdefault``, the dunders ``__setitem__``,
                ``__setattr__``, ``__iadd__``, ... anywhere, and ``.update``, ``.append``,
                ``.extend``, ``.add``, ``.sort``, ``.partition``, ``.resize``, ``.reverse``, ... when
                the result is discarded (a statement, a lambda body, the element of a discarded
                comprehension; ``jax.Array.sort`` and an optimizer's ``update`` return what is used);
                the unbound forms (``dict.update(state, d)``, ``list.append(state.xs, v)``,
                ``np.ndarray.fill(x, 0)``, ``object.__setattr__(x, ...)``, ``type(x).__setattr__``,
                ``super(C, x).__setattr__``) and :mod:`operator` (``operator.setitem``,
                ``operator.iadd``, ...); NumPy's in-place functions (``np.copyto``, ``np.put``,
                ``np.place``, ``np.putmask``, ``np.fill_diagonal``, ``np.put_along_axis``,
                ``np.random.shuffle``, ``np.<ufunc>.at``) on it; ``out=`` into it. Lambda bodies are
                checked, their parameters bound to what they get (the leaves of ``tree_map(fn,
                state)``, the arguments of a direct call of the lambda or of the name it is bound to).
                An alias is a name bound to an argument or a part or view of it without a copy
                (``crop = state.crop``, ``getattr(state, "crop")``, ``state.get("crop")``,
                ``next(iter(state.layers))``, ``np.asarray(x)``, ``np.array(x, copy=False)``,
                ``np.ravel`` / ``np.reshape``, ``x.view()``, ``x.T``, ``x.flat``, tuple unpacking,
                ``for`` / comprehension targets, ``match`` captures, walrus). A shallow copy
                (``dict(state)``, ``{**state}``, ``copy.copy(x)``, ``x.copy()``, ``list(x)``, a
                list / tuple / dict literal holding arguments, ``[*x]``) is a new container: setting
                or appending at its top level is fine, changing one of its elements is reported
                (``d = dict(state); d["crop"]["lai"] = v``). The statements are walked in order: a
                rebinding to a new object (``x * 2.0``, ``np.array(x)``, ``state.replace(...)``) ends
                an alias for the rest of its block, and after an ``if`` / loop / ``try`` / ``match``
                a name is an alias when it is one on any path that goes on (a branch that ends in
                ``return`` / ``raise`` does not reach the join; loop bodies are walked twice).
                ``x.at[i].set(v)`` chains are functional. Mutating an input either fails only when
                traced (equinox modules are frozen, JAX arrays immutable) or silently changes the
                caller's pytree (dict states, NumPy arrays, lists); return a new state with
                ``eqx.tree_at`` / ``.replace`` (``x.at[i].set(v)`` for an array element).

Known limits of AJ020 / AJ021 (by design, or not worth the machinery): a nested function (a
``lax.scan`` body) is checked with its own arguments only, so what it closes over from the
enclosing function is not followed; helpers outside the linted directories (``processes/``, and
the kit's ``kernel=True`` scope) are not scanned, and a host-side helper is recognised across
modules only through ``from <module> import <name>`` of the same source root (not through
``module.helper(...)``); static fields are known on ``self`` only, so ``np.asarray(params.<static
field>)`` is reported (read it with ``jnp`` or on the host); ``x.size`` is taken as the static
array size; ``math.*``, ``scipy.*``, ``float(x)`` and ``x.item()`` on traced values are out of
scope (they fail loudly under ``jit``); ``x.copy()`` is taken as a shallow copy (right for dicts and
lists; a NumPy copy is deep, so changing an element of a copied 2-D array may be reported).
Also not followed: a ``Tracer`` guard is taken at its word (``isinstance(state, Tracer)`` on a
record, or a local class named ``Tracer``, still makes the name concrete); a callback function
is known by name, so a method of the same name is exempt too; static fields and class attributes
are known on the class itself, not inherited ones; lambda parameters are always traced (a lambda
over constants, ``max(..., key=lambda k: np.abs(k))``, is reported); a loop target that rebinds
``np`` does not hide NumPy; NumPy functions kept in containers (``{"e": np.exp}``, ``f, g =
np.exp, np.log``) or as argument defaults are not followed; for AJ021, a ``with`` target, a
mutating call whose result is used in an expression (``_ = x.append(v)``, ``(x.append(v), 0)``,
``x.append(v) or y``), lambdas kept in containers or reached through a default argument, a
mutator bound to a name first (``up = state.update``, ``s = object.__setattr__``), a pytree
base class defined in another module, and the nested functions of processes outside
``processes/`` (they are not linted as kernels) are not followed.
A conformance-kit plugin whose modules are linted as kernels (``kernel=True``) marks its host-side
builders with the same structured :data:`AJ020_HOST_MARKER`.

The ids AJ013-AJ019 are used by the contract checks of :mod:`agrijax.iface.contract` (which also
reuses the id AJ012 for a check of its own, independent of lint rule AJ012); lint rules start again
at AJ020.

Usage::

    python -m agrijax.core.lint src/agrijax/processes [--all] [--strict] [--strict-aj007]
                                                      [--strict-aj011] [--ignore AJ007] [--aj007-report]
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

if not __package__:  # pragma: no cover - run as a file by the pre-commit hook
    # ``python src/agrijax/core/lint.py`` in the bare hook interpreter: bind ``agrijax`` and
    # ``agrijax.core`` to plain packages so the rule modules load without running
    # ``agrijax/__init__.py``, which imports JAX. The stubs are removed again once the rule modules
    # are loaded (below), so ``importlib.util.find_spec("agrijax")`` (AJ011's iface lookup) still
    # finds the installed package, as it did before the split
    import types

    _STUBS: list[str] = []
    for _name, _dir in (
        ("agrijax", Path(__file__).resolve().parents[1]),
        ("agrijax.core", Path(__file__).resolve().parent),
    ):
        if _name not in sys.modules:
            _pkg = types.ModuleType(_name)
            _pkg.__path__ = [str(_dir)]
            sys.modules[_name] = _pkg
            _STUBS.append(_name)

from agrijax.core._lint.ast_facts import (
    _call_name,
    _FileFacts,
    _is_process_decorated,
    _says_host_side,
    numpy_names,
)
from agrijax.core._lint.file_rules import (
    _check_aj008,
    _check_aj009,
    _check_aj010,
    _check_aj011,
    _check_aj012,
    _in_dirs,
    port_field_names,
)
from agrijax.core._lint.function_checker import _FunctionChecker
from agrijax.core._lint.imports import import_closure, import_slot, module_imports, slot_of_path
from agrijax.core._lint.rules import (
    _HOST_DIRS,
    _KERNEL_DIRS,
    _PROCESS_DECORATOR,
    AJ007_MAX_EXPONENT,
    AJ007_MAX_INDEX,
    AJ007_STRUCTURAL_CALLS,
    AJ007_STRUCTURAL_KEYWORDS,
    AJ007_TRIVIAL,
    AJ020_HOST_MARKER,
    AJ020_STATIC_CALLS,
    ALL_RULES,
    FILE_RULES,
    HOST_RULES,
    KERNEL_RULES,
    NOT_STRICT_RULES,
    RULES,
    Finding,
)

if not __package__:  # pragma: no cover - drop the run-as-a-file stubs again (see above)
    for _name in _STUBS:  # pyright: ignore[reportPossiblyUnbound]
        sys.modules.pop(_name, None)

__all__ = [
    "AJ007_MAX_EXPONENT",
    "AJ007_MAX_INDEX",
    "AJ007_STRUCTURAL_CALLS",
    "AJ007_STRUCTURAL_KEYWORDS",
    "AJ007_TRIVIAL",
    "AJ020_HOST_MARKER",
    "AJ020_STATIC_CALLS",
    "ALL_RULES",
    "FILE_RULES",
    "HOST_RULES",
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
    "numpy_names",
    "port_field_names",
    "slot_of_path",
]

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
    return any(d in _KERNEL_DIRS for d in Path(path).parts[:-1])


def _in_host_dir(path: str) -> bool:
    return any(d in _HOST_DIRS for d in Path(path).parts[:-1])


def _select(
    tree: ast.AST,
    path: str,
    all_functions: bool,
    kernel: bool | None = None,
    processes: frozenset[str] = frozenset(),
) -> Iterator[tuple[_FnNode, bool, frozenset[str]]]:
    kernel_file = _in_kernel_dir(path) if kernel is None else kernel
    host_file = kernel is None and not kernel_file and _in_host_dir(path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_proc = _is_process_decorated(node) or node.name in processes
            if all_functions or is_proc:
                yield node, is_proc, ALL_RULES
            elif kernel_file:
                yield node, is_proc, KERNEL_RULES
            elif host_file:
                yield node, is_proc, HOST_RULES


#: directories where a ``process(fn, ...)`` call makes ``fn`` a process for the lint
_PROCESS_CALL_DIRS: frozenset[str] = frozenset({"processes", "forcing", "models"})


def _called_processes(tree: ast.AST, path: str) -> frozenset[str]:
    """Names passed as the first argument of ``process(<name>, ...)`` in a file of
    :data:`_PROCESS_CALL_DIRS` (assembly entries built without the decorator)."""
    if not _in_dirs(path, _PROCESS_CALL_DIRS):
        return frozenset()
    return frozenset(
        n.args[0].id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and _call_name(n) == _PROCESS_DECORATOR
        and n.args
        and isinstance(n.args[0], ast.Name)
    )


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
    ``agrijax/iface/``; AJ012 on those of the layers it names.
    """
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:  # report as a finding instead of crashing pre-commit
        return [Finding("AJ000", "error", path, e.lineno or 0, e.offset or 0, f"syntax error: {e.msg}")]
    skip = frozenset(ignore)
    findings: list[Finding] = []
    procs = frozenset(processes) | _called_processes(tree, path)
    facts = _FileFacts.of(tree, path)

    def is_host(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Host-side: marked, a callback, in a host module, or nested in a host function."""
        while fn is not None:
            if _is_process_decorated(fn) or fn.name in procs:
                return False
            if facts.host_module or fn.name in facts.callbacks or _says_host_side(fn):
                return True
            fn = facts.enclosing.get(id(fn))  # type: ignore[assignment]
        return False

    for node, is_proc, rules in _select(tree, path, all_functions, kernel, procs):
        checker = _FunctionChecker(
            node, path, rules - skip, is_process=is_proc, host=not is_proc and is_host(node), facts=facts
        )
        findings.extend(checker.run())
    if "AJ008" not in skip:
        findings.extend(_check_aj008(tree, path))
    if "AJ009" not in skip:
        findings.extend(_check_aj009(tree, path))
    if "AJ010" not in skip:
        findings.extend(_check_aj010(tree, path))
    if "AJ011" not in skip:
        findings.extend(_check_aj011(tree, path))
    if "AJ012" not in skip:
        findings.extend(_check_aj012(tree, path))
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
        procs = _called_processes(tree, str(f))
        for node, is_proc, rules in _select(tree, str(f), all_functions, processes=procs):
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
        "--strict-aj011", action="store_true", help="treat AJ011 (forcing that shadows a port) as an error"
    )
    ap.add_argument(
        "--enforce", action="append", default=[], metavar="RULE", help="treat RULE as an error (repeatable)"
    )
    ap.add_argument(
        "--ignore", action="append", default=[], metavar="RULE", help="do not run RULE (repeatable)"
    )
    ap.add_argument("--aj007-report", action="store_true", help="print the AJ007 count per file")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    ns = ap.parse_args(argv)
    unknown = (set(ns.ignore) | set(ns.enforce)) - set(RULES)
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
    enforced = {*ns.enforce, *(("AJ011",) if ns.strict_aj011 else ())}
    n_enforced = sum(1 for f in findings if f.rule in enforced)
    n_strict = sum(1 for f in findings if f.level == "warning" and f.rule not in NOT_STRICT_RULES)
    n_proc = sum(1 for c in visited if c.is_process)
    if ns.aj007_report:
        for path, n in count_by_file(findings, "AJ007").items():
            print(f"AJ007 {n:5d}  {path}")
    print(
        f"agrijax lint: {n_err} error(s), {n_warn} warning(s) ({n_aj007} AJ007) in {len(visited)} "
        f"function(s) ({n_proc} @process, {len(visited) - n_proc} kernel)"
    )
    if n_err or (ns.strict and n_strict) or (ns.strict_aj007 and n_aj007) or n_enforced:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
