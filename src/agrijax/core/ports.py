"""Ports and bindings: how a module's state is placed in an assembled model (plan 19 A2; decision Q3).

A module's ``State`` declares the records it exchanges with other modules as **port fields**
(:func:`port`). An interface record is an ordinary ``State`` of arrays, so the three process
rules and ``AGRI_JAX_CHECK`` hold unchanged::

    class CropWaterIn(State):                       # interface record
        eop: Array = field(unit="mm d-1", dims="n_crop")
        trwup: Array = field(unit="cm d-1", dims="n_crop")

    class CropState(State):
        growth: Growth
        water_in: CropWaterIn = port()              # bound at assembly

    stress = bind(crop_stress, own="crops.maize", ports={"water_in": "iface.crop_water.maize"})

In the global (assembled) state the module's own subtree sits at ``own`` with its port fields
set to ``None``; each port's record sits at its bound global path. :func:`bind` wraps the
unchanged process: it gathers the module's view (own subtree plus each bound port), calls the
process on it, and scatters back **only the declared writes**. Its ``reads``/``writes`` are
rewritten onto the global paths, so ``AGRI_JAX_CHECK``, ``Model.dataflow()`` and
``Model.stale_reads()`` run on bound paths. Under ``AGRI_JAX_CHECK=1`` the wrapped process checks
its writes on the view (so a port not declared as a write must come back unchanged) and the
bound process checks them again on the global state. All of this is Python tree plumbing done
at trace time and erased by ``jit``, except that the bound port records pass through
``lax.optimization_barrier``: a port is an opaque value to the module, so XLA cannot fold the
producer's arithmetic into the consumer's, and a module computes bit for bit the same whether its
input record was produced in the same day step (coupled) or read from data (replay).

Replay and coupled runs are two bindings of the same code: in a replay run the port's global
record is written by a replay process from forcing; in a coupled run by the producing module.

There are no callables in state and no closures over params. Anything a consumer needs from
another module's parameters is published by that module into a record.

The top-level names of an assembled state are fixed (:data:`NAMESPACES`), plus names reserved for
later slots (:data:`RESERVED_NAMESPACES`).
"""

from __future__ import annotations

import dataclasses
import functools
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from jax import lax

from agrijax.core.model import Model
from agrijax.core.process import WILDCARD, Process
from agrijax.core.state import field, get_path, set_path

__all__ = [
    "NAMESPACES",
    "RESERVED_NAMESPACES",
    "Binding",
    "BindingError",
    "bind",
    "compose",
    "detach",
    "port",
    "port_names",
]

#: top-level names of an assembled state (plan 19 A2). ``water_supply`` and ``n_supply`` hold the
#: per-crop producers of P1 ``trwup`` (ROOTWU) and P10 (crop nitrogen) at ``<namespace>.<slot>``:
#: outside ``crops.<slot>``, so their state does not overlap the crop's subtree and their day entries
#: (``water_supply.<slot>.rootwu``, ``n_supply.<slot>.replay``) are modules of their own, which makes
#: ROOTWU's read of yesterday's root record a lag the day checks (M3 contract section 11, items 1-2).
NAMESPACES: tuple[str, ...] = ("soil_water", "crops", "water_supply", "n_supply", "iface", "ledger", "prev")
#: reserved for later slots, accepted but unused in M3
RESERVED_NAMESPACES: tuple[str, ...] = ("surface", "soil_heat", "solutes", "soil_om")


class BindingError(ValueError):
    """A binding that does not fit the module state or the global state."""


def port(*, description: str = "", default: Any = None, **kwargs: Any) -> Any:
    """Declare a port field of a module state: an interface record bound to a global path at
    assembly (:func:`bind`). Defaults to ``None`` (the value in the global state, where the
    record lives at its bound path), so declare ports after the module's own fields.

    Accepts every :func:`agrijax.core.state.field` keyword.
    """
    return field(description=description, port=True, default=default, **kwargs)


@functools.cache
def port_names(cls: type) -> tuple[str, ...]:
    """Names of the top-level port fields of the dataclass ``cls``."""
    if not dataclasses.is_dataclass(cls):
        return ()
    return tuple(f.name for f in dataclasses.fields(cls) if f.metadata.get("port", False))


@functools.cache
def _port_types(cls: type) -> dict[str, type | None]:
    """Declared class of each port field, when the annotation resolves to a class."""
    try:
        hints = typing.get_type_hints(cls)
    except Exception:  # unresolvable forward references: no type check
        hints = {}
    out: dict[str, type | None] = {}
    for name in port_names(cls):
        h = hints.get(name)
        out[name] = h if isinstance(h, type) else None
    return out


def detach(module_state: Any) -> Any:
    """``module_state`` with every port field set to ``None`` (its form in the global state)."""
    names = port_names(type(module_state))
    if not names:
        return module_state
    return dataclasses.replace(module_state, **{n: None for n in names})


def _head(path: str) -> str:
    return path.split(".", 1)[0]


def _check_namespace(path: str, what: str) -> None:
    head = _head(path)
    if head not in NAMESPACES and head not in RESERVED_NAMESPACES:
        raise BindingError(
            f"{what} {path!r}: top-level name {head!r} is not a namespace "
            f"(namespaces {NAMESPACES}, reserved {RESERVED_NAMESPACES})"
        )
    if path == head and what != "own":
        raise BindingError(f"{what} {path!r} must be below a namespace, e.g. 'iface.<record>.<slot>'")


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


@dataclass(frozen=True)
class Binding:
    """Where one module lives in the global state: its own subtree and the targets of its ports.

    ``params`` and ``forcing`` optionally select the module's parameter and forcing subtrees of
    the global ``params`` / ``forcing`` (``None`` passes them whole).
    """

    own: str
    ports: tuple[tuple[str, str], ...] = ()
    params: str | None = None
    forcing: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports", tuple((str(p), str(t)) for p, t in self.ports))
        if not self.own or "*" in self.own:
            raise BindingError(f"invalid own path {self.own!r}")
        _check_namespace(self.own, "own")
        names = [p for p, _ in self.ports]
        if len(set(names)) != len(names):
            raise BindingError(f"a port is bound twice: {names}")
        for p, t in self.ports:
            if not p.isidentifier():
                raise BindingError(f"port name {p!r} is not a field name")
            _check_namespace(t, f"target of port {p!r}")
            if _overlaps(t, self.own):
                raise BindingError(f"port {p!r} target {t!r} overlaps the module's own subtree {self.own!r}")
        targets = [t for _, t in self.ports]
        for i, a in enumerate(targets):
            for b in targets[i + 1 :]:
                if _overlaps(a, b):
                    raise BindingError(f"port targets {a!r} and {b!r} overlap")

    @property
    def port_map(self) -> dict[str, str]:
        return dict(self.ports)

    # ------------------------------------------------------------------ paths
    def to_global(self, path: str, module_cls: type | None = None) -> str:
        """Global path of a module-relative ``path`` (``"water_in.eop"`` -> the port's target)."""
        if path == WILDCARD:
            return WILDCARD
        head, _, rest = path.partition(".")
        target = self.port_map.get(head)
        if target is not None:
            return target + ("." + rest if rest else "")
        if module_cls is not None and head in port_names(module_cls):
            raise BindingError(
                f"{path!r} refers to port {head!r} of {module_cls.__name__}, which is not bound"
            )
        return f"{self.own}.{path}"

    def global_paths(self, paths: Iterable[str]) -> tuple[str, ...]:
        """:meth:`to_global` of each path; ``*`` becomes the own subtree plus every port target."""
        out: list[str] = []
        for p in paths:
            if p == WILDCARD:
                out.extend([self.own, *(t for _, t in self.ports)])
            else:
                out.append(self.to_global(p))
        return tuple(dict.fromkeys(out))

    # ------------------------------------------------------------------ tree plumbing
    def gather(self, tree: Any) -> Any:
        """The module's view of the global ``tree``: its own subtree with each bound port filled."""
        own_sub = get_path(tree, self.own)
        cls = type(own_sub)
        names = port_names(cls)
        if not self.ports:
            return own_sub
        types = _port_types(cls)
        values: dict[str, Any] = {}
        for p, target in self.ports:
            if p not in names:
                raise BindingError(f"{cls.__name__} has no port field {p!r} (ports: {list(names)})")
            if getattr(own_sub, p) is not None:
                raise BindingError(
                    f"{self.own}.{p} holds a value in the global state; a bound port lives only at its "
                    f"target {target!r} (place the module with Binding.entries / detach)"
                )
            value = get_path(tree, target)
            want = types.get(p)
            if want is not None and not isinstance(value, want):
                raise BindingError(
                    f"port {p!r} of {cls.__name__} expects {want.__name__}, but {target!r} holds "
                    f"{type(value).__name__}"
                )
            values[p] = value
        # a port record is an opaque value to the module: without the barrier XLA may fuse the
        # producer's arithmetic with the consumer's and fold constants across the boundary
        # ((x * 10) * 0.1), which rounds differently from the same record replayed from data
        values = lax.optimization_barrier(values)
        return dataclasses.replace(own_sub, **values)

    def scatter(self, tree: Any, view: Any, writes: Iterable[str]) -> Any:
        """``tree`` with the module-relative ``writes`` of ``view`` copied to their global paths."""
        cls = type(view)
        for w in writes:
            if w == WILDCARD:
                tree = set_path(tree, self.own, detach(view))
                for p, target in self.ports:
                    tree = set_path(tree, target, getattr(view, p))
                continue
            tree = set_path(tree, self.to_global(w, cls), get_path(view, w))
        return tree

    def entries(self, module_state: Any) -> dict[str, Any]:
        """``{global path: value}`` placing a full module state: the detached own subtree at
        ``own`` and each bound port's record at its target (feed to :func:`compose`)."""
        out = {self.own: detach(module_state)}
        for p, target in self.ports:
            out[target] = getattr(module_state, p)
        return out


def bind(
    proc: Process | Callable[..., Any],
    *,
    own: str,
    ports: Mapping[str, str] | None = None,
    params: str | None = None,
    forcing: str | None = None,
    name: str | None = None,
) -> Process:
    """Bind a module process to global paths (see the module docstring).

    Parameters
    ----------
    proc:
        The module's process, written against the module state (reads/writes relative to it).
    own:
        Global path of the module's own subtree, e.g. ``"crops.maize"``.
    ports:
        ``{port field: global path}``, e.g. ``{"water_in": "iface.crop_water.maize"}``. A
        declared read or write of an unbound port raises :class:`BindingError` when the bound
        process is first traced.
    params, forcing:
        Optional paths of the module's subtrees in the global params / forcing.
    name:
        Name of the bound process; defaults to ``"<own>.<process name>"`` (a day entry name).
    """
    inner = Model._as_process(proc)
    binding = Binding(own=own, ports=tuple((ports or {}).items()), params=params, forcing=forcing)
    module_writes = inner.writes

    def bound(state: Any, params_: Any, forcing_t: Any) -> Any:
        view = binding.gather(state)
        for p in (*inner.reads, *inner.writes):
            binding.to_global(p, type(view))  # raises for an unbound port of the module class
        p_sub = params_ if binding.params is None else get_path(params_, binding.params)
        f_sub = forcing_t if binding.forcing is None else get_path(forcing_t, binding.forcing)
        new_view = inner(view, p_sub, f_sub)  # checks the module-relative writes under AGRI_JAX_CHECK
        return binding.scatter(state, new_view, module_writes)

    bound.__name__ = f"bound_{inner.name}"
    bound.__qualname__ = bound.__name__
    return Process(
        fn=bound,
        name=name or f"{own}.{inner.name}",
        reads=binding.global_paths(inner.reads),
        writes=binding.global_paths(inner.writes),
        source=inner.source,
        fortran_name=inner.fortran_name,
        doc=inner.doc,
        info=inner.info,
    )


def compose(entries: Mapping[str, Any]) -> dict[str, Any]:
    """Build a global state (nested dicts) from ``{dotted path: subtree}``.

    Every path must start with a namespace (:data:`NAMESPACES` or :data:`RESERVED_NAMESPACES`)
    and no two paths may overlap. Nested dicts are ordinary pytrees: ``get_path``/``set_path``,
    ``tree_diff``, the write check and the dims check all work on them.
    """
    paths = list(entries)
    for p in paths:
        if not p or p.startswith(".") or p.endswith(".") or ".." in p:
            raise BindingError(f"invalid path {p!r}")
        _check_namespace(p, "own")
    for i, a in enumerate(paths):
        for b in paths[i + 1 :]:
            if _overlaps(a, b):
                raise BindingError(f"paths {a!r} and {b!r} overlap")
    root: dict[str, Any] = {}
    for p, value in entries.items():
        parts = p.split(".")
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):  # pragma: no cover - excluded by the overlap check
                raise BindingError(f"{p!r} passes through a non-dict node")
        node[parts[-1]] = value
    return root
