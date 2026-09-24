"""Named array axes (the ``Dims`` registry) and the shape check of State / Params / Forcing pytrees.

Every array field declares its axes with :func:`agrijax.core.state.field` ``dims=...``: a tuple
of axis names from :data:`DIMS` (``("n_crop", "n_layer")``), ``()`` for a scalar. A dims entry is

* a registered name, e.g. ``n_node``; its size is bound by the first leaf that uses it and every
  other leaf of the same tree must agree;
* a name with an offset, ``n_node-1`` (the ``n_node - 1`` faces between nodes);
* an integer literal, ``"4"`` (a fixed-size table);
* any of these with a ``?`` suffix: the axis may be absent (a scalar in single-horizon use).
  Optional axes must come first.

Declared dims describe the **trailing** axes of a leaf; extra leading axes are batch axes and are
ignored (the runtime vmaps, so the shapes it checks are per sample anyway). The check is a Python
walk over the dataclass fields and their metadata, done at trace time by
:func:`agrijax.core.runtime.run`: it reads ``.shape`` only and adds no operation to the traced
program. :func:`check_tree_dims` raises :class:`DimsError` naming the field path.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "DIMS",
    "DIM_ALIASES",
    "DimSpec",
    "DimsError",
    "check_tree_dims",
    "parse_dim",
    "parse_dims",
    "register_dim",
]


class _Dim(NamedTuple):
    description: str
    size: int | None  # fixed size, or None when bound per tree


#: The registry of named axes: ``{name: (description, fixed size or None)}``.
DIMS: dict[str, _Dim] = {
    "T": _Dim("time axis of a forcing: the days of the run", None),
    "hour": _Dim("hours of one day", 24),
    "n_crop": _Dim("crop slots (every crop field carries it first)", None),
    "n_cohort": _Dim("organ cohorts of a crop's organ queue (rank-indexed)", None),
    "n_node": _Dim("nodes of the Richards soil-water grid", None),
    "n_layer": _Dim("soil layers of the crop model (DSSAT layers)", None),
    "n_horizon": _Dim("soil horizons of the hydraulic parameters", None),
    "n_pool": _Dim("carbon / nitrogen pools of a pool-flow module", None),
}

#: Alternative spellings mapped onto a registered name.
DIM_ALIASES: dict[str, str] = {"n_day": "T"}


def register_dim(name: str, description: str, size: int | None = None) -> None:
    """Add a named axis to :data:`DIMS` (a module introducing a new axis calls this at import)."""
    if not re.fullmatch(r"[A-Za-z_]\w*", name):
        raise ValueError(f"not a valid axis name: {name!r}")
    old = DIMS.get(name)
    if old is not None and old.size != size:
        raise ValueError(f"axis {name!r} already registered with size {old.size}, not {size}")
    DIMS[name] = _Dim(description, size)
    _parse_cached.cache_clear()
    _class_fields.cache_clear()


class DimSpec(NamedTuple):
    """One parsed dims entry: axis ``name`` (None for a literal), ``offset`` or literal size, optional."""

    name: str | None
    offset: int
    optional: bool

    def describe(self) -> str:
        base = str(self.offset) if self.name is None else self.name
        if self.name is not None and self.offset:
            base += f"{self.offset:+d}"
        return base + ("?" if self.optional else "")


class DimsError(ValueError):
    """A dims declaration that does not parse, or a leaf shape that does not match it."""


_DIM_RE = re.compile(r"^\s*(?:(?P<lit>\d+)|(?P<name>[A-Za-z_]\w*)\s*(?P<off>[+-]\s*\d+)?)\s*(?P<opt>\?)?\s*$")


@functools.cache
def _parse_cached(token: str) -> DimSpec:
    m = _DIM_RE.match(token)
    if m is None:
        raise DimsError(f"cannot parse dims entry {token!r}")
    optional = m.group("opt") is not None
    if m.group("lit") is not None:
        return DimSpec(None, int(m.group("lit")), optional)
    name = DIM_ALIASES.get(m.group("name"), m.group("name"))
    if name not in DIMS:
        raise DimsError(f"unknown axis {m.group('name')!r} (registered: {sorted(DIMS)})")
    off = int(m.group("off").replace(" ", "")) if m.group("off") else 0
    fixed = DIMS[name].size
    if fixed is not None:  # a fixed-size name is a literal
        return DimSpec(None, fixed + off, optional)
    return DimSpec(name, off, optional)


def parse_dim(token: str | int) -> DimSpec:
    """Parse one dims entry (see the module docstring); raises :class:`DimsError`."""
    return _parse_cached(str(token))


def parse_dims(dims: Any) -> tuple[DimSpec, ...] | None:
    """Parse a dims tuple; optional axes must lead. ``None`` (undeclared) returns ``None``."""
    if dims is None:
        return None
    if isinstance(dims, (str, int)):
        dims = (dims,)
    specs = tuple(parse_dim(d) for d in dims)
    seen_required = False
    for s in specs:
        if s.optional and seen_required:
            raise DimsError(f"optional axes must come first: {tuple(dims)!r}")
        seen_required |= not s.optional
    return specs


# ------------------------------------------------------------------------------------ the check
_FieldInfo = tuple[str, bool, "tuple[DimSpec, ...] | None", "tuple[str, ...] | None"]


@functools.cache
def _class_fields(cls: type) -> tuple[_FieldInfo, ...]:
    """Per dataclass: ``(name, static, parsed dims or None, declared dims)`` for every field."""
    out: list[_FieldInfo] = []
    for f in dataclasses.fields(cls):
        dims = f.metadata.get("dims")
        declared = None if dims is None else tuple(str(d) for d in dims)
        static = bool(f.metadata.get("static", False))
        out.append((f.name, static, parse_dims(declared), declared))
    return tuple(out)


def _match(
    path: str,
    declared: tuple[str, ...],
    specs: tuple[DimSpec, ...],
    shape: tuple[int, ...],
    bind: dict[str, tuple[int, str]],
) -> None:
    n_req = sum(not s.optional for s in specs)
    if len(shape) < n_req:
        raise DimsError(f"{path}: declared dims {declared}, got shape {shape} (too few axes)")
    use = specs[max(0, len(specs) - len(shape)) :]
    trailing = shape[len(shape) - len(use) :] if use else ()
    for spec, size in zip(use, trailing, strict=True):
        if spec.name is None:
            if size != spec.offset:
                raise DimsError(
                    f"{path}: declared dims {declared}, got shape {shape} "
                    f"(axis {spec.describe()} must be {spec.offset})"
                )
            continue
        n = size - spec.offset
        prev = bind.get(spec.name)
        if prev is None:
            bind[spec.name] = (n, path)
        elif prev[0] != n:
            raise DimsError(
                f"{path}: declared dims {declared}, got shape {shape}: {spec.name} = {n}, "
                f"but {prev[1]} has {spec.name} = {prev[0]}"
            )


def _is_instance_dataclass(x: Any) -> bool:
    return dataclasses.is_dataclass(x) and not isinstance(x, type)


def _walk(node: Any, path: str, bind: dict[str, tuple[int, str]]) -> None:
    if node is None:
        return
    if _is_instance_dataclass(node):
        prefix = path + "." if path else ""
        for name, static, specs, declared in _class_fields(type(node)):
            if static:
                continue
            value = getattr(node, name, None)
            if value is None:
                continue
            if specs is None or declared is None or _is_instance_dataclass(value):
                _walk(value, prefix + name, bind)
                continue
            _match(prefix + name, declared, specs, tuple(np.shape(value)), bind)
        return
    if isinstance(node, Mapping):
        for k, v in node.items():
            _walk(v, f"{path}.{k}" if path else str(k), bind)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _walk(v, f"{path}.{i}" if path else str(i), bind)


def check_tree_dims(tree: Any, *, root: str = "") -> dict[str, int]:
    """Check every declared-dims leaf of ``tree`` (State / Params / Forcing, nested, or dicts of them).

    Sizes of named axes are bound across the whole tree, so passing ``{"state": s, "params": p}``
    checks that the state and the parameters agree on ``n_layer``. Works on concrete arrays and on
    tracers (only ``.shape`` is read). Returns ``{axis name: size}``; raises :class:`DimsError`.
    Fields without declared dims are skipped.
    """
    bind: dict[str, tuple[int, str]] = {}
    _walk(tree, root, bind)
    return {k: v[0] for k, v in bind.items()}
