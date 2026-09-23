"""Base classes for the State / Params / Forcing pytrees and field metadata.

All three are :class:`equinox.Module` subclasses: immutable dataclasses that are
also JAX pytrees. Fields carry metadata (``unit``, ``description``,
``fortran_name``) through :func:`field`, from which ``io.schema`` builds the
variable and parameter tables of the documentation and the file mapping.

Conventions (docs/en/02_architecture.md section 3.1):

* ``State`` holds everything that is carried from one day to the next.
* ``Params`` holds everything that is constant over a run (may be batched).
* ``Forcing`` holds time-varying inputs with the time axis first ``[T, ...]``;
  ``lax.scan`` slices it into a per-day forcing of the same class.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

import equinox as eqx
import jax.tree_util as jtu
from jax.core import Tracer

__all__ = [
    "Forcing",
    "Params",
    "State",
    "field",
    "field_metadata",
    "get_path",
    "leaf_paths",
    "set_path",
    "tree_diff",
]

_T = TypeVar("_T", bound="_Base")


def field(
    *,
    unit: str = "",
    description: str = "",
    fortran_name: str = "",
    dims: Sequence[str] | str | None = None,
    static: bool = False,
    converter: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Declare a pytree field with physical metadata.

    Parameters
    ----------
    unit:
        Physical unit, using the internal convention (cm, degC, MJ m-2 d-1, kg ha-1).
    description:
        One-line human-readable meaning.
    fortran_name:
        Name of the corresponding variable in the Fortran oracle (RZWQM or DSSAT), if any.
    dims:
        Symbolic dimension names, e.g. ``("n_crop", "n_node")``; documentation only.
    static:
        Passed through to :func:`equinox.field`; a static field is not a pytree leaf.
    converter:
        Passed through to :func:`equinox.field`.
    **kwargs:
        Any other :func:`dataclasses.field` keyword (``default``, ``default_factory``).
    """
    if isinstance(dims, str):
        dims = (dims,)
    metadata = {
        "unit": unit,
        "description": description,
        "fortran_name": fortran_name,
        "dims": tuple(dims) if dims is not None else None,
    }
    if converter is not None:
        kwargs["converter"] = converter
    return eqx.field(static=static, metadata=metadata, **kwargs)


def field_metadata(cls: type, *, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Return ``{dotted.path: metadata}`` for every dataclass field of ``cls``, recursing into nested modules.

    Fields declared without :func:`field` get empty metadata so that the table is complete.
    """
    out: dict[str, dict[str, Any]] = {}
    for f in dataclasses.fields(cls):
        path = f"{prefix}{f.name}"
        meta = {
            "unit": f.metadata.get("unit", ""),
            "description": f.metadata.get("description", ""),
            "fortran_name": f.metadata.get("fortran_name", ""),
            "dims": f.metadata.get("dims"),
            "static": bool(f.metadata.get("static", False)),
            "type": f.type,
        }
        out[path] = meta
        ftype = f.type
        if isinstance(ftype, type) and dataclasses.is_dataclass(ftype):
            out.update(field_metadata(ftype, prefix=path + "."))
    return out


class _Base(eqx.Module):
    """Common behaviour of State / Params / Forcing."""

    @classmethod
    def field_metadata(cls) -> dict[str, dict[str, Any]]:
        """Metadata table of this class, keyed by dotted field path."""
        return field_metadata(cls)

    def replace(self: _T, **changes: Any) -> _T:
        """Return a copy with the given top-level fields replaced (functional update)."""
        return dataclasses.replace(self, **changes)

    def leaf_paths(self) -> list[str]:
        """Dotted paths of every array leaf, in pytree order."""
        return leaf_paths(self)

    def get(self, path: str) -> Any:
        """Fetch a leaf or sub-tree by dotted path, e.g. ``state.get("crop.lai")``."""
        return get_path(self, path)

    def set(self: _T, path: str, value: Any) -> _T:
        """Functionally replace a leaf or sub-tree by dotted path (``eqx.tree_at`` under the hood)."""
        return set_path(self, path, value)

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate ``(dotted path, leaf)`` pairs."""
        for kp, leaf in jtu.tree_leaves_with_path(self):
            yield _keystr(kp), leaf


class State(_Base):
    """Base class for the model state carried from day to day.

    Subclasses declare fields with :func:`field`; nested modules group them
    (``state.soil.theta``, ``state.crop.lai``). Every crop field carries a
    leading ``n_crop`` axis.
    """


class Params(_Base):
    """Base class for run-constant parameters (the thing that gets batched and differentiated)."""


class Forcing(_Base):
    """Base class for time-varying inputs. Every leaf has the time axis first ``[T, ...]``.

    The runtime scans over the leading axis, handing each process a ``Forcing``
    of the same class whose leaves are one day's values.
    """

    @property
    def n_days(self) -> int:
        """Length of the leading time axis (taken from the first leaf)."""
        leaves = jtu.tree_leaves(self)
        if not leaves:
            raise ValueError("Forcing has no array leaves")
        return int(leaves[0].shape[0])


# ---------------------------------------------------------------------------
# path utilities
# ---------------------------------------------------------------------------


def _keystr(key_path: Sequence[Any]) -> str:
    parts: list[str] = []
    for k in key_path:
        if isinstance(k, jtu.GetAttrKey):
            parts.append(k.name)
        elif isinstance(k, jtu.DictKey):
            parts.append(str(k.key))
        elif isinstance(k, jtu.SequenceKey):
            parts.append(str(k.idx))
        elif isinstance(k, jtu.FlattenedIndexKey):
            parts.append(str(k.key))
        else:  # pragma: no cover - unknown key type
            parts.append(str(k))
    return ".".join(parts)


def leaf_paths(tree: Any) -> list[str]:
    """Dotted path of every leaf of ``tree`` in flatten order."""
    return [_keystr(kp) for kp, _ in jtu.tree_leaves_with_path(tree)]


def get_path(tree: Any, path: str) -> Any:
    """Fetch ``tree.a.b.c`` (or ``tree["a"]["b"]`` for dicts, ``tree[0]`` for sequences) by dotted path."""
    node = tree
    for part in path.split(".") if path else []:
        if isinstance(node, dict):
            node = node[part]
        elif isinstance(node, (list, tuple)):
            node = node[int(part)]
        else:
            node = getattr(node, part)
    return node


def set_path(tree: _T, path: str, value: Any) -> _T:
    """Return ``tree`` with the node at ``path`` replaced by ``value`` (functional, via ``eqx.tree_at``)."""
    return eqx.tree_at(lambda t: get_path(t, path), tree, value)


def tree_diff(before: Any, after: Any) -> list[str]:
    """Dotted paths of the leaves that differ between two pytrees of the same structure.

    Concrete leaves are compared by value; traced leaves (inside ``jit``/``scan``)
    by object identity, which is exact for processes that update with
    ``eqx.tree_at`` and conservative otherwise.
    """
    before_leaves = jtu.tree_leaves_with_path(before)
    after_leaves = jtu.tree_leaves_with_path(after)
    if jtu.tree_structure(before) != jtu.tree_structure(after):
        raise ValueError(
            "pytree structure changed:\n"
            f"  before: {jtu.tree_structure(before)}\n  after:  {jtu.tree_structure(after)}"
        )
    changed: list[str] = []
    for (kp, a), (_, b) in zip(before_leaves, after_leaves):
        if a is b:
            continue
        if isinstance(a, Tracer) or isinstance(b, Tracer):
            changed.append(_keystr(kp))
            continue
        try:
            import numpy as np

            same = np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) and (
                np.shape(a) == np.shape(b)
            )
        except (TypeError, ValueError):
            same = a == b
        if not same:
            changed.append(_keystr(kp))
    return changed
