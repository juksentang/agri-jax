"""The ``@process`` decorator, the process registry and the writes check.

A process is a pure function ``(state, params, forcing_t) -> state`` that obeys
the three rules (README section "Design in three rules"):

1. Everything read is in the arguments; everything changed is in the return value.
2. Branch with ``jnp.where`` / ``jnp.select``, never with Python ``if`` on state.
3. Never write a loop; time and samples are handled by the runtime.

The decorator records the declared ``reads`` and ``writes`` (dotted state paths),
registers the process, and, when the environment variable ``AGRI_JAX_CHECK=1``
is set, verifies after every call that the returned pytree differs from the
input only in the declared ``writes``.

Registry keys, variants and provenance
--------------------------------------
A process of the model library is registered under a versioned key
``slot/impl@ref_version:variant`` (:class:`ProcessKey`), for example
``soil_water/richards@rzwqm2-4.6:faithful`` or
``crop/ceres_maize.phenology@dssat-4.8.6.0:faithful``:

* ``slot`` is the role the process fills in an assembly (``soil_water``, ``pet``, ``crop``);
* ``impl`` names the implementation, dotted when one module contributes several processes
  (``ceres_maize.phenology``, ``ceres_maize.growth``);
* ``ref_version`` is the reference the implementation follows (a model build such as
  ``dssat-4.8.6.0`` or ``rzwqm2-4.6``, a standard such as ``asce-ewri-2005``), or ``none`` when
  there is no reference (bookkeeping, demonstrations);
* ``variant`` is ``faithful`` for the version pinned to the reference; any other variant reuses
  the faithful kernels and must list how it deviates.

A variant other than ``faithful`` of a key with a reference (``ref_version != none``) can only
be registered once its **faithful sibling** ``slot/impl@ref_version:faithful`` is registered:
:meth:`ProcessRegistry.add` raises :class:`MissingFaithfulError` otherwise, and removing a
faithful entry that still has variants raises too (M3 coupling contract, decision 7). The rule
binds every registration, including plugins, since they register through the same registry;
keys with ``ref_version = none`` are exempt (they have no reference to be faithful to). Define a
module's faithful process before its variants.

Selection between variants is static (Python-level, by key), never a traced flag. Numerics
settings (sub-step and iteration counts) are configuration, not variants.

Each keyed process carries a :class:`ProcessInfo`: the key, the provenance class
(:data:`PROVENANCE`: ``translated_bsd3`` for code translated from BSD-3 reference source,
``equations_only`` for code written from published equations alone,
``reference_only_conventions`` for code written from published equations where the reference
source, which carries no licence, was read for conventions only), the sources per equation, the
grid it runs on (:data:`GRIDS`), the list of known deviations from the reference and optionally
the reference build. The registry raises :class:`DuplicateProcessError` when a second, different
process claims a key or a name that is taken; redefining the same function (a module reload)
replaces the entry.

The registry is also a name-keyed mapping (``registry["richards_redistribution"]``) so the
lookups by function name keep working. :func:`lookup` resolves a key, :func:`list_processes`
filters the entries and :func:`metadata_problems` reports what an entry's metadata lacks.
"""

from __future__ import annotations

import inspect
import os
import re
from collections.abc import Callable, Iterable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agrijax.core.state import tree_diff

__all__ = [
    "CHECK_ENV",
    "FAITHFUL",
    "GRIDS",
    "NO_REFERENCE",
    "PROVENANCE",
    "Deviation",
    "DuplicateProcessError",
    "MissingFaithfulError",
    "Process",
    "ProcessInfo",
    "ProcessKey",
    "ProcessRegistry",
    "ProcessSignatureError",
    "ProcessWriteError",
    "Source",
    "check_enabled",
    "list_processes",
    "lookup",
    "metadata_problems",
    "process",
    "registry",
]

CHECK_ENV = "AGRI_JAX_CHECK"
WILDCARD = "*"

#: provenance classes: how the code of a process was derived
PROVENANCE: dict[str, str] = {
    "translated_bsd3": "translated from reference source code under BSD-3 (notice required)",
    "equations_only": "written from published equations only; the reference model's outputs validate it",
    "reference_only_conventions": (
        "written from published equations; the reference source (no licence file) was read privately for "
        "conventions only, never translated"
    ),
}

#: grids a process can run on (D3 replaces these names by grid objects)
GRIDS: dict[str, str] = {
    "point": "no spatial axis (one value per sample, crop or organ)",
    "rzwqm2_nodes": "RZWQM2 vertex-centred soil node grid (TL, DELZ)",
    "dssat_layers": "DSSAT soil layers (DLAYR), no surface layer 0",
    "rzwqm2_lyrset": (
        "the DSSAT crop layers of RZWQM2's embedded crop (LYRSET: layer bottoms 5, 15, 30, 45, 60, 90, "
        "120, 150 cm at CA-TPA), onto which the node water is mapped"
    ),
}

FAITHFUL = "faithful"
NO_REFERENCE = "none"

_KEY_RE = re.compile(
    r"^(?P<slot>[a-z][a-z0-9_]*)"
    r"/(?P<impl>[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*)"
    r"@(?P<ref>[a-z0-9](?:[a-z0-9.\-]*[a-z0-9])?)"
    r":(?P<variant>[a-z][a-z0-9_]*)$"
)


class ProcessWriteError(AssertionError):
    """A process modified state fields it did not declare in ``writes``."""


class ProcessSignatureError(TypeError):
    """A process does not accept exactly ``(state, params, forcing)``."""


class DuplicateProcessError(ValueError):
    """A different process is already registered under the same key or name."""


class MissingFaithfulError(ValueError):
    """A non-faithful variant without its registered faithful sibling (M3 contract, decision 7)."""


def check_enabled() -> bool:
    """True when ``AGRI_JAX_CHECK`` is ``1``/``true``/``yes`` (read at call time so tests can toggle it)."""
    return os.environ.get(CHECK_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _normalise(paths: str | Iterable[str] | None) -> tuple[str, ...]:
    if paths is None:
        return ()
    if isinstance(paths, str):
        return (paths,)
    out = tuple(str(p) for p in paths)
    for p in out:
        if not p or p.startswith(".") or p.endswith("."):
            raise ValueError(f"invalid state path {p!r}")
    return out


def _covered(leaf: str, declared: Sequence[str]) -> bool:
    """A declared path covers a leaf when it is equal to it or is a dotted prefix of it."""
    for d in declared:
        if d == WILDCARD or leaf == d or leaf.startswith(d + "."):
            return True
    return False


# ---------------------------------------------------------------------------------- metadata


@dataclass(frozen=True)
class ProcessKey:
    """A registry key ``slot/impl@ref_version:variant``."""

    slot: str
    impl: str
    ref_version: str
    variant: str = FAITHFUL

    def __post_init__(self) -> None:
        text = f"{self.slot}/{self.impl}@{self.ref_version}:{self.variant}"
        if not _KEY_RE.match(text):
            raise ValueError(
                f"invalid process key {text!r}: expected slot/impl@ref_version:variant with lower-case "
                "identifiers (impl may be dotted, ref_version may contain '.' and '-')"
            )

    @classmethod
    def parse(cls, text: str | ProcessKey) -> ProcessKey:
        """Parse ``slot/impl@ref_version:variant``; a :class:`ProcessKey` is returned as is."""
        if isinstance(text, ProcessKey):
            return text
        m = _KEY_RE.match(text)
        if m is None:
            raise ValueError(f"invalid process key {text!r}: expected slot/impl@ref_version:variant")
        return cls(m["slot"], m["impl"], m["ref"], m["variant"])

    def __str__(self) -> str:
        return f"{self.slot}/{self.impl}@{self.ref_version}:{self.variant}"

    @property
    def faithful(self) -> ProcessKey:
        """The ``faithful`` key of the same ``slot/impl@ref_version``."""
        return ProcessKey(self.slot, self.impl, self.ref_version, FAITHFUL)

    @property
    def needs_faithful_sibling(self) -> bool:
        """``True`` for a variant other than ``faithful`` of a key with a reference: it may only be
        registered next to its faithful sibling (``ref_version = none`` is exempt)."""
        return self.variant != FAITHFUL and self.ref_version != NO_REFERENCE


@dataclass(frozen=True)
class Source:
    """Where one equation (or block of equations) of a process comes from."""

    what: str
    ref: str


@dataclass(frozen=True)
class Deviation:
    """One known difference from the reference: what differs, why, and the evidence for it."""

    what: str
    why: str
    evidence: str


@dataclass(frozen=True)
class ProcessInfo:
    """Registry metadata of a keyed process (see the module docstring)."""

    key: ProcessKey
    provenance: str
    sources: tuple[Source, ...]
    grid: str
    deviates: tuple[Deviation, ...]
    ref_build: str = ""

    @property
    def slot(self) -> str:
        return self.key.slot

    @property
    def impl(self) -> str:
        return self.key.impl

    @property
    def ref_version(self) -> str:
        return self.key.ref_version

    @property
    def variant(self) -> str:
        return self.key.variant

    def problems(self) -> list[str]:
        """What is missing or inconsistent; empty when the metadata are complete."""
        out: list[str] = []
        if self.provenance not in PROVENANCE:
            out.append(f"provenance {self.provenance!r} not in {sorted(PROVENANCE)}")
        if not self.sources:
            out.append("no sources")
        for s in self.sources:
            if not (s.what.strip() and s.ref.strip()):
                out.append(f"incomplete source {s!r}")
        if self.grid not in GRIDS:
            out.append(f"grid {self.grid!r} not in {sorted(GRIDS)}")
        for d in self.deviates:
            if not (d.what.strip() and d.why.strip() and d.evidence.strip()):
                out.append(f"incomplete deviation {d!r}")
        if self.ref_version == NO_REFERENCE:
            if self.variant == FAITHFUL:
                out.append("a process without reference cannot be the 'faithful' variant")
            if self.provenance == "translated_bsd3":
                out.append("translated_bsd3 needs a reference version")
        elif self.variant != FAITHFUL and not self.deviates:
            out.append(f"variant {self.variant!r} lists no deviations from the faithful version")
        return out

    def as_dict(self) -> dict[str, Any]:
        """Plain-data form (for the validation matrix and result provenance)."""
        return {
            "key": str(self.key),
            "slot": self.slot,
            "impl": self.impl,
            "ref_version": self.ref_version,
            "variant": self.variant,
            "ref_build": self.ref_build,
            "provenance": self.provenance,
            "grid": self.grid,
            "sources": [{"what": s.what, "ref": s.ref} for s in self.sources],
            "deviates": [{"what": d.what, "why": d.why, "evidence": d.evidence} for d in self.deviates],
        }


def _as_source(s: Source | Sequence[str]) -> Source:
    if isinstance(s, Source):
        return s
    if isinstance(s, str) or len(s) != 2:
        raise TypeError(f"a source is Source(what, ref) or a (what, ref) pair, got {s!r}")
    return Source(str(s[0]), str(s[1]))


def _as_deviation(d: Deviation | Sequence[str]) -> Deviation:
    if isinstance(d, Deviation):
        return d
    if isinstance(d, str) or len(d) != 3:
        raise TypeError(f"a deviation is Deviation(what, why, evidence) or a triple, got {d!r}")
    return Deviation(str(d[0]), str(d[1]), str(d[2]))


# ---------------------------------------------------------------------------------- process


@dataclass(frozen=True)
class Process:
    """A registered process: the function plus its declarations.

    Instances are callable with the same signature as the wrapped function.
    """

    fn: Callable[..., Any]
    name: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    source: str = ""
    fortran_name: str = ""
    doc: str = field(default="", repr=False)
    info: ProcessInfo | None = field(default=None, repr=False)

    def __call__(self, state: Any, params: Any, forcing: Any) -> Any:
        new_state = self.fn(state, params, forcing)
        if check_enabled():
            self.check_writes(state, new_state)
        return new_state

    @property
    def key(self) -> str | None:
        """The registry key ``slot/impl@ref_version:variant``, or None for an unkeyed process."""
        return None if self.info is None else str(self.info.key)

    def check_writes(self, before: Any, after: Any) -> list[str]:
        """Raise :class:`ProcessWriteError` if ``after`` differs from ``before`` outside the declared writes.

        Returns the list of leaves that actually changed.
        """
        changed = tree_diff(before, after)
        undeclared = [p for p in changed if not _covered(p, self.writes)]
        if undeclared:
            raise ProcessWriteError(
                f"process {self.name!r} changed undeclared state fields {undeclared}; "
                f"declared writes = {list(self.writes)}"
            )
        return changed

    @property
    def __name__(self) -> str:  # pragma: no cover - trivial
        return self.name

    @property
    def __doc__(self) -> str:  # type: ignore[override]
        return self.doc

    def __get__(self, obj: Any, objtype: Any = None) -> Any:  # pragma: no cover - descriptor protocol
        return self


def metadata_problems(proc: Process) -> list[str]:
    """What the registry metadata of ``proc`` lack; empty when they are complete."""
    if proc.info is None:
        return ["no registry key (slot/impl@ref_version:variant) and no provenance metadata"]
    return proc.info.problems()


# ---------------------------------------------------------------------------------- registry


def _definition(fn: Callable[..., Any]) -> tuple[Any, ...]:
    """Identity of a function's definition that survives a module reload."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return ("object", id(fn))
    return (
        getattr(fn, "__module__", None),
        getattr(fn, "__qualname__", None),
        code.co_filename,
        code.co_firstlineno,
    )


class ProcessRegistry(MutableMapping[str, Process]):
    """All processes defined with :func:`process`, indexed by name and by registry key.

    As a mapping it is keyed by process name (iteration yields names); ``registry[k]`` also
    accepts a registry key. Adding a different process under a taken name or key raises
    :class:`DuplicateProcessError`.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, Process] = {}
        self._by_key: dict[str, Process] = {}

    # ---- mapping protocol (by name; keys also accepted for reads and deletes)
    def __getitem__(self, name_or_key: str) -> Process:
        proc = self._by_name.get(name_or_key)
        if proc is None:
            proc = self._by_key.get(name_or_key)
        if proc is None:
            raise KeyError(name_or_key)
        return proc

    def __setitem__(self, name: str, proc: Process) -> None:
        if name != proc.name:
            raise ValueError(f"registry name {name!r} differs from the process name {proc.name!r}")
        self.add(proc)

    def __delitem__(self, name_or_key: str) -> None:
        proc = self[name_or_key]
        orphans = self._variants_of(proc)
        if orphans:
            raise MissingFaithfulError(
                f"cannot remove {proc.key!r}: the variants {orphans} would lose their faithful sibling "
                "(remove them first)"
            )
        self._remove(proc)

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._by_name))

    def __len__(self) -> int:
        return len(self._by_name)

    # ---- registry operations
    def add(self, proc: Process) -> Process:
        """Register ``proc``; raise :class:`DuplicateProcessError` if its name or key is taken, and
        :class:`MissingFaithfulError` if it is a variant (other than ``faithful``, with a reference)
        whose faithful sibling is not registered."""
        key = proc.key
        if proc.info is not None and proc.info.key.needs_faithful_sibling:
            sibling = str(proc.info.key.faithful)
            if sibling not in self._by_key:
                raise MissingFaithfulError(
                    f"cannot register variant {key!r} ({proc.name!r}): its faithful sibling {sibling!r} is "
                    "not registered. Every non-faithful variant of a reference needs the faithful "
                    "implementation next to it (M3 coupling contract, decision 7); register the faithful "
                    "process first, or use ref_version 'none' for a process without a reference"
                )
        clash: list[str] = []
        taken = [self._by_name.get(proc.name)]
        if key is not None:
            taken.append(self._by_key.get(key))
        for old in taken:
            if old is not None and _definition(old.fn) != _definition(proc.fn):
                clash.append(f"{old.name!r} ({old.key or 'unkeyed'}, {getattr(old.fn, '__module__', '?')})")
        if clash:
            raise DuplicateProcessError(
                f"cannot register process {proc.name!r} ({key or 'unkeyed'}): already taken by "
                f"{', '.join(clash)}; give the new process its own key and a distinct name=..."
            )
        for old in taken:  # the same definition again (module reload): replace it
            if old is not None:
                self._remove(old)
        self._by_name[proc.name] = proc
        if key is not None:
            self._by_key[key] = proc
        return proc

    def _variants_of(self, proc: Process) -> list[str]:
        """Registered keys whose faithful sibling is ``proc`` (empty unless ``proc`` is faithful)."""
        if proc.info is None or proc.info.variant != FAITHFUL or self._by_key.get(str(proc.key)) is not proc:
            return []
        fk = proc.info.key
        return sorted(
            k
            for k, p in self._by_key.items()
            if p.info is not None and p.info.key.needs_faithful_sibling and p.info.key.faithful == fk
        )

    def _remove(self, proc: Process) -> None:
        if self._by_name.get(proc.name) is proc:
            del self._by_name[proc.name]
        key = proc.key
        if key is not None and self._by_key.get(key) is proc:
            del self._by_key[key]

    def lookup(self, key: str | ProcessKey) -> Process:
        """The process registered under ``key`` (``slot/impl@ref_version:variant``)."""
        k = str(ProcessKey.parse(key))
        try:
            return self._by_key[k]
        except KeyError:
            pk = ProcessKey.parse(k)
            near = sorted(q for q in self._by_key if q.startswith(f"{pk.slot}/{pk.impl}@"))
            hint = f"; registered for {pk.slot}/{pk.impl}: {near}" if near else ""
            raise KeyError(f"no process registered under {k!r}{hint}") from None

    def keyed(self) -> dict[str, Process]:
        """Copy of the key -> process table."""
        return dict(self._by_key)

    def select(
        self,
        *,
        slot: str | None = None,
        impl: str | None = None,
        ref_version: str | None = None,
        variant: str | None = None,
        provenance: str | None = None,
        grid: str | None = None,
        include_unkeyed: bool = False,
    ) -> list[Process]:
        """Keyed processes matching every given field, sorted by key (``impl`` also matches a
        dotted prefix: ``impl="ceres_maize"`` selects ``ceres_maize.phenology``)."""
        out: list[Process] = []
        for k in sorted(self._by_key):
            p = self._by_key[k]
            i = p.info
            assert i is not None
            if slot is not None and i.slot != slot:
                continue
            if impl is not None and not (i.impl == impl or i.impl.startswith(impl + ".")):
                continue
            if ref_version is not None and i.ref_version != ref_version:
                continue
            if variant is not None and i.variant != variant:
                continue
            if provenance is not None and i.provenance != provenance:
                continue
            if grid is not None and i.grid != grid:
                continue
            out.append(p)
        if include_unkeyed and all(f is None for f in (slot, impl, ref_version, variant, provenance, grid)):
            out.extend(self._by_name[n] for n in sorted(self._by_name) if self._by_name[n].info is None)
        return out


registry = ProcessRegistry()
"""Global table of every process defined with :func:`process` (by name and by registry key)."""


def lookup(key: str | ProcessKey) -> Process:
    """The process registered under ``key`` in the global :data:`registry`."""
    return registry.lookup(key)


def list_processes(**filters: Any) -> list[Process]:
    """Keyed processes of the global :data:`registry`; see :meth:`ProcessRegistry.select`."""
    return registry.select(**filters)


def _check_signature(fn: Callable[..., Any]) -> None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins, some callables
        return
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    required = [p for p in positional if p.default is inspect.Parameter.empty]
    if has_varargs:
        return
    if len(required) > 3 or len(positional) < 3:
        raise ProcessSignatureError(
            f"process {getattr(fn, '__name__', fn)!r} must accept (state, params, forcing); got {sig}"
        )


def _make_info(
    key: str | ProcessKey | None,
    provenance: str | None,
    sources: Iterable[Source | Sequence[str]] | None,
    grid: str | None,
    deviates: Iterable[Deviation | Sequence[str]] | None,
    ref_build: str,
) -> ProcessInfo | None:
    given = {
        "provenance": provenance,
        "sources": sources,
        "grid": grid,
        "deviates": deviates,
    }
    if key is None:
        extra = [n for n, v in given.items() if v is not None] + (["ref_build"] if ref_build else [])
        if extra:
            raise ValueError(f"registry metadata {extra} given without a key (slot/impl@ref_version:variant)")
        return None
    missing = [n for n, v in given.items() if v is None]
    if missing:
        raise ValueError(f"process key {str(key)!r} needs {missing} (deviates=() when there are none)")
    assert provenance is not None and sources is not None and grid is not None and deviates is not None
    info = ProcessInfo(
        key=ProcessKey.parse(key),
        provenance=provenance,
        sources=tuple(_as_source(s) for s in sources),
        grid=grid,
        deviates=tuple(_as_deviation(d) for d in deviates),
        ref_build=ref_build,
    )
    problems = info.problems()
    if problems:
        raise ValueError(f"incomplete metadata for process {str(info.key)!r}: {problems}")
    return info


def process(
    fn: Callable[..., Any] | None = None,
    *,
    reads: str | Iterable[str] | None = None,
    writes: str | Iterable[str] | None = None,
    source: str = "",
    fortran_name: str = "",
    name: str | None = None,
    register: bool = True,
    key: str | ProcessKey | None = None,
    provenance: str | None = None,
    sources: Iterable[Source | Sequence[str]] | None = None,
    grid: str | None = None,
    deviates: Iterable[Deviation | Sequence[str]] | None = None,
    ref_build: str = "",
) -> Any:
    """Declare a process function.

    Usage::

        @process(
            reads=("crop.stage", "soil.theta"),
            writes=("crop.lai",),
            source="DSSAT-CSM MZ_GROSUB.for",
            key="crop/leaf_growth@dssat-4.8.6.0:faithful",
            provenance="translated_bsd3",
            sources=[("leaf expansion", "MZ_GROSUB.for, eqs. 12-15")],
            grid="point",
            deviates=(),
        )
        def leaf_growth(state, params, forcing_t):
            '''...

            Source: DSSAT-CSM MZ_GROSUB.for, eqs. 12-15.
            '''
            return eqx.tree_at(lambda s: s.crop.lai, state, new_lai)

    Parameters
    ----------
    reads, writes:
        Dotted paths into the state (``"crop.lai"``) or sub-trees (``"crop"``).
        A sub-tree path covers every leaf below it; ``"*"`` covers everything.
    source:
        Literature or manual reference for the equations (one line).
    fortran_name:
        Name of the corresponding Fortran subroutine in the oracle.
    name:
        Registry name; defaults to the function name. Names are unique in the registry.
    register:
        Set to False for throwaway processes in tests so they do not shadow real ones.
    key:
        Registry key ``slot/impl@ref_version:variant`` (:class:`ProcessKey`). With a key,
        ``provenance``, ``sources``, ``grid`` and ``deviates`` are required and checked.
        Without a key the process is registered by name only (tests, ad-hoc processes).
    provenance:
        One of :data:`PROVENANCE`.
    sources:
        Per-equation sources, ``Source(what, ref)`` or ``(what, ref)`` pairs.
    grid:
        One of :data:`GRIDS`.
    deviates:
        Known deviations from the reference, ``Deviation(what, why, evidence)`` or triples;
        ``()`` when there are none. A variant other than ``faithful`` must list at least one.
    ref_build:
        The reference binary or source revision the key's ``ref_version`` was checked against.
    """
    info = _make_info(key, provenance, sources, grid, deviates, ref_build)

    def wrap(f: Callable[..., Any]) -> Process:
        _check_signature(f)
        pname = name or getattr(f, "__name__", repr(f))
        proc = Process(
            fn=f,
            name=pname,
            reads=_normalise(reads),
            writes=_normalise(writes),
            source=source,
            fortran_name=fortran_name,
            doc=inspect.getdoc(f) or "",
            info=info,
        )
        if register:
            registry.add(proc)
        return proc

    if fn is not None:
        return wrap(fn)
    return wrap
