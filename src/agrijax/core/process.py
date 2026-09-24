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
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agrijax.core.state import tree_diff

__all__ = [
    "CHECK_ENV",
    "Process",
    "ProcessSignatureError",
    "ProcessWriteError",
    "check_enabled",
    "process",
    "registry",
]

CHECK_ENV = "AGRI_JAX_CHECK"
WILDCARD = "*"


class ProcessWriteError(AssertionError):
    """A process modified state fields it did not declare in ``writes``."""


class ProcessSignatureError(TypeError):
    """A process does not accept exactly ``(state, params, forcing)``."""


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

    def __call__(self, state: Any, params: Any, forcing: Any) -> Any:
        new_state = self.fn(state, params, forcing)
        if check_enabled():
            self.check_writes(state, new_state)
        return new_state

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


registry: dict[str, Process] = {}
"""Global table of every process defined with :func:`process`, keyed by name."""


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


def process(
    fn: Callable[..., Any] | None = None,
    *,
    reads: str | Iterable[str] | None = None,
    writes: str | Iterable[str] | None = None,
    source: str = "",
    fortran_name: str = "",
    name: str | None = None,
    register: bool = True,
) -> Any:
    """Declare a process function.

    Usage::

        @process(reads=("crop.stage", "soil.theta"), writes=("crop.lai",))
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
        Literature or manual reference for the equations.
    fortran_name:
        Name of the corresponding Fortran subroutine in the oracle.
    name:
        Registry name; defaults to the function name.
    register:
        Set to False for throwaway processes in tests so they do not shadow real ones.
    """

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
        )
        if register:
            registry[pname] = proc
        return proc

    if fn is not None:
        return wrap(fn)
    return wrap
