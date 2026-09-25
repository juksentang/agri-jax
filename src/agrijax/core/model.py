"""``Model`` = a state definition plus an ordered list of processes; ``compile()`` builds ``day_step``.

Replacing a process is editing the list. The compiled day step is a single pure
function ``(state, params, forcing_t) -> (state, outputs_t)`` suitable for
``lax.scan``; the runtime never sees individual processes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from agrijax.core.process import Process, process
from agrijax.core.state import get_path

__all__ = ["DayStep", "Model"]


def _overlap(a: str, b: str) -> bool:
    """Two dotted paths overlap when equal or one is a dotted prefix of the other (``*`` overlaps all)."""
    return a == b or a == "*" or b == "*" or a.startswith(b + ".") or b.startswith(a + ".")


DayStep = Callable[[Any, Any, Any], tuple[Any, Any]]
OutputFn = Callable[[Any, Any, Any], Any]


class Model:
    """A state class and an ordered pipeline of processes.

    Parameters
    ----------
    state_spec:
        The ``State`` subclass this model evolves (documentation and validation only;
        the runtime works on whatever pytree ``state0`` is).
    processes:
        Ordered processes. Plain callables are wrapped with ``writes=("*",)`` so
        they are accepted but not write-checked.
    outputs:
        What the day step emits per day. ``None`` emits the full state; a sequence of
        dotted paths emits ``{path: value}``; a callable ``(state, params, forcing_t) -> pytree``
        emits whatever it returns.
    name:
        Optional label used in reports.
    day:
        The :class:`~agrijax.core.day.Day` declaration this model was compiled from, if any
        (phases, declared lags). Set by :meth:`Day.compile <agrijax.core.day.Day.compile>`;
        introspection only, it adds nothing to the compiled day step.
    """

    def __init__(
        self,
        state_spec: type | None,
        processes: Iterable[Process | Callable[..., Any]],
        *,
        outputs: Sequence[str] | OutputFn | None = None,
        name: str = "",
        day: Any = None,
    ) -> None:
        self.state_spec = state_spec
        self.processes: tuple[Process, ...] = tuple(self._as_process(p) for p in processes)
        names = [p.name for p in self.processes]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate process names in model: {dup}")
        self.name = name or (state_spec.__name__ if state_spec is not None else "Model")
        self._outputs = outputs
        self.day = day

    @staticmethod
    def _as_process(p: Process | Callable[..., Any]) -> Process:
        if isinstance(p, Process):
            return p
        if callable(p):
            return process(p, writes=("*",), register=False)
        raise TypeError(f"not a process or callable: {p!r}")

    # ------------------------------------------------------------------ introspection
    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.processes)

    def replace(self, old: str, new: Process | Callable[..., Any]) -> Model:
        """Return a new model with process ``old`` (by name) replaced by ``new``."""
        if old not in self.names:
            raise KeyError(old)
        procs = [self._as_process(new) if p.name == old else p for p in self.processes]
        # a replaced process may read or write differently, so the Day declaration is not carried over
        return Model(self.state_spec, procs, outputs=self._outputs, name=self.name)

    def dataflow(self) -> list[tuple[str, str, str]]:
        """Edges ``(writer, reader, path)`` within one day: a
        later process reads what an earlier one wrote."""
        edges: list[tuple[str, str, str]] = []
        for i, reader in enumerate(self.processes):
            for r in reader.reads:
                for writer in self.processes[:i]:
                    if any(r == w or r.startswith(w + ".") or w.startswith(r + ".") for w in writer.writes):
                        edges.append((writer.name, reader.name, r))
        return edges

    def stale_reads(self) -> list[tuple[str, str]]:
        """``(process, path)`` pairs read before any process of the day writes them.

        Such reads see the previous day's value, which is legitimate for true state but
        a bug for daily diagnostics. Returned for inspection, never raised.
        """
        written: set[str] = set()
        stale: list[tuple[str, str]] = []
        for p in self.processes:
            for r in p.reads:
                if not any(r == w or r.startswith(w + ".") or w.startswith(r + ".") for w in written):
                    stale.append((p.name, r))
            written.update(p.writes)
        return stale

    def writers(self, path: str) -> tuple[str, ...]:
        """Names of the processes whose declared writes overlap ``path`` (equal, above or below it)."""
        return tuple(p.name for p in self.processes if any(_overlap(path, w) for w in p.writes))

    def lagged_reads(self, module_of: Callable[[str], str] | None = None) -> list[tuple[str, str]]:
        """Stale reads that see **another module's** value from the previous day.

        A stale read ``(process, path)`` (:meth:`stale_reads`) is *lagged* when a process of a
        different module writes ``path`` during the day: the reader runs before that writer, so it
        sees yesterday's output. ``module_of`` maps a process name to its module (default: every
        process is its own module; :class:`~agrijax.core.day.Day` uses the entry name without its
        last component, so ``soil.watbal_rate`` and ``soil.watbal_integr`` are one module). These
        are the reads a ``Day`` must declare as :class:`~agrijax.core.day.Lag`.
        """
        mod = module_of or (lambda n: n)
        return [(p, r) for p, r in self.stale_reads() if any(mod(w) != mod(p) for w in self.writers(r))]

    def carried_reads(self, module_of: Callable[[str], str] | None = None) -> list[tuple[str, str]]:
        """Stale reads of true state: ``path`` is written only by the reader's own module or by
        nobody (a state variable carried across days, or an initial condition)."""
        lagged = set(self.lagged_reads(module_of))
        return [pr for pr in self.stale_reads() if pr not in lagged]

    # ------------------------------------------------------------------ compile
    def _output_fn(self) -> OutputFn:
        outputs = self._outputs
        if outputs is None:
            return lambda state, params, forcing_t: state
        if callable(outputs):
            return outputs
        paths = tuple(outputs)
        return lambda state, params, forcing_t: {p: get_path(state, p) for p in paths}

    def compile(self) -> DayStep:
        """Chain the processes into one pure ``day_step(state, params, forcing_t) -> (state, outputs_t)``."""
        procs = self.processes
        out_fn = self._output_fn()

        def day_step(state: Any, params: Any, forcing_t: Any) -> tuple[Any, Any]:
            for p in procs:  # Python loop over a static list of processes, unrolled at trace time
                state = p(state, params, forcing_t)
            return state, out_fn(state, params, forcing_t)

        day_step.__name__ = f"{self.name}_day_step"
        return day_step

    def __repr__(self) -> str:
        return f"Model({self.name}, processes=[{', '.join(self.names)}])"
