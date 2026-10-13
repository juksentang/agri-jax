"""Framework entries of an assembled day: a replay entry (a port written from a reference run's
record) and a no-op entry (a placeholder of the contract's day order).

Source: :mod:`agrijax.iface.contract` (the day table and the replay binding of a port).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agrijax.core.process import Process, process
from agrijax.core.state import get_path, set_path

__all__ = ["noop_entry", "replay_entry"]


def replay_entry(name: str, copies: Mapping[str, str], *, stands_in_reads: Sequence[str] = ()) -> Process:
    """A replay entry: copy ``{state path: forcing path}`` records from the day's forcing into the state.

    ``stands_in_reads`` are declared as its reads: the read set of the producer the replay stands
    in for, so the day's lag check sees the coupled dataflow. The copy itself reads no state.
    """
    pairs = tuple((str(t), str(s)) for t, s in copies.items())

    def _replay(state: Any, params: Any, forcing_t: Any) -> Any:
        """Write each target record from its forcing record (a reference run's values).

        Source: :mod:`agrijax.iface.contract` (replay and coupled bindings of the same ports).
        """
        for target, src in pairs:  # static loop over the declared copies
            state = set_path(state, target, get_path(forcing_t, src))
        return state

    return process(
        _replay,
        reads=tuple(stands_in_reads),
        writes=tuple(t for t, _ in pairs),
        name=name,
        register=False,
        source="replay of a reference run (the coupling contract's replay binding of a port)",
    )


def noop_entry(name: str, *, why: str) -> Process:
    """An entry of the contract's day whose producer does not exist yet and that changes nothing
    on the days it is run (``why`` says why): no reads, no writes, the state comes back as is."""

    def _noop(state: Any, params: Any, forcing_t: Any) -> Any:
        """Leave the state unchanged (a placeholder of the contract's day order).

        Source: :mod:`agrijax.iface.contract` day table (the entry's producer is not implemented yet).
        """
        return state

    return process(
        _noop,
        reads=(),
        writes=(),
        name=name,
        register=False,
        source=f"placeholder of the contract's day: {why}",
    )
