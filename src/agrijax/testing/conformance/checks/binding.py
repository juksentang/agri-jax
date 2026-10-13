"""Binding check: bound against unbound, replay against coupled, unbound ports raising."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax import lax

from agrijax.core.ports import Binding, BindingError, bind, compose, detach, port_names
from agrijax.core.process import (
    Process,
    process,
)
from agrijax.core.state import get_path, set_path

from ..case import ConformanceCase
from ._common import (
    _day,
    _diff_leaves,
    _fail,
    _inputs,
    _is_float,
    _jit_run,
    _main_dtype,
    _scan,
    _step_all,
    _under,
    _within_ulps,
)


# ------------------------------------------------------------------------------------ binding
def _producer(port_field: str, target: str) -> Process:
    def produce(state: Any, params: Any, forcing_t: Any) -> Any:
        """Kit fixture: the port record of today, computed in the same step (coupled binding).

        Source: conformance kit fixture.
        """
        rec0 = params["rec0"][port_field]
        factor = forcing_t["factor"]
        rec = jtu.tree_map(lambda x: x * factor.astype(x.dtype) if _is_float(x) else x, rec0)
        return set_path(state, target, rec)

    return process(produce, reads=(), writes=(target,), register=False, name=f"kit.produce.{port_field}")


def _replayer(port_field: str, target: str) -> Process:
    def replay(state: Any, params: Any, forcing_t: Any) -> Any:
        """Kit fixture: the port record of today, read from data (replay binding).

        Source: conformance kit fixture.
        """
        return set_path(state, target, forcing_t["replay"][port_field])

    return process(replay, reads=(), writes=(target,), register=False, name=f"kit.replay.{port_field}")


def check_binding(case: ConformanceCase) -> None:
    """Ports: the bound process equals the process on its own; a replay binding (port records
    from data) equals a coupled binding (records produced in the same step) bit for bit; a port
    the process uses but the binding leaves out raises :class:`~agrijax.core.ports.BindingError`.

    The replay covers every port the process reads, ``inout`` ones included: the producer writes
    the whole record, the replay feeds the record recorded at the end of the coupled step (with
    the process's own writes in it). The two agree only when the process depends on a port through
    the fields it reads and does not write; a process that reads back a field it writes into a
    shared port (its memory kept outside its own subtree) is not replayable and fails here."""
    name = "binding"
    proc = case.proc
    s0, p, f = _inputs(case, _main_dtype())
    ports = port_names(type(s0))
    pmap = case.port_map
    if not ports:
        return
    used = sorted({q.split(".", 1)[0] for q in (*proc.reads, *proc.writes)} & set(ports))
    if not pmap:
        if used:
            raise _fail(case, name, f"{proc.name} uses ports {used} and the case binds none")
        return
    own = case.own_path
    close = None if case.transforms_exact else _within_ulps
    binding = Binding(own, tuple(pmap.items()))
    g0 = compose(binding.entries(s0))
    bound = bind(proc, own=own, ports=pmap)

    # 1. bound == alone
    final_b, _ = jax.jit(lambda pp, ff, gg: _scan(bound, pp, ff, gg, case.n_days))(p, f, g0)
    final_a, _ = _jit_run(proc, case.n_days)(p, f, s0)
    bad = _diff_leaves(get_path(final_b, own), detach(final_a), close)
    for port_field, target in pmap.items():
        diff = _diff_leaves(get_path(final_b, target), getattr(final_a, port_field), close)
        bad += [f"{port_field}.{x}" for x in diff]
    if bad:
        raise _fail(case, name, f"bound and unbound runs differ at {bad}")

    # 2. replay == coupled, for every port the process reads (read-only and inout)
    reads_in = [q for q in used if any(_under(r, q) or _under(q, r) for r in proc.reads)]
    if reads_in:
        rng = np.random.default_rng([case.seed, 13])
        dt = _main_dtype()
        factor = jnp.asarray(1.0 + 0.01 * rng.uniform(-1.0, 1.0, case.n_days), dt)
        rec0 = {q: getattr(s0, q) for q in reads_in}
        gp = {"module": p, "rec0": rec0}
        inner = bind(proc, own=own, ports=pmap, params="module", forcing="module", name="kit.module")
        producers = [_producer(q, pmap[q]) for q in reads_in]
        coupled = _step_all([*producers, inner])

        def run_coupled(pp: Any, ff: Any, gg: Any) -> Any:
            def body(s: Any, f_t: Any) -> tuple[Any, Any]:
                s1 = coupled(s, pp, f_t)
                return s1, {q: get_path(s1, pmap[q]) for q in reads_in}

            return lax.scan(body, gg, ff, length=case.n_days)

        final_c, recs = jax.jit(run_coupled)(gp, {"module": f, "factor": factor}, g0)
        replay = _step_all([*(_replayer(q, pmap[q]) for q in reads_in), inner])
        final_r, _ = jax.jit(lambda pp, ff, gg: _scan(replay, pp, ff, gg, case.n_days))(
            gp, {"module": f, "replay": recs}, g0
        )
        rclose = None if case.binding_exact else _within_ulps
        bad = _diff_leaves(get_path(final_c, own), get_path(final_r, own), rclose)
        for port_field, target in pmap.items():
            diff = _diff_leaves(get_path(final_c, target), get_path(final_r, target), rclose)
            bad += [f"{port_field}.{x}" for x in diff]
        if bad:
            raise _fail(
                case,
                name,
                f"replay and coupled bindings differ at {bad} (replayed ports {reads_in}; a process must "
                "depend on a port only through the fields it reads and does not write)",
            )

    # 3. a used port left unbound fails at trace time
    f0 = _day(f, 0)
    for q in used:
        rest = {k: v for k, v in pmap.items() if k != q}
        partial = bind(proc, own=own, ports=rest)
        g_partial = compose(Binding(own, tuple(rest.items())).entries(s0))
        try:
            partial(g_partial, p, f0)
        except BindingError:
            continue
        except Exception as e:
            raise _fail(
                case, name, f"port {q!r} left unbound raised {type(e).__name__} ({e}) instead of BindingError"
            ) from None
        raise _fail(case, name, f"port {q!r} left unbound did not raise BindingError")
