"""The segment of the adaptive integrator: step search (primal) and masked replay (reverse pass).

Part of :mod:`~agrijax.processes.soil_water.richards_adaptive` (split out as a pure move); the
search-then-replay scheme and its ``custom_vjp`` are described in that module's docstring.
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard

from .adaptive_newton import _LIVE, _LIVE_LAST, NewtonInfo, adaptive_step
from .adaptive_stepping import AdaptiveStepping
from .hydraulics import AnyHydraulicParams
from .integrator import _ALPHA_CN, _ALPHA_FIRST
from .problem import (
    BC_CLIP,
    BC_FLUX,
    BC_PONDED,
    HOURS_PER_DAY,
    RichardsConfig,
    RichardsGrid,
    SinkOf,
    StepResult,
    _rate,
    _sink_provider,
    post_step,
)
from .sinks import SINK_CHANNELS, SinkChannels

_TIME_ATOL: float = numerical_guard(
    "richards.time_atol", 1.0e-9, "time [h] below which a segment counts as finished (float64)"
)
_TIME_ATOL_F32: float = numerical_guard(
    "richards.time_atol_f32", 1.0e-5, "time [h] below which a segment counts as finished (float32)"
)
_DT_MIN_RTOL: float = numerical_guard(
    "richards.dt_min_rtol", 1.0e-6, "relative tolerance of the test 'the step is at dt_min'"
)

# ---------------------------------------------------------------------------
# the segment: step search (primal) and masked replay (reverse pass)
# ---------------------------------------------------------------------------


class _SegIn(NamedTuple):
    """Differentiable inputs of a segment (``soil`` on the node axis)."""

    h: Array
    theta: Array
    pond: Array
    soil: AnyHydraulicParams
    grid: RichardsGrid
    supply: Array  # [24] cm h-1
    evaporation: Array  # [24] cm h-1
    channels: SinkChannels
    h_min: Array
    pond_max: Array
    pori: Array | None = None  # field-saturated porosity per node (RichardsConfig.drain_cap), else None
    config: RichardsConfig = RichardsConfig()  # the problem's settings (static: no leaves)


class _SegCtl(NamedTuple):
    """Non-differentiable controls of a segment."""

    t_start: Array
    t_end: Array
    dt0: Array
    first: Array


class StepTrace(NamedTuple):
    """Per-entry record of the step table, padded to ``max_steps + 1`` (empty entries: ``live = 0``).

    The flux entries are depths over the step [cm]; ``sinks``/``sinks_cut`` are ``[entry, channel]``.
    """

    t0: Array
    dt: Array
    alpha: Array
    bc: Array  # surface condition of the accepted solve (BC_CLIP, BC_PONDED, BC_FLUX)
    live: Array
    conv: Array
    k: Array
    n_clamp: Array
    clamp_last: Array
    n_active: Array
    residual: Array  # max_i |R_i| dt / tl_i at the last Newton evaluation
    balance_error: Array  # sum(tl dtheta) - dt (q_top - q_bot - sum tl S) of the accepted step [cm]
    theta_residual: Array  # max_i |R_i(h_new)| dt / tl_i of the accepted step
    supply: Array
    infiltration: Array
    evaporation: Array
    drainage: Array
    uptake: Array
    runoff: Array
    evaporation_deficit: Array
    uptake_cut: Array
    sinks: Array
    sinks_cut: Array
    drain_seepage: Array  # DRAIN cap seepage of the step [cm] (included in drainage)
    drain_moved: Array  # water the DRAIN cap passed down in the step [cm]


class _SegOut(NamedTuple):
    h: Array
    theta: Array
    pond: Array
    trace: StepTrace


class _SegMeta(NamedTuple):
    """Counters of the search (no gradient)."""

    n_rejects: Array
    n_newton: Array
    budget_exhausted: Array
    dt_next: Array


def _record(
    t0: Array,
    dt: Array,
    alpha: Array,
    bc: Array,
    sup: Array,
    r: StepResult,
    info: NewtonInfo,
    live: Array,
    seep: tuple[Array, Array] | None = None,
) -> StepTrace:
    """The trace entry of a step (zeros where ``live`` is 0); ``seep`` the DRAIN cap's
    ``(seepage, moved)`` [cm]."""

    def m(x: Array) -> Array:
        return jnp.where(live > 0.0, x, 0.0)

    return StepTrace(
        t0=t0,
        dt=m(dt),
        alpha=alpha,
        bc=m(bc),
        live=live,
        conv=m(info.conv),
        k=m(info.k),
        n_clamp=m(info.n_clamp),
        clamp_last=m(info.clamp_last),
        n_active=m(info.n_active),
        residual=m(info.residual),
        balance_error=m(r.balance_error),
        theta_residual=m(r.theta_residual),
        supply=m(sup * dt),
        infiltration=m(r.infiltration),
        evaporation=m(r.evaporation),
        drainage=m(r.drainage),
        uptake=m(r.uptake),
        runoff=m(r.runoff),
        evaporation_deficit=m(r.evaporation_deficit),
        uptake_cut=m(r.uptake_cut),
        sinks=m(r.sinks),
        sinks_cut=m(r.sinks_cut),
        drain_seepage=m(jnp.zeros_like(dt) if seep is None else seep[0]),
        drain_moved=m(jnp.zeros_like(dt) if seep is None else seep[1]),
    )


def _drained(
    inp: _SegIn, cfg: AdaptiveStepping, r: StepResult
) -> tuple[StepResult, tuple[Array, Array] | None]:
    """The step after RZWQM2's DRAIN cap (``RichardsConfig.drain_cap``; else unchanged, ``None``):
    heads and water contents capped, the seepage added to the drainage.

    Source: RZWQM2 ``DRAIN`` after every ``RICHRD`` step (``Rzrich.for:1092``), read for conventions;
    :func:`~agrijax.processes.soil_water.problem.post_step` (the problem's convention hooks).
    """
    return post_step(r, inp.soil, inp.grid.tl, inp.pori, inp.config)


def _step_at(
    inp: _SegIn,
    cfg: AdaptiveStepping,
    sink_of: SinkOf,
    h: Array,
    th: Array,
    pd: Array,
    t0: Array,
    dt: Array,
    alpha: Array,
    bc: Array | float = BC_CLIP,
    on: Array | bool = True,
) -> tuple[StepResult, NewtonInfo, Array, Array]:
    """One try at ``(t0, dt, alpha)`` under the surface condition ``bc``; ``on = False`` runs no
    Newton iteration. Returns the step, the Newton outcome, the supply rate and whether the solution
    solves the clipped problem (:func:`adaptive_step`)."""
    sup = _rate(inp.supply, t0, dt)
    eva = _rate(inp.evaporation, t0, dt)
    live = jnp.where(_last_resort(cfg, dt, alpha), _LIVE_LAST, _LIVE)
    live = jnp.where(on, live, 0.0).astype(th.dtype)
    r, info, consistent = adaptive_step(
        h, th, pd, inp.soil, inp.grid, sup, eva, sink_of(t0, dt, th, h), dt, alpha, inp.h_min,
        inp.pond_max, cfg, live, h, bc, inp.config,
    )  # fmt: skip
    return r, info, sup, consistent


def _converged(info: NewtonInfo, consistent: Array) -> Array:
    return (info.conv > 0.0) & (info.clamp_last == 0.0) & consistent


def _try_step(
    inp: _SegIn,
    cfg: AdaptiveStepping,
    sink_of: SinkOf,
    h: Array,
    th: Array,
    pd: Array,
    t0: Array,
    dt: Array,
    alpha: Array,
    bc_prev: Array,
    switch: bool,
) -> tuple[StepResult, NewtonInfo, Array, Array, Array, Array]:
    """A try of the search. With ``switch`` (``AdaptiveStepping.bc_switch``): the surface condition
    of the last accepted step ``bc_prev`` first, and when that fails to converge the others (clipped,
    ponded, flux) in turn, taking the first that converges to a solution of the clipped problem;
    without a converged solve, the clipped one. Without ``switch``: the clipped condition alone.
    Returns ``(step, Newton outcome, supply rate, converged, bc, Newton evaluations of all
    solves)``.

    Source: RZWQM2 ``CHKBC``
    (``Rzrich.for``: the surface condition switched outside the iteration), read for conventions.
    """
    if not switch:  # static: cfg.bc_switch
        r0, i0, sup, c0 = _step_at(inp, cfg, sink_of, h, th, pd, t0, dt, alpha)
        return r0, i0, sup, _converged(i0, c0), jnp.asarray(BC_CLIP, th.dtype), i0.k
    # the condition of the last accepted step first (sticky, as CHKBC), then the others in turn
    m0 = jnp.asarray(bc_prev, th.dtype)
    m1 = jnp.where(m0 == BC_CLIP, BC_PONDED, BC_CLIP).astype(th.dtype)
    m2 = jnp.where(m0 == BC_FLUX, BC_PONDED, BC_FLUX).astype(th.dtype)
    r0, i0, sup, c0 = _step_at(inp, cfg, sink_of, h, th, pd, t0, dt, alpha, m0)
    ok0 = _converged(i0, c0)
    r1, i1, _, c1 = _step_at(inp, cfg, sink_of, h, th, pd, t0, dt, alpha, m1, ~ok0)
    ok1 = ~ok0 & _converged(i1, c1)
    r2, i2, _, c2 = _step_at(inp, cfg, sink_of, h, th, pd, t0, dt, alpha, m2, ~ok0 & ~ok1)
    ok2 = ~ok0 & ~ok1 & _converged(i2, c2)
    ok = ok0 | ok1 | ok2
    # without a converged solve: the clipped one (m0 or m1), the problem the step solves
    take1 = ok1 | (~ok & (m1 == BC_CLIP))

    def pick(a: Any, b: Any, c: Any) -> Any:
        return jax.tree_util.tree_map(lambda x, y, z: jnp.where(take1, y, jnp.where(ok2, z, x)), a, b, c)

    r, info = pick((r0, i0), (r1, i1), (r2, i2))
    bc = jnp.where(take1, m1, jnp.where(ok2, m2, m0)).astype(th.dtype)
    return r, info, sup, ok, bc, i0.k + i1.k + i2.k


def _last_resort(cfg: AdaptiveStepping, dt: Array, alpha: Array) -> Array:
    """Whether a try at ``(dt, alpha)`` is the last resort of the search (accepted even unconverged)."""
    at_min = dt <= cfg.dt_min * (1.0 + _DT_MIN_RTOL)
    return at_min & ((alpha >= _ALPHA_FIRST) | (not cfg.cn_fallback))


class _SearchCarry(NamedTuple):
    t: Array
    dt: Array
    h: Array
    th: Array
    pond: Array
    first: Array
    prev_fail: Array
    bc: Array
    last_reset: Array
    trips: Array
    n_acc: Array
    n_rej: Array
    n_newton: Array
    trace: StepTrace


def _time_atol(f32: bool) -> float:
    return _TIME_ATOL_F32 if f32 else _TIME_ATOL


def _empty_trace(inp: _SegIn, n_tab: int, t_end: Array) -> StepTrace:
    dtype = inp.theta.dtype
    z = jnp.zeros((n_tab,), dtype)
    zc = jnp.zeros((n_tab, len(SINK_CHANNELS)), dtype)
    fields = {k: z for k in StepTrace._fields}
    fields.update(
        t0=jnp.full((n_tab,), t_end, dtype),
        alpha=jnp.full((n_tab,), _ALPHA_FIRST, dtype),
        sinks=zc,
        sinks_cut=zc,
    )
    return StepTrace(**fields)


def _put(trace: StepTrace, i: Array, rec: StepTrace, on: Array) -> StepTrace:
    """Write ``rec`` at entry ``i`` of ``trace`` where ``on``."""
    return StepTrace(*(a.at[i].set(jnp.where(on, b, a[i])) for a, b in zip(trace, rec, strict=True)))


def _search(cfg: AdaptiveStepping, inp: _SegIn, ctl: _SegCtl) -> tuple[_SegOut, _SegMeta]:
    """The step search over ``[t_start, t_end]``: a bounded while loop that also yields the values.

    Source: RZWQM2 ``ADJDT``
    (``Rzday.for``: limits, growth, restart after an event) and ``CNHEAD`` (``Rzrich.for``:
    halving on non-convergence), read for conventions.
    """
    dtype = inp.theta.dtype
    n_tab = cfg.max_steps + 1
    atol = _time_atol(dtype == jnp.float32)
    a_rest = _ALPHA_CN if cfg.time_scheme == "rzwqm" else _ALPHA_FIRST
    last_hour = int(HOURS_PER_DAY) - 1
    zero = jnp.zeros((), dtype)
    one = jnp.ones((), dtype)
    dt_lo = jnp.asarray(cfg.dt_min, dtype)
    dt_hi = jnp.asarray(cfg.dt_max, dtype)
    t_end = ctl.t_end
    supply = inp.supply
    sink_of = _sink_provider(inp.channels, inp.grid, dtype)

    def cond(c: _SearchCarry) -> Array:
        return (c.t < t_end - atol) & (c.trips < cfg.max_trips) & (c.n_acc < cfg.max_steps)

    def body(c: _SearchCarry) -> _SearchCarry:
        hour = jnp.clip(jnp.floor(c.t), 0, last_hour).astype(jnp.int32)
        hf = hour.astype(dtype)
        s_now = supply[hour]
        s_prev = jnp.where(hour > 0, supply[jnp.maximum(hour - 1, 0)], 0.0)
        onset = (c.t == hf) & (s_now > 0.0) & (s_prev <= 0.0) & (hour != c.last_reset)
        dt = jnp.where(onset, jnp.minimum(c.dt, cfg.dt_reset), c.dt)
        bp = jnp.minimum(hf + 1.0, t_end)
        rem = bp - c.t
        finish = rem <= cfg.breakpoint_stretch * dt
        dt_try = jnp.where(
            finish, rem, jnp.where(rem <= cfg.breakpoint_split * dt, rem / cfg.breakpoint_split, dt)
        )
        retry = c.prev_fail & cfg.cn_fallback
        alpha = jnp.where(c.first | retry, _ALPHA_FIRST, a_rest).astype(dtype)
        r, info, sup, ok, bc, k_all = _try_step(
            inp, cfg, sink_of, c.h, c.th, c.pond, c.t, dt_try, alpha, c.bc, cfg.bc_switch
        )
        r, seep = _drained(inp, cfg, r)
        alpha1 = alpha >= _ALPHA_FIRST
        accept = ok | _last_resort(cfg, dt_try, alpha)
        # the next step size (a constant of the replay)
        base = jnp.maximum(dt, dt_try)
        updates = info.k - 1.0  # Newton updates to convergence (k counts the polish evaluation)
        grown = jnp.where(c.prev_fail, base, cfg.dt_grow * base)
        dt_ok = jnp.where(
            updates <= cfg.iter_grow_max,
            grown,
            jnp.where(updates >= cfg.iter_shrink_min, cfg.dt_shrink_slow * base, base),
        )
        retry_cn = ~alpha1 & cfg.cn_fallback  # a failed CN step: same dt, alpha = 1
        dt_fail = jnp.where(retry_cn, dt, cfg.dt_shrink_fail * dt_try)
        dt_new = jnp.clip(jnp.where(accept, dt_ok, dt_fail), dt_lo, dt_hi)
        rec = _record(c.t, dt_try, alpha, bc, sup, r, info, one, seep)
        t_new = jnp.where(finish, bp, c.t + dt_try)
        return _SearchCarry(
            t=jnp.where(accept, t_new, c.t),
            dt=dt_new,
            h=jnp.where(accept, r.h, c.h),
            th=jnp.where(accept, r.theta, c.th),
            pond=jnp.where(accept, r.pond, c.pond),
            first=c.first & ~accept,
            prev_fail=~accept,
            bc=jnp.where(accept, bc, c.bc),
            last_reset=jnp.where(onset, hour, c.last_reset),
            trips=c.trips + 1,
            n_acc=c.n_acc + accept.astype(jnp.int32),
            n_rej=c.n_rej + (~accept).astype(dtype),
            n_newton=c.n_newton + k_all,
            trace=_put(c.trace, c.n_acc, rec, accept),
        )

    i0 = jnp.zeros((), jnp.int32)
    init = _SearchCarry(
        t=jnp.asarray(ctl.t_start, dtype),
        dt=jnp.clip(jnp.asarray(ctl.dt0, dtype), dt_lo, dt_hi),
        h=inp.h,
        th=inp.theta,
        pond=inp.pond,
        first=jnp.asarray(ctl.first, bool),
        prev_fail=jnp.zeros((), bool),
        bc=jnp.asarray(BC_CLIP, dtype),
        last_reset=i0 - 1,
        trips=i0,
        n_acc=i0,
        n_rej=zero,
        n_newton=zero,
        trace=_empty_trace(inp, n_tab, t_end),
    )
    c = lax.while_loop(cond, body, init)
    # budget exhausted: the rest of the segment is one step (Newton to kmax, accepted, counted)
    exhausted = c.t < t_end - atol

    def forced(c: _SearchCarry) -> tuple[Array, Array, Array, StepTrace, Array]:
        dt = jnp.where(exhausted, t_end - c.t, one)
        alpha = jnp.asarray(_ALPHA_FIRST, dtype)
        r, info, sup, _ = _step_at(inp, cfg, sink_of, c.h, c.th, c.pond, c.t, dt, alpha)
        r, seep = _drained(inp, cfg, r)
        rec = _record(c.t, dt, alpha, jnp.asarray(BC_CLIP, dtype), sup, r, info, one, seep)
        return r.h, r.theta, r.pond, _put(c.trace, c.n_acc, rec, exhausted), info.k

    def not_forced(c: _SearchCarry) -> tuple[Array, Array, Array, StepTrace, Array]:
        return c.h, c.th, c.pond, c.trace, zero

    h, th, pd, trace, k_forced = lax.cond(exhausted, forced, not_forced, c)
    meta = _SegMeta(
        n_rejects=c.n_rej,
        n_newton=c.n_newton + k_forced,
        budget_exhausted=exhausted.astype(dtype),
        dt_next=c.dt,
    )
    return _SegOut(h=h, theta=th, pond=pd, trace=trace), meta


Table = tuple[Array, Array, Array, Array]  # (t0, dt, alpha, bc) per entry


def _replay(cfg: AdaptiveStepping, inp: _SegIn, table: Table) -> _SegOut:
    """The segment along a frozen step table ``(t0, dt, alpha, bc)``: a masked scan (empty entries
    skipped).

    Every step is the same Newton solve as in the search, with its implicit-function VJP; this is
    the function the reverse pass differentiates.
    """
    dtype = inp.theta.dtype
    sink_of = _sink_provider(inp.channels, inp.grid, dtype)
    one = jnp.ones((), dtype)

    def live_step(h: Array, th: Array, pd: Array, x: Table) -> tuple[Array, Array, Array, StepTrace]:
        t0, dt, al, bc = x
        r, info, sup, _ = _step_at(inp, cfg, sink_of, h, th, pd, t0, dt, al, bc)
        r, seep = _drained(inp, cfg, r)
        return r.h, r.theta, r.pond, _record(t0, dt, al, bc, sup, r, info, one, seep)

    def body(carry: tuple[Array, Array, Array], x: Table) -> tuple[Any, StepTrace]:
        h, th, pd = carry
        t0, dt, al, bc = x
        live = dt > 0.0
        x_live = (t0, jnp.where(live, dt, 1.0), al, bc)
        shapes = jax.eval_shape(live_step, h, th, pd, x_live)

        def dead(h: Array, th: Array, pd: Array, x: Table) -> Any:
            rec = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), shapes[3])
            return h, th, pd, rec._replace(t0=x[0], alpha=x[2])

        h2, th2, pd2, rec = lax.cond(live, live_step, dead, h, th, pd, x_live)
        return (h2, th2, pd2), rec

    (h, th, pd), trace = lax.scan(body, (inp.h, inp.theta, inp.pond), table)
    return _SegOut(h=h, theta=th, pond=pd, trace=trace)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def _segment(cfg: AdaptiveStepping, inp: _SegIn, ctl: _SegCtl) -> tuple[_SegOut, _SegMeta]:
    return _search(cfg, inp, ctl)


def _segment_fwd(
    cfg: AdaptiveStepping, inp: _SegIn, ctl: _SegCtl
) -> tuple[tuple[_SegOut, _SegMeta], tuple[_SegIn, Table]]:
    out, meta = _search(cfg, inp, ctl)
    return (out, meta), (inp, (out.trace.t0, out.trace.dt, out.trace.alpha, out.trace.bc))


def _segment_bwd(
    cfg: AdaptiveStepping, res: tuple[_SegIn, Table], g: tuple[_SegOut, _SegMeta]
) -> tuple[_SegIn, None]:
    """Reverse pass: the VJP of the replay along the search's step table (the decisions are constants)."""
    inp, table = res
    _, vjp = jax.vjp(lambda i: _replay(cfg, i, table), inp)
    (g_inp,) = vjp(g[0])
    return g_inp, None


_segment.defvjp(_segment_fwd, _segment_bwd)
