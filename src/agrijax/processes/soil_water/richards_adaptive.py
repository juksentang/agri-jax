r"""The adaptive integrator of the Richards problem (``AdaptiveCN``, key ``:faithful``).

The fixed-step integrator (:mod:`~agrijax.processes.soil_water.fixed_cn`) takes ``n_sub`` sub-steps
a day with ``n_iter`` Newton iterations each and no convergence test. This integrator (config
:class:`AdaptiveStepping`, registry key ``soil_water/richards_time@rzwqm2-4.6:faithful``: RZWQM2's
time integration) iterates every sub-step to a tolerance and sets the step size from the outcome of the
Newton solve, like RZWQM2's ``ADJDT`` / ``CNHEAD``. It reads the physics only through
:mod:`~agrijax.processes.soil_water.problem` (residual, surface condition, Jacobian, sink cap, the
post-step DRAIN hook):

* **Newton to a tolerance.** Converged when the node water-content residual
  ``max_i |R_i| dt / tl_i <= newton_tol_theta`` and the sub-step balance
  ``|dt sum_i R_i| <= newton_tol_balance`` (1e-10 in float64), after which one more (polish)
  update is applied; at most ``newton_max_iter`` evaluations; a residual that rises
  ``newton_div_patience`` times in a row from evaluation ``newton_div_after`` on, a non-finite
  residual, or a clamp activation of the last applied update fails the solve (a clamp earlier on
  the path changes the path, not the root). Changes to the damped Newton of the fixed modes
  (measured necessary: without them the adaptive solver took up to 24000 steps a day, ran out of
  its step budget and accumulated 1-8 cm a year of balance error; they apply to this mode only, so
  the fixed modes stay bit for bit as they were):

  - **N1** the storage floor of the Jacobian is ``max(C, c_floor)`` instead of ``C + c_floor``
    (the step stays the exact Newton step wherever ``C >= c_floor``, which restores quadratic
    convergence);
  - **N2** a backtracking line search on the scaled residual ``sum((R dt / tl)^2)`` over the
    factors ``line_search`` (the first that lowers it, else the last), which breaks the 2-cycles
    seen at saturated and very dry fronts;
  - **N3** a conservative dry bound: the iterate is bounded below at ``dry_guard h_min`` (or the
    step's start head where that is drier), a divergence guard, not at ``h_min``. A node at
    ``h_min`` that still loses water (the gravity flux ``K(h_min)`` to the node below, the
    explicit half of Crank-Nicolson) drains below it, so the step conserves water; the uptake cap
    and the surface dry limit (ghost head ``h_min``) keep sinks and evaporation from taking a node
    past ``h_min``. Holding such nodes at ``h_min`` left their residual as balance error (up to
    3e-8 cm a step). A node at the guard whose update points further out is held there (an
    active set, ``n_active``; it should never occur).

* **Step size** (no time-error estimate, as in RZWQM2). Time weights: ``alpha =
  1`` on the first step of a segment, ``1/2`` after (``time_scheme = "rzwqm"``; ``"implicit"`` is
  ``alpha = 1`` throughout). A failed Crank-Nicolson solve is retried at the same ``dt`` with
  ``alpha = 1`` (``cn_fallback``), a failed ``alpha = 1`` solve is redone from the step's initial
  state with ``dt_shrink_fail dt``. After a converged step with ``k`` Newton updates the next
  step is ``dt_grow dt`` for ``k <= iter_grow_max``, ``dt_shrink_slow dt`` for ``k >=
  iter_shrink_min``, else ``dt``, never grown right after a failure, and always in
  ``[dt_min, dt_max]``. The last-resort solve
  (``dt_min``, ``alpha = 1``) may take ``newton_max_iter_last`` evaluations and has no divergence
  exit. Breakpoints: every hour boundary of the hourly forcing and the segment
  end; a remaining time of at most ``breakpoint_stretch dt`` is one step, of at most
  ``breakpoint_split dt`` is split into equal steps. At the onset of surface supply (an hour with
  supply after one without; the first hour of a day counts as one) and after an infiltration
  event the step restarts from ``dt_reset``. The last step size is carried to the next day in
  ``SoilWater.dt_next``.

* **Surface condition** (``bc_switch``). The residual of
  the fixed modes clips the requested surface flux to its ponded and dry limits, which is not
  smooth at the ponding switch: Newton 2-cycled there between the flux branch and the ponded
  branch floored at 0 (seen in the US_OPE storms). A try first solves under the condition of the
  last accepted step (clipped, ponded: ghost head 0, flux: the requested flux; sticky, as RZWQM2's
  ``CHKBC``), then under the others in turn, and takes the first that converges to a solution of
  the clipped problem (its clipped surface flux equals the condition's), so every accepted step
  solves the same equations as the fixed modes. The condition is part of the step table.

* **Failures stay finite and are counted.** A step that fails at ``dt_min`` with ``alpha = 1`` is
  accepted unconverged; a segment that runs out of accepted steps (``max_steps``) or tries
  (``max_trips``) ends in one unconverged step to the segment end. Both are counted
  (``n_unconverged``, ``budget_exhausted``) and raise under ``AGRI_JAX_CHECK=1`` in the process
  wrappers.

Implementation (search first, then replay): a **search** over the segment
with a bounded ``lax.while_loop`` decides the accepted steps ``(t0, dt, alpha, bc)`` and records them
in a table padded with empty steps to ``max_steps + 1``; a masked ``lax.scan`` **replays** the
table, and the segment is a ``custom_vjp`` whose value is the search's (every step solved once in
a forward-only run) and whose reverse pass is the VJP of the replay along the frozen table (the
same Newton solves; the step decisions are constants). Every replayed Newton solve is
differentiated by the implicit function theorem (``custom_vjp``: one transposed tridiagonal
solve per accepted step at the converged root, with the active-set rows as ``h_i`` = bound),
never through the iterations, so the gradient is that of the frozen step table. An empty step of
the replay is the identity. Forward-mode differentiation is not supported.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone
Water Quality Model, ch. 3; RZWQM2 ``ADJDT`` (``Rzday.for``, step limits and growth), ``CNHEAD``
(``Rzrich.for``, halving on non-convergence), read for conventions only; Celia, M.A., Bouloutas,
E.T., Zarba, R.L., 1990. Water Resour. Res. 26, 1483-1496 (mixed form); Dennis, J.E., Schnabel,
R.B., 1996. Numerical Methods for Unconstrained Optimization and Nonlinear Equations, SIAM, ch. 6
(backtracking line search); Griewank, A., Walther, A., 2008. Evaluating Derivatives, 2nd ed.,
SIAM, ch. 15 (implicit-function derivatives of an iterative solve).
"""

from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array

from agrijax.core.process import Deviation, Source, check_enabled

from .adaptive_newton import NewtonInfo, adaptive_step, newton_to_tolerance
from .adaptive_search import StepTrace, _replay, _SegCtl, _SegIn, _segment, _SegMeta, _SegOut
from .adaptive_stepping import DT_MAX_FAST, KEY, AdaptiveStepping
from .integrator import (
    Capabilities,
    DayDiagnostics,
    DayPlan,
    IntegratorInfo,
    combine_totals,
    register_integrator,
)
from .problem import HOURS_PER_DAY, RichardsProblem, SoilWater, SubstepTotals
from .sinks import SinkChannels

__all__ = [
    "DT_MAX_FAST",
    "KEY",
    "AdaptiveCN",
    "AdaptiveStats",
    "AdaptiveStepping",
    "NewtonInfo",
    "StepTrace",
    "adaptive_segment",
    "adaptive_step",
    "check_adaptive",
    "combine_stats",
    "empty_stats",
    "newton_to_tolerance",
    "replay_segment",
    "stats_fluxes",
]


class AdaptiveStats(NamedTuple):
    """Counters of the adaptive stepping of a segment or a day (floats)."""

    n_steps: Array
    n_rejects: Array
    n_newton: Array
    n_unconverged: Array
    budget_exhausted: Array
    n_active: Array
    dt_min_used: Array  # [h]; HOURS_PER_DAY for a segment without steps
    step_balance_max: Array  # [cm] largest |balance_error| of an accepted sub-step; 0 without steps


_NOT_SUMMED = ("dt_min_used", "step_balance_max")


def combine_stats(a: AdaptiveStats, b: AdaptiveStats) -> AdaptiveStats:
    """The counters of two segments of a day (sums, the smaller ``dt_min_used``, the larger
    ``step_balance_max``)."""
    summed = {k: getattr(a, k) + getattr(b, k) for k in AdaptiveStats._fields if k not in _NOT_SUMMED}
    return AdaptiveStats(
        **summed,
        dt_min_used=jnp.minimum(a.dt_min_used, b.dt_min_used),
        step_balance_max=jnp.maximum(a.step_balance_max, b.step_balance_max),
    )


def stats_fluxes(stats: AdaptiveStats) -> dict[str, Array]:
    """The diagnostic fields of :class:`~agrijax.processes.soil_water.richards.SoilWaterFluxes`."""
    return stats._asdict()


def check_adaptive(h: Array, stats: Any, where: str) -> Array:
    """Under ``AGRI_JAX_CHECK=1``: raise on an unconverged step or an exhausted budget (else identity).

    ``stats`` is an :class:`AdaptiveStats` or the day's ``SoilWaterFluxes`` (same field names).
    """
    if check_enabled():
        bad = (stats.n_unconverged > 0.0) | (stats.budget_exhausted > 0.0)
        return eqx.error_if(h, bad, f"{where}: adaptive Richards step did not converge (n_unconverged > 0)")
    return h


def _adaptive(stepping: Any) -> AdaptiveStepping:
    if not isinstance(stepping, AdaptiveStepping):
        raise TypeError(f"the adaptive kernels take an AdaptiveStepping, got {type(stepping).__name__}")
    return stepping


def _seg_in(problem: RichardsProblem, water: SoilWater) -> _SegIn:
    """The differentiable inputs of a segment: the problem of the day and the state at its start."""
    return _SegIn(
        h=water.h,
        theta=water.theta,
        pond=jnp.asarray(water.pond, water.theta.dtype),
        soil=problem.soil,
        grid=problem.grid,
        supply=problem.supply,
        evaporation=problem.evaporation,
        channels=problem.channels,
        h_min=problem.h_min,
        pond_max=problem.pond_max,
        pori=problem.pori,
        config=problem.config,
    )


def _inputs(
    water: SoilWater, params: Any, supply: Any, evaporation: Any, uptake: Any, pori: Any = None
) -> _SegIn:
    problem = RichardsProblem.of_day(params, supply, evaporation, uptake, water.theta.dtype, pori)
    return _seg_in(problem, water)


def replay_segment(
    water: SoilWater,
    params: Any,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
    trace: StepTrace,
    pori: Array | None = None,
) -> tuple[SoilWater, StepTrace]:
    """The segment replayed along the step table of ``trace`` (the function the gradient is of);
    ``pori`` as for :func:`adaptive_segment`.
    """
    table = (trace.t0, trace.dt, trace.alpha, trace.bc)
    inp = _inputs(water, params, supply, evaporation, uptake, pori)
    out = _replay(_adaptive(params.stepping), inp, table)
    return water.replace(h=out.h, theta=out.theta, pond=out.pond), out.trace


def adaptive_segment(
    water: SoilWater,
    params: Any,
    t_start: Any,
    t_end: Any,
    supply: Array,
    evaporation: Array,
    uptake: Array | SinkChannels,
    dt0: Any,
    first: Any = True,
    pori: Array | None = None,
) -> tuple[SoilWater, SubstepTotals, AdaptiveStats, StepTrace]:
    """Advance ``water`` over ``[t_start, t_end]`` [h] with adaptive sub-steps; returns
    ``(state, totals, counters, per-step trace)``.

    ``supply``/``evaporation`` hourly rates ``[24]`` [cm h-1], ``uptake`` the sink channels or the
    per-layer root water uptake [cm d-1] alone, ``dt0`` the starting step [h], ``first`` whether
    the first step takes ``alpha = 1``, ``pori`` the field-saturated porosity per node (needed with
    ``RichardsConfig.drain_cap``: the DRAIN cap after every accepted step), ``params.stepping`` an
    :class:`AdaptiveStepping`. The state carries the
    last step size in ``dt_next`` (no
    gradient) and keeps ``water.flux``. The value is the step search's; the reverse pass
    differentiates the replay of its step table (:func:`replay_segment`), so a forward-only run
    solves every step once. Sink callables (``SinkChannel.rate``) must not close over
    differentiated values.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2
    ``RICHRD`` / ``ADJDT`` (conventions).
    """
    problem = RichardsProblem.of_day(params, supply, evaporation, uptake, water.theta.dtype, pori)
    return segment(problem, _adaptive(params.stepping), water, t_start, t_end, dt0, first)


def segment(
    problem: RichardsProblem,
    cfg: AdaptiveStepping,
    water: SoilWater,
    t_start: Any,
    t_end: Any,
    dt0: Any,
    first: Any = True,
) -> tuple[SoilWater, SubstepTotals, AdaptiveStats, StepTrace]:
    """:func:`adaptive_segment` on the problem of the day (one segment,
    :meth:`~agrijax.processes.soil_water.problem.RichardsProblem.for_segment`).

    Source: RZWQM2 ``RICHRD`` / ``ADJDT`` (conventions).
    """
    dtype = water.theta.dtype
    inp = _seg_in(problem.for_segment(), water)
    sg = partial(jax.tree_util.tree_map, lax.stop_gradient)
    ctl = sg(
        _SegCtl(
            t_start=jnp.asarray(t_start, dtype),
            t_end=jnp.asarray(t_end, dtype),
            dt0=jnp.asarray(dt0, dtype),
            first=jnp.asarray(first, bool),
        )
    )
    out, meta = cast(tuple[_SegOut, _SegMeta], _segment(cfg, inp, ctl))
    meta = sg(meta)
    tr = out.trace
    totals = SubstepTotals(
        supply=jnp.sum(tr.supply),
        infiltration=jnp.sum(tr.infiltration),
        evaporation=jnp.sum(tr.evaporation),
        drainage=jnp.sum(tr.drainage),
        uptake=jnp.sum(tr.uptake),
        runoff=jnp.sum(tr.runoff),
        evaporation_deficit=jnp.sum(tr.evaporation_deficit),
        uptake_cut=jnp.sum(tr.uptake_cut),
        max_theta_residual=jnp.max(tr.theta_residual, initial=0.0),
        n_clamp=jnp.sum(tr.n_clamp),
        sinks=jnp.sum(tr.sinks, axis=0),
        sinks_cut=jnp.sum(tr.sinks_cut, axis=0),
        drain_seepage=jnp.sum(tr.drain_seepage),
        drain_moved=jnp.sum(tr.drain_moved),
    )
    live = lax.stop_gradient(tr.live)
    ok = lax.stop_gradient((tr.conv > 0.0) & (tr.clamp_last == 0.0))
    stats = AdaptiveStats(
        n_steps=jnp.sum(live),
        n_rejects=meta.n_rejects,
        n_newton=meta.n_newton,
        n_unconverged=jnp.sum(jnp.where(live > 0.0, (~ok).astype(dtype), 0.0)),
        budget_exhausted=meta.budget_exhausted,
        n_active=jnp.sum(jnp.where(lax.stop_gradient(tr.n_active) > 0.0, live, 0.0)),
        dt_min_used=jnp.min(jnp.where(live > 0.0, lax.stop_gradient(tr.dt), HOURS_PER_DAY)),
        step_balance_max=jnp.max(jnp.abs(lax.stop_gradient(tr.balance_error)), initial=0.0),
    )
    new = water.replace(h=out.h, theta=out.theta, pond=out.pond, dt_next=meta.dt_next)
    return new, totals, stats, tr


def empty_stats(dtype: Any) -> AdaptiveStats:
    """Counters of a segment without steps."""
    z = jnp.zeros((), dtype)
    return AdaptiveStats(z, z, z, z, z, z, jnp.asarray(HOURS_PER_DAY, dtype), z)


# ---------------------------------------------------------------------------
# the integrator
# ---------------------------------------------------------------------------


class AdaptiveCN:
    """The adaptive integrator (:class:`~agrijax.processes.soil_water.integrator.RichardsIntegrator`).

    A day without an event is one segment ``[0, 24]`` h from the carried step ``water.dt_next``. An
    event day: ``[0, t_ev]`` from ``water.dt_next`` (none with ``n_pre = 0``), the event at ``t_ev``,
    then ``[t_ev, 24]`` restarted from ``dt_reset`` after an event; each segment starts with ``alpha =
    1`` (after an event a deviation from RZWQM2, whose first post-event step has ``alpha = 1/2``;
    the event leaves a steep profile behind the front on which Crank-Nicolson may oscillate). ``t_ev`` is the
    storm start (``n_pre > 0``) or 0 (``n_pre = 0``, and on a day without an event), so a day without an
    event is one segment; the day's ``n_pre``, ``n_post`` and ``post_grading`` place no sub-steps here.
    """

    def __init__(self, config: AdaptiveStepping) -> None:
        self.config = _adaptive(config)

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            gradient="custom",
            dtypes=("float64", "float32"),
            batching="the step search is a while loop: a vmapped batch runs until its slowest lane ends "
            "(group lanes by site); the reverse pass replays each lane's own step table",
            forward_mode=False,
            conserved="theta",
            convergence=True,
        )

    def step_day(
        self, problem: RichardsProblem, water: SoilWater, plan: DayPlan
    ) -> tuple[SoilWater, DayDiagnostics]:
        """One day (:class:`~agrijax.processes.soil_water.integrator.RichardsIntegrator`).

        Source: RZWQM2 ``PHYSCL`` / ``ADJDT``
        (``Rzday.for``), ``EVNTRO`` / ``INFIL`` (``RZTEST.for``), read for conventions.
        """
        cfg = self.config
        dtype = water.theta.dtype
        ev = plan.event
        if ev is None:  # static: a day without an event, one segment
            new, tot, stats, _ = segment(problem, cfg, water, 0.0, HOURS_PER_DAY, water.dt_next)
            return new, DayDiagnostics(tot, stats)
        t_ev = jnp.where(ev.has_event, jnp.clip(jnp.asarray(ev.ts0, dtype), 0.0, HOURS_PER_DAY), 0.0)
        t_ev, w_pre, tot_pre, st_pre = _pre_segment(ev.n_pre, problem, cfg, water, t_ev)
        w_ev, result = ev.apply(w_pre)
        dt_post = jnp.where(ev.has_event, jnp.minimum(w_pre.dt_next, cfg.dt_reset), w_pre.dt_next)
        w_post, tot_post, st_post, _ = segment(problem, cfg, w_ev, t_ev, HOURS_PER_DAY, dt_post)
        stats = combine_stats(st_pre, st_post)
        return w_post, DayDiagnostics(combine_totals(tot_pre, tot_post), stats, result, t_ev)

    def check(self, h: Array, flux: Any, where: str) -> Array:
        """Under ``AGRI_JAX_CHECK=1``: raise on an unconverged step or an exhausted budget."""
        return check_adaptive(h, flux, where)


def _pre_segment(
    n_pre: int, problem: RichardsProblem, cfg: AdaptiveStepping, water: SoilWater, t_ev: Array
) -> tuple[Array, SoilWater, SubstepTotals | None, AdaptiveStats]:
    """The segment ``[0, t_ev]`` before the event, or none at all with ``n_pre = 0``."""
    if n_pre > 0:  # static: a pre-event segment exists (empty when t_ev = 0)
        w_pre, tot_pre, st_pre, _ = segment(problem, cfg, water, 0.0, t_ev, water.dt_next)
        return t_ev, w_pre, tot_pre, st_pre
    return jnp.zeros_like(t_ev), water, None, empty_stats(water.theta.dtype)


INFO = register_integrator(
    IntegratorInfo(
        key=KEY,
        config_type=AdaptiveStepping,
        factory=AdaptiveCN,
        summary="adaptive sub-steps from the Newton outcome (RZWQM2 ADJDT/CNHEAD), Newton to a tolerance "
        "with the fixes N1-N3, Crank-Nicolson with alpha = 1 on a segment's first step and on a retry",
        sources=(
            Source("step limits, growth and restart after an event", "RZWQM2 ADJDT (Rzday.for), conventions"),
            Source("halving on non-convergence", "RZWQM2 CNHEAD (Rzrich.for), conventions"),
            Source("backtracking line search", "Dennis & Schnabel (1996) ch. 6"),
            Source("implicit-function derivatives of a solve", "Griewank & Walther (2008) ch. 15"),
        ),
        deviates=(
            Deviation(
                "Newton on the transformed head with the exact Jacobian; RZWQM2 iterates modified Picard",
                "quadratic convergence to the 1e-10 tolerance (fixes N1-N3 of the module docstring)",
                "richards_adaptive.py module docstring (without N1-N3 the adaptive solver took up to 24000 "
                "steps a day); tests/unit/test_richards_adaptive.py",
            ),
            Deviation(
                "the first Richards step after an infiltration event has alpha = 1 (RZWQM2: 1/2)",
                "the event leaves a steep profile behind the front on which Crank-Nicolson may oscillate",
                "richards_adaptive.py AdaptiveCN docstring; tests/unit/test_richards_adaptive.py",
            ),
        ),
    )
)
