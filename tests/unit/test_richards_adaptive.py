"""The adaptive Richards integrator (``RichardsParams.stepping = AdaptiveStepping()``).

No data: the CA-TPA grid and soil of ``test_richards`` and synthetic forcing.

* every accepted sub-step converges (node residual and balance within the Newton tolerance) and
  closes its water balance to the tolerance; the day's balance closes;
* the step table hits every hour boundary exactly and never spans one; the step restarts from
  ``dt_reset`` at the onset of supply; ``dt_next`` is carried from day to day;
* failures are retried, halved and recovered; an exhausted budget is finite, counted and raises
  under ``AGRI_JAX_CHECK=1``;
* ``jit`` re-runs are bit for bit, ``vmap`` equals the loop, float32 runs and stays close to float64;
* step decisions carry no gradient; the implicit-function gradient equals central differences
  on the frozen step table;
* the soil-water day with a Green-Ampt event runs both adaptive segments and closes its balance;
* the day's largest per-step balance is reported (``step_balance_max``); output steps from
  step-table changes stay below 1e-3 cm on a storm day; central differences of the full adaptive
  run at two steps agree with AD where the step table does not move;
* a profile at h_min drains below it and every step conserves water (bounded at h_min, the
  held rows left the gravity flux as balance error); a start profile drier than h_min is not wetted
  by the bound; a saturated surface block under heavy supply (the Newton 2-cycle at the ponding
  switch) converges with the surface-condition switching.

Source: Celia et al. (1990) (per-step balance).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from agrijax.core import Model
from agrijax.core.runtime import run
from agrijax.processes.soil_water import richards_adaptive as A
from agrijax.processes.soil_water.day import DayConfig, soil_water_day, soil_water_day_kernel
from agrijax.processes.soil_water.richards import (
    DT_MAX_FAST,
    DT_START,
    AdaptiveStepping,
    FixedStepping,
    RichardsConfig,
    RichardsForcing,
    RichardsParams,
    RichardsState,
    SoilWater,
    richards_day,
    richards_redistribution,
)

from . import test_infiltration_day as TD
from .test_richards import catpa_grid, catpa_soil, synthetic_forcing

X64 = bool(jax.config.read("jax_enable_x64"))
FDT = jnp.float64 if X64 else jnp.float32
#: time atol of the step table in the current dtype
T_ATOL = 1e-12 if X64 else 1e-5


def _storm_day(amount: float = 2.5, hour: int = 3) -> tuple[jax.Array, jax.Array, jax.Array]:
    """A 2016-09-10-like day: ``amount`` cm in one hour on a dry profile, daytime evaporation, uptake."""
    supply = np.zeros(24)
    supply[hour] = amount
    hours = np.arange(24)
    evap = np.where((hours >= 6) & (hours < 18), 0.3 / 12, 0.0)
    uptake = np.zeros(37)
    uptake[:12] = 0.25 / 12
    return jnp.asarray(supply, FDT), jnp.asarray(evap, FDT), jnp.asarray(uptake, FDT)


def _params(**kw) -> RichardsParams:
    return RichardsParams(soil=catpa_soil(), grid=catpa_grid(), stepping=AdaptiveStepping(**kw))


def _dry(head: float = -3000.0) -> SoilWater:
    """A dry profile at a uniform head (above h_min in every horizon)."""
    return SoilWater.from_head(jnp.full(37, head, FDT), catpa_soil())


def _segment(params: RichardsParams, water: SoilWater, forcing=None, dt0=None):
    supply, evap, uptake = _storm_day() if forcing is None else forcing
    dt0 = water.dt_next if dt0 is None else dt0
    fn = jax.jit(lambda w, p: A.adaptive_segment(w, p, 0.0, 24.0, supply, evap, uptake, dt0))
    return fn(water, params)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_adaptive_config() -> None:
    cfg = AdaptiveStepping()
    assert (cfg.time_scheme, cfg.dt_max) == ("rzwqm", 0.1) and AdaptiveStepping.exact() == cfg
    assert AdaptiveStepping.fast().dt_max == DT_MAX_FAST == 0.25 == AdaptiveStepping.tier("fast").dt_max
    assert (cfg.dt_min, cfg.dt_reset, cfg.dt_grow, cfg.dt_shrink_fail, cfg.dt_shrink_slow) == (
        1e-4,
        1e-4,
        1.5,
        0.5,
        0.7,
    )
    assert (cfg.newton_max_iter, cfg.newton_tol_theta, cfg.newton_tol_balance, cfg.newton_polish) == (
        10,
        1e-10,
        1e-10,
        True,
    )
    assert isinstance(RichardsParams(soil=catpa_soil(), grid=catpa_grid()).stepping, FixedStepping)
    assert DT_START == 1e-4
    # the adaptive integrator has no gradient or Jacobian mode (always Newton with implicit-function VJPs)
    for kw in ({"stepping": "adaptive"}, {"grad": "implicit"}, {"jacobian": "picard"}, {"n_sub": 24}):
        with pytest.raises(TypeError):
            AdaptiveStepping(**kw)
    with pytest.raises(TypeError):  # the problem's config holds no numerics
        RichardsConfig(stepping="adaptive")  # type: ignore[call-arg]
    for kw in ({"dt_min": 1.0}, {"line_search": ()}, {"newton_max_iter": 1}, {"breakpoint_stretch": 3.0}):
        with pytest.raises(ValueError):
            AdaptiveStepping(**kw)
    with pytest.raises(ValueError):
        AdaptiveStepping(time_scheme="sometimes")
    with pytest.raises(ValueError):
        AdaptiveStepping.tier("slow")
    with pytest.raises(TypeError):
        RichardsParams(soil=catpa_soil(), grid=catpa_grid(), stepping=RichardsConfig())


# ---------------------------------------------------------------------------
# convergence and conservation per accepted step
# ---------------------------------------------------------------------------


def test_every_accepted_step_converges_and_closes_its_balance() -> None:
    params = _params()
    cfg = params.stepping
    new, tot, st, tr = _segment(params, _dry())
    live = np.asarray(tr.live) > 0
    n = int(live.sum())
    assert float(st.n_steps) == n and n >= 240  # dt_max = 0.1 h: at least 240 steps
    assert float(st.n_unconverged) == 0.0 and float(st.budget_exhausted) == 0.0
    assert float(tot.n_clamp) == 0.0 and float(st.n_active) == 0.0
    assert np.all(np.asarray(tr.conv)[live] == 1.0)
    tol_th = cfg.newton_tol_theta if X64 else cfg.newton_tol_theta_f32
    tol_b = cfg.newton_tol_balance if X64 else cfg.newton_tol_balance_f32
    assert np.all(np.asarray(tr.residual)[live] <= tol_th)
    # the accepted (polished) step closes its balance to the tolerance: sum(tl dtheta) - dt * fluxes
    bal = np.abs(np.asarray(tr.balance_error)[live])
    assert bal.max() <= tol_b, bal.max()
    print(f"per-step |balance| max {bal.max():.3e} median {np.median(bal):.3e}; steps {n}; "
          f"rejects {float(st.n_rejects)}; newton {float(st.n_newton)}")  # fmt: skip
    # the day: storage change = supply - evaporation - drainage - uptake - runoff
    w0 = _dry()
    grid = params.grid
    day_bal = (new.storage(grid) + new.pond - w0.storage(grid) - w0.pond) - (
        tot.supply - tot.evaporation - tot.drainage - tot.uptake - tot.runoff
    )
    assert abs(float(day_bal) - float(np.sum(np.asarray(tr.balance_error)))) <= (1e-11 if X64 else 1e-4)
    assert abs(float(day_bal)) <= (n * tol_b)
    assert abs(float(tot.supply) - 2.5) <= (1e-12 if X64 else 1e-5)


def test_richards_day_adaptive_equals_the_segment_and_fills_the_diagnostics() -> None:
    params = _params()
    supply, evap, uptake = _storm_day()
    w = jax.jit(lambda w: richards_day(w, params, supply, evap, uptake))(_dry())
    new, _, st, _ = _segment(params, _dry())
    # two compiled programs of the same computation: equal to rounding (measured 4e-16)
    np.testing.assert_allclose(
        np.asarray(w.theta), np.asarray(new.theta), rtol=0, atol=1e-14 if X64 else 1e-6
    )
    f = w.flux
    assert float(f.n_steps) == float(st.n_steps) and float(f.n_newton) == float(st.n_newton)
    assert float(f.n_unconverged) == 0.0 and float(f.budget_exhausted) == 0.0
    assert 0.0 < float(f.dt_min_used) <= params.stepping.dt_reset * (1 + 1e-6)  # restarted at the onset
    assert float(f.drainage) >= 0.0
    assert abs(float(f.balance_error)) <= float(f.n_steps) * (1e-10 if X64 else 3e-5)
    assert float(w.dt_next) == float(new.dt_next)
    assert 1e-4 * (1 - 1e-6) <= float(w.dt_next) <= 0.1 * (1 + 1e-6)


def test_the_value_is_the_replay_of_the_step_table() -> None:
    """The segment's value (the search) equals the replay of its table, which the gradient differentiates."""
    params = _params()
    supply, evap, uptake = _storm_day()
    new, _, _, tr = _segment(params, _dry())
    rep, tr2 = jax.jit(lambda w: A.replay_segment(w, params, supply, evap, uptake, tr))(_dry())
    np.testing.assert_allclose(
        np.asarray(rep.theta), np.asarray(new.theta), rtol=0, atol=1e-14 if X64 else 1e-6
    )
    live = np.asarray(tr.live) > 0
    np.testing.assert_array_equal(np.asarray(tr2.live), np.asarray(tr.live))
    np.testing.assert_array_equal(np.asarray(tr2.conv)[live], np.asarray(tr.conv)[live])
    np.testing.assert_array_equal(np.asarray(tr2.k)[live], np.asarray(tr.k)[live])
    np.testing.assert_allclose(
        np.asarray(tr2.drainage), np.asarray(tr.drainage), rtol=0, atol=1e-15 if X64 else 1e-7
    )


# ---------------------------------------------------------------------------
# the step table: breakpoints, reset at supply onset, carried step
# ---------------------------------------------------------------------------


def test_step_table_hits_every_hour_boundary_and_restarts_at_supply_onset() -> None:
    params = _params()
    _, _, _, tr = _segment(params, _dry())
    live = np.asarray(tr.live) > 0
    t0 = np.asarray(tr.t0)[live]
    dt = np.asarray(tr.dt)[live]
    assert t0[0] == 0.0 and np.all(np.diff(t0) > 0.0)
    for hour in range(24):
        assert np.any(t0 == float(hour)), hour  # every hour boundary is a step start, exactly
    # no step spans an hour boundary; the steps tile [0, 24]
    assert np.all(np.floor(t0) == np.floor(t0 + dt - T_ATOL))
    np.testing.assert_allclose(t0[1:], (t0 + dt)[:-1], rtol=0, atol=T_ATOL)
    assert abs(t0[-1] + dt[-1] - 24.0) <= T_ATOL
    # a remaining time of at most breakpoint_stretch dt before a breakpoint is one step
    assert dt.max() <= params.stepping.breakpoint_stretch * params.stepping.dt_max * (1 + 1e-6)
    # the storm hour (3) starts from dt_reset; the first step of the day has alpha = 1, the others 1/2
    k3 = int(np.flatnonzero(t0 == 3.0)[0])
    assert dt[k3] == pytest.approx(params.stepping.dt_reset, rel=1e-6)
    al = np.asarray(tr.alpha)[live]
    assert al[0] == 1.0 and np.mean(al == 0.5) > 0.9


def test_dt_next_is_carried_across_days() -> None:
    params = _params(dt_max=0.25)
    supply, evap, uptake = synthetic_forcing(3, seed=2)
    supply[:] = 0.0  # quiet days: the step grows to dt_max and is carried on
    forcing = RichardsForcing(
        supply=jnp.asarray(supply, FDT), evaporation=jnp.asarray(evap, FDT), uptake=jnp.asarray(uptake, FDT)
    )

    def body(w, f):
        w2 = richards_day(w, params, f.supply, f.evaporation, f.uptake)
        return w2, w2.dt_next

    w0 = SoilWater.from_theta(jnp.full(37, 0.25, FDT), params.soil)
    _, dts = jax.jit(lambda w, f: lax.scan(body, w, f))(w0, forcing)
    assert float(dts[0]) == pytest.approx(0.25) and float(dts[-1]) == pytest.approx(0.25)
    # day 2 starts from day 1's step: its first step is dt_max, not DT_START
    _, _, _, tr = jax.jit(
        lambda w: A.adaptive_segment(
            w, params, 0.0, 24.0, forcing.supply[1], forcing.evaporation[1], forcing.uptake[1], dts[0]
        )
    )(w0)
    assert float(tr.dt[0]) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# failures, budget, AGRI_JAX_CHECK
# ---------------------------------------------------------------------------


def test_failed_solves_are_retried_halved_and_recovered() -> None:
    ref = _segment(_params(), _dry())
    tight = _segment(_params(newton_max_iter=5), _dry())
    for _, _, st, _ in (ref, tight):
        assert float(st.n_unconverged) == 0.0 and float(st.budget_exhausted) == 0.0
    assert float(tight[2].n_rejects) > float(ref[2].n_rejects)
    d = abs(float(tight[0].storage(catpa_grid())) - float(ref[0].storage(catpa_grid())))
    print(f"storage difference, 3 vs 10 Newton evaluations: {d:.3e} cm; rejects {float(tight[2].n_rejects)}")
    assert d <= 1e-3


def test_budget_exhaustion_is_finite_counted_and_raises_under_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGRI_JAX_CHECK", raising=False)  # CI sets it for the whole run
    params = _params(max_trips=5)
    new, _, st, tr = _segment(params, _dry())
    assert float(st.budget_exhausted) == 1.0 and float(st.n_unconverged) >= 1.0
    assert float(st.n_steps) <= 6.0
    assert np.all(np.isfinite(np.asarray(new.h))) and np.all(np.isfinite(np.asarray(new.theta)))
    live = np.asarray(tr.live) > 0
    t_end = np.asarray(tr.t0)[live][-1] + np.asarray(tr.dt)[live][-1]
    assert abs(t_end - 24.0) <= T_ATOL  # the rest of the day is one (unconverged) step
    # the process wrapper raises under AGRI_JAX_CHECK=1 (and not without it)
    supply, evap, uptake = _storm_day()
    forcing = RichardsForcing(supply=supply, evaporation=evap, uptake=uptake)
    state = RichardsState(soil_water=_dry())
    out = richards_redistribution(state, params, forcing)
    assert float(out.soil_water.flux.budget_exhausted) == 1.0
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")
    with pytest.raises(Exception, match="did not converge"):
        jax.block_until_ready(richards_redistribution(state, params, forcing))


def test_process_wrapper_through_the_runtime_writes_dt_next(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _params(dt_max=0.25)
    supply, evap, uptake = synthetic_forcing(4, seed=11)
    forcing = RichardsForcing(
        supply=jnp.asarray(supply, FDT), evaporation=jnp.asarray(evap, FDT), uptake=jnp.asarray(uptake, FDT)
    )
    w0 = SoilWater.from_theta(jnp.full(37, 0.24, FDT), params.soil)
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")
    model = Model(
        RichardsState, [richards_redistribution], outputs=("soil_water.dt_next", "soil_water.flux.n_steps")
    )
    final, outs = jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(
        params, forcing, RichardsState(soil_water=w0)
    )
    w = w0
    for d in range(4):
        w = richards_day(w, params, forcing.supply[d], forcing.evaporation[d], forcing.uptake[d])
    np.testing.assert_allclose(
        np.asarray(final.soil_water.theta), np.asarray(w.theta), rtol=0, atol=1e-12 if X64 else 1e-6
    )
    assert float(outs["soil_water.dt_next"][-1]) == pytest.approx(float(w.dt_next))
    assert np.all(np.asarray(outs["soil_water.flux.n_steps"]) >= 96)


# ---------------------------------------------------------------------------
# determinism, vmap, float32
# ---------------------------------------------------------------------------


def test_rerun_is_bitwise_identical_and_vmap_equals_the_loop() -> None:
    params = _params(dt_max=0.25)
    a = _segment(params, _dry())
    b = _segment(params, _dry())
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    scales = jnp.asarray([0.9, 1.0, 1.1], FDT)
    supply, evap, uptake = _storm_day()

    def one(s):
        soil = params.soil.replace(ksat=params.soil.ksat * s)
        p = params.replace(soil=soil)
        new, _, st, _ = A.adaptive_segment(_dry(), p, 0.0, 24.0, supply, evap, uptake, DT_START)
        return new.theta, st.n_steps, st.n_newton

    th_v, n_v, k_v = jax.jit(jax.vmap(one))(scales)
    for i, s in enumerate(scales):
        th, n, k = jax.jit(one)(s)
        assert float(n_v[i]) == float(n)
        if X64:  # float32 rounding of the batched program may move a residual across the tolerance
            assert float(k_v[i]) == float(k)
        # measured: float64 equal to 1e-12; float32 up to 7e-6 (a decision moved by rounding)
        np.testing.assert_allclose(np.asarray(th_v[i]), np.asarray(th), rtol=0, atol=1e-12 if X64 else 5e-5)


@pytest.mark.allow_skip(reason="compares a float32 run against float64")
def test_float32_runs_and_stays_close_to_float64() -> None:
    if not X64:
        pytest.skip("compares against float64")
    supply, evap, uptake = (np.asarray(x) for x in _storm_day())
    w64 = richards_day(_dry(), _params(dt_max=0.25), *(jnp.asarray(x) for x in (supply, evap, uptake)))
    with jax.enable_x64(False):  # the float32 program, as in the float32 test tier
        p32 = _params(dt_max=0.25)
        w0 = SoilWater.from_head(jnp.full(37, -3000.0), p32.soil)
        f32 = [jnp.asarray(x, jnp.float32) for x in (supply, evap, uptake)]
        w32 = jax.jit(lambda w: richards_day(w, p32, *f32))(w0)
        assert w32.theta.dtype == jnp.float32
        s32 = float(w32.storage(p32.grid))
        flux32 = jax.tree_util.tree_map(float, w32.flux)
    assert np.all(np.isfinite(np.asarray(w32.theta)))
    d = abs(s32 - float(w64.storage(catpa_grid())))
    print(f"float32 - float64 storage {d:.3e} cm; f32 unconverged {flux32.n_unconverged}, "
          f"steps {flux32.n_steps} vs {float(w64.flux.n_steps)}")  # fmt: skip
    assert d <= 1e-3
    assert flux32.budget_exhausted == 0.0


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------


def test_step_decisions_carry_no_gradient() -> None:
    params = _params(dt_max=0.25)
    supply, evap, uptake = _storm_day()

    def loss(dt0, sup):
        new, _, _, _ = A.adaptive_segment(_dry(), params, 0.0, 24.0, sup, evap, uptake, dt0)
        return new.storage(params.grid) + new.dt_next

    g_dt, g_sup = jax.jit(jax.grad(loss, argnums=(0, 1)))(jnp.asarray(0.05, FDT), supply)
    assert float(g_dt) == 0.0  # the starting step only moves the (constant) step table
    assert np.all(np.isfinite(np.asarray(g_sup))) and float(g_sup[3]) > 0.0


FIELDS = ("lambda_", "hb", "ksat", "theta_r", "theta_s")


@pytest.mark.allow_skip(reason="central differences at 1e-5 relative need float64")
def test_implicit_gradient_matches_central_differences_on_the_frozen_table() -> None:
    if not X64:
        pytest.skip("needs float64")
    params = _params(dt_max=0.25)
    supply, evap, uptake = _storm_day(amount=1.2)
    w0 = SoilWater.from_theta(jnp.linspace(0.16, 0.30, 37), params.soil)

    def outputs(soil):
        new, tot, _, tr = A.adaptive_segment(
            w0, params.replace(soil=soil), 0.0, 24.0, supply, evap, uptake, DT_START
        )
        return new.storage(params.grid) + 10.0 * tot.drainage + 3.0 * tot.evaporation, (
            tr.t0,
            tr.dt,
            tr.alpha,
        )

    f = jax.jit(outputs)
    g = jax.jit(jax.grad(lambda s: outputs(s)[0]))(params.soil)
    _, table = f(params.soil)
    worst = 0.0
    for name in FIELDS:
        for hz in (0, 2, 4):
            x = getattr(params.soil, name)
            eps = 1e-4 * abs(float(x[hz]))
            (fp, tp), (fm, tm) = (
                f(params.soil.replace(**{name: x.at[hz].add(s * eps)})) for s in (1.0, -1.0)
            )
            for u, v, w in zip(tp, tm, table, strict=True):  # same (frozen) step table at both ends
                np.testing.assert_array_equal(np.asarray(u), np.asarray(w))
                np.testing.assert_array_equal(np.asarray(v), np.asarray(w))
            fd = (float(fp) - float(fm)) / (2.0 * eps)
            ad = float(getattr(g, name)[hz])
            assert np.isfinite(ad)
            # central differences carry the rounding of the loss (~1e-12 of |L|) divided by eps
            floor = 1e-12 * abs(float(fp)) / eps
            rel = abs(ad - fd) / max(abs(fd), 1e-300)
            if abs(ad - fd) > floor:
                worst = max(worst, rel)
            assert abs(ad - fd) <= 1e-5 * abs(fd) + floor, (name, hz, ad, fd, floor)
    print(f"worst relative AD - FD: {worst:.3e}")


def test_gradient_is_finite_in_the_switching_regimes() -> None:
    """Saturated and near-saturated profiles draining under a flux top, ponding and runoff, the dry end.

    The ponded start (the ponding switch of the surface boundary condition) and the fully saturated
    start can leave unconverged steps: the result and the gradient stay finite and the steps are
    counted.
    """
    params = _params(dt_max=0.25)
    uptake = jnp.zeros(37, FDT)
    zero = jnp.zeros(24, FDT)
    regimes = {
        "saturated_start": (
            SoilWater.from_head(jnp.zeros(37, FDT), params.soil),
            zero,
            jnp.full(24, 0.02, FDT),
        ),
        "near_saturated_draining": (_dry(-20.0), zero, jnp.full(24, 0.02, FDT)),
        "ponding_runoff": (_dry(-100.0), zero.at[2:4].set(20.0), jnp.full(24, 0.02, FDT)),
        "dry_end": (_dry(-14000.0), zero, jnp.full(24, 0.1, FDT)),
    }
    for name, (w0, sup, ev) in regimes.items():

        def loss(soil, w0=w0, sup=sup, ev=ev):
            new, tot, st, _ = A.adaptive_segment(
                w0, params.replace(soil=soil), 0.0, 24.0, sup, ev, uptake, DT_START
            )
            return new.storage(params.grid) + tot.drainage + tot.evaporation + tot.runoff, st

        (val, st), g = jax.jit(jax.value_and_grad(loss, has_aux=True))(params.soil)
        print(f"{name}: steps {float(st.n_steps)} rejects {float(st.n_rejects)} unconverged "
              f"{float(st.n_unconverged)} budget {float(st.budget_exhausted)} active {float(st.n_active)}")  # fmt: skip
        assert np.isfinite(float(val)), name
        for f in FIELDS:
            assert np.all(np.isfinite(np.asarray(getattr(g, f)))), (name, f)
        # the saturated start (C = 0 at every node, a1 = 0, a flux top and free drainage: J is
        # singular) stalls on the uniform-shift mode of the regularised
        # Jacobian; measured: float64 exhausts the budget with 2 unconverged steps, float32
        # converges. Finite and counted (an open item). The ponding regime converges because of
        # the surface-condition switching.
        if name != "saturated_start":
            assert float(st.n_unconverged) == 0.0 and float(st.budget_exhausted) == 0.0, name


# ---------------------------------------------------------------------------
# the soil-water day with the Green-Ampt event
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_pre", [0, 12])
def test_soil_water_day_with_events_runs_both_segments_and_closes(n_pre: int) -> None:
    params = TD._params(day=DayConfig(n_pre=n_pre, n_post=24 - n_pre), stepping=AdaptiveStepping())
    forcing = TD._forcing(6, seed=3, supply_scale=0.3)
    forcing = jax.tree_util.tree_map(lambda x: jnp.asarray(x, FDT), forcing)
    w0 = SoilWater.from_theta(jnp.full(37, 0.25, FDT), params.richards.soil)

    def body(w, f):
        w2, _, shift = soil_water_day_kernel(w, params, f.supply, f.evaporation, f.uptake, f.storm)
        return w2, (w2.flux, shift, w2.dt_next)

    _, (flux, shift, dtn) = jax.jit(lambda w, f: lax.scan(body, w, f))(w0, forcing)
    has_event = np.asarray(jnp.sum(forcing.storm.depth, axis=-1)) > 0
    assert has_event.any() and (~has_event).any()
    assert np.all(np.asarray(flux.n_unconverged) == 0) and np.all(np.asarray(flux.budget_exhausted) == 0)
    bal = np.abs(np.asarray(flux.balance_error))
    print(f"n_pre {n_pre}: day |balance| max {bal.max():.3e}; steps {np.asarray(flux.n_steps)}")
    assert bal.max() <= (1e-8 if X64 else 5e-3)
    assert np.all(np.asarray(flux.n_steps) >= 240)
    ts0 = np.asarray(forcing.storm.ts0)
    expect_shift = np.where(has_event, (np.clip(ts0, 0, 24) if n_pre else 0.0) - ts0, 0.0)
    np.testing.assert_allclose(np.asarray(shift), expect_shift, atol=1e-6)
    # an event restarts the step at dt_reset: the day's smallest step is dt_reset on event days
    dmin = np.asarray(flux.dt_min_used)
    assert np.all(dmin[has_event] <= 1e-4 * (1 + 1e-6))
    assert np.all(np.asarray(dtn) > 0)
    # the registered process gives the same state
    st = RichardsState(soil_water=w0)
    f0 = jax.tree_util.tree_map(lambda x: x[0], forcing)
    out = soil_water_day(st, params, f0)
    w1, _, _ = soil_water_day_kernel(w0, params, f0.supply, f0.evaporation, f0.uptake, f0.storm)
    np.testing.assert_array_equal(np.asarray(out.soil_water.theta), np.asarray(w1.theta))


# ---------------------------------------------------------------------------
# per-step balance diagnostic, output steps, two-step differences
# ---------------------------------------------------------------------------


def test_step_balance_max_is_the_largest_accepted_step_balance() -> None:
    params = _params(dt_max=0.25)
    supply, evap, uptake = _storm_day()
    _, _, st, tr = _segment(params, _dry())
    live = np.asarray(tr.live) > 0
    assert float(st.step_balance_max) == float(np.abs(np.asarray(tr.balance_error)[live]).max())
    w = jax.jit(lambda w: richards_day(w, params, supply, evap, uptake))(_dry())
    assert float(w.flux.step_balance_max) == pytest.approx(
        float(st.step_balance_max), rel=0, abs=1e-13 if X64 else 3e-5
    )
    assert float(w.flux.step_balance_max) <= (1e-10 if X64 else 3e-5)
    fixed = RichardsParams(soil=catpa_soil(), grid=catpa_grid(), stepping=FixedStepping())
    assert float(richards_day(_dry(), fixed, supply, evap, uptake).flux.step_balance_max) == 0.0
    # two segments of a day: the larger of the two
    a = A.empty_stats(FDT)._replace(step_balance_max=jnp.asarray(2.0, FDT))
    b = A.empty_stats(FDT)._replace(step_balance_max=jnp.asarray(3.0, FDT))
    assert float(A.combine_stats(a, b).step_balance_max) == 3.0


def _scan_day(params: RichardsParams, s: jax.Array):
    """Storage + drainage of the storm day with ksat scaled by ``s``, and the step-table fingerprint."""
    supply, evap, uptake = _storm_day(amount=1.5)
    soil = params.soil.replace(ksat=params.soil.ksat * s)
    new, tot, st, tr = A.adaptive_segment(
        _dry(), params.replace(soil=soil), 0.0, 24.0, supply, evap, uptake, DT_START
    )
    wts = jnp.linspace(1.0, 2.0, tr.dt.shape[0], dtype=FDT)
    table = lax.stop_gradient(jnp.sum(wts * (tr.dt + 3.0 * tr.alpha)))
    return new.storage(params.grid) + tot.drainage, (table, st.n_unconverged)


@pytest.mark.allow_skip(reason="the jump measurement needs float64")
def test_output_steps_from_step_count_changes_are_below_the_threshold() -> None:
    """Output steps on one storm day: ``J = |dy - ds (y'(s) + y'(s + ds)) / 2|`` over a 41-point ksat scan.

    Neighbours with the same step table differ by the smooth part (trapezoid error, measured ~5e-11
    cm) and, where a kink of the curves or of the surface switch lies between them, by the kink
    (measured 9.3e-7 cm on one pair); a change of the step table adds the jump (measured 1.0e-4 cm
    on 2 of 40 pairs, rorqual float64); the threshold is 1e-3 cm.
    """
    if not X64:
        pytest.skip("needs float64")
    params = _params(dt_max=0.25)
    s = jnp.linspace(0.95, 1.05, 41, dtype=FDT)
    (y, (table, unc)), dy = jax.jit(
        jax.vmap(jax.value_and_grad(lambda x: _scan_day(params, x), has_aux=True))
    )(s)
    y, dy, table = np.asarray(y), np.asarray(dy), np.asarray(table)
    assert np.all(np.asarray(unc) == 0.0)
    ds = np.diff(np.asarray(s))
    jump = np.abs(np.diff(y) - ds * 0.5 * (dy[:-1] + dy[1:]))
    same = table[1:] == table[:-1]
    print(f"table changes {int((~same).sum())} of {len(same)} pairs; J same-table max "
          f"{jump[same].max(initial=0.0):.3e}, changed-table max {jump[~same].max(initial=0.0):.3e} cm")  # fmt: skip
    assert (~same).any() and same.any()  # the scan crosses step-count changes
    assert np.median(jump[same]) <= 1e-9
    assert jump[same].max() <= 1e-5
    assert jump.max() <= 1e-3


@pytest.mark.allow_skip(reason="central differences at 1e-6 relative need float64")
def test_full_adaptive_differences_at_two_steps_match_the_gradient() -> None:
    """The two-step kink test on the full adaptive run: central differences at relative steps
    1e-4 and 1e-6 (the step table may move) against AD; where both ends keep the nominal step
    table the three agree, a kink or a step-count change inside the step would split them."""
    if not X64:
        pytest.skip("needs float64")
    params = _params(dt_max=0.25)
    f = jax.jit(lambda x: _scan_day(params, x))
    g = float(jax.jit(jax.grad(lambda x: _scan_day(params, x)[0]))(jnp.asarray(1.0, FDT)))
    y0, (t0, _) = f(jnp.asarray(1.0, FDT))
    fds = {}
    for h in (1e-4, 1e-6):
        (yp, (tp, _)), (ym, (tm, _)) = f(jnp.asarray(1.0 + h, FDT)), f(jnp.asarray(1.0 - h, FDT))
        fds[h] = ((float(yp) - float(ym)) / (2.0 * h), float(tp) == float(t0) == float(tm))
    print(f"AD {g:.10e}; FD {fds}")
    floor = 1e-13 * abs(float(y0))
    for h, (fd, same) in fds.items():
        if same:
            assert abs(fd - g) <= 1e-6 * abs(g) + floor / h, (h, fd, g)
    assert any(same for _, same in fds.values())


# ---------------------------------------------------------------------------
# the conservative dry bound, a start profile below h_min, a saturated block
# ---------------------------------------------------------------------------

_NO_FORCING = (jnp.zeros(24, FDT), jnp.zeros(24, FDT), jnp.zeros(37, FDT))


def _tol_b(params: RichardsParams) -> float:
    cfg = params.stepping
    return cfg.newton_tol_balance if X64 else cfg.newton_tol_balance_f32


def _closes(params: RichardsParams, water: SoilWater, new: SoilWater, tot) -> float:
    grid = params.grid
    return float(
        (new.storage(grid) + new.pond - water.storage(grid) - water.pond)
        - (tot.supply - tot.evaporation - tot.drainage - tot.uptake - tot.runoff)
    )


def test_profile_at_h_min_drains_below_it_and_every_step_closes() -> None:
    """The conservative dry bound: a profile at h_min loses water by gravity (``K(h_min)`` between
    layers and out of the bottom) with no sink or evaporation left to cut; the nodes drain below h_min
    and every step conserves water. Bounded at h_min instead (``dry_guard = 1``, the earlier
    behaviour), the held rows leave that flux as balance error."""
    params = _params()
    hmin = float(params.h_min)
    water = SoilWater.from_head(jnp.full(37, hmin, FDT), catpa_soil())
    new, tot, st, tr = _segment(params, water, _NO_FORCING)
    live = np.asarray(tr.live) > 0
    bal = np.abs(np.asarray(tr.balance_error)[live])
    assert float(st.n_unconverged) == 0.0 and float(st.budget_exhausted) == 0.0
    assert float(st.n_active) == 0.0
    assert bal.max() <= _tol_b(params), bal.max()
    assert float(tot.drainage) > 0.0
    assert float(jnp.min(new.h)) < hmin  # drained below h_min
    assert abs(_closes(params, water, new, tot)) <= float(st.n_steps) * _tol_b(params)
    print(f"drainage {float(tot.drainage):.3e} cm, min h {float(jnp.min(new.h)):.4f}, "
          f"step balance max {bal.max():.3e}")  # fmt: skip
    if X64:  # the h_min bound, for contrast (float32 rounds the gravity flux of a step away)
        held = _params(dry_guard=1.0)
        _, _, st1, tr1 = _segment(held, water, _NO_FORCING)
        bal1 = np.abs(np.asarray(tr1.balance_error)[np.asarray(tr1.live) > 0])
        print(f"dry_guard = 1: active steps {float(st1.n_active)}, step balance max {bal1.max():.3e}")
        assert float(st1.n_active) > 0.0 and bal1.max() > _tol_b(held)


def test_start_profile_below_h_min_is_not_wetted_by_the_bound() -> None:
    """Nodes that start drier than h_min (a restart from a reference profile) are bounded at
    their start head, so the first step creates no water and counts as converged."""
    params = _params()
    hmin = float(params.h_min)
    h = jnp.full(37, -3000.0, FDT).at[20:23].set(1.2 * hmin)
    water = SoilWater.from_head(h, catpa_soil())
    new, tot, st, tr = _segment(params, water, _NO_FORCING)
    live = np.asarray(tr.live) > 0
    bal = np.abs(np.asarray(tr.balance_error)[live])
    assert float(st.n_unconverged) == 0.0 and float(st.n_active) == 0.0 and float(tot.n_clamp) == 0.0
    assert bal.max() <= _tol_b(params), bal.max()
    assert abs(_closes(params, water, new, tot)) <= float(st.n_steps) * _tol_b(params)


def test_saturated_block_under_heavy_supply_converges() -> None:
    """A saturated surface block (positive heads) under 3 cm h-1 for two hours, then none: the block
    shifts as a whole at the ponding switch and when supply stops (the failures seen on the US_OPE
    scenario before the surface-condition switching); every step converges and closes its balance,
    and the budget holds."""
    params = _params()
    h = jnp.full(37, -5000.0, FDT).at[:8].set(2.0)
    water = SoilWater.from_head(h, catpa_soil())
    supply = jnp.zeros(24, FDT).at[:2].set(3.0)
    forcing = (supply, jnp.zeros(24, FDT), jnp.zeros(37, FDT))
    new, tot, st, tr = _segment(params, water, forcing)
    bal = np.abs(np.asarray(tr.balance_error)[np.asarray(tr.live) > 0])
    print(f"saturated block: steps {float(st.n_steps)} rejects {float(st.n_rejects)} newton "
          f"{float(st.n_newton)} unconverged {float(st.n_unconverged)} step balance max {bal.max():.3e}")  # fmt: skip
    assert float(st.n_unconverged) == 0.0 and float(st.budget_exhausted) == 0.0
    assert bal.max() <= _tol_b(params), bal.max()
    assert abs(_closes(params, water, new, tot)) <= float(st.n_steps) * _tol_b(params)
