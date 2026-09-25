"""The soil-water day as two redistribution segments around one infiltration event (plan 19 A6).

* ``replay_flux`` (the M1 day) is bit-identical to :func:`richards_day`;
* the event day equals its explicit composition (pre segment -> event -> post segment) to
  rounding, and on a day without a storm it equals a single uniform segment;
* the ledger closes: ``d(storage + pond) = supply + rain - evaporation - drainage - uptake -
  runoff`` to 1e-10 cm at a converged configuration, with runoff and saturated-profile seepage;
* the processes run through the core runtime with ``AGRI_JAX_CHECK=1`` (declared writes and
  finite heads), in float32 and float64, with finite gradients.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from agrijax.core import Model
from agrijax.core.process import lookup
from agrijax.core.runtime import run
from agrijax.processes.soil_water import infiltration as G
from agrijax.processes.soil_water.day import (
    DayConfig,
    SoilWaterDayForcing,
    SoilWaterDayParams,
    day_edges,
    infiltration_ga,
    soil_water_day,
    soil_water_day_kernel,
    soil_water_day_replay,
)
from agrijax.processes.soil_water.richards import (
    RichardsConfig,
    RichardsParams,
    RichardsState,
    SoilWater,
    day_alphas,
    richards_day,
    richards_substeps,
)

from .test_richards import catpa_grid, catpa_soil, nodes, synthetic_forcing

X64 = bool(jax.config.read("jax_enable_x64"))
AEF = 0.9


def _params(day: DayConfig | None = None, **cfg) -> SoilWaterDayParams:
    grid = catpa_grid()
    rp = RichardsParams(soil=catpa_soil(), grid=grid, config=RichardsConfig(**cfg))
    gp = G.GreenAmptParams(aef=jnp.asarray(AEF), config=G.GreenAmptConfig.for_grid(np.asarray(grid.tl)))
    return SoilWaterDayParams(richards=rp, infiltration=gp, config=DayConfig() if day is None else day)


def _storms(n_day: int, seed: int = 3) -> G.StormForcing:
    """NumPy storms: ~40 % of days, 1-3 breakpoint intervals, start anywhere in the day."""
    rng = np.random.default_rng(seed)
    ts0 = np.full(n_day, 24.0)
    dur = np.zeros((n_day, 3))
    dep = np.zeros((n_day, 3))
    for d in range(n_day):
        if rng.random() < 0.4:
            k = int(rng.integers(1, 4))
            ts0[d] = rng.uniform(0.0, 20.0)
            dur[d, :k] = rng.uniform(0.1, 1.2, k)
            dep[d, :k] = rng.gamma(1.3, 0.8, k)
    return G.StormForcing(ts0=jnp.asarray(ts0), duration=jnp.asarray(dur), depth=jnp.asarray(dep))


def _forcing(n_day: int, seed: int = 3, supply_scale: float = 0.0) -> SoilWaterDayForcing:
    supply, evap, uptake = synthetic_forcing(n_day, seed=seed)
    return SoilWaterDayForcing(
        supply=jnp.asarray(supply * supply_scale),
        evaporation=jnp.asarray(evap),
        uptake=jnp.asarray(uptake),
        storm=_storms(n_day, seed),
    )


def _run_days(params: SoilWaterDayParams, w0: SoilWater, forcing: SoilWaterDayForcing):
    def body(w, f):
        w2, ev, shift = soil_water_day_kernel(w, params, f.supply, f.evaporation, f.uptake, f.storm)
        return w2, (w2, ev, shift)

    return jax.jit(lambda w, f: lax.scan(body, w, f))(w0, forcing)


def test_day_params_refuse_a_slice_grid_that_does_not_span_the_cells() -> None:
    grid = catpa_grid()
    rp = RichardsParams(soil=catpa_soil(), grid=grid, config=RichardsConfig())
    depth = float(np.sum(np.asarray(grid.tl)))
    bad = G.GreenAmptParams(aef=jnp.asarray(AEF), config=G.GreenAmptConfig(n_slice=round(depth) + 10))
    with pytest.raises(ValueError, match="slices span"):
        SoilWaterDayParams(richards=rp, infiltration=bad)
    assert isinstance(_params(), SoilWaterDayParams)


def test_day_edges_place_the_event() -> None:
    cfg = DayConfig(n_pre=4, n_post=8, post_grading=2.0)
    t_pre, t_post, t_ev = day_edges(jnp.asarray(6.0), jnp.asarray(True), cfg, jnp.float32)
    np.testing.assert_allclose(t_pre, np.linspace(0.0, 6.0, 5), rtol=1e-6)
    assert float(t_post[0]) == 6.0 and float(t_post[-1]) == 24.0 and float(t_ev) == 6.0
    widths = np.diff(np.asarray(t_post))
    assert np.all(widths > 0) and np.all(np.diff(widths) > 0)  # graded towards the event
    # no storm: the uniform steps of one segment
    t_pre, t_post, _ = day_edges(jnp.asarray(6.0), jnp.asarray(False), cfg, jnp.float32)
    np.testing.assert_allclose(np.concatenate([t_pre, t_post[1:]]), np.arange(0.0, 26.0, 2.0), rtol=1e-6)
    # n_pre = 0: the event is placed at midnight
    _, t_post, t_ev = day_edges(
        jnp.asarray(6.0), jnp.asarray(True), DayConfig(n_pre=0, n_post=6), jnp.float32
    )
    assert float(t_ev) == 0.0 and float(t_post[0]) == 0.0
    with pytest.raises(ValueError):
        DayConfig(n_pre=-1)
    with pytest.raises(ValueError):
        DayConfig(post_grading=0.5)


def test_replay_variant_is_bit_identical_to_the_m1_day() -> None:
    """The same scan over days, once with :func:`richards_day` (M1) and once with the replay process."""
    params = _params(n_sub=12, n_iter=3)
    forcing = _forcing(10, supply_scale=1.0)
    w0 = SoilWater.from_theta(jnp.full(37, 0.22), params.richards.soil)

    def m1(w, f):
        w2 = richards_day(w, params.richards, f.supply, f.evaporation, f.uptake)
        return w2, w2

    def replay(st, f):
        st2 = soil_water_day_replay(st, params, f)
        return st2, st2.soil_water

    _, a = jax.jit(lambda w, f: lax.scan(m1, w, f))(w0, forcing)
    _, b = jax.jit(lambda s, f: lax.scan(replay, s, f))(RichardsState(soil_water=w0), forcing)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # through the core runtime (a different compiled program): equal to rounding
    model = Model(RichardsState, [soil_water_day_replay], outputs=("soil_water.flux.drainage",))
    final, outs = jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(
        params, forcing, RichardsState(soil_water=w0)
    )
    tol = 1e-13 if X64 else 1e-6
    np.testing.assert_allclose(final.soil_water.theta, a.theta[-1], rtol=0, atol=tol)
    np.testing.assert_allclose(outs["soil_water.flux.drainage"], a.flux.drainage, rtol=10 * tol, atol=tol)


def test_event_day_equals_its_composition() -> None:
    params = _params(DayConfig(n_pre=5, n_post=7), n_sub=12, n_iter=3)
    rp = params.richards
    f = jax.tree_util.tree_map(lambda x: x[0], _forcing(1))
    storm = G.StormForcing(
        ts0=jnp.asarray(9.5), duration=jnp.asarray([0.4, 0.8, 0.0]), depth=jnp.asarray([1.5, 2.5, 0.0])
    )
    w0 = SoilWater.from_theta(jnp.linspace(0.12, 0.32, 37), rp.soil)
    new, _, shift = soil_water_day_kernel(w0, params, f.supply, f.evaporation, f.uptake, storm)
    # by hand
    t_pre, t_post, t_ev = day_edges(storm.ts0, jnp.asarray(True), params.config, w0.theta.dtype)
    alphas = day_alphas(12, rp.config, w0.theta.dtype)
    a, _ = richards_substeps(w0, rp, t_pre, f.supply, f.evaporation, f.uptake, alphas[:5])
    soil = nodes(rp.soil, 37)
    e = G.green_ampt_event(a.theta, a.h, soil, rp.grid.tl, jnp.asarray(AEF), storm.duration, storm.depth,
                           params.infiltration.config)  # fmt: skip
    b, _ = richards_substeps(
        a.replace(theta=e.theta, h=e.h), rp, t_post, f.supply, f.evaporation, f.uptake, alphas[5:]
    )
    # equal to rounding (the kernel and the hand composition compile to different programs)
    tol = 1e-14 if X64 else 1e-6
    np.testing.assert_allclose(np.asarray(new.theta), np.asarray(b.theta), rtol=tol, atol=tol)
    np.testing.assert_allclose(np.asarray(new.h), np.asarray(b.h), rtol=10 * tol)
    assert float(t_ev) == 9.5 and float(shift) == 0.0
    assert float(new.flux.rain) == pytest.approx(4.0) and float(new.flux.event_infiltration) == float(
        e.infiltration
    )
    # the event moved to midnight with n_pre = 0 reports the shift
    p0 = _params(DayConfig(n_pre=0, n_post=12), n_sub=12, n_iter=3)
    _, _, shift0 = soil_water_day_kernel(w0, p0, f.supply, f.evaporation, f.uptake, storm)
    assert float(shift0) == -9.5


def test_day_without_storm_equals_one_uniform_segment() -> None:
    params = _params(DayConfig(n_pre=6, n_post=18), n_sub=24, n_iter=4)
    f = jax.tree_util.tree_map(lambda x: x[0], _forcing(1))
    none = G.StormForcing(ts0=jnp.asarray(24.0), duration=jnp.zeros(3), depth=jnp.zeros(3))
    w0 = SoilWater.from_theta(jnp.linspace(0.12, 0.32, 37), params.richards.soil)
    new, ev, _ = soil_water_day_kernel(w0, params, f.supply * 0.0, f.evaporation, f.uptake, none)
    ref = richards_day(w0, params.richards, f.supply * 0.0, f.evaporation, f.uptake)
    # the same 24 uniform sub-steps; only the edge arithmetic differs (linspace vs interpolation)
    np.testing.assert_allclose(new.theta, ref.theta, rtol=0, atol=1e-13 if X64 else 2e-6)
    assert float(ev.rain) == 0.0 and float(new.flux.event_infiltration) == 0.0


@pytest.mark.allow_skip(reason="the 1e-10 cm ledger closure is a float64 claim")
def test_ledger_closes_with_events_runoff_and_seepage() -> None:
    if not X64:
        pytest.skip("needs float64")
    params = _params(DayConfig(n_pre=12, n_post=36), n_sub=48, n_iter=10)
    forcing = _forcing(40, supply_scale=0.3)
    # an intense storm (runoff) on day 2 and a very large one on day 5
    dep = np.asarray(forcing.storm.depth).copy()
    dur = np.asarray(forcing.storm.duration).copy()
    ts0 = np.asarray(forcing.storm.ts0).copy()
    dep[2], dur[2], ts0[2] = [9.0, 3.0, 0.0], [0.4, 0.2, 0.0], 3.0
    dep[5], dur[5], ts0[5] = [30.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0
    forcing = forcing.replace(
        storm=G.StormForcing(ts0=jnp.asarray(ts0), duration=jnp.asarray(dur), depth=jnp.asarray(dep))
    )
    w0 = SoilWater.from_theta(jnp.full(37, 0.25), params.richards.soil)
    _, (days, ev, _) = _run_days(params, w0, forcing)
    fl = days.flux
    tl = np.asarray(params.richards.grid.tl)
    storage = np.concatenate([[np.asarray(w0.theta) @ tl], np.asarray(days.theta) @ tl])
    dpond = np.diff(np.concatenate([[0.0], np.asarray(days.pond)]))
    supplied = np.asarray(forcing.supply).sum(axis=1) + dep.sum(axis=1)
    out = sum(np.asarray(getattr(fl, k)) for k in ("evaporation", "drainage", "uptake", "runoff"))
    err = np.diff(storage) + dpond - (supplied - out)
    assert np.max(np.abs(err)) < 1e-10, np.max(np.abs(err))
    assert np.max(np.abs(np.asarray(fl.balance_error))) < 1e-10
    assert float(np.sum(fl.n_clamp)) == 0.0
    assert float(np.max(np.abs(np.asarray(ev.error)))) < 1e-12
    # the storms really exercised the event terms
    assert float(fl.event_runoff[2]) > 1.0 and float(fl.event_runoff[5]) > 10.0
    np.testing.assert_allclose(fl.rain, dep.sum(axis=1), rtol=1e-12)
    np.testing.assert_array_less(
        np.asarray(fl.event_infiltration) - 1e-12, np.asarray(fl.infiltration) + 1e-9
    )
    assert np.all(np.asarray(fl.seepage) <= np.asarray(fl.drainage) + 1e-12)
    np.testing.assert_allclose(days.theta, jax.vmap(lambda h: nodes_theta(h, params))(days.h), atol=1e-12)


@pytest.mark.allow_skip(reason="the 1e-10 cm ledger closure is a float64 claim")
def test_ledger_closes_with_saturated_profile_seepage() -> None:
    if not X64:
        pytest.skip("needs float64")
    params = _params(DayConfig(n_pre=12, n_post=36), n_sub=48, n_iter=10)
    soil = params.richards.soil
    porav = np.asarray(nodes(soil, 37).theta_s) * AEF
    w0 = SoilWater.from_theta(jnp.asarray(porav - 0.003), soil)
    f = jax.tree_util.tree_map(lambda x: x[0], _forcing(1))
    storm = G.StormForcing(
        ts0=jnp.asarray(2.0), duration=jnp.asarray([1.0, 0.0, 0.0]), depth=jnp.asarray([4.0, 0.0, 0.0])
    )
    new, _, _ = soil_water_day_kernel(w0, params, f.supply * 0.0, f.evaporation, f.uptake, storm)
    fl = new.flux
    # measured: 0.75 cm of the 4 cm storm passes through as seepage after the profile saturates
    assert float(fl.seepage) > 0.5 and float(fl.drainage) > float(fl.seepage)
    ds = float(new.storage(params.richards.grid) - w0.storage(params.richards.grid)) + float(new.pond)
    budget = 4.0 - float(fl.evaporation) - float(fl.drainage) - float(fl.uptake) - float(fl.runoff)
    assert ds == pytest.approx(budget, abs=1e-10)
    assert abs(float(fl.balance_error)) < 1e-10


def nodes_theta(h, params):
    from agrijax.processes.soil_water.hydraulics import theta_of_h

    return theta_of_h(h, nodes(params.richards.soil, 37))


def test_processes_registered_and_run_through_the_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, fn in (
        ("soil_water/day@rzwqm2-4.6:faithful", soil_water_day),
        ("soil_water/day@rzwqm2-4.6:replay_flux", soil_water_day_replay),
        ("soil_water/infiltration_ga@rzwqm2-4.6:faithful", infiltration_ga),
    ):
        p = lookup(key)
        assert p is fn
        assert p.info is not None and not p.info.problems()
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")
    params = _params(DayConfig(n_pre=4, n_post=8), n_sub=12, n_iter=3)
    forcing = _forcing(12)
    w0 = SoilWater.from_theta(jnp.full(37, 0.22), params.richards.soil)
    model = Model(RichardsState, [soil_water_day], outputs=("soil_water.theta", "soil_water.flux.rain"))
    final, outs = jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(
        params, forcing, RichardsState(soil_water=w0)
    )
    _, (days, _, _) = _run_days(params, w0, forcing)
    np.testing.assert_allclose(final.soil_water.theta, days.theta[-1], rtol=0, atol=1e-13 if X64 else 1e-6)
    np.testing.assert_allclose(
        outs["soil_water.flux.rain"], np.asarray(forcing.storm.depth).sum(axis=1), rtol=1e-6
    )
    # eager: the event process writes only what it declares
    st = RichardsState(soil_water=w0)
    f0 = jax.tree_util.tree_map(
        lambda x: x[int(np.argmax(np.asarray(forcing.storm.depth).sum(axis=1) > 0))], forcing
    )
    new = infiltration_ga(st, params, f0)
    changed = infiltration_ga.check_writes(st, new)
    assert changed and set(changed) <= {
        "soil_water.h", "soil_water.theta", "soil_water.flux.rain", "soil_water.flux.event_infiltration",
        "soil_water.flux.event_runoff", "soil_water.flux.seepage",
    }  # fmt: skip
    new_day = soil_water_day(st, params, f0)
    assert all(c.startswith("soil_water.") for c in soil_water_day.check_writes(st, new_day))


def test_event_day_float32_and_gradients() -> None:
    params = _params(DayConfig(n_pre=6, n_post=18), n_sub=24, n_iter=3)
    forcing = _forcing(6, seed=8)

    def loss(ksat, aef):
        p = params.replace(
            richards=params.richards.replace(soil=params.richards.soil.replace(ksat=ksat)),
            infiltration=params.infiltration.replace(aef=aef),
        )
        w0 = SoilWater.from_theta(jnp.full(37, 0.2, ksat.dtype), p.richards.soil)
        w, (days, _, _) = _run_days(p, w0, forcing)
        return jnp.sum(days.flux.drainage) + w.storage(p.richards.grid) + jnp.sum(days.flux.event_runoff)

    ks = params.richards.soil.ksat
    g_ks, g_aef = jax.grad(loss, argnums=(0, 1))(ks, jnp.asarray(AEF))
    assert np.all(np.isfinite(np.asarray(g_ks))) and np.isfinite(float(g_aef))
    assert float(jnp.abs(g_ks).sum()) > 0.0

    def storage(x64: bool) -> np.ndarray:
        with jax.enable_x64(x64):
            dt = np.float64 if x64 else np.float32
            p = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dt) if hasattr(a, "dtype") else a, params)
            f = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dt), forcing)
            w0 = SoilWater.from_theta(jnp.full(37, 0.2, dt), p.richards.soil)
            _, (days, _, _) = _run_days(p, w0, f)
            assert days.theta.dtype == dt
            return np.asarray(days.theta, np.float64) @ np.asarray(p.richards.grid.tl, np.float64)

    s64, s32 = storage(True), storage(False)
    assert np.all(np.isfinite(s32))
    assert np.max(np.abs(s32 - s64)) < 2e-4
