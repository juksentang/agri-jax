"""The two RZWQM2 conventions of ``soil_water.conventions``.

No data: the CA-TPA grid and soil of ``test_richards`` and synthetic profiles and forcing.

* DRAIN cap: equals a sequential transcription of the documented cascade (first node over
  ``pori + tol`` down, excess to the node below, bottom excess out), conserves water node by node,
  leaves no node at or below the first trigger above ``pori``, recomputes the head exactly on the
  changed nodes and leaves every other node bit for bit; a profile without excess is the identity;
* flux-mode limit: the analytic peak head gives the largest evaporation of the ghost-face Darcy
  flux over a dense grid of ghost heads; the limit is never below the ``h_min`` limit and equals it
  when ``m eps <= 1``;
* with the switches on, the fixed and the adaptive days close their water balance with the DRAIN
  seepage in the drainage, every accepted adaptive step converges, the profile ends each day at or
  below ``pori``; the flux-mode limit never delivers less evaporation than the ``h_min`` limit;
* the gradient through the cap (replay of a frozen step table) matches central differences;
* the switches are off by default; the convention variants are registered next to their faithful
  sibling, set the switches, and the faithful keys refuse a config with a switch on.

Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``DRAIN`` / ``CHKBC`` (conventions).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from agrijax.core.process import lookup
from agrijax.processes.soil_water import conventions as C
from agrijax.processes.soil_water import richards as R
from agrijax.processes.soil_water import richards_adaptive as A
from agrijax.processes.soil_water.day import (
    SoilWaterDayParams,
    soil_water_day,
    soil_water_day_drain_cap,
    soil_water_day_flux_evap,
    soil_water_day_kernel,
    soil_water_day_replay,
    soil_water_day_replay_conventions,
    soil_water_day_rzwqm2_conventions,
    with_conventions,
)
from agrijax.processes.soil_water.hydraulics import h_of_theta, k_of_h, theta_of_h
from agrijax.processes.soil_water.richards import (
    AdaptiveStepping,
    FixedStepping,
    RichardsConfig,
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
AEF = TD.AEF
N = 37


def _nodes():
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (N,)), catpa_soil().at_nodes())


def _pori() -> jax.Array:
    return C.field_saturation(_nodes(), jnp.asarray(AEF, FDT)).astype(FDT)


def _cascade_reference(theta: np.ndarray, tl: np.ndarray, pori: np.ndarray, tol: float):
    """A sequential transcription of the documented DRAIN cascade (NumPy, float64)."""
    th = theta.astype(float).copy()
    seep = 0.0
    first = next((i for i in range(len(th)) if th[i] - pori[i] > tol), None)
    if first is None:
        return th, seep
    for j in range(first, len(th)):
        d = th[j] - pori[j]
        if d > 0.0:
            th[j] = pori[j]
            if j + 1 < len(th):
                th[j + 1] = (th[j + 1] * tl[j + 1] + d * tl[j]) / tl[j + 1]
            else:
                seep = d * tl[j]
    return th, seep


def _wet_profile(seed: int, n_over: int = 6) -> tuple[jax.Array, jax.Array]:
    """A profile at 0.6-0.95 of pori with ``n_over`` nodes between pori and theta_s."""
    rng = np.random.default_rng(seed)
    pori = np.asarray(_pori(), float)
    ths = np.asarray(theta_of_h(jnp.zeros(N, FDT), _nodes()), float)
    th = pori * rng.uniform(0.6, 0.95, N)
    idx = rng.choice(N, n_over, replace=False)
    th[idx] = pori[idx] + rng.uniform(0.1, 1.0, n_over) * (ths[idx] - pori[idx])
    theta = jnp.asarray(th, FDT)
    return theta, h_of_theta(theta, _nodes())


# ---------------------------------------------------------------------------
# DRAIN cap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_drain_cap_is_the_sequential_cascade_and_conserves_water(seed: int) -> None:
    tl = catpa_grid().tl.astype(FDT)
    pori = _pori()
    theta, h = _wet_profile(seed)
    out = jax.jit(C.drain_cap)(theta, h, _nodes(), tl, pori)
    ref_th, ref_seep = _cascade_reference(
        np.asarray(theta, float), np.asarray(tl, float), np.asarray(pori, float), C.DRAIN_EXCESS_TOL
    )
    atol = 1e-14 if X64 else 2e-6
    np.testing.assert_allclose(np.asarray(out.theta, float), ref_th, rtol=0, atol=atol)
    assert float(out.seepage) == pytest.approx(ref_seep, abs=atol * 150)
    before = float(jnp.sum(theta * tl))
    after = float(jnp.sum(out.theta * tl)) + float(out.seepage)
    assert abs(after - before) <= (1e-12 if X64 else 5e-5)
    first = int(np.argmax(np.asarray(theta - pori) > C.DRAIN_EXCESS_TOL))
    assert np.all(np.asarray(out.theta - pori)[first:] <= (0.0 if X64 else 1e-7))
    changed = np.asarray(out.theta) != np.asarray(theta)
    assert changed.any()
    np.testing.assert_array_equal(np.asarray(out.h)[~changed], np.asarray(h)[~changed])
    np.testing.assert_array_equal(
        np.asarray(out.h)[changed], np.asarray(h_of_theta(out.theta, _nodes()))[changed]
    )


def test_drain_cap_without_excess_is_the_identity() -> None:
    tl = catpa_grid().tl.astype(FDT)
    pori = _pori()
    theta = pori * jnp.linspace(0.5, 1.0, N, dtype=FDT)  # the last node exactly at pori
    h = h_of_theta(theta, _nodes())
    out = C.drain_cap(theta, h, _nodes(), tl, pori)
    np.testing.assert_array_equal(np.asarray(out.theta), np.asarray(theta))
    np.testing.assert_array_equal(np.asarray(out.h), np.asarray(h))
    assert float(out.seepage) == 0.0


@pytest.mark.allow_skip(reason="the 1e-12 trigger is below float32 resolution")
@pytest.mark.skipif(not X64, reason="the 1e-12 trigger is below float32 resolution")
def test_drain_cap_starts_at_the_first_node_over_the_trigger() -> None:
    tl = catpa_grid().tl
    pori = _pori()
    theta = 0.8 * pori
    theta = theta.at[3].set(pori[3] + 0.5 * C.DRAIN_EXCESS_TOL)  # below the trigger: stays
    theta = theta.at[10].set(pori[10] + 0.01)  # the first node over it
    theta = theta.at[11].set(pori[11] + 0.25 * C.DRAIN_EXCESS_TOL)  # after it: capped
    out = C.drain_cap(theta, h_of_theta(theta, _nodes()), _nodes(), tl, pori)
    assert float(out.theta[3]) == float(theta[3])
    assert float(out.theta[10]) == float(pori[10])
    assert float(out.theta[11]) == float(pori[11])
    assert float(out.theta[12]) > float(theta[12])


# ---------------------------------------------------------------------------
# flux-mode evaporation limit
# ---------------------------------------------------------------------------


def _q_dry(ht0: float, eps_peak: bool, soil=None) -> float:
    soil = _nodes() if soil is None else soil
    h = jnp.full(N, ht0, FDT)
    th = theta_of_h(h, soil)
    a, _ = R._step_args(
        h, th, jnp.zeros((), FDT), soil, catpa_grid(), jnp.zeros((), FDT), jnp.asarray(1e3, FDT),
        jnp.zeros(N, FDT), jnp.asarray(0.1, FDT), jnp.ones((), FDT), jnp.asarray(R.H_CLAMP_RZWQM, FDT),
        RichardsConfig(evaporation_limit="flux_peak" if eps_peak else "hmin"), FixedStepping().h_upper,
    )  # fmt: skip
    _, _, q_dry = R.surface_fluxes(h, h, a)
    return float(q_dry)


@pytest.mark.parametrize("ht0", [-50.0, -400.0, -2000.0, -9000.0])
def test_flux_peak_is_the_largest_evaporation_over_the_ghost_head(ht0: float) -> None:
    soil = _nodes()
    dz = float(catpa_grid().dz_top)
    hmin = R.H_CLAMP_RZWQM
    s0 = jax.tree_util.tree_map(lambda x: x[:1], soil)
    k0 = float(k_of_h(jnp.asarray([ht0], FDT), s0)[0])
    hg = -np.geomspace(dz - ht0, -hmin, 200_001)
    s_g = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[:1], hg.shape), soil)
    kg = np.asarray(k_of_h(jnp.asarray(hg, FDT), s_g), float)
    q = -np.sqrt(k0 * kg) * ((ht0 - hg) / dz - 1.0)
    q_peak, q_hmin = _q_dry(ht0, True), _q_dry(ht0, False)
    rtol = 1e-9 if X64 else 1e-5
    assert q_peak <= q.min() * (1.0 - rtol)  # at least the grid's largest evaporation (q < 0)
    assert q_peak == pytest.approx(q.min(), rel=1e-7 if X64 else 1e-4)
    assert q_hmin == pytest.approx(q[-1], rel=1e-12 if X64 else 1e-5)
    assert q_peak <= q_hmin
    print(f"ht0 {ht0}: peak {q_peak:.6e} grid {q.min():.6e} h_min {q_hmin:.6e} cm/h")


def test_flux_peak_equals_the_h_min_limit_without_an_interior_peak() -> None:
    soil = _nodes()
    flat = soil.replace(eps=jnp.full(N, 1.8, FDT))  # m eps = 0.9 <= 1: monotone in the ghost head
    assert _q_dry(-500.0, True, flat) == _q_dry(-500.0, False, flat)


# ---------------------------------------------------------------------------
# the days with the switches on
# ---------------------------------------------------------------------------


def _wet_start() -> SoilWater:
    theta, _ = _wet_profile(7, n_over=10)
    return SoilWater.from_theta(theta, catpa_soil())


def _days(params: RichardsParams, water: SoilWater, n_day: int = 6, scale: float = 3.0):
    supply, evap, uptake = synthetic_forcing(n_day, seed=17)

    def body(w, f):
        w2 = richards_day(w, params, *f, aef=AEF)
        return w2, w2

    xs = tuple(jnp.asarray(x, FDT) for x in (supply * scale, evap, uptake))
    return jax.jit(lambda w: lax.scan(body, w, xs))(water)


@pytest.mark.parametrize("stepping", ["fixed", "adaptive"])
def test_days_with_both_conventions_close_the_balance_and_stay_below_pori(stepping: str) -> None:
    config = RichardsConfig(drain_cap=True, evaporation_limit="flux_peak")
    cfg = AdaptiveStepping.fast() if stepping == "adaptive" else FixedStepping(n_sub=48)
    params = RichardsParams(soil=catpa_soil(), grid=catpa_grid(), config=config, stepping=cfg)
    water = _wet_start()
    _, per = _days(params, water)
    fl = per.flux
    pori = np.asarray(_pori())
    assert float(jnp.sum(fl.drain_moved)) > 0.0
    assert np.all(np.asarray(fl.drain_seepage) <= np.asarray(fl.drainage))
    assert np.all(np.asarray(per.theta) <= pori + (0.0 if X64 else 1e-7))
    if stepping == "adaptive":
        assert isinstance(cfg, AdaptiveStepping)
        tol = cfg.newton_tol_balance if X64 else cfg.newton_tol_balance_f32
        assert float(jnp.sum(fl.n_unconverged)) == 0.0 and float(jnp.sum(fl.budget_exhausted)) == 0.0
        assert float(jnp.max(fl.step_balance_max)) <= tol
        assert np.all(np.abs(np.asarray(fl.balance_error)) <= np.asarray(fl.n_steps) * tol)
    print(f"{stepping}: drain seepage {np.asarray(fl.drain_seepage)} cm, day balance max "
          f"{float(jnp.max(jnp.abs(fl.balance_error))):.3e} cm")  # fmt: skip


def test_fixed_day_balance_with_the_cap_is_the_newton_residual() -> None:
    """The cap conserves water, so the fixed day's balance error is the one of its Newton solves:
    with the iterations run to convergence it is at rounding level."""
    kw = {"drain_cap": True, "evaporation_limit": "flux_peak"}
    off = RichardsParams(soil=catpa_soil(), grid=catpa_grid(), stepping=FixedStepping(n_sub=96, n_iter=20))
    on = dataclasses.replace(off, config=dataclasses.replace(off.config, **kw))
    _, p_on = _days(on, _wet_start())
    bal = float(jnp.max(jnp.abs(p_on.flux.balance_error)))
    print(f"fixed 96 x 20 with both conventions: day balance max {bal:.3e} cm")
    assert float(jnp.sum(p_on.flux.drain_moved)) > 0.0
    assert bal <= (1e-9 if X64 else 1e-3)


def test_flux_peak_never_evaporates_less_than_the_h_min_limit() -> None:
    """A dry surface under a large demand: the flux-mode limit delivers at least the h_min limit."""
    soil = catpa_soil()
    h = jnp.full(N, -300.0, FDT).at[:4].set(jnp.asarray([-12000.0, -8000.0, -3000.0, -1000.0], FDT))
    water = SoilWater.from_head(h, soil)
    evap = jnp.full(24, 0.05, FDT)
    zero24, zero_n = jnp.zeros(24, FDT), jnp.zeros(N, FDT)
    out = {}
    for lim in ("hmin", "flux_peak"):
        p = RichardsParams(
            soil=soil,
            grid=catpa_grid(),
            config=RichardsConfig(evaporation_limit=lim),
            stepping=AdaptiveStepping(),
        )
        out[lim] = jax.jit(lambda w, p=p: richards_day(w, p, zero24, evap, zero_n))(water).flux
    assert float(out["flux_peak"].evaporation) >= float(out["hmin"].evaporation)
    assert float(out["hmin"].evaporation_deficit) > 0.0
    print({k: (float(v.evaporation), float(v.evaporation_deficit)) for k, v in out.items()})


@pytest.mark.allow_skip(reason="central differences need float64")
@pytest.mark.skipif(not X64, reason="central differences need float64")
def test_gradient_through_the_cap_matches_central_differences_on_the_frozen_table() -> None:
    params = RichardsParams(
        soil=catpa_soil(),
        grid=catpa_grid(),
        config=RichardsConfig(drain_cap=True),
        stepping=AdaptiveStepping.fast(),
    )
    water = _wet_start()
    supply, evap, uptake = (jnp.asarray(x[0], FDT) for x in synthetic_forcing(1, seed=17))
    nodes = _nodes()

    def seg(aef):
        pori = C.field_saturation(nodes, aef)
        return A.adaptive_segment(water, params, 0.0, 24.0, supply, evap, uptake, water.dt_next, pori=pori)

    _, tot0, st0, tr = jax.jit(seg)(jnp.asarray(AEF))
    assert float(tot0.drain_moved) > 0.0 and float(st0.n_unconverged) == 0.0

    def frozen(aef):
        pori = C.field_saturation(nodes, aef)
        new, _ = A.replay_segment(water, params, supply, evap, uptake, tr, pori=pori)
        return new.storage(params.grid)

    def full(aef):
        new, _, _, _ = seg(aef)
        return new.storage(params.grid)

    g_full = float(jax.jit(jax.grad(full))(jnp.asarray(AEF)))
    g_frozen = float(jax.jit(jax.grad(frozen))(jnp.asarray(AEF)))
    e = 1e-6
    fd = (float(frozen(jnp.asarray(AEF + e))) - float(frozen(jnp.asarray(AEF - e)))) / (2 * e)
    assert np.isfinite(g_full) and g_full == pytest.approx(g_frozen, rel=1e-10)
    assert g_frozen == pytest.approx(fd, rel=1e-5, abs=1e-8)
    print(f"d storage / d aef: AD {g_frozen:.8e}, central difference {fd:.8e}")


# ---------------------------------------------------------------------------
# switches, registry keys and the faithful guard
# ---------------------------------------------------------------------------


VARIANTS = {
    "soil_water/day@rzwqm2-4.6:drain_cap": (soil_water_day_drain_cap, ("drain_cap",)),
    "soil_water/day@rzwqm2-4.6:flux_evap": (soil_water_day_flux_evap, ("flux_peak",)),
    "soil_water/day@rzwqm2-4.6:rzwqm2_conventions": (soil_water_day_rzwqm2_conventions, ("drain_cap", "flux_peak")),
    "soil_water/day@rzwqm2-4.6:replay_flux_conventions": (
        soil_water_day_replay_conventions,
        ("drain_cap", "flux_peak"),
    ),
}  # fmt: skip


def test_switches_are_off_by_default_and_checked() -> None:
    assert RichardsConfig().conventions == ()
    assert RichardsConfig(drain_cap=True, evaporation_limit="flux_peak").conventions == (
        "drain_cap",
        "flux_peak",
    )
    with pytest.raises(ValueError):
        RichardsConfig(evaporation_limit="peak")
    params = RichardsParams(soil=catpa_soil(), grid=catpa_grid(), config=RichardsConfig(drain_cap=True))
    w = SoilWater.from_theta(jnp.full(N, 0.2, FDT), catpa_soil())
    z = jnp.zeros(24, FDT)
    with pytest.raises(ValueError, match="aef"):
        richards_day(w, params, z, z, jnp.zeros(N, FDT))


@pytest.mark.parametrize("key", list(VARIANTS))
def test_variants_are_registered_next_to_the_faithful_key_and_set_the_switches(key: str) -> None:
    fn, on = VARIANTS[key]
    proc = lookup(key)
    assert proc.info is not None and proc.info.key.needs_faithful_sibling
    assert lookup(str(proc.info.key.faithful)) is not None
    assert proc.info.problems() == [] and len(proc.info.deviates) > 0
    params = with_conventions(TD._params(), drain="drain_cap" in on, flux_peak="flux_peak" in on)
    assert params.richards.config.conventions == on
    forcing = jax.tree_util.tree_map(lambda x: x[0], TD._forcing(1, seed=5, supply_scale=0.5))
    state = RichardsState(soil_water=_wet_start())
    new = fn(state, TD._params(), forcing)
    if "replay" in key:
        ref = richards_day(
            state.soil_water, params.richards, forcing.supply, forcing.evaporation, forcing.uptake, aef=AEF
        )
    else:
        ref, _, _ = soil_water_day_kernel(
            state.soil_water, params, forcing.supply, forcing.evaporation, forcing.uptake, forcing.storm
        )
    np.testing.assert_array_equal(np.asarray(new.soil_water.theta), np.asarray(ref.theta))
    assert ("drain_cap" in on) == (float(new.soil_water.flux.drain_moved) > 0.0)


def test_faithful_keys_refuse_the_switches() -> None:
    params = with_conventions(TD._params(), drain=True, flux_peak=False)
    forcing = jax.tree_util.tree_map(lambda x: x[0], TD._forcing(1, seed=5))
    state = RichardsState(soil_water=_wet_start())
    for fn in (soil_water_day, soil_water_day_replay):
        with pytest.raises(ValueError, match="convention"):
            fn(state, params, forcing)
    rp = params.richards
    with pytest.raises(ValueError, match="convention"):
        richards_redistribution(
            state,
            rp,
            R.RichardsForcing(supply=forcing.supply, evaporation=forcing.evaporation, uptake=forcing.uptake),
        )
    assert isinstance(params, SoilWaterDayParams)
