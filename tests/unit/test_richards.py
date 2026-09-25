"""Richards redistribution (``processes/soil_water/richards.py``): conservation, independent solutions, convergence.

Independent references used here, none derived from RZWQM output:

* **conservation**: the change of profile storage (``sum(theta * tl)`` of the state, summed with
  NumPy) equals the boundary fluxes and the sink over every day, to 1e-10 cm in float64 at a
  converged configuration; an unconverged configuration must report its imbalance honestly;
* **dense automatic differentiation**: the three-colour tridiagonal Jacobian equals
  ``jax.jacfwd`` of the residual;
* **steady layered profile**: the steady downward-flux solution of Darcy's law,
  ``dh/dz = 1 - q / K(h)``, integrated with ``scipy.integrate.solve_ivp`` from the free-drainage
  bottom (``K(h) = q``) through the five CA-TPA horizons; the error must fall with the grid size;
* **Philip early-time infiltration**: cumulative ponded infiltration ``I = S t^1/2 + A t`` with the
  sorptivity ``S`` of Parlange (1975), ``S^2 = int_{h_i}^0 (theta_s + theta - 2 theta_i) K dh``,
  computed by quadrature;
* **convergence**: sub-steps x iterations against a near-converged 96 x 8 run on NumPy-generated
  forcing (the CA-TPA 2015 version of the table is in ``tests/integration/test_richards_catpa.py``);
* **float32 vs float64** on the same NumPy inputs.

Source: Celia et al. (1990); Philip (1957); Parlange (1975); Ahuja et al. (2000) ch. 3.
"""

from __future__ import annotations

import itertools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from agrijax.core import Model
from agrijax.core.process import registry
from agrijax.core.runtime import run
from agrijax.processes.soil_water import richards as R
from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams, k_of_h, theta_of_h
from agrijax.processes.soil_water.richards import (
    RichardsConfig,
    RichardsForcing,
    RichardsGrid,
    RichardsParams,
    RichardsState,
    SoilWater,
    head_of_v,
    richards_day,
    richards_redistribution,
    substep_edges,
    tridiagonal_jacobian,
    v_of_head,
)

X64 = bool(jax.config.read("jax_enable_x64"))
#: relative tolerance of identities that hold to rounding (float64 / float32 unit-tier pass)
REL = 1e-12 if X64 else 2e-5

# CA-TPA grid (rzwqm.dat node records: layer bottom, distance to the next node) and horizons
CATPA_TLT = np.array(
    [1, 2, 4, 7, 11, 15, 19, 23, 26, 30, 34, 38, 43, 48, 53, 58, 63, 67, 70, 73, 77, 82, 86, 90, 94, 98,
     103, 108, 113, 118, 123, 128, 133, 138, 143, 147, 150], dtype=float
)  # fmt: skip
CATPA_DELZ = np.array(
    [1, 1, 3, 3, 5, 3, 5, 3, 3, 5, 3, 5, 5, 5, 5, 5, 5, 3, 3, 3, 5, 5, 3, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 3, 0], dtype=float
)  # fmt: skip
CATPA_HORIZON_BOTTOM = np.array([15.0, 30.0, 70.0, 90.0, 150.0])
CATPA_REC1 = np.array(
    [
        [14.6545, 0.22, 2.966, 5.41, 0.055, 0.453],
        [14.6545, 0.26, 2.966, 3.16, 0.032, 0.453],
        [14.6545, 0.36, 2.966, 3.31, 0.043, 0.453],
        [14.6545, 0.17, 2.966, 3.32, 0.048, 0.453],
        [14.6545, 0.322, 2.966, 2.59, 0.041, 0.453],
    ]
)
CATPA_REC2 = np.tile([0.0, 0.0, 0.0, 14.6545, 7440.01, 0.0, 0.0], (5, 1))


def catpa_grid() -> RichardsGrid:
    return RichardsGrid.from_rzwqm(CATPA_TLT, CATPA_DELZ)


def catpa_soil() -> SoilHydraulicParams:
    nh = np.searchsorted(CATPA_HORIZON_BOTTOM, CATPA_TLT, side="left")
    return SoilHydraulicParams.from_rzwqm_records(CATPA_REC1, CATPA_REC2, node_horizon=nh)


def synthetic_forcing(n_day: int, seed: int = 20260924) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy-generated daily inputs: rain events of 1-4 h, daytime evaporation demand, root uptake."""
    rng = np.random.default_rng(seed)
    supply = np.zeros((n_day, 24))
    for d in range(n_day):
        if rng.random() < 0.35:
            amount = rng.gamma(1.5, 0.8)  # cm
            dur = int(rng.integers(1, 5))
            start = int(rng.integers(0, 24 - dur))
            supply[d, start : start + dur] = amount / dur
    hours = np.arange(24)
    day_shape = np.where((hours >= 6) & (hours < 18), np.sin(np.pi * (hours - 6 + 0.5) / 12), 0.0)
    evap = np.outer(rng.uniform(0.05, 0.3, n_day), day_shape / day_shape.sum())
    uptake = np.zeros((n_day, 37))
    uptake[:, :14] = (
        rng.uniform(0.0, 0.25, (n_day, 1)) * np.linspace(2.0, 0.2, 14) / np.linspace(2.0, 0.2, 14).sum()
    )
    return supply, evap, uptake


def run_days(params: RichardsParams, water: SoilWater, supply, evap, uptake):
    """Scan ``richards_day`` over the days; returns the final state and the per-day states."""
    forcing = RichardsForcing(
        supply=jnp.asarray(supply), evaporation=jnp.asarray(evap), uptake=jnp.asarray(uptake)
    )

    def body(w, f):
        w2 = richards_day(w, params, f.supply, f.evaporation, f.uptake)
        return w2, w2

    return jax.jit(lambda w, f: lax.scan(body, w, f))(water, forcing)


def step_args(h, theta, soil, grid, q_demand, dt=1.0, alpha=1.0, sink=None):
    n = h.shape[0]
    return R._StepArgs(
        soil=soil,
        tl=grid.tl,
        delz=grid.delz,
        dz_top=grid.dz_top,
        theta_old=theta,
        h_old=h,
        q_demand=jnp.asarray(q_demand),
        sink=jnp.zeros(n) if sink is None else sink,
        dt=jnp.asarray(dt),
        alpha=jnp.asarray(alpha),
        h_min=jnp.asarray(-15000.0),
        h_hi=10.0 + grid.node_depth(),
    )


def nodes(soil: SoilHydraulicParams, n: int) -> SoilHydraulicParams:
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), soil.at_nodes())


# ---------------------------------------------------------------------------
# grid, transform, schedule, configuration
# ---------------------------------------------------------------------------


def test_catpa_grid_is_vertex_centred() -> None:
    g = catpa_grid()
    tl = np.asarray(g.tl)
    assert g.n_node == 37
    assert tl.sum() == pytest.approx(150.0, rel=REL)
    np.testing.assert_allclose(tl[:6], [1, 1, 2, 3, 4, 4])
    assert tl[-1] == 3.0
    zn = np.asarray(g.node_depth())
    np.testing.assert_allclose(zn[:6], [0.5, 1.5, 2.5, 5.5, 8.5, 13.5])
    assert zn[-1] == pytest.approx(148.5)
    # every cell boundary lies midway between its two nodes
    np.testing.assert_allclose(np.cumsum(tl)[:-1], 0.5 * (zn[:-1] + zn[1:]))
    assert float(g.dz_top) == 1.0


def test_grid_rejects_inconsistent_records() -> None:
    bad = CATPA_DELZ.copy()
    bad[5] = 4.0
    with pytest.raises(ValueError, match="vertex-centred"):
        RichardsGrid.from_rzwqm(CATPA_TLT, bad)


def test_config_validation() -> None:
    for kw in (
        {"n_sub": 0},
        {"jacobian": "lu"},
        {"time_scheme": "cn"},
        {"grad": "fd"},
        {"rain_fraction": 1.0},
    ):
        with pytest.raises(ValueError):
            RichardsConfig(**kw)
    with pytest.raises(ValueError):
        RichardsConfig(c_floor=-1.0)


def test_transformed_variable_round_trip_and_c1() -> None:
    s = jnp.asarray(14.6545)
    h = jnp.asarray(np.concatenate([-np.logspace(4.2, -3, 400), np.linspace(0.0, 10.0, 11)]))
    np.testing.assert_allclose(head_of_v(v_of_head(h, s), s), h, rtol=REL, atol=1e-10 if X64 else 1e-6)
    # value and slope continuous at v = 0 (h = -hb)
    d = jax.vmap(jax.grad(lambda v: head_of_v(v, s)))(jnp.asarray([-1e-9, 1e-9]))
    np.testing.assert_allclose(d, [14.6545, 14.6545], rtol=1e-6 if X64 else 1e-4)


def test_substep_edges() -> None:
    dry = substep_edges(jnp.zeros(24), 24, 0.5)
    np.testing.assert_allclose(dry, np.arange(25.0), atol=24 * REL)
    rain = jnp.zeros(24).at[:2].set(1.2)
    t = np.asarray(substep_edges(rain, 24, 0.5))
    assert t[0] == 0.0 and t[-1] == 24.0 and np.all(np.diff(t) > 0)
    assert np.sum(t[1:] <= 2.0 + 24 * REL) >= 12  # half of the sub-steps inside the two rain hours
    assert np.max(np.diff(t)) <= 24.0 / (0.5 * 24) + 24 * REL
    # the rates averaged over the schedule keep the daily totals
    means = R._interval_means(rain, jnp.asarray(t))
    assert float(jnp.sum(means * jnp.diff(jnp.asarray(t)))) == pytest.approx(2.4, rel=REL)


# ---------------------------------------------------------------------------
# residual, Jacobian, sign conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q_demand", [0.0, 0.5, 50.0, -0.3, -50.0])
@pytest.mark.parametrize("alpha", [1.0, 0.5])
def test_tridiagonal_jacobian_equals_dense_jacfwd(q_demand: float, alpha: float) -> None:
    grid = catpa_grid()
    soil = nodes(catpa_soil(), 37)
    rng = np.random.default_rng(3)
    h_old = -jnp.asarray(np.exp(rng.uniform(np.log(20.0), np.log(3000.0), 37)))
    h = h_old * jnp.asarray(rng.uniform(0.7, 1.3, 37))
    a = step_args(h_old, theta_of_h(h_old, soil), soil, grid, q_demand, dt=0.5, alpha=alpha)
    f = lambda x: R.richards_residual(x, x, a)  # noqa: E731
    r, dl, d, du = tridiagonal_jacobian(f, h)
    dense = np.asarray(jax.jacfwd(f)(h))
    band = np.diag(np.asarray(d)) + np.diag(np.asarray(dl)[1:], -1) + np.diag(np.asarray(du)[:-1], 1)
    np.testing.assert_allclose(band, dense, rtol=REL, atol=(1e-14 if X64 else 1e-6) * np.abs(dense).max())
    np.testing.assert_allclose(r, f(h), rtol=0, atol=0)
    assert float(dl[0]) == 0.0 and float(du[-1]) == 0.0


def test_darcy_signs_uniform_and_hydrostatic_profiles() -> None:
    grid = catpa_grid()
    soil = nodes(catpa_soil(), 37)
    zn = grid.node_depth()
    # uniform head: unit gradient drainage q = +K (downward) on every interior face and at the bottom
    h = jnp.full(37, -100.0)
    a = step_args(h, theta_of_h(h, soil), soil, grid, 0.0)
    q = R._face_fluxes(h, h, a)
    k = k_of_h(h, soil)
    kk = np.asarray(k)
    np.testing.assert_allclose(q[1:-1], np.sqrt(kk[:-1] * kk[1:]), rtol=REL)
    assert float(q[-1]) == pytest.approx(kk[-1], rel=REL)
    assert np.all(np.asarray(q[1:]) > 0)
    # hydrostatic (total head constant, h = z - const): no interior flux
    hs = zn - 200.0
    a = step_args(hs, theta_of_h(hs, soil), soil, grid, 0.0)
    q = R._face_fluxes(hs, hs, a)
    np.testing.assert_allclose(q[1:-1], 0.0, atol=1e-15)


def test_surface_limits() -> None:
    """Demand inside the limits is applied as given; outside, the Dirichlet limit is the flux."""
    grid = catpa_grid()
    soil = nodes(catpa_soil(), 37)
    h = jnp.full(37, -300.0)
    th = theta_of_h(h, soil)
    ks = float(soil.ksat[0])
    for qd in (0.2, -0.01):
        qt, qw, qdry = R.surface_fluxes(h, h, step_args(h, th, soil, grid, qd))
        assert float(qt) == pytest.approx(qd, rel=REL)
    qt, qw, _ = R.surface_fluxes(h, h, step_args(h, th, soil, grid, 1e4))
    assert float(qt) == float(qw) == pytest.approx(ks * (300.0 / 1.0 + 1.0), rel=REL)
    qt, _, qdry = R.surface_fluxes(h, h, step_args(h, th, soil, grid, -1e4))
    s1 = _h1(soil)
    kd = np.sqrt(float(k_of_h(jnp.asarray(-15000.0), s1)) * float(k_of_h(jnp.asarray(-300.0), s1)))
    assert (
        float(qt)
        == float(qdry)
        == pytest.approx(-kd * ((-300.0 + 15000.0) / 1.0 - 1.0), rel=1e-10 if X64 else 1e-4)
    )


def _h1(soil: SoilHydraulicParams) -> SoilHydraulicParams:
    return jax.tree_util.tree_map(lambda x: x[0], soil)


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------


@pytest.mark.allow_skip(reason="the 1e-10 cm closure is a float64 claim; float32 is covered below")
def test_mass_balance_converged_float64() -> None:
    if not X64:
        pytest.skip("needs float64")
    grid, soil = catpa_grid(), catpa_soil()
    supply, evap, uptake = synthetic_forcing(30)
    params = RichardsParams(soil=soil, grid=grid, config=RichardsConfig(n_sub=48, n_iter=8))
    w0 = SoilWater.from_theta(jnp.full(37, 0.25), soil)
    _, days = run_days(params, w0, supply, evap, uptake)
    tl = np.asarray(grid.tl)
    storage = np.concatenate([[np.asarray(w0.theta) @ tl], np.asarray(days.theta) @ tl])
    fl = days.flux
    supplied = supply.sum(axis=1)
    out = np.asarray(fl.evaporation) + np.asarray(fl.drainage) + np.asarray(fl.uptake) + np.asarray(fl.runoff)
    dpond = np.diff(np.concatenate([[0.0], np.asarray(days.pond)]))
    err = np.diff(storage) + dpond - (supplied - out)
    assert np.max(np.abs(err)) < 1e-10, np.max(np.abs(err))
    assert np.max(np.abs(np.asarray(fl.balance_error))) < 1e-10
    assert float(np.max(fl.max_theta_residual)) < 1e-10
    assert float(np.sum(fl.n_clamp)) == 0.0
    # the state is consistent: theta = theta(h)
    np.testing.assert_allclose(days.theta, theta_of_h(days.h, soil), rtol=0, atol=1e-15)
    # the inputs were really used
    assert np.sum(fl.infiltration) == pytest.approx(supplied.sum(), rel=1e-12)
    assert float(np.sum(fl.evaporation)) > 0.5 * evap.sum()


@pytest.mark.parametrize(("n_sub", "n_iter"), [(6, 1), (12, 2), (24, 3)])
def test_unconverged_balance_is_reported_not_hidden(n_sub: int, n_iter: int) -> None:
    """With few iterations the imbalance is non-zero; ``balance_error`` must equal the true bookkeeping error."""
    grid, soil = catpa_grid(), catpa_soil()
    supply, evap, uptake = synthetic_forcing(20)
    params = RichardsParams(soil=soil, grid=grid, config=RichardsConfig(n_sub=n_sub, n_iter=n_iter))
    w0 = SoilWater.from_theta(jnp.full(37, 0.25), soil)
    _, days = run_days(params, w0, supply, evap, uptake)
    tl = np.asarray(grid.tl, dtype=np.float64)
    storage = np.concatenate(
        [[np.asarray(w0.theta, np.float64) @ tl], np.asarray(days.theta, np.float64) @ tl]
    )
    fl = days.flux
    out = sum(np.asarray(getattr(fl, k), np.float64) for k in ("evaporation", "drainage", "uptake", "runoff"))
    dpond = np.diff(np.concatenate([[0.0], np.asarray(days.pond, np.float64)]))
    err = np.diff(storage) + dpond - (supply.sum(axis=1) - out)
    tol = 1e-9 if X64 else 2e-4
    np.testing.assert_allclose(np.asarray(fl.balance_error, np.float64), err, atol=tol, rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(days.h)))
    assert float(np.sum(fl.n_clamp)) == 0.0


def test_supply_limited_evaporation() -> None:
    """A dry surface cannot meet a large demand: the deficit is reported and equals demand - actual."""
    grid, soil = catpa_grid(), catpa_soil()
    params = RichardsParams(soil=soil, grid=grid, config=RichardsConfig(n_sub=48, n_iter=6))
    w = SoilWater.from_head(jnp.full(37, -8000.0), soil)
    demand = np.full(24, 1.0 / 24)  # 1 cm/d
    w1 = richards_day(w, params, jnp.zeros(24), jnp.asarray(demand), jnp.zeros(37))
    f = w1.flux
    assert 0.0 < float(f.evaporation) < 0.5
    assert float(f.evaporation) + float(f.evaporation_deficit) == pytest.approx(1.0, rel=REL)
    # a wet surface meets the same demand in full
    w = SoilWater.from_head(jnp.full(37, -50.0), soil)
    f = richards_day(w, params, jnp.zeros(24), jnp.asarray(demand), jnp.zeros(37)).flux
    assert float(f.evaporation) == pytest.approx(1.0, rel=REL)
    assert float(f.evaporation_deficit) == 0.0


def test_ponding_and_runoff() -> None:
    """Supply above the infiltration capacity runs off (pond_max = 0) or is stored and infiltrates later.

    The storm saturates the whole profile; the layered saturated profile then carries positive
    heads up to ~24 cm (above the old fixed clamp of +10 cm, which made the solve diverge on
    some machines), and after the storm the saturated surface drains across the air-entry kink.
    """
    grid, soil = catpa_grid(), catpa_soil()
    storm = np.zeros(24)
    storm[:1] = 40.0  # 40 cm in one hour on a nearly saturated profile
    w = SoilWater.from_head(jnp.full(37, -16.0), soil)
    tol = 1e-9 if X64 else 1e-3
    for n_iter, tol_iter in ((20, tol), (8, 1e-4 if X64 else 1e-3)):
        cfg = RichardsConfig(n_sub=96, n_iter=n_iter)
        f = richards_day(
            w,
            RichardsParams(soil=soil, grid=grid, config=cfg),
            jnp.asarray(storm),
            jnp.zeros(24),
            jnp.zeros(37),
        )
        assert float(f.flux.runoff) > 10.0
        assert float(f.pond) == 0.0
        assert float(f.flux.infiltration) + float(f.flux.runoff) == pytest.approx(40.0, abs=tol)
        assert abs(float(f.flux.balance_error)) < tol_iter, (n_iter, float(f.flux.balance_error))
        assert float(f.flux.n_clamp) == 0.0
        # with a 5 cm pond the excess of the storm hour is held and infiltrates in the following hours;
        # the pond-emptying sub-step drains a saturated surface across the air-entry kink (measured: 8
        # iterations leave 3e-6 cm with the kink chop, 0.015 cm without it)
        pp = RichardsParams(soil=soil, grid=grid, pond_max=5.0, config=cfg)
        g = richards_day(w, pp, jnp.asarray(storm), jnp.zeros(24), jnp.zeros(37))
        assert float(g.flux.infiltration) + float(g.flux.runoff) + float(g.pond) == pytest.approx(
            40.0, abs=tol
        )
        assert float(g.flux.infiltration) - float(f.flux.infiltration) == pytest.approx(5.0, abs=0.05)
        assert float(g.pond) == pytest.approx(0.0, abs=tol)
        assert abs(float(g.flux.balance_error)) < tol_iter, (n_iter, float(g.flux.balance_error))
    if X64:
        no_chop = RichardsParams(
            soil=soil, grid=grid, pond_max=5.0, config=RichardsConfig(n_sub=96, n_iter=8, chop=False)
        )
        g = richards_day(w, no_chop, jnp.asarray(storm), jnp.zeros(24), jnp.zeros(37))
        assert abs(float(g.flux.balance_error)) > 1e-3  # the failure the chop removes


@pytest.mark.allow_skip(reason="the 1e-8 cm comparison with the linear march needs float64")
def test_saturated_layered_steady_state_has_positive_heads() -> None:
    """Ponded, fully saturated CA-TPA profile at steady state against the linear march of Darcy's law.

    Saturated (``h >= -hb_k``, ``n1 = 0``) every node has ``K = K_s`` of its horizon, so the steady
    flux is the bottom one, ``q = K_s,bottom`` (unit gradient), and the heads follow from the top:
    ``h_0 = dz_top (1 - q / K_s,0)`` (ghost head 0), ``h_{i+1} = h_i + delz_i (1 - q / K_face,i)``
    with the geometric-mean face conductances. The deepest heads are ~24 cm, above +10 cm.
    """
    if not X64:
        pytest.skip("needs float64")
    grid = catpa_grid()
    soil = nodes(catpa_soil(), 37)
    ks = np.asarray(soil.ksat)
    q = ks[-1]
    h_ref = np.empty(37)
    h_ref[0] = float(grid.dz_top) * (1.0 - q / ks[0])
    k_face = np.sqrt(ks[:-1] * ks[1:])
    h_ref[1:] = h_ref[0] + np.cumsum(np.asarray(grid.delz) * (1.0 - q / k_face))
    assert h_ref.max() > 20.0 and np.all(h_ref > -np.asarray(soil.hb_k))
    h = jnp.zeros(37)
    r = R.richards_step(h, theta_of_h(h, soil), jnp.asarray(0.0), soil, grid, jnp.asarray(1e3), jnp.asarray(0.0),
                        jnp.zeros(37), jnp.asarray(1e6), jnp.asarray(1.0), jnp.asarray(-15000.0), jnp.asarray(0.0),
                        RichardsConfig(n_iter=30))  # fmt: skip
    np.testing.assert_allclose(r.h, h_ref, rtol=0, atol=1e-8)
    assert float(r.n_clamp) == 0.0
    assert float(r.drainage) / 1e6 == pytest.approx(q, rel=1e-10)


def test_uptake_is_capped_at_h_min() -> None:
    grid, soil = catpa_grid(), catpa_soil()
    params = RichardsParams(soil=soil, grid=grid, config=RichardsConfig(n_sub=24, n_iter=6))
    w = SoilWater.from_head(jnp.full(37, -14000.0), soil)
    uptake = jnp.zeros(37).at[:6].set(0.5)  # far more than the water above theta(h_min) in the top cells
    out = richards_day(w, params, jnp.zeros(24), jnp.zeros(24), uptake)
    f = out.flux
    assert float(f.uptake_cut) > 0.0
    assert float(f.uptake) + float(f.uptake_cut) == pytest.approx(
        float(jnp.sum(uptake)), rel=1e-10 if X64 else 1e-5
    )
    assert float(jnp.min(out.h)) >= -15000.0 * (1 + REL)
    # nodes end within 1e-9 of theta(h_min): the iterate touches the dry clamp there, as RZWQM's
    # H = MAX(H, HMIN) does, and the clamp is counted; its mass error is tiny (measured 1.5e-7 cm)
    assert float(f.n_clamp) > 0.0
    assert abs(float(f.balance_error)) < (1e-6 if X64 else 1e-3)


# ---------------------------------------------------------------------------
# independent solutions
# ---------------------------------------------------------------------------


def _layered_uniform(dz: float) -> tuple[RichardsGrid, SoilHydraulicParams, np.ndarray]:
    n = round(150.0 / dz)
    grid = RichardsGrid.uniform(n, dz)
    zc = (np.arange(n) + 0.5) * dz
    nh = np.searchsorted(CATPA_HORIZON_BOTTOM, zc, side="left")
    soil = SoilHydraulicParams.from_rzwqm_records(CATPA_REC1, CATPA_REC2, node_horizon=nh)
    return grid, nodes(soil, n), zc


def _steady_reference(zc: np.ndarray, q: float) -> tuple[float, np.ndarray]:
    horizons = [SoilHydraulicParams.from_rzwqm_records(CATPA_REC1[i], CATPA_REC2[i]) for i in range(5)]

    def k(h: float, i: int) -> float:
        return float(k_of_h(jnp.asarray(h), horizons[i]))

    h_bot = brentq(lambda h: k(h, 4) - q, -1e4, -1e-6, xtol=1e-13)

    def rhs(z: float, h: np.ndarray) -> list[float]:
        i = min(int(np.searchsorted(CATPA_HORIZON_BOTTOM, z, side="left")), 4)
        return [1.0 - q / k(float(h[0]), i)]

    sol = solve_ivp(rhs, [zc[-1], zc[0]], [h_bot], t_eval=zc[::-1], rtol=1e-11, atol=1e-10, max_step=0.5)
    assert sol.success
    return h_bot, sol.y[0][::-1]


@pytest.mark.allow_skip(reason="the ODE comparison at 1e-2 cm needs float64")
def test_steady_layered_profile_matches_darcy_ode() -> None:
    if not X64:
        pytest.skip("needs float64")
    q = 0.05  # cm/h downward
    errors = []
    for dz in (1.0, 0.5):
        grid, soil, zc = _layered_uniform(dz)
        h_bot, h_ref = _steady_reference(zc, q)
        h = jnp.full(zc.shape, h_bot)
        cfg = RichardsConfig(n_iter=60)
        for _ in range(2):  # one huge implicit step is the steady state; the second confirms it
            r = R.richards_step(h, theta_of_h(h, soil), jnp.asarray(0.0), soil, grid, jnp.asarray(q),
                                jnp.asarray(0.0), jnp.zeros(zc.shape), jnp.asarray(1e6), jnp.asarray(1.0),
                                jnp.asarray(-15000.0), jnp.asarray(0.0), cfg)  # fmt: skip
            h = r.h
        assert float(r.drainage) / 1e6 == pytest.approx(q, rel=1e-9)  # steady: out = in
        errors.append(float(np.max(np.abs(np.asarray(h) - h_ref))))
    # measured 0.026 cm (dz = 1) and 0.014 cm (dz = 0.5) on a profile from -64.4 to -55.5 cm head
    assert errors[0] < 0.04 and errors[1] < 0.02
    assert (
        errors[1] < 0.65 * errors[0]
    )  # first-order convergence (geometric-mean faces at the horizon breaks)


def _parlange_sorptivity(soil1: SoilHydraulicParams, h_i: float) -> float:
    th = lambda h: float(theta_of_h(jnp.asarray(h), soil1))  # noqa: E731
    kk = lambda h: float(k_of_h(jnp.asarray(h), soil1))  # noqa: E731
    ti, ts = th(h_i), th(0.0)
    f = lambda u: (ts + th(-np.exp(u)) - 2 * ti) * kk(-np.exp(u)) * np.exp(u)  # noqa: E731  (h = -e^u)
    s2 = quad(f, np.log(1e-8), np.log(-h_i), limit=400, epsabs=0.0, epsrel=1e-11)[0]
    return float(np.sqrt(s2))


@pytest.mark.allow_skip(reason="the 1 % sorptivity check needs float64")
@pytest.mark.parametrize(
    "rec1",
    [
        [14.6545, 0.22, 2.966, 5.41, 0.055, 0.453],  # CA-TPA horizon 1 (loamy sand)
        [37.3, 0.131, 2.393, 0.5, 0.09, 0.475],  # a fine-textured curve
    ],
)
def test_philip_early_time_infiltration(rec1: list[float]) -> None:
    if not X64:
        pytest.skip("needs float64")
    rec2 = [0.0, 0.0, 0.0, rec1[0], 0.0, 0.0, 0.0]
    soil1 = SoilHydraulicParams.from_rzwqm_records(np.asarray(rec1), np.asarray(rec2))
    h_i, dz, dt, n = -800.0, 0.05, 5e-4, 600
    grid = RichardsGrid.uniform(n, dz)
    soil = nodes(soil1, n)
    cfg = RichardsConfig(n_iter=12)
    h0 = jnp.full(n, h_i)

    def body(c, _):
        h, th = c
        r = R.richards_step(h, th, jnp.asarray(0.0), soil, grid, jnp.asarray(1e3), jnp.asarray(0.0),
                            jnp.zeros(n), jnp.asarray(dt), jnp.asarray(1.0), jnp.asarray(-15000.0),
                            jnp.asarray(0.0), cfg)  # fmt: skip
        return (r.h, r.theta), (r.infiltration, r.balance_error, r.n_clamp, r.h[-1])

    n_t = round(0.15 / dt)
    _, (inf, bal, clamp, h_bottom) = jax.jit(
        lambda: lax.scan(body, (h0, theta_of_h(h0, soil)), None, length=n_t)
    )()
    cum = np.cumsum(np.asarray(inf))
    t = dt * np.arange(1, n_t + 1)
    m = t >= 0.01
    s_num, a_num = np.linalg.lstsq(np.stack([np.sqrt(t[m]), t[m]], 1), cum[m], rcond=None)[0]
    s_ref = _parlange_sorptivity(soil1, h_i)
    ks = rec1[3]
    # measured: S within 0.44 % (loamy sand) and 0.18 % (fine) of Parlange; A = 0.63 Ks and 0.66 Ks
    assert s_num == pytest.approx(s_ref, rel=0.01)
    assert 0.3 * ks < a_num < 0.8 * ks  # Philip: A between Ks/3 and 2Ks/3 for the two-term series
    assert np.max(np.abs(np.asarray(bal))) < 1e-8
    assert float(np.sum(clamp)) == 0.0
    assert float(np.max(np.abs(np.asarray(h_bottom) - h_i))) < 1e-6  # the front never reached the bottom


# ---------------------------------------------------------------------------
# convergence and precision
# ---------------------------------------------------------------------------

#: max |daily storage difference| [cm] against 96 x 8 on the 40-day NumPy forcing below (17.9 cm of
#: rain), float64. Measured: 6 x 1 2.66, 12 x 2 0.030, 24 x 3 0.0079, 48 x 4 0.0034 (96 x 8 vs
#: 192 x 10: 0.0018); the bounds leave ~30 % headroom.
CONVERGENCE_BOUNDS = {(6, 1): 3.5, (12, 2): 0.04, (24, 3): 0.011, (48, 4): 0.0045}


@pytest.mark.allow_skip(reason="convergence errors below 1e-3 cm need float64")
def test_convergence_sub_steps_times_iterations() -> None:
    if not X64:
        pytest.skip("needs float64")
    grid, soil = catpa_grid(), catpa_soil()
    supply, evap, uptake = synthetic_forcing(40)
    w0 = SoilWater.from_theta(jnp.full(37, 0.20), soil)

    def storage(n_sub: int, n_iter: int, scheme: str = "implicit") -> np.ndarray:
        cfg = RichardsConfig(n_sub=n_sub, n_iter=n_iter, time_scheme=scheme)
        _, days = run_days(RichardsParams(soil=soil, grid=grid, config=cfg), w0, supply, evap, uptake)
        return np.asarray(days.theta) @ np.asarray(grid.tl)

    ref = storage(96, 8)
    assert np.max(np.abs(storage(192, 10) - ref)) < 0.003  # the reference itself is converged to this level
    errs = {}
    for cfg, bound in CONVERGENCE_BOUNDS.items():
        errs[cfg] = float(np.max(np.abs(storage(*cfg) - ref)))
        assert errs[cfg] < bound, (cfg, errs[cfg])
    order = list(CONVERGENCE_BOUNDS)
    assert all(errs[a] > errs[b] for a, b in itertools.pairwise(order)), errs
    # the design choice: RZWQM's Crank-Nicolson sub-steps (alpha = 1/2) do not damp an unconverged
    # iterate; at 24 x 3 they drift by centimetres (measured 11.8 cm) where the implicit steps stay
    # within 0.008 cm; converged (96 x 8) the two schemes agree to 0.004 cm
    assert np.max(np.abs(storage(24, 3, "rzwqm") - ref)) > 100 * errs[(24, 3)]
    assert np.max(np.abs(storage(96, 8, "rzwqm") - ref)) < 0.01


@pytest.mark.parametrize(("n_sub", "n_iter", "bal32_bound"), [(24, 3, 2e-3), (96, 8, 1e-4)])
def test_float32_matches_float64_on_numpy_inputs(n_sub: int, n_iter: int, bal32_bound: float) -> None:
    """Same NumPy forcing in both precisions: float32 must close the balance and track the float64 storage.

    Measured over 30 days: 24 x 3, max |storage difference| 6.7e-6 cm, float32 daily imbalance
    4.6e-4 cm (the unconverged residual, the same as float64's 4.6e-4); 96 x 8, storage 3.4e-5 cm,
    float32 imbalance 1.1e-5 cm per day (float64: 1.3e-14); cumulative drainage within 1.5e-6 cm.
    """
    supply, evap, uptake = synthetic_forcing(30, seed=7)

    def go(x64: bool):
        with jax.enable_x64(x64):
            dt = np.float64 if x64 else np.float32
            grid = RichardsGrid.from_rzwqm(CATPA_TLT, CATPA_DELZ)
            grid = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dt), grid)
            nh = np.searchsorted(CATPA_HORIZON_BOTTOM, CATPA_TLT, side="left")
            soil = SoilHydraulicParams.from_rzwqm_records(
                CATPA_REC1.astype(dt), CATPA_REC2.astype(dt), node_horizon=nh
            )
            soil = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dt), soil)
            cfg = RichardsConfig(n_sub=n_sub, n_iter=n_iter)
            params = RichardsParams(soil=soil, grid=grid, config=cfg)
            w0 = SoilWater.from_theta(jnp.full(37, 0.22, dt), soil)
            _, days = run_days(params, w0, supply.astype(dt), evap.astype(dt), uptake.astype(dt))
            assert days.theta.dtype == dt
            s = np.asarray(days.theta, np.float64) @ np.asarray(grid.tl, np.float64)
            return (
                s,
                np.asarray(days.flux.balance_error, np.float64),
                np.asarray(days.flux.drainage, np.float64),
                float(np.sum(days.flux.n_clamp)),
            )

    s64, b64, d64, c64 = go(True)
    s32, b32, d32, c32 = go(False)
    assert np.all(np.isfinite(s32))
    assert c32 == c64 == 0.0
    assert np.max(np.abs(s32 - s64)) < 2e-4
    assert np.max(np.abs(b32)) < bal32_bound
    assert np.max(np.abs(b32 - b64)) < 1e-4
    assert np.max(np.abs(np.cumsum(d32) - np.cumsum(d64))) < 1e-4


# ---------------------------------------------------------------------------
# process wrapper
# ---------------------------------------------------------------------------


def test_process_wrapper_registered_and_equal_to_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    assert registry["richards_redistribution"] is richards_redistribution
    assert richards_redistribution.fortran_name == "RICHRD"
    grid, soil = catpa_grid(), catpa_soil()
    params = RichardsParams(soil=soil, grid=grid)
    supply, evap, uptake = synthetic_forcing(8, seed=11)
    w0 = SoilWater.from_theta(jnp.full(37, 0.24), soil)
    forcing = RichardsForcing(
        supply=jnp.asarray(supply), evaporation=jnp.asarray(evap), uptake=jnp.asarray(uptake)
    )
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")  # the write check and the finite-head check run inside the scan
    model = Model(
        RichardsState, [richards_redistribution], outputs=("soil_water.theta", "soil_water.flux.drainage")
    )
    final, outs = jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(
        params, forcing, RichardsState(soil_water=w0)
    )
    w = w0
    for d in range(8):
        w = richards_day(w, params, forcing.supply[d], forcing.evaporation[d], forcing.uptake[d])
    np.testing.assert_allclose(final.soil_water.theta, w.theta, rtol=0, atol=1e-13 if X64 else 1e-6)
    np.testing.assert_allclose(
        outs["soil_water.flux.drainage"][-1], w.flux.drainage, rtol=1e-10 if X64 else 1e-5
    )
    # eager call: the write check sees only soil_water change
    st = RichardsState(soil_water=w0)
    new = richards_redistribution(st, params, jax.tree_util.tree_map(lambda x: x[0], forcing))
    changed = richards_redistribution.check_writes(st, new)
    assert changed and all(c.startswith("soil_water.") for c in changed)
    assert os.environ["AGRI_JAX_CHECK"] == "1"
