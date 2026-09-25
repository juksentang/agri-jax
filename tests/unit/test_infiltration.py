"""Green-Ampt infiltration event (``processes/soil_water/infiltration.py``) against independent solutions.

Independent references, none derived from RZWQM output (plan 19 A5, level 1):

* **wetting-front suction**: the closed form against ``scipy.integrate.quad`` of ``K(-s)/K_s``
  (the hydraulic functions are validated separately);
* **homogeneous soil, intense rain**: cumulative infiltration ``F(t)`` against the implicit
  Green-Ampt solution ``K t = F - psi dtheta ln(1 + F / (psi dtheta))`` (Green & Ampt 1911),
  converging as the slice thickness goes to 0;
* **constant rain r > K**: runoff starts at the Mein-Larson ponding depth
  ``F_p = psi dtheta / (r / K - 1)`` (ponding time ``t_p = F_p / r``), and after ponding ``F(t)``
  follows the shifted Green-Ampt curve (Mein & Larson 1973);
* **two layers**: the capacity against the closed-form series capacity
  ``f = (psi + z) / (L1/K1 + (z - L1)/K2)``, and ``F(t)`` against a quadrature of
  ``dt = dtheta dz / f(z)`` over both layers;
* **mass**: rain = infiltration + runoff + seepage, and the storage change equals the
  infiltration, to rounding; node <-> slice mapping conserves storage;
* **gradients** finite with respect to ``K_s, h_b, lambda, theta_s, AEF`` (float32 and float64)
  and equal to finite differences (float64).

In the model's Green-Ampt capacity the conductance is divided by ``VRCF = 2`` (RZWQM2), so the
analytic solutions use ``K = K_s / 2``.

Source: Green & Ampt (1911); Mein & Larson (1973); Ahuja et al. (2000) ch. 3.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from agrijax.processes.soil_water import infiltration as G
from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams, h_of_theta, k_of_h
from agrijax.processes.soil_water.richards import RichardsGrid

from .test_richards import CATPA_HORIZON_BOTTOM, CATPA_REC1, CATPA_REC2, catpa_grid, catpa_soil, nodes

X64 = bool(jax.config.read("jax_enable_x64"))
REL = 1e-12 if X64 else 2e-5
AEF = 0.9

# CA-TPA horizon 1 (loamy sand) and horizon 5 as the lower layer of the two-layer soil
REC1_TOP = CATPA_REC1[0]
REC1_LOW = CATPA_REC1[4]


def _soil(rec1: np.ndarray, n: int) -> SoilHydraulicParams:
    rec2 = np.array([0.0, 0.0, 0.0, rec1[0], 0.0, 0.0, 0.0])
    s = SoilHydraulicParams.from_rzwqm_records(np.asarray(rec1), rec2)
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (n,)), s)


def _event(theta, soil, tl, duration, depth, cfg, aef=AEF):
    th = jnp.asarray(theta)
    return G.green_ampt_event(
        th, h_of_theta(th, soil), soil, jnp.asarray(tl), jnp.asarray(aef), jnp.asarray(duration),
        jnp.asarray(depth), cfg,
    )  # fmt: skip


_event_jit = jax.jit(_event, static_argnames="cfg")


def _suction_quad(rec1: np.ndarray, theta_i: float) -> float:
    s1 = SoilHydraulicParams.from_rzwqm_records(np.asarray(rec1), np.array([0, 0, 0, rec1[0], 0, 0, 0.0]))
    s_i = -float(h_of_theta(jnp.asarray(min(theta_i, AEF * rec1[5])), s1))
    ks = rec1[3]
    kk = lambda s: float(k_of_h(jnp.asarray(-s), s1)) / ks  # noqa: E731
    return quad(kk, 0.0, s_i, points=[rec1[0]], limit=200, epsabs=0.0, epsrel=1e-12)[0]


# ---------------------------------------------------------------------------
# vectorised pieces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta_i", [0.08, 0.15, 0.25, 0.35, 0.40])
def test_wetting_front_suction_equals_quadrature(theta_i: float) -> None:
    soil = _soil(REC1_TOP, 1)
    sf = G.wetting_front_suction(jnp.full(1, theta_i), soil, soil.theta_s * AEF)
    # RZWQM2: S_f = 1 + int_1^{s_i} K/Ks ds = int_0^{s_i} K/Ks ds when K = Ks on [0, 1] (n1 = 0)
    assert float(sf[0]) == pytest.approx(_suction_quad(REC1_TOP, theta_i), rel=1e-9 if X64 else 1e-5)


def test_wetting_front_suction_branches_and_n1_eps_limits() -> None:
    soil = _soil(REC1_TOP, 3)
    # s_i <= 1 cm: the integral term is 1 (RZWQM2 convention), so S_f = 2
    near_sat = soil.replace(hb=jnp.full(3, 0.5), hb_k=jnp.full(3, 0.5))
    sf = G.wetting_front_suction(jnp.full(3, 0.40), near_sat, jnp.full(3, 0.41))
    np.testing.assert_allclose(sf, 2.0, rtol=REL)
    # eps = 1 and n1 = 1 use the logarithmic limit; values finite and continuous across the limit
    for name in ("eps", "n1"):
        vals = [
            float(G.wetting_front_suction(jnp.full(3, 0.2), soil.replace(**{name: jnp.full(3, v)}),
                                          soil.theta_s * AEF)[0])
            for v in (1.0 - 1e-4, 1.0, 1.0 + 1e-4)
        ]  # fmt: skip
        assert all(np.isfinite(vals))
        assert abs(vals[0] - vals[1]) < 1e-2 * abs(vals[1]) and abs(vals[2] - vals[1]) < 1e-2 * abs(vals[1])


def test_front_conductance_is_capped_and_non_increasing() -> None:
    soil = nodes(catpa_soil(), 37)
    c = np.asarray(G.front_conductance(soil))
    ks = np.asarray(soil.ksat)
    assert c[0] == ks[0]
    assert np.all(c <= ks + 1e-15) and np.all(np.diff(c) <= 0.0)
    # CA-TPA: K(-hb) = Ks (n1 = 0, hb_k = hb) -> running minimum of Ks over the horizons
    np.testing.assert_allclose(np.unique(c)[::-1], [5.41, 3.16, 2.59], rtol=REL)


def test_slice_mapping_covers_the_cells() -> None:
    grid = catpa_grid()
    cfg = G.GreenAmptConfig.for_grid(np.asarray(grid.tl))
    assert cfg.n_slice == 150
    node, zc = G.slice_nodes(grid.tl, cfg)
    counts = np.bincount(np.asarray(node), minlength=37)
    np.testing.assert_array_equal(counts, np.asarray(grid.tl).astype(int))
    tlt = np.cumsum(np.asarray(grid.tl))
    assert np.all(np.asarray(zc) <= tlt[np.asarray(node)]) and np.all(
        np.asarray(zc) > tlt[np.asarray(node)] - np.asarray(grid.tl)[np.asarray(node)]
    )
    with pytest.raises(ValueError, match="multiples"):
        G.GreenAmptConfig.for_grid([1.0, 1.5, 2.0])
    assert G.GreenAmptConfig.for_grid([1.0, 1.5, 2.0], ds=0.5).n_slice == 9


# ---------------------------------------------------------------------------
# independent solutions
# ---------------------------------------------------------------------------


def _ga_time(f: float, k: float, psi_dtheta: float) -> float:
    return (f - psi_dtheta * np.log1p(f / psi_dtheta)) / k


def _homogeneous(ds: float, depth_cm: float = 100.0):
    n = round(depth_cm)
    grid = RichardsGrid.uniform(n, 1.0)
    cfg = G.GreenAmptConfig.for_grid(np.asarray(grid.tl), ds=ds)
    return grid, _soil(REC1_TOP, n), cfg


@pytest.mark.allow_skip(reason="the O(ds^2) convergence check needs float64")
def test_homogeneous_intense_rain_follows_green_ampt() -> None:
    if not X64:
        pytest.skip("needs float64")
    theta_i, t_end, r = 0.15, 1.0, 1.0e4  # rain far above the capacity: the front runs at capacity
    psi = _suction_quad(REC1_TOP, theta_i)
    dtheta = AEF * REC1_TOP[5] - theta_i
    k = REC1_TOP[3] / G.VRCF
    f_ref = brentq(lambda f: _ga_time(f, k, psi * dtheta) - t_end, 1e-9, 200.0, xtol=1e-14)
    errors = []
    for ds in (1.0, 0.5, 0.25, 0.125):
        grid, soil, cfg = _homogeneous(ds)
        res = _event_jit(jnp.full(grid.n_node, theta_i), soil, grid.tl, [t_end], [r * t_end], cfg=cfg)
        assert float(res.duration) == pytest.approx(t_end, rel=1e-12)
        assert float(res.infiltration) + float(res.runoff) == pytest.approx(r * t_end, rel=1e-13)
        errors.append(abs(float(res.infiltration) - f_ref))
    # measured: 7.6e-4, 2.1e-4, 6.0e-5, 1.7e-5 cm of F = 7.48 cm (ratios 3.4-3.7: about second order)
    assert errors[-1] < 4e-5
    for a, b in itertools.pairwise(errors):
        assert 3.0 < a / b < 5.0, errors


@pytest.mark.allow_skip(reason="the ponding-depth bisection resolves 1e-3 cm, which needs float64")
def test_constant_rain_ponding_depth_and_post_ponding_curve() -> None:
    if not X64:
        pytest.skip("needs float64")
    theta_i = 0.15
    psi = _suction_quad(REC1_TOP, theta_i)
    dtheta = AEF * REC1_TOP[5] - theta_i
    k = REC1_TOP[3] / G.VRCF
    r = 3.0 * k
    f_p = psi * dtheta / (r / k - 1.0)  # Mein & Larson (1973)
    t_p = f_p / r
    ds = 0.05
    grid, soil, cfg = _homogeneous(ds)
    th = jnp.full(grid.n_node, theta_i)

    def runoff(depth: float) -> float:
        res = _event_jit(th, soil, grid.tl, [depth / r], [depth], cfg=cfg)
        return float(res.runoff)

    assert runoff(0.9 * f_p) < 1e-12
    assert runoff(1.2 * f_p) > 1e-4
    lo, hi = 0.9 * f_p, 1.2 * f_p
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if runoff(mid) < 1e-12 else (lo, mid)
    # runoff starts within one slice of F_p (the capacity is evaluated at slice centres)
    assert abs(hi - f_p) < dtheta * ds, (hi, f_p)
    # after ponding: K (t - t_p + t_s) = F - psi dtheta ln(1 + F / (psi dtheta)), t_s from F_p
    t_s = _ga_time(f_p, k, psi * dtheta)
    t_end = 4.0 * t_p
    f_ref = brentq(lambda f: _ga_time(f, k, psi * dtheta) - (t_end - t_p + t_s), f_p, 200.0, xtol=1e-14)
    res = _event_jit(th, soil, grid.tl, [t_end], [r * t_end], cfg=cfg)
    assert float(res.infiltration) == pytest.approx(f_ref, rel=2e-4)
    assert float(res.runoff) == pytest.approx(r * t_end - f_ref, rel=2e-3)


def _two_layer(ds: float, k_low: float):
    # nodes of 1-3 cm (non-uniform cells) over 10 cm of the top soil and 50 cm of the lower one
    tl = np.array([1, 1, 2, 3, 3] + [2] * 25, dtype=float)
    z_top = np.cumsum(tl) - tl
    n = len(tl)
    low = np.asarray(REC1_LOW, dtype=float).copy()
    low[3] = k_low
    horizon = (z_top >= 10.0).astype(int)
    rec1 = np.stack([REC1_TOP, low])
    rec2 = np.array([[0.0, 0.0, 0.0, REC1_TOP[0], 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, low[0], 0.0, 0.0, 0.0]])
    soil = SoilHydraulicParams.from_rzwqm_records(rec1, rec2, node_horizon=horizon)
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), soil.at_nodes())
    return tl, soil, G.GreenAmptConfig.for_grid(tl, ds=ds), rec1


@pytest.mark.parametrize("k_low", [1.2, 8.0])
def test_two_layer_capacity_equals_series_formula(k_low: float) -> None:
    tl, soil, cfg, rec1 = _two_layer(0.5, k_low)
    theta_i = 0.18
    th = jnp.full(len(tl), theta_i)
    suction = G.wetting_front_suction(th, soil, soil.theta_s * AEF)
    cond = G.front_conductance(soil)
    node, zc = G.slice_nodes(jnp.asarray(tl), cfg)
    cap = np.asarray(G.green_ampt_capacity(jnp.asarray(tl), cond, suction, node, zc, cfg))
    k1 = rec1[0, 3]
    k2 = min(k_low, k1)  # the front conductance is non-increasing with depth (RZWQM2 convention)
    z = np.asarray(zc)
    psi = np.asarray(suction)[np.asarray(node)]
    series = np.where(z < 10.0, k1 * (psi + z) / z, (psi + z) / (10.0 / k1 + (z - 10.0) / k2)) / G.VRCF
    np.testing.assert_allclose(cap, series, rtol=1e-12 if X64 else 1e-5)


@pytest.mark.allow_skip(reason="the two-layer quadrature comparison at 1e-4 needs float64")
def test_two_layer_infiltration_follows_series_quadrature() -> None:
    if not X64:
        pytest.skip("needs float64")
    k_low, theta_i, t_end = 1.2, 0.18, 1.5
    psi = [_suction_quad(REC1_TOP, theta_i), _suction_quad(np.r_[REC1_LOW[:3], k_low, REC1_LOW[4:]], theta_i)]
    dth = [AEF * REC1_TOP[5] - theta_i, AEF * REC1_LOW[5] - theta_i]
    k1, k2 = REC1_TOP[3] / G.VRCF, k_low / G.VRCF

    def f(z: float) -> float:
        return k1 * (psi[0] + z) / z if z < 10.0 else (psi[1] + z) / (10.0 / k1 + (z - 10.0) / k2)

    def t_of(zf: float) -> float:
        a = quad(lambda z: dth[0] / f(z), 0.0, min(zf, 10.0), epsrel=1e-12, limit=200)[0]
        b = quad(lambda z: dth[1] / f(z), 10.0, zf, epsrel=1e-12, limit=200)[0] if zf > 10.0 else 0.0
        return a + b

    z_ref = brentq(lambda z: t_of(z) - t_end, 1e-6, 59.0, xtol=1e-13)
    assert z_ref > 12.0  # the front is in the lower layer
    f_ref = dth[0] * 10.0 + dth[1] * (z_ref - 10.0)
    errs = []
    for ds in (0.5, 0.25, 0.125):
        tl, soil, cfg, _ = _two_layer(ds, k_low)
        res = _event_jit(jnp.full(len(tl), theta_i), soil, tl, [t_end], [1e4 * t_end], cfg=cfg)
        errs.append(abs(float(res.infiltration) - f_ref))
    # measured: 2.4e-4, 8.0e-5, 2.5e-5 cm of F = 5.71 cm (front at 25.1 cm, in the lower layer)
    assert errs[-1] < 5e-5
    assert errs[0] > 2.5 * errs[1] > 6.0 * errs[2]


# ---------------------------------------------------------------------------
# mass, identity, saturation, clamps
# ---------------------------------------------------------------------------


def _catpa_case():
    grid = catpa_grid()
    soil = nodes(catpa_soil(), 37)
    cfg = G.GreenAmptConfig.for_grid(np.asarray(grid.tl))
    return grid, soil, cfg


@pytest.mark.parametrize("seed", range(6))
def test_mass_balance_and_storage_on_random_storms(seed: int) -> None:
    grid, soil, cfg = _catpa_case()
    rng = np.random.default_rng(seed)
    th = jnp.asarray(rng.uniform(0.07, 0.40, 37))
    dur = rng.uniform(0.05, 1.5, 4) * (rng.random(4) > 0.2)
    dep = rng.gamma(1.2, 2.0, 4) * (dur > 0)
    res = _event_jit(th, soil, grid.tl, dur, dep, cfg=cfg)
    total = float(np.sum(dep))
    tol = (1e-12 if X64 else 3e-5) * max(total, 1.0)
    assert float(res.rain) == pytest.approx(total, rel=REL)
    assert abs(float(res.error)) < tol
    assert float(res.infiltration) + float(res.runoff) + float(res.seepage) == pytest.approx(total, abs=tol)
    stored = float(jnp.sum((res.theta - th) * grid.tl))
    assert stored == pytest.approx(float(res.infiltration), abs=tol)
    assert float(res.runoff) >= -tol and float(res.infiltration) >= 0.0
    # nothing above the available porosity, and only nodes that received water changed
    porav = np.asarray(soil.theta_s) * AEF
    assert np.all(np.asarray(res.theta) <= np.maximum(porav, np.asarray(th)) + 1e-12)
    same = np.asarray(res.theta) == np.asarray(th)
    np.testing.assert_array_equal(np.asarray(res.h)[same], np.asarray(h_of_theta(th, soil))[same])


def test_slice_grid_must_match_the_cells() -> None:
    """A config that does not span the grid is refused, not silently clipped into the last cell."""
    with pytest.raises(TypeError):
        G.GreenAmptParams(aef=jnp.asarray(AEF))  # type: ignore[call-arg]  # no default slice grid
    grid = RichardsGrid.uniform(50, 1.0)
    soil = _soil(REC1_TOP, 50)
    th = jnp.full(50, 0.1)
    for bad in (G.GreenAmptConfig(n_slice=150), G.GreenAmptConfig(n_slice=40)):
        with pytest.raises(ValueError, match="slices span"):
            _event(th, soil, grid.tl, [10.0], [30.0], bad)
    ok = G.GreenAmptConfig.for_grid(np.asarray(grid.tl))
    res = _event(th, soil, grid.tl, [10.0], [30.0], ok)
    # 30 cm of rain fills the 50 cm profile (about 15 cm of deficit); the rest leaves as seepage
    assert float(jnp.max(res.theta)) <= AEF * REC1_TOP[5] * (1.0 + REL) + 1e-12
    assert float(res.seepage) > 0.0
    # under jit the check runs at trace time on a concrete grid only; a traced grid is skipped
    res_jit = _event_jit(th, soil, grid.tl, jnp.asarray([10.0]), jnp.asarray([30.0]), ok)
    np.testing.assert_allclose(np.asarray(res_jit.theta), np.asarray(res.theta), rtol=REL)


def test_no_storm_is_the_identity_bit_for_bit() -> None:
    grid, soil, cfg = _catpa_case()
    th = jnp.asarray(np.linspace(0.08, 0.42, 37))
    h = h_of_theta(th, soil) * 1.0000001  # a head that is not h(theta) must survive unchanged too
    res = G.green_ampt_event(th, h, soil, grid.tl, jnp.asarray(AEF), jnp.zeros(3), jnp.zeros(3), cfg)
    assert bool(jnp.all(res.theta == th)) and bool(jnp.all(res.h == h))
    assert float(res.infiltration) == float(res.runoff) == float(res.seepage) == 0.0


def test_water_above_available_porosity_is_kept() -> None:
    """RZWQM2 would delete theta above theta_s*AEF (its DRAIN never lets it happen); we keep it."""
    grid, soil, cfg = _catpa_case()
    th = jnp.full(37, 0.2).at[:4].set(0.44)  # above 0.9 * 0.453 = 0.4077
    res = _event_jit(th, soil, grid.tl, [1.0], [1.0], cfg=cfg)
    np.testing.assert_array_equal(np.asarray(res.theta)[:4], np.asarray(th)[:4])
    assert float(jnp.sum((res.theta - th) * grid.tl)) == pytest.approx(1.0, rel=REL)


def test_saturated_profile_seepage_and_runoff() -> None:
    grid, soil, cfg = _catpa_case()
    porav = soil.theta_s * AEF
    # saturated before the storm: everything runs off (RZWQM2: "profile comes in fully saturated")
    res = _event_jit(porav, soil, grid.tl, [1.0], [3.0], cfg=cfg)
    assert float(res.runoff) == pytest.approx(3.0, rel=REL) and float(res.seepage) == 0.0
    # a small deficit and a large storm: the profile saturates, the rest passes through as seepage
    th = porav - 0.002
    deficit = float(jnp.sum(0.002 * grid.tl))
    res = _event_jit(th, soil, grid.tl, [1.0], [3.0], cfg=cfg)
    assert float(res.infiltration) == pytest.approx(deficit, rel=1e-9 if X64 else 1e-4)
    assert float(res.seepage) == pytest.approx(3.0 - deficit - float(res.runoff), rel=REL)
    assert float(res.seepage) > 2.0 and float(res.front_depth) == 150.0


def test_low_intensity_floor_and_short_step_floor() -> None:
    grid, soil, cfg = _catpa_case()
    th = jnp.full(37, 0.2)
    # 0.004 cm over 2 h is 0.002 cm/h, below the 0.01 cm/h floor: rain is consumed at 0.01 cm/h
    res = _event_jit(th, soil, grid.tl, [2.0], [0.004], cfg=cfg)
    assert float(res.infiltration) == pytest.approx(0.004, rel=REL)
    assert float(res.duration) == pytest.approx(0.4, rel=1e-9 if X64 else 1e-4)
    # a slice with a 1e-6 deficit (dq / v < dt_min) still costs at least dt_min of event time
    th2 = soil.theta_s * AEF - jnp.zeros(37).at[0].set(1e-6)
    res = _event_jit(th2, soil, grid.tl, [1.0], [5.0], cfg=cfg)
    assert float(res.duration) >= G.DT_MIN * (1 - 1e-6)


# ---------------------------------------------------------------------------
# gradients and precision
# ---------------------------------------------------------------------------

FIELDS = ("ksat", "hb", "lambda_", "theta_s")


def _loss(soil_h: SoilHydraulicParams, aef, th, grid, cfg, dur, dep):
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (37,)), soil_h.at_nodes())
    r = G.green_ampt_event(th, h_of_theta(th, soil), soil, grid.tl, aef, dur, dep, cfg)
    return r.infiltration + 0.3 * r.runoff + jnp.sum(r.theta * grid.tl * jnp.linspace(1.0, 2.0, 37))


@pytest.mark.parametrize("storm", [(0.5, 6.0), (2.0, 1.5)], ids=["runoff", "all_infiltrates"])
def test_gradients_finite_and_equal_to_finite_differences(storm: tuple[float, float]) -> None:
    grid = catpa_grid()
    soil = catpa_soil()
    cfg = G.GreenAmptConfig.for_grid(np.asarray(grid.tl))
    th = jnp.asarray(np.linspace(0.12, 0.3, 37))
    dur, dep = jnp.asarray([storm[0]]), jnp.asarray([storm[1]])
    aef = jnp.asarray(AEF)
    g_soil, g_aef = jax.grad(_loss, argnums=(0, 1))(soil, aef, th, grid, cfg, dur, dep)
    for name in (*FIELDS, "eps"):
        assert np.all(np.isfinite(np.asarray(getattr(g_soil, name)))), name
    assert np.isfinite(float(g_aef))
    assert any(float(jnp.abs(getattr(g_soil, n)).sum()) > 0 for n in FIELDS)
    if not X64:
        return
    # CA-TPA has hb == hb_k, where K(-hb) switches segments (a kink a central difference straddles);
    # the comparison is made at hb_k = 1.05 hb, where the loss is smooth in every parameter
    soil = soil.replace(hb_k=soil.hb_k * 1.05)
    g_soil, g_aef = jax.grad(_loss, argnums=(0, 1))(soil, aef, th, grid, cfg, dur, dep)
    f = jax.jit(lambda s, a: _loss(s, a, th, grid, cfg, dur, dep))
    for name in FIELDS:
        x = getattr(soil, name)
        for hz in (0, 1):
            eps = 1e-6 * abs(float(x[hz]))
            fd = (float(f(soil.replace(**{name: x.at[hz].add(eps)}), aef))
                  - float(f(soil.replace(**{name: x.at[hz].add(-eps)}), aef))) / (2 * eps)  # fmt: skip
            ad = float(getattr(g_soil, name)[hz])
            assert ad == pytest.approx(fd, rel=1e-4, abs=1e-7), (name, hz, ad, fd)
    e = 1e-7
    fd = (float(f(soil, aef + e)) - float(f(soil, aef - e))) / (2 * e)
    assert float(g_aef) == pytest.approx(fd, rel=1e-4)


def test_float32_matches_float64() -> None:
    def go(x64: bool):
        with jax.enable_x64(x64):
            dt = np.float64 if x64 else np.float32
            grid = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dt), catpa_grid())
            nh = np.searchsorted(CATPA_HORIZON_BOTTOM, np.cumsum(np.asarray(grid.tl)), side="left")
            soil = SoilHydraulicParams.from_rzwqm_records(
                CATPA_REC1.astype(dt), CATPA_REC2.astype(dt), node_horizon=nh
            )
            soil = jax.tree_util.tree_map(
                lambda a: jnp.broadcast_to(jnp.asarray(a, dt), (37,)), soil.at_nodes()
            )
            cfg = G.GreenAmptConfig.for_grid(np.asarray(grid.tl, np.float64))
            th = jnp.asarray(np.linspace(0.1, 0.35, 37), dt)
            r = _event(
                th, soil, grid.tl, np.asarray([0.3, 0.7], dt), np.asarray([2.5, 1.0], dt), cfg, aef=dt(AEF)
            )
            assert r.theta.dtype == dt
            return {k: np.asarray(getattr(r, k), np.float64) for k in ("theta", "infiltration", "runoff")}

    a, b = go(True), go(False)
    np.testing.assert_allclose(b["theta"], a["theta"], atol=2e-6)
    assert b["infiltration"] == pytest.approx(a["infiltration"], rel=2e-5)
    assert b["runoff"] == pytest.approx(a["runoff"], rel=2e-5, abs=2e-5)
