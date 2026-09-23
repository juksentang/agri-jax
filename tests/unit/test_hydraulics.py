"""Brooks-Corey hydraulics (RZWQM form): round trips, monotonicity, bounds, continuity, analytic
derivatives against autodiff, finite-difference gradients w.r.t. every parameter, hypothesis
property tests on realistic parameter ranges, and the CA-TPA rec2 derivation check (the scenario-file
version is in tests/integration/test_hydraulics_catpa.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agri_jax.processes.soil_water.hydraulics import (
    H_CLAMP_RZWQM,
    H_FC13,
    H_FC110,
    H_MIN,
    H_WP,
    SoilHydraulicParams,
    c2_of_params,
    c_of_h,
    derive_rzwqm,
    h_of_theta,
    k_of_h,
    theta_of_h,
)

X64 = bool(jax.config.jax_enable_x64)
RT_TOL = 1e-10 if X64 else 2e-5  # round-trip tolerance (task: 1e-10 in x64)
GRAD_TOL = 1e-8 if X64 else 1e-4  # analytic vs autodiff
FD_REL = 1e-4 if X64 else 5e-2  # autodiff vs central finite differences
FD_EPS = 1e-6 if X64 else 1e-3
JOINT_OFF = 1e-3 if X64 else 2e-2  # relative offset of the "around the joint" points; must exceed FD_EPS
EXACT = 1e-12 if X64 else 1e-5  # tolerance for algebraically identical evaluations

# CA-TPA rzwqm.dat "SOIL HORIZON HYDRAULIC PROPERTIES" block (5 horizons).
CATPA_REC1 = np.array(
    [
        [14.6545, 0.22, 2.966, 5.41, 0.055, 0.453],
        [14.6545, 0.26, 2.966, 3.16, 0.032, 0.453],
        [14.6545, 0.36, 2.966, 3.31, 0.043, 0.453],
        [14.6545, 0.17, 2.966, 3.32, 0.048, 0.453],
        [14.6545, 0.322, 2.966, 2.59, 0.041, 0.453],
    ]
)
CATPA_REC2 = np.array(
    [
        [0.255198, 0.315854, 0.141628, 14.6545, 7440.01, 0.0, 0.0],
        [0.218896, 0.287526, 0.101447, 14.6545, 7440.01, 0.0, 0.0],
        [0.176185, 0.248369, 0.076817, 14.6545, 7440.01, 0.0, 0.0],
        [0.286152, 0.340193, 0.172662, 14.6545, 7440.01, 0.0, 0.0],
        [0.1917, 0.262994, 0.085222, 14.6545, 7440.01, 0.0, 0.0],
    ]
)
# CA-TPA 37 nodes -> 5 horizons (lower depths 15/30/70/90/150 cm), from the rzwqm.dat node list.
CATPA_NODE_DEPTH = np.array(
    [1, 2, 4, 7, 11, 15, 19, 23, 27, 30, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101,
     105, 109, 113, 117, 121, 125, 129, 135, 150], dtype=float
)  # fmt: skip
CATPA_HORIZON_BOTTOM = np.array([15.0, 30.0, 70.0, 90.0, 150.0])

PRIMARY = ("hb", "lambda_", "eps", "ksat", "theta_r", "theta_s", "hb_k", "n1", "a1")
DERIVED = ("fc13", "fc110", "wp", "c2")


def make_params(
    hb=14.6545, lam=0.22, eps=2.966, ksat=5.41, theta_r=0.055, theta_s=0.453, hb_k=None, n1=0.0, a1=0.0
) -> SoilHydraulicParams:
    """Scalar (single horizon, no node map) parameter set; derived fields filled by RZWQM's rule."""
    f = jnp.asarray
    p = SoilHydraulicParams(
        hb=f(hb),
        lambda_=f(lam),
        eps=f(eps),
        ksat=f(ksat),
        theta_r=f(theta_r),
        theta_s=f(theta_s),
        fc13=f(0.0),
        fc110=f(0.0),
        wp=f(0.0),
        hb_k=f(hb if hb_k is None else hb_k),
        c2=f(0.0),
        n1=f(n1),
        a1=f(a1),
    )
    return derive_rzwqm(p)


def catpa_params(node_map: bool = False) -> SoilHydraulicParams:
    nh = np.searchsorted(CATPA_HORIZON_BOTTOM, CATPA_NODE_DEPTH, side="left") if node_map else None
    return SoilHydraulicParams.from_rzwqm_records(CATPA_REC1, CATPA_REC2, node_horizon=nh)


# a few contrasting parameter sets: CA-TPA horizon 1, a clay-like curve, a modified curve with a1>0 and n1>0
CASES = [
    make_params(),
    make_params(hb=37.3, lam=0.131, eps=2.393, ksat=0.05, theta_r=0.09, theta_s=0.475),
    make_params(hb=7.26, lam=0.592, eps=3.776, ksat=6.11, theta_r=0.02, theta_s=0.437),
    make_params(
        hb=20.76, lam=0.211, eps=2.633, ksat=1.0, theta_r=0.015, theta_s=0.501, a1=0.002, n1=0.3, hb_k=25.0
    ),
]


def h_grid(p: SoilHydraulicParams, n: int = 400) -> jnp.ndarray:
    """Heads from -1e6 cm to +10 cm, log-spaced in |h| plus the joints exactly and points around them."""
    hb = float(p.hb)
    hbk = float(p.hb_k)
    logs = -np.logspace(-3, 6, n)
    joints = np.array(
        [-hb, -hb * (1 - 1e-3), -hb * (1 + 1e-3), -hbk, -hbk * (1 - 1e-3), -hbk * (1 + 1e-3), 0.0]
    )
    return jnp.asarray(np.sort(np.concatenate([logs, joints, [-10.0 * hb, 1.0, 10.0]])))


# ---------------------------------------------------------------------------
# round trips, bounds, monotonicity, continuity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", CASES)
def test_roundtrip_theta_h_theta(p: SoilHydraulicParams) -> None:
    theta_s = float(p.theta_s)
    theta_min = float(theta_of_h(-1e25, p))  # smallest theta whose inverse is above the H_MIN floor
    assert theta_min < float(p.theta_r) + 1e-3
    theta = jnp.asarray(np.linspace(theta_min, theta_s, 500))
    back = theta_of_h(h_of_theta(theta, p), p)
    np.testing.assert_allclose(np.asarray(back), np.asarray(theta), rtol=0, atol=RT_TOL)


@pytest.mark.parametrize("p", CASES)
def test_roundtrip_h_theta_h(p: SoilHydraulicParams) -> None:
    """h -> theta -> h is exact on the Brooks-Corey segment (Se from 1e-6 to 1) and on the linear segment when a1 > 0.

    Below Se ~ 1e-6, theta - theta_r is at the resolution of theta itself, so no inverse can be exact.
    """
    hb, lam = float(p.hb), float(p.lambda_)
    se_floor = (hb / -H_MIN) ** lam  # Se at the H_MIN floor of h_of_theta (matters for small lambda)
    se_lo = 1e-6 if X64 else 0.05  # float32 has no digits left in theta - theta_r below Se ~ 0.05
    se = np.logspace(np.log10(max(se_lo, 1.01 * se_floor)), 0, 300)
    if float(p.a1) == 0.0:
        se = se[:-1]  # classical plateau: theta(-hb) = theta_s is not invertible (h_of_theta gives 0)
    h = -hb * se ** (-1.0 / lam)
    if float(p.a1) > 0.0:
        h = np.concatenate([h, -np.linspace(hb, 0.0, 50)])
    h = jnp.asarray(h)
    back = h_of_theta(theta_of_h(h, p), p)
    np.testing.assert_allclose(np.asarray(back), np.asarray(h), rtol=RT_TOL, atol=RT_TOL)


@pytest.mark.parametrize("p", CASES)
def test_theta_bounds_and_monotone(p: SoilHydraulicParams) -> None:
    h = h_grid(p)
    theta = np.asarray(theta_of_h(h, p))
    slack = 1e-15 if X64 else 1e-6
    assert np.all(np.isfinite(theta))
    assert np.all(theta >= float(p.theta_r) - slack)
    assert np.all(theta <= float(p.theta_s) + slack)
    assert np.all(np.diff(theta) >= -slack), "theta must be non-decreasing in h"
    assert theta[-1] == pytest.approx(float(p.theta_s))
    span = float(p.theta_s - p.theta_r - p.a1 * p.hb)
    assert theta[0] == pytest.approx(
        float(p.theta_r) + span * (float(p.hb) / 1e6) ** float(p.lambda_), rel=EXACT
    )
    assert np.all(np.asarray(c_of_h(h, p)) >= 0.0)


@pytest.mark.parametrize("p", CASES)
def test_k_bounds_and_monotone(p: SoilHydraulicParams) -> None:
    """K is positive, finite, <= ksat and non-decreasing in h.

    With n1 > 0 RZWQM's wet segment ``ksat |h|**-n1`` exceeds ksat as h -> 0- (POINTK has no guard; every
    reference class has n1 = 0), so those two properties are only asserted below -hb_k in that case.
    """
    h = h_grid(p)
    k = np.asarray(k_of_h(h, p))
    assert np.all(np.isfinite(k))
    assert np.all(k > 0.0)
    assert k[-1] == pytest.approx(float(p.ksat))
    sel = np.ones_like(k, dtype=bool) if float(p.n1) == 0.0 else np.asarray(h) <= -float(p.hb_k)
    assert np.all(k[sel] <= float(p.ksat) * (1.0 + 1e-12))
    assert np.all(np.diff(k[sel]) >= -1e-12 * float(p.ksat)), "K must be non-decreasing in h"


@pytest.mark.parametrize("p", CASES)
def test_h_of_theta_monotone(p: SoilHydraulicParams) -> None:
    theta = jnp.asarray(np.linspace(float(p.theta_r) - 0.01, float(p.theta_s) + 0.01, 400))
    h = np.asarray(h_of_theta(theta, p))
    assert np.all(np.isfinite(h))
    assert np.all(np.diff(h) >= 0.0)
    assert h[-1] == 0.0


@pytest.mark.parametrize("p", CASES)
def test_continuity_at_joints(p: SoilHydraulicParams) -> None:
    hb, hbk = float(p.hb), float(p.hb_k)
    d = 1e-9 if X64 else 1e-5
    rtol = 1e-6 if X64 else 1e-4
    # theta at -hb and at 0
    lo, hi = theta_of_h(jnp.asarray([-hb * (1 + d), -d]), p), theta_of_h(jnp.asarray([-hb * (1 - d), 0.0]), p)
    np.testing.assert_allclose(np.asarray(lo), np.asarray(hi), rtol=rtol)
    # K at -hb_k, and at 0 when n1 = 0 (with n1 > 0 RZWQM's wet segment is discontinuous at 0, see above)
    klo = k_of_h(jnp.asarray([-hbk * (1 + d), -d]), p)
    khi = k_of_h(jnp.asarray([-hbk * (1 - d), 0.0]), p)
    n = 2 if float(p.n1) == 0.0 else 1
    np.testing.assert_allclose(np.asarray(klo)[:n], np.asarray(khi)[:n], rtol=rtol)
    # exact values at the joint: both segments give ksat * hb_k**-n1
    k_joint = float(p.ksat) * hbk ** (-float(p.n1))
    assert float(k_of_h(-hbk, p)) == pytest.approx(k_joint, rel=EXACT)
    assert float(c2_of_params(p)) * hbk ** (-float(p.eps)) == pytest.approx(k_joint, rel=EXACT)


def test_k_continuity_does_not_depend_on_stored_c2() -> None:
    """The stored (possibly stale) c2 is never used: changing ksat keeps K continuous at -hb_k."""
    p = catpa_params()
    p2 = p.replace(ksat=p.ksat * 3.0)  # c2 field now stale, as in the CA-TPA file
    hbk = np.asarray(p.hb_k)
    lo = np.asarray(k_of_h(jnp.asarray(-hbk * (1 + 1e-9)), p2))
    hi = np.asarray(k_of_h(jnp.asarray(-hbk * (1 - 1e-9)), p2))
    np.testing.assert_allclose(lo, hi, rtol=1e-6)
    np.testing.assert_allclose(hi, np.asarray(p2.ksat), rtol=1e-6)


# ---------------------------------------------------------------------------
# derivatives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", CASES)
def test_c_of_h_equals_grad_theta(p: SoilHydraulicParams) -> None:
    h = h_grid(p)
    g = jax.vmap(jax.grad(lambda x: theta_of_h(x, p)))(h)
    c = c_of_h(h, p)
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_allclose(np.asarray(g), np.asarray(c), rtol=0, atol=GRAD_TOL)


def _split(p: SoilHydraulicParams) -> tuple[dict[str, jnp.ndarray], SoilHydraulicParams]:
    return {k: getattr(p, k) for k in PRIMARY}, p


def _with(p: SoilHydraulicParams, prim: dict[str, jnp.ndarray]) -> SoilHydraulicParams:
    return p.replace(**prim)


def _fd(f, x: float, eps: float, side: str) -> float:
    """Central or second-order one-sided finite difference of the scalar function ``f``."""
    if side == "central":
        return (f(x + eps) - f(x - eps)) / (2 * eps)
    if side == "backward":
        return (3 * f(x) - 4 * f(x - eps) + f(x - 2 * eps)) / (2 * eps)
    return (-3 * f(x) + 4 * f(x + eps) - f(x + 2 * eps)) / (2 * eps)


def _fd_points(p: SoilHydraulicParams, rng: np.random.Generator, n: int = 20) -> list[tuple[float, str, str]]:
    """20 random heads across the range plus the joints, each tagged with the FD scheme for ``hb`` and ``hb_k``.

    Central differences everywhere except exactly at a joint, where the curve has a kink and the
    derivative w.r.t. the joint location is one-sided: theta/C put ``h = -hb`` on the Brooks-Corey
    segment (``h > -hb`` is linear), which a *backward* step in ``hb`` stays on; K puts ``h = -hb_k``
    on the wet segment (``h >= -hb_k``), which a *forward* step in ``hb_k`` stays on.
    """
    hb, hbk = float(p.hb), float(p.hb_k)
    pts: list[tuple[float, str, str]] = [
        (-float(x), "central", "central") for x in 10 ** rng.uniform(-1, 5, n)
    ]
    pts += [(-hb, "backward", "forward" if hbk == hb else "central")]
    pts += [(-hb * (1 - JOINT_OFF), "central", "central"), (-hb * (1 + JOINT_OFF), "central", "central")]
    if hbk != hb:
        pts += [(-hbk, "central", "forward"), (-hbk * (1 - JOINT_OFF), "central", "central")]
        pts += [(-hbk * (1 + JOINT_OFF), "central", "central")]
    if float(p.a1) > 0.0:
        pts += [(-0.5 * hb, "central", "central")]
    return pts


def _check_fd(
    f, prim: dict[str, jnp.ndarray], x: jnp.ndarray, g: dict, schemes: dict[str, str], label: str
) -> int:
    """Compare autodiff ``g`` with finite differences of ``f(prim, x)`` w.r.t. every primary parameter."""
    f0 = float(f(prim, x))
    eps_mach = np.finfo(np.float64 if X64 else np.float32).eps
    n = 0
    for name in PRIMARY:
        got = float(g[name])
        assert np.isfinite(got), f"{label} d/d{name} not finite"
        x0 = float(prim[name])
        eps = FD_EPS * max(abs(x0), 1e-2)

        def f1(v, name=name):
            q = dict(prim)
            q[name] = jnp.asarray(v)
            return float(f(q, x))

        scheme = schemes.get(name, "central")
        if x0 == 0.0 and name in ("n1", "a1"):
            scheme = "forward"  # non-negative parameters: stay in range
        ref = _fd(f1, x0, eps, scheme)
        noise = 32.0 * eps_mach * max(abs(f0), 1e-300) / eps  # roundoff floor of the stencil
        assert abs(got - ref) <= FD_REL * max(abs(ref), abs(got)) + noise, (
            f"{label} d/d{name}: autodiff {got:.6e} vs FD {ref:.6e} ({scheme}, noise {noise:.1e})"
        )
        n += 1
    return n


@pytest.mark.parametrize("p", CASES)
@pytest.mark.parametrize("fn_name", ["theta_of_h", "k_of_h", "c_of_h"])
def test_grad_wrt_every_parameter_finite_and_matches_fd(p: SoilHydraulicParams, fn_name: str) -> None:
    fn = {"theta_of_h": theta_of_h, "k_of_h": k_of_h, "c_of_h": c_of_h}[fn_name]
    rng = np.random.default_rng(0)
    prim, base = _split(p)

    def f(prim_, h_):
        return fn(h_, _with(base, prim_))

    grad_fn = jax.jit(jax.grad(f))
    n_checked = 0
    for h, side_theta, side_k in _fd_points(p, rng):
        g = grad_fn(prim, jnp.asarray(h))
        schemes = {"hb": side_theta, "hb_k": side_k}
        n_checked += _check_fd(f, prim, jnp.asarray(h), g, schemes, f"{fn_name} at h={h}")
    assert n_checked >= 23 * len(PRIMARY)


@pytest.mark.parametrize("p", CASES)
def test_grad_h_of_theta_finite_and_matches_fd(p: SoilHydraulicParams) -> None:
    rng = np.random.default_rng(1)
    prim, base = _split(p)
    theta_r, theta_s = float(p.theta_r), float(p.theta_s)
    thetas = rng.uniform(theta_r + 0.02, theta_s - 1e-3, 20)

    def f(prim_, th_):
        return h_of_theta(th_, _with(base, prim_))

    grad_fn = jax.jit(jax.grad(f, argnums=(0, 1)))
    for th in thetas:
        g_prim, g_th = grad_fn(prim, jnp.asarray(th))
        assert np.isfinite(float(g_th))
        ref = _fd(lambda x: float(f(prim, jnp.asarray(x))), th, FD_EPS * 0.1, "central")
        assert abs(float(g_th) - ref) <= FD_REL * max(abs(ref), 1e-12)
        _check_fd(f, prim, jnp.asarray(th), g_prim, {}, f"h_of_theta at theta={th}")


def test_gradients_finite_at_extremes() -> None:
    """Saturated, far-dry and exactly-at-joint heads give finite gradients w.r.t. h and every parameter."""
    p = CASES[3]
    prim, base = _split(p)
    hs = jnp.asarray([5.0, 0.0, -1e-12, -float(p.hb_k), -float(p.hb), -1e7, -1e9])

    def loss(prim_, h_):
        q = _with(base, prim_)
        return jnp.sum(theta_of_h(h_, q) + k_of_h(h_, q) + c_of_h(h_, q))

    g_prim, g_h = jax.grad(loss, argnums=(0, 1))(prim, hs)
    assert np.all(np.isfinite(np.asarray(g_h)))
    for name, val in g_prim.items():
        assert np.all(np.isfinite(np.asarray(val))), name
    thetas = jnp.asarray([float(p.theta_r) - 0.1, float(p.theta_r), float(p.theta_s), float(p.theta_s) + 0.1])
    g2 = jax.grad(lambda pr, t: jnp.sum(h_of_theta(t, _with(base, pr))), argnums=(0, 1))(prim, thetas)
    assert np.all(np.isfinite(np.asarray(g2[1])))
    for name, val in g2[0].items():
        assert np.all(np.isfinite(np.asarray(val))), name


# ---------------------------------------------------------------------------
# hypothesis property tests on realistic ranges (all_parameters.csv across the 15 scenarios:
# lambda 0.14-0.87, ksat 0.05-6.11, theta_r 0.02-0.17; hb from the RZWQM reference classes 7-37 cm)
# ---------------------------------------------------------------------------

_H_REL = jnp.asarray(
    np.sort(np.concatenate([-np.logspace(-3, 5, 120), [-1.0, -(1 - 1e-6), -(1 + 1e-6), 0.0, 1.0]]))
)


@jax.jit
def _props(hb, lam, eps, ksat, theta_r, theta_s, a1, n1):
    p = SoilHydraulicParams(
        hb=hb, lambda_=lam, eps=eps, ksat=ksat, theta_r=theta_r, theta_s=theta_s,
        fc13=jnp.zeros(()), fc110=jnp.zeros(()), wp=jnp.zeros(()), hb_k=hb, c2=jnp.zeros(()), n1=n1, a1=a1,
    )  # fmt: skip
    h = _H_REL * hb  # covers 1e-3 hb .. 1e5 hb and the joint exactly
    theta = theta_of_h(h, p)
    k = k_of_h(h, p)
    c = c_of_h(h, p)
    g = jax.vmap(jax.grad(lambda x: theta_of_h(x, p)))(h)
    h_back = h_of_theta(theta, p)
    theta_grid = theta_r + (theta_s - theta_r) * jnp.linspace(1e-3, 1.0, 100)  # Se >= 1e-3 stays above H_MIN
    theta_back = theta_of_h(h_of_theta(theta_grid, p), p)
    gp = jax.grad(
        lambda q: jnp.sum(theta_of_h(h, q) + k_of_h(h, q) + c_of_h(h, q)) + jnp.sum(h_of_theta(theta_grid, q))
    )(p)
    return theta, k, c, g, h_back, theta_grid, theta_back, gp


realistic = dict(
    hb=st.floats(2.0, 60.0),
    lam=st.floats(0.14, 0.87),
    eps=st.floats(2.0, 4.5),
    ksat=st.floats(0.05, 6.11),
    theta_r=st.floats(0.02, 0.17),
    theta_s=st.floats(0.35, 0.55),
    a1_frac=st.floats(0.0, 0.5),  # a1 * hb as a fraction of (theta_s - theta_r); RZWQM class max a1 = 0.002
    n1=st.floats(0.0, 0.5),
)


@settings(max_examples=60, deadline=None)
@given(**realistic)
def test_property_realistic_parameters(hb, lam, eps, ksat, theta_r, theta_s, a1_frac, n1) -> None:
    a1 = a1_frac * (theta_s - theta_r) / hb
    theta, k, c, g, h_back, theta_grid, theta_back, gp = _props(hb, lam, eps, ksat, theta_r, theta_s, a1, n1)
    theta, k, c, g, h_back = (np.asarray(x) for x in (theta, k, c, g, h_back))
    h = np.asarray(_H_REL) * hb
    tol = 1e-9 if X64 else 1e-4
    # bounds and monotonicity
    assert np.all(np.isfinite(theta)) and np.all(theta >= theta_r - tol) and np.all(theta <= theta_s + tol)
    assert np.all(np.diff(theta) >= -tol)
    assert np.all(np.isfinite(k)) and np.all(k > 0.0)
    sel = (
        np.ones_like(k, dtype=bool) if n1 == 0.0 else h <= -hb
    )  # n1 > 0: RZWQM's wet segment is not monotone
    assert np.all(np.diff(k[sel]) >= -tol * ksat)
    assert np.all(c >= 0.0)
    # analytic capacity == autodiff
    np.testing.assert_allclose(g, c, rtol=0, atol=1e-8 if X64 else 1e-3)
    # K continuous at -hb_k (points at (1 +- 1e-6) hb)
    i_lo, i_hi = np.argmin(np.abs(h + hb * (1 + 1e-6))), np.argmin(np.abs(h + hb * (1 - 1e-6)))
    assert k[i_lo] == pytest.approx(k[i_hi], rel=1e-4 if X64 else 1e-2)
    # round trips
    np.testing.assert_allclose(np.asarray(theta_back), np.asarray(theta_grid), rtol=0, atol=RT_TOL)
    # strict: theta(-hb) = theta_s is the non-invertible plateau when a1 = 0; the error of the inverse grows
    # as 1/Se (theta - theta_r loses digits), so float32 is only checked down to Se = 0.05
    se_lo = 1e-6 if X64 else 0.05
    dry = (h < -hb) & (h >= -hb * se_lo ** (-1.0 / lam))
    np.testing.assert_allclose(h_back[dry], h[dry], rtol=1e-9 if X64 else 1e-4)
    # every parameter gradient finite
    for name in PRIMARY:
        assert np.isfinite(float(getattr(gp, name))), name


# ---------------------------------------------------------------------------
# node -> horizon map, batching
# ---------------------------------------------------------------------------


def test_node_horizon_gather_matches_per_horizon() -> None:
    p = catpa_params(node_map=True)
    assert p.node_horizon is not None and len(p.node_horizon) == 37
    assert p.node_horizon[:6] == (0, 0, 0, 0, 0, 0) and p.node_horizon[-1] == 4
    h = jnp.asarray(-np.logspace(0, 4, 37))
    theta = np.asarray(theta_of_h(h, p))
    k = np.asarray(k_of_h(h, p))
    hp = catpa_params()
    for i, j in enumerate(p.node_horizon):
        pj = jax.tree_util.tree_map(lambda a, j=j: a[j], hp.replace(node_horizon=None))
        assert theta[i] == pytest.approx(float(theta_of_h(h[i], pj)), rel=EXACT)
        assert k[i] == pytest.approx(float(k_of_h(h[i], pj)), rel=EXACT)
    q = p.at_nodes()
    assert q.node_horizon is None and q.hb.shape == (37,)
    assert q.at_nodes() is q  # idempotent (no map -> unchanged)


def test_vmap_over_parameter_batch_and_jit() -> None:
    p = catpa_params(node_map=True)
    batch = jax.tree_util.tree_map(lambda a: jnp.stack([a, a, a]), p)
    batch = batch.replace(ksat=jnp.stack([p.ksat, p.ksat * 1.1, p.ksat * 0.9]))
    assert batch.node_horizon == p.node_horizon  # static, shared across the batch
    h = jnp.asarray(-np.logspace(0, 4, 37))
    f = jax.jit(jax.vmap(lambda q: (theta_of_h(h, q), k_of_h(h, q))))
    theta, k = f(batch)
    assert theta.shape == (3, 37) and k.shape == (3, 37)
    np.testing.assert_allclose(np.asarray(theta[1]), np.asarray(theta_of_h(h, p)), rtol=EXACT)
    np.testing.assert_allclose(
        np.asarray(k[1]), np.asarray(k_of_h(h, p)) * 1.1, rtol=EXACT
    )  # K scales with ksat


def test_field_metadata_units() -> None:
    meta = SoilHydraulicParams.field_metadata()
    assert meta["hb"]["unit"] == "cm" and meta["hb"]["fortran_name"] == "SOILHP(1)"
    assert meta["ksat"]["unit"] == "cm hr-1"
    assert meta["theta_s"]["unit"] == "cm3 cm-3"
    assert meta["c2"]["fortran_name"] == "SOILHP(11)" and meta["a1"]["fortran_name"] == "SOILHP(13)"
    assert meta["node_horizon"]["static"] is True


# ---------------------------------------------------------------------------
# CA-TPA: what RZWQM does with rec2 (fc13 / fc110 / wp / c2)
# ---------------------------------------------------------------------------


def test_catpa_rec2_is_the_curve_at_333_100_15000_cm() -> None:
    """RZWQM's SOILPR (ITYPE=0) overwrites fc13/fc110/wp with theta(-333), theta(-100), theta(-15000)
    (RZTEST.for 4597-4616, 4858); the file values are exactly those to the 6 decimals written."""
    p = SoilHydraulicParams.from_rzwqm_records(CATPA_REC1, CATPA_REC2, derive=False)
    n = p.n_horizon
    fc13 = np.asarray(theta_of_h(jnp.full(n, H_FC13), p))
    fc110 = np.asarray(theta_of_h(jnp.full(n, H_FC110), p))
    wp = np.asarray(theta_of_h(jnp.full(n, H_WP), p))
    np.testing.assert_allclose(fc13, CATPA_REC2[:, 0], atol=1.5e-6, rtol=0)
    np.testing.assert_allclose(fc110, CATPA_REC2[:, 1], atol=1.5e-6, rtol=0)
    np.testing.assert_allclose(wp, CATPA_REC2[:, 2], atol=1.5e-6, rtol=0)
    d = derive_rzwqm(p)
    np.testing.assert_allclose(np.asarray(d.fc13), fc13, rtol=EXACT)
    np.testing.assert_allclose(np.asarray(d.fc110), fc110, rtol=EXACT)
    np.testing.assert_allclose(np.asarray(d.wp), wp, rtol=EXACT)
    assert d.node_horizon is None
    # from_rzwqm_records(derive=True) is the default and gives the same
    d2 = SoilHydraulicParams.from_rzwqm_records(CATPA_REC1, CATPA_REC2)
    np.testing.assert_allclose(np.asarray(d2.wp), wp, rtol=EXACT)


def test_catpa_file_c2_is_stale_and_overwritten() -> None:
    """The file's c2 = 7440.01 on every horizon equals 2.59 * hb_k**eps (horizon 5's Ksat); RZWQM recomputes
    c2 = ksat * hb_k**(eps - n1) per horizon (RZTEST.for line 4611), so only horizon 5 agrees."""
    p = SoilHydraulicParams.from_rzwqm_records(CATPA_REC1, CATPA_REC2, derive=False)
    c2 = np.asarray(c2_of_params(p))
    assert c2[4] == pytest.approx(7440.01, abs=0.1)
    assert np.all(c2[:4] > 9000.0)
    d = derive_rzwqm(p)
    np.testing.assert_allclose(np.asarray(d.c2), c2, rtol=EXACT)
    # and k_of_h uses the derived value, not the stale field
    hbk = np.asarray(p.hb_k)
    k_below = np.asarray(k_of_h(jnp.asarray(-2.0 * hbk), p))
    np.testing.assert_allclose(k_below, np.asarray(p.ksat) * 2.0 ** (-2.966), rtol=EXACT)


def test_rzwqm_dry_clamp_is_the_wilting_point_head() -> None:
    """RZWQM's active dry-end clamp is Hmin = -15000 cm (Rzmain.for line 4587, = HWP), not the
    commented-out -35000 PARAMETER; h_of_theta at the derived wilting point returns it."""
    assert H_CLAMP_RZWQM == H_WP == -15000.0
    d = derive_rzwqm(SoilHydraulicParams.from_rzwqm_records(CATPA_REC1, CATPA_REC2, derive=False))
    h = np.asarray(h_of_theta(d.wp, d))
    np.testing.assert_allclose(h, H_CLAMP_RZWQM, rtol=1e-9 if X64 else 1e-4)
    assert H_MIN < H_CLAMP_RZWQM  # the curve's own floor is only an overflow guard
