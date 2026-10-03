"""The calibration harness (agrijax.calib) on functions with known answers.

* parameter spaces: the logistic / log transforms round-trip and stay inside the bounds, and
  ``apply`` writes the values into a parameter tree;
* optimisers: batched Adam and CMA-ES find the minimum of quadratics (and CMA-ES of the
  Rosenbrock function) from several starts at once; random search keeps its best; the model-call
  counts are what the budgets say;
* the secant gradient equals the analytic gradient on a smooth function and is non-zero on a
  staircase whose AD derivative is 0;
* the gradient-trust report classifies a smooth function (level 3), a staircase ("step", the AD
  derivative 0 while the value moves), a function with one jump ("jumpy") and a constant
  ("inert"), and its finite differences agree with the analytic derivative;
* valid domains: a spec whose bounds (or log transform) reach outside its declared domain is
  rejected, and ``violations`` reports values outside bounds, domain or chain order;
* ordered chains (LL < DUL < SAT): every ``z`` maps to ordered values at least the gap apart and
  inside their own bounds, the map round-trips and is increasing in each coordinate, gradients
  are finite; ``index`` writes one element of an array leaf; the DSSAT soil specs carry sources;
* per-treatment losses sum to the objective; ``gradient_plan`` turns a report into AD / secant
  pairs (a jumpy pair falls back, a default-derivative-free parameter is derivative-free
  everywhere); ``pair_gradient`` is AD on the trusted pairs and the secant on the others.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib import (
    AdamConfig,
    CmaConfig,
    Domain,
    Group,
    Objective,
    OrderedChain,
    ParamSpace,
    ParamSpec,
    Target,
    TrustConfig,
    adam,
    batched_loss,
    batched_value_and_grad,
    cma_es,
    fd_check,
    first_day_index,
    gradient_plan,
    line_scan,
    pair_gradient,
    positive,
    random_search,
    secant_gradient,
    trust_report,
)
from agrijax.calib.soil import CN_BOUNDS, SWCON_BOUNDS, cn_spec, layer_water_chain, swcon_spec, u_spec

X64 = jax.config.jax_enable_x64

SPACE = ParamSpace(
    [
        ParamSpec("a.x", 1.0, 5.0, label="x"),
        ParamSpec("a.y", 0.0, 2.0, transform="log", label="y"),
        ParamSpec("b", -3.0, 3.0, transform="none"),
    ]
)


# ------------------------------------------------------------------------------ spaces
def test_space_round_trip_and_bounds():
    theta = np.array([[1.5, 0.3, -2.0], [4.9, 7.0, 2.5]])
    z = SPACE.to_unconstrained(jnp.asarray(theta))
    back = np.asarray(SPACE.to_physical(z))
    np.testing.assert_allclose(back, theta, rtol=1e-6 if not X64 else 1e-12)
    wild = np.asarray(SPACE.to_physical(jnp.array([[-50.0, -50.0, 0.0], [50.0, 5.0, 0.0]])))
    assert np.all(wild[:, 0] >= 1.0) and np.all(wild[:, 0] <= 5.0)
    assert np.all(wild[:, 1] >= 0.0)
    assert SPACE.names == ("x", "y", "b")


def test_space_rejects_bad_specs():
    with pytest.raises(ValueError):
        ParamSpec("p", 2.0, 1.0)
    with pytest.raises(ValueError):
        ParamSpec("p", -1.0, 1.0, transform="log")
    with pytest.raises(ValueError):
        ParamSpace([ParamSpec("p", 0.0, 1.0), ParamSpec("p", 0.0, 2.0)])


def test_space_apply_sets_the_tree():
    tree = {"a": {"x": jnp.asarray(2.0), "y": jnp.asarray(1.0)}, "b": jnp.asarray(0.0)}
    out = SPACE.apply(tree, jnp.array([3.0, 0.5, -1.0]))
    assert float(out["a"]["x"]) == 3.0 and float(out["a"]["y"]) == 0.5 and float(out["b"]) == -1.0
    np.testing.assert_array_equal(np.asarray(SPACE.get(out)), [3.0, 0.5, -1.0])
    g = jax.grad(lambda th: SPACE.apply(tree, th)["a"]["x"] * 2.0)(jnp.array([3.0, 0.5, -1.0]))
    np.testing.assert_array_equal(np.asarray(g), [2.0, 0.0, 0.0])


# ------------------------------------------------------------------------------ objective
def test_objective_targets_and_dates():
    stage = jnp.asarray(np.array([[7, 7], [9, 1], [1, 1], [4, 4], [4, 10]]))
    np.testing.assert_array_equal(np.asarray(first_day_index(stage, 4)), [3.0, 3.0])
    np.testing.assert_array_equal(np.asarray(first_day_index(stage, 10)), [5.0, 4.0])  # never: T

    def sim(theta):
        t = jnp.arange(5.0)[:, None]
        return {"y": theta[0] * t + jnp.zeros((5, 2)), "s": stage}

    mask = np.zeros((5, 2), bool)
    mask[2:, :] = True
    tg = (
        Target("yf", "y", "final", rel_scale=0.1),
        Target("ys", "y", "series", scale=1.0, mask=mask),
        Target("d", "s", "date", scale=2.0, code=4),
    )
    obs = {
        "yf": np.array([8.0, 8.0]),
        "ys": np.asarray(sim(jnp.array([2.0]))["y"]),
        "d": np.array([3.0, 3.0]),
    }
    obj = Objective((Group("g", sim, tg, obs),))
    assert float(obj(jnp.array([2.0]))) == 0.0
    # final: ((4 * 3 - 8) / 0.8)^2 = 25; series: mean over 6 masked days of (t)^2 = (4+9+16)*2/6
    br = obj.breakdown(jnp.array([3.0]))
    assert float(br["g/yf"]) == pytest.approx(25.0)
    assert float(br["g/ys"]) == pytest.approx((4 + 9 + 16) * 2 / 6)
    assert float(br["g/d"]) == 0.0
    with pytest.raises(ValueError):
        Target("bad", "s", "date")


# ------------------------------------------------------------------------------ optimisers
def _quad(z):
    c = jnp.array([0.5, -1.0, 2.0])
    return jnp.sum((z - c) ** 2 * jnp.array([1.0, 4.0, 0.25]))


def test_adam_batched_quadratic():
    z0 = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, -3.0], [-2.0, 1.0, 5.0]])
    res = adam(batched_value_and_grad(_quad), z0, AdamConfig(lr=0.1, steps=600))
    np.testing.assert_allclose(res.z_best, np.tile([0.5, -1.0, 2.0], (3, 1)), atol=2e-3)
    assert res.n_grad == 600 and res.n_forward == 0
    assert res.loss_history.shape == (600, 3) and res.z_history.shape == (600, 3, 3)
    assert np.all(res.loss_best <= res.loss_history[0])


def test_cma_es_quadratic_and_rosenbrock():
    z0 = np.array([[2.0, 2.0, 2.0], [-1.0, 0.0, 1.0]])
    res = cma_es(batched_loss(_quad), z0, CmaConfig(sigma0=1.0, max_evals=1500), seed=3)
    np.testing.assert_allclose(res.z_best, np.tile([0.5, -1.0, 2.0], (2, 1)), atol=1e-3)
    lam = res.extra["popsize"]
    assert lam == 4 + int(np.floor(3 * np.log(3)))
    assert res.n_forward == (1500 // lam) * lam and res.n_grad == 0

    def rosen(z):
        return jnp.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2)

    r2 = cma_es(batched_loss(rosen), np.zeros((2, 4)), CmaConfig(sigma0=0.5, max_evals=8000), seed=5)
    np.testing.assert_allclose(r2.z_best, np.ones((2, 4)), atol=1e-2)


def test_random_search_keeps_the_best():
    res = random_search(batched_loss(_quad), [-3, -3, -3], [3, 3, 3], 2, 256, batch=16, seed=0)
    assert res.n_forward == 256
    assert np.all(np.diff(res.loss_history, axis=0) <= 0.0)
    np.testing.assert_allclose(np.asarray(jax.vmap(_quad)(jnp.asarray(res.z_best))), res.loss_best)


def test_secant_gradient_smooth_and_staircase():
    vg = secant_gradient(_quad, [0, 2], [1e-3, 1e-3])
    z = jnp.array([1.0, 2.0, -1.0])
    v, g = vg(z)
    _, g_ad = jax.value_and_grad(_quad)(z)
    assert float(v) == float(_quad(z))
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_ad), rtol=1e-6 if X64 else 1e-2)

    def stairs(z):
        return jnp.floor(4.0 * z[0]) + z[1] ** 2

    v, g = secant_gradient(stairs, [0], [0.5])(jnp.array([0.3, 1.0]))
    assert float(jax.grad(stairs)(jnp.array([0.3, 1.0]))[0]) == 0.0
    assert float(g[0]) == pytest.approx(4.0 / 1.0 * 1.0, rel=0.5)  # secant over 2 steps / width 1
    assert float(g[1]) == pytest.approx(2.0)


# ------------------------------------------------------------------------------ trust report
def _mixed(theta):
    x, y, u, w = theta[0], theta[1], theta[2], theta[3]
    return jnp.stack(
        [
            x**2 + jnp.sin(y),  # smooth
            jnp.floor(5.0 * u) + 0.0 * x,  # staircase in u: AD 0, value moves
            jnp.where(w > 0.5, 3.0, 0.0) + w,  # one jump at w = 0.5
            jnp.asarray(7.0) + 0.0 * x,  # constant
        ]
    )


@pytest.mark.skipif(not X64, reason="finite differences at a 1e-5 step need float64")
@pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)")
def test_trust_report_classes():
    x = np.array([1.0, 0.2, 0.43, 0.47])
    lo, hi = np.zeros(4), np.ones(4) * 2.0
    rep = trust_report(_mixed, x, lo, hi, ["x", "y", "u", "w"], ["smooth", "stairs", "jump", "const"])
    p = rep["params"]
    assert p["x"]["outputs"]["smooth"]["class"] == "smooth" and p["x"]["outputs"]["smooth"]["level"] == 3
    assert p["y"]["outputs"]["smooth"]["level"] == 3
    st = p["u"]["outputs"]["stairs"]
    assert st["class"] == "step" and st["ad"] == 0.0 and st["zero_frac"] == 1.0 and st["zero_run_moves"]
    assert st["fd_large"] != 0.0 and st["level"] < 3
    jp = p["w"]["outputs"]["jump"]
    assert jp["class"] == "jumpy" and jp["n_jumps"] == 1 and jp["level"] < 3
    assert p["x"]["outputs"]["const"]["class"] == "inert"
    assert rep["outputs"] == ["smooth", "stairs", "jump", "const"]


def _quantised(theta):
    """A smooth term plus Fortran's ``REAL(INT(x*1000))/1000`` (straight-through in ``ste``)."""
    from agrijax.core.grad import trunc_st

    x = theta[0]
    return jnp.stack([x**2 + trunc_st(x * 1000.0) / 1000.0])


@pytest.mark.skipif(not X64, reason="finite differences at a 1e-5 step need float64")
@pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)")
def test_a_straight_through_derivative_is_checked_on_both_of_its_paths():
    """The ste derivative of a quantised function is the derivative of the unrounded model (2x + 1),
    not the program's own (2x, which a small-step difference between two quanta sees): level 2 tests
    the exact-mode derivative against the model's small-step difference and the unrounded model's
    derivative against the unrounded model's; the large step and the scan test the straight-through
    value on the real model (the G2 / G3 case of the DSSAT day, agrijax.facade_grad)."""
    from agrijax.calib.trust import counterparts
    from agrijax.core.grad import bind_gradient_mode, bind_unrounded

    x, lo, hi = np.array([0.4305]), np.zeros(1), np.ones(1) * 2.0
    bound = bind_gradient_mode(_quantised, "ste")
    assert counterparts(_quantised) == (None, None)
    fx, fu = counterparts(bound)
    assert fx is not None and fu is not None
    assert float(jax.jit(bind_unrounded(_quantised))(jnp.asarray(x))[0]) == pytest.approx(0.4305**2 + 0.4305)
    r = trust_report(bound, x, lo, hi, ["x"], ["y"])["params"]["x"]["outputs"]["y"]
    assert r["ad"] == pytest.approx(2 * 0.4305 + 1.0) and r["ad_exact"] == pytest.approx(2 * 0.4305)
    assert r["ad_unrounded"] == pytest.approx(r["ad"]) and abs(r["ste_offset"]) < 1e-12
    assert r["fd_small"] == pytest.approx(2 * 0.4305, rel=1e-6)  # between two quanta: the exact slope
    assert r["fd_small_unrounded"] == pytest.approx(2 * 0.4305 + 1.0, rel=1e-6)
    assert r["rel_err_small"] < 1e-6 and r["rel_err_large"] < 0.05
    assert (r["class"], r["level"], r["level2"]) == ("smooth", 3, "pass")
    # without the diagnostics (as the calibration runs it): no unrounded model, the same verdicts
    q = trust_report(bound, x, lo, hi, ["x"], ["y"], diagnostics=False)["params"]["x"]["outputs"]["y"]
    assert "ad_unrounded" not in q and (q["level"], q["level2"], q["ad_exact"]) == (3, "pass", r["ad_exact"])
    # unbound (no counterparts to test the program with): the surrogate fails the small steps, which
    # agree with each other (a derivative error as far as the check can tell)
    plain = trust_report(_quantised, x, lo, hi, ["x"], ["y"])["params"]["x"]["outputs"]["y"]
    assert plain["ad_exact"] == plain["ad"] and "ad_unrounded" not in plain
    assert (plain["level"], plain["level2"]) == (1, "fail")


@pytest.mark.skipif(not X64, reason="finite differences at a 1e-5 step need float64")
@pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)")
def test_level2_is_undecidable_when_the_small_steps_disagree_and_level3_decides():
    """Next to a quantum the 1e-4 difference straddles it and the 1e-5 / 1e-6 ones do not: the small-step
    differences disagree with each other, level 2 is undecidable (no step is picked), and the secants
    at 1 / 2 / 5 % and the scan decide: level 3 for the straight-through derivative, which predicts
    them; a derivative that does not (the exact one, 2x, read as if it were the one used) gets level 1."""
    from agrijax.calib.trust import L2_UNDECIDABLE
    from agrijax.core.grad import bind_gradient_mode

    x, lo, hi = np.array([0.43001]), np.zeros(1), np.ones(1) * 2.0  # 0.43001 - 2e-4 crosses 0.430
    r = trust_report(bind_gradient_mode(_quantised, "ste"), x, lo, hi, ["x"], ["y"])["params"]["x"]
    r = r["outputs"]["y"]
    assert r["level2"] == L2_UNDECIDABLE and r["fd_spread_small"] > 1e-3
    assert r["rel_err_large_max"] < 0.05 and r["level"] == 3
    exact = trust_report(bind_gradient_mode(_quantised, "exact"), x, lo, hi, ["x"], ["y"])["params"]["x"]
    assert exact["outputs"]["y"]["level2"] == L2_UNDECIDABLE and exact["outputs"]["y"]["level"] == 1


@pytest.mark.skipif(not X64, reason="finite differences at a 1e-5 step need float64")
@pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)")
def test_fd_check_and_line_scan():
    x = np.array([1.0, 0.2, 0.43, 0.47])
    fd = fd_check(_mixed, x, np.ones(4) * 2.0, TrustConfig())
    assert fd["ad"].shape == (4, 4)
    np.testing.assert_allclose(fd["ad"][0], [2.0, np.cos(0.2), 0.0, 0.0], rtol=1e-12)
    assert bool(fd["agree0"][0, 0]) and bool(fd["agree0"][0, 1])
    sc = line_scan(_mixed, x, 0, 0.5, 1.5, TrustConfig(n_scan=11))
    np.testing.assert_allclose(sc["g"][:, 0], 2.0 * sc["grid"], rtol=1e-12)
    assert sc["n_jumps"][0] == 0 and sc["finite"].all()
    np.testing.assert_allclose(sc["secant"][0], 2.0, rtol=1e-12)  # (1.5^2 - 0.5^2) / 1


# ------------------------------------------------------------------------------ domains and chains
def test_domain_is_checked_against_the_bounds():
    pos = positive("divides x")
    assert ParamSpec("p", 0.5, 2.0, domain=pos).domain is pos
    with pytest.raises(ValueError, match="valid domain"):
        ParamSpec("p", 0.0, 2.0, domain=pos)  # the logit reaches 0, the open end
    ParamSpec("p", 0.0, 2.0, transform="log", domain=pos)  # log: lower + exp(z) > 0
    with pytest.raises(ValueError, match="valid domain"):
        ParamSpec("p", 0.1, 2.0, transform="log", domain=Domain(0.0, 5.0))  # log is unbounded above
    with pytest.raises(ValueError, match="identity"):
        ParamSpec("p", 0.1, 2.0, transform="none", domain=pos)
    with pytest.raises(ValueError):
        Domain(1.0, 1.0)
    sp = ParamSpace([ParamSpec("p", 0.5, 2.0, domain=pos, label="p")])
    assert sp.violations([1.0]) == []
    assert len(sp.violations([-1.0])) == 2  # outside the bounds and the domain
    assert sp.table()[0]["domain"] == "(0, inf]" and sp.table()[0]["domain_source"] == "divides x"


def _soil_space():
    specs, chain = layer_water_chain(1, (0.05, 0.30), (0.10, 0.45), (0.30, 0.60), "test box")
    return ParamSpace([ParamSpec("other", 0.0, 1.0), *specs], ordered=[chain]), chain


def test_ordered_chain_keeps_the_order_for_every_z():
    sp, chain = _soil_space()
    assert chain.gap == 0.01 and sp.names == ("other", "LL1", "DUL1", "SAT1")
    rng = np.random.default_rng(0)
    z = rng.normal(0.0, 6.0, (2000, 4))
    z[:10, 1:] = [[40.0, -40.0, -40.0]] * 10  # LL at its top, DUL and SAT pushed to their floor
    th = np.asarray(sp.to_physical(jnp.asarray(z)))
    tol = 1e-12 if X64 else 1e-6
    assert np.all(th[:, 2] >= th[:, 1] + chain.gap - tol) and np.all(th[:, 3] >= th[:, 2] + chain.gap - tol)
    assert np.all(th >= sp.lower - tol) and np.all(th <= sp.upper + tol)
    ok = np.all(np.abs(z) < 5.0, axis=1)
    back = np.asarray(sp.to_unconstrained(jnp.asarray(th[ok])))
    np.testing.assert_allclose(back, z[ok], rtol=1e-6 if X64 else 1e-2, atol=1e-6 if X64 else 1e-2)
    # increasing in each coordinate, finite Jacobian (lower triangular)
    jac = np.asarray(jax.jacfwd(sp.to_physical)(jnp.asarray([0.2, -0.3, 0.1, 0.4])))
    assert np.all(np.isfinite(jac)) and np.all(np.diag(jac) > 0.0)
    assert np.allclose(np.triu(jac, 1), 0.0)
    assert sp.violations(th[20]) == []
    # saturated z: never past a bound (clipped after the transform), float32 values pass
    sat = np.asarray(sp.to_physical(jnp.asarray([[40.0, 40.0, 40.0, 40.0], [-40.0, -40.0, -40.0, -40.0]])))
    assert np.all(sat <= sp.upper.astype(sat.dtype)) and np.all(
        sat >= sp.lower.astype(sat.dtype)
    )  # bounds as stored
    assert sp.violations(sat[0]) == [] and sp.violations(sat[1]) == []
    assert sp.violations(np.asarray(th[20], dtype=np.float32)) == []
    near = np.array(sat[0], dtype=np.float64)
    near[3] = float(np.nextafter(np.float32(sp.upper[3]), np.float32(1.0)))  # 1 float32 ulp past SAT's upper
    assert sp.violations(near) != [] and sp.violations(near, dtype=np.float32) == []
    assert np.all(np.isfinite(np.asarray(jax.jacfwd(sp.to_physical)(jnp.asarray([40.0, 40.0, 40.0, 40.0])))))
    assert any("not >=" in v for v in sp.violations([0.5, 0.25, 0.2, 0.5]))


def test_ordered_chain_rejects_inconsistent_boxes():
    specs, chain = layer_water_chain(0, (0.05, 0.40), (0.10, 0.40), (0.30, 0.60), "test box")
    with pytest.raises(ValueError, match="upper"):
        ParamSpace(specs, ordered=[chain])
    specs, chain = layer_water_chain(0, (0.05, 0.30), (0.10, 0.45), (0.30, 0.60), "test box")
    with pytest.raises(ValueError, match="chain order"):
        ParamSpace(specs[::-1], ordered=[chain])
    with pytest.raises(ValueError):
        OrderedChain(("a", "b"), 0.0)
    with pytest.raises(ValueError):
        layer_water_chain(0, (0.05, 0.30), (0.10, 0.45), (0.30, 0.60), "")


def test_index_writes_one_layer():
    sp, _ = _soil_space()
    tree = {
        "other": jnp.asarray(0.5),
        "soil": {k: jnp.asarray([0.1, 0.2, 0.3]) for k in ("ll", "dul", "sat")},
    }
    out = sp.apply(tree, jnp.asarray([0.4, 0.11, 0.22, 0.44]))
    rt = 0.0 if X64 else 1e-6  # float32 leaves hold the float32 rounding of the values
    np.testing.assert_allclose(np.asarray(out["soil"]["ll"]), [0.1, 0.11, 0.3], rtol=rt)
    np.testing.assert_allclose(np.asarray(out["soil"]["sat"]), [0.1, 0.44, 0.3], rtol=rt)
    np.testing.assert_allclose(np.asarray(sp.get(out)), [0.4, 0.11, 0.22, 0.44], rtol=rt)


def test_dssat_soil_specs_carry_sources():
    cn, sw = cn_spec(), swcon_spec()
    assert (cn.lower, cn.upper) == CN_BOUNDS and cn.path == "soil.cn" and "Ritchie" in cn.bounds_source
    assert cn.domain.contains(25.0, 98.0) and "SOILDYN.for" in cn.domain.source
    assert (sw.lower, sw.upper) == SWCON_BOUNDS and sw.domain.lower_open
    with pytest.raises(ValueError, match="lower bound"):
        u_spec(2.0, "")
    u = u_spec(2.0, "site choice for the test", path="evap.u")
    assert u.upper == 12.0 and "FAO 1990" in u.bounds_source and u.domain.lower_open


# ------------------------------------------------------------------------------ pairs
def _two_treatments(theta):
    t = jnp.arange(4.0)[:, None]
    smooth = theta[0] * t + theta[1] ** 2 + jnp.zeros((4, 1))
    stairs = jnp.floor(4.0 * theta[0]) * t + theta[1] ** 2 + jnp.zeros((4, 1))
    return {"y": jnp.concatenate([smooth, stairs], axis=1)}


def test_treatment_losses_sum_to_the_objective():
    mask = np.zeros((4, 2), bool)
    mask[1:, :] = True
    tg = (
        Target("yf", "y", "final", rel_scale=0.1, weight=2.0),
        Target("ys", "y", "series", scale=1.0, mask=mask),
    )
    obs = {k: np.asarray(v) + 0.3 for k, v in {"yf": _two_treatments(jnp.asarray([1.0, 1.0]))["y"][-1],
                                                 "ys": _two_treatments(jnp.asarray([1.0, 1.0]))["y"]}.items()}  # fmt: skip
    obj = Objective((Group("g", _two_treatments, tg, obs, treatments=("a", "b")),))
    th = jnp.asarray([1.3, 0.7])
    per = np.asarray(obj.treatment_losses(th))
    assert per.shape == (2,) and float(np.sum(per)) == pytest.approx(
        float(obj(th)), rel=1e-12 if X64 else 1e-5
    )
    assert obj.treatment_names(th) == ("g/a", "g/b")


def test_gradient_plan_falls_back_per_pair():
    def rep(cls_a, cls_b, lev_a=3, lev_b=3):
        o = lambda c, lev: {"class": c, "level": lev, "n_jumps": 2 if c == "jumpy" else 0}  # noqa: E731
        return {"outputs": {"t1/y": o(cls_a, lev_a), "t2/y": o(cls_b, lev_b), "t2/z": o("inert", 3)}}

    report = {
        "outputs": ["t1/y", "t2/y", "t2/z"],
        "params": {"G3": rep("smooth", "jumpy", 3, 2), "RUE": rep("smooth", "smooth"), "P1": rep("step", "step", 2, 2),
                   "P5": rep("smooth", "smooth")},
    }  # fmt: skip
    plan = gradient_plan(report, {"t1": ["t1/y"], "t2": ["t2/y", "t2/z"]}, derivative_free=("P5",))
    assert plan.treatments == ("t1", "t2") and plan.params == ("G3", "RUE", "P1", "P5")
    np.testing.assert_array_equal(plan.use_ad, [[True, True, False, False], [False, True, False, False]])
    assert plan.method == {
        "G3": "hybrid",
        "RUE": "gradient",
        "P1": "derivative_free",
        "P5": "derivative_free",
    }
    assert plan.gradient_params == ("G3", "RUE") and plan.derivative_free_params == ("P1", "P5")
    assert any("jumpy" in r for r in plan.reasons["G3"]) and plan.reasons["P5"] == [
        "derivative-free by default"
    ]
    with pytest.raises(ValueError):
        gradient_plan(report, {"t1": ["nope"]})


def test_pair_gradient_is_ad_on_trusted_pairs_and_secant_elsewhere():
    def losses(z):
        smooth = (z[0] - 1.0) ** 2 + z[1] ** 2
        stairs = jnp.floor(4.0 * z[0]) + 3.0 * z[1]
        return jnp.stack([smooth, stairs])

    z = jnp.asarray([0.3, 0.5])
    mask = np.array([[False, False], [True, False]])  # treatment 2 x parameter 0: secant
    v, g = pair_gradient(losses, mask, 0.5)(z)
    assert float(v) == pytest.approx(float(jnp.sum(losses(z))))
    sec = (np.floor(4.0 * 0.8) - np.floor(4.0 * -0.2)) / 1.0
    np.testing.assert_allclose(
        np.asarray(g), [2.0 * (0.3 - 1.0) + sec, 2.0 * 0.5 + 3.0], rtol=1e-12 if X64 else 1e-5
    )
    _, g0 = pair_gradient(losses, np.zeros((2, 2), bool), 0.5)(z)
    np.testing.assert_allclose(
        np.asarray(g0), [2.0 * (0.3 - 1.0), 2.0 * 0.5 + 3.0], rtol=1e-12 if X64 else 1e-5
    )
    _, gb = jax.jit(jax.vmap(pair_gradient(losses, mask, 0.5)))(jnp.stack([z, z]))
    np.testing.assert_allclose(np.asarray(gb[1]), np.asarray(g), rtol=1e-12 if X64 else 1e-5)
