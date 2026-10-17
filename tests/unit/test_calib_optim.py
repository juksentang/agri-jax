"""Batched L-BFGS and Levenberg-Marquardt (agrijax.calib.optim) on functions with known answers.

* a quadratic: the L-BFGS directions with an exact line search reach the minimum in ``n`` steps;
  L-BFGS with its Armijo line search reaches it exactly at its second step when the Hessian is a
  multiple of the identity (the first pair gives the exact inverse Hessian), and within a few
  steps per dimension otherwise; LM solves a linear least-squares problem;
* the L-BFGS direction: the two-loop recursion equals the explicit inverse-Hessian update with
  ``H0 = (s.y / y.y) I``; a pair violating the curvature condition is not stored; a direction that
  is not a descent direction falls back to steepest descent and clears the memory;
* stop rules: each status (line search, rejects, damping, stall, ``ftol``, ``xtol``, gradient,
  ``max_iter``) with its exact call counts; LM uses Marquardt's scaling ``diag(J'J)``; the
  Armijo slope term, each clause of the MINPACK ``ftol`` test, the reset of every "in a row"
  counter, a target met with equality; an L-BFGS row at the minimum of an inconsistent
  least-squares problem stops on ``ftol`` instead of running to ``max_iter``; an LM ``xtol`` stop
  after a run of rejections is told apart by ``rejects_in_a_row``;
* the L-BFGS memory: more pairs than ``memory`` (the ring buffer wraps) gives the explicit BFGS
  update on the last ``memory`` pairs;
* LM against an independent NumPy reimplementation of its default stop rules (the plain
  per-row damping loop, BLAS sums): with the default settings, the same accept / reject
  decisions, statuses and call counts per row on a noisy exponential fit, and the iterates within
  rounding;
* the Rosenbrock function: both agree with the SciPy reference solutions (``L-BFGS-B`` and
  MINPACK ``lm``; SciPy is used only here);
* rows: with an objective whose rows do not depend on the batch (NumPy, elementwise), a row
  follows the same iterates, losses, counts and stop decision inside a batch as alone, bit for
  bit; with a compiled ``jit(vmap)`` objective, a row matches the same-shape run where only that
  row is active bit for bit (CPU), and a run at another batch size only within rounding; a row
  with a non-finite start or another start does not change the others; a stopped row keeps its
  state bit for bit and is no longer counted, and a row frozen from the start is evaluated in the
  fixed-shape batch but its results are discarded and never counted;
* counts: on one row the per-row counts equal the calls the optimiser made, including the forward
  calls inside a derivative call (secant columns, :func:`pair_gradient`); a budget is never
  exceeded and stops the row; a target stops the row at the first point reaching it.

Tests with tolerances at double-precision rounding or with exact call counts that depend on it
run only with float64 enabled (the float32 CI job skips them).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib import (
    LM_MINPACK_TOL,
    LbfgsConfig,
    LmConfig,
    batched_loss,
    batched_residual_jacobian,
    batched_value_and_grad,
    lbfgs,
    lm,
)

X64 = jax.config.jax_enable_x64


def needs_x64(fn):
    """Tolerances at double-precision rounding, or exact call counts that depend on it (the CI
    float32 pass skips these)."""
    fn = pytest.mark.skipif(not X64, reason="tolerances and call counts at float64 rounding")(fn)
    return pytest.mark.allow_skip(reason="float64 rounding (CI float32 pass)")(fn)


_C = jnp.array([0.5, -1.0, 2.0, 0.25])
_H = jnp.array([1.0, 4.0, 0.25, 2.0])


def _quad(z):
    return 0.5 * jnp.sum(_H * (z - _C) ** 2)


def _iso(z):
    return 1.5 * jnp.sum((z - _C) ** 2)


def _rosen(z):
    return jnp.sum(100.0 * (z[1:] - z[:-1] ** 2) ** 2 + (1.0 - z[:-1]) ** 2)


def _np_rosen3(z):
    """The 3-D Rosenbrock function and its gradient in NumPy, elementwise over the rows (no
    reduction, no compiled fused multiply-add), so that a row's value does not depend on the
    batch: the row tests check the optimisers alone."""
    z = np.asarray(z, dtype=float)
    z0, z1, z2 = z[:, 0], z[:, 1], z[:, 2]
    a0, a1 = z1 - z0**2, z2 - z1**2
    v = 100.0 * a0**2 + (1.0 - z0) ** 2 + 100.0 * a1**2 + (1.0 - z1) ** 2
    g = np.stack(
        [-400.0 * z0 * a0 - 2.0 * (1.0 - z0), 200.0 * a0 - 400.0 * z1 * a1 - 2.0 * (1.0 - z1), 200.0 * a1],
        axis=1,
    )
    return v, g


def _np_rosen3_loss(z):
    return _np_rosen3(z)[0]


def _np_rosen3_res(z):
    """Residuals of the 3-D Rosenbrock function (NumPy, elementwise over the rows)."""
    z = np.asarray(z, dtype=float)
    return np.stack(
        [10.0 * (z[:, 1] - z[:, 0] ** 2), 10.0 * (z[:, 2] - z[:, 1] ** 2), 1.0 - z[:, 0], 1.0 - z[:, 1]],
        axis=1,
    )


def _np_rosen3_rj(z):
    z = np.asarray(z, dtype=float)
    jac = np.zeros((z.shape[0], 4, 3))
    jac[:, 0, 0] = -20.0 * z[:, 0]
    jac[:, 0, 1] = 10.0
    jac[:, 1, 1] = -20.0 * z[:, 1]
    jac[:, 1, 2] = 10.0
    jac[:, 2, 0] = -1.0
    jac[:, 3, 1] = -1.0
    return _np_rosen3_res(z), jac


def _rosen_res(z):
    return jnp.concatenate([10.0 * (z[1:] - z[:-1] ** 2), 1.0 - z[:-1]])


_A = jnp.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0], [3.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.5, -2.0, 0.0]])
_B = jnp.array([1.0, -2.0, 0.5, 3.0, 1.0])


def _lin_res(z):
    return _A @ z - _B


def _counting(fn):
    """``fn`` with a call counter (``.calls``)."""

    def wrapped(z):
        wrapped.calls += 1
        return fn(z)

    wrapped.calls = 0
    return wrapped


_STARTS = np.array(
    [[0.0, 0.0, 0.0, 0.0], [3.0, -2.0, 1.0, 4.0], [-1.5, 2.5, -3.0, 0.5], [1.0, 1.0, 1.0, 1.0]]
)


# ------------------------------------------------------------------------------ convergence
@needs_x64
def test_lbfgs_isotropic_quadratic_exact_at_second_step():
    res = lbfgs(
        batched_value_and_grad(_iso), batched_loss(_iso), _STARTS, LbfgsConfig(gtol=1e-12, max_iter=10)
    )
    np.testing.assert_allclose(res.z_best, np.tile(np.asarray(_C), (4, 1)), atol=1e-12)
    # first step: steepest descent of unit length; second: the exact quasi-Newton step
    np.testing.assert_array_equal(res.extra["accepted"], 2)
    np.testing.assert_array_equal(res.extra["status"], "gradient")


def test_lbfgs_quadratic_converges():
    n = _STARTS.shape[1]
    res = lbfgs(
        batched_value_and_grad(_quad), batched_loss(_quad), _STARTS, LbfgsConfig(gtol=1e-10, max_iter=200)
    )
    np.testing.assert_allclose(res.z_best, np.tile(np.asarray(_C), (4, 1)), atol=1e-9)
    np.testing.assert_array_equal(res.extra["status"], "gradient")
    assert np.all(res.extra["accepted"] <= 4 * n), res.extra["accepted"]
    # accepted losses never increase
    assert np.all(np.diff(res.loss_history, axis=0) <= 0.0)


def test_lm_linear_least_squares():
    z0 = np.array([[0.0, 0.0, 0.0], [5.0, -5.0, 2.0]])
    res = lm(batched_residual_jacobian(_lin_res), batched_loss(_lin_res), z0, LmConfig(max_iter=60))
    ref = np.linalg.lstsq(np.asarray(_A), np.asarray(_B), rcond=None)[0]
    np.testing.assert_allclose(res.z_best, np.tile(ref, (2, 1)), atol=1e-8)
    assert np.all(res.extra["accepted"] >= 1)


@pytest.mark.parametrize("z0", [[-1.2, 1.0], [0.0, 0.0, 0.0, 0.0]])
def test_lbfgs_rosenbrock_matches_scipy(z0):
    from scipy.optimize import minimize

    def fun(x):
        v, g = jax.value_and_grad(_rosen)(jnp.asarray(x))
        return float(v), np.asarray(g)

    ref = minimize(
        fun, np.asarray(z0, float), jac=True, method="L-BFGS-B", options={"gtol": 1e-10, "ftol": 0.0}
    )
    res = lbfgs(
        batched_value_and_grad(_rosen),
        batched_loss(_rosen),
        np.asarray([z0], float),
        LbfgsConfig(gtol=1e-9, ftol=0.0, max_iter=2000),
    )
    np.testing.assert_allclose(res.z_best[0], ref.x, atol=1e-6)
    np.testing.assert_allclose(res.z_best[0], np.ones(len(z0)), atol=1e-6)
    assert res.loss_best[0] < 1e-12


@pytest.mark.parametrize("z0", [[-1.2, 1.0], [0.0, 0.0, 0.0, 0.0]])
def test_lm_rosenbrock_matches_scipy(z0):
    from scipy.optimize import least_squares

    ref = least_squares(
        lambda x: np.asarray(_rosen_res(jnp.asarray(x))),
        np.asarray(z0, float),
        jac=lambda x: np.asarray(jax.jacfwd(_rosen_res)(jnp.asarray(x))),  # pyright: ignore[reportArgumentType]
        method="lm",
        xtol=1e-15,
        ftol=1e-15,
        gtol=1e-15,
    )
    res = lm(
        batched_residual_jacobian(_rosen_res),
        batched_loss(_rosen_res),
        np.asarray([z0], float),
        LmConfig(max_iter=500),
        target=1e-24,
    )
    np.testing.assert_allclose(res.z_best[0], ref.x, atol=1e-8)
    np.testing.assert_allclose(res.z_best[0], np.ones(len(z0)), atol=1e-8)


# ------------------------------------------------------------------------------ rows
def _run(kind, z0, cfg=None, **kw):
    if kind == "lbfgs":
        return lbfgs(_np_rosen3, _np_rosen3_loss, z0, cfg or LbfgsConfig(gtol=1e-8), **kw)
    return lm(_np_rosen3_rj, _np_rosen3_res, z0, cfg or LmConfig(), **kw)


def _assert_row_same(batch, i, single, j=0):
    """Row ``i`` of ``batch`` = row ``j`` of ``single``, bit for bit; after the single run ended
    the batch row is frozen (its entries repeat the last one)."""
    k = single.loss_history.shape[0]
    assert batch.loss_history.shape[0] >= k
    np.testing.assert_array_equal(batch.loss_history[:k, i], single.loss_history[:, j])
    np.testing.assert_array_equal(batch.z_history[:k, i], single.z_history[:, j])
    np.testing.assert_array_equal(batch.loss_history[k:, i], single.loss_history[-1, j])
    np.testing.assert_array_equal(
        batch.z_history[k:, i], np.broadcast_to(single.z_history[-1, j], batch.z_history[k:, i].shape)
    )
    np.testing.assert_array_equal(batch.z_best[i], single.z_best[j])
    for key in ("n_forward_rows", "n_grad_rows", "status", "accepted"):
        np.testing.assert_array_equal(batch.extra[key][i], single.extra[key][j])


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_row_in_batch_same_as_alone_bitwise(kind):
    z0 = _STARTS[:, :3] * 0.5
    target = np.array([-np.inf, 1e-6, -np.inf, 1e-3])
    batch = _run(kind, z0, target=target)
    assert len(set(batch.extra["n_forward_rows"].tolist())) > 1  # the rows stop at different rounds
    for i in range(z0.shape[0]):
        single = _run(kind, z0[i : i + 1], target=target[i : i + 1])
        _assert_row_same(batch, i, single)


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_rows_do_not_affect_each_other(kind):
    z0 = _STARTS[:, :3] * 0.5
    ref = _run(kind, z0)
    other = z0.copy()
    other[1] = np.nan  # non-finite start: stops at once
    other[3] = [2.0, -1.0, 0.3]  # another start
    alt = _run(kind, other)
    assert alt.extra["status"][1] == "nonfinite"
    assert alt.extra["n_forward_rows"][1] == 0 and alt.extra["n_grad_rows"][1] == 1
    for i in (0, 2):
        k = min(ref.loss_history.shape[0], alt.loss_history.shape[0])
        np.testing.assert_array_equal(ref.loss_history[:k, i], alt.loss_history[:k, i])
        np.testing.assert_array_equal(ref.z_best[i], alt.z_best[i])
        np.testing.assert_array_equal(ref.extra["n_forward_rows"][i], alt.extra["n_forward_rows"][i])
        np.testing.assert_array_equal(ref.extra["n_grad_rows"][i], alt.extra["n_grad_rows"][i])


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_frozen_rows_unchanged_and_not_counted(kind):
    z0 = _STARTS[:, :3] * 0.5
    active = np.array([True, False, True, False])
    res = _run(kind, z0, active=active)
    for i in (1, 3):
        np.testing.assert_array_equal(res.z_best[i], z0[i])
        np.testing.assert_array_equal(res.z_history[:, i], np.broadcast_to(z0[i], res.z_history[:, i].shape))
        assert res.extra["status"][i] == "frozen"
        assert res.extra["n_forward_rows"][i] == 0 and res.extra["n_grad_rows"][i] == 0
    # the active rows are those of a run without the frozen ones
    sub = _run(kind, z0[active])
    for j, i in enumerate(np.nonzero(active)[0]):
        _assert_row_same(res, i, _run(kind, z0[i : i + 1]))
        np.testing.assert_array_equal(res.z_best[i], sub.z_best[j])


# ------------------------------------------------------------------------------ counts
@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_counts_equal_calls_on_one_row(kind):
    if kind == "lbfgs":
        der = _counting(batched_value_and_grad(_rosen))
        fwd = _counting(batched_loss(_rosen))
        res = lbfgs(der, fwd, np.array([[-1.2, 1.0]]), LbfgsConfig(gtol=1e-8))
        acc = int(res.extra["accepted"][0])
        # one gradient at the start and one per accepted step (none after the last if it stopped there)
        assert res.extra["n_grad_rows"][0] in (acc, acc + 1)
    else:
        der = _counting(batched_residual_jacobian(_rosen_res))
        fwd = _counting(batched_loss(_rosen_res))
        res = lm(der, fwd, np.array([[-1.2, 1.0]]), LmConfig())
    assert res.extra["n_forward_rows"][0] == fwd.calls == res.n_forward
    assert res.extra["n_grad_rows"][0] == der.calls == res.n_grad


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_budget_never_exceeded(kind):
    z0 = _STARTS[:, :3] * 0.5
    budget = np.array([7.0, 20.0, 13.5, 1e9])
    res = _run(kind, z0, budget=budget, grad_cost=2.5)
    cost = res.extra["n_forward_rows"] + 2.5 * res.extra["n_grad_rows"]
    assert np.all(cost <= budget)
    assert np.all(res.extra["status"][:3] == "budget")
    assert res.extra["status"][3] != "budget"
    for i in range(3):  # the row stops when its next call does not fit any more
        assert budget[i] - cost[i] < 2.5
    with pytest.raises(ValueError, match="grad_cost"):
        _run(kind, z0, budget=10.0)


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_target_stops_at_first_point_reaching_it(kind):
    z0 = _STARTS[:, :3] * 0.5
    free = _run(kind, z0)
    target = 1e-2
    res = _run(kind, z0, target=target)
    for i in range(z0.shape[0]):
        assert res.extra["status"][i] == "target"
        assert res.loss_best[i] <= target
        # the free run's history passes the same points: the stop is its first loss <= target
        k = int(np.argmax(free.loss_history[:, i] <= target))
        np.testing.assert_array_equal(res.loss_best[i], free.loss_history[k, i])
    # a target met at the start: one derivative call, no forward call
    at = _run(kind, np.ones((1, 3)), target=1e-30)
    assert at.extra["status"][0] == "target"
    assert at.extra["n_grad_rows"][0] == 1 and at.extra["n_forward_rows"][0] == 0


# ------------------------------------------------------------------------------ cost / time history
def _all_optimisers():
    """One run of every optimiser of the module (3 rows; L-BFGS and LM with a frozen row)."""
    from agrijax.calib import AdamConfig, CmaConfig, adam, cma_es, random_search

    z0 = _STARTS[:3, :3] * 0.5
    active = np.array([True, True, False])
    return {
        "adam": adam(batched_value_and_grad(_rosen), z0, AdamConfig(steps=25), forward_per_step=2),
        "cma_es": cma_es(batched_loss(_rosen), z0, CmaConfig(max_evals=60), seed=1),
        "random_search": random_search(batched_loss(_rosen), [-2] * 3, [2] * 3, 3, 64, batch=16, seed=0),
        "lbfgs": _run("lbfgs", z0, active=active),
        "lm": _run("lm", z0, active=active),
    }


def test_cost_and_time_history_every_optimiser():
    for name, res in _all_optimisers().items():
        steps, r = res.loss_history.shape
        assert res.cost_history.shape == (steps, r, 2), name
        assert res.time_history.shape == (steps,), name
        assert np.all(np.diff(res.cost_history, axis=0) >= 0), name
        assert np.all(res.cost_history >= 0), name
        # a real clock: strictly increasing and positive from the first entry
        assert np.all(np.diff(res.time_history) > 0.0) and res.time_history[0] > 0.0, name
        assert res.time_history[-1] <= res.wall_s, name
        if name in ("lbfgs", "lm"):  # per-row totals; n_forward / n_grad the largest
            np.testing.assert_array_equal(res.cost_history[-1, :, 0], res.extra["n_forward_rows"])
            np.testing.assert_array_equal(res.cost_history[-1, :, 1], res.extra["n_grad_rows"])
            assert res.cost_history[-1, :, 0].max() == res.n_forward, name
            assert res.cost_history[-1, :, 1].max() == res.n_grad, name
            np.testing.assert_array_equal(res.cost_history[:, 2], 0)  # the frozen row is never counted
        else:  # every row makes every call
            np.testing.assert_array_equal(res.cost_history[-1, :, 0], res.n_forward)
            np.testing.assert_array_equal(res.cost_history[-1, :, 1], res.n_grad)
        fe = res.forward_equivalents(2.5)
        np.testing.assert_allclose(fe[-1], res.cost_history[-1, :, 0] + 2.5 * res.cost_history[-1, :, 1])
        assert np.all(np.diff(fe, axis=0) >= 0.0), name


def test_cost_history_steps_of_the_fixed_budget_optimisers():
    res = _all_optimisers()
    # Adam: one value-and-gradient and forward_per_step forwards per step
    np.testing.assert_array_equal(res["adam"].cost_history[:, 0, 1], np.arange(1, 26))
    np.testing.assert_array_equal(res["adam"].cost_history[:, 0, 0], 2 * np.arange(1, 26))
    # CMA-ES: lambda forwards per generation
    lam = res["cma_es"].extra["popsize"]
    np.testing.assert_array_equal(res["cma_es"].cost_history[:, 1, 0], lam * np.arange(1, 60 // lam + 1))
    # random search: one batch per round
    np.testing.assert_array_equal(res["random_search"].cost_history[:, 0, 0], 16 * np.arange(1, 5))
    # L-BFGS / LM: a row's counts stop growing once it stopped (its own run alone ends there)
    z0 = _STARTS[:3, :3] * 0.5
    for name in ("lbfgs", "lm"):
        ch = res[name].cost_history
        for i in range(2):
            alone = _run(name, z0[i : i + 1])
            k = alone.cost_history.shape[0]
            assert k < ch.shape[0] or i == int(np.argmax(res[name].extra["n_forward_rows"]))
            np.testing.assert_array_equal(ch[:k, i], alone.cost_history[:, 0])
            np.testing.assert_array_equal(ch[k - 1 :, i], np.broadcast_to(ch[k - 1, i], ch[k - 1 :, i].shape))


# ------------------------------------------------------------------------------ batch shape (JAX objective)
_R8 = np.random.default_rng(7).uniform(-1.5, 1.5, size=(8, 3))


def _jax_run(kind, z0, **kw):
    if kind == "lbfgs":
        return lbfgs(batched_value_and_grad(_rosen), batched_loss(_rosen), z0, LbfgsConfig(gtol=1e-8), **kw)
    return lm(batched_residual_jacobian(_rosen_res), batched_loss(_rosen_res), z0, LmConfig(), **kw)


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_jax_objective_row_same_as_single_active_row_same_shape(kind):
    """The bitwise single-row reference of a compiled objective: the same [R, n] batch with only
    that row active (one program shape, so the same rounding for the row)."""
    batch = _jax_run(kind, _R8)
    for i in range(_R8.shape[0]):
        single = _jax_run(kind, _R8, active=np.arange(_R8.shape[0]) == i)
        _assert_row_same(batch, i, single, j=i)


@needs_x64
@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_jax_objective_other_batch_size_within_rounding(kind):
    """At another batch size a compiled objective may round a row differently (XLA picks the
    contraction and reduction order per program shape; seen on CPU at R = 4 and 8 for this
    function), so the iterates of a row run alone may part from those in the batch: only the
    objective values within rounding and the converged answer are the same."""
    vg = batched_value_and_grad(_rosen)
    vb, gb = (np.asarray(a) for a in vg(jnp.asarray(_R8)))
    for i in range(_R8.shape[0]):
        v1, g1 = (np.asarray(a) for a in vg(jnp.asarray(_R8[i : i + 1])))
        np.testing.assert_allclose(vb[i], v1[0], rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(gb[i], g1[0], rtol=1e-12, atol=1e-12)
    batch = _jax_run(kind, _R8)
    for i in range(_R8.shape[0]):
        single = _jax_run(kind, _R8[i : i + 1])
        assert batch.extra["status"][i] == single.extra["status"][0]
        np.testing.assert_allclose(batch.z_best[i], single.z_best[0], atol=1e-6)
        np.testing.assert_allclose(batch.z_best[i], np.ones(3), atol=1e-6)


# ------------------------------------------------------------------------------ forwards inside derivative calls
def _np_rosen3_rj_secant(mask, dz=0.1):
    """Residuals and Jacobian of the 3-D Rosenbrock function with central-secant columns where
    ``mask [R, 3]`` is true (two residual evaluations per secant column, in one batch per column);
    ``.inner`` counts the residual evaluations made for each row."""

    def rj(z):
        z = np.asarray(z, dtype=float)
        res, jac = _np_rosen3_rj(z)
        for c in range(3):
            if mask[:, c].any():
                e = np.zeros(3)
                e[c] = dz
                d = (_np_rosen3_res(z + e) - _np_rosen3_res(z - e)) / (2.0 * dz)
                jac[:, :, c] = np.where(mask[:, c, None], d, jac[:, :, c])
                rj.inner = rj.inner + 2 * mask[:, c]
        return res, jac

    rj.inner = np.zeros(mask.shape[0], dtype=np.int64)
    return rj


_SEC_MASK = np.array([[False, False, False], [True, False, False], [True, False, True], [True, True, True]])


def test_lm_secant_columns_counted_per_row():
    """Each Jacobian call of a row adds two forward calls per secant column of that row to its
    forward count (a central difference per secant column)."""
    z0 = _STARTS[:, :3] * 0.5
    fpd = 2 * _SEC_MASK.sum(axis=1)
    batch = lm(_np_rosen3_rj_secant(_SEC_MASK), _np_rosen3_res, z0, LmConfig(), forward_per_derivative=fpd)
    for i in range(z0.shape[0]):
        inner = _np_rosen3_rj_secant(_SEC_MASK[i : i + 1])
        rj = _counting(inner)
        fwd = _counting(_np_rosen3_res)
        single = lm(rj, fwd, z0[i : i + 1], LmConfig(), forward_per_derivative=int(fpd[i]))
        assert single.extra["n_grad_rows"][0] == rj.calls
        # trial forwards plus the secant forwards made inside the Jacobian calls
        assert single.extra["n_forward_rows"][0] == fwd.calls + int(inner.inner[0])
        assert single.extra["n_forward_rows"][0] == fwd.calls + fpd[i] * rj.calls
        _assert_row_same(batch, i, single)


def test_lbfgs_pair_gradient_forwards_counted():
    from agrijax.calib import pair_gradient, pair_gradient_forwards

    def losses(z):
        return jnp.stack(
            [
                100.0 * (z[1] - z[0] ** 2) ** 2 + (1.0 - z[0]) ** 2,
                100.0 * (z[2] - z[1] ** 2) ** 2 + (1.0 - z[1]) ** 2,
            ]
        )

    mask = np.array([[True, False, False], [False, False, True]])
    k2 = pair_gradient_forwards(mask)
    assert k2 == 4
    assert pair_gradient_forwards(np.zeros((2, 3), bool)) == 0
    der = _counting(jax.jit(jax.vmap(pair_gradient(losses, mask, 0.1))))
    fwd = _counting(batched_loss(lambda z: jnp.sum(losses(z))))
    res = lbfgs(der, fwd, np.array([[-1.2, 1.0, 0.5]]), LbfgsConfig(max_iter=30), forward_per_derivative=k2)
    assert res.extra["n_grad_rows"][0] == der.calls
    assert res.extra["n_forward_rows"][0] == fwd.calls + k2 * der.calls
    np.testing.assert_array_equal(res.cost_history[-1, 0], [res.extra["n_forward_rows"][0], der.calls])


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_budget_counts_forwards_inside_derivative_calls(kind):
    z0 = _STARTS[:, :3] * 0.5
    budget = np.array([30.0, 40.0, 25.0, 1e9])
    fpd = np.array([0, 2, 4, 6])
    res = _run(kind, z0, budget=budget, grad_cost=2.5, forward_per_derivative=fpd)
    plain = _run(kind, z0)
    cost = res.extra["n_forward_rows"] + 2.5 * res.extra["n_grad_rows"]
    assert np.all(cost <= budget)
    np.testing.assert_array_equal(res.extra["status"][:3], "budget")
    assert np.all(budget[:3] - cost[:3] < 2.5 + fpd[:3])  # the next call did not fit
    # the unbudgeted row: same iterates as without the extra count, forwards raised by fpd per derivative
    np.testing.assert_array_equal(res.z_best[3], plain.z_best[3])
    assert (
        res.extra["n_forward_rows"][3] == plain.extra["n_forward_rows"][3] + 6 * plain.extra["n_grad_rows"][3]
    )
    for bad in (-1, 1.5, np.array([0, 1, -2, 0])):
        with pytest.raises(ValueError, match="forward_per_derivative"):
            _run(kind, z0, forward_per_derivative=bad)


# ------------------------------------------------------------------------------ L-BFGS direction
def _quad_np(hmat, c):
    def vg(z):
        dz = z - c
        return 0.5 * np.einsum("ri,ij,rj->r", dz, hmat, dz), dz @ hmat

    return vg


def _spd(n, cond, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, n)))
    return (q * np.geomspace(1.0, cond, n)) @ q.T


@pytest.mark.parametrize("n", [4, 6])
def test_lbfgs_directions_exact_line_search_converge_in_n_steps(n):
    """With an exact line search (test-only: alpha = -g.d / d'Hd), the L-BFGS directions on a
    strictly convex quadratic reach the minimum in n steps (memory >= n, so the directions are
    those of BFGS, which are conjugate under exact line searches: finite termination, Nocedal &
    Wright 2006, Chapter 6)."""
    from agrijax.calib.optim import _lbfgs_direction

    hmat = np.diag(np.asarray(_H)) if n == 4 else _spd(n, 100.0, 3)
    c = np.asarray(_C) if n == 4 else np.linspace(-1.0, 2.0, n)
    vg = _quad_np(hmat, c)
    r, m = 4, 10
    z = np.random.default_rng(n).uniform(-3.0, 3.0, size=(r, n))
    z_prev = np.zeros((r, n))
    g_prev = np.zeros((r, n))
    s_mem = np.zeros((r, m, n))
    y_mem = np.zeros((r, m, n))
    rho = np.zeros((r, m))
    head = np.zeros(r, dtype=np.int64)
    g0 = vg(z)[1]
    for k in range(n):
        g = vg(z)[1]
        s_mem, y_mem, rho, head, d, gd = _lbfgs_direction(
            np.ones(r, bool), np.full(r, k > 0), z, g, z_prev, g_prev, s_mem, y_mem, rho, head, 1.0
        )
        assert np.all(gd < 0.0)
        alpha = -gd / np.einsum("ri,ij,rj->r", d, hmat, d)
        z_prev, g_prev = z, g
        z = z + alpha[:, None] * d
    g = vg(z)[1]
    assert np.all(np.max(np.abs(g), axis=1) <= 1e-9 * np.max(np.abs(g0), axis=1))
    np.testing.assert_allclose(z, np.tile(c, (r, 1)), atol=1e-8)


def _dir(z, g, z_prev, g_prev, s_mem, y_mem, rho, head, has_prev=True):
    from agrijax.calib.optim import _lbfgs_direction

    return _lbfgs_direction(
        np.array([True]), np.array([has_prev]), z, g, z_prev, g_prev, s_mem, y_mem, rho, head, 1.0
    )


def test_lbfgs_direction_is_the_scaled_inverse_hessian_update():
    """One stored pair: d = -H1 g with H1 = V' (gamma I) V + rho s s', V = I - rho y s',
    gamma = s.y / y.y (Nocedal & Wright 2006, eqs. 6.17 and 7.20); a badly scaled pair so gamma
    is far from 1. A second pair with s.y < 0 violates the curvature condition: not stored, the
    direction uses the first pair."""
    n, m = 3, 5
    s = np.array([[1.0, 0.2, 0.0]])
    y = np.array([[20.0, 0.5, 3.0]])  # s.y = 20.1, y.y = 409.25: gamma = 0.049
    g0 = np.array([[1.0, -2.0, 0.5]])
    z0 = np.zeros((1, n))
    mem = (np.zeros((1, m, n)), np.zeros((1, m, n)), np.zeros((1, m)), np.zeros(1, dtype=np.int64))
    s_mem, y_mem, rho, head, d, _ = _dir(z0 + s, g0 + y, z0, g0, *mem)
    g1 = g0 + y
    sy = float((s @ y.T)[0, 0])
    gamma = sy / float((y @ y.T)[0, 0])
    v = np.eye(n) - np.outer(y, s) / sy
    h1 = v.T @ (gamma * np.eye(n)) @ v + np.outer(s, s) / sy
    np.testing.assert_allclose(d[0], -h1 @ g1[0], rtol=1e-12)
    assert head[0] == 1 and rho[0, 0] == pytest.approx(1.0 / sy)
    assert abs(gamma - 1.0) > 0.5
    # a pair with s.y < 0: memory kept, same H1
    s2 = np.array([[0.5, 0.0, 0.0]])
    y2 = np.array([[-1.0, 0.0, 0.0]])
    z1 = z0 + s
    s_mem2, _, rho2, head2, d2, gd2 = _dir(z1 + s2, g1 + y2, z1, g1, s_mem, y_mem, rho, head)
    assert head2[0] == 1
    np.testing.assert_array_equal(rho2, rho)
    np.testing.assert_array_equal(s_mem2, s_mem)
    np.testing.assert_allclose(d2[0], -h1 @ (g1 + y2)[0], rtol=1e-12)
    assert gd2[0] < 0.0


def test_lbfgs_direction_non_descent_falls_back_to_steepest_descent():
    """A memory whose recursion gives an ascent direction (an inconsistent pair: rho > 0 while
    s.y < 0, so gamma < 0; d_qn = (3, 0) for g = (1, 0)) falls back to the steepest descent of
    length init_step and clears the memory."""
    n, m = 2, 3
    s_mem = np.zeros((1, m, n))
    y_mem = np.zeros((1, m, n))
    rho = np.zeros((1, m))
    s_mem[0, 0] = [1.0, 0.0]
    y_mem[0, 0] = [-1.0, 0.0]
    rho[0, 0] = 1.0
    head = np.array([1])
    g = np.array([[1.0, 0.0]])
    z = np.zeros((1, n))
    s_out, y_out, rho_out, head_out, d, gd = _dir(z, g, z, g, s_mem, y_mem, rho, head, has_prev=False)
    np.testing.assert_array_equal(d, [[-1.0, 0.0]])
    assert gd[0] == -1.0
    assert head_out[0] == 0 and np.all(rho_out == 0.0) and np.all(s_out == 0.0) and np.all(y_out == 0.0)


# ------------------------------------------------------------------------------ stop rules
def _bad_loss(z):
    """A finite loss far above every value of the gradient callable: every trial is rejected."""
    return np.full(np.asarray(z).shape[0], 1e300)


def _bad_res(z):
    return np.full((np.asarray(z).shape[0], 4), 1e100)


def test_lbfgs_linesearch_stop():
    z0 = _STARTS[:2, :3] * 0.5
    res = lbfgs(_np_rosen3, _bad_loss, z0, LbfgsConfig(max_backtracks=5))
    np.testing.assert_array_equal(res.extra["status"], "linesearch")
    np.testing.assert_array_equal(res.extra["n_forward_rows"], 5)
    np.testing.assert_array_equal(res.extra["n_grad_rows"], 1)
    np.testing.assert_array_equal(res.extra["accepted"], 0)
    np.testing.assert_array_equal(res.z_best, z0)


def test_lbfgs_max_iter_stop():
    res = _run("lbfgs", _STARTS[:, :3] * 0.5, LbfgsConfig(max_iter=2))
    np.testing.assert_array_equal(res.extra["status"], "max_iter")
    np.testing.assert_array_equal(res.extra["accepted"], 2)
    # a gradient at the start and after the first accepted step, none after the last
    np.testing.assert_array_equal(res.extra["n_grad_rows"], 2)


def test_lm_rejects_stop():
    z0 = _STARTS[:2, :3] * 0.5
    res = lm(_np_rosen3_rj, _bad_res, z0, LmConfig(max_rejects=4))
    np.testing.assert_array_equal(res.extra["status"], "rejects")
    np.testing.assert_array_equal(res.extra["n_forward_rows"], 4)
    np.testing.assert_array_equal(res.extra["n_grad_rows"], 1)
    np.testing.assert_array_equal(res.z_best, z0)


def test_lm_lambda_stop():
    z0 = _STARTS[:2, :3] * 0.5
    res = lm(_np_rosen3_rj, _bad_res, z0, LmConfig(lambda_max=1e2, max_rejects=100))
    # 1e-3 * 3^k > 1e2 first at k = 11
    np.testing.assert_array_equal(res.extra["status"], "lambda")
    np.testing.assert_array_equal(res.extra["n_forward_rows"], 11)
    np.testing.assert_allclose(res.extra["lambda"], 1e-3 * 3.0**11)


def test_lm_max_iter_stop():
    res = _run("lm", _STARTS[:, :3] * 0.5, LmConfig(max_iter=3))
    np.testing.assert_array_equal(res.extra["status"], "max_iter")
    np.testing.assert_array_equal(res.extra["n_forward_rows"], 3)


def test_lm_stall_stop():
    # stall_rel = 1: every accepted step counts as a stall step (none resets the count)
    res = _run("lm", _STARTS[:, :3] * 0.5, LmConfig(stall_rel=1.0, ftol=0.0, xtol=0.0))
    np.testing.assert_array_equal(res.extra["status"], "stall")
    np.testing.assert_array_equal(res.extra["accepted"], 3)
    np.testing.assert_array_equal(res.extra["n_grad_rows"], 3)


@needs_x64
def test_lm_converged_rows_stop_without_wasted_trials():
    """With the MINPACK tolerances switched on, at a minimum a row stops after one trial (ftol),
    not after max_rejects rejected trials: a least-squares problem with a noise floor, and a
    zero-residual fit at the rounding floor (xtol). A zero gradient at the start stops before any
    trial (gradient)."""
    z0 = np.array([[0.0, 0.0, 0.0], [5.0, -5.0, 2.0]])
    tols = LmConfig(ftol=LM_MINPACK_TOL, xtol=LM_MINPACK_TOL)
    floor = lm(batched_residual_jacobian(_lin_res), batched_loss(_lin_res), z0, tols)
    ref = np.linalg.lstsq(np.asarray(_A), np.asarray(_B), rcond=None)[0]
    np.testing.assert_allclose(floor.z_best, np.tile(ref, (2, 1)), atol=1e-8)
    np.testing.assert_array_equal(floor.extra["status"], "ftol")
    assert np.all(floor.extra["n_forward_rows"] <= 6), floor.extra["n_forward_rows"]
    # on a linear problem every trial decreases the loss but the last one may be lost in rounding
    assert np.all(floor.extra["n_forward_rows"] - floor.extra["accepted"] <= 1), floor.extra["accepted"]
    # a zero-residual fit
    b_fit = np.asarray(_A) @ np.array([0.5, -1.0, 2.0])

    def fit_res(z):
        return _A @ z - b_fit

    exact = lm(batched_residual_jacobian(fit_res), batched_loss(fit_res), z0, tols)
    np.testing.assert_allclose(exact.z_best, np.tile([0.5, -1.0, 2.0], (2, 1)), atol=1e-10)
    assert set(exact.extra["status"].tolist()) <= {"ftol", "xtol"}, exact.extra["status"]
    assert np.all(exact.extra["n_forward_rows"] < LmConfig().max_rejects), exact.extra["n_forward_rows"]
    # exactly zero residual at the start (halves and small integers: exact products and sums)
    assert np.all(np.asarray(fit_res(jnp.array([0.5, -1.0, 2.0]))) == 0.0)
    at = lm(batched_residual_jacobian(fit_res), batched_loss(fit_res), np.array([[0.5, -1.0, 2.0]]), tols)
    assert at.extra["status"][0] == "gradient"
    assert at.extra["n_forward_rows"][0] == 0 and at.extra["n_grad_rows"][0] == 1


def test_lm_gradient_stop():
    res = _run("lm", _STARTS[:, :3] * 0.5, LmConfig(gtol=1e-3, ftol=0.0, xtol=0.0))
    np.testing.assert_array_equal(res.extra["status"], "gradient")
    _, jac = _np_rosen3_rj(res.z_best)
    jtr = np.einsum("rkn,rk->rn", jac, _np_rosen3_res(res.z_best))
    assert np.all(np.max(np.abs(jtr), axis=1) <= 1e-3)


@needs_x64
def test_lm_marquardt_scaling():
    """One trial on a badly scaled linear problem with lambda0 = 1: the step is
    -(J'J + lambda diag(J'J))^-1 J'r, far from the step with identity damping."""
    a = np.array([[1.0, 0.0], [0.0, 1000.0], [1.0, 1.0]])
    b = np.array([1.0, 2.0, 3.0])

    def res(z):
        return jnp.asarray(a) @ z - jnp.asarray(b)

    z0 = np.array([[0.3, -0.2]])
    out = lm(batched_residual_jacobian(res), batched_loss(res), z0, LmConfig(lambda0=1.0, max_iter=1))
    r0 = a @ z0[0] - b
    jtj = a.T @ a
    step = -np.linalg.solve(jtj + np.diag(np.diag(jtj)), a.T @ r0)
    ident = -np.linalg.solve(jtj + np.eye(2), a.T @ r0)
    np.testing.assert_allclose(out.z_best[0], z0[0] + step, rtol=1e-12)
    assert np.max(np.abs(step - ident)) > 1e-2
    assert out.extra["status"][0] == "max_iter" and out.extra["accepted"][0] == 1
    assert out.extra["n_forward_rows"][0] == 1 and out.extra["n_grad_rows"][0] == 1


def test_lm_step_and_predicted_decrease():
    """The damped step solves (J'J + lam diag(J'J)) p = -J'r, and its predicted decrease equals the
    Gauss-Newton model's |r|^2 - |r + J p|^2 (exact for a linear residual: the actual decrease)."""
    from agrijax.calib.optim import _lm_step

    rng = np.random.default_rng(5)
    jac = rng.standard_normal((6, 3)) * np.array([1.0, 30.0, 0.1])
    r = rng.standard_normal(6)
    a_m, g = jac.T @ jac, jac.T @ r
    for lam in (1e-3, 1.0, 50.0):
        step, pred = _lm_step(a_m, g, lam)
        np.testing.assert_allclose((a_m + lam * np.diag(np.diag(a_m))) @ step, -g, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(pred, r @ r - np.sum((r + jac @ step) ** 2), rtol=1e-10)


# ------------------------------------------------------------------------------ pinned step and stop rules
def _scripted(values, width=None):
    """A batch callable for one row that returns ``values[k]`` at its ``k``-th call, whatever the
    point (a scalar loss, or a residual vector when ``width`` is given)."""

    def fn(z):
        v = np.asarray(values[fn.calls], dtype=float)
        fn.calls += 1
        r = np.asarray(z).shape[0]
        return np.full(r, float(v)) if width is None else np.broadcast_to(v, (r, width)).copy()

    fn.calls = 0
    return fn


def _const_vg(f, g):
    """value_and_grad of one coordinate: always the value ``f`` and the gradient ``g`` (so no pair
    is ever stored: s.y = 0, and every direction is the steepest descent of length init_step)."""

    def vg(z):
        r = np.asarray(z).shape[0]
        return np.full(r, f), np.full((r, 1), g)

    return vg


def test_lbfgs_armijo_slope_term():
    """The first trial is z - init_step g/|g| with g.d = -1 (f = 0, g = 1): a decrease of half
    c1 alpha |g.d| is rejected at every step length, twice of it is accepted at once."""
    c1 = LbfgsConfig().c1
    cfg = LbfgsConfig(max_backtracks=3, max_iter=1)
    z0 = np.zeros((1, 1))
    half = lbfgs(_const_vg(0.0, 1.0), lambda z: -0.5 * c1 * np.abs(np.asarray(z)[:, 0]), z0, cfg)
    assert half.extra["status"][0] == "linesearch" and half.extra["accepted"][0] == 0
    assert half.extra["n_forward_rows"][0] == 3
    twice = lbfgs(_const_vg(0.0, 1.0), lambda z: -2.0 * c1 * np.abs(np.asarray(z)[:, 0]), z0, cfg)
    assert twice.extra["status"][0] == "max_iter" and twice.extra["accepted"][0] == 1
    assert twice.extra["n_forward_rows"][0] == 1


def test_lbfgs_backtracks_counted_per_line_search():
    """max_backtracks counts rejected trials in a row within one line search: a reject before
    every accepted step (reject, accept, reject, accept, ...) never stops the row with
    max_backtracks = 2."""
    loss = _scripted([1.0, -1.0] * 3)
    res = lbfgs(_const_vg(0.0, 1.0), loss, np.zeros((1, 1)), LbfgsConfig(max_backtracks=2, max_iter=3))
    assert res.extra["status"][0] == "max_iter" and res.extra["accepted"][0] == 3
    assert res.extra["n_forward_rows"][0] == 6 and res.extra["n_grad_rows"][0] == 3


def test_lbfgs_ftol_stop():
    """An accepted step with a relative decrease below ftol stops the row (a tiny gradient, so
    the Armijo test accepts a tiny decrease); a larger decrease does not; with ftol = 0 only an
    accepted step without any decrease stops it."""
    z0 = np.zeros((1, 1))
    cfg = LbfgsConfig(max_iter=1)
    tiny = lbfgs(_const_vg(1.0, 1e-12), lambda z: np.full(np.asarray(z).shape[0], 1.0 - 1e-11), z0, cfg)
    assert tiny.extra["status"][0] == "ftol" and tiny.extra["accepted"][0] == 1
    big = lbfgs(_const_vg(1.0, 1e-12), lambda z: np.full(np.asarray(z).shape[0], 1.0 - 1e-6), z0, cfg)
    assert big.extra["status"][0] == "max_iter"
    off = LbfgsConfig(max_iter=1, ftol=0.0)
    same = lbfgs(_const_vg(1.0, 1e-20), lambda z: np.ones(np.asarray(z).shape[0]), z0, off)
    assert same.extra["status"][0] == "ftol" and same.extra["accepted"][0] == 1
    lower = lbfgs(_const_vg(1.0, 1e-12), lambda z: np.full(np.asarray(z).shape[0], 1.0 - 1e-11), z0, off)
    assert lower.extra["status"][0] == "max_iter"


@needs_x64
def test_lbfgs_inconsistent_least_squares_stops_at_minimum():
    """At the minimum of an inconsistent linear least-squares problem (nonzero loss floor, target
    never reached) every row stops on ftol long before max_iter, at the least-squares minimum (the
    loss to well within the relative ftol of 2.2e-9; the point to its square root)."""
    a = np.asarray(_A)
    b = np.asarray(_B)

    def vg(z):
        r = np.asarray(z) @ a.T - b
        return np.sum(r**2, axis=1), 2.0 * r @ a

    def loss(z):
        return vg(z)[0]

    z0 = np.array([[0.0, 0.0, 0.0], [5.0, -5.0, 2.0], [-3.0, 1.0, 4.0], [1.0, 1.0, 1.0]])
    res = lbfgs(vg, loss, z0, LbfgsConfig(max_iter=100))
    ref = np.linalg.lstsq(a, b, rcond=None)[0]
    np.testing.assert_allclose(res.loss_best, np.full(4, loss(ref[None])[0]), rtol=1e-9)
    np.testing.assert_allclose(res.z_best, np.tile(ref, (4, 1)), atol=1e-4)
    np.testing.assert_array_equal(res.extra["status"], "ftol")
    assert np.all(res.extra["accepted"] < 40), res.extra["accepted"]


def test_lm_rejects_counted_in_a_row():
    """max_rejects counts rejected trials in a row: a reject before every accepted step never
    stops the row with max_rejects = 2 (r = 1, J = 1 at every accepted point; trial residual 2:
    rejected, 0.5: accepted)."""

    def rj(z):
        k = np.asarray(z).shape[0]
        return np.ones((k, 1)), np.ones((k, 1, 1))

    res = lm(rj, _scripted([2.0, 0.5] * 3, width=1), np.zeros((1, 1)), LmConfig(max_rejects=2, max_iter=6))
    assert res.extra["status"][0] == "max_iter" and res.extra["accepted"][0] == 3
    assert res.extra["n_forward_rows"][0] == 6 and res.extra["rejects_in_a_row"][0] == 0


def test_lm_stall_counted_in_a_row():
    """stall_steps counts slow accepted steps in a row: slow (relative decrease 0.02), fast (0.75),
    slow, fast does not stop the row with stall_steps = 2; slow, slow does."""

    def rj(z):
        k = np.asarray(z).shape[0]
        return np.ones((k, 1)), np.ones((k, 1, 1))

    cfg = LmConfig(stall_rel=0.1, stall_steps=2, max_iter=4)
    mixed = lm(rj, _scripted([0.99, 0.5, 0.99, 0.5], width=1), np.zeros((1, 1)), cfg)
    assert mixed.extra["status"][0] == "max_iter" and mixed.extra["accepted"][0] == 4
    slow = lm(rj, _scripted([0.99, 0.99, 0.99, 0.99], width=1), np.zeros((1, 1)), cfg)
    assert slow.extra["status"][0] == "stall" and slow.extra["accepted"][0] == 2


def _lm_one_trial(delta, trial_res):
    """One LM trial (ftol = LM_MINPACK_TOL, max_iter = 1) from r = (1, delta), J = (0, 1)': the
    step -delta / (1 + lambda0) predicts a relative decrease of about delta^2, and the trial
    residual is ``trial_res``. Returns the status and the actual and predicted relative
    decreases."""
    lam = LmConfig().lambda0

    def rj(z):
        k = np.asarray(z).shape[0]
        jac = np.zeros((k, 2, 1))
        jac[:, 1, 0] = 1.0
        return np.tile([1.0, delta], (k, 1)), jac

    res = lm(rj, _scripted([trial_res], width=2), np.zeros((1, 1)), LmConfig(ftol=LM_MINPACK_TOL, max_iter=1))
    loss_old = 1.0 + delta**2
    p = -delta / (1.0 + lam)
    prered = (p**2 + 2.0 * lam * p**2) / loss_old
    actred = (loss_old - float(np.sum(np.asarray(trial_res) ** 2))) / loss_old
    return res.extra["status"][0], actred, prered


def test_lm_ftol_each_clause():
    """The MINPACK test |actred| <= ftol, prered <= ftol, actred <= 2 prered: met as a whole, it
    stops the row; failing any one clause alone, it does not."""
    ftol = LM_MINPACK_TOL
    st, act, pre = _lm_one_trial(1e-5, [1.0, 0.0])  # all three hold
    assert abs(act) <= ftol and pre <= ftol and act <= 2.0 * pre
    assert st == "ftol"
    st, act, pre = _lm_one_trial(1e-5, [np.sqrt(1.0 + 1e-10 + 1e-6), 0.0])  # |actred| fails alone
    assert abs(act) > ftol and pre <= ftol and act <= 2.0 * pre
    assert st == "max_iter"
    st, act, pre = _lm_one_trial(1e-3, [1.0, 1e-3])  # prered fails alone (no change in the loss)
    assert abs(act) <= ftol and pre > ftol and act <= 2.0 * pre
    assert st == "max_iter"
    st, act, pre = _lm_one_trial(1e-5, [np.sqrt((1.0 + 1e-10) * (1.0 - 1e-9)), 0.0])  # actred > 2 prered
    assert abs(act) <= ftol and pre <= ftol and act > 2.0 * pre
    assert st == "max_iter"


def test_lm_xtol_after_rejections_told_apart():
    """With the MINPACK tolerances on, a row whose trials are all rejected ends xtol once the
    growing damping has shrunk the step below xtol; ``rejects_in_a_row`` (equal to its trials, no
    step accepted) tells it from a converged row."""
    z0 = _STARTS[:2, :3] * 0.5
    cfg = LmConfig(
        ftol=LM_MINPACK_TOL, xtol=LM_MINPACK_TOL, max_rejects=1000, lambda_max=1e300, max_iter=1000
    )
    res = lm(_np_rosen3_rj, _bad_res, z0, cfg)
    np.testing.assert_array_equal(res.extra["status"], "xtol")
    np.testing.assert_array_equal(res.extra["accepted"], 0)
    np.testing.assert_array_equal(res.extra["rejects_in_a_row"], res.extra["n_forward_rows"])
    assert np.all(res.extra["n_forward_rows"] > 1)


@pytest.mark.parametrize("kind", ["lbfgs", "lm"])
def test_target_met_with_equality(kind):
    """A target equal to a loss the free run reaches stops the row at that point (<=, not <); a
    target equal to the starting loss stops it before any trial."""
    z0 = _STARTS[1:2, :3] * 0.5
    free = _run(kind, z0)
    hist = free.loss_history[:, 0]
    k = int(np.nonzero(hist < hist[0])[0][1])  # the second improvement of the free run
    res = _run(kind, z0, target=hist[k])
    assert res.extra["status"][0] == "target"
    assert res.loss_best[0] == hist[k]
    np.testing.assert_array_equal(res.loss_history[:, 0], hist[: k + 1])
    start = float(_np_rosen3_loss(z0)[0])
    at = _run(kind, z0, target=start)
    assert at.extra["status"][0] == "target" and at.extra["n_forward_rows"][0] == 0


def test_lbfgs_direction_ring_buffer_wraps():
    """More pairs than memory: after 7 pairs with m = 3 the ring buffer has wrapped twice, and the
    direction equals the explicit BFGS inverse update on the last 3 pairs, oldest to newest,
    from H0 = gamma I with gamma = s.y / y.y of the newest pair."""
    from agrijax.calib.optim import _lbfgs_direction

    n, m, k_pairs = 4, 3, 7
    hmat = _spd(n, 30.0, 11)
    rng = np.random.default_rng(2)
    zs = np.cumsum(rng.standard_normal((k_pairs + 1, n)), axis=0)
    gs = zs @ hmat  # gradient of 0.5 z'Hz: s.y = s'Hs > 0 for every pair
    s_mem = np.zeros((1, m, n))
    y_mem = np.zeros((1, m, n))
    rho = np.zeros((1, m))
    head = np.zeros(1, dtype=np.int64)
    d = np.zeros((1, n))
    for k in range(k_pairs + 1):
        s_mem, y_mem, rho, head, d, _ = _lbfgs_direction(
            np.array([True]),
            np.array([k > 0]),
            zs[k : k + 1],
            gs[k : k + 1],
            zs[max(k - 1, 0) : max(k - 1, 0) + 1],
            gs[max(k - 1, 0) : max(k - 1, 0) + 1],
            s_mem,
            y_mem,
            rho,
            head,
            1.0,
        )
    assert head[0] == k_pairs % m
    pairs = [(zs[k] - zs[k - 1], gs[k] - gs[k - 1]) for k in range(k_pairs - m + 1, k_pairs + 1)]
    s_n, y_n = pairs[-1]
    hinv = (s_n @ y_n) / (y_n @ y_n) * np.eye(n)
    for s, y in pairs:  # oldest to newest
        rho_k = 1.0 / (s @ y)
        v = np.eye(n) - rho_k * np.outer(y, s)
        hinv = v.T @ hinv @ v + rho_k * np.outer(s, s)
    np.testing.assert_allclose(d[0], -hinv @ gs[k_pairs], rtol=1e-10)


# ------------------------------------------------------------------------------ LM against the earlier damping loop
def _exp_fit():
    """A noisy exponential fit a exp(b t) (12 residuals, nonzero residual at the minimum), in
    NumPy elementwise over the rows."""
    t = np.linspace(0.0, 2.0, 12)
    noise = np.random.default_rng(4).standard_normal(12) * 0.05
    y = 2.0 * np.exp(-0.7 * t) + noise

    def res(z):
        z = np.asarray(z, dtype=float)
        return z[:, :1] * np.exp(z[:, 1:2] * t) - y

    def rj(z):
        z = np.asarray(z, dtype=float)
        e = np.exp(z[:, 1:2] * t)
        return z[:, :1] * e - y, np.stack([e, z[:, :1] * t * e], axis=2)

    return res, rj


def _reference_lm_loop(rj, res, z0, cfg):
    """An independent reimplementation of the default stop rules of ``lm`` (the plain per-row
    damping loop): one row at a time, J'J, J'r and sum(r^2) with BLAS products and np.sum; one
    Jacobian at a new accepted point, one trial per round; accept when the trial loss is lower;
    lambda / 3 (floored) after an accept, x 3 after a reject; stop on stall, rejects, damping or
    the round cap. Returns per row (status, forwards, Jacobians, accepted, accept / reject
    sequence, z, smallest relative gap |trial - loss| / loss of any decision)."""
    out = []
    for z in np.asarray(z0, dtype=float):
        lam, n_fwd, n_jac, n_acc, stall, rejects = cfg.lambda0, 0, 0, 0, 0, 0
        need_j, status, seq, gap = True, "max_iter", [], np.inf
        jj, rr = np.zeros((0, z.size)), np.zeros(0)
        loss = np.inf
        for _ in range(cfg.max_iter):
            if need_j:
                r0, j0 = rj(z[None])
                rr, jj = r0[0], j0[0]
                loss = float(np.sum(rr**2))
                n_jac += 1
                need_j = False
            a_m = jj.T @ jj
            g = jj.T @ rr
            dmat = np.diag(np.maximum(np.diag(a_m), 1e-12))
            step = -np.linalg.solve(a_m + lam * dmat, g)
            lt = float(np.sum(res((z + step)[None])[0] ** 2))
            n_fwd += 1
            gap = min(gap, abs(lt - loss) / loss)
            if np.isfinite(lt) and lt < loss:
                rel = (loss - lt) / max(loss, 1e-300)
                z, loss = z + step, lt
                lam = max(lam / cfg.lambda_factor, cfg.lambda_min)
                need_j, n_acc, rejects = True, n_acc + 1, 0
                stall = stall + 1 if rel < cfg.stall_rel else 0
                seq.append(True)
            else:
                lam *= cfg.lambda_factor
                rejects += 1
                seq.append(False)
            if stall >= cfg.stall_steps:
                status = "stall"
            elif rejects >= cfg.max_rejects:
                status = "rejects"
            elif lam > cfg.lambda_max:
                status = "lambda"
            if status != "max_iter":
                break
        out.append((status, n_fwd, n_jac, n_acc, seq, z, gap))
    return out


#: starts of the exponential fit: in the basin and far off (from a growing exponential with a small
#: amplitude, [0.2, 0.5], the first Gauss-Newton steps overshoot and are rejected, so the damping
#: grows before it shrinks)
_EXP_STARTS = np.array([[1.0, -0.2], [3.0, -1.5], [0.5, 1.0], [5.0, 0.8], [0.2, 0.5], [1.0, 1.5]])


@needs_x64
def test_lm_matches_reference_damping_loop_before_the_noise_floor():
    """With the default settings and a stall test that ends the rows well above the noise floor
    of the loss (stall_rel = 1e-2, stall_steps = 1), the accept / reject decisions, the status and
    the call counts of every row equal those of the reference loop, and the iterates agree within
    rounding (the sums are formed in another order). The test checks that the run has rejected
    trials and reaches the stall stop on every row, and that no decision sat on a knife edge."""
    res, rj = _exp_fit()
    cfg = LmConfig(stall_rel=1e-2, stall_steps=1)
    out = lm(rj, res, _EXP_STARTS, cfg)
    ref = _reference_lm_loop(rj, res, _EXP_STARTS, cfg)
    loss0 = np.sum(res(_EXP_STARTS) ** 2, axis=1)
    for i, (status, n_fwd, n_jac, n_acc, seq, z, gap) in enumerate(ref):
        # no accept / reject decision near the rounding of the loss (about 1e-16 relative)
        assert gap > 1e-6, (i, gap)
        assert out.extra["status"][i] == status == "stall"
        assert out.extra["n_forward_rows"][i] == n_fwd
        assert out.extra["n_grad_rows"][i] == n_jac
        assert out.extra["accepted"][i] == n_acc
        np.testing.assert_allclose(out.z_best[i], z, rtol=1e-10, atol=0.0)
        # one trial per round: a trial was accepted when the round's loss is below the one before
        hist = out.loss_history[:n_fwd, i]
        before = np.concatenate([[loss0[i]], hist[:-1]])
        assert (hist < before).tolist() == seq
    assert any(not a for row in ref for a in row[4])  # some trials were rejected


@needs_x64
def test_lm_reference_damping_loop_default_stall_agrees_to_the_floor():
    """With the default stall settings the rows run into the noise floor, where acceptance is
    decided by the last bits and the summation order may change a row's status and counts: there
    only the minimum is the same (loss within rounding, point within the floor's resolution)."""
    res, rj = _exp_fit()
    cfg = LmConfig()
    out = lm(rj, res, _EXP_STARTS, cfg)
    ref = _reference_lm_loop(rj, res, _EXP_STARTS, cfg)
    for i, row in enumerate(ref):
        z = row[5]
        np.testing.assert_allclose(out.z_best[i], z, rtol=1e-6)
        loss_ref = float(np.sum(res(z[None])[0] ** 2))
        np.testing.assert_allclose(out.loss_best[i], loss_ref, rtol=1e-12)
