"""Gradients of the Richards redistribution: unrolled, implicit-function-theorem, finite differences.

* The unrolled gradient (reverse mode through the fixed Newton iterations) must equal central
  finite differences of the same forward computation (float64, relative error < 1e-4) for the
  Brooks-Corey parameters lambda, h_b, K_sat, theta_r and theta_s of three horizons, at a
  coarse and a working configuration.
* At a converged state the implicit-function-theorem gradient (``grad="implicit"``: one
  tridiagonal adjoint solve per sub-step, no differentiation through the iterations) must
  equal the unrolled one, and both must equal finite differences.
* Gradients stay finite in the switching regimes (dry-limited evaporation, ponding and runoff,
  uptake cut at h_min) in float32 and float64, and through the core runtime with
  ``checkpoint=True``.

Source: Griewank & Walther (2008) ch. 15 (derivatives of fixed-point iterations); Ahuja et al.
(2000) ch. 3 for the model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from agrijax.core import Model
from agrijax.core.runtime import run
from agrijax.processes.soil_water.richards import (
    RichardsConfig,
    RichardsForcing,
    RichardsParams,
    RichardsState,
    SoilWater,
    richards_day,
    richards_redistribution,
)

from .test_richards import catpa_grid, catpa_soil

X64 = bool(jax.config.read("jax_enable_x64"))
FIELDS = ("lambda_", "hb", "ksat", "theta_r", "theta_s")
HORIZONS = (0, 2, 4)


def _forcing(n_day: int = 2) -> RichardsForcing:
    supply = np.zeros((n_day, 24))
    supply[0, 3:5] = 0.6
    evap = np.full((n_day, 24), 0.2 / 24)
    uptake = np.zeros((n_day, 37))
    uptake[:, :12] = 0.2 / 12
    return RichardsForcing(
        supply=jnp.asarray(supply), evaporation=jnp.asarray(evap), uptake=jnp.asarray(uptake)
    )


def _loss_fn(cfg: RichardsConfig, forcing: RichardsForcing, theta0: np.ndarray):
    grid = catpa_grid()

    def loss(soil):
        params = RichardsParams(soil=soil, grid=grid, config=cfg)
        w0 = SoilWater.from_theta(jnp.asarray(theta0), soil)

        def body(w, f):
            w2 = richards_day(w, params, f.supply, f.evaporation, f.uptake)
            return w2, 10.0 * (w2.flux.drainage + w2.flux.evaporation)

        w, per_day = lax.scan(body, w0, forcing)
        return w.storage(grid) + jnp.sum(per_day)

    return loss


def _fd(loss, soil, name: str, hz: int, rel_step: float = 1e-6) -> float:
    x = getattr(soil, name)
    eps = rel_step * abs(float(x[hz]))
    fp = float(loss(soil.replace(**{name: x.at[hz].add(eps)})))
    fm = float(loss(soil.replace(**{name: x.at[hz].add(-eps)})))
    return (fp - fm) / (2.0 * eps)


@pytest.mark.allow_skip(reason="central differences at rel 1e-4 need float64")
@pytest.mark.parametrize(("n_sub", "n_iter"), [(24, 3), (6, 1)])
def test_unrolled_gradient_matches_central_differences(n_sub: int, n_iter: int) -> None:
    if not X64:
        pytest.skip("needs float64")
    soil = catpa_soil()
    loss = _loss_fn(RichardsConfig(n_sub=n_sub, n_iter=n_iter), _forcing(), np.linspace(0.24, 0.30, 37))
    g = jax.jit(jax.grad(loss))(soil)
    lj = jax.jit(loss)
    worst = 0.0
    for name in FIELDS:
        for hz in HORIZONS:
            ad = float(getattr(g, name)[hz])
            fd = _fd(lj, soil, name, hz)
            assert np.isfinite(ad)
            rel = abs(ad - fd) / abs(fd)
            worst = max(worst, rel)
            assert rel < 1e-4, (name, hz, ad, fd)
    # measured: worst 3.2e-5 (24 x 3, d/d ksat of horizon 1, a derivative of 4e-5), typically 1e-8
    assert worst < 1e-4


@pytest.mark.allow_skip(reason="the 1e-8 agreement is a float64 claim")
def test_implicit_function_theorem_gradient_at_convergence() -> None:
    if not X64:
        pytest.skip("needs float64")
    soil = catpa_soil()
    forcing, theta0 = _forcing(), np.linspace(0.24, 0.30, 37)
    unrolled = _loss_fn(RichardsConfig(n_sub=24, n_iter=12), forcing, theta0)
    implicit = _loss_fn(RichardsConfig(n_sub=24, n_iter=12, grad="implicit"), forcing, theta0)
    assert float(unrolled(soil)) == float(implicit(soil))  # same forward pass
    gu = jax.jit(jax.grad(unrolled))(soil)
    gi = jax.jit(jax.grad(implicit))(soil)
    lj = jax.jit(unrolled)
    for name in FIELDS:
        a = np.asarray(getattr(gu, name))
        b = np.asarray(getattr(gi, name))
        # measured 1e-12 relative at 8 and 12 iterations (3e-6 at 4 iterations, before convergence)
        np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-12 * np.abs(a).max())
        for hz in HORIZONS:
            fd = _fd(lj, soil, name, hz)
            assert abs(float(b[hz]) - fd) < 1e-4 * abs(fd), (name, hz)
    # the other leaves (eps, hb_k, ...) are differentiated as well
    np.testing.assert_allclose(np.asarray(gi.eps), np.asarray(gu.eps), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(gi.hb_k), np.asarray(gu.hb_k), rtol=1e-8, atol=1e-14)


REGIMES = {
    # name: (initial head [cm], supply in hour 0 [cm/h], evaporation demand [cm/d], uptake of the top 6 cells [cm/d])
    "dry_evaporation": (-8000.0, 0.0, 1.0, 0.0),
    "ponding": (-16.0, 40.0, 0.0, 0.0),
    "uptake_cut": (-14000.0, 0.0, 0.0, 0.5),
}


def _regime_loss(cfg: RichardsConfig):
    grid = catpa_grid()

    def loss(soil, h0, rain, evap_day, upt):
        dtype = soil.hb.dtype
        params = RichardsParams(soil=soil, grid=grid, pond_max=1.0, config=cfg)
        w0 = SoilWater.from_head(jnp.full(37, 1.0, dtype) * h0, soil)
        supply = jnp.zeros(24, dtype).at[0].set(rain)
        evap = jnp.full(24, 1.0 / 24, dtype) * evap_day
        uptake = jnp.zeros(37, dtype).at[:6].set(upt)
        w = richards_day(w0, params, supply, evap, uptake)
        f = w.flux
        return w.storage(grid) + f.evaporation + f.runoff + f.uptake + f.drainage + w.pond, f

    return loss


@pytest.mark.parametrize("grad_mode", ["unrolled", "implicit"])
def test_gradients_finite_in_switching_regimes(grad_mode: str) -> None:
    """Every regime is reached (checked on the fluxes) and every gradient leaf is finite."""
    soil = catpa_soil()
    loss = _regime_loss(RichardsConfig(n_sub=24, n_iter=4, grad=grad_mode))
    vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True))
    for regime, args in REGIMES.items():
        (_, f), grads = vg(soil, *(jnp.asarray(a) for a in args))
        if regime == "dry_evaporation":
            assert float(f.evaporation_deficit) > 0.1
        elif regime == "ponding":
            assert float(f.runoff) > 1.0
        else:
            assert float(f.uptake_cut) > 0.0
        leaves = jax.tree_util.tree_leaves(grads)
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves), (regime, grad_mode)
        assert any(float(jnp.max(jnp.abs(x))) > 0 for x in jax.tree_util.tree_leaves(grads[0]))


def test_gradient_through_runtime_with_checkpoint() -> None:
    """``core.runtime.run(checkpoint=True)`` over the process: finite and equal to central differences."""
    soil = catpa_soil()
    grid = catpa_grid()
    forcing = _forcing(3)
    model = Model(RichardsState, [richards_redistribution], outputs=("soil_water.flux.drainage",))

    def via_runtime(s):
        params = RichardsParams(soil=s, grid=grid, config=RichardsConfig(n_sub=12, n_iter=3))
        st = RichardsState(soil_water=SoilWater.from_theta(jnp.full(37, 0.26), s))
        final, outs = run(model, params, forcing, st, checkpoint=True, return_final=True)
        return final.soil_water.storage(grid) + jnp.sum(outs["soil_water.flux.drainage"])

    g = jax.jit(jax.grad(via_runtime))(soil)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))
    if X64:
        lj = jax.jit(via_runtime)
        for name in ("lambda_", "ksat"):
            fd = _fd(lj, soil, name, 4)
            assert abs(float(getattr(g, name)[4]) - fd) < 1e-4 * abs(fd), name
