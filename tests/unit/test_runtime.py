"""Runtime: scan, vmap+jit, chunking, checkpointing and gradients on the toy model."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agri_jax.core import Model, run, run_batch, run_batch_chunked
from agri_jax.core.runtime import run_and_grad, run_sites, stack_days

from .toy import ToyForcing, ToyParams, ToyState, grow, infiltrate, toy_inputs, toy_model


def _reference(params: ToyParams, forcing: ToyForcing, state0: ToyState) -> tuple[np.ndarray, np.ndarray]:
    """Plain NumPy loop with the same arithmetic as the toy processes."""
    w = float(state0.water)
    b = float(state0.biomass)
    k, rue = float(params.k), float(params.rue)
    ws, bs = [], []
    for rain, srad in zip(np.asarray(forcing.rain), np.asarray(forcing.srad)):
        w = w + float(rain)
        uptake = k * w
        gain = rue * float(srad) * uptake if w > 0 else 0.0
        w, b = w - uptake, b + gain
        ws.append(w)
        bs.append(b)
    return np.array(ws), np.array(bs)


def test_model_introspection() -> None:
    m = toy_model()
    assert m.names == ("infiltrate", "grow")
    assert ("infiltrate", "grow", "water") in m.dataflow()
    assert ("infiltrate", "water") in m.stale_reads()
    assert ("grow", "biomass") in m.stale_reads()
    assert ("grow", "water") not in m.stale_reads()
    with pytest.raises(ValueError, match="duplicate"):
        Model(ToyState, [infiltrate, infiltrate])
    m2 = m.replace("grow", grow)
    assert m2.names == m.names
    with pytest.raises(KeyError):
        m.replace("missing", grow)


def test_compile_day_step_outputs_paths() -> None:
    params, forcing, state0 = toy_inputs(3)
    day = toy_model().compile()
    f0 = jax.tree_util.tree_map(lambda x: x[0], forcing)
    s1, out = day(state0, params, f0)
    assert set(out) == {"water", "biomass"}
    assert float(out["water"]) == pytest.approx(float(s1.water))
    full = Model(ToyState, [infiltrate, grow]).compile()
    _, out_full = full(state0, params, f0)
    assert isinstance(out_full, ToyState)


def test_run_matches_reference(tol: float) -> None:
    params, forcing, state0 = toy_inputs(12)
    out = run(toy_model(), params, forcing, state0)
    w_ref, b_ref = _reference(params, forcing, state0)
    assert out["water"].shape == (12,)
    np.testing.assert_allclose(np.asarray(out["water"]), w_ref, rtol=tol)
    np.testing.assert_allclose(np.asarray(out["biomass"]), b_ref, rtol=tol)
    final, _ = run(toy_model(), params, forcing, state0, return_final=True)
    assert float(final.water) == pytest.approx(w_ref[-1], rel=tol)


def test_run_checkpoint_matches_plain(tol: float) -> None:
    params, forcing, state0 = toy_inputs(8)
    a = run(toy_model(), params, forcing, state0)
    b = run(toy_model(), params, forcing, state0, checkpoint=True)
    np.testing.assert_allclose(np.asarray(a["biomass"]), np.asarray(b["biomass"]), rtol=tol)


def test_run_batch_over_params(tol: float) -> None:
    _, forcing, state0 = toy_inputs(6)
    n = 5
    batch = ToyParams(k=jnp.linspace(0.05, 0.3, n), rue=jnp.full((n,), 2.0))
    out = run_batch(toy_model(), batch, forcing, state0)
    assert out["water"].shape == (n, 6)
    for i in range(n):
        single = jax.tree_util.tree_map(lambda x, i=i: x[i], batch)
        w_ref, _ = _reference(single, forcing, state0)
        np.testing.assert_allclose(np.asarray(out["water"][i]), w_ref, rtol=tol)


def test_run_batch_chunked_matches_run_batch(tol: float) -> None:
    _, forcing, state0 = toy_inputs(5)
    n = 7  # not a multiple of chunk -> exercises the partial last chunk
    batch = ToyParams(k=jnp.linspace(0.05, 0.3, n), rue=jnp.linspace(1.0, 3.0, n))
    full = run_batch(toy_model(), batch, forcing, state0)
    chunked = run_batch_chunked(toy_model(), batch, forcing, state0, chunk=3)
    np.testing.assert_allclose(np.asarray(chunked["biomass"]), np.asarray(full["biomass"]), rtol=tol)
    with pytest.raises(ValueError):
        run_batch_chunked(toy_model(), batch, forcing, state0, chunk=0)


def test_run_sites_batches_everything() -> None:
    params, forcing, state0 = toy_inputs(4)
    n = 3
    tile = lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x)))  # noqa: E731
    out = run_sites(
        toy_model(),
        jax.tree_util.tree_map(tile, params),
        jax.tree_util.tree_map(tile, forcing),
        jax.tree_util.tree_map(tile, state0),
    )
    assert out["water"].shape == (n, 4)
    np.testing.assert_array_equal(np.asarray(out["water"][0]), np.asarray(out["water"][2]))


@pytest.mark.allow_skip(reason="the finite-difference half needs float64; skipped in the float32 CI pass")
def test_grad_finite_and_matches_finite_difference() -> None:
    params, forcing, state0 = toy_inputs(10)
    model = toy_model()

    def loss(outs):
        return jnp.sum(outs["biomass"])

    value, g = run_and_grad(model, loss, params, forcing, state0)
    assert jnp.isfinite(value)
    assert jnp.isfinite(g.k) and jnp.isfinite(g.rue)
    assert float(g.rue) > 0.0

    if not jax.config.jax_enable_x64:
        pytest.skip("finite-difference check needs float64")
    h = 1e-6
    for name in ("k", "rue"):
        base = getattr(params, name)
        plus = loss(run(model, params.replace(**{name: base + h}), forcing, state0))
        minus = loss(run(model, params.replace(**{name: base - h}), forcing, state0))
        fd = (plus - minus) / (2 * h)
        assert float(getattr(g, name)) == pytest.approx(float(fd), rel=1e-5)

    # gradient also flows through vmap + jit
    def batched_loss(p):
        return jnp.sum(run_batch(model, p, forcing, state0)["biomass"])

    batch = ToyParams(k=jnp.array([0.1, 0.2]), rue=jnp.array([2.0, 2.5]))
    gb = jax.grad(batched_loss)(batch)
    assert gb.k.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(gb.k)))


def test_stack_days_roundtrip() -> None:
    _, forcing, _ = toy_inputs(3)
    days = [jax.tree_util.tree_map(lambda x, i=i: x[i], forcing) for i in range(3)]
    stacked = stack_days(*days)
    np.testing.assert_array_equal(np.asarray(stacked.rain), np.asarray(forcing.rain))
    assert stacked.n_days == 3


# ---------------------------------------------------------------------------- compile reuse


def test_run_batch_reuses_one_compilation_across_calls(tol: float) -> None:
    """Independent references: JAX's own compile counter (``jax.monitoring``) and the NumPy loop."""
    import gc
    import weakref

    from jax import monitoring

    event = "/jax/core/compile/backend_compile_duration"
    n = [0]

    def listener(ev: str, duration_secs: float, **kw: object) -> None:
        if ev == event:
            n[0] += 1

    model = toy_model()
    params, forcing, state0 = toy_inputs(20)
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x * 1.1, x * 0.9]), params)
    batch2 = jax.tree_util.tree_map(lambda x: x * 1.01, batch)
    monitoring.register_event_duration_secs_listener(listener)
    try:
        jax.block_until_ready(run_batch(model, batch, forcing, state0))
        after_first = n[0]
        out2 = jax.block_until_ready(run_batch(model, batch2, forcing, state0))
        after_second = n[0]
    finally:
        monitoring.unregister_event_duration_listener(listener)
    assert after_first >= 1
    assert after_second == after_first, (
        f"run_batch recompiled {after_second - after_first}x for the same shapes"
    )
    for i in range(3):  # the reused runner still computes the new parameters
        p_i = jax.tree_util.tree_map(lambda x, i=i: x[i], batch2)
        w_ref, b_ref = _reference(p_i, forcing, state0)
        np.testing.assert_allclose(np.asarray(out2["water"])[i], w_ref, rtol=tol, atol=tol)
        np.testing.assert_allclose(np.asarray(out2["biomass"])[i], b_ref, rtol=tol, atol=tol)
    # the runners live on the model, so they do not keep it alive
    ref_model = weakref.ref(model)
    del model
    gc.collect()
    assert ref_model() is None
