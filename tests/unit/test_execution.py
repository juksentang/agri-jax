"""Execution settings (``agrijax.core.execution``): the backend-dependent ``depth_unroll``.

* Resolution: context over environment over the per-backend default; the target platform from the
  context, ``jax.default_device``, then the default backend; ``cuda``/``rocm`` are ``gpu``.
* ``depth_scan`` follows the setting (the lowered program has its while loop or not) and keeps an
  explicit ``unroll=``.
* The setting changes the layout, not the numbers: the tipping bucket's layer recurrences (``infil``,
  ``satflo``, ``up_flow``, vmapped over profiles as in a batch run) give bit-identical results
  unrolled and as loops on the backend the tests run on.
* The runtime runs its entry points on the platform of the inputs and keys its cached batch
  runners on the resolved settings.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import execution as E
from agrijax.core.depth_scan import depth_scan
from agrijax.core.runtime import _RUNNERS_ATTR, run_batch
from agrijax.processes.soil_water.bucket import kernels as K

from .test_bucket import profile
from .toy import toy_inputs, toy_model


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(E.DEPTH_UNROLL_ENV, raising=False)


def test_backend_defaults() -> None:
    assert E.resolve_depth_unroll("cpu") is False
    for p in ("gpu", "cuda", "rocm", "CUDA"):
        assert E.resolve_depth_unroll(p) is True
    assert E.resolve_depth_unroll("tpu") is False  # not measured: the loop, as before


def test_context_over_environment_over_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(E.DEPTH_UNROLL_ENV, "on")
    assert E.resolve_depth_unroll("cpu") is True
    monkeypatch.setenv(E.DEPTH_UNROLL_ENV, "auto")
    assert E.resolve_depth_unroll("gpu") is True and E.resolve_depth_unroll("cpu") is False
    monkeypatch.setenv(E.DEPTH_UNROLL_ENV, "0")
    assert E.resolve_depth_unroll("gpu") is False
    with E.execution(depth_unroll=True):
        assert E.resolve_depth_unroll("cpu") is True
        with E.execution(platform="gpu"):  # an inner context inherits the fields it leaves None
            assert E.current_execution() == E.Execution(platform="gpu", depth_unroll=True)
    monkeypatch.setenv(E.DEPTH_UNROLL_ENV, "maybe")
    with pytest.raises(ValueError, match=E.DEPTH_UNROLL_ENV):
        E.resolve_depth_unroll("cpu")


def test_target_platform() -> None:
    assert E.target_platform() == E._normalise(jax.default_backend())
    with jax.default_device(jax.devices("cpu")[0]):
        assert E.target_platform() == "cpu"
        with E.execution(platform="cuda"):
            assert E.target_platform() == "gpu"
            assert E.current_execution().depth_unroll is True
    with jax.default_device("cpu"):
        assert E.target_platform() == "cpu"


def test_platform_of_counts_only_committed_arrays() -> None:
    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.ones(3), cpu)
    assert x.committed
    assert E.platform_of({"a": np.ones(2), "b": x}) == "cpu"
    assert E.platform_of(np.ones(2), 1.0) is None
    u = jnp.ones(3)  # uncommitted: JAX runs it on the default device, wherever it lives
    assert not u.committed
    assert E.platform_of(u) is None
    assert E.platform_of(u, x) == "cpu"
    seen = []
    jax.jit(lambda t: seen.append(E.platform_of(t)) or t)(x)
    assert seen == [None]  # tracers carry no device
    with E.execution(platform="gpu"), E.on_platform_of(np.ones(2), u) as ex:
        assert ex.platform == "gpu"  # no committed array: the enclosing platform is kept


def test_uncommitted_inputs_follow_the_default_device() -> None:
    """Uncommitted inputs execute on ``jax.default_device``: the runtime must trace for that
    platform, not for the device the arrays were created on."""
    params, forcing, state0 = toy_inputs(5)
    params = jax.tree.map(lambda v: jnp.stack([v, v]), params)
    with E.execution(platform="gpu"):  # stands in for a GPU default backend
        assert E.platform_of(params, forcing, state0) is None
    model = toy_model()
    with jax.default_device(jax.devices("cpu")[0]):
        run_batch(model, params, forcing, state0)
    runners = model.__dict__[_RUNNERS_ATTR]
    got = {x for k in runners for x in k if isinstance(x, E.Execution)}
    assert got == {E.Execution("cpu", False)}


def _whiles(fn, *args) -> int:
    return jax.jit(fn).lower(*args).as_text().count("stablehlo.while")


def _recurrence(**kw):
    # a fresh function per call: jax.jit caches a trace by function, not by the execution setting
    return lambda x: depth_scan(lambda c, a: (c + a * jnp.sin(c), c), jnp.zeros((), x.dtype), x, **kw)


def test_depth_scan_follows_the_setting() -> None:
    x = jnp.ones(6)
    with E.execution(depth_unroll=False):
        assert _whiles(_recurrence(), x) == 1
        assert _whiles(_recurrence(unroll=True), x) == 0  # explicit value kept
    with E.execution(depth_unroll=True):
        assert _whiles(_recurrence(), x) == 0
        assert _whiles(_recurrence(unroll=1), x) == 1
    with E.execution(depth_unroll=True):
        on = jax.jit(_recurrence())(x)
    with E.execution(depth_unroll=False):
        off = jax.jit(_recurrence())(x)
    for a, b in zip(jax.tree.leaves(on), jax.tree.leaves(off), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _profiles(n_prof: int, seed: int):
    rng = np.random.default_rng(seed)
    cols: dict[str, list[np.ndarray]] = {}
    for i in range(n_prof):
        n, nlayr = 9, int(rng.integers(2, 9))
        dlayr, ds, ll, dul, sat, sw, swcn = profile(rng, n, nlayr, ("mid", "wet", "dry")[i % 3])
        dlayr[0] = rng.choice([3.0, 5.0, 10.0])
        avail = np.maximum(0.0, sw + np.where(dlayr > 0, rng.uniform(-0.02, 0.05, n), 0.0))
        row = {
            "dlayr": dlayr,
            "ds": ds,
            "ll": ll,
            "dul": dul,
            "sat": sat,
            "sw": sw,
            "swcn": swcn,
            "avail": avail,
            "swcon": np.asarray(rng.uniform(0.2, 0.8)),
            "pinf": np.asarray(rng.choice([0.00005, rng.uniform(0.2, 3.0), rng.uniform(4.0, 12.0)])),
            "actwtd": np.asarray(1000.0 if i % 3 else rng.uniform(20.0, 80.0)),
        }
        for k, v in row.items():
            cols.setdefault(k, []).append(np.asarray(v, dtype=float))
    return {k: jnp.asarray(np.stack(v)) for k, v in cols.items()}


def _bucket_layer_loops():
    def one(p):
        a = K.infil(
            p["dlayr"], p["ds"], p["dul"], p["sat"], p["sw"], p["swcn"], p["swcon"], p["pinf"], p["actwtd"]
        )
        b = K.satflo(p["dlayr"], p["dul"], p["sat"], p["sw"], p["swcn"], p["swcon"])
        u = K.up_flow(p["dlayr"], p["dul"], p["ll"], p["sat"], p["sw"], p["avail"])
        return a, b, u

    return jax.vmap(one)


def test_bucket_layer_loops_bit_identical_unrolled_and_looped() -> None:
    p = _profiles(48, seed=11)
    out = {}
    for on in (False, True):
        with E.execution(depth_unroll=on):
            fn = _bucket_layer_loops()
            out[on] = jax.jit(fn)(p)
            out[f"whiles_{on}"] = _whiles(_bucket_layer_loops(), p)
    # the two programs differ in structure (loops vs straight-line) ...
    assert out["whiles_False"] >= 4 and out["whiles_True"] == 0, out
    # ... and give the same numbers, bit for bit
    for a, b in zip(jax.tree.leaves(out[False]), jax.tree.leaves(out[True]), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_runtime_keys_runners_on_the_execution_setting() -> None:
    params, forcing, state0 = toy_inputs(5)
    params = jax.tree.map(lambda v: jnp.stack([v, v]), params)
    model = toy_model()
    with E.execution(depth_unroll=True):
        a = run_batch(model, params, forcing, state0)
    with E.execution(depth_unroll=False):
        b = run_batch(model, params, forcing, state0)
    plat = E.platform_of(params, forcing, state0) or E.target_platform()
    runners = model.__dict__[_RUNNERS_ATTR]
    assert {x for k in runners for x in k if isinstance(x, E.Execution)} == {
        E.Execution(plat, True),
        E.Execution(plat, False),
    }
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
