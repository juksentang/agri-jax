"""Gradient helpers of ``agrijax.core.grad``.

* The forward value of every helper is bit-identical across the gradient modes ``exact``,
  ``ste`` and ``implicit``, and equals an independent NumPy reference (``trunc``, Fortran ``NINT``
  half away from zero, the hard threshold, ``where``).
* Derivatives per mode: 0 through truncation / rounding / thresholds in ``exact``; identity
  (straight-through) through truncation and rounding in ``ste`` / ``implicit``; the one-step ramp
  ``-1/r`` on the crossing day only for :func:`event_ste`; the jump ``a - b`` times the indicator
  derivative through :func:`select_ste`. All finite, including a zero accumulator step.
* Mode resolution (argument, context, environment, default), the runtime cache key, and the old
  CERES location re-exporting the same objects.
* ``jax.jit`` keeps the mode of its first trace (the pitfall that made an ``exact`` calibration
  test reuse the ``ste`` program); :func:`bind_gradient_mode` fixes the mode with the function,
  so a jit of a bound function has its mode, per-mode bound functions really differ, and a bound
  function called inside an explicit context of another mode raises ``ModeMismatchError``.
* :func:`coef_div`: a Python-number divisor divides directly (the traced program of a default
  coefficient is unchanged; 0 is rejected), an array divisor is guarded before the division (the
  same quotient bit for bit where it is not 0, finite value and gradient where it is).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import grad as G
from agrijax.core.runtime import run_batch

from .toy import toy_inputs, toy_model

MODES = ("exact", "ste", "implicit")
X64 = bool(jax.config.read("jax_enable_x64"))


def _bits(x) -> bytes:
    return np.asarray(x).tobytes()


def _nint_ref(x: np.ndarray) -> np.ndarray:
    """Fortran NINT: half away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


XS = np.array(
    [-3.5, -2.5, -1.5, -0.5, -0.49, -0.3, 0.0, 0.3, 0.49, 0.5, 1.5, 2.5, 2.4999, 3.5001, 123.456, -987.654, 1e7 + 0.5]
)  # fmt: skip


# ------------------------------------------------------------------------------------ forward
@pytest.mark.parametrize("fn", ["trunc", "round0", "round3"])
def test_quantisers_forward_bit_identical_and_reference(fn: str) -> None:
    x = jnp.asarray(XS)
    outs = []
    for m in MODES:
        if fn == "trunc":
            outs.append(jax.jit(lambda v, m=m: G.trunc_st(v, mode=m))(x))
        elif fn == "round0":
            outs.append(jax.jit(lambda v, m=m: G.round_st(v, mode=m))(x))
        else:
            outs.append(jax.jit(lambda v, m=m: G.round_st(v, 3, mode=m))(x))
    assert _bits(outs[0]) == _bits(outs[1]) == _bits(outs[2])
    xn = np.asarray(x)  # the reference in the same precision as the run
    k = xn.dtype.type(1000.0)
    if fn == "trunc":
        ref = np.trunc(xn)
    elif fn == "round0":
        ref = _nint_ref(xn)
    else:
        ref = _nint_ref(xn * k) / k
    if fn == "round3" and jax.default_backend() != "cpu":
        # GPU float32 division is not correctly rounded (the CPU CI pass checks exact equality)
        np.testing.assert_array_max_ulp(np.asarray(outs[0]), ref, maxulp=2)
    else:
        np.testing.assert_array_equal(np.asarray(outs[0]), ref)


def test_nint_is_half_away_from_zero_not_half_even() -> None:
    x = jnp.asarray([0.5, 1.5, 2.5, -0.5, -2.5])
    np.testing.assert_array_equal(np.asarray(G.round_st(x)), [1.0, 2.0, 3.0, -1.0, -3.0])
    # the largest double below 0.5 rounds to 0 (floor(x + 0.5) would give 1)
    if X64:
        below = np.nextafter(0.5, 0.0)
        assert float(G.round_st(jnp.asarray(below))) == 0.0


def test_trunc_st_unchanged_from_the_ceres_version() -> None:
    """Same expression as before the move: the CERES outputs stay bit-identical."""
    x = jnp.asarray(XS * 1000.0)
    old = x + jax.lax.stop_gradient(jnp.trunc(x) - x)
    assert _bits(G.trunc_st(x)) == _bits(old)


def test_event_and_select_forward_bit_identical() -> None:
    margin = jnp.asarray([-5.0, -0.1, 0.0, 0.1, 3.0, 20.0])
    rate = jnp.asarray([10.0, 10.0, 10.0, 0.0, 10.0, 10.0])
    a = jnp.asarray([1.0, -0.0, 2.0, 3.0, 4.0, 5.0])
    b = jnp.asarray([-1.0, 7.0, 0.0, -0.0, 8.0, 9.0])
    inds, sels = [], []
    for m in MODES:
        ind = jax.jit(lambda mg, r, m=m: G.event_ste(mg, r, mode=m))(margin, rate)
        inds.append(ind)
        sels.append(jax.jit(lambda i, x, y, m=m: G.select_ste(i, x, y, mode=m))(ind, a, b))
    assert _bits(inds[0]) == _bits(inds[1]) == _bits(inds[2])
    assert _bits(sels[0]) == _bits(sels[1]) == _bits(sels[2])
    np.testing.assert_array_equal(np.asarray(inds[0]), (np.asarray(margin) >= 0).astype(float))
    ref = np.where(np.asarray(margin) >= 0, np.asarray(a), np.asarray(b))
    assert _bits(sels[0]) == _bits(ref.astype(np.asarray(sels[0]).dtype))  # keeps the sign of -0.0


# ---------------------------------------------------------------------------------- derivatives
@pytest.mark.parametrize("mode", MODES)
def test_quantiser_derivatives(mode: str) -> None:
    x = jnp.asarray([0.3, 1.7, -2.2, 12.5])
    g_t = jax.vmap(jax.grad(lambda v: G.trunc_st(v, mode=mode)))(x)
    g_r = jax.vmap(jax.grad(lambda v: G.round_st(v, 2, mode=mode)))(x)
    want = 0.0 if mode == "exact" else 1.0
    np.testing.assert_array_equal(np.asarray(g_t), want)
    np.testing.assert_allclose(np.asarray(g_r), want, rtol=1e-6)


def _staircase(threshold, rates, mode):
    """Days past the threshold on an accumulator S_d = cumsum(r): sum_d event_ste(S_d - P, r_d)."""
    s = jnp.cumsum(rates)
    return jnp.sum(G.event_ste(s - threshold, rates, mode=mode))


@pytest.mark.parametrize("mode", MODES)
def test_event_ste_derivative_is_the_one_step_ramp(mode: str) -> None:
    rates = jnp.asarray([12.0, 9.0, 15.0, 11.0, 13.0, 8.0])
    s = np.cumsum(np.asarray(rates))
    p = 40.0  # crossed on day 3 (S = 36, 47): r_3 = 11
    g = float(jax.grad(_staircase)(jnp.asarray(p), rates, mode))
    d = int(np.argmax(s >= p))
    assert d == 3
    if mode == "exact":
        assert g == 0.0
    else:
        assert g == pytest.approx(-1.0 / float(rates[d]), rel=1e-6)
    # per-day derivative: non-zero on the crossing day only
    per_day = jax.jacfwd(lambda q: G.event_ste(jnp.cumsum(rates) - q, rates, mode=mode))(jnp.asarray(p))
    nz = np.flatnonzero(np.asarray(per_day))
    assert list(nz) == ([] if mode == "exact" else [d])


def test_event_ste_matches_the_calibration_scale_secant() -> None:
    """The staircase count of days past P changes by one per accumulator step: the ramp slope
    -1/r_d is the one-step secant of the staircase at the crossing."""
    rates = jnp.full(30, 10.0)
    p = 123.0
    g = float(jax.grad(_staircase)(jnp.asarray(p), rates, "ste"))
    secant = (
        float(_staircase(p + 10.0, rates, "exact")) - float(_staircase(p - 10.0, rates, "exact"))
    ) / 20.0
    assert g == pytest.approx(secant, rel=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_event_ste_finite_for_degenerate_steps(mode: str) -> None:
    """A zero or negative step, or a zero margin, gives finite derivatives in every argument."""
    for margin, rate in [(0.0, 0.0), (1e-9, 0.0), (-1e-9, 0.0), (0.0, -3.0), (5.0, 1e-30), (0.0, 1e-30)]:
        g = jax.grad(lambda m_, r_: G.event_ste(m_, r_, mode=mode), argnums=(0, 1))(
            jnp.asarray(margin), jnp.asarray(rate)
        )
        assert all(np.isfinite(float(x)) for x in g), (margin, rate, g)


@pytest.mark.parametrize("mode", MODES)
def test_select_ste_carries_the_jump(mode: str) -> None:
    """Stage-dependent value v = a after the switch, b before: dv/dP = (a - b) * d(ind)/dP in ste."""
    rate, a, b = 10.0, 7.0, 2.0

    def v(p, s):
        ind = G.event_ste(s - p, rate, mode=mode)
        return G.select_ste(ind, a * p, b * p, mode=mode)  # both branches also depend on P

    p, s = 100.0, 104.0  # crossed today, margin 4 < rate
    g = float(jax.grad(v)(jnp.asarray(p), jnp.asarray(s)))
    taken = a  # d(a p)/dp on the taken branch
    if mode == "exact":
        assert g == pytest.approx(taken)
    else:
        assert g == pytest.approx(taken + (a * p - b * p) * (-1.0 / rate))
    # not the crossing day: the jump term vanishes in every mode
    g_late = float(jax.grad(v)(jnp.asarray(p), jnp.asarray(p + 3 * rate)))
    assert g_late == pytest.approx(a)


def test_select_ste_under_vmap_and_reverse_mode() -> None:
    p = jnp.linspace(90.0, 110.0, 9)
    s = jnp.asarray(100.0)

    def f(q):
        ind = G.event_ste(s - q, 10.0)
        return G.select_ste(ind, q**2, 3.0 * q)

    g_rev = jax.vmap(jax.grad(f))(p)
    g_fwd = jax.vmap(jax.jacfwd(f))(p)
    np.testing.assert_allclose(np.asarray(g_rev), np.asarray(g_fwd), rtol=1e-6)
    assert np.all(np.isfinite(np.asarray(g_rev)))


# ------------------------------------------------------------------------------ mode resolution
def test_mode_resolution_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(G.GRADIENT_MODE_ENV, raising=False)
    assert G.current_gradient_mode() == "ste"
    monkeypatch.setenv(G.GRADIENT_MODE_ENV, "exact")
    assert G.current_gradient_mode() == "exact"
    with G.gradient_mode("implicit"):
        assert G.current_gradient_mode() == "implicit"
        assert G.resolve_gradient_mode("ste") == "ste"
        assert G.solver_adjoint() == "implicit"
        with G.gradient_mode("ste"):
            assert G.solver_adjoint() == "unrolled"
    assert G.current_gradient_mode() == "exact"
    with pytest.raises(ValueError, match="unknown gradient mode"):
        G.resolve_gradient_mode("smooth")
    monkeypatch.setenv(G.GRADIENT_MODE_ENV, "bogus")
    with pytest.raises(ValueError):
        G.current_gradient_mode()


def test_context_mode_applies_at_trace_time() -> None:
    f = lambda v: G.trunc_st(v)  # noqa: E731
    with G.gradient_mode("exact"):
        g_exact = jax.grad(f)(jnp.asarray(1.3))
    g_ste = jax.grad(f)(jnp.asarray(1.3))
    assert float(g_exact) == 0.0 and float(g_ste) == 1.0


def test_runtime_cache_key_includes_the_mode() -> None:
    model = toy_model()
    params, forcing, state0 = toy_inputs(5)
    pb = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), params)
    run_batch(model, pb, forcing, state0)
    with G.gradient_mode("exact"):
        run_batch(model, pb, forcing, state0)
    with G.unrounded():
        run_batch(model, pb, forcing, state0)
    runners = model.__dict__["_agrijax_batch_runners"]
    # (gradient mode, unrounded): the unrounded model is a different program
    assert {k[-2:] for k in runners} == {("ste", False), ("exact", False), ("ste", True)}


def test_old_ceres_location_reexports_the_core_helpers() -> None:
    from agrijax.processes.crop.ceres_maize import _util

    for name in ("safe_div", "coef_div", "trunc_st", "curv_lin", "tabex"):
        assert getattr(_util, name) is getattr(G, name)


def test_safe_div_is_finite_everywhere() -> None:
    num = jnp.asarray([1.0, 0.0, -2.0, 3.0])
    den = jnp.asarray([2.0, 0.0, 0.0, -4.0])
    np.testing.assert_array_equal(np.asarray(G.safe_div(num, den, 9.0)), [0.5, 9.0, 9.0, -0.75])
    g = jax.grad(lambda d: jnp.sum(G.safe_div(num, d)))(den)
    assert np.all(np.isfinite(np.asarray(g)))


# ------------------------------------------------------------------------- the REAL*4 store
def _real4_ref(x: np.ndarray) -> np.ndarray:
    """NumPy's float32 cast, promoted back (host side, never folded)."""
    return np.asarray(x).astype(np.float32).astype(np.asarray(x).dtype)


def test_real4_store_survives_jit_on_the_default_backend() -> None:
    """``real4_store`` is the nearest binary32 value under jit and inside a fused comparison on
    the backend JAX runs on (a float64 -> float32 -> float64 convert pair is removed by XLA's GPU
    pipeline with the default ``--xla_allow_excess_precision``; ``reduce-precision`` is not)."""
    rng = np.random.default_rng(7)
    x = np.round(rng.uniform(0.02, 0.6, 4096) * 1e6) / 1e6  # 1e-6-rounded water contents
    x = x.astype(np.float64 if X64 else np.float32)
    want = _real4_ref(x)
    got = np.asarray(jax.jit(G.real4_store)(jnp.asarray(x)))
    assert got.dtype == x.dtype
    assert _bits(got) == _bits(want)

    if not X64:
        # float32: the store is the identity, and round_st's true division is not correctly rounded
        # on GPU in float32 (see agrijax.core.grad._divide); the kernel below is a float64 statement
        return

    # the use in the day: the stored content compared with a REAL*4 limit (SW .LE. LL)
    def kernel(v, ll):
        s = G.real4_store(G.round_st(v, 6))
        return s, s <= ll

    s, le = jax.jit(jax.vmap(kernel))(jnp.asarray(x)[None], jnp.asarray(want)[None])
    assert _bits(np.asarray(s)[0]) == _bits(want)
    assert bool(np.all(np.asarray(le)))
    assert int(np.sum(want != x)) > 0.9 * x.size  # the store is not a no-op in float64


def test_real4_store_edges_and_derivative() -> None:
    """Normal values and overflow as the cast; binary32 subnormals flush to a signed zero
    (documented); the derivative is that of the convert pair (identity, rounded)."""
    dt = np.float64 if X64 else np.float32
    x = np.array([0.0, -0.0, 1.0, -0.31, 2.0**-126, 3.0e38, np.inf, -np.inf], dtype=dt)
    got = np.asarray(jax.jit(G.real4_store)(jnp.asarray(x)))
    assert _bits(got) == _bits(_real4_ref(x))
    assert np.isnan(np.asarray(G.real4_store(jnp.asarray(np.nan, dtype=dt))))
    if X64:
        assert np.asarray(jax.jit(G.real4_store)(jnp.asarray(1e-40))) == 0.0
        assert float(jax.jit(G.real4_store)(jnp.asarray(1e39))) == np.inf
    g = jax.grad(lambda v: jnp.sum(G.real4_store(v) * 3.0))(jnp.asarray([0.25, 0.5], dtype=dt))
    np.testing.assert_array_equal(np.asarray(g), [3.0, 3.0])


def _quantised(x):
    return G.trunc_st(x * 10.0) * x  # d/dx = 10 x + trunc(10 x) in ste, trunc(10 x) in exact


def test_plain_jit_keeps_the_mode_of_its_first_trace() -> None:
    x = jnp.asarray(1.37)
    f = jax.jit(_quantised)
    with G.gradient_mode("ste"):
        g_ste = float(jax.grad(f)(x))
    with G.gradient_mode("exact"):
        g_cached = float(jax.grad(f)(x))  # the pitfall: the cached ste program
        g_fresh = float(jax.grad(jax.jit(lambda v: _quantised(v)))(x))
    assert g_ste == pytest.approx(10.0 * 1.37 + 13.0) and g_fresh == pytest.approx(13.0)
    assert g_cached == g_ste  # documents why simulators are bound to a mode


def test_bound_function_keeps_its_mode_and_refuses_another_context() -> None:
    x = jnp.asarray(1.37)
    bound = {m: G.bind_gradient_mode(_quantised, m) for m in ("ste", "exact")}
    jits = {m: b.jit() for m, b in bound.items()}
    assert bound["exact"].mode == "exact" and jits["ste"].mode == "ste"
    for _ in range(2):  # first call traces, second hits the cache; no context
        g = {m: float(jax.grad(j)(x)) for m, j in jits.items()}
        assert g["ste"] == pytest.approx(10.0 * 1.37 + 13.0) and g["exact"] == pytest.approx(13.0)
    for m in ("ste", "exact"):
        with G.gradient_mode(m):  # the matching context, nested in a fresh outer trace
            g_nested = float(jax.jit(jax.grad(lambda v, m=m: 2.0 * jits[m](v)))(x))
            assert g_nested == pytest.approx(2.0 * g[m])
    for inner, outer in (("ste", "exact"), ("exact", "ste")):
        with G.gradient_mode(outer):
            with pytest.raises(G.ModeMismatchError):
                jits[inner](x)  # cached: checked on the call
            with pytest.raises(G.ModeMismatchError):
                jax.jit(jax.grad(lambda v, i=inner: bound[i](v)))(x)  # traced
    assert _bits(jits["ste"](x)) == _bits(jits["exact"](x))  # forward bit-identical


def test_bind_resolves_the_mode_when_bound() -> None:
    with G.gradient_mode("exact"):
        b = G.bind_gradient_mode(_quantised)
    assert b.mode == "exact"
    assert float(jax.grad(jax.jit(b))(jnp.asarray(1.37))) == pytest.approx(13.0)


def test_coef_div_static_and_array_divisors() -> None:
    num = jnp.asarray([1.0, -2.5, 7.0, 0.3])
    np.testing.assert_array_equal(np.asarray(G.coef_div(num, 250.0)), np.asarray(num / 250.0))
    jaxpr_static = str(jax.make_jaxpr(lambda v: G.coef_div(v, 250.0))(num))
    jaxpr_plain = str(jax.make_jaxpr(lambda v: v / 250.0)(num))
    assert jaxpr_static == jaxpr_plain  # a default coefficient traces the same program
    with pytest.raises(ValueError):
        G.coef_div(num, 0.0)
    den = jnp.asarray([250.0, 0.0, -3.0, 1e-3])
    q = np.asarray(G.coef_div(num, den))
    ref = np.asarray(num) / np.where(np.asarray(den) != 0.0, np.asarray(den), 1.0)
    np.testing.assert_array_equal(q[[0, 2, 3]], ref[[0, 2, 3]])  # bit-identical where den != 0
    assert q[1] == -2.5  # the quotient by 1 where the divisor is 0
    g = jax.grad(lambda d: jnp.sum(G.coef_div(num, d)))(den)
    assert np.all(np.isfinite(np.asarray(g)))


def test_bound_jit_compiles_once() -> None:
    traces = []

    def f(x):
        traces.append(1)  # runs only when traced
        return G.trunc_st(x * 10.0) * x

    b = G.bind_gradient_mode(f, "ste")
    j = b.jit()
    assert j.fn is not f and j.mode == "ste" and b.fn is f
    x = jnp.asarray(1.37)
    for _ in range(3):
        j(x)
    assert len(traces) == 1  # compiled once, cached afterwards
    g = jax.jit(jax.grad(j))
    g(x)
    n = len(traces)
    for _ in range(3):
        g(x)
    assert len(traces) == n  # the derivative program is cached too
    with G.gradient_mode("exact"), pytest.raises(G.ModeMismatchError):
        j(x)  # forward-only call under another mode's context: raises by design


# ------------------------------------------------------------------------------ the unrounded model
def test_unrounded_removes_every_quantiser_and_only_inside_its_context():
    """Under ``unrounded`` (and a ``bind_unrounded`` function) ``trunc_st``, ``round_st`` and
    ``real4_store`` return their argument (value and derivative 1) in every gradient mode; outside it
    they quantise as before (the straight-through derivative is the unrounded model's)."""
    x = jnp.asarray(1.23456789012)

    def f(z):
        return G.trunc_st(z * 1000.0) / 1000.0 + G.round_st(z, 6) + G.real4_store(z)

    for mode in G.GRADIENT_MODES:
        bound = G.bind_gradient_mode(G.bind_unrounded(f), mode)
        y, dy = jax.jvp(bound, (x,), (jnp.ones(()),))
        assert float(y) == pytest.approx(3.0 * float(x), rel=1e-15 if X64 else 1e-6)
        assert float(dy) == pytest.approx(3.0, rel=1e-12 if X64 else 1e-6)
    with G.unrounded():
        assert G.unrounded_active() and float(G.trunc_st(jnp.asarray(2.7))) == 2.7
    assert not G.unrounded_active()
    assert float(G.trunc_st(jnp.asarray(2.7))) == 2.0
    y_ste, dy_ste = jax.jvp(G.bind_gradient_mode(f, "ste"), (x,), (jnp.ones(()),))
    assert float(y_ste) != pytest.approx(3.0 * float(x), rel=1e-6)
    assert float(dy_ste) == pytest.approx(3.0, rel=1e-6)
    assert "Unrounded" in repr(G.bind_unrounded(f))
