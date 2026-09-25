"""Gradient helpers of ``agrijax.core.grad`` (plan 19 A11; doc 17 section 5).

* The forward value of every helper is bit-identical across the gradient modes ``exact``,
  ``ste`` and ``implicit``, and equals an independent NumPy reference (``trunc``, Fortran ``NINT``
  half away from zero, the hard threshold, ``where``).
* Derivatives per mode: 0 through truncation / rounding / thresholds in ``exact``; identity
  (straight-through) through truncation and rounding in ``ste`` / ``implicit``; the one-step ramp
  ``-1/r`` on the crossing day only for :func:`event_ste`; the jump ``a - b`` times the indicator
  derivative through :func:`select_ste`. All finite, including a zero accumulator step.
* Mode resolution (argument, context, environment, default), the runtime cache key, and the old
  CERES location re-exporting the same objects.
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
    -1/r_d is the one-step secant of the staircase at the crossing (doc 17 section 5.2)."""
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
    runners = model.__dict__["_agrijax_batch_runners"]
    assert {k[-1] for k in runners} == {"ste", "exact"}


def test_old_ceres_location_reexports_the_core_helpers() -> None:
    from agrijax.processes.crop.ceres_maize import _util

    for name in ("safe_div", "trunc_st", "curv_lin", "tabex"):
        assert getattr(_util, name) is getattr(G, name)


def test_safe_div_is_finite_everywhere() -> None:
    num = jnp.asarray([1.0, 0.0, -2.0, 3.0])
    den = jnp.asarray([2.0, 0.0, 0.0, -4.0])
    np.testing.assert_array_equal(np.asarray(G.safe_div(num, den, 9.0)), [0.5, 9.0, 9.0, -0.75])
    g = jax.grad(lambda d: jnp.sum(G.safe_div(num, d)))(den)
    assert np.all(np.isfinite(np.asarray(g)))
