"""Post-tillage two-segment hydraulics (:class:`TilledSoilHydraulicParams`), no data.

* with ``original == current`` every function returns the single-segment values bit for bit,
  and so does a Richards step;
* the segments are the ones of RZWQM2 ``WC``/``SPMOIS``/``WCH``/``POINTK`` (Ahuja et al. 1998):
  current curve above ``-10 hb_current`` (``h = -10 hb`` itself on the current curve), original
  Brooks-Corey below; ``K`` original at and below ``-10 hb_k,current``; checked against a
  numpy evaluation of the published equations written here independently;
* ``C(h)`` is the derivative of ``theta(h)`` on every segment; ``h(theta)`` inverts ``theta(h)``
  on the original branch (below ``theta_o(-10 hb_o)``) and on the current one;
* gradients with respect to every parameter of both curves are finite and match central
  differences, so both curves are calibratable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.processes.soil_water import hydraulics as H
from agrijax.processes.soil_water.richards import RichardsConfig, RichardsGrid, richards_step

X64 = bool(jax.config.read("jax_enable_x64"))
EXACT = 1e-12 if X64 else 1e-5
#: float32: a head within this relative distance of a segment split is ambiguous (the split head
#: -10 hb rounds differently in float32 and in the float64 reference), so such probes are skipped
NEAR_SPLIT = 0.0 if X64 else 1e-5
INV_REL = 1e-9 if X64 else 2e-5
ROUND_REL = 1e-12 if X64 else 2e-6
GRAD_REL = 1e-8 if X64 else 1e-5
FD_REL = 1e-4 if X64 else 5e-2
FD_EPS = 1e-6 if X64 else 1e-3
F = 10.0  # the tillage split factor of RZWQM2 (H.RZWQM_HYDRAULICS.tillage_split_*)


def _p(hb, lam, eps, ksat, wr, ws, a1, hbk=None) -> H.SoilHydraulicParams:
    f = jnp.asarray
    n = np.shape(hb)
    z = f(np.zeros(n))
    return H.derive_rzwqm(
        H.SoilHydraulicParams(
            hb=f(hb), lambda_=f(lam), eps=f(eps), ksat=f(ksat), theta_r=f(wr), theta_s=f(ws),
            fc13=z, fc110=z, wp=z, hb_k=f(hb if hbk is None else hbk), c2=z, n1=z, a1=f(a1),
        )
    )  # fmt: skip


# two horizons: a loosened, wetter tilled curve over its start-up curve (CA-TPA-like values)
ORIG = _p(
    [14.6545, 20.0], [0.22, 0.30], [2.966, 3.2], [5.41, 2.0], [0.055, 0.04], [0.453, 0.42], [0.0, 0.002]
)
CUR = _p([11.0, 16.0], [0.19, 0.27], [2.8, 3.0], [7.3, 2.6], [0.055, 0.04], [0.49, 0.45], [0.0, 0.002])
TILLED = H.TilledSoilHydraulicParams(current=CUR, original=ORIG)


def _np_theta(h: np.ndarray, hb, lam, wr, ws, a1) -> np.ndarray:
    """Modified Brooks-Corey theta(h), Ahuja et al. (2000) ch. 3, written with numpy."""
    b = (ws - wr - a1 * hb) * hb**lam
    out = np.where(h >= 0.0, ws, np.where(h > -hb, ws + a1 * h, wr + b * np.abs(h) ** (-lam)))
    return out


def _heads(k: int) -> np.ndarray:
    hb_c = float(CUR.hb[k])
    return np.concatenate(
        [
            [1.0, 0.0, -0.5 * hb_c, -hb_c, -2.0 * hb_c, -F * hb_c, -F * hb_c * (1 + 1e-9)],
            -np.geomspace(F * hb_c * 1.001, 15000.0, 25),
        ]
    )


def _away(h: np.ndarray, *splits: float) -> np.ndarray:
    """Mask of probes farther than NEAR_SPLIT (relative) from every split head."""
    m = np.ones(np.shape(h), dtype=bool)
    for s in splits:
        m &= np.abs(h - s) > NEAR_SPLIT * abs(s)
    return m


def _col(p: H.SoilHydraulicParams, k: int) -> H.SoilHydraulicParams:
    return jax.tree_util.tree_map(lambda x: x[k], p)


@pytest.mark.parametrize("k", [0, 1])
def test_segments_are_rzwqm2s(k: int) -> None:
    h = _heads(k)
    c, o = _col(CUR, k), _col(ORIG, k)
    tp = H.TilledSoilHydraulicParams(current=c, original=o)
    got = np.asarray(H.theta_of_h(jnp.asarray(h), tp))
    cur = _np_theta(h, *(float(getattr(c, n)) for n in ("hb", "lambda_", "theta_r", "theta_s", "a1")))
    org = _np_theta(h, *(float(getattr(o, n)) for n in ("hb", "lambda_", "theta_r", "theta_s", "a1")))
    want = np.where(h < -F * float(c.hb), org, cur)
    m = _away(h, -F * float(c.hb))
    np.testing.assert_allclose(got[m], want[m], rtol=EXACT, atol=0)
    # h = -10 hb itself is on the current curve (WC: TH .LT. -10 hb), the jump is real;
    # only decidable in float64 (in float32 the split head itself rounds)
    j = int(np.flatnonzero(h == -F * float(c.hb))[0])
    assert abs(cur[j] - org[j]) > 1e-3
    if X64:
        assert got[j] == pytest.approx(cur[j], rel=EXACT)


@pytest.mark.parametrize("k", [0, 1])
def test_k_segments_are_rzwqm2s(k: int) -> None:
    c, o = _col(CUR, k), _col(ORIG, k)
    tp = H.TilledSoilHydraulicParams(current=c, original=o)
    hbk = float(c.hb_k)
    h = np.array([0.0, -0.5 * hbk, -2 * hbk, -F * hbk * (1 - 1e-9), -F * hbk, -3 * F * hbk, -15000.0])
    got = np.asarray(H.k_of_h(jnp.asarray(h), tp))
    c2_o = float(o.ksat) * float(o.hb_k) ** (float(o.eps) - float(o.n1))
    org = c2_o * np.abs(h) ** (-float(o.eps))
    cur = np.asarray(H.k_of_h(jnp.asarray(h), c))
    want = np.where(h <= -F * hbk, org, cur)  # POINTK: TH .LE. -10 hb_k
    m = _away(h, -F * hbk)
    np.testing.assert_allclose(got[m], want[m], rtol=EXACT, atol=0)
    if X64:  # the split head itself is on the start-up segment; float32 cannot decide it
        assert got[4] == pytest.approx(org[4], rel=EXACT)


def test_untilled_is_bit_identical_to_the_single_segment_curve() -> None:
    u = H.TilledSoilHydraulicParams.untilled(CUR)
    h = jnp.asarray(np.stack([_heads(0), _heads(1)], axis=1))
    for fn in (H.theta_of_h, H.c_of_h, H.k_of_h):
        np.testing.assert_array_equal(np.asarray(fn(h, u)), np.asarray(fn(h, CUR)))
    th = H.theta_of_h(h, CUR)
    np.testing.assert_array_equal(np.asarray(H.h_of_theta(th, u)), np.asarray(H.h_of_theta(th, CUR)))


def test_c_of_h_is_the_derivative_on_every_segment() -> None:
    for k in (0, 1):
        h = _heads(k)
        hb = float(CUR.hb[k])
        h = h[
            (h < 0.0)
            & (np.abs(h + F * hb) > max(1e-6, NEAR_SPLIT * F * hb))
            & (np.abs(h + hb) > max(1e-6, NEAR_SPLIT * hb))
        ]
        tk = jax.tree_util.tree_map(lambda x, k=k: x[k], TILLED)
        grad = jax.vmap(jax.grad(lambda x, tk=tk: H.theta_of_h(x, tk)))(jnp.asarray(h))
        np.testing.assert_allclose(np.asarray(H.c_of_h(jnp.asarray(h), tk)), np.asarray(grad), rtol=GRAD_REL)


def test_h_of_theta_inverts_both_branches() -> None:
    for k in (0, 1):
        tk = jax.tree_util.tree_map(lambda x, k=k: x[k], TILLED)
        o = tk.original
        th_split = float(H.theta_of_h(jnp.asarray(-F * float(o.hb)), o))
        # original branch: theta at or below WC10S2, heads well below both split heads
        h_dry = -np.geomspace(F * max(float(tk.current.hb), float(o.hb)) * 1.01, 15000.0, 12)
        th = H.theta_of_h(jnp.asarray(h_dry), o)
        assert bool(jnp.all(th <= th_split))
        np.testing.assert_allclose(np.asarray(H.h_of_theta(th, tk)), h_dry, rtol=INV_REL)
        np.testing.assert_allclose(
            np.asarray(H.theta_of_h(H.h_of_theta(th, tk), tk)), np.asarray(th), rtol=ROUND_REL
        )
        # current branch: above WC10S2 the inverse is the current curve's
        h_wet = -np.geomspace(float(tk.current.hb) * 1.01, F * float(tk.current.hb) * 0.99, 8)
        thw = H.theta_of_h(jnp.asarray(h_wet), tk.current)
        sel = np.asarray(thw > th_split)
        assert sel.any()
        np.testing.assert_array_equal(
            np.asarray(H.h_of_theta(thw, tk))[sel], np.asarray(H.h_of_theta(thw, tk.current))[sel]
        )


@pytest.mark.parametrize("fn_name", ["theta_of_h", "c_of_h", "k_of_h"])
def test_gradients_of_both_curves_finite_and_match_fd(fn_name: str) -> None:
    fn = getattr(H, fn_name)
    h = jnp.asarray([-5.0, -60.0, -130.0, -900.0])  # current (h > -10 hb) and original branches

    def f(p: H.TilledSoilHydraulicParams) -> jax.Array:
        return jnp.sum(fn(h, p))

    tk = jax.tree_util.tree_map(lambda x: x[0], TILLED)
    g = jax.grad(f)(tk)
    leaves, tdef = jax.tree_util.tree_flatten(tk)
    gl = jax.tree_util.tree_leaves(g)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in gl)
    touched = 0
    for i, (x, gx) in enumerate(zip(leaves, gl, strict=True)):
        step = FD_EPS * max(1.0, abs(float(x)))

        def at(v: float, i: int = i) -> float:
            ls = list(leaves)
            ls[i] = jnp.asarray(v)
            return float(f(jax.tree_util.tree_unflatten(tdef, ls)))

        fd = (at(float(x) + step) - at(float(x) - step)) / (2 * step)
        np.testing.assert_allclose(float(gx), fd, rtol=FD_REL, atol=1e-10 if X64 else 1e-3)
        touched += abs(float(gx)) > 0
    assert touched >= 4  # both curves feed the result


def test_node_gather_jit_and_vmap() -> None:
    nh = (0, 0, 1, 1, 1)
    tp = H.TilledSoilHydraulicParams(
        current=CUR.replace(node_horizon=nh), original=ORIG.replace(node_horizon=nh)
    )
    h = jnp.asarray([-3.0, -200.0, -20.0, -170.0, -5000.0])
    per_node = jax.jit(H.theta_of_h)(h, tp)
    want = [
        float(H.theta_of_h(h[i], jax.tree_util.tree_map(lambda x, j=j: x[j], TILLED)))
        for i, j in enumerate(nh)
    ]
    np.testing.assert_allclose(np.asarray(per_node), want, rtol=EXACT)
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), TILLED)
    out = jax.vmap(lambda p: H.k_of_h(jnp.asarray([-50.0, -500.0]), p))(batch)
    assert out.shape == (2, 2)


def test_node_maps_must_agree() -> None:
    with pytest.raises(ValueError, match="node"):
        H.TilledSoilHydraulicParams(current=CUR.replace(node_horizon=(0, 1)), original=ORIG)


def test_richards_step_untilled_is_bit_identical() -> None:
    n = 6
    tl = jnp.full(n, 2.0)
    grid = RichardsGrid(tl=tl, delz=jnp.full(n - 1, 2.0), dz_top=jnp.asarray(1.0))
    soil = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[0], (n,)), CUR)
    h0 = jnp.asarray([-80.0, -120.0, -160.0, -220.0, -300.0, -400.0])
    args = (
        h0, H.theta_of_h(h0, soil), jnp.asarray(0.0),
    )  # fmt: skip
    rest = (grid, jnp.asarray(0.05), jnp.asarray(0.01), jnp.zeros(n), jnp.asarray(0.5), jnp.asarray(1.0),
            jnp.asarray(-15000.0), jnp.asarray(0.0))  # fmt: skip
    cfg = RichardsConfig(n_iter=8)
    a = richards_step(*args, soil, *rest, cfg=cfg)
    b = richards_step(*args, H.TilledSoilHydraulicParams.untilled(soil), *rest, cfg=cfg)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    # and a real two-segment soil changes the step (the dry nodes sit below -10 hb)
    tilled = H.TilledSoilHydraulicParams(
        current=soil, original=jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x[0], (n,)), ORIG)
    )
    c = richards_step(h0, H.theta_of_h(h0, tilled), jnp.asarray(0.0), tilled, *rest, cfg=cfg)
    assert float(jnp.max(jnp.abs(c.theta - a.theta))) > 1e-4
    assert abs(float(c.balance_error)) < (1e-10 if X64 else 1e-4)
