"""Brooks-Corey hydraulics against an independent scalar reference, on the calibration parameter ranges.

``tests/unit/test_hydraulics.py`` checks the module against itself (round trips, autodiff vs its own
analytic capacity, FD of the JAX functions). This file adds checks against things the module does not
share code with:

* ``_ref_*``: a scalar pure-Python (``math``, always float64) transcription of the RZWQM equations
  (Ahuja et al. 2000, ch. 3; the ``WC``/``SPMOIS``/``WCH``/``POINTK`` segmentation), written from the
  equations and not from ``hydraulics.py``; the JAX curves must agree with it to rounding.
* the fundamental theorem of calculus: ``theta(h2) - theta(h1)`` equals the Gauss-Legendre quadrature
  of ``C(h)`` over ``[h1, h2]`` (a conservation statement: the water stored between two heads is the
  integral of the capacity; independent of autodiff);
* parameter gradients from ``jax.grad`` against central finite differences of the *reference*
  implementation at 50 random ``(h, params)`` points, including ``h = -hb`` exactly and just above and
  below it.

Parameter ranges (:data:`RANGES`) are the union of the calibration bounds in ``all_parameters.csv``
(``Pore Size`` = lambda, ``Ksat``, ``Residual WC``) and of the base values of every horizon of the 15
``RZWQM_sw_batch`` scenarios; ``tests/integration/test_hydraulics_reference.py`` checks that these
constants still cover both. The unit tier needs no data.
"""

from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from agrijax.processes.soil_water.hydraulics import (
    H_FC13,
    H_FC110,
    H_WP,
    SoilHydraulicParams,
    c_of_h,
    derive_rzwqm,
    h_of_theta,
    k_of_h,
    theta_of_h,
)

X64 = bool(jax.config.jax_enable_x64)
#: relative agreement with the float64 reference (float32: a few ulps times the |h|**-lambda conditioning)
REF_REL = 1e-12 if X64 else 2e-5
#: autodiff vs central FD of the reference
FD_REL = 1e-5 if X64 else 5e-3

#: (low, high) of every primary parameter; see the module docstring for the sources.
#: all_parameters.csv: Pore Size 0.14-0.87, Ksat 0.05-6.11, Residual WC 0.02-0.17 (union over the
#: 15 scenarios and 4 calibrated horizons). Scenario base values (105 horizons): hb = hb_k 8.69-92.09 cm,
#: lambda 0.101-0.474, eps 2.381-3.422, ksat 0.001-6.11 cm/hr, theta_r 0.015-0.09, theta_s 0.398-0.547,
#: a1 = n1 = 0. a1 and n1 are also sampled above 0 (RZWQM reference classes: a1 <= 0.002, n1 small).
RANGES: dict[str, tuple[float, float]] = {
    "hb": (8.0, 95.0),
    "lambda_": (0.10, 0.87),
    "eps": (2.3, 3.5),
    "ksat": (0.001, 6.11),
    "theta_r": (0.015, 0.17),
    "theta_s": (0.39, 0.55),
    "hb_k": (8.0, 95.0),
    "a1": (1e-5, 0.002),  # or exactly 0; a1 -> 0+ makes the linear inverse ill-conditioned (1/a1)
    "n1": (0.0, 0.3),
}
PRIMARY = ("hb", "lambda_", "eps", "ksat", "theta_r", "theta_s", "hb_k", "n1", "a1")


# ---------------------------------------------------------------------------
# independent scalar reference (float64, math only)
# ---------------------------------------------------------------------------


def _valid(q: dict[str, float]) -> bool:
    """Parameter set with a non-empty Brooks-Corey segment (theta_s - a1 hb > theta_r)."""
    return q["theta_s"] - q["a1"] * q["hb"] - q["theta_r"] > 0.02


def _ref_theta(h: float, q: dict[str, float]) -> float:
    ws, wr, hb, lam, a1 = q["theta_s"], q["theta_r"], q["hb"], q["lambda_"], q["a1"]
    if h >= 0.0:
        return ws
    if h > -hb:
        return ws + a1 * h
    return wr + (ws - wr - a1 * hb) * (hb / -h) ** lam


def _ref_c(h: float, q: dict[str, float]) -> float:
    ws, wr, hb, lam, a1 = q["theta_s"], q["theta_r"], q["hb"], q["lambda_"], q["a1"]
    if h >= 0.0:
        return 0.0
    if h > -hb:
        return a1
    return (ws - wr - a1 * hb) * lam * hb**lam * (-h) ** (-lam - 1.0)


def _ref_k(h: float, q: dict[str, float]) -> float:
    ks, hbk, eps, n1 = q["ksat"], q["hb_k"], q["eps"], q["n1"]
    if h >= 0.0:
        return ks
    if h >= -hbk:
        return ks * max(-h, 1e-6) ** (-n1)
    return ks * hbk ** (eps - n1) * (-h) ** (-eps)


def _ref_h(theta: float, q: dict[str, float]) -> float:
    ws, wr, hb, lam, a1 = q["theta_s"], q["theta_r"], q["hb"], q["lambda_"], q["a1"]
    if theta >= ws:
        return 0.0
    if a1 > 0.0 and theta >= ws - a1 * hb:
        return -(ws - theta) / a1
    return -hb * ((theta - wr) / (ws - wr - a1 * hb)) ** (-1.0 / lam)


def _params(q: dict[str, float]) -> SoilHydraulicParams:
    f = jnp.asarray
    z = f(0.0)
    return SoilHydraulicParams(
        hb=f(q["hb"]), lambda_=f(q["lambda_"]), eps=f(q["eps"]), ksat=f(q["ksat"]),
        theta_r=f(q["theta_r"]), theta_s=f(q["theta_s"]), fc13=z, fc110=z, wp=z,
        hb_k=f(q["hb_k"]), c2=z, n1=f(q["n1"]), a1=f(q["a1"]),
    )  # fmt: skip


def _as_seen(q: dict[str, float]) -> dict[str, float]:
    """The parameters as the JAX side stores them (rounded to float32 in the float32 pass)."""
    return {k: float(jnp.asarray(v)) for k, v in q.items()}


def _from_prim(prim: dict[str, jnp.ndarray]) -> SoilHydraulicParams:
    """Parameter pytree from the primary values (derived fields zero: the curves never read them)."""
    z = jnp.zeros(())
    return SoilHydraulicParams(
        hb=prim["hb"], lambda_=prim["lambda_"], eps=prim["eps"], ksat=prim["ksat"], theta_r=prim["theta_r"],
        theta_s=prim["theta_s"], fc13=z, fc110=z, wp=z, hb_k=prim["hb_k"], c2=z, n1=prim["n1"], a1=prim["a1"],
    )  # fmt: skip


def _close(got: float, ref: float, rel: float, abs_: float = 0.0) -> bool:
    return abs(got - ref) <= rel * max(abs(ref), abs(got)) + abs_


# ---------------------------------------------------------------------------
# hypothesis strategies
# ---------------------------------------------------------------------------


def _f(name: str) -> st.SearchStrategy[float]:
    lo, hi = RANGES[name]
    return st.floats(lo, hi, allow_nan=False, allow_infinity=False)


@st.composite
def param_sets(draw: st.DrawFn) -> dict[str, float]:
    q = {name: draw(_f(name)) for name in ("hb", "lambda_", "eps", "ksat", "theta_r", "theta_s")}
    # every scenario has hb_k == hb and a1 == n1 == 0: sample that case often, the general case too
    q["hb_k"] = draw(st.one_of(st.just(q["hb"]), _f("hb_k")))
    q["a1"] = draw(st.one_of(st.just(0.0), _f("a1")))
    q["n1"] = draw(st.one_of(st.just(0.0), _f("n1")))
    return q


#: heads relative to hb: 1e-3 hb .. 1e4 hb (|h| up to ~1e6 cm), the joints exactly and 1e-6 around them
_H_REL = np.sort(np.concatenate([-np.logspace(-3, 4, 60), [-1.0, -(1 - 1e-6), -(1 + 1e-6), 0.0, 2.0]]))


@jax.jit
def _curves(p: SoilHydraulicParams, h: jnp.ndarray, theta: jnp.ndarray):
    return theta_of_h(h, p), c_of_h(h, p), k_of_h(h, p), h_of_theta(theta, p)


@settings(derandomize=True, database=None, deadline=None, max_examples=80)
@given(q=param_sets())
def test_curves_match_independent_reference(q: dict[str, float]) -> None:
    """theta, C, K and h(theta) equal the scalar float64 transcription on 65 heads and 40 contents."""
    assume(_valid(q))
    q = _as_seen(q)
    h = _H_REL * q["hb"]
    h = np.concatenate([h, -q["hb_k"] * np.array([1.0, 1 - 1e-6, 1 + 1e-6])])
    span = q["theta_s"] - q["a1"] * q["hb"] - q["theta_r"]
    se_lo = max(1e-3, (q["hb"] / 1e20) ** q["lambda_"])  # keep |h| far from the H_MIN = -1e30 floor
    theta = np.concatenate(
        [
            q["theta_r"] + span * np.geomspace(se_lo, 1.0, 30)[:-1],
            q["theta_s"] - q["a1"] * q["hb"] * np.linspace(0, 1, 11),
        ]
    )
    th, c, k, hh = (np.asarray(x) for x in _curves(_params(q), jnp.asarray(h), jnp.asarray(theta)))
    for i, x in enumerate(h):
        x = float(jnp.asarray(x))  # the value the JAX side actually saw (float32 in the f32 pass)
        assert _close(float(th[i]), _ref_theta(x, q), REF_REL), (x, th[i], _ref_theta(x, q))
        assert _close(float(c[i]), _ref_c(x, q), REF_REL * 10, 1e-300), (x, c[i], _ref_c(x, q))
        assert _close(float(k[i]), _ref_k(x, q), REF_REL * 10), (x, k[i], _ref_k(x, q))
    # the inverse loses digits as 1/Se (theta - theta_r cancels): compare in theta space instead. On the
    # linear segment an ulp of theta moves h by ulp/a1, which lands on the Brooks-Corey side of -hb with
    # slope C(-hb): allow that much in float32 (in float64 it is below 1e-12 for a1 >= 1e-5).
    ulp = float(np.finfo(np.float64 if X64 else np.float32).eps) * q["theta_s"]
    c_joint = _ref_c(-q["hb"], q)
    rt_tol = (1e-12 if X64 else 2e-6) + (4 * ulp / q["a1"] * c_joint if q["a1"] > 0 and not X64 else 0.0)
    for i, t in enumerate(theta):
        t = float(jnp.asarray(t))
        ref = _ref_h(t, q)
        assert math.isfinite(float(hh[i]))
        if X64:
            assert _close(float(hh[i]), ref, 1e-9), (t, hh[i], ref)
        assert _close(_ref_theta(float(hh[i]), q), t, 0.0, rt_tol), (t, hh[i], ref)


@settings(derandomize=True, database=None, deadline=None, max_examples=60)
@given(q=param_sets())
def test_physical_properties(q: dict[str, float]) -> None:
    """Bounds, ordering theta_r < wp < fc13 < fc110 <= theta_s, monotone theta and K (n1 = 0), K <= ksat."""
    assume(_valid(q))
    q = _as_seen(q)
    p = derive_rzwqm(_params(q))
    wr, ws, ks = q["theta_r"], q["theta_s"], q["ksat"]
    assert wr < float(p.wp) < float(p.fc13) < float(p.fc110) <= ws
    # the derived values are the reference curve at RZWQM's heads
    for val, hh in ((p.fc13, H_FC13), (p.fc110, H_FC110), (p.wp, H_WP)):
        assert _close(float(val), _ref_theta(hh, q), REF_REL)
    h = jnp.asarray(_H_REL * q["hb"])
    th, c, k, _ = (np.asarray(x) for x in _curves(_params(q), h, jnp.asarray([ws])))
    slack = 1e-14 if X64 else 1e-6
    assert np.all((th >= wr - slack) & (th <= ws + slack))
    assert np.all(np.diff(th) >= -slack) and np.all(c >= 0.0)
    assert np.all(np.isfinite(k)) and np.all(k > 0.0)
    if q["n1"] == 0.0:
        assert np.all(k <= ks * (1 + slack))
        assert np.all(np.diff(k) >= -slack * ks)


_GL_X, _GL_W = np.polynomial.legendre.leggauss(40)


def _gl(f, a: float, b: float, n_sub: int = 40) -> float:
    """Composite 40-point Gauss-Legendre quadrature of the scalar-vectorised ``f`` on [a, b] (log-spaced)."""
    edges = -np.geomspace(-a, -b, n_sub + 1) if a < 0 and b < 0 else np.linspace(a, b, n_sub + 1)
    tot = 0.0
    for lo, hi in itertools.pairwise(edges):
        x = 0.5 * (hi - lo) * _GL_X + 0.5 * (hi + lo)
        tot += 0.5 * (hi - lo) * float(np.sum(_GL_W * np.asarray(f(x))))
    return tot


@settings(derandomize=True, database=None, deadline=None, max_examples=40)
@given(q=param_sets(), lo_rel=st.floats(1.0, 1e3), hi_rel=st.floats(1e-3, 1.0))
def test_capacity_integrates_to_storage(q: dict[str, float], lo_rel: float, hi_rel: float) -> None:
    """theta(h2) - theta(h1) = integral of C(h) dh over [h1, h2] (quadrature split at the -hb joint)."""
    assume(_valid(q))
    p = _params(q)
    hb = q["hb"]
    h1, h2 = -hb * (1.0 + lo_rel), -hb * hi_rel

    def cap(x: np.ndarray) -> jnp.ndarray:
        return c_of_h(jnp.asarray(x), p)

    stored = _gl(cap, h1, -hb) + _gl(cap, -hb, h2)
    delta = float(theta_of_h(h2, p) - theta_of_h(h1, p))
    assert stored == pytest.approx(delta, rel=1e-9 if X64 else 1e-4, abs=1e-13 if X64 else 1e-6)


# ---------------------------------------------------------------------------
# parameter gradients at 50 random (h, params) points vs central FD of the reference
# ---------------------------------------------------------------------------


def _random_points(n: int = 50, seed: int = 20260923) -> list[tuple[dict[str, float], float, str]]:
    """``(params, h, kind)``; ``kind`` in {random, at_hb, above_hb, below_hb, at_hbk}.

    Every fifth point is at the retention joint exactly; the "above"/"below" points sit at
    ``-hb (1 -+ 1e-4)``, far enough from the kink for a relative FD step of 1e-7.
    """
    rng = np.random.default_rng(seed)
    kinds = ["random", "at_hb", "above_hb", "below_hb", "at_hbk"]
    out: list[tuple[dict[str, float], float, str]] = []
    while len(out) < n:
        q = {k: float(rng.uniform(*RANGES[k])) for k in PRIMARY}
        if rng.random() < 0.5:
            q["hb_k"] = q["hb"]
        if rng.random() < 0.5:
            q["a1"] = 0.0
        if rng.random() < 0.5:
            q["n1"] = 0.0
        if not _valid(q):
            continue
        kind = kinds[len(out) % len(kinds)]
        hb = q["hb"]
        h = {
            "random": -(10 ** rng.uniform(-1.0, 5.0)),
            "at_hb": -hb,
            "above_hb": -hb * (1 - 1e-4),
            "below_hb": -hb * (1 + 1e-4),
            "at_hbk": -q["hb_k"],
        }[kind]
        out.append((q, h, kind))
    return out


POINTS = _random_points()

_JAX_FNS = {"theta": theta_of_h, "c": c_of_h, "k": k_of_h}
_REF_FNS = {"theta": _ref_theta, "c": _ref_c, "k": _ref_k}


def _grad_fn(name: str):
    fn = _JAX_FNS[name]

    def f(prim: dict[str, jnp.ndarray], h: jnp.ndarray) -> jnp.ndarray:
        p = _from_prim(prim)
        return fn(h, p)

    return jax.jit(jax.grad(f, argnums=(0, 1)))


_GRADS = {name: _grad_fn(name) for name in _JAX_FNS}


def _scheme(name: str, fn: str, q: dict[str, float], h: float) -> str:
    """Central FD, except for the parameter that *is* the kink location when h sits exactly on it.

    ``theta``/``C`` put ``h = -hb`` on the Brooks-Corey side (``h > -hb`` is linear), which a backward
    step in ``hb`` stays on; ``K`` puts ``h = -hb_k`` on the wet side (``h >= -hb_k``), which a forward
    step in ``hb_k`` stays on. A central difference there averages two different one-sided slopes.
    """
    if fn in ("theta", "c") and name == "hb" and h == -q["hb"]:
        return "backward"
    if fn == "k" and name == "hb_k" and h == -q["hb_k"]:
        return "forward"
    if name in ("a1", "n1") and q[name] == 0.0:
        return "forward"  # stay in the valid (non-negative) range
    if fn == "k" and name == "h" and h == -q["hb_k"]:
        return "forward"
    if fn in ("theta", "c") and name == "h" and h == -q["hb"]:
        return "backward"
    return "central"


def _fd(g, x: float, step: float, scheme: str) -> float:
    if scheme == "central":
        return (g(x + step) - g(x - step)) / (2 * step)
    if scheme == "backward":
        return (3 * g(x) - 4 * g(x - step) + g(x - 2 * step)) / (2 * step)
    return (-3 * g(x) + 4 * g(x + step) - g(x + 2 * step)) / (2 * step)


@pytest.mark.parametrize("fn", ["theta", "c", "k"])
def test_param_gradients_vs_reference_fd(fn: str) -> None:
    """jax.grad wrt every primary parameter and h equals the FD of the float64 reference at 50 points."""
    ref = _REF_FNS[fn]
    n_checked = 0
    kinds_seen: set[str] = set()
    for q, h, kind in POINTS:
        prim = {k: jnp.asarray(v) for k, v in q.items()}
        g_prim, g_h = _GRADS[fn](prim, jnp.asarray(h))
        f0 = abs(ref(h, q))
        for name in (*PRIMARY, "h"):
            got = float(g_h) if name == "h" else float(g_prim[name])
            assert math.isfinite(got), (fn, name, kind)
            x0 = h if name == "h" else q[name]
            step = 1e-7 * max(abs(x0), 1e-2)
            scheme = _scheme(name, fn, q, h)

            def g1(v: float, name: str = name, q: dict[str, float] = q, h: float = h) -> float:
                if name == "h":
                    return ref(v, q)
                return ref(h, {**q, name: v})

            want = _fd(g1, x0, step, scheme)
            noise = 1e-15 * f0 / step * 64  # float64 roundoff of the stencil
            assert _close(got, want, FD_REL, noise + (0.0 if X64 else 1e-3 * f0 / max(abs(x0), 1e-2))), (
                f"{fn} d/d{name} at {kind} h={h}: autodiff {got:.8e} vs FD {want:.8e} ({scheme})"
            )
            n_checked += 1
        kinds_seen.add(kind)
    assert n_checked == 50 * (len(PRIMARY) + 1)
    assert kinds_seen == {"random", "at_hb", "above_hb", "below_hb", "at_hbk"}


def test_h_of_theta_gradients_vs_reference_fd() -> None:
    """jax.grad of h(theta) wrt every retention parameter and theta vs FD of the reference inverse."""
    rng = np.random.default_rng(7)

    def f(prim: dict[str, jnp.ndarray], t: jnp.ndarray) -> jnp.ndarray:
        return h_of_theta(t, _from_prim(prim))

    grad = jax.jit(jax.grad(f, argnums=(0, 1)))
    names = ("hb", "lambda_", "theta_r", "theta_s", "a1")
    for q, _, _ in POINTS:
        span = q["theta_s"] - q["a1"] * q["hb"] - q["theta_r"]
        # a content on the Brooks-Corey segment (Se in [0.05, 0.95]) and, when a1 > 0, one on the line
        thetas = [q["theta_r"] + span * rng.uniform(0.05, 0.95)]
        if q["a1"] > 0.0:
            thetas.append(q["theta_s"] - q["a1"] * q["hb"] * rng.uniform(0.1, 0.9))
        for t in thetas:
            g_prim, g_t = grad({k: jnp.asarray(v) for k, v in q.items()}, jnp.asarray(t))
            h0 = abs(_ref_h(t, q))
            for name in (*names, "theta"):
                got = float(g_t) if name == "theta" else float(g_prim[name])
                x0 = t if name == "theta" else q[name]
                step = 1e-7 * max(abs(x0), 1e-3)
                scheme = "forward" if name == "a1" and q["a1"] == 0.0 else "central"

                def g1(v: float, name: str = name, t: float = t, q: dict[str, float] = q) -> float:
                    return _ref_h(v, q) if name == "theta" else _ref_h(t, {**q, name: v})

                want = _fd(g1, x0, step, scheme)
                noise = 64e-15 * h0 / step
                assert _close(
                    got, want, FD_REL * 10, noise + (0.0 if X64 else 1e-2 * h0 / max(abs(x0), 1e-3))
                ), f"h_of_theta d/d{name} at theta={t}: {got:.8e} vs FD {want:.8e}"
