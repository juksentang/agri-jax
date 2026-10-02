"""The gradient / sensitivity facade (``agrijax.facade_grad``) on functions with known answers, no data.

* the batched analysis (AD along each coefficient, central differences at two steps, line scans,
  class and level in one batch over scenarios) gives, scenario by scenario, the numbers and the
  classes of ``agrijax.calib.trust.trust_report`` on the same function;
* the trust labels: the phenology coefficients are ``experimental`` whatever the check finds; a
  growth coefficient is ``validated gradient`` on a smooth output, ``falls back`` (the derivative
  column is then the finite difference, not the AD value) on a jumpy one, and the coefficient's
  label is the weakest over its outputs and scenarios, as the calibration's gradient plan says;
* a finite-difference step that would leave the bounds is one-sided, a point outside the bounds
  widens them;
* the tables (per pair, summary over scenarios), the printed report, the outputs / parameters
  parsing and the evaluator cache;
* the two refactored halves of ``agrijax.calib.trust`` (``scan_summary``, ``fd_agreement``) equal
  ``line_scan`` / ``fd_check``.
"""

from __future__ import annotations

import gc
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax import dssat as ajd
from agrijax import facade_grad as fg
from agrijax.calib import TrustConfig, fd_check, line_scan, trust_report
from agrijax.calib.ceres import ceres_space
from agrijax.calib.dssat_day import CUL_ORDER
from agrijax.calib.trust import fd_agreement, scan_summary

pytestmark = [
    pytest.mark.skipif(
        not jax.config.jax_enable_x64, reason="finite differences at a 1e-5 step need float64"
    ),
    pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)"),
]

SPACE = ceres_space(CUL_ORDER)
#: P1 P2 P5 G2 G3 PHINT, inside the MINIMA-MAXIMA box
X = np.array([258.0, 0.5, 700.0, 650.0, 9.0, 46.0])
#: the toy's outputs under real output names: smooth, staircase in P1, jump in P5, jump in G2, constant
SMOOTH, STAIRS, P5_JUMP, G2_JUMP, CONST = "HWAM", "CWAM", "H#AM", "LAI@60", "GWAD@75"
OUTS = [SMOOTH, STAIRS, P5_JUMP, G2_JUMP, CONST]
SCN = 3


def _toy(theta, s):
    """Five outputs of six coefficients and a scenario index ``s`` (a float)."""
    p1, _, p5, g2, g3, phint = theta[0], theta[1], theta[2], theta[3], theta[4], theta[5]
    k = 1.0 + 0.3 * s
    smooth = k * (g2 / 100.0) ** 2 + g3 * jnp.sin(phint / 10.0)
    stairs = 3.0 * jnp.floor(p1 / 20.0) + 0.0 * theta[1]  # AD derivative 0, the value moves along P1
    p5_jump = 0.01 * p5 + 4.0 * (p5 > 703.0 + 2.0 * s)  # one jump inside the P5 scan
    g2_jump = 0.01 * g2 + 4.0 * (g2 > 652.0 + 3.0 * s)  # one jump inside the G2 scan
    const = 7.0 + 0.0 * jnp.sum(theta)
    return jnp.stack([smooth, stairs, p5_jump, g2_jump, const])


def _evaluate(theta, tan, scn):
    f = jax.jit(jax.vmap(lambda th, tg, s: jax.jvp(lambda t: _toy(t, s), (th,), (tg,))))
    y, dy = f(jnp.asarray(theta), jnp.asarray(tan), jnp.asarray(scn, dtype=float))
    return np.asarray(y), np.asarray(dy)


@pytest.fixture(scope="module")
def raw():
    cfg = TrustConfig(n_scan=101)
    return fg._analyse(_evaluate, X, range(6), SPACE.lower, SPACE.upper, SCN, len(OUTS), cfg)


def test_batched_analysis_equals_the_trust_report_per_scenario(raw):
    cfg = TrustConfig(n_scan=101)
    for s in range(SCN):
        rep = trust_report(
            lambda th, s=s: _toy(th, float(s)), X, SPACE.lower, SPACE.upper, CUL_ORDER, OUTS, cfg
        )
        for i, p in enumerate(CUL_ORDER):
            for e, o in enumerate(OUTS):
                r = rep["params"][p]["outputs"][o]
                tag = (s, p, o)
                assert raw["class"][s, e, i] == r["class"], tag
                assert raw["level"][s, e, i] == r["level"], tag
                assert raw["n_jumps"][s, e, i] == r["n_jumps"], tag
                np.testing.assert_allclose(
                    raw["ad"][s, e, i], r["ad"], rtol=1e-12, atol=1e-12, err_msg=str(tag)
                )
                np.testing.assert_allclose(
                    raw["fd_small"][s, e, i], r["fd_small"], rtol=1e-9, atol=1e-9, err_msg=str(tag)
                )
                np.testing.assert_allclose(
                    raw["fd_large"][s, e, i], r["fd_large"], rtol=1e-11, atol=1e-11, err_msg=str(tag)
                )
        np.testing.assert_allclose(raw["y0"][s], rep["y0"], rtol=1e-13)


def test_the_toy_has_the_classes_the_labels_rest_on(raw):
    i = {p: j for j, p in enumerate(CUL_ORDER)}
    o = {n: j for j, n in enumerate(OUTS)}
    cls, lvl = raw["class"], raw["level"]
    assert (cls[:, o[SMOOTH], i["G2"]] == "smooth").all() and (lvl[:, o[SMOOTH], i["G2"]] == 3).all()
    assert (cls[:, o[SMOOTH], i["G3"]] == "smooth").all()
    assert (cls[:, o[SMOOTH], i["P1"]] == "inert").all()
    assert (cls[:, o[STAIRS], i["P1"]] == "step").all()
    assert (cls[:, o[G2_JUMP], i["G2"]] == "jumpy").all()
    assert (cls[:, o[P5_JUMP], i["P5"]] == "jumpy").all()
    assert (cls[:, o[CONST]] == "inert").all()
    # the staircase: AD says 0, the finite difference over the large step does not
    assert raw["ad"][0, o[STAIRS], i["P1"]] == 0.0 and raw["fd_large"][0, o[STAIRS], i["P1"]] != 0.0


def test_labels_follow_the_calibration_plan(raw):
    names = [f"s{s}" for s in range(SCN)]
    lab, trust, reasons = fg._labels(raw, CUL_ORDER, OUTS, names)
    i = {p: j for j, p in enumerate(CUL_ORDER)}
    o = {n: j for j, n in enumerate(OUTS)}
    for p in fg.EXPERIMENTAL:  # whatever the check finds
        assert (lab[:, :, i[p]] == fg.LABEL_EXPERIMENTAL).all() and trust[p] == fg.LABEL_EXPERIMENTAL
    assert (lab[:, o[SMOOTH], i["G2"]] == fg.LABEL_VALIDATED).all()
    assert (lab[:, o[G2_JUMP], i["G2"]] == fg.LABEL_FALLBACK).all()
    assert (lab[:, o[CONST], i["G3"]] == fg.LABEL_VALIDATED).all()  # inert: a zero derivative that is right
    assert trust["G2"] == fg.LABEL_FALLBACK  # the weakest of its outputs and scenarios
    assert trust["G3"] == fg.LABEL_VALIDATED  # smooth, inert or flat-and-right on every output
    assert any(G2_JUMP in r for r in reasons["G2"]) and reasons["G3"] == []
    # the same labels from the plan of the whole report, as the calibration builds it
    assert set(trust.values()) == {fg.LABEL_EXPERIMENTAL, fg.LABEL_FALLBACK, fg.LABEL_VALIDATED}


def test_tables_and_summary(raw):
    names = [f"s{s}" for s in range(SCN)]
    lab, trust, _ = fg._labels(raw, CUL_ORDER, OUTS, names)
    rows = [{"scenario": s, "year": 1980 + s, "sowing_shift": 0} for s in range(SCN)]
    table = fg._tables(raw, lab, list(CUL_ORDER), OUTS, X, range(6), rows, np.ones(SCN, dtype=bool))
    assert len(table) == SCN * len(OUTS) * 6
    assert list(table.columns[:5]) == ["scenario", "year", "sowing_shift", "matured", "output"]
    g = table[(table["scenario"] == 1) & (table["output"] == SMOOTH) & (table["param"] == "G2")].iloc[0]
    k = 1.0 + 0.3
    assert g["value"] == pytest.approx(k * 6.5**2 + 9.0 * np.sin(4.6))
    assert g["derivative"] == pytest.approx(2 * k * 6.5 / 100.0, rel=1e-12) and g["source"] == "AD"
    width = SPACE.width[3]
    assert g["sensitivity"] == pytest.approx(g["derivative"] * width / g["value"])
    assert g["elasticity"] == pytest.approx(g["derivative"] * 650.0 / g["value"])
    assert g["trust"] == fg.LABEL_VALIDATED and g["class"] == "smooth" and g["level"] == 3
    assert g["unit"] == "kg ha-1 per (kernel plant-1)"
    # a pair that falls back reports the finite difference as its derivative
    f = table[(table["scenario"] == 0) & (table["output"] == G2_JUMP) & (table["param"] == "G2")].iloc[0]
    assert f["trust"] == fg.LABEL_FALLBACK and f["source"] == "central difference"
    assert f["derivative"] == f["fd"] and f["derivative"] != f["ad"]
    # an experimental pair keeps the AD value, the finite difference next to it
    e = table[(table["scenario"] == 0) & (table["output"] == STAIRS) & (table["param"] == "P1")].iloc[0]
    assert e["trust"] == fg.LABEL_EXPERIMENTAL and e["derivative"] == e["ad"] == 0.0 and e["fd"] != 0.0
    # the batch summary
    sens = fg.Sensitivity("T_t01", dict(zip(CUL_ORDER, X.tolist(), strict=True)), table, trust)
    sm = sens.summary
    assert list(sm.index.names) == ["output", "param"] and len(sm) == len(OUTS) * 6
    r = sm.loc[(SMOOTH, "G2")]
    assert r["scenarios"] == SCN and r["validated"] == SCN and r["trust"] == fg.LABEL_VALIDATED
    assert r["min derivative"] < r["mean derivative"] < r["max derivative"]
    assert sm.loc[(G2_JUMP, "G2"), "trust"] == fg.LABEL_FALLBACK and sm.loc[(G2_JUMP, "G2"), "validated"] == 0
    assert sm.loc[(SMOOTH, "P1"), "trust"] == fg.LABEL_EXPERIMENTAL
    text = str(sens)
    assert "experimental" in text and "validated gradient" in text and "Derivatives through phenology" in text
    assert f"over {SCN} scenarios" in text


def _gradient(raw):
    one = {
        k: (v[:1] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == SCN else v)
        for k, v in raw.items()
    }
    lab, trust, reasons = fg._labels(one, CUL_ORDER, OUTS, ["season"])
    table = fg._tables(one, lab, list(CUL_ORDER), OUTS, X, range(6), [{}], np.ones(1, dtype=bool))
    return fg.Gradient("T_t01", dict(zip(CUL_ORDER, X.tolist(), strict=True)), table, trust, reasons)


def test_gradient_text(raw):
    text = str(_gradient(raw))
    assert text.splitlines()[0].startswith("d output / d coefficient") and "falls back" in text
    assert "trust per coefficient: P1 experimental" in text and "Derivatives through phenology" in text


@pytest.mark.allow_skip(reason="the figures need matplotlib (pip install 'agrijax[plot]')")
def test_plots_draw_one_panel_per_output(raw):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = _gradient(raw)
    assert len(g.plot().axes) == len(OUTS)
    rows = [{"scenario": s, "year": 1980 + s, "sowing_shift": 0} for s in range(SCN)]
    lab, trust, _ = fg._labels(raw, CUL_ORDER, OUTS, [f"s{s}" for s in range(SCN)])
    table = fg._tables(raw, lab, list(CUL_ORDER), OUTS, X, range(6), rows, np.ones(SCN, dtype=bool))
    assert len(fg.Sensitivity("T_t01", g.point, table, trust).plot().axes) == len(OUTS)
    plt.close("all")


def test_a_step_that_leaves_the_range_is_one_sided_and_the_range_widens_around_the_point():
    seen = []

    def evaluate(theta, tan, scn):
        seen.append(np.asarray(theta))
        th = jnp.asarray(theta)
        y = jnp.stack([3.0 * th[:, 1] + th[:, 3] ** 2 / 1e3], axis=1)
        dy = jnp.stack([3.0 * jnp.asarray(tan)[:, 1] + 2 * th[:, 3] * jnp.asarray(tan)[:, 3] / 1e3], axis=1)
        return np.asarray(y), np.asarray(dy)

    cfg = TrustConfig(n_scan=11)
    x = X.copy()
    x[1] = 0.0  # P2 at its lower bound
    x[2] = 1000.0  # P5 above its MAXIMA (999): the published value outside the box
    out = fg._analyse(evaluate, x, [1, 2, 3], SPACE.lower, SPACE.upper, 1, 1, cfg)
    pts = np.concatenate(seen)
    assert pts[:, 1].min() >= SPACE.lower[1] and pts[:, 1].max() <= SPACE.upper[1]
    assert pts[:, 2].min() >= SPACE.lower[2] and pts[:, 2].max() <= 1000.0  # the box now holds the point
    # linear in P2: the one-sided differences are exact
    np.testing.assert_allclose(out["ad"][0, 0, 0], 3.0)
    np.testing.assert_allclose(out["fd_small"][0, 0, 0], 3.0, rtol=1e-9)
    np.testing.assert_allclose(out["fd_large"][0, 0, 0], 3.0, rtol=1e-9)
    assert out["class"][0, 0, 0] == "smooth" and out["level"][0, 0, 0] == 3
    # P5 does not enter this function: inert; G2 enters quadratically: smooth
    assert out["class"][0, 0, 1] == "inert" and out["class"][0, 0, 2] == "smooth"
    assert out["width"][1] == pytest.approx(1000.0 - SPACE.lower[2])


def test_scan_summary_and_fd_agreement_equal_the_originals():
    def f(th):
        return jnp.stack([th[0] ** 2, jnp.floor(th[1] * 4.0), 5.0 + 0.0 * th[0]])

    x = np.array([1.0, 0.43])
    cfg = TrustConfig(n_scan=21)
    sc = line_scan(f, x, 1, 0.2, 0.9, cfg, width=2.0)
    e = jnp.array([0.0, 1.0])
    grid = np.linspace(0.2, 0.9, 21)
    pts = np.repeat(x[None], 21, axis=0)
    pts[:, 1] = grid
    y, g = jax.vmap(lambda t: jax.jvp(f, (t,), (e,)))(jnp.asarray(pts))
    again = scan_summary(grid, np.asarray(y), np.asarray(g), 2.0, cfg)
    assert sc.keys() == again.keys()
    for k in sc:
        np.testing.assert_array_equal(sc[k], again[k], err_msg=k)
    fd = fd_check(f, x, np.array([2.0, 2.0]), cfg)
    for k in (0, 1):
        rel, agree = fd_agreement(fd["ad"], fd[f"fd{k}"], fd["y0"], np.array([2.0, 2.0]), k, cfg)
        np.testing.assert_array_equal(rel, fd[f"rel_err{k}"])
        # level 2 passes only where the small-step differences also agree with each other; level 3
        # needs every large-step secant, of which fd1 is one
        assert np.all(~fd[f"agree{k}"] | agree)
    np.testing.assert_array_equal(fd["agree0"], fd["status0"] == "pass")


def test_outputs_params_and_point_parsing():
    assert fg._parse_outputs("hwam") == ["HWAM"]
    assert fg._parse_outputs(["cwam", "lai@60", "Gwad@75", "H#AM"]) == ["CWAM", "LAI@60", "GWAD@75", "H#AM"]
    for bad in ([], ["HWAM", "hwam"], ["YIELD"], ["LAI"], ["LAI@x"], ["ADAT"]):
        with pytest.raises(ValueError):
            fg._parse_outputs(bad)
    assert fg._output_unit("HWAM") == "kg ha-1" and fg._output_unit("LAI@60") == "m2 m-2"
    assert fg._parse_params(["G3", "G2"], CUL_ORDER) == [4, 3] and fg._parse_params("PHINT", CUL_ORDER) == [5]
    for bad in ([], ["G2", "G2"], ["RUE"]):
        with pytest.raises(ValueError):
            fg._parse_params(bad, CUL_ORDER)
    pub = dict(zip(CUL_ORDER, X.tolist(), strict=True))
    np.testing.assert_array_equal(fg._point(pub, None), X)
    x = fg._point(pub, {"G2": 700.0})
    assert x[3] == 700.0 and x[4] == 9.0
    with pytest.raises(ValueError, match="unknown"):
        fg._point(pub, {"RUE": 4.0})
    with pytest.raises(ValueError, match="one value"):
        fg._point(pub, {"G2": [700.0, 800.0]})


def test_the_evaluator_is_kept_on_its_owner_for_as_long_as_it_lives():
    class Owner:
        pass

    calls = []
    a, b = Owner(), Owner()
    build = lambda: calls.append(1) or object()  # noqa: E731
    first = fg._cached(a, "k", build)
    assert fg._cached(a, "k", build) is first and len(calls) == 1
    assert fg._cached(a, "other", build) is not first and len(calls) == 2
    assert fg._cached(b, "k", build) is not first and len(calls) == 3
    n = len(fg._CACHE)
    del a
    gc.collect()
    assert len(fg._CACHE) == n - 1  # the dead owner's evaluators are released


def test_public_methods_delegate_with_the_module_defaults():
    for fn in (ajd.Experiment.gradient, ajd.Scenarios.sensitivity):
        sig = inspect.signature(fn).parameters
        assert sig["scan_points"].default == fg.SCAN_POINTS
        assert (
            tuple(sig["outputs"].default) == fg.DEFAULT_OUTPUTS and tuple(sig["params"].default) == CUL_ORDER
        )
        assert "experimental" in (fn.__doc__ or "") and "phenology" in (fn.__doc__ or "")
    assert fg.EXPERIMENTAL == ("P1", "P2", "P5", "PHINT")
    from agrijax.calib.ceres import CERES_DERIVATIVE_FREE

    assert fg.EXPERIMENTAL == CERES_DERIVATIVE_FREE
    assert "experimental" in (fg.gradient.__doc__ or "") and "experimental" in (fg.sensitivity.__doc__ or "")
    assert isinstance(fg.OUTPUTS["HWAM"], tuple) and set(fg.SERIES) == {"LAI", "CWAD", "GWAD"}


def test_gradients_refuse_execution_options_and_take_the_swap():
    """Gradients run in float64 on JAX's default device only: inside a float32 or device block they
    refuse (never silently ignore the options); ``exp.gradient`` takes ``soil_evaporation`` as
    ``exp.run`` does."""
    import agrijax as aj

    x64 = bool(jax.config.jax_enable_x64)
    with aj.options(precision="float32"):
        with pytest.raises(ValueError, match=r"exp\.gradient"):
            fg.gradient(object(), 4)
        with pytest.raises(ValueError, match=r"scen\.sensitivity"):
            fg.sensitivity(object())
    assert bool(jax.config.jax_enable_x64) is x64
    assert inspect.signature(ajd.Experiment.gradient).parameters["soil_evaporation"].default is None
    assert inspect.signature(fg.gradient).parameters["soil_evaporation"].default is None
