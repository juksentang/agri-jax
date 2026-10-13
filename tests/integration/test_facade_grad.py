"""``exp.gradient`` and ``scen.sensitivity`` on DSSAT's UFGA8201 example (slow, needs the DSSAT example
data; no DSSAT run: the reference program is not used).

* the values at the evaluation point are the season's (``Experiment.run``, ``Scenarios.run``);
* for every derivative the check labels ``validated gradient``, the program's own (exact-mode)
  derivative ``ad_exact`` equals a forward finite difference of the public ``run`` calls (an
  independent program: the single-season day, not the batched forward-mode one) at a small step, and
  the reported (straight-through) derivative the central difference over +-2 % of the range (column
  ``fd``, itself the central difference of those runs); in a water-limited season the two
  derivatives differ (the straight-through path through the root length density truncation);
* the classes and trust levels equal ``agrijax.calib.trust.trust_report`` on the same model function
  (the check the calibration runs), the labels follow the calibration's gradient plan: P1, P2, P5 and
  PHINT ``experimental``;
* a one-scenario batch (the base year, no shift) equals ``exp.gradient``; a batch's coefficient label is
  the weakest over its scenarios; a second call with the same outputs does not compile again;
* the evaluation point moves with ``cultivar=``.
"""

from __future__ import annotations

import contextlib

import jax
import numpy as np
import pytest

from agrijax import facade_grad as fg
from agrijax.port.run_fortran import DSSAT_ENGINE

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the DSSAT example data (AGRI_JAX_DSSAT)"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the DSSAT day runs in float64"),
]

OUTPUTS = ["HWAM", "CWAM", "H#AM", "LAI@60"]
#: forward difference step, as a fraction of the coefficient's MINIMA-MAXIMA range
FORWARD_STEP = 1e-4
#: validated derivative against the forward difference (the difference has O(h) truncation error)
FD_RTOL = 2e-3
#: central difference step of the batch check (fraction of the range): in a water-limited season the
#: response is a staircase of root length density / soil water quanta on top of a smooth curve, and a
#: difference interval must fall between two quanta to measure the program's derivative (a 1e-4 forward
#: step straddles one in 1979 +14 d); the central difference has O(h^2) error, hence the tight tolerance
CENTRAL_STEP = 1e-6
CENTRAL_RTOL = 1e-5


@pytest.fixture(scope="module")
def engine():
    if not (DSSAT_ENGINE / "example_data" / "Maize" / "UFGA8201.MZX").is_file():
        pytest.skip(f"DSSAT example data not found under {DSSAT_ENGINE}")
    return DSSAT_ENGINE


@pytest.fixture(scope="module")
def exp(engine):
    import agrijax as aj

    return aj.dssat.experiment("UFGA8201", data_root=engine)


@pytest.fixture(scope="module")
def grad(exp):
    return exp.gradient(treatment=4, outputs=OUTPUTS)


def _row(table, output, param):
    r = table[(table["output"] == output) & (table["param"] == param)]
    assert len(r) == 1
    return r.iloc[0]


def _season_value(exp, output, cultivar=None):
    """The output of ``Experiment.run`` (the single-season program), at the cultivar."""
    s = exp.run(treatment=4, cultivar=cultivar)
    if output in s.summary:
        return float(s.summary[output])
    assert output == "LAI@60"
    t = fg._planting_index(exp.inputs(4)) + 60
    return float(s.daily["lai"].iloc[t])


def test_the_table_and_the_values_at_the_published_cultivar(exp, grad):
    from agrijax.calib.dssat_day import CUL_ORDER

    t = grad.table
    assert len(t) == len(OUTPUTS) * 6 and "matured" not in t.columns
    assert list(grad.trust) == list(CUL_ORDER) and set(grad.trust.values()) <= {
        fg.LABEL_VALIDATED,
        fg.LABEL_FALLBACK,
        fg.LABEL_EXPERIMENTAL,
    }
    assert grad.point == exp.inputs(4).published()
    for o in OUTPUTS:
        v = _season_value(exp, o) if o != "H#AM" else None
        if v is not None:
            assert _row(t, o, "G2")["value"] == pytest.approx(v, rel=1e-9), o
    assert grad.timing["rows"] == 6 * (17 + 201)
    text = str(grad)
    assert "experimental" in text and "Derivatives through phenology" in text
    print("\n" + text)
    print(t.drop(columns=["unit"]).to_string())


def test_phenology_coefficients_are_experimental_and_the_others_follow_the_check(grad):
    t = grad.table
    for p in fg.EXPERIMENTAL:
        assert (t[t["param"] == p]["trust"] == fg.LABEL_EXPERIMENTAL).all() and (
            grad.trust[p] == fg.LABEL_EXPERIMENTAL
        )
    for p in ("G2", "G3"):
        sub = t[t["param"] == p]
        assert set(sub["trust"]) <= {fg.LABEL_VALIDATED, fg.LABEL_FALLBACK}
        ok = (sub["class"] == "inert") | ((sub["class"] == "smooth") & (sub["level"] == 3))
        assert ((sub["trust"] == fg.LABEL_VALIDATED) == ok).all()
        assert (grad.trust[p] == fg.LABEL_VALIDATED) == bool(ok.all())
    fall = t[t["trust"] == fg.LABEL_FALLBACK]
    assert (fall["derivative"] == fall["fd"]).all() and (fall["source"] == "central difference").all()
    rest = t[t["trust"] != fg.LABEL_FALLBACK]
    assert (rest["derivative"] == rest["ad"]).all() and (rest["source"] == "AD").all()


def test_validated_derivatives_equal_forward_finite_differences_of_the_runs(exp, grad):
    from agrijax.calib.ceres import CERES_SPECS

    t = grad.table
    checked = 0
    for o in ("HWAM", "CWAM", "LAI@60"):
        for p in ("G2", "G3"):
            r = _row(t, o, p)
            if r["trust"] != fg.LABEL_VALIDATED or r["class"] == "inert":
                continue
            h = FORWARD_STEP * (CERES_SPECS[p].upper - CERES_SPECS[p].lower)
            x = grad.point[p]
            fwd = (_season_value(exp, o, {p: x + h}) - _season_value(exp, o, {p: x})) / h
            assert r["ad_exact"] == pytest.approx(fwd, rel=FD_RTOL), (o, p)
            assert r["derivative"] == pytest.approx(r["fd"], rel=0.05), (o, p)
            checked += 1
    assert checked >= 2, "no validated G2 / G3 derivative to check (see the printed table)"


def test_the_fd_column_is_the_central_difference_of_the_runs(exp, grad):
    from agrijax.calib.ceres import CERES_SPECS

    for o, p in (("HWAM", "G2"), ("CWAM", "G3"), ("HWAM", "PHINT"), ("CWAM", "P5")):
        h = 0.02 * (CERES_SPECS[p].upper - CERES_SPECS[p].lower)
        x = grad.point[p]
        fd = (_season_value(exp, o, {p: x + h}) - _season_value(exp, o, {p: x - h})) / (2 * h)
        assert _row(grad.table, o, p)["fd"] == pytest.approx(fd, rel=1e-6, abs=1e-9), (o, p)


def test_classes_and_levels_equal_the_trust_report_of_the_same_function(exp, grad):
    import jax.numpy as jnp

    from agrijax.calib import TrustConfig, trust_report
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER
    from agrijax.core.grad import bind_gradient_mode

    rows = fg._cached(
        exp, (("gradient", 4, None), tuple(OUTPUTS)), lambda: pytest.fail("evaluator not cached")
    )
    sim, g = rows.sim, rows.group
    fn = sim._sim_fn(g)
    inputs, tab = sim.inputs[g], sim.tables[g]
    x = np.asarray([grad.point[n] for n in CUL_ORDER])
    sub = [0, 3, 4]  # P1, G2, G3: a scan of each costs one compile

    def f(th):
        full = jnp.asarray(x).at[jnp.asarray(sub)].set(th)
        return fn(inputs, tab, full[None, :], jnp.zeros(1, jnp.int32))[0, : len(OUTPUTS)]

    sp = ceres_space(CUL_ORDER)
    names = [CUL_ORDER[i] for i in sub]
    rep = trust_report(
        bind_gradient_mode(f, "ste"),
        x[sub],
        sp.lower[sub],
        sp.upper[sub],
        names,
        OUTPUTS,
        TrustConfig(n_scan=201),
    )
    for p in names:
        for o in OUTPUTS:
            r, mine = rep["params"][p]["outputs"][o], _row(grad.table, o, p)
            assert (mine["class"], mine["level"], mine["jumps"]) == (r["class"], r["level"], r["n_jumps"]), (
                p,
                o,
            )
            assert mine["ad"] == pytest.approx(r["ad"], rel=1e-9, abs=1e-12), (p, o)
            assert mine["ad_exact"] == pytest.approx(r["ad_exact"], rel=1e-9, abs=1e-12), (p, o)
            assert mine["ad_unrounded"] == pytest.approx(r["ad_unrounded"], rel=1e-9, abs=1e-12), (p, o)
            assert mine["fd"] == pytest.approx(r["fd_large"], rel=1e-9, abs=1e-9), (p, o)


def test_a_second_call_does_not_compile_and_the_point_moves_with_the_cultivar(exp, grad):
    # (the same program: 618 rows pad to the 512-row chunks of the first call's 1236)
    again = exp.gradient(treatment=4, outputs=OUTPUTS, params=["G2", "G3", "PHINT"])
    assert again.timing["compile_s"] == 0.0 and len(again.table) == len(OUTPUTS) * 3
    # the same point: the derivative is the same number whatever the scan resolution
    a, b = _row(again.table, "HWAM", "G2"), _row(grad.table, "HWAM", "G2")
    assert a["ad"] == pytest.approx(b["ad"], rel=1e-12) and a["value"] == b["value"]
    moved = exp.gradient(treatment=4, outputs=["HWAM"], params=["G2"], cultivar={"G2": 700.0}, scan_points=21)
    assert moved.point["G2"] == 700.0 and moved.point["G3"] == grad.point["G3"]
    assert _row(moved.table, "HWAM", "G2")["value"] == pytest.approx(
        _season_value(exp, "HWAM", {"G2": 700.0}), rel=1e-9
    )
    with pytest.raises(ValueError, match="after the end of the season"):
        exp.gradient(treatment=4, outputs=["LAI@999"], scan_points=5)
    with pytest.raises(ValueError, match="unknown output"):
        exp.gradient(treatment=4, outputs=["ADAT"], scan_points=5)


def test_scenario_batch(exp, grad):
    scen = exp.scenarios(treatment=4, years=[1979, 1982], sowing_shift=[-14, 0, 14])
    sens = scen.sensitivity(outputs=["HWAM", "CWAM"], params=["G2", "G3", "PHINT"], scan_points=101)
    t = sens.table
    assert len(t) == 6 * 2 * 3 and set(t["scenario"]) == set(range(6)) and t["matured"].all()
    # the values are the batch run's
    runs = scen.run().table.set_index("scenario")
    for s in range(6):
        for o in ("HWAM", "CWAM"):
            v = _row(t[t["scenario"] == s], o, "G2")["value"]
            assert v == pytest.approx(runs.loc[s, o], rel=1e-9), (s, o)
    # validated derivatives equal forward finite differences of the batch run
    from agrijax.calib.ceres import CERES_SPECS

    checked = 0
    for p in ("G2", "G3"):
        h = CENTRAL_STEP * (CERES_SPECS[p].upper - CERES_SPECS[p].lower)
        x = scen.published[p]
        two = scen.run({p: [x - h, x + h]}).table
        for s in range(6):
            for o in ("HWAM", "CWAM"):
                r = _row(t[t["scenario"] == s], o, p)
                if r["trust"] != fg.LABEL_VALIDATED or r["class"] == "inert":
                    continue
                y = two[two["scenario"] == s].sort_values("sample")[o].to_numpy()
                # the program's own derivative (exact mode) against the small step; the reported
                # (straight-through) derivative against the large-step secant
                assert r["ad_exact"] == pytest.approx((y[1] - y[0]) / (2 * h), rel=CENTRAL_RTOL), (s, o, p)
                assert r["derivative"] == pytest.approx(r["fd"], rel=0.05), (s, o, p)
                checked += 1
    assert checked >= 4, "too few validated derivatives to check (see the printed summary)"
    # labels: PHINT experimental; a coefficient is validated over the batch only if every pair is
    assert sens.trust["PHINT"] == fg.LABEL_EXPERIMENTAL
    for p in ("G2", "G3"):
        sub = t[t["param"] == p]
        assert (sens.trust[p] == fg.LABEL_VALIDATED) == bool((sub["trust"] == fg.LABEL_VALIDATED).all())
    sm = sens.summary
    assert len(sm) == 2 * 3 and (sm["scenarios"] == 6).all()
    assert (sm.xs("PHINT", level="param")["trust"] == fg.LABEL_EXPERIMENTAL).all()
    print("\n" + str(sens))
    print(sm.to_string())
    # one scenario (the base year, no shift) is the single-season gradient
    one = exp.scenarios(treatment=4, years=[1982], sowing_shift=[0]).sensitivity(
        outputs=["HWAM", "CWAM"], params=["G2", "G3", "PHINT"], scan_points=101
    )
    single = exp.gradient(
        treatment=4, outputs=["HWAM", "CWAM"], params=["G2", "G3", "PHINT"], scan_points=101
    )
    for o in ("HWAM", "CWAM"):
        for p in ("G2", "G3", "PHINT"):
            a, b = _row(one.table, o, p), _row(single.table, o, p)
            assert a["value"] == pytest.approx(b["value"], rel=1e-9)
            assert a["ad"] == pytest.approx(b["ad"], rel=1e-9, abs=1e-12)
            assert (a["trust"], a["class"], a["level"]) == (b["trust"], b["class"], b["level"]), (o, p)
    assert one.trust == single.trust
    # the straight-through derivative (regression: the small step used to be compared with the
    # straight-through value itself and these pairs fell back at trust level 1). Level 2 is decided on the
    # exact path, three-valued. 1979 -14 d is water limited: the straight-through yield derivative of G2
    # carries the root length density truncation's path (MZ_ROOTS, a GradientConvention of
    # crop/ceres_maize.roots), about 2.8 % off the exact one; level 2 passes and the pair is validated
    r = _row(t[(t["year"] == 1979) & (t["sowing_shift"] == -14)], "HWAM", "G2")
    assert abs(r["ad"] / r["ad_exact"] - 1.0) > 0.01 and r["level2"] == "pass" and r["err_small"] < 1e-3
    assert abs(r["ste_offset"]) < 0.02  # diagnostic: the unrounded model's derivative
    assert r["trust"] == fg.LABEL_VALIDATED and r["derivative"] == r["ad"] and r["err_large_max"] < 0.05
    # 1982 -14 d: the 1e-4 difference straddles a quantum, level 2 is undecidable, level 3 validates
    s82m = t[(t["year"] == 1982) & (t["sowing_shift"] == -14)]
    for p in ("G2", "G3"):
        r = _row(s82m, "HWAM", p)
        assert abs(r["ad"] / r["ad_exact"] - 1.0) > 0.01, p
        assert r["level2"] == "undecidable (FD unreliable)" and r["err_small"] < 1e-3, p
        assert r["trust"] == fg.LABEL_VALIDATED and r["level"] == 3 and r["err_large_max"] < 0.05, p
    # no derivative error anywhere: the small-step differences never agree with each other against the
    # exact derivative
    assert (t[t["param"].isin(["G2", "G3"])]["level2"] != "fail").all()
    # 1979 at 0 days: the small-step differences straddle a quantum (they disagree with each other), level 2
    # is undecidable and level 3 decides
    s79 = t[(t["year"] == 1979) & (t["sowing_shift"] == 0)]
    for o in ("HWAM", "CWAM"):
        for p in ("G2", "G3"):
            r = _row(s79, o, p)
            assert r["level2"] == "undecidable (FD unreliable)" and r["fd_spread_small"] > 1e-3, (o, p)
            assert (r["trust"] == fg.LABEL_VALIDATED) == (r["level"] == 3 and r["class"] == "smooth"), (o, p)
    # not water limited (1982, as sown): no straight-through path reaches yield
    s82 = t[(t["year"] == 1982) & (t["sowing_shift"] == 0)]
    for p in ("G2", "G3"):
        r = _row(s82, "HWAM", p)
        assert r["ad"] == pytest.approx(r["ad_exact"], rel=1e-9), p
    print(
        t[t["param"].isin(["G2", "G3"])][
            [
                "year",
                "sowing_shift",
                "output",
                "param",
                "ad",
                "ad_exact",
                "fd",
                "err_small",
                "err_large",
                "class",
                "level",
                "trust",
            ]
        ].to_string()
    )
    # a scenario that is not the first, against the calibration's trust report of the same function
    _same_as_the_trust_report(scen, sens, 1, ["G2", "G3"], ["HWAM", "CWAM"])


def _same_as_the_trust_report(scen, sens, s, params, outputs):
    import jax.numpy as jnp

    from agrijax.calib import TrustConfig, trust_report
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER
    from agrijax.core.grad import bind_gradient_mode

    rows = fg._cached(scen, (("sensitivity",), tuple(outputs)), lambda: pytest.fail("evaluator not cached"))
    sim, g = rows.sim, rows.group
    fn = sim._sim_fn(g)
    inputs, tab, loc = sim.inputs[g], sim.tables[g], int(rows.local[s])
    x = np.asarray([scen.published[n] for n in CUL_ORDER])
    sub = [CUL_ORDER.index(p) for p in params]

    def f(th):
        full = jnp.asarray(x).at[jnp.asarray(sub)].set(th)
        return fn(inputs, tab, full[None, :], jnp.full((1,), loc, jnp.int32))[0, : len(outputs)]

    sp = ceres_space(CUL_ORDER)
    rep = trust_report(
        bind_gradient_mode(f, "ste"),
        x[sub],
        sp.lower[sub],
        sp.upper[sub],
        params,
        outputs,
        TrustConfig(n_scan=101),
    )
    t = sens.table[sens.table["scenario"] == s]
    for p in params:
        for o in outputs:
            r, mine = rep["params"][p]["outputs"][o], _row(t, o, p)
            assert (mine["class"], mine["level"], mine["jumps"]) == (r["class"], r["level"], r["n_jumps"]), (
                p,
                o,
            )
            assert mine["ad"] == pytest.approx(r["ad"], rel=1e-9, abs=1e-12), (p, o)
            assert mine["ad_exact"] == pytest.approx(r["ad_exact"], rel=1e-9, abs=1e-12), (p, o)
            assert mine["ad_unrounded"] == pytest.approx(r["ad_unrounded"], rel=1e-9, abs=1e-12), (p, o)
            assert mine["fd"] == pytest.approx(r["fd_large"], rel=1e-9, abs=1e-9), (p, o)


@contextlib.contextmanager
def _registered_sites_exact():
    """Every registered straight-through call site (``GradientConvention``) traced in the exact mode:
    its helper, in the site's module, bound to ``mode="exact"`` (``real4_store`` has the cast's
    derivative in every mode)."""
    import functools
    import importlib

    from agrijax.core.process import registry

    saved = []
    try:
        for p in list(registry.values()):
            for g in () if p.info is None else p.info.gradient_conventions:
                if g.helper == "real4_store":
                    continue
                mod = importlib.import_module(g.site.rsplit(".", 1)[0])
                saved.append((mod, g.helper, getattr(mod, g.helper)))
                setattr(mod, g.helper, functools.partial(getattr(mod, g.helper), mode="exact"))
        yield len(saved)
    finally:
        for mod, name, fn in reversed(saved):
            setattr(mod, name, fn)


def test_the_registered_sites_are_the_whole_straight_through_derivative(exp):
    """No unregistered straight-through path: with every registered site in the exact mode, the ste
    program's values and derivatives equal the exact program's bit for bit, on the nine weather-year x
    sowing scenarios and all six coefficients (and without that, the two differ: the sites matter)."""
    scen = exp.scenarios(treatment=4, years=[1979, 1982, 1985], sowing_shift=[-14, 0, 14])
    outs = ["HWAM", "CWAM", "H#AM"]
    entries = [fg._entries(outs, r) for r in scen.runs]
    x = fg._point(scen.published, None)
    n = len(scen.runs)
    th = np.repeat(x[None], 6 * n, axis=0)
    tan = np.tile(np.eye(6), (n, 1))
    scn = np.repeat(np.arange(n), 6)
    with _registered_sites_exact() as n_sites:
        assert n_sites >= 4
        rows = fg._Rows(scen.runs, entries)  # programs traced inside the context
        y_s, d_s = rows(th, tan, scn)
        y_x, d_x = rows.exact(th, tan, scn)
    assert np.array_equal(y_s, y_x)
    assert np.array_equal(d_s, d_x), float(np.max(np.abs(d_s - d_x)))
    _, d_ste = fg._Rows(scen.runs, entries)(th, tan, scn)
    assert not np.array_equal(d_ste, d_x)
