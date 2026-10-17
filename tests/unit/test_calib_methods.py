"""The method plan (:mod:`agrijax.calib.methods`) on toy problems, no data.

The toy loss has two treatments and three parameters on ``[0, 10]``: ``s`` smooth on both,
``k`` a staircase (``floor``) on both, ``q`` a staircase on the first treatment and smooth on the
second (trusted on one treatment of two). The diagnosis runs in float64 (the tests that need it carry
the float64 marker); the refusals without float64 and the label-only plan run in both precisions.
"""

from __future__ import annotations

import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib import (
    MethodPlan,
    ParamSpace,
    ParamSpec,
    TrustConfig,
    gradient_plan,
    plan_methods,
)
from agrijax.calib.methods import CalibrationWarning
from agrijax.calib.objective import Group, Objective, Target

NAMES = ("s", "k", "q")
SPACE = ParamSpace([ParamSpec(f"toy.{n}", 0.0, 10.0, "-", "toy bounds", label=n) for n in NAMES])
THETA0 = np.asarray([6.0, 7.5, 6.5])
TREATMENTS = ("t1", "t2")


def toy(theta):
    s, k, q = theta[0], theta[1], theta[2]
    l1 = (s - 3.0) ** 2 + (jnp.floor(k) - 5.0) ** 2 + (jnp.floor(q) - 4.0) ** 2
    l2 = (s - 3.0) ** 2 + 0.5 * (jnp.floor(k) - 5.0) ** 2 + (q - 4.0) ** 2
    return jnp.stack([l1, l2])


def needs_x64(fn):
    """The trust diagnosis runs in float64 only (the CI float32 pass skips these)."""
    fn = pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the diagnosis runs in float64 only")(fn)
    return pytest.mark.allow_skip(reason="the trust diagnosis is float64-only (CI float32 pass)")(fn)


def plan_of(**kw):
    kw.setdefault("derivative_free", ())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        return plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS, **kw)


# ------------------------------------------------------------------ the split
@needs_x64
def test_smooth_gradient_staircase_derivative_free_and_pairs_as_gradient_plan():
    plan = plan_of()
    assert plan.role == {"s": "gradient", "k": "derivative_free", "q": "derivative_free"}
    assert set(plan.source.values()) == {"diagnosis"}
    assert plan.diagnosed == plan.role
    assert plan.gradient_params == ("s",) and plan.derivative_free_params == ("k", "q")
    ref = gradient_plan(plan.report)
    assert plan.pairs is not None
    np.testing.assert_array_equal(plan.pairs.use_ad, ref.use_ad)
    assert plan.pairs.method == ref.method
    # q: trusted on the second treatment only -> derivative-free here, "hybrid" in the pair plan
    assert plan.pairs.method["q"] == "hybrid"
    row = {r["param"]: r for r in plan.table()}
    assert row["q"]["ad_treatments"] == ["t2"] and row["k"]["ad_treatments"] == []
    assert row["s"]["worst_class"] == "smooth" and row["s"]["worst_level"] == 3
    assert row["k"]["worst_class"] in ("step", "jumpy")
    assert "1/2" in str(plan) and "diagnosis:" in str(plan)
    # GradientPlan counts a parameter trusted on some treatments; MethodPlan does not
    assert "q" in plan.pairs.gradient_params and "q" not in plan.gradient_params
    assert plan.calls["jvp"] == 3 * 201 and plan.calls["jacobian"] == 1
    assert plan.calls["forward"] == 2 + 2 * 3 * 6 and plan.calls["wall_s"] > 0


@needs_x64
def test_prior_labels_override_the_role_and_keep_the_diagnosed_class():
    plan = plan_of(derivative_free=("s",))
    assert plan.role["s"] == "derivative_free" and plan.source["s"] == "prior"
    assert plan.diagnosed["s"] == "gradient"
    assert "derivative-free by label" in plan.reasons["s"]


@needs_x64
def test_allow_gradient_releases_the_default_prior_and_the_diagnosis_decides():
    from agrijax.calib.ceres import ceres_space

    sp = ceres_space(["P1", "G2"])

    def f(theta):  # P1 a staircase, G2 smooth
        return jnp.stack([(jnp.floor(theta[0] / 10.0) - 20.0) ** 2 + (theta[1] / 100.0 - 6.0) ** 2])

    x0 = np.asarray([255.0, 700.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        default = plan_methods(f, sp, x0, treatments=["a"])
        released = plan_methods(f, sp, x0, treatments=["a"], allow_gradient=["P1"])
        diag_only = plan_methods(f, sp, x0, treatments=["a"], derivative_free=())
    assert default.source["P1"] == "prior" and default.role["P1"] == "derivative_free"
    assert "derivative-free by default" in default.reasons["P1"]
    assert default.role["G2"] == "gradient" and default.source["G2"] == "diagnosis"
    assert released.source["P1"] == "diagnosis" and released.role["P1"] == "derivative_free"
    assert "released from the default by allow_gradient" in released.reasons["P1"]
    assert diag_only.source["P1"] == "diagnosis" and diag_only.diagnosed == default.diagnosed
    assert default.config["derivative_free"] == ["P1"] and released.config["derivative_free"] == []


@needs_x64
def test_default_prior_matches_specs_not_names():
    from agrijax.calib.ceres import CERES_SPECS

    fake = ParamSpace([ParamSpec("crop.p1", 1.0, 2.0, "-", "toy", label="P1")])
    real = ParamSpace([CERES_SPECS["P1"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        p_fake = plan_methods(lambda t: jnp.stack([t[0] ** 2]), fake, [1.5], treatments=["a"])
        p_real = plan_methods(lambda t: jnp.stack([t[0] ** 2]), real, [200.0], treatments=["a"])
    assert p_fake.source["P1"] == "diagnosis" and p_real.source["P1"] == "prior"


def test_diagnose_false_uses_the_labels_alone_without_derivatives():
    plan = plan_of(derivative_free=("k",), diagnose=False)
    assert plan.role == {"s": "gradient", "k": "derivative_free", "q": "gradient"}
    assert plan.source == {"s": "assumed", "k": "prior", "q": "assumed"}
    assert set(plan.diagnosed.values()) == {"not_run"}
    assert plan.report is None and plan.pairs is None
    assert plan.calls["jvp"] == 0 and plan.calls["jacobian"] == 0 and plan.calls["forward"] <= 1
    rows = plan.table()
    assert all(r["worst_class"] is None and r["worst_level"] is None for r in rows)
    assert "not run" in str(plan)


# ------------------------------------------------------------------ problem kinds, records
def _objective():
    def sim(theta):
        y = jnp.stack([theta[0] + jnp.floor(theta[1]), theta[0] + theta[2]])
        return {"y": y[None, :]}

    g = Group("g", sim, (Target("y", "y", "final"),), {"y": np.asarray([8.0, 7.0])}, ("a", "b"))
    return Objective((g,))


@needs_x64
def test_objective_and_its_plain_function_give_the_same_plan():
    obj = _objective()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CalibrationWarning)
        a = plan_methods(obj, SPACE, THETA0, derivative_free=())
        b = plan_methods(obj.treatment_losses, SPACE, THETA0, treatments=["g/a", "g/b"], derivative_free=())
    assert a.treatments == ("g/a", "g/b")
    assert a.role == b.role and a.diagnosed == b.diagnosed
    np.testing.assert_array_equal(a.pairs.use_ad, b.pairs.use_ad)


@needs_x64
def test_round_trip_through_json():
    plan = plan_of()
    d = json.loads(json.dumps(plan.to_dict()))
    back = MethodPlan.from_dict(d)
    assert json.loads(json.dumps(back.to_dict())) == d
    assert back.role == plan.role and back.table() == plan.table() and str(back) == str(plan)
    np.testing.assert_array_equal(back.pairs.use_ad, plan.pairs.use_ad)
    with pytest.raises(ValueError, match="not a MethodPlan"):
        MethodPlan.from_dict({"params": ["s"]})
    cols = set(plan.table()[0])
    assert cols == {
        "param",
        "role",
        "source",
        "diagnosed",
        "inert",
        "ad_treatments",
        "worst_class",
        "worst_level",
        "reasons",
    }


@needs_x64
def test_repr_is_short_and_names_the_split():
    plan = plan_of()
    r = repr(plan)
    assert len(r) < 300, len(r)
    assert r.startswith("MethodPlan(params=('s', 'k', 'q')") and "diagnosed=True" in r
    assert "scan_y" not in r and str(plan) != r


def test_repr_of_a_label_only_plan():
    plan = plan_of(diagnose=False, derivative_free=("k",))
    assert repr(plan) == (
        "MethodPlan(params=('s', 'k', 'q'), gradient=('s', 'q'), derivative_free=('k',), treatments=2, "
        "diagnosed=False)"
    )


@needs_x64
def test_inert_parameter_warns_and_theta0_on_a_bound_warns_and_still_plans():
    sp = ParamSpace([*SPACE.specs, ParamSpec("toy.w", 0.0, 1.0, "-", "toy", label="w")])
    with pytest.warns(CalibrationWarning, match=r"inert on every treatment.*\['w'\]"):
        plan = plan_methods(
            lambda t: toy(t[:3]), sp, [*THETA0, 0.5], treatments=TREATMENTS, derivative_free=()
        )
    assert plan.inert["w"] and plan.role["w"] == "gradient"
    with pytest.warns(CalibrationWarning, match=r"outside or on the bounds for \['s'\]"):
        out = plan_methods(toy, SPACE, [10.5, 7.5, 6.5], treatments=TREATMENTS, derivative_free=())
    assert set(out.role) == set(NAMES)


# ------------------------------------------------------------------ refusals
def test_refusals():
    with pytest.raises(TypeError, match="not a string"):
        plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS, derivative_free="k", diagnose=False)
    with pytest.raises(TypeError, match="not a string"):
        plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS, allow_gradient="k", diagnose=False)
    with pytest.raises(ValueError, match="not in the space"):
        plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS, derivative_free=["x"], diagnose=False)
    with pytest.raises(ValueError, match="not in the space"):
        plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS, allow_gradient=["x"], diagnose=False)
    with pytest.raises(ValueError, match="both name"):
        plan_methods(
            toy,
            SPACE,
            THETA0,
            treatments=TREATMENTS,
            derivative_free=["k"],
            allow_gradient=["k"],
            diagnose=False,
        )
    with pytest.raises(ValueError, match="treatments="):
        plan_methods(toy, SPACE, THETA0, diagnose=False)
    with pytest.raises(ValueError, match="names for a loss of shape"):
        plan_methods(toy, SPACE, THETA0, treatments=["t1"], diagnose=False)
    with pytest.raises(ValueError, match="shape"):
        plan_methods(toy, SPACE, THETA0[:2], treatments=TREATMENTS, diagnose=False)
    with pytest.raises(ValueError, match="non-finite"):
        plan_methods(toy, SPACE, [np.nan, 7.5, 6.5], treatments=TREATMENTS, diagnose=False)
    with pytest.raises(ValueError, match=r"not finite on the treatments \['t1'\]"):
        plan_methods(
            lambda t: jnp.stack([jnp.log(t[0] - 7.0), t[1]]),
            SPACE,
            THETA0,
            treatments=TREATMENTS,
            diagnose=False,
        )
    with pytest.raises(ValueError, match="space and theta0"):
        plan_methods(toy, treatments=TREATMENTS, diagnose=False)
    with pytest.raises(TypeError, match="problem must be"):
        plan_methods(42, SPACE, THETA0, diagnose=False)
    with pytest.raises(TypeError, match="TrustConfig"):
        plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS, trust={"n_scan": 3})


@needs_x64
def test_non_finite_loss_at_theta0_is_refused_before_the_diagnosis(monkeypatch):
    import agrijax.calib.methods as methods

    def no_diagnosis(*a, **k):
        raise AssertionError("the trust diagnosis ran before the input check")

    monkeypatch.setattr(methods, "trust_report", no_diagnosis)
    with pytest.raises(ValueError, match=r"not finite on the treatments \['t1'\]"):
        plan_methods(lambda t: jnp.stack([jnp.log(t[0] - 7.0), t[1]]), SPACE, THETA0, treatments=TREATMENTS)


def test_a_cultivar_problem_is_refused_with_the_plain_function_form():
    """A cultivar problem is planned through its per-treatment loss function (``jax_loss``)."""
    from . import test_calib_workflow as tw

    pb = tw.make_problem()
    with pytest.raises(TypeError, match=r"function theta -> \[B\] with treatments="):
        plan_methods(pb, diagnose=False)
    with pytest.raises(TypeError, match="jax_loss"):
        plan_methods(pb, SPACE, THETA0, treatments=[t.name for t in pb.treatments], diagnose=False)


def test_diagnosis_without_float64_raises_and_labels_only_still_plans():
    with jax.enable_x64(False):
        with pytest.raises(RuntimeError, match="64-bit"):
            plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS)
        plan = plan_methods(toy, SPACE, THETA0, treatments=TREATMENTS, derivative_free=(), diagnose=False)
    assert plan.config["x64"] is False and set(plan.source.values()) == {"assumed"}


def test_trust_config_is_passed_through():
    plan = plan_of(diagnose=False, trust=TrustConfig(n_scan=11))
    assert plan.config["trust"]["n_scan"] == 11
