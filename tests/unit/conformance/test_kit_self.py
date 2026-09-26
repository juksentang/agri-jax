"""Self-test of the conformance kit: a clean toy passes every check, and every check fails on a
fixture that breaks the rule it guards (a check that cannot fail checks nothing).

The toy is a crop-slot water bucket reading the nitrogen port (``_toy``); ``_toy_bad`` holds one
broken variant per rule. Everything is synthetic (NumPy generator), no data.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from agrijax.core import ports as ports_mod
from agrijax.core.ports import port
from agrijax.core.state import State, field
from agrijax.iface.crop import CropNIn, CropWaterIn
from agrijax.testing.conformance import (
    CHECKS,
    Balance,
    ConformanceCase,
    ConformanceError,
    GradSpec,
    Tolerance,
    check_balance,
    check_binding,
    check_coefficients,
    check_grad_fd,
    check_grad_finite,
    check_lint,
    check_name,
    check_precision,
    check_reads,
    check_registry,
    check_shapes_dims,
    check_slot_contract,
    check_transforms,
    check_units,
    check_writes,
    lint_scope,
    run_checks,
)
from agrijax.testing.conformance.pytest_plugin import run_check

from . import _toy, _toy_bad

N_CROP, N_DAYS, N_LAYER = 2, 3, 4
X64 = bool(jax.config.jax_enable_x64)
needs_x64 = pytest.mark.allow_skip(reason="float64 part of the kit: covered by the float64 pass")


def make(rng: np.random.Generator, dtype, variant: str, *, state_cls=_toy.BucketState, coefs=None):
    state = state_cls(
        water=np.asarray(rng.uniform(1.0, 5.0, N_CROP), dtype),
        drained=np.zeros(N_CROP, dtype),
        n_in=CropNIn(nstres=np.asarray(rng.uniform(0.5, 1.0, N_CROP), dtype)),
    )
    cset = (coefs or _toy.BucketCoefficients)().as_arrays(dtype)
    params = _toy.BucketParams(coefficients=cset)
    forcing = _toy.BucketForcing(rain=np.asarray(rng.uniform(0.0, 2.0, (N_DAYS, N_CROP)), dtype))
    return state, params, forcing


BALANCE = Balance(
    "water",
    "cm",
    storage=lambda s, p: jnp.sum(s.water),
    inflow=lambda b, a, p, f: jnp.sum(f.rain),
    outflow=lambda b, a, p, f: jnp.sum(a.drained),
)


def case(proc=_toy.bucket, **kw) -> ConformanceCase:
    base = dict(
        key=proc.key,
        make=make,
        process=proc,
        n_days=N_DAYS,
        ports={"n_in": "iface.crop_n.{slot}"},
        balances=(BALANCE,),
        coefficient_sets=("coefficients",),
        origin="kit self-test",
        forcing_fields=("rain",),
    )
    base.update(kw)
    return ConformanceCase(**base)


GOOD = case()


@pytest.mark.parametrize("check", CHECKS, ids=check_name)
def test_the_clean_toy_passes_every_check(check) -> None:
    check(GOOD)


def test_run_checks_reports_every_check() -> None:
    out = run_checks(GOOD)
    assert list(out) == [check_name(c) for c in CHECKS]
    assert all(v is None for v in out.values()), out


class NoUnitState(State):
    water: Array = field(description="bucket water without a unit", dims=("n_crop",))
    drained: Array = field(unit="cm d-1", description="water drained today", dims=("n_crop",))
    n_in: CropNIn = port(description="nitrogen stress")


class NoDimsState(State):
    water: Array = field(unit="cm", description="bucket water without dims")
    drained: Array = field(unit="cm d-1", description="water drained today", dims=("n_crop",))
    n_in: CropNIn = port(description="nitrogen stress")


def make_bad_shape(rng, dtype, variant):
    s, p, f = make(rng, dtype, variant)
    return s.replace(water=np.ones(N_CROP + 1, dtype)), p, f


def make_out_of_bounds(rng, dtype, variant):
    s, p, f = make(rng, dtype, variant, coefs=_toy_bad.BoundedCoefficients)
    cset = eqx.tree_at(lambda c: c.k, p.coefficients, jnp.asarray(0.9, dtype))
    return s, dataclasses.replace(p, coefficients=cset), f


def make_supply(rng: np.random.Generator, dtype, variant: str):
    """The water-supply toy: a crop water record (sw, eop read; trwup written), no forcing."""
    water = CropWaterIn(
        sw=jnp.asarray(rng.uniform(0.1, 0.3, N_LAYER), dtype),
        eop=jnp.asarray(rng.uniform(2.0, 6.0, N_CROP), dtype),
        trwup=jnp.asarray(rng.uniform(0.1, 0.5, N_CROP), dtype),
    )
    state = _toy_bad.SupplyState(uptake=jnp.zeros(N_CROP, dtype), water=water)
    params = _toy_bad.SupplyParams(coefficients=_toy.BucketCoefficients().as_arrays(dtype))
    return state, params, None


def supply_case(proc=_toy_bad.supply) -> ConformanceCase:
    return ConformanceCase(
        key=proc.key,
        make=make_supply,
        process=proc,
        n_days=N_DAYS,
        ports={"water": "iface.crop_water.{slot}"},
        no_balance="kit fixture: a potential-uptake estimate",
        coefficient_sets=("coefficients",),
        origin="kit self-test",
    )


BAD = [
    ("registry", check_registry, case(_toy_bad.no_source), "Source"),
    ("registry", check_registry, case(_toy_bad.wrong_provenance), "reference_only_conventions"),
    ("registry", check_registry, case(_toy_bad.wrong_provenance), "ref_build"),
    ("registry", check_registry, case(origin=""), "origin"),
    ("registry", check_registry, case(key="crop/toy_other@none:kit_self"), "registered as"),
    ("registry", check_registry, case(_toy_bad.orphan), "faithful sibling"),
    ("lint", check_lint, case(_toy_bad.python_if), "AJ001"),
    (
        "coefficients",
        check_coefficients,
        case(make=lambda r, d, v: make(r, d, v, coefs=_toy_bad.OtherRefCoefficients)),
        "differs from the key",
    ),
    ("coefficients", check_coefficients, case(make=make_out_of_bounds), "outside bounds"),
    ("shapes_dims", check_shapes_dims, case(make=make_bad_shape), "n_crop"),
    ("shapes_dims", check_shapes_dims, case(n_days=N_DAYS - 1), "time axis"),
    (
        "shapes_dims",
        check_shapes_dims,
        case(make=lambda r, d, v: make(r, d, v, state_cls=NoDimsState)),
        "without declared dims",
    ),
    ("units", check_units, case(make=lambda r, d, v: make(r, d, v, state_cls=NoUnitState)), "no unit"),
    ("units", check_units, case(ports={"n_in": "iface.nitrogen.{slot}"}), "not a port"),
    ("slot_contract", check_slot_contract, case(_toy_bad.writes_in_port), "'in' port"),
    ("slot_contract", check_slot_contract, case(ports={"n_in": "iface.root.{slot}"}), "record CropNIn"),
    ("slot_contract", check_slot_contract, case(ports={}), "binds no global path"),
    # the n_supply slot publishes P10: a module that binds it and only reads it breaks the contract
    ("slot_contract", check_slot_contract, case(slot_contract="n_supply"), "never writes it"),
    ("slot_contract", check_slot_contract, case(_toy_bad.undeclared_read), "neither reads nor writes"),
    ("writes", check_writes, case(_toy_bad.undeclared_write), "undeclared"),
    ("reads", check_reads, case(_toy_bad.undeclared_read), "n_in"),
    ("reads", check_reads, case(forcing_fields=("snow",)), "forcing.rain"),
    ("balance", check_balance, case(_toy_bad.leak), "does not close"),
    ("transforms", check_transforms, case(_toy_bad.impure), "eager and jit"),
    ("transforms", check_transforms, case(_toy_bad.batch_mixing), "vmap"),
    ("precision", check_precision, case(_toy_bad.nonfinite), "not finite"),
    ("grad_finite", check_grad_finite, case(_toy_bad.nan_grad), "non-finite gradient"),
    ("grad_finite", check_grad_finite, case(grad=GradSpec(wrt=())), "nothing to differentiate"),
    ("binding", check_binding, case(ports={}), "binds none"),
    ("binding", check_binding, supply_case(_toy_bad.self_read), "replay and coupled bindings differ"),
]


@pytest.mark.parametrize(
    ("what", "check", "bad", "match"), BAD, ids=[f"{b[0]}-{i}" for i, b in enumerate(BAD)]
)
def test_every_check_fails_on_its_bad_fixture(what, check, bad, match) -> None:
    with pytest.raises(ConformanceError, match=match):
        check(bad)


def test_every_check_has_a_bad_fixture() -> None:
    covered = {b[0] for b in BAD} | {"grad_fd", "precision"}  # the float64-only ones below
    assert covered == {check_name(c) for c in CHECKS}


@needs_x64
def test_grad_fd_fails_on_a_wrong_derivative() -> None:
    if not X64:
        pytest.skip("the finite-difference check runs in float64")
    with pytest.raises(ConformanceError, match="smooth point"):
        check_grad_fd(case(_toy_bad.wrong_grad))


@needs_x64
def test_grad_fd_reports_a_kink_next_to_the_point() -> None:
    if not X64:
        pytest.skip("the finite-difference check runs in float64")
    with pytest.raises(ConformanceError, match="kink"):
        check_grad_fd(case(_toy_bad.jump))
    # the same process with the parameter exempted (straight-through or threshold) passes
    check_grad_fd(case(_toy_bad.jump, grad=GradSpec(fd_exempt=(("coefficients.k", "fixture step"),))))


def test_the_clean_supply_toy_passes_the_port_checks() -> None:
    """Positive control of the inout replay: reading sw, eop and writing trwup is replayable."""
    for check in (check_slot_contract, check_binding, check_reads, check_writes):
        check(supply_case())
    # the self-reading variant declares its reads honestly: only the binding check catches it
    check_reads(supply_case(_toy_bad.self_read))
    check_slot_contract(supply_case(_toy_bad.self_read))


def test_binding_fails_when_the_binding_alters_a_record(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression of the core binding (the module sees a changed record) fails 'bound == alone'."""
    orig = ports_mod.Binding.gather

    def lossy(self, tree):
        view = orig(self, tree)
        halve = {p: jax.tree_util.tree_map(lambda x: x * 0.5, getattr(view, p)) for p, _ in self.ports}
        return dataclasses.replace(view, **halve)

    monkeypatch.setattr(ports_mod.Binding, "gather", lossy)
    with pytest.raises(ConformanceError, match="bound and unbound runs differ"):
        check_binding(case())


def test_binding_fails_when_an_unbound_port_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regression of the core binding (an unbound port is not refused) fails the third part."""
    orig = ports_mod.Binding.to_global
    monkeypatch.setattr(ports_mod.Binding, "to_global", lambda self, path, module_cls=None: orig(self, path))
    with pytest.raises(ConformanceError, match="instead of BindingError"):
        check_binding(case())


@needs_x64
def test_precision_catches_a_float32_cancellation() -> None:
    if not X64:
        pytest.skip("the float32 against float64 comparison needs float64")
    with pytest.raises(ConformanceError, match="float32 differs from float64"):
        check_precision(case(_toy_bad.cancel))


@needs_x64
def test_a_float32_run_that_does_not_trace_fails_the_check() -> None:
    """An upcast inside an inner loop breaks its carry: balance, precision and gradients report it."""
    if not X64:
        pytest.skip("float64 does not exist with x64 disabled")
    bad = case(_toy_bad.scan_upcast)
    for check in (check_balance, check_precision, check_grad_finite):
        with pytest.raises(ConformanceError, match="does not trace"):
            check(bad)


def test_a_type_error_that_is_not_a_carry_mismatch_is_not_reported_as_a_failed_trace() -> None:
    """Only a loop carry that changes type becomes a ConformanceError; any other TypeError is a
    fault of the process or the kit and propagates (an exemption cannot swallow it)."""
    bad = case(_toy_bad.type_error)
    for check in (check_balance, check_precision, check_grad_finite):
        with pytest.raises(TypeError, match="toy_type_error"):
            check(bad)


F32_ONLY = "float32 gap (fixture)"


@needs_x64
def test_a_float32_exemption_runs_the_float64_part_unexempted() -> None:
    """``exempt_float32_x64``: the float32 part must fail (xfail), the float64 part must pass."""
    if not X64:
        pytest.skip("float64 does not exist with x64 disabled")
    split = ("balance", "precision", "grad_finite")
    checks = (check_balance, check_precision, check_grad_finite)
    exempt = case(_toy_bad.scan_upcast, exempt_float32_x64=dict.fromkeys(split, F32_ONLY))
    for check in checks:
        with pytest.raises(pytest.xfail.Exception, match=r"exempt \(float32 part\).*does not trace"):
            run_check(exempt, check)
    out = run_checks(exempt, checks)
    for n in split:
        assert str(out[n]).startswith(f"float32 part: exempt: float32 under x64: {F32_ONLY}"), out[n]
    # a float64 failure is not masked by the float32 exemption: the float64 part fails first
    leaky = case(_toy_bad.leak, exempt_float32_x64={"balance": F32_ONLY})
    with pytest.raises(ConformanceError, match="float64"):
        run_check(leaky, check_balance)
    assert str(run_checks(leaky, (check_balance,))["balance"]).startswith("float64 part: ")
    nan64 = case(_toy_bad.nonfinite, exempt_float32_x64={"precision": F32_ONLY})
    with pytest.raises(ConformanceError, match="not finite in float64"):
        run_check(nan64, check_precision)
    nan_grad64 = case(_toy_bad.nan_grad, exempt_float32_x64={"grad_finite": F32_ONLY})
    with pytest.raises(ConformanceError, match=r"non-finite gradient \(float64"):
        run_check(nan_grad64, check_grad_finite)
    # stale: the float32 part passes
    stale = case(exempt_float32_x64={"balance": F32_ONLY})
    with pytest.raises(ConformanceError, match=r"\(float32 part\): passes although exempt"):
        run_check(stale, check_balance)
    assert "remove the exemption" in str(run_checks(stale, (check_balance,))["balance"])


@pytest.mark.allow_skip(reason="x64-disabled part of the kit: covered by the float32 pass")
def test_a_float32_exemption_is_inert_with_x64_disabled() -> None:
    """With x64 disabled float32 is the only dtype: the check runs unexempted."""
    if X64:
        pytest.skip("covered by the float32 pass (AGRI_JAX_X64=0)")
    ok = case(exempt_float32_x64={"balance": F32_ONLY})
    run_check(ok, check_balance)
    assert run_checks(ok, (check_balance,))["balance"] is None
    with pytest.raises(ConformanceError, match="does not close"):
        run_check(case(_toy_bad.leak, exempt_float32_x64={"balance": F32_ONLY}), check_balance)


@needs_x64
def test_precision_catches_an_upcast() -> None:
    if not X64:
        pytest.skip("float64 does not exist with x64 disabled")
    with pytest.raises(ConformanceError, match="float64 out"):
        check_precision(case(_toy_bad.upcast))


# ------------------------------------------------------------------ case validation and exemptions
def test_a_case_needs_the_reasons_for_what_it_leaves_out() -> None:
    with pytest.raises(ValueError, match="no_balance"):
        case(balances=())
    with pytest.raises(ValueError, match="no_grad"):
        case(grad=None)
    with pytest.raises(ValueError, match="no_slot_contract"):
        case(slot_contract=None)
    with pytest.raises(ValueError, match="transforms_why"):
        case(transforms_tol=(Tolerance(1e-12, 0.0), Tolerance(1e-5, 0.0)))
    with pytest.raises(ValueError, match="without a reason"):
        case(exempt_checks={"reads": " "})
    with pytest.raises(ValueError, match=r"only .* have a float32 part"):
        case(exempt_float32_x64={"reads": "no float32 part"})
    with pytest.raises(ValueError, match="both exempt_checks and exempt_float32_x64"):
        case(exempt_checks={"balance": "gap"}, exempt_float32_x64={"balance": "gap"})
    with pytest.raises(ValueError, match="float32 exemption of 'balance' without a reason"):
        case(exempt_float32_x64={"balance": " "})
    with pytest.raises(ValueError, match="invalid process key"):
        case(key="not a key")
    ok = case(balances=(), no_balance="toy", grad=None, no_grad="toy")
    assert ok.own_path == "crops.maize" and ok.port_map == {"n_in": "iface.crop_n.maize"}


def test_an_exemption_must_fail_and_a_stale_one_is_an_error() -> None:
    exempt = case(_toy_bad.leak, exempt_checks={"balance": "tracked gap (fixture)"})
    with pytest.raises(pytest.xfail.Exception, match="tracked gap"):
        run_check(exempt, check_balance)
    assert str(run_checks(exempt, (check_balance,))["balance"]).startswith("exempt: tracked gap")
    stale = case(exempt_checks={"balance": "was a gap"})
    with pytest.raises(ConformanceError, match="remove the exemption"):
        run_check(stale, check_balance)
    assert "remove the exemption" in str(run_checks(stale, (check_balance,))["balance"])


def test_lint_scope_is_the_module_and_its_same_package_imports() -> None:
    files = {p.name for p in lint_scope(_toy.bucket)}
    assert files == {"_toy.py"}
    bad = {p.name for p in lint_scope(_toy_bad.python_if)}
    assert bad == {"_toy_bad.py", "_toy.py"}  # _toy_bad imports _toy from its own package


def test_the_reads_check_perturbs_nan_and_values_deterministically() -> None:
    a = dataclasses.replace(GOOD, seed=0)
    check_reads(a)
    check_reads(a)  # same generator, same result
