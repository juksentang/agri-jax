"""Self-test of the conformance kit: a clean toy passes every check, and every check fails on a
fixture that breaks the rule it guards (a check that cannot fail checks nothing).

The toy is a crop-slot water bucket reading the nitrogen port (``_toy``); ``_toy_bad`` holds one
broken variant per rule. Everything is synthetic (NumPy generator), no data.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.state import State, field
from agrijax.iface.crop import CropNIn
from agrijax.testing.conformance import (
    CHECKS,
    Balance,
    ConformanceCase,
    ConformanceError,
    GradSpec,
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

N_CROP, N_DAYS = 2, 3
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


BAD = [
    ("registry", check_registry, case(_toy_bad.no_source), "Source"),
    ("registry", check_registry, case(_toy_bad.wrong_provenance), "reference_only_conventions"),
    ("registry", check_registry, case(_toy_bad.wrong_provenance), "ref_build"),
    ("registry", check_registry, case(origin=""), "origin"),
    ("registry", check_registry, case(key="crop/toy_other@none:kit_self"), "registered as"),
    ("lint", check_lint, case(_toy_bad.python_if), "AJ001"),
    (
        "coefficients",
        check_coefficients,
        case(make=lambda r, d, v: make(r, d, v, coefs=_toy_bad.OtherRefCoefficients)),
        "differs from the key",
    ),
    ("shapes_dims", check_shapes_dims, case(make=make_bad_shape), "n_crop"),
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
    ("writes", check_writes, case(_toy_bad.undeclared_write), "undeclared"),
    ("reads", check_reads, case(_toy_bad.undeclared_read), "n_in"),
    ("reads", check_reads, case(forcing_fields=("snow",)), "forcing.rain"),
    ("balance", check_balance, case(_toy_bad.leak), "does not close"),
    ("transforms", check_transforms, case(_toy_bad.impure), "eager and jit"),
    ("precision", check_precision, case(_toy_bad.nonfinite), "not finite"),
    ("grad_finite", check_grad_finite, case(_toy_bad.nan_grad), "non-finite gradient"),
    ("grad_finite", check_grad_finite, case(grad=GradSpec(wrt=())), "nothing to differentiate"),
    ("binding", check_binding, case(ports={}), "binds none"),
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
    with pytest.raises(ValueError, match="without a reason"):
        case(exempt_checks={"reads": " "})
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
