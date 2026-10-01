"""The Richards physics/integrator split: protocol, registry, the day's dispatch and the
integrator conformance on the frozen problem (:mod:`agrijax.testing.conformance.integrators`).

* the problem's config holds only physics; each integrator has its own config, registered under a key
  with slot ``soil_water`` and a variant of the closed vocabulary; the day dispatches by config type;
* a new integrator plugs in by registration alone (a wrapper of the fixed steps gives the same day, bit
  for bit, through ``richards_day`` and the event day);
* every registered integrator has a conformance case and passes every check at its declared (measured)
  tolerances; the leaky fixture fails the balance check while its converged base passes;
* ``PROBLEM_VERSION`` pins the physics: the residual and fluxes of one sub-step on the frozen input.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.processes.soil_water import integrator as I
from agrijax.processes.soil_water.day import DayConfig, soil_water_day_kernel
from agrijax.processes.soil_water.fixed_cn import FixedCN, FixedStepping
from agrijax.processes.soil_water.problem import PROBLEM_VERSION, RichardsConfig, RichardsProblem
from agrijax.processes.soil_water.richards import RichardsParams, SoilWater, richards_day
from agrijax.processes.soil_water.richards_adaptive import AdaptiveStepping
from agrijax.testing.conformance import integrators as K
from agrijax.testing.conformance.case import ConformanceError

from . import test_infiltration_day as TD

X64 = bool(jax.config.read("jax_enable_x64"))


# ---------------------------------------------------------------------------
# layers and registry
# ---------------------------------------------------------------------------


def test_the_problem_config_holds_only_physics() -> None:
    assert {f.name for f in dataclasses.fields(RichardsConfig)} == {
        "sink_cutoff",
        "drain_cap",
        "evaporation_limit",
    }
    assert PROBLEM_VERSION == 1 and RichardsProblem.version == PROBLEM_VERSION


def test_registered_integrators_follow_the_protocol_and_the_vocabulary() -> None:
    assert set(I.INTEGRATORS) >= {
        "soil_water/richards_time@rzwqm2-4.6:faithful",
        "soil_water/richards_time@rzwqm2-4.6:alt_fixed_steps",
    }
    for key, info in I.INTEGRATORS.items():
        assert I.variant_problems(key) == [] and info.config_type.KEY == key and info.summary.strip()
        integ = info.factory(info.config_type())
        assert isinstance(integ, I.RichardsIntegrator)
        assert I.integrator_info(info.config_type()) is info and I.integrator_info(key) is info
    assert I.integrator_for(AdaptiveStepping()).capabilities.gradient == "custom"
    assert I.integrator_for(FixedStepping()).capabilities.gradient == "autodiff"
    assert I.integrator_for(FixedStepping(grad="implicit")).capabilities.gradient == "custom"
    assert I.integrator_for(AdaptiveStepping()).capabilities.convergence
    assert not I.integrator_for(FixedStepping()).capabilities.convergence


def _config_type(key: str) -> type:
    class Cfg(I.SteppingConfig):
        KEY: ClassVar[str] = key

    return Cfg


@pytest.mark.parametrize(
    ("key", "match"),
    [
        ("pet/richards_time@rzwqm2-4.6:alt_x", "slot"),
        ("soil_water/richards_time@rzwqm2-4.6:hybrid", "variant"),
        ("soil_water/other_time@rzwqm2-4.6:alt_x", "faithful"),
    ],
)
def test_registration_rejects_bad_keys(key: str, match: str) -> None:
    info = I.IntegratorInfo(key=key, config_type=_config_type(key), factory=FixedCN, summary="x")
    with pytest.raises(ValueError, match=match):
        I.register_integrator(info)
    assert key not in I.INTEGRATORS


def test_registration_rejects_a_taken_config_type_and_a_wrong_key() -> None:
    fixed = I.integrator_info(FixedStepping)
    with pytest.raises(ValueError, match="taken"):
        I.register_integrator(dataclasses.replace(fixed, key="soil_water/richards_time@none:alt_again"))
    with pytest.raises(ValueError, match="KEY"):
        I.register_integrator(
            I.IntegratorInfo(
                key="soil_water/richards_time@none:alt_y",
                config_type=_config_type("soil_water/richards_time@none:alt_z"),
                factory=FixedCN,
                summary="x",
            )
        )


# ---------------------------------------------------------------------------
# a new integrator plugs in by registration
# ---------------------------------------------------------------------------


class _WrappedSteps(FixedStepping):
    KEY: ClassVar[str] = "soil_water/richards_time@none:alt_test_wrapper"


class _Wrapped:
    """A plug-in integrator: delegates to the fixed steps (so the results must be bit for bit)."""

    def __init__(self, config: _WrappedSteps) -> None:
        self.config = config
        base = {f.name: getattr(config, f.name) for f in dataclasses.fields(FixedStepping)}
        self._inner = FixedCN(FixedStepping(**base))

    @property
    def capabilities(self) -> I.Capabilities:
        return self._inner.capabilities

    def step_day(self, problem: Any, water: Any, plan: Any) -> Any:
        return self._inner.step_day(problem, water, plan)

    def check(self, h: Any, flux: Any, where: str) -> Any:
        return h


@pytest.fixture
def wrapped() -> Any:
    info = I.register_integrator(
        I.IntegratorInfo(key=_WrappedSteps.KEY, config_type=_WrappedSteps, factory=_Wrapped, summary="test")
    )
    yield info
    I.INTEGRATORS.pop(info.key)


def test_a_registered_integrator_runs_the_days_without_changing_them(wrapped: Any) -> None:
    assert isinstance(I.integrator_for(_WrappedSteps(n_sub=12, n_iter=3)), _Wrapped)
    base = TD._params(DayConfig(n_pre=4, n_post=8), n_sub=12, n_iter=3)
    plug = dataclasses.replace(
        base, richards=dataclasses.replace(base.richards, stepping=_WrappedSteps(n_sub=12, n_iter=3))
    )
    forcing = TD._forcing(3, seed=5, supply_scale=0.3)
    w0 = SoilWater.from_theta(jnp.full(37, 0.25), base.richards.soil)
    for day in range(3):
        f = jax.tree_util.tree_map(lambda x, d=day: x[d], forcing)
        a = soil_water_day_kernel(w0, base, f.supply, f.evaporation, f.uptake, f.storm)[0]
        b = soil_water_day_kernel(w0, plug, f.supply, f.evaporation, f.uptake, f.storm)[0]
        c = richards_day(w0, base.richards, f.supply, f.evaporation, f.uptake)
        d = richards_day(w0, plug.richards, f.supply, f.evaporation, f.uptake)
        for x, y in ((a, b), (c, d)):
            for u, v in zip(jax.tree_util.tree_leaves(x), jax.tree_util.tree_leaves(y), strict=True):
                np.testing.assert_array_equal(np.asarray(u), np.asarray(v))


def test_richards_params_take_only_registered_integrator_configs() -> None:
    soil, grid = TD.catpa_soil(), TD.catpa_grid()
    with pytest.raises(KeyError):
        RichardsParams(soil=soil, grid=grid, stepping=_config_type("soil_water/unknown@none:alt_u")())
    with pytest.raises(TypeError):
        RichardsParams(soil=soil, grid=grid, config=FixedStepping())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# integrator conformance
# ---------------------------------------------------------------------------


def test_every_registered_integrator_has_a_conformance_case() -> None:
    assert K._registered_without_case() == []
    for c in K.integrator_cases():
        assert c.basis.strip() and c.key in I.INTEGRATORS


def _case_id(c: K.IntegratorCase) -> str:
    return c.name


@pytest.mark.parametrize("check", K.CHECKS, ids=lambda f: f.__name__)
@pytest.mark.parametrize("case", K.integrator_cases(), ids=_case_id)
def test_integrator_conformance(case: K.IntegratorCase, check: Callable[[K.IntegratorCase], None]) -> None:
    check(case)


def test_the_leaky_fixture_fails_the_balance_check() -> None:
    with pytest.raises(ConformanceError, match="balance"):
        K.check_balance(K.KNOWN_BAD)
    good = dataclasses.replace(K.KNOWN_BAD, name="converged_fixed", integrator=None)
    K.check_balance(good)  # the same steps without the leak close the balance
    m = K.measure(K.KNOWN_BAD)
    assert m["balance_theta_h_cm"] < m["balance_method_cm"]  # the heads do not see the leak


def test_the_out_of_budget_fixture_fails_the_convergence_check() -> None:
    with pytest.raises(ConformanceError, match="convergence"):
        K.check_convergence(K.KNOWN_UNCONVERGED)
    K.check_convergence(dataclasses.replace(K.KNOWN_UNCONVERGED, stepping=K.REFERENCE))


# ---------------------------------------------------------------------------
# PROBLEM_VERSION pins the physics
# ---------------------------------------------------------------------------

#: PROBLEM_VERSION -> the fingerprint of one sub-step on the frozen input (float64; tests/unit tier)
PHYSICS_PINS: dict[int, tuple[float, float, float, float]] = {
    1: (-2.6412463251045546, 0.35443740272337854, 0.0001516044412799034, 2.2802909061198275),
}


def _fingerprint() -> tuple[float, float, float, float]:
    theta0, _, _, uptake = K.frozen_inputs(np.float64)
    soil = K._soil(jnp.float64)
    grid = K._grid(jnp.float64)
    params = RichardsParams(soil=soil, grid=grid)
    supply = np.zeros(24)
    supply[3] = 2.0
    problem = RichardsProblem.of_day(params, supply, np.full(24, 0.02), uptake[0], jnp.float64)
    w = SoilWater.from_theta(jnp.asarray(theta0), soil)
    h = w.h * 1.01
    sup, eva = problem.rates(jnp.asarray(3.0), jnp.asarray(0.5))
    sink = problem.sink_of()(jnp.asarray(3.0), jnp.asarray(0.5), w.theta, w.h)
    a, _ = problem.step_args(w.h, w.theta, w.pond, sup, eva, sink, jnp.asarray(0.5), jnp.asarray(0.5), 10.0)
    r, dl, d, du = problem.jacobian(h, a)
    q = problem.face_fluxes(h, a)
    return (
        float(jnp.sum(r * grid.tl)),
        float(jnp.sum(jnp.abs(d))),
        float(jnp.sum(dl * du)),
        float(jnp.sum(q)),
    )


@pytest.mark.allow_skip(reason="the pins are float64 values")
@pytest.mark.skipif(not X64, reason="the pins are float64 values")
def test_problem_version_pins_the_physics() -> None:
    got = _fingerprint()
    pin = PHYSICS_PINS[PROBLEM_VERSION]
    assert np.allclose(got, pin, rtol=1e-12, atol=0.0), (
        f"the physics output changed ({got} against {pin}): bump PROBLEM_VERSION and pin the new values"
    )
