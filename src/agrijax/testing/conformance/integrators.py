"""Integrator conformance: every registered Richards integrator on the frozen problem.

A time integrator of the Richards problem (:mod:`agrijax.processes.soil_water.integrator`) plugs in
without touching the soil-water day, so it is checked on its own, against the physics it is given.
The **frozen problem** (:func:`frozen_inputs`, ``PROBLEM_VERSION`` 1) is three days on the CA-TPA
node grid (37 nodes, 150 cm, the five horizons of ``_builtin/soil_water.py``): an evaporation and
uptake day on a moderately wet profile, a 6 cm storm in one hour (the ponded infiltration capacity
binds, runoff), a redistribution day. Its **reference** is a converged reference scheme
(Crank-Nicolson with an ``alpha = 1`` fallback, 960 steps a day, Newton to 1e-10), run by the
faithful integrator with the step pinned at 24/960 h (:data:`REFERENCE`); it must
converge on every day. Being the faithful integrator's own code at a finer step, it is not independent
of the ``faithful`` cases: that independence is the data tier's
(``tests/integration/test_richards_integrators_catpa.py``: the kit's reference against the stored
output of a converged run on CA-TPA) and the ``PROBLEM_VERSION`` fingerprint.

An :class:`IntegratorCase` names one integrator config and declares its tolerances, each measured
(``basis`` names the measurement). The checks (:data:`CHECKS`):

* ``reference`` -- the daily storage [cm] and heads (``|h - h_ref| / max(|h_ref|, hb)``) agree with
  the reference within the declared tolerance, in the dtype of the run;
* ``convergence`` -- no unconverged sub-step and no exhausted step budget on any day (the counters
  ``n_unconverged`` and ``budget_exhausted`` of the day's fluxes; always 0 for an integrator without a
  convergence test);
* ``balance`` -- the cumulative water balance over the run closes in two views: the method's
  conserved quantity (the state ``theta``: the sum of the daily ``balance_error``) and the physical
  ``theta(h)`` of the final heads (storage from ``theta_of_h(h)`` against the booked fluxes);
* ``transforms`` -- ``vmap(jit)`` over three soils equals per-sample ``jit`` (bit for bit, or within the
  declared number of ulps), and the lanes of a batch are independent;
* ``dtypes`` -- every output keeps the input dtype and is finite;
* ``gradient`` -- when the integrator declares gradients, ``d loss / d (ksat, lambda scales)`` agrees
  with central differences within the declared relative tolerance (loss: final storage plus drainage),
  in float64; in float32 (central differences need float64) only a finite AD gradient is checked.

``KNOWN_BAD`` is an integrator that leaks water from the state (the state ``theta`` loses a fraction
of itself every day, the heads and fluxes do not); the ``balance`` check must fail on it
(``tests/unit/test_richards_integrators.py``). ``KNOWN_UNCONVERGED`` is the reference scheme at 1920
steps a day with the default step budget of 1000: every day runs out of budget, and the
``convergence`` check must fail on it.

Source: framework check (no reference equation); the reference is Crank-Nicolson time stepping with
Newton iteration on the mixed form of the Richards equation, at a fine step (960 a day) and a tight
Newton tolerance.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from agrijax.processes.soil_water.fixed_cn import FixedStepping
from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams, theta_of_h
from agrijax.processes.soil_water.integrator import (
    INTEGRATORS,
    DayPlan,
    RichardsIntegrator,
    integrator_for,
)
from agrijax.processes.soil_water.problem import PROBLEM_VERSION, RichardsGrid, RichardsProblem, SoilWater
from agrijax.processes.soil_water.richards import RichardsParams, richards_day_with
from agrijax.processes.soil_water.richards_adaptive import AdaptiveStepping

from .case import ConformanceError

__all__ = [
    "CHECKS",
    "KNOWN_BAD",
    "KNOWN_UNCONVERGED",
    "PROBLEM_VERSION",
    "REFERENCE",
    "FrozenRun",
    "IntegratorCase",
    "LeakyIntegrator",
    "check_balance",
    "check_convergence",
    "check_dtypes",
    "check_gradient",
    "check_reference",
    "check_transforms",
    "frozen_inputs",
    "gradient_ad",
    "integrator_cases",
    "measure",
    "run_frozen",
]

N_DAYS, N_ROOT, HOURS = 3, 14, 24
#: CA-TPA rzwqm.dat node records and horizons (as ``_builtin/soil_water.py``)
TLT = np.array(
    [1, 2, 4, 7, 11, 15, 19, 23, 26, 30, 34, 38, 43, 48, 53, 58, 63, 67, 70, 73, 77, 82, 86, 90, 94, 98,
     103, 108, 113, 118, 123, 128, 133, 138, 143, 147, 150], dtype=float
)  # fmt: skip
DELZ = np.array(
    [1, 1, 3, 3, 5, 3, 5, 3, 3, 5, 3, 5, 5, 5, 5, 5, 5, 3, 3, 3, 5, 5, 3, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5,
     5, 3, 0], dtype=float
)  # fmt: skip
HORIZON_BOTTOM = np.array([15.0, 30.0, 70.0, 90.0, 150.0])
#: cell thicknesses [cm] (RichardsGrid.from_rzwqm: diff([0, TLT]))
TL = np.diff(np.concatenate([[0.0], TLT]))
REC1 = np.array(
    [
        [14.6545, 0.22, 2.966, 5.41, 0.055, 0.453],
        [14.6545, 0.26, 2.966, 3.16, 0.032, 0.453],
        [14.6545, 0.36, 2.966, 3.31, 0.043, 0.453],
        [14.6545, 0.17, 2.966, 3.32, 0.048, 0.453],
        [14.6545, 0.322, 2.966, 2.59, 0.041, 0.453],
    ]
)
REC2 = np.tile([0.0, 0.0, 0.0, 14.6545, 7440.01, 0.0, 0.0], (5, 1))
#: the storm of day 1 [cm] in the hour STORM_HOUR
STORM_CM, STORM_HOUR = 2.5, 3
#: daily evaporation demand [cm d-1] (daytime sine) and root uptake [cm d-1] of the three days
EVAP_CM = (0.45, 0.10, 0.35)
UPTAKE_CM = (0.30, 0.05, 0.25)
#: the soils of the transforms check (common factors on ksat and lambda)
BATCH_SCALES = ((1.0, 1.0), (0.97, 1.02), (1.03, 0.98))
#: relative step of the central differences of the gradient check
FD_STEP = 1.0e-6

#: the converged reference scheme: CN, alpha = 1 fallback, 960 steps a day
REFERENCE = AdaptiveStepping(dt_min=HOURS / 960, dt_reset=HOURS / 960, dt_max=HOURS / 960)


# ------------------------------------------------------------------ the frozen problem
def _soil(dtype: Any, scales: Any = (1.0, 1.0)) -> SoilHydraulicParams:
    rec1 = REC1.copy()
    node_horizon = np.searchsorted(HORIZON_BOTTOM, TLT, side="left")
    soil = SoilHydraulicParams.from_rzwqm_records(rec1, REC2, node_horizon=node_horizon)
    soil = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), soil)
    sk, sl = scales
    return soil.replace(ksat=soil.ksat * sk, lambda_=soil.lambda_ * sl)


def _grid(dtype: Any) -> RichardsGrid:
    g = RichardsGrid.from_rzwqm(TLT, DELZ)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype), g)


def frozen_inputs(dtype: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(theta0[n], supply[D, 24], evaporation[D, 24], uptake[D, n])`` of the frozen problem (NumPy)."""
    node_horizon = np.searchsorted(HORIZON_BOTTOM, TLT, side="left")
    tr, ts = REC1[node_horizon, 4], REC1[node_horizon, 5]
    theta0 = tr + np.linspace(0.45, 0.65, len(TLT)) * (ts - tr)
    hours = np.arange(HOURS)
    shape = np.where((hours >= 6) & (hours < 18), np.sin(np.pi * (hours - 6 + 0.5) / 12), 0.0)
    evap = np.outer(EVAP_CM, shape / shape.sum())
    supply = np.zeros((N_DAYS, HOURS))
    supply[1, STORM_HOUR] = STORM_CM
    weights = np.linspace(2.0, 0.2, N_ROOT) / np.linspace(2.0, 0.2, N_ROOT).sum()
    uptake = np.zeros((N_DAYS, len(TLT)))
    uptake[:, :N_ROOT] = np.asarray(UPTAKE_CM)[:, None] * weights
    return tuple(np.asarray(x, dtype) for x in (theta0, supply, evap, uptake))  # type: ignore[return-value]


class FrozenRun(NamedTuple):
    """Daily outputs of a run of the frozen problem."""

    storage: Any  # [D] sum(theta tl) of the state [cm]
    storage_h: Any  # [D] sum(theta(h) tl) of the heads [cm]
    h: Any  # [D, n]
    theta: Any  # [D, n]
    pond: Any  # [D]
    balance_error: Any  # [D] the day's balance of the state (method view) [cm]
    inflow: Any  # [D] supply [cm]
    outflow: Any  # [D] evaporation + drainage + uptake + other sinks + runoff [cm]
    drainage: Any  # [D]
    n_unconverged: Any  # [D] unconverged sub-steps of the day (0 without a convergence test)
    budget_exhausted: Any  # [D] 1 when the day ran out of its step budget


def _run_fn(integrator: RichardsIntegrator, dtype: Any) -> Callable[[Any], FrozenRun]:
    """``scales[2] -> FrozenRun``: the frozen problem with ``integrator`` (ksat, lambda scaled)."""
    theta0, supply, evap, uptake = (jnp.asarray(x) for x in frozen_inputs(dtype))
    grid = _grid(dtype)

    def run(scales: Any) -> FrozenRun:
        soil = _soil(dtype, (scales[0], scales[1]))
        params = RichardsParams(soil=soil, grid=grid)
        w0 = SoilWater.from_theta(theta0, soil)

        def body(w: SoilWater, f: Any) -> tuple[SoilWater, Any]:
            w2 = richards_day_with(integrator, w, params, *f)
            fl = w2.flux
            out = fl.evaporation + fl.drainage + fl.uptake + fl.runoff
            out = out + fl.tile + fl.lateral + fl.subirrigation + fl.macropore_to_drain
            th_h = theta_of_h(w2.h, soil.at_nodes())
            return w2, FrozenRun(
                storage=w2.storage(grid),
                storage_h=jnp.sum(th_h * grid.tl),
                h=w2.h,
                theta=w2.theta,
                pond=w2.pond,
                balance_error=fl.balance_error,
                inflow=jnp.sum(f[0]),
                outflow=out,
                drainage=fl.drainage,
                n_unconverged=fl.n_unconverged,
                budget_exhausted=fl.budget_exhausted,
            )

        return lax.scan(body, w0, (supply, evap, uptake))[1]

    return run


def run_frozen(integrator: RichardsIntegrator, dtype: Any, scales: Any = (1.0, 1.0)) -> FrozenRun:
    """One jitted run of the frozen problem."""
    return jax.jit(_run_fn(integrator, dtype))(jnp.asarray(scales, dtype))


@functools.lru_cache(maxsize=4)
def _reference(dtype_name: str) -> FrozenRun:
    dtype = jnp.dtype(dtype_name)
    run = jax.tree_util.tree_map(np.asarray, run_frozen(integrator_for(REFERENCE), dtype))
    bad = _unconverged(run)
    if bad:
        raise ConformanceError(f"the reference run did not converge on the frozen problem: {bad}")
    return run


def reference(dtype: Any) -> FrozenRun:
    """The converged reference run of the frozen problem (cached per dtype)."""
    return _reference(jnp.dtype(dtype).name)


# ------------------------------------------------------------------ cases
@dataclasses.dataclass(frozen=True)
class IntegratorCase:
    """One integrator config and its declared (measured) tolerances on the frozen problem.

    ``storage_cm``/``head_rel``: agreement with the reference (float64; ``*_f32`` float32);
    ``balance_cm``: ``(method view, theta(h) view)`` of the cumulative balance over the run (float64,
    ``balance_cm_f32`` float32); ``transforms_ulps``: 0 is bit for bit; ``grad_rtol``: central
    differences, ``None`` when the integrator declares no gradient; ``basis``: the measurement.
    """

    name: str
    stepping: Any
    storage_cm: float
    head_rel: float
    storage_cm_f32: float
    head_rel_f32: float
    balance_cm: tuple[float, float]
    balance_cm_f32: tuple[float, float]
    transforms_ulps: int
    grad_rtol: float | None
    basis: str
    integrator: RichardsIntegrator | None = None  # a fixture outside the registry

    @property
    def key(self) -> str:
        return self.integrator_obj.config.KEY if self.integrator is None else "fixture"

    @property
    def integrator_obj(self) -> RichardsIntegrator:
        return integrator_for(self.stepping) if self.integrator is None else self.integrator


def _x64() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def _dtype() -> Any:
    return jnp.float64 if _x64() else jnp.float32


def _fail(case: IntegratorCase, check: str, msg: str) -> ConformanceError:
    return ConformanceError(f"{case.name} [{check}]: {msg}")


def measure(case: IntegratorCase, dtype: Any = None) -> dict[str, float]:
    """The metrics the checks compare with the declared tolerances (and a few more), in ``dtype``."""
    dt = _dtype() if dtype is None else dtype
    run = jax.tree_util.tree_map(np.asarray, run_frozen(case.integrator_obj, dt))
    ref = reference(dt)
    hb = np.asarray(_soil(dt).at_nodes().hb, np.float64)
    h, h_ref = np.asarray(run.h, np.float64), np.asarray(ref.h, np.float64)
    head = np.abs(h - h_ref) / np.maximum(np.abs(h_ref), hb)
    theta0 = frozen_inputs(dt)[0]
    s0 = float(np.sum(np.asarray(theta0, np.float64) * TL))
    h0 = SoilWater.from_theta(jnp.asarray(theta0), _soil(dt)).h
    s0_h = float(np.sum(np.asarray(theta_of_h(h0, _soil(dt).at_nodes()), np.float64) * TL))
    booked = float(np.sum(np.asarray(run.inflow, np.float64) - np.asarray(run.outflow, np.float64)))
    method = float(np.sum(np.asarray(run.balance_error, np.float64)))
    phys = float(run.storage_h[-1]) + float(run.pond[-1]) - s0_h - booked
    return {
        "storage_cm": float(np.max(np.abs(np.asarray(run.storage, np.float64) - np.asarray(ref.storage)))),
        "head_rel": float(np.max(head)),
        "balance_method_cm": abs(method),
        "balance_theta_h_cm": abs(phys),
        "state_storage_end_cm": float(run.storage[-1]),
        "initial_storage_cm": s0,
        "drainage_cm": float(np.sum(run.drainage)),
    }


def _unconverged(run: FrozenRun) -> str:
    """What did not converge in a run ('' when every day converged)."""
    n_bad = float(np.sum(np.asarray(run.n_unconverged)))
    n_out = float(np.sum(np.asarray(run.budget_exhausted)))
    if n_bad == 0.0 and n_out == 0.0:
        return ""
    return f"{n_bad:.0f} unconverged sub-steps, {n_out:.0f} days out of step budget"


def check_convergence(case: IntegratorCase) -> None:
    """An integrator that declares a convergence test converges on every day of the frozen problem
    (no unconverged sub-step, no exhausted step budget); one without it reports none."""
    run = run_frozen(case.integrator_obj, _dtype())
    bad = _unconverged(jax.tree_util.tree_map(np.asarray, run))
    if bad:
        caps = case.integrator_obj.capabilities
        what = "" if caps.convergence else " (the integrator declares no convergence test)"
        raise _fail(case, "convergence", bad + what)


def check_reference(case: IntegratorCase) -> None:
    m = measure(case)
    s_tol, h_tol = (case.storage_cm, case.head_rel) if _x64() else (case.storage_cm_f32, case.head_rel_f32)
    if not (m["storage_cm"] <= s_tol and m["head_rel"] <= h_tol):
        raise _fail(case, "reference", f"storage {m['storage_cm']:.3e} cm (tol {s_tol:.1e}), head "
                    f"{m['head_rel']:.3e} (tol {h_tol:.1e}) against the reference run")  # fmt: skip


def check_balance(case: IntegratorCase) -> None:
    m = measure(case)
    tm, th = case.balance_cm if _x64() else case.balance_cm_f32
    if not (m["balance_method_cm"] <= tm and m["balance_theta_h_cm"] <= th):
        raise _fail(case, "balance", f"cumulative balance: method view {m['balance_method_cm']:.3e} cm (tol "
                    f"{tm:.1e}), theta(h) view {m['balance_theta_h_cm']:.3e} cm (tol {th:.1e})")  # fmt: skip


def _ulps(a: np.ndarray, b: np.ndarray, scale: float = 0.0) -> float:
    """``max |a - b|`` in units of the spacing of the largest magnitude of the leaf, or of ``scale`` when
    larger (a balance error is measured against the storage it is a difference of); 0: bit for bit."""
    a, b = np.asarray(a), np.asarray(b)
    if not np.issubdtype(a.dtype, np.floating):
        return 0.0 if np.array_equal(a, b) else float("inf")
    scale = max(scale, float(np.max(np.abs(np.concatenate([a.ravel(), b.ravel()])), initial=0.0)))
    unit = float(np.spacing(a.dtype.type(scale))) if scale > 0.0 else float(np.finfo(a.dtype).tiny)
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0)) / unit


def transforms_ulps(case: IntegratorCase) -> float:
    """Largest difference between ``vmap(jit)`` over :data:`BATCH_SCALES` and per-sample ``jit``, and
    between a lane of the batch and the same lane in a batch with the others changed, in units of the
    spacing of each output's largest magnitude (0: bit for bit)."""
    dt = _dtype()
    fn = _run_fn(case.integrator_obj, dt)
    scales = jnp.asarray(BATCH_SCALES, dt)
    batched = jax.jit(jax.vmap(fn))(scales)
    storage = float(np.max(np.abs(np.asarray(batched.storage))))
    worst = 0.0

    def compare(x: FrozenRun, y: FrozenRun, i: int | None, j: int | None) -> float:
        out = 0.0
        for name in FrozenRun._fields:
            a, b = np.asarray(getattr(x, name)), np.asarray(getattr(y, name))
            a = a if i is None else a[i]
            b = b if j is None else b[j]
            out = max(out, _ulps(a, b, storage if name == "balance_error" else 0.0))
        return out

    single = jax.jit(fn)
    for i in range(len(BATCH_SCALES)):
        worst = max(worst, compare(batched, single(scales[i]), i, None))
    other = scales.at[1:].set(scales[1:] * jnp.asarray(1.01, dt))  # lane 0 unchanged
    worst = max(worst, compare(batched, jax.jit(jax.vmap(fn))(other), 0, 0))
    return worst


def check_transforms(case: IntegratorCase) -> None:
    u = transforms_ulps(case)
    if u > case.transforms_ulps:
        raise _fail(case, "transforms", f"vmap(jit) against jit / batch independence: {u:.0f} ulp "
                    f"(declared {case.transforms_ulps})")  # fmt: skip


def check_dtypes(case: IntegratorCase) -> None:
    dt = _dtype()
    run = run_frozen(case.integrator_obj, dt)
    bad = [i for i, x in enumerate(jax.tree_util.tree_leaves(run)) if x.dtype != dt]
    if bad:
        raise _fail(case, "dtypes", f"outputs {bad} are not {jnp.dtype(dt).name}")
    if not all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(run)):
        raise _fail(case, "dtypes", "non-finite output")
    caps = case.integrator_obj.capabilities
    if jnp.dtype(dt).name not in caps.dtypes:
        raise _fail(case, "dtypes", f"{jnp.dtype(dt).name} not among the declared dtypes {caps.dtypes}")


def _loss_fn(case: IntegratorCase) -> Callable[[Any], Any]:
    fn = _run_fn(case.integrator_obj, _dtype())

    def loss(sc: Any) -> Any:
        r = fn(sc)
        return r.storage[-1] + jnp.sum(r.drainage)

    return loss


def gradient_ad(case: IntegratorCase) -> np.ndarray:
    """``d loss / d scales`` by AD at the frozen soil (the loss of :func:`gradient_error`)."""
    return np.asarray(jax.jit(jax.grad(_loss_fn(case)))(jnp.ones(2, _dtype())), np.float64)


def gradient_error(case: IntegratorCase) -> float:
    """Largest relative difference of ``d loss / d scales`` (AD) and central differences."""
    dt = _dtype()
    loss = _loss_fn(case)
    x0 = jnp.ones(2, dt)
    g = gradient_ad(case)
    pts = jnp.stack([x0 + s * FD_STEP * e for e in jnp.eye(2, dtype=dt) for s in (1.0, -1.0)])
    v = np.asarray(jax.jit(jax.vmap(loss))(pts), np.float64).reshape(2, 2)
    fd = (v[:, 0] - v[:, 1]) / (2.0 * FD_STEP)
    return float(np.max(np.abs(g - fd) / np.maximum(np.abs(fd), 1e-12)))


def check_gradient(case: IntegratorCase) -> None:
    caps = case.integrator_obj.capabilities
    if caps.gradient == "none":
        if case.grad_rtol is not None:
            raise _fail(case, "gradient", "a tolerance is declared for an integrator without gradients")
        return
    if case.grad_rtol is None:
        raise _fail(
            case, "gradient", f"the integrator declares gradient={caps.gradient!r}: declare grad_rtol"
        )
    if not _x64():  # central differences need float64: here the AD gradient must be finite
        g = gradient_ad(case)
        if not np.all(np.isfinite(g)):
            raise _fail(case, "gradient", f"non-finite AD gradient in float32: {g}")
        return
    e = gradient_error(case)
    if not e <= case.grad_rtol:
        raise _fail(case, "gradient", f"AD against central differences: {e:.3e} (tol {case.grad_rtol:.1e})")


CHECKS: tuple[Callable[[IntegratorCase], None], ...] = (
    check_reference,
    check_convergence,
    check_balance,
    check_transforms,
    check_dtypes,
    check_gradient,
)


# ------------------------------------------------------------------ the known-bad fixture
class LeakyIntegrator:
    """A fixture that leaks water: the registered integrator of ``config`` whose state ``theta`` loses
    ``LEAK`` of itself every day while its heads and booked fluxes do not."""

    LEAK = 1.0e-4

    def __init__(self, config: Any) -> None:
        self.config = config
        self._inner = integrator_for(config)

    @property
    def capabilities(self) -> Any:
        return self._inner.capabilities

    def step_day(self, problem: RichardsProblem, water: SoilWater, plan: DayPlan) -> tuple[SoilWater, Any]:
        new, diag = self._inner.step_day(problem, water, plan)
        return new.replace(theta=new.theta * (1.0 - self.LEAK)), diag

    def check(self, h: Any, flux: Any, where: str) -> Any:
        return h


# ------------------------------------------------------------------ this repository's cases
def integrator_cases() -> list[IntegratorCase]:
    """The integrator cases of this repository (every registered integrator has at least one)."""
    return list(_CASES)


#: the ``basis`` of the cases: how the declared tolerances were set, from float64 and float32
#: measurements of each integrator
_RULE = (
    "measured for each integrator on a cluster CPU node, in float64 and float32: agreement "
    "rounded up in the second digit (above the reference's own limit, reference against 1920 steps a "
    "day: 7.6e-7 cm and 6.0e-6 in float64, 1.5e-5 cm and 2.0e-5 in float32); rounding-level balances, "
    "batch differences and gradient errors rounded up to the next power of ten"
)

_CASES: tuple[IntegratorCase, ...] = (
    IntegratorCase(
        "faithful_exact",
        AdaptiveStepping.exact(),
        storage_cm=1.4e-6,  # measured 1.31e-6
        head_rel=9.6e-5,  # 9.52e-5
        storage_cm_f32=2.7e-5,  # 2.67e-5
        head_rel_f32=1.5e-4,  # 1.44e-4
        balance_cm=(1e-13, 1e-13),  # 2.9e-14, 3.6e-14
        balance_cm_f32=(1e-4, 1e-4),  # 3.3e-5, 3.4e-5
        transforms_ulps=100,  # 47.5 (float64), 66 (float32)
        grad_rtol=1e-6,  # 1.26e-7
        basis=_RULE,
    ),
    IntegratorCase(
        "faithful_fast",
        AdaptiveStepping.fast(),
        storage_cm=1.2e-5,  # 1.17e-5
        head_rel=1.3e-4,  # 1.22e-4
        storage_cm_f32=4.6e-5,  # 4.58e-5
        head_rel_f32=6.6e-4,  # 6.53e-4
        balance_cm=(1e-13, 1e-13),  # 2.4e-14, 1.6e-14
        balance_cm_f32=(1e-5, 1e-5),  # 4.3e-6, 3.1e-6
        transforms_ulps=100,  # 72, 26
        grad_rtol=1e-7,  # 5.0e-8
        basis=_RULE,
    ),
    IntegratorCase(
        "fixed_24x3",
        FixedStepping.baseline(),
        storage_cm=0.013,  # 0.01251
        head_rel=0.027,  # 0.02656
        storage_cm_f32=0.013,  # 0.01255
        head_rel_f32=0.027,  # 0.02656
        balance_cm=(9.4e-4, 9.4e-4),  # 9.36e-4 both (the unconverged residual of 3 iterations)
        balance_cm_f32=(9.5e-4, 9.5e-4),  # 9.40e-4, 9.39e-4
        transforms_ulps=0,  # 0, 0
        grad_rtol=1e-7,  # 7.2e-8
        basis=_RULE,
    ),
    IntegratorCase(
        "fixed_96x8",
        FixedStepping.m1(),
        storage_cm=3.0e-3,  # 2.99e-3
        head_rel=7.3e-3,  # 7.26e-3
        storage_cm_f32=3.1e-3,  # 3.02e-3
        head_rel_f32=7.3e-3,  # 7.27e-3
        balance_cm=(1e-13, 1e-13),  # 1.2e-14, 2.0e-14
        balance_cm_f32=(1e-5, 1e-5),  # 3.9e-6, 5.1e-6
        transforms_ulps=10,  # 0, 1
        grad_rtol=1e-6,  # 2.5e-7
        basis=_RULE,
    ),
    IntegratorCase(
        "fixed_96x8_ift",
        FixedStepping.m1(grad="implicit"),  # the forward of fixed_96x8; the custom (IFT) gradient
        storage_cm=3.0e-3,
        head_rel=7.3e-3,
        storage_cm_f32=3.1e-3,
        head_rel_f32=7.3e-3,
        balance_cm=(1e-13, 1e-13),
        balance_cm_f32=(1e-5, 1e-5),
        transforms_ulps=10,
        grad_rtol=1e-6,  # 2.5e-7 (measured)
        basis=_RULE + "; forward as fixed_96x8 (grad changes only the reverse pass)",
    ),
    IntegratorCase(
        "fixed_cn_96x8",
        FixedStepping.m1(time_scheme="rzwqm"),
        storage_cm=1.1e-4,  # 1.08e-4
        head_rel=1.4e-3,  # 1.32e-3
        storage_cm_f32=1.5e-4,  # 1.45e-4
        head_rel_f32=1.4e-3,  # 1.32e-3
        balance_cm=(1e-13, 1e-13),  # 1.6e-14, 9.3e-15
        balance_cm_f32=(1e-5, 1e-5),  # 1.1e-6, 2.3e-6
        transforms_ulps=0,  # 0, 0
        grad_rtol=1e-6,  # 1.05e-7
        basis=_RULE,
    ),
)

#: the reference scheme at 1920 steps a day within the default budget of 1000 steps: out of budget
#: every day (measured: budget_exhausted = 1 on all three days, storage off by
#: 0.0166 cm; with max_steps = 2000 it converges and agrees with the reference to 7.6e-7 cm)
_OUT_OF_BUDGET = AdaptiveStepping(dt_min=HOURS / 1920, dt_reset=HOURS / 1920, dt_max=HOURS / 1920)

KNOWN_UNCONVERGED = IntegratorCase(
    name="out_of_budget_fixture",
    stepping=_OUT_OF_BUDGET,
    storage_cm=np.inf,
    head_rel=np.inf,
    storage_cm_f32=np.inf,
    head_rel_f32=np.inf,
    balance_cm=(np.inf, np.inf),
    balance_cm_f32=(np.inf, np.inf),
    transforms_ulps=2**62,
    grad_rtol=1.0,
    basis="kit fixture: must fail the convergence check",
)

#: the fixture's base closes the balance to rounding (the fast tier), so it fails by its leak alone
_LEAK_BASE = AdaptiveStepping.fast()

KNOWN_BAD = IntegratorCase(
    name="leaky_fixture",
    stepping=_LEAK_BASE,
    storage_cm=np.inf,
    head_rel=np.inf,
    storage_cm_f32=np.inf,
    head_rel_f32=np.inf,
    balance_cm=(1e-13, 1e-13),  # the base's (faithful_fast); the leak measured 1.2e-2 cm, 8.1e-3 cm
    balance_cm_f32=(1e-5, 1e-5),
    transforms_ulps=2**62,
    grad_rtol=1.0,
    basis="kit fixture: must fail the balance check",
    integrator=LeakyIntegrator(_LEAK_BASE),
)


def _registered_without_case() -> list[str]:
    have = {c.integrator_obj.config.KEY for c in _CASES}
    return sorted(set(INTEGRATORS) - have)
