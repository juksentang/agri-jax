"""Calibration and inference: parameter spaces, objectives, batched optimisers, gradient trust.

* :mod:`~agrijax.calib.space` - named, bounded parameters and their unconstrained coordinates, and
  shared plus per-site blocks of them laid out in one vector (several sites at once);
* :mod:`~agrijax.calib.objective` - observed quantities (final values, series, stage dates) and a
  weighted loss over batch groups;
* :mod:`~agrijax.calib.optim` - batched Adam, L-BFGS, Levenberg-Marquardt, CMA-ES and random
  search with model-call counts (per row for L-BFGS and LM, which stop and freeze rows one by one),
  and the secant replacement of untrustworthy gradient components;
* :mod:`~agrijax.calib.trust` - the gradient-trust report (FD agreement at two steps, line scans
  counting jumps and zero-derivative stretches, sensitivities, trust levels 1-3);
* :mod:`~agrijax.calib.methods` - which parameters get a gradient and which a derivative-free
  search, from the trust report at one point (:func:`plan_methods`, :class:`MethodPlan`);
* :mod:`~agrijax.calib.ceres` - the CERES-Maize adapter (cultivar bounds, batch simulators,
  the default gradient / derivative-free split); import it explicitly (it imports the crop model);
* :func:`calibrate` (:mod:`~agrijax.calib.workflow`) - one call from a DSSAT v4.8.6 maize example
  treatment (it needs the free-run input tables of the instrumented DSSAT build, not distributed) to
  calibrated CERES-Maize cultivar coefficients, a ``.CUL`` row and the check in DSSAT; its data-free
  core is :mod:`~agrijax.calib.fit`, the batched free-run day :mod:`~agrijax.calib.dssat_day`
  (imported on first use: they import the models);
* :mod:`~agrijax.calib.soil` - DSSAT soil parameters (CN, SWCON, U, SLPF, SRGF, SWCN bounds with
  sources; a layer's LL < DUL < SAT as one ordered chain; a site's space by ``.SOL`` layer, with the
  coordinates whose site value is an end of their domain left out on record);
* :mod:`~agrijax.calib.day_soil` - a soil written into every copy the free-run DSSAT day holds
  (``with_soil``; imported on first use: it imports the models).

**Simulators and the gradient mode.** The gradient mode (``exact`` / ``ste`` / ``implicit``,
:mod:`agrijax.core.grad`) is read when a function is traced, and ``jax.jit`` does not key its
cache on it: a jitted simulator keeps the mode of its first trace. The simulators of this package
therefore take the mode as a required argument and bind it
(:func:`agrijax.core.grad.bind_gradient_mode`): one simulator is one mode, and calling it inside
an explicit ``gradient_mode`` context of another mode raises
:class:`~agrijax.core.grad.ModeMismatchError`. To compare modes, build one simulator per mode.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from agrijax.calib.methods import MethodPlan, plan_methods
from agrijax.calib.objective import (
    Group,
    Objective,
    Target,
    evaluate_targets,
    first_day_index,
    synthetic_observations,
)
from agrijax.calib.optim import (
    LM_MINPACK_TOL,
    AdamConfig,
    CmaConfig,
    LbfgsConfig,
    LmConfig,
    OptResult,
    adam,
    batched_loss,
    batched_residual_jacobian,
    batched_value_and_grad,
    cma_es,
    lbfgs,
    lm,
    pair_gradient,
    pair_gradient_forwards,
    random_search,
    secant_gradient,
    time_calls,
)
from agrijax.calib.space import (
    UNBOUNDED,
    Domain,
    OrderedChain,
    ParamBlocks,
    ParamSpace,
    ParamSpec,
    non_negative,
    positive,
)
from agrijax.calib.trust import (
    GradientPlan,
    TrustConfig,
    fd_check,
    gradient_plan,
    line_scan,
    sensitivity,
    trust_report,
)

if TYPE_CHECKING:
    from agrijax.calib.observations import ObservationError
    from agrijax.calib.workflow import CalibrationResult, ScopeError, calibrate

#: names served lazily, with the submodule that defines them: :mod:`agrijax.calib.workflow` imports the
#: crop and day models; :class:`ObservationError` is defined in :mod:`agrijax.calib.observations`
#: (which does not, and which :mod:`~agrijax.calib.workflow` re-exports)
_LAZY = {
    "calibrate": "workflow",
    "CalibrationResult": "workflow",
    "ScopeError": "workflow",
    "ObservationError": "observations",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f"agrijax.calib.{_LAZY[name]}"), name)
    raise AttributeError(f"module 'agrijax.calib' has no attribute {name!r}")


__all__ = [
    "LM_MINPACK_TOL",
    "UNBOUNDED",
    "AdamConfig",
    "CalibrationResult",
    "CmaConfig",
    "Domain",
    "GradientPlan",
    "Group",
    "LbfgsConfig",
    "LmConfig",
    "MethodPlan",
    "Objective",
    "ObservationError",
    "OptResult",
    "OrderedChain",
    "ParamBlocks",
    "ParamSpace",
    "ParamSpec",
    "ScopeError",
    "Target",
    "TrustConfig",
    "adam",
    "batched_loss",
    "batched_residual_jacobian",
    "batched_value_and_grad",
    "calibrate",
    "cma_es",
    "evaluate_targets",
    "fd_check",
    "first_day_index",
    "gradient_plan",
    "lbfgs",
    "line_scan",
    "lm",
    "non_negative",
    "pair_gradient",
    "pair_gradient_forwards",
    "plan_methods",
    "positive",
    "random_search",
    "secant_gradient",
    "sensitivity",
    "synthetic_observations",
    "time_calls",
    "trust_report",
]
