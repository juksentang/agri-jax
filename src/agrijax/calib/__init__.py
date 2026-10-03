"""Calibration and inference: parameter spaces, objectives, batched optimisers, gradient trust.

* :mod:`~agrijax.calib.space` - named, bounded parameters and their unconstrained coordinates;
* :mod:`~agrijax.calib.objective` - observed quantities (final values, series, stage dates) and a
  weighted loss over batch groups;
* :mod:`~agrijax.calib.optim` - batched Adam, CMA-ES and random search with model-call counts,
  and the secant replacement of untrustworthy gradient components;
* :mod:`~agrijax.calib.trust` - the gradient-trust report (FD agreement at two steps, line scans
  counting jumps and zero-derivative stretches, sensitivities, trust levels 1-3);
* :mod:`~agrijax.calib.ceres` - the CERES-Maize adapter (cultivar bounds, batch simulators,
  the default gradient / derivative-free split); import it explicitly (it imports the crop model);
* :func:`calibrate` (:mod:`~agrijax.calib.workflow`) - one call from a DSSAT v4.8.6 maize example
  treatment (it needs the free-run input tables of the instrumented DSSAT build, not distributed) to
  calibrated CERES-Maize cultivar coefficients, a ``.CUL`` row and the check in DSSAT; its data-free
  core is :mod:`~agrijax.calib.fit`, the batched free-run day :mod:`~agrijax.calib.dssat_day`
  (imported on first use: they import the models);
* :mod:`~agrijax.calib.soil` - DSSAT soil parameters (CN, SWCON, U bounds with sources; a layer's
  LL < DUL < SAT as one ordered chain).

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

from agrijax.calib.objective import (
    Group,
    Objective,
    Target,
    evaluate_targets,
    first_day_index,
    synthetic_observations,
)
from agrijax.calib.optim import (
    AdamConfig,
    CmaConfig,
    OptResult,
    adam,
    batched_loss,
    batched_value_and_grad,
    cma_es,
    pair_gradient,
    random_search,
    secant_gradient,
    time_calls,
)
from agrijax.calib.space import UNBOUNDED, Domain, OrderedChain, ParamSpace, ParamSpec, non_negative, positive
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
    "UNBOUNDED",
    "AdamConfig",
    "CalibrationResult",
    "CmaConfig",
    "Domain",
    "GradientPlan",
    "Group",
    "Objective",
    "ObservationError",
    "OptResult",
    "OrderedChain",
    "ParamSpace",
    "ParamSpec",
    "ScopeError",
    "Target",
    "TrustConfig",
    "adam",
    "batched_loss",
    "batched_value_and_grad",
    "calibrate",
    "cma_es",
    "evaluate_targets",
    "fd_check",
    "first_day_index",
    "gradient_plan",
    "line_scan",
    "non_negative",
    "pair_gradient",
    "positive",
    "random_search",
    "secant_gradient",
    "sensitivity",
    "synthetic_observations",
    "time_calls",
    "trust_report",
]
