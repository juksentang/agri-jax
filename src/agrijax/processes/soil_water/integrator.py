r"""The time-integrator layer of the Richards redistribution: protocol, capabilities and registry.

The physics of a soil-water day is :class:`~agrijax.processes.soil_water.problem.RichardsProblem`
(one version, ``PROBLEM_VERSION``). How it is advanced in time is an **integrator**: an object with
a config of its own (a static :class:`SteppingConfig` subclass), a registry key and declared
:class:`Capabilities`, which the soil-water day calls only through :meth:`RichardsIntegrator.step_day`.
The config is the ``stepping`` field of :class:`~agrijax.processes.soil_water.richards.RichardsParams`,
a tagged union: its class is the tag, :func:`integrator_for` maps it to the registered integrator. No
code outside an integrator module tests which integrator runs.

Registered here (``soil_water/richards_time@rzwqm2-4.6:<variant>``):

* ``:faithful`` -- :class:`~agrijax.processes.soil_water.richards_adaptive.AdaptiveCN`
  (config ``AdaptiveStepping``): adaptive sub-steps from the Newton outcome with Newton to a
  tolerance and the three Newton fixes N1-N3 (described in
  :mod:`~agrijax.processes.soil_water.richards_adaptive`), Crank-Nicolson with ``alpha = 1`` on a
  segment's first step, RZWQM2's time integration (``ADJDT``, ``CNHEAD``);
* ``:alt_fixed_steps`` -- :class:`~agrijax.processes.soil_water.fixed_cn.FixedCN` (config
  ``FixedStepping``): a fixed number of sub-steps per day with a fixed number of damped Newton or
  Picard iterations, fully implicit (default) or Crank-Nicolson (``time_scheme="rzwqm"``): the 24 x 3
  baseline, the 96 x 8 configuration and the legacy modes, kept bit for bit so that their validated
  results stay as pinned.

Variants come from a closed vocabulary: ``faithful`` (RZWQM2's scheme), ``ref_<switch>`` (a reference
switch), ``alt_<name>`` (an alternative scheme of this or another package).

How to add an integrator
------------------------
1. Write a config class ``class MySteps(SteppingConfig)``: an ``equinox.Module`` whose fields are all
   static numerical settings (:func:`~agrijax.processes.soil_water.coefficients.setting_field`, with
   the origin of every value), and set ``KEY: ClassVar[str] = "soil_water/<impl>@<ref>:<variant>"``.
2. Write the integrator: a class with ``config``, a ``capabilities`` property and ``step_day(problem,
   water, plan) -> (water, DayDiagnostics)``. It runs each segment of the day (the day, or the parts
   before and after an event) on :meth:`~.problem.RichardsProblem.for_segment` of the problem and
   reads the physics only through it (residual,
   fluxes, Jacobian, sink cap, switch predicates, :meth:`~.problem.RichardsProblem.post_step` after
   every accepted sub-step) and returns the new ``h``, ``theta``, ``pond`` (and ``dt_next`` if it
   carries a step size) with ``water.flux`` untouched, plus the day's :class:`SubstepTotals` (the
   ``int q dt`` accumulators of both segments of an event day combined, :func:`combine_totals`) and
   its counters (a NamedTuple whose fields are ``SoilWaterFluxes`` diagnostics, or ``None``). With
   ``plan.event`` it runs the redistribution up to the event time it chooses, calls
   ``plan.event.apply`` once and continues to the end of the day. ``check(h, flux, where)`` raises
   under ``AGRI_JAX_CHECK=1`` when the day did not converge (the identity for a scheme without a
   convergence test).
3. Register it: ``register_integrator(IntegratorInfo(key=..., config_type=MySteps, factory=MyIntegrator,
   summary=..., sources=..., deviates=...))`` at import of its module (a plugin registers from its
   own package; nothing in the soil-water day changes).
4. Give it an integrator conformance case (:mod:`agrijax.testing.conformance.integrators`): its
   declared tolerances against a converged reference (Crank-Nicolson with the ``alpha = 1`` fallback
   and 960 sub-steps a day) on the frozen problem, the balance in both views, jit/vmap invariance,
   dtypes, and the gradient check when it declares gradients. The declared tolerances are measured,
   with the measurement named.

Design: what is solved (the physics layer) is kept apart from how it is advanced in time (the
integrator), so that a time scheme can be swapped without touching the physics.

Source: Hairer, E., Wanner, G., 1996. Solving Ordinary Differential Equations II, 2nd ed., Springer
(one-step methods for stiff semi-discrete problems).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, ClassVar, NamedTuple, Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.process import FAITHFUL, Deviation, ProcessKey, Source

from .coefficients import numerical_setting, rzwqm2
from .problem import RichardsProblem, SoilWater, SubstepTotals

__all__ = [
    "GRADIENTS",
    "INTEGRATORS",
    "SLOT",
    "Capabilities",
    "DayDiagnostics",
    "DayEvent",
    "DayPlan",
    "IntegratorInfo",
    "RichardsIntegrator",
    "SteppingConfig",
    "combine_totals",
    "integrator_for",
    "integrator_info",
    "register_integrator",
    "variant_problems",
]

#: slot of every Richards integrator
SLOT = "soil_water"
#: how an integrator is differentiated: not at all, by JAX through its operations, or by its own rule
GRADIENTS = ("none", "autodiff", "custom")
#: prefixes of the non-faithful variants of an integrator key (the closed vocabulary)
_VARIANT_PREFIXES = ("ref_", "alt_")

_ALPHA_FIRST: float = numerical_setting(
    "richards.alpha_first",
    1.0,
    "-",
    "time weight of the first sub-step of a day (fully implicit), and of every sub-step of "
    "time_scheme='implicit'",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/Rzrich.for:1653", "RICHRD", note="ALPH at the start of the day"),
)
_ALPHA_CN: float = numerical_setting(
    "richards.alpha_cn",
    0.5,
    "-",
    "time weight of the later sub-steps of time_scheme='rzwqm' (Crank-Nicolson, no water table)",
    origin="rzwqm2-4.6",
    provenance=rzwqm2("RZWQM/Rzrich.for:1655", "RICHRD", note="ALPH after the first step, ITBL != 1"),
)


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """What an integrator (with its config) supports.

    ``gradient`` one of :data:`GRADIENTS` (reverse mode); ``forward_mode`` whether ``jax.jvp`` works;
    ``dtypes`` the floating dtypes it runs in (float32 with x64 disabled); ``conserved`` the variable
    whose balance the method closes; ``batching`` how ``vmap`` behaves; ``convergence`` whether the
    day reports unconverged steps (``check`` raises on them); ``gradient_note`` where the gradient is
    valid (a custom rule exact only under a condition says which).
    """

    gradient: str
    dtypes: tuple[str, ...]
    batching: str
    forward_mode: bool = False
    conserved: str = "theta"
    convergence: bool = False
    gradient_note: str = ""

    def __post_init__(self) -> None:
        if self.gradient not in GRADIENTS:
            raise ValueError(f"gradient must be one of {GRADIENTS}, got {self.gradient!r}")
        if not self.dtypes or not self.batching.strip():
            raise ValueError("an integrator declares its dtypes and its batching behaviour")


class SteppingConfig(eqx.Module):
    """Base of the integrator configs (the tagged union of ``RichardsParams.stepping``).

    A subclass holds only static numerical settings and names its integrator by ``KEY``.
    """

    KEY: ClassVar[str] = ""


class DayEvent(NamedTuple):
    """The infiltration event of an event day, as the integrator sees it.

    ``ts0`` storm start [h] and ``has_event`` from the forcing; ``n_pre``, ``n_post``,
    ``post_grading`` the day's schedule (``DayConfig``: an integrator with fixed sub-steps places
    them by it; ``n_pre = 0`` places the event at ``t = 0`` for every integrator); ``apply(water)``
    runs the event on the state at the event time and returns ``(water, event result)``.
    """

    ts0: Array
    has_event: Array
    n_pre: int
    n_post: int
    post_grading: float
    apply: Callable[[SoilWater], tuple[SoilWater, Any]]


class DayPlan(NamedTuple):
    """What the day asks of the integrator besides the problem: an event, or none (a day without an event)."""

    event: DayEvent | None = None


class DayDiagnostics(NamedTuple):
    """The integrator's account of a day.

    ``totals`` the Richards sub-steps of the day (both segments combined), ``stats`` the counters
    (a NamedTuple of ``SoilWaterFluxes`` diagnostic fields, ``None`` when the integrator keeps none),
    ``event`` the event result and ``t_event`` the event time [h] (``None`` on a day without a plan
    event).
    """

    totals: SubstepTotals
    stats: Any = None
    event: Any = None
    t_event: Array | None = None


@runtime_checkable
class RichardsIntegrator(Protocol):
    """The protocol every Richards integrator implements (module docstring)."""

    config: Any

    @property
    def capabilities(self) -> Capabilities: ...

    def step_day(
        self, problem: RichardsProblem, water: SoilWater, plan: DayPlan
    ) -> tuple[SoilWater, DayDiagnostics]: ...

    def check(self, h: Array, flux: Any, where: str) -> Array: ...


@dataclasses.dataclass(frozen=True)
class IntegratorInfo:
    """Registry entry of an integrator: key, config type, factory and provenance."""

    key: str
    config_type: type[SteppingConfig]
    factory: Callable[[Any], RichardsIntegrator]
    summary: str
    sources: tuple[Source, ...] = ()
    deviates: tuple[Deviation, ...] = ()

    @property
    def parsed_key(self) -> ProcessKey:
        return ProcessKey.parse(self.key)


#: the registered integrators, by key
INTEGRATORS: dict[str, IntegratorInfo] = {}


def variant_problems(key: str) -> list[str]:
    """What is wrong with an integrator key: its slot and its variant vocabulary."""
    k = ProcessKey.parse(key)
    out = []
    if k.slot != SLOT:
        out.append(f"{key}: an integrator of the Richards problem has slot {SLOT!r}")
    if k.variant != FAITHFUL and not k.variant.startswith(_VARIANT_PREFIXES):
        out.append(f"{key}: variant must be 'faithful', 'ref_<switch>' or 'alt_<name>'")
    return out


def register_integrator(info: IntegratorInfo) -> IntegratorInfo:
    """Register an integrator (at import of its module); a re-import replaces its own entry.

    Raises on a malformed key, a variant outside the vocabulary, a non-faithful variant of a key with a
    reference whose ``faithful`` sibling is not registered, a key or a config type taken by another
    entry, or a config type whose ``KEY`` is not the entry's key.
    """
    _check_entry(info.key, info.config_type)
    INTEGRATORS[info.key] = info
    return info


def _check_entry(key: str, config_type: type) -> None:
    problems = variant_problems(key)
    k = ProcessKey.parse(key)
    own = getattr(config_type, "KEY", "")
    if own != key:
        problems.append(f"{key}: {config_type.__name__}.KEY is {own!r}")
    if k.needs_faithful_sibling and str(k.faithful) not in INTEGRATORS:
        problems.append(f"{key}: register {k.faithful} first")
    for other in INTEGRATORS.values():
        if other.key != key and other.config_type is config_type:
            problems.append(f"{key}: config type {config_type.__name__} taken by {other.key}")
    if problems:
        raise ValueError("; ".join(problems))


def integrator_info(what: str | SteppingConfig | type) -> IntegratorInfo:
    """The entry of a key, a config or a config type."""
    if isinstance(what, str):
        try:
            return INTEGRATORS[what]
        except KeyError:
            raise KeyError(f"no integrator {what!r} (registered: {sorted(INTEGRATORS)})") from None
    cls = what if isinstance(what, type) else type(what)
    for info in INTEGRATORS.values():
        if info.config_type is cls:
            return info
    raise KeyError(f"no integrator registered for config type {cls.__name__}")


def integrator_for(config: SteppingConfig) -> RichardsIntegrator:
    """The integrator of a stepping config (the tagged union's dispatch, static)."""
    return integrator_info(config).factory(config)


def combine_totals(pre: SubstepTotals | None, post: SubstepTotals) -> SubstepTotals:
    """The totals of an event day's two segments: ``post`` alone without a first segment, else the
    sums (the larger ``max_theta_residual``)."""
    if pre is None:  # static: no segment before the event
        return post
    names = [k for k in SubstepTotals._fields if k != "max_theta_residual"]
    summed = {k: getattr(pre, k) + getattr(post, k) for k in names}
    return SubstepTotals(
        **summed, max_theta_residual=jnp.maximum(pre.max_theta_residual, post.max_theta_residual)
    )
