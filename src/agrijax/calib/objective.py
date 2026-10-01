"""Calibration objectives: observed quantities, their model counterparts and a weighted loss.

A **simulator** is any function ``theta -> outputs`` where ``outputs`` is a dict of daily model
outputs, each ``[T, B]`` (days, treatments of one batch group) or ``[T, B, n_crop]`` (the crop
axis is summed away when it has size 1). A :class:`Target` turns the outputs into the
observed quantity:

``"final"``
    the value on the last simulated day (season yield, final biomass), one number per treatment;
``"series"``
    the value on the observation days (``mask[T, B]`` true), e.g. LAI or biomass samplings;
``"date"``
    the index of the first day a stage code is reached (``output`` is an integer stage code,
    ``code`` the stage), e.g. silking or maturity. The index is an integer: its derivative with
    respect to any parameter is 0 (the model's stage machine is discrete); a gradient-free or
    secant method is needed for the parameters that act only through dates.

The loss of one target is the mean of ``((sim - obs) / scale)^2`` over its observations; the
objective is the weighted sum over targets and batch groups. ``scale`` is the observation
standard deviation (absolute), or ``rel_scale * |obs|`` when ``rel_scale`` is given. The
objective splits exactly into per-treatment terms (:meth:`Objective.treatment_losses`: a final or
date target contributes ``r_b^2 / B`` of treatment ``b``, a series target its masked squares over
the target's observation count), which is what a per-(parameter, treatment) gradient-trust report
and :func:`agrijax.calib.optim.pair_gradient` work on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.coefficients import numerical_guard

__all__ = [
    "Group",
    "Objective",
    "Simulator",
    "Target",
    "evaluate_targets",
    "first_day_index",
    "synthetic_observations",
]

Simulator = Callable[[Array], Mapping[str, Array]]

#: floor of a target's observation scale (keeps the normalised residual finite)
_SCALE_FLOOR = numerical_guard("calib.scale_floor", 1e-12, "floor of an observation scale")


def _squeeze_crop(x: Array) -> Array:
    x = jnp.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    return x


def first_day_index(stage: Array, code: int) -> Array:
    """Index (float) of the first day ``stage == code`` along axis 0; ``T`` if never reached."""
    s = _squeeze_crop(stage)
    hit = s == code
    t = s.shape[0]
    idx = jnp.argmax(hit, axis=0)
    return jnp.where(jnp.any(hit, axis=0), idx, t).astype(float)


@dataclass(frozen=True)
class Target:
    """One observed quantity (see the module docstring for ``kind``)."""

    name: str
    output: str
    kind: Literal["final", "series", "date"]
    scale: float = 1.0
    rel_scale: float | None = None
    weight: float = 1.0
    code: int | None = None
    #: series observation days, ``[T, B]`` booleans (``kind="series"`` only)
    mask: Any = None

    def __post_init__(self) -> None:
        if self.kind == "date" and self.code is None:
            raise ValueError(f"target {self.name}: a date target needs the stage code")
        if self.kind == "series" and self.mask is None:
            raise ValueError(f"target {self.name}: a series target needs the observation mask")

    def value(self, outputs: Mapping[str, Array]) -> Array:
        """The simulated quantity: ``[B]`` (final, date) or ``[T, B]`` (series, masked later)."""
        x = outputs[self.output]
        if self.kind == "final":
            return _squeeze_crop(x)[-1]
        if self.kind == "date":
            return first_day_index(x, int(self.code))  # type: ignore[arg-type]
        return _squeeze_crop(x)


def evaluate_targets(targets: Sequence[Target], outputs: Mapping[str, Array]) -> dict[str, Array]:
    """``{target name: simulated quantity}``: what synthetic observations are made from."""
    return {t.name: t.value(outputs) for t in targets}


@dataclass(frozen=True)
class Group:
    """One batch group: a simulator (one ``vmap`` over its treatments) and its observations."""

    name: str
    simulate: Simulator
    targets: tuple[Target, ...]
    observed: Mapping[str, Any]
    #: names of the group's treatments (batch axis order), for reports; default ``0, 1, ...``
    treatments: tuple[str, ...] = ()

    def treatment_names(self, n: int) -> tuple[str, ...]:
        """``"<group>/<treatment>"`` for the ``n`` treatments of the batch."""
        names = self.treatments or tuple(str(b) for b in range(n))
        if len(names) != n:
            raise ValueError(f"group {self.name}: {len(names)} treatment names for {n} treatments")
        return tuple(f"{self.name}/{t}" for t in names)

    def residuals(self, theta: Array) -> dict[str, Array]:
        """Normalised residuals ``(sim - obs) / scale`` per target (series: 0 off the mask)."""
        out = self.simulate(theta)
        res = {}
        for t in self.targets:
            obs = jnp.asarray(self.observed[t.name])
            sim = t.value(out)
            if t.rel_scale is not None:
                scale = jnp.maximum(t.rel_scale * jnp.abs(obs), _SCALE_FLOOR)
            else:
                scale = jnp.maximum(jnp.asarray(t.scale, dtype=float), _SCALE_FLOOR)
            r = (sim - obs) / scale
            if t.kind == "series":
                r = jnp.where(jnp.asarray(t.mask), r, 0.0)
            res[t.name] = r
        return res

    def treatment_losses(self, theta: Array) -> Array:
        """``[B]``: the weighted loss of the group split by treatment (sums to the group's part
        of :class:`Objective`)."""
        res = self.residuals(theta)
        total = None
        for t in self.targets:
            r = res[t.name]
            if t.kind == "series":
                n = jnp.maximum(jnp.sum(jnp.asarray(t.mask)), 1)
                per = jnp.sum(r**2, axis=0) / n
            else:
                per = r**2 / r.shape[0]
            total = t.weight * per if total is None else total + t.weight * per
        if total is None:
            raise ValueError(f"group {self.name} has no targets")
        return total

    def target_losses(self, theta: Array) -> dict[str, Array]:
        """Mean squared normalised residual per target."""
        res = self.residuals(theta)
        out = {}
        for t in self.targets:
            r = res[t.name]
            if t.kind == "series":
                n = jnp.maximum(jnp.sum(jnp.asarray(t.mask)), 1)
                out[t.name] = jnp.sum(r**2) / n
            else:
                out[t.name] = jnp.mean(r**2)
        return out


@dataclass(frozen=True)
class Objective:
    """Weighted sum of target losses over batch groups, as a function of ``theta``."""

    groups: tuple[Group, ...]
    weights: Mapping[str, float] = field(default_factory=dict)

    def breakdown(self, theta: Array) -> dict[str, Array]:
        """``{"<group>/<target>": loss}``."""
        out = {}
        for g in self.groups:
            for name, v in g.target_losses(theta).items():
                out[f"{g.name}/{name}"] = v
        return out

    def treatment_losses(self, theta: Array) -> Array:
        """``[sum of B]``: the per-treatment losses of every group, concatenated in group order
        (names: :meth:`treatment_names`)."""
        return jnp.concatenate([jnp.atleast_1d(g.treatment_losses(theta)) for g in self.groups])

    def treatment_names(self, theta: Array) -> tuple[str, ...]:
        """Names of :meth:`treatment_losses` (evaluates the model once for the batch sizes)."""
        out: list[str] = []
        for g in self.groups:
            out.extend(g.treatment_names(int(np.shape(g.treatment_losses(theta))[0])))
        return tuple(out)

    def __call__(self, theta: Array) -> Array:
        total = jnp.asarray(0.0)
        for g in self.groups:
            losses = g.target_losses(theta)
            for t in g.targets:
                total = total + t.weight * losses[t.name]
        return total


def synthetic_observations(
    simulate: Simulator, targets: Sequence[Target], theta_true: Array
) -> dict[str, np.ndarray]:
    """Noise-free twin observations: the targets of the model run at ``theta_true``."""
    vals = evaluate_targets(targets, simulate(jnp.asarray(theta_true)))
    return {k: np.asarray(v) for k, v in vals.items()}
