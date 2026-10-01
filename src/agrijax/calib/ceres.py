"""CERES-Maize adapter of the calibration harness: cultivar parameter specs and batch simulators.

The crop runs as in the CERES-Maize comparison with DSSAT-CSM
(``tests/integration/test_ceres_dssat.py``): nitrogen off, the water port replayed from the
reference run (``ceres_water_replay``). A **batch group** is a set of treatments with the same soil
layering; their forcings are padded to a common length and the model runs as one ``vmap`` over the
treatments (:func:`stack_treatments`, :func:`group_simulator`). Treatments with different layer
counts go in different groups.

Bounds of the cultivar coefficients are the ``MINIMA`` / ``MAXIMA`` rows of the DSSAT-CSM
v4.8.6.0 cultivar file ``Genotype/MZCER048.CUL`` (lines 51-52: P1 5-450, P2 0-2, P5 580-999,
G2 248-990, G3 5-16.5, PHINT 38-75); the ecotype file ``MZCER048.ECO`` has no such rows, so the
RUE bounds are a harness choice (see :data:`CERES_SPECS`). Every spec also declares the valid
domain the crop equations need (e.g. ``G2 > 0``: it divides the kernel number).

**Gradient mode.** A simulator is built for one gradient mode and keeps it:
:func:`group_simulator` takes the mode as a required argument, so
``jax.jit(group_simulator(..., mode=m))`` differentiates in mode ``m`` wherever it is called, and
calling it inside an explicit ``gradient_mode`` context of another mode raises
:class:`~agrijax.core.grad.ModeMismatchError` (a loop over modes that reuses one simulator fails
loudly instead of running the first mode every time). Build one simulator, and one jit
(``sim.jit()``), per mode.

**Gradient or derivative-free, per parameter** (the defaults in code):
the phenology parameters :data:`CERES_DERIVATIVE_FREE` (P1, P2, P5, PHINT) are calibrated
derivative-free by default (their event gradients are experimental; P2 and a low-temperature end
of the season may carry no local signal at all); the growth parameters (G2, G3, RUE) use
gradients where the gradient-trust report passes, pair by pair (:func:`ceres_gradient_plan`): a
(parameter, treatment) pair with jumps or zero-derivative stretches falls back to central secants.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.grad import ModeBound, bind_gradient_mode
from agrijax.core.runtime import run
from agrijax.core.units import KG_HA_PER_G_M2
from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model
from agrijax.processes.crop.ceres_maize.constants import YRDOY_SCALE
from agrijax.sites.dssat_inputs import yrdoy_range

from .space import ParamSpace, ParamSpec, non_negative, positive
from .trust import GradientPlan, gradient_plan

__all__ = [
    "CERES_DERIVATIVE_FREE",
    "CERES_SPECS",
    "ISTAGE_MATURITY_OUT",
    "ISTAGE_SILKING_OUT",
    "CeresTreatment",
    "calib_outputs",
    "ceres_gradient_plan",
    "ceres_space",
    "group_simulator",
    "pad_forcing",
    "stack_treatments",
]

_CUL = "DSSAT-CSM v4.8.6.0 Genotype/MZCER048.CUL:51-52 (MINIMA / MAXIMA rows)"
_CM = "agrijax.processes.crop.ceres_maize"

#: valid domains: where the crop equations are defined for each coefficient
_DOMAIN = {
    "P1": non_negative(
        f"{_CM}.phenology juvenile_block: thermal time, SUMDTT >= P1 and SUMDTT / P1 (safe_div)"
    ),
    "P2": non_negative(
        f"{_CM}.phenology: photoperiod sensitivity; DJTI + P2 (TWILEN - P2O) is floored at DEN_MIN "
        "before 1 / (.)"
    ),
    "P5": positive(f"{_CM}.growth grain_fill_growth and phenology: divides SUMDTT (SUMDTT / P5)"),
    "G2": positive(f"{_CM}.phenology grain_number: divides GPP (GPP / G2, GPP / (0.15 G2), GPP / (0.5 G2))"),
    "G3": non_negative(f"{_CM}.growth grain_fill_rate: a potential kernel growth rate"),
    "PHINT": positive(f"{_CM}.growth leaf_appearance and phenology: divides DTT and SUMDTT (TI, TLNO)"),
    "RUE": non_negative(f"{_CM}.growth: radiation use efficiency (PCARB = IPAR RUE PCO2)"),
}

#: the calibratable CERES-Maize cultivar / ecotype coefficients with their bounds
CERES_SPECS: dict[str, ParamSpec] = {
    "P1": ParamSpec("cultivar.p1", 5.0, 450.0, "degC d", _CUL, label="P1", domain=_DOMAIN["P1"]),
    "P2": ParamSpec("cultivar.p2", 0.0, 2.0, "d h-1", _CUL, label="P2", domain=_DOMAIN["P2"]),
    "P5": ParamSpec("cultivar.p5", 580.0, 999.0, "degC d", _CUL, label="P5", domain=_DOMAIN["P5"]),
    "G2": ParamSpec("cultivar.g2", 248.0, 990.0, "kernel plant-1", _CUL, label="G2", domain=_DOMAIN["G2"]),
    "G3": ParamSpec("cultivar.g3", 5.0, 16.5, "mg kernel-1 d-1", _CUL, label="G3", domain=_DOMAIN["G3"]),
    "PHINT": ParamSpec("cultivar.phint", 38.0, 75.0, "degC d", _CUL, label="PHINT", domain=_DOMAIN["PHINT"]),
    "RUE": ParamSpec(
        "cultivar.rue",
        3.0,
        5.5,
        "g MJ-1",
        "harness choice: MZCER048.ECO has no MINIMA/MAXIMA rows; the file's ecotypes span 3.9-4.5 "
        "(IB0001 4.2, IB0004 '+5% RUE' 4.4), the box is wider on both sides",
        label="RUE",
        domain=_DOMAIN["RUE"],
    ),
}

#: parameters calibrated derivative-free by default: the stage parameters, whose event gradients
#: (straight-through estimates of the derivative through the event day, ``ste`` mode) are an
#: experimental option; P5's closer agreement with a smoothed finite-difference reference does not
#: open it automatically. Pass ``allow_gradient`` to :func:`ceres_gradient_plan` to try one.
CERES_DERIVATIVE_FREE: tuple[str, ...] = ("P1", "P2", "P5", "PHINT")

#: first-day stage codes of the phenology dates in the model output ``istage`` (as in
#: ``test_ceres_dssat.check_m2``: ADAT = first day of ISTAGE 4, MDAT = first day of ISTAGE 10)
ISTAGE_SILKING_OUT = 4
ISTAGE_MATURITY_OUT = 10


def ceres_space(names: Sequence[str]) -> ParamSpace:
    """:class:`ParamSpace` of the named :data:`CERES_SPECS`."""
    return ParamSpace([CERES_SPECS[n] for n in names])


def calib_outputs(state: CeresMaizeState, params: Any, forcing_t: Any) -> dict[str, Array]:
    """The daily outputs a calibration reads: LAI, CWAD, GWAD [kg ha-1] and the stage code
    (same expressions as ``plantgro_outputs``)."""
    g = state.growth
    return {
        "lai": g.lai,
        "cwad": g.biomas * KG_HA_PER_G_M2,
        "gwad": g.grnwt * state.phen.ears * KG_HA_PER_G_M2,
        "istage": state.phen.istage,
    }


@dataclass
class CeresTreatment:
    """One treatment: its parameters and forcing (from the reference run) and metadata."""

    name: str
    params: Any
    forcing: Any
    meta: Mapping[str, Any]


def pad_forcing(forcing: Any, n_days: int) -> Any:
    """``forcing`` extended to ``n_days`` days by repeating its last day (weather, soil water,
    EOP, TRWUP) with consecutive dates. Only for calibration batches: the padded days lie after
    the reference run's last day (maturity or harvest), so the reference season is unchanged."""
    t = int(np.shape(forcing.yrdoy)[0])
    if n_days < t:
        raise ValueError(f"cannot pad a {t}-day forcing to {n_days} days")
    if n_days == t:
        return forcing
    extra = n_days - t

    def pad(x: Any) -> Any:
        x = jnp.asarray(x)
        if x.ndim == 0 or x.shape[0] != t:
            return x
        return jnp.concatenate([x, jnp.repeat(x[-1:], extra, axis=0)], axis=0)

    out = jax.tree_util.tree_map(pad, forcing)
    last = int(np.asarray(forcing.yrdoy)[-1])
    days = yrdoy_range(last, _add_days(last, extra))
    yr = jnp.concatenate(
        [jnp.asarray(forcing.yrdoy), jnp.asarray(days[1:], dtype=jnp.asarray(forcing.yrdoy).dtype)]
    )
    return out.replace(yrdoy=yr)


def _add_days(yrdoy: int, n: int) -> int:
    d = date(yrdoy // YRDOY_SCALE, 1, 1) + timedelta(days=yrdoy % YRDOY_SCALE - 1 + n)
    return d.year * YRDOY_SCALE + d.timetuple().tm_yday


def stack_treatments(treatments: Sequence[CeresTreatment], n_days: int | None = None) -> tuple[Any, Any, Any]:
    """``(params, forcing, state0)`` of a batch group, each leaf with a leading treatment axis.

    Forcings are padded to ``n_days`` (default: the longest). All treatments need the same
    layer count."""
    nl = {int(np.shape(t.params.soil.dlayr)[-1]) for t in treatments}
    if len(nl) != 1:
        raise ValueError(f"a batch group needs one soil layering, got layer counts {sorted(nl)}")
    t_max = max(int(np.shape(t.forcing.yrdoy)[0]) for t in treatments)
    n = t_max if n_days is None else int(n_days)
    forc = [pad_forcing(t.forcing, n) for t in treatments]
    states = [CeresMaizeState.initial(t.params, 1) for t in treatments]

    def stack(*xs: Any) -> Any:
        return jnp.stack([jnp.asarray(x) for x in xs])

    params = jax.tree_util.tree_map(stack, *[t.params for t in treatments])
    forcing = jax.tree_util.tree_map(stack, *forc)
    state0 = jax.tree_util.tree_map(stack, *states)
    return params, forcing, state0


def group_simulator(
    space: ParamSpace,
    params: Any,
    forcing: Any,
    state0: Any,
    *,
    mode: str,
    outputs: Callable[..., dict[str, Array]] = calib_outputs,
) -> ModeBound[[Array], dict[str, Array]]:
    """``theta -> {output: [T, B]}``: the model on every treatment of the group (one ``vmap``),
    with the parameters of ``space`` set to ``theta`` in every treatment.

    The simulator is bound to the gradient ``mode`` (required: ``"exact"``, ``"ste"`` or
    ``"implicit"``, see :func:`agrijax.core.grad.bind_gradient_mode`); its ``.mode`` attribute
    says which, and calling it inside an explicit ``gradient_mode`` context of another mode
    raises :class:`agrijax.core.grad.ModeMismatchError`. Jit it per mode with its ``jit`` method
    (which keeps the check on every call): ``{m: group_simulator(..., mode=m).jit() for m in modes}``."""
    model = ceres_maize_model(outputs=outputs)

    def one(theta: Array, p: Any, f: Any, s0: Any) -> dict[str, Array]:
        out = run(model, space.apply(p, theta), f, s0)
        return {k: v[..., 0] for k, v in out.items()}  # drop the crop axis (n_crop = 1)

    batched = jax.vmap(one, in_axes=(None, 0, 0, 0), out_axes=1)

    def simulate(theta: Array) -> dict[str, Array]:
        return batched(theta, params, forcing, state0)

    return bind_gradient_mode(simulate, mode)


def ceres_gradient_plan(
    report: Mapping[str, Any],
    treatments: Mapping[str, Sequence[str]] | None = None,
    *,
    allow_gradient: Sequence[str] = (),
) -> GradientPlan:
    """:func:`agrijax.calib.trust.gradient_plan` with the CERES defaults: the parameters of
    :data:`CERES_DERIVATIVE_FREE` derivative-free unless named in ``allow_gradient``, the others
    by the trust report, pair by pair."""
    df = tuple(n for n in CERES_DERIVATIVE_FREE if n not in set(allow_gradient))
    return gradient_plan(report, treatments, derivative_free=df)
