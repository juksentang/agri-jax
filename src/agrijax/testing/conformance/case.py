"""Data structures of the conformance kit: a case, its conserved quantities and its gradient spec.

A :class:`ConformanceCase` is everything the generic checks need to exercise one registered
process (one registry key) without data: a ``make`` function that builds synthetic inputs from a
NumPy :class:`~numpy.random.Generator` (never ``jax.random``, whose numbers change with the x64
switch), the module binding (own subtree and ports) and what to expect (balances, gradients,
tolerances). Every field that switches a check off needs a written reason.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agrijax.core.process import Process, ProcessKey, lookup

__all__ = [
    "CALIBRATABLE",
    "DEFAULT_OWN",
    "DTYPE_SPLIT_CHECKS",
    "Balance",
    "ConformanceCase",
    "ConformanceError",
    "GradSpec",
    "Inputs",
    "Tolerance",
]

#: ``GradSpec.wrt`` entry meaning "every calibratable coefficient of the case's coefficient sets"
CALIBRATABLE = "<calibratable>"

#: default global path of a module's own subtree, by slot (``{slot}`` is the crop slot)
DEFAULT_OWN: dict[str, str] = {
    "crop": "crops.{slot}",
    "water_supply": "water_supply.{slot}",
    "n_supply": "n_supply.{slot}",
    "soil_water": "soil_water",
    "pet": "surface.pet",
    "snow": "surface.snow",
}

#: ``(state0, params, forcing)``; every forcing leaf has the time axis first
Inputs = tuple[Any, Any, Any]


class ConformanceError(AssertionError):
    """A conformance check failed (the message names the case, the check and what differs)."""


@dataclass(frozen=True)
class Tolerance:
    """``|a - b| <= atol + rtol * |b|``."""

    rtol: float
    atol: float

    def close(self, a: Any, b: Any) -> bool:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return bool(np.all(np.abs(a - b) <= self.atol + self.rtol * np.abs(b)))


@dataclass(frozen=True)
class Balance:
    """A conserved quantity: storage and the day's inflow and outflow, in the same unit.

    ``storage(state, params)`` is the stock at the end of a day; ``inflow`` and ``outflow`` are
    ``(before, after, params, forcing_t) -> amount`` of one day. The check asks
    ``|dS - (in - out)| <= atol + rtol * scale`` every day, with ``scale`` the stock plus the
    gross flows.
    """

    quantity: str
    unit: str
    storage: Callable[[Any, Any], Any]
    inflow: Callable[[Any, Any, Any, Any], Any]
    outflow: Callable[[Any, Any, Any, Any], Any]
    tol64: Tolerance = Tolerance(1e-12, 1e-10)
    tol32: Tolerance = Tolerance(1e-6, 1e-5)


@dataclass(frozen=True)
class GradSpec:
    """What to differentiate and how the finite-difference check is run.

    ``wrt`` are parameter paths; :data:`CALIBRATABLE` expands to every calibratable coefficient
    of the case's coefficient sets. ``loss`` maps the final state to a scalar (default: the sum of
    every floating leaf the process writes). The finite-difference check runs in gradient mode
    ``mode`` along random directions, with steps ``h`` and ``h / 10``; when the two disagree the
    point sits on a kink and the case must move it or exempt the parameter (``fd_exempt``: path
    and reason, e.g. a straight-through gradient). ``edge_variants`` are extra variants checked
    for finite gradients only (LAI = 0, dry soil, saturation).
    """

    wrt: tuple[str, ...] = (CALIBRATABLE,)
    loss: Callable[[Any], Any] | None = None
    mode: str = "exact"
    fd_rel_step: float = 1e-6
    fd_tol: Tolerance = Tolerance(1e-4, 1e-9)
    fd_exempt: tuple[tuple[str, str], ...] = ()
    edge_variants: tuple[str, ...] = ()
    fd_directions: int = 2


#: checks that run float64 and float32 separately (``ConformanceCase.exempt_float32_x64``)
DTYPE_SPLIT_CHECKS = ("balance", "precision", "grad_finite")


@dataclass(frozen=True)
class ConformanceCase:
    """One registry key and the synthetic inputs that exercise it (M3 coupling contract, section 4).

    Parameters
    ----------
    key:
        Registry key ``slot/impl@ref_version:variant``; the process is ``lookup(key)`` unless
        ``process`` is given (a kit fixture or an unregistered wrapper).
    make:
        ``(rng, dtype, variant) -> (state0, params, forcing)``. Must be deterministic in ``rng``;
        the forcing carries ``n_days`` days. ``dtype`` is the floating dtype of every input.
    variants:
        Input variants every check runs on (the first is the nominal one).
    own, ports, crop_slot:
        The module binding: the global path of the module's own subtree (default
        :data:`DEFAULT_OWN` of the slot) and ``{port field: global path}``; ``{slot}`` in either
        is replaced by ``crop_slot``.
    slot_contract:
        The slot contract to check against (default: the key's slot; ``None``: none, give
        ``no_slot_contract``).
    balances / no_balance, grad / no_grad:
        Conserved quantities and the gradient spec; an empty one needs the written reason.
    f32:
        Tolerance of float32 against float64 on the same inputs.
    transforms_exact:
        ``vmap(jit)`` must equal per-sample ``jit`` bit for bit (else within 4 ulp).
    transforms_tol / transforms_why:
        ``(float64, float32)`` tolerances that replace the ulp criterion of ``vmap(jit)`` against
        ``jit`` and the relative one of eager against ``jit``, for a process whose result is a
        long iteration (a fixed-count Newton solver, say) in which the batched and the eager
        programs round differently; the reason, with the measured differences, is required.
        Batch independence stays bit for bit.
    binding_exact:
        The replay binding must equal the coupled one bit for bit (else within 4 ulp: the two
        programs are compiled separately and XLA may round a reduction differently).
    forcing_fields:
        The forcing fields the process reads (listed in its docstring); when given, every other
        forcing leaf is perturbed and the output must not change.
    coefficient_sets:
        Parameter paths of the :class:`~agrijax.core.coefficients.Coefficients` sets the process
        uses (default: every set found in the parameters).
    exempt_checks:
        ``{check name: reason}``: the check is run and must fail (an exemption that passes is an
        error, so it is removed as soon as the gap is closed).
    exempt_float32_x64:
        ``{check name: reason}`` for a gap that exists only for float32 inputs with x64 enabled
        (an implicit upcast, say): the float32 part of the check is run and must fail (xfail, and
        an error once it passes), the float64 part is run as an ordinary check and must pass.
        Only the checks with a per-dtype part (:data:`DTYPE_SPLIT_CHECKS`); with x64 disabled
        float32 is the only dtype and the check runs unexempted.
    origin:
        Distribution and version that provides the case (filled by discovery).
    """

    key: str
    make: Callable[[np.random.Generator, Any, str], Inputs]
    variants: tuple[str, ...] = ("nominal",)
    n_days: int = 3
    process: Process | None = None
    slot_contract: str | None = ""
    no_slot_contract: str = ""
    own: str = ""
    ports: Mapping[str, str] = field(default_factory=dict)
    crop_slot: str = "maize"
    balances: tuple[Balance, ...] = ()
    no_balance: str = ""
    grad: GradSpec | None = GradSpec()
    no_grad: str = ""
    batch: int = 3
    f32: Tolerance = Tolerance(1e-4, 1e-6)
    transforms_exact: bool = True
    transforms_tol: tuple[Tolerance, Tolerance] | None = None
    transforms_why: str = ""
    binding_exact: bool = True
    forcing_fields: tuple[str, ...] = ()
    coefficient_sets: tuple[str, ...] | None = None
    exempt_checks: Mapping[str, str] = field(default_factory=dict)
    exempt_float32_x64: Mapping[str, str] = field(default_factory=dict)
    origin: str = ""
    seed: int = 0

    def __post_init__(self) -> None:
        ProcessKey.parse(self.key)  # raises on a malformed key
        if not self.variants:
            raise ValueError(f"{self.key}: at least one variant")
        if self.n_days < 1 or self.batch < 2:
            raise ValueError(f"{self.key}: n_days >= 1 and batch >= 2 required")
        if not self.balances and not self.no_balance.strip():
            raise ValueError(f"{self.key}: no balances and no reason (no_balance)")
        if self.grad is None and not self.no_grad.strip():
            raise ValueError(f"{self.key}: grad=None needs a reason (no_grad)")
        if self.slot_contract is None and not self.no_slot_contract.strip():
            raise ValueError(f"{self.key}: slot_contract=None needs a reason (no_slot_contract)")
        if self.transforms_tol is not None and not self.transforms_why.strip():
            raise ValueError(f"{self.key}: transforms_tol needs a reason (transforms_why)")
        for name, why in self.exempt_checks.items():
            if not str(why).strip():
                raise ValueError(f"{self.key}: exemption of {name!r} without a reason")
        for name, why in self.exempt_float32_x64.items():
            if name not in DTYPE_SPLIT_CHECKS:
                raise ValueError(
                    f"{self.key}: exempt_float32_x64 of {name!r}: "
                    f"only {DTYPE_SPLIT_CHECKS} have a float32 part"
                )
            if name in self.exempt_checks:
                raise ValueError(f"{self.key}: {name!r} is in both exempt_checks and exempt_float32_x64")
            if not str(why).strip():
                raise ValueError(f"{self.key}: float32 exemption of {name!r} without a reason")

    # ------------------------------------------------------------------ derived
    @property
    def parsed_key(self) -> ProcessKey:
        return ProcessKey.parse(self.key)

    @property
    def slot(self) -> str:
        return self.parsed_key.slot

    @property
    def proc(self) -> Process:
        """The process under test: ``process`` or the registered ``lookup(key)``."""
        return self.process if self.process is not None else lookup(self.key)

    @property
    def contract_name(self) -> str | None:
        return self.slot if self.slot_contract == "" else self.slot_contract

    @property
    def own_path(self) -> str:
        own = self.own or DEFAULT_OWN.get(self.slot, f"surface.{self.slot}")
        return own.format(slot=self.crop_slot)

    @property
    def port_map(self) -> dict[str, str]:
        return {k: v.format(slot=self.crop_slot) for k, v in self.ports.items()}

    def inputs(self, dtype: Any, variant: str | None = None, sample: int = 0) -> Inputs:
        """``make`` with the case's deterministic generator for ``sample`` (0 = nominal sample)."""
        v = self.variants[0] if variant is None else variant
        rng = np.random.default_rng([self.seed, sample])
        return self.make(rng, dtype, v)
