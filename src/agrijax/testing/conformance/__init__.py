"""The conformance kit: one command checks that a slot implementation can join an assembly.

Any implementation of a slot, a community plugin included, provides a
:class:`ConformanceCase` (synthetic inputs from a NumPy generator, no data) and runs every generic
check with::

    python -m agrijax.testing.conformance --key 'pet/*'           # this repository's cases
    python -m agrijax.testing.conformance --package my-agrijax-soil --no-builtin

The checks (M3 coupling contract, section 4.4; :data:`CHECKS`, in this order): registry metadata
and provenance, the three-rules lint with AJ007/AJ008 on the process module and its same-package
imports, coefficient labels, declared dims and shapes, units (and the port records against the
contract, unit string for unit string), the slot contract (reads and writes inside the own
subtree and the slot's ports), the writes under ``AGRI_JAX_CHECK=1``, the reads by perturbation,
conservation (:class:`Balance`), ``jit`` / ``vmap`` and batch independence, float32 against
float64, finite gradients, gradients against finite differences with a two-step kink test, and
replay against coupled binding. They are plain functions that raise :class:`ConformanceError`;
:mod:`.pytest_plugin` parametrises them by registry key.

A plugin registers its cases under the entry point group ``agrijax.conformance`` (a module with
``cases()`` or a callable). The kit guarantees that a module can join an assembly without
breaking it (level 0 of the validation levels); comparison against a reference model stays in the
integration tier.
"""

from __future__ import annotations

from .case import CALIBRATABLE, Balance, ConformanceCase, ConformanceError, GradSpec, Tolerance
from .checks import (
    CHECKS,
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
from .contracts import SLOT_CONTRACTS, SlotContract, SlotPort, slot_contract
from .discover import ENTRY_POINT_GROUP, EXEMPT, builtin_cases, discover, load_provider, select

__all__ = [
    "CALIBRATABLE",
    "CHECKS",
    "ENTRY_POINT_GROUP",
    "EXEMPT",
    "SLOT_CONTRACTS",
    "Balance",
    "ConformanceCase",
    "ConformanceError",
    "GradSpec",
    "SlotContract",
    "SlotPort",
    "Tolerance",
    "builtin_cases",
    "check_balance",
    "check_binding",
    "check_coefficients",
    "check_grad_fd",
    "check_grad_finite",
    "check_lint",
    "check_name",
    "check_precision",
    "check_reads",
    "check_registry",
    "check_shapes_dims",
    "check_slot_contract",
    "check_transforms",
    "check_units",
    "check_writes",
    "discover",
    "lint_scope",
    "load_provider",
    "run_checks",
    "select",
    "slot_contract",
]
