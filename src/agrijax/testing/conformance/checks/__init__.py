"""The generic checks of the conformance kit.

Each ``check_x(case)`` is a plain function of a :class:`~.case.ConformanceCase`: it returns
``None`` or raises :class:`~.case.ConformanceError` naming what differs. No check needs pytest or
data; every input comes from the case's ``make`` with a NumPy generator. :data:`CHECKS` lists them
in the order they run; :func:`run_checks` runs them outside pytest.

The checks run the process over the case's ``n_days`` with ``lax.scan`` (the runtime's day loop),
except where a check needs concrete values (the eager runs of :func:`check_writes` and
:func:`check_transforms`). Float64 parts run when JAX has x64 enabled; with ``AGRI_JAX_X64=0``
the float32 parts still run and the float64 comparisons are left to the float64 pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..case import ConformanceCase, ConformanceError
from ._common import check_parts, dtype_part
from .binding import check_binding
from .gradients import check_grad_fd, check_grad_finite
from .metadata import check_coefficients, check_lint, check_registry
from .metadata import lint_scope as lint_scope  # re-exported: the kit's __init__ imports it
from .runs import check_balance, check_precision, check_reads, check_transforms, check_writes
from .structure import check_shapes_dims, check_slot_contract, check_units

__all__ = [
    "CHECKS",
    "check_balance",
    "check_binding",
    "check_coefficients",
    "check_grad_fd",
    "check_grad_finite",
    "check_lint",
    "check_parts",
    "check_precision",
    "check_reads",
    "check_registry",
    "check_shapes_dims",
    "check_slot_contract",
    "check_transforms",
    "check_units",
    "check_writes",
    "dtype_part",
    "run_checks",
]


#: every check, in the order the kit runs them
CHECKS: tuple[Callable[[ConformanceCase], None], ...] = (
    check_registry,
    check_lint,
    check_coefficients,
    check_shapes_dims,
    check_units,
    check_slot_contract,
    check_writes,
    check_reads,
    check_balance,
    check_transforms,
    check_precision,
    check_grad_finite,
    check_grad_fd,
    check_binding,
)


def check_name(check: Callable[..., Any]) -> str:
    """``check_reads`` -> ``reads`` (the name used by ``exempt_checks`` and the pytest ids)."""
    return check.__name__.removeprefix("check_")


def run_checks(
    case: ConformanceCase, checks: Sequence[Callable[[ConformanceCase], None]] = CHECKS
) -> dict[str, str | None]:
    """Run ``checks`` on ``case`` outside pytest: ``{check name: None if it passed, else the
    message}``. An exempted check (``case.exempt_checks``) that fails reports ``"exempt: ..."``;
    one that passes is reported as a failure (the exemption is stale). A float32 exemption under
    x64 (``case.exempt_float32_x64``) runs the float64 part unexempted and the float32 part
    exempted, each message prefixed with its part."""
    out: dict[str, str | None] = {}
    for check in checks:
        n = check_name(check)
        failures: list[str] = []  # a part that fails unexempted, or an exempt part that passes
        expected: list[str] = []  # an exempt part that fails
        for label, why, ctx in check_parts(case, n):
            at = f"{label}: " if label else ""
            try:
                with ctx:
                    check(case)
            except ConformanceError as e:
                (failures if why is None else expected).append(
                    f"{at}{e}" if why is None else f"{at}exempt: {why}: {e}"
                )
            else:
                if why is not None:
                    failures.append(f"{at}exempt check passes; remove the exemption ({why})")
        out[n] = "; ".join(failures) if failures else ("; ".join(expected) if expected else None)
    return out
