"""The generic conformance test: every selected case against every check.

Run with ``python -m agrijax.testing.conformance`` or ``pytest --pyargs
agrijax.testing.conformance.test_conformance -p agrijax.testing.conformance.pytest_plugin``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agrijax.testing.conformance.case import ConformanceCase
from agrijax.testing.conformance.checks import CHECKS, check_name
from agrijax.testing.conformance.pytest_plugin import run_check


@pytest.mark.parametrize("check", CHECKS, ids=check_name)
def test_conformance(conformance_case: ConformanceCase, check: Callable[[ConformanceCase], None]) -> None:
    run_check(conformance_case, check)
