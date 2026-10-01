"""The gradient checks of the RZWQM2-side cases (Richards line) are slow tests.

``grad_finite`` and ``grad_fd`` of every ``...@rzwqm2-...`` case differentiate whole multi-day soil-water
days and take 20 to 190 s each on a CI runner, most of the unit tier's time. They are marked
``slow`` (run with ``--runslow``, on the cluster); the main-line (DSSAT-CSM) cases and every other
check of the RZWQM2-side cases stay in the unit tier.
"""

from __future__ import annotations

import pytest

_SLOW_CHECKS = ("grad_finite", "grad_fd")


def pytest_itemcollected(item: pytest.Item) -> None:
    callspec = getattr(item, "callspec", None)
    if item.originalname != "test_package_case" or callspec is None:  # type: ignore[attr-defined]
        return
    case_id = item.name
    if "@rzwqm2-" in case_id and any(f"-{c}]" in case_id for c in _SLOW_CHECKS):
        item.add_marker(pytest.mark.slow)
