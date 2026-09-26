"""pytest plugin of the conformance kit: parametrises the ``conformance_case`` fixture by registry key.

Load it with ``-p agrijax.testing.conformance.pytest_plugin`` (or ``pytest_plugins`` in a root
``conftest.py``). Options: ``--agrijax-key`` (glob over registry keys, default ``*``),
``--agrijax-package`` (only the cases of one distribution), ``--agrijax-no-builtin`` (only the
entry points' cases: a plugin testing itself). A test that takes ``conformance_case`` runs once per
selected case; :func:`run_check` runs one check honouring the case's exemptions.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from .case import ConformanceCase, ConformanceError
from .checks import check_name, check_parts
from .discover import discover, select

__all__ = ["run_check"]

X64_ENV = "AGRI_JAX_X64"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agrijax", "agrijax conformance kit")
    group.addoption("--agrijax-key", default="*", help="glob over registry keys, e.g. 'pet/*'")
    group.addoption("--agrijax-package", default=None, help="only the cases of this distribution")
    group.addoption(
        "--agrijax-no-builtin", action="store_true", default=False, help="skip this repository's own cases"
    )


def pytest_configure(config: pytest.Config) -> None:
    import jax

    x64 = os.environ.get(X64_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}
    jax.config.update("jax_enable_x64", x64)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "conformance_case" not in metafunc.fixturenames:
        return
    key = str(metafunc.config.getoption("--agrijax-key") or "*")
    cases = select(
        discover(builtin=not metafunc.config.getoption("--agrijax-no-builtin")),
        key=key,
        package=metafunc.config.getoption("--agrijax-package"),
    )
    if not cases:
        raise pytest.UsageError(f"no conformance case matches --agrijax-key {key!r}")
    metafunc.parametrize("conformance_case", cases, ids=[c.key for c in cases])


def run_check(case: ConformanceCase, check: Callable[[ConformanceCase], Any]) -> None:
    """Run ``check``; an exempted check must fail (reported as xfail) and fails the test when it
    passes, so a stale exemption cannot stay. A float32 exemption under x64
    (``case.exempt_float32_x64``) runs the float64 part first as an ordinary check (it must pass)
    and then the float32 part as an exempted one."""
    name = check_name(check)
    for label, reason, ctx in check_parts(case, name):
        at = f" ({label})" if label else ""
        if reason is None:
            with ctx:
                check(case)
            continue
        try:
            with ctx:
                check(case)
        except ConformanceError as e:
            pytest.xfail(f"exempt{at} ({reason}): {e}")
        raise ConformanceError(
            f"[{case.key}] {name}{at}: passes although exempt ({reason}); remove the exemption"
        )
