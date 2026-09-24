"""Shared pytest configuration.

* Enables ``jax_enable_x64`` before any test module imports JAX (set ``AGRI_JAX_X64=0``
  to run the unit tier in float32, as the CI second pass does).
* Adds ``--data-dir`` (default ``~/agri_jax_data``) and the ``data_dir`` fixture used by the
  diff / integration tiers, which skip when the data is absent.
* Applies the tier marker of each test's directory automatically.
* Keeps the unit tier data-free: a unit test that requests ``data_dir`` (directly or through
  another fixture) is a collection error (doc 05 section 4: the unit tier needs no data).
* Deselects ``@pytest.mark.slow`` tests by default (doc 05: the default run and the unit tier stay
  fast). They run with ``--runslow`` or ``AGRI_JAX_RUNSLOW=1``, whenever a ``-m`` expression is
  given (``pytest -m slow`` runs only them), or when a test is named explicitly by node id.
* Loads a deterministic hypothesis profile (``derandomize=True``, no example database), so every
  property test draws the same examples on every run and machine.
* With ``AGRI_JAX_NO_SKIP=1`` (set by the CI unit job) any skipped test fails the run, unless the
  test is marked ``@pytest.mark.allow_skip(reason=...)``; a silently skipped reader test would
  otherwise pass CI without running.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax
import pytest
from hypothesis import settings

settings.register_profile("agri_jax", derandomize=True, database=None, deadline=None)
settings.load_profile("agri_jax")

_X64 = os.environ.get("AGRI_JAX_X64", "1").strip().lower() not in {"0", "false", "no", "off"}
jax.config.update("jax_enable_x64", _X64)

DEFAULT_DATA_DIR = Path.home() / "agri_jax_data"
TIERS = ("unit", "diff", "integration", "gpu")
NO_SKIP_ENV = "AGRI_JAX_NO_SKIP"
RUNSLOW_ENV = "AGRI_JAX_RUNSLOW"
_DATA_FIXTURE = "data_dir"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--data-dir",
        action="store",
        default=os.environ.get("AGRI_JAX_DATA", str(DEFAULT_DATA_DIR)),
        help="root of the private data tree (scenarios, dumps, Fortran tools)",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help=f"also run @pytest.mark.slow tests (deselected by default; or set {RUNSLOW_ENV}=1)",
    )


def _run_slow(config: pytest.Config) -> bool:
    if config.getoption("--runslow"):
        return True
    if os.environ.get(RUNSLOW_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if config.getoption("markexpr", ""):  # an explicit -m expression decides by itself
        return True
    return any("::" in str(a) for a in config.args)  # a test named by node id


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", f"allow_skip(reason): this test may skip even under {NO_SKIP_ENV}=1 (give the reason)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark every test with the tier of its directory (tests/<tier>/...); keep the unit tier data-free."""
    root = Path(str(config.rootpath)) / "tests"
    offenders: list[str] = []
    for item in items:
        try:
            rel = Path(str(item.fspath)).relative_to(root)
        except ValueError:
            continue
        tier = rel.parts[0] if rel.parts else ""
        if tier in TIERS:
            item.add_marker(getattr(pytest.mark, tier))
        if tier == "unit" and _DATA_FIXTURE in getattr(item, "fixturenames", ()):
            offenders.append(item.nodeid)
    if not _run_slow(config):
        slow = [it for it in items if it.get_closest_marker("slow") is not None]
        if slow:
            config.hook.pytest_deselected(items=slow)
            items[:] = [it for it in items if it.get_closest_marker("slow") is None]
    if offenders:
        raise pytest.UsageError(
            "unit tests must not depend on the private data tree (fixture 'data_dir'); move them to "
            "tests/integration/: " + ", ".join(offenders)
        )


def _no_skip() -> bool:
    return os.environ.get(NO_SKIP_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


_unexpected_skips: list[str] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):  # type: ignore[no-untyped-def]
    outcome = yield
    report = outcome.get_result()
    if report.skipped and _no_skip() and item.get_closest_marker("allow_skip") is None:
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        _unexpected_skips.append(f"{item.nodeid}: {reason}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if _unexpected_skips and exitstatus == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:  # type: ignore[no-untyped-def]
    if _unexpected_skips:
        terminalreporter.section(f"{NO_SKIP_ENV}=1: unexpected skips (fail the run)", red=True)
        for line in _unexpected_skips:
            terminalreporter.line(line)


@pytest.fixture(scope="session")
def data_dir(request: pytest.FixtureRequest) -> Path:
    """Root of the private data tree; the test is skipped when it does not exist."""
    p = Path(request.config.getoption("--data-dir")).expanduser()
    if not p.is_dir():
        pytest.skip(f"data dir {p} not found (pass --data-dir or set AGRI_JAX_DATA)")
    return p


@pytest.fixture(scope="session")
def x64_enabled() -> bool:
    return bool(jax.config.jax_enable_x64)


@pytest.fixture(scope="session")
def tol(x64_enabled: bool) -> float:
    """Default relative tolerance for numerical comparisons: 1e-9 in x64, 1e-3 in float32."""
    return 1e-9 if x64_enabled else 1e-3
