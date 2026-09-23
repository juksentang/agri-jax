"""Differential-test tier: every test needs Fortran dumps under ``<data-dir>/dumps``; skipped when absent."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def dumps_dir(data_dir: Path) -> Path:
    p = data_dir / "dumps"
    if not p.is_dir():
        pytest.skip(f"no Fortran dumps at {p}")
    return p
