"""Expected values of the user's local scenario data, kept outside the repository.

Tests that pin a reader's output to known values of the local scenario files read those values
from ``<data-dir>/catpa/expected/<test module>.json`` instead of hard-coding them: the scenario
files (and so their values) are not public. A missing file skips the test like missing data
(the modules carry ``allow_skip``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

EXPECTED_DIR = Path("catpa/expected")


def expected(data_dir: Path, module: str) -> dict[str, Any]:
    """The expected values of test module ``module`` (its file stem) under ``data_dir``."""
    p = Path(data_dir) / EXPECTED_DIR / f"{Path(module).stem}.json"
    if not p.is_file():
        pytest.skip(f"{p} not found (expected values of the local scenario)")
    return json.loads(p.read_text(encoding="utf-8"))
