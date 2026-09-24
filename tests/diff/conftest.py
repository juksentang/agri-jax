"""Differential-test tier: every test needs Fortran dumps under ``<data-dir>/dumps``; skipped when absent.

Dumps are the ``npz`` cases written by :mod:`agrijax.port.dumps` (``dumps/<ROUTINE>/<case>.npz``).
Tests that take the ``dump_case`` argument are parametrised over every case of the routines in
:data:`ROUTINES`, or of those named by ``@pytest.mark.routines(...)`` (one skipped placeholder
per routine without dumps), so the tier collects the
same way with or without data. The reference runs the cases came from are kept next to them in
``dumps/_runs/<run>/`` (model outputs of the instrumented run, e.g. ``CA-TPA.ana``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agrijax.port import dumps as _dumps

ROUTINES: dict[str, str] = {
    "RICHRD": "rzwqm",
    "SINK": "rzwqm",
    "WC": "rzwqm",
    "POINTK": "rzwqm",
    "POTEVPHR": "rzwqm",
    "MZ_PHENOL": "ceres_maize",
    "MZ_GROSUB": "ceres_maize",
    "MZ_ROOTGR": "ceres_maize",  # CERES-Maize root growth (file MZ_ROOTS.for)
}
"""Instrumented routines -> name of their Fortran index (``port_index/<name>.json``)."""


def _data_root(config: pytest.Config) -> Path:
    return Path(str(config.getoption("--data-dir"))).expanduser()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "routines(*names): parametrise dump_case over these routines only")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "dump_case" not in metafunc.fixturenames:
        return
    root = _data_root(metafunc.config) / "dumps"
    mark = metafunc.definition.get_closest_marker("routines")
    wanted = list(mark.args) if mark is not None else list(ROUTINES)
    params = []
    for r in wanted:
        paths = _dumps.list_cases(root, r)
        if not paths:
            params.append(
                pytest.param(
                    None,
                    id=f"{r}-no-dumps",
                    marks=[
                        pytest.mark.skip(reason=f"no dumps for {r} under {root}"),
                        pytest.mark.allow_skip(reason="dumps are private data"),
                    ],
                )
            )
        else:
            params += [pytest.param(p, id=f"{r}-{p.stem}") for p in paths]
    metafunc.parametrize("dump_case", params)


@pytest.fixture(scope="session")
def dumps_dir(data_dir: Path) -> Path:
    p = data_dir / "dumps"
    if not p.is_dir():
        pytest.skip(f"no Fortran dumps at {p}")
    return p


@pytest.fixture(scope="session")
def index_entries(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Fortran index records of the instrumented routines (``port_index/*.json``), by name."""
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(set(ROUTINES.values())):
        p = data_dir / "port_index" / f"{name}.json"
        if p.is_file():
            for e in json.loads(p.read_text())["subroutines"]:
                out[str(e["name"]).upper()] = e
    return out
