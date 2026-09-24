"""Integration tier: whole-model comparison on scenario directories under ``<data-dir>/narval_mirror``.

Also the home of every test that reads the private data tree or runs a Fortran oracle (the unit
tier must not). Oracle runs are staged into private ``tmp_path_factory`` directories, never into
a shared ``<data-dir>/run/<name>``, so concurrent pytest processes cannot race on them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

CATPA = Path("narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario")

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
DSCSM = Path(
    os.environ.get("AGRI_JAX_DSCSM", str(DSSAT_ENGINE / "source" / "build486" / "bin" / "dscsm048"))
).expanduser()
UFGA_FILEX = DSSAT_ENGINE / "example_data" / "Maize" / "UFGA8201.MZX"


@pytest.fixture(scope="session")
def catpa_scenario(data_dir: Path) -> Path:
    p = data_dir / CATPA
    if not (p / "rzwqm.dat").is_file():
        pytest.skip(f"CA-TPA scenario not found at {p}")
    return p


@pytest.fixture(scope="session")
def stage_ufga8201() -> Callable[[Path], Path]:
    """Return ``stage(run_dir)``: copy everything ``dscsm048`` needs for UFGA8201 into ``run_dir``."""
    if not DSCSM.is_file():
        pytest.skip(f"dscsm048 binary not found at {DSCSM}")
    if not UFGA_FILEX.is_file():
        pytest.skip(f"{UFGA_FILEX} not found")
    from agrijax.io.dssat import stage_run_dir

    def stage(run_dir: Path) -> Path:
        return stage_run_dir(
            UFGA_FILEX,
            run_dir,
            data_dir=DSSAT_ENGINE / "source" / "Data",
            weather_dirs=[DSSAT_ENGINE / "example_data" / "Weather"],
            soil_dirs=[DSSAT_ENGINE / "example_data" / "Soil"],
            binary=DSCSM,
        )

    return stage


@pytest.fixture(scope="session")
def ufga_run(tmp_path_factory: pytest.TempPathFactory, stage_ufga8201: Callable[[Path], Path]) -> Path:
    """One UFGA8201 run of ``dscsm048`` per session, in a private temporary directory."""
    from agrijax.io.dssat import run_dssat

    run_dir = stage_ufga8201(tmp_path_factory.mktemp("ufga8201"))
    r = run_dssat(run_dir, UFGA_FILEX.name)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (run_dir / "Summary.OUT").is_file(), r.stdout[-2000:]
    return run_dir
