"""Round trip through the real RZWQM2 binary: our writer -> Fortran -> our .ana / OVERVIEW readers.

One-year (2015) CA-TPA runs in ``<data-dir>/run/pyt_<random>/<name>/``: a directory unique to
this pytest process (``tempfile.mkdtemp``), so concurrent runs never share or delete each
other's run directories, and short enough for the 80-character Fortran path buffers. It is
removed at the end of the module. Skipped when the scenario, the binary or the system loader
is absent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from agri_jax.io.rzwqm import (
    key_variables,
    params_from_dat,
    params_to_dat,
    read_ana,
    read_met,
    read_overview_yields,
    read_rzwqm_dat,
    set_value,
    write_rzwqm_dat,
)

pytestmark = pytest.mark.slow

TOOL = Path("narval_mirror/RZWQM_Tool")
LOADER = Path("/lib64/ld-linux-x86-64.so.2")
IPNAMES_FILES = (
    "cntrl.dat",
    "rzwqm.dat",
    "CA-TPA.MET",
    "CA-TPA.BRK",
    "rzinit.dat",
    "plgen.dat",
    "CA-TPA.sno",
    "CA-TPA.ana",
)
DATES_2015 = "1  1  2015  31  12  2015"


def _prepare_run_dir(scenario: Path, tool: Path, run_dir: Path) -> Path:
    if len(str(run_dir)) >= 80:
        pytest.skip(f"run dir path too long for RZWQM ({len(str(run_dir))} chars): {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    shutil.copytree(scenario, run_dir)
    shutil.copytree(tool / "DSSAT", run_dir / "DSSAT", dirs_exist_ok=True)
    shutil.copy2(tool / "main_ryzen5_avx512", run_dir / "main_ryzen5_avx512")
    shutil.copy2(tool / "DSSAT" / "MODEL.ERR", run_dir / "main_ryzenMODEL.ERR")

    ip = (run_dir / "IPNAMES.DAT").read_bytes().decode("latin-1").split("\r\n")
    for i, name in enumerate(IPNAMES_FILES):
        ip[i] = str(run_dir / name)
    ip[8] = DATES_2015
    (run_dir / "IPNAMES.DAT").write_bytes("\r\n".join(ip).encode("latin-1"))

    rzx = (run_dir / "MZDSSAT.RZX").read_bytes().decode("latin-1").split("\r\n")
    k = next(i for i, ln in enumerate(rzx) if "DATABASE FILE LOCATIONS" in ln)
    rzx[k + 2] = f"{run_dir}/DSSAT/"
    rzx[k + 3] = f"{run_dir}/"
    (run_dir / "MZDSSAT.RZX").write_bytes("\r\n".join(rzx).encode("latin-1"))
    return run_dir


def _run(run_dir: Path) -> None:
    env = dict(os.environ, FORT_BUFFERED="TRUE")
    with open(run_dir / "run.log", "wb") as log:
        subprocess.run(
            [str(LOADER), "./main_ryzen5_avx512"],
            cwd=run_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
    if not (run_dir / "CA-TPA.ana").is_file():
        tail = (run_dir / "run.log").read_text(errors="ignore")[-2000:]
        raise AssertionError(f"RZWQM produced no .ana in {run_dir}; log tail:\n{tail}")


@pytest.fixture(scope="module")
def tool(data_dir: Path) -> Path:
    t = data_dir / TOOL
    if not (t / "main_ryzen5_avx512").is_file() or not LOADER.exists():
        pytest.skip("RZWQM binary or system loader not available")
    return t


@pytest.fixture(scope="module")
def run_root(data_dir: Path) -> Iterator[Path]:
    """A run root private to this pytest process, under ``<data-dir>/run`` (short absolute path)."""
    parent = data_dir / "run"
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="pyt_", dir=parent))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def baseline(catpa_scenario: Path, tool: Path, run_root: Path) -> Path:
    """One-year run of the unmodified scenario."""
    rd = _prepare_run_dir(catpa_scenario, tool, run_root / "base")
    _run(rd)
    return rd


def test_ana_and_overview_from_real_run(baseline: Path, catpa_scenario: Path) -> None:
    ds = read_ana(baseline / "CA-TPA.ana")
    assert ds.sizes["time"] == 366  # 2015.000 (initial state) + 365 days
    assert len(ds.data_vars) == 138  # 139 header columns incl. the time column
    assert str(ds.time.values[0])[:10] == "2014-12-31" and str(ds.time.values[-1])[:10] == "2015-12-31"
    assert ds["stored_soil_water"].attrs["units"] == "CM"
    assert ds["actual_et"].attrs["column"] == 84

    k = key_variables(ds)
    assert (k.aet_cm[1:] <= k.pet_cm[1:] + 1e-6).all()
    assert (k.evap_cm >= 0).all() and (k.transp_cm >= 0).all() and (k.lai >= 0).all()
    np.testing.assert_allclose(k.aet_cm[1:], (k.evap_cm + k.transp_cm)[1:], atol=2e-3)

    met = read_met(catpa_scenario / "CA-TPA.MET")
    rain = met.rain_mm.reindex(k.time.values[1:]).to_numpy()
    np.testing.assert_allclose(10.0 * k.precip_cm.values[1:], rain, atol=0.25)
    assert abs(10.0 * float(k.precip_cm[1:].sum()) - rain.sum()) < 0.005 * rain.sum()

    ov = read_overview_yields(baseline / "OVERVIEW.OUT")
    assert ov.season.tolist() == [1]
    assert str(ov.planting_date[0])[:10] == "2015-04-28"
    assert str(ov.harvest_date[0])[:10] == "2015-09-25"
    assert ov.yield_kg_ha[0] == pytest.approx(float(k.grain_kg_ha.max()), abs=0.5)
    assert 5000 < ov.yield_kg_ha[0] < 15000


def test_rewritten_dat_reproduces_baseline(
    baseline: Path, catpa_scenario: Path, tool: Path, run_root: Path
) -> None:
    """Every hydraulic value written back (GenerateDat formatting, LF endings) -> identical .ana."""
    dat = read_rzwqm_dat(catpa_scenario / "rzwqm.dat")
    new = params_to_dat(dat, params_from_dat(dat), decimals=None)
    assert new.lines != dat.lines  # the hydraulic lines really were rewritten
    rd = _prepare_run_dir(catpa_scenario, tool, run_root / "same")
    write_rzwqm_dat(new, rd / "rzwqm.dat", newline="\n")
    _run(rd)
    a = read_ana(baseline / "CA-TPA.ana")
    b = read_ana(rd / "CA-TPA.ana")
    for v in a.data_vars:
        np.testing.assert_array_equal(a[v].values, b[v].values, err_msg=v)


def test_perturbed_dat_changes_outputs(
    baseline: Path, catpa_scenario: Path, tool: Path, run_root: Path
) -> None:
    """A write-back that matters: halving Ksat and raising the soil resistance changes the run."""
    dat = read_rzwqm_dat(catpa_scenario / "rzwqm.dat")
    p = params_from_dat(dat)
    p["ksat"] = p["ksat"] * 0.5
    new = params_to_dat(dat, p)
    addr = dat.pet_addresses()["soil_resistance"]
    new = set_value(new, *addr, 400.0)
    assert new.pet["soil_resistance"] == 400.0
    rd = _prepare_run_dir(catpa_scenario, tool, run_root / "pert")
    write_rzwqm_dat(new, rd / "rzwqm.dat")
    _run(rd)
    base = key_variables(read_ana(baseline / "CA-TPA.ana"))
    pert = key_variables(read_ana(rd / "CA-TPA.ana"))
    assert float(pert.evap_cm.sum()) < float(base.evap_cm.sum())
    assert not np.array_equal(pert.profile_water_cm.values, base.profile_water_cm.values)
