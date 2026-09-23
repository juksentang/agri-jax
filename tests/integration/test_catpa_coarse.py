"""First coarse comparison on CA-TPA 2015: PET and hydraulics modules against a fresh RZWQM2 run.

Drives ``poc/coarse_compare.py`` end to end: a one-year RZWQM2 run through
``agri_jax.port.run_fortran.run_rzwqm`` (about 3 s), the S-W PET and ASCE reference ET modules on
the ``.MET`` forcing, and the Brooks-Corey curves on the ``LAYER.PLT`` pressure heads. The
assertions are regression guards on facts measured on 2026-09-23 (see
``<data>/validation/catpa_2015_coarse.md``), deliberately loose where the gap is understood but
not yet closed; they are not tuned targets.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from agri_jax.port.run_fortran import RZWQM_BINARY, SYSTEM_LOADER, elf_interpreter

pytestmark = pytest.mark.slow

POC = Path(__file__).resolve().parents[2] / "poc" / "coarse_compare.py"


def _load_poc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("coarse_compare", POC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["coarse_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cc() -> ModuleType:
    return _load_poc()


@pytest.fixture(scope="module")
def result(cc: ModuleType, catpa_scenario: Path, tmp_path_factory: pytest.TempPathFactory):
    if not RZWQM_BINARY.is_file():
        pytest.skip(f"RZWQM binary {RZWQM_BINARY} not found")
    interp = elf_interpreter(RZWQM_BINARY)
    if interp is not None and not Path(interp).exists() and not SYSTEM_LOADER.exists():
        pytest.skip("no ELF loader for the RZWQM binary")
    out = tmp_path_factory.mktemp("catpa_coarse")
    ref_dir = cc.reference_run(out / "ref", rerun=True, scenario=catpa_scenario)
    res = cc.run_comparison(ref_dir, scenario=catpa_scenario)
    yield res
    shutil.rmtree(out, ignore_errors=True)


def test_report_structure(result) -> None:
    md = result.markdown
    for heading in ("PET: MET weather", "Profile water storage", "Measured facts behind the gaps"):
        assert heading in md
    rep = result.pet["met"]
    assert rep.periods == ["year (met)", "season (met)"]
    for m in rep:
        assert m.n == (365 if m.period.startswith("year") else 151)
        assert np.isfinite([m.rmse, m.bias, m.r2]).all()


def test_met_forcing_matches_ana_echo(result) -> None:
    w = result.weather
    for v in ("tmin", "tmax", "rh"):
        assert w[v].max_abs < 1e-9
    # srad differs by a few hundredths of MJ (reference-model preprocessing, not understood yet)
    assert w["srad"].max_abs < 0.2
    # wind differs only on the days where RZWQM floors the wind run at 100 km/d
    assert w["wind_run"].bias < 0.0 and w["wind_run"].max_abs < 40.0


def test_reference_et(result) -> None:
    ana = result.pet["ana"]
    met = result.pet["met"]
    for v in ("ref_et_tall_mm", "ref_et_short_mm"):
        assert ana[v, "year (ana)"].rmse < 1e-4  # REF_ET reproduced to print precision
        assert met[v, "year (met)"].rmse < 0.1  # MET forcing: the wind-floor days dominate


def test_shuttleworth_wallace_coarse(result) -> None:
    met = result.pet["met"]
    assert met["pet_mm", "year (met)"].rmse < 0.2
    assert met["pet_mm", "season (met)"].r2 > 0.97
    assert abs(met["pet_mm", "year (met)"].bias) < 0.05
    # with the start-of-day canopy (previous .ana row) PT is exact on every season day except the
    # day after harvest (canopy removed before the PET call) and the wind-floor days
    start, end = result.season
    pt_ref = result.pet_ref["pot_transp_mm"].to_pandas()
    err = (result.pet_sim["met-lag1"]["pot_transp_mm"].to_pandas() - pt_ref).abs()
    days = err.index
    skip = (days == end + pd.Timedelta(days=1)) | days.isin(
        pd.to_datetime(["2015-07-27", "2015-07-28", "2015-09-03"])
    )
    season = (days >= start) & (days <= end) & ~skip
    assert float(err[season].max()) < 1e-2


def test_profile_storage_sanity(result) -> None:
    st = result.storage
    assert st["storage_cm", "year"].rmse < 0.01  # theta_of_h(h) on the 37 nodes vs col 2
    assert st["storage_cm", "year (LAYER.PLT theta, io check)"].max_abs < 1e-3
    lo, hi = result.storage_range
    u = result.uniform
    assert u["wilting point h = -15000 cm"] < lo < hi < u["saturation (h = 0)"]
    assert u["residual (theta_r)"] < u["wilting point h = -15000 cm"] < u["field capacity h = -333 cm"]
    # the curves reproduce LAYER.PLT theta everywhere except just after tillage in horizon 1
    bad = result.theta_err_by_depth
    assert float(bad["max_abs_theta_err"].max()) < 5e-3
    assert set(bad.loc[bad["max_abs_theta_err"] > 1e-4, "depth_cm"]) <= {1.0, 2.0, 4.0, 7.0, 11.0, 15.0}
