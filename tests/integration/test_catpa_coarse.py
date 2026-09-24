"""Coarse comparison on CA-TPA 2015: PET and hydraulics modules against a fresh RZWQM2 run.

Drives ``poc/coarse_compare.py`` end to end: a one-year RZWQM2 run through
``agrijax.port.run_fortran.run_rzwqm`` (about 3 s), the S-W PET and ASCE reference ET modules on
three weather sources, an ablation of every input-timing convention, and the Brooks-Corey curves on
the ``LAYER.PLT`` pressure heads. Every tolerance below is an achieved value measured on 2026-09-23
(see ``<data>/validation/catpa_2015_coarse.md``) with a margin of about 2-3x, not a tuned target:

* primary run (prepared ``.MET``): PE RMSE 4.0e-4, max 3.0e-3; PT RMSE 6.6e-5, max 3.8e-4 mm/d;
* ``.ana``-weather run (module only): PE max 3.0e-3, PT max 3.9e-4 mm/d;
* the only unreproduced input is the reference's re-summed hourly solar radiation (a 0.993-1.005
  seasonal factor on the ``.MET`` value), whose effect is below 0.01 mm/d (known deviation).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax.port.run_fortran import RZWQM_BINARY, SYSTEM_LOADER, elf_interpreter

pytestmark = pytest.mark.slow

POC = Path(__file__).resolve().parents[2] / "poc" / "coarse_compare.py"

# achieved on 2026-09-23 (see the module docstring) with a 2-3x margin
PRIMARY_PE = dict(rmse=1.5e-3, max_abs=1.0e-2)
PRIMARY_PT = dict(rmse=3.0e-4, max_abs=1.5e-3)
MODULE_PE = dict(rmse=1.0e-3, max_abs=6.0e-3)
MODULE_PT = dict(rmse=2.0e-4, max_abs=1.0e-3)
SRAD_RESUM_EFFECT = 1.0e-2  # known deviation: re-summed hourly srad, mm/d on PE and PT
SRAD_RATIO = (0.99, 1.006)  # .ana col 88 / raw .MET srad over the year


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
def ref_dir(cc: ModuleType, catpa_scenario: Path, tmp_path_factory: pytest.TempPathFactory):
    if not RZWQM_BINARY.is_file():
        pytest.skip(f"RZWQM binary {RZWQM_BINARY} not found")
    interp = elf_interpreter(RZWQM_BINARY)
    if interp is not None and not Path(interp).exists() and not SYSTEM_LOADER.exists():
        pytest.skip("no ELF loader for the RZWQM binary")
    out = tmp_path_factory.mktemp("catpa_coarse")
    yield cc.reference_run(out / "ref", rerun=True, scenario=catpa_scenario)
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture(scope="module")
def result(cc: ModuleType, ref_dir: Path, catpa_scenario: Path):
    return cc.run_comparison(ref_dir, scenario=catpa_scenario)


def _err(result, run: str, var: str) -> pd.Series:
    return pd.Series((result.pet_sim[run][var] - result.pet_ref[var]).to_pandas()).abs()


def test_report_structure(result) -> None:
    md = result.markdown
    for heading in (
        "PET: prepared .MET weather",
        "PET: .ana-echoed weather",
        "Attribution: each convention",
        "Profile water storage",
        "Measured facts behind the conclusions",
    ):
        assert heading in md
    rep = result.pet["met"]
    assert rep.periods == ["year (met)", "season (met)"]
    for m in rep:
        assert m.n == (365 if m.period.startswith("year") else 151)
        assert np.isfinite([m.rmse, m.bias, m.r2]).all()


def test_forcing_facts(result) -> None:
    """Raw .MET vs the weather the reference model echoes: only the wind floor and the srad re-sum."""
    w = result.weather
    for v in ("tmin", "tmax", "rh"):
        assert w[v].max_abs < 1e-9
    f = result.facts
    lo, hi = f["srad_ratio_range"]
    assert SRAD_RATIO[0] < lo < hi < SRAD_RATIO[1]
    assert w["srad"].max_abs < 0.11
    # the raw wind differs from the echo exactly on the days below the 100 km/d floor
    assert f["wind_prepared_equals_ana"] is True
    assert len(f["wind_floor_days"]) == 3
    assert all(v < 100.0 for v in f["wind_floor_raw"])
    assert w["wind_run"].bias < 0.0 and w["wind_run"].max_abs < 40.0


def test_reference_et(result) -> None:
    ana = result.pet["ana"]
    met = result.pet["met"]
    for v in ("ref_et_tall_mm", "ref_et_short_mm"):
        assert ana[v, "year (ana)"].rmse < 1e-4  # REF_ET reproduced to print precision
        # prepared MET: the re-summed srad is the only difference left
        assert met[v, "year (met)"].rmse < 5e-3 and met[v, "year (met)"].max_abs < 2e-2


def test_shuttleworth_wallace_primary_run(result) -> None:
    """Prepared .MET forcing with the reference model's input timing: PE and PT at print precision."""
    met = result.pet["met"]
    for period in ("year (met)", "season (met)"):
        pe, pt = met["pot_evap_mm", period], met["pot_transp_mm", period]
        assert pe.rmse < PRIMARY_PE["rmse"] and pe.max_abs < PRIMARY_PE["max_abs"]
        assert pt.rmse < PRIMARY_PT["rmse"] and pt.max_abs < PRIMARY_PT["max_abs"]
        assert abs(pe.bias) < 5e-4 and abs(pt.bias) < 1e-4
    assert met["pet_mm", "year (met)"].r2 > 0.9999
    # no day above 0.01 mm in either flux, and the median PT error is zero (exact off-season)
    assert float(_err(result, "met", "pot_evap_mm").max()) < PRIMARY_PE["max_abs"]
    assert float(_err(result, "met", "pot_transp_mm").median()) == 0.0


def test_shuttleworth_wallace_module_only(result) -> None:
    """Identical forcing (.ana echo): what remains is the module itself, at print precision."""
    ana = result.pet["ana"]
    pe, pt = ana["pot_evap_mm", "year (ana)"], ana["pot_transp_mm", "year (ana)"]
    assert pe.rmse < MODULE_PE["rmse"] and pe.max_abs < MODULE_PE["max_abs"]
    assert pt.rmse < MODULE_PT["rmse"] and pt.max_abs < MODULE_PT["max_abs"]
    e_pe = _err(result, "ana", "pot_evap_mm")
    # the residual is the residue mass before the day's decomposition (col 72 is after it):
    # tiny, and confined to days with residue and exposed soil
    assert float(e_pe.quantile(0.95)) < 1.5e-3
    f = result.facts
    assert f["pe_err_after_harvest"] < 1e-3 and f["pe_err_before_planting"] < 5e-3


def test_srad_resummation_is_the_only_unreproduced_input(result) -> None:
    """Known deviation: the reference re-sums an hourly disaggregation of srad (Rzmain.for 1384-1389)."""
    se = result.facts["srad_effect_on_pet"]
    assert 0.0 < se["pot_evap_mm"] < SRAD_RESUM_EFFECT
    assert se["pot_transp_mm"] < SRAD_RESUM_EFFECT
    # and it shows up as the srad-driven part of the primary run's residual
    e_met = _err(result, "met", "pot_evap_mm")
    e_ana = _err(result, "ana", "pot_evap_mm")
    assert float((e_met - e_ana).abs().max()) < SRAD_RESUM_EFFECT


def test_wind_floor_is_the_whole_raw_met_gap(result) -> None:
    """Without INPDAY's 100 km/d floor the raw .MET run is off on exactly the three floor days."""
    e_pt = _err(result, "met-raw", "pot_transp_mm")
    floor = pd.DatetimeIndex(result.facts["wind_floor_days"])
    on = e_pt.index.isin(floor)
    assert float(e_pt[on].max()) > 0.5  # 2015-07-28: 0.749 mm/d
    assert float(e_pt[~on].max()) < PRIMARY_PT["max_abs"]
    off_days = pd.DatetimeIndex(e_pt.index)[e_pt.to_numpy() > 0.01]
    assert list(off_days) == list(floor)


ABLATION_EXPECTATIONS = {
    # label: (worst day, at least this error when the convention is removed)
    "same-row LAI and height": ("2015-09-14", 1.0),
    "canopy kept the day after harvest": ("2015-09-26", 2.0),
    "Jan 1 residue from .ana row 0": ("2015-01-01", 0.2),
    "residue type kept after harvest": (None, 0.1),
    "residue age not reset at harvest": ("2015-09-26", 0.05),
    "tillage-day residue from the previous row": ("2015-04-14", 0.05),
    "raw .MET wind (no 100 km/d floor)": ("2015-07-28", 0.5),
}


@pytest.mark.parametrize("label", list(ABLATION_EXPECTATIONS))
def test_attribution_of_former_outliers(result, label: str) -> None:
    """Removing one input-timing convention brings back a named outlier the primary run does not have."""
    a = result.ablations[label]
    day, at_least = ABLATION_EXPECTATIONS[label]
    assert a.max_abs > at_least, (label, a.max_abs)
    assert a.primary_on_worst_day < 2e-3, (label, a.worst_day, a.primary_on_worst_day)
    if day is not None:
        assert a.worst_day == pd.Timestamp(day), (label, a.worst_day)
    else:  # residue type: a plateau over the weeks after harvest, not a single day
        _, harvest = result.season
        assert a.worst_day > harvest
        after = a.errors[(a.errors.index > harvest) & (a.errors.index <= harvest + pd.Timedelta(days=60))]
        assert float(after.mean()) > 0.1


def test_harness_reads_the_scenario_not_hard_coded_facts(
    result, cc: ModuleType, catpa_scenario: Path
) -> None:
    f = result.facts
    assert f["tillage_dates"] == [pd.Timestamp("2015-04-14")]
    assert f["residue_type_before"] == "soybean" and f["residue_type_after"] == "corn"
    rc = f["residue_conditions"]
    assert rc["mass_t_ha"] == 2.0 and rc["age_d"] == 50.0 and rc["cover_factor"] == 2.5
    assert rc["stem_area_index"] == 0.0 and rc["standing_mass_t_ha"] == 0.0
    dat = cc.read_rzwqm_dat(catpa_scenario / "rzwqm.dat")
    assert cc.residue_cover_factor(dat) == 2.5


def test_gradients_finite_over_the_whole_year(
    result, cc: ModuleType, ref_dir: Path, catpa_scenario: Path
) -> None:
    """d(PT, PE_soil, PE_residue)/d(every PETParams field and every continuous input), every day of 2015."""
    inp = cc.load_inputs(ref_dir, catpa_scenario)
    x = cc.pet_inputs(inp)
    names = ("transpiration", "soil_evaporation", "residue_evaporation")
    w = x.weather

    def fluxes(params, tmin, tmax, srad, rh, wind, lai, height, theta, rm, age, rdia, rho, doy):
        r = cc.sw_day(
            x, tmin, tmax, srad, rh, wind, lai, height, theta, rm, age, rdia, rho, doy, params=params
        )
        return jnp.stack([getattr(r, n) for n in names])

    args = (
        w["tmin"],
        w["tmax"],
        w["srad"],
        w["rh"],
        w["wind_run"],
        x.lai,
        x.height,
        x.theta,
        x.residue_mass,
        x.residue_age,
        x.residue_diameter_cm,
        x.residue_density,
    )
    n_in = len(args)
    for k, name in enumerate(names):

        def scalar(params, *a, _k=k):
            return fluxes(params, *a)[_k]

        axes = (None, *([0] * (n_in + 1)))
        grads = jax.vmap(jax.grad(scalar, argnums=tuple(range(n_in + 1))), in_axes=axes)(
            x.params, *args, x.doy
        )
        gp, gin = grads[0], grads[1:]
        for field in (
            "albedo_dry",
            "albedo_wet",
            "albedo_maturity",
            "albedo_residue",
            "soil_resistance",
            "stomatal_resistance",
        ):
            g = np.asarray(getattr(gp, field))
            assert g.shape == (365,) and np.all(np.isfinite(g)), (name, field)
        for j, g in enumerate(gin):
            assert np.all(np.isfinite(np.asarray(g))), (name, j)
        if name == "transpiration":
            # physics sign checks: more stomatal resistance, less PT on every canopy day; more
            # radiation, more PT under a closed canopy (LAI >= 1.5). Under a sparse canopy the
            # sign can flip (extra radiation raises soil evaporation, lowers the source-height
            # deficit D0 and with it the small canopy flux: the S-W 1985 coupling), so the sparse
            # days are only required to be small in magnitude
            season = np.asarray(x.lai > 0.0) & np.asarray(x.height > 0.0)
            closed = season & np.asarray(x.lai >= 1.5)
            g_srad = np.asarray(gin[2])
            assert np.all(np.asarray(gp.stomatal_resistance)[season] <= 0.0)
            assert closed.sum() > 60 and np.all(g_srad[closed] > 0.0)
            assert np.all(np.abs(g_srad[season & ~closed]) < 1e-3)  # cm d-1 per MJ m-2 d-1
        if name == "soil_evaporation":
            assert np.all(np.asarray(gp.soil_resistance) <= 0.0)


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
    tillage = result.facts["tillage_dates"][0]
    bad_days = result.facts["theta_bad_days"]
    assert bad_days and all(tillage < d <= tillage + pd.Timedelta(days=10) for d in bad_days)
