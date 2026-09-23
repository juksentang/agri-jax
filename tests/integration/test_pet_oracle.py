"""PET kernels against the RZWQM2 ``.ana`` output of the canonical CA-TPA reference run.

The reference run is ``<data-dir>/catpa/ref_2015`` (one year, the shipped Scenario unchanged,
built by ``scripts/data/make_catpa_refs.py``); the site, PET and plant parameters are read from
the Scenario ``rzwqm.dat`` that run used, through :func:`agri_jax.io.rzwqm.read_rzwqm_dat`.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agri_jax.io.rzwqm import prepare_rzwqm_forcing, read_ana, read_met, read_rzwqm_dat
from agri_jax.io.rzwqm.layers import read_layer_output
from agri_jax.processes.pet import PETParams, asce_reference_et, shuttleworth_wallace

REF_RUN = Path("catpa/ref_2015")


def _block_values(text: str, marker: str) -> list[float]:
    """First data line (not starting with '=') after the header line containing ``marker``."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if marker in ln:
            for nxt in lines[i + 1 :]:
                s = nxt.strip()
                if s and not s.startswith("="):
                    return [float(t) for t in s.split()]
    raise KeyError(marker)


@pytest.fixture(scope="module")
def catpa_run(data_dir: Path, catpa_scenario: Path):
    run = data_dir / REF_RUN
    if not (run / "CA-TPA.ana").is_file():
        pytest.skip(f"{run}/CA-TPA.ana missing (run scripts/data/make_catpa_refs.py)")
    ds = read_ana(run / "CA-TPA.ana")
    cols = {int(k): v for k, v in ds.attrs["columns"].items()}
    dat_path = catpa_scenario / "rzwqm.dat"
    dat = read_rzwqm_dat(dat_path)
    phys, pet, plant = dat.physiography, dat.pet, dat.plant_site_params[0]
    hyd = dat.hydraulics
    cres = _block_values(dat_path.read_text(encoding="latin-1"), "C:P ratio of dominate residue material")[2]
    doy = pd.DatetimeIndex(ds.time.values).dayofyear.values
    return dict(
        run=run,
        ds=ds,
        col=lambda c: ds[cols[c]].values,
        doy=doy,
        elevation=float(phys["elevation_m"]),
        latitude=float(phys["latitude_rad"]),
        zone=int(phys["rainfall_zone"]),
        wc13=float(hyd["theta_fc33"][0]),
        wc15=float(hyd["theta_wp"][0]),
        cres=float(cres),
        params=PETParams(
            albedo_dry=jnp.asarray(pet["albedo_dry"]),
            albedo_wet=jnp.asarray(pet["albedo_wet"]),
            albedo_maturity=jnp.asarray(pet["albedo_crop"]),
            albedo_residue=jnp.asarray(pet["albedo_residue"]),
            soil_resistance=jnp.asarray(pet["soil_resistance"]),
            stomatal_resistance=jnp.asarray(plant["rs_min"]),
        ),
        wind_height=float(pet["wind_height_m"]),
        ipet=int(pet["pet_method"]),
    )


def test_prepared_met_wind_equals_ana(catpa_run, catpa_scenario: Path) -> None:
    """RZWQM floors the wind run at 100 km/d (INPDAY, UBREEZ); the prepared .MET equals .ana col 90."""
    c = catpa_run["col"]
    days = pd.DatetimeIndex(catpa_run["ds"].time.values)[1:]  # row 0 is the YYYY.000 initial state
    raw = read_met(catpa_scenario / "CA-TPA.MET")
    met = prepare_rzwqm_forcing(raw)
    np.testing.assert_array_equal(met.loc[days, "wind_run_km"].to_numpy(), c(90)[1:])
    np.testing.assert_array_equal(met.loc[days, "tmin"].to_numpy(), c(85)[1:])
    np.testing.assert_array_equal(met.loc[days, "tmax"].to_numpy(), c(86)[1:])
    np.testing.assert_array_equal(met.loc[days, "rh"].to_numpy(), c(89)[1:])
    # the floor is active on some days of the year, so the raw file differs there
    low = raw.loc[days, "wind_run_km"].to_numpy() < 100.0
    assert low.any() and not np.array_equal(raw.loc[days, "wind_run_km"].to_numpy(), c(90)[1:])
    pd.testing.assert_frame_equal(read_met(catpa_scenario / "CA-TPA.MET", prepare=True), met)


def test_asce_reference_et_against_ana(catpa_run) -> None:
    """Columns 81 (tall) and 82 (short) of the .ana file are REF_ET.FOR daily values in cm."""
    c = catpa_run["col"]
    sel = slice(1, None)  # row 0 is the YYYY.000 initial state
    tmin, tmax, srad, rh, wind_km = c(85)[sel], c(86)[sel], c(88)[sel], c(89)[sel], c(90)[sel]
    r = asce_reference_et(
        tmin,
        tmax,
        srad,
        rh,
        wind_km * 1.0e3 / 86400.0,
        elevation=catpa_run["elevation"],
        latitude=catpa_run["latitude"],
        doy=catpa_run["doy"][sel],
        wind_height=catpa_run["wind_height"],
        variant="rzwqm",
    )
    tall = c(81)[sel] * 10.0
    short = c(82)[sel] * 10.0
    rmse_tall = float(np.sqrt(np.mean((np.asarray(r.et_tall) - tall) ** 2)))
    rmse_short = float(np.sqrt(np.mean((np.asarray(r.et_short) - short) ** 2)))
    assert rmse_tall < 0.5 and rmse_short < 0.5
    # the reference model is reproduced to print precision (about 1e-5 mm/d)
    assert rmse_tall < 1e-3 and rmse_short < 1e-3


def test_shuttleworth_wallace_against_ana(catpa_run) -> None:
    """Coarse check of PT (col 9), PE (col 8) and PET = PE + PT (col 83) over one year.

    Inputs come from the .ana file itself (weather, LAI, height, residue mass) and the surface
    node water content from LAYER.PLT (previous day's end state, as the reference model uses it
    at the start of the day). Residue moisture, crusting and roughness are unknown (set 0).
    """
    if catpa_run["ipet"] != 0:
        pytest.skip("run does not use the Shuttleworth-Wallace PET option")
    plt = catpa_run["run"] / "LAYER.PLT"
    assert plt.is_file(), "LAYER.PLT (surface water content) missing from the reference run"
    theta1 = read_layer_output(plt)["soil_water_content"].isel(depth=0).values
    c = catpa_run["col"]
    n = min(len(theta1), len(c(43)) - 1)
    sel = slice(2, n + 1)  # .ana row k (k >= 2) uses the end state of day k-1 = LAYER.PLT day k-1
    prev = slice(1, n)
    tmin, tmax, srad, rh, wind_km = c(85)[sel], c(86)[sel], c(88)[sel], c(89)[sel], c(90)[sel]
    lai, height = c(43)[sel], c(62)[sel]
    residue = c(72)[prev]  # start-of-day residue mass (harvest adds residue at the end of the day)
    theta = theta1[: n - 1]
    doy = catpa_run["doy"][sel]
    age = 50.0 + np.arange(len(doy), dtype=float)  # residue albedo ageing; only a weak effect
    p = catpa_run["params"]
    residue_type = {2.0: "corn", 2.5: "soybean", 4.0: "wheat"}.get(catpa_run["cres"], "corn")

    def one(a, b, s, h_, w, l, hc, th, m, dd, ag):
        return shuttleworth_wallace(
            a,
            b,
            s,
            h_,
            w,
            l,
            hc,
            p,
            theta_surface=th,
            wc13=catpa_run["wc13"],
            wc15=catpa_run["wc15"],
            elevation=catpa_run["elevation"],
            latitude=catpa_run["latitude"],
            doy=dd,
            residue_mass=m,
            residue_age=ag,
            wind_height=catpa_run["wind_height"],
            rainfall_zone=catpa_run["zone"],
            residue_type=residue_type,
            residue_cover_factor=catpa_run["cres"],
        )

    r = jax.vmap(one)(tmin, tmax, srad, rh, wind_km, lai, height, theta, residue, doy, age)
    pt = np.asarray(r.transpiration) * 10.0
    pe = np.asarray(r.soil_evaporation + r.residue_evaporation) * 10.0
    pt_ref, pe_ref, pet_ref = c(9)[sel] * 10.0, c(8)[sel] * 10.0, c(83)[sel] * 10.0

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    assert rmse(pt + pe, pet_ref) < 0.5
    assert rmse(pt, pt_ref) < 0.3
    assert rmse(pe, pe_ref) < 0.3
    # residue cover (col 114 = 1 - CS): the same formula, but the reference model evaluates it
    # with the residue mass before the day's decomposition (col 72 is the end-of-day mass), so it
    # drifts by < 1e-3 before harvest; at harvest it also switches the residue-type constants
    # (IPR), which the PET module cannot know
    cover_diff = np.abs(np.asarray(1.0 - r.soil_fraction) - c(114)[prev])
    jumps = np.flatnonzero(np.diff(residue) > 1000.0)
    before_harvest = cover_diff[: jumps[0]] if len(jumps) else cover_diff
    assert float(np.median(before_harvest)) < 5e-3
    # transpiration is reproduced to print precision on the median day
    assert float(np.median(np.abs(pt - pt_ref))) < 1e-3
