"""The ASCE reference ET kernel against the RZWQM2 ``.ana`` output of the canonical CA-TPA reference run.

The reference run is ``<data-dir>/catpa/ref_2015`` (one year, the shipped Scenario unchanged,
built by ``scripts/data/make_catpa_refs.py``); the site, PET and plant parameters are read from
the Scenario ``rzwqm.dat`` that run used, through :func:`agrijax.io.rzwqm.read_rzwqm_dat`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrijax.io.rzwqm import prepare_rzwqm_forcing, read_ana, read_met, read_rzwqm_dat
from agrijax.processes.pet import asce_reference_et

REF_RUN = Path("catpa/ref_2015")


@pytest.fixture(scope="module")
def catpa_run(data_dir: Path, catpa_scenario: Path):
    run = data_dir / REF_RUN
    if not (run / "CA-TPA.ana").is_file():
        pytest.skip(f"{run}/CA-TPA.ana missing (run scripts/data/make_catpa_refs.py)")
    ds = read_ana(run / "CA-TPA.ana")
    cols = {int(k): v for k, v in ds.attrs["columns"].items()}
    dat_path = catpa_scenario / "rzwqm.dat"
    dat = read_rzwqm_dat(dat_path)
    phys, pet = dat.physiography, dat.pet
    doy = pd.DatetimeIndex(ds.time.values).dayofyear.values
    return dict(
        run=run,
        ds=ds,
        col=lambda c: ds[cols[c]].values,
        doy=doy,
        elevation=float(phys["elevation_m"]),
        latitude=float(phys["latitude_rad"]),
        wind_height=float(pet["wind_height_m"]),
    )


def test_prepared_met_wind_equals_ana(catpa_run, catpa_scenario: Path) -> None:
    """The reference floors the wind run at 100 km/d; the prepared .MET equals .ana col 90."""
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
