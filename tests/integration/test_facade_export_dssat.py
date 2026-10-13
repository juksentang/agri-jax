"""The export of facade results on UFGA8201 treatment 4, against ``dscsm048`` (slow).

* **Names.** Every DSSAT column the export writes (``agrijax.facade_export.SERIES``,
  ``SUMMARY_ORDER``) is a column of the real ``PlantGro.OUT`` / ``SoilWat.OUT`` / ``Summary.OUT`` of
  the reference run, in the same order there; ``DAP``, ``YEAR`` and ``DOY`` of the CSV file equal
  the reference run's on every day.
* **Values.** The columns of a season's CSV file are the season's numbers (exactly), and agree with the
  reference run's columns of the same names to the printed precision of DSSAT's files.
* **The tables.** pandas, xarray and the CSV files of the season, the reference run and a small batch are
  the same numbers.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pytest

from agrijax import facade_export as fx
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the DSSAT example data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the DSSAT day runs in float64"),
]

#: a season against dscsm048 (the free-run acceptance: yield within 2 %, measured 3e-5 here)
SEASON_YIELD_RTOL = 1e-3
#: the daily series the facade compares with dscsm048 (``Season.compare_daily``): RMSE / DSSAT maximum
DAILY_RMSE_OVER_MAX = 2e-3
#: every other mapped ``PlantGro.OUT`` column: largest absolute difference over DSSAT's maximum of the
#: column (measured on UFGA8201 t04: 4.5e-3 for RDPD, which DSSAT prints to 0.01 m, at most 2.3e-4 for
#: the others, 0 for the stages)
EXTRA_MAX_OVER_MAX = 1e-2


@pytest.fixture(scope="module")
def engine() -> Path:
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file():
        pytest.skip(f"dscsm048 not found under {DSSAT_ENGINE}")
    return DSSAT_ENGINE


@pytest.fixture(scope="module")
def exp(engine: Path):
    import agrijax as aj

    return aj.dssat.experiment("UFGA8201", data_root=engine)


@pytest.fixture(scope="module")
def season(exp):
    return exp.run(treatment=4)


@pytest.fixture(scope="module")
def ref(exp):
    return exp.reference(treatment=4)


def _positions(names: list[str], header: list[str]) -> list[int]:
    return [header.index(n) for n in names]


def test_column_names_are_those_of_dssat_s_files_in_their_order(ref):
    from agrijax.io.dssat import read_plantgro, read_soilwat, read_summary

    pg = list(read_plantgro(ref.out / "PlantGro.OUT").columns)
    sw = list(read_soilwat(ref.out / "SoilWat.OUT").columns)
    sm = list(read_summary(ref.out / "Summary.OUT").columns)
    plant = [s.dssat for s in fx.SERIES.values() if s.file == "PlantGro.OUT"]
    soil = [s.dssat for s in fx.SERIES.values() if s.file == "SoilWat.OUT"]
    for names, header, what in (
        (plant, pg, "PlantGro"),
        (soil, sw, "SoilWat"),
        (list(fx.SUMMARY_ORDER), sm, "Summary"),
    ):
        missing = [n for n in names if n not in header]
        assert not missing, f"{what}.OUT has no column {missing}: {header}"
        pos = _positions(names, header)
        assert pos == sorted(pos), (what, names, pos)  # the export keeps DSSAT's column order
    for n in ("YEAR", "DOY", "DAP"):
        assert n in pg
    for n in ("RUNNO", "TRNO", "EXNAME", "PDAT"):
        assert n in sm


def test_year_doy_dap_and_the_daily_series_agree_with_dssat(season, ref):
    from agrijax.io.dssat import read_plantgro

    pg = read_plantgro(ref.out / "PlantGro.OUT")
    d = season.to_frame(full=True, dssat_names=True)
    assert d["DAP"].iloc[0] == 0 and pg["DAP"].iloc[0] == 0 and (d["DAP"] >= 0).all()
    names = [n for n in (s.dssat for s in fx.SERIES.values() if s.file == "PlantGro.OUT") if n in d.columns]
    m = d.merge(pg[["YEAR", "DOY", "DAP", *names]], on=["YEAR", "DOY"], suffixes=("", "_dssat"))
    assert len(m) == len(pg), (len(m), len(pg))  # every day of DSSAT's run is a day of the season
    np.testing.assert_array_equal(m["DAP"], m["DAP_dssat"])
    print("\ncolumn   max |diff|   DSSAT max   max|diff| / max")
    for n in names:
        a, b = m[n].to_numpy(float), m[f"{n}_dssat"].to_numpy(float)
        top = float(np.nanmax(np.abs(b)))
        worst = float(np.nanmax(np.abs(a - b)))
        print(f"{n:6s} {worst:12.5g} {top:12.5g} {worst / top if top else float('nan'):12.3g}")
        if n in ("LAID", "CWAD", "GWAD"):
            rmse = float(np.sqrt(np.nanmean((a - b) ** 2)))
            assert rmse / top < DAILY_RMSE_OVER_MAX, (n, rmse, top)
        elif top:
            assert worst / top < EXTRA_MAX_OVER_MAX, (n, worst, top)
    # SoilWat: SWTD
    from agrijax.io.dssat import read_soilwat

    sw = read_soilwat(ref.out / "SoilWat.OUT")
    ms = d.merge(sw[["YEAR", "DOY", "SWTD"]], on=["YEAR", "DOY"], suffixes=("", "_dssat"))
    a, b = ms["SWTD"].to_numpy(float), ms["SWTD_dssat"].to_numpy(float)
    assert np.sqrt(np.nanmean((a - b) ** 2)) / np.nanmax(np.abs(b)) < DAILY_RMSE_OVER_MAX


def test_the_same_numbers_in_pandas_xarray_and_the_csv_files(season, ref, tmp_path):
    # pandas
    df = season.to_frame()
    assert df.equals(season.daily) and season.to_pandas().equals(df)
    full = season.to_frame(full=True)
    assert set(fx.SERIES) & set(season.outputs) <= set(full.columns)
    # xarray: season x day, a reference run beside the season
    ds = fx.to_xarray([season, ref])
    assert ds["lai"].dims == ("season", "day") and ds.sizes["season"] == 2
    assert ds["source"].values.tolist() == ["Agri-JAX", "DSSAT"]
    own = np.isfinite(ds["yrdoy"].values[0])  # the days the season has (the axis spans both runs)
    assert own.sum() == len(df)
    np.testing.assert_array_equal(ds["gwad"].values[0][own], df["gwad"])
    assert ds["planting"].values[0] == ds["planting"].values[1] == float(ref.summary["PDAT"])
    # the day axis counts from planting: on every day both runs have, they are the same calendar day
    yd = ds["yrdoy"].values
    both = np.isfinite(yd).all(axis=0)
    assert both.sum() > 100 and (yd[0, both] == yd[1, both]).all()
    assert ds["yrdoy"].sel(season=0, day=0).item() == ds["yrdoy"].sel(season=1, day=0).item()
    assert ds["day"].values[0] <= 0 and ds["day"].values[-1] > 100
    # the CSV files
    out = season.write_csv(tmp_path / "season")
    daily = fx.read_csv(out["daily"])
    want = season.to_frame(full=True, dssat_names=True)
    assert daily.columns.tolist() == want.columns.tolist()
    for c in want.columns:
        np.testing.assert_array_equal(daily[c].to_numpy(float), want[c].to_numpy(float), err_msg=c)
    assert (daily["TRNO"] == 4).all() and (daily["RUNNO"] == 1).all()
    s = fx.read_csv(out["summary"])
    assert s.loc[0, "EXNAME"] == "UFGA8201" and s.loc[0, "TRNO"] == 4 and s.loc[0, "SOURCE"] == "Agri-JAX"
    assert s.loc[0, "PDAT"] == ref.summary["PDAT"]
    assert s.loc[0, "ADAT"] == ref.summary["ADAT"] and s.loc[0, "MDAT"] == ref.summary["MDAT"]
    assert abs(s.loc[0, "HWAM"] / ref.summary["HWAM"] - 1) < SEASON_YIELD_RTOL
    # the reference run's own CSV carries DSSAT's printed summary values unchanged
    r = fx.read_csv(fx.write_summary_csv(ref, tmp_path / "ref_summary.csv"))
    for c in ("PDAT", "ADAT", "MDAT", "HWAM", "CWAM"):
        assert r.loc[0, c] == ref.summary[c], c
    assert r.loc[0, "SOURCE"] == "DSSAT"
    head = (tmp_path / "season" / "UFGA8201_t04_daily.csv").read_text().splitlines()[:6]
    assert all(ln.startswith("!") for ln in head[:5]) and "NOT a DSSAT output file" in head[1]


def test_a_batch_in_xarray_and_csv(exp, tmp_path):
    scen = exp.scenarios(treatment=4, years=[1979, 1982], sowing_shift=[-14, 0])
    g2 = [scen.published["G2"] * f for f in (0.95, 1.0, 1.05)]
    batch = scen.run({"G2": g2})
    assert batch.timing["seasons"] == 4 * 3
    t = batch.to_frame()
    ds = batch.to_xarray()
    assert ds.sizes["season"] == 12 and set(ds.data_vars) == {"ADAT", "MDAT", "HWAM", "CWAM"}
    np.testing.assert_array_equal(ds["HWAM"].values, t["HWAM"].to_numpy())
    np.testing.assert_array_equal(ds["G2"].values, t["G2"].to_numpy())
    shifted = ds.isel(season=(ds["sowing_shift"] == -14).values)
    assert shifted.sizes["season"] == 6 and set(shifted["year"].values.tolist()) == {1979, 1982}
    # one of the batch's G2 values is the published one: its yield is the single season's (same program)
    base = exp.run(treatment=4)
    row = t[
        (t["sample"] == 1)
        & (t["year"] == int(base.daily["yrdoy"].iloc[0]) // 1000)
        & (t["sowing_shift"] == 0)
    ]
    if len(row):
        assert abs(row["HWAM"].iloc[0] / base.summary["HWAM"] - 1) < SEASON_YIELD_RTOL
    out = batch.write_csv(tmp_path / "batch")
    assert set(out) == {"summary"}
    c = fx.read_csv(out["summary"])
    assert c.columns.tolist() == ["RUNNO", *t.columns]
    np.testing.assert_allclose(c["HWAM"], t["HWAM"], rtol=0, atol=0)
    text = out["summary"].read_text()
    assert text.startswith("! Agri-JAX export") and "NOT a DSSAT output file" in text
    assert pd.read_csv(out["summary"], comment="!", na_values=[-99]).shape == (12, 1 + t.shape[1])
