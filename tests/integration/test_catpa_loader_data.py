"""CA-TPA loaders against the local data tree: reference runs, management events, 100k-run LHS
arrays, LHS parameter matrix.

Skipped when the files under ``--data-dir`` are absent (build them with
``scripts/data/make_catpa_refs.py`` and ``scripts/data_sync.sh catpa_lhs``). The synthetic
tests of the same loaders are in ``tests/unit/test_catpa_loader.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_jax.io.catpa import (
    EVENT_COLUMNS,
    LHS_ANA_COLUMNS,
    N_DAY,
    N_RUN,
    N_SEASON,
    build_events,
    load_catpa_lhs,
    load_catpa_params,
    load_events,
    read_manage_out,
    write_events,
)
from agri_jax.io.rzwqm import key_variables, read_ana, read_overview_yields
from agri_jax.io.rzwqm.layers import (
    profile_storage_cm,
    read_layer_output,
)

SCEN = Path("narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario")


@pytest.fixture(scope="module")
def catpa_dir(data_dir: Path) -> Path:
    p = data_dir / "catpa"
    if not (p / "ref_2015" / "LAYER.PLT").is_file():
        pytest.skip(f"{p}/ref_2015 missing (run scripts/data/make_catpa_refs.py)")
    return p


def test_layer_plt_matches_ana_storage(catpa_dir: Path) -> None:
    """sum(theta * diff([0, depth])) reproduces .ana column 2 (stored soil water) day by day."""
    ds = read_layer_output(catpa_dir / "ref_2015" / "LAYER.PLT")
    assert ds.sizes == {"time": 365, "depth": 37}
    assert ds["depth"].values[0] == 1.0 and ds["depth"].values[-1] == 150.0
    assert float(ds["thickness"].sum()) == 150.0
    ana = key_variables(read_ana(catpa_dir / "ref_2015" / "CA-TPA.ana"))
    sw = ana["profile_water_cm"].sel(time=ds["time"])
    np.testing.assert_allclose(profile_storage_cm(ds).values, sw.values, atol=2e-4)
    theta = ds["soil_water_content"].values
    assert ((theta > 0.0) & (theta < 0.6)).all()


def test_events_match_manage_out(catpa_dir: Path, data_dir: Path) -> None:
    ev = build_events(data_dir / SCEN / "rzwqm.dat", "2015-01-01", "2023-12-31")
    assert tuple(ev.columns) == EVENT_COLUMNS
    counts = ev["event"].value_counts().to_dict()
    assert counts == {"planting": 7, "harvest": 7, "fertilizer_no3": 7, "pesticide": 7, "tillage": 7}
    assert ev["date"].min() >= pd.Timestamp("2015-01-01") and ev["date"].max() <= pd.Timestamp("2021-12-31")
    pl = pd.DataFrame(ev[ev["event"] == "planting"])
    assert list(pd.to_datetime(pl["date"]).dt.strftime("%Y-%m-%d")) == [
        *(f"{y}-04-28" for y in range(2015, 2020)),
        "2020-05-05",
        "2021-05-05",
    ]
    assert (pl["value"] == 80000.0).all() and (pl["unit"] == "seeds/ha").all()
    assert (ev.loc[ev["event"] == "fertilizer_no3", "value"] == 180.0).all()

    man = read_manage_out(catpa_dir / "base_2015_2023" / "MANAGE.OUT")
    man = man[~(man["event"].str.startswith("fertilizer_") & (man["value"] == 0.0))]
    key = ["date", "event"]
    a = pd.DataFrame(ev).sort_values(by=key).reset_index(drop=True)
    b = pd.DataFrame(man).sort_values(by=key).reset_index(drop=True)
    assert a[key].equals(b[key])
    both = ~a["event"].isin(["tillage", "harvest"])
    np.testing.assert_allclose(a.loc[both, "value"], b.loc[both, "value"])
    # the base OVERVIEW.OUT agrees on planting/harvest dates and density
    ov = read_overview_yields(catpa_dir / "base_2015_2023" / "OVERVIEW.OUT")
    np.testing.assert_array_equal(ov["planting_date"].to_numpy(), pl["date"].to_numpy())
    np.testing.assert_array_equal(ov["harvest_date"].values, ev.loc[ev["event"] == "harvest", "date"].values)
    assert (ov["plants_m2"] == 8.0).all()


def test_events_csv_roundtrip(catpa_dir: Path, data_dir: Path, tmp_path: Path) -> None:
    ev = build_events(data_dir / SCEN / "rzwqm.dat")
    back = load_events(write_events(ev, tmp_path / "events.csv"))
    pd.testing.assert_frame_equal(back, ev, check_dtype=False)
    if (catpa_dir / "events.csv").is_file():
        pd.testing.assert_frame_equal(load_events(catpa_dir / "events.csv"), ev, check_dtype=False)


@pytest.fixture(scope="module")
def lhs(data_dir: Path) -> dict[str, np.ndarray]:
    d = data_dir / "catpa_lhs"
    if not (d / "catpa_daily.npz").is_file():
        pytest.skip(f"{d}/catpa_daily.npz missing (scripts/data_sync.sh catpa_lhs)")
    return load_catpa_lhs(d)


def test_lhs_shapes(lhs: dict[str, np.ndarray]) -> None:
    assert lhs["run"].shape == (N_RUN,)
    np.testing.assert_array_equal(lhs["run"][:3], [0, 1, 2])
    assert (np.diff(lhs["run"]) == 1).all()
    assert lhs["time"].shape == (N_DAY,)
    assert lhs["time"][0] == np.datetime64("2015-01-01") and lhs["time"][-1] == np.datetime64("2023-12-31")
    for nm in LHS_ANA_COLUMNS:
        assert lhs[nm].shape == (N_RUN, N_DAY), nm
        assert lhs[nm].dtype == np.float32
    assert lhs["yields"].shape == (N_RUN, N_SEASON)
    np.testing.assert_array_equal(lhs["season_year"], np.arange(2015, 2022))


def test_lhs_params_rows(data_dir: Path) -> None:
    p = data_dir / "narval_mirror/RZWQM_sw_batch/CA-TPA/AutoAnalysis/parameter.csv"
    if not p.is_file():
        pytest.skip(f"{p} missing")
    df = load_catpa_params(p)
    assert len(df) == N_RUN and df.index.name == "run" and df.index[0] == 0
    assert df.shape[1] == 30
    assert df.iloc[0]["Pore Size (c2)"] == pytest.approx(0.17831)
    can = load_catpa_params(p, canonical=True)
    assert "lam_1" in can.columns and can["lam_1"].iloc[0] == pytest.approx(0.17831)


def test_lhs_run0_matches_local_rerun(lhs: dict[str, np.ndarray], catpa_dir: Path) -> None:
    """Run 0 = parameter.csv row 0, re-run locally (make_catpa_refs.py lhs_run0/).

    OVERVIEW yields agree to <= 2 kg/ha (the cluster and this machine differ in the last bits
    of floating point; observed diffs 0, 0, 0, 1, 0, 2, 0 kg/ha), daily series to float32 print
    precision plus the same small drift.
    """
    ov_path = catpa_dir / "lhs_run0" / "OVERVIEW.OUT"
    if not ov_path.is_file():
        pytest.skip(f"{ov_path} missing")
    ov = read_overview_yields(ov_path)["yield_kg_ha"].to_numpy()
    assert ov.shape == (N_SEASON,)
    assert np.abs(lhs["yields"][0] - ov).max() <= 2
    ana = read_ana(catpa_dir / "lhs_run0" / "CA-TPA.ana")
    kv = key_variables(ana, LHS_ANA_COLUMNS).isel(time=slice(1, None))  # drop the 2015.000 row
    assert kv.sizes["time"] == N_DAY
    np.testing.assert_array_equal(kv["time"].values.astype("datetime64[D]"), lhs["time"])
    for nm in ("sw_cm", "aet_cm", "lai"):
        np.testing.assert_allclose(np.asarray(lhs[nm][0], np.float64), kv[nm].values, atol=0.05, rtol=0.01)
    # .ana grain mass on each harvest date is that season's OVERVIEW yield, and 0 the day after
    ev = build_events()
    harvest = ev.loc[ev["event"] == "harvest", "date"].to_numpy()
    idx = np.searchsorted(lhs["time"], harvest.astype("datetime64[D]"))
    grain = np.asarray(lhs["grain_kg_ha"][0], np.float64)
    np.testing.assert_allclose(grain[idx], lhs["yields"][0], atol=1.0)
    assert (grain[idx + 1] == 0.0).all()


def test_lhs_runs_selection(data_dir: Path) -> None:
    d = data_dir / "catpa_lhs"
    if not (d / "npy" / "lai.npy").is_file():
        pytest.skip("npy cache not extracted")
    sub = load_catpa_lhs(d, variables=["lai"], runs=[5, 7])
    full = load_catpa_lhs(d, variables=["lai"])
    np.testing.assert_array_equal(sub["run"], [5, 7])
    np.testing.assert_array_equal(sub["lai"], full["lai"][[5, 7]])
    np.testing.assert_array_equal(sub["yields"], full["yields"][[5, 7]])
    assert set(sub) == {"run", "time", "yyyyddd", "lai", "yields", "season_year"}
    with pytest.raises(KeyError):
        load_catpa_lhs(d, variables=["nope"])
