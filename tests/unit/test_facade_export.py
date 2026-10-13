"""The export of facade results (``aj.export``): pandas, xarray and the DSSAT-style CSV files, on
synthetic ``Season`` / ``Reference`` / ``BatchResult`` objects (no data, no DSSAT, no JAX)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import agrijax as aj
from agrijax import dssat as ajd
from agrijax import facade_export as fx

PUBLISHED = {"P1": 320.0, "P2": 0.52, "P5": 940.0, "G2": 620.0, "G3": 8.5, "PHINT": 38.9}


def _yrdoy(first: int, n: int) -> list[int]:
    return [ajd._yrdoy_add(first, i) for i in range(n)]


def _season(
    key: str = "TEST8201_t01",
    first: int = 1982057,
    n: int = 6,
    g2: float = 620.0,
    planting: int | None = None,
) -> ajd.Season:
    """``n`` days from ``first``; planted on ``planting`` (default: the first day)."""
    days = _yrdoy(first, n)
    t = np.arange(n, dtype=float)
    daily = pd.DataFrame(
        {
            "date": ajd._dates(days),
            "yrdoy": days,
            "lai": 0.5 * t,
            "cwad": 100.0 * t,
            "gwad": 10.0 * t * t,
            "swtd": 200.0 - t,
        }
    )
    outputs = {
        "lai": (0.5 * t)[:, None],  # crop outputs are [days, 1]
        "lsd": t[:, None] + 1.0,
        "gstd": np.zeros((n, 1)),
        "lwad": 30.0 * t[:, None],
        "rdpd": 0.1 * t[:, None],
        "soil_sw": np.ones((n, 3)),  # not a PlantGro series: never exported
    }
    inputs = SimpleNamespace(
        exp="TEST8201",
        trno=int(key.split("_t")[1]),
        params_crop=SimpleNamespace(yrplt=np.asarray(planting or first)),
    )
    summary = {
        "ADAT": float(days[min(3, n - 1)]),
        "MDAT": float(days[-1]) if n >= 6 else np.nan,
        "HWAM": 10.0 * (n - 1) ** 2,
        "CWAM": 100.0 * (n - 1),
    }
    return ajd.Season(key, daily, summary, {**PUBLISHED, "G2": g2}, inputs, outputs)


def _reference() -> ajd.Reference:
    days = _yrdoy(1982057, 4)
    daily = pd.DataFrame(
        {
            "date": ajd._dates(days),
            "yrdoy": days,
            "lai": [0.0, 0.2, 0.4, 0.6],
            "cwad": [0.0, 90.0, 190.0, 290.0],
            "gwad": [0.0, 0.0, 10.0, 40.0],
            "swtd": [200.0, 199.0, 198.0, 197.0],
        }
    )
    summary = {"ADAT": float(days[3]), "MDAT": np.nan, "HWAM": 40.0, "CWAM": 290.0, "PDAT": 1982057.0}
    return ajd.Reference("TEST8201_t01", daily, summary, Path("."), 1.5)


def _batch() -> ajd.BatchResult:
    rows = []
    for s in range(2):
        for k, g2 in enumerate((800.0, 900.0, 1000.0)):
            rows.append(
                {
                    "scenario": s,
                    "year": 1978 + s,
                    "sowing_shift": (-14, 14)[s],
                    "sample": k,
                    **{**PUBLISHED, "G2": g2},
                    "ADAT": 1978200.0 + s + k,
                    "MDAT": np.nan if (s, k) == (1, 2) else 1978250.0 + s,
                    "HWAM": 7000.0 + 100.0 * s + k,
                    "CWAM": 15000.0,
                }
            )
    timing = {"compile_s": 1.5, "run_s": 0.25, "wall_s": 2.0, "seasons": 6, "devices": "1 x cpu"}
    return ajd.BatchResult(pd.DataFrame(rows), timing)


# ------------------------------------------------------------------------------------ pandas
def test_season_frame_is_the_daily_table():
    s = _season()
    df = s.to_frame()
    assert list(df.columns) == ["date", "yrdoy", "lai", "cwad", "gwad", "swtd"]
    assert df.equals(s.daily) and df.columns.equals(s.daily.columns)
    assert df is not s.daily and s.to_pandas().equals(df)  # a copy; to_pandas is the same table
    assert df.attrs["treatment"] == "TEST8201_t01" and df.attrs["source"] == "Agri-JAX"
    assert df.attrs["cultivar"]["G2"] == 620.0 and df.attrs["summary"]["HWAM"] == 250.0
    df.loc[0, "lai"] = 99.0  # editing the export leaves the season alone
    assert s.daily.loc[0, "lai"] == 0.0


def test_full_adds_the_plantgro_quantities_the_season_computed():
    s = _season()
    df = s.to_frame(full=True)
    assert list(df.columns)[:6] == ["date", "yrdoy", "lai", "cwad", "gwad", "swtd"]
    assert list(df.columns)[6:] == ["lsd", "gstd", "lwad", "rdpd"]  # soil_sw and missing ones are not
    np.testing.assert_array_equal(df["lwad"], 30.0 * np.arange(6))
    assert "lwad" not in s.to_frame().columns


def test_dssat_names_and_days_after_planting():
    s = _season()
    d = s.to_frame(dssat_names=True)
    assert list(d.columns) == ["RUNNO", "TRNO", "YEAR", "DOY", "DAP", "LAID", "GWAD", "CWAD", "SWTD"]
    assert d["RUNNO"].tolist() == [1] * 6 and d["TRNO"].tolist() == [1] * 6
    assert d["YEAR"].tolist() == [1982] * 6 and d["DOY"].tolist() == list(range(57, 63))
    assert d["DAP"].tolist() == list(range(6))
    np.testing.assert_array_equal(d["CWAD"], s.daily["cwad"])
    assert fx.to_frame(s, full=True, dssat_names=True).columns.tolist()[5:] == [
        "L#SD",
        "GSTD",
        "LAID",
        "LWAD",
        "GWAD",
        "CWAD",
        "RDPD",
        "SWTD",
    ]
    # across a year end DAP keeps counting, DOY and YEAR restart
    e = _season(first=1982363, n=4).to_frame(dssat_names=True)
    assert e["YEAR"].tolist() == [1982, 1982, 1982, 1983] and e["DOY"].tolist() == [363, 364, 365, 1]
    assert e["DAP"].tolist() == [0, 1, 2, 3]


def test_dap_is_zero_before_planting_as_in_dssat():
    # the season's first day is the day before planting (as in the facade's seasons)
    d = _season(first=1982056, n=4, planting=1982057).to_frame(dssat_names=True)
    assert d["DOY"].tolist() == [56, 57, 58, 59] and d["DAP"].tolist() == [0, 0, 1, 2]


def test_a_list_of_seasons_is_one_long_table():
    a, b = _season("TEST8201_t01"), _season("TEST8201_t02", n=3)
    df = fx.to_frame([a, b])
    assert df.columns.tolist()[:3] == ["season", "treatment", "date"] and len(df) == 9
    assert df["season"].tolist() == [0] * 6 + [1] * 3
    assert df["treatment"].tolist() == ["TEST8201_t01"] * 6 + ["TEST8201_t02"] * 3
    d = fx.to_frame([a, b], dssat_names=True)
    assert d["RUNNO"].tolist() == [1] * 6 + [2] * 3 and d["TRNO"].tolist() == [1] * 6 + [2] * 3


def test_batch_frame_is_its_table():
    b = _batch()
    df = b.to_frame()
    assert df.equals(b.table) and df.columns.equals(b.table.columns)
    assert df is not b.table and df.attrs["timing"]["seasons"] == 6
    assert b.to_pandas().equals(df)


def test_not_a_result_is_a_readable_error():
    with pytest.raises(TypeError, match="cannot export a dict"):
        fx.to_frame({"lai": 1})
    with pytest.raises(TypeError, match="cannot export a list"):
        fx.to_xarray(["season"])
    with pytest.raises(ValueError, match="empty"):
        fx.to_frame([])
    with pytest.raises(TypeError, match="no daily series"):
        fx.write_daily_csv(_batch(), "unused.csv")


# ------------------------------------------------------------------------------------ xarray
def test_xarray_of_seasons_is_season_by_day():
    a, b = _season("TEST8201_t01"), _season("TEST8201_t02", first=1983060, n=3, g2=700.0)
    ds = fx.to_xarray([a, b])
    assert dict(ds.sizes) == {"season": 2, "day": 6}
    assert ds["lai"].dims == ("season", "day") and ds["HWAM"].dims == ("season",)
    np.testing.assert_array_equal(ds["cwad"].sel(season=0), a.daily["cwad"])
    np.testing.assert_array_equal(ds["cwad"].sel(season=1, day=slice(0, 2)), b.daily["cwad"])
    assert np.isnan(ds["cwad"].sel(season=1, day=3)) and np.isnan(ds["yrdoy"].sel(season=1, day=5))
    assert ds["date"].dtype == np.dtype("datetime64[ns]") and np.isnat(ds["date"].values[1, 3])
    assert pd.Timestamp(ds["date"].values[1, 0]) == pd.Timestamp("1983-03-01")
    assert ds["yrdoy"].values[0, 0] == 1982057.0
    # coordinates per season
    assert ds["treatment"].values.tolist() == ["TEST8201_t01", "TEST8201_t02"]
    assert ds["source"].values.tolist() == ["Agri-JAX", "Agri-JAX"] and ds["trno"].values.tolist() == [1, 2]
    assert ds["planting"].values.tolist() == [1982057.0, 1983060.0]
    assert ds["G2"].values.tolist() == [620.0, 700.0] and ds["P1"].values.tolist() == [320.0, 320.0]
    assert np.isnan(ds["MDAT"].values[1]) and ds["ADAT"].values[0] == a.summary["ADAT"]
    # a coordinate is a selector: the season with the later-maturing cultivar
    assert ds.where(ds["G2"] > 650, drop=True)["treatment"].values.tolist() == ["TEST8201_t02"]
    # meaning, unit and DSSAT name travel with the variables
    assert ds["lai"].attrs["units"] == "m2 m-2" and ds["lai"].attrs["dssat_name"] == "LAID"
    assert ds["swtd"].attrs["dssat_file"] == "SoilWat.OUT" and ds["HWAM"].attrs["dssat_file"] == "Summary.OUT"
    assert "NOT a DSSAT output file" in ds.attrs["note"]
    assert "lwad" not in ds and "lwad" in fx.to_xarray([a, b], full=True)


def test_xarray_day_axis_counts_days_from_planting():
    # a starts on its planting day, c two days before its own (a reference run starts early the same way)
    a = _season("TEST8201_t01", first=1982057, n=4)
    c = _season("TEST8201_t02", first=1983058, n=5, planting=1983060)
    ds = fx.to_xarray([a, c])
    assert ds["day"].values.tolist() == [-2, -1, 0, 1, 2, 3]
    assert (
        ds["yrdoy"].sel(season=1, day=0).item() == 1983060.0
        and ds["yrdoy"].sel(season=0, day=0).item() == 1982057.0
    )
    assert np.isnan(ds["lai"].sel(season=0, day=-2)) and ds["lai"].sel(season=1, day=-2).item() == 0.0
    np.testing.assert_array_equal(ds["lai"].sel(season=1).values[:5], c.daily["lai"])  # c fills days -2 .. 2
    np.testing.assert_array_equal(ds["lai"].sel(season=0).values[2:], a.daily["lai"][:4])
    assert pd.Timestamp(ds["date"].sel(season=1, day=-2).values) == pd.Timestamp("1983-02-27")
    assert "planting" in ds["day"].attrs["long_name"]
    # one season by itself: the days it has
    assert fx.to_xarray(c)["day"].values.tolist() == [-2, -1, 0, 1, 2]
    # no planting date for one run: the day index of each run
    a.inputs = SimpleNamespace(exp="TEST8201", trno=1)
    ds = fx.to_xarray([a, c])
    assert ds["day"].values.tolist() == [0, 1, 2, 3, 4] and "index" in ds["day"].attrs["long_name"]
    assert ds["lai"].sel(season=1, day=0).item() == 0.0 and np.isnan(ds["lai"].sel(season=0, day=4))


def test_xarray_of_one_season_has_a_season_dimension_of_one():
    s = _season()
    ds = s.to_xarray()
    assert dict(ds.sizes) == {"season": 1, "day": 6}
    np.testing.assert_array_equal(ds["gwad"].values[0], s.daily["gwad"])
    assert ds["treatment"].values.tolist() == ["TEST8201_t01"]


def test_xarray_puts_a_reference_run_beside_the_season():
    ds = fx.to_xarray([_season(), _reference()])
    assert ds["source"].values.tolist() == ["Agri-JAX", "DSSAT"]
    assert np.isnan(ds["G2"].values[1])  # the published cultivar is not recorded by the reference run
    assert ds["planting"].values.tolist() == [1982057.0, 1982057.0]
    assert np.isnan(ds["cwad"].values[1, 4]) and ds["cwad"].values[1, 3] == 290.0
    both = fx.to_xarray([_season(), _reference()], full=True)
    assert np.isnan(both["lwad"].values[1]).all() and not np.isnan(both["lwad"].values[0]).any()


def test_xarray_of_a_batch_has_the_scenario_fields_as_coordinates():
    b = _batch()
    ds = b.to_xarray()
    assert dict(ds.sizes) == {"season": 6}
    assert set(ds.data_vars) == {"ADAT", "MDAT", "HWAM", "CWAM"}
    for c in ("scenario", "year", "sowing_shift", "sample", *ajd.CULTIVAR):
        assert c in ds.coords and ds[c].dims == ("season",), c
    np.testing.assert_array_equal(ds["G2"], b.table["G2"])
    np.testing.assert_array_equal(ds["HWAM"], b.table["HWAM"])
    assert np.isnan(ds["MDAT"].values[5]) and int(ds["year"].values[3]) == 1979
    assert ds["HWAM"].attrs["dssat_name"] == "HWAM"
    assert ds.attrs["timing_seasons"] == 6 and ds.attrs["timing_devices"] == "1 x cpu"
    assert ds.isel(season=(ds["sowing_shift"] == 14).values)["HWAM"].shape == (3,)
    assert fx.to_xarray(b).equals(fx.to_xarray(b))


def test_xarray_of_scenarios_keeps_the_planting_date_as_a_field():
    tab = pd.DataFrame(
        {"year": [1978, 1979], "sowing_shift": [0, 0], "PDAT": [1978057, 1979057], "ADAT": [1978200.0, np.nan],
         "MDAT": [1978250.0, 1979251.0], "HWAM": [7000.0, 6900.0], "CWAM": [15000.0, 14000.0]}
    )  # fmt: skip
    ds = fx.to_xarray(ajd.Scenarios("TEST8201_t01", tab, [], PUBLISHED))
    assert ds["PDAT"].dims == ("season",) and "PDAT" in ds.coords and "PDAT" not in ds.data_vars
    assert ds["year"].values.tolist() == [1978, 1979] and np.isnan(ds["ADAT"].values[1])


# ------------------------------------------------------------------------------------ CSV files
def _comment_lines(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln for ln in lines if ln.startswith("!")]


def test_xarray_and_csv_record_precision_and_soil_evaporation(tmp_path):
    """A float32 or swapped season stays recognisable once exported (coordinates and CSV header)."""
    a, b = _season("TEST8201_t01"), _season("TEST8201_t02")
    a.inputs.mesev, b.inputs.mesev, b.precision = "R", "S", "float32"
    ds = fx.to_xarray([a, b, _reference()])
    assert ds["precision"].values.tolist() == ["float64", "float32", ""]
    assert ds["mesev"].values.tolist() == ["R", "S", ""]
    head = "\n".join(_comment_lines(fx.write_summary_csv([a, b], tmp_path / "s.csv")))
    assert "RUNNO 1 TEST8201_t01: Agri-JAX float64, MESEV R" in head
    assert "RUNNO 2 TEST8201_t02: Agri-JAX float32, MESEV S" in head
    batch = _batch()
    batch.timing.update({"precision": "float32", "mesev": "S"})
    head = "\n".join(_comment_lines(fx.write_summary_csv(batch, tmp_path / "b.csv")))
    assert "runs: precision float32, mesev S" in head


def test_daily_csv_says_it_is_not_a_dssat_file_and_round_trips(tmp_path):
    s = _season()
    p = fx.write_daily_csv(s, tmp_path / "deep" / "dir" / "t01_daily.csv")
    assert p.is_file()
    head = _comment_lines(p)
    text = "\n".join(head)
    assert head[0].startswith("! Agri-JAX export, daily series")
    assert "NOT a DSSAT output file" in text and "not a genuine PlantGro.OUT" in text
    assert "independent implementation from published equations" in text
    assert (
        "validated against DSSAT-CSM (BSD-3) and RZWQM2 outputs" in text
        and "first" not in text.lower().split()
    )
    assert "LAID leaf area index [m2 m-2]" in text and "SWTD soil water in the profile [mm]" in text
    assert (
        p.read_text().splitlines()[len(head)].startswith("RUNNO,TRNO,YEAR,DOY,DAP,L#SD,GSTD,LAID,LWAD,GWAD")
    )
    got = fx.read_csv(p)  # the header lines skipped, and the '#' of L#SD survives
    want = s.to_frame(full=True, dssat_names=True)
    assert got.columns.tolist() == want.columns.tolist() and "L#SD" in got.columns
    for c in want.columns:
        np.testing.assert_array_equal(got[c].to_numpy(float), want[c].to_numpy(float), err_msg=c)
    ref = pd.read_csv(p, comment="!", na_values=[-99], float_precision="round_trip")  # as documented
    assert ref.equals(got)


def test_daily_csv_columns_and_missing_values(tmp_path):
    s = _season()
    s.outputs["lwad"][2, 0] = np.nan
    s.daily.loc[1, "swtd"] = np.nan
    full = fx.read_csv(fx.write_daily_csv(s, tmp_path / "full.csv"))
    assert full.columns.tolist() == [
        "RUNNO", "TRNO", "YEAR", "DOY", "DAP", "L#SD", "GSTD", "LAID", "LWAD", "GWAD", "CWAD", "RDPD", "SWTD",
    ]  # fmt: skip
    assert np.isnan(full.loc[2, "LWAD"]) and np.isnan(full.loc[1, "SWTD"]) and full["SWTD"].notna().sum() == 5
    rows = (tmp_path / "full.csv").read_text().splitlines()[-6:]
    assert rows[2].split(",")[8] == "-99" and rows[1].split(",")[-1] == "-99"  # DSSAT's missing value
    four = fx.read_csv(fx.write_daily_csv(s, tmp_path / "four.csv", full=False))
    assert four.columns.tolist() == ["RUNNO", "TRNO", "YEAR", "DOY", "DAP", "LAID", "GWAD", "CWAD", "SWTD"]
    other = fx.write_daily_csv(s, tmp_path / "nan.csv", na_rep="NA", full=False)
    assert other.read_text().splitlines()[-5].endswith(",NA") and "[NA]" in "\n".join(_comment_lines(other))
    assert np.isnan(fx.read_csv(other, na_rep="NA").loc[1, "SWTD"])
    assert pd.read_csv(other, comment="!").shape == (6, 9)


def test_daily_csv_of_several_runs_numbers_them(tmp_path):
    a, b = _season("TEST8201_t01"), _season("TEST8201_t02", n=3)
    p = fx.write_daily_csv([a, b, _reference()], tmp_path / "all.csv", full=False)
    d = fx.read_csv(p)
    assert d["RUNNO"].tolist() == [1] * 6 + [2] * 3 + [3] * 4
    assert d["TRNO"].tolist() == [1] * 6 + [2] * 3 + [1] * 4
    assert d["DAP"].tolist()[-4:] == [0, 1, 2, 3]
    text = "\n".join(_comment_lines(p))
    assert "SOURCE Agri-JAX" in text and "SOURCE DSSAT" in text and "dscsm048" in text


def test_a_run_without_a_planting_date_has_no_dap_column(tmp_path):
    s = _season()
    s.inputs = SimpleNamespace(exp="TEST8201", trno=1)
    d = fx.read_csv(fx.write_daily_csv(s, tmp_path / "d.csv", full=False))
    assert "DAP" not in d.columns and d.columns.tolist()[:4] == ["RUNNO", "TRNO", "YEAR", "DOY"]


def test_summary_csv_of_a_season(tmp_path):
    a, b = _season("TEST8201_t01"), _season("TEST8201_t02", first=1983060, n=3, g2=700.0)
    p = fx.write_summary_csv([a, b], tmp_path / "s.csv")
    head = "\n".join(_comment_lines(p))
    assert "NOT a DSSAT output file" in head and "ADAT silking date, YYYYDDD" in head
    assert "neither is a Summary.OUT column" in head
    lines = [ln for ln in p.read_text().splitlines() if not ln.startswith("!")]
    assert lines[0] == "RUNNO,TRNO,EXNAME,PDAT,ADAT,MDAT,CWAM,HWAM,P1,P2,P5,G2,G3,PHINT,SOURCE"
    assert lines[1].startswith("1,1,TEST8201,1982057,1982060,1982062,500.0,250.0,320.0,0.52,940.0,620.0")
    assert lines[2].startswith("2,2,TEST8201,1983060,1983062,-99,200.0,40.0,320.0,0.52,940.0,700.0")
    assert lines[2].endswith(",Agri-JAX")
    d = fx.read_csv(p)
    assert d["ADAT"].tolist() == [1982060, 1983062] and np.isnan(d["MDAT"][1])
    assert pd.api.types.is_integer_dtype(d["PDAT"])


def test_summary_csv_of_a_reference_run_keeps_its_dates(tmp_path):
    ref = _reference()
    ref.summary.update({"SDAT": 1982056.0, "EDAT": 1982065.0, "HDAT": 1982260.0})
    d = fx.read_csv(fx.write_summary_csv(ref, tmp_path / "ref.csv"))
    assert d.columns.tolist() == ["RUNNO", "TRNO", "EXNAME", "SDAT", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT",
                                  "CWAM", "HWAM", "SOURCE"]  # fmt: skip
    assert d["SOURCE"].tolist() == ["DSSAT"] and d["HDAT"].tolist() == [1982260]
    assert "P1" not in d.columns  # the reference run does not record the cultivar it ran with


def test_summary_csv_of_a_batch(tmp_path):
    b = _batch()
    p = fx.write_summary_csv(b, tmp_path / "b.csv")
    head = "\n".join(_comment_lines(p))
    assert "NOT a DSSAT output file" in head and "Agri-JAX scenario and cultivar fields" in head
    assert "not Summary.OUT columns" in head
    d = fx.read_csv(p)
    assert d.columns.tolist() == ["RUNNO", *b.table.columns]
    assert d["RUNNO"].tolist() == list(range(1, 7)) and len(d) == 6
    np.testing.assert_array_equal(d["HWAM"], b.table["HWAM"])
    assert np.isnan(d.loc[5, "MDAT"]) and d["ADAT"].tolist()[:2] == [1978200, 1978201]
    assert "1978200," in p.read_text() and "1978200.0" not in p.read_text()  # dates are integers


def test_write_csv_writes_both_files(tmp_path):
    s = _season()
    out = s.write_csv(tmp_path / "out")
    assert set(out) == {"daily", "summary"}
    assert out["daily"] == tmp_path / "out" / "TEST8201_t01_daily.csv"
    assert out["summary"] == tmp_path / "out" / "TEST8201_t01_summary.csv"
    assert all(p.is_file() for p in out.values())
    assert len(fx.read_csv(out["daily"])) == 6 and len(fx.read_csv(out["summary"])) == 1
    many = fx.write_csv([s, _reference()], tmp_path / "out", full=False)
    assert many["daily"].name == "agrijax_seasons_daily.csv"
    named = s.write_csv(tmp_path / "out", name="run7", full=False)
    assert named["daily"].name == "run7_daily.csv" and fx.read_csv(named["daily"]).shape[1] == 9
    only = _batch().write_csv(tmp_path / "out")
    assert set(only) == {"summary"} and only["summary"].name == "batch_summary.csv"
    assert _batch().write_csv(tmp_path / "out", name="g2")["summary"].name == "g2_summary.csv"


def test_csv_files_are_deterministic(tmp_path):
    a = fx.write_csv(_season(), tmp_path / "a")
    b = fx.write_csv(_season(), tmp_path / "b")
    for k in a:
        assert a[k].read_bytes() == b[k].read_bytes()


# ------------------------------------------------------------------------------------ names and light import
def test_dssat_names_cover_the_daily_series_and_are_unique():
    assert set(ajd.DAILY) <= set(fx.SERIES)
    assert fx.DSSAT_NAMES["lai"] == "LAID" and fx.DSSAT_NAMES["swtd"] == "SWTD"
    assert len(set(fx.DSSAT_NAMES.values())) == len(fx.DSSAT_NAMES)
    assert {fx.SERIES[k].file for k in ajd.DAILY} == {"PlantGro.OUT", "SoilWat.OUT"}
    assert set(ajd.SUMMARY) <= set(fx.SUMMARY_ORDER)
    for k in ajd.DAILY:  # the unit and the meaning are the ones the facade documents
        assert f"{fx.SERIES[k].long_name} [{fx.SERIES[k].units}]" == ajd.DAILY[k]


def test_export_is_lazy_and_light():
    code = (
        "import sys, agrijax as aj; assert 'jax' not in sys.modules; "
        "assert 'pandas' not in sys.modules; "
        "import agrijax.facade_export as fx; assert 'jax' not in sys.modules, 'jax imported'; "
        "assert 'xarray' not in sys.modules; "
        "print('export' in dir(aj), aj.export is fx)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["True", "True"]
    assert aj.export is fx


def test_every_public_name_has_a_docstring():
    for name in fx.__all__:
        obj = getattr(fx, name)
        if callable(obj):
            assert (obj.__doc__ or "").strip(), name
    for cls in (ajd.Season, ajd.BatchResult):
        for m in ("to_frame", "to_pandas", "to_xarray", "write_csv"):
            assert (getattr(cls, m).__doc__ or "").strip(), (cls.__name__, m)
