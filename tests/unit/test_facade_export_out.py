"""The DSSAT-format ``.OUT`` writer of ``aj.export`` (``write_dssat_out``), on synthetic seasons (no data,
no DSSAT, no JAX): the layout of the lines (checked against literal lines in DSSAT-CSM v4.8.6's layout),
the statement in the first lines that Agri-JAX wrote the files and DSSAT did not, the round trip through
the repository's own readers of DSSAT's output files (``agrijax.io.dssat``), missing and overflowing
values, the rows of each file, several runs in one file set, batches and what is refused."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from tests.unit.test_facade_export import _batch, _reference, _season

import agrijax as aj
from agrijax import dssat as ajd
from agrijax import facade_export as fx
from agrijax.io.dssat import read_plantgro, read_soilwat, read_summary

LABELS = {"experiment": "TEST8201MZ NIT X IRR, GAINESVILLE", "treatment": "IRRIGATED HIGH NITROGEN"}
#: the stage output: emergence (code 1) on the fourth day
STAGE = np.array([7, 8, 8, 1, 1, 2, 3, 4], dtype=float)[:, None]


def _out_season(
    key: str = "TEST8201_t04",
    *,
    first: int = 1982056,
    n: int = 8,
    planting: int | None = 1982057,
    sdat: int | None = 1982056,
    mesev: str = "R",
    precision: str = "float64",
    labels: dict[str, str] | None = None,
) -> ajd.Season:
    """A season that starts on the day before it is planted, with everything the writer can use."""
    s = _season(key, first=first, n=n, planting=planting)
    s.outputs["istage"] = STAGE[:n] if n <= len(STAGE) else np.ones((n, 1))
    ref = {"HDAT": float(s.daily["yrdoy"].iloc[-1])}
    if sdat is not None:
        ref["SDAT"] = float(sdat)
    s.inputs.summary = ref
    s.inputs.mesev = mesev
    s.precision = precision
    s.labels = dict(LABELS if labels is None else labels)
    if planting is None:
        s.inputs.params_crop = SimpleNamespace()
    return s


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="latin-1").split("\n")


# ---------------------------------------------------------------------------------- the layout
def test_plantgro_is_laid_out_as_dssat_lays_it_out(tmp_path):
    files = _out_season().write_dssat_out(tmp_path / "out")
    assert list(files) == ["PlantGro.OUT", "SoilWat.OUT", "Summary.OUT"]
    assert all(p.parent == tmp_path / "out" and p.is_file() for p in files.values())
    lines = _lines(files["PlantGro.OUT"])
    assert lines[0].startswith("*GROWTH ASPECTS OUTPUT FILE")
    assert lines[1] == "" and lines[2].startswith("*Agri-JAX ") and lines[3] == ""
    assert lines[4:10] == [  # the run header, character for character as dscsm048 writes it
        "*RUN   1        : IRRIGATED HIGH NITROGEN   MZCER048 TEST8201    4",
        " MODEL          : MZCER048 - Maize",
        " EXPERIMENT     : TEST8201 MZ NIT X IRR, GAINESVILLE",
        " DATA PATH      :",
        " TREATMENT  4   : IRRIGATED HIGH NITROGEN   MZCER048",
        "  ",
    ]
    assert lines[10].startswith("! ") and lines[11].startswith("! ")
    assert lines[12] == "@YEAR DOY   DAS   DAP   L#SD   GSTD   LAID   LWAD   GWAD   CWAD   RDPD"
    # the first day is the day before planting: PlantGro.OUT starts on the planting day, DAP 0, DAS 2
    assert lines[13] == " 1982 057     2     0    2.0      0   0.50     30     10    100   0.10"
    assert lines[14] == " 1982 058     3     1    3.0      0   1.00     60     40    200   0.20"
    assert (
        lines[-2] == " 1982 063     8     6    8.0      0   3.50    210    490    700   0.70"
        and lines[-1] == ""
    )
    assert len(lines) == 13 + 7 + 1


def test_soilwat_and_summary_are_laid_out_as_dssat_lays_them_out(tmp_path):
    files = _out_season().write_dssat_out(tmp_path)
    lines = _lines(files["SoilWat.OUT"])
    assert lines[0].startswith("*SOIL WATER DAILY OUTPUT FILE") and lines[2].startswith("*Agri-JAX ")
    assert lines[4] == "*RUN   1        : IRRIGATED HIGH NITROGEN   MZCER048 TEST8201    4"
    assert lines[12] == "@YEAR DOY   DAS    SWTD"
    assert lines[13] == " 1982 056     1     200"  # SoilWat.OUT starts on the first day of the run
    assert lines[14] == " 1982 057     2     199" and len(lines) == 13 + 8 + 1
    s = _lines(files["Summary.OUT"])
    assert (
        s[0].startswith("*SUMMARY : TEST8201MZ NIT X IRR, GAINESVILLE") and "Agri-JAX" in s[0] and s[1] == ""
    )
    head = ["@   RUNNO   TRNO CR MODEL... EXNAME.. TNAM.....................", "    SDAT    PDAT    EDAT"]
    head += ["    ADAT    MDAT    HDAT    CWAM    HWAM"]
    assert s[-3] == "".join(head)
    row = ["        1      4 MZ MZCER048 TEST8201 IRRIGATED HIGH NITROGEN  ", " 1982056 1982057 1982059"]
    row += [" 1982059 1982063 1982063     700     490"]
    assert s[-2] == "".join(row)
    assert s[-1] == ""


def test_the_first_lines_say_agri_jax_wrote_the_file_and_dssat_did_not(tmp_path):
    a = _out_season()
    b = _out_season("TEST8201_t05", mesev="S", precision="float32")
    for season, precision, method in (
        (a, "precision float64", "soil evaporation Ritchie (MESEV R)"),
        (b, "precision float32", "soil evaporation SALUS (MESEV S)"),
    ):
        for name, p in season.write_dssat_out(tmp_path / season.treatment).items():
            head = "\n".join(_lines(p)[:5])
            assert f"Agri-JAX {aj.__version__}" in head, name
            assert precision in head and method in head, name
            assert "NOT produced by the DSSAT program" in head and "in DSSAT-CSM v4.8.6 format" in head, name
            assert "DSSAT Cropping System Model Ver." not in head, name  # that line is DSSAT's own
    # a run made of several seasons states both of their options
    both = _lines(fx.write_dssat_out([a, b], tmp_path / "both")["Summary.OUT"])[0]
    assert "float32 and float64 (RUNNO lines)" in both and "Ritchie (MESEV R) and SALUS (MESEV S)" in both
    note = "\n".join(_lines(tmp_path / "both" / "Summary.OUT")[:6])
    assert "! RUNNO 2: TEST8201_t05, Agri-JAX float32, soil evaporation SALUS (MESEV S)" in note
    # nothing recorded: the file says so instead of guessing
    c = _out_season(mesev="", precision="")
    head = _lines(fx.write_dssat_out(c, tmp_path / "unknown")["PlantGro.OUT"])[2]
    assert "precision not recorded" in head and "soil evaporation not recorded" in head


def test_the_columns_not_written_are_named_in_the_file_and_in_the_module(tmp_path):
    files = _out_season().write_dssat_out(tmp_path)
    plant, soil = _lines(files["PlantGro.OUT"])[10], _lines(files["SoilWat.OUT"])[10]
    assert "VWAD GWGD HIAD" in plant and "RL1D" in plant and "SNW1C" in plant and "not -99 or 0" in plant
    assert "SWXD ROFC DRNC" in soil and "SW9D" in soil
    # nothing that is written is also said to be left out, and every export column has a DSSAT format
    written = {"YEAR", "DOY", "DAS", "DAP", *fx._OUT_FORMAT}
    assert not written & {c for cols in fx.OUT_NOT_WRITTEN.values() for c in cols}
    assert set(fx._OUT_FORMAT) == {s.dssat for s in fx.SERIES.values()}
    assert "write_dssat_out" in fx.NOT_DSSAT_OUT and "NOT a DSSAT output file" in fx.NOT_DSSAT_OUT


def test_every_exported_column_has_dssat_s_width_and_decimals():
    # read off dscsm048 v4.8.6's own PlantGro.OUT / SoilWat.OUT of UFGA8201
    want = {
        "L#SD": (7, 1), "GSTD": (7, 0), "LAID": (7, 2), "LWAD": (7, 0), "SWAD": (7, 0), "GWAD": (7, 0),
        "RWAD": (7, 0), "CWAD": (7, 0), "G#AD": (7, 0), "PWAD": (7, 0), "WSPD": (7, 3), "WSGD": (7, 3),
        "EWSD": (7, 3), "RDPD": (7, 2), "DTTD": (6, 2), "SWTD": (8, 0),
    }  # fmt: skip
    assert fx._OUT_FORMAT == want
    for f, text in (
        (fx._YEAR, " 1982"),
        (fx._DOY, " 057"),
        (fx._DAS, "     2"),
        (fx._Field("L#SD", 7, 1), "   18.0"),
        (fx._Field("LAID", 7, 2), "   3.85"),
        (fx._Field("WSPD", 7, 3), "  0.000"),
        (fx._Field("DTTD", 6, 2), " 13.15"),
        (fx._Field("SWTD", 8, 0), "     212"),
    ):
        value = {"YEAR": 1982, "DOY": 57, "DAS": 2, "L#SD": 18.0, "LAID": 3.85, "WSPD": 0.0, "DTTD": 13.15}
        assert fx._cell(f, value.get(f.name, 212.0)) == text, f.name


# ---------------------------------------------------------------------------------- the readers
def test_the_readers_of_dssat_files_read_them_back(tmp_path):
    s = _out_season()
    files = s.write_dssat_out(tmp_path)
    pg = read_plantgro(files["PlantGro.OUT"])
    assert pg.columns.tolist() == [
        "RUN", "TRNO", "YEAR", "DOY", "DAS", "DAP", "L#SD", "GSTD", "LAID", "LWAD", "GWAD", "CWAD", "RDPD", "DATE",
    ]  # fmt: skip
    assert pg["RUN"].tolist() == [1] * 7 and pg["TRNO"].tolist() == [4] * 7
    assert pg["YEAR"].tolist() == [1982] * 7 and pg["DOY"].tolist() == list(range(57, 64))
    assert pg["DAS"].tolist() == list(range(2, 9)) and pg["DAP"].tolist() == list(range(7))
    assert pg["DATE"].iloc[0] == pd.Timestamp("1982-02-26") and pg["DATE"].iloc[-1] == pd.Timestamp(
        "1982-03-04"
    )
    day = s.daily.iloc[1:].reset_index(drop=True)  # from the planting day
    np.testing.assert_array_equal(pg["LAID"], day["lai"])
    np.testing.assert_array_equal(pg["CWAD"], day["cwad"])
    np.testing.assert_array_equal(pg["GWAD"], day["gwad"])
    np.testing.assert_array_equal(pg["LWAD"], s.outputs["lwad"][1:, 0])
    np.testing.assert_array_equal(pg["L#SD"], s.outputs["lsd"][1:, 0])
    np.testing.assert_allclose(pg["RDPD"], s.outputs["rdpd"][1:, 0], atol=1e-12)
    sw = read_soilwat(files["SoilWat.OUT"])
    assert sw.columns.tolist() == ["RUN", "TRNO", "YEAR", "DOY", "DAS", "SWTD", "DATE"]
    assert sw["DAS"].tolist() == list(range(1, 9)) and sw["DOY"].tolist() == list(range(56, 64))
    np.testing.assert_array_equal(sw["SWTD"], s.daily["swtd"])
    sm = read_summary(files["Summary.OUT"])
    assert sm.columns.tolist() == [
        "RUNNO", "TRNO", "CR", "MODEL", "EXNAME", "TNAM", "SDAT", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT", "CWAM", "HWAM",
    ]  # fmt: skip
    row = sm.iloc[0]
    assert (row["RUNNO"], row["TRNO"], row["CR"], row["MODEL"], row["EXNAME"]) == (
        1,
        4,
        "MZ",
        "MZCER048",
        "TEST8201",
    )
    assert row["TNAM"] == "IRRIGATED HIGH NITROGEN"
    assert [row[c] for c in ("SDAT", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT")] == [
        1982056, 1982057, 1982059, 1982059, 1982063, 1982063,
    ]  # fmt: skip
    assert (row["CWAM"], row["HWAM"]) == (700, 490)


def _awkward_season() -> ajd.Season:
    """Values that are not already on DSSAT's grid, positive and negative, ties and not."""
    rng = np.random.default_rng(20261001)
    n = 40
    s = _out_season(n=n, first=1982056)
    s.outputs["istage"] = np.ones((n, 1))
    ties = np.array([0.5, 1.5, 2.5, 3.5, 1234.5, -0.5, -2.5, 99999.5])
    big = lambda lo, hi: rng.uniform(lo, hi, n)  # noqa: E731
    cwad = big(0.0, 25000.0)
    cwad[: len(ties)] = ties
    s.daily["cwad"] = cwad
    s.daily["gwad"] = big(0.0, 12000.0)
    s.daily["lai"] = big(0.0, 6.0)
    s.daily["swtd"] = big(150.0, 260.0)
    s.outputs["lsd"] = big(0.0, 21.0)[:, None]
    s.outputs["rdpd"] = big(0.0, 1.8)[:, None]
    s.outputs["wspd"] = big(0.0, 1.0)[:, None]
    s.outputs["dttd"] = big(0.0, 20.0)[:, None]
    s.outputs["gstd"] = rng.integers(0, 11, n).astype(float)[:, None]
    return s


def _printed(x: float, decimals: int) -> float:
    """DSSAT's printed value, rounded independently of the writer (``decimal``, on the exact value)."""
    q = Decimal(1).scaleb(-decimals)
    mode = ROUND_HALF_UP if decimals == 0 else ROUND_HALF_EVEN  # whole numbers: Fortran NINT
    return float(Decimal(x).quantize(q, rounding=mode))


def test_values_are_rounded_as_dssat_prints_them_and_read_back(tmp_path):
    s = _awkward_season()
    files = s.write_dssat_out(tmp_path)
    pg = read_plantgro(files["PlantGro.OUT"])
    day = s.daily.iloc[1:].reset_index(drop=True)  # from the planting day
    want = {
        "CWAD": (day["cwad"], 0), "GWAD": (day["gwad"], 0), "LAID": (day["lai"], 2),
        "L#SD": (pd.Series(s.outputs["lsd"][1:, 0]), 1), "RDPD": (pd.Series(s.outputs["rdpd"][1:, 0]), 2),
        "WSPD": (pd.Series(s.outputs["wspd"][1:, 0]), 3), "DTTD": (pd.Series(s.outputs["dttd"][1:, 0]), 2),
        "GSTD": (pd.Series(s.outputs["gstd"][1:, 0]), 0),
    }  # fmt: skip
    for c, (values, decimals) in want.items():
        expect = [_printed(v, decimals) for v in values]
        np.testing.assert_array_equal(pg[c].to_numpy(float), expect, err_msg=c)
    sw = read_soilwat(files["SoilWat.OUT"])
    np.testing.assert_array_equal(sw["SWTD"].to_numpy(float), [_printed(v, 0) for v in s.daily["swtd"]])
    # whole numbers are NINT: halves go away from zero, not to the even number
    cw = fx._Field("CWAD", 7)
    halves = [fx._cell(cw, v) for v in (0.5, 1.5, 2.5, -0.5, -2.5)]
    assert halves == ["      1", "      2", "      3", "     -1", "     -3"]
    assert pg["CWAD"].tolist()[:7] == [2, 3, 4, 1235, -1, -3, 100000]  # the halves of the season
    rows = [ln for ln in _lines(files["PlantGro.OUT"]) if ln.startswith(" 198")]
    assert len(rows) == len(pg) and all(len(r) == len(rows[0]) for r in rows)  # every row has the same width


def test_missing_and_overflowing_values(tmp_path):
    s = _out_season()
    s.daily.loc[2, "cwad"] = np.nan
    s.daily.loc[3, "lai"] = np.nan
    s.daily.loc[4, "swtd"] = np.nan
    s.outputs["rdpd"][4, 0] = np.nan
    s.daily.loc[5, "gwad"] = 1.0e8  # does not fit I7
    s.daily.loc[6, "lai"] = np.inf
    files = s.write_dssat_out(tmp_path)
    text = "\n".join(_lines(files["PlantGro.OUT"]))
    assert "    -99" in text and " -99.00" in text and "*******" in text  # I7, F7.2 missing; I7 overflow
    pg = read_plantgro(files["PlantGro.OUT"])  # the readers take -99 and asterisks for missing
    assert np.isnan(pg.loc[1, "CWAD"]) and np.isnan(pg.loc[2, "LAID"]) and np.isnan(pg.loc[3, "RDPD"])
    assert np.isnan(pg.loc[4, "GWAD"]) and np.isnan(pg.loc[5, "LAID"])
    assert pg["CWAD"].notna().sum() == 6 and pg["GWAD"].notna().sum() == 6
    sw = read_soilwat(files["SoilWat.OUT"])
    assert np.isnan(sw.loc[4, "SWTD"]) and sw["SWTD"].notna().sum() == 7
    assert "     -99" in "\n".join(_lines(files["SoilWat.OUT"]))  # I8
    # a missing value in a real field is written in that field's own format, as DSSAT writes -99
    assert fx._cell(fx._Field("DTTD", 6, 2), np.nan) == "-99.00"
    assert fx._cell(fx._Field("WSPD", 7, 3), np.nan) == "-99.000"
    assert fx._cell(fx._Field("LAID", 7, 2), -0.001) == "   0.00"  # no negative zero
    assert fx._cell(fx._Field("LAID", 7, 2), -0.4) == "  -0.40"


# ---------------------------------------------------------------------------------- the rows
def test_rows_start_where_dssat_files_start(tmp_path):
    # planted three days after the start: SoilWat.OUT keeps every day, PlantGro.OUT starts at DAP 0
    s = _out_season(first=1982056, n=8, planting=1982059)
    files = s.write_dssat_out(tmp_path)
    pg, sw = read_plantgro(files["PlantGro.OUT"]), read_soilwat(files["SoilWat.OUT"])
    assert pg["DOY"].tolist() == [59, 60, 61, 62, 63] and pg["DAP"].tolist() == [0, 1, 2, 3, 4]
    assert pg["DAS"].tolist() == [4, 5, 6, 7, 8] and sw["DOY"].tolist() == list(range(56, 64))
    # across a year end DAS and DAP keep counting while DOY restarts
    t = _out_season(first=1982363, n=5, planting=1982364, sdat=1982363)
    pg = read_plantgro(t.write_dssat_out(tmp_path / "ye")["PlantGro.OUT"])
    assert pg["YEAR"].tolist() == [1982, 1982, 1983, 1983] and pg["DOY"].tolist() == [364, 365, 1, 2]
    assert pg["DAP"].tolist() == [0, 1, 2, 3] and pg["DAS"].tolist() == [2, 3, 4, 5]


def test_dates_the_season_does_not_know_are_not_made_up(tmp_path):
    # no simulation start date: no DAS column; no planting date: no DAP column and every day
    s = _out_season(sdat=None, planting=None)
    files = s.write_dssat_out(tmp_path)
    pg, sw = read_plantgro(files["PlantGro.OUT"]), read_soilwat(files["SoilWat.OUT"])
    assert "DAS" not in pg.columns and "DAP" not in pg.columns and len(pg) == 8
    assert sw.columns.tolist() == ["RUN", "TRNO", "YEAR", "DOY", "SWTD", "DATE"]
    sm = read_summary(files["Summary.OUT"])
    assert "SDAT" not in sm.columns and "PDAT" not in sm.columns and sm["EDAT"].tolist() == [1982059]
    # a season whose inputs record nothing (as test_facade builds them): the four values of every season
    summary = {"ADAT": 1982060.0, "MDAT": np.nan, "HWAM": 5.0, "CWAM": 7.0}
    bare = ajd.Season("T_t01", _season().daily, summary, {}, None, {})
    sm = read_summary(bare.write_dssat_out(tmp_path / "bare")["Summary.OUT"])
    assert sm.columns.tolist() == ["RUNNO", "TRNO", "CR", "MODEL", "EXNAME", "ADAT", "MDAT", "CWAM", "HWAM"]
    assert np.isnan(sm.loc[0, "MDAT"]) and sm.loc[0, "HWAM"] == 5 and sm.loc[0, "EXNAME"] == "T"
    run = _lines(bare.write_dssat_out(tmp_path / "bare")["PlantGro.OUT"])
    # no treatment name known: its 25 characters stay blank
    assert run[4] == "*RUN   1        : " + " " * 26 + "MZCER048 " + "T" + " " * 11 + "1"
    assert " EXPERIMENT     : T" in run


def test_the_harvest_date_is_the_experiments_only_when_the_season_ends_on_it(tmp_path):
    s = _out_season()
    s.inputs.summary["HDAT"] = float(s.daily["yrdoy"].iloc[-1]) + 5  # the season stopped earlier (maturity)
    sm = read_summary(s.write_dssat_out(tmp_path)["Summary.OUT"])
    assert "HDAT" not in sm.columns  # nothing knows it: no column
    t = _out_season("TEST8201_t05")
    t.inputs.summary["HDAT"] = float(t.daily["yrdoy"].iloc[-1])
    both = read_summary(fx.write_dssat_out([s, t], tmp_path / "two")["Summary.OUT"])
    assert np.isnan(both.loc[0, "HDAT"]) and both.loc[1, "HDAT"] == 1982063  # -99 where one season lacks it


def test_a_long_treatment_name_is_cut_where_dssat_cuts_it(tmp_path):
    name = "N=56 KG/HA  POP=4.7 PL/M2 AND MORE"
    s = _out_season(labels={"experiment": "IUAF9901MZ IUAF9900MZ MAIZE KN", "treatment": name})
    files = s.write_dssat_out(tmp_path)
    run = _lines(files["PlantGro.OUT"])
    assert run[4] == "*RUN   1        : N=56 KG/HA  POP=4.7 PL/M2 MZCER048 TEST8201    4"
    assert run[6] == " EXPERIMENT     : IUAF9901 MZ IUAF9900MZ MAIZE KN"
    assert read_summary(files["Summary.OUT"]).loc[0, "TNAM"] == "N=56 KG/HA  POP=4.7 PL/M2"


# ---------------------------------------------------------------------------------- several runs
def test_several_seasons_are_one_multi_run_file_set(tmp_path):
    a = _out_season("TEST8201_t04")
    b = _out_season("TEST8201_t05", first=1983057, n=6, planting=1983058, sdat=1983057, labels={
        "experiment": LABELS["experiment"], "treatment": "RAINFED LOW NITROGEN"})  # fmt: skip
    files = fx.write_dssat_out([a, b], tmp_path)
    text = _lines(files["PlantGro.OUT"])
    runs = [i for i, ln in enumerate(text) if ln.startswith("*RUN")]
    assert (
        len(runs) == 2
        and text[runs[1]] == "*RUN   2        : RAINFED LOW NITROGEN      MZCER048 TEST8201    5"
    )
    assert text[runs[1] - 3] == "" and text[runs[1] - 2].startswith("*Agri-JAX") and text[runs[1] - 1] == ""
    assert sum(ln.startswith("@YEAR") for ln in text) == 2  # a table per run, as in a DSSAT batch
    pg = read_plantgro(files["PlantGro.OUT"])
    assert pg["RUN"].tolist() == [1] * 7 + [2] * 5 and pg["TRNO"].tolist() == [4] * 7 + [5] * 5
    assert pg["YEAR"].tolist() == [1982] * 7 + [1983] * 5 and pg["DAP"].tolist() == list(range(7)) + list(
        range(5)
    )
    sw = read_soilwat(files["SoilWat.OUT"])
    assert sw["RUN"].tolist() == [1] * 8 + [2] * 6 and sw["DAS"].tolist() == list(range(1, 9)) + list(
        range(1, 7)
    )
    sm = read_summary(files["Summary.OUT"])
    assert sm["RUNNO"].tolist() == [1, 2] and sm["TRNO"].tolist() == [4, 5]
    assert sm["TNAM"].tolist() == ["IRRIGATED HIGH NITROGEN", "RAINFED LOW NITROGEN"]
    assert sm["PDAT"].tolist() == [1982057, 1983058] and sm["SDAT"].tolist() == [1982056, 1983057]
    assert sum(ln.startswith("@") for ln in _lines(files["Summary.OUT"])) == 1  # one table of runs
    # the writer does not touch the seasons
    assert a.daily.shape == (8, 6) and list(a.labels) == ["experiment", "treatment"]


def test_per_run_writes_one_directory_per_season(tmp_path):
    a, b = _out_season("TEST8201_t04"), _out_season("TEST8201_t05")
    files = fx.write_dssat_out([a, b], tmp_path, per_run=True)
    assert list(files) == [f"{i}_TEST8201_t0{t}/{n}" for i, t in ((1, 4), (2, 5)) for n in
                           ("PlantGro.OUT", "SoilWat.OUT", "Summary.OUT")]  # fmt: skip
    assert all(p.is_file() and p.parent.parent == tmp_path for p in files.values())
    first = read_summary(files["1_TEST8201_t04/Summary.OUT"])
    second = read_summary(files["2_TEST8201_t05/Summary.OUT"])
    assert (
        first["TRNO"].tolist() == [4] and second["TRNO"].tolist() == [5] and second["RUNNO"].tolist() == [1]
    )
    assert read_plantgro(files["2_TEST8201_t05/PlantGro.OUT"])["RUN"].unique().tolist() == [1]
    # the same bytes as the file set of that season written alone
    alone = a.write_dssat_out(tmp_path / "alone")
    assert files["1_TEST8201_t04/PlantGro.OUT"].read_bytes() == alone["PlantGro.OUT"].read_bytes()


# ---------------------------------------------------------------------------------- batches
def test_a_batch_has_a_summary_with_a_run_per_season(tmp_path):
    batch = _batch()
    batch.treatment = "TEST8201_t04"
    batch.timing.update({"precision": "float32", "mesev": "S"})
    files = batch.write_dssat_out(tmp_path)
    assert list(files) == ["Summary.OUT"]  # a batch keeps no daily series
    text = _lines(files["Summary.OUT"])
    assert "precision float32" in text[0] and "SALUS (MESEV S)" in text[0]
    assert "NOT produced by the DSSAT program" in text[0]
    sm = read_summary(files["Summary.OUT"])
    assert sm.columns.tolist() == ["RUNNO", "TRNO", "CR", "MODEL", "EXNAME", "ADAT", "MDAT", "CWAM", "HWAM"]
    assert sm["RUNNO"].tolist() == list(range(1, 7)) and set(sm["TRNO"]) == {4}
    assert set(sm["EXNAME"]) == {"TEST8201"}
    np.testing.assert_array_equal(sm["HWAM"], np.round(batch.table["HWAM"]))
    assert sm["ADAT"].tolist()[:2] == [1978200, 1978201] and np.isnan(sm.loc[5, "MDAT"])
    # the scenario and cultivar fields are not Summary.OUT columns: they are listed in comment lines
    notes = [ln for ln in text if ln.startswith("! ")]
    assert any("not Summary.OUT columns" in ln for ln in notes)
    head = next(ln for ln in notes if ln.startswith("! RUNNO") and "scenario" in ln)
    assert head.split()[2:] == ["scenario", "year", "sowing_shift", "sample", *ajd.CULTIVAR]
    assert len([ln for ln in notes if ln.split()[1:2] and ln.split()[1].isdigit()]) == 6
    assert not any(ln.startswith("@") and "scenario" in ln for ln in text)
    # no treatment recorded: the identifiers say so (-99) instead of inventing them
    anon = fx.write_dssat_out(ajd.BatchResult(batch.table, batch.timing), tmp_path / "anon")["Summary.OUT"]
    assert bool(read_summary(anon)["TRNO"].isna().all())
    again = fx.write_dssat_out(batch, tmp_path / "again")["Summary.OUT"]
    assert again.read_bytes() == files["Summary.OUT"].read_bytes()  # deterministic


def test_scenarios_keep_their_planting_date(tmp_path):
    tab = pd.DataFrame(
        {"year": [1978, 1979], "sowing_shift": [0, 0], "PDAT": [1978057, 1979057], "ADAT": [1978200.0, np.nan],
         "MDAT": [1978250.0, 1979251.0], "HWAM": [7000.4, 6900.6], "CWAM": [15000.0, 14000.0]}
    )  # fmt: skip
    scen = ajd.Scenarios("TEST8201_t04", tab, [], {})
    sm = read_summary(fx.write_dssat_out(scen, tmp_path)["Summary.OUT"])
    assert sm.columns.tolist() == [
        "RUNNO",
        "TRNO",
        "CR",
        "MODEL",
        "EXNAME",
        "PDAT",
        "ADAT",
        "MDAT",
        "CWAM",
        "HWAM",
    ]
    assert sm["PDAT"].tolist() == [1978057, 1979057] and sm["HWAM"].tolist() == [7000, 6901]


# ---------------------------------------------------------------------------------- what is refused
def test_dssat_s_own_runs_are_not_written_as_agri_jax_output(tmp_path):
    with pytest.raises(TypeError, match="DSSAT's own run"):
        fx.write_dssat_out(_reference(), tmp_path)
    with pytest.raises(TypeError, match="DSSAT's own run"):
        fx.write_dssat_out([_out_season(), _reference()], tmp_path)
    with pytest.raises(TypeError, match=r"DSSAT's own Summary\.OUT"):
        fx.write_dssat_out(ajd.DssatBatch("TEST8201_t04", _batch().table, 1.0, 6), tmp_path)
    with pytest.raises(TypeError, match="cannot export a dict"):
        fx.write_dssat_out({"lai": 1}, tmp_path)
    with pytest.raises(ValueError, match="empty"):
        fx.write_dssat_out([], tmp_path)
    assert not list(tmp_path.iterdir())  # nothing was written for any of them


def test_the_csv_writers_point_to_the_dssat_format_writer(tmp_path):
    s = _out_season()
    head = "\n".join(_lines(fx.write_daily_csv(s, tmp_path / "d.csv"))[:6])
    assert "NOT a DSSAT output file" in head and "not a genuine PlantGro.OUT" in head
    assert "write_dssat_out" in head and "not produced by the DSSAT program" in head
    for doc in (
        fx.__doc__,
        fx.write_csv.__doc__,
        fx.write_daily_csv.__doc__,
        fx.write_summary_csv.__doc__,
        ajd.Season.write_csv.__doc__,
        ajd.BatchResult.write_csv.__doc__,
    ):
        assert "write_dssat_out" in (doc or "")
    assert "genuine" not in (ajd.Season.write_csv.__doc__ or "")
    assert "write_dssat_out" in fx.to_xarray(s).attrs["note"]
    # the CSV files are unchanged by the new writer
    daily = fx.read_csv(fx.write_daily_csv(s, tmp_path / "e.csv"))
    assert daily.columns.tolist()[:5] == ["RUNNO", "TRNO", "YEAR", "DOY", "DAP"]
