"""The DSSAT-format ``.OUT`` files of ``aj.export`` on UFGA8201 treatment 4, against the files of the
``dscsm048`` run of the same treatment (slow).

* **Layout.** ``PlantGro.OUT``, ``SoilWat.OUT`` and ``Summary.OUT`` written by
  :func:`agrijax.facade_export.write_dssat_out` have DSSAT's preamble (the ``*RUN`` block is the real
  one, line for line), the ``@`` line of the real file with the columns Agri-JAX does not simulate
  taken out (names, widths and alignment of the rest unchanged), data rows as wide as the header, and
  the decimals DSSAT prints in each column.
* **Readers.** ``read_plantgro``, ``read_soilwat`` and ``read_summary`` parse both files with the same
  columns (ours a subset, in DSSAT's order), the same types, the same ``RUN`` / ``TRNO`` / ``YEAR`` /
  ``DOY`` / ``DAS`` / ``DAP`` / ``DATE`` and the same dates in ``Summary.OUT``.
* **Numbers.** What the files hold is the season's numbers rounded as DSSAT prints them, within the
  acceptance of the other facade tests (``tests/integration/test_facade_export_dssat.py``).
* **Several runs.** Two seasons in one file set carry the real ``*RUN`` blocks of their treatments.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from agrijax import facade_export as fx
from agrijax.io.dssat import read_plantgro, read_soilwat, read_summary
from agrijax.io.dssat._fixed import header_tokens
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the DSSAT example data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the DSSAT day runs in float64"),
]

FILES = ("PlantGro.OUT", "SoilWat.OUT", "Summary.OUT")
READERS = {"PlantGro.OUT": read_plantgro, "SoilWat.OUT": read_soilwat, "Summary.OUT": read_summary}
#: a season against dscsm048 (the free-run acceptance: yield within 2 %, measured 3e-5 here)
SEASON_YIELD_RTOL = 1e-3
#: the daily series the facade compares with dscsm048: RMSE / DSSAT maximum
DAILY_RMSE_OVER_MAX = 2e-3
#: every other written PlantGro.OUT column: largest absolute difference over DSSAT's maximum of the column
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


@pytest.fixture(scope="module")
def ours(season, tmp_path_factory) -> dict[str, Path]:
    return season.write_dssat_out(tmp_path_factory.mktemp("dssat_out"))


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="latin-1").split("\n")


def _table(lines: list[str]) -> tuple[int, str, list[str]]:
    """Index of the first ``@`` line, the line, and the data rows under it."""
    i = next(k for k, ln in enumerate(lines) if ln.startswith("@"))
    rows = []
    for ln in lines[i + 1 :]:
        if not ln.strip() or ln.startswith(("@", "*")):
            break
        if not ln.lstrip().startswith("!"):
            rows.append(ln)
    return i, lines[i], rows


def _fields(header: str) -> dict[str, tuple[int, int]]:
    """Column -> the span of its field: from the end of the previous header token to the end of its own."""
    prev, out = 0, {}
    for name, _, end in header_tokens(header):
        out[name] = (prev, end)
        prev = end
    return out


def _decimals(rows: list[str], span: tuple[int, int]) -> set[int]:
    """The numbers of decimals DSSAT prints in a column (missing values, ``-99``, left out)."""
    found = set()
    for r in rows:
        t = r[span[0] : span[1]].strip()
        if t and t not in ("-99", "-99.0", "-99.00", "-99.000") and t.lstrip("-").replace(".", "").isdigit():
            found.add(len(t.split(".")[1]) if "." in t else 0)
    return found


# ---------------------------------------------------------------------------------- layout
@pytest.mark.parametrize("name", FILES)
def test_preamble_is_dssat_s_with_the_statement_that_agri_jax_wrote_it(name, ours, ref, season):
    mine, real = _lines(ours[name]), _lines(ref.out / name)
    i, _, _ = _table(mine)
    j, _, _ = _table(real)
    pre_mine = [ln for ln in mine[:i] if not ln.startswith("!")]
    pre_real = [ln for ln in real[:j] if not ln.startswith("!")]
    assert len(pre_mine) == len(pre_real), (pre_mine, pre_real)
    assert any(ln.startswith("!") for ln in mine[:i])  # the comment lines sit where DSSAT's own do
    if name == "Summary.OUT":
        title = "*SUMMARY : " + season.labels["experiment"]
        assert pre_mine[0].startswith(title) and pre_real[0].startswith(title)
        assert pre_mine[1] == pre_real[1] == ""
        statement = pre_mine[0]
    else:
        assert pre_mine[0].startswith(pre_real[0])  # the file's title
        assert pre_mine[1] == pre_real[1] == "" and pre_mine[3] == pre_real[3] == ""
        assert pre_real[2].startswith("*DSSAT Cropping System Model Ver. 4.8.6")  # DSSAT's own line
        assert pre_mine[4:] == pre_real[4:]  # the run header, line for line
        assert pre_mine[4].startswith("*RUN   1        : IRRIGATED HIGH NITROGEN   MZCER048 UFGA8201    4")
        statement = "\n".join(pre_mine[:3])
    assert "Agri-JAX" in statement and "NOT produced by the DSSAT program" in statement
    assert "precision float64" in statement and "soil evaporation Ritchie (MESEV R)" in statement
    assert "DSSAT Cropping System Model Ver." not in statement


@pytest.mark.parametrize("name", FILES)
def test_columns_widths_and_formats_are_those_of_the_real_file(name, ours, ref):
    _, oh, orows = _table(_lines(ours[name]))
    _, rh, rrows = _table(_lines(ref.out / name))
    of, rf = _fields(oh), _fields(rh)
    names = list(of)
    assert set(names) <= set(rf), sorted(set(names) - set(rf))
    assert names == [n for n in rf if n in of]  # DSSAT's order
    # the real "@" line without the columns that are not written: names, alignment, widths, positions
    bare = rh.replace("@", " ", 1)
    packed = "@" + "".join(bare[rf[n][0] : rf[n][1]] for n in names)[1:]
    assert oh == packed
    assert {n: (b - a) for n, (a, b) in of.items()} == {n: (b - a) for n in names for a, b in [rf[n]]}
    # data rows are as wide as their header, as in DSSAT's file
    assert {len(r) for r in rrows} == {len(rh)}
    assert {len(r) for r in orows} == {len(oh)} and len(orows) >= (1 if name == "Summary.OUT" else 100)
    # the same number of decimals in every column of numbers
    for n in names:
        if n in ("CR", "MODEL", "EXNAME", "TNAM"):
            continue
        assert _decimals(orows, of[n]) == _decimals(rrows, rf[n]), n
    if name == "Summary.OUT":  # the identifiers and the dates: the same characters as DSSAT's row
        for n in (
            "RUNNO",
            "TRNO",
            "CR",
            "MODEL",
            "EXNAME",
            "TNAM",
            "SDAT",
            "PDAT",
            "EDAT",
            "ADAT",
            "MDAT",
            "HDAT",
        ):
            assert orows[0][slice(*of[n])] == rrows[0][slice(*rf[n])], n


@pytest.mark.parametrize("name", FILES)
def test_the_readers_parse_ours_and_dssat_s_identically_shaped(name, ours, ref):
    a, b = READERS[name](ours[name]), READERS[name](ref.out / name)
    assert list(a.columns) == [c for c in b.columns if c in a.columns]  # a subset, in DSSAT's order
    assert not a.empty and len(a.columns) >= 7
    for c in a.columns:
        assert a[c].dtype.kind == b[c].dtype.kind, (c, a[c].dtype, b[c].dtype)
    if name == "Summary.OUT":
        assert len(a) == len(b) == 1
        for c in (
            "RUNNO",
            "TRNO",
            "CR",
            "MODEL",
            "EXNAME",
            "TNAM",
            "SDAT",
            "PDAT",
            "EDAT",
            "ADAT",
            "MDAT",
            "HDAT",
        ):
            assert a.loc[0, c] == b.loc[0, c], c
        for c in ("CWAM", "HWAM"):
            assert abs(a.loc[0, c] / b.loc[0, c] - 1) < SEASON_YIELD_RTOL, c
        return
    keys = ["RUN", "TRNO", "YEAR", "DOY", "DAS", *(["DAP"] if name == "PlantGro.OUT" else []), "DATE"]
    if name == "SoilWat.OUT":  # dscsm048 also writes the initial state, DAS 0
        assert b["DAS"].iloc[0] == 0 and a["DAS"].iloc[0] == 1
        b = b.iloc[1:].reset_index(drop=True)
    assert len(a) == len(b), (len(a), len(b))
    for c in keys:
        assert a[c].tolist() == b[c].tolist(), c


def test_the_numbers_are_the_seasons_and_agree_with_dssat(ours, ref, season):
    a, b = read_plantgro(ours["PlantGro.OUT"]), read_plantgro(ref.out / "PlantGro.OUT")
    d = season.to_frame(full=True, dssat_names=True)
    names = [c for c in a.columns if c in {s.dssat for s in fx.SERIES.values()}]
    # PlantGro.OUT starts on the planting day: every one of its days is a day of the season
    m = a.merge(d[["YEAR", "DOY", *names]], on=["YEAR", "DOY"], suffixes=("", "_season"))
    assert len(m) == len(a) == len(b)
    decimals = {"L#SD": 1, "LAID": 2, "WSPD": 3, "WSGD": 3, "EWSD": 3, "RDPD": 2, "DTTD": 2}
    print("\ncolumn   max |diff|   DSSAT max   max|diff| / max")
    for c in names:
        top = float(np.nanmax(np.abs(b[c])))
        worst = float(np.nanmax(np.abs(a[c] - b[c])))
        print(f"{c:6s} {worst:12.5g} {top:12.5g} {worst / top if top else float('nan'):12.3g}")
        if c in ("LAID", "CWAD", "GWAD"):
            rmse = float(np.sqrt(np.nanmean((a[c] - b[c]) ** 2)))
            assert rmse / top < DAILY_RMSE_OVER_MAX, (c, rmse, top)
        elif top:
            assert worst / top < EXTRA_MAX_OVER_MAX, (c, worst, top)
        # the file holds the season's number, rounded to the printed precision
        half = 0.5 * 10.0 ** -decimals.get(c, 0)
        np.testing.assert_allclose(m[c], m[f"{c}_season"], rtol=0, atol=half + 1e-9, err_msg=c)
    sa, sb = read_soilwat(ours["SoilWat.OUT"]), read_soilwat(ref.out / "SoilWat.OUT")
    sb = sb[sb["DAS"] >= 1].reset_index(drop=True)
    assert np.sqrt(np.mean((sa["SWTD"] - sb["SWTD"]) ** 2)) / np.max(np.abs(sb["SWTD"])) < DAILY_RMSE_OVER_MAX
    np.testing.assert_allclose(sa["SWTD"], season.daily["swtd"], rtol=0, atol=0.5 + 1e-9)


def test_two_seasons_in_one_file_set_carry_the_real_run_blocks(exp, season, ref, tmp_path):
    other = exp.run(treatment=6)
    ref6 = exp.reference(treatment=6)
    files = fx.write_dssat_out([season, other], tmp_path)
    for name in ("PlantGro.OUT", "SoilWat.OUT"):
        mine = _lines(files[name])
        starts = [i for i, ln in enumerate(mine) if ln.startswith("*RUN")]
        assert len(starts) == 2
        for k, (run, real) in enumerate(((ref, 4), (ref6, 6)), start=1):
            text = _lines(run.out / name)
            j = next(i for i, ln in enumerate(text) if ln.startswith("*RUN"))
            block = text[j : j + 6]
            block[0] = block[0].replace("*RUN   1", f"*RUN   {k}", 1)  # numbered in the order given
            assert mine[starts[k - 1] : starts[k - 1] + 6] == block, (name, real)
            assert mine[starts[k - 1] - 1] == "" and mine[starts[k - 1] - 2].startswith("*Agri-JAX")
    pg = read_plantgro(files["PlantGro.OUT"])
    assert sorted(pg["RUN"].unique().tolist()) == [1, 2] and sorted(pg["TRNO"].unique().tolist()) == [4, 6]
    sm = read_summary(files["Summary.OUT"])
    assert sm["RUNNO"].tolist() == [1, 2] and sm["TRNO"].tolist() == [4, 6]
    real = read_summary(ref6.out / "Summary.OUT")
    assert sm.loc[1, "TNAM"] == real.loc[0, "TNAM"] and sm.loc[1, "PDAT"] == real.loc[0, "PDAT"]
