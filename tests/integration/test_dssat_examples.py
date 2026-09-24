"""Every official Maize example of DSSAT-CSM v4.8.6.0, run and cross-checked.

All ``*.MZX`` of ``<engine>/example_data/Maize`` (every treatment, run mode ``A``) are run once per
session with the locally built ``dscsm048`` (``source/build486``, v4.8.6.0) through
:func:`agri_jax.port.run_fortran.run_dscsm`; each run stages into its own ``mkdtemp`` directory
under ``~/agri_jax_data/run`` (removed on success), outputs go to a private ``tmp_path_factory``
directory. The parsed outputs are then checked against references that do not go through the
parser being tested:

* ``Summary.OUT`` against ``Evaluate.OUT`` (a second table DSSAT writes from the same run, with
  other columns and layout): ``HWAM CWAM H#AM HWUM LAIX`` and anthesis / maturity as days after
  planting (``ADAT - PDAT`` = ``ADAPS``); the run list and ``TNAM`` against the FileX treatments.
* ``.MZA`` observations against the measured columns DSSAT copies into ``Evaluate.OUT`` (dates
  through DSSAT's ``READA_Dates`` rule, :func:`agri_jax.io.dssat.observed_date`).
* ``PlantGro.OUT`` final day against ``Summary.OUT`` (``CWAM``, ``HWAM``, max ``LAID`` ~ ``LAIX``,
  last day = ``HDAT``).
* ``SoilWat.OUT`` / ``ET.OUT``: the soil water balance closes (``d SWTD = d(PREC + IRRC - ROFC -
  DRNC - ETAC)``), layer contents times layer thicknesses sum to ``SWTD``, and the season totals
  equal the ``Summary.OUT`` ones (``PRCM IRCM ROCM DRCM ETCM EPCM ESCM``).
* ``read_wth(..., dssat_spans=True)`` of the station files against the weather DSSAT itself used
  (``Weather.OUT``).
* DSSAT as the reader of our writers: rerunning with every weather, soil and genotype file rewritten
  by ``write_wth`` / ``write_sol`` / ``write_cul`` / ``write_eco`` / ``write_spe`` gives identical
  outputs; a cultivar edited through ``write_cul`` is echoed by DSSAT (``OVERVIEW.OUT``) with the
  new values and equals the same edit made by hand in the text.
* Determinism: UFGA8201 five times in one process and four times in parallel processes gives
  identical outputs (timestamps masked).

Per-treatment results are recorded in ``~/agri_jax_data/validation/dssat_examples/summary.csv``;
when that file exists the test compares against it (``AGRI_JAX_UPDATE_BASELINE=1`` rewrites it).
The simulated columns are also compared with the reviewed, committed baseline
``tests/fixtures/dssat/dscsm048_examples_simulated.csv`` (independent of the private data tree).
"""

from __future__ import annotations

import csv
import io
import math
import multiprocessing as mp
import os
import re
import shutil
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_jax.io.dssat import (
    observed_date,
    parse_dssat_date,
    read_cul,
    read_eco,
    read_et,
    read_evaluate,
    read_filex,
    read_out,
    read_plantgro,
    read_soilwat,
    read_sol,
    read_spe,
    read_summary,
    read_wth,
    weather_stations,
    write_cul,
    write_eco,
    write_sol,
    write_spe,
    write_wth,
)
from agri_jax.io.dssat._fixed import read_lines

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
DSCSM = Path(
    os.environ.get("AGRI_JAX_DSCSM", str(DSSAT_ENGINE / "source" / "build486" / "bin" / "dscsm048"))
).expanduser()
MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
WEATHER = DSSAT_ENGINE / "example_data" / "Weather"
SOIL = DSSAT_ENGINE / "example_data" / "Soil"
DATA = DSSAT_ENGINE / "source" / "Data"
VALIDATION = (
    Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data"))).expanduser()
    / "validation"
    / "dssat_examples"
)
#: committed baseline (simulated columns of ``summary.csv``, dscsm048 v4.8.6.0, 12 experiments, 76 runs)
BASELINE = Path(__file__).resolve().parents[1] / "fixtures" / "dssat" / "dscsm048_examples_simulated.csv"
EXPERIMENTS = sorted(p.stem for p in MAIZE.glob("*.MZX")) if MAIZE.is_dir() else []
_STAMP = re.compile(r"[A-Z]{3} \d{1,2}, \d{4}[; ]+\d\d:\d\d:\d\d")
_ENGINE_FILES = ("DSCSM048.CTR", "MODEL.ERR", "DSSATPRO.v48", "DSSATPRO.L48", "DSSATPRO.L48.in")


# --------------------------------------------------------------------------- running


def make_engine(root: Path, *, genotype: Path | None = None) -> Path:
    """An engine tree ``root/bin`` in the layout ``run_dscsm`` expects, made of symlinks to the
    v4.8.6.0 ``source/Data`` files and the ``build486`` binary (``genotype`` replaces
    ``Data/Genotype``)."""
    b = root / "bin"
    b.mkdir(parents=True, exist_ok=True)
    for f in [*DATA.glob("*.CDE"), *(DATA / n for n in _ENGINE_FILES), DATA / "StandardData"]:
        if f.exists():
            (b / f.name).symlink_to(f)
    (b / "Genotype").symlink_to(genotype if genotype is not None else DATA / "Genotype")
    (b / "dscsm048").symlink_to(DSCSM)
    return root


def run_example(
    name: str,
    engine: Path,
    work: Path,
    *,
    weather_dir: Path = WEATHER,
    soil_dir: Path = SOIL,
) -> Path:
    """Run every treatment of ``<name>.MZX``; return the directory holding its ``*.OUT``."""
    from agri_jax.port.run_fortran import run_dscsm

    exp = work / "exp"
    exp.mkdir(parents=True)
    for f in MAIZE.glob(name + ".*"):
        shutil.copy2(f, exp / f.name)
    stations = {w[:4].upper() for w in weather_stations(exp / f"{name}.MZX")}
    extra = [f for f in sorted(weather_dir.glob("*.WTH")) if f.name[:4].upper() in stations]
    extra += sorted((DATA / "Pest").glob("*.PST"))  # pest-damage examples (IUAF9902/9903)
    r = run_dscsm(
        exp,
        work / "out",
        model="MZCER048",
        run_mode="A",
        experiment_file=f"{name}.MZX",
        engine=engine,
        weather_dir=weather_dir,
        soil_dir=soil_dir,
        extra_files=extra,
    )
    assert r.summary_path is not None
    return r.out_dir


def masked(path: Path) -> str:
    """File text with DSSAT's run timestamps removed."""
    return _STAMP.sub("<stamp>", path.read_text(errors="replace"))


@pytest.fixture(scope="session")
def engine486(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not DSCSM.is_file():
        pytest.skip(f"dscsm048 v4.8.6.0 binary not found at {DSCSM}")
    if not (DATA / "DSCSM048.CTR").is_file() or not MAIZE.is_dir():
        pytest.skip(f"DSSAT source Data / Maize examples not found under {DSSAT_ENGINE}")
    return make_engine(tmp_path_factory.mktemp("engine486"))


@pytest.fixture(scope="session")
def example_out(engine486: Path, tmp_path_factory: pytest.TempPathFactory) -> Callable[[str], Path]:
    """``example_out(name)``: output dir of the (cached) run of experiment ``name``."""
    cache: dict[str, Path] = {}

    def get(name: str) -> Path:
        if name not in cache:
            cache[name] = run_example(name, engine486, tmp_path_factory.mktemp(name))
        return cache[name]

    return get


@dataclass
class Outs:
    name: str
    out: Path
    summary: pd.DataFrame
    evaluate: pd.DataFrame

    @property
    def maize(self) -> pd.DataFrame:
        """Summary rows of the maize runs (fallow runs of EBPL8501 excluded), with Evaluate."""
        s = self.summary[self.summary["CR"] == "MZ"]
        return s.merge(self.evaluate, left_on="RUNNO", right_on="RUN", suffixes=("", "_ev"))


@pytest.fixture
def outs(request: pytest.FixtureRequest, example_out: Callable[[str], Path]) -> Outs:
    name = request.param
    out = example_out(name)
    return Outs(name, out, read_summary(out / "Summary.OUT"), read_evaluate(out / "Evaluate.OUT"))


def _dap(code: float, pdat: float) -> float:
    if np.isnan(code) or np.isnan(pdat):
        return math.nan
    return float((parse_dssat_date(str(int(code))) - parse_dssat_date(str(int(pdat)))).days)


def _eq_nan(a: np.ndarray, b: np.ndarray, *, atol: float = 0.0, rtol: float = 0.0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= atol + rtol * np.abs(b))


per_exp = pytest.mark.parametrize("outs", EXPERIMENTS, indirect=True)


def test_every_example_present() -> None:
    if not MAIZE.is_dir():
        pytest.skip(f"{MAIZE} not found")
    # the v4.8.6.0 tree ships 12 maize experiments; losing one would silently shrink the checks
    assert len(EXPERIMENTS) >= 12, EXPERIMENTS


# --------------------------------------------------------------------------- Summary vs Evaluate / FileX


@per_exp
def test_runs_are_the_filex_treatments(outs: Outs) -> None:
    trt = read_filex(MAIZE / f"{outs.name}.MZX")["TREATMENTS"]
    s = outs.summary
    assert list(s["RUNNO"]) == list(range(1, len(trt) + 1))
    assert list(s["TRNO"]) == [t["N"] for t in trt]
    # TNAM (A25 in Summary.OUT, may contain blanks and commas) is the FileX TNAME
    assert list(s["TNAM"]) == [str(t["TNAME"])[:25].strip() for t in trt]
    assert set(s["EXNAME"]) == {outs.name}
    ev = outs.evaluate.set_index("RUN")
    assert list(ev.index) == list(s["RUNNO"]) and list(ev["TN"]) == list(s["TRNO"])


@per_exp
def test_summary_matches_evaluate(outs: Outs) -> None:
    m = outs.maize
    assert len(m) >= 1
    for col, tol in (("HWAM", 0.0), ("CWAM", 0.0), ("H#AM", 0.0), ("HWUM", 5e-5)):
        ok = _eq_nan(m[col], m[col + "S"], atol=tol)
        assert ok.all(), (col, m.loc[~ok, ["RUNNO", col, col + "S"]].to_dict("records"))
    # LAIX: one decimal in Summary.OUT, two in Evaluate.OUT
    ok = _eq_nan(m["LAIX"], m["LAIXS"], atol=0.05 + 1e-9)
    assert ok.all(), m.loc[~ok, ["RUNNO", "LAIX", "LAIXS"]].to_dict("records")
    for date_col, dap_col in (("ADAT", "ADAPS"), ("MDAT", "MDAPS")):
        dap = np.array([_dap(d, p) for d, p in zip(m[date_col], m["PDAT"], strict=True)])
        ok = _eq_nan(dap, m[dap_col])
        assert ok.all(), (date_col, list(zip(dap[~ok], m.loc[~ok, dap_col], strict=True)))
    assert (m["HWAM"] >= 0).all() and (m["CWAM"] >= m["HWAM"]).all() and (m["CWAM"] > 0).all()


# --------------------------------------------------------------------------- observations


#: .MZA columns DSSAT does not copy into Evaluate.OUT (no simulated counterpart there)
_NOT_EVALUATED = {"BWAH", "L#SD", "CHTA", "HWAH", "TNAM", "-99"}


@per_exp
def test_observed_match_evaluate(outs: Outs) -> None:
    mza = MAIZE / f"{outs.name}.MZA"
    ev = outs.evaluate.set_index("TN")
    measured = [c for c in ev.columns if c.endswith("M") and c[:-1] + "S" in ev.columns]
    if not mza.is_file():  # pest-damage examples: no observations, DSSAT writes -99
        assert ev[measured].isna().all().all()
        return
    obs = read_out(mza).set_index("TRNO")
    s = outs.summary.set_index("TRNO")
    matched: list[str] = []
    for col in obs.columns:
        if col in _NOT_EVALUATED or col == "RUN":
            continue
        if col in ("ADAT", "MDAT"):
            key = col[0] + "DAPM"
            want = [
                math.nan
                if np.isnan(v)
                else float(
                    (observed_date(v, s.loc[t, "SDAT"]) - parse_dssat_date(str(int(s.loc[t, "PDAT"])))).days
                )
                for t, v in obs[col].items()
            ]
        else:
            key = col + "M"
            want = obs[col].to_numpy(dtype=float)
        assert key in ev.columns, f"{col}: no {key} in Evaluate.OUT"
        got = ev.loc[obs.index, key].to_numpy(dtype=float)
        ok = _eq_nan(got, np.asarray(want, dtype=float), atol=5e-4, rtol=5e-4)
        assert ok.all(), (col, list(zip(np.asarray(want)[~ok], got[~ok], strict=True)))
        matched.append(col)
    if "HWAM" in obs.columns:
        assert "HWAM" in matched and "CWAM" in matched


# --------------------------------------------------------------------------- PlantGro


@per_exp
def test_plantgro_final_day_matches_summary(outs: Outs) -> None:
    g = read_plantgro(outs.out / "PlantGro.OUT")
    m = outs.maize.set_index("RUNNO")
    assert sorted(set(g["RUN"])) == sorted(m.index)
    fin = g.groupby("RUN").tail(1).set_index("RUN").loc[m.index]
    np.testing.assert_array_equal((fin["YEAR"] * 1000 + fin["DOY"]).to_numpy(), m["HDAT"].to_numpy())
    rel = np.abs(fin["CWAD"] - m["CWAM"]) / m["CWAM"].clip(lower=1.0)
    assert (rel <= 0.01).all(), rel.to_dict()
    rel = np.abs(fin["GWAD"] - m["HWAM"]) / m["HWAM"].clip(lower=1.0)
    assert (rel <= 0.01).all(), rel.to_dict()
    lai_max = g.groupby("RUN")["LAID"].max().loc[m.index]
    assert (np.abs(lai_max - m["LAIX"]) <= 0.05 + 1e-9).all()  # LAIX has one decimal
    assert (np.abs(lai_max - m["LAIXS"]) <= 0.005 + 1e-9).all()  # Evaluate: two decimals


# --------------------------------------------------------------------------- water


def _layer_depths(path: Path) -> dict[int, list[tuple[int, int]]]:
    """Per run, the layer depths (cm) of the ``!  0-5  5-15 ...`` line above the SoilWat table."""
    out: dict[int, list[tuple[int, int]]] = {}
    run = 0
    for ln in read_lines(path):
        m = re.match(r"^\*RUN\s+(\d+)", ln)
        if m:
            run = int(m.group(1))
        elif ln.startswith("!") and re.search(r"\d+-\d+\s+\d+-\d+", ln):
            out[run] = [(int(a), int(b)) for a, b in re.findall(r"(\d+)-(\d+)", ln)]
    return out


def _efir(irr: dict[int, dict[str, object]], level: int) -> float:
    v = irr.get(level, {}).get("EFIR", -99) if level else -99
    x = float(v) if isinstance(v, (int, float)) else -99.0
    return 1.0 if x <= 0 else x


@per_exp
def test_soil_water_balance_and_totals(outs: Outs) -> None:
    sw_path = outs.out / "SoilWat.OUT"
    fx = read_filex(MAIZE / f"{outs.name}.MZX")
    if not sw_path.is_file():  # water balance off, or its daily output switched off (GAGR0201)
        ctrl = fx["SIMULATION CONTROLS"].values()
        assert all(c["OPTIONS"]["WATER"] == "N" or c["OUTPUTS"]["WAOUT"] == "N" for c in ctrl)
        return
    irr = fx.get("IRRIGATION AND WATER MANAGEMENT", {})
    mi = {t["N"]: t["MI"] for t in fx["TREATMENTS"]}
    w = read_soilwat(sw_path)
    et = read_et(outs.out / "ET.OUT")
    s = outs.summary.set_index("RUNNO")
    depths = _layer_depths(sw_path)
    for run, g in w.groupby("RUN"):
        q = et[et["RUN"] == run]
        m = g.merge(q[["DATE", "ETAC"]], on="DATE")
        assert len(m) >= len(g) - 1
        flux = m["PREC"] + m["IRRC"] - m["ROFC"] - m["DRNC"] - m["ETAC"]
        resid = (m["SWTD"] - m["SWTD"].iloc[0]) - (flux - flux.iloc[0])
        # integers (SWTD PREC IRRC ROFC DRNC) and 0.1 mm ETAC: at most 2*0.5 + 4*1 + 0.1 mm
        assert float(resid.abs().max()) <= 5.1, (run, float(resid.abs().max()))
        d = depths[int(run)]
        tot = sum(g[f"SW{i}D"] * (b - a) * 10.0 for i, (a, b) in enumerate(d, start=1))
        # SW*D have 3 decimals; observed max 1.35 mm, a shifted column is off by tens of mm
        assert float((tot - g["SWTD"]).abs().max()) <= 2.0, run
        fin, fe, sr = g.iloc[-1], q.iloc[-1], s.loc[run]
        pairs = {
            "PRCM": fin["PREC"],
            # IRCM is the water applied, IRRC what reaches the soil (EFIR efficiency)
            "IRCM": fin["IRRC"] / _efir(irr, mi[int(sr["TRNO"])]),
            "ROCM": fin["ROFC"],
            "DRCM": fin["DRNC"],
            "ETCM": fe["ETAC"],
            "EPCM": fe["EPAC"],
            "ESCM": fe["ESAC"] + fe["EMAC"] + fe["EFAC"],  # soil + mulch + flood evaporation
        }
        for k, v in pairs.items():
            tol = 1.0 + (1.0 if k == "IRCM" else 0.0)  # IRRC / EFIR amplifies its rounding
            assert abs(float(sr[k]) - float(v)) <= tol + 1e-9, (run, k, sr[k], v)


# --------------------------------------------------------------------------- weather as DSSAT reads it


_ENV_ORDER = ("EDAY", "ERAD", "EMAX", "EMIN", "ERAIN", "ECO2", "EDEW", "EWIND")
_ENV_OPS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "A": lambda x, v: x + v,
    "S": lambda x, v: x - v,
    "M": lambda x, v: x * v,
    "R": lambda x, v: np.full_like(x, v),
}


def _env_mods(filex: Path) -> dict[int, tuple[pd.Timestamp, dict[str, tuple[str, float]]]]:
    """``*ENVIRONMENT MODIFICATIONS`` levels as ``{level: (ODATE, {EDAY: (op, value), ...})}``
    (ops A/S/M/R are written glued to or before their value: ``R18.7``, ``R  25``)."""
    out: dict[int, tuple[pd.Timestamp, dict[str, tuple[str, float]]]] = {}
    sec = False
    for ln in read_lines(filex):
        if ln.startswith("*"):
            sec = ln.upper().startswith("*ENVIRONMENT")
            continue
        if not sec or ln.startswith("@") or not ln.strip() or ln.lstrip().startswith("!"):
            continue
        lev, odate, rest = ln.split(None, 2)
        pairs = re.findall(r"([ASMR])\s*(-?[\d.]+)", rest)
        ops = {k: (op, float(v)) for k, (op, v) in zip(_ENV_ORDER, pairs, strict=False)}
        out[int(lev)] = (pd.Timestamp(parse_dssat_date(odate)), ops)
    return out


@per_exp
def test_weather_as_dssat_reads_it(outs: Outs) -> None:
    """The daily weather DSSAT simulated with (Weather.OUT) is ``read_wth(dssat_spans=True)`` of
    the station files, after the FileX environment modifications (GAGR0201 growth chambers)."""
    filex = MAIZE / f"{outs.name}.MZX"
    stations = {w[:4].upper() for w in weather_stations(filex)}
    files = [f for f in sorted(WEATHER.glob("*.WTH")) if f.name[:4].upper() in stations]
    wx = pd.concat([read_wth(f, dssat_spans=True) for f in files]).drop_duplicates("date").set_index("date")
    wx0 = pd.concat([read_wth(f) for f in files]).drop_duplicates("date").set_index("date")
    used_all = read_out(outs.out / "Weather.OUT")
    me = {t["N"]: t["ME"] for t in read_filex(filex)["TREATMENTS"]}
    trno = outs.summary.set_index("RUNNO")["TRNO"]
    mods = _env_mods(filex)
    pairs = (
        ("srad", "SRAD", "ERAD"),
        ("tmax", "TMXD", "EMAX"),
        ("tmin", "TMND", "EMIN"),
        ("rain", "PRED", "ERAIN"),
    )
    checked = 0
    for run, used in used_all.groupby("RUN"):
        used = used.set_index("DATE")
        assert len(used) > 60
        got = wx.loc[used.index].copy()
        # the default (header-aligned) reader agrees with the DSSAT rule on these files
        pd.testing.assert_frame_equal(wx0.loc[used.index], got)
        level = me[int(trno[run])]
        if level:
            odate, ops = mods[level]
            after = got.index >= odate
            for mine, _, key in pairs:
                op, val = ops[key]
                got.loc[after, mine] = _ENV_OPS[op](got.loc[after, mine].to_numpy(), val)
        for mine, theirs, _ in pairs:
            ok = _eq_nan(got[mine].to_numpy(), used[theirs].to_numpy(), atol=0.05 + 1e-9)
            assert ok.all(), (run, mine, got.index[~ok][:5].tolist())
        checked += 1
    assert checked >= len(outs.maize)  # fallow runs (EBPL8501) write no Weather.OUT


# --------------------------------------------------------------------------- DSSAT reads our writers


def _rewrite_inputs(name: str, work: Path) -> tuple[Path, Path, Path]:
    """Weather, soil and genotype dirs with every file the experiment uses rewritten by our
    writers (read -> write, no edits)."""
    wdir, sdir, gdir = work / "wth", work / "sol", work / "gen"
    for d in (wdir, sdir, gdir):
        d.mkdir(parents=True)
    stations = {w[:4].upper() for w in weather_stations(MAIZE / f"{name}.MZX")}
    for f in sorted(WEATHER.glob("*.WTH")):
        if f.name[:4].upper() in stations:
            write_wth(read_wth(f), wdir / f.name)
    fields = read_filex(MAIZE / f"{name}.MZX")["FIELDS"]
    ids = {str(v["ID_SOIL"]) for v in fields.values()}
    for f in sorted(SOIL.glob("*.SOL")):
        text = f.read_text(errors="replace")
        if any(f"*{i}" in text for i in ids):
            profs = read_sol(f)
            write_sol([profs[i] for i in profs if i in ids], sdir / f.name)
    for f in (DATA / "Genotype").iterdir():
        shutil.copy2(f, gdir / f.name)
    write_cul(read_cul(gdir / "MZCER048.CUL"), gdir / "MZCER048.CUL")
    write_eco(read_eco(gdir / "MZCER048.ECO"), gdir / "MZCER048.ECO")
    write_spe(read_spe(gdir / "MZCER048.SPE"), gdir / "MZCER048.SPE")
    return wdir, sdir, gdir


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_dscsm_reads_our_writers(name: str, example_out: Callable[[str], Path], tmp_path: Path) -> None:
    base = example_out(name)
    wdir, sdir, gdir = _rewrite_inputs(name, tmp_path)
    engine = make_engine(tmp_path / "engine", genotype=gdir)
    out = run_example(name, engine, tmp_path / "run", weather_dir=wdir, soil_dir=sdir)
    for f in ("Summary.OUT", "PlantGro.OUT", "Evaluate.OUT", "SoilWat.OUT", "ET.OUT", "Weather.OUT"):
        if (base / f).is_file():
            assert masked(out / f) == masked(base / f), f


def test_write_cul_edit_is_what_dssat_reads(
    engine486: Path, example_out: Callable[[str], Path], tmp_path: Path
) -> None:
    """Edit cultivar IB0035 (UFGA8201) through write_cul: the file equals the same edit made by
    hand, and DSSAT echoes the new coefficients and simulates a different yield."""
    src = DATA / "Genotype" / "MZCER048.CUL"
    cul = read_cul(src)
    assert cul.loc["IB0035", "P1"] == 259.0 and cul.loc["IB0035", "G3"] == 8.168
    cul.loc["IB0035", "P1"] = 271.5
    cul.loc["IB0035", "G3"] = 7.25
    gdir = tmp_path / "gen"
    shutil.copytree(DATA / "Genotype", gdir)
    write_cul(cul, gdir / "MZCER048.CUL")
    # the same edit by hand, in the text
    lines = src.read_bytes().decode("latin-1").split("\r\n" if b"\r\n" in src.read_bytes() else "\n")
    k = next(i for i, ln in enumerate(lines) if ln.startswith("IB0035"))
    ln = lines[k]
    p1 = ln.index(" 259.0") + 1
    ln = ln[:p1] + "271.5" + ln[p1 + 5 :]
    g3 = ln.index(" 8.168") + 1
    ln = ln[:g3] + " 7.25" + ln[g3 + 5 :]
    lines[k] = ln
    by_hand = ("\r\n" if b"\r\n" in src.read_bytes() else "\n").join(lines).encode("latin-1")
    assert (gdir / "MZCER048.CUL").read_bytes() == by_hand
    assert read_cul(gdir / "MZCER048.CUL").loc["IB0035", ["P1", "G3"]].tolist() == [271.5, 7.25]

    out = run_example("UFGA8201", make_engine(tmp_path / "engine", genotype=gdir), tmp_path / "run")
    ov = (out / "OVERVIEW.OUT").read_text(errors="replace")
    assert "P1     : 271.50" in ov and "G3     :  7.250" in ov
    base = read_summary(example_out("UFGA8201") / "Summary.OUT")
    new = read_summary(out / "Summary.OUT")
    assert (new["HWAM"] != base["HWAM"]).any()


# --------------------------------------------------------------------------- determinism


def _run_ufga_masked(args: tuple[str, str]) -> dict[str, str]:
    engine, work = map(Path, args)
    out = run_example("UFGA8201", engine, work)
    return {p.name: masked(p) for p in sorted(out.glob("*.OUT"))}


def test_ufga8201_deterministic_in_process(engine486: Path, tmp_path: Path) -> None:
    runs = [_run_ufga_masked((str(engine486), str(tmp_path / f"r{i}"))) for i in range(5)]
    assert len(runs[0]) >= 10 and "Summary.OUT" in runs[0]
    for r in runs[1:]:
        assert r.keys() == runs[0].keys()
        for k in r:
            assert r[k] == runs[0][k], k


def test_ufga8201_deterministic_parallel_processes(engine486: Path, tmp_path: Path) -> None:
    """Four concurrent runs from separate processes (each stages its own mkdtemp run dir): no
    shared staging directory, no copy race, identical outputs."""
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        runs = list(ex.map(_run_ufga_masked, [(str(engine486), str(tmp_path / f"p{i}")) for i in range(4)]))
    for r in runs[1:]:
        assert r == runs[0]


# --------------------------------------------------------------------------- recorded yields


_CSV_COLUMNS = (
    "experiment", "run", "trno", "tnam", "crop", "model", "pdat", "adat", "mdat", "hdat",
    "hwam", "hwam_obs", "cwam", "cwam_obs", "adap", "adap_obs", "mdap", "mdap_obs",
    "laix", "prcm", "etcm",
)  # fmt: skip


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return "" if math.isnan(v) else (str(int(v)) if v.is_integer() else f"{v:.6g}")
    return str(v)


def test_record_summary_csv(example_out: Callable[[str], Path]) -> None:
    rows: list[list[str]] = []
    for name in EXPERIMENTS:
        out = example_out(name)
        s = read_summary(out / "Summary.OUT")
        ev = read_evaluate(out / "Evaluate.OUT").set_index("RUN")
        for _, r in s.iterrows():
            e = ev.loc[r["RUNNO"]]
            vals = [
                name, r["RUNNO"], r["TRNO"], r["TNAM"], r["CR"], r["MODEL"], r["PDAT"], r["ADAT"],
                r["MDAT"], r["HDAT"], r["HWAM"], e["HWAMM"], r["CWAM"], e["CWAMM"], e["ADAPS"],
                e["ADAPM"], e["MDAPS"], e["MDAPM"], r["LAIX"], r["PRCM"], r["ETCM"],
            ]  # fmt: skip
            rows.append([_fmt(float(v)) if isinstance(v, (int, float, np.number)) else _fmt(v) for v in vals])
    buf = io.StringIO()
    wr = csv.writer(buf, lineterminator="\n")
    wr.writerow(_CSV_COLUMNS)
    wr.writerows(rows)
    text = buf.getvalue()
    assert len(rows) >= 70  # 76 runs in the v4.8.6.0 examples
    # the reviewed baseline in git: the simulated columns only (model output of the BSD-3 binary;
    # the observed values and treatment names come from dssat-csm-data, which is not redistributed)
    with open(BASELINE, newline="") as fh:
        base = list(csv.reader(fh))
    idx = [_CSV_COLUMNS.index(c) for c in base[0]]
    fresh = [[r[i] for i in idx] for r in rows]
    assert base[1:] == fresh, (
        f"dscsm048 example outputs differ from the committed {BASELINE.name} (binary, data or parser "
        "changed?); review the diff and regenerate the fixture"
    )
    target = VALIDATION / "summary.csv"
    if target.is_file() and os.environ.get("AGRI_JAX_UPDATE_BASELINE", "") != "1":
        old = target.read_text()
        assert old == text, (
            f"{target} differs from this run (binary, data or parser changed?); "
            "rerun with AGRI_JAX_UPDATE_BASELINE=1 to record the new baseline"
        )
    elif VALIDATION.parent.is_dir():
        VALIDATION.mkdir(exist_ok=True)
        tmp = target.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(target)
