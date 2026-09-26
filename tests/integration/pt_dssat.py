"""Priestley-Taylor PET against dscsm048 (DSSAT-CSM v4.8.6.0) on the Maize examples (H1 item 2).

Helper module of ``test_pt_dssat.py`` and a script (``python tests/integration/pt_dssat.py``) that
writes the per-experiment table. Every ``*.MZX`` of ``<engine>/example_data/Maize`` is run, every
treatment, through :func:`agrijax.port.run_fortran.run_dscsm` with three FileX edits
(:func:`patch_filex`): ``EVAPO = R`` on every ``@N METHODS`` row (Priestley-Taylor; all twelve
examples already use it), and on every ``@N OUTPUTS`` row ``WAOUT = Y`` (water outputs; GAGR0201 has
them off) and ``VBOSE = D`` (detailed outputs, for ``SoilDyn.OUT``). Neither output switch changes
the simulation. The daily potential ET ``EO`` of ``PETPT`` is ``EOAA`` of ``ET.OUT``
(``SPAM/SPSUBS.for`` OPSPAM: the print-interval mean of ``EO``; the examples print daily,
``FROUT = 1``). The reference ET column ``REFA`` is -99 for this method (only the ASCE methods write
it), so there is no ``ET0`` to compare.

The kernel :func:`agrijax.processes.pet.priestley_taylor` is driven, day by day, with the inputs
``PETPT`` receives (``SPAM/SPAM.for`` line 300 -> ``SPAM/PET.for`` line 94):

* ``SRAD``, ``TMAX``, ``TMIN``: the station files read as DSSAT reads them
  (``read_wth(..., dssat_spans=True)``), checked against ``SRAA``/``TMAXA``/``TMINA`` of the same
  ``ET.OUT`` rows. Treatments with a FileX environment modification (all six of GAGR0201, growth
  chambers) are left out: their weather is not the station file.
* ``XHLAI``: the maize healthy LAI (``MZ_GROSUB.for`` line 1819, ``XHLAI = LAI``) of the
  previous day's ``PlantGro.OUT`` row (``LAID``; the plant integrates after SPAM in ``LAND.for``,
  so day ``t`` sees the end of day ``t - 1``); 0 before the first and after the last ``PlantGro``
  row (``plant.for`` RATE sets ``XHLAI = 0`` outside the crop season).
* ``ET_ALB = MSALB`` (``SPAM.for`` lines 290-297; no flood in these runs), set by
  ``SOILDYN.for`` ``ALBEDO_avg`` (lines 1528-1530 and 1555-1562): ``FF = (SW1 - 0.03) / (DUL(1) -
  0.03)``, ``FF = MAX(0.0, MIN(2.0, FF))``, ``SWALB = SALB * (1.0 - 0.45 * FF)`` and, because
  ``INFIL = S`` is one of ``'RSM'``, ``MSALB = MULCHCOVER * MULCHALB + (1.0 - MULCHCOVER) * SWALB``
  with ``MULCHALB = 0.45`` (``Mulch/MULCHLAYER.for`` line 47). ``ALBEDO_avg`` runs at the top of
  the SOILDYN RATE step (line 1053), before the soil, mulch and water modules of the day, so every
  input is the previous day's printed end-of-day value: ``SW1`` = ``SW1D`` (``SoilWat.OUT``),
  ``DUL(1)`` = ``DUL1`` (``SoilDyn.OUT``; DSSAT changes it daily with the soil organic matter,
  lines 1072-1152 and 1360), ``MULCHCOVER`` = ``MCFD`` (``Mulch.OUT``). On the first day the
  initial ``DUL(1)`` of ``OVERVIEW.OUT`` is used when ``SoilDyn.OUT`` has no day-0 row. ``SALB``
  (``SOIL ALBEDO``) comes from ``OVERVIEW.OUT``; runs without an ``OVERVIEW.OUT`` block (the fallow
  runs of EBPL8501) are left out. This is DSSAT's soil-albedo input to the method, reconstructed
  here; it is not part of the PET kernel.

Reference-side limit (the tolerance). DSSAT prints ``EOAA`` with 3 decimals (``F8.3``), ``LAID``
with 2 (``F7.2``), ``SW1D``, ``DUL1`` and ``MCFD`` with 3; the weather and ``SALB`` are exact (1-
and 2-decimal file values, echoed). ``EO`` is monotone in the albedo, the albedo
``0.23 - (0.23 - MSALB) exp(-0.75 LAI)`` is monotone in ``LAI`` and in ``MSALB``, and ``MSALB`` is
monotone in each of ``SW1``, ``DUL(1)`` and ``MULCHCOVER``, so over the print box (``LAI +- 0.005``
clipped at 0, the others ``+- 0.0005``, the cover clipped to [0, 1]) the kernel's extremes are at
the 16 corners. A day passes when ``EOAA`` lies within ``[min corner - 0.0005, max corner +
0.0005]``, widened by 1e-6 relative for DSSAT's single precision (a dozen REAL operations of 2^-24
each). No other slack.

Per experiment the table reports the day count, max |ours - EOAA| and RMSE with the printed inputs
at their centre values, the count of days beyond the pure ``EOAA`` print step (0.0005 mm), the
largest excess over the bound (<= 0: every day passes) and the largest bound half-width.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
WHEAT = DSSAT_ENGINE / "example_data" / "Wheat"
#: winter-wheat examples: the dump check of PETPT's cold branch (TMAX < 5 degC), which no Maize
#: example reaches
WHEAT_EXPERIMENTS = ("KSAS8101", "RORO7401", "SWSW7501")
WEATHER = DSSAT_ENGINE / "example_data" / "Weather"
SOIL = DSSAT_ENGINE / "example_data" / "Soil"
PEST = DSSAT_ENGINE / "source" / "Data" / "Pest"
OUT_SUBDIR = Path("validation") / "h1_2_pt_dssat"

#: print half-steps of the DSSAT outputs used: ET.OUT EOAA F8.3, PlantGro LAID F7.2, SoilWat SW1D
#: F8.3, SoilDyn DUL1 F8.3, Mulch MCFD F8.3
EO_HALF = 0.0005
LAI_HALF = 0.005
SW_HALF = 0.0005
DUL_HALF = 0.0005
COVER_HALF = 0.0005
#: ET.OUT prints SRAA / TMAXA / TMINA with 2 decimals
WEATHER_HALF = 0.005
#: DSSAT computes EO in REAL (float32): a dozen operations of 2^-24 relative rounding each
REAL_REL = 1.0e-6

# DSSAT-CSM v4.8.6.0 (BSD-3) Soil/SoilUtilities/SOILDYN.for, ALBEDO_avg, lines 1528-1530 and 1562:
#   FF = (SW1 - 0.03) / (SOILPROP % DUL(1) - 0.03)
#   FF = MAX(0.0, MIN(2.0, FF))
#   SWALB = SOILPROP % SALB * (1.0 - 0.45 * FF)
#   MSALB = MULCHCOVER * MULCHALB + (1.0 - MULCHCOVER) * SWALB
# and Soil/Mulch/MULCHLAYER.for line 47:  MULCHALB = 0.45
ALB_SW_OFFSET = 0.03
ALB_FF_MAX = 2.0
ALB_SW_SLOPE = 0.45
MULCH_ALBEDO = 0.45


def dssat_msalb(salb: np.ndarray | float, sw1: np.ndarray, dul1: np.ndarray, cover: np.ndarray) -> np.ndarray:
    """``MSALB`` of ``ALBEDO_avg`` (soil albedo with the top-layer water and mulch effects)."""
    ff = (np.asarray(sw1, dtype=float) - ALB_SW_OFFSET) / (np.asarray(dul1, dtype=float) - ALB_SW_OFFSET)
    ff = np.clip(ff, 0.0, ALB_FF_MAX)
    swalb = np.asarray(salb, dtype=float) * (1.0 - ALB_SW_SLOPE * ff)
    c = np.asarray(cover, dtype=float)
    return c * MULCH_ALBEDO + (1.0 - c) * swalb


def experiments() -> list[str]:
    return sorted(p.stem for p in MAIZE.glob("*.MZX")) if MAIZE.is_dir() else []


# --------------------------------------------------------------------------- running


#: FileX edits of every run: (header of the row, column, value)
FILEX_EDITS = (("@N METHODS", "EVAPO", "R"), ("@N OUTPUTS", "WAOUT", "Y"), ("@N OUTPUTS", "VBOSE", "D"))


def patch_filex(text: str, edits: Sequence[tuple[str, str, str]] = FILEX_EDITS) -> tuple[str, dict[str, int]]:
    """FileX text with ``column = value`` on every data row under a header line starting with
    ``header`` (single-character switches, right-aligned under the column name), and per column
    the number of rows that changed."""
    lines = text.splitlines(keepends=True)
    changed = {c: 0 for _, c, _ in edits}
    active: list[tuple[int, str, str]] = []
    for i, ln in enumerate(lines):
        if ln.startswith(("@", "*")):
            active = [
                (ln.index(c) + len(c) - 1, c, v)
                for h, c, v in edits
                if ln.startswith(h) and re.search(rf"\b{c}\b", ln)
            ]
            continue
        if not active or not ln.strip() or ln.lstrip().startswith("!"):
            continue
        for col, c, v in active:
            if len(ln) <= col or ln[col] == " " or (col + 1 < len(ln) and not ln[col + 1].isspace()):
                raise ValueError(f"no single-character {c} value at column {col + 1}: {ln!r}")
            if ln[col] != v:
                changed[c] += 1
                ln = ln[:col] + v + ln[col + 1 :]
        lines[i] = ln
    return "".join(lines), changed


def crop_of(name: str) -> tuple[Path, str, str]:
    """(example directory, FileX extension, model) of experiment ``name`` (Maize or Wheat)."""
    if (MAIZE / f"{name}.MZX").is_file():
        return MAIZE, ".MZX", "MZCER048"
    return WHEAT, ".WHX", "CSCER048"


def run_experiment(name: str, work: Path, engine: Path | None = None) -> Path:
    """Run every treatment of experiment ``name`` with the :data:`FILEX_EDITS`; return the
    output directory. ``engine`` (default: the v4.8.6.0 engine) may be the instrumented build."""
    from agrijax.io.dssat import weather_stations
    from agrijax.port.run_fortran import run_dscsm

    crop_dir, ext, model = crop_of(name)
    exp = work / "exp"
    if work.exists():
        shutil.rmtree(work)
    exp.mkdir(parents=True)
    for f in crop_dir.glob(name + ".*"):
        shutil.copy2(f, exp / f.name)
    fx = exp / f"{name}{ext}"
    text, _ = patch_filex(fx.read_text(errors="replace"))
    fx.write_text(text)
    stations = {w[:4].upper() for w in weather_stations(fx)}
    extra = [f for f in sorted(WEATHER.glob("*.WTH")) if f.name[:4].upper() in stations]
    extra += sorted(PEST.glob("*.PST"))  # pest-damage examples (IUAF9902/9903)
    r = run_dscsm(
        exp,
        work / "out",
        model=model,
        run_mode="A",
        experiment_file=fx.name,
        engine=engine if engine is not None else DSSAT_ENGINE,
        weather_dir=WEATHER,
        soil_dir=SOIL,
        extra_files=extra,
    )
    return r.out_dir


# --------------------------------------------------------------------------- reading

_OV_RUN = re.compile(r"^\*RUN\s+(\d+)")
_OV_SALB = re.compile(r"^SOIL ALBEDO\s*:\s*([-\d.]+)")
_OV_LAYER = re.compile(r"^\s*(\d+)-\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
_OV_ET = re.compile(r"\bET\s*:\s*(\S)")
_OV_INFIL = re.compile(r"\bINFIL\s*:\s*(\S)")


@dataclass
class RunSoil:
    salb: float
    ll1: float
    dul1: float
    top_cm: float
    meevp: str
    meinf: str


def read_overview(path: Path) -> dict[int, RunSoil]:
    """Per run: ``SOIL ALBEDO``, the first soil layer's LL / DUL and the ET / INFIL options."""
    out: dict[int, RunSoil] = {}
    run: int | None = None
    cur: dict[str, object] = {}
    for ln in path.read_text(errors="replace").splitlines():
        m = _OV_RUN.match(ln)
        if m:
            run = int(m.group(1))
            cur = {}
            continue
        if run is None:
            continue
        if (m := _OV_ET.search(ln)) and "SIMULATION OPT" not in ln and "PHOTO" in ln:
            cur["meevp"] = m.group(1)
        if (m := _OV_INFIL.search(ln)) and "PHOTO" in ln:
            cur["meinf"] = m.group(1)
        if "dul1" not in cur and (m := _OV_LAYER.match(ln)) and int(m.group(1)) == 0:
            cur.update(top_cm=float(m.group(2)), ll1=float(m.group(3)), dul1=float(m.group(4)))
        if (m := _OV_SALB.match(ln)) and run not in out:
            out[run] = RunSoil(
                salb=float(m.group(1)),
                ll1=float(cur["ll1"]),  # type: ignore[arg-type]
                dul1=float(cur["dul1"]),  # type: ignore[arg-type]
                top_cm=float(cur["top_cm"]),  # type: ignore[arg-type]
                meevp=str(cur.get("meevp", "?")),
                meinf=str(cur.get("meinf", "?")),
            )
    return out


def station_weather(name: str) -> pd.DataFrame:
    """Daily weather of the experiment's stations as DSSAT reads it, indexed by date."""
    from agrijax.io.dssat import read_wth, weather_stations

    stations = {w[:4].upper() for w in weather_stations(MAIZE / f"{name}.MZX")}
    files = [f for f in sorted(WEATHER.glob("*.WTH")) if f.name[:4].upper() in stations]
    wx = pd.concat([read_wth(f, dssat_spans=True) for f in files]).drop_duplicates("date")
    wx["date"] = pd.to_datetime(wx["date"])
    return wx.set_index("date")


def env_modified_treatments(name: str) -> set[int]:
    from agrijax.io.dssat import read_filex

    return {int(t["N"]) for t in read_filex(MAIZE / f"{name}.MZX")["TREATMENTS"] if t.get("ME")}


@dataclass
class RunDays:
    """The inputs PETPT received and the EO it printed, one row per simulated day of one run."""

    run: int
    trno: int
    days: pd.DataFrame  # date, srad, tmax, tmin, lai, sw1, dul1, cover, eo_ref, sraa, tmaxa, tmina
    soil: RunSoil


@dataclass
class Skipped:
    env_modified: list[int]
    no_overview: list[int]


def _series(df: pd.DataFrame, run: int, col: str) -> pd.Series:
    if not len(df) or col not in df.columns:
        return pd.Series(dtype=float)
    return df[df["RUN"] == run].sort_values("DATE").drop_duplicates("DATE").set_index("DATE")[col]


def _read(out: Path, name: str) -> pd.DataFrame:
    from agrijax.io.dssat import read_out

    return read_out(out / name) if (out / name).is_file() else pd.DataFrame()


def run_days(name: str, out: Path) -> tuple[list[RunDays], Skipped]:
    et = _read(out, "ET.OUT")
    sw, pg, mu, sd = (_read(out, f) for f in ("SoilWat.OUT", "PlantGro.OUT", "Mulch.OUT", "SoilDyn.OUT"))
    ov = read_overview(out / "OVERVIEW.OUT")
    wx = station_weather(name)
    env = env_modified_treatments(name)
    res: list[RunDays] = []
    skipped = Skipped([], [])
    for run, e in et.groupby("RUN"):
        run = int(run)
        trno = int(e["TRNO"].iloc[0])
        if trno in env:
            skipped.env_modified.append(run)
            continue
        if run not in ov:
            skipped.no_overview.append(run)
            continue
        e = e.sort_values("DATE").set_index("DATE")
        prev = e.index - pd.Timedelta(days=1)

        def previous(
            df: pd.DataFrame, col: str, what: str, run: int = run, prev: pd.DatetimeIndex = prev
        ) -> np.ndarray:
            v = _series(df, run, col).reindex(prev).to_numpy(dtype=float)
            if np.isnan(v).any():
                raise ValueError(f"{name} run {run}: {what} misses previous-day rows")
            return v

        sw1 = previous(sw, "SW1D", "SoilWat.OUT")
        cover = previous(mu, "MCFD", "Mulch.OUT")
        dul = _series(sd, run, "DUL1").reindex(prev).to_numpy(dtype=float, copy=True)
        if np.isnan(dul[0]):
            dul[0] = ov[run].dul1  # SoilDyn.OUT starts after day 0: the initial DUL(1)
        if np.isnan(dul).any():
            raise ValueError(f"{name} run {run}: SoilDyn.OUT misses previous-day rows")
        p = _series(pg, run, "LAID")
        lai = p.reindex(prev).to_numpy(dtype=float)
        # no PlantGro row: before the first row (not planted) or after the last (harvested), XHLAI = 0;
        # a missing row inside the season is an error
        missing = np.isnan(lai)
        if len(p) and ((prev[missing] >= p.index.min()) & (prev[missing] <= p.index.max())).any():
            raise ValueError(f"{name} run {run}: PlantGro.OUT has a gap inside the season")
        lai = np.where(missing, 0.0, lai)
        w = wx.reindex(e.index)
        d = pd.DataFrame(
            {
                "date": e.index,
                "srad": w["srad"].to_numpy(dtype=float),
                "tmax": w["tmax"].to_numpy(dtype=float),
                "tmin": w["tmin"].to_numpy(dtype=float),
                "lai": lai,
                "sw1": sw1,
                "dul1": dul,
                "cover": cover,
                "eo_ref": e["EOAA"].to_numpy(dtype=float),
                "sraa": e["SRAA"].to_numpy(dtype=float),
                "tmaxa": e["TMAXA"].to_numpy(dtype=float),
                "tmina": e["TMINA"].to_numpy(dtype=float),
            }
        )
        res.append(RunDays(run, trno, d, ov[run]))
    return res, skipped


# --------------------------------------------------------------------------- comparing


def _pt(
    srad: np.ndarray, tmax: np.ndarray, tmin: np.ndarray, lai: np.ndarray, msalb: np.ndarray
) -> np.ndarray:
    from agrijax.processes.pet import priestley_taylor

    return np.asarray(priestley_taylor(srad, tmax, tmin, lai, msalb), dtype=float)


def compare(r: RunDays) -> pd.DataFrame:
    """Day table with our EO at the printed inputs and its print-box bounds."""
    d = r.days.copy()
    sa = r.soil.salb
    args = (d["srad"].to_numpy(), d["tmax"].to_numpy(), d["tmin"].to_numpy())
    lai, sw1, dul, cov = (d[c].to_numpy() for c in ("lai", "sw1", "dul1", "cover"))
    d["msalb"] = dssat_msalb(sa, sw1, dul, cov)
    d["eo"] = _pt(*args, lai, d["msalb"].to_numpy())
    corners = [
        _pt(
            *args,
            np.maximum(lai + dl, 0.0),
            dssat_msalb(sa, sw1 + ds, dul + du, np.clip(cov + dc, 0.0, 1.0)),
        )
        for dl in (-LAI_HALF, LAI_HALF)
        for ds in (-SW_HALF, SW_HALF)
        for du in (-DUL_HALF, DUL_HALF)
        for dc in (-COVER_HALF, COVER_HALF)
    ]
    lo = np.min(corners, axis=0)
    hi = np.max(corners, axis=0)
    d["lo"] = lo - EO_HALF - REAL_REL * np.abs(lo)
    d["hi"] = hi + EO_HALF + REAL_REL * np.abs(hi)
    d["diff"] = d["eo"] - d["eo_ref"]
    d["excess"] = np.maximum(d["lo"] - d["eo_ref"], d["eo_ref"] - d["hi"])
    d["run"] = r.run
    d["trno"] = r.trno
    return d


def summarise(name: str, days: pd.DataFrame, n_runs: int) -> dict[str, object]:
    diff = days["diff"].to_numpy()
    k = int(np.argmax(np.abs(diff)))
    return {
        "experiment": name,
        "runs": n_runs,
        "days": len(days),
        "max_abs_mm": float(np.abs(diff).max()),
        "rmse_mm": float(np.sqrt(np.mean(diff**2))),
        "worst_run": int(days["run"].iloc[k]),
        "worst_date": str(pd.Timestamp(days["date"].iloc[k]).date()),
        "days_beyond_eo_print": int((np.abs(diff) > EO_HALF + 1e-12).sum()),
        "max_excess_mm": float(days["excess"].max()),
        "max_halfwidth_mm": float(((days["hi"] - days["lo"]) / 2).max()),
        "days_lai0": int((days["lai"] == 0).sum()),
        "max_abs_lai0_mm": float(np.abs(diff[days["lai"].to_numpy() == 0]).max(initial=0.0)),
        "max_cover": float(days["cover"].max()),
        "dul1_range": f"{days['dul1'].min():.3f}-{days['dul1'].max():.3f}",
        "days_hot": int((days["tmax"] > 35).sum()),
        "days_cold": int((days["tmax"] < 5).sum()),
        "eo_ref_max_mm": float(days["eo_ref"].max()),
    }


def analyse(name: str, out: Path) -> tuple[pd.DataFrame, dict[str, object], list[RunDays]]:
    runs, skipped = run_days(name, out)
    if not runs:
        row: dict[str, object] = {"experiment": name, "runs": 0}
    else:
        days = pd.concat([compare(r) for r in runs], ignore_index=True)
        row = summarise(name, days, len(runs))
    row["skipped_env_modified"] = len(skipped.env_modified)
    row["skipped_no_overview"] = len(skipped.no_overview)
    return (days if runs else pd.DataFrame()), row, runs


# --------------------------------------------------------------------------- PETPT dumps
#
# A private instrumented build of the same v4.8.6.0 source (``<data>/dssat_pt_dump/engine``: one
# WRITE after the ``CALL PETPT`` of ``SPAM/PET.for``, ``ES16.9``, i.e. every REAL exactly) prints
# the inputs PETPT received and the EO it returned, every day of every run (``PTDUMP.OUT``:
# RUN, YRDOY, ET_ALB, SRAD, TMAX, TMIN, XHLAI, EO). Built on rorqual with gfortran 12 and the
# project's CMake flags (the shipped binary: gfortran 13, locally); its printed outputs are
# compared with the shipped binary's, day by day.

DUMP_ENGINE_SUBDIR = Path("dssat_pt_dump") / "engine"
DUMP_COLUMNS = ("run", "yrdoy", "et_alb", "srad", "tmax", "tmin", "xhlai", "eo")
#: ES16.9 prints 10 significant digits: the float32 value exactly (9 suffice)
DUMP_DIGITS = 10


def read_ptdump(path: Path) -> pd.DataFrame:
    a = np.loadtxt(path, ndmin=2)
    d = pd.DataFrame(a, columns=list(DUMP_COLUMNS))
    d["run"] = d["run"].astype(int)
    d["yrdoy"] = d["yrdoy"].astype(int)
    d["date"] = pd.to_datetime(d["yrdoy"].astype(str), format="%Y%j")
    return d


def kernel_on_dump(dump: pd.DataFrame) -> dict[str, float]:
    """Our kernel on the dumped inputs against the dumped EO: max absolute and relative error."""
    eo = _pt(*(dump[c].to_numpy() for c in ("srad", "tmax", "tmin", "xhlai", "et_alb")))
    ref = dump["eo"].to_numpy()
    err = np.abs(eo - ref)
    return {
        "days": float(len(ref)),
        "max_abs_mm": float(err.max()),
        "max_rel": float((err / np.maximum(np.abs(ref), 1e-30)).max()),
        "eo_min_mm": float(ref.min()),
        "eo_max_mm": float(ref.max()),
        "days_hot": float((dump["tmax"] > 35).sum()),
        "days_cold": float((dump["tmax"] < 5).sum()),
        "days_floor": float((ref <= 1.0001e-4).sum()),
    }


def eoaa_vs_dump(out: Path, dump: pd.DataFrame) -> float:
    """Largest |EOAA - EO| over the days of ``ET.OUT`` (the print rounding of the dumped EO)."""
    et = _read(out, "ET.OUT")
    m = et.merge(dump, left_on=["RUN", "DATE"], right_on=["run", "date"], validate="one_to_one")
    if len(m) != len(et):
        raise ValueError(f"{len(et) - len(m)} ET.OUT days have no PETPT dump row")
    return float(np.abs(m["EOAA"] - m["eo"]).max())


def masked_lines(path: Path) -> list[str]:
    stamp = re.compile(r"[A-Z]{3} \d{1,2}, \d{4}[; ]+\d\d:\d\d:\d\d")
    return [stamp.sub("<stamp>", ln) for ln in path.read_text(errors="replace").splitlines()]


def reconstruction_vs_dump(days: pd.DataFrame, dump: pd.DataFrame) -> dict[str, float]:
    """The harness inputs (printed outputs of the shipped binary) against the dumped ones."""
    m = days.merge(dump, on=["run", "date"], suffixes=("", "_dump"), validate="one_to_one")
    f32 = lambda x: np.asarray(x, dtype=np.float32).astype(float)  # noqa: E731
    return {
        "days_matched": float(len(m)),
        "days_harness": float(len(days)),
        "lai_max_abs": float(np.abs(m["lai"] - m["xhlai"]).max()),
        "msalb_max_abs": float(np.abs(m["msalb"] - m["et_alb"]).max()),
        "weather_max_abs": float(
            max(np.abs(f32(m[c]) - m[f"{c}_dump"]).max() for c in ("srad", "tmax", "tmin"))
        ),
        "eoaa_vs_dump_max_abs": float(np.abs(m["eo_ref"] - m["eo_dump"]).max()),
        "ours_vs_dump_max_abs": float(np.abs(m["eo"] - m["eo_dump"]).max()),
    }


KEPT_OUTPUTS = (
    "ET.OUT",
    "SoilWat.OUT",
    "PlantGro.OUT",
    "OVERVIEW.OUT",
    "Summary.OUT",
    "Mulch.OUT",
    "SoilDyn.OUT",
)
#: outputs compared line by line between the shipped and the instrumented build
SAME_OUTPUTS = ("ET.OUT", "SoilWat.OUT", "PlantGro.OUT", "Mulch.OUT", "SoilDyn.OUT", "Summary.OUT")


def builds_differ(out: Path, out_dump: Path) -> dict[str, int]:
    """Per output file: number of lines (timestamps masked) that differ between the two builds."""
    res: dict[str, int] = {}
    for f in SAME_OUTPUTS:
        if not (out / f).is_file():
            continue
        a, b = masked_lines(out / f), masked_lines(out_dump / f)
        res[f] = abs(len(a) - len(b)) + sum(x != y for x, y in zip(a, b, strict=False))
    return res


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    import jax

    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
    ap.add_argument("--work", default=None, help="run root (default $AGRI_JAX_RUN_ROOT/pt_dssat)")
    ap.add_argument("--experiments", nargs="*", default=experiments())
    ap.add_argument("--wheat", nargs="*", default=list(WHEAT_EXPERIMENTS))
    a = ap.parse_args(argv)
    out_dir = Path(a.data_dir) / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(a.work) if a.work else Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "pt_dssat"
    dump_engine = Path(a.data_dir) / DUMP_ENGINE_SUBDIR
    have_dump = (dump_engine / "bin" / "dscsm048").is_file()
    rows: list[dict[str, object]] = []
    dump_rows: list[dict[str, object]] = []
    for name in [*a.experiments, *(a.wheat if have_dump else [])]:
        if name in a.wheat:  # dump-level kernel check only
            out = run_experiment(name, work / name)
            out_d = run_experiment(name, work / f"{name}_dump", engine=dump_engine)
            (out_dir / "outputs" / name).mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_d / "PTDUMP.OUT", out_dir / "outputs" / name / "PTDUMP.OUT")
            dump = read_ptdump(out_d / "PTDUMP.OUT")
            k = kernel_on_dump(dump)
            eo_print = eoaa_vs_dump(out, dump)
            diff = builds_differ(out, out_d)
            dump_rows.append(
                {"experiment": name, **k, "eoaa_vs_dump_max_abs": eo_print, "build_lines_differ": str(diff)}
            )
            print(
                f"{name}: DUMP days {int(k['days'])} kernel max {k['max_abs_mm']:.2e} mm rel "
                f"{k['max_rel']:.2e} (EO {k['eo_min_mm']:.4f}..{k['eo_max_mm']:.3f}, hot {int(k['days_hot'])} "
                f"cold {int(k['days_cold'])} floor {int(k['days_floor'])}) | EOAA-EOdump {eo_print:.2e} | "
                f"builds differ {diff}",
                flush=True,
            )
            continue
        out = run_experiment(name, work / name)
        keep = out_dir / "outputs" / name
        keep.mkdir(parents=True, exist_ok=True)
        for f in KEPT_OUTPUTS:
            if (out / f).is_file():
                shutil.copy2(out / f, keep / f)
        days, row, runs = analyse(name, out)
        if have_dump:
            out_d = run_experiment(name, work / f"{name}_dump", engine=dump_engine)
            shutil.copy2(out_d / "PTDUMP.OUT", keep / "PTDUMP.OUT")
            dump = read_ptdump(out_d / "PTDUMP.OUT")
            k = kernel_on_dump(dump)
            diff = builds_differ(out, out_d)
            drow: dict[str, object] = {"experiment": name, **k}
            drow["build_lines_differ"] = ";".join(f"{f}={n}" for f, n in diff.items())
            if runs:
                drow.update(reconstruction_vs_dump(days, dump))
            dump_rows.append(drow)
            print(
                f"{name}: DUMP days {int(k['days'])} kernel max {k['max_abs_mm']:.2e} mm rel "
                f"{k['max_rel']:.2e} (EO {k['eo_min_mm']:.4f}..{k['eo_max_mm']:.3f}, hot {int(k['days_hot'])} "
                f"cold {int(k['days_cold'])} floor {int(k['days_floor'])}) | builds differ {drow['build_lines_differ']}"
                + (
                    f" | harness: LAI {drow['lai_max_abs']:.2e} MSALB {drow['msalb_max_abs']:.2e} weather "
                    f"{drow['weather_max_abs']:.1e} EOAA-EOdump {drow['eoaa_vs_dump_max_abs']:.2e} "
                    f"ours-EOdump {drow['ours_vs_dump_max_abs']:.2e} ({int(drow['days_matched'])}/"
                    f"{int(drow['days_harness'])} days)"
                    if runs
                    else ""
                ),
                flush=True,
            )
        if not runs:
            print(f"{name}: no comparable run ({row})", flush=True)
            rows.append(row)
            continue
        days.to_csv(out_dir / f"{name}_daily.csv", index=False)
        soils = {(r.soil.salb, r.soil.dul1, r.soil.top_cm, r.soil.meevp, r.soil.meinf) for r in runs}
        row["soils"] = ";".join(f"SALB={s[0]} DUL1={s[1]} top={s[2]}cm ET={s[3]} INFIL={s[4]}" for s in soils)
        wmax = float(
            np.nanmax(
                np.abs(
                    days[["srad", "tmax", "tmin"]].to_numpy() - days[["sraa", "tmaxa", "tmina"]].to_numpy()
                )
            )
        )
        row["weather_max_abs"] = wmax
        rows.append(row)
        print(
            f"{name}: runs {row['runs']} days {row['days']} max {row['max_abs_mm']:.4f} rmse "
            f"{row['rmse_mm']:.5f} beyond-print {row['days_beyond_eo_print']} excess "
            f"{row['max_excess_mm']:.2e} halfwidth {row['max_halfwidth_mm']:.4f} | LAI=0 days "
            f"{row['days_lai0']} max {row['max_abs_lai0_mm']:.4f} | cover<={row['max_cover']:.3f} DUL1 "
            f"{row['dul1_range']} | weather {wmax:.3g} | skipped env {row['skipped_env_modified']} "
            f"no-overview {row['skipped_no_overview']} | {row['soils']}",
            flush=True,
        )
    for fname, rs in (("pt_dssat_summary.csv", rows), ("pt_dssat_dump_summary.csv", dump_rows)):
        if not rs:
            continue
        cols = list(dict.fromkeys(k for r in rs for k in r))
        with open(out_dir / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rs)
        print(f"wrote {out_dir / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
