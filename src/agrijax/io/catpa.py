"""CA-TPA site data: management events, the 100k-run LHS batch, and per-node profile output.

All data lives outside git under ``$AGRI_JAX_DATA`` (default ``~/agri_jax_data``;
layout in ``scripts/data/README.md``). Nothing here imports JAX.

Management events (:func:`build_events`, :func:`load_events`)
-------------------------------------------------------------
The only management source of the CA-TPA scenario is ``rzwqm.dat`` (``MZDSSAT.RZX`` carries
DSSAT cultivar/database settings, ``expdata.dat`` carries observations). Its blocks are
resolved by :func:`agrijax.io.rzwqm.events.read_management` the way ``Rzman.for`` schedules
them. CA-TPA lists 20 plantings 2002-2021, so the 2015-2023 window holds 7 seasons (2015-2021);
2022-2023 are fallow. Its 20 identical fertilizer records give one 180 kg N/ha NO3 broadcast
per season (first matching record of a day fires), its single pesticide record gives one
glyphosate application the day before each planting, and its tillage records are fixed dates.

The result is cross-checked against ``MANAGE.OUT`` of the 2015-2023 base run
(:func:`read_manage_out`; ``tests/unit/test_catpa_loader.py``).

LHS batch (:func:`load_catpa_lhs`, :func:`load_catpa_params`)
------------------------------------------------------------
``catpa_lhs/catpa_daily.npz`` (float32, ``(n_run, n_day)``, run-sorted) was extracted from the
``.ana`` files of the 100 000 runs by cluster extraction scripts (not part of this repository), which skip the
24 header lines, i.e. also the ``2015.000`` initial-state row: day 0 is 2015-01-01, day 3286 is 2023-12-31.
Mapping (:data:`LHS_ANA_COLUMNS`, 1-based ``.ana`` columns):

===============  ======  ==============================================
array            column  ``.ana`` header
===============  ======  ==============================================
``sw_cm``        2       STORED SOIL WATER (CM), 0-150 cm profile
``evap_cm``      6       ACTUAL EVAPORATION (CM/DAY)
``transp_cm``    7       ACTUAL TRANSPIRATION (CM/DAY)
``lai``          43      LEAF AREA INDEX
``grain_kg_ha``  44      BIOMASS OF GRAIN (KG/HA)
``aet_cm``       84      ACTUAL ET (CM), daily
===============  ======  ==============================================

``catpa_yield.npz`` holds the ``Maize YIELD : N kg/ha`` lines of each run's ``OVERVIEW.OUT``
(one per season, 2015..2021). Run ``i`` used row ``i`` (0-based) of
``AutoAnalysis/parameter.csv`` (``GenerateDat.py`` writes ``{i}_rzwqm.dat`` from row ``i``).

The six daily arrays are 1.3 GB each (7.9 GB together), more than a laptop's RAM, so by default
they are extracted once to ``catpa_lhs/npy/*.npy`` and returned as read-only memory maps.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rzwqm.dat import RzwqmDat
from .rzwqm.events import (
    EVENT_COLUMNS,
    FERT_METHOD,
    HARVEST_TYPE,
    PEST_METHOD,
    TILL_IMPLEMENT,
    TILL_OPERATION,
    _d,
    read_management,
)
from .rzwqm.layers import layer_thickness_cm, profile_storage_cm, read_layer_output

__all__ = [
    "CATPA_DIR",
    "DATA_ROOT",
    "EVENT_COLUMNS",
    "FERT_METHOD",
    "HARVEST_TYPE",
    "LHS_ANA_COLUMNS",
    "LHS_DIR",
    "N_DAY",
    "N_RUN",
    "N_SEASON",
    "PARAMETER_CSV",
    "PEST_METHOD",
    "SCENARIO_DIR",
    "TILL_IMPLEMENT",
    "TILL_OPERATION",
    "build_events",
    "extract_lhs_npy",
    "layer_thickness_cm",
    "load_catpa_lhs",
    "load_catpa_params",
    "load_events",
    "profile_storage_cm",
    "read_layer_output",
    "read_manage_out",
    "write_events",
]

DATA_ROOT = Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data"))).expanduser()
SCENARIO_DIR = DATA_ROOT / "narval_mirror" / "RZWQM_sw_batch" / "CA-TPA" / "Scenario"
PARAMETER_CSV = DATA_ROOT / "narval_mirror" / "RZWQM_sw_batch" / "CA-TPA" / "AutoAnalysis" / "parameter.csv"
CATPA_DIR = DATA_ROOT / "catpa"
LHS_DIR = DATA_ROOT / "catpa_lhs"

N_RUN = 100_000
N_DAY = 3287  # 2015-01-01 .. 2023-12-31
N_SEASON = 7  # maize 2015..2021
LHS_START = np.datetime64("2015-01-01", "D")

LHS_ANA_COLUMNS: dict[str, int] = {
    "sw_cm": 2,
    "evap_cm": 6,
    "transp_cm": 7,
    "lai": 43,
    "grain_kg_ha": 44,
    "aet_cm": 84,
}
"""Array name in ``catpa_daily.npz`` -> 1-based ``.ana`` column."""


# ============================================================================ events
def build_events(
    dat: RzwqmDat | str | Path | None = None,
    start: str | np.datetime64 = "2015-01-01",
    end: str | np.datetime64 = "2023-12-31",
) -> pd.DataFrame:
    """Management event table of the CA-TPA ``rzwqm.dat`` within ``[start, end]``.

    :func:`agrijax.io.rzwqm.events.read_management` with the CA-TPA scenario and the 2015-2023
    window as defaults (columns :data:`EVENT_COLUMNS`; module docstring).
    """
    if dat is None:
        dat = SCENARIO_DIR / "rzwqm.dat"
    return read_management(dat, start, end)


def write_events(df: pd.DataFrame, path: str | Path | None = None) -> Path:
    """Write the table as CSV (dates ISO, ``value`` with full precision)."""
    p = Path(path) if path is not None else CATPA_DIR / "events.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(p, index=False)
    return p


def load_events(path: str | Path | None = None) -> pd.DataFrame:
    """Read ``catpa/events.csv`` (default under ``$AGRI_JAX_DATA``); ``date`` parsed as datetime64."""
    p = Path(path) if path is not None else CATPA_DIR / "events.csv"
    df = pd.read_csv(
        p,
        dtype={"event": str, "unit": str, "source_line": str, "detail": str},
        keep_default_na=False,
        na_values={"value": [""]},
    )
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: missing columns {missing}")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return pd.DataFrame(df.loc[:, list(EVENT_COLUMNS)])


# ---------------------------------------------------------------- MANAGE.OUT
_MDATE = re.compile(r"^-{3,}\s*(\d{1,2})/\s*(\d{1,2})/(\d{4})\s*-+\s*(\d+)\s*-+")
_HDATE = re.compile(r"^\s*ON\s+(\d{1,2})/\s*(\d{1,2})/(\d{4})")
_AMT_FERT = re.compile(r"AMOUNT OF (NH4|NO3|UREA)\s*\[KG/HA\]\s*(-?[\d.]+(?:E[-+]?\d+)?)", re.I)
_AMT_PEST = re.compile(r"AMOUNT OF (.*?)\s*(-?\d+\.\d+(?:E[-+]?\d+)?)\s*\[KG/HA\]", re.I)
_DENS = re.compile(r"PLANTING DENSITY:\s*([\d.]+)")
_IMPL = re.compile(r"WITH IMPLEMENT:\s*(.*?)\s*$")


def read_manage_out(path: str | Path) -> pd.DataFrame:
    """Events reported in RZWQM2's ``MANAGE.OUT`` -> DataFrame(date, event, value, detail).

    ``event`` uses the names of :func:`build_events` (fertilizer split per N form, every form
    reported, including zeros). Harvests are the ``DSSAT Crop Harvest ... ON dd/mm/yyyy`` blocks.
    """
    text = Path(path).read_bytes().decode("latin-1", errors="ignore").replace("\x00", "")
    rows: list[dict[str, Any]] = []
    date: np.datetime64 | None = None
    cur: str | None = None
    harvest_pending = False
    for ln in text.splitlines():
        if "DSSAT Crop Harvest" in ln:
            harvest_pending, cur = True, None
            continue
        if harvest_pending and (m := _HDATE.match(ln)):
            rows.append({"date": _d(*m.group(1, 2, 3)), "event": "harvest", "value": np.nan, "detail": ""})
            harvest_pending = False
            continue
        if m := _MDATE.match(ln):
            date, cur = _d(*m.group(1, 2, 3)), None
            continue
        if "EVENT ==>" in ln:
            s = ln.upper()
            cur = next(
                (
                    e
                    for k, e in (
                        ("TILLAGE", "tillage"),
                        ("PESTICIDE", "pesticide"),
                        ("FERTILIZ", "fertilizer"),
                        ("PLANTING", "planting"),
                    )
                    if k in s
                ),
                "other",
            )
            if cur == "other":
                rows.append(
                    {"date": date, "event": "other", "value": np.nan, "detail": ln.split("==>", 1)[1].strip()}
                )
            continue
        if date is None or cur is None:
            continue
        if cur == "fertilizer" and (m := _AMT_FERT.search(ln)):
            rows.append(
                {
                    "date": date,
                    "event": f"fertilizer_{m.group(1).lower()}",
                    "value": float(m.group(2)),
                    "detail": "",
                }
            )
        elif cur == "pesticide" and (m := _AMT_PEST.search(ln)):
            rows.append(
                {"date": date, "event": "pesticide", "value": float(m.group(2)), "detail": m.group(1).strip()}
            )
        elif cur == "planting" and (m := _DENS.search(ln)):
            rows.append({"date": date, "event": "planting", "value": float(m.group(1)), "detail": ""})
        elif cur == "tillage" and (m := _IMPL.search(ln)):
            rows.append({"date": date, "event": "tillage", "value": np.nan, "detail": m.group(1).lower()})
    df = pd.DataFrame(rows, columns=["date", "event", "value", "detail"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ============================================================================ LHS batch
def _lhs_dir(data_dir: str | Path | None) -> Path:
    d = Path(data_dir) if data_dir is not None else LHS_DIR
    if (d / "catpa_lhs" / "catpa_daily.npz").is_file():  # tolerate the data root
        d = d / "catpa_lhs"
    return d


def extract_lhs_npy(data_dir: str | Path | None = None, names: Sequence[str] | None = None) -> Path:
    """Extract members of ``catpa_daily.npz`` to ``<data_dir>/npy/<name>.npy`` (once; streamed)."""
    d = _lhs_dir(data_dir)
    out = d / "npy"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(d / "catpa_daily.npz") as zf:
        members = {Path(i.filename).stem: i for i in zf.infolist()}
        for nm in names if names is not None else list(members):
            info = members[nm]
            dst = out / f"{nm}.npy"
            if dst.is_file() and dst.stat().st_size == info.file_size:
                continue
            tmp = dst.with_suffix(".npy.part")
            with zf.open(info) as src, open(tmp, "wb") as fh:
                shutil.copyfileobj(src, fh, length=16 << 20)
            tmp.replace(dst)
    return out


def _yyyyddd_to_dates(t: np.ndarray) -> np.ndarray:
    t64 = np.asarray(t, dtype=np.float64)
    year = np.floor(t64 + 1e-4).astype(np.int64)
    doy = np.rint((t64 - year) * 1000.0).astype(np.int64)
    jan1 = (year - 1970).astype("datetime64[Y]").astype("datetime64[D]")
    return jan1 + (doy - 1).astype("timedelta64[D]")


def load_catpa_lhs(
    data_dir: str | Path | None = None,
    *,
    variables: Sequence[str] | None = None,
    runs: slice | Sequence[int] | np.ndarray | None = None,
    mmap: bool = True,
) -> dict[str, np.ndarray]:
    """The CA-TPA 100k-run LHS results as a dict of arrays.

    Keys: ``run`` [n_run] int, ``time`` [n_day] datetime64[D] (2015-01-01 .. 2023-12-31),
    ``yyyyddd`` [n_day] float64, the daily arrays of :data:`LHS_ANA_COLUMNS` (or ``variables``)
    [n_run, n_day] float32, ``yields`` [n_run, 7] int64 (kg/ha, seasons 2015..2021) and
    ``season_year`` [7]. ``runs`` selects rows (a copy); otherwise, with ``mmap=True`` the daily
    arrays are read-only memory maps of ``npy/*.npy`` (extracted on first use, 7.9 GB), with
    ``mmap=False`` they are decompressed into RAM.
    """
    d = _lhs_dir(data_dir)
    names = list(LHS_ANA_COLUMNS) if variables is None else list(variables)
    unknown = [n for n in names if n not in LHS_ANA_COLUMNS]
    if unknown:
        raise KeyError(f"unknown LHS variables {unknown}; available {list(LHS_ANA_COLUMNS)}")
    sel: Any = slice(None) if runs is None else runs

    out: dict[str, np.ndarray] = {}
    if mmap:
        npy = extract_lhs_npy(d, [*names, "run", "time"])
        load = lambda nm: np.load(npy / f"{nm}.npy", mmap_mode="r")  # noqa: E731
    else:
        z = np.load(d / "catpa_daily.npz")
        load = lambda nm: z[nm]  # noqa: E731
    run = np.asarray(load("run"))
    t = np.asarray(load("time"), dtype=np.float64)
    time = _yyyyddd_to_dates(t)
    if not (np.diff(time) == np.timedelta64(1, "D")).all():
        raise ValueError("catpa_daily.npz time axis is not contiguous daily")
    out["run"] = run[sel]
    out["time"] = time
    out["yyyyddd"] = np.round(t, 3)
    for nm in names:
        a = load(nm)
        if a.shape != (run.size, time.size):
            raise ValueError(f"{nm}: shape {a.shape} != ({run.size}, {time.size})")
        out[nm] = a if runs is None else np.asarray(a[sel])

    y = np.load(d / "catpa_yield.npz")
    yr, ys, yv = y["run"], y["season"], y["yield_kg_ha"]
    n_season = int(ys.max()) + 1
    table = np.full((run.size, n_season), -1, dtype=np.int64)
    pos = np.searchsorted(run, yr)
    if not (run[np.clip(pos, 0, run.size - 1)] == yr).all():
        raise ValueError("catpa_yield.npz has runs missing from catpa_daily.npz")
    table[pos, ys] = yv
    if (table < 0).any():
        raise ValueError(f"{int((table < 0).any(axis=1).sum())} runs lack some seasonal yields")
    out["yields"] = table[sel]
    out["season_year"] = np.arange(2015, 2015 + n_season)
    return out


def load_catpa_params(path: str | Path | None = None, *, canonical: bool = False) -> pd.DataFrame:
    """``AutoAnalysis/parameter.csv`` (the LHS matrix) with index ``run`` = row number (0-based).

    ``canonical=True`` renames the columns to the field names of
    :func:`agrijax.io.rzwqm.params.canonical_name` (``"Pore Size (c2)"`` -> ``lam_1``).
    """
    p = Path(path) if path is not None else PARAMETER_CSV
    df = pd.read_csv(p, dtype=np.float64)
    df.index = pd.RangeIndex(len(df), name="run")
    if canonical:
        from .rzwqm.params import canonical_name

        df = df.rename(columns={c: canonical_name(c)[0] for c in df.columns})
    return df
