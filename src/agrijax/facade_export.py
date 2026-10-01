"""Export results: pandas, xarray, DSSAT-style CSV files and DSSAT-format ``.OUT`` files (``aj.export``).

The quick start of :mod:`agrijax.dssat` continued (the same objects, six lines)::

    import agrijax as aj

    exp = aj.dssat.experiment("UFGA8201")
    season = exp.run(treatment=4)
    batch = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14]).run(
        {"G2": [800.0, 900.0, 1000.0]}
    )
    df = season.to_frame()                  # the daily table (pandas); season.to_pandas() is the same
    ds = aj.export.to_xarray([season, exp.run(treatment=6)])  # xarray: dims season x day
    tab = batch.to_pandas()                 # one row per season: scenario and cultivar fields, SUMMARY
    bx = batch.to_xarray()                  # xarray: dim season, the scenario fields as coordinates
    files = season.write_csv("out")         # {"daily": ..._daily.csv, "summary": ..._summary.csv}
    outs = season.write_dssat_out("out")    # {"PlantGro.OUT": ..., "SoilWat.OUT": ..., "Summary.OUT": ...}

**What is exported.** A :class:`~agrijax.dssat.Season` (or a :class:`~agrijax.dssat.Reference`, the
``dscsm048`` run, so the two can sit side by side) keeps a daily table; a
:class:`~agrijax.dssat.BatchResult` (also :class:`~agrijax.dssat.Scenarios` and
:class:`~agrijax.dssat.DssatBatch`) keeps one row of end-of-season values per season and **no daily
series**, so its xarray has the dimension ``season`` only. ``to_xarray`` of seasons has the dimensions
``season`` x ``day`` (``day`` = calendar days from each run's planting date, negative before it, so
that seasons line up by crop stage whatever their start; shorter runs are padded with NaN) and the
coordinates ``date`` and ``yrdoy`` (``season`` x ``day``), ``treatment``, ``source``, ``trno``,
``planting`` and the cultivar coefficients (along ``season``).

**DSSAT-style CSV files** (:func:`write_csv`, :func:`write_daily_csv`, :func:`write_summary_csv`)
use DSSAT's column names where the quantity is the same (``YEAR``, ``DOY``, ``DAP``, ``LAID``, ``CWAD``,
``GWAD``, ... of ``PlantGro.OUT``, ``SWTD`` of ``SoilWat.OUT``, ``ADAT``, ``MDAT``, ``HWAM``, ``CWAM``
... of ``Summary.OUT``; the mapping is :data:`SERIES`). **They are tables, not DSSAT ``.OUT`` files**:
DSSAT's fixed-width layout, its ``*RUN`` blocks and its rounding are not reproduced, and the file
says so in its header (lines that start with ``!``, DSSAT's own comment character; ``#`` would clash
with the column names ``L#SD`` and ``G#AD``); the files of DSSAT's layout are the next paragraph's. Missing
values are ``-99``, DSSAT's convention. Read a file
with :func:`read_csv`, or ``pandas.read_csv(path, comment="!", na_values=[-99],
float_precision="round_trip")``. ``DAS`` is not written (a season starts on the day its simulation
starts, not necessarily DSSAT's simulation start day). The columns beyond
``PlantGro.OUT`` / ``SoilWat.OUT`` / ``Summary.OUT`` (``season``, ``SOURCE``, the cultivar
coefficients, the scenario fields) are named in the header. Only :data:`~agrijax.dssat.DAILY`
(``LAID``, ``CWAD``, ``GWAD``, ``SWTD``) is what :meth:`~agrijax.dssat.Season.compare_daily` compares with
DSSAT; the other ``PlantGro.OUT`` columns (``full=True``) are written as the model computes them, unrounded.

**DSSAT-format ``.OUT`` files** (:func:`write_dssat_out`, :meth:`~agrijax.dssat.Season.write_dssat_out`)
are ``PlantGro.OUT``, ``SoilWat.OUT`` and ``Summary.OUT`` in the fixed-width layout of DSSAT-CSM v4.8.6's
own files (the ``*RUN`` header, the ``@`` column line, the column widths and the number formats, read
off its ``UFGA8201`` output), so that tools which read DSSAT's output files read them. **Agri-JAX writes
them, the DSSAT program does not**: the first lines of each file say so, with the Agri-JAX version, the
precision and the soil evaporation method of the run. Only the columns Agri-JAX simulates are written
(``PlantGro.OUT``: ``YEAR`` ``DOY`` ``DAS`` ``DAP`` and the ``PlantGro.OUT`` columns of :data:`SERIES`;
``SoilWat.OUT``: ``YEAR`` ``DOY`` ``DAS`` ``SWTD``; ``Summary.OUT``: the identifiers and
:data:`SUMMARY_ORDER`); the others are left out, not filled with ``-99`` or zeros
(:data:`OUT_NOT_WRITTEN`, :func:`write_dssat_out`). A season is DSSAT-format output only through this
writer: a :class:`~agrijax.dssat.Reference` is DSSAT's own run, and its files are in ``reference.out``.

Pure host-side table code: pandas and xarray are imported on first use, JAX is never imported.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

import agrijax
from agrijax.dssat import (
    CULTIVAR,
    DAILY,
    SUMMARY,
    BatchResult,
    DssatBatch,
    Reference,
    Scenarios,
    Season,
)

__all__ = [
    "DSSAT_NAMES",
    "NOT_DSSAT_OUT",
    "OUT_NOT_WRITTEN",
    "SERIES",
    "read_csv",
    "to_frame",
    "to_pandas",
    "to_xarray",
    "write_csv",
    "write_daily_csv",
    "write_dssat_out",
    "write_summary_csv",
]


class _Series(NamedTuple):
    dssat: str
    file: str
    long_name: str
    units: str


def _daily_text(key: str) -> tuple[str, str]:
    name, _, rest = DAILY[key].partition(" [")
    return name, rest.rstrip("]")


#: the daily series of a season: Agri-JAX name -> DSSAT column, DSSAT file, meaning, unit. The first
#: four are :data:`agrijax.dssat.DAILY`; the others are the ``PlantGro.OUT`` quantities of
#: :func:`agrijax.processes.crop.ceres_maize.plantgro_outputs` (``Season.outputs``, unrounded). The
#: order is the order of the columns of the CSV file.
SERIES: dict[str, _Series] = {
    "lsd": _Series("L#SD", "PlantGro.OUT", "leaf number per stem (VSTAGE)", "1"),
    "gstd": _Series("GSTD", "PlantGro.OUT", "growth stage (RSTAGE)", "1"),
    "lai": _Series("LAID", "PlantGro.OUT", *_daily_text("lai")),
    "lwad": _Series("LWAD", "PlantGro.OUT", "leaf weight", "kg ha-1"),
    "swad": _Series("SWAD", "PlantGro.OUT", "stem weight", "kg ha-1"),
    "gwad": _Series("GWAD", "PlantGro.OUT", *_daily_text("gwad")),
    "rwad": _Series("RWAD", "PlantGro.OUT", "root weight", "kg ha-1"),
    "cwad": _Series("CWAD", "PlantGro.OUT", *_daily_text("cwad")),
    "g_ad": _Series("G#AD", "PlantGro.OUT", "grain number", "m-2"),
    "pwad": _Series("PWAD", "PlantGro.OUT", "ear weight", "kg ha-1"),
    "wspd": _Series("WSPD", "PlantGro.OUT", "water stress of photosynthesis (1 - SWFAC)", "1"),
    "wsgd": _Series("WSGD", "PlantGro.OUT", "water stress of expansion growth (1 - TURFAC)", "1"),
    "ewsd": _Series("EWSD", "PlantGro.OUT", "excess-water stress (SATFAC)", "1"),
    "rdpd": _Series("RDPD", "PlantGro.OUT", "rooting depth", "m"),
    "dttd": _Series("DTTD", "PlantGro.OUT", "thermal time of the day", "degC d"),
    "swtd": _Series("SWTD", "SoilWat.OUT", *_daily_text("swtd")),
}
#: Agri-JAX series name -> DSSAT column name
DSSAT_NAMES: dict[str, str] = {k: v.dssat for k, v in SERIES.items()}
#: the end-of-season columns of ``Summary.OUT`` an export writes, in their order there
SUMMARY_ORDER = ("SDAT", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT", "CWAM", "HWAM")
_SUMMARY_TEXT = {
    "SDAT": "simulation start date, YYYYDDD",
    "PDAT": "planting date, YYYYDDD",
    "EDAT": "emergence date, YYYYDDD",
    "ADAT": "silking date, YYYYDDD",
    "MDAT": "maturity date, YYYYDDD",
    "HDAT": "harvest date, YYYYDDD",
    "CWAM": "tops weight at maturity [kg ha-1]",
    "HWAM": "grain yield at maturity [kg ha-1]",
}
#: the sentence every written file carries in its header
NOT_DSSAT_OUT = (
    "NOT a DSSAT output file: this is not a genuine PlantGro.OUT, SoilWat.OUT or Summary.OUT; DSSAT's "
    "fixed-width layout, *RUN blocks and rounding are not reproduced here (write_dssat_out writes that "
    "layout; those files, too, are written by Agri-JAX and not produced by the DSSAT program)"
)
_AGRIJAX_SOURCE = (
    "Agri-JAX, an independent implementation from published equations, validated against DSSAT-CSM "
    "(BSD-3) and RZWQM2 outputs; the DSSAT-CSM v4.8.6 maize model (CERES-Maize), nitrogen off"
)
_DSSAT_SOURCE = (
    "a DSSAT-CSM v4.8.6.0 (dscsm048) run, its PlantGro.OUT / SoilWat.OUT / Summary.OUT values re-tabulated "
    "by Agri-JAX"
)
_KEY = re.compile(r"^(?P<exp>.+)_t(?P<n>\d+)$")
#: ``ISTAGE`` code of the emergence stage in a season's stage output (the first day it holds is EDAT)
_EMERGENCE_STAGE = 1
#: the date columns of a summary (``YYYYDDD`` integers)
_DATES = ("SDAT", "PDAT", "EDAT", "ADAT", "MDAT", "HDAT")


def _is_run(obj: Any) -> bool:
    return isinstance(obj, Season | Reference)


def _is_table(obj: Any) -> bool:
    return isinstance(obj, BatchResult | Scenarios | DssatBatch)


@dataclass(frozen=True)
class _Run:
    """One season or reference run, normalised: the daily table (``yrdoy`` and the series), the
    identity (``EXNAME``, ``TRNO``), the planting date (``YYYYDDD``), cultivar and summary."""

    source: str
    treatment: str
    exname: str
    trno: int | None
    planting: int | None
    cultivar: dict[str, float]
    summary: dict[str, float]
    daily: Any
    #: the precision of an Agri-JAX season (``"float64"`` / ``"float32"``); ``""`` for a DSSAT run
    precision: str = ""
    #: DSSAT's soil evaporation option the run used (``"R"`` / ``"S"``), ``""`` when not known
    mesev: str = ""
    #: the experiment's own names (``"experiment"``: the ``*EXP.DETAILS`` text, ``"treatment"``: TNAME)
    labels: dict[str, str] | None = None
    #: dates of the run as ``YYYYDDD`` where the run knows them (a season: the simulation start, the
    #: emergence from its stage output, the harvest of its experiment); ``None`` when not known
    sdat: int | None = None
    edat: int | None = None
    hdat: int | None = None


def _identity(key: str, inputs: Any) -> tuple[str, int | None]:
    exp, trno = getattr(inputs, "exp", None), getattr(inputs, "trno", None)
    if exp is not None and trno is not None:
        return str(exp), int(trno)
    m = _KEY.match(key)
    return (m["exp"], int(m["n"])) if m else (key, None)


def _planting(inputs: Any) -> int | None:
    v = getattr(getattr(inputs, "params_crop", None), "yrplt", None)
    if v is None:
        return None
    x = float(np.asarray(v).reshape(-1)[0])
    return int(x) if np.isfinite(x) else None


def _extra(outputs: Any, n: int) -> dict[str, np.ndarray]:
    """The ``SERIES`` of a season's ``outputs`` (``[days]`` or ``[days, 1]`` arrays), first ``n`` days."""
    out: dict[str, np.ndarray] = {}
    for k in SERIES:
        a = None if not outputs else outputs.get(k)
        if a is None:
            continue
        a = np.asarray(a, dtype=float)
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim == 1 and a.shape[0] >= n:
            out[k] = a[:n]
    return out


def _finite_date(v: Any) -> int | None:
    """A ``YYYYDDD`` date from a number, ``None`` when it is missing."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return int(x) if math.isfinite(x) and x > 0 else None


def _season_dates(obj: Any, daily: Any) -> tuple[int | None, int | None, int | None]:
    """``(SDAT, EDAT, HDAT)`` of a season: the simulation start and the harvest its inputs record (the
    harvest only when the season ends on that day, which a season of another cultivar need not), and
    the first day of the emergence stage of its stage output; ``None`` where not known."""
    ref = getattr(getattr(obj, "inputs", None), "summary", None) or {}
    days = daily["yrdoy"].to_numpy(dtype=float)
    sdat, hdat = _finite_date(ref.get("SDAT")), _finite_date(ref.get("HDAT"))
    if hdat is not None and not (len(days) and days[-1] == hdat):
        hdat = None
    edat = None
    stage = None if not getattr(obj, "outputs", None) else obj.outputs.get("istage")
    if stage is not None:
        st = np.asarray(stage, dtype=float).reshape(len(stage), -1)[:, 0][: len(days)]
        hit = np.nonzero(st == _EMERGENCE_STAGE)[0]
        edat = _finite_date(days[hit[0]]) if hit.size else None
    return sdat, edat, hdat


def _run(obj: Season | Reference, full: bool) -> _Run:
    daily = obj.daily.reset_index(drop=True).copy()
    if "yrdoy" not in daily.columns:
        raise ValueError("the daily table has no 'yrdoy' column (dates as YYYYDDD)")
    if full and isinstance(obj, Season):
        for k, v in _extra(obj.outputs, len(daily)).items():
            if k not in daily.columns:
                daily[k] = v
    summary = {str(k): float(v) for k, v in obj.summary.items()}
    inputs = getattr(obj, "inputs", None)
    exname, trno = _identity(obj.treatment, inputs)
    planting = _planting(inputs)
    if planting is None and np.isfinite(summary.get("PDAT", np.nan)):
        planting = int(summary["PDAT"])
    is_season = isinstance(obj, Season)
    sdat, edat, hdat = _season_dates(obj, daily) if is_season else (None, None, None)
    return _Run(
        "Agri-JAX" if is_season else "DSSAT",
        obj.treatment,
        exname,
        trno,
        planting,
        {str(k): float(v) for k, v in (obj.cultivar or {}).items()},
        summary,
        daily,
        str(getattr(obj, "precision", "")) if is_season else "",
        str(getattr(inputs, "mesev", "") or ""),
        dict(getattr(obj, "labels", None) or {}),
        sdat,
        edat,
        hdat,
    )


def _runs(obj: Any, full: bool) -> list[_Run]:
    if _is_run(obj):
        return [_run(obj, full)]
    if isinstance(obj, Sequence) and not isinstance(obj, str | bytes):
        items = list(obj)
        if not items:
            raise ValueError("nothing to export: the sequence of seasons is empty")
        if all(_is_run(o) for o in items):
            return [_run(o, full) for o in items]
    raise TypeError(
        f"cannot export a {type(obj).__name__}: pass a Season or Reference (or a list of them), or a "
        "BatchResult / Scenarios / DssatBatch"
    )


def _dates(yrdoy: np.ndarray) -> Any:
    """``datetime64[ns]`` of ``YYYYDDD`` values (NaT where not finite)."""
    import pandas as pd

    v = np.asarray(yrdoy, dtype=float)
    ok = np.isfinite(v)
    s = pd.Series([f"{int(x)}" if k else None for x, k in zip(v, ok, strict=True)], dtype=object)
    return pd.to_datetime(s, format="%Y%j").to_numpy().astype("datetime64[ns]")


def _dap(run: _Run) -> np.ndarray | None:
    """Calendar days from the planting date to each day of the run (negative before planting), or
    ``None`` when the planting date is not known."""
    import pandas as pd

    if run.planting is None:
        return None
    p = pd.to_datetime(f"{run.planting}", format="%Y%j")
    yd = run.daily["yrdoy"].to_numpy(dtype=float)
    if not np.all(np.isfinite(yd)):
        raise ValueError(f"{run.treatment}: the daily table has days without a date (yrdoy)")
    return np.asarray((pd.DatetimeIndex(_dates(yd)) - p).days.to_numpy(), dtype=np.int64)


def _numbers(values: Sequence[int | None]) -> np.ndarray:
    """Integers as an integer array, floats with NaN where one is missing."""
    if None in values:
        return np.asarray([np.nan if v is None else v for v in values], dtype=float)
    return np.asarray(values, dtype=np.int64)


def _xarray() -> Any:
    try:
        import xarray as xr
    except ImportError as e:  # pragma: no cover (xarray is a dependency of agrijax)
        raise ImportError("to_xarray needs xarray: pip install xarray") from e
    return xr


# ------------------------------------------------------------------------------ pandas
def to_frame(obj: Any, *, full: bool = False, dssat_names: bool = False) -> Any:
    """The result as a pandas table.

    * a :class:`~agrijax.dssat.Season` / :class:`~agrijax.dssat.Reference`: its daily table (``date``,
      ``yrdoy``, ``lai``, ``cwad``, ``gwad``, ``swtd``; the series of :data:`~agrijax.dssat.DAILY`), one
      row per day, with the treatment, source, cultivar and summary in ``DataFrame.attrs``;
    * a list of them: the daily tables stacked, with the columns ``season`` (position in the list) and
      ``treatment`` in front;
    * a :class:`~agrijax.dssat.BatchResult` (also ``Scenarios``, ``DssatBatch``): its table, one row
      per season (scenario and cultivar fields, then the end-of-season values), a copy.

    ``full=True`` adds the other ``PlantGro.OUT`` quantities a season computes (:data:`SERIES`,
    unrounded). ``dssat_names=True`` returns the daily table as the DSSAT-style CSV file writes it
    (:func:`write_daily_csv`: ``RUNNO``, ``TRNO``, ``YEAR``, ``DOY``, ``DAP`` and the DSSAT column names);
    it does not apply to a batch."""
    import pandas as pd

    if _is_table(obj):
        t = obj.table.reset_index(drop=True).copy()
        if isinstance(obj, BatchResult):
            t.attrs = {"timing": dict(obj.timing)}
        return t
    runs = _runs(obj, full)
    if dssat_names:
        return _dssat_daily(runs)
    if _is_run(obj):
        r = runs[0]
        r.daily.attrs = {
            "treatment": r.treatment,
            "source": r.source,
            "cultivar": dict(r.cultivar),
            "summary": dict(r.summary),
        }
        return r.daily
    frames = []
    for i, r in enumerate(runs):
        d = r.daily.copy()
        d.insert(0, "treatment", r.treatment)
        d.insert(0, "season", i)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def to_pandas(obj: Any, *, full: bool = False, dssat_names: bool = False) -> Any:
    """:func:`to_frame` (the same table; both names are offered)."""
    return to_frame(obj, full=full, dssat_names=dssat_names)


# ------------------------------------------------------------------------------ xarray
def to_xarray(obj: Any, *, full: bool = False) -> Any:
    """The result as an :class:`xarray.Dataset`.

    * a :class:`~agrijax.dssat.Season` / :class:`~agrijax.dssat.Reference`, or a list of them: the
      dimensions ``season`` (position in the list; one for a single season) x ``day`` (calendar days
      from the run's planting date, negative before it, so that seasons and a reference run line up by
      crop stage; shorter runs are padded with NaN; without a planting date for every run, the index of
      the day in the run). Data variables:
      the daily series (``lai``, ``cwad``, ``gwad``, ``swtd``; ``full=True`` adds the other
      ``PlantGro.OUT`` quantities, :data:`SERIES`) along (``season``, ``day``) and the end-of-season
      values (``ADAT``, ``MDAT``, ``HWAM``, ``CWAM``, dates as ``YYYYDDD``) along ``season``.
      Coordinates: ``date`` (``datetime64``) and ``yrdoy`` (``YYYYDDD``) along (``season``, ``day``);
      ``treatment``, ``source`` (``"Agri-JAX"`` or ``"DSSAT"``), ``trno``, ``planting`` (``YYYYDDD``),
      ``precision`` (``"float64"`` / ``"float32"`` of an Agri-JAX season, empty for DSSAT), ``mesev``
      (the soil evaporation, DSSAT's ``MESEV``: ``R`` / ``S``, so a swapped season is recognisable)
      and the cultivar coefficients (``P1`` ... ``PHINT``) along ``season``;
    * a :class:`~agrijax.dssat.BatchResult` (also ``Scenarios``, ``DssatBatch``): the dimension
      ``season`` (one per row of its table) only, a batch keeps no daily series. Data variables: the
      end-of-season values; coordinates: every other column of the table (``scenario``, ``year``,
      ``sowing_shift``, ``sample``, the cultivar coefficients, ``PDAT``), along ``season``. The batch's
      timing is in ``attrs``.

    Variables carry ``long_name``, ``units`` and ``dssat_name`` attributes; the dataset's ``attrs``
    say that these are Agri-JAX results, not DSSAT output files."""
    xr = _xarray()
    note: dict[str, Any] = {
        "title": "Agri-JAX results",
        "note": NOT_DSSAT_OUT,
        "package": f"agrijax {agrijax.__version__}",
    }
    if _is_table(obj):
        t = obj.table.reset_index(drop=True)
        data_cols = [c for c in SUMMARY_ORDER if c != "PDAT" and c in t.columns]
        fields = [c for c in t.columns if c not in data_cols]
        ds = xr.Dataset(
            {c: ("season", t[c].to_numpy(dtype=float)) for c in data_cols},
            coords={"season": np.arange(len(t)), **{c: ("season", t[c].to_numpy()) for c in fields}},
        )
        for c in data_cols:
            ds[c].attrs = {"long_name": _SUMMARY_TEXT[c], "dssat_name": c}
        ds.attrs = dict(note)
        if isinstance(obj, BatchResult):
            ds.attrs.update(
                {
                    f"timing_{k}": v if isinstance(v, int | float | str) else str(v)
                    for k, v in obj.timing.items()
                }
            )
        return ds

    runs = _runs(obj, full)
    n = len(runs)
    # the day axis counts calendar days from each run's planting date (negative before it), so seasons
    # (and a reference run, which starts a day or two earlier) line up by crop stage; without a planting
    # date for every run it is the index of the day in the run
    daps = [_dap(r) for r in runs]
    from_planting = all(d is not None for d in daps)
    offs = [np.asarray(d) for d in daps] if from_planting else [np.arange(len(r.daily)) for r in runs]
    lo, hi = min(int(o.min()) for o in offs), max(int(o.max()) for o in offs)
    m = hi - lo + 1
    names: list[str] = []
    for r in runs:
        names += [c for c in r.daily.columns if c not in ("date", "yrdoy") and c not in names]
    series = {k: np.full((n, m), np.nan) for k in names}
    yrdoy = np.full((n, m), np.nan)
    date = np.full((n, m), np.datetime64("NaT"), dtype="datetime64[ns]")
    for i, r in enumerate(runs):
        at = offs[i] - lo
        yd = r.daily["yrdoy"].to_numpy(dtype=float)
        yrdoy[i, at] = yd
        date[i, at] = _dates(yd)
        for c in names:
            if c in r.daily.columns:
                series[c][i, at] = r.daily[c].to_numpy(dtype=float)
    coords: dict[str, Any] = {
        "season": np.arange(n),
        "day": np.arange(lo, hi + 1),
        "date": (("season", "day"), date),
        "yrdoy": (("season", "day"), yrdoy),
        "treatment": ("season", [r.treatment for r in runs]),
        "source": ("season", [r.source for r in runs]),
        "trno": ("season", _numbers([r.trno for r in runs])),
        "planting": ("season", _numbers([r.planting for r in runs])),
        "precision": ("season", [r.precision for r in runs]),
        "mesev": ("season", [r.mesev for r in runs]),
    }
    for c in CULTIVAR:
        if any(c in r.cultivar for r in runs):
            coords[c] = ("season", np.asarray([r.cultivar.get(c, np.nan) for r in runs], dtype=float))
    final = [c for c in SUMMARY_ORDER if c != "PDAT" and (c in SUMMARY or any(c in r.summary for r in runs))]
    data: dict[str, Any] = {k: (("season", "day"), v) for k, v in series.items()}
    for c in final:
        data[c] = ("season", np.asarray([r.summary.get(c, np.nan) for r in runs], dtype=float))
    ds = xr.Dataset(data, coords=coords)
    for k in names:
        s = SERIES.get(k)
        if s is not None:
            ds[k].attrs = {
                "long_name": s.long_name,
                "units": s.units,
                "dssat_name": s.dssat,
                "dssat_file": s.file,
            }
    for c in final:
        ds[c].attrs = {"long_name": _SUMMARY_TEXT[c], "dssat_name": c, "dssat_file": "Summary.OUT"}
    ds["day"].attrs = {
        "long_name": "days from the planting date (negative: before planting)"
        if from_planting
        else "index of the day in the run (0 = its first day); no planting date known"
    }
    ds["yrdoy"].attrs = {"long_name": "date as YYYYDDD"}
    ds["planting"].attrs = {"long_name": "planting date as YYYYDDD"}
    ds["precision"].attrs = {"long_name": "precision of the Agri-JAX run (empty for a DSSAT run)"}
    ds["mesev"].attrs = {
        "long_name": "soil evaporation method (DSSAT's MESEV: R Ritchie, S SALUS; empty when not known)"
    }
    ds.attrs = dict(note)
    return ds


# ------------------------------------------------------------------------------ CSV files
def _dssat_daily(runs: list[_Run]) -> Any:
    """The daily CSV table: ``RUNNO``, ``TRNO``, ``YEAR``, ``DOY``, ``DAP`` (when a planting date is
    known) and the DSSAT column of every series, in :data:`SERIES` order."""
    import pandas as pd

    frames = []
    for i, r in enumerate(runs, start=1):
        yd = r.daily["yrdoy"].to_numpy(dtype=float)
        if not np.all(np.isfinite(yd)):
            raise ValueError(f"{r.treatment}: the daily table has days without a date (yrdoy)")
        yd = yd.astype(np.int64)
        n = len(yd)
        cols: dict[str, Any] = {
            "RUNNO": np.full(n, i, dtype=np.int64),
            "TRNO": pd.array([r.trno] * n, dtype="Int64"),
            "YEAR": yd // 1000,
            "DOY": yd % 1000,
        }
        dap = _dap(r)
        if dap is not None:  # PlantGro.OUT starts on the planting day; the days before it get 0
            cols["DAP"] = np.maximum(dap, 0)
        for k, s in SERIES.items():
            if k in r.daily.columns:
                cols[s.dssat] = r.daily[k].to_numpy(dtype=float)
        frames.append(pd.DataFrame(cols))
    d = pd.concat(frames, ignore_index=True)
    lead = ["RUNNO", "TRNO", "YEAR", "DOY", "DAP"]
    order = [c for c in lead if c in d.columns] + [s.dssat for s in SERIES.values() if s.dssat in d.columns]
    return d[order]


def _ints(df: Any, names: Sequence[str]) -> Any:
    """Columns of whole numbers (dates, ``YYYYDDD``) as integers with missing values kept."""
    import pandas as pd

    for c in names:
        if c in df.columns:
            s: Any = pd.to_numeric(df[c], errors="coerce")
            if bool((s.dropna() % 1 == 0).all()):
                df[c] = s.round().astype("Int64")
    return df


def _summary_frame(runs: list[_Run]) -> Any:
    """One row per run: ``RUNNO``, ``TRNO``, ``EXNAME``, the ``Summary.OUT`` columns known, the cultivar
    coefficients, ``SOURCE``."""
    import pandas as pd

    rows = []
    for i, r in enumerate(runs, start=1):
        s = dict(r.summary)
        if r.planting is not None:
            s.setdefault("PDAT", float(r.planting))
        rows.append({"RUNNO": i, "EXNAME": r.exname, **s, **r.cultivar, "SOURCE": r.source})
    d = pd.DataFrame(rows)
    d.insert(1, "TRNO", pd.array([r.trno for r in runs], dtype="Int64"))
    for c in SUMMARY:  # the values every season has are always columns
        if c not in d.columns:
            d[c] = np.nan
    summary = [c for c in SUMMARY_ORDER if c in SUMMARY or (c in d.columns and bool(d[c].notna().any()))]
    coef = [c for c in CULTIVAR if c in d.columns]
    return _ints(d[["RUNNO", "TRNO", "EXNAME", *summary, *coef, "SOURCE"]].copy(), _DATES)


def _write(path: str | os.PathLike[str], comments: Sequence[str], df: Any, na_rep: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        for ln in comments:
            f.write(f"! {ln}\n")
        df.to_csv(f, index=False, na_rep=na_rep, lineterminator="\n")
    return p


def _source_line(runs: Sequence[_Run]) -> str:
    src = {r.source for r in runs}
    parts = []
    if "Agri-JAX" in src:
        parts.append(f"SOURCE Agri-JAX: {_AGRIJAX_SOURCE}")
    if "DSSAT" in src:
        parts.append(f"SOURCE DSSAT: {_DSSAT_SOURCE}")
    return "; ".join(parts)


def _runs_line(runs: Sequence[_Run]) -> str:
    """How each run was made: ``RUNNO 1 UFGA8201_t04: Agri-JAX float64, MESEV R; ...``."""
    parts = []
    for i, r in enumerate(runs, start=1):
        how = " ".join(x for x in (r.source, r.precision) if x)
        parts.append(f"RUNNO {i} {r.treatment}: {how}" + (f", MESEV {r.mesev}" if r.mesev else ""))
    return "runs: " + "; ".join(parts)


def _header(
    kind: str, source: str, columns: str, na_rep: str, runs: Sequence[_Run] | None = None
) -> list[str]:
    return [
        f"Agri-JAX export, {kind} (CSV, written by agrijax {agrijax.__version__}).",
        NOT_DSSAT_OUT + ".",
        f"source: {source}",
        *([_runs_line(runs)] if runs else []),
        columns,
        f"Missing values are {na_rep} (DSSAT's convention). Read with "
        "agrijax.facade_export.read_csv(path) or "
        f'pandas.read_csv(path, comment="!", na_values=[{na_rep}], float_precision="round_trip").',
    ]


def write_daily_csv(
    obj: Any, path: str | os.PathLike[str], *, full: bool = True, na_rep: str = "-99"
) -> Path:
    """The daily series of a season (a :class:`~agrijax.dssat.Season`, a :class:`~agrijax.dssat.Reference`
    or a list of them) as a DSSAT-style CSV file at ``path`` (directories are created); returns the path.

    One row per day: ``RUNNO`` (1-based position in the list), ``TRNO``, ``YEAR``, ``DOY``, ``DAP`` (days
    after planting when the planting date is known, held at 0 on the days before planting, which
    ``PlantGro.OUT`` does not report) and the DSSAT column names of :data:`SERIES` (``full=True``: all of
    them; ``False``: the four of :data:`~agrijax.dssat.DAILY`). **Not a DSSAT ``.OUT`` file**: the
    header lines (starting with ``!``) say so, and :func:`write_dssat_out` writes the ``PlantGro.OUT`` and
    ``SoilWat.OUT`` layout; see the module docstring.
    Missing values are written as ``na_rep`` (``-99``). A batch keeps no daily series (``TypeError``)."""
    if _is_table(obj):
        raise TypeError(f"a {type(obj).__name__} keeps no daily series: write_summary_csv writes its table")
    runs = _runs(obj, full)
    d = _dssat_daily(runs)
    cols = [c for c in d.columns if c in {s.dssat for s in SERIES.values()}]
    text = (
        "columns follow DSSAT's PlantGro.OUT (SWTD: SoilWat.OUT) where the quantity is the same: "
        "RUNNO and TRNO as in Summary.OUT, YEAR and DOY the date, DAP days after planting, "
        + ", ".join(f"{c} {_series_of(c).long_name} [{_series_of(c).units}]" for c in cols)
        + ". RUNNO numbers the runs in the order given."
    )
    return _write(path, _header("daily series", _source_line(runs), text, na_rep, runs), d, na_rep)


def _series_of(dssat: str) -> _Series:
    return next(s for s in SERIES.values() if s.dssat == dssat)


def write_summary_csv(obj: Any, path: str | os.PathLike[str], *, na_rep: str = "-99") -> Path:
    """The end-of-season values as a DSSAT-style CSV file at ``path`` (directories are created); returns
    the path. **Not a DSSAT ``.OUT`` file** (:func:`write_dssat_out` writes the ``Summary.OUT`` layout; see
    the module docstring).

    * a season, a reference run or a list of them: one row per run with the ``Summary.OUT`` names
      ``RUNNO``, ``TRNO``, ``EXNAME``, ``PDAT`` (when known), ``ADAT``, ``MDAT``, ``CWAM``, ``HWAM``
      (dates as ``YYYYDDD``; a reference run also ``SDAT``, ``EDAT``, ``HDAT``), then the cultivar
      coefficients (``P1`` ... ``PHINT``, names of ``MZCER048.CUL``) and ``SOURCE``;
    * a :class:`~agrijax.dssat.BatchResult` (also ``Scenarios``, ``DssatBatch``): ``RUNNO`` and its table, one
      row per season (``scenario``, ``year``, ``sowing_shift``, ``sample``: Agri-JAX scenario fields, not
      ``Summary.OUT`` columns)."""
    if _is_table(obj):
        d = obj.table.reset_index(drop=True).copy()
        d.insert(0, "RUNNO", np.arange(1, len(d) + 1))
        d = _ints(d, _DATES)
        fields = [c for c in d.columns if c not in {"RUNNO", *SUMMARY_ORDER}]
        text = (
            "columns: RUNNO numbers the seasons (one per row of the batch table); "
            + ", ".join(c for c in d.columns if c in SUMMARY_ORDER)
            + " as in Summary.OUT ("
            + "; ".join(f"{c} {_SUMMARY_TEXT[c]}" for c in d.columns if c in SUMMARY_ORDER)
            + f"); {', '.join(fields)}: Agri-JAX scenario and cultivar fields (P1 ... PHINT are the names "
            "of MZCER048.CUL), not Summary.OUT columns."
        )
        head = _header("end-of-season values of a batch", _AGRIJAX_SOURCE, text, na_rep)
        timing = getattr(obj, "timing", None) or {}
        how = [f"{k} {timing[k]}" for k in ("precision", "mesev", "devices") if timing.get(k)]
        if how:
            head.insert(3, "runs: " + ", ".join(how))
        return _write(path, head, d, na_rep)
    runs = _runs(obj, False)
    d = _summary_frame(runs)
    shown = [c for c in d.columns if c in SUMMARY_ORDER]
    text = (
        "columns follow DSSAT's Summary.OUT where the quantity is the same: RUNNO, TRNO, EXNAME, "
        + "; ".join(f"{c} {_SUMMARY_TEXT[c]}" for c in shown)
        + ". P1 ... PHINT are the cultivar coefficients (names of MZCER048.CUL) and SOURCE the program, "
        "neither is a Summary.OUT column. RUNNO numbers the runs in the order given."
    )
    return _write(path, _header("end-of-season values", _source_line(runs), text, na_rep, runs), d, na_rep)


def write_csv(
    obj: Any,
    directory: str | os.PathLike[str],
    *,
    name: str | None = None,
    full: bool = True,
    na_rep: str = "-99",
) -> dict[str, Path]:
    """The daily and summary CSV files of ``obj`` in ``directory`` (created): ``<name>_daily.csv``
    (:func:`write_daily_csv`) and ``<name>_summary.csv`` (:func:`write_summary_csv`), as
    ``{"daily": path, "summary": path}``. A batch has the summary file only. ``name`` defaults to the
    treatment (``UFGA8201_t04``), ``agrijax_seasons`` for a list, ``batch`` for a batch. **Not DSSAT
    ``.OUT`` files**: :func:`write_dssat_out` writes those (see the module docstring)."""
    d = Path(directory)
    if _is_table(obj):
        stem = name or "batch"
        return {"summary": write_summary_csv(obj, d / f"{stem}_summary.csv", na_rep=na_rep)}
    runs = _runs(obj, False)
    stem = name or (runs[0].treatment if _is_run(obj) else "agrijax_seasons")
    return {
        "daily": write_daily_csv(obj, d / f"{stem}_daily.csv", full=full, na_rep=na_rep),
        "summary": write_summary_csv(obj, d / f"{stem}_summary.csv", na_rep=na_rep),
    }


def read_csv(path: str | os.PathLike[str], *, na_rep: str = "-99") -> Any:
    """A CSV file written by this module as a pandas table: the header lines (``!``) skipped, ``na_rep``
    (``-99``) read as NaN."""
    import pandas as pd

    return pd.read_csv(path, comment="!", na_values=[na_rep], float_precision="round_trip")


# ------------------------------------------------------------------------------ DSSAT .OUT files
class _Field(NamedTuple):
    """One column of a DSSAT fixed-width table: its name, the characters of its field (leading blanks
    included), the digits after the point of a real (0: a whole number), and how it is written."""

    name: str
    width: int
    decimals: int = 0
    #: ``"number"`` (right-aligned), ``"doy"`` (day of year, zero-padded) or ``"text"`` (one blank, then
    #: the text, left-aligned)
    kind: str = "number"
    #: the text of the ``@`` line when it is not the name (the text columns end in dots)
    head: str = ""


#: width and decimals of the columns of :data:`SERIES` in dscsm048 v4.8.6's own ``PlantGro.OUT`` and
#: ``SoilWat.OUT`` (read off its UFGA8201 output files; decimals 0 are whole numbers, rounded to the
#: nearest)
_OUT_FORMAT: dict[str, tuple[int, int]] = {
    "L#SD": (7, 1), "GSTD": (7, 0), "LAID": (7, 2), "LWAD": (7, 0), "SWAD": (7, 0), "GWAD": (7, 0),
    "RWAD": (7, 0), "CWAD": (7, 0), "G#AD": (7, 0), "PWAD": (7, 0), "WSPD": (7, 3), "WSGD": (7, 3),
    "EWSD": (7, 3), "RDPD": (7, 2), "DTTD": (6, 2), "SWTD": (8, 0),
}  # fmt: skip
_YEAR, _DOY = _Field("YEAR", 5), _Field("DOY", 4, kind="doy")
_DAS, _DAP = _Field("DAS", 6), _Field("DAP", 6)
#: the columns of ``Summary.OUT`` an export writes (:data:`SUMMARY_ORDER` follow the identifiers)
_SUMMARY_FIELDS: dict[str, _Field] = {
    "RUNNO": _Field("RUNNO", 9),
    "TRNO": _Field("TRNO", 7),
    "CR": _Field("CR", 3, kind="text"),
    "MODEL": _Field("MODEL", 9, kind="text", head="MODEL..."),
    "EXNAME": _Field("EXNAME", 9, kind="text", head="EXNAME.."),
    "TNAM": _Field("TNAM", 26, kind="text", head="TNAM" + "." * 21),
    **{c: _Field(c, 8) for c in SUMMARY_ORDER},
}
#: the model Agri-JAX re-implements (DSSAT's names for it): crop code and model name of the run header
_CROP, _MODEL = "MZ", "MZCER048"
_SOIL_EVAPORATION = {"R": "Ritchie", "S": "SALUS"}
#: the columns of dscsm048 v4.8.6's maize ``PlantGro.OUT`` and ``SoilWat.OUT`` (UFGA8201) that
#: :func:`write_dssat_out` does not write, in their order there
OUT_NOT_WRITTEN: dict[str, tuple[str, ...]] = {
    "PlantGro.OUT": tuple(
        (
            "VWAD GWGD HIAD P#AD NSTD PST1A PST2A KSTD LN%D SH%D HIPD PWDD PWTD SLAD CHTD CWID "
            "RL1D RL2D RL3D RL4D RL5D RL6D RL7D RL8D RL9D CDAD LDAD SDAD SNW0C SNW1C"
        ).split()
    ),
    "SoilWat.OUT": tuple(
        (
            "SWXD ROFC DRNC PREC IR#C IRRC LATFC DTWT DTWTM MWTD TDFD TDFC ROFD ROSD "
            "SW1D SW2D SW3D SW4D SW5D SW6D SW7D SW8D SW9D"
        ).split()
    ),
}
#: the title line of each file (the first line of dscsm048's own file; ours adds what it is)
_OUT_TITLE = {"PlantGro.OUT": "*GROWTH ASPECTS OUTPUT FILE", "SoilWat.OUT": "*SOIL WATER DAILY OUTPUT FILE"}
_NOT_DSSAT_PROGRAM = "NOT produced by the DSSAT program"


def _cell(f: _Field, v: Any) -> str:
    """One value in its field, the way DSSAT prints it: a whole number rounded to the nearest (halves
    away from zero, ``NINT``), a real fixed-point; a missing value (NaN) is ``-99``; an infinite value or
    one that does not fit its field is asterisks, as Fortran prints it."""
    if f.kind == "text":
        t = "" if v is None else str(v)
        return " " + t[: f.width - 1].ljust(f.width - 1)
    x = math.nan if v is None else float(v)
    if math.isnan(x):
        x = -99.0
    if math.isinf(x):
        return "*" * f.width
    if f.kind == "doy":
        t = f"{int(x):03d}"
    elif f.decimals == 0:
        n = math.floor(abs(x) + 0.5)
        t = str(-n if x < 0 and n else n)
    else:
        t = f"{x:.{f.decimals}f}"
        if not t.strip("-0."):  # no negative zero
            t = t.lstrip("-")
    return t.rjust(f.width) if len(t) <= f.width else "*" * f.width


def _head_line(fields: Sequence[_Field]) -> str:
    """The ``@`` line: the names right-aligned to their fields (text columns: left-aligned after one
    blank), the first character replaced by ``@``."""
    parts = [
        " " + (f.head or f.name).ljust(f.width - 1) if f.kind == "text" else (f.head or f.name).rjust(f.width)
        for f in fields
    ]
    return "@" + "".join(parts)[1:]


def _days_since(run: _Run, yrdoy: int | None) -> np.ndarray | None:
    """Calendar days from the date ``yrdoy`` to each day of the run, ``None`` without that date."""
    import pandas as pd

    if yrdoy is None:
        return None
    yd = run.daily["yrdoy"].to_numpy(dtype=float)
    if not np.all(np.isfinite(yd)):
        raise ValueError(f"{run.treatment}: the daily table has days without a date (yrdoy)")
    start = pd.to_datetime(f"{yrdoy}", format="%Y%j")
    return np.asarray((pd.DatetimeIndex(_dates(yd)) - start).days.to_numpy(), dtype=np.int64)


def _evaporation(r: _Run) -> str:
    """The soil evaporation method of a run (``Ritchie (MESEV R)``), empty when not recorded."""
    return f"{_SOIL_EVAPORATION.get(r.mesev, r.mesev)} (MESEV {r.mesev})" if r.mesev else ""


def _run_stamp(runs: Sequence[_Run], what: str) -> str:
    """The line (starting with ``*``, DSSAT's own marker) that says which program wrote a file, with
    its version, precision and soil evaporation method, and that DSSAT did not."""

    def one(values: list[str], label: str, empty: str) -> str:
        found = sorted({v for v in values if v})
        if not found:
            return f"{label} {empty}"
        return f"{label} {' and '.join(found)}" + (" (RUNNO lines)" if len(found) > 1 else "")

    return (
        f"*Agri-JAX {agrijax.__version__}: {what} in DSSAT-CSM v4.8.6 format; "
        f"{one([r.precision for r in runs], 'precision', 'not recorded')}; "
        f"{one([_evaporation(r) for r in runs], 'soil evaporation', 'not recorded')}; "
        f"generated by Agri-JAX, {_NOT_DSSAT_PROGRAM}"
    )


def _experiment_line(r: _Run) -> str:
    """The experiment as DSSAT's run header gives it: the 8-character code, a blank, then the rest of
    the ``*EXP.DETAILS`` text."""
    t = (r.labels or {}).get("experiment", "")
    return f"{t[:8]} {t[8:]}".rstrip() if t else r.exname


def _run_block(i: int, r: _Run) -> list[str]:
    """The ``*RUN`` block of one run in a daily file (the lines dscsm048 writes before each table)."""
    tnam = (r.labels or {}).get("treatment", "")[:25]
    trno = -99 if r.trno is None else r.trno
    return [
        f"*RUN {i:>3d}        : {tnam:<25s} {_MODEL:<8s} {r.exname[:8]:<8s}{trno:>5d}",
        f" MODEL          : {_MODEL} - Maize",
        f" EXPERIMENT     : {_experiment_line(r)}",
        " DATA PATH      :",
        f" TREATMENT{trno:>3d}   : {tnam:<25s} {_MODEL}",
        "  ",
    ]


def _series_keys(runs: Sequence[_Run], file: str) -> list[str]:
    return [
        k
        for k, s in SERIES.items()
        if s.file == file and s.dssat in _OUT_FORMAT and any(k in r.daily.columns for r in runs)
    ]


def _daily_out(file: str, runs: Sequence[_Run]) -> list[str]:
    """The lines of ``PlantGro.OUT`` or ``SoilWat.OUT``: one block per run, each with the run header
    and its own ``@`` table, as in a multi-treatment DSSAT run. ``PlantGro.OUT`` starts on the planting
    day (``DAP`` 0), ``SoilWat.OUT`` on the first day of the run."""
    plant = file == "PlantGro.OUT"
    keys = _series_keys(runs, file)
    das = any(r.sdat is not None for r in runs)
    dap = plant and any(r.planting is not None for r in runs)
    fields = [_YEAR, _DOY, *([_DAS] if das else []), *([_DAP] if dap else [])]
    fields += [_Field(SERIES[k].dssat, *_OUT_FORMAT[SERIES[k].dssat]) for k in keys]
    skipped = " ".join(OUT_NOT_WRITTEN[file])
    first_row = (
        "Rows start on the planting day (DAP 0)." if plant else "Rows start on the first day of the run."
    )
    out = [f"{_OUT_TITLE[file]} - Agri-JAX output in DSSAT format, {_NOT_DSSAT_PROGRAM}"]
    for i, r in enumerate(runs, start=1):
        yd = _finite_days(r)
        n = len(yd)
        nan = np.full(n, np.nan)
        counts_das = _days_since(r, r.sdat)
        counts_dap = _days_since(r, r.planting)
        cols: dict[str, np.ndarray] = {
            "YEAR": yd // 1000,
            "DOY": yd % 1000,
            "DAS": nan if counts_das is None else counts_das + 1,  # DSSAT counts its start day as DAS 1
            "DAP": nan if counts_dap is None else counts_dap,
        }
        for k in keys:
            cols[SERIES[k].dssat] = r.daily[k].to_numpy(dtype=float) if k in r.daily.columns else nan
        rows = np.nonzero(counts_dap >= 0)[0] if (plant and counts_dap is not None) else np.arange(n)
        out += ["", _run_stamp([r], f"{file} of one season"), "", *_run_block(i, r)]
        out += [
            f"! Only the columns Agri-JAX simulates are written; not written (not -99 or 0): {skipped}",
            f"! {first_row} DAS is 1 on the simulation start day, as in DSSAT, when that date is known.",
            _head_line(fields),
        ]
        out += ["".join(_cell(f, cols[f.name][j]) for f in fields) for j in rows]
    return out


def _finite_days(r: _Run) -> np.ndarray:
    yd = r.daily["yrdoy"].to_numpy(dtype=float)
    if not np.all(np.isfinite(yd)):
        raise ValueError(f"{r.treatment}: the daily table has days without a date (yrdoy)")
    return yd.astype(np.int64)


def _short(v: Any) -> str:
    if isinstance(v, str):
        return v
    x = float(v)
    return "-99" if math.isnan(x) else f"{x:.6g}"


def _summary_out(rows: list[dict[str, Any]], runs: Sequence[_Run], title: str, notes: list[str]) -> list[str]:
    """The lines of ``Summary.OUT``: the title line with the stamp, the comment lines, the ``@`` line and
    one row per run. A date or weight column is written when any run has a value (and always for the
    four values of every season, :data:`~agrijax.dssat.SUMMARY`)."""

    def known(c: str) -> bool:
        return c in SUMMARY or any(not math.isnan(float(r.get(c, math.nan))) for r in rows)

    names = ["RUNNO", "TRNO", "CR", "MODEL", "EXNAME"]
    if any(r.get("TNAM") for r in rows):
        names.append("TNAM")
    names += [c for c in SUMMARY_ORDER if known(c)]
    fields = [_SUMMARY_FIELDS[c] for c in names]
    stamp = _run_stamp(runs, "Summary.OUT").removeprefix("*")
    head = f"*SUMMARY : {title:<72}{stamp}"
    skipped = (
        "R# O# P# FNAM WSTA WYEAR SOIL_ID XLAT LONG ELEV HYEAR DWAP HWAH BWAH PWAM HWUM H#AM H#UM HIAM LAIX "
        "and every water, nitrogen, phosphorus, potassium, organic matter, productivity and seasonal "
        "environment column"
    )
    out = [
        head,
        "",
        f"! Only the columns Agri-JAX simulates are written; not written (not -99 or 0): {skipped}",
        *notes,
        _head_line(fields),
    ]
    out += ["".join(_cell(f, r.get(f.name)) for f in fields) for r in rows]
    return out


def _season_summary_rows(runs: Sequence[_Run]) -> list[dict[str, Any]]:
    rows = []
    for i, r in enumerate(runs, start=1):
        vals = {c: float(r.summary.get(c, np.nan)) for c in SUMMARY_ORDER}
        for c, v in (("SDAT", r.sdat), ("PDAT", r.planting), ("EDAT", r.edat), ("HDAT", r.hdat)):
            if v is not None and math.isnan(vals[c]):
                vals[c] = float(v)
        rows.append(
            {
                "RUNNO": i,
                "TRNO": r.trno,
                "CR": _CROP,
                "MODEL": _MODEL,
                "EXNAME": r.exname,
                "TNAM": (r.labels or {}).get("treatment", ""),
                **vals,
            }
        )
    return rows


def _batch_out(obj: Any) -> list[str]:
    """``Summary.OUT`` of a batch: one run per season, its scenario and cultivar fields (which are not
    ``Summary.OUT`` columns) in a comment table."""
    t = obj.table.reset_index(drop=True)
    key = _KEY.match(str(getattr(obj, "treatment", "") or ""))
    exname, trno = (key["exp"], int(key["n"])) if key else ("", None)
    rows = [
        {
            "RUNNO": i,
            "TRNO": trno,
            "CR": _CROP,
            "MODEL": _MODEL,
            "EXNAME": exname,
            **{c: float(t.loc[i - 1, c]) for c in SUMMARY_ORDER if c in t.columns},
        }
        for i in range(1, len(t) + 1)
    ]
    timing = getattr(obj, "timing", None) or {}
    pseudo = _Run(
        "Agri-JAX", str(getattr(obj, "treatment", "")), exname, trno, None, {}, {}, t,
        precision=str(timing.get("precision", "")), mesev=str(timing.get("mesev", "")),
    )  # fmt: skip
    fields = [c for c in t.columns if c not in SUMMARY_ORDER]
    cells = {c: [_short(v) for v in t[c]] for c in fields}
    wid = {c: max([len(c), *(len(x) for x in cells[c])]) for c in fields}
    notes = ["! RUNNO numbers the seasons of the batch (one per row of its table); its other fields, which "
             "are not Summary.OUT columns:"]  # fmt: skip
    notes.append("! RUNNO " + " ".join(c.rjust(wid[c]) for c in fields))
    notes += [
        f"! {i:>5d} " + " ".join(cells[c][i - 1].rjust(wid[c]) for c in fields) for i in range(1, len(t) + 1)
    ]
    title = f"{exname} batch of seasons" if exname else "batch of seasons"
    return _summary_out(rows, [pseudo], title, notes)


def _season_out(runs: Sequence[_Run]) -> list[str]:
    title = ((runs[0].labels or {}).get("experiment") or runs[0].exname).strip()
    notes = [
        f"! RUNNO {i}: {r.treatment}, Agri-JAX {r.precision or 'precision not recorded'}"
        + (f", soil evaporation {_evaporation(r)}" if r.mesev else "")
        for i, r in enumerate(runs, start=1)
    ]
    return _summary_out(_season_summary_rows(runs), runs, title, notes)


def _write_lines(path: Path, lines: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="latin-1", errors="replace", newline="") as f:  # one byte per column
        f.write("\n".join(lines) + "\n")
    return path


def _out_runs(obj: Any) -> list[_Run]:
    items = obj if isinstance(obj, Sequence) and not isinstance(obj, str | bytes) else [obj]
    if any(isinstance(o, Reference) for o in items):
        raise TypeError(
            "a Reference is DSSAT's own run: its PlantGro.OUT, SoilWat.OUT and Summary.OUT are in "
            "reference.out; write_dssat_out writes Agri-JAX results"
        )
    return _runs(obj, True)


def write_dssat_out(obj: Any, directory: str | os.PathLike[str], *, per_run: bool = False) -> dict[str, Path]:
    """Agri-JAX results as DSSAT-format output files in ``directory`` (created); returns
    ``{file name: path}`` (``PlantGro.OUT``, ``SoilWat.OUT``, ``Summary.OUT``).

    * a :class:`~agrijax.dssat.Season`, or a list of seasons: the three files, with the run header
      (``*RUN`` block), the ``@`` column line, the column widths and the number formats of DSSAT-CSM
      v4.8.6's own files (read off its ``UFGA8201`` output). A list is one multi-run file set, a
      ``*RUN`` block per season in ``PlantGro.OUT`` and ``SoilWat.OUT`` and a row per season in
      ``Summary.OUT`` (``RUNNO`` = position in the list), as a DSSAT run of several treatments writes
      them; ``per_run=True`` writes one directory per season instead (``<n>_<treatment>/``, ``n`` the
      position in the list, keys ``"<n>_<treatment>/PlantGro.OUT"`` ...);
    * a :class:`~agrijax.dssat.BatchResult` (also :class:`~agrijax.dssat.Scenarios`): ``Summary.OUT``
      only, one run per season; a batch keeps end-of-season values and no daily series, so it has no
      ``PlantGro.OUT`` or ``SoilWat.OUT`` (and ``per_run`` does not apply). Its scenario and cultivar
      fields are not ``Summary.OUT`` columns and are listed in comment lines.

    **These are not DSSAT's files.** Agri-JAX wrote them, the DSSAT program did not: the first lines
    of each file (``*`` and ``!`` lines, DSSAT's comment markers) say so and give the Agri-JAX
    version, the precision and the soil evaporation method (``MESEV``) of the run, and the run
    header's version line of dscsm048 is replaced by that statement. A
    :class:`~agrijax.dssat.Reference` (the ``dscsm048`` run, whose own files are in ``reference.out``)
    and a :class:`~agrijax.dssat.DssatBatch` are refused (``TypeError``).

    **Columns.** Only what Agri-JAX simulates is written, never filler: ``PlantGro.OUT`` has ``YEAR``,
    ``DOY``, ``DAS`` (when the simulation start date is known), ``DAP`` (when the planting date is
    known) and the columns of :data:`SERIES` of that file (``L#SD`` ``GSTD`` ``LAID`` ``LWAD`` ``SWAD``
    ``GWAD`` ``RWAD`` ``CWAD`` ``G#AD`` ``PWAD`` ``WSPD`` ``WSGD`` ``EWSD`` ``RDPD`` ``DTTD``);
    ``SoilWat.OUT`` has ``YEAR``, ``DOY``, ``DAS`` and ``SWTD``; ``Summary.OUT`` has ``RUNNO``,
    ``TRNO``, ``CR``, ``MODEL``, ``EXNAME``, ``TNAM`` (when the treatment name is known) and
    :data:`SUMMARY_ORDER` (``SDAT`` ``PDAT`` ``EDAT`` ``ADAT`` ``MDAT`` ``HDAT`` ``CWAM`` ``HWAM``; a
    date a season does not know is ``-99``, and a column nothing knows is left out). The other columns
    of DSSAT's maize files are **not written** (listed in :data:`OUT_NOT_WRITTEN` and in a comment line
    of each file): those Agri-JAX does not simulate (``NSTD``, ``PST1A``, ``KSTD``, ``SLAD``, ``CHTD``,
    the water-balance sums of ``SoilWat.OUT``, ...), those that follow from exported ones
    (``VWAD``, ``GWGD``, ``HIAD``) and the layer-resolved ``RL1D``-``RL9D`` and ``SW1D``-``SW9D``, which
    this version does not export. Of ``Summary.OUT`` the identifiers ``R#``, ``O#``, ``P#``, the site and
    weather columns, the other weights and yield components and all water, nitrogen, phosphorus,
    potassium, organic matter and environment columns are not written. Missing values are ``-99``, as
    in DSSAT; whole numbers are rounded to the nearest (halves away from zero), reals fixed-point, as
    DSSAT prints them; a value that does not fit its field is asterisks, as Fortran prints it. The
    numbers are therefore the season's rounded to DSSAT's printed precision, not its full precision
    (the CSV files keep that).

    **Rows.** ``PlantGro.OUT`` starts on the planting day (``DAP`` 0), ``SoilWat.OUT`` on the first
    simulated day (dscsm048 also writes the initial state as ``DAS`` 0, which a season does not have).
    ``DAS`` counts days from the simulation start date, which is ``DAS`` 1 (the facade's seasons start
    on it). ``HDAT`` is the harvest date of the experiment when the season ends on it. The readers
    :func:`agrijax.io.dssat.read_plantgro`, :func:`~agrijax.io.dssat.read_soilwat` and
    :func:`~agrijax.io.dssat.read_summary` read these files as they read DSSAT's.

    The CSV files of :func:`write_csv` remain: the tables with the Agri-JAX extras (cultivar
    coefficients, scenario fields), unrounded."""
    d = Path(directory)
    if isinstance(obj, DssatBatch):
        raise TypeError(
            "a DssatBatch holds DSSAT's own Summary.OUT values (dscsm048 ran it): write_dssat_out writes "
            "Agri-JAX results"
        )
    if _is_table(obj):
        return {"Summary.OUT": _write_lines(d / "Summary.OUT", _batch_out(obj))}
    runs = _out_runs(obj)
    if not per_run:
        return {
            "PlantGro.OUT": _write_lines(d / "PlantGro.OUT", _daily_out("PlantGro.OUT", runs)),
            "SoilWat.OUT": _write_lines(d / "SoilWat.OUT", _daily_out("SoilWat.OUT", runs)),
            "Summary.OUT": _write_lines(d / "Summary.OUT", _season_out(runs)),
        }
    out: dict[str, Path] = {}
    for i, r in enumerate(runs, start=1):
        sub = f"{i}_{r.treatment}"
        for name, lines in (
            ("PlantGro.OUT", _daily_out("PlantGro.OUT", [r])),
            ("SoilWat.OUT", _daily_out("SoilWat.OUT", [r])),
            ("Summary.OUT", _season_out([r])),
        ):
            out[f"{sub}/{name}"] = _write_lines(d / sub / name, lines)
    return out
