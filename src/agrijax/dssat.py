"""DSSAT-CSM maize experiments in a few calls: run a season, batch, calibrate, compare with DSSAT.

::

    import agrijax as aj

    exp = aj.dssat.experiment("UFGA8201")   # DSSAT's example experiment (its public files, cached)
    season = exp.run(treatment=4)           # the Agri-JAX DSSAT-CSM v4.8.6 maize day, one season
    scen = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14])
    batch = scen.run(cultivar={"G2": g2_samples})   # every sample on every scenario, batched
    res = aj.calibrate(exp, treatments=[4], holdout=[6])

    ref = exp.reference(treatment=4)        # the original DSSAT program on the same treatment
    season.compare_daily(ref)               # daily LAI, weights, soil water: differences
    aj.plot.season(season, reference=ref)
    res.check_dssat()                       # the calibrated .CUL row run in DSSAT

**What runs.** Agri-JAX's re-implementation of the DSSAT-CSM v4.8.6.0 maize model
(:mod:`agrijax.models.day_dssat486`: the soil water, evaporation, transpiration, root water uptake
and CERES-Maize), **nitrogen off**, on inputs built from DSSAT's own experiment files without
running DSSAT (:func:`agrijax.sites.dssat_free_run.dssat_free_inputs`: the FileX, ``.SOL``,
``.WTH``, ``.CUL`` / ``.ECO`` / ``.SPE`` and DSSAT's standard data; the season ends at the model's
own maturity). Treatments that need a DSSAT process Agri-JAX does not implement are refused with
the reason (:class:`~agrijax.sites.dssat_free_run.NativeInputError`); a treatment where DSSAT's
yield changes by more than 2 % with nitrogen on runs with a :class:`NitrogenWarning`. The day runs
in float64 (it is validated against DSSAT in float64): these calls switch ``jax_enable_x64`` on.
Opt-in float32 and the choice of device (``run(..., precision="float32", device="gpu")``,
``with aj.options(...)``), which leave your JAX settings as they were, are in
:mod:`agrijax.facade_execution`.

**DSSAT itself** (``dscsm048`` v4.8.6.0) is only the reference to compare with: :meth:`Experiment.reference`,
:meth:`Experiment.dssat_batch`, :meth:`Scenarios.reference` and the calibration's DSSAT check run it,
and get it on first use (:func:`install_reference`: a prebuilt build of the validated version).
DSSAT's data files come from DSSAT's public repositories (:func:`agrijax.io.examples.fetch_dssat_data`,
BSD-3). Runs are staged under ``AGRI_JAX_RUN_ROOT`` (default: the system temporary directory).
"""

from __future__ import annotations

import datetime as _dt
import os
import tempfile
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BatchResult",
    "DssatBatch",
    "DssatNotFoundError",
    "Experiment",
    "NitrogenWarning",
    "Reference",
    "Scenarios",
    "Season",
    "alternatives",
    "calibrate",
    "data",
    "engine",
    "experiment",
    "install_reference",
]

#: the cultivar coefficients of ``MZCER048.CUL`` (column order)
CULTIVAR = ("P1", "P2", "P5", "G2", "G3", "PHINT")
#: the daily series of a season (column: meaning [unit]); the same names in :attr:`Reference.daily`
DAILY = {
    "lai": "leaf area index [m2 m-2]",
    "cwad": "tops weight [kg ha-1]",
    "gwad": "grain weight [kg ha-1]",
    "swtd": "soil water in the profile [mm]",
}
#: the DSSAT output columns of the daily series (``PlantGro.OUT``, ``SoilWat.OUT``)
_DSSAT_DAILY = {"lai": "LAID", "cwad": "CWAD", "gwad": "GWAD", "swtd": "SWTD"}
#: the end-of-season values (``Summary.OUT`` names): silking and maturity dates (``YYYYDDD``), grain
#: yield and tops weight at maturity [kg ha-1]
SUMMARY = ("ADAT", "MDAT", "HWAM", "CWAM")
#: stage codes (CERES-Maize ``ISTAGE`` output) of the first day of silking and of maturity
_SILKING, _MATURITY = 4, 10
#: days simulated past the reference season when the cultivar changes (a later-maturing candidate)
PAD_DAYS = 60


class NitrogenWarning(UserWarning):
    """The model runs nitrogen off, and on this treatment DSSAT's yield changes with nitrogen on."""


class DssatNotFoundError(FileNotFoundError):
    """No ``dscsm048`` where one was asked for: :func:`install_reference` gets the reference program."""


def install_reference(
    root: str | os.PathLike[str] | None = None, *, build: bool = False, verbose: bool = True
) -> Path:
    """Get the DSSAT-CSM v4.8.6.0 reference program ``dscsm048`` to compare with
    (:func:`agrijax.io.examples.install_reference`: the prebuilt validated build, or built from
    source with ``build=True``); Agri-JAX's own runs do not need it."""
    from agrijax.io import examples

    return examples.install_reference(root, build=build, verbose=verbose)


def data(root: str | os.PathLike[str] | None = None) -> Path:
    """The root of DSSAT's public data files (:func:`agrijax.io.examples.fetch_dssat_data`: the example
    experiments, weather and soil, the maize genotype and standard data; downloaded once)."""
    from agrijax.io import examples

    return examples.fetch_dssat_data(root)


def _x64() -> None:
    import jax

    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


def engine(root: str | os.PathLike[str] | None = None) -> Path:
    """The root of the DSSAT reference program: ``root`` (must hold ``dscsm048``, else
    :class:`DssatNotFoundError`), else ``AGRI_JAX_DSSAT`` when it holds one, else
    :func:`install_reference` (downloads the prebuilt reference program on first use)."""
    from agrijax.port import run_fortran

    if root is not None:
        exe, _ = run_fortran.dscsm_paths(Path(root))
        if not exe.is_file():
            raise DssatNotFoundError(
                f"dscsm048 not found at {exe}: call agrijax.dssat.install_reference() to get the DSSAT "
                "reference program, or pass the root of an engine that has it"
            )
        return Path(root)
    eng = Path(os.environ.get("AGRI_JAX_DSSAT") or run_fortran.DSSAT_ENGINE)
    if run_fortran.dscsm_paths(eng)[0].is_file():
        return eng
    return install_reference()


def _work_dir(prefix: str) -> Path:
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=root))


def _yrdoy_add(yrdoy: int, days: int) -> int:
    d = _dt.date(yrdoy // 1000, 1, 1) + _dt.timedelta(days=yrdoy % 1000 - 1 + int(days))
    return d.year * 1000 + d.timetuple().tm_yday


def _dates(yrdoy: Sequence[float]) -> Any:
    import pandas as pd

    return pd.to_datetime([f"{int(d)}" if np.isfinite(d) else None for d in yrdoy], format="%Y%j")


def _first(stage: np.ndarray, code: int) -> int | None:
    hit = np.nonzero(stage == code)[0]
    return int(hit[0]) if hit.size else None


# ------------------------------------------------------------------ one season
@dataclass
class Season:
    """One season simulated by Agri-JAX: :attr:`daily` (one row per day: ``date``, ``yrdoy`` and the
    series of :data:`DAILY`), :attr:`summary` (:data:`SUMMARY`), the cultivar used and the inputs."""

    treatment: str
    daily: Any
    summary: dict[str, float]
    cultivar: dict[str, float]
    inputs: Any = field(repr=False)
    outputs: dict[str, np.ndarray] = field(repr=False)
    #: the precision the season was simulated in (``"float64"``, or ``"float32"`` when asked for)
    precision: str = "float64"
    #: the experiment's own names, for the run header of the DSSAT-format files
    #: (:meth:`write_dssat_out`): ``"experiment"`` (the ``*EXP.DETAILS`` text) and ``"treatment"`` (TNAME)
    labels: dict[str, str] = field(default_factory=dict, repr=False)

    def compare_daily(self, reference: Reference) -> Any:
        """Daily differences to ``reference`` (the DSSAT run) on the days both report, per series:
        days compared, RMSE, largest absolute difference, the reference's largest value and the
        RMSE relative to it (pandas)."""
        import pandas as pd

        m = self.daily.merge(reference.daily, on="yrdoy", suffixes=("", "_dssat"))
        rows = []
        for k, what in DAILY.items():
            a, b = m[k].to_numpy(float), m[f"{k}_dssat"].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]
            top = float(np.max(np.abs(b[ok]))) if ok.any() else float("nan")
            rmse = float(np.sqrt(np.mean(d * d))) if ok.any() else float("nan")
            rows.append(
                {
                    "series": k,
                    "meaning": what,
                    "days": int(ok.sum()),
                    "RMSE": rmse,
                    "max |diff|": float(np.max(np.abs(d))) if ok.any() else float("nan"),
                    "DSSAT max": top,
                    "RMSE / max": rmse / top if top else float("nan"),
                }
            )
        return pd.DataFrame(rows).set_index("series")

    def compare_summary(self, reference: Reference) -> Any:
        """End-of-season values of Agri-JAX and DSSAT (dates as ``YYYYDDD``; the difference is in days
        for the dates, relative for the weights) (pandas)."""
        import pandas as pd

        rows = []
        for k in SUMMARY:
            a, b = float(self.summary.get(k, np.nan)), float(reference.summary.get(k, np.nan))
            both = np.isfinite(a) and np.isfinite(b)
            if k in ("ADAT", "MDAT"):
                diff = float((_dates([a])[0] - _dates([b])[0]).days) if both else np.nan
            else:
                diff = (a - b) / b if both and b else np.nan
            if k in ("ADAT", "MDAT"):  # dates stay YYYYDDD integers in the table
                a, b = (int(v) if np.isfinite(v) else None for v in (a, b))
            rows.append({"value": k, "Agri-JAX": a, "DSSAT": b, "difference": diff})
        return pd.DataFrame(rows, dtype=object).set_index("value")

    def plot(self, reference: Reference | None = None, **kw: Any) -> Any:
        """:func:`agrijax.report.plot.season`."""
        from agrijax.report import plot

        return plot.season(self, reference=reference, **kw)

    def to_frame(self, *, full: bool = False, dssat_names: bool = False) -> Any:
        """:func:`agrijax.facade_export.to_frame`: the daily table (pandas); ``full=True`` adds the other
        ``PlantGro.OUT`` quantities, ``dssat_names=True`` the DSSAT column names."""
        from agrijax import facade_export

        return facade_export.to_frame(self, full=full, dssat_names=dssat_names)

    def to_pandas(self, *, full: bool = False, dssat_names: bool = False) -> Any:
        """:meth:`to_frame` (the same table)."""
        return self.to_frame(full=full, dssat_names=dssat_names)

    def to_xarray(self, *, full: bool = False) -> Any:
        """:func:`agrijax.facade_export.to_xarray`: an ``xarray.Dataset`` with the dimensions ``season``
        (one) x ``day``."""
        from agrijax import facade_export

        return facade_export.to_xarray(self, full=full)

    def write_csv(
        self, directory: str | os.PathLike[str], *, name: str | None = None, full: bool = True
    ) -> dict[str, Path]:
        """:func:`agrijax.facade_export.write_csv`: the daily and the summary CSV files in ``directory``
        (DSSAT's column names; these CSV files are not DSSAT-format ``.OUT`` files, which
        :meth:`write_dssat_out` writes)."""
        from agrijax import facade_export

        return facade_export.write_csv(self, directory, name=name, full=full)

    def write_dssat_out(self, directory: str | os.PathLike[str]) -> dict[str, Path]:
        """:func:`agrijax.facade_export.write_dssat_out`: ``PlantGro.OUT``, ``SoilWat.OUT`` and
        ``Summary.OUT`` of this season in ``directory``, in DSSAT-CSM v4.8.6's fixed-width layout and
        with the columns Agri-JAX simulates. The files say in their header that Agri-JAX wrote them and
        that the DSSAT program did not."""
        from agrijax import facade_export

        return facade_export.write_dssat_out(self, directory)


@dataclass
class Reference:
    """One ``dscsm048`` run (nitrogen off unless asked): :attr:`daily` (same columns as
    :attr:`Season.daily`, from ``PlantGro.OUT`` and ``SoilWat.OUT``), :attr:`summary`
    (``Summary.OUT``), the output directory and the wall time of the run (staging included)."""

    treatment: str
    daily: Any
    summary: dict[str, float]
    out: Path
    elapsed_s: float
    cultivar: dict[str, float] | None = None


def _reference_daily(out: Path, trno: int | None = None) -> Any:
    from agrijax.io.dssat import read_plantgro, read_soilwat

    pg, sw = read_plantgro(out / "PlantGro.OUT"), read_soilwat(out / "SoilWat.OUT")
    if trno is not None:
        pg = pg[pg["TRNO"] == trno] if "TRNO" in pg.columns else pg
        sw = sw[sw["TRNO"] == trno] if "TRNO" in sw.columns else sw
    key = lambda df: (df["YEAR"].astype(int) * 1000 + df["DOY"].astype(int)).to_numpy()  # noqa: E731
    a = pg.assign(yrdoy=key(pg))[["yrdoy", "LAID", "CWAD", "GWAD"]]
    b = sw.assign(yrdoy=key(sw))[["yrdoy", "SWTD"]]
    d = a.merge(b, on="yrdoy", how="outer").sort_values("yrdoy").reset_index(drop=True)
    d = d.rename(columns={v: k for k, v in _DSSAT_DAILY.items()})
    d.insert(0, "date", _dates(d["yrdoy"].tolist()))
    return d


def _summary_row(row: Any) -> dict[str, float]:
    return {k: float(row[k]) for k in (*SUMMARY, "SDAT", "PDAT", "EDAT", "HDAT") if k in row.index}


# ------------------------------------------------------------------ batches
@dataclass
class BatchResult:
    """A batch of seasons: :attr:`table` (one row per season: the scenario, the cultivar sample and
    :data:`SUMMARY`) and :attr:`timing` (``compile_s``: compiling the batched program, 0 when it was
    compiled before; ``run_s``: running it, results on the host; ``seasons``; ``devices``)."""

    table: Any
    timing: dict[str, Any]
    #: the treatment key the batch ran (``UFGA8201_t04``), for the run identifiers of
    #: :meth:`write_dssat_out`; empty when not known
    treatment: str = ""

    def __repr__(self) -> str:
        t = self.timing
        return (
            f"BatchResult({t['seasons']} seasons; compile {t['compile_s']:.2f} s, run {t['run_s']:.3f} s "
            f"= {1e3 * t['run_s'] / max(t['seasons'], 1):.3f} ms per season on {t['devices']})"
        )

    def to_frame(self) -> Any:
        """:func:`agrijax.facade_export.to_frame`: :attr:`table` as a pandas table (a copy)."""
        from agrijax import facade_export

        return facade_export.to_frame(self)

    def to_pandas(self) -> Any:
        """:meth:`to_frame` (the same table)."""
        return self.to_frame()

    def to_xarray(self) -> Any:
        """:func:`agrijax.facade_export.to_xarray`: an ``xarray.Dataset`` with the dimension ``season`` and
        the scenario fields as coordinates (a batch keeps end-of-season values, no daily series)."""
        from agrijax import facade_export

        return facade_export.to_xarray(self)

    def write_csv(self, directory: str | os.PathLike[str], *, name: str | None = None) -> dict[str, Path]:
        """:func:`agrijax.facade_export.write_csv`: the summary CSV file in ``directory`` (DSSAT's column
        names where they map; this CSV file is not a DSSAT-format ``.OUT`` file, which
        :meth:`write_dssat_out` writes)."""
        from agrijax import facade_export

        return facade_export.write_csv(self, directory, name=name)

    def write_dssat_out(self, directory: str | os.PathLike[str]) -> dict[str, Path]:
        """:func:`agrijax.facade_export.write_dssat_out`: ``Summary.OUT`` in ``directory`` in DSSAT-CSM
        v4.8.6's fixed-width layout, one run (row) per season of the batch. A batch keeps no daily
        series, so it has no ``PlantGro.OUT`` or ``SoilWat.OUT``."""
        from agrijax import facade_export

        return facade_export.write_dssat_out(self, directory)


def _samples(cultivar: Any, published: Mapping[str, float]) -> np.ndarray:
    """``[K, 6]`` cultivar samples from ``None`` (the published one), a mapping of arrays / values or
    a DataFrame (missing coefficients at the published values)."""
    if cultivar is None:
        return np.asarray([[published[n] for n in CULTIVAR]])
    cols = {str(k): np.atleast_1d(np.asarray(cultivar[k], dtype=float)) for k in cultivar}
    bad = sorted(set(cols) - set(CULTIVAR))
    if bad:
        raise ValueError(f"unknown cultivar coefficients {bad}; known: {list(CULTIVAR)}")
    k = max(len(v) for v in cols.values())
    if any(len(v) not in (1, k) for v in cols.values()):
        raise ValueError(f"cultivar samples of different lengths: { {n: len(v) for n, v in cols.items()} }")
    return np.stack([np.broadcast_to(cols.get(n, np.asarray([published[n]])), (k,)) for n in CULTIVAR], 1)


@dataclass
class Scenarios:
    """One treatment on several weather years and sowing dates (:meth:`Experiment.scenarios`):
    :attr:`table` (one row per scenario: year, sowing shift, planting date and Agri-JAX's season at
    the published cultivar, :data:`SUMMARY`) and the inputs of each. :meth:`run` simulates cultivar
    samples on all of them in one batched program (compiled on the first call); :meth:`reference`
    runs DSSAT on each scenario to compare with."""

    treatment: str
    table: Any
    runs: list[Any] = field(repr=False)
    published: dict[str, float]
    filex: list[Path] = field(default_factory=list, repr=False)
    #: wall time of each DSSAT run of :meth:`reference` [s]
    dssat_s: list[float] = field(default_factory=list)
    _exp: Any = field(default=None, repr=False)
    _sim: Any = field(default=None, repr=False)

    def reference(self) -> Any:
        """``dscsm048`` on every scenario (published cultivar, nitrogen off, one process per scenario;
        the wall time of each run, staging and DSSAT's output files included, in :attr:`dssat_s`):
        one row per scenario with DSSAT's :data:`SUMMARY` (pandas)."""
        import pandas as pd

        from agrijax.io.dssat import read_summary

        exp = self._exp
        trno = self.runs[0].trno
        rows, secs = [], []
        for i, fx in enumerate(self.filex):
            out, sec = exp._run_ref(trno, f"scn_ref_{i}", filex=fx)
            row = read_summary(out / "Summary.OUT").iloc[0]
            secs.append(sec)
            rows.append(
                {
                    "year": int(self.table.loc[i, "year"]),
                    "sowing_shift": int(self.table.loc[i, "sowing_shift"]),
                    **{k: float(row[k]) for k in SUMMARY},
                    "seconds": sec,
                }
            )
        self.dssat_s = secs
        return pd.DataFrame(rows)

    def run(
        self, cultivar: Any = None, *, precision: str | None = None, device: str | None = None
    ) -> BatchResult:
        """Every cultivar sample on every scenario (``cultivar``: ``None`` = the published one; a
        mapping ``{"G2": [...], ...}`` of equal-length samples, coefficients left out at the
        published value; or a DataFrame with those columns).

        ``precision`` (``"float64"``, the validated default, or ``"float32"``) and ``device``
        (``"cpu"``, ``"gpu"``, ``"gpu:<index>"``; default JAX's own) are the execution options of
        :func:`agrijax.options`: the compiled programs are kept per option, and your JAX settings are
        put back after the call (accuracy of float32: :mod:`agrijax.facade_execution`)."""
        if precision is not None or device is not None:
            from agrijax.facade_execution import options

            with options(precision, device):
                return self.run(cultivar)
        import pandas as pd

        from agrijax.calib.dssat_day import E_DATE, E_FINAL, OUT_NAMES, DaySimulator, Entry
        from agrijax.facade_execution import active, simulator

        _x64()
        t0 = time.perf_counter()
        theta = _samples(cultivar, self.published)
        k, n_s = theta.shape[0], len(self.runs)
        ents = [
            Entry(E_DATE, _SILKING),
            Entry(E_DATE, _MATURITY),
            Entry(E_FINAL, OUT_NAMES.index("gwad")),
            Entry(E_FINAL, OUT_NAMES.index("cwad")),
        ]
        sim = simulator(self, lambda **kw: DaySimulator(self.runs, [ents] * n_s, **kw))
        s0 = sim.stats()
        tid = np.repeat(np.arange(n_s), k)
        y = sim(np.tile(theta, (n_s, 1)), tid)
        s1 = sim.stats()
        pad = {i: sim.n_days[sim.where[i][0]] for i in range(n_s)}
        rows = []
        for j in range(y.shape[0]):
            i = int(tid[j])
            r = self.runs[i]
            ad, md = (int(v) for v in y[j, :2])
            rows.append(
                {
                    "scenario": i,
                    "year": int(self.table.loc[i, "year"]),
                    "sowing_shift": int(self.table.loc[i, "sowing_shift"]),
                    "sample": j % k,
                    **dict(zip(CULTIVAR, theta[j % k].tolist(), strict=True)),
                    "ADAT": _yrdoy_add(int(r.days[0]), ad) if ad < pad[i] else np.nan,
                    "MDAT": _yrdoy_add(int(r.days[0]), md) if md < pad[i] else np.nan,
                    "HWAM": float(y[j, 2]),
                    "CWAM": float(y[j, 3]),
                }
            )
        timing = {
            "compile_s": s1["compile_s"] - s0["compile_s"],
            "run_s": s1["call_s"] - s0["call_s"],
            "wall_s": time.perf_counter() - t0,
            "seasons": int(y.shape[0]),
            "devices": f"{sim.ndev} x {sim.devs[0].device_kind}",
            "precision": active().precision,
            "mesev": str(getattr(self.runs[0], "mesev", "")),  # the soil evaporation (a swap shows here)
        }
        return BatchResult(pd.DataFrame(rows), timing, self.treatment)

    def sensitivity(
        self,
        outputs: Sequence[str] | str = ("HWAM", "CWAM"),
        params: Sequence[str] | str = CULTIVAR,
        *,
        cultivar: Mapping[str, float] | None = None,
        scan_points: int = 201,
    ) -> Any:
        """The derivative of the end values (``"HWAM"``, ``"CWAM"``, ``"H#AM"``, or ``"LAI@60"``-style: the
        value 60 days after planting) with respect to the cultivar coefficients ``params``, on every
        scenario at the published cultivar (or ``cultivar``), in one batched program: a
        :class:`~agrijax.facade_grad.Sensitivity` (``.table`` per scenario, ``.summary`` over them).
        Every coefficient carries a trust label (``validated gradient`` / ``falls back`` /
        ``experimental``); derivatives through phenology events (``P1``, ``P2``, ``P5``, ``PHINT``)
        are experimental. Details: :mod:`agrijax.facade_grad`."""
        from agrijax.facade_grad import sensitivity

        return sensitivity(self, outputs=outputs, params=params, cultivar=cultivar, scan_points=scan_points)

    def weather_sensitivity(
        self,
        outputs: Sequence[str] | str = ("HWAM",),
        variables: Sequence[str] | str = ("SRAD", "TMAX", "TMIN", "RAIN"),
        *,
        check: bool = True,
        config: Any = None,
    ) -> Any:
        """The derivative of the end values with respect to every day's weather (``"SRAD"``, ``"TMAX"``,
        ``"TMIN"``, ``"RAIN"``, and ``"IRRD"`` the irrigation) on every scenario at the published cultivar:
        one reverse-mode pass over all of them, the trust checks batched. A
        :class:`~agrijax.facade_weather.WeatherSensitivities` (one
        :class:`~agrijax.facade_weather.WeatherSensitivity` per scenario, indexable by year). Temperature
        derivatives keep the growth-stage calendar fixed. Details: :mod:`agrijax.facade_weather`."""
        from agrijax import facade_weather as fw

        exp = self._exp
        files = (
            fw.station_files(exp.data / "example_data" / "Weather", exp.station) if exp is not None else []
        )
        names = [
            f"{self.treatment} {int(self.table.loc[i, 'year'])} {int(self.table.loc[i, 'sowing_shift']):+d} d"
            for i in range(len(self.runs))
        ]
        res = fw.WeatherSensitivities(
            fw.weather_sensitivity(
                self.runs,
                names,
                outputs=outputs,
                variables=variables,
                check=check,
                config=config,
                weather_files=files,
                owner=self,
                key=("weather",),
            )
        )
        res.years = [int(y) for y in self.table["year"]]
        return res


@dataclass
class DssatBatch:
    """Cultivar samples of one treatment run by ``dscsm048`` in one batch run
    (:meth:`Experiment.dssat_batch`): :attr:`table` (the coefficients as written to the ``.CUL``
    file and ``Summary.OUT``'s :data:`SUMMARY`), the wall time of the run (staging the run directory
    and writing DSSAT's daily output files included) and the seasons run."""

    treatment: str
    table: Any
    elapsed_s: float
    seasons: int

    @property
    def cultivar(self) -> dict[str, np.ndarray]:
        """The written coefficients, as a ``cultivar=`` argument (:meth:`Scenarios.run`)."""
        return {n: self.table[n].to_numpy(float) for n in CULTIVAR}


# ------------------------------------------------------------------ the experiment
@dataclass
class Experiment:
    """A DSSAT maize experiment (:func:`experiment`): its treatments, cultivars, soil and weather
    station; an ``os.PathLike`` of its ``.MZX`` file (so it can be passed where a path is expected,
    e.g. :func:`agrijax.calibrate`)."""

    name: str
    filex: Path
    #: the root of DSSAT's data the experiment's runs read (``example_data``, ``Data``)
    data: Path
    title: str
    treatments: dict[int, str]
    cultivars: dict[int, str]
    soil: str
    station: str
    _engine: Path | None = field(default=None, repr=False)
    _work: Path | None = field(default=None, repr=False)
    _refs: dict[Any, Reference] = field(default_factory=dict, repr=False)
    _inputs: dict[Any, Any] = field(default_factory=dict, repr=False)
    _warned: set[int] = field(default_factory=set, repr=False)

    def __fspath__(self) -> str:
        return str(self.filex)

    @property
    def engine(self) -> Path:
        """The DSSAT reference program's root (:func:`engine`: fetched on first use)."""
        if self._engine is None:
            self._engine = engine()
        return self._engine

    def __repr__(self) -> str:
        return f"Experiment({self.name}: {self.title}; {len(self.treatments)} treatments)"

    def table(self) -> Any:
        """One row per treatment: name, cultivar, whether the native inputs support it (else why)
        and DSSAT's measured yield change between nitrogen on and off (the model runs nitrogen off)."""
        import pandas as pd

        from agrijax.calib.workflow import NITROGEN_STRESS
        from agrijax.io.dssat import read_filex
        from agrijax.io.dssat.native_management import unsupported_features

        x = read_filex(self.filex)
        rows = []
        for n, tname in self.treatments.items():
            bad = unsupported_features(x, n)
            ns = NITROGEN_STRESS.get(f"{self.name}_t{n:02d}")
            rows.append(
                {
                    "treatment": n,
                    "name": tname,
                    "cultivar": self.cultivars[n],
                    "native inputs": "yes" if not bad else "; ".join(bad),
                    "yield change with nitrogen on": f"{ns:+.1%}" if ns is not None else "not measured",
                }
            )
        return pd.DataFrame(rows).set_index("treatment")

    def observed(self, treatment: int) -> Any:
        """The treatment's observed time series (the experiment's T file) of the daily series of
        :data:`DAILY` it holds (``LAID``, ``CWAD``, ``GWAD``): one row per observed date (pandas)."""
        import pandas as pd

        from agrijax.io.dssat.observed import read_observed

        trno = self._trno(treatment)
        ft = read_observed(self.filex).filet
        out = pd.DataFrame({"yrdoy": pd.Series(dtype=int)})
        if ft is not None:
            for k, code in _DSSAT_DAILY.items():
                if code in ft.codes:
                    ser = ft.series(trno, code)
                    s = pd.DataFrame({"yrdoy": ser["YRDOY"].to_numpy(int), k: ser["value"].to_numpy(float)})
                    out = out.merge(s, on="yrdoy", how="outer")
        out = out.sort_values("yrdoy").reset_index(drop=True)
        out.insert(0, "date", _dates(out["yrdoy"].tolist()))
        return out

    # -------------------------------------------------------------- internals
    def _dir(self, tag: str) -> Path:
        if self._work is None:
            self._work = _work_dir(f"aj_{self.name}_")
        d = self._work / tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _trno(self, treatment: int) -> int:
        t = int(treatment)
        if t not in self.treatments:
            raise ValueError(f"{self.name}: no treatment {t} (treatments {sorted(self.treatments)})")
        return t

    def _nitrogen_check(self, trno: int) -> None:
        """A :class:`NitrogenWarning` (once per treatment) when DSSAT's yield of ``trno`` changes by
        more than :data:`~agrijax.calib.workflow.NITROGEN_STRESS_OK` with nitrogen on."""
        from agrijax.calib.workflow import NITROGEN_STRESS, NITROGEN_STRESS_OK

        d = NITROGEN_STRESS.get(f"{self.name}_t{trno:02d}")
        if d is None or abs(d) <= NITROGEN_STRESS_OK or trno in self._warned:
            return
        self._warned.add(trno)
        warnings.warn(
            f"{self.name} treatment {trno}: the model runs nitrogen off, and DSSAT's yield on this "
            f"treatment changes by {d:+.1%} with nitrogen on; its simulated yield is the "
            "nitrogen-unlimited one (exp.table() lists the treatments where nitrogen matters)",
            NitrogenWarning,
            stacklevel=4,  # _nitrogen_check <- inputs <- run / scenarios <- the caller
        )

    def _run_ref(
        self, trno: int | Sequence[int], tag: str, *, filex: Path | None = None, **kw: Any
    ) -> tuple[Path, float]:
        from agrijax.sites.dssat_free_run import run_reference

        eng = self.engine
        ex = self.data / "example_data"
        t0 = time.perf_counter()
        out = run_reference(
            filex or self.filex,
            trno,
            self._dir(tag),
            engine=eng,
            weather_dir=ex / "Weather",
            soil_dir=ex / "Soil",
            **kw,
        )
        return out, time.perf_counter() - t0

    # -------------------------------------------------------------- DSSAT runs
    def reference(
        self,
        treatment: int,
        *,
        cultivar: Mapping[str, float] | None = None,
        nitrogen: bool = False,
        soil_evaporation: str | None = None,
    ) -> Reference:
        """``dscsm048`` on ``treatment`` (nitrogen off unless ``nitrogen=True``), with the published
        cultivar or the coefficients ``cultivar`` (written to a copy of ``MZCER048.CUL``; the values
        DSSAT reads, at the file's printed precision, are in :attr:`Reference.cultivar`).
        ``soil_evaporation`` runs DSSAT with the other ``MESEV`` (``"ritchie"`` or ``"salus"``,
        :func:`alternatives`), the swap :meth:`run` makes."""
        from agrijax import facade_swap
        from agrijax.io.dssat import read_summary

        trno = self._trno(treatment)
        swap = None if soil_evaporation is None else facade_swap.swap_code(self, trno, soil_evaporation)
        key = (trno, nitrogen, tuple(sorted((cultivar or {}).items())), swap)
        if key in self._refs:
            return self._refs[key]
        tag = f"ref_t{trno:02d}_{'n' if nitrogen else 'o'}_{len(self._refs)}"
        written = None
        kw: dict[str, Any] = {"nitrogen": nitrogen}
        if swap is not None:
            kw["filex_edit"] = facade_swap.filex_edit(self, trno, swap)
        if cultivar:
            ids, cul, table = self._cul_rows(trno, [dict(cultivar)], tag)
            kw.update(cultivar=(self.cultivars[trno], ids[0]), cul_file=cul)
            written = {n: float(table[0][n]) for n in CULTIVAR}
        out, secs = self._run_ref(trno, tag, **kw)
        row = read_summary(out / "Summary.OUT").iloc[0]
        ref = Reference(
            f"{self.name}_t{trno:02d}", _reference_daily(out), _summary_row(row), out, secs, written
        )
        self._refs[key] = ref
        return ref

    def _cul_rows(
        self, trno: int, samples: Sequence[Mapping[str, float]], tag: str
    ) -> tuple[list[str], Path, list[dict[str, float]]]:
        """A copy of the engine's ``MZCER048.CUL`` with one row per sample (ids ``AJ0001`` ...)."""
        from agrijax.io.dssat.cultivar_write import write_cultivar
        from agrijax.sites.dssat_free_run import dssat_data_dir

        src = dssat_data_dir(self.data) / "Genotype" / "MZCER048.CUL"
        base = self.cultivars[trno]
        d = self._dir(tag)
        ids, table = [], []
        cur = src
        for i, s in enumerate(samples, start=1):
            cid = f"AJ{i:04d}"
            dest = d / f"cul_{i % 2}" / "MZCER048.CUL"
            w = write_cultivar(cur, dest, cid, f"AgriJAX {base}"[:16], dict(s), base=base, overwrite=True)
            ids.append(cid)
            table.append(dict(w.written))
            cur = dest
        final = d / "cul" / "MZCER048.CUL"
        final.parent.mkdir(exist_ok=True)
        final.write_bytes(cur.read_bytes())
        return ids, final, table

    def dssat_batch(self, treatment: int, cultivar: Any, soil_evaporation: str | None = None) -> DssatBatch:
        """``dscsm048`` on ``treatment`` for each cultivar sample (``cultivar`` as in
        :meth:`Scenarios.run`, at most 99 samples), in one batch run (one process): the per-season
        cost of DSSAT itself on this machine, and its values for the same samples.
        ``soil_evaporation`` runs DSSAT with the other ``MESEV`` (:meth:`reference`)."""
        import pandas as pd

        from agrijax import facade_swap
        from agrijax.io.dssat import read_summary
        from agrijax.sites.dssat_free_run import cultivar_batch_filex

        trno = self._trno(treatment)
        swap = facade_swap.filex_edit(self, trno, soil_evaporation)
        pub = self.inputs(trno).published()
        theta = _samples(cultivar, pub)
        tag = f"dssat_batch_{len(self._refs)}_{time.perf_counter_ns()}"
        ids, cul, written = self._cul_rows(trno, [dict(zip(CULTIVAR, r, strict=True)) for r in theta], tag)
        k = len(ids)
        out, secs = self._run_ref(
            list(range(1, k + 1)),
            tag,
            cul_file=cul,
            filex_edit=lambda t: cultivar_batch_filex(swap(t) if swap else t, trno, ids),
        )
        s = read_summary(out / "Summary.OUT").sort_values("TRNO")
        rows = [{"sample": i, **written[i], **{c: float(s.iloc[i][c]) for c in SUMMARY}} for i in range(k)]
        return DssatBatch(f"{self.name}_t{trno:02d}", pd.DataFrame(rows), secs, k)

    # -------------------------------------------------------------- Agri-JAX runs
    def inputs(self, treatment: int, soil_evaporation: str | None = None) -> Any:
        """The treatment's inputs (:class:`~agrijax.sites.dssat_free_run.FreeRunInputs`), built from
        DSSAT's files without running DSSAT, with :data:`PAD_DAYS` days of real weather after the
        season. ``soil_evaporation`` swaps the soil evaporation (one of :func:`alternatives`; default
        the experiment's own): the inputs are built for the swapped model, whose season ends at its own
        maturity."""
        trno = self._trno(treatment)
        self._nitrogen_check(trno)
        if soil_evaporation is not None:
            from agrijax import facade_swap

            return facade_swap.inputs(self, trno, soil_evaporation)
        if trno not in self._inputs:
            self._inputs[trno] = self._build(self.filex, trno)
        return self._inputs[trno]

    def _build(self, filex: Path, trno: int) -> Any:
        from agrijax.sites.dssat_free_run import free_run_inputs

        _x64()
        return free_run_inputs(
            filex,
            trno,
            None,
            engine=self.data,
            source="native",
            soil_dirs=[filex.parent, self.data / "example_data" / "Soil"],
            extend_days=PAD_DAYS,
        )

    def run(
        self,
        treatment: int,
        *,
        cultivar: Mapping[str, float] | None = None,
        soil_evaporation: str | None = None,
        precision: str | None = None,
        device: str | None = None,
    ) -> Season:
        """One season of ``treatment`` simulated by Agri-JAX (published cultivar, or the
        coefficients ``cultivar`` replaced: the season then runs up to :data:`PAD_DAYS` days past the
        reference season and stops at maturity). ``soil_evaporation`` swaps the soil evaporation
        (``"ritchie"`` or ``"salus"``, :func:`alternatives`; default the experiment's own, DSSAT's
        ``MESEV``): the season is then the swapped model's, ending at its own maturity.

        ``precision`` (``"float64"``, the validated default, or ``"float32"``) and ``device``
        (``"cpu"``, ``"gpu"``, ``"gpu:<index>"``; default JAX's own) are the execution options of
        :func:`agrijax.options`; your JAX settings are put back after the call (accuracy of float32:
        :mod:`agrijax.facade_execution`; :attr:`Season.precision` records what ran)."""
        if precision is not None or device is not None:
            from agrijax.facade_execution import options

            with options(precision, device):
                return self.run(treatment, cultivar=cultivar, soil_evaporation=soil_evaporation)
        import pandas as pd

        from agrijax.core.units import MM_PER_CM
        from agrijax.facade_execution import active, simulate
        from agrijax.sites.dssat_free_run import _pad

        _x64()
        x = self.inputs(treatment, soil_evaporation)
        pad = PAD_DAYS if cultivar else 0
        o = simulate(x, cultivar, pad_days=pad)
        n = x.n_days + pad
        days = [int(d) for d in x.days] + [_yrdoy_add(int(x.days[-1]), i) for i in range(1, pad + 1)]
        stage = np.asarray(o["istage"])[:, 0]
        mat = _first(stage, _MATURITY)
        silk = _first(stage, _SILKING)
        dl = _pad(np.asarray(x.series["dlayr_end"], dtype=float), n)
        swtd = np.sum(np.asarray(o["soil_sw"])[:, : x.nl] * dl, axis=-1) * MM_PER_CM
        last = n if (mat is None or not cultivar) else mat + 1
        daily = pd.DataFrame(
            {
                "date": _dates(days[:last]),
                "yrdoy": days[:last],
                "lai": np.asarray(o["lai"])[:last, 0],
                "cwad": np.asarray(o["cwad"])[:last, 0],
                "gwad": np.asarray(o["gwad"])[:last, 0],
                "swtd": swtd[:last],
            }
        )
        t_end = mat if mat is not None else n - 1
        summary = {
            "ADAT": float(days[silk]) if silk is not None else np.nan,
            "MDAT": float(days[mat]) if mat is not None else np.nan,
            "HWAM": float(np.asarray(o["gwad"])[t_end, 0]),
            "CWAM": float(np.asarray(o["cwad"])[t_end, 0]),
        }
        cul = {**x.published(), **dict(cultivar or {})}
        labels = {"experiment": self.title, "treatment": self.treatments.get(int(x.trno), "")}
        return Season(x.key, daily, summary, cul, x, o, active().precision, labels)

    def scenarios(
        self,
        treatment: int,
        *,
        years: Iterable[int] | None = None,
        sowing_shift: Iterable[int] = (0,),
        soil_evaporation: str | None = None,
    ) -> Scenarios:
        """``treatment`` on the weather ``years`` (default: the experiment's own) and with the whole
        management calendar moved by each of ``sowing_shift`` days (the FileX dates,
        :func:`~agrijax.sites.dssat_free_run.shift_filex_dates`), each built from the shifted FileX
        without running DSSAT. These scenarios are not in the validation set: compare them with DSSAT
        through :meth:`Scenarios.reference`. ``soil_evaporation`` swaps the soil evaporation of every
        scenario (``"ritchie"`` or ``"salus"``, :func:`alternatives`; :meth:`Scenarios.reference` runs
        DSSAT on the same swap)."""
        import pandas as pd

        from agrijax import facade_swap
        from agrijax.sites.dssat_free_run import shift_filex_dates

        _x64()
        trno = self._trno(treatment)
        swap = facade_swap.swap_code(self, trno, soil_evaporation)
        text = facade_swap.swapped_text(self, trno, swap)
        base = self.inputs(trno)
        y0 = int(base.days[0]) // 1000
        ys = [y0] if years is None else [int(y) for y in years]
        wdir = self.data / "example_data" / "Weather"
        miss = [y for y in ys if not (wdir / f"{self.station}{y % 100:02d}01.WTH").is_file()]
        if miss:
            raise ValueError(f"no weather file {self.station}YY01.WTH in {wdir} for the years {miss}")
        rows, runs, files = [], [], []
        tag = f"_{swap}" if swap else ""  # the swapped files do not overwrite the experiment's own
        for y in ys:
            for sh in sowing_shift:
                fx = self._dir(f"scn_t{trno:02d}_{y}_{int(sh):+d}{tag}") / self.filex.name
                fx.write_text(shift_filex_dates(text, years=y - y0, days=int(sh)))
                r = self._build(fx, trno)
                runs.append(r)
                files.append(fx)
                rows.append(
                    {
                        "year": y,
                        "sowing_shift": int(sh),
                        "PDAT": int(np.asarray(r.params_crop.yrplt)),
                        **{k: r.summary[k] for k in SUMMARY},
                    }
                )
        return Scenarios(
            f"{self.name}_t{trno:02d}", pd.DataFrame(rows), runs, base.published(), files, _exp=self
        )

    def run_batch(
        self,
        treatment: int,
        cultivar: Any = None,
        *,
        years: Iterable[int] | None = None,
        sowing_shift: Iterable[int] = (0,),
        soil_evaporation: str | None = None,
        precision: str | None = None,
        device: str | None = None,
    ) -> BatchResult:
        """:meth:`scenarios` then :meth:`Scenarios.run` (one call; keep the scenarios to rerun
        without recompiling); ``soil_evaporation`` as in :meth:`scenarios`, ``precision`` and
        ``device`` as in :meth:`Scenarios.run`."""
        if precision is not None or device is not None:
            from agrijax.facade_execution import options

            with options(precision, device):
                return self.run_batch(
                    treatment,
                    cultivar,
                    years=years,
                    sowing_shift=sowing_shift,
                    soil_evaporation=soil_evaporation,
                )
        return self.scenarios(
            treatment, years=years, sowing_shift=sowing_shift, soil_evaporation=soil_evaporation
        ).run(cultivar)

    def gradient(
        self,
        treatment: int,
        outputs: Sequence[str] | str = ("HWAM", "CWAM"),
        params: Sequence[str] | str = CULTIVAR,
        *,
        cultivar: Mapping[str, float] | None = None,
        scan_points: int = 201,
        soil_evaporation: str | None = None,
    ) -> Any:
        """The derivative of one season's end values (``"HWAM"`` grain yield, ``"CWAM"`` tops weight,
        ``"H#AM"`` grain number, or ``"LAI@60"``-style: the value 60 days after planting) with respect
        to the cultivar coefficients ``params``, at the published cultivar (or ``cultivar``): a
        :class:`~agrijax.facade_grad.Gradient` (``.table``, ``.trust``). Every coefficient carries a
        trust label (``validated gradient`` / ``falls back`` / ``experimental``); derivatives through
        phenology events (``P1``, ``P2``, ``P5``, ``PHINT``) are experimental. ``soil_evaporation``
        as in :meth:`run`. Float64 on JAX's default device only. Details: :mod:`agrijax.facade_grad`."""
        from agrijax.facade_grad import gradient

        return gradient(
            self,
            treatment,
            outputs=outputs,
            params=params,
            cultivar=cultivar,
            scan_points=scan_points,
            soil_evaporation=soil_evaporation,
        )

    def weather_sensitivity(
        self,
        treatment: int,
        outputs: Sequence[str] | str = ("HWAM",),
        variables: Sequence[str] | str = ("SRAD", "TMAX", "TMIN", "RAIN"),
        *,
        year: int | None = None,
        sowing_shift: int = 0,
        check: bool = True,
        config: Any = None,
    ) -> Any:
        """The derivative of one season's end values (``"HWAM"``, ``"CWAM"``, ``"H#AM"``, ``"LAI@60"``-style)
        with respect to every day's weather, ``"SRAD"``, ``"TMAX"``, ``"TMIN"``, ``"RAIN"`` (and
        ``"IRRD"``, the irrigation applied), from the simulation start to maturity, in one reverse-mode
        pass on the full free-run day: a :class:`~agrijax.facade_weather.WeatherSensitivity` (``.daily``,
        ``.stages``, ``.trust``, ``.checks``, ``.attribution()``, ``.phenology_free()``, ``.plot()``).
        ``year`` / ``sowing_shift``: another weather year / sowing date (as :meth:`scenarios`).
        ``check``: run the trust check (single-day and whole-season reruns; ``config`` a
        :class:`~agrijax.facade_weather.WeatherTrustConfig`). Temperature derivatives keep the
        growth-stage calendar fixed (they exclude earlier / later development). Float64 on JAX's default
        devices only. Details: :mod:`agrijax.facade_weather`."""
        from agrijax import facade_weather as fw

        trno = self._trno(treatment)
        files = fw.station_files(self.data / "example_data" / "Weather", self.station)
        if year is None and int(sowing_shift) == 0:
            x = self.inputs(trno)
            name = f"{self.name}_t{trno:02d}"
            owner: Any = self
            key: Any = ("weather", trno)
            runs = [x]
        else:
            scen = self.scenarios(
                trno, years=None if year is None else [int(year)], sowing_shift=[int(sowing_shift)]
            )
            runs = scen.runs
            name = f"{self.name}_t{trno:02d} {int(scen.table.loc[0, 'year'])} {int(sowing_shift):+d} d"
            owner, key = scen, ("weather",)
        (res,) = fw.weather_sensitivity(
            runs,
            [name],
            outputs=outputs,
            variables=variables,
            check=check,
            config=config,
            weather_files=files,
            owner=owner,
            key=key,
        )
        return res


def experiment(
    name: str | os.PathLike[str], *, data_root: str | os.PathLike[str] | None = None
) -> Experiment:
    """A DSSAT maize experiment: an example name (``"UFGA8201"``, from DSSAT's example data,
    :func:`data`) or the path of a ``.MZX`` file (its weather files are read from the data's
    ``example_data/Weather``, its soil from the file's directory or ``example_data/Soil``)."""
    from agrijax.calib.workflow import _filex_path
    from agrijax.io.dssat import read_filex

    root = Path(data_root) if data_root is not None else data()
    fx = _filex_path(name, root)
    x = read_filex(fx)
    first = fx.read_text(errors="replace").splitlines()[0]
    title = first.split(":", 1)[1].strip() if ":" in first else fx.stem
    trs = {int(t["N"]): str(t.get("TNAME", "")).strip() for t in x["TREATMENTS"]}
    culs = {int(t["N"]): str(x["CULTIVARS"][t["CU"]]["INGENO"]).strip() for t in x["TREATMENTS"]}
    fl = next(iter(x["FIELDS"].values()))
    return Experiment(
        fx.stem.upper(),
        fx,
        root,
        title,
        trs,
        culs,
        str(fl.get("ID_SOIL", "")).strip(),
        str(fl.get("WSTA", "")).strip()[:4],
    )


def calibrate(
    experiment: Any,
    treatments: Any = None,
    cultivar: str | None = None,
    *,
    soil_evaporation: str | None = None,
    **kw: Any,
) -> Any:
    """:func:`agrijax.calib.workflow.calibrate` on an :class:`Experiment` (or a name / path) with its
    DSSAT data and float64 switched on; the native inputs (no DSSAT run) unless ``inputs=`` says
    otherwise. ``dssat_check=True`` / :meth:`~agrijax.calib.workflow.CalibrationResult.check_dssat`
    get the DSSAT reference program when they run. ``soil_evaporation`` calibrates on the swapped
    soil evaporation (``"ritchie"`` or ``"salus"``, :func:`alternatives`; needs the native inputs): the
    calibration and its DSSAT check both run on the swap."""
    from agrijax.calib.workflow import calibrate as _calibrate
    from agrijax.facade_execution import require_default_options

    require_default_options("aj.calibrate")
    _x64()
    if isinstance(experiment, Experiment):
        kw.setdefault("data", experiment.data)
        if experiment._engine is not None:
            kw.setdefault("engine", experiment._engine)
    else:
        kw.setdefault("data", data())
    kw.setdefault("inputs", "native")
    if soil_evaporation is None:
        return _calibrate(experiment, treatments, cultivar, **kw)
    from agrijax import facade_swap

    if kw["inputs"] != "native":
        raise facade_swap.SwapError(
            f"soil_evaporation={soil_evaporation!r} needs the native inputs, got inputs={kw['inputs']!r}: "
            "the table inputs hold the experiment's own soil evaporation"
        )
    staged = facade_swap.calibration_experiment(experiment, Path(kw["data"]), soil_evaporation)
    res = _calibrate(staged, treatments, cultivar, **kw)
    res.notes.append(facade_swap.note(soil_evaporation))
    return res


def alternatives(experiment: Any = None, treatment: int | None = None) -> Any:
    """The soil evaporation alternatives validated against ``dscsm048`` that ``soil_evaporation=`` of
    :meth:`Experiment.run`, :meth:`Experiment.scenarios`, :meth:`Experiment.reference` and
    :func:`calibrate` accepts (pandas: the value to pass, the DSSAT option, the method, how it removes
    the water, what it was validated against). With an ``experiment`` and a ``treatment``, a column says
    which one DSSAT uses on that treatment (:func:`agrijax.facade_swap.alternatives`)."""
    from agrijax import facade_swap

    return facade_swap.alternatives(experiment, treatment)
