"""The weather sensitivity calendar: gradients of season outputs with respect to every day's weather.

The user code this module enables (continuing the quick start of :mod:`agrijax.dssat`)::

    ws = exp.weather_sensitivity(treatment=2, outputs=["HWAM", "CWAM"],
                                 variables=["SRAD", "TMAX", "TMIN", "RAIN"])
    print(ws)                  # season totals, the most sensitive stage, a trust label per variable
    ws.daily                   # one row per (output, day, variable): weather, d output / d weather, stage
    ws.calendar("HWAM")        # the same as a day x variable table
    ws.stages                  # per growth stage (boundaries from the run): summed derivative, share
    ws.trust                   # {"SRAD": "validated gradient", "TMAX": "... (phenology calendar fixed; ...)"}
    ws.checks.single_day       # per-day perturbation reruns of the top-k days and random days, three-valued
    ws.checks.whole_season     # SRAD +-5 %, TMAX / TMIN +-1 degC, RAIN +-10 %: summed gradient vs rerun
    ws.phenology_free([-2, -1, 1, 2])  # companion: whole-season temperature offsets run free, batched
    att = ws.attribution()     # first-order climate attribution: gradient x this season's weather anomaly
    att.by_stage, att.total    #   (climatology from the station's weather files, or climatology= / anomaly=)
    ws.brute_force()           # every day x variable rerun one by one (the per-day perturbation approach)
    ws.plot()                  # aj.plot.weather_sensitivity(ws)

    many = exp.scenarios(treatment=2, years=[1979, 1982, 1985]).weather_sensitivity(outputs=["HWAM"])
    many[1982].stages          # one result per scenario, all in one batched program

``exp.weather_sensitivity`` is one season (the experiment's own, or ``year=`` / ``sowing_shift=``);
``scen.weather_sensitivity`` every scenario of a :class:`~agrijax.dssat.Scenarios`. Both are
:func:`weather_sensitivity` of this module.

**What is differentiated.** The end-of-season outputs of Agri-JAX's DSSAT-CSM v4.8.6 maize day
(``"HWAM"``, ``"CWAM"``, ``"H#AM"``, ``"LAI@60"``-style: as :mod:`agrijax.facade_grad`), on the full
free-run day with the native inputs (soil water, runoff by the curve number, infiltration, drainage,
evaporation, transpiration, root water uptake, CERES-Maize; nitrogen off), with respect to the
weather record of every simulated day, from the simulation start to maturity:

* ``SRAD`` [MJ m-2 d-1] - the radiation the crop and ``SPAM`` read;
* ``TMAX``, ``TMIN`` [degC] - the temperatures the crop, ``SPAM`` and ``WATBAL`` read, and through
  them ``HMET``'s hourly mean ``TAVG`` (``TAVG = wx TMAX + wn TMIN`` for the day's length:
  :func:`agrijax.forcing.dssat_weather.hourly_mean_temperature_weights`);
* ``RAIN`` [mm] - the rain ``WATBAL`` reads (runoff, infiltration, drainage, uptake);
* ``IRRD`` [mm] - the irrigation applied that day (as ``WATBAL`` reads it; useful on irrigated
  treatments, a management input rather than weather).

One reverse-mode pass (``jax.vjp`` of the season, one cotangent per output) gives the derivative of
every output with respect to all of them on every day. The tangent is applied to the weather record
as the model receives it (the values of the ``.WTH`` file, which carry one decimal, as the crop's
``Weather.OUT`` prints them; the perturbation reruns below change that record without rounding it
again). The derivative is the straight-through one of :mod:`agrijax.facade_grad` (``ste`` mode: the
identity through the quantisers registered as ``GradientConvention``; column ``derivative``); the
program's own exact-mode derivative is reported next to it (``derivative_exact``, 0 through the
quantisers): in water-limited seasons they differ by the paths through the root length density
truncation and the soil-water rounding.

**Temperature: phenology calendar fixed.** The CERES-Maize stage changes are selections on integer
days: the derivative keeps every stage on its day. A temperature derivative is the effect of a
warmer day *with the growth-stage calendar fixed*; it excludes the earlier or later development a
warmer or cooler season brings (shorter grain filling, ...). Every ``TMAX`` / ``TMIN`` label carries
that caveat (:data:`TEMPERATURE_CAVEAT`); :meth:`WeatherSensitivity.phenology_free` runs the
companion whole-season offsets with the phenology free, in one batch, for comparison. ``SRAD``
carries it too: while the growing point is below ground CERES-Maize's thermal time reads the soil
temperature ``TDSOIL = ACOEF TMAX + (1 - ACOEF) TMIN`` with ``ACOEF`` linear in ``SRAD``
(``MZ_PHENOL``), so radiation moves the early stage days (on UFGA8201 a whole-season ``SRAD`` +5 %
brings silking two days earlier; the whole-season check reports the shift). Rain and irrigation reach
the calendar only through germination (the seed layer's water, ``MZ_PHENOL`` stage 8); the
whole-season check reports any shift as well.

**Trust check** (separate from the coefficient trust report; :class:`WeatherTrustConfig`):

(a) single-day reruns: for each variable the ``top_k`` days of largest ``|d output / d variable|`` (first
    output) and ``random_days`` random season days are rerun with that day's value changed by three
    fixed steps (``SRAD`` 0.1 / 0.5 / 1 MJ m-2, temperatures 0.1 / 0.5 / 1 degC, ``RAIN`` / ``IRRD``
    1 / 2 / 5 mm; central differences, forward only where the value cannot go below 0). Three-valued:
    the differences agree with each other (to ``spread_rtol``) and with the derivative (to ``rtol``)
    -> ``pass``; with each other but not with it -> ``fail``; not with each other -> ``undecidable``
    (they straddle a threshold, a quantum or a stage day: they cannot judge the derivative). Both zero
    counts as agreeing. The derivative judged is the straight-through one, at the steps a user's
    question is about (as level 3 of :mod:`agrijax.calib.trust`); ``rel_err_exact`` reports the
    exact-mode derivative's agreement with the same difference (in water-limited seasons the
    straight-through paths through the quantisers make the two differ, and the small-step response is
    a staircase: the differences at 0.1 / 0.5 / 1 then disagree and the day is undecidable). Rainfall
    passes the runoff curve number, infiltration and the drainage thresholds: its outcome is reported
    as found;
(b) whole-season reruns: ``SRAD`` x(1 +- 5 %), ``TMAX`` / ``TMIN`` (and both) +- 1 degC, ``RAIN`` and
    ``IRRD`` x(1 +- 10 %) on every simulated day, against the summed gradient ``sum_t d output / d x_t
    * dx_t``; the gap (rerun central difference minus prediction) and its curvature part are reported,
    with the silking and maturity shifts of the reruns (the temperature gap includes the stage days
    moving, which the gradient excludes by construction);

A variable is ``"validated gradient"`` when every checked day of every output passes,
``"partly validated (FD undecidable on some days)"`` when some are undecidable and none fails,
``"not validated (FD disagrees on some days)"`` when one fails. The label is about the days checked,
at the check's steps; :meth:`WeatherSensitivity.brute_force` reruns every day.

**Attribution** (:meth:`WeatherSensitivity.attribution`): ``d output / d x_t * (x_t - climatology_t)``
per day and variable, summed by stage and over the season: a first-order estimate of how much each
day's departure from the climatology moved the output. The climatology is the day-of-year mean of the
station's weather files (or what the user passes), and the rerun on the climatological weather
(each variable alone and all together) shows how far first order goes. Temperature terms carry the
phenology caveat; rain anomalies of a daily climatology (a little rain every day against storms) are
far outside a linear range, which the rerun makes visible.

Runs in float64 on the host's JAX devices (``XLA_FLAGS=--xla_force_host_platform_device_count=<cores>``
for the batched reruns; the gradient pass runs on one device). Measured results (private validation
archive): the sensitivity calendars, trust outcomes and costs on UFGA8201 and CA-TPA.
"""

from __future__ import annotations

import datetime as _dt
import time
import weakref
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "LABEL_FAILS",
    "LABEL_PARTIAL",
    "LABEL_VALIDATED",
    "STAGE_NAMES",
    "TEMPERATURE_CAVEAT",
    "VARIABLES",
    "Attribution",
    "WeatherChecks",
    "WeatherSensitivities",
    "WeatherSensitivity",
    "WeatherTrustConfig",
    "climatology",
    "single_day_verdict",
    "station_files",
    "weather_sensitivity",
]

#: the daily inputs that can be differentiated: name -> (unit, meaning), in the perturbation's column order
VARIABLES: dict[str, tuple[str, str]] = {
    "SRAD": ("MJ m-2 d-1", "solar radiation"),
    "TMAX": ("degC", "maximum air temperature"),
    "TMIN": ("degC", "minimum air temperature"),
    "RAIN": ("mm", "rainfall"),
    "IRRD": ("mm", "irrigation applied"),
}
_VORDER: tuple[str, ...] = tuple(VARIABLES)
_IX = {v: i for i, v in enumerate(_VORDER)}
DEFAULT_VARIABLES: tuple[str, ...] = ("SRAD", "TMAX", "TMIN", "RAIN")
#: the variables of the weather record (attribution), and the temperatures (phenology caveat)
WEATHER: tuple[str, ...] = ("SRAD", "TMAX", "TMIN", "RAIN")
TEMPERATURE: tuple[str, ...] = ("TMAX", "TMIN")
#: the variables whose derivative keeps a stage calendar the variable itself moves (the caveat's labels)
PHENOLOGY: tuple[str, ...] = ("SRAD", "TMAX", "TMIN")
#: variables that cannot go below 0 (single-day differences forward only near 0)
_NONNEGATIVE: tuple[str, ...] = ("SRAD", "RAIN", "IRRD")
TEMPERATURE_CAVEAT = "phenology calendar fixed; excludes earlier/later development"
_TEMPERATURE_RERUN_NOTE = f"rerun includes earlier/later development; the gradient: {TEMPERATURE_CAVEAT}"

LABEL_VALIDATED = "validated gradient"
LABEL_PARTIAL = "partly validated (FD undecidable on some days)"
LABEL_FAILS = "not validated (FD disagrees on some days)"
_LABEL_RANK = {LABEL_FAILS: 0, LABEL_PARTIAL: 1, LABEL_VALIDATED: 2}
STATUS_PASS, STATUS_FAIL, STATUS_UNDECIDABLE = "pass", "fail", "undecidable"

#: CERES-Maize ``ISTAGE`` codes: the stage the crop is in (MZ_PHENOL.for INTEGR block comments,
#: :mod:`agrijax.processes.crop.ceres_maize.constants`)
STAGE_NAMES: dict[int, str] = {
    7: "before sowing / sowing day",
    8: "sowing to germination",
    9: "germination to emergence",
    1: "emergence to end of juvenile",
    2: "end of juvenile to tassel initiation",
    3: "tassel initiation to silking",
    4: "silking to beginning of grain fill",
    5: "effective grain filling",
    6: "end of grain fill to maturity",
    10: "after maturity",
}
_ISTAGE_FIRST = 7
#: rows of one compiled forward call (the reruns are cut into chunks of at most this many)
CHUNK_ROWS = 512
#: days simulated past the longest season (real weather first, then padding), as the calibration
PAD_DAYS = 60


@dataclass(frozen=True)
class WeatherTrustConfig:
    """Steps and thresholds of the weather trust check (harness choices, fixed in advance).

    * ``steps`` - the three single-day steps of each variable, in its unit;
    * ``spread_rtol`` - the single-day differences must agree with each other to this (relative to
      the largest) to judge the derivative; ``rtol`` - the derivative must agree with the difference
      at the smallest step to this;
    * ``abs_floor`` - a derivative or difference with ``|d| * largest step`` below ``abs_floor *
      max(|y|, 1)`` counts as zero (as :class:`agrijax.calib.trust.TrustConfig`);
    * ``top_k``, ``random_days``, ``seed`` - the days checked: the largest ``|derivative|`` of the first
      output, and random season days;
    * ``season`` - the whole-season perturbations: relative for ``SRAD`` / ``RAIN`` / ``IRRD``, in degC
      for the temperatures.
    """

    steps: Mapping[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "SRAD": (0.1, 0.5, 1.0),
            "TMAX": (0.1, 0.5, 1.0),
            "TMIN": (0.1, 0.5, 1.0),
            "RAIN": (1.0, 2.0, 5.0),
            "IRRD": (1.0, 2.0, 5.0),
        }
    )
    spread_rtol: float = 0.05
    rtol: float = 0.05
    abs_floor: float = 1e-10
    top_k: int = 8
    random_days: int = 4
    seed: int = 0
    season: Mapping[str, float] = field(
        default_factory=lambda: {"SRAD": 0.05, "TMAX": 1.0, "TMIN": 1.0, "RAIN": 0.10, "IRRD": 0.10}
    )


DEFAULT_CONFIG = WeatherTrustConfig()


def _season_label(v: str, cfg: WeatherTrustConfig) -> str:
    a = cfg.season[v]
    return f"{v} +-{a:g} degC" if v in TEMPERATURE else f"{v} x(1 +- {100 * a:g} %)"


def _with_caveat(v: str, label: str) -> str:
    return f"{label} ({TEMPERATURE_CAVEAT})" if v in PHENOLOGY else label


# ------------------------------------------------------------------------------ verdicts (no model)
def single_day_verdict(
    ad: Any, fds: Sequence[Any], y0: Any, h_scale: float, cfg: WeatherTrustConfig = DEFAULT_CONFIG
) -> dict[str, np.ndarray]:
    """The three-valued single-day verdict of derivatives ``ad`` against the differences ``fds`` at the
    three steps (smallest first), elementwise; ``y0`` the outputs (same shape), ``h_scale`` the largest
    step. ``status``: ``pass`` / ``fail`` / ``undecidable`` (module docstring, (a)); ``fd`` the
    difference at the smallest step, ``spread`` the differences' spread, ``rel_err`` the derivative's
    disagreement with ``fd``."""
    ad = np.asarray(ad, dtype=float)
    fs = np.stack([np.asarray(f, dtype=float) for f in fds])
    y0 = np.asarray(y0, dtype=float)
    floor = cfg.abs_floor * np.maximum(np.abs(y0), 1.0) / float(h_scale)
    big = np.abs(fs).max(axis=0)
    spread = (fs.max(axis=0) - fs.min(axis=0)) / np.maximum(big, 1e-300)
    all_zero = np.all(np.abs(fs) <= floor, axis=0)
    consistent = all_zero | (spread <= cfg.spread_rtol)
    fd = fs[0]
    rel = np.abs(ad - fd) / np.maximum(np.maximum(np.abs(ad), np.abs(fd)), 1e-300)
    both_zero = (np.abs(ad) <= floor) & (np.abs(fd) <= floor)
    agree = both_zero | (rel <= cfg.rtol)
    status = np.where(~consistent, STATUS_UNDECIDABLE, np.where(agree, STATUS_PASS, STATUS_FAIL)).astype(
        object
    )
    return {
        "status": status,
        "fd": fd,
        "spread": np.where(all_zero, 0.0, spread),
        "rel_err": np.where(both_zero, 0.0, rel),
    }


def _variable_label(statuses: Iterable[str]) -> str:
    st = list(statuses)
    if STATUS_FAIL in st:
        return LABEL_FAILS
    if STATUS_UNDECIDABLE in st:
        return LABEL_PARTIAL
    return LABEL_VALIDATED


def _stage_runs(stage: np.ndarray) -> list[tuple[int, int, int]]:
    """Consecutive runs ``(code, first, last)`` (inclusive day indices) of ``stage``."""
    out: list[tuple[int, int, int]] = []
    start = 0
    for t in range(1, len(stage) + 1):
        if t == len(stage) or stage[t] != stage[start]:
            out.append((int(stage[start]), start, t - 1))
            start = t
    return out


def _morning_stage(istage_end: np.ndarray) -> np.ndarray:
    """The stage the crop is in on entering each day: the previous day's end-of-day ``istage``."""
    ist = np.asarray(istage_end, dtype=int)
    return np.concatenate([[_ISTAGE_FIRST], ist[:-1]]) if ist.size else ist


# ------------------------------------------------------------------------------ climatology
def _leap_doy(d: _dt.date) -> int:
    """Day of a leap year (1-366) with the same month and day: Feb 29 has its own slot."""
    return _dt.date(2000, d.month, d.day).timetuple().tm_yday


def climatology(
    files: Sequence[str | Path], years: Iterable[int] | None = None, exclude: Iterable[int] = ()
) -> Any:
    """The day-of-year mean of ``SRAD``, ``TMAX``, ``TMIN``, ``RAIN`` over the weather files ``files``
    (DSSAT ``.WTH``; only the ``years`` given, default all, without ``exclude``): pandas, indexed by the
    day of a leap year (1-366, Feb 29 its own day, filled from Feb 28 / Mar 1 when no year has it),
    with the years used in ``attrs["years"]``."""
    import pandas as pd

    from agrijax.io.dssat import read_wth

    keep = None if years is None else {int(y) for y in years}
    drop = {int(y) for y in exclude}
    frames = []
    for f in files:
        df = read_wth(Path(f), dssat_spans=True)
        df = df[[(keep is None or d.year in keep) and d.year not in drop for d in df["date"].dt.date]]
        if len(df):
            frames.append(df)
    if not frames:
        raise ValueError("climatology: no weather in the files for the years asked")
    allw: Any = pd.concat(frames).drop_duplicates(subset="date")
    doy = [_leap_doy(d) for d in allw["date"].dt.date]
    cols = {"srad": "SRAD", "tmax": "TMAX", "tmin": "TMIN", "rain": "RAIN"}
    tab: Any = allw[list(cols)].rename(columns=cols).assign(doy=doy).groupby("doy").mean()
    tab = tab.reindex(range(1, 367))
    if tab.loc[60].isna().any():  # Feb 29 (leap day 60) from its neighbours
        tab.loc[60] = (tab.loc[59] + tab.loc[61]) / 2.0
    tab = tab.interpolate(limit_direction="both")
    tab.index.name = "doy"
    tab.attrs["years"] = sorted({d.year for d in allw["date"].dt.date})
    return tab


# ------------------------------------------------------------------------------ the programs
def _hook(f: Any, t: Any, tid: Any, delta: Any, wts: Any) -> Any:
    """The day's forcing of the samples with the weather perturbation ``delta [S, T, V]`` added on
    day ``t`` to every place the model reads it (``wts [B, T, 2]``: the TAVG weights)."""
    d = delta[:, t]
    w = wts[tid, t]

    def add(x: Any, v: Any) -> Any:
        return x + v.reshape(v.shape + (1,) * (x.ndim - v.ndim))

    srad, tmax, tmin, rain, irr = (d[:, _IX[v]] for v in _VORDER)
    soil = f["soil"]
    soil = soil.replace(
        rain=add(soil.rain, rain), irrigation=add(soil.irrigation, irr), tmax=add(soil.tmax, tmax)
    )
    spam = f["spam"]
    wx = spam["weather"]
    wx = wx.replace(
        srad=add(wx.srad, srad),
        tmax=add(wx.tmax, tmax),
        tmin=add(wx.tmin, tmin),
        tavg=add(wx.tavg, w[:, 0] * tmax + w[:, 1] * tmin),
    )
    c = f["crop"]
    crop = c.replace(srad=add(c.srad, srad), tmax=add(c.tmax, tmax), tmin=add(c.tmin, tmin))
    return {**f, "soil": soil, "spam": {**spam, "weather": wx}, "crop": crop}


class _Programs:
    """The weather-perturbed season of a batch of runs (one batch group): the forward values of rows
    ``(scenario, delta [T, V])`` on every device, and the reverse-mode derivative of the outputs with
    respect to ``delta`` on one device (``ste`` and ``exact`` modes)."""

    def __init__(self, runs: Sequence[Any], entries: Sequence[Sequence[Any]], n_out: int) -> None:
        import jax

        from agrijax.calib.dssat_day import CUL_ORDER, DaySimulator
        from agrijax.forcing.dssat_weather import hourly_mean_temperature_weights

        self.runs = list(runs)
        self.n_out = int(n_out)
        self._entries = [list(e) for e in entries]
        self.many = DaySimulator(self.runs, [list(e) for e in entries], pad_days=PAD_DAYS)
        self.one = DaySimulator(
            self.runs, [list(e) for e in entries], pad_days=PAD_DAYS, devices=jax.devices()[:1]
        )
        if len(self.many.groups) != 1:
            raise ValueError("the scenarios must share one soil layering and one evaporation method")
        (self.group,) = self.many.groups
        self.n_days = int(self.many.n_days[self.group])
        self.local = np.asarray([self.many.where[i][1] for i in range(len(self.runs))], dtype=np.int32)
        self.theta = np.asarray([[r.published()[n] for n in CUL_ORDER] for r in self.runs], dtype=float)
        _, f, _ = self.many.inputs[self.group]
        wx = f["spam"]["weather"]
        cols = (wx.srad, wx.tmax, wx.tmin, f["soil"].rain, f["soil"].irrigation)
        #: the weather the model reads [B (group order), T, V], and the TAVG weights [B, T, 2]
        self.weather = np.stack([np.asarray(a, dtype=float) for a in cols], axis=-1)
        w_x, w_n = hourly_mean_temperature_weights(np.asarray(f["crop"].dayl, dtype=float))
        self.wts = np.stack([w_x, w_n], axis=-1)
        self._wts_dev = {
            "many": jax.device_put(self.wts, self.many.rep),
            "one": jax.device_put(self.wts, self.one.rep),
        }
        self._fns = {k: s._sim_fn(self.group, _hook) for k, s in (("many", self.many), ("one", self.one))}
        self._progs: dict[Any, Any] = {}
        self.compile_s = 0.0
        self.run_s = 0.0

    def weather_of(self, i: int) -> np.ndarray:
        """The weather the model reads for run ``i``, ``[T, V]``."""
        return self.weather[self.local[i]]

    def _sim(self, which: str) -> Any:
        return self.many if which == "many" else self.one

    def _forward(self, which: str, n: int) -> Any:
        key = ("fwd", which, n)
        if key not in self._progs:
            import jax
            import jax.numpy as jnp

            from agrijax.calib.dssat_day import GRADIENT_MODE
            from agrijax.core.grad import bind_gradient_mode

            sim, fn = self._sim(which), self._fns[which]

            def f(inputs: Any, tab: Any, theta: Any, tid: Any, wts: Any, delta: Any) -> Any:
                return fn(inputs, tab, theta, tid, delta, wts)

            jf = jax.jit(
                bind_gradient_mode(f, GRADIENT_MODE),
                in_shardings=(sim.rep, sim.rep, sim.bsh, sim.bsh, sim.rep, sim.bsh),
                out_shardings=sim.bsh,
            )
            args = (
                sim.inputs[self.group],
                sim.tables[self.group],
                jnp.zeros((n, self.theta.shape[1])),
                jnp.zeros(n, jnp.int32),
                self._wts_dev[which],
                jnp.zeros((n, self.n_days, len(_VORDER))),
            )
            t0 = time.perf_counter()
            self._progs[key] = jf.lower(*args).compile()
            self.compile_s += time.perf_counter() - t0
        return self._progs[key]

    def values(self, delta: np.ndarray, scn: np.ndarray, which: str = "many") -> tuple[np.ndarray, float]:
        """Entry values ``[N, E]`` of the rows ``(scn [N], delta [N, T, V])`` and the wall time of the
        calls [s] (compilation excluded)."""
        import jax

        sim = self._sim(which)
        delta = np.asarray(delta, dtype=np.float64)
        scn = np.asarray(scn, dtype=np.int64)
        n = delta.shape[0]
        cap = sim.ndev
        while cap * 2 <= max(CHUNK_ROWS, sim.ndev):
            cap *= 2
        n_chunks = -(-n // cap)
        size = sim._pad(-(-n // n_chunks))
        prog = self._forward(which, size)
        out = np.zeros((n, sim.n_entries))
        secs = 0.0
        for c0 in range(0, n, size):
            m = min(size, n - c0)
            sl = slice(c0, c0 + m)

            def rows(a: np.ndarray, sl: slice = sl, m: int = m) -> np.ndarray:
                return np.concatenate([a[sl], np.repeat(a[sl][-1:], size - m, axis=0)]) if m < size else a[sl]

            t0 = time.perf_counter()
            y = prog(
                sim.inputs[self.group],
                sim.tables[self.group],
                jax.device_put(rows(self.theta[scn]), sim.bsh),
                jax.device_put(rows(self.local[scn]), sim.bsh),
                self._wts_dev[which],
                jax.device_put(rows(delta), sim.bsh),
            )
            out[sl] = np.asarray(y)[:m]
            secs += time.perf_counter() - t0
        self.run_s += secs
        return out, secs

    def _gradient_prog(self, mode: str, s_n: int) -> Any:
        key = ("grad", mode, s_n)
        if key not in self._progs:
            import jax
            import jax.numpy as jnp

            from agrijax.core.grad import bind_gradient_mode

            sim, fn, e_n = self.one, self._fns["one"], self.n_out

            def g(inputs: Any, tab: Any, theta: Any, tid: Any, wts: Any, delta: Any) -> Any:
                def f(d: Any) -> Any:
                    return fn(inputs, tab, theta, tid, d, wts)[:, :e_n]

                y, vjp = jax.vjp(f, delta)
                cts = jnp.broadcast_to(jnp.eye(e_n, dtype=y.dtype)[:, None, :], (e_n, *y.shape))
                (gr,) = jax.vmap(vjp)(cts)
                return y, gr

            jf = jax.jit(bind_gradient_mode(g, mode))
            args = (
                sim.inputs[self.group],
                sim.tables[self.group],
                jnp.zeros((s_n, self.theta.shape[1])),
                jnp.zeros(s_n, jnp.int32),
                self._wts_dev["one"],
                jnp.zeros((s_n, self.n_days, len(_VORDER))),
            )
            t0 = time.perf_counter()
            self._progs[key] = jf.lower(*args).compile()
            self.compile_s += time.perf_counter() - t0
        return self._progs[key]

    def gradient(self, scn: Sequence[int], mode: str) -> tuple[np.ndarray, np.ndarray, float]:
        """``(y [S, n_out], d y / d delta [S, n_out, T, V], seconds)`` at the unperturbed weather of the
        runs ``scn``, in gradient ``mode`` (one reverse-mode pass on one device)."""
        import jax
        import jax.numpy as jnp

        s = np.asarray(scn, dtype=np.int64)
        prog = self._gradient_prog(mode, int(s.size))
        sim = self.one
        t0 = time.perf_counter()
        y, gr = prog(
            sim.inputs[self.group],
            sim.tables[self.group],
            jnp.asarray(self.theta[s]),
            jnp.asarray(self.local[s]),
            self._wts_dev["one"],
            jnp.zeros((s.size, self.n_days, len(_VORDER))),
        )
        y, gr = jax.block_until_ready((y, gr))
        secs = time.perf_counter() - t0
        self.run_s += secs
        return np.asarray(y), np.transpose(np.asarray(gr), (1, 0, 2, 3)), secs


# ------------------------------------------------------------------------------ results
@dataclass
class WeatherChecks:
    """The weather trust check of one season (module docstring): :attr:`single_day` (one row per
    output, variable and day checked), :attr:`whole_season` (one row per output and perturbation),
    :attr:`summary` (per output and variable: the days that pass / fail / are undecidable, the label)."""

    single_day: Any
    whole_season: Any
    summary: Any


@dataclass
class Attribution:
    """First-order climate attribution of one season (:meth:`WeatherSensitivity.attribution`).

    :attr:`daily`: one row per (output, day, variable): the weather, the climatology, the anomaly and
    the contribution ``derivative x anomaly``; :attr:`by_stage`: contributions summed per growth stage;
    :attr:`total`: per output and variable, the summed contribution (``linear``) and the rerun on the
    climatological weather (``rerun``: the output minus the output with that variable, or all of them
    for ``ALL``, replaced by the climatology), with the caveat column. :attr:`years`: the climatology's
    years."""

    daily: Any
    by_stage: Any
    total: Any
    years: list[int]
    notes: list[str]

    def __str__(self) -> str:
        t = self.total.to_string(index=False, float_format=lambda v: f"{v:.5g}")
        return "\n".join(
            [f"first-order attribution against the climatology of {len(self.years)} years", t, *self.notes]
        )


@dataclass
class WeatherSensitivity:
    """The derivative of season outputs with respect to every day's weather on one season
    (:meth:`agrijax.dssat.Experiment.weather_sensitivity`; module docstring).

    :attr:`daily` has one row per (output, day, variable) from the simulation start to maturity:
    ``date``, ``day`` (index from the simulation start), ``dap`` (days after planting), ``stage`` and
    ``stage_name`` (the stage the crop is in that day), ``value`` (the weather), ``derivative`` (d output
    / d variable that day, straight-through), ``derivative_exact`` (the program's own), ``unit``,
    ``checked`` (the single-day verdict where that day was checked). :attr:`stages` sums it per growth
    stage: ``sum`` (the change of the output for +1 unit on every day of the stage), ``share`` (of the
    season's sum of ``|derivative|``), ``per_pct`` (for +1 % of the stage's values: radiation, rain,
    irrigation). :attr:`values` are the outputs, :attr:`trust` the label of each variable, :attr:`checks`
    the trust check, :attr:`timing` the cost."""

    name: str
    outputs: list[str]
    variables: list[str]
    values: dict[str, float]
    daily: Any
    stages: Any
    trust: dict[str, str]
    checks: WeatherChecks | None
    timing: dict[str, Any]
    dates: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    _ctx: Any = field(default=None, repr=False)

    # ---------------------------------------------------------------- views
    def calendar(self, output: str | None = None, *, exact: bool = False) -> Any:
        """``d output / d variable`` as a table: one row per day (date index, with the stage), one
        column per variable (``exact=True``: the exact-mode derivative)."""
        o = self.outputs[0] if output is None else output
        d = self.daily[self.daily["output"] == o]
        col = "derivative_exact" if exact else "derivative"
        tab = d.pivot_table(index=["date", "stage"], columns="variable", values=col, sort=False)
        return tab[[v for v in self.variables if v in tab.columns]]

    def __str__(self) -> str:
        lines = [
            f"d output / d daily weather, {self.name}: {self.dates['start']} to {self.dates['maturity']} "
            f"({self.dates['days']} days), one reverse-mode pass {self.timing.get('gradient_s', 0.0):.2f} s",
            ", ".join(f"{o} = {self.values[o]:.6g}" for o in self.outputs),
        ]
        rows = []
        for o in self.outputs:
            for v in self.variables:
                d = self.daily[(self.daily["output"] == o) & (self.daily["variable"] == v)]
                st = self.stages[(self.stages["output"] == o) & (self.stages["variable"] == v)]
                top = st.loc[st["share"].idxmax()] if len(st) else None
                rows.append(
                    {
                        "output": o,
                        "variable": v,
                        "season sum": float(d["derivative"].sum()),
                        "unit": d["unit"].iloc[0] if len(d) else "",
                        "top stage": f"{top['stage_name']} ({100 * top['share']:.0f} %)"
                        if top is not None
                        else "",
                    }
                )
        import pandas as pd

        lines.append(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        lines.append("trust: " + "; ".join(f"{v} {t}" for v, t in self.trust.items()))
        if self.checks is not None and len(self.checks.whole_season):
            ws = self.checks.whole_season
            cols = [
                "output",
                "perturbation",
                "predicted",
                "rerun",
                "gap",
                "mdat_shift_plus",
                "mdat_shift_minus",
            ]
            lines.append("whole season (summed gradient vs rerun):")
            lines.append(ws[cols].to_string(index=False, float_format=lambda v: f"{v:.5g}"))
        lines.extend(self.notes)
        return "\n".join(lines)

    def plot(self, output: str | None = None, **kw: Any) -> Any:
        """:func:`agrijax.report.plot.weather_sensitivity` (needs matplotlib)."""
        from agrijax.report.plot import weather_sensitivity as _p

        return _p(self, output=output, **kw)

    # ---------------------------------------------------------------- companions
    def _delta_rows(self, deltas: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        ctx = self._ctx
        n = len(deltas)
        return np.stack(deltas) if n else np.zeros((0, ctx.progs.n_days, len(_VORDER))), np.full(n, ctx.i)

    def _run_rows(self, deltas: Sequence[np.ndarray]) -> tuple[np.ndarray, float]:
        d, s = self._delta_rows(deltas)
        return self._ctx.progs.values(d, s)

    def phenology_free(self, offsets: Sequence[float] = (-2.0, -1.0, 1.0, 2.0)) -> Any:
        """The companion of the temperature derivatives: ``TMAX`` and ``TMIN`` shifted together by each
        of ``offsets`` [degC] on every simulated day, run with the phenology free (stage days move), in
        one batch, against the gradient's prediction ``offset x sum_t (d/dTMAX_t + d/dTMIN_t)`` (phenology
        calendar fixed). One row per output and offset: the rerun's change, the prediction, the gap, and
        the silking / maturity shifts [days] (pandas)."""
        import pandas as pd

        ctx = self._ctx
        offs = [float(o) for o in offsets]
        deltas = []
        for o in offs:
            d = np.zeros((ctx.progs.n_days, len(_VORDER)))
            d[:, _IX["TMAX"]] = o
            d[:, _IX["TMIN"]] = o
            deltas.append(d)
        y, secs = self._run_rows([np.zeros_like(deltas[0]), *deltas])
        rows = []
        for e, o in enumerate(self.outputs):
            pred_unit = float(ctx.g[e, :, _IX["TMAX"]].sum() + ctx.g[e, :, _IX["TMIN"]].sum())
            for j, off in enumerate(offs):
                yy = y[j + 1]
                rows.append(
                    {
                        "output": o,
                        "offset_degC": off,
                        "value": float(yy[e]),
                        "rerun": float(yy[e] - y[0, e]),
                        "predicted": off * pred_unit,
                        "gap": float(yy[e] - y[0, e]) - off * pred_unit,
                        "adat_shift": int(yy[ctx.e_silk] - y[0, ctx.e_silk]),
                        "mdat_shift": int(yy[ctx.e_mat] - y[0, ctx.e_mat]),
                    }
                )
        out = pd.DataFrame(rows)
        out.attrs["run_s"] = secs
        out.attrs["note"] = (
            "rerun: phenology free (the stage days move); predicted: the summed temperature gradient "
            f"({TEMPERATURE_CAVEAT})"
        )
        return out

    def brute_force(
        self,
        variables: Sequence[str] | None = None,
        step: Mapping[str, float] | None = None,
        *,
        one_device: bool = False,
    ) -> Any:
        """Every day from the start to maturity x every variable rerun with that day's value raised by
        ``step`` (default the middle single-day step of :class:`WeatherTrustConfig`): the per-day
        perturbation approach, batched. One row per (output, day, variable): the forward difference,
        the derivative and their relative difference (pandas; ``attrs``: rows run and their wall time,
        to compare with the one gradient pass in :attr:`timing`). ``one_device``: run them on one device,
        as the gradient pass (the same hardware, one core's program)."""
        import pandas as pd

        ctx = self._ctx
        vs = list(self.variables if variables is None else variables)
        st = {v: ctx.cfg.steps[v][1] for v in vs} | dict(step or {})
        t_cal = ctx.t_cal
        deltas = [np.zeros((ctx.progs.n_days, len(_VORDER)))]
        keys = []
        for v in vs:
            for t in range(t_cal):
                d = np.zeros((ctx.progs.n_days, len(_VORDER)))
                d[t, _IX[v]] = st[v]
                deltas.append(d)
                keys.append((v, t))
        d_rows, s_rows = self._delta_rows(deltas)
        which = "one" if one_device else "many"
        y, secs = ctx.progs.values(d_rows, s_rows, which)
        rows = []
        for e, o in enumerate(self.outputs):
            for j, (v, t) in enumerate(keys):
                fd = float((y[j + 1, e] - y[0, e]) / st[v])
                ad = float(ctx.g[e, t, _IX[v]])
                den = max(abs(fd), abs(ad), 1e-300)
                rows.append(
                    {
                        "output": o,
                        "day": t,
                        "date": ctx.dates[t],
                        "stage": int(ctx.stage[t]),
                        "variable": v,
                        "step": st[v],
                        "fd": fd,
                        "derivative": ad,
                        "rel_diff": abs(fd - ad) / den if den > 1e-300 else 0.0,
                    }
                )
        out = pd.DataFrame(rows)
        out.attrs.update(
            {"rows": len(deltas), "run_s": secs, "devices": 1 if one_device else ctx.progs.many.ndev}
        )
        return out

    def attribution(
        self,
        climatology: Any = None,
        *,
        anomaly: Any = None,
        years: Iterable[int] | None = None,
        rerun: bool = True,
    ) -> Attribution:
        """First-order climate attribution (module docstring): ``derivative x (weather - climatology)``
        per day for ``SRAD``, ``TMAX``, ``TMIN``, ``RAIN`` (those of :attr:`variables`).

        ``climatology``: a table indexed by the day of a leap year (1-366, :func:`climatology`) with
        those columns; default the day-of-year mean of the station's weather files (``years``: which
        ones; default all of them). ``anomaly``: instead, the anomaly itself (a table with those columns
        and one row per day from the simulation start, or indexed by ``YYYYDDD``). ``rerun``: also run
        the season on the climatological weather (each variable alone, and all together)."""
        import pandas as pd

        ctx = self._ctx
        vs = [v for v in self.variables if v in WEATHER]
        if not vs:
            raise ValueError("attribution: none of SRAD, TMAX, TMIN, RAIN among the variables")
        t_cal = ctx.t_cal
        x = ctx.progs.weather_of(ctx.i)[:t_cal]
        used_years: list[int] = []
        if anomaly is not None:
            an_t = pd.DataFrame(anomaly)
            if set(ctx.yrdoy[:t_cal]).issubset({int(k) for k in an_t.index}):
                an = np.stack([np.asarray(an_t.loc[ctx.yrdoy[:t_cal], v], dtype=float) for v in vs], axis=1)
            elif len(an_t) >= t_cal:
                an = np.stack([np.asarray(an_t[v], dtype=float)[:t_cal] for v in vs], axis=1)
            else:
                raise ValueError(f"anomaly: {len(an_t)} rows, the season has {t_cal} days")
            clim = x[:, [_IX[v] for v in vs]] - an
        else:
            if climatology is None:
                if not ctx.weather_files:
                    raise ValueError("attribution: no weather files known; pass climatology= or anomaly=")
                climatology_tab = _climatology(ctx.weather_files, years)
            else:
                climatology_tab = pd.DataFrame(climatology)
            used_years = list(climatology_tab.attrs.get("years", []))
            doy = [_leap_doy(d) for d in ctx.dates[:t_cal]]
            clim = np.stack([np.asarray(climatology_tab.loc[doy, v], dtype=float) for v in vs], axis=1)
            an = x[:, [_IX[v] for v in vs]] - clim
        contrib = np.stack([ctx.g[:, :t_cal, _IX[v]] * an[None, :, k] for k, v in enumerate(vs)], axis=-1)
        rows = []
        for e, o in enumerate(self.outputs):
            for t in range(t_cal):
                for k, v in enumerate(vs):
                    rows.append(
                        {
                            "output": o,
                            "day": t,
                            "date": ctx.dates[t],
                            "stage": int(ctx.stage[t]),
                            "stage_name": STAGE_NAMES.get(int(ctx.stage[t]), ""),
                            "variable": v,
                            "value": float(x[t, _IX[v]]),
                            "climatology": float(clim[t, k]),
                            "anomaly": float(an[t, k]),
                            "derivative": float(ctx.g[e, t, _IX[v]]),
                            "contribution": float(contrib[e, t, k]),
                        }
                    )
        daily = pd.DataFrame(rows)
        st_rows = []
        for e, o in enumerate(self.outputs):
            for code, a, b in _stage_runs(ctx.stage[:t_cal]):
                for k, v in enumerate(vs):
                    st_rows.append(
                        {
                            "output": o,
                            "stage": code,
                            "stage_name": STAGE_NAMES.get(code, ""),
                            "first": ctx.dates[a],
                            "last": ctx.dates[b],
                            "variable": v,
                            "mean_anomaly": float(an[a : b + 1, k].mean()),
                            "contribution": float(contrib[e, a : b + 1, k].sum()),
                        }
                    )
        by_stage = pd.DataFrame(st_rows)
        cf: dict[str, np.ndarray] = {}
        if rerun:
            deltas = [np.zeros((ctx.progs.n_days, len(_VORDER)))]
            for k, v in enumerate(vs):
                d = np.zeros_like(deltas[0])
                d[:t_cal, _IX[v]] = -an[:, k]
                deltas.append(d)
            d_all = np.sum(np.stack(deltas[1:]), axis=0)
            deltas.append(d_all)
            y, _ = self._run_rows(deltas)
            for k, v in enumerate(vs):
                cf[v] = y[0] - y[k + 1]
            cf["ALL"] = y[0] - y[-1]
        tot = []
        for e, o in enumerate(self.outputs):
            for k, v in enumerate([*vs, "ALL"]):
                lin = float(contrib[e, :, k].sum()) if v != "ALL" else float(contrib[e].sum())
                tot.append(
                    {
                        "output": o,
                        "variable": v,
                        "linear": lin,
                        "rerun": float(cf[v][e]) if v in cf else np.nan,
                        "caveat": TEMPERATURE_CAVEAT if v in PHENOLOGY or v == "ALL" else "",
                    }
                )
        notes = [
            "linear: sum over the days of derivative x (weather - climatology); rerun: the output minus the "
            "output on the climatological weather (phenology free).",
            f"temperature terms: {TEMPERATURE_CAVEAT}.",
            "rain against a daily climatology (a little rain every day) is far from a small perturbation: "
            "compare the linear and rerun columns before reading the rain terms.",
        ]
        return Attribution(daily, by_stage, pd.DataFrame(tot), used_years, notes)


def _climatology(files: Sequence[Path], years: Iterable[int] | None) -> Any:
    return climatology(files, years)


@dataclass
class _Context:
    """What the companions of one result need: the programs, the run's index, its derivative and its
    calendar."""

    progs: _Programs
    i: int
    g: np.ndarray  # [E, T, V] straight-through derivative
    t_cal: int
    stage: np.ndarray
    dates: list[Any]
    yrdoy: list[int]
    e_mat: int
    e_silk: int
    cfg: WeatherTrustConfig
    weather_files: list[Path]


# ------------------------------------------------------------------------------ the analysis
_CACHE: dict[int, tuple[Any, dict[Any, _Programs]]] = {}


def _cached(owner: Any, key: Any, build: Callable[[], _Programs]) -> _Programs:
    """The programs of ``key`` kept on ``owner`` (an experiment or a scenario set) while it lives."""
    if owner is None:
        return build()
    k = id(owner)
    hit = _CACHE.get(k)
    if hit is None or hit[0]() is not owner:
        store: dict[Any, _Programs] = {}
        _CACHE[k] = (weakref.ref(owner, lambda _r, k=k: _CACHE.pop(k, None)), store)
    else:
        store = hit[1]
    if key not in store:
        store[key] = build()
    return store[key]


def _parse_variables(variables: Sequence[str] | str) -> list[str]:
    vs = [variables] if isinstance(variables, str) else [str(v).upper() for v in variables]
    bad = [v for v in vs if v not in VARIABLES]
    if bad or not vs or len(set(vs)) != len(vs):
        raise ValueError(f"variables {vs}: distinct names out of {list(VARIABLES)}")
    return vs


def _date(yrdoy: int) -> _dt.date:
    return _dt.date(yrdoy // 1000, 1, 1) + _dt.timedelta(days=yrdoy % 1000 - 1)


def _single_day_rows(
    g0: np.ndarray, x: np.ndarray, vs: Sequence[str], t_cal: int, n_days: int, cfg: WeatherTrustConfig
) -> tuple[list[np.ndarray], list[tuple[str, int, str, int, int, float, float]]]:
    """The perturbation rows of the single-day check: per variable the top-k and random days, three
    steps, plus / minus rows (the minus row omitted, ``-1``, where the value cannot go below 0).
    Returns the deltas (the base row first) and per (variable, day, step) ``(v, day, kind, i_plus,
    i_minus, h_plus, h_minus)``."""
    rng = np.random.default_rng(cfg.seed)
    deltas = [np.zeros((n_days, len(_VORDER)))]
    keys: list[tuple[str, int, str, int, int, float, float]] = []
    for v in vs:
        score = np.abs(g0[:t_cal, _IX[v]])
        order = [int(t) for t in np.argsort(-score, kind="stable") if score[t] > 0][: cfg.top_k]
        rest = [t for t in range(t_cal) if t not in order]
        rnd = sorted(int(t) for t in rng.choice(rest, size=min(cfg.random_days, len(rest)), replace=False))
        for t, kind in [(t, "top") for t in order] + [(t, "random") for t in rnd]:
            for h in cfg.steps[v]:
                d = np.zeros((n_days, len(_VORDER)))
                d[t, _IX[v]] = h
                deltas.append(d)
                ip = len(deltas) - 1
                if v in _NONNEGATIVE and x[t, _IX[v]] - h < 0.0:
                    keys.append((v, t, kind, ip, -1, h, 0.0))
                    continue
                d = np.zeros((n_days, len(_VORDER)))
                d[t, _IX[v]] = -h
                deltas.append(d)
                keys.append((v, t, kind, ip, len(deltas) - 1, h, h))
    return deltas, keys


def _whole_season_deltas(
    vs: Sequence[str], x: np.ndarray, cfg: WeatherTrustConfig
) -> list[tuple[str, np.ndarray]]:
    """``(label, +delta [T, V])`` of the whole-season perturbations (the minus row is ``-delta``)."""
    out = []
    for v in vs:
        d = np.zeros_like(x)
        a = cfg.season[v]
        d[:, _IX[v]] = a if v in TEMPERATURE else a * x[:, _IX[v]]
        out.append((_season_label(v, cfg), d))
    if all(t in vs for t in TEMPERATURE):
        d = np.zeros_like(x)
        d[:, _IX["TMAX"]] = cfg.season["TMAX"]
        d[:, _IX["TMIN"]] = cfg.season["TMIN"]
        out.append((f"TMAX and TMIN +-{cfg.season['TMAX']:g} degC", d))
    return out


def _analyse(
    progs: _Programs,
    i: int,
    run: Any,
    name: str,
    out_names: Sequence[str],
    vs: Sequence[str],
    g: np.ndarray,
    g_exact: np.ndarray,
    grad_s: float,
    grad_exact_s: float,
    cfg: WeatherTrustConfig,
    check: bool,
    weather_files: Sequence[Path],
) -> WeatherSensitivity:
    """One season's result from its derivatives ``g``, ``g_exact`` ``[E, T, V]``: the base row, the
    calendar, the stage table and (``check``) the trust check."""
    import pandas as pd

    from agrijax.facade_grad import _output_unit, _planting_index

    e_n = len(out_names)
    e_mat, e_silk = e_n, e_n + 1
    x = progs.weather_of(i)
    n_days = progs.n_days
    deltas: list[np.ndarray] = [np.zeros((n_days, len(_VORDER)))]
    # the base row and the calendar
    t0 = time.perf_counter()
    yb, _ = progs.values(np.stack(deltas), np.asarray([i]))
    base = yb[0]
    t_mat = int(base[e_mat])
    t_cal = min(t_mat, n_days - 1) + 1
    o = run.simulate(pad_days=n_days - run.n_days)
    stage = _morning_stage(np.asarray(o["istage"])[:, 0])
    start = _date(int(run.days[0]))
    dates = [start + _dt.timedelta(days=t) for t in range(n_days)]
    yrdoy = [int(d.strftime("%Y%j")) for d in dates]
    t_plant = _planting_index(run)
    status_of: dict[tuple[str, str, int], str] = {}
    checks = None
    labels: dict[str, str] = {}
    check_s = 0.0
    n_check_rows = 0
    if check:
        sd_deltas, keys = _single_day_rows(g[0], x, vs, t_cal, n_days, cfg)
        ws = _whole_season_deltas(vs, x, cfg)
        all_rows = sd_deltas + [d for _, d in ws] + [-d for _, d in ws]
        yy, check_s = progs.values(np.stack(all_rows), np.full(len(all_rows), i))
        n_check_rows = len(all_rows)
        y0 = yy[0]
        sd_rows = []
        groups: dict[tuple[str, int], list[tuple[str, int, str, int, int, float, float]]] = {}
        for k in keys:
            groups.setdefault((k[0], k[1]), []).append(k)
        for (v, t), ks in groups.items():
            fds = np.stack(
                [
                    (yy[ip, :e_n] - (yy[im, :e_n] if im >= 0 else y0[:e_n])) / (hp + hm)
                    for (_, _, _, ip, im, hp, hm) in ks
                ]
            )  # [3, E]
            ad = g[:, t, _IX[v]]
            ver = single_day_verdict(ad, list(fds), y0[:e_n], max(cfg.steps[v]), cfg)
            for e, out in enumerate(out_names):
                status_of[(out, v, t)] = str(ver["status"][e])
                sd_rows.append(
                    {
                        "output": out,
                        "variable": v,
                        "day": t,
                        "date": dates[t],
                        "stage": int(stage[t]),
                        "selected": ks[0][2],
                        "value": float(x[t, _IX[v]]),
                        "derivative": float(ad[e]),
                        "derivative_exact": float(g_exact[e, t, _IX[v]]),
                        **{f"fd_{h:g}": float(fds[j, e]) for j, h in enumerate(cfg.steps[v])},
                        "one_sided": any(k[4] < 0 for k in ks),
                        "spread": float(ver["spread"][e]),
                        "rel_err": float(ver["rel_err"][e]),
                        "rel_err_exact": float(
                            abs(g_exact[e, t, _IX[v]] - fds[0, e])
                            / max(abs(g_exact[e, t, _IX[v]]), abs(fds[0, e]), 1e-300)
                        ),
                        "status": str(ver["status"][e]),
                    }
                )
        single = pd.DataFrame(sd_rows)
        n_sd, n_ws = len(sd_deltas), len(ws)
        ws_rows = []
        for j, (lab, d) in enumerate(ws):
            yp, ym = yy[n_sd + j], yy[n_sd + n_ws + j]
            for e, out in enumerate(out_names):
                pred = float(np.sum(g[e] * d))
                plus, minus = float(yp[e] - y0[e]), float(ym[e] - y0[e])
                central = (plus - minus) / 2.0
                ws_rows.append(
                    {
                        "output": out,
                        "perturbation": lab,
                        "predicted": pred,
                        "rerun": central,
                        "gap": central - pred,
                        "gap_rel": (central - pred) / abs(pred) if pred != 0.0 else np.nan,
                        "rerun_plus": plus,
                        "rerun_minus": minus,
                        "curvature": (plus + minus) / 2.0,
                        "adat_shift_plus": int(yp[e_silk] - y0[e_silk]),
                        "adat_shift_minus": int(ym[e_silk] - y0[e_silk]),
                        "mdat_shift_plus": int(yp[e_mat] - y0[e_mat]),
                        "mdat_shift_minus": int(ym[e_mat] - y0[e_mat]),
                        "note": _TEMPERATURE_RERUN_NOTE if "TM" in lab else "",
                    }
                )
        whole = pd.DataFrame(ws_rows)
        summ = []
        for out in out_names:
            for v in vs:
                st = [status_of[(out, v, t)] for (oo, vv, t) in status_of if oo == out and vv == v]
                summ.append(
                    {
                        "output": out,
                        "variable": v,
                        "days": len(st),
                        "pass": st.count(STATUS_PASS),
                        "fail": st.count(STATUS_FAIL),
                        "undecidable": st.count(STATUS_UNDECIDABLE),
                        "label": _with_caveat(v, _variable_label(st)),
                    }
                )
        summary = pd.DataFrame(summ)
        checks = WeatherChecks(single, whole, summary)
        for v in vs:
            st = [s for (oo, vv, _), s in status_of.items() if vv == v]
            labels[v] = _with_caveat(v, _variable_label(st))
    else:
        labels = {v: _with_caveat(v, "not checked") for v in vs}
    # the calendar
    rows = []
    for e, out in enumerate(out_names):
        unit_o = _output_unit(out)
        for t in range(t_cal):
            for v in vs:
                rows.append(
                    {
                        "output": out,
                        "date": dates[t],
                        "yrdoy": yrdoy[t],
                        "day": t,
                        "dap": t - t_plant,
                        "stage": int(stage[t]),
                        "stage_name": STAGE_NAMES.get(int(stage[t]), ""),
                        "variable": v,
                        "value": float(x[t, _IX[v]]),
                        "derivative": float(g[e, t, _IX[v]]),
                        "derivative_exact": float(g_exact[e, t, _IX[v]]),
                        "unit": f"{unit_o} per {VARIABLES[v][0]}",
                        "checked": status_of.get((out, v, t), ""),
                    }
                )
    daily = pd.DataFrame(rows)
    st_rows = []
    for e, out in enumerate(out_names):
        for v in vs:
            gv = g[e, :t_cal, _IX[v]]
            tot_abs = float(np.abs(gv).sum())
            for code, a, b in _stage_runs(stage[:t_cal]):
                seg = gv[a : b + 1]
                st_rows.append(
                    {
                        "output": out,
                        "variable": v,
                        "stage": code,
                        "stage_name": STAGE_NAMES.get(code, ""),
                        "first": dates[a],
                        "last": dates[b],
                        "days": b - a + 1,
                        "sum": float(seg.sum()),
                        "share": float(np.abs(seg).sum()) / tot_abs if tot_abs > 0 else 0.0,
                        "per_pct": float(np.sum(seg * x[a : b + 1, _IX[v]]) / 100.0)
                        if v in _NONNEGATIVE
                        else np.nan,
                    }
                )
    stages = pd.DataFrame(st_rows)
    silk = int(base[e_silk])
    info = {
        "start": dates[0],
        "planting": dates[t_plant],
        "silking": dates[silk] if silk < n_days else None,
        "maturity": dates[min(t_mat, n_days - 1)],
        "matured": t_mat < n_days,
        "days": t_cal,
    }
    timing = {
        "gradient_s": grad_s,
        "gradient_exact_s": grad_exact_s,
        "check_rows": n_check_rows,
        "check_s": check_s,
        "wall_s": time.perf_counter() - t0,
        "devices": progs.many.ndev,
    }
    notes = [
        f"temperature derivatives: {TEMPERATURE_CAVEAT} (compare ws.phenology_free()).",
        "labels hold for the days checked at the check's steps (ws.checks.single_day; ws.brute_force() "
        "reruns every day).",
    ]
    if "RAIN" in vs or "IRRD" in vs:
        notes.append(
            "rain and irrigation pass the runoff curve number, infiltration and drainage thresholds: see "
            "the single-day verdicts of RAIN / IRRD."
        )
    ctx = _Context(
        progs, i, g, t_cal, stage, dates, yrdoy, e_mat, e_silk, cfg, [Path(f) for f in weather_files]
    )
    return WeatherSensitivity(
        name,
        list(out_names),
        list(vs),
        {o: float(base[e]) for e, o in enumerate(out_names)},
        daily,
        stages,
        labels,
        checks,
        timing,
        info,
        notes,
        ctx,
    )


def weather_sensitivity(
    runs: Sequence[Any],
    names: Sequence[str],
    *,
    outputs: Sequence[str] | str = ("HWAM",),
    variables: Sequence[str] | str = DEFAULT_VARIABLES,
    check: bool = True,
    config: WeatherTrustConfig | None = None,
    weather_files: Sequence[str | Path] = (),
    owner: Any = None,
    key: Any = None,
) -> list[WeatherSensitivity]:
    """The weather sensitivity of every run of ``runs`` (:class:`~agrijax.sites.dssat_free_run.FreeRunInputs`
    of one soil layering and evaporation method; ``names`` their names): one reverse-mode pass per
    gradient mode over all of them, the checks batched (module docstring). ``weather_files``: the
    station's ``.WTH`` files, for the default climatology of :meth:`WeatherSensitivity.attribution`.
    ``owner`` / ``key``: keep the compiled programs on ``owner`` (a second call does not compile)."""
    from agrijax.calib.dssat_day import E_DATE, ISTAGE_SILKING, Entry
    from agrijax.dssat import _x64
    from agrijax.facade_execution import require_default_options
    from agrijax.facade_grad import _entries, _parse_outputs

    require_default_options("weather_sensitivity")
    _x64()
    if len(runs) != len(names) or not runs:
        raise ValueError(f"{len(runs)} runs, {len(names)} names")
    out_names = _parse_outputs(outputs)
    vs = _parse_variables(variables)
    cfg = config or DEFAULT_CONFIG
    ents = [[*_entries(out_names, r), Entry(E_DATE, ISTAGE_SILKING)] for r in runs]
    progs = _cached(owner, (key, tuple(out_names)), lambda: _Programs(runs, ents, len(out_names)))
    c0 = progs.compile_s
    scn = list(range(len(runs)))
    _, g, grad_s = progs.gradient(scn, "ste")
    _, g_x, grad_x_s = progs.gradient(scn, "exact")
    compile_grad = progs.compile_s - c0
    out = []
    for i, r in enumerate(runs):
        res = _analyse(
            progs,
            i,
            r,
            names[i],
            out_names,
            vs,
            g[i],
            g_x[i],
            grad_s,
            grad_x_s,
            cfg,
            check,
            [Path(f) for f in weather_files],
        )
        res.timing["compile_s"] = compile_grad
        res.timing["scenarios_in_pass"] = len(runs)
        out.append(res)
    return out


def station_files(directory: str | Path, station: str) -> list[Path]:
    """The station's yearly weather files ``<station><YY>01.WTH`` in ``directory`` (sorted)."""
    return sorted(Path(directory).glob(f"{station}??01.WTH"))


class WeatherSensitivities(list):  # type: ignore[type-arg]
    """The results of :meth:`agrijax.dssat.Scenarios.weather_sensitivity`, one per scenario (a list);
    index by position, or by year (``res[1982]``) where the year is unique."""

    years: list[int]

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, int) and k >= 1000 and k in getattr(self, "years", []):
            hits = [i for i, y in enumerate(self.years) if y == k]
            if len(hits) != 1:
                raise KeyError(f"year {k} names {len(hits)} scenarios: index by position")
            return super().__getitem__(hits[0])
        return super().__getitem__(k)
