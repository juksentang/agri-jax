"""The weather sensitivity calendar: gradients of season outputs with respect to every day's weather.

The user code this module enables (continuing the quick start of :mod:`agrijax.dssat`)::

    ws = exp.weather_sensitivity(treatment=2, outputs=["HWAM", "CWAM"],
                                 variables=["SRAD", "TMAX", "TMIN", "RAIN"])
    print(ws)                  # season totals, the most sensitive stage, a trust label per variable
    ws.daily                   # one row per (output, day, variable): weather, d output / d weather, stage
    ws.calendar("HWAM")        # the same as a day x variable table
    ws.stages                  # per growth stage (boundaries from the run): summed derivative, share
    ws.trust                   # {"SRAD": "validated gradient (...)", "RAIN": "not validated: 9 of 12 ..."}
    ws.checks.summary          # per output and variable: the shares of checked days at trust level 3 / 2 / 1
    ws.checks.single_day       # per-day perturbation reruns of the top-k days and random days, levels 2 and 3
    ws.checks.whole_season     # SRAD +-5 %, TMAX / TMIN +-1 degC, RAIN +-10 %: summed gradient vs rerun
    ws.phenology_free([-2, -1, 1, 2])  # companion: whole-season temperature offsets run free, batched
    att = ws.attribution()     # first-order climate attribution: gradient x this season's weather anomaly
    att.by_stage, att.total    #   (climatology from the station's weather files, or climatology= / anomaly=)
    ws.brute_force()           # the single-day check on every day; check=False: plain per-day reruns
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
* ``IRRD`` [mm] - the **effective** irrigation of that day, the amount reaching the soil (``IRRAMT``:
  the applied amount x the FileX efficiency ``EFIR``, as ``WATBAL`` reads it): divide by the efficiency
  for a derivative per applied mm. A management input rather than weather.

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

**Trust check** (separate from the coefficient trust report, with its levels and its comparisons,
:func:`agrijax.calib.trust.fd_diagnostics`; :class:`WeatherTrustConfig`):

(a) single-day reruns: for each variable the ``top_k`` days of largest ``|d output / d variable|`` (first
    output) and ``random_days`` random season days are rerun with that day's value changed by three small
    steps (1e-3 / 1e-4 / 1e-5 units) and three user-scale steps (``SRAD`` 0.1 / 0.5 / 1 MJ m-2, temperatures
    0.1 / 0.5 / 1 degC, ``RAIN`` / ``IRRD`` 1 / 2 / 5 mm). One difference scheme per day: central at every
    step, or forward at every step where the value minus the largest step would go below 0 (column
    ``one_sided``). **Level 2** (three-valued, on the exact-mode derivative): the small-step differences agree
    with each other (to 1e-3) and with the derivative -> pass, with each other but not with it -> fail, not
    with each other -> undecidable (a quantum or a threshold inside 1e-3 units). A threshold **at** the
    point (a weather value exactly on it, e.g. ``TMAX`` = 35.0 degC, the ``PETPT`` advection threshold:
    weather files hold one decimal) makes every central small-step difference the mean of the two slopes,
    steady across the steps; on central days the one-sided small-step differences are compared as well,
    and where each is steady but they differ level 2 is ``kink`` (undecidable) when the exact derivative is
    one of the two (column ``kink_side``: the side the program takes) and ``fail`` when it is neither. It is
    **not judged** where the exact-mode and straight-through derivatives differ by more than the level-2
    tolerance 1e-3 (a quantiser cuts the exact derivative there, to zero or to a residual), and on every day
    of a variable whose exact-mode derivative is zero on every season day while the straight-through one is
    not (rain and irrigation reach the outputs only through the soil-water rounding): a pass on the exact
    derivative says nothing about the reported one; not judged counts as undecidable, and a level-2 fail
    (outside a kink) is kept. **Level 3** (on the straight-through derivative, the one reported): it agrees
    with the secant at every user step (to 5 %), else the day fails: curvature, a threshold or a stage day
    inside a user step is a failure, not an undecidable case. The day's trust level is that of
    :mod:`agrijax.calib.trust` (3: level 3 passes and level 2 does not fail; 2: level 2 passes, level 3 fails;
    1: level 2 fails, or is undecidable / not judged and level 3 fails); a day passes at level 3. Both zero
    (below ``abs_floor``) counts as agreeing; a day where both derivatives and every difference are zero has
    status ``zero`` (no response: neither a pass nor a failure). Rainfall passes the runoff curve number,
    infiltration and the drainage thresholds: its outcome is reported as found;
(b) whole-season reruns: ``SRAD`` x(1 +- 5 %), ``TMAX`` / ``TMIN`` (and both) +- 1 degC, ``RAIN`` and
    ``IRRD`` x(1 +- 10 %) on every simulated day, against the summed gradient ``sum_t d output / d x_t
    * dx_t``; the gap (rerun central difference minus prediction) and its curvature part are reported,
    with the silking and maturity shifts of the reruns (the temperature gap includes the stage days
    moving, which the gradient excludes by construction);

``checks.summary`` gives, per output and variable, the zero days and, over the others, the shares at
level 3 / 2 / 1 and of the level-2 outcomes. A variable's label counts distinct days (a day fails when
any output fails there): ``"inert: ..."`` when every checked day is zero (no response, nothing
validated), ``"validated gradient"`` when no checked day fails, else ``"not validated: k of n checked
days fail"`` (with the caveats). The label is about the days checked;
:meth:`WeatherSensitivity.brute_force` runs the same check on every day.
Rain and irrigation reach the calendar through germination (the seed layer's water): their labels
carry :data:`WATER_CAVEAT`.

**Attribution** (:meth:`WeatherSensitivity.attribution`): ``d output / d x_t * (x_t - climatology_t)``
per day and variable, summed by stage and over the season: a first-order estimate of how much each
day's departure from the climatology moved the output. The climatology is the day-of-year mean of the
station's weather files without the season's own years (or what the user passes), and the rerun on
the climatological weather (each variable alone and all together) shows how far first order goes.
Every row carries :data:`LINEAR_CAVEAT` and the phenology / germination caveats of its variable; rain
anomalies of a daily climatology (a little rain every day against storms) are far outside a linear
range, which the rerun makes visible.

Runs in float64 on the host's JAX devices (``XLA_FLAGS=--xla_force_host_platform_device_count=<cores>``
for the batched reruns; the gradient pass runs on one device).

**Measured** (rorqual, ``scripts/diag/g1_weather_sensitivity.py``; UFGA8201 t2 / t4 in 1978-1987 and the
CA-TPA seasons 2015-2021: 27 seasons, 1468 checked day x variable rows of ``HWAM``, 127 of them with no
response). Where the season is not water limited (UFGA8201 t4 1982, CA-TPA 2021) the straight-through and
exact derivatives are equal, every checked ``SRAD`` day with a response is at trust level 3 and ``RAIN`` /
``IRRD`` are ``inert`` on the checked days (no response on 4 sampled days each; not "validated"). The
same check on every day (:meth:`WeatherSensitivity.brute_force`) puts, on UFGA8201 t4 1982, 92 % of the
``SRAD`` days with a response and 61-62 % of the temperature days at level 3 (the others at level 2: a
stage day or curvature inside the 0.1-1 degC steps), ``RAIN`` inert on all 130 days and one ``IRRD`` day
with a response (level 2), so ``IRRD`` "inert" holds for the sampled days only; on the water-limited UFGA8201
t2 1982 it puts 12 % (``SRAD``), 17 % (``TMAX``), 17 % (``TMIN``) and 0 % (``RAIN``) of the days with a
response at level 3. In water-limited seasons the response to one day's weather is a staircase (the
soil-water rounding, the ``TURFAC`` and root-length-density truncations, the runoff and drainage thresholds)
and the exact derivative is cut by those quantisers: of the **checked** days with a response (the top
``|derivative|`` days plus 4 random ones: a sample weighted to the largest derivatives, not an estimate over
all days; on t2 1982 it gives 9 / 27 / 50 % against the brute force's 12 / 17 / 17 %) 32 % (``SRAD``), 37 % /
40 % (``TMAX`` / ``TMIN``) and 2-4 % (``RAIN``, ``IRRD``) are at level 3, 0-11 % at level 2, the labels say
"not validated". Of the 1341 rows with a response, level 2 passes on 271 (all ``SRAD`` / ``TMAX`` /
``TMIN``, on days where the reported derivative equals the exact one to the level-2 tolerance 1e-3: there
the small steps test the reported derivative itself), is not judged on 1069 (the two derivatives differ by
more than 1e-3: on 159 of the 919 ``SRAD`` / ``TMAX`` / ``TMIN`` rows by 1e-3 to 5 %, on 488 by more) and is
``kink`` on 1; none fails. Two rows sit on a kink at the point: ``TMAX`` exactly 35.0 degC, the ``PETPT``
advection threshold (``TMAX .GT. 35.0``), UFGA8201 t4 1987 day 115 and t2 1980 day 113. There ``EO`` is
``1.1 EEQ`` on the left and ``EEQ ((TMAX - 35) 0.05 + 1.1)`` on the right, the exact derivative is the
left-hand one (-11.118 and -0.233 kg ha-1 per degC, equal to the backward small-step differences), the
forward differences give -12.557 and -1.232 (a warmer day), and the central differences their mean
(-11.837, -0.732) at every step; nothing is missing from the derivative. 94 of the 3846 calendar ``TMAX``
season days are exactly 35.0 (``daily["caveat"]``); on most of them ``EO`` does not reach the output that day
and the two sides agree. For ``RAIN`` and ``IRRD`` the exact derivative is zero on every day (cut by the
soil-water rounding), so their level 2 is not judged and their reported straight-through derivative is
tested at the user steps only, where it mostly fails: it is not shown correct.
Cost on one device: one call's two reverse passes (straight-through and exact) 0.13-0.26 s (first call) or
0.16-0.19 s (warm repeat) for 190 simulated days (130-day season and padding) x 5 variables x 2 outputs,
against 1.03-1.20 s for the 521-651 equivalent per-day reruns batched on the same device: 4.2-9.2x over four
runs; one forward season 0.008-0.015 s.
"""

from __future__ import annotations

import datetime as _dt
import numbers
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DAY_STATUSES",
    "LABEL_FAILS",
    "LABEL_INERT",
    "LABEL_VALIDATED",
    "LINEAR_CAVEAT",
    "STAGE_NAMES",
    "STATUS_NOT_JUDGED",
    "STATUS_ZERO",
    "TEMPERATURE_CAVEAT",
    "VARIABLES",
    "WATER_CAVEAT",
    "Attribution",
    "WeatherChecks",
    "WeatherSensitivities",
    "WeatherSensitivity",
    "WeatherTrustConfig",
    "climatology",
    "day_verdict",
    "station_files",
    "weather_sensitivity",
]

#: the daily inputs that can be differentiated: name -> (unit, meaning), in the perturbation's column order
VARIABLES: dict[str, tuple[str, str]] = {
    "SRAD": ("MJ m-2 d-1", "solar radiation"),
    "TMAX": ("degC", "maximum air temperature"),
    "TMIN": ("degC", "minimum air temperature"),
    "RAIN": ("mm", "rainfall"),
    "IRRD": ("mm", "effective irrigation (amount x efficiency)"),
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
#: the water inputs, and their caveat: they reach the calendar through germination (seed-layer water)
WATER: tuple[str, ...] = ("RAIN", "IRRD")
WATER_CAVEAT = "germination day fixed; excludes earlier/later germination"
#: the caveat of every attribution row
LINEAR_CAVEAT = "first-order (linearised) estimate; compare with the rerun"
_TEMPERATURE_RERUN_NOTE = f"rerun includes earlier/later development; the gradient: {TEMPERATURE_CAVEAT}"

LABEL_VALIDATED = "validated gradient"
LABEL_FAILS = "not validated"
#: a variable whose derivative and every difference are zero on every checked day: no response, nothing
#: validated (``calib/trust.py``'s ``inert`` class)
LABEL_INERT = "inert"
STATUS_PASS, STATUS_FAIL, STATUS_UNDECIDABLE = "pass", "fail", "undecidable"
#: level 2 does not test the reported derivative: the exact-mode derivative differs from the
#: straight-through one (cut by a quantiser on the path, to zero or to a residual), or is zero on every
#: day of the season
STATUS_NOT_JUDGED = "not judged"
#: level 2 at a kink at the point (a weather value exactly on a model threshold): the small-step
#: one-sided differences are each steady but differ, and the exact derivative equals one of them (the
#: one-sided derivative the program takes); counted as undecidable (level 3 decides)
STATUS_KINK = "kink"
#: calendar caveats of a ``TMAX`` day exactly on a ``PETPT`` threshold (``daily["caveat"]``)
TMAX_HOT_CAVEAT = (
    "TMAX exactly on the PETPT advection threshold (35 degC, PET.for 907 'TMAX .GT. 35.0'): EO has a kink "
    "there; where EO reaches the output that day the derivative is the left-hand one (a cooler day) and a "
    "warmer day's slope is steeper"
)
TMAX_COLD_CAVEAT = (
    "TMAX exactly on the PETPT cold threshold (5 degC, PET.for 909 'TMAX .LT. 5.0'): EO jumps just below "
    "it; where EO reaches the output that day the derivative is the right-hand one (a warmer day)"
)
#: a checked day whose derivatives and differences are all zero (below ``abs_floor``): no response
STATUS_ZERO = "zero"
#: checked-day statuses (``single_day["status"]``, ``daily["checked"]``)
DAY_STATUSES: tuple[str, ...] = (STATUS_PASS, STATUS_FAIL, STATUS_ZERO)
#: smallest ``YYYYDDD`` date (year 1000) and smallest year: an index at or above them is a date / a year,
#: not a position
_YYYYDDD_MIN = 1000001
_YEAR_MIN = 1000

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
    """Steps and thresholds of the weather trust check (harness choices, fixed in advance), the two
    levels of :mod:`agrijax.calib.trust` (:func:`~agrijax.calib.trust.fd_diagnostics` makes the verdicts):

    * ``small_steps`` - level 2: three adjacent small steps of each variable, in its unit; their
      differences must agree with each other to ``small_rtol`` to judge the exact-mode derivative
      (agreeing to ``small_rtol``: pass, else fail), and are *undecidable* when they do not;
    * ``steps`` - level 3: the three user-scale steps; the straight-through derivative must agree with
      the secant at **every** one of them to ``rtol``, else the day fails (curvature, a threshold or a
      stage day inside the step);
    * ``abs_floor`` - a derivative or difference with ``|d| * largest step`` below ``abs_floor *
      max(|y|, 1)`` counts as zero: it moves the output by less than a millionth of itself over the
      largest step (looser than the coefficient check's 1e-10, :class:`agrijax.calib.trust.TrustConfig`:
      a straight-through path through a quantiser leaves derivatives of 1e-5 kg ha-1 per mm on days
      where the model does not move at all);
    * ``top_k``, ``random_days``, ``seed`` - the days checked: the largest ``|derivative|`` of the first
      output, and random season days;
    * ``season`` - the whole-season perturbations: relative for ``SRAD`` / ``RAIN`` / ``IRRD``, in degC
      for the temperatures.
    """

    small_steps: Mapping[str, tuple[float, float, float]] = field(
        default_factory=lambda: {v: (1e-3, 1e-4, 1e-5) for v in ("SRAD", "TMAX", "TMIN", "RAIN", "IRRD")}
    )
    steps: Mapping[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "SRAD": (0.1, 0.5, 1.0),
            "TMAX": (0.1, 0.5, 1.0),
            "TMIN": (0.1, 0.5, 1.0),
            "RAIN": (1.0, 2.0, 5.0),
            "IRRD": (1.0, 2.0, 5.0),
        }
    )
    small_rtol: float = 1e-3
    rtol: float = 0.05
    abs_floor: float = 1e-6
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


def _caveats(v: str) -> list[str]:
    out = []
    if v in PHENOLOGY or v == "ALL":
        out.append(TEMPERATURE_CAVEAT)
    if v in WATER or v == "ALL":
        out.append(WATER_CAVEAT)
    return out


def _with_caveat(v: str, label: str) -> str:
    c = _caveats(v)
    return f"{label} ({'; '.join(c)})" if c else label


# ------------------------------------------------------------------------------ verdicts (no model)
def _pt_thresholds() -> tuple[float, float]:
    """The ``TMAX`` values of the ``PETPT`` thresholds (``hot_threshold`` / ``cold_threshold`` of
    :data:`agrijax.processes.pet.coefficients.DSSAT_PT`): ``EO`` has a kink there and, the comparisons
    being strict (``TMAX .GT. 35.0`` / ``TMAX .LT. 5.0``), the derivative is the left-hand one."""
    from agrijax.processes.pet.coefficients import DSSAT_PT

    return float(DSSAT_PT.hot_threshold), float(DSSAT_PT.cold_threshold)


def day_verdict(
    ad: Any,
    ad_exact: Any,
    small: Sequence[Any],
    large: Sequence[Any],
    y0: Any,
    h_scale: Any,
    cfg: WeatherTrustConfig = DEFAULT_CONFIG,
    judged: Any = None,
    right: Sequence[Any] | None = None,
    left: Sequence[Any] | None = None,
) -> dict[str, np.ndarray]:
    """The verdicts of days ``[E, D]`` (outputs x days): ``ad`` / ``ad_exact`` the straight-through and
    exact-mode derivatives, ``small`` / ``large`` the differences at the three small and the three
    user-scale steps (lists of ``[E, D]``), ``y0`` the outputs ``[E]``, ``h_scale`` the largest user step
    of each day ``[D]``. The comparisons are :func:`agrijax.calib.trust.fd_diagnostics` (level 2
    three-valued on ``ad_exact``; level 3 = ``ad`` agrees with every user-step secant); the day's
    ``level`` follows :mod:`agrijax.calib.trust`: 1 if level 2 fails, or is undecidable and level 3
    fails; 2 if level 2 passes and level 3 fails; 3 if level 3 passes (and level 2 does not fail).

    ``right`` / ``left`` (lists of ``[E, D]``, NaN on forward-only days): the one-sided differences
    ``(y(x + h) - y(x)) / h`` and ``(y(x) - y(x - h)) / h`` at the small steps. A **kink at the point**
    (a weather value exactly on a model threshold, e.g. ``TMAX`` = 35.0 degC, the ``PETPT`` advection
    threshold) makes the central small-step differences ``(right + left) / 2`` at every step, so they
    agree with each other and say nothing about either side. Where each side agrees with itself across
    the small steps (to ``cfg.small_rtol``) and the two sides differ (beyond ``cfg.small_rtol``), level 2
    is :data:`STATUS_KINK` (counted as undecidable: level 3 decides) when the exact derivative equals one
    side (``kink_side`` = ``"left"`` / ``"right"``: the one-sided derivative the program takes), and
    ``fail`` when it equals neither (``kink_side`` = ``"neither"``).

    Level 2 is **not judged** (:data:`STATUS_NOT_JUDGED`, counted as undecidable: level 3 decides) where
    the straight-through and exact-mode derivatives differ by more than the level-2 tolerance
    ``cfg.small_rtol`` (the exact one is cut by a quantiser, to zero or to a residual: a level-2 pass on
    it says nothing about the reported derivative), and wherever ``judged`` (``[E, D]`` bool) is False
    (the caller's: the exact derivative is zero on every day of the season). A level-2 **fail** is kept
    (outside a kink at the point: the small steps show the exact derivative wrong on both sides, whatever
    is reported).
    ``status`` is ``zero`` (:data:`STATUS_ZERO`) where both derivatives and every difference are zero
    (no response: neither a pass nor a failure), else ``pass`` at level 3 and ``fail`` otherwise."""
    from agrijax.calib.trust import _TINY, L2_FAIL, L2_UNDECIDABLE, TrustConfig, fd_diagnostics

    # the step values only index fd0 / fd1 here (the differences are given): positions 1 and 2
    tcfg = TrustConfig(
        fd_steps=(2.0, 20.0),
        fd_small_steps=(1.0, 2.0, 3.0),
        fd_large_steps=(10.0, 20.0, 30.0),
        fd_rtol=(cfg.small_rtol, cfg.rtol),
        abs_floor=cfg.abs_floor,
    )
    d = fd_diagnostics(ad, ad_exact, list(small), list(large), y0, h_scale, tcfg)
    ad_a, adx_a = np.abs(np.asarray(ad, dtype=float)), np.abs(np.asarray(ad_exact, dtype=float))
    floor = cfg.abs_floor * np.maximum(np.abs(np.asarray(y0, dtype=float)), 1.0)[:, None]
    floor = floor / np.asarray(h_scale, dtype=float)[None, :]
    status0 = np.asarray(d["status0"], dtype=object)
    adx = np.asarray(ad_exact, dtype=float)
    shape = np.shape(status0)
    nan = np.full(shape, np.nan)
    kink = np.zeros(shape, dtype=bool)
    side = np.full(shape, "", dtype=object)
    fd_r, fd_l = nan, nan
    if right is not None and left is not None:
        sr = np.stack([np.asarray(a, dtype=float) for a in right])
        sl = np.stack([np.asarray(a, dtype=float) for a in left])
        # the middle small step, as fd_small (calib/trust.py: fd_steps[0] = the second small step)
        fd_r, fd_l = sr[1], sl[1]

        def steady(f: np.ndarray) -> np.ndarray:
            ok = np.all(np.isfinite(f), axis=0)
            fz = np.where(np.isfinite(f), f, 0.0)
            spread = (fz.max(axis=0) - fz.min(axis=0)) / np.maximum(np.abs(fz).max(axis=0), _TINY)
            return ok & (np.all(np.abs(fz) <= floor[None], axis=0) | (spread <= cfg.small_rtol))

        def near(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            m = np.maximum(np.abs(a), np.abs(b))
            return (m <= floor) | (np.abs(a - b) <= cfg.small_rtol * m)

        fr0, fl0 = np.where(np.isfinite(fd_r), fd_r, 0.0), np.where(np.isfinite(fd_l), fd_l, 0.0)
        kink = steady(sr) & steady(sl) & ~near(fr0, fl0)
        on_l, on_r = near(adx, fl0), near(adx, fr0)
        side = np.where(kink, np.where(on_l, "left", np.where(on_r, "right", "neither")), "").astype(object)
        status0 = np.where(kink, np.where(on_l | on_r, STATUS_KINK, L2_FAIL), status0).astype(object)
    # level 2 judges the exact derivative: it says something about the reported one only where the two
    # agree to the level-2 tolerance (both below the floor agree). A failing level 2 is kept (it shows
    # the exact derivative wrong, whatever is reported)
    differ = (np.maximum(ad_a, adx_a) > floor) & (
        np.abs(np.asarray(ad, dtype=float) - adx) > cfg.small_rtol * np.maximum(ad_a, adx_a)
    )
    cut = differ
    if judged is not None:
        cut = cut | ~np.asarray(judged, dtype=bool)
    cut = cut & (status0 != L2_FAIL)
    fds = np.abs(np.stack([np.asarray(f, dtype=float) for f in (*small, *large)]))
    zero = (ad_a <= floor) & (adx_a <= floor) & np.all(fds <= floor[None], axis=0)
    l2 = np.where(cut, STATUS_NOT_JUDGED, np.where(status0 == L2_UNDECIDABLE, STATUS_UNDECIDABLE, status0))
    l3 = np.asarray(d["agree1"], dtype=bool)
    open_l2 = (l2 == STATUS_UNDECIDABLE) | (l2 == STATUS_NOT_JUDGED) | (l2 == STATUS_KINK)
    level = np.where(l2 == L2_FAIL, 1, np.where(l3, 3, np.where(open_l2, 1, 2)))
    return {
        "level2": l2.astype(object),
        "level3": l3,
        "level": level,
        "status": np.where(zero, STATUS_ZERO, np.where(level == 3, STATUS_PASS, STATUS_FAIL)).astype(object),
        "fd_small": d["fd0"],
        "fd_small_right": fd_r,
        "fd_small_left": fd_l,
        "kink_side": side,
        "rel_err_small": d["rel_err0"],
        "spread_small": d["spread0"],
        "rel_err_large_max": d["rel_err1_max"],
    }


def _variable_label(rows: Any) -> str:
    """The label of one variable from its checked rows (``day``, ``level``, ``status``; every output):
    counted in **distinct days** (a day fails when any output fails there). ``inert`` when every row is
    ``zero`` (no response: nothing to validate); ``validated gradient`` when no day fails."""
    day = [int(x) for x in rows["day"]]
    lv = [int(x) for x in rows["level"]]
    st = [str(x) for x in rows["status"]]
    days = set(day)
    if all(s == STATUS_ZERO for s in st):
        return f"{LABEL_INERT}: zero derivative and no response on all {len(days)} checked days"
    bad = {t for t, x, s in zip(day, lv, st, strict=True) if s != STATUS_ZERO and x != 3}
    return LABEL_VALIDATED if not bad else f"{LABEL_FAILS}: {len(bad)} of {len(days)} checked days fail"


def _day_caveat(v: str, value: float) -> str:
    """The calendar caveat of one day's value: a ``TMAX`` exactly on a ``PETPT`` threshold."""
    if v != "TMAX":
        return ""
    hot, cold = _pt_thresholds()
    return TMAX_HOT_CAVEAT if value == hot else TMAX_COLD_CAVEAT if value == cold else ""


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


#: :func:`climatology` under a name the ``climatology=`` argument of ``attribution`` does not shadow
_station_climatology = climatology


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
    output, variable and day checked: the differences, the level-2 and level-3 outcomes, the trust level),
    :attr:`whole_season` (one row per output and perturbation), :attr:`summary` (per output and variable:
    the shares of the checked days at trust level 3 / 2 / 1 and of the level-2 outcomes, the label)."""

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
    ``checked`` (``pass`` / ``fail`` where the single-day check reran that day), ``caveat`` (a ``TMAX``
    exactly on a ``PETPT`` threshold: at 35.0 degC the derivative is the left-hand one, a warmer day's
    slope is steeper, :data:`TMAX_HOT_CAVEAT`; at 5.0 the right-hand one). :attr:`stages` sums it
    per growth stage: ``sum`` (the change of the output for +1 unit on every day of the stage),
    ``share`` (of the season's sum of ``|derivative|``), ``per_pct`` (for +1 % of the stage's values:
    radiation, rain, irrigation). :attr:`values` are the outputs, :attr:`trust` the label of each
    variable, :attr:`checks` the trust check, :attr:`timing` the cost."""

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
    @staticmethod
    def _ready(what: str) -> None:
        """As the main call: refuse a non-default ``aj.options`` block, and run in float64 (a result made
        inside ``with aj.options():`` keeps working after the block put ``jax_enable_x64`` back)."""
        from agrijax.dssat import _x64
        from agrijax.facade_execution import require_default_options

        require_default_options(what)
        _x64()

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

        self._ready("weather_sensitivity(...).phenology_free")
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
        *,
        check: bool = True,
        step: Mapping[str, float] | None = None,
        one_device: bool = False,
    ) -> Any:
        """Every day from the start to maturity x every variable rerun (batched).

        ``check=True``: the single-day check of :attr:`checks` on **every** day, with the same steps,
        difference scheme and verdict (:func:`day_verdict`): one row per (output, variable, day) with the
        columns of ``checks.single_day``; ``attrs["shares"]`` the level shares per output and variable.
        ``check=False``: the plain per-day perturbation approach, one forward rerun per day at ``step``
        (default the middle user step), the difference and the derivative only (for the cost comparison).
        ``attrs``: rows run and their wall time; ``one_device``: on one device, as the gradient pass."""
        import pandas as pd

        self._ready("weather_sensitivity(...).brute_force")
        ctx = self._ctx
        vs = list(self.variables if variables is None else variables)
        which = "one" if one_device else "many"
        x = ctx.progs.weather_of(ctx.i)
        if check:
            days = [(v, t, "all") for v in vs for t in range(ctx.t_cal)]
            deltas, keys = _check_rows(days, x, ctx.progs.n_days, ctx.cfg)
            d_rows, s_rows = self._delta_rows(deltas)
            yy, secs = ctx.progs.values(d_rows, s_rows, which)
            res = _check_results(keys, yy, ctx.g, ctx.g_exact, len(self.outputs), ctx.cfg, ctx.t_cal)
            out = pd.DataFrame(
                _day_rows(keys, res, self.outputs, x, ctx.g, ctx.g_exact, ctx.dates, ctx.stage, ctx.cfg)
            )
            out.attrs["shares"] = _shares(out, ["output", "variable"])
        else:
            st = {v: ctx.cfg.steps[v][1] for v in vs} | dict(step or {})
            deltas = [np.zeros((ctx.progs.n_days, len(_VORDER)))]
            klist = []
            for v in vs:
                for t in range(ctx.t_cal):
                    d = np.zeros((ctx.progs.n_days, len(_VORDER)))
                    d[t, _IX[v]] = st[v]
                    deltas.append(d)
                    klist.append((v, t))
            d_rows, s_rows = self._delta_rows(deltas)
            y, secs = ctx.progs.values(d_rows, s_rows, which)
            rows = []
            for e, o in enumerate(self.outputs):
                for j, (v, t) in enumerate(klist):
                    rows.append(
                        {
                            "output": o,
                            "day": t,
                            "date": ctx.dates[t],
                            "stage": int(ctx.stage[t]),
                            "variable": v,
                            "step": st[v],
                            "fd": float((y[j + 1, e] - y[0, e]) / st[v]),
                            "derivative": float(ctx.g[e, t, _IX[v]]),
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
        leave_season_out: bool = True,
        rerun: bool = True,
    ) -> Attribution:
        """First-order climate attribution (module docstring): ``derivative x (weather - climatology)``
        per day for ``SRAD``, ``TMAX``, ``TMIN``, ``RAIN`` (those of :attr:`variables`).

        ``climatology``: a table indexed by the day of a leap year (1-366, :func:`climatology`) with
        those columns; default the day-of-year mean of the station's weather files (``years``: which
        ones; default all of them), **without the season's own years** (``leave_season_out``: an anomaly
        against a mean that contains the season itself is shrunk by about 1 / N). ``anomaly``: instead,
        the anomaly itself (a table with those columns: with a plain 0-based index, one row per day from
        the simulation start; else indexed by ``YYYYDDD`` covering every simulated day, or a
        ``ValueError`` names the missing days). ``rerun``: also run
        the season on the climatological weather (each variable alone, and all together)."""
        import pandas as pd

        if rerun:
            self._ready("weather_sensitivity(...).attribution")
        ctx = self._ctx
        vs = [v for v in self.variables if v in WEATHER]
        if not vs:
            raise ValueError("attribution: none of SRAD, TMAX, TMIN, RAIN among the variables")
        t_cal = ctx.t_cal
        x = ctx.progs.weather_of(ctx.i)[:t_cal]
        used_years: list[int] = []
        if anomaly is not None:
            an_t = pd.DataFrame(anomaly)
            idx = an_t.index
            if isinstance(idx, pd.RangeIndex) and idx.start == 0 and idx.step == 1:
                if len(an_t) < t_cal:
                    raise ValueError(f"anomaly: {len(an_t)} rows, the season has {t_cal} days")
                an = np.stack([np.asarray(an_t[v], dtype=float)[:t_cal] for v in vs], axis=1)
            else:
                try:
                    keys = {int(k) for k in idx}
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        "anomaly: index by YYYYDDD, or a plain 0-based RangeIndex (one row per day from the "
                        "simulation start)"
                    ) from err
                if not all(k >= _YYYYDDD_MIN for k in keys):
                    raise ValueError(
                        "anomaly: index by YYYYDDD (every day from the simulation start), or a plain 0-based "
                        "RangeIndex"
                    )
                missing = [int(d) for d in ctx.yrdoy[:t_cal] if int(d) not in keys]
                if missing:
                    raise ValueError(
                        f"anomaly: {len(missing)} simulated days missing from the YYYYDDD index (first "
                        f"{missing[:5]}); it must cover {ctx.yrdoy[0]}-{ctx.yrdoy[t_cal - 1]}"
                    )
                an = np.stack([np.asarray(an_t.loc[ctx.yrdoy[:t_cal], v], dtype=float) for v in vs], axis=1)
            clim = x[:, [_IX[v] for v in vs]] - an
        else:
            if climatology is None:
                if not ctx.weather_files:
                    raise ValueError("attribution: no weather files known; pass climatology= or anomaly=")
                own = sorted({d.year for d in ctx.dates[:t_cal]}) if leave_season_out else []
                years_l = None if years is None else [int(y) for y in years]
                try:
                    climatology_tab = _station_climatology(ctx.weather_files, years_l, own)
                except ValueError as err:
                    if not own:
                        raise
                    try:  # the same years with the season's own: is leave-one-out the cause?
                        _station_climatology(ctx.weather_files, years_l)
                        loo = True
                    except ValueError:
                        loo = False
                    if not loo:
                        raise
                    asked = "" if years_l is None else f" among years={sorted(years_l)}"
                    raise ValueError(
                        f"attribution: the station files hold only the season's own year(s) {own}{asked}, "
                        "which leave_season_out=True leaves out of the climatology: pass "
                        "leave_season_out=False, years= (other years), climatology= or anomaly="
                    ) from err
            else:
                climatology_tab = pd.DataFrame(climatology)
            used_years = list(climatology_tab.attrs.get("years", []))
            doy = [_leap_doy(d) for d in ctx.dates[:t_cal]]
            clim = np.stack([np.asarray(climatology_tab.loc[doy, v], dtype=float) for v in vs], axis=1)
            an = x[:, [_IX[v] for v in vs]] - clim
        contrib = np.stack([ctx.g[:, :t_cal, _IX[v]] * an[None, :, k] for k, v in enumerate(vs)], axis=-1)
        cav = {v: "; ".join([LINEAR_CAVEAT, *_caveats(v)]) for v in [*vs, "ALL"]}
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
                            "caveat": cav[v],
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
                            "caveat": cav[v],
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
                        "caveat": cav[v],
                    }
                )
        notes = [
            "linear: sum over the days of derivative x (weather - climatology); rerun: the output minus the "
            "output on the climatological weather (phenology free).",
            f"temperature and radiation terms: {TEMPERATURE_CAVEAT}; rain terms: {WATER_CAVEAT}.",
            "rain against a daily climatology (a little rain every day) is far from a small perturbation: "
            "compare the linear and rerun columns before reading the rain terms.",
        ]
        return Attribution(daily, by_stage, pd.DataFrame(tot), used_years, notes)


@dataclass
class _Context:
    """What the companions of one result need: the programs, the run's index, its derivative and its
    calendar."""

    progs: _Programs
    i: int
    g: np.ndarray  # [E, T, V] straight-through derivative
    g_exact: np.ndarray  # [E, T, V] exact-mode derivative
    t_cal: int
    stage: np.ndarray
    dates: list[Any]
    yrdoy: list[int]
    e_mat: int
    e_silk: int
    cfg: WeatherTrustConfig
    weather_files: list[Path]


# ------------------------------------------------------------------------------ the analysis
_CACHE: dict[int, tuple[Any, OrderedDict[Any, _Programs]]] = {}
#: compiled program sets kept per owner (least recently used dropped first): each holds its inputs on
#: the devices and its executables; a sweep over years belongs in one ``scenarios(...)`` call
CACHE_PER_OWNER = 4


def _cached(owner: Any, key: Any, build: Callable[[], _Programs]) -> _Programs:
    """The programs of ``key`` kept on ``owner`` (an experiment or a scenario set) while it lives, at
    most :data:`CACHE_PER_OWNER` keys per owner (the least recently used is dropped; results made from
    it keep their own reference)."""
    if owner is None:
        return build()
    k = id(owner)
    hit = _CACHE.get(k)
    if hit is None or hit[0]() is not owner:
        store: OrderedDict[Any, _Programs] = OrderedDict()
        _CACHE[k] = (weakref.ref(owner, lambda _r, k=k: _CACHE.pop(k, None)), store)
    else:
        store = hit[1]
    if key in store:
        store.move_to_end(key)
    else:
        store[key] = build()
        while len(store) > CACHE_PER_OWNER:
            store.popitem(last=False)
    return store[key]


def _parse_variables(variables: Sequence[str] | str) -> list[str]:
    vs = [variables] if isinstance(variables, str) else [str(v).upper() for v in variables]
    bad = [v for v in vs if v not in VARIABLES]
    if bad or not vs or len(set(vs)) != len(vs):
        raise ValueError(f"variables {vs}: distinct names out of {list(VARIABLES)}")
    return vs


def _date(yrdoy: int) -> _dt.date:
    return _dt.date(yrdoy // 1000, 1, 1) + _dt.timedelta(days=yrdoy % 1000 - 1)


def _check_days(
    g0: np.ndarray, vs: Sequence[str], t_cal: int, cfg: WeatherTrustConfig
) -> list[tuple[str, int, str]]:
    """The days of the single-day check: per variable the top-k ``|derivative|`` days (nonzero) of the
    first output, then random season days: ``(variable, day, "top" | "random")``. Each variable draws
    from its own generator (seeded by ``cfg.seed`` and the variable), so its days do not depend on which
    other variables are checked."""
    out: list[tuple[str, int, str]] = []
    for v in vs:
        rng = np.random.default_rng([cfg.seed, _IX[v]])
        score = np.abs(g0[:t_cal, _IX[v]])
        order = [int(t) for t in np.argsort(-score, kind="stable") if score[t] > 0][: cfg.top_k]
        rest = [t for t in range(t_cal) if t not in order]
        rnd = sorted(int(t) for t in rng.choice(rest, size=min(cfg.random_days, len(rest)), replace=False))
        out += [(v, t, "top") for t in order] + [(v, t, "random") for t in rnd]
    return out


def _check_rows(
    days: Sequence[tuple[str, int, str]], x: np.ndarray, n_days: int, cfg: WeatherTrustConfig
) -> tuple[list[np.ndarray], dict[tuple[str, int], dict[str, Any]]]:
    """The perturbation rows of ``days`` (the base row first): per day three small and three user-scale
    steps, plus and minus rows. **One difference scheme per day**: where the value minus the largest
    user step would go below 0 (``SRAD``, ``RAIN``, ``IRRD``), every step of that day is a forward
    difference (no minus rows), else every step is central."""
    deltas = [np.zeros((n_days, len(_VORDER)))]
    keys: dict[tuple[str, int], dict[str, Any]] = {}
    for v, t, kind in days:
        one = v in _NONNEGATIVE and float(x[t, _IX[v]]) - max(cfg.steps[v]) < 0.0
        ent: dict[str, Any] = {"kind": kind, "one_sided": one, "small": [], "large": []}
        for grp, hs in (("small", cfg.small_steps[v]), ("large", cfg.steps[v])):
            for h in hs:
                d = np.zeros((n_days, len(_VORDER)))
                d[t, _IX[v]] = h
                deltas.append(d)
                ip, im = len(deltas) - 1, -1
                if not one:
                    d = np.zeros((n_days, len(_VORDER)))
                    d[t, _IX[v]] = -h
                    deltas.append(d)
                    im = len(deltas) - 1
                ent[grp].append((ip, im, float(h)))
        keys[(v, t)] = ent
    return deltas, keys


def _check_results(
    keys: Mapping[tuple[str, int], Mapping[str, Any]],
    yy: np.ndarray,
    g: np.ndarray,
    g_exact: np.ndarray,
    e_n: int,
    cfg: WeatherTrustConfig,
    t_cal: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    """The differences and :func:`day_verdict` of every checked day (``yy`` the rows of
    :func:`_check_rows`, the base row first), per variable in one call: ``(v, t) -> {"fds_large",
    "verdict": {field: [E]}}``. Level 2 is not judged on any day of a variable (and output) whose
    exact-mode derivative is zero on every season day (``t_cal`` days) while the straight-through one is
    not: the exact derivative is cut by a quantiser on every path (rain and irrigation reach the outputs
    only through the soil-water rounding), so level 2 cannot test the reported derivative."""
    y0 = yy[0, :e_n]

    def fd(ip: int, im: int, h: float) -> np.ndarray:
        return (yy[ip, :e_n] - (yy[im, :e_n] if im >= 0 else y0)) / (2.0 * h if im >= 0 else h)

    def fd_right(ip: int, im: int, h: float) -> np.ndarray:
        return (yy[ip, :e_n] - y0) / h if im >= 0 else np.full(e_n, np.nan)

    def fd_left(ip: int, im: int, h: float) -> np.ndarray:
        return (y0 - yy[im, :e_n]) / h if im >= 0 else np.full(e_n, np.nan)

    out: dict[tuple[str, int], dict[str, Any]] = {}
    for v in dict.fromkeys(k[0] for k in keys):
        ts = [t for (vv, t) in keys if vv == v]
        if not ts:
            continue
        small = [np.stack([fd(*keys[(v, t)]["small"][j]) for t in ts], axis=1) for j in range(3)]
        right = [np.stack([fd_right(*keys[(v, t)]["small"][j]) for t in ts], axis=1) for j in range(3)]
        left = [np.stack([fd_left(*keys[(v, t)]["small"][j]) for t in ts], axis=1) for j in range(3)]
        large = [np.stack([fd(*keys[(v, t)]["large"][j]) for t in ts], axis=1) for j in range(3)]
        ad = np.stack([g[:, t, _IX[v]] for t in ts], axis=1)
        adx = np.stack([g_exact[:, t, _IX[v]] for t in ts], axis=1)
        floor = cfg.abs_floor * np.maximum(np.abs(y0), 1.0)[:, None] / max(cfg.steps[v])
        cut = np.all(np.abs(g_exact[:, :t_cal, _IX[v]]) <= floor, axis=1) & np.any(
            np.abs(g[:, :t_cal, _IX[v]]) > floor, axis=1
        )
        judged = np.repeat(~cut[:, None], len(ts), axis=1)
        ver = day_verdict(
            ad, adx, small, large, y0, np.full(len(ts), max(cfg.steps[v])), cfg, judged, right, left
        )
        for k, t in enumerate(ts):
            out[(v, t)] = {
                "fds_large": [f[:, k] for f in large],
                "verdict": {name: arr[:, k] for name, arr in ver.items()},
                "exact_cut": cut,
            }
    return out


def _day_rows(
    keys: Mapping[tuple[str, int], Mapping[str, Any]],
    res: Mapping[tuple[str, int], Mapping[str, Any]],
    out_names: Sequence[str],
    x: np.ndarray,
    g: np.ndarray,
    g_exact: np.ndarray,
    dates: Sequence[Any],
    stage: np.ndarray,
    cfg: WeatherTrustConfig,
) -> list[dict[str, Any]]:
    rows = []
    for (v, t), ent in keys.items():
        r = res[(v, t)]
        ver = r["verdict"]
        for e, out in enumerate(out_names):
            rows.append(
                {
                    "output": out,
                    "variable": v,
                    "day": t,
                    "date": dates[t],
                    "stage": int(stage[t]),
                    "selected": ent["kind"],
                    "value": float(x[t, _IX[v]]),
                    "one_sided": bool(ent["one_sided"]),
                    "derivative": float(g[e, t, _IX[v]]),
                    "derivative_exact": float(g_exact[e, t, _IX[v]]),
                    "fd_small": float(ver["fd_small"][e]),
                    "fd_small_right": float(ver["fd_small_right"][e]),
                    "fd_small_left": float(ver["fd_small_left"][e]),
                    "kink_side": str(ver["kink_side"][e]),
                    "exact_cut": bool(r["exact_cut"][e]),
                    **{f"fd_{h:g}": float(r["fds_large"][j][e]) for j, h in enumerate(cfg.steps[v])},
                    "level2": str(ver["level2"][e]),
                    "rel_err_small": float(ver["rel_err_small"][e]),
                    "level3": bool(ver["level3"][e]),
                    "rel_err_large_max": float(ver["rel_err_large_max"][e]),
                    "level": int(ver["level"][e]),
                    "status": str(ver["status"][e]),
                    "caveat": _day_caveat(v, float(x[t, _IX[v]])),
                }
            )
    return rows


def _shares(table: Any, by: Sequence[str]) -> Any:
    """Per group of ``by``: the rows checked (``days``), how many have no response (``zero_days``:
    derivatives and differences all zero), and over the **others** (``nonzero_days``; NaN when there
    are none) the shares at trust level 3 / 2 / 1 and of the level-2 outcomes (pass / fail /
    undecidable / not judged / kink), with the label they give."""
    import pandas as pd

    rows = []
    for key, grp in table.groupby(list(by), sort=False):
        n = len(grp)
        nz = grp["status"].to_numpy() != STATUS_ZERO
        lv = grp["level"].to_numpy()[nz]
        l2 = grp["level2"].to_numpy()[nz]
        v = key[list(by).index("variable")] if "variable" in by else ""

        def share(m: np.ndarray) -> float:
            return float(np.mean(m)) if m.size else float("nan")

        rows.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "days": n,
                "zero_days": int(n - nz.sum()),
                "nonzero_days": int(nz.sum()),
                "level3": share(lv == 3),
                "level2": share(lv == 2),
                "level1": share(lv == 1),
                "small_pass": share(l2 == STATUS_PASS),
                "small_fail": share(l2 == STATUS_FAIL),
                "small_undecidable": share(l2 == STATUS_UNDECIDABLE),
                "small_not_judged": share(l2 == STATUS_NOT_JUDGED),
                "small_kink": share(l2 == STATUS_KINK),
                "label": _with_caveat(str(v), _variable_label(grp)),
            }
        )
    return pd.DataFrame(rows)


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
        days = _check_days(g[0], vs, t_cal, cfg)
        sd_deltas, keys = _check_rows(days, x, n_days, cfg)
        ws = _whole_season_deltas(vs, x, cfg)
        all_rows = sd_deltas + [d for _, d in ws] + [-d for _, d in ws]
        yy, check_s = progs.values(np.stack(all_rows), np.full(len(all_rows), i))
        n_check_rows = len(all_rows)
        y0 = yy[0]
        res = _check_results(keys, yy[: len(sd_deltas)], g, g_exact, e_n, cfg, t_cal)
        single = pd.DataFrame(_day_rows(keys, res, out_names, x, g, g_exact, dates, stage, cfg))
        for o_, v_, t_, st_ in zip(
            single["output"], single["variable"], single["day"], single["status"], strict=True
        ):
            status_of[(str(o_), str(v_), int(t_))] = str(st_)
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
        summary = _shares(single, ["output", "variable"])
        checks = WeatherChecks(single, whole, summary)
        for v in vs:
            labels[v] = _with_caveat(v, _variable_label(single[single["variable"] == v]))
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
                        "caveat": _day_caveat(v, float(x[t, _IX[v]])),
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
    if checks is not None:
        sd = checks.single_day
        cut_of = {
            v: sorted(set(sd.loc[(sd["variable"] == v) & sd["exact_cut"], "output"].astype(str))) for v in vs
        }
        cut_vs = [f"{v} ({', '.join(o)})" for v, o in cut_of.items() if o]
        if cut_vs:
            notes.append(
                f"level 2 not judged for {'; '.join(cut_vs)}: the exact-mode derivative is zero on every "
                "season day while the straight-through one is not (cut by a quantiser on every path, e.g. "
                "the soil-water rounding); the reported derivative is checked at the user steps (level 3) "
                "only."
            )
        other = [
            v
            for v in vs
            if not cut_of[v]
            and len(sel := sd[(sd["variable"] == v) & (sd["status"] != STATUS_ZERO)])
            and (sel["level2"] == STATUS_NOT_JUDGED).all()
        ]
        if other:
            notes.append(
                f"level 2 not judged on every checked day of {', '.join(other)}: the reported "
                "(straight-through) and exact-mode derivatives differ there (beyond the level-2 tolerance); "
                "the reported derivative is checked at the user steps (level 3) only."
            )
        kinks = sd[sd["kink_side"] != ""]
        if len(kinks):
            notes.append(
                f"{len(kinks)} checked row(s) sit on a kink at the point (a weather value exactly on a model "
                "threshold, e.g. TMAX = 35 degC for PETPT): the derivative is one-sided (column kink_side; "
                "fd_small_right / fd_small_left give both slopes) and level 2 is 'kink' (undecidable) there."
            )
    ctx = _Context(
        progs, i, g, g_exact, t_cal, stage, dates, yrdoy, e_mat, e_silk, cfg, [Path(f) for f in weather_files]
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
    ``owner`` / ``key``: keep the compiled programs on ``owner`` (a second call does not compile); a key
    names its runs: a later call under the same key with other run objects raises ``ValueError``."""
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
    if len(progs.runs) != len(runs) or any(a is not b for a, b in zip(progs.runs, runs, strict=False)):
        raise ValueError(
            f"weather_sensitivity: the programs kept under key {key!r} were built for other runs; pass a "
            "key that names the runs (or owner=None)"
        )
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
    index by position, or by year (``res[1982]``, any integer type) where the year is unique; a year
    with no scenario raises ``KeyError``."""

    years: list[int]

    def __getitem__(self, k: Any) -> Any:
        if isinstance(k, numbers.Integral) and not isinstance(k, bool) and int(k) >= _YEAR_MIN:
            y = int(k)
            hits = [i for i, yy in enumerate(getattr(self, "years", [])) if yy == y]
            if not hits:
                raise KeyError(f"no scenario for year {y}")
            if len(hits) != 1:
                raise KeyError(f"year {y} names {len(hits)} scenarios: index by position")
            return super().__getitem__(hits[0])
        return super().__getitem__(k)
