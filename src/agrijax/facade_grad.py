"""Gradients and sensitivities of a season's end values to the CERES-Maize cultivar coefficients.

The user code this module enables (continuing the quick start of :mod:`agrijax.dssat`)::

    g = exp.gradient(treatment=4, outputs=["HWAM", "CWAM"], params=["G2", "G3", "PHINT"])
    print(g)                 # d output / d coefficient at the published cultivar, one trust label each
    g.table                  # one row per (output, coefficient): value, derivative, sensitivity, trust
    g.trust                  # {"G2": "validated gradient", "G3": ..., "PHINT": "experimental"}
    scen = exp.scenarios(treatment=4, years=range(1978, 1988), sowing_shift=[-14, 0, 14])
    sens = scen.sensitivity(outputs=["HWAM"], params=["G2", "G3"])
    sens.summary             # per output and coefficient: mean / min / max over the 30 scenarios, trust
    sens.table               # one row per (scenario, output, coefficient)
    sens.trust               # the label of each coefficient over all scenarios

``exp.gradient`` is one season of one treatment; ``scen.sensitivity`` is the same for every weather
year x sowing date of a :class:`~agrijax.dssat.Scenarios`, all of them in one batched program. Both
are :func:`gradient` / :func:`sensitivity` of this module.

**What is differentiated.** The end-of-season values of Agri-JAX's DSSAT-CSM v4.8.6 maize day
(nitrogen off): ``"HWAM"`` grain yield, ``"CWAM"`` tops weight and ``"H#AM"`` grain number on the
simulated maturity day, and ``"LAI@60"`` / ``"CWAD@60"`` / ``"GWAD@60"``: the value on a fixed day
after planting (60 here), which does not depend on when maturity falls. The derivative is the
forward-mode automatic derivative (AD) with respect to the six ``MZCER048.CUL`` coefficients ``P1``,
``P2``, ``P5``, ``G2``, ``G3``, ``PHINT``, all other inputs fixed. The maturity day itself is an
integer: the derivative of ``HWAM`` is that of the value on the day maturity falls, it does not
include the effect of the maturity day moving (the dates ``ADAT`` / ``MDAT`` have no derivative; run
:meth:`~agrijax.dssat.Scenarios.run` for them).

**Trust labels.** Every (output, coefficient) pair, and every coefficient, carries one of three
labels, from the gradient-trust check of :mod:`agrijax.calib.trust` run at the evaluation point
(the same check, thresholds and per-coefficient plan the calibration uses,
:func:`agrijax.calib.ceres.ceres_gradient_plan`):

* ``"validated gradient"`` - the AD derivative agrees with central finite differences at two steps
  (1e-5 and 2 % of the coefficient's ``MINIMA`` - ``MAXIMA`` range, to 1e-3 and 5 %; at the 1e-5
  step each of the two paths of the straight-through derivative is tested on its own model, see
  **Straight-through derivatives** below), and a line scan
  of ``scan_points`` points over +-10 % of the range finds no jump and no flat stretch the derivative
  does not explain (class ``smooth``, trust level 3), or the output does not move at all (``inert``).
  It holds at this point and at the scan's resolution: a jump narrower than the grid spacing can fall
  between two scan points and go unseen (the calibration scans 201 points for that reason);
* ``"falls back"`` - the check failed (jumps, a flat stretch while the output moves, a kink, or an
  AD / finite-difference disagreement): ``derivative`` is then the central finite difference over
  +-2 % of the range, not the AD value (which stays in column ``ad``);
* ``"experimental"`` - ``P1``, ``P2``, ``P5`` and ``PHINT`` act through the phenology events
  (emergence, silking, maturity: when a stage is reached). **Derivatives through phenology events are
  experimental**: the stage changes of the CERES-Maize phenology are selections on integer days
  (``jnp.where``, no :func:`~agrijax.core.grad.event_ste` ramp), so AD keeps every stage on its day
  and misses the jumps of a stage moving by a day (``ad`` equals ``ad_exact`` for these
  coefficients; the line scan finds the jumps, class ``jumpy`` or ``step``). The calibration does not
  trust them either (they are derivative-free by default there). ``derivative`` is that AD value; the
  central difference is in column ``fd`` to compare with, and the class and level the check found
  stay in the table.

A coefficient's label is the weakest of its pairs (over outputs, and over scenarios in a batch).
The labels are about the derivative, not the model: they say whether the local slope can be used,
and say nothing about how well the model reproduces a measured response.

**Straight-through derivatives.** The forward-mode program runs in the ``ste`` gradient mode
(:data:`agrijax.calib.dssat_day.GRADIENT_MODE`): through the quantisations DSSAT applies (the root
length density ``RLV = REAL(INT(RLV*1000))/1000`` of ``MZ_ROOTS``, the soil water
``SW = ANINT(SW*1E6)/1E6`` of ``WATBAL`` and the other quantiser call sites, each registered as a
``GradientConvention`` of its process) the derivative is the identity. Defined precisely, ``ad`` is the
derivative of the **unrounded model** (:func:`agrijax.core.grad.unrounded`: every quantiser the
identity) taken along the rounded trajectory; ``ad_exact`` is the derivative of the forward program
(exact mode: 0 through every quantum) and ``ad_unrounded`` that of the unrounded model along its own
trajectory. In a water-limited season the effect of ``G2`` / ``G3`` on root growth reaches yield
through root water uptake and ``ad`` differs from ``ad_exact`` (0.3 to 2.8 % for yield, up to 8 % for
tops weight). Level 2 tests both paths ``ad`` is made of, each against a small-step (1e-5) central
difference of a real function: ``ad_exact`` against the model's (``err_small_exact``), ``ad_unrounded``
against the unrounded model's (``err_small_unrounded``); ``err_small`` is the larger. ``ste_offset`` =
``ad / ad_unrounded - 1`` (0 to 1.1 % here) is the trajectory offset: the unrounded model's states drift
off the rounded ones by up to a quantum a day, so the two derivatives are not equal and are not
compared. The 2 % step and the line scan test ``ad`` on the real model (level 3), as
:func:`agrijax.calib.trust.trust_report` does for a mode-bound function. Where a difference does not
cross a discontinuity, ``ad_exact`` and ``ad_unrounded`` equal their central differences to 1e-9; the
unrounded model is not smooth either: in water-limited seasons its comparisons that the rounding held
exactly (a water content at the lower limit, for instance) switch at tiny steps, and its small-step
difference crosses such a jump in about a third of the cases (``scripts/diag/dssat_grad_gap.py
--sweep``), where level 2 cannot verify the path and the pair falls back.

**The large step.** 2 % of the range is the scale of the optimisers' steps: Adam (``calib.fit``, rate
0.05 in the logit coordinate) moves ``0.05 u (1 - u)`` of the range per step (at most 1.25 %, 0.40 % for
``G2`` and 1.00 % for ``G3`` at the published cultivar), the staged Levenberg-Marquardt's secants span
``+-0.1 u (1 - u)`` (0.8 % and 2.0 %). With the large step at 1 % of the range the same 15 of the 36
pairs below are validated as at 2 %; at 5 % also 15, one gained (1979 +14 ``CWAM``/``G2``) and one lost
(1985 +14 ``CWAM``/``G2``, whose 5 % secant crosses a jump): the labels do not hinge on the number.

Measured on rorqual (the tables ``tests/integration/test_facade_grad.py`` and
``scripts/diag/dssat_grad_gap.py --facade`` / ``--secant`` print; UFGA8201 treatment 4, published
cultivar, 201 scan points): for the experiment's own season (1982, planting as in the file) the AD
derivatives of ``HWAM``, ``CWAM`` and ``H#AM`` with respect to ``G2`` and ``G3`` are validated (no
straight-through path reaches them: ``ad`` equals ``ad_exact`` and ``ad_unrounded``; they equal the
central difference at a 1e-5 step to 1e-10). Over the 1979 / 1982 / 1985 x sowing -14 / 0 / +14 days
scenarios 15 of the 36 (scenario, output, coefficient) pairs of ``HWAM`` / ``CWAM`` x ``G2`` / ``G3``
are validated: all of 1982 at 0 and +14 days, 1982 -14 ``HWAM``, 1979 +14 ``HWAM``/``G2``, 1985 at 0
days ``G3``, 1985 +14 ``G2``. The other 21 fall back:

* the unrounded model's 1e-5 difference crosses one of its jumps (level 1): 1979 -14 (all four),
  1979 +14 ``HWAM``/``G3`` and ``CWAM``/``G3``, 1985 -14 (all four), 1985 at 0 days ``HWAM``/``G2`` and
  ``CWAM``/``G2``;
* the model's own 1e-5 difference straddles a quantum (level 1, 0.6 to 9 % off): 1979 at 0 days (all
  four);
* a real non-smoothness at the optimiser's scale (level 2): 1979 +14 ``CWAM``/``G2`` (2 % secant 7 %
  off), 1982 -14 ``CWAM``/``G2`` and ``CWAM``/``G3`` (30 % off), 1985 +14 ``HWAM``/``G3`` and
  ``CWAM``/``G3`` (a jump in the scan).

Over the batch ``G2`` and ``G3`` fall back. For ``PHINT`` the AD value and the finite difference over
+-2 % of the range differ by a factor of 2.4 on the experiment's own season (by sign in some scenarios):
the stage days moving, which AD does not see (above); hence ``experimental``.

Runs in float64 on the host's JAX devices (set ``XLA_FLAGS=--xla_force_host_platform_device_count=<cores>``
before importing JAX to use every CPU core). The cost is three batched forward-mode programs (the
``ste`` and ``exact`` modes and the unrounded model): per coefficient and scenario, three derivative
points, six finite-difference points and ``scan_points`` scan points.
"""

from __future__ import annotations

import re
import time
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_OUTPUTS",
    "EXPERIMENTAL",
    "LABEL_EXPERIMENTAL",
    "LABEL_FALLBACK",
    "LABEL_VALIDATED",
    "OUTPUTS",
    "SCAN_POINTS",
    "Gradient",
    "Sensitivity",
    "gradient",
    "sensitivity",
]

LABEL_VALIDATED = "validated gradient"
LABEL_FALLBACK = "falls back"
LABEL_EXPERIMENTAL = "experimental"

#: end-of-season outputs: name -> (unit, meaning); read on the simulated maturity day
OUTPUTS: dict[str, tuple[str, str]] = {
    "HWAM": ("kg ha-1", "grain yield at maturity"),
    "CWAM": ("kg ha-1", "tops weight at maturity"),
    "H#AM": ("kernels m-2", "grain number at maturity"),
}
#: daily series that can be read on a fixed day after planting (``"LAI@60"``): name -> (model output, unit)
SERIES: dict[str, tuple[str, str]] = {
    "LAI": ("lai", "m2 m-2"),
    "CWAD": ("cwad", "kg ha-1"),
    "GWAD": ("gwad", "kg ha-1"),
}
DEFAULT_OUTPUTS: tuple[str, ...] = ("HWAM", "CWAM")
#: line-scan points of the trust check (the calibration's own: ``agrijax.calib.fit.TRUST_SCAN``)
SCAN_POINTS = 201
#: the coefficients whose derivative runs through phenology events
#: (:data:`agrijax.calib.ceres.CERES_DERIVATIVE_FREE`)
EXPERIMENTAL: tuple[str, ...] = ("P1", "P2", "P5", "PHINT")
#: rows (seasons) of one compiled program call; a batch is cut into equal chunks of at most this many
CHUNK_ROWS = 512
#: rows per (scenario, coefficient) of the side programs of :class:`_Rows`: the exact-mode derivative
#: (1), the unrounded model's derivative and its two small-step points (3)
_SIDE_ROWS = {"exact": 1, "unrounded": 3}
_SERIES_RE = re.compile(r"^(LAI|CWAD|GWAD)@(\d+)$", re.IGNORECASE)


# ------------------------------------------------------------------------------ outputs
def _planting_index(run: Any) -> int:
    hit = np.nonzero(run.days == int(np.asarray(run.params_crop.yrplt)))[0]
    return int(hit[0]) if hit.size else 0


def _parse_outputs(outputs: Sequence[str] | str) -> list[str]:
    names = [outputs] if isinstance(outputs, str) else [str(o) for o in outputs]
    if not names:
        raise ValueError("outputs: name at least one output")
    if len(set(names)) != len(names):
        raise ValueError(f"outputs: duplicates in {names}")
    out = []
    for n in names:
        m = _SERIES_RE.match(n)
        key = f"{m.group(1).upper()}@{int(m.group(2))}" if m else n.upper()
        if not m and key not in OUTPUTS:
            raise ValueError(
                f"unknown output {n!r}; known: {list(OUTPUTS)} and "
                f"'<{'|'.join(SERIES)}>@<days after planting>'"
            )
        out.append(key)
    if len(set(out)) != len(out):
        raise ValueError(f"outputs: duplicates in {names}")
    return out


def _output_unit(name: str) -> str:
    return OUTPUTS[name][0] if name in OUTPUTS else SERIES[name.split("@")[0]][1]


def _entries(names: Sequence[str], run: Any) -> list[Any]:
    """The simulator entries of ``names`` on ``run``, then the maturity date (for the matured flag)."""
    from agrijax.calib.dssat_day import E_DATE, E_DAY, E_FINAL, ISTAGE_MATURITY, OUT_NAMES, Entry

    ents = []
    for n in names:
        if "@" in n:
            series, dap = n.split("@")
            t = _planting_index(run) + int(dap)
            if t >= run.n_days:
                raise ValueError(
                    f"{n}: day {t} from the start of the simulation is after the end of the season "
                    f"({run.n_days} days)"
                )
            ents.append(Entry(E_DAY, OUT_NAMES.index(SERIES[series][0]), t))
        else:
            out = {"HWAM": "gwad", "CWAM": "cwad", "H#AM": "g_ad"}[n]
            ents.append(Entry(E_FINAL, OUT_NAMES.index(out)))
    ents.append(Entry(E_DATE, ISTAGE_MATURITY))
    return ents


def _parse_params(params: Sequence[str] | str, names: Sequence[str]) -> list[int]:
    ps = [params] if isinstance(params, str) else [str(p) for p in params]
    bad = [p for p in ps if p not in names]
    if bad or not ps or len(set(ps)) != len(ps):
        raise ValueError(f"params {ps}: distinct names out of {list(names)}")
    return [list(names).index(p) for p in ps]


# ------------------------------------------------------------------------------ the batched rows
class _Rows:
    """The season's entries and their forward derivative on **rows** ``(theta [6], tangent [6],
    scenario)``: one compiled program (``jax.jvp`` of the batched day, gradient mode
    :data:`~agrijax.calib.dssat_day.GRADIENT_MODE`) on equal chunks, sharded over the JAX devices.
    Calling it returns ``(y, dy)`` ``[rows, entries]``; a zero tangent gives the plain forward values."""

    def __init__(self, runs: Sequence[Any], entries: Sequence[Sequence[Any]]) -> None:
        import jax

        from agrijax.calib.dssat_day import GRADIENT_MODE, DaySimulator
        from agrijax.core.grad import bind_gradient_mode, bind_unrounded

        self.sim = DaySimulator(list(runs), [list(e) for e in entries])
        if len(self.sim.groups) != 1:
            raise ValueError("the scenarios must share one soil layering and one evaporation method")
        (self.group,) = self.sim.groups
        self.local = np.asarray([self.sim.where[i][1] for i in range(len(runs))], dtype=np.int32)
        self.n_entries = max(len(e) for e in entries)
        fn = self.sim._sim_fn(self.group)

        def forward_and_tangent(inputs: Any, tab: Any, theta: Any, tan: Any, tid: Any) -> Any:
            return jax.jvp(lambda th: fn(inputs, tab, th, tid), (theta,), (tan,))

        self._bound = {
            GRADIENT_MODE: bind_gradient_mode(forward_and_tangent, GRADIENT_MODE),
            "exact": bind_gradient_mode(forward_and_tangent, "exact"),
            # the unrounded model (every quantiser the identity), in the straight-through mode
            "unrounded": bind_gradient_mode(bind_unrounded(forward_and_tangent), GRADIENT_MODE),
        }
        self._programs: dict[tuple[str, int], Any] = {}
        self.compile_s = 0.0
        self.run_s = 0.0

    @property
    def season_days(self) -> int:
        """The padded length of the simulated seasons (a maturity date at this index: not reached)."""
        return int(self.sim.n_days[self.group])

    def _program(self, n: int, mode: str) -> Any:
        if (mode, n) not in self._programs:
            import jax
            import jax.numpy as jnp

            from agrijax.calib.dssat_day import CUL_ORDER

            sim = self.sim
            jf = jax.jit(
                self._bound[mode],
                in_shardings=(sim.rep, sim.rep, sim.bsh, sim.bsh, sim.bsh),
                out_shardings=(sim.bsh, sim.bsh),
            )
            k = len(CUL_ORDER)
            args = (
                sim.inputs[self.group],
                sim.tables[self.group],
                jnp.zeros((n, k)),
                jnp.zeros((n, k)),
                jnp.zeros(n, jnp.int32),
            )
            t0 = time.perf_counter()
            self._programs[(mode, n)] = jf.lower(*args).compile()
            self.compile_s += time.perf_counter() - t0
        return self._programs[(mode, n)]

    def __call__(self, theta: np.ndarray, tan: np.ndarray, scn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from agrijax.calib.dssat_day import GRADIENT_MODE

        return self.evaluate(theta, tan, scn, GRADIENT_MODE)

    def exact(self, theta: np.ndarray, tan: np.ndarray, scn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """As calling the rows, with the derivative of the ``exact`` gradient mode: the derivative
        of the forward program itself (0 through Fortran's truncations and roundings), which the
        small-step finite difference tests (see :func:`_analyse`)."""
        return self.evaluate(theta, tan, scn, "exact")

    def unrounded(self, theta: np.ndarray, tan: np.ndarray, scn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """As calling the rows, on the unrounded model (:func:`agrijax.core.grad.unrounded`: every
        truncation, rounding and REAL*4 store the identity), whose derivative the straight-through
        one is along its own trajectory (see :func:`_analyse`)."""
        return self.evaluate(theta, tan, scn, "unrounded")

    def evaluate(
        self, theta: np.ndarray, tan: np.ndarray, scn: np.ndarray, mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        import jax

        from agrijax.calib.dssat_day import CUL_ORDER

        theta = np.asarray(theta, dtype=np.float64)
        tan = np.asarray(tan, dtype=np.float64)
        scn = np.asarray(scn, dtype=np.int64)
        n = theta.shape[0]
        cap = self.sim.ndev
        while cap * 2 <= max(CHUNK_ROWS, self.sim.ndev):
            cap *= 2
        n_chunks = -(-n // cap)
        size = self.sim._pad(-(-n // n_chunks))
        if mode in _SIDE_ROWS:
            # the exact rows are the AD rows only (one per scenario and coefficient), the unrounded rows
            # those and the two small-step points: one program size for every call of these rows,
            # whatever coefficients a call asks for
            rows_max = _SIDE_ROWS[mode] * len(CUL_ORDER) * len(self.local)
            size = max(size, min(cap, self.sim._pad(rows_max)))
        y = np.zeros((n, self.n_entries))
        dy = np.zeros((n, self.n_entries))
        prog = self._program(size, mode)
        for c0 in range(0, n, size):
            m = min(size, n - c0)
            sl = slice(c0, c0 + m)

            def rows(a: np.ndarray, sl: slice = sl, m: int = m) -> np.ndarray:
                return np.concatenate([a[sl], np.repeat(a[sl][-1:], size - m, axis=0)]) if m < size else a[sl]

            t0 = time.perf_counter()
            yy, dd = prog(
                self.sim.inputs[self.group],
                self.sim.tables[self.group],
                jax.device_put(rows(theta), self.sim.bsh),
                jax.device_put(rows(tan), self.sim.bsh),
                jax.device_put(rows(self.local[scn]), self.sim.bsh),
            )
            y[sl], dy[sl] = np.asarray(yy)[:m], np.asarray(dd)[:m]
            self.run_s += time.perf_counter() - t0
        return y, dy


# ------------------------------------------------------------------------------ the analysis
def _analyse(
    evaluate: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    x: Any,
    idx: Sequence[int],
    lower: Any,
    upper: Any,
    n_scn: int,
    n_out: int,
    cfg: Any,
    evaluate_exact: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    | None = None,
    evaluate_unrounded: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    | None = None,
) -> dict[str, Any]:
    """AD derivative, finite differences, scan diagnostics and class / level of every (scenario,
    output, parameter) around the point ``x [K]``, for the parameters at ``idx`` and ``n_scn``
    scenarios, in one call of ``evaluate(theta [N, K], tangent [N, K], scenario [N]) -> (y, dy)``
    (``[N, E]``: values and the derivative along ``tangent``). The first ``n_out`` entries are the
    outputs analysed; the others (``extra`` in the result) are only evaluated at ``x``.

    The same quantities and thresholds as :func:`agrijax.calib.trust.trust_report`, which makes
    them one parameter at a time for one function: the AD derivative (forward mode along each
    ``e_i``), central differences at the two ``cfg.fd_steps`` (one-sided where the step would leave
    the bounds), a line scan of ``cfg.n_scan`` points over ``+- cfg.scan_frac`` of the width (clipped
    to the bounds), and :func:`~agrijax.calib.trust.classify_pair`. The bounds are widened to hold
    ``x`` if it lies outside them. Arrays are ``[S, E, P]`` (``y0``: ``[S, E]``).

    ``evaluate_exact``, ``evaluate_unrounded``: for an ``evaluate`` whose derivative is
    straight-through, the same evaluation in the ``exact`` mode and on the unrounded model (as
    :func:`~agrijax.calib.trust.fd_check`'s ``f_exact`` / ``f_unrounded``): level 2 then needs the
    exact-mode derivative (``ad_exact``, on the AD rows) to agree with the small-step difference
    and the unrounded model's derivative (``ad_unrounded``) with the unrounded model's small-step
    difference (``fd_small_unrounded``); the large step and the scan test ``ad``. Without them
    ``ad_exact`` is ``ad`` and the unrounded arrays are NaN."""
    from agrijax.calib.trust import classify_pair, fd_agreement, scan_summary

    x = np.asarray(x, dtype=float)
    k, p_n, s_n = x.shape[0], len(idx), int(n_scn)
    ix = np.asarray(idx, dtype=int)
    lo = np.minimum(np.asarray(lower, dtype=float), x)
    hi = np.maximum(np.asarray(upper, dtype=float), x)
    w = (hi - lo)[ix]
    eye = np.eye(k)[ix]  # [P, K]
    scn = np.arange(s_n)
    n_scan = int(cfg.n_scan)

    # rows A: the AD derivative; B: central differences at the two steps (x + fwd e_i, x - bwd e_i);
    # C: the scan points with their directional derivatives
    h = np.stack([f * w for f in cfg.fd_steps])  # [2, P]
    fwd = np.where(x[ix] + h <= hi[ix], h, 0.0)
    bwd = np.where(x[ix] - h >= lo[ix], h, 0.0)
    if np.any(fwd + bwd <= 0.0):
        raise ValueError("a finite-difference step leaves the coefficient's range on both sides")
    a = np.maximum(lo[ix], x[ix] - cfg.scan_frac * w)
    b = np.minimum(hi[ix], x[ix] + cfg.scan_frac * w)
    grid = np.stack([np.linspace(a[p], b[p], n_scan) for p in range(p_n)])  # [P, n]

    th_a = np.broadcast_to(x, (s_n, p_n, k)).copy()
    tan_a = np.broadcast_to(eye, (s_n, p_n, k)).copy()
    th_b = np.broadcast_to(x, (s_n, p_n, 2, 2, k)).copy()
    th_c = np.broadcast_to(x, (s_n, p_n, n_scan, k)).copy()
    for p in range(p_n):
        for step in range(2):
            th_b[:, p, step, 0, ix[p]] += fwd[step, p]
            th_b[:, p, step, 1, ix[p]] -= bwd[step, p]
        th_c[:, p, :, ix[p]] = grid[p][None, :]
    tan_c = np.broadcast_to(eye[None, :, None, :], (s_n, p_n, n_scan, k))
    n_a, n_b = s_n * p_n, s_n * p_n * 4

    def ids(shape: tuple[int, ...]) -> np.ndarray:
        return np.broadcast_to(scn.reshape((-1,) + (1,) * (len(shape) - 1)), shape).reshape(-1)

    y, dy = evaluate(
        np.concatenate([th_a.reshape(-1, k), th_b.reshape(-1, k), th_c.reshape(-1, k)]),
        np.concatenate([tan_a.reshape(-1, k), np.zeros((n_b, k)), tan_c.reshape(-1, k)]),
        np.concatenate([ids((s_n, p_n)), ids((s_n, p_n, 4)), ids((s_n, p_n, n_scan))]),
    )
    e_n = int(n_out)
    e_all = y.shape[1]
    y_a = y[:n_a].reshape(s_n, p_n, e_all)
    dy_a = dy[:n_a, :e_n].reshape(s_n, p_n, e_n)
    y_b = y[n_a : n_a + n_b, :e_n].reshape(s_n, p_n, 2, 2, e_n)
    y_c = y[n_a + n_b :, :e_n].reshape(s_n, p_n, n_scan, e_n)
    g_c = dy[n_a + n_b :, :e_n].reshape(s_n, p_n, n_scan, e_n)

    y0 = y_a[:, 0, :e_n]  # [S, E]
    ad = dy_a.transpose(0, 2, 1)  # [S, E, P]
    if evaluate_exact is None:
        ad_exact = ad
    else:
        _, dy_x = evaluate_exact(th_a.reshape(-1, k), tan_a.reshape(-1, k), ids((s_n, p_n)))
        ad_exact = dy_x[:, :e_n].reshape(s_n, p_n, e_n).transpose(0, 2, 1)
    shape = (s_n, e_n, p_n)
    ad_u = np.full(shape, np.nan)
    fd_u = np.full(shape, np.nan)
    rel_u = np.zeros(shape)
    agree_u = np.ones(shape, dtype=bool)
    if evaluate_unrounded is not None:
        th_u = np.concatenate([th_a.reshape(-1, k), th_b[:, :, 0].reshape(-1, k)])
        tan_u = np.concatenate([tan_a.reshape(-1, k), np.zeros((2 * n_a, k))])
        y_u, dy_u = evaluate_unrounded(th_u, tan_u, np.concatenate([ids((s_n, p_n)), ids((s_n, p_n, 2))]))
        ad_u = dy_u[:n_a, :e_n].reshape(s_n, p_n, e_n).transpose(0, 2, 1)
        yb_u = y_u[n_a:, :e_n].reshape(s_n, p_n, 2, e_n)
        fd_u = ((yb_u[:, :, 0] - yb_u[:, :, 1]) / (fwd[0] + bwd[0])[None, :, None]).transpose(0, 2, 1)
        y0_u = y_u[:n_a, :e_n].reshape(s_n, p_n, e_n)[:, 0]
        for s in range(s_n):
            rel_u[s], agree_u[s] = fd_agreement(ad_u[s], fd_u[s], y0_u[s], w, 0, cfg)
    fds = [
        ((y_b[:, :, st, 0, :] - y_b[:, :, st, 1, :]) / (fwd[st] + bwd[st])[None, :, None]).transpose(0, 2, 1)
        for st in range(2)
    ]
    rel = [np.zeros(shape), np.zeros(shape)]
    agree = [np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)]
    for s in range(s_n):
        for st in range(2):
            rel[st][s], agree[st][s] = fd_agreement(
                (ad_exact if st == 0 else ad)[s], fds[st][s], y0[s], w, st, cfg
            )
    rel_x = rel[0]
    rel[0] = np.maximum(rel[0], rel_u)
    agree[0] = agree[0] & agree_u
    cls = np.empty(shape, dtype=object)
    level = np.zeros(shape, dtype=int)
    jumps = np.zeros(shape, dtype=int)
    zero_frac = np.zeros(shape)
    scan_range = np.zeros(shape)
    for s in range(s_n):
        for p in range(p_n):
            sc = scan_summary(grid[p], y_c[s, p], g_c[s, p], w[p], cfg)
            for e in range(e_n):
                cls[s, e, p], level[s, e, p] = classify_pair(
                    (bool(agree[0][s, e, p]), bool(agree[1][s, e, p])), sc, e, cfg
                )
            jumps[s, :, p] = sc["n_jumps"]
            zero_frac[s, :, p] = sc["zero_frac"]
            scan_range[s, :, p] = sc["range"]
    return {
        "y0": y0,
        "extra": y_a[:, 0, e_n:],
        "ad": ad,
        "ad_exact": ad_exact,
        "ad_unrounded": ad_u,
        "fd_small_unrounded": fd_u,
        "rel_err_small_exact": rel_x,
        "rel_err_small_unrounded": rel_u,
        "fd_small": fds[0],
        "fd_large": fds[1],
        "rel_err_small": rel[0],
        "rel_err_large": rel[1],
        "class": cls,
        "level": level,
        "n_jumps": jumps,
        "zero_frac": zero_frac,
        "scan_range": scan_range,
        "width": w,
        "scan": np.stack([a, b], axis=1),
    }


def _labels(
    raw: Mapping[str, Any], params: Sequence[str], out_names: Sequence[str], scn_names: Sequence[str]
) -> tuple[np.ndarray, dict[str, str], dict[str, list[str]]]:
    """Per-pair labels ``[S, E, P]`` and per-parameter labels / reasons, from the calibration's own
    :func:`~agrijax.calib.ceres.ceres_gradient_plan` on the classes and levels of ``raw``: a pair is
    ``validated`` where the plan differentiates it by AD, ``experimental`` for the phenology
    coefficients, ``falls back`` otherwise; a coefficient is ``validated`` when the plan says
    ``gradient`` (AD on every output of every scenario)."""
    from agrijax.calib.ceres import CERES_DERIVATIVE_FREE, ceres_gradient_plan

    s_n, e_n, _ = raw["class"].shape
    keys = [(s, e) for s in range(s_n) for e in range(e_n)]
    names = [f"{scn_names[s]}/{out_names[e]}" for s, e in keys]
    rep = {
        "outputs": names,
        "params": {
            p: {
                "outputs": {
                    nm: {
                        "class": raw["class"][s, e, i],
                        "level": int(raw["level"][s, e, i]),
                        "n_jumps": int(raw["n_jumps"][s, e, i]),
                    }
                    for nm, (s, e) in zip(names, keys, strict=True)
                }
            }
            for i, p in enumerate(params)
        },
    }
    pair = ceres_gradient_plan(rep)  # every (scenario, output) its own group: the pair
    groups = {sn: [f"{sn}/{o}" for o in out_names] for sn in scn_names}
    plan = ceres_gradient_plan(rep, groups)
    lab = np.empty(raw["class"].shape, dtype=object)
    for i, p in enumerate(params):
        for b, (s, e) in enumerate(keys):
            lab[s, e, i] = (
                LABEL_VALIDATED
                if pair.use_ad[b, i]
                else LABEL_EXPERIMENTAL
                if p in CERES_DERIVATIVE_FREE
                else LABEL_FALLBACK
            )
    trust = {
        p: LABEL_VALIDATED
        if plan.method[p] == "gradient"
        else LABEL_EXPERIMENTAL
        if p in CERES_DERIVATIVE_FREE
        else LABEL_FALLBACK
        for p in params
    }
    return lab, trust, {p: list(plan.reasons.get(p, [])) for p in params}


# ------------------------------------------------------------------------------ results
_NOTE = (
    "Derivatives through phenology events (P1, P2, P5, PHINT) are experimental; "
    "'validated gradient' holds at this point and at the scan's resolution."
)


@dataclass
class Gradient:
    """The derivative of season outputs with respect to cultivar coefficients on one treatment
    (:meth:`agrijax.dssat.Experiment.gradient`).

    :attr:`table` has one row per (output, coefficient): ``value`` (the output at the point),
    ``derivative`` (d output / d coefficient: the AD value, or the central difference where
    ``trust`` is ``"falls back"``; column ``source`` says which), ``unit``, ``sensitivity``
    (derivative x the coefficient's ``MINIMA``-``MAXIMA`` range / ``value``: the relative change of
    the output across the whole range, linearised), ``elasticity`` (derivative x coefficient /
    ``value``: relative change per relative change), ``ad``, ``ad_exact`` and ``fd`` (the AD
    derivative, straight-through as reported, the program's own exact-mode derivative, and the
    central finite difference over +-2 % of the range), ``trust`` (``"validated gradient"``,
    ``"falls back"`` or ``"experimental"``: see :mod:`agrijax.facade_grad`), and the check's
    ``class`` (``smooth``, ``kinked``, ``jumpy``, ``step``, ``inert``), trust ``level`` (0-3: 0 = a
    non-finite value, 1 = the exact-mode derivative ``ad_exact`` and the 1e-5 central difference
    disagree, 2 = they
    agree but the 2 % step or the scan finds a difference, 3 = all agree), the ``jumps`` its scan
    found and ``err_small`` / ``err_large`` (the relative AD / finite-difference disagreement at the
    two steps).
    :attr:`trust` is the label of each coefficient (the weakest over the outputs), :attr:`reasons`
    why a coefficient is not validated, :attr:`point` the cultivar coefficients the derivatives are
    taken at, :attr:`timing` the cost."""

    treatment: str
    point: dict[str, float]
    table: Any
    trust: dict[str, str]
    reasons: dict[str, list[str]] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    scan_points: int = SCAN_POINTS

    def __str__(self) -> str:
        return _text(f"d output / d coefficient, {self.treatment}", self.table, self.trust, self.timing)

    def plot(self, **kw: Any) -> Any:
        """A bar chart of the normalised sensitivities (:attr:`table` ``sensitivity``), one panel per
        output, each bar marked by its trust label (needs matplotlib)."""
        return _plot(
            self.table, self.trust, f"{self.treatment}: sensitivity to the cultivar coefficients", **kw
        )


@dataclass
class Sensitivity:
    """The derivative of season outputs with respect to cultivar coefficients over the scenarios of a
    batch (:meth:`agrijax.dssat.Scenarios.sensitivity`).

    :attr:`table` has one row per (scenario, output, coefficient) with the columns of
    :attr:`Gradient.table` after ``scenario``, ``year``, ``sowing_shift`` and ``matured`` (the season
    reached maturity within the simulated days; if not, ``HWAM`` etc. are the values on the last
    day). :attr:`summary` has one row per (output, coefficient): the mean, minimum and maximum
    derivative over the scenarios, the mean ``sensitivity``, how many scenarios are
    ``validated`` and the ``trust`` label (``"validated gradient"`` only when every scenario is).
    :attr:`trust` is the label of each coefficient (the weakest over outputs and scenarios),
    :attr:`reasons` why a coefficient is not validated."""

    treatment: str
    point: dict[str, float]
    table: Any
    trust: dict[str, str]
    reasons: dict[str, list[str]] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    scan_points: int = SCAN_POINTS

    @property
    def summary(self) -> Any:
        """One row per (output, coefficient) over the scenarios (see the class)."""
        import pandas as pd

        t = self.table
        rows = []
        for (o, p), g in t.groupby(["output", "param"], sort=False):
            lab = set(g["trust"])
            rows.append(
                {
                    "output": o,
                    "param": p,
                    "scenarios": len(g),
                    "mean derivative": float(g["derivative"].mean()),
                    "min derivative": float(g["derivative"].min()),
                    "max derivative": float(g["derivative"].max()),
                    "mean sensitivity": float(g["sensitivity"].mean()),
                    "validated": int((g["trust"] == LABEL_VALIDATED).sum()),
                    "trust": LABEL_EXPERIMENTAL
                    if p in EXPERIMENTAL
                    else LABEL_VALIDATED
                    if lab == {LABEL_VALIDATED}
                    else LABEL_FALLBACK,
                    "unit": g["unit"].iloc[0],
                }
            )
        return pd.DataFrame(rows).set_index(["output", "param"])

    def __str__(self) -> str:
        n = int(self.table["scenario"].nunique())
        return _text(
            f"d output / d coefficient over {n} scenarios, {self.treatment}",
            self.summary.reset_index(),
            self.trust,
            self.timing,
            cols=(
                "output",
                "param",
                "mean derivative",
                "min derivative",
                "max derivative",
                "validated",
                "trust",
            ),
        )

    def plot(self, **kw: Any) -> Any:
        """A bar chart of the mean normalised sensitivities (:attr:`summary`), one panel per output,
        each bar marked by its trust label (needs matplotlib)."""
        s = self.summary.reset_index().rename(columns={"mean sensitivity": "sensitivity"})
        return _plot(s, self.trust, f"{self.treatment}: mean sensitivity over the scenarios", **kw)


def _text(
    title: str,
    table: Any,
    trust: Mapping[str, str],
    timing: Mapping[str, Any],
    cols: Sequence[str] | None = None,
) -> str:
    cols = cols or ("output", "param", "value", "derivative", "unit", "sensitivity", "trust")
    shown = table[[c for c in cols if c in table.columns]]
    body = shown.to_string(index=False, float_format=lambda v: f"{v:.5g}")
    lines = [title, body, "trust per coefficient: " + ", ".join(f"{p} {t}" for p, t in trust.items())]
    if timing:
        lines.append(
            f"{timing.get('rows', 0)} model runs: compile {timing.get('compile_s', 0.0):.1f} s, "
            f"run {timing.get('run_s', 0.0):.2f} s"
        )
    lines.append(_NOTE)
    return "\n".join(lines)


_COLOURS = {LABEL_VALIDATED: "#2a7ab9", LABEL_FALLBACK: "#d98c1f", LABEL_EXPERIMENTAL: "#8c8c8c"}


def _plot(table: Any, trust: Mapping[str, str], title: str, **kw: Any) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    outs = list(dict.fromkeys(table["output"]))
    fig, axes = plt.subplots(
        1, len(outs), figsize=kw.pop("figsize", (4.2 * len(outs) + 1.5, 3.4)), squeeze=False
    )
    for ax, o in zip(axes[0], outs, strict=True):
        t = table[table["output"] == o]
        ax.bar(t["param"], t["sensitivity"], color=[_COLOURS[v] for v in t["trust"]])
        ax.axhline(0.0, color="#444444", lw=0.6)
        ax.set_title(o)
        ax.set_ylabel("sensitivity (output change over the range / output)")
    fig.suptitle(title)
    fig.legend(
        handles=[Patch(color=c, label=k) for k, c in _COLOURS.items()],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


def _tables(
    raw: Mapping[str, Any],
    lab: np.ndarray,
    params: Sequence[str],
    out_names: Sequence[str],
    x: np.ndarray,
    idx: Sequence[int],
    scn_rows: Sequence[Mapping[str, Any]],
    matured: np.ndarray,
) -> Any:
    import pandas as pd

    from agrijax.calib.ceres import CERES_SPECS
    from agrijax.calib.dssat_day import CUL_ORDER

    rows = []
    for s, sr in enumerate(scn_rows):
        for e, o in enumerate(out_names):
            y = float(raw["y0"][s, e])
            for i, p in enumerate(params):
                fall = lab[s, e, i] == LABEL_FALLBACK
                ad, fd = float(raw["ad"][s, e, i]), float(raw["fd_large"][s, e, i])
                der = fd if fall else ad
                theta = float(x[idx[i]])
                rows.append(
                    {
                        **sr,
                        "matured": bool(matured[s]),
                        "output": o,
                        "param": p,
                        "value": y,
                        "derivative": der,
                        "unit": f"{_output_unit(o)} per ({CERES_SPECS[CUL_ORDER[idx[i]]].unit})",
                        "sensitivity": der * float(raw["width"][i]) / y if y != 0.0 else np.nan,
                        "elasticity": der * theta / y if y != 0.0 else np.nan,
                        "source": "central difference" if fall else "AD",
                        "ad": ad,
                        "ad_exact": float(raw["ad_exact"][s, e, i]),
                        "ad_unrounded": float(raw["ad_unrounded"][s, e, i]),
                        "ste_offset": ad / float(raw["ad_unrounded"][s, e, i]) - 1.0
                        if float(raw["ad_unrounded"][s, e, i]) != 0.0
                        else 0.0,
                        "fd": fd,
                        "trust": str(lab[s, e, i]),
                        "class": str(raw["class"][s, e, i]),
                        "level": int(raw["level"][s, e, i]),
                        "jumps": int(raw["n_jumps"][s, e, i]),
                        "err_small": float(raw["rel_err_small"][s, e, i]),
                        "err_small_exact": float(raw["rel_err_small_exact"][s, e, i]),
                        "err_small_unrounded": float(raw["rel_err_small_unrounded"][s, e, i]),
                        "err_large": float(raw["rel_err_large"][s, e, i]),
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ evaluators, cached
_CACHE: dict[int, tuple[Any, dict[Any, _Rows]]] = {}


def _cached(owner: Any, key: Any, build: Callable[[], _Rows]) -> _Rows:
    """The evaluator of ``key`` kept on ``owner`` (an experiment or a scenario set) for as long as it
    lives: a second call with the same outputs does not compile again."""
    k = id(owner)
    hit = _CACHE.get(k)
    if hit is None or hit[0]() is not owner:
        store: dict[Any, _Rows] = {}
        _CACHE[k] = (weakref.ref(owner, lambda _r, k=k: _CACHE.pop(k, None)), store)
    else:
        store = hit[1]
    if key not in store:
        store[key] = build()
    return store[key]


def _scenario_name(row: Mapping[str, Any]) -> str:
    return f"{row['year']} {row['sowing_shift']:+d} d" if "year" in row else "season"


def _point(published: Mapping[str, float], cultivar: Mapping[str, float] | None) -> np.ndarray:
    from agrijax.calib.dssat_day import CUL_ORDER

    vals = dict(published)
    for n, v in (cultivar or {}).items():
        if n not in CUL_ORDER:
            raise ValueError(f"unknown cultivar coefficient {n!r}; known: {list(CUL_ORDER)}")
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 0:
            raise ValueError(f"cultivar[{n!r}]: one value per coefficient (got shape {arr.shape})")
        vals[n] = float(arr)
    return np.asarray([vals[n] for n in CUL_ORDER], dtype=float)


def _run(
    owner: Any,
    key: Any,
    runs: Sequence[Any],
    scn_rows: Sequence[Mapping[str, Any]],
    published: Mapping[str, float],
    outputs: Sequence[str] | str,
    params: Sequence[str] | str,
    cultivar: Mapping[str, float] | None,
    scan_points: int,
) -> tuple[Any, dict[str, str], dict[str, list[str]], dict[str, Any], dict[str, float]]:
    from agrijax.calib.ceres import ceres_space
    from agrijax.calib.dssat_day import CUL_ORDER
    from agrijax.calib.trust import TrustConfig

    out_names = _parse_outputs(outputs)
    idx = _parse_params(params, CUL_ORDER)
    p_names = [CUL_ORDER[i] for i in idx]
    if int(scan_points) < 3:
        raise ValueError("scan_points must be at least 3")
    cfg = TrustConfig(n_scan=int(scan_points))
    x = _point(published, cultivar)
    rows = _cached(
        owner, (key, tuple(out_names)), lambda: _Rows(runs, [_entries(out_names, r) for r in runs])
    )
    sp = ceres_space(CUL_ORDER)
    c0, r0 = rows.compile_s, rows.run_s
    t0 = time.perf_counter()
    raw = _analyse(
        rows, x, idx, sp.lower, sp.upper, len(runs), len(out_names), cfg, rows.exact, rows.unrounded
    )
    matured = raw["extra"][:, 0] < rows.season_days  # the maturity date (last entry) was reached
    names_s = [_scenario_name(r) for r in scn_rows]
    lab, trust, reasons = _labels(raw, p_names, out_names, names_s)
    table = _tables(raw, lab, p_names, out_names, x, idx, scn_rows, matured)
    if not matured.all():
        late = [int(i) for i in np.nonzero(~matured)[0]]
        warnings.warn(
            f"scenarios {late} do not reach maturity within the simulated days: "
            "their end values are the values on the last day (column 'matured')",
            stacklevel=3,
        )
    timing = {
        "compile_s": rows.compile_s - c0,
        "run_s": rows.run_s - r0,
        "wall_s": time.perf_counter() - t0,
        "rows": int(len(runs) * len(idx) * (9 + cfg.n_scan)),
    }
    point = dict(zip(CUL_ORDER, x.tolist(), strict=True))
    return table, trust, reasons, timing, point


# ------------------------------------------------------------------------------ public functions
def gradient(
    experiment: Any,
    treatment: int,
    *,
    outputs: Sequence[str] | str = DEFAULT_OUTPUTS,
    params: Sequence[str] | str = ("P1", "P2", "P5", "G2", "G3", "PHINT"),
    cultivar: Mapping[str, float] | None = None,
    scan_points: int = SCAN_POINTS,
    soil_evaporation: str | None = None,
) -> Gradient:
    """The derivative of one season's end values with respect to cultivar coefficients, each with its
    trust label (:meth:`agrijax.dssat.Experiment.gradient`; see the module docstring for the labels).

    ``outputs``: ``"HWAM"`` grain yield, ``"CWAM"`` tops weight, ``"H#AM"`` grain number (on the
    maturity day), or ``"LAI@60"`` / ``"CWAD@60"`` / ``"GWAD@60"`` (the value 60 days after planting,
    any day number). ``params``: coefficients out of ``P1 P2 P5 G2 G3 PHINT`` (default all six).
    ``cultivar``: coefficients moved from the published values to evaluate the derivative elsewhere
    (``{"G2": 700.0}``). ``scan_points``: points of the trust check's line scan over +-10 % of each
    coefficient's range (more points find narrower jumps, and cost proportionally more).

    **Derivatives through phenology events are experimental**: ``P1``, ``P2``, ``P5`` and ``PHINT``
    are labelled ``"experimental"`` whatever the check finds, and ``G2`` / ``G3`` are validated or fall
    back per output at this point. ``soil_evaporation``: the season of the swapped soil evaporation
    (``"ritchie"`` or ``"salus"``, :func:`agrijax.dssat.alternatives`), as in
    :meth:`~agrijax.dssat.Experiment.run`. Needs the treatment's native inputs (no DSSAT run). Runs in
    float64 on JAX's default device only: it refuses an :func:`agrijax.options` block asking for
    float32 or a device (gradients were not tested in float32)."""
    from agrijax import facade_swap
    from agrijax.dssat import _x64
    from agrijax.facade_execution import require_default_options

    require_default_options("exp.gradient")
    _x64()
    trno = experiment._trno(treatment)
    x = experiment.inputs(trno, soil_evaporation)
    key = ("gradient", trno, facade_swap.swap_code(experiment, trno, soil_evaporation))
    table, trust, reasons, timing, point = _run(
        experiment,
        key,
        [x],
        [{}],
        x.published(),
        outputs,
        params,
        cultivar,
        scan_points,
    )
    table = table.drop(columns=["matured"]) if bool(table["matured"].all()) else table
    return Gradient(f"{experiment.name}_t{trno:02d}", point, table, trust, reasons, timing, int(scan_points))


def sensitivity(
    scenarios: Any,
    *,
    outputs: Sequence[str] | str = DEFAULT_OUTPUTS,
    params: Sequence[str] | str = ("P1", "P2", "P5", "G2", "G3", "PHINT"),
    cultivar: Mapping[str, float] | None = None,
    scan_points: int = SCAN_POINTS,
) -> Sensitivity:
    """The derivative of the end values with respect to cultivar coefficients on every scenario of
    ``scenarios`` (:meth:`agrijax.dssat.Scenarios.sensitivity`), each with its trust label: the
    arguments and the labels are those of :func:`gradient`; all scenarios run in one batched program.

    **Derivatives through phenology events are experimental** (``P1``, ``P2``, ``P5``, ``PHINT``). A
    coefficient is ``"validated gradient"`` over the batch only where every scenario's check passed;
    :attr:`Sensitivity.summary` counts them. The scenarios of ``scenarios`` are not in the
    validation set (compare them with DSSAT through :meth:`~agrijax.dssat.Scenarios.reference`). Runs
    in float64 on JAX's default device only (it refuses a non-default :func:`agrijax.options` block)."""
    from agrijax.dssat import _x64
    from agrijax.facade_execution import require_default_options

    require_default_options("scen.sensitivity")
    _x64()
    t = scenarios.table
    scn_rows = [
        {"scenario": i, "year": int(t.loc[i, "year"]), "sowing_shift": int(t.loc[i, "sowing_shift"])}
        for i in range(len(scenarios.runs))
    ]
    table, trust, reasons, timing, point = _run(
        scenarios,
        ("sensitivity",),
        scenarios.runs,
        scn_rows,
        scenarios.published,
        outputs,
        params,
        cultivar,
        scan_points,
    )
    return Sensitivity(scenarios.treatment, point, table, trust, reasons, timing, int(scan_points))
