"""The cultivar-calibration core behind :func:`agrijax.calib.calibrate`: data-free, simulator-agnostic.

A :class:`CultivarProblem` is a batch simulator ``simulate(theta [S, 6], tid [S]) -> [S, E]`` of the
six ``MZCER048.CUL`` coefficients (:data:`~agrijax.calib.dssat_day.CUL_ORDER`) on the treatments of
the problem, the observations (one :class:`Observed` per observed number, pointing at one entry of
its treatment) and the published (starting) cultivar. :func:`fit_cultivar` calibrates it:

1. **objective** - per observed code ``c`` (``ADAT``, ``HWAM``, ``LAID`` ...) the mean over its
   observations on the calibration treatments of the squared normalised residual, summed over the
   codes; the scale of a date code is one day, of any other code the mean absolute observed value (a
   squared normalised RMSE). Dates are integer day indices: the objective is piecewise constant in
   the phenology coefficients.
2. **sensitivity check** at the published cultivar: every coefficient scanned alone over its
   ``MINIMA`` / ``MAXIMA`` box (:data:`SCAN_BOX` points) and over the start region
   (+-:data:`START_SPREAD`, :data:`SCAN_START` points). **Inert** = the objective does not move over
   the box; **flat** = it does not move over the start region (the data carry no information there;
   e.g. P2 without a photoperiod signal). With ``params="auto"`` inert and flat coefficients are
   fixed at the published value (reported); with explicit ``params`` they are calibrated with a
   warning.
3. **optimisation** from ``starts`` points (the published cultivar and ``starts - 1`` points
   ``published x (1 + U(-START_SPREAD, START_SPREAD))``, inside the box less :data:`BOX_MARGIN`), all
   rows batched in one simulator call per generation:

   * ``"cma"`` (default): joint (mu/mu_w, lambda)-CMA-ES on every free coefficient (the strategy
     parameters of :func:`agrijax.calib.optim.cma_es`, Hansen 2016 Table 1) in the unconstrained
     (logit) coordinates, with **restarts** (a row whose best loss has not improved by
     :data:`CMA_GAIN_REL` for :data:`CMA_STALL` generations restarts from a new start point; the best
     point of the run that ended in the restart is kept as a candidate), :data:`CMA_BUDGET`
     evaluations per start;
   * ``"staged"``: stage 1 CMA-ES on the free phenology coefficients (P1, P2, P5, PHINT) against the
     dates and the LAI observed before the observed silking day (which the grain coefficients cannot
     move); stage 2 CMA-ES on the free growth coefficients (G2, G3) against the other targets; stage 3
     joint CMA-ES on all free coefficients from the stage-2 point with a small step
     (:data:`SIGMA_STAGE3`). Every stage is derivative-free (a gradient second stage is not part of
     this interface);
   * ``"adam"``: Adam on the coefficients whose gradient the trust report accepts on every treatment
     (:func:`agrijax.calib.ceres.ceres_gradient_plan`: P1, P2, P5, PHINT are derivative-free by
     default, the event gradients are experimental; G2 / G3 pass or fail pair by pair); the other
     free coefficients stay at the published value (reported). Needs a problem with ``jax_loss``.
4. **selection on the written coefficients**: the candidates are, per start, the best point of each
   CMA-ES run that ended in a restart, the start's overall best point and its :data:`TOP_K` best
   points of all its evaluations (unrounded objective), plus the published cultivar itself (the
   last run of a start does not end in a restart and adds no candidate of its own). Every
   candidate is rounded to the printed precision of a ``.CUL`` row (``problem.written``) and the
   one with the lowest objective **at the written values**
   is returned (a date on a knife edge can move by a day when the coefficients are rounded; the
   objective of the unrounded point is reported next to it). When no written candidate beats the
   published cultivar, the published cultivar is returned (``improved = False``, with a warning).

Nothing here reads files or runs DSSAT; :mod:`agrijax.calib.workflow` builds the problem from a DSSAT
experiment. The tuning constants below are documented defaults (harness choices), not model
coefficients.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agrijax.calib.dssat_day import CUL_ORDER, E_DATE, E_FINAL, OUT_NAMES, Entry

__all__ = [
    "BOX_MARGIN",
    "CMA_BUDGET",
    "CMA_GAIN_REL",
    "CMA_STALL",
    "METHODS",
    "PROBES",
    "SCAN_BOX",
    "SCAN_START",
    "SIGMA0",
    "START_SPREAD",
    "CalibrationWarning",
    "CultivarProblem",
    "FitResult",
    "Observed",
    "Treatment",
    "fit_cultivar",
]

# ------------------------------------------------------------------ tuning defaults (harness choices)
#: starts: the published cultivar and ``published x (1 + U(-START_SPREAD, START_SPREAD))``
START_SPREAD = 0.25
#: starts and restarts stay this fraction of the bound width inside either bound
BOX_MARGIN = 0.05
#: CMA-ES initial step [unconstrained (logit) units]
SIGMA0 = 0.5
#: stage-3 step of the staged method [logit units]: a local refinement of the stage-2 point
SIGMA_STAGE3 = 0.1
#: generations without an improvement of CMA_GAIN_REL (relative) before a CMA-ES row restarts
CMA_STALL = 25
CMA_GAIN_REL = 1e-2
#: a CMA-ES row whose step (times the largest covariance axis) falls below this restarts [logit units]
CMA_SIGMA_MIN = 1e-10
#: joint CMA-ES evaluations per start (the budget of the measured real-data calibrations)
CMA_BUDGET = 4000
#: staged method: evaluations per start of stages 1, 2, 3 (``budget`` scales all three)
STAGED_BUDGET = (2400, 400, 1200)
#: candidates per start kept for the selection on written values: the TOP_K best points (unrounded
#: objective) of all the start's evaluations, the best point of each CMA-ES run that ended in a
#: restart, and the start's overall best point
TOP_K = 8
#: Adam steps per start and learning rate [logit units per step]
ADAM_STEPS = 200
ADAM_LR = 0.05
#: sensitivity check: points of the box scan and of the start-region scan
SCAN_BOX = 41
SCAN_START = 101
#: a coefficient is inert when the objective's range over the box is at most this, relative to
#: max(1, the largest objective value on the scan)
INERT_RTOL = 1e-12
#: line-scan points of the gradient-trust report (method ``adam``; the default 41 misses jumps
#: narrower than 2.3 % of the bound width)
TRUST_SCAN = 201
#: floor of a target scale (a code observed as 0 everywhere)
SCALE_FLOOR = 1e-12
METHODS = ("cma", "staged", "adam")
PHENOLOGY = ("P1", "P2", "P5", "PHINT")
GROWTH = ("G2", "G3")

#: entries every treatment carries first (predictions for reports and the DSSAT check): silking,
#: maturity and emergence dates, yield, tops weight and grain number at simulated maturity
PROBES: tuple[tuple[str, Entry], ...] = (
    ("ADAT", Entry(E_DATE, 4)),
    ("MDAT", Entry(E_DATE, 10)),
    ("EDAT", Entry(E_DATE, 1)),
    ("HWAM", Entry(E_FINAL, OUT_NAMES.index("gwad"))),
    ("CWAM", Entry(E_FINAL, OUT_NAMES.index("cwad"))),
    ("H#AM", Entry(E_FINAL, OUT_NAMES.index("g_ad"))),
)


class CalibrationWarning(UserWarning):
    """Something the calibration did or found that the user should know (also in the result)."""


@dataclass(frozen=True)
class Treatment:
    """One treatment: its name, the simulated days with real forcing (``YYYYDDD``) and its entries
    (:data:`PROBES` first, then one per observation)."""

    name: str
    days: np.ndarray
    entries: tuple[Entry, ...]


@dataclass(frozen=True)
class Observed:
    """One observed number: treatment index, code, entry index in the treatment, value (a date as a
    day index on the treatment's days) and the observation day (``YYYYDDD``; for a date target the
    observed date)."""

    treatment: int
    code: str
    entry: int
    value: float
    day: int


@dataclass
class CultivarProblem:
    """What :func:`fit_cultivar` calibrates (module docstring)."""

    simulate: Callable[[np.ndarray, np.ndarray], np.ndarray]
    treatments: list[Treatment]
    observed: list[Observed]
    #: the published cultivar in CUL_ORDER
    published: np.ndarray
    #: ``theta [R, 6] -> [R, 6]``: the values a written ``.CUL`` row holds
    written: Callable[[np.ndarray], np.ndarray]
    #: ``(obs [B, E], winv [B, E]) -> (theta [6] -> [B])``: the per-treatment loss in JAX (method adam)
    jax_loss: Callable[[np.ndarray, np.ndarray], Callable[[Any], Any]] | None = None
    #: simulator statistics (compile / call time, program calls, seasons)
    stats: Callable[[], dict[str, Any]] | None = None


# ------------------------------------------------------------------ objective
class _Objective:
    """Observed arrays ``obs``, weights ``winv`` ``[B, E]`` of a set of treatments (module docstring)."""

    def __init__(
        self,
        problem: CultivarProblem,
        tset: Sequence[int],
        *,
        scales: dict[str, float] | None = None,
        keep: Callable[[Observed], bool] | None = None,
    ) -> None:
        b_n = len(problem.treatments)
        e_n = max(len(t.entries) for t in problem.treatments)
        self.tset = tuple(int(t) for t in tset)
        self.obs = np.zeros((b_n, e_n))
        self.winv = np.zeros((b_n, e_n))
        mine = [o for o in problem.observed if o.treatment in self.tset]
        self.codes = sorted({o.code for o in mine})
        self.n_obs: dict[str, int] = {}
        self.scales: dict[str, float] = {}
        for c in self.codes:
            obs_c = [o for o in mine if o.code == c]
            date = problem.treatments[obs_c[0].treatment].entries[obs_c[0].entry].kind == E_DATE
            if scales is not None and c in scales:
                s = scales[c]
            else:
                s = 1.0 if date else max(float(np.mean([abs(o.value) for o in obs_c])), SCALE_FLOOR)
            self.scales[c] = s
            self.n_obs[c] = len(obs_c)
            for o in obs_c:
                self.obs[o.treatment, o.entry] = o.value
                if keep is None or keep(o):
                    self.winv[o.treatment, o.entry] = 1.0 / (s * math.sqrt(len(obs_c)))

    def view(self, problem: CultivarProblem, keep: Callable[[Observed], bool]) -> _Objective:
        """The same objective (same scales and counts) on the observations ``keep`` accepts."""
        return _Objective(problem, self.tset, scales=self.scales, keep=keep)

    def sample_losses(self, y: np.ndarray, tid: np.ndarray) -> np.ndarray:
        return np.sum(((y - self.obs[tid]) * self.winv[tid]) ** 2, axis=1)


class _Evaluator:
    """Objective values of candidate batches ``theta [C, 6]`` on the treatments of one objective."""

    def __init__(self, problem: CultivarProblem, tset: Sequence[int]) -> None:
        self.problem = problem
        self.tset = np.asarray(tset, dtype=np.int64)
        self.candidates = 0

    def values(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Entry values ``[C * B, E]`` and treatment indices of the samples (candidate-major)."""
        theta = np.asarray(theta, dtype=float)
        c_n, b_n = theta.shape[0], self.tset.size
        tid = np.tile(self.tset, c_n)
        y = self.problem.simulate(np.repeat(theta, b_n, axis=0), tid)
        self.candidates += c_n
        return np.asarray(y), tid

    def losses(self, theta: np.ndarray, *objs: _Objective) -> list[np.ndarray]:
        y, tid = self.values(theta)
        c_n, b_n = np.asarray(theta).shape[0], self.tset.size
        return [o.sample_losses(y, tid).reshape(c_n, b_n).sum(axis=1) for o in objs]


# ------------------------------------------------------------------ coordinates
def _space() -> Any:
    from agrijax.calib.ceres import ceres_space

    return ceres_space(CUL_ORDER)


def _to_theta(sp: Any, z: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp

    return np.array(sp.to_physical(jnp.asarray(z)))  # a writable copy


def _to_z(sp: Any, theta: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp

    return np.array(sp.to_unconstrained(jnp.asarray(theta)))


def _inner_box(sp: Any) -> tuple[np.ndarray, np.ndarray]:
    return sp.lower + BOX_MARGIN * sp.width, sp.upper - BOX_MARGIN * sp.width


def _starts(
    sp: Any, center: np.ndarray, n: int, rng: np.random.Generator, *, first_center: bool
) -> np.ndarray:
    lo, hi = _inner_box(sp)
    th = center[None] * (1.0 + rng.uniform(-START_SPREAD, START_SPREAD, (n, center.size)))
    if first_center and n:
        th[0] = center
    return np.clip(th, lo, hi)


# ------------------------------------------------------------------ CMA-ES with restarts (batched rows)
def _cma_rows(
    evaluate: Callable[[np.ndarray], np.ndarray],
    z0: np.ndarray,
    dims: Sequence[int],
    *,
    max_evals: int,
    tol: float,
    seed: int,
    sigma0: float = SIGMA0,
    restart: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """(mu/mu_w, lambda)-CMA-ES on the coordinates ``dims`` of ``z`` (others fixed at ``z0``), one
    search per row, with restarts (module docstring). ``evaluate(z [M, 6]) -> loss [M]``; a row
    stops when its loss is <= ``tol`` or its budget is used."""
    from agrijax.calib.optim import _cma_defaults

    r_n = z0.shape[0]
    d = list(dims)
    n = len(d)
    lam = 4 + int(np.floor(3 * np.log(max(n, 1))))
    p = _cma_defaults(n, lam)
    rng = np.random.default_rng(seed)
    mean = z0[:, d].copy()
    sigma = np.full(r_n, sigma0)
    c = np.tile(np.eye(n), (r_n, 1, 1))
    pc = np.zeros((r_n, n))
    ps = np.zeros((r_n, n))
    best_z = z0.copy()
    best_l = np.full(r_n, np.inf)
    evals = np.zeros(r_n, int)
    done = np.zeros(r_n, bool) if max_evals >= lam else np.ones(r_n, bool)
    gen = np.zeros(r_n, int)
    last_gain = np.zeros(r_n, int)
    run_best = np.full(r_n, np.inf)
    run_best_z = z0.copy()
    restarts = np.zeros(r_n, int)
    top_z = np.repeat(z0[:, None, :], TOP_K, axis=1)
    top_l = np.full((r_n, TOP_K), np.inf)
    restart_z: list[np.ndarray] = []
    restart_row: list[np.ndarray] = []
    while not done.all():
        rows = np.nonzero(~done)[0]
        ev_, bv = np.linalg.eigh(c[rows])
        dsq = np.sqrt(np.maximum(ev_, 0.0))
        y = np.einsum("rij,rkj->rki", bv * dsq[:, None, :], rng.standard_normal((rows.size, lam, n)))
        x = mean[rows][:, None, :] + sigma[rows][:, None, None] * y
        zfull = np.repeat(z0[rows][:, None, :], lam, axis=1)
        zfull[:, :, d] = x
        f = np.asarray(evaluate(zfull.reshape(-1, z0.shape[1])), dtype=float)
        f = np.where(np.isfinite(f), f, np.inf).reshape(rows.size, lam)
        evals[rows] += lam
        order = np.argsort(f, axis=1)
        gb = f[np.arange(rows.size), order[:, 0]]
        better = gb < best_l[rows]
        best_l[rows[better]] = gb[better]
        best_z[rows[better]] = zfull[np.arange(rows.size), order[:, 0]][better]
        gain = gb < run_best[rows] * (1.0 - CMA_GAIN_REL)
        run_better = gb < run_best[rows]
        run_best_z[rows[run_better]] = zfull[np.arange(rows.size), order[:, 0]][run_better]
        run_best[rows] = np.minimum(run_best[rows], gb)
        # the TOP_K best points of every row so far (candidates of the written-value selection)
        all_l = np.concatenate([top_l[rows], f], axis=1)
        all_z = np.concatenate([top_z[rows], zfull], axis=1)
        keep = np.argsort(all_l, axis=1)[:, :TOP_K]
        top_l[rows] = np.take_along_axis(all_l, keep, axis=1)
        top_z[rows] = np.take_along_axis(all_z, keep[:, :, None], axis=1)
        gen[rows] += 1
        last_gain[rows[gain]] = gen[rows[gain]]
        ysel = np.take_along_axis(y, order[:, : p["mu"], None], axis=1)
        yw = np.einsum("m,rmn->rn", p["w"], ysel)
        mean[rows] = mean[rows] + sigma[rows][:, None] * yw
        inv_sqrt = np.einsum("rij,rj,rkj->rik", bv, 1.0 / np.maximum(dsq, 1e-300), bv)
        ps[rows] = (1 - p["cs"]) * ps[rows] + np.sqrt(p["cs"] * (2 - p["cs"]) * p["mueff"]) * np.einsum(
            "rij,rj->ri", inv_sqrt, yw
        )
        psn = np.linalg.norm(ps[rows], axis=1)
        hsig = (psn / np.sqrt(1 - (1 - p["cs"]) ** (2 * gen[rows])) / p["chin"] < 1.4 + 2 / (n + 1)).astype(
            float
        )
        pc[rows] = (1 - p["cc"]) * pc[rows] + hsig[:, None] * np.sqrt(
            p["cc"] * (2 - p["cc"]) * p["mueff"]
        ) * yw
        rank_mu = np.einsum("m,rmi,rmj->rij", p["w"], ysel, ysel)
        dh = (1 - hsig) * p["cc"] * (2 - p["cc"])
        cr = (
            (1 - p["c1"] - p["cmu"]) * c[rows]
            + p["c1"] * (np.einsum("ri,rj->rij", pc[rows], pc[rows]) + dh[:, None, None] * c[rows])
            + p["cmu"] * rank_mu
        )
        c[rows] = 0.5 * (cr + np.transpose(cr, (0, 2, 1)))
        sigma[rows] = sigma[rows] * np.exp((p["cs"] / p["damps"]) * (psn / p["chin"] - 1))
        small = sigma[rows] * np.sqrt(np.max(np.diagonal(c[rows], axis1=1, axis2=2), axis=1)) < CMA_SIGMA_MIN
        done[rows] = (best_l[rows] <= tol) | (evals[rows] + lam > max_evals)
        stuck = rows[~done[rows] & (small | (gen[rows] - last_gain[rows] >= CMA_STALL))]
        if stuck.size:
            if restart is None:
                done[stuck] = True
            else:
                restart_z.append(run_best_z[stuck].copy())
                restart_row.append(stuck.copy())
                mean[stuck] = restart(stuck)[:, d]
                sigma[stuck] = sigma0
                c[stuck] = np.eye(n)
                pc[stuck] = 0.0
                ps[stuck] = 0.0
                gen[stuck] = 0
                last_gain[stuck] = 0
                run_best[stuck] = np.inf
                restarts[stuck] += 1
    fin = np.isfinite(top_l)
    cand_z = [top_z[fin], *restart_z, best_z]
    cand_row = [np.nonzero(fin)[0], *restart_row, np.arange(r_n)]
    return {
        "z_best": best_z,
        "loss_best": best_l,
        "evals": evals,
        "popsize": lam,
        "restarts": restarts,
        "candidates": np.concatenate(cand_z),
        "candidate_row": np.concatenate(cand_row),
    }


# ------------------------------------------------------------------ result
@dataclass
class FitResult:
    """What :func:`fit_cultivar` found (the fields of :class:`agrijax.calib.CalibrationResult` that do
    not need DSSAT)."""

    theta_written: np.ndarray
    theta_unrounded: np.ndarray
    published: np.ndarray
    free: tuple[str, ...]
    fixed: dict[str, str]
    loss: dict[str, float]
    per_start: list[dict[str, Any]]
    sensitivity: dict[str, dict[str, Any]]
    trust: dict[str, Any]
    fit: list[dict[str, Any]]
    predictions: dict[str, dict[str, dict[str, float]]]
    holdout: dict[str, Any] | None
    targets: dict[str, int]
    scales: dict[str, float]
    method: str
    calls: dict[str, Any]
    wall_s: float
    #: False when no written candidate beat the published cultivar (then it is what is returned)
    improved: bool = True
    warnings: list[str] = field(default_factory=list)


def _warn(msgs: list[str], msg: str) -> None:
    msgs.append(msg)
    warnings.warn(msg, CalibrationWarning, stacklevel=3)


def _sensitivity(
    ev: _Evaluator, obj: _Objective, sp: Any, pub: np.ndarray
) -> tuple[dict[str, dict[str, Any]], float]:
    """Scans of every coefficient alone at the published cultivar (module docstring)."""
    grids = []
    for i in range(len(CUL_ORDER)):
        box = np.linspace(sp.lower[i], sp.upper[i], SCAN_BOX)
        start = np.clip(
            pub[i] * (1 + np.linspace(-START_SPREAD, START_SPREAD, SCAN_START)), sp.lower[i], sp.upper[i]
        )
        grids.append((box, start))
    pts = []
    for i, (box, start) in enumerate(grids):
        for gv in (*box, *start):
            t = pub.copy()
            t[i] = gv
            pts.append(t)
    pts.append(pub.copy())
    (loss,) = ev.losses(np.asarray(pts), obj)
    loss_pub = float(loss[-1])
    out: dict[str, dict[str, Any]] = {}
    k = 0
    for n in CUL_ORDER:
        f_box = loss[k : k + SCAN_BOX]
        k += SCAN_BOX
        f_start = loss[k : k + SCAN_START]
        k += SCAN_START
        inert = float(np.ptp(f_box)) <= INERT_RTOL * max(1.0, float(np.max(np.abs(f_box))))
        flat = bool(np.all(f_start == f_start[SCAN_START // 2]))
        out[n] = {
            "inert": bool(inert),
            "flat": flat,
            "loss_range_box": [float(np.min(f_box)), float(np.max(f_box))],
            "loss_range_start": [float(np.min(f_start)), float(np.max(f_start))],
        }
    return out, loss_pub


def _index_to_yrdoy(days: np.ndarray, idx: float) -> int:
    """``YYYYDDD`` of day index ``idx`` counted from the first simulated day (past the real days too)."""
    from datetime import date, timedelta

    d0 = int(days[0])
    d = date(d0 // 1000, 1, 1) + timedelta(days=d0 % 1000 - 1 + int(idx))
    return d.year * 1000 + d.timetuple().tm_yday


def _yrdoy_days_between(a: int, b: int) -> int:
    """Days from ``YYYYDDD`` ``b`` to ``a`` (positive when ``a`` is later)."""
    from datetime import date, timedelta

    da = date(a // 1000, 1, 1) + timedelta(days=a % 1000 - 1)
    db = date(b // 1000, 1, 1) + timedelta(days=b % 1000 - 1)
    return (da - db).days


def _predictions(problem: CultivarProblem, y: np.ndarray, tid: np.ndarray) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for s, b in enumerate(tid):
        tr = problem.treatments[int(b)]
        rec: dict[str, float] = {}
        for j, (code, e) in enumerate(PROBES):
            v = float(y[s, j])
            rec[code] = float(_index_to_yrdoy(tr.days, v)) if e.kind == E_DATE else v
        rec["MDAT_index"] = float(y[s, 1])
        rec["n_real_days"] = float(tr.days.size)
        out[tr.name] = rec
    return out


def _fit_table(
    problem: CultivarProblem,
    tset: Sequence[int],
    label: str,
    y_cal: np.ndarray,
    y_pub: np.ndarray,
    tid: np.ndarray,
) -> list[dict[str, Any]]:
    row_of = {int(b): s for s, b in enumerate(tid)}
    rows = []
    for o in problem.observed:
        if o.treatment not in tset:
            continue
        tr = problem.treatments[o.treatment]
        e = tr.entries[o.entry]
        s = row_of[o.treatment]
        cal, pub = float(y_cal[s, o.entry]), float(y_pub[s, o.entry])
        if e.kind == E_DATE:
            obs_v: float = float(_index_to_yrdoy(tr.days, o.value))
            cal, pub = float(_index_to_yrdoy(tr.days, cal)), float(_index_to_yrdoy(tr.days, pub))
            kind = "date"
        else:
            obs_v = o.value
            kind = "final" if e.kind == E_FINAL else "series"
        rows.append(
            {
                "set": label,
                "treatment": tr.name,
                "code": o.code,
                "kind": kind,
                "day": int(o.day),
                "observed": obs_v,
                "calibrated": cal,
                "published": pub,
            }
        )
    return rows


def fit_cultivar(
    problem: CultivarProblem,
    calibration: Sequence[int],
    *,
    holdout: Sequence[int] = (),
    params: str | Sequence[str] = "auto",
    method: str = "cma",
    starts: int = 8,
    seed: int = 0,
    budget: int | None = None,
    tol: float = 0.0,
) -> FitResult:
    """Calibrate ``problem`` on the treatments ``calibration`` (indices; module docstring).

    ``budget``: evaluations per start (CMA-ES; the staged stages scale with it) or Adam steps;
    ``None`` = :data:`CMA_BUDGET` / :data:`STAGED_BUDGET` / :data:`ADAM_STEPS`. ``tol``: a CMA-ES row
    stops once its objective is at most ``tol`` (0: the budget is used; synthetic problems with an
    exact solution can stop early). ``holdout`` treatments are evaluated, never fitted."""
    t0 = time.perf_counter()
    if method not in METHODS:
        raise ValueError(f"method {method!r}: one of {METHODS}")
    if starts < 1:
        raise ValueError("starts must be >= 1")
    cal = tuple(int(b) for b in calibration)
    hold = tuple(int(b) for b in holdout)
    if not cal:
        raise ValueError("no calibration treatment")
    if set(cal) & set(hold):
        raise ValueError(f"treatments both calibrated and held out: {sorted(set(cal) & set(hold))}")
    msgs: list[str] = []
    sp = _space()
    pub = np.asarray(problem.published, dtype=float).copy()
    outside = {
        n: (float(pub[i]), float(sp.lower[i]), float(sp.upper[i]))
        for i, n in enumerate(CUL_ORDER)
        if not sp.lower[i] <= pub[i] <= sp.upper[i]
    }
    if outside:
        _warn(
            msgs,
            f"published values outside the MINIMA / MAXIMA box (value, min, max): {outside}; the calibration "
            "searches inside the box",
        )
    ev = _Evaluator(problem, cal)
    obj = _Objective(problem, cal)
    if not obj.codes:
        raise ValueError("no observation on the calibration treatments")
    # ---------------------------------------------------------------- sensitivity and free set
    sens, loss_pub = _sensitivity(ev, obj, sp, pub)
    fixed: dict[str, str] = {}
    if isinstance(params, str):
        if params != "auto":
            raise ValueError(f"params: 'auto' or a sequence of {CUL_ORDER}, got {params!r}")
        wanted = list(CUL_ORDER)
        for n in CUL_ORDER:
            if sens[n]["inert"]:
                fixed[n] = "inert: the objective does not change over the MINIMA-MAXIMA box"
            elif sens[n]["flat"]:
                fixed[n] = (
                    f"flat: the objective does not change within +-{START_SPREAD:.0%} of the published value"
                )
        if fixed:
            _warn(msgs, f"fixed at the published values (no information in these observations): {fixed}")
    else:
        wanted = [str(n) for n in params]
        bad = [n for n in wanted if n not in CUL_ORDER]
        if bad or not wanted or len(set(wanted)) != len(wanted):
            raise ValueError(f"params {wanted}: distinct names out of {CUL_ORDER}")
        for n in CUL_ORDER:
            if n not in wanted:
                fixed[n] = "not requested"
        weak = [n for n in wanted if sens[n]["inert"] or sens[n]["flat"]]
        if weak:
            _warn(
                msgs,
                f"calibrating {weak} although the objective does not move with them (inert or flat); "
                "their calibrated values carry no information",
            )
    free = [n for n in CUL_ORDER if n in wanted and n not in fixed]
    trust: dict[str, Any] = {
        "derivative_free_by_default": list(PHENOLOGY),
        "note": "P1, P2, P5, PHINT: event gradients are experimental; derivative-free unless named",
    }
    # ---------------------------------------------------------------- adam: gradient plan first
    if method == "adam" and free:
        from agrijax.calib.ceres import ceres_gradient_plan
        from agrijax.calib.trust import TrustConfig, trust_report

        if problem.jax_loss is None:
            raise ValueError("method 'adam' needs a problem with a JAX loss (jax_loss)")
        # the mode-bound loss itself (trust_report jits it and derives its exact-mode counterpart)
        f = problem.jax_loss(obj.obs[list(cal)], obj.winv[list(cal)])
        names = [problem.treatments[b].name for b in cal]
        # the exact counterpart is derived (level 2 judges it); the unrounded model is a diagnostic only
        rep = trust_report(
            f, pub, sp.lower, sp.upper, CUL_ORDER, names, TrustConfig(n_scan=TRUST_SCAN), diagnostics=False
        )
        plan = ceres_gradient_plan(rep)
        trust.update(
            {
                "plan": plan.table(),
                "classes": {
                    n: {
                        t: (o["class"], o["level"], o["n_jumps"])
                        for t, o in rep["params"][n]["outputs"].items()
                    }
                    for n in CUL_ORDER
                },
            }
        )
        trusted = [n for n in free if plan.method[n] == "gradient"]
        for n in free:
            if n not in trusted:
                fixed[n] = f"gradient not trusted ({plan.method[n]}): kept at the published value"
        if not trusted:
            raise ValueError(
                f"method 'adam': no free coefficient has a trusted gradient on every treatment "
                f"({ {n: plan.method[n] for n in free} }); use method='cma'"
            )
        _warn(
            msgs,
            f"adam calibrates {trusted} only (trusted gradients); {sorted(set(free) - set(trusted))} kept",
        )
        free = trusted
    if not free:
        raise ValueError(f"no coefficient left to calibrate (fixed: {fixed})")
    if set(free) & set(GROWTH) and not ({"H#AM", "G#AD"} & set(obj.codes)):
        _warn(
            msgs,
            "G2 / G3 are free but no grain number (H#AM or G#AD) is observed: yield, biomass and "
            "LAI do not separate kernel number from kernel weight, so G2 and G3 trade off along a ridge",
        )
    dims = [CUL_ORDER.index(n) for n in free]
    fixed_mask = np.asarray([n not in free for n in CUL_ORDER], dtype=bool)
    # ---------------------------------------------------------------- optimisation
    rng = np.random.default_rng(seed)
    th0 = _starts(sp, pub, starts, rng, first_center=True)
    th0[:, fixed_mask] = pub[fixed_mask]
    z0 = _to_z(sp, th0)

    def theta_of(z: np.ndarray) -> np.ndarray:
        th = _to_theta(sp, z)
        th[:, fixed_mask] = pub[fixed_mask]
        return th

    def restart(rows: np.ndarray) -> np.ndarray:
        return _to_z(sp, _starts(sp, pub, rows.size, rng, first_center=False))

    stage: dict[str, Any] = {}
    cand_z: list[np.ndarray] = []
    cand_row: list[np.ndarray] = []
    if method == "cma":
        n_ev = CMA_BUDGET if budget is None else int(budget)
        r = _cma_rows(
            lambda z: ev.losses(theta_of(z), obj)[0],
            z0,
            dims,
            max_evals=n_ev,
            tol=tol,
            seed=seed + 1,
            restart=restart,
        )
        stage["cma"] = {
            "evals": r["evals"].tolist(),
            "popsize": r["popsize"],
            "restarts": r["restarts"].tolist(),
        }
        cand_z.append(r["candidates"])
        cand_row.append(r["candidate_row"])
    elif method == "staged":
        total_b = sum(STAGED_BUDGET)
        budgets = (
            STAGED_BUDGET
            if budget is None
            else tuple(max(1, round(budget * b / total_b)) for b in STAGED_BUDGET)
        )
        adat = {o.treatment: o.value for o in problem.observed if o.code == "ADAT" and o.treatment in cal}

        def is_date(o: Observed) -> bool:
            return problem.treatments[o.treatment].entries[o.entry].kind == E_DATE

        def stage1(o: Observed) -> bool:
            e = problem.treatments[o.treatment].entries[o.entry]
            pre = o.code == "LAID" and o.treatment in adat and e.t < adat[o.treatment]
            return is_date(o) or pre

        plan = (
            (PHENOLOGY, obj.view(problem, stage1), budgets[0], SIGMA0, True),
            (GROWTH, obj.view(problem, lambda o: not is_date(o)), budgets[1], SIGMA0, True),
            (CUL_ORDER, obj, budgets[2], SIGMA_STAGE3, False),
        )
        z = z0.copy()
        for k, (group, o_view, n_ev, sig, restarts_on) in enumerate(plan, start=1):
            dd = [CUL_ORDER.index(n) for n in group if n in free]
            if not dd:
                stage[f"stage{k}"] = {"skipped": "no free coefficient"}
            else:
                zc = z.copy()

                def restart_k(rows: np.ndarray, dd: list[int] = dd, zc: np.ndarray = zc) -> np.ndarray:
                    out = zc[rows].copy()
                    out[:, dd] = restart(rows)[:, dd]
                    return out

                r = _cma_rows(
                    lambda zz, o_view=o_view: ev.losses(theta_of(zz), o_view)[0],
                    z,
                    dd,
                    max_evals=n_ev,
                    tol=0.0,
                    seed=seed + 1 + k,
                    sigma0=sig,
                    restart=restart_k if restarts_on else None,
                )
                z = r["z_best"].copy()
                stage[f"stage{k}"] = {
                    "params": [CUL_ORDER[i] for i in dd],
                    "evals": r["evals"].tolist(),
                    "restarts": r["restarts"].tolist(),
                }
            if k == 1:  # stage 2 starts from the start values of the growth coefficients
                g_dd = [CUL_ORDER.index(n) for n in GROWTH]
                z[:, g_dd] = z0[:, g_dd]
            if k >= 2 and dd:  # the stage-2 and the stage-3 points are candidates
                cand_z.append(r["candidates"])
                cand_row.append(r["candidate_row"])
        if not cand_z:
            cand_z.append(z.copy())
            cand_row.append(np.arange(starts))
    else:  # adam
        import jax
        import jax.numpy as jnp

        from agrijax.calib.optim import AdamConfig, adam

        assert problem.jax_loss is not None
        f = problem.jax_loss(obj.obs[list(cal)], obj.winv[list(cal)])
        zfix = jnp.asarray(z0[0])
        idx = jnp.asarray(dims)

        def total(zs: Any) -> Any:
            zz = zfix.at[idx].set(zs)
            th = sp.to_physical(zz)
            th = jnp.where(jnp.asarray(fixed_mask), jnp.asarray(pub), th)
            return jnp.sum(f(th))

        vg = jax.jit(jax.vmap(jax.value_and_grad(total)))
        steps = ADAM_STEPS if budget is None else int(budget)
        r_adam = adam(vg, z0[:, dims], AdamConfig(lr=ADAM_LR, steps=steps))
        z = z0.copy()
        z[:, dims] = r_adam.z_best
        stage["adam"] = {"steps": steps, "n_nonfinite": int(r_adam.extra["n_nonfinite"])}
        ev.candidates += steps * starts
        cand_z.append(z)
        cand_row.append(np.arange(starts))
    # ---------------------------------------------------------------- selection on written values
    # every candidate at written precision, and the published cultivar itself (row -1)
    th_unr = np.concatenate([*[theta_of(zz) for zz in cand_z], pub[None]])
    rows_of = np.concatenate([*cand_row, [-1]])
    th_w = np.asarray(problem.written(th_unr[:-1]), dtype=float)
    th_w[:, fixed_mask] = pub[fixed_mask]
    th_w = np.concatenate([th_w, pub[None]])
    l_w, l_u = (ev.losses(th_w, obj)[0], ev.losses(th_unr, obj)[0])
    best = int(np.argmin(l_w))
    per_start = [
        {
            "start": int(rows_of[k]),
            "loss_written": float(np.min(l_w[rows_of == rows_of[k]])),
            "loss_unrounded": float(np.min(l_u[rows_of == rows_of[k]])),
            "n_candidates": int(np.sum(rows_of == rows_of[k])),
        }
        for k in sorted({int(np.nonzero(rows_of == r)[0][0]) for r in np.unique(rows_of)})
    ]
    improved = bool(l_w[best] < loss_pub) and rows_of[best] >= 0
    if not improved:
        best = th_w.shape[0] - 1
        _warn(
            msgs,
            f"no candidate improves on the published cultivar at written precision (best written "
            f"{float(np.min(l_w[:-1])):.4g} vs published {loss_pub:.4g}): the published cultivar is returned",
        )
    n_c = th_w.shape[0] - 1
    crossed = [k for k in range(n_c) if l_w[k] > 1.1 * l_u[k] + 1e-12]
    if crossed:
        _warn(
            msgs,
            f"{len(crossed)} of {n_c} candidates change their objective by more than 10 % when rounded to "
            "the .CUL precision (a date on a knife edge); the selection uses the written values",
        )
    # ---------------------------------------------------------------- reports
    everything = tuple(cal) + hold
    rep_ev = _Evaluator(problem, everything)
    y, tid = rep_ev.values(np.stack([th_w[best], pub]))
    b_n = len(everything)
    y_cal, y_pub, tid1 = y[:b_n], y[b_n:], tid[:b_n]
    fit = _fit_table(problem, cal, "calibration", y_cal, y_pub, tid1)
    preds = {
        "calibrated": _predictions(problem, y_cal, tid1),
        "published": _predictions(problem, y_pub, tid1),
    }
    late = [n for n, p in preds["calibrated"].items() if p["MDAT_index"] >= p["n_real_days"]]
    if late:
        _warn(
            msgs,
            f"calibrated maturity falls on padded forcing days (after the reference season) for {late}: "
            "those seasons are simulated on repeated weather, DSSAT will differ there",
        )
    holdout_res = None
    if hold:
        obj_h = _Objective(problem, hold, scales=obj.scales)
        lh_w = lh_p = float("nan")
        if obj_h.codes:
            (lh,) = _Evaluator(problem, hold).losses(np.stack([th_w[best], pub]), obj_h)
            lh_w, lh_p = float(lh[0]), float(lh[1])
        holdout_res = {
            "treatments": [problem.treatments[b].name for b in hold],
            "loss_written": float(lh_w),
            "loss_published": float(lh_p),
            "targets": dict(obj_h.n_obs),
            "fit": _fit_table(problem, hold, "holdout", y_cal, y_pub, tid1),
        }
    calls: dict[str, Any] = {
        "candidates": ev.candidates + rep_ev.candidates,
        "starts": starts,
        "stages": stage,
    }
    if problem.stats is not None:
        calls.update(problem.stats())
    return FitResult(
        theta_written=th_w[best],
        theta_unrounded=th_unr[best],
        published=pub,
        free=tuple(free),
        fixed=fixed,
        loss={"written": float(l_w[best]), "unrounded": float(l_u[best]), "published": loss_pub},
        per_start=per_start,
        sensitivity=sens,
        trust=trust,
        fit=fit,
        predictions=preds,
        holdout=holdout_res,
        targets=dict(obj.n_obs),
        scales=dict(obj.scales),
        method=method,
        calls=calls,
        wall_s=time.perf_counter() - t0,
        improved=improved,
        warnings=msgs,
    )
