"""Acceptance of the adaptive Richards mode on the M1 replays: year tables, gradients, staircase scan.

Helper module of ``test_richards_adaptive_years.py`` / ``test_richards_adaptive_grad_catpa.py`` and a
script (``python tests/integration/richards_adaptive_years.py years|grad|staircase ...``) that writes
the tables under ``<data>/validation/w1_c/``.

The replay is the M1 recipe of :mod:`richards_years` (RZWQM2's own infiltration, evaporation and
uptake prescribed; drainage and profile computed) in ``restart`` mode: every calendar year starts
from RZWQM2's profile of the previous 31 December. The years of a site run as the lanes of one
``vmap`` (:func:`year_lanes`; a lane is padded with empty days after its year's end) through
:func:`~agrijax.processes.soil_water.richards.richards_day` with
``AdaptiveStepping.tier("exact" | "fast")`` (``dt_max`` 0.1 h / 0.25 h).

Compared per site-year:

* against RZWQM2: the daily storage RMSE (M1: < 0.05 cm) and the per-layer theta RMSE;
* against the converged reference (``validation/w1_e0/ref/<site>_m1_fix_cnfb_960.npz``: Crank-Nicolson with
  an alpha = 1 fallback, 960 steps a day, Newton to 1e-10; *clean* site-years had no unconverged
  step and agree with 1920 steps a day to <= 1.1e-5 cm on CA-TPA, ``e0_reference_years.csv``): the
  largest daily storage difference, i.e. the numerical error of the adaptive stepping;
* the counters (steps, rejects, Newton evaluations, unconverged steps, exhausted budgets, active-set
  steps), the largest per-step balance error and the day balance.

:func:`year_function` and :func:`frozen_year_function` give one year as a function of common factors
on the five Brooks-Corey parameter classes (``SCALED``, all horizons at once), for the gradient
checks and the staircase scan: the value is
the adaptive run; the frozen variant replays the step tables of a given run
(:func:`~agrijax.processes.soil_water.richards_adaptive.replay_segment`), which is the function the
gradient is of.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # script use
    sys.path.insert(0, str(HERE))

from richards_years import (  # noqa: E402
    BATCH,
    M1_STORAGE_RMSE,
    RunReplay,
    catpa_replay,
    run_scenario,
)

#: the 8 comparable scenarios besides CA-TPA (test_richards_years.COMPARABLE)
OTHER_SITES = (
    "CA-ER1",
    "CA-MA1",
    "US-Mj1",
    "US-S2",
    "US-manilacotton",
    "US_OPE",
    "US_Rockfish",
    "US_Rockford_Alfalfa",
)
SITES = ("CA-TPA", *OTHER_SITES)
TIERS = ("exact", "fast")
#: outputs of the converged reference (solutions and their self-check)
E0_SUB = Path("validation/w1_e0")
#: saved RZWQM2 runs of the scenarios (the RZWQM2 binary re-runs them when absent)
H14_RUNS = Path("rzwqm_runs/h1_4_final")
OUT_SUB = Path("validation/w1_c")
HOURS = 24
#: the Brooks-Corey parameter classes scaled by a common factor (all horizons)
SCALED = ("lambda_", "hb", "ksat", "theta_r", "theta_s")
#: per-day flux fields kept from a run
FLUX_KEYS = (
    "drainage",
    "runoff",
    "evaporation_deficit",
    "uptake_cut",
    "balance_error",
    "n_clamp",
    "n_steps",
    "n_rejects",
    "n_newton",
    "n_unconverged",
    "budget_exhausted",
    "n_active",
    "dt_min_used",
    "step_balance_max",
)


def replay_for(site: str, data_dir: Path) -> RunReplay:
    """The reference run of a site: CA-TPA 2015-2023, or the saved run (re-run when absent)."""
    if site == "CA-TPA":
        return catpa_replay(data_dir)
    run_dir = data_dir / H14_RUNS / site
    staged = data_dir / H14_RUNS / "_stage" / site / "Scenario"
    scen = staged if staged.is_dir() else data_dir / BATCH / site / "Scenario"
    if not run_dir.is_dir():
        root = Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "w1b_runs"
        rec = run_scenario(site, data_dir, root, root / "_stage")
        if not rec["ok"]:
            raise RuntimeError(f"{site}: {rec['error']}")
        run_dir, scen = Path(rec["out"]), Path(rec["source"])
    return RunReplay(site, run_dir, scen)


# ------------------------------------------------------------------------------ year lanes


class Lanes(NamedTuple):
    """The years of a run as ``vmap`` lanes (restart mode), padded to the longest year."""

    th0: np.ndarray  # [Y, n] start profile of each year
    sup: np.ndarray  # [Y, D, 24] hourly supply [cm h-1]
    eva: np.ndarray  # [Y, D, 24] hourly evaporation [cm h-1]
    upt: np.ndarray  # [Y, D, n] uptake per layer [cm d-1]
    idx: np.ndarray  # [Y, D] day index into the run, -1 on padding
    years: np.ndarray  # [Y]


def year_lanes(rep: RunReplay, years: Sequence[int] | None = None) -> Lanes:
    """Restart lanes: year ``y`` starts from RZWQM2's ``LAYER.PLT`` profile of the day before (the
    first year of the run from ``rzinit.dat``), exactly as ``RunReplay.run(mode="restart")``."""
    reset, start = rep.restart_profiles("restart")
    ys = np.unique(rep.years) if years is None else np.asarray(years)
    d_max = max(int((rep.years == y).sum()) for y in ys)
    n_y, n = len(ys), rep.grid.n_node
    idx = -np.ones((n_y, d_max), int)
    sup = np.zeros((n_y, d_max, HOURS))
    eva = np.zeros((n_y, d_max, HOURS))
    upt = np.zeros((n_y, d_max, n))
    th0 = np.zeros((n_y, n))
    for j, y in enumerate(ys):
        ii = np.flatnonzero(rep.years == y)
        idx[j, : len(ii)] = ii
        sup[j, : len(ii)] = rep.supply[ii]
        eva[j, : len(ii)] = rep.evap_hourly[ii]
        upt[j, : len(ii)] = rep.uptake[ii]
        th0[j] = rep.theta_init if ii[0] == 0 else start[ii[0]]
        assert ii[0] == 0 or reset[ii[0]]
    return Lanes(th0, sup, eva, upt, idx, ys)


def adaptive_params(soil: Any, grid: Any, tier: str, **overrides: Any) -> Any:
    """``RichardsParams`` with ``AdaptiveStepping.tier(tier)``; ``overrides`` that are fields of the
    problem's ``RichardsConfig`` (the conventions) go to it, the rest to the stepping."""
    import dataclasses

    from agrijax.processes.soil_water.richards import AdaptiveStepping, RichardsConfig, RichardsParams

    phys = {f.name for f in dataclasses.fields(RichardsConfig)}
    config = RichardsConfig(**{k: v for k, v in overrides.items() if k in phys})
    stepping = AdaptiveStepping.tier(tier, **{k: v for k, v in overrides.items() if k not in phys})
    return RichardsParams(soil=soil, grid=grid, config=config, stepping=stepping)


def run_years(rep: RunReplay, tier: str, **overrides: Any) -> tuple[dict[str, np.ndarray], float, float]:
    """All years of ``rep`` in one ``vmap`` through ``richards_day`` with ``AdaptiveStepping.tier(tier)``.

    Returns the per-day outputs in run order (``storage``, ``theta`` and :data:`FLUX_KEYS`), the
    compile and the run wall time [s].
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    from agrijax.processes.soil_water.richards import SoilWater, richards_day

    lanes = year_lanes(rep)
    params = adaptive_params(rep.soil, rep.grid, tier, **overrides)
    grid, soil = rep.grid, rep.soil

    def lane(th0: Any, sup: Any, eva: Any, upt: Any) -> Any:
        def body(w: Any, f: Any) -> Any:
            w2 = richards_day(w, params, *f)
            return w2, {
                "storage": w2.storage(grid),
                "theta": w2.theta,
                **{k: getattr(w2.flux, k) for k in FLUX_KEYS},
            }

        return lax.scan(body, SoilWater.from_theta(th0, soil), (sup, eva, upt))[1]

    args = tuple(jnp.asarray(x) for x in (lanes.th0, lanes.sup, lanes.eva, lanes.upt))
    t0 = time.perf_counter()
    comp = jax.jit(jax.vmap(lane)).lower(*args).compile()
    t_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = jax.block_until_ready(comp(*args))
    t_r = time.perf_counter() - t0
    m = lanes.idx >= 0
    order = np.argsort(lanes.idx[m])
    return {k: np.asarray(v)[m][order] for k, v in out.items()}, t_c, t_r


# ------------------------------------------------------------------------------ converged reference


class Reference(NamedTuple):
    storage: np.ndarray  # [day]
    theta: np.ndarray  # [day, n]
    clean: dict[int, bool]  # year -> the reference has no unconverged step
    self_check: dict[int, float]  # year -> |960 - 1920 steps a day| storage [cm]


def e0_reference(data_dir: Path, site: str) -> Reference | None:
    """The converged reference of a site (None when its outputs are not in the data tree)."""
    path = data_dir / E0_SUB / "ref" / f"{site}_m1_fix_cnfb_960.npz"
    table = data_dir / E0_SUB / "e0_reference_years.csv"
    if not path.is_file() or not table.is_file():
        return None
    ref = np.load(path)
    t = pd.read_csv(table)
    t = t[t.site == site].set_index("year")
    return Reference(
        np.asarray(ref["storage"]),
        np.asarray(ref["theta"]),
        {int(y): bool(v) for y, v in t["clean"].items()},
        {int(y): float(v) for y, v in t["ref_vs_1920_cm"].items()},
    )


def year_table(rep: RunReplay, r: dict[str, np.ndarray], ref: Reference | None, **tags: Any) -> pd.DataFrame:
    """Per-year metrics of one adaptive run ``r``."""
    rows = []
    for y in np.unique(rep.years):
        k = rep.years == y
        ds = r["storage"][k] - rep.storage[k]
        conv_days = r["n_unconverged"][k] == 0.0
        row: dict[str, Any] = {
            "site": rep.site,
            **tags,
            "year": int(y),
            "n_days": int(k.sum()),
            "storage_rmse_cm": float(np.sqrt(np.mean(ds**2))),
            "theta_rmse": float(np.sqrt(np.mean((r["theta"][k] - rep.theta[k]) ** 2))),
            "M1_pass": bool(np.sqrt(np.mean(ds**2)) < M1_STORAGE_RMSE),
            "drainage_cm": float(r["drainage"][k].sum()),
            "ref_drainage_cm": float(rep.deep_seepage[k].sum()),
            "runoff_cm": float(r["runoff"][k].sum()),
            "evaporation_deficit_cm": float(r["evaporation_deficit"][k].sum()),
            "uptake_cut_cm": float(r["uptake_cut"][k].sum()),
            **{
                c: float(r[c][k].sum())
                for c in (
                    "n_steps",
                    "n_rejects",
                    "n_newton",
                    "n_unconverged",
                    "budget_exhausted",
                    "n_active",
                    "n_clamp",
                )
            },
            "newton_per_day": float(r["n_newton"][k].mean()),
            "steps_per_day": float(r["n_steps"][k].mean()),
            "dt_min_used_h": float(r["dt_min_used"][k].min()),
            "n_days_unconverged": int((~conv_days).sum()),
            # the largest per-step balance of the accepted steps, on fully converged days and overall
            "step_balance_max_conv_cm": float(r["step_balance_max"][k][conv_days].max(initial=0.0)),
            # ... and on the converged days without a node held at the dry bound (``n_active == 0``)
            "step_balance_max_free_cm": float(
                r["step_balance_max"][k][conv_days & (r["n_active"][k] == 0.0)].max(initial=0.0)
            ),
            "n_days_active": int((r["n_active"][k] > 0.0).sum()),
            "step_balance_max_cm": float(r["step_balance_max"][k].max()),
            "day_balance_max_conv_cm": float(np.abs(r["balance_error"][k][conv_days]).max(initial=0.0)),
            "day_balance_max_cm": float(np.abs(r["balance_error"][k]).max()),
            "year_balance_conv_cm": float(np.abs(r["balance_error"][k][conv_days]).sum()),
            "year_balance_cm": float(np.abs(r["balance_error"][k]).sum()),
            "worst_day": str(rep.days[k][int(np.argmax(np.abs(ds)))]),
        }
        if ref is not None:
            e = r["storage"][k] - ref.storage[k]
            rmse_ref = float(np.sqrt(np.mean((ref.storage[k] - rep.storage[k]) ** 2)))
            row.update(
                ref_clean=ref.clean.get(int(y), False),
                ref_self_check_cm=ref.self_check.get(int(y), np.nan),
                err_ref_max_cm=float(np.max(np.abs(e))),
                err_ref_end_cm=float(e[-1]),
                err_ref_theta_max=float(np.max(np.abs(r["theta"][k] - ref.theta[k]))),
                ref_storage_rmse_cm=rmse_ref,
                d_rmse_vs_ref_cm=row["storage_rmse_cm"] - rmse_ref,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def day_row(rep: RunReplay, r: dict[str, np.ndarray], ref: Reference | None, day: str) -> dict[str, Any]:
    """The counters and errors of one day (the 2016-09-10 storm of CA-TPA)."""
    i = int(np.flatnonzero(rep.days == np.datetime64(day))[0])
    row = {"day": day, "infiltration_cm": float(rep.infiltration[i])}
    row.update({k: float(r[k][i]) for k in FLUX_KEYS})
    row["storage_diff_rzwqm_cm"] = float(r["storage"][i] - rep.storage[i])
    if ref is not None:
        row["err_ref_cm"] = float(r["storage"][i] - ref.storage[i])
        row["err_ref_prev_cm"] = float(r["storage"][i - 1] - ref.storage[i - 1])
    return row


def site_years(site: str, data_dir: Path, tiers: Sequence[str] = TIERS, **overrides: Any) -> dict[str, Any]:
    """One site: per-year tables of every tier, the run outputs and the timings
    (``overrides`` of :func:`adaptive_params`, for variants)."""
    rep = replay_for(site, data_dir)
    ref = e0_reference(data_dir, site)
    out: dict[str, Any] = {"rep": rep, "ref": ref, "tables": {}, "runs": {}, "walls": {}}
    for tier in tiers:
        r, t_c, t_r = run_years(rep, tier, **overrides)
        out["runs"][tier] = r
        out["walls"][tier] = (t_c, t_r)
        out["tables"][tier] = year_table(rep, r, ref, tier=tier)
    if set(tiers) == set(
        TIERS
    ):  # the two tiers against each other (a check where the reference is not clean)
        d = np.abs(out["runs"]["exact"]["storage"] - out["runs"]["fast"]["storage"])
        by_year = {int(y): float(d[rep.years == y].max()) for y in np.unique(rep.years)}
        for tier in tiers:
            t = out["tables"][tier]
            t["d_exact_fast_max_cm"] = t["year"].map(by_year)
    return out


def all_site_years(
    data_dir: Path, sites: Sequence[str] = SITES, workers: int | None = None, **overrides: Any
) -> dict[str, Any]:
    """:func:`site_years` of several sites, concurrently (one XLA program per site and tier)."""
    n = workers or min(len(sites), os.cpu_count() or 1)
    with ThreadPoolExecutor(n) as ex:
        res = list(ex.map(lambda s: site_years(s, data_dir, **overrides), sites))
    return dict(zip(sites, res, strict=True))


# ------------------------------------------------------------------------------ one year as a function


class YearOut(NamedTuple):
    storage_end: Any  # 31 December storage [cm]
    drainage: Any  # the year's drainage [cm]
    m1_loss: Any  # daily storage RMSE against RZWQM2 [cm]
    n_steps: Any  # [D] accepted steps per day
    n_rejects: Any  # [D]
    n_newton: Any  # [D]
    n_unconverged: Any  # [D]
    table_hash: Any  # [D] a fingerprint of each day's step table (t0, dt, alpha, bc)


#: a restart year, or a window ``(first day, last day)`` restarted from RZWQM2's profile of the day
#: before (``LAYER.PLT``)
Span = int | tuple[str, str]


def _year_inputs(rep: RunReplay, year: Span) -> tuple[Any, ...]:
    import jax.numpy as jnp

    if isinstance(year, tuple):  # a window
        i0 = int(np.flatnonzero(rep.days == np.datetime64(year[0]))[0])
        i1 = int(np.flatnonzero(rep.days == np.datetime64(year[1]))[0]) + 1
        sl = slice(i0, i1)
        th0 = rep.theta_init if i0 == 0 else rep.theta[i0 - 1]
        return tuple(
            jnp.asarray(x)
            for x in (th0, rep.supply[sl], rep.evap_hourly[sl], rep.uptake[sl], rep.storage[sl])
        )
    lanes = year_lanes(rep, [year])
    n_d = int((lanes.idx[0] >= 0).sum())
    ii = lanes.idx[0, :n_d]
    return (
        jnp.asarray(lanes.th0[0]),
        jnp.asarray(lanes.sup[0, :n_d]),
        jnp.asarray(lanes.eva[0, :n_d]),
        jnp.asarray(lanes.upt[0, :n_d]),
        jnp.asarray(rep.storage[ii]),
    )


def scaled_soil(soil: Any, scales: Any) -> Any:
    """``soil`` with each class of :data:`SCALED` multiplied by ``scales[i]`` (all horizons)."""
    return soil.replace(**{f: getattr(soil, f) * scales[i] for i, f in enumerate(SCALED)})


def _table_weights(n_tab: int) -> Any:
    import jax.numpy as jnp

    rng = np.random.default_rng(20260926)
    return jnp.asarray(rng.integers(1, 2**31, (4, n_tab)), jnp.int64)


def table_fingerprint(tr: Any, wts: Any) -> Any:
    """An exact fingerprint of a day's step table: an int64 (wrapping) weighted sum of the bit
    patterns of ``t0``, ``dt``, ``alpha`` and ``bc``, independent of the summation order, so equal tables
    give equal fingerprints in any compiled program (a float sum would not, under ``vmap``)."""
    import jax.numpy as jnp
    from jax import lax

    cols = (tr.t0, tr.dt, tr.alpha, tr.bc)
    bits = [lax.bitcast_convert_type(lax.stop_gradient(x), jnp.int64) for x in cols]
    return jnp.sum(sum(b * w for b, w in zip(bits, wts, strict=True)))


def year_function(
    rep: RunReplay, year: Span, tier: str, uptake_scale: float = 1.0, **overrides: Any
) -> tuple[Callable[..., Any], Any]:
    """``f(scales[5]) -> (YearOut, tables)``: one restart year of the adaptive run.

    ``tables`` (``[D, n_tab]`` each of ``t0``, ``dt``, ``alpha``, ``bc``) are the step tables of the run, for
    :func:`frozen_year_function`. The gradient of ``f`` (through the segment's ``custom_vjp``) is
    that of the frozen tables.
    """
    import jax.numpy as jnp
    from jax import lax

    from agrijax.processes.soil_water import richards_adaptive as A
    from agrijax.processes.soil_water.richards import SoilWater

    th0, sup, eva, upt, ref = _year_inputs(rep, year)
    upt = upt * uptake_scale
    cfg = adaptive_params(rep.soil, rep.grid, tier, **overrides).stepping
    wts = _table_weights(cfg.max_steps + 1)
    grid = rep.grid

    def f(scales: Any) -> tuple[YearOut, Any]:
        soil = scaled_soil(rep.soil, scales)
        params = adaptive_params(soil, grid, tier, **overrides)

        def body(w: Any, x: Any) -> Any:
            new, tot, st, tr = A.adaptive_segment(w, params, 0.0, float(HOURS), *x, w.dt_next)
            th = table_fingerprint(tr, wts)
            return new, (new.storage(grid), tot.drainage, st, th, (tr.t0, tr.dt, tr.alpha, tr.bc))

        w0 = SoilWater.from_theta(th0, soil)
        _, (storage, drainage, st, th, tables) = lax.scan(body, w0, (sup, eva, upt))
        out = YearOut(
            storage_end=storage[-1],
            drainage=jnp.sum(drainage),
            m1_loss=jnp.sqrt(jnp.mean((storage - ref) ** 2)),
            n_steps=st.n_steps,
            n_rejects=st.n_rejects,
            n_newton=st.n_newton,
            n_unconverged=st.n_unconverged,
            table_hash=th,
        )
        return out, tables

    return f, cfg


def frozen_year_function(
    rep: RunReplay, year: Span, tier: str, uptake_scale: float = 1.0, **overrides: Any
) -> Callable[..., Any]:
    """``g(scales[5], tables) -> (storage_end, drainage, m1_loss)``: the year replayed along fixed tables."""
    import jax.numpy as jnp
    from jax import lax

    from agrijax.processes.soil_water import richards_adaptive as A
    from agrijax.processes.soil_water.richards import SoilWater

    th0, sup, eva, upt, ref = _year_inputs(rep, year)
    upt = upt * uptake_scale
    grid = rep.grid

    def g(scales: Any, tables: Any) -> tuple[Any, Any, Any]:
        soil = scaled_soil(rep.soil, scales)
        params = adaptive_params(soil, grid, tier, **overrides)

        def body(w: Any, x: Any) -> Any:
            (s, e, u), (t0, dt, al, bc) = x
            n = t0.shape[0]
            z = jnp.zeros((n,), t0.dtype)
            tr = A.StepTrace(**{k: z for k in A.StepTrace._fields})._replace(t0=t0, dt=dt, alpha=al, bc=bc)
            new, tt = A.replay_segment(w, params, s, e, u, tr)
            return new, (new.storage(grid), jnp.sum(tt.drainage))

        w0 = SoilWater.from_theta(th0, soil)
        _, (storage, drainage) = lax.scan(body, w0, ((sup, eva, upt), tables))
        return storage[-1], jnp.sum(drainage), jnp.sqrt(jnp.mean((storage - ref) ** 2))

    return g


# ------------------------------------------------------------------------ gradients and the staircase scan

#: outputs of one year checked for gradients and steps
OUTS = ("storage_end", "drainage", "m1_loss")
#: relative steps of the central differences on the frozen tables (the two-step kink test)
FD_FROZEN = (1e-4, 1e-6)
#: relative steps of the central differences of the full adaptive run (1e-6, 1e-4; secants 1 %, 5 %)
FD_FREE = (1e-6, 1e-4, 1e-2, 5e-2)
#: the staircase scan: s in [1 - SCAN_HALF, 1 + SCAN_HALF] in N_SCAN points
N_SCAN = 401
SCAN_HALF = 0.1


def _outs(o: YearOut) -> Any:
    import jax.numpy as jnp

    return jnp.stack([getattr(o, k) for k in OUTS])


def _pm_points(steps: Sequence[float]) -> np.ndarray:
    """``1 +- h e_i`` for every step ``h`` and class ``i`` (order: step, class, sign)."""
    eye = np.eye(len(SCALED))
    return np.stack([1.0 + sg * h * eye[i] for h in steps for i in range(len(SCALED)) for sg in (1.0, -1.0)])


def grad_table(rep: RunReplay, year: Span, tier: str, uptake_scale: float = 1.0) -> pd.DataFrame:
    """Gradients at ``s = 1``: AD against central differences, per class of :data:`SCALED` and output.

    * ``fd_frozen_<h>``: the frozen step tables of the nominal run (the function AD is of) at the
      two relative steps of :data:`FD_FROZEN`; ``rel_frozen_two_step`` is their disagreement (a kink
      within the step splits them);
    * ``fd_free_<h>``: the full adaptive run (tables may move) at :data:`FD_FREE`, with
      ``same_tables_<h>`` (both ends keep every day's nominal step table) and ``days_changed_<h>``.
    """
    import jax
    import jax.numpy as jnp

    f, _ = year_function(rep, year, tier, uptake_scale)
    g = frozen_year_function(rep, year, tier, uptake_scale)
    s0 = jnp.ones(len(SCALED))
    walls: dict[str, float] = {}
    t0 = time.perf_counter()
    o0, tables = jax.block_until_ready(jax.jit(f)(s0))
    walls["t_fwd_s"] = time.perf_counter() - t0
    y0 = np.asarray(_outs(o0))
    t0 = time.perf_counter()
    jac = np.asarray(jax.block_until_ready(jax.jit(jax.jacrev(lambda s: _outs(f(s)[0])))(s0)))
    walls["t_ad_s"] = time.perf_counter() - t0
    frozen0 = np.asarray(jax.jit(lambda s: jnp.stack(g(s, tables)))(s0))
    n_c, n_o = len(SCALED), len(OUTS)

    t0 = time.perf_counter()
    fz = jax.jit(jax.vmap(lambda s: jnp.stack(g(s, tables))))(jnp.asarray(_pm_points(FD_FROZEN)))
    vals = np.asarray(jax.block_until_ready(fz)).reshape(len(FD_FROZEN), n_c, 2, n_o)
    walls["t_frozen_fd_s"] = time.perf_counter() - t0
    fd_frozen = (vals[:, :, 0] - vals[:, :, 1]) / (2.0 * np.asarray(FD_FROZEN)[:, None, None])

    t0 = time.perf_counter()
    of, _ = jax.block_until_ready(jax.jit(jax.vmap(f))(jnp.asarray(_pm_points(FD_FREE))))
    walls["t_free_fd_s"] = time.perf_counter() - t0
    shape = (len(FD_FREE), n_c, 2, -1)
    fv = np.asarray(jax.vmap(_outs)(of)).reshape(len(FD_FREE), n_c, 2, n_o)
    th = np.asarray(of.table_hash).reshape(shape)
    nu = np.asarray(of.n_unconverged).reshape(shape)
    nn = np.asarray(of.n_newton).reshape(shape)
    newton0 = np.asarray(o0.n_newton)
    fd_free = (fv[:, :, 0] - fv[:, :, 1]) / (2.0 * np.asarray(FD_FREE)[:, None, None])
    hash0 = np.asarray(o0.table_hash)

    def rel(a: float, b: float) -> float:
        return abs(a - b) / max(abs(b), 1e-300)

    label = year if isinstance(year, int) else "/".join(year)
    rows = []
    for i, cls in enumerate(SCALED):
        for j, out in enumerate(OUTS):
            ad = float(jac[j, i])
            row: dict[str, Any] = {
                "site": rep.site, "year": label, "tier": tier, "class": cls, "output": out,
                "nominal": float(y0[j]), "frozen_minus_search": float(frozen0[j] - y0[j]), "ad": ad,
            }  # fmt: skip
            for a, h in enumerate(FD_FROZEN):
                row[f"fd_frozen_{h:g}"] = float(fd_frozen[a, i, j])
                row[f"rel_frozen_{h:g}"] = rel(ad, float(fd_frozen[a, i, j]))
            row["rel_frozen_two_step"] = rel(float(fd_frozen[1, i, j]), float(fd_frozen[0, i, j]))
            for a, h in enumerate(FD_FREE):
                changed = (th[a, i, 0] != hash0) | (th[a, i, 1] != hash0)
                row[f"fd_free_{h:g}"] = float(fd_free[a, i, j])
                row[f"rel_free_{h:g}"] = rel(ad, float(fd_free[a, i, j]))
                row[f"same_tables_{h:g}"] = not bool(changed.any())
                row[f"days_changed_{h:g}"] = int(changed.sum())
                row[f"newton_changed_{h:g}"] = int(
                    ((nn[a, i, 0] != newton0) | (nn[a, i, 1] != newton0)).sum()
                )
                row[f"unconverged_{h:g}"] = float(nu[a, i].sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    for k, v in walls.items():
        df[k] = v
    df["nominal_unconverged"] = float(np.asarray(o0.n_unconverged).sum())
    return df


def scan_points(rep: RunReplay, year: int, tier: str, cls: str, idx: np.ndarray) -> dict[str, np.ndarray]:
    """The outputs, their AD derivatives and every day's step-table fingerprint at scan points ``idx``."""
    import jax
    import jax.numpy as jnp

    f, _ = year_function(rep, year, tier)
    i = SCALED.index(cls)
    s = np.linspace(1.0 - SCAN_HALF, 1.0 + SCAN_HALF, N_SCAN)[idx]

    def outs(z: Any) -> tuple[Any, YearOut]:
        o, _ = f(jnp.ones(len(SCALED)).at[i].set(z))
        return _outs(o), o

    def point(x: Any) -> Any:
        return jax.jacrev(outs, has_aux=True)(x)

    t0 = time.perf_counter()
    jac, o = jax.block_until_ready(jax.jit(jax.vmap(point))(jnp.asarray(s)))
    wall = time.perf_counter() - t0
    return {
        "s": s,
        "idx": np.asarray(idx),
        "vals": np.asarray(jax.vmap(_outs)(o)),
        "ad": np.asarray(jac),
        "table_hash": np.asarray(o.table_hash),
        "n_steps": np.asarray(o.n_steps),
        "n_rejects": np.asarray(o.n_rejects),
        "n_newton": np.asarray(o.n_newton),
        "n_unconverged": np.asarray(o.n_unconverged),
        "wall": np.asarray(wall),
    }


def staircase_tables(scan: dict[str, np.ndarray], **tags: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Neighbour pairs of a scan: ``J = |y(s + ds) - y(s) - ds (y'(s) + y'(s + ds)) / 2|`` and whether
    any day's step table changed between the two; and the summary per output and signature change."""
    s, vals, ad, th = scan["s"], scan["vals"], scan["ad"], scan["table_hash"]
    ds = np.diff(s)
    changed = np.any(th[1:] != th[:-1], axis=1)
    days = (th[1:] != th[:-1]).sum(axis=1)
    rows = []
    for j, name in enumerate(OUTS):
        jump = np.abs(np.diff(vals[:, j]) - ds * 0.5 * (ad[:-1, j] + ad[1:, j]))
        for k in range(len(ds)):
            rows.append({
                **tags, "output": name, "s": s[k], "y": vals[k, j], "ad": ad[k, j], "jump": jump[k],
                "signature_changed": bool(changed[k]), "days_changed": int(days[k]),
                "newton_diff": float(scan["n_newton"][k + 1].sum() - scan["n_newton"][k].sum()),
            })  # fmt: skip
    df = pd.DataFrame(rows)
    summ = []
    for (name, ch), gdf in df.groupby(["output", "signature_changed"]):
        jmp = gdf.jump.to_numpy()
        summ.append({
            **tags, "output": name, "signature_changed": ch, "n_pairs": len(gdf),
            "jump_median": float(np.median(jmp)), "jump_p95": float(np.quantile(jmp, 0.95)),
            "jump_max": float(jmp.max()), "s_at_max": float(gdf.s.to_numpy()[int(np.argmax(jmp))]),
            "n_above_1e-3": int((jmp > 1e-3).sum()), "ad_min": float(gdf.ad.min()), "ad_max": float(gdf.ad.max()),
        })  # fmt: skip
    sm = pd.DataFrame(summ)
    sm["unconverged_points"] = int((scan["n_unconverged"].sum(axis=1) > 0).sum())
    return df, sm


# ------------------------------------------------------------------------------ script


def _data_dir() -> Path:
    return Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))


def main_years(sites: Sequence[str]) -> None:
    """The tables; ``W1_OVERRIDES`` (JSON) sets :func:`adaptive_params` overrides and ``W1_TAG``
    a suffix of the file names (variants)."""
    import json

    data = _data_dir()
    out = data / OUT_SUB
    out.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(os.environ.get("W1_OVERRIDES", "{}"))
    tag = os.environ.get("W1_TAG", "")
    t0 = time.perf_counter()
    res = all_site_years(data, sites, **overrides)
    tabs, walls, days = [], [], []
    for site, r in res.items():
        for tier in TIERS:
            tabs.append(r["tables"][tier])
            walls.append(
                {"site": site, "tier": tier, "compile_s": r["walls"][tier][0], "run_s": r["walls"][tier][1]}
            )
            if site == "CA-TPA":
                days.append({"tier": tier, **day_row(r["rep"], r["runs"][tier], r["ref"], "2016-09-10")})
    y = pd.concat(tabs, ignore_index=True)
    y.to_csv(out / f"years{tag}.csv", index=False, float_format="%.6g")
    pd.DataFrame(walls).to_csv(out / f"walls{tag}.csv", index=False, float_format="%.4g")
    pd.DataFrame(days).to_csv(out / f"catpa_2016-09-10{tag}.csv", index=False, float_format="%.6g")
    print(f"years done in {time.perf_counter() - t0:.0f} s", flush=True)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 400):
        cols = [
            "site",
            "tier",
            "year",
            "storage_rmse_cm",
            "err_ref_max_cm",
            "ref_clean",
            "n_unconverged",
            "budget_exhausted",
            "step_balance_max_conv_cm",
            "step_balance_max_cm",
            "newton_per_day",
        ]
        print(y[[c for c in cols if c in y]].to_string())
        print(pd.DataFrame(days).T.to_string())
        print(pd.DataFrame(walls).to_string())


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "years"
    if cmd == "years":
        main_years(sys.argv[2:] or SITES)
    else:
        raise SystemExit(f"unknown command {cmd}")
