"""Calibration harness on CERES-Maize 4.8.6 with replayed water (nitrogen off).

Steps (``python scripts/bench/calib/ceres_twin.py <step> [--out DIR]``):

``prep``
    Run dscsm048 (build486, nitrogen off) for the treatments of :data:`PROBLEMS`, build the
    crop's parameters and forcing exactly as the CERES-Maize integration test does
    (``tests/integration/test_ceres_dssat.py``: weather and SW from the run's outputs, EOP / TRWUP
    from the instrumented-engine SPAM dump tables), check that the model at the INP cultivar
    reproduces DSSAT's yield and dates, and pickle the treatments.
``trust``
    Gradient-trust report (agrijax.calib.trust) of the season functionals (yield, final
    biomass, integrated LAI, silking and maturity day of every treatment) at the true cultivar,
    in the gradient modes exact / ste / implicit, and of the twin loss at a perturbed point.
``twin``
    Twin experiment: synthetic observations at the true cultivar, R perturbed starts, recovery
    with batched Adam (ste, exact, and ste with secants for the zero-gradient parameters), CMA-ES
    and random search at equal model calls and at equal wall time.

Everything is written under ``--out`` (default ``$AGRI_JAX_DATA/validation/aj_dcal``). The steps are
sized for a multi-core CPU node of a cluster (float64, many model evaluations), not for a laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from agrijax.calib import (  # noqa: E402
    AdamConfig,
    CmaConfig,
    Group,
    Objective,
    Target,
    TrustConfig,
    adam,
    batched_loss,
    batched_value_and_grad,
    cma_es,
    random_search,
    secant_gradient,
    time_calls,
    trust_report,
)
from agrijax.calib.ceres import (  # noqa: E402
    ISTAGE_MATURITY_OUT,
    ISTAGE_SILKING_OUT,
    CeresTreatment,
    ceres_space,
    group_simulator,
    stack_treatments,
)
from agrijax.calib.objective import evaluate_targets, first_day_index  # noqa: E402
from agrijax.core.grad import gradient_mode  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
#: calibration problems: one cultivar each, its treatments share the soil layering (one vmap)
PROBLEMS: dict[str, list[tuple[str, int]]] = {
    "UFGA8201": [("UFGA8201", 1), ("UFGA8201", 3), ("UFGA8201", 5)],  # rainfed, irrigated, veg. stress
    "IUAF9901": [("IUAF9901", 1), ("IUAF9901", 3)],  # Ames 1999, 4.7 and 7.5 plants m-2
}
#: P2 is left out: the first trust run (2026-09-27) measured it inert on every
#: UFGA8201 functional (the daylength stays below P2O) and a 2-step staircase at IUAF9901 (P2 = 0.1)
PARAMS = ("P1", "P5", "G2", "G3", "PHINT", "RUE")
#: days appended after the reference run's last day (a perturbed cultivar may mature later)
PAD_DAYS = 40
MODES = ("exact", "ste", "implicit")

# twin-experiment settings (harness choices, recorded in the result)
N_REP = 8  # perturbed starts per problem
PERTURB = 0.25  # starts: theta* * (1 + U(-PERTURB, PERTURB)), clipped into the inner box
MARGIN = 0.05  # the inner box keeps starts this fraction of the bound width from either bound
ADAM_STEPS = 500
ADAM_LR = 0.05
SECANT_DZ = (0.1, 0.3)  # secant half-widths in z units (logit): 0.1 is ~2.5 % of the bound width at mid-box
LAI_EVERY = 7  # LAI observed every 7th day between emergence and maturity
REL_SCALE = 0.05  # yield and final biomass: 5 % observation scale
LAI_SCALE = 0.2  # m2 m-2
DATE_SCALE = 2.0  # d


def out_dir(args: argparse.Namespace) -> Path:
    base = args.out or os.path.join(
        os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data"), "validation", "aj_dcal"
    )
    p = Path(base).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_info() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job": os.environ.get("SLURM_JOB_ID"),
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
    }


def to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, tree)


def to_jax(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, tree)


# ------------------------------------------------------------------------------------ prep
def step_prep(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(ROOT / "tests" / "integration"))
    import test_ceres_dssat as tcd  # type: ignore[import-not-found]  # the test's staging and forcing, reused as is

    data = Path(os.environ["AGRI_JAX_DATA"])
    tables = data / "dumps" / "tables" / "dssat486"
    run_root = Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "dcal"
    out = out_dir(args)
    treatments: dict[str, list[CeresTreatment]] = {}
    checks = []
    for prob, trts in PROBLEMS.items():
        treatments[prob] = []
        for exp, trno in trts:
            d = tcd.run_reference(exp, trno, run_root / f"{exp}_{trno}")
            p, f, res, row = tcd.simulate(d, trno, exp=exp, tables=tables)
            days = np.asarray(f.yrdoy)
            stage = res["istage"][:, 0]

            def first(code: int, stage=stage, days=days) -> int:
                hit = np.nonzero(stage == code)[0]
                return int(days[hit[0]]) if hit.size else -99

            yld = float(res["gwad"][-1, 0])
            c = {
                "treatment": f"{exp}_t{trno}",
                "n_days": len(days),
                "n_layer": int(p.soil.dlayr.shape[0]),
                "yield_model": yld,
                "yield_dssat": float(row["HWAM"]),
                "yield_rel_err": abs(yld - float(row["HWAM"])) / float(row["HWAM"]),
                "adat_model": first(ISTAGE_SILKING_OUT),
                "adat_dssat": int(row["ADAT"]),
                "mdat_model": first(ISTAGE_MATURITY_OUT),
                "mdat_dssat": int(row["MDAT"]),
                "cultivar": {k: float(getattr(p.cultivar, k.lower())) for k in PARAMS},
            }
            checks.append(c)
            print(json.dumps(c))
            meta = {"exp": exp, "trno": trno, "summary": {k: float(row[k]) for k in ("HWAM", "ADAT", "MDAT")}}
            treatments[prob].append(CeresTreatment(f"{exp}_t{trno}", to_numpy(p), to_numpy(f), meta))
    with open(out / "prep.pkl", "wb") as fh:
        pickle.dump(treatments, fh)
    (out / "prep_checks.json").write_text(json.dumps({"env": env_info(), "checks": checks}, indent=1))


def load_problems(out: Path) -> dict[str, list[CeresTreatment]]:
    with open(out / "prep.pkl", "rb") as fh:
        tr = pickle.load(fh)
    return {
        k: [CeresTreatment(t.name, to_jax(t.params), to_jax(t.forcing), t.meta) for t in v]
        for k, v in tr.items()
    }


def build(prob: list[CeresTreatment], names: tuple[str, ...] = PARAMS, *, mode: str):
    """Space, simulator and true theta of one problem (the INP cultivar of its treatments); the
    parameters not in ``names`` stay at their true (INP) values. The simulator is bound to the
    gradient ``mode``: build one per mode."""
    space = ceres_space(names)
    n_days = max(int(np.shape(t.forcing.yrdoy)[0]) for t in prob) + PAD_DAYS
    params, forcing, state0 = stack_treatments(prob, n_days)
    sim = group_simulator(space, params, forcing, state0, mode=mode)
    truth = np.asarray(space.get(prob[0].params), dtype=float)
    for t in prob[1:]:
        assert np.array_equal(np.asarray(space.get(t.params)), truth), "one cultivar per problem"
    return space, sim, truth


def functionals(sim):
    """theta -> [5 * B]: yield, final biomass, integrated LAI, silking day, maturity day."""

    def f(theta):
        o = sim(theta)
        return jnp.concatenate(
            [
                o["gwad"][-1],
                o["cwad"][-1],
                jnp.sum(o["lai"], axis=0),
                first_day_index(o["istage"], ISTAGE_SILKING_OUT),
                first_day_index(o["istage"], ISTAGE_MATURITY_OUT),
            ]
        )

    return f


FUNC_NAMES = ("yield", "cwad_final", "lai_sum", "silk_day", "mat_day")


def targets_for(sim, truth) -> tuple[tuple[Target, ...], dict[str, np.ndarray]]:
    out = sim(jnp.asarray(truth))
    st = np.asarray(out["istage"])
    t = st.shape[0]
    emerged = np.isin(st, [1, 2, 3, 4, 5, 6])
    mask = emerged & (np.arange(t)[:, None] % LAI_EVERY == 0)
    tg = (
        Target("yield", "gwad", "final", rel_scale=REL_SCALE),
        Target("cwad", "cwad", "final", rel_scale=REL_SCALE),
        Target("lai", "lai", "series", scale=LAI_SCALE, mask=jnp.asarray(mask)),
        Target("silk", "istage", "date", scale=DATE_SCALE, code=ISTAGE_SILKING_OUT),
        Target("mat", "istage", "date", scale=DATE_SCALE, code=ISTAGE_MATURITY_OUT),
    )
    obs = {k: np.asarray(v) for k, v in evaluate_targets(tg, out).items()}
    return tg, obs


def starts(space, truth, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    th = truth[None] * (1.0 + rng.uniform(-PERTURB, PERTURB, (N_REP, truth.size)))
    lo = space.lower + MARGIN * space.width
    hi = space.upper - MARGIN * space.width
    return np.clip(th, lo, hi)


# ------------------------------------------------------------------------------------ trust
def step_trust(args: argparse.Namespace) -> None:
    out = out_dir(args)
    problems = load_problems(out)
    report: dict[str, Any] = {"env": env_info(), "problems": {}}
    cfg = TrustConfig()
    for pname, prob in problems.items():
        names = [f"{t.name}/{fn}" for fn in FUNC_NAMES for t in prob]
        rep: dict[str, Any] = {"params": list(PARAMS), "modes": {}}
        for mode in MODES:
            space, sim, truth = build(prob, mode=mode)  # one simulator per mode
            rep["truth"] = truth.tolist()
            with gradient_mode(mode):
                t0 = time.perf_counter()
                r = trust_report(functionals(sim), truth, space.lower, space.upper, PARAMS, names, cfg)
                r["wall_s"] = time.perf_counter() - t0
            rep["modes"][mode] = r
            print(pname, mode, f"{r['wall_s']:.1f} s")
            for pn in PARAMS:
                cls = {k: v["class"] for k, v in r["params"][pn]["outputs"].items()}
                lev = {k: v["level"] for k, v in r["params"][pn]["outputs"].items()}
                print(f"  {pn}: classes {sorted(set(cls.values()))} levels {sorted(set(lev.values()))}")
        # the twin loss at the first perturbed start, ste mode
        space, sim, truth = build(prob, mode="ste")
        tg, obs = targets_for(sim, truth)
        obj = Objective((Group(pname, sim, tg, obs),))
        x0 = starts(space, truth, seed=1)[0]
        with gradient_mode("ste"):
            r = trust_report(
                lambda th, obj=obj: obj(th)[None], x0, space.lower, space.upper, PARAMS, ["loss"], cfg
            )
        rep["loss_at_start"] = r
        report["problems"][pname] = rep
    (out / "trust.json").write_text(json.dumps(report))
    print("wrote", out / "trust.json")


# ------------------------------------------------------------------------------------ twin
def summarise(space, res, truth, t_per_call: dict[str, float]) -> dict[str, Any]:
    th = np.asarray(space.to_physical(jnp.asarray(res.z_best)))
    rel = space.relative_error(th, truth)
    wid = space.width_error(th, truth)
    return {
        "theta_best": th.tolist(),
        "loss_best": res.loss_best.tolist(),
        "rel_err": rel.tolist(),
        "rel_err_median": np.median(rel, axis=0).tolist(),
        "rel_err_max": rel.max(axis=0).tolist(),
        "width_err_median": np.median(wid, axis=0).tolist(),
        "n_recovered_1pct": int(np.sum(np.all(rel <= 0.01, axis=1))),
        "loss_history_median": np.median(res.loss_history, axis=1).tolist(),
        "n_forward": res.n_forward,
        "n_grad": res.n_grad,
        "wall_s": res.wall_s,
        "per_call_s": t_per_call,
        "extra": {
            k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in res.extra.items()
        },
    }


#: parameter subsets of the twin experiment: all six, and the three the trust report finds
#: smooth on yield / biomass (the phenology coefficients fixed at the truth)
SUBSETS: dict[str, tuple[str, ...]] = {"full": PARAMS, "growth": ("G2", "G3", "RUE")}
PLATEAU_FRAC = 0.05  # zero-loss plateau scan: truth +- 5 % of the true value
PLATEAU_POINTS = 401
PLATEAU_TOL = 1e-12  # a loss at most this is "zero" (normalised residuals ~1e-6)


def plateau(obj, truth: np.ndarray) -> dict[str, Any]:
    """Per parameter: the interval around the truth on which the twin loss stays 0 (the
    resolution limit of the observations for that parameter, others at the truth)."""
    f = jax.jit(jax.vmap(obj))
    out = {}
    for i, pn in enumerate(PARAMS):
        grid = truth[i] * (1.0 + np.linspace(-PLATEAU_FRAC, PLATEAU_FRAC, PLATEAU_POINTS))
        pts = np.repeat(truth[None], PLATEAU_POINTS, axis=0)
        pts[:, i] = grid
        loss = np.asarray(f(jnp.asarray(pts)))
        c = PLATEAU_POINTS // 2
        lo = c
        while lo > 0 and loss[lo - 1] <= PLATEAU_TOL:
            lo -= 1
        hi = c
        while hi < PLATEAU_POINTS - 1 and loss[hi + 1] <= PLATEAU_TOL:
            hi += 1
        out[pn] = {
            "zero_interval": [float(grid[lo]), float(grid[hi])],
            "rel_width": float((grid[hi] - grid[lo]) / truth[i]),
            "loss_at_truth": float(loss[c]),
            "hits_scan_edge": bool(lo == 0 or hi == PLATEAU_POINTS - 1),
        }
    return out


def run_subset(names, pname, prob, pi, trust) -> dict[str, Any]:
    space, sim, truth = build(prob, names, mode="ste")
    tg, obs = targets_for(sim, truth)
    obj = Objective((Group(pname, sim, tg, obs),))

    def loss_z(z):
        return obj(space.to_physical(z))

    def loss_in(mode: str):
        """The same twin loss on a simulator bound to ``mode`` (forward bit-identical)."""
        if mode == "ste":
            return loss_z
        _, sim_m, _ = build(prob, names, mode=mode)
        obj_m = Objective((Group(pname, sim_m, tg, obs),))
        return lambda z: obj_m(space.to_physical(z))

    z0 = np.asarray(space.to_unconstrained(jnp.asarray(starts(space, truth, seed=1 + pi))))
    pr: dict[str, Any] = {
        "truth": truth.tolist(),
        "treatments": [t.name for t in prob],
        "space": space.table(),
        "start_theta": np.asarray(space.to_physical(jnp.asarray(z0))).tolist(),
        "methods": {},
    }
    if names == PARAMS:
        pr["plateau"] = plateau(obj, truth)
    # the parameters whose loss derivative the trust report does not trust (the twin loss at the
    # first perturbed start, ste mode): class "step" (AD 0 while the loss moves) or "jumpy" (jumps
    # the AD derivative does not explain) get central secants in the hybrid runs
    flagged = []
    if trust is not None:
        rl = trust["problems"][pname]["loss_at_start"]["params"]
        flagged = [k for k, pn in enumerate(names) if rl[pn]["outputs"]["loss"]["class"] in ("step", "jumpy")]
    pr["secant_params"] = [names[k] for k in flagged]

    fwd = batched_loss(loss_z)
    t_fwd = time_calls(fwd, jnp.asarray(z0))
    pr["forward_batch_s"] = {"first": t_fwd[0], "steady": t_fwd[1], "batch": N_REP}
    print(pname, names, "forward R-batch", t_fwd, "secant params", pr["secant_params"])

    runs: list[tuple[str, str, float]] = [("adam_ste", "ste", 0.0), ("adam_exact", "exact", 0.0)]
    if flagged:
        runs += [(f"adam_ste_secant{dz:g}", "ste", dz) for dz in SECANT_DZ]
    for label, mode, dz in runs:
        loss_m = loss_in(mode)
        with gradient_mode(mode):
            if dz > 0.0:
                vg = jax.jit(jax.vmap(secant_gradient(loss_m, flagged, [dz] * len(flagged))))
            else:
                vg = batched_value_and_grad(loss_m)
            t_vg = time_calls(vg, jnp.asarray(z0))
            res = adam(
                vg,
                z0,
                AdamConfig(lr=ADAM_LR, steps=ADAM_STEPS),
                forward_per_step=2 * len(flagged) if dz > 0.0 else 0,
            )
        s = summarise(space, res, truth, {"value_and_grad_first": t_vg[0], "value_and_grad_steady": t_vg[1]})
        pr["methods"][label] = s
        print(pname, label, f"wall {res.wall_s:.1f}s", "median rel err", np.round(s["rel_err_median"], 4))

    # gradient-free baselines: equal model calls (one value-and-gradient call = one call), equal
    # wall time (the plain Adam's) and, with secants, the hybrid's call count
    lam = 4 + int(np.floor(3 * np.log(len(names))))
    cma_f = batched_loss(loss_z)
    t_cma = time_calls(cma_f, jnp.asarray(np.repeat(z0, lam, axis=0)))
    per_eval = t_cma[1] / lam  # wall per evaluation per problem (R problems batched)
    wall_budget = pr["methods"]["adam_ste"]["wall_s"]
    budgets = {"cma_equal_calls": ADAM_STEPS, "cma_equal_wall": max(lam, int(wall_budget / per_eval))}
    if flagged:
        budgets["cma_equal_hybrid_calls"] = ADAM_STEPS * (1 + 2 * len(flagged))
    pr["cma_population_batch_s"] = {"first": t_cma[0], "steady": t_cma[1], "batch": N_REP * lam}
    for label, budget in budgets.items():
        res = cma_es(cma_f, z0, CmaConfig(sigma0=0.5, max_evals=budget), seed=11 + pi)
        s = summarise(space, res, truth, {"population_steady": t_cma[1]})
        s["budget"] = budget
        pr["methods"][label] = s
        print(
            pname,
            label,
            budget,
            f"wall {res.wall_s:.1f}s",
            "median rel err",
            np.round(s["rel_err_median"], 4),
        )
    # random search in the box of the starts (theta* +- PERTURB), equal wall time
    lo_th = space.clip(truth * (1 - PERTURB))
    hi_th = space.clip(truth * (1 + PERTURB))
    lo_z = np.asarray(space.to_unconstrained(jnp.asarray(np.minimum(lo_th, hi_th))))
    hi_z = np.asarray(space.to_unconstrained(jnp.asarray(np.maximum(lo_th, hi_th))))
    rs_batch = 16
    rs_f = batched_loss(loss_z)
    jax.block_until_ready(rs_f(jnp.zeros((N_REP * rs_batch, len(names)))))
    res = random_search(rs_f, lo_z, hi_z, N_REP, budgets["cma_equal_wall"], batch=rs_batch, seed=21 + pi)
    s = summarise(space, res, truth, {})
    s["budget"] = budgets["cma_equal_wall"]
    pr["methods"]["random_equal_wall"] = s
    print(pname, "random", f"wall {res.wall_s:.1f}s", "median rel err", np.round(s["rel_err_median"], 4))
    return pr


def step_twin(args: argparse.Namespace) -> None:
    out = out_dir(args)
    problems = load_problems(out)
    trust = json.loads((out / "trust.json").read_text()) if (out / "trust.json").exists() else None
    result: dict[str, Any] = {
        "env": env_info(),
        "settings": {
            "subsets": SUBSETS,
            "n_rep": N_REP,
            "perturb": PERTURB,
            "margin": MARGIN,
            "adam_steps": ADAM_STEPS,
            "adam_lr": ADAM_LR,
            "secant_dz": SECANT_DZ,
            "lai_every": LAI_EVERY,
            "rel_scale": REL_SCALE,
            "lai_scale": LAI_SCALE,
            "date_scale": DATE_SCALE,
            "pad_days": PAD_DAYS,
            "plateau": [PLATEAU_FRAC, PLATEAU_POINTS, PLATEAU_TOL],
        },
        "subsets": {},
    }
    for sname, names in SUBSETS.items():
        result["subsets"][sname] = {}
        for pi, (pname, prob) in enumerate(problems.items()):
            result["subsets"][sname][pname] = run_subset(names, pname, prob, pi, trust)
            (out / "twin.json").write_text(json.dumps(result))
    print("wrote", out / "twin.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "trust", "twin", "all"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    steps = ("prep", "trust", "twin") if args.step == "all" else (args.step,)
    for s in steps:
        t0 = time.perf_counter()
        {"prep": step_prep, "trust": step_trust, "twin": step_twin}[s](args)
        print(f"step {s}: {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
