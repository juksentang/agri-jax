"""D3-1: staged calibration of CERES-Maize cultivar coefficients on the validated free-run DSSAT day.

Measurement / driver only; nothing in ``src/`` changes.

**Model.** :mod:`agrijax.models.day_dssat486` exactly as the D2-1a acceptance runs it (configuration
``free`` of ``tests/integration/day_dssat486_free_harness.py``: every entry its own process, REAL*4 soil
values, the labelled SOILDYN / residue / hourly TAVG replays, ``exact_lags=True``), f64, gradient mode
``ste`` bound to every program (:func:`agrijax.core.grad.bind_gradient_mode`). The forcing of a run is
padded :data:`PAD_DAYS` days past the longest season of its (layer count, MESEV) group (the last day
repeated, rain and irrigation 0: the harness convention), so a candidate cultivar that matures later
than the reference season still reaches maturity; every treatment here is harvested at maturity
(``HARVS = M``), and the simulated harvest is the simulated maturity day.

**Calibrated coefficients.** The six ``MZCER048.CUL`` coefficients P1, P2, P5, G2, G3, PHINT inside the
file's ``MINIMA`` / ``MAXIMA`` rows (:data:`agrijax.calib.ceres.CERES_SPECS`). RUE is left out: it is an
ecotype coefficient (``MZCER048.ECO``), that file has no bound rows, and the ``.CUL`` writer (the
round trip to DSSAT) writes cultivar rows only. A coefficient whose outputs do not move over its whole
box on a problem (measured, step ``check``: P2 without a photoperiod signal) is not calibrated there
(kept at its start value) and reported as not identifiable.

**Observations -> targets.** One entry per observed quantity of a treatment (types: stage date = first
day of an ISTAGE code; value at simulated maturity; value on a day), normalised as
:mod:`agrijax.calib.objective` / :func:`agrijax.calib.observations.observation_targets` do: dates in
days (scale 1 d), other quantities by the mean absolute observed value of the problem (a squared
normalised RMSE); the loss of a problem is the sum over targets of the mean squared normalised residual.
The **original objective** is that loss over every target, unsmoothed (dates are integer day indices).
The entries are gathered on the device, so a model call returns ``[samples, entries]`` values only.

**Methods** (all problems and starts batched in one program call per (group, shape)):

* ``staged``: stage 1 CMA-ES (derivative-free, :data:`agrijax.calib.ceres.CERES_DERIVATIVE_FREE`) on
  P1, P2, P5, PHINT against the stage dates (variant ``laipre``: also the LAI observed before silking,
  which the grain coefficients cannot move), G2 / G3 at the start values; then a gradient-trust report
  (:mod:`agrijax.calib.trust` classes and levels, per (treatment, parameter), at the stage-2 start of
  every row) and :func:`agrijax.calib.trust.gradient_plan`; stage 2 Levenberg-Marquardt on G2, G3
  against yield, biomass, LAI (and grain number in variant ``gn``), with the residual Jacobian from
  forward-mode AD for the pairs the plan trusts and central secants for the others.
* ``cma``: CMA-ES on all six coefficients against the original objective (the derivative-free baseline).
* ``random``: uniform random search (baseline).

Every evaluation also yields the original objective, so every method has a best-so-far trajectory of
the same quantity; quality targets are defined before the runs (:data:`Q_TWIN`, :data:`Q_REAL_FACTOR`).

Benchmark switches (wt/grad_gap): ``AJ_D31_DIR`` writes everything to a separate directory, and
``AJ_D31_TRUST`` (``legacy`` / ``split``, :data:`TRUST_LEVEL2`) selects the level-2 test of the stage-2
trust report.

Steps: ``prep`` (dscsm048 reference runs + inputs), ``check`` (published cultivars against DSSAT,
identifiability scans, trust at the truth), ``twin`` / ``real`` (one fresh process per method: compile
counted, no compile cache), ``roundtrip`` (calibrated cultivar -> .CUL copy -> dscsm048), ``table``.
Everything goes under ``$AGRI_JAX_DATA/validation/aj_d31``. Runs on rorqual only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

T_START = time.perf_counter()

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "src", REPO / "tests" / "integration"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ============================================================================ settings (harness choices)
PARAMS = ("P1", "P2", "P5", "PHINT", "G2", "G3")
PHEN = ("P1", "P2", "P5", "PHINT")  # stage 1 (derivative-free by default)
GROWTH = ("G2", "G3")  # stage 2 (gradient plan)
I_PHEN = tuple(PARAMS.index(n) for n in PHEN)
I_GROWTH = tuple(PARAMS.index(n) for n in GROWTH)
FIELDS = {"P1": "p1", "P2": "p2", "P5": "p5", "G2": "g2", "G3": "g3", "PHINT": "phint"}
MODE = "ste"
PAD_DAYS = 60
N_STARTS = 8
PERTURB = 0.25  # twin starts: truth * (1 + U(-PERTURB, PERTURB)), clipped into the inner box
MARGIN = 0.05  # inner box: this fraction of the bound width from either bound
LAI_EVERY = 14  # twin: LAI observed every 14th day from emergence to maturity
E_MAX = 64  # entries per treatment (padded)
#: quality target of the twin (defined before the runs): the original objective <= Q_TWIN, i.e. every
#: stage date exact (one day off in one of <= 3 treatments costs >= 1/3) and every continuous target
#: within about 1 % normalised RMSE
Q_TWIN = 1e-4
#: recovery: an identifiable coefficient within 2 % of its true value
REC_TOL = 0.02
#: quality target of the real observations: within 5 % of the lowest loss any method found (per problem)
Q_REAL_FACTOR = 1.05
SIGMA0 = 0.5  # CMA-ES initial step [z units]
CMA_STALL = 25  # generations without an improvement of CMA_GAIN_REL before a CMA-ES restart
CMA_GAIN_REL = 1e-2
CMA2_EVALS = 400  # stage 2 CMA-ES budget per row (problems whose plan has no AD pair)
CMA1_EVALS = 2400  # stage 1 budget per row
CMA_ALL_EVALS = 4000  # all-coefficient CMA-ES budget per row
RS_EVALS = 8000  # random search budget per row
RS_BATCH = 64
TOL1_DATES = 1e-12  # stage 1 (dates): every date exact
TOL1_LAIPRE = 1e-6  # stage 1 (dates + pre-silking LAI)
LM_ITERS = 40
LM_LAMBDA0 = 1e-3
LM_TOL = Q_TWIN * 1e-3
LM3_ITERS = 25  # stage-3 (joint refinement) iterations
SECANT_DZ = 0.1  # central-secant half width [z units] (D3-0: about 2.5 % of the width at mid-box)
TRUST_SCAN = 201  # line-scan points (default 41 misses jumps narrower than 2.3 % of the width)
#: level-2 test of the stage-2 trust report (environment AJ_D31_TRUST): "legacy" compares the
#: straight-through AD derivative with the small-step difference of the model (the check before
#: wt/grad_gap); "split" compares the exact-mode derivative with it and the unrounded model's
#: derivative with the unrounded model's small-step difference (agrijax.calib.trust, level 2)
TRUST_LEVEL2 = os.environ.get("AJ_D31_TRUST", "legacy")
#: calls of the published cultivar's DSSAT cost model (D4-1, rorqual, dscsm048 build486, daily outputs):
#: batch 18.2 ms / season, one treatment per invocation 65.1 ms -> start-up 46.9 ms per invocation;
#: node 192 cores: start-up 55 ms + ceil(seasons / 192) x 18.2 ms per round
DSSAT_SEASON_S = 0.0182
DSSAT_INVOCATION_S = 0.0651 - 0.0182
DSSAT_NODE_START_S = 0.055
DSSAT_NODE_CORES = 192

#: twin problems: one per cultivar of the 58 M2 treatments; nitrogen-level treatments are the same run
#: with nitrogen off, so one of them is used (GHWA0401, IBWA8301). GAGR0201 (IB0071) is left out: its
#: growth-chamber seasons end 74 days after sowing, before silking (check step: HWAM 0, simulated
#: maturity on padded days), so its dates and yield would come from the padded forcing
TWIN: dict[str, list[str]] = {
    "IB0171": ["BRPI0202_t01", "BRPI0202_t05"],
    "IB0173": ["BRPI0202_t02", "BRPI0202_t06"],
    "IB0172": ["BRPI0202_t03", "BRPI0202_t07"],
    "IB0174": ["BRPI0202_t04", "BRPI0202_t08"],
    "IB0012": ["FLSC8101_t01", "FLSC8101_t02"],
    "GH0010": ["GHWA0401_t01"],
    "IB0063": ["IBWA8301_t03"],
    "IB0060": ["IBWA8301_t06"],
    "IB1052": ["IUAF9901_t01", "IUAF9901_t03"],
    "ZA0002": ["SIAZ9501_t01", "SIAZ9501_t03", "SIAZ9601_t01"],
    "IB0035": ["UFGA8201_t02", "UFGA8201_t04", "UFGA8201_t06"],
}
#: real-observation problems: nitrogen-sufficient treatments with observed dates (the supported scope);
#: IUAF9901 (no ADAT / MDAT), GAGR0201 (canopy height only), GHWA0401 (low input) are left out
REAL: dict[str, list[str]] = {
    "UFGA8201/IB0035": ["UFGA8201_t02", "UFGA8201_t04", "UFGA8201_t06"],
    "FLSC8101/IB0012": ["FLSC8101_t01"],
    "IBWA8301/IB0063": ["IBWA8301_t03"],
    "IBWA8301/IB0060": ["IBWA8301_t06"],
    "SIAZ95-96/ZA0002": ["SIAZ9501_t01", "SIAZ9601_t01"],
    "BRPI0202/IB0171": ["BRPI0202_t05"],
    "BRPI0202/IB0173": ["BRPI0202_t06"],
    "BRPI0202/IB0172": ["BRPI0202_t07"],
    "BRPI0202/IB0174": ["BRPI0202_t08"],
}
#: cost of one JVP call relative to a forward call in the budget view (two tangent directions)
JVP_COST = 2.0
#: the CUL MINIMA / MAXIMA of the grain coefficients (the contour box)
CERES_BOX = {"G2": (248.0, 990.0), "G3": (5.0, 16.5)}
#: identifiability case: UFGA8201 calibrated on t02 / t04, t06 held out
UFGA_CAL: dict[str, list[str]] = {"UFGA8201/IB0035": ["UFGA8201_t02", "UFGA8201_t04"]}
UFGA_ALL: dict[str, list[str]] = {"UFGA8201/IB0035": ["UFGA8201_t02", "UFGA8201_t04", "UFGA8201_t06"]}
UFGA_HOLDOUT = "UFGA8201_t06"
#: observed codes used; variant "gn" adds the grain number
CODES_BASE = ("ADAT", "MDAT", "HWAM", "CWAM", "LAID")
CODES_GN = (*CODES_BASE, "H#AM")
STAGE1_CODES = ("ADAT", "MDAT")
STAGE2_CODES = ("HWAM", "CWAM", "LAID", "H#AM")

# model outputs gathered by the entries (index = ``out`` of a value entry)
OUT_NAMES = ("lai", "cwad", "gwad", "g_ad", "lsd")
CODE_OUT = {"HWAM": "gwad", "CWAM": "cwad", "H#AM": "g_ad", "LAID": "lai", "CWAD": "cwad", "GWAD": "gwad"}
ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY = 1, 4, 10
CODE_STAGE = {"ADAT": ISTAGE_SILKING, "MDAT": ISTAGE_MATURITY}
# entry types
E_PAD, E_DATE, E_FINAL, E_DAY = 0, 1, 2, 3


def out_root() -> Path:
    p = Path(os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")).expanduser() / "validation" / "aj_d31"
    if os.environ.get("AJ_D31_DIR"):  # a separate result directory (the before / after benchmark)
        p = Path(os.environ["AJ_D31_DIR"]).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_info() -> dict[str, Any]:
    import jax

    return {
        "host": platform.node(),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "job": os.environ.get("SLURM_JOB_ID"),
        "jax": jax.__version__,
        "n_devices": len(jax.devices()),
        "x64": bool(jax.config.jax_enable_x64),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "gradient_mode": MODE,
    }


def dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, default=_json_default))


def _json_default(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


# ============================================================================ prep
def maize_dir() -> Path:
    from agrijax.port.run_fortran import DSSAT_ENGINE

    return DSSAT_ENGINE / "example_data" / "Maize"


def cultivar_of(exp: str, trno: int) -> str:
    from agrijax.io.dssat.filex import read_filex

    x = read_filex(maize_dir() / f"{exp}.MZX")
    tr = next(t for t in x["TREATMENTS"] if int(t["N"]) == trno)
    return str(x["CULTIVARS"][tr["CU"]]["INGENO"]).strip()


def _np_tree(t: Any) -> Any:
    import jax

    return jax.tree.map(lambda x: np.asarray(x), t)


def step_prep(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import day_dssat486_free_harness as H

    data = Path(os.environ["AGRI_JAX_DATA"])
    work = Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "d31"
    keys = H.a12_keys(data)
    refs = H.run_references(keys, work, data, a.jobs)
    runs = [H.build(e, t, refs[H.key_of(e, t)], data) for e, t in keys]
    soil_values, soildyn, real4_sw = H.CONFIGS["free"]
    n_group: dict[tuple[int, str], int] = {}
    for r in runs:
        g = (r.nl, r.mesev)
        n_group[g] = max(n_group.get(g, 0), r.n_days)
    items = []
    for r in runs:
        n = n_group[(r.nl, r.mesev)] + PAD_DAYS
        ps = H.params_of(r, soil_values, real4_sw)
        cul = ps["crop"].cultivar
        items.append(
            {
                "key": r.key,
                "exp": r.exp,
                "trno": r.trno,
                "cultivar": cultivar_of(r.exp, r.trno),
                "theta_pub": [float(np.asarray(getattr(cul, FIELDS[n_]))) for n_ in PARAMS],
                "nl": r.nl,
                "mesev": r.mesev,
                "n_days": r.n_days,
                "n_padded": n,
                "days": np.asarray(r.days, dtype=np.int64),
                "summary": {
                    c: float(r.row[c]) for c in ("HWAM", "CWAM", "H#AM", "EDAT", "ADAT", "MDAT", "HDAT")
                },
                "params": _np_tree(ps),
                "forcing": _np_tree(H.forcing_of(r, n, soil_values, soildyn)),
                "state": _np_tree(H.state_of(r, ps, soil_values)),
            }
        )
    out = out_root() / "inputs.pkl"
    with open(out, "wb") as fh:
        pickle.dump(items, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta = [{k: v for k, v in it.items() if k not in ("params", "forcing", "state", "days")} for it in items]
    dump(out_root() / "inputs_meta.json", {"env": env_info(), "pad_days": PAD_DAYS, "items": meta})
    print("wrote", out, len(items), "runs", flush=True)


def load_items(keys: Sequence[str]) -> list[dict[str, Any]]:
    with open(out_root() / "inputs.pkl", "rb") as fh:
        items = pickle.load(fh)
    by = {it["key"]: it for it in items}
    return [by[k] for k in keys]


# ============================================================================ evaluator (device)
def _first(x: Any) -> Any:
    import jax.numpy as jnp

    return jnp.ravel(x)[0]


def calib_outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
    """Daily outputs a calibration reads (``plantgro_outputs``: the PlantGro.OUT quantities)."""
    from agrijax.core.state import get_path
    from agrijax.models.day_dssat486 import SLOT
    from agrijax.processes.crop.ceres_maize import plantgro_outputs

    pg = plantgro_outputs(get_path(state, f"crops.{SLOT}"), params["crop"], forcing_t["crop"])
    return {**{n: _first(pg[n]) for n in OUT_NAMES}, "istage": _first(pg["istage"])}


def set_cultivar(params: Any, theta: Any) -> Any:
    """``params`` (leading sample axis) with the six cultivar coefficients set to ``theta [S, 6]``."""
    import equinox as eqx

    c = params["crop"].cultivar
    names = [FIELDS[n] for n in PARAMS]
    vals = tuple(
        theta[:, i].astype(getattr(c, f).dtype).reshape(getattr(c, f).shape) for i, f in enumerate(names)
    )
    cul = eqx.tree_at(lambda x: tuple(getattr(x, f) for f in names), c, vals)
    crop = eqx.tree_at(lambda x: x.cultivar, params["crop"], cul)
    return {**params, "crop": crop}


def _first_idx(hit: Any) -> Any:
    import jax.numpy as jnp

    t = hit.shape[0]
    return jnp.where(jnp.any(hit, axis=0), jnp.argmax(hit, axis=0), t)


def _gather_one(y: Any, ist: Any, etype: Any, eout: Any, et: Any) -> Any:
    """One sample: entry values ``[E]`` from its daily outputs ``y [O, T]`` and stage codes ``ist [T]``."""
    import jax.numpy as jnp

    t_n = ist.shape[0]
    codes = jnp.asarray([ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY])
    firsts = jnp.stack([_first_idx(ist == c) for c in (ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY)])
    date = jnp.sum(jnp.where(eout[:, None] == codes[None, :], firsts[None, :], 0), axis=1).astype(y.dtype)
    t_mat = jnp.minimum(firsts[2], t_n - 1)
    t_eff = jnp.where(etype == E_FINAL, t_mat, jnp.clip(et, 0, t_n - 1))
    o = jnp.clip(eout, 0, y.shape[0] - 1)
    val = y[o, t_eff]
    return jnp.where(etype == E_DATE, date, jnp.where(etype == E_PAD, 0.0, val))


class Evaluator:
    """The free-run day on samples ``(theta [6], treatment)``: forward and forward-mode JVP programs,
    one per (group, padded sample count, kind), sharded over the host devices; compile time and call
    time are accumulated."""

    def __init__(self, items: list[dict[str, Any]], entries: dict[str, dict[str, np.ndarray]]) -> None:
        import jax
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as PS

        self.items = items
        self.index = {it["key"]: i for i, it in enumerate(items)}
        self.devs = jax.devices()
        self.ndev = len(self.devs)
        mesh = Mesh(np.asarray(self.devs), ("b",))
        self.rep = NamedSharding(mesh, PS())
        self.bsh = NamedSharding(mesh, PS("b"))
        self.ksh = NamedSharding(mesh, PS(None, "b"))
        groups: dict[tuple[int, str], list[int]] = {}
        for i, it in enumerate(items):
            groups.setdefault((it["nl"], it["mesev"]), []).append(i)
        self.groups = groups
        self.item_group = {}
        self.inputs: dict[tuple[int, str], Any] = {}
        for g, idx in groups.items():
            for b, i in enumerate(idx):
                self.item_group[i] = (g, b)
            gi = [items[i] for i in idx]
            assert len({it["n_padded"] for it in gi}) == 1, g
            stack = lambda trees: jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *trees)  # noqa: E731
            args = (
                stack([it["params"] for it in gi]),
                stack([it["forcing"] for it in gi]),
                stack([it["state"] for it in gi]),
            )
            self.inputs[g] = jax.device_put(args, self.rep)
        self.set_entries(entries)
        self.programs: dict[tuple[Any, ...], Any] = {}
        self.compile_s = 0.0
        self.call_s = 0.0
        self.n_compiled = 0
        self.seasons = {"fwd": 0, "jvp": 0}
        self.calls = {"fwd": 0, "jvp": 0}

    def set_entries(self, entries: dict[str, dict[str, np.ndarray]]) -> None:
        import jax

        self.tables = {}
        for g, idx in self.groups.items():
            tab = {
                k: np.stack([entries[self.items[i]["key"]][k] for i in idx]).astype(np.int32)
                for k in ("type", "out", "t")
            }
            self.tables[g] = jax.device_put(tab, self.rep)

    # ---------------------------------------------------------------- programs
    def _program(self, g: tuple[int, str], sp: int, k: int, mode: str = MODE) -> Any:
        key = (g, sp, k, mode)
        if key in self.programs:
            return self.programs[key]
        import jax
        import jax.numpy as jnp

        from agrijax.core.grad import bind_gradient_mode, bind_unrounded
        from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_processes

        def bind(fn: Any) -> Any:
            """``fn`` in gradient mode ``mode``; ``"unrounded"``: the unrounded model in :data:`MODE`."""
            if mode == "unrounded":
                return bind_gradient_mode(bind_unrounded(fn), MODE)
            return bind_gradient_mode(fn, mode)

        model = day_dssat486(SLOT).compile(
            day_processes(SLOT, mesev=g[1]), outputs=calib_outputs, exact_lags=True
        )
        step = jax.vmap(model.compile(), in_axes=(0, 0, 0))
        n_days = int(self.items[self.groups[g][0]]["n_padded"])

        def sim(inputs: Any, tab: Any, theta: Any, tid: Any) -> Any:
            params_tr, forcing_tr, state_tr = inputs
            params = set_cultivar(jax.tree.map(lambda x: x[tid], params_tr), theta)
            state0 = jax.tree.map(lambda x: x[tid], state_tr)

            def body(s: Any, t: Any) -> Any:
                return step(s, params, jax.tree.map(lambda x: x[tid, t], forcing_tr))

            _, outs = jax.lax.scan(body, state0, jnp.arange(n_days))
            y = jnp.stack([outs[n] for n in OUT_NAMES], axis=0)  # [O, T, S]
            return jax.vmap(_gather_one, in_axes=(2, 1, 0, 0, 0))(
                y, outs["istage"], tab["type"][tid], tab["out"][tid], tab["t"][tid]
            )

        if k == 0:
            fn = bind(sim)
            jf = jax.jit(fn, in_shardings=(self.rep, self.rep, self.bsh, self.bsh), out_shardings=self.bsh)
            args = (self.inputs[g], self.tables[g], jnp.zeros((sp, 6)), jnp.zeros(sp, jnp.int32))
        else:

            def sim_jvp(inputs: Any, tab: Any, theta: Any, tid: Any, v: Any) -> Any:
                f = lambda th: sim(inputs, tab, th, tid)  # noqa: E731
                y, dy = jax.vmap(lambda vv: jax.jvp(f, (theta,), (vv,)), in_axes=0, out_axes=(0, 0))(v)
                return y[0], dy

            fn = bind(sim_jvp)
            jf = jax.jit(
                fn,
                in_shardings=(self.rep, self.rep, self.bsh, self.bsh, self.ksh),
                out_shardings=(self.bsh, self.ksh),
            )
            args = (
                self.inputs[g],
                self.tables[g],
                jnp.zeros((sp, 6)),
                jnp.zeros(sp, jnp.int32),
                jnp.zeros((k, sp, 6)),
            )
        t0 = time.perf_counter()
        compiled = jf.lower(*args).compile()
        self.compile_s += time.perf_counter() - t0
        self.n_compiled += 1
        self.programs[key] = compiled
        return compiled

    def _pad(self, s: int) -> int:
        """Padded sample count: a multiple of the device count on the ladder ndev * 2^j."""
        m = self.ndev
        while m < s:
            m *= 2
        return m

    #: largest sample count of one program call (memory), rounded down to the ladder of :meth:`_pad`
    CHUNK_TARGET = 12288

    @property
    def chunk(self) -> int:
        m = self.ndev
        while m * 2 <= max(self.CHUNK_TARGET, self.ndev):
            m *= 2
        return m

    def run(
        self, theta: np.ndarray, keys: np.ndarray, v: np.ndarray | None = None, mode: str = MODE
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Entry values ``[S, E_MAX]`` of samples ``(theta[s], item keys[s])``; with directions
        ``v [k, S, 6]`` also the tangents ``[k, S, E_MAX]`` (forward-mode AD in physical units)."""
        import jax

        s_n = theta.shape[0]
        out = np.zeros((s_n, E_MAX))
        dout = None if v is None else np.zeros((v.shape[0], s_n, E_MAX))
        item = np.asarray([self.index[k] for k in keys])
        kind = "fwd" if v is None else "jvp"
        for g in self.groups:
            pos = np.nonzero(np.asarray([self.item_group[i][0] == g for i in item]))[0]
            if pos.size == 0:
                continue
            for c0 in range(0, pos.size, self.chunk):
                p = pos[c0 : c0 + self.chunk]
                k = 0 if v is None else v.shape[0]
                # reuse the smallest compiled size that fits (no recompile when fewer rows are active)
                have = sorted(
                    s_
                    for (gg, s_, kk, mm) in self.programs
                    if gg == g and kk == k and mm == mode and s_ >= p.size
                )
                sp = have[0] if have else self._pad(p.size)
                th = np.zeros((sp, 6))
                th[: p.size] = theta[p]
                th[p.size :] = theta[p[0]]
                tid = np.zeros(sp, np.int32)
                tid[: p.size] = [self.item_group[i][1] for i in item[p]]
                tid[p.size :] = tid[0]
                prog = self._program(g, sp, k, mode)
                t0 = time.perf_counter()
                dth = jax.device_put(th, self.bsh)
                dtid = jax.device_put(tid, self.bsh)
                if v is None:
                    y = np.asarray(prog(self.inputs[g], self.tables[g], dth, dtid))
                    out[p] = y[: p.size]
                else:
                    vv = np.zeros((k, sp, 6))
                    vv[:, : p.size] = v[:, p]
                    y, dy = prog(self.inputs[g], self.tables[g], dth, dtid, jax.device_put(vv, self.ksh))
                    out[p] = np.asarray(y)[: p.size]
                    dout[:, p] = np.asarray(dy)[:, : p.size]  # type: ignore[index]
                self.call_s += time.perf_counter() - t0
                self.calls[kind] += 1
                self.seasons[kind] += int(p.size)
        return out, dout

    def stats(self) -> dict[str, Any]:
        return {
            "compile_s": self.compile_s,
            "call_s": self.call_s,
            "n_programs": self.n_compiled,
            "program_calls": dict(self.calls),
            "seasons": dict(self.seasons),
            "n_devices": self.ndev,
        }


# ============================================================================ observations -> entries
def probe_entries(it: dict[str, Any]) -> dict[str, np.ndarray]:
    """Entries that read the stage dates, the values at maturity and on the last real day."""
    ent = [(E_DATE, c, 0) for c in (ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY)]
    ent += [(E_FINAL, o, 0) for o in range(len(OUT_NAMES))]
    ent += [(E_DAY, o, it["n_days"] - 1) for o in range(len(OUT_NAMES))]
    ent += [(E_DAY, o, it["n_padded"] - 1) for o in range(len(OUT_NAMES))]
    return _table(ent)


PROBE_NAMES = (
    ["edate", "adate", "mdate"]
    + [f"{o}_mat" for o in OUT_NAMES]
    + [f"{o}_last_real" for o in OUT_NAMES]
    + [f"{o}_last_pad" for o in OUT_NAMES]
)


def _table(ent: list[tuple[int, int, int]]) -> dict[str, np.ndarray]:
    if len(ent) > E_MAX:
        raise ValueError(f"{len(ent)} entries > E_MAX = {E_MAX}")
    tab = {k: np.zeros(E_MAX, np.int32) for k in ("type", "out", "t")}
    for j, (ty, o, t) in enumerate(ent):
        tab["type"][j], tab["out"][j], tab["t"][j] = ty, o, t
    return tab


class Obs:
    """Per treatment: entry table (device) and ``code``, ``obs`` per entry (host)."""

    def __init__(self) -> None:
        self.tab: dict[str, dict[str, np.ndarray]] = {}
        self.code: dict[str, list[str]] = {}
        self.obs: dict[str, np.ndarray] = {}
        self.day: dict[str, np.ndarray] = {}

    def add(self, key: str, rows: list[tuple[str, int, int, int, float]]) -> None:
        """``rows``: (code, type, out/stage, day, observed value)."""
        self.tab[key] = _table([(ty, o, t) for _, ty, o, t, _ in rows])
        self.code[key] = [r[0] for r in rows]
        self.obs[key] = np.asarray([r[4] for r in rows] + [0.0] * (E_MAX - len(rows)))
        self.day[key] = np.asarray([r[3] for r in rows] + [-1] * (E_MAX - len(rows)))


def twin_obs(ev: Evaluator, problems: dict[str, list[str]], codes: Sequence[str]) -> Obs:
    """Noise-free observations at the published cultivar (the truth): stage dates, values at maturity
    and LAI every :data:`LAI_EVERY` days from emergence to maturity (on the real season)."""
    keys = [k for ks in problems.values() for k in ks]
    items = {it["key"]: it for it in ev.items}
    ev.set_entries({k: probe_entries(items[k]) for k in ev.index})
    th = np.asarray([items[k]["theta_pub"] for k in keys])
    y, _ = ev.run(th, np.asarray(keys))
    obs = Obs()
    for s, k in enumerate(keys):
        e_i, m_i = int(y[s, 0]), int(y[s, 2])
        rows: list[tuple[str, int, int, int, float]] = []
        for c in codes:
            if c in CODE_STAGE:
                rows.append((c, E_DATE, CODE_STAGE[c], 0, float(y[s, 1 if c == "ADAT" else 2])))
            elif c == "LAID":
                for t in range(e_i, min(m_i, items[k]["n_days"] - 1) + 1, LAI_EVERY):
                    rows.append((c, E_DAY, OUT_NAMES.index("lai"), t, math.nan))
            else:
                o = OUT_NAMES.index(CODE_OUT[c])
                rows.append((c, E_FINAL, o, 0, float(y[s, 3 + o])))
        obs.add(k, rows)
    # the LAI series values: one more forward at the truth with the final tables
    ev.set_entries({k: obs.tab.get(k, probe_entries(items[k])) for k in ev.index})
    y, _ = ev.run(th, np.asarray(keys))
    for s, k in enumerate(keys):
        n = len(obs.code[k])
        o = obs.obs[k]
        o[:n] = np.where(np.isnan(o[:n]), y[s, :n], o[:n])
        assert np.array_equal(o[:n], y[s, :n]), k  # the twin observations are the model at the truth
    return obs


def real_obs(
    ev: Evaluator, problems: dict[str, list[str]], codes: Sequence[str]
) -> tuple[Obs, dict[str, Any]]:
    """Observations of the DSSAT A / T files (:func:`agrijax.calib.observations.observation_targets`)
    on each treatment's real days (padded days never match an observation date)."""
    from agrijax.calib.observations import observation_targets
    from agrijax.io.dssat.observed import read_observed

    items = {it["key"]: it for it in ev.items}
    obs = Obs()
    info: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    for ks in problems.values():
        for k in ks:
            it = items[k]
            if it["exp"] not in cache:
                cache[it["exp"]] = read_observed(maize_dir() / f"{it['exp']}.MZX")
            days = np.full(it["n_padded"], -1, np.int64)
            days[: it["n_days"]] = it["days"]
            ot = observation_targets(
                cache[it["exp"]], [it["trno"]], days[None], codes=codes, sim_start=[int(it["days"][0])]
            )
            rows: list[tuple[str, int, int, int, float]] = []
            for t in ot.targets:
                v = np.asarray(ot.observed[t.name])
                if t.kind == "date":
                    rows.append((t.name, E_DATE, int(t.code), 0, float(v[0])))
                elif t.kind == "final":
                    rows.append((t.name, E_FINAL, OUT_NAMES.index(t.output), 0, float(v[0])))
                else:
                    m = np.asarray(t.mask)[:, 0]
                    if t.name in ("HWAM", "CWAM", "H#AM"):  # a last-day series of a partly observed final
                        rows.append((t.name, E_FINAL, OUT_NAMES.index(t.output), 0, float(v[m, 0][0])))
                        continue
                    for d in np.nonzero(m)[0]:
                        rows.append((t.name, E_DAY, OUT_NAMES.index(t.output), int(d), float(v[d, 0])))
            obs.add(k, rows)
            # A-file quantities observed for this treatment but not usable (e.g. an observed date that
            # lies after the last day of the reference season, whose forcing the dump tables hold)
            fa = cache[it["exp"]].filea
            dropped = []
            for c in codes:
                if c in ("LAID", "CWAD", "GWAD") or fa is None or c not in fa.codes:
                    continue
                v = fa.value(it["trno"], c)
                have = v is not None and np.isfinite(v) and (v > 0 if c in CODE_STAGE else True)
                if have and c not in [r_[0] for r_ in rows]:
                    dropped.append(c)
            t_drop = [n for n in ot.notes if "outside the simulated days" in n]
            if dropped or t_drop:
                print(
                    f"WARNING {k}: observed A-file targets dropped {dropped}; T-file notes {t_drop}",
                    flush=True,
                )
            info[k] = {
                "n_obs": ot.n_obs,
                "codes_used": sorted({r_[0] for r_ in rows}),
                "notes": list(ot.notes),
                "unsupported": sorted(ot.unsupported),
                "dropped_a": dropped,
            }
    return obs, info


# ============================================================================ objective (host)
class Objective:
    """The losses of candidates of several problems from entry values (module docstring)."""

    def __init__(self, obs: Obs, problems: dict[str, list[str]]) -> None:
        self.obs = obs
        self.problems = problems
        self.pnames = list(problems)
        self.winv: dict[str, np.ndarray] = {k: np.zeros(E_MAX) for ks in problems.values() for k in ks}
        self.scale_info: dict[str, dict[str, float]] = {}
        for p, ks in problems.items():
            codes = sorted({c for k in ks for c in obs.code[k]})
            self.scale_info[p] = {}
            for c in codes:
                vals = [obs.obs[k][j] for k in ks for j, cc in enumerate(obs.code[k]) if cc == c]
                scale = 1.0 if c in CODE_STAGE else max(float(np.mean(np.abs(vals))), 1e-12)
                self.scale_info[p][c] = scale
                for k in ks:
                    for j, cc in enumerate(obs.code[k]):
                        if cc == c:
                            self.winv[k][j] = 1.0 / (scale * math.sqrt(len(vals)))
        self.codes = {k: np.asarray(obs.code[k] + [""] * (E_MAX - len(obs.code[k]))) for k in self.winv}

    def view(self, codes: Sequence[str] | None, lai_pre: bool = False) -> dict[str, np.ndarray]:
        """``{key: [E_MAX] 0/1}``: the entries of ``codes`` (``None``: all); ``lai_pre`` adds the LAI
        observations before the observed silking day."""
        out = {}
        for k, cc in self.codes.items():
            m = np.isin(cc, list(codes)) if codes is not None else cc != ""
            if lai_pre:
                ad = [self.obs.obs[k][j] for j, c in enumerate(cc) if c == "ADAT"]
                if ad:
                    m = m | ((cc == "LAID") & (self.obs.day[k] < ad[0]))
            out[k] = m.astype(float)
        return out

    def layout(self, prob_of: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Samples of candidates ``c`` of problems ``prob_of[c]``: (candidate index, item key)."""
        cand, keys = [], []
        for c, p in enumerate(prob_of):
            for k in self.problems[self.pnames[int(p)]]:
                cand.append(c)
                keys.append(k)
        return np.asarray(cand), np.asarray(keys)

    def residuals(
        self, y: np.ndarray, keys: np.ndarray, view: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """Normalised residuals ``[S, E_MAX]`` (0 off the view and on padding)."""
        obs = np.stack([self.obs.obs[k] for k in keys])
        w = np.stack([self.winv[k] for k in keys])
        if view is not None:
            w = w * np.stack([view[k] for k in keys])
        return (y - obs) * w

    def weights(self, keys: np.ndarray, view: dict[str, np.ndarray] | None = None) -> np.ndarray:
        w = np.stack([self.winv[k] for k in keys])
        if view is not None:
            w = w * np.stack([view[k] for k in keys])
        return w

    def per_candidate(self, per_sample: np.ndarray, cand: np.ndarray, n: int) -> np.ndarray:
        out = np.zeros(n)
        np.add.at(out, cand, per_sample)
        return out


class Problem:
    """Evaluator + objective for candidate batches ``theta [C, 6]`` with ``prob_of [C]``: every
    evaluation returns the loss of a view and the original objective, and counts calls per row."""

    def __init__(self, ev: Evaluator, obj: Objective) -> None:
        self.ev = ev
        self.obj = obj
        ev.set_entries({k: obj.obs.tab.get(k, _table([])) for k in ev.index})
        self.full = obj.view(None)

    def evaluate(
        self, theta: np.ndarray, prob_of: np.ndarray, view: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        cand, keys = self.obj.layout(prob_of)
        y, _ = self.ev.run(theta[cand], keys)
        r_v = self.obj.residuals(y, keys, view)
        r_f = self.obj.residuals(y, keys, self.full)
        c = theta.shape[0]
        return (
            self.obj.per_candidate(np.sum(r_v**2, axis=1), cand, c),
            self.obj.per_candidate(np.sum(r_f**2, axis=1), cand, c),
        )


# ============================================================================ parameter space
def space() -> Any:
    from agrijax.calib.ceres import ceres_space

    return ceres_space(PARAMS)


def to_theta(sp: Any, z: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp

    return np.asarray(sp.to_physical(jnp.asarray(z)))


def to_z(sp: Any, theta: np.ndarray) -> np.ndarray:
    import jax.numpy as jnp

    return np.asarray(sp.to_unconstrained(jnp.asarray(theta)))


def dtheta_dz(sp: Any, z: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-z))
    return sp.width * s * (1.0 - s)


def make_starts(sp: Any, center: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    """``N_STARTS`` starting points: ``"perturb"`` (center * (1 + U(-PERTURB, PERTURB))) or ``"box"``
    (uniform), both inside the inner box."""
    lo = sp.lower + MARGIN * sp.width
    hi = sp.upper - MARGIN * sp.width
    if mode == "perturb":
        th = center[None] * (1.0 + rng.uniform(-PERTURB, PERTURB, (N_STARTS, center.size)))
    else:
        th = lo + (hi - lo) * rng.random((N_STARTS, center.size))
    return np.clip(th, lo, hi)


# ============================================================================ optimisers (batched rows)
class Tracker:
    """Per row: forward / JVP calls, best-so-far original objective after each call batch, wall."""

    def __init__(self, n_rows: int, t0: float) -> None:
        self.n_fwd = np.zeros(n_rows)  # float: a shared trust report is split over the rows
        self.n_jvp = np.zeros(n_rows)
        self.best_full = np.full(n_rows, np.inf)
        self.best_full_z = np.zeros((n_rows, 6))
        self.hist: list[tuple[float, list[float], list[float], list[float]]] = []
        self.t0 = t0

    def record(
        self, rows: np.ndarray, full: np.ndarray, z: np.ndarray, *, fwd: float = 0, jvp: float = 0
    ) -> None:
        """``full [len(rows), m]``: the original objective of the ``m`` points ``z [len(rows), m, 6]``."""
        self.n_fwd[rows] += fwd
        self.n_jvp[rows] += jvp
        j = np.argmin(full, axis=1)
        fb = full[np.arange(rows.size), j]
        better = fb < self.best_full[rows]
        self.best_full[rows[better]] = fb[better]
        self.best_full_z[rows[better]] = z[better, j[better]]
        self.snap()

    def snap(self) -> None:
        self.hist.append(
            (time.perf_counter() - self.t0, self.n_fwd.tolist(), self.n_jvp.tolist(), self.best_full.tolist())
        )

    def calls_to(self, q: np.ndarray) -> dict[str, Any]:
        """Per row: forward / JVP calls and wall when the best original objective first was <= q[row]."""
        n = self.n_fwd.size
        fwd = np.full(n, -1)
        jvp = np.full(n, -1)
        wall = np.full(n, np.nan)
        for t, f, g, b in self.hist:
            hit = (np.asarray(b) <= q) & (fwd < 0)
            fwd[hit] = np.asarray(f)[hit]
            jvp[hit] = np.asarray(g)[hit]
            wall[hit] = t
        return {"fwd": fwd, "jvp": jvp, "wall_s": wall}


class _SubTracker(Tracker):
    """A :class:`Tracker` view on the rows ``rows`` of ``parent`` (row ``i`` here = ``rows[i]``)."""

    def __init__(self, parent: Tracker, rows: np.ndarray) -> None:
        self.parent = parent
        self.map = np.asarray(rows)

    @property
    def n_fwd(self) -> Any:  # type: ignore[override]
        return _Proxy(self.parent.n_fwd, self.map)

    @property
    def n_jvp(self) -> Any:  # type: ignore[override]
        return _Proxy(self.parent.n_jvp, self.map)

    def record(
        self, rows: np.ndarray, full: np.ndarray, z: np.ndarray, *, fwd: float = 0, jvp: float = 0
    ) -> None:
        self.parent.record(self.map[rows], full, z, fwd=fwd, jvp=jvp)

    def snap(self) -> None:
        self.parent.snap()


class _Proxy:
    """``arr[map[i]]`` for item access (the ``+=`` of the optimisers)."""

    def __init__(self, arr: np.ndarray, m: np.ndarray) -> None:
        self.arr, self.m = arr, m

    def __getitem__(self, i: Any) -> Any:
        return self.arr[self.m[i]]

    def __setitem__(self, i: Any, v: Any) -> None:
        self.arr[self.m[i]] = v


def cma_rows(
    evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    z0: np.ndarray,
    dims: Sequence[int],
    tracker: Tracker,
    *,
    max_evals: int,
    tol: np.ndarray,
    seed: int,
    restart: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """(mu/mu_w, lambda)-CMA-ES on the coordinates ``dims`` of ``z`` (others fixed at ``z0``), one
    search per row, the default strategy parameters of :func:`agrijax.calib.optim.cma_es` (Hansen 2016,
    Table 1), with **restarts**: a row whose best loss has not improved for :data:`CMA_STALL`
    generations (a plateau of the piecewise-constant date loss, or convergence to a local minimum), or
    whose step fell below 1e-10, restarts from ``restart(rows)`` (new means, ``z`` units; the same
    distribution as the starts) with the initial step and an identity covariance, the best point kept.
    A row stops when its loss is <= ``tol[row]`` or its budget is used.
    ``evaluate(z [M, 6], rows [M]) -> (loss, original objective)``."""
    from agrijax.calib.optim import _cma_defaults

    r_n = z0.shape[0]
    d = list(dims)
    n = len(d)
    lam = 4 + int(np.floor(3 * np.log(n)))
    p = _cma_defaults(n, lam)
    rng = np.random.default_rng(seed)
    mean = z0[:, d].copy()
    sigma = np.full(r_n, SIGMA0)
    c = np.tile(np.eye(n), (r_n, 1, 1))
    pc = np.zeros((r_n, n))
    ps = np.zeros((r_n, n))
    best_z = z0.copy()
    best_l = np.full(r_n, np.inf)
    evals = np.zeros(r_n, int)
    done = np.zeros(r_n, bool)
    gen = np.zeros(r_n, int)
    last_gain = np.zeros(r_n, int)
    run_best = np.full(r_n, np.inf)
    restarts = np.zeros(r_n, int)
    while not done.all():
        rows = np.nonzero(~done)[0]
        ev_, bv = np.linalg.eigh(c[rows])
        dsq = np.sqrt(np.maximum(ev_, 0.0))
        y = np.einsum("rij,rkj->rki", bv * dsq[:, None, :], rng.standard_normal((rows.size, lam, n)))
        x = mean[rows][:, None, :] + sigma[rows][:, None, None] * y
        zfull = np.repeat(z0[rows][:, None, :], lam, axis=1)
        zfull[:, :, d] = x
        f, full = evaluate(zfull.reshape(-1, 6), np.repeat(rows, lam))
        f = np.where(np.isfinite(f), f, np.inf).reshape(rows.size, lam)
        tracker.record(rows, full.reshape(rows.size, lam), zfull, fwd=lam)
        evals[rows] += lam
        order = np.argsort(f, axis=1)
        gb = f[np.arange(rows.size), order[:, 0]]
        better = gb < best_l[rows]
        best_l[rows[better]] = gb[better]
        best_z[rows[better]] = zfull[np.arange(rows.size), order[:, 0]][better]
        gain = gb < run_best[rows] * (1.0 - CMA_GAIN_REL)
        run_best[rows] = np.minimum(run_best[rows], gb)
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
        small = sigma[rows] * np.sqrt(np.max(np.diagonal(c[rows], axis1=1, axis2=2), axis=1)) < 1e-10
        done[rows] = (best_l[rows] <= tol[rows]) | (evals[rows] + lam > max_evals)
        stuck = rows[~done[rows] & (small | (gen[rows] - last_gain[rows] >= CMA_STALL))]
        if stuck.size:
            if restart is None:
                done[stuck] = True
            else:
                mean[stuck] = restart(stuck)[:, d]
                sigma[stuck] = SIGMA0
                c[stuck] = np.eye(n)
                pc[stuck] = 0.0
                ps[stuck] = 0.0
                gen[stuck] = 0
                last_gain[stuck] = 0
                run_best[stuck] = np.inf
                restarts[stuck] += 1
    return {"z_best": best_z, "loss_best": best_l, "evals": evals, "popsize": lam, "restarts": restarts}


def random_rows(
    evaluate: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    lo_z: np.ndarray,
    hi_z: np.ndarray,
    tracker: Tracker,
    *,
    max_evals: int,
    tol: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Uniform random search in the box ``[lo_z, hi_z]`` (per row), ``RS_BATCH`` per call; a row stops
    at ``tol`` or its budget."""
    r_n = lo_z.shape[0]
    rng = np.random.default_rng(seed)
    best_z = np.zeros((r_n, 6))
    best_l = np.full(r_n, np.inf)
    evals = np.zeros(r_n, int)
    done = np.zeros(r_n, bool)
    while not done.all():
        rows = np.nonzero(~done)[0]
        x = lo_z[rows][:, None] + (hi_z - lo_z)[rows][:, None] * rng.random((rows.size, RS_BATCH, 6))
        f, full = evaluate(x.reshape(-1, 6), np.repeat(rows, RS_BATCH))
        f = f.reshape(rows.size, RS_BATCH)
        tracker.record(rows, full.reshape(rows.size, RS_BATCH), x, fwd=RS_BATCH)
        evals[rows] += RS_BATCH
        j = np.argmin(f, axis=1)
        fb = f[np.arange(rows.size), j]
        better = fb < best_l[rows]
        best_l[rows[better]] = fb[better]
        best_z[rows[better]] = x[np.arange(rows.size), j][better]
        done[rows] = (best_l[rows] <= tol[rows]) | (evals[rows] + RS_BATCH > max_evals)
    return {"z_best": best_z, "loss_best": best_l, "evals": evals}


# ---------------------------------------------------------------------------- trust (batched)
def _scan_diag(
    grid: np.ndarray, y: np.ndarray, g: np.ndarray, width: float, cfg: Any
) -> dict[str, np.ndarray]:
    """The diagnostics of :func:`agrijax.calib.trust.line_scan` from scan values ``y [P, m]`` and
    directional AD derivatives ``g [P, m]`` (same formulas, computed from batched evaluations)."""
    dx = np.diff(grid)[:, None]
    actual = np.diff(y, axis=0)
    pred = 0.5 * (g[1:] + g[:-1]) * dx
    rng = y.max(axis=0) - y.min(axis=0)
    scale = np.maximum(np.abs(y).max(axis=0), 1.0)
    unexplained = np.abs(actual - pred)
    jump = unexplained > cfg.jump_frac * np.maximum(rng, cfg.abs_floor * scale)
    zero = np.abs(g) * width <= cfg.abs_floor * scale
    runs = np.zeros(y.shape[1], dtype=int)
    moves = np.zeros(y.shape[1], dtype=bool)
    for j in range(y.shape[1]):
        best, cur, start, best_start = 0, 0, 0, 0
        for k, zz in enumerate(zero[:, j]):
            if zz:
                if cur == 0:
                    start = k
                cur += 1
                if cur > best:
                    best, best_start = cur, start
            else:
                cur = 0
        runs[j] = best
        if best > 1:
            seg = y[best_start : best_start + best, j]
            moves[j] = bool(seg.max() - seg.min() > cfg.abs_floor * scale[j])
    return {
        "grid": grid,
        "y": y,
        "g": g,
        "finite": np.all(np.isfinite(y), axis=0) & np.all(np.isfinite(g), axis=0),
        "n_jumps": jump.sum(axis=0),
        "max_jump": (unexplained / np.maximum(rng, 1e-300)).max(axis=0),
        "zero_frac": zero.mean(axis=0),
        "zero_run": runs,
        "zero_run_moves": moves,
        "range": rng,
        "secant": (y[-1] - y[0]) / (grid[-1] - grid[0]),
        "ad_mean": g.mean(axis=0),
    }


def _fd_diag(
    ad: np.ndarray, y0: np.ndarray, fds: list[np.ndarray], width: np.ndarray, cfg: Any
) -> dict[str, np.ndarray]:
    """The agreement flags of :func:`agrijax.calib.trust.fd_check` (``ad``, ``fd`` ``[m, n]``)."""
    out: dict[str, np.ndarray] = {"ad": ad, "y0": y0}
    for k, fd in enumerate(fds):
        den = np.maximum(np.maximum(np.abs(ad), np.abs(fd)), 1e-300)
        rel = np.abs(ad - fd) / den
        floor = cfg.abs_floor * np.maximum(np.abs(y0), 1.0)[:, None] / width[None, :]
        both_zero = (np.abs(ad) <= floor) & (np.abs(fd) <= floor)
        out[f"fd{k}"] = fd
        out[f"rel_err{k}"] = np.where(both_zero, 0.0, rel)
        out[f"agree{k}"] = both_zero | (rel <= cfg.fd_rtol[k])
    return out


def trust_rows(
    pb: Problem,
    theta: np.ndarray,
    rows_prob: np.ndarray,
    pidx: Sequence[int],
    view: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Gradient-trust report (format of :func:`agrijax.calib.trust.trust_report`) of the per-treatment
    losses (``view``) of the problems ``rows_prob[r]`` with respect to the coefficients ``pidx`` at
    ``theta[r]``: AD (forward mode) against central differences at the two steps of the config, and a
    line scan of :data:`TRUST_SCAN` points over +- scan_frac of the width per coefficient. Everything
    is batched; the AD values come from one ``len(pidx)``-direction JVP program (the scans use the
    direction of their coefficient), so the stage-2 JVP program is the same compiled program."""
    from agrijax.calib.trust import TrustConfig, _classify, _level

    cfg = TrustConfig(n_scan=TRUST_SCAN)
    sp = space()
    lo, hi, width = sp.lower, sp.upper, sp.width
    r_n = theta.shape[0]
    n = len(pidx)
    obj = pb.obj
    keys_of = [obj.problems[obj.pnames[int(p)]] for p in rows_prob]
    # samples: [centres | scan points]; the tangents of all n directions for every sample
    pts, keys = [], []
    for r in range(r_n):
        for k in keys_of[r]:
            pts.append(theta[r])
            keys.append(k)
    n_c = len(pts)
    grids = np.zeros((r_n, n, cfg.n_scan))
    for r in range(r_n):
        for a, i in enumerate(pidx):
            a0 = max(lo[i], theta[r, i] - cfg.scan_frac * width[i])
            b0 = min(hi[i], theta[r, i] + cfg.scan_frac * width[i])
            grids[r, a] = np.linspace(a0, b0, cfg.n_scan)
            for gval in grids[r, a]:
                t = theta[r].copy()
                t[i] = gval
                for k in keys_of[r]:
                    pts.append(t)
                    keys.append(k)
    pts = np.asarray(pts)
    keys = np.asarray(keys)
    v = np.zeros((n, pts.shape[0], 6))
    for a, i in enumerate(pidx):
        v[a, :, i] = 1.0
    y, dy = pb.ev.run(pts, keys, v)
    r_ = obj.residuals(y, keys, view)
    w = obj.weights(keys, view)
    l_all = np.sum(r_**2, axis=1)
    g_all = 2.0 * np.sum(r_[None] * dy * w[None], axis=2)  # type: ignore[index]  # [n, S]
    l0, g0 = l_all[:n_c], g_all[:, :n_c]
    ls, gs = l_all[n_c:], g_all[:, n_c:]
    # central differences at the two steps
    fd_pts, fd_keys = [], []
    for frac in cfg.fd_steps:
        for i in pidx:
            for sgn in (1.0, -1.0):
                t = pts[:n_c].copy()
                t[:, i] += sgn * frac * width[i]
                fd_pts.append(t)
                fd_keys.append(keys[:n_c])
    kf = np.concatenate(fd_keys)
    yf, _ = pb.ev.run(np.concatenate(fd_pts), kf)
    lf = np.sum(obj.residuals(yf, kf, view) ** 2, axis=1).reshape(len(cfg.fd_steps), n, 2, n_c)
    split = TRUST_LEVEL2 == "split"
    if split:
        # the two paths of the straight-through derivative: the exact-mode derivative at the centres,
        # and the unrounded model's derivative and small-step central difference
        def loss_grad(yy: np.ndarray, dd: np.ndarray, kk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            rr = obj.residuals(yy, kk, view)
            return np.sum(rr**2, axis=1), 2.0 * np.sum(rr[None] * dd * obj.weights(kk, view)[None], axis=2)

        yx, dyx = pb.ev.run(pts[:n_c], keys[:n_c], v[:, :n_c], mode="exact")
        _, gx0 = loss_grad(yx, dyx, keys[:n_c])  # type: ignore[arg-type]
        yu, dyu = pb.ev.run(pts[:n_c], keys[:n_c], v[:, :n_c], mode="unrounded")
        lu0, gu0 = loss_grad(yu, dyu, keys[:n_c])  # type: ignore[arg-type]
        nf = n * 2 * n_c
        yfu, _ = pb.ev.run(np.concatenate(fd_pts[: n * 2]), kf[:nf], mode="unrounded")
        lfu = np.sum(obj.residuals(yfu, kf[:nf], view) ** 2, axis=1).reshape(n, 2, n_c)
    reports = []
    pos = 0
    spos = 0
    for r in range(r_n):
        ks = keys_of[r]
        m = len(ks)
        sl = slice(pos, pos + m)
        ad = g0[:, sl].T  # [m, n]
        fds = [
            (lf[s_, :, 0, sl] - lf[s_, :, 1, sl]).T / (2 * cfg.fd_steps[s_] * width[list(pidx)])[None, :]
            for s_ in range(2)
        ]
        fd = _fd_diag(ad, l0[sl], fds, width[list(pidx)], cfg)
        if split:
            wp = width[list(pidx)]
            fx = _fd_diag(gx0[:, sl].T, l0[sl], fds[:1], wp, cfg)
            fdu = (lfu[:, 0, sl] - lfu[:, 1, sl]).T / (2 * cfg.fd_steps[0] * wp)[None, :]
            fu = _fd_diag(gu0[:, sl].T, lu0[sl], [fdu], wp, cfg)
            fd["agree0"] = fx["agree0"] & fu["agree0"]
            fd["rel_err0"] = np.maximum(fx["rel_err0"], fu["rel_err0"])
        params = {}
        for a, i in enumerate(pidx):
            cnt = cfg.n_scan * m
            yy = ls[spos : spos + cnt].reshape(cfg.n_scan, m)
            gg = gs[a, spos : spos + cnt].reshape(cfg.n_scan, m)
            spos += cnt
            sc = _scan_diag(grids[r, a], yy, gg, float(width[i]), cfg)
            outs = {}
            for j, k in enumerate(ks):
                outs[k] = {
                    "ad": float(ad[j, a]),
                    "fd_small": float(fds[0][j, a]),
                    "fd_large": float(fds[1][j, a]),
                    "rel_err_small": float(fd["rel_err0"][j, a]),
                    "rel_err_large": float(fd["rel_err1"][j, a]),
                    "n_jumps": int(sc["n_jumps"][j]),
                    "max_jump_frac": float(sc["max_jump"][j]),
                    "zero_frac": float(sc["zero_frac"][j]),
                    "zero_run_moves": bool(sc["zero_run_moves"][j]),
                    "scan_range": float(sc["range"][j]),
                    "class": _classify(fd, sc, j, a, cfg),
                    "level": _level(fd, sc, j, a, cfg),
                }
            params[PARAMS[i]] = {"x": float(theta[r, i]), "outputs": outs}
        reports.append({"params": params, "outputs": list(ks)})
        pos += m
    return reports


#: model calls of one trust report per problem: one JVP at the centre and n x n_scan scan JVPs (all
#: with the n directions of the stage-2 program), 2 x 2 x n central-difference forwards
def trust_calls(n: int) -> tuple[int, int]:
    if TRUST_LEVEL2 == "split":  # + the unrounded small-step forwards, the exact and unrounded JVPs
        return 6 * n, 3 + n * TRUST_SCAN
    return 4 * n, 1 + n * TRUST_SCAN


def lm_rows(
    pb: Problem,
    z0: np.ndarray,
    rows_prob: np.ndarray,
    pidx: Sequence[int],
    view: dict[str, np.ndarray],
    secant: list[np.ndarray],
    tracker: Tracker,
    *,
    free: np.ndarray,
    fixed: np.ndarray,
    ad_idx: Sequence[int] = I_GROWTH,
    iters: int = LM_ITERS,
    tol: float = LM_TOL,
) -> dict[str, Any]:
    """Levenberg-Marquardt on the coordinates ``pidx`` of ``z`` (Marquardt 1963 scaling, lambda x3 /
    /3), per row, on the residuals of ``view``. The residual Jacobian column of coefficient
    ``pidx[a]`` on treatment ``b`` is forward-mode AD (one JVP program with the directions ``ad_idx``)
    where ``secant[row][b, a]`` is false, else the central secant of that treatment's residuals at
    +- :data:`SECANT_DZ` in ``z``; columns of coefficients not free on the row's problem are 0. Calls
    per iteration: one JVP at a new point, one forward trial, two forwards per column with a secant
    pair."""
    sp = space()
    obj = pb.obj
    r_n = z0.shape[0]
    n = len(pidx)
    col_ad = {a: ad_idx.index(i) for a, i in enumerate(pidx) if i in ad_idx}
    keys_of = [obj.problems[obj.pnames[int(p)]] for p in rows_prob]
    fr = [free[rows_prob[r]][list(pidx)] for r in range(r_n)]
    fr_all = free[rows_prob]

    def th_of(zz: np.ndarray, rr: np.ndarray) -> np.ndarray:
        """Physical values: the coefficients not free on a row's problem at their fixed values."""
        return np.where(fr_all[rr], to_theta(sp, zz), fixed[rr])

    sec = [np.asarray(secant[r], bool) & fr[r][None, :] for r in range(r_n)]
    for r in range(r_n):  # a column without AD is a secant column
        for a in range(n):
            if a not in col_ad:
                sec[r][:, a] = fr[r][a]
    z = z0.copy()
    lam = np.full(r_n, LM_LAMBDA0)
    active = np.ones(r_n, bool)
    need_j = np.ones(r_n, bool)
    cur_r: list[np.ndarray | None] = [None] * r_n
    cur_j: list[np.ndarray | None] = [None] * r_n
    loss = np.full(r_n, np.inf)
    n_acc = np.zeros(r_n, int)
    stall = np.zeros(r_n, int)  # accepted steps in a row with a relative decrease < 1e-6
    rejects = np.zeros(r_n, int)  # rejected trials in a row
    for _ in range(iters):
        rows = np.nonzero(active & need_j)[0]
        if rows.size:
            th = th_of(z[rows], rows)
            dz = dtheta_dz(sp, z[rows])
            th_s, keys_s, row_s = [], [], []
            for a_, r in enumerate(rows):
                for k in keys_of[r]:
                    th_s.append(th[a_])
                    keys_s.append(k)
                    row_s.append(a_)
            th_s = np.asarray(th_s)
            keys_s = np.asarray(keys_s)
            row_s = np.asarray(row_s)
            v = np.zeros((len(ad_idx), th_s.shape[0], 6))
            for a, i in enumerate(ad_idx):
                v[a, :, i] = 1.0
            y, dy = pb.ev.run(th_s, keys_s, v)
            w = obj.weights(keys_s, view)
            res = obj.residuals(y, keys_s, view)
            jac = np.zeros((n, th_s.shape[0], E_MAX))
            for a, q in col_ad.items():
                jac[a] = dy[q] * w * dz[row_s, pidx[a]][:, None]  # type: ignore[index]  # d r / d z
            b_of = np.asarray([keys_of[rows[row_s[s_]]].index(keys_s[s_]) for s_ in range(keys_s.size)])
            for a in range(n):  # non-free columns and secant pairs start from 0
                off_ = np.asarray(
                    [
                        not fr[rows[row_s[s_]]][a] or sec[rows[row_s[s_]]][b_of[s_], a]
                        for s_ in range(keys_s.size)
                    ]
                )
                jac[a, off_] = 0.0
            sec_cols = sorted({a for r in rows for a in range(n) if sec[r][:, a].any()})
            n_sec = np.zeros(rows.size, int)
            if sec_cols:
                pts, kk, sel_of = [], [], []
                for a in sec_cols:
                    # the samples of the rows with a secant pair in this column (both signs)
                    need = np.asarray([sec[rows[row_s[s_]]][:, a].any() for s_ in range(keys_s.size)])
                    sel = np.nonzero(need)[0]
                    sel_of.append(sel)
                    for sgn in (1.0, -1.0):
                        zz = z[rows].copy()
                        zz[:, pidx[a]] += sgn * SECANT_DZ
                        pts.append(th_of(zz, rows)[row_s[sel]])
                        kk.append(keys_s[sel])
                ysec, _ = pb.ev.run(np.concatenate(pts), np.concatenate(kk))
                rsec = obj.residuals(ysec, np.concatenate(kk), view)
                off = 0
                for q, a in enumerate(sec_cols):
                    sel = sel_of[q]
                    rp = rsec[off : off + sel.size]
                    rm = rsec[off + sel.size : off + 2 * sel.size]
                    off += 2 * sel.size
                    d_col = (rp - rm) / (2 * SECANT_DZ)
                    use = np.asarray([sec[rows[row_s[s_]]][b_of[s_], a] for s_ in sel], dtype=bool)
                    jac[a, sel[use]] = d_col[use]
                for a_, r in enumerate(rows):
                    n_sec[a_] = 2 * sum(1 for a in sec_cols if sec[r][:, a].any())
            for a_, r in enumerate(rows):
                sel = row_s == a_
                cur_r[r] = res[sel].reshape(-1)
                cur_j[r] = jac[:, sel].reshape(n, -1).T
                loss[r] = float(np.sum(cur_r[r] ** 2))  # type: ignore[operator]
            full_c = pb.obj.per_candidate(
                np.sum(obj.residuals(y, keys_s, pb.full) ** 2, axis=1), row_s, rows.size
            )
            tracker.record(rows, full_c[:, None], z[rows][:, None], jvp=1)
            for a_, r in enumerate(rows):
                tracker.n_fwd[r] += n_sec[a_]
            need_j[rows] = False
        rows = np.nonzero(active)[0]
        if rows.size == 0:
            break
        trial = z.copy()
        for r in rows:
            jj, rr = cur_j[r], cur_r[r]
            a_m = jj.T @ jj  # type: ignore[union-attr]
            g = jj.T @ rr  # type: ignore[union-attr]
            dmat = np.diag(np.maximum(np.diag(a_m), 1e-12))
            try:
                step = -np.linalg.solve(a_m + lam[r] * dmat, g)
            except np.linalg.LinAlgError:
                step = np.zeros(n)
            step = np.where(fr[r], step, 0.0)
            trial[r, list(pidx)] = z[r, list(pidx)] + step
        lt, ft = pb.evaluate(th_of(trial[rows], rows), rows_prob[rows], view)
        tracker.record(rows, ft[:, None], trial[rows][:, None], fwd=1)
        for a_, r in enumerate(rows):
            if np.isfinite(lt[a_]) and lt[a_] < loss[r]:
                rel = (loss[r] - lt[a_]) / max(loss[r], 1e-300)
                z[r] = trial[r]
                loss[r] = lt[a_]
                lam[r] = max(lam[r] / 3.0, 1e-12)
                need_j[r] = True
                n_acc[r] += 1
                rejects[r] = 0
                stall[r] = stall[r] + 1 if rel < 1e-6 else 0
            else:
                lam[r] *= 3.0
                rejects[r] += 1
            if loss[r] <= tol or stall[r] >= 3 or rejects[r] >= 12 or lam[r] > 1e10:
                active[r] = False
    return {"z_best": z, "loss_best": loss, "accepted": n_acc, "lambda": lam}


# ============================================================================ experiment runner
def recovery(theta: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.abs(theta - truth) / np.maximum(np.abs(truth), 1e-12)


def build(kind: str, codes: Sequence[str]) -> tuple[Problem, dict[str, list[str]], dict[str, Any]]:
    """Problems of ``kind`` (``twin``, ``real``, ``ufga``: the identifiability case, calibration
    treatments only; ``ufga_all``: its three treatments). A real problem with an observed A-file target
    that cannot be used (dropped) is excluded, with the reason in ``info["excluded"]``."""
    problems = {"twin": TWIN, "real": REAL, "ufga": UFGA_CAL, "ufga_all": UFGA_ALL}[kind]
    keys = sorted({k for ks in problems.values() for k in ks})
    items = load_items(keys)
    ev = Evaluator(items, {it["key"]: _table([]) for it in items})
    info: dict[str, Any] = {}
    if kind == "twin":
        obs = twin_obs(ev, problems, codes)
    else:
        obs, info = real_obs(ev, problems, codes)
        excl = {
            pn: {k: info[k]["dropped_a"] for k in ks if info[k]["dropped_a"]} for pn, ks in problems.items()
        }
        excl = {pn: v for pn, v in excl.items() if v}
        if excl:
            print(
                f"WARNING excluded problems (observed targets outside the simulated days): {excl}", flush=True
            )
            problems = {pn: ks for pn, ks in problems.items() if pn not in excl}
        info["excluded"] = excl
    pb = Problem(ev, Objective(obs, problems))
    return pb, problems, info


def truth_of(pb: Problem) -> np.ndarray:
    items = {it["key"]: it for it in pb.ev.items}
    out = []
    for ks in pb.obj.problems.values():
        th = [items[k]["theta_pub"] for k in ks]
        assert all(t == th[0] for t in th), ks
        out.append(th[0])
    return np.asarray(out)


def free_mask(kind: str, problems: dict[str, list[str]], cs: str) -> np.ndarray:
    """Coefficients calibrated per problem: not the ones the check step measured inert over the box or
    flat over the start region (for the observation set ``cs``)."""
    m = np.ones((len(problems), 6), bool)
    f = out_root() / f"check_{'real' if kind.startswith('ufga') else kind}_{cs}.json"
    if f.exists():
        chk = json.loads(f.read_text())
        by = dict(zip(chk["problems"], chk["identifiability"], strict=True))
        for p, pn in enumerate(problems):
            for i, n in enumerate(PARAMS):
                m[p, i] = not (by[pn][n]["inert"] or by[pn][n].get("flat_start", False))
    return m


def run_method(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    t0 = time.perf_counter()
    tokens = set(a.variant.split("+"))
    codes = CODES_GN if "gn" in tokens else CODES_BASE
    stage1_laipre = "dates" not in tokens
    pb, problems, info = build(a.kind, codes)
    t_build = time.perf_counter()
    sp = space()
    p_n = len(problems)
    truth = truth_of(pb)
    free = free_mask(a.kind, problems, "gn" if "gn" in tokens else "base")
    rng = np.random.default_rng(a.seed)
    rows_prob = np.repeat(np.arange(p_n), N_STARTS)
    th0 = np.concatenate([make_starts(sp, truth[p], rng, "perturb") for p in range(p_n)])
    z0 = to_z(sp, th0)
    r_n = rows_prob.size
    tracker = Tracker(r_n, T_START)
    # coefficients the data carry no information on (check step): fixed at the reference value (the
    # truth of the twin, the published value of the real problems)
    fixed_th = truth[rows_prob].copy()
    start_mode = "perturb"  # twin: around the truth; real: around the published cultivar

    def restart(rows: np.ndarray) -> np.ndarray:
        """New CMA-ES means from the distribution of the starts."""
        return np.concatenate(
            [to_z(sp, make_starts(sp, truth[rows_prob[r]], rng, start_mode)[:1]) for r in rows]
        )

    def theta_of(z: np.ndarray, rows: np.ndarray) -> np.ndarray:
        th = to_theta(sp, z)
        f = free[rows_prob[rows]]
        return np.where(f, th, fixed_th[rows])

    def evaluator(
        view: dict[str, np.ndarray],
    ) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        def ev(z: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return pb.evaluate(theta_of(z, rows), rows_prob[rows], view)

        return ev

    full_view = pb.full
    res: dict[str, Any] = {"method": a.method, "variant": a.variant, "kind": a.kind, "seed": a.seed}
    # ------------------------------------------------ quality targets (defined before the runs)
    if a.kind == "twin":
        q = np.full(r_n, Q_TWIN)
    else:
        q = np.full(r_n, -np.inf)  # real: set after the runs (Q_REAL_FACTOR x the best loss found)
    stage: dict[str, Any] = {}
    if a.method == "staged":
        v1 = pb.obj.view(STAGE1_CODES, lai_pre=stage1_laipre)
        tol1 = np.full(r_n, TOL1_LAIPRE if stage1_laipre else TOL1_DATES)
        s0 = time.perf_counter()
        r1 = cma_rows(
            evaluator(v1),
            z0,
            I_PHEN,
            tracker,
            max_evals=CMA1_EVALS,
            tol=tol1,
            seed=a.seed + 1,
            restart=restart,
        )
        stage["stage1"] = {
            "wall_s": time.perf_counter() - s0,
            "evals": r1["evals"],
            "restarts": r1["restarts"],
            "loss": r1["loss_best"],
            "reached_tol": r1["loss_best"] <= tol1,
        }
        z1 = r1["z_best"].copy()
        z1[:, list(I_GROWTH)] = z0[:, list(I_GROWTH)]  # stage 2 starts from the start values of G2, G3
        z1 = np.where(free[rows_prob], z1, to_z(sp, fixed_th))  # coefficients not calibrated: reference
        th1 = theta_of(z1, np.arange(r_n))
        v2 = pb.obj.view(STAGE2_CODES)
        s0 = time.perf_counter()
        # one trust report per problem, at the coordinate-wise median of its rows' stage-2 starts; its
        # calls are split over the problem's rows
        th_med = np.stack([np.median(th1[rows_prob == p], axis=0) for p in range(p_n)])
        reports = trust_rows(pb, th_med, np.arange(p_n), I_GROWTH, v2)
        tf, tj = trust_calls(len(I_GROWTH))
        tracker.n_fwd += tf / N_STARTS
        tracker.n_jvp += tj / N_STARTS
        tracker.snap()
        from agrijax.calib.trust import gradient_plan

        plans = [gradient_plan(rep, None, derivative_free=()) for rep in reports]
        secant = [~plans[p].use_ad for p in rows_prob]
        stage["trust"] = {
            "wall_s": time.perf_counter() - s0,
            "theta": th_med,
            "plans": [pl.table() for pl in plans],
            "classes": [
                {
                    p: {
                        k: (o["class"], o["level"], o["n_jumps"])
                        for k, o in rep["params"][p]["outputs"].items()
                    }
                    for p in GROWTH
                }
                for rep in reports
            ],
        }
        s0 = time.perf_counter()
        # stage 2: LM with the plan's Jacobian; a problem whose plan has no AD pair at all (every pair
        # jumpy or kinked) runs CMA-ES on G2, G3 instead (derivative-free, robust to the jumps)
        no_ad = np.asarray([not plans[p].use_ad.any() for p in rows_prob])
        z2 = z1.copy()
        lm_r = np.nonzero(~no_ad)[0]
        cma_r = np.nonzero(no_ad)[0]
        r2: dict[str, Any] = {"loss_best": np.full(r_n, np.nan), "accepted": np.zeros(r_n, int)}
        if lm_r.size:
            r2l = lm_rows(
                pb,
                z1[lm_r],
                rows_prob[lm_r],
                I_GROWTH,
                v2,
                [secant[r] for r in lm_r],
                _SubTracker(tracker, lm_r),
                free=free,
                fixed=fixed_th[lm_r],
            )
            z2[lm_r] = r2l["z_best"]
            r2["loss_best"][lm_r] = r2l["loss_best"]
            r2["accepted"][lm_r] = r2l["accepted"]
        if cma_r.size:

            def ev2(z: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                g = cma_r[rows]
                return pb.evaluate(theta_of(z, g), rows_prob[g], v2)

            def restart2(rows: np.ndarray) -> np.ndarray:
                zz = z1[cma_r[rows]].copy()
                zz[:, list(I_GROWTH)] = restart(cma_r[rows])[:, list(I_GROWTH)]
                return zz

            r2c = cma_rows(
                ev2,
                z1[cma_r],
                I_GROWTH,
                _SubTracker(tracker, cma_r),
                max_evals=CMA2_EVALS,
                tol=np.full(cma_r.size, LM_TOL if a.kind == "twin" else 0.0),
                seed=a.seed + 5,
                restart=restart2,
            )
            z2[cma_r] = r2c["z_best"]
            r2["loss_best"][cma_r] = r2c["loss_best"]
        _, full2 = pb.evaluate(theta_of(z2, np.arange(r_n)), rows_prob, full_view)  # not counted (known)
        stage["stage2"] = {
            "wall_s": time.perf_counter() - s0,
            "loss": r2["loss_best"],
            "full": full2,
            "accepted": r2["accepted"],
            "method_rows": np.where(no_ad, "cma", "lm"),
        }
        # stage 3: joint refinement of every free coefficient on the original objective, for the rows
        # not converged (twin: original objective > LM_TOL; real: every row): LM with the stage
        # coefficients' columns as central secants (derivative-free by default) and the growth columns
        # by the plan
        s0 = time.perf_counter()
        todo = np.nonzero(full2 > LM_TOL)[0] if a.kind == "twin" else np.arange(r_n)
        z_final = z2.copy()
        if todo.size:
            sec3 = []
            for r in todo:
                m = np.ones((len(pb.obj.problems[pb.obj.pnames[rows_prob[r]]]), 6), bool)
                m[:, list(I_GROWTH)] = secant[r]
                sec3.append(m)
            r3 = lm_rows(
                pb,
                z2[todo],
                rows_prob[todo],
                range(6),
                full_view,
                sec3,
                _SubTracker(tracker, todo),
                free=free,
                fixed=fixed_th[todo],
                iters=LM3_ITERS,
            )
            z_final[todo] = r3["z_best"]
        stage["stage3"] = {"wall_s": time.perf_counter() - s0, "rows": todo}
    elif a.method == "cma":
        tolc = q.copy() * 1e-3 if a.kind == "twin" else np.full(r_n, 0.0)
        r = cma_rows(
            evaluator(full_view),
            z0,
            range(6),
            tracker,
            max_evals=CMA_ALL_EVALS,
            tol=tolc,
            seed=a.seed + 2,
            restart=restart,
        )
        stage["cma"] = {
            "evals": r["evals"],
            "loss": r["loss_best"],
            "popsize": r["popsize"],
            "restarts": r["restarts"],
        }
        z_final = r["z_best"]
    elif a.method == "random":
        lo_th = sp.lower + MARGIN * sp.width
        hi_th = sp.upper - MARGIN * sp.width
        if a.kind == "twin":  # the box of the twin starts (favourable to random search)
            lo_th = np.maximum(lo_th, truth * (1 - PERTURB))
            hi_th = np.minimum(hi_th, truth * (1 + PERTURB))
        lo_z = to_z(sp, np.repeat(np.minimum(lo_th, hi_th), N_STARTS, axis=0))
        hi_z = to_z(sp, np.repeat(np.maximum(lo_th, hi_th), N_STARTS, axis=0))
        r = random_rows(
            evaluator(full_view),
            lo_z,
            hi_z,
            tracker,
            max_evals=RS_EVALS,
            tol=q.copy() * 1e-3 if a.kind == "twin" else np.full(r_n, 0.0),
            seed=a.seed + 3,
        )
        stage["random"] = {"evals": r["evals"], "loss": r["loss_best"]}
        z_final = r["z_best"]
    else:
        raise ValueError(a.method)
    t_end = time.perf_counter()
    th_final = theta_of(z_final, np.arange(r_n))
    # the original objective of the returned points (one more forward, not counted)
    _, full_final = pb.evaluate(th_final, rows_prob, full_view)
    # the published cultivar on the same objective (the reference solution of the real problems)
    _, full_pub = pb.evaluate(truth, np.arange(p_n), full_view)
    # the coefficients as a .CUL row holds them (printed precision) and their objective
    th_written = written_theta(th_final)
    _, full_written = pb.evaluate(th_written, rows_prob, full_view)
    res.update(
        {
            "env": env_info(),
            "problems": problems,
            "info": info,
            "scales": pb.obj.scale_info,
            "truth": truth,
            "free": free,
            "rows_prob": rows_prob,
            "theta_start": th0,
            "theta_final": th_final,
            "loss_final": full_final,
            "loss_published": full_pub,
            "theta_written": th_written,
            "loss_written": full_written,
            "best_full": tracker.best_full,
            "best_full_theta": theta_of(tracker.best_full_z, np.arange(r_n)),
            "n_fwd": tracker.n_fwd,
            "n_jvp": tracker.n_jvp,
            "hist": tracker.hist,
            "stage": stage,
            "timing": {
                "process_s": t_end - T_START,
                "build_s": t_build - t0,
                "optimise_s": t_end - t_build,
                **pb.ev.stats(),
            },
            "n_treatments": {p: len(ks) for p, ks in problems.items()},
            "q_twin": Q_TWIN if a.kind == "twin" else None,
        }
    )
    name = f"{a.kind}_{a.method}_{a.variant}_s{a.seed}.json"
    dump(out_root() / name, res)
    print("wrote", name, f"process {t_end - T_START:.1f} s", json.dumps(pb.ev.stats()), flush=True)


def launch(kind: str, method: str, variant: str, seed: int) -> int:
    """One method in a fresh process (compile counted, no compile cache)."""
    env = dict(os.environ)
    env.pop("AGRI_JAX_CHECK", None)
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    cores = len(os.sched_getaffinity(0))
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "method",
        "--kind",
        kind,
        "--method",
        method,
        "--variant",
        variant,
        "--seed",
        str(seed),
    ]
    print("launch", " ".join(cmd[2:]), flush=True)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, env=env, check=False)
    print(f"  done rc={p.returncode} wall {time.perf_counter() - t0:.1f} s", flush=True)
    return p.returncode


def step_runs(a: argparse.Namespace) -> None:
    rc = 0
    for m in a.methods.split(","):
        for v in a.variants.split(","):
            if m != "staged" and "dates" in v.split("+"):
                continue
            rc |= launch(a.kind, m, v, a.seed)
    if rc:
        raise SystemExit(rc)


# ============================================================================ check
def step_check(a: argparse.Namespace) -> None:
    """Published cultivars against DSSAT (all 58 runs), and per twin / real problem: which coefficients
    move the original objective at all over their box (inert = not identifiable), the zero-loss plateau
    around the truth, and the trust report of every coefficient at the truth (information: the stage
    coefficients are derivative-free by default)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    with open(out_root() / "inputs.pkl", "rb") as fh:
        items = pickle.load(fh)
    ev = Evaluator(items, {it["key"]: probe_entries(it) for it in items})
    th = np.asarray([it["theta_pub"] for it in items])
    t0 = time.perf_counter()
    y, _ = ev.run(th, np.asarray([it["key"] for it in items]))
    t_pub = time.perf_counter() - t0
    runs = []
    for s, it in enumerate(items):
        d = dict(zip(PROBE_NAMES, y[s].tolist(), strict=False))
        n = it["n_days"]
        yld = d["gwad_last_real"]
        runs.append(
            {
                "key": it["key"],
                "cultivar": it["cultivar"],
                "group": f"nl{it['nl']}_{it['mesev']}",
                "n_days": n,
                "n_padded": it["n_padded"],
                "hwam": it["summary"]["HWAM"],
                "yield_last_real": yld,
                "yield_at_maturity": d["gwad_mat"],
                "yield_last_pad": d["gwad_last_pad"],
                "yield_rel_err": abs(yld - it["summary"]["HWAM"]) / max(it["summary"]["HWAM"], 1e-9),
                "maturity_index": d["mdate"],
                "maturity_is_last_real_day": int(d["mdate"]) == n - 1,
                "cwad_mat": d["cwad_mat"],
                "cwam": it["summary"]["CWAM"],
                "g_ad_mat": d["g_ad_mat"],
                "h_am": it["summary"]["H#AM"],
            }
        )
    out: dict[str, Any] = {"env": env_info(), "runs": runs, "published_forward_s": t_pub, "stats": ev.stats()}
    print(
        json.dumps({r["key"]: [round(r["yield_rel_err"], 4), r["maturity_is_last_real_day"]] for r in runs}),
        flush=True,
    )
    dump(out_root() / "check_pub.json", out)
    for kind, cs in (("twin", "base"), ("twin", "gn"), ("real", "base"), ("real", "gn")):
        pb, problems, info = build(kind, CODES_GN if cs == "gn" else CODES_BASE)
        truth = truth_of(pb)
        p_n = len(problems)
        sp = space()
        res: dict[str, Any] = {"problems": problems, "info": info, "identifiability": [], "plateau": []}
        # one coefficient at a time around the truth / published values: full-box scan (41 points: inert
        # = no effect anywhere), the start region +-PERTURB (101 points: flat = no information in the
        # data there, the coefficient is then fixed), the +-5 % plateau (401 points: the resolution)
        scans = (("box", 41, None), ("start", 101, PERTURB), ("plateau", 401, 0.05))
        rows_of: dict[str, list[dict[str, Any]]] = {lab: [] for lab, _, _ in scans}
        for label, npts, rel in scans:
            pts, prob = [], []
            for p in range(p_n):
                for i in range(6):
                    grid = (
                        np.linspace(sp.lower[i], sp.upper[i], npts)
                        if rel is None
                        else np.clip(
                            truth[p, i] * (1 + np.linspace(-rel, rel, npts)), sp.lower[i], sp.upper[i]
                        )
                    )
                    for gv in grid:
                        t = truth[p].copy()
                        t[i] = gv
                        pts.append(t)
                        prob.append(p)
            _, full = pb.evaluate(np.asarray(pts), np.asarray(prob), pb.full)
            full = full.reshape(p_n, 6, npts)
            for p in range(p_n):
                row = {}
                for i, n in enumerate(PARAMS):
                    f = full[p, i]
                    c = npts // 2
                    if label == "box":
                        inert = float(np.ptp(f)) <= 1e-12 * max(1.0, float(np.max(np.abs(f))))
                        row[n] = {
                            "inert": bool(inert),
                            "loss_max": float(np.max(f)),
                            "loss_min": float(np.min(f)),
                        }
                    elif label == "start":
                        row[n] = {"flat_start": bool(np.all(f == f[c])), "start_loss_max": float(np.max(f))}
                    else:  # the interval around the truth on which the loss does not change at all
                        lo_ = c
                        while lo_ > 0 and f[lo_ - 1] == f[c]:
                            lo_ -= 1
                        hi_ = c
                        while hi_ < npts - 1 and f[hi_ + 1] == f[c]:
                            hi_ += 1
                        row[n] = {
                            "zero_rel_halfwidth": float((hi_ - lo_) / (npts - 1) * rel),  # type: ignore[operator]
                            "loss_at_truth": float(f[c]),
                            "hits_edge": bool(lo_ == 0 or hi_ == npts - 1),
                        }
                rows_of[label].append(row)
        res["identifiability"] = [
            {n: {**b[n], **s_[n]} for n in PARAMS}
            for b, s_ in zip(rows_of["box"], rows_of["start"], strict=True)
        ]
        res["plateau"] = rows_of["plateau"]
        # trust of every coefficient at the truth, per treatment, original objective
        reports = trust_rows(pb, truth, np.arange(p_n), range(6), pb.full)
        res["trust_at_truth"] = [
            {
                n: {k: (o["class"], o["level"], o["n_jumps"]) for k, o in rep["params"][n]["outputs"].items()}
                for n in PARAMS
            }
            for rep in reports
        ]
        from agrijax.calib.ceres import ceres_gradient_plan

        res["plan_at_truth"] = [ceres_gradient_plan(rep).table() for rep in reports]
        res["stats"] = pb.ev.stats()
        dump(out_root() / f"check_{kind}_{cs}.json", res)
        for p, pn in enumerate(problems):
            print(
                kind,
                cs,
                pn,
                {
                    n: (
                        "INERT"
                        if res["identifiability"][p][n]["inert"]
                        else "FLAT"
                        if res["identifiability"][p][n]["flat_start"]
                        else round(res["plateau"][p][n]["zero_rel_halfwidth"], 4)
                    )
                    for n in PARAMS
                },
                flush=True,
            )


# ============================================================================ round trip
def _filex_with_cultivar(text: str, old: str, new: str) -> str:
    out, sec = [], ""
    for ln in text.splitlines():
        if ln.startswith("*"):
            sec = ln
        if sec.startswith("*CULTIVARS") and not ln.startswith(("*", "@", "!")) and f" {old} " in ln:
            ln = ln.replace(f" {old} ", f" {new} ", 1)
        out.append(ln)
    return "\n".join(out) + "\n"


def cul_source() -> Path:
    from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths

    return dscsm_paths(DSSAT_ENGINE)[1] / "Genotype" / "MZCER048.CUL"


def written_theta(theta: np.ndarray) -> np.ndarray:
    """``theta [R, 6]`` rounded as :func:`agrijax.io.dssat.cultivar_write.cul_line` prints it into
    ``MZCER048.CUL`` (what DSSAT reads back)."""
    from agrijax.io.dssat.cultivar_write import _fields, cul_line, cul_precision

    dec = cul_precision(cul_source())
    out = np.zeros_like(theta)
    for r in range(theta.shape[0]):
        vals = dict(zip(PARAMS, theta[r].tolist(), strict=True))
        f = _fields(cul_line("AJ9999", "x", vals, ecotype="IB0001", decimals=dec))
        out[r] = [float(f[n]) for n in PARAMS]
    return out


def step_roundtrip(a: argparse.Namespace) -> None:
    """The best calibrated cultivar of every real problem (lowest original objective over the staged
    runs' starts) -> a copy of MZCER048.CUL (:func:`agrijax.io.dssat.cultivar_write.write_cultivar`)
    -> dscsm048 (nitrogen off) on the problem's treatments; compared with Agri-JAX at the written
    (rounded) values and with the published cultivar's DSSAT run."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import test_ceres_dssat as m2

    from agrijax.io.dssat import read_summary
    from agrijax.io.dssat.cultivar_write import write_cultivar
    from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths, run_dscsm

    src = dscsm_paths(DSSAT_ENGINE)[1] / "Genotype" / "MZCER048.CUL"
    res = json.loads((out_root() / f"real_{a.method}_{a.variant}_s{a.seed}.json").read_text())
    rows_prob = np.asarray(res["rows_prob"])
    # selection on the objective of the WRITTEN coefficients (what DSSAT will read)
    loss = np.asarray(res["loss_written"])
    theta = np.asarray(res["theta_written"])
    problems: dict[str, list[str]] = res["problems"]
    items = {it["key"]: it for it in load_items(sorted({k for ks in problems.values() for k in ks}))}
    work = Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / f"rt_{a.method[:3]}"
    out: dict[str, Any] = {"source_cul": str(src), "problems": {}}
    wtheta: dict[str, list[float]] = {}
    for p, pn in enumerate(problems):
        rows = np.nonzero(rows_prob == p)[0]
        best = rows[np.argmin(loss[rows])]
        vals = dict(zip(PARAMS, theta[best].tolist(), strict=True))
        ks = problems[pn]
        pub = items[ks[0]]["cultivar"]
        cid = f"AJ{p + 1:04d}"
        dest = out_root() / f"roundtrip_{a.method}" / pn.replace("/", "_") / "MZCER048.CUL"
        dest.parent.mkdir(parents=True, exist_ok=True)
        w = write_cultivar(
            src,
            dest,
            cid,
            f"AgriJAX {pub}",
            {k: vals[k] for k in ("P1", "P2", "P5", "G2", "G3", "PHINT")},
            base=pub,
            overwrite=True,
        )
        wtheta[pn] = [w.written[n] for n in PARAMS]
        trts = []
        for k in ks:
            it = items[k]
            for label, cul, cfile in (("calibrated", cid, dest), ("published", pub, None)):
                dd = work / f"{k}_{label[:3]}"
                if dd.exists():
                    shutil.rmtree(dd)
                dd.mkdir(parents=True)
                for f in m2.MAIZE.glob(it["exp"] + ".MZ*"):
                    shutil.copy2(f, dd / f.name)
                x = dd / f"{it['exp']}.MZX"
                txt = m2._nitrogen_off(x.read_text(errors="replace"))
                if cfile is not None:
                    txt = _filex_with_cultivar(txt, pub, cid)
                    shutil.copy2(cfile, dd / "MZCER048.CUL")
                x.write_text(txt)
                batch = "$BATCH(MAIZE)\n!\n@FILEX" + " " * 88 + "TRTNO     RP     SQ     OP     CO\n"
                batch += f"{it['exp']}.MZX".ljust(92) + f"{it['trno']:7d}      1      0      0      0\n"
                (dd / "DSSBatch.v48").write_text(batch)
                run_dscsm(
                    dd,
                    dd / "out",
                    run_mode="B",
                    experiment_file="DSSBatch.v48",
                    extra_files=sorted(m2.WEATHER.glob(it["exp"][:4] + "*.WTH")),
                    keep_files=("*.OUT", "DSSAT48.INP"),
                )
                inp = (dd / "out" / "DSSAT48.INP").read_text(errors="replace")
                row = read_summary(dd / "out" / "Summary.OUT").iloc[0]
                trts.append(
                    {
                        "key": k,
                        "label": label,
                        "cultivar_in_inp": cul in inp,
                        **{c: float(row[c]) for c in ("HWAM", "CWAM", "H#AM", "ADAT", "MDAT")},
                    }
                )
        out["problems"][pn] = {
            "row": int(best),
            "loss_written": float(loss[best]),
            "loss_unrounded_of_that_row": float(res["loss_final"][best]),
            "cultivar_id": cid,
            "line": w.line,
            "written": w.written,
            "rounded": w.rounded,
            "out_of_range": w.out_of_range,
            "dssat": trts,
        }
    # Agri-JAX at the written values
    pb, _, _ = build("real", CODES_GN)
    pb.ev.set_entries({k: probe_entries(it) for k, it in ((it["key"], it) for it in pb.ev.items)})
    keys, ths = [], []
    for pn, ks in problems.items():
        for k in ks:
            keys.append(k)
            ths.append(wtheta[pn])
    y, _ = pb.ev.run(np.asarray(ths), np.asarray(keys))
    for s, k in enumerate(keys):
        d = dict(zip(PROBE_NAMES, y[s].tolist(), strict=False))
        pn = next(p for p, ks in problems.items() if k in ks)
        it = items[k]
        rec = next(t for t in out["problems"][pn]["dssat"] if t["key"] == k and t["label"] == "calibrated")

        def ydoy(i: float, it: dict[str, Any] = it) -> int:
            d0 = date(int(it["days"][0]) // 1000, 1, 1) + timedelta(
                days=int(it["days"][0]) % 1000 - 1 + int(i)
            )
            return d0.year * 1000 + d0.timetuple().tm_yday

        rec["agrijax"] = {
            "HWAM": d["gwad_mat"],
            "CWAM": d["cwad_mat"],
            "H#AM": d["g_ad_mat"],
            "ADAT": ydoy(d["adate"]),
            "MDAT": ydoy(d["mdate"]),
        }
        rec["yield_rel_diff"] = abs(d["gwad_mat"] - rec["HWAM"]) / max(rec["HWAM"], 1e-9)
        rec["dates_equal"] = rec["agrijax"]["ADAT"] == int(rec["ADAT"]) and rec["agrijax"]["MDAT"] == int(
            rec["MDAT"]
        )
    dump(out_root() / f"roundtrip_{a.method}.json", out)
    print(json.dumps(out, default=_json_default, indent=1)[:6000], flush=True)


# ============================================================================ diagnostics
def step_diag(a: argparse.Namespace) -> None:
    """The outputs at the published cultivar with P2 replaced by 0 and by tiny positive values (the
    logit round trip of P2 = 0 gives 2e-9), for the treatments whose published P2 is 0."""
    import jax

    jax.config.update("jax_enable_x64", True)
    with open(out_root() / "inputs.pkl", "rb") as fh:
        items = [it for it in pickle.load(fh) if it["theta_pub"][1] == 0.0 or it["key"] == "UFGA8201_t02"]
    ev = Evaluator(items, {it["key"]: probe_entries(it) for it in items})
    vals = (0.0, 1e-12, 2e-9, 1e-6, 1e-3, 1e-2)
    th, keys = [], []
    for it in items:
        for v in vals:
            t = list(it["theta_pub"])
            t[1] = v
            th.append(t)
            keys.append(it["key"])
    y, _ = ev.run(np.asarray(th), np.asarray(keys))
    out = []
    for s_, (k, t) in enumerate(zip(keys, th, strict=True)):
        d = dict(zip(PROBE_NAMES, y[s_].tolist(), strict=False))
        out.append(
            {
                "key": k,
                "P2": t[1],
                **{n: d[n] for n in ("edate", "adate", "mdate", "gwad_mat", "cwad_mat", "lai_mat")},
            }
        )
        print(json.dumps(out[-1]), flush=True)
    dump(out_root() / "diag_p2.json", out)


# ============================================================================ tables
def _fmt(x: float, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    if abs(x) >= 1e4 or (abs(x) < 1e-2 and x != 0):
        return f"{x:.{d}e}"
    return f"{x:.{d + 1}g}"


def dssat_cost(r: dict[str, Any], upto: float | None) -> dict[str, float]:
    """DSSAT + the same derivative-free trajectory: every candidate one dscsm048 invocation over the
    problem's treatments (D4-1 rates: :data:`DSSAT_INVOCATION_S` + K x :data:`DSSAT_SEASON_S`), (a)
    serially on one core, (b) each call batch spread over a 192-core node (makespan
    ceil(invocations / 192) x the mean invocation cost + start-up). ``upto``: stop at that wall time
    of the JAX run (equal quality)."""
    k_of = np.asarray([r["n_treatments"][pn] for pn in r["problems"]])[np.asarray(r["rows_prob"])]
    prev = np.zeros(len(r["rows_prob"]))
    serial = node = serial_b = node_b = 0.0
    inv_total = 0.0
    seasons = 0.0
    for t, f, _, _ in r["hist"]:
        if upto is not None and t > upto + 1e-9:
            break
        f = np.asarray(f)
        d = f - prev
        prev = f
        if d.sum() <= 0:
            continue
        cost = DSSAT_INVOCATION_S + k_of * DSSAT_SEASON_S
        serial += float(np.sum(d * cost))
        inv = float(d.sum())
        inv_total += inv
        seasons += float(np.sum(d * k_of))
        node += DSSAT_NODE_START_S + math.ceil(inv / DSSAT_NODE_CORES) * float(np.sum(d * cost) / inv)
        # batch basis (D4-1 batch rates: 18.2 ms per season, no per-candidate start-up)
        sz = float(np.sum(d * k_of))
        serial_b += sz * DSSAT_SEASON_S
        node_b += DSSAT_NODE_START_S + math.ceil(sz / DSSAT_NODE_CORES) * DSSAT_SEASON_S
    return {
        "serial_s": serial,
        "node_s": node,
        "serial_batch_s": serial_b,
        "node_batch_s": node_b,
        "invocations": inv_total,
        "seasons": seasons,
    }


def summarise(r: dict[str, Any], q: np.ndarray, plateau: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows_prob = np.asarray(r["rows_prob"])
    truth = np.asarray(r["truth"])
    th = np.asarray(r["theta_final"])
    loss = np.asarray(r["loss_final"])
    free = np.asarray(r["free"])
    names = list(r["problems"])
    ok = loss <= q
    tr = Tracker(len(rows_prob), 0.0)
    tr.hist = [tuple(h) for h in r["hist"]]  # type: ignore[misc]
    ct = tr.calls_to(q)
    per_problem = []
    for p, pn in enumerate(names):
        rows = rows_prob == p
        rel = np.abs(th[rows] - truth[p]) / np.maximum(np.abs(truth[p]), 1e-12)
        wid = np.abs(th[rows] - truth[p]) / space().width
        tol = np.asarray(
            [max(REC_TOL, plateau.get(pn, {}).get(n, {}).get("zero_rel_halfwidth", 0.0)) for n in PARAMS]
        )
        rec = rel <= tol[None, :]
        succ = ok[rows]
        per_problem.append(
            {
                "problem": pn,
                "n_trt": r["n_treatments"][pn],
                "success": int(succ.sum()),
                "n_rows": int(rows.sum()),
                "loss_median": float(np.median(loss[rows])),
                "rel_err_median": np.median(rel, axis=0).tolist(),
                "width_err_median": np.median(wid, axis=0).tolist(),
                "recovered": rec.sum(axis=0).tolist(),
                "free": free[p].tolist(),
                "fwd_median": float(np.median(np.asarray(r["n_fwd"])[rows])),
                "jvp_median": float(np.median(np.asarray(r["n_jvp"])[rows])),
                "fwd_to_q_median": float(np.median(ct["fwd"][rows][ct["fwd"][rows] >= 0]))
                if (ct["fwd"][rows] >= 0).any()
                else None,
                "jvp_to_q_median": float(np.median(ct["jvp"][rows][ct["fwd"][rows] >= 0]))
                if (ct["fwd"][rows] >= 0).any()
                else None,
            }
        )
    reached = ct["fwd"] >= 0
    t_last = float(np.nanmax(ct["wall_s"])) if reached.any() else None
    return {
        "per_problem": per_problem,
        "success": int(ok.sum()),
        "reached_q_any_point": int(reached.sum()),
        "n_rows": int(ok.size),
        "wall_to_last_q": t_last,
        "wall_to_half": float(np.sort(ct["wall_s"][reached])[len(ok) // 2 - 1])
        if reached.sum() >= len(ok) // 2
        else None,
        "timing": r["timing"],
        "dssat_full": dssat_cost(r, None),
        "dssat_to_last_q": dssat_cost(r, t_last) if t_last is not None else None,
    }


def step_table3(a: argparse.Namespace) -> None:
    """Twin: the methods ``--methods`` x ``--variants`` over the seeds ``--seeds`` (quality Q_TWIN)."""
    root = out_root()
    seeds = [int(x) for x in a.seeds.split(",")]
    out: dict[str, Any] = {}
    lines = []
    for m in a.methods.split(","):
        for v in a.variants.split(","):
            files = [root / f"twin_{m}_{v}_s{sd}.json" for sd in seeds]
            files = [f for f in files if f.exists()]
            if not files:
                continue
            chk = json.loads((root / f"check_twin_{v}.json").read_text())
            plateau = dict(zip(chk["problems"], chk["plateau"], strict=True))
            per = []
            for f in files:
                r = json.loads(f.read_text())
                sm = summarise(r, np.full(len(r["rows_prob"]), Q_TWIN), plateau)
                sm["fwd_to_q_all"] = [pp["fwd_to_q_median"] for pp in sm["per_problem"]]
                per.append(sm)
            succ = [p_["success"] for p_ in per]
            key = f"{m}_{v}"
            out[key] = per
            probs = [pp["problem"] for pp in per[0]["per_problem"]]
            ps = {pn: sum(p_["per_problem"][i]["success"] for p_ in per) for i, pn in enumerate(probs)}
            rec = {
                n: int(
                    sum(
                        np.sum([pp["recovered"][j] for pp in p_["per_problem"] if pp["free"][j]])
                        for p_ in per
                    )
                )
                for j, n in enumerate(PARAMS)
            }
            nfree = {
                n: int(sum(np.sum([pp["n_rows"] for pp in p_["per_problem"] if pp["free"][j]]) for p_ in per))
                for j, n in enumerate(PARAMS)
            }
            lines.append(
                f"| {key} | {sum(succ)}/{sum(p_['n_rows'] for p_ in per)} ({', '.join(map(str, succ))}) | "
                f"{np.median([p_['timing']['process_s'] for p_ in per]):.0f} s | "
                f"{np.median([p_['timing']['compile_s'] for p_ in per]):.0f} s | "
                f"{np.median([p_['timing']['call_s'] for p_ in per]):.0f} s | "
                f"{_fmt(float(np.median([p_['wall_to_last_q'] or np.nan for p_ in per])))} s | "
                f"{np.median([p_['timing']['seasons']['fwd'] for p_ in per]):.0f} / "
                f"{np.median([p_['timing']['seasons']['jvp'] for p_ in per]):.0f} | "
                + (
                    f"{np.median([(p_['dssat_to_last_q'] or p_['dssat_full'])['serial_s'] for p_ in per]) / 3600:.2f} h / "  # noqa: E501
                    f"{np.median([(p_['dssat_to_last_q'] or p_['dssat_full'])['node_s'] for p_ in per]):.0f} s"  # noqa: E501
                    if m != "staged"
                    else "n/a"
                )
                + " | "
                + " ".join(f"{n} {rec[n]}/{nfree[n]}" for n in PARAMS if nfree[n])
                + " |"
            )
            lines.append(
                "|   per problem success | " + ", ".join(f"{pn} {ps[pn]}" for pn in probs) + " |||||||| "
            )
    head = (
        "| method | success (per seed) | process wall | compile | call time | wall to last row at Q | "
        "seasons fwd / jvp | DSSAT model 1 core / node | recovered (free rows) |\n|---|---|---|---|---|---|---|---|---|"  # noqa: E501
    )
    txt = head + "\n" + "\n".join(lines)
    print(txt)
    (root / "tables_twin3.md").write_text(txt)
    dump(root / "summary_twin3.json", out)


def step_real_table(a: argparse.Namespace) -> None:
    """Real observations: per problem the best row of each method against the published cultivar."""
    root = out_root()
    lines = [
        "| problem | method | loss published | loss best, written coefficients (median of 8) | unrounded | "
        + " | ".join(f"{n} pub -> cal" for n in PARAMS)
        + " |",
        "|" + "---|" * (5 + len(PARAMS)),
    ]
    for m in a.methods.split(","):
        f = root / f"real_{m}_{a.variant}_s{a.seed}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rp = np.asarray(r["rows_prob"])
        loss = np.asarray(r["loss_written"])
        l_un = np.asarray(r["loss_final"])
        th = np.asarray(r["theta_written"])
        free = np.asarray(r["free"])
        if r["info"].get("excluded"):
            lines.append(f"| excluded | {m} | {r['info']['excluded']} |||||||||")
        for p, pn in enumerate(r["problems"]):
            rows = np.nonzero(rp == p)[0]
            b = rows[np.argmin(loss[rows])]
            cells = [
                f"{r['truth'][p][i]:.4g} -> {th[b, i]:.4g}"
                if free[p, i]
                else f"{r['truth'][p][i]:.4g} (fixed)"
                for i in range(6)
            ]
            lines.append(
                f"| {pn} | {m} | {_fmt(r['loss_published'][p])} | {_fmt(float(loss[b]))} "
                f"({_fmt(float(np.median(loss[rows])))}) | {_fmt(float(l_un[b]))} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append(
            f"| timing {m} | process {r['timing']['process_s']:.0f} s, compile {r['timing']['compile_s']:.0f} s, "  # noqa: E501
            f"calls {r['timing']['call_s']:.0f} s |||||||||| "
        )
    txt = "\n".join(lines)
    print(txt)
    (root / f"tables_real_{a.variant}.md").write_text(txt)


def step_budget(a: argparse.Namespace) -> None:
    """Twin: success (original objective <= Q_TWIN reached) against the per-row call budget
    (forward + JVP_COST x JVP) and against wall clock, per method and seed."""
    root = out_root()
    seeds = [int(x) for x in a.seeds.split(",")]
    budgets = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000]
    out: dict[str, Any] = {"budgets": budgets, "jvp_cost": JVP_COST, "curves": {}, "wall": {}}
    for m in a.methods.split(","):
        for v in a.variants.split(","):
            key = f"{m}_{v}"
            cur, wall = [], []
            for sd in seeds:
                r = json.loads((root / f"twin_{m}_{v}_s{sd}.json").read_text())
                n = len(r["rows_prob"])
                cost = np.full(n, np.inf)
                t_hit = np.full(n, np.inf)
                for t, f, g, b in r["hist"]:
                    hit = (np.asarray(b) <= Q_TWIN) & ~np.isfinite(cost)
                    cost[hit] = (np.asarray(f) + JVP_COST * np.asarray(g))[hit]
                    t_hit[hit] = t
                cur.append([int(np.sum(cost <= bb)) for bb in budgets])
                wall.append(sorted(t_hit[np.isfinite(t_hit)].tolist()))
            out["curves"][key] = cur
            out["wall"][key] = wall
    lines = ["| method | " + " | ".join(str(b) for b in budgets) + " |", "|" + "---|" * (1 + len(budgets))]
    for key, cur in out["curves"].items():
        tot = np.sum(np.asarray(cur), axis=0)
        lines.append(f"| {key} | " + " | ".join(str(int(x)) for x in tot) + " |")
    # wall-clock view: when does each method reach the other's final success count (per seed)
    keys = list(out["wall"])
    for k1 in keys:
        for k2 in keys:
            if k1 == k2 or k1.split("_")[1] != k2.split("_")[1]:
                continue
            tt = []
            for w1, w2 in zip(out["wall"][k1], out["wall"][k2], strict=True):
                need = len(w2)
                tt.append(w1[need - 1] if len(w1) >= need else None)
            lines.append(f"wall until {k1} reaches the success count of {k2} (per seed): {tt}")
    txt = "\n".join(lines)
    print(txt)
    (root / "tables_budget.md").write_text(txt)
    dump(root / "budget.json", out)


def per_code(pb: Problem, theta: np.ndarray, pidx: int) -> dict[str, dict[str, float]]:
    """Per treatment and observed code: the mean squared normalised residual of ``theta`` and the
    simulated / observed values (dates as day indices)."""
    keys = pb.obj.problems[pb.obj.pnames[pidx]]
    y, _ = pb.ev.run(np.repeat(theta[None], len(keys), axis=0), np.asarray(keys))
    out: dict[str, dict[str, Any]] = {}
    for s_, k in enumerate(keys):
        cc = pb.obj.codes[k]
        r_ = pb.obj.residuals(y[s_ : s_ + 1], np.asarray([k]))[0]
        d: dict[str, Any] = {}
        for c in sorted({c for c in cc if c}):
            m = cc == c
            d[c] = {
                "loss": float(np.sum(r_[m] ** 2)),
                "sim": y[s_, m].tolist(),
                "obs": pb.obj.obs.obs[k][m].tolist(),
            }
        out[k] = d
    return out


def step_ident(a: argparse.Namespace) -> None:
    """Identifiability case on UFGA8201 t02 / t04 / t06, forward only:
    G2 x G3 loss contours (a conditional slice: the other coefficients at the calibrated best, written
    values) of the objective without and with grain number, and the simulated grain number; the points
    of the real-data calibration and of the ``ufga`` recalibrations (base / gn, 3 seeds, t06 held out)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    root = out_root()
    pb, _, _ = build("ufga_all", CODES_GN)
    base_v = pb.obj.view(CODES_BASE)
    gn_v = pb.full
    pts: dict[str, Any] = {}
    for m in ("staged", "cma"):
        r = json.loads((root / f"real_{m}_base_s0.json").read_text())
        pi = list(r["problems"]).index("UFGA8201/IB0035")
        rows = np.nonzero(np.asarray(r["rows_prob"]) == pi)[0]
        th = np.asarray(r["theta_written"])[rows]
        lw = np.asarray(r["loss_written"])[rows]
        pts[f"real_{m}"] = {"theta": th, "loss": lw, "best": int(np.argmin(lw))}
        pts["published"] = np.asarray(r["truth"][pi])
    cb = pts["real_cma"]
    center = cb["theta"][cb["best"]]
    g2 = np.linspace(CERES_BOX["G2"][0], CERES_BOX["G2"][1], a.grid)
    g3 = np.linspace(CERES_BOX["G3"][0], CERES_BOX["G3"][1], a.grid)
    th = np.repeat(center[None], a.grid * a.grid, axis=0)
    gg2, gg3 = np.meshgrid(g2, g3, indexing="ij")
    th[:, PARAMS.index("G2")] = gg2.ravel()
    th[:, PARAMS.index("G3")] = gg3.ravel()
    keys = pb.obj.problems["UFGA8201/IB0035"]
    cand, kk = pb.obj.layout(np.zeros(th.shape[0], int))
    y, _ = pb.ev.run(th[cand], kk)
    lb = pb.obj.per_candidate(np.sum(pb.obj.residuals(y, kk, base_v) ** 2, axis=1), cand, th.shape[0])
    lg = pb.obj.per_candidate(np.sum(pb.obj.residuals(y, kk, gn_v) ** 2, axis=1), cand, th.shape[0])
    gn_sim = {}
    for k in keys:
        j = int(np.nonzero(pb.obj.codes[k] == "H#AM")[0][0])
        sel = kk == k
        gn_sim[k] = {"sim": y[sel, j].reshape(a.grid, a.grid), "obs": float(pb.obj.obs.obs[k][j])}
    res: dict[str, Any] = {
        "center": center,
        "g2": g2,
        "g3": g3,
        "loss_base": lb.reshape(a.grid, a.grid),
        "loss_gn": lg.reshape(a.grid, a.grid),
        "grain_number": gn_sim,
        "points": pts,
        "scales": pb.obj.scale_info,
    }
    # the recalibrations on t02 / t04 (t06 held out), base and gn, per seed
    recal: dict[str, Any] = {}
    for v in ("base", "gn"):
        for sd in (0, 1, 2):
            f = root / f"ufga_cma_{v}_s{sd}.json"
            if not f.exists():
                continue
            r = json.loads(f.read_text())
            thw = np.asarray(r["theta_written"])
            lw = np.asarray(r["loss_written"])
            b = int(np.argmin(lw))
            recal[f"{v}_s{sd}"] = {
                "theta": thw,
                "loss_cal": lw,
                "best": b,
                "per_code_best": per_code(pb, thw[b], 0),
                "timing": r["timing"],
            }
    res["recal"] = recal
    res["per_code_published"] = per_code(pb, pts["published"], 0)
    res["per_code_real_cma_best"] = per_code(pb, center, 0)
    dump(root / "ident.json", res)
    print(
        "wrote ident.json",
        {k: (v["theta"][v["best"]].tolist(), float(v["loss_cal"][v["best"]])) for k, v in recal.items()},
    )


def step_table(a: argparse.Namespace) -> None:
    """Markdown tables of every result file of ``--kind`` (to stdout and ``tables_<kind>.md``)."""
    root = out_root()
    files = sorted(root.glob(f"{a.kind}_*_s{a.seed}.json"))
    res = {f.stem: json.loads(f.read_text()) for f in files}
    lines: list[str] = []
    summ: dict[str, Any] = {}
    if a.kind == "real":  # quality: Q_REAL_FACTOR x the lowest loss any method found, per problem
        best: dict[tuple[str, str], float] = {}
        for r in res.values():
            v = "gn" if "gn" in r["variant"].split("+") else "base"
            for p, pn in enumerate(r["problems"]):
                rows = np.asarray(r["rows_prob"]) == p
                lb = float(np.min(np.asarray(r["best_full"])[rows]))
                best[(v, pn)] = min(best.get((v, pn), np.inf), lb)
    for name, r in res.items():
        cs = "gn" if "gn" in r["variant"].split("+") else "base"
        chk = json.loads((root / f"check_{a.kind}_{cs}.json").read_text())
        plateau = dict(zip(chk["problems"], chk["plateau"], strict=True))
        rows_prob = np.asarray(r["rows_prob"])
        if a.kind == "twin":
            q = np.full(rows_prob.size, Q_TWIN)
        else:
            q = np.asarray([Q_REAL_FACTOR * best[(cs, pn)] for pn in r["problems"]])[rows_prob]
        sm = summarise(r, q, plateau)
        summ[name] = sm
        tm = sm["timing"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            f"success {sm['success']}/{sm['n_rows']} (any evaluated point: {sm['reached_q_any_point']}); "
            f"process {tm['process_s']:.1f} s (compile {tm['compile_s']:.1f} s, {tm['n_programs']} programs; "
            f"model calls {tm['call_s']:.1f} s); wall to the last row at quality "
            f"{_fmt(sm['wall_to_last_q'])} s, to half the rows {_fmt(sm['wall_to_half'])} s; "
            f"seasons simulated fwd {tm['seasons']['fwd']} / jvp {tm['seasons']['jvp']}; "
            f"devices {tm['n_devices']}"
        )
        d = sm["dssat_to_last_q"] or sm["dssat_full"]
        lines.append(
            f"DSSAT cost model on this trajectory (to the last row at quality): serial 1 core "
            f"{d['serial_s'] / 3600:.2f} h, 192-core node {d['node_s']:.0f} s, "
            f"{d['invocations']:.0f} invocations"
        )
        lines.append("")
        lines.append(
            "| problem | trt | success | loss med | fwd med | jvp med | fwd to Q | "
            + " | ".join(f"{n} rel err (rec)" for n in PARAMS)
            + " |"
        )
        lines.append("|" + "---|" * (7 + len(PARAMS)))
        for pp in sm["per_problem"]:
            cells = []
            for i in range(len(PARAMS)):
                if not pp["free"][i]:
                    cells.append("fixed")
                else:
                    cells.append(f"{_fmt(pp['rel_err_median'][i])} ({pp['recovered'][i]}/{pp['n_rows']})")
            lines.append(
                f"| {pp['problem']} | {pp['n_trt']} | {pp['success']}/{pp['n_rows']} | "
                f"{_fmt(pp['loss_median'])} | "
                f"{_fmt(pp['fwd_median'])} | {_fmt(pp['jvp_median'])} | {_fmt(pp['fwd_to_q_median'])} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
        if r["method"] == "staged":
            st = r["stage"]
            ad = {n: [0, 0] for n in GROWTH}
            for pl in st["trust"]["plans"]:
                for row in pl:
                    ad[row["param"]][0] += len(row["ad_treatments"])
            for pn in r["problems"]:
                for n in GROWTH:
                    ad[n][1] += r["n_treatments"][pn]
            lines.append(
                "methods: P1, P2, P5, PHINT stage 1 CMA-ES (restarts "
                f"{int(np.sum(st['stage1'].get('restarts', [0])))} over all rows; stage-1 tolerance reached "
                f"{int(np.sum(st['stage1']['reached_tol']))}/{len(rows_prob)} rows) and stage 3 "
                "central secants "
                f"({len(st.get('stage3', {}).get('rows', []))} rows); "
                + "; ".join(
                    f"{n} AD on {v[0]}/{v[1]} (treatment, problem) pairs, secants on the rest"
                    for n, v in ad.items()
                )
                + "; stage 2: "
                + ", ".join(
                    f"{m_} on {int(np.sum(np.asarray(st['stage2'].get('method_rows', [])) == m_))} rows"
                    for m_ in ("lm", "cma")
                )
                + f"; stage walls: 1 {st['stage1']['wall_s']:.1f} s, trust {st['trust']['wall_s']:.1f} s, "
                f"2 {st['stage2']['wall_s']:.1f} s, 3 {st.get('stage3', {}).get('wall_s', 0.0):.1f} s"
            )
            lines.append("")
    txt = "\n".join(lines)
    print(txt)
    (root / f"tables_{a.kind}.md").write_text(txt)
    dump(root / f"summary_{a.kind}.json", summ)


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "step",
        choices=(
            "prep",
            "check",
            "runs",
            "method",
            "roundtrip",
            "table",
            "diag",
            "table3",
            "realtable",
            "budget",
            "ident",
        ),
    )
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--kind", default="twin", choices=("twin", "real", "ufga"))
    ap.add_argument("--method", default="staged")
    ap.add_argument("--methods", default="staged,cma,random")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--variants", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--grid", type=int, default=41)
    a = ap.parse_args()
    {
        "prep": step_prep,
        "check": step_check,
        "runs": step_runs,
        "method": run_method,
        "roundtrip": step_roundtrip,
        "table": step_table,
        "diag": step_diag,
        "table3": step_table3,
        "realtable": step_real_table,
        "budget": step_budget,
        "ident": step_ident,
    }[a.step](a)


if __name__ == "__main__":
    main()
