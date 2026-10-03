"""Faithful against smoothed phenology on the DSSAT maize examples (measurement / driver only).

The smoothed model is the free-run DSSAT day of the staged-calibration driver (``d3_1_staged.py``: configuration
``free`` of ``tests/integration/day_dssat486_free_harness.py``, f64, gradient mode ``ste``) with the crop
entries ``phenology`` and ``growth`` replaced by the non-faithful ``alt_smoothed`` variant
(:func:`agrijax.models.day_dssat486.smoothed_crop_processes`): every stage threshold is a logistic gate of
scale ``width`` [degC d] (``CeresMaizeParams.smoothing``). A stage date of the smoothed model is either the
hard date (first day of the stage code; the gate crosses 0.5) or the soft one (expected passage day, the
sum of ``1 - passage`` over every simulated day, padded days included); the smoothed calibration and
gradient checks use the soft dates, which are the differentiable ones.

Steps (all under ``$AGRI_JAX_DATA/validation/aj_g0b``, rorqual only):

* ``prep`` - the staged-calibration inputs (dscsm048 reference runs of the 58 treatments; ``d3_1_staged.step_prep``) and
  the identifiability files of the staged-calibration study (``check_*.json``: the coefficients each problem leaves fixed);
* ``forward`` - (a) at the published cultivars, per width: yield, biomass, grain number and the
  emergence / silking / maturity dates of the smoothed model against dscsm048 and the faithful model;
* ``grad`` - (b) per width (and the faithful model): the gradient-trust verdicts of
  :mod:`agrijax.calib.trust` (three-valued level 2 at 1e-4 / 1e-5 / 1e-6 of the CUL width, level 3 at the
  1 / 2 / 5 % secants and a 41-point scan over +-10 %) of yield, biomass, grain number and the silking and
  maturity dates with respect to P1, P2, P5, PHINT, G2, G3 at the published cultivar of every treatment;
* ``adam`` - (c) the smoothed model calibrated by Adam (forward-mode AD gradient of the objective, all free
  coefficients) on the staged-calibration twin problems (``--kind twin``) or UFGA8201 t04 (``--kind ufga4``, grain number
  included), from the staged-calibration starts; the result is also evaluated in the faithful model; ``--self-twin``:
  observations from the smoothed model itself; ``--restart none``: one trajectory per row (no cold
  restarts); reproduce a run with ``--iters`` = its recorded ``budget.iterations``; ``--refine N``: the
  faithful selection (Adam iterates only, no fresh starts) refined by N CMA-ES evaluations on the faithful
  model (the start kept where the search does not beat it), and the same CMA-ES from the staged-calibration starts as
  its baseline, each with ``--refine-streams`` random streams;
* ``d31`` - the faithful baselines with the staged-calibration driver itself (``--method cma``: CMA-ES on all coefficients;
  ``--method staged``: the hybrid ``gradient_plan``), with ``ufga4`` = UFGA8201 t04; ``--self-twin
  --width W``: the same methods on the smoothed self-twin of ``adam --self-twin`` (same objective, starts
  and budget); ``--smoothed-model --width W``: the same methods on the smoothed model against the ordinary
  twin (faithful observations, the objective of ``adam`` without ``--self-twin``), the result also scored
  in the faithful model (the smoothed model's bias, separated from Adam's optimisation error);
* ``rescore`` - every self-twin result scored with the hard dates of the smoothed model;
* ``holdout`` - UFGA8201 t06 (held out) and t04 for the best row of every ``ufga4`` result, in the faithful
  model, the smoothed model of its width and dscsm048 (the calibrated cultivar written into a copy of
  ``MZCER048.CUL``); the twin results' best rows in dscsm048 against the truth;
* ``table`` - the tables of the result note.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_spec = importlib.util.spec_from_file_location("d3_1_staged", HERE / "d3_1_staged.py")
assert _spec is not None and _spec.loader is not None
D = importlib.util.module_from_spec(_spec)
sys.modules["d3_1_staged"] = D
_spec.loader.exec_module(D)

#: the gate scales swept [degC d]: about a fifth of a day of thermal time, a third of a day, torchcrop's
#: k = 50 on a development scale of 850 degC d per unit (emergence to silking), van Bree's slope 50 on the
#: season-length normalised sum (about 1700 degC d), and twice that
WIDTHS = (2.0, 5.0, 17.0, 35.0, 70.0)
#: the gradient check also runs below a day of thermal time (where the soft stage-4 duration underflows)
GRAD_WIDTHS = (0.3, 1.0, *WIDTHS)
#: staged-calibration result directory whose identifiability files (check_*.json) fix the same coefficients here
D31_CHECKS = "aj_d31_gradgap/three"
#: UFGA8201: calibrated on t04 (high nitrogen, irrigated), t06 held out
UFGA4 = {"UFGA8201/IB0035": ["UFGA8201_t04"]}
UFGA6 = {"UFGA8201/IB0035": ["UFGA8201_t06"]}
#: Adam (z units: the logit coordinates of the CUL box, staged-calibration ``space``)
ADAM_ITERS = 300  # fixed-iteration runs (--iters); the default is budget-aligned
#: Adam iterations per segment (a restart from a fresh start after each one, unless at the target)
ADAM_SEGMENT = 150
ADAM_LR0 = 0.05
ADAM_LR_MIN = 0.0025
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
#: faithful objective evaluated along the Adam path every this many iterations
FAITHFUL_EVERY = 25
#: a faithful checkpoint counts for the selection only after this many Adam steps since the row's last
#: (re)start: a fresh start evaluated at a checkpoint is a random-search point, not an Adam iterate
FAITHFUL_MIN_STEPS = 1
#: independent random streams of the faithful CMA-ES refinement and of its baseline from the staged-calibration starts
#: (the spread of the refined success count across streams is the refinement's own noise)
REFINE_STREAMS = 3
#: offset between the CMA-ES seeds of two refinement streams
REFINE_STREAM_STRIDE = 1000
#: passage index of a stage code of the staged-calibration entries (emergence 1, silking 4, maturity 10)
CODE_PASSAGE = {D.ISTAGE_EMERGENCE: 0, D.ISTAGE_SILKING: 3, D.ISTAGE_MATURITY: 6}
#: gradient check outputs: (name, entry type, out / stage code)
GRAD_OUTPUTS = (
    ("yield", D.E_FINAL, D.OUT_NAMES.index("gwad")),
    ("biomass", D.E_FINAL, D.OUT_NAMES.index("cwad")),
    ("grain_number", D.E_FINAL, D.OUT_NAMES.index("g_ad")),
    ("silking", D.E_DATE, D.ISTAGE_SILKING),
    ("maturity", D.E_DATE, D.ISTAGE_MATURITY),
)
FORWARD_OUTPUTS = (
    ("emergence", D.E_DATE, D.ISTAGE_EMERGENCE),
    ("silking", D.E_DATE, D.ISTAGE_SILKING),
    ("maturity", D.E_DATE, D.ISTAGE_MATURITY),
    ("yield", D.E_FINAL, D.OUT_NAMES.index("gwad")),
    ("biomass", D.E_FINAL, D.OUT_NAMES.index("cwad")),
    ("grain_number", D.E_FINAL, D.OUT_NAMES.index("g_ad")),
)


def out_root() -> Path:
    return D.out_root()


def _setup_env() -> None:
    data = Path(os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")).expanduser()
    os.environ.setdefault("AJ_D31_DIR", str(data / "validation" / "aj_g0b"))
    os.environ.setdefault("AJ_D31_TRUST", "three")
    D.TRUST_LEVEL2 = os.environ["AJ_D31_TRUST"]


# ============================================================================ evaluator
def _smooth_outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
    from agrijax.core.state import get_path
    from agrijax.models.day_dssat486 import SLOT

    out = D.calib_outputs(state, params, forcing_t)
    soft = get_path(state, f"crops.{SLOT}").soft
    return {**out, "passage": soft.passage[0]}


def _gather_soft(y: Any, ist: Any, pas: Any, etype: Any, eout: Any, et: Any, soft: bool) -> Any:
    """:func:`d3_1_staged._gather_one` with the soft dates (expected passage day) when ``soft``."""
    import jax.numpy as jnp

    hard = D._gather_one(y, ist, etype, eout, et)
    if not soft:
        return hard
    # the expected passage day over the first ``et`` days (a date entry's ``t``, set by
    # :meth:`SmoothedEvaluator.set_entries`: every simulated day by default, the real season days only with
    # ``soft_span="real"``)
    real = (jnp.arange(pas.shape[0])[:, None] < jnp.max(jnp.where(etype == D.E_DATE, et, 0))).astype(
        pas.dtype
    )
    expected = jnp.sum((1.0 - pas) * real, axis=0)  # [7]
    idx = jnp.select(
        [eout == D.ISTAGE_EMERGENCE, eout == D.ISTAGE_SILKING, eout == D.ISTAGE_MATURITY],
        [CODE_PASSAGE[D.ISTAGE_EMERGENCE], CODE_PASSAGE[D.ISTAGE_SILKING], CODE_PASSAGE[D.ISTAGE_MATURITY]],
        0,
    )
    return jnp.where(etype == D.E_DATE, expected[idx], hard)


def _smooth_item(it: dict[str, Any], width: float) -> dict[str, Any]:
    """The item's params and state for the smoothed day: gate scale ``width``, zero soft clocks."""
    import jax.numpy as jnp

    from agrijax.core.state import get_path, set_path
    from agrijax.models.day_dssat486 import SLOT
    from agrijax.processes.crop.ceres_maize import SmoothingCoefficients, with_soft_state

    sm = SmoothingCoefficients().as_arrays(jnp.float64)
    sm = sm.replace(width=jnp.asarray(width, jnp.float64))
    params = {**it["params"], "crop": it["params"]["crop"].replace(smoothing=sm)}
    path = f"crops.{SLOT}"
    state = set_path(it["state"], path, with_soft_state(get_path(it["state"], path)))
    return {**it, "params": params, "state": state}


class SmoothedEvaluator(D.Evaluator):
    """The staged-calibration evaluator on the smoothed day (``smoothed``), soft or hard dates, the gate scale settable
    without a recompile (:meth:`set_width`: the scale is an input array of the compiled programs)."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        entries: dict[str, dict[str, np.ndarray]],
        *,
        width: float,
        soft_dates: bool = True,
        soft_span: str = "all",
    ) -> None:
        if soft_span not in ("all", "real"):
            raise ValueError(soft_span)
        self.width = width
        self.soft_dates = soft_dates
        self.soft_span = soft_span
        super().__init__([_smooth_item(it, width) for it in items], entries)

    def set_entries(self, entries: dict[str, dict[str, np.ndarray]]) -> None:
        """The staged-calibration entry tables with the day count the soft dates are summed over in ``t`` of every date
        entry: every simulated day (``soft_span="all"``, the default: a candidate that matures after the
        reference season is penalised by its lateness, as by the hard date) or the real season days only
        (``"real"``, the round-1 definition: blind to a maturity after the reference season, which ends
        at maturity in most treatments); the hard dates do not read ``t``."""
        ent = {}
        span = "n_padded" if self.soft_span == "all" else "n_days"
        for k, tab in entries.items():
            tab = {kk: np.array(vv) for kk, vv in tab.items()}
            it = self.items[self.index[k]]
            tab["t"] = np.where(tab["type"] == D.E_DATE, int(it[span]), tab["t"]).astype(np.int32)
            ent[k] = tab
        super().set_entries(ent)

    def set_width(self, width: float) -> None:
        import jax
        import jax.numpy as jnp

        self.width = width
        for g, (p, f, s) in list(self.inputs.items()):
            crop = p["crop"]
            sm = crop.smoothing
            p2 = {**p, "crop": crop.replace(smoothing=sm.replace(width=jnp.full_like(sm.width, width)))}
            self.inputs[g] = jax.device_put((p2, f, s), self.rep)

    def _program(self, g: tuple[int, str], sp: int, k: int, mode: str = D.MODE) -> Any:
        key = (g, sp, k, mode)
        if key in self.programs:
            return self.programs[key]
        import jax
        import jax.numpy as jnp

        from agrijax.core.grad import bind_gradient_mode, bind_unrounded
        from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_processes, smoothed_crop_processes

        def bind(fn: Any) -> Any:
            """``fn`` in gradient mode ``mode``; ``"unrounded"``: the unrounded model in ``D.MODE``."""
            if mode == "unrounded":
                return bind_gradient_mode(bind_unrounded(fn), D.MODE)
            return bind_gradient_mode(fn, mode)

        procs = day_processes(SLOT, mesev=g[1], replace=smoothed_crop_processes(SLOT))
        model = day_dssat486(SLOT).compile(procs, outputs=_smooth_outputs, exact_lags=True)
        step = jax.vmap(model.compile(), in_axes=(0, 0, 0))
        n_days = int(self.items[self.groups[g][0]]["n_padded"])
        soft = self.soft_dates

        def sim(inputs: Any, tab: Any, theta: Any, tid: Any) -> Any:
            params_tr, forcing_tr, state_tr = inputs
            params = D.set_cultivar(jax.tree.map(lambda x: x[tid], params_tr), theta)
            state0 = jax.tree.map(lambda x: x[tid], state_tr)

            def body(s: Any, t: Any) -> Any:
                return step(s, params, jax.tree.map(lambda x: x[tid, t], forcing_tr))

            _, outs = jax.lax.scan(body, state0, jnp.arange(n_days))
            y = jnp.stack([outs[n] for n in D.OUT_NAMES], axis=0)  # [O, T, S]
            return jax.vmap(
                lambda yy, ii, pp, a, b, c: _gather_soft(yy, ii, pp, a, b, c, soft),
                in_axes=(2, 1, 1, 0, 0, 0),
            )(y, outs["istage"], outs["passage"], tab["type"][tid], tab["out"][tid], tab["t"][tid])

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


def _entries(spec: Sequence[tuple[str, int, int]]) -> dict[str, np.ndarray]:
    return D._table([(ty, o, 0) for _, ty, o in spec])


def _all_items() -> list[dict[str, Any]]:
    import pickle

    with open(out_root() / "inputs.pkl", "rb") as fh:
        return pickle.load(fh)


def _day_index(it: dict[str, Any], yrdoy: float) -> float:
    if not np.isfinite(yrdoy) or yrdoy <= 0:
        return float("nan")
    days = np.asarray(it["days"])
    hit = np.nonzero(days == int(yrdoy))[0]
    return float(hit[0]) if hit.size else float("nan")


# ============================================================================ prep
def step_prep(a: argparse.Namespace) -> None:
    D.step_prep(a)
    src = Path(os.environ["AGRI_JAX_DATA"]) / "validation" / D31_CHECKS
    found = sorted(src.glob("check_*.json"))
    if not found:  # free_mask would silently free all six coefficients: not the staged-calibration problems
        raise SystemExit(f"prep: no staged-calibration identifiability files check_*.json in {src}")
    for f in found:
        shutil.copy2(f, out_root() / f.name)
        print("copied", f.name, flush=True)


def _check_file(kind: str) -> tuple[str, str]:
    """``(kind, observation set)`` of :func:`d3_1_staged.free_mask` for ``kind`` (``twin`` / ``ufga4``),
    refusing to go on without the staged-calibration identifiability file: without it ``free_mask`` frees every
    coefficient, a different problem from the staged-calibration one."""
    kk, cs = ("real", "gn") if kind == "ufga4" else (kind, "base")
    f = out_root() / f"check_{kk}_{cs}.json"
    if not f.exists():
        raise SystemExit(f"{f} is missing: run the prep step (the staged-calibration identifiability files)")
    return kk, cs


def _free_mask(kind: str, problems: dict[str, list[str]]) -> np.ndarray:
    """:func:`d3_1_staged.free_mask` of ``kind``, with the file check of :func:`_check_file`."""
    kk, cs = _check_file(kind)
    return D.free_mask(kk, problems, cs)


# ============================================================================ (a) forward departure
def step_forward(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    items = _all_items()
    keys = np.asarray([it["key"] for it in items])
    theta = np.asarray([it["theta_pub"] for it in items])
    ent = {it["key"]: _entries(FORWARD_OUTPUTS) for it in items}
    names = [n for n, _, _ in FORWARD_OUTPUTS]
    t0 = time.perf_counter()
    ev_f = D.Evaluator(items, ent)
    y_f, _ = ev_f.run(theta, keys)
    res: dict[str, Any] = {
        "keys": keys.tolist(),
        "names": names,
        "faithful": y_f[:, : len(names)],
        "faithful_stats": ev_f.stats(),
    }
    dss = np.zeros((len(items), len(names)))
    for s_, it in enumerate(items):
        sm = it["summary"]
        dss[s_] = [
            _day_index(it, sm["EDAT"]),
            _day_index(it, sm["ADAT"]),
            _day_index(it, sm["MDAT"]),
            sm["HWAM"],
            sm["CWAM"],
            sm["H#AM"],
        ]
    res["dscsm048"] = dss
    ev_h = SmoothedEvaluator(items, ent, width=WIDTHS[0], soft_dates=False)
    ev_s = SmoothedEvaluator(items, ent, width=WIDTHS[0], soft_dates=True)
    ev_r = SmoothedEvaluator(items, ent, width=WIDTHS[0], soft_dates=True, soft_span="real")
    for w in WIDTHS:
        ev_h.set_width(w)
        ev_s.set_width(w)
        ev_r.set_width(w)
        yh, _ = ev_h.run(theta, keys)
        ys, _ = ev_s.run(theta, keys)
        yr, _ = ev_r.run(theta, keys)
        res[f"smoothed_{w:g}"] = {
            "hard": yh[:, : len(names)],
            "soft": ys[:, : len(names)],
            "soft_real_days": yr[:, : len(names)],
        }
        print(f"width {w:g}: done ({time.perf_counter() - t0:.0f} s)", flush=True)
    # the vanishing scale (the faithful limit) as a check of the setup: hard and soft dates
    ev_h.set_width(1e-12)
    ev_s.set_width(1e-12)
    y0, _ = ev_h.run(theta, keys)
    y0s, _ = ev_s.run(theta, keys)
    res["smoothed_1e-12"] = {"hard": y0[:, : len(names)], "soft": y0s[:, : len(names)]}
    res["smoothed_stats"] = ev_h.stats()
    D.dump(out_root() / "forward.json", res)
    print("wrote forward.json", flush=True)
    # at the vanishing scale every soft date must be the hard one (outside GAGR0201, where the limit does
    # not hold on its repeated-date padded days): checked after the dump, so the numbers stay inspectable
    keep = np.asarray([not str(k).startswith("GAGR") for k in keys])
    hard0, soft0 = y0[keep, :3], y0s[keep, :3]
    bad = ~((hard0 == soft0) | (np.isnan(hard0) & np.isnan(soft0)))
    if bad.any():
        rows = [(str(keys[keep][i]), names[j], hard0[i, j], soft0[i, j]) for i, j in np.argwhere(bad)]
        raise SystemExit(f"soft != hard dates at scale 1e-12: {rows}")
    print("scale 1e-12: soft dates == hard dates on", int(keep.sum()), "treatments", flush=True)


# ============================================================================ (b) gradient quality
def _grad_quality(ev: Any, items: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Trust verdicts per (treatment, output, coefficient) at the published cultivars."""
    from agrijax.calib.trust import TrustConfig, _classify, _level, fd_diagnostics, scan_summary

    cfg = TrustConfig()
    sp = D.space()
    lo, hi, width = np.asarray(sp.lower), np.asarray(sp.upper), np.asarray(sp.width)
    keys = np.asarray([it["key"] for it in items])
    th0 = np.asarray([it["theta_pub"] for it in items])
    s_n, n, m = len(items), 6, len(GRAD_OUTPUTS)
    eye = np.eye(n)
    v = np.repeat(eye[:, None, :], s_n, axis=1)  # [6, S, 6]
    t0 = time.perf_counter()
    y0, ad = ev.run(th0, keys, v)
    _, adx = ev.run(th0, keys, v, mode="exact")
    y0 = y0[:, :m]
    ad = np.transpose(ad[:, :, :m], (1, 2, 0))  # [S, m, n]
    adx = np.transpose(adx[:, :, :m], (1, 2, 0))
    nonfinite = {
        "treatments_ad": int((~np.all(np.isfinite(ad), axis=(1, 2))).sum()),
        "treatments_ad_exact": int((~np.all(np.isfinite(adx), axis=(1, 2))).sum()),
        "values_ad": int((~np.isfinite(ad)).sum()),
        "values": int(ad.size),
    }
    steps = (*cfg.fd_small_steps, *cfg.fd_large_steps)
    pts, kk = [], []
    hstep = np.zeros((len(steps), n, s_n))
    for si, frac in enumerate(steps):
        for i in range(n):
            tp = th0.copy()
            tm = th0.copy()
            tp[:, i] = np.minimum(th0[:, i] + frac * width[i], hi[i])
            tm[:, i] = np.maximum(th0[:, i] - frac * width[i], lo[i])
            hstep[si, i] = tp[:, i] - tm[:, i]
            pts += [tp, tm]
            kk += [keys, keys]
    yf, _ = ev.run(np.concatenate(pts), np.concatenate(kk))
    yf = yf[:, :m].reshape(len(steps), n, 2, s_n, m)
    fd = (yf[:, :, 0] - yf[:, :, 1]) / np.maximum(hstep, 1e-300)[..., None]  # [steps, n, S, m]
    # scans: 41 points over +-10 % of the width per coefficient, AD along that coefficient
    grids = np.zeros((n, s_n, cfg.n_scan))
    sp_pts, sp_keys, sp_v = [], [], []
    for i in range(n):
        for s_ in range(s_n):
            a0 = max(lo[i], th0[s_, i] - cfg.scan_frac * width[i])
            b0 = min(hi[i], th0[s_, i] + cfg.scan_frac * width[i])
            grids[i, s_] = np.linspace(a0, b0, cfg.n_scan)
            t = np.repeat(th0[s_][None], cfg.n_scan, axis=0)
            t[:, i] = grids[i, s_]
            sp_pts.append(t)
            sp_keys += [keys[s_]] * cfg.n_scan
            sp_v.append(np.repeat(eye[i][None], cfg.n_scan, axis=0))
    ys, gs = ev.run(np.concatenate(sp_pts), np.asarray(sp_keys), np.concatenate(sp_v)[None])
    ys = ys[:, :m].reshape(n, s_n, cfg.n_scan, m)
    gs = gs[0, :, :m].reshape(n, s_n, cfg.n_scan, m)
    wall = time.perf_counter() - t0
    nonfinite["scan_points_ad"] = int((~np.isfinite(gs)).sum())
    nonfinite["scan_points"] = int(gs.size)
    ns = len(cfg.fd_small_steps)
    rows = []
    for s_ in range(s_n):
        fdd = fd_diagnostics(
            ad[s_],
            adx[s_],
            [fd[k, :, s_].T for k in range(ns)],
            [fd[k, :, s_].T for k in range(ns, len(steps))],
            y0[s_],
            width,
            cfg,
        )
        for i in range(n):
            sc = scan_summary(grids[i, s_], ys[i, s_], gs[i, s_], float(width[i]), cfg)
            for j, (oname, _, _) in enumerate(GRAD_OUTPUTS):
                large = [float(fd[k, i, s_, j]) for k in range(ns, len(steps))]
                rows.append(
                    {
                        "key": str(keys[s_]),
                        "param": D.PARAMS[i],
                        "output": oname,
                        "y0": float(y0[s_, j]),
                        "ad": float(ad[s_, j, i]),
                        "ad_exact": float(adx[s_, j, i]),
                        "secants": large,
                        "fd_small": [float(fd[k, i, s_, j]) for k in range(ns)],
                        "status0": str(fdd["status0"][j, i]),
                        "rel_err_2pct": float(fdd["rel_err1"][j, i]),
                        "rel_err_large_max": float(fdd["rel_err1_max"][j, i]),
                        "agree_large": bool(fdd["agree1"][j, i]),
                        "n_jumps": int(sc["n_jumps"][j]),
                        "zero_frac": float(sc["zero_frac"][j]),
                        "scan_range": float(sc["range"][j]),
                        "class": _classify(fdd, sc, j, i, cfg),
                        "level": _level(fdd, sc, j, i, cfg),
                    }
                )
    return {"label": label, "rows": rows, "wall_s": wall, "stats": ev.stats(), "nonfinite": nonfinite}


def step_grad(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    items = _all_items()
    ent = {it["key"]: _entries(GRAD_OUTPUTS) for it in items}
    out: dict[str, Any] = {}
    out["faithful"] = _grad_quality(D.Evaluator(items, ent), items, "faithful (hard dates)")
    print("faithful done", flush=True)
    ev = SmoothedEvaluator(items, ent, width=GRAD_WIDTHS[0], soft_dates=True)
    for w in GRAD_WIDTHS:
        ev.set_width(w)
        out[f"smoothed_{w:g}"] = _grad_quality(ev, items, f"smoothed width {w:g} (soft dates)")
        print(f"width {w:g} done", flush=True)
    D.dump(out_root() / "grad.json", out)
    print("wrote grad.json", flush=True)


# ============================================================================ (c) calibration
def _build(kind: str) -> tuple[Any, dict[str, list[str]], dict[str, Any], tuple[str, ...]]:
    """The faithful staged-calibration problem of ``kind`` (``twin``: base codes; ``ufga4``: UFGA8201 t04 with grain
    number; ``ufga6``: its held-out t06)."""
    if kind == "twin":
        codes = D.CODES_BASE
        pb, problems, info = D.build("twin", codes)
    else:
        codes = D.CODES_GN
        D.UFGA_CAL = UFGA4 if kind == "ufga4" else UFGA6
        pb, problems, info = D.build("ufga", codes)
    return pb, problems, info, codes


def _smoothed_build(width: float, self_obs: bool, holder: dict[str, Any], soft_dates: bool = True) -> Any:
    """A replacement for :func:`d3_1_staged.build` whose ``twin`` problem has the smoothed model of scale
    ``width`` as the evaluator. ``self_obs``: the smoothed self-twin (observations from the smoothed model
    at the truth, the problem :func:`step_adam` solves with ``--self-twin``); otherwise the ordinary twin
    (observations from the faithful model: the smoothed model's bias included). The faithful problem is
    left in ``holder["faithful"]`` (to score the result in the faithful model)."""
    orig = D.build

    def build(kind: str, codes: Sequence[str]) -> tuple[Any, dict[str, list[str]], dict[str, Any]]:
        if kind != "twin":
            raise SystemExit("--self-twin / --smoothed-model need --kind twin")
        pb_f, problems, info = orig(kind, codes)
        holder["faithful"] = pb_f
        ev_s = SmoothedEvaluator(
            pb_f.ev.items, {k: D._table([]) for k in pb_f.ev.index}, width=width, soft_dates=soft_dates
        )
        obj = D.Objective(D.twin_obs(ev_s, problems, codes), problems) if self_obs else pb_f.obj
        return D.Problem(ev_s, obj), problems, info

    return build


def step_d31(a: argparse.Namespace) -> None:
    """A staged-calibration method on ``twin`` / ``ufga4`` with the staged-calibration driver itself (fresh process: compile counted).

    With ``--self-twin`` the method runs on the smoothed self-twin of scale ``--width`` (the observations
    and the evaluator from the smoothed model, soft dates: the same objective, starts and budget as
    ``adam --self-twin``); the result is ``twin_self_<method>_w<width>_s<seed>.json``.

    With ``--smoothed-model`` the method runs on the smoothed model of scale ``--width`` against the
    ordinary twin's observations (from the faithful model; the objective ``adam`` without ``--self-twin``
    optimises). Its returned coefficients are also scored in the faithful model
    (``loss_faithful_final``): with a strong optimiser this separates the smoothed model's bias from
    Adam's optimisation error. The result is ``twin_sm_<method>_w<width>_s<seed>.json``.
    """
    _check_file(a.kind)
    if a.self_twin or a.smoothed_model:
        if a.kind != "twin":
            raise SystemExit("--self-twin / --smoothed-model need --kind twin")
        if a.self_twin and a.smoothed_model:
            raise SystemExit("--self-twin and --smoothed-model exclude each other")
        import jax

        jax.config.update("jax_enable_x64", True)
        holder: dict[str, Any] = {}
        setattr(D, "build", _smoothed_build(float(a.width), bool(a.self_twin), holder))  # noqa: B010
        tag = "self" if a.self_twin else "sm"
        dst = out_root() / f"twin_{tag}_{a.method}_w{float(a.width):g}_s{a.seed}.json"
        dump0 = D.dump
        extra: dict[str, Any] = {"self_twin": bool(a.self_twin), "width": float(a.width)}

        def dump(path: Path, obj: dict[str, Any]) -> None:  # one file per run
            if a.smoothed_model:  # the smoothed-model result scored in the faithful model
                pb_f = holder["faithful"]
                rp = np.asarray(obj["rows_prob"])
                _, lf = pb_f.evaluate(np.asarray(obj["theta_final"]), rp, pb_f.full)
                _, lfw = pb_f.evaluate(np.asarray(obj["theta_written"]), rp, pb_f.full)
                obj = {
                    **obj,
                    "smoothed_model": True,
                    "loss_smoothed_final": obj["loss_final"],
                    "loss_faithful_final": lf,
                    "loss_faithful_written": lfw,
                }
            dump0(dst, {**obj, **extra})

        setattr(D, "dump", dump)  # noqa: B010
        D.run_method(argparse.Namespace(kind="twin", method=a.method, variant="base", seed=a.seed))
        print("wrote", dst.name, flush=True)
        return
    if a.kind == "ufga4":
        D.UFGA_CAL = UFGA4
    ns = argparse.Namespace(
        kind="ufga" if a.kind == "ufga4" else a.kind,
        method=a.method,
        variant="gn" if a.kind == "ufga4" else "base",
        seed=a.seed,
    )
    D.run_method(ns)
    src = out_root() / f"{ns.kind}_{ns.method}_{ns.variant}_s{ns.seed}.json"
    dst = out_root() / f"{a.kind}_{a.method}_s{a.seed}.json"
    if src != dst:
        src.rename(dst)
    print("wrote", dst.name, flush=True)


def step_adam(a: argparse.Namespace) -> None:
    """Adam on the smoothed model (module docstring), budget-aligned with the staged-calibration CMA-ES.

    Budget: ``--budget`` forward-evaluation equivalents per row (default the staged-calibration CMA-ES budget,
    :data:`d3_1_staged.CMA_ALL_EVALS`); one Adam iteration is one ``6``-direction JVP call, whose cost in
    forwards is measured on the run's own batch (median of 3 timed calls of each kind after compile), so
    the iteration count is ``budget / (t_jvp / t_fwd)``. The budget unit is one forward of the *smoothed*
    model on Adam's own batch; the faithful checkpoint forwards (selection) are counted separately
    (``budget.faithful_checkpoint_calls``) and the ``--refine`` evaluations on top. The timed ratio is
    noisy (2.2-3.2 across identical configurations), so a run is reproduced with ``--iters`` set to the
    recorded ``budget.iterations`` (with the same ``--seed`` the starts and restarts are then the same).

    Restarts (``--restart cold``, the default): the iterations are split into segments of
    :data:`ADAM_SEGMENT`; after a segment a row that has not reached the quality target (twin
    :data:`d3_1_staged.Q_TWIN` on the smoothed objective; none on ``ufga4``, so every row restarts) restarts
    from a fresh staged-calibration start with fresh moments and a fresh cosine schedule; the best point of every row is
    kept over all segments. ``--restart none``: one trajectory per row over all the iterations, one cosine
    schedule from :data:`ADAM_LR0` to :data:`ADAM_LR_MIN` over the whole run (the control without cold
    restarts; result ``<kind>_[self_]adam_single_w<width>_s<seed>.json``).

    Selection: the best iterate on the smoothed objective (``theta_final``) and, among the checkpoints
    (every :data:`FAITHFUL_EVERY` iterations and each segment end), the best on the faithful objective
    (``theta_best_faithful``). A checkpoint counts only for rows that have taken at least
    :data:`FAITHFUL_MIN_STEPS` Adam steps since their last (re)start: the iterate of a checkpoint that
    falls on the first iteration after a restart is the fresh random start itself, and selecting it on
    the faithful objective would credit Adam with a random search in the faithful model. The number of
    Adam steps behind each selected point is recorded (``final_steps``, ``best_faithful_steps``; 0 = a
    start). With ``--refine N`` the faithful selection is refined by CMA-ES on the faithful model (``N``
    evaluations per row, ``theta_refined``; the start is evaluated and kept where the search does not beat
    it, the search's own best point is ``loss_faithful_refined_raw``), and the same CMA-ES is run with the
    same budget from the staged-calibration starts (``baseline_cma``: what the refinement evaluations reach without
    Adam); both with ``--refine-streams`` independent random streams (``streams``; the top-level fields
    are stream 0). Non-finite gradients
    are counted per iteration and row (the row's update is skipped, nothing is zeroed silently) and
    reported.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    t_start = time.perf_counter()
    width = float(a.width)
    pb_f, problems, info, codes = _build(a.kind)
    obj = pb_f.obj
    ev_s = SmoothedEvaluator(pb_f.ev.items, {k: D._table([]) for k in pb_f.ev.index}, width=width, soft_dates=True)
    if a.self_twin:  # the twin observations from the smoothed model itself: no model mismatch
        if a.kind != "twin":
            raise SystemExit("--self-twin needs --kind twin")
        obj = D.Objective(D.twin_obs(ev_s, problems, codes), problems)
        pb_f = D.Problem(pb_f.ev, obj)
    pb_s = D.Problem(ev_s, obj)
    t_build = time.perf_counter()
    sp = D.space()
    p_n = len(problems)
    truth = D.truth_of(pb_f)
    free = _free_mask(a.kind, problems)
    rng = np.random.default_rng(a.seed)  # the staged-calibration starts (same generator calls as run_method)
    rows_prob = np.repeat(np.arange(p_n), D.N_STARTS)
    th0 = np.concatenate([D.make_starts(sp, truth[p], rng, "perturb") for p in range(p_n)])
    r_n = rows_prob.size
    fixed_th = truth[rows_prob].copy()
    fr = free[rows_prob]
    q_target = D.Q_TWIN if a.kind == "twin" else -np.inf

    def theta_of(z: np.ndarray) -> np.ndarray:
        return np.where(fr, D.to_theta(sp, z), fixed_th)

    cand, keys = obj.layout(rows_prob)
    v = np.repeat(np.eye(6)[:, None, :], cand.size, axis=1)
    z = D.to_z(sp, th0)

    def jvp_eval(zz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th = theta_of(zz)
        y, dy = pb_s.ev.run(th[cand], keys, v)
        r_ = obj.residuals(y, keys, pb_s.full)
        w_ = obj.weights(keys, pb_s.full)
        loss = obj.per_candidate(np.sum(r_**2, axis=1), cand, r_n)
        g_s = 2.0 * np.sum(r_[None] * dy * w_[None], axis=2)  # [6, S]
        g_th = np.stack([obj.per_candidate(g_s[i], cand, r_n) for i in range(6)], axis=1)
        return loss, np.where(fr, g_th * D.dtheta_dz(sp, zz), 0.0)

    # ---- cost of one JVP iteration in forward evaluations (after compile)
    jvp_eval(z)
    pb_s.evaluate(theta_of(z), rows_prob, pb_s.full)
    t_j, t_f = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        jvp_eval(z)
        t_j.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        pb_s.evaluate(theta_of(z), rows_prob, pb_s.full)
        t_f.append(time.perf_counter() - t0)
    cost = float(np.median(t_j) / np.median(t_f))
    budget = float(a.budget)
    n_iter = int(a.iters) if a.iters else max(int(budget / cost), 1)
    seg_len = n_iter if a.restart == "none" else ADAM_SEGMENT
    print(
        f"JVP / forward cost {cost:.2f}: {n_iter} iterations for {budget:g} forward equivalents", flush=True
    )

    m1 = np.zeros_like(z)
    m2 = np.zeros_like(z)
    seg_it = np.zeros(r_n, int)
    best_loss = np.full(r_n, np.inf)
    best_z = z.copy()
    best_f = np.full(r_n, np.inf)
    best_fz = z.copy()
    steps = np.zeros(r_n, int)  # Adam steps since the row's last (re)start
    best_steps = np.zeros(r_n, int)  # ... behind the best smoothed point
    best_fsteps = np.full(r_n, -1)  # ... behind the best faithful checkpoint (-1: none counted)
    restarts = np.zeros(r_n, int)
    frozen = np.zeros(r_n, bool)  # at the target after a segment: no further updates
    nonfinite_rows = np.zeros(r_n, int)
    nonfinite_iters = 0
    hist: list[dict[str, Any]] = []
    t_opt = time.perf_counter()
    n_jvp = 0
    n_fwd_faithful = 0
    for it_ in range(n_iter + 1):
        loss, g_z = jvp_eval(z)
        n_jvp += 1
        bad = ~np.all(np.isfinite(g_z), axis=1) | ~np.isfinite(loss)
        if bad.any():
            nonfinite_iters += 1
            nonfinite_rows[bad] += 1
            if nonfinite_iters <= 5:
                pbad = sorted({obj.pnames[int(rows_prob[r])] for r in np.nonzero(bad)[0]})
                print(
                    f"WARNING iter {it_}: non-finite gradient or loss on {int(bad.sum())} rows {pbad}",
                    flush=True,
                )
        better = np.isfinite(loss) & (loss < best_loss)
        best_loss[better] = loss[better]
        best_z[better] = z[better]
        best_steps[better] = steps[better]
        rec: dict[str, Any] = {
            "iter": it_,
            "wall_s": time.perf_counter() - t_opt,
            "loss_smoothed": loss.tolist(),
        }
        seg_end = (~frozen & (seg_it + 1 >= seg_len)) | (it_ == n_iter)
        if it_ % FAITHFUL_EVERY == 0 or seg_end.any():
            _, lf = pb_f.evaluate(theta_of(z), rows_prob, pb_f.full)
            n_fwd_faithful += 1
            rec["loss_faithful"] = lf.tolist()
            bf = np.isfinite(lf) & (lf < best_f) & (steps >= FAITHFUL_MIN_STEPS)
            best_f[bf] = lf[bf]
            best_fz[bf] = z[bf]
            best_fsteps[bf] = steps[bf]
        hist.append(rec)
        if it_ == n_iter:
            break
        frac = seg_it / seg_len
        lr = ADAM_LR_MIN + 0.5 * (ADAM_LR0 - ADAM_LR_MIN) * (1.0 + np.cos(np.pi * frac))
        skip = bad | frozen
        g_u = np.where(skip[:, None], 0.0, g_z)
        m1 = np.where(skip[:, None], m1, ADAM_B1 * m1 + (1 - ADAM_B1) * g_u)
        m2 = np.where(skip[:, None], m2, ADAM_B2 * m2 + (1 - ADAM_B2) * g_u**2)
        t_ = (seg_it + 1)[:, None]
        step = lr[:, None] * (m1 / (1 - ADAM_B1**t_)) / (np.sqrt(m2 / (1 - ADAM_B2**t_)) + ADAM_EPS)
        z = np.where(skip[:, None], z, z - step)
        steps = steps + (~skip).astype(int)
        seg_it = seg_it + 1
        # restarts at a segment end: rows not at the target start again from a fresh staged-calibration start
        # (``--restart none``: no restart, the single trajectory ends at the last iteration)
        redo = seg_end & (best_loss > q_target) & (a.restart != "none")
        done = seg_end & ~redo
        if redo.any():
            idx = np.nonzero(redo)[0]
            z[idx] = np.concatenate(
                [D.to_z(sp, D.make_starts(sp, truth[rows_prob[r]], rng, "perturb")[:1]) for r in idx]
            )
            m1[idx] = 0.0
            m2[idx] = 0.0
            steps[idx] = 0
            restarts[idx] += 1
        seg_it[seg_end] = 0
        if done.any():  # rows at the target stay at their best point
            z[done] = best_z[done]
            steps[done] = best_steps[done]
            frozen |= done
        if it_ % 100 == 0:
            print(
                f"iter {it_}: median best smoothed {np.median(best_loss):.4g}, median best faithful "
                f"{np.median(best_f):.4g}, restarts {int(restarts.sum())} "
                f"({time.perf_counter() - t_opt:.0f} s)",
                flush=True,
            )
    t_adam = time.perf_counter()
    th_final = theta_of(best_z)
    th_bf = theta_of(best_fz)
    _, loss_s_final = pb_s.evaluate(th_final, rows_prob, pb_s.full)
    _, loss_f_final = pb_f.evaluate(th_final, rows_prob, pb_f.full)
    _, loss_f_bf = pb_f.evaluate(th_bf, rows_prob, pb_f.full)
    _, loss_s_bf = pb_s.evaluate(th_bf, rows_prob, pb_s.full)
    th_written = D.written_theta(th_final)
    _, loss_f_written = pb_f.evaluate(th_written, rows_prob, pb_f.full)
    _, loss_pub = pb_f.evaluate(truth, np.arange(p_n), pb_f.full)
    refined: dict[str, Any] = {}
    baseline: dict[str, Any] = {}
    if a.refine > 0:  # CMA-ES on the faithful model from the best faithful checkpoint

        def ev_f(zz: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return pb_f.evaluate(
                np.where(fr[rows], D.to_theta(sp, zz), fixed_th[rows]), rows_prob[rows], pb_f.full
            )

        def cma_from(z_start: np.ndarray, stream: int) -> dict[str, Any]:
            """CMA-ES (the staged-calibration ``cma_rows``, initial step SIGMA0, no restarts) from ``z_start`` on the
            faithful model, random stream ``stream``. ``cma_rows`` neither evaluates nor keeps its start,
            so the start is evaluated here (one more faithful forward per row) and kept where the search
            did not beat it: the refinement never returns a point worse than its start (``loss_kept``);
            the search's own best point is ``loss_raw``."""
            tracker = D.Tracker(r_n, time.perf_counter())
            t0 = time.perf_counter()
            rr = D.cma_rows(
                ev_f,
                z_start,
                range(6),
                tracker,
                max_evals=int(a.refine),
                tol=np.full(r_n, q_target * 1e-3 if a.kind == "twin" else 0.0),
                seed=a.seed + 7 + REFINE_STREAM_STRIDE * stream,
                restart=None,
            )
            th_r = theta_of(rr["z_best"])
            _, l_r = pb_f.evaluate(th_r, rows_prob, pb_f.full)
            th_0 = theta_of(z_start)
            _, l_0 = pb_f.evaluate(th_0, rows_prob, pb_f.full)
            keep = l_0 <= l_r
            th_k = np.where(keep[:, None], th_0, th_r)
            return {
                "stream": stream,
                "theta": th_k,
                "theta_written": D.written_theta(th_k),
                "loss_kept": np.minimum(l_0, l_r),
                "loss_raw": l_r,
                "loss_start": l_0,
                "start_kept": keep,
                "wall_s": time.perf_counter() - t0,
                "n_fwd": tracker.n_fwd + 1,
            }

        n_streams = max(int(a.refine_streams), 1)
        streams = [cma_from(best_fz, j) for j in range(n_streams)]
        r0 = streams[0]
        refined = {
            "evals": int(a.refine),
            "theta_refined": r0["theta"],
            "theta_refined_written": r0["theta_written"],
            "loss_faithful_refined": r0["loss_kept"],
            "loss_faithful_refined_raw": r0["loss_raw"],
            "wall_s": r0["wall_s"],
            "n_fwd": r0["n_fwd"],
            "incumbent_kept": True,
            "streams": [{k: v for k, v in r.items() if k != "theta_written"} for r in streams],
        }
        # the same CMA-ES with the same budget from the staged-calibration starts: the refinement's share without Adam
        base_streams = [cma_from(D.to_z(sp, th0), j) for j in range(n_streams)]
        b0 = base_streams[0]
        baseline = {
            "evals": int(a.refine),
            "theta": b0["theta"],
            "theta_written": b0["theta_written"],
            "loss_faithful": b0["loss_kept"],
            "loss_faithful_raw": b0["loss_raw"],
            "wall_s": b0["wall_s"],
            "n_fwd": b0["n_fwd"],
            "incumbent_kept": True,
            "streams": [{k: v for k, v in r.items() if k != "theta_written"} for r in base_streams],
        }
    t_end = time.perf_counter()
    res = {
        "method": "adam_smoothed",
        "self_twin": bool(a.self_twin),
        "width": width,
        "kind": a.kind,
        "seed": a.seed,
        "codes": list(codes),
        "env": D.env_info(),
        "problems": problems,
        "info": info,
        "truth": truth,
        "free": free,
        "rows_prob": rows_prob,
        "theta_start": th0,
        "theta_final": th_final,
        "theta_written": th_written,
        "theta_best_faithful": th_bf,
        "theta_best_faithful_written": D.written_theta(th_bf),
        "loss_smoothed_final": loss_s_final,
        "loss_faithful_final": loss_f_final,
        "loss_faithful_written": loss_f_written,
        "loss_faithful_best_faithful": loss_f_bf,
        "loss_smoothed_best_faithful": loss_s_bf,
        "loss_published_faithful": loss_pub,
        "refine": refined,
        "baseline_cma": baseline,
        "final_steps": best_steps,
        "best_faithful_steps": best_fsteps,
        "faithful_selection": {
            "min_adam_steps": FAITHFUL_MIN_STEPS,
            "rows_without_counted_checkpoint": int((best_fsteps < 0).sum()),
        },
        "hist": hist,
        "budget": {
            "forward_equivalents": budget,
            "jvp_over_fwd": cost,
            "iterations": n_iter,
            "segment": seg_len,
            "restart": a.restart,
            "forward_unit": "smoothed forward on the Adam batch (checkpoints, refine extra)",
            "jvp_calls": n_jvp,
            "faithful_checkpoint_calls": n_fwd_faithful,
            "refine_evals_per_row": int(a.refine),
        },
        "restarts": restarts,
        "nonfinite": {"iterations": nonfinite_iters, "per_row": nonfinite_rows},
        "adam": {"lr0": ADAM_LR0, "lr_min": ADAM_LR_MIN},
        "timing": {
            "process_s": t_end - t_start,
            "build_s": t_build - t_start,
            "optimise_s": t_adam - t_opt,
            "refine_s": t_end - t_adam,
            "smoothed": ev_s.stats(),
            "faithful": pb_f.ev.stats(),
        },
    }
    single = "_single" if a.restart == "none" else ""
    name = f"{a.kind}_{'self_' if a.self_twin else ''}adam{single}_w{width:g}_s{a.seed}.json"
    D.dump(out_root() / name, res)
    print(
        "wrote",
        name,
        f"process {t_end - t_start:.1f} s, non-finite iterations {nonfinite_iters}, "
        f"rows {int((nonfinite_rows > 0).sum())}",
        flush=True,
    )


def step_rescore(a: argparse.Namespace) -> None:
    """The self-twin results (every ``twin_self_*`` file) scored with hard dates: the observations and the
    candidates' dates are the smoothed model's hard dates (first day of the stage code) at the result's
    scale, the other outputs unchanged. A date residual is then 0 anywhere on the plateau of the right day,
    as in the faithful model, so a success here is comparable with the faithful twin's; on the soft dates
    of the self-twin objective 1e-4 needs a soft-date error of about 0.01 d. Writes
    ``self_twin_hard.json`` (per result stem: ``loss_hard`` per row)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    pb_f, problems, _, codes = _build("twin")
    files = [f for f in _results("twin") if "_self_" in f.name]
    by_w: dict[float, list[Path]] = {}
    for f in files:
        by_w.setdefault(float(json.loads(f.read_text())["width"]), []).append(f)
    out: dict[str, Any] = {}
    ev_h = None
    for w, fs in sorted(by_w.items()):
        if ev_h is None:
            ev_h = SmoothedEvaluator(
                pb_f.ev.items, {k: D._table([]) for k in pb_f.ev.index}, width=w, soft_dates=False
            )
        else:
            ev_h.set_width(w)
        pb_h = D.Problem(ev_h, D.Objective(D.twin_obs(ev_h, problems, codes), problems))
        for f in fs:
            res = json.loads(f.read_text())
            _, lh = pb_h.evaluate(np.asarray(res["theta_final"]), np.asarray(res["rows_prob"]), pb_h.full)
            out[f.stem] = {"width": w, "loss_hard": lh}
            print(f.stem, f"hard-date success {int((lh <= D.Q_TWIN).sum())}/{lh.size}", flush=True)
    D.dump(out_root() / "self_twin_hard.json", out)
    print("wrote self_twin_hard.json", flush=True)


# ============================================================================ holdout and DSSAT round trip
def _run_dssat(label: str, problems: dict[str, list[str]], theta_by_pn: dict[str, np.ndarray]) -> list[dict]:
    """dscsm048 (nitrogen off) on every treatment of ``problems`` with the cultivar ``theta_by_pn[pn]``
    written into a copy of MZCER048.CUL (the staged-calibration round trip)."""
    import test_ceres_dssat as m2

    from agrijax.io.dssat import read_summary
    from agrijax.io.dssat.cultivar_write import write_cultivar
    from agrijax.port.run_fortran import run_dscsm

    src = D.cul_source()
    items = {it["key"]: it for it in _all_items()}
    work = Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "g0b"
    out = []
    for p, (pn, ks) in enumerate(problems.items()):
        vals = dict(zip(D.PARAMS, np.asarray(theta_by_pn[pn]).tolist(), strict=True))
        pub = items[ks[0]]["cultivar"]
        cid = f"AJ{p + 1:04d}"
        dest = out_root() / "dssat" / label / pn.replace("/", "_") / "MZCER048.CUL"
        dest.parent.mkdir(parents=True, exist_ok=True)
        w = write_cultivar(src, dest, cid, f"AgriJAX {pub}", vals, base=pub, overwrite=True)
        for k in ks:
            it = items[k]
            dd = work / f"{label[:12]}_{k}"
            if dd.exists():
                shutil.rmtree(dd)
            dd.mkdir(parents=True)
            for f in m2.MAIZE.glob(it["exp"] + ".MZ*"):
                shutil.copy2(f, dd / f.name)
            x = dd / f"{it['exp']}.MZX"
            txt = D._filex_with_cultivar(m2._nitrogen_off(x.read_text(errors="replace")), pub, cid)
            shutil.copy2(dest, dd / "MZCER048.CUL")
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
            out.append(
                {
                    "problem": pn,
                    "key": k,
                    "cultivar_in_inp": cid in inp,
                    "written": w.written,
                    **{c: float(row[c]) for c in ("HWAM", "CWAM", "H#AM")},
                    "ADAT": _day_index(it, float(row["ADAT"])),
                    "MDAT": _day_index(it, float(row["MDAT"])),
                }
            )
    return out


def _results(kind: str) -> list[Path]:
    return sorted(out_root().glob(f"{kind}_*_s*.json"))


def _best_rows(res: dict[str, Any]) -> dict[str, int]:
    """Per problem: the row with the lowest objective of the method's own model (smoothed for Adam)."""
    rows_prob = np.asarray(res["rows_prob"])
    loss = np.asarray(res["loss_smoothed_final"] if "loss_smoothed_final" in res else res["loss_final"])
    out = {}
    for p, pn in enumerate(res["problems"]):
        rows = np.nonzero(rows_prob == p)[0]
        out[pn] = int(rows[np.nanargmin(loss[rows])])
    return out


def _candidates(res: dict[str, Any], label: str) -> dict[str, dict[str, np.ndarray]]:
    """``{label: {problem: written coefficients}}`` of a result: its best row per problem on the method's
    own objective; for Adam also the best faithful checkpoint (``:faithful_sel``, selected on the faithful
    objective) and the CMA-ES refinement (``:refined``) when present."""
    rows_prob = np.asarray(res["rows_prob"])
    sets = [("", "theta_written", res.get("loss_smoothed_final", res.get("loss_final")))]
    if "theta_best_faithful_written" in res:
        sets.append((":faithful_sel", "theta_best_faithful_written", res["loss_faithful_best_faithful"]))
    if res.get("refine"):
        sets.append((":refined", None, res["refine"]["loss_faithful_refined"]))
    if res.get("baseline_cma"):  # the same CMA-ES from the staged-calibration starts (one label per seed)
        sets.append(("@", "baseline", res["baseline_cma"]["loss_faithful"]))
    out: dict[str, dict[str, np.ndarray]] = {}
    for suffix, key, loss in sets:
        if key is None:
            th = np.asarray(res["refine"]["theta_refined_written"])
        elif key == "baseline":
            th = np.asarray(res["baseline_cma"]["theta_written"])
        else:
            th = np.asarray(res[key])
        loss = np.asarray(loss, dtype=float)
        sel = {}
        for p, pn in enumerate(res["problems"]):
            rows = np.nonzero(rows_prob == p)[0]
            sel[pn] = th[rows[np.nanargmin(loss[rows])]]
        out[f"{BASELINE_LABEL}_s{res['seed']}" if suffix == "@" else label + suffix] = sel
    return out


def step_holdout(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, str(REPO / "tests" / "integration"))
    report: dict[str, Any] = {}
    # ---- UFGA8201: t04 calibrated, t06 held out
    pb4, prob4, _, _ = _build("ufga4")
    pb6, prob6, _, _ = _build("ufga6")
    pn = next(iter(prob4))
    pub = D.truth_of(pb4)[0]
    cands: dict[str, np.ndarray] = {"published": pub}
    widths: dict[str, float | None] = {"published": None}
    for f in _results("ufga4"):
        res = json.loads(f.read_text())
        for label, sel in _candidates(res, f.stem.replace("ufga4_", "")).items():
            cands[label] = sel[pn]
            widths[label] = res.get("width") if ":" not in label and BASELINE_LABEL not in label else None
    ev_s4 = SmoothedEvaluator(pb4.ev.items, {k: D._table([]) for k in pb4.ev.index}, width=WIDTHS[0])
    ev_s6 = SmoothedEvaluator(pb6.ev.items, {k: D._table([]) for k in pb6.ev.index}, width=WIDTHS[0])
    pbs4, pbs6 = D.Problem(ev_s4, pb4.obj), D.Problem(ev_s6, pb6.obj)
    for label, th in cands.items():
        rec: dict[str, Any] = {"theta": th, "width": widths[label]}
        rec["faithful_t04"] = D.per_code(pb4, th, 0)
        rec["faithful_t06"] = D.per_code(pb6, th, 0)
        if widths[label] is not None:
            ev_s4.set_width(float(widths[label]))  # type: ignore[arg-type]
            ev_s6.set_width(float(widths[label]))  # type: ignore[arg-type]
            rec["smoothed_t04"] = D.per_code(pbs4, th, 0)
            rec["smoothed_t06"] = D.per_code(pbs6, th, 0)
        both = {pn: [*prob4[pn], *prob6[pn]]}
        rec["dssat"] = _run_dssat(f"ufga_{label.replace(':', '_')}", both, {pn: th})
        report[label] = rec
        print("ufga", label, "done", flush=True)
    obs = {}
    for k, pb in (("UFGA8201_t04", pb4), ("UFGA8201_t06", pb6)):
        cc = pb.obj.codes[k]
        obs[k] = {c: pb.obj.obs.obs[k][cc == c].tolist() for c in sorted({c for c in cc if c})}
    report_all: dict[str, Any] = {"ufga": report, "ufga_obs": obs}
    # ---- twin: the best row of every problem in dscsm048, against dscsm048 at the truth
    pbt, probt, _, _ = _build("twin")
    truth = D.truth_of(pbt)
    tw: dict[str, Any] = {"truth": _run_dssat("twin_truth", probt, dict(zip(probt, truth, strict=True)))}
    for f in _results("twin"):
        if "_self_" in f.name:
            continue
        res = json.loads(f.read_text())
        for label, sel in _candidates(res, f.stem.replace("twin_", "")).items():
            if label in tw:  # the baseline: the same for every Adam run of a seed
                continue
            tw[label] = _run_dssat(f"twin_{label.replace(':', '_')}", probt, sel)
            print("twin", label, "done", flush=True)
    report_all["twin_dssat"] = tw
    D.dump(out_root() / "holdout.json", report_all)
    print("wrote holdout.json", flush=True)


# ============================================================================ tables
def _f(x: float, d: int = 2) -> str:
    return D._fmt(float(x), d)


def _forward_table(lines: list[str]) -> None:
    r = json.loads((out_root() / "forward.json").read_text())
    names = r["names"]
    keep = np.asarray([not k.startswith("GAGR") for k in r["keys"]])
    f = np.asarray(r["faithful"])[keep]
    dss = np.asarray(r["dscsm048"], dtype=float)[keep]
    lines += [
        f"### (a) Forward departure at the published cultivars ({int(keep.sum())} treatments)",
        "",
        "dates: mean |diff| / max |diff| / mean diff [d]; yield, biomass, grain number: mean |rel| / "
        "max |rel| / mean rel [%]; against dscsm048",
        "",
        "| model | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]

    def row(y: np.ndarray) -> str:
        out = []
        for j in range(len(names)):
            a, b = y[:, j], dss[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if j < 3:
                d = a[ok] - b[ok]
                out.append(f"{np.mean(np.abs(d)):.2f} / {np.max(np.abs(d)):.0f} / {np.mean(d):+.2f}")
            else:
                ok &= b > 0
                d = (a[ok] - b[ok]) / b[ok] * 100
                out.append(f"{np.mean(np.abs(d)):.1f} / {np.max(np.abs(d)):.1f} / {np.mean(d):+.1f}")
        return " | ".join(out)

    lines.append(f"| faithful | {row(f)} |")
    for mode in ("hard", "soft"):
        if mode in r["smoothed_1e-12"]:
            y0 = np.asarray(r["smoothed_1e-12"][mode])[keep]
            lines.append(f"| smoothed, scale 1e-12 (limit), {mode} dates | {row(y0)} |")
    label = {"hard": "hard dates", "soft": "soft dates", "soft_real_days": "soft dates, real days only"}
    for w in WIDTHS:
        for mode in ("hard", "soft", "soft_real_days"):
            if mode in r[f"smoothed_{w:g}"]:
                y = np.asarray(r[f"smoothed_{w:g}"][mode])[keep]
                lines.append(f"| smoothed {w:g} degC d, {label[mode]} | {row(y)} |")
    lines.append("")


def _grad_table(lines: list[str]) -> None:
    g = json.loads((out_root() / "grad.json").read_text())
    outs = [n for n, _, _ in GRAD_OUTPUTS]
    # GAGR0201 left out as in the forward table: its silking, maturity and final values are read on
    # padded days (a 74-day growth-chamber season; dscsm048 has no silking or maturity there)
    keys = sorted({x["key"] for x in next(iter(g.values()))["rows"]})
    kept = [k for k in keys if not k.startswith("GAGR")]
    lines += [
        f"### (b) Gradient quality at the published cultivars ({len(kept)} treatments; GAGR0201 left out, "
        "padded-day artefact)",
        "",
        "Per (coefficient, output): share of treatments at trust level 3 (AD agrees with the 1, 2 and 5 % "
        "secants within 5 %, no unexplained jump on the +-10 % scan); in brackets the median relative "
        "AD / 2 % secant disagreement. Inert pairs (output does not move) count as level 3 and are listed "
        "in the class table.",
        "",
    ]
    for res in g.values():
        rows = [x for x in res["rows"] if not x["key"].startswith("GAGR")]
        lines += [
            f"**{res['label']}**",
            "",
            "| coefficient | " + " | ".join(outs) + " |",
            "|---|" + "---|" * len(outs),
        ]
        for p in D.PARAMS:
            cells = []
            for o in outs:
                rr = [x for x in rows if x["param"] == p and x["output"] == o]
                l3 = np.mean([x["level"] == 3 for x in rr])
                med = np.median([x["rel_err_2pct"] for x in rr])
                inert = np.mean([x["class"] == "inert" for x in rr])
                cells.append(f"{l3:.2f} ({med:.2g}){' i' + format(inert, '.2f') if inert > 0 else ''}")
            lines.append(f"| {p} | " + " | ".join(cells) + " |")
        nf = res.get("nonfinite")
        if nf:
            lines += [
                "",
                f"non-finite AD (all {len(keys)} treatments, GAGR0201 included): {nf['treatments_ad']} "
                f"treatments ({nf['values_ad']} of {nf['values']} values; scan points {nf['scan_points_ad']} "
                f"of {nf['scan_points']})",
            ]
        cls: dict[str, int] = {}
        for x in rows:
            cls[x["class"]] = cls.get(x["class"], 0) + 1
        lines += ["", "classes: " + ", ".join(f"{k} {v}" for k, v in sorted(cls.items())), ""]


def _method_of(stem: str, kind: str) -> tuple[str, int]:
    """``(method label without the seed, seed)`` of a result file stem."""
    body = stem.replace(f"{kind}_", "", 1)
    head, _, seed = body.rpartition("_s")
    return head, int(seed)


def _budget(res: dict[str, Any], refined: bool = False) -> str:
    """Per-row budget of one result. Adam: iterations x the timed JVP / forward ratio = smoothed-model
    forwards on its own batch, the faithful checkpoint forwards (selection) separately, and the CMA-ES
    refinement only for the refined variant; staged-calibration methods: faithful (or, on the self-twin, smoothed)
    forwards and JVPs per row (median)."""
    if "budget" in res:
        b = res["budget"]
        txt = (
            f"{b['iterations']} it x {b['jvp_over_fwd']:.2f} = {b['iterations'] * b['jvp_over_fwd']:.0f} sm"
            f" + {b['faithful_checkpoint_calls']} F ckpt"
        )
        if refined and b.get("refine_evals_per_row"):
            txt += f" + {b['refine_evals_per_row']} F CMA"
        return txt
    fwd = np.asarray(res["n_fwd"], dtype=float)
    jvp = np.asarray(res["n_jvp"], dtype=float)
    txt = f"{np.median(fwd):.0f} fwd"
    if jvp.sum() > 0:
        txt += f" + {np.median(jvp):.0f} JVP"
    return txt


def _self_hard() -> dict[str, Any]:
    f = out_root() / "self_twin_hard.json"
    return json.loads(f.read_text()) if f.exists() else {}


#: label of the CMA-ES run from the staged-calibration starts with the refinement's budget (one per seed and kind)
BASELINE_LABEL = "cma_refine_budget_from_starts"


def _twin_rows(self_twin: bool) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = {}
    hard = _self_hard() if self_twin else {}
    baseline_seeds: set[int] = set()
    for f in _results("twin"):
        if ("_self_" in f.name) != self_twin:
            continue
        res = json.loads(f.read_text())
        m, seed = _method_of(f.stem, "twin")
        truth = np.asarray(res["truth"])
        rp = np.asarray(res["rows_prob"])
        free = np.asarray(res["free"])[rp]
        is_adam = "loss_smoothed_final" in res
        if self_twin:  # the method's own objective is the smoothed one (Adam and the staged-calibration methods alike)
            ls = res["loss_smoothed_final"] if is_adam else res["loss_final"]
            variants = [("", np.asarray(res["theta_final"]), None, ls)]
        else:
            variants = [
                (
                    "",
                    np.asarray(res["theta_final"]),
                    res.get("loss_faithful_final", res.get("loss_final")),
                    res.get("loss_smoothed_final"),
                )
            ]
            if "theta_best_faithful" in res:
                variants.append(
                    (
                        " [faithful sel]",
                        np.asarray(res["theta_best_faithful"]),
                        res["loss_faithful_best_faithful"],
                        res.get("loss_smoothed_best_faithful"),
                    )
                )
            if res.get("refine"):
                variants.append(
                    (
                        " [+CMA refine]",
                        np.asarray(res["refine"]["theta_refined"]),
                        res["refine"]["loss_faithful_refined"],
                        None,  # the smoothed objective of the refined coefficients is not computed
                    )
                )
        base = {} if self_twin else (res.get("baseline_cma") or {})  # faithful twin only
        if base and seed not in baseline_seeds:  # the same for every Adam run of a seed (same starts)
            baseline_seeds.add(seed)
            variants.append(("@" + BASELINE_LABEL, np.asarray(base["theta"]), base["loss_faithful"], None))
        for suffix, th, lf, ls in variants:
            crit = np.asarray(ls if self_twin else lf, dtype=float)
            rec = (D.recovery(th, truth[rp]) <= D.REC_TOL) & free
            lh = hard.get(f.stem, {}).get("loss_hard")
            is_base = suffix.startswith("@")
            by.setdefault(suffix[1:] if is_base else m + suffix, []).append(
                {
                    "seed": seed,
                    "success": int((crit <= D.Q_TWIN).sum()),
                    "success_hard": None if lh is None else int((np.asarray(lh) <= D.Q_TWIN).sum()),
                    "lh_med": None if lh is None else float(np.median(lh)),
                    "n": int(crit.size),
                    "lf_med": None if lf is None else float(np.median(np.asarray(lf, dtype=float))),
                    "ls_med": None if ls is None else float(np.median(np.asarray(ls, dtype=float))),
                    "rec": [int(rec[:, i].sum()) for i in range(6)],
                    "free": [int(free[:, i].sum()) for i in range(6)],
                    "wall": float(base["wall_s"] if is_base else res["timing"]["process_s"]),
                    "opt": float(base["wall_s"] if is_base else res["timing"]["optimise_s"]),
                    "budget": f"{base['evals']} F CMA"
                    if is_base
                    else _budget(res, refined=suffix == " [+CMA refine]"),
                    "nonfinite": None if is_base else res.get("nonfinite"),
                }
            )
    return by


def _med_or_dash(rs: list[dict[str, Any]], k: str) -> str:
    x = [r[k] for r in rs if r[k] is not None]
    return _f(np.median(x)) if x else "-"


def _twin_block(lines: list[str], title: str, by: dict[str, list[dict[str, Any]]], self_twin: bool) -> None:
    if self_twin:
        crit = (
            "success: the method's own (smoothed, soft-date) objective <= 1e-4; success hard: the same "
            "self-twin scored with the hard dates of the smoothed model (observations and candidates; a "
            "date residual is then 0 on the plateau of the right day, as in the faithful model)"
        )
        head = "| method | seeds | success per seed | success hard per seed | smoothed objective median | hard-date objective median | "  # noqa: E501
        ncol = 11
    else:
        crit = (
            "success: faithful objective <= 1e-4 (the smoothed calibrations are judged in the faithful "
            "model: their model bias)"
        )
        head = (
            "| method | seeds | success per seed | faithful objective median | smoothed objective median | "
        )
        ncol = 10
    lines += [
        title,
        "",
        f"{crit}; recovered: free coefficient within 2 % of the truth (sum over seeds); budget per row per "
        "seed (sm = smoothed-model forwards on the Adam batch, F ckpt = faithful checkpoint forwards, F CMA "
        "= faithful refinement evaluations); wall: whole process, mean over seeds, compile included; opt: "
        "optimisation loop only",
        "",
        head + "recovered P1 / P2 / P5 / PHINT / G2 / G3 | budget per row (per seed) | wall / opt [s] | "
        "non-finite gradient iterations |",
        "|" + "---|" * ncol,
    ]
    for m, rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        succ = ", ".join(f"{r['success']}/{r['n']}" for r in rs)
        rec = " / ".join(f"{sum(r['rec'][i] for r in rs)}/{sum(r['free'][i] for r in rs)}" for i in range(6))
        nf = [r["nonfinite"]["iterations"] for r in rs if r["nonfinite"]]
        bud = "; ".join(r["budget"] for r in rs)
        if len({r["budget"] for r in rs}) == 1:
            bud = rs[0]["budget"] + " (each)"
        wall = f"{np.mean([r['wall'] for r in rs]):.0f} / {np.mean([r['opt'] for r in rs]):.0f}"
        if self_twin:
            sh = [r["success_hard"] for r in rs]
            shtxt = ", ".join(f"{x}/{r['n']}" for x, r in zip(sh, rs, strict=True)) if None not in sh else "-"
            mid = f"{succ} | {shtxt} | {_med_or_dash(rs, 'ls_med')} | {_med_or_dash(rs, 'lh_med')}"
        else:
            mid = f"{succ} | {_med_or_dash(rs, 'lf_med')} | {_med_or_dash(rs, 'ls_med')}"
        lines.append(
            f"| {m} | {len(rs)} | {mid} | {rec} | {bud} | {wall} | {', '.join(map(str, nf)) if nf else '-'} |"
        )
    lines.append("")


def _cma_at(res: dict[str, Any], budget: float) -> np.ndarray:
    """Per row: the best faithful objective of a staged-calibration CMA-ES result after at most ``budget`` forwards of
    that row (from its ``hist``; ``inf`` before the first record)."""
    best = np.full(len(res["rows_prob"]), np.inf)
    for _, f, _, b in res["hist"]:
        ok = np.asarray(f, dtype=float) <= budget
        best = np.where(ok, np.minimum(best, np.asarray(b, dtype=float)), best)
    return best


def _refine_table(lines: list[str]) -> None:
    """(c1c): per ordinary-twin Adam run and seed, the faithful selection, the refinement per random
    stream (start kept / the search's own best point), its baseline from the staged-calibration starts per stream, and
    the staged-calibration CMA-ES (with restarts) at the same faithful evaluation count as the refined pipeline
    (refinement + start evaluation + Adam's faithful checkpoints)."""
    q = D.Q_TWIN
    rows: list[str] = []
    for f in _results("twin"):
        if "_self_" in f.name or "_adam" not in f.name:
            continue
        res = json.loads(f.read_text())
        ref = res.get("refine") or {}
        if not ref.get("streams"):
            continue
        m, seed = _method_of(f.stem, "twin")
        cma_f = out_root() / f"twin_cma_s{seed}.json"
        n = len(res["rows_prob"])

        def cnt(x: Any) -> int:
            return int((np.asarray(x, dtype=float) <= q).sum())

        sel = cnt(res["loss_faithful_best_faithful"])
        kept = [cnt(r["loss_kept"]) for r in ref["streams"]]
        raw = [cnt(r["loss_raw"]) for r in ref["streams"]]
        lost = [
            int(
                (
                    (np.asarray(res["loss_faithful_best_faithful"]) <= q) & (np.asarray(r["loss_raw"]) > q)
                ).sum()
            )
            for r in ref["streams"]
        ]
        worse = [int((np.asarray(r["loss_raw"]) > np.asarray(r["loss_start"])).sum()) for r in ref["streams"]]
        base = res.get("baseline_cma") or {}
        bk = [cnt(r["loss_kept"]) for r in base.get("streams", [])]
        matched = ref["evals"] + 1 + res["budget"]["faithful_checkpoint_calls"]
        d31 = "-"
        if cma_f.exists():
            cres = json.loads(cma_f.read_text())
            if len(cres["rows_prob"]) == n:
                d31 = f"{cnt(_cma_at(cres, matched))} ({matched} F)"
        rows.append(
            f"| {m} | {seed} | {sel} | {', '.join(map(str, kept))} (mean {np.mean(kept):.1f}) | "
            f"{', '.join(map(str, raw))} | {', '.join(map(str, worse))} | {', '.join(map(str, lost))} | "
            f"{', '.join(map(str, bk)) or '-'} | {d31} |"
        )
    if not rows:
        return
    lines += [
        "### (c1c) Ordinary twin: refinement per random stream and the staged-calibration CMA-ES at matched faithful "
        "evaluations (successes of 88)",
        "",
        "sel: Adam's faithful selection; refined: CMA-ES from the selection, start kept where not beaten, "
        "one count per random stream; raw: the search's own best point (no start kept); worse: rows whose "
        "raw point is worse than the start; lost: selection successes the raw point loses; baseline: the "
        "same CMA-ES from the staged-calibration starts, per stream; staged-calibration CMA-ES: the staged-calibration run (restarts) at the refined "
        "pipeline's faithful evaluation count (refinement + start + Adam's checkpoints)",
        "",
        "| Adam run | seed | sel | refined per stream | raw per stream | worse than start | lost | "
        "baseline per stream | staged-calibration CMA-ES matched |",
        "|---|---|---|---|---|---|---|---|---|",
        *sorted(rows),
        "",
    ]


def _twin_table(lines: list[str]) -> None:
    _twin_block(
        lines,
        "### (c1a) Ordinary twin: observations from the faithful model (11 problems x 8 starts = 88 rows)",
        _twin_rows(False),
        False,
    )
    _twin_block(
        lines,
        "### (c1b) Self-twin: observations from the smoothed model itself (no model bias; optimiser only; "
        "every method on the same smoothed objective, starts and nominal budget)",
        _twin_rows(True),
        True,
    )
    _refine_table(lines)


def _ufga_table(lines: list[str]) -> None:
    h = json.loads((out_root() / "holdout.json").read_text())
    obs = h["ufga_obs"]
    o6 = obs["UFGA8201_t06"]
    lines += [
        "### (c2) UFGA8201: calibrated on t04 (ADAT, MDAT, HWAM, CWAM, LAI, H#AM), t06 held out",
        "",
        "best row per method and seed (``[faithful sel]``: selected on the faithful objective; ``[+CMA "
        "refine]``: then refined by CMA-ES on the faithful model); written coefficients; per method the median "  # noqa: E501
        f"over seeds and in brackets the range. Observed t06: HWAM {o6['HWAM'][0]:.0f}, H#AM {o6['H#AM'][0]:.0f}, "  # noqa: E501
        f"ADAT {o6['ADAT'][0]:.0f}, MDAT {o6['MDAT'][0]:.0f} (day indices). F = faithful model, DSSAT = dscsm048.",  # noqa: E501
        "",
        "| method | seeds | t04 objective F | t06 HWAM F | t06 HWAM DSSAT | t06 H#AM DSSAT | t06 ADAT / MDAT error "  # noqa: E501
        "DSSAT [d] | t06 ADAT / MDAT error F [d] | F vs DSSAT max yield diff [%] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for label, rec in h["ufga"].items():
        m = label
        if "_s" in label.split(":")[0]:
            head, _, rest = label.partition(":")
            base, _, _seed = head.rpartition("_s")
            m = base + (":" + rest if rest else "")
        t4 = sum(v["loss"] for v in rec["faithful_t04"]["UFGA8201_t04"].values())
        f6 = rec["faithful_t06"]["UFGA8201_t06"]
        ds = {d["key"]: d for d in rec["dssat"]}
        d6 = ds["UFGA8201_t06"]
        fy = [
            abs(rec[f"faithful_{t}"][f"UFGA8201_{t}"]["HWAM"]["sim"][0] - ds[f"UFGA8201_{t}"]["HWAM"])
            / max(ds[f"UFGA8201_{t}"]["HWAM"], 1.0)
            * 100
            for t in ("t04", "t06")
        ]
        groups.setdefault(m, []).append(
            {
                "t4": t4,
                "hf": f6["HWAM"]["sim"][0],
                "hd": d6["HWAM"],
                "gd": d6["H#AM"],
                "ad": d6["ADAT"] - o6["ADAT"][0],
                "md": d6["MDAT"] - o6["MDAT"][0],
                "af": f6["ADAT"]["sim"][0] - o6["ADAT"][0],
                "mf": f6["MDAT"]["sim"][0] - o6["MDAT"][0],
                "fy": max(fy),
            }
        )

    def med(rs: list[dict[str, Any]], k: str, d: int = 3) -> str:
        """Median (range) over the seeds; a NaN (a dscsm048 date outside the reference run's days) is
        counted, never dropped silently."""
        x = np.asarray([r[k] for r in rs], dtype=float)
        nan = int(np.isnan(x).sum())
        if nan == x.size:
            return f"n/a ({nan} of {x.size} outside the reference run)"
        if x.size == 1:
            return _f(x[0], d)
        txt = f"{_f(np.nanmedian(x), d)} ({_f(np.nanmin(x), d)}-{_f(np.nanmax(x), d)})"
        if nan:
            txt += f", {nan} of {x.size} outside the reference run"
        return txt

    for m, rs in sorted(groups.items()):
        lines.append(
            f"| {m} | {len(rs)} | {med(rs, 't4')} | {med(rs, 'hf', 4)} | {med(rs, 'hd', 4)} | {med(rs, 'gd', 4)} | "  # noqa: E501
            f"{med(rs, 'ad', 1)} / {med(rs, 'md', 1)} | {med(rs, 'af', 1)} / {med(rs, 'mf', 1)} | "
            f"{max(r['fy'] for r in rs):.2f} |"
        )
    lines.append("")
    tw = h["twin_dssat"]
    truth = {d["key"]: d for d in tw["truth"]}
    lines += [
        "### (c3) Ordinary twin, best row per problem in dscsm048 against dscsm048 at the truth",
        "",
        "dates outside the reference run's days (no day index) are counted separately, not as wrong",
        "",
        "| method (seed) | HWAM mean / max abs rel [%] | ADAT, MDAT exact | wrong | outside the reference days |",  # noqa: E501
        "|---|---|---|---|---|",
    ]
    for label, rows in sorted(tw.items()):
        if label == "truth":
            continue
        rel = [
            abs(d["HWAM"] - truth[d["key"]]["HWAM"]) / max(truth[d["key"]]["HWAM"], 1.0) * 100 for d in rows
        ]
        outside = sum(not (np.isfinite(d["ADAT"]) and np.isfinite(d["MDAT"])) for d in rows)
        ex = sum(
            np.isfinite(d["ADAT"])
            and np.isfinite(d["MDAT"])
            and d["ADAT"] == truth[d["key"]]["ADAT"]
            and d["MDAT"] == truth[d["key"]]["MDAT"]
            for d in rows
        )
        lines.append(
            f"| {label} | {np.mean(rel):.2f} / {np.max(rel):.2f} | {ex}/{len(rows)} | "
            f"{len(rows) - ex - outside} | {outside} |"
        )
    lines.append("")


def step_table(a: argparse.Namespace) -> None:
    lines: list[str] = []
    for fn in (_forward_table, _grad_table, _twin_table, _ufga_table):
        try:
            fn(lines)
        except FileNotFoundError as e:
            lines += [f"(missing: {e.filename})", ""]
    text = "\n".join(lines)
    (out_root() / "tables.md").write_text(text)
    print(text)


# ============================================================================ main
def main() -> None:
    _setup_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prep", "forward", "grad", "adam", "d31", "rescore", "holdout", "table"))
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--kind", default="twin", choices=("twin", "ufga4"))
    ap.add_argument("--method", default="cma")
    ap.add_argument("--width", type=float, default=17.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=D.CMA_ALL_EVALS, help="adam: forward equivalents per row")
    ap.add_argument(
        "--iters",
        type=int,
        default=0,
        help="adam: fixed iteration count (overrides --budget; reproduces a run with its budget.iterations)",
    )
    ap.add_argument(
        "--restart",
        default="cold",
        choices=("cold", "none"),
        help="adam: cold restarts every ADAM_SEGMENT iterations, or one trajectory over the whole budget",
    )
    ap.add_argument(
        "--refine", type=int, default=0, help="adam: CMA-ES evaluations per row on the faithful model"
    )
    ap.add_argument(
        "--refine-streams",
        type=int,
        default=REFINE_STREAMS,
        help="adam: independent random streams of the refinement and its baseline (noise of the count)",
    )
    ap.add_argument(
        "--self-twin", action="store_true", help="adam, d31: twin observations from the smoothed model"
    )
    ap.add_argument(
        "--smoothed-model",
        action="store_true",
        help="d31: the staged-calibration method on the smoothed model against the ordinary twin (faithful observations)",
    )
    a = ap.parse_args()
    {
        "prep": step_prep,
        "forward": step_forward,
        "grad": step_grad,
        "adam": step_adam,
        "d31": step_d31,
        "rescore": step_rescore,
        "holdout": step_holdout,
        "table": step_table,
    }[a.step](a)


if __name__ == "__main__":
    main()
