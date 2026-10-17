"""Soil parameters written into the free-run DSSAT day (``calib/day_soil.py``) on two DSSAT examples,
against the native rebuild and against ``dscsm048``.

For UFGA8201 treatment 2 (rainfed, the Millhopper sand IBMZ910014: eight ``.SOL`` layers, nine model
layers) and IBWA8301 treatment 3 (IBMZ910013: seven ``.SOL`` layers, nine model layers), five soils
are drawn in boxes around the site's ``.SOL`` values (``calib.soil.soil_space`` with
``SITE_BOUNDS``: CN, SWCON, U, SLPF, LL, DUL and SAT of every layer, SRGF of every layer with a
nonzero value, the top one included; site values on a bound accepted, the draws stay in the inner
half of each box), rounded to the precision ``DSSAT48.INP`` prints, and written to a
``.SOL`` (``write_sol``). For each soil:

* ``dscsm048`` runs the treatment on that ``.SOL`` (nitrogen off);
* **independent check, bit for bit**: ``with_soil`` on the original run's trees equals the trees
  the native chain builds from the new ``.SOL`` and the new run's ``DSSAT48.INP`` (whose soil the
  native chain checks against its own, layer by layer): every parameter, the morning state and the
  forcing of the new season;
* **against dscsm048**: the season simulated from ``with_soil``'s trees has the harvest-day grain
  weight within 2 % of ``HWAM``, the emergence, silking and maturity days of ``Summary.OUT``, and the
  layer soil water of ``SoilWat.OUT`` (printed to 0.001) within :data:`SW_HALF_STEP` on every
  layer-day: the print half-step 0.0005 plus the free run's measured end-of-day water difference
  (``SECONDARY_BOUNDS["m2"]["dump.sw_end_of_day_max_abs"]``, 9.2e-5).

  **One named exception, recognised by its cause, not by the soil's index**: a soil may exceed
  :data:`SW_HALF_STEP` only if, at the start of its first layer-day beyond it (the end of the day
  before, or the initial water), one of its layers sits at ``SATFLO``'s drainage threshold
  ``DUL + 0.003`` (``SATFLO.for``; the bucket's ``dul_margin``): our water within
  :data:`THRESHOLD_TOL` of it. Such a tie can fall either way on a new soil (the free run's
  end-of-day water differs from the reference by up to 9e-5, and soils rounded to 0.001 put the
  threshold on a FileX initial water now and then), so DSSAT drains the layer's excess and we keep
  it, or the reverse; the difference then stays within :data:`SW_FLIP_MAX`. Measured: IBWA8301
  treatment 3 soil 0 has DUL 0.367 in model layers 4 and 5, whose initial water is 0.370: max
  0.0038 on 41 layer-days, yield 0.007 % off, dates unchanged; the other nine soils at most
  0.000505. The bit-for-bit native-rebuild check guards ``with_soil`` itself, so this check only
  has to tell a threshold flip of the model from a ``with_soil`` error.

The fifth soil of IBWA8301 moves ``LL`` above and ``SAT`` below the initial water, so ``SOILDYN``'s
limits on the initial water are compared with ``dscsm048`` too: its limited initial water lies
above the FileX value in one layer and below it in another. The comparison numbers are written to
``<data-dir>/validation/day_soil/with_soil_dscsm048.json``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import day_dssat486_free_harness as h
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib.day_soil import soil_layer_map, with_soil
from agrijax.calib.soil import SITE_BOUNDS, soil_space
from agrijax.core.runtime import run_sites
from agrijax.io.dssat import read_filex, read_soilwat, read_summary, write_sol
from agrijax.io.dssat.native_management import treatment_levels
from agrijax.io.dssat.native_soil import find_soil_profile
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
from agrijax.processes.soil_water.bucket.coefficients import WATBAL_COEFFICIENTS
from agrijax.sites.dssat_free_run import _season_model, free_run_inputs, run_reference, stack_trees

pytestmark = [
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the DSSAT example data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
SOIL = DSSAT_ENGINE / "example_data" / "Soil"
CASES = [("UFGA8201", 2), ("IBWA8301", 3)]
N_SOILS = 5
EXTEND_DAYS = 60
YIELD_REL = 0.02
#: half the print step of SoilWat.OUT's SW (F8.3)
PRINT_HALF_STEP = 0.0005
#: SoilWat.OUT prints SW to 0.001 (F8.3): half a step, plus the free run's measured end-of-day
#: water difference from the reference (it can move a value across a rounding tie)
SW_HALF_STEP = PRINT_HALF_STEP + h.SECONDARY_BOUNDS["m2"]["dump.sw_end_of_day_max_abs"][0]
#: a layer's water this close to DUL + dul_margin sits at the SATFLO threshold (a few REAL*4 quanta
#: of the water content; module docstring)
THRESHOLD_TOL = 5e-6
#: largest layer water difference after a SATFLO tie: twice the threshold's margin (the excess the
#: tied layer drains or keeps, and the layer below that receives it); measured 0.0038
SW_FLIP_MAX = 2 * WATBAL_COEFFICIENTS.dul_margin
#: the soil whose LL and SAT bracket the initial water (``_clip_soil``)
CLIP_CASE = ("IBWA8301_t03", N_SOILS - 1)
#: decimals ``DSSAT48.INP`` prints each quantity with (``optempy2k.for`` 505-560, 810)
DECIMALS = {"cn": 0, "u": 1, "swcon": 2, "slpf": 2, "ll": 3, "dul": 3, "sat": 3, "srgf": 3}
COLUMN = {"ll": "SLLL", "dul": "SDUL", "sat": "SSAT", "srgf": "SRGF"}
SURFACE = {"cn": "SLRO", "swcon": "SLDR", "u": "SLU1", "slpf": "SLPF"}


def _draws(site: dict[str, np.ndarray], seed: int) -> list[dict[str, np.ndarray]]:
    """Soils drawn in the inner half of the site boxes, rounded to the printed precision (site
    values on a bound accepted: SRGF 1.0 of the top layers, SLPF 1.0 of IBMZ910013)."""
    n = site["ll"].size
    srgf = [h_ for h_ in range(n) if site["srgf"][h_] > 0]
    sp = soil_space(
        site,
        ("cn", "swcon", "u", "slpf", "ll", "dul", "sat", "srgf"),
        {"ll": range(n), "dul": range(n), "sat": range(n), "srgf": srgf},
        SITE_BOUNDS,
        site_on_bound=True,
    )
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < N_SOILS:
        theta = np.asarray(
            sp.to_physical(jnp.asarray(rng.uniform(-1.1, 1.1, sp.n)))
        )  # sigmoid in [0.25, 0.75]
        soil = {k: np.array(site[k], dtype=float, copy=True) for k in DECIMALS}
        for s, v in zip(sp.specs, theta, strict=True):
            if s.index is None:
                soil[s.path] = np.asarray(v)
            else:
                soil[s.path][s.index] = v
        soil = {k: np.round(v, DECIMALS[k]) for k, v in soil.items()}
        if np.all(soil["dul"] > soil["ll"]) and np.all(soil["sat"] > soil["dul"]):
            out.append(soil)
    return out


def _clip_soil(site: dict[str, np.ndarray], sw_input: np.ndarray, layer_of: np.ndarray) -> dict[str, Any]:
    """A soil with LL above the initial water in one deep layer and SAT below it in another."""
    soil = {k: np.array(site[k], dtype=float, copy=True) for k in DECIMALS}
    sw = {int(layer_of[i]): float(sw_input[i]) for i in range(sw_input.size) if np.isfinite(sw_input[i])}
    lo, hi = 2, 4  # .SOL layers 3 and 5
    soil["ll"][lo] = round(sw[lo] + 0.01, 3)
    soil["dul"][lo] = max(soil["dul"][lo], round(soil["ll"][lo] + 0.03, 3))
    soil["sat"][lo] = max(soil["sat"][lo], round(soil["dul"][lo] + 0.05, 3))
    soil["sat"][hi] = round(sw[hi] - 0.02, 3)
    soil["dul"][hi] = min(soil["dul"][hi], round(soil["sat"][hi] - 0.03, 3))
    soil["ll"][hi] = min(soil["ll"][hi], round(soil["dul"][hi] - 0.05, 3))
    return soil


def _profile_with(prof: Any, soil: dict[str, np.ndarray]) -> Any:
    p = copy.deepcopy(prof)
    for k, col in COLUMN.items():
        p.layers[col] = np.asarray(soil[k], dtype=float)
    for k, col in SURFACE.items():
        p.surface[col] = float(soil[k])
    return p


def _leaves(tree: Any) -> list[tuple[str, np.ndarray]]:
    return [
        (jax.tree_util.keystr(k), np.asarray(v)) for k, v in jax.tree_util.tree_flatten_with_path(tree)[0]
    ]


def _differences(a: Any, b: Any) -> list[str]:
    la, lb = _leaves(a), _leaves(b)
    if [k for k, _ in la] != [k for k, _ in lb]:
        return ["tree structure"]
    return [
        k
        for (k, x), (_, y) in zip(la, lb, strict=True)
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True)
    ]


def _first(stage: np.ndarray, days: np.ndarray, code: int) -> int:
    hit = np.nonzero(stage == code)[0]
    return int(days[hit[0]]) if hit.size else -99


@pytest.fixture(scope="module")
def results(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[dict[str, Any]]]:
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file() or not MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    root = tmp_path_factory.mktemp("ws")
    report: dict[str, list[dict[str, Any]]] = {}
    for exp, trno in CASES:
        fx = MAIZE / f"{exp}.MZX"
        x = read_filex(fx)
        tr = treatment_levels(x, trno)
        prof = find_soil_profile(str(x["FIELDS"][int(tr["FL"])]["ID_SOIL"]).strip(), [MAIZE, SOIL])
        out0 = run_reference(fx, trno, root / f"{exp[:4]}0")
        orig = free_run_inputs(fx, trno, out0, source="native", extend_days=EXTEND_DAYS)
        lm = soil_layer_map(orig, fx)
        assert not lm.soildyn
        site = {k: np.asarray(v, dtype=float) for k, v in lm.sol.items()}
        soils = _draws(site, seed=sum(map(ord, exp)) + trno)
        if exp == "IBWA8301":
            soils[-1] = _clip_soil(site, lm.sw_input, lm.sol_layer)
        n_max = orig.n_days + orig.n_ext
        p0, s0 = orig.params(), orig.state()
        f0 = orig.forcing(n_max)
        rows, trees = [], []
        for k, soil in enumerate(soils):
            sdir = root / f"{exp[:4]}s{k}"
            sdir.mkdir()
            write_sol([_profile_with(prof, soil)], sdir / "SOIL.SOL")
            out = run_reference(fx, trno, root / f"{exp[:4]}{k + 1}", soil_dir=sdir)
            new = free_run_inputs(fx, trno, out, source="native", soil_dirs=[sdir], extend_days=0)
            n = new.n_days
            assert n <= n_max, (exp, k, n, n_max)
            values = {q: np.asarray(v) for q, v in soil.items()}
            ours = with_soil(p0, f0, s0, values, layers=lm)
            pn = new.params()
            short = jax.tree.map(lambda a, n=n: a[:n], ours[1])
            diff = {
                "params": _differences(ours[0], pn),
                "state": _differences(ours[2], new.state(pn)),
                "forcing": _differences(short, new.forcing(n)),
            }
            trees.append(ours)
            sw_in = np.where(np.isnan(lm.sw_input), np.asarray(ours[0]["soil"].soil.dul), lm.sw_input)
            sw0 = np.asarray(ours[2]["soil_water"].sw)
            rows.append(
                {
                    "soil": k,
                    "n_days": n,
                    "out": out,
                    "diff": diff,
                    "sw0_raised": int(np.sum(sw0 > sw_in + 1e-6)),
                    "sw0_lowered": int(np.sum(sw0 < sw_in - 1e-6)),
                    "row": read_summary(out / "Summary.OUT"),
                }
            )
        sim = run_sites(_season_model(orig.mesev), *(stack_trees([t[i] for t in trees]) for i in range(3)))
        days_all = np.asarray(f0["crop"].yrdoy)
        for k, r in enumerate(rows):
            n, out = r["n_days"], r["out"]
            summ = r.pop("row")
            summ = summ[summ["TRNO"] == trno].iloc[0]
            days = days_all[:n]
            ist = np.asarray(sim["istage"])[k, :n, 0]
            y = float(np.asarray(sim["gwad"])[k, n - 1, 0])
            hwam = float(summ["HWAM"])
            sw = read_soilwat(out / "SoilWat.OUT")
            sw = sw[sw["TRNO"] == trno] if "TRNO" in sw.columns else sw
            cols = [j for j in range(orig.nl) if f"SW{j + 1}D" in sw.columns]
            key = np.asarray(sw["YEAR"], dtype=int) * 1000 + np.asarray(sw["DOY"], dtype=int)
            pos = {int(d): i for i, d in enumerate(key.tolist())}
            at = [pos.get(int(d)) for d in days]
            ok = np.asarray([a is not None for a in at])
            idx = np.asarray([a for a in at if a is not None], dtype=int)
            table = np.stack([np.asarray(sw[f"SW{j + 1}D"], dtype=float) for j in cols], -1)
            ref = table[idx]
            soil_sw = np.asarray(sim["soil_sw"])[k, :n]
            ours_sw = np.take(
                np.take(soil_sw, np.nonzero(ok)[0], axis=0), np.asarray(cols, dtype=int), axis=1
            )
            diff_sw = np.abs(ours_sw - ref)
            beyond = diff_sw > SW_HALF_STEP
            # a SATFLO tie: at the start of the first day beyond the bound (the end of the day
            # before, or the initial water), a layer's water lies at DUL + dul_margin
            thr = np.asarray(trees[k][0]["soil"].soil.dul) + WATBAL_COEFFICIENTS.dul_margin
            tie_day, tie_layers = -99, []
            if beyond.any():
                i0 = int(np.nonzero(ok)[0][np.argmax(beyond.any(axis=1))])
                start = soil_sw[i0 - 1] if i0 > 0 else np.asarray(trees[k][2]["soil_water"].sw)
                tie_day = int(days[i0])
                tie_layers = [int(j_) + 1 for j_ in np.nonzero(np.abs(start - thr) <= THRESHOLD_TOL)[0]]
            r.update(
                sw_tie_day=tie_day,
                sw_tie_layers=tie_layers,
                yield_ours=y,
                hwam=hwam,
                yield_rel=abs(y - hwam) / hwam,
                dates={
                    name: {"ours": _first(ist, days, code), "ref": int(summ[col])}
                    for name, code, col in (
                        ("emergence", 1, "EDAT"),
                        ("silking", 4, "ADAT"),
                        ("maturity", 10, "MDAT"),
                    )
                },
                sw_layers=len(cols),
                sw_days=int(ok.sum()),
                sw_max_abs=float(np.max(np.abs(ours_sw - ref))),
                sw_over_bound=int(np.sum(np.abs(ours_sw - ref) > SW_HALF_STEP)),
                sw_cells=int(ref.size),
                out=str(out),
            )
        report[f"{exp}_t{trno:02d}"] = rows
    dest = data_dir / "validation" / "day_soil"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "with_soil_dscsm048.json").write_text(json.dumps(report, indent=1, default=str))
    for key, rows in report.items():
        for r in rows:
            print(
                f"{key} soil {r['soil']}: n={r['n_days']} yield {r['yield_ours']:.1f} vs {r['hwam']:.1f} "
                f"({100 * r['yield_rel']:.3f} %), dates "
                f"{ {k: v['ours'] - v['ref'] if v['ref'] > 0 else None for k, v in r['dates'].items()} }, "
                f"SW max |d| {r['sw_max_abs']:.5f} over {r['sw_layers']} layers x {r['sw_days']} days "
                f"({r['sw_over_bound']} beyond {SW_HALF_STEP:.6f}; SATFLO tie in layers "
                f"{r['sw_tie_layers']} on {r['sw_tie_day']}), "
                f"initial water raised in {r['sw0_raised']} / lowered in {r['sw0_lowered']} layers, "
                f"differing leaves {sum(len(v) for v in r['diff'].values())}"
            )
    return report


def test_with_soil_equals_the_native_rebuild_bit_for_bit(results) -> None:
    bad = {
        (key, r["soil"]): r["diff"] for key, rows in results.items() for r in rows if any(r["diff"].values())
    }
    assert not bad, bad


def test_yield_within_two_percent_of_dscsm048(results) -> None:
    bad = {
        (key, r["soil"]): r["yield_rel"]
        for key, rows in results.items()
        for r in rows
        if r["yield_rel"] >= YIELD_REL
    }
    assert not bad, bad


def test_stage_dates_equal_dscsm048(results) -> None:
    bad = {
        (key, r["soil"]): r["dates"]
        for key, rows in results.items()
        for r in rows
        if any(v["ours"] != v["ref"] for v in r["dates"].values() if v["ref"] > 0)
    }
    assert not bad, bad


def test_layer_soil_water_at_print_precision(results) -> None:
    """Every layer-day within :data:`SW_HALF_STEP`, except in a soil that starts its first day beyond
    it with a layer at the SATFLO threshold (module docstring); there within :data:`SW_FLIP_MAX`."""
    rows = {(key, r["soil"]): r for key, rs in results.items() for r in rs}
    untied = {
        case: (r["sw_max_abs"], r["sw_over_bound"])
        for case, r in rows.items()
        if r["sw_max_abs"] > SW_HALF_STEP and not r["sw_tie_layers"]
    }
    assert not untied, untied
    tied = {
        case: (r["sw_max_abs"], r["sw_over_bound"], r["sw_tie_layers"], r["sw_tie_day"])
        for case, r in rows.items()
        if r["sw_tie_layers"]
    }
    assert all(m <= SW_FLIP_MAX for m, *_ in tied.values()), tied
    assert all(r["sw_layers"] > 0 and r["sw_days"] > 0 for r in rows.values())


def test_the_soils_differ_and_the_limits_were_exercised(results) -> None:
    for rows in results.values():
        assert len(rows) == N_SOILS
        assert len({round(r["yield_ours"], 3) for r in rows}) > 1
    key, k = CLIP_CASE
    r = results[key][k]
    assert r["sw0_raised"] > 0 and r["sw0_lowered"] > 0, r
