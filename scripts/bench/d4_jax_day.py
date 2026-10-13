"""Bench assembly: the validated free-run DSSAT day as the speed benchmark's model.

Measurement only (no change to the package). The day is
:func:`agrijax.models.day_dssat486.day_dssat486` with every entry's own process
(:func:`~agrijax.models.day_dssat486.day_processes`, ``mesev`` of the run: SOILDYN albedo, the
tipping bucket, PETPT, SPAM PSE / MULCH_EVAP / SOILEV or ESR / TRANS, ROOTWU, XTRACT, CERES-Maize,
the ledger), compiled with ``exact_lags=True``: the configuration ``free`` of
``tests/integration/day_dssat486_free_harness.py`` that passed the free-run acceptance against
DSSAT-CSM (65 runs: the 58 CERES-Maize treatments and the 7 nitrogen-off CA-TPA seasons). Nothing
of the soil water, the evapotranspiration or the uptake is replayed; the labelled replays of the
acceptance run stay (residue record, SOILDYN soil properties of the runs whose soil changes,
hourly-mean TAVG).

This file only builds the harness's inputs and adds the benchmark's outputs, perturbation and
batched simulator; the model is the harness's.

Inputs: built once (``prepare``) through the harness (``build``: one ``dscsm048`` reference run
per treatment for CERES-Maize's parameters and the crop weather, the instrumented-engine WATBAL
and SPAM dump tables for the soil and SPAM inputs) with ``params_of`` / ``forcing_of`` /
``state_of`` of the ``free`` configuration (REAL*4 soil values, SOILDYN replay on), the forcing
padded to the longest season of the run's group (``(layer count, MESEV)``; rain and irrigation zero
after the season, as in the acceptance run), and pickled as NumPy trees so that the timing jobs
load them without running DSSAT.
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src", REPO / "tests" / "integration"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from agrijax.core.state import get_path  # noqa: E402
from agrijax.core.units import MM_PER_CM  # noqa: E402
from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_processes  # noqa: E402
from agrijax.processes.crop.ceres_maize import plantgro_outputs  # noqa: E402

CONFIG = "free"  # the free-run acceptance configuration of the harness
P = {
    "crop_water": f"iface.crop_water.{SLOT}",
    "crop": f"crops.{SLOT}",
}


# ============================================================================ model
def _first(x: Any) -> Any:
    return jnp.ravel(x)[0]


def outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
    """Ten daily series per run (the benchmark's output set): LAI, CWAD, GWAD, the
    growth stage, the profile water [cm], ES, EP, EOP [mm d-1], runoff and drainage [mm d-1]."""
    crop = get_path(state, P["crop"])
    pg = plantgro_outputs(crop, params["crop"], forcing_t["crop"])
    b = get_path(state, "soil_water")
    ev = get_path(state, "iface.evaporation")
    pet = get_path(state, "iface.pet")
    soil = forcing_t["soil"].soil
    dl = params["soil"].soil.dlayr if soil is None else soil.dlayr
    return {
        "lai": _first(pg["lai"]),
        "cwad": _first(pg["cwad"]),
        "gwad": _first(pg["gwad"]),
        "istage": _first(pg["istage"]),
        "tsw_cm": jnp.sum(b.sw * dl, axis=-1),
        "es_mm": ev.soil_evaporation * MM_PER_CM,
        "ep_mm": ev.transpiration,
        "eop_mm": pet.transpiration * MM_PER_CM,
        "runoff_mm": b.flux.runoff,
        "drain_mm": b.flux.drain,
    }


@dataclass
class BenchDay:
    model: Any
    lag_check: str


def bench_model(mesev: str) -> BenchDay:
    """The free-run DSSAT day (every entry its own process), lag check with exact lags on."""
    procs = day_processes(SLOT, mesev=mesev)
    m = day_dssat486(SLOT).compile(procs, outputs=outputs, exact_lags=True)
    return BenchDay(m, "passed (exact_lags=True)")


# ============================================================================ inputs
def _np_tree(t: Any) -> Any:
    return jax.tree.map(lambda x: np.asarray(x), t)


def prepare(data_dir: Path, work: Path, out: Path, jobs: int) -> dict[str, Any]:
    """Build the 65 acceptance runs' inputs (one ``dscsm048`` run per treatment via the harness)."""
    jax.config.update("jax_enable_x64", True)
    import day_dssat486_free_harness as H

    keys = H.a12_keys(data_dir) + H.catpa_keys()
    refs = H.run_references(keys, work, data_dir, jobs)
    runs = [H.build(e, t, refs[H.key_of(e, t)], data_dir) for e, t in keys]
    soil_values, soildyn, real4_sw = H.CONFIGS[CONFIG]
    n_group: dict[tuple[int, str], int] = {}
    for r in runs:
        g = (r.nl, r.mesev)
        n_group[g] = max(n_group.get(g, 0), r.n_days)
    items = []
    for r in runs:
        n = n_group[(r.nl, r.mesev)]
        ps = H.params_of(r, soil_values, real4_sw)
        items.append(
            {
                "key": r.key,
                "nl": r.nl,
                "mesev": r.mesev,
                "n_days": r.n_days,
                "n_padded": n,
                "days": np.asarray(r.days, dtype=np.int64),
                "hwam": float(r.row["HWAM"]),
                "ref_dates": {c: float(r.row[c]) for c in ("EDAT", "ADAT", "MDAT")},
                "params": _np_tree(ps),
                "forcing": _np_tree(H.forcing_of(r, n, soil_values, soildyn)),
                "state": _np_tree(H.state_of(r, ps, soil_values)),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(items, fh, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "config": CONFIG,
        "n_runs": len(items),
        "groups": {
            f"nl{nl}_{m}": {
                "runs": sum(1 for it in items if it["nl"] == nl and it["mesev"] == m),
                "n_padded": n_group[(nl, m)],
            }
            for nl, m in sorted(n_group)
        },
        "n_days": [it["n_days"] for it in items],
        "n_days_total": int(sum(it["n_days"] for it in items)),
    }
    (out.parent / "inputs_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ============================================================================ batched run
def stack(trees: list[Any]) -> Any:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *trees)


def cast(tree: Any, dtype: Any) -> Any:
    def f(x: Any) -> Any:
        x = np.asarray(x)
        return x.astype(dtype) if np.issubdtype(x.dtype, np.floating) else x

    return jax.tree.map(f, tree)


def group_inputs(items: list[dict[str, Any]], dtype: Any) -> dict[str, Any]:
    """Stack one group's runs (same layer count and MESEV; forcing already padded to the group)."""
    n = items[0]["n_padded"]
    assert all(it["n_padded"] == n for it in items)
    return {
        "keys": [it["key"] for it in items],
        "n_days": n,
        "real_days": [it["n_days"] for it in items],
        "params": cast(stack([it["params"] for it in items]), dtype),
        "forcing": cast(stack([it["forcing"] for it in items]), dtype),
        "state": cast(stack([it["state"] for it in items]), dtype),
    }


#: parameters perturbed per sample (multiplicative factor U(1 - SPREAD, 1 + SPREAD))
PERTURBED = ("p1", "p5", "g2", "g3", "phint", "cn", "swcon")
SPREAD = 0.1


def perturb(params: Any, mult: Any) -> Any:
    """Apply the sample multipliers ``mult`` ([B, 7]) to the cultivar and the static bucket soil."""
    c = params["crop"].cultivar
    cul = eqx.tree_at(
        lambda x: (x.p1, x.p5, x.g2, x.g3, x.phint),
        c,
        (c.p1 * mult[:, 0], c.p5 * mult[:, 1], c.g2 * mult[:, 2], c.g3 * mult[:, 3], c.phint * mult[:, 4]),
    )
    crop = eqx.tree_at(lambda x: x.cultivar, params["crop"], cul)
    s = params["soil"].soil
    soil = eqx.tree_at(lambda x: (x.cn, x.swcon), s, (s.cn * mult[:, 5], s.swcon * mult[:, 6]))
    return {**params, "crop": crop, "soil": eqx.tree_at(lambda x: x.soil, params["soil"], soil)}


def perturb_day_soil(f_t: Any, mult: Any) -> Any:
    """The same CN and SWCON factors on the day's SOILPROP forcing (the bucket reads it every day)."""
    sf = f_t["soil"]
    if sf.soil is None:
        return f_t
    s = sf.soil
    soil = eqx.tree_at(lambda x: (x.cn, x.swcon), s, (s.cn * mult[:, 5], s.swcon * mult[:, 6]))
    return {**f_t, "soil": eqx.tree_at(lambda x: x.soil, sf, soil)}


def simulator(model: Any, n_days: int) -> Any:
    """``sim(params_tr, forcing_tr, state_tr, tid, mult) -> outputs [T, B]``: sample ``b`` runs
    treatment ``tid[b]`` with its parameters times ``mult[b]``; the day's forcing of every treatment
    is gathered per day inside the scan (no [B, T] forcing copy)."""
    step = jax.vmap(model.compile(), in_axes=(0, 0, 0))

    def sim(params_tr: Any, forcing_tr: Any, state_tr: Any, tid: Any, mult: Any) -> Any:
        params = perturb(jax.tree.map(lambda x: x[tid], params_tr), mult)
        state0 = jax.tree.map(lambda x: x[tid], state_tr)

        def body(s: Any, t: Any) -> Any:
            f_t = perturb_day_soil(jax.tree.map(lambda x: x[tid, t], forcing_tr), mult)
            return step(s, params, f_t)

        _, outs = jax.lax.scan(body, state0, jnp.arange(n_days))
        return outs

    return sim


# ============================================================================ accuracy helpers
def first_day(stage: np.ndarray, days: np.ndarray, code: int) -> int:
    hit = np.nonzero(np.asarray(stage) == code)[0]
    return int(days[hit[0]]) if hit.size else -99


def yrdoy_diff(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None if a == b else 9999
    da = date(a // 1000, 1, 1) + timedelta(days=a % 1000 - 1)
    db = date(b // 1000, 1, 1) + timedelta(days=b % 1000 - 1)
    return (da - db).days


#: the dates of the acceptance (``day_dssat486_free_harness._dates``): name, ISTAGE code, Summary column
DATES = (("emergence", 1, "EDAT"), ("silking", 4, "ADAT"), ("maturity", 10, "MDAT"))


def run_record(it: dict[str, Any], gwad: np.ndarray, istage: np.ndarray) -> dict[str, Any]:
    """One run's harvest-day yield and dates (the acceptance quantities) from its daily outputs."""
    n = it["n_days"]
    y = float(gwad[n - 1])
    dates = {}
    for name, code, col in DATES:
        ref = it["ref_dates"][col]
        refi = int(ref) if ref > 0 else -99
        ours = first_day(istage[:n], it["days"], code)
        dates[name] = {"ours": ours, "ref": refi, "diff_days": yrdoy_diff(ours, refi)}
    return {"key": it["key"], "yield": y, "hwam": it["hwam"], "dates": dates}
