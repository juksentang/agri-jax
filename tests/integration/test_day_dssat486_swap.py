"""Swapping the soil evaporation of the free DSSAT day: Ritchie two-stage
``soil_water/soilev`` and SALUS ``soil_water/esr_soilevap``, both validated against ``dscsm048``.

Each run of the free-run harness (:mod:`day_dssat486_free_harness`, configuration ``free``) runs
twice: with its own implementation (the reference's ``MESEV``: the acceptance run of
``test_day_dssat486_free.py``) and with the other one, chosen by registry key
(``day_processes(soil_evaporation=...)`` and ``day_params(soil_evaporation=...)``). The native run
must meet the acceptance criterion (yield within 2 % of ``HWAM``); the swapped run is another model,
so only its closure is asserted: finite, and the daily water ledger closed (the bucket and ``XTRACT``
book the evaporation the way the swapped implementation removes it). The differences (yield, season
ES, EP, profile water, maturity date) are the report
``<data-dir>/validation/aj_dint/swap_soil_evaporation[_small].json``: what a user changes when they
swap. The swapped runs keep the reference run's residue and ``SOILDYN`` replays (forcing), which the
other evaporation would change slightly in ``dscsm048``.

The default tier runs one treatment of each ``MESEV`` (UFGA8201 t1, GHWA0401 t1); the slow tier all 65.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import day_dssat486_free_harness as h
import jax
import numpy as np
import pytest

from agrijax.models.day_dssat486 import SOIL_EVAPORATION_KEYS

pytestmark = [
    pytest.mark.allow_skip(
        reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT), the DSSAT dump tables (WATBAL, SPAM, ROOTWU) and "
        "the CA-TPA DSSAT case"
    ),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

SMALL = [("UFGA8201", 1), ("GHWA0401", 1)]
#: the other implementation of each MESEV
OTHER = {"R": "S", "S": "R"}
#: CERES-Maize ISTAGE code of physiological maturity (the harness's MDAT)
MATURITY = 10


def _season(r: h.Run, o: dict[str, np.ndarray]) -> dict[str, Any]:
    """Season totals of one run's outputs ``o`` (its slice of the batch)."""
    n = r.n_days
    swtd = np.sum(o["soil_sw"][:n, : r.nl] * r.series["dlayr_end"], axis=-1) * h.MM_PER_CM
    stage = o["istage"][:n, 0]
    mat = h._first(stage, r.days, MATURITY)
    return {
        "yield": float(o["gwad"][n - 1, 0]),
        "es_mm": float(np.sum(o["es"][:n])),
        "ep_mm": float(np.sum(o["ep"][:n])),
        "swtd_end_mm": float(swtd[-1]),
        "swtd": swtd,
        "sw1": o["soil_sw"][:n, 0],
        "maturity": mat,
        "ledger_max_abs_residual_mm": float(np.max(np.abs(o["residual"][:n]))) * h.MM_PER_CM,
        "finite": bool(all(np.all(np.isfinite(o[k][:n])) for k in ("gwad", "soil_sw", "es", "ep"))),
    }


def _row(r: h.Run, native: dict[str, np.ndarray], swapped: dict[str, np.ndarray]) -> dict[str, Any]:
    a, b = _season(r, native), _season(r, swapped)
    hwam = float(r.row["HWAM"])
    return {
        "run": r.key,
        "mesev": r.mesev,
        "native": SOIL_EVAPORATION_KEYS[r.mesev],
        "swapped": SOIL_EVAPORATION_KEYS[OTHER[r.mesev]],
        "hwam": hwam,
        "native_yield": a["yield"],
        "native_yield_rel": abs(a["yield"] - hwam) / hwam if hwam > 0 else None,
        "swapped_yield": b["yield"],
        "yield_change_rel": (b["yield"] - a["yield"]) / a["yield"] if a["yield"] > 0 else None,
        "native_es_mm": a["es_mm"],
        "swapped_es_mm": b["es_mm"],
        "native_ep_mm": a["ep_mm"],
        "swapped_ep_mm": b["ep_mm"],
        "native_swtd_end_mm": a["swtd_end_mm"],
        "swapped_swtd_end_mm": b["swtd_end_mm"],
        "swtd_max_abs_change_mm": float(np.max(np.abs(b["swtd"] - a["swtd"]))),
        "sw1_max_abs_change": float(np.max(np.abs(b["sw1"] - a["sw1"]))),
        "maturity_shift_days": h._yrdoy_diff(b["maturity"], a["maturity"]),
        "native_ledger_mm": a["ledger_max_abs_residual_mm"],
        "swapped_ledger_mm": b["ledger_max_abs_residual_mm"],
        "finite": a["finite"] and b["finite"],
    }


@pytest.fixture(scope="module", params=["small", pytest.param("all", marks=pytest.mark.slow)])
def table(request: pytest.FixtureRequest, data_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    if not h.m2.DSCSM.is_file() or not h.m2.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {h.m2.DSSAT_ENGINE}")
    for sub in (h.DSW, h.DET, h.A12, h.CATPA_CASE):
        if not (data_dir / sub).is_dir():
            pytest.skip(f"{data_dir / sub} not found")
    keys = SMALL if request.param == "small" else h.a12_keys(data_dir) + h.catpa_keys()
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    work = tmp_path_factory.mktemp("swap")
    outs = h.run_references(keys, work, data_dir, jobs)
    runs = [h.build(e, t, outs[h.key_of(e, t)], data_dir) for e, t in keys]
    rows: dict[str, dict[str, Any]] = {}
    for grp in sorted({(r.nl, r.mesev) for r in runs}):
        rs = [r for r in runs if (r.nl, r.mesev) == grp]
        native = h.run_group(rs, "free")
        swapped = h.run_group(rs, "free", soil_evaporation=SOIL_EVAPORATION_KEYS[OTHER[grp[1]]])
        for i, r in enumerate(rs):
            rows[r.key] = _row(r, {k: v[i] for k, v in native.items()}, {k: v[i] for k, v in swapped.items()})
    table = [rows[h.key_of(e, t)] for e, t in keys]
    out = data_dir / h.REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    name = "swap_soil_evaporation.json" if request.param == "all" else "swap_soil_evaporation_small.json"
    (out / name).write_text(json.dumps({"step": "soil evaporation swap", "runs": table}, indent=1))
    return table


def test_both_implementations_are_exercised(table) -> None:
    assert {r["mesev"] for r in table} == {"R", "S"}


def test_native_run_meets_the_acceptance_criterion(table) -> None:
    bad = {
        r["run"]: r["native_yield_rel"]
        for r in table
        if r["hwam"] > 0 and r["native_yield_rel"] >= h.YIELD_REL
    }
    assert not bad, bad


def test_swapped_run_is_finite_and_its_ledger_closes(table) -> None:
    assert all(r["finite"] for r in table)
    bad = {
        r["run"]: (r["native_ledger_mm"], r["swapped_ledger_mm"])
        for r in table
        if max(r["native_ledger_mm"], r["swapped_ledger_mm"]) > h.LEDGER_ATOL_MM
    }
    assert not bad, bad


def test_the_swap_changes_the_soil_evaporation(table) -> None:
    """The swap is not a no-op: every run's season ES moves (the two methods differ by design)."""
    assert all(r["swapped_es_mm"] != r["native_es_mm"] for r in table)
