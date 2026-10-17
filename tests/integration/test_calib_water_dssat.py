"""Soil water and evapotranspiration as calibration outputs (``calib/dssat_day.py`` ``et``, ``sw<k>``;
``calib/observations.py`` with ``soil_layers``) against ``dscsm048``.

Three DSSAT v4.8.6.0 example treatments with nine model layers and ``MESEV = R`` (one batch group):
UFGA8201 treatment 2 (rainfed) and 6 (irrigated), BRPI0202 treatment 5 (four ``.SOL`` layers split
into nine). Each runs with ``dscsm048`` (nitrogen off), its free-run inputs are built from the
public files (``source="native"``) and one :class:`DaySimulator` call at the published cultivar
returns, for every simulated day, ``et`` and the soil water of every layer.

* **ET** against ``ET.OUT`` ``ETAA`` (daily output: the day's ``ET = EVAP + EP``, ``SPAM.for``,
  printed ``F8.3``): within :data:`ET_ATOL` and the season totals within :data:`ET_SUM_REL`, both
  pinned at about 1.5 times the measured agreement on these three treatments (as the free-run
  harness pins its bounds), so that an error of about a print step, e.g. a dropped ``EM`` on a day
  it exceeds that, fails.
* **Soil water** against ``SoilWat.OUT`` ``SW<k>D`` (printed ``F8.3``): within the free run's pinned
  layer bound ``h.SECONDARY_BOUNDS["m2"]["soil_water.SW_layers_max_abs"]``.
* **Observations by depth**: the ``SoilWat.OUT`` soil water, given as observations at the middle
  depth of each layer (``soil_water``, with the model layer bottoms ``DS`` as ``soil_layers``), go
  back to the layers they came from (target ``SW<k>D`` observed where ``SoilWat.OUT`` prints layer
  ``k``), and the simulator's entries built from those targets reproduce them within the soil-water
  bound.

The measured numbers are written to ``<data-dir>/validation/calib_water/water_dscsm048.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import day_dssat486_free_harness as h
import jax
import numpy as np
import pandas as pd
import pytest

from agrijax.calib.dssat_day import CUL_ORDER, E_DAY, OUT_NAMES, DaySimulator, Entry
from agrijax.calib.observations import observation_targets
from agrijax.io.dssat import read_soilwat
from agrijax.io.dssat.observed import ObservedData
from agrijax.io.dssat.outputs import read_et
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
from agrijax.sites.dssat_free_run import free_run_inputs, run_reference

pytestmark = [
    pytest.mark.allow_skip(reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the DSSAT example data"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
CASES = [("UFGA8201", 2), ("UFGA8201", 6), ("BRPI0202", 5)]
#: half the print step of ET.OUT's ETAA, ESAA, EPAA, EMAA (F8.3)
PRINT_HALF_STEP = 0.0005
_B = h.SECONDARY_BOUNDS["m2"]
#: daily |ET - ETAA| [mm d-1]: measured 4.94e-4 .. 4.98e-4 on the three treatments (ETAA's print
#: half-step, ``water_dscsm048.json`` 2026-10-04), pinned at about 1.5 times the largest
ET_ATOL = 1.5 * PRINT_HALF_STEP
#: season total of ET against the sum of the printed ETAA (relative): measured 4.5e-6 .. 2.4e-5,
#: pinned at about 1.5 times the largest
ET_SUM_REL = 3.6e-5
#: layer soil water against SoilWat.OUT [cm3 cm-3]: the free run's pinned bound on these treatments
SW_ATOL = _B["soil_water.SW_layers_max_abs"][1]


def _by_day(df: pd.DataFrame, col: str, days: np.ndarray) -> np.ndarray:
    key = np.asarray(df["YEAR"], dtype=int) * 1000 + np.asarray(df["DOY"], dtype=int)
    m = dict(zip(key.tolist(), np.asarray(df[col], dtype=float).tolist(), strict=True))
    return np.asarray([m.get(int(d), np.nan) for d in days], dtype=float)


@pytest.fixture(scope="module")
def results(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file() or not MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    root = tmp_path_factory.mktemp("wat")
    runs, outs = [], []
    for exp, trno in CASES:
        out = run_reference(MAIZE / f"{exp}.MZX", trno, root / f"{exp[:4]}{trno}")
        runs.append(free_run_inputs(MAIZE / f"{exp}.MZX", trno, out, source="native"))
        outs.append(out)
    assert len({(r.nl, r.mesev) for r in runs}) == 1, [(r.nl, r.mesev) for r in runs]
    i_et = OUT_NAMES.index("et")
    entries = []
    for r in runs:
        n = r.n_days
        e = [Entry(E_DAY, i_et, t) for t in range(n)]
        e += [Entry(E_DAY, OUT_NAMES.index(f"sw{k}"), t) for k in range(1, r.nl + 1) for t in range(n)]
        entries.append(e)
    sim = DaySimulator(runs, entries)
    theta = np.asarray([[r.published()[c] for c in CUL_ORDER] for r in runs])
    y = sim(theta, np.arange(len(runs)))
    report: dict[str, Any] = {}
    for b, (r, out) in enumerate(zip(runs, outs, strict=True)):
        n, nl, days = r.n_days, r.nl, r.days
        et = y[b, :n]
        sw = y[b, n : n + nl * n].reshape(nl, n).T
        ref_et = read_et(out / "ET.OUT")
        ref_et = ref_et[ref_et["TRNO"] == r.trno] if "TRNO" in ref_et.columns else ref_et
        etaa = _by_day(ref_et, "ETAA", days)
        sws = read_soilwat(out / "SoilWat.OUT")
        sws = sws[sws["TRNO"] == r.trno] if "TRNO" in sws.columns else sws
        cols = [k for k in range(nl) if f"SW{k + 1}D" in sws.columns]
        sw_ref = np.stack([_by_day(sws, f"SW{k + 1}D", days) for k in cols], -1)
        ok_et = np.isfinite(etaa)
        ok_sw = np.isfinite(sw_ref)
        # observations by depth: the printed soil water at the middle of each layer
        ds = np.asarray(r.site.ds, dtype=float)
        mid = (np.concatenate([[0.0], ds[:-1]]) + ds) / 2.0
        rows = [
            (r.trno, int(days[t]), float(mid[k]), float(sw_ref[t, j]))
            for j, k in enumerate(cols)
            for t in range(n)
            if ok_sw[t, j]
        ]
        table = pd.DataFrame(rows, columns=["TRNO", "YRDOY", "DEPTH", "SW"])
        ot = observation_targets(
            ObservedData(r.key), [r.trno], days[None], soil_layers=[ds], soil_water=table
        )
        placed: dict[str, Any] = {}
        sub_entries, sub_obs = [], []
        for t in ot.targets:
            k = int(t.output[2:])
            mask = np.asarray(t.mask)[:, 0]
            placed[t.name] = {
                "layer": k,
                "same_days_as_printed": bool(np.array_equal(mask, ok_sw[:, cols.index(k - 1)]))
                if k - 1 in cols
                else False,
                "values_equal_printed": bool(
                    np.array_equal(np.asarray(ot.observed[t.name])[mask, 0], sw_ref[mask, cols.index(k - 1)])
                )
                if k - 1 in cols
                else False,
            }
            for d in np.nonzero(mask)[0]:
                sub_entries.append(Entry(E_DAY, OUT_NAMES.index(t.output), int(d)))
                sub_obs.append(float(np.asarray(ot.observed[t.name])[d, 0]))
        one = DaySimulator([r], [sub_entries])
        y_obs = one(theta[b : b + 1], np.zeros(1, np.int64))[0, : len(sub_entries)]
        report[r.key] = {
            "n_days": n,
            "n_layers": nl,
            "sw_layers_printed": len(cols),
            "et_days": int(ok_et.sum()),
            "et_max_abs": float(np.max(np.abs(et[ok_et] - etaa[ok_et]))),
            "et_sum_ours": float(np.sum(et[ok_et])),
            "et_sum_ref": float(np.sum(etaa[ok_et])),
            "et_sum_rel": abs(float(np.sum(et[ok_et])) - float(np.sum(etaa[ok_et])))
            / float(np.sum(etaa[ok_et])),
            "sw_cells": int(ok_sw.sum()),
            "sw_max_abs": float(np.nanmax(np.abs(sw[:, cols] - sw_ref))),
            "depth_targets": placed,
            "depth_target_names": sorted(placed),
            "depth_notes": list(ot.notes),
            "depth_entries_max_abs": float(np.max(np.abs(y_obs - np.asarray(sub_obs)))),
            "expected_targets": sorted(f"SW{k + 1}D" for k in cols),
        }
    dest = data_dir / "validation" / "calib_water"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "water_dscsm048.json").write_text(json.dumps(report, indent=1, default=str))
    for key, row in report.items():
        print(
            f"{key}: ET max |d| {row['et_max_abs']:.5f} (bound {ET_ATOL:.5f}) over {row['et_days']} days, "
            f"season {row['et_sum_ours']:.2f} vs {row['et_sum_ref']:.2f} mm; SW max |d| {row['sw_max_abs']:.6f} "
            f"(bound {SW_ATOL}) over {row['sw_layers_printed']} layers x {row['n_days']} days; "
            f"depth entries max |d| {row['depth_entries_max_abs']:.6f}"
        )
    return report


def test_every_day_and_layer_compared(results) -> None:
    for key, row in results.items():
        assert row["et_days"] == row["n_days"], key
        assert row["sw_layers_printed"] == row["n_layers"] == 9, key
        assert row["sw_cells"] == row["n_days"] * row["n_layers"], key


def test_et_against_dscsm048(results) -> None:
    bad = {k: r["et_max_abs"] for k, r in results.items() if not r["et_max_abs"] <= ET_ATOL}
    assert not bad, (bad, ET_ATOL)
    bad = {k: r["et_sum_rel"] for k, r in results.items() if not r["et_sum_rel"] <= ET_SUM_REL}
    assert not bad, (bad, ET_SUM_REL)


def test_layer_soil_water_against_dscsm048(results) -> None:
    bad = {k: r["sw_max_abs"] for k, r in results.items() if not r["sw_max_abs"] <= SW_ATOL}
    assert not bad, (bad, SW_ATOL)


def test_observations_by_depth_go_back_to_their_layers(results) -> None:
    for key, r in results.items():
        assert r["depth_target_names"] == r["expected_targets"], key
        assert all(
            p["same_days_as_printed"] and p["values_equal_printed"] for p in r["depth_targets"].values()
        ), (
            key,
            r["depth_targets"],
        )
        assert not r["depth_notes"], (key, r["depth_notes"])
        assert r["depth_entries_max_abs"] <= SW_ATOL, key
