"""Per-call check of the post-tillage curves against RZWQM2 ``WC`` and ``POINTK`` dumps (grade C).

Each dump is one function call of the instrumented CA-TPA 2015 run: ``WC(H, HYDP, I, ISTAT)`` with
its saved start-up copy ``TRHYDP(13, MAXHOR)``, and ``POINTK(HEAD, HYDP, J, PORI)`` with its saved
``C22``/``SN22``. :func:`~agrijax.processes.soil_water.hydraulics.theta_of_h` and
:func:`~agrijax.processes.soil_water.hydraulics.k_of_h` on
:class:`~agrijax.processes.soil_water.hydraulics.TilledSoilHydraulicParams` (current ``HYDP``,
original ``TRHYDP``) must return the dumped value to a few units in the last place: the
arithmetic is the same, and the only difference is the ``pow`` of the Fortran runtime against
XLA's (``REL_TOL``, measured).

The saved copies are checked as well: every ``POINTK`` dump holds ``C22 = C2(TRHYDP)`` and
``SN22 = TRHYDP(3)`` (``C2 = ksat hb_k**(eps - n1)``, :func:`c2_of_params`), so the start-up
``K`` curve is the one of the start-up ``SOILHP``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from tillage_dumps import run_of_case, saved_c22_sn22, saved_trhydp

from agrijax.processes.soil_water.hydraulics import (
    RZWQM_HYDRAULICS,
    SoilHydraulicParams,
    TilledSoilHydraulicParams,
    c2_of_params,
    k_of_h,
    theta_of_h,
)

pytestmark = [
    pytest.mark.allow_skip(reason="WC / POINTK dumps are private data"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="dump comparison is float64"),
]

REL_TOL = 1e-14


def _params(hp: np.ndarray) -> SoilHydraulicParams:
    hp = np.asarray(hp, float).reshape(13)
    return SoilHydraulicParams.from_rzwqm_records(hp[None, 0:6], hp[None, 6:13])


def _case_files(data_dir: Path, routine: str) -> list[Path]:
    d = data_dir / "dumps" / routine
    return sorted(d.glob("*.npz")) if d.is_dir() else []


def _i(x: Any) -> int:
    return int(np.asarray(x).reshape(-1)[0])


def _f(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


@pytest.fixture(scope="module")
def wc_table(data_dir: Path) -> pd.DataFrame:
    from agrijax.port.dumps import load_case

    files = _case_files(data_dir, "WC")
    if not files:
        pytest.skip("no WC dumps")
    rows = []
    for f in files:
        c = load_case(f)
        e = c.entry
        i = _i(e["I"]) - 1
        cur = np.asarray(e["HYDP"], float).reshape(13)
        orig = cur if _i(e["ISTAT"]) == -1 else np.asarray(e["TRHYDP"], float)[:, i]
        h = _f(e["H"])
        p = TilledSoilHydraulicParams(current=_params(cur), original=_params(orig))
        ours = float(np.asarray(theta_of_h(jnp.asarray([h]), p))[0])
        single = float(np.asarray(theta_of_h(jnp.asarray([h]), _params(cur)))[0])
        ref = _f(c.exit["WC"])
        split = RZWQM_HYDRAULICS.tillage_split_retention * cur[0]
        rows.append(
            {
                "case": f.name,
                "tilled": not np.array_equal(cur, orig),
                "original_branch": h < -split,
                "h_cm": h,
                "wc_ref": ref,
                "rel_err": abs(ours - ref) / abs(ref),
                "rel_err_single": abs(single - ref) / abs(ref),
            }
        )
    t = pd.DataFrame(rows)
    out = os.environ.get("AGRI_JAX_RICHARDS_REPORT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        t.to_csv(Path(out) / "wc_dump_calls.csv", index=False, float_format="%.6g")
    return t


@pytest.fixture(scope="module")
def pointk_table(data_dir: Path) -> pd.DataFrame:
    from agrijax.port.dumps import load_case

    files = _case_files(data_dir, "POINTK")
    if not files:
        pytest.skip("no POINTK dumps")
    trhydp = {r: saved_trhydp(data_dir, r) for r in {run_of_case(f) for f in files}}
    rows = []
    for f in files:
        tr = trhydp[run_of_case(f)]
        if tr is None:
            pytest.skip(f"no WC dumps (TRHYDP) for the run of {f.name}")
        c = load_case(f)
        e = c.entry
        j = _i(e["J"]) - 1
        cur = np.asarray(e["HYDP"], float).reshape(13)
        first = j + 1 > _i(e["IRST"])  # POINTK saves C22/SN22 from HYDP on this call
        c22 = cur[10] if first else _f(np.asarray(e["C22"], float)[j])
        sn22 = cur[2] if first else _f(np.asarray(e["SN22"], float)[j])
        orig = tr[:, j]
        po = _params(orig)
        p = TilledSoilHydraulicParams(current=_params(cur), original=po)
        h = _f(e["HEAD"])
        ours = float(np.asarray(k_of_h(jnp.asarray([h]), p))[0])
        single = float(np.asarray(k_of_h(jnp.asarray([h]), _params(cur)))[0])
        ref = _f(c.exit["POINTK"])
        split = RZWQM_HYDRAULICS.tillage_split_conductivity * cur[9]
        rows.append(
            {
                "case": f.name,
                "tilled": not np.array_equal(cur, orig),
                "original_branch": h <= -split,
                "h_cm": h,
                "k_ref": ref,
                "rel_err": abs(ours - ref) / abs(ref),
                "rel_err_single": abs(single - ref) / abs(ref),
                "c22_rel_err": abs(float(np.asarray(c2_of_params(po))[0]) - c22) / abs(c22),
                "sn22_equal": sn22 == orig[2],
            }
        )
    t = pd.DataFrame(rows)
    out = os.environ.get("AGRI_JAX_RICHARDS_REPORT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        t.to_csv(Path(out) / "pointk_dump_calls.csv", index=False, float_format="%.6g")
    return t


def test_wc_calls_match_the_two_segment_curve(wc_table: pd.DataFrame) -> None:
    t = wc_table
    assert float(np.max(t["rel_err"].to_numpy())) < REL_TOL, t.sort_values("rel_err").tail(5).to_string()


def test_wc_dumps_exercise_the_original_branch(wc_table: pd.DataFrame) -> None:
    """At least one tilled call below ``-10 hb``, where the single-segment curve is wrong."""
    hit = wc_table.loc[wc_table["tilled"] & wc_table["original_branch"]]
    if len(hit) == 0:
        pytest.skip("no WC dump on the original branch of a tilled horizon")
    assert float(np.max(hit["rel_err_single"].to_numpy())) > 1e3 * REL_TOL


def test_pointk_calls_match_the_two_segment_curve(pointk_table: pd.DataFrame) -> None:
    t = pointk_table
    assert float(np.max(t["rel_err"].to_numpy())) < REL_TOL, t.sort_values("rel_err").tail(5).to_string()


def test_pointk_saved_copies_are_the_start_up_curve(pointk_table: pd.DataFrame) -> None:
    t = pointk_table
    assert bool(t["sn22_equal"].all())
    assert float(np.max(t["c22_rel_err"].to_numpy())) < REL_TOL


def test_saved_c22_is_constant_over_the_run(data_dir: Path) -> None:
    files = _case_files(data_dir, "POINTK")
    if not files:
        pytest.skip("no POINTK dumps")
    for run in {run_of_case(f) for f in files}:
        assert saved_c22_sn22(data_dir, run) is not None
