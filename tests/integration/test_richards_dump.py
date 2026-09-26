"""Per-call comparison with RZWQM2 ``RICHRD`` entry/exit dumps (grade C), when the dumps exist.

The instrumentation track writes ``<data>/dumps/RICHRD/<case>.npz`` (:mod:`agrijax.port.dumps`:
``in.<NAME>`` / ``out.<NAME>`` per Fortran argument, arrays in Fortran shape). Without that
directory every test here skips.

One ``RICHRD`` call advances the profile by one time step ``DELT`` [h] with RZWQM's modified
Picard iteration converged to ``RICH_EPS``. The same step is taken here with
:func:`agrijax.processes.soil_water.richards.richards_step`, converged (20 Newton iterations).
Mapping of the dumped arguments (checked on the CA-TPA 2015 dumps):

* grid ``TL[:NN]``, ``DELZ[:NN-1]``, node map ``NDXN2H`` (1-based), hydraulics
  ``SOILHP(13, horizon)`` (rows 1-6 rec1, rows 7-13 rec2, re-derived as ``SOILPR`` does);
* start heads ``in.H[:NN]`` (``HOLD``/``TH`` are the ghost-augmented arrays, node ``i`` at
  index ``i + 1``), start water content ``in.THETA``, step ``out.DELT`` (the step RZWQM actually
  took), time weight ``in.ALPH``, evaporation demand ``-in.EVAP`` [cm h-1]
  (``EVAP = -(PES + PER) <= 0``), sink ``in.QS`` [cm h-1 cm-1], ``HMIN``.

Two classes of call, told apart by the start-up copy ``TRHYDP`` of the hydraulic parameters
(read from the run's ``WC`` dumps, :mod:`tillage_dumps`):

* *untilled*: ``SOILHP == TRHYDP`` bit for bit on every horizon, i.e. the curve is the
  single-segment curve of ``SOILHP``. These calls run on :class:`SoilHydraulicParams` exactly as
  before the post-tillage variant existed (bit-identical results);
* *tilled*: tillage (``MATILL``) and reconsolidation have changed ``SOILHP``; RZWQM2's ``WC``,
  ``SPMOIS`` and ``POINTK`` then evaluate the current curve above ``-10 hb`` and the pre-tillage
  curve below (Ahuja et al. 1998). These calls run on
  :class:`~agrijax.processes.soil_water.hydraulics.TilledSoilHydraulicParams`.

Every call, of both classes, must satisfy ``in.THETA == theta(in.H)`` to 1e-12 on every node
(the curve RZWQM evaluates is the one used here) and reproduce ``out.THETA`` to ``THETA_TOL``,
the storage change to ``STORAGE_TOL`` and ``out.H`` to ``HEAD_TOL``. CA-TPA 2015 (100 calls):
15 untilled, 85 tilled; 61 of the tilled calls have a node below ``-10 hb`` on the original
curve (``in.THETA`` differs from the single-segment ``theta(in.H)`` by up to 4.8e-3). Before the
variant those 61 ran on the single-segment curve and were held only to a looser 1e-4 in theta
(worst 5.3e-5 in theta, 9.7e-5 cm in storage, 141 cm in head); on the two-segment curve all but
one are grade C (the ``*_single`` columns of the table keep the single-segment result).

The exception is an open difference of the surface boundary, not of the curve: on
``catpa2015_d2015244_c0000066473`` the dry surface limit of the solver cuts 1.54e-5 cm of the
evaporation RZWQM2 delivered in full (``AEVAP = EVAP``); the storage difference equals that
deficit to 3e-16 cm on both curves. Such *surface-limited* calls (our evaporation deficit > 0
while ``-AEVAP DELT`` is the full demand) are checked for exactly that and nothing looser. Calls
where both models cut the evaporation (three on day 144) are grade C like the others.

With ``AGRI_JAX_RICHARDS_REPORT`` set (a directory) the per-call table is written there as
``richrd_dump_calls.csv``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from tillage_dumps import run_of_case, saved_trhydp

from agrijax.processes.soil_water.hydraulics import (
    AnyHydraulicParams,
    SoilHydraulicParams,
    TilledSoilHydraulicParams,
    theta_of_h,
)
from agrijax.processes.soil_water.richards import RichardsConfig, RichardsGrid, richards_step

pytestmark = [
    pytest.mark.allow_skip(reason="RICHRD dumps are private data"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="dump comparison is float64"),
]

ROUTINE = "RICHRD"
# measured on the 39 untilled CA-TPA 2015 calls: max 2.1e-10 in theta, 1.4e-13 cm in storage,
# 2.6e-7 cm in head (the Newton root vs RZWQM's Picard root converged to RICH_EPS)
THETA_TOL = 1e-8
STORAGE_TOL = 1e-10
HEAD_TOL = 1e-5
CONSISTENT = 1e-12
REQUIRED_IN = ("DELZ", "TL", "NDXN2H", "NN", "SOILHP", "H", "THETA", "EVAP", "QS", "HMIN", "ALPH")
REQUIRED_OUT = ("H", "THETA", "DELT")

_step = jax.jit(richards_step, static_argnames=("cfg",))


def _cases(data_dir: Path) -> list[Path]:
    d = data_dir / "dumps" / ROUTINE
    return sorted(d.glob("*.npz")) if d.is_dir() else []


def _scalar(x: np.ndarray) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _soil(hp: np.ndarray, n_hor: int, nn: int, nh: np.ndarray) -> SoilHydraulicParams:
    s = SoilHydraulicParams.from_rzwqm_records(hp[0:6, :n_hor].T, hp[6:13, :n_hor].T, node_horizon=nh)
    return jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (nn,)), s.at_nodes())


def _run_step(soil: AnyHydraulicParams, e: dict[str, Any], dt: float, nn: int, grid: RichardsGrid) -> Any:
    return _step(
        jnp.asarray(np.asarray(e["H"], float)[:nn]),
        jnp.asarray(np.asarray(e["THETA"], float)[:nn]),
        jnp.asarray(0.0),
        soil,
        grid,
        jnp.asarray(0.0),
        jnp.asarray(max(-_scalar(e["EVAP"]), 0.0)),
        jnp.asarray(np.asarray(e["QS"], float)[:nn]),
        jnp.asarray(dt),
        jnp.asarray(_scalar(e["ALPH"])),
        jnp.asarray(_scalar(e["HMIN"])),
        jnp.asarray(0.0),
        cfg=RichardsConfig(n_iter=20),
    )


def _errors(r: Any, x: dict[str, Any], tl: np.ndarray, nn: int, suffix: str = "") -> dict[str, float]:
    d = np.asarray(r.theta) - np.asarray(x["THETA"], float)[:nn]
    return {
        f"max_abs_dtheta{suffix}": float(np.max(np.abs(d))),
        f"abs_dstorage_cm{suffix}": float(abs(np.dot(d, tl))),
        f"max_abs_dh_cm{suffix}": float(np.max(np.abs(np.asarray(r.h) - np.asarray(x["H"], float)[:nn]))),
    }


def compare_case(path: Path, trhydp: np.ndarray | None) -> dict[str, float | str]:
    """One row of the per-call table (also used by the H1-3 measurement script)."""
    from agrijax.port.dumps import load_case

    c = load_case(path)
    e, x = c.entry, c.exit
    missing = [f"in.{k}" for k in REQUIRED_IN if k not in e] + [
        f"out.{k}" for k in REQUIRED_OUT if k not in x
    ]
    if missing:
        return {"case": path.name, "skip": f"missing fields {missing}"}
    if "IREBOT" in e and int(_scalar(e["IREBOT"])) != 2:
        return {"case": path.name, "skip": "bottom boundary is not free drainage"}
    if "ITBL" in e and int(_scalar(e["ITBL"])) != 0:
        return {"case": path.name, "skip": "water table present"}
    nn = int(_scalar(e["NN"]))
    tl = np.asarray(e["TL"], float)[:nn]
    delz = np.asarray(e["DELZ"], float)[: nn - 1]
    grid = RichardsGrid(tl=jnp.asarray(tl), delz=jnp.asarray(delz), dz_top=jnp.asarray(delz[0]))
    nh = np.asarray(e["NDXN2H"], int)[:nn] - 1
    hp = np.asarray(e["SOILHP"], float)  # SOILHP(13, MAXHOR), Fortran shape
    n_hor = int(nh.max()) + 1
    single = _soil(hp, n_hor, nn, nh)
    tilled = trhydp is not None and not np.array_equal(hp[:, :n_hor], trhydp[:, :n_hor])
    if trhydp is None:
        soil: AnyHydraulicParams = single
        cls = "no_trhydp"
    elif tilled:
        soil = TilledSoilHydraulicParams(current=single, original=_soil(trhydp, n_hor, nn, nh))
        cls = "tilled"
    else:
        soil, cls = single, "untilled"
    h_old = jnp.asarray(np.asarray(e["H"], float)[:nn])
    theta_in = np.asarray(e["THETA"], float)[:nn]
    offset = float(np.max(np.abs(np.asarray(theta_of_h(h_old, soil)) - theta_in)))
    offset_single = float(np.max(np.abs(np.asarray(theta_of_h(h_old, single)) - theta_in)))
    dt = _scalar(x["DELT"])
    r = _run_step(soil, e, dt, nn, grid)
    theta_ref = np.asarray(x["THETA"], float)[:nn]
    row: dict[str, float | str] = {
        "case": path.name,
        "skip": "",
        "class": cls,
        "delt_h": dt,
        "alpha": _scalar(e["ALPH"]),
        "curve_offset": offset,
        "curve_offset_single": offset_single,
        "max_dtheta_ref": float(np.max(np.abs(theta_ref - theta_in))),
        **_errors(r, x, tl, nn),
        "balance_error_cm": float(r.balance_error),
        "n_clamp": float(r.n_clamp),
        # evaporation our dry surface limit could not supply, and what RZWQM2 took (AEVAP, <= 0)
        "evap_deficit_cm": float(r.evaporation_deficit),
        "evap_demand_cm": max(-_scalar(e["EVAP"]), 0.0) * dt,
        "aevap_ref_cm": -_scalar(x["AEVAP"]) * dt,
    }
    if tilled:
        rs = _run_step(single, e, dt, nn, grid)
        row |= _errors(rs, x, tl, nn, "_single")
        row["same_as_single"] = bool(
            np.array_equal(np.asarray(r.theta), np.asarray(rs.theta))
            and np.array_equal(np.asarray(r.h), np.asarray(rs.h))
        )
    else:
        row |= {k + "_single": v for k, v in _errors(r, x, tl, nn).items()}
        row["same_as_single"] = True
    return row


@pytest.fixture(scope="module")
def table(data_dir: Path) -> pd.DataFrame:
    found = _cases(data_dir)
    if not found:
        pytest.skip(f"no {ROUTINE} dumps under {data_dir / 'dumps' / ROUTINE}")
    runs = {run_of_case(p) for p in found}
    trhydp = {r: saved_trhydp(data_dir, r) for r in runs}
    t = pd.DataFrame([compare_case(p, trhydp[run_of_case(p)]) for p in found])
    out = os.environ.get("AGRI_JAX_RICHARDS_REPORT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        t.to_csv(Path(out) / "richrd_dump_calls.csv", index=False, float_format="%.6g")
    return t


def _select(table: pd.DataFrame, cls: str) -> pd.DataFrame:
    mask = (table["skip"] == "") & (table.get("class", pd.Series("", index=table.index)) == cls)
    return cast(pd.DataFrame, table.loc[mask])


def _surface_limited(t: pd.DataFrame) -> pd.Series:
    """Our dry surface limit cut the evaporation while RZWQM2 delivered the full demand."""
    return (t["evap_deficit_cm"] > 0.0) & (t["aevap_ref_cm"] == t["evap_demand_cm"])


def _assert_grade_c(t: pd.DataFrame) -> None:
    t = cast(pd.DataFrame, t.loc[~_surface_limited(t)])
    bad = cast(
        pd.DataFrame,
        t.loc[
            (t["curve_offset"] >= CONSISTENT)
            | (t["max_abs_dtheta"] >= THETA_TOL)
            | (t["abs_dstorage_cm"] >= STORAGE_TOL)
            | (t["max_abs_dh_cm"] >= HEAD_TOL)
        ],
    )
    assert len(bad) == 0, f"{len(bad)} of {len(t)} RICHRD calls outside grade C:\n{bad.to_string()}"
    assert bool((t["n_clamp"] == 0).all())
    assert float(np.max(np.abs(t["balance_error_cm"].to_numpy()))) < 1e-12
    # the steps are not trivial: on every call RZWQM changed theta by at least 100 times our difference
    assert bool((t["max_dtheta_ref"] > 100 * t["max_abs_dtheta"]).all())


def test_every_richrd_call_has_the_start_up_curve(table: pd.DataFrame) -> None:
    """Every comparable call knows its pre-tillage curve (the run's WC dumps hold ``TRHYDP``)."""
    t = table.loc[table["skip"] == ""]
    if len(t) == 0:
        pytest.skip(f"no comparable {ROUTINE} case: {sorted(set(table['skip']))}")
    assert set(t["class"]) <= {"untilled", "tilled"}, sorted(set(t["class"]))


def test_richrd_untilled_calls_grade_c(table: pd.DataFrame) -> None:
    """Calls on the single-segment curve: converged steps agree with RZWQM2 to rounding level."""
    t = _select(table, "untilled")
    if len(t) == 0:
        pytest.skip(f"no comparable untilled {ROUTINE} case: {sorted(set(table['skip']))}")
    _assert_grade_c(t)


def test_richrd_tilled_calls_grade_c(table: pd.DataFrame) -> None:
    """Calls after tillage, on RZWQM2's two-segment curve: the same grade C as the untilled calls,
    and the single-segment curve would not reach it (the variant is what closes the gap)."""
    t = _select(table, "tilled")
    if len(t) == 0:
        pytest.skip("no tilled call among the dumps")
    _assert_grade_c(t)
    assert float(np.max(t["curve_offset_single"].to_numpy())) > 1e3 * CONSISTENT
    assert float(np.max(t["max_abs_dtheta_single"].to_numpy())) > 100 * THETA_TOL


def test_richrd_surface_limited_calls_differ_by_the_evaporation_deficit_only(table: pd.DataFrame) -> None:
    """Open difference (not the retention curve): on a call where the dry surface limit of
    :func:`~agrijax.processes.soil_water.richards.surface_fluxes` (ghost head ``HMIN``, geometric-mean
    ``K``) cuts the evaporation, RZWQM2 still delivered the full demand (``AEVAP = EVAP``). The storage
    difference is then exactly the evaporation deficit, on the single-segment and the two-segment
    curve alike; everything else of the step agrees."""
    t = cast(pd.DataFrame, table.loc[(table["skip"] == "") & _surface_limited(table)])
    if len(t) == 0:
        pytest.skip("no surface-limited call")
    diff = np.abs(t["abs_dstorage_cm"].to_numpy() - t["evap_deficit_cm"].to_numpy())
    assert float(np.max(diff)) < STORAGE_TOL, diff
    assert float(np.max(t["curve_offset"].to_numpy())) < CONSISTENT
