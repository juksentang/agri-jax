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

Two classes of call:

* *consistent*: ``in.THETA == theta_of_h(in.H)`` to 1e-12 on every node, i.e. the retention
  curve RZWQM evaluates is the single-segment curve of ``SOILHP``. Here the converged step
  must reproduce ``out.THETA`` to ``THETA_TOL`` and the storage change to ``STORAGE_TOL``
  (measured on 39 calls: max 2.1e-10 in theta, 1.4e-13 cm in storage, 2.6e-7 cm in head);
* *tillage-modified*: after a tillage event RZWQM2's ``WC`` evaluates the tilled curve
  (``SOILHP``) only above ``-10 hb`` and the pre-tillage curve below it (Ahuja et al. 1998,
  two-segment reconsolidation curve), which :mod:`hydraulics` does not model. Those calls are
  reported with their curve offset and only held to the looser ``TILLED_THETA_TOL``
  (measured on 61 calls: max 5.3e-5 in theta, 9.7e-5 cm in storage, where the curve offset
  is up to 4.8e-3).

With ``AGRI_JAX_RICHARDS_REPORT`` set (a directory) the per-call table is written there as
``richrd_dump_calls.csv``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams, theta_of_h
from agrijax.processes.soil_water.richards import RichardsConfig, RichardsGrid, richards_step

pytestmark = [
    pytest.mark.allow_skip(reason="RICHRD dumps are private data"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="dump comparison is float64"),
]

ROUTINE = "RICHRD"
THETA_TOL = 1e-8
STORAGE_TOL = 1e-10
HEAD_TOL = 1e-5
TILLED_THETA_TOL = 1e-4
CONSISTENT = 1e-12
REQUIRED_IN = ("DELZ", "TL", "NDXN2H", "NN", "SOILHP", "H", "THETA", "EVAP", "QS", "HMIN", "ALPH")
REQUIRED_OUT = ("H", "THETA", "DELT")

_step = jax.jit(richards_step, static_argnames=("cfg",))


def _cases(data_dir: Path) -> list[Path]:
    d = data_dir / "dumps" / ROUTINE
    return sorted(d.glob("*.npz")) if d.is_dir() else []


def _scalar(x: np.ndarray) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _compare_case(path: Path) -> dict[str, float | str]:
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
    soil = SoilHydraulicParams.from_rzwqm_records(hp[0:6, :n_hor].T, hp[6:13, :n_hor].T, node_horizon=nh)
    soil_n = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (nn,)), soil.at_nodes())
    h_old = np.asarray(e["H"], float)[:nn]
    theta_in = np.asarray(e["THETA"], float)[:nn]
    offset = float(np.max(np.abs(np.asarray(theta_of_h(jnp.asarray(h_old), soil_n)) - theta_in)))
    dt = _scalar(x["DELT"])
    r = _step(
        jnp.asarray(h_old),
        jnp.asarray(theta_in),
        jnp.asarray(0.0),
        soil_n,
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
    theta_ref = np.asarray(x["THETA"], float)[:nn]
    d = np.asarray(r.theta) - theta_ref
    return {
        "case": path.name,
        "skip": "",
        "class": "consistent" if offset < CONSISTENT else "tillage_modified",
        "delt_h": dt,
        "alpha": _scalar(e["ALPH"]),
        "curve_offset": offset,
        "max_dtheta_ref": float(np.max(np.abs(theta_ref - theta_in))),
        "max_abs_dtheta": float(np.max(np.abs(d))),
        "abs_dstorage_cm": float(abs(np.dot(d, tl))),
        "max_abs_dh_cm": float(np.max(np.abs(np.asarray(r.h) - np.asarray(x["H"], float)[:nn]))),
        "balance_error_cm": float(r.balance_error),
        "n_clamp": float(r.n_clamp),
    }


@pytest.fixture(scope="module")
def table(data_dir: Path) -> pd.DataFrame:
    found = _cases(data_dir)
    if not found:
        pytest.skip(f"no {ROUTINE} dumps under {data_dir / 'dumps' / ROUTINE}")
    t = pd.DataFrame([_compare_case(p) for p in found])
    out = os.environ.get("AGRI_JAX_RICHARDS_REPORT")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        t.to_csv(Path(out) / "richrd_dump_calls.csv", index=False, float_format="%.6g")
    return t


def _select(table: pd.DataFrame, cls: str) -> pd.DataFrame:
    mask = (table["skip"] == "") & (table.get("class", pd.Series("", index=table.index)) == cls)
    return cast(pd.DataFrame, table.loc[mask])


def test_richrd_consistent_calls_grade_c(table: pd.DataFrame) -> None:
    """Calls on the single-segment curve: converged steps agree with RZWQM2 to rounding level."""
    t = _select(table, "consistent")
    if len(t) == 0:
        pytest.skip(f"no comparable {ROUTINE} case: {sorted(set(table['skip']))}")
    bad = cast(
        pd.DataFrame,
        t.loc[
            (t["max_abs_dtheta"] >= THETA_TOL)
            | (t["abs_dstorage_cm"] >= STORAGE_TOL)
            | (t["max_abs_dh_cm"] >= HEAD_TOL)
        ],
    )
    assert len(bad) == 0, f"{len(bad)} of {len(t)} RICHRD calls outside grade C:\n{bad.to_string()}"
    assert bool((t["n_clamp"] == 0).all())
    assert float(np.max(np.abs(t["balance_error_cm"].to_numpy()))) < 1e-12
    # the steps are not trivial: RZWQM changed theta by more than the tolerance on every call
    assert bool((t["max_dtheta_ref"] > 100 * THETA_TOL).all())


def test_richrd_tillage_modified_calls_bounded(table: pd.DataFrame) -> None:
    """Calls on RZWQM2's two-segment tilled curve: the residual difference is bounded, not zero."""
    t = _select(table, "tillage_modified")
    if len(t) == 0:
        pytest.skip("no tillage-modified call among the dumps")
    worst = float(np.max(t["max_abs_dtheta"].to_numpy()))
    assert worst < TILLED_THETA_TOL, worst
    assert bool((t["n_clamp"] == 0).all())
