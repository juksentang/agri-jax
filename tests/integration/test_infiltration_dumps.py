"""Plan 19 A5 level 2: the Green-Ampt event kernel against RZWQM2 4.6 ``EVNTRO``/``INFIL`` dumps.

The reference-data track instrumented RZWQM2 4.6 (plan 19, note of 2026-09-24) and wrote one
``npz`` case per storm event (:mod:`agrijax.port.dumps`: ``in.<NAME>`` / ``out.<NAME>``, Fortran
shapes) for two CA-TPA runs:

* ``<data>/dumps/{EVNTRO,INFIL}/catpa2015_*.npz`` (2015: 107 events);
* ``<data>/dumps/tables/rzwqm46_catpa2015_2023/cases/{EVNTRO,INFIL}/catpa2015_2023_*.npz``
  (2015-2023: 1024 events).

``EVNTRO`` and ``INFIL`` cases of the same event share the file name. Each storm is run through
:func:`~agrijax.processes.soil_water.infiltration.green_ampt_event` from its dumped entry state
(resynchronised per event): grid ``TLT[:NN]``, hydraulics ``SOILHP(13, horizon)`` with the node map
``NDXN2H``, ``THETA``/``H`` on the nodes, ``AEF``, and the breakpoints ``BPWHEN``/``BPMUCH``
(cumulative time [h] and depth [cm], ``NBP`` of them). Compared per event:

* cumulative infiltration ``CII`` (``EVNTRO`` exit) and the ``INFIL`` cumulative curve at its
  last front step ``CI(NTIM)``;
* runoff ``ROI``;
* front position: ``INFIL`` ``DWF`` is the centre of the deepest slice filled, our
  ``front_depth`` is its bottom, so ``front_depth = DWF + ds/2``; and the event duration on the
  Green-Ampt clock ``TR(NTIM)``;
* post-event water contents ``THETA[:NN]`` (``EVNTRO`` exit).

Scope: rain storms only. Events that carry snowmelt (``SMELT > 0``) or irrigation (``AIRR > 0``)
enter ``EVNTRO`` without breakpoints (``NBP`` points at zero depth) through the snowmelt/irrigation
entry, which is not part of the M3 event kernel (L-snow; snow is added in the M3 assembly). They
are counted, and the count is pinned so a change in the reference is noticed.

Tolerances. The plan (A5 level 2) asks for the dumped quantities to be reproduced from the dumped
entry state; the kernel follows the same arithmetic, so in float64 the bound is rounding level:
``TOL_CM = 1e-12`` cm, ``TOL_THETA = 1e-12`` and ``TOL_H = 1e-12`` h (measured 2026-09-24: max
1.8e-15 cm in ``CII``, 1.7e-15 cm in ``ROI``, 2.5e-16 in ``THETA``, front positions identical,
over 87 + 829 rain storms). In float32 (``AGRI_JAX_X64=0``) the bounds are ``1e-5``
(measured max 5.5e-7 cm, 3.0e-7 in ``THETA``).

No ``allow_skip``: the test skips when the dumps are missing, and ``AGRI_JAX_NO_SKIP=1`` turns
that skip into a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.port.dumps import DumpCase, load_case
from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams
from agrijax.processes.soil_water.infiltration import GreenAmptConfig, green_ampt_event

#: run -> (EVNTRO dir, INFIL dir, rain storms, snowmelt/irrigation events), relative to ``<data>/dumps``
RUNS: dict[str, tuple[Path, Path, int, int]] = {
    "catpa2015": (Path("EVNTRO"), Path("INFIL"), 87, 20),
    "catpa2015_2023": (
        Path("tables/rzwqm46_catpa2015_2023/cases/EVNTRO"),
        Path("tables/rzwqm46_catpa2015_2023/cases/INFIL"),
        829,
        195,
    ),
}

_event = jax.jit(green_ampt_event, static_argnames="cfg")


def _tolerances() -> tuple[float, float, float]:
    """``(cm, theta, h)`` bounds: rounding level in float64, float32 rounding otherwise."""
    return (1e-12, 1e-12, 1e-12) if jax.config.read("jax_enable_x64") else (1e-5, 1e-5, 1e-5)


def _f(x: np.ndarray) -> float:
    return float(np.asarray(x).reshape(-1)[0])


@dataclass(frozen=True)
class EventResult:
    name: str
    d_cii: float
    d_ci_last: float
    d_roi: float
    d_front: float
    d_duration: float
    d_theta: float
    runoff: float


def _is_rain_storm(e: dict[str, np.ndarray]) -> bool:
    return _f(e["SMELT"]) <= 0.0 and _f(e["AIRR"]) <= 0.0


def compare_event(name: str, ev: DumpCase, inf: DumpCase) -> EventResult:
    """Run one dumped rain storm through the kernel from its entry state; differences ours - RZWQM2."""
    e, x = ev.entry, ev.exit
    nn = int(_f(e["NN"]))
    tlt = np.asarray(e["TLT"], float)[:nn]
    tl = np.diff(np.concatenate([[0.0], tlt]))
    nh = np.asarray(e["NDXN2H"], int)[:nn] - 1
    hp = np.asarray(e["SOILHP"], float)  # SOILHP(13, MAXHOR), Fortran shape
    n_hor = int(nh.max()) + 1
    soil = SoilHydraulicParams.from_rzwqm_records(hp[0:6, :n_hor].T, hp[6:13, :n_hor].T, node_horizon=nh)
    soil_n = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (nn,)), soil.at_nodes())
    nbp = int(_f(e["NBP"]))
    when = np.asarray(e["BPWHEN"], float)[:nbp]
    much = np.asarray(e["BPMUCH"], float)[:nbp]
    cfg = GreenAmptConfig.for_grid(tl)
    r = _event(
        jnp.asarray(np.asarray(e["THETA"], float)[:nn]),
        jnp.asarray(np.asarray(e["H"], float)[:nn]),
        soil_n,
        jnp.asarray(tl),
        jnp.asarray(_f(e["AEF"])),
        jnp.asarray(np.diff(np.concatenate([[0.0], when]))),
        jnp.asarray(np.diff(np.concatenate([[0.0], much]))),
        cfg=cfg,
    )
    nt = int(_f(inf.exit["NTIM"]))
    return EventResult(
        name=name,
        d_cii=float(r.infiltration) - _f(x["CII"]),
        d_ci_last=float(r.infiltration) - float(np.asarray(inf.exit["CI"])[nt - 1]),
        d_roi=float(r.runoff) - _f(x["ROI"]),
        d_front=float(r.front_depth) - (_f(inf.exit["DWF"]) + 0.5 * cfg.ds),
        d_duration=float(r.duration) - float(np.asarray(inf.exit["TR"])[nt - 1]),
        d_theta=float(np.max(np.abs(np.asarray(r.theta) - np.asarray(x["THETA"], float)[:nn]))),
        runoff=_f(x["ROI"]),
    )


@pytest.mark.parametrize("run", sorted(RUNS))
def test_event_dumps(data_dir: Path, run: str) -> None:
    """Level 2: every dumped rain storm from its dumped entry profile; CII, CI, ROI, front, THETA."""
    ev_dir, inf_dir, n_rain, n_other = RUNS[run]
    root = data_dir / "dumps"
    files = sorted((root / ev_dir).glob(f"{run}_d*.npz"))
    if not files:
        pytest.skip(f"no EVNTRO dumps for {run} under {root / ev_dir}")
    tol_cm, tol_theta, tol_h = _tolerances()
    results: list[EventResult] = []
    other = 0
    for f in files:
        ev = load_case(f)
        if not _is_rain_storm(ev.entry):
            other += 1
            continue
        inf_path = root / inf_dir / f.name
        assert inf_path.is_file(), f"INFIL case missing for {f.name}"
        results.append(compare_event(f.name, ev, load_case(inf_path)))
    assert (len(results), other) == (n_rain, n_other)

    def worst(attr: str) -> tuple[float, str]:
        k = int(np.argmax([abs(getattr(r, attr)) for r in results]))
        return abs(getattr(results[k], attr)), results[k].name

    assert worst("d_cii")[0] < tol_cm, worst("d_cii")
    assert worst("d_ci_last")[0] < tol_cm, worst("d_ci_last")
    assert worst("d_roi")[0] < tol_cm, worst("d_roi")
    assert worst("d_front")[0] == 0.0, worst("d_front")
    assert worst("d_duration")[0] < tol_h, worst("d_duration")
    assert worst("d_theta")[0] < tol_theta, worst("d_theta")
    # the storms with runoff are part of the comparison (2 of 1024 events in 2015-2023, none in 2015)
    n_runoff = sum(r.runoff > 0.0 for r in results)
    assert n_runoff == (0 if run == "catpa2015" else 2)
