"""H1-5: the Green-Ampt event against RZWQM2 4.6 ``EVNTRO``/``INFIL`` dumps on 19 scenarios, runoff storms included.

Helper module of ``test_green_ampt_scenarios.py`` and a script
(``python tests/integration/green_ampt_scenarios.py``) that writes the per-event table
``events.csv`` and the per-scenario summary ``summary.json`` to ``<data>/validation/h1_5_green_ampt/``.

Reference data. RZWQM2 4.6 (source tree ``RZWQM_Linux_Ver45``, ifx 2025.2, ``-fp-model precise``)
instrumented at ``EVNTRO`` and ``INFIL`` (the A12 build of the reference-data track; every call
dumped, entry and exit), run with :func:`agrijax.port.run_fortran.run_rzwqm` over the whole
``IPNAMES.DAT`` period of

* the 15 scenarios of ``narval_mirror/RZWQM_sw_batch`` (:data:`BATCH`);
* the 4 tile-drained scenarios shared by the research group (narval
  ``/project/def-zhiming/sharing/RZWQM_Scenario_Runtime``, mirrored to
  ``narval_mirror/tile_scenarios``): ``Ohio-td_1`` and ``Ohio-td_2`` (TM2 Ohio, 2011-2019; the two
  folders hold the same model inputs, so their runs are identical), ``Lanna-td`` (Lanna plot 71,
  1998-2014) and ``Melby-td`` (Mellby, 1987-2004) (:data:`TILE`).

The instrumented build's ``.ana`` (and ``LAYER.PLT`` where the scenario writes one; the Ohio
scenarios do not) is identical to that of the plain build with the same flags on every scenario
(checked by the private collection script, recorded in ``collect.json``). One ``npz`` case per
storm event and routine, with the fields read here, is exported to
``<data>/dumps/h1_5_green_ampt/<site>/{EVNTRO,INFIL}/`` (:mod:`agrijax.port.dumps`).

Per event, as in ``test_infiltration_dumps.py``: the storm is run through
:func:`~agrijax.processes.soil_water.infiltration.green_ampt_event` from its dumped entry state
(grid ``TLT[:NN]``, ``SOILHP`` with ``NDXN2H``, ``THETA``/``H``, ``AEF``, breakpoints
``BPWHEN``/``BPMUCH``), and compared with

* infiltration ``CII`` and the ``INFIL`` cumulative curve at its last front step ``CI(NTIM)``;
* runoff ``ROI``;
* the infiltration capacity, three ways: the wetting-front suction per node ``SWF[:NN]``
  (``INFIL`` entry, the values the capacity uses) against :func:`wetting_front_suction`; the front
  conductance per node ``CNN[:NN]`` (``INFIL`` entry) against :func:`front_conductance`; and the
  rate of the last front step ``VFIN = min(V, max(RR, rr_min))`` (``INFIL`` exit, ``V`` the layered
  capacity at the final front slice ``ID``) against :func:`green_ampt_capacity` at the same slice
  and rain intensity (when the profile did not saturate, ``ISAT = 0``);
* the front position (``DWF + ds/2``), the Green-Ampt clock ``TR(NTIM)`` (a sum of ``dq / v`` over
  every front step, i.e. every step's rate), and the post-event water contents ``THETA[:NN]``.

Tillage. Where a storm's ``SOILHP`` differs from the run's start-up copy (tillage and
reconsolidation), RZWQM2 reads the initial suction from the two-segment retention curve (``WCH``
with ``TRHYDP``), and the event is given :class:`TilledSoilHydraulicParams`. The start-up copy is
taken as the ``SOILHP`` of the run's first storm (every run starts on 1 January, before the
first tillage); the pre-tillage storms, which use it untilled, and the post-tillage storms, which
use it as ``TRHYDP``, all agree to rounding on the 15 batch scenarios, which they would not with a
wrong copy.

Event classes. Snowmelt events (``SMELT > 0``) enter ``EVNTRO`` without breakpoints (L-snow);
irrigation events (``AIRR > 0``) are left out too, but not because they lack breakpoints: 10,216 of
the 11,245 carry ``BPWHEN(1) != 0`` and still do not match the kernel from their entry state (the
entry breakpoints probably do not hold the irrigation water, or EVNTRO treats it elsewhere; open
item L-irr, to be traced before any irrigated site is compared); only the 1,011 at
``US_Rockford_Alfalfa`` take the ponded entry. Both are counted and left out, as at CA-TPA, and so are
the *empty* events that enter the same way with no water (``BPWHEN(1) = 0``, ``AIRR = SMELT = 0``;
371 at ``US_Rockford_Alfalfa``, with ``CII`` of order 1e-17 cm and no runoff). For every
Green-Ampt branch of RZWQM2 we have not ported (:data:`UNPORTED`, deferred as L-GA+) the dumped
switches tell whether it is on and whether it acts on the storm (e.g. the water-table branch
only runs when the saturated zone is inside the infiltration grid); the decision reads the
switches and the entry state, never the size of the difference. A rain storm on which no unported
branch acts is *checked* against the tolerance; the others are *excluded* and reported per branch
with their differences.

Tolerance: the kernel follows the same arithmetic, so in float64 the bound is rounding level,
:data:`TOL_CM` = 1e-12 (cm, cm h-1, h; the level of ``test_infiltration_dumps.py``, measured there
1.8e-15 cm).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.port.dumps import DumpCase, load_case
from agrijax.processes.soil_water.hydraulics import (
    AnyHydraulicParams,
    SoilHydraulicParams,
    TilledSoilHydraulicParams,
)
from agrijax.processes.soil_water.infiltration import (
    RR_MIN,
    GreenAmptConfig,
    front_conductance,
    green_ampt_capacity,
    green_ampt_event,
    slice_nodes,
    wetting_front_suction,
)

BATCH = (
    "CA-ER1",
    "CA-MA1",
    "CA-TPA",
    "US-LYS_NW",
    "US-LYS_SE",
    "US-LYS_SW",
    "US-Mj1",
    "US-S2",
    "US-TW3",
    "US-Tw2",
    "US-UA1_HartFarm",
    "US-manilacotton",
    "US_OPE",
    "US_Rockfish",
    "US_Rockford_Alfalfa",
)
TILE = ("Ohio-td_1", "Ohio-td_2", "Lanna-td", "Melby-td")
SITES = BATCH + TILE
SOURCE = {
    **{s: "RZWQM_sw_batch (viveka scenarios from sw, group share, prepared 2026-09-23)" for s in BATCH},
    **{
        s: "research group tile-drain scenario (narval def-zhiming share RZWQM_Scenario_Runtime)"
        for s in TILE
    },
}
CASES = Path("dumps/h1_5_green_ampt")
OUT = Path("validation/h1_5_green_ampt")
PREFIX = "h1ga"
TOL_CM = 1e-12
#: breakpoint arrays are padded with zero-length, zero-depth intervals to a multiple of this (fewer
#: compilations; the padding leaves the event bit for bit unchanged: zero depth adds nothing to the
#: cumulative rain, and the intensity lookup finds a real interval first)
_BP_PAD = 8


def _f(x: Any) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _i(x: Any) -> int:
    return round(_f(x))


@dataclass(frozen=True)
class _Ev:
    """The four dumped records of one storm: EVNTRO entry/exit, INFIL entry/exit."""

    e: dict[str, np.ndarray]
    x: dict[str, np.ndarray]
    i: dict[str, np.ndarray]
    ix: dict[str, np.ndarray]


def _water_table_in_profile(c: _Ev) -> bool:
    """``ITBL = 1`` and the top of the saturated zone inside the infiltration grid: INFIL then calls
    SATFLO/FULFLO at every front step above it and never takes the saturated-profile branch."""
    return _i(c.e["ITBL"]) == 1 and int(np.asarray(c.i["H2OTAB"], float).reshape(-1)[1]) <= _i(c.i["NSLT"])


def _pori_limited(c: _Ev) -> bool:
    nn = _i(c.e["NN"])
    nh = np.asarray(c.e["NDXN2H"], int)[:nn] - 1
    avail = np.asarray(c.e["SOILHP"], float)[5, nh] * _f(c.e["AEF"])
    return bool(np.any(np.asarray(c.e["PORI"], float)[:nn] != avail))


#: Green-Ampt branches of RZWQM2 4.6 that the kernel does not have (deferred, L-GA+):
#: name -> (what, where, switched on, acts on this storm). "Switched on" reads the scenario switches
#: from the dumped records; "acts" is the condition under which the branch's code runs and can change
#: the event (from the dumped entry state; for macropores, the dumped macropore flow). A storm whose
#: switched-on branches do not act runs only the ported code, so it stays in the checked set.
#: Line numbers: RZWQM_Linux_Ver45/src/RZWQM/RZTEST.for.
UNPORTED: dict[str, tuple[str, str, Any, Any]] = {
    "crust": (
        "surface crust: conductance of the top node from the crust conductivity CRUSTK",
        "EVNTRO RZTEST.for:656-666 (ICRUST /= 0 and CRUSTK /= 0)",
        lambda c: _i(c.e["ICRUST"]) != 0 and _f(c.e["CRUSTK"]) != 0.0,
        lambda c: True,
    ),
    "macropore": (
        "macropores: flow of the excess rain (MPFLOW) and the EVNTRO macropore redistribution after INFIL",
        "INFIL RZTEST.for:1468-1488, EVNTRO RZTEST.for:845-931, MPFLOW RZTEST.for:2537 (YESMAC >= 1)",
        lambda c: _i(c.e["YESMAC"]) >= 1,
        lambda c: True,
    ),
    "water_table": (
        "water table in the profile: saturated flow below the front and tile drainage (SATFLO, FULFLO)",
        "INFIL RZTEST.for:1490-1529, SATFLO :2167, FULFLO :2347 (ITBL = 1; acts when INT(H2OTAB(2)) <= NSLT)",
        lambda c: _i(c.e["ITBL"]) == 1,
        _water_table_in_profile,
    ),
    "ugflow": (
        "unit-gradient flow below the front (UGFLOW)",
        "INFIL RZTEST.for:1530-1536, UGFLOW RZTEST.for:3281 (IREBOT = 2 and IUGFLW = 1; acts when the "
        "water-table branch does not and the front stops above the bottom slice)",
        lambda c: _i(c.e["IREBOT"]) == 2 and _i(c.i["IUGFLW"]) == 1,
        lambda c: not _water_table_in_profile(c) and _i(c.ix["ID"]) < _i(c.i["NSLT"]),
    ),
    "bottom_flux": (
        "rate through a saturated profile capped by the bottom flux BOTFLX",
        "INFIL RZTEST.for:1387, 1537-1600 (IREBOT = 3; acts when the profile saturates, ISAT = 1, "
        "with no water table in it)",
        lambda c: _i(c.e["IREBOT"]) == 3,
        lambda c: _i(c.ix["ISAT"]) == 1 and not _water_table_in_profile(c),
    ),
    "ice": (
        "available porosity PORI below theta_s AEF (ice) on a node; conductivity limited by PORI",
        "EVNTRO RZTEST.for:429-436 and 684-691 (PORI /= SOILHP(6) AEF)",
        _pori_limited,
        lambda c: True,
    ),
    "plastic": (
        "plastic mulch: rain reduced by min(1, -10 (pplastic - 1))",
        "EVNTRO RZTEST.for:474-504 (PPLASTIC > 0.9)",
        lambda c: _f(c.e["PPLASTIC"]) > 0.9,
        lambda c: True,
    ),
}


@dataclass
class EventRow:
    site: str
    case: str
    date: int
    kind: str  # rain | snowmelt | irrigation
    switched: str = ""  # ';'-joined UNPORTED names switched on
    branches: str = ""  # ';'-joined UNPORTED names that act on this storm
    tilled: bool = False  # SOILHP differs from the run's start-up copy: two-segment retention (H1-3)
    error: str = ""  # exception raised by the kernel or the setup
    rain: float = math.nan
    cii_ref: float = math.nan
    roi_ref: float = math.nan
    cii: float = math.nan
    roi: float = math.nan
    d_cii: float = math.nan
    d_ci_last: float = math.nan
    d_roi: float = math.nan
    d_swf: float = math.nan
    d_cnn: float = math.nan
    d_vfin: float = math.nan
    d_front: float = math.nan
    d_duration: float = math.nan
    d_theta: float = math.nan
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def runoff(self) -> bool:
        return self.roi_ref > 0.0

    @property
    def portable(self) -> bool:
        """A rain storm on which no unported branch acts, run without error: the checked set."""
        return self.kind == "rain" and not self.branches and not self.error


def event_kind(e: dict[str, np.ndarray]) -> str:
    """``snowmelt``, ``irrigation``, ``empty`` (no breakpoint rain, ``BPWHEN(1) = 0``, and no water:
    EVNTRO's ponded entry with ``AIRR = 0``, RZTEST.for:506-513) or ``rain``."""
    if _f(e["SMELT"]) > 0.0:
        return "snowmelt"
    if _f(e["AIRR"]) > 0.0:
        return "irrigation"
    if _f(np.asarray(e["BPWHEN"]).reshape(-1)[0]) == 0.0:
        return "empty"
    return "rain"


def _capacity(theta: Any, soil: AnyHydraulicParams, tl: Any, aef: Any, cfg: GreenAmptConfig) -> Any:
    cur = soil.current if isinstance(soil, TilledSoilHydraulicParams) else soil
    suction = wetting_front_suction(theta, soil, cur.theta_s * aef)
    cond = front_conductance(soil)
    node, zc = slice_nodes(tl, cfg)
    cap = green_ampt_capacity(tl, cond, suction, node, zc, cfg)
    return suction, cond, cap


_event = jax.jit(green_ampt_event, static_argnames="cfg")
_cap = jax.jit(_capacity, static_argnames="cfg")


def _pad(x: np.ndarray) -> np.ndarray:
    n = -(-len(x) // _BP_PAD) * _BP_PAD
    return np.concatenate([x, np.zeros(n - len(x))])


def _soil(hp: np.ndarray, nh: np.ndarray, nn: int) -> SoilHydraulicParams:
    """``SOILHP(13, MAXHOR)`` (Fortran shape) -> parameters gathered onto the ``nn`` nodes."""
    n_hor = int(nh.max()) + 1
    soil = SoilHydraulicParams.from_rzwqm_records(hp[0:6, :n_hor].T, hp[6:13, :n_hor].T, node_horizon=nh)
    return jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (nn,)), soil.at_nodes())


def compare_event(site: str, name: str, ev: DumpCase, inf: DumpCase, startup_soilhp: np.ndarray) -> EventRow:
    """One dumped storm from its entry state; differences are ours - RZWQM2.

    ``startup_soilhp`` is the run's start-up ``SOILHP`` (``TRHYDP``): when the storm's ``SOILHP``
    differs (tillage, reconsolidation), the event gets the two-segment retention curve RZWQM2 uses
    after tillage (:class:`TilledSoilHydraulicParams`).
    """
    e, x = ev.entry, ev.exit
    c = _Ev(e, x, inf.entry, inf.exit)
    row = EventRow(site=site, case=name, date=ev.date, kind=event_kind(e))
    on = [k for k, (_, _, sw, _) in UNPORTED.items() if sw(c)]
    row.switched = ";".join(on)
    row.branches = ";".join(k for k in on if UNPORTED[k][3](c))
    row.cii_ref, row.roi_ref = _f(x["CII"]), _f(x["ROI"])
    row.extra = {
        "fmp_in": _f(e["FMP"]),
        "fmp_out": _f(x["FMP"]),
        "isat": float(_f(inf.exit["ISAT"])),
        "h2otab": float(np.asarray(inf.entry["H2OTAB"], float).reshape(-1)[1]),
        "nslt": _f(inf.entry["NSLT"]),
        "id_out": _f(inf.exit["ID"]),
    }
    if row.kind != "rain":
        return row
    try:
        nn = _i(e["NN"])
        tlt = np.asarray(e["TLT"], float)[:nn]
        tl = np.diff(np.concatenate([[0.0], tlt]))
        nh = np.asarray(e["NDXN2H"], int)[:nn] - 1
        hp = np.asarray(e["SOILHP"], float)  # SOILHP(13, MAXHOR), Fortran shape
        n_hor = int(nh.max()) + 1
        soil: AnyHydraulicParams = _soil(hp, nh, nn)
        row.tilled = not np.array_equal(hp[:, :n_hor], startup_soilhp[:, :n_hor])
        if row.tilled:
            soil = TilledSoilHydraulicParams(current=soil, original=_soil(startup_soilhp, nh, nn))
        nbp = _i(e["NBP"])
        when = np.asarray(e["BPWHEN"], float)[:nbp]
        much = np.asarray(e["BPMUCH"], float)[:nbp]
        cfg = GreenAmptConfig.for_grid(tl)
        theta0 = jnp.asarray(np.asarray(e["THETA"], float)[:nn])
        aef = jnp.asarray(_f(e["AEF"]))
        r = _event(
            theta0,
            jnp.asarray(np.asarray(e["H"], float)[:nn]),
            soil,
            jnp.asarray(tl),
            aef,
            jnp.asarray(_pad(np.diff(np.concatenate([[0.0], when])))),
            jnp.asarray(_pad(np.diff(np.concatenate([[0.0], much])))),
            cfg=cfg,
        )
        suction, cond, cap = (np.asarray(a) for a in _cap(theta0, soil, jnp.asarray(tl), aef, cfg=cfg))
        cur_theta_s = (soil.current if isinstance(soil, TilledSoilHydraulicParams) else soil).theta_s
    except Exception as ex:  # reported per event
        row.error = f"{type(ex).__name__}: {ex}"
        return row
    nt = _i(inf.exit["NTIM"])
    row.rain = float(r.rain)
    row.cii, row.roi = float(r.infiltration), float(r.runoff)
    row.d_cii = row.cii - row.cii_ref
    row.d_ci_last = row.cii - float(np.asarray(inf.exit["CI"])[nt - 1])
    row.d_roi = row.roi - row.roi_ref
    swf_ref = inf.entry["SWF"] if "SWF" in inf.entry else x["SWF"]  # INFIL entry: the suction INFIL uses
    d_swf = np.abs(suction - np.asarray(swf_ref, float)[:nn])
    # only nodes below the available porosity can hold the front; on a node at or above it (all its
    # slices saturated, skipped by the front) RZWQM2's WCH returns its input head -hb when a1 = 0,
    # the kernel h = 0 (hydraulics.h_of_theta), and the suction there never enters the capacity
    unsat = np.asarray(theta0) < np.asarray(cur_theta_s) * _f(e["AEF"])
    row.d_swf = float(np.max(d_swf, where=unsat, initial=0.0))
    row.extra["d_swf_saturated_nodes"] = float(np.max(d_swf, where=~unsat, initial=0.0))
    row.d_cnn = float(np.max(np.abs(cond - np.asarray(inf.entry["CNN"], float)[:nn])))
    row.d_front = float(r.front_depth) - (_f(inf.exit["DWF"]) + 0.5 * cfg.ds)
    row.d_duration = float(r.duration) - float(np.asarray(inf.exit["TR"])[nt - 1])
    row.d_theta = float(np.max(np.abs(np.asarray(r.theta) - np.asarray(x["THETA"], float)[:nn])))
    j = _i(inf.exit["ID"]) - 1  # RZWQM2's last front slice (ID, 1-based): the capacity at the same front
    if _i(inf.exit["ISAT"]) == 0 and _i(inf.exit["NTIM"]) > 0 and 0 <= j < cfg.n_slice:
        v = min(float(cap[j]), max(_f(inf.exit["RR"]), RR_MIN))
        if _i(e["IREBOT"]) == 3:  # INFIL caps the reported final rate at the bottom flux (RZTEST.for:1387)
            v = min(v, _f(e["BOTFLX"]))
        row.d_vfin = v - _f(inf.exit["VFIN"])
    return row


def site_rows(data_dir: Path, site: str) -> list[EventRow]:
    """Every dumped event of one scenario (empty when the cases are missing)."""
    root = data_dir / CASES / site
    files = sorted((root / "EVNTRO").glob(f"{PREFIX}_d*.npz"))
    rows: list[EventRow] = []
    startup: np.ndarray | None = None
    for f in files:
        inf = root / "INFIL" / f.name
        if not inf.is_file():
            raise FileNotFoundError(f"INFIL case missing for {site}/{f.name}")
        ev = load_case(f)
        if startup is None:  # start-up SOILHP: the first storm of the run (module docstring, "Tillage")
            startup = np.asarray(ev.entry["SOILHP"], float)
        rows.append(compare_event(site, f.name, ev, load_case(inf), startup))
    return rows


DIFFS = ("d_cii", "d_ci_last", "d_roi", "d_swf", "d_cnn", "d_vfin", "d_front", "d_duration", "d_theta")


def _stats(rows: list[EventRow]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for k in DIFFS:
        v = np.array([getattr(r, k) for r in rows], float)
        v = v[np.isfinite(v)]
        out[k] = {
            "n": int(v.size),
            "max_abs": float(np.max(np.abs(v))) if v.size else None,
            "rmse": float(np.sqrt(np.mean(v**2))) if v.size else None,
        }
    return out


def _exact(r: EventRow) -> bool:
    return not r.error and all(abs(getattr(r, k)) < TOL_CM for k in ("d_cii", "d_roi", "d_theta"))


def summary(rows: list[EventRow]) -> dict[str, Any]:
    """Counts and error statistics of one scenario's events."""
    rain = [r for r in rows if r.kind == "rain"]
    port = [r for r in rain if r.portable]
    excl = [r for r in rain if r.branches]
    s: dict[str, Any] = {
        "events": len(rows),
        "snowmelt": sum(r.kind == "snowmelt" for r in rows),
        "irrigation": sum(r.kind == "irrigation" for r in rows),
        "empty": sum(r.kind == "empty" for r in rows),
        "rain": len(rain),
        "rain_runoff": sum(r.runoff for r in rain),
        "rain_errors": sum(bool(r.error) for r in rain),
        "checked": len(port),
        "checked_runoff": sum(r.runoff for r in port),
        "checked_tilled": sum(r.tilled for r in port),
        "checked_exact": sum(_exact(r) for r in port),
        "checked_runoff_ref_cm": float(sum(r.roi_ref for r in port if r.runoff)),
        "checked_stats": _stats(port),
        "checked_runoff_stats": _stats([r for r in port if r.runoff]),
        "excluded": len(excl),
        "excluded_runoff": sum(r.runoff for r in excl),
        "excluded_exact": sum(_exact(r) for r in excl),
        "excluded_stats": _stats([r for r in excl if not r.error]),
        "excluded_runoff_stats": _stats([r for r in excl if r.runoff and not r.error]),
        "branches": {},
    }
    for name in UNPORTED:
        sw = [r for r in rain if name in r.switched.split(";")]
        on = [r for r in rain if name in r.branches.split(";")]
        if sw:
            s["branches"][name] = {
                "switched_on": len(sw),
                "acts": len(on),
                "acts_runoff": sum(r.runoff for r in on),
                "acts_exact": sum(_exact(r) for r in on),
                "acts_stats": _stats([r for r in on if not r.error]),
            }
    s["error_kinds"] = sorted({r.error[:160] for r in rain if r.error})
    return s


def write_report(data_dir: Path, sites: tuple[str, ...] = SITES) -> dict[str, Any]:
    out = data_dir / OUT
    out.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {
        "tol_cm": TOL_CM,
        "unported": {k: list(v[:2]) for k, v in UNPORTED.items()},
        "sites": {},
    }
    all_rows: list[EventRow] = []
    for site in sites:
        rows = site_rows(data_dir, site)
        all_rows += rows
        rep["sites"][site] = {"source": SOURCE[site], **summary(rows)}
        keys = ("events", "rain", "rain_runoff", "checked", "checked_runoff", "checked_exact", "excluded")
        print(site, json.dumps({k: rep["sites"][site][k] for k in (*keys, "rain_errors")}), flush=True)
    rep["all"] = summary(all_rows)
    with open(out / "events.csv", "w", newline="") as fh:
        cols = [k for k in asdict(all_rows[0]) if k != "extra"] if all_rows else []
        ex = sorted({k for r in all_rows for k in r.extra})
        w = csv.writer(fh)
        w.writerow([*cols, *ex])
        for r in all_rows:
            d = asdict(r)
            w.writerow([d[c] for c in cols] + [r.extra.get(k, "") for k in ex])
    (out / "summary.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    d = Path(os.environ.get("AGRI_JAX_DATA", Path.home() / "agri_jax_data"))
    write_report(d, tuple(sys.argv[1:]) or SITES)
