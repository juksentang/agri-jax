"""The DSSAT tipping bucket (``processes/soil_water/bucket``) day by day against dscsm048 v4.8.6.0.

Reference: the dumps of ``WATBAL`` and its callees (``bucket_dssat.py``): a gfortran 13.3
build of the BSD-3 v4.8.6.0 source instrumented at ``WATBAL``, ``RNOFF``, ``INFIL``, ``SATFLO``,
``UP_FLOW``, ``SNOWFALL`` and ``MULCHWATER``, whose printed outputs equal build486's for every run
of the set: the 58 M2 maize treatments (nitrogen off, one-treatment batch), the 14 runs of the
CA-TPA DSSAT case and the winter-wheat examples KSAS8101 (6 runs, the only ones with snow) and
SWSW7501 (14 runs): 92 runs, 40 598 days. The crop is replayed: ``SPAM``'s soil evaporation, root
extraction and mulch evaporation of every day come from the dump, and so do the day's soil
properties (``SOILDYN``).

Two comparisons, both batched over all runs (one ``vmap``):

* **one day** (every day of every run a sample): from the reference's ``WATBAL`` RATE entry state
  (water content, snow, mulch water) one RATE, replay and INTEGR, against the day's exits;
* **free run**: each run's whole season from its initial state, against the exits of every day.

Tolerances come from the reference's own arithmetic. ``WATBAL`` rounds the water content to
``1e-6`` in REAL (``ANINT(SW*1e6)/1e6``) and computes in REAL*4 (unit roundoff
``u = 2^-24``); our kernels run the same statements in float64. So a day's water content may land
one quantum ``q = 1e-6`` apart, plus the REAL representation of the quantum (``u |SW|``); every
other quantity is within ``16 u S`` of the reference, ``S`` the scale of the numbers it is computed
from (the profile's saturated water ``sum(SAT DLAYR)`` plus the day's water input; 16 REAL
operations per result). In the free run the differences of the water content accumulate at most
one quantum per day, and a flux follows its inputs with a Lipschitz constant of at most 2 (cm of
flux per cm of water-content difference): ``|d flux| <= 16 u S + 2 sum(DLAYR |d SW_entry|)``.
The measured numbers go to ``<data-dir>/validation/aj_dsw/d1a/bucket_report.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bucket_dssat as bd
import jax
import numpy as np
import pytest

from agrijax.core.runtime import run_batch

pytestmark = [
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
    pytest.mark.allow_skip(reason="needs the DSSAT WATBAL dumps in the data directory"),
]

U32 = 2.0**-24
QUANTUM = 1e-6
K_OPS = 16
LIPSCHITZ = 2.0
REPORT = Path("validation/aj_dsw/d1a/bucket_report.json")
#: quantities of :func:`bd.outputs` compared, and the unit of their scale (mm or cm per layer)
MM = ("runoff", "infiltration", "drain", "watavl", "snow", "mulch_wat")
CM_LAYER = ("drn", "upflow")
VOL = ("swdelts", "swdeltu")


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tables(data_dir: Path) -> Path:
    t = data_dir / bd.TABLES
    if not (t.parent / "collect_report.json").is_file():
        pytest.skip(f"no WATBAL dumps under {t}")
    return t


@pytest.fixture(scope="module")
def runs(tables: Path) -> list[bd.Run]:
    return [bd.load_run(tables, k) for k in bd.run_keys(tables)]


@pytest.fixture(scope="module")
def n_layer(runs: list[bd.Run]) -> int:
    return max(r.nl for r in runs)


@pytest.fixture(scope="module")
def free(runs: list[bd.Run], n_layer: int) -> dict[str, np.ndarray]:
    p, f, s = bd.inputs(runs, n_layer)
    res = run_batch(bd.replay_model(), p, f, s, in_axes=(0, 0, 0))
    return {k: np.asarray(v) for k, v in res.items()}


@pytest.fixture(scope="module")
def one_day(runs: list[bd.Run], n_layer: int) -> tuple[dict[str, np.ndarray], list[tuple[int, int]]]:
    p, f, s, idx = bd.day_inputs(runs, n_layer)
    res = run_batch(bd.replay_model(), p, f, s, in_axes=(0, 0, 0))
    return {k: np.asarray(v)[:, 0] for k, v in res.items()}, idx


@pytest.fixture(scope="module")
def report(data_dir: Path):
    rep: dict[str, Any] = {}
    yield rep
    out = data_dir / REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text()) if out.is_file() else {}
    old.update(rep)
    out.write_text(json.dumps(old, indent=1))


def _salus(r: bd.Run) -> bool:
    return bd._char(r.v("wb_rate_in", "MESEV")[0]) == "S"


def _scale_cm(r: bd.Run, n_layer: int) -> np.ndarray:
    """Per day: the profile's saturated water plus the day's water input [cm]."""
    sat = bd._layers(r, "wb_rate_in", "SP_SAT", r.n_days, n_layer)
    dl = bd._layers(r, "wb_rate_in", "SP_DLAYR", r.n_days, n_layer)
    inp = (
        np.asarray(r.v("wb_rate_in", "RAIN"), float) + np.asarray(r.v("wb_rate_in", "IRRAMT"), float)
    ) * 0.1
    return np.sum(sat * dl, axis=1) + inp + np.asarray(r.v("wb_rate_in", "SNOW"), float) * 0.1


def _check(sim: dict[str, np.ndarray], r: bd.Run, n_layer: int, dsw_entry: np.ndarray | None):
    """Per quantity: max |sim - ref|, the tolerance margin (max of |d| / tol) and the day."""
    ref = bd.reference(r, n_layer)
    nl = r.nl
    s_cm = _scale_cm(r, n_layer)
    dl = bd._layers(r, "wb_rate_in", "SP_DLAYR", r.n_days, n_layer)
    # the propagated entry difference of the free run [cm]
    prop = np.zeros(r.n_days) if dsw_entry is None else LIPSCHITZ * np.sum(dl * np.abs(dsw_entry), axis=1)
    out: dict[str, dict[str, float]] = {}
    for q in ("sw", *MM, *CM_LAYER, *VOL):
        if q == "upflow" and _salus(r):
            continue  # with MESEV = 'S' UPFLOW belongs to SPAM's ESR_SoilEvap (not called in WATBAL)
        d = np.abs(sim[q] - ref[q])
        if q == "sw":
            if dsw_entry is None:
                tol = QUANTUM + U32 * np.abs(ref[q])
            else:  # accumulated quanta: at most one new quantum per day
                tol = (np.arange(r.n_days)[:, None] + 1) * (QUANTUM + U32 * np.abs(ref[q]))
            d, tol = d[:, :nl], tol[:, :nl]
        elif q in MM:
            tol = K_OPS * U32 * s_cm * 10.0 + prop * 10.0
        elif q in CM_LAYER:
            tol = (K_OPS * U32 * s_cm + prop)[:, None] * np.ones((1, n_layer))
            d, tol = d[:, :nl], tol[:, :nl]
        else:  # volumetric changes, as depths per layer
            d = (d * dl)[:, :nl]
            tol = ((K_OPS * U32 * s_cm + prop)[:, None] * np.ones((1, n_layer)))[:, :nl]
        ratio = d / tol
        k = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        out[q] = {"max": float(d.max()), "margin": float(ratio.max()), "date": int(r.date[k[0]])}
    return out


def _summary(per_run: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, Any]]:
    qs = sorted({q for v in per_run.values() for q in v})
    out = {}
    for q in qs:
        rows = [(v[q]["margin"], v[q]["max"], k, v[q]["date"]) for k, v in per_run.items() if q in v]
        m = max(rows)
        out[q] = {"max_abs": max(r[1] for r in rows), "max_margin": m[0], "worst": f"{m[2]}@{m[3]}"}
    return out


# ----------------------------------------------------------------------------- the reference set
def test_collection_is_complete_and_every_output_identical(tables: Path) -> None:
    rep = json.loads((tables.parent / "collect_report.json").read_text())
    assert rep["n_treatments"] == 58 and rep["n_identical"] == 58
    assert rep["catpa"]["out_diffs"] == [] and len(rep["catpa"]["runs"]) == 14
    extra = json.loads((tables.parent / "collect_report_extra.json").read_text())
    assert {k: v["out_diffs"] for k, v in extra["wheat"].items()} == {"KSAS8101": [], "SWSW7501": []}
    keys = bd.run_keys(tables)
    assert len(keys) == 58 + 14 + 6 + 14, len(keys)


def test_reference_set_stays_in_the_supported_scope(runs: list[bd.Run], report: dict) -> None:
    """No flooding, tile drainage, water table, tillage, plastic mulch or zonal energy balance
    (``MEEVP = 'Z'``, whose WATBAL INTEGR branch is not ported) in any run; the replayed
    SOILDYN properties change in some runs, snow falls in the wheat runs, both evaporation methods
    occur."""
    n_salus = n_soil_change = n_snow_days = 0
    for r in runs:
        ri, ii = r.t["wb_rate_in"], r.t["wb_integr_in"]
        assert np.all(ri["FLOOD"] == 0.0) and not np.any(ri["PUDDLED"]) and not np.any(ri["BUNDED"]), r.key
        assert np.all(ri["NTIL"] == 0) and np.all(ri["TDLNO"] <= 0), r.key
        for n in ("SWDELTL", "SWDELTT", "SWDELTW"):
            assert np.all(ii[n] == 0.0), (r.key, n)
        assert np.all(ri["ACTWTD"] >= ri["SP_DS"][:, r.nl - 1]), r.key
        assert np.all(r.t["rnoff_in"]["PMFRACTION"] == 0.0), r.key
        assert bd._char(ri["MEINF"][0]) in ("R", "S", "M"), r.key
        # WATBAL INTEGR's MEEVP = 'Z' branch (the zonal ETPHOT soil evaporation) is not ported
        meevp = {bd._char(x) for x in np.asarray(ii["MEEVP"]).reshape(-1)}
        assert "Z" not in meevp, (r.key, meevp)
        n_salus += _salus(r)
        n_soil_change += bool(
            np.any(ri["SP_DUL"] != ri["SP_DUL"][0]) or np.any(ii["SP_DLAYR"] != ri["SP_DLAYR"])
        )
        n_snow_days += int(np.sum(r.t["wb_rate_out"]["SNOW"] > 0.0))
    report["scope"] = {
        "runs": len(runs),
        "salus": n_salus,
        "soil_changes": n_soil_change,
        "snow_days": n_snow_days,
    }
    assert n_salus > 0 and n_soil_change > 0 and n_snow_days > 0


# ----------------------------------------------------------------------------- one day
def test_one_day_steps_match_the_dumps(runs, n_layer, one_day, report) -> None:
    res, idx = one_day
    per_run = {}
    flips = 0
    for i, r in enumerate(runs):
        rows = [k for k, (ri, _) in enumerate(idx) if ri == i]
        sim = {q: v[rows] for q, v in res.items()}
        per_run[r.key] = _check(sim, r, n_layer, None)
        ref = bd.reference(r, n_layer)["sw"][:, : r.nl]
        flips += int(np.sum(np.abs(sim["sw"][:, : r.nl] - ref) > 0.5 * QUANTUM))
    summ = _summary(per_run)
    report["one_day"] = {
        "summary": summ,
        "sw_quantum_flips": flips,
        "layer_days": sum(r.n_days * r.nl for r in runs),
    }
    bad = {q: v for q, v in summ.items() if v["max_margin"] > 1.0}
    assert not bad, bad


def test_one_day_truncation_equals_the_reference_residual(runs, n_layer, one_day, report) -> None:
    """Our ``truncation`` (the water the reference's rounding and cuts create or remove) equals the
    reference's own daily balance residual: its storage change minus (inputs - outputs)."""
    res, idx = one_day
    worst = 0.0
    for i, r in enumerate(runs):
        rows = [k for k, (ri, _) in enumerate(idx) if ri == i]
        nd, nl = r.n_days, n_layer
        ri, ro, ii, io = "wb_rate_in", "wb_rate_out", "wb_integr_in", "wb_integr_out"
        dy = bd._layers(r, ii, "DLAYR_YEST", nd, nl)
        dl = bd._layers(r, ii, "SP_DLAYR", nd, nl)
        s0 = np.sum(bd._layers(r, ri, "SW", nd, nl) * dy, axis=1)
        s1 = np.sum(bd._layers(r, io, "SW", nd, nl) * dl, axis=1)
        m0 = np.asarray(r.v("mulch_rate_in", "MULCHWAT"), float) * 0.1
        m1 = np.asarray(r.v("mulch_integr_out", "MULCHWAT"), float) * 0.1
        sn0 = np.asarray(r.v(ri, "SNOW"), float) * 0.1
        sn1 = np.asarray(r.v(ro, "SNOW"), float) * 0.1
        inflow = (np.asarray(r.v(ri, "RAIN"), float) + np.asarray(r.v(ri, "IRRAMT"), float)) * 0.1
        inflow = inflow + np.asarray(r.v("mulch_rate_out", "RESWATADD"), float) * 0.1
        up = -np.sum(bd._layers(r, ii, "SWDELTX", nd, nl) * dy, axis=1)
        if _salus(r):
            ev = -np.sum(bd._layers(r, ii, "SWDELTU", nd, nl) * dy, axis=1)
        else:
            ev = np.asarray(r.v(ii, "ES"), float) * 0.1
        em = np.asarray(r.v("mulch_integr_in", "MULCHEVAP"), float) * 0.1
        out = (
            (np.asarray(r.v(ro, "RUNOFF"), float) + np.asarray(r.v(ro, "DRAIN"), float)) * 0.1 + up + ev + em
        )
        ref_trunc = inflow - out - ((s1 + m1 + sn1) - (s0 + m0 + sn0))
        d = np.abs(res["truncation"][rows] - ref_trunc)
        # a quantum flip of the water content moves the rounding term by the same water depth
        flip = np.sum(np.abs(res["sw"][rows] - bd._layers(r, io, "SW", nd, nl)) * dl, axis=1)
        tol = K_OPS * U32 * _scale_cm(r, n_layer) + flip
        worst = max(worst, float(np.max(d / tol)))
    report["one_day_truncation_margin"] = worst
    assert worst <= 1.0


# ----------------------------------------------------------------------------- free run
def test_free_run_matches_the_dumps_every_day(runs, n_layer, free, report) -> None:
    per_run = {}
    for i, r in enumerate(runs):
        sim = {q: v[i, : r.n_days] for q, v in free.items()}
        ref_sw = bd.reference(r, n_layer)["sw"]
        entry = np.zeros_like(ref_sw)
        entry[1:] = sim["sw"][:-1] - ref_sw[:-1]
        per_run[r.key] = _check(sim, r, n_layer, entry)
    summ = _summary(per_run)
    report["free_run"] = {"summary": summ, "days": sum(r.n_days for r in runs)}
    bad = {q: v for q, v in summ.items() if v["max_margin"] > 1.0}
    assert not bad, bad


def test_free_run_closes_its_own_water_balance(runs, n_layer, free, report) -> None:
    """Every day: storage change = rain + irrigation + residue water - runoff - drainage - uptake -
    evaporation - mulch evaporation - truncation, to float64 rounding."""
    _, f, _ = bd.inputs(runs, n_layer)
    worst = 0.0
    for i, r in enumerate(runs):
        nd = r.n_days
        st = free["storage"][i, :nd]
        s_init = float(
            np.sum(
                bd._layers(r, "wb_rate_in", "SW", 1, n_layer)[0]
                * bd._layers(r, "wb_integr_in", "DLAYR_YEST", 1, n_layer)[0]
            )
        )
        s_init += (float(r.v("wb_rate_in", "SNOW")[0]) + float(r.v("mulch_rate_in", "MULCHWAT")[0])) * 0.1
        prev = np.concatenate([[s_init], st[:-1]])
        fb = jax.tree_util.tree_map(lambda x, i=i, nd=nd: np.asarray(x)[i, :nd], f)
        salus = _salus(r)
        inflow = (fb.bucket.rain + fb.bucket.irrigation + free["residue_water"][i, :nd]) * 0.1
        evap = np.sum(fb.evap_layers, axis=1) if salus else fb.es
        out = (
            (free["runoff"][i, :nd] + free["drain"][i, :nd]) * 0.1 + np.sum(fb.uptake, axis=1) + evap + fb.em
        )
        out = out + free["truncation"][i, :nd]
        res = (st - prev) - (inflow - out)
        worst = max(worst, float(np.max(np.abs(res))))
    report["free_run_balance_max_abs_cm"] = worst
    assert worst <= 1e-10


# ----------------------------------------------------------------------------- the callees
def _callee(runs: list[bd.Run], name: str, fields: tuple[str, ...], n_layer: int):
    """Stack the entry and exit values of a callee over all calls of all runs (layers padded)."""
    ins: dict[str, list[np.ndarray]] = {k: [] for k in fields}
    outs: dict[str, list[np.ndarray]] = {k: [] for k in fields}
    nl = []
    for r in runs:
        tab = r.t.get(f"{name}_in", {})
        if not tab or not len(next(iter(tab.values()))):
            continue
        a, b = r.t[f"{name}_in"], r.t[f"{name}_out"]
        n = len(next(iter(a.values())))
        for k in fields:
            for src, dst in ((a, ins), (b, outs)):
                v = np.asarray(src[k], float)
                if v.ndim == 2:
                    w = np.zeros((n, n_layer))
                    w[:, : r.nl] = v[:, : r.nl]
                    v = w
                dst[k].append(v)
        nl += [r.nl] * n
    return (
        {k: np.concatenate(v) for k, v in ins.items()},
        {k: np.concatenate(v) for k, v in outs.items()},
        np.asarray(nl),
    )


def dumps_dates(r: bd.Run, table: str) -> np.ndarray:
    """The dates of one of the run's callee tables."""
    return r.dates[table]


def _mask(nl: np.ndarray, n: int) -> np.ndarray:
    return np.arange(n)[None, :] < nl[:, None]


def test_infil_satflo_upflow_rnoff_calls_match_their_dumps(runs, n_layer, report) -> None:
    from agrijax.processes.soil_water.bucket import kernels as K

    margins = {}
    f = (
        "DLAYR",
        "DS",
        "DUL",
        "SAT",
        "SW",
        "SWCN",
        "SWCON",
        "PINF",
        "ACTWTD",
        "SWDELTS",
        "DRN",
        "DRAIN",
        "EXCS",
    )
    a, b, nl = _callee(runs, "infil", f, n_layer)
    m = _mask(nl, n_layer)
    dl = np.where(m, a["DLAYR"], 0.0)
    got = jax.jit(jax.vmap(K.infil))(
        dl, a["DS"], a["DUL"], a["SAT"], a["SW"], a["SWCN"], a["SWCON"], a["PINF"], a["ACTWTD"]
    )
    scale = np.sum(a["SAT"] * dl, axis=1) + a["PINF"]
    tol = K_OPS * U32 * scale
    margins["infil"] = max(
        float(np.max(np.abs(np.asarray(got.swdelts) - b["SWDELTS"]) * dl / tol[:, None])),
        float(np.max(np.where(m, np.abs(np.asarray(got.drn) - b["DRN"]), 0.0) / tol[:, None])),
        float(np.max(np.abs(np.asarray(got.drain) - b["DRAIN"]) / (10 * tol))),
        float(np.max(np.abs(np.asarray(got.excs) - b["EXCS"]) / tol)),
    )
    f = ("DLAYR", "DUL", "SAT", "SW", "SWCN", "SWCON", "SWDELTS", "DRN", "DRAIN")
    a, b, nl = _callee(runs, "satflo", f, n_layer)
    m = _mask(nl, n_layer)
    dl = np.where(m, a["DLAYR"], 0.0)
    got = jax.jit(jax.vmap(K.satflo))(dl, a["DUL"], a["SAT"], a["SW"], a["SWCN"], a["SWCON"])
    tol = K_OPS * U32 * np.sum(a["SAT"] * dl, axis=1)
    margins["satflo"] = max(
        float(np.max(np.abs(np.asarray(got.swdelts) - b["SWDELTS"]) * dl / tol[:, None])),
        float(np.max(np.where(m, np.abs(np.asarray(got.drn) - b["DRN"]), 0.0) / tol[:, None])),
        float(np.max(np.abs(np.asarray(got.drain) - b["DRAIN"]) / (10 * tol))),
    )
    f = ("DLAYR", "DUL", "LL", "SAT", "SW", "SW_AVAIL", "SWDELTU", "UPFLOW")
    a, b, nl = _callee(runs, "up_flow", f, n_layer)
    m = _mask(nl, n_layer)
    dl = np.where(m, a["DLAYR"], 0.0)
    got = jax.jit(jax.vmap(K.up_flow))(dl, a["DUL"], a["LL"], a["SAT"], a["SW"], a["SW_AVAIL"])
    tol = K_OPS * U32 * np.sum(a["SAT"] * dl, axis=1)
    margins["up_flow"] = max(
        float(np.max(np.abs(np.asarray(got.swdeltu) - b["SWDELTU"]) * dl / tol[:, None])),
        float(np.max(np.where(m, np.abs(np.asarray(got.upflow) - b["UPFLOW"]), 0.0) / tol[:, None])),
    )
    f = ("CN", "LL", "SAT", "SW", "WATAVL", "MULCH%MULCHCOVER", "RUNOFF")
    a, b, nl = _callee(runs, "rnoff", f, n_layer)
    meinf = np.ones_like(
        a["CN"]
    )  # MEINF = 'S' in every run (test_reference_set_stays_in_the_supported_scope)
    got = jax.jit(jax.vmap(lambda *x: K.rnoff(*x).runoff))(
        a["CN"], a["LL"], a["SAT"], a["SW"], a["WATAVL"], a["MULCH%MULCHCOVER"], meinf
    )
    tol = K_OPS * U32 * (a["WATAVL"] + 254.0 * (100.0 / a["CN"]))
    margins["rnoff"] = float(np.max(np.abs(np.asarray(got) - b["RUNOFF"]) / tol))
    report["callee_margins"] = margins
    assert all(v <= 1.0 for v in margins.values()), margins


def test_snowfall_and_mulch_calls_match_their_dumps(runs, n_layer, report) -> None:
    from agrijax.processes.soil_water.bucket import kernels as K

    a, b, _ = _callee(runs, "snow", ("TMAX", "RAIN", "SNOW", "WATAVL"), n_layer)
    assert a["TMAX"].size > 0
    got = jax.jit(jax.vmap(K.snowfall))(a["TMAX"], a["RAIN"], a["SNOW"])
    n_snow = int(np.sum((a["TMAX"] <= 1.0) | (a["SNOW"] > 0.0)))
    # scales in mm, floored at 1 mm (a day without snow, rain or residue has nothing to round)
    tol = K_OPS * U32 * (a["SNOW"] + a["RAIN"] + np.abs(a["TMAX"]) + 1.0)
    m_snow = max(
        float(np.max(np.abs(np.asarray(got.snow) - b["SNOW"]) / tol)),
        float(np.max(np.abs(np.asarray(got.watavl) - b["WATAVL"]) / tol)),
    )
    # MULCHWATER's DYNAMIC is a local copied from CONTROL after the entry dump, so the table selected
    # with DYNAMIC = RATE at entry ("mulch_rate_in") holds the entries of the INTEGR call, whose saved
    # locals are the RATE call's inputs and results (MULCHEVAP, MULCHMASS, MULCHCOVER, WATFAC; MULWATADD,
    # NRAIN) and whose MULCHWAT is unchanged by RATE. The RATE input WATAVL is the water after the snow
    # stage: RAIN, or SNOWFALL's WATAVL on the days it runs.
    cols: dict[str, list[np.ndarray]] = {
        k: [] for k in ("w", "mw", "ev", "ms", "cv", "wf", "add", "nrain", "new")
    }
    for r in runs:
        t = r.t["mulch_rate_in"]
        w = np.asarray(r.v("wb_rate_in", "RAIN"), float).copy()
        if r.t.get("snow_out"):
            k = np.searchsorted(r.date, dumps_dates(r, "snow_out"))
            w[k] = np.asarray(r.t["snow_out"]["WATAVL"], float)
        for key, v in zip(cols, (w, t["MULCHWAT"], t["MULCHEVAP"], t["MULCHMASS"], t["MULCHCOVER"], t["WATFAC"],
                                 t["MULWATADD"], t["NRAIN"], r.v("wb_rate_in", "M_NEW")), strict=True):  # fmt: skip
            cols[key].append(np.asarray(v, float))
    c = {k: np.concatenate(v) for k, v in cols.items()}
    one = np.ones_like(c["w"])
    got = jax.jit(jax.vmap(K.mulch_rate))(c["w"], c["mw"], c["ev"], c["ms"], c["cv"], c["new"], c["wf"], one)
    wet = (c["ms"] > 0.01) & (c["w"] > 0.0)
    tol = K_OPS * U32 * (c["w"] + c["wf"] * 1e-4 * c["ms"] + c["mw"] + 1.0)
    m_mulch = max(
        float(np.max(np.abs(np.asarray(got.mulwatadd) - np.where(wet, c["add"], 0.0)) / tol)),
        float(np.max(np.where(wet, np.abs(np.asarray(got.watavl) - c["nrain"]), 0.0) / tol)),
    )
    a = {"WATAVL": c["w"]}
    report["callee_margins_surface"] = {
        "snowfall": m_snow,
        "mulch_rate": m_mulch,
        "snowfall_active_calls": n_snow,
        "mulch_calls": int(a["WATAVL"].size),
    }
    assert n_snow > 0 and m_snow <= 1.0 and m_mulch <= 1.0
