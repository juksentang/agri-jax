"""RZWQM2 4.6 daily full-state dumps (plan A12): conventions the M3 bindings rely on.

The instrumented build dumps, on every day of a CA-TPA run, the entry and exit of ``PHYSCL`` (soil
physics of the day), ``MAPLNT`` (plant), ``DSSATDRV`` (the embedded DSSAT crop driver), ``ROOTWU``
and ``MZ_GROSUB`` (embedded CERES-Maize), every storm event (``EVNTRO``/``INFIL``) and every step
of the redistribution clock (``ADJDT``). Records carry a run-wide sequence number, so the order
of the routines within a day is part of the data. The daily tables are in
``dumps/tables/rzwqm46_<run>/`` (see :mod:`agrijax.port.dumps`); the run's outputs and the
instrumentation report in ``dumps/_runs/<run>/``.

Every test checks a fact of the reference model against the reference model itself (its own
outputs, or one dumped quantity against another), independently of any Agri-JAX process. Each
fact answers an open verification item of the extensibility plan (O-RZ1 ... O-RZ6) and fails if a
new RZWQM2 build changes it.
"""

from __future__ import annotations

import functools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from agrijax.io.rzwqm.ana import KEY_COLUMNS, read_ana
from agrijax.port import dumps

pytestmark = pytest.mark.allow_skip(reason="dumps are private data")

RUNS = ("catpa2015", "catpa2015_2023")


@pytest.fixture(params=RUNS)
def run(request: pytest.FixtureRequest, dumps_dir: Path) -> Path:
    d = dumps_dir / "tables" / f"rzwqm46_{request.param}"
    if not (d / "physcl_exit.npz").is_file():
        pytest.skip(f"no A12 tables at {d}")
    return d


@functools.lru_cache(maxsize=32)
def _t(run: Path, name: str) -> dumps.DailyTable:
    return dumps.load_table(run / f"{name}.npz")[0]


def _day_sequences(run: Path) -> dict[int, list[tuple[str, int]]]:
    order = np.load(run / "record_order.npz")["order"]
    per: dict[int, list[tuple[str, int]]] = {}
    for r in order:
        per.setdefault(int(r["date"]), []).append((str(r["routine"]), int(r["phase"])))
    return per


def _overlap(src_bot: np.ndarray, dst_bot: np.ndarray) -> np.ndarray:
    """``O[i, j]`` = thickness of target cell ``i`` inside source cell ``j`` (cells from depth 0)."""
    s_top = np.concatenate([[0.0], src_bot[:-1]])
    d_top = np.concatenate([[0.0], dst_bot[:-1]])
    lo = np.maximum(d_top[:, None], s_top[None, :])
    hi = np.minimum(dst_bot[:, None], src_bot[None, :])
    return np.clip(hi - lo, 0.0, None)


def test_instrumented_build_leaves_outputs_unchanged(run: Path) -> None:
    rep = json.loads((run / "report.json").read_text())
    assert rep["ana_identical"] and rep["layer_plt_identical"], rep
    assert rep["overview_identical"], "OVERVIEW.OUT differs beyond time stamps and run-dir paths"


def test_daily_order_crop_after_physics(run: Path) -> None:
    """O-RZ2: PHYSCL opens the day, storm events run inside it, the crop (MAPLNT -> DSSATDRV ->
    ROOTWU, MZ_GROSUB) runs after PHYSCL returns; at most one storm event per day."""
    per = _day_sequences(run)
    n_crop = 0
    for d, seq in per.items():
        assert seq[0] == ("PHYSCL", 0), d
        ph_exit = seq.index(("PHYSCL", 1))
        ev = [i for i, x in enumerate(seq) if x[0] in ("EVNTRO", "INFIL")]
        assert all(0 < i < ph_exit for i in ev), d
        assert sum(1 for x in seq if x == ("EVNTRO", 0)) <= 1, d
        ms = [i for i, x in enumerate(seq) if x[0] == "MAPLNT"]
        assert ms and ms[0] > ph_exit, d
        crop = [i for i, x in enumerate(seq) if x[0] in ("DSSATDRV", "ROOTWU", "MZ_GROSUB")]
        assert all(ms[0] < i < ms[-1] for i in crop), d
        names = [x[0] for x in seq if x[1] == 0]
        if "ROOTWU" in names:
            n_crop += 1
            # O-RZ3: potential uptake is computed before the crop grows (except on the first crop
            # day, when MZ_GROSUB is initialised first)
            first_crop_day = d == min(
                k for k, s in per.items() if ("ROOTWU", 0) in s and d // 1000 == k // 1000
            )
            if not first_crop_day:
                assert names.index("ROOTWU") < names.index("MZ_GROSUB"), d
    assert n_crop >= 100


def test_storm_event_is_instantaneous_on_the_redistribution_clock(run: Path) -> None:
    """O-RZ2 / O-RZ6: the redistribution steps of every day add up to 24 h whatever the storm
    duration, and the step after an event starts where the step before it ended."""
    a = np.load(run / "adjdt_calls.npz")
    ex = a["phase"] == 1
    tot: dict[int, float] = {}
    for d, dt in zip(a["date"][ex], a["delt"][ex], strict=True):
        tot[int(d)] = tot.get(int(d), 0.0) + float(dt)
    assert max(abs(v - 24.0) for v in tot.values()) < 1e-9
    order = np.load(run / "record_order.npz")["order"]
    ev = order[(order["routine"] == "EVNTRO") & (order["phase"] == 0)]
    ph = _t(run, "physcl_exit")
    assert len(ev) > 50
    dur = []
    for e in ev:
        same = a["date"] == e["date"]
        before = np.nonzero(same & ex & (a["seq"] < e["seq"]))[0]
        after = np.nonzero(same & (a["phase"] == 0) & (a["seq"] > e["seq"]))[0]
        t_end = float(a["daytim"][before[-1]] + a["delt"][before[-1]]) if before.size else 0.0
        assert after.size, int(e["date"])
        assert float(a["daytim"][after[0]]) == t_end, int(e["date"])
        dur.append(float(ph.values["TMSTM"][int(ph.index_of([int(e["date"])])[0])]))
    assert max(dur) > 0.0  # storms have a duration (up to 2 h at CA-TPA) that the clock does not spend
    np.testing.assert_array_equal(ph.values["DAYTIM"], 24.0)


def test_istress0_sink_is_yesterdays_dssat_uptake(run: Path) -> None:
    """O-RZ2: with ISTRESS = 0 the Richards sink is ``qsr * 24 * |ptrans / petplant|`` from the
    ``qsr`` that DSSATDRV wrote at the end of the previous day (lag 1)."""
    ph_in, ph_out, dd_out = _t(run, "physcl_entry"), _t(run, "physcl_exit"), _t(run, "dssatdrv_exit")
    for k in ("ISTRESS", "IHOURLY", "ISHAW"):
        assert set(ph_out.values[k].tolist()) == {0}, k
    nn = int(ph_out.values["NN"][0])
    pp, pt = ph_out.values["PETPLANT"], ph_out.values["PTRANS"]
    act = np.abs(pp) > 0
    pred = ph_out.values["QSR"][act, :nn].astype(np.float64) * 24.0 * np.abs(pt[act] / pp[act])[:, None]
    assert act.sum() > 100
    qs = ph_out.values["QSX"][act]
    # RICHRD sets the sink of a node to 0 in place when its Newton update is clamped at HMIN;
    # those nodes are at the dry limit. Everywhere else the formula holds exactly.
    off = qs != pred
    h = ph_out.values["H"][act, :nn]
    hmin = np.broadcast_to(ph_out.values["HMIN"][act][:, None], h.shape)
    assert np.all(qs[off] == 0.0) and np.all(h[off] < hmin[off] + 50.0)
    assert off.mean() < 0.02
    rows = [(k, int(j)) for k, j in enumerate(ph_in.index_of(dd_out.date + 1)) if j >= 0]
    assert len(rows) > 100
    for k, j in rows:
        np.testing.assert_array_equal(ph_in.values["QSR"][j], dd_out.values["QSR"][k])


def test_physcl_limits_yesterdays_uptake_by_todays_pet(run: Path) -> None:
    """A4 chain, step missing from plan rev 2: before the time loop PHYSCL scales the node ``qsr``
    by ``WUF = min(1, PET / TRWUP)`` (today's PET, yesterday's TRWUP; REAL*4 product) on nodes
    with ``THETA > SOILHP(9)`` and zeroes nodes with ``THETA < SOILHP(9)``, when PET > 0 and
    TRWUP != 0; the sink uses the scaled ``qsr``."""
    ph_in, ph_out = _t(run, "physcl_entry"), _t(run, "physcl_exit")
    nn = int(ph_out.values["NN"][0])
    pet = ph_out.values["PETPLANT"]
    tr = ph_in.values["TRWUP"].astype(np.float64)
    on = (pet > 0) & (tr != 0)
    wuf = np.where(on, np.where(pet <= tr, pet / np.where(tr != 0, tr, 1.0), 1.0), 1.0).astype(np.float32)
    jh = ph_in.values["NDXN2H"][:, :nn] - 1
    th9 = np.take_along_axis(ph_in.values["SOILHP"][:, 8, :], jh, axis=1)
    th = ph_in.values["THETA"][:, :nn]
    q0 = ph_in.values["QSR"][:, :nn]
    q = np.where(on[:, None] & (th > th9), q0 * wuf[:, None], np.where(on[:, None] & (th < th9), 0.0, q0))
    np.testing.assert_array_equal(ph_out.values["QSR"][:, :nn], q.astype(np.float32))
    assert (wuf < 1).sum() > 100  # the limiter is active on most crop days at CA-TPA


def test_realmatch_maps_nodes_to_crop_layers_and_back(run: Path) -> None:
    """O-RZ2 / A7: the crop layers are LYRSET (5, 15, 30, 45, 60, +30 cm to the profile); SW at
    ROOTWU entry is the thickness-weighted mean of the end-of-physics node THETA, and the node
    ``qsr`` is the thickness-weighted mean of the layer rates ``rwu / 24 / dlayr`` (``SW > LL``)."""
    ph_out, dd_out, rw_in = _t(run, "physcl_exit"), _t(run, "dssatdrv_exit"), _t(run, "rootwu_entry")
    nn, nl = int(ph_out.values["NN"][0]), int(rw_in.values["NLAYR"][0])
    ds = dd_out.values["SOILPROP%DS"][0, :nl].astype(np.float64)
    tlt = ph_out.values["TLT"][0, :nn].astype(np.float64)
    assert ds.tolist() == [5.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0][:nl] and ds[-1] == tlt[-1]
    o = _overlap(tlt, ds)
    w = o / o.sum(axis=1, keepdims=True)
    j = ph_out.index_of(rw_in.date)
    sw_pred = ph_out.values["THETA"][j, :nn] @ w.T
    np.testing.assert_allclose(rw_in.values["SW"][:, :nl], sw_pred, rtol=0, atol=2e-7)  # REAL*4 of the mean
    ot = _overlap(ds, tlt)
    wt = ot / ot.sum(axis=1, keepdims=True)
    v = dd_out.values
    sw, ll = v["SW"][:, :nl], v["SOILPROP%LL"][:, :nl]
    ql = np.where(sw > ll, v["RWU"][:, :nl] / 24.0 / v["SOILPROP%DLAYR"][:, :nl], 0.0).astype(np.float32)
    # quirk: a layer with SW == LL exactly is left alone, so it keeps element L of the array as
    # DSSATDRV found it -- the *node* L value of yesterday's qsr after today's PHYSCL limiter
    dd_in = _t(run, "dssatdrv_entry")
    assert np.array_equal(dd_in.call, dd_out.call)
    ql = np.where(sw == ll, dd_in.values["QSR"][:, :nl], ql)
    np.testing.assert_allclose(v["QSR"][:, :nn], ql.astype(np.float64) @ wt.T, rtol=1e-6, atol=1e-12)


def test_rootwu_uses_yesterdays_root_length_density(run: Path) -> None:
    """O-RZ3: RLV at ROOTWU entry is the crop's RLV of the previous day, not today's."""
    rw_in, dd_out = _t(run, "rootwu_entry"), _t(run, "dssatdrv_exit")
    nl = int(rw_in.values["NLAYR"][0])
    j = dd_out.index_of(rw_in.date - 1)
    have = j >= 0
    assert have.sum() > 100
    np.testing.assert_array_equal(rw_in.values["RLV"][have, :nl], dd_out.values["RLV"][j[have], :nl])
    same = dd_out.index_of(rw_in.date)
    assert np.abs(rw_in.values["RLV"][:, :nl] - dd_out.values["RLV"][same, :nl]).max() > 0.01


def test_profile_storage_matches_ana(run: Path, dumps_dir: Path) -> None:
    """PHYSCL exit sum(THETA * TL) is the ``.ana`` profile storage (6 significant digits)."""
    ph = _t(run, "physcl_exit")
    ana = dumps_dir / "_runs" / run.name.removeprefix("rzwqm46_") / "CA-TPA.ana"
    if not ana.is_file():
        pytest.skip(f"no {ana}")
    ds = read_ana(ana)
    col = ds.attrs["columns"][str(KEY_COLUMNS["profile_water_cm"])]
    days = np.array([round(float(t) * 1000) for t in ds["yyyyddd"].values])
    ref = np.asarray(ds[col].values, dtype=np.float64)[np.searchsorted(days, ph.date)]
    nn = int(ph.values["NN"][0])
    st = np.sum(ph.values["THETA"][:, :nn] * ph.values["TL"][:, :nn], axis=1)
    half_unit = 0.5 * 10.0 ** (np.floor(np.log10(np.abs(ref))) - 5) * (1 + 1e-9)
    assert np.all(np.abs(st - ref) <= half_unit)


def test_morning_state_is_the_previous_evening_except_on_soil_changes(run: Path) -> None:
    """For resynchronisation the morning state is the PHYSCL *entry* table: it equals the previous
    day's exit to round-off (HYDPAR re-evaluates theta(h)) except on days when the soil hydraulic
    properties change (tillage), when theta and h jump."""
    ph_in, ph_out = _t(run, "physcl_entry"), _t(run, "physcl_exit")
    nn = int(ph_out.values["NN"][0])
    j = ph_out.index_of(ph_in.date - 1)
    j[ph_in.date % 1000 == 1] = -1  # year boundary
    k = np.nonzero(j >= 0)[0]
    dth = np.abs(ph_in.values["THETA"][k, :nn] - ph_out.values["THETA"][j[k], :nn]).max(axis=1)
    dsoil = np.abs(ph_in.values["SOILHP"][k] - ph_out.values["SOILHP"][j[k]]).reshape(len(k), -1).max(axis=1)
    assert np.all(dth[dsoil == 0] < 1e-15)
    assert np.all(dth[dsoil > 0] > 1e-6)


def test_embedded_ceres_is_the_dssat40_lineage(run: Path, dumps_dir: Path) -> None:
    """O-RZ1: ``CARBO = PCARB * min(PRFT, SWFAC, NSTRES) * SLPF`` holds exactly on every day
    (the DSSAT 4.0 form; 4.8.6 also multiplies PSTRES1 and KSTRES), and OVERVIEW.OUT names
    DSSAT-CSM 4.0.2."""
    v = _t(run, "mz_grosub_exit").values
    car = v["CARBO"]
    f = (v["PCARB"] * np.minimum(np.minimum(v["PRFT"], v["SWFAC"]), v["NSTRES"]) * v["SLPF"]).astype(
        np.float32
    )
    assert (car > 0).sum() > 100
    np.testing.assert_array_equal(car, f)
    ov = dumps_dir / "_runs" / run.name.removeprefix("rzwqm46_") / "OVERVIEW.OUT"
    if ov.is_file():
        assert "DSSAT Cropping System Model Ver. 4.0.2" in ov.read_text(errors="replace")


def test_reference_crop_is_nitrogen_limited(run: Path) -> None:
    """O-RZ5 (risk K13): RZWQM's CA-TPA maize runs with nitrogen on and NSTRES is well below 1 on
    most crop days, so an N-off CERES cannot reproduce its growth without N (or NSTRES replay)."""
    v = _t(run, "mz_grosub_exit").values
    assert set(np.char.strip(v["ISWNIT"].astype(str)).tolist()) == {"Y"}
    ns = v["NSTRES"]
    assert ns.min() < 0.6 and (ns < 0.99).mean() > 0.5
    assert np.all((ns >= 0) & (ns <= 1))


@pytest.mark.routines("EVNTRO")
def test_event_water_balance(dump_case: Path) -> None:
    """O-RZ6: every storm puts rain into infiltration or runoff, and the node storage rises by the
    infiltration (no drainage within the event at CA-TPA); one breakpoint per storm."""
    c = dumps.load_case(dump_case)
    nn = int(c.entry["NN"])
    tl = np.diff(np.concatenate([[0.0], c.entry["TLT"][:nn]]))
    ds = np.sum((c.exit["THETA"][:nn] - c.entry["THETA"][:nn]) * tl)
    assert abs(float(c.exit["ZRFDD"]) - float(c.exit["CII"]) - float(c.exit["ROI"])) < 1e-12
    assert abs(ds - float(c.exit["CII"]) + float(c.exit["CDNCI"]) + float(c.exit["DRSEEP"])) < 1e-12
    assert int(c.entry["NBP"]) >= 1 and int(c.entry["NSLT"]) == round(float(c.entry["TLT"][nn - 1]))


@pytest.mark.routines("INFIL")
def test_infil_front_times(dump_case: Path) -> None:
    """O-RZ6: the front-advance times are increasing and every step is above DTMIN = 1e-5 h; the
    last slice is corrected (CORR) so cumulative infiltration ends at the storm total."""
    c = dumps.load_case(dump_case)
    n = int(c.exit["NTIM"])
    tr = np.concatenate([[0.0], c.exit["TR"][:n]])
    assert n >= 1 and np.all(np.diff(tr) > 1e-5)
    assert float(c.exit["CORR"]) >= 0.0


def test_event_summary_counts(run: Path) -> None:
    """The CA-TPA storm facts the M3 event process is scoped on (plan A5): a single breakpoint per
    storm, storms start at 00:00, no storm spans midnight."""
    ph = _t(run, "physcl_exit")
    assert not ph.values["SPAN"].any() and set(ph.values["ISEG"].tolist()) == {1}
    per = _day_sequences(run)
    n = Counter(sum(1 for x in s if x == ("EVNTRO", 0)) for s in per.values())
    assert set(n) <= {0, 1}
