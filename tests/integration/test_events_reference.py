"""Management events keep N species, depth, method and tillage against the reference inputs.

Every check compares the event path (:func:`agrijax.io.rzwqm.events.read_management`,
:func:`agrijax.io.dssat.events.read_treatment_management`, :class:`~agrijax.core.events.EventTable`)
with something that does not go through it:

==========================================  ==============================================================
check                                       independent reference
==========================================  ==============================================================
15 RZWQM2 scenarios, fertilizer + tillage   dates, N forms and implements: ``MANAGE.OUT`` of a full-period
                                            RZWQM2 4.6 run of each scenario (slow); species, method, depth
                                            and codes: the source record re-read with the stdlib only
                                            (comment-block search, ``str.split``)
DSSAT maize FileX, all treatments           the FileX sections re-read with ``str.split`` (treatment factor
                                            levels, simulation options, ``*FERTILIZERS``, ``*RESIDUES``,
                                            ``*TILLAGE``, ``*PLANTING``) and ``FERCH048.SDA`` split on
                                            whitespace; ``Fert_Place``'s day loop (leave at the first record
                                            dated after today) replayed literally
DSSAT applied N and residue                 dscsm048 v4.8.6.0 itself: daily cumulative applied N ``NAPC``
                                            and application count ``NI#M`` of ``SoilNi.OUT``, season totals
                                            ``NICM`` / ``RECM`` of ``Summary.OUT``, for every treatment
CA-TPA ``events.csv``                       totals and days counted with the stdlib ``csv`` module
==========================================  ==============================================================
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from agrijax.core.events import EventTable, source_code
from agrijax.io.rzwqm.events import read_management

BATCH = Path("narval_mirror/RZWQM_sw_batch")
SITES = (
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

pytestmark = pytest.mark.allow_skip(
    reason="needs the private RZWQM2 scenarios / the local DSSAT example data"
)


# ------------------------------------------------------------------ RZWQM2: stdlib reference


def _rz_block(lines: list[str], key: str) -> list[tuple[int, list[str]]]:
    """Records (1-based line, tokens) of the data block after the first comment block holding ``key``."""
    i, n = 0, len(lines)
    while i < n:
        if lines[i].startswith("="):
            j = i
            while j < n and lines[j].startswith("="):
                j += 1
            if key.upper() in "\n".join(lines[i:j]).upper():
                count = int(lines[j].split()[0])
                return [(j + 2 + k, lines[j + 1 + k].split()) for k in range(count)]
            i = j
        else:
            i += 1
    raise KeyError(key)


def _rz_records(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Fertilizer and tillage records of the raw ``rzwqm.dat`` text, by 1-based line."""
    lines = path.read_bytes().decode("latin-1").splitlines()

    def rest(t: list[str]) -> tuple[date | None, list[str]]:
        if int(t[1]) == 5:  # ref 5 dd mm yyyy ...
            return date(int(t[4]), int(t[3]), int(t[2])), t[5:]
        return None, t[3:]  # ref when offset ...

    fert, till = {}, {}
    for ln, t in _rz_block(lines, "F E R T I L I Z E R   M A N A G E M E N T"):
        d, r = rest(t)
        fert[ln] = {
            "date": d,
            "method": int(r[0]),
            "no3": float(r[1]),
            "nh4": float(r[2]),
            "urea": float(r[3]),
        }
    for ln, t in _rz_block(lines, "T I L L A G E   M A N A G E M E N T"):
        d, r = rest(t)
        till[ln] = {"date": d, "implement": int(r[0]), "depth": float(r[1]), "operation": int(r[3])}
    return fert, till


def _rz_period(scenario: Path) -> tuple[date, date]:
    """Simulation period of ``IPNAMES.DAT`` (the ``DD MM YYYY DD MM YYYY`` line), stdlib only."""
    ip = next(p for p in scenario.iterdir() if p.name.upper() == "IPNAMES.DAT")
    for ln in ip.read_bytes().decode("latin-1").splitlines()[8:12]:
        t = ln.split()
        if len(t) >= 6 and all(x.isdigit() for x in t[:6]):
            return date(int(t[2]), int(t[1]), int(t[0])), date(int(t[5]), int(t[4]), int(t[3]))
    raise ValueError(f"{ip}: no simulation period")


@pytest.mark.parametrize("site", SITES)
def test_rzwqm_scenario_event_values(data_dir: Path, site: str) -> None:
    """Every fertilizer / tillage event carries the species, method, depth and codes of its record.

    The dates are checked against RZWQM2 itself (:func:`test_rzwqm_events_match_manage_out`);
    here each event row is matched, by its ``source_line``, with the record re-read with the
    stdlib, and the :class:`EventTable` leaves of its day with that record.
    """
    scenario = data_dir / BATCH / site / "Scenario"
    path = scenario / "rzwqm.dat"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    fert, till = _rz_records(path)
    start, end = _rz_period(scenario)
    df = read_management(path, start, end, skip=("irrigation",), simulation_start=start)
    ev_rows = df[df["event"].str.startswith(("fertilizer_", "tillage"))]
    if ev_rows.empty:
        return
    dates = pd.date_range(start, end, freq="D")
    ev = EventTable.from_frame(df, dates, ignore=("pesticide",), outside="drop")
    a = {k: np.asarray(getattr(ev, k)) for k in ev.field_metadata()}
    t_of = {d.date(): t for t, d in enumerate(dates)}
    exp_n = np.zeros(len(dates))
    exp_till = np.zeros(len(dates), dtype=bool)
    seen_fert: set[date] = set()
    for d, e, v, src in zip(
        ev_rows["date"], ev_rows["event"], ev_rows["value"], ev_rows["source_line"], strict=True
    ):
        day, ln = d.date(), int(src.rsplit(":", 1)[1])
        t = t_of[day]
        if e == "tillage":
            r = till[ln]
            assert r["date"] in (None, day), (site, day, ln)
            assert not exp_till[t], (site, day)  # one tillage record per day (GOTO 120)
            exp_till[t] = True
            assert v == r["depth"] == a["till_depth_cm"][t], (site, day)
            assert a["till_implement"][t] == source_code("rzwqm2", r["implement"]), (site, day)
            assert a["till_operation"][t] == r["operation"], (site, day)
            continue
        r = fert[ln]
        assert r["date"] in (None, day), (site, day, ln)
        assert v == r[e.removeprefix("fertilizer_")], (site, day, e)
        if day in seen_fert:
            continue
        seen_fert.add(day)
        for s in ("no3", "nh4", "urea"):
            assert a[f"fert_{s}_kg_ha"][t] == r[s], (site, day, s)
        exp_n[t] = r["no3"] + r["nh4"] + r["urea"]
        if exp_n[t] > 0:
            assert a["fert_method"][t] == source_code("rzwqm2", r["method"]), (site, day)
            assert r["method"] == 1 and a["fert_depth_cm"][t] == 0.0, (site, day)  # surface broadcast
    np.testing.assert_array_equal(np.asarray(ev.fert_kg_ha), exp_n)
    np.testing.assert_array_equal(a["tillage"], exp_till)
    assert a["fert_org_kg_ha"].sum() == 0.0 and a["fert_unspecified_kg_ha"].sum() == 0.0


# ------------------------------------------------------------------ RZWQM2: MANAGE.OUT reference

_MANAGE_DATE = re.compile(r"^-+\s*(\d+)/\s*(\d+)/\s*(\d{4})\s+-+")


def _manage_out(path: Path) -> tuple[dict[date, dict[str, float]], dict[date, str]]:
    """Fertilizer (N forms) and tillage (implement name) per day of RZWQM2's ``MANAGE.OUT``."""
    fert: dict[date, dict[str, float]] = {}
    till: dict[date, str] = {}
    day: date | None = None
    cur = None
    for ln in path.read_bytes().decode("latin-1", "ignore").replace("\x00", "").splitlines():
        if m := _MANAGE_DATE.match(ln):
            day, cur = date(int(m[3]), int(m[2]), int(m[1])), None
        elif "EVENT ==>" in ln:
            u = ln.upper()
            cur = "tillage" if "TILLAGE" in u else "fert" if "FERTILIZ" in u else None
            if cur == "fert":
                assert day is not None and day not in fert, (path, day)
                fert[day] = {}
        elif cur == "tillage" and "WITH IMPLEMENT:" in ln:
            assert day is not None and day not in till, (path, day)
            till[day] = ln.split(":", 1)[1].strip().lower()
        elif cur == "fert" and "AMOUNT OF" in ln:
            assert day is not None
            form = ln.split("OF", 1)[1].split("[", 1)[0].strip().lower()
            fert[day][form] = float(ln.split("]", 1)[1])
    return fert, till


@pytest.fixture(scope="module")
def manage_runs(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path | str]:
    """``MANAGE.OUT`` of a full-period RZWQM2 run of every scenario (or the error)."""
    from test_rzwqm_all_scenarios import TOOL, _stage_source

    import agrijax.port.run_fortran as rf

    if not (data_dir / BATCH).is_dir():
        pytest.skip(f"{data_dir / BATCH} not found")
    if not (data_dir / TOOL / "main_ryzen5_avx512").is_file():
        pytest.skip("RZWQM binary not found")
    out_root = tmp_path_factory.mktemp("rz_manage_out")
    stage_root = tmp_path_factory.mktemp("rz_manage_src")
    # the reference binary needs a run dir <= 45 characters: honour AGRI_JAX_RUN_ROOT like run_fortran
    run_root = Path(os.environ.get("AGRI_JAX_RUN_ROOT", str(data_dir / "run")))

    def one(site: str) -> Path | str:
        try:
            src = _stage_source(site, data_dir / BATCH / site / "Scenario", stage_root, run_root)
            rf.run_rzwqm(src, out_root / site, keep_files=("MANAGE.OUT",), timeout=1800, run_root=run_root)
            return out_root / site / "MANAGE.OUT"
        except Exception as e:  # reported per scenario
            return f"{type(e).__name__}: {e}"

    # threads: the work is the Fortran subprocess (no fork after JAX started its threads)
    with ThreadPoolExecutor(max(1, min(len(SITES), os.cpu_count() or 1))) as ex:
        return dict(zip(SITES, ex.map(one, SITES), strict=True))


@pytest.mark.slow
@pytest.mark.parametrize("site", SITES)
def test_rzwqm_events_match_manage_out(manage_runs: dict[str, Path | str], data_dir: Path, site: str) -> None:
    """Fertilizer days and N forms, tillage days and implements: RZWQM2's own ``MANAGE.OUT``.

    Over the whole simulation period of the scenario: this covers ``MAQUE``'s plant-reference
    gate (records of a crop that is not being managed do not fire, fixed dates included), the
    first-match-per-day rules and the relative timings.
    """
    out = manage_runs[site]
    if isinstance(out, str):
        pytest.fail(f"{site}: RZWQM2 run failed: {out}")
    scenario = data_dir / BATCH / site / "Scenario"
    start, end = _rz_period(scenario)
    fert, till = _manage_out(out)
    df = read_management(scenario / "rzwqm.dat", start, end, skip=("irrigation",), simulation_start=start)
    got_fert: dict[date, dict[str, float]] = {}
    got_till: dict[date, str] = {}
    for d, e, v, det in zip(df["date"], df["event"], df["value"], df["detail"], strict=True):
        if e.startswith("fertilizer_"):
            got_fert.setdefault(d.date(), {})[e.removeprefix("fertilizer_")] = float(v)
        elif e == "tillage":
            got_till[d.date()] = dict(p.split("=", 1) for p in det.split(";"))["implement"]
    assert sorted(got_fert) == sorted(fert), site
    for d, amounts in fert.items():
        assert {k: v for k, v in amounts.items() if v} == {k: v for k, v in got_fert[d].items() if v}, (
            site,
            d,
        )
    assert got_till == till, site


def test_catpa_events_csv_totals(data_dir: Path) -> None:
    path = data_dir / "catpa" / "events.csv"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="D")
    ev = EventTable.from_csv(path, dates)
    no3 = [float(r["value"]) for r in rows if r["event"] == "fertilizer_no3"]
    till = [r for r in rows if r["event"] == "tillage"]
    assert float(ev.fert_no3_kg_ha.sum()) == sum(no3) == 7 * 180.0
    assert float(ev.fert_kg_ha.sum()) == sum(
        float(r["value"]) for r in rows if r["event"].startswith("fertilizer")
    )
    assert int(ev.tillage.sum()) == len(till) == 7
    assert float(ev.till_depth_cm.sum()) == sum(float(r["value"]) for r in till)
    till_days = sorted(pd.Timestamp(r["date"]) for r in till)
    assert list(dates[np.asarray(ev.tillage)]) == till_days
    assert set(np.asarray(ev.fert_method)[np.asarray(ev.fert_kg_ha) > 0]) == {source_code("rzwqm2", 1)}
    assert float(ev.fert_nh4_kg_ha.sum() + ev.fert_urea_kg_ha.sum()) == 0.0


# ------------------------------------------------------------------ DSSAT: stdlib reference


def _dssat_engine() -> Path:
    from agrijax.port.run_fortran import DSSAT_ENGINE

    return DSSAT_ENGINE


def _maize_files() -> list[Path]:
    d = _dssat_engine() / "example_data" / "Maize"
    return sorted(d.glob("*.MZX")) if d.is_dir() else []


def _sections(path: Path) -> dict[str, list[tuple[list[str], list[str]]]]:
    """``{section: [(header tokens, raw data lines), ...]}`` with the stdlib only."""
    out: dict[str, list[tuple[list[str], list[str]]]] = {}
    sec = None
    for ln in path.read_bytes().decode("latin-1").splitlines():
        if ln.startswith("*"):
            sec = ln[1:].split("  ")[0].strip().upper()
            out.setdefault(sec, [])
        elif sec is None or not ln.strip() or ln.startswith("!"):
            continue
        elif ln.startswith("@"):
            out[sec].append((ln[1:].split(), []))
        elif out.get(sec):
            out[sec][-1][1].append(ln)
    return out


def _sec(s: dict, prefix: str) -> list[tuple[list[str], list[str]]]:
    return next((v for k, v in s.items() if k.startswith(prefix)), [])


def _yrdoy(code: str) -> date:
    v = int(code)
    yy, doy = divmod(v, 1000)
    return date(2000 + yy if yy <= 35 else 1900 + yy, 1, 1) + timedelta(days=doy - 1)


def _ferch() -> dict[int, tuple[float, float, float]]:
    p = _dssat_engine() / "source" / "Data" / "StandardData" / "FERCH048.SDA"
    out = {}
    for ln in p.read_bytes().decode("latin-1").splitlines():
        if ln[:2] == "FE" and ln[2:5].isdigit():
            nums = ln[44:].split()  # after (6X,A35,3X): names may run into the 3X gap
            out[int(ln[2:5])] = (float(nums[0]), float(nums[1]), float(nums[2]))
    return out


def _tilop() -> dict[str, float]:
    """``TILOP048.SDA``: implement -> its deepest layer (last ``SLB`` row, whitespace split)."""
    p = _dssat_engine() / "source" / "Data" / "StandardData" / "TILOP048.SDA"
    out: dict[str, float] = {}
    cur, in_slb = None, False
    for ln in p.read_bytes().decode("latin-1").splitlines():
        if ln.startswith("*"):
            cur, in_slb = ln[1:6], False
        elif ln.startswith("@"):
            in_slb = ln[1:].split()[:1] == ["SLB"]
        elif cur and in_slb and ln.strip() and not ln.startswith("!"):
            slb = float(ln.split()[0])
            if slb < 0.01:
                cur = None
            else:
                out[cur] = slb
    return out


def _dssat_reference(path: Path, trno: int) -> dict[str, Any]:
    """Expected per-day fertilizer / residue / tillage of one treatment, stdlib only."""
    s = _sections(path)
    trt = None
    for _hdr, rows in _sec(s, "TREATMENTS"):
        for ln in rows:
            if int(ln[:3]) == trno:
                toks = ln.split()[-13:]  # CU FL SA IC MP MI MF MR MC MT ME MH SM
                trt = dict(zip("CU FL SA IC MP MI MF MR MC MT ME MH SM".split(), map(int, toks), strict=True))
    assert trt is not None
    opts: dict[str, str] = {"FERTI": "R", "RESID": "R", "TILL": "Y", "PLANT": "R"}
    for hdr, rows in _sec(s, "SIMULATION"):
        if len(hdr) > 1 and hdr[1] in ("MANAGEMENT", "OPTIONS"):
            for ln in rows:
                t = ln.split()
                if int(t[0]) == trt["SM"]:
                    opts.update(dict(zip(hdr[2:], t[2:], strict=False)))
                    break
    planting = None
    for hdr, rows in _sec(s, "PLANTING"):
        for ln in rows:
            t = ln.split()
            if int(t[0]) == trt["MP"] and hdr[1] == "PDATE":
                planting = _yrdoy(t[1])

    def rows_of(prefix: str, level: int) -> list[list[str]]:
        return [ln.split() for _h, rows in _sec(s, prefix) for ln in rows if int(ln.split()[0]) == level]

    def day_loop(recs: list[date]) -> list[int]:
        """Indices fired by a DSSAT day loop that leaves at the first record dated after today."""
        fired = []
        for d in sorted(set(recs)):
            for i, di in enumerate(recs):
                if di == d:
                    fired.append(i)
                elif di > d:
                    break
        return sorted(fired)

    def when(code: str, mode: str) -> date:
        if mode == "D":
            assert planting is not None
            return planting + timedelta(days=int(code))
        return _yrdoy(code)

    ferch = _ferch()
    fert: dict[date, dict[str, float]] = {}
    if trt["MF"] and opts["FERTI"] in ("R", "D"):
        recs = rows_of("FERTI", trt["MF"])  # F FDATE FMCD FACD FDEP FAMN ...
        ds = [when(r[1], opts["FERTI"]) for r in recs]
        for i in day_loop(ds):
            r = recs[i]
            famn = float(r[5])
            if famn == 0:
                continue
            pct = ferch[int(r[2][2:])]
            f = fert.setdefault(ds[i], {"no3": 0.0, "nh4": 0.0, "urea": 0.0})
            for k, p in zip(("no3", "nh4", "urea"), pct, strict=True):
                f[k] += famn * p / 100.0
            meth = int(r[3][2:]) if r[3].startswith("AP") else 1
            f["method"] = source_code("dssat", meth)
            f["depth"] = float(r[4])
    res: dict[date, dict[str, float]] = {}
    if trt["MR"] and opts["RESID"] in ("R", "D"):
        recs = rows_of("RESID", trt["MR"])  # R RDATE RCOD RAMT RESN RESP RESK RINP RDEP RMET
        ds = [when(r[1], opts["RESID"]) for r in recs]
        for i in day_loop(ds):
            r = recs[i]
            amt, resn, rinp, rdep = float(r[3]), float(r[4]), float(r[7]), float(r[8])
            if amt <= 0:
                continue
            if rinp < 0:
                rinp, rdep = (100.0, max(rdep, 15.0)) if rdep > 0 else (0.0, rdep)
            res[ds[i]] = {"amount": amt, "n": amt * resn / 100.0, "depth": max(rdep, 0.0), "incorp": rinp}
    till: list[tuple[date, float, int]] = []
    if trt["MT"] and opts["TILL"] in ("Y", "R"):
        deepest = _tilop()
        for r in rows_of("TILLA", trt["MT"]):  # T TDATE TIMPL TDEP
            depth = float(r[3])
            if depth - deepest[r[2]] > 0.5:  # Tillage.for: TDEP capped at the implement's last SLB
                depth = deepest[r[2]]
            if depth > 0:
                till.append((_yrdoy(r[1]), depth, int(r[2][2:])))
    return {"fert": fert, "residue": res, "tillage": till, "planting": planting}


def _cases() -> list[tuple[Path, int]]:
    out = []
    for f in _maize_files():
        for _hdr, rows in _sec(_sections(f), "TREATMENTS"):
            out.extend((f, int(ln[:3])) for ln in rows)
    return out


@pytest.mark.parametrize(
    ("filex", "trno"), _cases() or [pytest.param(None, 0, marks=pytest.mark.skip("no data"))]
)
def test_dssat_filex_events(filex: Path, trno: int) -> None:
    from agrijax.io.dssat.events import read_treatment_management

    ref = _dssat_reference(filex, trno)
    df = read_treatment_management(filex, trno)
    days = [*ref["fert"], *ref["residue"], *(d for d, _, _ in ref["tillage"])] + (
        [ref["planting"]] if ref["planting"] else []
    )
    if not days:
        assert df.empty
        return
    dates = pd.date_range(min(days), max(days), freq="D")
    ev = EventTable.from_frame(df, dates, ignore=())
    a = {k: np.asarray(getattr(ev, k)) for k in ev.field_metadata()}
    t_of = {d.date(): t for t, d in enumerate(dates)}
    exp_total = np.zeros(len(dates))
    for d, f in ref["fert"].items():
        t = t_of[d]
        for s in ("no3", "nh4", "urea"):
            np.testing.assert_allclose(a[f"fert_{s}_kg_ha"][t], f[s], rtol=0, atol=1e-12, err_msg=f"{d} {s}")
        exp_total[t] = f["no3"] + f["nh4"] + f["urea"]
        assert a["fert_method"][t] == f["method"], d
        assert a["fert_depth_cm"][t] == f["depth"], d
    np.testing.assert_allclose(np.asarray(ev.fert_kg_ha), exp_total, rtol=0, atol=1e-12)
    exp_res = np.zeros(len(dates))
    for d, r in ref["residue"].items():
        t = t_of[d]
        exp_res[t] = r["amount"]
        np.testing.assert_allclose(a["residue_n_kg_ha"][t], r["n"], rtol=1e-14)
        assert (a["residue_depth_cm"][t], a["residue_incorp_pct"][t]) == (r["depth"], r["incorp"]), d
    np.testing.assert_array_equal(a["residue_kg_ha"], exp_res)
    _check_tillage(ref, a, dates)
    if ref["planting"] is not None:
        assert list(dates[a["sow"]]) == [pd.Timestamp(ref["planting"])]


def _check_tillage(ref: dict[str, Any], a: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> None:
    assert sorted(d.date() for d in dates[a["tillage"]]) == sorted({d for d, _, _ in ref["tillage"]})
    t_of = {d.date(): t for t, d in enumerate(dates)}
    for d, depth, implement in ref["tillage"]:
        t = t_of[d]
        assert a["till_depth_cm"][t] == depth, d
        assert a["till_implement"][t] == source_code("dssat", implement), d


def _tillage_cases() -> list[tuple[Path, int]]:
    """Treatments with an applied tillage event in the example FileX of every crop."""
    d = _dssat_engine() / "example_data"
    out = []
    for f in sorted(d.glob("*/*.??X")) if d.is_dir() else []:
        if f.suffix.upper() == ".SQX":  # sequence runs: out of the reader's scope (single season)
            continue
        s = _sections(f)
        if not _sec(s, "TILLA"):
            continue
        out.extend((f, trno) for _h, rows in _sec(s, "TREATMENTS") for trno in (int(ln[:3]) for ln in rows))
    return [(f, n) for f, n in out if _dssat_reference(f, n)["tillage"]]


@pytest.mark.parametrize(
    ("filex", "trno"), _tillage_cases() or [pytest.param(None, 0, marks=pytest.mark.skip("no data"))]
)
def test_dssat_filex_tillage(filex: Path, trno: int) -> None:
    """Tillage days, depth (capped at the implement's deepest ``TILOP048`` layer) and implement.

    The maize examples have no applied tillage, so the example FileX of all crops are used.
    """
    from agrijax.io.dssat.events import read_treatment_management

    ref = _dssat_reference(filex, trno)
    df = read_treatment_management(filex, trno)
    till = pd.DataFrame(df[df["event"] == "tillage"])
    rows = [
        (d.date(), float(v), int(det.split(":")[1]))
        for d, v, det in zip(till["date"], till["value"], till["detail"], strict=True)
    ]
    assert sorted(rows) == sorted(ref["tillage"])
    days = [d for d, _, _ in ref["tillage"]]
    if len(set(days)) < len(days):  # same-day operations: one tillage placement per day in EventTable
        with pytest.raises(ValueError, match="one placement per kind and day"):
            EventTable.from_frame(till, pd.date_range(min(days), max(days), freq="D"), ignore=())
        return
    dates = pd.date_range(min(days), max(days), freq="D")
    ev = EventTable.from_frame(till, dates, ignore=())
    _check_tillage(ref, {k: np.asarray(getattr(ev, k)) for k in ev.field_metadata()}, dates)


# ------------------------------------------------------------------ DSSAT: dscsm048 reference


def _dscsm() -> Path:
    from agrijax.port.run_fortran import dscsm_paths

    return dscsm_paths()[0]


@pytest.mark.parametrize("filex", _maize_files() or [pytest.param(None, marks=pytest.mark.skip("no data"))])
def test_dssat_applied_n_matches_dscsm048(filex: Path, tmp_path: Path) -> None:
    """Daily cumulative applied N and application count, season N and residue totals, per treatment."""
    if not _dscsm().is_file():
        pytest.skip(f"dscsm048 not found at {_dscsm()}")
    from agrijax.io.dssat import read_out, read_summary, weather_stations
    from agrijax.io.dssat.events import read_treatment_management
    from agrijax.port.run_fortran import run_dscsm

    eng = _dssat_engine()
    exp = tmp_path / "exp"
    exp.mkdir()
    for f in filex.parent.glob(filex.stem + ".*"):
        shutil.copy2(f, exp / f.name)
    stations = {w[:4].upper() for w in weather_stations(exp / filex.name)}
    extra = [
        w for w in sorted((eng / "example_data" / "Weather").glob("*.WTH")) if w.name[:4].upper() in stations
    ]
    extra += sorted((eng / "source" / "Data" / "Pest").glob("*.PST"))
    r = run_dscsm(
        exp,
        tmp_path / "out",
        experiment_file=filex.name,
        extra_files=extra,
        keep_files=("SoilNi.OUT", "Summary.OUT"),
        run_root=tmp_path,
        check=False,
    )
    assert r.summary_path is not None
    summary = read_summary(r.summary_path)
    soil_ni = r.out_dir / "SoilNi.OUT"
    ni = read_out(soil_ni) if soil_ni.is_file() else None
    assert len(summary) > 0
    for rec in summary.to_dict("records"):
        trno = int(rec["TRNO"])
        df = read_treatment_management(filex, trno)
        fert = pd.DataFrame(df[df["event"].str.startswith("fertilizer_")])
        res = pd.DataFrame(df[df["event"] == "residue"])
        assert float(np.asarray(fert["value"], dtype=float).sum()) == pytest.approx(
            float(rec["NICM"]), abs=0.5
        ), trno
        assert float(np.asarray(res["value"], dtype=float).sum()) == pytest.approx(
            float(rec["RECM"]), abs=0.5
        ), trno
        if ni is None:  # nitrogen not simulated: no daily file
            continue
        g = pd.DataFrame(ni[ni["TRNO"] == trno])
        ev = EventTable.from_frame(df, pd.DatetimeIndex(g["DATE"]), ignore=())
        napc = np.asarray(g["NAPC"], dtype=float)
        np.testing.assert_allclose(np.cumsum(np.asarray(ev.fert_kg_ha)), napc, atol=0.5)
        counts = fert.groupby(["date", "source_line"]).ngroups
        assert int(np.asarray(g["NI#M"])[-1]) == counts, trno
