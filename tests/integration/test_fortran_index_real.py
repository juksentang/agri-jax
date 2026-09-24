"""Cross-check of the Fortran index on the real RZWQM2 and CERES-Maize sources.

Evidence, each independent of the fparser index (``agrijax.port.fortran_index``):

* a second, line-based scanner (``agrijax.port.fortran_xcheck``) must reproduce every routine
  name, argument list (order and names), ``CALL`` target, ``COMMON`` block, explicit ``SAVE`` and
  call-graph edge of the stored index JSON (``<data-dir>/port_index/{rzwqm,ceres_maize}.json``);
* the stored JSON must be what the current indexer produces (no stale index);
* the topological order must put every callee before its caller (checked by a separate DFS);
* five routines were read by hand (WC, POINTK, RICHRD, POTEVPHR in RZWQM2; MZ_PHENOL in
  CERES-Maize). Their argument intents and state-carrying locals are written below as a
  person's judgement, with the reasoning, and the index must agree.

The source is private (RZWQM2) or external (DSSAT) and never copied here; the comments describe
what the code does, by variable name, without quoting it. Everything skips when the sources are
absent.

Intent convention (the index's, see ``fortran_index``): ``in`` = never written in the routine or
its indexed callees; ``out`` = written somewhere and never read before a write on any path;
``inout`` = read before a definite write on some path and written somewhere. A "conditional
out" (written only on some paths, so the caller's old value survives on the others) is ``out``
under this convention; those are listed separately per routine because a JAX port must treat
them as pass-through (inout).

Slow whole-tree scans (all RZWQM ``*.for``, all DSSAT ``Plant/*/*.for``) pin the disagreements
found outside the indexed subset; each is explained where it is listed.
"""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

from agrijax.port.fortran_xcheck import (
    check_topological_order,
    compare,
    kahn_is_acyclic,
    logical_statements,
    scan_files,
)

pytest.importorskip("fparser")

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
RZ_SRC = Path("narval_mirror/RZWQM_Linux_Ver45/src/RZWQM")

# fixed-form statement field: RZWQM2 is built with ifort defaults (72 columns, no
# -extend-source in Makefile.mak); DSSAT with -ffixed-line-length-none.
LINE_LENGTH = {"rzwqm": 72, "ceres_maize": None}


def _load(data_dir: Path, name: str) -> dict[str, Any]:
    p = data_dir / "port_index" / f"{name}.json"
    if not p.is_file():
        pytest.skip(f"index {p} not found")
    index = json.loads(p.read_text())
    missing = [f for f in index["files"] if not Path(f).is_file()]
    if missing:
        pytest.skip(f"Fortran sources of {name} not found: {missing[:2]}")
    return index


@pytest.fixture(scope="module", params=["rzwqm", "ceres_maize"])
def index_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def rzwqm(data_dir: Path) -> dict[str, Any]:
    return _load(data_dir, "rzwqm")


@pytest.fixture(scope="module")
def ceres(data_dir: Path) -> dict[str, Any]:
    return _load(data_dir, "ceres_maize")


def _sub(index: dict[str, Any], name: str) -> dict[str, Any]:
    return next(s for s in index["subroutines"] if s["name"] == name)


def _state(s: dict[str, Any]) -> set[str]:
    return {v["name"] for v in s["saved_vars"] if v["read_before_assigned"]}


# ---------------------------------------------------------------------------
# Independent scanner vs index
# ---------------------------------------------------------------------------
def test_scanner_agrees_with_index(data_dir: Path, index_name: str) -> None:
    index = _load(data_dir, index_name)
    assert index["errors"] == [] and index["duplicates"] == {}
    units = scan_files(index["files"], line_length=LINE_LENGTH[index_name])
    assert len(units) == index["n_subroutines"]
    diffs = compare(index, units)
    assert diffs == [], "\n".join(map(str, diffs))
    # the comparison is not vacuous: it saw real routines, arguments, edges and COMMONs
    assert sum(len(u.args) for u in units) > 300
    assert sum(len(v) for v in index["call_graph"].values()) >= 9


def test_index_json_is_current(data_dir: Path, index_name: str) -> None:
    """The stored JSON is exactly what the current indexer produces from the same files."""
    from agrijax.port.fortran_index import index_tree

    index = _load(data_dir, index_name)
    fresh = index_tree(index["files"]).to_dict(list(index["subtrees"]))
    assert fresh == index


def test_rzwqm_has_no_code_past_column_72(rzwqm: dict[str, Any]) -> None:
    """fparser reads whole lines while ifort stops at column 72: on these files both see the same code."""
    for f in rzwqm["files"]:
        text = Path(f).read_text(encoding="latin-1")
        assert logical_statements(text, line_length=72) == logical_statements(text, line_length=None), f


def test_topological_order(data_dir: Path, index_name: str) -> None:
    index = _load(data_dir, index_name)
    g = index["call_graph"]
    assert check_topological_order(g, index["topological_order"]) == []
    assert kahn_is_acyclic(g) == (index["cycles"] == [])
    for root, sub in index["subtrees"].items():
        keep = set(sub["order"])
        assert root in keep
        assert check_topological_order({k: [v for v in g[k] if v in keep] for k in keep}, sub["order"]) == []


# ---------------------------------------------------------------------------
# Hand-verified routines
# ---------------------------------------------------------------------------
def _assert_routine(
    s: dict[str, Any],
    *,
    args: Sequence[str],
    intents: dict[str, str],
    true_state: Iterable[str],
    conservative: Iterable[str] = (),
    saved_not_state: Iterable[str] = (),
) -> None:
    assert s["args"] == list(args)
    assert s["intent_guess"] == intents
    # Soundness: every variable I judged to carry state is flagged. Precision: every other flag
    # is one I explained as conservative (the must-define analysis cannot see the correlation).
    assert _state(s) == set(true_state) | set(conservative)
    explicit_no_state = {
        v["name"] for v in s["saved_vars"] if not v["read_before_assigned"] and v["kind"] == "save"
    }
    assert explicit_no_state == set(saved_not_state)


def test_hand_wc(rzwqm: dict[str, Any]) -> None:
    # WC(H, HYDP, I, ISTAT), Brooks-Corey theta(h), Rzrich.for.
    # Arguments: none is ever assigned -> all 'in'. ISTAT = -1 is a "store the tillage-reference
    # curve for horizon I" request, H the head, HYDP the 13 current curve parameters.
    # State: TRHYDP(13, MAXHOR) is in a SAVE statement; it is written only when ISTAT == -1 and
    # read on the dry branch (h below -10 * bubbling pressure) on later calls -> real state (a
    # per-horizon copy of the pre-tillage parameters). B and TH are assigned before use on every
    # branch that reads them; J is the store-loop counter -> not state.
    _assert_routine(
        _sub(rzwqm, "WC"),
        args=["H", "HYDP", "I", "ISTAT"],
        intents=dict.fromkeys(["H", "HYDP", "I", "ISTAT"], "in"),
        true_state={"TRHYDP"},
    )
    assert _sub(rzwqm, "WC")["function_refs"] == ["BCBETA"]


def test_hand_pointk(rzwqm: dict[str, Any]) -> None:
    # POINTK(HEAD, HYDP, J, PORI), point hydraulic conductivity K(h), Rzrich.for.
    # Arguments: none assigned -> all 'in'; PORI is not used at all (its use is commented out).
    # State (SAVE IRST, C22, SN22; DATA IRST /0/): IRST is compared with J before it is
    # updated -> state. C22(J) / SN22(J) are refreshed only when J > IRST (first visit of a
    # higher horizon index) and read on the very dry branch -> state: once cached, a horizon's
    # K(h) tail never picks up later changes of HYDP(11) / HYDP(3). TH is set from HEAD first.
    # COMMON /HYDROL/ is declared but only its storage is shared; nothing of it is used here.
    s = _sub(rzwqm, "POINTK")
    _assert_routine(
        s,
        args=["HEAD", "HYDP", "J", "PORI"],
        intents=dict.fromkeys(["HEAD", "HYDP", "J", "PORI"], "in"),
        true_state={"IRST", "C22", "SN22"},
    )
    assert [c["name"] for c in s["common_blocks"]] == ["HYDROL"]
    assert s["calls"] == [] and s["function_refs"] == []


RICHRD_ARGS = (
    "DELT DELZ H HOLD NDXN2H NN QF QS HKBAR SOILHP TL THETA START EVAP AEVAP IREBOT BOTHED "
    "BOTFLX ALPH CNVG H2OTAB DAYTIM ITBL QT IMAGIC RICH_EPS MAXITER_RICH MAXCYCLE_RICH PORI QSN "
    "ISHAW EVAPS HMIN NSP QSSI ISTRESS"
).split()


def test_hand_richrd(rzwqm: dict[str, Any]) -> None:
    # RICHRD: one Richards-equation step on the redistribution grid, Rzrich.for.
    # Body: first-call init of BOTHED, REDGRD (true grid -> computational grid, calls INITCOND),
    # CHKBC (boundary conditions), CNHEAD (Picard/Newton solve), NODFLX, WCNODS, TRUGRD (back).
    # inout, with the path that makes them so:
    #   DELT   read by REDGRD/CHKBC, halved by CNHEAD on non-convergence (adaptive step).
    #   H      read by REDGRD, overwritten by TRUGRD.       HOLD  read by REDGRD, reset here.
    #   QS, THETA, QT, QSSI  read by REDGRD, overwritten by TRUGRD.
    #   BOTHED set from H(NN) only on the first call, read by CHKBC / CNHEAD.
    #   ALPH   set only when DAYTIM <= 0 or ITBL /= 1; otherwise the caller's weight is used.
    #   CNVG   CNHEAD reads it (resets its counter when true) and clears it on mass-balance failure.
    # out: QF, HKBAR, AEVAP -- written unconditionally by TRUGRD, never read before.
    # in: everything else (never written here nor by a callee; BOTFLX is only read by CHKBC).
    inout = {"DELT", "H", "HOLD", "QS", "THETA", "BOTHED", "ALPH", "CNVG", "QT", "QSSI"}
    out = {"QF", "HKBAR", "AEVAP"}
    intents = {a: "inout" if a in inout else "out" if a in out else "in" for a in RICHRD_ARGS}
    # State:
    #   FIRST (SAVE + DATA .TRUE.)  first-call flag -> state.
    #   NNC (SAVE)  REDGRD sets it (NN + 2) only when START, otherwise uses the saved value -> state.
    #   TH (local, implicitly saved under -save)  passed to CNHEAD as its H, which on the
    #     flux-bottom branch writes H(NN) only if a denominator is non-zero and then reads H(NN):
    #     in that degenerate case the previous step's TH(NN) leaks in -> real (edge-case) state.
    #   TTL, TDELZ, NDXHOR are in the SAVE list but INITCOND rewrites them on every call before
    #     use -> saved, not state.
    _assert_routine(
        _sub(rzwqm, "RICHRD"),
        args=RICHRD_ARGS,
        intents=intents,
        true_state={"FIRST", "NNC", "TH"},
        saved_not_state={"TTL", "TDELZ", "NDXHOR"},
    )


POTEVPHR_ARGS = (
    "ASPECT CS ELEV EPAN FT FTR JDAY PET PER PES RR RTS S THETA TMIN TMAX U WRES LAI XLAT PP "
    "ICRUST RH WC13 WC15 IPL HEIGHT AS ESN RCS PKTEMP IRTYPE TLAI IPR SDEAD_HEIGHT SDEAD TL "
    "CSHSLAB THERMK DELT PFIRST H IPENFLUX AEVAP TAIR TZ1 TZLB ZNLB DAYTIM CLOUDS GFLUX ISHAW "
    "IYYY WCSAT IHFLAG IHOURLY TM1 TM2 HKBAR NN RDF QS ATRANS HROOT PUP TRTS PPOP STEMAI ZSTUBL "
    "STUBLW PA XLH RHORB HRM CR AR UNEW NSP NR RHOSP ZSP ORTS DELZ COR ITIME IWZONE HRTS CO2R "
    "TRAT RTH HRTH JPENFLUX TM4 RTSTOT"
).split()


def test_hand_potevphr(rzwqm: dict[str, Any]) -> None:
    # POTEVPHR: hourly/daily Shuttleworth-Wallace PET with optional PENFLUX energy balance, Rzpet.for.
    # inout:
    #   RTS, RTH  copied to ORTS/ORTH first, then replaced by SWSUN(...) or restored.
    #   HRTH      element ITIME written only when hourly and RTSTOT == 0; read for RSRATIO.
    #   PFIRST    PENFLUX's first-call flag (read and cleared there).
    #   HRM       RESISTHR updates it.
    # out (written before any read): PET, PER, PES (pan branch sets all three then jumps to the
    #   end; otherwise S-W sets them and they are scaled), FT, FTR, CR, AR, UNEW, AS, ORTS,
    #   GFLUX (zeroed first), and callee outputs CS, STEMAI, ZSTUBL, STUBLW, RHORB (RESISTHR),
    #   PA, XLH (ECONST), PP, RCS (MAXSW; RCS is then raised to at least RTS), plus the
    #   conditional outs below.
    # Conditional outs (index 'out'; a port must pass the old value through):
    #   ESN            not written on the pan-evaporation path (early jump past it).
    #   TM1, TM2, TM4  written only when PENFLUX runs (IPENFLUX == JPENFLUX == 1).
    #   HRTS           only element ITIME, only when hourly and RTSTOT == 0.
    # in: all others (many are unused: DAYTIM, IHFLAG, HKBAR, NN, RDF, QS, ATRANS, HROOT, PUP,
    #   TRTS, PPOP, NR, RHOSP, ZSP, DELZ, COR, RTSTOT only in a test, ...).
    inout = {"RTS", "RTH", "HRTH", "PFIRST", "HRM"}
    out = set(
        "PET PER PES FT FTR CR AR UNEW AS ORTS GFLUX CS STEMAI ZSTUBL STUBLW RHORB PA XLH PP RCS "
        "ESN TM1 TM2 TM4 HRTS".split()
    )
    intents = {a: "inout" if a in inout else "out" if a in out else "in" for a in POTEVPHR_ARGS}
    # State that really crosses calls:
    #   PENFLUXE, PENFLUXT (SAVE)  hourly accumulators, reset after hour 24 -> state.
    #   RCSHR (local hourly array)  MAXSW fills it only on sloped ground (S /= 0); on flat ground
    #     it keeps its initial value (0 under -init=zero) and is still read when hourly -> state
    #     (constant, compiler-init dependent).
    # Flagged but not real state (conservative: the analysis cannot correlate conditions):
    #   W1, W2, XWNEW  set under "REPHEIGHT > 0" / "ELSEIF REPHEIGHT <= 0" (exhaustive, no ELSE).
    #   RAA, RAS       RESISTHR sets them under "LAI > 0 and height > 0 (or SAI > 0)" and again
    #                  under "LAI == 0 or height == 0": exhaustive for physical inputs.
    #   RSNSW, RSNRW, RSNCW, GFLW  set in the ISHAW/IPENFLUX block; read only in the PENFLUX
    #                  output block, which implies that block ran.
    #   RLNSW, RLNRW, RLNCW  PENFLUX outputs read in the ISHAW/IPENFLUX block even when PENFLUX
    #                  did not run this call, but only into RLNS/RLNR/RLNC, which are dead.
    #   EFLUX, HTFLUX  PENFLUX outputs read every call, but their values only reach PENFLUXE/T and
    #                  output under the same condition that made PENFLUX run.
    # WNDADJ is SAVEd (DATA 1.0) but recomputed before use -> saved, not state.
    _assert_routine(
        _sub(rzwqm, "POTEVPHR"),
        args=POTEVPHR_ARGS,
        intents=intents,
        true_state={"PENFLUXE", "PENFLUXT", "RCSHR"},
        conservative="W1 W2 XWNEW RAA RAS RSNSW RSNRW RSNCW GFLW RLNSW RLNRW RLNCW EFLUX HTFLUX".split(),
        saved_not_state={"WNDADJ"},
    )
    s = _sub(rzwqm, "POTEVPHR")
    assert [c["name"] for c in s["common_blocks"]] == ["IPOTEV", "RESID"]
    assert s["calls"] == ["ECONST", "MAXSW", "NETRAD", "PENFLUX", "RESISTHR"]


MZ_PHENOL_ARGS = (
    "DYNAMIC ISWWAT FILEIO IDETO CUMDEP DAYL DLAYR LEAFNO LL NLAYR PLTPOP SDEPTH SNOW SRAD SUMP SW "
    "TMAX TMIN TWILEN XN YRDOY YRSIM CUMDTT DTT EARS GPP ISDATE ISTAGE MDATE STGDOY SUMDTT XNTI "
    "TLNO XSTAGE YREMRG RUE KCAN KEP P3 TSEN CDAY SEEDFRAC VEGFRAC CROPSTATUS"
).split()


def test_hand_mz_phenol(ceres: dict[str, Any]) -> None:
    # MZ_PHENOL: CERES-Maize phenology, DSSAT DYNAMIC dispatch (RUNINIT/SEASINIT read files and
    # reset; otherwise the daily thermal time and the ISTAGE state machine). The index analyses
    # the whole routine as one call, so an argument written in one phase and read in another
    # (on a later call) is inout.
    # inout:
    #   PLTPOP, SDEPTH, YREMRG  read from FILEIO in RUNINIT (only if the section is found), read
    #     later (seed depth -> layer, emergence date test, ear number).
    #   KCAN     read from the ecotype file in RUNINIT; KEP is derived from it in SEASINIT too.
    #   CUMDTT, SUMDTT  accumulators (reset at init, += DTT daily).
    #   GPP      stage 6 tests "GPP <= 0" on a value set on an earlier day (stage 4).
    #   ISTAGE   the state-machine selector.   P3  set in stage 2, read in stage 3 (later days).
    #   VEGFRAC  MAX(VEGFRAC, ...) keeps the running maximum.
    #   FILEIO   judged 'in' by me (a file name, only passed to OPEN / ERROR); the index says
    #     'inout' because ERROR is outside the indexed files (unresolved callee -> conservative).
    #     The index records this in unresolved_call_args (asserted below).
    # out: CUMDEP (stage 7), DTT (every arm of the thermal-time IF sets it), EARS, ISDATE, MDATE,
    #   STGDOY (element writes), XNTI, TLNO, XSTAGE, RUE, KEP, TSEN, CDAY, SEEDFRAC, CROPSTATUS.
    #   All are conditional in DSSAT's phase sense (e.g. MDATE only at maturity or crop failure,
    #   STGDOY one element per stage), so they are pass-through for a port.
    inout = {"PLTPOP", "SDEPTH", "YREMRG", "KCAN", "CUMDTT", "SUMDTT", "GPP", "ISTAGE", "P3", "VEGFRAC"}
    out = set(
        "CUMDEP DTT EARS ISDATE MDATE STGDOY XNTI TLNO XSTAGE RUE KEP TSEN CDAY SEEDFRAC CROPSTATUS".split()
    )
    mine = {a: "inout" if a in inout else "out" if a in out else "in" for a in MZ_PHENOL_ARGS}
    s = _sub(ceres, "MZ_PHENOL")
    assert s["args"] == MZ_PHENOL_ARGS
    assert {a: v for a, v in s["intent_guess"].items() if a != "FILEIO"} == {
        a: v for a, v in mine.items() if a != "FILEIO"
    }
    assert mine["FILEIO"] == "in" and s["intent_guess"]["FILEIO"] == "inout"
    assert "ERROR:2:FILEIO" in s["unresolved_call_args"] and "ERROR" in ceres["external_calls"]["MZ_PHENOL"]
    # State: a bare SAVE saves every local. The ones read before written in a call:
    #   parameters read in RUNINIT and used on every later (rate) call:
    #     cultivar P1 P2 P5 G2 PHINT; species DSGT DGET SWCG; ecotype TBASE TOPT ROPT P2O DJTI GDDE DSGFT;
    #     the output unit NOUTDO (GETLUN in RUNINIT, WRITE on crop failure).
    #   state-machine memory: NDAS (days in stage), L0 (seed layer, stage 7 -> 8), P9 (stage 8 -> 9),
    #     SIND (photoperiod induction, stage 2), SUMDTT_2 (stage 2 -> 3), IDURP (stage 4 counter),
    #     DUMMY (counter used to set PDTT once).
    #   ECONO is read from the cultivar line and used in the same RUNINIT call to find the ecotype;
    #     flagged only because the READ sits in the ELSE of "section not found" (whose branch
    #     calls ERROR, which stops the run) -> conservative.
    true_state = set(
        "P1 P2 P5 G2 PHINT DSGT DGET SWCG TBASE TOPT ROPT P2O DJTI GDDE DSGFT NOUTDO "
        "NDAS L0 P9 SIND SUMDTT_2 IDURP DUMMY".split()
    )
    assert _state(s) == true_state | {"ECONO"}
    assert {v["kind"] for v in s["saved_vars"]} == {"save_all"}


# ---------------------------------------------------------------------------
# Whole trees (slow): pin the disagreements found outside the indexed subset
# ---------------------------------------------------------------------------
# RZWQM2 src/RZWQM/*.for. fparser cannot parse two files at all (reported by index_tree):
#   Rzmain.for  text in the sequence field (columns 73-80) -- ifort ignores it, fparser does not;
#   Rznutr.for  "STOP3" (blank-insensitive fixed form for STOP 3).
# Remaining disagreements, all index-side:
#   save:*      SAVEd names missing from the index -- it only lists SAVEd names that are locals,
#               and a name that is undeclared under a wildcard USE (MAQUE, MASSBL, PHYSCL: USE
#               VARIABLE) or never assigned (QUICKTURF STARTDAY) is not a local for it.
#   edge:ENVSTR zero-argument function references EPHOP() / EHERBP() are not seen by the index.
RZ_PARSE_FAILURES = {"Rzmain.for", "Rznutr.for"}
RZ_KNOWN = {
    ("save", "MAQUE"),
    ("save", "MASSBL"),
    ("save", "PHYSCL"),
    ("save", "QUICKTURF"),
    ("edge", "ENVSTR"),
}
# DSSAT source/Plant: the CERES family plus NWHEAT / NTEF (a full Plant/*/*.for scan, 290 files,
# was run once during development: 14 files fparser cannot parse -- PAUSE, "TYPE (T)NAME"
# without a blank, a blank inside a name -- and no disagreement beyond the one below; it takes
# ~40 s, so the test keeps this subset). The one disagreement: NWHEATS_RTLV receives
# NWHEATS_LEVEL as a dummy procedure; the index draws an edge to the global function of that
# name, the scanner does not (scope difference, conservative for ordering).
DSSAT_PLANT_DIRS = ("CERES-*", "NWHEAT", "NTEF")
DSSAT_PLANT_KNOWN = {("edge", "NWHEATS_RTLV")}


def _whole_tree(files: list[Path], line_length: int | None):
    from agrijax.port.fortran_index import index_tree

    idx = index_tree(files)
    bad = {e["file"] for e in idx.errors}
    index = idx.to_dict()
    units = scan_files([f for f in map(str, files) if f not in bad], line_length=line_length)
    return idx, index, compare(index, units)


@pytest.mark.slow
def test_rzwqm_whole_tree(data_dir: Path) -> None:
    src = data_dir / RZ_SRC
    files = sorted(src.glob("*.for"))
    if not files:
        pytest.skip(f"no RZWQM sources under {src}")
    idx, index, diffs = _whole_tree(files, 72)
    assert {Path(e["file"]).name for e in idx.errors} == RZ_PARSE_FAILURES
    assert {(d.what, d.routine) for d in diffs} == RZ_KNOWN, "\n".join(map(str, diffs))
    # the save disagreements are only ever names missing on the index side
    for d in diffs:
        if d.what == "save":
            left, right = d.detail.split(" vs scanner ")
            assert set(ast.literal_eval(left.removeprefix("index "))) < set(ast.literal_eval(right))
    assert check_topological_order(index["call_graph"], index["topological_order"]) == []


@pytest.mark.slow
def test_dssat_plant_whole_tree() -> None:
    plant = DSSAT_ENGINE / "source" / "Plant"
    files = sorted(f for d in DSSAT_PLANT_DIRS for f in plant.glob(f"{d}/*.for"))
    if not files:
        pytest.skip(f"no DSSAT Plant sources under {plant}")
    idx, index, diffs = _whole_tree(files, None)
    assert len(idx.subroutines) > 150
    assert {Path(e["file"]).name for e in idx.errors} == {"CER_Output.for"}  # PAUSE statement
    assert {(d.what, d.routine) for d in diffs} == DSSAT_PLANT_KNOWN, "\n".join(map(str, diffs))
    assert check_topological_order(index["call_graph"], index["topological_order"]) == []
