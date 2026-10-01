"""Tests for the Fortran dump instrumenter (agrijax.port.instrument).

Pure-Python tests cover the statement scanner and the rewrite rules. The end-to-end tests compile
small self-written Fortran programs with gfortran (plain and instrumented), check that the
instrumented program prints exactly what the plain one prints, and read the dumps back with
agrijax.port.dumps. They skip when gfortran is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from agrijax.port import dumps
from agrijax.port import instrument as ins

GFORTRAN = shutil.which("gfortran")


# ---------------------------------------------------------------------------
# scanner and classification
# ---------------------------------------------------------------------------
FIXED = """\
C comment line
      SUBROUTINE FOO(A, B,
     &               N)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION A(N), W(3)
      CHARACTER*8 TAG
      SAVE W
      DATA TAG /'x!y'/
      SQ(X) = X*X
   10 A(1) = SQ(B) ! comment with RETURN
      IF (B .GT. 0.0D0) RETURN
      B = 1; W(1) = B
  20  RETURN
      END
"""


def test_scan_fixed_form_continuation_comments_and_semicolons() -> None:
    st = ins.scan_statements(FIXED.splitlines(keepends=True))
    keys = [s.key for s in st]
    assert keys[0] == "SUBROUTINEFOO(A,B,N)"
    assert "DATATAG/'x!y'/" in keys  # '!' inside a string is not a comment
    assert "A(1)=SQ(B)" in keys  # inline comment stripped
    assert "B=1" in keys and "W(1)=B" in keys  # ';' splits statements
    lab = [s for s in st if s.label == "10"]
    assert len(lab) == 1 and lab[0].key == "A(1)=SQ(B)"


def test_find_unit_first_executable_skips_data_and_statement_functions() -> None:
    st = ins.scan_statements(FIXED.splitlines(keepends=True))
    u = ins.find_unit(st, "FOO")
    assert u.kind == "SUBROUTINE" and u.dummies == ["A", "B", "N"]
    assert u.first_exec is not None and u.first_exec.key == "A(1)=SQ(B)"
    assert [r.key for r in u.returns] == ["IF(B.GT.0.0D0)RETURN", "RETURN"]
    assert u.decls["A"].dims == ["N"] and u.decls["W"].dims == ["3"]
    assert 10 in u.labels and 20 in u.labels


def test_tab_format_lines() -> None:
    src = "      SUBROUTINE T(X)\n\tREAL X\n\tX = 1.0 +\n\t12.0\n\tRETURN\n      END\n"
    st = ins.scan_statements(src.splitlines(keepends=True))
    assert [s.key for s in st] == ["SUBROUTINET(X)", "REALX", "X=1.0+2.0", "RETURN", "END"]


def test_columns_beyond_line_length_are_ignored() -> None:
    src = "      X = 1" + " " * 61 + "SEQ00010\n"
    assert ins.scan_statements([src], line_length=72)[0].key == "X=1"
    assert ins.scan_statements([src], line_length=None)[0].key == "X=1SEQ00010"


def test_free_form_scan() -> None:
    src = "subroutine g(x, &\n   & n) ! hi\n  integer :: n\n  real :: x(n)\n  x = 1.0; return\nend subroutine g\n"
    st = ins.scan_statements(src.splitlines(keepends=True), free=True)
    assert [s.key for s in st] == [
        "SUBROUTINEG(X,N)",
        "INTEGER::N",
        "REAL::X(N)",
        "X=1.0",
        "RETURN",
        "ENDSUBROUTINEG",
    ]


def test_interface_body_is_not_the_unit() -> None:
    src = (
        "module m\n interface\n  subroutine s(a)\n   real a\n  end subroutine\n end interface\nend module\n"
        "subroutine s(a)\n real a\n a = 2.0\nend subroutine s\n"
    )
    st = ins.scan_statements(src.splitlines(keepends=True), free=True)
    u = ins.find_unit(st, "S")
    assert u.header.first_line == 7
    assert u.first_exec is not None and u.first_exec.key == "A=2.0"


# ---------------------------------------------------------------------------
# rewriting
# ---------------------------------------------------------------------------
def _entry(name: str, args: list[str], **kw: object) -> dict[str, object]:
    return {"name": name, "args": args, **kw}


def test_rewrite_returns_and_labelled_end() -> None:
    src = (
        "      SUBROUTINE R(X, K)\n"
        "      IF (K .EQ. 1) GOTO 30\n"
        "      IF (K .EQ. 2) RETURN\n"
        "      X = X + 1.0\n"
        "   30 END\n"
    )
    new, res = ins.instrument_routine(src, _entry("R", ["X", "K"]), 1)
    lines = new.splitlines()
    assert "      USE AJDUMP" in lines[1]
    assert "      IF (K .EQ. 2) GOTO 99971" in lines
    # the END label moved to a CONTINUE in front of the exit block
    i30 = lines.index("30    CONTINUE")
    assert lines[i30 + 1] == "99971 CONTINUE"
    assert lines[-1].strip() == "END" and not lines[-1].lstrip().startswith("30")
    assert res.n_returns == 1 and res.exit_label == 99971


def test_exit_label_avoids_existing_labels() -> None:
    src = "      SUBROUTINE R(X)\n99971 X = 1.0\n      RETURN\n      END\n"
    _, res = ins.instrument_routine(src, _entry("R", ["X"]), 1)
    assert res.exit_label == 99972


def test_long_return_line_is_continued_within_72_columns() -> None:
    cond = "IF (AVERYLONGVARIABLENAME .GT. ANOTHERVERYLONGNAME + 1.0D0) RETURN"
    src = f"      SUBROUTINE R(X)\n      {cond}\n      END\n"
    assert len(f"      {cond}") <= 72
    new, _ = ins.instrument_routine(src, _entry("R", ["X"]), 1)
    for ln in new.splitlines():
        if ln[:1] not in "Cc*!":
            assert len(ln) <= 72, ln
    st = ins.scan_statements(new.splitlines(keepends=True))
    assert any(s.key == "IF(AVERYLONGVARIABLENAME.GT.ANOTHERVERYLONGNAME+1.0D0)GOTO99971" for s in st)


def test_generated_lines_wrap_at_72_columns() -> None:
    long = "A" * 30
    src = f"      SUBROUTINE R({long}B)\n      {long}B = 1.0\n      END\n"
    new, _ = ins.instrument_routine(src, _entry("R", [f"{long}B"]), 1)
    assert all(len(ln) <= 72 for ln in new.splitlines())
    st = ins.scan_statements(new.splitlines(keepends=True))
    assert sum(s.key == f"CALLAJD_PUT('{long}B',{long}B)" for s in st) == 2


def test_plan_variables_kinds_extents_and_skips() -> None:
    src = (
        "      SUBROUTINE P(X, Y, F, N)\n"
        "      DIMENSION X(*), Y(13,*)\n"
        "      EXTERNAL F\n"
        "      COMMON /C/ CA, CB\n"
        "      X(1) = CA\n"
        "      END\n"
    )
    e = _entry(
        "P",
        ["X", "Y", "F", "N"],
        saved_vars=[{"name": "S1", "kind": "implicit_save"}],
        common_blocks=[{"name": "C", "vars": ["CA", "CB"]}],
        read_vars=["CA"],
        assigned_vars=["X"],
        intent_guess={"X": "out"},
    )
    st = ins.scan_statements(src.splitlines(keepends=True))
    u = ins.find_unit(st, "P")
    specs, notes = ins.plan_variables(e, u, ins.RoutineRequest("P", extents={"X": "N"}))
    by = {v.name: v for v in specs}
    assert by["X"].expr == "X(1:N)" and by["X"].intent == "out"
    assert "Y" not in by and any("Y: assumed-size" in n for n in notes)
    assert "F" not in by and any("F: dummy procedure" in n for n in notes)
    assert by["S1"].kind == "implicit_save"
    assert by["CA"].kind == "common" and by["CA"].block == "C"
    assert "CB" not in by  # COMMON member neither read nor written


def test_only_and_extra_narrow_and_widen_the_plan() -> None:
    src = """      SUBROUTINE STEP(X, N, IDAY, S)
      DIMENSION X(N)
      COMMON /BLK/ CSUM, NCALL
      S = X(1)
      END
"""
    lines = src.splitlines(keepends=True)
    unit = ins.find_unit(ins.scan_statements(lines), "STEP")
    req = ins.RoutineRequest("STEP", only=["s", "CSUM", "NOPE"], extra={"x1": "X(1)"})
    specs, notes = ins.plan_variables(STEP_ENTRY, unit, req)
    assert [(v.name, v.expr, v.kind) for v in specs] == [
        ("S", "S", "arg"),
        ("CSUM", "CSUM", "common"),
        ("X1", "X(1)", "extra"),
    ]
    assert notes == ["NOPE: requested by 'only' but not in the plan"]
    with pytest.raises(ins.InstrumentError, match="already dumped"):
        ins.plan_variables(STEP_ENTRY, unit, ins.RoutineRequest("STEP", extra={"S": "S"}))
    specs, _ = ins.plan_variables(
        STEP_ENTRY, unit, ins.RoutineRequest("STEP", skip=["X"], extra={"X": "X(1:N)"})
    )
    assert [v.name for v in specs if v.kind == "extra"] == ["X"]


def test_cli_only_skip_extra(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "toy.f").write_text(TOY)
    idx = {"files": [str(src / "toy.f")], "subroutines": [{**STEP_ENTRY, "file": str(src / "toy.f")}]}
    (tmp_path / "idx.json").write_text(json.dumps(idx))
    out = tmp_path / "out"
    rc = ins.main(
        [
            "tree",
            "--index",
            str(tmp_path / "idx.json"),
            "--src",
            str(src),
            "--out",
            str(out),
            "--index-root",
            str(src),
            "--routine",
            "STEP@IDAY",
            "--only",
            "STEP:X,S,ACC",
            "--skip",
            "STEP:ACC",
            "--extra",
            "STEP:XSUM=SUM(X(1:N))",
        ]
    )
    assert rc == 0
    man = json.loads((out / "_patches" / "manifest.json").read_text())
    assert [v["name"] for v in man["routines"][0]["variables"]] == ["X", "S", "XSUM"]
    assert man["routines"][0]["request"] == {
        "date_expr": "IDAY",
        "only": ["X", "S", "ACC"],
        "skip": ["ACC"],
        "extra": {"XSUM": "SUM(X(1:N))"},
    }
    assert "CALL AJD_PUT('XSUM',SUM(X(1:N)))" in (out / "toy.f").read_text()


def test_derived_type_components_are_expanded() -> None:
    types = ins.parse_type_definitions(
        [
            "      TYPE Inner\n        REAL A(3)\n        INTEGER, POINTER :: P\n      END TYPE Inner\n"
            "      TYPE Outer\n        SEQUENCE\n        CHARACTER (len=2) C\n        TYPE (Inner) IN\n"
            "      END TYPE Outer\n"
        ],
        line_length=None,
    )
    assert [c.name for c in types["OUTER"]] == ["C", "IN"]
    src = "      SUBROUTINE D(T)\n      USE M\n      TYPE (Outer) T\n      T%C = 'ab'\n      END\n"
    st = ins.scan_statements(src.splitlines(keepends=True))
    u = ins.find_unit(st, "D")
    specs, notes = ins.plan_variables(_entry("D", ["T"]), u, ins.RoutineRequest("D"), types)
    assert [v.name for v in specs] == ["T%C", "T%IN%A"]
    assert any("T%IN%P" in n for n in notes)


def test_entry_and_alternate_return_are_rejected() -> None:
    src = "      SUBROUTINE E(X)\n      X = 1\n      ENTRY E2(X)\n      END\n"
    with pytest.raises(ins.InstrumentError, match="ENTRY"):
        ins.instrument_routine(src, _entry("E", ["X"]), 1)
    src = "      SUBROUTINE E(X, *)\n      X = 1\n      RETURN 1\n      END\n"
    with pytest.raises(ins.InstrumentError, match="alternate RETURN"):
        ins.instrument_routine(src, _entry("E", ["X", "*"]), 1)


def test_missing_unit_and_bad_id() -> None:
    with pytest.raises(ins.InstrumentError, match="not found"):
        ins.instrument_routine("      END\n", _entry("NOPE", []), 1)
    with pytest.raises(ins.InstrumentError, match="routine id"):
        ins.instrument_routine("      SUBROUTINE A\n      END\n", _entry("A", []), 0)


def test_date_hook_inserted_before_first_statement() -> None:
    src = "      SUBROUTINE H(IY, JD)\n      INTEGER IY, JD\n      JD = JD + 1\n      END\n"
    new = ins.insert_date_hook(src, "H", "IY*1000+JD")
    lines = new.splitlines()
    assert lines[1] == "      USE AJDUMP"
    assert lines.index("      CALL AJD_SETDATE(INT(IY*1000+JD))") < lines.index("      JD = JD + 1")


def test_module_source_covers_types_and_ranks() -> None:
    src = ins.ajdump_module_source()
    for tag in ("r8", "r4", "i4", "i8", "l4", "ch"):
        for r in range(ins._MAX_RANK + 1):
            assert f"module procedure ajd_put_{tag}_{r}\n" in src
    assert "module ajdump" in src and "end module ajdump" in src


def test_select_dates_is_deterministic_and_grouped() -> None:
    dates = list(range(2015001, 2015061))
    rain = [10.0 if d % 7 == 0 else 0.0 for d in dates]
    season = [d > 2015030 for d in dates]
    a = ins.select_dates(dates, rain, season=season, n_rain=3, n_dry=3, n_season=4)
    b = ins.select_dates(dates, rain, season=season, n_rain=3, n_dry=3, n_season=4)
    assert a == b
    assert len(a["rain"]) == 3 and all(rain[dates.index(d)] >= 5 for d in a["rain"])
    assert all(rain[dates.index(d)] == 0.0 for d in a["dry"])
    assert all(d > 2015030 for d in a["season"])
    assert not set(a["season"]) & (set(a["rain"]) | set(a["dry"]))
    with pytest.raises(ValueError):
        ins.select_dates(dates, rain[:-1])


def test_write_config(tmp_path: Path) -> None:
    p = ins.write_config(
        tmp_path / "ajdump.cfg",
        dates=[3, 1, 2, 2],
        routines={"step": (2, 1, 10), "evntro": (5, 1, 100, True), "f2": (1, 1, 1, False)},
        default=(0, 1, 0),
    )
    txt = p.read_text().splitlines()
    assert "ROUTINE STEP 2 1 10" in txt and "DEFAULT 0 1 0" in txt
    assert "ROUTINE EVNTRO 5 1 100 1" in txt and "ROUTINE F2 1 1 1" in txt
    assert txt[txt.index("DATES 3") + 1] == "1 2 3"


# ---------------------------------------------------------------------------
# end to end with gfortran
# ---------------------------------------------------------------------------
TOY = """\
      PROGRAM MAIN
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION X(4)
      COMMON /BLK/ CSUM, NCALL
      DO 20 IDAY = 1, 6
        DO 10 I = 1, 4
          X(I) = DBLE(I*IDAY) / 3.0D0
   10   CONTINUE
        DO 15 K = 1, 3
          CALL STEP(X, 4, IDAY, S)
   15   CONTINUE
        Y = F2(S)
        WRITE(*,'(I3,3ES24.16)') IDAY, S, Y, CSUM
   20 CONTINUE
      END
C     SAVE state, an implicitly static local (-fno-automatic), COMMON, early RETURN
      SUBROUTINE STEP(X, N, IDAY, S)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION X(N)
      LOGICAL FIRST
      CHARACTER*4 TAG
      COMMON /BLK/ CSUM, NCALL
      SAVE FIRST, ACC
      DATA FIRST /.TRUE./
      IF (FIRST) THEN
        ACC = 0.0D0
        FIRST = .FALSE.
      ENDIF
      S = 0.0D0
      DO 10 I = 1, N
        S = S + X(I)
   10 CONTINUE
      ACC = ACC + S
      CSUM = ACC
      NCALL = NCALL + 1
      TAG = 'DAY'
      IF (IDAY .EQ. 3) RETURN
      X(1) = -X(1)
      PREV = S
      RETURN
      END
      DOUBLE PRECISION FUNCTION F2(S)
      DOUBLE PRECISION S
      F2 = S*S
      END
"""

STEP_ENTRY = {
    "name": "STEP",
    "args": ["X", "N", "IDAY", "S"],
    "saved_vars": [
        {"name": "FIRST", "kind": "save"},
        {"name": "ACC", "kind": "save"},
        {"name": "PREV", "kind": "implicit_save"},
        {"name": "TAG", "kind": "implicit_save"},
    ],
    "common_blocks": [{"name": "BLK", "vars": ["CSUM", "NCALL"]}],
    "read_vars": ["CSUM", "NCALL"],
    "assigned_vars": ["CSUM", "NCALL"],
    "intent_guess": {"X": "inout", "N": "in", "IDAY": "in", "S": "out"},
}


def _build(tmp: Path, name: str, sources: list[Path], flags: tuple[str, ...] = ()) -> Path:
    exe = tmp / name
    subprocess.run(
        [GFORTRAN or "gfortran", "-O2", "-fno-automatic", *flags, "-o", str(exe), *map(str, sources)],
        check=True,
        cwd=tmp,
        capture_output=True,
        text=True,
    )
    return exe


def _run(exe: Path, cwd: Path) -> str:
    return subprocess.run([str(exe)], check=True, cwd=cwd, capture_output=True, text=True).stdout


def _instrument_toy(tmp: Path) -> tuple[Path, Path]:
    plain = tmp / "toy.f"
    plain.write_text(TOY)
    new, res = ins.instrument_routine(
        TOY, STEP_ENTRY, 1, request=ins.RoutineRequest("STEP", date_expr="IDAY")
    )
    new, res2 = ins.instrument_routine(new, {"name": "F2", "args": ["S"]}, 2)
    assert res.notes == [] and res2.variables[-1].kind == "result"
    inst = tmp / "toy_i.f"
    inst.write_text(new)
    mod = tmp / "ajdump.f90"
    mod.write_text(ins.ajdump_module_source())
    return plain, inst


def _reference_calls() -> list[dict[str, object]]:
    """The toy's call sequence recomputed in Python (independent of the Fortran)."""
    calls = []
    acc, ncall, first, prev, tag = 0.0, 0, True, 0.0, "\x00" * 4  # static storage starts zeroed
    s = 0.0
    for iday in range(1, 7):
        x = [i * iday / 3.0 for i in range(1, 5)]
        for _ in range(3):
            entry = {
                "X": list(x),
                "S": s,
                "ACC": acc,
                "NCALL": ncall,
                "FIRST": first,
                "PREV": prev,
                "TAG": tag,
            }
            if first:
                acc, first = 0.0, False
            s = sum(x)
            acc += s
            ncall += 1
            tag = "DAY "
            if iday != 3:
                x[0] = -x[0]
                prev = s
            exit_ = {
                "X": list(x),
                "S": s,
                "ACC": acc,
                "NCALL": ncall,
                "FIRST": first,
                "PREV": prev,
                "TAG": tag,
            }
            calls.append({"date": iday, "entry": entry, "exit": exit_})
    return calls


@pytest.mark.allow_skip(reason="end-to-end instrumentation needs gfortran")
@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not installed")
def test_gfortran_end_to_end_all_calls(tmp_path: Path) -> None:
    plain, inst = _instrument_toy(tmp_path)
    out_plain = _run(_build(tmp_path, "plain", [plain]), tmp_path)
    exe = _build(tmp_path, "inst", [tmp_path / "ajdump.f90", inst])
    ref_dir = tmp_path / "all"
    ref_dir.mkdir()
    (ref_dir / "ajdump.cfg").write_text("DEFAULT 1000 1 1000\n")
    out_inst = _run(exe, ref_dir)
    assert out_inst == out_plain  # instrumentation does not change results

    cases = dumps.pair_records(dumps.read_dump(ref_dir / "ajdump_STEP.bin", strict=True))
    ref = _reference_calls()
    assert len(cases) == len(ref) == 18
    for k, (c, r) in enumerate(zip(cases, ref, strict=True)):
        assert c.routine == "STEP" and c.call == k + 1 and c.date == r["date"]
        assert c.day_call == k % 3 + 1
        for phase, got in (("entry", c.entry), ("exit", c.exit)):
            want = r[phase]
            assert isinstance(want, dict)
            np.testing.assert_array_equal(got["X"], np.array(want["X"]))
            assert got["X"].dtype == np.float64 and got["X"].shape == (4,)
            if k > 0 or phase == "exit":  # entry S of the very first call is an undefined dummy
                assert float(got["S"]) == want["S"]
            assert float(got["ACC"]) == want["ACC"] and float(got["CSUM"]) == want["ACC"]
            assert int(got["NCALL"]) == want["NCALL"] and got["NCALL"].dtype == np.int32
            assert bool(got["FIRST"]) == want["FIRST"] and got["FIRST"].dtype == np.bool_
            assert float(got["PREV"]) == want["PREV"]
            assert got["TAG"].tobytes().decode() == want["TAG"]
            assert int(got["N"]) == 4 and int(got["IDAY"]) == r["date"]
    f2 = dumps.pair_records(dumps.read_dump(ref_dir / "ajdump_F2.bin", strict=True))
    assert len(f2) == 6
    for c in f2:
        assert float(c.exit["F2"]) == float(c.entry["S"]) ** 2
        assert "F2" not in c.entry  # the result is only dumped at exit


@pytest.mark.allow_skip(reason="end-to-end instrumentation needs gfortran")
@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not installed")
def test_gfortran_sampling_config(tmp_path: Path) -> None:
    _, inst = _instrument_toy(tmp_path)
    exe = _build(tmp_path, "inst", [tmp_path / "ajdump.f90", inst])
    run = tmp_path / "sel"
    run.mkdir()
    ins.write_config(run / "ajdump.cfg", dates=[2, 5], routines={"STEP": (1, 2, 10)}, default=(0, 1, 0))
    _run(exe, run)
    cases = dumps.pair_records(dumps.read_dump(run / "ajdump_STEP.bin", strict=True))
    # dates 2 and 5 only, first call of each (every=2 allows calls 1 and 3, per_day=1 keeps call 1)
    assert [(c.date, c.day_call, c.call) for c in cases] == [(2, 1, 4), (5, 1, 13)]
    assert dumps.read_dump(run / "ajdump_F2.bin") == []  # default max_total 0

    run2 = tmp_path / "every"
    run2.mkdir()
    ins.write_config(run2 / "ajdump.cfg", routines={"STEP": (5, 2, 3)})
    _run(exe, run2)
    cases = dumps.pair_records(dumps.read_dump(run2 / "ajdump_STEP.bin", strict=True))
    assert [(c.date, c.day_call) for c in cases] == [(1, 1), (1, 3), (2, 1)]  # max_total 3


FREE = """\
module toytypes
  implicit none
  type state
    real(8) :: w(3)
    integer :: n
    logical :: wet
  end type state
end module toytypes

subroutine upd(st, rain, k)
  use toytypes
  implicit none
  type(state) :: st
  real, intent(in) :: rain
  integer, intent(in) :: k
  integer, save :: nwet = 0
  st%n = st%n + 1
  if (rain <= 0.0) then
    st%wet = .false.
    return
  end if
  st%w(k) = st%w(k) + rain
  st%wet = .true.
  nwet = nwet + 1
end subroutine upd

program p
  use toytypes
  implicit none
  type(state) :: s
  integer :: i
  s%w = 0d0; s%n = 0; s%wet = .false.
  do i = 1, 5
    call upd(s, real(mod(i, 2)) * 1.5, mod(i, 3) + 1)
  end do
  print '(3es24.16, i4, l2)', s%w, s%n, s%wet
end program p
"""


@pytest.mark.allow_skip(reason="end-to-end instrumentation needs gfortran")
@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not installed")
def test_gfortran_free_form_derived_type(tmp_path: Path) -> None:
    plain = tmp_path / "free.f90"
    plain.write_text(FREE)
    types = ins.parse_type_definitions([FREE], free=True)
    entry = {"name": "UPD", "args": ["ST", "RAIN", "K"], "saved_vars": [{"name": "NWET", "kind": "save"}]}
    new, res = ins.instrument_routine(FREE, entry, 1, free=True, line_length=None, types=types)
    assert [v.name for v in res.variables] == ["ST%W", "ST%N", "ST%WET", "RAIN", "K", "NWET"]
    inst = tmp_path / "free_i.f90"
    inst.write_text(new)
    (tmp_path / "ajdump.f90").write_text(ins.ajdump_module_source())
    out_plain = _run(_build(tmp_path, "plain", [plain]), tmp_path)
    out_inst = _run(_build(tmp_path, "inst", [tmp_path / "ajdump.f90", inst]), tmp_path)
    assert out_inst == out_plain
    cases = dumps.pair_records(dumps.read_dump(tmp_path / "ajdump_UPD.bin", strict=True))
    assert len(cases) == 5
    w = np.zeros(3)
    nwet = 0
    for i, c in enumerate(cases, start=1):
        rain = (i % 2) * 1.5
        np.testing.assert_array_equal(c.entry["ST%W"], w)
        assert c.entry["RAIN"].dtype == np.float32 and float(c.entry["RAIN"]) == rain
        assert int(c.entry["NWET"]) == nwet
        if rain > 0:
            w[i % 3] += rain
            nwet += 1
        np.testing.assert_array_equal(c.exit["ST%W"], w)
        assert int(c.exit["ST%N"]) == i and bool(c.exit["ST%WET"]) == (rain > 0)
        assert int(c.exit["NWET"]) == nwet


@pytest.mark.allow_skip(reason="end-to-end instrumentation needs gfortran")
@pytest.mark.skipif(GFORTRAN is None, reason="gfortran not installed")
def test_gfortran_sequence_all_dates_and_extra(tmp_path: Path) -> None:
    """Version-2 ``seq`` orders records across routines; the all-dates flag bypasses ``DATES``;
    an ``extra`` expression is dumped at entry and exit."""
    plain = tmp_path / "toy.f"
    plain.write_text(TOY)
    req = ins.RoutineRequest("STEP", date_expr="IDAY", only=["X", "S"], extra={"XSUM": "SUM(X(1:N))"})
    new, res = ins.instrument_routine(TOY, STEP_ENTRY, 1, request=req)
    assert [v.name for v in res.variables] == ["X", "S", "XSUM"]
    new, _ = ins.instrument_routine(new, {"name": "F2", "args": ["S"]}, 2)
    inst = tmp_path / "toy_i.f"
    inst.write_text(new)
    (tmp_path / "ajdump.f90").write_text(ins.ajdump_module_source())
    exe = _build(tmp_path, "inst", [tmp_path / "ajdump.f90", inst])
    assert _run(exe, tmp_path) == _run(_build(tmp_path, "plain", [plain]), tmp_path)
    run = tmp_path / "r"
    run.mkdir()
    ins.write_config(run / "ajdump.cfg", dates=[2], routines={"STEP": (1, 1, 100), "F2": (1, 1, 100, True)})
    _run(exe, run)
    step = dumps.pair_records(dumps.read_dump(run / "ajdump_STEP.bin", strict=True))
    f2 = dumps.pair_records(dumps.read_dump(run / "ajdump_F2.bin", strict=True))
    assert [c.date for c in step] == [2]  # DATES applies to STEP
    assert [c.date for c in f2] == [1, 2, 3, 4, 5, 6]  # F2 is dumped on every date
    for c in step:
        assert float(c.entry["XSUM"]) == float(np.sum(c.entry["X"]))
        assert float(c.exit["XSUM"]) == float(np.sum(c.exit["X"]))
        assert set(c.entry) == {"X", "S", "XSUM"}
    # F2 of day d is called after the three STEP calls of day d: the merged order shows it
    order = dumps.event_order([run / "ajdump_STEP.bin", run / "ajdump_F2.bin"])
    assert list(order["seq"]) == list(range(1, len(order) + 1))
    seq = [(str(r["routine"]), int(r["phase"]), int(r["date"])) for r in order]
    assert seq[:4] == [("F2", 0, 1), ("F2", 1, 1), ("STEP", 0, 2), ("STEP", 1, 2)]
    assert seq[4:6] == [("F2", 0, 2), ("F2", 1, 2)]
