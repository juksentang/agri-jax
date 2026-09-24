"""Unit tests for agri_jax.port.fortran_xcheck: the line-based scanner that cross-checks the index.

Two layers of evidence, both on embedded fixtures written for these tests:

1. The scanner itself is pinned against hand-derived expectations (fixed-form columns,
   continuation, comments, literals, headers, CALL forms, SAVE / COMMON syntax).
2. The scanner and the fparser-based index (``fortran_index``) are run on the same fixtures and
   must agree fact by fact (:func:`compare`). The comparator is mutation-tested: every kind of
   perturbation of an index is detected. Behaviours where the two methods are known to differ
   are pinned as ``xfail(strict=True)`` with the reason, so a fix in either tool shows up.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import pytest

from agri_jax.port.fortran_xcheck import (
    check_topological_order,
    compare,
    kahn_is_acyclic,
    logical_statements,
    scan_edges,
    scan_text,
)

# ---------------------------------------------------------------------------
# Fixtures (toy code written for these tests; statements start in column 7)
# ---------------------------------------------------------------------------
FIXED = """\
C     upper-case C comment: CALL NOPE1
c     lower-case c comment: CALL NOPE2
*     star comment: CALL NOPE3
!     bang comment: CALL NOPE4
      SUBROUTINE ALPHA(A, B, *, C)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      COMMON /BLK1/ X1, X2(3), /BLK2/ Y1
      COMMON Z1, Z2
      DIMENSION ARR(5)
      DOUBLE PRECISION S2
      SAVE S1, S2, /BLK1/
      DATA S1 /0.0D0/, S2 /1.0D0/
      S1 = S1 + A
      WRITE(*,*) 'CALL NOPE5(X) ! not a comment either'
      WRITE(*,*) 'A literal that
     +continues CALL NOPE6'
      IF (A .GT. 0.0D0) CALL BETA(A,
     +    B)
      IF (B .GT. 0.0D0) THEN
         CALL GAMMA
      ENDIF
      CALLED = 1.0D0
      ARR(1) = GAMFN(A) + ARR(2)
     &       + S2
         ! a comment-only line between continuation lines
     1       + KAPPA(1, 2)
      C = A + CALLED ! trailing comment CALL NOPE7(X)
   10 CONTINUE
      IF (C .LT. 0.0D0) RETURN 1
      RETURN
      END

      DOUBLE PRECISION FUNCTION GAMFN(X)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      GAMFN = X * 2.0D0
      RETURN
      END

      INTEGER FUNCTION KAPPA(N, M)
      KAPPA = N + M + LAMBDA(N)
      END

      FUNCTION LAMBDA(N)
      LAMBDA = N
      END

      SUBROUTINE BETA(P, Q)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
C     LAMBDA is a local array here, not the function
      DIMENSION LAMBDA(3)
      LAMBDA(1) = P
      Q = LAMBDA(1) + GAMFN(P)
      CALL EXTERN(Q)
      END

      SUBROUTINE GAMMA
      SAVE
      INTEGER K
      K = K + 1
      END
"""

FREE = """\
module kinds_mod
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  type :: pt
     real(dp) :: x
  end type pt
  interface
     subroutine ext_sub(a)
       real :: a
     end subroutine ext_sub
  end interface
contains
  subroutine mod_sub(a, b)
    real(dp), intent(in) :: a
    real(dp), intent(out) :: b
    real(dp), save :: acc = 0.0_dp
    acc = acc + a   ! 'call nope(x)' in a comment
    b = acc + helper(a)
    call inner(b)
    if (b > 1.0_dp) call ext_sub(real(b))
  contains
    subroutine inner(z)
      real(dp), intent(inout) :: z
      z = z * 0.5_dp
    end subroutine inner
  end subroutine mod_sub

  real(dp) function helper(x) result(y)
    real(dp), intent(in) :: x
    y = x + &
        & 1.0_dp
  end function helper
end module kinds_mod
"""


def _units(text: str, **kw):
    return {u.name: u for u in scan_text(text, **kw)}


# ---------------------------------------------------------------------------
# Logical statements
# ---------------------------------------------------------------------------
def test_fixed_form_comments_are_dropped() -> None:
    texts = [s.text for s in logical_statements(FIXED)]
    assert not any("NOPE" in t.replace("'?'", "") for t in texts), texts


def test_fixed_form_continuation_and_comment_line_between() -> None:
    stmts = {s.line: s.text for s in logical_statements(FIXED)}
    # the three physical lines plus a comment-only line in between form one statement
    arr = next(t for t in stmts.values() if t.startswith("ARR(1)"))
    assert arr == "ARR(1) = GAMFN(A) + ARR(2) + S2 + KAPPA(1, 2)"
    call = next(t for t in stmts.values() if "CALL BETA" in t)
    assert call == "IF (A .GT. 0.0D0) CALL BETA(A, B)"


def test_literals_are_masked_across_continuation() -> None:
    stmts = [s.text for s in logical_statements(FIXED)]
    assert "WRITE(*,*) '?'" in stmts  # '!' inside a literal is not a comment
    assert stmts.count("WRITE(*,*) '?'") == 2  # the literal continued onto the next line


def test_labels_and_line_numbers() -> None:
    stmts = logical_statements(FIXED)
    cont = next(s for s in stmts if s.text == "CONTINUE")
    assert cont.label == "10"
    head = next(s for s in stmts if s.text.startswith("SUBROUTINE ALPHA"))
    assert head.line == 5


def test_column_72_truncation_and_sequence_field() -> None:
    line = "      X = A + B" + " " * 57 + "SEQ00010"
    assert len(line) == 80
    assert [s.text for s in logical_statements(line)] == ["X = A + B"]
    assert [s.text for s in logical_statements(line, line_length=None)] == ["X = A + B SEQ00010"]


def test_col6_zero_is_not_continuation_and_debug_lines() -> None:
    src = "      X = 1\n     0Y = 2\nD     PRINT *, X\n     *Z = 3\n"
    # '0' in column 6 starts a new statement; any other character continues; D = debug comment
    assert [s.text for s in logical_statements(src)] == ["X = 1", "Y = 2Z = 3"]


def test_tab_format() -> None:
    src = "\tX = 1 +\n\t1 2\n10\tY = X\n"
    stmts = logical_statements(src)
    assert [(s.label, s.text) for s in stmts] == [("", "X = 1 + 2"), ("10", "Y = X")]


def test_preprocessor_and_semicolons() -> None:
    src = "#ifdef FOO\n      A = 1; B = 2\n#endif\n"
    assert [s.text for s in logical_statements(src)] == ["A = 1", "B = 2"]


def test_free_form_continuation() -> None:
    stmts = [s.text for s in logical_statements(FREE, free=True)]
    assert "Y = X + 1.0_DP" in stmts
    assert not any("NOPE" in t for t in stmts)


# ---------------------------------------------------------------------------
# Units, arguments, calls, SAVE, COMMON
# ---------------------------------------------------------------------------
def test_scan_fixed_units_and_arguments() -> None:
    u = _units(FIXED)
    assert list(u) == ["ALPHA", "GAMFN", "KAPPA", "LAMBDA", "BETA", "GAMMA"]
    assert u["ALPHA"].args == ["A", "B", "C"]  # alternate return '*' dropped
    assert u["GAMFN"].kind == "function" and u["GAMFN"].args == ["X"]
    assert u["KAPPA"].kind == "function" and u["KAPPA"].args == ["N", "M"]
    assert u["GAMMA"].args == [] and u["GAMMA"].kind == "subroutine"


def test_scan_fixed_calls_and_edges() -> None:
    u = _units(FIXED)
    # CALLED = ... is an assignment, the literal and comment CALLs are masked
    assert u["ALPHA"].calls == ["BETA", "GAMMA"]
    assert u["BETA"].calls == ["EXTERN"]
    g = scan_edges(list(u.values()))
    assert g["ALPHA"] == ["BETA", "GAMFN", "GAMMA", "KAPPA"]
    assert g["KAPPA"] == ["LAMBDA"]
    assert g["BETA"] == ["GAMFN"]  # LAMBDA is a local array in BETA, not an edge
    assert g["GAMMA"] == [] and g["LAMBDA"] == [] and g["GAMFN"] == []


def test_scan_fixed_save_and_common() -> None:
    u = _units(FIXED)
    a = u["ALPHA"]
    assert a.save_names == ["S1", "S2"]  # /BLK1/ is a common block, not a local
    assert not a.save_all
    assert a.common_blocks == [
        {"name": "BLK1", "vars": ["X1", "X2"]},
        {"name": "BLK2", "vars": ["Y1"]},
        {"name": "", "vars": ["Z1", "Z2"]},
    ]
    assert u["GAMMA"].save_all


def test_scan_free_module_units() -> None:
    u = _units(FREE, free=True)
    assert list(u) == ["MOD_SUB", "INNER", "HELPER"]  # interface body is not a unit
    assert u["MOD_SUB"].kind == "module_subroutine"
    assert u["INNER"].kind == "internal_subroutine"
    assert u["HELPER"].kind == "module_function" and u["HELPER"].args == ["X"]
    assert u["MOD_SUB"].calls == ["EXT_SUB", "INNER"]
    assert u["MOD_SUB"].save_attr_names == ["ACC"]
    assert scan_edges(list(u.values()))["MOD_SUB"] == ["HELPER", "INNER"]


def test_header_variants() -> None:
    src = (
        "      REAL*8 FUNCTION F1(X)\n      F1 = X\n      END\n"
        "      RECURSIVE SUBROUTINE S1(N)\n      END\n"
        "      CHARACTER*(*) FUNCTION F2()\n      F2 = 'A'\n      END\n"
        "      DOUBLEPRECISION FUNCTION F3(Y) \n      F3 = Y\n      END\n"
    )
    u = _units(src)
    assert {n: (v.kind, v.args) for n, v in u.items()} == {
        "F1": ("function", ["X"]),
        "S1": ("subroutine", ["N"]),
        "F2": ("function", []),
        "F3": ("function", ["Y"]),
    }


def test_zero_argument_function_reference_and_statement_function() -> None:
    src = (
        "      SUBROUTINE CALLER(Z)\n"
        "      SQ(T) = T * T\n"
        "      Z = ZEROF() + SQ(Z)\n"
        "      END\n"
        "      FUNCTION ZEROF()\n      ZEROF = 1.0\n      END\n"
        "      FUNCTION SQ(T)\n      SQ = T\n      END\n"
    )
    u = _units(src)
    assert u["CALLER"].stmt_functions == {"SQ"}
    # ZEROF() is a reference; SQ( is the statement function of CALLER, not the external SQ
    assert scan_edges(list(u.values()))["CALLER"] == ["ZEROF"]


def test_dummy_procedure_is_not_an_edge() -> None:
    src = (
        "      SUBROUTINE USEF(F, X, Y)\n      EXTERNAL F\n      Y = F(X)\n      END\n"
        "      FUNCTION F(X)\n      F = X\n      END\n"
    )
    assert scan_edges(list(_units(src).values()))["USEF"] == []


# ---------------------------------------------------------------------------
# Agreement with the fparser index on the same fixtures
# ---------------------------------------------------------------------------
fi = pytest.importorskip("agri_jax.port.fortran_index")
pytest.importorskip("fparser")


def _index(tmp_path: Path, name: str, text: str) -> dict:
    p = tmp_path / name
    p.write_text(text)
    return fi.index_tree([p]).to_dict()


@pytest.mark.parametrize(
    ("name", "text", "free"), [("fixed.for", FIXED, False), ("free.f90", FREE, True)], ids=["fixed", "free"]
)
def test_scanner_agrees_with_index(tmp_path: Path, name: str, text: str, free: bool) -> None:
    index = _index(tmp_path, name, text)
    assert index["errors"] == []
    units = scan_text(text, file=name, free=free)
    assert compare(index, units) == []
    assert check_topological_order(index["call_graph"], index["topological_order"]) == []


# Mutations of a correct index: the comparator must flag each one with the right category.
def _mut_args_order(ix: dict) -> None:
    s = next(s for s in ix["subroutines"] if s["name"] == "KAPPA")
    s["args"] = s["args"][::-1]


def _mut_args_name(ix: dict) -> None:
    s = next(s for s in ix["subroutines"] if s["name"] == "ALPHA")
    s["args"] = ["A", "B", "D"]


def _mut_drop_routine(ix: dict) -> None:
    ix["subroutines"] = [s for s in ix["subroutines"] if s["name"] != "LAMBDA"]


def _mut_drop_edge(ix: dict) -> None:
    ix["call_graph"]["KAPPA"] = []


def _mut_add_edge(ix: dict) -> None:
    ix["call_graph"]["BETA"] = ["GAMFN", "LAMBDA"]


def _mut_calls(ix: dict) -> None:
    s = next(s for s in ix["subroutines"] if s["name"] == "BETA")
    s["calls"] = []


def _mut_common(ix: dict) -> None:
    s = next(s for s in ix["subroutines"] if s["name"] == "ALPHA")
    s["common_blocks"][0]["vars"] = ["X2", "X1"]


def _mut_save(ix: dict) -> None:
    s = next(s for s in ix["subroutines"] if s["name"] == "ALPHA")
    s["saved_vars"] = [v for v in s["saved_vars"] if v["name"] != "S2"]


def _mut_save_all(ix: dict) -> None:
    s = next(s for s in ix["subroutines"] if s["name"] == "GAMMA")
    s["saved_vars"] = []


def _mut_kind(ix: dict) -> None:
    s = next(s for s in ix["subroutines"] if s["name"] == "GAMFN")
    s["kind"] = "subroutine"


@pytest.mark.parametrize(
    ("mutate", "what", "routine"),
    [
        (_mut_args_order, "args", "KAPPA"),
        (_mut_args_name, "args", "ALPHA"),
        (_mut_drop_routine, "routine", "LAMBDA"),
        (_mut_drop_edge, "edge", "KAPPA"),
        (_mut_add_edge, "edge", "BETA"),
        (_mut_calls, "calls", "BETA"),
        (_mut_common, "common", "ALPHA"),
        (_mut_save, "save", "ALPHA"),
        (_mut_save_all, "save", "GAMMA"),
        (_mut_kind, "kind", "GAMFN"),
    ],
)
def test_compare_detects_mutations(tmp_path: Path, mutate, what: str, routine: str) -> None:
    index = _index(tmp_path, "fixed.for", FIXED)
    units = scan_text(FIXED)
    assert compare(index, units) == []
    bad = copy.deepcopy(index)
    mutate(bad)
    diffs = compare(bad, units)
    assert any(d.what == what and d.routine == routine for d in diffs), diffs


# Known differences between the two methods, pinned so that a fix in either one is noticed.
@pytest.mark.xfail(strict=True, reason="fortran_index misses zero-argument function references F()")
def test_index_zero_argument_function_edge(tmp_path: Path) -> None:
    src = "      SUBROUTINE CALLER(Z)\n      Z = ZEROF()\n      END\n      FUNCTION ZEROF()\n      ZEROF = 1.0\n      END\n"
    index = _index(tmp_path, "z.for", src)
    assert index["call_graph"]["CALLER"] == ["ZEROF"]


@pytest.mark.xfail(
    strict=True,
    reason="fortran_index drops SAVEd names that are undeclared when the unit has a wildcard USE",
)
def test_index_save_under_wildcard_use(tmp_path: Path) -> None:
    src = (
        "      MODULE M\n      REAL G\n      END MODULE M\n"
        "      SUBROUTINE S(X)\n      USE M\n      SAVE CNT\n      CNT = CNT + X\n      G = CNT\n      END\n"
    )
    index = _index(tmp_path, "w.for", src)
    s = next(s for s in index["subroutines"] if s["name"] == "S")
    assert [v["name"] for v in s["saved_vars"]] == ["CNT"]


@pytest.mark.xfail(
    strict=True, reason="fortran_index drops SAVEd names that are never assigned and not declared"
)
def test_index_save_of_read_only_name(tmp_path: Path) -> None:
    src = "      SUBROUTINE S(X)\n      SAVE C\n      DATA C /2.0/\n      X = C\n      END\n"
    index = _index(tmp_path, "r.for", src)
    assert [v["name"] for v in index["subroutines"][0]["saved_vars"]] == ["C"]


@pytest.mark.xfail(
    strict=True, reason="fortran_index does not recognise statement functions (they appear as assignments)"
)
def test_index_statement_function(tmp_path: Path) -> None:
    src = "      SUBROUTINE S(X, Y)\n      SQ(T) = T * T\n      Y = SQ(X)\n      END\n"
    index = _index(tmp_path, "sf.for", src)
    s = index["subroutines"][0]
    assert "SQ" not in s["assigned_vars"] and "T" not in s["read_vars"]


@pytest.mark.xfail(
    strict=True, reason="fparser reads fixed-form lines past column 72 (ifort default ignores them)"
)
def test_index_ignores_sequence_field(tmp_path: Path) -> None:
    src = "      SUBROUTINE S(X)" + " " * 51 + "SEQ00010\n      X = 1\n      END\n"
    index = _index(tmp_path, "seq.for", src)
    assert index["errors"] == []


def test_dummy_procedure_edge_difference(tmp_path: Path) -> None:
    """The index draws an edge to a global function that shares its name with a dummy procedure.

    The scanner does not (the callee is whatever the caller passes). Pinned as a documented
    difference in scope, not a bug in either: it is conservative for ordering.
    """
    src = (
        "      SUBROUTINE USEF(F, X, Y)\n      EXTERNAL F\n      Y = F(X)\n      END\n"
        "      FUNCTION F(X)\n      F = X\n      END\n"
    )
    index = _index(tmp_path, "d.for", src)
    diffs = compare(index, scan_text(src))
    assert [str(d) for d in diffs] == ["edge:USEF: index-only edge USEF -> F"]


# ---------------------------------------------------------------------------
# Topological order property
# ---------------------------------------------------------------------------
def test_topological_checker_on_hand_cases() -> None:
    g = {"A": ["B", "C"], "B": ["C"], "C": []}
    assert check_topological_order(g, ["C", "B", "A"]) == []
    assert check_topological_order(g, ["B", "C", "A"]) == ["C (callee) is after B (caller)"]
    assert check_topological_order(g, ["C", "A"])  # not a permutation
    cyc = {"P": ["Q"], "Q": ["P"], "T": ["P"]}
    assert check_topological_order(cyc, ["P", "Q", "T"]) == []  # a cycle cannot be ordered
    assert check_topological_order(cyc, ["T", "P", "Q"]) == ["P (callee) is after T (caller)"]
    assert kahn_is_acyclic(g) and not kahn_is_acyclic(cyc)


@pytest.mark.parametrize("seed", range(20))
def test_index_topological_order_on_random_graphs(seed: int) -> None:
    """fortran_index.topological_order vs the independent checker on random graphs (some cyclic)."""
    rng = random.Random(seed)
    n = rng.randint(1, 30)
    names = [f"R{i}" for i in range(n)]
    p = rng.uniform(0.02, 0.3)
    cyclic = seed % 3 == 0
    g = {
        a: sorted({b for j, b in enumerate(names) if (j > i or cyclic) and rng.random() < p})
        for i, a in enumerate(names)
    }
    order, cycles = fi.topological_order(g)
    assert check_topological_order(g, order) == []
    assert (cycles == []) == kahn_is_acyclic(g)
