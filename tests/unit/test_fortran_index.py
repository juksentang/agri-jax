"""Unit tests for agrijax.port.fortran_index on small embedded Fortran fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fparser")

from agrijax.port.fortran_index import index_file, index_tree, main, topological_order

# Fixed-form F77 fixture. Columns matter: statements start in column 7.
FIXED = """\
C     Toy soil routine with every hidden-state pattern the indexer must see.
      SUBROUTINE STEP(DT, N, H, Q, TOT)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (MX=10)
      DIMENSION H(N), Q(N), W(MX), TMP(MX)
      LOGICAL FIRST
      CHARACTER*40 MSG
      COMMON /SOILP/ THS(MX), NLAY
      SAVE FIRST, ACC
      DATA FIRST /.TRUE./
      DATA KOUNT /0/
C     classic first-call initialisation: ACC and OLD carry state
      IF (FIRST) THEN
        ACC = 0.0D0
        OLD = 0.0D0
        FIRST = .FALSE.
      ENDIF
C     both branches assign BOTH: not state
      IF (DT .GT. 1.0D0) THEN
        BOTH = 1.0D0
      ELSE
        BOTH = 2.0D0
      ENDIF
C     only one branch assigns PART, read after: state
      IF (DT .GT. 5.0D0) PART = 3.0D0
      KOUNT = KOUNT + 1
C     FILL writes W (out): W is not state; UNKNWN is not indexed
      CALL FILL(N, W)
      CALL UNKNWN(LUN)
      DO 10 I = 1, N
        TMP(I) = W(I) * DT
        Q(I) = TMP(I) + BOTH + PART + THS(I)
        H(I) = H(I) + Q(I)
   10 CONTINUE
      ACC = ACC + DT
      TOT = ACC + OLD + FSQ(DT) + DBLE(LUN) + DBLE(KOUNT)
      OLD = TOT
      WRITE(MSG, '(A)') 'done'
      WRITE(LUN, *) MSG
      RETURN
      END

      SUBROUTINE FILL(N, W)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION W(N)
      DO 20 I = 1, N
        W(I) = 1.0D0
   20 CONTINUE
      RETURN
      END

      DOUBLE PRECISION FUNCTION FSQ(X)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      FSQ = X * X
      RETURN
      END
"""

CYCLE = """\
      SUBROUTINE PING(K)
      INTEGER K
      IF (K .GT. 0) CALL PONG(K)
      END

      SUBROUTINE PONG(K)
      INTEGER K
      K = K - 1
      CALL PING(K)
      END

      SUBROUTINE TOP(K)
      INTEGER K
      CALL PING(K)
      CALL LEAF(K)
      END

      SUBROUTINE LEAF(K)
      INTEGER K
      K = 1
      END
"""

FREE = """\
module toymod
  implicit none
  real :: gstate = 0.0
contains
  subroutine upd(x, y, z)
    use othermod, only: helper
    real, intent(in) :: x
    real, intent(out) :: y
    real :: z
    real :: c
    c = x
    y = c + z
    z = 0.0
  end subroutine upd
end module toymod
"""

BROKEN = """\
      SUBROUTINE BAD(X
      X = = 1
      END
"""


@pytest.fixture()
def fixed_file(tmp_path: Path) -> Path:
    p = tmp_path / "toy.for"
    p.write_text(FIXED)
    return p


def _by_name(subs):
    return {s.name: s for s in subs}


def test_index_file_structure(fixed_file: Path) -> None:
    subs = _by_name(index_file(fixed_file))
    assert set(subs) == {"STEP", "FILL", "FSQ"}
    step = subs["STEP"]
    assert step.kind == "subroutine"
    assert subs["FSQ"].kind == "function"
    assert step.args == ["DT", "N", "H", "Q", "TOT"]
    assert step.line_start == 2
    assert step.line_end == 41
    assert step.calls == ["FILL", "UNKNWN"]
    assert "FSQ" in step.function_refs
    assert "DBLE" not in step.function_refs  # intrinsic
    assert step.common_blocks == [{"name": "SOILP", "vars": ["THS", "NLAY"]}]
    assert "MX" in step.parameters
    assert "MX" not in step.locals
    assert "THS" not in step.locals  # COMMON member


def test_intent_guess_after_tree(fixed_file: Path) -> None:
    idx = index_tree([fixed_file])
    subs = idx.by_name()
    assert subs["STEP"].intent_guess == {"DT": "in", "N": "in", "H": "inout", "Q": "out", "TOT": "out"}
    assert subs["FILL"].intent_guess == {"N": "in", "W": "out"}
    assert subs["FSQ"].intent_guess == {"X": "in"}


def test_saved_vars(fixed_file: Path) -> None:
    idx = index_tree([fixed_file])
    step = idx.by_name()["STEP"]
    kinds = {s.name: s.kind for s in step.saved_vars}
    rba = {s.name for s in step.saved_vars if s.read_before_assigned}
    assert kinds["FIRST"] == "save"
    assert kinds["ACC"] == "save"
    assert kinds["KOUNT"] == "data_init"
    # conditionally initialised then read -> implicit SAVE under -save
    assert set(step.implicit_save) == {"OLD", "PART"}
    assert rba == {"FIRST", "ACC", "KOUNT", "OLD", "PART"}
    # assigned on every path / by an out-argument / by an unknown call / internal WRITE
    for name in ("BOTH", "W", "TMP", "LUN", "MSG", "I"):
        assert name in step.locals
        assert name not in kinds
    first = next(s for s in step.saved_vars if s.name == "OLD")
    assert first.first_read_line == 36


def test_unknown_callee_is_conservative_for_dummies(tmp_path: Path) -> None:
    p = tmp_path / "u.for"
    p.write_text("      SUBROUTINE U(A, B)\n      CALL EXTERN(A)\n      B = 1.0\n      END\n")
    (u,) = index_file(p)
    assert u.intent_guess == {"A": "inout", "B": "out"}
    assert u.unresolved_call_args == ["EXTERN:0:A"]


def test_call_graph_order_and_cycles(tmp_path: Path) -> None:
    p = tmp_path / "cyc.for"
    p.write_text(CYCLE)
    idx = index_tree([p])
    assert idx.cycles == [["PING", "PONG"]]
    order = idx.order
    assert order.index("LEAF") < order.index("TOP")
    assert order.index("PING") < order.index("TOP")
    assert idx.subtree_order("PING") == ["PING", "PONG"]
    assert idx.subtree_cycles("TOP") == [["PING", "PONG"]]
    # PONG decrements K; the intent propagates through the cycle to TOP.
    assert idx.by_name()["TOP"].intent_guess == {"K": "inout"}


def test_topological_order_plain_graph() -> None:
    order, cycles = topological_order({"A": ["B", "C"], "B": ["C"], "C": [], "D": ["D"]})
    assert order.index("C") < order.index("B") < order.index("A")
    assert cycles == [["D"]]


def test_free_form_module(tmp_path: Path) -> None:
    p = tmp_path / "m.f90"
    p.write_text(FREE)
    (upd,) = index_file(p)
    assert upd.name == "UPD"
    assert upd.kind == "module_subroutine"
    assert upd.intent_declared == {"X": "in", "Y": "out"}
    assert upd.intent_guess == {"X": "in", "Y": "out", "Z": "inout"}
    mods = {u["module"]: u["only"] for u in upd.module_uses}
    assert mods == {"TOYMOD": None, "OTHERMOD": ["HELPER"]}
    assert upd.saved_vars == []


def test_parse_failure_is_recorded(tmp_path: Path, fixed_file: Path) -> None:
    bad = tmp_path / "bad.for"
    bad.write_text(BROKEN)
    idx = index_tree([bad, fixed_file])
    assert len(idx.errors) == 1
    assert idx.errors[0]["file"] == str(bad)
    assert {s.name for s in idx.subroutines} == {"STEP", "FILL", "FSQ"}


def test_preprocessor_lines_tolerated(tmp_path: Path) -> None:
    p = tmp_path / "pp.for"
    p.write_text("#ifdef FOO\n      SUBROUTINE PP(X)\n#endif\n      X = 2.0\n      END\n")
    (pp,) = index_file(p)
    assert pp.name == "PP"
    assert pp.intent_guess == {"X": "out"}


def test_cli_writes_json(tmp_path: Path, fixed_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "idx" / "index.json"
    rc = main([str(fixed_file), "-o", str(out), "--root", "STEP"])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["n_subroutines"] == 3
    assert data["errors"] == []
    sub = data["subtrees"]["STEP"]
    assert sub["order"] == ["FILL", "FSQ", "STEP"]
    assert set(sub["hidden_state"]["STEP"]) == {"ACC", "FIRST", "KOUNT", "OLD", "PART"}
    step = next(s for s in data["subroutines"] if s["name"] == "STEP")
    assert set(step["implicit_save"]) == {"OLD", "PART"}
    assert "[STEP] order: FILL FSQ STEP" in capsys.readouterr().out
