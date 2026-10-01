"""The citations of the radiation coefficients point at the right reference source lines.

* DSSAT-CSM (``dssat-4.8.6.0``, BSD-3): every quoted statement is the statement on the cited line
  of ``<AGRI_JAX_DSSAT>/source/<file>`` (whitespace and case aside), inside the cited routine, and
  holds the coefficient's value as a literal.
* RZWQM2 (``rzwqm2-4.6``): the cited ``<file>:<line>`` of the RZWQM2 source tree
  (``<data>/narval_mirror/RZWQM_Linux_Ver45/src``) lies in the cited routine and the line (or its
  fixed-form continuation lines) holds the value as a Fortran literal, up to sign. Nothing of the
  RZWQM2 source is copied into the repository: the check reads the tree at test time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agrijax.core.coefficients import Provenance, iter_coefficients
from agrijax.forcing import radiation as R
from agrijax.port.run_fortran import DEFAULT_DSSAT_ENGINE

pytestmark = pytest.mark.allow_skip(reason="needs the RZWQM2 source tree and the DSSAT-CSM source")

RZ_SOURCE = Path("narval_mirror/RZWQM_Linux_Ver45/src")
DSSAT_SOURCE = Path(os.environ.get("AGRI_JAX_DSSAT", str(DEFAULT_DSSAT_ENGINE))).expanduser() / "source"
_UNIT = re.compile(r"^\s+(?:[A-Z*0-9 ]+\s)?(?:PROGRAM|SUBROUTINE|FUNCTION)\s+(\w+)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])(\d+\.?\d*(?:[DE][-+]?\d+)?|\.\d+(?:[DE][-+]?\d+)?)", re.IGNORECASE)

CITES = [(p, v, f.metadata["provenance"]) for p, f, v in iter_coefficients(R.RZWQM_RADIATION)]


def _lines(path: Path) -> list[str]:
    return path.read_bytes().decode("latin-1").replace("\r", "").split("\n")


def _unit_at(lines: list[str], line: int) -> str:
    for k in range(line - 1, -1, -1):
        if lines[k][:1] in "cC*!":
            continue
        m = _UNIT.match(lines[k])
        if m:
            return m.group(1).upper()
    return ""


def _code(lines: list[str], line: int) -> str:
    code = [lines[line - 1].split("!")[0]]
    for nxt in lines[line:]:
        if len(nxt) > 5 and nxt[:1] not in "cC*!" and nxt[5] not in " 0":
            code.append(nxt[6:].split("!")[0])
        else:
            break
    return " ".join(code)


def _numbers(code: str) -> set[float]:
    return {abs(float(t.upper().replace("D", "E"))) for t in _NUMBER.findall(code)}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def test_every_coefficient_cites_a_reference() -> None:
    assert len(CITES) >= 30
    refs = {p.ref_version for _, _, p in CITES}
    assert refs == {R.REF_DSSAT, R.REF_RZWQM}
    for path, _, p in CITES:
        assert p.file and p.line, path
        if p.ref_version == R.REF_DSSAT:
            assert p.statement, path
        else:
            assert not p.statement and p.paper, path


@pytest.mark.parametrize(("name", "value", "prov"), [c for c in CITES if c[2].ref_version == R.REF_DSSAT])
def test_dssat_statement_is_on_the_cited_line(name: str, value: float, prov: Provenance) -> None:
    src = DSSAT_SOURCE / prov.file
    if not src.is_file():
        pytest.skip(f"DSSAT-CSM source not found at {src}")
    assert prov.line is not None
    lines = _lines(src)
    assert _norm(lines[prov.line - 1]) == _norm(prov.statement), (name, lines[prov.line - 1])
    assert _unit_at(lines, prov.line) == prov.routine.upper(), name
    assert abs(value) in _numbers(prov.statement), (name, value)


@pytest.mark.parametrize(("name", "value", "prov"), [c for c in CITES if c[2].ref_version == R.REF_RZWQM])
def test_rzwqm2_citation_holds_the_value(name: str, value: float, prov: Provenance, data_dir: Path) -> None:
    src = data_dir / RZ_SOURCE / prov.file
    if not src.is_file():
        pytest.skip(f"RZWQM2 source not found at {src}")
    assert prov.line is not None
    lines = _lines(src)
    assert _unit_at(lines, prov.line) == prov.routine.upper(), name
    assert abs(value) in _numbers(_code(lines, prov.line)), (name, value, _code(lines, prov.line))
