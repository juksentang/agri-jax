"""The reference citations of the PET coefficients point at the right source lines.

* RZWQM2 (``rzwqm2-4.6``): for every Shuttleworth-Wallace coefficient and every ``REF_ET.FOR``
  constant, the cited ``<file>:<line>`` of the RZWQM2 source tree
  (``<data>/narval_mirror/RZWQM_Linux_Ver45/src``) lies in the cited routine, and the line (or its
  Fortran continuation lines) holds the declared value as a Fortran literal, up to the sign the
  equation applies (except where the value is implied, :data:`IMPLIED`). Nothing of the source is
  copied into the repository: the check reads the private source tree at test time.
* DSSAT-CSM (``dssat-4.8.6.0``, BSD-3): every ``PETPT`` coefficient's quoted statement is the
  statement on the cited line of ``SPAM/PET.for`` (whitespace and case aside) and holds the value.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agrijax.core.coefficients import iter_coefficients
from agrijax.processes.pet import coefficients as PC

pytestmark = pytest.mark.allow_skip(reason="needs the private RZWQM2 source tree and the DSSAT-CSM source")

RZ_SOURCE = Path("narval_mirror/RZWQM_Linux_Ver45/src")
DSSAT_SOURCE = (
    Path(os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")).expanduser()
    / "source"
)
#: citations whose line implies the value instead of writing it
IMPLIED = {
    "sw.maxsw.hours_per_radian": "PPCNST = 12/PI (the port uses math.pi)",
    "sw.wind.canopy_height_factor": "TWOTRD = 2/3",
    "sw.resist.ras_height_fraction": "written as a division by 2",
}
_ROUTINE = re.compile(r"^\s+(?:[A-Z ]+\s)?(?:SUBROUTINE|FUNCTION)\s+(\w+)", re.IGNORECASE)
_NUMBER = re.compile(
    r"(?<![\w])(\d+\.\d*(?:[DE][-+]?\d+)?|\d+[DE][-+]?\d+)|(?<![\w.])(\.\d+(?:[DE][-+]?\d+)?)", re.IGNORECASE
)


def _citations(ref: str) -> list[tuple[str, float, object]]:
    return [
        (path, value, f.metadata["provenance"])
        for path, f, value in iter_coefficients(PC.PET_COEFFICIENTS)
        if f.metadata["provenance"].ref_version == ref
    ]


def _routine_at(lines: list[str], line: int) -> str:
    for k in range(line - 1, -1, -1):
        text = lines[k]
        if text[:1] in "cC*!":
            continue
        m = _ROUTINE.match(text)
        if m:
            return m.group(1).upper()
    return ""


def _statement(lines: list[str], line: int) -> str:
    """The code of ``line`` and its fixed-form continuation lines (comments stripped)."""
    code = [lines[line - 1].split("!")[0]]
    for nxt in lines[line:]:
        if len(nxt) > 5 and nxt[:1] not in "cC*!" and nxt[5] not in " 0":
            code.append(nxt[6:].split("!")[0])
        else:
            break
    return " ".join(code)


def _numbers(code: str) -> list[float]:
    return [float((a or b).upper().replace("D", "E")) for a, b in _NUMBER.findall(code)]


@pytest.mark.parametrize(
    ("name", "value", "prov"), _citations(PC.REF_RZWQM), ids=lambda x: x if isinstance(x, str) else ""
)
def test_rzwqm2_citation_is_in_the_routine_and_holds_the_value(
    data_dir: Path, name: str, value: float, prov: object
) -> None:
    src = data_dir / RZ_SOURCE / prov.file  # type: ignore[attr-defined]
    if not src.is_file():
        pytest.skip(f"{src} not found")
    lines = src.read_text(encoding="latin-1").splitlines()
    line = prov.line  # type: ignore[attr-defined]
    assert _routine_at(lines, line) == prov.routine.upper(), name  # type: ignore[attr-defined]
    if name in IMPLIED:
        return
    numbers = _numbers(_statement(lines, line))
    assert any(n == abs(float(value)) for n in numbers), (name, value, numbers)


@pytest.mark.parametrize(
    ("name", "value", "prov"), _citations(PC.REF_DSSAT), ids=lambda x: x if isinstance(x, str) else ""
)
def test_dssat_statement_is_on_the_cited_line_and_holds_the_value(
    name: str, value: float, prov: object
) -> None:
    src = DSSAT_SOURCE / prov.file  # type: ignore[attr-defined]
    if not src.is_file():
        pytest.skip(f"{src} not found")
    lines = src.read_text(encoding="latin-1").splitlines()
    line = prov.line  # type: ignore[attr-defined]
    assert _routine_at(lines, line) == "PETPT"

    def norm(s: str) -> str:
        return re.sub(r"\s+", "", s).upper()

    assert norm(prov.statement) == norm(lines[line - 1].split("!")[0]), name  # type: ignore[attr-defined]
    assert any(n == float(value) for n in _numbers(prov.statement)), (name, value)  # type: ignore[attr-defined]
