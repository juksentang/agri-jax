"""The RZWQM2 citations of the reference heads of the hydraulic functions point at the right lines.

For every :class:`~agrijax.processes.soil_water.hydraulics.HydraulicsCoefficients` field, the
cited ``<file>:<line>`` of the RZWQM2 source tree (``<data>/narval_mirror/RZWQM_Linux_Ver45/src``)
lies in the cited routine and holds the magnitude of the head as a Fortran literal. The check reads
the private source tree at test time; nothing of it is copied into the repository.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agrijax.core.coefficients import iter_coefficients
from agrijax.processes.soil_water import hydraulics as H

pytestmark = pytest.mark.allow_skip(reason="needs the private RZWQM2 source tree under the data dir")

SOURCE = Path("narval_mirror/RZWQM_Linux_Ver45/src")
_ROUTINE = re.compile(r"^\s+(?:[A-Z ]+\s)?(?:SUBROUTINE|FUNCTION)\s+(\w+)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w])(\d+\.\d*(?:[DE][-+]?\d+)?|\d+[DE][-+]?\d+)", re.IGNORECASE)


def _routine_at(lines: list[str], line: int) -> str:
    for k in range(line - 1, -1, -1):
        text = lines[k]
        if text[:1] in "cC*!":
            continue
        m = _ROUTINE.match(text)
        if m:
            return m.group(1).upper()
    return ""


@pytest.mark.parametrize(
    ("name", "value", "prov"),
    [(p, v, f.metadata["provenance"]) for p, f, v in iter_coefficients(H.RZWQM_HYDRAULICS)],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_rzwqm2_citation_is_in_the_routine_and_holds_the_head(
    data_dir: Path, name: str, value: float, prov: object
) -> None:
    src = data_dir / SOURCE / prov.file  # type: ignore[attr-defined]
    if not src.is_file():
        pytest.skip(f"{src} not found")
    lines = src.read_text(encoding="latin-1").splitlines()
    line = prov.line  # type: ignore[attr-defined]
    assert _routine_at(lines, line) == prov.routine.upper(), name  # type: ignore[attr-defined]
    code = lines[line - 1].split("!")[0]
    numbers = [float(x.upper().replace("D", "E")) for x in _NUMBER.findall(code)]
    assert any(n == abs(float(value)) for n in numbers), (name, value, numbers)
