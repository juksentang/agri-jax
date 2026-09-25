"""The RZWQM2 citations of the soil-water coefficients and settings point at the right source lines.

For every Green-Ampt coefficient and every numerical setting whose origin is ``rzwqm2-4.6``, the
cited ``<file>:<line>`` of the RZWQM2 source tree (``<data>/narval_mirror/RZWQM_Linux_Ver45/src``)
lies in the cited routine, and the line holds the declared value as a Fortran literal (except
where the value is implied rather than written, listed in :data:`IMPLIED`). Nothing of the source
is copied into the repository: the check reads the private source tree at test time.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from agrijax.core.coefficients import iter_coefficients
from agrijax.processes.soil_water import coefficients as SC
from agrijax.processes.soil_water import day, infiltration, richards  # noqa: F401  (declare settings)

pytestmark = pytest.mark.allow_skip(reason="needs the private RZWQM2 source tree under the data dir")

SOURCE = Path("narval_mirror/RZWQM_Linux_Ver45/src")
#: citations whose line implies the value instead of writing it
IMPLIED = {
    "richards.face_k_mean_exponent": "the geometric mean is written as a square root",
    "green_ampt.ds": "the 1-cm infiltration grid is stated in the routine's header",
    "suction_dry_limit": "the integral term is DC1 / DC1 = 1",
}
_ROUTINE = re.compile(r"^\s+(?:[A-Z ]+\s)?(?:SUBROUTINE|FUNCTION)\s+(\w+)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w])(\d+\.\d*(?:[DE][-+]?\d+)?|\d+[DE][-+]?\d+)", re.IGNORECASE)


def _citations() -> list[tuple[str, float, object]]:
    out: list[tuple[str, float, object]] = []
    for path, f, value in iter_coefficients(SC.RZWQM2_GREEN_AMPT):
        out.append((path, value, f.metadata["provenance"]))
    for name, s in SC.SETTINGS.items():
        if s.origin == SC.REF_VERSION:
            out.append((name, s.value, s.provenance))
    return out


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
    ("name", "value", "prov"), _citations(), ids=lambda x: x if isinstance(x, str) else ""
)
def test_rzwqm2_citation_is_in_the_routine_and_holds_the_value(
    data_dir: Path, name: str, value: float, prov: object
) -> None:
    src = data_dir / SOURCE / prov.file  # type: ignore[attr-defined]
    if not src.is_file():
        pytest.skip(f"{src} not found")
    lines = src.read_text(encoding="latin-1").splitlines()
    line = prov.line  # type: ignore[attr-defined]
    assert _routine_at(lines, line) == prov.routine.upper()  # type: ignore[attr-defined]
    if name in IMPLIED:
        return
    code = lines[line - 1].split("!")[0]
    numbers = [float(x.upper().replace("D", "E")) for x in _NUMBER.findall(code)]
    assert any(n == float(value) for n in numbers), (name, value, numbers)
