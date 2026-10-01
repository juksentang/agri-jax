"""The citations of the native DSSAT input coefficients point at the right DSSAT-CSM v4.8.6.0 lines.

For every coefficient of :mod:`agrijax.io.dssat.native_soil`, :mod:`~agrijax.io.dssat.native_weather`,
:mod:`~agrijax.io.dssat.native_management` and :mod:`agrijax.forcing.dssat_weather`: the cited
``<file>:<line>`` of ``<AGRI_JAX_DSSAT>/source`` lies in the cited routine and the line (with its
fixed-form continuation lines) holds the coefficient's value as a literal, up to sign. No statement
is quoted in the repository: the check reads the source tree at test time.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agrijax.core.coefficients import iter_coefficients
from agrijax.forcing.dssat_weather import DSSAT486_WEATHER
from agrijax.io.dssat.native_management import DSSAT_MANAGEMENT_INPUT
from agrijax.io.dssat.native_soil import DSSAT_SOIL_INPUT
from agrijax.io.dssat.native_weather import DSSAT_WEATHER_INPUT
from agrijax.port.run_fortran import DEFAULT_DSSAT_ENGINE

pytestmark = pytest.mark.allow_skip(reason="needs the DSSAT-CSM v4.8.6.0 source tree (AGRI_JAX_DSSAT)")

SOURCE = Path(os.environ.get("AGRI_JAX_DSSAT", str(DEFAULT_DSSAT_ENGINE))).expanduser() / "source"
_UNIT = re.compile(r"^\s+(?:[A-Z*0-9 ]+\s)?(?:PROGRAM|SUBROUTINE|FUNCTION|MODULE)\s+(\w+)", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])(\d+\.?\d*(?:[DE][-+]?\d+)?|\.\d+(?:[DE][-+]?\d+)?)", re.IGNORECASE)

CITES = [
    (f"{type(tree).__name__}.{p}", v, f.metadata["provenance"])
    for tree in (DSSAT_SOIL_INPUT, DSSAT_WEATHER_INPUT, DSSAT_MANAGEMENT_INPUT, DSSAT486_WEATHER)
    for p, f, v in iter_coefficients(tree)
]


def _lines(path: Path) -> list[str]:
    return path.read_bytes().decode("latin-1").replace("\r", "").split("\n")


_END_INTERFACE = re.compile(r"^\s+END\s*INTERFACE", re.IGNORECASE)
_INTERFACE = re.compile(r"^\s+(ABSTRACT\s+)?INTERFACE\b", re.IGNORECASE)
_DOTTED = re.compile(r"\.(GE|GT|LE|LT|EQ|NE|AND|OR|NOT|EQV|NEQV)\.", re.IGNORECASE)


def _unit_at(lines: list[str], line: int) -> str:
    """The program unit the line is in (interface blocks of the unit are skipped)."""
    k = line - 1
    while k >= 0:
        ln = lines[k]
        if ln[:1] not in "cC*!":
            if _END_INTERFACE.match(ln):
                while k >= 0 and not _INTERFACE.match(lines[k]):
                    k -= 1
            else:
                m = _UNIT.match(ln)
                if m:
                    return m.group(1).upper()
        k -= 1
    return ""


def _code(lines: list[str], line: int) -> str:
    code = [lines[line - 1].split("!")[0]]
    for nxt in lines[line:]:
        if len(nxt) > 5 and nxt[:1] not in "cC*!" and nxt[5] not in " 0" and not nxt.lstrip().startswith("!"):
            code.append(nxt[6:].split("!")[0])
        else:
            break
    return " ".join(code)


def _numbers(code: str) -> set[float]:
    code = _DOTTED.sub(" ", code)  # CUMDEP .GE.200. -> CUMDEP 200.
    return {abs(float(t.upper().replace("D", "E"))) for t in _NUMBER.findall(code)}


def test_every_coefficient_cites_a_dssat_line_without_quoting_it() -> None:
    assert len(CITES) >= 60
    for name, _, p in CITES:
        assert p.ref_version == "dssat-4.8.6.0" and p.file and p.line and p.routine, name
        assert not p.statement, name


@pytest.mark.parametrize(("name", "value", "prov"), CITES, ids=[c[0] for c in CITES])
def test_cited_line_holds_the_value(name: str, value: float, prov) -> None:
    path = SOURCE / prov.file
    if not path.is_file():
        pytest.skip(f"{path} not found")
    lines = _lines(path)
    assert prov.line <= len(lines), name
    assert _unit_at(lines, prov.line) == prov.routine.upper(), (name, _unit_at(lines, prov.line))
    assert abs(float(value)) in _numbers(_code(lines, prov.line)), (name, value, prov.file, prov.line)
