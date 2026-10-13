"""Every coefficient of the SPAM partition (:mod:`agrijax.processes.pet.spam_dssat`) and of the soil /
mulch evaporation, the root extraction and the soil albedo
(:mod:`agrijax.processes.soil_water.bucket_evap`) equals a literal of the DSSAT-CSM
v4.8.6.0 source statement it quotes, on the line it cites.

For each row of the coefficient tables the cited line (``SPAM/*.for``, ``Weather/HMET.for``,
``Soil/Mulch/MULCHEVAP.for`` of ``$AGRI_JAX_DSSAT/source``, the tree build486 was compiled from)
must hold the quoted statement (case and whitespace aside, trailing ``!`` comments dropped), must
not be a comment line, and the coefficient's default (its magnitude, for the negative exponents)
must be one of the numeric literals of that statement.

Skips (``allow_skip``) when the DSSAT source tree is absent (it is not part of the repository).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agrijax.core.coefficients import coefficient_table
from agrijax.port.run_fortran import DEFAULT_DSSAT_ENGINE
from agrijax.processes.pet.spam_dssat import SPAM_COEFFICIENTS
from agrijax.processes.soil_water.bucket_evap import (
    ALBEDO_COEFFICIENTS,
    EVAP_COEFFICIENTS,
    XTRACT_COEFFICIENTS,
)

DSSAT_ENGINE = Path(os.environ.get("AGRI_JAX_DSSAT", str(DEFAULT_DSSAT_ENGINE))).expanduser()
SOURCE = DSSAT_ENGINE / "source"
_NUM = re.compile(r"(?<![A-Za-z_0-9])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)")

pytestmark = [
    pytest.mark.allow_skip(reason="needs the DSSAT-CSM v4.8.6.0 source tree (not in the repository)"),
    pytest.mark.skipif(not (SOURCE / "SPAM").is_dir(), reason=f"DSSAT-CSM source not found under {SOURCE}"),
]

ROWS = [
    *coefficient_table(SPAM_COEFFICIENTS),
    *coefficient_table(EVAP_COEFFICIENTS),
    # EP / XTRACT thresholds (SPAM.for) and the SOILDYN soil albedo (SOILDYN.for, MULCHLAYER.for)
    *coefficient_table(XTRACT_COEFFICIENTS),
    *coefficient_table(ALBEDO_COEFFICIENTS),
]
_LINES: dict[str, list[str]] = {}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def _line(rel: str, lineno: int) -> str:
    if rel not in _LINES:
        _LINES[rel] = (SOURCE / rel).read_text(encoding="latin-1").splitlines()
    return _LINES[rel][lineno - 1]


def test_every_coefficient_cites_a_statement() -> None:
    assert len(ROWS) >= 50
    for r in ROWS:
        assert r["fortran"], r["path"]
        assert "DSSAT-CSM v4.8.6.0" in r["source"], r["path"]


@pytest.mark.parametrize("row", ROWS, ids=[r["path"] for r in ROWS])
def test_default_equals_cited_fortran_literal(row) -> None:
    assert row["ref_version"] == "dssat-4.8.6.0", row["path"]
    where = f"{row['file']}:{row['line']}"
    line = _line(row["file"], int(row["line"]))
    assert line[:1] not in "!cC*", f"{where} is a comment line: {line!r}"
    code = line.split("!")[0]
    assert _norm(row["fortran"]) in _norm(code), f"{where}: {line.strip()!r} lacks {row['fortran']!r}"
    lits = {float(m) for m in _NUM.findall(row["fortran"])}
    assert abs(float(row["value"])) in lits, (where, row["value"], sorted(lits))
