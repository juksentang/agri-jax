"""Every hoisted CERES-Maize coefficient equals the literal in the DSSAT-CSM source it cites.

For each row of :func:`agrijax.processes.crop.ceres_maize.coefficient_table` the cited line
(``MZ_GROSUB.for`` / ``MZ_PHENOL.for`` / ``MZ_ROOTS.for`` of DSSAT-CSM v4.8.6.0, the tree the
local ``dscsm048`` was built from) must hold the quoted Fortran statement (case and whitespace
aside, trailing ``!`` comments dropped), and the coefficient's default must be one of the numeric
literals of that statement. The two constants ``MZ_GROSUB`` sets in SEASINIT and the crop takes
as species parameters (``CANHT_POT``, ``BSGDD``) are checked the same way. (The species-file
coefficients are read from ``MZCER048.SPE`` at run time, not hoisted, and are covered by the
reference comparison.)

Skips (``allow_skip``) when the DSSAT source tree is absent (it is not part of the repository).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agrijax.processes.crop.ceres_maize import coefficient_table
from agrijax.processes.crop.ceres_maize.coefficients import BSGDD, CANHT_POT

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
SOURCE = DSSAT_ENGINE / "source" / "Plant" / "CERES-Maize"
_NUM = re.compile(r"(?<![A-Za-z_0-9])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+|\d+)")

pytestmark = [
    pytest.mark.allow_skip(reason="needs the local DSSAT-CSM v4.8.6.0 source tree (not in the repository)"),
    pytest.mark.skipif(not SOURCE.is_dir(), reason=f"DSSAT-CSM source not found under {SOURCE}"),
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


def _code(line: str) -> str:
    """A fixed-form Fortran line without its trailing ``!`` comment."""
    return line.split("!")[0]


_LINES: dict[str, list[str]] = {}


def _line(fname: str, lineno: int) -> str:
    if fname not in _LINES:
        _LINES[fname] = (SOURCE / fname).read_text(errors="replace").splitlines()
    return _LINES[fname][lineno - 1]


ROWS = [
    *coefficient_table(),
    {
        "path": "species.canht_pot (CANHT_POT)",
        "value": CANHT_POT,
        "source": "DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for:665",
        "fortran": "CANHT_POT = 1.6",
    },
    {
        "path": "species.bsgdd (BSGDD)",
        "value": BSGDD,
        "source": "DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for:656",
        "fortran": "BSGDD = 250.0",
    },
]


@pytest.mark.parametrize("row", ROWS, ids=[r["path"] for r in ROWS])
def test_default_equals_cited_fortran_literal(row):
    fname, lineno = row["source"].rsplit("/", 1)[1].split(":")
    line = _line(fname, int(lineno))
    assert line[:1] not in "!cC*", f"{row['source']} is a comment line: {line!r}"
    assert _norm(row["fortran"]) in _norm(_code(line)), (
        f"{row['source']}: {line.strip()!r} lacks {row['fortran']!r}"
    )
    literals = [float(x) for x in _NUM.findall(row["fortran"])]
    assert float(row["value"]) in literals, (row["path"], row["value"], row["fortran"])


def test_cited_lines_are_in_the_integrate_section():
    """Every coefficient of a DYNAMIC = INTEGR routine cites a line after ``DYNAMIC.EQ.INTEGR``."""
    for fname in ("MZ_GROSUB.for", "MZ_ROOTS.for"):
        lines = (SOURCE / fname).read_text(errors="replace").splitlines()
        start = next(i for i, ln in enumerate(lines, 1) if "DYNAMIC.EQ.INTEGR" in ln.replace(" ", "").upper())
        for r in coefficient_table():
            f, n = r["source"].rsplit("/", 1)[1].split(":")
            if f == fname:
                assert int(n) > start, (r["path"], r["source"], start)
