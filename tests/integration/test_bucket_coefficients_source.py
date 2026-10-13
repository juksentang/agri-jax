"""Every tipping-bucket coefficient equals a literal of the DSSAT-CSM statement it cites.

For each row of ``WATBAL_COEFFICIENTS.table()`` (``processes/soil_water/bucket/coefficients.py``)
the cited line of the DSSAT-CSM v4.8.6.0 source (``$AGRI_JAX_DSSAT/source``, the tree build486 and
the instrumented build were compiled from) must hold the quoted statement (case and blanks
aside; the quote may carry the line's trailing ``!`` comment), and the default must be one of the
numeric literals of that statement.

Skips (``allow_skip``) when the DSSAT source tree is absent (it is not part of the repository).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from agrijax.port.run_fortran import DEFAULT_DSSAT_ENGINE
from agrijax.processes.soil_water.bucket import WATBAL_COEFFICIENTS

SOURCE = Path(os.environ.get("AGRI_JAX_DSSAT", str(DEFAULT_DSSAT_ENGINE))).expanduser() / "source"
_NUM = re.compile(r"(?<![A-Za-z_0-9])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+|\d+)")

pytestmark = [
    pytest.mark.allow_skip(reason="needs the local DSSAT-CSM v4.8.6.0 source tree (not in the repository)"),
    pytest.mark.skipif(
        not (SOURCE / "Soil" / "SoilWater").is_dir(), reason=f"DSSAT-CSM source not found under {SOURCE}"
    ),
]

ROWS = WATBAL_COEFFICIENTS.table()
#: coefficients that are an implicit factor 1 of their statement (``SNOMLT = TMAX + RAIN*0.4``: the
#: melt per degree of TMAX) and static decimal counts (``ANINT(SW * 1.e6) / 1.e6``: 10**6)
IMPLICIT_ONE = {"melt_per_degc"}
DECIMALS = {"sw_decimals"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()


@pytest.mark.parametrize("row", ROWS, ids=[r["path"] for r in ROWS])
def test_default_equals_cited_fortran_literal(row) -> None:
    assert row["ref_version"] == "dssat-4.8.6.0" and row["file"] and row["line"], row
    line = (SOURCE / row["file"]).read_text(errors="replace").splitlines()[row["line"] - 1]
    assert _norm(row["fortran"]) in _norm(line), (
        f"{row['file']}:{row['line']}: {line.strip()!r} lacks {row['fortran']!r}"
    )
    code = line if row["fortran"].lstrip().startswith("&") else line.split("!")[0]
    values = {float(m) for m in _NUM.findall(code)}
    v = float(row["value"])
    if row["path"] in IMPLICIT_ONE:
        assert v == 1.0, row
    elif row["path"] in DECIMALS:
        assert 10.0**v in values, (row["path"], v, sorted(values))
    else:
        assert v in values, (row["path"], v, sorted(values))


def test_every_number_of_the_kernels_is_a_declared_coefficient() -> None:
    """The coefficient set is complete: lint rule AJ007 finds no bare literal in the bucket package."""
    from agrijax.core.lint import lint_paths

    pkg = Path(__file__).resolve().parents[2] / "src" / "agrijax" / "processes" / "soil_water" / "bucket"
    findings = [f for f in lint_paths([pkg]) if f.rule == "AJ007"]
    assert findings == [], findings
