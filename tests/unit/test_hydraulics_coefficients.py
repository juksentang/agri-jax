"""Coefficients of the Brooks-Corey hydraulic functions (``soil_water/hydraulics.py``).

* the dry-end clamp of the state (15 bar) is declared once with unit, meaning and an RZWQM2
  provenance (file, line, routine, published source; no source text); it is a convention of the
  reference, not calibrated;
* the module constant ``H_CLAMP_RZWQM`` is the declared default, and the numerical floors
  (``H_MIN``, ``A1_MIN``, ...) are declared numerical guards;
* ``hydraulics.py`` carries no bare numeric literal (lint rule AJ007).
"""

from __future__ import annotations

from pathlib import Path

from agrijax.core.coefficients import GUARDS, coefficient_table
from agrijax.core.lint import lint_paths
from agrijax.processes.soil_water import hydraulics as H

FILE = Path(H.__file__)


def test_dry_end_clamp_is_declared_with_rzwqm2_provenance() -> None:
    rows = coefficient_table(H.RZWQM_HYDRAULICS)
    assert [r["path"] for r in rows] == ["h_clamp"]
    assert [r["value"] for r in rows] == [-15000.0]
    for r in rows:
        assert r["unit"] == "cm" and r["description"].strip()
        assert r["ref_version"] == "rzwqm2-4.6" and r["statement"] == "" and r["paper"]
        assert r["file"] == "RZWQM/Rzmain.for" and r["line"] and r["routine"]
        assert not r["calibratable"] and not r["static"]


def test_module_constants_are_the_declared_defaults_and_guards() -> None:
    c = H.RZWQM_HYDRAULICS
    assert H.H_CLAMP_RZWQM == c.h_clamp
    assert H.HydraulicsCoefficients() == c
    for name, value in (("hydraulics.h_min", H.H_MIN), ("hydraulics.a1_min", H.A1_MIN)):
        assert GUARDS[name][0] == value
    assert "hydraulics.se_floor" in GUARDS and "hydraulics.absh_wet_floor" in GUARDS


def test_no_bare_numeric_literal_in_the_hydraulic_functions() -> None:
    found = [f for f in lint_paths([FILE]) if f.rule == "AJ007"]
    assert found == [], [str(f) for f in found]
