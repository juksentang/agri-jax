"""Coefficients and numerical settings of the soil-water modules (``soil_water/coefficients.py``).

* every numerical setting of the Richards redistribution is declared once, with unit, meaning and
  origin: an RZWQM2 convention cites file, line, routine and the published description (never the
  source text), an own choice names its basis;
* the static config fields carry their setting, and the field default is the declared value;
* the soil-water kernels carry no bare numeric literal (lint rule AJ007).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import pytest

import agrijax.processes.soil_water  # noqa: F401  (registers the processes)
from agrijax.core.lint import lint_paths
from agrijax.processes.soil_water import coefficients as SC
from agrijax.processes.soil_water.richards import AdaptiveStepping, FixedStepping, RichardsConfig

X64 = bool(jax.config.read("jax_enable_x64"))
SRC = Path(__file__).resolve().parents[2] / "src" / "agrijax"
FILES = [
    SRC / "processes" / "soil_water" / name
    for name in (
        "richards.py",
        "problem.py",
        "integrator.py",
        "fixed_cn.py",
        "richards_adaptive.py",
        "adaptive_stepping.py",
        "adaptive_newton.py",
        "adaptive_search.py",
        "sinks.py",
        "coefficients.py",
    )
] + [SRC / "core" / "depth_scan.py"]


def test_every_setting_has_unit_meaning_and_origin() -> None:
    assert len(SC.SETTINGS) >= 20
    for name, s in SC.SETTINGS.items():
        assert s.name == name and s.description.strip()
        assert s.origin in {"rzwqm2-4.6", "agrijax"}
        assert s.source
        if s.origin == "rzwqm2-4.6":
            p = s.provenance
            assert p is not None and p.ref_version == SC.REF_VERSION
            assert p.file.startswith("RZWQM/") and p.line and p.routine
            assert p.paper == SC.AHUJA_2000 and not p.statement
        else:
            assert s.basis or s.provenance is not None
    rows = SC.settings_table()
    assert [r["name"] for r in rows] == list(SC.SETTINGS)


@pytest.mark.parametrize("cls", [RichardsConfig, FixedStepping, AdaptiveStepping])
def test_config_fields_are_declared_settings(cls: type) -> None:
    inst = cls()
    for f in dataclasses.fields(cls):
        s = f.metadata["setting"]
        assert SC.SETTINGS[s.name] is s
        assert getattr(inst, f.name) == s.value and f.default == s.value
        assert f.metadata["unit"] == s.unit


def test_redeclaring_a_setting_differently_raises() -> None:
    SC.numerical_setting("richards.alpha_cn", 0.5, **_same("richards.alpha_cn"))
    with pytest.raises(ValueError, match="already declared"):
        SC.numerical_setting("richards.alpha_cn", 0.25, **_same("richards.alpha_cn"))
    with pytest.raises(ValueError, match="basis"):
        SC.NumericalSetting("x", 1.0, "-", "a setting", "agrijax")
    with pytest.raises(ValueError, match="source file"):
        SC.NumericalSetting("x", 1.0, "-", "a setting", "rzwqm2-4.6")


def _same(name: str) -> dict:
    s = SC.SETTINGS[name]
    return {"unit": s.unit, "description": s.description, "origin": s.origin, "provenance": s.provenance}


def test_no_bare_numeric_literal_in_the_soil_water_kernels() -> None:
    found = [f for f in lint_paths(FILES) if f.rule == "AJ007"]
    assert found == [], [f.format() for f in found]
