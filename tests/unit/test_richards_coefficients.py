"""Coefficients and numerical settings of the soil-water modules (``soil_water/coefficients.py``).

* every numerical setting (Richards, Green-Ampt, soil-water day) is declared once, with unit,
  meaning and origin: an RZWQM2 convention cites file, line, routine and the published
  description (never the source text), an own choice names its basis;
* the static config fields carry their setting, and the field default is the declared value;
* the Green-Ampt coefficients are a calibratable :class:`~agrijax.core.coefficients.Coefficients`
  set whose ``ref_version`` is the registry key of the Green-Ampt process; array leaves give the
  same event as the default floats, and the gradient with respect to ``vrcf`` is finite and equals
  finite differences;
* the soil-water kernels carry no bare numeric literal (lint rule AJ007).

The cited source lines are checked against the RZWQM2 source tree in
``tests/integration/test_richards_coefficients_source.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import agrijax.processes.soil_water  # noqa: F401  (registers the processes)
from agrijax.core.lint import lint_paths
from agrijax.core.process import list_processes
from agrijax.processes.soil_water import coefficients as SC
from agrijax.processes.soil_water import infiltration as G
from agrijax.processes.soil_water.day import DayConfig, soil_water_day_kernel
from agrijax.processes.soil_water.richards import RichardsConfig, SoilWater

from .test_infiltration_day import _forcing, _params
from .test_richards import catpa_grid, catpa_soil

X64 = bool(jax.config.read("jax_enable_x64"))
SRC = Path(__file__).resolve().parents[2] / "src" / "agrijax"
FILES = [
    SRC / "processes" / "soil_water" / name
    for name in ("richards.py", "day.py", "infiltration.py", "sinks.py", "coefficients.py")
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


@pytest.mark.parametrize("cls", [RichardsConfig, DayConfig])
def test_config_fields_are_declared_settings(cls: type) -> None:
    inst = cls()
    for f in dataclasses.fields(cls):
        s = f.metadata["setting"]
        assert SC.SETTINGS[s.name] is s
        assert getattr(inst, f.name) == s.value and f.default == s.value
        assert f.metadata["unit"] == s.unit


def test_green_ampt_config_fields_name_their_settings() -> None:
    cfg = G.GreenAmptConfig(n_slice=150)
    for name in ("ds", "dt_min", "rr_min"):
        f = next(f for f in dataclasses.fields(G.GreenAmptConfig) if f.name == name)
        s = SC.SETTINGS[f.metadata["setting"]]
        assert getattr(cfg, name) == s.value and f.metadata["unit"] == s.unit
    assert (G.DT_MIN, G.RR_MIN, G.SLICE_DS) == (1.0e-5, 1.0e-2, 1.0)


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


def test_green_ampt_coefficients_table_and_registry_reference() -> None:
    rows = SC.green_ampt_coefficient_table()
    assert [r["path"] for r in rows] == ["vrcf", "suction_offset", "suction_dry_limit", "suction_lower"]
    for r in rows:
        assert r["ref_version"] == SC.REF_VERSION and r["statement"] == "" and r["paper"] == SC.AHUJA_2000
        assert r["file"] == "RZWQM/RZTEST.for" and r["routine"] in {"INFIL", "EVNTRO"}
    assert SC.RZWQM2_GREEN_AMPT.calibratable_paths() == ["vrcf", "suction_offset", "suction_dry_limit"]
    assert G.VRCF == SC.RZWQM2_GREEN_AMPT.vrcf == 2.0
    keys = {str(p.key) for p in list_processes() if str(p.key).startswith("soil_water/")}
    assert "soil_water/infiltration_ga@rzwqm2-4.6:faithful" in keys
    assert all(k.split("@")[1].split(":")[0] == SC.REF_VERSION for k in keys)
    assert G.GreenAmptParams(config=G.GreenAmptConfig(n_slice=1)).coef() is SC.RZWQM2_GREEN_AMPT


def _storm_event(coefs: SC.GreenAmptCoefficients | None, theta: float = 0.2) -> G.GAResult:
    grid = catpa_grid()
    soil = catpa_soil()
    n = grid.n_node
    soil_n = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n,)), soil.at_nodes())
    th = jnp.full(n, theta)
    cfg = G.GreenAmptConfig.for_grid(np.asarray(grid.tl))
    return G.green_ampt_event(
        th,
        SoilWater.from_theta(th, soil).h,
        soil_n,
        grid.tl,
        jnp.asarray(0.9),
        jnp.asarray([0.2, 0.5]),
        jnp.asarray([6.0, 3.0]),
        cfg,
        coefs,
    )


def test_array_coefficients_give_the_default_event() -> None:
    ref = _storm_event(None)
    arr = _storm_event(SC.RZWQM2_GREEN_AMPT.as_arrays())
    assert float(ref.runoff) > 0.1  # the capacity binds, so vrcf matters
    for a, b in zip(ref, arr):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12 if X64 else 1e-6, atol=0.0)


def test_day_kernel_reads_the_coefficients_of_the_params() -> None:
    dp = _params(DayConfig(n_pre=0, n_post=12))
    fo = _forcing(1, supply_scale=0.0)
    storm = G.StormForcing(ts0=jnp.asarray(0.0), duration=jnp.asarray([0.2]), depth=jnp.asarray([6.0]))
    w0 = SoilWater.from_theta(jnp.full(dp.richards.grid.n_node, 0.2), dp.richards.soil)
    args = (fo.supply[0], fo.evaporation[0], fo.uptake[0], storm)
    _, ev0, _ = soil_water_day_kernel(w0, dp, *args)
    halved = SC.RZWQM2_GREEN_AMPT.replace(vrcf=1.0)
    dp1 = dp.replace(infiltration=dp.infiltration.replace(coefficients=halved))
    _, ev1, _ = soil_water_day_kernel(w0, dp1, *args)
    assert float(ev1.infiltration) > float(ev0.infiltration)  # a smaller reduction infiltrates more


def test_gradient_with_respect_to_vrcf_is_finite_and_matches_fd() -> None:
    base = SC.RZWQM2_GREEN_AMPT.as_arrays()

    def infil(v: jax.Array) -> jax.Array:
        return _storm_event(base.replace(vrcf=v)).infiltration

    v0 = jnp.asarray(2.0)
    g = float(jax.grad(infil)(v0))
    assert np.isfinite(g) and g < 0.0
    if X64:
        eps = 1e-6
        fd = (float(infil(v0 + eps)) - float(infil(v0 - eps))) / (2 * eps)
        assert g == pytest.approx(fd, rel=1e-5)


def test_no_bare_numeric_literal_in_the_soil_water_kernels() -> None:
    found = [f for f in lint_paths(FILES) if f.rule == "AJ007"]
    assert found == [], [f.format() for f in found]
