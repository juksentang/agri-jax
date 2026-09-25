"""Coefficients of the Brooks-Corey hydraulic functions (``soil_water/hydraulics.py``).

* the reference heads of RZWQM2 (1/3 bar, 1/10 bar, 15 bar, dry-end clamp) are declared once with
  unit, meaning and an RZWQM2 provenance (file, line, routine, published source; no source text);
  they are conventions of the reference, not calibrated;
* the module constants ``H_FC13 H_FC110 H_WP H_CLAMP_RZWQM`` are the declared defaults, and the
  numerical floors (``H_MIN``, ``A1_MIN``, ...) are declared numerical guards;
* :func:`derive_rzwqm` evaluates the curve at the heads of the coefficient set it is given;
* ``hydraulics.py`` carries no bare numeric literal (lint rule AJ007).

The cited source lines are checked in ``tests/integration/test_hydraulics_coefficients_source.py``.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.coefficients import GUARDS, coefficient_table
from agrijax.core.lint import lint_paths
from agrijax.processes.soil_water import hydraulics as H

X64 = bool(jax.config.read("jax_enable_x64"))
FILE = Path(H.__file__)


def _params() -> H.SoilHydraulicParams:
    f = jnp.asarray
    return H.SoilHydraulicParams(
        hb=f([20.0, 45.0]), lambda_=f([0.3, 0.15]), eps=f([2.9, 3.4]), ksat=f([2.6, 0.4]),
        theta_r=f([0.04, 0.06]), theta_s=f([0.42, 0.46]), fc13=f([0.0, 0.0]), fc110=f([0.0, 0.0]),
        wp=f([0.0, 0.0]), hb_k=f([20.0, 45.0]), c2=f([0.0, 0.0]), n1=f([0.0, 0.0]), a1=f([0.002, 0.0]),
    )  # fmt: skip


def test_reference_heads_are_declared_with_rzwqm2_provenance() -> None:
    rows = coefficient_table(H.RZWQM_HYDRAULICS)
    assert [r["path"] for r in rows] == ["h_fc13", "h_fc110", "h_wp", "h_clamp"]
    assert [r["value"] for r in rows] == [-333.0, -100.0, -15000.0, -15000.0]
    for r in rows:
        assert r["unit"] == "cm" and r["description"].strip()
        assert r["ref_version"] == "rzwqm2-4.6" and r["statement"] == "" and r["paper"]
        assert r["file"] in {"RZWQM/RZTEST.for", "RZWQM/Rzmain.for"} and r["line"] and r["routine"]
        assert not r["calibratable"] and not r["static"]


def test_module_constants_are_the_declared_defaults_and_guards() -> None:
    c = H.RZWQM_HYDRAULICS
    assert (H.H_FC13, H.H_FC110, H.H_WP, H.H_CLAMP_RZWQM) == (c.h_fc13, c.h_fc110, c.h_wp, c.h_clamp)
    assert H.HydraulicsCoefficients() == c
    for name, value in (("hydraulics.h_min", H.H_MIN), ("hydraulics.a1_min", H.A1_MIN)):
        assert GUARDS[name][0] == value
    assert "hydraulics.se_floor" in GUARDS and "hydraulics.absh_wet_floor" in GUARDS


def test_derive_rzwqm_uses_the_heads_of_its_coefficients() -> None:
    p = _params()
    ref = H.derive_rzwqm(p)
    # array leaves give the same derived values as the default floats
    arr = H.derive_rzwqm(p, H.RZWQM_HYDRAULICS.as_arrays())
    for a, b in zip(jax.tree_util.tree_leaves(arr), jax.tree_util.tree_leaves(ref), strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12 if X64 else 1e-6)
    # FAO's 1/3 bar (-340 cm) instead of RZWQM's -333 cm moves fc13 only
    fao = H.derive_rzwqm(p, H.HydraulicsCoefficients(h_fc13=-340.0))
    np.testing.assert_allclose(np.asarray(fao.fc13), np.asarray(H.theta_of_h(jnp.full(2, -340.0), p)))
    assert bool(jnp.all(fao.fc13 < ref.fc13))
    np.testing.assert_array_equal(np.asarray(fao.fc110), np.asarray(ref.fc110))
    np.testing.assert_array_equal(np.asarray(fao.wp), np.asarray(ref.wp))
    # the derived wilting point is differentiable with respect to its head
    g = jax.grad(lambda h: jnp.sum(H.derive_rzwqm(p, H.HydraulicsCoefficients(h_wp=h)).wp))(-15000.0)
    np.testing.assert_allclose(float(g), float(jnp.sum(H.c_of_h(jnp.full(2, -15000.0), p))), rtol=1e-6)


def test_no_bare_numeric_literal_in_the_hydraulic_functions() -> None:
    found = [f for f in lint_paths([FILE]) if f.rule == "AJ007"]
    assert found == [], [str(f) for f in found]
