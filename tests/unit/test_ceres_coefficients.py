"""The hoisted CERES-Maize coefficients (``agrijax.processes.crop.ceres_maize.coefficients``).

Data-free checks; the check of every default against the DSSAT-CSM source it cites (the local
reference tree) is ``tests/integration/test_ceres_coefficients_source.py``.

* every coefficient has a unit, a description, a ``<file>:<line>`` source in ``MZ_GROSUB`` /
  ``MZ_PHENOL`` / ``MZ_ROOTS`` and a Fortran statement in which its default is a literal;
* ``CeresMaizeParams.coefficients = None`` adds no pytree leaf and means the DSSAT values; a
  season with explicit (array) coefficients equals the default season to rounding (1e-12 in
  float64: runtime operands instead of compile-time literals may change XLA's constant folding);
* coefficients are differentiable: the season yield has a finite, non-zero gradient with respect
  to leaf-area, grain-fill and light-interception coefficients, equal to a central difference.
"""

from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import run
from agrijax.processes.crop.ceres_maize import (
    DSSAT_COEFFICIENTS,
    CeresCoefficients,
    CeresMaizeState,
    ceres_maize_model,
    coefficient_table,
)

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import make_params

X64 = jax.config.jax_enable_x64
_NUM = re.compile(r"(?<![A-Za-z_0-9])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+|\d+)")

MODEL = ceres_maize_model(
    outputs=lambda s, p, f: {"gwad": s.growth.grnwt * s.phen.ears * 10.0, "lai": s.growth.lai}
)


def test_every_coefficient_is_documented_and_cites_its_statement():
    rows = coefficient_table()
    assert len(rows) == len({r["path"] for r in rows}) > 100
    for r in rows:
        assert r["unit"] and r["description"], r["path"]
        m = re.fullmatch(
            r"DSSAT-CSM v4\.8\.6\.0 Plant/CERES-Maize/(MZ_GROSUB|MZ_PHENOL|MZ_ROOTS)\.for:(\d+)", r["source"]
        )
        assert m, (r["path"], r["source"])
        group = r["path"].split(".")[0]
        assert m.group(1) == {"grosub": "MZ_GROSUB", "phenol": "MZ_PHENOL", "roots": "MZ_ROOTS"}[group], r[
            "path"
        ]
        literals = [float(x) for x in _NUM.findall(r["fortran"])]
        assert float(r["value"]) in literals, (r["path"], r["value"], r["fortran"])
        assert isinstance(r["value"], int) == r["static"], r["path"]  # integer thresholds are static


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("grosub.sla_leaf", 267.0),
        ("grosub.cumph_emergence", 0.514),
        ("grosub.groear_max", 0.81),
        ("grosub.groear_mid", 210.0),
        ("grosub.swmin_frac", 0.85),
        ("grosub.grolf_max_frac", 0.75),
        ("grosub.grort1_min_frac", 0.25),
        ("grosub.grort4_frac", 0.08),
        ("phenol.gpp_psker", 7200.0),
        ("roots.rtdep_cumdtt", 275.0),
    ],
)
def test_named_defaults(path, value):
    group, name = path.split(".")
    assert getattr(getattr(DSSAT_COEFFICIENTS, group), name) == value


def test_none_adds_no_leaf_and_means_dssat():
    p = make_params()
    assert p.coefficients is None and p.coef() is DSSAT_COEFFICIENTS
    n0 = len(jax.tree_util.tree_leaves(p))
    q = p.replace(coefficients=CeresCoefficients().as_arrays())
    n_coef = sum(1 for r in coefficient_table() if not r["static"])
    assert len(jax.tree_util.tree_leaves(q)) == n0 + n_coef
    assert all(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(q.coefficients))
    assert q.coef().grosub.cold_days == 6  # static fields stay Python ints


def _season(p, f):
    out = run(MODEL, p, f, CeresMaizeState.initial(p, 1))
    return out["gwad"][-1, 0], out["lai"][:, 0]


def test_explicit_coefficients_reproduce_the_default_season():
    f, w = season_forcing(3, stress=True)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    y0, lai0 = jax.jit(_season)(p, f)
    y1, lai1 = jax.jit(_season)(p.replace(coefficients=CeresCoefficients().as_arrays()), f)
    assert float(y0) > 1000.0
    if X64:  # same values and operations; only XLA's folding of literals may differ (last bit)
        np.testing.assert_allclose(float(y1), float(y0), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(lai1), np.asarray(lai0), rtol=1e-12, atol=1e-12)
    else:  # float32: a literal product such as 3.0 * 3.1 is rounded once in Python, twice in float32
        np.testing.assert_allclose(np.asarray(lai1), np.asarray(lai0), rtol=1e-4, atol=1e-4)
    # a changed coefficient changes the season
    c = DSSAT_COEFFICIENTS
    c2 = c.replace(grosub=c.grosub.replace(sla_leaf=240.0))
    y2, _ = jax.jit(_season)(p.replace(coefficients=c2), f)
    assert float(y2) != float(y0)


@pytest.mark.skipif(not X64, reason="central differences need float64")
@pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)")
def test_yield_gradient_wrt_coefficients_matches_central_difference():
    f, w = season_forcing(5, stress=True, waterlog=False)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    names = ("sla_leaf", "grogrn_sw_base", "lifac_max")  # ear weight (groear_*) does not feed grain

    def yield_of(x):
        c = DSSAT_COEFFICIENTS.as_arrays()
        g = c.grosub.replace(**{n: x[i] for i, n in enumerate(names)})
        return _season(p.replace(coefficients=c.replace(grosub=g)), f)[0]

    x0 = jnp.asarray([getattr(DSSAT_COEFFICIENTS.grosub, n) for n in names])
    grad = np.asarray(jax.jit(jax.grad(yield_of))(x0))
    assert np.all(np.isfinite(grad)) and np.all(grad != 0.0), grad
    h = 1e-6 * np.abs(np.asarray(x0))
    fd = np.array(
        [
            (float(yield_of(x0.at[i].add(h[i]))) - float(yield_of(x0.at[i].add(-h[i])))) / (2 * h[i])
            for i in range(len(names))
        ]
    )
    np.testing.assert_allclose(grad, fd, rtol=1e-4)
