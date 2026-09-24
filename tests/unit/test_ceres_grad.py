"""Gradients through CERES-Maize: central differences and finiteness over every ``jnp.where`` branch.

* ``jax.jacfwd`` of season outputs (final grain yield, integrated LAI, final above-ground
  biomass) with respect to the cultivar coefficients P1, P5, G2, G3 and PHINT equals central
  finite differences (float64). The finite-difference step is checked not to move any stage
  transition or leaf-number step, so both sides differentiate the same smooth piece. P1 acts only
  through the discrete end-of-juvenile date (nitrogen off: ``XSTAGE`` feeds nothing), so its
  derivative is 0 on both sides; P5 acts continuously only on the stage-5 leaf senescence.
  Waterlogging is left out of these seasons: through ``SATFAC`` the root length density feeds
  back on growth, and ``RLV`` is truncated to 1e-3 as in the Fortran with a straight-through
  (identity) derivative, which a finite difference of the truncated model does not see.
* The gradient of one day of the model with respect to every float parameter, state and forcing
  leaf is finite on a grid of states collected from synthetic seasons that visit every stage and
  every failure / stress path (drought, waterlogging, cold, seed-reserve exhaustion, slow grain
  fill, germination and emergence failure), plus degenerate states (empty crop, zero
  population, zero assimilation). The thermal-time kernel is checked on a temperature grid that
  crosses every branch boundary.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import run
from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model, thermal_time

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import a, make_params

X64 = jax.config.jax_enable_x64


def needs_x64(fn):
    """Central differences need float64 (the CI float32 pass skips them)."""
    fn = pytest.mark.skipif(not X64, reason="central differences need float64")(fn)
    return pytest.mark.allow_skip(reason="finite differences are float64-only (CI float32 pass)")(fn)


MODEL = ceres_maize_model(
    outputs=lambda s, p, f: {
        "lai": s.growth.lai,
        "cwad": s.growth.biomas * 10.0,
        "gwad": s.growth.grnwt * s.phen.ears * 10.0,
        "istage": s.phen.istage,
        "leafno": s.growth.leafno,
    }
)
PARAMS = ("p1", "p5", "g2", "g3", "phint")


def _set(p, name, value):
    return p.replace(cultivar=p.cultivar.replace(**{name: value}))


def _losses(p, f):
    out = run(MODEL, p, f, CeresMaizeState.initial(p, 1))
    return (
        jnp.stack([out["gwad"][-1, 0], jnp.sum(out["lai"][:, 0]), out["cwad"][-1, 0]]),
        (out["istage"], out["leafno"]),
    )


def _season_derivatives(seed: int, stress: bool):
    """AD Jacobian and central differences of the 3 season outputs w.r.t. the 5 coefficients.

    One compilation each: ``jacfwd`` over the coefficient vector, and the model evaluated at the
    base point and the 10 points ``x +- h e_i`` under ``vmap``.
    """
    # water stress on photosynthesis and expansion, no waterlogging: with SATFAC = 0 the root
    # length density (truncated to 1e-3 with a straight-through derivative) feeds nothing back
    f, w = season_forcing(seed, stress=stress, waterlog=False)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    x0 = np.array([float(getattr(p.cultivar, n)) for n in PARAMS])

    def with_x(x):
        return p.replace(cultivar=p.cultivar.replace(**{n: x[k] for k, n in enumerate(PARAMS)}))

    jac = np.asarray(jax.jit(jax.jacfwd(lambda x: _losses(with_x(x), f)[0]))(a(x0)))
    h = 1e-6 * np.maximum(np.abs(x0), 1.0)
    pts = np.concatenate([x0[None], x0 + np.diag(h), x0 - np.diag(h)])
    vals, (stages, leafno) = jax.jit(jax.vmap(lambda x: _losses(with_x(x), f)))(a(pts))
    vals, stages, leafno = np.asarray(vals), np.asarray(stages), np.asarray(leafno)
    fd = ((vals[1:6] - vals[6:11]) / (2 * h[:, None])).T  # [output, coefficient]
    return jac, fd, vals[0], stages, leafno


_DERIVS: dict[tuple[int, bool], tuple] = {}


@needs_x64
@pytest.mark.parametrize(("seed", "stress"), [(31, False), (32, True)])
def test_gradients_equal_central_differences(seed, stress):
    if (seed, stress) not in _DERIVS:
        _DERIVS[(seed, stress)] = _season_derivatives(seed, stress)
    ad, fd, base, stages, leafno = _DERIVS[(seed, stress)]
    # the steps must not move a stage date or a leaf-number step (same smooth piece on both sides)
    for k in range(1, 11):
        assert np.array_equal(stages[k], stages[0]) and np.array_equal(leafno[k], leafno[0])
    assert np.all(np.isfinite(ad))
    # absolute floor: the rounding noise of the central difference, ~ eps * |loss| / h
    np.testing.assert_allclose(ad, fd, rtol=2e-4, atol=1e-8 * np.abs(base).max())
    col = {n: k for k, n in enumerate(PARAMS)}
    assert np.all(ad[:, col["p1"]] == 0.0)
    for n in ("g2", "g3", "phint"):
        assert abs(ad[0, col[n]]) > 0.0  # kernel number, kernel growth rate and leaf number act on yield
    # P5 acts continuously only through the stage-5 leaf senescence SLAN (integrated LAI); grain
    # growth does not depend on assimilation, and the season length moves in whole days
    assert abs(ad[1, col["p5"]]) > 0.0 and ad[0, col["p5"]] == 0.0


@needs_x64
def test_p1_acts_through_the_juvenile_end_date():
    """A step of P1 large enough to move the end of the juvenile phase changes the yield."""
    f, w = season_forcing(31)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    y0 = float(_losses(p, f)[0][0])
    y1 = float(_losses(_set(p, "p1", a(300.0)), f)[0][0])
    assert y1 != y0


# ------------------------------------------------------------------ finiteness over every branch
def _collect_states():
    """Daily (state, forcing) pairs from seasons covering every stage and stress path."""
    step_states = []
    model_s = ceres_maize_model(outputs=lambda s, p, f: s)
    runner = jax.jit(lambda p_, f_, s_: run(model_s, p_, f_, s_))
    specs: list[dict[str, Any]] = [
        dict(seed=41, stress=False, yrplt=2),
        dict(seed=42, stress=True, yrplt=2),
        dict(seed=43, stress=True, cold=True, yrplt=0),
        dict(seed=44, stress=False, always_dry=True, yrplt=0),  # germination failure
        dict(seed=45, stress=False, dry_days=6, yrplt=0),
    ]
    for sp in specs:
        yrplt = sp.pop("yrplt")
        f, w = season_forcing(**sp)
        p = make_params(yrplt=int(w["yrdoy"][yrplt]))
        s0 = CeresMaizeState.initial(p, 1)
        states = runner(p, f, s0)
        # the state entering day t is the output of day t - 1 (s0 for day 0)
        prev = jax.tree_util.tree_map(lambda x0, xs: jnp.concatenate([x0[None], xs[:-1]]), s0, states)
        step_states.append((p, prev, f))
    return step_states


def _float(*trees):
    """Float leaves of gradient pytrees (integer leaves have ``float0`` cotangents)."""
    leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(trees)]
    return [x for x in leaves if np.issubdtype(x.dtype, np.floating)]


def _float_leaves_sum(tree):
    return sum(jnp.sum(x) for x in jax.tree_util.tree_leaves(tree) if jnp.issubdtype(x.dtype, jnp.floating))


def test_one_day_gradient_is_finite_on_every_visited_state():
    step = ceres_maize_model(outputs=lambda s, p, f: s).compile()

    def scalar(p, s, f):
        s_new, _ = step(s, p, f)
        return _float_leaves_sum(s_new)

    grad = jax.jit(jax.vmap(jax.grad(scalar, argnums=(0, 1, 2), allow_int=True), in_axes=(None, 0, 0)))
    seen: set[int] = set()
    flags = dict(satfac=False, swfac=False, spent=False, cold=False, lai4=False, fail=False)
    for p, prev, f in _collect_states():
        gp, gs, gf = grad(p, prev, f)
        for leaf in _float(gp, gs, gf):
            assert np.all(np.isfinite(leaf)), "non-finite gradient"
        seen |= set(np.asarray(prev.phen.istage).ravel().tolist())
        flags["satfac"] |= bool(np.any(np.asarray(prev.stress.satfac) > 0))
        flags["swfac"] |= bool(np.any(np.asarray(prev.stress.swfac) < 0.1))
        flags["spent"] |= bool(
            np.any((np.asarray(prev.growth.seedrv) == 0) & (np.asarray(prev.phen.istage) == 1))
        )
        flags["cold"] |= bool(np.any(np.asarray(f.tmin) <= 6.0))
        flags["lai4"] |= bool(np.any(np.asarray(prev.growth.lai) > 4.0))
        flags["fail"] |= bool(np.any(np.isin(np.asarray(prev.phen.crop_status), [12, 13, 32, 33])))
    assert seen >= {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, seen
    assert all(flags.values()), flags


def test_one_day_gradient_is_finite_on_degenerate_states():
    """Empty crop, zero population, zero leaf mass / area and zero radiation in every stage."""
    step = ceres_maize_model(outputs=lambda s, p, f: s).compile()
    f, w = season_forcing(46, n=10)
    p = make_params(yrplt=int(w["yrdoy"][0]), pltpop=7.2)
    ft = jax.tree_util.tree_map(lambda x: x[3], f).replace(srad=a(0.0))
    n = 11
    s = CeresMaizeState.initial(p, n)
    stage = jnp.asarray([7, 8, 9, 1, 2, 3, 4, 5, 6, 10, 1], dtype=jnp.int32)
    zero_pop = jnp.asarray([0.0] * 10 + [0.0])
    s = s.replace(phen=s.phen.replace(istage=stage), growth=s.growth.replace(pltpop=zero_pop))

    def scalar(p_, s_, f_):
        s_new, _ = step(s_, p_, f_)
        return _float_leaves_sum(s_new)

    grad = jax.jit(jax.grad(scalar, argnums=(0, 1, 2), allow_int=True))
    for pp in (p, p.replace(cultivar=p.cultivar.replace(p5=a(0.0), g2=a(0.0), djti=a(0.0)))):
        gp, gs, gf = grad(pp, s, ft)
        for leaf in _float(gp, gs, gf):
            assert np.all(np.isfinite(leaf))


def test_thermal_time_gradient_is_finite_across_every_branch_boundary():
    p = make_params()
    tmax = jnp.linspace(-10.0, 45.0, 56)
    tmin = jnp.linspace(-15.0, 40.0, 56)
    tx, tn = jnp.meshgrid(tmax, tmin, indexing="ij")
    ok = tn <= tx
    tx, tn = tx[ok], tn[ok]

    def dtt(tx_, tn_, srad, dayl, snow, leafno, istage):
        return thermal_time(tx_, tn_, srad, dayl, snow, leafno, istage, p.cultivar)

    for leafno in (3, 12):
        for snow in (0.0, 10.0):
            for istage in (1, 5):
                g = jax.vmap(
                    jax.grad(dtt, argnums=(0, 1, 2, 3, 4)), in_axes=(0, 0, None, None, None, None, None)
                )(tx, tn, a(15.0), a(13.0), a(snow), jnp.asarray(leafno), jnp.asarray(istage))
                for leaf in g:
                    assert np.all(np.isfinite(np.asarray(leaf)))


@needs_x64
@pytest.mark.parametrize(("seed", "stress"), [(33, True)])
def test_hardcoded_coefficients_are_calibratable(seed, stress):
    """Every hoisted DSSAT coefficient is a differentiable leaf: the AD Jacobian of the season
    outputs w.r.t. all of them is finite, many act, and the five with the largest effect agree
    with central differences taken on the same smooth piece (no stage date or leaf step moved)."""
    from agrijax.processes.crop.ceres_maize.coefficients import DSSAT_COEFFICIENTS

    f, w = season_forcing(seed, stress=stress, waterlog=False)
    p = make_params(yrplt=int(w["yrdoy"][2])).replace(coefficients=DSSAT_COEFFICIENTS.as_arrays())
    leaves, treedef = jax.tree_util.tree_flatten(p.coefficients)
    x0 = np.array([float(v) for v in leaves])

    def with_x(x):
        return p.replace(
            coefficients=jax.tree_util.tree_unflatten(treedef, [x[k] for k in range(len(leaves))])
        )

    jac = np.asarray(jax.jit(jax.jacfwd(lambda x: _losses(with_x(x), f)[0]))(a(x0)))
    assert jac.shape == (3, len(leaves)) and len(leaves) > 100
    assert np.all(np.isfinite(jac))
    acting = np.flatnonzero(np.any(jac != 0.0, axis=0))
    assert acting.size >= 20, acting.size
    top = np.argsort(-np.abs(jac[2] * np.maximum(np.abs(x0), 1e-12)))[:5]
    h = 1e-6 * np.maximum(np.abs(x0[top]), 1.0)
    pts = np.repeat(x0[None], 11, axis=0)
    for i, k in enumerate(top):
        pts[1 + i, k] += h[i]
        pts[6 + i, k] -= h[i]
    vals, (stages, leafno) = jax.jit(jax.vmap(lambda x: _losses(with_x(x), f)))(a(pts))
    vals, stages, leafno = np.asarray(vals), np.asarray(stages), np.asarray(leafno)
    for k in range(1, 11):
        assert np.array_equal(stages[k], stages[0]) and np.array_equal(leafno[k], leafno[0])
    fd = ((vals[1:6] - vals[6:11]) / (2 * h[:, None])).T
    np.testing.assert_allclose(jac[:, top], fd, rtol=2e-4, atol=1e-8 * np.abs(vals[0]).max())
