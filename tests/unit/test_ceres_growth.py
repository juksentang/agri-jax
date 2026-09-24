"""CERES-Maize growth (``ceres_stress``, ``ceres_growth``) and the assembled day.

Independent checks, all data-free:

* the stress kernels ``water_stress_factors`` and ``saturation_factor`` and the table helpers
  ``tabex`` / ``curv_lin`` against plain-Python transcriptions of the Fortran (loops, ``if``);
* conservation of assimilate: on every growing day of synthetic seasons the organ increments
  recovered from the state trajectory sum to the day's ``CARBO`` (stage 2: leaf + root; stage 3:
  leaf + stem + ear + root; stage 4: ear + stem + root; stage 5: grain + stem + root while stem
  growth is positive), and the root weight follows ``RTWT + 0.5 GRORT - 0.005 RTWT``;
* the leaf pool in the organ queue: ``aggregate(leaf, PLTPOP)`` gives the total (green +
  senesced) leaf area index and ``LAI = (PLA - SENLA) PLTPOP 1e-4``;
* runtime behaviour: jit equals eager, ``vmap`` over a parameter batch equals a loop, identical
  crops on the ``n_crop`` axis equal one crop, and ``AGRI_JAX_CHECK=1`` accepts the declared writes.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import registry, run
from agrijax.core.organs import aggregate
from agrijax.core.process import CHECK_ENV
from agrijax.processes.crop.ceres_maize import (
    CeresForcing,
    CeresMaizeState,
    ceres_growth,
    ceres_maize_model,
    ceres_phenology,
    ceres_roots,
    ceres_stress,
    saturation_factor,
    water_stress_factors,
)
from agrijax.processes.crop.ceres_maize._util import curv_lin, tabex

from .test_ceres_phenology import DLAYR, SPE, a, make_params, season

X64 = jax.config.jax_enable_x64
TOL = 1e-9 if X64 else 5e-4


# ------------------------------------------------------------------ kernel references
def ref_water_stress(eop: float, trwup: float, rwuep1: float) -> tuple[float, float]:
    swfac = 1.0
    turfac = 1.0
    if eop > 0.0:
        ep1 = eop * 0.1
        if trwup / ep1 < rwuep1:
            turfac = (1.0 / rwuep1) * trwup / ep1
        if ep1 >= trwup:
            swfac = trwup / ep1
    return swfac, int(turfac * 1000) / 1000


def ref_satfac(sw, sat, dlayr, rlv, tss, pormin):
    tss = list(tss)
    sumex = sumrl = 0.0
    for l in range(len(sw)):
        tss[l] = 0.0 if sat[l] - sw[l] >= pormin else tss[l] + 1.0
        swexf = max((sat[l] - sw[l]) / pormin, 0.0) if tss[l] > 2.0 else 1.0
        swexf = min(swexf, 1.0)
        sumex += dlayr[l] * rlv[l] * (1.0 - swexf)
        sumrl += dlayr[l] * rlv[l]
    satfac = sumex / sumrl if sumrl > 0.0 else 0.0
    return min(max(satfac, 0.0), 1.0), tss


def ref_tabex(val, arg, x):
    k = len(arg)
    j = k - 1
    for jj in range(1, k):
        if not x > arg[jj]:
            j = jj
            break
    return (x - arg[j - 1]) * (val[j] - val[j - 1]) / (arg[j] - arg[j - 1]) + val[j - 1]


def ref_curv(xb, x1, x2, xm, x):
    c = 0.0
    if xb < x < x1:
        c = (x - xb) / (x1 - xb)
    if x1 <= x <= x2:
        c = 1.0
    if x2 < x < xm:
        c = 1.0 - (x - x2) / (xm - x2)
    return min(max(c, 0.0), 1.0)


def test_water_stress_factors_equal_reference():
    rng = np.random.default_rng(0)
    eop = np.concatenate([[0.0, 0.0, 5.0, 5.0, 5.0], rng.uniform(0.0, 9.0, 300)])
    trwup = np.concatenate([[0.0, 0.3, 0.5, 0.2, 0.0], rng.uniform(0.0, 1.2, 300)])
    sw, tu = water_stress_factors(a(eop), a(trwup), a(1.5))
    for i in range(len(eop)):
        ws, wt = ref_water_stress(float(eop[i]), float(trwup[i]), 1.5)
        assert float(sw[i]) == pytest.approx(ws, rel=TOL, abs=TOL)
        assert float(tu[i]) == pytest.approx(wt, abs=1e-3 if not X64 else 1e-12)


def test_saturation_factor_equals_reference_loop():
    rng = np.random.default_rng(1)
    nl = len(DLAYR)
    sat = np.full(nl, 0.40)
    for _ in range(50):
        sw = sat - rng.uniform(0.0, 0.12, nl)
        rlv = rng.uniform(0.0, 3.0, (2, nl)) * (rng.random((2, nl)) < 0.8)
        tss = rng.integers(0, 5, (2, nl)).astype(float)
        got_s, got_t = saturation_factor(a(sw), a(sat), a(DLAYR), a(rlv), a(tss), a(0.05))
        for c in range(2):
            ws, wt = ref_satfac(list(sw), list(sat), DLAYR, list(rlv[c]), list(tss[c]), 0.05)
            assert float(got_s[c]) == pytest.approx(ws, rel=TOL, abs=TOL)
            np.testing.assert_array_equal(np.asarray(got_t[c]), wt)


def test_tabex_and_curv_equal_reference():
    co2x, co2y = SPE["co2x"], SPE["co2y"]
    for x in [-50.0, 0.0, 100.0, 220.0, 250.0, 330.0, 331.0, 400.0, 555.5, 990.0, 5000.0, 12000.0]:
        assert float(tabex(a(co2y), a(co2x), a(x))) == pytest.approx(
            ref_tabex(co2y, co2x, x), rel=TOL, abs=TOL
        )
    xb, x1, x2, xm = SPE["prftc"]
    for x in np.linspace(0.0, 50.0, 101):
        assert float(curv_lin(a(xb), a(x1), a(x2), a(xm), a(x))) == pytest.approx(
            ref_curv(xb, x1, x2, xm, float(x)), rel=TOL, abs=TOL
        )


# ------------------------------------------------------------------ synthetic seasons through the model
def season_forcing(
    seed: int, *, stress: bool = False, waterlog: bool | None = None, **kw
) -> tuple[CeresForcing, dict]:
    w = season(seed, **kw)
    n = len(w["yrdoy"])
    rng = np.random.default_rng(seed + 1000)
    swfac = np.ones(n)
    turfac = np.ones(n)
    sw = w["sw"].copy()
    if stress:
        dry = (np.arange(n) % 37) > 22
        swfac = np.where(dry, rng.uniform(0.05, 0.9, n), 1.0)
        turfac = np.floor(np.where(dry, rng.uniform(0.0, 0.9, n), 1.0) * 1000) / 1000
    if stress if waterlog is None else waterlog:
        wet = (np.arange(n) % 29) < 5
        sw[wet, :3] = 0.225  # near saturation in the top layers
    f = CeresForcing(
        yrdoy=jnp.asarray(w["yrdoy"]),
        tmax=a(w["tmax"]),
        tmin=a(w["tmin"]),
        srad=a(w["srad"]),
        dayl=a(w["dayl"]),
        twilen=a(w["twilen"]),
        co2=a(np.full(n, 380.0)),
        snow=a(w["snow"]),
        sw=a(sw),
        swfac=a(swfac),
        turfac=a(turfac),
    )
    return f, w


_STATE_MODEL = ceres_maize_model(outputs=lambda s, p, f: s)
_RUN_STATES = jax.jit(lambda p, f, s: run(_STATE_MODEL, p, f, s))  # one compilation per shape


def _states(params, forcing, n_crop=1):
    return _RUN_STATES(params, forcing, CeresMaizeState.initial(params, n_crop))


@pytest.mark.parametrize(("seed", "stress"), [(11, False), (12, True), (13, True)])
def test_assimilate_is_conserved_by_partitioning(seed, stress):
    f, w = season_forcing(seed, stress=stress)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s = _states(p, f)
    g, ph = s.growth, s.phen
    lfwt = np.asarray(g.leaf.mass[:, 0, 0])
    stmwt = np.asarray(g.stmwt[:, 0])
    earwt = np.asarray(g.earwt[:, 0])
    grnwt = np.asarray(g.grnwt[:, 0])
    rtwt = np.asarray(g.rtwt[:, 0])
    slan = np.asarray(g.slan[:, 0])
    carbo = np.asarray(g.carbo[:, 0])
    grort = np.asarray(g.grort[:, 0])
    seedrv = np.asarray(g.seedrv[:, 0])
    stage = np.asarray(ph.istage[:, 0])
    swmax = np.asarray(g.swmax[:, 0])
    checked = {2: 0, 3: 0, 4: 0, 5: 0}
    for t in range(1, len(stage)):
        st = stage[t]
        prev_st = stage[t - 1]
        if st != prev_st or carbo[t] <= 0.001 or st not in checked:
            continue  # growth ran in the new stage's block on a transition day; skip for clarity
        dl = lfwt[t] - lfwt[t - 1]
        ds = stmwt[t] - stmwt[t - 1]
        de = earwt[t] - earwt[t - 1]
        dg = grnwt[t] - grnwt[t - 1]
        # root weight integration
        np.testing.assert_allclose(
            rtwt[t], max(rtwt[t - 1] + 0.5 * grort[t] - 0.005 * rtwt[t - 1], 0.0), rtol=TOL, atol=TOL
        )
        if st == 2:
            total = dl + slan[t] / 600.0 + grort[t]
        elif st == 3:
            total = dl + slan[t] / 600.0 + ds + de + grort[t]
        elif st == 4:
            total = de + ds + grort[t]
        else:
            np.testing.assert_allclose(de, dg, rtol=TOL, atol=TOL)  # grain growth goes to the ear
            if dg > carbo[t]:
                continue  # grain fed from the stem (GROSTM < 0): no root growth that day
            # GROSTM = CARBO - GROGRN >= 0 is split half stem (capped at SWMAX), half root
            total = dg + 2.0 * grort[t]
            assert stmwt[t] <= swmax[t] + 1e-12
        np.testing.assert_allclose(total, carbo[t], rtol=TOL * 100, atol=TOL, err_msg=f"stage {st} day {t}")
        checked[st] += 1
    assert all(v > 3 for v in checked.values()), checked
    assert np.all(np.diff(seedrv[stage == 3]) == 0.0)  # the seed reserve is only used in stage 1


def test_leaf_pool_lives_in_the_organ_queue():
    f, w = season_forcing(14)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s = _states(p, f)
    g = s.growth
    t = -1
    last = jax.tree_util.tree_map(lambda x: x[t], g)
    agg = aggregate(last.leaf, last.pltpop)
    pla = float(last.leaf.area[0, 0])
    assert bool(last.leaf.alive[0, 0]) and int(last.leaf.n_active[0]) == 1
    assert float(agg["lai"][0]) == pytest.approx(pla * float(last.pltpop[0]) * 1e-4, rel=TOL)
    assert float(last.lai[0]) == pytest.approx(
        (pla - float(last.senla[0])) * float(last.pltpop[0]) * 1e-4, rel=TOL
    )
    assert float(agg["leaf_mass"][0]) == pytest.approx(
        float(last.leaf.mass[0, 0]) * float(last.pltpop[0]) * 10.0, rel=TOL
    )
    # before emergence no cohort has appeared
    first = jax.tree_util.tree_map(lambda x: x[0], g)
    assert int(first.leaf.n_active[0]) == 0


def test_whole_season_is_finite_and_reaches_maturity():
    f, w = season_forcing(15, stress=True)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s = _states(p, f)
    for leaf in jax.tree_util.tree_leaves(s):
        assert np.all(np.isfinite(np.asarray(leaf, dtype=float)))
    stage = np.asarray(s.phen.istage[:, 0])
    assert stage[-1] == 10 and int(s.phen.crop_status[-1, 0]) == 1
    assert float(s.growth.grnwt[-1, 0]) > 0.0
    assert float(s.stress.satfac[:, 0].max()) > 0.0  # waterlogging days exercised the SATFAC path


def test_jit_equals_eager_and_vmap_equals_loop():
    f, w = season_forcing(16, stress=True, n=180)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    model = ceres_maize_model()
    s0 = CeresMaizeState.initial(p, 1)
    eager = run(model, p, f, s0)
    jitted = jax.jit(lambda p_, f_, s_: run(model, p_, f_, s_))(p, f, s0)
    for k in eager:
        np.testing.assert_allclose(np.asarray(eager[k]), np.asarray(jitted[k]), rtol=1e-12 if X64 else 1e-5)
    g3s = np.array([6.0, 8.17, 10.0])
    batch = jax.tree_util.tree_map(lambda x: jnp.stack([x] * 3), p)
    batch = batch.replace(cultivar=batch.cultivar.replace(g3=a(g3s)))
    vm = jax.jit(jax.vmap(lambda p_: run(model, p_, f, s0)))(batch)
    for i, g3 in enumerate(g3s):
        pi = p.replace(cultivar=p.cultivar.replace(g3=a(g3)))
        one = run(model, pi, f, s0)
        np.testing.assert_allclose(
            np.asarray(vm["gwad"][i]), np.asarray(one["gwad"]), rtol=1e-10 if X64 else 1e-4
        )
    assert float(vm["gwad"][0][-1, 0]) < float(vm["gwad"][2][-1, 0])


def test_identical_crops_on_the_n_crop_axis_equal_one_crop():
    f, w = season_forcing(17, stress=True, n=200)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    model = ceres_maize_model()
    one = run(model, p, f, CeresMaizeState.initial(p, 1))
    two = run(model, p, f, CeresMaizeState.initial(p, 2))
    for k in one:
        # equal to rounding: XLA may vectorise a length-2 axis differently from a length-1 axis
        tol = 1e-12 if X64 else 1e-5
        np.testing.assert_allclose(np.asarray(two[k])[:, 0], np.asarray(one[k])[:, 0], rtol=tol, atol=tol)
        np.testing.assert_allclose(np.asarray(two[k])[:, 1], np.asarray(one[k])[:, 0], rtol=tol, atol=tol)


def test_declared_writes_hold_under_agri_jax_check(monkeypatch):
    monkeypatch.setenv(CHECK_ENV, "1")
    f, w = season_forcing(18, stress=True, n=150)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s = CeresMaizeState.initial(p, 2)
    procs = [ceres_phenology, ceres_stress, ceres_growth, ceres_roots]
    for t in range(60):  # eager calls: the writes check compares concrete values
        ft = jax.tree_util.tree_map(lambda x, t=t: x[t], f)
        for proc in procs:
            s = proc(s, p, ft)
    assert int(s.phen.istage[0]) != 7


def test_processes_are_registered_with_their_fortran_names():
    assert registry["ceres_phenology"].fortran_name == "MZ_PHENOL"
    assert registry["ceres_growth"].fortran_name == "MZ_GROSUB"
    assert registry["ceres_stress"].fortran_name == "MZ_GROSUB"
    assert registry["ceres_roots"].fortran_name == "MZ_ROOTGR"
    for name in ("ceres_phenology", "ceres_growth", "ceres_stress", "ceres_roots"):
        assert "DSSAT-CSM" in registry[name].source and "BSD-3" in registry[name].source


def test_cold_failure_and_drought_failure_paths():
    """ICOLD >= CDAY ends the crop (status 32); more than 10 days of SWFAC <= 0.1 with LAI <= 0.1
    before silking ends it too (status 33)."""
    f, w = season_forcing(19, n=120)
    p = make_params(yrplt=int(w["yrdoy"][0]))
    cold = f.replace(
        tmin=jnp.where(jnp.arange(120) > 25, -1.0, f.tmin), tmax=jnp.where(jnp.arange(120) > 25, 15.0, f.tmax)
    )
    s = _states(p, cold)
    assert 32 in set(np.asarray(s.phen.crop_status[:, 0]).tolist())
    dry = f.replace(
        swfac=jnp.where(jnp.arange(120) > 12, 0.05, 1.0), turfac=jnp.where(jnp.arange(120) > 12, 0.0, 1.0)
    )
    s2 = _states(p, dry)
    assert 33 in set(np.asarray(s2.phen.crop_status[:, 0]).tolist())
    for leaf in jax.tree_util.tree_leaves((s, s2)):
        assert np.all(np.isfinite(np.asarray(leaf, dtype=float)))


def test_slow_grain_fill_triggers_early_maturity():
    """Stage 5 with RGFILL <= RSGR for more than RSGRT days sets SUMDTT = P5 (early maturity)."""
    f, w = season_forcing(20, n=260)
    p = make_params(yrplt=int(w["yrdoy"][0]))
    s = _states(p, f)
    stage = np.asarray(s.phen.istage[:, 0])
    t5 = int(np.argmax(stage == 5))
    cool = f.replace(
        tmax=jnp.where(jnp.arange(260) > t5 + 3, 8.0, f.tmax),
        tmin=jnp.where(jnp.arange(260) > t5 + 3, 3.0, f.tmin),
    )
    s2 = _states(p, cool)
    st2 = np.asarray(s2.phen.istage[:, 0])
    assert (st2 == 6).any() and np.argmax(st2 == 6) <= t5 + 3 + int(SPE["rsgrt"]) + 3
    assert math.isfinite(float(s2.growth.grnwt[-1, 0]))
