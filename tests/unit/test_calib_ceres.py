"""The CERES-Maize adapter of the calibration harness on synthetic seasons (no data).

* A batch group (``stack_treatments`` + ``group_simulator``, one ``vmap`` over treatments with
  forcings padded to a common length) gives, on every treatment's own days, the outputs of the
  model run on that treatment alone: the same stage on every day, and LAI / biomass / yield to
  rounding (the batched program may order a reduction differently). Measured on rorqual in
  float64: on these synthetic seasons up to 5 ulp (6.0e-16 relative) on LAI
  and 1 ulp on CWAD and GWAD; on the DSSAT problems GWAD differs by up to 4 ulp
  (IUAF9901 t3 on 75 days, and a 2- vs a 1-treatment group). The test allows ``RTOL`` (1e-12).
* ``pad_forcing`` appends consecutive dates across the year end.
* The twin loss is 0 at the true cultivar, its AD gradient is finite, non-zero for G2, G3 and RUE
  and exactly 0 for P1 (P1 acts only through the whole-day end of the juvenile phase, nitrogen
  off; ``test_ceres_grad``), in the default ``ste`` mode and in ``exact`` mode, each with its own
  simulator.
* A simulator is one gradient mode (a required argument): per-mode simulators (jitted) give
  bit-identical outputs but different derivatives of the root length (RLV is stored to 1e-3 by
  ``trunc_st``: identity derivative in ``ste``, 0 in ``exact``), and a simulator called inside a
  ``gradient_mode`` context of the other mode raises (fresh trace, or its ``jit()`` on a cache
  hit) instead of silently running its own mode.
* The CUL bounds of :data:`CERES_SPECS` contain the test cultivar and lie inside every declared
  valid domain; the default plan is derivative-free for P1, P2, P5 and PHINT.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib import Group, Objective, Target, batched_value_and_grad
from agrijax.calib.ceres import (
    CERES_SPECS,
    ISTAGE_MATURITY_OUT,
    ISTAGE_SILKING_OUT,
    CeresTreatment,
    calib_outputs,
    ceres_gradient_plan,
    ceres_space,
    group_simulator,
    pad_forcing,
    stack_treatments,
)
from agrijax.calib.objective import evaluate_targets
from agrijax.core import run
from agrijax.core.grad import ModeMismatchError, gradient_mode
from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import make_params

NAMES = ("P1", "P5", "G2", "G3", "PHINT", "RUE")
MODES = ("ste", "exact")
X64 = jax.config.jax_enable_x64
RTOL = 1e-12 if X64 else 1e-5


def _treatments() -> list[CeresTreatment]:
    out = []
    for seed, stress, n_cut in ((31, False, 0), (32, True, 17)):
        f, w = season_forcing(seed, stress=stress, waterlog=False)
        if n_cut:  # a shorter season: the batch pads it
            f = jax.tree_util.tree_map(lambda x, n_cut=n_cut: x[:-n_cut], f)
        p = make_params(yrplt=int(w["yrdoy"][2]))
        out.append(CeresTreatment(f"s{seed}", p, f, {}))
    return out


TREATMENTS = _treatments()
SPACE = ceres_space(NAMES)


def test_bounds_contain_the_cultivar():
    theta = np.asarray(SPACE.get(TREATMENTS[0].params))
    assert np.all(theta >= SPACE.lower) and np.all(theta <= SPACE.upper)
    assert CERES_SPECS["P1"].bounds_source.startswith("DSSAT-CSM v4.8.6.0 Genotype/MZCER048.CUL")
    assert SPACE.violations(theta) == []
    for name in ("P5", "G2", "PHINT"):  # divisors: open at 0
        d = CERES_SPECS[name].domain
        assert d.lower == 0.0 and d.lower_open and "divides" in d.source, name


def test_default_plan_is_derivative_free_for_phenology():
    ok = {"class": "smooth", "level": 3, "n_jumps": 0}
    names = ("P1", "P2", "P5", "PHINT", "G2", "G3", "RUE")
    report = {"outputs": ["t/y"], "params": {n: {"outputs": {"t/y": ok}} for n in names}}
    plan = ceres_gradient_plan(report)
    assert plan.derivative_free_params == ("P1", "P2", "P5", "PHINT")
    assert plan.gradient_params == ("G2", "G3", "RUE")
    assert ceres_gradient_plan(report, allow_gradient=("P5",)).method["P5"] == "gradient"


def test_pad_forcing_dates_cross_the_year():
    f = TREATMENTS[0].forcing
    f2 = f.replace(yrdoy=jnp.asarray(np.array([2001363 + k for k in range(3)] + [2002001]), dtype=jnp.int32))
    f2 = jax.tree_util.tree_map(lambda x: x[:4], f2)
    g = pad_forcing(f2, 7)
    np.testing.assert_array_equal(
        np.asarray(g.yrdoy), [2001363, 2001364, 2001365, 2002001, 2002002, 2002003, 2002004]
    )
    np.testing.assert_array_equal(np.asarray(g.tmax)[4:], np.full(3, float(f2.tmax[-1])))
    with pytest.raises(ValueError):
        pad_forcing(f2, 3)


def test_group_equals_single_runs():
    params, forcing, state0 = stack_treatments(TREATMENTS, n_days=270)
    sim = group_simulator(SPACE, params, forcing, state0, mode="ste").jit()
    theta = SPACE.get(TREATMENTS[0].params)
    out = sim(theta)
    model = ceres_maize_model(outputs=calib_outputs)
    for b, t in enumerate(TREATMENTS):
        ref = jax.jit(lambda p, f, s: run(model, p, f, s))(
            t.params, t.forcing, CeresMaizeState.initial(t.params, 1)
        )
        n = int(np.shape(t.forcing.yrdoy)[0])
        np.testing.assert_array_equal(np.asarray(out["istage"])[:n, b], np.asarray(ref["istage"])[:, 0])
        for k in ("lai", "cwad", "gwad"):  # vmap may reorder a reduction: a few ulp (see the docstring)
            np.testing.assert_allclose(
                np.asarray(out[k])[:n, b], np.asarray(ref[k])[:, 0], rtol=RTOL, err_msg=k
            )
    assert np.asarray(out["gwad"]).shape == (270, 2)


def _objective(mode: str):
    params, forcing, state0 = stack_treatments(TREATMENTS)
    sim = group_simulator(SPACE, params, forcing, state0, mode=mode)
    assert sim.mode == mode
    truth = np.asarray(SPACE.get(TREATMENTS[0].params))
    tg = (
        Target("yield", "gwad", "final", rel_scale=0.05),
        Target("cwad", "cwad", "final", rel_scale=0.05),
        Target("silk", "istage", "date", scale=2.0, code=ISTAGE_SILKING_OUT),
        Target("mat", "istage", "date", scale=2.0, code=ISTAGE_MATURITY_OUT),
    )
    obs = {k: np.asarray(v) for k, v in evaluate_targets(tg, sim(jnp.asarray(truth))).items()}
    assert np.all(obs["mat"] < np.shape(forcing.yrdoy)[1])  # both seasons mature
    return Objective((Group("synthetic", sim, tg, obs),)), truth


@pytest.mark.parametrize("mode", ["ste", "exact"])
def test_twin_loss_gradient(mode):
    obj, truth = _objective(mode)

    def loss_z(z):
        return obj(SPACE.to_physical(z))

    z_true = SPACE.to_unconstrained(jnp.asarray(truth))
    z = np.stack([np.asarray(z_true), np.asarray(z_true) + np.array([0.0, 0.0, 0.05, -0.05, 0.0, 0.05])])
    v, g = batched_value_and_grad(loss_z)(jnp.asarray(z))
    v, g = np.asarray(v), np.asarray(g)
    # the loss at the truth is 0 up to the logit round trip of theta
    assert v[0] == pytest.approx(0.0, abs=1e-18 if X64 else 1e-8) and v[1] > v[0]
    assert np.all(np.isfinite(g))
    col = {n: k for k, n in enumerate(NAMES)}
    assert g[1, col["P1"]] == 0.0
    for n in ("G2", "G3", "RUE"):
        assert g[1, col[n]] != 0.0, n


def _rlv_outputs(state, params, forcing_t):
    out = calib_outputs(state, params, forcing_t)
    out["rlv"] = jnp.sum(state.roots.rlv, axis=-1)  # summed over layers, [n_crop]
    return out


def test_simulators_are_per_mode():
    f, w = season_forcing(33, stress=True)
    t = CeresTreatment("dry", make_params(yrplt=int(w["yrdoy"][2])), f, {})
    params, forcing, state0 = stack_treatments([t])
    space = ceres_space(("RUE", "G3"))
    theta = space.get(t.params)
    with pytest.raises(TypeError):
        group_simulator(space, params, forcing, state0)  # type: ignore[call-arg]  # the mode is required
    sims = {
        m: group_simulator(space, params, forcing, state0, outputs=_rlv_outputs, mode=m).jit() for m in MODES
    }

    def root_length(sim):  # RLV is stored to 1e-3 (trunc_st): identity derivative in ste, 0 in exact
        return lambda th: sim(th)["rlv"][-1, 0]

    out = {m: sims[m](theta) for m in sims}
    for k in ("lai", "cwad", "gwad", "istage", "rlv"):
        assert np.asarray(out["ste"][k]).tobytes() == np.asarray(out["exact"][k]).tobytes(), k
    g = {m: np.asarray(jax.grad(root_length(sims[m]))(theta)) for m in sims}
    assert g["ste"][0] > 0.0 and np.all(g["exact"] == 0.0), g
    for inner, outer in (("ste", "exact"), ("exact", "ste")):
        with gradient_mode(outer):
            with pytest.raises(ModeMismatchError):  # cached jit: checked on the call
                sims[inner](theta)
            with pytest.raises(ModeMismatchError):  # a fresh outer trace
                jax.jit(jax.grad(root_length(sims[inner])))(theta)
        with gradient_mode(inner):  # the matching context is fine
            np.testing.assert_array_equal(np.asarray(jax.grad(root_length(sims[inner]))(theta)), g[inner])
