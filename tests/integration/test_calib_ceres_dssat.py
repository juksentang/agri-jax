"""The calibration harness on a DSSAT reference problem: UFGA8201 treatments 1, 3, 5.

Built exactly as the M2 test (``test_ceres_dssat``: dscsm048 build486, nitrogen off, water
replayed from the run and the SPAM dumps) and batched by ``agrijax.calib.ceres``:

* the batch group (one ``vmap`` over the three treatments, forcings padded by 40 days) at the INP
  cultivar reproduces DSSAT's yield (Summary.OUT HWAM, printed as an integer: within 0.5 %, the
  M2 criterion) and silking / maturity dates on every treatment;
* the twin loss is 0 at the truth and its gradient is finite;
* the gradient-trust report classifies the yield response as measured on rorqual:
  P1 a staircase (AD derivative 0 on the whole scan while the yield moves), G3 and
  RUE smooth at trust level 3, and the same classes in the exact and ste modes. Each mode has its
  own simulator and jit (``group_simulator(..., mode=m).jit()``): a jitted simulator keeps the
  mode it was built in, so one shared jit would test the first mode twice, and calling it
  under another mode's context raises;
* the modes are different programs: on IUAF9901 treatment 3 (7.5 plants m-2), where the RLV
  quantisation reaches the biomass, ``d CWAD / d RUE`` at the INP cultivar differs between the
  ``ste`` and ``exact`` simulators (trust run: 3606.10 and 3604.12 kg ha-1 per g MJ-1), the
  exact one equals the central finite difference of the forward program at a 1e-5 step, and the
  forward outputs are bit-identical;
* the jump fallback: on IUAF9901 treatments 1 and 3 the trust report of yield and final
  biomass per treatment finds G3 jumpy on both (the stage-5 stem floor of MZ_GROSUB, STMWT = SWMIN
  and GROGRN = CARBO, switches on one more day at each jump) and G2 jumpy on treatment 3 only, RUE
  smooth; the CERES gradient plan makes RUE a gradient parameter, G2 hybrid (AD on treatment 1,
  secants on treatment 3), G3 derivative-free, and the pair gradient of the per-treatment twin
  losses is finite.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_ceres_dssat as tcd

from agrijax.calib import (
    Group,
    Objective,
    Target,
    TrustConfig,
    batched_value_and_grad,
    pair_gradient,
    trust_report,
)
from agrijax.calib.ceres import (
    ISTAGE_MATURITY_OUT,
    ISTAGE_SILKING_OUT,
    CeresTreatment,
    ceres_gradient_plan,
    ceres_space,
    group_simulator,
    stack_treatments,
)
from agrijax.calib.objective import evaluate_targets, first_day_index
from agrijax.core.grad import ModeMismatchError, gradient_mode

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"
)

TRNOS = (1, 3, 5)
NAMES = ("P1", "G3", "RUE")
PAD = 40
MODES = ("ste", "exact")


def _stage(tmp_path_factory, data_dir, exp, trnos):
    if not tcd.DSCSM.is_file() or not tcd.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {tcd.DSSAT_ENGINE}")
    tables = data_dir / "dumps" / "tables" / "dssat486"
    if not (tables / f"{exp}_t{trnos[0]:02d}_spam.npz").is_file():
        pytest.skip(f"SPAM dump tables not found under {tables}")
    trs, rows = [], []
    for t in trnos:
        out = tcd.run_reference(exp, t, tmp_path_factory.mktemp(f"cal_{exp}_{t}"))
        p, f, _, row = tcd.simulate(out, t, exp=exp, tables=tables)
        trs.append(CeresTreatment(f"t{t}", p, f, {}))
        rows.append(row)
    return trs, rows


@pytest.fixture(scope="module")
def group(tmp_path_factory, data_dir):
    trs, rows = _stage(tmp_path_factory, data_dir, "UFGA8201", TRNOS)
    space = ceres_space(NAMES)
    n = max(int(np.shape(t.forcing.yrdoy)[0]) for t in trs) + PAD
    params, forcing, state0 = stack_treatments(trs, n)
    sims = {m: group_simulator(space, params, forcing, state0, mode=m).jit() for m in MODES}
    truth = np.asarray(space.get(trs[0].params))
    return space, sims, truth, trs, rows


def test_batch_group_reproduces_dssat(group):
    _, sims, truth, trs, rows = group
    out = sims["ste"](jnp.asarray(truth))
    gw = np.asarray(out["gwad"])[-1]
    silk = np.asarray(first_day_index(out["istage"], ISTAGE_SILKING_OUT)).astype(int)
    mat = np.asarray(first_day_index(out["istage"], ISTAGE_MATURITY_OUT)).astype(int)
    for b, (t, row) in enumerate(zip(trs, rows, strict=True)):
        days = np.asarray(t.forcing.yrdoy)
        assert abs(gw[b] - float(row["HWAM"])) / float(row["HWAM"]) < 0.005, (t.name, gw[b], row["HWAM"])
        assert int(days[silk[b]]) == int(row["ADAT"]) and int(days[mat[b]]) == int(row["MDAT"]), t.name


def test_twin_loss_zero_at_truth(group):
    space, sims, truth, _, _ = group
    sim = sims["ste"]
    tg = (
        Target("yield", "gwad", "final", rel_scale=0.05),
        Target("cwad", "cwad", "final", rel_scale=0.05),
        Target("silk", "istage", "date", scale=2.0, code=ISTAGE_SILKING_OUT),
    )
    obs = {k: np.asarray(v) for k, v in evaluate_targets(tg, sim(jnp.asarray(truth))).items()}
    obj = Objective((Group("ufga", sim, tg, obs),))
    z = space.to_unconstrained(jnp.asarray(truth))
    v, g = batched_value_and_grad(lambda zz: obj(space.to_physical(zz)))(jnp.stack([z, z + 0.05]))
    assert float(v[0]) < 1e-20 and float(v[1]) > 0.0
    assert np.all(np.isfinite(np.asarray(g)))


@pytest.mark.parametrize("mode", MODES)
def test_trust_classes_of_yield(group, mode):
    space, sims, truth, trs, _ = group
    sim = sims[mode]
    assert sim.mode == mode

    def yields(theta):
        return sim(theta)["gwad"][-1]

    rep = trust_report(yields, truth, space.lower, space.upper, NAMES, [t.name for t in trs], TrustConfig())
    for t in trs:
        p1 = rep["params"]["P1"]["outputs"][t.name]
        assert p1["class"] == "step" and p1["ad"] == 0.0 and p1["zero_frac"] == 1.0 and p1["scan_range"] > 0
        for n in ("G3", "RUE"):
            o = rep["params"][n]["outputs"][t.name]
            assert o["class"] == "smooth" and o["level"] == 3, (n, t.name, o)


@pytest.fixture(scope="module")
def iuaf(tmp_path_factory, data_dir):
    trs, _ = _stage(tmp_path_factory, data_dir, "IUAF9901", (1, 3))
    n = max(int(np.shape(t.forcing.yrdoy)[0]) for t in trs) + PAD
    return trs, stack_treatments(trs, n)


def test_modes_are_different_programs(iuaf):
    trs, (params, forcing, state0) = iuaf
    space = ceres_space(("RUE",))
    sims = {m: group_simulator(space, params, forcing, state0, mode=m).jit() for m in MODES}
    th = jnp.asarray(np.asarray(space.get(trs[0].params)))
    outs = {m: sims[m](th) for m in MODES}
    for k in ("lai", "cwad", "gwad", "istage"):
        assert np.asarray(outs["ste"][k]).tobytes() == np.asarray(outs["exact"][k]).tobytes(), k

    def cwad(sim):  # treatment 3
        return lambda t: sim(t)["cwad"][-1, 1]

    g = {m: float(jax.grad(cwad(sims[m]))(th)[0]) for m in MODES}
    h = 1e-5 * float(space.width[0])
    fd = (float(cwad(sims["ste"])(th + h)) - float(cwad(sims["ste"])(th - h))) / (2 * h)
    assert abs(g["ste"] - g["exact"]) > 1.0, g  # measured: 3606.10 vs 3604.12
    assert abs(g["exact"] - fd) / abs(fd) < 1e-6 < abs(g["ste"] - fd) / abs(fd), (g, fd)
    with gradient_mode("exact"), pytest.raises(ModeMismatchError):  # a ste simulator refuses exact
        jax.grad(cwad(sims["ste"]))(th)


def test_jumpy_pairs_fall_back(iuaf):
    trs, (params, forcing, state0) = iuaf
    names = ("G2", "G3", "RUE")
    space = ceres_space(names)
    sim = group_simulator(space, params, forcing, state0, mode="ste").jit()
    truth = np.asarray(space.get(trs[0].params))

    def f(theta):
        o = sim(theta)
        return jnp.concatenate([o["gwad"][-1], o["cwad"][-1]])

    tn = [t.name for t in trs]
    outs = [f"{t}/yield" for t in tn] + [f"{t}/cwad" for t in tn]
    rep = trust_report(f, truth, space.lower, space.upper, names, outs, TrustConfig())
    plan = ceres_gradient_plan(rep, {t: [f"{t}/yield", f"{t}/cwad"] for t in tn})
    assert plan.method == {"G2": "hybrid", "G3": "derivative_free", "RUE": "gradient"}, plan.table()
    np.testing.assert_array_equal(plan.use_ad, [[True, False, True], [False, False, True]])
    tg = (Target("yield", "gwad", "final", rel_scale=0.05), Target("cwad", "cwad", "final", rel_scale=0.05))
    obs = {k: np.asarray(v) for k, v in evaluate_targets(tg, sim(jnp.asarray(truth))).items()}
    obj = Objective((Group("iuaf", sim, tg, obs, treatments=tuple(tn)),))
    z0 = space.to_unconstrained(jnp.asarray(truth * np.array([1.05, 0.95, 1.05])))
    losses = jax.jit(lambda z: obj.treatment_losses(space.to_physical(z)))
    dz = 0.1
    v, g = pair_gradient(losses, plan.secant_mask, dz)(z0)
    g = np.asarray(g)
    assert float(v) > 0.0 and np.all(np.isfinite(g)) and np.all(g != 0.0)
    # the secant components are central finite differences of the losses at the same step:
    # G3 (secant on both treatments) = the full-loss FD; G2 = AD on t1 + the FD of t3's loss
    e = [jnp.zeros(3).at[i].set(dz) for i in range(3)]
    atol = 1e-10 * float(np.abs(g).max())  # summation order of the per-treatment terms
    fd = [(np.asarray(losses(z0 + e[i])) - np.asarray(losses(z0 - e[i]))) / (2 * dz) for i in range(3)]
    np.testing.assert_allclose(g[1], fd[1].sum(), rtol=1e-10, atol=atol)
    ad_t1 = float(jax.grad(lambda z: losses(z)[0])(z0)[0])
    np.testing.assert_allclose(g[0], ad_t1 + fd[0][1], rtol=1e-10, atol=atol)
    np.testing.assert_allclose(
        g[2], float(jax.grad(lambda z: jnp.sum(losses(z)))(z0)[2]), rtol=1e-10, atol=atol
    )
