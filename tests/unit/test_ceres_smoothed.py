"""The smoothed CERES-Maize phenology and growth (``crop/ceres_maize.*@dssat-4.8.6.0:alt_smoothed``).

Independent references:

* the faithful model itself: as the gate scale tends to 0 every logistic gate is the reference's step,
  so the smoothed day must reproduce the faithful day (same integer stages and stage dates on every
  day, the same yield, LAI and biomass to rounding) on a synthetic season that reaches maturity, with
  ``YRDOY`` strictly increasing (the limit does not hold on days that repeat a date: the faithful
  ``SUMP`` reset at ``YRDOY == STGDOY(3)`` fires on every repeated day, the soft stage-4 sums do not),
  also after a crop failure (a drought failure before silking, a cold failure after silking in stage 4
  and in stage 5: the failure forces the soft passages, the stage-6 clock is the carried ``SUMDTT``);
* the structure of the soft passages: each is a probability (in ``[0, 1]``), never decreases over the
  season and never exceeds the passage of the boundary before it, and the expected passage day
  (the sum of ``1 - passage``) moves towards the hard stage date as the scale shrinks;
* an early maturity (a cold spell in effective grain filling: growth's ``SUMDTT = P5`` or the stage-6
  ``DTT < 2`` rule) while the soft clock is still below ``P5``: the forced passages stay at 1 after the
  hard maturity, and at the vanishing scale the soft dates equal the hard ones;
* derivatives: with a finite scale the yield has a finite, nonzero derivative with respect to every
  phenology coefficient (``P1``, ``P2``, ``P5``, ``PHINT``) where the faithful model's is zero, and AD
  agrees with central differences at small steps.

The faithful processes never read ``state.soft`` or ``params.smoothing`` (both ``None`` by default).
The limit and derivative checks are float64 comparisons (skipped in the float32 tier); the gate limits,
the soft-state check and a smoke test of the passage chain run in both precisions.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.model import Model
from agrijax.core.runtime import run
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    CROP_PROCESSES_SMOOTHED,
    REPLAY_PROCESSES,
    CeresMaizeState,
    SmoothingCoefficients,
    ceres_phenology_smoothed,
    with_soft_state,
)
from agrijax.processes.crop.ceres_maize.smoothed import gate, soft_clock, stage_occupancy
from agrijax.processes.crop.ceres_maize.smoothed_params import PASSAGE_NAMES
from agrijax.testing.conformance._builtin.crop import params as crop_params
from agrijax.testing.conformance._builtin.crop import weather

X64 = jax.config.jax_enable_x64
DTYPE = jnp.float64 if X64 else jnp.float32


def x64_only(fn: Any) -> Any:
    """Skip a float64 comparison in the float32 tier (an allowed skip under AGRI_JAX_NO_SKIP=1)."""
    fn = pytest.mark.skipif(not X64, reason="the limit and derivative checks are float64 comparisons")(fn)
    return pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier")(fn)


N_DAYS = 200  # long enough for the synthetic season to reach maturity
SEEDS = (3, 11)
FAITHFUL = Model(CeresMaizeState, [*REPLAY_PROCESSES, *CROP_PROCESSES], outputs=lambda s, p, f: s)
SMOOTHED = Model(CeresMaizeState, [*REPLAY_PROCESSES, *CROP_PROCESSES_SMOOTHED], outputs=lambda s, p, f: s)


def _inputs(seed: int) -> tuple[Any, Any, Any]:
    dtype = DTYPE
    f = weather(seed, dtype, n=N_DAYS)
    p = crop_params(dtype, int(np.asarray(f.yrdoy)[2]))
    return p, f, CeresMaizeState.initial(p, 1, dtype=dtype)


def _smoothed_params(p: Any, width: float) -> Any:
    sm = SmoothingCoefficients().as_arrays(DTYPE)
    return p.replace(smoothing=sm.replace(width=jnp.asarray(width, DTYPE)))


@jax.jit
def _run_faithful(p: Any, f: Any, s0: Any) -> Any:
    return run(FAITHFUL, p, f, s0)


@jax.jit
def _run_smoothed(p: Any, f: Any, s0: Any) -> Any:
    return run(SMOOTHED, p, f, with_soft_state(s0))


def _yield(traj: Any) -> Any:
    return traj.growth.grnwt[-1, 0] * traj.phen.ears[-1, 0]


@x64_only
@pytest.mark.parametrize("seed", SEEDS)
def test_vanishing_scale_reproduces_the_faithful_model(seed: int) -> None:
    p, f, s0 = _inputs(seed)
    ref = _run_faithful(p, f, s0)
    got = _run_smoothed(_smoothed_params(p, 1e-12), f, s0)
    ist = np.asarray(ref.phen.istage[:, 0])
    assert ist[-1] == 10, "the synthetic season must reach maturity"
    np.testing.assert_array_equal(np.asarray(got.phen.istage[:, 0]), ist)
    np.testing.assert_array_equal(np.asarray(got.phen.stgdoy[-1]), np.asarray(ref.phen.stgdoy[-1]))
    np.testing.assert_array_equal(np.asarray(got.phen.mdate[-1]), np.asarray(ref.phen.mdate[-1]))
    for name, a, b in (
        ("yield", _yield(got), _yield(ref)),
        ("gpp", got.phen.gpp[-1], ref.phen.gpp[-1]),
        ("lai", got.growth.lai, ref.growth.lai),
        ("biomass", got.growth.biomas, ref.growth.biomas),
        ("rtwt", got.growth.rtwt, ref.growth.rtwt),
        ("tlno", got.phen.tlno[-1], ref.phen.tlno[-1]),
        ("p3", got.phen.p3[-1], ref.phen.p3[-1]),
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-9, err_msg=name)
    # the soft dates (expected passage days) of the vanishing scale are the hard stage dates
    codes = (1, 2, 3, 4, 5, 6, 10)
    hard = np.asarray([int(np.argmax(ist == k)) for k in codes], dtype=float)
    np.testing.assert_array_equal(np.sum(1.0 - np.asarray(got.soft.passage[:, 0, :]), axis=0), hard)


@x64_only
@pytest.mark.parametrize("seed", SEEDS)
def test_passages_are_ordered_probabilities_and_their_expected_days_converge(seed: int) -> None:
    p, f, s0 = _inputs(seed)
    ref = _run_faithful(p, f, s0)
    ist = np.asarray(ref.phen.istage[:, 0])
    # hard stage codes reached at each passage index (emergence 1, ..., maturity 10)
    codes = (1, 2, 3, 4, 5, 6, 10)
    hard = {k: int(np.argmax(ist == c)) for k, c in zip(PASSAGE_NAMES, codes, strict=True)}
    errs = []
    for width in (2.0, 17.0, 70.0):
        traj = _run_smoothed(_smoothed_params(p, width), f, s0)
        pas = np.asarray(traj.soft.passage[:, 0, :])  # [T, 7]
        assert np.all(np.isfinite(pas))
        assert np.all((pas >= 0.0) & (pas <= 1.0))
        assert np.all(np.diff(pas, axis=0) >= -1e-12), "a passage decreased over the season"
        assert np.all(np.diff(pas[:, 1:], axis=1) <= 1e-12), "a later boundary passed before an earlier one"
        expected = np.sum(1.0 - pas, axis=0)
        errs.append([abs(expected[i] - hard[k]) for i, k in enumerate(PASSAGE_NAMES)])
        occ = np.stack([np.asarray(o) for o in stage_occupancy(jnp.asarray(pas))], axis=-1)
        assert np.all(occ >= -1e-12) and np.all(occ.sum(axis=-1) <= 1.0 + 1e-12)
    errs = np.asarray(errs)  # [width, boundary]
    assert errs[0].max() <= 2.0  # the narrowest gate: within two days of the hard date
    assert errs[0].mean() <= errs[-1].mean() + 0.5  # the widest gate is not closer on average


#: the cold spell that forces an early maturity: starts this many days after the beginning of effective
#: grain filling, lasts COLD_DAYS days (more than RSGRT = 5 slow-fill days, fewer than CDAY = 15 cold
#: days), at TMAX / TMIN [degC] (no thermal time, RGFILL 0)
COLD_START, COLD_DAYS, COLD_TMAX, COLD_TMIN = 3, 9, 5.0, 1.0


def _early_maturity_inputs(seed: int) -> tuple[Any, Any, Any, int]:
    """Seed ``seed``'s season with a cold spell in effective grain filling, and the faithful model's
    maturity day of the unchanged season."""
    p, f, s0 = _inputs(seed)
    ist = np.asarray(_run_faithful(p, f, s0).phen.istage[:, 0])
    d5 = int(np.argmax(ist == 5))
    days = np.arange(N_DAYS)
    cold = jnp.asarray((days >= d5 + COLD_START) & (days < d5 + COLD_START + COLD_DAYS))
    f = f.replace(tmax=jnp.where(cold, COLD_TMAX, f.tmax), tmin=jnp.where(cold, COLD_TMIN, f.tmin))
    return p, f, s0, int(np.argmax(ist == 10))


@x64_only
@pytest.mark.parametrize("seed", SEEDS)
def test_early_maturity_keeps_the_passages_and_the_soft_dates(seed: int) -> None:
    p, f, s0, mat_normal = _early_maturity_inputs(seed)
    ref = _run_faithful(p, f, s0)
    ist = np.asarray(ref.phen.istage[:, 0])
    assert ist[-1] == 10
    mat = int(np.argmax(ist == 10))
    assert mat < mat_normal - COLD_START, (mat, mat_normal)  # the cold spell matured the crop early
    codes = (1, 2, 3, 4, 5, 6, 10)
    hard = np.asarray([int(np.argmax(ist == c)) for c in codes], dtype=float)
    for width in (1e-12, 2.0, 17.0):
        traj = _run_smoothed(_smoothed_params(p, width), f, s0)
        pas = np.asarray(traj.soft.passage[:, 0, :])
        assert np.all(np.isfinite(pas)) and np.all((pas >= 0.0) & (pas <= 1.0))
        assert np.all(np.diff(pas, axis=0) >= -1e-12), f"a passage decreased over the season ({width:g})"
        assert np.all(np.diff(pas[:, 1:], axis=1) <= 1e-12), f"passages out of order ({width:g})"
        ist_s = np.asarray(traj.phen.istage[:, 0])
        assert ist_s[-1] == 10
        mat_s = int(np.argmax(ist_s == 10))  # the smoothed model's own (forced) maturity day
        assert mat_s < mat_normal - COLD_START, (width, mat_s, mat_normal)
        assert np.all(pas[mat_s:, -2:] == 1.0), f"end of grain filling / maturity not kept at 1 ({width:g})"
        if width == 1e-12:
            np.testing.assert_array_equal(np.asarray(traj.phen.istage[:, 0]), ist)
            np.testing.assert_array_equal(np.sum(1.0 - pas, axis=0), hard)  # soft dates = hard dates


#: the drought that fails the crop before silking: no water uptake (TRWUP = 0) for DROUGHT_DAYS days
#: from the day after emergence (seed 3: the faithful crop fails in stage 1 and matures on day 63)
DROUGHT_DAYS = 16
#: the cold spell that fails the crop after silking: COLD_FAIL_DAYS days (at least CDAY = 15) at
#: TMAX / TMIN [degC] (TMIN <= TSEN = 6, still thermal time; RGFILL stays above the slow-fill limit, so no
#: early maturity), from one day after the hard start of stage 4 or 5
COLD_FAIL_DAYS, COLD_FAIL_TMAX, COLD_FAIL_TMIN = 16, 25.0, 5.0
#: passage index -> hard stage code (emergence 1, ..., maturity 10)
CODES = (1, 2, 3, 4, 5, 6, 10)


def _drought_inputs(seed: int) -> tuple[Any, Any, Any]:
    """Seed ``seed``'s season with no water uptake for :data:`DROUGHT_DAYS` days from the day after
    emergence."""
    p, f, s0 = _inputs(seed)
    ist0 = np.asarray(_run_faithful(p, f, s0).phen.istage[:, 0])
    start = int(np.argmax(ist0 == 1)) + 1
    days = np.arange(N_DAYS)
    dry = jnp.asarray((days >= start) & (days < start + DROUGHT_DAYS))
    return p, f.replace(trwup=jnp.where(dry, 0.0, f.trwup)), s0


def _cold_inputs(seed: int, stage: int) -> tuple[Any, Any, Any]:
    """Seed ``seed``'s season with a cold spell of :data:`COLD_FAIL_DAYS` days from one day after the hard
    start of ``stage``."""
    p, f, s0 = _inputs(seed)
    ist0 = np.asarray(_run_faithful(p, f, s0).phen.istage[:, 0])
    start = int(np.argmax(ist0 == stage)) + 1
    days = np.arange(N_DAYS)
    cold = jnp.asarray((days >= start) & (days < start + COLD_FAIL_DAYS))
    f = f.replace(tmax=jnp.where(cold, COLD_FAIL_TMAX, f.tmax), tmin=jnp.where(cold, COLD_FAIL_TMIN, f.tmin))
    return p, f, s0


def _failure_cases() -> list[Any]:
    cases = [pytest.param("drought", SEEDS[0], 0, id="drought-before-silking")]
    cases += [pytest.param("cold", sd, st, id=f"cold-stage{st}-seed{sd}") for sd in SEEDS for st in (4, 5)]
    return cases


def _failure_inputs(kind: str, seed: int, stage: int) -> tuple[Any, Any, Any]:
    return _drought_inputs(seed) if kind == "drought" else _cold_inputs(seed, stage)


@x64_only
@pytest.mark.parametrize(("kind", "seed", "stage"), _failure_cases())
def test_the_failure_cases_fail_the_faithful_crop(kind: str, seed: int, stage: int) -> None:
    """The preconditions of the limit test below: the faithful crop fails (status 32 cold or 33 drought
    on its stage-6 days), in the intended stage, and then reaches maturity (stage 10)."""
    p, f, s0 = _failure_inputs(kind, seed, stage)
    ref = _run_faithful(p, f, s0)
    ist = np.asarray(ref.phen.istage[:, 0])
    status = np.asarray(ref.phen.crop_status[:, 0])
    assert ist[-1] == 10
    d6 = int(np.argmax(ist == 6))
    assert status[d6] == (33 if kind == "drought" else 32), status[d6]
    before = ist[d6 - 1]
    if kind == "drought":
        assert before < 4 and not np.any((ist >= 2) & (ist <= 5))
    else:
        assert before == stage, (before, stage)


@x64_only
@pytest.mark.parametrize(("kind", "seed", "stage"), _failure_cases())
def test_crop_failure_at_the_vanishing_scale(kind: str, seed: int, stage: int) -> None:
    """After a crop failure the vanishing scale still reproduces the faithful model: the same stages and
    maturity date, the same growth (no grain filled after the failure), and soft dates equal to the hard
    ones (a boundary the failed crop never reaches is passed on the failure day)."""
    p, f, s0 = _failure_inputs(kind, seed, stage)
    ref = _run_faithful(p, f, s0)
    got = _run_smoothed(_smoothed_params(p, 1e-12), f, s0)
    ist = np.asarray(ref.phen.istage[:, 0])
    np.testing.assert_array_equal(np.asarray(got.phen.istage[:, 0]), ist)
    np.testing.assert_array_equal(np.asarray(got.phen.mdate[-1]), np.asarray(ref.phen.mdate[-1]))
    np.testing.assert_array_equal(np.asarray(got.phen.stgdoy[-1]), np.asarray(ref.phen.stgdoy[-1]))
    for name, a, b in (
        ("yield", _yield(got), _yield(ref)),
        ("grain weight", got.growth.grnwt, ref.growth.grnwt),
        ("lai", got.growth.lai, ref.growth.lai),
        ("biomass", got.growth.biomas, ref.growth.biomas),
        ("rtwt", got.growth.rtwt, ref.growth.rtwt),
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-9, err_msg=name)
    d6 = int(np.argmax(ist == 6))
    hard = np.asarray([int(np.argmax(ist == c)) if np.any(ist == c) else d6 for c in CODES], dtype=float)
    np.testing.assert_array_equal(np.sum(1.0 - np.asarray(got.soft.passage[:, 0, :]), axis=0), hard)


def test_gate_and_clock_limits() -> None:
    x = jnp.asarray([-50.0, -1.0, 0.0, 1.0, 50.0])
    # an exact tie passes, as the reference's >= (the gate's TIE offset)
    np.testing.assert_array_equal(np.asarray(gate(x, 1e-12)), [0.0, 0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(np.asarray(soft_clock(x, 1e-6)), [0.0, 0.0, 0.0, 1.0, 50.0], atol=1e-6)
    g = jax.vmap(jax.grad(lambda t: soft_clock(t, 17.0)))(x)
    np.testing.assert_allclose(
        np.asarray(g), np.asarray(jax.nn.sigmoid(x / 17.0)), rtol=1e-12 if X64 else 1e-6
    )


def test_passage_chain_in_the_default_precision() -> None:
    """Smoke test in the tier's precision (float32 in the float32 tier): at 17 degC d the season reaches
    maturity and the passages are finite, ordered probabilities that never decrease (to rounding)."""
    p, f, s0 = _inputs(SEEDS[0])
    traj = _run_smoothed(_smoothed_params(p, 17.0), f, s0)
    pas = np.asarray(traj.soft.passage[:, 0, :])
    tol = 1e-12 if X64 else 1e-6
    assert pas.dtype == np.dtype(DTYPE)
    assert np.all(np.isfinite(pas)) and np.all((pas >= 0.0) & (pas <= 1.0))
    assert np.all(np.diff(pas, axis=0) >= -tol), "a passage decreased over the season"
    assert np.all(np.diff(pas[:, 1:], axis=1) <= tol), "a later boundary passed before an earlier one"
    assert int(np.asarray(traj.phen.istage[-1, 0])) == 10
    assert np.isfinite(float(_yield(traj))) and float(_yield(traj)) > 0.0


@x64_only
@pytest.mark.parametrize("seed", SEEDS[:1])
def test_yield_has_finite_nonzero_phenology_derivatives(seed: int) -> None:
    p, f, s0 = _inputs(seed)
    names = ("p1", "p2", "p5", "phint", "g2", "g3")
    theta0 = jnp.asarray([float(np.asarray(getattr(p.cultivar, n))) for n in names])

    def yield_of(theta: Any, width: float, smoothed: bool) -> Any:
        cul = p.cultivar.replace(**{n: theta[i] for i, n in enumerate(names)})
        pp = p.replace(cultivar=cul)
        traj = _run_smoothed(_smoothed_params(pp, width), f, s0) if smoothed else _run_faithful(pp, f, s0)
        return _yield(traj)

    g_s = np.asarray(jax.jacfwd(lambda t: yield_of(t, 17.0, True))(theta0))
    g_f = np.asarray(jax.jacfwd(lambda t: yield_of(t, 17.0, False))(theta0))
    assert np.all(np.isfinite(g_s))
    # P1 and P2 move the faithful yield only through the integer stage dates: derivative 0 there
    assert g_f[0] == 0.0 and g_f[1] == 0.0
    for i, n in enumerate(names[:4]):
        assert abs(g_s[i]) > 1e-3, n  # [kg ha-1 per unit]: every phenology coefficient moves the yield
    # P5 sets the grain-filling duration: its derivative is much larger once the stage end is a gate
    assert abs(g_s[2]) > 10.0 * abs(g_f[2]), (g_s[2], g_f[2])
    # AD against central differences at a small step (the smoothed model is smooth at this point)
    for i, n in enumerate(names):
        h = 1e-4 * float(theta0[i])
        e = jnp.zeros(6).at[i].set(h)
        fd = (float(yield_of(theta0 + e, 17.0, True)) - float(yield_of(theta0 - e, 17.0, True))) / (2 * h)
        np.testing.assert_allclose(g_s[i], fd, rtol=2e-3, atol=1e-6, err_msg=n)


def test_the_smoothed_phenology_needs_the_soft_state() -> None:
    p, f, s0 = _inputs(SEEDS[0])
    ft = jax.tree_util.tree_map(lambda x: x[0], f)
    with pytest.raises(ValueError, match="soft"):
        ceres_phenology_smoothed(s0, p, ft)
    assert s0.soft is None and p.smoothing is None  # the faithful defaults carry neither


@x64_only
@pytest.mark.parametrize("width", (0.3, 1.0, 3.0))
def test_derivatives_are_finite_at_small_scales(width: float) -> None:
    """At scales below a day of thermal time the soft stage-4 duration of a crop not yet in stage 4 is a
    logistic tail (down to 1e-300); its floor DUR4_MIN keeps SUMP / DUR4 and every derivative finite."""
    p, f, s0 = _inputs(SEEDS[0])
    names = ("p1", "p2", "p5", "phint", "g2", "g3")
    theta0 = jnp.asarray([float(np.asarray(getattr(p.cultivar, n))) for n in names])

    def outputs(theta: Any) -> Any:
        cul = p.cultivar.replace(**{n: theta[i] for i, n in enumerate(names)})
        traj = _run_smoothed(_smoothed_params(p.replace(cultivar=cul), width), f, s0)
        soft_dates = jnp.sum(1.0 - traj.soft.passage[:, 0, :], axis=0)
        return jnp.concatenate([_yield(traj)[None], traj.growth.biomas[-1], traj.phen.gpp[-1], soft_dates])

    jac = np.asarray(jax.jacfwd(outputs)(theta0))
    assert np.all(np.isfinite(jac)), np.argwhere(~np.isfinite(jac))
