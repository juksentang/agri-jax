"""The core runtime on a real model: CA-TPA 2015-2023, Shuttleworth-Wallace PET + bookkeeping.

Model: :mod:`agri_jax.models.catpa_pet_demo` (3287 days). Every runtime path is checked against
a path that does not go through it:

* ``run`` (``lax.scan`` over the compiled day step) vs a Python day loop that calls the S-W kernel
  directly and does the bookkeeping in Python floats (no Model, no @process, no scan);
* the forcing vs the files it was built from, and the reference model's own echo of it (.ana);
* the running totals vs a NumPy cumulative sum (conservation of the bookkeeping);
* transpiration of 2015 vs the RZWQM2 .ana column (coarse; a pipeline test, not a science test);
* ``run_batch`` (vmap) over 512 parameter sets vs a Python loop of ``run``;
* ``run_batch_chunked(chunk=100)`` vs ``run_batch``; ``checkpoint=True`` vs ``False``;
* ``run_and_grad`` vs central differences on three PET parameters;
* compilation counts from ``jax.monitoring`` (no recompilation per call, scan not unrolled);
* the ``AGRI_JAX_CHECK=1`` writes check on a deliberately wrong wrapper inside ``run``.

Timings are printed (run with ``-s``) and collected in ``TIMINGS``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import monitoring

from agri_jax.core import run, run_batch, run_batch_chunked
from agri_jax.core.process import ProcessWriteError, process
from agri_jax.core.runtime import run_and_grad
from agri_jax.io.rzwqm import prepare_rzwqm_forcing, read_ana, read_met, read_rzwqm_dat
from agri_jax.io.rzwqm.layers import read_layer_output
from agri_jax.models.catpa_pet_demo import (
    ANA_HEIGHT,
    ANA_LAI,
    ANA_RESIDUE,
    OUTPUTS,
    DemoState,
    accumulate_totals,
    build_forcing,
    catpa_pet_model,
    initial_state,
    site_params_from_dat,
    sw_pet_from_forcing,
)
from agri_jax.processes.pet import shuttleworth_wallace

REF_DIRS = (Path("validation/catpa_2015_ref"), Path("catpa/ref_2015"))
START, END = "2015-01-01", "2023-12-31"
N_BATCH = 512
TIMINGS: dict[str, float] = {}

X64 = bool(jax.config.jax_enable_x64)
RTOL = 1e-12 if X64 else 1e-5

pytestmark = pytest.mark.skipif(not X64, reason="real-pipeline checks are defined in float64")


def _timed(key: str, fn: Any, *args: Any) -> Any:
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    TIMINGS[key] = time.perf_counter() - t0
    print(f"[timing] {key}: {TIMINGS[key]:.3f} s")
    return out


def _ana_col(ana: Any, col: int) -> np.ndarray:
    cols = {int(k): v for k, v in ana.attrs["columns"].items()}
    return np.asarray(ana[cols[col]].values, dtype=float)


@pytest.fixture(scope="module")
def pipe(data_dir: Path, catpa_scenario: Path) -> dict[str, Any]:
    ref = next((data_dir / d for d in REF_DIRS if (data_dir / d / "CA-TPA.ana").is_file()), None)
    if ref is None or not (ref / "LAYER.PLT").is_file():
        pytest.skip(f"CA-TPA 2015 reference run (.ana + LAYER.PLT) not found under {data_dir}")
    met_raw = read_met(catpa_scenario / "CA-TPA.MET")
    dat = read_rzwqm_dat(catpa_scenario / "rzwqm.dat")
    phys = dat.physiography
    # the whole INPDAY chain plus the hourly re-sum of the radiation (what the model uses)
    met = prepare_rzwqm_forcing(
        met_raw, latitude_rad=phys["latitude_rad"], slope_rad=phys["slope_rad"], aspect_rad=phys["aspect_rad"]
    )
    ana = read_ana(ref / "CA-TPA.ana")
    theta = read_layer_output(ref / "LAYER.PLT", start="2015-01-01")["soil_water_content"]
    theta1 = np.asarray(theta.isel(depth=0).values, dtype=float)
    params = site_params_from_dat(dat, catpa_scenario / "rzwqm.dat")
    forcing = build_forcing(met, ana, START, END)
    state0 = initial_state(theta1[0])
    model = catpa_pet_model()
    single = jax.jit(lambda p: run(model, p, forcing, state0))
    outs = _timed("run_9y_compile_and_run", single, params)
    _timed("run_9y_cached", single, params)
    return dict(
        met=met,
        met_raw=met_raw,
        ana=ana,
        theta1=theta1,
        params=params,
        forcing=forcing,
        state0=state0,
        model=model,
        single=single,
        outs=outs,
    )


def _batch_params(params: Any, n: int, seed: int = 20150101) -> Any:
    """``n`` parameter sets: three PET parameters stratified-uniform, the rest broadcast."""
    rng = np.random.default_rng(seed)

    def strat(lo: float, hi: float) -> np.ndarray:  # one draw per stratum, shuffled (LHS)
        u = (rng.permutation(n) + rng.random(n)) / n
        return lo + (hi - lo) * u

    rs, rss, a0 = strat(100.0, 400.0), strat(20.0, 150.0), strat(0.15, 0.35)
    batched = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x))), params)
    batched = batched.set("pet.stomatal_resistance", jnp.asarray(rs))
    batched = batched.set("pet.soil_resistance", jnp.asarray(rss))
    return batched.set("pet.albedo_dry", jnp.asarray(a0))


def _index(tree: Any, i: int) -> Any:
    return jax.tree_util.tree_map(lambda x: x[i], tree)


# ------------------------------------------------------------------------------ forcing


def test_forcing_matches_met_and_ana(pipe: dict[str, Any]) -> None:
    """The stacked forcing is the prepared .MET (which the reference model echoes in .ana) and the
    2015 canopy, repeated by day of year."""
    f, met, raw, ana = pipe["forcing"], pipe["met"], pipe["met_raw"], pipe["ana"]
    assert f.n_days == 3287 == len(met.loc[START:END])
    sl = met.loc[START:END]
    np.testing.assert_array_equal(np.asarray(f.tmax), sl["tmax"].to_numpy())
    np.testing.assert_array_equal(np.asarray(f.srad), sl["srad_mj"].to_numpy())
    np.testing.assert_array_equal(
        np.asarray(f.wind_run), np.maximum(raw.loc[START:END, "wind_run_km"], 100.0)
    )
    # the reference model's own echo of the 2015 weather (.ana cols 85 tmin, 86 tmax, 89 rh, 90 wind)
    for attr, col in (("tmin", 85), ("tmax", 86), ("rh", 89), ("wind_run", 90)):
        np.testing.assert_array_equal(
            np.asarray(getattr(f, attr))[:365], _ana_col(ana, col)[1:], err_msg=attr
        )
    # .ana col 88 is not a plain echo of the .MET radiation (-0.7 % .. +0.5 % day to day): it is the
    # re-sum of the model's hourly disaggregation, which prepare_rzwqm_forcing(latitude_rad=...)
    # reproduces to the 6-digit print precision
    np.testing.assert_allclose(np.asarray(f.srad)[:365], _ana_col(ana, 88)[1:], rtol=1e-5, atol=1e-5)
    met_ratio = _ana_col(ana, 88)[1:] / raw.loc["2015", "srad_mj"].to_numpy()
    assert np.abs(met_ratio - 1.0).max() > 5e-3  # the raw .MET value is measurably different
    lai, height, res = (_ana_col(ana, c) for c in (ANA_LAI, ANA_HEIGHT, ANA_RESIDUE))
    np.testing.assert_array_equal(np.asarray(f.lai)[:365], lai[1:])
    np.testing.assert_array_equal(np.asarray(f.height_cm)[:365], height[1:])
    np.testing.assert_array_equal(np.asarray(f.residue_mass)[:365], res[:-1])
    # 2016 (leap): 1 Jan .. 30 Dec repeat 2015 by day of year, 31 Dec (doy 366) reuses 31 Dec 2015
    np.testing.assert_array_equal(np.asarray(f.lai)[365 : 365 + 365], lai[1:])
    assert float(f.lai[365 + 365]) == lai[-1] and float(f.doy[365 + 365]) == 366.0
    assert float(np.max(np.asarray(f.lai))) > 1.0  # a real crop season is in there


# ------------------------------------------------------------------------------ run vs independent loop


def _python_day_loop(params: Any, forcing: Any, theta: float, n_days: int, *, jit_kernel: bool) -> dict:
    """Independent path: call the S-W kernel day by day, bookkeeping in Python floats."""

    def kernel(tmin, tmax, srad, rh, wind, lai, hc, doy, rm, age):
        return shuttleworth_wallace(
            tmin,
            tmax,
            srad,
            rh,
            wind,
            lai,
            hc,
            params.pet,
            theta_surface=theta,
            wc13=params.wc13,
            wc15=params.wc15,
            elevation=params.elevation,
            latitude=params.latitude,
            doy=doy,
            tlai=lai,
            residue_mass=rm,
            residue_age=age,
            wind_height=params.wind_height,
            trat=params.trat,
            rainfall_zone=params.rainfall_zone,
            residue_type=params.residue_type,
            residue_cover_factor=params.residue_cover_factor,
        )

    k = jax.jit(kernel) if jit_kernel else kernel
    cols = [
        np.asarray(getattr(forcing, a))
        for a in (
            "tmin",
            "tmax",
            "srad",
            "rh",
            "wind_run",
            "lai",
            "height_cm",
            "doy",
            "residue_mass",
            "residue_age",
        )
    ]
    out: dict[str, list[float]] = {p: [] for p in OUTPUTS}
    tt = te = 0.0
    for d in range(n_days):
        r = k(*(float(c[d]) for c in cols))
        pt, ps, pr = float(r.transpiration), float(r.soil_evaporation), float(r.residue_evaporation)
        tt = tt + pt
        te = te + ps + pr
        for key, v in zip(OUTPUTS, (pt, ps, pr, tt, te, float(d + 1))):
            out[key].append(v)
    return {key: np.asarray(v) for key, v in out.items()}


def test_run_equals_python_day_loop(pipe: dict[str, Any]) -> None:
    """lax.scan over Model.compile() == a hand loop over the jitted kernel, all 3287 days, 1e-12."""
    outs = pipe["outs"]
    t0 = time.perf_counter()
    ref = _python_day_loop(pipe["params"], pipe["forcing"], float(pipe["theta1"][0]), 3287, jit_kernel=True)
    TIMINGS["python_day_loop_jit_kernel_9y"] = time.perf_counter() - t0
    print(f"[timing] python_day_loop_jit_kernel_9y: {TIMINGS['python_day_loop_jit_kernel_9y']:.3f} s")
    for key in OUTPUTS:
        np.testing.assert_allclose(np.asarray(outs[key]), ref[key], rtol=RTOL, atol=1e-15, err_msg=key)
    assert ref["totals.transpiration"][-1] > 100.0  # 9 seasons of real transpiration, cm


def test_run_equals_eager_kernel_first_60_days(pipe: dict[str, Any]) -> None:
    """Same, with no jit at all on the reference side (op-by-op dispatch), first 60 days."""
    outs = pipe["outs"]
    ref = _python_day_loop(pipe["params"], pipe["forcing"], float(pipe["theta1"][0]), 60, jit_kernel=False)
    for key in OUTPUTS:
        np.testing.assert_allclose(np.asarray(outs[key])[:60], ref[key], rtol=RTOL, atol=1e-15, err_msg=key)


def test_totals_conserve_daily_fluxes(pipe: dict[str, Any]) -> None:
    """Bookkeeping conservation: running sums == NumPy cumulative sums of the daily fluxes."""
    o = {k: np.asarray(v) for k, v in pipe["outs"].items()}
    np.testing.assert_allclose(o["totals.transpiration"], np.cumsum(o["pet.transpiration"]), rtol=1e-12)
    ev = o["pet.soil_evaporation"] + o["pet.residue_evaporation"]
    np.testing.assert_allclose(o["totals.evaporation"], np.cumsum(ev), rtol=1e-12)
    np.testing.assert_array_equal(o["totals.days"], np.arange(1, 3288, dtype=float))
    for k in ("pet.transpiration", "pet.soil_evaporation", "pet.residue_evaporation"):
        assert np.all(np.isfinite(o[k])) and np.all(o[k] >= 0.0), k


def test_transpiration_2015_against_reference_model(pipe: dict[str, Any]) -> None:
    """Coarse agreement of 2015 potential transpiration with RZWQM2 (.ana col 9, cm d-1).

    The demo holds the surface water content constant and ages the residue by day of year, so only
    transpiration is compared, with the bound of tests/integration/test_pet_oracle.py.
    """
    pt = np.asarray(pipe["outs"]["pet.transpiration"])[1:365] * 10.0
    ref = _ana_col(pipe["ana"], 9)[2:] * 10.0
    rmse = float(np.sqrt(np.mean((pt - ref) ** 2)))
    print(f"[metric] 2015 PT RMSE vs .ana col 9: {rmse:.4f} mm/d; sums {pt.sum():.1f} vs {ref.sum():.1f} mm")
    assert rmse < 0.3
    assert abs(pt.sum() - ref.sum()) < 0.1 * ref.sum()


# ------------------------------------------------------------------------------ batch paths


@pytest.fixture(scope="module")
def batch(pipe: dict[str, Any]) -> dict[str, Any]:
    params = _batch_params(pipe["params"], N_BATCH)
    m, f, s0 = pipe["model"], pipe["forcing"], pipe["state0"]
    outs = _timed(f"run_batch_{N_BATCH}x9y", lambda p: run_batch(m, p, f, s0), params)
    return dict(params=params, outs=outs)


def test_run_batch_512_equals_loop_of_run(pipe: dict[str, Any], batch: dict[str, Any]) -> None:
    single, params, outs = pipe["single"], batch["params"], batch["outs"]
    t0 = time.perf_counter()
    loop = [jax.block_until_ready(single(_index(params, i))) for i in range(N_BATCH)]
    TIMINGS[f"loop_of_run_{N_BATCH}x9y"] = time.perf_counter() - t0
    print(f"[timing] loop_of_run_{N_BATCH}x9y: {TIMINGS[f'loop_of_run_{N_BATCH}x9y']:.3f} s")
    for key in OUTPUTS:
        ref = np.stack([np.asarray(o[key]) for o in loop])
        np.testing.assert_allclose(np.asarray(outs[key]), ref, rtol=RTOL, atol=1e-15, err_msg=key)
    # the three parameters actually move the result
    tot = np.asarray(outs["totals.transpiration"])[:, -1]
    assert tot.std() > 0.01 * tot.mean()


def test_run_batch_chunked_equals_run_batch(pipe: dict[str, Any], batch: dict[str, Any]) -> None:
    """chunk=100 over 512 samples (five full chunks + a partial one) is bitwise run_batch."""
    m, f, s0 = pipe["model"], pipe["forcing"], pipe["state0"]
    ch = _timed(
        f"run_batch_chunked_{N_BATCH}x9y_chunk100",
        lambda p: run_batch_chunked(m, p, f, s0, chunk=100),
        batch["params"],
    )
    for key in OUTPUTS:
        np.testing.assert_array_equal(np.asarray(ch[key]), np.asarray(batch["outs"][key]), err_msg=key)


def test_checkpoint_is_exact(pipe: dict[str, Any]) -> None:
    m, f, s0, p = pipe["model"], pipe["forcing"], pipe["state0"], pipe["params"]
    ck = jax.jit(lambda q: run(m, q, f, s0, checkpoint=True))(p)
    for key in OUTPUTS:
        np.testing.assert_array_equal(np.asarray(ck[key]), np.asarray(pipe["outs"][key]), err_msg=key)

    def loss(o: dict) -> Any:
        return o["totals.transpiration"][-1] + o["totals.evaporation"][-1]

    g_ck = jax.jit(lambda q: run_and_grad(m, loss, q, f, s0, checkpoint=True))(p)
    g_no = jax.jit(lambda q: run_and_grad(m, loss, q, f, s0, checkpoint=False))(p)
    leaves_ck, leaves_no = jax.tree_util.tree_leaves(g_ck), jax.tree_util.tree_leaves(g_no)
    assert len(leaves_ck) == len(leaves_no)
    for a, b in zip(leaves_ck, leaves_no):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=RTOL, atol=0.0)


# ------------------------------------------------------------------------------ gradients


GRAD_FIELDS = ("pet.stomatal_resistance", "pet.soil_resistance", "pet.albedo_dry")


def test_run_and_grad_matches_central_differences(pipe: dict[str, Any]) -> None:
    m, f, s0, p = pipe["model"], pipe["forcing"], pipe["state0"], pipe["params"]

    def loss(o: dict) -> Any:  # 9-year potential transpiration + half the evaporation, mm
        return 10.0 * (o["totals.transpiration"][-1] + 0.5 * o["totals.evaporation"][-1])

    vg = jax.jit(lambda q: run_and_grad(m, loss, q, f, s0))
    value, grad = _timed("run_and_grad_9y_compile_and_run", vg, p)
    _timed("run_and_grad_9y_cached", vg, p)
    for leaf in jax.tree_util.tree_leaves(grad):
        assert np.all(np.isfinite(np.asarray(leaf)))
    fwd = jax.jit(lambda q: loss(run(m, q, f, s0)))
    np.testing.assert_allclose(float(value), float(fwd(p)), rtol=1e-13)
    for path in GRAD_FIELDS:
        x = float(p.get(path))
        h = 1e-5 * abs(x)
        fd = (float(fwd(p.set(path, jnp.asarray(x + h)))) - float(fwd(p.set(path, jnp.asarray(x - h))))) / (
            2 * h
        )
        g = float(grad.get(path))
        print(f"[grad] d loss / d {path}: ad {g:.10e} fd {fd:.10e}")
        assert g != 0.0
        np.testing.assert_allclose(g, fd, rtol=1e-5, err_msg=path)
    assert float(grad.get("pet.stomatal_resistance")) < 0.0  # more resistance, less transpiration


# ------------------------------------------------------------------------------ compilation


class _CompileCounter:
    EVENT = "/jax/core/compile/backend_compile_duration"

    def __init__(self) -> None:
        self.n = 0

    def __call__(self, event: str, duration_secs: float, **kwargs: str | int) -> None:
        if event == self.EVENT:
            self.n += 1

    def __enter__(self) -> _CompileCounter:
        monitoring.register_event_duration_secs_listener(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        monitoring.unregister_event_duration_listener(self)


def test_compilation_is_bounded(pipe: dict[str, Any]) -> None:
    m, f, s0, p = pipe["model"], pipe["forcing"], pipe["state0"], pipe["params"]
    single = jax.jit(lambda q: run(m, q, f, s0))
    with _CompileCounter() as c:
        for rs in (150.0, 200.0, 250.0, 300.0, 350.0):
            jax.block_until_ready(single(p.set("pet.stomatal_resistance", jnp.asarray(rs))))
    assert c.n == 1, f"a held jitted run compiled {c.n} times for 5 parameter values"

    bp = _batch_params(p, 16)
    with _CompileCounter() as c:
        jax.block_until_ready(run_batch(m, bp, f, s0))
        first = c.n
        jax.block_until_ready(run_batch(m, _batch_params(p, 16, seed=7), f, s0))
    print(f"[compile] run_batch: first call {first}, second call {c.n - first}")
    assert first <= 1 and c.n - first == 0  # one compilation, reused by the second call (same shapes)

    # the scan is not unrolled: the traced program has the same size for 1 and 9 years
    def n_eqns(n_days: int) -> int:
        fd = jax.tree_util.tree_map(lambda x: x[:n_days], f)
        jaxpr = jax.make_jaxpr(lambda q: run(m, q, fd, s0))(p)
        return len(jaxpr.jaxpr.eqns)

    assert n_eqns(365) == n_eqns(3287)


# ------------------------------------------------------------------------------ writes check


@process(reads=("theta_surface",), writes=("pet",), register=False, source="test: deliberately wrong")
def _wrong_pet(state: DemoState, params: Any, forcing_t: Any) -> DemoState:
    """Correct fluxes, but also dries the surface node without declaring it.

    Source: test only.
    """
    new = sw_pet_from_forcing.fn(state, params, forcing_t)
    return eqx.tree_at(lambda s: s.theta_surface, new, new.theta_surface - 0.001)


def test_writes_check_catches_wrong_wrapper_inside_run(
    pipe: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    m, f, s0, p = pipe["model"], pipe["forcing"], pipe["state0"], pipe["params"]
    bad = m.replace("sw_pet_from_forcing", _wrong_pet)
    f30 = jax.tree_util.tree_map(lambda x: x[:30], f)
    monkeypatch.delenv("AGRI_JAX_CHECK", raising=False)
    run(bad, p, f30, s0)  # silently wrong without the check
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")
    with pytest.raises(ProcessWriteError, match="theta_surface"):
        run(bad, p, f30, s0)
    with pytest.raises(ProcessWriteError, match="theta_surface"):
        _wrong_pet(s0, p, _index(f, 180))
    # the real processes pass the check on the full real forcing (traced inside the scan)
    out = run(m, p, f, s0)
    np.testing.assert_array_equal(np.asarray(out["totals.days"]), np.asarray(pipe["outs"]["totals.days"]))
    assert accumulate_totals.check_writes(s0, accumulate_totals(s0, p, _index(f, 0))) == ["totals.days"]


def test_report_timings() -> None:
    """Print every timing collected by this module (run last; with -s)."""
    for k, v in TIMINGS.items():
        print(f"[timing-summary] {k}: {v:.3f} s")
