"""Water ledger (plan 19 A10): cumulative channels as state, daily residual, closure under check.

* A synthetic bucket (rain in; evaporation, per-slot transpiration and drainage out) closes to
  1e-10 cm per day and over the run (x64); the cumulative channels equal NumPy float64 sums.
* The Richards process with the ledger as the last day entry (converged numerics, CA-TPA grid)
  closes to 1e-10 cm per day, and its ledger residual equals the process's own
  ``balance_error`` bookkeeping.
* Under ``AGRI_JAX_CHECK=1`` a leaking process makes the ledger raise inside ``jit``/``scan``;
  without the check it runs and reports the residual.
* In float32 the compensated cumulative sums stay within a few float32 ulps of the float64 sum
  where a naive float32 accumulation drifts.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from agrijax.core import Day, Forcing, Phase, State, WaterLedger, field, process, run, water_ledger
from agrijax.core.ledger import _neumaier, ledger_dtype
from agrijax.core.process import CHECK_ENV

X64 = bool(jax.config.read("jax_enable_x64"))


class Bucket(State):
    w: Array = field(dims=(), unit="cm")
    evap: Array = field(dims=(), unit="cm d-1")
    transp: Array = field(dims="n_crop", unit="cm d-1")
    drain: Array = field(dims=(), unit="cm d-1")


class BucketState(State):
    soil_water: Bucket
    ledger: dict


class Rain(Forcing):
    rain: Array = field(dims="T", unit="cm d-1")
    pet: Array = field(dims="T", unit="cm d-1")


def _bucket_day(leak: float):
    def bucket(s, p, f):
        """Rain in; evaporation, two-slot transpiration and drainage out, each limited by storage.

        Source: fixture (water balance dW = in - out).
        """
        w = s.soil_water.w + f.rain
        ev = jnp.minimum(0.4 * f.pet, 0.5 * w)
        w = w - ev
        tr = jnp.minimum(jnp.array([0.6, 0.3]) * f.pet, 0.2 * w)
        w = w - jnp.sum(tr)
        dr = 0.05 * jnp.maximum(w - 5.0, 0.0)
        w = w - dr + leak  # leak: water appearing from nowhere
        return eqx.tree_at(lambda t: t.soil_water, s, Bucket(w=w, evap=ev, transp=tr, drain=dr))

    return process(bucket, reads=("soil_water",), writes=("soil_water",), register=False, source="t")


LEDGER = water_ledger(
    storage="soil_water.w",
    inflows={"rain": lambda s, p, f: f.rain},
    outflows={
        "evaporation": "soil_water.evap",
        "transpiration": "soil_water.transp",
        "drainage": "soil_water.drain",
    },
)


def bucket_model(leak: float = 0.0):
    day = Day(
        ref="none",
        phases=(Phase("physcl", ("soil_water.bucket",)), Phase("ledger", ("ledger.close",))),
    )
    return day.compile(
        {"soil_water.bucket": _bucket_day(leak), "ledger.close": LEDGER},
        outputs=lambda s, p, f: s.ledger["water"],
    )


def bucket_inputs(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    rain = np.where(rng.random(n) < 0.3, rng.gamma(1.2, 1.0, n), 0.0)
    pet = rng.uniform(0.1, 0.6, n)
    w0 = 8.0
    led0 = WaterLedger.init(
        w0, inflows=["rain"], outflows={"evaporation": (), "transpiration": (2,), "drainage": ()}
    )
    zero = jnp.zeros(())
    s0 = BucketState(
        soil_water=Bucket(w=jnp.asarray(w0), evap=zero, transp=jnp.zeros(2), drain=zero),
        ledger={"water": led0},
    )
    return Rain(rain=jnp.asarray(rain), pet=jnp.asarray(pet)), s0, rain


def test_ledger_entry_declarations() -> None:
    assert LEDGER.name == "ledger.close"
    assert LEDGER.writes == ("ledger.water",)
    assert set(LEDGER.reads) == {
        "soil_water.w",
        "soil_water.evap",
        "soil_water.transp",
        "soil_water.drain",
        "ledger.water",
    }
    with pytest.raises(ValueError, match="both in and out"):
        water_ledger(storage="a.b", inflows={"x": "a.x"}, outflows={"x": "a.y"})


def test_bucket_ledger_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECK_ENV, "1")  # the per-day assertion runs inside the scan
    f, s0, rain = bucket_inputs()
    out = jax.jit(lambda f_, s_: run(bucket_model(), None, f_, s_))(f, s0)
    tol = 1e-10 if X64 else 2e-5
    assert float(jnp.max(jnp.abs(out.residual))) < tol
    last = jax.tree_util.tree_map(lambda x: x[-1], out)
    assert abs(float(last.closure())) < tol
    assert last.inflow["rain"].total.dtype == ledger_dtype()
    np.testing.assert_allclose(float(last.total_in()["rain"]), rain.sum(), rtol=1e-12 if X64 else 1e-6)
    assert last.outflow["transpiration"].total.shape == (2,)  # per slot
    assert float(jnp.sum(last.total_out()["drainage"])) > 0.0  # every channel really used


def test_leak_fails_under_check_and_is_reported_without(monkeypatch: pytest.MonkeyPatch) -> None:
    f, s0, _ = bucket_inputs(20)
    model = bucket_model(leak=1e-3)
    monkeypatch.setenv(CHECK_ENV, "1")
    with pytest.raises(RuntimeError, match=r"water ledger at 'ledger\.water' does not close"):
        jax.block_until_ready(jax.jit(lambda f_, s_: run(model, None, f_, s_))(f, s0))
    monkeypatch.setenv(CHECK_ENV, "0")
    out = jax.jit(lambda f_, s_: run(bucket_model(leak=1e-3), None, f_, s_))(f, s0)
    np.testing.assert_allclose(np.asarray(out.residual), 1e-3, rtol=1e-3)
    assert float(out.max_abs_residual[-1]) == pytest.approx(1e-3, rel=1e-3)


def test_tolerance_can_be_set_per_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    f, s0, _ = bucket_inputs(20)
    loose = water_ledger(
        storage="soil_water.w",
        inflows={"rain": lambda s, p, f: f.rain},
        outflows={
            "evaporation": "soil_water.evap",
            "transpiration": "soil_water.transp",
            "drainage": "soil_water.drain",
        },
        atol=1e-2,
    )
    day = Day(
        ref="none", phases=(Phase("physcl", ("soil_water.bucket",)), Phase("ledger", ("ledger.close",)))
    )
    model = day.compile(
        {"soil_water.bucket": _bucket_day(1e-3), "ledger.close": loose},
        outputs=lambda s, p, f: s.ledger["water"].residual,
    )
    monkeypatch.setenv(CHECK_ENV, "1")
    res = run(model, None, f, s0)
    assert float(jnp.max(res)) == pytest.approx(1e-3, rel=1e-3)


@pytest.mark.allow_skip(reason="the 1e-10 cm closure is a float64 claim")
def test_richards_ledger_closes_and_matches_balance_error(monkeypatch: pytest.MonkeyPatch) -> None:
    if not X64:
        pytest.skip("needs float64")
    from agrijax.processes.soil_water.richards import (
        RichardsConfig,
        RichardsForcing,
        RichardsParams,
        SoilWater,
        richards_redistribution,
    )

    from .test_richards import catpa_grid, catpa_soil, synthetic_forcing

    class RState(State):
        soil_water: SoilWater
        ledger: dict

    grid, soil = catpa_grid(), catpa_soil()
    supply, evap, uptake = synthetic_forcing(12)
    params = RichardsParams(soil=soil, grid=grid, config=RichardsConfig(n_sub=48, n_iter=8))
    w0 = SoilWater.from_theta(jnp.full(37, 0.25), soil)

    def storage(s, p, f):
        return jnp.sum(s.soil_water.theta * p.grid.tl) + s.soil_water.pond

    fl = "soil_water.flux."
    ledger = water_ledger(
        storage=storage,
        inflows={"supply": lambda s, p, f: jnp.sum(f.supply)},  # hourly rates over 24 h of 1 h
        outflows={k: fl + k for k in ("evaporation", "drainage", "uptake", "runoff")},
        reads=("soil_water.theta", "soil_water.pond"),
    )
    day = Day(
        ref="rzwqm2-4.6",
        phases=(Phase("physcl", ("soil_water.richards",)), Phase("ledger", ("ledger.close",))),
    )
    model = day.compile(
        {"soil_water.richards": richards_redistribution, "ledger.close": ledger},
        outputs=lambda s, p, f: (s.ledger["water"], s.soil_water.flux.balance_error),
    )
    s0 = RState(
        soil_water=w0,
        ledger={
            "water": WaterLedger.init(
                storage(RState(soil_water=w0, ledger={}), params, None),
                inflows=["supply"],
                outflows=["evaporation", "drainage", "uptake", "runoff"],
            )
        },
    )
    forcing = RichardsForcing(
        supply=jnp.asarray(supply), evaporation=jnp.asarray(evap), uptake=jnp.asarray(uptake)
    )
    monkeypatch.setenv(CHECK_ENV, "1")
    led, balance_error = jax.jit(lambda f_, s_: run(model, params, f_, s_))(forcing, s0)
    assert float(jnp.max(jnp.abs(led.residual))) < 1e-10
    np.testing.assert_allclose(np.asarray(led.residual), np.asarray(balance_error), rtol=0, atol=1e-11)
    last = jax.tree_util.tree_map(lambda x: x[-1], led)
    assert abs(float(last.closure())) < 1e-10
    assert float(last.total_in()["supply"]) == pytest.approx(supply.sum(), rel=1e-12)


def test_compensated_sum_in_the_ledger_dtype() -> None:
    """10^4 daily amounts of ~0.1 cm: the compensated total stays at float64-sum accuracy."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 0.2, 10_000)
    dt = ledger_dtype()
    xs = jnp.asarray(x, dt)

    def body(carry, v):
        t, c = _neumaier(carry[0], carry[1], v)
        return (t, c), None

    (t, c), _ = jax.lax.scan(body, (jnp.zeros((), dt), jnp.zeros((), dt)), xs)
    exact = float(np.sum(np.asarray(xs, np.float64)))  # the float64 sum of the same (rounded) inputs
    naive = float(jax.lax.scan(lambda a, v: (a + v, None), jnp.zeros((), dt), xs)[0])
    comp = float(t + c)
    eps = float(jnp.finfo(dt).eps)
    assert abs(comp - exact) <= 2 * eps * exact
    if dt == jnp.float32:
        assert abs(naive - exact) > 10 * abs(comp - exact)  # the naive float32 sum drifts
