"""The water ledger of the soil-water day books every sink channel (plan 19 A6/A10, H1 item 6).

* :func:`soil_water_ledger` reads the daily total of every channel of ``SINK_LEDGER_OUTFLOWS``
  (``uptake``, ``tile``, ``lateral``, ``subirrigation``, ``macropore_to_drain``), each as its own
  cumulative outflow, plus evaporation, drainage and runoff; inflows are supply and event rain.
* A synthetic run with Green-Ampt storms and non-zero tile, lateral, subirrigation (a negative
  sink) and macropore channels closes to 1e-10 cm per day under ``AGRI_JAX_CHECK=1`` (float64,
  converged numerics); each cumulative channel equals the NumPy sum of its daily flux, and the
  ledger residual equals the day's own ``balance_error``.
* Negative control: a ledger that books only ``uptake`` (the pre-H1 form) misses exactly the other
  channels' daily totals and raises under the check.
* At the dry end (``h_min`` cut) the ledger books the channels after the cut; its residual equals the
  day's ``balance_error`` (the day itself closes only to 1.5e-7 cm there: strict xfail, open issue).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from agrijax.core import Day, Forcing, Phase, State, WaterLedger, field, process, run, water_ledger
from agrijax.core.process import CHECK_ENV
from agrijax.processes.soil_water import (
    SINK_CHANNELS,
    SINK_LEDGER_OUTFLOWS,
    SOIL_WATER_LEDGER_INFLOWS,
    SOIL_WATER_LEDGER_OUTFLOWS,
    DayConfig,
    SinkChannel,
    SinkChannels,
    soil_water_day_kernel,
    soil_water_ledger,
    soil_water_ledger_init,
)
from agrijax.processes.soil_water import infiltration as G
from agrijax.processes.soil_water.richards import SoilWater

from .test_infiltration_day import _params, _storms
from .test_richards import catpa_soil, synthetic_forcing

X64 = bool(jax.config.read("jax_enable_x64"))
N_DAY = 10
G_DAY = DayConfig(n_pre=12, n_post=36)
OTHER = ("tile", "lateral", "subirrigation", "macropore_to_drain")


class ChannelForcing(Forcing):
    storm: G.StormForcing
    supply: Array = field(unit="cm h-1", dims=("T", "hour"))
    evaporation: Array = field(unit="cm h-1", dims=("T", "hour"))
    uptake: Array = field(unit="cm d-1", dims=("T", "n_node"))
    tile: Array = field(unit="cm d-1", dims=("T", "n_node"))
    lateral: Array = field(unit="cm d-1", dims=("T", "n_node"))
    subirrigation: Array = field(unit="cm d-1", dims=("T", "n_node"))
    macropore_to_drain: Array = field(unit="cm d-1", dims=("T", "n_node"))


class LState(State):
    soil_water: SoilWater
    ledger: dict


def _day_with_channels(state, params, f):
    """The soil-water day with every sink channel given as a daily per-layer amount.

    Source: fixture (plan 19 A6 sink channels, soil_water_day_kernel).
    """
    s = SinkChannels(
        uptake=SinkChannel(daily=f.uptake, carries_solute=False),
        tile=SinkChannel(daily=f.tile),
        lateral=SinkChannel(daily=f.lateral),
        subirrigation=SinkChannel(daily=f.subirrigation),
        macropore_to_drain=SinkChannel(daily=f.macropore_to_drain),
    )
    new, _, _ = soil_water_day_kernel(state.soil_water, params, f.supply, f.evaporation, s, f.storm)
    return eqx.tree_at(lambda t: t.soil_water, state, new)


DAY_ENTRY = process(
    _day_with_channels,
    reads=("soil_water.h", "soil_water.theta", "soil_water.pond"),
    writes=("soil_water",),
    register=False,
    source="fixture",
    name="soil_water.day",
)


def _storage(s, p, f):
    return jnp.sum(s.soil_water.theta * p.richards.grid.tl) + s.soil_water.pond


#: the pre-H1 ledger: one uptake term, the other sink channels not booked
UPTAKE_ONLY = water_ledger(
    storage=_storage,
    inflows={"supply": lambda s, p, f: jnp.sum(f.supply), "rain": "soil_water.flux.rain"},
    outflows={k: f"soil_water.flux.{k}" for k in ("evaporation", "drainage", "runoff", "uptake")},
    reads=("soil_water.theta", "soil_water.pond"),
    at="ledger.uptake_only",
    name="ledger.uptake_only",
)


def _model(with_uptake_only: bool = False):
    entries = ("ledger.close", "ledger.uptake_only") if with_uptake_only else ("ledger.close",)
    day = Day(ref="rzwqm2-4.6", phases=(Phase("physcl", ("soil_water.day",)), Phase("ledger", entries)))
    procs = {"soil_water.day": DAY_ENTRY, "ledger.close": soil_water_ledger()}
    if with_uptake_only:
        procs["ledger.uptake_only"] = UPTAKE_ONLY
    return day.compile(procs, outputs=lambda s, p, f: (s.ledger, s.soil_water.flux))


def _channels(n: int, n_day: int, scale: float = 1.0) -> dict[str, np.ndarray]:
    """Daily per-layer amounts [cm d-1]: tile and macropore at depth, lateral mid-profile, subirrigation adds."""
    z = np.zeros((n_day, n))
    tile, lat, sub, mac = z.copy(), z.copy(), z.copy(), z.copy()
    tile[:, 25:30] = 0.01 * scale
    lat[:, 15:20] = 0.004 * scale
    sub[:, 30:34] = -0.006 * scale
    mac[:, 20:22] = 0.003 * scale
    return {"tile": tile, "lateral": lat, "subirrigation": sub, "macropore_to_drain": mac}


def _inputs(theta0: float = 0.25):
    params = _params(G_DAY, n_sub=48, n_iter=10)
    n = params.richards.grid.n_node
    supply, evap, uptake = synthetic_forcing(N_DAY, seed=5)
    ch = _channels(n, N_DAY)
    forcing = ChannelForcing(
        supply=jnp.asarray(0.5 * supply),
        evaporation=jnp.asarray(evap),
        uptake=jnp.asarray(uptake),
        storm=_storms(N_DAY, seed=5),
        **{k: jnp.asarray(v) for k, v in ch.items()},
    )
    w0 = SoilWater.from_theta(jnp.full(n, theta0), catpa_soil())
    led0 = soil_water_ledger_init(w0, params.richards.grid)
    s0 = LState(soil_water=w0, ledger={"water": led0, "uptake_only": _uptake_only_init(w0, params)})
    return params, forcing, s0, ch


def _uptake_only_init(w0: SoilWater, params):
    return WaterLedger.init(
        w0.storage(params.richards.grid) + w0.pond,
        inflows=["supply", "rain"],
        outflows=["evaporation", "drainage", "runoff", "uptake"],
    )


def _run(model, params, forcing, s0):
    return jax.jit(lambda f_, s_: run(model, params, f_, s_))(forcing, s0)


def test_ledger_channels_cover_every_sink_channel() -> None:
    """Declarations: every ``SINK_LEDGER_OUTFLOWS`` path is read and booked under its own name."""
    led = soil_water_ledger()
    assert led.name == "ledger.close" and led.writes == ("ledger.water",)
    for name, path in SINK_LEDGER_OUTFLOWS.items():
        assert SOIL_WATER_LEDGER_OUTFLOWS[name] == path
        assert path in led.reads, name
    assert set(SINK_CHANNELS) <= set(SOIL_WATER_LEDGER_OUTFLOWS)
    assert set(SOIL_WATER_LEDGER_INFLOWS) == {"supply", "rain"}
    params = _params(G_DAY, n_sub=48, n_iter=10)
    w0 = SoilWater.from_theta(jnp.full(params.richards.grid.n_node, 0.25), catpa_soil())
    l0 = soil_water_ledger_init(w0, params.richards.grid)
    assert tuple(l0.outflow) == tuple(SOIL_WATER_LEDGER_OUTFLOWS)
    assert tuple(l0.inflow) == SOIL_WATER_LEDGER_INFLOWS
    assert float(l0.storage) == float(w0.storage(params.richards.grid))


@pytest.mark.allow_skip(reason="the 1e-10 cm closure is a float64 claim")
def test_ledger_closes_with_tile_lateral_subirrigation_macropore(monkeypatch: pytest.MonkeyPatch) -> None:
    if not X64:
        pytest.skip("needs float64")
    params, forcing, s0, ch = _inputs()
    monkeypatch.setenv(CHECK_ENV, "1")  # the per-day closure assertion runs inside the scan
    ledgers, flux = _run(_model(), params, forcing, s0)
    led = ledgers["water"]
    res = np.asarray(led.residual)
    print("synthetic channels: max |daily residual| [cm] =", np.max(np.abs(res)))
    assert np.max(np.abs(res)) < 1e-10
    # the ledger residual is the day's own balance_error, recomputed from the booked channels
    np.testing.assert_allclose(res, np.asarray(flux.balance_error), rtol=0, atol=1e-12)
    last = jax.tree_util.tree_map(lambda x: x[-1], led)
    print("synthetic channels: whole-run closure [cm] =", float(last.closure()))
    assert abs(float(last.closure())) < 1e-10
    tout = last.total_out()
    for name in SOIL_WATER_LEDGER_OUTFLOWS:
        daily = np.asarray(getattr(flux, name), np.float64)
        assert float(tout[name]) == pytest.approx(float(np.sum(daily)), rel=1e-12, abs=1e-14), name
    # every channel is really used: no cut on a wet profile, each is its prescribed amount
    assert float(np.sum(np.asarray(flux.sink_cut))) == 0.0
    for name in OTHER:
        assert float(tout[name]) == pytest.approx(float(ch[name].sum()), rel=1e-9), name
    assert float(tout["tile"]) > 0.0 and float(tout["lateral"]) > 0.0
    assert float(tout["macropore_to_drain"]) > 0.0 and float(tout["subirrigation"]) < 0.0
    tin = last.total_in()
    assert float(tin["rain"]) > 0.0 and float(tin["supply"]) > 0.0
    assert float(tin["supply"]) == pytest.approx(float(np.sum(forcing.supply)), rel=1e-12)


@pytest.mark.allow_skip(reason="the negative control is measured against the float64 closure")
def test_uptake_only_ledger_misses_the_other_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pre-H1 single-uptake ledger is off by exactly the unbooked channels, and the check catches it."""
    if not X64:
        pytest.skip("needs float64")
    params, forcing, s0, _ = _inputs()
    monkeypatch.setenv(CHECK_ENV, "1")
    with pytest.raises(RuntimeError, match=r"ledger\.uptake_only"):
        jax.block_until_ready(_run(_model(with_uptake_only=True), params, forcing, s0))
    monkeypatch.setenv(CHECK_ENV, "0")
    ledgers, flux = _run(_model(with_uptake_only=True), params, forcing, s0)
    missing = sum(np.asarray(getattr(flux, k), np.float64) for k in OTHER)
    gap = np.asarray(ledgers["uptake_only"].residual) - np.asarray(ledgers["water"].residual)
    np.testing.assert_allclose(gap, -missing, rtol=0, atol=1e-12)  # unbooked outflows show as a deficit
    assert np.min(np.abs(np.asarray(ledgers["uptake_only"].residual))) > 1e-2
    assert np.max(np.abs(np.asarray(ledgers["water"].residual))) < 1e-10


def _dry_inputs():
    """Dry profile (h = -14000 cm), uptake and tile of 0.5 cm d-1 in the top 6 layers: both are cut at h_min."""
    params = _params(G_DAY, n_sub=48, n_iter=10)
    n = params.richards.grid.n_node
    n_day = 3
    w0 = SoilWater.from_head(jnp.full(n, -14000.0), catpa_soil())
    upt = np.zeros((n_day, n))
    upt[:, :6] = 0.5
    ch = _channels(n, n_day)
    ch["tile"][:, :6] = 0.5
    forcing = ChannelForcing(
        supply=jnp.zeros((n_day, 24)),
        evaporation=jnp.zeros((n_day, 24)),
        uptake=jnp.asarray(upt),
        storm=G.StormForcing(
            ts0=jnp.full(n_day, 24.0), duration=jnp.zeros((n_day, 1)), depth=jnp.zeros((n_day, 1))
        ),
        **{k: jnp.asarray(v) for k, v in ch.items()},
    )
    s0 = LState(
        soil_water=w0,
        ledger={
            "water": soil_water_ledger_init(w0, params.richards.grid),
            "uptake_only": _uptake_only_init(w0, params),
        },
    )
    return params, forcing, s0, ch


@pytest.mark.allow_skip(reason="compared against the float64 balance_error")
def test_ledger_books_channels_after_the_h_min_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the dry end the ledger books the channels after the cut, consistent with the day's own balance."""
    if not X64:
        pytest.skip("needs float64")
    params, forcing, s0, ch = _dry_inputs()
    monkeypatch.setenv(CHECK_ENV, "0")
    ledgers, flux = _run(_model(), params, forcing, s0)
    led = ledgers["water"]
    print("h_min cut: residual", np.asarray(led.residual), "n_clamp", np.asarray(flux.n_clamp))
    # the ledger adds no error of its own: its residual is the day's balance_error
    np.testing.assert_allclose(np.asarray(led.residual), np.asarray(flux.balance_error), rtol=0, atol=1e-12)
    assert float(np.sum(np.asarray(flux.sink_cut))) > 0.0 and float(np.sum(np.asarray(flux.uptake_cut))) > 0.0
    last = jax.tree_util.tree_map(lambda x: x[-1], led)
    for name in ("uptake", "tile"):
        booked = float(last.total_out()[name])
        assert booked == pytest.approx(float(np.sum(np.asarray(getattr(flux, name)))), rel=1e-12), name
    assert float(last.total_out()["tile"]) < float(ch["tile"].sum())


@pytest.mark.allow_skip(reason="the 1e-10 cm closure is a float64 claim")
@pytest.mark.xfail(
    strict=True,
    reason="open (soil-water numerics, not the ledger): the Richards day with the h_min cut active and "
    "head-iterate clamps (n_clamp 475-539 per day) leaves balance_error 1.5e-7 cm/d (measured on rorqual, "
    "n_sub=48, n_iter=10); the ledger reports it exactly",
)
def test_dry_end_day_closes_to_1e_10(monkeypatch: pytest.MonkeyPatch) -> None:
    if not X64:
        pytest.skip("needs float64")
    params, forcing, s0, _ = _dry_inputs()
    monkeypatch.setenv(CHECK_ENV, "0")
    ledgers, _ = _run(_model(), params, forcing, s0)
    assert np.max(np.abs(np.asarray(ledgers["water"].residual))) < 1e-10
