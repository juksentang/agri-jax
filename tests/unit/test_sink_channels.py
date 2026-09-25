"""The typed sink-channel record of the soil-water step (plan 19 A6, ``soil_water/sinks.py``).

Independent references: exact equality with the single-uptake path (the M1 day), conservation
(storage change against the channel totals summed with NumPy), the closed-form channel totals of
a prescribed daily or per-sub-step rate when no node is near ``h_min``, and the order of the
``h_min`` cut.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.state import get_path
from agrijax.processes.soil_water import (
    SINK_CHANNELS,
    SINK_LEDGER_OUTFLOWS,
    DayConfig,
    GreenAmptConfig,
    GreenAmptParams,
    SinkChannel,
    SinkChannels,
    SoilWaterDayParams,
    StormForcing,
    as_sink_channels,
    soil_water_day_kernel,
)
from agrijax.processes.soil_water.richards import (
    RichardsConfig,
    RichardsParams,
    RichardsState,
    SoilWater,
    richards_day,
)

from .test_richards import CATPA_TLT, catpa_grid, catpa_soil

X64 = bool(jax.config.read("jax_enable_x64"))
REL = 1e-12 if X64 else 2e-5
BAL = 1e-9 if X64 else 2e-3

N = len(CATPA_TLT)
TL = np.diff(np.concatenate([[0.0], CATPA_TLT]))
UPTAKE = np.zeros(N)
UPTAKE[:12] = 0.02  # [cm d-1] per layer, root zone
EVAP = np.where((np.arange(24) >= 6) & (np.arange(24) < 18), 0.02, 0.0)  # [cm h-1]
SUPPLY = np.zeros(24)
SUPPLY[3:5] = 0.4  # [cm h-1], 0.8 cm of prescribed surface supply


def _params() -> RichardsParams:
    return RichardsParams(soil=catpa_soil(), grid=catpa_grid(), config=RichardsConfig(n_sub=24, n_iter=6))


def _water(theta: float = 0.25) -> SoilWater:
    return SoilWater.from_theta(jnp.full(N, theta), catpa_soil())


def _day(sinks: object, theta: float = 0.25) -> SoilWater:
    return richards_day(_water(theta), _params(), jnp.asarray(SUPPLY), jnp.asarray(EVAP), sinks)  # type: ignore[arg-type]


def _zeros_everywhere() -> SinkChannels:
    """Every channel present (daily arrays of zeros) except uptake: forces the stacked path."""
    z = jnp.zeros(N)
    return SinkChannels(
        uptake=SinkChannel(daily=jnp.asarray(UPTAKE), carries_solute=False),
        tile=SinkChannel(daily=z),
        lateral=SinkChannel(daily=z),
        subirrigation=SinkChannel(daily=z),
        macropore_to_drain=SinkChannel(daily=z),
    )


def _storage_change(a: SoilWater, b: SoilWater) -> float:
    return float(np.sum((np.asarray(b.theta, np.float64) - np.asarray(a.theta, np.float64)) * TL))


def test_record_and_names() -> None:
    assert SINK_CHANNELS == ("subirrigation", "uptake", "tile", "lateral", "macropore_to_drain")
    s = SinkChannels.from_uptake(jnp.asarray(UPTAKE))
    assert s.only_daily_uptake and s.daily_uptake_only() is not None
    assert not s.uptake.carries_solute
    assert all(getattr(s, n).carries_solute for n in SINK_CHANNELS if n != "uptake")
    assert s.solute_channels() == ()  # only uptake is active in M3, and it carries no solute
    assert as_sink_channels(s) is s
    assert as_sink_channels(jnp.asarray(UPTAKE)).only_daily_uptake
    t = SinkChannels(tile=SinkChannel(daily=jnp.zeros(N)))
    assert not t.only_daily_uptake and t.daily_uptake_only() is None
    assert t.solute_channels() == ("tile",)
    # every channel has its own daily total in the fluxes, which the ledger reads
    w = _day(s)
    state = RichardsState(soil_water=w)
    for name, path in SINK_LEDGER_OUTFLOWS.items():
        assert np.ndim(get_path(state, path)) == 0, name


def test_array_and_uptake_record_are_bit_identical() -> None:
    """A plain array, ``from_uptake`` and the stacked path with zero channels give the M1 day exactly."""
    a = _day(jnp.asarray(UPTAKE))
    b = _day(SinkChannels.from_uptake(jnp.asarray(UPTAKE)))
    c = _day(_zeros_everywhere())
    for other in (b, c):
        np.testing.assert_array_equal(np.asarray(a.theta), np.asarray(other.theta))
        np.testing.assert_array_equal(np.asarray(a.h), np.asarray(other.h))
    for f in ("infiltration", "evaporation", "drainage", "runoff", "balance_error"):
        assert float(getattr(a.flux, f)) == float(getattr(b.flux, f)), f
        assert float(getattr(a.flux, f)) == float(getattr(c.flux, f)), f
    # the reported uptake of the stacked path is a different reduction of the same numbers
    assert float(c.flux.uptake) == pytest.approx(float(a.flux.uptake), rel=REL)
    for n in ("tile", "lateral", "subirrigation", "macropore_to_drain", "sink_cut"):
        assert float(getattr(a.flux, n)) == 0.0 and float(getattr(c.flux, n)) == 0.0


def test_each_channel_reported_separately_and_balance_closes() -> None:
    tile = np.zeros(N)
    tile[25:30] = 0.01  # [cm d-1]
    lateral = np.zeros(N)
    lateral[15:20] = 0.004
    subirr = np.zeros(N)
    subirr[30:34] = -0.006  # adds water
    mac = np.zeros(N)
    mac[20:22] = 0.003
    s = SinkChannels(
        uptake=SinkChannel(daily=jnp.asarray(UPTAKE), carries_solute=False),
        tile=SinkChannel(daily=jnp.asarray(tile)),
        lateral=SinkChannel(daily=jnp.asarray(lateral)),
        subirrigation=SinkChannel(daily=jnp.asarray(subirr)),
        macropore_to_drain=SinkChannel(daily=jnp.asarray(mac)),
    )
    w0 = _water()
    w1 = _day(s)
    f = w1.flux
    # no node near h_min: every channel is its prescribed daily amount
    for name, amount in (
        ("tile", tile),
        ("lateral", lateral),
        ("subirrigation", subirr),
        ("macropore_to_drain", mac),
    ):
        assert float(getattr(f, name)) == pytest.approx(float(amount.sum()), rel=1e-9 if X64 else 1e-5), name
    assert float(f.uptake) == pytest.approx(float(UPTAKE.sum()), rel=1e-9 if X64 else 1e-5)
    assert float(f.uptake_cut) == 0.0 and float(f.sink_cut) == 0.0
    assert s.solute_channels() == ("subirrigation", "tile", "lateral", "macropore_to_drain")
    # conservation with NumPy sums of the reported channels
    out = sum(float(getattr(f, k)) for k in ("evaporation", "drainage", "runoff", *SINK_CHANNELS))
    ds = _storage_change(w0, w1) + float(w1.pond) - float(w0.pond)
    assert ds == pytest.approx(float(np.sum(SUPPLY)) - out, abs=BAL)
    assert abs(float(f.balance_error)) < BAL


def test_per_substep_callable_equals_its_daily_form() -> None:
    """A callable that returns the daily rate / 24 reproduces the daily channel (rounding only)."""
    tile = np.zeros(N)
    tile[25:30] = 0.01

    def tile_rate(t0: jax.Array, dt: jax.Array, theta: jax.Array, h: jax.Array) -> jax.Array:
        return jnp.asarray(tile / 24.0, theta.dtype) + 0.0 * theta

    daily = SinkChannels(uptake=SinkChannel(daily=jnp.asarray(UPTAKE), carries_solute=False),
                         tile=SinkChannel(daily=jnp.asarray(tile)))  # fmt: skip
    call = SinkChannels(uptake=SinkChannel(daily=jnp.asarray(UPTAKE), carries_solute=False),
                        tile=SinkChannel(rate=tile_rate))  # fmt: skip
    a, b = _day(daily), _day(call)
    np.testing.assert_allclose(np.asarray(a.theta), np.asarray(b.theta), rtol=REL, atol=0)
    assert float(b.flux.tile) == pytest.approx(float(tile.sum()), rel=1e-9 if X64 else 1e-5)

    # a state-dependent callable: drains only what is above 0.2 in the node, closes the balance
    def wet_rate(t0: jax.Array, dt: jax.Array, theta: jax.Array, h: jax.Array) -> jax.Array:
        excess = jnp.where(theta > 0.2, theta - 0.2, 0.0)
        return jnp.zeros_like(theta).at[25:30].set(0.05 * excess[25:30] * TL[25:30])

    w1 = _day(SinkChannels(uptake=SinkChannel(daily=jnp.asarray(UPTAKE), carries_solute=False),
                           tile=SinkChannel(rate=wet_rate)))  # fmt: skip
    assert float(w1.flux.tile) > 0.0
    assert abs(float(w1.flux.balance_error)) < BAL


def test_h_min_cut_order_uptake_before_drains() -> None:
    """At the dry end the uptake takes what is available first; the tile then gets the rest (none)."""
    params = _params()
    w = SoilWater.from_head(jnp.full(N, -14000.0), catpa_soil())
    upt = jnp.zeros(N).at[:6].set(0.5)
    tile = jnp.zeros(N).at[:6].set(0.5)
    s = SinkChannels(uptake=SinkChannel(daily=upt, carries_solute=False), tile=SinkChannel(daily=tile))
    out = richards_day(w, params, jnp.zeros(24), jnp.zeros(24), s)
    ref = richards_day(w, params, jnp.zeros(24), jnp.zeros(24), upt)
    f = out.flux
    assert float(f.uptake) == pytest.approx(float(ref.flux.uptake), rel=1e-6 if X64 else 1e-3)
    assert float(f.tile) < 1e-6 * float(f.uptake)
    assert float(f.sink_cut) == pytest.approx(float(jnp.sum(tile)) - float(f.tile), rel=1e-6)
    assert float(f.uptake) + float(f.uptake_cut) == pytest.approx(float(jnp.sum(upt)), rel=1e-6)


def test_event_day_accepts_the_record_and_reports_channels() -> None:
    """``soil_water_day_kernel``: the record passes through both segments; ledger closes with a tile channel."""
    rp = _params()
    params = SoilWaterDayParams(
        richards=rp,
        infiltration=GreenAmptParams(aef=jnp.asarray(0.9), config=GreenAmptConfig.for_grid(TL)),
        config=DayConfig(n_pre=6, n_post=18),
    )
    storm = StormForcing(
        ts0=jnp.asarray(4.0), duration=jnp.asarray([1.0, 1.0]), depth=jnp.asarray([0.6, 0.3])
    )
    tile = jnp.zeros(N).at[25:30].set(0.01)
    w0 = _water()
    a, _, _ = soil_water_day_kernel(w0, params, jnp.zeros(24), jnp.asarray(EVAP), jnp.asarray(UPTAKE), storm)
    b, _, _ = soil_water_day_kernel(
        w0, params, jnp.zeros(24), jnp.asarray(EVAP), SinkChannels.from_uptake(jnp.asarray(UPTAKE)), storm
    )
    np.testing.assert_array_equal(np.asarray(a.theta), np.asarray(b.theta))
    s = SinkChannels(
        uptake=SinkChannel(daily=jnp.asarray(UPTAKE), carries_solute=False), tile=SinkChannel(daily=tile)
    )
    c, _, _ = soil_water_day_kernel(w0, params, jnp.zeros(24), jnp.asarray(EVAP), s, storm)
    assert float(c.flux.tile) == pytest.approx(float(jnp.sum(tile)), rel=1e-9 if X64 else 1e-5)
    assert abs(float(c.flux.balance_error)) < BAL
    assert float(a.flux.tile) == 0.0


def test_gradient_through_a_channel_is_finite() -> None:
    params = _params()
    w0 = _water()

    def storage(scale: jax.Array) -> jax.Array:
        s = SinkChannels(
            uptake=SinkChannel(daily=jnp.asarray(UPTAKE), carries_solute=False),
            tile=SinkChannel(daily=scale * jnp.zeros(N).at[25:30].set(0.01)),
        )
        w = richards_day(w0, params, jnp.asarray(SUPPLY), jnp.asarray(EVAP), s)
        return w.storage(params.grid)

    g = float(jax.grad(storage)(jnp.asarray(1.0)))
    assert np.isfinite(g)
    # against a central difference; slightly less than the 0.05 cm removed, as drainage responds
    eps = 1e-3 if X64 else 5e-2
    fd = (float(storage(jnp.asarray(1.0 + eps))) - float(storage(jnp.asarray(1.0 - eps)))) / (2 * eps)
    assert g == pytest.approx(fd, rel=1e-5 if X64 else 5e-2)
    assert -0.05 < g < -0.045
