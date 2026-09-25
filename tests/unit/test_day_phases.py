"""The day as reference-declared phases (plan 19 A1): order, same-phase reads, declared lags.

Two synthetic assemblies stand in for the two reference days:

* an RZWQM-order day (management -> physics -> plant -> publish), in which the soil reads the
  root uptake the crop published the day before: one declared lag;
* a DSSAT-order day (RATE then INTEGR), in which the extraction reads the drainage rate the
  soil water balance wrote earlier in the same RATE phase, the soil module contributes two
  entries, and one function serves two entries.

The tests check the compiled order, the ``dataflow()`` edges, that ``stale_reads()`` equals the
declared lags plus the carried state exactly, that undeclared and surplus lags are rejected,
that a lagged reader really sees yesterday's value when the model runs (compared with a NumPy
loop), the ``start_of_day`` discipline, and the ``prev.*`` snapshot.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from agrijax.core import Day, Forcing, Lag, Phase, State, compose, field, process, run, snapshot
from agrijax.core.day import DayError, DayLagError, DayOrderError
from agrijax.core.state import get_path, set_path


# ------------------------------------------------------------------------ the RZWQM-order fixture
class Soil(State):
    w: Array = field(dims=(), unit="cm", description="storage")
    added: Array = field(dims=(), unit="cm", description="management water of the day")


class Crop(State):
    lai: Array = field(dims=(), unit="m2 m-2")


class Uptake(State):
    q: Array = field(dims=(), unit="cm d-1", description="root uptake published for tomorrow")


class Weather(Forcing):
    rain: Array = field(dims="T", unit="cm")
    irr: Array = field(dims="T", unit="cm")


def _p(fn, reads, writes):
    return process(fn, reads=reads, writes=writes, register=False, source="test fixture")


def _events(s, p, f):
    """Source: fixture."""
    return set_path(s, "soil_water.added", f.irr)


def _soil_day(s, p, f):
    """Source: fixture."""
    w = get_path(s, "soil_water.w") + get_path(s, "soil_water.added") + f.rain
    return set_path(s, "soil_water.w", w - get_path(s, "iface.root_uptake.maize.q"))


def _growth(s, p, f):
    """Source: fixture."""
    lai = get_path(s, "crops.maize.lai")
    return set_path(s, "crops.maize.lai", lai + 0.01 * get_path(s, "soil_water.w"))


def _publish(s, p, f):
    """Source: fixture."""
    return set_path(s, "iface.root_uptake.maize.q", 0.1 * get_path(s, "crops.maize.lai"))


RZ_PROCS = {
    "events.apply": _p(_events, (), ("soil_water.added",)),
    "soil_water.day": _p(
        _soil_day, ("soil_water.w", "soil_water.added", "iface.root_uptake.maize.q"), ("soil_water.w",)
    ),
    "crops.maize.growth": _p(_growth, ("soil_water.w", "crops.maize.lai"), ("crops.maize.lai",)),
    "crops.maize.publish_uptake": _p(_publish, ("crops.maize.lai",), ("iface.root_uptake.maize.q",)),
}
UPTAKE_LAG = Lag("soil_water.day", "iface.root_uptake.maize.q", evidence="rz DSSATDRV qsr used the next day")


def rz_day(lags=(UPTAKE_LAG,)) -> Day:
    return Day(
        ref="rzwqm2-4.6",
        phases=(
            Phase("management", ("events.apply",)),
            Phase("physcl", ("soil_water.day",)),
            Phase("plant", ("crops.maize.growth", "crops.maize.publish_uptake")),
        ),
        lags=lags,
    )


def rz_state0():
    return compose(
        {
            "soil_water": Soil(w=jnp.asarray(10.0), added=jnp.asarray(0.0)),
            "crops.maize": Crop(lai=jnp.asarray(0.5)),
            "iface.root_uptake.maize": Uptake(q=jnp.asarray(0.2)),
        }
    )


def test_rzwqm_order_compiles_with_order_dataflow_and_stale_reads() -> None:
    day = rz_day()
    model = day.compile(RZ_PROCS)
    assert model.names == day.entries
    assert model.day is day
    assert [day.phase_of(n) for n in model.names] == ["management", "physcl", "plant", "plant"]
    assert model.dataflow() == [
        ("events.apply", "soil_water.day", "soil_water.added"),
        ("soil_water.day", "crops.maize.growth", "soil_water.w"),
        ("crops.maize.growth", "crops.maize.publish_uptake", "crops.maize.lai"),
    ]
    lagged = set(day.lagged_reads(model))
    carried = set(day.carried_reads(model))
    assert lagged == day.declared_lags() == {("soil_water.day", "iface.root_uptake.maize.q")}
    assert carried == {("soil_water.day", "soil_water.w"), ("crops.maize.growth", "crops.maize.lai")}
    assert set(model.stale_reads()) == lagged | carried
    assert not lagged & carried


def test_undeclared_lag_is_rejected() -> None:
    with pytest.raises(
        DayLagError, match=r"undeclared lags.*soil_water\.day <- iface\.root_uptake\.maize\.q"
    ):
        rz_day(lags=()).compile(RZ_PROCS)


def test_declared_lag_the_order_does_not_produce_is_rejected() -> None:
    bogus = Lag("crops.maize.growth", "soil_water.w", evidence="none")  # written earlier today: fresh
    with pytest.raises(DayLagError, match="declared lags that the order does not produce"):
        rz_day(lags=(UPTAKE_LAG, bogus)).compile(RZ_PROCS)
    carried = Lag("crops.maize.growth", "crops.maize.lai", evidence="none")  # own state, not a lag
    with pytest.raises(DayLagError, match="does not produce"):
        rz_day(lags=(UPTAKE_LAG, carried)).compile(RZ_PROCS)


def test_lagged_reader_sees_yesterdays_value() -> None:
    """Run the compiled day and compare with a NumPy loop that uses yesterday's published uptake."""
    n = 8
    rain = np.array([0.0, 1.2, 0.0, 0.0, 2.5, 0.3, 0.0, 0.7])
    irr = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0])
    f = Weather(rain=jnp.asarray(rain), irr=jnp.asarray(irr))
    model = rz_day().compile(
        RZ_PROCS, outputs=("soil_water.w", "crops.maize.lai", "iface.root_uptake.maize.q")
    )
    out = run(model, None, f, rz_state0())
    w, lai, q = 10.0, 0.5, 0.2
    ws, lais, qs = [], [], []
    for d in range(n):
        w = w + irr[d] + rain[d] - q  # q is yesterday's (the state's value on day 0)
        lai = lai + 0.01 * w
        q = 0.1 * lai
        ws.append(w)
        lais.append(lai)
        qs.append(q)
    np.testing.assert_allclose(np.asarray(out["soil_water.w"]), ws, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out["crops.maize.lai"]), lais, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(out["iface.root_uptake.maize.q"]), qs, rtol=1e-6)
    assert np.asarray(out["soil_water.w"])[0] == pytest.approx(10.0 + 0.0 + 0.0 - 0.2)


def test_entry_and_lag_declarations_are_validated() -> None:
    with pytest.raises(DayError, match="entries without a process"):
        rz_day().compile({k: v for k, v in RZ_PROCS.items() if k != "crops.maize.growth"})
    with pytest.raises(DayError, match="processes not in the day"):
        rz_day().compile({**RZ_PROCS, "crops.maize.extra": RZ_PROCS["crops.maize.growth"]})
    with pytest.raises(DayError, match="entries listed twice"):
        Day("x", (Phase("a", ("m.p",)), Phase("b", ("m.p",))))
    with pytest.raises(DayError, match="duplicate phase names"):
        Day("x", (Phase("a", ("m.p",)), Phase("a", ("m.q",))))
    with pytest.raises(DayError, match="evidence"):
        Lag("m.p", "x.y")
    with pytest.raises(DayError, match="days=1"):
        Lag("m.p", "x.y", days=2, evidence="e")
    with pytest.raises(DayError, match="not entries of the day"):
        Day("x", (Phase("a", ("m.p",)),), lags=(Lag("m.q", "x.y", evidence="e"),))
    with pytest.raises(DayError, match="unknown discipline"):
        Phase("a", ("m.p",), discipline="pure")  # type: ignore[arg-type]
    with pytest.raises(DayError, match="invalid entry"):
        Phase("a", ("m..p",))


def test_compile_accepts_an_iterable_of_named_processes() -> None:
    import dataclasses

    procs = [dataclasses.replace(p, name=k) for k, p in RZ_PROCS.items()]
    model = rz_day().compile(procs)
    assert model.names == rz_day().entries


# ------------------------------------------------------------------------ the DSSAT-order fixture
class DSoil(State):
    sw: Array = field(dims=(), unit="cm")
    swdelts: Array = field(dims=(), unit="cm d-1", description="drainage rate (RATE)")
    swdeltx: Array = field(dims=(), unit="cm d-1", description="root extraction rate (RATE)")
    n_calls: Array = field(dims=(), unit="-", description="management calls (RATE and INTEGR)")


def _watbal_rate(s, p, f):
    """Source: fixture (DSSAT WATBAL RATE: drainage from the start-of-day SW)."""
    return set_path(s, "soil_water.swdelts", -0.1 * get_path(s, "soil_water.sw"))


def _xtract(s, p, f):
    """Source: fixture (DSSAT SPAM XTRACT reads SW + SWDELTS of the same RATE pass)."""
    avail = jnp.maximum(get_path(s, "soil_water.sw") + get_path(s, "soil_water.swdelts"), 0.0)
    return set_path(s, "soil_water.swdeltx", -0.05 * avail)


def _watbal_integr(s, p, f):
    """Source: fixture (DSSAT WATBAL INTEGR applies both rates)."""
    sw = get_path(s, "soil_water.sw")
    return set_path(
        s, "soil_water.sw", sw + get_path(s, "soil_water.swdelts") + get_path(s, "soil_water.swdeltx")
    )


def _mgmt(s, p, f):
    """Source: fixture (one function called in RATE and in INTEGR)."""
    return set_path(s, "soil_water.n_calls", get_path(s, "soil_water.n_calls") + 1.0)


MGMT = _p(_mgmt, ("soil_water.n_calls",), ("soil_water.n_calls",))
DS_PROCS = {
    "mgmt.rate": MGMT,
    "soil.watbal_rate": _p(_watbal_rate, ("soil_water.sw",), ("soil_water.swdelts",)),
    "spam.xtract": _p(_xtract, ("soil_water.sw", "soil_water.swdelts"), ("soil_water.swdeltx",)),
    "soil.watbal_integr": _p(
        _watbal_integr,
        ("soil_water.sw", "soil_water.swdelts", "soil_water.swdeltx"),
        ("soil_water.sw",),
    ),
    "mgmt.integr": MGMT,
}
SPAM_SW_LAG = Lag(
    "spam.xtract", "soil_water.sw", evidence="dssat SPAM.for:270-272 (SW is the start-of-day state)"
)


def ds_day(discipline=None) -> Day:
    return Day(
        ref="dssat-4.8.6.0",
        phases=(
            Phase("rate", ("mgmt.rate", "soil.watbal_rate", "spam.xtract"), discipline=discipline),
            Phase("integr", ("soil.watbal_integr", "mgmt.integr")),
        ),
        lags=(SPAM_SW_LAG,),
    )


def test_dssat_same_phase_read_passes_and_modules_appear_twice() -> None:
    day = ds_day()
    model = day.compile(DS_PROCS)
    assert model.names == (
        "mgmt.rate",
        "soil.watbal_rate",
        "spam.xtract",
        "soil.watbal_integr",
        "mgmt.integr",
    )
    assert ("soil.watbal_rate", "spam.xtract", "soil_water.swdelts") in model.dataflow()
    assert model.processes[0].fn is model.processes[-1].fn  # one function, two entries
    # the soil module reads its own state in RATE and updates it in INTEGR: carried, not a lag
    assert set(day.carried_reads(model)) == {
        ("mgmt.rate", "soil_water.n_calls"),
        ("soil.watbal_rate", "soil_water.sw"),
        ("soil.watbal_integr", "soil_water.sw"),
    }
    assert set(model.stale_reads()) == set(day.lagged_reads(model)) | set(day.carried_reads(model))
    assert set(day.lagged_reads(model)) == {("spam.xtract", "soil_water.sw")}
    s0 = {
        "soil_water": DSoil(
            sw=jnp.asarray(20.0), swdelts=jnp.asarray(0.0), swdeltx=jnp.asarray(0.0), n_calls=jnp.asarray(0.0)
        )
    }
    out = run(model, None, Weather(rain=jnp.zeros(3), irr=jnp.zeros(3)), s0)
    sw = 20.0
    for d in range(3):
        dr = -0.1 * sw
        dx = -0.05 * max(sw + dr, 0.0)
        sw = sw + dr + dx
        assert float(out["soil_water"].sw[d]) == pytest.approx(sw, rel=1e-6)
    np.testing.assert_array_equal(np.asarray(out["soil_water"].n_calls), [2.0, 4.0, 6.0])


def test_same_fixture_fails_under_start_of_day_discipline() -> None:
    with pytest.raises(
        DayOrderError, match=r"\[rate\] spam.xtract reads soil_water.swdelts written earlier by"
    ):
        ds_day(discipline="start_of_day").compile(DS_PROCS)
    # the order check can be deferred and run by hand
    model = ds_day(discipline="start_of_day").compile(DS_PROCS, check=False)
    assert ds_day(discipline="start_of_day").order_violations(model) == [
        ("rate", "soil.watbal_rate", "spam.xtract", "soil_water.swdelts")
    ]


# ------------------------------------------------------------------------------------ snapshot
class Prev(State):
    sw: Array = field(dims=(), unit="cm")


def _overwrite(s, p, f):
    """Source: fixture."""
    return set_path(s, "soil_water.sw", get_path(s, "soil_water.sw") * 0.5)


def _read_start(s, p, f):
    """Source: fixture (needs the start-of-day SW although the soil already overwrote it)."""
    return set_path(s, "soil_water.swdeltx", get_path(s, "prev.soil_water.sw"))


def test_snapshot_gives_later_entries_the_start_of_day_value() -> None:
    day = Day(
        ref="none",
        phases=(Phase("day", ("prev.snapshot", "soil.halve", "crop.start_of_day_read")),),
    )
    procs = {
        "prev.snapshot": snapshot("prev.snapshot", {"soil_water.sw": "prev.soil_water.sw"}),
        "soil.halve": _p(_overwrite, ("soil_water.sw",), ("soil_water.sw",)),
        "crop.start_of_day_read": _p(_read_start, ("prev.soil_water.sw",), ("soil_water.swdeltx",)),
    }
    model = day.compile(procs)
    # the snapshot reads soil_water.sw before soil.halve writes it: a start-of-day copy, not a lag;
    # the reader of prev.* reads what the snapshot wrote earlier today
    assert day.lagged_reads(model) == []
    assert set(day.carried_reads(model)) == {
        ("prev.snapshot", "soil_water.sw"),
        ("soil.halve", "soil_water.sw"),
    }
    assert ("prev.snapshot", "crop.start_of_day_read", "prev.soil_water.sw") in model.dataflow()
    s0 = {
        "soil_water": DSoil(
            sw=jnp.asarray(8.0), swdelts=jnp.asarray(0.0), swdeltx=jnp.asarray(0.0), n_calls=jnp.asarray(0.0)
        ),
        "prev": {"soil_water": Prev(sw=jnp.asarray(-1.0))},
    }
    out = run(model, None, Weather(rain=jnp.zeros(3), irr=jnp.zeros(3)), s0)
    np.testing.assert_allclose(np.asarray(out["soil_water"].swdeltx), [8.0, 4.0, 2.0])
    np.testing.assert_allclose(np.asarray(out["soil_water"].sw), [4.0, 2.0, 1.0])
    with pytest.raises(DayError, match="overlap"):
        snapshot("prev.bad", {"soil_water": "soil_water.sw"})


def test_compiled_day_is_the_plain_model() -> None:
    """Phases add no runtime structure: the compiled day equals the flat model bit for bit."""
    import jax

    from agrijax.core import Model

    f = Weather(rain=jnp.asarray([0.0, 1.0, 2.0]), irr=jnp.asarray([0.5, 0.0, 0.0]))
    a = run(rz_day().compile(RZ_PROCS), None, f, rz_state0())
    b = run(Model(None, list(RZ_PROCS.values())), None, f, rz_state0())
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb) == 4
    for x, y in zip(la, lb, strict=True):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
