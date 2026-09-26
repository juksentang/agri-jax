"""The skeleton day ``DAY_RZWQM46`` (contract smoke assembly): declaration and lag check, no JAX run.

The day compiles from the processes of :func:`agrijax.models.day_rzwqm46.day_processes` with the
contract's allowed lags (``agrijax.iface.contract.allowed_lags``). It is the contract's whole day
(``agrijax.iface.contract.DAY_TABLE``): weather, management, physcl, plant, ledger. What the check
finds is part of the contract validation and is pinned here:

* all five allowed lags are used: PET reads yesterday's canopy and surface water content, the
  uptake limit yesterday's node uptake and TRWUP, and ROOTWU (``water_supply.maize.rootwu``, a
  module of its own outside ``crops.maize``) yesterday's root record (P2);
* removing any of them, in particular the P2 lag, makes :meth:`Day.check` fail.
"""

from __future__ import annotations

import dataclasses

import pytest

from agrijax.core.day import Day, DayLagError
from agrijax.iface.contract import DAY_TABLE, allowed_lags
from agrijax.models.day_rzwqm46 import SLOT, day_processes, day_rzwqm46, replay_entry

P2 = ("water_supply.maize.rootwu", "iface.root.maize")
USED = {
    ("pet.sw_daily", "iface.canopy.maize"),
    ("pet.sw_daily", "soil_water.theta"),
    ("soil_water.uptake_limit", "iface.root_uptake.maize"),
    ("soil_water.uptake_limit", "iface.crop_water.maize.trwup"),
    P2,
}


def _compiled():
    day = day_rzwqm46(SLOT)
    return day, day.compile(day_processes(SLOT))


def _without(day: Day, pair: tuple[str, str]) -> Day:
    lags = tuple(lag for lag in day.lags if lag.pair != pair)
    assert len(lags) == len(day.lags) - 1
    return dataclasses.replace(day, lags=lags)


def test_day_follows_the_contract_order_and_lags() -> None:
    day, model = _compiled()
    assert day.ref == "rzwqm2-4.6"
    assert [p.name for p in day.phases] == ["weather", "management", "physcl", "plant", "ledger"]
    assert day.entries == tuple(e.name(SLOT) for e in DAY_TABLE)
    assert day.entries[:3] == ("weather.radiation", "events.apply", "crops.maize.season_init")
    assert day.entries[3:7] == ("pet.sw_daily", "snow.prms", "soil_water.uptake_limit", "soil_water.day")
    assert day.entries[-1] == "ledger.close"
    plant = next(p for p in day.phases if p.name == "plant").entries
    assert plant[0] == "n_supply.maize.replay"
    assert (
        plant.index("crops.maize.eop")
        < plant.index("water_supply.maize.rootwu")
        < plant.index("crops.maize.phenology")
    )
    assert (
        plant.index("crops.maize.publish")
        < plant.index("crops.maize.canopy")
        < plant.index("crops.maize.publish_uptake")
    )
    assert model.names == day.entries
    assert {lag.pair for lag in day.lags} == {lag.pair for lag in allowed_lags(SLOT)}


def test_lag_check_uses_all_five_allowed_lags() -> None:
    day, model = _compiled()
    report = day.check(model)
    assert set(report.used) == USED
    assert report.unused_pairs == ()
    # ROOTWU and the crop's publish entry are different modules, so the P2 read is a lag
    assert Day.module_of("water_supply.maize.rootwu") != Day.module_of("crops.maize.publish")
    assert P2 in day.lagged_reads(model) and P2 not in day.carried_reads(model)
    assert model.writers("iface.root.maize") == ("crops.maize.publish",)
    day.check(model, exact_lags=True)


def test_day_check_fails_when_the_p2_lag_is_removed() -> None:
    day, model = _compiled()
    no_p2 = _without(day, P2)
    with pytest.raises(DayLagError, match=r"water_supply\.maize\.rootwu <- iface\.root\.maize"):
        no_p2.check(model)
    with pytest.raises(DayLagError, match=r"water_supply\.maize\.rootwu <- iface\.root\.maize"):
        no_p2.compile(day_processes(SLOT))


@pytest.mark.parametrize("pair", sorted(USED - {P2}))
def test_day_check_fails_when_any_other_used_lag_is_removed(pair: tuple[str, str]) -> None:
    day, model = _compiled()
    with pytest.raises(DayLagError, match="undeclared lags"):
        _without(day, pair).check(model)


def test_every_crop_water_field_has_its_producer_before_the_crop() -> None:
    _, model = _compiled()
    flows = model.dataflow()
    for writer in ("crops.maize.remap_in", "crops.maize.eop", "water_supply.maize.rootwu"):
        assert (writer, "crops.maize.stress", "iface.crop_water.maize") in flows, writer
    assert ("n_supply.maize.replay", "crops.maize.growth", "iface.crop_n.maize") in flows
    assert ("soil_water.uptake_limit", "soil_water.day", "soil_water.sink_in.uptake") in flows
    assert ("snow.prms", "soil_water.day", "iface.snow.melt") in flows
    assert ("water_supply.maize.rootwu", "crops.maize.publish_uptake", "water_supply.maize.rwu") in flows


def test_placeholder_entries_touch_nothing() -> None:
    _, model = _compiled()
    by_name = {p.name: p for p in model.processes}
    for name in ("weather.radiation", "events.apply", "crops.maize.season_init"):
        assert by_name[name].reads == () and by_name[name].writes == (), name


def test_replacing_a_producer_by_a_replay_keeps_the_day() -> None:
    """The replay binding of P1 (the comparison of the smoke test) compiles into the same day."""
    day = day_rzwqm46(SLOT)
    rep = {
        "crops.maize.remap_in": replay_entry(
            "crops.maize.remap_in", {"iface.crop_water.maize.sw": "replay.p1.sw"}
        ),
        "crops.maize.eop": replay_entry("crops.maize.eop", {"iface.crop_water.maize.eop": "replay.p1.eop"}),
        "water_supply.maize.rootwu": replay_entry(
            "water_supply.maize.rootwu",
            {
                "iface.crop_water.maize.trwup": "replay.p1.trwup",
                "water_supply.maize.tss": "replay.p1.tss",
                "water_supply.maize.rwu": "replay.p1.rwu",
            },
        ),
    }
    model = day.compile(day_processes(SLOT, replace=rep))
    report = day.check(model)
    # the replay of ROOTWU reads no root record, so the P2 lag is allowed but unused
    assert set(report.used) == USED - {P2}
    assert report.unused_pairs == (P2,)
