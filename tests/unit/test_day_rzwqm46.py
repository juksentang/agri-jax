"""The skeleton day ``DAY_RZWQM46`` (contract smoke assembly): declaration and lag check, no JAX run.

The day compiles from the processes of :func:`agrijax.models.day_rzwqm46.day_processes` with the
contract's allowed lags (``agrijax.iface.contract.allowed_lags``). What the check finds is part of
the contract validation and is pinned here:

* four of the five allowed lags are used (PET reads the canopy and the surface water content of
  yesterday; the uptake limit reads yesterday's node uptake and TRWUP);
* the allowed lag of P2 (``crops.maize.rootwu <- iface.root.maize``) is **not** seen as a lag:
  with the contract's entry names ROOTWU and the crop's publish entry are one module
  (``crops.maize``), so Day.check classes the read as the module's own carried state.
"""

from __future__ import annotations

from agrijax.core.day import Day
from agrijax.iface.contract import allowed_lags
from agrijax.models.day_rzwqm46 import SLOT, day_processes, day_rzwqm46, replay_entry

USED = {
    ("pet.sw_daily", "iface.canopy.maize"),
    ("pet.sw_daily", "soil_water.theta"),
    ("soil_water.uptake_limit", "iface.root_uptake.maize"),
    ("soil_water.uptake_limit", "iface.crop_water.maize.trwup"),
}


def _compiled():
    day = day_rzwqm46(SLOT)
    return day, day.compile(day_processes(SLOT))


def test_day_follows_the_contract_order_and_lags() -> None:
    day, model = _compiled()
    assert day.ref == "rzwqm2-4.6"
    assert [p.name for p in day.phases] == ["physcl", "plant", "ledger"]
    assert day.entries[:4] == ("pet.sw_daily", "snow.prms", "soil_water.uptake_limit", "soil_water.day")
    assert day.entries[-1] == "ledger.close"
    assert model.names == day.entries
    assert {lag.pair for lag in day.lags} == {lag.pair for lag in allowed_lags(SLOT)}


def test_lag_check_uses_four_allowed_lags_and_reports_the_p2_lag_unused() -> None:
    day, model = _compiled()
    report = day.check(model)
    assert set(report.used) == USED
    assert report.unused_pairs == (("crops.maize.rootwu", "iface.root.maize"),)
    # the P2 read is there, but as the crop module's own carried state (same module crops.maize)
    assert ("crops.maize.rootwu", "iface.root.maize") in day.carried_reads(model)
    assert Day.module_of("crops.maize.rootwu") == Day.module_of("crops.maize.publish")


def test_every_crop_water_field_has_its_producer_before_the_crop() -> None:
    _, model = _compiled()
    flows = model.dataflow()
    for field_, writer in (("sw", "remap_in"), ("eop", "eop"), ("trwup", "rootwu")):
        assert (f"crops.maize.{writer}", "crops.maize.stress", "iface.crop_water.maize") in flows, field_
    assert ("n_supply.replay", "crops.maize.growth", "iface.crop_n.maize") in flows
    assert ("soil_water.uptake_limit", "soil_water.day", "soil_water.sink_in.uptake") in flows
    assert ("snow.prms", "soil_water.day", "iface.snow.melt") in flows


def test_replacing_a_producer_by_a_replay_keeps_the_day() -> None:
    """The replay binding of P1 (the comparison of the smoke test) compiles into the same day."""
    day = day_rzwqm46(SLOT)
    rep = {
        "crops.maize.remap_in": replay_entry(
            "crops.maize.remap_in", {"iface.crop_water.maize.sw": "replay.p1.sw"}
        ),
        "crops.maize.eop": replay_entry("crops.maize.eop", {"iface.crop_water.maize.eop": "replay.p1.eop"}),
        "crops.maize.rootwu": replay_entry(
            "crops.maize.rootwu",
            {
                "iface.crop_water.maize.trwup": "replay.p1.trwup",
                "crops.maize_rootwu.tss": "replay.p1.tss",
                "crops.maize_rootwu.rwu": "replay.p1.rwu",
            },
        ),
    }
    model = day.compile(day_processes(SLOT, replace=rep))
    report = day.check(model)
    assert set(report.used) == USED
