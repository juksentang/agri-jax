"""The owner's writes by execution phase, no data.

A field has one owning module; the owner's extra writes of a field other modules read are declared
(:class:`~agrijax.core.day.PhasedWrite`: entry, path, execution phase, meaning), and a season-end
or reset write runs after the producer and after every same-day consumer.

The fixture is P1 ``trwup`` on a harvest day, in the contract's order: yesterday's value read by the
uptake limit (lag 1), ROOTWU's uptake (the producer), the crop, the daily output record (an entry)
and the water ledger reading the day's value, and ROOTWU's own season end (a fixture process: on
the harvest flag ``TSS = RWU = 0`` and ``trwup = 0``) last:

* on the harvest day the non-zero ``trwup`` is what the crop, the daily output record and the
  ledger see (the ledger closes with it); at the end of the day and the next morning it is 0;
* :meth:`Day.check` rejects an undeclared extra write, a season end before the producer, before
  each same-day consumer (crop, daily output, ledger), and an end-of-day output path that would
  show the reset value; with the season end before the ledger the run shows why (the ledger no
  longer closes and the output is 0).
"""

from __future__ import annotations

import dataclasses
import re

import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import Day, Lag, Phase, bind, compose, process, run
from agrijax.core.day import DayError, DayWriteError, PhasedWrite, snapshot
from agrijax.core.events import EventTable
from agrijax.core.ledger import WaterLedger, water_ledger
from agrijax.core.process import CHECK_ENV
from agrijax.core.state import get_path, set_path
from agrijax.iface.crop import CropWaterIn
from agrijax.processes.water_supply import RootwuState

TRWUP = "iface.crop_water.maize.trwup"
UPTAKE = (0.3, 0.4, 0.5, 0.6)  # the producer's TRWUP of each day [cm d-1]
HARVEST = 1  # day index of the harvest
TRWUP0 = 0.2  # yesterday's TRWUP on the first morning
W0 = 10.0  # soil storage on the first morning [cm]
END = "water_supply.maize.season_end"
DECL = PhasedWrite(END, TRWUP, "season_end", "TRWUP = 0 at the end of a harvest day, after its readers")
LAG = Lag("soil_water.uptake_limit", TRWUP, evidence="fixture: the reader takes the previous day's TRWUP")


def _p(fn, reads, writes):
    return process(fn, reads=reads, writes=writes, register=False, source="fixture")


def _limit(s, p, f):
    """Source: fixture (the uptake limit: what the morning sees of yesterday's TRWUP)."""
    return set_path(s, "soil_water.seen_trwup", get_path(s, TRWUP))


def _rootwu(s, p, f):
    """Source: fixture (ROOTWU: the day's TRWUP and layer uptake, TSS counting days)."""
    t = jnp.asarray(f["uptake"]) * jnp.ones_like(get_path(s, TRWUP))
    s = set_path(s, TRWUP, t)
    s = set_path(
        s, "water_supply.maize.rwu", t[:, None] * jnp.ones_like(get_path(s, "water_supply.maize.rwu"))
    )
    return set_path(s, "water_supply.maize.tss", get_path(s, "water_supply.maize.tss") + 1.0)


def _extract(s, p, f):
    """Source: fixture (the soil loses the day's TRWUP)."""
    return set_path(s, "soil_water.w", get_path(s, "soil_water.w") - jnp.sum(get_path(s, TRWUP)))


def _season_end(s, p, f):
    """Source: fixture (ROOTWU's own season end: on the harvest flag TSS = RWU = 0 and P1 trwup = 0)."""
    h = jnp.asarray(f.harvest)

    def zero(x):
        return jnp.where(h, jnp.zeros_like(x), x)

    return s.replace(tss=zero(s.tss), rwu=zero(s.rwu), water=s.water.replace(trwup=zero(s.water.trwup)))


def _crop(s, p, f):
    """Source: fixture (the crop's water stress input: the day's TRWUP)."""
    return set_path(s, "crops.maize.seen_trwup", get_path(s, TRWUP))


PROCS = {
    "soil_water.uptake_limit": _p(_limit, (TRWUP,), ("soil_water.seen_trwup",)),
    "water_supply.maize.rootwu": _p(
        _rootwu,
        (TRWUP, "water_supply.maize.rwu", "water_supply.maize.tss"),
        (TRWUP, "water_supply.maize.rwu", "water_supply.maize.tss"),
    ),
    "soil_water.extract": _p(_extract, ("soil_water.w", TRWUP), ("soil_water.w",)),
    "crops.maize.stress": _p(_crop, (TRWUP,), ("crops.maize.seen_trwup",)),
    # the daily output record: the day's value copied by an entry before the season end
    "prev.daily_output": snapshot("prev.daily_output", {TRWUP: "prev.output.trwup"}),
    # the water ledger books the day's uptake as TRWUP (it closes only with the day's value)
    "ledger.close": water_ledger(
        storage="soil_water.w", inflows={"rain": "soil_water.rain"}, outflows={"uptake": TRWUP}
    ),
    END: bind(
        process(
            _season_end,
            reads=("tss", "rwu", "water.trwup"),
            writes=("tss", "rwu", "water.trwup"),
            register=False,
            source="fixture",
        ),
        own="water_supply.maize",
        ports={"water": "iface.crop_water.maize"},
        forcing="events",
    ),
}
ORDER = (
    ("physcl", ("soil_water.uptake_limit",)),
    ("plant", ("water_supply.maize.rootwu", "soil_water.extract", "crops.maize.stress", "prev.daily_output")),
    ("ledger", ("ledger.close",)),
    ("season_end", (END,)),
)


def _day(order=ORDER, phased=(DECL,)) -> Day:
    return Day(ref="none", phases=tuple(Phase(n, e) for n, e in order), lags=(LAG,), phased_writes=phased)


def _order_with_end_before(entry: str):
    """``ORDER`` with the season end moved to just before ``entry`` (one phase for the whole day)."""
    flat = [e for _, es in ORDER for e in es if e != END]
    flat.insert(flat.index(entry), END)
    return (("day", tuple(flat)),)


def _state0():
    one = jnp.ones(1)
    return compose(
        {
            "soil_water": {"w": jnp.asarray(W0), "rain": jnp.asarray(0.0), "seen_trwup": jnp.zeros(1)},
            "water_supply.maize": RootwuState(tss=jnp.full((1, 2), 3.0), rwu=jnp.full((1, 2), 0.1)),
            "iface.crop_water.maize": CropWaterIn(sw=jnp.zeros(2), eop=one, trwup=TRWUP0 * one),
            "crops.maize": {"seen_trwup": jnp.zeros(1)},
            "prev.output": {"trwup": jnp.zeros(1)},
            "ledger.water": WaterLedger.init(jnp.asarray(W0), inflows=["rain"], outflows={"uptake": (1,)}),
        }
    )


def _forcing():
    harvest = np.zeros(len(UPTAKE), bool)
    harvest[HARVEST] = True
    events = EventTable.empty(len(UPTAKE)).replace(harvest=jnp.asarray(harvest))
    return {"uptake": jnp.asarray(UPTAKE), "events": events}


def _outputs(s, p, f):
    return {
        "morning": get_path(s, "soil_water.seen_trwup")[0],
        "crop": get_path(s, "crops.maize.seen_trwup")[0],
        "output": get_path(s, "prev.output.trwup")[0],
        "ledger_uptake": get_path(s, "ledger.water").outflow["uptake"].value[0],
        "residual": get_path(s, "ledger.water").residual,
        "end": get_path(s, TRWUP)[0],
        "tss": get_path(s, "water_supply.maize.tss")[0, 0],
        "rwu": get_path(s, "water_supply.maize.rwu")[0, 0],
    }


#: what :func:`_outputs` reads; ``TRWUP`` at the end of the day is the season end's value on purpose
OUT_READS = (
    "soil_water.seen_trwup",
    "crops.maize.seen_trwup",
    "prev.output.trwup",
    "ledger.water",
    TRWUP,
    "water_supply.maize.tss",
    "water_supply.maize.rwu",
)
OUT = {"outputs": _outputs, "output_reads": OUT_READS, "end_of_day_reads": (TRWUP,)}


def _compile(day: Day, procs=PROCS, **kw):
    return day.compile(procs, **{**OUT, **kw})


def _fx(fn, reads, writes):
    return process(fn, reads=reads, writes=writes, register=False, source="fixture")


def _zero_trwup(s, p, f):
    """Source: fixture (another entry zeroing TRWUP)."""
    return set_path(s, TRWUP, 0.0 * get_path(s, TRWUP))


def _keep(s, p, f):
    """Source: fixture (an entry that changes nothing)."""
    return s


def _run(day: Day, *, check: bool = True):
    model = _compile(day, check=check)
    return {k: np.asarray(v) for k, v in run(model, None, _forcing(), _state0()).items()}


def test_harvest_day_trwup_is_seen_by_the_crop_the_output_and_the_ledger_and_the_next_day_starts_from_0() -> (
    None
):
    out = _run(_day())
    up = np.asarray(UPTAKE)
    h = HARVEST
    # the harvest day: the crop, the daily output record and the ledger see the non-zero TRWUP
    assert up[h] > 0.0
    np.testing.assert_allclose(out["crop"], up, rtol=1e-6)
    np.testing.assert_allclose(out["output"], up, rtol=1e-6)
    np.testing.assert_allclose(np.diff(out["ledger_uptake"], prepend=0.0), up, rtol=1e-6)
    np.testing.assert_allclose(out["residual"], 0.0, atol=1e-6)  # the ledger closes with the day's value
    # the end of the harvest day is the season end's; the next morning starts from 0
    assert out["end"][h] == 0.0 and out["tss"][h] == 0.0 and out["rwu"][h] == 0.0
    np.testing.assert_allclose(out["end"][np.arange(len(up)) != h], np.delete(up, h), rtol=1e-6)
    assert out["morning"][h + 1] == 0.0
    np.testing.assert_allclose(out["morning"][: h + 1], [TRWUP0, *up[:h]], rtol=1e-6)
    # after the harvest ROOTWU's own carried TSS restarts from 0 (its own entries write it)
    assert out["tss"][h + 1] == 1.0 and out["tss"][h - 1] == 3.0 + h


def test_an_undeclared_extra_write_of_the_owner_is_rejected() -> None:
    with pytest.raises(
        DayWriteError,
        match=r"the write of iface\.crop_water\.maize\.trwup by water_supply\.maize\.season_end after the "
        r"producer water_supply\.maize\.rootwu is undeclared",
    ):
        _compile(_day(phased=()))
    assert _day().write_phase_problems(_compile(_day())) == []
    assert _day().owner_problems(_compile(_day())) == []


@pytest.mark.parametrize(
    "consumer", ["soil_water.extract", "crops.maize.stress", "prev.daily_output", "ledger.close"]
)
def test_a_season_end_before_a_same_day_consumer_is_rejected(consumer: str) -> None:
    with pytest.raises(DayWriteError, match="before its same-day consumer " + re.escape(consumer)):
        _compile(_day(_order_with_end_before(consumer)))


def test_a_season_end_before_the_ledger_breaks_the_harvest_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why the rule: unchecked, a season end before the ledger leaves the ledger booking 0 on the
    harvest day while the soil lost the day's uptake (the residual is the uptake)."""
    monkeypatch.delenv(CHECK_ENV, raising=False)  # the ledger's closure check would stop the run
    out = _run(_day(_order_with_end_before("ledger.close")), check=False)
    h = HARVEST
    assert out["crop"][h] == pytest.approx(UPTAKE[h])  # the crop still ran before the reset
    np.testing.assert_allclose(out["residual"][h], -UPTAKE[h], rtol=1e-6)
    np.testing.assert_allclose(np.delete(out["residual"], h), 0.0, atol=1e-6)


def test_the_producer_is_the_first_writer_and_undeclared() -> None:
    # the season end before the producer: the first writer is a declared write
    with pytest.raises(DayWriteError, match="the first writer is the undeclared producer"):
        _compile(_day(_order_with_end_before("water_supply.maize.rootwu")))
    # the producer declared (a mislabel) so that the season end would be the producer by elimination
    mislabel = PhasedWrite("water_supply.maize.rootwu", TRWUP, "season_end", "mislabel")
    with pytest.raises(DayWriteError, match="the first writer is the undeclared producer"):
        _compile(_day(_order_with_end_before("crops.maize.stress"), phased=(mislabel,)))


def test_a_daily_extra_write_completes_the_value_before_the_first_consumer() -> None:
    """A declared ``daily`` extra write (DSSAT RATE then INTEGR, as the bucket's truncation) runs
    after the producer and before the first same-day consumer of another module; after any
    consumer it is rejected (the consumers would see two different values of the day)."""
    daily = dataclasses.replace(DECL, when="daily")
    # right after the producer, before the first consumer (soil_water.extract): sanctioned
    _compile(_day(_order_with_end_before("soil_water.extract"), phased=(daily,)))
    for before in ("crops.maize.stress", "prev.daily_output", "ledger.close"):
        with pytest.raises(
            DayWriteError, match=r"runs after its same-day consumer soil_water\.extract \(a daily extra write"
        ):
            _compile(_day(_order_with_end_before(before), phased=(daily,)))
    with pytest.raises(DayWriteError, match="runs after its same-day consumer"):
        _compile(_day(phased=(daily,)))


def test_a_declaration_covers_its_path_only() -> None:
    """A season end writing the whole P1 record is not excused by the ``trwup`` declaration."""
    procs = {**PROCS, END: _fx(_keep, (), ("iface.crop_water.maize",))}
    with pytest.raises(
        DayWriteError, match=r"iface\.crop_water\.maize by water_supply\.maize\.season_end .*undeclared"
    ):
        _compile(_day(), procs)
    assert DECL.covers(END, TRWUP) and DECL.covers(END, TRWUP + ".x")
    assert not DECL.covers(END, "iface.crop_water.maize") and not DECL.covers("x.y", TRWUP)


@pytest.mark.parametrize("where", ["after the consumers", "before the output and the ledger"])
def test_a_second_module_writing_the_path_is_rejected(where: str) -> None:
    """AJ013 on the compiled day: another module zeroing TRWUP, declared or not."""
    procs = {**PROCS, "crops.maize.clobber": _fx(_zero_trwup, (TRWUP,), (TRWUP,))}
    if where == "after the consumers":
        order = (*ORDER, ("crop_end", ("crops.maize.clobber",)))
    else:
        order = (
            ORDER[0],
            ("plant", (*ORDER[1][1][:3], "crops.maize.clobber", ORDER[1][1][3])),
            *ORDER[2:],
        )
    excuse = PhasedWrite("crops.maize.clobber", TRWUP, "season_end", "fixture")
    for day in (_day(order), _day(order, phased=(DECL, excuse))):
        probs = day.owner_problems(_compile(day, procs, check=False))
        assert any(p.startswith("two modules write one path: water_supply.maize.rootwu") for p in probs), (
            probs
        )
        # (Day.check may reject it first as a lag: ROOTWU's read of TRWUP becomes another module's)
        with pytest.raises(DayError):
            _compile(day, procs)


def test_writes_under_another_module_or_against_the_owner_table_are_rejected() -> None:
    procs = {**PROCS, "crops.maize.harvest": _fx(_keep, (), ("water_supply.maize.rwu",))}
    day = _day((*ORDER, ("crop_end", ("crops.maize.harvest",))))
    # (Day.check rejects it first as a lag: ROOTWU's read of its own rwu becomes another module's)
    probs = day.owner_problems(_compile(day, procs, check=False))
    assert any(
        p.startswith(
            "crops.maize.harvest writes water_supply.maize.rwu: the state of module water_supply.maize"
        )
        for p in probs
    ), probs
    with pytest.raises(DayError):
        _compile(day, procs)
    # the contract's owner table: iface paths belong to their port's writer module
    owned = dataclasses.replace(_day(), owners=((TRWUP, "crops.maize"),))
    with pytest.raises(DayWriteError, match=r"is owned by module crops\.maize"):
        _compile(owned)
    _compile(dataclasses.replace(_day(), owners=((TRWUP, "water_supply.maize"),)))


def test_an_unchecked_writer_is_rejected_in_a_day_with_owner_declarations() -> None:
    procs = {**PROCS, "crops.maize.harvest": _zero_trwup}  # a plain callable: writes '*'
    day = _day((*ORDER, ("crop_end", ("crops.maize.harvest",))))
    probs = day.owner_problems(_compile(day, procs, check=False))
    assert any(p.startswith("crops.maize.harvest declares no writes") for p in probs), probs
    with pytest.raises(DayError):
        _compile(day, procs)
    # without owner declarations an unchecked writer stays outside the ownership check
    plain = dataclasses.replace(day, phased_writes=())
    assert not any(
        "declares no writes" in p for p in plain.owner_problems(_compile(plain, procs, check=False))
    )


def test_outputs_must_show_the_day_values() -> None:
    """The outputs are read at the end of the day, after the season end: a path list or declared
    reads that include the reset path would show 0 on the harvest day; the daily output record is
    an entry before the reset (``prev.daily_output``). Undeclared outputs are rejected."""
    day = _day()
    with pytest.raises(DayWriteError, match=r"end-of-day output iface\.crop_water\.maize\.trwup"):
        day.compile(PROCS, outputs=(TRWUP, "soil_water.w"))
    with pytest.raises(DayWriteError, match=r"end-of-day output iface\.crop_water\.maize "):
        day.compile(PROCS, outputs=("iface.crop_water.maize",))
    day.compile(PROCS, outputs=("prev.output.trwup", "soil_water.w"))
    with pytest.raises(DayWriteError, match="the full state"):
        day.compile(PROCS)
    with pytest.raises(DayWriteError, match="a callable without output_reads"):
        day.compile(PROCS, outputs=_outputs)
    with pytest.raises(DayWriteError, match=r"end-of-day output iface\.crop_water\.maize\.trwup"):
        day.compile(PROCS, outputs=_outputs, output_reads=OUT_READS)
    _compile(day)  # the end-of-day TRWUP declared deliberate
    # a day that performs no reset (the season end a stand-in that writes nothing) needs no declaration
    day.compile({**PROCS, END: _fx(_keep, (), ())}, outputs=_outputs)
    with pytest.raises(ValueError, match="callable outputs only"):
        day.compile(PROCS, outputs=(TRWUP,), output_reads=(TRWUP,))
    # end_of_day_reads name declared output paths (or paths below them), never '*' or a parent
    for bad in (("*",), ("iface",), ("iface.crop_water.maize",), ("soil_water.w",)):
        with pytest.raises(ValueError, match="end_of_day_reads"):
            day.compile(PROCS, outputs=_outputs, output_reads=OUT_READS, end_of_day_reads=bad)
    with pytest.raises(ValueError, match="end_of_day_reads"):
        day.compile(PROCS, end_of_day_reads=(TRWUP,))


def test_prev_is_reserved_for_snapshot_entries() -> None:
    procs = {**PROCS, "prev.daily_output": _fx(_keep, (TRWUP,), ("prev.output.trwup",))}
    probs = _day().owner_problems(_compile(_day(), procs, check=False))
    assert any("prev.* holds start-of-day copies" in p for p in probs), probs
    with pytest.raises(DayWriteError, match=r"prev\.\* holds"):
        _compile(_day(), procs)


def test_a_contract_day_uses_the_contract_tables() -> None:
    """The RZWQM2-order and DSSAT days take their port owners and phased writes from the contract; a day
    whose tables differ is rejected, and in a contract day an iface write no port owns is too."""
    from agrijax.iface.contract import dssat_aj013_problems
    from agrijax.models.day_dssat486 import day_dssat486

    rz = _contract_day("maize")
    assert ("iface.crop_water.maize.trwup", "water_supply.maize") in rz.owners
    lie = tuple((p, "crops.maize") if p == "iface.crop_water.maize.trwup" else (p, o) for p, o in rz.owners)
    with pytest.raises(DayError, match="owners differ from the contract"):
        dataclasses.replace(rz, owners=lie)
    ds = day_dssat486("maize")
    assert ("soil_water.sink_in.uptake", "spam") in ds.owners
    assert ("soil_water.evap_layers", "spam") in ds.owners
    assert [(w.entry, w.path, w.when) for w in ds.phased_writes] == [
        ("soil_water.integrate", "soil_water.flux.truncation", "daily")
    ]
    assert dssat_aj013_problems() == []
    procs = _noop_processes(rz)
    h = procs["crops.maize.canopy"]
    procs["crops.maize.canopy"] = dataclasses.replace(h, writes=("iface.nobody.x",))
    probs = rz.owner_problems(rz.compile(procs, check=False))
    assert any("an iface path that no port of the contract owns" in p for p in probs), probs


def _contract_day(slot: str) -> Day:
    """The contract's day in the RZWQM2 4.6 order for ``slot`` (its tables: port owners, lags)."""
    from agrijax.iface.contract import allowed_lags, day_entries

    return Day(
        ref="rzwqm2-4.6",
        phases=tuple(Phase(ph, entries) for ph, entries in day_entries(slot)),
        lags=allowed_lags(slot),
        contract_slot=slot,
    )


def _noop_processes(day: Day) -> dict:
    """A placeholder process for every entry of ``day`` (no reads, no writes)."""
    from agrijax.models.entries import noop_entry

    return {e: noop_entry(e, why="fixture") for ph in day.phases for e in ph.entries}


def test_dssat_aj013_rejects_an_undeclared_or_late_integrate_write(monkeypatch: pytest.MonkeyPatch) -> None:
    import agrijax.iface.contract as c

    monkeypatch.setattr(c, "DSSAT_PHASED_WRITES", ())
    monkeypatch.setattr(c, "DSSAT_SHARED_STATE", {})
    assert c.dssat_aj013_problems() == []  # the truncation is not a port field: nothing to order
    late = (
        dataclasses.replace(
            c.PhasedWrite("soil_water.integrate", "soil_water.flux.truncation", "daily", "x")
        ),
    )
    monkeypatch.setattr(c, "DSSAT_PHASED_WRITES", late)
    monkeypatch.setattr(
        c, "DSSAT_SHARED_STATE", {"soil_water.flux.truncation": ("soil_water.rate", ("spam.xtract",))}
    )
    probs = c.dssat_aj013_problems()
    assert any(
        "daily write of soil_water.integrate runs after the consumer(s) ['spam.xtract']" in p for p in probs
    )
    monkeypatch.setattr(c, "DSSAT_SHARED_STATE", {})
    assert any("without a shared-state row" in p for p in c.dssat_aj013_problems())


def test_phased_writes_are_validated_by_the_day() -> None:
    with pytest.raises(DayError, match="not entries of the day"):
        _day(phased=(PhasedWrite("water_supply.maize.nowhere", TRWUP, "reset", "fixture"),))
    with pytest.raises(DayError, match="declared twice"):
        _day(phased=(DECL, DECL))
    with pytest.raises(DayError, match="state of another module"):
        _day(phased=(PhasedWrite("crops.maize.stress", "water_supply.maize.tss", "reset", "fixture"),))
    with pytest.raises(DayError, match="give a path and a module"):
        dataclasses.replace(_day(), owners=((TRWUP, ""),))


def test_a_lag_reader_sees_the_reset_value_the_next_morning() -> None:
    """By design: a reader of another module before the producer (an allowed lag) is not a same-day
    consumer; the morning after a harvest it sees the season end's 0."""
    procs = {**PROCS, "diag.yesterday": _fx(_keep, (TRWUP,), ())}
    order = (("physcl", ("diag.yesterday", "soil_water.uptake_limit")), *ORDER[1:])
    lag = Lag("diag.yesterday", TRWUP, evidence="fixture")
    d = Day(ref="none", phases=tuple(Phase(n, e) for n, e in order), lags=(LAG, lag), phased_writes=(DECL,))
    _compile(d, procs)


def test_a_contract_is_registered_once_and_its_reference_days_are_contract_days() -> None:
    """A registered contract table cannot be replaced after import, and a
    day of a registered reference is the contract's day unless it says it is a fixture (bare)."""
    from agrijax.core.day import register_contract
    from agrijax.iface.contract import port_owners

    with pytest.raises(DayError, match="already registered"):
        register_contract("rzwqm2-4.6", lambda slot: (port_owners(slot), ()))
    rz = _contract_day("maize")
    # a crop entry writing TRWUP is still rejected (the table was not replaced)
    procs = _noop_processes(rz)
    h = procs["crops.maize.canopy"]
    procs["crops.maize.canopy"] = dataclasses.replace(h, writes=("iface.crop_water.maize.trwup",))
    assert any(
        "is owned by module water_supply.maize" in p
        for p in rz.owner_problems(rz.compile(procs, check=False))
    )
    with pytest.raises(DayError, match="registered contract"):
        Day(ref="rzwqm2-4.6", phases=rz.phases)
    with pytest.raises(DayError, match="registered contract"):
        dataclasses.replace(rz, contract_slot="", owners=(), phased_writes=())
    with pytest.raises(DayError, match="a bare day has no contract_slot"):
        dataclasses.replace(rz, bare=True)
    Day(ref="rzwqm2-4.6", phases=rz.phases, bare=True)
    Day(ref="none", phases=rz.phases)  # an unregistered reference needs neither
