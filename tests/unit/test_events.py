"""EventTable (docs/en/02_architecture.md section 3.5): building, round trip, day slicing, csv hook."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import lax

from agri_jax.core.events import NO_PRIMING, EventTable
from agri_jax.core.organs import OrganQueue, aggregate, appear, prime, top
from agri_jax.io.catpa import CATPA_DIR, EVENT_COLUMNS, write_events

DATES = pd.date_range("2023-04-01", periods=20, freq="D")
RECORDS = [
    ("2023-04-02", "planting", 80000.0),
    ("2023-04-05", "fertilizer_no3", 100.0),
    ("2023-04-05", "fertilizer_nh4", 20.0),
    (pd.Timestamp("2023-04-06"), "irrigation", 2.5),
    ("2023-04-06", "irrig", 0.5),
    ("2023-04-12", "topping", 0.0),
    ("2023-04-14", "priming", (0, 3)),
    ("2023-04-17", "priming", (3, 6)),
    ("2023-04-20", "harvest", 1.0),
]


def test_from_records_expands_per_day() -> None:
    ev = EventTable.from_records(RECORDS, DATES)
    assert ev.n_days == 20
    for leaf in jax.tree_util.tree_leaves(ev):
        assert leaf.shape == (20,)
    assert np.flatnonzero(np.asarray(ev.sow)).tolist() == [1]
    assert np.flatnonzero(np.asarray(ev.harvest)).tolist() == [19]
    assert np.flatnonzero(np.asarray(ev.topping)).tolist() == [11]
    assert float(ev.fert_kg_ha[4]) == 120.0 and float(ev.fert_kg_ha.sum()) == 120.0
    assert float(ev.irrig_cm[5]) == pytest.approx(3.0)
    assert (int(ev.priming_lo[13]), int(ev.priming_hi[13])) == (0, 3)
    assert (int(ev.priming_lo[16]), int(ev.priming_hi[16])) == (3, 6)
    assert int((ev.priming_lo == NO_PRIMING).sum()) == 18
    assert EventTable.field_metadata()["irrig_cm"]["unit"] == "cm"


def test_from_records_errors_and_options() -> None:
    with pytest.raises(ValueError, match="unknown event kind"):
        EventTable.from_records([("2023-04-02", "mowing", 1.0)], DATES)
    ev = EventTable.from_records([("2023-04-02", "mowing", 1.0)], DATES, ignore=("mowing",))
    assert not bool(ev.sow.any())
    with pytest.raises(ValueError, match="outside"):
        EventTable.from_records([("2024-01-01", "sow", 0.0)], DATES)
    ev = EventTable.from_records([("2024-01-01", "sow", 0.0)], DATES, outside="drop")
    assert not bool(ev.sow.any())
    with pytest.raises(ValueError, match="two primings"):
        EventTable.from_records([("2023-04-02", "priming", (0, 1)), ("2023-04-02", "priming", (1, 2))], DATES)
    with pytest.raises(ValueError, match="priming ranks"):
        EventTable.from_records([("2023-04-02", "priming", (3, 1))], DATES)
    assert not bool(jax.tree_util.tree_reduce(lambda a, b: a | jnp.any(b > 0), EventTable.empty(4), False))


def test_round_trip() -> None:
    ev = EventTable.from_records(RECORDS, DATES)
    recs = ev.to_records(DATES)
    kinds = [k for _, k, _ in recs]
    assert kinds == ["sow", "fertilizer", "irrigation", "topping", "priming", "priming", "harvest"]
    back = EventTable.from_records(recs, DATES)
    for a, b in zip(jax.tree_util.tree_leaves(ev), jax.tree_util.tree_leaves(back)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    with pytest.raises(ValueError):
        ev.to_records(DATES[:5])


def test_day_slice_python_and_traced_index() -> None:
    ev = EventTable.from_records(RECORDS, DATES)
    d = ev.day(13)
    assert isinstance(d, EventTable)
    assert d.priming_lo.shape == () and (int(d.priming_lo), int(d.priming_hi)) == (0, 3)
    fert = jax.jit(lambda t: ev.day(t).fert_kg_ha)(jnp.asarray(4))
    assert float(fert) == 120.0
    # the slice lax.scan hands a process equals day(t)
    _, per_day = lax.scan(lambda c, e: (c, e.priming_hi), 0, ev)
    assert int(per_day[16]) == int(ev.day(16).priming_hi) == 6


def test_events_drive_organ_queue_in_scan() -> None:
    """A day step reading the event slice: leaves appear daily, topping caps, primings harvest."""
    ev = EventTable.from_records(RECORDS, DATES)

    def step(q: OrganQueue, e: EventTable) -> tuple[OrganQueue, jax.Array]:
        q = appear(q, True, 10.0, 1.0)
        q = top(q, q.n_active, mask=e.topping)
        q, harvested = prime(q, e.priming_lo, e.priming_hi)
        return q, harvested[0]

    q_final, harvested = jax.jit(lambda q: lax.scan(step, q, ev))(OrganQueue.empty(1, 30))
    assert int(q_final.n_active[0]) == 12  # appeared on days 0..11, topped on day 11
    np.testing.assert_allclose(np.asarray(harvested)[[13, 16]], [3.0, 3.0])
    assert float(harvested.sum()) == 6.0
    assert float(aggregate(q_final, 1.0)["lai"][0]) == pytest.approx(6 * 10.0 * 1e-4)


def test_from_csv_synthetic(tmp_path) -> None:
    df = pd.DataFrame(
        [
            ("2023-04-01", "tillage", 15.0, "cm", "x:1", "implement=disk"),
            ("2023-04-02", "planting", 80000.0, "seeds/ha", "x:2", "plant_ref=1"),
            ("2023-04-03", "pesticide", 2.5, "kg a.i./ha", "x:3", ""),
            ("2023-04-05", "fertilizer_no3", 180.0, "kg/ha", "x:4", "method=broadcast-surface"),
            ("2023-04-06", "irrigation", 25.0, "mm", "x:5", ""),
            ("2023-04-14", "priming", 0.0, "rank", "x:6", "rank_lo=2;rank_hi=5"),
            ("2023-04-20", "harvest", 1.0, "fraction", "x:7", "type=seeds"),
            ("2025-01-01", "harvest", 1.0, "fraction", "x:8", "outside the forcing window"),
        ],
        columns=list(EVENT_COLUMNS),
    )
    df["date"] = pd.to_datetime(df["date"])
    path = write_events(df, tmp_path / "events.csv")
    ev = EventTable.from_csv(path, DATES)
    assert np.flatnonzero(np.asarray(ev.sow)).tolist() == [1]
    assert np.flatnonzero(np.asarray(ev.harvest)).tolist() == [19]
    assert float(ev.fert_kg_ha[4]) == 180.0
    assert float(ev.irrig_cm[5]) == pytest.approx(2.5)
    assert (int(ev.priming_lo[13]), int(ev.priming_hi[13])) == (2, 5)
    with pytest.raises(ValueError, match="unknown event kind"):
        EventTable.from_csv(path, DATES, ignore=())


@pytest.mark.allow_skip(reason="needs the private CA-TPA events.csv under $AGRI_JAX_DATA")
def test_from_csv_catpa() -> None:
    path = CATPA_DIR / "events.csv"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="D")
    ev = EventTable.from_csv(path, dates)
    assert ev.n_days == len(dates)
    sow_days = dates[np.asarray(ev.sow)]
    assert len(sow_days) == 7 and sorted({d.year for d in sow_days}) == list(range(2015, 2022))
    assert int(ev.harvest.sum()) == 7
    np.testing.assert_allclose(np.asarray(ev.fert_kg_ha[ev.fert_kg_ha > 0]), 180.0)
    assert float(ev.irrig_cm.sum()) == 0.0 and not bool(ev.topping.any())
    assert bool((ev.priming_lo == NO_PRIMING).all())
