"""EventTable: building, round trip, day slicing, csv hook; species / depth / tillage payloads and
the RZWQM2 / DSSAT event readers on synthetic inputs (the reference-input checks are in
``tests/integration/test_events_reference.py``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import lax

from agrijax.core.events import (
    EVENT_REGISTRY,
    FERT_SPECIES,
    NO_PRIMING,
    EventTable,
    decode_code,
    source_code,
)
from agrijax.core.organs import OrganQueue, aggregate, appear, prime, top
from agrijax.io.catpa import CATPA_DIR, EVENT_COLUMNS, write_events

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
            ("2023-04-01", "tillage", 15.0, "cm", "x:1", "implement=tandem disk;operation=secondary"),
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
    assert float(ev.fert_kg_ha[4]) == 180.0 == float(ev.fert_no3_kg_ha[4])
    assert int(ev.fert_method[4]) == source_code("rzwqm2", 1) and float(ev.fert_depth_cm[4]) == 0.0
    assert float(ev.irrig_cm[5]) == pytest.approx(2.5)
    assert (int(ev.priming_lo[13]), int(ev.priming_hi[13])) == (2, 5)
    assert np.flatnonzero(np.asarray(ev.tillage)).tolist() == [0]
    assert float(ev.till_depth_cm[0]) == 15.0 and int(ev.till_operation[0]) == 2
    assert decode_code(int(ev.till_implement[0])) == ("rzwqm2", 5)  # tandem disk
    with pytest.raises(ValueError, match="unknown event kind"):
        EventTable.from_csv(path, DATES, ignore=())
    no_till = EventTable.from_csv(path, DATES, ignore=("pesticide", "tillage"))
    assert not bool(no_till.tillage.any())
    df.loc[0, "detail"] = "implement=disk"
    with pytest.raises(ValueError, match="not an RZWQM2 name"):
        EventTable.from_csv(write_events(df, tmp_path / "bad.csv"), DATES)


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


# ------------------------------------------------------------------ species, depth, codes


def test_fertilizer_species_depth_and_combined_accessor() -> None:
    recs = [
        ("2023-04-05", "fertilizer_no3", 100.0),
        ("2023-04-05", "fertilizer_nh4", 20.0),
        ("2023-04-05", "fertilizer", {"urea": 5.0, "depth_cm": 7.5, "method": "dssat:AP004"}),
        ("2023-04-06", "fert", 3.0),
        ("2023-04-07", "fertilizer_org", 11.0),
    ]
    ev = EventTable.from_records(recs, DATES)
    assert float(ev.fert_no3_kg_ha[4]) == 100.0 and float(ev.fert_nh4_kg_ha[4]) == 20.0
    assert float(ev.fert_urea_kg_ha[4]) == 5.0 and float(ev.fert_depth_cm[4]) == 7.5
    assert int(ev.fert_method[4]) == 2004 == source_code("dssat", "AP004")
    assert float(ev.fert_unspecified_kg_ha[5]) == 3.0 and int(ev.fert_method[5]) == 0
    assert float(ev.fert_org_kg_ha[6]) == 11.0
    # the combined field of the first table: total N per day, also inside jit / scan slices
    np.testing.assert_array_equal(
        np.asarray(ev.fert_kg_ha), sum(np.asarray(getattr(ev, f"fert_{s}_kg_ha")) for s in FERT_SPECIES)
    )
    assert float(ev.fert_kg_ha[4]) == 125.0 and float(jax.jit(lambda e: e.fert_kg_ha.sum())(ev)) == 139.0
    _, per_day = lax.scan(lambda c, e: (c, e.fert_kg_ha), 0, ev)
    np.testing.assert_array_equal(np.asarray(per_day), np.asarray(ev.fert_kg_ha))
    for leaf in jax.tree_util.tree_leaves(ev):
        assert leaf.shape == (20,)
    assert EventTable.field_metadata()["fert_nh4_kg_ha"]["unit"] == "kg ha-1"


def test_same_day_placement_must_agree() -> None:
    base = ("2023-04-05", "fertilizer", {"no3": 10.0, "depth_cm": 5.0, "method": 2001})
    with pytest.raises(ValueError, match="different depth_cm"):
        EventTable.from_records([base, ("2023-04-05", "fertilizer", {"nh4": 1.0, "depth_cm": 6.0})], DATES)
    with pytest.raises(ValueError, match="different method"):
        EventTable.from_records([base, ("2023-04-05", "fertilizer", {"nh4": 1.0, "method": 2004})], DATES)
    ok = EventTable.from_records([base, ("2023-04-05", "fertilizer_nh4", 1.0)], DATES)  # unset: no conflict
    assert float(ok.fert_kg_ha[4]) == 11.0 and int(ok.fert_method[4]) == 2001
    till = ("2023-04-02", "tillage", {"depth_cm": 15.0, "implement": "rzwqm2:1"})
    with pytest.raises(ValueError, match="different implement"):
        EventTable.from_records([till, ("2023-04-02", "till", {"depth_cm": 15.0, "implement": 1002})], DATES)
    with pytest.raises(ValueError, match="no payload key"):
        EventTable.from_records([("2023-04-02", "tillage", {"depth": 1.0})], DATES)


def test_source_codes() -> None:
    assert source_code("rzwqm2", 14) == 1014 and source_code("dssat", "TI003") == 2003
    assert decode_code(2020) == ("dssat", 20) and decode_code(0) is None
    for bad in (("apsim", 1), ("dssat", 0), ("dssat", 1000)):
        with pytest.raises(ValueError):
            source_code(*bad)
    with pytest.raises(ValueError):
        decode_code(3001)
    with pytest.raises(ValueError):
        EventTable.from_records([("2023-04-05", "fertilizer", {"no3": 1.0, "method": "AP004"})], DATES)


def test_tillage_residue_round_trip() -> None:
    recs = [
        *RECORDS,
        ("2023-04-01", "tillage", {"depth_cm": 20.0, "implement": "dssat:TI003"}),
        ("2023-04-03", "till", 12.0),
        (
            "2023-04-03",
            "residue",
            {"amount_kg_ha": 2000.0, "n_kg_ha": 30.0, "depth_cm": 15.0, "incorp_pct": 100.0},
        ),
        ("2023-04-05", "fertilizer", {"urea": 5.0, "depth_cm": 7.5, "method": 2004}),
        ("2023-04-08", "tillage", {}),
    ]
    ev = EventTable.from_records(recs, DATES)
    assert np.flatnonzero(np.asarray(ev.tillage)).tolist() == [0, 2, 7]
    assert float(ev.till_depth_cm[2]) == 12.0 and int(ev.till_implement[2]) == 0
    assert float(ev.residue_n_kg_ha[2]) == 30.0 and float(ev.residue_incorp_pct[2]) == 100.0
    back = EventTable.from_records(ev.to_records(DATES), DATES)
    for (path, a), b in zip(
        jax.tree_util.tree_leaves_with_path(ev), jax.tree_util.tree_leaves(back), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=str(path))
    kinds = {k for _, k, _ in ev.to_records(DATES)}
    assert kinds <= set(EVENT_REGISTRY) and {"tillage", "residue", "fertilizer"} <= kinds


# ------------------------------------------------------------------ RZWQM2 reader (synthetic)

_RZ_DAT = """= PLANT MODEL CONTROL
1
7000  maize TEST
= planting control and harvest parameters
2
1 28 4 2015 76 3 80000 1
3 0 0 0 25 9 2015
30 1.0 3 0 0
1 28 4 2016 76 3 80000 1
3 0 0 0 20 9 2016
30 1.0 3 0 0
= M A N U R E   M A N A G E M E N T
0
= F E R T I L I Z E R   M A N A G E M E N T
3
1 1 0 1 180.0 0.0 0.0 1 1 0 0.0 0.0 0.0
1 1 0 1 90.0 0.0 0.0 1 1 0 0.0 0.0 0.0
1 5 10 5 2016 {meth} 0.0 120.0 0.0 1 1 0 0.0 0.0 0.0
= B M P   M A N A G E M E N T
3 150.0 2.0 36.0 0 10.0 0.0 1.0 1.22 1.0 12.0
= GENERAL PESTICIDE INFORMATION
0
= P E S T I C I D E   M A N A G E M E N T
0
= T I L L A G E   M A N A G E M E N T
2
1 1 3 14 15.0 0.25 1 0.0
1 1 3 5 10.0 0.25 2 0.0
= I R R I G A T I O N   M A N A G E M E N T
1 0 0 0.0
"""


def _rz(meth: int = 3):
    from agrijax.io.rzwqm.dat import RzwqmDat

    return RzwqmDat(_RZ_DAT.format(meth=meth).splitlines(keepends=True))


def test_rzwqm_read_management_synthetic() -> None:
    from agrijax.io.rzwqm.events import read_management

    with pytest.raises(NotImplementedError, match="irrigation"):
        read_management(_rz())
    df = read_management(_rz(), skip=("irrigation",))
    fert = pd.DataFrame(df[df["event"].str.startswith("fertilizer")])
    # relative records fire for both plantings, the same-day duplicate (90 kg) does not
    assert fert["value"].tolist() == [180.0, 180.0, 120.0]
    assert fert["event"].tolist() == ["fertilizer_no3", "fertilizer_no3", "fertilizer_nh4"]
    assert fert["detail"].tolist()[2] == "method=injected-NH3;depth_cm=12"  # BMPTIL
    till = pd.DataFrame(df[df["event"] == "tillage"])
    assert list(pd.DatetimeIndex(till["date"]).strftime("%Y-%m-%d")) == [
        "2015-04-25",
        "2016-04-25",
    ]  # first match only
    assert (till["value"] == 15.0).all()
    dates = pd.date_range("2015-01-01", "2016-12-31", freq="D")
    ev = EventTable.from_frame(df, dates)
    t = dates.get_loc(pd.Timestamp("2016-05-10"))
    assert float(ev.fert_nh4_kg_ha[t]) == 120.0 and float(ev.fert_depth_cm[t]) == 12.0
    assert int(ev.fert_method[t]) == source_code("rzwqm2", 3)
    t = dates.get_loc(pd.Timestamp("2016-04-25"))
    assert int(ev.till_implement[t]) == source_code("rzwqm2", 14) and int(ev.till_operation[t]) == 1
    assert float(ev.fert_kg_ha.sum()) == 480.0 and int(ev.sow.sum()) == 2
    part = read_management(_rz(), "2016-01-01", "2016-04-30", skip=("irrigation",))
    assert part["date"].min() >= pd.Timestamp("2016-01-01") and part["date"].max() <= pd.Timestamp(
        "2016-04-30"
    )
    for m in (2, 5, 6):
        with pytest.raises(NotImplementedError, match=f"method {m}"):
            read_management(_rz(m), skip=("irrigation",))


_RZ_ROTATION = """= PLANT MODEL CONTROL
2
7000  maize TEST
7001  wheat TEST
= planting control and harvest parameters
2
2 20 10 2015 20 2 3500000 1
3 0 0 0 30 4 2016
10 1.0 3 0 0
1 10 5 2016 76 3 80000 1
3 0 0 0 30 9 2016
30 1.0 3 0 0
= M A N U R E   M A N A G E M E N T
0
= F E R T I L I Z E R   M A N A G E M E N T
5
1 5 1 3 2016 1 10.0 0.0 0.0 1 1 0 0.0 0.0 0.0
2 5 2 3 2016 1 20.0 0.0 0.0 1 1 0 0.0 0.0 0.0
1 5 1 5 2016 1 30.0 0.0 0.0 1 1 0 0.0 0.0 0.0
1 5 2 5 2016 1 40.0 0.0 0.0 1 1 0 0.0 0.0 0.0
1 2 5 1 50.0 0.0 0.0 1 1 0 0.0 0.0 0.0
= B M P   M A N A G E M E N T
3 150.0 2.0 36.0 0 10.0 0.0 1.0 1.22 1.0 12.0
= GENERAL PESTICIDE INFORMATION
0
= P E S T I C I D E   M A N A G E M E N T
0
= T I L L A G E   M A N A G E M E N T
1
1 5 2 3 2016 14 15.0 0.25 1 0.0
= I R R I G A T I O N   M A N A G E M E N T
0
"""


def test_rzwqm_plant_reference_gate() -> None:
    """MAQUE checks a record only while its plant reference is the crop being managed."""
    from agrijax.io.rzwqm.dat import RzwqmDat
    from agrijax.io.rzwqm.events import read_management

    dat = RzwqmDat(_RZ_ROTATION.splitlines(keepends=True))

    def applied(df: pd.DataFrame, kind: str) -> dict[str, float]:
        rows = df[df["event"].str.startswith(kind)]
        return dict(zip(pd.DatetimeIndex(rows["date"]).strftime("%m-%d"), rows["value"], strict=True))

    # wheat (ref 2) in the field until its harvest (Apr 30) plus one day (PLANT runs after MAQUE):
    # maize records (ref 1) of Mar 1 and May 1 do not fire, the wheat one of Mar 2 does; from
    # May 2 the next planting (maize, ref 1) is managed; the relative record fires 5 d after it
    df = read_management(dat)
    assert applied(df, "fertilizer") == {"03-02": 20.0, "05-02": 40.0, "05-15": 50.0}
    assert applied(df, "tillage") == {}  # maize tillage on Mar 2, while wheat is managed
    # a run starting in 2016 never planted the 2015 wheat: this year's next planting (maize) is managed
    df = read_management(dat, simulation_start="2016-01-01")
    assert applied(df, "fertilizer") == {"03-01": 10.0, "05-01": 30.0, "05-02": 40.0, "05-15": 50.0}
    assert applied(df, "tillage") == {"03-02": 15.0}


# ------------------------------------------------------------------ DSSAT reader (synthetic)

_FILEX = """*EXP.DETAILS: TEST0101MZ SYNTHETIC

*TREATMENTS                        -------------FACTOR LEVELS------------
@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM
 1 1 0 0 REPORTED DATES             1  1  0  1  1  0  1  1  0  1  0  0  1
 2 1 0 0 DAYS AFTER PLANTING        1  1  0  1  1  0  2  1  0  1  0  0  2

*PLANTING DETAILS
@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL
 1 15100   -99   7.5   7.5     S     R    80     0     7   -99   -99   -99   -99     0

*FERTILIZERS (INORGANIC)
@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME
 1 15100 FE001 AP002    10    60   -99   -99   -99   -99   -99 -99
 1 15130 FE005 AP004     5    40   -99   -99   -99   -99   -99 -99
 1 15120 FE005 AP004     5    30   -99   -99   -99   -99   -99 -99
 2     0 FE005   -99     0    20   -99   -99   -99   -99   -99 -99
 2    30 FE005 AP001     0    25   -99   -99   -99   -99   -99 -99
 3 15100 FE099 AP001     0    25   -99   -99   -99   -99   -99 -99

*RESIDUES AND ORGANIC FERTILIZER
@R RDATE  RCOD  RAMT  RESN  RESP  RESK  RINP  RDEP  RMET RENAME
 1 15095 RE001  2000  1.50   -99   -99   -99    10   -99 -99

*TILLAGE AND ROTATIONS
@T TDATE TIMPL  TDEP TNAME
 1 15098 TI003    30 -99

*SIMULATION CONTROLS
@N GENERAL     NYERS NREPS START SDATE RSEED SNAME....................
 1 GE              1     1     S 15090  2150 SYNTHETIC
@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2
 1 OP              Y     Y     N     N     N     N     N     Y     M
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 1 MA              R     N     R     R     M
@N GENERAL     NYERS NREPS START SDATE RSEED SNAME....................
 2 GE              1     1     S 15090  2150 SYNTHETIC
@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2
 2 OP              Y     Y     N     N     N     N     N     N     M
@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS
 2 MA              R     N     D     N     M
"""


def _ferch_line(n: int, name: str, pct: tuple[float, ...], fert_n: str) -> str:
    """One ``FERCH048.SDA`` row in DSSAT's ``(A5,1X,A35,3X,8F6.0,3X,A3,F6.0,6A6)`` layout."""
    return f"FE{n:03d} {name:<35}   " + "".join(f"{v:6g}" for v in pct) + "   LIN     0" + f"{fert_n:>6}"


def _dssat_inputs(tmp_path):
    from agrijax.io.dssat.events import read_fertilizer_types, read_tillage_depths

    ferch = tmp_path / "FERCH048.SDA"
    ferch.write_text(
        "*Fertilizer properties lookup table\n! comment\n@CDE  Description ...\n"
        + "\n".join(
            [
                _ferch_line(1, "Ammonium nitrate", (50, 50, 0, 0, 0, 0, 0, 0), "34"),
                "!     comment inside the table",
                _ferch_line(5, "Urea", (0, 0, 100, 0, 0, 0, 0, 0), "46.4"),
                _ferch_line(13, "Single super phosphate", (0, 0, 0, 0, 0, 0, 0, 0), "0"),
                _ferch_line(99, "Polymer-coated urea", (0, 0, 100, 0, 0, 0, 0, 30), "44"),
            ]
        )
        + "\n"
    )
    tilop = tmp_path / "TILOP048.SDA"
    tilop.write_text(
        "*Tillage Operations\n! notes\n\n*TI003 Moldboard plow 20 cm depth       One pass\n"
        "@ CN2T  RINP  SSDT  MIXT  HPAN\n -10.0    95   100    90     0\n@  SLB  SBDT  SKST\n"
        " 15.00 -10.0  10.0\n 20.00   5.0  -5.0\n\n*TI004 Chisel plow\n@ CN2T\n -10.0 30 100 40 0\n"
        "@  SLB\n 15.00 -20.0  20.0\n"
    )
    types, depths = read_fertilizer_types(ferch), read_tillage_depths(tilop)
    assert types[1].pct == (50.0, 50.0, 0.0) and types[1].has_n and not types[1].special
    assert not types[13].has_n and types[99].special
    assert depths == {"TI003": 20.0, "TI004": 15.0}
    filex = tmp_path / "TEST0101.MZX"
    filex.write_text(_FILEX)
    return filex, {"fertilizer_types": types, "tillage_depths": depths}


def test_dssat_treatment_management_synthetic(tmp_path) -> None:
    from agrijax.io.dssat.events import read_treatment_management, treatment_events

    filex, kw = _dssat_inputs(tmp_path)
    dates = pd.date_range("2015-04-01", "2015-06-30", freq="D")
    ev = treatment_events(filex, 1, dates, **kw)
    t = {d: dates.get_loc(pd.Timestamp(d)) for d in ("2015-04-10", "2015-05-10", "2015-04-30", "2015-04-05")}
    # FE001: 60 kg N as 50 % NO3 / 50 % NH4, incorporated (AP002) to 10 cm
    assert float(ev.fert_no3_kg_ha[t["2015-04-10"]]) == 30.0 == float(ev.fert_nh4_kg_ha[t["2015-04-10"]])
    assert int(ev.fert_method[t["2015-04-10"]]) == 2002 and float(ev.fert_depth_cm[t["2015-04-10"]]) == 10.0
    assert (
        float(ev.fert_urea_kg_ha[t["2015-05-10"]]) == 40.0 and float(ev.fert_depth_cm[t["2015-05-10"]]) == 5.0
    )
    # the 30 kg row listed after the later-dated 40 kg row never fires (Fert_Place leaves its loop)
    assert float(ev.fert_kg_ha[t["2015-04-30"]]) == 0.0 and float(ev.fert_kg_ha.sum()) == 100.0
    # residue: RINP missing with RDEP > 0 -> 100 % incorporated to at least 15 cm
    r = t["2015-04-05"]
    assert (float(ev.residue_kg_ha[r]), float(ev.residue_n_kg_ha[r])) == (2000.0, 30.0)
    assert (float(ev.residue_incorp_pct[r]), float(ev.residue_depth_cm[r])) == (100.0, 15.0)
    # tillage deeper than the implement's deepest layer is capped at it
    k = dates.get_loc(pd.Timestamp("2015-04-08"))
    assert bool(ev.tillage[k]) and float(ev.till_depth_cm[k]) == 20.0 and int(ev.till_implement[k]) == 2003
    assert np.flatnonzero(np.asarray(ev.sow)).tolist() == [t["2015-04-10"]]

    # treatment 2: FERTI = D (days after planting, 0 = planting day), RESID = N, TILL = N
    ev2 = treatment_events(filex, 2, dates, **kw)
    assert (
        float(ev2.fert_urea_kg_ha[t["2015-04-10"]]) == 20.0 and int(ev2.fert_method[t["2015-04-10"]]) == 2001
    )
    assert float(ev2.fert_urea_kg_ha[t["2015-05-10"]]) == 25.0
    assert float(ev2.residue_kg_ha.sum()) == 0.0 and not bool(ev2.tillage.any())
    with pytest.raises(ValueError, match="no treatment"):
        read_treatment_management(filex, 9, **kw)


def test_dssat_unsupported_inputs_raise(tmp_path) -> None:
    from agrijax.io.dssat.events import read_treatment_management

    _, kw = _dssat_inputs(tmp_path)
    cr = tmp_path / "CR010101.MZX"
    cr.write_text(
        _FILEX.replace(" 1  1  0  1  1  0  1  1  0  1  0  0  1", " 1  1  0  1  1  0  3  1  0  1  0  0  1")
    )
    with pytest.raises(NotImplementedError, match="controlled-release"):
        read_treatment_management(cr, 1, **kw)
    neg = tmp_path / "NG010101.MZX"
    neg.write_text(_FILEX.replace(" 1 15130 FE005 AP004     5    40", " 1 15130 FE005 AP004     5   -99"))
    with pytest.raises(ValueError, match="negative"):
        read_treatment_management(neg, 1, **kw)
    auto = tmp_path / "AU010101.MZX"
    auto.write_text(
        _FILEX.replace(
            " 1 MA              R     N     R     R     M", " 1 MA              R     N     A     R     M"
        )
    )
    with pytest.raises(NotImplementedError, match="fertilizer option 'A'"):
        read_treatment_management(auto, 1, **kw)
