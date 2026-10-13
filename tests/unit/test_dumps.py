"""Tests for the dump stream reader and the npz case files (agrijax.port.dumps); data-free."""

from __future__ import annotations

import io
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from agrijax.port import dumps


def _stream(records: Sequence[tuple[str, int, int, int, int, Mapping[str, object]]]) -> bytes:
    buf = io.BytesIO()
    for r in records:
        dumps.write_record(buf, *r)
    return buf.getvalue()


VALUES_IN: dict[str, object] = {
    "THETA": np.array([0.3, 0.31, 0.32]),
    "SOILHP": np.arange(6, dtype=np.float64).reshape(2, 3, order="F"),
    "NN": np.int32(3),
    "START": np.bool_(True),
    "TAG": np.array(b"ab  ", dtype="S4"),
    "SW": np.array([0.1, 0.2], dtype=np.float32),
    "CNT": np.int64(7),
    "CONTROL%YRDOY": np.int32(2015123),
}


def test_write_read_roundtrip_types_and_shapes() -> None:
    raw = _stream([("RICHRD", 0, 5, 2015123, 2, VALUES_IN)])
    recs = list(dumps.iter_records(io.BytesIO(raw), strict=True))
    assert len(recs) == 1
    r = recs[0]
    assert (r.routine, r.phase, r.call, r.date, r.day_call) == ("RICHRD", 0, 5, 2015123, 2)
    assert list(r.values) == list(VALUES_IN)
    for k, v in VALUES_IN.items():
        got = r.values[k]
        want = np.asarray(v)
        assert got.shape == want.shape, k
        assert got.dtype == want.dtype, k
        np.testing.assert_array_equal(got, want)
    # Fortran order: element (i, j) of SOILHP is at i + 2*j in the stream
    assert r.values["SOILHP"][1, 2] == 5.0


def test_stream_layout_is_the_documented_one() -> None:
    buf = io.BytesIO()
    dumps.write_record(buf, "WC", 1, 9, 7, 1, {"H": np.float64(-10.0)}, seq=123)
    raw = buf.getvalue()
    assert raw[:4] == b"AJDR"
    assert struct.unpack("=i", raw[4:8]) == (2,)
    assert raw[8:40].decode().strip() == "WC"
    phase, call, date, dcall = struct.unpack("=iqii", raw[40:60])
    assert (phase, call, date, dcall) == (1, 9, 7, 1)
    assert struct.unpack("=q", raw[60:68]) == (123,)
    assert raw[68:72] == b"AJDV"
    assert raw[72:136].decode().strip() == "H"
    code, rank, isz = struct.unpack("=iii", raw[136:148])
    assert (code, rank, isz) == (1, 0, 8)
    assert struct.unpack("=d", raw[148:156]) == (-10.0,)
    assert raw[156:] == b"AJDE" and len(raw) == 160
    assert next(dumps.iter_records(io.BytesIO(raw))).seq == 123


def test_version_1_streams_are_still_read() -> None:
    """Version 1 (no ``seq``) is what the 2026-09 RICHRD/WC/POINTK/POTEVPHR collection wrote."""
    buf = io.BytesIO()
    dumps.write_record(buf, "WC", 1, 9, 7, 1, {"H": np.float64(-10.0)}, version=1)
    raw = buf.getvalue()
    assert struct.unpack("=i", raw[4:8]) == (1,) and raw[60:64] == b"AJDV" and len(raw) == 152
    (r,) = list(dumps.iter_records(io.BytesIO(raw), strict=True))
    assert (r.routine, r.phase, r.call, r.date, r.day_call, r.seq) == ("WC", 1, 9, 7, 1, -1)
    assert float(r.values["H"]) == -10.0
    with pytest.raises(ValueError, match="unsupported"):
        dumps.write_record(io.BytesIO(), "WC", 1, 9, 7, 1, {}, version=3)


def test_pairing_and_truncation(tmp_path: Path) -> None:
    recs = [
        ("SINK", 0, 1, 10, 1, {"X": np.float64(1.0)}),
        ("SINK", 1, 1, 10, 1, {"X": np.float64(2.0)}),
        ("SINK", 0, 2, 10, 2, {"X": np.float64(3.0)}),
        ("SINK", 0, 3, 11, 1, {"X": np.float64(5.0)}),
        ("SINK", 1, 3, 11, 1, {"X": np.float64(6.0)}),
    ]
    raw = _stream(recs)
    p = tmp_path / "ajdump_SINK.bin"
    p.write_bytes(raw[:-10])  # cut inside the last record
    got = dumps.read_dump(p)
    assert len(got) == 4
    with pytest.raises(dumps.DumpFormatError, match="truncated"):
        dumps.read_dump(p, strict=True)
    p.write_bytes(raw)
    cases = dumps.pair_records(dumps.read_dump(p, strict=True))
    assert [(c.call, float(c.entry["X"]), float(c.exit["X"])) for c in cases] == [
        (1, 1.0, 2.0),
        (3, 5.0, 6.0),
    ]


def test_bad_tags_and_version() -> None:
    with pytest.raises(dumps.DumpFormatError, match="bad record tag"):
        list(dumps.iter_records(io.BytesIO(b"XXXX" + b"\0" * 60)))
    raw = bytearray(_stream([("A", 0, 1, 0, 1, {"X": np.float64(1.0)})]))
    raw[4:8] = struct.pack("=i", 99)
    with pytest.raises(dumps.DumpFormatError, match="version 99"):
        list(dumps.iter_records(io.BytesIO(bytes(raw))))
    raw = bytearray(_stream([("A", 0, 1, 0, 1, {"X": np.float64(1.0)})]))
    raw[68:72] = b"ZZZZ"
    with pytest.raises(dumps.DumpFormatError, match="bad variable tag"):
        list(dumps.iter_records(io.BytesIO(bytes(raw))))


INDEX_ENTRY = {
    "name": "RICHRD",
    "args": ["THETA", "NN", "START"],
    "intent_guess": {"THETA": "inout", "NN": "in", "START": "in"},
    "saved_vars": [{"name": "TH", "kind": "implicit_save"}],
    "common_blocks": [{"name": "HYDROL", "vars": ["SOILHP"]}],
}


def test_npz_case_roundtrip(tmp_path: Path) -> None:
    ent = {"THETA": np.array([0.3, 0.4]), "NN": np.int32(2), "TH": np.zeros(3), "CONTROL%YRDOY": np.int32(5)}
    ext = {**ent, "THETA": np.array([0.31, 0.41])}
    case = dumps.DumpCase("RICHRD", 42, 2015100, 3, ent, ext)
    kinds = dumps.field_kinds(INDEX_ENTRY)
    p = dumps.write_case_npz(case, tmp_path, kinds=kinds, prefix="catpa")
    assert p == tmp_path / "RICHRD" / "catpa_d2015100_c0000000042.npz"
    back = dumps.load_case(p)
    assert (back.routine, back.call, back.date, back.day_call) == ("RICHRD", 42, 2015100, 3)
    np.testing.assert_array_equal(back.exit["THETA"], [0.31, 0.41])
    assert back.entry["NN"].dtype == np.int32
    assert back.fields["THETA"] == {"kind": "arg", "intent": "inout", "block": None}
    assert back.fields["TH"]["kind"] == "implicit_save"
    assert back.fields["CONTROL%YRDOY"]["kind"] == "unknown"
    with np.load(p, allow_pickle=False) as z:
        assert "in.THETA" in z.files and "out.THETA" in z.files
        assert json.loads(str(z["meta.fields"]))["NN"]["intent"] == "in"
    assert dumps.list_cases(tmp_path, "richrd") == [p]
    assert dumps.list_cases(tmp_path, "NOPE") == []


def test_field_kinds_manifest_fills_components() -> None:
    man = {"variables": [{"name": "ISWITCH%ISWNIT", "kind": "arg", "intent": "in", "block": None}]}
    k = dumps.field_kinds(INDEX_ENTRY, man)
    assert k["ISWITCH%ISWNIT"]["kind"] == "arg"
    assert k["SOILHP"] == {"kind": "common", "intent": None, "block": "HYDROL"}


def _case(call: int, date: int) -> dumps.DumpCase:
    return dumps.DumpCase("X", call, date, 1, {}, {})


def test_thin_cases_is_deterministic_and_keeps_last_of_day() -> None:
    cases = [_case(c, 100 + (c - 1) // 10) for c in range(1, 51)]  # 5 dates x 10 calls
    a = dumps.thin_cases(cases, 10)
    assert a == dumps.thin_cases(list(reversed(cases)), 10)
    assert len(a) == 10 and len({c.call for c in a}) == 10
    b = dumps.thin_cases(cases, 8, keep_last_per_date=True)
    assert {10, 20, 30, 40, 50} <= {c.call for c in b} and len(b) == 8
    assert dumps.thin_cases(cases[:3], 10) == cases[:3]


def test_export_writes_one_npz_per_case(tmp_path: Path) -> None:
    raw = _stream(
        [
            ("RICHRD", 0, 1, 2015001, 1, {"THETA": np.array([0.2]), "NN": np.int32(1)}),
            ("RICHRD", 1, 1, 2015001, 1, {"THETA": np.array([0.25]), "NN": np.int32(1)}),
            ("RICHRD", 0, 2, 2015002, 1, {"THETA": np.array([0.25]), "NN": np.int32(1)}),
            ("RICHRD", 1, 2, 2015002, 1, {"THETA": np.array([0.24]), "NN": np.int32(1)}),
            ("RICHRD", 0, 3, 2015002, 2, {"THETA": np.array([0.24]), "NN": np.int32(1)}),
            ("RICHRD", 1, 3, 2015002, 2, {"THETA": np.array([0.23]), "NN": np.int32(1)}),
        ]
    )
    f = tmp_path / "ajdump_RICHRD.bin"
    f.write_bytes(raw)
    paths = dumps.export(f, tmp_path / "dumps", index={"subroutines": [INDEX_ENTRY]})
    assert [p.name for p in paths] == [f"d201500{d}_c000000000{c}.npz" for d, c in ((1, 1), (2, 2), (2, 3))]
    assert dumps.load_case(paths[1]).fields["THETA"]["intent"] == "inout"
    assert [dumps.load_case(p).last_of_date for p in paths] == [True, False, True]
    assert dumps.summary(dumps.read_dump(f))["cases"] == 3
    assert dumps.main(["export", str(f), "--out", str(tmp_path / "cli"), "--max-cases", "1"]) == 0
    assert len(dumps.list_cases(tmp_path / "cli", "RICHRD")) == 1


# ---------------------------------------------------------------------------- daily tables, order
def _day_stream(path: Path) -> None:
    """Three dates, two calls per date, with a run-wide seq shared with a second routine."""
    buf = io.BytesIO()
    seq = 0
    call = 0
    for date in (2015001, 2015002, 2015004):
        for dc in (1, 2):
            call += 1
            for phase in (0, 1):
                seq += 1
                vals = {
                    "THETA": np.full(3, date % 1000 + dc / 10 + phase / 100),
                    "NN": np.int32(3),
                    "BIG": np.zeros((50, 4)),
                }
                dumps.write_record(buf, "PHYSCL", phase, call, date, dc, vals, seq=seq)
            seq += 1  # a record of another routine between the calls
    path.write_bytes(buf.getvalue())


def test_field_selection_skips_values(tmp_path: Path) -> None:
    p = tmp_path / "ajdump_PHYSCL.bin"
    _day_stream(p)
    recs = dumps.read_dump(p, strict=True, fields=["theta"])
    assert len(recs) == 12 and all(list(r.values) == ["THETA"] for r in recs)
    heads = list(dumps.iter_headers(p, strict=True))
    assert [h.seq for h in heads][:4] == [1, 2, 4, 5] and all(h.values == {} for h in heads)


def test_daily_table_first_last_entry_exit(tmp_path: Path) -> None:
    p = tmp_path / "ajdump_PHYSCL.bin"
    _day_stream(p)
    last = dumps.daily_table(p, ["THETA", "NN"], phase="exit", which="last")
    assert last.routine == "PHYSCL" and len(last) == 3
    np.testing.assert_array_equal(last.date, [2015001, 2015002, 2015004])
    np.testing.assert_array_equal(last.n_calls, [2, 2, 2])
    np.testing.assert_allclose(last.values["THETA"][:, 0], [1.21, 2.21, 4.21])
    assert last.values["THETA"].shape == (3, 3) and last.values["NN"].dtype == np.int32
    first = dumps.daily_table(p, ["THETA"], phase=0, which="first", dates=[2015002])
    np.testing.assert_allclose(first.values["THETA"][:, 0], [2.1])
    np.testing.assert_array_equal(first.call, [3])
    np.testing.assert_array_equal(last.index_of([2015002, 2015003, 2015004]), [1, -1, 2])
    sel = dumps.daily_table(p, ["THETA"], which="last", select=lambda r: r.day_call == 1)
    np.testing.assert_allclose(sel.values["THETA"][:, 0], [1.11, 2.11, 4.11])
    np.testing.assert_array_equal(sel.n_calls, [1, 1, 1])
    whole = dumps.daily_table(p, None)
    assert set(whole.values) == {"THETA", "NN", "BIG"} and whole.values["BIG"].shape == (3, 50, 4)
    with pytest.raises(dumps.DumpFormatError, match="missing"):
        dumps.daily_table(p, ["NOPE"])
    with pytest.raises(ValueError):
        dumps.daily_table(p, ["THETA"], phase="middle")
    with pytest.raises(ValueError):
        dumps.daily_table(p, ["THETA"], which="median")


def test_table_npz_roundtrip_and_cli(tmp_path: Path) -> None:
    p = tmp_path / "ajdump_PHYSCL.bin"
    _day_stream(p)
    t = dumps.daily_table(p, ["THETA"], phase="entry")
    out = dumps.save_table(t, tmp_path / "t" / "physcl.npz", meta={"run": "toy"})
    back, meta = dumps.load_table(out)
    assert meta == {"run": "toy"} and back.phase == 0 and back.which == "last"
    np.testing.assert_array_equal(back.values["THETA"], t.values["THETA"])
    np.testing.assert_array_equal(back.seq, t.seq)
    cli = tmp_path / "cli.npz"
    assert dumps.main(["table", str(p), "--out", str(cli), "--field", "NN", "--which", "first"]) == 0
    tab, meta = dumps.load_table(cli)
    assert list(tab.values) == ["NN"] and meta["source"] == str(p)


def test_event_order_merges_streams_by_seq(tmp_path: Path) -> None:
    a, b = tmp_path / "ajdump_A.bin", tmp_path / "ajdump_B.bin"
    ba, bb = io.BytesIO(), io.BytesIO()
    dumps.write_record(ba, "A", 0, 1, 5, 1, {}, seq=1)
    dumps.write_record(bb, "B", 0, 1, 5, 1, {}, seq=2)
    dumps.write_record(bb, "B", 1, 1, 5, 1, {}, seq=3)
    dumps.write_record(ba, "A", 1, 1, 5, 1, {}, seq=4)
    a.write_bytes(ba.getvalue())
    b.write_bytes(bb.getvalue())
    order = dumps.event_order([a, b])
    assert [(str(r["routine"]), int(r["phase"])) for r in order] == [("A", 0), ("B", 0), ("B", 1), ("A", 1)]
    v1 = tmp_path / "ajdump_V1.bin"
    buf = io.BytesIO()
    dumps.write_record(buf, "V1", 0, 1, 5, 1, {}, version=1)
    v1.write_bytes(buf.getvalue())
    with pytest.raises(dumps.DumpFormatError, match="sequence"):
        dumps.event_order([v1])


def test_case_keeps_sequence_numbers(tmp_path: Path) -> None:
    buf = io.BytesIO()
    dumps.write_record(buf, "R", 0, 1, 5, 1, {"X": np.float64(1)}, seq=7)
    dumps.write_record(buf, "R", 1, 1, 5, 1, {"X": np.float64(2)}, seq=9)
    (case,) = dumps.pair_records(list(dumps.iter_records(io.BytesIO(buf.getvalue()))))
    assert (case.seq_in, case.seq_out) == (7, 9)
    back = dumps.load_case(dumps.write_case_npz(case, tmp_path))
    assert (back.seq_in, back.seq_out) == (7, 9)
