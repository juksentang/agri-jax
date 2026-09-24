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
    raw = _stream([("WC", 1, 9, 7, 1, {"H": np.float64(-10.0)})])
    assert raw[:4] == b"AJDR"
    assert struct.unpack("=i", raw[4:8]) == (1,)
    assert raw[8:40].decode().strip() == "WC"
    phase, call, date, dcall = struct.unpack("=iqii", raw[40:60])
    assert (phase, call, date, dcall) == (1, 9, 7, 1)
    assert raw[60:64] == b"AJDV"
    assert raw[64:128].decode().strip() == "H"
    code, rank, isz = struct.unpack("=iii", raw[128:140])
    assert (code, rank, isz) == (1, 0, 8)
    assert struct.unpack("=d", raw[140:148]) == (-10.0,)
    assert raw[148:] == b"AJDE"


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
    raw[60:64] = b"ZZZZ"
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
