"""Read the entry/exit dump streams written by instrumented Fortran and turn them into ``npz`` cases.

The stream format is defined in :mod:`agrijax.port.instrument` (one file ``ajdump_<ROUTINE>.bin``
per routine, records ``AJDR ... AJDE``). One call of a routine gives an entry record and an exit
record with the same call index; :func:`pair_records` joins them into a :class:`DumpCase`, and
:func:`write_case_npz` stores it as ``<root>/<ROUTINE>/<case>.npz`` with the fields

``in.<NAME>`` / ``out.<NAME>``
    entry / exit value of every dumped variable, named as in the Fortran index (upper case;
    derived-type components as ``VAR%COMP``). Reals keep their Fortran kind (``float64`` or
    ``float32``), integers are ``int32`` (or their own kind), logicals ``bool``, characters
    fixed-width bytes. Arrays keep the Fortran shape (column-major data, returned as NumPy arrays
    with ``order='F'`` semantics: element ``[i, j]`` is Fortran ``(i+1, j+1)``).
``meta.routine``, ``meta.call``, ``meta.date``, ``meta.day_call``
    routine name, 1-based call index in the run, simulation date as produced by the date hook
    (``YYYYDDD`` in the reference runs; 0 when no hook set it) and call index within that date.
``meta.seq_in``, ``meta.seq_out``
    (version 2 streams) position of the entry and exit record in the run-wide record sequence
    shared by all instrumented routines, so cases of different routines can be ordered.
``meta.last_of_date``
    whether no later call of the same date was in the dump stream (written by :func:`export`).
``meta.fields``
    JSON: ``{NAME: {"kind": "arg"|"save"|..., "intent": ..., "block": ...}}`` from the index
    record (and the instrumentation manifest when given).

Daily tables
------------
For whole-run comparisons one ``npz`` per call is too many files. :func:`daily_table` streams a
dump file once and keeps, per date, one call (the first or the last of the date) of one phase and
a chosen set of variables, stacked along a leading day axis: ``date [n]``, ``call [n]``,
``seq [n]``, ``n_calls [n]`` (calls of the date in the stream) and one array ``[n, *shape]`` per
variable. :func:`save_table` / :func:`load_table` store it as one ``npz``. :func:`event_order`
merges the record headers of several streams by ``seq`` (the run order of entries and exits).

CLI::

    python -m agrijax.port.dumps summary ajdump_RICHRD.bin
    python -m agrijax.port.dumps export ajdump_RICHRD.bin --out dumps/ --index rzwqm.json \\
        [--manifest manifest.json] [--case-prefix catpa2015] [--max-cases 100]
    python -m agrijax.port.dumps table ajdump_PHYSCL.bin --out physcl_exit.npz --phase exit \\
        --which last --field THETA --field H
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

__all__ = [
    "DailyTable",
    "DumpCase",
    "DumpFormatError",
    "DumpRecord",
    "case_name",
    "daily_table",
    "event_order",
    "field_kinds",
    "iter_headers",
    "iter_records",
    "list_cases",
    "load_case",
    "load_table",
    "pair_records",
    "read_dump",
    "save_table",
    "thin_cases",
    "write_case_npz",
    "write_record",
]

VERSION = 2
SUPPORTED_VERSIONS = (1, 2)
"""Version 2 adds the run-wide record sequence number ``seq`` (int64) after ``day_call``."""
_DTYPES: dict[int, str] = {
    1: "<f8",
    2: "<f4",
    3: "<i4",
    4: "<i8",
    5: "<i2",
    6: "<i4",
    7: "S",
    8: "<i1",
    9: "<c16",
    10: "<c8",
}
_CODE_OF: dict[str, int] = {
    "float64": 1,
    "float32": 2,
    "int32": 3,
    "int64": 4,
    "int16": 5,
    "bool": 6,
    "int8": 8,
}


class DumpFormatError(ValueError):
    """The stream is not a valid ajdump stream."""


@dataclass
class DumpRecord:
    routine: str
    phase: int  # 0 entry, 1 exit
    call: int
    date: int
    day_call: int
    values: dict[str, np.ndarray] = field(default_factory=dict)
    seq: int = -1
    """Run-wide record sequence number (version 2 streams); -1 for version 1."""


@dataclass
class DumpCase:
    routine: str
    call: int
    date: int
    day_call: int
    entry: dict[str, np.ndarray]
    exit: dict[str, np.ndarray]
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_of_date: bool | None = None
    """True when no later call of the same date is in the stream the case came from (so, when
    every call of that date was dumped, the case is the last call of the day)."""
    seq_in: int = -1
    seq_out: int = -1


def _native(dt: str) -> str:
    """Native-endian version of a little-endian dtype string (the stream is native order)."""
    return dt.replace("<", "=") if sys.byteorder == "little" else dt.replace("<", ">")


class _Reader:
    def __init__(self, fh: BinaryIO, name: str) -> None:
        self.fh = fh
        self.name = name

    def take(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError
        return b

    def i4(self, n: int = 1) -> tuple[int, ...]:
        return struct.unpack(f"={n}i", self.take(4 * n))

    def i8(self) -> int:
        return struct.unpack("=q", self.take(8))[0]


def iter_records(
    fh: BinaryIO,
    *,
    name: str = "<stream>",
    strict: bool = False,
    fields: Collection[str] | None = None,
) -> Iterator[DumpRecord]:
    """Records of a stream; a truncated last record ends the iteration (raises when ``strict``).

    With ``fields`` only those variables are decoded (the others are skipped over), which keeps
    whole-run scans of large records cheap; ``fields=()`` reads the headers only."""
    r = _Reader(fh, name)
    want = None if fields is None else {f.upper() for f in fields}
    while True:
        start = fh.tell()
        tag = fh.read(4)
        if not tag:
            return
        try:
            if tag != b"AJDR":
                raise DumpFormatError(f"{name}: bad record tag {tag!r} at byte {start}")
            (ver,) = r.i4()
            if ver not in SUPPORTED_VERSIONS:
                raise DumpFormatError(
                    f"{name}: dump version {ver}, reader supports {', '.join(map(str, SUPPORTED_VERSIONS))}"
                )
            routine = r.take(32).decode("latin-1").strip()
            (phase,) = r.i4()
            call = r.i8()
            date, dcall = r.i4(2)
            seq = r.i8() if ver >= 2 else -1
            rec = DumpRecord(routine, phase, call, date, dcall, seq=seq)
            while True:
                vt = r.take(4)
                if vt == b"AJDE":
                    break
                if vt != b"AJDV":
                    raise DumpFormatError(f"{name}: bad variable tag {vt!r} at byte {fh.tell() - 4}")
                vname = r.take(64).decode("latin-1").strip()
                code, rank, isz = r.i4(3)
                shape = r.i4(rank) if rank else ()
                if code not in _DTYPES:
                    raise DumpFormatError(f"{name}: unknown type code {code} for {vname}")
                n = int(np.prod(shape)) if rank else 1
                if want is not None and vname not in want:
                    fh.seek(n * isz, io.SEEK_CUR)
                    continue
                raw = r.take(n * isz)
                if code == 7:
                    arr = np.frombuffer(raw, dtype=f"S{isz}") if isz > 0 else np.full(n, b"", dtype="S1")
                else:
                    arr = np.frombuffer(raw, dtype=_native(_DTYPES[code]))
                    if code == 6:
                        arr = arr != 0
                arr = arr.reshape(shape, order="F") if rank else arr.reshape(())
                rec.values[vname] = np.array(arr)  # own the memory, writable
            yield rec
        except EOFError:
            if strict:
                raise DumpFormatError(f"{name}: truncated record at byte {start}") from None
            return


def read_dump(
    path: str | Path, *, strict: bool = False, fields: Collection[str] | None = None
) -> list[DumpRecord]:
    """All records of one ``ajdump_<ROUTINE>.bin`` file (``fields``: see :func:`iter_records`)."""
    p = Path(path)
    with p.open("rb") as fh:
        return list(iter_records(fh, name=str(p), strict=strict, fields=fields))


def iter_headers(path: str | Path, *, strict: bool = False) -> Iterator[DumpRecord]:
    """Records of a file without their values (header fields only)."""
    p = Path(path)
    with p.open("rb") as fh:
        yield from iter_records(fh, name=str(p), strict=strict, fields=())


def write_record(
    fh: BinaryIO,
    routine: str,
    phase: int,
    call: int,
    date: int,
    day_call: int,
    values: Mapping[str, Any],
    *,
    seq: int = 0,
    version: int = VERSION,
) -> None:
    """Write one record in the stream format (the Python mirror of the Fortran writer, for tests)."""
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"unsupported dump version {version}")
    fh.write(b"AJDR")
    fh.write(struct.pack("=i", version))
    fh.write(routine.upper().ljust(32)[:32].encode("latin-1"))
    fh.write(struct.pack("=iqii", phase, call, date, day_call))
    if version >= 2:
        fh.write(struct.pack("=q", seq))
    for nm, v in values.items():
        a = np.asarray(v)
        if a.dtype.kind == "S":
            code, isz = 7, a.dtype.itemsize
            data = np.asfortranarray(a).tobytes(order="F")
        elif a.dtype == np.bool_:
            code, isz = 6, 4
            data = np.asarray(a, dtype=np.int32).tobytes(order="F")
        else:
            code = _CODE_OF[a.dtype.name]
            isz = a.dtype.itemsize
            data = a.astype(_native(_DTYPES[code])).tobytes(order="F")
        fh.write(b"AJDV")
        fh.write(nm.upper().ljust(64)[:64].encode("latin-1"))
        fh.write(struct.pack(f"=iii{a.ndim}i", code, a.ndim, isz, *a.shape))
        fh.write(data)
    fh.write(b"AJDE")


def pair_records(records: Sequence[DumpRecord]) -> list[DumpCase]:
    """Join entry and exit records of the same (routine, call); unmatched records are dropped."""
    entries: dict[tuple[str, int], DumpRecord] = {}
    cases: list[DumpCase] = []
    for rec in records:
        k = (rec.routine, rec.call)
        if rec.phase == 0:
            entries[k] = rec
        elif rec.phase == 1 and k in entries:
            e = entries.pop(k)
            cases.append(
                DumpCase(
                    rec.routine,
                    rec.call,
                    e.date,
                    e.day_call,
                    e.values,
                    rec.values,
                    seq_in=e.seq,
                    seq_out=rec.seq,
                )
            )
    cases.sort(key=lambda c: c.call)
    return cases


def field_kinds(
    index_entry: Mapping[str, Any] | None, manifest_entry: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """``{NAME: {"kind", "intent", "block"}}`` from an index record (and the manifest)."""
    out: dict[str, dict[str, Any]] = {}
    if index_entry:
        intents = {k.upper(): v for k, v in index_entry.get("intent_guess", {}).items()}
        intents.update({k.upper(): v for k, v in index_entry.get("intent_declared", {}).items()})
        for a in index_entry.get("args", []):
            out[a.upper()] = {"kind": "arg", "intent": intents.get(a.upper()), "block": None}
        for s in index_entry.get("saved_vars", []):
            out.setdefault(s["name"].upper(), {"kind": s.get("kind", "save"), "intent": None, "block": None})
        for blk in index_entry.get("common_blocks", []):
            for v in blk.get("vars", []):
                out.setdefault(v.upper(), {"kind": "common", "intent": None, "block": blk.get("name")})
    if manifest_entry:
        for v in manifest_entry.get("variables", []):
            out.setdefault(v["name"], {"kind": v["kind"], "intent": v.get("intent"), "block": v.get("block")})
    return out


def _kind_of(name: str, kinds: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    if name in kinds:
        return kinds[name]
    base = name.split("%", 1)[0]
    return kinds.get(base, {"kind": "unknown", "intent": None, "block": None})


def case_name(case: DumpCase, prefix: str = "") -> str:
    """``<prefix>_d<date>_c<call>`` (zero padded, sorts by date then call)."""
    stem = f"d{case.date:07d}_c{case.call:010d}"
    return f"{prefix}_{stem}" if prefix else stem


def write_case_npz(
    case: DumpCase,
    root: str | Path,
    *,
    kinds: Mapping[str, dict[str, Any]] | None = None,
    prefix: str = "",
) -> Path:
    """Write ``<root>/<ROUTINE>/<case>.npz``; returns the path."""
    kinds = kinds or {}
    arrays: dict[str, Any] = {}
    names = list(dict.fromkeys([*case.entry, *case.exit]))
    for nm in case.entry:
        arrays[f"in.{nm}"] = case.entry[nm]
    for nm in case.exit:
        arrays[f"out.{nm}"] = case.exit[nm]
    arrays["meta.routine"] = np.array(case.routine)
    arrays["meta.call"] = np.array(case.call, dtype=np.int64)
    arrays["meta.date"] = np.array(case.date, dtype=np.int32)
    arrays["meta.day_call"] = np.array(case.day_call, dtype=np.int32)
    arrays["meta.fields"] = np.array(json.dumps({nm: _kind_of(nm, kinds) for nm in names}))
    if case.last_of_date is not None:
        arrays["meta.last_of_date"] = np.array(case.last_of_date)
    if case.seq_in >= 0 or case.seq_out >= 0:
        arrays["meta.seq_in"] = np.array(case.seq_in, dtype=np.int64)
        arrays["meta.seq_out"] = np.array(case.seq_out, dtype=np.int64)
    d = Path(root) / case.routine
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{case_name(case, prefix)}.npz"
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    p.write_bytes(buf.getvalue())
    return p


def load_case(path: str | Path) -> DumpCase:
    """Read one ``npz`` case back."""
    with np.load(Path(path), allow_pickle=False) as z:
        entry = {k[3:]: z[k] for k in z.files if k.startswith("in.")}
        exit_ = {k[4:]: z[k] for k in z.files if k.startswith("out.")}
        return DumpCase(
            routine=str(z["meta.routine"]),
            call=int(z["meta.call"]),
            date=int(z["meta.date"]),
            day_call=int(z["meta.day_call"]),
            entry=entry,
            exit=exit_,
            fields=json.loads(str(z["meta.fields"])),
            last_of_date=bool(z["meta.last_of_date"]) if "meta.last_of_date" in z.files else None,
            seq_in=int(z["meta.seq_in"]) if "meta.seq_in" in z.files else -1,
            seq_out=int(z["meta.seq_out"]) if "meta.seq_out" in z.files else -1,
        )


def list_cases(root: str | Path, routine: str) -> list[Path]:
    """Sorted ``npz`` case files of one routine under a dumps root (empty when absent)."""
    d = Path(root) / routine.upper()
    return sorted(d.glob("*.npz")) if d.is_dir() else []


def thin_cases(
    cases: Sequence[DumpCase], max_cases: int, *, keep_last_per_date: bool = False
) -> list[DumpCase]:
    """Deterministic subset of at most ``max_cases`` cases, spread evenly over the call order.

    With ``keep_last_per_date`` the last call of every date is always kept (the call whose exit
    state matches the end-of-day outputs of the model), and the remaining budget is spread over
    the other calls.
    """
    cases = sorted(cases, key=lambda c: c.call)
    if len(cases) <= max_cases:
        return list(cases)
    keep: list[DumpCase] = []
    rest = cases
    if keep_last_per_date:
        last: dict[int, DumpCase] = {}
        for c in cases:
            last[c.date] = c
        keep = sorted(last.values(), key=lambda c: c.call)[:max_cases]
        ids = {c.call for c in keep}
        rest = [c for c in cases if c.call not in ids]
    n = max_cases - len(keep)
    if n > 0 and rest:
        step = len(rest) / n
        keep += [rest[int(i * step + step / 2)] for i in range(min(n, len(rest)))]
    return sorted(keep, key=lambda c: c.call)


def _index_entry(index: Mapping[str, Any] | None, routine: str) -> dict[str, Any] | None:
    if index is None:
        return None
    subs = index.get("subroutines", index)
    items = subs if isinstance(subs, list) else list(subs.values())
    for e in items:
        if str(e.get("name", "")).upper() == routine.upper():
            return dict(e)
    return None


def export(
    dump_file: str | Path,
    out_root: str | Path,
    *,
    index: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    prefix: str = "",
    max_cases: int | None = None,
    keep_last_per_date: bool = False,
) -> list[Path]:
    """Dump stream -> ``npz`` cases under ``out_root/<ROUTINE>/``."""
    cases = pair_records(read_dump(dump_file))
    if not cases:
        return []
    routine = cases[0].routine
    ment = None
    if manifest is not None:
        ment = next((r for r in manifest.get("routines", []) if r["name"] == routine), None)
    kinds = field_kinds(_index_entry(index, routine), ment)
    last: dict[int, int] = {}
    for c in cases:
        last[c.date] = max(last.get(c.date, 0), c.day_call)
    for c in cases:
        c.last_of_date = c.day_call == last[c.date]
    if max_cases is not None:
        cases = thin_cases(cases, max_cases, keep_last_per_date=keep_last_per_date)
    return [write_case_npz(c, out_root, kinds=kinds, prefix=prefix) for c in cases]


@dataclass
class DailyTable:
    """One call per date of one routine and phase, stacked along a leading day axis."""

    routine: str
    phase: int
    which: str
    date: np.ndarray
    call: np.ndarray
    seq: np.ndarray
    n_calls: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.date.shape[0])

    def index_of(self, dates: Sequence[int] | np.ndarray) -> np.ndarray:
        """Row of each date (``-1`` where the table has no row for it)."""
        d = np.asarray(dates, dtype=np.int64)
        pos = np.searchsorted(self.date, d)
        pos = np.clip(pos, 0, max(len(self) - 1, 0))
        hit = (self.date[pos] == d) if len(self) else np.zeros(d.shape, bool)
        return np.where(hit, pos, -1)


def daily_table(
    path: str | Path,
    fields: Collection[str] | None,
    *,
    phase: int | str = "exit",
    which: str = "last",
    dates: Collection[int] | None = None,
    select: Callable[[DumpRecord], bool] | None = None,
    strict: bool = False,
) -> DailyTable:
    """Stream ``path`` once and keep one record per date (see the module docstring).

    ``phase``: ``"entry"``/``0`` or ``"exit"``/``1``. ``which``: ``"first"`` or ``"last"`` call
    of the date. ``fields``: variables to keep (``None`` = all). ``dates``: restrict to these.
    ``select``: keep only records for which it returns true (it sees the decoded ``fields``, so a
    variable it tests must be among them), e.g. the RATE call of a DSSAT routine.
    ``n_calls`` counts the selected calls of the date. A variable missing from a kept record
    raises :class:`DumpFormatError` (a table must be
    rectangular)."""
    ph = {"entry": 0, "exit": 1}.get(str(phase), phase)
    if ph not in (0, 1):
        raise ValueError(f"phase must be entry/exit or 0/1, got {phase!r}")
    if which not in ("first", "last"):
        raise ValueError(f"which must be 'first' or 'last', got {which!r}")
    keep_dates = None if dates is None else {int(d) for d in dates}
    rows: dict[int, DumpRecord] = {}
    counts: dict[int, int] = {}
    routine = ""
    p = Path(path)
    with p.open("rb") as fh:
        for rec in iter_records(fh, name=str(p), strict=strict, fields=fields):
            routine = routine or rec.routine
            if rec.phase != ph or (keep_dates is not None and rec.date not in keep_dates):
                continue
            if select is not None and not select(rec):
                continue
            counts[rec.date] = counts.get(rec.date, 0) + 1
            if which == "last" or rec.date not in rows:
                rows[rec.date] = rec
    ds = sorted(rows)
    recs = [rows[d] for d in ds]
    names = list(fields) if fields is not None else (list(recs[0].values) if recs else [])
    values: dict[str, np.ndarray] = {}
    for nm in (n.upper() for n in names):
        missing = [r.date for r in recs if nm not in r.values]
        if missing:
            raise DumpFormatError(f"{p}: {nm} missing in the records of dates {missing[:5]}")
        values[nm] = np.stack([r.values[nm] for r in recs]) if recs else np.zeros((0,))
    return DailyTable(
        routine=routine,
        phase=int(ph),
        which=which,
        date=np.asarray(ds, dtype=np.int64),
        call=np.asarray([r.call for r in recs], dtype=np.int64),
        seq=np.asarray([r.seq for r in recs], dtype=np.int64),
        n_calls=np.asarray([counts[d] for d in ds], dtype=np.int64),
        values=values,
    )


def save_table(table: DailyTable, path: str | Path, *, meta: Mapping[str, Any] | None = None) -> Path:
    """Write a :class:`DailyTable` (plus optional JSON ``meta``) as one compressed ``npz``."""
    arrays: dict[str, Any] = {f"v.{k}": v for k, v in table.values.items()}
    arrays.update(
        {
            "meta.routine": np.array(table.routine),
            "meta.phase": np.array(table.phase, dtype=np.int32),
            "meta.which": np.array(table.which),
            "meta.json": np.array(json.dumps(dict(meta or {}))),
            "date": table.date,
            "call": table.call,
            "seq": table.seq,
            "n_calls": table.n_calls,
        }
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    p.write_bytes(buf.getvalue())
    return p


def load_table(path: str | Path) -> tuple[DailyTable, dict[str, Any]]:
    """Read a table written by :func:`save_table`; returns ``(table, meta)``."""
    with np.load(Path(path), allow_pickle=False) as z:
        t = DailyTable(
            routine=str(z["meta.routine"]),
            phase=int(z["meta.phase"]),
            which=str(z["meta.which"]),
            date=z["date"],
            call=z["call"],
            seq=z["seq"],
            n_calls=z["n_calls"],
            values={k[2:]: z[k] for k in z.files if k.startswith("v.")},
        )
        return t, json.loads(str(z["meta.json"]))


def event_order(paths: Sequence[str | Path]) -> np.ndarray:
    """Headers of several streams merged by ``seq``: a structured array with fields
    ``seq, routine, phase, call, date, day_call`` (version 2 streams only)."""
    rows: list[tuple[int, str, int, int, int, int]] = []
    for p in paths:
        for h in iter_headers(p):
            if h.seq < 0:
                raise DumpFormatError(f"{p}: version 1 stream has no sequence numbers")
            rows.append((h.seq, h.routine, h.phase, h.call, h.date, h.day_call))
    rows.sort()
    names = ("seq", "routine", "phase", "call", "date", "day_call")
    dt = np.dtype(list(zip(names, ("i8", "U32", "i4", "i8", "i4", "i4"), strict=True)))
    return np.array(rows, dtype=dt)


def summary(records: Sequence[DumpRecord]) -> dict[str, Any]:
    cases = pair_records(records)
    dates = sorted({c.date for c in cases})
    return {
        "records": len(records),
        "cases": len(cases),
        "routines": sorted({r.routine for r in records}),
        "dates": len(dates),
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "variables": len(records[0].values) if records else 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agrijax.port.dumps", description=(__doc__ or "").split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("dump")
    e = sub.add_parser("export")
    e.add_argument("dump", nargs="+")
    e.add_argument("--out", required=True)
    e.add_argument("--index")
    e.add_argument("--manifest")
    e.add_argument("--case-prefix", default="")
    e.add_argument("--max-cases", type=int)
    e.add_argument("--keep-last-per-date", action="store_true")
    t = sub.add_parser("table", help="one call per date, stacked, into one npz")
    t.add_argument("dump")
    t.add_argument("--out", required=True)
    t.add_argument("--phase", default="exit", choices=["entry", "exit"])
    t.add_argument("--which", default="last", choices=["first", "last"])
    t.add_argument("--field", action="append", default=None, help="variable to keep (repeatable)")
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        print(json.dumps(summary(read_dump(a.dump)), indent=1))
        return 0
    if a.cmd == "table":
        tab = daily_table(a.dump, a.field, phase=a.phase, which=a.which)
        save_table(tab, a.out, meta={"source": str(a.dump)})
        print(f"{a.dump}: {len(tab)} dates, {len(tab.values)} variables -> {a.out}")
        return 0
    index = json.loads(Path(a.index).read_text()) if a.index else None
    manifest = json.loads(Path(a.manifest).read_text()) if a.manifest else None
    for d in a.dump:
        paths = export(
            d,
            a.out,
            index=index,
            manifest=manifest,
            prefix=a.case_prefix,
            max_cases=a.max_cases,
            keep_last_per_date=a.keep_last_per_date,
        )
        print(f"{d}: {len(paths)} cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
