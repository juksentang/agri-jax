"""Instrument Fortran routines with entry/exit state dumps (step 2 of the porting procedure).

Given the index record of a routine (see :mod:`agrijax.port.fortran_index`), this module rewrites
the routine's source so that every call can write the values of

* its explicit arguments,
* the variables that persist between calls (``SAVE``, ``DATA``-initialised, and the locals made
  static by ``-save``; the index lists them in ``saved_vars``),
* the ``COMMON`` members it reads or writes,
* and, for a function, its result,

once at entry and once at exit, to a binary stream file ``ajdump_<ROUTINE>.bin``. The reading
side is :mod:`agrijax.port.dumps`. Nothing is dumped by the routine itself: the generated code
calls the generic ``AJD_PUT`` of a small Fortran module (:func:`ajdump_module_source`), so type,
kind and rank are resolved by the compiler and the instrumenter needs no symbol table beyond the
declarations it scans (array shapes of assumed-size dummies, derived-type components, ``EXTERNAL``
dummies).

Rewrite rules (fixed or free source form)
-----------------------------------------
* ``USE AJDUMP`` is inserted right after the ``SUBROUTINE`` / ``FUNCTION`` statement.
* The entry block (``IF (AJD_BEGIN(id,'NAME')) THEN ... CALL AJD_PUT(...) ... END IF``) goes
  before the first executable statement, after all specification statements, ``DATA`` and
  statement functions.
* Every ``RETURN`` of the routine body (including ``IF (...) RETURN``) becomes a ``GOTO`` to a
  fresh label placed before ``END`` (or ``CONTAINS``); the exit block follows that label. A label
  on the ``END`` statement moves to a ``CONTINUE`` in front of the exit block, so ``GOTO`` targets
  keep their meaning.
* Optionally ``CALL AJD_SETDATE(expr)`` is inserted before the entry block (``date_expr``), or in
  another routine as a date hook (:func:`insert_date_hook`), so records carry the simulation date.

The instrumented code only reads the dumped variables (they are passed to ``INTENT(IN)``
dummies), so a correct instrumentation leaves the outputs of the model unchanged; that is checked
per build by comparing the model outputs of the instrumented and the uninstrumented build.

Which calls are written is decided at run time by ``ajdump.cfg`` in the working directory (or the
file named by the environment variable ``AJDUMP_CFG``), see :func:`write_config`. Without a
configuration file the first 100 calls of each routine are written.

Stream record format (native byte order, ``ACCESS='STREAM'``)::

    record   = 'AJDR' int32 version  char(32) routine  int32 phase (0 entry, 1 exit)
               int64 call_index  int32 date  int32 day_call  { variable }  'AJDE'
    variable = 'AJDV' char(64) name  int32 type_code  int32 rank  int32 itemsize
               int32 shape[rank]  data (itemsize * prod(shape) bytes, column-major)

Type codes are listed in :data:`TYPE_CODES`. Reals are written with their own kind (no floating
point conversion happens in the dump code, so signalling values and FP traps are not an issue);
logicals are written as int32 0/1; characters as their bytes.

CLI::

    python -m agrijax.port.instrument tree --index idx.json --src SRC --out OUT \\
        --routine RICHRD --hook 'PHYSCL=IYYY*1000+JDAY' [--copy-tree] [--line-length 72]
    python -m agrijax.port.instrument module OUT/ajdump.f90
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

__all__ = [
    "DUMP_VERSION",
    "EXIT_LABEL",
    "TYPE_CODES",
    "DateHook",
    "InstrumentError",
    "InstrumentResult",
    "RoutineRequest",
    "VarSpec",
    "ajdump_module_source",
    "find_unit",
    "insert_date_hook",
    "instrument_routine",
    "instrument_tree",
    "main",
    "parse_type_definitions",
    "plan_variables",
    "scan_statements",
    "select_dates",
    "write_config",
]

DUMP_VERSION = 1
EXIT_LABEL = 99971
"""Preferred label of the exit block (the next free one is used if the routine already has it)."""

TYPE_CODES: dict[int, tuple[str, int | None]] = {
    1: ("float64", 8),
    2: ("float32", 4),
    3: ("int32", 4),
    4: ("int64", 8),
    5: ("int16", 2),
    6: ("bool", 4),  # logical of any kind, written as int32 0/1
    7: ("char", None),  # itemsize = character length
    8: ("int8", 1),
    9: ("complex128", 16),
    10: ("complex64", 8),
}
"""Stream type code -> (numpy dtype name, itemsize)."""

MAX_ROUTINES = 64
"""Routine ids are 1..MAX_ROUTINES (size of the tables in the Fortran module)."""

FREE_SUFFIXES = {".f90", ".f95", ".f03", ".f08"}


class InstrumentError(ValueError):
    """The routine cannot be instrumented as requested (unsupported construct, missing unit...)."""


# ---------------------------------------------------------------------------
# Statement scanner
# ---------------------------------------------------------------------------
@dataclass
class Stmt:
    """One Fortran statement, with a map from its characters back to the physical source."""

    text: str
    """Statement text without label, continuation marks and comments (original case)."""
    pos: list[tuple[int, int]]
    """``pos[k]`` = (0-based physical line, column) of ``text[k]``."""
    label: str | None
    first_line: int
    last_line: int
    label_pos: tuple[int, int, int] | None = None  # (line, start col, end col) of the label

    @cached_property
    def key(self) -> str:
        """Upper-case text with blanks removed (blanks are insignificant in fixed form)."""
        return _squeeze(self.text)


def _squeeze(text: str) -> str:
    """Upper-case, blanks removed outside character literals."""
    out: list[str] = []
    quote: str | None = None
    for ch in text:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            out.append(ch)
        elif not ch.isspace():
            out.append(ch.upper())
    return "".join(out)


def _strip_comment(content: str, quote: str | None) -> tuple[int, str | None]:
    """End index of the code part of ``content`` (before a ``!`` comment) and the quote state."""
    for i, ch in enumerate(content):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return i, quote
    return len(content), quote


def _split_semicolons(st: Stmt) -> list[Stmt]:
    parts: list[tuple[int, int]] = []
    quote: str | None = None
    start = 0
    for i, ch in enumerate(st.text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == ";":
            parts.append((start, i))
            start = i + 1
    if not parts:
        return [st]
    parts.append((start, len(st.text)))
    out: list[Stmt] = []
    for n, (a, b) in enumerate(parts):
        text = st.text[a:b]
        if not text.strip():
            continue
        pos = st.pos[a:b]
        out.append(
            Stmt(
                text=text,
                pos=pos,
                label=st.label if n == 0 else None,
                first_line=pos[0][0],
                last_line=pos[-1][0],
                label_pos=st.label_pos if n == 0 else None,
            )
        )
    return out


def _fixed_fields(line: str) -> tuple[str, bool, int] | None:
    """(label field, is continuation, content start column) of a fixed-form line, or None."""
    head = line[:6]
    if "\t" in head:
        t = line.index("\t")
        if line[:t].strip() and not line[:t].strip().isdigit():
            return None
        rest = line[t + 1 : t + 2]
        if rest and rest in "123456789":  # tab format: tab + nonzero digit = continuation
            return line[:t], True, t + 2
        return line[:t], False, t + 1
    label = line[:5]
    cont = line[5:6]
    return label, bool(cont) and cont not in " 0", 6


def scan_statements(
    lines: Sequence[str],
    *,
    free: bool = False,
    line_length: int | None = 72,
    d_lines_comment: bool = True,
) -> list[Stmt]:
    """Logical statements of a source file (fixed or free form), with positions.

    ``line_length`` is the fixed-form statement field limit (``None`` = unlimited, as with
    ``-ffixed-line-length-none``); characters beyond it are ignored, like the compiler does.
    Preprocessor lines (``#...``) are treated as comments.
    """
    stmts: list[Stmt] = []
    cur_text: list[str] = []
    cur_pos: list[tuple[int, int]] = []
    cur_label: str | None = None
    cur_label_pos: tuple[int, int, int] | None = None
    quote: str | None = None
    open_ = False

    def flush() -> None:
        nonlocal cur_text, cur_pos, cur_label, cur_label_pos, open_, quote
        if open_ and "".join(cur_text).strip():
            st = Stmt(
                text="".join(cur_text),
                pos=list(cur_pos),
                label=cur_label,
                first_line=cur_pos[0][0] if cur_pos else 0,
                last_line=cur_pos[-1][0] if cur_pos else 0,
                label_pos=cur_label_pos,
            )
            # trim leading/trailing blanks from text (keep positions aligned)
            a = len(st.text) - len(st.text.lstrip())
            b = len(st.text.rstrip())
            st.text, st.pos = st.text[a:b], st.pos[a:b]
            if st.pos:
                st.first_line, st.last_line = st.pos[0][0], st.pos[-1][0]
            stmts.extend(_split_semicolons(st))
        cur_text, cur_pos, cur_label, cur_label_pos = [], [], None, None
        open_ = False
        quote = None

    free_cont = False
    for ln, raw in enumerate(lines):
        line = raw.rstrip("\r\n")
        if line.lstrip().startswith("#"):
            continue
        if not free:
            if line_length is not None:
                line = line[:line_length]
            if not line.strip():
                continue
            c0 = line[:1]
            if c0 in "Cc*!" or (d_lines_comment and c0 in "Dd"):
                continue
            if line.lstrip().startswith("!") and "\t" not in line[:1]:
                # '!' anywhere in the label field starts a comment line
                idx = len(line) - len(line.lstrip())
                if idx < 6 or not line[:6].strip():
                    continue
            fields = _fixed_fields(line)
            if fields is None:
                continue
            label, is_cont, start = fields
            content = line[start:]
            if is_cont and open_:
                end, quote = _strip_comment(content, quote)
            else:
                flush()
                open_ = True
                lab = label.strip()
                cur_label = lab or None
                if lab:
                    s = label.index(lab[0])
                    cur_label_pos = (ln, s, s + len(lab))
                end, quote = _strip_comment(content, None)
            for k in range(end):
                cur_text.append(content[k])
                cur_pos.append((ln, start + k))
        else:
            stripped = line.strip()
            if not stripped or stripped.startswith("!"):
                continue
            start = len(line) - len(line.lstrip())
            content = line[start:]
            if not free_cont:
                flush()
                open_ = True
                m = re.match(r"(\d{1,5})\s+", content)
                if m:
                    cur_label = m.group(1)
                    cur_label_pos = (ln, start, start + len(m.group(1)))
                    start += m.end()
                    content = line[start:]
            else:
                if content.startswith("&"):
                    start += 1
                    content = line[start:]
            end, quote = _strip_comment(content, quote)
            code = content[:end].rstrip()
            free_cont = code.endswith("&")
            if free_cont:
                code = code[:-1]
            for k in range(len(code)):
                cur_text.append(code[k])
                cur_pos.append((ln, start + k))
    flush()
    return stmts


# ---------------------------------------------------------------------------
# Statement classification
# ---------------------------------------------------------------------------
_SPEC_KW = (
    "USE",
    "IMPLICIT",
    "INTEGER",
    "REAL",
    "DOUBLEPRECISION",
    "DOUBLECOMPLEX",
    "COMPLEX",
    "LOGICAL",
    "CHARACTER",
    "BYTE",
    "TYPE",
    "CLASS",
    "DIMENSION",
    "COMMON",
    "PARAMETER",
    "SAVE",
    "DATA",
    "EXTERNAL",
    "INTRINSIC",
    "EQUIVALENCE",
    "INCLUDE",
    "NAMELIST",
    "ALLOCATABLE",
    "POINTER",
    "TARGET",
    "OPTIONAL",
    "INTENT",
    "VOLATILE",
    "PROCEDURE",
    "IMPORT",
    "AUTOMATIC",
    "STATIC",
    "PUBLIC",
    "PRIVATE",
    "SEQUENCE",
)
_NEUTRAL_KW = ("FORMAT", "ENTRY")
_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
_TYPE_RE = re.compile(
    r"^(INTEGER|REAL|DOUBLEPRECISION|DOUBLECOMPLEX|COMPLEX|LOGICAL|CHARACTER|BYTE)"
    r"(\*\(\*\)|\*\d+|\((?:[^()]|\([^()]*\))*\))?"
)
_DERIVED_RE = re.compile(r"^(?:TYPE|CLASS)\((\w+)\)")


def _match_paren(s: str, i: int) -> int:
    """Index just past the parenthesis group opening at ``s[i]`` (quotes respected)."""
    depth = 0
    quote: str | None = None
    for j in range(i, len(s)):
        ch = s[j]
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(s)


def _split_top(s: str, sep: str = ",") -> list[str]:
    """Split on ``sep`` outside parentheses, brackets and quotes."""
    out: list[str] = []
    depth = 0
    quote: str | None = None
    cur: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            cur.append(ch)
        elif ch in "([":
            depth += 1
            cur.append(ch)
        elif ch in ")]":
            depth -= 1
            cur.append(ch)
        elif depth == 0 and s.startswith(sep, i):
            out.append("".join(cur))
            cur = []
            i += len(sep)
            continue
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


def _assignment(key: str) -> tuple[str, str | None] | None:
    """(target name, first subscript group or None) when ``key`` is an assignment statement."""
    m = re.match(r"[A-Z_][A-Z0-9_%]*", key)
    if not m:
        return None
    i = m.end()
    group: str | None = None
    while i < len(key) and key[i] == "(":
        j = _match_paren(key, i)
        if group is None:
            group = key[i + 1 : j - 1]
        i = j
        # a derived-type component after a subscript: A(1)%B = ...
        m2 = re.match(r"%[A-Z_][A-Z0-9_]*", key[i:])
        if m2:
            i += m2.end()
    if i < len(key) and key[i] == "=" and key[i + 1 : i + 2] not in ("=", ">"):
        return m.group(0), group
    return None


def _is_end(key: str) -> bool:
    return key == "END" or key.startswith(("ENDSUBROUTINE", "ENDFUNCTION", "ENDPROGRAM"))


_HEADER_RE = re.compile(
    r"^(?:(?:RECURSIVE|PURE|ELEMENTAL|IMPURE|MODULE)|"
    r"(?:INTEGER|REAL|DOUBLEPRECISION|DOUBLECOMPLEX|COMPLEX|LOGICAL|CHARACTER)"
    r"(?:\*\(\*\)|\*\d+|\((?:[^()]|\([^()]*\))*\))?|TYPE\(\w+\))*"
    r"(SUBROUTINE|FUNCTION)([A-Z_][A-Z0-9_]*)"
)


def _header_name(key: str) -> tuple[str, str] | None:
    if key.startswith(("CALL", "END")):
        return None
    m = _HEADER_RE.match(key)
    if not m:
        return None
    rest = key[m.end() :]
    if rest and not rest.startswith(("(", "RESULT", "BIND")):
        return None
    return m.group(1), m.group(2)


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------
@dataclass
class Decl:
    """What the instrumenter needs to know about one declared name."""

    name: str
    dims: list[str] | None = None
    derived: str | None = None
    allocatable: bool = False
    pointer: bool = False
    external: bool = False
    parameter: bool = False

    @property
    def assumed_size(self) -> bool:
        return bool(self.dims) and (self.dims[-1] == "*" or self.dims[-1].endswith(":*"))


def _entities(s: str) -> list[tuple[str, str | None]]:
    """``name[(dims)][*len][=init|/init/]`` entities -> [(name, dims or None)]."""
    out: list[tuple[str, str | None]] = []
    for ent in _split_top(s):
        m = _NAME_RE.match(ent)
        if not m:
            continue
        dims = None
        rest = ent[m.end() :]
        if rest.startswith("("):
            j = _match_paren(rest, 0)
            dims = rest[1 : j - 1]
        out.append((m.group(0), dims))
    return out


def _dims_list(d: str | None) -> list[str] | None:
    return None if d is None else [x for x in _split_top(d)]


class _Decls:
    def __init__(self) -> None:
        self.table: dict[str, Decl] = {}

    def get(self, name: str) -> Decl:
        if name not in self.table:
            self.table[name] = Decl(name)
        return self.table[name]

    def add_stmt(self, key: str) -> None:
        if key.startswith("DIMENSION"):
            body = key[len("DIMENSION") :].removeprefix("::")
            for n, d in _entities(body):
                if d is not None:
                    self.get(n).dims = _dims_list(d)
            return
        if key.startswith("COMMON"):
            body = re.sub(r"/[A-Z0-9_]*/", ",", key[len("COMMON") :])
            for n, d in _entities(body):
                dec = self.get(n)
                if d is not None:
                    dec.dims = _dims_list(d)
            return
        if key.startswith("EXTERNAL"):
            for n in _split_top(key[len("EXTERNAL") :].removeprefix("::")):
                if n:
                    self.get(n).external = True
            return
        for kw, attr in (("ALLOCATABLE", "allocatable"), ("POINTER", "pointer")):
            if key.startswith(kw) and not key.startswith(kw + "("):
                for n, d in _entities(key[len(kw) :].removeprefix("::")):
                    dec = self.get(n)
                    setattr(dec, attr, True)
                    if d is not None:
                        dec.dims = _dims_list(d)
                return
        derived: str | None = None
        m = _DERIVED_RE.match(key)
        if m:
            derived = m.group(1)
            rest = key[m.end() :]
        else:
            m = _TYPE_RE.match(key)
            if not m:
                return
            rest = key[m.end() :]
        attrs: list[str] = []
        if "::" in rest:
            parts = _split_top(rest, "::")
            attr_s, rest = parts[0], "::".join(parts[1:])
            attrs = [a for a in _split_top(attr_s) if a]
        elif rest.startswith(","):  # malformed without '::'
            return
        default_dims: str | None = None
        flags = {"allocatable": False, "pointer": False, "external": False, "parameter": False}
        for a in attrs:
            if a.startswith("DIMENSION("):
                default_dims = a[len("DIMENSION(") : -1]
            elif a in ("ALLOCATABLE", "POINTER", "EXTERNAL", "PARAMETER"):
                flags[a.lower()] = True
        for n, d in _entities(rest):
            dec = self.get(n)
            if derived:
                dec.derived = derived
            dd = d if d is not None else default_dims
            if dd is not None:
                dec.dims = _dims_list(dd)
            for k, v in flags.items():
                if v:
                    setattr(dec, k, True)


@dataclass
class TypeComponent:
    name: str
    derived: str | None
    dims: list[str] | None
    allocatable: bool
    pointer: bool


def parse_type_definitions(
    texts: Iterable[str], *, free: bool = False, line_length: int | None = None
) -> dict[str, list[TypeComponent]]:
    """``TYPE name ... END TYPE`` blocks of the given sources -> {TYPENAME: components}."""
    types: dict[str, list[TypeComponent]] = {}
    for text in texts:
        stmts = scan_statements(text.splitlines(), free=free, line_length=line_length)
        cur: str | None = None
        decls = _Decls()
        order: list[str] = []
        for st in stmts:
            k = st.key
            if cur is None:
                if k.startswith("TYPE") and not k.startswith("TYPE("):
                    rest = k[4:]
                    if "::" in rest:
                        rest = rest.split("::", 1)[1]
                    m = _NAME_RE.match(rest)
                    if m and not rest.startswith("IS"):
                        cur = m.group(0)
                        decls, order = _Decls(), []
                continue
            if k.startswith("ENDTYPE"):
                types[cur] = [
                    TypeComponent(
                        n,
                        decls.table[n].derived,
                        decls.table[n].dims,
                        decls.table[n].allocatable,
                        decls.table[n].pointer,
                    )
                    for n in order
                ]
                cur = None
                continue
            if k in ("SEQUENCE",) or k.startswith(("PRIVATE", "CONTAINS", "PROCEDURE")):
                continue
            before = set(decls.table)
            decls.add_stmt(k)
            order.extend(n for n in decls.table if n not in before and n not in order)
    return types


# ---------------------------------------------------------------------------
# Routine structure
# ---------------------------------------------------------------------------
@dataclass
class UnitInfo:
    name: str
    kind: str  # 'SUBROUTINE' | 'FUNCTION'
    header: Stmt
    result: str | None
    dummies: list[str]
    first_exec: Stmt | None
    end: Stmt  # END statement or CONTAINS
    end_is_contains: bool
    returns: list[Stmt]
    decls: dict[str, Decl]
    labels: set[int]
    stmts: list[Stmt] = field(repr=False, default_factory=list)


def find_unit(stmts: Sequence[Stmt], name: str, *, line_hint: int | None = None) -> UnitInfo:
    """Locate program unit ``name`` and classify its statements.

    ``line_hint`` (1-based header line, e.g. ``line_start`` of the index) picks the right unit
    when the name also appears in an interface body.
    """
    name = name.upper()
    cands: list[int] = []
    depth_iface = 0
    for i, st in enumerate(stmts):
        k = st.key
        if k.startswith(("INTERFACE", "ABSTRACTINTERFACE")):
            depth_iface += 1
            continue
        if k.startswith("ENDINTERFACE"):
            depth_iface -= 1
            continue
        if depth_iface:
            continue
        h = _header_name(k)
        if h and h[1] == name:
            cands.append(i)
    if not cands:
        raise InstrumentError(f"program unit {name} not found")
    idx = cands[0]
    if line_hint is not None:
        exact = [i for i in cands if stmts[i].first_line + 1 == line_hint]
        if exact:
            idx = exact[0]
    header = stmts[idx]
    hk = header.key
    kind, _ = _header_name(hk)  # type: ignore[misc]
    m = re.search(rf"{kind}{name}", hk)
    assert m is not None
    rest = hk[m.end() :]
    dummies: list[str] = []
    if rest.startswith("("):
        j = _match_paren(rest, 0)
        dummies = [d for d in _split_top(rest[1 : j - 1]) if d]
        rest = rest[j:]
    result: str | None = None
    if kind == "FUNCTION":
        mr = re.search(r"RESULT\((\w+)\)", rest)
        result = mr.group(1) if mr else name

    decls = _Decls()
    first_exec: Stmt | None = None
    end: Stmt | None = None
    end_is_contains = False
    returns: list[Stmt] = []
    labels: set[int] = set()
    body: list[Stmt] = []
    in_iface = 0
    in_type = False
    known_arrays: set[str] = set()
    dummy_set = set(dummies)
    for st in stmts[idx + 1 :]:
        k = st.key
        body.append(st)
        if st.label and st.label.isdigit():
            labels.add(int(st.label))
        if in_iface:
            if k.startswith(("INTERFACE", "ABSTRACTINTERFACE")):
                in_iface += 1
            elif k.startswith("ENDINTERFACE"):
                in_iface -= 1
            continue
        if in_type:
            if k.startswith("ENDTYPE"):
                in_type = False
            continue
        if k == "CONTAINS":
            end, end_is_contains = st, True
            break
        if _is_end(k):
            end = st
            break
        if first_exec is None:
            kind_s = _classify(k, known_arrays, dummy_set)
            if kind_s == "iface":
                in_iface = 1
                continue
            if kind_s == "typedef":
                in_type = True
                continue
            if kind_s == "spec":
                decls.add_stmt(k)
                known_arrays = {n for n, d in decls.table.items() if d.dims}
                continue
            if kind_s in ("neutral", "stmtfn"):
                continue
            first_exec = st
        if k == "RETURN" or _is_if_return(k):
            returns.append(st)
        elif re.match(r"RETURN[0-9(]", k) or (k.startswith("IF(") and re.search(r"\)RETURN\w", k)):
            raise InstrumentError(f"{name}: alternate RETURN at line {st.first_line + 1} is not supported")
        if k.startswith("ENTRY"):
            raise InstrumentError(f"{name}: ENTRY statements are not supported")
    if end is None:
        raise InstrumentError(f"{name}: no END statement found")
    return UnitInfo(
        name=name,
        kind=kind,
        header=header,
        result=result,
        dummies=dummies,
        first_exec=first_exec,
        end=end,
        end_is_contains=end_is_contains,
        returns=returns,
        decls=decls.table,
        labels=labels,
        stmts=body,
    )


def _is_if_return(key: str) -> bool:
    if not key.startswith("IF("):
        return False
    j = _match_paren(key, 2)
    return key[j:] == "RETURN"


def _classify(key: str, arrays: set[str], dummies: set[str]) -> str:
    if len(_split_top(key, "::")) > 1:
        if key.startswith("TYPE") and not key.startswith("TYPE("):
            return "typedef"
        return "spec"
    a = _assignment(key)
    if a is not None:
        name, group = a
        if (
            group is not None
            and "%" not in name
            and name not in arrays
            and name not in dummies
            and (group == "" or re.fullmatch(r"[A-Z_][A-Z0-9_]*(,[A-Z_][A-Z0-9_]*)*", group))
        ):
            return "stmtfn"
        return "exec"
    if key.startswith(("INTERFACE", "ABSTRACTINTERFACE")):
        return "iface"
    if key.startswith("TYPE") and not key.startswith("TYPE("):
        return "typedef"
    if key.startswith(_NEUTRAL_KW):
        return "neutral"
    if key.startswith(_SPEC_KW):
        return "spec"
    return "exec"


# ---------------------------------------------------------------------------
# Variable plan
# ---------------------------------------------------------------------------
@dataclass
class VarSpec:
    """One dumped quantity: record name, Fortran expression, provenance."""

    name: str
    expr: str
    kind: str  # 'arg' | 'save' | 'save_all' | 'data_init' | 'implicit_save' | 'common' | 'result'
    intent: str | None = None
    block: str | None = None
    exit_only: bool = False


@dataclass
class RoutineRequest:
    """What to instrument in one routine."""

    name: str
    date_expr: str | None = None
    extents: dict[str, str] = field(default_factory=dict)
    """Upper bound of the last dimension for assumed-size dummies, e.g. ``{"X": "NN"}``."""
    skip: list[str] = field(default_factory=list)
    include_common: bool = True
    include_saved: bool = True


@dataclass
class DateHook:
    """Insert ``CALL AJD_SETDATE(expr)`` at the entry of ``routine`` (in ``file``)."""

    routine: str
    expr: str
    file: str | None = None


def plan_variables(
    entry: Mapping[str, Any],
    unit: UnitInfo,
    req: RoutineRequest,
    types: Mapping[str, list[TypeComponent]] | None = None,
) -> tuple[list[VarSpec], list[str]]:
    """Variables to dump for one routine from its index record; returns (specs, notes)."""
    types = types or {}
    notes: list[str] = []
    skip = {s.upper() for s in req.skip}
    extents = {k.upper(): v for k, v in req.extents.items()}
    intents = {k.upper(): v for k, v in entry.get("intent_guess", {}).items()}
    intents.update({k.upper(): v for k, v in entry.get("intent_declared", {}).items()})
    raw: list[VarSpec] = []
    for a in entry.get("args", []):
        a = a.upper()
        if a != "*":
            raw.append(VarSpec(a, a, "arg", intents.get(a)))
    if req.include_saved:
        for s in entry.get("saved_vars", []):
            raw.append(VarSpec(s["name"].upper(), s["name"].upper(), s.get("kind", "save")))
    if req.include_common:
        used = {v.upper() for v in entry.get("read_vars", [])} | {
            v.upper() for v in entry.get("assigned_vars", [])
        }
        for blk in entry.get("common_blocks", []):
            for v in blk.get("vars", []):
                v = v.upper()
                if v in used:
                    raw.append(VarSpec(v, v, "common", block=blk.get("name")))
    if unit.result:
        raw.append(VarSpec(unit.result, unit.result, "result", exit_only=True))
    seen: set[str] = set()
    specs: list[VarSpec] = []
    for v in raw:
        if v.name in seen or v.name in skip:
            continue
        seen.add(v.name)
        d = unit.decls.get(v.name)
        if d is not None and d.external:
            notes.append(f"{v.name}: dummy procedure, not dumped")
            continue
        if d is not None and d.parameter:
            continue
        if d is not None and (d.allocatable or d.pointer):
            notes.append(f"{v.name}: allocatable/pointer, not dumped")
            continue
        if d is not None and d.assumed_size:
            if v.name not in extents:
                notes.append(f"{v.name}: assumed-size array without an extent, not dumped")
                continue
            dims = d.dims or []
            last = dims[-1]
            lb = last.split(":")[0] if ":" in last else "1"
            sect = [":"] * (len(dims) - 1) + [f"{lb}:{extents[v.name]}"]
            v.expr = f"{v.name}({','.join(sect)})"
        if d is not None and d.derived:
            specs.extend(_expand_derived(v, d.derived, types, notes, depth=0))
            continue
        specs.append(v)
    return specs, notes


def _expand_derived(
    v: VarSpec, tname: str, types: Mapping[str, list[TypeComponent]], notes: list[str], depth: int
) -> list[VarSpec]:
    comps = types.get(tname.upper())
    if comps is None:
        notes.append(f"{v.name}: derived type {tname} has no known definition, not dumped")
        return []
    if depth > 4:
        notes.append(f"{v.name}: derived type nesting too deep, not dumped")
        return []
    out: list[VarSpec] = []
    for c in comps:
        nm = f"{v.name}%{c.name}"
        ex = f"{v.expr}%{c.name}"
        if c.allocatable or c.pointer:
            notes.append(f"{nm}: allocatable/pointer component, not dumped")
            continue
        sub = VarSpec(nm, ex, v.kind, v.intent, v.block, v.exit_only)
        if c.derived:
            out.extend(_expand_derived(sub, c.derived, types, notes, depth + 1))
        else:
            out.append(sub)
    return out


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------
def _emit(code: str, *, free: bool, indent: int = 0, label: str | None = None, width: int = 72) -> list[str]:
    """Physical lines for one generated statement (continuation-wrapped)."""
    if free:
        pad = " " * (6 + indent)
        first = (f"{label} " if label else "") + pad
        limit = 100
        chunks: list[str] = []
        s = code
        while len(s) > limit:
            chunks.append(s[:limit])
            s = s[limit:]
        chunks.append(s)
        if len(chunks) == 1:
            return [first + chunks[0] + "\n"]
        out = [first + chunks[0] + "&\n"]
        for c in chunks[1:-1]:
            out.append(pad + "&" + c + "&\n")
        out.append(pad + "&" + chunks[-1] + "\n")
        return out
    lab = (label or "").ljust(5)[:5]
    body = " " * indent + code
    room = width - 6
    out = [lab + " " + body[:room] + "\n"]
    s = body[room:]
    while s:
        out.append("     &" + s[:room] + "\n")
        s = s[room:]
    return out


def _put_lines(specs: Sequence[VarSpec], *, exit_: bool, free: bool) -> list[str]:
    lines: list[str] = []
    for v in specs:
        if v.exit_only and not exit_:
            continue
        lines += _emit(f"CALL AJD_PUT('{v.name}',{v.expr})", free=free, indent=2)
    return lines


def _fresh_label(used: set[int]) -> int:
    lab = EXIT_LABEL
    while lab in used:
        lab += 1
        if lab > 99999:
            lab = 90000
    return lab


def _replace_return(lines: list[str], st: Stmt, label: int, *, free: bool, width: int | None) -> None:
    """Rewrite the trailing RETURN keyword of ``st`` into ``GOTO label`` in place."""
    idx = st.text.upper().rstrip().rfind("RETURN")
    if idx < 0:
        raise InstrumentError(f"RETURN not found in statement at line {st.first_line + 1}")
    (l0, c0), (l1, c1) = st.pos[idx], st.pos[idx + 5]
    if l0 != l1 or c1 != c0 + 5:
        raise InstrumentError(f"RETURN keyword split across lines at line {l0 + 1}; not supported")
    raw = lines[l0]
    nl = "\n" if raw.endswith("\n") else ""
    text = raw.rstrip("\r\n")
    if not free and width is not None:
        text = text[:width]
    new = text[:c0] + f"GOTO {label}" + text[c0 + 6 :]
    if free or width is None or len(new.rstrip()) <= width:
        lines[l0] = new + nl
        return
    # too long for the fixed-form field: continue the statement on a new line
    head = text[:c0].rstrip()
    tail = text[c0 + 6 :]
    lines[l0] = head + nl + "     &GOTO " + str(label) + tail + nl


def _date_line(expr: str, free: bool) -> list[str]:
    return _emit(f"CALL AJD_SETDATE(INT({expr}))", free=free)


@dataclass
class InstrumentResult:
    routine: str
    routine_id: int
    file: str
    variables: list[VarSpec]
    notes: list[str]
    exit_label: int
    n_returns: int


def instrument_routine(
    text: str,
    entry: Mapping[str, Any],
    routine_id: int,
    *,
    request: RoutineRequest | None = None,
    free: bool = False,
    line_length: int | None = 72,
    types: Mapping[str, list[TypeComponent]] | None = None,
    file: str = "",
) -> tuple[str, InstrumentResult]:
    """Instrument one routine inside ``text`` (the whole source file); returns (new text, result).

    ``entry`` is the index record of the routine (``args``, ``saved_vars``, ``common_blocks``,
    ``read_vars``, ``assigned_vars``, ``intent_guess``, ``line_start``). It can be a hand-written
    dict with just ``args`` for tests.
    """
    if not 1 <= routine_id <= MAX_ROUTINES:
        raise InstrumentError(f"routine id {routine_id} outside 1..{MAX_ROUTINES}")
    name = str(entry["name"]).upper()
    req = request or RoutineRequest(name)
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    stmts = scan_statements(lines, free=free, line_length=line_length)
    unit = find_unit(stmts, name, line_hint=entry.get("line_start"))
    specs, notes = plan_variables(entry, unit, req, types)
    label = _fresh_label(unit.labels)
    width = None if free else line_length

    # 1) RETURN -> GOTO (in place, positions unaffected for other lines except splits: go bottom-up)
    inserts: dict[int, list[str]] = {}  # insert before physical line
    for st in sorted(unit.returns, key=lambda s: s.first_line, reverse=True):
        _replace_return(lines, st, label, free=free, width=width)

    # 2) exit block before END/CONTAINS
    end = unit.end
    exit_block: list[str] = []
    if end.label and not unit.end_is_contains:
        exit_block += _emit("CONTINUE", free=free, label=end.label)
        assert end.label_pos is not None
        ln, a, b = end.label_pos
        lines[ln] = lines[ln][:a] + " " * (b - a) + lines[ln][b:]
    exit_block += _emit("CONTINUE", free=free, label=str(label))
    exit_block += _emit(f"IF (AJD_EXIT({routine_id})) THEN", free=free)
    exit_block += _put_lines(specs, exit_=True, free=free)
    exit_block += _emit("CALL AJD_END()", free=free, indent=2)
    exit_block += _emit("END IF", free=free)
    inserts.setdefault(end.first_line, []).extend(exit_block)

    # 3) entry block before the first executable statement (or before the exit block)
    entry_block: list[str] = []
    if req.date_expr:
        entry_block += _date_line(req.date_expr, free)
    entry_block += _emit(f"IF (AJD_BEGIN({routine_id},'{name}')) THEN", free=free)
    entry_block += _put_lines(specs, exit_=False, free=free)
    entry_block += _emit("CALL AJD_END()", free=free, indent=2)
    entry_block += _emit("END IF", free=free)
    at = unit.first_exec.first_line if unit.first_exec is not None else end.first_line
    inserts.setdefault(at, [])[0:0] = entry_block

    # 4) USE after the header
    use_at = unit.header.last_line + 1
    inserts.setdefault(use_at, [])[0:0] = _emit("USE AJDUMP", free=free)

    out: list[str] = []
    for i, ln_text in enumerate(lines):
        if i in inserts:
            out.extend(inserts[i])
        out.append(ln_text)
    if len(lines) in inserts:
        out.extend(inserts[len(lines)])
    res = InstrumentResult(
        routine=name,
        routine_id=routine_id,
        file=file,
        variables=specs,
        notes=notes,
        exit_label=label,
        n_returns=len(unit.returns),
    )
    return "".join(out), res


def insert_date_hook(
    text: str,
    routine: str,
    expr: str,
    *,
    free: bool = False,
    line_length: int | None = 72,
    line_hint: int | None = None,
) -> str:
    """Insert ``USE AJDUMP`` and ``CALL AJD_SETDATE(INT(expr))`` at the entry of ``routine``."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    stmts = scan_statements(lines, free=free, line_length=line_length)
    unit = find_unit(stmts, routine, line_hint=line_hint)
    at = unit.first_exec.first_line if unit.first_exec is not None else unit.end.first_line
    inserts = {unit.header.last_line + 1: _emit("USE AJDUMP", free=free)}
    inserts.setdefault(at, []).extend(_date_line(expr, free))
    out: list[str] = []
    for i, t in enumerate(lines):
        out.extend(inserts.get(i, []))
        out.append(t)
    return "".join(out)


# ---------------------------------------------------------------------------
# Fortran runtime module
# ---------------------------------------------------------------------------
_PUT_TYPES: list[tuple[str, str, int, str, str]] = [
    # tag, declaration, type code, itemsize expression, write expression
    ("r8", "real(real64)", 1, "8", "x"),
    ("r4", "real(real32)", 2, "4", "x"),
    ("i4", "integer(int32)", 3, "4", "x"),
    ("i8", "integer(int64)", 4, "8", "x"),
    ("i2", "integer(int16)", 5, "2", "x"),
    ("i1", "integer(int8)", 8, "1", "x"),
    ("l4", "logical(4)", 6, "4", "merge(1_int32, 0_int32, x)"),
    ("l8", "logical(8)", 6, "4", "merge(1_int32, 0_int32, x)"),
    ("l2", "logical(2)", 6, "4", "merge(1_int32, 0_int32, x)"),
    ("l1", "logical(1)", 6, "4", "merge(1_int32, 0_int32, x)"),
    ("ch", "character(len=*)", 7, "len(x)", "x"),
    ("c8", "complex(real64)", 9, "16", "x"),
    ("c4", "complex(real32)", 10, "8", "x"),
]
_MAX_RANK = 7


def ajdump_module_source() -> str:
    """Free-form source of module ``AJDUMP`` (compile it before the instrumented files)."""
    specifics: list[str] = []
    bodies: list[str] = []
    for tag, decl, code, isz, wexpr in _PUT_TYPES:
        for r in range(_MAX_RANK + 1):
            nm = f"ajd_put_{tag}_{r}"
            specifics.append(nm)
            dims = "" if r == 0 else "(" + ",".join([":"] * r) + ")"
            shp = "[integer ::]" if r == 0 else "shape(x)"
            bodies.append(
                f"""  subroutine {nm}(name, x)
    character(len=*), intent(in) :: name
    {decl}, intent(in) :: x{dims}
    if (.not. writing) return
    call vhead(name, {code}, {r}, int({isz}, int32), {shp})
    write(curu) {wexpr}
  end subroutine {nm}
"""
            )
    iface = "\n".join(f"    module procedure {s}" for s in specifics)
    return f"""! Generated by agrijax.port.instrument (dump format version {DUMP_VERSION}). Do not edit.
module ajdump
  use, intrinsic :: iso_fortran_env, only: real32, real64, int8, int16, int32, int64
  implicit none
  private
  public :: ajd_begin, ajd_exit, ajd_end, ajd_put, ajd_setdate

  integer, parameter :: maxr = {MAX_ROUTINES}, maxcfg = 256
  logical, save :: inited = .false.
  integer(int32), save :: cur_date = 0
  integer, save :: ndates = -1
  integer(int32), allocatable, save :: dates(:)
  logical, save :: day_ok = .true.
  character(len=32), save :: rname(maxr) = ' '
  integer(int64), save :: ncall(maxr) = 0, ndumped(maxr) = 0, act_call(maxr) = 0
  integer(int64), save :: nday(maxr) = 0, dumped_day(maxr) = 0
  integer(int32), save :: last_date(maxr) = -huge(1_int32), act_date(maxr) = 0, act_dcall(maxr) = 0
  integer(int64), save :: perday(maxr) = 0, every(maxr) = 1, maxtot(maxr) = 0
  logical, save :: active(maxr) = .false.
  integer, save :: unit(maxr) = 0
  logical, save :: opened(maxr) = .false.
  integer, save :: curu = 0
  logical, save :: writing = .false.
  character(len=1024), save :: outdir = '.'
  integer(int64), save :: def_perday = huge(1_int64), def_every = 1, def_maxtot = 100
  integer, save :: ncfg = 0
  character(len=32), save :: cfg_name(maxcfg) = ' '
  integer(int64), save :: cfg_perday(maxcfg) = 0, cfg_every(maxcfg) = 1, cfg_maxtot(maxcfg) = 0

  interface ajd_put
{iface}
  end interface ajd_put

contains

  subroutine upcase(s)
    character(len=*), intent(inout) :: s
    integer :: i, c
    do i = 1, len(s)
      c = iachar(s(i:i))
      if (c >= 97 .and. c <= 122) s(i:i) = achar(c - 32)
    end do
  end subroutine upcase

  subroutine init()
    character(len=1024) :: fname, line, kw
    character(len=32) :: nm
    integer :: u, ios, n, i
    integer(int64) :: a, b, c
    logical :: ex
    inited = .true.
    fname = 'ajdump.cfg'
    call get_environment_variable('AJDUMP_CFG', line, status=ios)
    if (ios == 0 .and. len_trim(line) > 0) fname = line
    inquire(file=trim(fname), exist=ex)
    if (.not. ex) return
    open(newunit=u, file=trim(fname), status='old', action='read', iostat=ios)
    if (ios /= 0) return
    do
      read(u, '(A)', iostat=ios) line
      if (ios /= 0) exit
      line = adjustl(line)
      if (len_trim(line) == 0 .or. line(1:1) == '#') cycle
      kw = line
      i = index(kw, ' ')
      if (i > 0) kw = kw(1:i-1)
      call upcase(kw)
      select case (trim(kw))
      case ('DIR')
        outdir = adjustl(line(4:))
      case ('DEFAULT')
        read(line(8:), *, iostat=ios) a, b, c
        if (ios == 0) then
          def_perday = a; def_every = max(1_int64, b); def_maxtot = c
        end if
      case ('ROUTINE')
        read(line(8:), *, iostat=ios) nm, a, b, c
        if (ios == 0 .and. ncfg < maxcfg) then
          call upcase(nm)
          ncfg = ncfg + 1
          cfg_name(ncfg) = nm; cfg_perday(ncfg) = a; cfg_every(ncfg) = max(1_int64, b)
          cfg_maxtot(ncfg) = c
        end if
      case ('DATES')
        read(line(6:), *, iostat=ios) n
        if (ios == 0 .and. n >= 0) then
          if (allocated(dates)) deallocate(dates)
          allocate(dates(n))
          if (n > 0) read(u, *, iostat=ios) dates
          if (ios /= 0) then
            ndates = 0
          else
            ndates = n
          end if
        end if
      end select
    end do
    close(u)
    call select_day()
  end subroutine init

  subroutine select_day()
    if (ndates < 0) then
      day_ok = .true.
    else
      day_ok = any(dates == cur_date)
    end if
  end subroutine select_day

  subroutine ajd_setdate(d)
    integer, intent(in) :: d
    if (.not. inited) call init()
    if (int(d, int32) /= cur_date) then
      cur_date = int(d, int32)
      call select_day()
    end if
  end subroutine ajd_setdate

  subroutine register(id, name)
    integer, intent(in) :: id
    character(len=*), intent(in) :: name
    integer :: i, ios
    character(len=32) :: nm
    nm = name
    call upcase(nm)
    rname(id) = nm
    perday(id) = def_perday; every(id) = def_every; maxtot(id) = def_maxtot
    do i = 1, ncfg
      if (cfg_name(i) == nm) then
        perday(id) = cfg_perday(i); every(id) = cfg_every(i); maxtot(id) = cfg_maxtot(i)
      end if
    end do
    open(newunit=unit(id), file=trim(outdir)//'/ajdump_'//trim(nm)//'.bin', access='stream', &
         form='unformatted', status='replace', action='write', iostat=ios)
    opened(id) = ios == 0
  end subroutine register

  logical function ajd_begin(id, name)
    integer, intent(in) :: id
    character(len=*), intent(in) :: name
    ajd_begin = .false.
    if (.not. inited) call init()
    if (id < 1 .or. id > maxr) return
    if (rname(id) == ' ') call register(id, name)
    ncall(id) = ncall(id) + 1
    active(id) = .false.
    if (last_date(id) /= cur_date) then
      last_date(id) = cur_date; nday(id) = 0; dumped_day(id) = 0
    end if
    nday(id) = nday(id) + 1
    if (.not. opened(id) .or. .not. day_ok) return
    if (ndumped(id) >= maxtot(id)) return
    if (mod(nday(id) - 1, every(id)) /= 0) return
    if (dumped_day(id) >= perday(id)) return
    dumped_day(id) = dumped_day(id) + 1
    ndumped(id) = ndumped(id) + 1
    active(id) = .true.
    act_call(id) = ncall(id); act_date(id) = cur_date; act_dcall(id) = int(nday(id), int32)
    call rhead(id, 0)
    ajd_begin = .true.
  end function ajd_begin

  logical function ajd_exit(id)
    integer, intent(in) :: id
    ajd_exit = .false.
    if (id < 1 .or. id > maxr) return
    if (.not. active(id)) return
    active(id) = .false.
    call rhead(id, 1)
    ajd_exit = .true.
  end function ajd_exit

  subroutine rhead(id, phase)
    integer, intent(in) :: id, phase
    curu = unit(id)
    writing = .true.
    write(curu) 'AJDR', int({DUMP_VERSION}, int32), rname(id), int(phase, int32), act_call(id), &
                act_date(id), act_dcall(id)
  end subroutine rhead

  subroutine ajd_end()
    if (.not. writing) return
    write(curu) 'AJDE'
    flush(curu)
    writing = .false.
  end subroutine ajd_end

  subroutine vhead(name, code, rank, itemsize, shp)
    character(len=*), intent(in) :: name
    integer, intent(in) :: code, rank
    integer(int32), intent(in) :: itemsize
    integer, intent(in) :: shp(:)
    character(len=64) :: nm
    nm = name
    write(curu) 'AJDV', nm, int(code, int32), int(rank, int32), itemsize, int(shp, int32)
  end subroutine vhead

{"".join(bodies)}
end module ajdump
"""


# ---------------------------------------------------------------------------
# Run-time configuration and sampling
# ---------------------------------------------------------------------------
def write_config(
    path: str | Path,
    *,
    dates: Sequence[int] | None = None,
    routines: Mapping[str, tuple[int, int, int]] | None = None,
    default: tuple[int, int, int] | None = None,
    out_dir: str | None = None,
) -> Path:
    """Write ``ajdump.cfg``.

    ``dates``: simulation dates (as produced by the date expression, e.g. ``YYYYDDD``) on which
    calls are dumped; ``None`` = every date. ``routines``: ``{NAME: (per_day, every, max_total)}``
    -- at most ``per_day`` dumps per date, only every ``every``-th call of the date (counting from
    the first), at most ``max_total`` in the run. ``default`` applies to unlisted routines.
    """
    lines = ["# ajdump configuration (agrijax.port.instrument)"]
    if out_dir:
        lines.append(f"DIR {out_dir}")
    if default is not None:
        lines.append("DEFAULT {} {} {}".format(*default))
    for nm, (pd, ev, mx) in (routines or {}).items():
        lines.append(f"ROUTINE {nm.upper()} {pd} {ev} {mx}")
    if dates is not None:
        ds = sorted({int(d) for d in dates})
        lines.append(f"DATES {len(ds)}")
        for i in range(0, len(ds), 8):
            lines.append(" ".join(str(d) for d in ds[i : i + 8]))
    p = Path(path)
    p.write_text("\n".join(lines) + "\n")
    return p


def select_dates(
    dates: Sequence[int],
    rain: Sequence[float],
    *,
    season: Sequence[bool] | None = None,
    n_rain: int = 12,
    n_dry: int = 12,
    n_season: int = 12,
    rain_threshold: float = 5.0,
    dry_days: int = 5,
) -> dict[str, list[int]]:
    """Deterministic spread of sample dates: rainy days, dry spells and growing-season days.

    * ``rain``: days with ``rain >= rain_threshold``.
    * ``dry``: days preceded by at least ``dry_days`` rain-free days (rain < 0.1).
    * ``season``: days with ``season`` true that are in neither of the other groups.

    Each group is thinned by taking evenly spaced elements, so the result does not depend on any
    random state. Returns ``{"rain": [...], "dry": [...], "season": [...]}``.
    """
    if len(dates) != len(rain):
        raise ValueError("dates and rain differ in length")
    ds = [int(d) for d in dates]
    wet = [d for d, r in zip(ds, rain, strict=True) if r >= rain_threshold]
    dry: list[int] = []
    run = 0
    for d, r in zip(ds, rain, strict=True):
        if run >= dry_days and r < 0.1:
            dry.append(d)
        run = run + 1 if r < 0.1 else 0
    taken = set(_spread(wet, n_rain)) | set(_spread(dry, n_dry))
    seas: list[int] = []
    if season is not None:
        if len(season) != len(ds):
            raise ValueError("season and dates differ in length")
        seas = [d for d, s in zip(ds, season, strict=True) if s and d not in taken]
    return {"rain": _spread(wet, n_rain), "dry": _spread(dry, n_dry), "season": _spread(seas, n_season)}


def _spread(xs: Sequence[int], n: int) -> list[int]:
    if n <= 0 or not xs:
        return []
    if len(xs) <= n:
        return list(xs)
    step = len(xs) / n
    return [xs[int(i * step + step / 2)] for i in range(n)]


# ---------------------------------------------------------------------------
# Tree driver
# ---------------------------------------------------------------------------
def _index_entries(index: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    subs = index.get("subroutines", index)
    items = subs if isinstance(subs, list) else list(subs.values())
    return {str(e["name"]).upper(): e for e in items}


def _rel_to(path: str, src_root: Path, orig_roots: Sequence[Path]) -> Path:
    p = Path(path)
    for r in [src_root, *orig_roots]:
        try:
            return p.relative_to(r)
        except ValueError:
            continue
    raise InstrumentError(f"{path} is not under {src_root}")


def instrument_tree(
    index: Mapping[str, Any] | str | Path,
    src_root: str | Path,
    out_root: str | Path,
    requests: Sequence[RoutineRequest],
    *,
    hooks: Sequence[DateHook] = (),
    index_root: str | Path | None = None,
    copy_tree: bool = False,
    line_length: int | None = 72,
    type_sources: Sequence[str | Path] = (),
    module_path: str | Path | None = None,
) -> dict[str, Any]:
    """Instrument several routines of a source tree; returns the manifest (also written to disk).

    Files are read from ``src_root`` and written to ``out_root`` under the same relative path
    (with ``copy_tree`` the whole tree is copied first, giving a buildable tree). The index's
    absolute file paths are resolved relative to ``index_root`` (default: the common root of the
    index's ``files``). Unified diffs of every changed file go to ``out_root/_patches/`` and the
    manifest to ``out_root/_patches/manifest.json``. The ``AJDUMP`` module is written to
    ``module_path`` (relative to ``out_root``; default ``ajdump.f90``).
    """
    idx: Mapping[str, Any] = index if isinstance(index, Mapping) else json.loads(Path(index).read_text())
    entries = _index_entries(idx)
    src_root, out_root = Path(src_root).resolve(), Path(out_root)
    files = [Path(f) for f in idx.get("files", [])]
    roots: list[Path] = []
    if index_root is not None:
        roots.append(Path(index_root).resolve())
    elif files:
        roots.append(Path(os.path.commonpath([str(f.parent) for f in files])))
    if copy_tree:
        if out_root.exists():
            shutil.rmtree(out_root)
        shutil.copytree(src_root, out_root, symlinks=True)
    out_root.mkdir(parents=True, exist_ok=True)
    types: dict[str, list[TypeComponent]] = {}
    for ts in type_sources:
        tp = Path(ts)
        types.update(
            parse_type_definitions(
                [tp.read_text(encoding="latin-1")],
                free=tp.suffix.lower() in FREE_SUFFIXES,
                line_length=None if tp.suffix.lower() in FREE_SUFFIXES else line_length,
            )
        )

    texts: dict[Path, str] = {}
    originals: dict[Path, str] = {}

    def load(rel: Path) -> str:
        if rel not in texts:
            originals[rel] = (src_root / rel).read_text(encoding="latin-1")
            texts[rel] = originals[rel]
        return texts[rel]

    results: list[InstrumentResult] = []
    jobs: list[tuple[int, RoutineRequest, dict[str, Any], Path]] = []
    for rid, req in enumerate(requests, start=1):
        e = entries.get(req.name.upper())
        if e is None:
            raise InstrumentError(f"{req.name} is not in the index")
        jobs.append((rid, req, e, _rel_to(e["file"], src_root, roots)))
    # bottom-up within a file, so the index line numbers of the remaining routines stay valid
    jobs.sort(key=lambda j: (str(j[3]), -int(j[2].get("line_start") or 0)))
    for rid, req, e, rel in jobs:
        free = rel.suffix.lower() in FREE_SUFFIXES
        new, res = instrument_routine(
            load(rel),
            e,
            rid,
            request=req,
            free=free,
            line_length=None if free else line_length,
            types=types,
            file=str(rel),
        )
        texts[rel] = new
        results.append(res)
    results.sort(key=lambda r: r.routine_id)
    for h in hooks:
        if h.file is not None:
            rel = Path(h.file)
        else:
            e = entries.get(h.routine.upper())
            if e is None:
                raise InstrumentError(f"hook routine {h.routine} is not in the index; give its file")
            rel = _rel_to(e["file"], src_root, roots)
        free = rel.suffix.lower() in FREE_SUFFIXES
        texts[rel] = insert_date_hook(
            load(rel), h.routine, h.expr, free=free, line_length=None if free else line_length
        )

    pdir = out_root / "_patches"
    pdir.mkdir(parents=True, exist_ok=True)
    for rel, new in texts.items():
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(new, encoding="latin-1")
        diff = difflib.unified_diff(
            originals[rel].splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
        (pdir / (str(rel).replace("/", "__") + ".patch")).write_text("".join(diff), encoding="latin-1")
    mod = out_root / (module_path or "ajdump.f90")
    mod.parent.mkdir(parents=True, exist_ok=True)
    mod.write_text(ajdump_module_source())
    manifest = {
        "version": DUMP_VERSION,
        "src_root": str(src_root),
        "module": str(mod),
        "routines": [
            {
                "id": r.routine_id,
                "name": r.routine,
                "file": r.file,
                "exit_label": r.exit_label,
                "n_returns": r.n_returns,
                "notes": r.notes,
                "variables": [asdict(v) for v in r.variables],
            }
            for r in results
        ],
        "hooks": [asdict(h) for h in hooks],
        "files": sorted(str(r) for r in texts),
    }
    (pdir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_routine(s: str) -> RoutineRequest:
    # NAME[@date_expr][,EXT=VAR=expr...]  e.g. RICHRD  or  MZ_PHENOL@YRDOY
    name, _, date = s.partition("@")
    return RoutineRequest(name=name.upper(), date_expr=date or None)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agrijax.port.instrument", description=(__doc__ or "").split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tree", help="instrument routines of a source tree")
    t.add_argument("--index", required=True, help="index JSON from agrijax.port.fortran_index")
    t.add_argument("--src", required=True, help="source root (read-only)")
    t.add_argument("--out", required=True, help="output root (instrumented files, patches, manifest)")
    t.add_argument("--index-root", help="root the index's absolute paths are relative to")
    t.add_argument("--routine", action="append", default=[], help="NAME or NAME@date_expr (repeatable)")
    t.add_argument("--hook", action="append", default=[], help="ROUTINE=expr[@file] date hook (repeatable)")
    t.add_argument("--extent", action="append", default=[], help="ROUTINE:VAR=expr for assumed-size dummies")
    t.add_argument("--copy-tree", action="store_true", help="copy the whole source tree first")
    t.add_argument("--line-length", default="72", help="fixed-form line length, or 'none'")
    t.add_argument("--type-source", action="append", default=[], help="file with derived-type definitions")
    t.add_argument("--module-path", default=None, help="where to write ajdump.f90 (relative to --out)")
    m = sub.add_parser("module", help="write the AJDUMP Fortran module")
    m.add_argument("path")
    a = ap.parse_args(argv)
    if a.cmd == "module":
        Path(a.path).write_text(ajdump_module_source())
        return 0
    reqs = [_parse_routine(r) for r in a.routine]
    by = {r.name: r for r in reqs}
    for x in a.extent:
        rn, _, rest = x.partition(":")
        var, _, expr = rest.partition("=")
        by[rn.upper()].extents[var.upper()] = expr
    hooks: list[DateHook] = []
    for h in a.hook:
        rn, _, rest = h.partition("=")
        expr, _, file = rest.partition("@")
        hooks.append(DateHook(rn.upper(), expr, file or None))
    ll = None if str(a.line_length).lower() == "none" else int(a.line_length)
    man = instrument_tree(
        a.index,
        a.src,
        a.out,
        reqs,
        hooks=hooks,
        index_root=a.index_root,
        copy_tree=a.copy_tree,
        line_length=ll,
        type_sources=a.type_source,
        module_path=a.module_path,
    )
    for r in man["routines"]:
        nv = len(r["variables"])
        print(f"{r['name']:12s} id={r['id']} vars={nv} returns={r['n_returns']} file={r['file']}")
        for n in r["notes"]:
            print(f"    note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
