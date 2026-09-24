"""Independent line-based cross-check of :mod:`agrijax.port.fortran_index`.

``fortran_index`` builds its call graph and argument lists from an fparser syntax tree. This
module re-derives the same facts with a completely different method -- a regular-expression
scanner over *logical statements* -- and compares the two. Agreement of two independent
implementations is the evidence that the index is right; every disagreement is either a bug in
one of them or a documented difference in scope (see :func:`compare`).

Scanner model
-------------
Physical lines are turned into logical statements:

* fixed form: a ``C``/``c``/``*``/``!`` (or ``D``/``d`` debug line, which ifort and gfortran
  drop by default) in column 1 makes a comment line; columns 1-5 hold the label; a character
  other than blank or ``0`` in column 6 marks a continuation; the statement field is columns
  7..``line_length`` (72 for ifort defaults, ``None`` = unlimited as DSSAT's
  ``-ffixed-line-length-none``). DEC tab format is accepted: a TAB in columns 1-6 starts the
  statement field, and a digit 1-9 right after the TAB marks a continuation.
* free form: ``!`` comments, trailing ``&`` continues, a leading ``&`` on the continuation line
  is dropped.
* both: ``!`` outside a character literal starts a comment (quote state is carried across
  continuation lines), character literals are masked to ``'?'`` so their contents can never
  look like code, ``;`` separates statements, ``#`` preprocessor lines are skipped, and the text
  is upper-cased.

On the statements the scanner recognises program-unit headers (``SUBROUTINE``, ``FUNCTION``
with optional type prefix and ``RESULT``, ``PROGRAM``), ``END`` statements, ``MODULE`` /
``CONTAINS`` / ``INTERFACE`` / derived-``TYPE`` blocks (so interface bodies are not units),
``CALL`` statements (also as the action of a logical ``IF``), ``SAVE`` statements and ``SAVE``
attributes, ``COMMON`` statements, array declarators (``DIMENSION``, type declarations,
``COMMON`` objects) and statement-function definitions. Function references are found as
``NAME(`` for every scanned ``FUNCTION`` name that is not an array or statement function of the
referencing unit.

CLI::

    python -m agrijax.port.fortran_xcheck index.json [--line-length 72]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Disagreement",
    "ScannedUnit",
    "Statement",
    "check_topological_order",
    "compare",
    "index_edges",
    "logical_statements",
    "main",
    "scan_edges",
    "scan_file",
    "scan_files",
    "scan_text",
]

FREE_SUFFIXES = {".f90", ".f95", ".f03", ".f08"}


# ---------------------------------------------------------------------------
# Logical statements
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Statement:
    """One logical statement: 1-based line of its first physical line, masked upper-case text."""

    line: int
    text: str
    label: str = ""


def _strip_comment_and_mask(content: str, quote: str | None) -> tuple[str, str | None]:
    """Remove a ``!`` comment and mask literals; ``quote`` is the open quote carried in/out."""
    out: list[str] = []
    i = 0
    n = len(content)
    while i < n:
        ch = content[i]
        if quote is not None:
            if ch == quote:
                if i + 1 < n and content[i + 1] == quote:  # doubled quote inside the literal
                    i += 2
                    continue
                quote = None
                out.append("?'")
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append("'")
        elif ch == "!":
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out), quote


def _fixed_fields(raw: str, line_length: int | None) -> tuple[str, bool, str] | None:
    """(label, is_continuation, statement field) of a fixed-form line, or None for a comment."""
    if not raw.strip():
        return None
    if raw[0] in "Cc*!Dd":
        return None
    head = raw[:6]
    if "\t" in head:
        k = raw.index("\t")
        label, rest = raw[:k], raw[k + 1 :]
        if rest[:1] in tuple("123456789"):
            return label.strip(), True, rest[1:]
        return label.strip(), False, rest
    if head[:5].lstrip().startswith("!"):
        return None
    body = raw[6:] if line_length is None else raw[6:line_length]
    cont = len(raw) > 5 and raw[5] not in (" ", "0")
    return head[:5].strip(), cont, body


def logical_statements(text: str, *, free: bool = False, line_length: int | None = 72) -> list[Statement]:
    """Split Fortran source text into logical statements (see the module docstring)."""
    stmts: list[Statement] = []
    cur: list[str] = []
    cur_line = 0
    cur_label = ""
    quote: str | None = None
    pending_free = False  # free form: previous line ended with '&'

    def flush() -> None:
        nonlocal cur, quote
        if cur:
            joined = "".join(cur)
            for part in joined.split(";"):
                t = " ".join(part.split()).upper()
                if t:
                    stmts.append(Statement(cur_line, t, cur_label))
        cur = []
        quote = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        raw = raw.rstrip("\r\n")
        if raw.startswith("#"):
            continue
        if free:
            s = raw.strip()
            if not s or s.startswith("!"):
                continue
            if pending_free:
                if s.startswith("&"):
                    s = s[1:]
            else:
                flush()
                cur_line, cur_label = lineno, ""
                m = re.match(r"(\d{1,5})\s+", s)
                if m:
                    cur_label, s = m.group(1), s[m.end() :]
            body, quote = _strip_comment_and_mask(s, quote)
            b = body.rstrip()
            pending_free = b.endswith("&")
            if pending_free:
                b = b[:-1]
            cur.append(b + " " if quote is None else b)
            continue
        fields = _fixed_fields(raw, line_length)
        if fields is None:
            continue
        label, cont, body = fields
        rest = body.strip()
        if not cont and not label and (not rest or rest.startswith("!")):
            continue  # a line holding only a comment (any column but 6) is a comment line
        if not cont or not cur:
            flush()
            cur_line, cur_label = lineno, label
        masked, quote = _strip_comment_and_mask(body, quote if cont else None)
        cur.append(masked)
    flush()
    return stmts


# ---------------------------------------------------------------------------
# Statement recognisers
# ---------------------------------------------------------------------------
_TYPE = r"(?:INTEGER|REAL|DOUBLE\s*PRECISION|DOUBLE\s*COMPLEX|COMPLEX|LOGICAL|CHARACTER|TYPE\s*\(\s*\w+\s*\))"
_KIND = r"(?:\s*\*\s*(?:\d+|\(\s*\*\s*\))|\s*\([^()]*\))?"
_PREFIX = r"(?:(?:RECURSIVE|PURE|ELEMENTAL|IMPURE)\s+)*"
_HEADER = re.compile(
    rf"^{_PREFIX}(?:{_TYPE}{_KIND}\s*)?{_PREFIX}(SUBROUTINE|FUNCTION|PROGRAM)\s*([A-Z]\w*)\s*"
    r"(?:\(([^()]*)\))?\s*(?:RESULT\s*\(\s*([A-Z]\w*)\s*\))?\s*(?:BIND\s*\(.*\))?$"
)
_END = re.compile(r"^END(?:\s*(SUBROUTINE|FUNCTION|PROGRAM|MODULE|INTERFACE|TYPE|BLOCK\s*DATA)\b.*)?$")
_MODULE = re.compile(r"^MODULE\s+([A-Z]\w*)$")
_INTERFACE = re.compile(r"^(?:ABSTRACT\s+)?INTERFACE\b")
_TYPE_BLOCK = re.compile(r"^TYPE\s*(?:,[^:]*)?(?:::)?\s*([A-Z]\w*)$")
_BLOCK_DATA = re.compile(r"^BLOCK\s*DATA\b")
_CALL = re.compile(r"^CALL\s*([A-Z]\w*(?:\s*%\s*[A-Z]\w*)*)\s*(\(.*\))?$")
_SAVE = re.compile(r"^SAVE\b\s*(?:::)?\s*(.*)$")
_COMMON = re.compile(r"^COMMON\b\s*(.*)$")
_DIMENSION = re.compile(r"^DIMENSION\b\s*(?:::)?\s*(.*)$")
_TYPE_DECL = re.compile(rf"^{_TYPE}(?=[\s*(,:]){_KIND}\s*(,[^:]*)?(?:::)?\s*(.*)$")
_STMT_FUNC = re.compile(r"^([A-Z]\w*)\s*\(\s*[A-Z]\w*(?:\s*,\s*[A-Z]\w*)*\s*\)\s*=")
_INCLUDE = re.compile(r"^INCLUDE\s*'")
_ENTRY = re.compile(r"^ENTRY\s+([A-Z]\w*)")


def _split_top(s: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    out: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


def _strip_logical_if(t: str) -> str:
    """Return the action statement of ``IF (cond) action`` (repeatedly), else ``t``."""
    while True:
        m = re.match(r"^(?:ELSE\s*)?IF\s*\(", t)
        if not m:
            return t
        depth = 1
        i = m.end()
        while i < len(t) and depth:
            if t[i] == "(":
                depth += 1
            elif t[i] == ")":
                depth -= 1
            i += 1
        rest = t[i:].strip()
        if not rest or rest == "THEN" or re.match(r"^\d", rest):  # block IF or arithmetic IF
            return ""
        t = rest


def _declarator_names(items: str) -> list[tuple[str, bool]]:
    """(name, has_dims) of each entity in a declaration list."""
    out: list[tuple[str, bool]] = []
    for it in _split_top(items):
        it = it.split("=")[0].strip()
        m = re.match(r"([A-Z]\w*)\s*(\*\s*\d+\s*)?(\()?", it)
        if m:
            out.append((m.group(1), m.group(3) is not None))
    return out


@dataclass
class ScannedUnit:
    """What the scanner saw of one program unit."""

    name: str
    kind: str  # 'subroutine' | 'function' | 'program', with 'module_' / 'internal_' prefix
    file: str
    line_start: int
    args: list[str]
    calls: list[str] = field(default_factory=list)  # sorted, unique, as the index stores them
    save_names: list[str] = field(default_factory=list)
    save_all: bool = False
    save_attr_names: list[str] = field(default_factory=list)
    common_blocks: list[dict[str, Any]] = field(default_factory=list)
    arrays: set[str] = field(default_factory=set)
    stmt_functions: set[str] = field(default_factory=set)
    externals: set[str] = field(default_factory=set)
    includes: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    body: list[Statement] = field(default_factory=list, repr=False)

    @property
    def base_kind(self) -> str:
        return self.kind.split("_")[-1]


def _parse_common(rest: str) -> list[dict[str, Any]]:
    """``/A/ X, Y(3), /B/ Z`` -> [{'name': 'A', 'vars': [...]}, ...]; blank common has name ''."""
    parts = re.split(r"/\s*(\w*)\s*/", rest)
    blocks: list[dict[str, Any]] = []
    lead = parts[0].strip().strip(",")
    if lead:
        blocks.append({"name": "", "vars": [n for n, _ in _declarator_names(lead)]})
    for k in range(1, len(parts), 2):
        blocks.append(
            {"name": parts[k], "vars": [n for n, _ in _declarator_names(parts[k + 1].strip().strip(","))]}
        )
    return blocks


def scan_text(
    text: str, *, file: str = "<string>", free: bool = False, line_length: int | None = 72
) -> list[ScannedUnit]:
    """Scan one source text and return its program units in source order."""
    units: list[ScannedUnit] = []
    # scope stack entries: ('unit', ScannedUnit) | ('module', name) | ('interface', None) |
    # ('iface_proc', None) | ('type', None) | ('blockdata', None)
    stack: list[tuple[str, Any]] = []

    def cur_unit() -> ScannedUnit | None:
        for tag, obj in reversed(stack):
            if tag == "unit":
                return obj
            if tag in ("interface", "iface_proc", "type"):
                return None
        return None

    for st in logical_statements(text, free=free, line_length=line_length):
        t = st.text
        top = stack[-1][0] if stack else None
        if top == "type":
            if _END.match(t) and re.match(r"^END\s*TYPE\b", t):
                stack.pop()
            continue
        if top in ("interface", "iface_proc"):
            if top == "interface" and re.match(r"^END\s*INTERFACE\b", t):
                stack.pop()
            elif _HEADER.match(t):
                stack.append(("iface_proc", None))
            elif top == "iface_proc" and _END.match(t):
                stack.pop()
            continue
        m = _HEADER.match(t)
        if m and not t.startswith("MODULE PROCEDURE"):
            kind = m.group(1).lower()
            if stack and stack[-1][0] == "module":
                kind = f"module_{kind}"
            elif stack and stack[-1][0] == "unit":
                kind = f"internal_{kind}"
            args = [a for a in (x.strip() for x in _split_top(m.group(3) or "")) if a and a != "*"]
            u = ScannedUnit(m.group(2), kind, file, st.line, args)
            units.append(u)
            stack.append(("unit", u))
            continue
        if _MODULE.match(t) and not t.startswith("MODULE PROCEDURE"):
            stack.append(("module", _MODULE.match(t).group(1)))  # type: ignore[union-attr]
            continue
        if _INTERFACE.match(t):
            stack.append(("interface", None))
            continue
        if _TYPE_BLOCK.match(t) and not _TYPE_DECL.match(t):
            stack.append(("type", None))
            continue
        if _BLOCK_DATA.match(t):
            stack.append(("blockdata", None))
            continue
        if t == "CONTAINS":  # the next headers nest inside the unit / module on the stack
            continue
        if _END.match(t):
            if stack:
                stack.pop()
            continue
        u = cur_unit()
        if u is None:
            continue
        u.body.append(st)
        _scan_statement(u, t)

    for u in units:
        u.calls = sorted(set(u.calls))
    return units


def _scan_statement(u: ScannedUnit, t: str) -> None:
    action = _strip_logical_if(t)
    m = _CALL.match(action)
    if m:
        u.calls.append(re.sub(r"\s", "", m.group(1)).split("%")[-1])
        return
    m = _SAVE.match(t)
    if m and not re.match(r"^SAVE\s*=", t):
        rest = m.group(1).strip()
        if not rest:
            u.save_all = True
        else:
            for it in _split_top(rest):
                it = it.strip()
                if not it.startswith("/"):
                    u.save_names.append(it)
        return
    m = _COMMON.match(t)
    if m and not re.match(r"^COMMON\s*=", t):
        for blk in _parse_common(m.group(1)):
            u.common_blocks.append(blk)
            u.arrays |= {n for n, d in _declarator_names(m.group(1)) if d}
        return
    m = _DIMENSION.match(t)
    if m and not re.match(r"^DIMENSION\s*=", t):
        u.arrays |= {n for n, d in _declarator_names(m.group(1)) if d}
        return
    if re.match(r"^EXTERNAL\b", t):
        u.externals |= set(re.findall(r"[A-Z]\w*", t[len("EXTERNAL") :]))
        return
    if _INCLUDE.match(t):
        u.includes.append(t)
        return
    m = _ENTRY.match(t)
    if m:
        u.entries.append(m.group(1))
        return
    m = _TYPE_DECL.match(t)
    if m and not re.match(r"^\w+\s*=", t) and "FUNCTION" not in t.split("(")[0]:
        attrs, items = m.group(1) or "", m.group(2)
        names = _declarator_names(items)
        u.arrays |= {n for n, d in names if d}
        if re.search(r"\bDIMENSION\b", attrs):
            u.arrays |= {n for n, _ in names}
        if re.search(r"\bSAVE\b", attrs):
            u.save_attr_names.extend(n for n, _ in names)
        return
    m = _STMT_FUNC.match(t)
    if m and m.group(1) not in u.arrays and m.group(1) != u.name:
        # a statement function definition, unless the name is (later found to be) an array;
        # arrays are declared before the executable part so this order is safe.
        u.stmt_functions.add(m.group(1))


def _is_free(path: Path, free: bool | None) -> bool:
    return path.suffix.lower() in FREE_SUFFIXES if free is None else free


def scan_file(
    path: str | Path, *, free: bool | None = None, line_length: int | None = 72
) -> list[ScannedUnit]:
    """Scan one file (latin-1, as the Fortran sources are not all UTF-8)."""
    p = Path(path)
    return scan_text(
        p.read_text(encoding="latin-1"), file=str(p), free=_is_free(p, free), line_length=line_length
    )


def scan_files(
    paths: Iterable[str | Path], *, free: bool | None = None, line_length: int | None = 72
) -> list[ScannedUnit]:
    out: list[ScannedUnit] = []
    for p in paths:
        out.extend(scan_file(p, free=free, line_length=line_length))
    return out


def _function_refs(u: ScannedUnit, functions: set[str]) -> set[str]:
    """Scanned FUNCTION names referenced as ``NAME(`` in ``u`` (arrays / statement functions excluded)."""
    cands = functions - u.arrays - u.stmt_functions - set(u.args)
    if not cands:
        return set()
    pat = re.compile(r"(?<![\w%])(" + "|".join(sorted(map(re.escape, cands))) + r")\s*\(")
    found: set[str] = set()
    for st in u.body:
        if _TYPE_DECL.match(st.text) and not re.match(r"^\w+\s*=", st.text):
            continue  # a declaration: NAME( here is a declarator, not a reference
        found |= set(pat.findall(st.text))
    return found


def scan_edges(units: Sequence[ScannedUnit]) -> dict[str, list[str]]:
    """Call graph among the scanned units: CALL targets plus references to scanned FUNCTIONs."""
    by_name: dict[str, ScannedUnit] = {}
    for u in units:
        by_name.setdefault(u.name, u)
    functions = {n for n, u in by_name.items() if u.base_kind == "function"}
    graph: dict[str, list[str]] = {}
    for name, u in by_name.items():
        refs = {c for c in u.calls if c in by_name} | _function_refs(u, functions)
        graph[name] = sorted(refs)
    return graph


# ---------------------------------------------------------------------------
# Comparison with the index
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Disagreement:
    """One fact on which the scanner and the index differ."""

    what: str  # 'routine' | 'kind' | 'args' | 'calls' | 'edge' | 'common' | 'save' | 'unsupported'
    routine: str
    detail: str

    def __str__(self) -> str:
        return f"{self.what}:{self.routine}: {self.detail}"


def index_edges(index: dict[str, Any]) -> dict[str, list[str]]:
    return {k: sorted(v) for k, v in index["call_graph"].items()}


def compare(index: dict[str, Any], units: Sequence[ScannedUnit]) -> list[Disagreement]:
    """Every disagreement between an index JSON (``TreeIndex.to_dict()``) and scanned units.

    Compared: the set of routine names (with multiplicity), per routine the kind, the ordered
    argument list, the full sorted list of ``CALL`` targets (internal and external), ``COMMON``
    blocks with their members in order, explicit ``SAVE`` names (the index lists them as
    ``saved_vars`` with ``kind='save'``), bare ``SAVE`` (``kind='save_all'``), and the call-graph
    edges. Scanner features the index expands but the scanner does not (``INCLUDE``, ``ENTRY``)
    are reported as ``unsupported`` instead of silently compared.
    """
    out: list[Disagreement] = []
    isubs: list[dict[str, Any]] = index["subroutines"]
    inames = sorted(s["name"] for s in isubs)
    snames = sorted(u.name for u in units)
    for n in sorted(set(inames) | set(snames)):
        ci, cs = inames.count(n), snames.count(n)
        if ci != cs:
            out.append(Disagreement("routine", n, f"index has {ci}, scanner has {cs}"))

    iby: dict[str, dict[str, Any]] = {}
    for s in isubs:
        iby.setdefault(s["name"], s)
    sby: dict[str, ScannedUnit] = {}
    for u in units:
        sby.setdefault(u.name, u)

    for n in sorted(set(iby) & set(sby)):
        s, u = iby[n], sby[n]
        if u.includes or u.entries:
            out.append(Disagreement("unsupported", n, f"INCLUDE={u.includes} ENTRY={u.entries}"))
        if s["kind"] != u.kind:
            out.append(Disagreement("kind", n, f"index {s['kind']!r} vs scanner {u.kind!r}"))
        if s["args"] != u.args:
            out.append(Disagreement("args", n, f"index {s['args']} vs scanner {u.args}"))
        if sorted(s["calls"]) != u.calls:
            only_i = sorted(set(s["calls"]) - set(u.calls))
            only_s = sorted(set(u.calls) - set(s["calls"]))
            out.append(Disagreement("calls", n, f"index-only {only_i}, scanner-only {only_s}"))
        icommon = [(c["name"], list(c["vars"])) for c in s["common_blocks"]]
        scommon = [(c["name"], list(c["vars"])) for c in u.common_blocks]
        if icommon != scommon:
            out.append(Disagreement("common", n, f"index {icommon} vs scanner {scommon}"))
        isave = sorted(v["name"] for v in s["saved_vars"] if v["kind"] == "save")
        ssave = sorted(set(u.save_names) | set(u.save_attr_names))
        if isave != ssave:
            out.append(Disagreement("save", n, f"index {isave} vs scanner {ssave}"))
        # bare SAVE: the index gives every assigned local (not explicitly saved) kind='save_all'
        isave_all = sorted(v["name"] for v in s["saved_vars"] if v["kind"] == "save_all")
        expect = sorted((set(s["locals"]) & set(s["assigned_vars"])) - set(ssave)) if u.save_all else []
        if isave_all != expect:
            out.append(
                Disagreement("save", n, f"bare SAVE={u.save_all}: index save_all {isave_all} vs {expect}")
            )

    ig = index_edges(index)
    sg = scan_edges(units)
    for n in sorted(set(ig) | set(sg)):
        a, b = set(ig.get(n, [])), set(sg.get(n, []))
        for v in sorted(a - b):
            out.append(Disagreement("edge", n, f"index-only edge {n} -> {v}"))
        for v in sorted(b - a):
            out.append(Disagreement("edge", n, f"scanner-only edge {n} -> {v}"))
    return out


def check_topological_order(graph: dict[str, list[str]], order: Sequence[str]) -> list[str]:
    """Violations of "callee before caller" in ``order``; empty when the property holds.

    Written without the index's Tarjan code: ``order`` must be a permutation of the graph's
    nodes, and for every edge ``u -> v`` either ``v`` precedes ``u`` or ``u`` and ``v`` lie on a
    common cycle (mutual reachability, found by plain DFS), in which case no order can satisfy
    both directions.
    """
    problems: list[str] = []
    nodes = set(graph)
    if sorted(order) != sorted(nodes):
        problems.append(f"order is not a permutation of the graph nodes: {sorted(set(order) ^ nodes)}")
    pos = {n: i for i, n in enumerate(order)}

    def reach(src: str) -> set[str]:
        seen: set[str] = set()
        todo = list(graph.get(src, []))
        while todo:
            v = todo.pop()
            if v in seen:
                continue
            seen.add(v)
            todo.extend(graph.get(v, []))
        return seen

    reach_cache: dict[str, set[str]] = {}
    for u, succ in graph.items():
        for v in succ:
            if v not in pos or u not in pos or v == u:
                continue
            if pos[v] < pos[u]:
                continue
            ru = reach_cache.setdefault(u, reach(u))
            rv = reach_cache.setdefault(v, reach(v))
            if not (v in ru and u in rv):
                problems.append(f"{v} (callee) is after {u} (caller)")
    return problems


def kahn_is_acyclic(graph: dict[str, list[str]]) -> bool:
    """Kahn's algorithm: True when every node can be removed, i.e. the graph has no cycle."""
    indeg = {n: 0 for n in graph}
    for succ in graph.values():
        for v in succ:
            if v in indeg:
                indeg[v] += 1
    todo = [n for n, d in indeg.items() if d == 0]
    removed = 0
    while todo:
        n = todo.pop()
        removed += 1
        for v in graph[n]:
            if v in indeg:
                indeg[v] -= 1
                if indeg[v] == 0:
                    todo.append(v)
    return removed == len(graph)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agrijax.port.fortran_xcheck", description=(__doc__ or "").split("\n")[0]
    )
    ap.add_argument("index", help="index JSON written by agrijax.port.fortran_index")
    ap.add_argument(
        "--line-length", type=int, default=72, help="fixed-form statement field end (0 = unlimited)"
    )
    ns = ap.parse_args(argv)
    index = json.loads(Path(ns.index).read_text())
    units = scan_files(index["files"], line_length=ns.line_length or None)
    diffs = compare(index, units)
    topo = check_topological_order(index["call_graph"], index["topological_order"])
    print(f"scanned {len(units)} units; index has {len(index['subroutines'])}")
    for d in diffs:
        print(d)
    for p in topo:
        print(f"topological: {p}")
    return 1 if diffs or topo else 0


if __name__ == "__main__":
    sys.exit(main())
