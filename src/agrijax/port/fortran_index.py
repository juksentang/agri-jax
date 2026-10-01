"""Index Fortran sources with fparser: call graph, argument intent and hidden state.

This is step 1 ("Index") of the porting procedure.
For every ``SUBROUTINE`` / ``FUNCTION`` / ``PROGRAM`` it records a :class:`SubroutineInfo`:
arguments, ``CALL`` targets, ``COMMON`` blocks, ``USE`` statements, a guessed ``intent`` per
argument, and the variables that carry state between calls (``saved_vars``).

Dataflow model
--------------
The execution part is lowered to a tiny IR (read / write / call / branch) and evaluated with
a *must-define* analysis in source order:

* ``IF`` / ``ELSE IF`` / ``SELECT CASE`` / ``WHERE`` branches: a variable is defined after the
  construct only if every branch defines it (a missing ``ELSE`` is an empty branch). This is what
  catches the classic ``IF (FIRST) THEN; X = 0; ENDIF; X = X + 1`` state carrier.
* ``DO`` loops are assumed to execute at least once (so initialisation loops define arrays).
* Writing an array element (``A(I) = ...``) counts as defining ``A``.
* ``GOTO`` / arithmetic ``IF`` are ignored (straight-line approximation).
* Actual arguments of a ``CALL`` use the callee's intent when the callee is indexed (see
  :func:`index_tree`, which propagates intents bottom-up through the call graph); for an unknown
  callee the argument is treated as ``inout`` for the caller's dummies (conservative) and as a
  plain definition for locals (so ``CALL GETLUN('OUT', LUN)`` does not flag ``LUN``).

Intent guess (per dummy argument): assigned somewhere and never read before it is definitely
assigned -> ``'out'``; read before definitely assigned and assigned later -> ``'inout'``;
never assigned -> ``'in'``. An explicit ``INTENT(...)`` declaration overrides the guess.

Saved variables: names in ``SAVE`` statements / ``SAVE`` attributes (``kind='save'``; a bare
``SAVE`` gives every assigned local ``kind='save_all'``), locals initialised by ``DATA`` or a
declaration initialiser and assigned in the body (``kind='data_init'``, implicitly saved by the
standard), and, because the binaries are compiled with ``-save``, every other local that is read
before it is definitely assigned and assigned somewhere (``kind='implicit_save'``). Each entry
also says whether it is ``read_before_assigned`` (the ones that really carry state).

Locals are declared names plus, when the unit has no wildcard ``USE`` (``USE M`` without
``ONLY``), undeclared names that are assigned (implicit typing). Dummies, ``COMMON`` members,
``PARAMETER`` constants, the function result and statement functions are never locals.

CLI::

    python -m agrijax.port.fortran_index <files...> -o index.json [--root RICHRD --root POTEVP]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "SavedVar",
    "SubroutineInfo",
    "TreeIndex",
    "index_file",
    "index_tree",
    "main",
    "parse_file",
    "topological_order",
]

FREE_SUFFIXES = {".f90", ".f95", ".f03", ".f08"}

# IO keywords whose value is written by the statement.
_IO_WRITE_KEYWORDS = {"IOSTAT", "IOMSG", "SIZE", "NEWUNIT", "STAT", "ERRMSG"}
# INQUIRE: every specifier except these is an output.
_INQUIRE_INPUT_KEYWORDS = {"UNIT", "FILE", "ID", "ERR", None}

Ev = tuple[Any, ...]
"""IR event: ``("R", name, line)``, ``("W", name, line)``,
``("C", callee, ((key, var), ...), line)`` with key an ``int`` position or ``str`` keyword,
``("B", (branch_events, ...), exhaustive)``."""

CalleeSig = Callable[[str], "tuple[list[str], dict[str, str]] | None"]


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class SavedVar:
    """A variable that may persist between calls."""

    name: str
    kind: str  # 'save' | 'save_all' | 'data_init' | 'implicit_save'
    read_before_assigned: bool
    first_read_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "read_before_assigned": self.read_before_assigned,
            "first_read_line": self.first_read_line,
        }


@dataclass
class SubroutineInfo:
    """Index record of one program unit."""

    name: str
    file: str
    line_start: int
    line_end: int
    kind: str  # subroutine | function | program | module_subroutine | internal_function | ...
    args: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    common_blocks: list[dict[str, Any]] = field(default_factory=list)
    module_uses: list[dict[str, Any]] = field(default_factory=list)
    saved_vars: list[SavedVar] = field(default_factory=list)
    assigned_vars: list[str] = field(default_factory=list)
    read_vars: list[str] = field(default_factory=list)
    intent_guess: dict[str, str] = field(default_factory=dict)
    intent_declared: dict[str, str] = field(default_factory=dict)
    function_refs: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    locals: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    unresolved_call_args: list[str] = field(default_factory=list)
    # internal analysis state (not serialised)
    _flow: list[Ev] = field(default_factory=list, repr=False)
    _ctx: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def implicit_save(self) -> list[str]:
        return [s.name for s in self.saved_vars if s.kind == "implicit_save"]

    @property
    def state_vars(self) -> list[str]:
        """Saved variables that are read before being assigned (the real hidden state)."""
        return [s.name for s in self.saved_vars if s.read_before_assigned]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": self.file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "lines": self.line_end - self.line_start + 1,
            "kind": self.kind,
            "args": list(self.args),
            "intent_guess": dict(self.intent_guess),
            "intent_declared": dict(self.intent_declared),
            "calls": list(self.calls),
            "function_refs": list(self.function_refs),
            "common_blocks": [dict(c) for c in self.common_blocks],
            "module_uses": [dict(u) for u in self.module_uses],
            "saved_vars": [s.to_dict() for s in self.saved_vars],
            "implicit_save": self.implicit_save,
            "assigned_vars": list(self.assigned_vars),
            "read_vars": list(self.read_vars),
            "locals": list(self.locals),
            "parameters": list(self.parameters),
            "entries": list(self.entries),
            "unresolved_call_args": list(self.unresolved_call_args),
        }

    # -- analysis ------------------------------------------------------------
    def analyse(self, callee_sig: CalleeSig | None = None) -> None:
        """(Re)compute read/assigned sets, intents and saved variables from the IR."""
        st = _FlowState(callee_sig or (lambda _n: None))
        st.run(self._flow, set())
        ctx = self._ctx
        args = self.args
        argset = set(args)
        self.read_vars = sorted(st.read)
        self.assigned_vars = sorted(st.assigned)
        self.unresolved_call_args = sorted(st.unresolved)

        intents: dict[str, str] = {}
        for a in args:
            exposed = a in st.exposed or a in st.exposed_unknown
            if a in st.assigned:
                intents[a] = "inout" if exposed else "out"
            else:
                intents[a] = "in"
        intents.update({k: v for k, v in self.intent_declared.items() if k in argset})
        self.intent_guess = intents

        excluded = argset | ctx["common_vars"] | ctx["parameters"] | ctx["stmt_functions"]
        excluded |= ctx["externals"] | ctx["result_names"] | set(self.entries)
        declared: set[str] = ctx["declared"]
        local_set = {n for n in declared if n not in excluded}
        if not ctx["wildcard_use"]:
            callees = set(self.calls) | set(self.function_refs)
            local_set |= {n for n in st.assigned if n not in excluded and n not in callees}
        self.locals = sorted(local_set)

        explicit: set[str] = ctx["save_names"]
        data_init: set[str] = ctx["data_init"]
        saved: list[SavedVar] = []
        for n in sorted(local_set):
            exposed_line = st.exposed.get(n)
            rba = exposed_line is not None and n in st.assigned
            if n in explicit:
                kind = "save"
            elif ctx["save_all"] and n in st.assigned:
                kind = "save_all"
            elif n in data_init and n in st.assigned:
                kind = "data_init"
            elif rba:
                kind = "implicit_save"
            else:
                continue
            saved.append(SavedVar(n, kind, rba, exposed_line if rba else None))
        self.saved_vars = saved


class _FlowState:
    """Must-define evaluation of the IR."""

    def __init__(self, callee_sig: CalleeSig) -> None:
        self.callee_sig = callee_sig
        self.read: set[str] = set()
        self.assigned: set[str] = set()
        self.exposed: dict[str, int] = {}
        self.exposed_unknown: dict[str, int] = {}
        self.unresolved: set[str] = set()

    def _r(self, name: str, line: int, defined: set[str]) -> None:
        self.read.add(name)
        if name not in defined:
            self.exposed.setdefault(name, line)

    def _w(self, name: str, defined: set[str]) -> None:
        self.assigned.add(name)
        defined.add(name)

    def run(self, seq: Sequence[Ev], defined: set[str]) -> set[str]:
        for ev in seq:
            tag = ev[0]
            if tag == "R":
                self._r(ev[1], ev[2], defined)
            elif tag == "W":
                self._w(ev[1], defined)
            elif tag == "C":
                self._call(ev[1], ev[2], ev[3], defined)
            elif tag == "B":
                outs = [self.run(br, set(defined)) for br in ev[1]]
                if not ev[2]:
                    outs.append(set(defined))
                new = set.intersection(*outs) if outs else set(defined)
                defined.clear()
                defined.update(new)
        return defined

    def _call(self, callee: str, args: Sequence[tuple[int | str, str]], line: int, defined: set[str]) -> None:
        sig = self.callee_sig(callee)
        writes: list[str] = []
        for key, var in args:
            intent: str | None = None
            if sig is not None:
                names, intents = sig
                dummy = names[key] if isinstance(key, int) and key < len(names) else key
                intent = intents.get(dummy) if isinstance(dummy, str) else None
            if intent is None:
                self.unresolved.add(f"{callee}:{key}:{var}")
                self.read.add(var)
                if var not in defined:
                    self.exposed_unknown.setdefault(var, line)
                writes.append(var)
            else:
                if intent in ("in", "inout"):
                    self._r(var, line, defined)
                if intent in ("out", "inout"):
                    writes.append(var)
        for var in writes:
            self._w(var, defined)


# ---------------------------------------------------------------------------
# fparser front end
# ---------------------------------------------------------------------------
def _fparser() -> Any:
    from fparser.two.parser import ParserFactory

    global _PARSER
    if _PARSER is None:
        logging.getLogger("fparser").setLevel(logging.ERROR)
        _PARSER = ParserFactory().create(std="f2003")
    return _PARSER


_PARSER: Any = None


def _is_free(path: Path, free: bool | None) -> bool:
    return path.suffix.lower() in FREE_SUFFIXES if free is None else free


def parse_file(path: str | Path, *, free: bool | None = None, include_dirs: Sequence[str] = ()) -> Any:
    """Parse one file with fparser; on failure retry with preprocessor lines blanked out."""
    from fparser.common.readfortran import FortranFileReader, FortranStringReader
    from fparser.common.sourceinfo import FortranFormat
    from fparser.two.symbol_table import SYMBOL_TABLES

    path = Path(path)
    parser = _fparser()
    is_free = _is_free(path, free)
    incs = [str(path.parent), *include_dirs]
    if sys.getrecursionlimit() < 10000:
        sys.setrecursionlimit(10000)

    def _parse(reader: Any) -> Any:
        SYMBOL_TABLES.clear()
        reader.set_format(FortranFormat(is_free, False))
        tree = parser(reader)
        if tree is None:
            raise ValueError("fparser returned no tree")
        return tree

    try:
        return _parse(FortranFileReader(str(path), ignore_comments=True, include_dirs=incs))
    except Exception as first:  # fparser raises many unrelated exception types
        text = path.read_text(encoding="latin-1")
        comment = "!" if is_free else "C"
        lines = [comment + ln[1:] if ln.startswith("#") else ln for ln in text.splitlines()]
        cleaned = "\n".join(lines) + "\n"
        if cleaned == text or cleaned == text + "\n":
            raise first
        try:
            return _parse(FortranStringReader(cleaned, ignore_comments=True, include_dirs=incs))
        except Exception:
            raise first from None


def _cls(node: Any) -> str:
    return type(node).__name__


def _up(node: Any) -> str:
    return str(node).strip().upper()


def _children(node: Any) -> Iterator[Any]:
    """Children of a fparser node, flattening tuples/lists (e.g. Loop_Control, Common_Stmt)."""
    from fparser.two.utils import Base

    items = node.children if isinstance(node, Base) else node
    for c in items:
        if isinstance(c, (list, tuple)):
            yield from _children(c)
        elif c is not None:
            yield c


def _walk(node: Any, stop: Callable[[Any], bool] = lambda _n: False) -> Iterator[Any]:
    from fparser.two.utils import Base

    for c in _children(node):
        if isinstance(c, Base):
            yield c
            if not stop(c):
                yield from _walk(c, stop)


def _line(node: Any, default: int) -> int:
    item = getattr(node, "item", None)
    span = getattr(item, "span", None)
    return int(span[0]) if span else default


def _line_end(node: Any, default: int) -> int:
    item = getattr(node, "item", None)
    span = getattr(item, "span", None)
    return int(span[1]) if span else default


class _Unit:
    """Builds the IR and the declaration context of one program unit."""

    def __init__(self, node: Any, kind: str, file: str, host_uses: Sequence[dict[str, Any]] = ()) -> None:
        self.node = node
        self.kind = kind
        self.file = file
        self.arrays: set[str] = set()
        self.chars: set[str] = set()
        self.declared: set[str] = set()
        self.common: list[dict[str, Any]] = []
        self.common_vars: set[str] = set()
        self.uses: list[dict[str, Any]] = [dict(u, inherited=True) for u in host_uses]
        self.save_names: set[str] = set()
        self.save_all = False
        self.data_init: set[str] = set()
        self.parameters: set[str] = set()
        self.externals: set[str] = set()
        self.stmt_functions: set[str] = set()
        self.intent_declared: dict[str, str] = {}
        self.calls: list[str] = []
        self.function_refs: list[str] = []
        self.entries: list[str] = []
        self.result_names: set[str] = set()

    # -- declarations ---------------------------------------------------------
    def spec(self, spec: Any) -> None:
        for st in _walk(spec, stop=lambda n: _cls(n) == "Interface_Block"):
            name = _cls(st)
            if name == "Use_Stmt":
                items = st.items
                only = items[4]
                self.uses.append(
                    {
                        "module": _up(items[2]),
                        "only": sorted({_up(x).split("=>")[0].strip() for x in _children(only)})
                        if only is not None
                        else None,
                    }
                )
            elif name == "Type_Declaration_Stmt":
                self._type_decl(st)
            elif name == "Dimension_Stmt":
                self._dimension(st)
            elif name == "Common_Stmt":
                self._common(st)
            elif name == "Save_Stmt":
                ents = st.items[1] if len(st.items) > 1 else None
                if ents is None:
                    self.save_all = True
                else:
                    for e in _children(ents):
                        if _cls(e) == "Name":
                            self.save_names.add(_up(e))
            elif name == "Data_Stmt":
                self.data_init |= self._data_names(st)
            elif name == "Parameter_Stmt":
                for c in _walk(st):
                    if _cls(c) == "Named_Constant_Def":
                        self.parameters.add(_up(c.items[0]))
            elif name in ("External_Stmt", "Intrinsic_Stmt"):
                for c in _walk(st):
                    if _cls(c) == "Name":
                        self.externals.add(_up(c))
            elif name == "Stmt_Function_Stmt":
                self.stmt_functions.add(_up(st.items[0]))
            elif name == "Intent_Stmt":
                spec_, names = st.items[0], st.items[1]
                for c in _children(names):
                    self.intent_declared[_up(c)] = _up(spec_).replace(" ", "").lower()

    def _dimension(self, st: Any) -> None:
        for c in _children(st):
            if _cls(c) == "Name":
                self.arrays.add(_up(c))
                self.declared.add(_up(c))

    def _type_decl(self, st: Any) -> None:
        attrs = st.items[1]
        attr_strs = [_up(a) for a in _children(attrs)] if attrs is not None else []
        is_dim = any(s.startswith("DIMENSION") for s in attr_strs)
        is_param = "PARAMETER" in attr_strs
        is_save = "SAVE" in attr_strs
        is_ext = "EXTERNAL" in attr_strs
        intent = next((s for s in attr_strs if s.startswith("INTENT")), None)
        is_char = _up(st.items[0]).startswith("CHARACTER")
        for ent in _walk(st.items[2]):
            if _cls(ent) != "Entity_Decl":
                continue
            n = _up(ent.items[0])
            self.declared.add(n)
            if is_char:
                self.chars.add(n)
            if is_dim or ent.items[1] is not None:
                self.arrays.add(n)
            if is_param:
                self.parameters.add(n)
            elif len(ent.items) > 3 and ent.items[3] is not None:
                self.data_init.add(n)
            if is_save:
                self.save_names.add(n)
            if is_ext:
                self.externals.add(n)
            if intent:
                self.intent_declared[n] = intent.replace("INTENT", "").strip("() ").replace(" ", "").lower()

    def _common(self, st: Any) -> None:
        for blk, objs in st.items[0]:
            bname = _up(blk) if blk is not None else ""
            names: list[str] = []
            for o in _children(objs):
                if _cls(o) == "Name":
                    names.append(_up(o))
                elif _cls(o) == "Common_Block_Object":
                    n = _up(o.items[0])
                    names.append(n)
                    self.arrays.add(n)
            self.common.append({"name": bname, "vars": names})
            self.common_vars |= set(names)

    def _data_names(self, st: Any) -> set[str]:
        out: set[str] = set()
        for c in _walk(st):
            if _cls(c) == "Data_Stmt_Object_List":
                for o in _children(c):
                    b = self._base_name(o)
                    if b:
                        out.add(b)
        return out

    @staticmethod
    def _base_name(node: Any) -> str | None:
        c = _cls(node)
        if c == "Name":
            return _up(node)
        if c in ("Part_Ref", "Data_Ref", "Array_Section", "Substring_Range"):
            return _Unit._base_name(node.items[0])
        return None

    # -- expressions ------------------------------------------------------------
    def _is_var_ref(self, part_ref: Any) -> bool:
        n = _up(part_ref.items[0])
        if n in self.arrays:
            return True
        subs = part_ref.items[1]
        return any(_cls(s) in ("Subscript_Triplet", "Substring_Range") for s in _children(subs))

    def reads(self, node: Any, line: int, out: list[Ev]) -> None:
        c = _cls(node)
        if c == "Name":
            out.append(("R", _up(node), line))
        elif c == "Part_Ref":
            n = _up(node.items[0])
            if self._is_var_ref(node):
                out.append(("R", n, line))
            elif n not in self.stmt_functions:
                self.function_refs.append(n)
            for s in _children(node.items[1]):
                self.reads(s, line, out)
        elif c == "Function_Reference":
            self.function_refs.append(_up(node.items[0]))
            for s in _children(node.items[1:]):
                self.reads(s, line, out)
        elif c == "Data_Ref":
            parts = list(_children(node))
            base = self._base_name(parts[0])
            if base:
                out.append(("R", base, line))
            for p in parts:
                if _cls(p) == "Part_Ref":
                    for s in _children(p.items[1]):
                        self.reads(s, line, out)
        elif c == "Intrinsic_Function_Reference":
            for s in _children(node.items[1:]):
                self.reads(s, line, out)
        elif c == "Actual_Arg_Spec":
            self.reads(node.items[1], line, out)
        elif c in ("Keyword", "Label", "Intrinsic_Name", "Type_Name", "Alt_Return_Spec"):
            return
        elif c == "Io_Implied_Do":
            self._implied_do(node, line, out, write_items=False)
        elif hasattr(node, "children"):
            for s in _children(node):
                if not isinstance(s, str):
                    self.reads(s, line, out)

    def _subscript_reads(self, node: Any, line: int, out: list[Ev]) -> None:
        c = _cls(node)
        if c == "Part_Ref":
            for s in _children(node.items[1]):
                self.reads(s, line, out)
        elif c in ("Data_Ref", "Array_Section", "Substring_Range"):
            for p in _children(node):
                if not isinstance(p, str):
                    self._subscript_reads(p, line, out)

    def writes(self, node: Any, line: int, out: list[Ev]) -> None:
        c = _cls(node)
        if c == "Io_Implied_Do":
            self._implied_do(node, line, out, write_items=True)
            return
        self._subscript_reads(node, line, out)
        base = self._base_name(node)
        if base:
            out.append(("W", base, line))
        else:
            self.reads(node, line, out)

    def _implied_do(self, node: Any, line: int, out: list[Ev], *, write_items: bool) -> None:
        objs, ctrl = node.items[0], node.items[1]
        cparts = list(_children(ctrl))
        for b in cparts[1:]:
            self.reads(b, line, out)
        if cparts:
            out.append(("W", _up(cparts[0]), line))
        for o in _children(objs):
            (self.writes if write_items else self.reads)(o, line, out)

    # -- statements ---------------------------------------------------------------
    def block(self, nodes: Iterable[Any], line: int) -> list[Ev]:
        out: list[Ev] = []
        for n in nodes:
            if isinstance(n, str):
                continue
            ln = _line(n, line)
            out.extend(self.node_events(n, ln))
            line = _line_end(n, ln)
        return out

    def node_events(self, n: Any, line: int) -> list[Ev]:
        c = _cls(n)
        if c == "If_Construct":
            return self._if_construct(n, line)
        if c == "Case_Construct":
            return self._case_construct(n, line)
        if c in ("Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct", "Action_Term_Do_Construct"):
            return self._do_construct(n, line)
        if c == "Where_Construct":
            return self._where_construct(n, line)
        if c.endswith("_Construct") or c == "Execution_Part":
            return self.block(n.children, line)
        return self.stmt(n, line)

    def _if_construct(self, n: Any, line: int) -> list[Ev]:
        out: list[Ev] = []
        branches: list[list[Ev]] = []
        prefix: list[Ev] = []  # reads of the conditions tested before the current branch
        cur: list[Any] | None = None
        cur_line = line
        exhaustive = False
        for ch in n.children:
            c = _cls(ch)
            ln = _line(ch, cur_line)
            if c in ("If_Then_Stmt", "Else_If_Stmt"):
                if cur is not None:
                    branches.append([*prefix, *self.block(cur, cur_line)])
                cond: list[Ev] = []
                self.reads(ch.items[0], ln, cond)
                if c == "If_Then_Stmt":
                    out.extend(cond)
                else:
                    prefix = [*prefix, *cond]
                cur, cur_line = [], ln
            elif c == "Else_Stmt":
                if cur is not None:
                    branches.append([*prefix, *self.block(cur, cur_line)])
                cur, cur_line, exhaustive = [], ln, True
            elif c == "End_If_Stmt":
                continue
            elif cur is not None:
                cur.append(ch)
        if cur is not None:
            branches.append([*prefix, *self.block(cur, cur_line)])
        if not exhaustive:
            branches.append(list(prefix))
            exhaustive = True
        out.append(("B", tuple(tuple(b) for b in branches), exhaustive))
        return out

    def _case_construct(self, n: Any, line: int) -> list[Ev]:
        out: list[Ev] = []
        branches: list[list[Ev]] = []
        cur: list[Any] | None = None
        cur_line = line
        exhaustive = False
        for ch in n.children:
            c = _cls(ch)
            ln = _line(ch, cur_line)
            if c == "Select_Case_Stmt":
                self.reads(ch.items[0], ln, out)
            elif c == "Case_Stmt":
                if cur is not None:
                    branches.append(self.block(cur, cur_line))
                if "DEFAULT" in _up(ch):
                    exhaustive = True
                cur, cur_line = [], ln
            elif c == "End_Select_Stmt":
                continue
            elif cur is not None:
                cur.append(ch)
        if cur is not None:
            branches.append(self.block(cur, cur_line))
        out.append(("B", tuple(tuple(b) for b in branches), exhaustive))
        return out

    def _do_construct(self, n: Any, line: int) -> list[Ev]:
        out: list[Ev] = []
        kids = list(n.children)
        head = kids[0]
        lc = next((x for x in _children(head) if _cls(x) == "Loop_Control"), None)
        if lc is not None:
            while_expr, counter = lc.items[0], lc.items[1]
            if counter is not None:
                var, bounds = counter
                for b in bounds:
                    self.reads(b, line, out)
                out.append(("W", _up(var), line))
            if while_expr is not None:
                self.reads(while_expr, line, out)
        body = [k for k in kids[1:] if _cls(k) != "End_Do_Stmt"]
        out.extend(self.block(body, line))  # assumed to execute at least once
        return out

    def _where_construct(self, n: Any, line: int) -> list[Ev]:
        out: list[Ev] = []
        branches: list[list[Ev]] = []
        cur: list[Any] = []
        for ch in n.children:
            c = _cls(ch)
            ln = _line(ch, line)
            if c in ("Where_Construct_Stmt", "Masked_Elsewhere_Stmt", "Elsewhere_Stmt"):
                if cur:
                    branches.append(self.block(cur, line))
                    cur = []
                for x in _children(ch):
                    if not isinstance(x, str):
                        self.reads(x, ln, out)
            elif c == "End_Where_Stmt":
                continue
            else:
                cur.append(ch)
        if cur:
            branches.append(self.block(cur, line))
        out.append(("B", tuple(tuple(b) for b in branches), False))
        return out

    def stmt(self, n: Any, line: int) -> list[Ev]:
        c = _cls(n)
        out: list[Ev] = []
        if c in ("Assignment_Stmt", "Pointer_Assignment_Stmt"):
            self.reads(n.items[2], line, out)
            self.writes(n.items[0], line, out)
        elif c == "Call_Stmt":
            self._call(n, line, out)
        elif c in ("If_Stmt", "Where_Stmt"):
            self.reads(n.items[0], line, out)
            out.append(("B", (tuple(self.stmt(n.items[1], line)),), False))
        elif c == "Read_Stmt":
            self._io_specs(n.items[0], line, out, inquire=False)
            if len(n.items) > 1 and n.items[1] is not None and _cls(n.items[1]) != "Io_Control_Spec_List":
                # READ fmt, list  ->  items[1] is the format
                self.reads(n.items[1], line, out)
            items = n.items[2] if len(n.items) > 2 else None
            for it in _children(items) if items is not None else ():
                self.writes(it, line, out)
        elif c in ("Write_Stmt", "Print_Stmt"):
            for part in n.items:
                if part is None or isinstance(part, str):
                    continue
                if _cls(part) == "Io_Control_Spec_List":
                    self._io_specs(part, line, out, inquire=False, internal_unit=c == "Write_Stmt")
                else:
                    self.reads(part, line, out)
        elif c == "Inquire_Stmt":
            for part in n.items:
                if part is not None and not isinstance(part, str):
                    self._io_specs(part, line, out, inquire=True)
        elif c in (
            "Open_Stmt",
            "Close_Stmt",
            "Rewind_Stmt",
            "Backspace_Stmt",
            "Endfile_Stmt",
            "Flush_Stmt",
            "Wait_Stmt",
        ):
            for part in n.items:
                if part is not None and not isinstance(part, str):
                    self._io_specs(part, line, out, inquire=False)
        elif c in ("Allocate_Stmt", "Deallocate_Stmt", "Nullify_Stmt"):
            for x in _walk(n):
                b = self._base_name(x) if _cls(x) in ("Name", "Part_Ref", "Data_Ref") else None
                if b:
                    out.append(("W", b, line))
                    break
        elif c == "Entry_Stmt":
            self.entries.append(_up(n.items[0]))
        elif c in (
            "Continue_Stmt",
            "Goto_Stmt",
            "Format_Stmt",
            "Data_Stmt",
            "Exit_Stmt",
            "Cycle_Stmt",
            "End_Do_Stmt",
            "End_If_Stmt",
            "End_Select_Stmt",
            "Stop_Stmt",
            "Comment",
            "Include_Stmt",
        ):
            if c == "Data_Stmt":
                self.data_init |= self._data_names(n)
        elif c == "Return_Stmt":
            for x in _children(n):
                if not isinstance(x, str):
                    self.reads(x, line, out)
        else:
            # Arithmetic_If, Computed_Goto, and anything unforeseen: every name is a read.
            for x in _children(n):
                if not isinstance(x, str):
                    self.reads(x, line, out)
        return out

    def _io_specs(
        self, specs: Any, line: int, out: list[Ev], *, inquire: bool, internal_unit: bool = False
    ) -> None:
        if specs is None:
            return
        if _cls(specs).endswith("_List"):
            parts = list(_children(specs))
        else:
            parts = [specs]
        for k, sp in enumerate(parts):
            items = getattr(sp, "items", None)
            if (
                internal_unit
                and items
                and len(items) == 2
                and (items[0] == "UNIT" or (k == 0 and items[0] is None))
            ):
                # WRITE(CHARVAR, ...) is an internal write: the character variable is assigned.
                if self._base_name(items[1]) in self.chars:
                    self.writes(items[1], line, out)
                    continue
            items = getattr(sp, "items", None)
            if not items or len(items) != 2 or not (items[0] is None or isinstance(items[0], str)):
                self.reads(sp, line, out)
                continue
            kw = items[0].upper() if isinstance(items[0], str) else None
            is_write = kw not in _INQUIRE_INPUT_KEYWORDS if inquire else kw in _IO_WRITE_KEYWORDS
            if kw in ("ERR", "END", "EOR", "FMT") and _cls(items[1]) == "Label":
                continue
            if is_write:
                self.writes(items[1], line, out)
            else:
                self.reads(items[1], line, out)

    def _call(self, n: Any, line: int, out: list[Ev]) -> None:
        designator = n.items[0]
        callee = _up(designator).split("%")[-1]
        self.calls.append(callee)
        args: list[tuple[int | str, str]] = []
        arglist = n.items[1] if len(n.items) > 1 else None
        pos = 0
        for a in _children(arglist) if arglist is not None else ():
            if isinstance(a, str):
                continue
            key: int | str = pos
            expr = a
            if _cls(a) == "Actual_Arg_Spec":
                key = _up(a.items[0])
                expr = a.items[1]
            pos += 1
            if _cls(expr) == "Alt_Return_Spec":
                continue
            var = self._actual_var(expr)
            if var is None:
                self.reads(expr, line, out)
            else:
                self._subscript_reads(expr, line, out)
                args.append((key, var))
        out.append(("C", callee, tuple(args), line))

    def _actual_var(self, expr: Any) -> str | None:
        c = _cls(expr)
        if c == "Name":
            n = _up(expr)
            return None if n in self.parameters else n
        if c == "Part_Ref":
            return _up(expr.items[0]) if self._is_var_ref(expr) else None
        if c in ("Data_Ref", "Array_Section"):
            return self._base_name(expr)
        return None


# ---------------------------------------------------------------------------
# Units discovery
# ---------------------------------------------------------------------------
_UNIT_KINDS = {
    "Subroutine_Subprogram": ("subroutine", "Subroutine_Stmt"),
    "Function_Subprogram": ("function", "Function_Stmt"),
    "Main_Program": ("program", "Program_Stmt"),
}


def _units(tree: Any) -> Iterator[tuple[Any, str, list[dict[str, Any]]]]:
    """Yield (node, kind, host_uses) for every program unit, including contained ones."""

    def rec(node: Any, prefix: str, host: list[dict[str, Any]]) -> Iterator[tuple[Any, str, list]]:
        for ch in _children(node):
            c = _cls(ch)
            if c in _UNIT_KINDS:
                kind = _UNIT_KINDS[c][0]
                yield ch, f"{prefix}{kind}", host
                inner = next((x for x in _children(ch) if _cls(x) == "Internal_Subprogram_Part"), None)
                if inner is not None:
                    yield from rec(inner, "internal_", host)
            elif c == "Module":
                mname = _up(ch.children[0].items[1])
                mh = [*host, {"module": mname, "only": None}]
                for x in _children(ch):
                    if _cls(x) == "Module_Subprogram_Part":
                        yield from rec(x, "module_", mh)
            elif c in ("Interface_Block",):
                continue
            elif hasattr(ch, "children") and c in ("Program", "Internal_Subprogram_Part"):
                yield from rec(ch, prefix, host)

    yield from rec(tree, "", [])


def _index_unit(node: Any, kind: str, file: str, host_uses: list[dict[str, Any]]) -> SubroutineInfo:
    head = node.children[0]
    items = head.items
    if _cls(head) == "Program_Stmt":
        name, dummies, result = _up(items[1]), None, None
    else:
        name, dummies = _up(items[1]), items[2]
        result = None
        if _cls(head) == "Function_Stmt" and len(items) > 3 and items[3] is not None:
            for x in _walk(items[3]):
                if _cls(x) == "Name":
                    result = _up(x)
                    break
    args = [_up(a) for a in _children(dummies)] if dummies is not None else []
    args = [a for a in args if a != "*"]
    u = _Unit(node, kind, file, host_uses)
    if "function" in kind:
        u.result_names = {name} | ({result} if result else set())
    start = _line(head, 0)
    end = _line_end(node.children[-1], start)
    flow: list[Ev] = []
    for part in node.children[1:]:
        c = _cls(part)
        if c == "Specification_Part":
            u.spec(part)
        elif c == "Execution_Part":
            flow = u.block(part.children, start)
    info = SubroutineInfo(
        name=name,
        file=file,
        line_start=start,
        line_end=end,
        kind=kind,
        args=args,
        calls=sorted(set(u.calls)),
        common_blocks=u.common,
        module_uses=u.uses,
        intent_declared={k: v for k, v in u.intent_declared.items() if k in set(args)},
        function_refs=sorted(set(u.function_refs) - u.stmt_functions),
        entries=u.entries,
        parameters=sorted(u.parameters),
        _flow=flow,
        _ctx={
            "declared": u.declared,
            "common_vars": u.common_vars,
            "parameters": u.parameters,
            "stmt_functions": u.stmt_functions,
            "externals": u.externals,
            "result_names": u.result_names,
            "save_names": u.save_names,
            "save_all": u.save_all,
            "data_init": u.data_init,
            "wildcard_use": any(x["only"] is None for x in u.uses),
        },
    )
    info.analyse()
    return info


def index_file(
    path: str | Path, *, free: bool | None = None, include_dirs: Sequence[str] = ()
) -> list[SubroutineInfo]:
    """Index every program unit of one file (raises on parse failure; see :func:`index_tree`)."""
    path = Path(path)
    tree = parse_file(path, free=free, include_dirs=include_dirs)
    return [_index_unit(n, k, str(path), h) for n, k, h in _units(tree)]


# ---------------------------------------------------------------------------
# Call graph
# ---------------------------------------------------------------------------
def _sccs(graph: dict[str, list[str]]) -> list[list[str]]:
    """Tarjan's SCCs (iterative); emitted leaves first (reverse topological order)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in sorted(graph):
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            v, i = work.pop()
            if i == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on.add(v)
            succ = graph.get(v, [])
            if i < len(succ):
                work.append((v, i + 1))
                w = succ[i]
                if w not in index:
                    work.append((w, 0))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            if low[v] == index[v]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(sorted(comp))
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
    return out


def topological_order(graph: dict[str, list[str]]) -> tuple[list[str], list[list[str]]]:
    """Leaves-first order of ``graph`` (callee before caller) and its cycles.

    Members of a cycle are emitted together (alphabetically) at the position of their SCC.
    A cycle is an SCC with more than one member or a self-call.
    """
    comps = _sccs(graph)
    order = [n for comp in comps for n in comp]
    cycles = [c for c in comps if len(c) > 1 or c[0] in graph.get(c[0], [])]
    return order, cycles


@dataclass
class TreeIndex:
    """Result of :func:`index_tree`."""

    files: list[str]
    subroutines: list[SubroutineInfo]
    errors: list[dict[str, str]]
    call_graph: dict[str, list[str]]
    external_calls: dict[str, list[str]]
    order: list[str]
    cycles: list[list[str]]
    duplicates: dict[str, list[str]]

    def by_name(self) -> dict[str, SubroutineInfo]:
        out: dict[str, SubroutineInfo] = {}
        for s in self.subroutines:
            out.setdefault(s.name, s)
        return out

    def reachable(self, root: str) -> set[str]:
        root = root.upper()
        seen: set[str] = set()
        todo = [root]
        while todo:
            v = todo.pop()
            if v in seen or v not in self.call_graph:
                continue
            seen.add(v)
            todo.extend(self.call_graph[v])
        return seen

    def subtree_order(self, root: str) -> list[str]:
        """Leaves-first order of the indexed routines reachable from ``root`` (inclusive)."""
        keep = self.reachable(root)
        return [n for n in self.order if n in keep]

    def subtree_cycles(self, root: str) -> list[list[str]]:
        keep = self.reachable(root)
        return [c for c in self.cycles if set(c) & keep]

    def hidden_state(self, root: str) -> dict[str, list[str]]:
        """Saved variables read before assigned, per routine of the subtree of ``root``."""
        subs = self.by_name()
        return {n: subs[n].state_vars for n in self.subtree_order(root) if subs[n].state_vars}

    def to_dict(self, roots: Sequence[str] = ()) -> dict[str, Any]:
        return {
            "files": list(self.files),
            "n_subroutines": len(self.subroutines),
            "errors": list(self.errors),
            "duplicates": dict(self.duplicates),
            "cycles": [list(c) for c in self.cycles],
            "topological_order": list(self.order),
            "call_graph": {k: list(v) for k, v in self.call_graph.items()},
            "external_calls": {k: list(v) for k, v in self.external_calls.items() if v},
            "subtrees": {
                r.upper(): {
                    "order": self.subtree_order(r),
                    "cycles": self.subtree_cycles(r),
                    "hidden_state": self.hidden_state(r),
                }
                for r in roots
            },
            "subroutines": [s.to_dict() for s in self.subroutines],
        }


def index_tree(
    paths: Iterable[str | Path], *, free: bool | None = None, include_dirs: Sequence[str] = ()
) -> TreeIndex:
    """Index many files, build the call graph, a leaves-first order and refined intents.

    Parse failures are caught per file and reported in ``errors``. Intents are propagated
    bottom-up: each routine is re-analysed with the intents of the (indexed) routines it calls,
    iterating inside call cycles until stable.
    """
    files = [str(p) for p in paths]
    subs: list[SubroutineInfo] = []
    errors: list[dict[str, str]] = []
    for f in files:
        try:
            subs.extend(index_file(f, free=free, include_dirs=include_dirs))
        except Exception as e:  # per-file robustness is the point here
            msg = " ".join(str(e).split())[:500]
            errors.append({"file": f, "error": f"{type(e).__name__}: {msg}"})

    by_name: dict[str, SubroutineInfo] = {}
    seen_in: dict[str, list[str]] = {}
    for s in subs:
        seen_in.setdefault(s.name, []).append(f"{s.file}:{s.line_start}")
        by_name.setdefault(s.name, s)
    duplicates = {k: v for k, v in seen_in.items() if len(v) > 1}

    graph: dict[str, list[str]] = {}
    external: dict[str, list[str]] = {}
    for name, s in by_name.items():
        refs = set(s.calls) | {f for f in s.function_refs if f in by_name and "function" in by_name[f].kind}
        graph[name] = sorted(r for r in refs if r in by_name)
        external[name] = sorted(c for c in s.calls if c not in by_name)
    order, cycles = topological_order(graph)

    def sig(callee: str) -> tuple[list[str], dict[str, str]] | None:
        t = by_name.get(callee)
        return (t.args, t.intent_guess) if t is not None else None

    for comp in _sccs(graph):
        for _ in range(len(comp) + 2):
            before = [dict(by_name[n].intent_guess) for n in comp]
            for n in comp:
                by_name[n].analyse(sig)
            if before == [by_name[n].intent_guess for n in comp]:
                break
    for s in subs:  # duplicates that are not the canonical entry
        if by_name[s.name] is not s:
            s.analyse(sig)
    return TreeIndex(files, subs, errors, graph, external, order, cycles, duplicates)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agrijax.port.fortran_index", description=(__doc__ or "").split("\n")[0]
    )
    ap.add_argument("files", nargs="+", help="Fortran source files")
    ap.add_argument("-o", "--output", required=True, help="output JSON path")
    ap.add_argument("--root", action="append", default=[], help="report the call subtree of this routine")
    ap.add_argument("--free", action="store_true", default=None, help="force free-form parsing")
    ap.add_argument("-I", "--include", action="append", default=[], help="include directory")
    ap.add_argument("-q", "--quiet", action="store_true")
    ns = ap.parse_args(argv)

    idx = index_tree(ns.files, free=ns.free, include_dirs=ns.include)
    out = Path(ns.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(idx.to_dict(ns.root), indent=1) + "\n")
    if not ns.quiet:
        print(f"indexed {len(idx.subroutines)} units from {len(idx.files)} files -> {out}")
        for e in idx.errors:
            print(f"PARSE FAILURE {e['file']}: {e['error']}")
        if idx.cycles:
            print(f"cycles: {idx.cycles}")
        for r in ns.root:
            print(f"[{r.upper()}] order: {' '.join(idx.subtree_order(r))}")
            for n, v in idx.hidden_state(r).items():
                print(f"    {n}: {' '.join(v)}")
    return 1 if idx.errors and not idx.subroutines else 0


if __name__ == "__main__":
    raise SystemExit(main())
