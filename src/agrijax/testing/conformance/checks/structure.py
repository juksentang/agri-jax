"""Structure checks: declared dims and shapes, units and port records, the slot contract."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from agrijax.core.dims import DimsError, check_tree_dims
from agrijax.core.ports import bind, port_names
from agrijax.core.process import (
    Process,
)
from agrijax.core.units import UnitError, parse_unit
from agrijax.iface.contract import DSSAT_PORTS, PORTS, PortSpec

from ..case import ConformanceCase
from ..contracts import slot_contract
from ._common import (
    _day,
    _fail,
    _inputs,
    _is_array_leaf,
    _jit_run,
    _main_dtype,
    _paths_leaves,
    _under,
)


# ------------------------------------------------------------------------------------ shapes and units
def _walk_fields(node: Any, prefix: str, visit: Callable[[str, dataclasses.Field[Any], Any], None]) -> None:
    """Call ``visit(path, field, value)`` for every non-static field of every dataclass in ``node``."""
    if node is None:
        return
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        for f in dataclasses.fields(node):
            if f.metadata.get("static", False):
                continue
            value = getattr(node, f.name, None)
            path = f"{prefix}.{f.name}" if prefix else f.name
            if value is None:
                continue
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                _walk_fields(value, path, visit)
            elif isinstance(value, Mapping):
                for k, v in value.items():
                    _walk_fields(v, f"{path}.{k}", visit)
            elif isinstance(value, (list, tuple)) and not _is_array_leaf(value):
                for i, v in enumerate(value):
                    if dataclasses.is_dataclass(v):
                        _walk_fields(v, f"{path}.{i}", visit)
                    else:
                        visit(f"{path}.{i}", f, v)
            else:
                visit(path, f, value)
    elif isinstance(node, Mapping):
        for k, v in node.items():
            _walk_fields(v, f"{prefix}.{k}" if prefix else str(k), visit)


def _is_static_field(f: dataclasses.Field[Any]) -> bool:
    return bool(f.metadata.get("static", False))


def _step_problems(proc: Process, s0: Any, p: Any, f: Any, after: Any = None) -> list[str]:
    """Tree structure, shape or dtype changes between ``s0`` and one jitted step (or ``after``)."""
    if after is None:
        after = jax.jit(lambda s, pp, ff: proc(s, pp, ff))(s0, p, _day(f, 0))
    if jtu.tree_structure(after) != jtu.tree_structure(s0):
        return ["the state's tree structure changed"]
    out = []
    for (path, a), b in zip(_paths_leaves(s0), jtu.tree_leaves(after), strict=True):
        if np.shape(a) != np.shape(b):
            out.append(f"{path}: shape {np.shape(a)} -> {np.shape(b)}")
        elif jnp.result_type(a) != jnp.result_type(b):
            out.append(f"{path}: {jnp.result_type(a)} in, {jnp.result_type(b)} out")
    return out


def check_shapes_dims(case: ConformanceCase) -> None:
    """Declared dims match the shapes, every array leaf declares its dims, and ``n_days`` days keep
    the state's tree structure, shapes and dtypes."""
    name = "shapes_dims"
    dtype = _main_dtype()
    for variant in case.variants:
        s0, p, f = _inputs(case, dtype, variant)
        try:
            check_tree_dims({"params": p, "forcing": f, "state": s0})
        except DimsError as e:
            raise _fail(case, name, f"variant {variant!r}: {e}") from None
        undeclared: list[str] = []

        def visit(path: str, fld: dataclasses.Field[Any], value: Any, _u: list[str] = undeclared) -> None:
            if _is_array_leaf(value) and fld.metadata.get("dims") is None:
                _u.append(path)

        for root, tree in (("state", s0), ("params", p), ("forcing", f)):
            _walk_fields(tree, root, visit)
        if undeclared:
            raise _fail(case, name, f"variant {variant!r}: array leaves without declared dims: {undeclared}")
        leaves = jtu.tree_leaves(f)
        if leaves and any(np.shape(x)[:1] != (case.n_days,) for x in leaves):
            raise _fail(case, name, f"variant {variant!r}: forcing leaves need a time axis of {case.n_days}")
        problems = _step_problems(case.proc, s0, p, f)
        if problems:
            raise _fail(case, name, f"variant {variant!r}: " + "; ".join(problems))
        final, _ = _jit_run(case.proc, case.n_days)(p, f, s0)
        problems = _step_problems(case.proc, s0, p, f, after=final)
        if problems:
            raise _fail(case, name, f"variant {variant!r}, after {case.n_days} days: " + "; ".join(problems))


def _contract_port(target: str, slot: str, *, dssat: bool = False) -> PortSpec | None:
    # the ports of the RZWQM2 4.6 day first, then those of the DSSAT-CSM day (DSSAT_PORTS: its own
    # records PD1-PD7 and the DSSAT views of P1, P4-P7, P9; a DSSAT-day process binds those); for a
    # process of the DSSAT-CSM reference (``dssat=True``) the DSSAT day's view of a port comes first
    specs = (*PORTS.values(), *DSSAT_PORTS.values())
    if dssat:
        specs = (*DSSAT_PORTS.values(), *PORTS.values())
    for spec in specs:
        try:
            path = spec.global_path(slot)
        except ValueError:
            continue
        if spec.kind == "state" and path == target:
            return spec
    return None


def _is_dssat_case(case: ConformanceCase) -> bool:
    """A process of the DSSAT-CSM reference (registry key ``...@dssat-<version>:...``)."""
    return "@dssat-" in case.key


def _record_vs_spec(cls: type, spec: PortSpec) -> list[str]:
    """Field metadata of record class ``cls`` against the contract's fields (unit strings equal,
    dims equal, grid equal when the contract fixes one)."""
    meta = {f.name: f.metadata for f in dataclasses.fields(cls)}
    out = []
    for fname, fs in spec.fields:
        m = meta.get(fname)
        if m is None:
            out.append(f"{spec.id}.{fname}: missing in {cls.__name__}")
            continue
        if m.get("unit") != fs.unit:
            out.append(f"{spec.id}.{fname}: unit {m.get('unit')!r}, contract {fs.unit!r}")
        else:
            try:
                if not parse_unit(m["unit"]).same_dimension(parse_unit(fs.unit)):
                    out.append(f"{spec.id}.{fname}: dimension differs from {fs.unit!r}")
            except UnitError as e:
                out.append(f"{spec.id}.{fname}: {e}")
        if m.get("dims") != fs.dims:
            out.append(f"{spec.id}.{fname}: dims {m.get('dims')!r}, contract {fs.dims!r}")
        if fs.grid is not None and m.get("grid") != fs.grid:
            out.append(f"{spec.id}.{fname}: grid {m.get('grid')!r}, contract {fs.grid!r}")
    return out


def check_units(case: ConformanceCase) -> None:
    """Every array field declares a unit that parses; every bound port's record matches the
    contract field by field (unit string, dimension, dims, grid)."""
    name = "units"
    s0, p, f = _inputs(case, _main_dtype())
    problems: list[str] = []

    def visit(path: str, fld: dataclasses.Field[Any], value: Any) -> None:
        if not _is_array_leaf(value):
            return
        unit = fld.metadata.get("unit", "")
        if not str(unit).strip():
            problems.append(f"{path}: no unit")
            return
        try:
            parse_unit(unit)
        except UnitError as e:
            problems.append(f"{path}: {e}")

    for root, tree in (("state", s0), ("params", p), ("forcing", f)):
        _walk_fields(tree, root, visit)
    for port_field, target in case.port_map.items():
        spec = _contract_port(target, case.crop_slot, dssat=_is_dssat_case(case))
        if spec is None:
            problems.append(f"port {port_field} -> {target}: not a port of the coupling contract")
            continue
        value = getattr(s0, port_field, None)
        if value is None:
            problems.append(f"port {port_field}: the case's state0 leaves it empty")
            continue
        if spec.record is not None:
            problems.extend(_record_vs_spec(type(value), spec))
    if problems:
        raise _fail(case, name, "; ".join(problems))


# ------------------------------------------------------------------------------------ slot contract


def check_slot_contract(case: ConformanceCase) -> None:
    """Bound reads and writes stay inside the own subtree and the slot's contract ports; ``in``
    ports are never written; every bound ``out`` port is written; the case binds exactly the ports
    the process uses (so an ``out`` port cannot be left unchecked by leaving it bound but unused);
    each bound port has the contract's path and record class."""
    name = "slot_contract"
    cname = case.contract_name
    if cname is None:
        return
    contract = slot_contract(cname)
    proc = case.proc
    s0, _, _ = _inputs(case, _main_dtype())
    cls = type(s0)
    ports = port_names(cls)
    pmap = case.port_map
    problems: list[str] = []
    unknown = sorted(set(pmap) - set(ports))
    if unknown:
        raise _fail(case, name, f"{cls.__name__} has no port fields {unknown}")
    used = {path.split(".", 1)[0] for path in (*proc.reads, *proc.writes)}
    for path in (*proc.reads, *proc.writes):
        head = path.split(".", 1)[0]
        if head in ports and head not in pmap:
            problems.append(f"{proc.name} uses port {head!r} ({path}) but the case binds no global path")
        if path == "*":
            problems.append(f"{proc.name} declares the wildcard {path!r}; a slot process names its paths")
    for port_field in sorted(set(pmap) - used):
        problems.append(
            f"the case binds port {port_field!r} that {proc.name} neither reads nor writes "
            "(bind only the ports the process uses)"
        )
    allowed: dict[str, Any] = {}
    for sp in contract.ports:
        try:
            allowed[sp.spec.global_path(case.crop_slot)] = sp
        except ValueError:
            continue
    for port_field, target in pmap.items():
        sp = allowed.get(target)
        if sp is None:
            problems.append(f"port {port_field} -> {target}: not a port of slot {cname!r} {sorted(allowed)}")
            continue
        value = getattr(s0, port_field, None)
        if sp.spec.record is not None and value is not None and type(value) is not sp.spec.record:
            problems.append(
                f"port {port_field}: record {type(value).__name__}, the contract's {sp.port} is "
                f"{sp.spec.record.__name__}"
            )
    if problems:
        raise _fail(case, name, "; ".join(problems))
    bound = bind(proc, own=case.own_path, ports=pmap)
    own = case.own_path
    for r in bound.reads:
        if not (_under(r, own) or any(_under(r, t) for t in allowed)):
            problems.append(f"reads {r}: outside {own} and the ports of slot {cname!r}")
    for w in bound.writes:
        if _under(w, own):
            continue
        hit = [(t, sp) for t, sp in allowed.items() if _under(w, t) or _under(t, w)]
        if not hit:
            problems.append(f"writes {w}: outside {own} and the ports of slot {cname!r}")
            continue
        for t, sp in hit:
            rest = w[len(t) + 1 :] if w.startswith(t + ".") else None
            fname = rest.split(".", 1)[0] if rest else None
            if not sp.writable(fname):
                what = "an 'in' port" if sp.direction == "in" else f"field {fname or '<record>'} of {sp.port}"
                problems.append(f"writes {w}: {what} (slot {cname!r} may write {sp.writes or 'all fields'})")
    for port_field, target in pmap.items():
        sp = allowed.get(target)
        if sp is None or sp.direction != "out":
            continue
        if not any(_under(w, target) or _under(target, w) for w in bound.writes):
            problems.append(
                f"port {port_field} -> {target} is an 'out' port of slot {cname!r} ({sp.port}) but "
                f"{proc.name} never writes it"
            )
    if problems:
        raise _fail(case, name, "; ".join(problems))
