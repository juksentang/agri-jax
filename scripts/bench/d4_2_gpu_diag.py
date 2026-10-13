"""Where the steady GPU time of the free-run DSSAT day goes.

Measurement only. The task and the compiled programs are those of ``d4_jax_scaling.py`` (the
65-run mix at ``B`` seasons, one program per (layer count, MESEV) group, f64). For each group:

* compile (excluded), one warm-up call, the median of ``--repeats`` steady calls
  (``block_until_ready``, inputs on the device);
* ``jax.profiler`` trace of one further steady call (Perfetto JSON), parsed here: device kernels
  per simulated day, device busy time (union over streams) against the traced span (idle
  fraction), memcpy count, the kernel-name totals, and the per-iteration time (period of the loop
  body, from the kernel that runs once per day);
* the compiled HLO: the instructions of the day loop's body (static kernel count per day, the
  bytes their results write) and ``cost_analysis`` of the program;
* the bytes that one day must at least move (the scan carry read and written, the day's forcing
  gathered, the outputs written), to compare with the kernel time.

``where``: a static estimate of what the ``jnp.where`` / ``lax.select_n`` sites of one day step cost
because both branches are evaluated. For every ``select_n`` of the step's jaxpr (per sample,
nested jaxprs followed, ``scan`` bodies times their length), the exclusive cone of each case
operand (the equations whose results feed only that case) is costed in output elements
(transcendentals counted apart); the selects are grouped by their first ``src/agrijax`` source
frames. The cost of the cheaper case is a lower bound of the work thrown away per call; XLA may
fuse or simplify parts of it, so this is an attribution, not a timing.

Usage::

    python scripts/bench/d4_2_gpu_diag.py profile --B 10000 100000 --tag gpufull
    python scripts/bench/d4_2_gpu_diag.py where --tag where
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
OUT = DATA / "validation" / os.environ.get("AJ_DBENCH_DIR", "aj_dbench_d42")
INPUTS = DATA / "validation" / "aj_dbench_d41" / "inputs" / "d4_inputs.pkl"
TRACES = DATA / "d42_traces"


def _groups(items: list[dict[str, Any]], B: int) -> dict[tuple[int, str], int]:
    """The ``--group mix`` allocation of ``d4_jax_scaling.worker`` (largest remainder)."""
    groups: dict[tuple[int, str], int] = Counter((it["nl"], it["mesev"]) for it in items)
    n_all = len(items)
    exact = {k: B * groups[k] / n_all for k in sorted(groups)}
    alloc = {k: int(exact[k]) for k in exact}
    for k in sorted(exact, key=lambda k: -(exact[k] - alloc[k]))[: B - sum(alloc.values())]:
        alloc[k] += 1
    return {k: v for k, v in alloc.items() if v > 0}


# ============================================================================ trace parsing
def _load_trace(d: Path) -> dict[str, Any]:
    fs = sorted(d.rglob("perfetto_trace.json.gz"), key=lambda f: f.stat().st_mtime)
    if not fs:
        raise FileNotFoundError(f"no trace under {d}: {[str(x) for x in d.rglob('*')][:20]}")
    with gzip.open(fs[-1], "rt") as fh:
        return json.load(fh)


def _union(iv: list[tuple[float, float]]) -> float:
    tot, end = 0.0, -1e300
    for s, e in sorted(iv):
        if e <= end:
            continue
        tot += e - max(s, end)
        end = e
    return tot


def parse_trace(tr: dict[str, Any], n_days: int) -> dict[str, Any]:
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    pname: dict[Any, str] = {}
    tname: dict[tuple[Any, Any], str] = {}
    for e in ev:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pname[e["pid"]] = e.get("args", {}).get("name", "")
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tname[(e["pid"], e["tid"])] = e.get("args", {}).get("name", "")
    threads: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "dur_us": 0.0})
    stream: list[dict[str, Any]] = []
    for e in ev:
        if e.get("ph") != "X":
            continue
        p = pname.get(e["pid"], str(e["pid"]))
        t = tname.get((e["pid"], e["tid"]), str(e["tid"]))
        k = f"{p} | {t}"
        threads[k]["n"] += 1
        threads[k]["dur_us"] += float(e.get("dur", 0))
        if ("GPU" in p or "gpu" in p) and "Stream" in t:
            stream.append(e)
    out: dict[str, Any] = {"threads": dict(threads), "n_days": n_days}
    if not stream:
        return out
    stream.sort(key=lambda e: float(e["ts"]))
    is_copy = [bool(re.search(r"memcpy|memset", e["name"], re.I)) for e in stream]
    kern = [e for e, c in zip(stream, is_copy, strict=True) if not c]
    copies = [e for e, c in zip(stream, is_copy, strict=True) if c]
    iv = [(float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in stream]
    span = max(b for _, b in iv) - min(a for a, _ in iv)
    busy = _union(iv)
    kbusy = _union([(float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in kern])
    durs = sorted(float(e.get("dur", 0)) for e in kern)
    names: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "dur_us": 0.0})
    for e in kern:
        names[e["name"]]["n"] += 1
        names[e["name"]]["dur_us"] += float(e.get("dur", 0))
    top = sorted(names.items(), key=lambda kv: -kv[1]["dur_us"])
    cnames: Counter[str] = Counter(e["name"] for e in copies)
    # loop period: the kernel names that run once per day; successive starts of the first one
    per_day = [n for n, v in names.items() if abs(v["n"] - n_days) <= 1]
    it_stats: dict[str, Any] = {}
    if per_day:
        marker = min(per_day, key=lambda n: next(float(e["ts"]) for e in kern if e["name"] == n))
        ts = [float(e["ts"]) for e in kern if e["name"] == marker]
        dt = sorted(b - a for a, b in itertools.pairwise(ts))
        if dt:
            q = lambda f: dt[min(len(dt) - 1, int(f * len(dt)))]  # noqa: E731
            it_stats = {
                "marker": marker,
                "iterations": len(ts),
                "p50_us": q(0.5),
                "p90_us": q(0.9),
                "p99_us": q(0.99),
                "max_us": dt[-1],
                "min_us": dt[0],
                "sum_us": sum(dt),
                "top1pct_share": sum(dt[-max(1, len(dt) // 100) :]) / sum(dt),
            }
    gaps = []
    for (_, a1), (b0, _) in itertools.pairwise(sorted(iv)):
        gaps.append(max(0.0, b0 - a1))
    gaps.sort()
    out |= {
        "stream_events": len(stream),
        "kernels": len(kern),
        "copies": len(copies),
        "copy_names": dict(cnames.most_common(8)),
        "copy_dur_us": sum(float(e.get("dur", 0)) for e in copies),
        "kernels_per_day": len(kern) / n_days,
        "copies_per_day": len(copies) / n_days,
        "span_us": span,
        "busy_us": busy,
        "kernel_busy_us": kbusy,
        "idle_fraction": 1 - busy / span if span > 0 else None,
        "kernel_dur_us_p50": durs[len(durs) // 2],
        "kernel_dur_us_p90": durs[int(0.9 * len(durs))],
        "kernel_dur_us_max": durs[-1],
        "kernel_dur_us_mean": sum(durs) / len(durs),
        "gap_us_p50": gaps[len(gaps) // 2] if gaps else None,
        "gap_us_mean": sum(gaps) / len(gaps) if gaps else None,
        "per_day_us": span / n_days,
        "top_kernels": [
            {"name": n[:120], "n": v["n"], "dur_us": v["dur_us"], "share": v["dur_us"] / max(kbusy, 1e-9)}
            for n, v in top[:25]
        ],
        "top5_share": sum(v["dur_us"] for _, v in top[:5]) / max(sum(v["dur_us"] for _, v in top), 1e-9),
        "distinct_kernels": len(names),
        "iteration": it_stats,
    }
    return out


# ============================================================================ HLO body
_SHAPE = re.compile(r"\b(pred|s8|s16|s32|s64|u8|u32|u64|f16|bf16|f32|f64)\[([0-9,]*)\]")
_BYTES = {"pred": 1, "s8": 1, "u8": 1, "s16": 2, "f16": 2, "bf16": 2, "s32": 4, "u32": 4, "f32": 4}
_FREE = ("parameter(", "get-tuple-element(", "tuple(", "constant(", "bitcast(", "after-all(", "partition-id(")


def _shape_bytes(s: str) -> int:
    tot = 0
    for dt, dims in _SHAPE.findall(s):
        n = 1
        for d in filter(None, dims.split(",")):
            n *= int(d)
        tot += n * _BYTES.get(dt, 8)
    return tot


def hlo_body(text: str) -> dict[str, Any]:
    """The largest ``while`` body of the compiled module: its top-level instructions (op kinds,
    the bytes their results occupy), nested while loops with their known trip counts."""
    comps: dict[str, list[str]] = {}
    cur = None
    for ln in text.splitlines():
        m = re.match(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s.*\{\s*$", ln)
        if m and not ln.startswith(" "):
            cur = m.group(1)
            comps[cur] = []
            continue
        if ln.strip() == "}":
            cur = None
            continue
        if cur is not None and "=" in ln:
            comps[cur].append(ln)
    whiles = []
    for c, lines in comps.items():
        for ln in lines:
            if " while(" in ln:
                b = re.search(r"body=%?([\w.\-]+)", ln)
                n = re.search(r'"n":"?(\d+)', ln)
                whiles.append(
                    {"in": c, "body": b.group(1) if b else None, "trip": int(n.group(1)) if n else None}
                )

    def summary(name: str) -> dict[str, Any]:
        kinds: Counter[str] = Counter()
        nbytes = 0
        for ln in comps.get(name, []):
            rhs = ln.split("=", 1)[1]
            if any(f in rhs for f in _FREE) and "fusion(" not in rhs:
                continue
            op = re.search(r"\s([a-z][\w\-]*)\(", rhs)
            k = op.group(1) if op else "?"
            if k == "fusion":
                kk = re.search(r"kind=(k\w+)", rhs)
                k = f"fusion/{kk.group(1) if kk else '?'}"
            kinds[k] += 1
            nbytes += _shape_bytes(
                rhs.split(" ", 2)[1] if rhs.strip().startswith("(") is False else rhs.split(")")[0]
            )
        return {"ops": sum(kinds.values()), "kinds": dict(kinds.most_common()), "result_bytes": nbytes}

    bodies = [w for w in whiles if w["body"] in comps]
    if not bodies:
        return {"whiles": whiles}
    main = max(bodies, key=lambda w: len(comps[w["body"]]))
    return {"whiles": whiles, "main_body": main, "main": summary(main["body"])} | {
        "nested": [w | summary(w["body"]) for w in bodies if w["in"] == main["body"]]
    }


# ============================================================================ profile
def _simulator(D: Any, model: Any, n_days: int, unroll: int) -> Any:
    """``d4_jax_day.simulator`` with ``lax.scan(..., unroll=unroll)`` (a probe of the loop organisation)."""
    import jax
    import jax.numpy as jnp

    step = jax.vmap(model.compile(), in_axes=(0, 0, 0))

    def sim(params_tr: Any, forcing_tr: Any, state_tr: Any, tid: Any, mult: Any) -> Any:
        params = D.perturb(jax.tree.map(lambda x: x[tid], params_tr), mult)
        state0 = jax.tree.map(lambda x: x[tid], state_tr)

        def body(s: Any, t: Any) -> Any:
            f_t = D.perturb_day_soil(jax.tree.map(lambda x: x[tid, t], forcing_tr), mult)
            return step(s, params, f_t)

        _, outs = jax.lax.scan(body, state0, jnp.arange(n_days), unroll=unroll)
        return outs

    return sim


def profile(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    import numpy as np

    sys.path.insert(0, str(HERE))
    import d4_jax_day as D

    # the layer-loop unrolling is the execution setting ``depth_unroll`` (agrijax.core.execution);
    # ``--depth-unroll on|off`` overrides the backend default through AGRI_JAX_DEPTH_UNROLL
    if a.depth_unroll != "auto":
        os.environ["AGRI_JAX_DEPTH_UNROLL"] = "1" if a.depth_unroll == "on" else "0"
    from agrijax.core.execution import resolve_depth_unroll

    items = D.load(INPUTS)
    by: dict[tuple[int, str], list[Any]] = {}
    for it in items:
        by.setdefault((it["nl"], it["mesev"]), []).append(it)
    dev = jax.devices()[0]
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {
        "device": dev.device_kind,
        "backend": jax.default_backend(),
        "jax": jax.__version__,
    }
    res |= {"unroll": a.unroll, "depth_unroll": resolve_depth_unroll()}
    res |= {"xla_flags": os.environ.get("XLA_FLAGS", "")}
    res |= {"host": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "runs": []}
    for B in a.B:
        for key, Bg in _groups(items, B).items():
            g = D.group_inputs(by[key], np.float64)
            n_tr = len(g["keys"])
            rng = np.random.default_rng(0)
            mult = rng.uniform(1 - D.SPREAD, 1 + D.SPREAD, size=(Bg, len(D.PERTURBED)))
            tid = (np.arange(Bg) % n_tr).astype(np.int32)
            bd = D.bench_model(key[1])
            sim = (
                D.simulator(bd.model, g["n_days"])
                if a.unroll == 1
                else _simulator(D, bd.model, g["n_days"], a.unroll)
            )
            dargs = jax.device_put((g["params"], g["forcing"], g["state"], tid, mult))
            fn = jax.jit(sim)
            compiled = fn.lower(*dargs).compile()
            o = compiled(*dargs)
            gw = np.asarray(o["gwad"])[-1]
            del o
            times = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(compiled(*dargs))
                times.append(time.perf_counter() - t0)
            name = f"nl{key[0]}_{key[1]}"
            rec: dict[str, Any] = {
                "B": B,
                "group": name,
                "B_group": Bg,
                "n_days": g["n_days"],
                "steady_s": float(np.median(times)),
                "steady_all_s": times,
                "gwad_final_sum": float(gw.sum()),
            }
            # bytes one day must move at least: the carry read + written, the day's forcing, outputs
            carry_b = int(sum(np.asarray(x[0]).nbytes for x in jax.tree.leaves(g["state"]))) * Bg
            f_day = int(sum(np.asarray(x[:, 0]).nbytes for x in jax.tree.leaves(g["forcing"]))) * Bg // n_tr
            rec["min_bytes_per_day"] = {"carry_rw": 2 * carry_b, "forcing": f_day, "outputs": 10 * 8 * Bg}
            try:
                ca = compiled.cost_analysis()
                ca = ca[0] if isinstance(ca, list) else ca
                rec["cost_analysis"] = {
                    k: float(v) for k, v in ca.items() if k in ("flops", "bytes accessed", "transcendentals")
                }
            except Exception as ex:
                rec["cost_analysis"] = str(ex)
            try:
                rec["hlo"] = hlo_body(compiled.as_text())
            except Exception as ex:
                rec["hlo"] = str(ex)
            if a.trace and jax.default_backend() == "gpu":
                td = TRACES / f"{a.tag}_{os.environ.get('SLURM_JOB_ID', os.getpid())}_B{B}_{name}"
                td.mkdir(parents=True, exist_ok=True)
                with jax.profiler.trace(str(td), create_perfetto_trace=True):
                    t0 = time.perf_counter()
                    jax.block_until_ready(compiled(*dargs))
                    rec["traced_call_s"] = time.perf_counter() - t0
                rec["trace"] = parse_trace(_load_trace(td), g["n_days"])
                rec["trace_dir"] = str(td)
            brief = {k: v for k, v in rec.items() if k not in ("hlo",)}
            if "trace" in brief:
                brief["trace"] = {
                    k: v for k, v in rec["trace"].items() if k not in ("top_kernels", "threads")
                }
            print(json.dumps(brief), flush=True)
            res["runs"].append(rec)
            del dargs, compiled
    (OUT / f"gpu_diag_{a.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT / f'gpu_diag_{a.tag}.json'}")


# ============================================================================ where
_ZERO = {
    "broadcast_in_dim", "reshape", "squeeze", "expand_dims", "convert_element_type", "stop_gradient",
    "copy", "copy_p", "transpose", "slice", "iota", "reduce_precision",
}  # fmt: skip
_TRANSC = {
    "exp", "exp2", "log", "log1p", "expm1", "pow", "tanh", "logistic", "sin", "cos", "tan", "atan2",
    "sqrt", "rsqrt", "cbrt", "erf", "erfc", "lgamma", "digamma", "asin", "acos", "atan", "sinh", "cosh",
}  # fmt: skip


def _numel(v: Any) -> int:
    import numpy as np

    return int(np.prod(getattr(v.aval, "shape", ()) or (1,)))


_LOOPS = {"scan", "while", "cond"}


def _call_jaxpr(eqn: Any) -> Any:
    """The inner jaxpr of a call-like equation (pjit, closed_call, custom_jvp/vjp, remat), else None."""
    from jax.extend import core as jc

    if eqn.primitive.name in _LOOPS:
        return None
    for k in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        j = eqn.params.get(k)
        if j is None:
            continue
        j = j.jaxpr if isinstance(j, jc.ClosedJaxpr) else j
        if len(j.invars) == len(eqn.invars) and len(j.outvars) == len(eqn.outvars):
            return j
    return None


def _loop_jaxprs(eqn: Any) -> list[tuple[Any, float]]:
    from jax.extend import core as jc

    p = eqn.params
    name = eqn.primitive.name
    if name == "scan":
        return [(p["jaxpr"].jaxpr, float(p["length"]))]
    if name == "while":
        return [(p["body_jaxpr"].jaxpr, 1.0), (p["cond_jaxpr"].jaxpr, 1.0)]
    if name == "cond":
        return [(b.jaxpr if isinstance(b, jc.ClosedJaxpr) else b, 1.0) for b in p["branches"]]
    return []


def _site(src: Any) -> str | None:
    from jax._src import source_info_util as si

    try:
        fr = [f for f in si.user_frames(src.traceback) if "/agrijax/" in f.file_name]
    except Exception:
        return None
    fr = [
        f for f in fr if "/core/grad.py" not in f.file_name and "/core/depth_scan.py" not in f.file_name
    ] or fr
    if not fr:
        return None
    f = fr[0]
    return f"{f.file_name.split('/src/')[-1]}:{f.start_line} ({f.function_name})"


def _is_library(j: Any) -> bool:
    di = getattr(j, "debug_info", None)
    src = str(getattr(di, "func_src_info", "") or "")
    return "/jax/" in src or "/equinox/" in src or not src


class WhereCost:
    """A flat data-flow graph of one jaxpr (call-like equations inlined; scan/while/cond bodies
    analysed as their own graphs, their cost times the trip count) and the exclusive cones of
    every ``select_n`` case."""

    def __init__(self) -> None:
        self.sites: dict[str, dict[str, float]] = defaultdict(
            lambda: {"selects": 0, "cone_min": 0.0, "cone_sum": 0.0, "transc_min": 0.0, "transc_sum": 0.0}
        )
        # deduplicated (nested selects' cones overlap): union of all case cones, union of the cheaper cones
        self.union_all = 0.0
        self.union_min = 0.0
        self.union_all_transc = 0.0

    def graph(self, j: Any, mult: float) -> tuple[float, float]:
        from jax.extend import core as jc

        nodes: list[dict[str, Any]] = []  # ins, outs (value ids), cost, transc, site, select
        counter = [0]

        def new() -> int:
            counter[0] += 1
            return counter[0]

        def emit(jx: Any, ins: list[int | None], ctx: str | None, lib: bool) -> list[int | None]:
            env: dict[Any, int | None] = {}
            for v, x in zip(jx.invars, ins, strict=True):
                env[v] = x
            for v in jx.constvars:
                env[v] = new()

            def rd(v: Any) -> int | None:
                return env.get(v) if isinstance(v, jc.Var) else None

            for e in jx.eqns:
                own = None if lib else _site(e.source_info)
                site = own or ctx
                inner = _call_jaxpr(e)
                if inner is not None:
                    outs = emit(inner, [rd(v) for v in e.invars], site, lib or _is_library(inner))
                    for v, o in zip(e.outvars, outs, strict=True):
                        env[v] = o if o is not None else new()
                    continue
                cost = tr = 0.0
                if e.primitive.name in _LOOPS:
                    for sub, m in _loop_jaxprs(e):
                        c, t = self.graph(sub, mult * m)
                        cost += c * m
                        tr += t * m
                elif e.primitive.name not in _ZERO:
                    cost = float(sum(_numel(v) for v in e.outvars))
                    tr = cost if e.primitive.name in _TRANSC else 0.0
                outs_ids = [new() for _ in e.outvars]
                for v, o in zip(e.outvars, outs_ids, strict=True):
                    env[v] = o
                nodes.append(
                    {
                        "ins": [rd(v) for v in e.invars],
                        "outs": outs_ids,
                        "cost": cost,
                        "tr": tr,
                        "site": site or "?",
                        "select": e.primitive.name == "select_n",
                    }
                )
            return [rd(v) for v in jx.outvars]

        top_ins: list[int | None] = [new() for _ in j.invars]
        outs = {o for o in emit(j, top_ins, None, False) if o is not None}
        producer: dict[int, int] = {}
        consumers: dict[int, set[int]] = defaultdict(set)
        for i, n in enumerate(nodes):
            for o in n["outs"]:
                producer[o] = i
            for x in n["ins"]:
                if x is not None:
                    consumers[x].add(i)
        u_all: set[int] = set()
        u_min: set[int] = set()
        for i, n in enumerate(nodes):
            if not n["select"]:
                continue
            cones = []
            sets = []
            for case in n["ins"][1:]:
                cone = self._cone(nodes, i, case, producer, consumers, outs)
                sets.append(cone)
                cones.append((sum(nodes[k]["cost"] for k in cone), sum(nodes[k]["tr"] for k in cone)))
            u_all.update(*sets)
            u_min.update(sets[min(range(len(sets)), key=lambda k: cones[k][0])])
            s = self.sites[n["site"]]
            s["selects"] += mult
            s["cone_min"] += mult * min(c for c, _ in cones)
            s["cone_sum"] += mult * sum(c for c, _ in cones)
            s["transc_min"] += mult * min(t for _, t in cones)
            s["transc_sum"] += mult * sum(t for _, t in cones)
        self.union_all += mult * sum(nodes[k]["cost"] for k in u_all)
        self.union_min += mult * sum(nodes[k]["cost"] for k in u_min)
        self.union_all_transc += mult * sum(nodes[k]["tr"] for k in u_all)
        return sum(n["cost"] for n in nodes), sum(n["tr"] for n in nodes)

    @staticmethod
    def _cone(nodes: list, i: int, case: int | None, producer: dict, consumers: dict, outs: set) -> set[int]:
        """Nodes whose results feed only case ``case`` of select ``i`` (and nothing else)."""
        if case is None or case in outs or case not in producer:
            return set()
        if sum(x == case for x in nodes[i]["ins"]) != 1 or consumers[case] != {i}:
            return set()
        root = producer[case]
        if any(o in outs or consumers[o] for o in nodes[root]["outs"] if o != case):
            return set()
        cone = {root}
        changed = True
        while changed:
            changed = False
            cands = {
                producer[x] for q in cone for x in nodes[q]["ins"] if x is not None and x in producer
            } - cone
            for q in cands:
                if all(o not in outs and consumers[o] <= cone for o in nodes[q]["outs"]):
                    cone.add(q)
                    changed = True
        return cone


def where(a: argparse.Namespace) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, str(HERE))
    import d4_jax_day as D

    items = D.load(INPUTS)
    by: dict[tuple[int, str], list[Any]] = {}
    for it in items:
        by.setdefault((it["nl"], it["mesev"]), []).append(it)
    res: dict[str, Any] = {"groups": {}}
    for key in sorted(by):
        it = by[key][0]
        bd = D.bench_model(key[1])
        step = bd.model.compile()
        f0 = jax.tree.map(lambda x: x[0], it["forcing"])
        cj = jax.make_jaxpr(step)(it["state"], it["params"], f0)
        wc = WhereCost()
        tot, tr = wc.graph(cj.jaxpr, 1.0)
        top = sorted(wc.sites.items(), key=lambda kv: -kv[1]["cone_min"])
        rec = {
            "day_elements": tot,
            "day_transcendentals": tr,
            "selects": sum(v["selects"] for v in wc.sites.values()),
            "cond_work_union": wc.union_all,
            "cond_work_union_min": wc.union_min,
            "cond_transc_union": wc.union_all_transc,
            "cone_min_total": sum(v["cone_min"] for v in wc.sites.values()),
            "cone_sum_total": sum(v["cone_sum"] for v in wc.sites.values()),
            "transc_min_total": sum(v["transc_min"] for v in wc.sites.values()),
            "top": [{"site": k, **v} for k, v in top[: a.top]],
            "top_by_sum": [
                {"site": k, **v}
                for k, v in sorted(wc.sites.items(), key=lambda kv: -kv[1]["cone_sum"])[: a.top]
            ],
        }
        name = f"nl{key[0]}_{key[1]}"
        res["groups"][name] = rec
        print(name, json.dumps({k: v for k, v in rec.items() if not k.startswith("top")}), flush=True)
        for r in rec["top"][:12]:
            print("   ", json.dumps(r), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"where_{a.tag}.json").write_text(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--B", type=int, nargs="+", default=[10000, 100000])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tag", required=True)
    p.add_argument("--no-trace", dest="trace", action="store_false")
    p.add_argument("--unroll", type=int, default=1, help="probe: scan unroll factor of the day loop")
    p.add_argument(
        "--depth-unroll",
        choices=("auto", "on", "off"),
        default="auto",
        help="depth_scan layer loops: the backend default (GPU unrolled), or forced on / off",
    )
    w = sub.add_parser("where")
    w.add_argument("--tag", default="where")
    w.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    {"profile": profile, "where": where}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
