"""JAX scaling of the free-run DSSAT day (``d4_jax_day.py``) over parameter samples.

Protocol: every configuration runs in a fresh Python process (no
compile cache, ``jax_enable_compilation_cache = False``), and reports separately

* ``h2d_s``: putting the inputs on the device(s), once (the only device copy of the inputs);
* ``lower_s`` and ``compile_s``: ``jax.jit(sim).lower(...)`` and ``.compile()``;
* ``first_call_s``: the first call of the compiled function (``block_until_ready``);
* ``to_host_s``: copying the first call's daily outputs to NumPy;
* ``steady_s``: the median of ``repeats`` further calls (inputs already on the device);
* ``e2e_s``: a single pass, from the driver starting the subprocess to the first call's outputs on
  the host (Python start, imports, unpickling the inputs, building the model, input transfer,
  lowering, compiling, one call, outputs on the host); the steady-state repeats and the bookkeeping
  after them are excluded (``excluded_s``; with several groups, each group's repeats are excluded);
  ``process_wall_s`` is the wall time of the whole subprocess including the repeats;
* ``peak_mem_bytes``: GPU ``peak_bytes_in_use`` (one input copy, the previous call's outputs freed
  before the next call); CPU the process ``ru_maxrss``.

Task: ``B`` season-runs. ``--group mix`` (the default task, the same season mix as the DSSAT
baseline): the 65 acceptance runs replicated, ``B`` split over the groups (layer count, MESEV) in
proportion to their run counts (largest remainder), sample ``b`` of a group running the group's run
``b mod n``; ``--group largest``: every sample in the largest group. Each group is
its own compiled program (days padded to the group's longest season). Every sample's cultivar P1,
P5, G2, G3, PHINT and soil CN, SWCON are multiplied by U(0.9, 1.1) factors (seed 0; ``--unperturbed``
all 1); outputs: ten daily series per run (``d4_jax_day.outputs``), returned to the host.

Devices: ``cpu1`` one core (``taskset`` to one core, ``--xla_cpu_multi_thread_eigen=false``);
``cpuN`` the allocated cores as N host devices (``--xla_force_host_platform_device_count``) with the
batch sharded over them (B padded to a multiple of N; the padding is reported); ``gpu`` one device.

Warm start (``warm``): the same worker with the persistent compilation cache on
(``--cache-dir``, a fresh directory per configuration; ``jax_persistent_cache_min_compile_time_secs
= 0`` unless ``--cache-min-compile-s`` is given, entry-size minimum 0), run twice in two fresh
processes at the same B (fixed batch shapes): process 1 = cold (empty cache, writes it), process 2
= warm (reads it). Hits and misses are counted from ``jax.monitoring`` events
(``/jax/compilation_cache/cache_hits``, ``cache_misses``, ``compile_requests_use_cache``).

Usage::

    python scripts/bench/d4_jax_scaling.py prepare --jobs 8
    python scripts/bench/d4_jax_scaling.py sweep --device cpu1 --precision f64 --B 1 10 100 1000
    python scripts/bench/d4_jax_scaling.py equal --device cpu1 --precision f64
    python scripts/bench/d4_jax_scaling.py warm --device cpuN --precision f64 --B 100 10000 --tag w
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

T_START = time.perf_counter()
HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
OUT = DATA / "validation" / os.environ.get("AJ_DBENCH_DIR", "aj_dbench_d41")
INPUTS = DATA / "validation" / "aj_dbench_d41" / "inputs" / "d4_inputs.pkl"


def cpu_model() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, check=False).stdout
        return next(
            (ln.split(":", 1)[1].strip() for ln in out.splitlines() if ln.startswith("Model name")), ""
        )
    except OSError:
        return ""


def node_info() -> dict[str, Any]:
    return {
        "host": platform.node(),
        "cpu_model": cpu_model(),
        "affinity_cores": len(os.sched_getaffinity(0)),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
    }


# ============================================================================ worker
def worker(a: argparse.Namespace) -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", a.precision == "f64")
    events: dict[str, float] = {}
    if a.cache_dir:
        from jax import monitoring

        def _ev(name: str, **_: Any) -> None:
            if name.startswith("/jax/compilation_cache/"):
                events[name] = events.get(name, 0) + 1

        def _dur(name: str, secs: float, **_: Any) -> None:
            if name.startswith("/jax/compilation_cache/"):
                events[name] = events.get(name, 0.0) + secs

        monitoring.register_event_listener(_ev)
        monitoring.register_event_duration_secs_listener(_dur)
        jax.config.update("jax_enable_compilation_cache", True)
        jax.config.update("jax_compilation_cache_dir", a.cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", a.cache_min_compile_s)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    else:
        jax.config.update("jax_enable_compilation_cache", False)
    import numpy as np

    sys.path.insert(0, str(HERE))
    import d4_jax_day as D

    # the layer-loop unrolling is the execution setting ``depth_unroll`` of agrijax.core.execution
    # (GPU: unrolled, CPU: loops); ``AGRI_JAX_DEPTH_UNROLL=0|1`` in the environment overrides it
    from agrijax.core.execution import resolve_depth_unroll

    t_imp = time.perf_counter()
    items = D.load(Path(a.inputs))
    dtype = np.float64 if a.precision == "f64" else np.float32
    groups: dict[tuple[int, str], list[Any]] = {}
    for it in items:
        groups.setdefault((it["nl"], it["mesev"]), []).append(it)
    n_all = len(items)
    alloc: dict[tuple[int, str], int] = {}
    if a.group in ("all", "mix"):
        chosen = sorted(groups)
        if a.group == "mix" and a.B is not None:
            B = int(a.B)
            exact = {k: B * len(groups[k]) / n_all for k in chosen}
            alloc = {k: int(exact[k]) for k in chosen}
            rest = B - sum(alloc.values())
            for k in sorted(chosen, key=lambda k: -(exact[k] - alloc[k]))[:rest]:
                alloc[k] += 1
            chosen = [k for k in chosen if alloc[k] > 0]
    elif a.group == "largest":
        chosen = [max(groups, key=lambda k: (len(groups[k]), -k[0]))]
    else:
        nl, m = a.group.split("_")
        chosen = [(int(nl.removeprefix("nl")), m)]
    t_load = time.perf_counter()
    devs = jax.devices()
    ndev = len(devs)
    res: dict[str, Any] = {
        "device": a.device,
        "precision": a.precision,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "n_devices": ndev,
        "device_kind": devs[0].device_kind,
        "import_s": t_imp - T_START,
        "load_inputs_s": t_load - t_imp,
        "group_mode": a.group,
        "B_total": a.B,
        "groups": [],
        **node_info(),
    }
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as PS

    mesh = Mesh(np.asarray(devs), ("b",))
    rep = NamedSharding(mesh, PS())
    bsh = NamedSharding(mesh, PS("b"))
    osh = NamedSharding(mesh, PS(None, "b"))
    excluded = 0.0
    e2e_single = None
    for key in chosen:
        g = D.group_inputs(groups[key], dtype)
        n_tr = len(g["keys"])
        B = n_tr if a.B is None else alloc.get(key, int(a.B))
        Bp = -(-B // ndev) * ndev
        rng = np.random.default_rng(0)
        if a.unperturbed:
            mult = np.ones((Bp, len(D.PERTURBED)))
        else:
            mult = rng.uniform(1 - D.SPREAD, 1 + D.SPREAD, size=(Bp, len(D.PERTURBED)))
        tid = (np.arange(Bp) % n_tr).astype(np.int32)
        mult = mult.astype(dtype)
        t0 = time.perf_counter()
        bd = D.bench_model(key[1])
        sim = D.simulator(bd.model, g["n_days"])
        t1 = time.perf_counter()
        args = (g["params"], g["forcing"], g["state"], tid, mult)
        # the inputs on the device once; the compile and every call use this one copy
        if ndev > 1:
            dargs = tuple(jax.device_put(x, s) for x, s in zip(args, (rep, rep, rep, bsh, bsh), strict=True))
            fn = jax.jit(sim, in_shardings=(rep, rep, rep, bsh, bsh), out_shardings=osh)
        else:
            dargs = jax.device_put(args)
            fn = jax.jit(sim)
        jax.block_until_ready(dargs)
        t_h2d = time.perf_counter()
        lowered = fn.lower(*dargs)
        t2 = time.perf_counter()
        compiled = lowered.compile()
        t3 = time.perf_counter()
        out = compiled(*dargs)
        jax.block_until_ready(out)
        t4 = time.perf_counter()
        host = {k: np.asarray(v) for k, v in out.items()}
        t5 = time.perf_counter()
        del out
        # single pass done: outputs of one call on the host
        e2e_single = time.time() - a.launch_epoch - excluded
        first = t4 - t3
        to_host = t5 - t4
        reps = a.repeats if first < a.long_call_s else 1
        times = []
        for _ in range(reps):
            s0 = time.perf_counter()
            o = compiled(*dargs)
            jax.block_until_ready(o)
            times.append(time.perf_counter() - s0)
            del o
        steady = float(np.median(times))
        gw = host["gwad"][-1, :B]
        # per-run harvest yield and dates of the unperturbed task (the acceptance quantities)
        sanity = None
        if a.unperturbed and B == n_tr:
            sanity = [
                D.run_record(it, host["gwad"][:, i], host["istage"][:, i]) for i, it in enumerate(groups[key])
            ]
        mem = None
        try:
            ms = devs[0].memory_stats()
            if ms:
                mem = int(ms.get("peak_bytes_in_use", 0))
        except Exception:
            mem = None
        run_days = B * g["n_days"]
        res["groups"].append(
            {
                "group": f"nl{key[0]}_{key[1]}",
                "n_treatments": n_tr,
                "B": B,
                "B_padded": Bp,
                "n_days_padded": g["n_days"],
                "real_days_mean": float(np.mean(g["real_days"])),
                "real_days_of_samples": int(sum(g["real_days"][int(i)] for i in tid[:B])),
                "run_days": run_days,
                "lag_check": bd.lag_check,
                "build_s": t1 - t0,
                "h2d_s": t_h2d - t1,
                "lower_s": t2 - t_h2d,
                "compile_s": t3 - t2,
                "first_call_s": first,
                "steady_s": steady,
                "steady_all_s": times,
                "to_host_s": to_host,
                "steady_per_season_run_s": steady / B,
                "steady_per_1e5_s": steady / B * 1e5,
                "out_bytes": int(sum(v.nbytes for v in host.values())),
                "peak_device_bytes": mem,
                "gwad_final_mean": float(np.mean(gw)),
                "gwad_final_finite": int(np.isfinite(gw).sum()),
                "sanity": sanity,
            }
        )
        del dargs, compiled, lowered
        excluded += time.perf_counter() - t5
    res["depth_unroll"] = resolve_depth_unroll()
    res["cache_dir"] = a.cache_dir
    res["cache_min_compile_s"] = a.cache_min_compile_s if a.cache_dir else None
    res["cache_events"] = events
    if a.cache_dir and Path(a.cache_dir).is_dir():
        fs = [f for f in Path(a.cache_dir).rglob("*") if f.is_file()]
        res["cache_files"] = len(fs)
        res["cache_bytes"] = int(sum(f.stat().st_size for f in fs))
    res["e2e_def"] = "single_pass"
    res["e2e_single_pass_s"] = e2e_single
    res["excluded_s"] = excluded
    res["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    res["worker_wall_s"] = time.perf_counter() - T_START
    return res


# ============================================================================ driver
def launch(
    device: str,
    precision: str,
    B: int | None,
    group: str,
    *,
    unperturbed: bool,
    cores: int,
    timeout: float,
    cache_dir: str = "",
    cache_min_compile_s: float = 0.0,
    repeats: int = 3,
) -> dict[str, Any]:
    env = dict(os.environ)
    env.pop("AGRI_JAX_CHECK", None)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--device",
        device,
        "--precision",
        precision,
        "--group",
        group,
        "--inputs",
        str(INPUTS),
        "--launch-epoch",
        repr(time.time()),
    ]
    if B is not None:
        cmd += ["--B", str(B)]
    cmd += ["--repeats", str(repeats)]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir, "--cache-min-compile-s", repr(cache_min_compile_s)]
    if unperturbed:
        cmd += ["--unperturbed"]
    if device == "cpu1":
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        core = sorted(os.sched_getaffinity(0))[0]
        cmd = ["taskset", "-c", str(core), *cmd]
    elif device == "cpuN":
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cores}"
    elif device == "gpu":
        env.pop("JAX_PLATFORMS", None)
    t0 = time.perf_counter()
    cmd[cmd.index("--launch-epoch") + 1] = repr(time.time())
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    e2e = time.perf_counter() - t0
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")), None)
    if p.returncode != 0 or line is None:
        return {
            "device": device,
            "precision": precision,
            "B": B,
            "group": group,
            "error": (p.stderr or p.stdout)[-3000:],
            "e2e_s": e2e,
        }
    r = json.loads(line)
    r["process_wall_s"] = e2e
    r["e2e_s"] = r["e2e_single_pass_s"]
    r["xla_flags"] = env.get("XLA_FLAGS", "")
    return r


def sweep(a: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"jax_{a.tag}.jsonl"
    cores = len(os.sched_getaffinity(0))
    last: dict[str, Any] | None = None
    for B in a.B:
        if last is not None and "groups" in last:
            gs = last["groups"]
            predicted = sum(g["steady_s"] for g in gs) * B / sum(g["B"] for g in gs)
            if predicted > a.budget_s:
                rec = {
                    "device": a.device,
                    "precision": a.precision,
                    "B": B,
                    "skipped": True,
                    "predicted_steady_s": predicted,
                    "budget_s": a.budget_s,
                }
                print(json.dumps(rec), flush=True)
                with open(path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                continue
        r = launch(a.device, a.precision, B, a.group, unperturbed=False, cores=cores, timeout=a.timeout)
        r["tag"] = a.tag
        brief = [{k: v for k, v in g.items() if k != "sanity"} for g in r.get("groups", [])]
        print(json.dumps({k: v for k, v in r.items() if k != "groups"} | {"groups": brief}), flush=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        if "groups" in r:
            last = r


def warm(a: argparse.Namespace) -> None:
    """Cold then warm e2e at each B: two fresh processes sharing one fresh persistent cache."""
    import shutil

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"jax_warm_{a.tag}.jsonl"
    cores = len(os.sched_getaffinity(0))
    base = Path(a.cache_root or os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "jc"
    for B in a.B:
        cdir = base / f"{a.tag}_B{B}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True)
        for phase in a.phases:
            r = launch(
                a.device,
                a.precision,
                B,
                a.group,
                unperturbed=False,
                cores=cores,
                timeout=a.timeout,
                cache_dir=str(cdir),
                cache_min_compile_s=a.cache_min_compile_s,
                repeats=a.repeats,
            )
            r |= {"tag": a.tag, "phase": phase}
            brief = [{k: v for k, v in g.items() if k != "sanity"} for g in r.get("groups", [])]
            print(json.dumps({k: v for k, v in r.items() if k != "groups"} | {"groups": brief}), flush=True)
            with open(path, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        shutil.rmtree(cdir, ignore_errors=True)


def equal(a: argparse.Namespace) -> None:
    """The 65 acceptance runs once, unperturbed, every group (the task DSSAT's batch runs do)."""
    OUT.mkdir(parents=True, exist_ok=True)
    cores = len(os.sched_getaffinity(0))
    r = launch(a.device, a.precision, None, "all", unperturbed=True, cores=cores, timeout=a.timeout)
    r["tag"] = a.tag
    print(json.dumps(r)[:4000], flush=True)
    with open(OUT / f"jax_equal_{a.tag}.json", "w") as fh:
        json.dump(r, fh, indent=1)


def prepare_cmd(a: argparse.Namespace) -> None:
    sys.path.insert(0, str(HERE))
    import d4_jax_day as D

    work = Path(os.environ.get("AGRI_JAX_RUN_ROOT", "/tmp")) / "d41p"
    meta = D.prepare(DATA, work, INPUTS, a.jobs)
    print(json.dumps(meta))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--device", required=True)
    w.add_argument("--precision", default="f64")
    w.add_argument("--B", type=int, default=None)
    w.add_argument("--group", default="mix")
    w.add_argument("--inputs", default=str(INPUTS))
    w.add_argument("--repeats", type=int, default=3)
    w.add_argument("--long-call-s", type=float, default=60.0)
    w.add_argument("--unperturbed", action="store_true")
    w.add_argument("--launch-epoch", type=float, required=True)
    w.add_argument("--cache-dir", default="")
    w.add_argument("--cache-min-compile-s", type=float, default=0.0)
    wm = sub.add_parser("warm")
    wm.add_argument("--device", required=True)
    wm.add_argument("--precision", default="f64")
    wm.add_argument("--group", default="mix")
    wm.add_argument("--tag", required=True)
    wm.add_argument("--timeout", type=float, default=5400)
    wm.add_argument("--B", type=int, nargs="+", required=True)
    wm.add_argument("--repeats", type=int, default=3)
    wm.add_argument("--phases", nargs="+", default=["cold", "warm"])
    wm.add_argument("--cache-root", default="")
    wm.add_argument("--cache-min-compile-s", type=float, default=0.0)
    for name in ("sweep", "equal"):
        s = sub.add_parser(name)
        s.add_argument("--device", required=True)
        s.add_argument("--precision", default="f64")
        s.add_argument("--group", default="mix")
        s.add_argument("--tag", required=True)
        s.add_argument("--timeout", type=float, default=5400)
        s.add_argument("--budget-s", type=float, default=1500)
        if name == "sweep":
            s.add_argument("--B", type=int, nargs="+", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "worker":
        print(json.dumps(worker(a)), flush=True)
    elif a.cmd == "sweep":
        sweep(a)
    elif a.cmd == "equal":
        equal(a)
    elif a.cmd == "warm":
        warm(a)
    else:
        prepare_cmd(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
