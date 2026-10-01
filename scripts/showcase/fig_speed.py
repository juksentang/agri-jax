#!/usr/bin/env python3
"""Speed figures of the showcase.

``speed_unified_{en,zh}.svg``
    Agri-JAX and DSSAT-CSM on one timing boundary: the 65 acceptance runs replicated to B seasons (B = 1e4
    and 1e5), on one core, 32 cores, a whole 192-core node and one H100. JAX: first run (fresh process,
    empty compilation cache), warm start (second process reading the cache) and repeat call; DSSAT: whole
    run (staging the run directories to the last ``dscsm048`` invocation, daily output files written) and
    repeat run (directories already staged).

``speed_unroll_{en,zh}.svg``
    The GPU story in three steps: where the time went (kernels per simulated day, idle device), the change
    (layer recurrences of the soil-water bucket unrolled) and what it costs (compile time, first run),
    with the agreement checks of the two layouts.

Inputs (``<data-dir>/validation/``; result files of the benchmark scripts in ``scripts/bench`` and of the DSSAT
wall-time runs, all measured on the cluster; the JAX records are one JSON line per process)::

    aj_dbench_d42/jax_warm_cpu{1,32,192}.jsonl   JAX cold / warm / steady, CPU
    aj_dbench_d42/dssat_e2e_cpu{1,32,192}.json   DSSAT whole run / repeat run, CPU
    aj_dbench_d42/gpu_diag_gpufull.json          GPU profile, layer loops
    aj_dbench_d43/jax_warm_gpufull_{on,off}.jsonl  JAX on the H100, layer loops unrolled (on) / kept (off)
    aj_dbench_d43/gpu_diag_gpufull_unrolled.json   GPU profile, unrolled
    aj_dbench_d43/d2_1a_free_run_gpu_*.json      65-run acceptance on the GPU (agreement of the layouts)

Usage::

    python scripts/showcase/fig_speed.py --data-dir ~/agri_jax_data
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svg as S

#: dates of the measurements (the benchmark runs on the cluster; the records carry no date)
MEASURED = {"cpu": "2026-09-28", "gpu": "2026-09-29"}

TEXT = {
    "en": {
        "u_title": "Simulating B seasons on one timing boundary",
        "u_sub": "65 acceptance runs replicated to B seasons, float64. Seconds on a log axis, shorter is faster.",
        "col": "B = {} seasons",
        "g_cpu1": "1 core",
        "g_cpu1_d": "one pinned core of a node",
        "g_cpu32": "32 cores",
        "g_cpu32_d": "32 cores of a node",
        "g_cpu192": "192-core node",
        "g_cpu192_d": "whole node",
        "g_gpu": "One H100",
        "g_gpu_d": "whole GPU, layer loops unrolled (the GPU default)",
        "r_jfirst": "JAX first run",
        "r_jwarm": "JAX warm start",
        "r_jrep": "JAX repeat call",
        "r_dwhole": "DSSAT whole run",
        "r_drep": "DSSAT repeat run",
        "u_foot1": "First run: fresh process, empty compilation cache. Warm start: compilation cache read. Repeat call: same "
        "process. DSSAT whole run: staging, runs and daily output files; repeat run: directories already staged.",
        "u_foot2": "{cpu}, {date_cpu}; {gpu} ({host} host), {date_gpu}; JAX {jax}; one measurement per cell. The sides do "
        "different work: DSSAT writes 303 kB of daily text per season, JAX returns ten daily series in memory.",
        "u_desc": "Horizontal bars on a log axis: seconds for JAX first run, warm start and repeat call, and for DSSAT whole "
        "run and repeat run, for 1e4 and 1e5 seasons on one core, 32 cores, a 192-core node and one H100 GPU.",
        "o_title": "GPU: find the bottleneck, change the layout, check the model",
        "o_sub": "One H100, float64, the 65-run acceptance task replicated to B seasons; six programs.",
        "o_h1": "1  Profile: short kernels and idle gaps",
        "o_h2": "2  Result: steady call, six programs",
        "o_h3": "3  Cost, and the check that the model is unchanged",
        "o_p_kern": "Kernels per simulated day",
        "o_p_idle": "Device idle between kernels",
        "o_p_steady": "Seconds per call",
        "o_p_cost": "Seconds at B = 1e5",
        "loops": "layer loops",
        "unrolled": "unrolled",
        "compile": "XLA compile",
        "first": "first run",
        "warm": "warm start",
        "faster": "{}× faster",
        "o_chk": "Same model. GPU unrolled against CPU: stage dates identical in {n} of {n} runs, yield differs by at most "
        "{y} (relative). Unrolled against loops on the GPU: yield, soil water, runoff and drainage bit for bit; "
        "{b} of {n} runs bit for bit in every output, the rest differ in the last bits (largest: biomass {c} kg ha⁻¹).",
        "o_desc": "Paired bars for layer loops and unrolled layers on the H100: kernels per day {k0} to {k1} at 1e5 seasons, "
        "idle share {i0} to {i1}, steady call 3.1 times faster, compile and first run longer.",
        "B": ["B = 1e4", "B = 1e5"],
    },
    "zh": {
        "u_title": "同一计时边界下模拟 B 个季",
        "u_sub": "65 个验收运行复制到 B 个季，float64。横轴为秒（对数轴），越短越快。",
        "col": "B = {} 个季",
        "g_cpu1": "1 核",
        "g_cpu1_d": "节点上的一个绑定核",
        "g_cpu32": "32 核",
        "g_cpu32_d": "一个节点中的 32 核",
        "g_cpu192": "192 核整节点",
        "g_cpu192_d": "整个节点",
        "g_gpu": "一块 H100",
        "g_gpu_d": "整块 GPU，层循环展开（GPU 默认设置）",
        "r_jfirst": "JAX 首次运行",
        "r_jwarm": "JAX 热启动",
        "r_jrep": "JAX 重复调用",
        "r_dwhole": "DSSAT 完整运行",
        "r_drep": "DSSAT 重复运行",
        "u_foot1": "首次运行：新进程、编译缓存为空。热启动：读取编译缓存。重复调用：同一进程内再次调用。"
        "DSSAT 完整运行含目录准备、运行与逐日输出文件；重复运行：目录已备好。",
        "u_foot2": "{cpu}，{date_cpu}；{gpu}（主机 {host}），{date_gpu}；JAX {jax}；每格一次测量。"
        "两边工作量不同：DSSAT 每季写出 303 kB 逐日文本，JAX 在内存中返回十个逐日序列。",
        "u_desc": "对数轴上的横向条形图：1e4 与 1e5 个季在 1 核、32 核、192 核整节点和一块 H100 上，JAX 首次运行、热启动、"
        "重复调用与 DSSAT 完整运行、重复运行所需的秒数。",
        "o_title": "GPU：定位瓶颈、调整程序布局、核对模型不变",
        "o_sub": "一块 H100，float64，65 个验收运行复制到 B 个季；六个程序。",
        "o_h1": "1  剖析：内核短而多、设备有空闲",
        "o_h2": "2  结果：稳定调用，六个程序",
        "o_h3": "3  代价，以及模型不变的核对",
        "o_p_kern": "每个模拟日的内核数",
        "o_p_idle": "内核之间设备空闲的比例",
        "o_p_steady": "每次调用的秒数",
        "o_p_cost": "B = 1e5 时的秒数",
        "loops": "层循环",
        "unrolled": "展开",
        "compile": "XLA 编译",
        "first": "首次运行",
        "warm": "热启动",
        "faster": "快 {} 倍",
        "o_chk": "模型不变。GPU 展开版与 CPU 相比：{n} 个运行中阶段日期全部一致，产量最多相差 {y}（相对）。"
        "GPU 上展开与循环相比：产量、土壤水、径流与排水逐位一致；{n} 个运行中 {b} 个在全部输出上逐位一致，"
        "其余只在末几位不同（最大为生物量 {c} kg ha⁻¹）。",
        "o_desc": "H100 上层循环与展开层的成对条形图：1e5 个季时每日内核数由 {k0} 降至 {k1}，空闲比例由 {i0} 降至 {i1}，"
        "稳定调用快 3.1 倍，编译与首次运行变长。",
        "B": ["B = 1e4", "B = 1e5"],
    },
}


# ----------------------------------------------------------------------------- data
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _jax_cells(path: Path) -> dict[int, dict[str, float]]:
    """{B: first run, warm start, steady} of a JAX scaling file: end-to-end seconds of the cold and the
    warm process; steady = summed median call of the six programs, mean of the two processes."""
    out: dict[int, dict] = {}
    for r in _jsonl(path):
        c = out.setdefault(r["B_total"], {})
        c[r["phase"]] = (
            r["e2e_s"],
            sum(g["steady_s"] for g in r["groups"]),
            sum(g["compile_s"] for g in r["groups"]),
        )
    return {
        b: {
            "first": c["cold"][0],
            "warm": c["warm"][0],
            "steady": 0.5 * (c["cold"][1] + c["warm"][1]),
            "compile": c["cold"][2],
        }
        for b, c in out.items()
        if "cold" in c and "warm" in c
    }


def _meta(path: Path) -> dict[str, str]:
    """Hardware and library of the records of a JAX scaling file (one value each, checked)."""
    recs = _jsonl(path)
    cpu = {
        re.sub(r"\s+\d+-Core Processor|\(R\)|\(TM\)|\s+CPU\b|\s+@.*", "", r["cpu_model"]).strip()
        for r in recs
    }
    dev = {r["device_kind"].replace("NVIDIA ", "") for r in recs if r.get("backend") == "gpu"}
    jax = {r["jax"] for r in recs}
    if len(cpu) != 1 or len(jax) != 1 or len(dev) > 1:
        raise ValueError(f"{path}: mixed hardware or library versions {cpu} {dev} {jax}")
    return {"cpu": cpu.pop(), "gpu": dev.pop() if dev else "", "jax": jax.pop()}


def _dssat_cells(path: Path) -> dict[int, dict[str, float]]:
    d = json.loads(path.read_text())
    return {r["B"]: {"whole": r["e2e_s"], "repeat": r["run_s"]} for r in d["rows"] if r["output"] == "daily"}


def _dirs(dd: Path, d42: str | None, d43: str | None) -> tuple[Path, Path]:
    v = dd / "validation"
    return (Path(d42) if d42 else v / "aj_dbench_d42"), (Path(d43) if d43 else v / "aj_dbench_d43")


def load_unified(dd: Path, d42: str | None = None, d43: str | None = None) -> dict:
    d42, d43 = _dirs(dd, d42, d43)
    data = {}
    for key, n in (("cpu1", 1), ("cpu32", 32), ("cpu192", 192)):
        jx = _jax_cells(d42 / f"jax_warm_cpu{n}.jsonl")
        ds = _dssat_cells(d42 / f"dssat_e2e_cpu{n}.json")
        data[key] = {b: {"jax": jx[b], "dssat": ds[b]} for b in (10000, 100000)}
    gpu = _jax_cells(d43 / "jax_warm_gpufull_on.jsonl")
    data["gpu"] = {b: {"jax": gpu[b]} for b in (10000, 100000)}
    cpu_meta, gpu_meta = _meta(d42 / "jax_warm_cpu1.jsonl"), _meta(d43 / "jax_warm_gpufull_on.jsonl")
    assert all(_meta(d42 / f"jax_warm_cpu{n}.jsonl") == cpu_meta for n in (32, 192))
    data["meta"] = {
        "cpu": cpu_meta["cpu"],
        "gpu": gpu_meta["gpu"],
        "host": gpu_meta["cpu"],
        "jax": gpu_meta["jax"]
        if gpu_meta["jax"] == cpu_meta["jax"]
        else f"{cpu_meta['jax']} / {gpu_meta['jax']}",
        "date_cpu": MEASURED["cpu"],
        "date_gpu": MEASURED["gpu"],
    }
    # the cells must be the ones of the benchmark's own unified table
    uni = d42 / "unified.json"
    if uni.is_file():
        u = json.loads(uni.read_text())
        for key in ("cpu1", "cpu32", "cpu192"):
            for b in (10000, 100000):
                j = u[key]["jax"]
                assert abs(j[f"{b}/cold"]["e2e"] - data[key][b]["jax"]["first"]) < 1e-9
                assert abs(u[key]["dssat"][f"{b}/daily"]["e2e_s"] - data[key][b]["dssat"]["whole"]) < 1e-9
    return data


# ----------------------------------------------------------------------------- unified bars
def unified_figure(lang: str, data: dict) -> str:
    t = TEXT[lang]
    body = [S.txt(28, 40, t["u_title"], 20, "k b"), S.txt(28, 64, t["u_sub"], 15, "m")]
    col_x = [196.0, 584.0]
    col_w = 360.0
    sx = S.Scale(0.1, 1e4, 0, col_w, log=True)  # 5 decades; the last one holds the value labels
    top = 112.0
    pitch, bh, hh, gap = 18.0, 11.0, 18.0, 6.0
    groups = [
        (
            "cpu1",
            [
                ("jfirst", "jax", "first"),
                ("jwarm", "jax", "warm"),
                ("jrep", "jax", "steady"),
                ("dwhole", "dssat", "whole"),
                ("drep", "dssat", "repeat"),
            ],
        ),
        ("cpu32", None),
        ("cpu192", None),
        ("gpu", [("jfirst", "jax", "first"), ("jwarm", "jax", "warm"), ("jrep", "jax", "steady")]),
    ]
    groups[1] = ("cpu32", groups[0][1])
    groups[2] = ("cpu192", groups[0][1])
    rows_total = sum(len(rows) for _, rows in groups)
    ybottom = top + rows_total * pitch + len(groups) * hh + (len(groups) - 1) * gap
    for ci in range(2):
        x0 = col_x[ci]
        body.append(S.txt(x0, 86, t["col"].format(("1e4", "1e5")[ci]), 15, "k w"))
        for e in (-1, 0, 1, 2, 3):
            px = x0 + sx(10.0**e)
            body.append(S.line(px, 98, px, ybottom, "grid"))
            body.append(
                S.txt(px, 106, S.fmt_s(10.0**e).replace(".0", "") if e >= 0 else "0.1 s", 15, "n", "middle")
            )
    y = top
    for key, rows in groups:
        body.append(S.txt(28, y + 13, t[f"g_{key}"], 15, "k b"))
        body.append(S.txt(28 + S.tw(t[f"g_{key}"], 15) + 14, y + 13, t[f"g_{key}_d"], 15, "m"))
        y += hh
        for label, tool, field in rows:
            body.append(S.txt(40, y + 13, t[f"r_{label}"], 15, "k"))
            for ci, b in enumerate((10000, 100000)):
                v = data[key][b][tool][field]
                x0 = col_x[ci]
                body.append(
                    S.hbar(x0, x0 + sx(v), y + (pitch - bh) / 2, bh, "fteal" if tool == "jax" else "frust", 3)
                )
                body.append(S.txt(x0 + sx(v) + 7, y + 13, S.fmt_s(v), 15, "n k"))
            y += pitch
        y += gap
    els, yend = S.paragraph(28, ybottom + 30, t["u_foot1"], 930, lead=20)
    body += els
    els, yend = S.paragraph(28, yend, t["u_foot2"].format(**data["meta"]), 930, lead=20)
    body += els
    S.check_fit(yend, 20)
    return S.svg_doc("speed-unified", lang, t["u_title"], t["u_desc"], body)


# ----------------------------------------------------------------------------- unrolling story
def _profile(path: Path) -> dict[int, dict[str, float]]:
    d = json.loads(path.read_text())
    out = {}
    for b in (10000, 100000):
        rs = [r for r in d["runs"] if r["B"] == b and r.get("trace")]
        nd = sum(r["trace"]["n_days"] for r in rs)
        out[b] = {
            "kernels": sum(r["trace"]["kernels"] for r in rs) / nd,
            "idle": 1.0 - sum(r["trace"]["busy_us"] for r in rs) / sum(r["trace"]["span_us"] for r in rs),
        }
    return out


def load_unroll(dd: Path, d42: str | None = None, d43: str | None = None) -> dict:
    d42, d43 = _dirs(dd, d42, d43)
    on = _jax_cells(d43 / "jax_warm_gpufull_on.jsonl")
    off = _jax_cells(d43 / "jax_warm_gpufull_off.jsonl")
    acc = sorted(glob.glob(str(d43 / "d2_1a_free_run_gpu_22003447.json")))
    chk = json.loads(Path(acc[0]).read_text())["runs"] if acc else []
    rows = chk
    return {
        "loops": _profile(d42 / "gpu_diag_gpufull.json"),
        "unrolled": _profile(d43 / "gpu_diag_gpufull_unrolled.json"),
        "on": on,
        "off": off,
        "check": {
            "n": len(rows),
            "stages": all(r["istage_equal"] and r["gpu_loops_istage_equal"] for r in rows),
            "yield_gpu_cpu": max((r["yield_gpu_cpu_rel"] for r in rows), default=None),
            "yield_loops": max((r["gpu_loops_yield_rel"] for r in rows), default=None),
            "bit": sum(1 for r in rows if r["gpu_loops_bit_identical"]),
            "soil_runoff_drain": max(
                (max(r["gpu_loops_series_max_abs"][k] for k in ("soil_sw", "runoff", "drain")) for r in rows),
                default=None,
            ),
            "cwad": max((r["gpu_loops_series_max_abs"]["cwad"] for r in rows), default=None),
        },
    }


def _pairs(body, x, y, w, h, title, groups, fmt, ymax):
    """A vertical paired-bar panel. ``groups``: [(label, before, after, note)]; before gray, after teal."""
    body.append(S.txt(x, y + 16, title, 15, "k w"))
    top, bot = y + 44, y + h - 44
    sy = S.Scale(0, ymax, bot, top)
    body.append(S.line(x, bot, x + w, bot, "axis"))
    gw = w / len(groups)
    bw = min(56.0, gw / 2.6)
    for i, (label, v0, v1, note) in enumerate(groups):
        cx = x + gw * (i + 0.5)
        for j, (v, cls) in enumerate(((v0, "fneu"), (v1, "fteal"))):
            bx = cx - bw - 3 + j * (bw + 6)
            body.append(S.vbar(bx, bot, sy(v), bw, cls, 4))
            body.append(S.txt(bx + bw / 2, sy(v) - 6, fmt(v), 15, "n k h", "middle"))
        body.append(S.txt(cx, bot + 20, label, 15, "m", "middle"))
        if note:
            body.append(S.txt(cx, bot + 40, note, 15, "k w", "middle"))


def unroll_figure(lang: str, d: dict) -> str:
    t = TEXT[lang]
    body = [S.txt(28, 40, t["o_title"], 20, "k b"), S.txt(28, 64, t["o_sub"], 15, "m")]
    lo, un, on, off = d["loops"], d["unrolled"], d["on"], d["off"]
    B = t["B"]
    # legend, right of the subtitle
    lx = 760
    body.append(S.rect(lx, 54, 14, 10, "fneu", 2))
    body.append(S.txt(lx + 22, 64, t["loops"], 15, "k"))
    lx2 = lx + 22 + S.tw(t["loops"]) + 20
    body.append(S.rect(lx2, 54, 14, 10, "fteal", 2))
    body.append(S.txt(lx2 + 22, 64, t["unrolled"], 15, "k"))
    pw, y1, y2, ph = 224.0, 92.0, 356.0, 228.0
    gx = [28.0, 28.0 + pw + 30, 28.0 + 2 * (pw + 30)]
    wide = 972.0 - gx[2]
    body.append(S.txt(28, y1 + 4, t["o_h1"], 17, "k b"))
    body.append(S.txt(gx[2], y1 + 4, t["o_h2"], 17, "k b"))
    body.append(S.line(28, y1 + 12, 972, y1 + 12, "grid"))
    _pairs(
        body,
        gx[0],
        y1 + 14,
        pw,
        ph - 14,
        t["o_p_kern"],
        [
            (B[0], lo[10000]["kernels"], un[10000]["kernels"], ""),
            (B[1], lo[100000]["kernels"], un[100000]["kernels"], ""),
        ],
        lambda v: f"{v:.0f}",
        560,
    )
    _pairs(
        body,
        gx[1],
        y1 + 14,
        pw,
        ph - 14,
        t["o_p_idle"],
        [
            (B[0], lo[10000]["idle"], un[10000]["idle"], ""),
            (B[1], lo[100000]["idle"], un[100000]["idle"], ""),
        ],
        lambda v: f"{100 * v:.1f} %",
        0.56,
    )
    st = [
        (
            B[i],
            off[b]["steady"],
            on[b]["steady"],
            t["faster"].format(f"{off[b]['steady'] / on[b]['steady']:.1f}"),
        )
        for i, b in enumerate((10000, 100000))
    ]
    _pairs(body, gx[2], y1 + 14, wide, ph - 14, t["o_p_steady"], st, lambda v: f"{v:.2f} s", 1.6)
    body.append(S.txt(28, y2 + 4, t["o_h3"], 17, "k b"))
    body.append(S.line(28, y2 + 12, 972, y2 + 12, "grid"))
    cost = [
        (t["compile"], off[100000]["compile"], on[100000]["compile"], ""),
        (t["first"], off[100000]["first"], on[100000]["first"], ""),
        (t["warm"], off[100000]["warm"], on[100000]["warm"], ""),
    ]
    _pairs(body, 28, y2 + 14, 440, ph - 34, t["o_p_cost"], cost, S.fmt_s, 92)
    c = d["check"]
    assert c["n"] == 65 and c["stages"] and c["yield_loops"] == 0.0 and c["soil_runoff_drain"] == 0.0
    txtc = t["o_chk"].format(
        n=c["n"], y=S.fmt_sci(c["yield_gpu_cpu"], 1), b=c["bit"], c=S.fmt_sci(c["cwad"], 1)
    )
    els, yend = S.paragraph(510, y2 + 50, txtc, 462, lead=22, cls="k")
    body += els
    S.check_fit(yend)
    desc = t["o_desc"].format(
        k0=f"{lo[100000]['kernels']:.0f}",
        k1=f"{un[100000]['kernels']:.0f}",
        i0=f"{100 * lo[100000]['idle']:.0f} %",
        i1=f"{100 * un[100000]['idle']:.1f} %",
    )
    return S.svg_doc("speed-unroll", lang, t["o_title"], desc, body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_common_args(ap)
    ap.add_argument("--bench-d42", help="CPU benchmark results (default <data-dir>/validation/aj_dbench_d42)")
    ap.add_argument("--bench-d43", help="GPU benchmark results (default <data-dir>/validation/aj_dbench_d43)")
    a = ap.parse_args()
    dd, out = S.data_dir(a.data_dir), S.fig_dir(a.out)
    uni, unr = load_unified(dd, a.bench_d42, a.bench_d43), load_unroll(dd, a.bench_d42, a.bench_d43)
    for lang in a.lang:
        (out / f"speed_unified_{lang}.svg").write_text(unified_figure(lang, uni))
        (out / f"speed_unroll_{lang}.svg").write_text(unroll_figure(lang, unr))


if __name__ == "__main__":
    main()
