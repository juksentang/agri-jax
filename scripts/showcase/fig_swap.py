#!/usr/bin/env python3
"""Swap figure of the showcase: replacing the soil-evaporation process of the DSSAT day.

The day has two registered soil-evaporation methods that both reproduce ``dscsm048``: Ritchie two-stage
(``MESEV R``, 39 of the 65 acceptance runs) and SALUS layered (``MESEV S``, 26 runs). Every run is simulated
twice, with its own method and with the other one; the swapped run is a different model, so only a finite
result and a closed daily water ledger are asserted for it.

``swap_soil_evaporation_{en,zh}.svg``
    Season soil evaporation, own method against swapped (one point per run), and the change in grain yield.

Input: ``<data-dir>/validation/aj_dint/swap_soil_evaporation.json``, written by
``tests/integration/test_day_dssat486_swap.py`` (run it with ``--runslow``).

Usage::

    python scripts/showcase/fig_swap.py --data-dir ~/agri_jax_data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svg as S

TEXT = {
    "en": {
        "title": "Swapping the soil-evaporation process of the DSSAT day",
        "sub": "Each of the 65 runs is simulated with its own method and with the other one, Ritchie or SALUS.",
        "p_es": "Season soil evaporation (mm)",
        "p_es_s": "own method on x, the other method on y; points on the line are unchanged; key: median change",
        "xl": "own method, mm",
        "yl": "other method, mm",
        "p_y": "Grain yield, change when swapped",
        "p_y_s": "runs with a yield above 0; percent of the run's own-method yield; label: largest change",
        "d_rs": "Ritchie → SALUS",
        "d_sr": "SALUS → Ritchie",
        "runs": "{n} runs",
        "median": "median {}",
        "foot": "The daily water ledger closes in both layouts (largest residual {l0} mm with the own method, {l1} mm swapped) "
        "and all 65 runs are finite. With its own method every run is within {yr} of the DSSAT yield. The swapped "
        "run is a different model: only finiteness and the closed ledger are asserted for it, not agreement with DSSAT.",
        "desc": "Left, season soil evaporation with the own method against the swapped method for 65 runs. Right, the "
        "change in grain yield: Ritchie to SALUS median {a} percent, range {a0} to {a1}; SALUS to Ritchie median "
        "{b} percent, range {b0} to {b1}.",
    },
    "zh": {
        "title": "替换 DSSAT 日模型中的土面蒸发过程",
        "sub": "65 个运行各用自身方法与另一种方法（Ritchie 或 SALUS）模拟一次。",
        "p_es": "全季土面蒸发 (mm)",
        "p_es_s": "横轴为自身方法，纵轴为另一种方法；落在对角线上表示不变；图例中为变化中位数",
        "xl": "自身方法，mm",
        "yl": "另一种方法，mm",
        "p_y": "替换后籽粒产量的变化",
        "p_y_s": "产量大于 0 的运行；相对各自自身方法产量的百分数；标注最大变化",
        "d_rs": "Ritchie → SALUS",
        "d_sr": "SALUS → Ritchie",
        "runs": "{n} 个运行",
        "median": "中位数 {}",
        "foot": "两种方法下每日水量账均闭合（自身方法最大残差 {l0} mm，替换后 {l1} mm），65 个运行的结果均有限。"
        "自身方法下每个运行的产量都在 DSSAT 产量的 {yr} 以内。替换后的运行是另一个模型：只断言结果有限、水量账闭合，"
        "不要求与 DSSAT 一致。",
        "desc": "左：65 个运行自身方法与替换方法的全季土面蒸发。右：籽粒产量的变化：Ritchie 换 SALUS 中位数 {a} %，范围 {a0} 至 {a1}；"
        "SALUS 换 Ritchie 中位数 {b} %，范围 {b0} 至 {b1}。",
    },
}


def stats(runs: list[dict]) -> dict:
    out = {}
    for me, key in (("R", "rs"), ("S", "sr")):
        rr = [r for r in runs if r["mesev"] == me]
        yc = np.array([100.0 * r["yield_change_rel"] for r in rr if r["yield_change_rel"] is not None])
        es = np.array([100.0 * (r["swapped_es_mm"] / r["native_es_mm"] - 1.0) for r in rr])
        out[key] = {
            "n": len(rr),
            "n_yield": len(yc),
            "yield": yc,
            "es_pct": es,
            "es_med": float(np.median(es)),
            "y_med": float(np.median(yc)),
            "y_min": float(yc.min()),
            "y_max": float(yc.max()),
            "led0": max(r["native_ledger_mm"] for r in rr),
            "led1": max(r["swapped_ledger_mm"] for r in rr),
            "own_yield_rel": max(r["native_yield_rel"] for r in rr if r["native_yield_rel"] is not None),
            "own": np.array([r["native_es_mm"] for r in rr]),
            "other": np.array([r["swapped_es_mm"] for r in rr]),
        }
    return out


def _marker(kind: str, x: float, y: float, cls: str = "fteal", r: float = 4.5) -> str:
    """Circle for Ritchie → SALUS, diamond for SALUS → Ritchie."""
    if kind == "rs":
        return S.circle(x, y, r, cls)
    d = r * 1.35
    return S.path(
        f"M{x:.1f},{y - d:.1f}L{x + d:.1f},{y:.1f}L{x:.1f},{y + d:.1f}L{x - d:.1f},{y:.1f}Z", f"{cls} ring"
    )


def _fmt_pct(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f} %".replace("-", "−")


def swap_figure(lang: str, st: dict) -> str:
    t = TEXT[lang]
    body = [S.txt(28, 40, t["title"], 20, "k b"), S.txt(28, 64, t["sub"], 15, "m")]
    # ---- left: season soil evaporation, own against other
    x, y = 28.0, 92.0
    body.append(S.txt(x, y + 4, t["p_es"], 17, "k w"))
    els, _ = S.paragraph(x, y + 26, t["p_es_s"], 440, lead=20)
    body += els
    size = 250.0
    x0, y1 = x + 58, y + 82
    y0 = y1 + size
    top = 500.0
    sx, sy = S.Scale(0, top, x0, x0 + size), S.Scale(0, top, y0, y1)
    for tv in (0, 100, 200, 300, 400, 500):
        body.append(S.line(x0, sy(tv), x0 + size, sy(tv), "grid"))
        body.append(S.line(sx(tv), y1, sx(tv), y0, "grid"))
        body.append(S.txt(x0 - 9, sy(tv) + 5, f"{tv}", 15, "n", "end"))
        body.append(S.txt(sx(tv), y0 + 22, f"{tv}", 15, "n", "middle"))
    body.append(S.line(sx(0), sy(0), sx(top), sy(top), "ink"))
    body.append(S.txt(x0 + size / 2, y0 + 44, t["xl"], 15, "m", "middle"))
    body.append(S.txt(x0, y1 - 8, t["yl"], 15, "m"))
    for key in ("rs", "sr"):
        for a, b in zip(st[key]["own"], st[key]["other"], strict=True):
            body.append(_marker(key, sx(a), sy(b)))
    # marker key to the right of the plot
    kx = x0 + size + 18
    for i, key in enumerate(("rs", "sr")):
        yy = y1 + 18 + i * 84
        body.append(_marker(key, kx + 6, yy - 5))
        body.append(S.txt(kx + 20, yy, t["d_" + key], 15, "k"))
        body.append(S.txt(kx + 20, yy + 20, t["runs"].format(n=st[key]["n"]), 15, "m"))
        body.append(S.txt(kx + 20, yy + 40, t["median"].format(_fmt_pct(st[key]["es_med"], 0)), 15, "n k"))
    # ---- right: yield change, two strips
    rx = 514.0
    body.append(S.txt(rx, y + 4, t["p_y"], 17, "k w"))
    els, _ = S.paragraph(rx, y + 26, t["p_y_s"], 440, lead=20)
    body += els
    ax0, ax1 = rx + 16, rx + 458 - 10
    sv = S.Scale(-30, 20, ax0, ax1)
    strip_h = 118.0
    tops = [y + 88, y + 88 + strip_h + 26]
    for tv in (-30, -20, -10, 0, 10, 20):
        body.append(S.line(sv(tv), tops[0] - 8, sv(tv), tops[1] + strip_h - 8, "ink" if tv == 0 else "grid"))
    for key, ty in zip(("rs", "sr"), tops, strict=True):
        s = st[key]
        body.append(S.txt(ax0, ty + 6, t["d_" + key], 15, "k w"))
        body.append(S.txt(ax0 + S.tw(t["d_" + key]) + 10, ty + 6, t["runs"].format(n=s["n_yield"]), 15, "m"))
        px = [sv(v) for v in s["yield"]]
        cy = ty + 62
        # a strip plot: deterministic vertical jitter, translucent dots so that the dense centre reads darker
        for i, p in enumerate(px):
            body.append(f'<g class="soft">{_marker(key, p, cy + ((i * 37) % 11 - 5) * 3.0, r=4)}</g>')
        ext = int(np.argmax(np.abs(s["yield"])))  # the largest change
        body.append(
            S.txt(
                px[ext],
                cy + ((ext * 37) % 11 - 5) * 3.0 - 13,
                _fmt_pct(float(s["yield"][ext])),
                15,
                "n k",
                "middle",
            )
        )
        body.append(S.txt(ax1, ty + 6, t["median"].format(_fmt_pct(s["y_med"], 2)), 15, "n k", "end"))
    ay = tops[1] + strip_h - 8
    body.append(S.line(ax0, ay, ax1, ay, "axis"))
    for tv in (-30, -20, -10, 0, 10, 20):
        body.append(S.line(sv(tv), ay, sv(tv), ay + 5, "axis"))
        body.append(S.txt(sv(tv), ay + 22, f"{tv:+d} %".replace("-", "−") if tv else "0", 15, "n", "middle"))
    # ---- footer
    led0 = max(st["rs"]["led0"], st["sr"]["led0"])
    led1 = max(st["rs"]["led1"], st["sr"]["led1"])
    yr = max(st["rs"]["own_yield_rel"], st["sr"]["own_yield_rel"])
    foot = t["foot"].format(l0=S.fmt_sci(led0), l1=S.fmt_sci(led1), yr=f"{100 * yr:.1f} %")
    els, yend = S.paragraph(28, 530, foot, 930, lead=20)
    body += els
    S.check_fit(yend, 20)
    a, b = st["rs"], st["sr"]
    desc = t["desc"].format(
        a=f"{a['y_med']:.2f}",
        a0=f"{a['y_min']:.1f}",
        a1=f"{a['y_max']:.1f}",
        b=f"{b['y_med']:.2f}",
        b0=f"{b['y_min']:.1f}",
        b1=f"{b['y_max']:.1f}",
    )
    return S.svg_doc("swap-soil-evaporation", lang, t["title"], desc, body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_common_args(ap)
    ap.add_argument("--swap-json", help="default <data-dir>/validation/aj_dint/swap_soil_evaporation.json")
    a = ap.parse_args()
    dd, out = S.data_dir(a.data_dir), S.fig_dir(a.out)
    runs = json.loads(Path(a.swap_json or dd / "validation/aj_dint/swap_soil_evaporation.json").read_text())[
        "runs"
    ]
    assert len(runs) == 65 and all(r["finite"] for r in runs)
    st = stats(runs)
    for lang in a.lang:
        (out / f"swap_soil_evaporation_{lang}.svg").write_text(swap_figure(lang, st))


if __name__ == "__main__":
    main()
