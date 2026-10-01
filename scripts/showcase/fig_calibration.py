#!/usr/bin/env python3
"""Calibration figure of the showcase: how grain number changes what G2 and G3 can be told apart.

``calibration_g2g3_{en,zh}.svg``
    Loss contours over the two CERES-Maize grain coefficients G2 (kernels per plant) and G3 (mg per kernel
    per day) for the UFGA8201 treatments 2, 4 and 6 (rainfed, irrigated, stressed in the vegetative phase; nitrogen off), left for the
    observations used by default (phenology dates, yield, biomass, LAI), right with the observed grain number
    (H#AM) added. The contours are a conditional slice on a 41 x 41 grid of forward runs (the other
    coefficients at the best values of a fit to real observations), not a posterior. Markers: the published
    cultivar and the 24 recalibrations (3 seeds x 8 starts, joint CMA-ES on treatments 2 and 4, treatment 6 held out).

Input: the result file of the identifiability check of the calibration study (``--ident``, default
``<data-dir>/validation/aj_d31/ident.json``; keys ``g2``, ``g3``, ``loss_base``, ``loss_gn`` [g2 x g3 grids],
``points.published`` and ``recal.{base,gn}_s{0,1,2}`` with ``theta`` and ``per_code_best``).
Needs ``numpy`` and ``contourpy`` >= 1.3 (an older build mis-handles NumPy 2; the script checks).

Usage::

    python scripts/showcase/fig_calibration.py --data-dir ~/agri_jax_data
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import contourpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svg as S


def _check_contourpy() -> None:
    """contourpy builds older than 1.1 return broken fill metadata under NumPy 2; fail with a clear message."""
    x = np.linspace(0.0, 1.0, 5)
    pl, ol = cp.contour_generator(x, x, np.add.outer(x, x), fill_type=cp.FillType.OuterOffset).filled(
        0.5, 1.0
    )
    if not (len(pl) == 1 and list(ol[0]) == [0, len(pl[0])]):
        raise RuntimeError(
            f"contourpy {cp.__version__} returns broken filled contours here; upgrade contourpy (>= 1.3)"
        )


G2, G3 = 4, 5  # columns of the coefficient vectors (P1, P2, P5, PHINT, G2, G3)
EDGES = [0.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9]  # bands of loss / minimum loss of the panel
BAND_TEAL = [8, 22, 38, 54, 72, 92]  # percent of the page's teal in each band, valley to worst

TEXT = {
    "en": {
        "title": "How grain number changes the identifiability of G2 and G3",
        "sub": "UFGA8201 treatments 2, 4 and 6, nitrogen off. Loss contours on a 41 × 41 grid of forward runs.",
        "p_a": "Dates, yield, biomass and LAI",
        "p_a_s": "a long valley: many (G2, G3) pairs fit equally well",
        "p_b": "The same, plus grain number (H#AM)",
        "p_b_s": "the valley closes to one basin around the published values",
        "xl": "G2, kernels per plant",
        "yl": "G3, mg per kernel per day",
        "n_a": "24 recalibrations: G2 {g2}, G3 {g3}. Held-out grain number (treatment 6), best fit of each seed: {h}.",
        "n_b": "24 recalibrations: G2 {g2}, G3 {g3}. Held-out grain number (treatment 6), best fit of each seed: {h}.",
        "lg_band": "loss relative to the minimum of the panel",
        "lg_pub": "published cultivar",
        "lg_fit": "recalibration (3 seeds × 8 starts)",
        "lg_in": "within 1.5× of the minimum",
        "foot": "Conditional slice: P1, P5 and PHINT at the best values of a fit to the real observations (254.6, 954.5, 45.84), "
        "P2 fixed; not a posterior. The contours use treatments 2, 4 and 6; the recalibrations used 2 and 4. "
        "Simulated grain number depends on G2 only, so adding it pins G2 and the ridge closes.",
        "desc": "Two contour maps of the loss over G2 (x, kernels per plant) and G3 (y, mg per kernel per day). Without grain "
        "number the low-loss region is a long curved valley from G2 430 to 990; with grain number it is a basin "
        "around the published G2 {pg2} and G3 {pg3}.",
    },
    "zh": {
        "title": "粒数如何改变 G2 与 G3 的可辨识性",
        "sub": "UFGA8201 处理 2、4、6，氮过程关闭。41 × 41 网格上前向运行得到的损失等高线。",
        "p_a": "日期、产量、生物量与 LAI",
        "p_a_s": "狭长谷地：许多 (G2, G3) 组合拟合同样好",
        "p_b": "同上，再加入粒数 (H#AM)",
        "p_b_s": "谷地收拢为以已发布值为中心的单一盆地",
        "xl": "G2，粒 株⁻¹",
        "yl": "G3，mg 粒⁻¹ d⁻¹",
        "n_a": "24 次重新标定：G2 {g2}，G3 {g3}。处理 6 留出的粒数（各种子的最优拟合）：{h}。",
        "n_b": "24 次重新标定：G2 {g2}，G3 {g3}。处理 6 留出的粒数（各种子的最优拟合）：{h}。",
        "lg_band": "损失相对本图最小值的倍数",
        "lg_pub": "已发布品种系数",
        "lg_fit": "重新标定（3 个种子 × 8 个起点）",
        "lg_in": "最小值的 1.5 倍以内",
        "foot": "条件切片：P1、P5、PHINT 取自对实测观测拟合的最优值 (254.6, 954.5, 45.84)，P2 固定；不是后验。"
        "等高线用处理 2、4、6，重新标定用处理 2 与 4。模拟粒数只取决于 G2，因此加入粒数即固定了 G2，山脊随之合拢。",
        "desc": "G2（横轴，粒 株⁻¹）与 G3（纵轴，mg 粒⁻¹ d⁻¹）上损失的两张等高线图。不含粒数时，低损失区是从 G2 430 到 990 的"
        "狭长弯曲谷地；加入粒数后收拢为以已发布值 G2 {pg2}、G3 {pg3} 为中心的盆地。",
    },
}


def _star(cx: float, cy: float, r: float) -> str:
    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.42
        pts.append(f"{cx + rr * math.cos(a):.1f},{cy + rr * math.sin(a):.1f}")
    return f'<polygon points="{" ".join(pts)}" class="frust ring" stroke-linejoin="round"/>'


def _rings(pts_list, off_list, px, py) -> str:
    d = []
    for pts, off in zip(pts_list, off_list, strict=True):
        for a, b in itertools.pairwise(off):
            ring = pts[a:b]
            d.append("M" + "L".join(f"{px(x):.1f},{py(y):.1f}" for x, y in ring) + "Z")
    return "".join(d)


def _range(v, fmt) -> str:
    return f"{fmt(min(v))}–{fmt(max(v))}"


def calibration_figure(lang: str, ident: dict) -> str:
    t = TEXT[lang]
    g2, g3 = np.asarray(ident["g2"], float), np.asarray(ident["g3"], float)
    pub = ident["points"]["published"]
    css = "".join(
        f".ajfig.ajfig .b{k}{{fill:color-mix(in srgb,var(--f-teal) {p}%,var(--f-bg))}}\n"
        for k, p in enumerate(BAND_TEAL)
    )
    body = [S.txt(28, 40, t["title"], 20, "k b"), S.txt(28, 64, t["sub"], 15, "m")]
    pw = 458.0
    specs = [
        ("loss_base", "base", "p_a", "p_a_s", "n_a", 28.0),
        ("loss_gn", "gn", "p_b", "p_b_s", "n_b", 514.0),
    ]
    px0, py0, ph = 54.0, 140.0, 240.0  # plot box offset in the panel, and its height
    for key, tag, tk, sk, nk, x in specs:
        loss = np.asarray(ident[key], float)  # [g2, g3]
        rel = loss / loss.min()
        x0, x1, y1, y0 = x + px0, x + pw - 8, py0, py0 + ph
        sx = S.Scale(g2[0], g2[-1], x0, x1)
        sy = S.Scale(g3[0], g3[-1], y0, y1)
        body.append(S.txt(x, 88, t[tk], 17, "k w"))
        body.append(S.txt(x, 108, t[sk], 15, "m"))
        body.append(S.txt(x, 132, t["yl"], 15, "m"))
        gen = cp.contour_generator(g2, g3, rel.T, fill_type=cp.FillType.OuterOffset)
        for k in range(6):
            pl, ol = gen.filled(EDGES[k], EDGES[k + 1])
            body.append(S.path(_rings(pl, ol, sx, sy), f"b{k}") if pl else "")
        for ln in cp.contour_generator(g2, g3, rel.T).lines(1.5):
            body.append(S.path("M" + "L".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in ln), "ink2"))
        # frame and ticks
        body.append(S.rect(x0, y1, x1 - x0, y0 - y1, "axis"))
        for tv in (300, 500, 700, 900):
            body.append(S.line(sx(tv), y0, sx(tv), y0 + 5, "axis"))
            body.append(S.txt(sx(tv), y0 + 22, f"{tv}", 15, "n", "middle"))
        for tv in (6, 8, 10, 12, 14, 16):
            body.append(S.line(x0 - 5, sy(tv), x0, sy(tv), "axis"))
            body.append(S.txt(x0 - 9, sy(tv) + 5, f"{tv}", 15, "n", "end"))
        body.append(S.txt((x0 + x1) / 2, y0 + 44, t["xl"], 15, "m", "middle"))
        # the recalibrations of this objective, then the published cultivar on top
        fits = np.vstack([np.asarray(ident["recal"][f"{tag}_s{i}"]["theta"], float) for i in range(3)])
        for a, b in zip(fits[:, G2], fits[:, G3], strict=True):
            body.append(S.circle(sx(a), sy(b), 3.5, "fink"))
        body.append(_star(sx(pub[G2]), sy(pub[G3]), 9.5))
        # measured spread, from the same records
        h = []
        for i in range(3):
            r = ident["recal"][f"{tag}_s{i}"]
            pc = r["per_code_best"]["UFGA8201_t06"]["H#AM"]
            h.append(100.0 * (pc["sim"][0] - pc["obs"][0]) / pc["obs"][0])
        hs = f"{min(h):+.0f}…{max(h):+.0f} %".replace("-", "−")
        note = t[nk].format(
            g2=_range(fits[:, G2], lambda v: f"{math.floor(v + 0.5):d}"),
            g3=_range(fits[:, G3], lambda v: f"{v:.1f}"),
            h=hs,
        )
        els, _ = S.paragraph(x, y0 + 70, note, pw - 10, lead=20, cls="k")
        body += els
    # legend: bands on the left, markers on the right
    ly = 500.0
    body.append(S.txt(28, ly, t["lg_band"], 15, "m"))
    labels = ["≤1.5", "2", "3", "5", "10", ">10"]
    for k in range(6):
        body.append(S.rect(28 + k * 54, ly + 10, 50, 14, f"b{k}"))
        body.append(S.txt(28 + k * 54 + 25, ly + 42, labels[k], 15, "n", "middle"))
    mx = 540.0
    body.append(_star(mx + 9, ly - 4, 8.5))
    body.append(S.txt(mx + 28, ly + 1, t["lg_pub"], 15, "k"))
    body.append(S.circle(mx + 9, ly + 17, 3.5, "fink"))
    body.append(S.txt(mx + 28, ly + 22, t["lg_fit"], 15, "k"))
    body.append(S.line(mx - 1, ly + 38, mx + 19, ly + 38, "ink2"))
    body.append(S.txt(mx + 28, ly + 43, t["lg_in"], 15, "k"))
    els, yend = S.paragraph(28, ly + 72, t["foot"], 930, lead=20)
    body += els
    S.check_fit(yend, 20)
    return S.svg_doc(
        "calibration-g2g3",
        lang,
        t["title"],
        t["desc"].format(pg2=f"{pub[G2]:.0f}", pg3=f"{pub[G3]:.2f}"),
        body,
        css,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_common_args(ap)
    ap.add_argument(
        "--ident", help="identifiability result (default <data-dir>/validation/aj_d31/ident.json)"
    )
    a = ap.parse_args()
    _check_contourpy()
    dd, out = S.data_dir(a.data_dir), S.fig_dir(a.out)
    ident = json.loads(Path(a.ident or dd / "validation/aj_d31/ident.json").read_text())
    for lang in a.lang:
        (out / f"calibration_g2g3_{lang}.svg").write_text(calibration_figure(lang, ident))


if __name__ == "__main__":
    main()
