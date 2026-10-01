#!/usr/bin/env python3
"""Validation figures of the showcase: Agri-JAX against DSSAT-CSM v4.8.6.0 (``dscsm048``).

``validation_runs_{en,zh}.svg``
    The 65 acceptance runs of the free-run DSSAT day (58 treatments of the DSSAT maize example data and
    7 nitrogen-off field seasons): empirical distribution, over the runs, of the grain-yield error, the
    LAI and profile soil-water RMSE and the largest residual of the daily water ledger. Read from the
    per-run report ``<data-dir>/validation/aj_dint/d2_1a_free_run.json`` written by
    ``tests/integration/test_day_dssat486_free.py`` (run it with ``--runslow``).

``validation_daily_{en,zh}.svg``
    One treatment (default ``UFGA8201_t04``, irrigated, high nitrogen) day by day: LAI, biomass and grain mass, and the
    profile soil water of Agri-JAX over those of DSSAT, with the difference below each panel. Agri-JAX's
    daily series are read from ``<data-dir>/validation/aj_dint/daily/<run>.npz`` (same test); DSSAT's from
    the ``PlantGro.OUT`` and ``SoilWat.OUT`` of a reference run, ``--reference-dir``. ``--run-reference``
    makes that reference run here (``dscsm048`` from ``$AGRI_JAX_DSSAT``, nitrogen off, daily outputs on,
    as the test stages it), which takes a fraction of a second.

Usage::

    python scripts/showcase/fig_validation.py --data-dir ~/agri_jax_data --run-reference
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _svg as S

TEXT = {
    "en": {
        "runs_title": "65 free-run seasons against DSSAT-CSM 4.8.6",
        "runs_sub": "Nitrogen off, float64 on CPU. Each curve is the share of runs at or below a value.",
        "p_yield": "Grain yield, relative error",
        "p_yield_u": "59 runs with a DSSAT yield above 0",
        "p_lai": "Leaf area index, RMSE over the season",
        "p_lai_u": "m² m⁻², 65 runs",
        "p_sw": "Soil water of the profile, RMSE",
        "p_sw_u": "mm, 65 runs (DSSAT prints whole millimetres)",
        "p_led": "Daily water ledger, largest residual",
        "p_led_u": "units of 1e-14 mm, 65 runs",
        "limit": "acceptance limit 2 %",
        "median": "median {}",
        "max": "max {}",
        "foot_a": "Emergence, silking and maturity dates are equal in {n} of {n} runs. {z} runs have no grain yield in "
        "DSSAT (HWAM 0); Agri-JAX gives 0 for them too.",
        "foot_b": "Free run: soil water, evaporation and uptake are computed by Agri-JAX, not taken from DSSAT. "
        "Residue records and soil-property changes are replayed from DSSAT (those modules are not ported). "
        "Reference: dscsm048 build 486 outputs for 58 maize treatments of the DSSAT example experiments "
        "and 7 seasons at the AmeriFlux CA-TPA corn site.",
        "runs_desc": "Four cumulative distribution curves over the 65 runs: grain yield relative error (median {ym}, "
        "largest {yx}, acceptance limit 2 percent), LAI RMSE (largest {lx}), profile soil water RMSE "
        "(largest {sx} mm) and largest daily water-ledger residual (largest {gx} mm).",
        "d_title": "One season day by day: Agri-JAX over DSSAT-CSM 4.8.6",
        "d_sub": "{run}, nitrogen off. Harvest grain yield {ours} kg ha⁻¹ against {ref} kg ha⁻¹ in DSSAT.",
        "lg_dssat": "DSSAT-CSM 4.8.6",
        "lg_jax": "Agri-JAX",
        "r_lai": "Leaf area index (m² m⁻²)",
        "r_bio": "Above-ground biomass and grain mass (kg ha⁻¹)",
        "r_sw": "Soil water of the profile (mm)",
        "cwad": "biomass",
        "gwad": "grain",
        "silk": "silking",
        "mat": "maturity",
        "maxd": "max |Δ|",
        "d_foot": "Δ = Agri-JAX minus DSSAT. DSSAT prints LAI to 0.01, biomass and grain mass to 1 kg ha⁻¹ and soil "
        "water to 1 mm, so differences of that size are the rounding of its output files. DSSAT is drawn "
        "wide, Agri-JAX thin on top. Treatment 4 of UFGA8201 (irrigated, high nitrogen), one of the 65 runs.",
        "daily_desc": "Daily LAI, above-ground biomass, grain mass and profile soil water of {run} from Agri-JAX and "
        "from DSSAT-CSM, overlaid, with the difference below each panel.",
        "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    },
    "zh": {
        "runs_title": "65 个自由运行季对 DSSAT-CSM 4.8.6 的偏差",
        "runs_sub": "氮过程关闭，CPU 上 float64。每条曲线表示不大于某一数值的运行所占比例。",
        "p_yield": "籽粒产量相对误差",
        "p_yield_u": "DSSAT 产量大于 0 的 59 个运行",
        "p_lai": "叶面积指数全季 RMSE",
        "p_lai_u": "m² m⁻²，65 个运行",
        "p_sw": "剖面土壤水 RMSE",
        "p_sw_u": "mm，65 个运行（DSSAT 输出取整毫米）",
        "p_led": "每日水量账最大残差",
        "p_led_u": "单位 1e-14 mm，65 个运行",
        "limit": "验收上限 2 %",
        "median": "中位数 {}",
        "max": "最大 {}",
        "foot_a": "{n} 个运行的出苗、吐丝、成熟日期全部一致。{z} 个运行在 DSSAT 中无籽粒产量（HWAM 为 0），Agri-JAX 同样给出 0。",
        "foot_b": "自由运行：土壤水、蒸发与吸水均由 Agri-JAX 计算，不取自 DSSAT；残留物记录与土壤性质变化取自 DSSAT 回放（这两个模块尚未移植）。"
        "参照为 dscsm048 build 486 的输出，涵盖 DSSAT 示例试验的 58 个玉米处理和 AmeriFlux CA-TPA 玉米站点的 7 个季。",
        "runs_desc": "65 个运行上的四条累积分布曲线：籽粒产量相对误差（中位数 {ym}，最大 {yx}，验收上限 2 %）、"
        "LAI 的 RMSE（最大 {lx}）、剖面土壤水的 RMSE（最大 {sx} mm）和每日水量账最大残差（最大 {gx} mm）。",
        "d_title": "一个生长季逐日对照：Agri-JAX 与 DSSAT-CSM 4.8.6",
        "d_sub": "{run}，氮过程关闭。收获时籽粒产量 {ours} kg ha⁻¹，DSSAT 为 {ref} kg ha⁻¹。",
        "lg_dssat": "DSSAT-CSM 4.8.6",
        "lg_jax": "Agri-JAX",
        "r_lai": "叶面积指数 (m² m⁻²)",
        "r_bio": "地上部生物量与籽粒质量 (kg ha⁻¹)",
        "r_sw": "剖面土壤水 (mm)",
        "cwad": "生物量",
        "gwad": "籽粒",
        "silk": "吐丝",
        "mat": "成熟",
        "maxd": "最大 |Δ|",
        "d_foot": "Δ = Agri-JAX 减 DSSAT。DSSAT 输出 LAI 取两位小数，生物量与籽粒质量取整 kg ha⁻¹，土壤水取整 mm，"
        "同量级的差异来自其输出文件的取整。DSSAT 用粗线，Agri-JAX 用细线叠在上面。"
        "UFGA8201 处理 4（灌溉、高氮），65 个运行之一。",
        "daily_desc": "{run} 的逐日 LAI、地上部生物量、籽粒质量和剖面土壤水：Agri-JAX 与 DSSAT-CSM 叠画，各图下方为两者之差。",
        "months": ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"],
    },
}


# ----------------------------------------------------------------------------- the 65-run distribution
def _panel(body, t, x, y, w, h, title, unit, values, scale, ticks, tick_fmt, val_fmt, limit=None):
    """One ECDF panel at (x, y) of size w x h."""
    pl, pr, pt, pb = 50, 16, 64, 30
    x0, x1, y0, y1 = x + pl, x + w - pr, y + h - pb, y + pt  # plot box, y0 = bottom
    sx = S.Scale(scale[0], scale[1], x0, x1, log=scale[2])
    sy = S.Scale(0, 1, y0, y1)
    body.append(S.txt(x, y + 18, title, 17, "k w"))
    body.append(S.txt(x, y + 40, unit, 15, "m"))
    for f, lab in ((0, "0 %"), (0.5, "50 %"), (1, "100 %")):
        body.append(S.line(x0, sy(f), x1, sy(f), "grid"))
        body.append(S.txt(x0 - 8, sy(f) + 5, lab, 15, "n", "end"))
    for tv in ticks:
        body.append(S.line(sx(tv), y0, sx(tv), y0 + 5, "axis"))
        body.append(S.txt(sx(tv), y0 + 22, tick_fmt(tv), 15, "n", "middle"))
    body.append(S.line(x0, y0, x1, y0, "axis"))
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    xs = [sx(max(a, scale[0])) for a in v]
    d = f"M{x0:.1f},{y0:.1f}"
    for i, px in enumerate(xs):
        d += f"H{px:.1f}V{sy((i + 1) / n):.1f}"
    d += f"H{x1:.1f}"
    body.append(S.path(d, "steal"))
    med, mx = float(np.median(v)), float(v[-1])
    k = int(np.searchsorted(v, med, side="right"))
    body.append(S.circle(sx(med), sy(k / n), 4, "fteal"))
    body.append(S.txt(sx(med) + 9, sy(k / n) + 20, t["median"].format(val_fmt(med)), 15, "k n"))
    body.append(S.circle(sx(mx), sy(1.0), 4, "fteal"))
    body.append(S.txt(sx(mx), sy(1.0) - 10, t["max"].format(val_fmt(mx)), 15, "k n", "middle"))
    if limit is not None:
        body.append(S.line(sx(limit), y1, sx(limit), y0, "ink"))
        body.append(S.txt(sx(limit) - 6, y0 - 8, t["limit"], 15, "k", "end"))
    return med, mx


def runs_figure(lang: str, report: dict) -> str:
    t = TEXT[lang]
    rows = report["runs"]["free"]
    n = len(rows)
    yl = [r["yield_rel"] for r in rows if r["yield_rel"] is not None]
    lai = [r["lai"]["rmse"] for r in rows]
    sw = [r["soil_water"]["SWTD"]["rmse"] for r in rows]
    led = [r["ledger"]["max_abs_residual_mm"] * 1e14 for r in rows]
    zero = sum(1 for r in rows if r["yield_rel"] is None)
    summ = report["summary"]["free"]
    # the figure is drawn from the rows; they must agree with the report's own summary
    assert n == summ["n_runs"] and len(yl) == summ["yield_ok"] - zero
    assert (
        abs(max(yl) - summ["yield_rel_max"]) < 1e-15
        and abs(float(np.median(yl)) - summ["yield_rel_median"]) < 1e-15
    )
    assert abs(max(lai) - summ["lai_rmse_max"]) < 1e-15 and abs(max(sw) - summ["swtd_rmse_max_mm"]) < 1e-12
    assert summ["dates_equal"] == n and summ["yield_fail"] == []
    body = [
        S.txt(28, 40, t["runs_title"], 20, "k b"),
        S.txt(28, 64, t["runs_sub"], 15, "m"),
    ]
    pw, ph = 458, 204
    gx, gy = 28, 80
    f1 = lambda v: S.fmt_sci(v, 1)  # noqa: E731
    y_m, y_x = _panel(
        body,
        t,
        gx,
        gy,
        pw,
        ph,
        t["p_yield"],
        t["p_yield_u"],
        yl,
        (1e-6, 1e-1, True),
        [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        lambda v: f"1e{round(np.log10(v))}",
        f1,
        limit=0.02,
    )
    _, l_x = _panel(
        body,
        t,
        gx + pw + 28,
        gy,
        pw,
        ph,
        t["p_lai"],
        t["p_lai_u"],
        lai,
        (0, 0.007, False),
        [0, 0.002, 0.004, 0.006],
        lambda v: "0" if v == 0 else f"{v:.3f}",
        lambda v: f"{v:.4f}",
    )
    _, s_x = _panel(
        body,
        t,
        gx,
        gy + ph + 22,
        pw,
        ph,
        t["p_sw"],
        t["p_sw_u"],
        sw,
        (0, 0.4, False),
        [0, 0.1, 0.2, 0.3, 0.4],
        lambda v: "0" if v == 0 else f"{v:.1f}",
        lambda v: f"{v:.2f}",
    )
    _, g_x = _panel(
        body,
        t,
        gx + pw + 28,
        gy + ph + 22,
        pw,
        ph,
        t["p_led"],
        t["p_led_u"],
        led,
        (0, 7, False),
        [0, 2, 4, 6],
        lambda v: f"{v:g}",
        lambda v: f"{v:.1f}",
    )
    fy = gy + 2 * ph + 22 + 26
    els, fy = S.paragraph(28, fy, t["foot_a"].format(n=summ["dates_equal"], z=zero), 930)
    body += els
    els, fy = S.paragraph(28, fy, t["foot_b"], 930)
    body += els
    S.check_fit(fy)
    desc = t["runs_desc"].format(
        ym=S.fmt_sci(y_m), yx=S.fmt_sci(y_x), lx=f"{l_x:.4f}", sx=f"{s_x:.2f}", gx=S.fmt_sci(g_x * 1e-14)
    )
    return S.svg_doc("validation-runs", lang, t["runs_title"], desc, body)


# ----------------------------------------------------------------------------- one season day by day
def _table(path: Path, first: str = "@YEAR") -> dict[str, np.ndarray]:
    """A DSSAT daily output file as {column: array}, plus ``yrdoy``."""
    lines = path.read_text(errors="replace").splitlines()
    hi = [i for i, ln in enumerate(lines) if ln.startswith(first)]
    if len(hi) != 1:
        raise ValueError(f"{path}: expected one '{first}' header, found {len(hi)}")
    cols = [c.lstrip("@") for c in lines[hi[0]].split()]
    rows = []
    for ln in lines[hi[0] + 1 :]:
        if not ln.strip() or ln[0] in "*!":
            break
        rows.append(ln.split())
    out = {c: np.array([float(r[i]) for r in rows]) for i, c in enumerate(cols)}
    out["yrdoy"] = (out["YEAR"] * 1000 + out["DOY"]).astype(int)
    return out


def _layer_thickness(inp: Path) -> np.ndarray:
    """Layer thicknesses [cm] of the soil profile in a run's ``DSSAT48.INP`` (``*SOIL`` block)."""
    lines = inp.read_text(errors="replace").splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("*SOIL"))
    bottoms = []
    for ln in lines[i + 4 :]:  # 3 header records, then the first layer table (SLB, ...)
        if not ln.strip():
            break
        bottoms.append(float(ln[:6]))
    return np.diff(np.r_[0.0, bottoms])


def _nitrogen_off(text: str) -> str:
    """The FileX with NITRO = N and daily growth / water outputs (as ``test_ceres_dssat._nitrogen_off``)."""
    out, opt, outp = [], "", ""
    for ln in text.splitlines():
        if ln.startswith("@N OPTIONS"):
            opt = ln
        elif ln.startswith("@N OUTPUTS"):
            outp = ln
        elif opt and ln.split()[1:2] == ["OP"]:
            k = opt.index("NITRO") + 4
            ln, opt = ln[:k] + "N" + ln[k + 1 :], ""
        elif outp and ln.split()[1:2] == ["OU"]:
            for name, val in (("FROPT", "1"), ("GROUT", "Y"), ("WAOUT", "Y")):
                k = outp.index(name) + 4
                ln = ln[:k] + val + ln[k + 1 :]
            outp = ""
        out.append(ln)
    return "\n".join(out) + "\n"


def run_reference(run: str, work: Path) -> Path:
    """Run ``dscsm048`` (nitrogen off) for ``<EXP>_t<NN>``; returns the directory of its output files."""
    from agrijax.port.run_fortran import DSSAT_ENGINE, run_dscsm

    exp, trno = run.split("_t")[0], int(run.split("_t")[1])
    maize = DSSAT_ENGINE / "example_data" / "Maize"
    weather = DSSAT_ENGINE / "example_data" / "Weather"
    dest = work / run
    dest.mkdir(parents=True, exist_ok=True)
    for f in maize.glob(exp + ".MZ*"):
        shutil.copy2(f, dest / f.name)
    x = dest / f"{exp}.MZX"
    x.write_text(_nitrogen_off(x.read_text(errors="replace")))
    batch = "$BATCH(MAIZE)\n!\n@FILEX" + " " * 88 + "TRTNO     RP     SQ     OP     CO\n"
    batch += f"{exp}.MZX".ljust(92) + f"{trno:7d}      1      0      0      0\n"
    (dest / "DSSBatch.v48").write_text(batch)
    res = run_dscsm(
        dest,
        dest / "out",
        run_mode="B",
        experiment_file="DSSBatch.v48",
        extra_files=sorted(weather.glob(exp[:4] + "*.WTH")),
        keep_files=("*.OUT", "DSSAT48.INP"),
        run_root=work / "rr",
    )
    return res.out_dir


def _yrdoy_date(v: int) -> date:
    return date(v // 1000, 1, 1) + timedelta(days=v % 1000 - 1)


def _align(days, ours, ref_yrdoy, ref):
    """(day index of ours, ours, ref) on the days both have."""
    pos = {int(d): i for i, d in enumerate(ref_yrdoy)}
    k = [i for i, d in enumerate(days) if int(d) in pos]
    return np.array(k), ours[k], np.array([ref[pos[int(days[i])]] for i in k])


def daily_figure(lang: str, report: dict, run: str, npz: Path, ref_dir: Path) -> str:
    t = TEXT[lang]
    z = np.load(npz)
    days = z["days"].astype(int)
    pg = _table(ref_dir / "PlantGro.OUT")
    sw = _table(ref_dir / "SoilWat.OUT")
    dl = _layer_thickness(ref_dir / "DSSAT48.INP")
    row = next(r for r in report["runs"]["free"] if r["run"] == run)
    nl = len(dl)
    swtd = (z["ours.soil_sw"][:, :nl] * dl).sum(axis=1) * 10.0  # cm3 cm-3 x cm -> mm
    series = {
        "lai": (z["ours.lai"][:, 0], pg["yrdoy"], pg["LAID"]),
        "cwad": (z["ours.cwad"][:, 0], pg["yrdoy"], pg["CWAD"]),
        "gwad": (z["ours.gwad"][:, 0], pg["yrdoy"], pg["GWAD"]),
        "swtd": (swtd, sw["yrdoy"], sw["SWTD"]),
    }
    al = {k: _align(days, *v) for k, v in series.items()}
    # the curves must be the ones of the report: same RMSE as the per-run table
    rm = float(np.sqrt(np.mean((al["swtd"][1] - al["swtd"][2]) ** 2)))
    assert abs(rm - row["soil_water"]["SWTD"]["rmse"]) < 1e-6, (rm, row["soil_water"]["SWTD"]["rmse"])

    x0, x1 = 92.0, 846.0  # plot box; the gutter to the right carries the end labels
    gx = x1 + 12
    d0 = _yrdoy_date(int(days[0]))
    dn = (_yrdoy_date(int(days[-1])) - d0).days
    sx = lambda yd: x0 + (_yrdoy_date(int(yd)) - d0).days / dn * (x1 - x0)  # noqa: E731
    body = [S.txt(28, 40, t["d_title"], 20, "k b")]
    ours_y, ref_y = row["yield"], row["hwam"]
    body.append(
        S.txt(28, 64, t["d_sub"].format(run=run, ours=S.grouped(ours_y), ref=S.grouped(ref_y)), 15, "m")
    )
    # legend under the title, right
    lx = 690
    body.append(S.line(lx, 35, lx + 26, 35, "srust w4"))
    body.append(S.txt(lx + 34, 40, t["lg_dssat"], 15, "k"))
    lx2 = lx + 34 + S.tw(t["lg_dssat"]) + 22
    body.append(S.line(lx2, 35, lx2 + 26, 35, "steal"))
    body.append(S.txt(lx2 + 34, 40, t["lg_jax"], 15, "k"))

    blocks = [(92.0, "lai", t["r_lai"]), (238.0, "bio", t["r_bio"]), (384.0, "swtd", t["r_sw"])]
    ph, rh = 80.0, 28.0
    events = {"silk": row["dates"]["silking"]["ours"], "mat": row["dates"]["maturity"]["ours"]}
    months = []
    m = date(d0.year, d0.month, 1)
    while m <= d0 + timedelta(days=dn):
        if m >= d0:
            months.append(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    mx_of = lambda mo: sx(mo.year * 1000 + mo.timetuple().tm_yday)  # noqa: E731

    for by, key, title in blocks:
        top, bot = by + 18, by + 18 + ph
        body.append(S.txt(x0, by + 10, title, 15, "k w"))
        if key == "lai":
            body.append(S.txt(sx(events["silk"]) + 6, by + 10, t["silk"], 15, "m"))
            body.append(S.txt(sx(events["mat"]) - 6, by + 10, t["mat"], 15, "m", "end"))
        keys = {"lai": ["lai"], "bio": ["cwad", "gwad"], "swtd": ["swtd"]}[key]
        allv = np.concatenate([np.r_[al[k][1], al[k][2]] for k in keys])
        if key == "swtd":
            lo = np.floor(allv.min() / 50) * 50
            hi = np.ceil(allv.max() / 50) * 50
            ticks = list(np.arange(lo, hi + 1, 50))
        elif key == "bio":
            lo, hi = 0.0, np.ceil(allv.max() * 1.02 / 5000) * 5000
            ticks = list(np.arange(0, hi + 1, 10000))
        else:
            lo, hi = 0.0, S.nice_ceil(allv.max() * 1.02)
            ticks = list(np.arange(0, hi + 1e-9, 1.0))
        sy = S.Scale(lo, hi, bot, top)
        for tv in ticks:
            body.append(S.line(x0, sy(tv), x1, sy(tv), "grid"))
            lab = S.grouped(tv) if tv >= 1000 else f"{tv:g}"
            body.append(S.txt(x0 - 8, sy(tv) + 5, lab, 15, "n", "end"))
        for mo in months:
            body.append(S.line(mx_of(mo), top, mx_of(mo), bot, "grid"))
        for ev in ("silk", "mat"):
            body.append(S.line(sx(events[ev]), top, sx(events[ev]), bot, "ink"))
        for k in keys:
            idx, ours, ref = al[k]
            xs = [sx(days[i]) for i in idx]
            body.append(S.polyline(xs, [sy(v) for v in ref], "srust w4"))
            body.append(S.polyline(xs, [sy(v) for v in ours], "steal w15"))
        if key == "bio":
            for k in ("cwad", "gwad"):
                _, ours, _ = al[k]
                body.append(S.txt(gx, sy(ours[-1]) + 5, t[k], 15, "k"))
        # difference strip under the panel
        rtop, rbot = bot + 6, bot + 6 + rh
        rmid = (rtop + rbot) / 2
        mxd = max(float(np.max(np.abs(al[k][1] - al[k][2]))) for k in keys)
        lim = S.nice_ceil(mxd * 1.15)
        sr = S.Scale(-lim, lim, rbot, rtop)
        body.append(S.line(x0, rmid, x1, rmid, "grid"))
        for k in keys:
            idx, ours, ref = al[k]
            body.append(S.polyline([sx(days[i]) for i in idx], [sr(v) for v in ours - ref], "steal w15"))
        body.append(S.txt(x0 - 8, rmid + 5, f"±{lim:g}", 15, "n", "end"))
        body.append(S.txt(x0 + 6, rtop + 13, "Δ", 15, "m"))
        body.append(S.txt(gx, rtop + 11, t["maxd"], 15, "m"))
        body.append(S.txt(gx, rtop + 27, f"{mxd:.2g}", 15, "k n"))
    # shared month axis
    ay = blocks[-1][0] + 18 + ph + 6 + rh + 8
    body.append(S.line(x0, ay, x1, ay, "axis"))
    for mo in months:
        body.append(S.line(mx_of(mo), ay, mx_of(mo), ay + 5, "axis"))
        body.append(S.txt(mx_of(mo), ay + 22, t["months"][mo.month - 1], 15, "n", "middle"))
    els, yend = S.paragraph(28, ay + 50, t["d_foot"], 930)
    body += els
    S.check_fit(yend)
    return S.svg_doc("validation-daily", lang, t["d_title"], t["daily_desc"].format(run=run), body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S.add_common_args(ap)
    ap.add_argument(
        "--report", help="per-run report (default <data-dir>/validation/aj_dint/d2_1a_free_run.json)"
    )
    ap.add_argument(
        "--run", default="UFGA8201_t04", help="treatment of the daily figure (default UFGA8201_t04)"
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--reference-dir", help="directory with the PlantGro.OUT, SoilWat.OUT, DSSAT48.INP of the run"
    )
    g.add_argument(
        "--run-reference", action="store_true", help="run dscsm048 for --run here ($AGRI_JAX_DSSAT)"
    )
    a = ap.parse_args()
    dd, out = S.data_dir(a.data_dir), S.fig_dir(a.out)
    rep = json.loads(Path(a.report or dd / "validation/aj_dint/d2_1a_free_run.json").read_text())
    for lang in a.lang:
        (out / f"validation_runs_{lang}.svg").write_text(runs_figure(lang, rep))
    npz = dd / "validation/aj_dint/daily" / f"{a.run}.npz"
    ref = Path(a.reference_dir) if a.reference_dir else None
    tmp = None
    if a.run_reference:
        # short path: the Fortran binaries read run-directory paths into fixed-length records
        tmp = tempfile.mkdtemp(prefix="ajv", dir=os.environ.get("AJ_SHORT_TMP", "/tmp"))
        ref = run_reference(a.run, Path(tmp))
    if ref is None:
        print("daily figure skipped: pass --reference-dir or --run-reference", file=sys.stderr)
    else:
        for lang in a.lang:
            (out / f"validation_daily_{lang}.svg").write_text(daily_figure(lang, rep, a.run, npz, ref))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
