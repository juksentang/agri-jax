"""Figures of Agri-JAX results (``aj.plot``; needs matplotlib: ``pip install "agrijax[plot]"``).

The functions only draw: they take what :mod:`agrijax.dssat` returns (a :class:`~agrijax.dssat.Season`,
a :class:`~agrijax.dssat.Reference`, a :class:`~agrijax.dssat.BatchResult`, a
:class:`~agrijax.calib.workflow.CalibrationResult`) or plain tables with the same columns, and
return the matplotlib ``Figure``.

* :func:`season` - the season's LAI, tops and grain weight, soil water, with DSSAT dashed;
* :func:`compare` - Agri-JAX against DSSAT day by day (1:1 panels, the RMSE in each title);
* :func:`batch` - the distribution of a batch's yields per scenario group;
* :func:`calibration` - observed, published-cultivar and calibrated values per target;
* :func:`weather_sensitivity` - the weather sensitivity calendar, one panel per variable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["batch", "calibration", "compare", "season", "weather_sensitivity"]

#: series colors (categorical slots 1 and 2 of a colour-vision-deficiency-checked palette)
AGRIJAX = "#2a78d6"
DSSAT = "#eb6834"
MUTED = "#52514e"
#: the daily panels: column -> label
PANELS = {
    "lai": "LAI [m² m⁻²]",
    "cwad": "tops weight [kg ha⁻¹]",
    "gwad": "grain weight [kg ha⁻¹]",
    "swtd": "soil water [mm]",
}


def _plt() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover (depends on the environment)
        raise ImportError('agrijax.plot needs matplotlib: pip install "agrijax[plot]"') from e
    return plt


def _frame(x: Any) -> Any:
    """The daily table of a Season / Reference, or ``x`` itself (a DataFrame)."""
    return getattr(x, "daily", x)


def _style(ax: Any) -> None:
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def season(
    result: Any,
    reference: Any = None,
    *,
    observed: Any = None,
    variables: Sequence[str] = tuple(PANELS),
    title: str | None = None,
) -> Any:
    """One panel per daily series of ``result`` (Agri-JAX, solid), with ``reference`` (DSSAT, dashed)
    and ``observed`` points (a table with ``date`` and some of the series' columns) when given."""
    plt = _plt()
    d = _frame(result)
    ref = _frame(reference) if reference is not None else None
    fig, axes = plt.subplots(len(variables), 1, figsize=(7.5, 1.9 * len(variables)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, variables, strict=True):
        ax.plot(d["date"], d[k], color=AGRIJAX, linewidth=2, label="Agri-JAX")
        if ref is not None and k in ref:
            ax.plot(ref["date"], ref[k], color=DSSAT, linewidth=2, linestyle=(0, (4, 3)), label="DSSAT-CSM")
        if observed is not None and k in observed:
            ax.plot(
                observed["date"],
                observed[k],
                "o",
                color=MUTED,
                markersize=5,
                fillstyle="none",
                label="observed",
            )
        ax.set_ylabel(PANELS.get(k, k), fontsize=8)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.suptitle(title or getattr(result, "treatment", ""), fontsize=10)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def compare(result: Any, reference: Any, *, variables: Sequence[str] = tuple(PANELS)) -> Any:
    """Agri-JAX (y) against DSSAT (x) on the days both report, one 1:1 panel per series, the RMSE in
    each panel's title."""
    plt = _plt()
    m = _frame(result).merge(_frame(reference), on="yrdoy", suffixes=("", "_dssat"))
    fig, axes = plt.subplots(1, len(variables), figsize=(3.0 * len(variables), 3.1))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, variables, strict=True):
        x, y = m[f"{k}_dssat"].to_numpy(float), m[k].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        lo, hi = (float(np.min(x[ok])), float(np.max(x[ok]))) if ok.any() else (0.0, 1.0)
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
        ax.plot(x[ok], y[ok], "o", color=AGRIJAX, markersize=3)
        rmse = float(np.sqrt(np.mean((y[ok] - x[ok]) ** 2))) if ok.any() else float("nan")
        ax.set_title(f"{PANELS.get(k, k)}\nRMSE {rmse:.3g}", fontsize=8)
        ax.set_xlabel("DSSAT-CSM", fontsize=8)
        _style(ax)
    axes[0].set_ylabel("Agri-JAX", fontsize=8)
    fig.tight_layout()
    return fig


def batch(result: Any, *, by: str = "year", value: str = "HWAM", label: str = "grain yield [kg ha⁻¹]") -> Any:
    """The distribution of ``value`` over a batch (a :class:`~agrijax.dssat.BatchResult` or its
    table): one box per ``by`` group (e.g. the weather year), every season a dot."""
    plt = _plt()
    t = getattr(result, "table", result)
    groups = sorted(t[by].unique())
    data = [t.loc[t[by] == g, value].to_numpy(float) for g in groups]
    fig, ax = plt.subplots(figsize=(max(4.0, 0.6 * len(groups) + 2), 3.4))
    ax.boxplot(
        data,
        widths=0.55,
        showfliers=False,
        medianprops={"color": AGRIJAX, "linewidth": 2},
        boxprops={"color": MUTED},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
    )
    rng = np.random.default_rng(0)
    for i, v in enumerate(data, start=1):
        ax.plot(i + rng.uniform(-0.18, 0.18, v.size), v, ".", color=AGRIJAX, alpha=0.25, markersize=3)
    ax.set_xticks(range(1, len(groups) + 1), [str(g) for g in groups], fontsize=8)
    ax.set_xlabel(by, fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    _style(ax)
    fig.tight_layout()
    return fig


def calibration(res: Any, *, set_: str = "calibration") -> Any:
    """Observed, published-cultivar and calibrated values of each target of a calibration
    (``res.fit``; dates as days after the first observed date of the code), one panel per code."""
    plt = _plt()
    fit = res.fit if hasattr(res, "fit") else res
    f = fit[fit["set"] == set_] if "set" in fit else fit
    codes = list(dict.fromkeys(f["code"]))
    fig, axes = plt.subplots(1, len(codes), figsize=(2.4 * len(codes), 3.0))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, codes, strict=True):
        g = f[f["code"] == c]
        vals = {k: g[k].to_numpy(float) for k in ("observed", "published", "calibrated")}
        if (g["kind"] == "date").all():
            import pandas as pd

            base = pd.to_datetime(g["observed"].astype(int).astype(str), format="%Y%j").min()
            vals = {
                k: (
                    pd.to_datetime(pd.Series(v.astype(int).astype(str)), format="%Y%j") - base
                ).dt.days.to_numpy()
                for k, v in vals.items()
            }
        x = np.arange(len(g))
        ax.plot(x, vals["observed"], "o", color=MUTED, fillstyle="none", markersize=7, label="observed")
        ax.plot(x, vals["published"], "s", color=DSSAT, markersize=5, label="published")
        ax.plot(x, vals["calibrated"], "D", color=AGRIJAX, markersize=5, label="calibrated")
        ax.set_title(c, fontsize=9)
        ax.set_xticks([])
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig


def weather_sensitivity(ws: Any, *, output: str | None = None, variables: Sequence[str] | None = None) -> Any:
    """The weather sensitivity calendar of a :class:`~agrijax.facade_weather.WeatherSensitivity`: one
    panel per variable, the daily derivative of ``output`` (default the first) as bars, the growth
    stages shaded alternately and named, the days the trust check reran marked (filled: pass, open:
    undecidable, cross: fail), the variable's trust label in the panel title."""
    plt = _plt()
    o = ws.outputs[0] if output is None else output
    vs = list(ws.variables if variables is None else variables)
    d = ws.daily[ws.daily["output"] == o]
    fig, axes = plt.subplots(len(vs), 1, figsize=(8.0, 1.9 * len(vs) + 0.6), sharex=True, squeeze=False)
    st = ws.stages[(ws.stages["output"] == o) & (ws.stages["variable"] == vs[0])]
    marks = {"pass": ("o", "full"), "undecidable": ("o", "none"), "fail": ("x", "full")}
    for ax, v in zip(axes[:, 0], vs, strict=True):
        t = d[d["variable"] == v]
        for k, (_, row) in enumerate(st.iterrows()):
            if k % 2:
                ax.axvspan(row["first"], row["last"], color="#f0efeb", lw=0)
        ax.bar(t["date"], t["derivative"], width=1.0, color=AGRIJAX)
        for status, (m, fill) in marks.items():
            c = t[t["checked"] == status]
            if len(c):
                ax.plot(c["date"], c["derivative"], m, color=DSSAT, fillstyle=fill, markersize=4, ls="none")
        ax.axhline(0.0, color=MUTED, lw=0.6)
        unit = t["unit"].iloc[0] if len(t) else ""
        ax.set_ylabel(f"d{o}/d{v}\n[{unit}]", fontsize=7)
        ax.set_title(f"{v}: {ws.trust.get(v, '')}", fontsize=8, loc="left")
        _style(ax)
    top = axes[0, 0]
    for _, row in st.iterrows():
        top.annotate(
            str(row["stage"]),
            (row["first"], 1.0),
            xycoords=("data", "axes fraction"),
            fontsize=6,
            color=MUTED,
            va="bottom",
        )
    fig.suptitle(f"{ws.name}: sensitivity of {o} to the daily weather (stage codes on top)", fontsize=9)
    fig.tight_layout()
    return fig
