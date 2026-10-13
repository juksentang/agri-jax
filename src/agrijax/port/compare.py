"""Series-by-series comparison of simulated and reference-model outputs, with a Markdown report.

Step 5/"whole-model validation" helper of the reference-validation procedure: given
two ``xarray`` objects on a shared ``time`` axis (Agri-JAX output and a reference-model output such
as the RZWQM2 ``.ana`` file), :func:`compare_series` aligns them, drops pairs where either side is
missing and computes, per variable,

* ``n``: number of paired values;
* ``bias`` = mean(sim - ref); ``mae`` = mean |sim - ref|; ``rmse`` = sqrt(mean (sim - ref)^2);
  ``max_abs`` = max |sim - ref|;
* ``pbias`` = 100 sum(sim - ref) / sum(ref) [%] (NaN when sum(ref) = 0);
* ``r2``: squared Pearson correlation of sim and ref (NaN when either side is constant);
* ``nse``: Nash-Sutcliffe efficiency 1 - SSE / sum (ref - mean ref)^2 (NaN for a constant ref).

Optional per-variable :class:`Tolerance` limits turn each row into pass / fail. Reports for
several periods (e.g. "year" and "season") are concatenated with :meth:`CompareReport.concat` or
``+`` and rendered with :meth:`CompareReport.to_markdown`.

This is plain NumPy (no JAX): it post-processes trajectories and is not a ``@process``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

__all__ = [
    "CompareReport",
    "SeriesMetrics",
    "Tolerance",
    "compare_series",
    "series_metrics",
]

_METRICS = ("n", "bias", "mae", "rmse", "max_abs", "pbias", "r2", "nse")


@dataclass(frozen=True)
class Tolerance:
    """Acceptance limits for one variable; ``None`` disables a check.

    ``rmse``, ``mae``, ``max_abs`` and ``abs_bias`` are upper limits in the variable's units,
    ``abs_pbias`` in percent, ``r2_min`` and ``nse_min`` are lower limits.
    """

    rmse: float | None = None
    mae: float | None = None
    max_abs: float | None = None
    abs_bias: float | None = None
    abs_pbias: float | None = None
    r2_min: float | None = None
    nse_min: float | None = None

    def check(self, m: SeriesMetrics) -> tuple[str, ...]:
        """Names of the limits that ``m`` violates (a NaN metric violates any limit set on it)."""
        failed: list[str] = []

        def upper(name: str, value: float, limit: float | None) -> None:
            if limit is not None and not (value <= limit):
                failed.append(name)

        def lower(name: str, value: float, limit: float | None) -> None:
            if limit is not None and not (value >= limit):
                failed.append(name)

        upper("rmse", m.rmse, self.rmse)
        upper("mae", m.mae, self.mae)
        upper("max_abs", m.max_abs, self.max_abs)
        upper("abs_bias", abs(m.bias), self.abs_bias)
        upper("abs_pbias", abs(m.pbias), self.abs_pbias)
        lower("r2_min", m.r2, self.r2_min)
        lower("nse_min", m.nse, self.nse_min)
        return tuple(failed)

    def describe(self) -> str:
        """Compact text such as ``rmse<=0.1, r2>=0.9``; empty when no limit is set."""
        parts: list[str] = []
        for f in fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if f.name.endswith("_min"):
                parts.append(f"{f.name[:-4]}>={v:g}")
            else:
                parts.append(f"{f.name}<={v:g}")
        return ", ".join(parts)


@dataclass(frozen=True)
class SeriesMetrics:
    """Metrics of one simulated series against its reference over one period."""

    variable: str
    ref_variable: str
    period: str
    units: str
    n: int
    bias: float
    mae: float
    rmse: float
    max_abs: float
    pbias: float
    r2: float
    nse: float
    ref_mean: float
    sim_mean: float
    tolerance: Tolerance | None = None
    failed: tuple[str, ...] = ()

    @property
    def passed(self) -> bool | None:
        """``True``/``False`` against the tolerance; ``None`` when no tolerance was given."""
        if self.tolerance is None:
            return None
        return not self.failed

    def as_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "tolerance"}
        d["tolerance"] = "" if self.tolerance is None else self.tolerance.describe()
        d["passed"] = self.passed
        return d


def series_metrics(sim: Any, ref: Any) -> dict[str, float]:
    """Metrics of two equally long 1-D arrays; pairs with a NaN on either side are dropped."""
    s = np.asarray(sim, dtype=float).ravel()
    r = np.asarray(ref, dtype=float).ravel()
    if s.shape != r.shape:
        raise ValueError(f"sim and ref differ in length: {s.shape} vs {r.shape}")
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    n = int(s.size)
    nan = float("nan")
    if n == 0:
        empty: dict[str, float] = {k: nan for k in (*_METRICS, "ref_mean", "sim_mean")}
        empty["n"] = 0
        return empty
    d = s - r
    sum_ref = float(np.sum(r))
    ss_ref = float(np.sum((r - r.mean()) ** 2))
    ss_sim = float(np.sum((s - s.mean()) ** 2))
    if ss_ref > 0.0 and ss_sim > 0.0:
        cov = float(np.sum((s - s.mean()) * (r - r.mean())))
        r2 = cov * cov / (ss_ref * ss_sim)
    else:
        r2 = nan
    return dict(
        n=n,
        bias=float(d.mean()),
        mae=float(np.abs(d).mean()),
        rmse=float(np.sqrt(np.mean(d * d))),
        max_abs=float(np.abs(d).max()),
        pbias=100.0 * float(d.sum()) / sum_ref if sum_ref != 0.0 else nan,
        r2=r2,
        nse=1.0 - float(np.sum(d * d)) / ss_ref if ss_ref > 0.0 else nan,
        ref_mean=float(r.mean()),
        sim_mean=float(s.mean()),
    )


def _fmt(v: Any, digits: int) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    x = float(v)
    if math.isnan(x):
        return "n/a"
    if x != 0.0 and (abs(x) < 10.0 ** (-digits) or abs(x) >= 1e6):
        return f"{x:.{max(digits - 1, 1)}e}"
    return f"{x:.{digits}f}"


@dataclass
class CompareReport:
    """A list of :class:`SeriesMetrics` rows (one per variable and period) plus a title and notes."""

    rows: list[SeriesMetrics] = field(default_factory=list)
    title: str = ""
    notes: list[str] = field(default_factory=list)

    # ---- access ---------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __add__(self, other: CompareReport) -> CompareReport:
        return CompareReport.concat([self, other], title=self.title or other.title)

    def get(self, variable: str, period: str | None = None) -> SeriesMetrics:
        """The row of ``variable`` (and ``period`` when several periods are present)."""
        hits = [m for m in self.rows if m.variable == variable and (period is None or m.period == period)]
        if len(hits) != 1:
            raise KeyError(f"{len(hits)} rows match variable={variable!r} period={period!r}")
        return hits[0]

    def __getitem__(self, key: str | tuple[str, str]) -> SeriesMetrics:
        if isinstance(key, tuple):
            return self.get(key[0], key[1])
        return self.get(key)

    @property
    def passed(self) -> bool:
        """``True`` when no row with a tolerance failed (rows without a tolerance are ignored)."""
        return all(m.passed is not False for m in self.rows)

    @property
    def periods(self) -> list[str]:
        return list(dict.fromkeys(m.period for m in self.rows))

    @classmethod
    def concat(cls, reports: Iterable[CompareReport], title: str = "") -> CompareReport:
        reps = list(reports)
        rows = [m for r in reps for m in r.rows]
        notes = [n for r in reps for n in r.notes]
        return cls(rows=rows, title=title or next((r.title for r in reps if r.title), ""), notes=notes)

    # ---- rendering ------------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([m.as_dict() for m in self.rows])

    def to_markdown(
        self,
        metrics: Sequence[str] = ("n", "ref_mean", "sim_mean", "bias", "rmse", "max_abs", "r2", "nse"),
        *,
        digits: int = 3,
        heading_level: int = 2,
    ) -> str:
        """Markdown table (one row per variable and period) with a status column when tolerances exist."""
        labels = {
            "n": "n",
            "ref_mean": "ref mean",
            "sim_mean": "sim mean",
            "bias": "bias",
            "mae": "MAE",
            "rmse": "RMSE",
            "max_abs": "max abs",
            "pbias": "PBIAS %",
            "r2": "R²",
            "nse": "NSE",
        }
        unknown = [k for k in metrics if k not in labels]
        if unknown:
            raise ValueError(f"unknown metrics {unknown}; choose from {list(labels)}")
        has_tol = any(m.tolerance is not None for m in self.rows)
        head = ["period", "variable", "reference", "units", *(labels[k] for k in metrics)]
        if has_tol:
            head += ["tolerance", "status"]
        out: list[str] = []
        if self.title:
            out += [f"{'#' * heading_level} {self.title}", ""]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join("---" if i < 4 else "---:" for i in range(len(head))) + "|")
        for m in self.rows:
            cells = [m.period, f"`{m.variable}`", f"`{m.ref_variable}`", m.units]
            cells += [_fmt(getattr(m, k), digits) for k in metrics]
            if has_tol:
                if m.tolerance is None:
                    cells += ["", ""]
                else:
                    status = "pass" if m.passed else "FAIL (" + ", ".join(m.failed) + ")"
                    cells += [m.tolerance.describe(), status]
            out.append("| " + " | ".join(cells) + " |")
        if self.notes:
            out.append("")
            out += [f"- {n}" for n in self.notes]
        return "\n".join(out) + "\n"

    def __str__(self) -> str:
        return self.to_markdown()


def _as_dataset(obj: xr.Dataset | xr.DataArray, default_name: str) -> xr.Dataset:
    if isinstance(obj, xr.Dataset):
        return obj
    if isinstance(obj, xr.DataArray):
        return obj.to_dataset(name=obj.name if obj.name is not None else default_name)
    raise TypeError(f"expected an xarray Dataset or DataArray, got {type(obj).__name__}")


def _normalise_vars(
    vars: Sequence[str] | Mapping[str, str] | None, sim: xr.Dataset, ref: xr.Dataset
) -> dict[str, str]:
    if vars is None:
        common = [v for v in sim.data_vars if v in ref.data_vars]
        if not common:
            raise ValueError("sim and ref share no variable names; pass `vars`")
        return {str(v): str(v) for v in common}
    if isinstance(vars, str):
        return {vars: vars}
    if isinstance(vars, Mapping):
        return {str(k): str(v) for k, v in vars.items()}
    return {str(v): str(v) for v in vars}


def _tolerance_for(
    tolerances: Mapping[str, Tolerance | float] | Tolerance | float | None, var: str
) -> Tolerance | None:
    if tolerances is None:
        return None
    if isinstance(tolerances, Mapping):
        t = tolerances.get(var)
    else:
        t = tolerances
    if t is None:
        return None
    if isinstance(t, Tolerance):
        return t
    return Tolerance(rmse=float(t))


def compare_series(
    sim: xr.Dataset | xr.DataArray,
    ref: xr.Dataset | xr.DataArray,
    vars: Sequence[str] | Mapping[str, str] | None = None,
    tolerances: Mapping[str, Tolerance | float] | Tolerance | float | None = None,
    *,
    dim: str = "time",
    period: str = "all",
    where: xr.DataArray | np.ndarray | slice | None = None,
    units: Mapping[str, str] | None = None,
    title: str = "",
) -> CompareReport:
    """Compare simulated against reference series variable by variable.

    Parameters
    ----------
    sim, ref : ``xarray`` Dataset (or a single named DataArray) with a ``dim`` coordinate. They
        are inner-joined on ``dim``, so only common time stamps are compared.
    vars : variable names present in both, or a mapping ``{sim_name: ref_name}``; default = all
        shared names.
    tolerances : per-variable :class:`Tolerance` (a bare float is an RMSE limit), or one
        tolerance for every variable; ``None`` = no pass/fail.
    period : label of this comparison window (shown in the report).
    where : optional selection on ``dim`` applied after alignment: a ``slice`` of coordinate
        labels, or a boolean mask on ``dim`` (DataArray indexed by ``dim`` or a NumPy array of
        the aligned length).
    units : units per sim variable; default taken from the ``units`` attribute (sim, then ref).
    title : report title.

    Every variable must be 1-D along ``dim`` after alignment; NaNs are dropped pairwise.
    """
    sim_ds = _as_dataset(sim, "value")
    ref_ds = _as_dataset(ref, "value")
    mapping = _normalise_vars(vars, sim_ds, ref_ds)
    for s_name, r_name in mapping.items():
        if s_name not in sim_ds:
            raise KeyError(f"{s_name!r} not in sim")
        if r_name not in ref_ds:
            raise KeyError(f"{r_name!r} not in ref")
    sim_sel = sim_ds[list(mapping)]
    ref_sel = ref_ds[list(dict.fromkeys(mapping.values()))]
    sim_a, ref_a = xr.align(sim_sel, ref_sel, join="inner")
    if isinstance(where, slice):
        sim_a = sim_a.sel({dim: where})
        ref_a = ref_a.sel({dim: where})
    elif where is not None:
        mask = np.asarray(where.reindex({dim: sim_a[dim]}) if isinstance(where, xr.DataArray) else where)
        mask = mask.astype(bool)
        if mask.shape != (sim_a.sizes[dim],):
            raise ValueError(f"mask of shape {mask.shape} does not match {dim} length {sim_a.sizes[dim]}")
        sim_a = sim_a.isel({dim: mask})
        ref_a = ref_a.isel({dim: mask})

    rows: list[SeriesMetrics] = []
    for s_name, r_name in mapping.items():
        s = sim_a[s_name]
        r = ref_a[r_name]
        if s.ndim != 1 or r.ndim != 1:
            raise ValueError(f"{s_name!r}/{r_name!r} must be 1-D along {dim!r}, got {s.dims} and {r.dims}")
        met = series_metrics(s.values, r.values)
        u = (units or {}).get(s_name) or s.attrs.get("units") or r.attrs.get("units") or ""
        tol = _tolerance_for(tolerances, s_name)
        base = SeriesMetrics(
            variable=s_name,
            ref_variable=r_name,
            period=period,
            units=str(u),
            n=int(met["n"]),
            bias=met["bias"],
            mae=met["mae"],
            rmse=met["rmse"],
            max_abs=met["max_abs"],
            pbias=met["pbias"],
            r2=met["r2"],
            nse=met["nse"],
            ref_mean=met["ref_mean"],
            sim_mean=met["sim_mean"],
            tolerance=tol,
        )
        if tol is not None:
            base = SeriesMetrics(**{**base.__dict__, "failed": tol.check(base)})
        rows.append(base)
    return CompareReport(rows=rows, title=title)
