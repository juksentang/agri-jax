"""``agrijax.port.compare`` against implementations that share no code with it.

* exact rational arithmetic (:class:`fractions.Fraction`) for bias, MAE, MSE, max |d|, PBIAS, NSE
  and the squared correlation, on 200 seeded random series with NaN gaps (the float values are
  converted exactly, so the reference has no rounding at all);
* ``scipy.stats.pearsonr`` and ``statistics.correlation`` (stdlib) for R2;
* a ``pandas`` inner join on the time index plus ``dropna`` for the alignment and NaN handling of
  :func:`compare_series`, with shifted, gappy time axes and a boolean mask.
"""

from __future__ import annotations

import math
import statistics
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

from agrijax.port.compare import compare_series, series_metrics

N_CASES = 200


def _case(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 400))
    ref = rng.gamma(2.0, 2.0, n) * rng.choice([1.0, 1e-3, 1e3])
    sim = ref * rng.uniform(0.5, 1.5) + rng.normal(0.0, ref.std() + 1e-9, n) * rng.uniform(0, 1)
    for arr in (sim, ref):  # NaN gaps on either side
        arr[rng.random(n) < 0.1] = np.nan
    return sim, ref


def _exact(sim: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    pairs = [
        (Fraction(float(s)), Fraction(float(r)))
        for s, r in zip(sim, ref, strict=True)
        if np.isfinite(s) and np.isfinite(r)
    ]
    n = len(pairs)
    d = [s - r for s, r in pairs]
    sum_r = sum(r for _, r in pairs)
    mean_r = sum_r / n
    mean_s = sum(s for s, _ in pairs) / n
    ss_r = sum((r - mean_r) ** 2 for _, r in pairs)
    ss_s = sum((s - mean_s) ** 2 for s, _ in pairs)
    cov = sum((s - mean_s) * (r - mean_r) for s, r in pairs)
    sse = sum(x * x for x in d)
    return dict(
        n=n,
        bias=float(sum(d) / n),
        mae=float(sum(abs(x) for x in d) / n),
        rmse=math.sqrt(float(sse / n)),
        max_abs=float(max(abs(x) for x in d)),
        pbias=float(100 * sum(d) / sum_r),
        nse=float(1 - sse / ss_r),
        r2=float(cov * cov / (ss_r * ss_s)),
        ref_mean=float(mean_r),
        sim_mean=float(mean_s),
    )


def test_metrics_equal_exact_rational_reference() -> None:
    for seed in range(N_CASES):
        sim, ref = _case(seed)
        m = series_metrics(sim, ref)
        e = _exact(sim, ref)
        assert m["n"] == e["n"], seed
        for k in ("bias", "mae", "rmse", "max_abs", "pbias", "nse", "r2", "ref_mean", "sim_mean"):
            scale = max(abs(e[k]), abs(e["ref_mean"]) if k in ("bias", "mae", "rmse", "max_abs") else 1.0)
            assert m[k] == pytest.approx(e[k], rel=1e-9, abs=1e-12 * scale), (seed, k, m[k], e[k])


@pytest.mark.parametrize("seed", range(0, N_CASES, 10))
def test_r2_equals_scipy_and_stdlib(seed: int) -> None:
    sim, ref = _case(seed)
    ok = np.isfinite(sim) & np.isfinite(ref)
    r_scipy = stats.pearsonr(sim[ok], ref[ok]).statistic
    r_std = statistics.correlation(sim[ok].tolist(), ref[ok].tolist())
    m = series_metrics(sim, ref)
    assert m["r2"] == pytest.approx(r_scipy**2, rel=1e-9)
    assert m["r2"] == pytest.approx(r_std**2, rel=1e-9)


@pytest.mark.parametrize("seed", range(0, N_CASES, 20))
def test_compare_series_alignment_equals_pandas_join(seed: int) -> None:
    rng = np.random.default_rng(10_000 + seed)
    sim, ref = _case(seed)
    t_ref = pd.date_range("2015-01-01", periods=len(ref), freq="D")
    shift = int(rng.integers(-5, 6))
    t_sim = t_ref + pd.Timedelta(days=shift)
    keep = rng.random(len(sim)) > 0.05  # sim also misses some time stamps entirely
    s_ser = pd.Series(sim[keep], index=t_sim[keep], name="x")
    r_ser = pd.Series(ref, index=t_ref, name="x_ref")
    joined = pd.concat([s_ser, r_ser], axis=1, join="inner").dropna()
    mask_days = joined.index[joined.index.dayofweek < 5]  # weekdays only, as a mask
    sim_ds = xr.Dataset({"x": ("time", s_ser.to_numpy())}, coords={"time": s_ser.index})
    ref_ds = xr.Dataset({"x_ref": ("time", r_ser.to_numpy())}, coords={"time": r_ser.index})
    rep = compare_series(sim_ds, ref_ds, {"x": "x_ref"})
    row = rep.get("x")
    e = _exact(joined["x"].to_numpy(), joined["x_ref"].to_numpy()) if len(joined) >= 2 else None
    assert row.n == len(joined)
    if e is not None:
        assert row.rmse == pytest.approx(e["rmse"], rel=1e-9)
        assert row.bias == pytest.approx(e["bias"], rel=1e-9, abs=1e-12 * abs(e["ref_mean"]))
    common = xr.align(sim_ds, ref_ds, join="inner")[0]["time"]
    wd = xr.DataArray(pd.DatetimeIndex(common.values).dayofweek < 5, coords={"time": common})
    row_w = compare_series(sim_ds, ref_ds, {"x": "x_ref"}, where=wd).get("x")
    sub = joined.loc[mask_days]
    assert row_w.n == len(sub)
    if len(sub) >= 2:
        assert row_w.mae == pytest.approx(
            _exact(sub["x"].to_numpy(), sub["x_ref"].to_numpy())["mae"], rel=1e-9
        )
