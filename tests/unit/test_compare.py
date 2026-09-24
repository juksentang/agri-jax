"""agrijax.port.compare on synthetic series: metric values, alignment, masks, tolerances, Markdown."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from agrijax.port.compare import CompareReport, Tolerance, compare_series, series_metrics


def _ds(**vars_: np.ndarray) -> xr.Dataset:
    n = len(next(iter(vars_.values())))
    t = pd.date_range("2015-01-01", periods=n, freq="D")
    return xr.Dataset({k: ("time", np.asarray(v, float), {"units": "mm d-1"}) for k, v in vars_.items()},
                      coords={"time": t})  # fmt: skip


def test_series_metrics_known_values() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    sim = ref + np.array([0.5, -0.5, 0.5, 1.5])
    m = series_metrics(sim, ref)
    assert m["n"] == 4
    assert m["bias"] == pytest.approx(0.5)
    assert m["mae"] == pytest.approx(0.75)
    assert m["rmse"] == pytest.approx(np.sqrt((0.25 + 0.25 + 0.25 + 2.25) / 4))
    assert m["max_abs"] == pytest.approx(1.5)
    assert m["pbias"] == pytest.approx(100 * 2.0 / 10.0)
    assert m["r2"] == pytest.approx(np.corrcoef(sim, ref)[0, 1] ** 2)
    assert m["nse"] == pytest.approx(1 - 3.0 / 5.0)


def test_series_metrics_perfect_linear_and_constant() -> None:
    ref = np.linspace(0, 10, 50)
    m = series_metrics(ref, ref)
    assert m["rmse"] == 0.0 and m["r2"] == pytest.approx(1.0) and m["nse"] == pytest.approx(1.0)
    # an affine transform keeps R2 = 1 but not NSE
    m2 = series_metrics(2 * ref + 1, ref)
    assert m2["r2"] == pytest.approx(1.0) and m2["nse"] < 0.0
    # constant reference: R2 and NSE undefined, PBIAS undefined for a zero-sum reference
    m3 = series_metrics(np.ones(5), np.zeros(5))
    assert np.isnan(m3["r2"]) and np.isnan(m3["nse"]) and np.isnan(m3["pbias"])
    assert m3["bias"] == 1.0


def test_series_metrics_drops_nan_pairs_and_checks_length() -> None:
    m = series_metrics([1.0, np.nan, 3.0, 4.0], [1.0, 2.0, np.nan, 5.0])
    assert m["n"] == 2 and m["bias"] == pytest.approx(-0.5)
    empty = series_metrics([np.nan], [1.0])
    assert empty["n"] == 0 and np.isnan(empty["rmse"])
    with pytest.raises(ValueError):
        series_metrics([1.0, 2.0], [1.0])


def test_compare_series_aligns_on_time_and_maps_names() -> None:
    rng = np.random.default_rng(0)
    ref = _ds(obs=rng.normal(size=30))
    sim = _ds(a=ref.obs.values + 0.1).isel(time=slice(5, None))  # starts 5 days later
    rep = compare_series(sim, ref, {"a": "obs"}, period="all")
    m = rep["a"]
    assert m.n == 25 and m.ref_variable == "obs" and m.units == "mm d-1"
    assert m.bias == pytest.approx(0.1) and m.rmse == pytest.approx(0.1)
    assert m.r2 == pytest.approx(1.0)
    assert m.passed is None and rep.passed


def test_compare_series_default_vars_and_dataarray_input() -> None:
    ref = _ds(x=np.arange(10.0), y=np.ones(10), only_ref=np.zeros(10))
    sim = _ds(x=np.arange(10.0) + 1.0, y=np.ones(10))
    rep = compare_series(sim, ref)
    assert [m.variable for m in rep] == ["x", "y"]
    rep_da = compare_series(sim["x"], ref["x"])
    assert rep_da["x"].bias == pytest.approx(1.0)
    with pytest.raises(KeyError):
        compare_series(sim, ref, ["only_ref"])
    with pytest.raises(TypeError):
        compare_series(np.zeros(3), ref)  # type: ignore[arg-type]


def test_compare_series_where_slice_and_mask() -> None:
    err = np.r_[np.zeros(10), np.full(10, 2.0)]
    ref = _ds(v=np.arange(20.0))
    sim = _ds(v=np.arange(20.0) + err)
    first = compare_series(sim, ref, ["v"], where=slice("2015-01-01", "2015-01-10"), period="first")
    assert first["v"].n == 10 and first["v"].rmse == 0.0
    mask = xr.DataArray(err > 0, coords={"time": ref.time}, dims="time")
    second = compare_series(sim, ref, ["v"], where=mask, period="second")
    assert second["v"].n == 10 and second["v"].bias == pytest.approx(2.0)
    with pytest.raises(ValueError):
        compare_series(sim, ref, ["v"], where=np.ones(3, bool))
    both = first + second
    assert both.periods == ["first", "second"]
    assert both["v", "second"].bias == pytest.approx(2.0)
    with pytest.raises(KeyError):
        both["v"]  # ambiguous without the period


def test_tolerances_pass_fail_and_float_shorthand() -> None:
    ref = _ds(a=np.arange(10.0), b=np.arange(10.0))
    sim = _ds(a=np.arange(10.0) + 0.05, b=np.arange(10.0) + 1.0)
    rep = compare_series(sim, ref, ["a", "b"], {"a": 0.1, "b": Tolerance(rmse=0.5, r2_min=0.9)})
    assert rep["a"].passed is True
    assert rep["b"].passed is False and rep["b"].failed == ("rmse",)
    assert not rep.passed
    one_for_all = compare_series(
        sim, ref, ["a", "b"], Tolerance(abs_bias=0.1, nse_min=0.95)
    )  # b: NSE = 1 - 10/82.5
    assert one_for_all["a"].passed and one_for_all["b"].failed == ("abs_bias", "nse_min")
    # a NaN metric fails any limit set on it
    const = compare_series(_ds(c=np.ones(5)), _ds(c=np.ones(5)), ["c"], Tolerance(r2_min=0.5))
    assert const["c"].failed == ("r2_min",)
    assert Tolerance(rmse=0.1, r2_min=0.9).describe() == "rmse<=0.1, r2>=0.9"


def test_markdown_rendering() -> None:
    ref = _ds(a=np.arange(10.0), b=np.arange(10.0))
    sim = _ds(a=np.arange(10.0) + 0.05, b=np.arange(10.0) + 1e-9)
    rep = compare_series(sim, ref, ["a", "b"], {"a": 0.01}, period="year", title="Synthetic")
    rep.notes.append("synthetic data")
    md = rep.to_markdown()
    lines = md.splitlines()
    assert lines[0] == "## Synthetic"
    header = next(ln for ln in lines if ln.startswith("| period"))
    assert "RMSE" in header and "R²" in header and "status" in header
    rows = [ln for ln in lines if ln.startswith("| year")]
    assert len(rows) == 2
    assert "FAIL (rmse)" in rows[0] and rows[1].rstrip().endswith("|  |  |")
    assert "e-" in rows[1]  # tiny numbers in scientific notation
    assert lines[-1] == "- synthetic data"
    ncol = header.count("|")
    assert all(ln.count("|") == ncol for ln in lines if ln.startswith("|"))
    df = rep.to_dataframe()
    assert list(df["variable"]) == ["a", "b"] and df.loc[0, "passed"] is False
    with pytest.raises(ValueError):
        rep.to_markdown(metrics=("n", "nope"))
    assert str(CompareReport()).startswith("| period")
