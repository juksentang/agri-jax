"""agri_jax.io.ameriflux on the real CA-TPA AmeriFlux files under ``<data-dir>/ameriflux``.

BASE HH 3-5 and FLUXNET FLUXMET v1.3_r1 (2020-2023). Every test skips when its file is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_jax.io import ameriflux as af

pytestmark = pytest.mark.allow_skip(reason="needs the private AmeriFlux CA-TPA files under the data dir")

BASE_REL = Path("ameriflux/base/AMF_CA-TPA_BASE_HH_3-5.csv")
FLUXNET_REL = Path("ameriflux/fluxnet")
HEIGHTS_REL = Path("ameriflux/CA-TPA_measurement_height.csv")
POLICY_REL = FLUXNET_REL / "DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt"


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"{p} not found")
    return p


@pytest.fixture(scope="module")
def base(data_dir: Path) -> pd.DataFrame:
    return af.read_base_hh(_need(data_dir / BASE_REL))


@pytest.fixture(scope="module")
def dd(data_dir: Path) -> pd.DataFrame:
    d = _need(data_dir / FLUXNET_REL)
    try:
        return af.read_fluxnet(d, "DD")
    except FileNotFoundError as e:
        pytest.skip(str(e))


def test_base_rows_and_units(base: pd.DataFrame) -> None:
    assert len(base) == 70128  # 1461 days * 48
    assert base.index[0] == pd.Timestamp("2020-01-01 00:00")
    assert base.index[-1] == pd.Timestamp("2023-12-31 23:30")
    assert (pd.Series(base.index).diff().iloc[1:] == pd.Timedelta("30min")).all()
    assert base.attrs["site"] == "CA-TPA" and base.attrs["version"] == "3-5"
    assert base.attrs["units"]["SWC_1_2_1"].startswith("%")
    # SWC is % (0-100), not m3 m-3
    swc = base[["SWC_1_2_1", "SWC_1_3_1", "SWC_1_5_1"]]
    top = float(np.nanmax(swc.to_numpy()))
    assert 1.0 < top <= 100.0
    assert float(np.nanmedian(base["PA"].to_numpy())) == pytest.approx(99.0, abs=2.0)  # kPa


def test_base_swc_coverage_2021(base: pd.DataFrame) -> None:
    cov = float(base.loc["2021-01-01":"2021-12-31", "SWC_1_2_1"].notna().mean())
    print(f"SWC_1_2_1 coverage 2021: {cov:.3f}")
    assert cov > 0.85


def test_measurement_heights_swc_depths(data_dir: Path, base: pd.DataFrame) -> None:
    hts = af.read_measurement_heights(_need(data_dir / HEIGHTS_REL), site="CA-TPA")
    tab = af.base_variable_positions(base.columns, hts)
    assert tab.loc[["SWC_1_2_1", "SWC_1_3_1", "SWC_1_5_1"], "height_m"].tolist() == pytest.approx(
        [-0.05, -0.20, -0.50]
    )
    for col, depth in af.CATPA_DEPTH_M.items():
        assert tab.at[col, "depth_m"] == pytest.approx(depth), col


def test_fluxnet_dd_rows_and_le_corr_days(dd: pd.DataFrame) -> None:
    assert len(dd) == 1461
    assert dd.attrs["units"].get("LE_CORR") == "W m-2"
    assert dd.attrs["units"].get("SWC_F_MDS_1") == "%"
    years = np.asarray(dd.index.values).astype("datetime64[Y]").astype(int) + 1970
    days = {y: int(dd["LE_CORR"].notna().to_numpy()[years == y].sum()) for y in range(2020, 2024)}
    assert days == {2020: 206, 2021: 365, 2022: 365, 2023: 365}


def test_fluxnet_hh_row_count(data_dir: Path) -> None:
    hh = af.read_fluxnet(_need(data_dir / FLUXNET_REL), "HH", columns=["LE_F_MDS_QC"])
    assert len(hh) == 70128


def test_annual_et_2021(dd: pd.DataFrame) -> None:
    et = af.daily_et(dd, "LE_CORR")["et_mm"]
    annual = float(et.loc["2021-01-01":"2021-12-31"].sum())
    may_sep = float(et.loc["2021-05-01":"2021-09-30"].sum())
    print(f"2021 ET (LE_CORR, lambda 2.45): annual {annual:.1f} mm, May-Sep {may_sep:.1f} mm")
    assert annual == pytest.approx(658.0, abs=2.0)
    assert may_sep == pytest.approx(450.0, abs=2.0)


def test_closure_2021_base(base: pd.DataFrame) -> None:
    c = af.energy_balance_closure(base, "Y")
    r = c.loc[pd.Timestamp("2021-01-01")]
    print(
        f"2021 closure (BASE HH, all four terms present): ratio {r['ratio']:.3f}, "
        f"slope {r['slope']:.3f}, intercept {r['intercept']:.1f} W m-2, n {int(r['n'])}"
    )
    assert 0.6 < r["ratio"] < 1.1
    daily = af.energy_balance_closure(base.loc["2021-07-01":"2021-07-31"], "D")
    assert len(daily) == 31 and (daily["n"] <= 48).all()


def test_catpa_validation_set(data_dir: Path) -> None:
    _need(data_dir / FLUXNET_REL)
    v = af.catpa_validation_set(data_dir, years=(2020, 2021))
    assert set(v["season"]) == {2020, 2021}
    assert v.index[0] == pd.Timestamp(af.CATPA_FLUX_START)
    s21 = v.loc[v["season"] == 2021]
    assert s21.index[0] == pd.Timestamp("2021-05-05") and s21.index[-1] == pd.Timestamp("2021-10-20")
    assert len(s21) == 169 and not np.isnan(np.asarray(s21["et_mm"], dtype=float)).any()
    assert ((v["qc"] >= 0) & (v["qc"] <= 1)).all()
    q = v[["et_lo", "et_hi"]].dropna()
    assert len(q) > 0.9 * len(v) and (q["et_lo"] <= q["et_hi"]).all()
    print(
        "maize-season ET: "
        + ", ".join(f"{y} {g['et_mm'].sum():.1f} mm / {len(g)} d" for y, g in v.groupby("season"))
    )


def test_acknowledgments_verbatim_in_policy(data_dir: Path) -> None:
    text = " ".join(_need(data_dir / POLICY_REL).read_text(encoding="latin-1").split())
    assert af.ACK_FLUXNET in text
    assert af.ACK_AMERIFLUX in text
    assert af.DATA_AVAILABILITY_TEMPLATE in text
