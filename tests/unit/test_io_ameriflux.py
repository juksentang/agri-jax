"""agri_jax.io.ameriflux on small synthetic CSVs written by the tests (no data).

The checks against the real CA-TPA BASE / FLUXNET files are in
``tests/integration/test_ameriflux_catpa.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agri_jax.io import ameriflux as af

_BASE = """# Site: XX-TST
# Version: 1-2
TIMESTAMP_START,TIMESTAMP_END,LE,H,NETRAD,G_1_1_1,TA,SWC_1_2_1,SWC_1_5_1,TS_1_1_1,P_PI_1,NEE_PI
202107010000,202107010030,10.0,5.0,20.0,5.0,15.5,-9999,30.0,18.0,0,-9999
202107010030,202107010100,-9999,5.0,20.0,5.0,15.0,20.5,30.1,17.9,0.2,-1.5
202107010100,202107010130,40.0,20.0,100.0,20.0,-9999,20.4,30.2,17.8,1.0,-2.5
202107020000,202107020030,30.0,30.0,80.0,0.0,16.0,20.3,-9999,17.7,0,-3.0
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# ----------------------------------------------------------------------------- BASE


def test_read_base_hh_header_missing_units(tmp_path: Path) -> None:
    df = af.read_base_hh(_write(tmp_path, "AMF_XX-TST_BASE_HH_1-2.csv", _BASE))
    assert df.attrs["site"] == "XX-TST" and df.attrs["version"] == "1-2"
    assert df.index.name == "TIMESTAMP_START"
    assert df.index[0] == pd.Timestamp("2021-07-01 00:00") and pd.DatetimeIndex(df.index).tz is None
    assert df["TIMESTAMP_END"].iloc[1] == pd.Timestamp("2021-07-01 01:00")
    assert len(df) == 4 and "TIMESTAMP_START" not in df.columns
    assert np.isnan(df["LE"].iloc[1]) and np.isnan(df["SWC_1_2_1"].iloc[0]) and np.isnan(df["TA"].iloc[2])
    assert df["LE"].iloc[0] == 10.0 and df["P_PI_1"].iloc[2] == 1.0
    assert (df.drop(columns="TIMESTAMP_END") != af.MISSING).all().all()
    u = df.attrs["units"]
    assert u["LE"] == "W m-2" and u["TA"] == "deg C" and u["TS_1_1_1"] == "deg C"
    assert u["SWC_1_2_1"].startswith("%") and u["P_PI_1"].startswith("mm") and u["NEE_PI"].startswith("umol")
    assert df.attrs["positions"]["SWC_1_5_1"] == (1, 5, 1)


def test_read_base_hh_rejects_bad_header(tmp_path: Path) -> None:
    bad = "# Site: X\nTIME,LE\n202101010000,1\n"
    with pytest.raises(ValueError, match="TIMESTAMP_START"):
        af.read_base_hh(_write(tmp_path, "bad.csv", bad))


def test_base_variable_positions_suffix() -> None:
    pos = af.base_variable_positions(["LE", "P_PI_1", "SWC_1_2_1", "TS_2_10_3", "G_1_1_1", "NEE_PI"])
    assert pos == {"SWC_1_2_1": (1, 2, 1), "TS_2_10_3": (2, 10, 3), "G_1_1_1": (1, 1, 1)}


_HEIGHTS = """Site_ID,Variable,Start_Date,Height,Instrument_Model,Instrument_Model2,Comment,BASE_Version
XX-TST,LE,,5.0,GA_OP-LI-COR LI-7500,SA-Campbell CSAT-3,20200629,1-2
XX-TST,SWC_1_2_1,,-0.05,SWC-TDR,,20200629,1-2
XX-TST,SWC_1_5_1,201901010000,-0.40,SWC-TDR,,old,1-2
XX-TST,SWC_1_5_1,202001010000,-0.5,SWC-TDR,,moved,1-2
YY-OTH,SWC_1_2_1,,-0.3,SWC-TDR,,,1-2
"""


def test_read_measurement_heights_and_join(tmp_path: Path) -> None:
    hts = af.read_measurement_heights(_write(tmp_path, "h.csv", _HEIGHTS), site="XX-TST")
    assert set(hts["variable"]) == {"LE", "SWC_1_2_1", "SWC_1_5_1"}
    by = hts.set_index("variable")
    assert by.at["SWC_1_5_1", "height_m"] == -0.5  # latest Start_Date wins
    assert by.at["LE", "instrument"] == "GA_OP-LI-COR LI-7500;SA-Campbell CSAT-3"
    everything = af.read_measurement_heights(tmp_path / "h.csv", site=None)
    assert len(everything) == 4 and set(everything["site"]) == {"XX-TST", "YY-OTH"}
    with pytest.raises(ValueError, match="several sites"):
        af.base_variable_positions(["SWC_1_2_1"], everything)
    tab = af.base_variable_positions(["LE", "SWC_1_2_1", "SWC_1_5_1", "TS_1_1_1"], hts)
    assert list(tab.index) == ["SWC_1_2_1", "SWC_1_5_1", "TS_1_1_1"]
    assert tab.at["SWC_1_2_1", "depth_m"] == pytest.approx(0.05)
    assert tab.at["SWC_1_5_1", "v"] == 5 and tab.at["SWC_1_5_1", "depth_m"] == pytest.approx(0.5)
    assert np.isnan(tab.at["TS_1_1_1", "height_m"])


# ----------------------------------------------------------------------------- FLUXNET

_DD = """TIMESTAMP,TA_F,LE_F_MDS,LE_F_MDS_QC,LE_CORR,LE_CORR_25,LE_CORR_75,LE_RANDUNC,H_F_MDS,H_F_MDS_QC,NETRAD,G_F_MDS,G_F_MDS_QC
20210701,20.0,100.0,1.0,120.0,110.0,130.0,5.0,40.0,1.0,200.0,10.0,1.0
20210702,10.0,50.0,0.5,-9999,-9999,-9999,-9999,20.0,0.5,100.0,0.0,1.0
"""

_HH_ROWS = [
    "TIMESTAMP_START,TIMESTAMP_END,LE_F_MDS,LE_F_MDS_QC,H_F_MDS,H_F_MDS_QC,NETRAD,G_F_MDS,G_F_MDS_QC",
    "202107010000,202107010030,50,0,50,0,150,50,0",
    "202107010030,202107010100,60,1,40,0,150,50,0",
    "202107010100,202107010130,70,0,30,0,-9999,50,0",
    "202107010130,202107010200,80,0,20,0,200,0,0",
]


def test_read_fluxnet_dd_and_units(tmp_path: Path) -> None:
    _write(tmp_path, "AMF_XX-TST_FLUXNET_FLUXMET_DD_2021-2021_v1.3_r1.csv", _DD)
    info = (
        "SITE_ID,GROUP_ID,VARIABLE_GROUP,VARIABLE,DATAVALUE\n"
        "XX-TST,1,GRP_ONEFLUX,PRODUCT_FIRST_YEAR,2021\n"
        "XX-TST,2,GRP_VAR_INFO,VAR_INFO_VARNAME,LE_CORR\n"
        "XX-TST,2,GRP_VAR_INFO,VAR_INFO_UNIT,W m-2\n"
        "XX-TST,2,GRP_VAR_INFO,VAR_INFO_HEIGHT,5\n"
    )
    _write(tmp_path, "AMF_XX-TST_FLUXNET_BIFVARINFO_DD_2021-2021_v1.3_r1.csv", info)
    dd = af.read_fluxnet(tmp_path, "DD")
    assert list(dd.index) == [pd.Timestamp("2021-07-01"), pd.Timestamp("2021-07-02")]
    assert dd.index.name == "TIMESTAMP" and np.isnan(dd["LE_CORR"].iloc[1])
    assert dd.attrs["units"] == {"LE_CORR": "W m-2"} and dd.attrs["resolution"] == "DD"
    assert af.read_bifvarinfo(tmp_path / "AMF_XX-TST_FLUXNET_BIFVARINFO_DD_2021-2021_v1.3_r1.csv").at[
        "LE_CORR", "HEIGHT"
    ] == pytest.approx(5.0)
    with pytest.raises(FileNotFoundError):
        af.read_fluxnet(tmp_path, "MM")
    with pytest.raises(ValueError, match="resolution"):
        af.read_fluxnet(tmp_path, "XX")  # type: ignore[arg-type]


def test_read_fluxnet_other_resolutions(tmp_path: Path) -> None:
    _write(tmp_path, "A_FLUXNET_FLUXMET_HH_x.csv", "\n".join(_HH_ROWS) + "\n")
    _write(
        tmp_path,
        "A_FLUXNET_FLUXMET_WW_x.csv",
        "TIMESTAMP_START,TIMESTAMP_END,LE_F_MDS\n20210101,20210107,1\n",
    )
    _write(tmp_path, "A_FLUXNET_FLUXMET_MM_x.csv", "TIMESTAMP,LE_F_MDS\n202102,1\n")
    _write(tmp_path, "A_FLUXNET_FLUXMET_YY_x.csv", "TIMESTAMP,LE_F_MDS\n2021,1\n")
    hh = af.read_fluxnet(tmp_path, "HH", columns=["LE_F_MDS"])
    assert list(hh.columns) == ["TIMESTAMP_END", "LE_F_MDS"]
    assert hh.index[1] == pd.Timestamp("2021-07-01 00:30")
    assert af.read_fluxnet(tmp_path, "WW")["TIMESTAMP_END"].iloc[0] == pd.Timestamp("2021-01-07")
    assert af.read_fluxnet(tmp_path, "MM").index[0] == pd.Timestamp("2021-02-01")
    assert af.read_fluxnet(tmp_path, "YY").index[0] == pd.Timestamp("2021-01-01")


# ----------------------------------------------------------------------------- ET


def test_daily_et_hand_computation(tmp_path: Path) -> None:
    dd = af.read_fluxnet(_write(tmp_path, "dd.csv", _DD), "DD")
    et = af.daily_et(dd)
    # 120 W m-2 * 86400 s / 2.45e6 J kg-1 = 4.2318... mm
    assert et["et_mm"].iloc[0] == pytest.approx(120.0 * 86400 / 2.45e6)
    assert et["et_mm"].iloc[0] == pytest.approx(4.231836734693878)
    assert et["et_lo"].iloc[0] == pytest.approx(110.0 * 86400 / 2.45e6)
    assert et["et_hi"].iloc[0] == pytest.approx(130.0 * 86400 / 2.45e6)
    assert et["randunc_mm"].iloc[0] == pytest.approx(5.0 * 86400 / 2.45e6)
    assert np.isnan(et["et_mm"].iloc[1]) and et["qc"].tolist() == [1.0, 0.5]
    assert et.attrs["qc_kind"] == "LE_F_MDS_QC_dd"

    mds = af.daily_et(dd, source="LE_F_MDS", lambda_mj_kg=2.5)
    assert mds["et_mm"].tolist() == pytest.approx([100.0 * 86400 / 2.5e6, 50.0 * 86400 / 2.5e6])
    assert bool(mds["et_lo"].isna().all())

    t_dep = af.daily_et(dd, lambda_mj_kg="ta")
    lam = (2.501 - 0.002361 * 20.0) * 1e6
    assert t_dep["et_mm"].iloc[0] == pytest.approx(120.0 * 86400 / lam)
    assert af.latent_heat_mj_kg(0.0) == pytest.approx(2.501)


def test_daily_et_measured_fraction_from_hh(tmp_path: Path) -> None:
    dd = af.read_fluxnet(_write(tmp_path, "dd.csv", _DD), "DD")
    hh = af.read_fluxnet(_write(tmp_path, "hh.csv", "\n".join(_HH_ROWS) + "\n"), "HH")
    et = af.daily_et(dd, hh=hh)
    assert et["qc"].iloc[0] == pytest.approx(0.75)  # 3 of 4 half-hours QC 0
    assert np.isnan(et["qc"].iloc[1]) and et.attrs["qc_kind"] == "measured_fraction_hh"


# ----------------------------------------------------------------------------- closure


def test_energy_balance_closure_base(tmp_path: Path) -> None:
    df = af.read_base_hh(_write(tmp_path, "b.csv", _BASE))
    c = af.energy_balance_closure(df, "D")
    assert list(c.index) == [pd.Timestamp("2021-07-01"), pd.Timestamp("2021-07-02")]
    # day 1: row 2 lacks LE -> (15 + 60) / (15 + 80); day 2: 60 / 80
    assert c["ratio"].tolist() == pytest.approx([75.0 / 95.0, 0.75])
    assert c["n"].tolist() == [2, 1] and c["coverage"].iloc[0] == pytest.approx(2 / 3)
    assert np.isnan(c["slope"].iloc[0])  # fewer than 3 points
    y = af.energy_balance_closure(df, "Y")
    assert y["ratio"].iloc[0] == pytest.approx(135.0 / 175.0) and y["n"].iloc[0] == 3
    with pytest.raises(ValueError, match="window"):
        af.energy_balance_closure(df, "W")  # type: ignore[arg-type]


def test_energy_balance_closure_slope_constructed() -> None:
    idx = pd.date_range("2021-06-01", periods=96, freq="30min")
    avail = np.linspace(-50.0, 500.0, 96)
    g = np.full(96, 10.0)
    turb = 0.8 * avail + 5.0
    df = pd.DataFrame({"H": 0.3 * turb, "LE": 0.7 * turb, "NETRAD": avail + g, "G_1_1_1": g}, index=idx)
    c = af.energy_balance_closure(df, "D")
    assert c["slope"].tolist() == pytest.approx([0.8, 0.8])
    assert c["intercept"].tolist() == pytest.approx([5.0, 5.0])
    day1 = slice(0, 48)
    assert c["ratio"].iloc[0] == pytest.approx(turb[day1].sum() / avail[day1].sum())


def test_energy_balance_closure_fluxnet_hh_measured_only(tmp_path: Path) -> None:
    hh = af.read_fluxnet(_write(tmp_path, "hh.csv", "\n".join(_HH_ROWS) + "\n"), "HH")
    c = af.energy_balance_closure(hh, "D")
    # row 2 gap-filled LE (QC 1), row 3 missing NETRAD -> rows 1 and 4
    assert c["n"].iloc[0] == 2
    assert c["ratio"].iloc[0] == pytest.approx((100.0 + 100.0) / (100.0 + 200.0))
    assert c.attrs["terms"]["LE"] == "LE_F_MDS"


# ----------------------------------------------------------------------------- seasons / policy


def test_growing_season_mask() -> None:
    idx = pd.date_range("2021-05-01", "2021-05-10 12:00", freq="12h")
    m = af.growing_season_mask(idx, "2021-05-03", "2021-05-05")
    assert sorted({d.day for d in idx[m]}) == [3, 4, 5]
    m2 = af.growing_season_mask(idx, ["2021-05-01", "2021-05-09"], ["2021-05-01", "2021-05-10"])
    assert int(m2.sum()) == 6
    with pytest.raises(ValueError, match="before sowing"):
        af.growing_season_mask(idx, "2021-05-05", "2021-05-01")
    with pytest.raises(ValueError, match="sowing dates"):
        af.growing_season_mask(idx, ["2021-05-01"], ["2021-05-02", "2021-05-03"])


def test_citations_and_acknowledgments() -> None:
    assert "10.17190/AMF/2563529" in af.CITATION_BASE and "Ver. 3-5" in af.CITATION_BASE
    assert "10.17190/AMF/3027362" in af.CITATION_FLUXNET and "1.3_r1" in af.CITATION_FLUXNET
    assert af.acknowledgments(fluxnet=False) == af.ACK_AMERIFLUX
    assert af.acknowledgments().startswith("FLUXNET data products were produced")
    s = af.data_availability("the AmeriFlux data portal (https://ameriflux.lbl.gov)", "23 September 2026")
    assert s.endswith(
        "downloaded from the AmeriFlux data portal (https://ameriflux.lbl.gov) on 23 September 2026."
    )


def test_catpa_validation_set_unknown_year(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no maize season"):
        af.catpa_validation_set(tmp_path, years=(2022,))
