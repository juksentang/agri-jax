"""DSSAT readers/writers against real files (skipped when absent).

Inputs: MZCER040.* and DSSATWTH.WTH of the RZWQM CA-TPA scenario, the dssat-csm-data example
tree (UFGA8201.MZX, UFGA8201.WTH, SOIL.SOL) and the UFGA8201 ``dscsm048`` run of the session fixture
``ufga_run`` (tests/integration/conftest.py, private temporary directory). The unit tier covers the same readers
on inline snippets and the public MZCER048 fixtures (``tests/unit/test_io_dssat.py``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrijax.io.dssat import (
    read_cul,
    read_eco,
    read_et,
    read_filex,
    read_plantgro,
    read_soilwat,
    read_sol,
    read_spe,
    read_summary,
    read_wth,
    write_sol,
    write_wth,
)

DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
CATPA = Path("narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario")


def _need(p: Path) -> Path:
    if not p.exists():
        pytest.skip(f"{p} not found")
    return p


@pytest.fixture(scope="module")
def catpa(data_dir: Path) -> Path:
    return _need(data_dir / CATPA)


# --------------------------------------------------------------------------- weather


def test_read_wth_catpa(catpa: Path) -> None:
    w = read_wth(_need(catpa / "DSSATWTH.WTH"))
    assert list(w.columns) == ["date", "srad", "tmax", "tmin", "rain", "rhum", "wind"]
    assert w["date"].iloc[0] == pd.Timestamp("2015-01-01")
    assert w["date"].iloc[-1] == pd.Timestamp("2023-12-31")
    assert len(w) == (w["date"].iloc[-1] - w["date"].iloc[0]).days + 1
    assert w.notna().all().all()
    # 2015-11-13: RHUM and WIND touch in the file ("77.41004.0")
    row = w.loc[w["date"] == pd.Timestamp("2015-11-13")].iloc[0]
    assert (row["rhum"], row["wind"]) == (77.4, 1004.0)
    assert (w["tmax"] >= w["tmin"]).all()
    site = w.attrs["site"]
    assert site["LAT"] == pytest.approx(42.695) and site["INSI"] is None and site["TAV"] is None


def test_read_wth_ufga_and_roundtrip(tmp_path: Path) -> None:
    src = _need(DSSAT_ENGINE / "example_data" / "Weather" / "UFGA8201.WTH")
    w = read_wth(src)
    assert "par" in w.columns and w.attrs["site"]["INSI"] == "UFGA"
    assert w["date"].iloc[0] == pd.Timestamp("1982-01-01")
    for four in (False, True):
        out = write_wth(w, tmp_path / f"X{int(four)}.WTH", four_digit_year=four)
        w2 = read_wth(out)
        pd.testing.assert_frame_equal(w, w2, check_exact=False, atol=1e-9)
        assert w2.attrs["site"] == w.attrs["site"]


# --------------------------------------------------------------------------- soil


def test_read_sol_ibmz910014(tmp_path: Path) -> None:
    sol = read_sol(_need(DSSAT_ENGINE / "example_data" / "Soil" / "SOIL.SOL"))
    p = sol["IBMZ910014"]
    assert (p.source, p.depth, p.description) == ("Gainesville", 180.0, "Millhopper Fine Sand")
    assert p.site["COUNTRY"] == "USA" and p.site["LAT"] == pytest.approx(29.63)
    assert p.site["SCS FAMILY"].startswith("Loamy,silic")
    assert p.surface["SLRO"] == 60.0 and p.surface["SMHB"] == "IB001"
    lay = p.layers
    assert list(lay["SLB"]) == [5, 15, 30, 60, 90, 120, 150, 180]
    np.testing.assert_allclose(lay["SDUL"].to_numpy()[:3], [0.096, 0.086, 0.086])
    np.testing.assert_allclose(lay["SLLL"].iloc[-1], 0.070)
    np.testing.assert_allclose(lay["SSAT"].iloc[-1], 0.360)
    assert bool(lay["SSKS"].isna().to_numpy().all())  # -99 in the file
    for col in ("SLB", "SLLL", "SDUL", "SSAT", "SRGF", "SSKS", "SBDM", "SLOC", "SLCL", "SLSI"):
        assert col in lay.columns
    assert (lay["SLLL"] < lay["SDUL"]).all() and (lay["SDUL"] < lay["SSAT"]).all()


@pytest.mark.filterwarnings("ignore:.*pairs the tables by row:UserWarning")  # CN.SOL, ET.SOL
def test_write_sol_roundtrip_all_examples(tmp_path: Path) -> None:
    files = sorted((_need(DSSAT_ENGINE / "example_data" / "Soil")).glob("*.SOL"))
    n = 0
    for f in files[:12]:
        a = read_sol(f)
        b = read_sol(write_sol(a, tmp_path / f.name))
        assert a.keys() == b.keys()
        for k in a:
            pa, pb = a[k], b[k]
            assert (pa.source, pa.texture, pa.description) == (pb.source, pb.texture, pb.description)
            assert pa.depth == pb.depth or (math.isnan(pa.depth) and math.isnan(pb.depth))
            pd.testing.assert_frame_equal(pa.layers, pb.layers)
            assert pa.surface == pb.surface
            n += 1
    assert n > 0


# --------------------------------------------------------------------------- genotype


def test_read_cul_040(catpa: Path) -> None:
    c = read_cul(_need(catpa / "MZCER040.CUL"))
    assert list(c.index) == ["IB0012"]
    r = c.loc["IB0012"]
    assert r["VRNAME"] == "PIO 3382" and r["ECO#"] == "IB0001"
    assert (r["P1"], r["P2"], r["P5"], r["G2"], r["G3"], r["PHINT"]) == (
        250.0, 0.7, 890.0, 825.0, 8.5, 38.9
    )  # fmt: skip
    assert (r["HTMAX"], r["BIOHALF"]) == (244.6, 43.07)


def test_read_eco_040(catpa: Path) -> None:
    e0 = read_eco(_need(catpa / "MZCER040.ECO"))
    assert list(e0.columns) == [
        "ECONAME", "TBASE", "TOPT", "ROPT", "P20", "DJTI", "GDDE", "DSGFT", "RUE", "KCAN"
    ]  # fmt: skip
    assert e0.loc["IB0001", "P20"] == 12.5 and e0.loc["IB0002", "RUE"] == 4.5


def test_read_spe_040(catpa: Path) -> None:
    s0 = read_spe(_need(catpa / "MZCER040.SPE"))
    assert s0["PRFTC"] == (6.2, 16.5, 33.0, 44.0)
    assert s0["RGFIL"] == (5.5, 16.0, 39.0, 48.5)
    assert s0["PARSR"] == 0.5 and len(s0["CO2X"]) == len(s0["CO2Y"]) == 10  # type: ignore[arg-type]
    assert "TEMPERATURE EFFECTS" in s0.sections


# --------------------------------------------------------------------------- FileX


def test_read_filex_ufga8201() -> None:
    x = read_filex(_need(DSSAT_ENGINE / "example_data" / "Maize" / "UFGA8201.MZX"))
    trt = x["TREATMENTS"]
    assert [t["N"] for t in trt] == [1, 2, 3, 4, 5, 6]
    assert trt[3]["TNAME"] == "IRRIGATED HIGH NITROGEN" and trt[3]["MI"] == 2 and trt[3]["MF"] == 2
    assert x["CULTIVARS"][1] == {"CR": "MZ", "INGENO": "IB0035", "CNAME": "McCurdy 84aa"}
    f = x["FIELDS"][1]
    assert (f["ID_FIELD"], f["WSTA"], f["ID_SOIL"], f["SLDP"]) == ("UFGA0002", "UFGA", "IBMZ910014", 180)
    assert f["XCRD"] == pytest.approx(29.638) and f["FLNAME"] == "Field section"
    pl = x["PLANTING DETAILS"][1]
    assert (pl["PDATE"], pl["PPOP"], pl["PLME"], pl["PLRS"], pl["PLDP"]) == (82057, 7.2, "S", 61, 7)
    sc = x["SIMULATION CONTROLS"][1]
    assert sc["GENERAL"]["SDATE"] == 82056 and sc["GENERAL"]["START"] == "S"
    assert sc["OPTIONS"]["WATER"] == "Y" and sc["METHODS"]["EVAPO"] == "R"
    assert sc["MANAGEMENT"]["IRRIG"] == "R" and sc["HARVEST"]["HLAST"] == 83057
    ic = x["INITIAL CONDITIONS"][1]
    assert ic["ICDAT"] == 82056 and [r["ICBL"] for r in ic["rows"]][-1] == 180
    irr = x["IRRIGATION AND WATER MANAGEMENT"]
    assert len(irr[2]["rows"]) == 16 and irr[1]["rows"] == [{"IDATE": 82063, "IROP": "IR001", "IRVAL": 13}]


# --------------------------------------------------------------------------- run outputs


def test_read_outputs_ufga_run(ufga_run: Path) -> None:
    s = read_summary(ufga_run / "Summary.OUT")
    assert list(s["TRNO"]) == [1, 2, 3, 4, 5, 6]
    assert s["TNAM"].iloc[0] == "RAINFED LOW NITROGEN"
    assert s["HWAM"].dtype.kind == "i" and (s["HWAM"] > 0).all()
    # every numeric column after SOIL_ID must equal a plain whitespace split of that tail
    lines = [ln for ln in (ufga_run / "Summary.OUT").read_text().splitlines() if ln[:9].strip().isdigit()]
    cols = list(s.columns)
    tail = cols[cols.index("SOIL_ID") + 1 :]
    tail_values = s[tail].to_numpy(dtype=float)
    for ln, got in zip(lines, tail_values, strict=True):
        toks = ln.split()
        ref = toks[len(toks) - len(tail) :]
        for r, g in zip(ref, got, strict=True):
            rv = float(r)
            assert (np.isnan(g) and rv == -99) or float(g) == pytest.approx(rv), (r, g)

    g = read_plantgro(ufga_run / "PlantGro.OUT")
    assert set(g["TRNO"]) == {1, 2, 3, 4, 5, 6}
    last = g.groupby("TRNO").tail(1).set_index("TRNO")
    np.testing.assert_array_equal(last["GWAD"].to_numpy(), s.set_index("TRNO")["HWAM"].to_numpy())
    lmax = np.asarray(g.groupby("TRNO")["LAID"].max(), dtype=float)
    np.testing.assert_allclose(lmax, s["LAIX"].to_numpy(), atol=0.051)

    w = read_soilwat(ufga_run / "SoilWat.OUT")
    assert {"SWTD", "PREC", "SW1D", "DATE"} <= set(w.columns)
    assert (w["SW1D"].between(0.0, 0.5)).all()

    e = read_et(ufga_run / "ET.OUT")
    assert {"EOAA", "ETAA", "EPAA", "ESAA"} <= set(e.columns)
    assert (e["ETAA"] <= e["EOAA"] + 1e-6).all()
    # cumulative ET over the season ~ Summary ETCM (mm)
    etcm = np.asarray(e.groupby("TRNO")["ETAA"].sum(), dtype=float)
    np.testing.assert_allclose(etcm, s["ETCM"].to_numpy(), rtol=0.02, atol=2)
