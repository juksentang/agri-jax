"""Readers / writers of agrijax.io.rzwqm against the CA-TPA scenario files, the batch parameter
map ``all_parameters.csv`` and the original ``GenerateDat.py`` (skipped when absent).

Moved from the unit tier, which must not depend on the private data tree.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrijax.io.rzwqm import (
    layout_param_map,
    load_param_map,
    param_map_from_csv,
    params_from_dat,
    params_to_dat,
    read_brk,
    read_met,
    read_rzwqm_dat,
    save_param_map,
    set_value,
    write_rzwqm_dat,
)
from agrijax.io.rzwqm.dat import HYDRAULIC_FIELDS
from agrijax.io.rzwqm.params import ParamSpec

BATCH = Path("narval_mirror/RZWQM_sw_batch")
TOOL = Path("narval_mirror/RZWQM_Tool")


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def scenario(data_dir: Path) -> Path:
    p = data_dir / BATCH / "CA-TPA" / "Scenario"
    if not (p / "rzwqm.dat").is_file():
        pytest.skip(f"CA-TPA scenario not found at {p}")
    return p


@pytest.fixture(scope="module")
def param_csv(data_dir: Path) -> Path:
    p = data_dir / BATCH / "all_parameters.csv"
    if not p.is_file():
        pytest.skip(f"{p} not found")
    return p


@pytest.fixture(scope="module")
def generate_dat(data_dir: Path):
    """The original GenerateDat.py module (imported from the RZWQM_Tool mirror)."""
    p = data_dir / TOOL / "LHS_ana_Gen" / "GenerateDat.py"
    if not p.is_file():
        pytest.skip(f"{p} not found")
    spec = importlib.util.spec_from_file_location("_generatedat_ref", p)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_generatedat_ref"] = mod
    spec.loader.exec_module(mod)
    return mod


def _generatedat_output(mod, src: Path, rows: list[dict], out: Path) -> bytes:
    """Exactly what GenerateDat.main writes for one sample of one .dat source."""
    content = mod.apply_sample(str(src), rows, False)
    with open(out, "w", encoding="latin-1") as f:
        f.writelines(content)
    return out.read_bytes()


# ----------------------------------------------------------------------------- rzwqm.dat
def test_dat_roundtrip_byte_identical(scenario: Path, tmp_path: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    out = write_rzwqm_dat(dat, tmp_path / "rzwqm.dat")
    assert out.read_bytes() == (scenario / "rzwqm.dat").read_bytes()
    assert dat.newline == "\r\n"
    assert len(dat.lines) == 1095


def test_dat_parsed_sections(scenario: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    phys = dat.physiography
    assert phys["latitude_rad"] == pytest.approx(0.745163)
    assert phys["co2_ppm"] == 330.0
    assert dat.n_horizon == 5 and dat.profile_depth_cm == 150.0
    np.testing.assert_array_equal(dat.horizon_depths_cm, [15.0, 30.0, 70.0, 90.0, 150.0])
    assert dat.n_node == 37
    assert dat.node_depths_cm[0] == 1.0 and dat.node_depths_cm[-1] == 150.0
    assert np.all(np.diff(dat.node_depths_cm) > 0)
    assert dat.node_spacing_cm[-1] == 0.0

    sp = dat.soil_physical
    assert sp["texture"] == ["Loamy sand"] * 5
    np.testing.assert_allclose(sp["bulk_density"], 1.45)

    hyd = dat.hydraulics
    assert set(hyd) == set(HYDRAULIC_FIELDS)
    assert all(v.shape == (5,) for v in hyd.values())
    np.testing.assert_allclose(hyd["lam"], [0.22, 0.26, 0.36, 0.17, 0.322])
    np.testing.assert_allclose(hyd["ksat"], [5.41, 3.16, 3.31, 3.32, 2.59])
    np.testing.assert_allclose(hyd["theta_r"], [0.055, 0.032, 0.043, 0.048, 0.041])
    np.testing.assert_allclose(hyd["theta_s"], 0.453)
    np.testing.assert_allclose(hyd["theta_fc33"][0], 0.255198)
    np.testing.assert_allclose(hyd["theta_wp"][4], 0.085222)
    np.testing.assert_allclose(hyd["hb"], 14.6545)
    np.testing.assert_allclose(hyd["c2"], 7440.01)
    np.testing.assert_allclose(hyd["ksat_lat"], 2.59)
    assert np.all(hyd["theta_r"] < hyd["theta_wp"])
    assert np.all(hyd["theta_wp"] < hyd["theta_fc33"])
    assert np.all(hyd["theta_fc33"] < hyd["theta_s"])

    pet = dat.pet
    assert (pet["albedo_dry"], pet["albedo_wet"], pet["albedo_crop"], pet["albedo_residue"]) == (
        0.68,
        0.74,
        0.88,
        0.31,
    )
    assert pet["soil_resistance"] == 54.0 and pet["pet_method"] == 0.0

    assert dat.plants == ["7000  maize IB0012 PIO 3382"]
    assert dat.plant_site_params[0]["rs_min"] == 224.0

    pl = dat.plantings
    assert len(pl) == 20
    assert pl[0].planting_date == np.datetime64("2002-04-28")
    assert pl[0].harvest_date == np.datetime64("2002-09-20")
    assert pl[-1].planting_date == np.datetime64("2021-05-05")
    assert pl[-1].harvest_date == np.datetime64("2021-10-20")
    assert all(p.density_seeds_ha == 80000.0 and p.row_spacing_cm == 76.0 for p in pl)
    assert all(p.harvest_option == 3 and p.harvest_type == 3 for p in pl)
    assert dat.get(*pl[0].density_address) == "80000.0"


def test_dat_addresses_match_all_parameters_csv(scenario: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    h = dat.hydraulic_addresses()
    assert h["lam"][:4] == [(135, 2), (138, 2), (141, 2), (144, 2)]
    assert h["ksat"][0] == (135, 4) and h["theta_r"][0] == (135, 5)
    assert h["theta_fc33"][0] == (136, 0) and h["theta_fc10"][0] == (136, 1) and h["theta_wp"][0] == (136, 2)
    assert dat.pet_addresses()["albedo_dry"] == (291, 0)
    assert dat.pet_addresses()["soil_resistance"] == (291, 10)
    assert dat.plant_site_addresses()[0]["rs_min"] == (545, 7)


def test_set_value_semantics(scenario: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    orig = list(dat.lines)
    new = set_value(dat, 135, 2, 0.4)
    assert dat.lines == orig  # not in place by default
    assert new.lines[134] == "1 14.6545 0.400 2.966 5.41 0.055 0.453\r\n"
    assert new.hydraulics["lam"][0] == 0.4
    assert sum(a != b for a, b in zip(new.lines, orig, strict=True)) == 1
    same = set_value(dat, 291, 10, 300.0, inplace=True)
    assert same is dat and dat.pet["soil_resistance"] == 300.0
    with pytest.raises(IndexError):
        set_value(dat, 135, 7, 1.0)
    with pytest.raises(IndexError):
        set_value(dat, 10_000, 0, 1.0)
    with pytest.raises(ValueError, match="comment"):
        set_value(dat, 1, 0, 1.0)


def test_set_value_equals_generatedat_single_edit(scenario: Path, generate_dat, tmp_path: Path) -> None:
    src = scenario / "rzwqm.dat"
    value = float(np.round(0.3456789, 5))
    ref = _generatedat_output(
        generate_dat, src, [{"line_number": 135, "location_at_line": 2, "value": value}], tmp_path / "g.dat"
    )
    ours = write_rzwqm_dat(set_value(read_rzwqm_dat(src), 135, 2, value), tmp_path / "o.dat", newline="\n")
    assert ours.read_bytes() == ref


def test_params_to_dat_equals_generatedat_lhs(
    scenario: Path, param_csv: Path, generate_dat, tmp_path: Path
) -> None:
    """Three LHS samples of all 30 CA-TPA parameters: our write-back == GenerateDat's files."""
    src = scenario / "rzwqm.dat"
    specs = param_map_from_csv(param_csv, "CA-TPA")
    assert len(specs) == 30
    setting = {
        "total_iteration": 3,
        "target_parameter": {
            s.source_name: {
                "minimum_value": s.minimum,
                "maximum_value": s.maximum,
                "parameter_file_path": "rzwqm.dat",
                "line_number": s.line_number,
                "location_at_line": s.location_at_line,
            }
            for s in specs
        },
    }
    lhs, order = generate_dat.latin_hypercube_sampling(setting)
    by_source = {s.source_name: s for s in specs}
    dat = read_rzwqm_dat(src)
    for i in range(lhs.shape[0]):
        rows = [
            {
                "line_number": by_source[n].line_number,
                "location_at_line": by_source[n].location_at_line,
                "value": lhs[i, j],
            }
            for j, n in enumerate(order)
        ]
        ref = _generatedat_output(generate_dat, src, rows, tmp_path / f"{i}_g.dat")
        params = {by_source[n].name: lhs[i, j] for j, n in enumerate(order)}
        new = params_to_dat(dat, params, specs)
        ours = write_rzwqm_dat(new, tmp_path / f"{i}_o.dat", newline="\n")
        assert ours.read_bytes() == ref, f"sample {i}"
        back = params_from_dat(new, specs)
        for j, n in enumerate(order):
            s = by_source[n]
            got = back[s.field][s.horizon - 1] if s.horizon is not None else back[s.name]
            assert float(got) == pytest.approx(lhs[i, j], abs=1e-12)


# ----------------------------------------------------------------------------- params


def test_params_from_dat_with_csv_map(scenario: Path, param_csv: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    specs = param_map_from_csv(param_csv, "CA-TPA")
    assert sorted({s.horizon for s in specs if s.horizon}) == [1, 2, 3, 4]
    p = params_from_dat(dat, specs)
    for f in HYDRAULIC_FIELDS:
        assert p[f].shape == (5,)
    np.testing.assert_allclose(p["ksat"], [5.41, 3.16, 3.31, 3.32, 2.59])
    assert float(p["albedo_dry"]) == 0.68 and float(p["soil_resistance"]) == 54.0
    assert float(p["rs_min_corn"]) == 224.0


def test_param_map_rejects_stale_address(scenario: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    bad = [ParamSpec(name="lam_1", field="lam", line_number=138, location_at_line=2, horizon=1)]
    with pytest.raises(ValueError, match="hydraulic block"):
        params_from_dat(dat, bad)
    bad_pet = [ParamSpec(name="albedo_dry", field="albedo_dry", line_number=291, location_at_line=1)]
    with pytest.raises(ValueError, match="PET block"):
        params_from_dat(dat, bad_pet)


def test_params_roundtrip_full_layout(scenario: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    p = params_from_dat(dat)
    new = params_to_dat(dat, p, decimals=None)
    for f in HYDRAULIC_FIELDS:
        np.testing.assert_array_equal(params_from_dat(new)[f], p[f])
    lines_changed = {i for i, (a, b) in enumerate(zip(dat.lines, new.lines, strict=True)) if a != b}
    hyd = dat.section("hydraulics")
    assert all(hyd.start - 1 <= i <= hyd.stop - 1 for i in lines_changed)

    p2 = {k: v.copy() for k, v in p.items()}
    p2["ksat"][4] = 1.23456789
    new2 = params_to_dat(dat, p2)
    assert new2.hydraulics["ksat"][4] == 1.23457
    np.testing.assert_array_equal(new2.hydraulics["ksat"][:4], p["ksat"][:4])


def test_param_map_yaml_roundtrip(scenario: Path, param_csv: Path, tmp_path: Path) -> None:
    specs = param_map_from_csv(param_csv, "CA-TPA")
    back = load_param_map(save_param_map(specs, tmp_path / "map.yaml"))
    assert back == specs
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    full = layout_param_map(dat)
    assert len(full) == len(HYDRAULIC_FIELDS) * 5
    assert load_param_map(save_param_map(full, tmp_path / "full.yaml")) == full


def _all_scenarios(root: Path) -> list[Path]:
    return sorted(root.glob("*/Scenario/rzwqm.dat"))


def test_every_batch_scenario_parses_and_matches_csv(data_dir: Path, param_csv: Path, tmp_path: Path) -> None:
    files = _all_scenarios(data_dir / BATCH)
    if len(files) < 2:
        pytest.skip("batch scenarios not mirrored")
    for f in files:
        name = f.parent.parent.name
        dat = read_rzwqm_dat(f)
        assert write_rzwqm_dat(dat, tmp_path / f"{name}.dat").read_bytes() == f.read_bytes(), name
        specs = param_map_from_csv(param_csv, name)
        p = params_from_dat(dat, specs)  # raises if any CSV address disagrees with the layout
        assert all(p[k].shape == (dat.n_horizon,) for k in HYDRAULIC_FIELDS), name
        assert len(dat.node_depths_cm) == dat.n_node and dat.horizon_depths_cm[-1] == dat.profile_depth_cm
        assert len(dat.plant_site_params) == len(dat.plants), name
        assert len(dat.plantings) >= 1, name


# ----------------------------------------------------------------------------- MET / BRK
def test_read_met(scenario: Path) -> None:
    met = read_met(scenario / "CA-TPA.MET")
    assert list(met.columns) == ["tmin", "tmax", "wind_run_km", "srad_mj", "epan", "rh", "par", "rain_mm"]
    assert met.index.name == "date"
    assert met.index[0] == pd.Timestamp("2001-01-01") and met.index[-1] == pd.Timestamp("2023-12-31")
    assert len(met) == len(pd.date_range("2001-01-01", "2023-12-31"))
    assert (met.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert met.attrs["begin"] == pd.Timestamp("2001-01-01") and met.attrs["met_flag"] == 0
    first = met.iloc[0]
    assert (first.tmin, first.tmax, first.wind_run_km, first.srad_mj, first.rh) == (
        -10.0,
        -3.49,
        294.6,
        6.83,
        91.39,
    )
    assert met.loc["2001-01-03", "rain_mm"] == 3.17
    assert (met.tmax >= met.tmin).all()


def test_read_brk_consistent_with_met(scenario: Path) -> None:
    brk = read_brk(scenario / "CA-TPA.BRK")
    assert brk.calendar_code == 1
    ev = brk.events
    assert ev.iloc[0][["year", "doy", "n_breakpoints", "midnight"]].tolist() == [2001, 3, 2, 0]
    assert ev.iloc[0].depth_in == 0.125
    assert len(brk.breakpoints) == int(ev.n_breakpoints.sum())
    last = brk.breakpoints.groupby("event").cum_depth_in.max()
    np.testing.assert_allclose(last.to_numpy(), ev.depth_in.to_numpy())
    met = read_met(scenario / "CA-TPA.MET")
    daily = brk.daily_rain_mm()
    assert set(daily.index) <= set(met.index[met.rain_mm > 0])
    # BRK depths are inches with 3 decimals -> at most 0.0005 in = 0.0127 mm rounding
    np.testing.assert_allclose(daily.to_numpy(), met.rain_mm.reindex(daily.index).to_numpy(), atol=0.0128)


# ----------------------------------------------------------------------------- .ana


# ----------------------------------------------------------------------------- OVERVIEW.OUT
