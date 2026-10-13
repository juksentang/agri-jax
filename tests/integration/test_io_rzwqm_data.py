"""Readers / writers of agrijax.io.rzwqm against the CA-TPA scenario files, the batch parameter
map ``all_parameters.csv`` and the original ``GenerateDat.py`` (skipped when absent).

Moved from the unit tier, which must not depend on the data tree (kept outside the repository).
The values of the local scenario a test pins are read from
``<data-dir>/catpa/expected/test_io_rzwqm_data.json`` (:mod:`_scenario_expected`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _scenario_expected import expected

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
def exp(data_dir: Path) -> dict:
    return expected(data_dir, __file__)


def _addr(a: list[int]) -> tuple[int, int]:
    return int(a[0]), int(a[1])


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
def test_dat_roundtrip_byte_identical(scenario: Path, tmp_path: Path, exp: dict) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    out = write_rzwqm_dat(dat, tmp_path / "rzwqm.dat")
    assert out.read_bytes() == (scenario / "rzwqm.dat").read_bytes()
    assert dat.newline == "\r\n"
    assert len(dat.lines) == exp["dat"]["n_lines"]


def test_dat_parsed_sections(scenario: Path, exp: dict) -> None:
    e = exp["dat"]
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    phys = dat.physiography
    assert phys["latitude_rad"] == pytest.approx(e["latitude_rad"])
    assert phys["co2_ppm"] == e["co2_ppm"]
    assert dat.n_horizon == e["n_horizon"] and dat.profile_depth_cm == e["profile_depth_cm"]
    np.testing.assert_array_equal(dat.horizon_depths_cm, e["horizon_depths_cm"])
    assert dat.n_node == e["n_node"]
    assert (
        dat.node_depths_cm[0] == e["first_node_depth_cm"] and dat.node_depths_cm[-1] == dat.profile_depth_cm
    )
    assert np.all(np.diff(dat.node_depths_cm) > 0)
    assert dat.node_spacing_cm[-1] == 0.0

    sp = dat.soil_physical
    assert sp["texture"] == e["texture"]
    np.testing.assert_allclose(sp["bulk_density"], e["bulk_density"])

    hyd = dat.hydraulics
    assert set(hyd) == set(HYDRAULIC_FIELDS)
    assert all(v.shape == (dat.n_horizon,) for v in hyd.values())
    # every value is the token at its address in the file (line number, 1-based, and token index)
    for k, addrs in dat.hydraulic_addresses().items():
        want = [dat.get_float(ln, col) for ln, col in addrs]
        np.testing.assert_allclose(hyd[k], want, err_msg=k)
    assert np.all(hyd["hb"] > 0.0) and np.all(hyd["ksat"] > 0.0) and np.all(hyd["c2"] > 0.0)
    assert np.all(hyd["theta_r"] < hyd["theta_wp"])
    assert np.all(hyd["theta_wp"] < hyd["theta_fc33"])
    assert np.all(hyd["theta_fc33"] < hyd["theta_s"])

    pet = dat.pet
    assert [pet["albedo_dry"], pet["albedo_wet"], pet["albedo_crop"], pet["albedo_residue"]] == e["albedos"]
    assert pet["soil_resistance"] == e["soil_resistance"] and pet["pet_method"] == e["pet_method"]

    assert dat.plants == e["plants"]
    assert dat.plant_site_params[0]["rs_min"] == e["rs_min"]

    pl = dat.plantings
    assert len(pl) == e["n_plantings"]
    assert pl[0].planting_date == np.datetime64(e["first_planting"])
    assert pl[0].harvest_date == np.datetime64(e["first_harvest"])
    assert pl[-1].planting_date == np.datetime64(e["last_planting"])
    assert pl[-1].harvest_date == np.datetime64(e["last_harvest"])
    assert all(
        p.density_seeds_ha == e["density_seeds_ha"] and p.row_spacing_cm == e["row_spacing_cm"] for p in pl
    )
    assert all(p.harvest_option == e["harvest_option"] and p.harvest_type == e["harvest_type"] for p in pl)
    assert dat.get(*pl[0].density_address) == e["density_token"]


def test_dat_addresses_match_all_parameters_csv(scenario: Path, exp: dict) -> None:
    e = exp["addresses"]
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    h = dat.hydraulic_addresses()
    assert h["lam"][:4] == [_addr(a) for a in e["lam_first4"]]
    assert h["ksat"][0] == _addr(e["ksat_0"]) and h["theta_r"][0] == _addr(e["theta_r_0"])
    assert h["theta_fc33"][0] == _addr(e["theta_fc33_0"]) and h["theta_fc10"][0] == _addr(e["theta_fc10_0"])
    assert h["theta_wp"][0] == _addr(e["theta_wp_0"])
    assert dat.pet_addresses()["albedo_dry"] == _addr(e["albedo_dry"])
    assert dat.pet_addresses()["soil_resistance"] == _addr(e["soil_resistance"])
    assert dat.plant_site_addresses()[0]["rs_min"] == _addr(e["rs_min_0"])


def test_set_value_semantics(scenario: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    orig = list(dat.lines)
    ln, col = dat.hydraulic_addresses()["lam"][0]
    new = set_value(dat, ln, col, 0.4)
    assert dat.lines == orig  # not in place by default
    old_tok, new_tok = orig[ln - 1].split(), new.lines[ln - 1].split()
    assert (
        new_tok[col] == "0.400" and new_tok[:col] + new_tok[col + 1 :] == old_tok[:col] + old_tok[col + 1 :]
    )
    assert new.lines[ln - 1].endswith("\r\n")
    assert new.hydraulics["lam"][0] == 0.4
    assert sum(a != b for a, b in zip(new.lines, orig, strict=True)) == 1
    same = set_value(dat, *dat.pet_addresses()["soil_resistance"], 300.0, inplace=True)
    assert same is dat and dat.pet["soil_resistance"] == 300.0
    with pytest.raises(IndexError):
        set_value(dat, ln, len(dat.tokens(ln)), 1.0)
    with pytest.raises(IndexError):
        set_value(dat, 10_000, 0, 1.0)
    with pytest.raises(ValueError, match="comment"):
        set_value(dat, 1, 0, 1.0)


def test_set_value_equals_generatedat_single_edit(scenario: Path, generate_dat, tmp_path: Path) -> None:
    src = scenario / "rzwqm.dat"
    value = float(np.round(0.3456789, 5))
    dat = read_rzwqm_dat(src)
    ln, col = dat.hydraulic_addresses()["lam"][0]
    ref = _generatedat_output(
        generate_dat, src, [{"line_number": ln, "location_at_line": col, "value": value}], tmp_path / "g.dat"
    )
    ours = write_rzwqm_dat(set_value(dat, ln, col, value), tmp_path / "o.dat", newline="\n")
    assert ours.read_bytes() == ref


def test_params_to_dat_equals_generatedat_lhs(
    scenario: Path, param_csv: Path, generate_dat, tmp_path: Path, exp: dict
) -> None:
    """Three LHS samples of all CA-TPA batch parameters: our write-back == GenerateDat's files."""
    src = scenario / "rzwqm.dat"
    specs = param_map_from_csv(param_csv, "CA-TPA")
    assert len(specs) == exp["batch"]["n_params"]
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


def test_params_from_dat_with_csv_map(scenario: Path, param_csv: Path, exp: dict) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    specs = param_map_from_csv(param_csv, "CA-TPA")
    assert sorted({s.horizon for s in specs if s.horizon}) == exp["batch"]["param_horizons"]
    p = params_from_dat(dat, specs)
    for f in HYDRAULIC_FIELDS:
        assert p[f].shape == (dat.n_horizon,)
    np.testing.assert_allclose(p["ksat"], dat.hydraulics["ksat"])
    assert float(p["albedo_dry"]) == dat.pet["albedo_dry"]
    assert float(p["soil_resistance"]) == dat.pet["soil_resistance"]
    assert float(p["rs_min_corn"]) == dat.plant_site_params[0]["rs_min"]


def test_param_map_rejects_stale_address(scenario: Path) -> None:
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    ln2, col2 = dat.hydraulic_addresses()["lam"][1]  # horizon 2's address given for horizon 1
    bad = [ParamSpec(name="lam_1", field="lam", line_number=ln2, location_at_line=col2, horizon=1)]
    with pytest.raises(ValueError, match="hydraulic block"):
        params_from_dat(dat, bad)
    ln_pet = dat.pet_addresses()["albedo_dry"][0]
    bad_pet = [ParamSpec(name="albedo_dry", field="albedo_dry", line_number=ln_pet, location_at_line=1)]
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
    p2["ksat"][-1] = 1.23456789
    new2 = params_to_dat(dat, p2)
    assert new2.hydraulics["ksat"][-1] == 1.23457
    np.testing.assert_array_equal(new2.hydraulics["ksat"][:-1], p["ksat"][:-1])


def test_param_map_yaml_roundtrip(scenario: Path, param_csv: Path, tmp_path: Path) -> None:
    specs = param_map_from_csv(param_csv, "CA-TPA")
    back = load_param_map(save_param_map(specs, tmp_path / "map.yaml"))
    assert back == specs
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    full = layout_param_map(dat)
    assert len(full) == len(HYDRAULIC_FIELDS) * dat.n_horizon
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
def test_read_met(scenario: Path, exp: dict) -> None:
    e = exp["met"]
    met = read_met(scenario / "CA-TPA.MET")
    assert list(met.columns) == ["tmin", "tmax", "wind_run_km", "srad_mj", "epan", "rh", "par", "rain_mm"]
    assert met.index.name == "date"
    assert met.index[0] == pd.Timestamp(e["first_day"]) and met.index[-1] == pd.Timestamp(e["last_day"])
    assert len(met) == len(pd.date_range(e["first_day"], e["last_day"]))
    assert (met.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert met.attrs["begin"] == pd.Timestamp(e["first_day"]) and met.attrs["met_flag"] == 0
    first = met.iloc[0]
    assert [first.tmin, first.tmax, first.wind_run_km, first.srad_mj, first.rh] == e["first_row"]
    assert met.loc[e["rain_day"], "rain_mm"] == e["rain_day_mm"]
    assert (met.tmax >= met.tmin).all()


def test_read_brk_consistent_with_met(scenario: Path, exp: dict) -> None:
    e = exp["brk"]
    brk = read_brk(scenario / "CA-TPA.BRK")
    assert brk.calendar_code == 1
    ev = brk.events
    assert ev.iloc[0][["year", "doy", "n_breakpoints", "midnight"]].tolist() == e["first_event"]
    assert ev.iloc[0].depth_in == e["first_depth_in"]
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
