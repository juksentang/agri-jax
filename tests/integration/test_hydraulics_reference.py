"""Brooks-Corey hydraulics against the reference data: every horizon of the 15 RZWQM scenarios and the
RZWQM2 binary itself.

1. ``SOILPR`` semantics on all 105 horizons: ``theta_of_h`` at -333, -100 and -15000 cm reproduces the
   rec2 ``FC1/3``, ``FC1/10`` and ``WP`` written in each ``rzwqm.dat`` (written with 6 decimals; the
   largest difference over 315 values is 6.95e-7). A 1 % change of the suction head moves theta by more
   than 3.7e-5 on every horizon, so the check resolves *which* heads RZWQM uses.
2. The parameter-inertness experiment, automated: on CA-TPA 2015, FC1/3, FC1/10 and WP of horizon 1 at
   -30 % leave all 138 ``.ana`` columns and the yield bit-identical to the base run, while Ksat at -50 %
   (and theta_r at -30 %, a curve parameter) change them: RZWQM re-derives rec2 from the curve.
3. K(h) is continuous and monotone across the ``hb_k`` junction on every horizon.
4. ``h_of_theta`` inverts ``theta_of_h`` to 1e-10 on every horizon over
   ``[theta_r + 1e-4, theta_s - 1e-4]``, except on 6 small-lambda horizons within 6.2e-4 of theta_r, where
   the exact inverse is below the ``H_MIN = -1e30`` cm guard and ``H_MIN`` is returned (checked).
5. The parameter ranges of the unit-tier property tests cover ``all_parameters.csv`` and every horizon.

RZWQM runs are staged under ``<data-dir>/run/`` (the Fortran reads 80-character paths); outputs are
copied to ``tmp_path``. Each run of one year takes about 2.5 s.
"""

from __future__ import annotations

import csv
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.io.rzwqm.ana import read_ana
from agrijax.io.rzwqm.dat import RzwqmDat, read_rzwqm_dat, set_value, write_rzwqm_dat
from agrijax.io.rzwqm.params import canonical_name
from agrijax.port.run_fortran import RZWQM_BINARY, parse_overview_yields, run_rzwqm
from agrijax.processes.soil_water.hydraulics import (
    H_CLAMP_RZWQM,
    H_FC13,
    H_FC110,
    H_MIN,
    H_WP,
    SoilHydraulicParams,
    h_of_theta,
    k_of_h,
    theta_of_h,
)

BATCH = Path("narval_mirror/RZWQM_sw_batch")
SCENARIOS = (
    "CA-ER1", "CA-MA1", "CA-TPA", "US-LYS_NW", "US-LYS_SE", "US-LYS_SW", "US-Mj1", "US-S2", "US-TW3",
    "US-Tw2", "US-UA1_HartFarm", "US-manilacotton", "US_OPE", "US_Rockfish", "US_Rockford_Alfalfa",
)  # fmt: skip
X64 = bool(jax.config.jax_enable_x64)
#: the rec2 values are written with 6 decimals: agreement to within one unit of the 6th decimal
REC2_ATOL = 1.0e-6


def _dat(data_dir: Path, name: str) -> RzwqmDat:
    f = data_dir / BATCH / name / "Scenario" / "rzwqm.dat"
    if not f.is_file():
        pytest.skip(f"{f} not found")
    return read_rzwqm_dat(f)


@pytest.fixture(scope="module")
def all_hydraulics(data_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    root = data_dir / BATCH
    if not root.is_dir():
        pytest.skip(f"{root} not found")
    return {s: _dat(data_dir, s).hydraulics for s in SCENARIOS}


def _p(hyd: dict[str, np.ndarray]) -> SoilHydraulicParams:
    return SoilHydraulicParams.from_rzwqm_dat(hyd, derive=False)


# ---------------------------------------------------------------------------
# 1. SOILPR: rec2 is the curve at -333 / -100 / -15000 cm, on every horizon
# ---------------------------------------------------------------------------


def test_scenario_set_is_complete(all_hydraulics: dict[str, dict[str, np.ndarray]]) -> None:
    n = sum(len(h["hb"]) for h in all_hydraulics.values())
    assert len(all_hydraulics) == 15 and n == 105


def test_file_c2_is_stale_on_26_horizons(all_hydraulics: dict[str, dict[str, np.ndarray]]) -> None:
    """The rec2 c2 is not ksat * hb_k**(eps - n1) (what SOILPR writes over it) on 26 of 105 horizons."""
    stale = 0
    for hyd in all_hydraulics.values():
        p = SoilHydraulicParams.from_rzwqm_dat(hyd)  # derive=True: c2 recomputed as SOILPR does
        stale += int(np.sum(~np.isclose(hyd["c2"], np.asarray(p.c2), rtol=1e-4)))
    assert stale == 26
    catpa = all_hydraulics["CA-TPA"]
    np.testing.assert_allclose(catpa["c2"], 2.59 * catpa["hb_k"] ** catpa["eps"], rtol=1e-5)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_rec2_is_the_curve_on_every_horizon(data_dir: Path, scenario: str) -> None:
    """ITYPE = 0 and hb < 333 on every horizon (the SOILPR branch used), and rec2 = theta(-333, -100, -15000)."""
    dat = _dat(data_dir, scenario)
    assert np.all(dat.soil_physical["texture_code"] == 0), "SOILPR ITYPE > 0 (texture scaling) is not ported"
    hyd = dat.hydraulics
    p = _p(hyd)
    n = p.n_horizon
    assert np.all(hyd["hb"] < -H_FC13)
    tol = REC2_ATOL if X64 else 3e-6
    for head, key in ((H_FC13, "theta_fc33"), (H_FC110, "theta_fc10"), (H_WP, "theta_wp")):
        got = np.asarray(theta_of_h(jnp.full(n, head), p))
        np.testing.assert_allclose(got, hyd[key], rtol=0, atol=tol, err_msg=f"{scenario} {key}")
        # resolution: a 1 % error in the head would be visible on every horizon
        off = np.asarray(theta_of_h(jnp.full(n, head * 1.01), p))
        assert np.all(np.abs(off - hyd[key]) > 10 * REC2_ATOL), f"{scenario} {key}: check cannot resolve 1 %"


# ---------------------------------------------------------------------------
# 3. K(h) across the hb_k junction; 4. inverse round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_k_continuous_and_monotone_at_hb_k(data_dir: Path, scenario: str) -> None:
    hyd = _dat(data_dir, scenario).hydraulics
    p = _p(hyd)
    hbk, ks = hyd["hb_k"], hyd["ksat"]
    assert np.all(hyd["n1"] == 0.0)  # flat wet segment: K = ksat on [-hb_k, 0]
    d = 1e-9 if X64 else 1e-5
    for i in range(p.n_horizon):
        pi = jax.tree_util.tree_map(lambda a, i=i: a[i], p)
        lo = float(k_of_h(-hbk[i] * (1 + d), pi))
        at = float(k_of_h(-hbk[i], pi))
        hi = float(k_of_h(-hbk[i] * (1 - d), pi))
        assert at == pytest.approx(ks[i], rel=1e-12 if X64 else 1e-6)
        assert lo == pytest.approx(at, rel=1e-7 if X64 else 1e-3) and hi == pytest.approx(
            at, rel=1e-12 if X64 else 1e-6
        )
        assert lo <= at <= hi
        # monotone on a fine grid straddling the junction and over the whole range
        h = -np.concatenate([hbk[i] * np.linspace(0.5, 2.0, 301), np.logspace(-3, 6, 400)])
        h = np.sort(h)
        k = np.asarray(k_of_h(jnp.asarray(h), pi))
        assert np.all(np.isfinite(k)) and np.all(k > 0.0) and np.all(k <= ks[i] * (1 + 1e-12))
        assert np.all(np.diff(k) >= -1e-12 * ks[i]), f"{scenario} horizon {i + 1}: K not monotone"
        # dry side is the power law through (hb_k, ksat) with exponent eps: K(-10 hb_k) = ksat 10**-eps
        assert float(k_of_h(-10 * hbk[i], pi)) == pytest.approx(
            ks[i] * 10 ** -hyd["eps"][i], rel=1e-10 if X64 else 1e-5
        )


#: horizons whose lambda is so small that theta_r + 1e-4 maps below the H_MIN = -1e30 cm overflow guard of
#: h_of_theta (|h| up to 6.6e37 cm for US-Tw2 horizon 1, lambda = 0.101); 6 of 105, see the test below.
FLOORED = {("CA-MA1", 1), ("CA-MA1", 3), ("US-LYS_SW", 3), ("US-LYS_SW", 4), ("US-Tw2", 1), ("US-Tw2", 3)}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_h_of_theta_round_trip(data_dir: Path, scenario: str) -> None:
    """theta -> h -> theta to 1e-10 on [theta_r + 1e-4, theta_s - 1e-4], 2001 contents per horizon.

    Where the exact inverse is below ``H_MIN = -1e30`` cm (small lambda, theta within 6.2e-4 of theta_r)
    h_of_theta returns ``H_MIN`` (to rounding); that band is far outside the model's range (RZWQM clamps h at
    ``H_CLAMP_RZWQM = -15000`` cm, and theta(-15000) is more than 0.01 above it on every horizon).
    """
    hyd = _dat(data_dir, scenario).hydraulics
    p = _p(hyd)
    tol = 1e-10 if X64 else 2e-6
    margins: list[float] = []
    for i in range(p.n_horizon):
        pi = jax.tree_util.tree_map(lambda a, i=i: a[i], p)
        theta = np.linspace(hyd["theta_r"][i] + 1e-4, hyd["theta_s"][i] - 1e-4, 2001)
        h = np.asarray(h_of_theta(jnp.asarray(theta), pi))
        theta_floor = float(theta_of_h(H_MIN, pi))
        ok = theta > theta_floor
        assert (not ok.all()) == ((scenario, i + 1) in FLOORED), (scenario, i + 1, theta_floor)
        np.testing.assert_allclose(h[~ok], H_MIN, rtol=1e-12)
        assert np.all(np.isfinite(h)) and np.all(h < 0.0)
        assert np.all(np.diff(h[ok]) > 0.0)
        np.testing.assert_allclose(
            np.asarray(theta_of_h(jnp.asarray(h[ok]), pi)),
            theta[ok],
            rtol=0,
            atol=tol,
            err_msg=f"{scenario} {i + 1}",
        )
        margins.append(float(theta_of_h(H_CLAMP_RZWQM, pi)) - theta_floor)
    assert min(margins) > 0.01, margins


# ---------------------------------------------------------------------------
# 5. the unit-tier property ranges cover the calibration ranges and every horizon
# ---------------------------------------------------------------------------


def _unit_props() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "unit" / "test_hydraulics_props.py"
    spec = importlib.util.spec_from_file_location("_hydraulics_props", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_property_ranges_cover_the_data(
    data_dir: Path, all_hydraulics: dict[str, dict[str, np.ndarray]]
) -> None:
    ranges: dict[str, tuple[float, float]] = _unit_props().RANGES
    field = {"lam": "lambda_", "ksat": "ksat", "theta_r": "theta_r"}
    n_rows = 0
    with open(data_dir / BATCH / "all_parameters.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                _, fld, _ = canonical_name(row["parameter"])
            except KeyError:
                continue
            if fld in field:
                lo, hi = ranges[field[fld]]
                assert lo <= float(row["min"]) and float(row["max"]) <= hi, row
                n_rows += 1
    assert n_rows == 15 * 4 * 3
    dat_key = {"hb": "hb", "lambda_": "lam", "eps": "eps", "ksat": "ksat", "theta_r": "theta_r",
               "theta_s": "theta_s", "hb_k": "hb_k"}  # fmt: skip
    for hyd in all_hydraulics.values():
        for name, key in dat_key.items():
            lo, hi = ranges[name]
            assert np.all((hyd[key] >= lo) & (hyd[key] <= hi)), (name, hyd[key])
        assert np.all(hyd["a1"] == 0.0) and np.all(hyd["n1"] == 0.0)


# ---------------------------------------------------------------------------
# 2. parameter inertness in the RZWQM2 binary (CA-TPA, 2015)
# ---------------------------------------------------------------------------

#: name -> (fields of horizon 1, factor)
PERTURBATIONS: dict[str, tuple[tuple[str, ...], float]] = {
    "base": ((), 1.0),
    "rec2_m30": (("theta_fc33", "theta_fc10", "theta_wp"), 0.7),
    "ksat_m50": (("ksat",), 0.5),
    "theta_r_m30": (("theta_r",), 0.7),
}


def _perturb(dat: RzwqmDat, fields: tuple[str, ...], factor: float) -> RzwqmDat:
    addr = dat.hydraulic_addresses()
    out = dat
    for fld in fields:
        line, tok = addr[fld][0]
        out = set_value(out, line, tok, round(dat.get_float(line, tok) * factor, 6), decimals=6)
    return out


@pytest.fixture(scope="module")
def inertness_runs(
    data_dir: Path, catpa_scenario: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[Path, list[float], RzwqmDat]]:
    if not RZWQM_BINARY.is_file():
        pytest.skip(f"RZWQM binary not found at {RZWQM_BINARY}")
    run_root = data_dir / "run"
    run_root.mkdir(exist_ok=True)
    assert len(str(run_root.resolve())) < 60  # + "/rz_xxxxxxxx/CA-TPA.MET" must stay < 80
    tmp = tmp_path_factory.mktemp("hyd_inert")
    base = read_rzwqm_dat(catpa_scenario / "rzwqm.dat")
    dats = {name: _perturb(base, f, x) for name, (f, x) in PERTURBATIONS.items()}

    def run(name: str) -> tuple[str, tuple[Path, list[float], RzwqmDat]]:
        dat_file = write_rzwqm_dat(dats[name], tmp / f"{name}.dat")
        r = run_rzwqm(
            catpa_scenario, tmp / name, dat_override=dat_file, start="2015-01-01", end="2015-12-31",
            timeout=120, run_root=run_root,
        )  # fmt: skip
        assert r.overview_path is not None
        return name, (r.ana_path, parse_overview_yields(r.overview_path), dats[name])

    with ThreadPoolExecutor(max_workers=len(dats)) as ex:
        return dict(ex.map(run, dats))


def _changed(a: Path, b: Path) -> list[str]:
    da, db = read_ana(a), read_ana(b)
    assert list(da.data_vars) == list(db.data_vars)
    return [str(v) for v in da.data_vars if not np.array_equal(da[v].values, db[v].values)]


def test_perturbations_reach_the_file(inertness_runs: dict[str, tuple[Path, list[float], RzwqmDat]]) -> None:
    """Non-vacuous: the perturbed dat files carry the new values, and nothing else changed."""
    base = inertness_runs["base"][2].hydraulics
    for name, (fields, factor) in PERTURBATIONS.items():
        hyd = inertness_runs[name][2].hydraulics
        for key, v in hyd.items():
            want = base[key].copy()
            if key in fields:
                want[0] = round(want[0] * factor, 6)
            np.testing.assert_allclose(v, want, rtol=0, atol=5e-7, err_msg=f"{name} {key}")


def test_base_run_matches_reference(inertness_runs: dict[str, tuple[Path, list[float], RzwqmDat]]) -> None:
    ana, yields, _ = inertness_runs["base"]
    assert len(read_ana(ana).data_vars) == 138
    assert yields == [9916.0]  # CA-TPA 2015 maize, also the first season of the 2015-2023 run


def test_rec2_retention_values_are_inert(
    inertness_runs: dict[str, tuple[Path, list[float], RzwqmDat]],
) -> None:
    """FC1/3, FC1/10, WP of horizon 1 at -30 %: all 138 columns and the yield are bit-identical."""
    base_ana, base_y, _ = inertness_runs["base"]
    ana, y, _ = inertness_runs["rec2_m30"]
    assert _changed(base_ana, ana) == []
    assert ana.read_bytes() == base_ana.read_bytes()
    assert y == base_y


@pytest.mark.parametrize(("name", "min_cols"), [("ksat_m50", 30), ("theta_r_m30", 30)])
def test_curve_parameters_are_live(
    inertness_runs: dict[str, tuple[Path, list[float], RzwqmDat]], name: str, min_cols: int
) -> None:
    """Positive controls: Ksat -50 % (and theta_r -30 %) of horizon 1 change >= 30 columns and the yield."""
    base_ana, base_y, _ = inertness_runs["base"]
    ana, y, _ = inertness_runs[name]
    changed = _changed(base_ana, ana)
    print(f"{name}: {len(changed)} of 138 columns changed, yield {base_y} -> {y}")
    assert len(changed) >= min_cols, (name, len(changed))
    assert y != base_y, (name, y, base_y)
