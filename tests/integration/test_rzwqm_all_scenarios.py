"""All 15 RZWQM2 scenarios of ``RZWQM_sw_batch``: one-year Fortran runs checked against independent references.

Each scenario ``<data-dir>/narval_mirror/RZWQM_sw_batch/<site>/Scenario`` is run for the first
calendar year of its ``IPNAMES.DAT`` period with :func:`agrijax.port.run_fortran.run_rzwqm`
(all 15 in a thread pool; each run is a subprocess, ~15 s wall on 4 cores). The checks, per scenario:

========================  ======================================================================
check                     independent reference
========================  ======================================================================
run completes             exit status 0 **and** no ``Program will have to stop`` in ``run.log``
                          (the binary exits 0 after a Fortran ``STOP ' '``)
``.ana`` layout           138 variables; ``ndays + 1`` rows (``YYYY.000`` initial state)
``OVERVIEW.OUT``          every season matches a planting of ``rzwqm.dat``; a season harvested
                          inside the year has a finite yield equal to ``.ana`` column 44
                          (biomass of grain) to print precision; no OVERVIEW only when
                          ``rzwqm.dat`` has no harvest inside the year
``LAYER.PLT`` grid        node count and node depths equal the ``rzwqm.dat`` numerical grid
``LAYER.PLT`` storage     ``sum(theta * dz)`` equals ``.ana`` STORED SOIL WATER (<= 1e-3 cm, daily)
soil water balance        ``dS = infil - E - T - seepage - tile - lateral + added`` from ``.ana``
                          fluxes closes daily (<= 3e-4 cm; conservation law)
annual ET                 ``.ana`` column 84 (ACTUAL ET) equals columns 6 + 7 (E + T) daily
``rzwqm.dat`` round trip  read -> write is byte-identical
parameter map             every ``all_parameters.csv`` entry of the scenario resolves through
                          :func:`params_from_dat`; values equal the raw token at the CSV
                          (line, token) address and the ``cur`` columns of
                          ``parameter_ranges_review.csv`` (a separately produced table)
========================  ======================================================================

The run-dir length and the exit-0 STOP are handled by :func:`run_rzwqm` itself: the Cropsim-CERES
module (wheat, canola: ``DSSAT40/CSCER/CSCER040.FOR``) builds the ecotype path in
``CHARACTER*64 ECDIRFLE`` = ``<rundir>/DSSAT/`` + ``WHCER040.ECO``, so the run dir must be <= 45
characters (a longer one truncates the name, the model prints ``Could not find input file!`` and
STOPs with exit status 0). ``run_rzwqm`` creates ``<run>/rXXXXXXXX`` and refuses a longer path,
and raises :class:`FortranRunError` on a STOP marker in ``run.log`` or a short ``.ana``.
:func:`test_default_run_dir_fits_cscer_ecotype_buffer` checks the buffer size against the Fortran
declaration and :func:`test_truncated_run_is_detected` forces the old 46-character dir on CA-MA1.

One workaround is applied in the *staging* only, never to the scenario on disk:

* **US_Rockford_Alfalfa** ships Windows-cased names (``ipnames.dat``, ``aldssat.rzx``, ...) and
  43-character weather file names that overflow the 80-character IPNAMES records: it is staged
  with the renames of its own ``scenario_linuxized.py`` and the MET/BRK files shortened.

A per-scenario summary (runtime, nodes, yield, annual ET, check residuals) is written to
``<data-dir>/validation/rzwqm_scenarios/rzwqm_scenarios_1yr.csv`` plus ``<site>_daily.csv``.
Failed runs are recorded there with their ``run.log`` tail and fail their tests (never skipped).
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import agrijax.port.run_fortran as rf
from agrijax.io.rzwqm import (
    param_map_from_csv,
    params_from_dat,
    prepare_rzwqm_forcing,
    read_ana,
    read_met,
    read_overview_yields,
    read_rzwqm_dat,
    write_rzwqm_dat,
)
from agrijax.io.rzwqm.layers import profile_storage_cm, read_layer_output, simulation_start

BATCH = Path("narval_mirror/RZWQM_sw_batch")
TOOL = Path("narval_mirror/RZWQM_Tool")
CSCER_SRC = Path("narval_mirror/RZWQM_Linux_Ver45/src/DSSAT40/CSCER/CSCER040.FOR")
OUT_SUBDIR = Path("validation/rzwqm_scenarios")

SITES = (
    "CA-ER1",
    "CA-MA1",
    "CA-TPA",
    "US-LYS_NW",
    "US-LYS_SE",
    "US-LYS_SW",
    "US-Mj1",
    "US-S2",
    "US-TW3",
    "US-Tw2",
    "US-UA1_HartFarm",
    "US-manilacotton",
    "US_OPE",
    "US_Rockfish",
    "US_Rockford_Alfalfa",
)
N_ANA_VARIABLES = 138
#: ``CHARACTER*64 ECDIRFLE`` (CSCER040.FOR) holds ``<rundir>/DSSAT/`` + a 12-character file name.
ECO_BUFFER = 64
ECO_NAME_LEN = len("WHCER040.ECO")
MAX_RUN_DIR_LEN = ECO_BUFFER - len("/DSSAT/") - ECO_NAME_LEN
STORAGE_TOL_CM = 1e-3
BALANCE_TOL_CM = 3e-4  # stored soil water is printed with 6 significant digits (~1e-4 cm at 10-100 cm)
ET_TOL_CM = 1e-5  # G15.6 print precision of daily fluxes (< 1 cm/day)
KEEP = ("*.ana", "OVERVIEW.OUT", "LAYER.PLT", "WORK.OUT")

# --------------------------------------------------------------------------- pool worker


def _find_ci(d: Path, name: str) -> Path | None:
    return next((p for p in d.iterdir() if p.name.upper() == name.upper()), None)


def _stage_source(site: str, scenario: Path, stage_root: Path, run_root: Path) -> Path:
    """Scenario as the Linux binary expects it; a staged copy only when renames are needed."""
    if (scenario / "IPNAMES.DAT").is_file():
        ip_lines = (scenario / "IPNAMES.DAT").read_bytes().decode("latin-1").splitlines()
        long_names = any(
            len(f"{run_root.resolve()}/r12345678/" + re.split(r"[\\/]", ln.strip())[-1]) > rf.MAX_PATH_LEN
            for ln in ip_lines[:8]
        )
        if not long_names:
            return scenario
    src = stage_root / site / "Scenario"
    shutil.copytree(scenario, src)
    # scenario_linuxized.py renames: ipnames.dat -> IPNAMES.DAT, *.rzx / *.cul -> upper case
    ip = _find_ci(src, "IPNAMES.DAT")
    assert ip is not None, f"{site}: no IPNAMES.DAT"
    if ip.name != "IPNAMES.DAT":
        ip = ip.rename(src / "IPNAMES.DAT")
    for p in list(src.iterdir()):
        if p.is_file() and p.suffix.lower() in (".rzx", ".cul") and p.name != p.name.upper():
            p.rename(src / p.name.upper())
    raw = ip.read_bytes().decode("latin-1")
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)
    for i, ext in ((2, "MET"), (3, "BRK")):  # shorten weather names that overflow 80-char records
        base = re.split(r"[\\/]", lines[i].strip())[-1]
        f = _find_ci(src, base)
        if f is not None and len(base) > 16:
            f.rename(src / f"wx.{ext}")
            lines[i] = f"C:\\x\\wx.{ext}"
    ip.write_bytes(nl.join(lines).encode("latin-1"))
    return src


def _first_year(ipnames: Path) -> tuple[_dt.date, _dt.date]:
    start = simulation_start(ipnames).astype(object)
    assert isinstance(start, _dt.date)
    return start, _dt.date(start.year, 12, 31)


def _short_run_root(data_dir: str | Path) -> Path:
    """Run root for the reference binary (<= 45 characters): AGRI_JAX_RUN_ROOT if set, like run_fortran."""
    return Path(os.environ.get("AGRI_JAX_RUN_ROOT", str(Path(data_dir) / "run")))


def _run_one(site: str, data_dir: str, out_root: str, stage_root: str) -> dict[str, Any]:
    """Pool worker: stage + run one scenario; never raises (failures are returned with the log)."""
    rec: dict[str, Any] = {"site": site, "ok": False, "error": "", "log_tail": ""}
    out = Path(out_root) / site
    try:
        scenario = Path(data_dir) / BATCH / site / "Scenario"
        run_root = _short_run_root(data_dir)
        src = _stage_source(site, scenario, Path(stage_root), run_root)
        start, end = _first_year(src / "IPNAMES.DAT")
        rec.update(start=start.isoformat(), end=end.isoformat(), source=str(src))
        t0 = time.perf_counter()
        r = rf.run_rzwqm(src, out, start=start, end=end, keep_files=KEEP, timeout=300, run_root=run_root)
        rec.update(
            elapsed_s=r.elapsed_s,
            wall_s=time.perf_counter() - t0,
            ana=str(r.ana_path),
            overview=str(r.overview_path) if r.overview_path else "",
            layer=str(out / "LAYER.PLT") if (out / "LAYER.PLT").is_file() else "",
            log=str(r.run_log),
        )
        # run_rzwqm raises on a STOP marker or a short .ana; the log is re-checked here anyway
        log = r.run_log.read_text(errors="replace").lower()
        assert not any(m in log for m in rf.RZWQM_STOP_MARKERS), "STOP marker in run.log"
        rec["ok"] = True
    except Exception as e:  # every failure is reported per scenario, never raised
        rec["error"] = f"{type(e).__name__}: {e}"
        log = out / "run.log"
        if log.is_file():
            rec["log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-30:])
    return rec


# --------------------------------------------------------------------------- analysis


def _ana_checks(ds: Any) -> dict[str, float]:
    f = {k: ds[k].values[1:] for k in ds.data_vars}
    s = ds["stored_soil_water"].values
    soil_in = f["infiltration"] + f["water_added_due_to_using_measured_swc"]
    soil_out = (
        f["actual_evaporation"]
        + f["actual_transpiration"]
        + f["deep_seepage"]
        + f["tile_drainage"]
        + f["lateral_water_flow"]
    )
    bal = np.diff(s) - (soil_in - soil_out)
    et = f["actual_et"]
    et2 = f["actual_evaporation"] + f["actual_transpiration"]
    surf = f["precipitation"] + f["irrigation"] - f["infiltration"] - f["runoff"]
    return {
        "annual_et_cm": float(et.sum()),
        "annual_precip_cm": float(f["precipitation"].sum()),
        "annual_irrig_cm": float(f["irrigation"].sum()),
        "soil_balance_max_abs_cm": float(np.abs(bal).max()),
        "et_col84_vs_e_plus_t_max_abs_cm": float(np.abs(et - et2).max()),
        # interception / snow / ponding storage: not closed by the .ana fluxes, reported only
        "surface_remainder_cm": float(surf.sum()),
        "grain_max_kg_ha": float(np.nanmax(ds["biomass_of_grain"].values)),
    }


def _summarise(rec: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    row: dict[str, Any] = {k: rec.get(k, "") for k in ("site", "ok", "start", "end", "elapsed_s", "error")}
    if not rec["ok"]:
        row["log_tail"] = rec.get("log_tail", "").replace("\n", " | ")[-800:]
        return row
    scen = Path(data_dir) / BATCH / rec["site"] / "Scenario"
    dat = read_rzwqm_dat(scen / "rzwqm.dat")
    ds = read_ana(rec["ana"])
    row.update(n_node=dat.n_node, n_ana_rows=ds.sizes["time"])
    row.update(_ana_checks(ds))
    if rec["layer"]:
        lay = read_layer_output(rec["layer"], start=rec["start"])
        stor = profile_storage_cm(lay).values
        ana_s = ds["stored_soil_water"].values[1:]
        if stor.shape == ana_s.shape:
            row["layer_storage_max_abs_cm"] = float(np.abs(stor - ana_s).max())
            daily = pd.DataFrame(
                {
                    "date": pd.DatetimeIndex(ds["time"].values[1:]).strftime("%Y-%m-%d"),
                    "ana_stored_soil_water_cm": ana_s,
                    "layer_theta_dz_cm": stor,
                    "actual_et_cm": ds["actual_et"].values[1:],
                    "precipitation_cm": ds["precipitation"].values[1:],
                }
            )
            daily.to_csv(data_dir / OUT_SUBDIR / f"{rec['site']}_daily.csv", index=False)
    row["crop"], row["yield_kg_ha"] = "", float("nan")
    if rec["overview"]:
        seasons = read_overview_yields(rec["overview"]).to_dict("records")
        hit = [s for s in seasons if np.isfinite(s["yield_kg_ha"])]
        if hit:
            row["crop"], row["yield_kg_ha"] = str(hit[0]["crop"]), float(hit[0]["yield_kg_ha"])
        elif seasons:
            row["crop"] = "<no yield line>"
    return row


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def batch_dir(data_dir: Path) -> Path:
    b = data_dir / BATCH
    if not b.is_dir():
        pytest.skip(f"{b} not found")
    missing = [s for s in SITES if not (b / s / "Scenario").is_dir()]
    assert not missing, f"scenario folders missing from {b}: {missing}"
    if not (data_dir / TOOL / "main_ryzen5_avx512").is_file():
        pytest.skip("RZWQM binary not found")
    return b


@pytest.fixture(scope="module")
def runs(
    batch_dir: Path, data_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, dict[str, Any]]]:
    out_root = tmp_path_factory.mktemp("rz_all_out")
    stage_root = tmp_path_factory.mktemp("rz_all_src")
    workers = max(1, min(len(SITES), os.cpu_count() or 1))
    # threads, not processes: the work is the Fortran subprocess, and fork() after JAX has
    # started its threads is unsafe
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_run_one, s, str(data_dir), str(out_root), str(stage_root)) for s in SITES]
        recs = {r["site"]: r for r in (f.result() for f in futs)}
    out_dir = data_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_summarise(recs[s], data_dir) for s in SITES]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "rzwqm_scenarios_1yr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    yield recs


def _need(runs: dict[str, dict[str, Any]], site: str) -> dict[str, Any]:
    rec = runs[site]
    if not rec["ok"]:
        pytest.fail(f"{site}: {rec['error']}\n--- run.log tail ---\n{rec.get('log_tail', '')}")
    return rec


def _day(x: Any) -> np.datetime64 | None:
    s = str(x)
    return None if s in ("NaT", "nan", "None", "") else np.datetime64(s[:10], "D")


def _days(rec: dict[str, Any]) -> int:
    return (_dt.date.fromisoformat(rec["end"]) - _dt.date.fromisoformat(rec["start"])).days + 1


# --------------------------------------------------------------------------- tests

site_param = pytest.mark.parametrize("site", SITES)


@pytest.mark.slow
@site_param
def test_run_completes(runs: dict[str, dict[str, Any]], site: str) -> None:
    rec = _need(runs, site)
    assert rec["elapsed_s"] < 60


@pytest.mark.slow
@site_param
def test_ana_layout(runs: dict[str, dict[str, Any]], site: str) -> None:
    rec = _need(runs, site)
    ds = read_ana(rec["ana"])
    assert len(ds.data_vars) == N_ANA_VARIABLES
    assert ds.sizes["time"] == _days(rec) + 1
    tok = ds["yyyyddd"].values
    year = rec["start"][:4]
    assert tok[0] == f"{year}.000" and tok[-1] == f"{year}.{_days(rec):03d}"
    assert pd.Timestamp(ds["time"].values[-1]).date() == _dt.date.fromisoformat(rec["end"])
    assert np.isfinite(ds["stored_soil_water"].values).all()


@pytest.mark.slow
@site_param
def test_overview_yields(runs: dict[str, dict[str, Any]], batch_dir: Path, site: str) -> None:
    rec = _need(runs, site)
    start, end = (np.datetime64(rec[k], "D") for k in ("start", "end"))
    dat = read_rzwqm_dat(batch_dir / site / "Scenario" / "rzwqm.dat")
    harvests_in_year = [
        p for p in dat.plantings if p.harvest_date is not None and start <= p.harvest_date <= end
    ]
    if not rec["overview"]:
        # RZWQM writes OVERVIEW.OUT at a season's end; none when nothing is harvested this year
        assert not harvests_in_year, f"{site}: no OVERVIEW.OUT but rzwqm.dat harvests {harvests_in_year}"
        return
    seasons = read_overview_yields(rec["overview"]).to_dict("records")
    assert len(seasons) >= 1
    ana = read_ana(rec["ana"])
    grain = ana["biomass_of_grain"].values
    plant_dates = {p.planting_date for p in dat.plantings}
    for s in seasons:
        pdate, hdate = _day(s["planting_date"]), _day(s["harvest_date"])
        assert pdate in plant_dates, f"{site}: OVERVIEW planting {pdate} not in rzwqm.dat"
        if hdate is not None and start <= hdate <= end:
            assert s["crop"], f"{site}: harvested season without a crop name"
            assert np.isfinite(s["yield_kg_ha"]) and s["yield_kg_ha"] > 0
            # OVERVIEW prints kg/ha as an integer; .ana column 44 carries the same state
            assert abs(s["yield_kg_ha"] - np.nanmax(grain)) <= 1.0


@pytest.mark.slow
@site_param
def test_layer_plt_grid(runs: dict[str, dict[str, Any]], batch_dir: Path, site: str) -> None:
    rec = _need(runs, site)
    assert rec["layer"], f"{site}: LAYER.PLT not written"
    lay = read_layer_output(rec["layer"], start=rec["start"])
    dat = read_rzwqm_dat(batch_dir / site / "Scenario" / "rzwqm.dat")
    assert lay.sizes["depth"] == dat.n_node
    np.testing.assert_allclose(lay["depth"].values, dat.node_depths_cm, atol=1e-4)
    assert lay.sizes["time"] == _days(rec)
    assert (lay["day"].values == np.arange(1, _days(rec) + 1)).all()


@pytest.mark.slow
@site_param
def test_layer_storage_matches_ana(runs: dict[str, dict[str, Any]], site: str) -> None:
    rec = _need(runs, site)
    lay = read_layer_output(rec["layer"], start=rec["start"])
    ana = read_ana(rec["ana"]).isel(time=slice(1, None))  # drop the YYYY.000 initial state
    np.testing.assert_array_equal(lay["time"].values, ana["time"].values)
    err = np.abs(profile_storage_cm(lay).values - ana["stored_soil_water"].values)
    assert err.max() <= STORAGE_TOL_CM, f"{site}: max |sum(theta dz) - ana| = {err.max():.2e} cm"


@pytest.mark.slow
@site_param
def test_soil_water_balance_and_et(runs: dict[str, dict[str, Any]], site: str) -> None:
    rec = _need(runs, site)
    c = _ana_checks(read_ana(rec["ana"]))
    assert c["soil_balance_max_abs_cm"] <= BALANCE_TOL_CM, c
    assert c["et_col84_vs_e_plus_t_max_abs_cm"] <= ET_TOL_CM, c
    assert 0 < c["annual_et_cm"] < c["annual_precip_cm"] + c["annual_irrig_cm"] + 100


@site_param
def test_dat_roundtrip_bytes(batch_dir: Path, site: str, tmp_path: Path) -> None:
    src = batch_dir / site / "Scenario" / "rzwqm.dat"
    out = write_rzwqm_dat(read_rzwqm_dat(src), tmp_path / "rzwqm.dat")
    assert out.read_bytes() == src.read_bytes()


def _review_rows(batch_dir: Path, site: str) -> list[dict[str, str]]:
    p = batch_dir / "parameter_ranges_review.csv"
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return [r for r in csv.DictReader(fh) if r["scenario"] == site]


_REVIEW_COLS = {
    "ksat": "cur Ksat",
    "lam": "cur lambda",
    "theta_r": "cur wr",
    "theta_fc33": "cur FC1/3",
    "theta_fc10": "cur FC1/10",
    "theta_wp": "cur WP",
}


@site_param
def test_params_from_dat_csv_map(batch_dir: Path, site: str) -> None:
    dat_path = batch_dir / site / "Scenario" / "rzwqm.dat"
    dat = read_rzwqm_dat(dat_path)
    specs = param_map_from_csv(batch_dir / "all_parameters.csv", site)
    assert len(specs) >= 30
    params = params_from_dat(dat, specs)
    raw = dat_path.read_bytes().decode("latin-1").splitlines()
    for s in specs:
        v = params[s.field][s.horizon - 1] if s.horizon is not None else params[s.name]
        tok = raw[s.line_number - 1].split()[s.location_at_line]  # plain split, no block parser
        assert float(v) == pytest.approx(float(tok.replace("D", "E")), rel=1e-12), s
    review = _review_rows(batch_dir, site)
    for r in review:  # present for the 6 scenarios whose Setting.json was generated
        h = int(r["horizon(label)"].split()[0].lstrip("H"))
        assert int(r["rec1_line"]) == dat.hydraulic_addresses()["lam"][h - 1][0]
        for field, col in _REVIEW_COLS.items():
            assert params[field][h - 1] == pytest.approx(float(r[col]), abs=1e-9), (h, field)


def test_default_run_dir_fits_cscer_ecotype_buffer(data_dir: Path, tmp_path: Path) -> None:
    """The buffer size comes from the Fortran declaration, the path from the real ``_make_run_dir``."""
    src = data_dir / CSCER_SRC
    if not src.is_file():
        pytest.skip(f"{src} not found")
    decl = re.search(r"CHARACTER\*(\d+)\s+ECDIRFLE", src.read_text(errors="replace"))
    assert decl is not None and int(decl.group(1)) == ECO_BUFFER
    assert rf.MAX_RZWQM_RUN_DIR_LEN == MAX_RUN_DIR_LEN
    d = rf._make_run_dir("r", None, max_len=rf.MAX_RZWQM_RUN_DIR_LEN)
    try:
        assert len(str(d)) + len("/DSSAT/") + ECO_NAME_LEN <= ECO_BUFFER
    finally:
        shutil.rmtree(d, ignore_errors=True)
    too_deep = tmp_path / ("x" * (MAX_RUN_DIR_LEN + 1))
    with pytest.raises(rf.FortranRunError, match="characters"):
        rf._make_run_dir("r", too_deep, max_len=rf.MAX_RZWQM_RUN_DIR_LEN)


def _long_run_dir(prefix: str, run_root: Path | None, max_len: int | None = None) -> Path:
    """The pre-fix run dir: ``<run>/rz_XXXXXXXX`` padded to 46 characters, length check bypassed."""
    root = Path(run_root) if run_root is not None else rf.RUN_ROOT
    root.mkdir(parents=True, exist_ok=True)
    pad = max(1, MAX_RUN_DIR_LEN + 1 - len(str(root.resolve())) - 1 - 8)
    return Path(tempfile.mkdtemp(prefix="z" * pad, dir=root)).resolve()


@pytest.mark.slow
def test_truncated_run_is_detected(batch_dir: Path, data_dir: Path, tmp_path: Path) -> None:
    """Real binary, real failure: CA-MA1 (canola) with a 46-character run dir STOPs with exit 0.

    ``run_rzwqm`` must raise instead of returning a truncated ``.ana``; the kept run dir shows the
    reference model's own ``Could not find input file`` / ``Program will have to stop`` text.
    """
    scen = batch_dir / "CA-MA1" / "Scenario"
    start, end = _first_year(scen / "IPNAMES.DAT")
    with mock.patch.object(rf, "_make_run_dir", _long_run_dir):
        with pytest.raises(rf.FortranRunError) as ei:
            rf.run_rzwqm(
                scen, tmp_path / "out", start=start, end=end, timeout=120, run_root=_short_run_root(data_dir)
            )
    run_dir = ei.value.run_dir
    assert run_dir is not None and len(str(run_dir)) == MAX_RUN_DIR_LEN + 1
    try:
        log = (run_dir / "run.log").read_text(errors="replace").lower()
        assert any(m in log for m in rf.RZWQM_STOP_MARKERS), log[-2000:]
        assert "run.log reports" in str(ei.value) or "truncated run" in str(ei.value)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


SRAD_RTOL = 2e-5  # .ana prints 6 significant digits; float32 hourly arithmetic of DSSAT HMET


@pytest.mark.slow
@site_param
def test_ana_srad_is_hourly_resum_of_met(runs: dict[str, dict[str, Any]], batch_dir: Path, site: str) -> None:
    """``.ana`` column 88 (the model's own solar radiation) == :func:`rzwqm_daily_srad` of the ``.MET``.

    Reference: the binary's printed radiation on 15 scenarios (one with a 2-degree slope, which
    exercises the SHAW direct/diffuse partition); the plain ``.MET`` value is 0.7 % away.
    """
    rec = _need(runs, site)
    src = Path(rec["source"])
    ip = (src / "IPNAMES.DAT").read_bytes().decode("latin-1").splitlines()
    met_name = re.split(r"[\\/]", ip[2].strip())[-1]
    met_path = _find_ci(src, met_name)
    assert met_path is not None, f"{site}: {met_name} not in {src}"
    phys = read_rzwqm_dat(batch_dir / site / "Scenario" / "rzwqm.dat").physiography
    met = read_met(met_path)
    prepared = prepare_rzwqm_forcing(
        met,
        latitude_rad=phys["latitude_rad"],
        slope_rad=phys["slope_rad"],
        aspect_rad=phys["aspect_rad"],
    ).loc[rec["start"] : rec["end"]]
    ds = read_ana(rec["ana"])
    cols = {int(k): v for k, v in ds.attrs["columns"].items()}
    ana = ds[cols[88]].values[1:]
    ours = prepared["srad_mj"].to_numpy()
    assert ours.shape == ana.shape
    np.testing.assert_allclose(ours, ana, rtol=SRAD_RTOL, atol=1e-5, err_msg=site)
    # and the .MET value alone is measurably different (the correction is not a no-op)
    raw = prepared["srad_mj_met"].to_numpy()
    assert np.abs(raw - ana).max() > 10 * np.abs(ours - ana).max()
