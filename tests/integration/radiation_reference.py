"""The RZWQM2 4.6 daily radiation reconstruction against the reference model's own values.

Helper module of ``test_forcing_radiation_reference.py`` and a script
(``python tests/integration/radiation_reference.py [--data-dir D] [--no-runs]``) that writes the
report ``<data-dir>/validation/aj_w2/radiation_reference.json`` and the per-day CSV tables next to
it. :func:`agrijax.forcing.radiation.rzwqm_radiation` is driven with the raw ``.MET`` daily
radiation and the ``rzwqm.dat`` physiography (latitude, slope, aspect) and compared with:

=====================  ==============================  =============================================
source                 values                          reference-side precision
=====================  ==============================  =============================================
``physcl_entry`` table RTS, RTH, HRTH (24 h), HRTS     full double (instrumented binary whose
CA-TPA 2015-2023       (24 h), CLOUDS; 3287 days       outputs are identical to the plain binary)
``POTEVPHR`` dumps     the same five, 40 days of 2015  full double
CA-TPA 2015
``rzwqm46_pet_scen``   RTS, RTH at the POTEVPHR entry  full double
9 scenarios x 3 years  (1095-1096 days each)
``.ana`` column 88     RTS                             G15.6: 6 significant digits
CA-TPA 2015-2023
``.ana`` column 88     RTS                             G15.6
15 scenarios, 1st year (fresh runs of the binary; one scenario, US_Rockford_Alfalfa, has a
                       2-degree slope: the SHAW partition is not the identity there)
=====================  ==============================  =============================================

Errors are reported per quantity as the largest absolute and relative difference, the number of
days that are bit-identical, and for the printed column the largest ratio of the error to half a
unit of the printed last digit (``<= 1`` means equal to print precision).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agrijax.forcing.radiation import DailyRadiation, rzwqm_radiation
from agrijax.io.rzwqm import read_ana, read_met, read_rzwqm_dat

BATCH = Path("narval_mirror/RZWQM_sw_batch")
TABLES = Path("dumps/tables")
PHYSCL_2015_2023 = TABLES / "rzwqm46_catpa2015_2023" / "physcl_entry.npz"
PET_SCEN = TABLES / "rzwqm46_pet_scen"
POTEVPHR_DUMPS = Path("dumps/POTEVPHR")
CATPA_ANA_2015_2023 = Path("catpa/base_2015_2023/CA-TPA.ana")
OUT_SUBDIR = Path("validation/aj_w2")
#: the scenarios whose first year is run fresh for the ``.ana`` comparison (all 15)
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
#: G15.6 of the ``.ana`` file: significant digits printed
ANA_DIGITS = 6


# --------------------------------------------------------------------------- inputs
def _find_ci(d: Path, name: str) -> Path | None:
    return next((p for p in d.iterdir() if p.name.upper() == name.upper()), None)


def scenario_met_path(scenario: Path) -> Path:
    """The ``.MET`` file ``IPNAMES.DAT`` line 3 names, found case-insensitively in the scenario folder."""
    ip = _find_ci(scenario, "IPNAMES.DAT")
    assert ip is not None, f"no IPNAMES.DAT in {scenario}"
    line = ip.read_bytes().decode("latin-1").splitlines()[2]
    name = re.split(r"[\\/]", line.strip())[-1]
    p = _find_ci(scenario, name)
    assert p is not None, f"{name} not in {scenario}"
    return p


def scenario_inputs(data_dir: Path, site: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """``(read_met frame, physiography)`` of a batch scenario."""
    scen = data_dir / BATCH / site / "Scenario"
    phys = read_rzwqm_dat(scen / "rzwqm.dat").physiography
    geo = {k: float(phys[k]) for k in ("latitude_rad", "slope_rad", "aspect_rad")}
    return read_met(scenario_met_path(scen)), geo


def reconstruct(met: pd.DataFrame, geo: dict[str, float], dates: Iterable[Any]) -> DailyRadiation:
    """:func:`rzwqm_radiation` of the ``.MET`` frame on ``dates`` (a date-like sequence)."""
    idx = pd.DatetimeIndex(list(dates))
    m = met.loc[idx]
    d = pd.Series(idx).dt
    return rzwqm_radiation(
        m["srad_mj"].to_numpy(),
        d.dayofyear.to_numpy(),
        geo["latitude_rad"],
        geo["slope_rad"],
        geo["aspect_rad"],
        month=d.month.to_numpy(),
    )


def yyyyddd_dates(code: np.ndarray) -> pd.DatetimeIndex:
    code = np.asarray(code).astype(int)
    years, days = code // 1000, code % 1000
    return pd.DatetimeIndex(
        [
            pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=int(dd) - 1)
            for y, dd in zip(years, days, strict=True)
        ]
    )


# --------------------------------------------------------------------------- statistics
def f32_ulps(ours: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance in float32 units in the last place (both are float32 values stored as double)."""
    a = np.asarray(ours, dtype=np.float32).view(np.int32).astype(np.int64)
    b = np.asarray(ref, dtype=np.float32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, np.int64(-(2**31)) - a, a)
    b = np.where(b < 0, np.int64(-(2**31)) - b, b)
    return np.abs(a - b)


def stats(ours: np.ndarray, ref: np.ndarray, *, per_day_axis: int | None = None) -> dict[str, Any]:
    """Largest absolute and relative difference; days (rows) that are bit-identical."""
    ours = np.asarray(ours, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    diff = np.abs(ours - ref)
    rel = np.where(ref != 0.0, diff / np.where(ref != 0.0, np.abs(ref), 1.0), np.where(diff > 0, np.inf, 0.0))
    same = ours == ref
    if ours.ndim > 1:
        same = same.reshape(same.shape[0], -1).all(axis=1)
    worst = int(np.unravel_index(np.argmax(diff), diff.shape)[0]) if diff.size else -1
    return {
        "n": int(same.shape[0]),
        "n_bit_identical": int(same.sum()),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "max_rel": float(rel.max()) if rel.size else 0.0,
        "worst_row": worst,
    }


def print_half_unit(x: np.ndarray, digits: int = ANA_DIGITS) -> np.ndarray:
    """Half a unit of the last printed digit of a G-format value with ``digits`` significant digits."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    e = np.floor(np.log10(np.where(ax > 0, ax, 1.0)))
    return 0.5 * 10.0 ** (e - (digits - 1))


def excess_over_print(ours: np.ndarray, printed: np.ndarray) -> np.ndarray:
    """Per day: the part of ``|ours - printed|`` beyond half a printed unit, relative to the value."""
    ours = np.asarray(ours, dtype=np.float64)
    printed = np.asarray(printed, dtype=np.float64)
    return np.maximum(np.abs(ours - printed) - print_half_unit(printed), 0.0) / np.abs(printed)


def printed_stats(ours: np.ndarray, printed: np.ndarray) -> dict[str, Any]:
    ours = np.asarray(ours, dtype=np.float64)
    printed = np.asarray(printed, dtype=np.float64)
    err = np.abs(ours - printed)
    ratio = err / print_half_unit(printed)
    # a value on a rounding boundary of the reference's double (printed from double) may round
    # either way by one unit: ratio <= 1 up to the rounding of the ratio itself
    return {
        "n": int(err.size),
        "max_abs": float(err.max()),
        "max_ratio_to_half_unit": float(ratio.max()),
        "n_within_print": int((ratio <= 1.0 + 1e-9).sum()),
        "n_printed_equal": int((np.abs(np.round(ours, 12) - printed) <= 0).sum()),
        "worst_row": int(np.argmax(ratio)),
        "max_rel_excess_over_print": float(excess_over_print(ours, printed).max()),
    }


# --------------------------------------------------------------------------- comparisons
def compare_physcl_catpa(data_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """CA-TPA 2015-2023: the five quantities at the PHYSCL entry, every day."""
    t = np.load(data_dir / PHYSCL_2015_2023)
    assert np.all(np.asarray(t["n_calls"]) == 1)
    dates = yyyyddd_dates(t["date"])
    met, geo = scenario_inputs(data_dir, "CA-TPA")
    assert np.all(np.asarray(t["v.XLAT"]) == geo["latitude_rad"])
    r = reconstruct(met, geo, dates)
    out: dict[str, Any] = {
        "srad (RTS)": stats(r.srad, t["v.RTS"]),
        "srad_horizontal (RTH)": stats(r.srad_horizontal, t["v.RTH"]),
        "hourly_horizontal (HRTH)": stats(r.hourly_horizontal, t["v.HRTH"]),
        "hourly_slope (HRTS)": stats(r.hourly_slope, t["v.HRTS"]),
        "clouds (CLOUDS)": stats(r.clouds, t["v.CLOUDS"]),
    }
    out["hourly_horizontal (HRTH)"]["max_f32_ulps"] = int(f32_ulps(r.hourly_horizontal, t["v.HRTH"]).max())
    daily = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "met_srad": met.loc[dates, "srad_mj"].to_numpy(),
            "rth_ref": t["v.RTH"],
            "rth_ours": r.srad_horizontal,
            "rts_ref": t["v.RTS"],
            "rts_ours": r.srad,
            "rts_abs_err": np.abs(r.srad - t["v.RTS"]),
            "clouds_ref": t["v.CLOUDS"],
            "clouds_ours": r.clouds,
            "hrth_max_f32_ulps": f32_ulps(r.hourly_horizontal, t["v.HRTH"]).max(axis=1),
            "hrts_max_abs_err": np.abs(r.hourly_slope - t["v.HRTS"]).max(axis=1),
        }
    )
    return out, daily


def compare_potevphr_dumps(data_dir: Path) -> dict[str, Any]:
    """CA-TPA 2015: the POTEVPHR entry dumps (40 days)."""
    files = sorted((data_dir / POTEVPHR_DUMPS).glob("catpa2015_d*.npz"))
    assert files, f"no POTEVPHR dumps under {data_dir / POTEVPHR_DUMPS}"
    recs = [np.load(p) for p in files]
    code = np.array([int(f["in.IYYY"]) * 1000 + int(f["in.JDAY"]) for f in recs])
    met, geo = scenario_inputs(data_dir, "CA-TPA")
    r = reconstruct(met, geo, yyyyddd_dates(code))

    def col(k: str) -> np.ndarray:
        return np.stack([np.asarray(f[k], dtype=np.float64) for f in recs])

    return {
        "n_dumps": len(recs),
        "srad (RTS)": stats(r.srad, col("in.RTS")),
        "srad_horizontal (RTH)": stats(r.srad_horizontal, col("in.RTH")),
        "hourly_horizontal (HRTH)": stats(r.hourly_horizontal, col("in.HRTH")),
        "hourly_slope (HRTS)": stats(r.hourly_slope, col("in.HRTS")),
        "clouds (CLOUDS)": stats(r.clouds, col("in.CLOUDS")),
    }


def pet_scen_sites(data_dir: Path) -> list[str]:
    return sorted(p.name.removesuffix("_entry.npz") for p in (data_dir / PET_SCEN).glob("*_entry.npz"))


def compare_pet_scen(data_dir: Path, site: str) -> dict[str, Any]:
    """RTS and RTH at the POTEVPHR entry of the 3-year PET scenario runs (first call of each day)."""
    t = np.load(data_dir / PET_SCEN / f"{site}_entry.npz")
    code = np.asarray(t["date"]).astype(int)
    assert np.unique(code).size == code.size, f"{site}: one record per day expected"
    rts, rth = np.asarray(t["v.RTS"]), np.asarray(t["v.RTH"])
    met, geo = scenario_inputs(data_dir, site)
    assert np.allclose(np.asarray(t["v.XLAT"]), geo["latitude_rad"], rtol=0, atol=0)
    r = reconstruct(met, geo, yyyyddd_dates(code))
    return {
        "days": f"{code[0]}..{code[-1]}",
        "slope_rad": geo["slope_rad"],
        "srad (RTS)": stats(r.srad, rts),
        "srad_horizontal (RTH)": stats(r.srad_horizontal, rth),
    }


def compare_ana(
    ana_path: Path, met: pd.DataFrame, geo: dict[str, float]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """``.ana`` column 88 (RTS as printed) against the reconstruction, every day of the file."""
    ds = read_ana(ana_path)
    cols = {int(k): v for k, v in ds.attrs["columns"].items()}
    printed = ds[cols[88]].values[1:]
    dates = pd.DatetimeIndex(np.asarray(ds["time"].values[1:]).astype("datetime64[D]"))
    r = reconstruct(met, geo, dates)
    daily = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "ana_col88": printed,
            "rts_ours": r.srad,
            "abs_err": np.abs(r.srad - printed),
            "half_unit": print_half_unit(printed),
        }
    )
    return printed_stats(r.srad, printed), daily


def ana_rts(ana_path: Path) -> pd.Series:
    """``.ana`` column 88 by date (``YYYY-MM-DD``)."""
    ds = read_ana(ana_path)
    cols = {int(k): v for k, v in ds.attrs["columns"].items()}
    dates = pd.DatetimeIndex(np.asarray(ds["time"].values[1:]).astype("datetime64[D]"))
    return pd.Series(ds[cols[88]].values[1:], index=dates.strftime("%Y-%m-%d"))


def table_rts(path: Path) -> pd.Series:
    """RTS of a dump table (``physcl_entry`` or a PET scenario entry) by date."""
    t = np.load(path)
    return pd.Series(np.asarray(t["v.RTS"]), index=yyyyddd_dates(t["date"]).strftime("%Y-%m-%d"))


def build_excess(table: pd.Series, printed: pd.Series) -> dict[str, Any]:
    """The reference's own build-to-build limit: dumped RTS vs the shipped binary's ``.ana``.

    The dumps come from the binary rebuilt from source (instrumented; its ``.ana`` is identical
    to the plain rebuild's); the ``.ana`` files from the shipped binary ``main_ryzen5_avx512``. Where
    the two differ by more than the print rounding, the excess is a lower bound of their
    difference (relative to the value).
    """
    common = table.index.intersection(printed.index)
    ex = excess_over_print(table.loc[common].to_numpy(), printed.loc[common].to_numpy())
    return {"n": int(ex.size), "max_rel_excess": float(ex.max()), "n_days_beyond_print": int((ex > 0).sum())}


def run_first_years(data_dir: Path, sites: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Fresh first-year runs of the batch scenarios (the staging of ``test_rzwqm_all_scenarios``)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import test_rzwqm_all_scenarios as tras

    sites = list(sites)
    out_root = Path(tempfile.mkdtemp(prefix="aj_w2_rad_out_"))
    stage_root = Path(tempfile.mkdtemp(prefix="aj_w2_rad_src_"))
    with ThreadPoolExecutor(max(1, min(len(sites), os.cpu_count() or 1))) as ex:
        futs = [ex.submit(tras._run_one, s, str(data_dir), str(out_root), str(stage_root)) for s in sites]
        return {r["site"]: r for r in (f.result() for f in futs)}


def report(data_dir: Path, *, runs: bool = True) -> dict[str, Any]:
    out_dir = data_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {}
    rep["physcl_catpa_2015_2023"], daily = compare_physcl_catpa(data_dir)
    daily.to_csv(out_dir / "radiation_physcl_catpa_2015_2023_daily.csv", index=False)
    rep["potevphr_dumps_catpa_2015"] = compare_potevphr_dumps(data_dir)
    rep["pet_scen"] = {s: compare_pet_scen(data_dir, s) for s in pet_scen_sites(data_dir)}
    met, geo = scenario_inputs(data_dir, "CA-TPA")
    rep["ana_catpa_2015_2023"], daily = compare_ana(data_dir / CATPA_ANA_2015_2023, met, geo)
    daily.to_csv(out_dir / "radiation_ana_catpa_2015_2023_daily.csv", index=False)
    rep["reference_build_excess"] = {
        "CA-TPA 2015-2023": build_excess(
            table_rts(data_dir / PHYSCL_2015_2023), ana_rts(data_dir / CATPA_ANA_2015_2023)
        )
    }
    if runs:
        recs = run_first_years(data_dir, SITES)
        for s in pet_scen_sites(data_dir):
            if recs.get(s, {}).get("ok"):
                rep["reference_build_excess"][f"{s} first year"] = build_excess(
                    table_rts(data_dir / PET_SCEN / f"{s}_entry.npz"), ana_rts(Path(recs[s]["ana"]))
                )
        rep["ana_first_year"] = {}
        for s in SITES:
            rec = recs[s]
            if not rec["ok"]:
                rep["ana_first_year"][s] = {"error": rec["error"], "log_tail": rec.get("log_tail", "")[-800:]}
                continue
            met, geo = scenario_inputs(data_dir, s)
            st, daily = compare_ana(Path(rec["ana"]), met, geo)
            st.update(slope_rad=geo["slope_rad"], start=rec["start"], end=rec["end"])
            rep["ana_first_year"][s] = st
            daily.to_csv(out_dir / f"radiation_ana_{s}_daily.csv", index=False)
    (out_dir / "radiation_reference.json").write_text(json.dumps(rep, indent=1, default=_json))
    return rep


def _json(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not math.isfinite(float(x)) else float(x)
    raise TypeError(type(x))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data"))
    ap.add_argument("--no-runs", action="store_true", help="skip the fresh first-year runs")
    a = ap.parse_args()
    print(json.dumps(report(Path(a.data_dir).expanduser(), runs=not a.no_runs), indent=1, default=_json))
