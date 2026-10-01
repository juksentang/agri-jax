"""The PRMS snowpack (``agrijax.processes.snow``) against RZWQM2 4.6 itself.

Helper module of ``test_snow_prms_reference.py`` and a script
(``python tests/integration/snow_reference.py [--data-dir D] [--no-runs]``) that writes the report
``<data-dir>/validation/aj_w2/snow_reference.json`` and per-day CSV tables next to it.

References:

=====================================  ================================  ===========================
source                                 values                            reference-side precision
=====================================  ================================  ===========================
``physcl_entry`` / ``physcl_exit``     SNOWPK (pack water equivalent)    full double (the rebuilt,
tables, CA-TPA 2015-2023 (3287 days)   at the exit, OMSEA(105) (SMELT)   instrumented binary)
``.ana`` columns 92 and 105,           SNP, SMELT                        G15.6 (6 significant
CA-TPA 2015-2023 (shipped binary)                                        digits)
``rzwqm46_pet_scen`` POTEVPHR entry    PKTEMP (the previous day's pack   full double
tables (3-year runs)                   temperature)
``.ana`` columns 3, 85, 86, 92, 105    precipitation, TMIN, TMAX, SNP,   G15.6
of fresh full-period runs of the 15    SMELT
batch scenarios
=====================================  ================================  ===========================

Two input chains drive the port. **dump**: TMIN, TMAX, RTS at the PHYSCL entry and the day's
precipitation DAYRAIN at the PHYSCL exit, i.e. the reference's own inputs, which isolates the snow
routine. **files**: the ``.MET`` temperatures with the ``IPNAMES.DAT`` modifiers and the
``INPDAY`` bounds, RTS rebuilt by :mod:`agrijax.forcing.radiation`, the day's storms from the
``.BRK`` file by :mod:`agrijax.forcing.precipitation`: the chain an Agri-JAX run uses. All sites
of a chain run in one ``vmap`` over sites (padded at the end with warm dry days).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import radiation_reference as rr

import agrijax.port.run_fortran as rf
from agrijax.forcing.precipitation import daily_storm_precipitation
from agrijax.forcing.radiation import rzwqm_radiation
from agrijax.io.rzwqm import read_ana, read_met, read_rzwqm_dat
from agrijax.io.rzwqm.met import INPDAY_BOUNDS, read_brk
from agrijax.io.rzwqm.storms import storm_depths
from agrijax.processes.snow import PrmsSnowParams, SnowForcing, SnowState, read_sno, snow_prms

BATCH = rr.BATCH
TABLES = Path("dumps/tables")
PHYSCL_ENTRY = TABLES / "rzwqm46_catpa2015_2023" / "physcl_entry.npz"
PHYSCL_EXIT = TABLES / "rzwqm46_catpa2015_2023" / "physcl_exit.npz"
PET_SCEN = TABLES / "rzwqm46_pet_scen"
CATPA_ANA = rr.CATPA_ANA_2015_2023
ERA5_DD = Path("ameriflux/fluxnet/AMF_CA-TPA_FLUXNET_ERA5_DD_1981-2025_v1.3_r1.csv")
BASE_HH = Path("ameriflux/base/AMF_CA-TPA_BASE_HH_3-5.csv")
OUT_SUBDIR = Path("validation/aj_w2")
SITES = rr.SITES
#: ``.ana`` columns used here
COL_PRECIP, COL_TMIN, COL_TMAX, COL_SWE, COL_MELT = 3, 85, 86, 92, 105
#: ``OMSEA`` index of SMELT (column 105, 0-based 104)
OMSEA_MELT = 104
#: warm, dry padding day (no pack can form: mean temperature above 0 degC, no precipitation)
PAD_T = 20.0


# --------------------------------------------------------------------------- scenario inputs
def _find_ci(d: Path, name: str) -> Path | None:
    return next((p for p in d.iterdir() if p.name.upper() == name.upper()), None)


def ipnames_lines(scenario: Path) -> list[str]:
    ip = _find_ci(scenario, "IPNAMES.DAT")
    assert ip is not None, f"no IPNAMES.DAT in {scenario}"
    return ip.read_bytes().decode("latin-1").splitlines()


def ipnames_file(scenario: Path, line: int) -> Path:
    """The file ``IPNAMES.DAT`` names on (0-based) ``line``, found case-insensitively in the scenario."""
    name = re.split(r"[\\/]", ipnames_lines(scenario)[line].strip())[-1]
    p = _find_ci(scenario, name)
    assert p is not None, f"{name} not in {scenario}"
    return p


def metmod(scenario: Path) -> np.ndarray:
    """The 8 x 12 weather modifiers of ``IPNAMES.DAT`` (lines 10-17: TMIN, TMAX offsets; wind, RTS,
    pan, RH, precipitation, CO2 in percent)."""
    rows = [ln.split() for ln in ipnames_lines(scenario)[9:17]]
    return np.array([[float(t) for t in r[:12]] for r in rows])


def scenario_ipet(scenario: Path) -> int:
    """``IPET``: item 19 of the evaporation record of ``rzwqm.dat`` (the first data line after the
    "SHAW module" label; ``Rzmain.for`` 4717-4731)."""
    lines = (scenario / "rzwqm.dat").read_bytes().decode("latin-1").splitlines()
    i = next(k for k, ln in enumerate(lines) if "SHAW" in ln.upper() and "MODULE" in ln.upper())
    rec = next(ln.split() for ln in lines[i:] if ln.strip() and not ln.strip().startswith("="))
    return int(float(rec[18]))


def site_inputs(data_dir: Path, site: str, dates: pd.DatetimeIndex) -> dict[str, Any]:
    """The files chain of ``site`` on ``dates``: forcing arrays, the ``.sno`` parameters, latitude."""
    scen = data_dir / BATCH / site / "Scenario"
    dat = read_rzwqm_dat(scen / "rzwqm.dat")
    geo = {k: float(dat.physiography[k]) for k in ("latitude_rad", "slope_rad", "aspect_rad")}
    mm = metmod(scen)
    met = read_met(ipnames_file(scen, 2)).loc[dates]
    brk = read_brk(ipnames_file(scen, 3))
    sno = read_sno(ipnames_file(scen, 6))
    mon = dates.month.to_numpy() - 1
    lo_t, hi_t = INPDAY_BOUNDS["tmin"][0], INPDAY_BOUNDS["tmax"][1]
    tmin = np.maximum(met["tmin"].to_numpy(dtype=float) + mm[0, mon], lo_t)
    tmax = np.minimum(met["tmax"].to_numpy(dtype=float) + mm[1, mon], hi_t)
    rad = rzwqm_radiation(
        met["srad_mj"].to_numpy(dtype=float),
        dates.dayofyear.to_numpy(),
        geo["latitude_rad"],
        geo["slope_rad"],
        geo["aspect_rad"],
        month=dates.month.to_numpy(),
        metmod_srad_pct=mm[3],
    )
    precip = daily_storm_precipitation(storm_depths(brk, met_modifiers=mm), dates)
    return {
        "site": site,
        "dates": dates,
        "tmin": tmin,
        "tmax": tmax,
        "srad": rad.srad,
        "precipitation": precip,
        "doy": dates.dayofyear.to_numpy().astype(float),
        "sno": sno,
        "ipet": scenario_ipet(scen),
        "latitude": geo["latitude_rad"],
        "metmod": mm,
        "met_rain_cm": met["rain_mm"].to_numpy(dtype=float) / 10.0 if "rain_mm" in met else None,
    }


# --------------------------------------------------------------------------- the port, batched
def simulate(runs: Sequence[dict[str, Any]], swe0: Sequence[float]) -> list[dict[str, np.ndarray]]:
    """:func:`simulate_batch` per group of runs with the same static parameters (``IPET``)."""
    out: list[dict[str, np.ndarray] | None] = [None] * len(runs)
    for ipet in sorted({r.get("ipet", 0) for r in runs}):
        idx = [i for i, r in enumerate(runs) if r.get("ipet", 0) == ipet]
        for i, o in zip(idx, simulate_batch([runs[i] for i in idx], [swe0[i] for i in idx]), strict=True):
            out[i] = o
    return [o for o in out if o is not None]


def simulate_batch(runs: Sequence[dict[str, Any]], swe0: Sequence[float]) -> list[dict[str, np.ndarray]]:
    """Run the snow process over every site of ``runs`` in one ``vmap`` (float64).

    Each run gives ``tmin``, ``tmax``, ``srad``, ``precipitation``, ``doy`` (length T_i), ``sno`` and
    ``latitude``; shorter runs are padded at the end with warm dry days. Returns per run the daily
    ``swe``, ``smelt`` (melt + melt runoff), ``melt``, ``melt_runoff``, ``sublimation``, ``cover``,
    ``intercepted`` [cm] and ``pk_temp`` [degC] after each day.
    """
    n = max(len(r["tmin"]) for r in runs)

    def pad(x: np.ndarray, fill: float) -> np.ndarray:
        return np.concatenate([np.asarray(x, dtype=np.float64), np.full(n - len(x), fill)])

    forcing = SnowForcing(
        tmin=jnp.asarray(np.stack([pad(r["tmin"], PAD_T) for r in runs])),
        tmax=jnp.asarray(np.stack([pad(r["tmax"], PAD_T) for r in runs])),
        srad=jnp.asarray(np.stack([pad(r["srad"], PAD_T) for r in runs])),
        precipitation=jnp.asarray(np.stack([pad(r["precipitation"], 0.0) for r in runs])),
        doy=jnp.asarray(np.stack([pad(r["doy"], 1.0) for r in runs])),
    )
    plist = [
        PrmsSnowParams.from_sno(r["sno"], r["latitude"], ipet=r.get("ipet", 0), dtype=jnp.float64)
        for r in runs
    ]
    params = jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *plist)
    s0 = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[SnowState.initial(w, jnp.float64) for w in swe0])

    def one(p: Any, f: Any, s: Any) -> Any:
        def body(st: Any, ft: Any) -> tuple[Any, Any]:
            new = snow_prms(st, p, ft)
            return new, new

        return jax.lax.scan(body, s, jax.tree_util.tree_map(lambda x: x, f))[1]

    traj = jax.jit(jax.vmap(one))(params, forcing, s0)
    out = []
    for i, r in enumerate(runs):
        m = len(r["tmin"])
        o = traj.out
        out.append(
            {
                "swe": np.asarray(traj.swe[i, :m]),
                "melt": np.asarray(o.melt[i, :m]),
                "melt_runoff": np.asarray(o.melt_runoff[i, :m]),
                "smelt": np.asarray(o.melt[i, :m] + o.melt_runoff[i, :m]),
                "sublimation": np.asarray(o.sublimation[i, :m]),
                "cover": np.asarray(traj.cover[i, :m]),
                "intercepted": np.asarray(traj.intercepted[i, :m]),
                "pk_temp": np.asarray(traj.pk_temp[i, :m]),
                "swe_mm": np.asarray(o.swe[i, :m]),
            }
        )
    return out


# --------------------------------------------------------------------------- reference tables
def catpa_tables(data_dir: Path) -> dict[str, Any]:
    """The CA-TPA 2015-2023 PHYSCL entry/exit tables: inputs and the pack at the exit."""
    a = np.load(data_dir / PHYSCL_ENTRY)
    b = np.load(data_dir / PHYSCL_EXIT)
    assert np.all(np.asarray(a["n_calls"]) == 1) and np.array_equal(a["date"], b["date"])
    dates = rr.yyyyddd_dates(a["date"])
    return {
        "dates": dates,
        "tmin": np.asarray(a["v.TMIN"]),
        "tmax": np.asarray(a["v.TMAX"]),
        "srad": np.asarray(a["v.RTS"]),
        "precipitation": np.asarray(b["v.DAYRAIN"]),
        "doy": np.asarray(a["v.JDAY"]).astype(float),
        "swe_entry": np.asarray(a["v.SNOWPK"]),
        "swe": np.asarray(b["v.SNOWPK"]),
        "smelt": np.asarray(b["v.OMSEA"])[:, OMSEA_MELT],
        "swe_omsea92": np.asarray(b["v.OMSEA"])[:, COL_SWE - 1],
        "xlat": float(np.asarray(a["v.XLAT"])[0]),
    }


def ana_columns(
    path: Path, cols: Iterable[int]
) -> tuple[pd.DatetimeIndex, dict[int, np.ndarray], dict[int, float]]:
    """Daily values (rows after the initial ``YYYY.000`` row) and the initial row of ``.ana`` columns."""
    ds = read_ana(path)
    names = {int(k): v for k, v in ds.attrs["columns"].items()}
    dates = pd.DatetimeIndex(np.asarray(ds["time"].values[1:]).astype("datetime64[D]"))
    vals = {c: np.asarray(ds[names[c]].values[1:], dtype=float) for c in cols}
    first = {c: float(ds[names[c]].values[0]) for c in cols}
    return dates, vals, first


# --------------------------------------------------------------------------- statistics
def stats(ours: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Largest absolute and relative difference, days bit-identical, worst day."""
    s = rr.stats(ours, ref)
    return s


def printed(ours: np.ndarray, printed_vals: np.ndarray) -> dict[str, Any]:
    """Against a G15.6 column: error in half units of the last printed digit (``<= 1``: equal to print)."""
    return rr.printed_stats(ours, printed_vals)


def event_counts(swe: np.ndarray, smelt: np.ndarray) -> dict[str, int]:
    return {"pack_days": int((swe > 0.0).sum()), "melt_days": int((smelt > 0.0).sum())}


def yearly(dates: pd.DatetimeIndex, swe: np.ndarray, smelt: np.ndarray, frac_infil: float) -> dict[int, Any]:
    out: dict[int, Any] = {}
    for y in sorted(set(dates.year)):
        m = np.asarray(dates.year == y)
        out[int(y)] = {
            "pack_days": int((swe[m] > 0).sum()),
            "melt_days": int((smelt[m] > 0).sum()),
            "melt_cm": float(smelt[m].sum()),
            "melt_runoff_cm": float((smelt[m] - smelt[m] * frac_infil).sum()),
            "max_swe_cm": float(swe[m].max()),
        }
    return out


# --------------------------------------------------------------------------- reference-side limits
def rts_build_bound(data_dir: Path) -> float:
    """The reference's own build-to-build limit of RTS (relative): the rebuilt binary's dumped RTS
    against the shipped binary's ``.ana`` column 88, CA-TPA 2015-2023 (``radiation_reference``)."""
    b = rr.build_excess(rr.table_rts(data_dir / PHYSCL_ENTRY), rr.ana_rts(data_dir / CATPA_ANA))
    return float(b["max_rel_excess"])


def excess_abs(full: np.ndarray, printed_vals: np.ndarray) -> np.ndarray:
    """Per day: how far ``full`` is from the printed value beyond half a printed unit [same unit]."""
    return np.maximum(np.abs(np.asarray(full) - printed_vals) - rr.print_half_unit(printed_vals), 0.0)


def perturbed(run: dict[str, Any], factor: float) -> dict[str, Any]:
    return {**run, "srad": np.asarray(run["srad"]) * factor}


def spread(
    nominal: dict[str, np.ndarray], plus: dict[str, np.ndarray], minus: dict[str, np.ndarray], k: str
) -> np.ndarray:
    """Per day: how far the output ``k`` moves when RTS moves by the build bound either way."""
    return np.maximum(np.abs(plus[k] - nominal[k]), np.abs(minus[k] - nominal[k]))


# --------------------------------------------------------------------------- CA-TPA comparisons
def compare_catpa_dumps(data_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """CA-TPA 2015-2023 through both input chains against the full-double PHYSCL exit tables."""
    t = catpa_tables(data_dir)
    files = site_inputs(data_dir, "CA-TPA", t["dates"])
    assert files["latitude"] == t["xlat"]
    sno = files["sno"]
    dump_run = {k: t[k] for k in ("tmin", "tmax", "srad", "precipitation", "doy")}
    dump_run.update(sno=sno, latitude=t["xlat"], ipet=files["ipet"])
    bound = rts_build_bound(data_dir)
    s0 = t["swe_entry"][0]
    ours_dump, ours_files, f_plus, f_minus = simulate(
        [dump_run, files, perturbed(files, 1.0 + bound), perturbed(files, 1.0 - bound)], [s0] * 4
    )
    rep: dict[str, Any] = {"n_days": len(t["dates"]), "rts_build_bound": bound, "ipet": files["ipet"]}
    rep["inputs_files_vs_dump"] = {
        "tmin": stats(files["tmin"], t["tmin"]),
        "tmax": stats(files["tmax"], t["tmax"]),
        "srad (RTS)": stats(files["srad"], t["srad"]),
        "precipitation (BRK storms vs DAYRAIN)": stats(files["precipitation"], t["precipitation"]),
    }
    rep["reference"] = {
        **event_counts(t["swe"], t["smelt"]),
        "omsea92_equals_snowpk": bool(np.array_equal(t["swe_omsea92"], t["swe"])),
        "yearly": yearly(t["dates"], t["swe"], t["smelt"], sno.frac_infil),
    }
    for name, o in (("dump_chain", ours_dump), ("files_chain", ours_files)):
        rep[name] = {
            "swe": stats(o["swe"], t["swe"]),
            "smelt": stats(o["smelt"], t["smelt"]),
            **event_counts(o["swe"], o["smelt"]),
            "pack_day_mismatch": int(((o["swe"] > 0) != (t["swe"] > 0)).sum()),
            "melt_day_mismatch": int(((o["smelt"] > 0) != (t["smelt"] > 0)).sum()),
            "yearly": yearly(t["dates"], o["swe"], o["smelt"], sno.frac_infil),
            "swe_mm_is_10x": bool(np.array_equal(o["swe_mm"], o["swe"] * 10.0)),
            "melt_split_max_abs": float(np.abs(o["melt"] - sno.frac_infil * o["smelt"]).max()),
            "sublimation_cm": float(o["sublimation"].sum()),
            "balance_max_abs": float(
                np.abs(
                    np.diff(np.concatenate([[s0], o["swe"]]))
                    - (o["intercepted"] - o["smelt"] - o["sublimation"])
                ).max()
            ),
        }
    for k in ("swe", "smelt"):
        sp = spread(ours_files, f_plus, f_minus, k)
        err = np.abs(ours_files[k] - t[k])
        rep["files_chain"][f"{k}_rts_spread_max"] = float(sp.max())
        rep["files_chain"][f"{k}_beyond_rts_spread_max"] = float(np.maximum(err - sp, 0.0).max())
        rep["files_chain"][f"{k}_n_beyond_rts_spread"] = int((err > sp).sum())
    daily = pd.DataFrame(
        {
            "date": t["dates"].strftime("%Y-%m-%d"),
            "tmin": t["tmin"],
            "tmax": t["tmax"],
            "rts": t["srad"],
            "dayrain": t["precipitation"],
            "swe_ref": t["swe"],
            "swe_dump_chain": ours_dump["swe"],
            "swe_files_chain": ours_files["swe"],
            "swe_rts_spread": spread(ours_files, f_plus, f_minus, "swe"),
            "smelt_ref": t["smelt"],
            "smelt_dump_chain": ours_dump["smelt"],
            "smelt_files_chain": ours_files["smelt"],
            "smelt_rts_spread": spread(ours_files, f_plus, f_minus, "smelt"),
            "sublimation": ours_dump["sublimation"],
            "cover": ours_dump["cover"],
            "intercepted": ours_dump["intercepted"],
            "pk_temp": ours_dump["pk_temp"],
        }
    )
    return rep, daily


def compare_catpa_ana(data_dir: Path, daily: pd.DataFrame) -> dict[str, Any]:
    """The shipped binary's CA-TPA ``.ana`` (columns 92, 105) against the port and the dump tables.

    ``build_excess_abs``: the largest distance of the rebuilt binary's full-double value (the dump)
    from the shipped binary's printed value beyond half a printed unit: the reference's own
    build-to-build limit on that column."""
    dates, vals, _ = ana_columns(data_dir / CATPA_ANA, (COL_SWE, COL_MELT))
    assert list(dates.strftime("%Y-%m-%d")) == list(daily["date"])
    out: dict[str, Any] = {}
    for col, k in ((COL_SWE, "swe"), (COL_MELT, "smelt")):
        p = vals[col]
        ref = daily[f"{k}_ref"].to_numpy()
        ex = excess_abs(ref, p)
        nz = p != 0.0
        out[f"col{col}"] = {
            "build_excess_abs": float(ex.max()),
            "build_excess_rel": float((ex[nz] / np.abs(p[nz])).max()),
            "n_days_dump_beyond_print": int((ex > 0).sum()),
            "port_vs_printed": printed(daily[f"{k}_dump_chain"].to_numpy(), p),
            "dump_vs_printed": printed(ref, p),
            "files_chain_vs_printed": printed(daily[f"{k}_files_chain"].to_numpy(), p),
            "port_excess_abs_max": float(excess_abs(daily[f"{k}_dump_chain"].to_numpy(), p).max()),
            "files_chain_excess_abs_max": float(excess_abs(daily[f"{k}_files_chain"].to_numpy(), p).max()),
        }
    return out


def compare_pktemp(data_dir: Path, bound: float, rel_build: float) -> dict[str, Any]:
    """PKTEMP at the POTEVPHR entry (the pack temperature of the previous snow call) of the 3-year
    PET scenario runs against the port's pack temperature (files chain), per site; with how far
    the port's value moves when RTS moves by the build bound ``bound`` either way. PKTEMP is not
    printed, so its build-to-build limit is taken relative: ``rel_build`` (the relative excess of
    the pack water equivalent, column 92) times the value, and at least one double unit at the
    scale of the site's pack temperatures (the resolution the reference carries it with)."""
    out: dict[str, Any] = {}
    for site in rr.pet_scen_sites(data_dir):
        t = np.load(data_dir / PET_SCEN / f"{site}_entry.npz")
        code = np.asarray(t["date"]).astype(int)
        dates = rr.yyyyddd_dates(code)
        pk = np.asarray(t["v.PKTEMP"])
        run = site_inputs(data_dir, site, dates)
        o, op, om = simulate([run, perturbed(run, 1.0 + bound), perturbed(run, 1.0 - bound)], [0.0] * 3)

        def prev(x: dict[str, np.ndarray]) -> np.ndarray:
            # the entry value of day d is the pack temperature after the call of day d-1
            return np.concatenate([[0.0], x["pk_temp"][:-1]])

        sp = np.maximum(np.abs(prev(op) - prev(o)), np.abs(prev(om) - prev(o)))
        err = np.abs(prev(o) - pk)
        lim = sp + rel_build * np.abs(pk) + np.finfo(np.float64).eps * float(np.abs(pk).max())
        out[site] = {
            "days": f"{code[0]}..{code[-1]}",
            "pack_days_port": int((o["swe"] > 0).sum()),
            "pktemp_nonzero_ref": int((pk != 0).sum()),
            "pktemp": stats(prev(o), pk),
            "rts_spread_max": float(sp.max()),
            "n_beyond_rts_spread": int((err > sp).sum()),
            "beyond_rts_spread_max": float(np.maximum(err - sp, 0.0).max()),
            "n_beyond_limit": int((err > lim).sum()),
            "beyond_limit_max": float(np.maximum(err - lim, 0.0).max()),
        }
    return out


# --------------------------------------------------------------------------- fresh scenario runs
def run_scenarios(data_dir: Path, sites: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Fresh full-period runs (the ``IPNAMES.DAT`` period) of the batch scenarios, in a thread pool."""
    import test_rzwqm_all_scenarios as tras

    sites = list(sites)
    out_root = Path(tempfile.mkdtemp(prefix="aj_w2_snow_out_"))
    stage_root = Path(tempfile.mkdtemp(prefix="aj_w2_snow_src_"))

    def one(site: str) -> dict[str, Any]:
        rec: dict[str, Any] = {"site": site, "ok": False}
        try:
            scenario = data_dir / BATCH / site / "Scenario"
            run_root = tras._short_run_root(data_dir)
            src = tras._stage_source(site, scenario, stage_root, run_root)
            t0 = time.perf_counter()
            r = rf.run_rzwqm(src, out_root / site, keep_files=("*.ana",), timeout=1800, run_root=run_root)
            rec.update(ok=True, ana=str(r.ana_path), wall_s=time.perf_counter() - t0)
        except Exception as e:  # reported per scenario
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    with ThreadPoolExecutor(max(1, min(len(sites), os.cpu_count() or 1))) as ex:
        return {r["site"]: r for r in ex.map(one, sites)}


def compare_scenarios(
    data_dir: Path, recs: dict[str, dict[str, Any]], bound: float, build_abs: dict[str, float]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Every fresh run: inputs against the printed ``.ana``, then SNP and SMELT.

    Per day the limit is half a printed unit, plus the reference's build-to-build excess of the
    column (``build_abs``, measured on CA-TPA), plus how far the port's value moves when RTS moves
    by the reference's RTS build bound ``bound`` either way."""
    runs, swe0, meta = [], [], []
    rep: dict[str, Any] = {}
    for site, rec in recs.items():
        if not rec.get("ok"):
            rep[site] = {"error": rec.get("error", "")}
            continue
        dates, vals, first = ana_columns(
            Path(rec["ana"]), (COL_PRECIP, COL_TMIN, COL_TMAX, COL_SWE, COL_MELT)
        )
        run = site_inputs(data_dir, site, dates)
        mm = run["metmod"]
        rep[site] = {
            "period": f"{dates[0].date()}..{dates[-1].date()}",
            "n_days": len(dates),
            "metmod_identity": bool(np.all(mm[:2] == 0.0) and np.all(mm[2:] == 100.0)),
            "precipitation": printed(run["precipitation"], vals[COL_PRECIP]),
            "tmin": printed(run["tmin"], vals[COL_TMIN]),
            "tmax": printed(run["tmax"], vals[COL_TMAX]),
            "ipet": run["ipet"],
            "reference": {
                **event_counts(vals[COL_SWE], vals[COL_MELT]),
                "max_swe_cm": float(vals[COL_SWE].max()),
                "initial_swe_cm": first[COL_SWE],
            },
        }
        runs.append(run)
        swe0.append(first[COL_SWE])
        meta.append((site, dates, vals))
    batch = [*runs, *[perturbed(r, 1.0 + bound) for r in runs], *[perturbed(r, 1.0 - bound) for r in runs]]
    outs = simulate(batch, swe0 * 3) if runs else []
    n = len(runs)
    tables: dict[str, pd.DataFrame] = {}
    for i, (site, dates, vals) in enumerate(meta):
        o, op, om = outs[i], outs[n + i], outs[2 * n + i]
        r = rep[site]
        r["port"] = {
            **event_counts(o["swe"], o["smelt"]),
            "swe": printed(o["swe"], vals[COL_SWE]),
            "smelt": printed(o["smelt"], vals[COL_MELT]),
            "pack_day_mismatch": int(((o["swe"] > 0) != (vals[COL_SWE] > 0)).sum()),
            "melt_day_mismatch": int(((o["smelt"] > 0) != (vals[COL_MELT] > 0)).sum()),
            "sublimation_cm": float(o["sublimation"].sum()),
        }
        for k, col in (("swe", COL_SWE), ("smelt", COL_MELT)):
            ex = excess_abs(o[k], vals[col])
            lim = build_abs[k] + spread(o, op, om, k)
            r["port"][f"{k}_excess_abs_max"] = float(ex.max())
            r["port"][f"{k}_rts_spread_max"] = float(spread(o, op, om, k).max())
            r["port"][f"{k}_n_beyond_limit"] = int((ex > lim).sum())
            r["port"][f"{k}_beyond_limit_max"] = float(np.maximum(ex - lim, 0.0).max())
        if r["reference"]["pack_days"] or r["port"]["pack_days"]:
            tables[site] = pd.DataFrame(
                {
                    "date": dates.strftime("%Y-%m-%d"),
                    "swe_ana": vals[COL_SWE],
                    "swe_port": o["swe"],
                    "smelt_ana": vals[COL_MELT],
                    "smelt_port": o["smelt"],
                }
            )
    return rep, tables


# --------------------------------------------------------------------------- the 2022 weather anomaly
def _winter(dates: pd.DatetimeIndex) -> np.ndarray:
    """DJFM winter label: December counts to the next year's winter; 0 outside DJFM."""
    m, y = dates.month.to_numpy(), dates.year.to_numpy()
    return np.where(m == 12, y + 1, np.where(m <= 3, y, 0))


def catpa_winter_weather(data_dir: Path) -> dict[str, Any]:
    """Precipitation on freezing days per DJFM winter at CA-TPA: the reference weather (``.MET``
    temperatures, ``.BRK`` storms, ``.MET`` rain column), ERA5 (FLUXNET ERA5 daily) and the tower
    weighing gauge ``P_PI_1`` with the tower albedo (2020-2023)."""
    dates = pd.date_range("2014-12-01", "2023-12-31", freq="D")
    run = site_inputs(data_dir, "CA-TPA", dates)
    tavg = (run["tmin"] + run["tmax"]) / 2.0
    frz = tavg <= 0.0
    w = _winter(dates)
    rep: dict[str, Any] = {"winters": {}}
    era = None
    if (data_dir / ERA5_DD).is_file():
        e = pd.read_csv(data_dir / ERA5_DD, na_values=[-9999])
        e.index = pd.to_datetime(e["TIMESTAMP"].astype(str), format="%Y%m%d")
        era = e.reindex(dates)
    gauge = None
    albedo_days = None
    if (data_dir / BASE_HH).is_file():
        h = pd.read_csv(data_dir / BASE_HH, comment="#", na_values=[-9999])
        ts = pd.to_datetime(h["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M")
        h.index = ts
        p = h["P_PI_1"]
        cnt = p.groupby(ts.dt.floor("D").to_numpy()).count()
        gauge = p.groupby(ts.dt.floor("D").to_numpy()).sum(min_count=1).where(cnt >= 46).reindex(dates) / 10.0
        hour = ts.dt.hour.to_numpy()
        mid = h[(hour >= 10) & (hour < 14)]
        alb = (mid["SW_OUT"] / mid["SW_IN"]).where(mid["SW_IN"] > 50.0)
        albedo_days = alb.groupby(mid.index.floor("D")).mean().reindex(dates)
    for y in range(2015, 2024):
        m = w == y
        r: dict[str, Any] = {
            "days": int(m.sum()),
            "freezing_days_met": int((m & frz).sum()),
            "brk_precip_cm": float(run["precipitation"][m].sum()),
            "brk_precip_freezing_cm": float(run["precipitation"][m & frz].sum()),
        }
        if run["met_rain_cm"] is not None:
            r["met_rain_cm"] = float(run["met_rain_cm"][m].sum())
            r["met_rain_freezing_cm"] = float(run["met_rain_cm"][m & frz].sum())
        if era is not None:
            ta, pe = era["TA_ERA"].to_numpy(), era["P_ERA"].to_numpy() / 10.0
            r["era5_precip_cm"] = float(np.nansum(pe[m]))
            r["era5_freezing_days"] = int((m & (ta <= 0.0)).sum())
            r["era5_precip_freezing_cm"] = float(np.nansum(pe[m & (ta <= 0.0)]))
            r["era5_precip_on_met_freezing_days_cm"] = float(np.nansum(pe[m & frz]))
        if gauge is not None:
            g = gauge.to_numpy()
            ok = m & np.isfinite(g)
            if ok.any():
                r["gauge_days"] = int(ok.sum())
                r["gauge_precip_cm"] = float(g[ok].sum())
                r["gauge_precip_on_met_freezing_days_cm"] = float(g[ok & frz].sum())
                r["brk_precip_on_gauge_days_cm"] = float(run["precipitation"][ok].sum())
        if albedo_days is not None:
            a = albedo_days.to_numpy()
            ok = m & np.isfinite(a)
            if ok.any():
                r["albedo_days_with_data"] = int(ok.sum())
                r["albedo_gt_0p5_days"] = int((ok & (np.nan_to_num(a) > 0.5)).sum())
        rep["winters"][str(y)] = r
    return rep


# --------------------------------------------------------------------------- report
def report(data_dir: Path, *, runs: bool = True) -> dict[str, Any]:
    out_dir = data_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict[str, Any] = {}
    rep["catpa_2015_2023"], daily = compare_catpa_dumps(data_dir)
    daily.to_csv(out_dir / "snow_catpa_2015_2023_daily.csv", index=False)
    rep["catpa_2015_2023_ana"] = compare_catpa_ana(data_dir, daily)
    rep["pet_scen_pktemp"] = compare_pktemp(
        data_dir,
        rep["catpa_2015_2023"]["rts_build_bound"],
        rep["catpa_2015_2023_ana"][f"col{COL_SWE}"]["build_excess_rel"],
    )
    rep["catpa_winter_weather"] = catpa_winter_weather(data_dir)
    if runs:
        recs = run_scenarios(data_dir, SITES)
        build_abs = {
            "swe": rep["catpa_2015_2023_ana"][f"col{COL_SWE}"]["build_excess_abs"],
            "smelt": rep["catpa_2015_2023_ana"][f"col{COL_MELT}"]["build_excess_abs"],
        }
        rep["scenarios"], tables = compare_scenarios(
            data_dir, recs, rep["catpa_2015_2023"]["rts_build_bound"], build_abs
        )
        for s, df in tables.items():
            df.to_csv(out_dir / f"snow_ana_{s}_daily.csv", index=False)
    (out_dir / "snow_reference.json").write_text(json.dumps(rep, indent=1, default=_json))
    return rep


def _json(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x))


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--data-dir", default=os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data"))
    ap.add_argument("--no-runs", action="store_true", help="skip the fresh scenario runs")
    a = ap.parse_args()
    print(json.dumps(report(Path(a.data_dir).expanduser(), runs=not a.no_runs), indent=1, default=_json))
