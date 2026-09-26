"""Shuttleworth-Wallace PET against RZWQM2 on the 15 scenarios of ``RZWQM_sw_batch``, three years each.

Helper module of ``test_pet_scenarios.py`` and a script (``python tests/integration/pet_scenarios.py``)
that writes the per-scenario, per-year table. For every scenario the RZWQM2 binary is run for the
first three calendar years of its ``IPNAMES.DAT`` period (:func:`agrijax.port.run_fortran.run_rzwqm`,
``.ana`` + ``LAYER.PLT`` + ``MANAGE.OUT`` kept); the daily S-W module
(:func:`agrijax.processes.pet.shuttleworth_wallace`) is then driven with the reference run's own
start-of-day state, turned into module inputs with the conventions measured on CA-TPA 2015
(``poc/coarse_compare.py``, generalised here to several seasons and crops):

* weather: ``.ana`` cols 85/86/88/89/90 (the reference model's echo: identical forcing, module-only
  check) and, second, the prepared ``.MET`` (:func:`prepare_rzwqm_forcing` with the site latitude,
  slope and aspect: the whole input chain). Col 88 is the field radiation ``RTS`` (hourly re-sum);
  the measured horizontal radiation ``RTH`` of the net long-wave cloudiness ratio is not printed
  and is taken from the ``.MET`` file in both cases (``srad_horizontal`` of the kernel);
* LAI (col 43) and canopy height (col 62) of the previous ``.ana`` row (crop growth follows the PET
  call); the day after a harvest (``MANAGE.OUT``) starts with no canopy;
* flat residue mass (col 72) of the previous row, except on a tillage day (``MANAGE.OUT``: management
  precedes the PET call) and on the first day (``rzwqm.dat`` initial residue; ``.ana`` row 0 is
  all zeros);
* residue age: the file's initial age + 1 per day (Rzday.for line 1284, before the PET call),
  restarting at every harvest (``RESAGE = 0``); residue type: the ``CRES`` default (Rzmain.for
  lines 5245-5247) until the first harvest, then the harvested crop's ``IRTYPE`` (plant-name table,
  Rzmain.for lines 5870-5917): the residue-type index takes the harvested plant's value when the
  residue age has just been reset to zero (Rzman.for line 5936);
* surface-node water content: ``LAYER.PLT`` node 1 at the end of the previous day; on a tillage
  day (primary or secondary operation, or an incorporation event) the tillage-zone mean of the
  previous day's profile, as ``MATILL`` mixes the soil water before the PET call (Rzman.for
  lines 2704-3121);
* minimum stomatal resistance ``RST(IPL)`` of the plant in the field (the plant of the latest
  planting), one per plant row of the SITE SPECIFIC PARAMETERS block (RESISThr, Rzpet.for).

Scenario classes (from ``rzwqm.dat``): the daily S-W path (``pet_method = 0``, hourly weather,
SHAW and PENFLUX off) is the module's; ``pet_method = 1`` (crop coefficient x tall reference ET,
``C_R_S_Cover``) and ``hourly_weather = 1`` (``POTEVPHR`` once per hour on the hourly weather,
Rzday.for lines 1197-1276) are other methods of the reference model, reported but not a module
check.

The kernel itself is checked against the reference routine's own inputs and outputs
(``POTEVPHR`` entry/exit dumps of the same scenarios and years, ``test_pet_dumps.py``); this
harness adds the reconstruction of those inputs from printed output, whose error on the dumped
run is measured input by input there (:func:`pet_dumps.attribution`).

Compared with ``.ana`` col 8 (potential evaporation PE = soil + residue), col 9 (potential
transpiration PT) and col 83 (PET = PE + PT), converted from cm to mm d-1. Day 1 of each run is
reported but left out of the statistics: its start-of-day state (initial canopy of a perennial or
winter crop, initial surface water content) is not printed anywhere.

Reference-side limits: the ``.ana`` prints 6 significant digits (G15.6), i.e. <= 5e-7 cm = 5e-6 mm
on a 1 mm/d flux, and the inputs are echoed with the same precision.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # script use: import the sibling test module's staging helpers
    sys.path.insert(0, str(HERE))

BATCH = Path("narval_mirror/RZWQM_sw_batch")
OUT_SUBDIR = Path("validation/pet_scenarios")
N_YEARS = 3
KEEP = ("*.ana", "LAYER.PLT", "MANAGE.OUT", "OVERVIEW.OUT", "WEATHERD.OUT")
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
PET_COLUMNS = {"pe": 8, "pt": 9, "pet": 83}
WEATHER_COLUMNS = {"tmin": 85, "tmax": 86, "srad": 88, "rh": 89, "wind_run": 90}
MET_NAMES = {"tmin": "tmin", "tmax": "tmax", "srad": "srad_mj", "rh": "rh", "wind_run": "wind_run_km"}
#: residue type (``IPR``) of the ``CRES`` values Rzmain.for lines 5245-5247 recognise; else IPR = 1
RESIDUE_TYPE_OF_CRES = {2.0: "corn", 2.5: "soybean", 4.0: "wheat"}
RESIDUE_TYPE_OF_IPR = {1: "corn", 2: "soybean", 3: "wheat"}
#: ``IRTYPE`` of a plant by name (Rzmain.for lines 5870-5917, first match wins; all others 2)
IRTYPE_BY_NAME = (
    (("corn", "maize"), 1),
    (("soybean",), 2),
    (("wheat",), 3),
    (("bermudagrass",), 3),
    (("apple",), 3),
    (("sunflower",), 1),
    (("alfalfa",), 2),
    (("cotton",), 1),
    (("sugarcane",), 1),
    (("sorghum",), 1),
)
IRTYPE_DEFAULT = 2


def irtype(plant_name: str) -> int:
    """Residue-type index of a plant, Rzmain.for lines 5870-5917 (case-insensitive substring match)."""
    s = plant_name.lower()
    for keys, t in IRTYPE_BY_NAME:
        if any(k in s for k in keys):
            return t
    return IRTYPE_DEFAULT


# ------------------------------------------------------------------------------ MANAGE.OUT

_DATE_HDR = re.compile(r"^-{3,}\s*(\d{1,2})/\s*(\d{1,2})/\s*(\d{4})\s*-{3,}")
_HARVEST_HDR = re.compile(r"^-{3,}(?P<what>[^-].*?)-{3,}(?P<crop>.*)$")
_ON_DATE = re.compile(r"^\s*ON\s+(\d{1,2})/\s*(\d{1,2})/\s*(\d{4})")


@dataclass
class Management:
    """Dated events of a run, from the reference model's own ``MANAGE.OUT`` log."""

    tillage: list[pd.Timestamp] = field(default_factory=list)
    tillage_implement: list[str] = field(default_factory=list)  # "WITH IMPLEMENT:" text, same order
    tillage_kind: list[str] = field(default_factory=list)  # event text, same order
    harvest: list[tuple[pd.Timestamp, str]] = field(default_factory=list)  # (date, crop text)
    kinds: dict[str, int] = field(default_factory=dict)  # every event / harvest header seen


def read_manage_out(path: Path) -> Management:
    m = Management()
    cur: pd.Timestamp | None = None
    pending: str | None = None
    for ln in path.read_bytes().decode("latin-1").splitlines():
        d = _DATE_HDR.match(ln)
        if d:
            cur = pd.Timestamp(int(d.group(3)), int(d.group(2)), int(d.group(1)))
            continue
        if pending is not None:
            on = _ON_DATE.match(ln)
            if on:
                m.harvest.append(
                    (pd.Timestamp(int(on.group(3)), int(on.group(2)), int(on.group(1))), pending)
                )
                pending = None
                continue
        h = _HARVEST_HDR.match(ln.strip())
        if h and "HARVEST" in h.group("what").upper():
            key = h.group("what").strip()
            m.kinds[key] = m.kinds.get(key, 0) + 1
            pending = h.group("crop").strip()
            continue
        s = ln.strip()
        if s.upper().startswith("WITH IMPLEMENT:") and len(m.tillage_implement) < len(m.tillage):
            m.tillage_implement.append(s.split(":", 1)[1].strip().lower())
            continue
        if s.startswith("EVENT ==>"):
            what = s[len("EVENT ==>") :].strip()
            m.kinds[what] = m.kinds.get(what, 0) + 1
            if "TILLAGE" in what.upper() and cur is not None:
                m.tillage.append(cur)
                m.tillage_kind.append(what.upper())
                if "INCORPORATION" in what.upper():  # no implement line follows (MANOUT)
                    m.tillage_implement.append("")
            if "HARVEST" in what.upper() and cur is not None:
                m.harvest.append((cur, what))
    return m


# ------------------------------------------------------------------------------ runs


def _short_run_root(data_dir: Path) -> Path:
    return Path(os.environ.get("AGRI_JAX_RUN_ROOT", str(Path(data_dir) / "run")))


def run_site(
    site: str, data_dir: Path, out_root: Path, stage_root: Path, n_years: int = N_YEARS
) -> dict[str, Any]:
    """Stage + run one scenario for ``n_years`` calendar years; never raises (failures returned)."""
    from test_rzwqm_all_scenarios import _stage_source

    import agrijax.port.run_fortran as rf
    from agrijax.io.rzwqm.layers import simulation_start

    rec: dict[str, Any] = {"site": site, "ok": False, "error": ""}
    out = out_root / site
    try:
        scenario = data_dir / BATCH / site / "Scenario"
        run_root = _short_run_root(data_dir)
        src = _stage_source(site, scenario, stage_root, run_root)
        start = simulation_start(src / "IPNAMES.DAT").astype(object)
        assert isinstance(start, _dt.date)
        end = _dt.date(start.year + n_years - 1, 12, 31)
        rec.update(start=start, end=end, source=src, out=out)
        t0 = time.perf_counter()
        rf.run_rzwqm(src, out, start=start, end=end, keep_files=KEEP, timeout=1800, run_root=run_root)
        rec["wall_s"] = time.perf_counter() - t0
        rec["ok"] = True
    except Exception as e:  # reported per scenario
        rec["error"] = f"{type(e).__name__}: {e}"
        log = out / "run.log"
        if log.is_file():
            rec["log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
    return rec


def run_all(
    data_dir: Path, out_root: Path, stage_root: Path, sites: Sequence[str] = SITES
) -> dict[str, dict]:
    workers = max(1, min(len(sites), os.cpu_count() or 1))
    with ThreadPoolExecutor(workers) as ex:  # threads: the work is the Fortran subprocess
        futs = [ex.submit(run_site, s, data_dir, out_root, stage_root) for s in sites]
        return {r["site"]: r for r in (f.result() for f in futs)}


# ------------------------------------------------------------------------------ inputs


def _data_lines_after(lines: Sequence[str], marker: str) -> list[list[str]]:
    for i, ln in enumerate(lines):
        if marker in ln:
            out: list[list[str]] = []
            for nxt in lines[i + 1 :]:
                s = nxt.strip()
                if s.startswith("="):
                    if out:
                        break
                    continue
                if s:
                    out.append(s.split())
            return out
    raise LookupError(marker)


def residue_conditions(dat: Any) -> dict[str, float]:
    vals = [float(t) for t in _data_lines_after(dat.lines, "C:P ratio of dominate residue material")[0]]
    keys = (
        "mass_t_ha",
        "age_d",
        "cover_factor",
        "height_cm",
        "cn_ratio",
        "sai",
        "standing_height_cm",
        "standing_mass_t_ha",
        "cp_ratio",
    )
    return dict(zip(keys, vals, strict=False))


def tillage_records(dat: Any) -> list[dict[str, Any]]:
    """Records of the ``rzwqm.dat`` tillage block (items 2.1-2.8 of its header comment)."""
    from agrijax.io.rzwqm.events import TILL_IMPLEMENT

    recs = _data_lines_after(dat.lines, "number of tillage operations")
    n = int(float(recs[0][0]))
    out = []
    for r in recs[1 : 1 + n]:
        when = int(float(r[1]))
        j = 5 if when == 5 else 3  # a fixed date takes three tokens (dd mm yyyy), an offset one
        imp = int(float(r[j]))
        out.append(
            dict(
                when=when,
                implement=imp,
                implement_name=TILL_IMPLEMENT.get(imp, "").lower(),
                depth_cm=float(r[j + 1]),
                intensity=float(r[j + 2]),
                operation=int(float(r[j + 3])),
            )
        )
    return out


def _find_met(src: Path) -> Path:
    from test_rzwqm_all_scenarios import _find_ci

    ip = (src / "IPNAMES.DAT").read_bytes().decode("latin-1").splitlines()
    name = re.split(r"[\\/]", ip[2].strip())[-1]
    p = _find_ci(src, name)
    if p is None:
        raise FileNotFoundError(f"{name} not in {src}")
    return p


@dataclass
class SiteInputs:
    site: str
    days: pd.DatetimeIndex
    weather: dict[str, dict[str, np.ndarray]]  # source -> var -> array
    lai: np.ndarray
    height: np.ndarray
    theta: np.ndarray
    residue_mass: np.ndarray
    residue_age: np.ndarray
    residue_type: np.ndarray
    ref: dict[str, np.ndarray]  # mm d-1
    extra: dict[str, np.ndarray]  # diagnostics: snow depth, standing dead, ...
    site_consts: dict[str, Any]
    flags: dict[str, Any]
    manage: Management


def site_inputs(rec: dict[str, Any], data_dir: Path) -> SiteInputs:
    from agrijax.io.rzwqm import prepare_rzwqm_forcing, read_ana, read_met, read_rzwqm_dat
    from agrijax.io.rzwqm.layers import read_layer_output

    site, out, src = rec["site"], Path(rec["out"]), Path(rec["source"])
    ana_path = next(p for p in out.iterdir() if p.suffix.lower() == ".ana")
    ds = read_ana(ana_path)
    col = {int(k): v for k, v in ds.attrs["columns"].items()}
    ana_days = pd.DatetimeIndex(ds.time.values)
    days = ana_days[1:]  # row 0 = YYYY.000 initial state (all zeros)

    def c(k: int) -> np.ndarray:
        return ds[col[k]].values.astype(float)

    dat = read_rzwqm_dat(data_dir / BATCH / site / "Scenario" / "rzwqm.dat")
    phys, pet, hyd = dat.physiography, dat.pet, dat.hydraulics
    manage = read_manage_out(out / "MANAGE.OUT") if (out / "MANAGE.OUT").is_file() else Management()
    met = prepare_rzwqm_forcing(
        read_met(_find_met(src)),
        latitude_rad=phys["latitude_rad"],
        slope_rad=phys["slope_rad"],
        aspect_rad=phys["aspect_rad"],
    ).reindex(days)
    weather = {
        "ana": {k: c(v)[1:] for k, v in WEATHER_COLUMNS.items()},
        "met": {k: met[v].to_numpy(dtype=float) for k, v in MET_NAMES.items()},
    }
    # RTH (measured horizontal radiation, INPDAY) is not echoed in the .ana: from the .MET file
    srad_h = met["srad_mj_met"].to_numpy(dtype=float)
    for w in weather.values():
        w["srad_h"] = srad_h
    n = len(days)
    lai, height = c(43)[:-1].copy(), c(62)[:-1].copy()  # previous row = start of day
    harvest_days = sorted({d for d, _ in manage.harvest})
    for h in harvest_days:
        nxt = h + pd.Timedelta(days=1)
        if nxt in days:
            k = days.get_loc(nxt)
            lai[k] = 0.0
            height[k] = 0.0
    residue = c(72)[:-1].copy()
    rc = residue_conditions(dat)
    residue[0] = rc["mass_t_ha"] * 1.0e3
    for t in manage.tillage:
        if t in days:
            k = days.get_loc(t)
            residue[k] = c(72)[k + 1]
    lay = read_layer_output(out / "LAYER.PLT", start=rec["start"])
    prof = lay["soil_water_content"].to_pandas().reindex(days).to_numpy(dtype=float)  # [day, node]
    th_end = prof[:, 0]
    theta = np.concatenate([th_end[:1], th_end[:-1]])
    theta_unmixed = theta.copy()
    # tillage day: MATILL (Rzman.for lines 2704-3121, before the PET call) replaces the water
    # content of every node of the tillage zone by the thickness-weighted zone mean; the zone is
    # horizon 1, or horizons 1-2 for a primary operation deeper than horizon 1 that is not an
    # incorporation event (lines 2834-2839); primary and secondary operations and incorporation
    # events mix (line 2849), other operations only compact
    recs = tillage_records(dat)
    hor = dat.horizon_depths_cm
    zdepth = np.asarray(dat.node_depths_cm, dtype=float)
    tl = np.diff(np.concatenate([[0.0], zdepth]))
    assert len(manage.tillage_implement) == len(manage.tillage) == len(manage.tillage_kind)
    for t, imp, kind in zip(manage.tillage, manage.tillage_implement, manage.tillage_kind, strict=True):
        if t not in days or days.get_loc(t) == 0:
            continue
        k = days.get_loc(t)
        if "INCORPORATION" in kind:
            iht = 1
        else:
            cand = [r for r in recs if r["implement_name"] == imp]
            if not cand:
                raise LookupError(f"{site}: tillage implement {imp!r} of {t.date()} not in rzwqm.dat")
            r0 = cand[0]
            if r0["operation"] not in (1, 2):
                continue
            iht = 2 if (r0["operation"] < 2 and r0["depth_cm"] > hor[0] and len(hor) > 1) else 1
        zone = zdepth <= hor[iht - 1] + 1e-9
        theta[k] = float((prof[k - 1, zone] * tl[zone]).sum() / hor[iht - 1])
    age = rc["age_d"] + np.arange(1, n + 1, dtype=float)
    rtype = np.array([RESIDUE_TYPE_OF_CRES.get(rc["cover_factor"], "corn")] * n, dtype=object)
    plants = dat.plants
    for h, crop_text in manage.harvest:
        after = days > h
        age[after] = (days[after] - h).days.to_numpy().astype(float)
        name = crop_text or ""
        matches = [p for p in plants if p.split(maxsplit=1)[-1].lower()[:5] in name.lower()]
        pname = matches[0] if matches else name
        rtype[after] = RESIDUE_TYPE_OF_IPR[irtype(pname)]
    # plant in the field (``IPL``): the plant of the latest planting on or before the day; its own
    # minimum stomatal resistance RST(IPL) enters RSC (RESISThr, Rzpet.for)
    rs_rows = [float(r["rs_min"]) for r in dat.plant_site_params]
    ipl = np.zeros(n, dtype=int)
    for p in sorted(dat.plantings, key=lambda p: p.planting_date):
        ipl[days >= pd.Timestamp(p.planting_date)] = p.plant_ref
    rs = np.array([rs_rows[i - 1] if 1 <= i <= len(rs_rows) else rs_rows[0] for i in ipl], dtype=float)
    ref = {k: c(v)[1:] * 10.0 for k, v in PET_COLUMNS.items()}
    extra = {
        "theta_unmixed": theta_unmixed,
        "ipl": ipl.astype(float),
        "rs_min": rs,
        "precip_cm": c(3)[1:],
        "irrig_cm": c(4)[1:],
        "infil_cm": c(5)[1:],
        "act_e_cm": c(6)[1:],
        "act_t_cm": c(7)[1:],
        "ice_cm": c(91)[1:],
        "residue_temp": c(93)[1:],
        "snow_depth_cm": c(92)[1:],
        "standing_dead_kg_ha": c(73)[1:],
        "residue_end_kg_ha": c(72)[1:],
        "lai_row": c(43)[1:],
        "ref_et_tall_mm": c(81)[1:] * 10.0,
        "ref_et_short_mm": c(82)[1:] * 10.0,
    }
    consts = dict(
        wc13=float(hyd["theta_fc33"][0]),
        wc15=float(hyd["theta_wp"][0]),
        elevation=float(phys["elevation_m"]),
        latitude=float(phys["latitude_rad"]),
        wind_height=float(pet["wind_height_m"]),
        rainfall_zone=int(phys["rainfall_zone"]),
        cover_factor=float(rc["cover_factor"]),
    )
    flags = {
        k: pet.get(k)
        for k in (
            "pet_method",
            "hourly_weather",
            "use_shaw",
            "use_penflux",
            "plastic_cover_frac",
            "wind_height_m",
            "soil_resistance",
        )
    }
    flags.update(
        co2_ppm=float(phys["co2_ppm"]),
        slope_rad=float(phys["slope_rad"]),
        plants=";".join(plants),
        residue_sai=rc.get("sai", 0.0),
        standing_residue_t_ha=rc.get("standing_mass_t_ha", 0.0),
    )
    return SiteInputs(
        site, days, weather, lai, height, theta, residue, age, rtype, ref, extra, consts, flags, manage
    )


# ------------------------------------------------------------------------------ S-W


def simulate(
    x: SiteInputs,
    source: str,
    *,
    theta: np.ndarray | None = None,
    srad_horizontal: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Daily PE, PT, PET [mm d-1] of the S-W module on the inputs ``x`` with weather ``source``.

    ``theta`` replaces the start-of-day surface water content, ``srad_horizontal`` the measured
    horizontal radiation ``RTH`` (default: the ``.MET`` value).
    """
    import jax
    import jax.numpy as jnp

    from agrijax.processes.pet import PETParams, shuttleworth_wallace
    from agrijax.processes.pet.shuttleworth_wallace import RESIDUE_DENSITY_G_CM3, RESIDUE_DIAMETER_CM

    params = x.site_consts["params"]
    assert isinstance(params, PETParams)
    w = x.weather[source]
    k = x.site_consts
    rdia = np.array([RESIDUE_DIAMETER_CM[t] for t in x.residue_type], dtype=float)
    rho = np.array([RESIDUE_DENSITY_G_CM3[t] for t in x.residue_type], dtype=float)
    srad = w["srad"]
    th_in = x.theta if theta is None else theta
    srh = w["srad_h"] if srad_horizontal is None else np.asarray(srad_horizontal, dtype=float)

    def one(tmin, tmax, sr, srh_d, rh, wind, lai, hc, th, rm, age, rd, rr, doy, rsd):
        p = PETParams(
            albedo_dry=params.albedo_dry,
            albedo_wet=params.albedo_wet,
            albedo_maturity=params.albedo_maturity,
            albedo_residue=params.albedo_residue,
            soil_resistance=params.soil_resistance,
            stomatal_resistance=rsd,  # RST(IPL) of the plant in the field
        )
        r = shuttleworth_wallace(
            tmin,
            tmax,
            sr,
            rh,
            wind,
            lai,
            hc,
            p,
            theta_surface=th,
            wc13=k["wc13"],
            wc15=k["wc15"],
            elevation=k["elevation"],
            latitude=k["latitude"],
            doy=doy,
            srad_horizontal=srh_d,
            residue_mass=rm,
            residue_age=age,
            wind_height=k["wind_height"],
            rainfall_zone=k["rainfall_zone"],
            residue_cover_factor=k["cover_factor"],
            residue_diameter_cm=rd,
            residue_density=rr,
        )
        return r

    r = jax.jit(jax.vmap(one))(
        *(jnp.asarray(v) for v in (w["tmin"], w["tmax"], srad, srh, w["rh"], w["wind_run"])),
        jnp.asarray(x.lai),
        jnp.asarray(x.height),
        jnp.asarray(th_in),
        jnp.asarray(x.residue_mass),
        jnp.asarray(x.residue_age),
        jnp.asarray(rdia),
        jnp.asarray(rho),
        jnp.asarray(x.days.dayofyear.to_numpy()),
        jnp.asarray(x.extra["rs_min"]),
    )
    pt = np.asarray(r.transpiration) * 10.0
    pe = np.asarray(r.soil_evaporation + r.residue_evaporation) * 10.0
    return {"pe": pe, "pt": pt, "pet": pe + pt, "rn": np.asarray(r.rn)}


def pet_params(dat: Any):
    import jax.numpy as jnp

    from agrijax.processes.pet import PETParams

    pet = dat.pet
    return PETParams(
        albedo_dry=jnp.asarray(pet["albedo_dry"]),
        albedo_wet=jnp.asarray(pet["albedo_wet"]),
        albedo_maturity=jnp.asarray(pet["albedo_crop"]),
        albedo_residue=jnp.asarray(pet["albedo_residue"]),
        soil_resistance=jnp.asarray(pet["soil_resistance"]),
        stomatal_resistance=jnp.asarray(dat.plant_site_params[0]["rs_min"]),
    )


def year_stats(x: SiteInputs, sim: dict[str, np.ndarray], source: str) -> list[dict[str, Any]]:
    """Per calendar year: max |error|, RMSE, bias [mm d-1] of PE, PT, PET over every day but day 1."""
    rows = []
    years = x.days.year.to_numpy()
    valid = np.ones(len(x.days), dtype=bool)
    valid[0] = False  # day 1: start-of-day state not printed
    for y in np.unique(years):
        m = (years == y) & valid
        row: dict[str, Any] = {"site": x.site, "year": int(y), "weather": source, "n_days": int(m.sum())}
        for v in ("pe", "pt", "pet"):
            e = sim[v][m] - x.ref[v][m]
            row[f"{v}_max_abs_mm"] = float(np.abs(e).max())
            row[f"{v}_rmse_mm"] = float(np.sqrt(np.mean(e**2)))
            row[f"{v}_bias_mm"] = float(e.mean())
            row[f"{v}_ref_sum_mm"] = float(x.ref[v][m].sum())
            k = int(np.argmax(np.abs(e)))
            row[f"{v}_worst_day"] = str(x.days[m][k].date())
            row[f"{v}_n_gt_1e-2"] = int((np.abs(e) > 1e-2).sum())
        rows.append(row)
    return rows


def daily_frame(x: SiteInputs, sims: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    d: dict[str, Any] = {"date": x.days.strftime("%Y-%m-%d")}
    for v in ("pe", "pt", "pet"):
        d[f"ref_{v}"] = x.ref[v]
        for s, sim in sims.items():
            d[f"{s}_{v}"] = sim[v]
    for s, sim in sims.items():
        for extra_key in ("rn",):
            if extra_key in sim:
                d[f"{s}_{extra_key}"] = sim[extra_key]
    d.update(
        lai=x.lai,
        height=x.height,
        theta=x.theta,
        residue=x.residue_mass,
        residue_age=x.residue_age,
        residue_type=x.residue_type,
    )
    for k, v in x.extra.items():
        d[k] = v
    for k, v in x.weather["ana"].items():
        d[f"ana_{k}"] = v
    return pd.DataFrame(d)


def reference_et(x: SiteInputs, source: str = "ana") -> dict[str, np.ndarray]:
    """ASCE tall / short reference ET [mm d-1] (``variant="rzwqm"``, REF_ET.FOR) with ``trat`` = 1."""
    from agrijax.processes.pet import asce_reference_et

    w = x.weather[source]
    k = x.site_consts
    r = asce_reference_et(
        w["tmin"],
        w["tmax"],
        w["srad"],
        w["rh"],
        w["wind_run"] * 1.0e3 / 86400.0,
        elevation=k["elevation"],
        latitude=k["latitude"],
        doy=x.days.dayofyear.to_numpy(),
        wind_height=k["wind_height"],
        variant="rzwqm",
    )
    return {"tall": np.asarray(r.et_tall), "short": np.asarray(r.et_short)}


def refet_stats(x: SiteInputs, ret: dict[str, np.ndarray]) -> dict[str, float]:
    """Max / RMSE of the reference ET against ``.ana`` cols 81/82, all days and days with LAI < 0.01.

    ``REF_ET`` receives ``TRAT`` (DSSAT ``TRATIO``, Rzday.for line 1195), which is 1 at 330 ppm CO2
    and whenever the total LAI is below 0.01; the module is run with ``trat = 1``.
    """
    valid = np.ones(len(x.days), dtype=bool)
    valid[0] = False
    bare = valid & (x.extra["lai_row"] < 0.01) & (x.lai < 0.01)
    out: dict[str, float] = {}
    for v, col in (("tall", "ref_et_tall_mm"), ("short", "ref_et_short_mm")):
        e = ret[v] - x.extra[col]
        out[f"refet_{v}_max_abs_mm"] = float(np.abs(e[valid]).max())
        out[f"refet_{v}_rmse_mm"] = float(np.sqrt(np.mean(e[valid] ** 2)))
        out[f"refet_{v}_max_abs_bare_mm"] = float(np.abs(e[bare]).max()) if bare.any() else float("nan")
    return out


def analyse(
    rec: dict[str, Any], data_dir: Path
) -> tuple[SiteInputs, dict[str, dict[str, np.ndarray]], list[dict]]:
    from agrijax.io.rzwqm import read_rzwqm_dat

    x = site_inputs(rec, data_dir)
    x.site_consts["params"] = pet_params(
        read_rzwqm_dat(data_dir / BATCH / rec["site"] / "Scenario" / "rzwqm.dat")
    )
    sims = {s: simulate(x, s) for s in ("ana", "met")}
    # ablations: tillage-day water content without the MATILL mixing; the field radiation RTS
    # also in the long-wave cloudiness ratio (one radiation input, as before 2026-09-25)
    sims["ana_unmixed"] = simulate(x, "ana", theta=x.extra["theta_unmixed"])
    sims["ana_rts_only"] = simulate(x, "ana", srad_horizontal=x.weather["ana"]["srad"])
    ret = reference_et(x)
    x.extra["our_ref_et_tall_mm"], x.extra["our_ref_et_short_mm"] = ret["tall"], ret["short"]
    rows = [r for s in ("ana", "met") for r in year_stats(x, sims[s], s)]
    rstats = refet_stats(x, ret)
    for r in rows:
        r.update(rstats)
    return x, sims, rows


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    import jax

    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
    ap.add_argument("--runs", default=None, help="run output root (default <data>/pet_scenario_runs)")
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--reuse", action="store_true", help="reuse existing run outputs")
    a = ap.parse_args(argv)
    data_dir = Path(a.data_dir)
    runs_root = Path(a.runs) if a.runs else data_dir / "pet_scenario_runs"
    stage_root = runs_root / "_stage"
    out_dir = data_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    if a.reuse:
        from agrijax.io.rzwqm.layers import simulation_start

        recs = {}
        for s in a.sites:
            src = stage_root / s / "Scenario"
            if not src.is_dir():
                src = data_dir / BATCH / s / "Scenario"
            st = simulation_start(src / "IPNAMES.DAT").astype(object)
            recs[s] = dict(
                site=s,
                ok=(runs_root / s / "LAYER.PLT").is_file(),
                start=st,
                end=_dt.date(st.year + N_YEARS - 1, 12, 31),
                source=src,
                out=runs_root / s,
                error="",
            )
    else:
        import shutil

        shutil.rmtree(stage_root, ignore_errors=True)
        stage_root.mkdir(parents=True)
        recs = run_all(data_dir, runs_root, stage_root, a.sites)
    rows: list[dict] = []
    for s in a.sites:
        r = recs[s]
        if not r["ok"]:
            print(f"{s}: RUN FAILED {r['error']}\n{r.get('log_tail', '')}")
            rows.append({"site": s, "error": r["error"]})
            continue
        try:
            x, sims, st = analyse(r, data_dir)
        except Exception as e:
            import traceback

            traceback.print_exc()
            rows.append({"site": s, "error": f"analysis {type(e).__name__}: {e}"})
            continue
        daily_frame(x, sims).to_csv(out_dir / f"{s}_daily.csv", index=False)
        for row in st:
            row.update({f"flag_{k}": v for k, v in x.flags.items()})
            row["manage_kinds"] = ";".join(f"{k}={v}" for k, v in x.manage.kinds.items())
            row["harvests"] = ";".join(f"{d.date()}:{t}" for d, t in x.manage.harvest)
        rows.extend(st)
        for row in st:
            print(
                f"{s:20s} {row['year']} {row['weather']:4s} PE max {row['pe_max_abs_mm']:.2e} rmse "
                f"{row['pe_rmse_mm']:.2e} | PT max {row['pt_max_abs_mm']:.2e} rmse {row['pt_rmse_mm']:.2e} "
                f"({row['pt_worst_day']}) | PET max {row['pet_max_abs_mm']:.2e} ({row['pe_worst_day']})"
            )
        print(f"{s}: flags {x.flags}; manage {x.manage.kinds}; harvests {x.manage.harvest}")
        print(
            f"{s}: reference ET "
            + " ".join(f"{k}={v:.2e}" for k, v in st[0].items() if k.startswith("refet"))
        )
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "pet_scenarios_3yr.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir / 'pet_scenarios_3yr.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
