"""Coarse comparison of Agri-JAX components against RZWQM2 on CA-TPA 2015.

What it does (no tuning; every parameter comes from the scenario files as they stand):

1. Reference run: a one-year (2015) CA-TPA run of the RZWQM2 binary through
   ``agrijax.port.run_fortran.run_rzwqm``, cached in ``<data>/validation/catpa_2015_ref/``
   (``CA-TPA.ana``, ``OVERVIEW.OUT``, ``LAYER.PLT``); reused when present.
2. PET: the daily Shuttleworth-Wallace module (``processes.pet.shuttleworth_wallace``) and the
   ASCE reference ET (``processes.pet.asce_reference_et``, ``variant="rzwqm"``) driven day by day
   with the state of the reference run turned into the module's start-of-day inputs the way the
   reference model does it at its PET call (:class:`HarnessOptions`; measured, not assumed: every
   convention is ablated in the attribution table). Compared with ``.ana`` cols 8 (PE), 9 (PT),
   83 (PET = PE + PT), 81 (tall reference ET), 82 (short reference ET). Three weather sources:
   the prepared ``.MET`` forcing (primary; 100 km/d wind floor of ``INPDAY``), the weather echoed
   in ``.ana`` cols 85-90 (module-only check: identical forcing) and the raw ``.MET`` file.
3. Hydraulics sanity check: the modified Brooks-Corey curves (``processes.soil_water.hydraulics``)
   with the CA-TPA horizon parameters from ``rzwqm.dat`` (``io.rzwqm.dat`` hydraulics block),
   mapped onto the 37 numerical nodes; profile storage sum(theta * dz) for (a) uniform-potential
   profiles (saturation, -100 cm, -333 cm, -15000 cm, residual) as plausibility bounds and (b) the
   theta(h) profile computed from the daily pressure heads of ``LAYER.PLT``, compared with ``.ana``
   col 2 (stored soil water, cm).

Writes the Markdown report to ``<data>/validation/catpa_2015_coarse.md`` and prints it.

Usage::

    uv run --project /home/yushentang/Agri_JAX python poc/coarse_compare.py [--rerun] [--out FILE]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from agrijax.io.rzwqm import read_ana, read_met, read_overview_yields, read_rzwqm_dat  # noqa: E402
from agrijax.io.rzwqm.dat import RzwqmDat  # noqa: E402
from agrijax.io.rzwqm.layers import layer_thickness_cm, read_layer_output  # noqa: E402
from agrijax.port.compare import CompareReport, compare_series  # noqa: E402
from agrijax.port.run_fortran import run_rzwqm  # noqa: E402
from agrijax.processes.pet import PETParams, SWResult, asce_reference_et, shuttleworth_wallace  # noqa: E402
from agrijax.processes.pet.shuttleworth_wallace import (  # noqa: E402
    RESIDUE_DENSITY_G_CM3,
    RESIDUE_DIAMETER_CM,
)
from agrijax.processes.soil_water.hydraulics import (  # noqa: E402
    H_FC13,
    H_FC110,
    H_WP,
    SoilHydraulicParams,
    theta_of_h,
)

DATA = Path(os.environ.get("AGRI_JAX_DATA", "/home/yushentang/agri_jax_data"))
SCENARIO = DATA / "narval_mirror" / "RZWQM_sw_batch" / "CA-TPA" / "Scenario"
REF_DIR = DATA / "validation" / "catpa_2015_ref"
REPORT_PATH = DATA / "validation" / "catpa_2015_coarse.md"
YEAR = 2015
KEEP = ("*.ana", "OVERVIEW.OUT", "LAYER.PLT")

#: .ana column (1-based) of each reference variable used here, and the name used in the report.
PET_COLUMNS = {
    "pot_evap_mm": 8,
    "pot_transp_mm": 9,
    "pet_mm": 83,
    "ref_et_tall_mm": 81,
    "ref_et_short_mm": 82,
}
WEATHER_COLUMNS = {"tmin": 85, "tmax": 86, "srad": 88, "rh": 89, "wind_run": 90}
MET_NAMES = {"tmin": "tmin", "tmax": "tmax", "srad": "srad_mj", "rh": "rh", "wind_run": "wind_run_km"}
WEATHER_UNITS = {"tmin": "degC", "tmax": "degC", "srad": "MJ m-2 d-1", "rh": "%", "wind_run": "km d-1"}
#: residue-type constants chosen from the residue cover factor CRES at start-up
#: (Rzmain.for lines 5245-5247; the DATA statement initialises IPR to 1 = corn otherwise)
RESIDUE_TYPE_OF_CRES = {2.0: "corn", 2.5: "soybean", 4.0: "wheat"}
#: residue type of the harvested crop, from the plant name of the rzwqm.dat plant record
CROP_RESIDUE_TYPE = (("maize", "corn"), ("corn", "corn"), ("soy", "soybean"), ("wheat", "wheat"))


@dataclass(frozen=True)
class HarnessOptions:
    """How the reference run's state becomes the PET module's start-of-day inputs.

    The defaults are the conventions of the reference model's daily loop (measured on every
    ``.ana`` row of CA-TPA 2015, see the attribution table of the report): management (tillage)
    runs before the PET call, crop growth, harvest and the residue-type switch run after it.
    """

    weather: str = "met"  # "met" = prepared .MET (wind floor), "met-raw" = raw .MET, "ana" = echoed
    canopy_lag: int = 1  # LAI / height from the previous .ana row (start of day); 0 = same row
    harvest_canopy_zero: bool = True  # the day after harvest starts without a canopy
    jan1_residue_from_file: bool = True  # first day: initial flat residue of rzwqm.dat, not row 0
    residue_type_switch: bool = True  # harvested crop's residue constants after harvest
    residue_age_reset: bool = True  # residue age restarts at harvest (HARVST sets RESAGE = 0)
    tillage_same_row: bool = True  # tillage day: post-tillage residue mass (same row)


REFERENCE_TIMING = HarnessOptions()
#: PET runs: key -> (options, report title)
PET_RUNS: dict[str, tuple[HarnessOptions, str]] = {
    "met": (REFERENCE_TIMING, "PET: prepared .MET weather, reference-model input timing (primary)"),
    "ana": (
        replace(REFERENCE_TIMING, weather="ana"),
        "PET: .ana-echoed weather, reference-model input timing (module-only check)",
    ),
    "met-raw": (
        replace(REFERENCE_TIMING, weather="met-raw"),
        "PET: raw .MET weather without the 100 km/d wind floor (diagnostic)",
    ),
}
#: ablations of the primary run: label -> (options, variable it hits, named cause)
ABLATIONS: dict[str, tuple[HarnessOptions, str, str]] = {
    "same-row LAI and height": (
        replace(REFERENCE_TIMING, canopy_lag=0),
        "pot_transp_mm",
        "crop growth follows the PET call in the reference day loop: PET sees the start-of-day canopy",
    ),
    "canopy kept the day after harvest": (
        replace(REFERENCE_TIMING, harvest_canopy_zero=False),
        "pot_transp_mm",
        "harvest removes the canopy after the .ana LAI gate of the harvest day; the next PET call sees "
        "LAI = 0",
    ),
    "Jan 1 residue from .ana row 0": (
        replace(REFERENCE_TIMING, jan1_residue_from_file=False),
        "pot_evap_mm",
        ".ana row 0 is all zeros, not the initial state; the first day uses the rzwqm.dat initial residue",
    ),
    "residue type kept after harvest": (
        replace(REFERENCE_TIMING, residue_type_switch=False),
        "pot_evap_mm",
        "IPR switches from the CRES-derived type (2.5 -> soybean) to the harvested crop's (maize -> corn)",
    ),
    "residue age not reset at harvest": (
        replace(REFERENCE_TIMING, residue_age_reset=False),
        "pot_evap_mm",
        "HARVST sets RESAGE = 0 (Rzman.for line 6615); the residue albedo ages from the harvest",
    ),
    "tillage-day residue from the previous row": (
        replace(REFERENCE_TIMING, tillage_same_row=False),
        "pot_evap_mm",
        "management (tillage, residue incorporation) precedes the PET call within the day",
    ),
    "raw .MET wind (no 100 km/d floor)": (
        replace(REFERENCE_TIMING, weather="met-raw"),
        "pot_transp_mm",
        "INPDAY floors the wind run at UBREEZ = 100 km/d (Rzmain.for line 3543)",
    ),
}


# --------------------------------------------------------------------------------------- inputs
def reference_run(ref_dir: Path = REF_DIR, *, rerun: bool = False, scenario: Path = SCENARIO) -> Path:
    """Directory holding ``CA-TPA.ana`` + ``LAYER.PLT`` of a 2015 RZWQM2 run (run once, then cached)."""
    ana = ref_dir / "CA-TPA.ana"
    if not rerun and ana.is_file() and (ref_dir / "LAYER.PLT").is_file():
        return ref_dir
    run_rzwqm(
        scenario,
        ref_dir,
        start=_dt.date(YEAR, 1, 1),
        end=_dt.date(YEAR, 12, 31),
        keep_files=KEEP,
    )
    return ref_dir


@dataclass
class Inputs:
    ana: xr.Dataset  # full .ana (row 0 = YYYY.000 initial state)
    col: dict[int, str]  # 1-based column -> variable name
    met: pd.DataFrame  # prepared MET rows of YEAR (INPDAY wind floor and bounds)
    met_raw: pd.DataFrame  # MET rows of YEAR as written in the file
    layers: xr.Dataset  # LAYER.PLT, one row per day of YEAR
    dat: RzwqmDat
    ref_dir: Path
    yields: pd.DataFrame | None

    def column(self, c: int) -> xr.DataArray:
        return self.ana[self.col[c]]

    @property
    def days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.met.index)


def load_inputs(ref_dir: Path, scenario: Path = SCENARIO) -> Inputs:
    ana = read_ana(ref_dir / "CA-TPA.ana")
    col = {int(k): v for k, v in ana.attrs["columns"].items()}
    span = slice(f"{YEAR}-01-01", f"{YEAR}-12-31")
    met_raw = read_met(scenario / "CA-TPA.MET").loc[span]
    met = read_met(scenario / "CA-TPA.MET", prepare=True).loc[span]
    layers = read_layer_output(ref_dir / "LAYER.PLT", start=f"{YEAR}-01-01")
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    ov = ref_dir / "OVERVIEW.OUT"
    yields = read_overview_yields(ov) if ov.is_file() else None
    return Inputs(ana, col, met, met_raw, layers, dat, ref_dir, yields)


def growing_season(inp: Inputs) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Planting and harvest dates of the YEAR planting in rzwqm.dat."""
    for p in inp.dat.plantings:
        if p.planting_date is not None and pd.Timestamp(p.planting_date).year == YEAR:
            return pd.Timestamp(p.planting_date), pd.Timestamp(p.harvest_date)
    raise LookupError(f"no {YEAR} planting in rzwqm.dat")


def _data_lines_after(dat: RzwqmDat, marker: str) -> list[list[str]]:
    """Data records (not starting with '=') that follow the header line containing ``marker``."""
    lines = [ln.rstrip("\r\n") for ln in dat.lines]
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


def residue_conditions(dat: RzwqmDat) -> dict[str, float]:
    """The 'Current Residue Conditions' record: initial flat residue, its age, CRES, ... (9 items)."""
    vals = [float(t) for t in _data_lines_after(dat, "C:P ratio of dominate residue material")[0]]
    keys = (
        "mass_t_ha",
        "age_d",
        "cover_factor",
        "height_cm",
        "cn_ratio",
        "stem_area_index",
        "standing_height_cm",
        "standing_mass_t_ha",
        "cp_ratio",
    )
    return dict(zip(keys, vals, strict=False))


def residue_cover_factor(dat: RzwqmDat) -> float:
    """``CRES``, item 3 of the 'Current Residue Conditions' record (corn 2.0, soybean 2.5, wheat 4.0)."""
    return residue_conditions(dat)["cover_factor"]


def tillage_dates(dat: RzwqmDat, year: int = YEAR) -> list[pd.Timestamp]:
    """Dates of the tillage operations of ``year`` given as specified dates (timing code 5).

    Record layout: plant ref, timing code, dd mm yyyy (code 5) or offset, implement, depth,
    intensity, operation, P mixing. Operations timed relative to planting (codes 1-4) are not
    resolved and are skipped.
    """
    recs = _data_lines_after(dat, "number of tillage operations")
    n = int(float(recs[0][0]))
    out = []
    for r in recs[1 : 1 + n]:
        if int(float(r[1])) == 5:
            d, m, y = (int(float(t)) for t in r[2:5])
            if y == year:
                out.append(pd.Timestamp(year=y, month=m, day=d))
    return out


def crop_residue_type(dat: RzwqmDat) -> str:
    """Residue type ('corn' | 'soybean' | 'wheat') of the first plant record, from its name."""
    name = str(dat.plants[0]).lower() if dat.plants else ""
    for key, rtype in CROP_RESIDUE_TYPE:
        if key in name:
            return rtype
    return "corn"


def pet_params(dat: RzwqmDat) -> PETParams:
    pet = dat.pet
    return PETParams(
        albedo_dry=jnp.asarray(pet["albedo_dry"]),
        albedo_wet=jnp.asarray(pet["albedo_wet"]),
        albedo_maturity=jnp.asarray(pet["albedo_crop"]),
        albedo_residue=jnp.asarray(pet["albedo_residue"]),
        soil_resistance=jnp.asarray(pet["soil_resistance"]),
        stomatal_resistance=jnp.asarray(dat.plant_site_params[0]["rs_min"]),
    )


# ----------------------------------------------------------------------------------------- PET
def _weather(inp: Inputs, source: str) -> dict[str, np.ndarray]:
    if source == "met":
        return {k: inp.met[v].to_numpy(dtype=float) for k, v in MET_NAMES.items()}
    if source == "met-raw":
        return {k: inp.met_raw[v].to_numpy(dtype=float) for k, v in MET_NAMES.items()}
    if source == "ana":
        return {
            k: inp.column(c).sel(time=inp.days).to_numpy().astype(float) for k, c in WEATHER_COLUMNS.items()
        }
    raise ValueError(source)


def weather_dataset(inp: Inputs, source: str) -> xr.Dataset:
    w = _weather(inp, source)
    return xr.Dataset(
        {k: ("time", v, {"units": WEATHER_UNITS[k]}) for k, v in w.items()},
        coords={"time": inp.days.values},
    )


@dataclass
class PETInputs:
    """Start-of-day inputs of the S-W module for every day of YEAR (numpy, time axis first)."""

    days: pd.DatetimeIndex
    weather: dict[str, np.ndarray]  # tmin, tmax, srad, rh, wind_run
    lai: np.ndarray
    height: np.ndarray
    theta: np.ndarray
    residue_mass: np.ndarray
    residue_age: np.ndarray
    residue_diameter_cm: np.ndarray
    residue_density: np.ndarray
    doy: np.ndarray
    params: PETParams
    site: dict[str, float]  # wc13, wc15, elevation, latitude, wind_height
    rainfall_zone: int
    cover_factor: float
    residue_type: np.ndarray  # object array of 'corn' | 'soybean' | 'wheat'


def pet_inputs(inp: Inputs, opt: HarnessOptions = REFERENCE_TIMING) -> PETInputs:
    """Turn the reference run's outputs into the module's inputs following ``opt``."""
    dat = inp.dat
    days = inp.days
    n = len(days)
    w = _weather(inp, opt.weather)
    _, harvest = growing_season(inp)
    canopy_days = days - pd.Timedelta(days=opt.canopy_lag)
    lai = inp.column(43).sel(time=canopy_days).to_numpy().astype(float)
    height = inp.column(62).sel(time=canopy_days).to_numpy().astype(float)
    if opt.harvest_canopy_zero:
        gone = days == harvest + pd.Timedelta(days=1)
        lai[gone] = 0.0
        height[gone] = 0.0
    # start-of-day residue mass = previous .ana row (row 0 is the YYYY.000 initial-state row, all zeros)
    prev_days = days - pd.Timedelta(days=1)
    residue = inp.column(72).sel(time=prev_days).to_numpy().astype(float)
    rc = residue_conditions(dat)
    if opt.jan1_residue_from_file:
        residue[0] = rc["mass_t_ha"] * 1.0e3
    if opt.tillage_same_row:
        for t in tillage_dates(dat):
            if t in days:
                residue[days.get_loc(t)] = float(inp.column(72).sel(time=t))
    # surface-node water content at the start of the day = end state of the previous day;
    # LAYER.PLT has no initial row, so Jan 1 uses its own end state (one day affected)
    theta_end = inp.layers["soil_water_content"].isel(depth=0).sel(time=days).to_numpy().astype(float)
    theta = np.concatenate([theta_end[:1], theta_end[:-1]])
    doy = days.dayofyear.to_numpy()
    # residue age: file value + 1 on the first day (Rzday.for line 1284 increments before the
    # PET call), restarting at harvest (HARVST, Rzman.for line 6615)
    age = rc["age_d"] + np.arange(1, n + 1, dtype=float)
    after = days > harvest
    if opt.residue_age_reset:
        age[after] = (days[after] - harvest).days.to_numpy().astype(float)
    cres = rc["cover_factor"]
    rtype = np.array([RESIDUE_TYPE_OF_CRES.get(cres, "corn")] * n, dtype=object)
    if opt.residue_type_switch:
        rtype[after] = crop_residue_type(dat)
    rdia = np.array([RESIDUE_DIAMETER_CM[t] for t in rtype], dtype=float)
    rho = np.array([RESIDUE_DENSITY_G_CM3[t] for t in rtype], dtype=float)
    hyd, phys, pet = dat.hydraulics, dat.physiography, dat.pet
    site = dict(
        wc13=float(hyd["theta_fc33"][0]),
        wc15=float(hyd["theta_wp"][0]),
        elevation=float(phys["elevation_m"]),
        latitude=float(phys["latitude_rad"]),
        wind_height=float(pet["wind_height_m"]),
    )
    return PETInputs(
        days=days,
        weather=w,
        lai=lai,
        height=height,
        theta=theta,
        residue_mass=residue,
        residue_age=age,
        residue_diameter_cm=rdia,
        residue_density=rho,
        doy=doy,
        params=pet_params(dat),
        site=site,
        rainfall_zone=int(phys["rainfall_zone"]),
        cover_factor=float(cres),
        residue_type=rtype,
    )


def sw_day(
    x: PETInputs,
    tmin,
    tmax,
    srad,
    rh,
    wind,
    lai,
    height,
    theta,
    residue_mass,
    residue_age,
    rdia,
    rho,
    doy,
    params: PETParams | None = None,
) -> SWResult:
    """The S-W kernel for one day with the site constants of ``x`` (differentiable in every array)."""
    return shuttleworth_wallace(
        tmin,
        tmax,
        srad,
        rh,
        wind,
        lai,
        height,
        x.params if params is None else params,
        theta_surface=theta,
        wc13=x.site["wc13"],
        wc15=x.site["wc15"],
        elevation=x.site["elevation"],
        latitude=x.site["latitude"],
        doy=doy,
        residue_mass=residue_mass,
        residue_age=residue_age,
        wind_height=x.site["wind_height"],
        rainfall_zone=x.rainfall_zone,
        residue_cover_factor=x.cover_factor,
        residue_diameter_cm=rdia,
        residue_density=rho,
    )


def sw_result(x: PETInputs) -> SWResult:
    """The S-W kernel over every day of ``x`` (jit + vmap)."""
    w = x.weather

    def one(*args):
        return sw_day(x, *args)

    return jax.jit(jax.vmap(one))(
        w["tmin"],
        w["tmax"],
        w["srad"],
        w["rh"],
        w["wind_run"],
        x.lai,
        x.height,
        x.theta,
        x.residue_mass,
        x.residue_age,
        x.residue_diameter_cm,
        x.residue_density,
        x.doy,
    )


def pet_simulation(inp: Inputs, opt: HarnessOptions = REFERENCE_TIMING) -> xr.Dataset:
    """Daily S-W PE, PT, PE + PT and ASCE tall/short reference ET [mm d-1] for every day of YEAR."""
    x = pet_inputs(inp, opt)
    r = sw_result(x)
    w = x.weather
    ref = asce_reference_et(
        w["tmin"],
        w["tmax"],
        w["srad"],
        w["rh"],
        w["wind_run"] * 1.0e3 / 86400.0,
        elevation=x.site["elevation"],
        latitude=x.site["latitude"],
        doy=x.doy,
        wind_height=x.site["wind_height"],
        variant="rzwqm",
    )
    pt = np.asarray(r.transpiration) * 10.0
    pe = np.asarray(r.soil_evaporation + r.residue_evaporation) * 10.0
    out = {
        "pot_evap_mm": pe,
        "pot_transp_mm": pt,
        "pet_mm": pe + pt,
        "ref_et_tall_mm": np.asarray(ref.et_tall),
        "ref_et_short_mm": np.asarray(ref.et_short),
    }
    return xr.Dataset(
        {k: ("time", v, {"units": "mm d-1"}) for k, v in out.items()},
        coords={"time": x.days.values},
        attrs={"weather": opt.weather},
    )


def pet_reference(inp: Inputs) -> xr.Dataset:
    """The .ana PET columns of every day of YEAR, converted from cm to mm d-1."""
    days = inp.days
    return xr.Dataset(
        {
            k: ("time", inp.column(c).sel(time=days).to_numpy().astype(float) * 10.0, {"units": "mm d-1"})
            for k, c in PET_COLUMNS.items()
        },
        coords={"time": days.values},
    )


# ---------------------------------------------------------------------------------- hydraulics
def hydraulic_params(dat: RzwqmDat) -> tuple[SoilHydraulicParams, np.ndarray, np.ndarray]:
    """Node-mapped hydraulic parameters, node depths and layer thicknesses [cm]."""
    nodes = np.asarray(dat.node_depths_cm, dtype=float)
    bottoms = np.asarray(dat.horizon_depths_cm, dtype=float)
    node_horizon = np.minimum(np.searchsorted(bottoms, nodes, side="left"), len(bottoms) - 1)
    p = SoilHydraulicParams.from_rzwqm_dat(dat.hydraulics, node_horizon=node_horizon)
    return p.at_nodes(), nodes, layer_thickness_cm(nodes)


def uniform_profile_storage(dat: RzwqmDat) -> dict[str, float]:
    """Storage [cm] of profiles at one uniform matric potential (plausibility bounds)."""
    p, nodes, dz = hydraulic_params(dat)
    out: dict[str, float] = {}
    for label, h in (
        ("saturation (h = 0)", 0.0),
        ("h = -100 cm (1/10 bar)", H_FC110),
        ("field capacity h = -333 cm", H_FC13),
        ("wilting point h = -15000 cm", H_WP),
    ):
        th = theta_of_h(jnp.full(nodes.shape, h), p)
        out[label] = float(jnp.sum(th * dz))
    out["residual (theta_r)"] = float(jnp.sum(p.theta_r * dz))
    return out


def storage_from_heads(inp: Inputs) -> xr.Dataset:
    """Daily profile storage [cm] from theta_of_h(LAYER.PLT pressure head), and from LAYER.PLT theta."""
    p, nodes, dz = hydraulic_params(inp.dat)
    lay = inp.layers.sel(time=inp.days)
    depth = lay["depth"].to_numpy()
    if not np.allclose(depth, nodes):
        raise ValueError(f"LAYER.PLT depths {depth} differ from rzwqm.dat nodes {nodes}")
    h = jnp.asarray(lay["pressure_head"].to_numpy())
    theta = jax.vmap(lambda hh: theta_of_h(hh, p))(h)
    theta_np = np.asarray(theta)
    theta_ref = lay["soil_water_content"].to_numpy()
    storage = (theta_np * dz).sum(axis=1)
    return xr.Dataset(
        {
            "storage_cm": ("time", storage, {"units": "cm"}),
            "storage_layer_theta_cm": ("time", (theta_ref * dz).sum(axis=1), {"units": "cm"}),
            "theta_err": (("time", "depth"), theta_np - theta_ref, {"units": "cm3 cm-3"}),
        },
        coords={"time": inp.days.values, "depth": depth},
    )


def storage_reference(inp: Inputs) -> xr.Dataset:
    return xr.Dataset(
        {"storage_cm": ("time", inp.column(2).sel(time=inp.days).to_numpy().astype(float), {"units": "cm"})},
        coords={"time": inp.days.values},
    )


# -------------------------------------------------------------------------------------- report
@dataclass
class Ablation:
    label: str
    variable: str
    cause: str
    rmse: float  # year RMSE of the ablated run [mm d-1]
    max_abs: float  # largest daily |error| of the ablated run
    worst_day: pd.Timestamp
    primary_on_worst_day: float  # |error| of the primary run on that day
    errors: pd.Series  # daily |sim - ref| of the ablated run


@dataclass
class CoarseResult:
    pet: dict[str, CompareReport]  # run key -> report (year + season rows)
    pet_sim: dict[str, xr.Dataset]  # run key -> simulated PET series
    pet_ref: xr.Dataset
    ablations: dict[str, Ablation]
    weather: CompareReport
    storage: CompareReport
    uniform: dict[str, float]
    storage_range: tuple[float, float]
    theta_err_by_depth: pd.DataFrame
    season: tuple[pd.Timestamp, pd.Timestamp]
    facts: dict[str, object]  # measured numbers quoted in the report and asserted in the tests
    markdown: str


def _abs_err(sim: xr.Dataset, ref: xr.Dataset, var: str) -> pd.Series:
    return (sim[var] - ref[var]).to_pandas().abs()


def ablation_table(inp: Inputs, ref: xr.Dataset, primary: xr.Dataset) -> dict[str, Ablation]:
    out: dict[str, Ablation] = {}
    for label, (opt, var, cause) in ABLATIONS.items():
        sim = pet_simulation(inp, opt)
        e = _abs_err(sim, ref, var)
        worst = e.idxmax()
        out[label] = Ablation(
            label=label,
            variable=var,
            cause=cause,
            rmse=float(np.sqrt(np.mean(e.to_numpy() ** 2))),
            max_abs=float(e.max()),
            worst_day=pd.Timestamp(worst),
            primary_on_worst_day=float(_abs_err(primary, ref, var)[worst]),
            errors=e,
        )
    return out


def measured_facts(inp: Inputs, res_pet: dict[str, xr.Dataset], ref: xr.Dataset, season) -> dict[str, object]:
    days = inp.days
    start, end = season
    raw_wind = inp.met_raw["wind_run_km"].to_numpy(dtype=float)
    ana_wind = inp.column(90).sel(time=days).to_numpy().astype(float)
    floor_days = list(days[raw_wind < 100.0])
    ratio = inp.column(88).sel(time=days).to_numpy().astype(float) / inp.met_raw["srad_mj"].to_numpy(
        dtype=float
    )
    met, ana = res_pet["met"], res_pet["ana"]
    srad_effect = {v: float(np.abs(met[v] - ana[v]).max()) for v in ("pot_evap_mm", "pot_transp_mm")}
    e_pe = _abs_err(met, ref, "pot_evap_mm")
    e_pt = _abs_err(met, ref, "pot_transp_mm")
    after = (days > end) & (days <= end + pd.Timedelta(days=60))
    pre = (days < start) & (days.month >= 4)
    theta_err = np.abs(storage_from_heads(inp)["theta_err"])
    err = theta_err.max("depth").to_pandas()
    bad_days = err[err > 1e-3]
    bad_depths = theta_err["depth"].to_numpy()[(theta_err > 1e-3).any("time").to_numpy()]
    return {
        "wind_floor_days": floor_days,
        "wind_floor_raw": [float(v) for v in raw_wind[raw_wind < 100.0]],
        "wind_prepared_equals_ana": bool(
            np.array_equal(inp.met["wind_run_km"].to_numpy(dtype=float), ana_wind)
        ),
        "srad_ratio_range": (float(ratio.min()), float(ratio.max())),
        "srad_effect_on_pet": srad_effect,
        "pe_err_after_harvest": float(e_pe[after].mean()),
        "pe_err_before_planting": float(e_pe[pre].mean()),
        "worst_pe": [
            (pd.Timestamp(d), float(v)) for d, v in e_pe.sort_values(ascending=False).head(3).items()
        ],
        "worst_pt": [
            (pd.Timestamp(d), float(v)) for d, v in e_pt.sort_values(ascending=False).head(3).items()
        ],
        "tillage_dates": tillage_dates(inp.dat),
        "residue_conditions": residue_conditions(inp.dat),
        "residue_type_before": RESIDUE_TYPE_OF_CRES.get(residue_cover_factor(inp.dat), "corn"),
        "residue_type_after": crop_residue_type(inp.dat),
        "theta_bad_days": [pd.Timestamp(d) for d in bad_days.index],
        "theta_bad_depths": (float(bad_depths.min()), float(bad_depths.max())) if bad_depths.size else None,
    }


def run_comparison(ref_dir: Path, scenario: Path = SCENARIO) -> CoarseResult:
    inp = load_inputs(ref_dir, scenario)
    start, end = growing_season(inp)
    season = slice(start, end)
    ref = pet_reference(inp)
    pet_reports: dict[str, CompareReport] = {}
    pet_sims: dict[str, xr.Dataset] = {}
    for key, (opt, title) in PET_RUNS.items():
        sim = pet_simulation(inp, opt)
        pet_sims[key] = sim
        vars_ = list(PET_COLUMNS)
        rep = compare_series(sim, ref, vars_, period=f"year ({key})")
        rep += compare_series(sim, ref, vars_, period=f"season ({key})", where=season)
        rep.title = title
        pet_reports[key] = rep
    ablations = ablation_table(inp, ref, pet_sims["met"])
    weather = compare_series(
        weather_dataset(inp, "met-raw"),
        weather_dataset(inp, "ana"),
        list(WEATHER_COLUMNS),
        period="year",
        title="Raw .MET forcing vs weather echoed in .ana (cols 85-90)",
    )
    st_sim = storage_from_heads(inp)
    st_ref = storage_reference(inp)
    storage = compare_series(
        st_sim[["storage_cm"]], st_ref, ["storage_cm"], period="year", title="Profile water storage"
    )
    storage += compare_series(
        st_sim[["storage_layer_theta_cm"]].rename(storage_layer_theta_cm="storage_cm"),
        st_ref,
        ["storage_cm"],
        period="year (LAYER.PLT theta, io check)",
    )
    uniform = uniform_profile_storage(inp.dat)
    col2 = st_ref["storage_cm"].to_numpy()
    err = st_sim["theta_err"]
    by_depth = pd.DataFrame(
        {
            "depth_cm": err["depth"].to_numpy(),
            "max_abs_theta_err": np.abs(err).max("time").to_numpy(),
            "days_abs_err_gt_1e-3": (np.abs(err) > 1e-3).sum("time").to_numpy(),
        }
    )
    res = CoarseResult(
        pet=pet_reports,
        pet_sim=pet_sims,
        pet_ref=ref,
        ablations=ablations,
        weather=weather,
        storage=storage,
        uniform=uniform,
        storage_range=(float(col2.min()), float(col2.max())),
        theta_err_by_depth=by_depth,
        season=(start, end),
        facts=measured_facts(inp, pet_sims, ref, (start, end)),
        markdown="",
    )
    res.markdown = render(res, inp)
    return res


def render(res: CoarseResult, inp: Inputs) -> str:
    start, end = res.season
    today = _dt.date.today().isoformat()
    f = res.facts
    rc = f["residue_conditions"]
    lines = [
        f"# CA-TPA {YEAR}: coarse comparison against RZWQM2",
        "",
        f"Generated {today} by `poc/coarse_compare.py`. Reference run: `{inp.ref_dir}` "
        f"(RZWQM2 binary, IPNAMES period {YEAR}-01-01 to {YEAR}-12-31). Growing season = planting "
        f"{start.date()} to harvest {end.date()} from `rzwqm.dat`. Nothing was tuned: all parameters "
        "are the scenario values. Units mm d-1 unless stated. R² is the squared Pearson "
        "correlation; NSE = 1 - SSE/SST; bias = mean(sim - ref).",
        "",
    ]
    if inp.yields is not None and len(inp.yields):
        y = inp.yields.iloc[-1]
        lines += [f"Reference-run maize yield (OVERVIEW.OUT): {y['yield_kg_ha']:.0f} kg/ha.", ""]
    lines += [
        "Inputs of the PET module (start-of-day state, the reference model's input timing, see the "
        "attribution table): LAI (col 43) and canopy height (col 62) of the previous `.ana` row, "
        f"zero on the day after harvest; flat residue mass (col 72) of the previous row, the rzwqm.dat "
        f"initial residue ({rc['mass_t_ha']:g} t/ha) on the first day and the post-tillage mass (same "
        f"row) on tillage days ({', '.join(str(t.date()) for t in f['tillage_dates'])}); surface-node "
        f"theta from `LAYER.PLT` of the previous day; residue age = file age ({rc['age_d']:g} d) + day "
        f"count, restarting at harvest; residue-type constants `{f['residue_type_before']}` (from CRES = "
        f"{rc['cover_factor']:g}) before harvest and `{f['residue_type_after']}` (the crop's) after it. "
        "PET params from the `rzwqm.dat` PET block, rs_min of the plant block, rainfall zone and CRES "
        "from the file. Weather: prepared `.MET` (100 km/d wind floor of INPDAY; primary), the `.ana` "
        "echo (identical forcing, module-only check) or the raw `.MET`.",
        "",
    ]
    for key in PET_RUNS:
        lines.append(res.pet[key].to_markdown(heading_level=2))
    lines += error_distribution_table(res)
    lines += attribution_table(res)
    lines.append(res.weather.to_markdown(heading_level=2))
    lines.append(res.storage.to_markdown(heading_level=2))
    lines += [
        "Storage for uniform-potential profiles (hydraulics module, CA-TPA horizons on the 37 nodes, "
        "sum theta dz over 0-150 cm):",
        "",
        "| profile | storage (cm) |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v:.3f} |" for k, v in res.uniform.items()]
    lo, hi = res.storage_range
    lines += [
        f"| `.ana` col 2, {YEAR} min / max | {lo:.3f} / {hi:.3f} |",
        "",
        "theta_of_h(LAYER.PLT pressure head) - LAYER.PLT theta, per node (nodes with any |err| > 1e-4):",
        "",
        "| node depth (cm) | max abs err | days with abs err > 1e-3 |",
        "|---:|---:|---:|",
    ]
    bad = res.theta_err_by_depth[res.theta_err_by_depth["max_abs_theta_err"] > 1e-4]
    for _, r in bad.iterrows():
        lines.append(
            f"| {r['depth_cm']:.0f} | {r['max_abs_theta_err']:.2e} | {int(r['days_abs_err_gt_1e-3'])} |"
        )
    if bad.empty:
        lines.append("| (none) | | |")
    worst = float(res.theta_err_by_depth["max_abs_theta_err"].max())
    lines += ["", f"Largest per-node |theta error| over all nodes and days: {worst:.2e}.", ""]
    lines += diagnostics(res, inp)
    lines += CONCLUSIONS
    return "\n".join(lines) + "\n"


#: Hand-written reading of the 2026-09-23 run (the numbers above are regenerated).
CONCLUSIONS = [
    "## Conclusions (hand-written, 2026-09-23 run)",
    "",
    "1. The daily Shuttleworth-Wallace branch (`POTEVPHR`, `ipet = 0`, `ihourly = 0`) is reproduced "
    "to the print precision of the `.ana` file once the module is fed what the reference model "
    "feeds it: with the echoed weather the whole-year PT error is below 1e-3 mm/d on every day and "
    "the PE error below 5e-3 mm/d (the residual is the residue mass before the day's decomposition, "
    "which col 72 reports after it). No equation, constant or branch of the module was changed to "
    "get there; every former outlier was an input-timing convention of the reference day loop, "
    "listed with its measured effect in the attribution table.",
    "2. Input timing. Management (tillage, residue incorporation) runs before the PET call; crop "
    "growth, harvest (canopy removal, harvest residue, residue age reset) and the switch of the "
    "residue-type constants to the harvested crop's run after it. In a coupled `Model` the process "
    "order must therefore be: management, PET, crop growth, harvest. The same-row convention is wrong "
    "by up to 1.7 mm/d on days of fast canopy change and by 2.6 mm/d the day after harvest.",
    "3. Forcing preparation belongs to the io layer, not the PET process: the 100 km/d wind floor "
    "(`prepare_rzwqm_forcing`) is the whole raw-`.MET` gap in PT (0.75 mm/d on 2015-07-28). The "
    "solar radiation the reference uses is not the `.MET` value but the daily sum of its hourly "
    "disaggregation (SOLAR_SHAW, Rzmain.for lines 1384-1389), a smooth seasonal factor of "
    "0.993-1.005; its effect on PE and PT is below 0.01 mm/d and is not reproduced (known "
    "deviation, encoded with that tolerance).",
    "4. Rejected hypotheses (tested one at a time, each made the agreement worse or changed nothing): "
    "the post-harvest stubble height that HARVST stores in HEIGHT (30 cm) is not what the PET call "
    "sees (height 0 reproduces the reference); the snow long-wave branch never runs in the daily "
    "path (its switch NSP is only set by SHAW; Jan 1 has no snow anyway); residue wetness never "
    "applies (WRES is reset to 0 every day before the call); the night rule's threshold is exactly "
    "the reference's (hourly W m-2 test on the daily total, never true); the rainfall-zone (a, b) "
    "constants of the file (zone 3) are right; the surface theta of the previous day is right "
    "(Jan 1 is the only day without it and is now within 2e-3 mm/d).",
    "5. Hydraulics. The Brooks-Corey curves with the rzwqm.dat horizon parameters reproduce the "
    "reference theta from its own pressure heads on all 37 nodes to 1e-4, except in horizon 1 "
    "(0-15 cm) for about a week after the 2015-04-14 tillage (15 cm deep). This is consistent with "
    "the tillage-modified curve (TRHYDP) that the module deliberately does not implement. Profile "
    "storage from theta(h) matches col 2 to a few hundredths of a cm, and the 2015 range of col 2 "
    "lies between the uniform wilting-point and saturated storages, as it should.",
    "",
]


def error_distribution_table(res: CoarseResult) -> list[str]:
    """Median / 95th percentile of the daily |sim - ref| and the worst days, per run."""
    out = [
        "## Distribution of daily absolute errors (whole year)",
        "",
        "| run | variable | median abs | p95 abs | days abs > 0.01 mm | worst days (abs err, mm) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for src, sim in res.pet_sim.items():
        err = np.abs(sim - res.pet_ref).to_dataframe()
        for v in PET_COLUMNS:
            e = err[v]
            worst = e.sort_values(ascending=False).head(3)
            wtxt = ", ".join(f"{d.date()} ({x:.3f})" for d, x in worst.items())
            out.append(
                f"| {src} | `{v}` | {e.median():.2e} | {e.quantile(0.95):.2e} "
                f"| {int((e > 0.01).sum())} | {wtxt} |"
            )
    return [*out, ""]


def attribution_table(res: CoarseResult) -> list[str]:
    """Each input-timing convention removed from the primary run, with the error it brings back."""
    out = [
        "## Attribution: each convention of the primary run removed one at a time",
        "",
        "| convention removed | variable | year RMSE | max abs | worst day (primary run there) | cause |",
        "|---|---|---:|---:|---|---|",
    ]
    for a in res.ablations.values():
        out.append(
            f"| {a.label} | `{a.variable}` | {a.rmse:.3f} | {a.max_abs:.3f} | "
            f"{a.worst_day.date()} ({a.primary_on_worst_day:.1e}) | {a.cause} |"
        )
    se = res.facts["srad_effect_on_pet"]
    lo, hi = res.facts["srad_ratio_range"]
    out.append(
        f"| `.MET` srad instead of the re-summed hourly srad (`.ana` col 88) | `pot_evap_mm` / "
        f"`pot_transp_mm` | - | {se['pot_evap_mm']:.3f} / {se['pot_transp_mm']:.3f} | - | "
        f"Rzmain.for lines 1384-1389 re-sum the SOLAR_SHAW hourly disaggregation; ratio .ana/.MET "
        f"{lo:.4f}-{hi:.4f} over the year; not reproduced (known deviation) |"
    )
    return [*out, ""]


def diagnostics(res: CoarseResult, inp: Inputs) -> list[str]:
    """Measured facts behind the conclusions (asserted in tests/integration/test_catpa_coarse.py)."""
    f = res.facts
    _, end = res.season
    floor_txt = ", ".join(
        f"{d.date()} ({v:.1f})" for d, v in zip(f["wind_floor_days"], f["wind_floor_raw"], strict=False)
    )
    lo, hi = f["srad_ratio_range"]
    se = f["srad_effect_on_pet"]
    bd = f["theta_bad_depths"]
    node_txt = f"nodes {bd[0]:g}-{bd[1]:g} cm" if bd else "no node"
    top_bottom = float(np.asarray(inp.dat.horizon_depths_cm)[0])
    worst_pe = "; ".join(f"{d.date()} ({v:.1e})" for d, v in f["worst_pe"])
    worst_pt = "; ".join(f"{d.date()} ({v:.1e})" for d, v in f["worst_pt"])
    return [
        "## Measured facts behind the conclusions",
        "",
        f"- Wind: {len(f['wind_floor_days'])} days where the raw `.MET` wind run is below 100 km/d "
        f"(raw values in km/d): {floor_txt or 'none'}. The prepared `.MET` wind equals `.ana` col 90 on "
        f"every day: {f['wind_prepared_equals_ana']}.",
        f"- Solar radiation: `.ana` col 88 / raw `.MET` srad ranges {lo:.4f}-{hi:.4f} over the year "
        f"(smooth seasonal factor). Largest effect on the primary run: PE {se['pot_evap_mm']:.1e}, "
        f"PT {se['pot_transp_mm']:.1e} mm/d (primary vs `.ana`-weather run).",
        f"- Primary run, mean |PE error| in the 60 days after harvest ({end.date()}): "
        f"{f['pe_err_after_harvest']:.1e} mm/d, vs {f['pe_err_before_planting']:.1e} mm/d from April to "
        f"planting. Worst PE days: {worst_pe}. Worst PT days: {worst_pt}.",
        f"- Residue: initial {f['residue_conditions']['mass_t_ha']:g} t/ha, age "
        f"{f['residue_conditions']['age_d']:g} d, CRES {f['residue_conditions']['cover_factor']:g} -> "
        f"`{f['residue_type_before']}` constants before harvest; `{f['residue_type_after']}` after "
        "(measured on col 114: the post-harvest cover 0.920 for 11.94 t/ha is the corn diameter and "
        "density, the soybean ones would give 0.988). No standing residue (items 6-8 are 0).",
        f"- theta_of_h vs LAYER.PLT theta: {len(f['theta_bad_days'])} days with a node error > 1e-3 "
        f"({', '.join(str(d.date()) for d in f['theta_bad_days'])}); "
        f"{node_txt} (horizon 1 ends at {top_bottom:g} cm).",
        "",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref-dir", type=Path, default=REF_DIR)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--rerun", action="store_true", help="re-run RZWQM2 even when cached outputs exist")
    a = ap.parse_args(argv)
    ref_dir = reference_run(a.ref_dir, rerun=a.rerun)
    res = run_comparison(ref_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(res.markdown)
    print(res.markdown)
    print(f"written: {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
