"""First coarse comparison of Agri-JAX components against RZWQM2 on CA-TPA 2015.

What it does (no tuning; every parameter comes from the scenario files as they stand):

1. Reference run: a one-year (2015) CA-TPA run of the RZWQM2 binary through
   ``agri_jax.port.run_fortran.run_rzwqm``, cached in ``<data>/validation/catpa_2015_ref/``
   (``CA-TPA.ana``, ``OVERVIEW.OUT``, ``LAYER.PLT``); reused when present.
2. PET: the daily Shuttleworth-Wallace module (``processes.pet.shuttleworth_wallace``) and the
   ASCE reference ET (``processes.pet.asce_reference_et``, ``variant="rzwqm"``) driven day by day
   by the ``.MET`` forcing, with the canopy state (LAI col 43, height col 62), the flat residue
   mass (col 72, previous row) and the surface-node water content (``LAYER.PLT``, previous day)
   taken from the reference run. Compared with ``.ana`` cols 8 (PE), 9 (PT), 83 (PET = PE + PT),
   81 (tall reference ET), 82 (short reference ET). As a diagnostic the same is repeated with the
   weather echoed in the ``.ana`` file (cols 85, 86, 88, 89, 90) instead of the ``.MET`` file, and
   the two weather inputs are compared with each other.
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
from dataclasses import dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from agri_jax.io.rzwqm import read_ana, read_met, read_overview_yields, read_rzwqm_dat  # noqa: E402
from agri_jax.io.rzwqm.dat import RzwqmDat  # noqa: E402
from agri_jax.io.rzwqm.layers import layer_thickness_cm, read_layer_output  # noqa: E402
from agri_jax.port.compare import CompareReport, compare_series  # noqa: E402
from agri_jax.port.run_fortran import run_rzwqm  # noqa: E402
from agri_jax.processes.pet import PETParams, asce_reference_et, shuttleworth_wallace  # noqa: E402
from agri_jax.processes.soil_water.hydraulics import (  # noqa: E402
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
#: PET runs: key -> (weather source, canopy lag in days, report title). "met" is the run the task
#: specifies (MET weather, LAI/height of the same .ana row); the others are diagnostics.
PET_RUNS = {
    "met": ("met", 0, "PET: MET weather, same-row LAI/height (primary)"),
    "ana": ("ana", 0, "PET: .ana-echoed weather, same-row LAI/height (diagnostic)"),
    "met-lag1": ("met", 1, "PET: MET weather, previous-row (start-of-day) LAI/height (diagnostic)"),
}
WEATHER_UNITS = {"tmin": "degC", "tmax": "degC", "srad": "MJ m-2 d-1", "rh": "%", "wind_run": "km d-1"}


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
    met: pd.DataFrame  # MET rows of YEAR
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
    met = read_met(scenario / "CA-TPA.MET")
    met = met.loc[f"{YEAR}-01-01" : f"{YEAR}-12-31"]
    layers = read_layer_output(ref_dir / "LAYER.PLT", start=f"{YEAR}-01-01")
    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    ov = ref_dir / "OVERVIEW.OUT"
    yields = read_overview_yields(ov) if ov.is_file() else None
    return Inputs(ana, col, met, layers, dat, ref_dir, yields)


def growing_season(inp: Inputs) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Planting and harvest dates of the YEAR planting in rzwqm.dat."""
    for p in inp.dat.plantings:
        if p.planting_date is not None and pd.Timestamp(p.planting_date).year == YEAR:
            return pd.Timestamp(p.planting_date), pd.Timestamp(p.harvest_date)
    raise LookupError(f"no {YEAR} planting in rzwqm.dat")


def residue_cover_factor(dat: RzwqmDat) -> float:
    """``CRES``, item 3 of the 'Current Residue Conditions' record (corn 2.0, soybean 2.5, wheat 4.0)."""
    lines = [ln.rstrip("\r\n") for ln in dat.lines]
    for i, ln in enumerate(lines):
        if "C:P ratio of dominate residue material" in ln:
            for nxt in lines[i + 1 :]:
                s = nxt.strip()
                if s and not s.startswith("="):
                    return float(s.split()[2])
    raise LookupError("residue block not found")


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


def pet_simulation(inp: Inputs, weather: str = "met", *, canopy_lag: int = 0) -> xr.Dataset:
    """Daily S-W PE, PT, PE + PT and ASCE tall/short reference ET [mm d-1] for every day of YEAR.

    ``canopy_lag = 1`` takes LAI and height from the previous ``.ana`` row (diagnostic only).
    """
    dat = inp.dat
    days = inp.days
    w = _weather(inp, weather)
    canopy_days = days - pd.Timedelta(days=canopy_lag)
    lai = inp.column(43).sel(time=canopy_days).to_numpy().astype(float)
    height = inp.column(62).sel(time=canopy_days).to_numpy().astype(float)
    # start-of-day residue mass = previous .ana row (row 0 is the YYYY.000 initial state)
    prev_days = days - pd.Timedelta(days=1)
    residue = inp.column(72).sel(time=prev_days).to_numpy().astype(float)
    # surface-node water content at the start of the day = end state of the previous day;
    # LAYER.PLT has no initial row, so Jan 1 uses its own end state (one day affected)
    theta_end = inp.layers["soil_water_content"].isel(depth=0).sel(time=days).to_numpy().astype(float)
    theta = np.concatenate([theta_end[:1], theta_end[:-1]])
    doy = days.dayofyear.to_numpy()
    # residue albedo ageing: the residue age is not an output; the file's initial age (50 d) plus
    # the day count is used (weak effect: it only moves the residue albedo)
    age = 50.0 + np.arange(len(days), dtype=float)

    hyd = dat.hydraulics
    phys = dat.physiography
    pet = dat.pet
    cres = residue_cover_factor(dat)
    residue_type = {2.0: "corn", 2.5: "soybean", 4.0: "wheat"}.get(cres, "corn")
    params = pet_params(dat)
    site = dict(
        wc13=float(hyd["theta_fc33"][0]),
        wc15=float(hyd["theta_wp"][0]),
        elevation=float(phys["elevation_m"]),
        latitude=float(phys["latitude_rad"]),
        wind_height=float(pet["wind_height_m"]),
    )
    zone = int(phys["rainfall_zone"])

    def one(tmin, tmax, srad, rh, wind, l_, hc, th, m, dd, ag):
        return shuttleworth_wallace(
            tmin,
            tmax,
            srad,
            rh,
            wind,
            l_,
            hc,
            params,
            theta_surface=th,
            wc13=site["wc13"],
            wc15=site["wc15"],
            elevation=site["elevation"],
            latitude=site["latitude"],
            doy=dd,
            residue_mass=m,
            residue_age=ag,
            wind_height=site["wind_height"],
            rainfall_zone=zone,
            residue_type=residue_type,
            residue_cover_factor=cres,
        )

    r = jax.jit(jax.vmap(one))(
        w["tmin"], w["tmax"], w["srad"], w["rh"], w["wind_run"], lai, height, theta, residue, doy, age
    )
    ref = asce_reference_et(
        w["tmin"],
        w["tmax"],
        w["srad"],
        w["rh"],
        w["wind_run"] * 1.0e3 / 86400.0,
        elevation=site["elevation"],
        latitude=site["latitude"],
        doy=doy,
        wind_height=site["wind_height"],
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
        coords={"time": days.values},
        attrs={"weather": weather},
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
class CoarseResult:
    pet: dict[str, CompareReport]  # weather source -> report (year + season rows)
    pet_sim: dict[str, xr.Dataset]  # weather source -> simulated PET series
    pet_ref: xr.Dataset
    weather: CompareReport
    storage: CompareReport
    uniform: dict[str, float]
    storage_range: tuple[float, float]
    theta_err_by_depth: pd.DataFrame
    season: tuple[pd.Timestamp, pd.Timestamp]
    markdown: str


def run_comparison(ref_dir: Path, scenario: Path = SCENARIO) -> CoarseResult:
    inp = load_inputs(ref_dir, scenario)
    start, end = growing_season(inp)
    season = slice(start, end)
    ref = pet_reference(inp)
    pet_reports: dict[str, CompareReport] = {}
    pet_sims: dict[str, xr.Dataset] = {}
    for key, (src, lag, title) in PET_RUNS.items():
        sim = pet_simulation(inp, src, canopy_lag=lag)
        pet_sims[key] = sim
        vars_ = list(PET_COLUMNS)
        rep = compare_series(sim, ref, vars_, period=f"year ({key})")
        rep += compare_series(sim, ref, vars_, period=f"season ({key})", where=season)
        rep.title = title
        pet_reports[key] = rep
    weather = compare_series(
        weather_dataset(inp, "met"),
        weather_dataset(inp, "ana"),
        list(WEATHER_COLUMNS),
        period="year",
        title=".MET forcing vs weather echoed in .ana (cols 85-90)",
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
        weather=weather,
        storage=storage,
        uniform=uniform,
        storage_range=(float(col2.min()), float(col2.max())),
        theta_err_by_depth=by_depth,
        season=(start, end),
        markdown="",
    )
    res.markdown = render(res, inp)
    return res


def render(res: CoarseResult, inp: Inputs) -> str:
    start, end = res.season
    today = _dt.date.today().isoformat()
    lines = [
        f"# CA-TPA {YEAR}: first coarse comparison against RZWQM2",
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
        "Inputs of the PET modules: weather from the `.MET` file (primary) or from the `.ana` echo "
        "(diagnostic); LAI (col 43) and canopy height (col 62) of the same row, flat residue mass "
        "(col 72) of the previous row, surface-node theta from `LAYER.PLT` of the previous day, "
        "residue age = 50 d + day count (not an output). PET params from the `rzwqm.dat` PET block, "
        "rs_min of the plant block, rainfall zone and CRES from the file.",
        "",
    ]
    for key in PET_RUNS:
        lines.append(res.pet[key].to_markdown(heading_level=2))
    lines += error_distribution_table(res)
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
    lines += INTERPRETATION
    return "\n".join(lines) + "\n"


#: Hand-written reading of the 2026-09-23 run (qualitative; the numbers above are regenerated).
INTERPRETATION = [
    "## Interpretation (hand-written, 2026-09-23 run; hypotheses, not conclusions)",
    "",
    "1. Hourly vs daily PET is *not* the explanation here: the CA-TPA PET block has hourly weather "
    "off (`hourly_weather = 0`), and with the weather echoed in `.ana` the daily ASCE reference ET "
    "matches cols 81/82 to print precision. So the daily POTEVPHR branch is the right target.",
    "2. Forcing preprocessing. RZWQM floors the daily wind run at 100 km/d (every `.MET` value "
    "below 100 is echoed as 100); the Agri-JAX forcing layer does not, which is the whole "
    "MET-vs-ana difference in reference ET. It also uses a solar radiation that differs from the "
    "2-decimal `.MET` value by up to about 0.1 MJ/m2/d (unexplained: maybe a unit round trip or "
    "the hourly disaggregation re-summed); its effect on PET is second order. A wind floor belongs "
    "in the io/forcing layer, not in the PET process: it is now "
    "`agri_jax.io.rzwqm.prepare_rzwqm_forcing` (INPDAY, `UBREEZ = 100`, Rzmain.for line 3543), "
    "checked day by day against `.ana` col 90 in tests/integration/test_pet_oracle.py; the raw-MET "
    "table above deliberately does not apply it.",
    "3. Canopy timing. The reference evaluates PET with the start-of-day canopy (LAI and height of "
    "the previous `.ana` row), like residue mass and surface theta. The same-row driver (the "
    "primary table, as the task specified) is wrong by up to 1.7 mm/d on days of fast LAI or "
    "height change. The exception is the day after harvest, where the canopy is already gone at "
    "the PET call. When PET is coupled to the crop module in a `Model`, the process order has to "
    "put PET before the crop update.",
    "4. Residue after harvest. The remaining PE error is concentrated in the weeks after harvest, "
    "when RZWQM switches the residue-type constants (IPR) and adds harvest residue within the day, "
    "which the PET module cannot see from its inputs. Jan 1 is an artefact of this harness "
    "(LAYER.PLT has no initial-state row, so Jan 1 uses its own end-of-day surface theta).",
    "5. Hydraulics. The Brooks-Corey curves with the rzwqm.dat horizon parameters reproduce the "
    "reference theta from its own pressure heads on all 37 nodes to 1e-4, except in horizon 1 "
    "(0-15 cm) for about a week after the 2015-04-14 tillage (15 cm deep). This is consistent with "
    "the tillage-modified curve (TRHYDP) that the module deliberately does not implement. Profile "
    "storage from theta(h) matches col 2 to a few hundredths of a cm, and the 2015 range of col 2 "
    "lies between the uniform wilting-point and saturated storages, as it should.",
    "",
]


def error_distribution_table(res: CoarseResult) -> list[str]:
    """Median / 95th percentile of the daily |sim - ref| and the worst days, per weather source."""
    out = [
        "## Distribution of daily absolute errors (whole year)",
        "",
        "| weather | variable | median abs | p95 abs | days abs > 0.1 mm | worst days (abs err, mm) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for src, sim in res.pet_sim.items():
        err = np.abs(sim - res.pet_ref).to_dataframe()
        for v in PET_COLUMNS:
            e = err[v]
            worst = e.sort_values(ascending=False).head(3)
            wtxt = ", ".join(f"{d.date()} ({x:.3f})" for d, x in worst.items())
            out.append(
                f"| {src} | `{v}` | {e.median():.2e} | {e.quantile(0.95):.3f} "
                f"| {int((e > 0.1).sum())} | {wtxt} |"
            )
    return [*out, ""]


def diagnostics(res: CoarseResult, inp: Inputs) -> list[str]:
    """Measured facts behind the hypotheses for the remaining gaps."""
    days = inp.days
    met_wind = inp.met["wind_run_km"].to_numpy(dtype=float)
    ana_wind = inp.column(90).sel(time=days).to_numpy().astype(float)
    low = met_wind < ana_wind - 1.0
    echoed = sorted({round(float(a), 3) for a in ana_wind[low]})
    floor_txt = f"The echoed values there are {', '.join(f'{a:g}' for a in echoed)} km/d." if echoed else ""
    min_echo = float(ana_wind.min())
    clamp_txt = ", ".join(
        f"{d.date()} (MET {m:.1f}, .ana {a:.1f})" for d, m, a in zip(days[low], met_wind[low], ana_wind[low])
    )
    start, end = res.season
    after = (days > end) & (days <= end + pd.Timedelta(days=60))
    e_pe = np.abs(res.pet_sim["ana"]["pot_evap_mm"] - res.pet_ref["pot_evap_mm"]).to_numpy()
    pre = (days < start) & (days.month >= 4)
    theta_err = np.abs(storage_from_heads(inp)["theta_err"])
    err = theta_err.max("depth").to_pandas()
    bad_days = err[err > 1e-3]
    bad_depths = theta_err["depth"].to_numpy()[(theta_err > 1e-3).any("time").to_numpy()]
    depth_txt = f"nodes {bad_depths.min():g}-{bad_depths.max():g} cm" if bad_depths.size else "no node"
    top_bottom = float(np.asarray(inp.dat.horizon_depths_cm)[0])
    # PT outliers: the same days with the canopy state (LAI, height) of the previous .ana row
    pt_ref = res.pet_ref["pot_transp_mm"].to_pandas()
    pt0 = res.pet_sim["ana"]["pot_transp_mm"].to_pandas()
    pt1 = pet_simulation(inp, "ana", canopy_lag=1)["pot_transp_mm"].to_pandas()
    e0, e1 = (pt0 - pt_ref).abs(), (pt1 - pt_ref).abs()
    worst = e0.sort_values(ascending=False).head(3).index
    lag_txt = "; ".join(f"{d.date()}: {e0[d]:.3f} -> {e1[d]:.2e} mm" for d in worst)
    season_mask = (pt_ref.index >= start) & (pt_ref.index <= end)
    n_lag_better = int(((e1 < e0 - 1e-3) & season_mask).sum())
    n_lag_worse = int(((e1 > e0 + 1e-3) & season_mask).sum())
    return [
        "## Measured facts behind the gaps",
        "",
        f"- Wind: {int(low.sum())} days where the `.MET` wind run is below the value echoed in `.ana`: "
        f"{clamp_txt or 'none'}. {floor_txt} Minimum echoed wind run over the year: {min_echo:g} km/d.",
        f"- Mean |PE error| (`.ana` weather) in the 60 days after harvest ({end.date()}): "
        f"{float(e_pe[after].mean()):.3f} mm/d, "
        f"vs {float(e_pe[pre].mean()):.3f} mm/d from April to planting.",
        f"- theta_of_h vs LAYER.PLT theta: {len(bad_days)} days with a node error > 1e-3 "
        f"({', '.join(str(d.date()) for d in bad_days.index)}); {depth_txt} "
        f"(horizon 1 ends at {top_bottom:g} cm).",
        f"- PT outliers (`.ana` weather) recomputed with the previous row's LAI and height: {lag_txt}. "
        f"Over the season that lag improves {n_lag_better} days and worsens {n_lag_worse} days "
        "(by > 1e-3 mm): the reference evaluates PET with the start-of-day canopy (previous row), "
        "the same convention as for residue mass and surface theta.",
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
