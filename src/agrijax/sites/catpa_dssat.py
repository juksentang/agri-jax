"""CA-TPA as a DSSAT-CSM v4.8.6.0 experiment: the seven maize seasons 2015-2021 of the RZWQM2
scenario, converted from the scenario's own inputs, for an independent 4.8.6 reference run.

The DSSAT input files are data: :func:`build_case` writes them under a directory of the data
tree (``$AGRI_JAX_DATA/dssat_catpa/<genotype set>``), never into the repository. One case
directory holds:

=====================  =====================================================================
file                   content and source
=====================  =====================================================================
``CTPA1501.MZX``       the experiment (:func:`write_filex`): 14 treatments, the 7 seasons with
                       water on / N off (1-7) and water on / N on (8-14); sowing, density,
                       depth, row spacing and fixed-date harvest from the season table
                       (:func:`agrijax.sites.catpa_m3.season_table`); the scenario's fertiliser
                       (:func:`fertilizer_levels`); per-season initial conditions from the
                       RZWQM2 state of the day before sowing (:func:`initial_conditions`)
``CT.SOL``             the soil (:func:`catpa_soil_profile`): the ``rzwqm.dat`` horizons on the
                       crop's DSSAT layers, LL / DUL from the Brooks-Corey curve
                       (:func:`brooks_corey_theta`) at the scenario's wilting-point and
                       field-capacity heads, SAT = theta_s, SKS = Ksat
``CTPAyy01.WTH``       one file per year (:func:`catpa_weather`): the ``.MET`` record after
                       the daily preparation (SRAD the prepared ``.MET`` value), RAIN = the
                       day's ``.BRK`` storms after the skip threshold; CO2
                       from the site line (``CCO2``, run with ``CO2 = W``)
``MZCER048.CUL/.ECO/   genotype set ``catpa`` only (:func:`write_catpa_genotype`): the 4.8.6
.SPE``                 files with the CA-TPA cultivar row ``CT0012`` (the embedded crop's
                       ``MZCER040.CUL`` values) and the species values of the embedded crop's
                       ``MZCER040.SPE``; every difference is listed (:func:`genotype_differences`)
=====================  =====================================================================

Genotype sets: ``catpa`` runs the parameter set our CERES-Maize uses for CA-TPA
(:func:`agrijax.sites.catpa_m3.ceres_maize_params`: the embedded crop's 4.0 cultivar, ecotype and
species values, ``TSEN`` / ``CDAY`` from 4.8.6); ``stock`` runs the unmodified 4.8.6 files with
the stock ``IB0012 PIO 3382`` row.

Conversion choices (every one listed in :data:`DSSAT_ONLY_SETTINGS` or in the docstrings below):

* layers: the crop's DSSAT layers of RZWQM2 (5, 15, 30, 45, 60, then 30 cm steps; RZWQM2 4.5
  DSSATDRV.for:557-564) to the profile depth; horizon and node values are averaged over each
  layer by overlap thickness, the way DSSATDRV maps RZWQM2 nodes to the crop's layers
  (``realMATCH``, DSSATDRV.for:579-584) -- checked against the reference's ``SOILPROP``;
* LL / DUL: RZWQM2 gives the embedded crop the water content of its Brooks-Corey curve at the
  control record's wilting-point and field-capacity heads (``rzwqm.dat`` items 18-19, CA-TPA
  -15000 and -333 cm; RZWQM2 4.5 Rzman.for:5812-5813, curve ``WC`` Rzrich.for:1971); SAT is the
  horizon's theta_s (Rzman.for:5806); SKS is the horizon's Ksat [cm h-1];
* soil organic C and N, pH: the reference's values on the crop's first day (``DSSATDRV`` entry
  ``RZOC``, ``RZON``, ``CGPH`` per node, passed in as :class:`OrganicProfile`);
* the DSSAT-only surface parameters (albedo, stage-1 evaporation limit, drainage rate, runoff
  curve number) have no RZWQM2 counterpart and come from the DSSAT loamy sand profile
  ``UFWH940004`` (:data:`SURFACE_TEMPLATE`); ``SLNF`` / ``SLPF`` and the root growth factors
  from the scenario's ``MZDSSAT.RZX`` (:func:`agrijax.io.rzwqm.rzx.read_rzx`);
* not converted: tillage (the chisel passes of ``rzwqm.dat``; DSSAT runs with ``TILL = N``),
  pesticide, surface residue (none at the simulation start).

Nothing here imports JAX. Validated in ``tests/integration/test_io_catpa_dssat.py`` (soil
against the reference ``DSSATDRV`` table, weather against ``catpa_m3``, genotype against
``ceres_maize_params``, and the dscsm048 echo of every input).
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agrijax.io.dssat.genotype import read_cul, read_eco, read_spe, write_cul, write_eco, write_spe
from agrijax.io.dssat.sol import SoilProfile, write_sol
from agrijax.io.dssat.wth import write_wth
from agrijax.io.rzwqm.dat import RzwqmDat, read_rzwqm_dat
from agrijax.io.rzwqm.events import read_management
from agrijax.io.rzwqm.met import _R2D, prepare_rzwqm_forcing, read_brk, read_met
from agrijax.io.rzwqm.rzx import RzxControl, read_rzx
from agrijax.io.rzwqm.storms import read_met_modifiers, storm_arrays
from agrijax.sites.catpa_m3 import CATPA_VARNO, CERES_SPECIES_KEYS, CatpaPaths, SeasonTable, season_table

__all__ = [
    "CASE_FILEX",
    "CASE_INSI",
    "CASE_SOIL_ID",
    "CATPA_CULTIVAR_ID",
    "DSSAT_LAYER_BOTTOMS_CM",
    "DSSAT_ONLY_SETTINGS",
    "GENOTYPE_SETS",
    "N_OPTIONS",
    "SURFACE_TEMPLATE",
    "CaseFiles",
    "GenotypeDiff",
    "OrganicProfile",
    "Setting",
    "brooks_corey_theta",
    "build_case",
    "catpa_soil_profile",
    "catpa_weather",
    "fertilizer_levels",
    "genotype_differences",
    "initial_conditions",
    "layer_average",
    "read_overview_cultivar",
    "read_overview_soil",
    "read_overview_stages",
    "run_case",
    "season_results",
    "treatment_numbers",
    "weather_site",
    "write_catpa_genotype",
    "write_filex",
    "write_weather_files",
]

#: institute + site code of the case (``WSTA``, weather files ``CTPAyy01.WTH``, experiment name)
CASE_INSI = "CTPA"
#: the experiment file (``<INSI><yy><nn>.MZX``: first season 2015, experiment 01)
CASE_FILEX = "CTPA1501.MZX"
#: soil profile id; DSSAT looks it up in ``<first two letters>.SOL`` (``CT.SOL``)
CASE_SOIL_ID = "CTPA150001"
#: cultivar id of the CA-TPA row added to ``MZCER048.CUL`` (the embedded crop's ``IB0012`` values)
CATPA_CULTIVAR_ID = "CT0012"
#: genotype sets: ``(cultivar id, cultivar name)``
GENOTYPE_SETS: dict[str, tuple[str, str]] = {
    "catpa": (CATPA_CULTIVAR_ID, "PIO 3382 RZWQM2"),
    "stock": (CATPA_VARNO, "PIO 3382"),
}
#: the two water / nitrogen configurations: ``(DSSAT NITRO switch, label)``; water is on in both
N_OPTIONS: tuple[tuple[str, str], ...] = (("N", "water on, N off"), ("Y", "water on, N on"))

#: bottoms of the crop's DSSAT soil layers [cm]: ``DS`` = 5, 15, 30, 45, 60, then +30 cm
#: (RZWQM2 4.5 DSSATDRV.for:557-564), cut at the profile depth
_DS_FIRST_CM = (5.0, 15.0, 30.0, 45.0, 60.0)
_DS_STEP_CM = 30.0
_CATPA_PROFILE_CM = 150.0
DSSAT_LAYER_BOTTOMS_CM: np.ndarray = np.concatenate(
    [np.array(_DS_FIRST_CM), np.arange(_DS_FIRST_CM[-1] + _DS_STEP_CM, _CATPA_PROFILE_CM + 1.0, _DS_STEP_CM)]
)
#: unit conversion fraction -> percent (``rzwqm.dat`` sand/silt/clay fractions -> DSSAT %)
_PERCENT = 100.0
#: unit conversion cm -> mm (storm depths -> DSSAT RAIN)
_MM_PER_CM = 10.0
#: decimals written to the ``.WTH`` columns (DSSAT reads them as REAL; SRAD and the temperatures
#: change by at most 5e-4 in their unit)
#: decimals of the ``.SOL`` layer values (the reference's REAL values to 5e-5)
SOL_DECIMALS = 4
WTH_DECIMALS: dict[str, int] = {"srad": 3, "tmax": 3, "tmin": 3, "rain": 3, "rhum": 2, "wind": 1}


@dataclass(frozen=True)
class Setting:
    """A DSSAT input with no RZWQM2 counterpart (or not converted), its value and why."""

    name: str
    value: Any
    source: str


#: surface line of the soil: DSSAT-only parameters from the DSSAT-CSM v4.8.6.0 loamy sand profile
#: ``UFWH940004`` (Eustis loamy sand; ``example_data/Soil/SOIL.SOL:1111``); ``SLNF`` and ``SLPF``
#: are set from ``MZDSSAT.RZX`` in :func:`catpa_soil_profile`
SURFACE_TEMPLATE: dict[str, Any] = {
    "SCOM": "BN",
    "SALB": 0.13,
    "SLU1": 8.5,
    "SLDR": 0.53,
    "SLRO": 64.0,
    "SLNF": 1.0,
    "SLPF": 1.0,
    "SMHB": "IB001",
    "SMPX": "IB001",
    "SMKE": "IB001",
}

DSSAT_ONLY_SETTINGS: tuple[Setting, ...] = (
    Setting(
        "SOL SALB, SLU1, SLDR, SLRO",
        "0.13, 8.5 mm, 0.53 d-1, 64",
        "DSSAT-CSM 4.8.6 example_data/Soil/SOIL.SOL:1111 (UFWH940004 Eustis loamy sand); RZWQM2 has "
        "no tipping-bucket drainage, runoff curve number or Ritchie stage-1 limit (Green-Ampt + "
        "Richards); its 'albedo of the dry soil' (passed to the embedded crop as SALB) is not "
        "a DSSAT soil albedo",
    ),
    Setting(
        "SOL SSAT, SBDM",
        "theta_s, bulk density of the untilled horizons",
        "on crop days the embedded crop sees the tilled top 15 cm (the tilled SAT and BD of SOILPROP); the "
        "static .SOL keeps the rzwqm.dat horizon values",
    ),
    Setting(
        "SOL SSKS",
        "thickness-weighted Ksat",
        "the 60-90 cm layer mixes two horizons; RZWQM2 passes no Ksat to the crop",
    ),
    Setting("SOL SLHB, SCEC, SADC, SLCF", "-99, -99, -99, 0", "not in rzwqm.dat (no coarse fraction)"),
    Setting(
        "FileX initial SNH4",
        "0 ppm",
        "LAYER.PLT holds no NH4; the reference DSSATDRV entry NH4R on the sowing days is <= 0.003 per node",
    ),
    Setting(
        "FileX initial residue", "none", "the run starts on the sowing day; RZWQM2 residue pools not mapped"
    ),
    Setting("FileX TILL", "N", "the rzwqm.dat chisel passes (15 cm) are not converted"),
    Setting(
        "FileX EVAPO",
        "R (Priestley-Taylor)",
        "the DSSAT PET that agrijax ports (processes/pet/priestley_taylor.py); "
        "RZWQM2 uses Shuttleworth-Wallace",
    ),
    Setting("FileX MESOM", "G (Godwin SOM)", "DSSAT default soil organic matter method"),
    Setting(
        "FileX CO2", "W (site line CCO2)", "RZWQM2 passes CO2 = 330 ppm x METMOD row 8 (100 %) to the crop"
    ),
    Setting("FileX HARVS", "R (reported date)", "RZWQM2 harvest option 3 (fixed date)"),
)


# ============================================================================ soil
def brooks_corey_theta(
    h_cm: float | np.ndarray,
    hb: float | np.ndarray,
    lam: float | np.ndarray,
    theta_r: float | np.ndarray,
    theta_s: float | np.ndarray,
) -> np.ndarray:
    """Water content of the Brooks-Corey curve at suction ``|h_cm|`` >= ``hb`` [cm]:
    ``theta_r + (theta_s - theta_r) (hb / |h|)^lam`` (Brooks and Corey 1964, Hydrology Paper 3,
    eq. for the effective saturation ``Se = (hb / h)^lam``; the curve RZWQM2 evaluates in ``WC``,
    RZWQM2 4.5 Rzrich.for:1971, whose modified branch ``|h| < hb`` is not needed here and
    raises)."""
    h = np.abs(np.asarray(h_cm, dtype=float))
    hbv = np.asarray(hb, dtype=float)
    if np.any(h < hbv):
        raise ValueError(
            "brooks_corey_theta: suction below the bubbling pressure (modified branch not implemented)"
        )
    return np.asarray(theta_r, float) + (np.asarray(theta_s, float) - np.asarray(theta_r, float)) * (
        hbv / h
    ) ** np.asarray(lam, float)


def layer_average(
    src_bottoms: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    dst_bottoms: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Values of the layers ``src_bottoms`` (bottoms [cm], top at 0) on the layers ``dst_bottoms``:
    the overlap-thickness (times ``weights``) average of every source layer a target layer
    touches (the mapping of RZWQM2 4.5 ``realMATCH``, DSSATDRV.for:579-584). The target profile
    must not be deeper than the source."""
    zs = np.concatenate([[0.0], np.asarray(src_bottoms, float)])
    zd = np.concatenate([[0.0], np.asarray(dst_bottoms, float)])
    if zd[-1] > zs[-1] + 1e-9:
        raise ValueError(f"target profile ({zd[-1]} cm) deeper than the source ({zs[-1]} cm)")
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    lo = np.maximum(zd[:-1, None], zs[None, :-1])
    hi = np.minimum(zd[1:, None], zs[None, 1:])
    ov = np.clip(hi - lo, 0.0, None) * w[None, :]
    return (ov @ v) / ov.sum(axis=1)


@dataclass(frozen=True)
class OrganicProfile:
    """Soil organic C and N [%] and pH on the RZWQM2 nodes (bottoms ``node_bottoms_cm``): the
    reference's ``DSSATDRV`` entry ``RZOC``, ``RZON``, ``CGPH`` of the crop's first day."""

    node_bottoms_cm: np.ndarray
    oc_pct: np.ndarray
    on_pct: np.ndarray
    ph: np.ndarray


def _horizon_hydraulics(dat: RzwqmDat) -> dict[str, np.ndarray]:
    """LL, DUL, SAT, SKS per horizon (module docstring): Brooks-Corey at the control heads."""
    h = dat.hydraulics
    heads = dat.water_suction_heads_cm
    ll = brooks_corey_theta(heads["hwp"], h["hb"], h["lam"], h["theta_r"], h["theta_s"])
    dul = brooks_corey_theta(heads["hfc"], h["hb"], h["lam"], h["theta_r"], h["theta_s"])
    return {"ll": ll, "dul": dul, "sat": h["theta_s"].astype(float), "sks": h["ksat"].astype(float)}


def catpa_soil_profile(
    dat: RzwqmDat,
    rzx: RzxControl,
    organic: OrganicProfile,
    *,
    bottoms: Sequence[float] | np.ndarray = DSSAT_LAYER_BOTTOMS_CM,
    soil_id: str = CASE_SOIL_ID,
) -> SoilProfile:
    """The CA-TPA soil as a DSSAT profile (module docstring): the ``rzwqm.dat`` horizons on the
    layers ``bottoms``; ``SRGF`` from ``rzx`` (one factor per layer), ``SLOC`` / ``SLNI`` /
    ``SLHW`` from ``organic``."""
    zb = np.asarray(bottoms, float)
    if len(rzx.srgf) != len(zb):
        raise ValueError(f"{len(rzx.srgf)} root growth factors in the .RZX for {len(zb)} layers")
    hz = dat.horizon_depths_cm
    hyd = _horizon_hydraulics(dat)
    phy = dat.soil_physical

    def on(v: np.ndarray) -> np.ndarray:
        return np.round(layer_average(hz, v, zb), SOL_DECIMALS)

    def on_nodes(v: np.ndarray) -> np.ndarray:
        return np.round(layer_average(organic.node_bottoms_cm, v, zb), SOL_DECIMALS)

    layers = pd.DataFrame(
        {
            "SLB": zb,
            "SLMH": ["-99"] * len(zb),
            "SLLL": on(hyd["ll"]),
            "SDUL": on(hyd["dul"]),
            "SSAT": on(hyd["sat"]),
            "SRGF": np.asarray(rzx.srgf, float),
            "SSKS": on(hyd["sks"]),
            "SBDM": on(phy["bulk_density"]),
            "SLOC": on_nodes(organic.oc_pct),
            "SLCL": on(phy["clay"] * _PERCENT),
            "SLSI": on(phy["silt"] * _PERCENT),
            "SLCF": np.zeros(len(zb)),
            "SLNI": on_nodes(organic.on_pct),
            "SLHW": on_nodes(organic.ph),
            "SLHB": np.full(len(zb), np.nan),
            "SCEC": np.full(len(zb), np.nan),
            "SADC": np.full(len(zb), np.nan),
        }
    )
    ph = dat.physiography
    surface = dict(SURFACE_TEMPLATE, SLNF=rzx.slnf, SLPF=rzx.slpf)
    return SoilProfile(
        id=soil_id,
        source="RZWQM2",
        texture="LS",
        depth=float(zb[-1]),
        description="CA-TPA loamy sand from RZWQM2 rzwqm.dat",
        site={
            "SITE": "CA-TPA",
            "COUNTRY": "Canada",
            "LAT": round(float(ph["latitude_rad"]) * _R2D, 4),
            "LONG": round(float(ph["longitude_rad"]) * _R2D, 4),
            "SCS FAMILY": "rzwqm.dat horizons, Brooks-Corey LL/DUL",
        },
        surface=surface,
        layers=layers,
        layer_groups=[list(layers.columns)],
    )


# ============================================================================ initial conditions
def initial_conditions(
    layer_plt: Any,
    day: np.datetime64 | str,
    *,
    bottoms: Sequence[float] | np.ndarray = DSSAT_LAYER_BOTTOMS_CM,
) -> pd.DataFrame:
    """``ICBL SH2O SNH4 SNO3`` on the layers ``bottoms`` from the RZWQM2 state at the end of
    ``day`` (``LAYER.PLT`` read by :func:`agrijax.io.rzwqm.layers.read_layer_output`): soil water
    averaged over thickness, NO3-N [ug g-1] over soil mass (thickness x bulk density), NH4 zero
    (:data:`DSSAT_ONLY_SETTINGS`)."""
    d = pd.Timestamp(_day(day))
    s = layer_plt.sel(time=d)
    z = np.asarray(layer_plt["depth"].values, float)
    zb = np.asarray(bottoms, float)
    sw = layer_average(z, s["soil_water_content"].values, zb)
    no3 = layer_average(z, s["no3_n_conc"].values, zb, weights=s["soil_bulk_density"].values)
    return pd.DataFrame({"ICBL": zb, "SH2O": sw, "SNH4": np.zeros(len(zb)), "SNO3": no3})


# ============================================================================ weather
def catpa_weather(
    paths: CatpaPaths, dat: RzwqmDat, start: str | np.datetime64, end: str | np.datetime64
) -> pd.DataFrame:
    """Daily DSSAT weather of ``[start, end]`` (``date srad tmax tmin rain rhum wind``): the
    ``.MET`` record after the daily preparation with the ``IPNAMES.DAT`` modifiers
    (:func:`~agrijax.io.rzwqm.met.prepare_rzwqm_forcing`, ``srad`` the prepared ``.MET`` value
    [MJ m-2 d-1]), wind run [km d-1], RH [%], and ``rain`` [mm] = the day's ``.BRK`` storm depth
    after the skip threshold and the rainfall modifier (:func:`~agrijax.io.rzwqm.storms.storm_arrays`)."""
    metmod = read_met_modifiers(paths["IPNAMES.DAT"])
    met = prepare_rzwqm_forcing(read_met(paths["CA-TPA.MET"]), met_modifiers=metmod)
    days = np.arange(_day(start), _day(end) + np.timedelta64(1, "D"))
    idx = pd.DatetimeIndex(days)
    m = met.loc[idx]
    storms = storm_arrays(days, read_brk(paths["CA-TPA.BRK"]), met_modifiers=metmod)
    return pd.DataFrame(
        {
            "date": idx,
            "srad": m["srad_mj"].to_numpy(float),
            "tmax": m["tmax"].to_numpy(float),
            "tmin": m["tmin"].to_numpy(float),
            "rain": storms.total_cm * _MM_PER_CM,
            "rhum": m["rh"].to_numpy(float),
            "wind": m["wind_run_km"].to_numpy(float),
        }
    )


def weather_site(dat: RzwqmDat, weather: pd.DataFrame) -> dict[str, Any]:
    """The ``.WTH`` site line: ``INSI LAT LONG ELEV`` from ``rzwqm.dat``; ``TAV`` [degC] the mean of
    the daily mean temperature of ``weather`` and ``AMP`` [degC] the range of its monthly means
    (DSSAT uses ``TAV + AMP cos(.) / 2``, SPAM/STEMP.for:351); ``REFHT`` / ``WNDHT`` the
    ``rzwqm.dat`` wind measurement height [m]; ``CCO2`` [ppm] the ambient CO2 of ``rzwqm.dat``."""
    ph = dat.physiography
    tav: Any = 0.5 * (weather["tmax"] + weather["tmin"])
    months: Any = pd.to_datetime(weather["date"]).dt.month
    monthly: Any = tav.groupby(months.to_numpy()).mean()
    zw = float(dat.pet["wind_height_m"])
    return {
        "INSI": CASE_INSI,
        "LAT": round(float(ph["latitude_rad"]) * _R2D, 4),
        "LONG": round(float(ph["longitude_rad"]) * _R2D, 4),
        "ELEV": float(ph["elevation_m"]),
        "TAV": round(float(tav.mean()), 1),
        "AMP": round(float(monthly.max() - monthly.min()), 1),
        "REFHT": zw,
        "WNDHT": zw,
        "CCO2": float(ph["co2_ppm"]),
    }


def write_weather_files(weather: pd.DataFrame, site: Mapping[str, Any], out_dir: str | Path) -> list[Path]:
    """One ``CTPAyy01.WTH`` per calendar year of ``weather`` (values rounded to :data:`WTH_DECIMALS`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = weather.copy()
    for c, d in WTH_DECIMALS.items():
        w[c] = w[c].round(d)
    w.attrs["decimals"] = {c: 1 for c in WTH_DECIMALS}
    years: Any = pd.to_datetime(w["date"]).dt.year.to_numpy()
    paths = []
    for y in sorted(set(years)):
        part = w.loc[years == y].reset_index(drop=True)
        part.attrs = dict(w.attrs)
        name = f"{site['INSI']}{y % 100:02d}01.WTH"
        paths.append(
            write_wth(
                part, out / name, site=site, title="CA-TPA from RZWQM2 .MET/.BRK", four_digit_year=False
            )
        )
    return paths


# ============================================================================ genotype
@dataclass(frozen=True)
class GenotypeDiff:
    """A CERES-Maize genotype value that differs between the RZWQM2 4.0 file and DSSAT 4.8.6.

    ``rzwqm_40`` / ``dssat_486`` are ``None`` where the file has no such entry; ``used`` is the
    value of the ``catpa`` genotype set."""

    file: str
    name: str
    rzwqm_40: Any
    dssat_486: Any
    used: Any


def _num_equal(a: Any, b: Any) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (
            isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) and all(map(_num_equal, a, b))
        )
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip() == str(b).strip()
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-12)) or (
        np.isnan(float(a)) and np.isnan(float(b))
    )


def _species_used(name: str) -> bool:
    """True for the species entries our CERES reads from the 4.0 file (catpa_m3.ceres_maize_params)."""
    return name in CERES_SPECIES_KEYS or name in ("PORM", "RWMX")


def genotype_differences(
    paths: CatpaPaths, genotype48: str | Path, *, varno: str = CATPA_VARNO
) -> list[GenotypeDiff]:
    """Every cultivar (``varno``), ecotype (its ``ECO#``) and species value that differs between
    the embedded crop's 4.0 files (project ``MZCER040.CUL``, database ``MZCER040.ECO/.SPE``,
    :class:`~agrijax.sites.catpa_m3.CatpaPaths`) and the DSSAT-CSM 4.8.6 ``MZCER048`` files in
    ``genotype48``, with the value the ``catpa`` set uses."""
    g = Path(genotype48)
    out: list[GenotypeDiff] = []
    c40 = read_cul(paths["MZCER040.CUL"]).loc[varno]
    c48 = read_cul(g / "MZCER048.CUL").loc[varno]
    for k in sorted(set(c40.index) | set(c48.index), key=lambda s: (s not in c40.index, s)):
        if k in ("VRNAME", "EXPNO", "ECO#"):
            continue
        a, b = c40.get(k), c48.get(k)
        if a is None or b is None or not _num_equal(a, b):
            out.append(GenotypeDiff("CUL", k, a, b, a if k in c48.index else None))
    eco = str(c40["ECO#"])
    e40 = read_eco(paths.species_db / "MZCER040.ECO").loc[eco]
    e48 = read_eco(g / "MZCER048.ECO").loc[eco]
    for k in list(e48.index) + [x for x in e40.index if x not in e48.index]:
        if k == "ECONAME":
            continue
        a, b = e40.get(k), e48.get(k)
        if a is None or b is None or not _num_equal(a, b):
            out.append(GenotypeDiff("ECO", k, a, b, b if a is None else a))
    s40 = read_spe(paths.species_db / "MZCER040.SPE").params
    s48 = read_spe(g / "MZCER048.SPE").params
    for k in list(s48) + [x for x in s40 if x not in s48]:
        a, b = s40.get(k), s48.get(k)
        if a is None or b is None or not _num_equal(a, b):
            used = a if (_species_used(k) and a is not None) else b
            out.append(GenotypeDiff("SPE", k, a, b, used))
    return out


def write_catpa_genotype(
    paths: CatpaPaths,
    genotype48: str | Path,
    out_dir: str | Path,
    *,
    varno: str = CATPA_VARNO,
    cultivar_id: str = CATPA_CULTIVAR_ID,
    cultivar_name: str = GENOTYPE_SETS["catpa"][1],
) -> list[Path]:
    """The ``catpa`` genotype files in ``out_dir``: the 4.8.6 ``MZCER048.CUL`` with a row
    ``cultivar_id`` carrying the 4.0 cultivar values of ``varno``; ``MZCER048.ECO`` with the
    4.0 values of its ecotype (``TSEN`` / ``CDAY`` stay 4.8.6's); ``MZCER048.SPE`` with the 4.0
    values of the entries our CERES reads from the 4.0 file (``CERES_SPECIES_KEYS``, ``PORM``,
    ``RWMX``). Written layout preserving (DSSAT's fixed formats)."""
    g, out = Path(genotype48), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c40 = read_cul(paths["MZCER040.CUL"]).loc[varno]
    cul = read_cul(g / "MZCER048.CUL")
    row = cul.loc[[varno]].copy()
    row.index = pd.Index([cultivar_id], name=cul.index.name)
    row["VRNAME"] = cultivar_name
    for k in ("P1", "P2", "P5", "G2", "G3", "PHINT"):
        row[k] = float(c40[k])
    cul2 = pd.concat([cul, row])
    cul2.attrs = cul.attrs
    files = [write_cul(cul2, out / "MZCER048.CUL")]
    eco = read_eco(g / "MZCER048.ECO")
    e40 = read_eco(paths.species_db / "MZCER040.ECO").loc[str(c40["ECO#"])]
    for k in e40.index:
        if k != "ECONAME" and k in eco.columns:
            eco.loc[str(c40["ECO#"]), k] = e40[k]
    files.append(write_eco(eco, out / "MZCER048.ECO"))
    spe = read_spe(g / "MZCER048.SPE")
    s40 = read_spe(paths.species_db / "MZCER040.SPE").params
    for k, v in s40.items():
        if _species_used(k) and k in spe.params:
            spe.params[k] = v
    files.append(write_spe(spe, out / "MZCER048.SPE"))
    return files


# ============================================================================ FileX
_FX_TEXT = frozenset(
    {
        "TNAME",
        "CNAME",
        "ID_FIELD",
        "WSTA",
        "SLTX",
        "ID_SOIL",
        "FLNAME",
        "ICNAME",
        "PLNAME",
        "FERNAME",
        "HNAME",
        "SNAME",
        "SMODEL",
    }
)
#: second tokens of the simulation-control headers: their two-letter code sits at the token start
_FX_GROUPS = frozenset(
    {
        "GENERAL",
        "OPTIONS",
        "METHODS",
        "MANAGEMENT",
        "OUTPUTS",
        "PLANTING",
        "IRRIGATION",
        "NITROGEN",
        "RESIDUES",
        "HARVEST",
    }
)


def _fx_num(v: Any, width: int) -> str:
    """Text of a FileX number in ``width`` columns: integers as is; reals with a decimal point and
    as many decimals (at most 4) as fit (DSSAT reads the fields with ``F`` formats)."""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    x = float(v)
    if np.isnan(x):
        return "-99"
    if x == 0.0:
        return "0"
    if x.is_integer() and len(str(int(x))) <= width:
        return str(int(x))
    for d in range(4, -1, -1):
        t = f"{x:.{d}f}" if d else f"{x:.0f}."
        if d and "." in t:
            t = t.rstrip("0")
        if t.startswith("0."):
            t = t[1:]
        elif t.startswith("-0."):
            t = "-" + t[2:]
        if len(t) <= width:
            return t
    raise ValueError(f"{x!r} does not fit a {width}-column FileX field")


def _fx_line(header: str, values: Sequence[Any]) -> str:
    """A FileX data line under ``header``: codes and numbers right-aligned to the end of their
    header token (each in the columns after the previous token's end and a blank), names
    (:data:`_FX_TEXT`) left-aligned at the token start; the second token of a simulation-control
    header (``GENERAL`` ...) takes its two-letter code at the token start."""
    toks = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", header)]
    if len(values) > len(toks):
        raise ValueError(f"{len(values)} values for {len(toks)} header tokens: {header!r}")
    line = [" "] * (max(e for _, _, e in toks) + 40)
    prev_end = 0
    for i, ((name, a, b), v) in enumerate(zip(toks, values, strict=False)):
        key = name.strip(".")
        if key in _FX_TEXT or (i == 1 and key in _FX_GROUPS):
            t = str(v)
            start = a
        else:
            t = _fx_num(v, b - prev_end - 1 if i else b)
            start = b - len(t)
        if start < prev_end + (1 if i else 0):
            raise ValueError(f"value {t!r} of {name} overlaps the previous field in {header!r}")
        end = start + len(t)
        if end > len(line):
            line.extend([" "] * (end - len(line)))
        line[start:end] = list(t)
        prev_end = max(b, end)
    return "".join(line).rstrip()


def _day(x: Any) -> Any:
    """``x`` (str, date or datetime64) as a ``datetime64[D]`` scalar."""
    return np.asarray(x, dtype="datetime64[D]")[()]


def _yyddd(d: np.datetime64 | date) -> int:
    t = pd.Timestamp(d)
    return int((t.year % 100) * 1000 + t.dayofyear)


_H_TRT = "@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM"
_H_CUL = "@C CR INGENO CNAME"
_H_FLD = "@L ID_FIELD WSTA....  FLSA  FLOB  FLDT  FLDD  FLDS  FLST SLTX  SLDP  ID_SOIL    FLNAME"
_H_IC = "@C   PCR ICDAT  ICRT  ICND  ICRN  ICRE  ICWD ICRES ICREN ICREP ICRIP ICRID ICNAME"
_H_ICL = "@C  ICBL  SH2O  SNH4  SNO3"
_H_PL = (
    "@P PDATE EDATE  PPOP  PPOE  PLME  PLDS  PLRS  PLRD  PLDP  PLWT  PAGE  PENV  PLPH  SPRL"
    "                        PLNAME"
)
_H_FE = "@F FDATE  FMCD  FACD  FDEP  FAMN  FAMP  FAMK  FAMC  FAMO  FOCD FERNAME"
_H_HA = "@H HDATE  HSTG  HCOM HSIZE   HPC  HBPC HNAME"
_H_SIM = (
    "@N GENERAL     NYERS NREPS START SDATE RSEED SNAME....................",
    "@N OPTIONS     WATER NITRO SYMBI PHOSP POTAS DISES  CHEM  TILL   CO2",
    "@N METHODS     WTHER INCON LIGHT EVAPO INFIL PHOTO HYDRO NSWIT MESOM MESEV MESOL",
    "@N MANAGEMENT  PLANT IRRIG FERTI RESID HARVS",
    "@N OUTPUTS     FNAME OVVEW SUMRY FROPT GROUT CAOUT WAOUT NIOUT MIOUT DIOUT VBOSE CHOUT OPOUT FMOPT",
)
_H_AUTO = (
    "@N PLANTING    PFRST PLAST PH2OL PH2OU PH2OD PSTMX PSTMN",
    "@N IRRIGATION  IMDEP ITHRL ITHRU IROFF IMETH IRAMT IREFF",
    "@N NITROGEN    NMDEP NMTHR NAMNT NCODE NAOFF",
    "@N RESIDUES    RIPCN RTIME RIDEP",
    "@N HARVEST     HFRST HLAST HPCNP HPCNR",
)
#: DSSAT fertilizer material of each RZWQM2 N form (DSSAT-CSM 4.8.6 FERCH048.SDA): FE008 calcium
#: nitrate (100 % NO3-N), FE002 ammonium sulfate (100 % NH4-N), FE005 urea (100 % urea-N)
FERT_MATERIAL: dict[str, str] = {
    "fertilizer_no3": "FE008",
    "fertilizer_nh4": "FE002",
    "fertilizer_urea": "FE005",
}
#: DSSAT application method of each RZWQM2 fertilizer method (DETAIL.CDE: AP001 broadcast, not
#: incorporated; Fert_Place.for puts it all in the top layer)
FERT_METHOD: dict[str, str] = {"broadcast-surface": "AP001"}
#: RZWQM2 random-number seed placeholder for DSSAT's weather generator (not used: WTHER = M)
_RSEED = 2150
#: harvest percentage of the product [%] (RZWQM2 harvest efficiency 1 = 100 %)
_HPC = 100.0


def fertilizer_levels(
    dat: RzwqmDat, seasons: SeasonTable
) -> list[list[tuple[np.datetime64, str, str, float]]]:
    """Per season the fertilizer applications of ``rzwqm.dat`` between the previous harvest (or
    the season's sowing year start) and its harvest: ``(date, FMCD, FACD, kg N ha-1)``
    (:data:`FERT_MATERIAL`, :data:`FERT_METHOD`; other forms or methods raise)."""
    start = np.datetime64(f"{pd.Timestamp(seasons.sow[0]).year}-01-01", "D")
    df = read_management(dat, start, seasons.harvest[-1])
    fert: Any = df[df["event"].str.startswith("fertilizer")]
    out: list[list[tuple[np.datetime64, str, str, float]]] = []
    for k in range(seasons.n_season):
        lo = seasons.harvest[k - 1] if k else start - np.timedelta64(1, "D")
        rows = []
        for r in fert.to_dict("records"):
            d = _day(r["date"])
            if not (lo < d <= seasons.harvest[k]):
                continue
            method = dict(kv.split("=", 1) for kv in str(r["detail"]).split(";") if "=" in kv).get(
                "method", ""
            )
            if r["event"] not in FERT_MATERIAL or method not in FERT_METHOD:
                raise NotImplementedError(f"fertilizer {r['event']} method {method!r} on {d} not converted")
            if d < seasons.sow[k]:
                raise NotImplementedError(
                    f"fertilizer on {d} before the sowing on {seasons.sow[k]} (run starts at sowing)"
                )
            rows.append((d, FERT_MATERIAL[r["event"]], FERT_METHOD[method], float(r["value"])))
        out.append(rows)
    return out


def treatment_numbers(n_season: int) -> dict[tuple[int, str], int]:
    """``(season index, NITRO switch) -> treatment number`` of :func:`write_filex`."""
    return {(k, nit): j * n_season + k + 1 for j, (nit, _) in enumerate(N_OPTIONS) for k in range(n_season)}


def write_filex(
    path: str | Path,
    seasons: SeasonTable,
    ics: Sequence[pd.DataFrame],
    fertilizers: Sequence[Sequence[tuple[np.datetime64, str, str, float]]],
    *,
    cultivar: tuple[str, str] = GENOTYPE_SETS["catpa"],
    soil_depth_cm: float = float(DSSAT_LAYER_BOTTOMS_CM[-1]),
) -> Path:
    """The experiment file: per season (and per :data:`N_OPTIONS`) one treatment starting on the
    sowing day, planting and harvest from ``seasons``, initial conditions ``ics[k]``
    (:func:`initial_conditions`), fertilizer ``fertilizers[k]`` (N on only)."""
    n = seasons.n_season
    if len(ics) != n or len(fertilizers) != n:
        raise ValueError("one initial-condition table and one fertilizer list per season")
    trt = treatment_numbers(n)
    L: list[str] = [f"*EXP.DETAILS: {CASE_FILEX[:8]}MZ CA-TPA MAIZE 2015-2021 FROM RZWQM2", "", "*TREATMENTS"]
    L.append(_H_TRT)
    for nit, label in N_OPTIONS:
        for k in range(n):
            year = pd.Timestamp(seasons.sow[k]).year
            mf = k + 1 if (nit == "Y" and fertilizers[k]) else 0
            L.append(
                _fx_line(
                    _H_TRT,
                    [
                        trt[(k, nit)],
                        1,
                        0,
                        0,
                        f"{year} {label}"[:25],
                        1,
                        1,
                        0,
                        k + 1,
                        k + 1,
                        0,
                        mf,
                        0,
                        0,
                        0,
                        0,
                        k + 1,
                        trt[(k, nit)],
                    ],
                )
            )
    L += ["", "*CULTIVARS", _H_CUL, _fx_line(_H_CUL, [1, "MZ", cultivar[0], cultivar[1]])]
    L += ["", "*FIELDS", _H_FLD]
    L.append(
        _fx_line(
            _H_FLD,
            [
                1,
                "CTPA0001",
                CASE_INSI,
                -99,
                0,
                "DR000",
                0,
                0,
                "00000",
                "LS",
                int(soil_depth_cm),
                CASE_SOIL_ID,
                "CA-TPA",
            ],
        )
    )
    L += ["", "*INITIAL CONDITIONS"]
    for k, ic in enumerate(ics):
        L.append(_H_IC)
        L.append(
            _fx_line(
                _H_IC,
                [k + 1, "MZ", _yyddd(seasons.sow[k]), 0, 0, 1, 1, -99, 0, 0, 0, 0, 0, "RZWQM2 LAYER.PLT"],
            )
        )
        L.append(_H_ICL)
        for _, r in ic.iterrows():
            L.append(_fx_line(_H_ICL, [k + 1, r["ICBL"], r["SH2O"], r["SNH4"], r["SNO3"]]))
    L += ["", "*PLANTING DETAILS", _H_PL]
    for k in range(n):
        pop = float(seasons.pltpop[k])
        L.append(
            _fx_line(
                _H_PL,
                [
                    k + 1,
                    _yyddd(seasons.sow[k]),
                    -99,
                    pop,
                    pop,
                    "S",
                    "R",
                    float(seasons.rowspc[k]),
                    0,
                    float(seasons.sdepth[k]),
                    -99,
                    -99,
                    -99,
                    -99,
                    0,
                    "-99",
                ],
            )
        )
    if any(fertilizers):
        L += ["", "*FERTILIZERS (INORGANIC)", _H_FE]
        for k, rows in enumerate(fertilizers):
            for d, fmcd, facd, kg in rows:
                L.append(_fx_line(_H_FE, [k + 1, _yyddd(d), fmcd, facd, 0, kg, 0, 0, 0, 0, "-99", "-99"]))
    L += ["", "*HARVEST DETAILS", _H_HA]
    for k in range(n):
        L.append(_fx_line(_H_HA, [k + 1, _yyddd(seasons.harvest[k]), "GS000", "-99", "-99", _HPC, 0, "-99"]))
    L += ["", "*SIMULATION CONTROLS"]
    for nit, label in N_OPTIONS:
        for k in range(n):
            s = trt[(k, nit)]
            sow = pd.Timestamp(seasons.sow[k])
            yl = _yyddd(np.datetime64(f"{sow.year + 1}-{sow.month:02d}-{sow.day:02d}", "D"))
            vals = (
                [s, "GE", 1, 1, "S", _yyddd(seasons.sow[k]), _RSEED, f"CA-TPA {sow.year} {label}"[:25]],
                [s, "OP", "Y", nit, "N", "N", "N", "N", "N", "N", "W"],
                [s, "ME", "M", "M", "E", "R", "S", "R", "R", 1, "G", "R", 2],
                [s, "MA", "R", "N", "R", "N", "R"],
                [
                    s,
                    "OU",
                    "N",
                    "Y",
                    "Y",
                    1,
                    "Y",
                    "N",
                    "Y",
                    "Y" if nit == "Y" else "N",
                    "N",
                    "N",
                    "Y",
                    "N",
                    "N",
                    "A",
                ],
            )
            auto = (
                [s, "PL", _yyddd(seasons.sow[k]), _yyddd(seasons.sow[k]), 40, 100, 30, 40, 10],
                [s, "IR", 30, 50, 100, "GS000", "IR001", 10, 1],
                [s, "NI", 30, 50, 25, "FE001", "GS000"],
                [s, "RE", 100, 1, 20],
                [s, "HA", 0, yl, 100, 0],
            )
            for h, v in zip(_H_SIM, vals, strict=True):
                L += [h, _fx_line(h, v)]
            L.append("")
            L.append("@  AUTOMATIC MANAGEMENT")
            for h, v in zip(_H_AUTO, auto, strict=True):
                L += [h, _fx_line(h, v)]
            L.append("")
    p = Path(path)
    p.write_text("\n".join(L) + "\n", encoding="latin-1")
    return p


# ============================================================================ the case
@dataclass(frozen=True)
class CaseFiles:
    """What :func:`build_case` wrote."""

    case_dir: Path
    genotype_set: str
    filex: Path
    sol: Path
    weather: tuple[Path, ...]
    genotype: tuple[Path, ...]
    seasons: SeasonTable
    soil: SoilProfile
    ics: tuple[pd.DataFrame, ...]
    fertilizers: tuple[tuple[tuple[np.datetime64, str, str, float], ...], ...]


def build_case(
    case_dir: str | Path,
    *,
    paths: CatpaPaths,
    organic: OrganicProfile,
    layer_plt: Any,
    genotype48: str | Path,
    genotype_set: str = "catpa",
    start: str = "2015-01-01",
    end: str = "2021-12-31",
) -> CaseFiles:
    """Write the CA-TPA DSSAT case of the seasons sown in ``[start, end]`` into ``case_dir``
    (module docstring): FileX, ``CT.SOL``, the ``.WTH`` files of the season years, and for
    ``genotype_set == "catpa"`` the genotype files. ``layer_plt``: the base run's ``LAYER.PLT``
    (:func:`agrijax.io.rzwqm.layers.read_layer_output`); ``genotype48``: the 4.8.6 ``Genotype``
    directory."""
    if genotype_set not in GENOTYPE_SETS:
        raise ValueError(f"genotype_set {genotype_set!r} not in {sorted(GENOTYPE_SETS)}")
    d = Path(case_dir)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    dat = read_rzwqm_dat(paths["rzwqm.dat"])
    rzx = read_rzx(paths["MZDSSAT.RZX"])
    seasons = season_table(dat, start, end)
    soil = catpa_soil_profile(dat, rzx, organic)
    sol = write_sol([soil], d / f"{CASE_SOIL_ID[:2]}.SOL", title="*SOILS: CA-TPA from RZWQM2")
    ics = tuple(initial_conditions(layer_plt, s - np.timedelta64(1, "D")) for s in seasons.sow)
    ferts = fertilizer_levels(dat, seasons)
    first = np.datetime64(f"{pd.Timestamp(seasons.sow[0]).year}-01-01", "D")
    last = np.datetime64(f"{pd.Timestamp(seasons.harvest[-1]).year}-12-31", "D")
    weather = catpa_weather(paths, dat, first, last)
    wfiles = write_weather_files(weather, weather_site(dat, weather), d)
    geno: list[Path] = []
    if genotype_set == "catpa":
        geno = write_catpa_genotype(paths, genotype48, d)
    fx = write_filex(d / CASE_FILEX, seasons, ics, ferts, cultivar=GENOTYPE_SETS[genotype_set])
    return CaseFiles(
        case_dir=d,
        genotype_set=genotype_set,
        filex=fx,
        sol=sol,
        weather=tuple(wfiles),
        genotype=tuple(geno),
        seasons=seasons,
        soil=soil,
        ics=ics,
        fertilizers=tuple(tuple(r) for r in ferts),
    )


def run_case(
    case: CaseFiles,
    out_dir: str | Path,
    *,
    run_root: str | Path | None = None,
    engine: str | Path | None = None,
) -> Any:
    """Run ``dscsm048 MZCER048 A`` on the case (:func:`agrijax.port.run_fortran.run_dscsm`, every
    output file kept in ``out_dir``); returns its :class:`DscsmResult`."""
    from agrijax.port.run_fortran import run_dscsm  # lazy: only the reference run needs port

    return run_dscsm(
        case.case_dir,
        out_dir,
        model="MZCER048",
        experiment_file=CASE_FILEX,
        weather_dir=case.case_dir,
        soil_dir=case.case_dir,
        keep_files=("*.OUT",),
        run_root=run_root,
        engine=engine,
    )


# ============================================================================ outputs
#: ``OVERVIEW.OUT`` stage names -> ``STGDOY`` index of CERES-Maize (the stage that ends that day)
OVERVIEW_STAGE_INDEX: dict[str, int] = {
    "Sowing": 7,
    "Germinate": 8,
    "Emergence": 9,
    "End Juveni": 1,
    "Floral Ini": 2,
    "75% Silkin": 3,
    "Beg Gr Fil": 4,
    "End Gr Fil": 5,
    "Maturity": 6,
}
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")
    )
}
_RUN_RE = re.compile(r"^\*RUN\s+(\d+)")
_TRT_RE = re.compile(r"^\s*TREATMENT\s*(\d+)")
_START_RE = re.compile(r"^\s*STARTING DATE\s*:\s*([A-Z]{3})\s+(\d+)\s+(\d{4})")
_STAGE_RE = re.compile(r"^\s*(\d{1,2}) ([A-Z]{3})\s+(-?\d+) (.{10})")
_SOIL_RE = re.compile(r"^\s*(\d+)-\s*(\d+)((?:\s+-?\d*\.\d+){11})\s*$")
OVERVIEW_SOIL_COLUMNS = (
    "LL",
    "DUL",
    "SAT",
    "EXTR_SW",
    "INIT_SW",
    "ROOT_DIST",
    "BD",
    "PH",
    "NO3",
    "NH4",
    "ORG_C",
)


def _overview_runs(path: str | Path) -> list[tuple[int, list[str]]]:
    lines = Path(path).read_bytes().decode("latin-1").splitlines()
    runs: list[tuple[int, list[str]]] = []
    for ln in lines:
        m = _RUN_RE.match(ln)
        if m:
            runs.append((int(m.group(1)), []))
        if runs:
            runs[-1][1].append(ln)
    return runs


def read_overview_stages(path: str | Path) -> pd.DataFrame:
    """The stage table of every run of an ``OVERVIEW.OUT`` (``SIMULATED CROP AND SOIL STATUS AT
    MAIN DEVELOPMENT STAGES``): ``RUN TRNO STAGE STGDOY_INDEX DATE``; the year of a ``DD MON`` date
    is the starting year's, plus one for dates before the start (a season crossing a year end)."""
    rows = []
    for run, lines in _overview_runs(path):
        trno, start = None, None
        in_tab = False
        for ln in lines:
            if trno is None and (m := _TRT_RE.match(ln)):
                trno = int(m.group(1))
            if start is None and (m := _START_RE.match(ln)):
                start = date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
            if "RSTG" in ln and "STAGE" in ln:
                in_tab = True
                continue
            if in_tab:
                if not ln.strip():
                    if rows and rows[-1]["RUN"] == run:
                        break
                    continue
                m = _STAGE_RE.match(ln)
                if m is None:
                    continue
                name = m.group(4).strip()
                if start is None:
                    raise ValueError(f"{path}: run {run} has no STARTING DATE line")
                d = date(start.year, _MONTHS[m.group(2)], int(m.group(1)))
                if d < start:
                    d = date(start.year + 1, d.month, d.day)
                rows.append(
                    {
                        "RUN": run,
                        "TRNO": trno,
                        "STAGE": name,
                        "STGDOY_INDEX": OVERVIEW_STAGE_INDEX.get(name, -1),
                        "DATE": d,
                    }
                )
    return pd.DataFrame(rows, columns=["RUN", "TRNO", "STAGE", "STGDOY_INDEX", "DATE"])


def read_overview_soil(path: str | Path) -> pd.DataFrame:
    """The soil table of every run of an ``OVERVIEW.OUT`` (``SUMMARY OF SOIL AND GENETIC INPUT
    PARAMETERS``, what dscsm048 read): ``RUN TOP BOTTOM`` + :data:`OVERVIEW_SOIL_COLUMNS`."""
    rows = []
    for run, lines in _overview_runs(path):
        for ln in lines:
            m = _SOIL_RE.match(ln)
            if m is None:
                continue
            vals = [float(t) for t in m.group(3).split()]
            rows.append(
                {
                    "RUN": run,
                    "TOP": float(m.group(1)),
                    "BOTTOM": float(m.group(2)),
                    **dict(zip(OVERVIEW_SOIL_COLUMNS, vals, strict=True)),
                }
            )
    return pd.DataFrame(rows)


_CUL_ECHO_RE = re.compile(r"\b(P1|P2|P5|G2|G3|PHINT)\s*:\s*(-?\d*\.?\d+)")


def read_overview_cultivar(path: str | Path) -> pd.DataFrame:
    """The cultivar echo of every run of an ``OVERVIEW.OUT`` (``P1 P2 P5 G2 G3 PHINT`` as dscsm048
    read them): ``RUN`` + those columns."""
    rows = []
    for run, lines in _overview_runs(path):
        vals: dict[str, float] = {}
        for ln in lines:
            for m in _CUL_ECHO_RE.finditer(ln):
                vals.setdefault(m.group(1), float(m.group(2)))
        rows.append({"RUN": run, **vals})
    return pd.DataFrame(rows)


def _yrdoy(d: Any) -> int:
    t = pd.Timestamp(d)
    return int(t.year * 1000 + t.dayofyear)


def season_results(out_dir: str | Path, seasons: SeasonTable) -> list[dict[str, Any]]:
    """Per treatment of :func:`write_filex`: stage dates (``STGDOY`` index -> ``YYYYDDD``), LAI max
    (``PlantGro`` ``LAID``), biomass at harvest [g m-2] (``CWAD`` / 10 on the harvest day), yield
    [kg ha-1] (``Summary`` ``HWAM``), the seasonal means of the water and N stress factors
    (``WSPD`` = 1 - SWFAC, ``WSGD`` = 1 - TURFAC, ``NSTD`` = 1 - NSTRES) and the season's
    ``Summary`` rain / ET / drainage [mm]."""
    from agrijax.io.dssat.outputs import read_plantgro, read_summary

    out = Path(out_dir)
    summ: Any = read_summary(out / "Summary.OUT")
    pg: Any = read_plantgro(out / "PlantGro.OUT")
    st: Any = read_overview_stages(out / "OVERVIEW.OUT")
    rows = []
    for (k, nit), tr in sorted(treatment_numbers(seasons.n_season).items(), key=lambda x: x[1]):
        s = summ[summ["TRNO"] == tr]
        if len(s) != 1:
            raise ValueError(f"treatment {tr}: {len(s)} Summary rows")
        s0 = s.iloc[0]
        g = pg[pg["TRNO"] == tr]
        stages = st[st["TRNO"] == tr]
        stg = {
            int(r["STGDOY_INDEX"]): _yrdoy(r["DATE"]) for _, r in stages.iterrows() if r["STGDOY_INDEX"] > 0
        }
        on_h = g[pd.to_datetime(g["DATE"]) == pd.Timestamp(seasons.harvest[k])]
        last = on_h.iloc[0] if len(on_h) else g.iloc[-1]

        def mean(c: str, g: Any = g) -> float | None:
            return float(g[c].mean()) if c in g.columns else None

        rows.append(
            {
                "trno": tr,
                "season": int(pd.Timestamp(seasons.sow[k]).year),
                "nitro": nit,
                "sow": _yrdoy(seasons.sow[k]),
                "harvest": _yrdoy(seasons.harvest[k]),
                "pdat": int(s0["PDAT"]),
                "hdat": int(s0["HDAT"]),
                "stgdoy": {str(i): stg[i] for i in sorted(stg)},
                "lai_max": float(g["LAID"].max()),
                "biomass_harvest_g_m2": float(last["CWAD"]) / _MM_PER_CM,
                "yield_kg_ha": float(s0["HWAM"]),
                "cwam_kg_ha": float(s0["CWAM"]),
                "mean_wspd": mean("WSPD"),
                "mean_wsgd": mean("WSGD"),
                "mean_nstd": mean("NSTD"),
                "prcm_mm": float(s0["PRCM"]) if "PRCM" in s0 else None,
                "etcm_mm": float(s0["ETCM"]) if "ETCM" in s0 else None,
                "epcm_mm": float(s0["EPCM"]) if "EPCM" in s0 else None,
                "drcm_mm": float(s0["DRCM"]) if "DRCM" in s0 else None,
            }
        )
    return rows
