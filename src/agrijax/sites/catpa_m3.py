"""CA-TPA inputs: the 2015-2023 forcing pytree, the season table and the CERES-Maize parameters of
the crop of the RZWQM2 scenario.

Everything is built from the scenario's own input files (``rzwqm.dat``, ``IPNAMES.DAT``,
``CA-TPA.MET``, ``CA-TPA.BRK``, the embedded crop's DSSAT 4.0 ``MZCER040.CUL`` of the project and
``MZCER040.{ECO,SPE}`` of the scenario's DSSAT database, :class:`CatpaPaths`) plus two things the
input files do not hold:

* the soil the embedded crop sees (DSSAT ``SOILPROP``: ``DLAYR``, ``LL``, ``DUL``, ``SAT``,
  ``WR`` -> ``SHF``, ``SLPF``), which RZWQM2 derives inside ``DSSATDRV``; it is taken from the
  ``DSSATDRV`` table of the instrumented reference run (:func:`ceres_soil_from_dssatdrv`);
* the two ecotype values the 4.0 ecotype file lacks (``TSEN``, ``CDAY``), taken from the
  DSSAT-CSM v4.8.6.0 ``MZCER048.ECO`` row of the same ecotype (:data:`CERES_SUBSTITUTIONS`).

Forcing pytree of :func:`catpa_m3_inputs` (``[T]`` leading axis, units of the coupling
contract (port P8) and of the record classes):

=====================  ===============================================  ======================
key                    record                                           source
=====================  ===============================================  ======================
``weather``            :class:`~agrijax.iface.surface.DailyWeather`     ``.MET`` after the daily
                       (``srad`` and ``srad_horizontal``, MJ m-2 d-1)   preparation
``events``             :class:`~agrijax.core.events.EventTable`         ``rzwqm.dat`` management
                       (``sow`` / ``harvest`` flags, ``irrig_cm``)
``crop``               :class:`~agrijax.processes.crop.ceres_maize.     weather + ``DAYLEN`` /
                       state.CeresForcing` (replay fields zero)         ``TWILIGHT``; CO2
=====================  ===============================================  ======================

The weather stays in the forcing preprocessing, outside the differentiable processes (NumPy, not
differentiable). Snow on the ground for the crop (``CeresForcing.snow``) is zero here: in a coupled
day the crop reads the snow port P9. The breakpoint storms of the ``.BRK`` file are in
:attr:`CatpaM3Inputs.storms` (:func:`~agrijax.io.rzwqm.storms.storm_arrays`).

The season table (:func:`season_table`) holds the seven CA-TPA maize seasons 2015-2021 of the
``PLANT MANAGEMENT`` block: sowing and (fixed-date) harvest days, and the per-season planting
values the crop reads (``PLTPOP``, ``SDEPTH``, ``ROWSPC``, ``YRPLT``). Validated against ``MANAGE.OUT`` of the
2015-2023 base run and the ``DSSATDRV`` / ``MZ_GROSUB`` tables of the instrumented run
(``tests/integration/test_io_m3_catpa.py``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from agrijax.io.dssat.genotype import read_cul, read_eco, read_spe
from agrijax.io.rzwqm.dat import Planting, RzwqmDat, read_rzwqm_dat
from agrijax.io.rzwqm.events import event_table_from_frame, read_management
from agrijax.io.rzwqm.met import _R2D, prepare_rzwqm_forcing, read_brk, read_met
from agrijax.io.rzwqm.storms import (
    StormArrays,
    irrigation_cm,
    read_met_modifiers,
    storm_arrays,
)
from agrijax.port.run_fortran import DEFAULT_DSSAT_ENGINE

__all__ = [
    "CATPA_VARNO",
    "CERES_SPECIES_KEYS",
    "CERES_SUBSTITUTIONS",
    "CatpaM3Inputs",
    "CatpaPaths",
    "SeasonTable",
    "Substitution",
    "catpa_m3_inputs",
    "ceres_forcing",
    "ceres_maize_params",
    "ceres_soil_from_dssatdrv",
    "daily_weather",
    "event_table",
    "latitude_deg",
    "season_table",
]

#: the CA-TPA cultivar (``MANAGE.OUT``: "CROP PLANTED: MAIZE IB0012 PIO 3382"; ``DSSATDRV``
#: ``VARNOR`` of the reference run)
CATPA_VARNO: str = "IB0012"
#: unit conversion ha -> m2 (seeds ha-1 -> plants m-2 of the planting density)
M2_PER_HA: float = 1.0e4
#: RZWQM2 harvest option 3 = fixed date (``rzwqm.dat`` PLANT MANAGEMENT record 2, item 1)
_HARVEST_FIXED_DATE = 3
#: 0-based ``METMOD`` row of the CO2 modifier [percent] of ``IPNAMES.DAT``
_CO2_MODIFIER_ROW = 7
_PERCENT_TO_FRACTION = 1.0e-2

#: 4.0 ``.SPE`` names read into :class:`CeresSpecies` fields of the same (lower-case) name
CERES_SPECIES_KEYS: tuple[str, ...] = (
    "PRFTC",
    "RGFIL",
    "PARSR",
    "CO2X",
    "CO2Y",
    "FSLFW",
    "FSLFN",
    "RSGR",
    "RSGRT",
    "CARBOT",
    "DSGT",
    "DGET",
    "SWCG",
    "STMWTE",
    "RTWTE",
    "LFWTE",
    "SEEDRVE",
    "LEAFNOE",
    "PLAE",
    "RLWR",
    "RWUEP1",
)
#: 4.0 ``.ECO`` columns read into :class:`CeresCultivar` fields of the same (lower-case) name
_ECO_KEYS: tuple[str, ...] = ("TBASE", "TOPT", "ROPT", "DJTI", "GDDE", "DSGFT", "RUE")
_CUL_KEYS: tuple[str, ...] = ("P1", "P2", "P5", "G2", "G3", "PHINT")


@dataclass(frozen=True)
class Substitution:
    """A crop parameter not taken from the embedded crop's own 4.0 files, and why."""

    name: str
    source: str
    reason: str


CERES_SUBSTITUTIONS: tuple[Substitution, ...] = (
    Substitution(
        "cultivar.tsen",
        "DSSAT-CSM v4.8.6.0 Data/Genotype/MZCER048.ECO, column TSEN of the cultivar's ecotype row",
        "the RZWQM2-embedded DSSAT 4.0 MZCER040.ECO has no TSEN column; the 4.8.6 kernels read it",
    ),
    Substitution(
        "cultivar.cday",
        "DSSAT-CSM v4.8.6.0 Data/Genotype/MZCER048.ECO, column CDAY of the cultivar's ecotype row",
        "as tsen: no CDAY column in MZCER040.ECO",
    ),
    Substitution(
        "cultivar.p2o",
        "MZCER040.ECO column 'P20' (header spelling of the file; DSSAT reads the ecotype record by "
        "position as P2O)",
        "name only: the value is the 4.0 file's",
    ),
    Substitution(
        "species.canht_pot",
        "DSSAT-CSM v4.8.6.0 MZ_GROSUB.for:665 (CANHT_POT = 1.6, SEASINIT; "
        "processes/crop/ceres_maize/coefficients.py)",
        "a 4.8.6 SEASINIT constant with no 4.0 species-file entry",
    ),
    Substitution(
        "species.bsgdd",
        "DSSAT-CSM v4.8.6.0 MZ_GROSUB.for:656 (BSGDD = 250.0, SEASINIT; "
        "processes/crop/ceres_maize/coefficients.py)",
        "a 4.8.6 SEASINIT constant with no 4.0 species-file entry",
    ),
    Substitution(
        "species.pormin, species.rwumx",
        "MZCER040.SPE entries PORM and RWMX",
        "name only: the 4.0 file's names of PORMIN and RWUMX",
    ),
    Substitution(
        "soil.*",
        "DSSATDRV SOILPROP (DLAYR, LL, DUL, SAT, WR -> SHF, SLPF) of the instrumented RZWQM2 4.6 "
        "run (ceres_soil_from_dssatdrv)",
        "RZWQM2 derives the crop's layer soil inside DSSATDRV; not rebuilt from rzwqm.dat here",
    ),
)
"""Every CA-TPA CERES-Maize parameter that does not come from the embedded crop's 4.0 files as
written (the 4.0 species file differs from 4.8.6 in ``RGFIL`` and the CO2 table: the 4.0 values
are used, since the reference crop reads them)."""


# ============================================================================ paths
@dataclass(frozen=True)
class CatpaPaths:
    """Input files of the CA-TPA reference scenario under a data root (``$AGRI_JAX_DATA``).

    ``species_db`` is the DSSAT database directory of the scenario (``MZDSSAT.RZX``, "DATABASE
    FILE LOCATIONS", first line; ``Scenario/DSSAT`` in the mirror): at the start of a run RZWQM2
    copies the crop's ``.SPE`` and ``.ECO`` from there into the project directory
    (RZWQM2 4.5 readrzx.for:44-68), so the copies in the scenario root are not what the crop
    reads (CA-TPA's root ``MZCER040.SPE`` has ``RSGR = 0.1``, ``PORM = 0.05``; the database file
    and the reference's ``MZ_GROSUB`` have 0.9 and 0.01). The cultivar file is the project's own
    (the second location, the scenario root).
    """

    scenario: Path
    eco48: Path

    @property
    def species_db(self) -> Path:
        """The DSSAT species / ecotype database directory of the scenario."""
        return self.scenario / "DSSAT"

    @classmethod
    def under(cls, data_dir: str | Path | None = None, dssat_engine: str | Path | None = None) -> CatpaPaths:
        """``<data>/narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario`` and the 4.8.6 ecotype file of
        ``dssat_engine`` (default ``$AGRI_JAX_DSSAT``)."""
        root = Path(data_dir or os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")).expanduser()
        eng = Path(dssat_engine or os.environ.get("AGRI_JAX_DSSAT", str(DEFAULT_DSSAT_ENGINE))).expanduser()
        return cls(
            scenario=root / "narval_mirror" / "RZWQM_sw_batch" / "CA-TPA" / "Scenario",
            eco48=eng / "source" / "Data" / "Genotype" / "MZCER048.ECO",
        )

    def __getitem__(self, name: str) -> Path:
        return self.scenario / name


# ============================================================================ seasons and events
def _as_day(x: Any) -> Any:
    """``x`` (str, date or datetime64) as a ``datetime64[D]`` scalar."""
    return np.asarray(x, dtype="datetime64[D]")[()]


def _yrdoy(d: np.datetime64) -> int:
    t = pd.Timestamp(d)
    return int(t.year * 1000 + t.dayofyear)


@dataclass(frozen=True)
class SeasonTable:
    """The crop seasons of a run window, one entry per sowing (``[n_season]`` numpy arrays).

    ``sow`` / ``harvest`` are ``datetime64[D]``; ``yrplt`` and ``harvest_yrdoy`` the same days as
    ``YYYYDDD``; ``pltpop`` [plants m-2], ``sdepth`` [cm] and ``rowspc`` [cm] what the crop reads.
    """

    sow: np.ndarray
    harvest: np.ndarray
    yrplt: np.ndarray
    harvest_yrdoy: np.ndarray
    pltpop: np.ndarray
    sdepth: np.ndarray
    rowspc: np.ndarray

    @property
    def n_season(self) -> int:
        """Number of seasons."""
        return len(self.sow)

    def season_index(self, days: Sequence[np.datetime64] | np.ndarray) -> np.ndarray:
        """Index of the latest season sown on or before each day (``int32 [T]``, -1 before the first)."""
        d = np.asarray(days, dtype="datetime64[D]")
        return (np.searchsorted(self.sow, d, side="right") - 1).astype(np.int32)

    def in_crop(self, days: Sequence[np.datetime64] | np.ndarray) -> np.ndarray:
        """True on the days from a sowing to its harvest, both included (``bool [T]``)."""
        d = np.asarray(days, dtype="datetime64[D]")
        k = self.season_index(d)
        kk = np.maximum(k, 0)
        return (k >= 0) & (d <= self.harvest[kk])


def season_table(
    plantings: RzwqmDat | Sequence[Planting],
    start: str | np.datetime64,
    end: str | np.datetime64,
) -> SeasonTable:
    """The seasons sown in ``[start, end]`` from the ``PLANT MANAGEMENT`` block (``rzwqm.dat`` or
    its :attr:`~agrijax.io.rzwqm.dat.RzwqmDat.plantings`). Only fixed-date harvests (option 3)
    are supported; the planting depth is record 1 item 5 in cm (equal to the crop's ``SDEPTH`` on
    every CA-TPA season, ``DSSATDRV`` table), the density seeds ha-1 over :data:`M2_PER_HA`."""
    pl = plantings.plantings if isinstance(plantings, RzwqmDat) else list(plantings)
    lo, hi = _as_day(start), _as_day(end)
    sel = sorted((p for p in pl if lo <= p.planting_date <= hi), key=lambda p: p.planting_date)
    for p in sel:
        if p.harvest_option != _HARVEST_FIXED_DATE or p.harvest_date is None:
            raise NotImplementedError(
                f"planting of {p.planting_date} (line {p.line_no}): harvest option {p.harvest_option}; "
                "only fixed-date harvests (option 3) are supported"
            )
    sow = np.array([p.planting_date for p in sel], dtype="datetime64[D]")
    harvest = np.array([p.harvest_date for p in sel], dtype="datetime64[D]")
    if np.any(harvest < sow) or np.any(sow[1:] <= harvest[:-1]):
        raise ValueError("seasons overlap or harvest precedes sowing")
    return SeasonTable(
        sow=sow,
        harvest=harvest,
        yrplt=np.array([_yrdoy(d) for d in sow], dtype=np.int64),
        harvest_yrdoy=np.array([_yrdoy(d) for d in harvest], dtype=np.int64),
        pltpop=np.array([p.density_seeds_ha / M2_PER_HA for p in sel], dtype=float),
        sdepth=np.array([float(p.planting_depth_layer) for p in sel], dtype=float),
        rowspc=np.array([p.row_spacing_cm for p in sel], dtype=float),
    )


def event_table(dat: RzwqmDat, days: Sequence[np.datetime64] | np.ndarray) -> Any:
    """The :class:`~agrijax.core.events.EventTable` of ``days``: the management of ``rzwqm.dat``
    (:func:`~agrijax.io.rzwqm.events.read_management`, the path validated against ``MANAGE.OUT``)
    with the per-day irrigation of :func:`~agrijax.io.rzwqm.storms.irrigation_cm` (zero at CA-TPA)."""
    d = np.asarray(days, dtype="datetime64[D]")
    irrigation_cm(dat, d)  # raises when the scenario has irrigation operations (not parsed)
    df = read_management(dat, d[0], d[-1])
    return event_table_from_frame(df, pd.DatetimeIndex(d))


# ============================================================================ weather
def latitude_deg(dat: RzwqmDat) -> float:
    """Site latitude [deg] as RZWQM2 passes it to the crop (``XLATR = XLAT * R2D``, Rzman.for:5862)."""
    return float(dat.physiography["latitude_rad"]) * _R2D


def daily_weather(
    met_path: str | Path,
    dat: RzwqmDat,
    days: Sequence[np.datetime64] | np.ndarray,
    *,
    met_modifiers: np.ndarray | None = None,
) -> tuple[Any, np.ndarray]:
    """``(DailyWeather, RTH)`` of ``days``: the ``.MET`` record after the daily preparation
    (:func:`~agrijax.io.rzwqm.met.prepare_rzwqm_forcing`: wind floor, the ``met_modifiers`` of
    ``IPNAMES.DAT``, bounds); ``srad`` and RTH [MJ m-2 d-1] are both the prepared ``.MET``
    radiation (no hourly disaggregation)."""
    from agrijax.iface.surface import DailyWeather

    met = prepare_rzwqm_forcing(read_met(met_path), met_modifiers=met_modifiers)
    d = np.asarray(days, dtype="datetime64[D]")
    idx = pd.DatetimeIndex(d)
    missing = idx.difference(pd.DatetimeIndex(met.index))
    if len(missing):
        first = str(missing[0])[:10]
        raise ValueError(f"{met_path}: no weather record for {len(missing)} days, first {first}")
    m = met.loc[idx]
    doy = np.array([t.dayofyear for t in idx], dtype=float)
    rth = m["srad_mj"].to_numpy(float)
    w = DailyWeather(
        tmin=jnp.asarray(m["tmin"].to_numpy(float)),
        tmax=jnp.asarray(m["tmax"].to_numpy(float)),
        srad=jnp.asarray(m["srad_mj"].to_numpy(float)),
        rh=jnp.asarray(m["rh"].to_numpy(float)),
        wind_run=jnp.asarray(m["wind_run_km"].to_numpy(float)),
        doy=jnp.asarray(doy),
    )
    return w, rth


def ceres_forcing(
    weather: Any,
    days: Sequence[np.datetime64] | np.ndarray,
    *,
    lat_deg: float,
    co2_ppm: float | np.ndarray,
    n_layer: int,
) -> Any:
    """The crop's :class:`CeresForcing` of ``days`` from ``weather`` (tmax, tmin, srad):
    ``DAYLEN`` / ``TWILIGHT`` daylengths at ``lat_deg``, ``co2_ppm`` (scalar or ``[T]``). The record is
    the weather only: the crop's snow and water come through the ports P9 and P1 in the coupled day
    (the replay fields live in ``CeresReplayForcing``). ``n_layer`` is accepted for the
    callers of the earlier record, which carried a ``[T, n_layer]`` replay ``sw``; it is not used."""
    from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength
    from agrijax.processes.crop.ceres_maize.state import CeresForcing

    d = np.asarray(days, dtype="datetime64[D]")
    n = len(d)
    yrdoy = np.array([_yrdoy(x) for x in d], dtype=np.int32)
    doy = weather.doy
    del n_layer  # the weather-only record has no layer axis
    return CeresForcing(
        yrdoy=jnp.asarray(yrdoy),
        tmax=weather.tmax,
        tmin=weather.tmin,
        srad=weather.srad,
        dayl=daylength(doy, lat_deg),
        twilen=twilight_daylength(doy, lat_deg),
        co2=jnp.broadcast_to(jnp.asarray(co2_ppm, dtype=weather.tmax.dtype), (n,)),
    )


# ============================================================================ CERES-Maize parameters
def ceres_soil_from_dssatdrv(values: Mapping[str, Any], row: int = 0) -> dict[str, np.ndarray]:
    """The crop's layer soil from a ``DSSATDRV`` table row (``SOILPROP%...`` fields, Fortran
    padded to ``NL``; cut to ``SOILPROP%NLAYR``): ``{dlayr, ll, dul, sat, shf, slpf}``."""
    nl = int(np.asarray(values["SOILPROP%NLAYR"])[row])

    def lay(k: str) -> np.ndarray:
        return np.asarray(values[f"SOILPROP%{k}"], dtype=float)[row, :nl]

    return {
        "dlayr": lay("DLAYR"),
        "ll": lay("LL"),
        "dul": lay("DUL"),
        "sat": lay("SAT"),
        "shf": lay("WR"),
        "slpf": np.asarray(np.asarray(values["SOILPROP%SLPF"], dtype=float)[row]),
    }


def ceres_maize_params(
    paths: CatpaPaths,
    soil: Mapping[str, Any],
    seasons: SeasonTable,
    season: int,
    *,
    varno: str = CATPA_VARNO,
) -> Any:
    """CERES-Maize parameters of season ``season`` (0-based) of ``seasons``: cultivar ``varno``
    from the project's ``MZCER040.CUL``, its ecotype from the database ``MZCER040.ECO`` (``TSEN``
    and ``CDAY`` from the 4.8.6 ``MZCER048.ECO``), the species from the database ``MZCER040.SPE``
    (:attr:`CatpaPaths.species_db`), the layer ``soil``
    (:func:`ceres_soil_from_dssatdrv`) and the season's planting values; the 4.8.6 coefficients.
    Every value not read from a 4.0 file as written is listed in :data:`CERES_SUBSTITUTIONS`."""
    from agrijax.processes.crop.ceres_maize.coefficients import BSGDD, CANHT_POT, DSSAT_COEFFICIENTS
    from agrijax.processes.crop.ceres_maize.state import (
        CeresCultivar,
        CeresMaizeParams,
        CeresSoil,
        CeresSpecies,
    )

    def a(x: Any) -> Any:
        return jnp.asarray(np.asarray(x, dtype=float))

    cul_tab = read_cul(paths["MZCER040.CUL"])
    if varno not in cul_tab.index:
        raise KeyError(f"cultivar {varno} not in {paths['MZCER040.CUL']}")
    cul = cul_tab.loc[varno]
    eco = read_eco(paths.species_db / "MZCER040.ECO").loc[cul["ECO#"]]
    eco48 = read_eco(paths.eco48).loc[cul["ECO#"]]
    spe = read_spe(paths.species_db / "MZCER040.SPE")
    cultivar = CeresCultivar(
        **{k.lower(): a(cul[k]) for k in _CUL_KEYS},
        **{k.lower(): a(eco[k]) for k in _ECO_KEYS},
        p2o=a(eco["P20"]),
        tsen=a(eco48["TSEN"]),
        cday=a(eco48["CDAY"]),
    )
    species = CeresSpecies(
        **{k.lower(): a(spe[k]) for k in CERES_SPECIES_KEYS},
        pormin=a(spe["PORM"]),
        rwumx=a(spe["RWMX"]),
        canht_pot=a(CANHT_POT),
        bsgdd=a(BSGDD),
    )
    k = int(season)
    return CeresMaizeParams(
        cultivar=cultivar,
        species=species,
        soil=CeresSoil(**{f: a(soil[f]) for f in ("dlayr", "ll", "dul", "sat", "shf", "slpf")}),
        pltpop=a(seasons.pltpop[k]),
        sdepth=a(seasons.sdepth[k]),
        rowspc=a(seasons.rowspc[k]),
        yrplt=jnp.asarray(int(seasons.yrplt[k]), dtype=jnp.int32),
        coefficients=DSSAT_COEFFICIENTS.as_arrays(),
    )


# ============================================================================ the whole run
@dataclass(frozen=True)
class CatpaM3Inputs:
    """Everything :func:`catpa_m3_inputs` builds (module docstring)."""

    days: np.ndarray
    forcing: dict[str, Any]
    storms: StormArrays
    seasons: SeasonTable
    crop_params: tuple[Any, ...]
    substitutions: tuple[Substitution, ...]
    lat_deg: float


def catpa_m3_inputs(
    soil: Mapping[str, Any],
    *,
    paths: CatpaPaths | None = None,
    start: str | np.datetime64 = "2015-01-01",
    end: str | np.datetime64 = "2023-12-31",
) -> CatpaM3Inputs:
    """The CA-TPA forcing pytree of ``[start, end]`` (default 2015-2023, 3287 days), the season
    table and one CERES-Maize parameter set per season. ``soil`` is the crop's layer soil
    (:func:`ceres_soil_from_dssatdrv`). Raises when the rainfall modifier of ``IPNAMES.DAT``
    changes with the month (not reproduced, :func:`storm_arrays`) or the scenario has irrigation
    operations (not parsed)."""
    p = paths or CatpaPaths.under()
    dat = read_rzwqm_dat(p["rzwqm.dat"])
    metmod = read_met_modifiers(p["IPNAMES.DAT"])
    days = np.arange(_as_day(start), _as_day(end) + np.timedelta64(1, "D"))
    weather, rth = daily_weather(p["CA-TPA.MET"], dat, days, met_modifiers=metmod)
    storms = storm_arrays(days, read_brk(p["CA-TPA.BRK"]), met_modifiers=metmod)
    seasons = season_table(dat, days[0], days[-1])
    lat = latitude_deg(dat)
    month = np.asarray(days, dtype="datetime64[M]").astype(np.int64) % 12
    co2 = float(dat.physiography["co2_ppm"]) * metmod[_CO2_MODIFIER_ROW, month] * _PERCENT_TO_FRACTION
    n_layer = len(np.asarray(soil["dlayr"]))
    forcing: dict[str, Any] = {
        "events": event_table(dat, days),
        "crop": ceres_forcing(weather, days, lat_deg=lat, co2_ppm=co2, n_layer=n_layer),
    }
    forcing["weather"] = weather.replace(srad_horizontal=jnp.asarray(rth))
    crop = tuple(ceres_maize_params(p, soil, seasons, k) for k in range(seasons.n_season))
    return CatpaM3Inputs(
        days=days,
        forcing=forcing,
        storms=storms,
        seasons=seasons,
        crop_params=crop,
        substitutions=CERES_SUBSTITUTIONS,
        lat_deg=lat,
    )
