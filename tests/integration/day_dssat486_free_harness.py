"""Harness of ``test_day_dssat486_free.py``: the DSSAT-CSM v4.8.6.0 day run free against
``dscsm048`` on the 58 maize treatments of M2 and the 7 nitrogen-off CA-TPA seasons.

Inputs of one run (``<EXP>_t<NN>``):

* the reference run itself (``dscsm048`` build486, nitrogen off, one-treatment batch:
  ``test_ceres_dssat.run_reference``; CA-TPA: treatment 1-7 of ``$AGRI_JAX_DATA/dssat_catpa/catpa``),
  which gives CERES-Maize's parameters (``DSSAT48.INP``, ``.ECO``, ``.SPE``), the crop weather
  (``Weather.OUT``) and the reference outputs (``PlantGro``, ``SoilWat``, ``ET``, ``Summary``);
* the soil and the soil water's inputs as ``dscsm048`` holds them, from the ``WATBAL`` tables
  (``validation/aj_dsw/d1a/tables/<key>/``): the ``SOILPROP`` of ``SEASINIT`` (the static soil), the
  initial ``SW``, snow and mulch water, the day's ``RAIN``, ``TMAX``, ``IRRAMT`` and the residue
  record (``MULCHMASS``, ``MULCHCOVER``, ``NEWMULCH``, ``MUL_WATFAC``: **replay** of the organic
  matter module) and, where ``SOILDYN`` changes them, the day's ``SOILPROP`` (**replay**);
* from the ``SPAM`` entry tables (``validation/aj_det/tables/<key>_spam_in.npz``): the
  weather record ``SPAM`` reads (``SRAD``, ``TMAX``, ``TMIN``, ``WINDSP``, ``CO2`` and the hourly
  mean ``TAVG`` of ``HMET``: **replay** of the weather module's derived value), ``SALB``, ``U``,
  ``MULCH_AM``, ``MUL_EXTFAC`` (**replay**, residue) and ``KSEVAP = KTRANS`` (CERES-Maize's KEP).

Nothing of the soil water, the evaporation, the transpiration or the uptake comes from the
reference: the crop forcing's ``sw``, ``eop`` and ``trwup`` are NaN (the crop reads P1 only).

Configurations (all batched with ``vmap`` over the runs of one layer count and ``MESEV``):

* ``free`` (the acceptance run): REAL*4 soil values (:func:`agrijax.models.day_dssat486.dssat_soil_values`),
  the ``SOILDYN`` replay on for the runs whose soil changes;
* ``input``: the soil values as the decimal inputs (the effect of the REAL*4 convention);
* ``static``: REAL*4, ``SOILDYN`` replay off (the effect of the soil-property replay).

Acceptance: the harvest-day grain weight within 2 % of ``HWAM`` (the reference's own yield
definition; four of the seven CA-TPA seasons are harvested before maturity); secondary: LAI and
biomass curves (RMSE), emergence / silking / maturity dates; and ET, soil water and the daily water
ledger. The per-run table goes to
``<data-dir>/validation/aj_dint/d2_1a_free_run.json``.

One-day checks against the SPAM dumps (every simulated day of the 58 treatments, the dumped REAL*4
inputs promoted to float64): ``XTRACT`` (the layer extraction ``-SWDELTX DLAYR``, ``EP``, ``TRWU``;
ROOTWU's ``RWU`` from the ``ROOTWU`` exit tables), the ``SOILDYN`` albedo ``MSALB`` and ``PETPT``
on it; tolerance: the REAL*4 rounding of the Fortran expressions (``spam_evap_dssat.OPS`` style).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_ceres_dssat as m2  # noqa: E402

from agrijax.core.runtime import run_sites  # noqa: E402
from agrijax.io.dssat import read_out, read_plantgro, read_soilwat, read_summary, read_wth  # noqa: E402
from agrijax.models.day_dssat486 import (  # noqa: E402
    SLOT,
    SOIL_EVAPORATION_KEYS,
    DssatSite,
    day_dssat486,
    day_outputs,
    day_params,
    day_processes,
    dssat_soil_values,
    initial_state,
    soil_evaporation_params_problems,
)
from agrijax.port import dumps  # noqa: E402
from agrijax.processes.crop.ceres_maize import CeresMaizeState  # noqa: E402
from agrijax.processes.pet.priestley_taylor import priestley_taylor  # noqa: E402
from agrijax.processes.pet.spam_dssat import SpamWeather  # noqa: E402
from agrijax.processes.soil_water.bucket import (  # noqa: E402
    BucketForcing,
    BucketSoil,
    BucketState,
    MulchForcing,
)
from agrijax.processes.soil_water.bucket_evap import (  # noqa: E402
    SoilEvapState,
    actual_transpiration,
    extraction_supply,
    soil_albedo,
    xtract,
)
from agrijax.processes.water_supply import RootwuState  # noqa: E402
from agrijax.sites.dssat_inputs import ceres_forcing, ceres_params  # noqa: E402

#: experiments whose daylength is replaced by the FileX environment modification (growth chamber)
DAYLENGTH_FROM_OUTPUT = {"GAGR0201"}
#: the ``WATBAL`` tables, the ``SPAM`` entry tables and the ``SPAM`` / ``ROOTWU`` exit tables under the
#: data directory
DSW = Path("validation/aj_dsw/d1a/tables")
DET = Path("validation/aj_det/tables")
A12 = Path("dumps/tables/dssat486")
CATPA_CASE = Path("dssat_catpa/catpa")
CATPA_FILEX = "CTPA1501.MZX"
REPORT_DIR = Path("validation/aj_dint")
#: acceptance: harvest-day grain weight within 2 % of HWAM
YIELD_REL = 0.02
#: the daily water ledger closes to this [mm]: the float64 rounding of the day's terms (measured on
#: rorqual: at most 1.8e-13 mm on every day of the 65 runs in every configuration)
LEDGER_ATOL_MM = 1e-11
#: runs of the free configuration whose yield is not within YIELD_REL, with the measured reason
#: (none: 65 of 65 within 2 %, the largest BRPI0202 t04 at 1.54 %)
KNOWN_YIELD_FAILURES: dict[str, str] = {}
#: runs whose emergence / silking / maturity date differs from the reference, in days (none)
KNOWN_DATE_DIFFS: dict[str, dict[str, int]] = {}
#: the runs the mixed precision (REAL*4 soil limits against a float64 water content, configuration
#: ``real4_limits``) moves out of the 2 % (measured): CA-TPA 2015-2020, 2.3 % to 20 %
MIXED_PRECISION_FAILURES: frozenset[str] = frozenset(f"CTPA1501_t{k:02d}" for k in range(1, 7))
#: the secondary acceptance metrics of the free configuration, per
#: group of runs (``"m2"``: the 58 M2 treatments, ``"catpa"``: the 7 CA-TPA seasons), each
#: ``(largest value over the group's runs, pinned bound)``: the bound is the measured value plus a
#: margin (two rorqual runs, the second on the merged main: the same values). LAI
#: [m2 m-2] and CWAD (relative to the season maximum) against PlantGro.OUT; ET.OUT [mm d-1, printed
#: F8.3 or F7.2]; SoilWat.OUT SWTD [mm, printed to 1 mm: half-step 0.5] and the printed layer SW
#: [cm3 cm-3, F8.3]; the end-of-day SW against the next day's SPAM entry dump (full precision). The
#: M2 treatments sit at the print resolution; the CA-TPA seasons 5 to 18 times above it in EO, EOS,
#: EP, ES and the end-of-day SW (their largest values: 2018 and 2020, t04 and t06).
SECONDARY_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "m2": {
        "lai_rmse": (0.00334, 0.005),
        "cwad_rmse_rel_to_max": (0.0034, 0.005),
        "et.EO.max_abs": (0.000543, 0.00075),
        "et.EOS.max_abs": (0.000634, 0.001),
        "et.EOP.max_abs": (0.000572, 0.001),
        "et.ES.max_abs": (0.000821, 0.0012),
        "et.EP.max_abs": (0.00435, 0.0065),
        "et.EM.max_abs": (0.0005, 0.00075),
        "soil_water.SWTD.max_abs": (0.5, 0.75),
        "soil_water.SW_layers_max_abs": (0.000576, 0.00075),
        "dump.sw_end_of_day_max_abs": (0.000092, 0.00015),
    },
    "catpa": {
        "lai_rmse": (0.00606, 0.009),
        "cwad_rmse_rel_to_max": (0.000746, 0.0012),
        "et.EO.max_abs": (0.00115, 0.0018),
        "et.EOS.max_abs": (0.00588, 0.009),
        "et.EOP.max_abs": (0.0052, 0.008),
        "et.ES.max_abs": (0.0101, 0.015),
        "et.EP.max_abs": (0.0582, 0.085),
        "et.EM.max_abs": (0.0, 0.0005),
        "soil_water.SWTD.max_abs": (0.751, 1.1),
        "soil_water.SW_layers_max_abs": (0.00113, 0.0017),
        "dump.sw_end_of_day_max_abs": (0.000941, 0.0014),
    },
}


def bound_group(run: str) -> str:
    """The :data:`SECONDARY_BOUNDS` group of a run key."""
    return "catpa" if run.startswith("CTPA") else "m2"


#: TRWUP (P1, ours) against the SPAM exit dump. The 58 M2 treatments: at most 0.0020 cm d-1 on any
#: day (BRPI0202 t03). The CA-TPA seasons deviate more (measured, reported here): the season
#: total by at most 1.48 % (2020, t06; 0.21 % in 2018, t04; <= 0.05 % in the other five), single days
#: by up to 11 % of the day's value and 0.68 cm d-1 (t06); the yield stays within 2 % (t06: 0.03 %).
#: Pinned: the season total within TRWUP_SUM_REL_CATPA, the M2 days within TRWUP_ATOL_M2 [cm d-1].
TRWUP_SUM_REL_CATPA = 0.02
TRWUP_ATOL_M2 = 0.005
#: the soil properties SOILDYN may change (SOILPROP components the day's modules read)
SOILPROP = ("dlayr", "ds", "ll", "dul", "sat", "swcn")
MM_PER_CM = 10.0
#: unit roundoff of REAL*4
U32 = 2.0**-24
#: rounded REAL*4 operations on the longest dependency path (Fortran statements cited), plus one
#: for the rounding of the dumped result:
#: XTRACT layer extraction: SW_AVAIL (SPAM.for:272, 2 adds + MAX), the ES adjustment (426 / 430: 3),
#: - LL (SPSUBS.for:440), WUF (458: 2), RWU * WUF (466), the cap (467-468: 2), SWTEMP (470: 2),
#: SWDELTX = SWTEMP - SW (483), our -SWDELTX DLAYR (1): 16
OPS_UPTAKE = 16
#: EP = 10 TRWU: the layer sum (NLAYR additions, at most 20) after the layer terms
OPS_EP = 16 + 20
#: ALBEDO_avg: FF (3), MIN/MAX, SWALB (3), MSALB (4)
OPS_MSALB = 12
#: PETPT (spam_evap_dssat.OPS["EO"])
OPS_EO = 16


# ------------------------------------------------------------------------ keys and reference runs
def a12_keys(data_dir: Path) -> list[tuple[str, int]]:
    """``(experiment, treatment)`` of the 58 M2 treatments (the SPAM dump tables)."""
    out = []
    for p in sorted((data_dir / A12).glob("*_spam.npz")):
        exp, t = p.name.removesuffix("_spam.npz").rsplit("_t", 1)
        out.append((exp, int(t)))
    return out


def catpa_keys() -> list[tuple[str, int]]:
    """The 7 nitrogen-off CA-TPA seasons 2015-2021 (treatments 1-7 of CTPA1501.MZX)."""
    return [("CTPA1501", k) for k in range(1, 8)]


def key_of(exp: str, trno: int) -> str:
    return f"{exp}_t{trno:02d}"


def run_catpa(trno: int, dest: Path, data_dir: Path) -> Path:
    """One nitrogen-off CA-TPA season as a one-treatment batch of the case (daily outputs on)."""
    from agrijax.port.run_fortran import run_dscsm

    case = data_dir / CATPA_CASE
    exp = dest / "exp"
    exp.mkdir(parents=True, exist_ok=True)
    for f in case.iterdir():
        if f.is_file() and not f.name.endswith(".OUT"):
            shutil.copy2(f, exp / f.name)
    x = exp / CATPA_FILEX
    x.write_text(m2._nitrogen_off(x.read_text(errors="replace")))
    batch = "$BATCH(MAIZE)\n!\n@FILEX" + " " * 88 + "TRTNO     RP     SQ     OP     CO\n"
    batch += CATPA_FILEX.ljust(92) + f"{trno:7d}      1      0      0      0\n"
    (exp / "DSSBatch.v48").write_text(batch)
    out = dest / "out"
    run_dscsm(
        exp,
        out,
        run_mode="B",
        experiment_file="DSSBatch.v48",
        weather_dir=exp,
        soil_dir=exp,
        keep_files=("*.OUT", "DSSAT48.INP"),
    )
    return out


def run_references(keys: list[tuple[str, int]], work: Path, data_dir: Path, jobs: int) -> dict[str, Path]:
    """The reference run of every key, in parallel (one ``dscsm048`` process per run)."""

    def one(k: tuple[str, int]) -> Path:
        exp, t = k
        dest = work / key_of(exp, t)
        if exp == "CTPA1501":
            return run_catpa(t, dest, data_dir)
        return m2.run_reference(exp, t, dest)

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        outs = list(ex.map(one, keys))
    return {key_of(e, t): o for (e, t), o in zip(keys, outs, strict=True)}


# ------------------------------------------------------------------------ one run's inputs
def _chr(a: Any) -> str:
    v = np.asarray(a).reshape(-1)
    if v.dtype.kind in "SU":
        s = v[0]
        return (s.decode() if isinstance(s, bytes) else str(s)).strip()
    return np.asarray(a).tobytes().decode(errors="replace").strip("\x00 ")


def decimal_of(x: Any) -> np.ndarray:
    """The shortest decimal that rounds to the REAL*4 value (the number as written in the input file)."""
    a = np.asarray(x, dtype=np.float32)
    return np.asarray([float(str(v)) for v in a.reshape(-1)], dtype=np.float64).reshape(a.shape)


def _table(d: Path, name: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t, _ = dumps.load_table(d / f"{name}.npz")
    return np.asarray(t.date, dtype=np.int64), {k: np.asarray(v) for k, v in t.values.items()}


def _next_day(yrdoy: int) -> int:
    d = date(yrdoy // 1000, 1, 1) + timedelta(days=yrdoy % 1000)
    return d.year * 1000 + d.timetuple().tm_yday


@dataclass
class Run:
    """Host-side inputs of one run and its reference values."""

    key: str
    exp: str
    trno: int
    out: Path
    params_crop: Any
    forcing_crop: Any
    m2_res: dict[str, np.ndarray] | None
    row: Any
    nl: int
    mesev: str
    site: DssatSite
    soildyn: bool
    series: dict[str, np.ndarray]
    sw0: np.ndarray
    snow0: float
    mulch0: float
    mulch_evap0: float
    ksevap: float
    spam_in: dict[str, np.ndarray]
    spam_out: dict[str, np.ndarray]
    rootwu_rwu: np.ndarray | None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def days(self) -> np.ndarray:
        return np.asarray(self.forcing_crop.yrdoy, dtype=np.int64)

    @property
    def n_days(self) -> int:
        return int(self.days.shape[0])


def _catpa_crop(out: Path, trno: int, data_dir: Path, supply: tuple[np.ndarray, np.ndarray, np.ndarray]):
    """CERES-Maize's parameters and forcing of a CA-TPA season (the case's own genotype files)."""
    case = data_dir / CATPA_CASE
    inp = out / "DSSAT48.INP"
    p = ceres_params(str(inp), str(case / "MZCER048.ECO"), str(case / "MZCER048.SPE"))
    summ = read_summary(out / "Summary.OUT")
    row = summ[summ["TRNO"] == trno].iloc[0]
    text = inp.read_text(errors="replace").splitlines()
    wname = next(ln.split()[1] for ln in text if ln.startswith("WEATHERW"))
    lat = float(read_wth(case / wname).attrs["site"]["LAT"])
    last = int(np.nanmax([row["MDAT"], row["HDAT"]]))
    f = ceres_forcing(str(out), trno, int(row["SDAT"]), last, int(p.soil.dlayr.shape[0]), lat, supply=supply)
    return p, f, row


def build(exp: str, trno: int, out: Path, data_dir: Path) -> Run:
    """The inputs of one run (module docstring) on the crop's days."""
    key = key_of(exp, trno)
    si_dates, si = _table(data_dir / DET, f"{key}_spam_in")
    so_dates, so = _table(data_dir / DET, f"{key}_spam_out")
    assert np.array_equal(si_dates, so_dates), key
    if exp == "CTPA1501":
        supply = (so["YRDOY"].astype(np.int64), so["EOP"].astype(np.float64), so["TRWUP"].astype(np.float64))
        p, f, row = _catpa_crop(out, trno, data_dir, supply)
        res = None
    else:
        p, f, res, row = m2.simulate(
            out, trno, daylength_from_output=exp in DAYLENGTH_FROM_OUTPUT, exp=exp, tables=data_dir / A12
        )
    days = np.asarray(f.yrdoy, dtype=np.int64)
    d = data_dir / DSW / key
    rate_dates, ri = _table(d, "wb_rate_in")
    _, ii = _table(d, "wb_integr_in")
    _, mri = _table(d, "mulch_rate_in")
    _, si0 = _table(d, "wb_seasinit_in")
    pmf = 0.0
    if (d / "rnoff_in.npz").is_file():
        pmf = float(_table(d, "rnoff_in")[1]["PMFRACTION"][0])
    pos = {int(x): i for i, x in enumerate(rate_dates.tolist())}
    spos = {int(x): i for i, x in enumerate(si_dates.tolist())}
    missing = [int(x) for x in days if int(x) not in pos or int(x) not in spos]
    assert not missing, (key, "crop days without a WATBAL / SPAM call", missing[:5])
    at = np.asarray([pos[int(x)] for x in days])
    sat_ = np.asarray([spos[int(x)] for x in days])
    nl = int(ri["SP_NLAYR"][0])
    assert nl == int(p.soil.dlayr.shape[0]), (key, nl, p.soil.dlayr.shape)

    def lay(tab: dict[str, np.ndarray], name: str, idx: Any = None) -> np.ndarray:
        v = np.asarray(tab[name], dtype=np.float64)[:, :nl]
        return v if idx is None else v[idx]

    def sca(tab: dict[str, np.ndarray], name: str, idx: Any = None) -> np.ndarray:
        v = np.asarray(tab[name], dtype=np.float64).reshape(-1)
        return v if idx is None else v[idx]

    src = si0 if "SP_DLAYR" in si0 else ri
    static = {k: lay(src, f"SP_{k.upper()}")[0] for k in SOILPROP}
    static_cn, static_swcon = float(sca(src, "SP_CN")[0]), float(sca(src, "SP_SWCON")[0])
    daily = {k: lay(ri, f"SP_{k.upper()}", at) for k in SOILPROP}
    daily["cn"] = sca(ri, "SP_CN", at)
    daily["swcon"] = sca(ri, "SP_SWCON", at)
    dlayr_end = lay(ii, "SP_DLAYR", at)
    changes = any(np.any(daily[k] != static[k][None, :]) for k in SOILPROP) or bool(
        np.any(dlayr_end != static["dlayr"][None, :])
    )
    changes |= bool(np.any(daily["cn"] != static_cn) or np.any(daily["swcon"] != static_swcon))
    mesev = _chr(si["MESEV"][0])
    meinf = _chr(ri["MEINF"][0])
    site = DssatSite(
        dlayr=decimal_of(static["dlayr"]),
        ds=decimal_of(static["ds"]),
        ll=decimal_of(static["ll"]),
        dul=decimal_of(static["dul"]),
        sat=decimal_of(static["sat"]),
        swcn=decimal_of(static["swcn"]),
        cn=float(decimal_of(static_cn)),
        swcon=float(decimal_of(static_swcon)),
        salb=float(decimal_of(si["SALB_S"][0])),
        u=float(decimal_of(si["U_S"][0])),
        mesev=mesev,
        meinf=meinf,
        actwtd=float(sca(ri, "ACTWTD")[0]),
        pmfraction=pmf,
    )
    series = {
        "rain": sca(ri, "RAIN", at),
        "tmax": sca(ri, "TMAX", at),
        "irrigation": sca(ri, "IRRAMT", at),
        "m_mass": sca(ri, "M_MASS", at),
        "m_cover": sca(ri, "M_COVER", at),
        "m_new": sca(ri, "M_NEW", at),
        "m_watfac": sca(ri, "M_WATFAC", at),
        "mulch_am": sca(si, "MULCH_AM", sat_),
        "mulch_extfac": sca(si, "MUL_EXTFAC", sat_),
        "tavg": sca(si, "TAVG_W", sat_),
        "wind_run": sca(si, "WINDSP_W", sat_),
        "co2": sca(si, "CO2_W", sat_),
        "srad": sca(si, "SRAD_W", sat_),
        "tmax_w": sca(si, "TMAX_W", sat_),
        "tmin_w": sca(si, "TMIN_W", sat_),
        "dlayr_end": dlayr_end,
        **{f"soil_{k}": v for k, v in daily.items()},
    }
    ks = np.unique(np.asarray(si["KSEVAP"], dtype=np.float64))
    kt = np.unique(np.asarray(si["KTRANS"], dtype=np.float64))
    assert ks.size == 1 and kt.size == 1 and ks[0] == kt[0], (key, ks, kt)
    rw = None
    a12 = data_dir / A12 / f"{key}_rootwu_out.npz"
    if a12.is_file():
        t, _ = dumps.load_table(a12)
        rpos = {int(x): i for i, x in enumerate(np.asarray(t.date).tolist())}
        rwu = np.zeros((len(si_dates), nl))
        for i, x in enumerate(si_dates.tolist()):
            if int(x) in rpos:
                rwu[i] = np.asarray(t.values["RWU"], dtype=np.float64)[rpos[int(x)], :nl]
        rw = rwu
    run = Run(
        key=key,
        exp=exp,
        trno=trno,
        out=out,
        params_crop=p,
        forcing_crop=f,
        m2_res=res,
        row=row,
        nl=nl,
        mesev=mesev,
        site=site,
        soildyn=bool(changes),
        series=series,
        sw0=lay(ri, "SW", at)[0],
        snow0=float(sca(ri, "SNOW")[at[0]]),
        mulch0=float(sca(mri, "MULCHWAT")[at[0]]),
        mulch_evap0=float(sca(mri, "MULCHEVAP")[at[0]]),
        ksevap=float(ks[0]),
        spam_in={k: np.asarray(v)[sat_] for k, v in si.items()},
        spam_out={k: np.asarray(v)[sat_] for k, v in so.items()},
        rootwu_rwu=None if rw is None else rw[sat_],
    )
    run.notes = {
        "mesev": mesev,
        "meinf": meinf,
        "soildyn_replay": run.soildyn,
        "mulch_days": int(np.sum(series["m_mass"] > 0.0)),
        "first_day_is_simulation_start": int(days[0]) == int(rate_dates[0]),
    }
    return run


# ------------------------------------------------------------------------ batched inputs
def _pad(x: Any, n: int) -> np.ndarray:
    a = np.asarray(x)
    idx = np.minimum(np.arange(n), a.shape[0] - 1)
    return a[idx]


def _stack(trees: list[Any]) -> Any:
    return jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *trees)


def params_of(
    r: Run, soil_values: str, real4_sw: bool | None = None, soil_evaporation: str | None = None
) -> dict[str, Any]:
    p = day_params(
        r.site,
        r.params_crop,
        ksevap=r.ksevap,
        ktrans=r.ksevap,
        soil_values=soil_values,
        soil_evaporation=soil_evaporation,
    )
    if real4_sw is not None:  # the mixed-precision configuration (measurement only)
        p["soil"] = p["soil"].replace(real4_sw=real4_sw)
    return p


def forcing_of(r: Run, n: int, soil_values: str, soildyn: bool) -> dict[str, Any]:
    """The run's global forcing, padded to ``n`` days (the last day repeated; never compared)."""
    s = r.series
    j = lambda x: jnp.asarray(_pad(np.asarray(x, dtype=np.float64), n))  # noqa: E731
    conv = lambda x: dssat_soil_values(x, soil_values)  # noqa: E731
    if soildyn and r.soildyn:
        # the dumped REAL*4 values under the run's soil-value convention, as the static soil
        # (``real4``: unchanged; ``input``: the decimal each value was read from)
        soil = BucketSoil(**{k: j(conv(decimal_of(s[f"soil_{k}"]))) for k in (*SOILPROP, "cn", "swcon")})
        dlayr_end = j(conv(decimal_of(s["dlayr_end"])))
    else:
        st = r.site
        tile = lambda v: np.broadcast_to(conv(v), (r.n_days, *np.shape(v)))  # noqa: E731
        soil = BucketSoil(
            dlayr=j(tile(st.dlayr)),
            ds=j(tile(st.ds)),
            ll=j(tile(st.ll)),
            dul=j(tile(st.dul)),
            sat=j(tile(st.sat)),
            swcn=j(tile(st.swcn)),
            cn=j(tile(st.cn)),
            swcon=j(tile(st.swcon)),
        )
        dlayr_end = soil.dlayr
    mulch = MulchForcing(
        mass=j(s["m_mass"]), cover=j(s["m_cover"]), new_mass=j(s["m_new"]), watfac=j(s["m_watfac"])
    )
    after = np.arange(n) >= r.n_days
    zero_after = lambda x: jnp.where(jnp.asarray(after), 0.0, j(x))  # noqa: E731
    bf = BucketForcing(
        rain=zero_after(s["rain"]),
        tmax=j(s["tmax"]),
        irrigation=zero_after(s["irrigation"]),
        mulch=mulch,
        soil=soil,
        dlayr_end=dlayr_end,
    )
    weather = SpamWeather(
        tavg=j(s["tavg"]),
        wind_run=j(s["wind_run"]),
        co2=j(s["co2"]),
        srad=j(s["srad"]),
        tmax=j(s["tmax_w"]),
        tmin=j(s["tmin_w"]),
    )
    fc = jax.tree.map(lambda x: jnp.asarray(_pad(np.asarray(x), n)), r.forcing_crop)
    fc = fc.replace(
        sw=jnp.full_like(fc.sw, jnp.nan),
        eop=jnp.full_like(fc.eop, jnp.nan),
        trwup=jnp.full_like(fc.trwup, jnp.nan),
    )
    return {
        "crop": fc,
        "soil": bf,
        "spam": {"weather": weather, "mulch_am": j(s["mulch_am"]), "mulch_extfac": j(s["mulch_extfac"])},
    }


def state_of(r: Run, params: dict[str, Any], soil_values: str) -> dict[str, Any]:
    sw0 = jnp.asarray(dssat_soil_values(r.sw0, soil_values))
    b = BucketState.initial(
        sw0, snow=r.snow0, mulch_wat=r.mulch0, mulch_evap_prev=r.mulch_evap0, dtype=jnp.float64
    )
    soil = params["soil"].soil
    se = SoilEvapState.initial(sw0, soil.dlayr, soil.ds, soil.dul, soil.ll, params["evap"].u)
    storage0 = jnp.sum(sw0 * soil.dlayr) + (r.snow0 + r.mulch0) / MM_PER_CM
    return initial_state(
        bucket=b,
        soil_evap=se,
        crop=CeresMaizeState.initial(params["crop"], 1),
        rootwu=RootwuState.initial(1, r.nl, jnp.float64),
        salb=params["albedo"].salb,
        storage0=storage0,
    )


#: ``{name: (soil values, SOILDYN replay, REAL*4 water content override)}``: ``free`` the faithful
#: configuration; ``input`` the decimal soil values with a float64 water content; ``real4_limits``
#: the REAL*4 soil values with a float64 water content (the mixed precision the convention avoids);
#: ``static`` the faithful one without the SOILDYN replay
CONFIGS: dict[str, tuple[str, bool, bool | None]] = {
    "free": ("real4", True, None),
    "input": ("input", True, None),
    "real4_limits": ("real4", True, False),
    "static": ("real4", False, None),
}


def run_group(rs: list[Run], config: str, soil_evaporation: str | None = None) -> dict[str, np.ndarray]:
    """One configuration on the runs ``rs`` (one layer count, one MESEV), batched. ``soil_evaporation``
    swaps the soil evaporation (a registry key; default: the runs' own MESEV, the reference's)."""
    soil_values, soildyn, real4_sw = CONFIGS[config]
    mesev = rs[0].mesev
    assert all(r.mesev == mesev for r in rs)
    impl = SOIL_EVAPORATION_KEYS[mesev] if soil_evaporation is None else soil_evaporation
    n = max(r.n_days for r in rs)
    ps = [params_of(r, soil_values, real4_sw, soil_evaporation=impl) for r in rs]
    params = _stack(ps)
    # the coupling flags themselves, on the host (inside jit only the static record is readable)
    assert not soil_evaporation_params_problems(params, impl), soil_evaporation_params_problems(params, impl)
    state = _stack([state_of(r, p, soil_values) for r, p in zip(rs, ps, strict=True)])
    forcing = _stack([forcing_of(r, n, soil_values, soildyn) for r in rs])
    model = day_dssat486(SLOT).compile(
        day_processes(SLOT, soil_evaporation=impl), outputs=day_outputs(SLOT), exact_lags=True
    )
    out = run_sites(model, params, forcing, state)
    return {k: np.asarray(v) for k, v in out.items()}


# ------------------------------------------------------------------------ comparisons
def _first(stage: np.ndarray, days: np.ndarray, code: int) -> int:
    hit = np.nonzero(stage == code)[0]
    return int(days[hit[0]]) if hit.size else -99


def _yrdoy_diff(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None if a == b else 9999
    da = date(a // 1000, 1, 1) + timedelta(days=a % 1000 - 1)
    db = date(b // 1000, 1, 1) + timedelta(days=b % 1000 - 1)
    return (da - db).days


def _dates(r: Run, istage: np.ndarray) -> dict[str, dict[str, Any]]:
    out = {}
    for name, code, col in (("emergence", 1, "EDAT"), ("silking", 4, "ADAT"), ("maturity", 10, "MDAT")):
        ref = float(r.row[col])
        ours = _first(istage, r.days, code)
        refi = int(ref) if ref > 0 else -99
        out[name] = {"ours": ours, "ref": refi, "diff_days": _yrdoy_diff(ours, refi)}
    return out


def _by_day(df: pd.DataFrame, col: str, days: np.ndarray) -> np.ndarray:
    key = np.asarray(df["YEAR"], dtype=int) * 1000 + np.asarray(df["DOY"], dtype=int)
    m = dict(zip(key.tolist(), np.asarray(df[col], dtype=float).tolist(), strict=True))
    return np.asarray([m.get(int(d), np.nan) for d in days], dtype=float)


def _err(ours: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(ref)
    d = ours[ok] - ref[ok]
    if not d.size:
        return {"n": 0}
    scale = float(np.max(np.abs(ref[ok]))) or 1.0
    return {
        "n": int(d.size),
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "rmse_rel_to_max": float(np.sqrt(np.mean(d * d)) / scale),
        "sum_ours": float(np.sum(ours[ok])),
        "sum_ref": float(np.sum(ref[ok])),
    }


def _sum_rel(ours: np.ndarray, ref: np.ndarray) -> float:
    """``|sum(ours) - sum(ref)| / sum(ref)`` (0 when the reference total is 0)."""
    total = float(np.sum(ref))
    return abs(float(np.sum(ours)) - total) / total if total > 0.0 else 0.0


def secondary(row: dict[str, Any]) -> dict[str, float]:
    """The secondary metrics of one run's row that :data:`SECONDARY_BOUNDS` pins."""
    get = {
        "lai_rmse": lambda r: r["lai"].get("rmse", 0.0),
        "cwad_rmse_rel_to_max": lambda r: r["cwad"].get("rmse_rel_to_max", 0.0),
        **{
            f"et.{k}.max_abs": (lambda r, k=k: r["et"].get(k, {}).get("max_abs", 0.0))
            for k in ("EO", "EOS", "EOP", "ES", "EP", "EM")
        },
        "soil_water.SWTD.max_abs": lambda r: r["soil_water"]["SWTD"].get("max_abs", 0.0),
        "soil_water.SW_layers_max_abs": lambda r: r["soil_water"]["SW_layers_max_abs"] or 0.0,
        "dump.sw_end_of_day_max_abs": lambda r: r["dump"]["sw_end_of_day_max_abs"],
    }
    return {k: float(f(row)) for k, f in get.items()}


def compare(r: Run, o: dict[str, np.ndarray]) -> dict[str, Any]:
    """One run's row of the report from the batched outputs ``o`` (this run's slice)."""
    n = r.n_days
    days = r.days
    hwam = float(r.row["HWAM"])
    y = float(o["gwad"][n - 1, 0])
    pg = read_plantgro(r.out / "PlantGro.OUT")
    pg = pg[pg["TRNO"] == r.trno]
    et = read_out(r.out / "ET.OUT")
    et = et[et["TRNO"] == r.trno] if "TRNO" in et.columns else et
    sw = read_soilwat(r.out / "SoilWat.OUT")
    sw = sw[sw["TRNO"] == r.trno] if "TRNO" in sw.columns else sw
    res = {
        k: o[k][:n]
        for k in ("lai", "cwad", "gwad", "lwad", "swad", "rwad", "g_ad", "rdpd", "gstd", "lsd", "istage")
    }
    errs = m2.daily_errors(
        r.out, r.trno, r.forcing_crop, {**res, "ewsd": o["ewsd"][:n], "g_ad": o["g_ad"][:n]}
    )
    row: dict[str, Any] = {
        "run": r.key,
        "n_days": n,
        "n_layer": r.nl,
        **r.notes,
        "hwam": hwam,
        "yield": y,
        "yield_rel": abs(y - hwam) / hwam if hwam > 0 else None,
        "yield_ok": (abs(y - hwam) / hwam < YIELD_REL) if hwam > 0 else abs(y - hwam) < 1.0,
        "dates": _dates(r, o["istage"][:n, 0]),
        "plantgro": {c: errs[c] for c in ("LAID", "CWAD", "GWAD", "LWAD", "SWAD", "RWAD", "GSTD", "L#SD")},
        "lai": _err(o["lai"][:n, 0], _by_day(pg, "LAID", days)),
        "cwad": _err(o["cwad"][:n, 0], _by_day(pg, "CWAD", days)),
        "cwam": {"ours": float(o["cwad"][n - 1, 0]), "ref": float(r.row["CWAM"])},
    }
    if r.m2_res is not None:
        row["yield_m2"] = float(r.m2_res["gwad"][-1, 0])
        row["yield_m2_rel"] = abs(row["yield_m2"] - hwam) / hwam if hwam > 0 else None
    # ET.OUT (mm d-1, F8.3 or F7.2) and SoilWat.OUT (mm, SW 3 decimals)
    row["et"] = {
        name: _err(o[ours][:n], _by_day(et, col, days))
        for name, ours, col in (
            ("EO", "eo", "EOAA"),
            ("EOS", "eos", "EOSA"),
            ("EOP", "eop", "EOPA"),
            ("ES", "es", "ESAA"),
            ("EP", "ep", "EPAA"),
            ("EM", "em", "EMAA"),
        )
        if col in et.columns
    }
    dl_end = r.series["dlayr_end"]
    swtd = np.sum(o["soil_sw"][:n, : r.nl] * dl_end, axis=-1) * MM_PER_CM
    layer_cols = [k for k in range(r.nl) if f"SW{k + 1}D" in sw.columns]
    sw_ref = np.stack([_by_day(sw, f"SW{k + 1}D", days) for k in layer_cols], -1) if layer_cols else None
    row["soil_water"] = {
        "SWTD": _err(swtd, _by_day(sw, "SWTD", days)),
        "SW_layers_printed": len(layer_cols),
        "SW_layers_max_abs": float(np.nanmax(np.abs(o["soil_sw"][:n, layer_cols] - sw_ref)))
        if sw_ref is not None
        else None,
        "ROFD": _err(o["runoff"][:n], _by_day(sw, "ROFD", days)) if "ROFD" in sw.columns else None,
    }
    # full-precision diagnostics against the SPAM entry / exit dumps
    si, so = r.spam_in, r.spam_out
    nxt = np.arange(1, n)
    row["dump"] = {
        "sw_end_of_day_max_abs": float(np.max(np.abs(o["soil_sw"][: n - 1, : r.nl] - si["SW"][nxt, : r.nl]))),
        "msalb_max_abs": float(np.max(np.abs(o["msalb"][:n] - si["MSALB_S"].astype(float)))),
        "eo_max_abs": float(np.max(np.abs(o["eo"][:n] - so["EO"].astype(float)))),
        "es_max_abs": float(np.max(np.abs(o["es"][:n] - so["ES"].astype(float)))),
        "ep_max_abs": float(np.max(np.abs(o["ep"][:n] - so["EP"].astype(float)))),
        "eop_max_abs": float(np.max(np.abs(o["p1_eop"][:n, 0] - so["EOP"].astype(float)))),
        "trwup_max_abs": float(np.max(np.abs(o["p1_trwup"][:n, 0] - so["TRWUP"].astype(float)))),
        "trwup_sum_rel": _sum_rel(o["p1_trwup"][:n, 0], so["TRWUP"].astype(float)),
        "xhlai_next_max_abs": float(np.max(np.abs(o["p6_lai"][: n - 1, 0] - si["XHLAI"][nxt].astype(float)))),
        "mulch_wat_start_max_abs": float(
            np.max(
                np.abs(np.concatenate([[r.mulch0], o["mulch_wat"][: n - 1]]) - si["MULCHWAT"].astype(float))
            )
        ),
    }
    resid = o["residual"][:n] * MM_PER_CM
    row["ledger"] = {"max_abs_residual_mm": float(np.max(np.abs(resid))), "closure_mm": float(np.sum(resid))}
    row["finite"] = bool(
        all(np.all(np.isfinite(o[k][:n])) for k in ("lai", "cwad", "gwad", "soil_sw", "es", "ep"))
    )
    return row


def evaluate(
    keys: list[tuple[str, int]],
    data_dir: Path,
    work: Path,
    jobs: int,
    configs: tuple[str, ...] = tuple(CONFIGS),
) -> dict[str, list[dict[str, Any]]]:
    """Reference runs, every configuration batched by (layer count, MESEV), per-run rows."""
    outs = run_references(keys, work, data_dir, jobs)
    runs = [build(e, t, outs[key_of(e, t)], data_dir) for e, t in keys]
    table: dict[str, list[dict[str, Any]]] = {}
    for cfg in configs:
        rows: dict[str, dict[str, Any]] = {}
        for grp in sorted({(r.nl, r.mesev) for r in runs}):
            rs = [r for r in runs if (r.nl, r.mesev) == grp]
            o = run_group(rs, cfg)
            for i, r in enumerate(rs):
                oi = {k: v[i] for k, v in o.items()}
                rows[r.key] = compare(r, oi)
                if cfg == "free":
                    save_daily(r, oi, data_dir)
        table[cfg] = [rows[key_of(e, t)] for e, t in keys]
    table["xtract_one_day"] = [one_day_checks(r) for r in runs]
    return table


DAILY_KEYS = (
    "gwad",
    "cwad",
    "lai",
    "wspd",
    "wsgd",
    "istage",
    "p1_trwup",
    "p1_eop",
    "ep",
    "es",
    "soil_sw",
    "rlv",
)


def save_daily(r: Run, o: dict[str, np.ndarray], data_dir: Path) -> Path:
    """The free run's daily series of one run next to the reference's (SPAM exit TRWUP, EOP, EP; the
    M2 crop on the replayed water) for the per-day diagnosis of a deviation."""
    n = r.n_days
    out = data_dir / REPORT_DIR / "daily"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"ours.{k}": np.asarray(o[k][:n]) for k in DAILY_KEYS if k in o}
    arrays |= {f"ref.{k}": np.asarray(r.spam_out[k], dtype=np.float64) for k in ("TRWUP", "EOP", "EP", "ES")}
    arrays["ref.SW_next"] = np.asarray(r.spam_in["SW"], dtype=np.float64)[:, : r.nl]
    if r.m2_res is not None:
        arrays |= {
            f"m2.{k}": np.asarray(r.m2_res[k])
            for k in ("gwad", "cwad", "lai", "wspd", "wsgd")
            if k in r.m2_res
        }
    arrays["days"] = r.days
    path = out / f"{r.key}.npz"
    np.savez_compressed(path, **arrays)
    return path


# ------------------------------------------------------------------------ one-day checks
def one_day_checks(r: Run) -> dict[str, Any]:
    """XTRACT, ALBEDO_avg and PETPT on the dumped REAL*4 inputs of every day, against the dumped
    outputs, in units of ``U32 * scale`` (the REAL*4 limits :data:`OPS_UPTAKE`, :data:`OPS_EP`,
    :data:`OPS_MSALB`, :data:`OPS_EO`)."""
    si, so = r.spam_in, r.spam_out
    f = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)  # noqa: E731
    nl = r.nl
    sw = f(si["SW"])[:, :nl]
    dl = f(si["DLAYR_S"])[:, :nl]
    ll = f(si["LL_S"])[:, :nl]
    salus = r.mesev == "S"
    out: dict[str, Any] = {"run": r.key}
    # albedo and PETPT (every day)
    msalb, _ = soil_albedo(
        sw[:, 0], f(si["DUL_S"])[:, 0], f(si["SALB_S"]), f(si["MULCHCOVER"]), r.site.mulch_on
    )
    ms_ref = f(si["MSALB_S"])
    ratio_ms = np.abs(np.asarray(msalb) - ms_ref) / (U32 * np.maximum(f(si["SALB_S"]), 1e-30))
    out["msalb_ulp_max"] = float(ratio_ms.max())
    out["msalb_ok_days"] = int(np.sum(ratio_ms <= OPS_MSALB))
    eo = priestley_taylor(f(si["SRAD_W"]), f(si["TMAX_W"]), f(si["TMIN_W"]), f(si["XHLAI"]), ms_ref)
    eo_ref = f(so["EO"])
    ratio_eo = np.abs(np.asarray(eo) - eo_ref) / (U32 * np.maximum(np.abs(eo_ref), 1e-30))
    out["eo_ulp_max"] = float(ratio_eo.max())
    out["eo_ok_days"] = int(np.sum(ratio_eo <= OPS_EO))
    out["n_days"] = int(sw.shape[0])
    if r.rootwu_rwu is None:
        out["xtract"] = "no ROOTWU exit tables for this run"
        return out
    ep = actual_transpiration(f(so["EOP"]), f(so["TRWUP"]), f(si["XHLAI"]))
    ep_ref_in = np.asarray(ep)
    es_lyr = f(so["ES_LYR_L"])[:, :nl] if "ES_LYR_L" in so else np.zeros_like(sw)
    avail = extraction_supply(
        sw,
        f(si["SWDELTS"])[:, :nl],
        f(si["SWDELTU"])[:, :nl],
        dl,
        f(so["ES"]) / MM_PER_CM,
        es_lyr / MM_PER_CM,
        salus,
    )
    res = xtract(sw, avail, ll, dl, f(so["TRWUP"]), r.rootwu_rwu, ep_ref_in)
    up_ref = -f(so["SWDELTX"])[:, :nl] * dl
    scale = np.maximum(sw * dl + np.asarray(r.rootwu_rwu), 1e-30)
    ratio_up = np.abs(np.asarray(res.uptake) - up_ref) / (U32 * scale)
    ep_ref = f(so["EP"])
    ep_scale = np.maximum(MM_PER_CM * np.sum(scale, axis=-1), 1e-30)
    ratio_ep = np.abs(np.asarray(res.ep) - ep_ref) / (U32 * ep_scale)
    out.update(
        xtract_days=int(sw.shape[0]),
        uptake_ulp_max=float(ratio_up.max()),
        uptake_bad_layer_days=int(np.sum(ratio_up > OPS_UPTAKE)),
        uptake_max_abs_cm=float(np.max(np.abs(np.asarray(res.uptake) - up_ref))),
        ep_ulp_max=float(ratio_ep.max()),
        ep_bad_days=int(np.sum(ratio_ep > OPS_EP)),
        trwu_max_abs_cm=float(np.max(np.abs(np.asarray(res.trwu) - f(so["TRWU"])))),
        ep_days=int(np.sum(ep_ref > 0.0)),
        dry_layer_days=int(np.sum((sw <= ll) & (np.asarray(r.rootwu_rwu) > 0.0))),
    )
    return out


# ------------------------------------------------------------------------ report
def summary(table: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cfg in CONFIGS:
        rows = table.get(cfg)
        if not rows:
            continue
        yrel = [r["yield_rel"] for r in rows if r["yield_rel"] is not None]
        out[cfg] = {
            "n_runs": len(rows),
            "yield_ok": sum(bool(r["yield_ok"]) for r in rows),
            "yield_fail": [
                (r["run"], r["yield_rel"], r["hwam"], r["yield"]) for r in rows if not r["yield_ok"]
            ],
            "yield_rel_max": max(yrel) if yrel else None,
            "yield_rel_median": float(np.median(yrel)) if yrel else None,
            "dates_equal": sum(all(v["diff_days"] in (0, None) for v in r["dates"].values()) for r in rows),
            "dates_diff": {
                r["run"]: {
                    k: v["diff_days"] for k, v in r["dates"].items() if v["diff_days"] not in (0, None)
                }
                for r in rows
                if any(v["diff_days"] not in (0, None) for v in r["dates"].values())
            },
            "lai_rmse_max": max(r["lai"].get("rmse", 0.0) for r in rows),
            "cwad_rmse_rel_max": max(r["cwad"].get("rmse_rel_to_max", 0.0) for r in rows),
            "swtd_rmse_max_mm": max(r["soil_water"]["SWTD"].get("rmse", 0.0) for r in rows),
            "ledger_max_abs_residual_mm": max(r["ledger"]["max_abs_residual_mm"] for r in rows),
            "finite": sum(r["finite"] for r in rows),
        }
    one = table.get("xtract_one_day", [])
    xs = [x for x in one if "uptake_ulp_max" in x]
    out["one_day"] = {
        "runs": len(one),
        "xtract_runs": len(xs),
        "xtract_days": sum(x["xtract_days"] for x in xs),
        "uptake_ulp_max": max((x["uptake_ulp_max"] for x in xs), default=None),
        "uptake_bad_layer_days": sum(x["uptake_bad_layer_days"] for x in xs),
        "ep_ulp_max": max((x["ep_ulp_max"] for x in xs), default=None),
        "ep_bad_days": sum(x["ep_bad_days"] for x in xs),
        "msalb_ulp_max": max(x["msalb_ulp_max"] for x in one) if one else None,
        "msalb_bad_days": sum(x["n_days"] - x["msalb_ok_days"] for x in one),
        "eo_ulp_max": max(x["eo_ulp_max"] for x in one) if one else None,
        "eo_bad_days": sum(x["n_days"] - x["eo_ok_days"] for x in one),
    }
    return out


def write_report(table: dict[str, Any], data_dir: Path, name: str) -> Path:
    out = data_dir / REPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    engine = Path(os.environ.get("AGRI_JAX_DSSAT", str(m2.DSSAT_ENGINE)))
    path.write_text(
        json.dumps(
            {
                "step": "DSSAT day, free run",
                "reference": "DSSAT-CSM v4.8.6.0 dscsm048 (build486), nitrogen off",
                "engine": str(engine),
                "x64": bool(jax.config.jax_enable_x64),
                "configs": {
                    k: {"soil_values": v[0], "soildyn_replay": v[1], "real4_sw_override": v[2]}
                    for k, v in CONFIGS.items()
                },
                "summary": summary(table),
                "runs": table,
            },
            indent=1,
            default=float,
        )
    )
    return path
