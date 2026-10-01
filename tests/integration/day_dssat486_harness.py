"""Harness of ``test_day_dssat486.py``: the DSSAT-CSM v4.8.6.0 day on the 58 maize treatments of M2.

Every treatment with a SPAM dump table (``<data-dir>/dumps/tables/dssat486/<EXP>_t<NN>_spam.npz``)
is run twice through ``dscsm048`` (nitrogen off, as M2): once exactly as
``test_ceres_dssat.run_reference`` (the M2 run) and once with ``VBOSE = D`` added, which makes
``SoilWatBal.OUT`` print the daily water balance (``WBAL``: ``IRRD PRED RESAD LFLOD MEVAP DRND
ROFD FROD ESAD EPAD EFAD TDFD`` and the stores ``SWTD FWTD SNOWD MWTD``, 0.01 mm). The two runs must
print the same ``PlantGro``, ``SoilWat``, ``ET`` and ``Summary`` files (an output switch).

The day (:mod:`agrijax.models.day_dssat486`) is then run on all treatments at once (``vmap`` over
the treatments of the same layer count, days padded by repeating the last day) in two configurations:

* **A, the M2 configuration**: the soil water is ``SoilWat.OUT``'s printed ``SW`` of the day (M2's
  input), ``EOP`` and ``TRWUP`` are the SPAM dump's (ROOTWU replaced by a replay of ``TRWUP``).
  The inputs are M2's, so the CERES outputs must equal ``test_ceres_dssat.simulate`` bit for bit.
* **B, the coupled day**: ROOTWU is ours, on our CERES root record of the day before and the
  start-of-day soil water; the soil water is the full-precision end-of-day ``SW`` (SPAM's
  start-of-day ``SW`` of the next day; on the last simulated day, which has no next SPAM call, the
  printed one), and the other replays are the SPAM dump (``EO``, ``ES``, ``RWU`` after ``XTRACT``,
  ``XHLAI`` / ``XLAI`` of the next call), ``ET.OUT`` (``EMAA``) and ``SoilWatBal.OUT``. The crop
  forcing's own ``sw``, ``eop`` and ``trwup`` are NaN: the crop reads its water only from P1. The
  CERES outputs must meet the M2 criteria of the 58-treatment test (LAI and CWAD within 1 % where
  printed with 3 digits, every growth column within 2.5 print units, stage and leaf number equal,
  yield within 0.5 %), and the water ledger of the replayed soil must close every day within the
  print half-steps of the printed balance terms.

The per-treatment table goes to ``<data-dir>/validation/aj_dday/d2_0_day_dssat486.json``.
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

import pt_dssat  # noqa: E402
import test_ceres_dssat as m2  # noqa: E402

from agrijax.core.runtime import run_sites  # noqa: E402
from agrijax.iface.crop import CanopyRecord  # noqa: E402
from agrijax.io.dssat import read_out, read_plantgro  # noqa: E402
from agrijax.models.day_dssat486 import (  # noqa: E402
    SLOT,
    DssatSite,
    day_dssat486,
    day_outputs,
    day_params,
    day_processes,
    initial_state,
    ledger_entry,
    replay_processes,
    trwup_replay_entry,
)
from agrijax.models.day_rzwqm46 import noop_entry  # noqa: E402
from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model  # noqa: E402
from agrijax.processes.soil_water.bucket import BucketForcing, BucketState  # noqa: E402
from agrijax.processes.soil_water.bucket_evap import SoilEvapState  # noqa: E402
from agrijax.processes.water_supply import RootwuParams, RootwuState  # noqa: E402

#: experiments whose daylength is replaced by the FileX environment modification (growth chamber)
DAYLENGTH_FROM_OUTPUT = {"GAGR0201"}
#: files the M2 run and the detailed-output run must print identically
SAME_FILES = ("PlantGro.OUT", "SoilWat.OUT", "ET.OUT", "Summary.OUT")
#: the M2 criteria of ``test_ceres_dssat.test_every_maize_example_experiment``
M2_REL_BIG = 0.01
M2_UNITS_ALL = 2.5
M2_YIELD_REL = 0.005
M2_UNIT_COLUMNS = ("LAID", "CWAD", "LWAD", "SWAD", "GWAD", "RWAD")
#: WBAL daily FORMAT (WBAL.for:229-235): stores and flows F7.2 / F8.2 -> half-step 0.005 mm;
#: ET.OUT EMAA F8.3 -> 0.0005 mm
PRINT_HALF_MM = 0.005
EMAA_HALF_MM = 0.0005
#: WATBAL INTEGR rounds SW to 1e-6 (WATBAL.for:503-505): at most 5e-7 per layer, times DLAYR [cm] * 10 mm
SW_ROUND = 5e-7
#: SoilWat.OUT prints SW with F8.3 (the end-of-day SW of the last simulated day comes from it)
SW_PRINT_HALF = 5e-4
#: DSSAT's REAL arithmetic of the day's integration (a few operations of 2^-24 on stores of ~1e2 mm
#: per layer) and the float32 dump values: 1e-3 mm per day in total (measured residuals are below)
REAL_SLACK_MM = 1e-3
#: closure tolerance of configuration B's ledger entry under AGRI_JAX_CHECK=1 [cm]: the largest
#: residual measured on the 58 treatments is 0.037 cm (0.37 mm, on a last simulated day, whose
#: end-of-day SW is the F8.3 print; 0.018 mm on every other day; validation/aj_dday/
#: d2_0_day_dssat486.json), plus a margin
LEDGER_ATOL_CM = 0.05
#: SoilWatBal columns that are zero for these treatments (flood, lateral flow, tile drainage)
ZERO_COLUMNS = ("FWTD", "LFLOD", "FROD", "EFAD", "TDFD")
MM_PER_CM = 10.0


def tables_dir(data_dir: Path) -> Path:
    return data_dir / "dumps" / "tables" / "dssat486"


def treatments(tables: Path) -> list[tuple[str, int]]:
    """``(experiment, treatment)`` of every SPAM dump table."""
    out = []
    for p in sorted(tables.glob("*_spam.npz")):
        exp, t = p.name.removesuffix("_spam.npz").rsplit("_t", 1)
        out.append((exp, int(t)))
    return out


# ------------------------------------------------------------------------ reference runs
def run_detailed(exp: str, trno: int, dest: Path) -> Path:
    """``test_ceres_dssat.run_reference`` with ``VBOSE = D`` (daily ``SoilWatBal.OUT``)."""
    from agrijax.port.run_fortran import run_dscsm

    dest.mkdir(parents=True, exist_ok=True)
    for f in m2.MAIZE.glob(exp + ".MZ*"):
        shutil.copy2(f, dest / f.name)
    x = dest / f"{exp}.MZX"
    text = m2._nitrogen_off(x.read_text(errors="replace"))
    text, n = pt_dssat.patch_filex(text, (("@N OUTPUTS", "VBOSE", "D"),))
    assert n, exp
    x.write_text(text)
    batch = "$BATCH(MAIZE)\n!\n@FILEX" + " " * 88 + "TRTNO     RP     SQ     OP     CO\n"
    batch += f"{exp}.MZX".ljust(92) + f"{trno:7d}      1      0      0      0\n"
    (dest / "DSSBatch.v48").write_text(batch)
    out = dest / "out"
    run_dscsm(
        dest,
        out,
        run_mode="B",
        experiment_file="DSSBatch.v48",
        extra_files=sorted(m2.WEATHER.glob(exp[:4] + "*.WTH")),
        keep_files=("*.OUT", "DSSAT48.INP"),
    )
    return out


def _strip(path: Path) -> list[str]:
    return [
        ln
        for ln in path.read_text(errors="replace").splitlines()
        if "DSSAT Cropping System" not in ln and not ln.startswith("*SUMMARY")
    ]


def run_references(
    keys: list[tuple[str, int]], work: Path, jobs: int
) -> dict[tuple[str, int], tuple[Path, Path]]:
    """The M2 run and the detailed run of every treatment (in parallel: one process per run)."""

    def one(k: tuple[str, int]) -> tuple[Path, Path]:
        exp, t = k
        base = work / f"{exp}_{t}"
        return m2.run_reference(exp, t, base / "m2"), run_detailed(exp, t, base / "wb")

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        return dict(zip(keys, ex.map(one, keys), strict=True))


# ------------------------------------------------------------------------ one treatment
def _next_day(yrdoy: int) -> int:
    d = date(yrdoy // 1000, 1, 1) + timedelta(days=yrdoy % 1000)
    return d.year * 1000 + d.timetuple().tm_yday


def _by_day(df: pd.DataFrame, col: str, days: np.ndarray, fill: float = 0.0) -> np.ndarray:
    key = np.asarray(df["YEAR"], dtype=int) * 1000 + np.asarray(df["DOY"], dtype=int)
    m = dict(zip(key.tolist(), np.asarray(df[col], dtype=float).tolist(), strict=True))
    return np.asarray([m.get(int(d), fill) for d in days], dtype=float)


@dataclass
class Treatment:
    """Inputs of one treatment for both configurations (host-side NumPy) and its M2 result."""

    exp: str
    trno: int
    out: Path
    params: Any
    forcing: Any
    m2_res: dict[str, np.ndarray]
    row: Any
    nl: int
    dlayr: np.ndarray
    replay_a: dict[str, Any]
    replay_b: dict[str, Any]
    water: dict[str, np.ndarray]
    sw0: np.ndarray
    snow0: float
    mulch0: float
    wbal: pd.DataFrame
    printed_sw_days: np.ndarray
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.exp}_t{self.trno:02d}"

    @property
    def n_days(self) -> int:
        return int(np.asarray(self.forcing.yrdoy).shape[0])


def build(exp: str, trno: int, out: Path, wb_out: Path, tables: Path) -> Treatment:
    """Parameters, forcing and the replays of one treatment; M2's own result on the same run."""
    p, f, res, row = m2.simulate(
        out, trno, daylength_from_output=exp in DAYLENGTH_FROM_OUTPUT, exp=exp, tables=tables
    )
    nl = int(p.soil.dlayr.shape[0])
    dl = np.asarray(p.soil.dlayr, dtype=float)
    days = np.asarray(f.yrdoy, dtype=int)
    n = len(days)
    with np.load(tables / f"{exp}_t{trno:02d}_spam.npz", allow_pickle=False) as z:
        meta = json.loads(str(z["meta.json"]))
        assert (
            meta["experiment"] == exp and int(meta["trno"]) == trno and meta["outputs_identical_to_reference"]
        )
        sp = {k.removeprefix("v."): np.asarray(z[k]) for k in z.files if k.startswith("v.")}
    pos = {int(d): i for i, d in enumerate(sp["YRDOY"].tolist())}
    at = np.asarray([pos.get(int(d), -1) for d in days])
    nxt = np.asarray([pos.get(_next_day(int(d)), -1) for d in days])
    have, have_next = at >= 0, nxt >= 0

    def daily(name: str, cols: slice | None = None) -> np.ndarray:
        v = sp[name].astype(np.float64)
        v = v[:, :nl] if cols is not None else v
        out_ = np.zeros((n, *v.shape[1:]))
        out_[have] = v[at[have]]
        return out_

    sw_print = np.asarray(f.sw, dtype=float)
    sw_full = sw_print.copy()
    sw_full[have_next] = sp["SW"][nxt[have_next], :nl].astype(np.float64)
    sw0 = sp["SW"][at[0], :nl].astype(np.float64) if have[0] else sw_print[0]
    # the canopy SPAM reads the next day is PLANT's end-of-day XHLAI / XLAI; the last day: PlantGro
    pg = read_plantgro(out / "PlantGro.OUT")
    pg = pg[pg["TRNO"] == trno]
    laid = _by_day(pg, "LAID", days)
    lai = laid.copy()
    tlai = laid.copy()
    lai[have_next] = sp["XHLAI"][nxt[have_next]].astype(np.float64)
    tlai[have_next] = sp["XLAI"][nxt[have_next]].astype(np.float64)
    et = read_out(out / "ET.OUT")
    et = et[et["TRNO"] == trno]
    wb = read_out(wb_out / "SoilWatBal.OUT")
    wb = wb[wb["TRNO"] == trno] if "TRNO" in wb.columns else wb
    wb = wb[wb["YEAR"].notna()]
    first = int(days[0])
    d0 = date(first // 1000, 1, 1) + timedelta(days=first % 1000 - 2)
    prev = d0.year * 1000 + d0.timetuple().tm_yday
    init = _by_day(wb, "SNOWD", np.asarray([prev])), _by_day(wb, "MWTD", np.asarray([prev]))
    col = {
        c: _by_day(wb, c, days) for c in ("PRED", "IRRD", "RESAD", "ROFD", "DRND", "SNOWD", "MWTD", "MEVAP")
    }
    col |= {c: _by_day(wb, c, days) for c in ZERO_COLUMNS if c in wb.columns}
    col["ESAD"], col["EPAD"] = _by_day(wb, "ESAD", days), _by_day(wb, "EPAD", days)
    col["SWTD"] = _by_day(wb, "SWTD", days)
    emaa = _by_day(et, "EMAA", days)
    one = np.ones((n, 1))
    canopy = {"lai": lai[:, None], "tlai": tlai[:, None], "height": np.zeros((n, 1))}
    common = {
        "soil": {
            "runoff": col["ROFD"],
            "drain": col["DRND"],
            "residue_water": col["RESAD"],
            "snow": col["SNOWD"],
            "mulch_wat": col["MWTD"],
        },
        "spam": {
            "eo": daily("EO"),
            "es": daily("ES") / MM_PER_CM,
            "em": emaa / MM_PER_CM,
            "eop": np.asarray(f.eop, dtype=float)[:, None] * one,
            "uptake": daily("RWU", slice(None)),
        },
        "canopy": canopy,
        "trwup": np.asarray(f.trwup, dtype=float)[:, None],
    }
    replay_a = _with_sw(common, sw_print)
    replay_b = _with_sw(common, sw_full)
    water = {"rain": col["PRED"], "irrigation": col["IRRD"]}
    t = Treatment(
        exp=exp,
        trno=trno,
        out=out,
        params=p,
        forcing=f,
        m2_res=res,
        row=row,
        nl=nl,
        dlayr=dl,
        replay_a=replay_a,
        replay_b=replay_b,
        water=water,
        sw0=sw0,
        snow0=float(init[0][0]),
        mulch0=float(init[1][0]),
        wbal=pd.DataFrame(col | {"EMAA": emaa, "TRWU": daily("TRWU"), "EP": daily("EP")}),
        printed_sw_days=np.nonzero(~have_next)[0],
    )
    t.notes = {
        "sw_printed_days_in_b": int((~have_next).sum()),
        "spam_days": int(have.sum()),
        "nonzero_zero_columns": sorted(c for c in ZERO_COLUMNS if c in col and np.any(col[c] != 0.0)),
        "mevap_vs_emaa_max_mm": float(np.max(np.abs(col["MEVAP"] - emaa))) if n else 0.0,
        "epad_vs_uptake_max_mm": float(
            np.max(np.abs(col["EPAD"] - daily("RWU", slice(None)).sum(1) * MM_PER_CM))
        ),
        "esad_vs_es_max_mm": float(np.max(np.abs(col["ESAD"] - daily("ES")))),
    }
    return t


def _with_sw(common: dict[str, Any], sw: np.ndarray) -> dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in common.items()}
    out["soil"] = dict(common["soil"]) | {"sw": sw}
    return out


# ------------------------------------------------------------------------ batched runs
def _pad(x: np.ndarray, n: int) -> np.ndarray:
    idx = np.minimum(np.arange(n), x.shape[0] - 1)
    return np.asarray(x)[idx]


def _stack(trees: list[Any]) -> Any:
    return jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *trees)


#: the replayed daily flows (``replay.<group>.<name>``) and the water inputs that a padding day
#: carries as zero, so the ledger books nothing on the days after a treatment's last day (its
#: stores, repeated, stay unchanged: the residual of a padding day is 0)
PAD_ZERO_FLOWS: tuple[tuple[str, str], ...] = (
    ("soil", "runoff"),
    ("soil", "drain"),
    ("soil", "residue_water"),
    ("spam", "es"),
    ("spam", "em"),
    ("spam", "uptake"),
)


def _zero_after(x: np.ndarray, n_days: int) -> np.ndarray:
    """``x`` padded by repeating the last day, with the padding days set to 0."""
    a = np.array(x, dtype=float)
    a[n_days:] = 0.0
    return a


def _forcing(t: Treatment, replay: dict[str, Any], n: int, *, nan_water: bool) -> dict[str, Any]:
    """The treatment's forcing padded to ``n`` days: every record repeats its last day, except the
    daily flows of :data:`PAD_ZERO_FLOWS` and the rain and irrigation, which are 0 on the padding
    days (never compared; a scan is causal, so the real days are unchanged)."""
    f = jax.tree.map(lambda x: jnp.asarray(_pad(np.asarray(x), n)), t.forcing)
    if nan_water:
        f = f.replace(
            sw=jnp.full_like(f.sw, jnp.nan),
            eop=jnp.full_like(f.eop, jnp.nan),
            trwup=jnp.full_like(f.trwup, jnp.nan),
        )
    padded = jax.tree.map(lambda x: _pad(np.asarray(x, dtype=float), n), replay)
    for group, name in PAD_ZERO_FLOWS:
        padded[group][name] = _zero_after(padded[group][name], t.n_days)
    rep = jax.tree.map(jnp.asarray, padded)
    rep["canopy"] = CanopyRecord(**rep["canopy"])
    water = {k: jnp.asarray(_zero_after(_pad(v, n), t.n_days)) for k, v in t.water.items()}
    soil = BucketForcing(rain=water["rain"], tmax=f.tmax, irrigation=water["irrigation"])
    return {"crop": f, "soil": soil, "replay": rep}


def real4(x: Any) -> jnp.ndarray:
    """The value DSSAT holds for a number it read into a REAL (float32), promoted exactly to float64."""
    return jnp.asarray(np.asarray(x, dtype=np.float32).astype(np.float64))


def _site(t: Treatment) -> DssatSite:
    """The soil of the replayed day: the crop's INP layers (the bucket, the soil evaporation and
    XTRACT are replayed; MESEV = R so the ledger books the replayed total ES)."""
    s = t.params.soil
    dl = np.asarray(s.dlayr, dtype=float)
    return DssatSite(
        dlayr=dl,
        ds=np.cumsum(dl),
        ll=np.asarray(s.ll, dtype=float),
        dul=np.asarray(s.dul, dtype=float),
        sat=np.asarray(s.sat, dtype=float),
        swcn=np.zeros(dl.size),
        cn=0.0,
        swcon=0.0,
        salb=0.0,
        u=0.0,
    )


def _params(t: Treatment) -> dict[str, Any]:
    """M2's crop parameters as they are (soil values as input); ROOTWU's soil layers as the REAL*4
    values ROOTWU compares the (REAL*4) soil water with: on a layer that ``XTRACT`` has dried to the
    lower limit, DSSAT's ``SW(L) .LE. LL(L)`` holds exactly in REAL*4, while the float64 ``LL`` of the
    INP number (0.026) lies 1e-9 below the REAL*4 ``SW`` (0.0260000005) and the layer would take up
    water. (The free day applies this as the model's convention, ``dssat_soil_values``.)"""
    p = t.params
    out = day_params(_site(t), p, ksevap=0.0, ktrans=0.0, soil_values="input")
    out["rootwu"] = RootwuParams(dlayr=real4(p.soil.dlayr), ll=real4(p.soil.ll), sat=real4(p.soil.sat))
    return out


def _state(t: Treatment) -> dict[str, Any]:
    sw0 = jnp.asarray(t.sw0)
    dl = jnp.asarray(t.dlayr)
    storage0 = float(np.sum(t.sw0 * t.dlayr) + (t.snow0 + t.mulch0) / MM_PER_CM)
    return initial_state(
        bucket=BucketState.initial(sw0, snow=t.snow0, mulch_wat=t.mulch0, dtype=jnp.float64),
        soil_evap=SoilEvapState.initial(sw0, dl, jnp.cumsum(dl), sw0, sw0, jnp.asarray(0.0)),
        crop=CeresMaizeState.initial(t.params, 1),
        rootwu=RootwuState.initial(1, t.nl, jnp.float64),
        salb=0.0,
        storage0=storage0,
    )


def run_group(ts: list[Treatment]) -> dict[str, dict[str, np.ndarray]]:
    """Both configurations and the stand-alone crop on the treatments ``ts`` (one layer count), batched."""
    n = max(t.n_days for t in ts)
    params = _stack([_params(t) for t in ts])
    state = _stack([_state(t) for t in ts])
    day = day_dssat486(SLOT)
    reps = replay_processes(SLOT)
    # A replays M2's printed soil water with fluxes of another precision: not a water balance, so
    # its ledger entry books nothing; B's replayed soil closes within LEDGER_ATOL_CM (checked under
    # AGRI_JAX_CHECK=1; the per-day print bounds are checked in compare)
    ledger_a = noop_entry("ledger.close", why="configuration A's replayed soil and fluxes are not a balance")
    model_a = day.compile(
        day_processes(
            SLOT,
            replace={
                **reps,
                f"water_supply.{SLOT}.rootwu": trwup_replay_entry(SLOT),
                "ledger.close": ledger_a,
            },
        ),
        outputs=day_outputs(SLOT),
        exact_lags=True,
    )
    model_b = day.compile(
        day_processes(SLOT, replace={**reps, "ledger.close": ledger_entry(atol=LEDGER_ATOL_CM, rtol=0.0)}),
        outputs=day_outputs(SLOT),
        exact_lags=True,
    )
    fa = _stack([_forcing(t, t.replay_a, n, nan_water=False) for t in ts])
    fb = _stack([_forcing(t, t.replay_b, n, nan_water=True) for t in ts])
    out_a = run_sites(model_a, params, fa, state)
    out_b = run_sites(model_b, params, fb, state)
    alone = run_sites(
        ceres_maize_model(),
        params["crop"],
        _stack([jax.tree.map(lambda x: jnp.asarray(_pad(np.asarray(x), n)), t.forcing) for t in ts]),
        _stack([CeresMaizeState.initial(t.params, 1) for t in ts]),
    )
    np_ = lambda d: {k: np.asarray(v) for k, v in d.items()}  # noqa: E731
    return {"a": np_(out_a), "b": np_(out_b), "alone": np_(alone)}


# ------------------------------------------------------------------------ comparisons
def _first(stage: np.ndarray, days: np.ndarray, code: int) -> int:
    hit = np.nonzero(stage == code)[0]
    return int(days[hit[0]]) if hit.size else -99


def _dates(t: Treatment, istage: np.ndarray) -> dict[str, list[int]]:
    days = np.asarray(t.forcing.yrdoy)
    out = {}
    for name, code, col in (("emergence", 1, "EDAT"), ("silking", 4, "ADAT"), ("maturity", 10, "MDAT")):
        ref = float(t.row[col])
        out[name] = [_first(istage, days, code), int(ref) if ref > 0 else -99]
    return out


def compare(
    t: Treatment, a: dict[str, np.ndarray], b: dict[str, np.ndarray], alone: dict[str, np.ndarray]
) -> dict[str, Any]:
    """One treatment's row of the report."""
    n = t.n_days
    ref = t.m2_res
    keys = sorted(ref)
    a_eq = {k: bool(np.array_equal(a[k][:n], ref[k])) for k in keys}
    alone_eq = {k: bool(np.array_equal(alone[k][:n], ref[k])) for k in keys}
    a_alone_eq = {k: bool(np.array_equal(a[k][:n], alone[k][:n])) for k in keys}
    res_b = {k: b[k][:n] for k in keys}
    errs_b = m2.daily_errors(t.out, t.trno, t.forcing, res_b)
    errs_m2 = m2.daily_errors(t.out, t.trno, t.forcing, ref)
    hwam = float(t.row["HWAM"])
    y_b, y_m2 = float(res_b["gwad"][-1, 0]), float(ref["gwad"][-1, 0])
    ok = all(errs_b[c]["max_rel_big"] < M2_REL_BIG for c in ("LAID", "CWAD"))
    ok &= all(errs_b[c]["max_units"] <= M2_UNITS_ALL for c in M2_UNIT_COLUMNS)
    ok &= errs_b["GSTD"]["max_units"] == 0.0 and errs_b["L#SD"]["max_units"] == 0.0
    if hwam > 0:
        ok &= abs(y_b - hwam) / hwam < M2_YIELD_REL
    # TRWUP: ours (P1) against the SPAM dump (the reference's ROOTWU on its own SW and RLV)
    tr_b = b["p1_trwup"][:n, 0]
    tr_ref = np.asarray(t.forcing.trwup, dtype=float)
    pos = tr_ref > 0
    rel = np.abs(tr_b[pos] - tr_ref[pos]) / tr_ref[pos]
    # the ledger of the replayed soil
    wb = t.wbal
    resid_mm = b["residual"][:n] * MM_PER_CM
    printed = [c for c in ("IRRD", "RESAD", "ROFD", "DRND") if np.any(wb[c] != 0.0)]
    stores = [c for c in ("SNOWD", "MWTD") if np.any(wb[c] != 0.0)]
    bound = (
        PRINT_HALF_MM * (len(printed) + 2 * len(stores))
        + EMAA_HALF_MM * float(np.any(wb["EMAA"] != 0.0))
        + SW_ROUND * float(t.dlayr.sum()) * MM_PER_CM
        + REAL_SLACK_MM
    )
    # the days whose end-of-day SW is the printed one (no SPAM call the next day): its F8.3 half-step
    days_ = np.asarray(t.forcing.yrdoy)
    bounds = np.full(n, bound)
    bounds[t.printed_sw_days] += SW_PRINT_HALF * float(t.dlayr.sum()) * MM_PER_CM
    full = np.ones(n, dtype=bool)
    full[t.printed_sw_days] = False
    return {
        "treatment": t.key,
        "n_days": n,
        "n_layer": t.nl,
        "a_bit_identical_to_m2": all(a_eq.values()),
        "a_columns_not_identical": sorted(k for k, v in a_eq.items() if not v),
        "standalone_batched_bit_identical_to_m2": all(alone_eq.values()),
        "a_bit_identical_to_standalone_batched": all(a_alone_eq.values()),
        "b_meets_m2_criteria": bool(ok),
        "b_max_units": {c: errs_b[c]["max_units"] for c in (*M2_UNIT_COLUMNS, "G#AD", "GSTD", "L#SD")},
        "b_max_rel_big": {c: errs_b[c]["max_rel_big"] for c in ("LAID", "CWAD")},
        "m2_max_units": {c: errs_m2[c]["max_units"] for c in (*M2_UNIT_COLUMNS, "G#AD", "GSTD", "L#SD")},
        "m2_max_rel_big": {c: errs_m2[c]["max_rel_big"] for c in ("LAID", "CWAD")},
        "hwam": hwam,
        "yield_b": y_b,
        "yield_m2": y_m2,
        "yield_b_rel": abs(y_b - hwam) / hwam if hwam > 0 else None,
        "dates_b": _dates(t, b["istage"][:n, 0]),
        "dates_m2": _dates(t, ref["istage"][:, 0]),
        "b_vs_m2_max_abs": {
            k: float(np.max(np.abs(res_b[k] - ref[k]))) for k in ("lai", "cwad", "gwad", "rwad", "rdpd")
        },
        "trwup_days": int(pos.sum()),
        "trwup_rel_max": float(rel.max()) if rel.size else 0.0,
        "trwup_rel_median": float(np.median(rel)) if rel.size else 0.0,
        "trwup_zero_mismatch_days": int(np.sum((tr_ref == 0) != (tr_b == 0))),
        "ledger_max_abs_residual_mm": float(np.max(np.abs(resid_mm[full]))),
        "ledger_bound_mm": bound,
        "ledger_printed_sw_days": [int(days_[k]) for k in t.printed_sw_days],
        "ledger_printed_sw_day_residual_mm": [float(resid_mm[k]) for k in t.printed_sw_days],
        "ledger_printed_sw_day_bound_mm": [float(bounds[k]) for k in t.printed_sw_days],
        "ledger_closure_mm": float(np.sum(resid_mm)),
        "ledger_ok": bool(np.all(np.abs(resid_mm) <= bounds)),
        **t.notes,
    }


def evaluate(keys: list[tuple[str, int]], data_dir: Path, work: Path, jobs: int) -> list[dict[str, Any]]:
    """Reference runs, the two configurations batched by layer count, and the per-treatment rows."""
    tables = tables_dir(data_dir)
    runs = run_references(keys, work, jobs)
    diffs = {
        k: [name for name in SAME_FILES if _strip(o / name) != _strip(w / name)] for k, (o, w) in runs.items()
    }
    ts = [build(exp, t, *runs[(exp, t)], tables) for exp, t in keys]
    rows: dict[str, dict[str, Any]] = {}
    for nl in sorted({t.nl for t in ts}):
        group = [t for t in ts if t.nl == nl]
        out = run_group(group)
        for i, t in enumerate(group):
            pick = {c: {k: v[i] for k, v in out[c].items()} for c in out}
            row = compare(t, pick["a"], pick["b"], pick["alone"])
            row["detailed_run_differs_in"] = diffs[(t.exp, t.trno)]
            rows[t.key] = row
    return [rows[f"{e}_t{t:02d}"] for e, t in keys]


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over the treatments."""
    return {
        "n_treatments": len(rows),
        "a_bit_identical_to_m2": sum(r["a_bit_identical_to_m2"] for r in rows),
        "a_bit_identical_to_standalone_batched": sum(
            r["a_bit_identical_to_standalone_batched"] for r in rows
        ),
        "standalone_batched_bit_identical_to_m2": sum(
            r["standalone_batched_bit_identical_to_m2"] for r in rows
        ),
        "b_meets_m2_criteria": sum(r["b_meets_m2_criteria"] for r in rows),
        "b_max_units_worst": {c: max(r["b_max_units"][c] for r in rows) for c in rows[0]["b_max_units"]},
        "m2_max_units_worst": {c: max(r["m2_max_units"][c] for r in rows) for c in rows[0]["m2_max_units"]},
        "b_max_rel_big_worst": {c: max(r["b_max_rel_big"][c] for r in rows) for c in ("LAID", "CWAD")},
        "yield_b_rel_worst": max((r["yield_b_rel"] or 0.0) for r in rows),
        "dates_b_equal_reference": sum(all(v[0] == v[1] for v in r["dates_b"].values()) for r in rows),
        "dates_m2_equal_reference": sum(all(v[0] == v[1] for v in r["dates_m2"].values()) for r in rows),
        "trwup_rel_max_worst": max(r["trwup_rel_max"] for r in rows),
        "trwup_rel_median_worst": max(r["trwup_rel_median"] for r in rows),
        "trwup_zero_mismatch_days": sum(r["trwup_zero_mismatch_days"] for r in rows),
        "ledger_ok": sum(r["ledger_ok"] for r in rows),
        "ledger_max_abs_residual_mm_worst": max(r["ledger_max_abs_residual_mm"] for r in rows),
        "ledger_bound_mm_min": min(r["ledger_bound_mm"] for r in rows),
        "detailed_run_differs": sum(bool(r["detailed_run_differs_in"]) for r in rows),
        "nonzero_zero_columns": sorted({c for r in rows for c in r["nonzero_zero_columns"]}),
        "sw_printed_days_in_b": sum(r["sw_printed_days_in_b"] for r in rows),
    }


def write_report(rows: list[dict[str, Any]], data_dir: Path, name: str = "d2_0_day_dssat486.json") -> Path:
    out = data_dir / "validation" / "aj_dday"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    engine = Path(os.environ.get("AGRI_JAX_DSSAT", str(m2.DSSAT_ENGINE)))
    path.write_text(
        json.dumps(
            {
                "step": "DSSAT day, skeleton",
                "reference": "DSSAT-CSM v4.8.6.0 dscsm048 (build486), nitrogen off",
                "engine": str(engine),
                "x64": bool(jax.config.jax_enable_x64),
                "summary": summary(rows),
                "treatments": rows,
            },
            indent=1,
        )
    )
    return path
