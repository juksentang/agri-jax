"""SPAM partition and soil / mulch evaporation against the instrumented dscsm048, helpers.

Helper module of ``test_spam_evap_dssat.py`` and a script (``python
tests/integration/spam_evap_dssat.py``) that prints the per-quantity error table.

Data (kept outside the repository): ``<data>/validation/aj_det/tables`` (written by the collection
script on rorqual): a gfortran 13 build of the DSSAT-CSM v4.8.6.0 BSD-3 source, instrumented at
entry and exit of ``SPAM``, ``SOILEV``, ``ESR_SoilEvap`` and ``MULCH_EVAP`` (:mod:`agrijax.port.
instrument`), ran the 58 maize treatments of M2 (``test_ceres_dssat.run_reference``: nitrogen off,
one-treatment batch) and the 7 nitrogen-off seasons of the CA-TPA DSSAT case (treatments 1-7 of
``CTPA1501.MZX``, :mod:`agrijax.sites.catpa_dssat`); every ``*.OUT`` of each instrumented run equals
the build486 reference run (``collect_report.json``). One table per routine and phase with the
RATE call of every simulated day (``<KEY>_{spam_in,spam_out,soilev_in,soilev_out,soilev_init,
esr_in,esr_out,mulch_in,mulch_out}.npz``); ``_runs/<KEY>/`` holds the run's printed outputs.

Tolerances: the reference computes in REAL*4. Every kernel here is driven with the dumped float32
inputs promoted to float64, and the result is compared with the dumped float32 output. The
reference-side limit of a quantity is its REAL*4 rounding: ``n`` rounded operations on the longest
dependency path of the Fortran expression, each at most one unit roundoff ``2**-24`` of the
largest magnitude on that path (the "scale"), plus the half-ulp rounding of the dumped result.
:data:`OPS` lists ``n`` per quantity (counted from the Fortran statements) and :func:`scale_of`
the scale; a day passes when ``|ours - ref| <= OPS * U32 * scale``. Branch decisions of the
reference (``IF``) are taken from the dumped state where the float32 and float64 values of a
compared quantity straddle a threshold (none found: the counts are reported).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

TABLES_SUBDIR = Path("validation") / "aj_det" / "tables"
#: unit roundoff of REAL*4
U32 = 2.0**-24
#: rounded REAL*4 operations on the longest dependency path of each compared quantity (the Fortran
#: statements cited), plus one for the rounding of the dumped result
OPS: dict[str, int] = {
    # PETPT: TD (3), ALBEDO (4, EXP), SLANG (1), EEQ (5), EO (1-4), MAX; PET.for:895-913
    "EO": 16,
    # PSE: -KSEVAP*XLAI, EXP, *EO, MAX; PET.for:1482-1493
    "EOS": 5,
    # MULCH_EVAP: MAI (3), EXP (2), EOM (2), MIN/MAX, *COVER, EOS3 (6), then SPAM EOS3 - EM (1)
    "EM": 10,
    "EOS_SOIL": 14,
    # SOILEV: ESUP (4) or stage 2 (sqrt, 5), AWEV1 (4), SWMIN (4), limits; SOILEV.for:96-174
    "ES": 16,
    "SUMES1": 12,
    "SUMES2": 12,
    "T": 16,
    # ESR: SWTEMP (2), MEANDEP (2), coefficient (pow: 3), SWDELTU (3), ES_LYR (3), sum over the
    # layers (NLAYR additions, bounded by 20), reduction (3), UPFLOW (NLAYR additions)
    "ES_LYR": 20,
    "SWDELTU": 16,
    "UPFLOW": 24,
    "ESR_ES": 28,
    # TRATIO (VPSAT 5, VPSLOP 7, DELTA 1, LHV 2, GAMMA 3, resistances 8, ratio 7) and TRANS
    # (FDINT 3, EOP 4, EOP_max 3, MIN/MAX); TRANS.for:82-142, 211-286, HMET.for:643, 677
    "TRAT": 32,
    "EOP": 40,
}


def tables_dir(data_dir: Path | str) -> Path:
    return Path(data_dir) / TABLES_SUBDIR


@dataclass
class Run:
    """The daily tables of one reference run (float32 values as dumped)."""

    key: str
    spam_in: dict[str, np.ndarray]
    spam_out: dict[str, np.ndarray]
    dates: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)

    @property
    def mesev(self) -> str:
        return _chr(self.spam_in["MESEV"][0])

    @property
    def nlayr(self) -> int:
        return int(self.spam_in["NLAYR_S"][0])


def _chr(a: np.ndarray) -> str:
    return np.asarray(a).tobytes().decode(errors="replace").strip("\x00 ")


def run_keys(tables: Path) -> list[str]:
    return sorted(p.name.removesuffix("_spam_out.npz") for p in tables.glob("*_spam_out.npz"))


def load_run(tables: Path, key: str) -> Run:
    from agrijax.port import dumps

    si, _ = dumps.load_table(tables / f"{key}_spam_in.npz")
    so, meta = dumps.load_table(tables / f"{key}_spam_out.npz")
    assert np.array_equal(si.date, so.date), key
    r = Run(key, si.values, so.values, so.date, extra={"meta": meta})
    for suf in ("soilev_in", "soilev_out", "soilev_init", "esr_in", "esr_out", "mulch_in", "mulch_out"):
        p = tables / f"{key}_{suf}.npz"
        if p.is_file():
            r.tables[suf] = dumps.load_table(p)[0]
    return r


def load_runs(tables: Path, keys: Iterable[str] | None = None) -> list[Run]:
    return [load_run(tables, k) for k in (run_keys(tables) if keys is None else keys)]


def f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def ratio(ours: Any, ref: Any, scale: Any) -> np.ndarray:
    """``|ours - ref|`` in units of ``U32 * scale`` (``scale`` floored at the smallest normal)."""
    s = np.maximum(np.abs(np.asarray(scale, dtype=np.float64)), np.finfo(np.float32).tiny)
    return np.abs(np.asarray(ours, dtype=np.float64) - np.asarray(ref, dtype=np.float64)) / (U32 * s)


def cat(runs: list[Run], get) -> np.ndarray:
    parts = [get(r) for r in runs]
    parts = [p for p in parts if p is not None and len(p)]
    return np.concatenate(parts) if parts else np.zeros(0)


# ------------------------------------------------------------------------ per-quantity checks
def check_eo(runs: list[Run]) -> dict[str, np.ndarray]:
    """``PETPT`` (``pet/priestley_taylor``) on the inputs ``PET`` received (WEATHER, XHLAI, ET_ALB)."""
    from agrijax.processes.pet import priestley_taylor

    v = {k: cat(runs, lambda r, k=k: r.spam_in[k]) for k in ("SRAD_W", "TMAX_W", "TMIN_W", "XHLAI")}
    alb = cat(runs, lambda r: r.spam_out["ET_ALB_L"])
    ref = cat(runs, lambda r: r.spam_out["EO"])
    eo = np.asarray(
        priestley_taylor(f64(v["SRAD_W"]), f64(v["TMAX_W"]), f64(v["TMIN_W"]), f64(v["XHLAI"]), f64(alb))
    )
    return {"EO": ratio(eo, ref, ref)}


def check_eos(runs: list[Run]) -> dict[str, np.ndarray]:
    """``PSE`` on the dumped ``EO``, ``XLAI``, ``KSEVAP``."""
    from agrijax.processes.pet.spam_dssat import potential_soil_evaporation

    eo = cat(runs, lambda r: r.spam_out["EO"])
    xlai = cat(runs, lambda r: r.spam_in["XLAI"])
    ks = cat(runs, lambda r: r.spam_in["KSEVAP"])
    ref = cat(runs, lambda r: r.spam_out["EOS"])
    eos = np.asarray(potential_soil_evaporation(f64(eo), f64(xlai), f64(ks)))
    return {"EOS": ratio(eos, ref, eo)}


def check_mulch(runs: list[Run]) -> dict[str, np.ndarray]:
    """``MULCH_EVAP`` on its dumped entry (``EM``, returned ``EOS``) and SPAM's mulch step on the
    SPAM entry / exit (``EM``, ``EOS_SOIL``)."""
    from agrijax.processes.soil_water.bucket_evap import mulch_evaporation, spam_mulch_step

    def g(tab: str, name: str):
        return cat(runs, lambda r: r.tables[tab].values[name] if tab in r.tables else None)

    mi = {
        n: g("mulch_in", n)
        for n in (
            "EOS",
            "MULCH%MULCHMASS",
            "MULCH%MULCHCOVER",
            "MULCH%MULCHWAT",
            "MULCH%MULCH_AM",
            "MULCH%MUL_EXTFAC",
        )
    }
    mo = {n: g("mulch_out", n) for n in ("EM", "EOS")}
    r = mulch_evaporation(
        f64(mi["EOS"]),
        f64(mi["MULCH%MULCHMASS"]),
        f64(mi["MULCH%MULCHCOVER"]),
        f64(mi["MULCH%MULCHWAT"]),
        f64(mi["MULCH%MULCH_AM"]),
        f64(mi["MULCH%MUL_EXTFAC"]),
    )
    out = {"MULCH_EM": ratio(r.em, mo["EM"], mi["EOS"]), "MULCH_EOS": ratio(r.eos, mo["EOS"], mi["EOS"])}
    si = {
        n: cat(runs, lambda rr, n=n: rr.spam_in[n])
        for n in ("MULCHMASS", "MULCHCOVER", "MULCHWAT", "MULCH_AM", "MUL_EXTFAC")
    }
    eos = cat(runs, lambda rr: rr.spam_out["EOS"])
    s = spam_mulch_step(f64(eos), *(f64(si[n]) for n in si), mulch_active=True)
    em_ref = cat(runs, lambda rr: rr.spam_out["EM_L"])
    eos_soil_ref = cat(runs, lambda rr: rr.spam_out["EOS_SOIL_L"])
    out["EM"] = ratio(s.em, em_ref, eos)
    out["EOS_SOIL"] = ratio(s.eos, eos_soil_ref, eos)
    out["_mulch_days"] = np.asarray([int((si["MULCHMASS"] > 0.1).sum())])
    return out


def check_soilev(runs: list[Run]) -> dict[str, np.ndarray]:
    """``SOILEV`` RATE on its dumped entry (store, EOS, WINF, SW, LL, DLAYR, SW_AVAIL, U,
    PMFRACTION) against its exit; ``SEASINIT`` against the init table; SPAM's ``SW_AVAIL(1)``."""
    from agrijax.processes.soil_water.bucket_evap import SoilevStore, soilev_init, soilev_rate

    rr = [r for r in runs if "soilev_in" in r.tables]
    i = {
        n: cat(rr, lambda r, n=n: r.tables["soilev_in"].values[n])
        for n in ("SUMES1", "SUMES2", "T", "SWEF", "EOS", "WINF", "SW_AVAIL", "U", "PMFRACTION")
    }
    sw1 = cat(rr, lambda r: r.tables["soilev_in"].values["SW"][:, 0])
    ll1 = cat(rr, lambda r: r.tables["soilev_in"].values["LL"][:, 0])
    dl1 = cat(rr, lambda r: r.tables["soilev_in"].values["DLAYR"][:, 0])
    o = {n: cat(rr, lambda r, n=n: r.tables["soilev_out"].values[n]) for n in ("ES", "SUMES1", "SUMES2", "T")}
    store = SoilevStore(f64(i["SUMES1"]), f64(i["SUMES2"]), f64(i["T"]), f64(i["SWEF"]))
    es, new = soilev_rate(
        store,
        f64(i["EOS"]),
        f64(i["WINF"]),
        f64(sw1),
        f64(ll1),
        f64(dl1),
        f64(i["SW_AVAIL"]),
        f64(i["U"]),
        f64(i["PMFRACTION"]),
    )
    sc = np.max(
        np.abs(
            np.stack(
                [
                    i["EOS"],
                    i["SUMES1"],
                    i["SUMES2"],
                    i["U"],
                    i["WINF"],
                    o["SUMES1"],
                    o["SUMES2"],
                    3.5 * np.sqrt(np.maximum(o["T"], 0.0)),
                ]
            )
        ),
        axis=0,
    )
    out = {
        "ES": ratio(es, o["ES"], sc),
        "SUMES1": ratio(new.sumes1, o["SUMES1"], sc),
        "SUMES2": ratio(new.sumes2, o["SUMES2"], sc),
        # T = (SUMES2 / 3.5)**2: relative to T itself (twice the relative error of SUMES2)
        # T = (SUMES2 / 3.5)**2 carries the error of SUMES2 times dT/dSUMES2 = 2 sqrt(T) / 3.5
        "T": ratio(
            new.t, o["T"], np.maximum(np.abs(o["T"]), 2.0 * np.sqrt(np.maximum(o["T"], 0.0)) * sc / 3.5)
        ),
    }
    # SW_AVAIL(1) = MAX(0, SW(1) + SWDELTS(1) + SWDELTU(1)) (SPAM.for:361-363), from SPAM's entry
    days = [r.tables["soilev_in"].date for r in rr]
    sa = []
    for r, d in zip(rr, days, strict=True):
        j = np.searchsorted(r.dates, d)
        s = f64(r.spam_in["SW"][j, 0]) + f64(r.spam_in["SWDELTS"][j, 0]) + f64(r.spam_in["SWDELTU"][j, 0])
        sa.append(np.maximum(s, 0.0))
    out["SW_AVAIL"] = ratio(np.concatenate(sa), i["SW_AVAIL"], i["SW_AVAIL"])
    # SEASINIT
    ini = [r for r in rr if "soilev_init" in r.tables and len(r.tables["soilev_init"])]
    if ini:
        v = {
            n: cat(ini, lambda r, n=n: r.tables["soilev_init"].values[n])
            for n in ("SUMES1", "SUMES2", "T", "SWEF", "U")
        }
        sw = cat(ini, lambda r: r.tables["soilev_init"].values["SW"][:, 0])
        ll = cat(ini, lambda r: r.tables["soilev_init"].values["LL"][:, 0])
        du = cat(ini, lambda r: r.tables["soilev_init"].values["DUL"][:, 0])
        dl = cat(ini, lambda r: r.tables["soilev_init"].values["DLAYR"][:, 0])
        st = soilev_init(f64(sw), f64(ll), f64(du), f64(dl), f64(v["U"]))
        scale = np.maximum(np.abs(v["U"]), (f64(du) - f64(sw)) * f64(dl) * 10.0)
        out["INIT_SUMES1"] = ratio(st.sumes1, v["SUMES1"], scale)
        out["INIT_SUMES2"] = ratio(st.sumes2, v["SUMES2"], scale)
        out["INIT_T"] = ratio(st.t, v["T"], np.maximum(np.abs(v["T"]), 1.0))
        out["INIT_SWEF"] = ratio(st.swef, v["SWEF"], v["SWEF"])
    return out


def check_esr(runs: list[Run]) -> dict[str, np.ndarray]:
    """``ESR_SoilEvap`` on its dumped entry against its exit, grouped by layer count."""
    from agrijax.processes.soil_water.bucket_evap import esr_soil_evaporation

    rr = [r for r in runs if "esr_in" in r.tables]
    acc: dict[str, list[np.ndarray]] = {k: [] for k in ("ESR_ES", "ES_LYR", "SWDELTU", "UPFLOW", "PROFILE")}
    for nl in sorted({r.nlayr for r in rr}):
        g = [r for r in rr if r.nlayr == nl]

        def lay(name: str, tab: str, g=g, nl=nl) -> np.ndarray:
            return cat(g, lambda x: x.tables[tab].values[name][:, :nl])

        eos = cat(g, lambda r: r.tables["esr_in"].values["EOS"])
        res = esr_soil_evaporation(
            f64(eos),
            f64(lay("SW", "esr_in")),
            f64(lay("SWDELTS", "esr_in")),
            f64(lay("DLAYR_S", "esr_in")),
            f64(lay("DS_S", "esr_in")),
            f64(lay("DUL_S", "esr_in")),
            f64(lay("LL_S", "esr_in")),
            f64(cat(g, lambda r: r.tables["esr_out"].values["PMFRACTION"])),
        )
        es_ref = cat(g, lambda r: r.tables["esr_out"].values["ES"])
        lyr_ref = lay("ES_LYR", "esr_out")
        sc = np.maximum(np.abs(eos), np.abs(es_ref))
        acc["ESR_ES"].append(ratio(res.es, es_ref, sc))
        acc["ES_LYR"].append(ratio(res.es_lyr, lyr_ref, sc[:, None]).max(axis=-1))
        dl = f64(lay("DLAYR_S", "esr_in"))
        acc["SWDELTU"].append(
            ratio(res.swdeltu, lay("SWDELTU", "esr_out"), (sc[:, None] / (10.0 * dl))).max(axis=-1)
        )
        acc["UPFLOW"].append(ratio(res.upflow, lay("UPFLOW", "esr_out"), sc[:, None] / 10.0).max(axis=-1))
        prof = cat(g, lambda r: r.tables["esr_out"].values["PROFILETYPE"])
        acc["PROFILE"].append((np.asarray(res.profile) != prof).astype(float))
    return {k: np.concatenate(v) for k, v in acc.items() if v}


def check_trans(runs: list[Run]) -> dict[str, np.ndarray]:
    """``TRATIO`` / ``TRANS`` on the dumped ``EO``, ``XHLAI``, ``KTRANS``, weather and ``EVAP``."""
    from agrijax.processes.pet.spam_dssat import potential_transpiration, transpiration_ratio

    i = {
        n: cat(runs, lambda r, n=n: r.spam_in[n]) for n in ("XHLAI", "KTRANS", "CO2_W", "TAVG_W", "WINDSP_W")
    }
    o = {n: cat(runs, lambda r, n=n: r.spam_out[n]) for n in ("EO", "EOP", "EVAP_L")}
    trat = transpiration_ratio(
        f64(i["CO2_W"]), f64(i["TAVG_W"]), f64(i["WINDSP_W"]), f64(i["XHLAI"]), c4=True
    )
    eop = potential_transpiration(f64(o["EO"]), f64(i["XHLAI"]), f64(i["KTRANS"]), trat, f64(o["EVAP_L"]))
    cap = (
        f64(o["EO"])
        - f64(o["EO"]) * (1 - np.exp(-f64(i["KTRANS"]) * f64(i["XHLAI"]))) * (1 - np.asarray(trat))
        - f64(o["EVAP_L"])
    )
    uncapped = f64(o["EO"]) * (1 - np.exp(-f64(i["KTRANS"]) * f64(i["XHLAI"]))) * np.asarray(trat)
    return {
        "EOP": ratio(eop, o["EOP"], o["EO"]),
        "_cap_binding_days": np.asarray([int(((cap < uncapped) & (f64(i["XHLAI"]) > 1e-6)).sum())]),
        "_trat_ne_1_days": np.asarray([int((np.abs(np.asarray(trat) - 1) > 1e-7).sum())]),
    }


def summary(results: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out = {}
    for k, a in results.items():
        a = np.asarray(a, dtype=float)
        if k.startswith("_"):
            out[k] = {"value": float(a.sum())}
            continue
        lim = OPS.get(k.replace("MULCH_", "").replace("INIT_", ""), np.nan)
        out[k] = {
            "n": float(a.size),
            "max_ulp_of_scale": float(a.max(initial=0.0)),
            "p99": float(np.quantile(a, 0.99)) if a.size else 0.0,
            "ops": float(lim),
            "n_beyond_ops": float((a > lim).sum()) if np.isfinite(lim) else float("nan"),
        }
    return out


def main() -> int:
    import jax

    jax.config.update("jax_enable_x64", True)
    data = Path(os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")).expanduser()
    t = tables_dir(data)
    runs = load_runs(t)
    res: dict[str, np.ndarray] = {}
    for f in (check_eo, check_eos, check_mulch, check_soilev, check_esr, check_trans):
        res.update(f(runs))
    s = summary(res)
    print(json.dumps(s, indent=1))
    (data / "validation" / "aj_det" / "measure.json").write_text(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
