"""The canopy record (P6) against the RZWQM2 4.6 reference, CA-TPA 2015-2021 day by day.

The PET of RZWQM2 reads the ``LAI``, ``TLAI`` and ``HEIGHT`` that the plant manager ``MAPLNT``
left at the end of the previous day. This test runs the canopy entry
(``crop/ceres_maize.canopy@rzwqm2-4.6:faithful``) and the harvest reset
(``crop/ceres_maize.harvest@rzwqm2-4.6:faithful``) in the contract's order over 2015-001 to
2021-365 in one ``scan``, on the crop state of the reference itself: a replay entry writes the
embedded CERES's end-of-day ``XHLAI``, ``BIOMAS``, ``GRNWT``, ``EARS`` (DSSATDRV exit) and
``MDATE`` (MZ_GROSUB exit) into the crop state (zeros and ``MDATE = -99`` between seasons, the
``SEASINIT`` values). A PET stand-in (``pet.sw_daily``, in the ``physcl`` phase before the crop) copies the
canopy record it reads into its own subtree. The per-season table has the reference's planting population and planned harvest
date; ``HTMAX`` and ``BIOHALF`` are the cultivar file's (as the reference reads them, REAL*4).

Asserted:

* reference facts: the ``PHYSCL`` entry ``LAI``, ``TLAI``, ``HEIGHT`` of day ``t`` equal the
  ``MAPLNT`` exit of day ``t - 1`` on all 3287 days, ``TLAI == LAI`` on every day, all three are 0
  on the harvest days and between seasons; the dumped ``PLHGHT`` and ``PLALFA`` are those of the
  cultivar file's ``HTMAX`` and ``BIOHALF``;
* the canopy record at the end of every day equals the ``MAPLNT`` exit of that day: 0 exactly on
  harvest days and between seasons; ``lai`` within one REAL*4 rounding of the reference (the
  reference rounds the declined LAI to REAL*4), ``height`` within the REAL*4 rounding of the two
  single-precision operations of the stalk mass (``BIOMAS*10``, the subtraction), propagated to
  the height (``d ln h / d ln S <= 1``) and through the running maximum; ``tlai == lai``;
* the post-maturity decline is exercised: 27 days of 2020-2021 with ``YRDOY > MDATE`` and
  ``HDATE > MDATE``, on which the record differs from the green LAI;
* what the PET stand-in reads on day ``t`` equals the reference ``PHYSCL`` entry of day ``t``
  (same tolerances), and ``Day.check`` reports the contract's lag ``pet.sw_daily`` ->
  ``iface.canopy.maize`` as used.

Measured and recorded (``$AGRI_JAX_DATA/validation/aj_w4/canopy_catpa.json``, printed with
``-s``): the largest differences, absolute and relative, of LAI and height.

Data (kept outside the repository, ``allow_skip``): the RZWQM2 dump tables and scenario inputs of
``test_day_rzwqm46_smoke.py`` (whose ``Reference`` builds the CA-TPA crop parameters).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import Day, Lag, Phase, bind, compose, process, run
from agrijax.core.day import snapshot
from agrijax.core.events import EventTable
from agrijax.core.ports import Binding
from agrijax.core.state import get_path, set_path
from agrijax.iface.contract import phased_writes
from agrijax.iface.crop import CanopyRecord
from agrijax.io.dssat import read_cul
from agrijax.processes.crop.ceres_maize import (
    CanopyCoefficients,
    CeresCanopyParams,
    CeresForcing,
    CeresMaizeState,
    CeresSeasons,
    ceres_canopy,
    ceres_harvest,
)
from agrijax.sites.dssat_inputs import yrdoy_range

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the RZWQM2 full-state dump tables of CA-TPA 2015-2023"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="the reference comparison is float64"),
]

FIRST, LAST = 2015001, 2021365
SLOT = "maize"
OWN = f"crops.{SLOT}"
CANOPY = f"iface.canopy.{SLOT}"
SEEN = "surface.pet.canopy_seen"
#: the day's canopy record, recorded before the harvest reset (the daily output record)
DAY_CANOPY = "prev.day.canopy"
HARVEST_PORTS = {"root_out": f"iface.root.{SLOT}", "canopy_out": CANOPY}
#: the ports of the initial crop state that carry a record (bound to their global paths)
BOUND = (
    ("water_in", f"iface.crop_water.{SLOT}"),
    ("root_out", HARVEST_PORTS["root_out"]),
    ("snow_in", "iface.snow"),
)
REPORT = "validation/aj_w4/canopy_catpa.json"
#: one REAL*4 rounding (half an ulp relative to the value)
U32 = 2.0**-24
#: float64 slack of the comparison (a few ulp of our own evaluation)
U64 = 8 * 2.0**-52
#: the post-maturity decline days of the reference (2020: MDATE 2020281, HDATE 2020299; 2021:
#: 2021284, 2021293), counted from the dump tables
N_DECLINE = 27

DAY = Day(
    ref="rzwqm2-4.6",
    bare=True,  # part of the contract's day with stand-ins
    phases=(
        Phase("physcl", ("pet.sw_daily",)),
        Phase("plant", (f"{OWN}.crop_replay", f"{OWN}.canopy", "prev.day_output", f"{OWN}.harvest")),
    ),
    lags=(
        Lag(
            "pet.sw_daily",
            CANOPY,
            evidence="dumps: PHYSCL entry LAI, TLAI, HEIGHT of day t = MAPLNT exit of day t - 1 (3287 days)",
        ),
    ),
    # the harvest's bare-soil write of P6 is the crop's declared season end
    phased_writes=tuple(w for w in phased_writes(SLOT) if (w.entry, w.path) == (f"{OWN}.harvest", CANOPY)),
)


def _crop_replay(state: CeresMaizeState, params: Any, forcing_t: Any) -> CeresMaizeState:
    """Write the reference crop's end-of-day XHLAI, BIOMAS, GRNWT, EARS and MDATE into the state.

    Source: RZWQM2 dump tables (DSSATDRV exit, MZ_GROSUB exit) of the 4.6 CA-TPA run."""
    g, ph = state.growth, state.phen
    return state.replace(
        growth=g.replace(
            lai=forcing_t["lai"] * jnp.ones_like(g.lai),
            biomas=forcing_t["biomas"] * jnp.ones_like(g.biomas),
            grnwt=forcing_t["grnwt"] * jnp.ones_like(g.grnwt),
        ),
        phen=ph.replace(
            ears=forcing_t["ears"] * jnp.ones_like(ph.ears),
            mdate=forcing_t["mdate"] * jnp.ones_like(ph.mdate),
        ),
    )


crop_replay = process(
    _crop_replay,
    reads=(),
    writes=("growth.lai", "growth.biomas", "growth.grnwt", "phen.ears", "phen.mdate"),
    name="crop_replay",
    register=False,
    source="replay of the reference crop state (RZWQM2 dump tables)",
)


def _pet_standin(state: Any, params: Any, forcing_t: Any) -> Any:
    """What the PET entry reads: the canopy record at the start of the day, copied aside.

    Source: coupling contract, port P6 (PET reads the canopy with a one-day lag)."""
    return set_path(state, SEEN, get_path(state, CANOPY))


pet_standin = process(
    _pet_standin,
    reads=(CANOPY,),
    writes=(SEEN,),
    name="pet_standin",
    register=False,
    source="stand-in of the PET entry's read of P6",
)


def _model() -> Any:
    procs = {
        "pet.sw_daily": pet_standin,
        f"{OWN}.crop_replay": bind(crop_replay, own=OWN, forcing="rep"),
        f"{OWN}.canopy": bind(
            ceres_canopy, own=OWN, ports={"canopy_out": CANOPY}, params="crop", forcing="crop"
        ),
        # the daily output record: the day's canopy record before the harvest's bare-soil reset
        "prev.day_output": snapshot("prev.day_output", {CANOPY: DAY_CANOPY}),
        f"{OWN}.harvest": bind(ceres_harvest, own=OWN, ports=HARVEST_PORTS, params="crop", forcing="events"),
    }

    def outputs(s: Any, p: Any, f: Any) -> dict[str, Any]:
        return {
            "end": get_path(s, CANOPY),
            "seen": get_path(s, SEEN),
            "season": get_path(s, OWN).season,
            "day": get_path(s, DAY_CANOPY),
        }

    # "end" is compared with the reference's MAPLNT exit (bare soil after a harvest) on purpose
    return DAY.compile(
        procs,
        outputs=outputs,
        output_reads=(CANOPY, SEEN, f"{OWN}.season", DAY_CANOPY),
        end_of_day_reads=(CANOPY,),
    )


def _f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@pytest.fixture(scope="module")
def canopy_run(data_dir: Path) -> dict[str, Any]:
    from test_day_rzwqm46_smoke import SCENARIO, Reference

    ref = Reference(data_dir)
    mg, dx, mx, pe = (ref.t[n] for n in ("mz_grosub_exit", "dssatdrv_exit", "maplnt_exit", "physcl_entry"))
    crop_days = np.asarray(dx.date, dtype=np.int64)
    assert np.array_equal(crop_days, np.asarray(mg.date, dtype=np.int64))
    days = np.asarray(yrdoy_range(FIRST, LAST), dtype=np.int64)
    n = len(days)
    pos = {int(d): t for t, d in enumerate(days)}
    it = np.asarray([pos[int(d)] for d in crop_days])
    v, m = dx.values, mg.values
    # ---- the reference's seasons: first and last crop day, population and planned harvest date
    yr = crop_days // 1000
    sow, harvest, pop, hdate = [], [], [], []
    for y in np.unique(yr):
        i = np.nonzero(yr == y)[0]
        sow.append(int(crop_days[i[0]]))
        harvest.append(int(crop_days[i[-1]]))
        pop.append(float(v["PLANTVAR%PLTPOP"][i[0]]))
        hdate.append(int(v["HDATE"][i[0], 0]))
    # ---- the cultivar's HTMAX and BIOHALF, read as REAL*4 as the reference does
    vr = v["VRNAME"][0]
    vr = (bytes(vr).decode() if isinstance(vr, (bytes, np.bytes_)) else str(vr)).strip()
    cul = read_cul(data_dir / SCENARIO / "MZCER040.CUL")
    row = cul[cul["VRNAME"].str.strip() == vr].iloc[0]
    htmax = float(np.float32(row["HTMAX"]))
    biohalf = float(np.float32(row["BIOHALF"]))
    coef = CanopyCoefficients(htmax=htmax, biohalf=biohalf).as_arrays(jnp.float64)
    p0 = ref.crop
    table = CeresSeasons(
        yrplt=jnp.asarray(sow, jnp.int32),
        pltpop=jnp.asarray(pop, jnp.float64),
        sdepth=jnp.asarray([float(v["PLANTVAR%SDEPTH"][0])] * len(sow), jnp.float64),
        rowspc=jnp.asarray([float(v["PLANTVAR%ROWSPC"][0])] * len(sow), jnp.float64),
    )
    params = {
        "crop": p0.replace(
            seasons=table, canopy=CeresCanopyParams(hdate=jnp.asarray(hdate, jnp.int32), coefficients=coef)
        )
    }
    # ---- forcing: the replayed crop state, the date, the harvest flags
    host: dict[str, np.ndarray] = {k: np.zeros(n) for k in ("lai", "biomas", "grnwt", "ears")}
    host["mdate"] = np.full(n, -99, dtype=np.int32)
    host["lai"][it] = _f64(v["XHLAI"])
    host["biomas"][it] = _f64(v["BIOMAS"])
    host["grnwt"][it] = _f64(v["GRNWT"])
    host["ears"][it] = _f64(v["EARS"])
    host["mdate"][it] = np.asarray(m["MDATE"], dtype=np.int32)
    rep = {k: jnp.asarray(x) for k, x in host.items()}
    z = jnp.zeros(n)
    crop = CeresForcing(yrdoy=jnp.asarray(days, jnp.int32), tmax=z, tmin=z, srad=z, dayl=z, twilen=z, co2=z)
    h_on = np.zeros(n, bool)
    h_on[[pos[h] for h in harvest]] = True
    events = EventTable.empty(n).replace(harvest=jnp.asarray(h_on))
    s0 = CeresMaizeState.initial(params["crop"], 1, events=True)
    g = {
        **Binding(OWN, BOUND).entries(s0),
        CANOPY: CanopyRecord.zeros(1),
        SEEN: CanopyRecord.zeros(1),
        DAY_CANOPY: CanopyRecord.zeros(1),
    }
    model = _model()
    report = DAY.check(model)
    out = jax.jit(lambda pp, ff, ss: run(model, pp, ff, ss))(
        params, {"crop": crop, "rep": rep, "events": events}, compose(g)
    )
    out = jax.tree_util.tree_map(np.asarray, out)
    # ---- the reference record of each day (MAPLNT exit) and what PET read (PHYSCL entry)
    mi = {int(d): t for t, d in enumerate(np.asarray(mx.date))}
    pi = {int(d): t for t, d in enumerate(np.asarray(pe.date))}
    ref_end = {
        k: np.asarray([mx.values[k][mi[int(d)]] for d in days], np.float64) for k in ("LAI", "TLAI", "HEIGHT")
    }
    ref_seen = {
        k: np.asarray([pe.values[k][pi[int(d)]] for d in days], np.float64) for k in ("LAI", "TLAI", "HEIGHT")
    }
    # ---- tolerances from the reference's REAL*4 operations
    lai_tol = U32 * np.abs(ref_end["LAI"]) + U64 * np.abs(ref_end["LAI"])
    b10 = _f64(rep["biomas"]) * 10.0
    grain = np.floor(_f64(rep["grnwt"]) * _f64(rep["ears"]) * 10.0 + 0.5)
    stalk = b10 - grain
    rel = np.where(stalk > 0.0, U32 * (1.0 + b10 / np.where(stalk > 0.0, stalk, 1.0)), 0.0)
    season_of = np.searchsorted(np.asarray(sow), days, side="right")
    rel_run = np.zeros(n)
    for k in np.unique(season_of):  # running maximum of the bound within each season (host side)
        sel = season_of == k
        rel_run[sel] = np.maximum.accumulate(rel[sel])
    h_tol = (rel_run + U64) * np.abs(ref_end["HEIGHT"])
    return {
        "days": days,
        "pos": pos,
        "it": it,
        "out": out,
        "report": report,
        "ref_end": ref_end,
        "ref_seen": ref_seen,
        "lai_tol": lai_tol,
        "h_tol": h_tol,
        "tables": {"dx": dx, "mg": mg, "mx": mx, "pe": pe},
        "facts": {"sow": sow, "harvest": harvest, "hdate": hdate, "pop": pop},
        "cul": (htmax, biohalf),
        "rep": {k: np.asarray(x) for k, x in rep.items()},
    }


def test_reference_pet_reads_yesterdays_maplnt_exit(canopy_run: dict) -> None:
    mx, pe = canopy_run["tables"]["mx"], canopy_run["tables"]["pe"]
    assert np.array_equal(np.asarray(mx.date), np.asarray(pe.date)) and len(mx.date) == 3287
    for k in ("LAI", "TLAI", "HEIGHT"):
        np.testing.assert_array_equal(pe.values[k][1:], mx.values[k][:-1])
        assert float(pe.values[k][0]) == 0.0
    np.testing.assert_array_equal(mx.values["TLAI"], mx.values["LAI"])
    days, it = canopy_run["days"], canopy_run["it"]
    crop = np.zeros(len(days), bool)
    crop[it] = True
    ends = canopy_run["ref_end"]
    for k in ("LAI", "TLAI", "HEIGHT"):
        assert np.all(ends[k][~crop] == 0.0), k
        assert np.all(ends[k][[canopy_run["pos"][h] for h in canopy_run["facts"]["harvest"]]] == 0.0), k


def test_the_daily_output_record_has_the_harvest_days_canopy(canopy_run: dict) -> None:
    """The daily output record, written before the harvest's reset,
    differs from the end-of-day record only on harvest days, where it keeps the day's canopy (the
    running-maximum height is not 0) while the end of the day is bare soil."""
    day, end = canopy_run["out"]["day"], canopy_run["out"]["end"]
    h = np.asarray([canopy_run["pos"][d] for d in canopy_run["facts"]["harvest"]])
    other = np.ones(len(canopy_run["days"]), bool)
    other[h] = False
    for k in ("lai", "tlai", "height"):
        a, b = np.asarray(getattr(day, k))[:, 0], np.asarray(getattr(end, k))[:, 0]
        assert a[other].tobytes() == b[other].tobytes(), k
        assert np.all(b[h] == 0.0), k
    assert np.all(np.asarray(day.height)[h, 0] > 0.0)


def test_cultivar_height_parameters_are_the_dumped_ones(canopy_run: dict) -> None:
    htmax, biohalf = canopy_run["cul"]
    assert (htmax, biohalf) == (float(np.float32(244.6)), float(np.float32(43.07)))
    v = canopy_run["tables"]["dx"].values
    np.testing.assert_array_equal(v["PLHGHT"], htmax)
    np.testing.assert_array_equal(v["PLALFA"], -2.0 * htmax * math.log(0.5) / biohalf)


def test_canopy_record_is_the_maplnt_exit(canopy_run: dict) -> None:
    end, ref = canopy_run["out"]["end"], canopy_run["ref_end"]
    lai, tlai, h = (np.asarray(getattr(end, k))[:, 0] for k in ("lai", "tlai", "height"))
    np.testing.assert_array_equal(tlai, lai)
    zero = ref["HEIGHT"] == 0.0
    assert np.all(lai[ref["LAI"] == 0.0] == 0.0) and np.all(h[zero] == 0.0)
    d_lai = np.abs(lai - ref["LAI"])
    d_h = np.abs(h - ref["HEIGHT"])
    bad_l = np.nonzero(d_lai > canopy_run["lai_tol"])[0]
    bad_h = np.nonzero(d_h > canopy_run["h_tol"])[0]
    days = canopy_run["days"]
    assert bad_l.size == 0, (days[bad_l][:10], d_lai[bad_l][:10])
    assert bad_h.size == 0, (days[bad_h][:10], d_h[bad_h][:10])
    rel_h = d_h[~zero] / ref["HEIGHT"][~zero]
    rel_l = d_lai[ref["LAI"] > 0] / ref["LAI"][ref["LAI"] > 0]
    measured = {
        "days": len(days),
        "crop_days": len(canopy_run["it"]),
        "nonzero_days": int(np.sum(~zero)),
        "lai_max_abs": float(d_lai.max()),
        "lai_max_rel": float(rel_l.max()),
        "height_max_abs_cm": float(d_h.max()),
        "height_max_rel": float(rel_h.max()),
        "tolerance": "LAI: 2**-24 |ref| + 8 eps64 |ref|; height: 2**-24 (1 + 10 BIOMAS / stalk) running max + 8 eps64",
    }
    print("W4B canopy", json.dumps(measured))
    root = Path(os.environ.get("AGRI_JAX_DATA", "~/agri_jax_data")).expanduser()
    out = root / REPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measured, indent=2))


def test_post_maturity_decline_is_exercised(canopy_run: dict) -> None:
    days, rep = canopy_run["days"], canopy_run["rep"]
    hd = np.zeros(len(days), np.int64)
    sow = np.asarray(canopy_run["facts"]["sow"])
    k = np.searchsorted(sow, days, side="right") - 1
    hd[k >= 0] = np.asarray(canopy_run["facts"]["hdate"])[k[k >= 0]]
    m = rep["mdate"].astype(np.int64)
    decline = (days > m) & (hd > m) & (m > 0)
    assert int(decline.sum()) == N_DECLINE
    assert set((days[decline] // 1000).tolist()) == {2020, 2021}
    lai = np.asarray(canopy_run["out"]["end"].lai)[:, 0]
    assert np.all(lai[decline] < rep["lai"][decline])
    assert np.all(np.abs(lai - canopy_run["ref_end"]["LAI"])[decline] <= canopy_run["lai_tol"][decline])


def test_pet_reads_the_previous_day_record(canopy_run: dict) -> None:
    seen, ref = canopy_run["out"]["seen"], canopy_run["ref_seen"]
    lai_tol = np.concatenate([[0.0], canopy_run["lai_tol"][:-1]])
    h_tol = np.concatenate([[0.0], canopy_run["h_tol"][:-1]])
    for k, tol in (("lai", lai_tol), ("tlai", lai_tol), ("height", h_tol)):
        x = np.asarray(getattr(seen, k))[:, 0]
        assert np.all(np.abs(x - ref[k.upper()]) <= tol), k
    rep = canopy_run["report"]
    assert ("pet.sw_daily", CANOPY) in set(rep.used)
