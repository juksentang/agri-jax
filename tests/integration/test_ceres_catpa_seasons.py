"""Seven CA-TPA maize seasons (2015-2021) in one run against the RZWQM2 4.6 embedded CERES.

The acceptance test of the season boundaries: CERES-Maize runs
from 2015-001 to 2021-365 in one ``scan`` with the per-season parameter table of the reference's
seven seasons (``CeresSeasons``), sowing and harvest from an event table, the crop's water record
(P1) replayed from the reference run (ROOTWU entry ``SW``, DSSATDRV exit ``EOP``, ROOTWU exit
``TRWUP``), its nitrogen stress through P10 (the reference's ``NSTRES``, ``n_supply`` replay) and its
snow through P9 (``.ana`` column 92, ``x 10`` mm). The day is the crop side of the contract's day:
``season_init`` (management), the replays, the five CERES entries (growth ``nstress_replay``),
``harvest`` and ROOTWU's own season end (``water_supply.<slot>.season_end``) at the
end of the plant phase.

One vmapped program runs nine scenarios: the seven seasons together, each season alone (its own
sowing and harvest flags, its row first in the table), and the seven seasons with every sow flag
five days after the table's ``YRPLT``. The crop state is event-driven
(``CeresMaizeState.initial(..., events=True)``); the canopy record and ROOTWU's state start at a
non-zero value, so that their zeros after each harvest are the season ends' (the crop's for the
canopy, ROOTWU's own for its state and ``TRWUP``). Asserted:

* season structure of the reference: seven seasons, the first crop day is ``YRPLT`` and the last
  the harvest date ``HDATE`` of each (1088 crop days);
* each season of the seven-season run equals that season run alone, bit for bit, from its sowing
  day to its harvest day (the reset leaves nothing of the previous season);
* at the end of every day the crop is past the sowing stage exactly from each sowing day to the
  day before its harvest, and the season index counts the harvests;
* the crop water record's ``TRWUP`` at the end of each crop day equals the reference DSSATDRV exit
  ``TRWUP`` (0 on the 7 harvest days, ROOTWU's value otherwise) exactly, and is 0 between seasons;
* the root record (P2) between a harvest and the next sowing has no roots and ``XHLAI = 0``,
  equal to the reference ROOTWU entry ``RLV`` of each sowing day (0); ROOTWU's ``TSS`` is 0 at
  each harvest's end, as the reference ROOTWU entry ``TSS`` of each sowing day; the canopy record
  (P6) at the end of each harvest day is 0, equal to the reference MAPLNT exit ``LAI``, ``TLAI``,
  ``HEIGHT`` of that day;
* in every season the dates on which stages 7, 8, 9, 1, 2, 3 and 4 end (``STGDOY``: sowing to the
  beginning of effective grain filling) are the embedded CERES's, to the day;
* the sow flags start the crop: with the flags five days late every season starts on its flag
  day, not on ``YRPLT``; ``check_sowing_dates`` accepts the reference inputs and rejects those.

Measured and recorded (``$AGRI_JAX_DATA/validation/aj_w4/catpa_seasons.json``, printed with
``-s``), not asserted: per season the later stage dates, the days with the same ``ISTAGE``, LAI,
biomass and yield of our DSSAT-CSM 4.8.6 CERES against the RZWQM2 embedded DSSAT 4.0.2 CERES
(different model versions: no reference-side tolerance applies to the growth).

Data (kept outside the repository, ``allow_skip``): the RZWQM2 dump tables and scenario inputs of
``test_day_rzwqm46_smoke.py`` (whose ``Reference`` builds the CA-TPA crop parameters and weather).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import Day, Phase, bind, compose, run
from agrijax.core.day import snapshot
from agrijax.core.events import EventTable
from agrijax.core.ports import Binding
from agrijax.core.state import get_path
from agrijax.iface.contract import phased_writes
from agrijax.iface.crop import CanopyRecord, CropNIn
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES_NSTRESS_REPLAY,
    NOT_REACHED,
    CeresMaizeState,
    CeresReplayForcing,
    CeresSeasons,
    ceres_crop_water_replay,
    ceres_harvest,
    ceres_season_init,
    ceres_snow_replay,
    check_sowing_dates,
    yield_kg_ha,
)
from agrijax.processes.crop.ceres_maize.season import OWN_FIELDS
from agrijax.processes.n_supply import CropNReplayState, crop_n_replay
from agrijax.processes.water_supply import RootwuState, rootwu_season_end, rootwu_trwup_replay
from agrijax.sites.dssat_inputs import yrdoy_range

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the RZWQM2 full-state dump tables of CA-TPA 2015-2023"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="the seasons run is float64"),
]

FIRST, LAST = 2015001, 2021365
N_SEASONS = 7
SLOT = "maize"
OWN = f"crops.{SLOT}"
PORTS = {
    "water_in": f"iface.crop_water.{SLOT}",
    "root_out": f"iface.root.{SLOT}",
    "n_in": f"iface.crop_n.{SLOT}",
    "snow_in": "iface.snow",
}
HARVEST_PORTS = {"root_out": f"iface.root.{SLOT}", "canopy_out": f"iface.canopy.{SLOT}"}
#: ROOTWU's own state; its season end zeroes it and P1 trwup on harvest days
SUPPLY = f"water_supply.{SLOT}"
NAMES = ("phenology", "stress", "growth", "roots", "publish")
PRE_HARVEST = "prev.crop_before_harvest"
REPORT = "validation/aj_w4/catpa_seasons.json"
#: lane of the scenario whose sow flags are LATE days after the table's YRPLT
LATE_LANE, LATE = 8, 5
#: start value of the canopy record and of ROOTWU's state (only the season ends write them)
PREFILL = 1.0

#: P1 trwup and the day's value of it, recorded before the season end (the daily output record)
TRWUP = f"{PORTS['water_in']}.trwup"
DAY_TRWUP = "prev.day.trwup"
PLANT = (
    f"n_supply.{SLOT}.replay",
    # the replay of P1: sw and eop for the crop-side producers, trwup for ROOTWU (its owning module)
    f"{OWN}.water_replay",
    f"{SUPPLY}.rootwu",
    f"{OWN}.snow_replay",
    *(f"{OWN}.{n}" for n in NAMES),
    f"{OWN}.before_harvest",
    "prev.day_output",
    f"{OWN}.harvest",
    f"{SUPPLY}.season_end",
)
DAY = Day(
    ref="rzwqm2-4.6",
    bare=True,  # part of the contract's day with stand-ins
    phases=(Phase("management", (f"{OWN}.season_init",)), Phase("plant", PLANT)),
    # the contract's declared season ends of the crop (P2, P6) and of ROOTWU (P1 trwup, rwu, tss)
    phased_writes=tuple(w for w in phased_writes(SLOT) if w.entry in PLANT),
)


def _model() -> Any:
    procs = {
        f"{OWN}.season_init": bind(ceres_season_init, own=OWN, params="crop", forcing="events"),
        f"n_supply.{SLOT}.replay": bind(
            crop_n_replay, own=f"n_supply.{SLOT}", ports={"n_out": PORTS["n_in"]}, forcing="n"
        ),
        **{
            f"{OWN}.{n}": bind(p, own=OWN, ports=PORTS, params="crop", forcing="crop")
            for n, p in zip(
                ("water_replay", "snow_replay", *NAMES),
                (ceres_crop_water_replay, ceres_snow_replay, *CROP_PROCESSES_NSTRESS_REPLAY),
                strict=True,
            )
        },
        f"{SUPPLY}.rootwu": bind(
            rootwu_trwup_replay, own=SUPPLY, ports={"water": PORTS["water_in"]}, forcing="crop"
        ),
        # the crop of the day before the harvest reset (the harvest day's own values)
        f"{OWN}.before_harvest": snapshot(f"{OWN}.before_harvest", {OWN: PRE_HARVEST}),
        "prev.day_output": snapshot("prev.day_output", {TRWUP: DAY_TRWUP}),
        f"{OWN}.harvest": bind(ceres_harvest, own=OWN, ports=HARVEST_PORTS, params="crop", forcing="events"),
        f"{SUPPLY}.season_end": bind(
            rootwu_season_end, own=SUPPLY, ports={"water": PORTS["water_in"]}, forcing="events"
        ),
    }

    def outputs(s: Any, p: Any, f: Any) -> dict[str, Any]:
        c = get_path(s, OWN)
        pre = get_path(s, PRE_HARVEST)
        return {
            "istage": c.phen.istage[0],
            "season": c.season,
            "pre": {
                "istage": pre.phen.istage[0],
                "lai": pre.growth.lai[0],
                "biomas": pre.growth.biomas[0],
                "yield": yield_kg_ha(pre)[0],
                "rtdep": pre.roots.rtdep[0],
                "stgdoy": pre.phen.stgdoy[0],
            },
            "own": {n: getattr(c, n) for n in OWN_FIELDS},
            "trwup": get_path(s, f"{PORTS['water_in']}.trwup")[0],
            "root": get_path(s, PORTS["root_out"]),
            "canopy": get_path(s, HARVEST_PORTS["canopy_out"]),
            "tss": get_path(s, f"water_supply.{SLOT}.tss")[0],
            "day_trwup": get_path(s, DAY_TRWUP)[0],
        }

    # the end-of-day TRWUP, TSS and records are compared with the reference's exit values (after
    # its harvest-day reset) on purpose; the day's TRWUP is DAY_TRWUP
    end = (TRWUP, PORTS["root_out"], HARVEST_PORTS["canopy_out"], f"{SUPPLY}.tss")
    return DAY.compile(
        procs, outputs=outputs, output_reads=(OWN, PRE_HARVEST, DAY_TRWUP, *end), end_of_day_reads=end
    )


def _bytes_equal(a: Any, b: Any) -> list[str]:
    fa = jax.tree_util.tree_flatten_with_path(a)[0]
    fb = jax.tree_util.tree_flatten_with_path(b)[0]
    return [
        jax.tree_util.keystr(p)
        for (p, x), (_, y) in zip(fa, fb, strict=True)
        if np.asarray(x).tobytes() != np.asarray(y).tobytes()
    ]


@pytest.fixture(scope="module")
def seasons_run(data_dir: Path) -> dict[str, Any]:
    from test_day_rzwqm46_smoke import Reference  # the CA-TPA reference inputs of the smoke test

    ref = Reference(data_dir)
    mg, dx, rx, re, mx = (
        ref.t[n] for n in ("mz_grosub_exit", "dssatdrv_exit", "rootwu_exit", "rootwu_entry", "maplnt_exit")
    )
    crop_days = np.asarray(dx.date, dtype=np.int64)
    days = np.asarray(yrdoy_range(FIRST, LAST), dtype=np.int64)
    n = len(days)
    # ---- the reference's seasons
    yr = crop_days // 1000
    sow, harvest, rows = [], [], []
    for y in np.unique(yr):
        idx = np.nonzero(yr == y)[0]
        sow.append(int(crop_days[idx[0]]))
        harvest.append(int(crop_days[idx[-1]]))
        i0 = int(idx[0])
        rows.append(
            (
                int(mg.values["YRPLT"][i0]),
                float(dx.values["PLANTVAR%PLTPOP"][i0]),
                float(dx.values["PLANTVAR%SDEPTH"][i0]),
                float(dx.values["PLANTVAR%ROWSPC"][i0]),
                int(dx.values["HDATE"][i0, 0]),
            )
        )
    facts = {"sow": sow, "harvest": harvest, "rows": rows}
    # ---- forcing: weather and the replayed P1 (crop days), P9 snow, P10 NSTRES
    crop = ref.crop_forcing([int(d) for d in days], replay=True)
    assert isinstance(crop, CeresReplayForcing)
    swe = np.asarray([ref.ana[92][ref.ana_i(int(d)) + 1] * 10.0 for d in days])
    crop = crop.replace(snow=jnp.asarray(swe))
    nf = ref.nstres([int(d) for d in days])
    pos = {int(d): t for t, d in enumerate(days)}

    def table(order: list[int]) -> CeresSeasons:
        rs = [rows[k][:4] for k in order] + [(NOT_REACHED, *rows[order[-1]][1:4])] * (N_SEASONS - len(order))
        y, pop, sd, rsp = (np.asarray(c) for c in zip(*rs, strict=True))
        return CeresSeasons(
            yrplt=jnp.asarray(y, jnp.int32),
            pltpop=jnp.asarray(pop, jnp.float64),
            sdepth=jnp.asarray(sd, jnp.float64),
            rowspc=jnp.asarray(rsp, jnp.float64),
        )

    def events(order: list[int], late: int = 0) -> EventTable:
        ev = EventTable.empty(n)
        s_on = np.zeros(n, bool)
        h_on = np.zeros(n, bool)
        for k in order:
            s_on[pos[sow[k]] + late] = True
            h_on[pos[harvest[k]]] = True
        return ev.replace(sow=jnp.asarray(s_on), harvest=jnp.asarray(h_on))

    # lanes 0-7: the seven seasons together, then each alone; lane 8: the seven seasons with every
    # sow flag LATE days after the table's YRPLT (the flags, not YRPLT, start the crop)
    scenarios = [list(range(N_SEASONS))] + [[k] for k in range(N_SEASONS)] + [list(range(N_SEASONS))]
    p0 = ref.crop
    params = [{"crop": p0.replace(seasons=table(o))} for o in scenarios]
    evs = [events(o, LATE if lane == LATE_LANE else 0) for lane, o in enumerate(scenarios)]
    nl = ref.nl

    def state0(pp: Any) -> dict[str, Any]:
        s0 = CeresMaizeState.initial(pp["crop"], 1, events=True).replace(n_in=CropNIn.initial(1))
        # the canopy record and the uptake producer's state start non-zero: only the season ends
        # write them here, so their zeros after each harvest are the resets' work
        full = lambda x: jnp.full_like(x, PREFILL)  # noqa: E731
        g = {
            **Binding(OWN, tuple(PORTS.items())).entries(s0),
            HARVEST_PORTS["canopy_out"]: jax.tree_util.tree_map(full, CanopyRecord.zeros(1)),
            f"water_supply.{SLOT}": jax.tree_util.tree_map(full, RootwuState.initial(1, nl)),
            f"n_supply.{SLOT}": CropNReplayState(),
        }
        g[PRE_HARVEST] = g[OWN]
        g["prev.day"] = {"trwup": jnp.zeros(1)}
        return compose(g)

    stack = lambda xs: jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)  # noqa: E731
    model = _model()
    report = DAY.check(model)
    go = jax.jit(
        jax.vmap(
            lambda pp, ev, s: run(model, pp, {"crop": crop, "n": nf, "events": ev}, s),
            in_axes=(0, 0, 0),
        )
    )
    out = go(stack(params), stack(evs), stack([state0(pp) for pp in params]))
    out = jax.tree_util.tree_map(np.asarray, out)
    return {
        "ref": ref,
        "facts": facts,
        "days": days,
        "pos": pos,
        "out": out,
        "report": report,
        "tables": {"mg": mg, "dx": dx, "rx": rx, "re": re, "mx": mx},
        "swe": swe,
        "replay_trwup": np.asarray(crop.trwup),
        "sow_flags": [np.asarray(e.sow) for e in evs],
        "seasons": [pp["crop"].seasons for pp in params],
    }


def test_reference_has_seven_seasons_from_sowing_to_harvest(seasons_run: dict) -> None:
    f = seasons_run["facts"]
    assert len(f["sow"]) == N_SEASONS and len(seasons_run["tables"]["dx"].date) == 1088
    for (yrplt, _, _, _, hdate), s, h in zip(f["rows"], f["sow"], f["harvest"], strict=True):
        assert yrplt == s and hdate == h
    assert seasons_run["report"].used == ()  # the crop side has no lag of its own


def test_each_season_equals_the_season_alone(seasons_run: dict) -> None:
    out, pos, f = seasons_run["out"], seasons_run["pos"], seasons_run["facts"]
    for k in range(N_SEASONS):
        a, b = pos[f["sow"][k]], pos[f["harvest"][k]]
        for part in ("own", "pre"):
            together = jax.tree_util.tree_map(lambda x, a=a, b=b: x[0, a:b], out[part])
            alone = jax.tree_util.tree_map(lambda x, k=k, a=a, b=b: x[k + 1, a:b], out[part])
            assert _bytes_equal(together, alone) == [], (k, part)
        # the harvest day's own values (before the reset) too
        for key in ("lai", "biomas", "yield", "stgdoy"):
            assert out["pre"][key][0, b].tobytes() == out["pre"][key][k + 1, b].tobytes(), (k, key)


def test_crop_is_active_on_the_reference_crop_days_and_counts_seasons(seasons_run: dict) -> None:
    out, pos, f, days = seasons_run["out"], seasons_run["pos"], seasons_run["facts"], seasons_run["days"]
    n = len(days)
    past = np.zeros(n, bool)
    h_on = np.zeros(n, bool)
    for s, h in zip(f["sow"], f["harvest"], strict=True):
        past[pos[s] : pos[h]] = True
        h_on[pos[h]] = True
    np.testing.assert_array_equal(out["istage"][0] != 7, past)
    np.testing.assert_array_equal(out["season"][0], np.cumsum(h_on))
    # the reference's crop days are the sowing-to-harvest windows
    crop_days = np.asarray(seasons_run["tables"]["dx"].date, dtype=np.int64)
    on = np.isin(days, crop_days)
    np.testing.assert_array_equal(on, past | h_on)


def test_trwup_is_the_reference_crop_exit_trwup(seasons_run: dict) -> None:
    out, pos = seasons_run["out"], seasons_run["pos"]
    dx = seasons_run["tables"]["dx"]
    crop_days = np.asarray(dx.date, dtype=np.int64)
    idx = np.asarray([pos[int(d)] for d in crop_days])
    ours = out["trwup"][0]
    np.testing.assert_array_equal(ours[idx], np.asarray(dx.values["TRWUP"], dtype=np.float64))
    off = np.ones(len(ours), bool)
    off[idx] = False
    assert np.all(ours[off] == 0.0)
    assert int(np.sum(ours[idx] == 0.0)) >= N_SEASONS
    # on the harvest days the replayed (ROOTWU exit) TRWUP is not 0: the zeros there are the reset's
    f = seasons_run["facts"]
    h = np.asarray([pos[d] for d in f["harvest"]])
    assert np.all(np.asarray(seasons_run["replay_trwup"])[h] > 0.0)
    np.testing.assert_array_equal(ours[h], 0.0)
    # the daily output record (before the season end) has the day's TRWUP, the ROOTWU exit value,
    # also on the harvest days
    day = out["day_trwup"][0]
    np.testing.assert_array_equal(day, np.asarray(seasons_run["replay_trwup"], dtype=day.dtype))
    assert np.all(day[h] > 0.0)


def test_records_between_seasons_are_the_no_crop_records(seasons_run: dict) -> None:
    out, pos, f = seasons_run["out"], seasons_run["pos"], seasons_run["facts"]
    t = seasons_run["tables"]
    root = out["root"]
    for k in range(N_SEASONS):
        h = pos[f["harvest"][k]]
        end = pos[f["sow"][k + 1]] if k + 1 < N_SEASONS else len(seasons_run["days"])
        for x in (root.rlv, root.rtdep, root.xhlai):
            assert np.all(x[0, h:end] == 0.0), k
        # the reference: ROOTWU entry RLV and TSS of the next sowing day are 0, as our records
        if k + 1 < N_SEASONS:
            j = int(t["re"].index_of([f["sow"][k + 1]])[0])
            np.testing.assert_array_equal(
                np.asarray(t["re"].values["RLV"][j][: seasons_run["ref"].nl], float), root.rlv[0, end - 1, 0]
            )
            np.testing.assert_array_equal(
                np.asarray(t["re"].values["TSS"][j][: seasons_run["ref"].nl], float), out["tss"][0, end - 1]
            )
        assert np.all(out["tss"][0, h] == 0.0)
        if k == 0:  # before the first harvest ROOTWU's state is the prefill: the zeros are the reset's
            assert np.all(out["tss"][0, h - 1] == PREFILL)
            assert float(out["canopy"].lai[0, h - 1, 0]) == PREFILL
        # the canopy after the harvest day: 0, as the reference MAPLNT exit of the harvest day
        jm = int(t["mx"].index_of([f["harvest"][k]])[0])
        for ours, key in (
            (out["canopy"].lai, "LAI"),
            (out["canopy"].tlai, "TLAI"),
            (out["canopy"].height, "HEIGHT"),
        ):
            assert float(ours[0, h, 0]) == float(t["mx"].values[key][jm]) == 0.0, (k, key)
        assert float(np.max(root.xhlai[0, pos[f["sow"][k]] : h])) > 0.0


def test_growth_against_the_embedded_ceres_is_recorded(seasons_run: dict, data_dir: Path) -> None:
    """Measured, not asserted: DSSAT-CSM 4.8.6 CERES (ours) against the RZWQM2 embedded DSSAT 4.0.2
    CERES, both driven by the same crop water and NSTRES."""
    out, pos, f = seasons_run["out"], seasons_run["pos"], seasons_run["facts"]
    mg = seasons_run["tables"]["mg"]
    ref_days = np.asarray(mg.date, dtype=np.int64)
    rows = []
    for k in range(N_SEASONS):
        a, b = pos[f["sow"][k]], pos[f["harvest"][k]]
        sel = (ref_days >= f["sow"][k]) & (ref_days <= f["harvest"][k])
        r_lai = np.asarray(mg.values["LAI"][sel], float)
        r_bio = np.asarray(mg.values["BIOMAS"][sel], float)
        r_st = np.asarray(mg.values["ISTAGE"][sel])
        o_lai = out["pre"]["lai"][0, a : b + 1]
        o_bio = out["pre"]["biomas"][0, a : b + 1]
        o_st = out["pre"]["istage"][0, a : b + 1]
        r_stg = np.asarray(mg.values["STGDOY"][sel][-1])
        o_stg = out["pre"]["stgdoy"][0, b]
        stage_diff = {
            str(s + 1): int(_doy_diff(int(o_stg[s]), int(r_stg[s])))
            for s in range(min(len(o_stg), len(r_stg)))
            if 0 < int(r_stg[s]) < 9999999 and 0 < int(o_stg[s]) < 9999999
        }
        rows.append(
            {
                "season": int(f["sow"][k] // 1000),
                "sow": f["sow"][k],
                "harvest": f["harvest"][k],
                "days": int(b - a + 1),
                "istage_days_equal": int(np.sum(o_st == r_st)),
                "stage_date_diff_days": stage_diff,
                "lai_max": [float(o_lai.max()), float(r_lai.max())],
                "lai_rmse": float(np.sqrt(np.mean((o_lai - r_lai) ** 2))),
                "biomass_harvest_g_m2": [float(o_bio[-1]), float(r_bio[-1])],
                "yield_harvest_kg_ha": [
                    float(out["pre"]["yield"][0, b]),
                    float(np.asarray(mg.values["YIELD"][sel], float)[-1]),
                ],
                "snow_days_in_season": int(np.sum(seasons_run["swe"][a : b + 1] > 0.0)),
            }
        )
    rep = {"what": "ours [DSSAT-CSM 4.8.6 CERES] vs RZWQM2 4.6 embedded DSSAT 4.0.2 CERES", "seasons": rows}
    seasons_run["growth_rows"] = rows
    path = Path(os.environ.get("AGRI_JAX_DATA", str(data_dir))) / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1) + "\n")
    print("W4A " + json.dumps(rep))
    assert len(rows) == N_SEASONS


def _doy_diff(a: int, b: int) -> int:
    """Days from ``YYYYDDD`` ``b`` to ``a`` (same or adjacent years)."""
    import datetime

    def d(x: int) -> datetime.date:
        return datetime.date(x // 1000, 1, 1) + datetime.timedelta(days=x % 1000 - 1)

    return (d(a) - d(b)).days


def test_phenology_to_silking_is_the_embedded_ceres(seasons_run: dict, data_dir: Path) -> None:
    """The dates on which stages 7, 8, 9, 1, 2, 3 and 4 end (STGDOY: sowing to the beginning of
    effective grain filling) are the reference's, to the day, in all seven seasons; the later
    stages depend on grain growth, which differs between the two CERES versions, and are recorded
    only."""
    if "growth_rows" not in seasons_run:
        test_growth_against_the_embedded_ceres_is_recorded(seasons_run, data_dir)
    for row in seasons_run["growth_rows"]:
        diff = row["stage_date_diff_days"]
        for stage in ("7", "8", "9", "1", "2", "3", "4"):
            assert diff.get(stage) == 0, (row["season"], stage, diff)


def test_the_sow_flags_start_the_crop(seasons_run: dict) -> None:
    """The sowing decision is the event table's. With every sow flag LATE days
    after the table's YRPLT (lane LATE_LANE), each season starts on its flag day: the crop is past
    the sowing stage from the flag day to the day before harvest, and the sowing stage ends
    (STGDOY(7)) on the flag day. ``check_sowing_dates`` accepts the reference lanes and rejects
    the late one."""
    out, pos, f, days = seasons_run["out"], seasons_run["pos"], seasons_run["facts"], seasons_run["days"]
    past = np.zeros(len(days), bool)
    for s_, h in zip(f["sow"], f["harvest"], strict=True):
        past[pos[s_] + LATE : pos[h]] = True
    np.testing.assert_array_equal(out["istage"][LATE_LANE] != 7, past)
    for s_, h in zip(f["sow"], f["harvest"], strict=True):
        assert int(out["pre"]["stgdoy"][LATE_LANE, pos[h], 7 - 1]) == int(days[pos[s_] + LATE])
        assert int(out["pre"]["stgdoy"][0, pos[h], 7 - 1]) == s_
    for lane in range(LATE_LANE):
        check_sowing_dates(seasons_run["sow_flags"][lane], days, seasons_run["seasons"][lane])
    with pytest.raises(ValueError, match="not the season table's YRPLT"):
        check_sowing_dates(seasons_run["sow_flags"][LATE_LANE], days, seasons_run["seasons"][LATE_LANE])
