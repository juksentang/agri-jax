"""CA-TPA 2015-2021 with the full nitrogen replay and the embedded crop's species file.

The nitrogen replay variant of CERES-Maize (``crop/ceres_maize.growth@dssat-4.8.6.0:
nstress_replay``) reads, through port P10, every nitrogen factor the RZWQM2 4.6 embedded CERES uses
for the crop's mass: ``NSTRES`` (assimilation), ``AGEFAC`` (stage 2-4 expansion and, through
``SLFN``, senescence in every stage), ``NDEF3`` and ``NPOOL`` (the grain-number cap of the first
effective grain-fill day), replayed per day from the reference's MZ_GROSUB exit (the OUTPUT call,
after that day's INTEGR; ``NFAC`` changes only the crop's nitrogen and is not replayed). The crop
parameters read the species file the embedded crop reads, ``Scenario/DSSAT/MZCER040.SPE``
(``RSGR`` 0.9, ``PORM`` 0.01; the dumped ``FILECC``).

One vmapped program runs the seven seasons in three lanes, each with the day of
``test_ceres_catpa_seasons.py`` (event-table sowing and harvest, the per-season table, P1 crop water,
P9 snow and P10 replayed):

* ``full_n``: all four factors and the embedded crop's species file (the RZWQM2-day configuration);
* ``nstres_only``: ``NSTRES`` only (the other factors at the record's defaults), same species;
* ``nstres_only_w4a_spe``: ``NSTRES`` only with ``Scenario/MZCER040.SPE`` (``RSGR`` 0.1,
  ``PORM`` 0.05; not the file the embedded crop reads), kept for comparison.

Asserted:

* each lane reproduces the diagnosis lane of the same configuration (a separate diagnosis script ran
  its own copy of the growth day with switches): ``full_n`` =
  ``L5_all_N+rzwqm_spe``, ``nstres_only`` = ``L11_base+rzwqm_spe``, ``nstres_only_w4a_spe`` =
  ``L0_base_nstres``, per season yield and biomass at harvest, maximum LAI and ``STGDOY``
  (:data:`D1`, pinned); when the diagnosis daily file is present, also every day's
  LAI, biomass, yield, grain number, stem and ear weight and ``CARBO`` of the three lanes;
* ``full_n``: in all seven seasons the stage dates ``STGDOY`` 7, 8, 9, 1, 2, 3, 4 and, where the
  season reaches it, 5 are the embedded CERES's to the day (the 2020 stage 5 was 4 days late with
  the wrong species file: ``RSGR`` 0.1 instead of 0.9 delays early maturity).

Measured and recorded (``$AGRI_JAX_DATA/validation/aj_w4/catpa_n_replay.json``, printed with
``-s``): per season and lane the yield, biomass at harvest, maximum LAI, LAI and biomass RMSE and
``STGDOY(5)`` against RZWQM2. The remaining ``full_n`` gap to RZWQM2 is the DSSAT 4.0 -> 4.8
ear-growth change (a diagnosis run with the two 4.0 ear formulas added matches RZWQM2 to 0.1 kg/ha
over the seven seasons; those formulas are not in ``src/``). No tolerance against
RZWQM2 is asserted on the growth: the two CERES versions differ.

Data (kept outside the repository, ``allow_skip``): the RZWQM2 dump tables and scenario inputs of
``test_day_rzwqm46_smoke.py``; the diagnosis daily file ``$AGRI_JAX_DATA/validation/d1/d1_nreplay_daily.npz``
(optional).
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
from agrijax.io.dssat import read_spe
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES_NSTRESS_REPLAY,
    CeresMaizeState,
    CeresReplayForcing,
    CeresSeasons,
    ceres_crop_water_replay,
    ceres_harvest,
    ceres_season_init,
    ceres_snow_replay,
    yield_kg_ha,
)
from agrijax.processes.n_supply import CropNReplayForcing, CropNReplayState, crop_n_replay
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
REPORT = "validation/aj_w4/catpa_n_replay.json"
D1_DAILY = "validation/d1/d1_nreplay_daily.npz"
#: the species file of the third lane, not the one the embedded crop reads (relative to the scenario
#: directory)
W4A_SPECIES_FILE = "MZCER040.SPE"
#: lane -> (nitrogen factors replayed, species file, the diagnosis lane of the same configuration)
LANES: dict[str, tuple[str, str, str]] = {
    "full_n": ("all", "embedded", "L5_all_N+rzwqm_spe"),
    "nstres_only": ("nstres", "embedded", "L11_base+rzwqm_spe"),
    "nstres_only_w4a_spe": ("nstres", "w4a", "L0_base_nstres"),
}
#: the seasons' results of the diagnosis run (``validation/d1/d1_nreplay.json``, a run outside this
#: repository): per season (2015-2021) yield and biomass at harvest [kg ha-1, g m-2], maximum LAI
#: and STGDOY(1..10)
D1: dict[str, list[tuple[float, float, float, list[int]]]] = {
    "L5_all_N+rzwqm_spe": [
        (10823.422018523323, 2298.466465465578, 5.209236510164814,
         [2015158, 2015165, 2015212, 2015226, 9999999, 9999999, 2015118, 2015119, 2015128, 9999999]),
        (10393.213065298116, 2198.8640565294536, 5.2512497853186915,
         [2016162, 2016169, 2016210, 2016222, 9999999, 9999999, 2016119, 2016120, 2016138, 9999999]),
        (5994.912801318501, 1615.6657771513837, 4.124608167125966,
         [2017164, 2017171, 2017217, 2017231, 9999999, 9999999, 2017118, 2017119, 2017138, 9999999]),
        (10744.088647788649, 2187.4993457206156, 5.108662975799874,
         [2018153, 2018160, 2018200, 2018213, 2018261, 9999999, 2018118, 2018119, 2018128, 9999999]),
        (10493.72643028326, 2152.867623354141, 4.825645656310095,
         [2019167, 2019174, 2019212, 2019225, 9999999, 9999999, 2019118, 2019119, 2019138, 9999999]),
        (12230.998673962218, 2348.395713147624, 5.188332084536062,
         [2020162, 2020169, 2020205, 2020216, 2020280, 2020281, 2020126, 2020127, 2020143, 9999999]),
        (12544.964182367194, 2465.781167562267, 5.215283080829783,
         [2021160, 2021167, 2021209, 2021223, 2021280, 2021284, 2021125, 2021126, 2021140, 9999999]),
    ],
    "L11_base+rzwqm_spe": [
        (11316.023272941547, 2533.3471324391235, 5.589269220576045,
         [2015158, 2015165, 2015212, 2015226, 9999999, 9999999, 2015118, 2015119, 2015128, 9999999]),
        (11307.976484946277, 2454.5687033709964, 5.607692826524034,
         [2016162, 2016169, 2016210, 2016222, 9999999, 9999999, 2016119, 2016120, 2016138, 9999999]),
        (9018.161036063539, 2248.5414266588155, 5.492576048184331,
         [2017164, 2017171, 2017217, 2017231, 9999999, 9999999, 2017118, 2017119, 2017138, 9999999]),
        (11352.137088587002, 2392.448055548997, 5.463267416358109,
         [2018153, 2018160, 2018200, 2018213, 2018261, 9999999, 2018118, 2018119, 2018128, 9999999]),
        (11404.804278009375, 2424.1901171636046, 5.4384546433820145,
         [2019167, 2019174, 2019212, 2019225, 9999999, 9999999, 2019118, 2019119, 2019138, 9999999]),
        (13572.448589106269, 2579.740678216109, 5.407082864196605,
         [2020162, 2020169, 2020205, 2020216, 2020280, 2020281, 2020126, 2020127, 2020143, 9999999]),
        (13128.215068958069, 2627.3193976117595, 5.397430657172849,
         [2021160, 2021167, 2021209, 2021223, 2021280, 2021284, 2021125, 2021126, 2021140, 9999999]),
    ],
    "L0_base_nstres": [
        (11316.023272941547, 2533.3471324391235, 5.589269220576045,
         [2015158, 2015165, 2015212, 2015226, 9999999, 9999999, 2015118, 2015119, 2015128, 9999999]),
        (11307.976484946277, 2454.5687033709964, 5.607692826524034,
         [2016162, 2016169, 2016210, 2016222, 9999999, 9999999, 2016119, 2016120, 2016138, 9999999]),
        (9018.161036063539, 2248.5414266588155, 5.492576048184331,
         [2017164, 2017171, 2017217, 2017231, 9999999, 9999999, 2017118, 2017119, 2017138, 9999999]),
        (11352.137088587002, 2392.448055548997, 5.463267416358109,
         [2018153, 2018160, 2018200, 2018213, 2018261, 9999999, 2018118, 2018119, 2018128, 9999999]),
        (11404.804278009375, 2424.1901171636046, 5.4384546433820145,
         [2019167, 2019174, 2019212, 2019225, 9999999, 9999999, 2019118, 2019119, 2019138, 9999999]),
        (13855.257225292951, 2608.021541834777, 5.407082864196605,
         [2020162, 2020169, 2020205, 2020216, 2020284, 2020290, 2020126, 2020127, 2020143, 9999999]),
        (13128.215068958069, 2627.3193976117595, 5.397430657172849,
         [2021160, 2021167, 2021209, 2021223, 2021280, 2021284, 2021125, 2021126, 2021140, 9999999]),
    ],
}  # fmt: skip
#: the diagnosis daily series compared day by day (its key -> our output key)
DAILY = {
    "lai": "lai",
    "biomas": "biomas",
    "yield": "yield",
    "gpp": "gpp",
    "stmwt": "stmwt",
    "earwt": "earwt",
    "carbo": "carbo",
}
NOT_REACHED = 9999999

DAY = Day(
    ref="rzwqm2-4.6",
    bare=True,  # part of the contract's day with stand-ins
    phases=(
        Phase("management", (f"{OWN}.season_init",)),
        Phase(
            "plant",
            (
                f"n_supply.{SLOT}.replay",
                # the replay of P1: sw and eop for the crop-side producers, trwup for ROOTWU (its owner)
                f"{OWN}.water_replay",
                f"{SUPPLY}.rootwu",
                f"{OWN}.snow_replay",
                *(f"{OWN}.{n}" for n in NAMES),
                f"{OWN}.before_harvest",
                f"{OWN}.harvest",
                f"{SUPPLY}.season_end",
            ),
        ),
    ),
    # the contract's declared season ends of the crop (P2, P6) and of ROOTWU (P1 trwup, rwu, tss)
    phased_writes=tuple(w for w in phased_writes(SLOT) if w.entry.startswith((f"{OWN}.", f"{SUPPLY}."))),
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
        f"{OWN}.before_harvest": snapshot(f"{OWN}.before_harvest", {OWN: PRE_HARVEST}),
        f"{OWN}.harvest": bind(ceres_harvest, own=OWN, ports=HARVEST_PORTS, params="crop", forcing="events"),
        f"{SUPPLY}.season_end": bind(
            rootwu_season_end, own=SUPPLY, ports={"water": PORTS["water_in"]}, forcing="events"
        ),
    }

    def outputs(s: Any, p: Any, f: Any) -> dict[str, Any]:
        pre = get_path(s, PRE_HARVEST)
        return {
            "lai": pre.growth.lai[0],
            "biomas": pre.growth.biomas[0],
            "yield": yield_kg_ha(pre)[0],
            "gpp": pre.phen.gpp[0],
            "stmwt": pre.growth.stmwt[0],
            "earwt": pre.growth.earwt[0],
            "carbo": pre.growth.carbo[0],
            "stgdoy": pre.phen.stgdoy[0],
            "n_in": get_path(s, PORTS["n_in"]),
        }

    return DAY.compile(procs, outputs=outputs, output_reads=(PRE_HARVEST, PORTS["n_in"]))


@pytest.fixture(scope="module")
def n_run(data_dir: Path) -> dict[str, Any]:
    from test_day_rzwqm46_smoke import SCENARIO, Reference

    ref = Reference(data_dir)
    mg, dx = ref.t["mz_grosub_exit"], ref.t["dssatdrv_exit"]
    crop_days = np.asarray(dx.date, dtype=np.int64)
    days = np.asarray(yrdoy_range(FIRST, LAST), dtype=np.int64)
    n = len(days)
    pos = {int(d): t for t, d in enumerate(days)}
    yr = crop_days // 1000
    sow, harvest, rows = [], [], []
    for y in np.unique(yr):
        idx = np.nonzero(yr == y)[0]
        i0 = int(idx[0])
        sow.append(int(crop_days[i0]))
        harvest.append(int(crop_days[idx[-1]]))
        rows.append(
            (
                int(mg.values["YRPLT"][i0]),
                float(dx.values["PLANTVAR%PLTPOP"][i0]),
                float(dx.values["PLANTVAR%SDEPTH"][i0]),
                float(dx.values["PLANTVAR%ROWSPC"][i0]),
            )
        )
    assert len(sow) == N_SEASONS
    crop = ref.crop_forcing([int(d) for d in days], replay=True)
    assert isinstance(crop, CeresReplayForcing)
    swe = np.asarray([ref.ana[92][ref.ana_i(int(d)) + 1] * 10.0 for d in days])
    crop = crop.replace(snow=jnp.asarray(swe))
    full = ref.n_factors([int(d) for d in days])
    only = CropNReplayForcing(
        nstres=full.nstres,
        agefac=jnp.ones_like(full.nstres),
        ndef3=jnp.ones_like(full.nstres),
        npool=jnp.full_like(full.nstres, float(np.asarray(CropNIn.initial(1).npool)[0])),
    )
    # the NSTRES-only replay is ref.nstres (the series of the seasons test)
    np.testing.assert_array_equal(
        np.asarray(ref.nstres([int(d) for d in days]).nstres), np.asarray(full.nstres)
    )
    y_, pop, sd, rsp = (np.asarray(c) for c in zip(*rows, strict=True))
    seasons = CeresSeasons(
        yrplt=jnp.asarray(y_, jnp.int32),
        pltpop=jnp.asarray(pop, jnp.float64),
        sdepth=jnp.asarray(sd, jnp.float64),
        rowspc=jnp.asarray(rsp, jnp.float64),
    )
    s_on, h_on = np.zeros(n, bool), np.zeros(n, bool)
    for k in range(N_SEASONS):
        s_on[pos[sow[k]]] = True
        h_on[pos[harvest[k]]] = True
    ev = EventTable.empty(n).replace(sow=jnp.asarray(s_on), harvest=jnp.asarray(h_on))
    # species: the fixture's (Scenario/DSSAT/MZCER040.SPE) and the other one (Scenario/MZCER040.SPE)
    p0 = ref.crop.replace(seasons=seasons)
    w4a = read_spe(data_dir / SCENARIO / W4A_SPECIES_FILE)
    sp_w4a = p0.species.replace(
        rsgr=jnp.asarray(float(np.asarray(w4a["RSGR"], float))),
        pormin=jnp.asarray(float(np.asarray(w4a["PORM"], float))),
    )
    species = {"embedded": p0.species, "w4a": sp_w4a}
    names = list(LANES)
    params = [{"crop": p0.replace(species=species[LANES[k][1]])} for k in names]
    nforc = [full if LANES[k][0] == "all" else only for k in names]
    nl = ref.nl

    def state0(pp: Any) -> Any:
        s0 = CeresMaizeState.initial(pp["crop"], 1, events=True).replace(n_in=CropNIn.initial(1))
        g = {
            **Binding(OWN, tuple(PORTS.items())).entries(s0),
            HARVEST_PORTS["canopy_out"]: CanopyRecord.zeros(1),
            f"water_supply.{SLOT}": RootwuState.initial(1, nl),
            f"n_supply.{SLOT}": CropNReplayState(),
        }
        g[PRE_HARVEST] = g[OWN]
        return compose(g)

    stack = lambda xs: jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)  # noqa: E731
    model = _model()
    DAY.check(model)
    go = jax.jit(jax.vmap(lambda pp, nf, s: run(model, pp, {"crop": crop, "n": nf, "events": ev}, s)))
    out = go(stack(params), stack(nforc), stack([state0(pp) for pp in params]))
    out = jax.tree_util.tree_map(np.asarray, out)
    return {
        "names": names,
        "out": out,
        "days": days,
        "pos": pos,
        "sow": sow,
        "harvest": harvest,
        "mg": mg,
        "full": full,
        "species": {k: (float(v.rsgr), float(v.pormin), float(v.fslfn)) for k, v in species.items()},
    }


def _season_rows(r: dict[str, Any], lane: int) -> list[dict[str, Any]]:
    out, pos, mg = r["out"], r["pos"], r["mg"]
    ref_days = np.asarray(mg.date, dtype=np.int64)
    rows = []
    for k in range(N_SEASONS):
        a, b = pos[r["sow"][k]], pos[r["harvest"][k]]
        sel = (ref_days >= r["sow"][k]) & (ref_days <= r["harvest"][k])
        r_lai = np.asarray(mg.values["LAI"][sel], float)
        r_bio = np.asarray(mg.values["BIOMAS"][sel], float)
        lai = out["lai"][lane, a : b + 1]
        bio = out["biomas"][lane, a : b + 1]
        stg = [int(x) for x in out["stgdoy"][lane, b]]
        r_stg = [int(x) for x in np.asarray(mg.values["STGDOY"][sel][-1])[:10]]
        rows.append(
            {
                "season": r["sow"][k] // 1000,
                "yield": float(out["yield"][lane, b]),
                "biomass": float(out["biomas"][lane, b]),
                "lai_max": float(lai.max()),
                "stgdoy": stg,
                "rzwqm": {
                    "yield": float(np.asarray(mg.values["YIELD"][sel], float)[-1]),
                    "biomass": float(r_bio[-1]),
                    "lai_max": float(r_lai.max()),
                    "stgdoy": r_stg,
                },
                "lai_rmse": float(np.sqrt(np.mean((lai - r_lai) ** 2))),
                "biomass_rmse_g_m2": float(np.sqrt(np.mean((bio - r_bio) ** 2))),
                "stgdoy5_minus_rzwqm_days": (
                    None if NOT_REACHED in (stg[4], r_stg[4]) else _doy_diff(stg[4], r_stg[4])
                ),
            }
        )
    return rows


def _doy_diff(a: int, b: int) -> int:
    import datetime

    def d(x: int) -> datetime.date:
        return datetime.date(x // 1000, 1, 1) + datetime.timedelta(days=x % 1000 - 1)

    return (d(a) - d(b)).days


def test_replay_carries_every_nitrogen_factor(n_run: dict) -> None:
    """P10 holds the replayed factors on the crop days: the ``full_n`` lane's record is the MZ_GROSUB exit's
    ``NSTRES``, ``AGEFAC``, ``NDEF3``, ``NPOOL``; the NSTRES-only lanes keep the defaults."""
    out, pos, mg = n_run["out"], n_run["pos"], n_run["mg"]
    idx = np.asarray([pos[int(d)] for d in np.asarray(mg.date)])
    rec = out["n_in"]
    for name, col in (("nstres", "NSTRES"), ("agefac", "AGEFAC"), ("ndef3", "NDEF3"), ("npool", "NPOOL")):
        want = np.asarray(mg.values[col], dtype=np.float64)
        np.testing.assert_array_equal(getattr(rec, name)[0, idx, 0], want, err_msg=name)
    for col in ("NSTRES", "AGEFAC", "NDEF3"):  # every factor is below 1 on some crop days
        assert np.any(np.asarray(mg.values[col], float) < 0.99), col
    for lane in (1, 2):
        np.testing.assert_array_equal(rec.agefac[lane], 1.0)
        np.testing.assert_array_equal(rec.ndef3[lane], 1.0)
    assert n_run["species"]["embedded"][:2] == (0.9, 0.01)
    assert n_run["species"]["w4a"][:2] == (0.1, 0.05)


def test_each_lane_reproduces_its_d1_lane(n_run: dict, data_dir: Path) -> None:
    names, out = n_run["names"], n_run["out"]
    for lane, name in enumerate(names):
        d1 = D1[LANES[name][2]]
        rows = _season_rows(n_run, lane)
        for row, (y, bio, lai, stg) in zip(rows, d1, strict=True):
            got = (row["yield"], row["biomass"], row["lai_max"])
            assert got == (y, bio, lai), (name, row["season"], got, (y, bio, lai))
            assert row["stgdoy"] == stg, (name, row["season"], row["stgdoy"], stg)
    daily = Path(os.environ.get("AGRI_JAX_DATA", str(data_dir))) / D1_DAILY
    if not daily.is_file():
        return
    with np.load(daily) as z:
        assert np.array_equal(z["days"], n_run["days"])
        for lane, name in enumerate(names):
            for dk, ok in DAILY.items():
                np.testing.assert_array_equal(
                    out[ok][lane], z[f"{LANES[name][2]}.{dk}"], err_msg=f"{name} {dk}"
                )
    n_run["d1_daily_checked"] = True


def test_full_replay_stage_dates_are_the_embedded_ceres(n_run: dict) -> None:
    for row in _season_rows(n_run, n_run["names"].index("full_n")):
        ours, theirs = row["stgdoy"], row["rzwqm"]["stgdoy"]
        for s in (7, 8, 9, 1, 2, 3, 4, 5):
            if s == 5 and theirs[s - 1] == NOT_REACHED:
                assert ours[s - 1] == NOT_REACHED, row["season"]
                continue
            assert ours[s - 1] == theirs[s - 1], (row["season"], s, ours, theirs)


def test_against_rzwqm_is_recorded(n_run: dict, data_dir: Path) -> None:
    """Measured, not asserted: per season and lane against the RZWQM2 4.6 embedded CERES."""
    rep = {
        "what": (
            "DSSAT-CSM 4.8.6 CERES (ours, nstress_replay growth) with the RZWQM2 nitrogen factors "
            "replayed through P10, against RZWQM2 4.6 embedded DSSAT 4.0 CERES, CA-TPA 2015-2021"
        ),
        "lanes": {
            name: {"config": LANES[name], "seasons": _season_rows(n_run, lane)}
            for lane, name in enumerate(n_run["names"])
        },
        "species": n_run["species"],
        "d1_daily_checked": bool(n_run.get("d1_daily_checked", False)),
    }
    for lane in rep["lanes"].values():
        s = lane["seasons"]
        lane["total_yield_minus_rzwqm_kg_ha"] = float(sum(r["yield"] - r["rzwqm"]["yield"] for r in s))
    path = Path(os.environ.get("AGRI_JAX_DATA", str(data_dir))) / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1) + "\n")
    print("W4C " + json.dumps(rep))
    assert len(rep["lanes"]) == len(LANES)
