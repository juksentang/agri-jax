"""Season boundaries of CERES-Maize (per-season management, season start and end), no data.

* the per-season parameter table (``CeresSeasons``): without a table and with a one-row table
  equal to the scalar management the crop computes the same bits (``n_season = 1``);
  past the last row no season starts;
* ``ceres_season_init`` on a sowing day gives ``CeresMaizeState.initial`` of the crop's season bit
  for bit and leaves every other day alone; ``ceres_harvest`` on a harvest day sets the crop to the
  next season's SEASINIT state, P2 to the no-crop root record, P6 to bare soil, and counts the
  season, touching nothing outside the crop slot; ROOTWU's own ``rootwu_season_end`` sets its TSS,
  RWU and P1 ``trwup`` to 0 (a module resets only its own state);
* two seasons in one run: each season equals that season run alone, bit for bit (one vmapped
  program over the three scenarios); between the seasons the root record has no roots and
  ``XHLAI = 0``;
* the event table's ``sow`` flag starts the crop: with the flags later than
  ``YRPLT`` the crop starts on the flags and the run equals the one whose ``YRPLT`` is the flag
  days (``YRPLT`` is not read); without the season initialisation an event-driven crop never
  starts; a state that is not event-driven is rejected; ``check_sowing_dates`` compares the two;
* the contract's day with the season entries: ``Day.check`` passes with the five allowed lags,
  ROOTWU's TSS stays its own carried state (only ROOTWU's entries write it; a reset of another
  module's state is an undeclared lag that the check rejects), and a season initialisation that
  wrote the root and canopy records at the start of the day is rejected as hiding the P2 and P6
  lags.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import Day, Lag, Model, Phase, bind, compose, process, run
from agrijax.core.day import DayError, DayLagError, DayWriteError, PhasedWrite, snapshot
from agrijax.core.events import EventTable
from agrijax.core.ports import Binding
from agrijax.core.state import get_path, set_path
from agrijax.iface.contract import path_consumers, phased_writes
from agrijax.iface.crop import CanopyRecord
from agrijax.iface.surface import SnowOut
from agrijax.models.day_rzwqm46 import SLOT, day_processes, day_rzwqm46
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    NOT_REACHED,
    CeresMaizeState,
    CeresReplayForcing,
    CeresSeasons,
    ceres_crop_water_replay,
    ceres_harvest,
    ceres_maize_model,
    ceres_season_init,
    ceres_snow_replay,
    check_sowing_dates,
    root_record,
)
from agrijax.processes.crop.ceres_maize.season import OWN_FIELDS
from agrijax.processes.water_supply import RootwuState, rootwu_season_end, rootwu_trwup_replay
from agrijax.sites.dssat_inputs import yrdoy_range

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import make_params

P2 = ("water_supply.maize.rootwu", "iface.root.maize")
USED = {
    ("pet.sw_daily", "iface.canopy.maize"),
    ("pet.sw_daily", "soil_water.theta"),
    ("soil_water.uptake_limit", "iface.root_uptake.maize"),
    ("soil_water.uptake_limit", "iface.crop_water.maize.trwup"),
    P2,
}


def _leaves_equal(a, b) -> list[str]:
    """Paths of the leaves of ``a`` and ``b`` whose bytes differ (same structure required)."""
    fa = jax.tree_util.tree_flatten_with_path(a)[0]
    fb = jax.tree_util.tree_flatten_with_path(b)[0]
    assert len(fa) == len(fb)
    return [
        jax.tree_util.keystr(pa)
        for (pa, x), (_, y) in zip(fa, fb, strict=True)
        if np.asarray(x).tobytes() != np.asarray(y).tobytes() or np.asarray(x).dtype != np.asarray(y).dtype
    ]


# ------------------------------------------------------------------ the per-season table
def test_in_season_without_a_table_or_index_is_the_same_params() -> None:
    p = make_params()
    assert p.in_season(None) is p
    assert p.in_season(jnp.asarray(0, jnp.int32)) is p  # no table: the scalars are the only season
    t = p.replace(seasons=CeresSeasons.single(p))
    assert t.in_season(None) is t


def test_in_season_picks_the_row_and_no_season_after_the_last() -> None:
    p = make_params(yrplt=2001100)
    table = CeresSeasons(
        yrplt=jnp.asarray([2001100, 2002110], jnp.int32),
        pltpop=jnp.asarray([7.2, 8.0]),
        sdepth=jnp.asarray([7.0, 5.0]),
        rowspc=jnp.asarray([61.0, 76.0]),
    )
    q = p.replace(seasons=table)
    for k, (y, pop, sd, rs) in enumerate(((2001100, 7.2, 7.0, 61.0), (2002110, 8.0, 5.0, 76.0))):
        r = q.in_season(jnp.asarray(k, jnp.int32))
        assert int(r.yrplt) == y
        assert (float(r.pltpop), float(r.sdepth), float(r.rowspc)) == pytest.approx((pop, sd, rs), rel=1e-6)
        assert r.yrplt.dtype == jnp.int32 and r.yrplt.shape == ()
    after = q.in_season(jnp.asarray(2, jnp.int32))
    assert int(after.yrplt) == NOT_REACHED and float(after.pltpop) == pytest.approx(8.0)
    s = CeresMaizeState.initial(q, 2, season=1)
    assert int(np.asarray(s.season)) == 1 and np.all(np.asarray(s.growth.pltpop) == 8.0)
    assert CeresMaizeState.initial(p, 1).season is None
    assert int(np.asarray(CeresMaizeState.initial(q, 1).season)) == 0


def test_one_row_table_is_the_single_season_bit_for_bit() -> None:
    """A one-row season table (``n_season = 1``) changes no bit of the crop (states and PlantGro outputs)."""
    f, w = season_forcing(71, stress=True, waterlog=True, n=200)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    q = p.replace(seasons=CeresSeasons.single(p))
    model = ceres_maize_model(outputs=lambda s, pp, ff: s)
    go = jax.jit(lambda pp, ff, s: run(model, pp, ff, s))
    a = go(p, f, CeresMaizeState.initial(p, 1))
    b = go(q, f, CeresMaizeState.initial(q, 1))
    assert np.all(np.asarray(b.season) == 0)
    assert _leaves_equal(a, b.replace(season=None)) == []
    assert float(np.max(np.asarray(a.growth.biomas))) > 0.0


# ------------------------------------------------------------------ the two processes
def _mid_season(seed: int = 72, day: int = 80):
    """A grain-fill state with a two-row table (season 0 running) and filled ports."""
    f, w = season_forcing(seed, stress=True, n=day + 5)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    q = p.replace(
        seasons=CeresSeasons(
            yrplt=jnp.asarray([int(w["yrdoy"][2]), 2002100], jnp.int32),
            pltpop=jnp.asarray([7.2, 9.0]),
            sdepth=jnp.asarray([7.0, 4.0]),
            rowspc=jnp.asarray([61.0, 70.0]),
        )
    )
    traj = jax.jit(lambda pp, ff, s: run(ceres_maize_model(outputs=lambda s_, p_, f_: s_), pp, ff, s))(
        q, f, CeresMaizeState.initial(q, 2)
    )
    s = jax.tree_util.tree_map(lambda x: x[day], traj)
    rng = np.random.default_rng(seed)
    n_crop = s.roots.rlv.shape[0]
    s = s.replace(
        canopy_out=CanopyRecord(
            lai=jnp.asarray(rng.uniform(1, 4, n_crop)),
            tlai=jnp.asarray(rng.uniform(1, 4, n_crop)),
            height=jnp.asarray(rng.uniform(50, 200, n_crop)),
        ),
        sowing=jnp.asarray(False),  # an event-driven crop (its sowing leaf is the day's sow flag)
    )
    assert float(np.max(np.asarray(s.growth.lai))) > 0.5 and int(s.season) == 0
    return s, q


def _events(**flags: bool) -> EventTable:
    ev = EventTable.empty(1)
    ev = ev.replace(**{k: jnp.asarray([v]) for k, v in flags.items()})
    return ev.day(0)


def test_season_init_on_a_sowing_day_is_seasinit() -> None:
    s, q = _mid_season()
    new = ceres_season_init(s, q, _events(sow=True))
    fresh = CeresMaizeState.initial(q, 2, season=0)
    for n in OWN_FIELDS:
        assert _leaves_equal(getattr(new, n), getattr(fresh, n)) == [], n
    # the sowing leaf is the day's flag; the ports and the season index are untouched
    assert bool(new.sowing) is True
    kept = dataclasses.replace(new, sowing=s.sowing, **{n: getattr(s, n) for n in OWN_FIELDS})
    assert _leaves_equal(kept, s) == []
    same = ceres_season_init(s, q, _events(sow=False))
    assert _leaves_equal(same, s) == [] and bool(same.sowing) is False


def test_season_init_rejects_a_crop_that_is_not_event_driven() -> None:
    """Without the ``sowing`` leaf the crop would start on YRPLT and ignore the sow flags."""
    s, q = _mid_season()
    with pytest.raises(ValueError, match="event-driven"):
        ceres_season_init(s.replace(sowing=None), q, _events(sow=True))


def test_harvest_resets_the_crop_and_its_records() -> None:
    s, q = _mid_season()
    assert float(np.max(np.asarray(s.water_in.trwup))) > 0.0
    new = ceres_harvest(s, q, _events(harvest=True))
    assert int(new.season) == 1
    fresh = CeresMaizeState.initial(q, 2, season=1)
    for n in OWN_FIELDS:
        assert _leaves_equal(getattr(new, n), getattr(fresh, n)) == [], n
    np.testing.assert_allclose(np.asarray(new.growth.pltpop), 9.0, rtol=1e-6)  # the next season's population
    # P1 is another slot's record: the crop's harvest leaves it alone (ROOTWU zeroes its trwup)
    assert _leaves_equal(new.water_in, s.water_in) == []
    assert _leaves_equal(new.root_out, root_record(fresh, q)) == []
    for x in (new.root_out.rlv, new.root_out.rtdep, new.root_out.xhlai):
        np.testing.assert_array_equal(np.asarray(x), 0.0)
    for x in jax.tree_util.tree_leaves(new.canopy_out):
        np.testing.assert_array_equal(np.asarray(x), 0.0)
    assert _leaves_equal(new.snow_in, s.snow_in) == [] and _leaves_equal(new.n_in, s.n_in) == []
    assert _leaves_equal(ceres_harvest(s, q, _events(harvest=False)), s) == []
    bound = bind(ceres_harvest, own="crops.maize", ports=HARVEST_PORTS, forcing="events", params="crop")
    assert all(w.startswith(("crops.maize.", "iface.root.maize", "iface.canopy.maize")) for w in bound.writes)


def test_rootwu_season_end_resets_its_own_state_and_trwup() -> None:
    """On a harvest day ROOTWU's TSS, RWU and P1 trwup are 0; sw, eop, the root
    record and every other day are untouched."""
    rng = np.random.default_rng(75)
    n_crop, n_layer = 2, 6
    u = lambda *shape: jnp.asarray(rng.uniform(0.1, 3.0, shape))  # noqa: E731
    from agrijax.iface.crop import CropWaterIn, RootRecord

    s = RootwuState(
        tss=u(n_crop, n_layer),
        rwu=u(n_crop, n_layer),
        root=RootRecord(
            rlv=u(n_crop, n_layer), rtdep=u(n_crop), rwumx=u(n_crop), pormin=u(n_crop), xhlai=u(n_crop)
        ),
        water=CropWaterIn(sw=u(n_layer), eop=u(n_crop), trwup=u(n_crop)),
    )
    new = rootwu_season_end(s, None, _events(harvest=True))
    for x in (new.tss, new.rwu, new.water.trwup):
        np.testing.assert_array_equal(np.asarray(x), 0.0)
    assert _leaves_equal(new.water.replace(trwup=s.water.trwup), s.water) == []
    assert _leaves_equal(new.root, s.root) == []
    assert _leaves_equal(rootwu_season_end(s, None, _events(harvest=False)), s) == []
    bound = bind(
        rootwu_season_end,
        own="water_supply.maize",
        ports={"water": "iface.crop_water.maize"},
        forcing="events",
    )
    assert set(bound.writes) == {
        "water_supply.maize.tss",
        "water_supply.maize.rwu",
        "iface.crop_water.maize.trwup",
    }


# ------------------------------------------------------------------ two seasons in one run
S1_SOW, S1_HARVEST, S2_SOW, S2_HARVEST, N_DAYS = 2, 150, 400, 540, 560
PORTS = {"water_in": "iface.crop_water.maize", "root_out": "iface.root.maize", "snow_in": "iface.snow"}
HARVEST_PORTS = {"root_out": "iface.root.maize", "canopy_out": "iface.canopy.maize"}
TRWUP = "iface.crop_water.maize.trwup"
#: the day's value of P1 trwup, recorded before the season end (the daily output record)
DAY_TRWUP = "prev.day.trwup"
SEASON_ENTRIES = (
    "crops.maize.season_init",
    "crops.maize.water_replay",
    "water_supply.maize.rootwu",
    "crops.maize.snow_replay",
    *(f"crops.maize.{n}" for n in ("phenology", "stress", "growth", "roots", "publish")),
    "prev.day_output",
    "crops.maize.harvest",
    "water_supply.maize.season_end",
)
SEASON_DAY = Day(
    ref="none",
    phases=(
        Phase("management", SEASON_ENTRIES[:1]),
        Phase("plant", SEASON_ENTRIES[1:]),
    ),
    # the contract's declared season ends of the crop (P2, P6) and of ROOTWU (P1 trwup, rwu, tss)
    phased_writes=tuple(w for w in phased_writes(SLOT) if w.entry in SEASON_ENTRIES),
)


def _season_model(init=ceres_season_init) -> Model:
    own = "crops.maize"
    procs = {
        "crops.maize.season_init": bind(init, own=own, forcing="events", params="crop"),
        # the replay of P1: sw and eop for the crop-side producers, trwup for ROOTWU (its owner)
        "crops.maize.water_replay": bind(ceres_crop_water_replay, own=own, ports=PORTS, forcing="crop"),
        "water_supply.maize.rootwu": bind(
            rootwu_trwup_replay, own="water_supply.maize", ports={"water": PORTS["water_in"]}, forcing="crop"
        ),
        "crops.maize.snow_replay": bind(
            ceres_snow_replay, own=own, ports=PORTS, forcing="crop", params="crop"
        ),
        **{
            f"crops.maize.{n}": bind(proc, own=own, ports=PORTS, forcing="crop", params="crop")
            for n, proc in zip(
                ("phenology", "stress", "growth", "roots", "publish"), CROP_PROCESSES, strict=True
            )
        },
        "prev.day_output": snapshot("prev.day_output", {TRWUP: DAY_TRWUP}),
        "crops.maize.harvest": bind(
            ceres_harvest, own=own, ports=HARVEST_PORTS, forcing="events", params="crop"
        ),
        "water_supply.maize.season_end": bind(
            rootwu_season_end,
            own="water_supply.maize",
            ports={"water": "iface.crop_water.maize"},
            forcing="events",
        ),
    }

    def outputs(s, pp, ff):
        return {
            "crop": get_path(s, own),
            "root": get_path(s, "iface.root.maize"),
            "trwup": get_path(s, TRWUP),
            "day_trwup": get_path(s, DAY_TRWUP),
            "tss": get_path(s, "water_supply.maize.tss"),
        }

    # the end-of-day root record, TRWUP and TSS are the season ends' values on harvest days on purpose
    end = ("iface.root.maize", TRWUP, "water_supply.maize.tss")
    return SEASON_DAY.compile(
        procs, outputs=outputs, output_reads=(own, *end, DAY_TRWUP), end_of_day_reads=end
    )


def _two_season_inputs(sow_shift=(0, 0), yrplt_shift=(0, 0)):
    """Inputs of the three scenarios; the ``k``-th season's sow flag is ``sow_shift[k]`` days after
    its row's ``YRPLT``, which is ``yrplt_shift[k]`` days after the season's default sowing day."""
    f, _ = season_forcing(73, stress=True, n=N_DAYS)
    yrdoy = np.asarray(yrdoy_range(2001090, 2001090 + 10_000)[:N_DAYS], np.int32)  # monotonic dates
    f = f.replace(yrdoy=jnp.asarray(yrdoy))
    p = make_params(yrplt=int(yrdoy[S1_SOW + yrplt_shift[0]]))
    y1, y2 = int(yrdoy[S1_SOW + yrplt_shift[0]]), int(yrdoy[S2_SOW + yrplt_shift[1]])
    f1, f2 = S1_SOW + yrplt_shift[0] + sow_shift[0], S2_SOW + yrplt_shift[1] + sow_shift[1]

    def table(rows):
        rows = list(rows) + [(NOT_REACHED, 7.2, 7.0, 61.0)] * (2 - len(rows))
        y, pop, sd, rs = (np.asarray(c) for c in zip(*rows, strict=True))
        return CeresSeasons(
            yrplt=jnp.asarray(y, jnp.int32),
            pltpop=jnp.asarray(pop),
            sdepth=jnp.asarray(sd),
            rowspc=jnp.asarray(rs),
        )

    s1, s2 = (y1, 7.2, 7.0, 61.0), (y2, 8.1, 5.5, 70.0)
    scen = [  # (table, sow days, harvest days): both seasons, the first alone, the second alone
        (table([s1, s2]), (f1, f2), (S1_HARVEST, S2_HARVEST)),
        (table([s1]), (f1,), (S1_HARVEST,)),
        (table([s2]), (f2,), (S2_HARVEST,)),
    ]
    params, events = [], []
    for t, sows, harvests in scen:
        params.append({"crop": p.replace(seasons=t)})
        ev = EventTable.empty(N_DAYS)
        events.append(
            ev.replace(
                sow=jnp.asarray(np.isin(np.arange(N_DAYS), sows)),
                harvest=jnp.asarray(np.isin(np.arange(N_DAYS), harvests)),
            )
        )
    stack = lambda xs: jax.tree_util.tree_map(lambda *a: jnp.stack(a), *xs)  # noqa: E731
    forcing = {"crop": f, "events": stack(events)}
    n_layer = int(p.soil.dlayr.shape[0])
    states = []
    for pp in params:
        s0 = CeresMaizeState.initial(pp["crop"], 1, events=True)
        states.append(
            compose(
                {
                    **Binding("crops.maize", tuple(PORTS.items())).entries(s0),
                    "iface.canopy.maize": CanopyRecord.zeros(1),
                    "water_supply.maize": RootwuState.initial(1, n_layer),
                    "prev.day": {"trwup": jnp.zeros(1)},
                }
            )
        )
    return stack(params), forcing, stack(states), yrdoy


def test_two_seasons_in_one_run_equal_each_season_alone() -> None:
    model = _season_model()
    params, forcing, states, yrdoy = _two_season_inputs()
    run_all = jax.jit(
        jax.vmap(
            lambda pp, ev, s: run(model, pp, {"crop": forcing["crop"], "events": ev}, s), in_axes=(0, 0, 0)
        )
    )
    out = run_all(params, forcing["events"], states)
    crop, root = out["crop"], out["root"]
    both = jax.tree_util.tree_map(lambda x: x[0], crop)
    for lane, (a, b) in ((1, (S1_SOW, S1_HARVEST)), (2, (S2_SOW, S2_HARVEST))):
        alone = jax.tree_util.tree_map(lambda x, lane=lane: x[lane], crop)
        win = slice(a, b)  # the end-of-day states from sowing to the day before harvest
        for n in ("phen", "stress", "growth", "roots"):
            got = jax.tree_util.tree_map(lambda x, win=win: x[win], getattr(both, n))
            want = jax.tree_util.tree_map(lambda x, win=win: x[win], getattr(alone, n))
            assert _leaves_equal(got, want) == [], (lane, n)
        assert float(np.max(np.asarray(alone.growth.biomas[win]))) > 0.0  # the season really grew
    istage = np.asarray(both.phen.istage)[:, 0]
    season = np.asarray(both.season)
    crop_day = np.zeros(N_DAYS, bool)
    crop_day[S1_SOW : S1_HARVEST + 1] = crop_day[S2_SOW : S2_HARVEST + 1] = True
    # at the end of each day the crop is past the sowing stage (7) from the sowing day to the day
    # before harvest; the harvest day ends in the next season's SEASINIT state
    past_sowing = crop_day & ~np.isin(np.arange(N_DAYS), (S1_HARVEST, S2_HARVEST))
    np.testing.assert_array_equal(istage != 7, past_sowing)
    np.testing.assert_array_equal(season, np.cumsum(np.isin(np.arange(N_DAYS), (S1_HARVEST, S2_HARVEST))))
    rl = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), root)
    off = ~past_sowing  # the root record written at the end of these days is the no-crop record
    for x in (rl.rlv, rl.rtdep, rl.xhlai):
        assert np.all(x[off] == 0.0)
    assert (
        float(np.max(rl.xhlai[S1_SOW:S1_HARVEST])) > 0.0 and float(np.max(rl.xhlai[S2_SOW:S2_HARVEST])) > 0.0
    )
    trwup = np.asarray(out["trwup"])[0, :, 0]
    assert np.all(trwup[[S1_HARVEST, S2_HARVEST]] == 0.0) and trwup[S1_HARVEST - 1] > 0.0
    # the daily output record (before the season end) has the day's TRWUP, also on harvest days;
    # the end of the day differs from it only there
    day_trwup = np.asarray(out["day_trwup"])[0, :, 0]
    want = np.asarray(forcing["crop"].trwup, dtype=day_trwup.dtype)
    np.testing.assert_array_equal(day_trwup, want)
    assert np.all(day_trwup[[S1_HARVEST, S2_HARVEST]] > 0.0)
    np.testing.assert_array_equal(np.nonzero(day_trwup != trwup)[0], [S1_HARVEST, S2_HARVEST])
    assert np.all(np.asarray(out["tss"])[0, [S1_HARVEST, S2_HARVEST]] == 0.0)
    # the second season starts on its own sowing date (row 1 of the table), not before
    assert istage[S2_SOW - 1] == 7 and istage[S2_SOW] != 7 and int(yrdoy[S2_SOW]) != int(yrdoy[S1_SOW])


def _run_three(model, sow_shift=(0, 0), yrplt_shift=(0, 0)):
    params, forcing, states, _ = _two_season_inputs(sow_shift, yrplt_shift)
    go = jax.jit(
        jax.vmap(
            lambda pp, ev, s: run(model, pp, {"crop": forcing["crop"], "events": ev}, s), in_axes=(0, 0, 0)
        )
    )
    return go(params, forcing["events"], states), forcing, params


def test_the_sow_flag_starts_the_crop_and_yrplt_is_not_read() -> None:
    """The sowing decision is the event's. With the sow flags 3 and 5 days
    after the table's YRPLT the crop starts on the flag days (not on YRPLT), and the run is the
    same bit for bit as with a table whose YRPLT is the flag days."""
    model = _season_model()
    late, _, _ = _run_three(model, sow_shift=(3, 5))
    same, _, _ = _run_three(model, yrplt_shift=(3, 5))
    assert _leaves_equal(late, same) == []
    istage = np.asarray(late["crop"].phen.istage)[0, :, 0]
    for flag in (S1_SOW + 3, S2_SOW + 5):
        assert istage[flag - 1] == 7 and istage[flag] != 7, flag
    stg = np.asarray(late["crop"].phen.stgdoy)[0, :, 0]  # STGDOY of the sowing stage (7): the flag day
    yrdoy = np.asarray(_two_season_inputs()[3])
    assert int(stg[S1_HARVEST - 1, 7 - 1]) == int(yrdoy[S1_SOW + 3])
    assert int(stg[S2_HARVEST - 1, 7 - 1]) == int(yrdoy[S2_SOW + 5])
    # the season really grew from the late flag on
    assert float(np.max(np.asarray(late["crop"].growth.biomas)[0, S1_SOW + 3 : S1_HARVEST])) > 0.0


def _no_season_init(state, params, forcing_t):
    """Fixture: a season initialisation that does nothing.

    Source: fixture (a day without the season entry).
    """
    return state


def test_without_season_init_the_event_driven_crop_never_starts() -> None:
    """The season initialisation is what starts an event-driven crop: without it (an identity in
    its place) the crop stays at stage 7 all run long, whatever the table's YRPLT."""
    idle = process(_no_season_init, reads=(), writes=(), name="idle", register=False, source="fixture")
    out, _, _ = _run_three(_season_model(init=idle))
    assert np.all(np.asarray(out["crop"].phen.istage) == 7)
    assert np.all(np.asarray(out["crop"].growth.biomas) == 0.0)


def test_check_sowing_dates() -> None:
    params, forcing, _, yrdoy = _two_season_inputs()
    ev = forcing["events"]
    for lane in range(3):
        check_sowing_dates(
            np.asarray(ev.sow)[lane],
            yrdoy,
            jax.tree_util.tree_map(lambda x, lane=lane: x[lane], params["crop"].seasons),
        )
    _, forcing_late, _, _ = _two_season_inputs(sow_shift=(3, 5))
    with pytest.raises(ValueError, match="not the season table's YRPLT"):
        check_sowing_dates(
            np.asarray(forcing_late["events"].sow)[0],
            yrdoy,
            jax.tree_util.tree_map(lambda x: x[0], params["crop"].seasons),
        )
    with pytest.raises(ValueError, match="differ in shape"):
        check_sowing_dates(np.zeros(3, bool), yrdoy, params["crop"].seasons)


# ------------------------------------------------------------------ the contract's day
def test_contract_day_with_the_season_entries_keeps_the_five_lags() -> None:
    day = day_rzwqm46(SLOT)
    model = day.compile(day_processes(SLOT, seasons=True), outputs=())
    report = day.check(model)
    assert set(report.used) == USED and report.unused_pairs == ()
    assert day.hidden_lags(model) == []
    assert model.writers("iface.root.maize") == ("crops.maize.publish", "crops.maize.harvest")
    assert model.writers("iface.canopy.maize") == ("crops.maize.canopy", "crops.maize.harvest")
    by = {p.name: p for p in model.processes}
    assert by["crops.maize.season_init"].writes == (
        *(f"crops.maize.{n}" for n in OWN_FIELDS),
        "crops.maize.sowing",
    )
    # ROOTWU's TSS stays its own carried state, written only by ROOTWU's entries,
    # and P1 trwup has one writer module (ROOTWU: its uptake and its season end)
    assert ("water_supply.maize.rootwu", "water_supply.maize.tss") in day.carried_reads(model)
    assert model.writers("water_supply.maize.tss") == (
        "water_supply.maize.rootwu",
        "water_supply.maize.season_end",
    )
    assert model.writers("iface.crop_water.maize.trwup") == (
        "water_supply.maize.rootwu",
        "water_supply.maize.season_end",
    )
    for name in ("crops.maize.season_init", "crops.maize.harvest"):
        assert all(w.startswith(("crops.maize.", "iface.root.", "iface.canopy.")) for w in by[name].writes), (
            name
        )


def _harvest_resetting_rootwu(state, params, forcing_t):
    """Fixture: a crop harvest that also clears ROOTWU's TSS (a cross-module reset).

    Source: fixture (an arrangement the day check rejects: no module resets another module's state).
    """
    tss = get_path(state, "water_supply.maize.tss")
    return set_path(
        state, "water_supply.maize.tss", jnp.where(forcing_t["events"].harvest, jnp.zeros_like(tss), tss)
    )


def test_a_crop_entry_resetting_rootwu_state_is_rejected() -> None:
    """A crop entry that writes
    ``water_supply.<slot>.tss`` (ROOTWU's state) makes ROOTWU's read of its own carried TSS a lag of
    another module, which the day check rejects; the declaration that would excuse it is rejected
    too (a module declares writes of its own state and out ports only)."""
    day = day_rzwqm46(SLOT)
    bad = process(
        _harvest_resetting_rootwu,
        reads=("water_supply.maize.tss",),
        writes=("water_supply.maize.tss",),
        name="crops.maize.harvest",
        register=False,
        source="fixture",
    )
    model = day.compile(day_processes(SLOT, seasons=True, replace={"crops.maize.harvest": bad}), check=False)
    assert ("water_supply.maize.rootwu", "water_supply.maize.tss") in day.lagged_reads(model)
    with pytest.raises(DayLagError, match=r"water_supply\.maize\.rootwu <- water_supply\.maize\.tss"):
        day.check(model)
    excuse = PhasedWrite("crops.maize.harvest", "water_supply.maize.tss", "season_end", "fixture")
    with pytest.raises(DayError, match="state of another module"):
        dataclasses.replace(day, phased_writes=(*day.phased_writes, excuse))


def test_the_contract_day_rejects_its_season_ends_undeclared_or_early() -> None:
    """Without the phased-write declarations the season ends of ROOTWU and of the crop are
    undeclared extra writes of paths other modules read; ROOTWU's season end moved before the
    crop's readers of P1 is a reset before a same-day consumer."""
    day = day_rzwqm46(SLOT)
    procs = day_processes(SLOT, seasons=True)
    model = day.compile(procs, outputs=())
    assert day.write_phase_problems(model) == [] and day.owner_problems(model) == []
    # a contract day takes the contract's declarations: a bare day is not a contract day
    with pytest.raises(DayError, match="phased writes differ from the contract"):
        dataclasses.replace(day, phased_writes=day.phased_writes[:1])
    # opting out of the contract takes the explicit fixture flag, not an empty contract_slot
    with pytest.raises(DayError, match="registered contract"):
        dataclasses.replace(day, phased_writes=(), owners=(), contract_slot="")
    bare = dataclasses.replace(day, phased_writes=(), owners=(), contract_slot="", bare=True)
    probs = bare.write_phase_problems(model)
    for path in (
        "iface.crop_water.maize.trwup",
        "water_supply.maize.rwu",
        "iface.root.maize",
        "iface.canopy.maize",
    ):
        assert any(path in p and "undeclared" in p for p in probs), (path, probs)
    with pytest.raises(DayWriteError, match="undeclared"):
        bare.check(model)
    early = [e for e in day.entries if e != "water_supply.maize.season_end"]
    early.insert(early.index("water_supply.maize.rootwu") + 1, "water_supply.maize.season_end")
    moved = dataclasses.replace(
        day,
        phases=(
            *day.phases[:-2],
            Phase("plant", tuple(e for e in early if day.phase_of(e) == "plant")),
            day.phases[-1],
        ),
    )
    m2 = moved.compile(procs, check=False, outputs=())
    with pytest.raises(DayWriteError, match=r"before its same-day consumer crops\.maize\.stress"):
        moved.check(m2)


@pytest.mark.parametrize("path", ["iface.crop_water.maize.trwup", "water_supply.maize.rwu"])
def test_a_crop_harvest_writing_rootwu_output_or_state_is_rejected(path: str) -> None:
    """On the compiled day, the crop's harvest with an extra write of P1 trwup (owned
    by the water-supply module) or of ROOTWU's rwu is a second writer module."""
    day = day_rzwqm46(SLOT)
    procs = day_processes(SLOT, seasons=True)
    h = procs["crops.maize.harvest"]
    procs["crops.maize.harvest"] = dataclasses.replace(h, writes=(*h.writes, path))
    probs = day.owner_problems(day.compile(procs, check=False, outputs=()))
    assert any(p.startswith(f"crops.maize.harvest writes {path}") for p in probs), probs
    with pytest.raises(DayError):
        day.compile(procs, outputs=())


def test_the_contract_consumers_are_the_compiled_readers() -> None:
    """AJ013 (contract) and Day.check (compiled day) see the same consumers: for every declared
    season-end path, the entries of other modules that read it in the compiled contract day are the
    contract's consumers of the path (port field consumers, SHARED_STATE readers)."""
    day = day_rzwqm46(SLOT)
    model = day.compile(day_processes(SLOT, seasons=True), outputs=())
    from agrijax.core.model import _overlap

    for d in day.phased_writes:
        readers = {
            p.name
            for p in model.processes
            if Day.module_of(p.name) != Day.module_of(d.entry) and any(_overlap(r, d.path) for r in p.reads)
        }
        want = {c.format(slot=SLOT) for c in path_consumers(d.path.replace(SLOT, "{slot}"))}
        assert readers == want, (d.path, readers, want)


def _start_of_day_reset(state, params, forcing_t):
    """Fixture: a season initialisation that also resets the root and canopy records.

    Source: fixture (an arrangement that hides lags from the day check).
    """
    for path in ("iface.root.maize", "iface.canopy.maize"):
        rec = get_path(state, path)
        rec = jax.tree_util.tree_map(lambda x: jnp.where(forcing_t["events"].sow, jnp.zeros_like(x), x), rec)
        state = set_path(state, path, rec)
    return state


def test_a_start_of_day_reset_of_the_records_is_a_hidden_lag() -> None:
    day = day_rzwqm46(SLOT)
    bad = process(
        _start_of_day_reset,
        reads=("iface.root.maize", "iface.canopy.maize"),
        writes=("iface.root.maize", "iface.canopy.maize"),
        name="crops.maize.season_init",
        register=False,
        source="fixture",
    )
    model = day.compile(
        day_processes(SLOT, seasons=True, replace={"crops.maize.season_init": bad}), check=False
    )
    # without the guard the P2 and P6 lags would silently disappear from the order analysis
    lagged = set(day.lagged_reads(model))
    assert P2 not in lagged and ("pet.sw_daily", "iface.canopy.maize") not in lagged
    hidden = {(r, p) for r, p, _, _ in day.hidden_lags(model)}
    assert hidden == {P2, ("pet.sw_daily", "iface.canopy.maize")}
    with pytest.raises(DayLagError, match="hidden by an earlier writer"):
        day.check(model)


# ------------------------------------------------------------------ cross-module resets and hidden lags
def _w(path, read=True):
    def fn(s, p, f):
        """Source: fixture."""
        return set_path(s, path, (get_path(s, path) if read else 0.0) + 1.0)

    return fn


def test_a_reset_of_another_modules_state_is_an_undeclared_lag() -> None:
    """There is no cross-module reset declaration. Module ``a`` writing module
    ``b``'s carried ``b.x`` at the end of the day makes ``b``'s read a lag of another module,
    which the day check rejects; ``b`` resetting its own state is carried state."""
    procs = {
        "b.step": process(_w("b.x"), reads=("b.x",), writes=("b.x",), register=False, source="fixture"),
        "a.reset": process(_w("b.x", False), reads=(), writes=("b.x",), register=False, source="fixture"),
    }
    day = Day(ref="none", phases=(Phase("p", ("b.step", "a.reset")),))
    with pytest.raises(DayLagError, match=r"b\.step <- b\.x"):
        day.compile(procs)
    own = {
        "b.step": procs["b.step"],
        "b.reset": process(_w("b.x", False), reads=(), writes=("b.x",), register=False, source="fixture"),
    }
    day = Day(ref="none", phases=(Phase("p", ("b.step", "b.reset")),))
    model = day.compile(own)
    assert day.lagged_reads(model) == [] and day.carried_reads(model) == [("b.step", "b.x")]


def test_hidden_lag_needs_an_allowed_lag_and_writers_on_both_sides() -> None:
    procs = {
        "a.early": process(_w("c.x"), reads=(), writes=("c.x",), register=False, source="fixture"),
        "b.read": process(_w("b.y"), reads=("c.x",), writes=("b.y",), register=False, source="fixture"),
        "c.late": process(_w("c.x"), reads=(), writes=("c.x",), register=False, source="fixture"),
    }
    phases = (Phase("p", ("a.early", "b.read", "c.late")),)
    day = Day(ref="none", phases=phases, lags=(Lag("b.read", "c.x", evidence="fixture"),))
    model = day.compile(procs, check=False)
    assert day.hidden_lags(model) == [("b.read", "c.x", "a.early", "c.late")]
    with pytest.raises(DayLagError, match="hidden"):
        day.check(model)
    # no allowed lag on the read: an ordinary same-day read, nothing hidden
    plain = Day(ref="none", phases=phases)
    assert plain.hidden_lags(plain.compile(procs, check=False)) == []
    # no late writer: the allowed lag is merely unused
    procs2 = {k: v for k, v in procs.items() if k != "c.late"}
    day2 = Day(ref="none", phases=(Phase("p", ("a.early", "b.read")),), lags=day.lags)
    assert day2.hidden_lags(day2.compile(procs2, check=False)) == []
    assert day2.check(day2.compile(procs2)).unused_pairs == (("b.read", "c.x"),)


def test_snow_port_record_is_read_by_the_phenology() -> None:
    """The crop's ``SNOW`` is the ``swe`` of the P9 snow record; the crop's own forcing has no snow field."""
    f, w = season_forcing(74, n=40)
    p = make_params(yrplt=int(w["yrdoy"][0]))
    cold = jnp.arange(40) < 15
    f = f.replace(
        tmin=jnp.where(cold, -3.0, f.tmin),
        tmax=jnp.where(cold, 20.0, f.tmax),  # a warm day: the crown temperature of a frosty night counts
        snow=jnp.where(cold, 12.0, 0.0).astype(f.tmax.dtype),
    )
    model = ceres_maize_model(outputs=lambda s, pp, ff: s)
    go = jax.jit(lambda pp, ff, s0: run(model, pp, ff, s0))
    s = go(p, f, CeresMaizeState.initial(p, 1))
    np.testing.assert_array_equal(np.asarray(s.snow_in.swe), np.asarray(f.snow))
    no_snow = go(p, f.replace(snow=jnp.zeros_like(f.snow)), CeresMaizeState.initial(p, 1))
    dtt, dtt0 = np.asarray(s.phen.dtt)[:15, 0], np.asarray(no_snow.phen.dtt)[:15, 0]
    assert np.all(dtt > 0.0) and np.any(dtt != dtt0)  # under snow the crown temperature replaces the soil one
    assert isinstance(CeresMaizeState.initial(p, 1).snow_in, SnowOut) and isinstance(f, CeresReplayForcing)
