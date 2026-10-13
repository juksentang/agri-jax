"""Season start of CERES-Maize (per-season management), no data.

* the per-season parameter table (``CeresSeasons``): without a table and with a one-row table
  equal to the scalar management the crop computes the same bits (``n_season = 1``);
  past the last row no season starts;
* ``ceres_season_init`` on a sowing day gives ``CeresMaizeState.initial`` of the crop's season bit
  for bit and leaves every other day alone; a state that is not event-driven is rejected;
* ``check_sowing_dates`` compares the event table's sow flags with the season table;
* the crop reads its snow from the P9 record.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import compose, run
from agrijax.core.events import EventTable
from agrijax.core.ports import Binding
from agrijax.iface.crop import CanopyRecord
from agrijax.iface.surface import SnowOut
from agrijax.processes.crop.ceres_maize import (
    NOT_REACHED,
    CeresMaizeState,
    CeresReplayForcing,
    CeresSeasons,
    ceres_maize_model,
    ceres_season_init,
    check_sowing_dates,
)
from agrijax.processes.crop.ceres_maize.season import OWN_FIELDS
from agrijax.processes.water_supply import RootwuState
from agrijax.sites.dssat_inputs import yrdoy_range

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import make_params


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


# ------------------------------------------------------------------ the season start
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


# ------------------------------------------------------------------ the sowing dates of two seasons
S1_SOW, S1_HARVEST, S2_SOW, S2_HARVEST, N_DAYS = 2, 150, 400, 540, 560
PORTS = {"water_in": "iface.crop_water.maize", "root_out": "iface.root.maize", "snow_in": "iface.snow"}


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
