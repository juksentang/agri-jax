"""The ``nstress_replay`` growth variant of CERES-Maize and the P10 replay producer (M3 contract
decision 2), data-free.

* with ``NSTRES = 1`` the variant is the faithful nitrogen-off growth bit for bit, one day at a
  time and over a whole season;
* ``CARBO = PCARB min(PRFT, SWFAC, NSTRES) SLPF``: on every growing day of a synthetic season,
  from the same pre-growth state, a small ``NSTRES`` (below ``PRFT`` and ``SWFAC``) scales
  ``CARBO`` linearly, a large one leaves it unchanged, and ``CARBO`` never increases with
  stress; nothing but the growth state and the phenology it writes changes;
* bound in an assembly, the replay producer writes ``iface.crop_n.maize`` from the forcing and
  the variant reads it the same day: the bound run with ``NSTRES = 1`` equals the faithful bound
  run bit for bit, a stressed series lowers biomass, and ``AGRI_JAX_CHECK=1`` accepts the writes;
* the faithful state is unchanged: ``CeresMaizeState.initial`` leaves ``n_in`` empty.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import Day, Model, Phase, bind, compose, run
from agrijax.core.ports import Binding
from agrijax.core.process import CHECK_ENV
from agrijax.core.state import get_path, leaf_paths
from agrijax.iface.crop import CropNIn
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    CROP_PROCESSES_NSTRESS_REPLAY,
    CeresMaizeState,
    ceres_growth,
    ceres_growth_nstress_replay,
    ceres_phenology,
    ceres_stress,
    ceres_water_replay,
    plantgro_outputs,
)
from agrijax.processes.n_supply import CropNReplayForcing, CropNReplayState, crop_n_replay

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import make_params

_STATES = Model(CeresMaizeState, [ceres_water_replay, *CROP_PROCESSES], outputs=lambda s, p, f: s)
_STATES_V = Model(
    CeresMaizeState, [ceres_water_replay, *CROP_PROCESSES_NSTRESS_REPLAY], outputs=lambda s, p, f: s
)


def _season(seed: int, n: int = 200):
    f, w = season_forcing(seed, stress=True, n=n)
    return f, make_params(yrplt=int(w["yrdoy"][2]))


def _with_n(s: CeresMaizeState, nstres) -> CeresMaizeState:
    ns = jnp.broadcast_to(jnp.asarray(nstres, dtype=s.growth.lai.dtype), s.growth.lai.shape)
    return dataclasses.replace(s, n_in=CropNIn(nstres=ns))


def _bytes_equal(a, b) -> list[str]:
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    names = leaf_paths(a)
    return [
        n for n, x, y in zip(names, la, lb, strict=True) if np.asarray(x).tobytes() != np.asarray(y).tobytes()
    ]


def test_initial_state_leaves_the_nitrogen_port_empty() -> None:
    p = make_params()
    s0 = CeresMaizeState.initial(p, 1)
    assert s0.n_in is None
    assert not any(k.startswith("n_in") for k in leaf_paths(s0))


def test_nstres_one_is_the_faithful_season_bitwise() -> None:
    f, p = _season(61)
    s0 = CeresMaizeState.initial(p, 1)
    faithful = jax.jit(lambda pp, ff, s: run(_STATES, pp, ff, s))(p, f, s0)
    variant = jax.jit(lambda pp, ff, s: run(_STATES_V, pp, ff, s))(p, f, _with_n(s0, 1.0))
    assert variant.n_in is not None
    np.testing.assert_array_equal(np.asarray(variant.n_in.nstres), 1.0)
    assert _bytes_equal(dataclasses.replace(variant, n_in=None), faithful) == []
    assert float(np.max(np.asarray(faithful.growth.biomas))) > 0.0


def _pre_growth_states(seed: int):
    """Every day's state just before growth (after the water replay, phenology and stress), from
    the faithful trajectory, stacked on the day axis."""
    f, p = _season(seed)
    s0 = CeresMaizeState.initial(p, 1)
    traj = jax.jit(lambda pp, ff, s: run(_STATES, pp, ff, s))(p, f, s0)
    prev = jax.tree_util.tree_map(lambda x: x[:-1], traj)
    ft = jax.tree_util.tree_map(lambda x: x[1:], f)

    def pre(s, ff):
        s = ceres_water_replay(s, p, ff)
        return ceres_stress(ceres_phenology(s, p, ff), p, ff)

    return jax.jit(jax.vmap(pre))(prev, ft), ft, p


def test_carbo_takes_the_minimum_with_nstres() -> None:
    pre, ft, p = _pre_growth_states(62)

    def day(s, ff, ns):
        return ceres_growth_nstress_replay(_with_n(s, ns), p, ff)

    step = jax.jit(jax.vmap(day, in_axes=(0, 0, None)))
    faithful = jax.jit(jax.vmap(lambda s, ff: ceres_growth(s, p, ff)))(pre, ft)
    one = step(pre, ft, 1.0)
    assert _bytes_equal(dataclasses.replace(one, n_in=None), faithful) == []

    c_f = np.asarray(faithful.growth.carbo)[:, 0]
    istage = np.asarray(pre.phen.istage)[:, 0]
    grows = (istage >= 1) & (istage <= 5) & (c_f > 1e-3 * c_f.max())
    assert grows.sum() > 60
    small1, small2 = step(pre, ft, 1e-6), step(pre, ft, 2e-6)
    c1 = np.asarray(small1.growth.carbo)[:, 0][grows]
    c2 = np.asarray(small2.growth.carbo)[:, 0][grows]
    # NSTRES below PRFT and SWFAC: CARBO = PCARB NSTRES SLPF, linear in NSTRES
    np.testing.assert_allclose(c2, 2.0 * c1, rtol=1e-12 if jax.config.jax_enable_x64 else 1e-5)
    np.testing.assert_array_less(c1, c_f[grows])
    # NSTRES above 1 never binds (PRFT, SWFAC <= 1)
    big = step(pre, ft, 1.5)
    assert _bytes_equal(dataclasses.replace(big, n_in=None), faithful) == []
    # a moderate NSTRES never raises CARBO
    mid = np.asarray(step(pre, ft, 0.6).growth.carbo)[:, 0]
    assert np.all(mid[grows] <= c_f[grows]) and np.any(mid[grows] < c_f[grows])
    # the variant writes what the faithful process writes and nothing else
    changed = set(_bytes_equal(dataclasses.replace(step(pre, ft, 0.3), n_in=None), faithful))
    assert changed and all(n.startswith(("growth", "phen")) for n in changed)


# ------------------------------------------------------------------ bound in an assembly
PORTS = {"water_in": "iface.crop_water.maize", "root_out": "iface.root.maize"}
PORTS_N = {**PORTS, "n_in": "iface.crop_n.maize"}
NAMES = ("phenology", "stress", "growth", "roots", "publish")


def _assembly(nstress: bool) -> tuple[Model, Day]:
    water = bind(
        ceres_water_replay, own="crops.maize", ports=PORTS, forcing="crop", name="crops.maize.water_replay"
    )
    procs = {"crops.maize.water_replay": water}
    entries = ["crops.maize.water_replay"]
    if nstress:
        procs["n_supply.replay"] = bind(
            crop_n_replay,
            own="crops.n_replay",
            ports={"n_out": "iface.crop_n.maize"},
            forcing="n",
            name="n_supply.replay",
        )
        entries.insert(0, "n_supply.replay")
    crop = CROP_PROCESSES_NSTRESS_REPLAY if nstress else CROP_PROCESSES
    for n, proc in zip(NAMES, crop, strict=True):
        ports = PORTS_N if (nstress and n == "growth") else PORTS
        procs[f"crops.maize.{n}"] = bind(
            proc, own="crops.maize", ports=ports, forcing="crop", name=f"crops.maize.{n}"
        )
    day = Day(
        ref="none",
        phases=(
            Phase("management", tuple(entries)),
            Phase("plant", tuple(f"crops.maize.{n}" for n in NAMES)),
        ),
    )

    def outputs(s, pp, ff):
        o = plantgro_outputs(get_path(s, "crops.maize"), pp, ff["crop"])
        if nstress:
            o["nstres"] = get_path(s, "iface.crop_n.maize.nstres")
        return o

    return day.compile(procs, outputs=outputs), day


def _global0(p, nstress: bool) -> dict:
    s0 = CeresMaizeState.initial(p, 1)
    extra = {}
    if nstress:
        extra = {"crops.n_replay": CropNReplayState(), "iface.crop_n.maize": CropNIn.initial(1)}
    return compose({**Binding("crops.maize", tuple(PORTS.items())).entries(s0), **extra})


def test_bound_replay_producer_feeds_the_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    f, p = _season(63)
    n = int(f.yrdoy.shape[0])
    rng = np.random.default_rng(63)
    series = jnp.asarray(rng.uniform(0.3, 1.0, n), dtype=f.tmax.dtype)
    faithful_model, _ = _assembly(False)
    variant_model, day = _assembly(True)
    assert day.lag_report(variant_model).used == ()
    assert ("n_supply.replay", "crops.maize.growth", "iface.crop_n.maize") in variant_model.dataflow()
    go = jax.jit(lambda m_p, ff, g: run(variant_model, m_p, ff, g))
    ref = jax.jit(lambda m_p, ff, g: run(faithful_model, m_p, ff, g))(p, {"crop": f}, _global0(p, False))

    ones = go(p, {"crop": f, "n": CropNReplayForcing(nstres=jnp.ones_like(series))}, _global0(p, True))
    for k in ref:
        assert np.asarray(ones[k]).tobytes() == np.asarray(ref[k]).tobytes(), k

    stressed = go(p, {"crop": f, "n": CropNReplayForcing(nstres=series)}, _global0(p, True))
    np.testing.assert_array_equal(np.asarray(stressed["nstres"])[:, 0], np.asarray(series))
    assert float(stressed["cwad"][-1, 0]) < float(ref["cwad"][-1, 0])

    # eager calls under AGRI_JAX_CHECK=1: every bound process checks its writes on concrete values
    monkeypatch.setenv(CHECK_ENV, "1")
    forcing = {"crop": f, "n": CropNReplayForcing(nstres=series)}
    g = _global0(p, True)
    for t in range(40):
        ft = jax.tree_util.tree_map(lambda x, t=t: x[t], forcing)
        for proc in variant_model.processes:
            g = proc(g, p, ft)
    assert float(get_path(g, "iface.crop_n.maize.nstres")[0]) == pytest.approx(float(series[39]))


def test_replay_producer_alone() -> None:
    s = CropNReplayState.initial(2)
    out = crop_n_replay(s, None, CropNReplayForcing(nstres=jnp.asarray(0.7)))
    np.testing.assert_allclose(np.asarray(out.n_out.nstres), [0.7, 0.7])
    assert out.n_out.nstres.dtype == s.n_out.nstres.dtype
