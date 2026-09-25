"""Ports and bindings (plan 19 A2): typed port fields, ``bind()`` to global paths, records as state.

* One module (a toy crop with a ``water_in`` input port and a ``root_out`` output port) runs
  through a **replay** binding (its input record written from forcing) and a **coupled**
  binding (written by a soil module): the crop's outputs are bit-identical when the replayed
  record equals the coupled one.
* The CERES-Maize processes bound into a global state at ``crops.maize`` give bit-identical
  outputs to the unbound CERES model over a synthetic season (bind is erased by ``jit``).
* Bound ``reads``/``writes`` are global paths, so ``dataflow()``, ``stale_reads()`` and a
  :class:`~agrijax.core.day.Day` with a declared lag work on them.
* Under ``AGRI_JAX_CHECK=1`` a module that writes an undeclared port fails.
* Binding errors: unknown port, unbound port, namespace, overlapping targets, a port value left
  in the own subtree, a record of the wrong type.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from agrijax.core import (
    Day,
    Forcing,
    Lag,
    Model,
    Params,
    Phase,
    State,
    bind,
    compose,
    field,
    port,
    process,
    run,
    run_batch,
)
from agrijax.core.ports import Binding, BindingError, detach, port_names
from agrijax.core.process import CHECK_ENV, ProcessWriteError
from agrijax.core.state import get_path, set_path


# --------------------------------------------------------------------------- interface records
class CropWaterIn(State):
    """Interface record read by the crop (plan 19 A2/A3)."""

    eop: Array = field(dims=(), unit="mm d-1", description="potential transpiration")
    trwup: Array = field(dims=(), unit="cm d-1", description="potential root water uptake")


class RootRecord(State):
    """Interface record published by the crop."""

    rlv: Array = field(dims="n_layer", unit="cm cm-3")


# --------------------------------------------------------------------------------- the crop
class Growth(State):
    biomass: Array = field(dims=(), unit="g m-2")
    swfac: Array = field(dims=(), unit="-")


class CropState(State):
    growth: Growth
    water_in: CropWaterIn = port(description="EOP and TRWUP for the stress factors")
    root_out: RootRecord = port(description="root record published each day")


class CropParams(Params):
    rwuep1: Array = field(dims=(), unit="-")
    rue: Array = field(dims=(), unit="g MJ-1")


class Weather(Forcing):
    srad: Array = field(dims="T", unit="MJ m-2 d-1")
    pet: Array = field(dims="T", unit="mm d-1")


@process(reads=("water_in", "growth.swfac"), writes=("growth.swfac",), register=False, source="test")
def crop_stress(s: CropState, p: CropParams, f: Weather) -> CropState:
    """SWFAC from EOP and TRWUP (the DSSAT form ``min(1, 10 TRWUP / EOP)`` without RWUEP1 tricks).

    Source: fixture after DSSAT-CSM MZ_GROSUB.for (SWFAC).
    """
    ep1 = s.water_in.eop * 0.1
    ratio = jnp.where(ep1 > 0.0, s.water_in.trwup / jnp.where(ep1 > 0.0, ep1, 1.0), 1.0)
    return set_path(s, "growth.swfac", jnp.clip(ratio * p.rwuep1, 0.0, 1.0))


@process(reads=("growth",), writes=("growth.biomass",), register=False, source="test")
def crop_growth(s: CropState, p: CropParams, f: Weather) -> CropState:
    """Radiation-use growth limited by SWFAC.

    Source: fixture.
    """
    return set_path(s, "growth.biomass", s.growth.biomass + p.rue * f.srad * s.growth.swfac)


@process(reads=("growth.biomass",), writes=("root_out",), register=False, source="test")
def crop_publish(s: CropState, p: CropParams, f: Weather) -> CropState:
    """Publish a root record derived from the biomass (the crop's own numbers, as state).

    Source: fixture.
    """
    rlv = 1e-3 * s.growth.biomass * jnp.array([1.0, 0.6, 0.3])
    return set_path(s, "root_out", RootRecord(rlv=rlv))


CROP_PORTS = {"water_in": "iface.crop_water.maize", "root_out": "iface.root.maize"}


# ------------------------------------------------------------------------ the two producers
class SoilState(State):
    sw: Array = field(dims="n_layer", unit="cm3 cm-3")


class SupplyForcing(Forcing):
    eop: Array = field(dims="T", unit="mm d-1")
    trwup: Array = field(dims="T", unit="cm d-1")


class ReplayForcing(Forcing):
    weather: Weather
    supply: SupplyForcing


def _soil_supply(s, p, f):
    """Coupled producer: TRWUP from the soil water and yesterday's root record; EOP from PET.

    Source: fixture.
    """
    rlv = get_path(s, "iface.root.maize.rlv")
    sw = get_path(s, "soil_water.sw")
    trwup = jnp.sum(0.05 * rlv * jnp.maximum(sw - 0.1, 0.0)) + 0.01
    sw_new = jnp.maximum(sw - 0.002 * rlv, 0.05)
    s = set_path(s, "soil_water.sw", sw_new)
    return set_path(s, "iface.crop_water.maize", CropWaterIn(eop=f.pet * 10.0, trwup=trwup))


SOIL_SUPPLY = process(
    _soil_supply,
    reads=("soil_water.sw", "iface.root.maize.rlv"),
    writes=("soil_water.sw", "iface.crop_water.maize"),
    register=False,
    source="test",
)


def _replay_supply(s, p, f):
    """Replay producer: the interface record straight from forcing.

    Source: fixture.
    """
    return set_path(s, "iface.crop_water.maize", CropWaterIn(eop=f.supply.eop, trwup=f.supply.trwup))


REPLAY_SUPPLY = process(
    _replay_supply, reads=(), writes=("iface.crop_water.maize",), register=False, source="t"
)


def crop_entries(forcing: str | None = None) -> list:
    return [
        bind(fn, own="crops.maize", ports=CROP_PORTS, forcing=forcing, name=f"crops.maize.{n}")
        for n, fn in (("stress", crop_stress), ("growth", crop_growth), ("publish", crop_publish))
    ]


def crop0() -> CropState:
    return CropState(
        growth=Growth(biomass=jnp.asarray(1.0), swfac=jnp.asarray(1.0)),
        water_in=CropWaterIn(eop=jnp.asarray(0.0), trwup=jnp.asarray(0.0)),
        root_out=RootRecord(rlv=jnp.asarray([0.1, 0.05, 0.0])),
    )


def global0():
    b = Binding("crops.maize", tuple(CROP_PORTS.items()))
    return compose({"soil_water": SoilState(sw=jnp.asarray([0.3, 0.25, 0.2])), **b.entries(crop0())})


PARAMS = CropParams(rwuep1=jnp.asarray(1.5), rue=jnp.asarray(1.2))
N_DAYS = 40


def weather() -> Weather:
    t = np.arange(N_DAYS, dtype=float)
    return Weather(
        srad=jnp.asarray(15.0 + 5.0 * np.sin(t / 5.0)), pet=jnp.asarray(0.3 + 0.1 * np.cos(t / 3.0))
    )


CROP_OUT = ("crops.maize.growth.biomass", "crops.maize.growth.swfac", "iface.root.maize.rlv")
IFACE_OUT = ("iface.crop_water.maize.eop", "iface.crop_water.maize.trwup")

RZ_LIKE_DAY = Day(
    ref="none",
    phases=(
        Phase("physcl", ("soil_water.supply",)),
        Phase("plant", tuple(f"crops.maize.{n}" for n in ("stress", "growth", "publish"))),
    ),
    lags=(
        Lag("soil_water.supply", "iface.root.maize.rlv", evidence="fixture: root record of the previous day"),
    ),
)


def coupled_model() -> Model:
    procs = {"soil_water.supply": SOIL_SUPPLY, **{p.name: p for p in crop_entries()}}
    return RZ_LIKE_DAY.compile(procs, outputs=CROP_OUT + IFACE_OUT)


# ------------------------------------------------------------------------------------ tests
def test_port_fields_are_declared_and_detached() -> None:
    assert port_names(CropState) == ("water_in", "root_out")
    d = detach(crop0())
    assert d.water_in is None and d.root_out is None
    assert float(d.growth.biomass) == 1.0
    g = global0()
    assert get_path(g, "crops.maize").water_in is None
    assert isinstance(get_path(g, "iface.crop_water.maize"), CropWaterIn)


def test_bound_paths_are_global() -> None:
    stress, _, publish = crop_entries()
    assert stress.reads == ("iface.crop_water.maize", "crops.maize.growth.swfac")
    assert stress.writes == ("crops.maize.growth.swfac",)
    assert publish.writes == ("iface.root.maize",)
    assert stress.name == "crops.maize.stress" and stress.fn is not crop_stress.fn
    model = coupled_model()
    assert ("soil_water.supply", "crops.maize.stress", "iface.crop_water.maize") in model.dataflow()
    assert RZ_LIKE_DAY.lagged_reads(model) == [("soil_water.supply", "iface.root.maize.rlv")]


def test_replay_and_coupled_bindings_are_bit_identical() -> None:
    """The same crop code under two bindings: coupled (soil producer) and replay (from forcing)."""
    f = weather()
    coupled = run(coupled_model(), PARAMS, f, global0())
    replay_day = Day(
        ref="none",
        phases=(
            Phase("replay", ("iface.replay",)),
            Phase("plant", tuple(f"crops.maize.{n}" for n in ("stress", "growth", "publish"))),
        ),
    )
    procs = {"iface.replay": REPLAY_SUPPLY, **{p.name: p for p in crop_entries(forcing="weather")}}
    replay_model = replay_day.compile(procs, outputs=CROP_OUT)
    rf = ReplayForcing(
        weather=f,
        supply=SupplyForcing(
            eop=coupled["iface.crop_water.maize.eop"], trwup=coupled["iface.crop_water.maize.trwup"]
        ),
    )
    replay = run(replay_model, PARAMS, rf, global0())
    for k in CROP_OUT:
        assert np.asarray(coupled[k]).tobytes() == np.asarray(replay[k]).tobytes(), k
    # the coupling is not trivial: stress varies over the run
    swfac = np.asarray(coupled["crops.maize.growth.swfac"])
    assert swfac.min() < 0.99 * swfac.max()


def test_bound_module_equals_the_unbound_module_bitwise() -> None:
    """A module state run on its own (ports inside it) equals the bound run."""
    f = weather()
    supply = SupplyForcing(
        eop=jnp.asarray(np.linspace(2.0, 5.0, N_DAYS)), trwup=jnp.asarray(np.linspace(0.5, 0.1, N_DAYS))
    )

    def feed(s, p, fr):
        """Source: fixture."""
        return set_path(s, "water_in", CropWaterIn(eop=fr.supply.eop, trwup=fr.supply.trwup))

    feed_p = process(feed, reads=(), writes=("water_in",), register=False, source="t")
    alone = Model(
        CropState,
        [
            feed_p,
            process(
                lambda s, p, fr: crop_stress(s, p, fr.weather),
                reads=crop_stress.reads,
                writes=crop_stress.writes,
                register=False,
                name="stress",
            ),
            process(
                lambda s, p, fr: crop_growth(s, p, fr.weather),
                reads=crop_growth.reads,
                writes=crop_growth.writes,
                register=False,
                name="growth",
            ),
            process(
                lambda s, p, fr: crop_publish(s, p, fr.weather),
                reads=crop_publish.reads,
                writes=crop_publish.writes,
                register=False,
                name="publish",
            ),
        ],
        outputs=lambda s, p, fr: {"b": s.growth.biomass, "rlv": s.root_out.rlv},
    )
    rf = ReplayForcing(weather=f, supply=supply)
    a = run(alone, PARAMS, rf, crop0())
    bound = Model(
        None,
        [REPLAY_SUPPLY, *crop_entries(forcing="weather")],
        outputs=lambda s, p, fr: {
            "b": get_path(s, "crops.maize.growth.biomass"),
            "rlv": get_path(s, "iface.root.maize.rlv"),
        },
    )
    b = run(bound, PARAMS, rf, global0())
    for k in ("b", "rlv"):
        assert np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes(), k


def test_bound_ceres_is_bit_identical_to_the_ceres_model() -> None:
    """The unchanged CERES-Maize processes, bound at ``crops.maize``, over a synthetic season."""
    from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model
    from agrijax.processes.crop.ceres_maize.model import plantgro_outputs

    from .test_ceres_growth import season_forcing
    from .test_ceres_phenology import make_params

    f, w = season_forcing(3, stress=True, waterlog=True)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s0 = CeresMaizeState.initial(p, 1)
    plain = ceres_maize_model()
    ref = run(plain, p, f, s0)
    # the plain model keeps its ports (water_in replayed from forcing, root_out published) in its own
    # state; bound, they are wired to interface paths outside the subtree
    ports = {"water_in": "iface.crop_water.maize", "root_out": "iface.root.maize"}
    bound = Model(
        None,
        [bind(proc, own="crops.maize", ports=ports) for proc in plain.processes],
        outputs=lambda s, pp, ff: plantgro_outputs(get_path(s, "crops.maize"), pp, ff),
    )
    out = run(bound, p, f, compose(Binding("crops.maize", tuple(ports.items())).entries(s0)))
    assert set(out) == set(ref)
    for k in ref:
        assert np.asarray(out[k]).tobytes() == np.asarray(ref[k]).tobytes(), k
    assert float(np.max(np.asarray(ref["cwad"]))) > 0.0  # the season really grew


def test_bound_run_batch_and_grad() -> None:
    model = coupled_model()
    f = weather()
    pb = jax.tree_util.tree_map(lambda x: jnp.stack([x, 1.1 * x]), PARAMS)
    out = run_batch(model, pb, f, global0())
    assert out["crops.maize.growth.biomass"].shape == (2, N_DAYS)

    def loss(rue):
        o = run(model, PARAMS.replace(rue=rue), f, global0())
        return o["crops.maize.growth.biomass"][-1]

    g = jax.grad(loss)(jnp.asarray(1.2))
    assert np.isfinite(float(g)) and float(g) > 0.0


def test_undeclared_port_write_fails_under_check(monkeypatch: pytest.MonkeyPatch) -> None:
    @process(reads=("water_in",), writes=("growth.swfac",), register=False, source="test")
    def leaky(s: CropState, p: CropParams, f: Weather) -> CropState:
        """Writes its input port without declaring it.

        Source: fixture.
        """
        s = set_path(s, "water_in.eop", s.water_in.eop + 1.0)
        return set_path(s, "growth.swfac", jnp.asarray(0.5))

    bound = bind(leaky, own="crops.maize", ports=CROP_PORTS)
    g = global0()
    fday = jax.tree_util.tree_map(lambda x: x[0], weather())
    monkeypatch.setenv(CHECK_ENV, "1")
    with pytest.raises(ProcessWriteError, match=r"water_in\.eop"):
        bound(g, PARAMS, fday)
    monkeypatch.setenv(CHECK_ENV, "0")
    out = bound(g, PARAMS, fday)  # unchecked: the undeclared write is not scattered back
    assert float(get_path(out, "iface.crop_water.maize.eop")) == float(
        get_path(g, "iface.crop_water.maize.eop")
    )
    monkeypatch.setenv(CHECK_ENV, "1")
    ok = bind(crop_stress, own="crops.maize", ports=CROP_PORTS)(g, PARAMS, fday)
    for x, y in zip(
        jax.tree_util.tree_leaves(get_path(ok, "iface.crop_water.maize")),
        jax.tree_util.tree_leaves(get_path(g, "iface.crop_water.maize")),
        strict=True,
    ):
        assert x is y  # the input port comes back untouched


def test_binding_errors() -> None:
    g = global0()
    fday = jax.tree_util.tree_map(lambda x: x[0], weather())
    with pytest.raises(BindingError, match="not a namespace"):
        bind(crop_stress, own="crop", ports=CROP_PORTS)
    with pytest.raises(BindingError, match="not a namespace"):
        bind(crop_stress, own="crops.maize", ports={"water_in": "records.water"})
    with pytest.raises(BindingError, match="overlap"):
        bind(crop_stress, own="crops.maize", ports={"water_in": "crops.maize.w"})
    with pytest.raises(BindingError, match="overlap"):
        Binding("crops.maize", (("water_in", "iface.a"), ("root_out", "iface.a.b")))
    with pytest.raises(BindingError, match="no port field"):
        bind(crop_stress, own="crops.maize", ports={**CROP_PORTS, "soil_in": "iface.soil.maize"})(
            g, PARAMS, fday
        )
    with pytest.raises(BindingError, match="not bound"):
        bind(crop_stress, own="crops.maize", ports={"root_out": "iface.root.maize"})(g, PARAMS, fday)
    left = compose(
        {
            "soil_water": SoilState(sw=jnp.ones(3)),
            "crops.maize": crop0(),
            "iface.crop_water.maize": crop0().water_in,
            "iface.root.maize": crop0().root_out,
        }
    )
    with pytest.raises(BindingError, match="holds a value in the global state"):
        bind(crop_stress, own="crops.maize", ports=CROP_PORTS)(left, PARAMS, fday)
    wrong = set_path(g, "iface.crop_water.maize", RootRecord(rlv=jnp.ones(3)))
    with pytest.raises(BindingError, match="expects CropWaterIn"):
        bind(crop_stress, own="crops.maize", ports=CROP_PORTS)(wrong, PARAMS, fday)
    with pytest.raises(BindingError, match="overlap"):
        compose({"crops.maize": 1, "crops.maize.x": 2})
    with pytest.raises(BindingError, match="not a namespace"):
        compose({"other": 1})
