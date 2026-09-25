"""CERES-Maize ports (plan 19 A2-A4): water in, stress computed by the crop, root record out.

* ``CeresForcing`` carries no stress factors; the crop computes ``SWFAC``/``TURFAC`` from the
  ``EOP`` and ``TRWUP`` of its ``water_in`` port with ``RWUEP1`` (DSSAT ``MZ_GROSUB``), on every
  day it runs the stress block;
* ``ceres_publish`` puts today's ``RLV``, ``RTDEP``, ``XHLAI = LAI`` and the species ``RWUMX``,
  ``PORMIN`` into ``root_out``;
* one CERES code path, three bindings: on its own (ports inside the state, the replay writes
  ``water_in``), bound with its ports on ``iface`` paths and the replay bound too (bit-identical),
  and **coupled** in a DSSAT-order day where a soil fixture writes ``sw``/``eop``, the ``ROOTWU``
  producer writes ``trwup`` from yesterday's root record (a declared lag), and the crop follows.
  Replaying the coupled run's interface records into the crop alone reproduces its crop outputs
  bit for bit.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import Day, Lag, Model, Phase, bind, compose, process, run
from agrijax.core.coefficients import coefficient_table
from agrijax.core.ports import Binding
from agrijax.core.state import get_path, set_path
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    CeresForcing,
    CeresMaizeState,
    ceres_maize_model,
    ceres_water_replay,
    plantgro_outputs,
    water_stress_factors,
)
from agrijax.processes.crop.ceres_maize.growth import WATER_STRESS_COEFFICIENTS, WaterStressCoefficients
from agrijax.processes.soil_water.uptake import RootwuParams, RootwuState, rootwu_supply

from .test_ceres_growth import season_forcing
from .test_ceres_phenology import make_params

PORTS = {"water_in": "iface.crop_water.maize", "root_out": "iface.root.maize"}
_STATE_MODEL = ceres_maize_model(outputs=lambda s, p, f: s)


def test_forcing_has_no_stress_fields() -> None:
    names = {f.name for f in dataclasses.fields(CeresForcing)}
    assert {"eop", "trwup", "sw"} <= names
    assert not {"swfac", "turfac"} & names


def test_stress_factors_are_computed_from_eop_and_trwup() -> None:
    f, w = season_forcing(51, stress=True, n=200)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s = run(_STATE_MODEL, p, f, CeresMaizeState.initial(p, 1))
    st = np.asarray(s.phen.istage[:, 0])
    prev = np.concatenate([[7], st[:-1]])  # the stage entering each day
    mdate = np.asarray(s.phen.mdate[:, 0])
    run_days = (prev >= 1) & (prev <= 6) & (mdate != np.asarray(f.yrdoy))
    sw_want, tu_want = water_stress_factors(f.eop, f.trwup, p.species.rwuep1)
    got_sw = np.asarray(s.stress.swfac[:, 0])
    got_tu = np.asarray(s.stress.turfac[:, 0])
    np.testing.assert_array_equal(got_sw[run_days], np.asarray(sw_want)[run_days])
    np.testing.assert_array_equal(got_tu[run_days], np.asarray(tu_want)[run_days])
    assert run_days.sum() > 50 and got_sw[run_days].min() < 0.5  # stressed days were visited
    # the port holds what the replay wrote
    np.testing.assert_array_equal(np.asarray(s.water_in.trwup[:, 0]), np.asarray(f.trwup))
    np.testing.assert_array_equal(np.asarray(s.water_in.sw), np.asarray(f.sw))


def test_the_root_record_is_published_each_day() -> None:
    f, w = season_forcing(52, n=150)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s = run(_STATE_MODEL, p, f, CeresMaizeState.initial(p, 2))
    r = s.root_out
    assert np.asarray(r.rlv).tobytes() == np.asarray(s.roots.rlv).tobytes()
    assert np.asarray(r.rtdep).tobytes() == np.asarray(s.roots.rtdep).tobytes()
    assert np.asarray(r.xhlai).tobytes() == np.asarray(s.growth.lai).tobytes()
    np.testing.assert_array_equal(np.asarray(r.rwumx), float(p.species.rwumx))
    np.testing.assert_array_equal(np.asarray(r.pormin), float(p.species.pormin))
    assert float(np.max(np.asarray(r.rlv))) > 0.0 and float(np.max(np.asarray(r.xhlai))) > 0.0


def _bound(forcing: str | None = None, params: str | None = None) -> list:
    return [
        bind(proc, own="crops.maize", ports=PORTS, forcing=forcing, params=params, name=f"crops.maize.{n}")
        for n, proc in zip(("phenology", "stress", "growth", "roots", "publish"), CROP_PROCESSES, strict=True)
    ]


def _global0(s0: CeresMaizeState, extra: dict | None = None) -> dict:
    return compose({**Binding("crops.maize", tuple(PORTS.items())).entries(s0), **(extra or {})})


def _crop_out(s, p, f) -> dict:
    return plantgro_outputs(get_path(s, "crops.maize"), p, f)


def test_replay_bound_crop_equals_the_crop_alone_bitwise() -> None:
    f, w = season_forcing(53, stress=True, waterlog=True, n=200)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    s0 = CeresMaizeState.initial(p, 1)
    alone = run(ceres_maize_model(), p, f, s0)
    replay = bind(ceres_water_replay, own="crops.maize", ports=PORTS, name="iface.replay")
    assert replay.writes == ("iface.crop_water.maize",)
    model = Model(None, [replay, *_bound()], outputs=_crop_out)
    out = run(model, p, f, _global0(s0))
    assert set(out) == set(alone)
    for k in alone:
        assert np.asarray(out[k]).tobytes() == np.asarray(alone[k]).tobytes(), k
    assert float(np.max(np.asarray(alone["cwad"]))) > 0.0


# ------------------------------------------------------------------ coupled, DSSAT order
def _soil_supply(s, p, f):
    """Soil fixture: today's soil water and potential transpiration into the crop water record.

    Source: fixture (the SPAM / WATBAL outputs of a DSSAT day).
    """
    rec = get_path(s, "iface.crop_water.maize")
    rec = rec.replace(sw=jnp.asarray(f.sw), eop=jnp.asarray(f.eop) * jnp.ones_like(rec.eop))
    return set_path(s, "iface.crop_water.maize", rec)


SOIL_SUPPLY = process(
    _soil_supply,
    reads=("iface.crop_water.maize",),
    writes=("iface.crop_water.maize.sw", "iface.crop_water.maize.eop"),
    register=False,
    source="fixture",
)

DSSAT_LIKE_DAY = Day(
    ref="none",
    phases=(
        Phase("spam", ("soil_water.supply", "soil_water.rootwu")),
        Phase(
            "plant", tuple(f"crops.maize.{n}" for n in ("phenology", "stress", "growth", "roots", "publish"))
        ),
    ),
    lags=(
        Lag(
            "soil_water.rootwu",
            "iface.root.maize",
            evidence="DSSAT-CSM LAND.for: PLANT outputs RLV, RWUMX, PORMIN, XHLAI that SPAM reads on its next call",
        ),
    ),
)
IFACE = ("iface.crop_water.maize.sw", "iface.crop_water.maize.eop", "iface.crop_water.maize.trwup")


def _coupled(p, f):
    n_layer = int(p.soil.dlayr.shape[0])
    rootwu = bind(rootwu_supply, own="soil_water.rootwu", params="rootwu", name="soil_water.rootwu",
                  ports={"root": "iface.root.maize", "water": "iface.crop_water.maize"})  # fmt: skip
    procs = {
        "soil_water.supply": SOIL_SUPPLY,
        "soil_water.rootwu": rootwu,
        **{q.name: q for q in _bound(params="crop")},
    }

    def outputs(s, pp, ff):
        o = _crop_out(s, pp["crop"], ff)
        o.update({k: get_path(s, k) for k in IFACE})
        return o

    model = DSSAT_LIKE_DAY.compile(procs, outputs=outputs)
    params = {"crop": p, "rootwu": RootwuParams(dlayr=p.soil.dlayr, ll=p.soil.ll, sat=p.soil.sat)}
    g0 = _global0(CeresMaizeState.initial(p, 1), {"soil_water.rootwu": RootwuState.initial(1, n_layer)})
    return model, params, g0


def test_coupled_day_with_the_rootwu_producer() -> None:
    f, w = season_forcing(54, n=180)
    # a drying soil, so TRWUP from ROOTWU falls below the demand and the crop is stressed
    frac = np.clip(np.arange(180) / 120.0, 0.0, 1.0)[:, None]
    ll = np.asarray(make_params().soil.ll)
    sw = np.asarray(f.sw) * (1.0 - frac) + (ll + 0.005) * frac
    f = f.replace(sw=jnp.asarray(sw))
    p = make_params(yrplt=int(w["yrdoy"][2]))
    model, params, g0 = _coupled(p, f)
    assert DSSAT_LIKE_DAY.lagged_reads(model) == [("soil_water.rootwu", "iface.root.maize")]
    assert ("soil_water.rootwu", "crops.maize.stress", "iface.crop_water.maize") in model.dataflow()
    out = jax.jit(lambda pp, ff, gg: run(model, pp, ff, gg))(params, f, g0)
    trwup = np.asarray(out["iface.crop_water.maize.trwup"])[:, 0]
    assert trwup.max() > 0.0 and trwup[-1] < 0.5 * trwup.max()
    assert float(np.min(1.0 - np.asarray(out["wspd"]))) < 0.9  # the producer's TRWUP stressed the crop

    # the same crop code driven by a replay of the coupled run's records: bit for bit
    rf = f.replace(
        sw=out["iface.crop_water.maize.sw"],
        eop=out["iface.crop_water.maize.eop"][:, 0],
        trwup=out["iface.crop_water.maize.trwup"][:, 0],
    )
    alone = jax.jit(lambda pp, ff, s: run(ceres_maize_model(), pp, ff, s))(
        p, rf, CeresMaizeState.initial(p, 1)
    )
    for k in alone:
        assert np.asarray(out[k]).tobytes() == np.asarray(alone[k]).tobytes(), k


def test_coupled_gradient_reaches_the_published_rwumx() -> None:
    """d(season biomass)/d(RWUMX) flows through the root record into ROOTWU and the stress: on a
    moist soil the uptake per unit root length is capped at RWUMX, so while the root system is
    small TRWUP = RWUMX sum(DLAYR RLV) limits transpiration."""
    f, w = season_forcing(55, n=150)
    p = make_params(yrplt=int(w["yrdoy"][2]))
    model, params, g0 = _coupled(p, f)

    def loss(rwumx):
        pp = {**params, "crop": p.replace(species=p.species.replace(rwumx=rwumx))}
        return run(model, pp, f, g0)["cwad"][-1, 0]

    g = jax.grad(loss)(jnp.asarray(0.03))
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_water_stress_numbers_are_declared_coefficients() -> None:
    """The TURFAC storage precision and the last stress-block stage are coefficients with their
    DSSAT-CSM statement (checked against the source in tests/integration/test_rootwu_dssat.py);
    ``EP1 = EOP * 0.1`` is the mm -> cm adapter."""
    rows = {r["path"]: r for r in coefficient_table(WaterStressCoefficients)}
    assert set(rows) == {"turfac_scale", "grosub_last_stage"}
    for r in rows.values():
        assert r["ref_version"] == "dssat-4.8.6.0" and r["line"] and r["statement"] and not r["calibratable"]
    assert rows["grosub_last_stage"]["static"] and WATER_STRESS_COEFFICIENTS.grosub_last_stage == 6
    eop = jnp.asarray([4.0, 4.0, 0.0])
    trwup = jnp.asarray([0.2123456, 0.5, 0.1])
    sw, tu = water_stress_factors(eop, trwup, 1.5)
    np.testing.assert_allclose(np.asarray(sw), [0.2123456 / 0.4, 1.0, 1.0], rtol=1e-6)
    assert float(tu[0]) == pytest.approx(np.trunc(0.2123456 / 0.4 / 1.5 * 1000.0) / 1000.0, rel=1e-6)
    coarse = WaterStressCoefficients(turfac_scale=10.0)
    _, tu10 = water_stress_factors(eop, trwup, 1.5, coarse)
    assert float(tu10[0]) == pytest.approx(np.trunc(0.2123456 / 0.4 / 1.5 * 10.0) / 10.0, rel=1e-6)
