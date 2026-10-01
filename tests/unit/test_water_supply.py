"""The water-supply slot (``processes/water_supply``): the moved ROOTWU, the RZWQM2 uptake limit
``WUF`` and the layer -> node publish with its ``SW == LL`` quirk. Data-free.

* the old path ``processes/soil_water/uptake.py`` is gone (no alias across slots), the producer is
  registered from ``processes/water_supply/rootwu.py``;
* :func:`rzwqm_uptake_limit` against a plain-Python transcription of the reference's rule
  (per crop and node ``if``/``else``, Python floats): ``PET > 0`` and ``TRWUP != 0`` gate, ``PET``
  above and below ``TRWUP``, nodes above, below and exactly at the wilting point; the sum over the
  crops of a slot; finite gradients and finite differences with respect to the input records;
* :func:`rzwqm_publish_uptake` against a plain-Python transcription of the crop driver's layer loop
  and a loop-based thickness-weighted mean: layers above, below and at ``LL`` (the quirk keeps the
  node value of the day's sink), the ``PET <= 0`` day, the REAL*4 comparison (a float64 ``SW`` one
  float32 rounding above ``LL`` counts as ``SW == LL``), conservation of the layer amounts, finite
  gradients with respect to ``rwu``;
* both processes bound on the contract's ports in the skeleton day: ``Day.check`` sees the two
  lags of the uptake limit (P3 and P1 ``trwup``) and nothing else changes.

The day-by-day comparison with RZWQM2 4.6 is ``tests/integration/test_water_supply_rzwqm.py``.
"""

from __future__ import annotations

import importlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import bind
from agrijax.core.grids import SoilGrid
from agrijax.core.process import CHECK_ENV, lookup
from agrijax.iface.crop import CropWaterIn
from agrijax.iface.soil import NodeUptake, SinkInputs
from agrijax.iface.surface import PETFluxes
from agrijax.processes import water_supply as ws
from agrijax.processes.water_supply import (
    PublishUptakeParams,
    RootwuState,
    UptakeLimitParams,
    UptakeLimitState,
    rzwqm_publish_uptake,
    rzwqm_uptake_limit,
)

X64 = bool(jax.config.read("jax_enable_x64"))
DT = jnp.float64 if X64 else jnp.float32
REL = 1e-12 if X64 else 2e-6
ABS = 1e-15 if X64 else 1e-9


# ------------------------------------------------------------------ the moved module
def test_old_path_is_gone_and_the_producer_lives_in_water_supply() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agrijax.processes.soil_water.uptake")
    new = importlib.import_module("agrijax.processes.water_supply.rootwu")
    assert len(new.__all__) == 13
    assert ws.rootwu_supply is new.rootwu_supply
    p = lookup("water_supply/rootwu@dssat-4.8.6.0:faithful")
    assert p is new.rootwu_supply and p.fn.__module__ == "agrijax.processes.water_supply.rootwu"


# ------------------------------------------------------------------ the uptake limit (WUF)
def ref_limit(q, trwup, pet, theta, theta_wp):
    """The reference's rule in plain Python: per crop the ``WUF`` gate and value, per node the
    wilting-point branch; the slot's sink is the sum over its crops."""
    n_crop, n_node = len(q), len(theta)
    out = [0.0] * n_node
    for c in range(n_crop):
        on = pet > 0.0 and trwup[c] != 0.0
        wuf = (pet / trwup[c] if pet <= trwup[c] else 1.0) if on else 1.0
        for i in range(n_node):
            v = q[c][i]
            if on:
                if theta[i] > theta_wp[i]:
                    v = v * wuf
                elif theta[i] < theta_wp[i]:
                    v = 0.0
            out[i] += v
    return out


def _limit_state(q, trwup, pet, theta, dtype=DT) -> UptakeLimitState:
    z = jnp.zeros((), dtype)
    n_node = len(theta)
    return UptakeLimitState(
        theta=jnp.asarray(theta, dtype),
        root_uptake=NodeUptake(uptake=jnp.asarray(q, dtype)),
        water=CropWaterIn(
            sw=jnp.zeros(3, dtype), eop=jnp.zeros(len(trwup), dtype), trwup=jnp.asarray(trwup, dtype)
        ),
        pet=PETFluxes(
            transpiration=jnp.asarray(pet, dtype),
            soil_evaporation=z,
            residue_evaporation=z,
            reference_short=z,
            reference_tall=z,
            eo_priestley_taylor=z,
        ),
        sink_in=SinkInputs.zeros(n_node, dtype),
    )


_LIMIT = jax.jit(rzwqm_uptake_limit)


@pytest.mark.parametrize("seed", range(40))
def test_uptake_limit_matches_the_plain_transcription(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_crop, n_node = int(rng.integers(1, 4)), 11
    theta_wp = rng.uniform(0.05, 0.15, n_node)
    theta = theta_wp + rng.uniform(-0.05, 0.2, n_node)
    theta_wp[0] = theta[0]  # a node exactly at the wilting point keeps its uptake
    q = rng.uniform(0.0, 0.05, (n_crop, n_node))
    kind = seed % 4
    pet = [0.3, 0.0, -0.1, 0.3][kind]
    trwup = rng.uniform(0.05, 0.8, n_crop)
    if kind == 3:
        trwup[0] = 0.0  # no potential uptake yesterday: that crop is not limited
    st = _limit_state(q, trwup, pet, theta)
    got = np.asarray(_LIMIT(st, UptakeLimitParams(theta_wp=jnp.asarray(theta_wp, DT)), None).sink_in.uptake)
    want = ref_limit(q.tolist(), trwup.tolist(), pet, theta.tolist(), theta_wp.tolist())
    np.testing.assert_allclose(got, want, rtol=REL, atol=ABS)


def test_uptake_limit_branches() -> None:
    theta_wp = np.full(4, 0.1)
    theta = np.asarray([0.2, 0.05, 0.1, 0.3])  # above, below, at, above
    q = np.asarray([[1.0, 2.0, 3.0, 4.0]]) * 1e-2
    p = UptakeLimitParams(theta_wp=jnp.asarray(theta_wp, DT))
    # PET = TRWUP / 4: WUF = 0.25 on nodes above, 0 below, kept at theta_wp
    out = np.asarray(_LIMIT(_limit_state(q, [0.4], 0.1, theta), p, None).sink_in.uptake)
    np.testing.assert_allclose(out, [0.25e-2, 0.0, 3e-2, 1e-2], rtol=REL)
    # PET > TRWUP: WUF = 1, but the nodes below the wilting point are still zeroed
    out = np.asarray(_LIMIT(_limit_state(q, [0.05], 0.1, theta), p, None).sink_in.uptake)
    np.testing.assert_allclose(out, [1e-2, 0.0, 3e-2, 4e-2], rtol=REL)
    # gate off (PET = 0, or TRWUP = 0): the uptake passes unchanged, below the wilting point too
    for pet, tr in ((0.0, 0.4), (0.1, 0.0)):
        out = np.asarray(_LIMIT(_limit_state(q, [tr], pet, theta), p, None).sink_in.uptake)
        np.testing.assert_array_equal(out, np.asarray(q[0], out.dtype))


def test_uptake_limit_one_crop_is_the_crop_bit_for_bit_and_writes_only_the_sink() -> None:
    rng = np.random.default_rng(3)
    theta_wp = rng.uniform(0.05, 0.1, 7)
    theta = theta_wp + rng.uniform(0.01, 0.1, 7)
    q = rng.uniform(0.0, 0.05, (1, 7))
    st = _limit_state(q, [0.5], 0.2, theta)
    p = UptakeLimitParams(theta_wp=jnp.asarray(theta_wp, DT))
    new = _LIMIT(st, p, None)
    w = jnp.asarray(0.2, DT) / jnp.asarray(0.5, DT)
    np.testing.assert_array_equal(np.asarray(new.sink_in.uptake), np.asarray(jnp.asarray(q[0], DT) * w))
    for other in ("tile", "lateral", "subirrigation", "macropore_to_drain"):
        assert float(jnp.max(jnp.abs(getattr(new.sink_in, other)))) == 0.0
    assert new.sink_in.uptake.dtype == DT


def test_uptake_limit_gradients_are_finite_and_match_finite_differences() -> None:
    rng = np.random.default_rng(5)
    n = 6
    theta_wp = rng.uniform(0.05, 0.1, n)
    theta = theta_wp + rng.uniform(-0.03, 0.1, n)
    p = UptakeLimitParams(theta_wp=jnp.asarray(theta_wp, DT))

    def loss(q, tr, pet):
        st = _limit_state(q, tr, pet, theta).replace(
            root_uptake=NodeUptake(uptake=q),
            water=CropWaterIn(sw=jnp.zeros(3, DT), eop=jnp.zeros(tr.shape, DT), trwup=tr),
        )
        st = st.replace(pet=st.pet.replace(transpiration=pet))
        return jnp.sum(rzwqm_uptake_limit(st, p, None).sink_in.uptake ** 2)

    q = jnp.asarray(rng.uniform(0.01, 0.05, (2, n)), DT)
    for tr, pet in (([0.5, 0.1], 0.2), ([0.0, 0.1], 0.2), ([0.5, 0.1], 0.0)):  # incl. the TRWUP = 0 edge
        args = (q, jnp.asarray(tr, DT), jnp.asarray(pet, DT))
        g = jax.grad(loss, argnums=(0, 1, 2))(*args)
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in g), (tr, pet)
        if X64 and pet > 0.0:
            for k in range(3):
                d = jnp.asarray(np.random.default_rng(k).normal(size=np.shape(args[k])), DT)
                h = 1e-6
                a_p = list(args)
                a_m = list(args)
                a_p[k] = args[k] + h * d
                a_m[k] = args[k] - h * d
                fd = (loss(*a_p) - loss(*a_m)) / (2 * h)
                np.testing.assert_allclose(float(jnp.sum(g[k] * d)), float(fd), rtol=1e-6, atol=1e-12)


# ------------------------------------------------------------------ the layer -> node publish
NODES = SoilGrid.from_thickness("rzwqm2_nodes", (2.0, 3.0, 5.0, 5.0, 10.0, 10.0, 15.0, 20.0, 30.0))
LAYERS = SoilGrid.from_thickness("rzwqm2_lyrset", (5.0, 10.0, 15.0, 30.0, 40.0))


def ref_publish(rwu, sw, ll, eop, found, dlayr, tl, node_bot, layer_bot):
    """The crop driver's layer loop and ``REALMATCH`` in plain Python: node amounts ``[n_crop][n_node]``."""
    n_node, n_layer = len(tl), len(dlayr)
    out = []
    for c in range(len(rwu)):
        rate = [0.0] * n_layer
        if eop[c] > 0.0:
            for L in range(n_layer):
                s32, l32 = np.float32(sw[L]), np.float32(ll[L])
                if s32 > l32:
                    rate[L] = rwu[c][L] / dlayr[L]
                elif s32 < l32:
                    rate[L] = 0.0
                else:
                    rate[L] = found[L] / tl[L]  # element L of the node array as found
        row = []
        for i in range(n_node):
            top_i = node_bot[i - 1] if i else 0.0
            num = den = 0.0
            for L in range(n_layer):
                top_l = layer_bot[L - 1] if L else 0.0
                o = max(0.0, min(node_bot[i], layer_bot[L]) - max(top_i, top_l))
                num += o * rate[L]
                den += o
            row.append((num / den if den > 0.0 else 0.0) * tl[i])
        out.append(row)
    return out


def _publish_state(rwu, sw, eop, found, dtype=DT) -> RootwuState:
    n_crop, n_layer = np.shape(rwu)
    return RootwuState(
        tss=jnp.zeros((n_crop, n_layer), dtype),
        rwu=jnp.asarray(rwu, dtype),
        water=CropWaterIn(
            sw=jnp.asarray(sw, dtype), eop=jnp.asarray(eop, dtype), trwup=jnp.zeros(n_crop, dtype)
        ),
        root_uptake=NodeUptake.zeros(n_crop, NODES.n, dtype),
        sink_in=SinkInputs.zeros(NODES.n, dtype).replace(uptake=jnp.asarray(found, dtype)),
    )


_PUBLISH = jax.jit(rzwqm_publish_uptake)


@pytest.mark.parametrize("seed", range(30))
def test_publish_matches_the_plain_transcription(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    n_crop = int(rng.integers(1, 3))
    ll = np.round(rng.uniform(0.05, 0.15, LAYERS.n), 3)
    sw = ll + rng.uniform(-0.03, 0.15, LAYERS.n)
    ll[seed % LAYERS.n] = sw[seed % LAYERS.n]  # one layer at SW == LL
    rwu = rng.uniform(0.0, 0.1, (n_crop, LAYERS.n))
    eop = rng.uniform(1.0, 6.0, n_crop)
    if seed % 3 == 0:
        eop[0] = 0.0  # PET <= 0: no uptake from that crop
    found = rng.uniform(0.0, 0.05, NODES.n)
    st = _publish_state(rwu, sw, eop, found)
    p = PublishUptakeParams(nodes=NODES, layers=LAYERS, ll=jnp.asarray(ll, DT))
    got = np.asarray(_PUBLISH(st, p, None).root_uptake.uptake)
    want = ref_publish(
        rwu.tolist(), sw.tolist(), ll.tolist(), eop.tolist(), found.tolist(),
        LAYERS.thickness.tolist(), NODES.thickness.tolist(), list(NODES.bottom), list(LAYERS.bottom),
    )  # fmt: skip
    np.testing.assert_allclose(got, want, rtol=REL, atol=ABS)


def test_publish_quirk_uses_node_l_of_the_days_sink_and_the_real4_comparison() -> None:
    ll = np.asarray([0.1, 0.1, 0.1, 0.1, 0.1])
    rwu = np.asarray([[0.05, 0.04, 0.03, 0.02, 0.01]])
    found = np.arange(1, NODES.n + 1) * 1e-3
    p = PublishUptakeParams(nodes=NODES, layers=LAYERS, ll=jnp.asarray(ll, DT))
    base = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2])
    at_ll = base.copy()
    at_ll[2] = 0.1  # layer 3 (15-30 cm) at LL: its rate is node 3's rate of the day's sink
    out = np.asarray(_PUBLISH(_publish_state(rwu, at_ll, [4.0], found), p, None).root_uptake.uptake)
    ref = np.asarray(_PUBLISH(_publish_state(rwu, base, [4.0], found), p, None).root_uptake.uptake)
    # node 5 (15-25 cm) lies in layer 3 only: rate found[2] / TL[2] instead of rwu / dlayr
    tl = NODES.thickness
    np.testing.assert_allclose(out[0, 4], found[2] / tl[2] * tl[4], rtol=REL)
    np.testing.assert_allclose(ref[0, 4], rwu[0, 2] / 15.0 * tl[4], rtol=REL)
    if X64:
        # one float64 step above LL, the same REAL*4 value: the reference's SW == LL
        near = base.copy()
        near[2] = np.nextafter(0.1, 1.0)
        assert near[2] > 0.1 and np.float32(near[2]) == np.float32(0.1)
        o2 = np.asarray(_PUBLISH(_publish_state(rwu, near, [4.0], found), p, None).root_uptake.uptake)
        np.testing.assert_array_equal(o2, out)
    below = base.copy()
    below[2] = 0.09
    o3 = np.asarray(_PUBLISH(_publish_state(rwu, below, [4.0], found), p, None).root_uptake.uptake)
    np.testing.assert_array_equal(o3[0, 4], 0.0)


def test_publish_conserves_the_layer_amounts_and_zero_pet_gives_no_uptake() -> None:
    rng = np.random.default_rng(9)
    ll = np.full(LAYERS.n, 0.1)
    sw = np.full(LAYERS.n, 0.25)
    rwu = rng.uniform(0.0, 0.1, (2, LAYERS.n))
    p = PublishUptakeParams(nodes=NODES, layers=LAYERS, ll=jnp.asarray(ll, DT))
    out = np.asarray(
        _PUBLISH(_publish_state(rwu, sw, [3.0, 0.0], np.ones(NODES.n)), p, None).root_uptake.uptake
    )
    np.testing.assert_allclose(out[0].sum(), rwu[0].sum(), rtol=REL)  # the layer grid ends on a node bottom
    np.testing.assert_array_equal(out[1], 0.0)


def test_publish_gradients_with_respect_to_rwu_are_finite_and_linear() -> None:
    rng = np.random.default_rng(11)
    ll = np.round(rng.uniform(0.05, 0.15, LAYERS.n), 3)
    sw = ll + np.asarray([0.1, -0.01, 0.0, 0.05, 0.2])
    p = PublishUptakeParams(nodes=NODES, layers=LAYERS, ll=jnp.asarray(ll, DT))
    found = rng.uniform(0.0, 0.05, NODES.n)

    def loss(rwu):
        st = _publish_state(np.zeros((1, LAYERS.n)), sw, [4.0], found).replace(rwu=rwu)
        return jnp.sum(rzwqm_publish_uptake(st, p, None).root_uptake.uptake)

    rwu = jnp.asarray(rng.uniform(0.0, 0.1, (1, LAYERS.n)), DT)
    g = np.asarray(jax.grad(loss)(rwu))
    assert np.all(np.isfinite(g))
    # d(total node uptake)/d(rwu_L) = 1 on SW > LL layers, 0 on SW < LL and on the SW == LL layer
    np.testing.assert_allclose(g[0], [1.0, 0.0, 0.0, 1.0, 1.0], rtol=REL, atol=ABS)


# ------------------------------------------------------------------ in the contract's day
def test_both_processes_fit_the_contract_day_and_its_lags() -> None:
    from agrijax.models.day_rzwqm46 import SLOT, day_processes, day_rzwqm46

    limit = bind(
        rzwqm_uptake_limit,
        own="water_supply.uptake_limit",
        ports={
            "theta": "soil_water.theta",
            "root_uptake": f"iface.root_uptake.{SLOT}",
            "water": f"iface.crop_water.{SLOT}",
            "pet": "iface.pet",
            "sink_in": "soil_water.sink_in",
        },
        params="uptake_limit",
        name="soil_water.uptake_limit",
    )
    publish = bind(
        rzwqm_publish_uptake,
        own=f"water_supply.{SLOT}",
        ports={
            "water": f"iface.crop_water.{SLOT}",
            "root_uptake": f"iface.root_uptake.{SLOT}",
            "sink_in": "soil_water.sink_in",
        },
        params="publish_uptake",
        name=f"crops.{SLOT}.publish_uptake",
    )
    d = day_rzwqm46(SLOT)
    procs = day_processes(
        SLOT, replace={"soil_water.uptake_limit": limit, f"crops.{SLOT}.publish_uptake": publish}
    )
    report = d.check(d.compile(procs))
    assert {
        ("soil_water.uptake_limit", f"iface.root_uptake.{SLOT}"),
        ("soil_water.uptake_limit", f"iface.crop_water.{SLOT}.trwup"),
    } <= set(report.used)
    assert report.unused_pairs == ()
    assert set(limit.reads) == {
        "soil_water.theta",
        f"iface.root_uptake.{SLOT}",
        f"iface.crop_water.{SLOT}.trwup",
        "iface.pet.transpiration",
    }
    assert limit.writes == ("soil_water.sink_in.uptake",)
    assert publish.writes == (f"iface.root_uptake.{SLOT}.uptake",)


def test_processes_pass_the_write_check() -> None:
    old = os.environ.get(CHECK_ENV)
    os.environ[CHECK_ENV] = "1"
    try:
        rng = np.random.default_rng(1)
        theta_wp = rng.uniform(0.05, 0.1, NODES.n)
        st = _limit_state(rng.uniform(0, 0.05, (1, NODES.n)), [0.5], 0.2, theta_wp + 0.05)
        rzwqm_uptake_limit(st, UptakeLimitParams(theta_wp=jnp.asarray(theta_wp, DT)), None)
        ps = _publish_state(
            rng.uniform(0, 0.1, (1, LAYERS.n)), np.full(LAYERS.n, 0.2), [3.0], np.zeros(NODES.n)
        )
        rzwqm_publish_uptake(
            ps, PublishUptakeParams(nodes=NODES, layers=LAYERS, ll=jnp.full(LAYERS.n, 0.1, DT)), None
        )
    finally:
        if old is None:
            os.environ.pop(CHECK_ENV, None)
        else:
            os.environ[CHECK_ENV] = old
