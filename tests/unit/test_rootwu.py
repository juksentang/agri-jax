"""DSSAT ``ROOTWU`` potential root water uptake (plan 19 A4): kernel, carried state, process.

Data-free checks:

* :func:`rootwu_estimate` against a plain-Python transcription of ``SPAM/ROOTWU.for`` (DSSAT-CSM
  v4.8.6.0, BSD-3): per-layer ``if``/``else``, ``math.exp``/``math.log``, Python floats and a
  ``SAVE``d counter kept in a dict. It shares no code with the vectorised kernel. Random states
  visit every branch: no roots (``RLV <= 1e-5``), dry layers (``SW <= LL``), ``LL > 0.30``,
  ``RLV > exp(SWCON3)``, the exponent cap at 40, near-saturated layers before and after the
  two-day delay, ``PORMIN = 0`` (flooded rice), the ``RWUMX`` cap and the canopy gate
  (``XHLAI = 0``: ``ROOTWU`` not called, counter untouched);
* the saturation-day counter carried over a sequence of calls equals the Fortran's ``SAVE``;
* gradients with respect to every float input are finite on every visited state, including the
  branch boundaries (``SW = LL``, ``RLV = 1e-5``, ``SAT - SW = PORMIN``);
* the process :func:`rootwu_supply` writes ``TRWUP`` into the crop water record and nothing else
  of it, alone and bound to ``iface`` paths, and passes ``AGRI_JAX_CHECK=1``.

The comparison with the reference model's own ``ROOTWU`` (dumps of ``dscsm048`` on 58 maize
treatments and of RZWQM2 4.6 at CA-TPA) is ``tests/integration/test_rootwu_dssat.py``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import bind, compose, registry
from agrijax.core.ports import Binding
from agrijax.core.process import CHECK_ENV
from agrijax.core.state import get_path
from agrijax.processes.soil_water.uptake import (
    CropWaterIn,
    RootRecord,
    RootwuParams,
    RootwuState,
    SoilView,
    rootwu_estimate,
    rootwu_supply,
)

X64 = jax.config.jax_enable_x64
REL = 1e-12 if X64 else 2e-5
ABS = 1e-15 if X64 else 1e-8


# ------------------------------------------------------------------ plain-Python ROOTWU
def ref_rootwu(dlayr, ll, rlv, rwumx, sat, sw, pormin, tss, xhlai):
    """One SPAM RATE call of ROOTWU for one crop; ``tss`` is the SAVEd list, updated in place."""
    n = len(dlayr)
    if not xhlai > 0.0:
        return [0.0] * n, 0.0
    swcon1, swcon3 = 1.32e-3, 7.01
    trwup = 0.0
    rwu = [0.0] * n
    for L in range(n):
        swcon2 = 120.0 - 250.0 * ll[L]
        if ll[L] > 0.30:
            swcon2 = 45.0
        if rlv[L] <= 0.00001 or sw[L] <= ll[L]:
            r = 0.0
        else:
            if rlv[L] > math.exp(swcon3):
                den = swcon3 - math.log(swcon3)
            else:
                den = swcon3 - math.log(rlv[L])
            r = swcon1 * math.exp(min(swcon2 * (sw[L] - ll[L]), 40.0)) / den
            if (sat[L] - sw[L]) >= pormin or pormin < 1e-6:
                tss[L] = 0.0
            else:
                tss[L] = tss[L] + 1.0
            if tss[L] > 2.0:
                swexf = max((sat[L] - sw[L]) / pormin, 0.0)
            else:
                swexf = 1.0
            swexf = min(swexf, 1.0)
            r = min(r, rwumx * swexf)
            r = min(r, rwumx)
        rwu[L] = r * dlayr[L] * rlv[L]
        trwup += rwu[L]
    return rwu, trwup


N_LAYER = 9
DLAYR = [5.0, 10.0, 15.0, 15.0, 15.0, 30.0, 30.0, 30.0, 30.0]


def random_state(rng: np.random.Generator, n_crop: int) -> dict:
    """Soil shared by the crops, roots per crop; every branch of ROOTWU is likely."""
    ll = rng.uniform(0.02, 0.36, N_LAYER)
    sat = ll + rng.uniform(0.15, 0.35, N_LAYER)
    kind = rng.integers(0, 4, N_LAYER)
    sw = np.select(
        [kind == 0, kind == 1, kind == 2],
        [ll - rng.uniform(0.0, 0.02, N_LAYER), sat - rng.uniform(0.0, 0.04, N_LAYER), ll + 0.3],
        ll + rng.uniform(0.0, 0.1, N_LAYER),
    )
    sw = np.minimum(sw, sat)
    rlv = rng.uniform(0.0, 3.0, (n_crop, N_LAYER))
    rlv[rng.uniform(size=rlv.shape) < 0.15] = 0.0
    rlv[rng.uniform(size=rlv.shape) < 0.05] = 5e-6
    rlv[rng.uniform(size=rlv.shape) < 0.02] = 2000.0  # > exp(SWCON3): the capped denominator
    return dict(
        dlayr=np.asarray(DLAYR),
        ll=ll,
        sat=sat,
        sw=sw,
        rlv=rlv,
        rwumx=rng.choice([0.03, 0.05, 1e-3], n_crop),
        pormin=rng.choice([0.05, 0.02, 0.0], n_crop),
        xhlai=rng.choice([0.0, 0.8, 3.0], n_crop, p=[0.15, 0.4, 0.45]),
        tss=rng.integers(0, 5, (n_crop, N_LAYER)).astype(float),
    )


def _records(c: dict):
    a = jnp.asarray
    root = RootRecord(
        rlv=a(c["rlv"]),
        rtdep=a(np.zeros(len(c["rwumx"]))),
        rwumx=a(c["rwumx"]),
        pormin=a(c["pormin"]),
        xhlai=a(c["xhlai"]),
    )
    soil = SoilView(dlayr=a(c["dlayr"]), ll=a(c["ll"]), sat=a(c["sat"]), sw=a(c["sw"]))
    return root, soil


_EST = jax.jit(rootwu_estimate)


@pytest.mark.parametrize("seed", range(8))
def test_estimate_equals_the_reference_loop(seed: int) -> None:
    rng = np.random.default_rng(700 + seed)
    hits = dict(dry=0, noroot=0, llhigh=0, capden=0, sat_delay=0, sat_on=0, flood=0, gate=0, capmx=0)
    for _ in range(40):
        c = random_state(rng, 3)
        if not X64:  # the reference sees the float32-rounded inputs the kernel sees
            c = {k: np.asarray(v, dtype=np.float32).astype(np.float64) for k, v in c.items()}
        root, soil = _records(c)
        got = _EST(root, soil, jnp.asarray(c["tss"]))
        for k in range(3):
            tss = list(c["tss"][k])
            rwu, trwup = ref_rootwu(
                list(c["dlayr"]),
                list(c["ll"]),
                list(c["rlv"][k]),
                float(c["rwumx"][k]),
                list(c["sat"]),
                list(c["sw"]),
                float(c["pormin"][k]),
                tss,
                float(c["xhlai"][k]),
            )
            np.testing.assert_allclose(np.asarray(got.rwu[k]), rwu, rtol=REL, atol=ABS)
            np.testing.assert_allclose(float(got.trwup[k]), trwup, rtol=REL, atol=ABS)
            np.testing.assert_array_equal(np.asarray(got.tss[k]), tss)
            on = c["xhlai"][k] > 0
            hits["gate"] += not on
            for L in range(N_LAYER):
                act = on and c["rlv"][k][L] > 1e-5 and c["sw"][L] > c["ll"][L]
                hits["dry"] += on and c["sw"][L] <= c["ll"][L]
                hits["noroot"] += on and c["rlv"][k][L] <= 1e-5
                hits["llhigh"] += act and c["ll"][L] > 0.30
                hits["capden"] += act and c["rlv"][k][L] > math.exp(7.01)
                hits["flood"] += act and c["pormin"][k] == 0.0
                wet = act and c["sat"][L] - c["sw"][L] < c["pormin"][k]
                hits["sat_delay"] += wet and tss[L] <= 2
                hits["sat_on"] += wet and tss[L] > 2
                hits["capmx"] += act and rwu[L] == pytest.approx(
                    c["rwumx"][k] * c["dlayr"][L] * c["rlv"][k][L]
                )
    assert all(v > 0 for v in hits.values()), hits


def test_saturation_counter_is_carried_like_the_fortran_save() -> None:
    """30 calls with a wetting and drying layer: the counter returned by each call, fed to the
    next, equals the Fortran's SAVEd TSS; days without canopy leave it untouched."""
    rng = np.random.default_rng(42)
    c = random_state(rng, 2)
    c["rlv"] = np.abs(c["rlv"]) + 0.5
    c["pormin"] = np.asarray([0.05, 0.05])
    tss_j = jnp.zeros((2, N_LAYER))
    tss_r = [[0.0] * N_LAYER, [0.0] * N_LAYER]
    maxed = 0.0
    for day in range(30):
        wet = (day // 4) % 2 == 0
        sw = np.where(np.arange(N_LAYER) < 4, c["sat"] - (0.01 if wet else 0.12), c["ll"] + 0.05)
        c["sw"] = sw
        c["xhlai"] = np.asarray([1.0, 0.0 if day in (5, 6, 17) else 2.0])
        root, soil = _records(c)
        got = _EST(root, soil, tss_j)
        for k in range(2):
            ref_rootwu(
                list(c["dlayr"]), list(c["ll"]), list(c["rlv"][k]), 0.03, list(c["sat"]), list(sw),
                0.05, tss_r[k], float(c["xhlai"][k]),
            )  # fmt: skip
        np.testing.assert_array_equal(np.asarray(got.tss), np.asarray(tss_r))
        tss_j = got.tss
        maxed = max(maxed, float(np.max(np.asarray(tss_j))))
    assert maxed >= 3.0  # the two-day delay was passed


@pytest.mark.parametrize("seed", range(3))
def test_gradients_are_finite_on_every_state_and_boundary(seed: int) -> None:
    rng = np.random.default_rng(900 + seed)

    def scalar(rlv, rwumx, pormin, xhlai, dlayr, ll, sat, sw, tss):
        root = RootRecord(rlv=rlv, rtdep=jnp.zeros_like(rwumx), rwumx=rwumx, pormin=pormin, xhlai=xhlai)
        r = rootwu_estimate(root, SoilView(dlayr=dlayr, ll=ll, sat=sat, sw=sw), tss)
        return jnp.sum(r.rwu) + jnp.sum(r.trwup) + jnp.sum(r.tss)

    grad = jax.jit(jax.grad(scalar, argnums=tuple(range(9))))
    for i in range(30):
        c = random_state(rng, 3)
        if i % 3 == 0:  # exactly on the branch boundaries
            c["sw"][:3] = c["ll"][:3]
            c["rlv"][:, 3] = 1e-5
            c["sw"][4] = c["sat"][4] - 0.05
            c["pormin"][:] = 0.05
        args = [
            jnp.asarray(c[k], dtype=float)
            for k in ("rlv", "rwumx", "pormin", "xhlai", "dlayr", "ll", "sat", "sw", "tss")
        ]
        for g in grad(*args):
            assert np.all(np.isfinite(np.asarray(g)))
    # the gradient reaches the published crop numbers where they bind
    c = random_state(np.random.default_rng(1), 1)
    c["xhlai"][:] = 1.0
    c["sw"] = c["ll"] + 0.3  # exponent capped: uptake = RWUMX
    c["rlv"][:] = 1.0
    c["tss"][:] = 0.0
    args = [
        jnp.asarray(c[k], dtype=float)
        for k in ("rlv", "rwumx", "pormin", "xhlai", "dlayr", "ll", "sat", "sw", "tss")
    ]
    g = grad(*args)
    assert float(g[1][0]) > 0.0


# ------------------------------------------------------------------ the process
def _producer(c: dict, k_crop: int = 2) -> tuple[RootwuState, RootwuParams]:
    root, soil = _records(c)
    params = RootwuParams(dlayr=soil.dlayr, ll=soil.ll, sat=soil.sat)
    water = CropWaterIn(sw=soil.sw, eop=jnp.full((k_crop,), 4.0), trwup=jnp.zeros((k_crop,)))
    st = RootwuState.initial(k_crop, N_LAYER).replace(tss=jnp.asarray(c["tss"]), root=root, water=water)
    return st, params


def test_process_writes_trwup_into_the_water_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHECK_ENV, "1")
    c = random_state(np.random.default_rng(5), 2)
    c["xhlai"][:] = 1.0
    st, params = _producer(c)
    out = rootwu_supply(st, params, None)  # eager: the writes check compares values
    want = rootwu_estimate(
        st.root, SoilView(dlayr=params.dlayr, ll=params.ll, sat=params.sat, sw=st.water.sw), st.tss
    )
    assert np.asarray(out.water.trwup).tobytes() == np.asarray(want.trwup).tobytes()
    assert np.asarray(out.rwu).tobytes() == np.asarray(want.rwu).tobytes()
    assert np.asarray(out.tss).tobytes() == np.asarray(want.tss).tobytes()
    for a_, b_ in zip(jax.tree_util.tree_leaves((out.water.sw, out.water.eop, out.root)),
                      jax.tree_util.tree_leaves((st.water.sw, st.water.eop, st.root)), strict=True):  # fmt: skip
        assert np.asarray(a_).tobytes() == np.asarray(b_).tobytes()  # read-only parts unchanged
    p = registry["rootwu_supply"]
    assert p.key == "water_supply/rootwu@dssat-4.8.6.0:faithful" and p.fortran_name == "ROOTWU"
    assert p.info is not None and p.info.provenance == "translated_bsd3"


def test_process_bound_to_interface_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound at ``soil_water.rootwu`` with its ports on ``iface``: same numbers, global paths,
    and under AGRI_JAX_CHECK only the declared paths change."""
    c = random_state(np.random.default_rng(6), 2)
    c["xhlai"][:] = 1.5
    st, params = _producer(c)
    ports = {"root": "iface.root.maize", "water": "iface.crop_water.maize"}
    bound = bind(rootwu_supply, own="soil_water.rootwu", ports=ports)
    assert bound.writes == ("soil_water.rootwu.tss", "soil_water.rootwu.rwu", "iface.crop_water.maize.trwup")
    assert set(bound.reads) == {"soil_water.rootwu.tss", "iface.root.maize", "iface.crop_water.maize.sw"}
    g = compose(Binding("soil_water.rootwu", tuple(ports.items())).entries(st))
    monkeypatch.setenv(CHECK_ENV, "1")
    out = bound(g, params, None)
    alone = rootwu_supply(st, params, None)
    assert (
        np.asarray(get_path(out, "iface.crop_water.maize.trwup")).tobytes()
        == np.asarray(alone.water.trwup).tobytes()
    )
    assert np.asarray(get_path(out, "soil_water.rootwu.tss")).tobytes() == np.asarray(alone.tss).tobytes()
    for a_, b_ in zip(
        jax.tree_util.tree_leaves(get_path(out, "iface.root.maize")),
        jax.tree_util.tree_leaves(get_path(g, "iface.root.maize")),
        strict=True,
    ):
        assert np.asarray(a_).tobytes() == np.asarray(b_).tobytes()  # the input port is unchanged
