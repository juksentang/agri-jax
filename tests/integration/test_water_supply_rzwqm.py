"""The water-supply slot against RZWQM2 4.6 at CA-TPA 2015-2023, day by day.

Data (kept outside the repository, ``allow_skip``): the daily entry/exit tables of the instrumented
RZWQM2 4.6 run of CA-TPA 2015-2023,
``<data>/dumps/tables/rzwqm46_catpa2015_2023/{physcl,dssatdrv}_{entry,exit}.npz`` (the instrumented
build gives the same outputs as the plain one, ``tests/diff/test_a12_rzwqm.py``).

1. **Uptake limit** (:func:`rzwqm_uptake_limit`, ``soil_water/wuf@rzwqm2-4.6:faithful``) on all
   3287 days, all days in one ``vmap``: inputs the ``PHYSCL`` entry node uptake ``QSR`` and
   ``TRWUP`` (yesterday's, the contract's two lags), the entry ``THETA``, today's ``PETPLANT`` and
   ``SOILHP(9)`` of each node's horizon; output against the ``PHYSCL`` exit ``QSR``. The zero
   pattern is identical and the values agree within two REAL*4 roundings (the reference rounds
   ``WUF`` and the product to REAL*4). The limit acts on 963 days, all crop days, with
   ``WUF < 1`` on 954 of the 1088 crop days.
2. **Publish** (:func:`rzwqm_publish_uptake`, ``crop_iface/publish_uptake@rzwqm2-4.6:faithful``) on
   all 1088 crop days: inputs the ``DSSATDRV`` exit ``RWU``, ``SW``, ``EOP``, ``SOILPROP%LL`` and the
   day's sink (the ``DSSATDRV`` entry ``QSR``, equal to the ``PHYSCL`` exit ``QSR``); output against
   the ``DSSATDRV`` exit ``QSR`` (the reference's ``REALMATCH`` in REAL*4).
3. **The SW == LL quirk**: 22 crop days (2016-243 to 2016-264, 33 layer-days on layers 4
   and 5) have a layer at ``SW == LL`` exactly; with the quirk the publish matches on all 1088
   days, without it (the layer gets 0 where the reference keeps the array's earlier value) exactly those
   22 days fail.
   The precision trap: the same comparison on ``SW`` remapped in float64 from the node ``THETA``
   is counted in float64 and on REAL*4-rounded values.
4. **The chain over the lag**: our publish of day ``d - 1`` feeds our limit of day ``d``
   (with the reference's ``TRWUP`` of ``d - 1``) and gives the ``PHYSCL`` exit ``QSR`` of day ``d``.

The measured numbers are printed with the prefix ``water_supply`` (``-s``).
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core.grids import remap_intensive, rzwqm_lyrset, rzwqm_nodes
from agrijax.iface.crop import CropWaterIn
from agrijax.iface.soil import NodeUptake, SinkInputs
from agrijax.iface.surface import PETFluxes
from agrijax.port import dumps
from agrijax.processes.water_supply import (
    PublishUptakeParams,
    RootwuState,
    UptakeLimitParams,
    UptakeLimitState,
    rzwqm_publish_uptake,
    rzwqm_uptake_limit,
)

pytestmark = [
    pytest.mark.allow_skip(reason="needs the RZWQM2 full-state dump tables of CA-TPA 2015-2023"),
    pytest.mark.skipif(
        not jax.config.read("jax_enable_x64"), reason="the day-by-day comparison runs in float64"
    ),
]

TABLES = Path("dumps/tables/rzwqm46_catpa2015_2023")
HOURS = 24.0  # RZWQM2 qsr is per hour: node amount [cm d-1] = qsr * 24 * TL
#: two REAL*4 roundings (WUF to REAL*4, then the REAL*4 product): 2 * 2**-24 relative
REAL4_TWO_ROUNDINGS = 2.0**-23
#: the reference's REALMATCH is REAL*4 (layer rate, weighted sum, quotient): tolerance of the
#: layer -> node map measured on the reference itself (tests/diff/test_a12_rzwqm.py, rtol 1e-6)
REAL4_REALMATCH_RTOL = 1e-6


def _f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _next(d: int) -> int:
    t = datetime.date(d // 1000, 1, 1) + datetime.timedelta(days=d % 1000)
    return t.year * 1000 + t.timetuple().tm_yday


class Ref:
    def __init__(self, data_dir: Path) -> None:
        tab = data_dir / TABLES
        names = ("physcl_entry", "physcl_exit", "dssatdrv_entry", "dssatdrv_exit", "rootwu_entry")
        for n in names:
            if not (tab / f"{n}.npz").is_file():
                pytest.skip(f"{tab / n}.npz not found")
        self.t = {n: dumps.load_table(tab / f"{n}.npz")[0] for n in names}
        po = self.t["physcl_exit"].values
        self.nn = int(po["NN"][0])
        self.nl = int(self.t["rootwu_entry"].values["NLAYR"][0])
        self.tl = _f64(po["TL"][0, : self.nn])
        self.nodes = rzwqm_nodes(_f64(po["TLT"][0, : self.nn]))
        self.layers = rzwqm_lyrset(self.nodes)
        dx = self.t["dssatdrv_exit"].values
        np.testing.assert_array_equal(self.layers.thickness, _f64(dx["SOILPROP%DLAYR"][0, : self.nl]))
        ll = dx["SOILPROP%LL"][:, : self.nl]
        assert np.all(ll == ll[0])  # the crop layers' LL does not change over the run
        self.ll = _f64(ll[0])

    def node_amount(self, qsr: np.ndarray) -> np.ndarray:
        """RZWQM2 ``qsr`` [cm h-1 per cm] -> node amount [cm d-1] (extensive)."""
        return _f64(qsr)[..., : self.nn] * HOURS * self.tl


@pytest.fixture(scope="module")
def ref(data_dir: Path) -> Ref:
    return Ref(data_dir)


def _z(shape: tuple[int, ...]) -> jnp.ndarray:
    return jnp.zeros(shape)


def _limit_inputs(ref: Ref, q_in: np.ndarray, trwup: np.ndarray, rows: np.ndarray):
    """Batched (one day per row) inputs of the limit on PHYSCL rows ``rows``."""
    pi, po = ref.t["physcl_entry"].values, ref.t["physcl_exit"].values
    n, nn = len(rows), ref.nn
    jh = pi["NDXN2H"][rows, :nn] - 1
    theta_wp = np.take_along_axis(_f64(pi["SOILHP"][rows, 8, :]), jh, axis=1)  # SOILHP(9) of the horizon
    state = UptakeLimitState(
        theta=jnp.asarray(_f64(pi["THETA"][rows, :nn])),
        root_uptake=NodeUptake(uptake=jnp.asarray(q_in)[:, None, :]),
        water=CropWaterIn(sw=_z((n, ref.nl)), eop=_z((n, 1)), trwup=jnp.asarray(trwup)[:, None]),
        pet=PETFluxes(
            transpiration=jnp.asarray(_f64(po["PETPLANT"][rows])),
            soil_evaporation=_z((n,)),
            residue_evaporation=_z((n,)),
            reference_short=_z((n,)),
            reference_tall=_z((n,)),
            eo_priestley_taylor=_z((n,)),
        ),
        sink_in=SinkInputs(**{k: _z((n, nn)) for k in SinkInputs.__dataclass_fields__}),
    )
    return state, UptakeLimitParams(theta_wp=jnp.asarray(theta_wp))


_LIMIT = jax.jit(jax.vmap(rzwqm_uptake_limit, in_axes=(0, 0, None)))
_PUBLISH = jax.jit(jax.vmap(rzwqm_publish_uptake, in_axes=(0, None, None)))


def _rel(got: np.ndarray, want: np.ndarray) -> float:
    nz = want != 0.0
    return float(np.max(np.abs(got[nz] - want[nz]) / np.abs(want[nz]))) if nz.any() else 0.0


# ------------------------------------------------------------------ 1. the uptake limit
def test_uptake_limit_day_by_day(ref: Ref) -> None:
    pi, po = ref.t["physcl_entry"].values, ref.t["physcl_exit"].values
    rows = np.arange(len(ref.t["physcl_entry"]))
    np.testing.assert_array_equal(ref.t["physcl_entry"].date, ref.t["physcl_exit"].date)
    state, params = _limit_inputs(ref, ref.node_amount(pi["QSR"]), _f64(pi["TRWUP"]), rows)
    got = np.asarray(_LIMIT(state, params, None).sink_in.uptake)
    want = ref.node_amount(po["QSR"])
    assert got.shape == want.shape == (3287, ref.nn)
    np.testing.assert_array_equal(got == 0.0, want == 0.0)
    rel = _rel(got, want)
    # the gate and the factor, counted on the reference's own inputs
    pet, tr = _f64(po["PETPLANT"]), _f64(pi["TRWUP"])
    on = (pet > 0.0) & (tr != 0.0)
    wuf = np.where(on & (pet <= tr), pet / np.where(on, tr, 1.0), 1.0)
    crop = ref.t["physcl_entry"].index_of(ref.t["rootwu_entry"].date)
    changed = np.any(ref.node_amount(pi["QSR"]) != want, axis=1)
    out = {
        "days": len(rows),
        "days_limit_on": int(on.sum()),
        "crop_days": len(crop),
        "crop_days_limit_on": int(on[crop].sum()),
        "crop_days_wuf_lt_1": int((wuf[crop] < 1.0).sum()),
        "wuf_min": float(wuf.min()),
        "days_qsr_changed_by_physcl": int(changed.sum()),
        "max_rel_error_vs_physcl_exit_qsr": rel,
        "nodes_below_wilting_point_on_limited_days": int(
            np.sum(on[:, None] & (_f64(pi["THETA"][:, : ref.nn]) < np.asarray(params.theta_wp)))
        ),
    }
    print("water_supply " + json.dumps({"uptake_limit": out}))
    assert rel <= REAL4_TWO_ROUNDINGS, rel
    assert out["days_limit_on"] == out["crop_days_limit_on"] == 963
    assert out["crop_days"] == 1088 and out["crop_days_wuf_lt_1"] == 954


# ------------------------------------------------------------------ 2-3. the publish and the quirk
def _publish_inputs(ref: Ref, sw: np.ndarray, sink: np.ndarray) -> RootwuState:
    dx = ref.t["dssatdrv_exit"].values
    n, nl, nn = len(dx["EOP"]), ref.nl, ref.nn
    return RootwuState(
        tss=_z((n, 1, nl)),
        rwu=jnp.asarray(_f64(dx["RWU"][:, :nl]))[:, None, :],
        water=CropWaterIn(sw=jnp.asarray(sw), eop=jnp.asarray(_f64(dx["EOP"]))[:, None], trwup=_z((n, 1))),
        root_uptake=NodeUptake(uptake=_z((n, 1, nn))),
        sink_in=SinkInputs(**{k: _z((n, nn)) for k in SinkInputs.__dataclass_fields__}).replace(
            uptake=jnp.asarray(sink)
        ),
    )


def _publish(ref: Ref, sw: np.ndarray, sink: np.ndarray) -> np.ndarray:
    p = PublishUptakeParams(nodes=ref.nodes, layers=ref.layers, ll=jnp.asarray(ref.ll))
    return np.asarray(_PUBLISH(_publish_inputs(ref, sw, sink), p, None).root_uptake.uptake)[:, 0, :]


def _day_fails(ref: Ref, got: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Days where the node uptake misses the reference beyond the REAL*4 ``REALMATCH`` tolerance
    (rtol 1e-6, atol 1e-12 in ``qsr`` units as in ``tests/diff/test_a12_rzwqm.py``)."""
    tol = REAL4_REALMATCH_RTOL * np.abs(want) + 1e-12 * HOURS * ref.tl
    return np.any(np.abs(got - want) > tol, axis=1)


def test_publish_day_by_day_with_the_sw_eq_ll_quirk(ref: Ref) -> None:
    dd_in, dx = ref.t["dssatdrv_entry"].values, ref.t["dssatdrv_exit"].values
    po = ref.t["physcl_exit"]
    days = ref.t["dssatdrv_exit"].date
    np.testing.assert_array_equal(ref.t["dssatdrv_entry"].date, days)
    j = po.index_of(days)
    # the array DSSATDRV finds is the morning's limited qsr: P4 of the day
    np.testing.assert_array_equal(dd_in["QSR"][:, : ref.nn], po.values["QSR"][j, : ref.nn])
    # the PET gate of DSSATDRV and EOP > 0 select the same days (EOP = 10 PET, ISTRESS = 0)
    np.testing.assert_array_equal(_f64(dx["PET"]) <= 0.0, _f64(dx["EOP"]) <= 0.0)
    sink = ref.node_amount(dd_in["QSR"])
    sw = _f64(dx["SW"][:, : ref.nl])
    want = ref.node_amount(dx["QSR"])
    got = _publish(ref, sw, sink)
    no_quirk = _publish(ref, sw, np.zeros_like(sink))  # SW == LL layers get 0 (no quirk)
    eq = dx["SW"][:, : ref.nl] == dx["SOILPROP%LL"][:, : ref.nl]
    eq_days = np.any(eq, axis=1)
    fail, fail_nq = _day_fails(ref, got, want), _day_fails(ref, no_quirk, want)
    tot = np.sum(want, axis=1)
    dev = np.abs(no_quirk - want).sum(axis=1)
    out = {
        "crop_days": len(days),
        "days_pet_le_0": int(np.sum(_f64(dx["PET"]) <= 0.0)),
        "max_rel_error": _rel(got, want),
        "max_abs_error_where_ref_zero_cm": float(np.max(np.abs(got[want == 0.0])))
        if np.any(want == 0)
        else 0.0,
        "days_failing_with_quirk": int(fail.sum()),
        "days_failing_without_quirk": int(fail_nq.sum()),
        "sw_eq_ll_days": int(eq_days.sum()),
        "sw_eq_ll_layer_days": int(eq.sum()),
        "sw_eq_ll_layers_1based": sorted({int(k) + 1 for k in np.nonzero(eq)[1]}),
        "sw_eq_ll_first_last": [int(days[eq_days][0]), int(days[eq_days][-1])],
        "without_quirk_profile_uptake_missed_cm_max": float(dev[eq_days].max()),
        "without_quirk_profile_uptake_missed_frac_max": float((dev[eq_days] / tot[eq_days]).max()),
        "without_quirk_profile_uptake_missed_frac_median": float(np.median(dev[eq_days] / tot[eq_days])),
    }
    print("water_supply " + json.dumps({"publish": out}))
    assert out["crop_days"] == 1088 and out["days_failing_with_quirk"] == 0
    np.testing.assert_array_equal(fail_nq, eq_days)
    assert out["sw_eq_ll_days"] == 22 and out["sw_eq_ll_layer_days"] == 33


def test_sw_eq_ll_precision_trap_of_the_float64_chain(ref: Ref) -> None:
    """The ``SW == LL`` precision trap, with numbers: ``SW`` remapped in float64 from the node ``THETA``
    (our remap_in) against the REAL*4 ``LL``: how many of the reference's 33 ``SW == LL`` layer-days each
    comparison finds, and the publish from that ``SW`` against the reference."""
    dx = ref.t["dssatdrv_exit"].values
    po = ref.t["physcl_exit"]
    days = ref.t["dssatdrv_exit"].date
    j = po.index_of(days)
    sw64 = np.asarray(
        remap_intensive(jnp.asarray(_f64(po.values["THETA"][j, : ref.nn])), ref.nodes, ref.layers)
    )
    eq_ref = dx["SW"][:, : ref.nl] == dx["SOILPROP%LL"][:, : ref.nl]
    ll = ref.ll[None, :]
    eq64 = sw64 == ll
    gt64 = sw64 > ll
    eq32 = sw64.astype(np.float32) == ll.astype(np.float32)
    want = ref.node_amount(dx["QSR"])
    got = _publish(ref, sw64, ref.node_amount(ref.t["dssatdrv_entry"].values["QSR"]))
    fail = _day_fails(ref, got, want)
    out = {
        "ref_eq_layer_days": int(eq_ref.sum()),
        "float64_eq_on_ref_eq": int((eq64 & eq_ref).sum()),
        "float64_gt_on_ref_eq": int((gt64 & eq_ref).sum()),
        "real4_eq_on_ref_eq": int((eq32 & eq_ref).sum()),
        "real4_eq_not_ref_eq": int((eq32 & ~eq_ref).sum()),
        "max_abs_sw64_minus_ll_on_ref_eq": float(np.max(np.abs(sw64 - ll)[eq_ref])),
        "publish_from_sw64_days_failing": int(fail.sum()),
        "publish_from_sw64_max_rel_error": _rel(got, want),
    }
    print("water_supply " + json.dumps({"precision_trap": out}))
    assert out["ref_eq_layer_days"] == 33
    assert out["real4_eq_on_ref_eq"] == 33 and out["real4_eq_not_ref_eq"] == 0
    assert out["publish_from_sw64_days_failing"] == 0


# ------------------------------------------------------------------ 4. the chain over the lag
def test_publish_of_yesterday_through_the_limit_gives_todays_sink(ref: Ref) -> None:
    dx, pi = ref.t["dssatdrv_exit"], ref.t["physcl_entry"]
    ours = _publish(
        ref, _f64(dx.values["SW"][:, : ref.nl]), ref.node_amount(ref.t["dssatdrv_entry"].values["QSR"])
    )
    nxt = np.asarray([_next(int(d)) for d in dx.date])
    rows = pi.index_of(nxt)
    k = np.nonzero(rows >= 0)[0]
    rows = rows[k]
    # yesterday's TRWUP as the crop water record holds it at the end of the day (DSSATDRV exit)
    state, params = _limit_inputs(ref, ours[k], _f64(dx.values["TRWUP"][k]), rows)
    got = np.asarray(_LIMIT(state, params, None).sink_in.uptake)
    want = ref.node_amount(ref.t["physcl_exit"].values["QSR"][rows])
    fail = _day_fails(ref, got, want)
    out = {"day_pairs": len(k), "days_failing": int(fail.sum()), "max_rel_error": _rel(got, want)}
    print("water_supply " + json.dumps({"chain": out}))
    assert len(k) > 1000 and out["days_failing"] == 0
