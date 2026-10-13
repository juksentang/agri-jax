"""The nitrogen factors of the ``nstress_replay`` growth besides ``NSTRES``, data-free.

DSSAT-CSM v4.8.6.0 ``MZ_GROSUB`` uses three more outputs of ``MZ_NFACTO`` for the crop's mass,
which the P10 record (:class:`~agrijax.iface.crop.CropNIn`) now carries next to ``nstres``:

* ``AGEFAC`` in the expansion factor ``min(AGEFAC, TURFAC, 1 - SATFAC)`` of stages 2-4 (not stage
  1): on every stage 2-4 day of a synthetic season, the variant with ``AGEFAC = a`` (and
  ``FSLFN = 0``, so that the senescence factor is 1) is the faithful day with ``TURFAC`` replaced by
  ``min(TURFAC, a)``, bit for bit; on stage 1 and 5 days it is the faithful day;
* ``AGEFAC`` in the senescence factor ``SLFN = (1 - FSLFN) + FSLFN AGEFAC`` of every stage: on the
  stage 1 and 5 days the senesced area equals ``min(max(SENLA + (PLA - SENLA)(1 - min(SLFW, SLFC,
  SLFT, SLFN)), SLAN), PLA)`` written out in NumPy;
* ``NDEF3`` and ``NPOOL`` in the grain-number cap ``GPP = min(GPP NDEF3, NPOOL / (0.062 x 0.0095))``
  once, on the first day of effective grain filling: over a season the grain number is the
  faithful one before that day and the capped one from it on;
* the record's defaults (``1``, ``NPOOL_NO_CAP``) and the replay producer's optional series.

With the defaults the variant is the faithful growth bit for bit
(``tests/unit/test_ceres_nstress_replay.py``).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core import Model, run
from agrijax.iface.crop import NPOOL_NO_CAP, CropNIn
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES,
    CROP_PROCESSES_NSTRESS_REPLAY,
    DSSAT_COEFFICIENTS,
    REPLAY_PROCESSES,
    CeresMaizeState,
    ceres_growth,
    ceres_growth_nstress_replay,
)
from agrijax.processes.crop.ceres_maize.growth import grain_number_cap, nitrogen_senescence_factor
from agrijax.processes.n_supply import CropNReplayForcing, CropNReplayState, crop_n_replay

from .test_ceres_nstress_replay import _bytes_equal, _pre_growth_states, _season

X64 = bool(jax.config.read("jax_enable_x64"))
_STATES = Model(CeresMaizeState, [*REPLAY_PROCESSES, *CROP_PROCESSES], outputs=lambda s, p, f: s)
_STATES_V = Model(
    CeresMaizeState, [*REPLAY_PROCESSES, *CROP_PROCESSES_NSTRESS_REPLAY], outputs=lambda s, p, f: s
)
GC = DSSAT_COEFFICIENTS.grosub


def _n(s: CeresMaizeState, **kw) -> CeresMaizeState:
    """``s`` with a nitrogen record whose given factors are ``kw`` (the others the defaults)."""
    like = s.growth.lai
    vals = {k: jnp.broadcast_to(jnp.asarray(v, dtype=like.dtype), like.shape) for k, v in kw.items()}
    vals.setdefault("nstres", jnp.ones_like(like))
    return dataclasses.replace(s, n_in=CropNIn(**vals))


def _no_n(s: CeresMaizeState) -> CeresMaizeState:
    return dataclasses.replace(s, n_in=None)


# ------------------------------------------------------------------ the record and the producer
def test_record_defaults_are_no_stress() -> None:
    dt = jnp.float32
    r = CropNIn(nstres=jnp.full((2,), 0.7, dt))
    for x in (r.agefac, r.ndef3):
        assert x.dtype == dt and x.shape == (2,)
        np.testing.assert_array_equal(np.asarray(x), 1.0)
    assert r.npool.dtype == dt
    np.testing.assert_array_equal(np.asarray(r.npool), np.float32(NPOOL_NO_CAP))
    i = CropNIn.initial(3)
    for x in (i.nstres, i.agefac, i.ndef3):
        np.testing.assert_array_equal(np.asarray(x), 1.0)
    np.testing.assert_array_equal(np.asarray(i.npool), NPOOL_NO_CAP)
    # the no-cap pool never binds: NPOOL / (0.062 x 0.0095) is far above any grain number
    assert NPOOL_NO_CAP / (GC.gpp_npool_a * GC.gpp_npool_b) > 1e9
    full = CropNIn.like(i, 0.5, 0.6, 0.7, 0.2)
    np.testing.assert_allclose(
        [np.asarray(full.nstres), np.asarray(full.agefac), np.asarray(full.ndef3), np.asarray(full.npool)],
        np.repeat([[0.5], [0.6], [0.7], [0.2]], 3, axis=1),
    )
    part = CropNIn.like(i, 0.5)
    assert jax.tree_util.tree_structure(part) == jax.tree_util.tree_structure(i)
    np.testing.assert_array_equal(np.asarray(part.agefac), 1.0)
    np.testing.assert_array_equal(np.asarray(part.npool), NPOOL_NO_CAP)


def test_replay_producer_writes_every_factor() -> None:
    s = CropNReplayState.initial(2)
    f = CropNReplayForcing(
        nstres=jnp.asarray(0.7), agefac=jnp.asarray(0.6), ndef3=jnp.asarray(0.5), npool=jnp.asarray(0.3)
    )
    out = crop_n_replay(s, None, f).n_out
    for name, v in (("nstres", 0.7), ("agefac", 0.6), ("ndef3", 0.5), ("npool", 0.3)):
        x = getattr(out, name)
        assert x.dtype == s.n_out.nstres.dtype and x.shape == (2,), name
        np.testing.assert_allclose(np.asarray(x), [v, v], err_msg=name)
    # a series left out is the default: the NSTRES-only replay writes what it wrote before
    only = crop_n_replay(s, None, CropNReplayForcing(nstres=jnp.asarray(0.7))).n_out
    np.testing.assert_array_equal(np.asarray(only.agefac), 1.0)
    np.testing.assert_array_equal(np.asarray(only.ndef3), 1.0)
    np.testing.assert_array_equal(np.asarray(only.npool), NPOOL_NO_CAP)


# ------------------------------------------------------------------ kernels
def test_kernels_are_the_fortran_statements() -> None:
    a = np.linspace(0.0, 1.0, 11)
    for fslfn in (0.05, 0.0, 0.3):
        got = np.asarray(nitrogen_senescence_factor(jnp.asarray(a), jnp.asarray(fslfn)))
        np.testing.assert_allclose(got, (1.0 - fslfn) + fslfn * a, rtol=1e-12 if X64 else 1e-6)
    # AGEFAC = 1 with the species value FSLFN = 0.05: exactly 1 (the nitrogen-off senescence)
    assert float(nitrogen_senescence_factor(jnp.asarray(1.0), jnp.asarray(0.05))) == 1.0
    gpp, ndef3, npool = (
        np.asarray([600.0, 600.0, 50.0]),
        np.asarray([0.5, 1.0, 1.0]),
        np.asarray([1.0, 0.2, 1.0]),
    )
    got = np.asarray(grain_number_cap(jnp.asarray(gpp), jnp.asarray(ndef3), jnp.asarray(npool), GC))
    np.testing.assert_allclose(
        got, np.minimum(gpp * ndef3, npool / (0.062 * 0.0095)), rtol=1e-12 if X64 else 1e-6
    )


# ------------------------------------------------------------------ AGEFAC in expansion
def test_agefac_limits_expansion_in_stages_2_to_4_only() -> None:
    pre, ft, p = _pre_growth_states(71)
    # FSLFN = 0: SLFN = 1 whatever AGEFAC, so only the expansion factor sees it
    p0 = p.replace(species=p.species.replace(fslfn=jnp.zeros_like(p.species.fslfn)))
    a = 0.5
    variant = jax.jit(jax.vmap(lambda s, ff: ceres_growth_nstress_replay(_n(s, agefac=a), p0, ff)))(pre, ft)
    faithful = jax.jit(jax.vmap(lambda s, ff: ceres_growth(s, p0, ff)))(pre, ft)

    def cut_turfac(s: CeresMaizeState) -> CeresMaizeState:
        return s.replace(stress=s.stress.replace(turfac=jnp.minimum(s.stress.turfac, a)))

    limited = jax.jit(jax.vmap(lambda s, ff: ceres_growth(cut_turfac(s), p0, ff)))(pre, ft)
    istage = np.asarray(pre.phen.istage)[:, 0]
    stages = {k: np.nonzero(istage == k)[0] for k in range(1, 6)}
    assert all(len(stages[k]) > 3 for k in range(1, 6)), {k: len(v) for k, v in stages.items()}

    def day(tree, t):
        return jax.tree_util.tree_map(lambda x: x[t], tree)

    for k, days in stages.items():
        for t in days:
            got = _no_n(day(variant, t))
            # stages 2-4: min(AGEFAC, TURFAC, 1 - SATFAC) = min(min(TURFAC, AGEFAC), 1 - SATFAC)
            want = day(limited, t) if k in (2, 3, 4) else day(faithful, t)
            want = want.replace(stress=got.stress)  # the growth process does not write the stress
            assert _bytes_equal(got, want) == [], (k, t)
    # and it binds: less leaf area over stages 2-3 (in stage 2 the leaf growth is often capped at
    # 0.75 CARBO, where PLA follows the capped leaf weight and AGEFAC does not act), less ear growth
    # in stage 4
    area = lambda s: np.asarray(s.growth.leaf.area)[:, 0, 0]  # noqa: E731
    d = np.concatenate([stages[2], stages[3]])
    assert np.all(area(variant)[d] <= area(faithful)[d]) and np.any(area(variant)[d] < area(faithful)[d])
    d = stages[4]
    ear = lambda s: np.asarray(s.growth.earwt)[:, 0]  # noqa: E731
    assert np.any(ear(variant)[d] < ear(faithful)[d])


# ------------------------------------------------------------------ AGEFAC in senescence
def test_slfn_in_senescence_every_stage() -> None:
    pre, ft, p = _pre_growth_states(72)
    a = 0.2
    variant = jax.jit(jax.vmap(lambda s, ff: ceres_growth_nstress_replay(_n(s, agefac=a), p, ff)))(pre, ft)
    faithful = jax.jit(jax.vmap(lambda s, ff: ceres_growth(s, p, ff)))(pre, ft)
    istage = np.asarray(pre.phen.istage)[:, 0]
    # stage 1 and 5 days (no AGEFAC in expansion), crop past emergence (day-start PLA > 0)
    pla0 = np.asarray(pre.growth.leaf.area)[:, 0, 0]
    grows = np.asarray(pre.phen.mdate)[:, 0] != np.asarray(ft.yrdoy)  # not the maturity day
    days = np.nonzero(((istage == 1) | (istage == 5)) & (pla0 > 0.0) & grows)[0]
    assert len(days) > 20
    g = variant.growth
    pla = np.asarray(g.leaf.area)[:, 0, 0][days]
    slan = np.asarray(g.slan)[:, 0][days]
    senla0 = np.asarray(pre.growth.senla)[:, 0][days]
    lai0 = np.asarray(pre.growth.lai)[:, 0][days]
    swfac = np.asarray(pre.stress.swfac)[:, 0][days]
    tmin = np.asarray(ft.tmin)[days]
    fslfw, fslfn = float(p.species.fslfw), float(p.species.fslfn)
    slfw = (1.0 - fslfw) + fslfw * swfac
    slfn = (1.0 - fslfn) + fslfn * a
    slfc = np.where(lai0 > 4.0, 1.0 - 0.008 * (lai0 - 4.0), 1.0)
    slft = np.where(tmin <= 6.0, np.maximum(0.0, 1.0 - 0.01 * (tmin - 6.0) ** 2), 1.0)
    fac = np.minimum.reduce([slfw, slfc, slft, np.full_like(slfw, slfn)])
    senla = np.minimum(np.maximum(senla0 + (pla - senla0) * (1.0 - fac), slan), pla)
    np.testing.assert_allclose(np.asarray(g.senla)[:, 0][days], senla, rtol=1e-12 if X64 else 1e-5, atol=1e-9)
    # it binds: more senescence than the nitrogen-off day where SLFN is the smallest factor
    more = np.asarray(g.senla)[:, 0][days] > np.asarray(faithful.growth.senla)[:, 0][days]
    assert np.any(more) and np.all(
        np.asarray(g.senla)[:, 0][days] >= np.asarray(faithful.growth.senla)[:, 0][days]
    )


# ------------------------------------------------------------------ NDEF3 / NPOOL grain-number cap
def test_grain_number_cap_on_the_first_effective_grain_fill_day() -> None:
    f, p = _season(73)
    s0 = CeresMaizeState.initial(p, 1)
    go_f = jax.jit(lambda pp, ff, s: run(_STATES, pp, ff, s))
    go_v = jax.jit(lambda pp, ff, s: run(_STATES_V, pp, ff, s))
    faithful = go_f(p, f, s0)
    yrdoy = np.asarray(f.yrdoy)
    stg4 = np.asarray(faithful.phen.stgdoy)[:, 0, 3]
    t0 = int(np.argmax(stg4 == yrdoy))
    assert stg4[t0] == yrdoy[t0] and int(np.asarray(faithful.phen.istage)[t0, 0]) == 5
    assert float(np.asarray(faithful.growth.grogrn)[t0, 0]) != 0.0
    gpp_f = np.asarray(faithful.phen.gpp)[:, 0]
    assert gpp_f[t0] > 0.0 and np.all(gpp_f[t0:] == gpp_f[t0])
    npool_small = 0.1
    for kw, want in (
        ({"ndef3": 0.6}, np.asarray(gpp_f[t0], dtype=gpp_f.dtype) * np.asarray(0.6, dtype=gpp_f.dtype)),
        ({"npool": npool_small}, min(gpp_f[t0], npool_small / (0.062 * 0.0095))),
    ):
        v = go_v(p, f, _n(s0, **kw))
        # before the first effective grain-fill day: the faithful crop, bit for bit
        before = jax.tree_util.tree_map(lambda x, t0=t0: x[:t0], _no_n(v))
        assert _bytes_equal(before, jax.tree_util.tree_map(lambda x, t0=t0: x[:t0], faithful)) == [], kw
        gpp_v = np.asarray(v.phen.gpp)[:, 0]
        # from it on the capped grain number, set once
        np.testing.assert_allclose(gpp_v[t0:], want, rtol=1e-12 if X64 else 1e-6, err_msg=str(kw))
        assert want < gpp_f[t0]
        assert float(np.asarray(v.growth.grnwt)[-1, 0]) < float(np.asarray(faithful.growth.grnwt)[-1, 0])
