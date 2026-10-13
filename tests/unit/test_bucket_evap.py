"""DSSAT-CSM soil and mulch evaporation kernels (``bucket_evap``) against a line-by-line scalar
transcription of the Fortran (``SPAM/SOILEV.for``, ``SPAM/ESR_SoilEvap.for``,
``Soil/Mulch/MULCHEVAP.for``, ``SPAM/SPAM.for``), on random states that reach every branch.

The transcriptions below are a second implementation (plain Python ``if`` / ``else``, one
statement per Fortran statement, BSD-3 source); the vectorised kernels must equal them to float64
rounding. The reference comparison against the instrumented dscsm048 is
``tests/integration/test_spam_evap_dssat.py``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.iface.surface import PETFluxes
from agrijax.processes.soil_water.bucket_evap import (
    EVAP_COEFFICIENTS,
    SoilEvapParams,
    SoilEvapState,
    SoilevStore,
    esr_soil_evaporation,
    mulch_evaporation,
    soil_evaporation_esr,
    soil_evaporation_mulch,
    soil_evaporation_soilev,
    soilev_init,
    soilev_rate,
    spam_mulch_step,
)

pytestmark = [
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 transcription comparison"),
    pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier"),
]


# --------------------------------------------------------------------------- Fortran transcriptions
def f_esup(eos, sumes1, sumes2, u, es, t):
    sumes1 = sumes1 + eos
    if sumes1 > u:
        es = eos - 0.4 * (sumes1 - u)
        sumes2 = 0.6 * (sumes1 - u)
        t = (sumes2 / 3.5) ** 2
        sumes1 = u
    else:
        es = eos
    return sumes1, sumes2, es, t


def f_soilev_init(sw1, ll1, dul1, dlayr1, u):
    swr = max(0.0, (sw1 - ll1) / (dul1 - ll1))
    usoil = (dul1 - sw1) * dlayr1 * 10.0
    if swr >= 1.0:
        sumes1 = sumes2 = t = 0.0
    elif usoil <= u:
        sumes2 = 0.0
        t = 0.0
        sumes1 = usoil
    else:
        sumes2 = usoil - u
        sumes1 = u
        t = (sumes2 / 3.5) ** 2
    swef = 0.9 - 0.00038 * (dlayr1 - 30.0) ** 2
    return sumes1, sumes2, t, swef


def f_soilev_rate(sumes1, sumes2, t, swef, eos, winf, sw1, ll1, dlayr1, sw_avail, u, pmfraction):
    es = 0.0
    if sumes1 >= u and winf >= sumes2:
        winfmod = winf - sumes2
        sumes1 = u - winfmod
        sumes2 = 0.0
        t = 0.0
        if winfmod > u:
            sumes1 = 0.0
        sumes1, sumes2, es, t = f_esup(eos, sumes1, sumes2, u, es, t)
    elif sumes1 >= u and winf < sumes2:
        t = t + 1.0
        es = 3.5 * t**0.5 - sumes2
        if winf > 0.0:
            esx = 0.8 * winf
            if esx <= es:
                esx = es + winf
            if esx > eos:
                esx = eos
            es = esx
        elif es > eos:
            es = eos
        sumes2 = sumes2 + es - winf
        t = (sumes2 / 3.5) ** 2
    elif winf >= sumes1:
        sumes1 = 0.0
        sumes1, sumes2, es, t = f_esup(eos, sumes1, sumes2, u, es, t)
    else:
        sumes1 = sumes1 - winf
        sumes1, sumes2, es, t = f_esup(eos, sumes1, sumes2, u, es, t)
    awev1 = max(0.0, (sw1 - ll1 * swef) * dlayr1 * 10.0)
    if awev1 < es:
        if sumes1 >= u and sumes2 > es:
            sumes2 = sumes2 - es + awev1
            t = (sumes2 / 3.5) ** 2
            es = awev1
        elif sumes1 >= u and sumes2 < es and sumes2 > 0:
            sumes1 = sumes1 - (es - sumes2)
            sumes2 = max(sumes1 + awev1 - u, 0.0)
            sumes1 = min(sumes1 + awev1, u)
            t = (sumes2 / 3.5) ** 2
            es = awev1
        else:
            sumes1 = sumes1 - es + awev1
            es = awev1
    if pmfraction > 1e-6:
        es = es * (1.0 - pmfraction)
    swmin = max(0.0, sw_avail - swef * ll1)
    if es > swmin * dlayr1 * 10.0:
        es = swmin * dlayr1 * 10.0
    es = max(es, 0.0)
    return es, sumes1, sumes2, t


def f_esr(eos, sw, swdelts, dlayr, ds, dul, ll, pmfraction):
    n = len(sw)
    swad = [0.30 * ll[i] for i in range(n)]
    meandep = [ds[i] - dlayr[i] / 2.0 for i in range(n)]
    swtemp = [sw[i] + 0.5 * swdelts[i] if swdelts[i] > 0.0 else sw[i] + swdelts[i] for i in range(n)]
    ptype = 3
    for i in range(n):
        if meandep[i] < 100.0 and swtemp[i] > dul[i]:
            ptype = 1
    if ptype == 1:
        thr = 0.275 * dul[0] + 1.165 * dul[0] * dul[0] + (1.2 * dul[0] ** 3.75) * meandep[0]
        if swtemp[0] < thr:
            ptype = 2
    es = 0.0
    swdeltu, es_lyr = [0.0] * n, [0.0] * n
    for i in range(n):
        if ptype == 3:
            a = 0.5 + 0.24 * dul[i]
            b = -2.04 + 0.20 * dul[i]
            coef = a * meandep[i] ** b
        elif ptype == 2:
            coef = 0.011
        else:
            coef = 0.26 * meandep[i] ** -0.70
        swdeltu[i] = -(swtemp[i] - swad[i]) * coef
        if pmfraction > 1e-6:
            swdeltu[i] = swdeltu[i] * (1.0 - pmfraction)
        sw_avail = sw[i] + swdelts[i] - swad[i]
        if -swdeltu[i] > sw_avail:
            swdeltu[i] = -sw_avail
        swdeltu[i] = min(0.0, swdeltu[i])
        es_lyr[i] = -swdeltu[i] * dlayr[i] * 10.0
        es = es + es_lyr[i]
    if es > eos:
        red = eos / es
        es_lyr = [x * red for x in es_lyr]
        swdeltu = [x * red for x in swdeltu]
        es = eos
    upflow = [0.0] * n
    upflow[n - 1] = es_lyr[n - 1] / 10.0
    for i in range(n - 2, -1, -1):
        upflow[i] = upflow[i + 1] + es_lyr[i] / 10.0
    return es, es_lyr, swdeltu, upflow, ptype


def f_mulch(eos, mass, cover, wat, am, extfac):
    if mass > 0.1:
        mai = (am * 1.0e-5 * mass) / cover if cover > 1e-6 else 0.0
        eom = eos * (1.0 - math.exp(-extfac * mai))
        em2 = min(eom, wat * 0.85)
        em2 = max(em2, 0.0) * cover
        eos2 = eos - em2
        eos3 = eos * math.exp(-extfac * mai) * cover + eos * (1.0 - cover)
        return em2, min(eos2, eos3)
    return 0.0, eos


# --------------------------------------------------------------------------- random states
def _states(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(2.0, 12.0, n)
    kind = rng.integers(0, 4, n)
    s1 = np.where(kind == 0, rng.uniform(0.0, 1.0, n) * u, u)  # 0: stage 1
    s2 = np.where(kind == 0, 0.0, rng.uniform(0.0, 20.0, n))
    s2 = np.where(kind == 3, rng.uniform(0.0, 0.5, n), s2)  # small SUMES2: the AWEV1 branch b
    t = (s2 / 3.5) ** 2
    winf = np.where(rng.random(n) < 0.5, 0.0, rng.uniform(0.0, 25.0, n))
    eos = rng.uniform(0.0, 9.0, n)
    ll1 = rng.uniform(0.05, 0.25, n)
    dlayr1 = rng.choice([5.0, 10.0, 15.0, 20.0], n)
    sw1 = ll1 * rng.uniform(0.2, 2.5, n)  # below the air-dry limit up to above DUL
    sw_avail = np.maximum(0.0, sw1 + rng.uniform(-0.05, 0.05, n))
    swef = 0.9 - 0.00038 * (dlayr1 - 30.0) ** 2
    pm = np.where(rng.random(n) < 0.2, rng.uniform(0.1, 0.9, n), 0.0)
    return dict(
        s1=s1,
        s2=s2,
        t=t,
        swef=swef,
        eos=eos,
        winf=winf,
        sw1=sw1,
        ll1=ll1,
        dlayr1=dlayr1,
        sw_avail=sw_avail,
        u=u,
        pm=pm,
    )


def test_soilev_rate_equals_the_fortran_transcription_on_every_branch() -> None:
    d = _states(4000)
    es, new = soilev_rate(
        SoilevStore(d["s1"], d["s2"], d["t"], d["swef"]),
        d["eos"],
        d["winf"],
        d["sw1"],
        d["ll1"],
        d["dlayr1"],
        d["sw_avail"],
        d["u"],
        d["pm"],
    )
    ref = np.array(
        [
            f_soilev_rate(
                *(
                    d[k][i]
                    for k in (
                        "s1",
                        "s2",
                        "t",
                        "swef",
                        "eos",
                        "winf",
                        "sw1",
                        "ll1",
                        "dlayr1",
                        "sw_avail",
                        "u",
                        "pm",
                    )
                )
            )
            for i in range(4000)
        ]
    )
    np.testing.assert_allclose(np.asarray(es), ref[:, 0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(new.sumes1), ref[:, 1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(new.sumes2), ref[:, 2], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(new.t), ref[:, 3], rtol=1e-12, atol=1e-12)
    # the random states reach every branch: stages 1 / 2, rain refills, the AWEV1 and SWMIN limits
    st2 = d["s1"] >= d["u"]
    assert (st2 & (d["winf"] >= d["s2"])).any() and (st2 & (d["winf"] < d["s2"]) & (d["winf"] > 0)).any()
    assert (~st2 & (d["winf"] >= d["s1"])).any() and (~st2 & (d["winf"] < d["s1"])).any()
    awev1 = np.maximum(0.0, (d["sw1"] - d["ll1"] * d["swef"]) * d["dlayr1"] * 10)
    assert (np.asarray(es) == awev1).sum() > 100 and (np.asarray(es) == 0).sum() > 100


def test_soilev_init_equals_the_fortran_transcription() -> None:
    rng = np.random.default_rng(1)
    n = 500
    ll = rng.uniform(0.05, 0.2, n)
    dul = ll + rng.uniform(0.05, 0.2, n)
    sw = ll + rng.uniform(-0.05, 1.3, n) * (dul - ll)
    dl = rng.choice([5.0, 10.0, 15.0], n)
    u = rng.uniform(2.0, 12.0, n)
    st = soilev_init(sw, ll, dul, dl, u)
    ref = np.array([f_soilev_init(sw[i], ll[i], dul[i], dl[i], u[i]) for i in range(n)])
    for k, v in enumerate((st.sumes1, st.sumes2, st.t, st.swef)):
        np.testing.assert_allclose(np.asarray(v), ref[:, k], rtol=1e-12, atol=1e-12)
    s1, s2 = np.asarray(st.sumes1), np.asarray(st.sumes2)
    assert (s2 > 0).any() and ((s2 == 0) & (s1 > 0)).any() and ((s1 == 0) & (s2 == 0)).any()  # 3 branches


def test_esr_equals_the_fortran_transcription_on_every_profile_type() -> None:
    rng = np.random.default_rng(2)
    n, nl = 600, 6
    dlayr = np.tile([5.0, 10.0, 15.0, 30.0, 30.0, 30.0], (n, 1))
    ds = np.cumsum(dlayr, axis=1)
    ll = rng.uniform(0.05, 0.2, (n, nl))
    dul = ll + rng.uniform(0.08, 0.2, (n, nl))
    wetness = rng.choice([0.5, 1.05, 1.2], (n, 1))
    sw = ll + wetness * (dul - ll) + rng.uniform(-0.02, 0.02, (n, nl))
    sw[: n // 3, 0] = ll[: n // 3, 0] * 0.9  # top layer dry: the intermediate profile
    swdelts = np.where(rng.random((n, nl)) < 0.5, rng.uniform(-0.01, 0.01, (n, nl)), 0.0)
    eos = rng.uniform(0.0, 8.0, n)
    pm = np.where(rng.random(n) < 0.2, 0.5, 0.0)
    r = esr_soil_evaporation(eos, sw, swdelts, dlayr, ds, dul, ll, pm)
    types = set()
    for i in range(n):
        es, lyr, du, up, pt = f_esr(eos[i], sw[i], swdelts[i], dlayr[i], ds[i], dul[i], ll[i], pm[i])
        types.add(pt)
        assert int(r.profile[i]) == pt
        np.testing.assert_allclose(float(r.es[i]), es, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(r.es_lyr[i]), lyr, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(r.swdeltu[i]), du, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(np.asarray(r.upflow[i]), up, rtol=1e-12, atol=1e-15)
    assert types == {1, 2, 3}
    assert (np.asarray(r.es) == eos).sum() > 50 and (np.asarray(r.es) < eos).sum() > 50


def test_mulch_equals_the_fortran_transcription_and_spam_subtracts_em_again() -> None:
    rng = np.random.default_rng(3)
    n = 400
    eos = rng.uniform(0.0, 6.0, n)
    mass = np.where(rng.random(n) < 0.2, 0.05, rng.uniform(0.0, 6000.0, n))
    cover = np.where(rng.random(n) < 0.1, 0.0, rng.uniform(0.0, 1.0, n))
    wat = rng.uniform(0.0, 3.0, n)
    am, ext = np.full(n, 4.0), np.full(n, 0.8)
    r = mulch_evaporation(eos, mass, cover, wat, am, ext)
    ref = np.array([f_mulch(eos[i], mass[i], cover[i], wat[i], am[i], ext[i]) for i in range(n)])
    np.testing.assert_allclose(np.asarray(r.em), ref[:, 0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(r.eos), ref[:, 1], rtol=1e-12, atol=1e-14)
    s = spam_mulch_step(eos, mass, cover, wat, am, ext, mulch_active=True)
    left = np.where(ref[:, 1] > ref[:, 0], ref[:, 1] - ref[:, 0], 0.0)
    called = eos > 1e-6
    np.testing.assert_allclose(np.asarray(s.eos), np.where(called, left, eos), rtol=1e-12, atol=1e-14)
    off = spam_mulch_step(eos, mass, cover, wat, am, ext, mulch_active=False)
    np.testing.assert_array_equal(np.asarray(off.eos), eos)
    assert np.all(np.asarray(off.em) == 0)


def _state(dtype=jnp.float64, **kw) -> tuple[SoilEvapState, SoilEvapParams]:
    dl = jnp.asarray([5.0, 10.0, 15.0], dtype)
    ll = jnp.asarray([0.1, 0.1, 0.12], dtype)
    dul = jnp.asarray([0.22, 0.23, 0.24], dtype)
    sw = jnp.asarray([0.15, 0.18, 0.2], dtype)
    s = SoilEvapState.initial(
        sw, dl, jnp.cumsum(dl), dul, ll, jnp.asarray(8.0, dtype), pet=PETFluxes.zeros(dtype)
    )
    import equinox as eqx

    for k, v in kw.items():
        s = eqx.tree_at(lambda x, k=k: getattr(x, k), s, jnp.asarray(v, dtype))
    return s, SoilEvapParams(u=jnp.asarray(8.0, dtype), pmfraction=jnp.asarray(0.0, dtype))


def test_processes_gate_on_eos_soil_and_write_evap() -> None:
    s, p = _state(eos_soil=0.0, em=0.2, sumes1=8.0, sumes2=4.0, t=(4.0 / 3.5) ** 2)
    out = soil_evaporation_soilev(s, p, None)
    assert float(out.es) == 0.0 and float(out.evap) == 0.2 and float(out.sumes2) == 4.0  # not called
    s2, _ = _state(eos_soil=3.0, em=0.2, sumes1=8.0, sumes2=4.0, t=(4.0 / 3.5) ** 2)
    out = soil_evaporation_soilev(s2, p, None)
    assert float(out.es) > 0.0 and float(out.evap) == pytest.approx(float(out.es) + 0.2)
    swd = jnp.asarray([-0.001, -0.002, 0.0])
    s3, _ = _state(eos_soil=0.0, swdeltu=swd)
    out = soil_evaporation_esr(s3, p, None)
    np.testing.assert_array_equal(np.asarray(out.swdeltu), np.asarray(swd))  # not called: unchanged
    s4, _ = _state(eos_soil=2.0)
    out = soil_evaporation_esr(s4, p, None)
    np.testing.assert_allclose(float(out.upflow[0]) * 10.0, float(out.es), rtol=1e-12)
    np.testing.assert_allclose(float(jnp.sum(out.es_lyr)), float(out.es), rtol=1e-12)


def test_mulch_process_reads_eos_from_the_pet_port_in_cm() -> None:
    import equinox as eqx

    s, p = _state(mulch_mass=3000.0, mulch_cover=0.6, mulch_water=1.0, mulch_am=4.0, mulch_extfac=0.8)
    s = eqx.tree_at(lambda x: x.pet.soil_evaporation, s, jnp.asarray(0.4))
    out = soil_evaporation_mulch(s, p, None)
    ref = spam_mulch_step(4.0, 3000.0, 0.6, 1.0, 4.0, 0.8, mulch_active=True)
    assert float(out.em) == pytest.approx(float(ref.em)) and float(out.eos_soil) == pytest.approx(
        float(ref.eos)
    )
    assert float(out.em) > 0.0


def test_gradients_are_finite_at_the_edges() -> None:
    c = EVAP_COEFFICIENTS.as_arrays(jnp.float64)

    def loss(cc, eos):
        es, st = soilev_rate(
            SoilevStore(8.0, 0.0, 0.0, 0.84), eos, 0.0, 0.1, 0.1, 5.0, 0.1, 8.0, 0.0, cc.soilev
        )
        r = esr_soil_evaporation(
            eos,
            jnp.asarray([0.1, 0.2]),
            jnp.zeros(2),
            jnp.asarray([5.0, 10.0]),
            jnp.asarray([5.0, 15.0]),
            jnp.asarray([0.2, 0.2]),
            jnp.asarray([0.1, 0.1]),
            0.0,
            cc.esr,
        )
        m = mulch_evaporation(eos, 0.0, 0.0, 0.0, 4.0, 0.8, cc.mulch)
        return es + st.t + r.es + m.em + m.eos

    for eos in (0.0, 1e-9, 3.0):
        g = jax.grad(loss)(c, jnp.asarray(eos))
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g)), eos
