"""DSSAT-CSM SPAM partition (``pet/spam_pse``, ``pet/spam_trans``) against a scalar transcription
of the Fortran (``SPAM/PET.for`` PSE, ``SPAM/TRANS.for`` TRANS / TRATIO, ``Weather/HMET.for``
VPSAT / VPSLOP, BSD-3) and the port conventions (P5 in cm d-1, EO in mm d-1).
The reference comparison is ``tests/integration/test_spam_evap_dssat.py``."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.iface.crop import CanopyRecord
from agrijax.iface.surface import PETFluxes
from agrijax.processes.pet.spam_dssat import (
    SPAM_COEFFICIENTS,
    SpamParams,
    SpamPSEState,
    SpamTransState,
    SpamWeather,
    potential_soil_evaporation,
    potential_transpiration,
    spam_potential_soil_evaporation,
    spam_potential_transpiration,
    transpiration_ratio,
)

pytestmark = [
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="float64 transcription comparison"),
    pytest.mark.allow_skip(reason="float64 comparison, skipped in the float32 tier"),
]


def f_pse(eo, ksevap, xlai):
    if ksevap <= 0.0:
        eos = eo * (1.0 - 0.39 * xlai) if xlai <= 1.0 else eo / 1.1 * math.exp(-0.4 * xlai)
    else:
        eos = eo * math.exp(-ksevap * xlai)
    return max(eos, 0.0)


def f_vpsat(t):
    return 610.78 * math.exp(17.269 * t / (t + 237.30))


def f_vpslop(t):
    return 18.0 * (2501.0 - 2.373 * t) * f_vpsat(t) / (8.314 * (t + 273.0) ** 2)


def f_tratio(c4, co2, tavg, windsp, xhlai):
    if xhlai < 0.01:
        return 1.0
    uavg = windsp / 86.4
    rb = 10.0
    if c4:
        rlf = (1.0 / (0.0328 - 5.49e-5 * 330.0 + 2.96e-8 * 330.0**2)) + rb
        rlfc = (1.0 / (0.0328 - 5.49e-5 * co2 + 2.96e-8 * co2**2)) + rb
    else:
        rlf = 9.72 + 0.0757 * 330.0 + 10.0
        rlfc = 9.72 + 0.0757 * co2 + 10.0
    rl = rlf / (0.5 * 2.88)
    rlc = rlfc / (0.5 * 2.88)
    ra = 208.0 / uavg
    delta = f_vpslop(tavg) / 100.0
    lhv = 2500.9 - 2.345 * tavg
    gamma = 1013.0 * 1.005 / (lhv * 0.622)
    return (delta + gamma * (1.0 + rl / ra)) / (delta + gamma * (1.0 + rlc / ra))


def f_trans(eo, xhlai, ktrans, trat, evap):
    if not xhlai > 1e-6:
        return 0.0
    fdint = 1.0 - math.exp(-ktrans * xhlai)
    eop = eo * fdint
    eop_reduc = eop * (1.0 - trat)
    eop = eop * trat
    eop = min(eop, eo - eop_reduc - evap)
    return max(eop, 0.0)


def test_pse_both_forms() -> None:
    rng = np.random.default_rng(0)
    n = 500
    eo = rng.uniform(0.0, 9.0, n)
    xlai = np.where(rng.random(n) < 0.1, 0.0, rng.uniform(0.0, 6.0, n))
    ks = np.where(rng.random(n) < 0.3, -99.0, rng.uniform(0.4, 1.0, n))
    got = np.asarray(potential_soil_evaporation(eo, xlai, ks))
    ref = np.array([f_pse(eo[i], ks[i], xlai[i]) for i in range(n)])
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-15)
    assert ((ks <= 0) & (xlai <= 1)).any() and ((ks <= 0) & (xlai > 1)).any()


@pytest.mark.parametrize("c4", [True, False])
def test_tratio_and_trans(c4: bool) -> None:
    rng = np.random.default_rng(1)
    n = 400
    co2 = rng.uniform(300.0, 800.0, n)
    tavg = rng.uniform(-5.0, 35.0, n)
    wind = rng.uniform(5.0, 400.0, n)
    xhlai = np.where(rng.random(n) < 0.1, rng.uniform(0.0, 0.02, n), rng.uniform(0.0, 6.0, n))
    trat = np.asarray(transpiration_ratio(co2, tavg, wind, xhlai, c4=c4))
    ref = np.array([f_tratio(c4, co2[i], tavg[i], wind[i], xhlai[i]) for i in range(n)])
    np.testing.assert_allclose(trat, ref, rtol=1e-13)
    # TRATIO = 1 at the reference CO2
    one = np.asarray(transpiration_ratio(330.0, 20.0, 150.0, 3.0, c4=c4))
    assert float(one) == pytest.approx(1.0, abs=1e-15)
    eo = rng.uniform(0.0, 9.0, n)
    evap = eo * rng.uniform(0.0, 1.0, n)
    kt = np.full(n, 0.6854839)
    eop = np.asarray(potential_transpiration(eo, xhlai, kt, trat, evap))
    ref = np.array([f_trans(eo[i], xhlai[i], kt[i], trat[i], evap[i]) for i in range(n)])
    np.testing.assert_allclose(eop, ref, rtol=1e-13, atol=1e-15)
    capped = eo * (1 - np.exp(-kt * xhlai)) * trat > eo - eo * (1 - np.exp(-kt * xhlai)) * (1 - trat) - evap
    assert capped.any() and (~capped).any()


def _canopy(lai: float) -> CanopyRecord:
    v = jnp.asarray([lai])
    return CanopyRecord(lai=v, tlai=v, height=v)


def test_processes_write_p5_in_cm_from_eo_in_mm() -> None:
    z = jnp.asarray(0.0)
    pet = PETFluxes(z, z, z, z, z, jnp.asarray(5.0))
    params = SpamParams(ksevap=jnp.asarray(0.7), ktrans=jnp.asarray(0.7))
    out = spam_potential_soil_evaporation(SpamPSEState(canopy=_canopy(2.0), pet=pet), params, None)
    assert float(out.pet.soil_evaporation) == pytest.approx(f_pse(5.0, 0.7, 2.0) / 10.0, rel=1e-14)
    assert float(out.pet.transpiration) == 0.0
    wx = SpamWeather(tavg=jnp.asarray(22.0), wind_run=jnp.asarray(150.0), co2=jnp.asarray(400.0))
    st = SpamTransState(canopy=_canopy(2.0), pet=out.pet, evaporation=jnp.asarray(1.0))
    tr = spam_potential_transpiration(st, params, wx)
    ref = f_trans(5.0, 2.0, 0.7, f_tratio(True, 400.0, 22.0, 150.0, 2.0), 1.0)
    assert float(tr.pet.transpiration) == pytest.approx(ref / 10.0, rel=1e-13)
    assert float(tr.pet.soil_evaporation) == float(out.pet.soil_evaporation)


def test_gradients_finite_at_zero_lai_and_zero_wind() -> None:
    c = SPAM_COEFFICIENTS.as_arrays(jnp.float64)

    def loss(cc, lai, wind):
        eos = potential_soil_evaporation(4.0, lai, 0.7, cc.pse)
        trat = transpiration_ratio(400.0, 20.0, wind, lai, c4=True, c=cc.trans)
        return eos + potential_transpiration(4.0, lai, 0.7, trat, 0.5, cc.trans)

    for lai, wind in ((0.0, 100.0), (3.0, 0.0), (1e-7, 1.0)):
        g = jax.grad(loss, argnums=(0, 1, 2))(c, jnp.asarray(lai), jnp.asarray(wind))
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g)), (lai, wind)
