"""Potential ET: ASCE reference ET and Priestley-Taylor (DSSAT).

Priestley-Taylor against a hand computation, ASCE against ``pyet``, vmap and gradients. The
comparison with the RZWQM2 ``.ana`` output of the CA-TPA reference run is in
``tests/integration/test_pet_oracle.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.processes.pet import asce_reference_et, priestley_taylor, wind_to_2m

# a mid-latitude site: elevation m, latitude rad
SITE_ELEV = 200.0
SITE_LAT = 0.745163


# --------------------------------------------------------------------------------------
# Priestley-Taylor (DSSAT PETPT) against a hand computation
# --------------------------------------------------------------------------------------
def test_priestley_taylor_hand_computation(tol: float) -> None:
    srad, tmax, tmin, lai, msalb = 20.0, 30.0, 15.0, 2.0, 0.13
    td = 0.6 * 30.0 + 0.4 * 15.0  # 24.0
    albedo = 0.23 - (0.23 - 0.13) * np.exp(-0.75 * 2.0)
    slang = 20.0 * 23.923
    eeq = slang * (2.04e-4 - 1.83e-4 * albedo) * (td + 29.0)
    assert float(priestley_taylor(srad, tmax, tmin, lai, msalb)) == pytest.approx(1.1 * eeq, rel=tol)
    # hot day: factor (Tmax - 35) 0.05 + 1.1
    eeq_hot = slang * (2.04e-4 - 1.83e-4 * albedo) * (0.6 * 38.0 + 0.4 * 15.0 + 29.0)
    assert float(priestley_taylor(srad, 38.0, tmin, lai, msalb)) == pytest.approx(
        eeq_hot * (3.0 * 0.05 + 1.1), rel=tol
    )
    # cold day: 0.01 exp(0.18 (Tmax + 20))
    eeq_cold = slang * (2.04e-4 - 1.83e-4 * albedo) * (0.6 * 2.0 + 0.4 * (-5.0) + 29.0)
    assert float(priestley_taylor(srad, 2.0, -5.0, lai, msalb)) == pytest.approx(
        eeq_cold * 0.01 * np.exp(0.18 * 22.0), rel=tol
    )
    # bare soil uses the soil albedo
    eeq_bare = slang * (2.04e-4 - 1.83e-4 * 0.13) * (td + 29.0)
    assert float(priestley_taylor(srad, tmax, tmin, 0.0, msalb)) == pytest.approx(1.1 * eeq_bare, rel=tol)
    assert float(priestley_taylor(0.0, tmax, tmin, lai, msalb)) == pytest.approx(1e-4)


# --------------------------------------------------------------------------------------
# ASCE reference ET against pyet
# --------------------------------------------------------------------------------------
def test_asce_reference_et_matches_pyet() -> None:
    pyet = pytest.importorskip("pyet")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    n = 30
    tmin = rng.uniform(-5.0, 20.0, n)
    tmax = tmin + rng.uniform(3.0, 15.0, n)
    srad = rng.uniform(2.0, 30.0, n)
    rh = rng.uniform(30.0, 95.0, n)
    wind = rng.uniform(0.3, 6.0, n)
    doy = rng.integers(1, 366, n)
    idx = pd.to_datetime([f"2015-{int(d):03d}" for d in doy], format="%Y-%j")
    u2 = np.asarray(wind_to_2m(wind, 2.0))
    kw = dict(
        rs=pd.Series(srad, idx),
        tmax=pd.Series(tmax, idx),
        tmin=pd.Series(tmin, idx),
        rh=pd.Series(rh, idx),
        elevation=SITE_ELEV,
        lat=SITE_LAT,
        clip_zero=False,
    )
    tmean = pd.Series(0.5 * (tmin + tmax), idx)
    ref_short = pyet.pm_asce(tmean, pd.Series(u2, idx), etype="os", **kw).values
    ref_tall = pyet.pm_asce(tmean, pd.Series(u2, idx), etype="rs", **kw).values
    mine = asce_reference_et(tmin, tmax, srad, rh, wind, elevation=SITE_ELEV, latitude=SITE_LAT, doy=doy)
    assert np.abs(np.asarray(mine.et_short) - ref_short).max() < 1e-3
    assert np.abs(np.asarray(mine.et_tall) - ref_tall).max() < 1e-3
    assert bool(jnp.all(mine.et_tall > mine.et_short))


def test_asce_reference_et_vmap_and_grad() -> None:
    def f(w):
        return asce_reference_et(10.0, 25.0, 20.0, 50.0, w, elevation=200.0, latitude=0.7, doy=180).et_short

    g = jax.grad(f)(2.0)
    assert np.isfinite(float(g)) and float(g) > 0.0  # more wind, more ET
    batch = jax.vmap(f)(jnp.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.asarray(batch), [float(f(w)) for w in (1.0, 2.0, 3.0)])
