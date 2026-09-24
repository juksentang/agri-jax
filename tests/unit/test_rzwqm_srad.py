"""``rzwqm_daily_srad``: the daily radiation RZWQM2 re-sums from its hourly disaggregation.

Data-free checks against independent references (the data-backed comparison with the binary's
``.ana`` column 88 on 15 scenarios is ``tests/integration/test_rzwqm_all_scenarios.py``):

* the daily normalisation integral ``ISINB`` of Spitters' eq. 6 (a closed form in DSSAT ``SOLAR``)
  equals ``scipy.integrate.quad`` of ``sinB (1 + 0.4 sinB)`` over the day;
* the whole-hour sum is a rectangle rule of a curve whose integral is the daily value, so it stays
  within 1.5 % of it at every day for |latitude| <= 45 deg (5 % at 60 deg);
* the result is linear in the daily value;
* the slope branch (SHAW direct/diffuse partition) tends to the flat branch as the slope -> 0 and
  has the right sign: a south-facing slope gets more winter radiation in the northern hemisphere,
  a north-facing one less.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from agri_jax.io.rzwqm.met import _hourly_radiation_dssat40, rzwqm_daily_srad

PI = 3.14159  # DSSAT's PARAMETER value
DOYS = np.arange(1, 366)


def _sun(doy: int, lat_deg: float) -> tuple[float, float, float, float]:
    rad = PI / 180.0
    dec = -23.45 * math.cos(2.0 * PI * (doy + 10.0) / 365.0)
    soc = max(-1.0, min(1.0, math.tan(rad * dec) * math.tan(rad * lat_deg)))
    dayl = 12.0 + 24.0 * math.asin(soc) / PI
    ssin = math.sin(rad * dec) * math.sin(rad * lat_deg)
    ccos = math.cos(rad * dec) * math.cos(rad * lat_deg)
    return dayl, ssin, ccos, rad


@pytest.mark.parametrize("lat_deg", [-45.0, 0.0, 23.0, 42.7, 55.0])
@pytest.mark.parametrize("doy", [15, 80, 172, 266, 355])
def test_isinb_closed_form_equals_quadrature(lat_deg: float, doy: int) -> None:
    dayl, ssin, ccos, rad = _sun(doy, lat_deg)
    t0, t1 = 12.0 - dayl / 2.0, 12.0 + dayl / 2.0

    def shape(t_h: float) -> float:
        sinb = max(ssin + ccos * math.cos((t_h - 12.0) * PI / 12.0), 0.0)
        return sinb * (1.0 + 0.4 * sinb)

    integral_s = 3600.0 * quad(shape, t0, t1, epsabs=1e-12, epsrel=1e-12, limit=200)[0]
    srad = 20.0
    hourly = _hourly_radiation_dssat40(np.array([srad]), np.array([doy]), np.float32(lat_deg))[0]
    for h in range(1, 25):
        sinb_true = ssin + ccos * math.cos((h - 12.0) * PI / 12.0)
        if hourly[h - 1] > 0 and math.asin(sinb_true) / rad > 1.0:  # HANG raises beta to >= 1 deg
            implied_isinb = sinb_true * (1.0 + 0.4 * sinb_true) * srad * 1e6 / float(hourly[h - 1])
            assert implied_isinb == pytest.approx(integral_s, rel=2e-5), (h, lat_deg, doy)


@pytest.mark.parametrize("lat_deg", [-60.0, -30.0, 0.0, 30.0, 42.7, 60.0])
def test_hourly_resum_close_to_daily_value(lat_deg: float) -> None:
    srad = np.full(DOYS.shape, 18.0)
    out = rzwqm_daily_srad(srad, DOYS, math.radians(lat_deg))
    ratio = out / srad
    assert np.all(np.isfinite(ratio))
    # short polar-ish winter days are sampled by few whole hours, hence the looser bound at 60 deg
    assert np.abs(ratio - 1.0).max() < (1.5e-2 if abs(lat_deg) <= 45.0 else 5e-2)
    # at CA-TPA's latitude the ratio spans roughly 0.993..1.005 (as in the .ana file)
    if lat_deg == 42.7:
        assert 0.99 < ratio.min() < 0.999 and 1.001 < ratio.max() < 1.01


def test_linear_in_daily_value() -> None:
    lat = 0.745163
    a = rzwqm_daily_srad(np.full(DOYS.shape, 10.0), DOYS, lat)
    b = rzwqm_daily_srad(np.full(DOYS.shape, 25.0), DOYS, lat)
    np.testing.assert_allclose(b / a, 2.5, rtol=1e-6)
    assert np.all(rzwqm_daily_srad(np.zeros(5), np.arange(1, 6), lat) == 0.0)


def test_slope_branch_limits_and_sign() -> None:
    lat = 0.745163  # 42.7 N
    srad = np.full(DOYS.shape, 15.0)
    flat = rzwqm_daily_srad(srad, DOYS, lat)
    tiny = rzwqm_daily_srad(srad, DOYS, lat, slope_rad=1e-9, aspect_rad=0.0)
    np.testing.assert_allclose(tiny, flat, rtol=1e-6)
    winter = np.r_[0:40, 330:365]
    # SHAW azimuth: measured from north (AZM = pi - asin(...) near noon in the north), so aspect
    # pi faces south
    south = rzwqm_daily_srad(srad, DOYS, lat, slope_rad=math.radians(15.0), aspect_rad=math.pi)
    north = rzwqm_daily_srad(srad, DOYS, lat, slope_rad=math.radians(15.0), aspect_rad=0.0)
    assert np.all(south[winter] > flat[winter])
    assert np.all(north[winter] < flat[winter])
