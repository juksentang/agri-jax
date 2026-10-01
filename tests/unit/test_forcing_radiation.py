"""``agrijax.forcing.radiation``: data-free checks against independent references.

The comparison with the reference model (dump tables, ``.ana`` column 88) is
``tests/integration/test_forcing_radiation_reference.py``. Here:

* the daily integral ``ISINB`` of the hourly shape (closed form of DSSAT ``SOLAR``) equals
  ``scipy.integrate.quad`` of ``sinB (1 + 0.4 sinB)`` over the day;
* the whole-hour sum is a rectangle rule of a curve whose integral is the daily value: within
  1.5 % of it for |latitude| <= 45 degrees;
* RTH follows ``INPDAY``: modifier, then bounds;
* on a flat surface SHAW's direct + diffuse is the horizontal hourly value (to double rounding),
  and the slope branch has the right sign and limit;
* CLOUDS is the Flerchinger-Yu line of the daily transmissivity, clipped to [0, 1];
* batching: a (sites x days) call equals the per-site calls; the coefficient table is complete.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from agrijax.core.units import parse_unit
from agrijax.forcing.radiation import (
    N_HOURS,
    RZWQM_RADIATION,
    coefficient_table,
    horizontal_radiation,
    hourly_horizontal_radiation,
    radiation_from_met,
    rzwqm_radiation,
    shaw_slope_partition,
)

PI = 3.14159  # DSSAT's PARAMETER value
DOYS = np.arange(1, 366)
LAT_CATPA = 0.745163  # CA-TPA, rzwqm.dat


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
    lat_rad = math.radians(lat_deg)
    hourly = hourly_horizontal_radiation(np.array([srad]), np.array([doy]), lat_rad)[0]
    for h in range(1, N_HOURS + 1):
        sinb_true = ssin + ccos * math.cos((h - 12.0) * PI / 12.0)
        if hourly[h - 1] > 0 and math.asin(sinb_true) / rad > 1.0:  # HANG raises beta to >= 1 deg
            implied = sinb_true * (1.0 + 0.4 * sinb_true) * srad * 1e6 / float(hourly[h - 1])
            assert implied == pytest.approx(integral_s, rel=2e-5), (h, lat_deg, doy)


@pytest.mark.parametrize("lat_deg", [-45.0, -30.0, 0.0, 30.0, 42.7, 45.0])
def test_hourly_resum_close_to_daily_value(lat_deg: float) -> None:
    out = rzwqm_radiation(np.full(DOYS.shape, 18.0), DOYS, math.radians(lat_deg))
    ratio = out.srad / 18.0
    assert np.all(np.isfinite(ratio))
    assert np.abs(ratio - 1.0).max() < 1.5e-2
    assert out.hourly_horizontal.shape == (DOYS.size, N_HOURS)
    assert np.all(out.hourly_horizontal >= 0.0)


def test_rth_modifier_then_bounds() -> None:
    srad = np.array([-1.0, 0.0, 5.51, 44.9, 60.0])
    np.testing.assert_array_equal(horizontal_radiation(srad), np.clip(srad * 100.0 * 0.01, 0.0, 45.0))
    pct = np.full(12, 100.0)
    pct[6] = 50.0  # July
    month = np.array([7, 7, 7, 1, 7])
    np.testing.assert_allclose(horizontal_radiation(srad, month, pct), [0.0, 0.0, 2.755, 44.9, 30.0])
    with pytest.raises(ValueError, match="month"):
        horizontal_radiation(srad, None, pct)
    with pytest.raises(ValueError, match="12"):
        horizontal_radiation(srad, month, pct[:6])
    with pytest.raises(ValueError, match=r"1\.\.12"):
        horizontal_radiation(srad, month * 0, pct)


def test_linear_in_the_daily_value_and_zero_days() -> None:
    a = rzwqm_radiation(np.full(DOYS.shape, 10.0), DOYS, LAT_CATPA).srad
    b = rzwqm_radiation(np.full(DOYS.shape, 25.0), DOYS, LAT_CATPA).srad
    np.testing.assert_allclose(b / a, 2.5, rtol=1e-6)
    with pytest.raises(ValueError, match="RTS = 0"):
        rzwqm_radiation(np.zeros(5), np.arange(1, 6), LAT_CATPA)
    z = rzwqm_radiation(np.zeros(5), np.arange(1, 6), LAT_CATPA, allow_zero=True)
    assert np.all(z.srad == 0.0) and np.all(z.clouds == 1.0)


def test_flat_partition_is_the_horizontal_value() -> None:
    out = rzwqm_radiation(np.linspace(1.0, 30.0, DOYS.size), DOYS, LAT_CATPA)
    np.testing.assert_allclose(out.hourly_slope, out.hourly_horizontal, rtol=1e-14, atol=0.0)
    direct, diffuse = shaw_slope_partition(out.hourly_horizontal, DOYS, LAT_CATPA)
    assert np.all(direct >= 0.0) and np.all(diffuse >= 0.0)
    np.testing.assert_array_equal(direct + diffuse, out.hourly_slope)
    # RTS is the hour-by-hour sum of HRTS in MJ
    np.testing.assert_allclose(out.srad, out.hourly_slope.sum(axis=1) * 3600.0 / 1e6, rtol=1e-14)


def test_slope_branch_limits_and_sign() -> None:
    srad = np.full(DOYS.shape, 15.0)
    flat = rzwqm_radiation(srad, DOYS, LAT_CATPA).srad
    tiny = rzwqm_radiation(srad, DOYS, LAT_CATPA, slope_rad=1e-9, aspect_rad=0.0).srad
    np.testing.assert_allclose(tiny, flat, rtol=1e-6)
    winter = np.r_[0:40, 330:365]
    # SHAW's azimuth is measured from north: aspect pi faces south
    south = rzwqm_radiation(srad, DOYS, LAT_CATPA, slope_rad=math.radians(15.0), aspect_rad=math.pi).srad
    north = rzwqm_radiation(srad, DOYS, LAT_CATPA, slope_rad=math.radians(15.0), aspect_rad=0.0).srad
    assert np.all(south[winter] > flat[winter])
    assert np.all(north[winter] < flat[winter])
    # RTH is the horizontal value whatever the slope
    np.testing.assert_array_equal(
        rzwqm_radiation(srad, DOYS, LAT_CATPA, slope_rad=0.3).srad_horizontal, horizontal_radiation(srad)
    )


def test_clouds_line_of_the_daily_transmissivity() -> None:
    c = RZWQM_RADIATION.shaw
    doy = np.array([172])
    # SUNMAX of CLOUDY for CA-TPA on day 172, then totals that give chosen transmissivities
    declin = c.declination_amplitude * math.sin(2 * c.pi * (172 - 80) / 365.0)
    haf = math.acos(-math.tan(LAT_CATPA) * math.tan(declin))
    sunmax = (
        24.0
        * c.solar_constant
        * (
            haf * math.sin(LAT_CATPA) * math.sin(declin)
            + math.cos(LAT_CATPA) * math.cos(declin) * math.sin(haf)
        )
        / c.pi
    )
    for srad in (1.0, 10.0, 20.0, 30.0):
        out = rzwqm_radiation(np.array([srad]), doy, LAT_CATPA)
        tt = out.hourly_horizontal.sum() / sunmax
        expect = min(max(float(np.float32(1.333)) - float(np.float32(1.666)) * tt, 0.0), 1.0)
        assert out.clouds[0] == pytest.approx(expect, abs=1e-12)
    assert 0.0 < rzwqm_radiation(np.array([15.0]), doy, LAT_CATPA).clouds[0] < 1.0
    assert rzwqm_radiation(np.array([0.5]), doy, LAT_CATPA).clouds[0] == 1.0
    assert rzwqm_radiation(np.array([40.0]), doy, LAT_CATPA).clouds[0] == 0.0


def test_batch_over_sites_equals_single_calls() -> None:
    rng = np.random.default_rng(0)
    lats = np.array([0.57731, 0.745163, 0.87441])[:, None]  # (sites, 1)
    srad = rng.uniform(1.0, 30.0, (3, DOYS.size))
    days = np.broadcast_to(DOYS, srad.shape)
    batch = rzwqm_radiation(srad, days, lats, slope_rad=np.array([0.0, 0.034907, 0.0])[:, None])
    assert batch.srad.shape == (3, DOYS.size) and batch.hourly_slope.shape == (3, DOYS.size, N_HOURS)
    for i, slope in enumerate((0.0, 0.034907, 0.0)):
        one = rzwqm_radiation(srad[i], DOYS, float(lats[i, 0]), slope_rad=slope)
        np.testing.assert_array_equal(batch.srad[i], one.srad)
        np.testing.assert_array_equal(batch.clouds[i], one.clouds)


def test_polar_days_stay_finite() -> None:
    summer = rzwqm_radiation(np.full(3, 25.0), np.array([170, 172, 174]), math.radians(80.0))
    assert np.all(np.isfinite(summer.srad)) and np.all(summer.hourly_horizontal > 0.0)
    night = rzwqm_radiation(np.full(3, 1.0), np.array([355, 356, 357]), math.radians(80.0), allow_zero=True)
    assert np.all(night.srad == 0.0) and np.all(np.isfinite(night.clouds))


def test_radiation_from_met_frame() -> None:
    idx = pd.date_range("2015-01-01", "2015-12-31", freq="D", name="date")
    met = pd.DataFrame({"srad_mj": np.linspace(2.0, 28.0, idx.size)}, index=idx)
    out = radiation_from_met(met, LAT_CATPA)
    direct = rzwqm_radiation(met["srad_mj"].to_numpy(), np.arange(1, 366), LAT_CATPA)
    np.testing.assert_array_equal(out["srad"].to_numpy(), direct.srad)
    np.testing.assert_array_equal(out["srad_horizontal"].to_numpy(), direct.srad_horizontal)
    assert list(out.columns) == ["srad", "srad_horizontal", "clouds"]


def test_coefficient_table_is_complete() -> None:
    rows = coefficient_table()
    assert len(rows) >= 30
    for r in rows:
        parse_unit(r["unit"])
        assert r["source"] and r["description"], r["path"]
        assert r["ref_version"] in ("dssat-4.8.6.0", "rzwqm2-4.6"), r["path"]
        if r["ref_version"] == "rzwqm2-4.6":
            assert not r["statement"] and r["paper"], r["path"]  # no RZWQM2 statement is quoted
        else:
            assert r["statement"], r["path"]
    paths = {r["path"] for r in rows}
    assert {"hmet.spitters_b", "shaw.max_transmissivity", "shaw.cloud_slope", "inpday.srad_max"} <= paths
