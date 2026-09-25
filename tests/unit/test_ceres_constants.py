"""The named codes, conversions and small coefficient groups of the CERES-Maize port.

* :class:`SolarCoefficients` and :class:`XstageCoefficients` are declared coefficients (unit,
  meaning, DSSAT-CSM v4.8.6.0 file:line and statement; the statements are checked against the
  source in ``tests/integration/test_ceres_constants_source.py``), leaves kept out of the
  calibration vector;
* ``daylength`` / ``twilight_daylength`` with the coefficients as arrays reproduce the defaults and
  are differentiable with respect to them;
* the codes are the DSSAT values and the guard is registered;
* the CERES-Maize interface modules and ``ROOTWU`` carry no bare numeric literal (lint rule AJ007).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.core.coefficients import GUARDS, coefficient_table
from agrijax.core.lint import lint_paths
from agrijax.processes.crop.ceres_maize import constants as k
from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength

X64 = jax.config.jax_enable_x64
_NUM = re.compile(r"(?<![A-Za-z_0-9])(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+|\d+)")
SRC = Path(__file__).resolve().parents[2] / "src" / "agrijax"
OWNED = [
    SRC / "processes" / "soil_water" / "uptake.py",
    *sorted((SRC / "processes" / "crop" / "ceres_maize").glob("*.py")),
]


def test_solar_and_xstage_numbers_are_declared_coefficients() -> None:
    solar = {r["path"]: r for r in coefficient_table(k.SolarCoefficients)}
    assert set(solar) == {
        "pi", "dec_amplitude", "dec_doy_offset", "year_days", "tw_deg_to_rad", "tw_dec_amplitude",
        "tw_dec_rate", "tw_dec_doy0", "tw_sin_depression", "tw_dlv_min", "tw_hours_per_rad",
    }  # fmt: skip
    xs = {r["path"]: r for r in coefficient_table(k.XstageCoefficients)}
    assert set(xs) == {"ti_base", "ti_slope", "silk_base", "silk_slope", "efg_base", "efg_slope"}
    for r in [*solar.values(), *xs.values()]:
        assert r["ref_version"] == "dssat-4.8.6.0" and r["file"] and r["line"] and r["routine"]
        assert r["unit"] and r["description"] and r["statement"]
        assert abs(r["value"]) in [float(x) for x in _NUM.findall(r["statement"])], r
        assert not r["calibratable"] and not r["static"]
    for r in solar.values():
        assert r["file"] == "Weather/SOLAR.for" and r["routine"] in ("DAYLEN", "TWILIGHT")
    for r in xs.values():
        assert r["file"] == "Plant/CERES-Maize/MZ_PHENOL.for" and r["routine"] == "MZ_PHENOL"
    assert k.SOLAR_COEFFICIENTS.calibratable_paths() == []
    assert k.XSTAGE_COEFFICIENTS.calibratable_paths() == []
    assert len(jax.tree_util.tree_leaves(k.SOLAR_COEFFICIENTS)) == len(solar)  # leaves, not static


def test_daylength_with_array_coefficients_reproduces_the_defaults_and_is_differentiable() -> None:
    doy = jnp.arange(1.0, 367.0)
    arr = k.SolarCoefficients().as_arrays()
    for lat in (-60.0, 0.0, 43.0, 70.0):
        for fn in (daylength, twilight_daylength):
            base = np.asarray(fn(doy, lat))
            got = np.asarray(jax.jit(fn)(doy, lat, arr))
            np.testing.assert_allclose(got, base, rtol=1e-12 if X64 else 1e-5, atol=1e-12 if X64 else 1e-4)
    # the DSSAT statements (SOLAR.for DAYLEN) by hand at 43 N on DOY 172
    dec = -23.45 * math.cos(2.0 * 3.14159 * (172 + 10.0) / 365.0)
    rad = 3.14159 / 180.0
    soc = math.tan(rad * dec) * math.tan(rad * 43.0)
    want = 12.0 + 24.0 * math.asin(soc) / 3.14159
    np.testing.assert_allclose(float(daylength(172.0, 43.0)), want, rtol=1e-12 if X64 else 1e-6)
    g = jax.grad(lambda c: jnp.sum(daylength(doy, 43.0, c)))(arr)
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(g))
    assert float(g.dec_amplitude) != 0.0  # a larger declination amplitude changes the daylength


def test_codes_are_the_dssat_values() -> None:
    stages = [
        k.ISTAGE_SOWING, k.ISTAGE_GERMINATION, k.ISTAGE_EMERGENCE, k.ISTAGE_JUVENILE,
        k.ISTAGE_END_JUVENILE, k.ISTAGE_TASSEL_INIT, k.ISTAGE_END_LEAF_GROWTH, k.ISTAGE_EFG,
        k.ISTAGE_MATURITY, k.ISTAGE_AFTER_MATURITY,
    ]  # fmt: skip
    assert stages == [7, 8, 9, 1, 2, 3, 4, 5, 6, 10]  # the MZ_PHENOL stage cycle
    status = [
        k.CROP_STATUS_MATURE, k.CROP_STATUS_NO_GERMINATION, k.CROP_STATUS_NO_EMERGENCE,
        k.CROP_STATUS_COLD, k.CROP_STATUS_DROUGHT,
    ]  # fmt: skip
    assert status == [1, 12, 13, 32, 33]
    assert all(isinstance(x, int) for x in [*stages, *status, k.MDATE_NONE, k.YRDOY_SCALE])
    assert (k.XSTAGE_SEASINIT, k.MDATE_NONE) == (0.1, -99)
    assert (k.MG_PER_G, k.G_PER_MG, k.RLV_PRECISION) == (1000.0, 0.001, 1000.0)
    assert (k.PAIR_MEAN_DIVISOR, k.PAIR_MEAN_WEIGHT, k.HOURS_PER_HALF_DAY) == (2.0, 0.5, 12.0)
    assert GUARDS["ceres_maize.den_min"][0] == k.DEN_MIN == 1e-6
    assert GUARDS["ceres_maize.eps"][0] == 1e-12


def test_no_bare_numeric_literal_in_the_ceres_interface_and_rootwu() -> None:
    assert len(OWNED) > 5
    found = [f for f in lint_paths(OWNED) if f.rule == "AJ007"]
    assert found == [], [f.format() for f in found]
