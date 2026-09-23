"""Brooks-Corey hydraulics against the CA-TPA ``rzwqm.dat`` read through the io layer.

The same check on inline copies of the records runs in ``tests/unit/test_hydraulics.py``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from agri_jax.io.rzwqm.dat import read_rzwqm_dat
from agri_jax.processes.soil_water.hydraulics import (
    H_CLAMP_RZWQM,
    H_FC13,
    H_FC110,
    H_WP,
    SoilHydraulicParams,
    theta_of_h,
)


def _soil_physics_control(text: str) -> list[float]:
    """The data line of the soil-physics control record (the one that ends with Hmin, HFC, HWP)."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if "Wilting Point water suction head" in ln:
            for nxt in lines[i + 1 :]:
                s = nxt.strip()
                if s and not s.startswith("="):
                    return [float(t) for t in s.split()]
    raise LookupError("soil-physics control record not found")


def test_catpa_from_scenario_file(catpa_scenario: Path) -> None:
    """theta(-333), theta(-100), theta(-15000) reproduce the file's rec2; c2 of horizon 5 is the file value."""

    hyd = read_rzwqm_dat(catpa_scenario / "rzwqm.dat").hydraulics
    p = SoilHydraulicParams.from_rzwqm_dat(hyd, derive=False)
    n = p.n_horizon
    assert n == 5
    np.testing.assert_allclose(np.asarray(theta_of_h(jnp.full(n, H_FC13), p)), hyd["theta_fc33"], atol=1.5e-6)
    np.testing.assert_allclose(
        np.asarray(theta_of_h(jnp.full(n, H_FC110), p)), hyd["theta_fc10"], atol=1.5e-6
    )
    np.testing.assert_allclose(np.asarray(theta_of_h(jnp.full(n, H_WP), p)), hyd["theta_wp"], atol=1.5e-6)
    d = SoilHydraulicParams.from_rzwqm_dat(hyd)
    np.testing.assert_allclose(np.asarray(d.c2)[4], hyd["c2"][4], rtol=1e-5)


def test_catpa_hmin_is_the_rzwqm_clamp(catpa_scenario: Path) -> None:
    """CA-TPA's 19-item control record gives Hmin (item 17) = HWP (item 19) = H_CLAMP_RZWQM."""
    rec = _soil_physics_control((catpa_scenario / "rzwqm.dat").read_text(encoding="latin-1"))
    assert len(rec) == 19
    hmin, hfc, hwp = rec[16:19]
    assert hmin == H_CLAMP_RZWQM and hwp == H_WP and hfc == H_FC13
