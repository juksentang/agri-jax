"""Unit constants: round trips and the MJ/W factor."""

import pytest

from agri_jax.core import units as u


@pytest.mark.parametrize(
    "fwd, back, x",
    [
        (u.mm_to_cm, u.cm_to_mm, 12.5),
        (u.mj_m2_day_to_w_m2, u.w_m2_to_mj_m2_day, 20.0),
        (u.celsius_to_kelvin, u.kelvin_to_celsius, 25.0),
        (u.kg_ha_to_g_m2, u.g_m2_to_kg_ha, 8000.0),
        (u.mj_m2_day_to_mm_water, u.mm_water_to_mj_m2_day, 15.0),
    ],
)
def test_roundtrip(fwd, back, x) -> None:
    assert back(fwd(x)) == pytest.approx(x, rel=1e-12)


def test_known_values() -> None:
    assert u.mm_to_cm(10.0) == pytest.approx(1.0)
    assert u.mj_m2_day_to_w_m2(1.0) == pytest.approx(11.574, rel=1e-4)
    assert u.kg_ha_to_g_m2(10.0) == pytest.approx(1.0)
    assert u.celsius_to_kelvin(0.0) == pytest.approx(273.15)
