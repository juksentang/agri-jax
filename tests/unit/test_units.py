"""Unit constants: round trips and the MJ/W factor."""

import math

import pytest

from agrijax.core import units as u


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


# ======================================================================== declared constants
def _module_numbers() -> set[str]:
    return {
        n
        for n, v in vars(u).items()
        if n.isupper() and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def test_every_numeric_constant_is_declared() -> None:
    assert set(u.CONSTANTS) == _module_numbers()
    for name, c in u.CONSTANTS.items():
        assert c.name == name
        assert c.value == getattr(u, name)
        assert c.description.strip()
        u.parse_unit(c.unit)  # the unit string parses


@pytest.mark.parametrize("name", sorted(n for n, c in u.CONSTANTS.items() if c.kind == "conversion"))
def test_conversion_constant_matches_its_units(name: str) -> None:
    c = u.CONSTANTS[name]
    assert math.isclose(c.value, u.conversion_factor(c.src, c.dst), rel_tol=1e-15)
    assert u.parse_unit(c.unit).same_dimension(u.parse_unit("-"))  # dst per src is a pure number
    # the reverse factor, when declared, is its reciprocal (round trip)
    back = [r for r in u.CONSTANTS.values() if r.kind == "conversion" and (r.src, r.dst) == (c.dst, c.src)]
    for r in back:
        assert math.isclose(c.value * r.value, 1.0, rel_tol=1e-15)


@pytest.mark.parametrize("name", sorted(n for n, c in u.CONSTANTS.items() if c.kind != "conversion"))
def test_physical_constant_has_provenance(name: str) -> None:
    from agrijax.core.coefficients import Provenance

    c = u.CONSTANTS[name]
    assert c.kind in {"physical", "coefficient"}
    assert not (c.src or c.dst)
    if c.paper or c.file:
        # the same validation as a model coefficient's provenance (reference key, file:line, paper)
        Provenance(c.ref_version, file=c.file, line=c.line, paper=c.paper, equation=c.equation)
    else:
        assert c.note.startswith("derived:"), f"{name} cites neither a paper, a source nor a derivation"


def test_derived_constants() -> None:
    assert u.KG_PER_M2_PER_MM == u.WATER_DENSITY_KG_M3 * u.conversion_factor("mm", "m")
    assert math.isclose(u.W_PER_M2_PER_MJ_M2_DAY, 1.0e6 / u.SECONDS_PER_DAY, rel_tol=1e-15)
    # the FAO-56 daily Stefan-Boltzmann constant is the SI one rounded to four digits
    si_per_day = u.STEFAN_BOLTZMANN_W_M2_K4 * u.MJ_M2_DAY_PER_W_PER_M2
    assert math.isclose(u.STEFAN_BOLTZMANN_MJ_M2_K4_DAY, si_per_day, rel_tol=1e-3)


def test_area_adapters() -> None:
    assert u.cm2_to_m2(1.0e4) == pytest.approx(1.0, rel=1e-15)
    assert u.m2_to_cm2(u.cm2_to_m2(123.4)) == pytest.approx(123.4, rel=1e-15)
