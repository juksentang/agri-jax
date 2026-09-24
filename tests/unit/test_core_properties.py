"""Hypothesis property tests for ``core.state`` (get / set / tree_diff) and ``core.units``.

Deterministic: every test runs with ``derandomize=True`` and no example database, so the same
examples are drawn on every machine. The references are independent of the code under test:
the state checks compare against a NumPy leaf-by-leaf comparison of flattened pytrees and the
declared field layout; the unit checks compare against conversion factors written out from their
definitions (1 cm = 10 mm, 1 d = 86400 s, 1 ha = 10^4 m2, 0 degC = 273.15 K) with exact
rational arithmetic.
"""

from __future__ import annotations

import math
from fractions import Fraction

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import Array

from agri_jax.core import units
from agri_jax.core.state import Params, State, field, get_path, leaf_paths, set_path, tree_diff

SETTINGS = settings(derandomize=True, database=None, deadline=None, max_examples=150)
FLOAT_DTYPE = np.float64 if jax.config.jax_enable_x64 else np.float32


# ------------------------------------------------------------------------------ a nested state


class _Soil(State):
    theta: Array = field(unit="cm3 cm-3", dims="n_node")
    wtd: Array = field(unit="cm")


class _Crop(State):
    lai: Array = field(unit="m2 m-2", dims="n_crop")
    biomass: Array = field(unit="kg ha-1", dims=("n_crop", "4"))


class _S(State):
    soil: _Soil
    crop: _Crop
    day: Array = field(unit="d")


class _P(Params):
    k: Array = field(unit="d-1")
    n_node: int = field(static=True, default=3)


PATHS = ("soil.theta", "soil.wtd", "crop.lai", "crop.biomass", "day")
SHAPES = {"soil.theta": (5,), "soil.wtd": (), "crop.lai": (2,), "crop.biomass": (2, 4), "day": ()}

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=32)


def _arr(draw: st.DrawFn, shape: tuple[int, ...]) -> np.ndarray:
    n = int(np.prod(shape)) if shape else 1
    vals = draw(st.lists(finite, min_size=n, max_size=n))
    return np.asarray(vals, dtype=FLOAT_DTYPE).reshape(shape)


@st.composite
def states(draw: st.DrawFn) -> _S:
    a = {p: jnp.asarray(_arr(draw, SHAPES[p])) for p in PATHS}
    return _S(
        soil=_Soil(theta=a["soil.theta"], wtd=a["soil.wtd"]),
        crop=_Crop(lai=a["crop.lai"], biomass=a["crop.biomass"]),
        day=a["day"],
    )


@st.composite
def state_path_value(draw: st.DrawFn) -> tuple[_S, str, jnp.ndarray]:
    s = draw(states())
    path = draw(st.sampled_from(PATHS))
    return s, path, jnp.asarray(_arr(draw, SHAPES[path]))


def _numpy_diff(a: object, b: object) -> list[str]:
    """Independent reference for tree_diff: flatten both and compare leaf values with NumPy."""
    la, lb = jtu.tree_leaves(a), jtu.tree_leaves(b)
    return [
        PATHS[i]
        for i, (x, y) in enumerate(zip(la, lb))
        if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
    ]


def test_leaf_paths_follow_the_declared_layout() -> None:
    s = _S(
        soil=_Soil(theta=jnp.zeros(5), wtd=jnp.asarray(0.0)),
        crop=_Crop(lai=jnp.zeros(2), biomass=jnp.zeros((2, 4))),
        day=jnp.asarray(0.0),
    )
    assert leaf_paths(s) == list(PATHS) == s.leaf_paths()
    assert [p for p, _ in s.items()] == list(PATHS)
    assert leaf_paths(_P(k=jnp.asarray(1.0))) == ["k"]  # static fields are not leaves


@SETTINGS
@given(state_path_value())
def test_set_then_get_returns_the_value(spv: tuple[_S, str, jnp.ndarray]) -> None:
    s, path, v = spv
    new = s.set(path, v)
    assert get_path(new, path) is v and new.get(path) is v
    assert jtu.tree_structure(new) == jtu.tree_structure(s)
    for other in PATHS:  # every other leaf is the same object
        if other != path:
            assert get_path(new, other) is get_path(s, other)
    # setting the old value back restores the original leaf-for-leaf
    back = set_path(new, path, get_path(s, path))
    assert all(x is y for x, y in zip(jtu.tree_leaves(back), jtu.tree_leaves(s)))


@SETTINGS
@given(state_path_value())
def test_tree_diff_equals_numpy_reference(spv: tuple[_S, str, jnp.ndarray]) -> None:
    s, path, v = spv
    new = s.set(path, v)
    d = tree_diff(s, new)
    assert d == _numpy_diff(s, new)
    same = np.array_equal(np.asarray(v), np.asarray(get_path(s, path)))
    assert d == ([] if same else [path])
    assert tree_diff(new, s) == d  # symmetric
    assert tree_diff(s, s) == []
    # a value-equal copy (different objects) is not a change
    copy = jtu.tree_map(lambda x: jnp.array(np.asarray(x)), s)
    assert tree_diff(s, copy) == []


@SETTINGS
@given(states(), st.lists(st.sampled_from(PATHS), min_size=1, max_size=5, unique=True))
def test_tree_diff_finds_every_changed_leaf(s: _S, paths: list[str]) -> None:
    new = s
    for p in paths:
        new = new.set(p, get_path(s, p) + 1.0)  # |x| <= 1e6 in float32 still changes by 1
    assert sorted(tree_diff(s, new)) == sorted(paths) == sorted(_numpy_diff(s, new))


def test_tree_diff_nan_and_structure() -> None:
    s = _S(
        soil=_Soil(theta=jnp.full(5, jnp.nan), wtd=jnp.asarray(0.0)),
        crop=_Crop(lai=jnp.zeros(2), biomass=jnp.zeros((2, 4))),
        day=jnp.asarray(0.0),
    )
    assert tree_diff(s, s.set("soil.theta", jnp.full(5, jnp.nan))) == []  # NaN == NaN for the diff
    assert tree_diff(s, s.set("soil.theta", jnp.zeros(5))) == ["soil.theta"]
    assert tree_diff(s, s.set("soil.wtd", jnp.zeros(3))) == ["soil.wtd"]  # shape change is a change
    with pytest.raises(ValueError, match="structure"):
        tree_diff(s, {"soil": s.soil})


def test_tree_diff_under_trace_is_identity_based() -> None:
    s = _S(
        soil=_Soil(theta=jnp.zeros(5), wtd=jnp.asarray(0.0)),
        crop=_Crop(lai=jnp.zeros(2), biomass=jnp.zeros((2, 4))),
        day=jnp.asarray(0.0),
    )
    seen: list[list[str]] = []

    def f(t: _S) -> jnp.ndarray:
        new = t.set("crop.lai", t.crop.lai * 1.0)
        seen.append(tree_diff(t, new))
        return new.crop.lai

    jax.jit(f)(s)
    assert seen == [["crop.lai"]]  # conservative: a traced replacement counts as a write


# ------------------------------------------------------------------------------ units

# normal floats only: a subnormal input underflows to 0 in a conversion, which is not a round-trip bug
reals = st.floats(
    min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False, allow_subnormal=False
)
positive = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)

# conversion factors from their definitions, exact
_MM_PER_CM = Fraction(10)
_S_PER_DAY = Fraction(24 * 60 * 60)
_W_M2_PER_MJ_M2_D = Fraction(10**6) / _S_PER_DAY
_G_M2_PER_KG_HA = Fraction(1000) / Fraction(10**4)
_K0 = Fraction(27315, 100)

FORWARD_BACK = [
    (units.mm_to_cm, units.cm_to_mm, 1 / _MM_PER_CM),
    (units.mj_m2_day_to_w_m2, units.w_m2_to_mj_m2_day, _W_M2_PER_MJ_M2_D),
    (units.kg_ha_to_g_m2, units.g_m2_to_kg_ha, _G_M2_PER_KG_HA),
    (units.mj_m2_day_to_mm_water, units.mm_water_to_mj_m2_day, 1 / Fraction(245, 100)),
]


@pytest.mark.parametrize(("fwd", "back", "factor"), FORWARD_BACK, ids=lambda x: getattr(x, "__name__", ""))
@SETTINGS
@given(x=reals)
def test_linear_conversions_match_definitions_and_round_trip(fwd, back, factor: Fraction, x: float) -> None:
    exact = float(Fraction(x) * factor)
    assert math.isclose(fwd(x), exact, rel_tol=4e-16, abs_tol=0.0)
    assert math.isclose(back(fwd(x)), x, rel_tol=1e-15, abs_tol=0.0)
    assert math.isclose(fwd(back(x)), x, rel_tol=1e-15, abs_tol=0.0)


@SETTINGS
@given(x=reals)
def test_temperature_round_trip(x: float) -> None:
    assert math.isclose(units.celsius_to_kelvin(x), float(Fraction(x) + _K0), rel_tol=4e-16, abs_tol=1e-13)
    assert math.isclose(units.kelvin_to_celsius(units.celsius_to_kelvin(x)), x, rel_tol=1e-15, abs_tol=1e-12)


@SETTINGS
@given(e=positive, lam=st.floats(min_value=2.2, max_value=2.6))
def test_energy_water_with_custom_latent_heat(e: float, lam: float) -> None:
    mm = units.mj_m2_day_to_mm_water(e, latent_heat=lam)
    assert math.isclose(mm, float(Fraction(e) / Fraction(lam)), rel_tol=4e-16)
    assert math.isclose(units.mm_water_to_mj_m2_day(mm, latent_heat=lam), e, rel_tol=1e-15)


def test_constants_are_mutually_consistent() -> None:
    assert units.CM_PER_MM * units.MM_PER_CM == 1.0
    assert units.CM_PER_M * units.M_PER_CM == 1.0
    assert units.SECONDS_PER_DAY == units.SECONDS_PER_HOUR * units.HOURS_PER_DAY
    assert math.isclose(units.W_PER_M2_PER_MJ_M2_DAY * units.MJ_M2_DAY_PER_W_PER_M2, 1.0, rel_tol=1e-15)
    assert units.G_M2_PER_KG_HA * units.KG_HA_PER_G_M2 == 1.0
    # Stefan-Boltzmann in both units: 5.670374419e-8 W m-2 K-4 * 86400 s / 1e6 = 4.899e-9 (FAO-56 rounds 4.903e-9)
    daily = units.STEFAN_BOLTZMANN_W_M2_K4 * units.MJ_M2_DAY_PER_W_PER_M2
    assert abs(daily - units.STEFAN_BOLTZMANN_MJ_M2_K4_DAY) / daily < 1e-3


@SETTINGS
@given(st.lists(reals, min_size=1, max_size=16))
def test_conversions_are_elementwise_on_arrays(xs: list[float]) -> None:
    """Array inputs give the element-by-element scalar result (what processes rely on)."""
    a = np.asarray(xs, dtype=np.float64)
    for fwd, back, _ in FORWARD_BACK:
        np.testing.assert_array_equal(fwd(a), np.array([fwd(x) for x in xs]))
        np.testing.assert_array_equal(back(a), np.array([back(x) for x in xs]))
    np.testing.assert_array_equal(
        units.celsius_to_kelvin(a), np.array([units.celsius_to_kelvin(x) for x in xs])
    )
