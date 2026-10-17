"""Soil parameter specs and spaces by ``.SOL`` layer (``calib/soil.py``), no data.

* Every spec of a site's space carries bounds and a non-empty source; the site values lie inside
  their bounds and every layer's ``LL + gap <= DUL``, ``DUL + gap <= SAT`` holds at them, so the
  site maps to ``z`` and back.
* ``ParamSpace.apply`` writes each path of the space into the site's value mapping (one ``.SOL``
  layer at a time).
* ``swcn_spec`` refuses a layer whose ``SSKS`` is missing; a site value outside its bounds, or a
  calibrated member that leaves no room for a fixed neighbour, is refused.
* A site value on or near an end of its logistic box (saturated ``z``) is refused unless accepted;
  for the soil blocks of both sites every site value maps to a finite ``z`` well inside the
  logistic range, where ``d theta / d z`` is not small.
* The default paths and bounds of the existing specs are unchanged.

The site values below are those of the two ``.SOL`` profiles of the DSSAT examples UFGA8201
(IBMZ910014, eight layers, sand) and IBWA8301 (IBMZ910013, seven layers), copied as test data.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.calib.soil import (
    CN_BOUNDS,
    LAYER_QUANTITIES,
    SCALAR_QUANTITIES,
    SITE_BOUNDS,
    SITE_MARGIN,
    SLPF_BOUNDS,
    SWCON_BOUNDS,
    U_ALL_SOL_MIN,
    U_SOIL_SOL_RANGE,
    U_UPPER,
    Box,
    RelativeBox,
    WidenedBox,
    cn_spec,
    layer_water_chain,
    slpf_spec,
    soil_space,
    srgf_spec,
    swcn_spec,
    swcon_spec,
)
from agrijax.calib.space import ParamSpace

X64 = bool(jax.config.jax_enable_x64)
NAN = float("nan")

SAND = {  # IBMZ910014 (UFGA8201)
    "cn": 60.0,
    "swcon": 0.65,
    "u": 2.0,
    "slpf": 0.92,
    "ll": np.array([0.026, 0.025, 0.025, 0.025, 0.028, 0.028, 0.029, 0.070]),
    "dul": np.array([0.096, 0.086, 0.086, 0.086, 0.090, 0.090, 0.130, 0.258]),
    "sat": np.array([0.23] * 7 + [0.36]),
    "srgf": np.array([1.0, 1.0, 0.7, 0.3, 0.05, 0.03, 0.002, 0.0]),
    "swcn": np.full(8, NAN),
}
LOAM = {  # IBMZ910013 (IBWA8301)
    "cn": 60.0,
    "swcon": 0.60,
    "u": 5.0,
    "slpf": 1.0,
    "ll": np.array([0.22, 0.23, 0.24, 0.25, 0.26, 0.25, 0.26]),
    "dul": np.array([0.35, 0.35, 0.35, 0.37, 0.38, 0.38, 0.40]),
    "sat": np.array([0.55, 0.55, 0.55, 0.48, 0.46, 0.46, 0.48]),
    "srgf": np.array([1.0, 1.0, 0.8, 0.4, 0.2, 0.05, 0.02]),
    "swcn": np.array([1.2, 0.9, 0.6, 0.4, 0.3, 0.3, 0.2]),  # synthetic: the file has none
}
#: a 26-parameter soil block: CN, SWCON, U, SLPF, LL and DUL of every layer, SAT of the top three,
#: SRGF of layers 3-5 (layers 1 and 2 have SRGF 1.0, on the cap of its domain; the eighth has 0)
LAYERS = {"ll": range(8), "dul": range(8), "sat": range(3), "srgf": range(2, 5)}


def _site_vector(space: ParamSpace, site: dict) -> np.ndarray:
    return np.asarray(
        [
            float(np.asarray(site[s.path])[s.index] if s.index is not None else site[s.path])
            for s in space.specs
        ]
    )


def _sand_space() -> ParamSpace:
    return soil_space(SAND, ("cn", "swcon", "u", "slpf", "ll", "dul", "sat", "srgf"), LAYERS, SITE_BOUNDS)


def test_site_space_has_sources_and_is_feasible_at_the_site() -> None:
    sp = _sand_space()
    assert sp.n == 26
    assert sp.names[:4] == ("CN", "SWCON", "U", "SLPF")
    assert sp.names[4:9] == ("LL_H1", "DUL_H1", "SAT_H1", "LL_H2", "DUL_H2")
    assert "SRGF_H3" in sp.names and "SRGF_H2" not in sp.names and "SAT_H4" not in sp.names
    for row in sp.table():
        assert row["lower"] < row["upper"], row
        assert row["bounds_source"].strip(), row
        assert row["domain_source"].strip(), row
    assert len(sp.ordered) == 8  # one chain per .SOL layer
    theta = _site_vector(sp, SAND)
    assert sp.violations(theta) == []
    back = np.asarray(sp.to_physical(sp.to_unconstrained(jnp.asarray(theta))))
    assert np.all((theta > sp.lower) & (theta < sp.upper))
    np.testing.assert_allclose(back, theta, rtol=1e-9 if X64 else 1e-5)
    # the boxes: CN widened around the site value 60, LL of the sand on its floor-free relative box
    spec = dict(zip(sp.names, sp.specs, strict=True))
    assert (spec["CN"].lower, spec["CN"].upper) == (50.0, 94.0)
    assert spec["LL_H1"].lower == pytest.approx(0.026 * 0.6) and spec["LL_H1"].upper == pytest.approx(
        0.026 * 1.4
    )
    assert spec["SAT_H1"].lower == pytest.approx(0.18) and spec["SAT_H1"].upper == pytest.approx(0.28)
    assert (spec["U"].lower, spec["U"].upper) == (U_ALL_SOL_MIN, U_UPPER)
    assert U_ALL_SOL_MIN < SAND["u"] == U_SOIL_SOL_RANGE[0]  # the SOIL.SOL minimum is the sand's U
    assert (spec["SLPF"].lower, spec["SLPF"].upper) == SLPF_BOUNDS
    assert "*.SOL" in spec["U"].bounds_source and "SOIL.SOL" in spec["SLPF"].bounds_source
    assert spec["SRGF_H3"].upper == 1.0 and spec["SRGF_H3"].lower == pytest.approx(0.35)


def test_every_draw_keeps_each_layer_ordered() -> None:
    sp = _sand_space()
    z = np.random.default_rng(1).normal(0.0, 3.0, (500, sp.n))
    th = np.asarray(sp.to_physical(jnp.asarray(z)))
    names = list(sp.names)
    tol = 1e-12 if X64 else 1e-6
    for h in range(8):
        ll, dul = th[:, names.index(f"LL_H{h + 1}")], th[:, names.index(f"DUL_H{h + 1}")]
        assert np.all(dul >= ll + 0.01 - tol)
        sat = th[:, names.index(f"SAT_H{h + 1}")] if f"SAT_H{h + 1}" in names else SAND["sat"][h]
        assert np.all(sat >= dul + 0.01 - tol)
    assert all(sp.violations(t) == [] for t in th[:50])


def test_apply_writes_one_sol_layer_of_the_site_values() -> None:
    sp = _sand_space()
    tree = {k: jnp.asarray(v) for k, v in SAND.items()}
    theta = _site_vector(sp, SAND).copy()
    i = sp.names.index("DUL_H3")
    theta[i] = 0.1
    out = sp.apply(tree, jnp.asarray(theta))
    want = SAND["dul"].copy()
    want[2] = 0.1
    np.testing.assert_allclose(np.asarray(out["dul"]), want, rtol=0 if X64 else 1e-6)
    np.testing.assert_array_equal(np.asarray(out["swcn"]), np.asarray(tree["swcn"]))  # untouched
    np.testing.assert_allclose(np.asarray(sp.get(out)), theta, rtol=0 if X64 else 1e-6)


def test_prefix_puts_the_paths_under_a_node() -> None:
    sp = soil_space(SAND, ("cn", "ll"), {"ll": [0]}, SITE_BOUNDS, prefix="sol")
    assert sp.paths == ("sol.cn", "sol.ll") and sp.specs[1].index == 0


def test_swcn_needs_a_given_ssks() -> None:
    with pytest.raises(ValueError, match="missing"):
        swcn_spec(0, NAN)
    with pytest.raises(ValueError, match="missing"):
        swcn_spec(0, -99.0)
    with pytest.raises(ValueError, match="missing"):
        soil_space(SAND, ("swcn",), [0], SITE_BOUNDS)
    s = swcn_spec(2, 0.6, prefix=None)
    assert s.transform == "log" and s.path == "swcn" and s.index == 2 and s.name == "SWCN_H3"
    assert (s.lower, s.upper) == pytest.approx((0.15, 2.4)) and s.domain.lower_open
    sp = soil_space(LOAM, ("swcn", "srgf"), [2, 3], SITE_BOUNDS)
    assert sp.names == ("SRGF_H3", "SWCN_H3", "SRGF_H4", "SWCN_H4")


def test_site_values_outside_the_bounds_are_refused() -> None:
    # the sand's CN 60 lies below the estimation table's 61: the default bounds do not hold it
    with pytest.raises(ValueError, match="CN: site value 60"):
        soil_space(SAND, ("cn",))
    sp = soil_space(SAND, ("cn",), bounds={"cn": SITE_BOUNDS["cn"]})
    assert (sp.specs[0].lower, sp.specs[0].upper) == (50.0, 94.0)
    with pytest.raises(ValueError, match="needs its bounds"):
        soil_space(SAND, ("ll",), [0])
    with pytest.raises(ValueError, match="unknown"):
        soil_space(SAND, ("ksat",))
    with pytest.raises(ValueError, match="outside the 8"):
        soil_space(SAND, ("ll",), [8], SITE_BOUNDS)
    with pytest.raises(ValueError, match=r"no \.SOL layers given for ll"):
        soil_space(SAND, ("ll", "dul"), {"dul": [0]}, SITE_BOUNDS)
    with pytest.raises(ValueError, match="not among"):
        soil_space(SAND, ("ll",), {"dul": [0]}, SITE_BOUNDS)


def test_a_calibrated_member_must_leave_room_for_a_fixed_neighbour() -> None:
    wide = {"dul": Box(0.05, 0.35, "test box"), "ll": Box(0.01, 0.2, "test box")}
    with pytest.raises(ValueError, match="fixed SAT"):
        soil_space(SAND, ("dul",), [0], wide)  # DUL up to 0.35 against SAT 0.23
    with pytest.raises(ValueError, match="fixed DUL"):
        soil_space(SAND, ("ll",), [0], wide)  # LL up to 0.2 against DUL 0.096
    low_sat = {"sat": Box(0.05, 0.5, "test box")}
    with pytest.raises(ValueError, match="fixed DUL"):
        soil_space(SAND, ("sat",), [0], low_sat)
    broken = {**SAND, "dul": SAND["ll"] + 0.005}
    with pytest.raises(ValueError, match="gap"):
        soil_space(broken, ("ll", "dul"), [0], SITE_BOUNDS)


def test_the_boxes() -> None:
    r = RelativeBox(0.6, 1.4, "t", half_width=0.05, floor=0.01)
    assert r.bounds(0.025) == pytest.approx((0.015, 0.035))
    assert r.bounds(0.35) == pytest.approx((0.30, 0.40))  # the half-width caps the relative box
    assert r.bounds(0.012) == pytest.approx((0.01, 0.0168))  # the floor
    assert SITE_BOUNDS["srgf"].bounds(0.8) == pytest.approx((0.4, 1.0))  # capped at 1
    w = WidenedBox(*CN_BOUNDS, 10.0, "t")
    assert (
        w.bounds(60.0) == (50.0, 94.0) and w.bounds(80.0) == (61.0, 94.0) and w.bounds(90.0) == (61.0, 100.0)
    )
    assert Box(1.0, 2.0, "t").bounds(5.0) == (1.0, 2.0)
    assert set(SITE_BOUNDS) <= set(SCALAR_QUANTITIES + LAYER_QUANTITIES)
    assert all(b.source.strip() for b in SITE_BOUNDS.values())


def test_new_specs_and_the_unchanged_defaults() -> None:
    cn = cn_spec()
    assert cn.path == "soil.cn" and (cn.lower, cn.upper) == CN_BOUNDS
    assert swcon_spec().path == "soil.swcon"
    with pytest.raises(ValueError, match="source"):
        cn_spec(bounds=(50.0, 94.0))
    c2 = cn_spec(bounds=(50.0, 94.0), bounds_source="site value - 10")
    assert (c2.lower, c2.upper) == (50.0, 94.0) and c2.bounds_source == "site value - 10"
    specs, chain = layer_water_chain(1, (0.05, 0.30), (0.10, 0.45), (0.30, 0.60), "test box")
    assert [s.path for s in specs] == ["soil.ll", "soil.dul", "soil.sat"] and chain.names == (
        "LL1",
        "DUL1",
        "SAT1",
    )
    s = slpf_spec()
    assert s.path == "soil.slpf" and (s.lower, s.upper) == SLPF_BOUNDS and "IBSG910010" in s.bounds_source
    assert s.domain.lower_open and s.domain.upper == 1.0
    with pytest.raises(ValueError, match="source"):
        slpf_spec(bounds=(0.5, 1.0))
    with pytest.raises(ValueError, match="domain"):
        slpf_spec(bounds=(0.5, 1.2), bounds_source="t")
    g = srgf_spec(3, 0.3, prefix="crop.soil")
    assert g.path == "crop.soil.srgf" and g.index == 3 and (g.lower, g.upper) == pytest.approx((0.15, 0.45))


#: the soil blocks of the two sites: every scalar, LL and DUL of every layer, SAT of the top five,
#: SRGF of the layers below the second with a nonzero value (the loam without SLPF, see below)
BLOCKS = {
    "sand": (SAND, ("cn", "swcon", "u", "slpf", "ll", "dul", "sat", "srgf"),
             {"ll": range(8), "dul": range(8), "sat": range(5), "srgf": range(2, 7)}),
    "loam": (LOAM, ("cn", "swcon", "u", "ll", "dul", "sat", "srgf"),
             {"ll": range(7), "dul": range(7), "sat": range(5), "srgf": range(2, 7)}),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(BLOCKS))
def test_the_site_maps_to_an_unsaturated_z(name: str) -> None:
    site, quantities, layers = BLOCKS[name]
    sp = soil_space(site, quantities, layers, SITE_BOUNDS)
    theta = _site_vector(sp, site)
    z = np.asarray(sp.to_unconstrained(jnp.asarray(theta)))
    zmax = float(np.log((1.0 - SITE_MARGIN) / SITE_MARGIN))
    assert np.all(np.isfinite(z)) and np.all(np.abs(z) <= zmax + 1e-9), dict(zip(sp.names, z, strict=True))
    # d theta / d z at the site is not small against the box width (a saturated coordinate gives
    # about 1e-9 of it)
    diag = np.diag(np.asarray(jax.jacfwd(sp.to_physical)(jnp.asarray(z))))
    assert np.all(diag >= 1e-3 * sp.width), dict(zip(sp.names, diag / sp.width, strict=True))


def test_a_site_value_on_a_bound_is_refused_unless_accepted() -> None:
    # SRGF 1.0 of the top layers sits on the cap of its domain; the loam's SLPF 1.0 on the SOIL.SOL
    # upper bound (also its domain's)
    with pytest.raises(ValueError, match="SRGF_H2 1"):
        soil_space(SAND, ("srgf",), [1], SITE_BOUNDS)
    with pytest.raises(ValueError, match="SLPF 1"):
        soil_space(LOAM, ("slpf",))
    # the old default U box (the SOIL.SOL minimum 2.0) puts the sand's U on its lower bound
    with pytest.raises(ValueError, match="U 2"):
        soil_space(SAND, ("u",), bounds={"u": Box(U_SOIL_SOL_RANGE[0], U_UPPER, "SOIL.SOL")})
    sp = soil_space(SAND, ("srgf",), [1], SITE_BOUNDS, site_on_bound=True)
    assert sp.names == ("SRGF_H2",) and (sp.specs[0].lower, sp.specs[0].upper) == (0.5, 1.0)
    # near, not on: within SITE_MARGIN of the width
    near = {**SAND, "swcon": SWCON_BOUNDS[1] - 0.5 * SITE_MARGIN * (SWCON_BOUNDS[1] - SWCON_BOUNDS[0])}
    with pytest.raises(ValueError, match="SWCON"):
        soil_space(near, ("swcon",))
    # a chain member near the previous member's site value plus the gap
    tight = {**SAND, "dul": SAND["ll"] + 0.0101}
    with pytest.raises(ValueError, match="DUL_H1"):
        soil_space(tight, ("ll", "dul"), [0], {"ll": Box(0.01, 0.03, "t"), "dul": Box(0.03, 0.2, "t")})
