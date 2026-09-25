"""Coefficients of the PET kernels (``processes/pet/coefficients.py``).

* every coefficient of the Shuttleworth-Wallace, ASCE reference ET and Priestley-Taylor equations
  is declared once, with unit, meaning and a provenance whose ``ref_version`` is the registry key
  of the process that uses it; RZWQM2 coefficients cite file, line, routine and the published
  equation (never the source text), DSSAT-CSM coefficients quote the Fortran statement;
* physical and astronomical constants are labelled but not calibrated;
* the module aliases (``RESIDUE_DIAMETER_CM``, ``ASCE_SHORT``, ...) are the declared defaults;
* a coefficient set with array leaves gives the result of the default floats, the processes read
  the set of their params, and gradients with respect to the calibratable coefficients are finite
  (and equal finite differences in float64);
* the PET kernels carry no bare numeric literal (lint rule AJ007).

The cited source lines are checked against the reference sources in
``tests/integration/test_pet_coefficients_source.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import agrijax.processes.pet  # noqa: F401  (registers the processes)
from agrijax.core.coefficients import GUARDS, iter_coefficients
from agrijax.core.lint import lint_paths
from agrijax.core.process import list_processes
from agrijax.processes.pet import SWResult
from agrijax.processes.pet import coefficients as PC
from agrijax.processes.pet.daily import (
    DailyWeather,
    PETSiteParams,
    PETState,
    SurfaceState,
    pet_asce_reference,
    pet_priestley_taylor,
    pet_shuttleworth_wallace,
)

from .test_pet import params, site, summer_day

sw = importlib.import_module("agrijax.processes.pet.shuttleworth_wallace")
pm = importlib.import_module("agrijax.processes.pet.penman_monteith")
pt = importlib.import_module("agrijax.processes.pet.priestley_taylor")

X64 = bool(jax.config.read("jax_enable_x64"))
RTOL = 1e-12 if X64 else 1e-5
PET_DIR = Path(__file__).resolve().parents[2] / "src" / "agrijax" / "processes" / "pet"


def _rows(tree: object) -> list[dict]:
    return PC.coefficient_table(tree)


def _pet_keys() -> dict[str, str]:
    """``{kernel name: ref_version}`` of the registered PET processes."""
    out = {}
    for p in list_processes():
        key = str(p.key)
        if key.startswith("pet/"):
            out[key.split("/")[1].split("@")[0]] = key.split("@")[1].split(":")[0]
    return out


# ------------------------------------------------------------------------------ declaration
def test_every_coefficient_cites_the_reference_of_its_process() -> None:
    keys = _pet_keys()
    assert keys == {
        "shuttleworth_wallace": PC.REF_RZWQM,
        "asce_reference": PC.REF_ASCE,
        "priestley_taylor": PC.REF_DSSAT,
    }
    for r in _rows(PC.RZWQM_SW):
        assert r["ref_version"] == keys["shuttleworth_wallace"], r["path"]
    for r in _rows(PC.DSSAT_PT):
        assert r["ref_version"] == keys["priestley_taylor"], r["path"]
    for r in _rows(PC.ASCE_2005):
        # the REF_ET.FOR departures of variant="rzwqm" cite the RZWQM2 source
        expected = PC.REF_RZWQM if r["path"].endswith("_rzwqm") else keys["asce_reference"]
        assert r["ref_version"] == expected, r["path"]


def test_rows_follow_the_licence_of_their_reference() -> None:
    rows = _rows(PC.PET_COEFFICIENTS)
    assert len(rows) == len({r["path"] for r in rows}) == 156
    for r in rows:
        assert r["description"].strip() and r["unit"], r["path"]
        if r["ref_version"] == PC.REF_RZWQM:
            # RZWQM2 source has no licence file: file, line, routine and the published equation, no statement
            assert r["statement"] == "" and r["fortran"] == "", r["path"]
            assert r["file"] in {PC.RZPET, PC.REFET} and r["line"] and r["routine"] and r["paper"], r["path"]
        elif r["ref_version"] == PC.REF_DSSAT:
            assert r["file"] == PC.PETFOR and r["routine"] == "PETPT" and r["statement"], r["path"]
        elif r["path"] == "asce.stefan_boltzmann":
            # the 'asce' sigma is the FAO-56 value (eq. 39); ASCE eq. 17 states 4.901e-9
            assert r["ref_version"] == PC.REF_ASCE and r["paper"] == PC.FAO56 and r["equation"] == "39"
            assert "4.901e-9" in r["note"] and not r["file"]
        else:
            assert r["ref_version"] == PC.REF_ASCE and r["paper"] == PC.ASCE and not r["file"], r["path"]
            assert r["equation"], r["path"]


PHYSICAL = {
    "sw.econst.gravity",
    "sw.econst.gas_constant",
    "sw.econst.virtual_t_offset",
    "sw.econst.virtual_vapour",
    "sw.econst.mw_ratio",
    "sw.econst.cp_air",
    "sw.maxsw.hours_per_radian",
    "sw.maxsw.solar_noon_hour",
    "sw.maxsw.radians_per_hour",
    "sw.resist.von_karman",
    "sw.resist.vapour_diffusivity",
    "sw.potevp.stefan_boltzmann",
    "asce.days_per_year",
    "asce.t_kelvin_longwave",
    "asce.stefan_boltzmann",
    "asce.t_kelvin",
    "asce.stefan_boltzmann_rzwqm",
    "asce.day_angle_rzwqm",
    "asce.t_kelvin_rzwqm",
}


def test_physical_constants_are_labelled_but_not_calibrated() -> None:
    rows = {r["path"]: r for r in _rows(PC.PET_COEFFICIENTS)}
    fixed = {p for p, r in rows.items() if not r["calibratable"]}
    assert fixed == PHYSICAL
    calibratable = PC.PET_COEFFICIENTS.calibratable_paths()
    assert len(calibratable) == len(rows) - len(PHYSICAL)
    assert "sw.potevp.canopy_extinction" in calibratable and "pt.alpha" in calibratable


def test_module_aliases_are_the_declared_defaults() -> None:
    r = PC.RZWQM_SW.resist
    assert sw.RESIDUE_DIAMETER_CM == {"corn": 1.0, "soybean": 0.5, "wheat": 0.25}
    assert sw.RESIDUE_DENSITY_G_CM3 == {"corn": 0.15, "soybean": 0.17, "wheat": 0.18}
    assert sw.RESIDUE_RANDOMNESS == r.residue_cover_default == 1.32
    assert sw.LONGWAVE_COEFFS == {1: (1.2, -0.2), 2: (1.1, -0.1), 3: (1.0, 0.0)}
    assert (sw.VON_KARMAN, sw.Z0_BARE_SOIL, sw.EDDY_DECAY, sw.DRAG_COEFF) == (0.41, 0.01, 2.5, 0.07)
    assert sw.STEFAN_BOLTZMANN == 4.903e-9 and sw.CP_AIR == 1.013e-3 and sw.CANOPY_EXTINCTION == 0.594
    assert sw.TWO_THIRDS == 2.0 / 3.0 and sw.PEN123 == 0.123 and sw.SOLAR_CONST_HOURLY == 4.9212
    assert sw._HOURS_PER_RADIAN == 12.0 / np.pi
    assert pm.ASCE_SHORT == (900.0, 0.34) and pm.ASCE_TALL == (1600.0, 0.38)
    assert pt.EO_FLOOR_MM == PC.DSSAT_PT.eo_floor == 1.0e-4
    assert PC.PET_COEFFICIENTS == PC.PETCoefficients()


def test_numerical_guards_are_declared() -> None:
    for name in ("pet.sw.tiny", "pet.sw.wind_floor", "pet.sw.height_floor", "pet.asce.acos_margin"):
        assert name in GUARDS and GUARDS[name][1].strip()


def test_no_bare_numeric_literal_in_the_pet_kernels() -> None:
    files = sorted(PET_DIR.glob("*.py"))
    assert {f.name for f in files} >= {"shuttleworth_wallace.py", "penman_monteith.py", "priestley_taylor.py"}
    found = [f for f in lint_paths(files) if f.rule == "AJ007"]
    assert found == [], [str(f) for f in found]


# ------------------------------------------------------------------------------ kernels
def _sw(coefficients: PC.SWCoefficients, **kw) -> SWResult:
    d = summer_day(residue_mass=kw.pop("residue_mass", 3000.0))
    d.update(kw)
    return sw.shuttleworth_wallace(
        **d,
        params=params(),
        theta_surface=0.2,
        doy=180,
        residue_age=30.0,
        coefficients=coefficients,
        **site(),
    )


def _close(a: object, b: object) -> None:
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("lai", [0.0, 0.5, 3.0])
def test_array_coefficients_give_the_default_shuttleworth_wallace(lai: float) -> None:
    _close(_sw(PC.RZWQM_SW.as_arrays(), lai=lai), _sw(PC.RZWQM_SW, lai=lai))


@pytest.mark.parametrize("variant", ["asce", "rzwqm"])
def test_array_coefficients_give_the_default_reference_et(variant: str) -> None:
    kw = dict(elevation=200.0, latitude=0.745, doy=180, variant=variant)
    ref = pm.asce_reference_et(12.0, 28.0, 22.0, 60.0, 2.0, **kw)
    _close(
        pm.asce_reference_et(12.0, 28.0, 22.0, 60.0, 2.0, coefficients=PC.ASCE_2005.as_arrays(), **kw), ref
    )


@pytest.mark.parametrize("tmax", [2.0, 20.0, 38.0])
def test_array_coefficients_give_the_default_priestley_taylor(tmax: float) -> None:
    ref = pt.priestley_taylor(20.0, tmax, tmax - 10.0, 2.0, 0.2)
    _close(pt.priestley_taylor(20.0, tmax, tmax - 10.0, 2.0, 0.2, PC.DSSAT_PT.as_arrays()), ref)


def test_coefficients_change_the_result_in_the_expected_direction() -> None:
    base = _sw(PC.RZWQM_SW)
    denser = PC.RZWQM_SW.from_vector(jnp.asarray([0.8]), ["potevp.canopy_extinction"])
    more = _sw(denser)
    assert float(more.canopy_fraction) > float(base.canopy_fraction)
    assert float(more.transpiration) > float(base.transpiration)
    # a larger Priestley-Taylor coefficient raises EO proportionally in the normal branch
    hi = PC.DSSAT_PT.from_vector(jnp.asarray([1.32]), ["alpha"])
    ratio = pt.priestley_taylor(20.0, 25.0, 15.0, 2.0, 0.2, hi) / pt.priestley_taylor(
        20.0, 25.0, 15.0, 2.0, 0.2
    )
    np.testing.assert_allclose(float(ratio), 1.2, rtol=RTOL)


def _state_site(coefficients: PC.PETCoefficients | None) -> tuple[PETState, PETSiteParams, DailyWeather]:
    f = jnp.asarray
    surface = SurfaceState(
        lai=f(3.0), tlai=f(3.2), height_cm=f(150.0), theta_surface=f(0.2), residue_mass=f(3000.0),
        residue_age=f(30.0), residue_wet=f(0.0),
    )  # fmt: skip
    p = PETSiteParams(
        pet=params(), elevation=f(200.0), latitude=f(0.745), wc13=f(0.255), wc15=f(0.142), wind_height=f(2.0),
        albedo_soil=f(0.2), trat=f(1.0), rainfall_zone=3, asce_variant="rzwqm", coefficients=coefficients,
    )  # fmt: skip
    w = DailyWeather(tmin=f(15.0), tmax=f(28.0), srad=f(22.0), rh=f(60.0), wind_run=f(150.0), doy=f(180.0))
    return PETState.zeros_like_surface(surface), p, w


@pytest.mark.parametrize("proc", [pet_shuttleworth_wallace, pet_asce_reference, pet_priestley_taylor])
def test_processes_read_the_coefficients_of_their_params(proc) -> None:
    ref = proc(*_state_site(None))
    assert _state_site(None)[1].coeffs is PC.PET_COEFFICIENTS
    _close(proc(*_state_site(PC.PETCoefficients().as_arrays())), ref)
    changed = PC.PET_COEFFICIENTS.from_vector(
        jnp.asarray([0.8, 0.40, 1.3]), ["sw.potevp.canopy_extinction", "asce.net_shortwave", "pt.alpha"]
    )
    out = proc(*_state_site(changed))
    diff = [
        not np.allclose(a, b)
        for a, b in zip(jax.tree_util.tree_leaves(out.pet), jax.tree_util.tree_leaves(ref.pet))
    ]
    assert any(diff)


def _fd(fun, x0: float, h: float) -> float:
    return (fun(x0 + h) - fun(x0 - h)) / (2.0 * h)


def test_gradient_wrt_the_calibratable_sw_coefficients_is_finite_and_matches_fd() -> None:
    paths, vec = PC.RZWQM_SW.to_vector()
    assert len(paths) == sum(1 for _ in iter_coefficients(PC.RZWQM_SW)) - 12

    def loss(v: jax.Array) -> jax.Array:
        r = _sw(PC.RZWQM_SW.from_vector(v, paths))
        return r.transpiration + r.soil_evaporation + r.residue_evaporation

    g = jax.grad(loss)(vec)
    assert bool(jnp.all(jnp.isfinite(g)))
    for name in ("potevp.canopy_extinction", "resist.leaf_boundary_resistance", "albedo.residue_ageing_rate"):
        i = paths.index(name)
        assert float(g[i]) != 0.0, name
        if X64:
            x0 = float(vec[i])
            fd = _fd(lambda x, i=i: float(loss(vec.at[i].set(x))), x0, 1e-6 * max(abs(x0), 1.0))
            np.testing.assert_allclose(float(g[i]), fd, rtol=1e-5, err_msg=name)


def test_gradient_wrt_the_asce_and_pt_coefficients_is_finite() -> None:
    paths, vec = PC.ASCE_2005.to_vector()

    def asce(v: jax.Array) -> jax.Array:
        c = PC.ASCE_2005.from_vector(v, paths)
        r = pm.asce_reference_et(
            12.0, 28.0, 22.0, 60.0, 2.0, elevation=200.0, latitude=0.745, doy=180, coefficients=c
        )
        return r.et_short + r.et_tall

    assert bool(jnp.all(jnp.isfinite(jax.grad(asce)(vec))))
    ppaths, pvec = PC.DSSAT_PT.to_vector()
    for tmax in (2.0, 20.0, 38.0):
        g = jax.grad(
            lambda v, t=tmax: pt.priestley_taylor(
                20.0, t, t - 10.0, 2.0, 0.2, PC.DSSAT_PT.from_vector(v, ppaths)
            )
        )(pvec)
        assert bool(jnp.all(jnp.isfinite(g)))


def test_invalid_rainfall_zone_and_variant_raise() -> None:
    kw = {**site(), "rainfall_zone": 0}
    with pytest.raises(KeyError, match="rainfall_zone"):
        sw.shuttleworth_wallace(**summer_day(), params=params(), theta_surface=0.2, doy=180, **kw)
    kw = {**site(), "residue_type": "rice"}
    with pytest.raises(KeyError, match="residue_type"):
        sw.shuttleworth_wallace(**summer_day(), params=params(), theta_surface=0.2, doy=180, **kw)
    with pytest.raises(KeyError, match="variant"):
        pm.asce_reference_et(
            12.0,
            28.0,
            22.0,
            60.0,
            2.0,
            elevation=200.0,
            latitude=0.7,
            doy=180,
            variant="fao",  # type: ignore[arg-type]
        )
