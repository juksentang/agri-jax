"""Coefficients of the PET kernels (``processes/pet/coefficients.py``).

* every coefficient of the ASCE reference ET and Priestley-Taylor equations is declared once, with
  unit, meaning and a provenance whose ``ref_version`` is the registry key of the process that uses
  it; the three ``REF_ET`` constants of RZWQM2 cite file, line, routine and the published equation
  (never the source text), DSSAT-CSM coefficients quote the Fortran statement;
* physical and astronomical constants are labelled but not calibrated;
* the module aliases (``ASCE_SHORT``, ...) are the declared defaults;
* a coefficient set with array leaves gives the result of the default floats, the processes read
  the set of their params, and gradients with respect to the calibratable coefficients are finite
  (and equal finite differences in float64);
* the PET kernels carry no bare numeric literal (lint rule AJ007).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import agrijax.processes.pet  # noqa: F401  (registers the processes)
from agrijax.core.coefficients import GUARDS
from agrijax.core.lint import lint_paths
from agrijax.core.process import list_processes
from agrijax.iface.crop import CanopyRecord
from agrijax.processes.pet import coefficients as PC
from agrijax.processes.pet.daily import (
    DailyWeather,
    PETSiteParams,
    PETState,
    pet_asce_reference,
    pet_priestley_taylor,
)
from agrijax.processes.pet.spam_dssat import SPAM_COEFFICIENTS  # registers pet/spam_*

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
        "asce_reference": PC.REF_ASCE,
        "priestley_taylor": PC.REF_DSSAT,
        # the SPAM partition; its coefficients live in spam_dssat.py (SPAM_COEFFICIENTS)
        "spam_pse": PC.REF_DSSAT,
        "spam_trans": PC.REF_DSSAT,
    }
    for r in _rows(SPAM_COEFFICIENTS):
        assert r["ref_version"] == PC.REF_DSSAT and r["statement"], r["path"]
    for r in _rows(PC.DSSAT_PT):
        assert r["ref_version"] == keys["priestley_taylor"], r["path"]
    for r in _rows(PC.ASCE_2005):
        # the REF_ET.FOR departures of variant="rzwqm" cite the RZWQM2 source
        expected = PC.REF_RZWQM if r["path"].endswith("_rzwqm") else keys["asce_reference"]
        assert r["ref_version"] == expected, r["path"]


def test_rows_follow_the_licence_of_their_reference() -> None:
    rows = _rows(PC.PET_COEFFICIENTS)
    assert len(rows) == len({r["path"] for r in rows}) == 51
    for r in rows:
        assert r["description"].strip() and r["unit"], r["path"]
        if r["ref_version"] == PC.REF_RZWQM:
            # RZWQM2 is closed source: file, line, routine and the published equation, no statement
            assert r["statement"] == "" and r["fortran"] == "", r["path"]
            assert r["file"] == PC.REFET and r["line"] and r["routine"] and r["paper"], r["path"]
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
    assert "asce.net_shortwave" in calibratable and "pt.alpha" in calibratable


def test_module_aliases_are_the_declared_defaults() -> None:
    assert pm.ASCE_SHORT == (900.0, 0.34) and pm.ASCE_TALL == (1600.0, 0.38)
    assert pt.EO_FLOOR_MM == PC.DSSAT_PT.eo_floor == 1.0e-4
    assert PC.PET_COEFFICIENTS == PC.PETCoefficients()


def test_numerical_guards_are_declared() -> None:
    for name in ("pet.asce.acos_margin", "pet.asce.rso_floor", "pet.asce.sqrt_floor"):
        assert name in GUARDS and GUARDS[name][1].strip()


def test_no_bare_numeric_literal_in_the_pet_kernels() -> None:
    files = sorted(PET_DIR.glob("*.py"))
    assert {f.name for f in files} >= {"penman_monteith.py", "priestley_taylor.py"}
    found = [f for f in lint_paths(files) if f.rule == "AJ007"]
    assert found == [], [str(f) for f in found]


# ------------------------------------------------------------------------------ kernels
def _close(a: object, b: object) -> None:
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=RTOL, atol=0.0)


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
    # a larger Priestley-Taylor coefficient raises EO proportionally in the normal branch
    hi = PC.DSSAT_PT.from_vector(jnp.asarray([1.32]), ["alpha"])
    ratio = pt.priestley_taylor(20.0, 25.0, 15.0, 2.0, 0.2, hi) / pt.priestley_taylor(
        20.0, 25.0, 15.0, 2.0, 0.2
    )
    np.testing.assert_allclose(float(ratio), 1.2, rtol=RTOL)


def _state_site(coefficients: PC.PETCoefficients | None) -> tuple[PETState, PETSiteParams, DailyWeather]:
    f = jnp.asarray
    canopy = CanopyRecord(lai=f([3.0]), tlai=f([3.2]), height=f([150.0]))
    p = PETSiteParams(
        elevation=f(200.0), latitude=f(0.745), wind_height=f(2.0), albedo_soil=f(0.2), trat=f(1.0),
        asce_variant="rzwqm", coefficients=coefficients,
    )  # fmt: skip
    w = DailyWeather(tmin=f(15.0), tmax=f(28.0), srad=f(22.0), rh=f(60.0), wind_run=f(150.0), doy=f(180.0))
    return PETState.module(canopy, f([0.2, 0.25])), p, w


@pytest.mark.parametrize("proc", [pet_asce_reference, pet_priestley_taylor])
def test_processes_read_the_coefficients_of_their_params(proc) -> None:
    ref = proc(*_state_site(None))
    assert _state_site(None)[1].coeffs is PC.PET_COEFFICIENTS
    _close(proc(*_state_site(PC.PETCoefficients().as_arrays())), ref)
    changed = PC.PET_COEFFICIENTS.from_vector(jnp.asarray([0.40, 1.3]), ["asce.net_shortwave", "pt.alpha"])
    out = proc(*_state_site(changed))
    diff = [
        not np.allclose(a, b)
        for a, b in zip(jax.tree_util.tree_leaves(out.pet), jax.tree_util.tree_leaves(ref.pet))
    ]
    assert any(diff)


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


def test_invalid_variant_raises() -> None:
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
