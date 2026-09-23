"""Day-level PET processes (``agri_jax.processes.pet.daily``): the kernels behind the process signature.

Registry and declarations, exact agreement with the array kernels, the writes check, jit / vmap
over a parameter batch and finite gradients through the process.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agri_jax.core import registry
from agri_jax.core.process import ProcessWriteError
from agri_jax.processes.pet import (
    DailyWeather,
    PETParams,
    PETSiteParams,
    PETState,
    SurfaceState,
    asce_reference_et,
    pet_asce_reference,
    pet_priestley_taylor,
    pet_shuttleworth_wallace,
    priestley_taylor,
    shuttleworth_wallace,
)
from agri_jax.processes.pet.shuttleworth_wallace import KM_DAY_TO_M_S


def _f(x: float) -> jnp.ndarray:
    return jnp.asarray(x, dtype=float)


def site(**kw) -> PETSiteParams:
    pet = PETParams(
        albedo_dry=_f(0.25),
        albedo_wet=_f(0.15),
        albedo_maturity=_f(0.23),
        albedo_residue=_f(0.30),
        soil_resistance=_f(54.0),
        stomatal_resistance=_f(224.0),
    )
    base = dict(
        pet=pet,
        elevation=_f(200.0),
        latitude=_f(0.745163),
        wc13=_f(0.255198),
        wc15=_f(0.141628),
        wind_height=_f(2.0),
        albedo_soil=_f(0.13),
        trat=_f(1.0),
        rainfall_zone=3,
    )
    base.update(kw)
    return PETSiteParams(**base)


def state(lai: float = 3.0, residue: float = 2500.0) -> PETState:
    surface = SurfaceState(
        lai=_f(lai),
        tlai=_f(lai),
        height_cm=_f(150.0),
        theta_surface=_f(0.2),
        residue_mass=_f(residue),
        residue_age=_f(30.0),
        residue_wet=_f(0.0),
    )
    return PETState.zeros_like_surface(surface)


def weather() -> DailyWeather:
    return DailyWeather(
        tmin=_f(15.0), tmax=_f(28.0), srad=_f(22.0), rh=_f(60.0), wind_run=_f(150.0), doy=_f(180)
    )


def test_registered_with_declarations() -> None:
    for proc in (pet_shuttleworth_wallace, pet_asce_reference, pet_priestley_taylor):
        assert registry[proc.name] is proc
        assert proc.writes and all(w.startswith("pet.") for w in proc.writes)
        assert "Source:" in proc.doc
    assert pet_shuttleworth_wallace.fortran_name == "POTEVPHR"


def test_processes_equal_the_kernels() -> None:
    s, p, w = state(), site(), weather()
    out = pet_shuttleworth_wallace(s, p, w)
    ref = shuttleworth_wallace(
        w.tmin,
        w.tmax,
        w.srad,
        w.rh,
        w.wind_run,
        s.surface.lai,
        s.surface.height_cm,
        p.pet,
        theta_surface=s.surface.theta_surface,
        wc13=p.wc13,
        wc15=p.wc15,
        elevation=p.elevation,
        latitude=p.latitude,
        doy=w.doy,
        tlai=s.surface.tlai,
        residue_mass=s.surface.residue_mass,
        residue_age=s.surface.residue_age,
        rainfall_zone=3,
    )
    assert float(out.pet.transpiration) == float(ref.transpiration) > 0.0
    assert float(out.pet.soil_evaporation) == float(ref.soil_evaporation)
    assert float(out.pet.residue_evaporation) == float(ref.residue_evaporation) > 0.0

    out = pet_asce_reference(s, p, w)
    r = asce_reference_et(
        w.tmin,
        w.tmax,
        w.srad,
        w.rh,
        w.wind_run * KM_DAY_TO_M_S,
        elevation=p.elevation,
        latitude=p.latitude,
        doy=w.doy,
    )
    assert float(out.pet.reference_short) == float(r.et_short) > 0.0
    assert float(out.pet.reference_tall) == float(r.et_tall) > float(r.et_short)
    out_rz = pet_asce_reference(s, site(asce_variant="rzwqm"), w)
    assert float(out_rz.pet.reference_short) != float(out.pet.reference_short)

    out = pet_priestley_taylor(s, p, w)
    assert float(out.pet.eo_priestley_taylor) == float(
        priestley_taylor(w.srad, w.tmax, w.tmin, _f(3.0), _f(0.13))
    )


@pytest.mark.parametrize("proc", [pet_shuttleworth_wallace, pet_asce_reference, pet_priestley_taylor])
def test_writes_only_what_is_declared(proc, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")
    s = state()
    new = proc(s, site(), weather())  # raises ProcessWriteError on an undeclared change
    changed = proc.check_writes(s, new)
    assert changed and all(c.startswith("pet.") for c in changed)
    bad = new.set("surface.lai", new.surface.lai + 1.0)
    with pytest.raises(ProcessWriteError):
        proc.check_writes(s, bad)


def test_jit_vmap_over_parameter_batch_and_grad() -> None:
    s, w = state(lai=1.5), weather()
    rs = jnp.linspace(100.0, 400.0, 5)

    def transp(r):
        p = site()
        p = p.set("pet.stomatal_resistance", r)
        return pet_shuttleworth_wallace(s, p, w).pet.transpiration

    batch = jax.jit(jax.vmap(transp))(rs)
    single = np.array([float(transp(r)) for r in rs])
    np.testing.assert_allclose(np.asarray(batch), single, rtol=1e-12 if jax.config.jax_enable_x64 else 1e-5)
    assert np.all(np.diff(np.asarray(batch)) < 0.0)  # more stomatal resistance, less transpiration
    g = jax.grad(transp)(_f(224.0))
    assert np.isfinite(float(g)) and float(g) < 0.0
    # bare soil: the canopy branch is masked and its gradient must stay finite (no NaN through where)
    s0 = state(lai=0.0, residue=0.0)
    g0 = jax.grad(
        lambda r: (
            pet_shuttleworth_wallace(s0, site().set("pet.stomatal_resistance", r), w).pet.soil_evaporation
        )
    )(_f(224.0))
    assert np.isfinite(float(g0))
