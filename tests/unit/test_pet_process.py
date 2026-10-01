"""Day-level PET processes (``agrijax.processes.pet.daily``): the kernels behind the process signature.

Registry and declarations, exact agreement with the array kernels (canopy from the P6 port, the
surface node of P7, RTH from the weather, the residue parameters), the writes check, jit / vmap
over a parameter batch, finite gradients through the process and no float32 upcast.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import registry
from agrijax.core.process import ProcessWriteError
from agrijax.iface.crop import CanopyRecord
from agrijax.processes.pet import (
    RESIDUE_KINDS,
    RZWQM_SW,
    DailyWeather,
    PETParams,
    PETSiteParams,
    PETState,
    SurfaceResidue,
    asce_reference_et,
    pet_asce_reference,
    pet_priestley_taylor,
    pet_shuttleworth_wallace,
    priestley_taylor,
    shuttleworth_wallace,
)
from agrijax.processes.pet.shuttleworth_wallace import (
    KM_DAY_TO_M_S,
    RESIDUE_DENSITY_G_CM3,
    RESIDUE_DIAMETER_CM,
)

X64 = bool(jax.config.read("jax_enable_x64"))


def _f(x) -> jnp.ndarray:
    return jnp.asarray(x, dtype=float)


def residue(mass: float = 2500.0, kind: str = "corn") -> SurfaceResidue:
    return SurfaceResidue(mass=_f(mass), age=_f(30.0), wet=_f(0.0), kind=_f(RESIDUE_KINDS[kind]))


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
        residue=residue(),
    )
    base.update(kw)
    return PETSiteParams(**base)


def state(lai: float = 3.0, height: float = 150.0) -> PETState:
    canopy = CanopyRecord(lai=_f([lai]), tlai=_f([lai + 0.2]), height=_f([height]))
    return PETState.module(canopy, _f([0.2, 0.27, 0.3]))


def weather(**kw) -> DailyWeather:
    base = dict(tmin=_f(15.0), tmax=_f(28.0), srad=_f(22.0), rh=_f(60.0), wind_run=_f(150.0), doy=_f(180))
    base.update(kw)
    return DailyWeather(**base)


def _kernel(s: PETState, p: PETSiteParams, w: DailyWeather, kind: str = "corn", **kw):
    return shuttleworth_wallace(
        w.tmin,
        w.tmax,
        w.srad,
        w.rh,
        w.wind_run,
        s.canopy.lai[0],
        s.canopy.height[0],
        p.pet,
        theta_surface=s.theta[0],
        wc13=p.wc13,
        wc15=p.wc15,
        elevation=p.elevation,
        latitude=p.latitude,
        doy=w.doy,
        tlai=s.canopy.tlai[0],
        srad_horizontal=w.srad_horizontal,
        residue_age=p.residue.age,
        residue_wet=p.residue.wet,
        rainfall_zone=3,
        residue_diameter_cm=RESIDUE_DIAMETER_CM[kind],
        residue_density=RESIDUE_DENSITY_G_CM3[kind],
        **kw,
    )


def test_registered_with_declarations() -> None:
    for proc in (pet_shuttleworth_wallace, pet_asce_reference, pet_priestley_taylor):
        assert registry[proc.name] is proc
        assert proc.writes and all(w.startswith("pet.") for w in proc.writes)
        assert "Source:" in proc.doc
        assert proc.info is not None and proc.info.ref_build.strip()
    assert pet_shuttleworth_wallace.fortran_name == "POTEVPHR"
    assert pet_shuttleworth_wallace.reads == ("canopy", "theta")
    assert pet_priestley_taylor.reads == ("canopy.lai",)
    assert pet_asce_reference.reads == ()


@pytest.mark.parametrize("kind", ["corn", "soybean", "wheat"])
def test_sw_process_equals_the_kernel(kind: str) -> None:
    s, w = state(), weather(srad_horizontal=_f(22.08))
    p = site(residue=residue(kind=kind))
    out = pet_shuttleworth_wallace(s, p, w)
    ref = _kernel(s, p, w, kind, residue_mass=p.residue.mass)
    assert float(out.pet.transpiration) == float(ref.transpiration) > 0.0
    assert float(out.pet.soil_evaporation) == float(ref.soil_evaporation)
    assert float(out.pet.residue_evaporation) == float(ref.residue_evaporation) > 0.0
    assert float(out.pet.eo_priestley_taylor) == 0.0  # untouched P5 fields


def test_residue_kind_zero_switches_the_residue_branch_off() -> None:
    s, w = state(), weather()
    out = pet_shuttleworth_wallace(s, site(residue=residue(kind="none")), w)
    ref = _kernel(s, site(), w, "corn", residue_mass=0.0)
    assert float(out.pet.residue_evaporation) == 0.0 == float(ref.residue_evaporation)
    assert float(out.pet.soil_evaporation) == float(ref.soil_evaporation)
    # the default residue record is no residue
    none = pet_shuttleworth_wallace(s, site(residue=SurfaceResidue.none()), w)
    assert float(none.pet.residue_evaporation) == 0.0


def test_srad_horizontal_defaults_to_srad_and_is_read() -> None:
    w = weather()
    assert w.srad_horizontal is w.srad
    s, p = state(), site()
    a = pet_shuttleworth_wallace(s, p, w)
    b = pet_shuttleworth_wallace(s, p, weather(srad_horizontal=w.srad))
    assert float(a.pet.transpiration) == float(b.pet.transpiration)
    c = pet_shuttleworth_wallace(s, p, weather(srad_horizontal=_f(18.0)))
    assert float(c.pet.transpiration) != float(a.pet.transpiration)
    # the other processes ignore RTH
    other = pet_asce_reference(s, p, weather(srad_horizontal=_f(18.0)))
    assert float(other.pet.reference_short) == float(pet_asce_reference(s, p, w).pet.reference_short)


def test_canopy_of_several_crops_is_one_source() -> None:
    """Two crops: LAI and TLAI summed, the tallest height (one shared source, not split between crops)."""
    theta = state().theta
    two = CanopyRecord(lai=_f([1.0, 2.0]), tlai=_f([1.1, 2.1]), height=_f([80.0, 150.0]))
    one = CanopyRecord(lai=_f([3.0]), tlai=_f([3.2]), height=_f([150.0]))
    a = pet_shuttleworth_wallace(PETState.module(two, theta), site(), weather())
    b = pet_shuttleworth_wallace(PETState.module(one, theta), site(), weather())
    rtol = 1e-12 if X64 else 1e-6
    np.testing.assert_allclose(float(a.pet.transpiration), float(b.pet.transpiration), rtol=rtol)


def test_other_processes_equal_their_kernels() -> None:
    s, p, w = state(), site(), weather()
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
    bad = new.set("canopy.lai", new.canopy.lai + 1.0)
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
    np.testing.assert_allclose(np.asarray(batch), single, rtol=1e-12 if X64 else 1e-5)
    assert np.all(np.diff(np.asarray(batch)) < 0.0)  # more stomatal resistance, less transpiration
    g = jax.grad(transp)(_f(224.0))
    assert np.isfinite(float(g)) and float(g) < 0.0
    # bare soil: the canopy branch is masked and its gradient must stay finite (no NaN through where)
    s0 = state(lai=0.0, height=0.0)
    p0 = site(residue=residue(mass=0.0))
    g0 = jax.grad(
        lambda r: pet_shuttleworth_wallace(s0, p0.set("pet.stomatal_resistance", r), w).pet.soil_evaporation
    )(_f(224.0))
    assert np.isfinite(float(g0))


def _f32(tree):
    def cast(x):
        a = jnp.asarray(x)
        return a.astype(jnp.float32) if jnp.issubdtype(a.dtype, jnp.floating) else x

    return jax.tree_util.tree_map(cast, tree)


@pytest.mark.parametrize("proc", [pet_shuttleworth_wallace, pet_asce_reference, pet_priestley_taylor])
def test_float32_in_float32_out(proc) -> None:
    """Float32 inputs give float32 fluxes, also with x64 enabled."""
    w = weather(srad_horizontal=_f(22.1))
    out = proc(_f32(state()), _f32(site()), _f32(w))
    for leaf in jax.tree_util.tree_leaves(out.pet):
        assert leaf.dtype == jnp.float32
    ref = proc(state(), site(), w)
    for a, b in zip(jax.tree_util.tree_leaves(out.pet), jax.tree_util.tree_leaves(ref.pet), strict=True):
        np.testing.assert_allclose(float(a), float(b), rtol=1e-5, atol=1e-7)


def test_kernels_keep_float32() -> None:
    """The three kernels and the clear-sky quadrature keep float32 (x64 on or off)."""
    f = jnp.float32
    r = shuttleworth_wallace(
        f(12.0), f(26.0), f(20.0), f(60.0), f(150.0), f(2.0), f(120.0), _f32(site().pet),
        theta_surface=f(0.2), wc13=f(0.25), wc15=f(0.14), elevation=f(200.0), latitude=f(0.745),
        doy=f(180.0), coefficients=RZWQM_SW,
    )  # fmt: skip
    assert r.transpiration.dtype == f and r.rn.dtype == f
    assert priestley_taylor(f(20.0), f(25.0), f(15.0), f(2.0), f(0.2)).dtype == f
    e = asce_reference_et(
        f(12.0), f(28.0), f(22.0), f(60.0), f(2.0), elevation=f(200.0), latitude=f(0.7), doy=f(180)
    )
    assert e.et_short.dtype == f
