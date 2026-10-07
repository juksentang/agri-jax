"""Day-level PET processes (``agrijax.processes.pet.daily``): the kernels behind the process signature.

Registry and declarations, exact agreement with the array kernels, the writes check, jit / vmap over
a parameter batch, finite gradients through the process and no float32 upcast.
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
    DailyWeather,
    PETSiteParams,
    PETState,
    asce_reference_et,
    pet_asce_reference,
    pet_priestley_taylor,
    priestley_taylor,
)
from agrijax.processes.pet.daily import KM_DAY_TO_M_S

X64 = bool(jax.config.read("jax_enable_x64"))


def _f(x) -> jnp.ndarray:
    return jnp.asarray(x, dtype=float)


def site(**kw) -> PETSiteParams:
    base = dict(
        elevation=_f(200.0),
        latitude=_f(0.745163),
        wind_height=_f(2.0),
        albedo_soil=_f(0.13),
        trat=_f(1.0),
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


def test_registered_with_declarations() -> None:
    for proc in (pet_asce_reference, pet_priestley_taylor):
        assert registry[proc.name] is proc
        assert proc.writes and all(w.startswith("pet.") for w in proc.writes)
        assert "Source:" in proc.doc
        assert proc.info is not None and proc.info.ref_build.strip()
    assert pet_priestley_taylor.reads == ("canopy.lai",)
    assert pet_asce_reference.reads == ()


def test_srad_horizontal_is_not_read() -> None:
    w = weather()
    assert w.srad_horizontal is w.srad
    s, p = state(), site()
    other = pet_asce_reference(s, p, weather(srad_horizontal=_f(18.0)))
    assert float(other.pet.reference_short) == float(pet_asce_reference(s, p, w).pet.reference_short)


def test_processes_equal_their_kernels() -> None:
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


def test_canopy_of_several_crops_is_one_source() -> None:
    """Two crops: the Priestley-Taylor LAI is their sum (one shared source, not split between crops)."""
    theta = state().theta
    two = CanopyRecord(lai=_f([1.0, 2.0]), tlai=_f([1.1, 2.1]), height=_f([80.0, 150.0]))
    one = CanopyRecord(lai=_f([3.0]), tlai=_f([3.2]), height=_f([150.0]))
    a = pet_priestley_taylor(PETState.module(two, theta), site(), weather())
    b = pet_priestley_taylor(PETState.module(one, theta), site(), weather())
    rtol = 1e-12 if X64 else 1e-6
    np.testing.assert_allclose(float(a.pet.eo_priestley_taylor), float(b.pet.eo_priestley_taylor), rtol=rtol)


@pytest.mark.parametrize("proc", [pet_asce_reference, pet_priestley_taylor])
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
    albs = jnp.linspace(0.1, 0.3, 5)

    def eo(alb):
        return pet_priestley_taylor(s, site(albedo_soil=alb), w).pet.eo_priestley_taylor

    batch = jax.jit(jax.vmap(eo))(albs)
    single = np.array([float(eo(a)) for a in albs])
    np.testing.assert_allclose(np.asarray(batch), single, rtol=1e-12 if X64 else 1e-5)
    assert np.all(np.diff(np.asarray(batch)) < 0.0)  # a brighter soil, less energy, less EO
    g = jax.grad(eo)(_f(0.13))
    assert np.isfinite(float(g)) and float(g) < 0.0


def _f32(tree):
    def cast(x):
        a = jnp.asarray(x)
        return a.astype(jnp.float32) if jnp.issubdtype(a.dtype, jnp.floating) else x

    return jax.tree_util.tree_map(cast, tree)


@pytest.mark.parametrize("proc", [pet_asce_reference, pet_priestley_taylor])
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
    """The two kernels keep float32 (x64 on or off)."""
    f = jnp.float32
    assert priestley_taylor(f(20.0), f(25.0), f(15.0), f(2.0), f(0.2)).dtype == f
    e = asce_reference_et(
        f(12.0), f(28.0), f(22.0), f(60.0), f(2.0), elevation=f(200.0), latitude=f(0.7), doy=f(180)
    )
    assert e.et_short.dtype == f
