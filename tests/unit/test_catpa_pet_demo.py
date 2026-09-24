"""Data-free checks of :mod:`agri_jax.models.catpa_pet_demo` (the real-data checks are in
tests/integration/test_core_real_pipeline.py).

Synthetic 40-day forcing: the model's processes against the kernel called directly, the running
sums against a Python sum, the declarations and dataflow, the writes check on a deliberately wrong
wrapper, and the three-rules lint in strict mode on the module file.
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agri_jax.core import lint, registry, run
from agri_jax.core.process import ProcessWriteError, process
from agri_jax.models import catpa_pet_demo as demo
from agri_jax.processes.pet import PETParams, PETSiteParams, shuttleworth_wallace

RTOL = 1e-12 if jax.config.jax_enable_x64 else 2e-5
N = 40


def _params() -> PETSiteParams:
    f = lambda x: jnp.asarray(x, dtype=float)  # noqa: E731
    return PETSiteParams(
        pet=PETParams(f(0.25), f(0.15), f(0.23), f(0.30), f(54.0), f(224.0)),
        elevation=f(200.0),
        latitude=f(0.745163),
        wc13=f(0.255198),
        wc15=f(0.141628),
        wind_height=f(2.0),
        albedo_soil=f(0.13),
        trat=f(1.0),
        rainfall_zone=3,
    )


def _forcing() -> demo.DemoForcing:
    t = np.arange(N, dtype=float)
    doy = 150.0 + t
    lai = np.clip((t - 5.0) / 8.0, 0.0, 4.0)  # bare soil first, then a growing canopy
    return demo.DemoForcing(
        tmin=jnp.asarray(12.0 + 3.0 * np.sin(t)),
        tmax=jnp.asarray(27.0 + 4.0 * np.cos(t)),
        srad=jnp.asarray(20.0 + 5.0 * np.sin(0.3 * t)),
        rh=jnp.asarray(55.0 + 20.0 * np.sin(0.7 * t)),
        wind_run=jnp.asarray(120.0 + 40.0 * np.cos(0.5 * t)),
        doy=jnp.asarray(doy),
        lai=jnp.asarray(lai),
        height_cm=jnp.asarray(40.0 * lai),
        residue_mass=jnp.asarray(np.where(t < 20, 3000.0, 1500.0)),
        residue_age=jnp.asarray(50.0 + t),
    )


def test_declarations_and_dataflow() -> None:
    m = demo.catpa_pet_model()
    assert m.names == ("sw_pet_from_forcing", "accumulate_totals")
    assert registry["sw_pet_from_forcing"] is demo.sw_pet_from_forcing
    assert registry["accumulate_totals"] is demo.accumulate_totals
    assert ("sw_pet_from_forcing", "accumulate_totals", "pet") in m.dataflow()
    assert demo.sw_pet_from_forcing.fortran_name == "POTEVPHR"
    # theta_surface and the totals are true state (read before written); the fluxes are not stale
    assert ("accumulate_totals", "pet") not in m.stale_reads()


def test_run_equals_kernel_loop_on_synthetic_forcing() -> None:
    p, f, s0 = _params(), _forcing(), demo.initial_state(0.2)
    out = run(demo.catpa_pet_model(), p, f, s0)
    tt = te = 0.0
    for d in range(N):
        fd = jax.tree_util.tree_map(lambda x, d=d: x[d], f)
        r = shuttleworth_wallace(
            fd.tmin,
            fd.tmax,
            fd.srad,
            fd.rh,
            fd.wind_run,
            fd.lai,
            fd.height_cm,
            p.pet,
            theta_surface=0.2,
            wc13=p.wc13,
            wc15=p.wc15,
            elevation=p.elevation,
            latitude=p.latitude,
            doy=fd.doy,
            residue_mass=fd.residue_mass,
            residue_age=fd.residue_age,
            rainfall_zone=3,
        )
        tt += float(r.transpiration)
        te += float(r.soil_evaporation) + float(r.residue_evaporation)
        np.testing.assert_allclose(float(out["pet.transpiration"][d]), float(r.transpiration), rtol=RTOL)
        np.testing.assert_allclose(
            float(out["pet.soil_evaporation"][d]), float(r.soil_evaporation), rtol=RTOL
        )
        np.testing.assert_allclose(float(out["totals.transpiration"][d]), tt, rtol=RTOL, atol=1e-30)
        np.testing.assert_allclose(float(out["totals.evaporation"][d]), te, rtol=RTOL)
        assert float(out["totals.days"][d]) == d + 1
    assert float(out["pet.transpiration"][0]) == 0.0 < float(out["pet.transpiration"][-1])


@process(reads=("theta_surface",), writes=("pet",), register=False, source="test: deliberately wrong")
def _wrong(state: demo.DemoState, params: PETSiteParams, forcing_t: demo.DemoForcing) -> demo.DemoState:
    """Correct fluxes, but also changes the undeclared surface water content.

    Source: test only.
    """
    new = demo.sw_pet_from_forcing.fn(state, params, forcing_t)
    return eqx.tree_at(lambda s: s.theta_surface, new, new.theta_surface * 0.5)


def test_writes_check_triggers_on_wrong_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    p, f, s0 = _params(), _forcing(), demo.initial_state(0.2)
    bad = demo.catpa_pet_model().replace("sw_pet_from_forcing", _wrong)
    monkeypatch.delenv("AGRI_JAX_CHECK", raising=False)
    out = run(bad, p, f, s0)  # no check: runs, silently wrong
    assert np.all(np.isfinite(np.asarray(out["pet.transpiration"])))
    monkeypatch.setenv("AGRI_JAX_CHECK", "1")
    with pytest.raises(ProcessWriteError, match="theta_surface"):
        run(bad, p, f, s0)
    with pytest.raises(ProcessWriteError, match="theta_surface"):
        jax.jit(lambda q: run(bad, q, f, s0))(p)
    run(demo.catpa_pet_model(), p, f, s0)  # the real processes pass


def test_module_passes_strict_lint() -> None:
    path = Path(demo.__file__)
    assert lint.main([str(path), "--strict"]) == 0
