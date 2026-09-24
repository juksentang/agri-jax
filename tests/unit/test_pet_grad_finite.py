"""Every ``jnp.where`` branch of the PET kernels has finite gradients on both sides.

A grid of synthetic conditions that exercises each branch of ``shuttleworth_wallace`` (canopy /
no canopy, residue / no residue, corn and soybean residue constants, the "night" rule, negative
net radiation, zero radiation, zero humidity, the 45 km/d wind floor and gale winds, frozen and
hot days, wet and dry surfaces), and the gradient of each flux with respect to every
``PETParams`` field and every continuous input is checked to be finite on every grid point. The
same for the ASCE reference ET and the Priestley-Taylor kernels. No data, no reference model:
this is the differentiability contract of the module (doc 04 section 2.5).
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

from agrijax.processes.pet import (
    PETParams,
    asce_reference_et,
    priestley_taylor,
    shuttleworth_wallace,
)
from agrijax.processes.pet.shuttleworth_wallace import RESIDUE_DENSITY_G_CM3, RESIDUE_DIAMETER_CM

WC13, WC15, ELEVATION, LATITUDE = 0.255198, 0.141628, 200.0, 0.745163  # CA-TPA layer 1 and site
PARAM_FIELDS = (
    "albedo_dry",
    "albedo_wet",
    "albedo_maturity",
    "albedo_residue",
    "soil_resistance",
    "stomatal_resistance",
)


def params() -> PETParams:
    return PETParams(
        albedo_dry=jnp.asarray(0.25),
        albedo_wet=jnp.asarray(0.15),
        albedo_maturity=jnp.asarray(0.23),
        albedo_residue=jnp.asarray(0.31),
        soil_resistance=jnp.asarray(54.0),
        stomatal_resistance=jnp.asarray(224.0),
    )


def grid() -> dict[str, np.ndarray]:
    """Cartesian grid of conditions (about 1.5e4 points) covering every branch."""
    temps = ((-20.0, -10.0), (2.0, 12.0), (15.0, 30.0))
    lai = (0.0, 0.3, 2.5, 6.0)
    height = (0.0, 30.0, 220.0)
    wind = (45.0, 1000.0)
    srad = (0.0, 0.02, 5.0, 28.0)  # 0.02 MJ m-2 d-1 is below the (daily-total) night threshold 0.036
    rh = (0.0, 60.0, 100.0)
    theta = (0.05, 0.2, 0.45)
    residue = (0.0, 2000.0, 12000.0)
    rtype = ("corn", "soybean")
    rows = list(itertools.product(temps, lai, height, wind, srad, rh, theta, residue, rtype))
    return {
        "tmin": np.array([r[0][0] for r in rows]),
        "tmax": np.array([r[0][1] for r in rows]),
        "lai": np.array([r[1] for r in rows]),
        "height": np.array([r[2] for r in rows]),
        "wind": np.array([r[3] for r in rows]),
        "srad": np.array([r[4] for r in rows]),
        "rh": np.array([r[5] for r in rows]),
        "theta": np.array([r[6] for r in rows]),
        "residue": np.array([r[7] for r in rows]),
        "rdia": np.array([RESIDUE_DIAMETER_CM[r[8]] for r in rows]),
        "rho": np.array([RESIDUE_DENSITY_G_CM3[r[8]] for r in rows]),
        "doy": np.array([15 + 37 * (i % 9) for i in range(len(rows))]),
    }


def _sw(p, tmin, tmax, srad, rh, wind, lai, height, theta, rm, rdia, rho, doy):
    return shuttleworth_wallace(
        tmin,
        tmax,
        srad,
        rh,
        wind,
        lai,
        height,
        p,
        theta_surface=theta,
        doy=doy,
        residue_mass=rm,
        residue_age=30.0,
        residue_diameter_cm=rdia,
        residue_density=rho,
        residue_cover_factor=2.5,
        rainfall_zone=3,
        wc13=WC13,
        wc15=WC15,
        elevation=ELEVATION,
        latitude=LATITUDE,
    )


def test_grid_exercises_every_branch() -> None:
    g = grid()
    r = jax.vmap(lambda *a: _sw(params(), *a))(
        g["tmin"],
        g["tmax"],
        g["srad"],
        g["rh"],
        g["wind"],
        g["lai"],
        g["height"],
        g["theta"],
        g["residue"],
        g["rdia"],
        g["rho"],
        g["doy"],
    )
    canopy = (g["lai"] > 0) & (g["height"] > 0)
    assert canopy.any() and (~canopy).any()
    assert (g["residue"] > 0).any() and (g["residue"] == 0).any()
    night = g["srad"] * 1.0e6 / 3.6e3 < 10.0
    assert night.any() and (~night).any()
    rn = np.asarray(r.rn)
    assert (rn > 0).any() and (rn <= 0).any()  # the "RN < 0 -> TCAN RTS / 3" replacement fires
    assert (np.asarray(r.transpiration)[canopy] > 0).any() and np.all(
        np.asarray(r.transpiration)[~canopy] == 0
    )
    assert np.all(np.asarray(r.residue_evaporation)[g["residue"] == 0] == 0)
    for name in ("transpiration", "soil_evaporation", "residue_evaporation", "rn", "raa", "ras", "rsr"):
        assert np.all(np.isfinite(np.asarray(getattr(r, name)))), name


def test_shuttleworth_wallace_gradients_finite_on_the_grid() -> None:
    g = grid()
    args = (
        g["tmin"],
        g["tmax"],
        g["srad"],
        g["rh"],
        g["wind"],
        g["lai"],
        g["height"],
        g["theta"],
        g["residue"],
        g["rdia"],
        g["rho"],
    )
    p = params()
    for k, name in enumerate(("transpiration", "soil_evaporation", "residue_evaporation")):

        def scalar(pp, *a, _k=k):
            r = _sw(pp, *a)
            return (r.transpiration, r.soil_evaporation, r.residue_evaporation)[_k]

        axes = (None, *([0] * (len(args) + 1)))
        grads = jax.jit(jax.vmap(jax.grad(scalar, argnums=tuple(range(len(args) + 1))), in_axes=axes))(
            p, *args, g["doy"]
        )
        gp, gin = grads[0], grads[1:]
        for field in PARAM_FIELDS:
            v = np.asarray(getattr(gp, field))
            assert v.shape == (len(g["doy"]),) and np.all(np.isfinite(v)), (name, field)
        for j, v in enumerate(gin):
            assert np.all(np.isfinite(np.asarray(v))), (name, j)
    # the gradients are not trivially zero: transpiration responds to radiation on canopy days
    canopy = (g["lai"] > 0) & (g["height"] > 0) & (g["srad"] > 1.0)
    g_srad = jax.vmap(jax.grad(lambda s, *a: _sw(p, a[0], a[1], s, *a[2:]).transpiration))(
        g["srad"],
        g["tmin"],
        g["tmax"],
        g["rh"],
        g["wind"],
        g["lai"],
        g["height"],
        g["theta"],
        g["residue"],
        g["rdia"],
        g["rho"],
        g["doy"],
    )
    assert (np.asarray(g_srad)[canopy] > 0).mean() > 0.9


def test_reference_et_and_priestley_taylor_gradients_finite() -> None:
    tmax = np.array([-5.0, 20.0, 40.0] * 8)
    tmin = tmax - 10.0
    srad = np.array([0.0, 20.0] * 12)
    lai = np.array([0.0, 0.0, 2.0, 5.0] * 6)
    rh = np.array([0.0, 50.0, 100.0] * 8)
    wind = np.array([0.5, 5.0] * 12)
    doy = np.arange(1, 25) * 15

    def et(a, b, s, h, u, d):
        r = asce_reference_et(a, b, s, h, u, elevation=200.0, latitude=0.745163, doy=d, wind_height=2.0)
        return r.et_tall + r.et_short

    g = jax.vmap(jax.grad(et, argnums=(0, 1, 2, 3, 4)))(tmin, tmax, srad, rh, wind, doy)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g)

    def pt(s, a, b, l_, alb):
        return priestley_taylor(s, a, b, l_, alb)

    g2 = jax.vmap(jax.grad(pt, argnums=(0, 1, 2, 3, 4)))(srad, tmax, tmin, lai, np.full(24, 0.13))
    assert all(np.all(np.isfinite(np.asarray(v))) for v in g2)
