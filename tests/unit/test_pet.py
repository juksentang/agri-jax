"""Potential ET: Shuttleworth-Wallace (RZWQM2), ASCE reference ET, Priestley-Taylor (DSSAT).

Pure-function properties (energy closure, limits, non-negativity, finite-difference gradients,
vmap consistency) and ASCE against ``pyet``. The comparison with the RZWQM2 ``.ana`` output of
the CA-TPA reference run is in ``tests/integration/test_pet_oracle.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agri_jax.processes.pet import (
    PETParams,
    asce_reference_et,
    clear_sky_radiation,
    energy_constants,
    priestley_taylor,
    shuttleworth_wallace,
    soil_albedo,
    wind_adjustment,
    wind_to_2m,
)

# CA-TPA site (rzwqm.dat): elevation m, latitude rad, layer-1 1/3-bar and 15-bar water contents
CATPA_ELEV = 200.0
CATPA_LAT = 0.745163
CATPA_WC13 = 0.255198
CATPA_WC15 = 0.141628


def params(**kw) -> PETParams:
    base = dict(
        albedo_dry=0.25,
        albedo_wet=0.15,
        albedo_maturity=0.23,
        albedo_residue=0.30,
        soil_resistance=54.0,
        stomatal_resistance=224.0,
    )
    base.update(kw)
    return PETParams(**{k: jnp.asarray(v, dtype=float) for k, v in base.items()})


def site() -> dict:
    return dict(
        wc13=CATPA_WC13,
        wc15=CATPA_WC15,
        elevation=CATPA_ELEV,
        latitude=CATPA_LAT,
        rainfall_zone=3,
    )


def summer_day(**kw) -> dict:
    d = dict(tmin=15.0, tmax=28.0, srad=22.0, rh=60.0, wind_run=150.0, lai=3.0, height_cm=150.0)
    d.update(kw)
    return d


# --------------------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------------------
def test_energy_constants_reasonable() -> None:
    ec = energy_constants(15.0, 28.0, 60.0, 200.0)
    assert 1.5 < float(ec.ea) < 3.0  # mean e_sat at 15 / 28 degC ~ 2.7 kPa
    assert float(ec.ed) == pytest.approx(0.6 * float(ec.ea))
    assert 98.0 < float(ec.pressure) < 100.0  # 200 m elevation
    assert 1.1 < float(ec.rho_air) < 1.25
    assert 2.44 < float(ec.latent_heat) < 2.47
    assert 0.064 < float(ec.gamma) < 0.068


def test_clear_sky_radiation_bounds() -> None:
    doy = jnp.arange(1, 366)
    csr = clear_sky_radiation(doy, CATPA_LAT)
    assert bool(jnp.all(csr.total > 0.0))
    assert bool(jnp.all(csr.direct <= csr.extraterrestrial))
    assert bool(jnp.all(csr.total < csr.extraterrestrial))
    # summer solstice at 42.7 N: ~40 MJ m-2 d-1 top of atmosphere, ~30 clear sky
    assert 39.0 < float(csr.extraterrestrial[171]) < 43.0
    assert 26.0 < float(csr.total[171]) < 34.0
    assert float(csr.total[171]) > float(csr.total[354])
    assert 14.0 < float(csr.sunset_hour[171]) - float(csr.sunrise_hour[171]) < 16.0


def test_soil_albedo_clips_between_wet_and_dry() -> None:
    a = soil_albedo(jnp.array([0.05, CATPA_WC15, 0.2, CATPA_WC13, 0.4]), CATPA_WC13, CATPA_WC15, 0.25, 0.15)
    assert float(a[0]) == pytest.approx(0.25)
    assert float(a[1]) == pytest.approx(0.25)
    assert float(a[3]) == pytest.approx(0.15)
    assert float(a[4]) == pytest.approx(0.15)
    assert 0.15 < float(a[2]) < 0.25
    assert float(soil_albedo(0.05, CATPA_WC13, CATPA_WC15, 0.25, 0.15, crust=1.0)) == pytest.approx(0.5)


def test_wind_adjustment_bare_and_canopy() -> None:
    bare = wind_adjustment(100.0, 0.0, 2.0)
    assert float(bare.reference_height) == pytest.approx(2.0)
    assert float(bare.factor) == pytest.approx(np.log(200.0) / np.log(1.93 / 0.0123))
    crop = wind_adjustment(100.0, 150.0, 2.0)
    assert float(crop.reference_height) == pytest.approx(1.93 + 1.0)
    assert float(crop.factor) == pytest.approx(np.log(1.93 / (0.123 * 1.5)) / np.log(1.93 / 0.0123))
    assert float(crop.wind_run) < float(bare.wind_run)


# --------------------------------------------------------------------------------------
# Priestley-Taylor (DSSAT PETPT) against a hand computation
# --------------------------------------------------------------------------------------
def test_priestley_taylor_hand_computation(tol: float) -> None:
    srad, tmax, tmin, lai, msalb = 20.0, 30.0, 15.0, 2.0, 0.13
    td = 0.6 * 30.0 + 0.4 * 15.0  # 24.0
    albedo = 0.23 - (0.23 - 0.13) * np.exp(-0.75 * 2.0)
    slang = 20.0 * 23.923
    eeq = slang * (2.04e-4 - 1.83e-4 * albedo) * (td + 29.0)
    assert float(priestley_taylor(srad, tmax, tmin, lai, msalb)) == pytest.approx(1.1 * eeq, rel=tol)
    # hot day: factor (Tmax - 35) 0.05 + 1.1
    eeq_hot = slang * (2.04e-4 - 1.83e-4 * albedo) * (0.6 * 38.0 + 0.4 * 15.0 + 29.0)
    assert float(priestley_taylor(srad, 38.0, tmin, lai, msalb)) == pytest.approx(
        eeq_hot * (3.0 * 0.05 + 1.1), rel=tol
    )
    # cold day: 0.01 exp(0.18 (Tmax + 20))
    eeq_cold = slang * (2.04e-4 - 1.83e-4 * albedo) * (0.6 * 2.0 + 0.4 * (-5.0) + 29.0)
    assert float(priestley_taylor(srad, 2.0, -5.0, lai, msalb)) == pytest.approx(
        eeq_cold * 0.01 * np.exp(0.18 * 22.0), rel=tol
    )
    # bare soil uses the soil albedo
    eeq_bare = slang * (2.04e-4 - 1.83e-4 * 0.13) * (td + 29.0)
    assert float(priestley_taylor(srad, tmax, tmin, 0.0, msalb)) == pytest.approx(1.1 * eeq_bare, rel=tol)
    assert float(priestley_taylor(0.0, tmax, tmin, lai, msalb)) == pytest.approx(1e-4)


# --------------------------------------------------------------------------------------
# ASCE reference ET against pyet
# --------------------------------------------------------------------------------------
def test_asce_reference_et_matches_pyet() -> None:
    pyet = pytest.importorskip("pyet")
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    n = 30
    tmin = rng.uniform(-5.0, 20.0, n)
    tmax = tmin + rng.uniform(3.0, 15.0, n)
    srad = rng.uniform(2.0, 30.0, n)
    rh = rng.uniform(30.0, 95.0, n)
    wind = rng.uniform(0.3, 6.0, n)
    doy = rng.integers(1, 366, n)
    idx = pd.to_datetime([f"2015-{int(d):03d}" for d in doy], format="%Y-%j")
    u2 = np.asarray(wind_to_2m(wind, 2.0))
    kw = dict(
        rs=pd.Series(srad, idx),
        tmax=pd.Series(tmax, idx),
        tmin=pd.Series(tmin, idx),
        rh=pd.Series(rh, idx),
        elevation=CATPA_ELEV,
        lat=CATPA_LAT,
        clip_zero=False,
    )
    tmean = pd.Series(0.5 * (tmin + tmax), idx)
    ref_short = pyet.pm_asce(tmean, pd.Series(u2, idx), etype="os", **kw).values
    ref_tall = pyet.pm_asce(tmean, pd.Series(u2, idx), etype="rs", **kw).values
    mine = asce_reference_et(tmin, tmax, srad, rh, wind, elevation=CATPA_ELEV, latitude=CATPA_LAT, doy=doy)
    assert np.abs(np.asarray(mine.et_short) - ref_short).max() < 1e-3
    assert np.abs(np.asarray(mine.et_tall) - ref_tall).max() < 1e-3
    assert bool(jnp.all(mine.et_tall > mine.et_short))


def test_asce_reference_et_vmap_and_grad() -> None:
    def f(w):
        return asce_reference_et(10.0, 25.0, 20.0, 50.0, w, elevation=200.0, latitude=0.7, doy=180).et_short

    g = jax.grad(f)(2.0)
    assert np.isfinite(float(g)) and float(g) > 0.0  # more wind, more ET
    batch = jax.vmap(f)(jnp.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.asarray(batch), [float(f(w)) for w in (1.0, 2.0, 3.0)])


# --------------------------------------------------------------------------------------
# Shuttleworth-Wallace properties
# --------------------------------------------------------------------------------------
def test_sw_energy_closure_components_sum_to_total(x64_enabled: bool) -> None:
    """The component fluxes evaluated with the source-height deficit D0 sum to the S-W total."""
    p = params()
    rel = 1e-8 if x64_enabled else 1e-4
    for kw in (
        summer_day(),
        summer_day(lai=0.5, height_cm=30.0, residue_mass=3000.0),
        summer_day(lai=0.0, height_cm=0.0, residue_mass=5000.0),
        summer_day(rh=90.0, wind_run=400.0, lai=6.0, height_cm=250.0),
    ):
        r = shuttleworth_wallace(**kw, params=p, theta_surface=0.2, doy=180, **site())
        total_cm = float(r.transpiration + r.soil_evaporation + r.residue_evaporation)
        assert total_cm == pytest.approx(float(r.latent_total) / (10.0 * float(r.latent_heat)), rel=rel)


def test_sw_energy_closure_no_deficit_bounded_by_net_radiation() -> None:
    """With zero atmospheric vapour pressure deficit the total latent heat cannot exceed Rn - G."""
    p = params()
    rng = np.random.default_rng(1)
    for _ in range(20):
        kw = summer_day(
            tmin=rng.uniform(5, 20),
            tmax=rng.uniform(21, 35),
            srad=rng.uniform(5, 30),
            rh=100.0,
            wind_run=rng.uniform(50, 500),
            lai=rng.uniform(0, 6),
            height_cm=rng.uniform(5, 250),
            residue_mass=rng.uniform(0, 8000),
        )
        r = shuttleworth_wallace(**kw, params=p, theta_surface=0.2, doy=int(rng.integers(120, 270)), **site())
        lam_total = (
            float(r.transpiration + r.soil_evaporation + r.residue_evaporation) * 10.0 * float(r.latent_heat)
        )
        assert lam_total <= float(r.rn) + 1e-9
        assert float(r.rns + r.rnr) <= float(r.rn) + 1e-9


def test_sw_limits() -> None:
    p = params()
    bare = shuttleworth_wallace(
        **summer_day(lai=0.0, height_cm=0.0), params=p, theta_surface=0.2, doy=180, **site()
    )
    assert float(bare.transpiration) == 0.0
    assert float(bare.soil_evaporation) > 0.1
    assert float(bare.residue_evaporation) == 0.0
    assert float(bare.soil_fraction) == 1.0
    assert float(bare.canopy_fraction) == 0.0
    full = shuttleworth_wallace(
        **summer_day(lai=8.0, height_cm=250.0), params=p, theta_surface=0.2, doy=180, tlai=8.0, **site()
    )
    assert float(full.transpiration) > 0.3
    # under a closed canopy the soil term is a small fraction of the field ET (the S-W aerodynamic
    # term through ras keeps it above zero; 13 % here) and far below bare-soil evaporation
    assert float(full.soil_evaporation) < 0.2 * float(full.transpiration)
    assert float(full.soil_evaporation) < 0.3 * float(bare.soil_evaporation)
    assert float(full.canopy_fraction) > 0.99
    # a residue layer reduces soil evaporation and adds residue evaporation
    res = shuttleworth_wallace(
        **summer_day(lai=0.0, height_cm=0.0, residue_mass=5000.0),
        params=p,
        theta_surface=0.2,
        doy=180,
        **site(),
    )
    assert 0.0 < float(res.soil_fraction) < 1.0
    assert float(res.residue_evaporation) > 0.0
    assert float(res.soil_evaporation) < float(bare.soil_evaporation)
    # a wetter surface has a lower albedo and hence higher soil evaporation
    wet = shuttleworth_wallace(
        **summer_day(lai=0.0, height_cm=0.0), params=p, theta_surface=0.4, doy=180, **site()
    )
    assert float(wet.albedo_soil) < float(bare.albedo_soil)
    assert float(wet.soil_evaporation) > float(bare.soil_evaporation)


def test_sw_non_negative_and_finite_over_random_inputs(x64_enabled: bool) -> None:
    p = params()
    vm_rel = 1e-9 if x64_enabled else 1e-5
    rng = np.random.default_rng(2)
    n = 200
    tmin = rng.uniform(-20.0, 25.0, n)
    tmax = tmin + rng.uniform(0.5, 18.0, n)
    srad = rng.uniform(0.0, 32.0, n)
    rh = rng.uniform(10.0, 100.0, n)
    wind = rng.uniform(5.0, 1000.0, n)
    lai = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.uniform(0.0, 7.0, n))
    height = np.where(lai > 0, rng.uniform(1.0, 300.0, n), 0.0)
    rm = np.where(rng.uniform(size=n) < 0.3, 0.0, rng.uniform(0.0, 12000.0, n))
    theta = rng.uniform(0.05, 0.45, n)
    doy = rng.integers(1, 366, n)

    def one(a, b, c, d, e, l, h, m, th, dd):
        return shuttleworth_wallace(
            a, b, c, d, e, l, h, p, theta_surface=th, doy=dd, residue_mass=m, **site()
        )

    r = jax.vmap(one)(tmin, tmax, srad, rh, wind, lai, height, rm, theta, doy)
    for name in (
        "transpiration",
        "soil_evaporation",
        "residue_evaporation",
        "rn",
        "raa",
        "ras",
        "rac",
        "rsc",
        "rsr",
    ):
        v = np.asarray(getattr(r, name))
        assert np.all(np.isfinite(v)), name
    for name in ("transpiration", "soil_evaporation", "residue_evaporation", "raa", "ras", "rsr"):
        assert np.all(np.asarray(getattr(r, name)) >= 0.0), name
    assert np.all(np.asarray(r.transpiration + r.soil_evaporation + r.residue_evaporation) < 2.5)  # < 25 mm/d
    # vmap agrees with scalar evaluation
    for i in (0, 17, 99):
        s = one(tmin[i], tmax[i], srad[i], rh[i], wind[i], lai[i], height[i], rm[i], theta[i], doy[i])
        assert float(s.transpiration) == pytest.approx(float(r.transpiration[i]), rel=vm_rel, abs=1e-12)
        assert float(s.soil_evaporation) == pytest.approx(float(r.soil_evaporation[i]), rel=vm_rel, abs=1e-12)


@pytest.mark.parametrize("case", ["canopy", "sparse_with_residue", "bare_with_residue"])
def test_sw_gradients_wrt_all_parameters_finite_difference(case: str, x64_enabled: bool) -> None:
    kw = {
        "canopy": summer_day(),
        "sparse_with_residue": summer_day(lai=0.8, height_cm=40.0, residue_mass=2500.0),
        "bare_with_residue": summer_day(lai=0.0, height_cm=0.0, residue_mass=6000.0),
    }[case]
    p = params()

    def total(pp: PETParams) -> jax.Array:
        r = shuttleworth_wallace(**kw, params=pp, theta_surface=0.2, doy=180, **site())
        return r.transpiration + 2.0 * r.soil_evaporation + 3.0 * r.residue_evaporation

    grads = jax.grad(total)(p)
    rel = 1e-5 if x64_enabled else 5e-2
    eps = 1e-6 if x64_enabled else 1e-2
    for name in (
        "albedo_dry",
        "albedo_wet",
        "albedo_maturity",
        "albedo_residue",
        "soil_resistance",
        "stomatal_resistance",
    ):
        g = float(getattr(grads, name))
        assert np.isfinite(g), name
        h = eps * max(1.0, abs(float(getattr(p, name))))
        plus = total(p.replace(**{name: getattr(p, name) + h}))
        minus = total(p.replace(**{name: getattr(p, name) - h}))
        fd = float((plus - minus) / (2.0 * h))
        assert g == pytest.approx(fd, rel=rel, abs=1e-8 if x64_enabled else 1e-3), (case, name)
    # gradients wrt forcing / state inputs are finite too
    g_in = jax.grad(
        lambda th, l, w: jnp.sum(
            shuttleworth_wallace(
                **{**kw, "lai": l, "wind_run": w}, params=p, theta_surface=th, doy=180, **site()
            ).soil_evaporation
        ),
        argnums=(0, 1, 2),
    )(0.2, kw["lai"], kw["wind_run"])
    assert all(np.isfinite(float(v)) for v in g_in)


def test_sw_parameter_sensitivity_signs() -> None:
    p = params()
    base = shuttleworth_wallace(**summer_day(), params=p, theta_surface=0.2, doy=180, **site())
    more_rs = shuttleworth_wallace(
        **summer_day(), params=params(stomatal_resistance=400.0), theta_surface=0.2, doy=180, **site()
    )
    assert float(more_rs.transpiration) < float(base.transpiration)
    more_rss = shuttleworth_wallace(
        **summer_day(), params=params(soil_resistance=500.0), theta_surface=0.2, doy=180, **site()
    )
    assert float(more_rss.soil_evaporation) < float(base.soil_evaporation)
    brighter = shuttleworth_wallace(
        **summer_day(), params=params(albedo_maturity=0.4), theta_surface=0.2, doy=180, **site()
    )
    assert float(brighter.rn) < float(base.rn)
