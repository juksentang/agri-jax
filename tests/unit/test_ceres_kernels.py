"""The small CERES-Maize kernels against values computed by hand from the DSSAT-CSM equations.

Each expected value is the Fortran statement of DSSAT-CSM v4.8.6.0 (``MZ_GROSUB.for``,
``MZ_PHENOL.for``, ``MZ_ROOTS.for``; BSD-3) evaluated in plain Python with the literal numbers
of the Fortran, not with the coefficient objects, so a wrong hoisted default or a changed
operation shows up here. Inputs are chosen to visit every branch of each kernel.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.processes.crop.ceres_maize import DSSAT_COEFFICIENTS, CeresCultivar, CeresSpecies
from agrijax.processes.crop.ceres_maize.growth import (
    LeafAppearance,
    OrganGrowth,
    assimilation,
    canopy_height,
    crop_failure,
    ear_growth_fraction,
    early_maturity,
    floral_induction_growth,
    grain_fill_growth,
    grain_fill_rate,
    juvenile_growth,
    leaf_appearance,
    leaf_senescence,
    select_stage_block,
    silk_efg_growth,
    stage3_demand,
    stage3_partition,
)
from agrijax.processes.crop.ceres_maize.phenology import (
    grain_number,
    growing_point_thermal_time,
    hourly_thermal_time,
    leaf_number_at_ti,
    photoperiod_rate,
)
from agrijax.processes.crop.ceres_maize.roots import (
    emergence_rlv,
    root_front_advance,
    root_length_growth,
    root_water_deficit,
    waterlogging_survival,
)

from .test_ceres_phenology import CUL, SPE, a

X64 = jax.config.jax_enable_x64
REL = 1e-12 if X64 else 2e-6
GC = DSSAT_COEFFICIENTS.grosub
PC = DSSAT_COEFFICIENTS.phenol
RC = DSSAT_COEFFICIENTS.roots
SPECIES = CeresSpecies(**{k: a(v) for k, v in SPE.items()})
CULTIVAR = CeresCultivar(**{k: a(v) for k, v in CUL.items()})


def close(got, want, rel=REL, abs_=None):
    np.testing.assert_allclose(
        np.asarray(got, dtype=float), want, rtol=rel, atol=abs_ if abs_ is not None else rel
    )


def organs(pla=0.0, lfwt=0.0, stmwt=0.0, earwt=0.0, grort=0.0, slan=0.0, cls=0.0) -> OrganGrowth:
    return OrganGrowth(*(a(x) for x in (pla, lfwt, stmwt, earwt, grort, slan, cls)))


# ------------------------------------------------------------------ MZ_GROSUB
def test_assimilation_by_hand():
    lifac = 1.5 - 0.768 * ((75.0 * 0.01) ** 2 * 7.0) ** 0.1
    pcarb = 20.0 * 0.5 / 7.0 * (1.0 - math.exp(-lifac * 2.0)) * 4.2 * 1.0  # PCO2(330) = 1.00
    # TAVGD = 0.25*20 + 0.75*30 = 27.5: PRFT = 1, min(PRFT, SWFAC) = 0.8
    got = assimilation(20.0, 30.0, 20.0, 330.0, a(2.0), a(7.0), a(75.0), a(0.8), 0.9, 4.2, SPECIES, GC)
    close(got, pcarb * 0.8 * 0.9)
    # TAVGD = 0.25*36 + 0.75*40 = 39 on the falling limb: PRFT = 1 - (39 - 33) / (44 - 33)
    got = assimilation(20.0, 40.0, 36.0, 330.0, a(2.0), a(7.0), a(75.0), a(1.0), 1.0, 4.2, SPECIES, GC)
    close(got, pcarb * (1.0 - 6.0 / 11.0))
    got = assimilation(20.0, 30.0, 20.0, 330.0, a(2.0), a(0.0), a(75.0), a(1.0), 1.0, 4.2, SPECIES, GC)
    close(got, 0.0)  # no plants: IPAR = 0


def test_leaf_appearance_by_hand():
    la = leaf_appearance(a(2.0), a(20.0), 40.0, a(100.0), a(150.0), GC)
    ti = 20.0 / (40.0 * (0.66 + 0.068 * 2.0))
    close(la.ti, ti)
    close(la.cumph, 2.0 + ti)
    close(la.xn, 3.0 + ti)
    close(la.cumph3, 2.0)  # SUMDTT = 100 > P3 - 2 PHINT = 70: CUMPH = CUMPH - TI
    la = leaf_appearance(a(6.0), a(20.0), 40.0, a(10.0), a(150.0), GC)
    close(la.ti, 0.5)  # PC = 1 from 5 phyllochrons on
    close(la.cumph3, 6.5)
    close(la.xn3, 7.5)


def _la(ti, xn, xn3=None):
    x3 = xn if xn3 is None else xn3
    return LeafAppearance(ti=a(ti), cumph=a(xn - 1.0), xn=a(xn), cumph3=a(x3 - 1.0), xn3=a(x3))


def test_juvenile_growth_branches():
    # enough assimilate: XLFWT = max((PLA/250)^1.25, LFWT) = LFWT, GROLF = 0, SLAN kept
    b, seed = juvenile_growth(
        organs(pla=20.0, lfwt=0.5, slan=0.3), a(1.0), _la(0.5, 5.0), a(1.0), a(0.2), a(50.0), a(7.0), GC
    )
    close(b.pla, 20.0 + 3.0 * 5.0 * 5.0 * 0.5)
    close(b.grort, 1.0)
    close(b.slan, 0.3)
    close(b.lfwt, 0.5 - 0.3 / 600.0)
    close(b.cls, 0.3 / 600.0 * 7.0 * 10.0)
    close(seed, 0.2)
    # roots at their 25 % floor, drawing on the seed reserve (not exhausted)
    pla = 200.0 + 37.5
    grolf = (pla / 250.0) ** 1.25 - 0.3
    b, seed = juvenile_growth(
        organs(pla=200.0, lfwt=0.3), a(0.5), _la(0.5, 5.0), a(1.0), a(1.0), a(50.0), a(7.0), GC
    )
    close(b.grort, 0.125)
    close(seed, 1.0 + 0.5 - grolf - 0.125)
    close(b.pla, pla)
    close(b.slan, 50.0 * pla / 10000.0)
    close(b.lfwt, 0.3 + grolf - 50.0 * pla / 10000.0 / 600.0)
    # the seed reserve runs out: GROLF = 0.75 CARBO, PLA = (LFWT + GROLF)^0.8 267
    b, seed = juvenile_growth(
        organs(pla=200.0, lfwt=0.3), a(0.5), _la(0.5, 5.0), a(1.0), a(0.2), a(50.0), a(7.0), GC
    )
    close(seed, 0.0)
    pla = (0.3 + 0.375) ** 0.8 * 267.0
    close(b.pla, pla)
    close(b.lfwt, 0.675 - 50.0 * pla / 10000.0 / 600.0)
    # XN < 4: PLAG = 4 XN TI
    b, _ = juvenile_growth(
        organs(pla=2.0, lfwt=0.5), a(1.0), _la(0.5, 3.0), a(0.5), a(0.2), a(0.0), a(7.0), GC
    )
    close(b.pla, 2.0 + 4.0 * 3.0 * 0.5 * 0.5)


def test_floral_induction_growth_capped():
    b = floral_induction_growth(
        organs(pla=300.0, lfwt=1.0), a(0.2), _la(0.5, 8.0), a(1.0), a(40.0), a(7.0), GC
    )
    # PLA = 300 + 3.5 x 64 x 0.5 = 412 -> GROLF = (412/267)^1.25 - 1 > 0.75 CARBO: capped at 0.15
    pla = (1.0 + 0.15) ** 0.8 * 267.0
    close(b.pla, pla)
    close(b.grort, 0.2 - 0.15)
    close(b.slan, 40.0 * pla / 10000.0)
    close(b.lfwt, 1.15 - 40.0 * pla / 10000.0 / 600.0)
    b = floral_induction_growth(
        organs(pla=300.0, lfwt=1.0), a(5.0), _la(0.5, 8.0), a(1.0), a(40.0), a(7.0), GC
    )
    close(b.pla, 412.0)
    close(b.grort, 5.0 - ((412.0 / 267.0) ** 1.25 - 1.0))


def test_ear_growth_fraction_by_hand():
    close(ear_growth_fraction(a(210.0), GC), 0.405)
    close(ear_growth_fraction(a(0.0), GC), 0.81 / (1.0 + math.exp(-0.02 * (0.0 - 210.0))))


def test_stage3_demand_three_branches():
    pla = 400.0
    for xn, tlno, plag, late in [
        (10.0, 20.0, 3.5 * 10.0 * 10.0 * 0.5 * 0.9, False),  # XN < 12
        (14.0, 20.0, 3.5 * 170.0 * 0.5 * 0.9, False),  # XN < TLNO - 2.9999
        (18.0, 20.0, 170.0 * 3.5 / (18.0 + 3.0 - 20.0) ** 0.5 * 0.5 * 0.9, True),
    ]:
        grolf, grostm = stage3_demand(_la(0.5, xn, xn), a(0.9), a(tlno), a(8.0), a(pla), GC)
        g = 0.00116 * plag * pla**0.25
        close(grolf, g)
        close(grostm, 3.0 * 3.1 * 0.5 * 0.9 if late else g * 0.0182 * (xn - 8.0) ** 2)


def test_stage3_partition_branches():
    lf, st, ea, rt = stage3_partition(a(2.0), a(0.3), a(0.5), a(0.2), a(1.0), GC)  # stem covers the ear
    close([lf, st, ea, rt], [0.3, 0.3, 0.2, 1.2])
    lf, st, ea, rt = stage3_partition(a(2.0), a(0.3), a(0.1), a(0.3), a(1.0), GC)  # ear > stem: halves
    close([lf, st, ea, rt], [0.3, 0.05, 0.05, 1.6])
    # roots at or below 10 %: shoot scaled to 90 % of CARBO (GRF = 0.9 / 1.0)
    lf, st, ea, rt = stage3_partition(a(1.0), a(0.5), a(0.5), a(0.1), a(1.0), GC)
    close([lf, st, ea, rt], [0.45, 0.36, 0.09, 0.1])
    lf, st, ea, rt = stage3_partition(a(1.0), a(0.5), a(0.5), a(0.1), a(0.0), GC)  # TURFAC = 0: no scaling
    close([lf, st, ea, rt], [0.5, 0.4, 0.1, 0.0], abs_=1e-12 if X64 else 1e-6)


def test_silk_efg_growth_by_hand():
    org = organs(pla=500.0, lfwt=3.0, stmwt=20.0, earwt=1.0)
    b, cum, sump = silk_efg_growth(
        org, a(2.0), a(1.0), a(10.0), a(100.0), a(200.0), a(4.0), a(7.0), a(1.5), GC
    )
    close(cum, 210.0)
    close(sump, 6.0)
    close(b.earwt, 1.0 + 0.81)  # 0.405 x CARBO
    close(b.grort, 0.16)
    close(b.stmwt, 20.0 + 2.0 - 0.81 - 0.16)
    slan = 500.0 * (0.05 + 100.0 / 200.0 * 0.05)
    close(b.slan, slan)
    close(b.lfwt, 3.0 - slan / 600.0)
    close(b.cls, slan / 600.0 * 7.0 * 10.0 + 1.5)
    b, _, _ = silk_efg_growth(org, a(0.0), a(1.0), a(10.0), a(100.0), a(200.0), a(4.0), a(7.0), a(1.5), GC)
    close([b.grort, b.stmwt], [0.0, 20.0])  # no assimilate: (CARBO - GROEAR) x 0.5 each


def test_grain_fill_rate_by_hand():
    rgfill, grogrn = grain_fill_rate(a(20.0), a(0.5), a(500.0), 8.0, SPECIES.rgfil, GC)
    close([rgfill, grogrn], [1.0, 500.0 * 8.0 * 0.001 * (0.45 + 0.55 * 0.5)])
    rgfill, _ = grain_fill_rate(a(30.0), a(1.0), a(500.0), 8.0, SPECIES.rgfil, GC)
    close(rgfill, 1.0 - (30.0 - 27.0) / (35.0 - 27.0))


def test_early_maturity_counters():
    i = lambda x: jnp.asarray(x, dtype=jnp.int32)  # noqa: E731
    act = jnp.asarray([True, True, True, False, False])
    rg = a([0.5, 0.05, 0.05, 0.5, 0.5])
    m = early_maturity(act, rg, i([3, 2, 5, 4, 4]), i([0, 0, 0, 6, 2]), SPECIES)
    np.testing.assert_array_equal(np.asarray(m.emat), [0, 3, 0, 0, 4])  # EMAT 6 > RSGRT = 5: reset
    np.testing.assert_array_equal(np.asarray(m.cmat), [0, 0, 0, 7, 3])
    np.testing.assert_array_equal(np.asarray(m.early), [False, False, True, True, False])
    np.testing.assert_array_equal(np.asarray(m.early_idle), [False, False, False, True, False])


def test_grain_fill_growth_surplus_and_depleted_stem():
    i0 = jnp.asarray(0, dtype=jnp.int32)
    args: dict[str, Any] = dict(
        tempm=a(20.0),
        swfac=a(0.5),
        sumdtt=a(400.0),
        gpp=a(500.0),
        grnwt=a(10.0),
        grogrn_prev=a(0.0),
        emat=i0,
        cmat=i0,
        cul_g3=8.0,
        cul_p5=800.0,
        spe=SPECIES,
        c=GC,
    )
    pot = 500.0 * 8.0 * 0.001 * (0.45 + 0.55 * 0.5)  # 2.9
    org = organs(pla=600.0, lfwt=20.0, stmwt=10.0, earwt=30.0, grort=0.3, slan=5.0)
    b, gf = grain_fill_growth(org, a(5.0), swmin=a(9.5), swmax=a(11.0), **args)
    close(gf.grogrn, pot)
    close(gf.grnwt, 10.0 + pot)
    close(b.earwt, 30.0 + pot)
    close(b.grort, (5.0 - pot) * 0.5)
    close(b.stmwt, min(10.0 + (5.0 - pot) * 0.5, 11.0))
    close(b.slan, 600.0 * (0.1 + 0.6 * (400.0 / 800.0) ** 3))
    # deficit: STMWT + CARBO - GROGRN = 8.1 <= 1.07 SWMIN, + 0.005 LFWT = 8.2 < SWMIN -> SWMIN, GROGRN = CARBO
    b, gf = grain_fill_growth(org, a(1.0), swmin=a(9.5), swmax=a(11.0), **args)
    close([b.stmwt, gf.grogrn, b.grort], [9.5, 1.0, 0.0])


def test_select_stage_block_picks_each_stage():
    blocks = tuple(organs(pla=k, lfwt=10.0 * k) for k in (1.0, 2.0, 3.0, 4.0, 5.0))
    keep = organs(pla=-1.0, lfwt=-10.0)
    s = jnp.asarray([1, 3, 5, 6])
    got = select_stage_block([s == k for k in range(1, 6)], blocks, keep)
    close(got.pla, [1.0, 3.0, 5.0, -1.0])
    close(got.lfwt, [10.0, 30.0, 50.0, -10.0])


def test_leaf_senescence_by_hand():
    senla, lai = leaf_senescence(a(5000.0), a(1000.0), a(200.0), a(5.0), a(0.5), a(4.0), a(7.0), 0.05, GC)
    slf = min(0.95 + 0.05 * 0.5, 1.0 - 0.008 * (5.0 - 4.0), max(0.0, 1.0 - 0.01 * (4.0 - 6.0) ** 2))
    s = 1000.0 + (5000.0 - 1000.0) * (1.0 - slf)
    close([senla, lai], [s, (5000.0 - s) * 7.0 * 0.0001])
    senla, _ = leaf_senescence(a(5000.0), a(1000.0), a(3000.0), a(3.0), a(1.0), a(20.0), a(7.0), 0.05, GC)
    close(senla, 3000.0)  # no stress: SENLA = max(SENLA, SLAN)


def test_crop_failure_cold_and_drought():
    i = lambda x: jnp.asarray(x, dtype=jnp.int32)  # noqa: E731
    grow = jnp.asarray([True, True, True, True, False])
    f = crop_failure(
        grow, i([3, 3, 2, 2, 3]), i([-99] * 5), i([0] * 5), i(2001200), i([5, 5, 5, 5, 5]),
        a([0.0, 1.0, 0.05, 0.05, 0.0]), a([2.0, 2.0, 20.0, 20.0, 2.0]), a([1.0, 1.0, 0.05, 0.05, 1.0]),
        i([6, 14, 0, 0, 6]), i([0, 0, 10, 9, 0]), 6.0, 15.0, GC,
    )  # fmt: skip
    np.testing.assert_array_equal(np.asarray(f.icold), [7, 15, 0, 0, 6])
    np.testing.assert_array_equal(np.asarray(f.nwsd), [0, 0, 11, 10, 0])
    np.testing.assert_array_equal(np.asarray(f.istage), [6, 6, 6, 2, 3])
    np.testing.assert_array_equal(np.asarray(f.status), [32, 32, 33, 0, 0])
    np.testing.assert_array_equal(np.asarray(f.mdate), [2001200, 2001200, 2001200, -99, -99])


def test_canopy_height_by_hand():
    close(canopy_height(a(3.0), a(7.0), 1.6, GC), 3.0 / (0.4238 * 7.0 + 0.3424) * 1.6)
    close(canopy_height(a(10.0), a(7.0), 1.6, GC), 1.6)


# ------------------------------------------------------------------ MZ_PHENOL
def test_photoperiod_rate_and_leaf_number():
    close(photoperiod_rate(14.0, CULTIVAR), 1.0 / (4.0 + 1.193 * (14.0 - 12.5)))
    close(photoperiod_rate(12.0, CULTIVAR), 0.25)
    tlno, p3 = leaf_number_at_ti(a(300.0), 43.0, PC)
    close(tlno, 300.0 / (43.0 * 0.5) + 5.0)
    close(p3, (300.0 / 21.5 + 5.0 + 0.5) * 43.0 - 300.0)


def test_grain_number_branches():
    g2 = 924.3
    i = lambda x: jnp.asarray(x, dtype=jnp.int32)  # noqa: E731
    gpp, ears = grain_number(a(50.0), i(10), g2, a(7.0), PC)  # no ear loss
    close([gpp, ears], [g2 * (50.0 * 1000.0 / 10 * 3.4 / 5.0) / 7200.0 + 50.0, 7.0])
    gpp, ears = grain_number(a(1.0), i(10), g2, a(7.0), PC)  # GPP < 0.15 G2
    want = g2 * 68.0 / 7200.0 + 50.0
    close([gpp, ears], [want, 7.0 * (want / (g2 * 0.15)) ** 0.33])
    gpp, ears = grain_number(a(20.0), i(10), g2, a(15.0), PC)  # dense and below 0.5 G2: barren plants
    want = g2 * 1360.0 / 7200.0 + 50.0
    barfac = 0.0085 * (1.0 - want / g2) * 15.0**1.5
    close([gpp, ears], [want, 15.0 * (want / (g2 * 0.5)) ** barfac])
    gpp, _ = grain_number(a(0.0), i(10), g2, a(7.0), PC)
    close(gpp, 51.0)  # at least 51 kernels


def test_growing_point_thermal_time_by_hand():
    acoef = 0.01061 * 20.0 + 0.5902
    tdsoil = acoef * 20.0 + (1.0 - acoef) * 10.0
    tnsoil = max(0.36354 * 20.0 + 0.63646 * 10.0, 8.0)
    got = growing_point_thermal_time(a(20.0), a(10.0), 20.0, 12.0, 0.0, 8.0, a(34.0), PC)
    close(got, (tnsoil + tdsoil) / 2.0 - 8.0)
    # snow: crown temperatures 2 + T (0.4 + 0.0018 (10 - 15)^2)
    fac = 0.4 + 0.0018 * (10.0 - 15.0) ** 2
    got = growing_point_thermal_time(a(-1.0), a(-4.0), 5.0, 9.0, 10.0, 8.0, a(34.0), PC)
    close(got, ((2.0 - 4.0 * fac) + (2.0 - 1.0 * fac)) / 2.0 - 8.0)


def test_hourly_thermal_time_by_hand():
    close(hourly_thermal_time(a(20.0), a(20.0), 8.0, a(34.0), PC), 12.0)
    want = 0.0
    for h in range(1, 25):
        th = min(max(30.0 + 10.0 * math.sin(3.14 / 12.0 * h), 8.0), 34.0)
        want += (th - 8.0) / 24.0
    close(hourly_thermal_time(a(40.0), a(20.0), 8.0, a(34.0), PC), want)


# ------------------------------------------------------------------ MZ_ROOTGR
def test_emergence_rlv_by_hand():
    got = emergence_rlv(a([8.0]), a([7.0]), a([5.0, 10.0, 15.0]), RC)
    close(got, [[0.2 * 7.0 / 5.0, 0.2 * 7.0 / 10.0 * (1.0 - (15.0 - 8.0) / 10.0), 0.0]])


def test_root_water_deficit_and_survival():
    got = root_water_deficit(a([0.30, 0.12, 0.09]), a([0.10, 0.10, 0.10]), a([0.30, 0.30, 0.30]), RC)
    close(got, [1.0, 4.0 * 0.02 / 0.2, 0.0], abs_=1e-12 if X64 else 1e-6)
    close(waterlogging_survival(a(0.40), a(0.38), 0.05, RC), 1.0 - 0.1 * (1.0 - 0.02 / 0.05))
    close(waterlogging_survival(a(0.40), a(0.30), 0.05, RC), 1.0)


def test_root_front_advance_by_hand():
    close(root_front_advance(a(30.0), a(10.0), a(100.0), a(1.0), a(1.0), a(1.0), a(200.0), RC), 31.0)
    got = root_front_advance(a(30.0), a(10.0), a(300.0), a(0.64), a(0.2), a(1.0), a(200.0), RC)
    close(got, 30.0 + 10.0 * 0.2 * math.sqrt(0.64 * 0.4))
    close(root_front_advance(a(199.5), a(10.0), a(300.0), a(1.0), a(1.0), a(1.0), a(200.0), RC), 200.0)


def test_root_length_growth_by_hand():
    # RNLF = RLNEW / TRLDF = 2.001 / 4; values kept away from the 1e-3 truncation steps
    rlv, spread = root_length_growth(
        a([[1.0, 0.5]]), a([[3.0, 1.0]]), a([2.001]), a([5.0, 10.0]), a([1.0]), RC
    )
    rnlf = 2.001 / 4.0
    want = [math.floor((1.0 + 3.0 * rnlf / 5.0 - 0.005 * 1.0) * 1000.0) / 1000.0,
            math.floor((0.5 + 1.0 * rnlf / 10.0 - 0.005 * 0.5) * 1000.0) / 1000.0]  # fmt: skip
    close(rlv, [want], abs_=1e-12 if X64 else 1e-6)
    assert bool(spread[0])
    _, spread = root_length_growth(a([[1.0]]), a([[0.0]]), a([2.0]), a([5.0]), a([1.0]), RC)
    assert not bool(spread[0])  # TRLDF < 1e-5 RLNEW: no update


@pytest.mark.parametrize("x", [0.0, 1e-6, 250.0])
def test_kernels_finite_gradients_at_edges(x):
    """d/dx of kernels at their clamp edges stays finite (both where branches finite).

    Not below 1e-6: d(PAR / PLTPOP) / dPLTPOP = -PAR / PLTPOP^2 of the (unchanged, b65f1a7)
    interception overflows float32 for a population below about 1e-19 plant m-2."""
    fns = [
        lambda v: assimilation(20.0, 30.0, 20.0, 330.0, v, v, a(75.0), a(1.0), 1.0, 4.2, SPECIES, GC),
        lambda v: stage3_demand(_la(0.5, 18.0, 18.0), a(1.0), 21.0 - v * 0.0, a(8.0), v, GC)[0],
        lambda v: grain_number(v, jnp.asarray(0, dtype=jnp.int32), 924.3, v, PC)[1],
        lambda v: root_front_advance(a(30.0), a(10.0), a(100.0), v, v, v, a(200.0), RC),
    ]
    for fn in fns:
        g = jax.grad(lambda v, fn=fn: jnp.sum(fn(v)))(a(x))
        assert np.isfinite(float(g))
