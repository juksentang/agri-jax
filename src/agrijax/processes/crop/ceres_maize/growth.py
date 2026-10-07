"""CERES-Maize growth: stress factors, leaf area, assimilation, partitioning and senescence.

Two processes port ``MZ_GROSUB`` (``DYNAMIC = INTEGR``) with nitrogen, phosphorus, potassium and
pests off (``ISWNIT = ISWPHO = ISWPOT = ISWDIS = N``: ``AGEFAC = NSTRES = NDEF3 = PSTRES1 =
PSTRES2 = KSTRES = 1``, no pest damage):

* :func:`ceres_stress` - the water-stress block (``SWFAC``, ``TURFAC``) and the excess-water
  factor ``SATFAC`` with its per-layer saturation-day counters ``TSS``. The water-stress factors
  are computed here, as in DSSAT, from the potential transpiration ``EOP`` and the potential root
  water uptake ``TRWUP`` of the crop's ``water_in`` port with the species parameter ``RWUEP1``
  (:func:`water_stress_factors`); who produces ``EOP`` and ``TRWUP`` (a replay of the reference
  run, DSSAT ``ROOTWU``, an RZWQM2 sink) is a binding choice outside the crop, which only reads the port.
* :func:`ceres_growth` - stage-date initialisations, daily assimilation ``CARBO`` (intercepted
  PAR x RUE x CO2 x temperature / water stress x ``SLPF``), leaf appearance, the per-stage leaf,
  stem, ear, grain and root growth, leaf senescence, cold / drought crop failure and the state
  totals.
* :func:`ceres_growth_nstress_replay` - the same day with the nitrogen factors of the crop's
  ``n_in`` port (:class:`~agrijax.iface.crop.CropNIn`, ``iface.crop_n.<slot>``) where
  ``MZ_GROSUB`` uses them for the crop's mass: ``NSTRES`` in ``CARBO = PCARB min(PRFT, SWFAC,
  NSTRES) SLPF``, ``AGEFAC`` in the stage 2-4 expansion factor and (``SLFN``) in the leaf
  senescence, ``NDEF3`` and ``NPOOL`` in the grain-number cap of the first day of effective grain
  filling. An assembly can fill the port with a replay of a reference run's nitrogen factors
  because the nitrogen cycle is not built yet (:mod:`agrijax.processes.n_supply`); with the
  record's no-stress defaults it is the faithful process bit for bit.

``ceres_growth`` is a thin process over kernels with one responsibility each, in the order of
the Fortran: :func:`stage_date_init`, :func:`emergence_init`, :func:`assimilation`,
:func:`leaf_appearance`, one block per stage (:func:`juvenile_growth`,
:func:`floral_induction_growth`, :func:`tassel_silk_growth` with :func:`ear_growth_fraction`,
:func:`stage3_demand`, :func:`stage3_partition`; :func:`silk_efg_growth`;
:func:`grain_fill_growth` with :func:`grain_fill_rate`, :func:`early_maturity`), the stage
selection (:func:`select_stage_block`, :func:`grosub_blocks`), :func:`leaf_senescence`,
:func:`crop_failure`, :func:`canopy_height` and :func:`growth_totals`. Every number
``MZ_GROSUB`` hard-codes is a field of
:class:`~agrijax.processes.crop.ceres_maize.coefficients.GrosubCoefficients`
(``params.coef().grosub``), with its source line.

As in ``MZ_CERES`` the growth routine runs only in stages 1-6; on the maturity (or failure) day
and in stage 6 it stops after the stress block, so growth ends when effective grain filling
ends. Each stage block is computed for every crop and selected with ``jnp.where`` on the stage
mask; every power, square root and division has a clamped argument so both branches of every
``where`` stay finite.

The kernels live in sibling modules and are re-exported here, so every name keeps its
``growth`` import path: the stress block in :mod:`~agrijax.processes.crop.ceres_maize.stress`,
the initialisations, assimilation, leaf appearance and stage blocks in
:mod:`~agrijax.processes.crop.ceres_maize.grosub_stages`, and the stage selection, senescence,
nitrogen factors, crop failure, canopy height and totals in
:mod:`~agrijax.processes.crop.ceres_maize.grosub_day`. This module keeps the two growth processes.

Source: DSSAT-CSM v4.8.6.0 ``Plant/CERES-Maize/MZ_GROSUB.for`` and ``MZ_CERES.for`` (BSD-3,
Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center); Jones & Kiniry (1986) CERES-Maize; ear growth after J. I. Lizaso (2006).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.process import process

from .constants import ISTAGE_EFG
from .grosub_day import (
    CropFailure,
    GrosubDay,
    _nonneg,
    canopy_height,
    crop_failure,
    grain_number_cap,
    grosub_blocks,
    growth_totals,
    leaf_senescence,
    nitrogen_senescence_factor,
)
from .grosub_stages import (
    EarlyMaturity,
    EmergenceInit,
    GrainFill,
    LeafAppearance,
    OrganGrowth,
    StageDateInit,
    assimilation,
    ear_growth_fraction,
    early_maturity,
    emergence_init,
    floral_induction_growth,
    grain_fill_growth,
    grain_fill_rate,
    juvenile_growth,
    leaf_appearance,
    select_stage_block,
    silk_efg_growth,
    stage3_demand,
    stage3_partition,
    stage_date_init,
    tassel_silk_growth,
)
from .state import CeresForcing, CeresGrowthState, CeresMaizeParams, CeresMaizeState, CeresPhenologyState
from .stress import (
    WATER_STRESS_COEFFICIENTS,
    WaterStressCoefficients,
    ceres_stress,
    saturation_factor,
    water_stress_factors,
)

__all__ = [
    "WATER_STRESS_COEFFICIENTS",
    "CropFailure",
    "EarlyMaturity",
    "EmergenceInit",
    "GrainFill",
    "GrosubDay",
    "LeafAppearance",
    "OrganGrowth",
    "StageDateInit",
    "WaterStressCoefficients",
    "assimilation",
    "canopy_height",
    "ceres_growth",
    "ceres_growth_nstress_replay",
    "ceres_stress",
    "crop_failure",
    "ear_growth_fraction",
    "early_maturity",
    "emergence_init",
    "floral_induction_growth",
    "grain_fill_growth",
    "grain_fill_rate",
    "grain_number_cap",
    "grosub_blocks",
    "growth_totals",
    "juvenile_growth",
    "leaf_appearance",
    "leaf_senescence",
    "nitrogen_senescence_factor",
    "saturation_factor",
    "select_stage_block",
    "silk_efg_growth",
    "stage3_demand",
    "stage3_partition",
    "stage_date_init",
    "tassel_silk_growth",
    "water_stress_factors",
]


def _growth_day(
    state: CeresMaizeState,
    params: CeresMaizeParams,
    forcing_t: CeresForcing,
    photo_stress: Array,
    agefac: ArrayLike = 1.0,
    slfn: ArrayLike = 1.0,
) -> tuple[CeresPhenologyState, CeresGrowthState, Array]:
    """The body of :func:`ceres_growth` with the ``CARBO`` stress factor ``photo_stress``, the
    nitrogen factor ``agefac`` of leaf expansion (:func:`grosub_blocks`) and the nitrogen factor
    ``slfn`` of leaf senescence (:func:`leaf_senescence`), both ``1`` with nitrogen off: the day's
    new phenology and growth state (growth, senescence, failure and totals) and the mask of the
    crops that grew.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, DYNAMIC = INTEGR (BSD-3).
    """
    ph = state.phen
    g = state.growth
    cul = params.cultivar
    c = params.coef().grosub
    d = grosub_blocks(state, params, forcing_t, c, photo_stress, agefac)
    grow, em = d.grow, d.em
    tmin = jnp.asarray(forcing_t.tmin)
    swfac = state.stress.swfac

    senla, lai = leaf_senescence(
        d.organs.pla, em.senla, d.organs.slan, em.lai, swfac, tmin, g.pltpop, params.species.fslfw, c, slfn
    )
    senla = jnp.where(grow, senla, em.senla)
    lai = jnp.where(grow, lai, em.lai)
    fail = crop_failure(
        grow,
        ph.istage,
        ph.mdate,
        ph.crop_status,
        jnp.asarray(forcing_t.yrdoy),
        d.leafno,
        lai,
        tmin,
        swfac,
        g.icold,
        g.nwsd,
        cul.tsen,
        cul.cday,
        c,
    )
    new_growth = growth_totals(g, d, senla, lai, fail, params.species.canht_pot, c)
    new_phen = eqx.tree_at(
        lambda p: (p.istage, p.mdate, p.crop_status, p.sumdtt, p.ears, p.gpp),
        ph,
        (
            fail.istage.astype(ph.istage.dtype),
            fail.mdate.astype(ph.mdate.dtype),
            fail.status.astype(ph.crop_status.dtype),
            d.sumdtt,
            _nonneg(grow, d.sd.ears),
            _nonneg(grow, ph.gpp),
        ),
    )
    return new_phen, new_growth, grow


@process(
    reads=("phen", "stress", "growth", "season"),
    writes=(
        "growth",
        "phen.istage",
        "phen.mdate",
        "phen.crop_status",
        "phen.sumdtt",
        "phen.ears",
        "phen.gpp",
    ),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for (BSD-3)",
    fortran_name="MZ_GROSUB",
    key="crop/ceres_maize.growth@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        ("stage-date initialisations, emergence initialisation", "MZ_GROSUB.for INTEGR (BSD-3)"),
        ("assimilation CARBO (PAR x RUE x CO2 x temperature / water stress x SLPF)", "MZ_GROSUB.for INTEGR"),
        (
            "leaf appearance and the per-stage leaf, stem, ear, grain, root growth",
            "MZ_GROSUB.for ISTAGE 1-5 blocks",
        ),
        ("ear growth", "MZ_GROSUB.for GROEAR; after J. I. Lizaso (2006)"),
        ("leaf senescence, cold / drought failure, canopy height, totals", "MZ_GROSUB.for INTEGR"),
        ("CERES-Maize model description", "Jones & Kiniry (1986)"),
    ),
    deviates=(
        (
            "nitrogen, phosphorus, potassium and pests off: AGEFAC = NSTRES = NDEF3 = PSTRES1 = PSTRES2 = "
            "KSTRES = 1, no pest damage",
            "the nutrient and pest modules are not built in this implementation",
            "growth.py module docstring",
        ),
        (
            "DSSAT single precision (REAL*4) is not reproduced",
            "the kernels run in float64 (float32 with AGRI_JAX_X64=0)",
            "tests/integration/test_ceres_dssat.py tolerances",
        ),
    ),
)
def ceres_growth(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """One day of ``MZ_GROSUB`` (``DYNAMIC = INTEGR``) after the stress block, for every crop.

    In the Fortran order: stage-date resets, assimilation, leaf appearance and the block of the
    crop's stage (:func:`grosub_blocks`: :func:`juvenile_growth`, :func:`floral_induction_growth`,
    :func:`tassel_silk_growth`, :func:`silk_efg_growth`, :func:`grain_fill_growth`, all
    evaluated and selected by stage), leaf senescence, cold / drought failure and the state
    totals (:func:`growth_totals`). The hard-coded coefficients come from
    ``params.coef().grosub``; the row spacing is that of the crop's season
    (:meth:`CeresMaizeParams.in_season`). Forcing read: ``yrdoy``, ``tmax``, ``tmin``, ``srad``,
    ``co2``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, DYNAMIC = INTEGR (BSD-3).
    """
    new_phen, new_growth, _ = _growth_day(
        state, params.in_season(state.season), forcing_t, state.stress.swfac
    )
    return eqx.tree_at(lambda x: (x.phen, x.growth), state, (new_phen, new_growth))


@process(
    reads=("phen", "stress", "growth", "season", "n_in"),
    writes=(
        "growth",
        "phen.istage",
        "phen.mdate",
        "phen.crop_status",
        "phen.sumdtt",
        "phen.ears",
        "phen.gpp",
    ),
    source=(
        "DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for (BSD-3), NSTRES, AGEFAC, NDEF3 and NPOOL "
        "from the crop nitrogen port"
    ),
    fortran_name="MZ_GROSUB",
    key="crop/ceres_maize.growth@dssat-4.8.6.0:nstress_replay",
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        ("everything of the faithful growth process", "MZ_GROSUB.for INTEGR (BSD-3), as ceres_growth"),
        (
            "CARBO = PCARB * AMIN1(PRFT, SWFAC, NSTRES, PSTRES1, KSTRES) * SLPF with PSTRES1 = KSTRES = 1",
            "MZ_GROSUB.for INTEGR, 'Calculate Potential Photosynthesis'",
        ),
        (
            "AMIN1(AGEFAC, TURFAC, (1.0-SATFAC), PStres2, KSTRES) in the stage 2-4 PLAG, the late stage-3 "
            "GROSTM and the stage-4 GROEAR (stage 1 without AGEFAC), PStres2 = KSTRES = 1",
            "MZ_GROSUB.for:1210-1215, 1282-1283, 1337-1355, 1404-1405",
        ),
        ("SLFN = (1-FSLFN) + FSLFN*AGEFAC in the leaf senescence, every stage", "MZ_GROSUB.for:1612, 1632"),
        (
            "GPP = AMIN1(GPP*NDEF3, NPOOL/(0.062*0.0095)) when ICSDUR = 1 (first day of effective grain "
            "filling) and NSINK = GROGRN*GNP is not 0",
            "MZ_GROSUB.for:875-880, 928-939 (ICSDUR = 1 on STGDOY(4)), 1543-1557",
        ),
    ),
    deviates=(
        (
            "NSTRES, AGEFAC, NDEF3 and NPOOL are read from the crop nitrogen port iface.crop_n.<slot> "
            "(a replay of the reference run in the RZWQM2-4.6 day) instead of computed by MZ_NFACTO and "
            "the grain nitrogen block from the crop's nitrogen state; NFAC (it changes only the crop's "
            "nitrogen) is not used, and phosphorus, potassium and pests stay off",
            "the RZWQM2 reference maize at CA-TPA is nitrogen-limited and there is no nitrogen module "
            "yet; replaying NSTRES alone leaves out what the other three factors do to the crop's mass",
            "tests/unit/test_ceres_nstress_replay.py and tests/unit/test_ceres_n_factors.py (the "
            "no-stress record is the faithful process bit for bit; each factor acts where MZ_GROSUB "
            "uses it); CA-TPA 2015-2021 against RZWQM2 4.6 outputs (data tier outside this "
            "repository): the full replay and the NSTRES-only replay each reproduce a separate "
            "diagnostic run of the same growth day with switches",
        ),
        (
            "DSSAT single precision (REAL*4) is not reproduced",
            "the kernels run in float64 (float32 with AGRI_JAX_X64=0)",
            "tests/integration/test_ceres_dssat.py tolerances",
        ),
    ),
)
def ceres_growth_nstress_replay(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """:func:`ceres_growth` with the nitrogen factors of the ``n_in`` port where ``MZ_GROSUB``
    uses them for the crop's mass (:class:`~agrijax.iface.crop.CropNIn`):

    * ``CARBO = PCARB min(PRFT, SWFAC, NSTRES) SLPF``;
    * the expansion factor ``min(AGEFAC, TURFAC, 1 - SATFAC)`` of stages 2-4 (leaf expansion
      ``PLAG``, the late stage-3 stem ``GROSTM``, the stage-4 ear ``GROEAR``;
      :func:`grosub_blocks`);
    * ``SLFN = (1 - FSLFN) + FSLFN AGEFAC`` in the leaf senescence of every stage
      (:func:`nitrogen_senescence_factor`, :func:`leaf_senescence`);
    * on the first day of effective grain filling (``ICSDUR = 1``: stage 5 on ``STGDOY(4)``)
      with grain growth (``NSINK = GROGRN GNP`` not 0, ``GNP > 0``), after that day's grain growth,
      ``GPP = min(GPP NDEF3, NPOOL / (0.062 x 0.0095))`` (:func:`grain_number_cap`).

    Everything else is the faithful growth day; with the record's no-stress defaults
    (``CropNIn.initial``: ``NSTRES = AGEFAC = NDEF3 = 1``, ``NPOOL = NPOOL_NO_CAP``) it is
    :func:`ceres_growth` bit for bit. Bind ``n_in`` to ``iface.crop_n.<slot>``, or fill it in the
    crop's own state. ``FSLFN`` is the species file's. Forcing read: ``yrdoy``, ``tmax``,
    ``tmin``, ``srad``, ``co2``.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for, DYNAMIC = INTEGR (BSD-3).
    """
    p = params.in_season(state.season)
    swfac = state.stress.swfac
    n = state.n_in
    nstres = jnp.asarray(n.nstres, dtype=swfac.dtype)
    agefac = jnp.asarray(n.agefac, dtype=swfac.dtype)
    slfn = nitrogen_senescence_factor(agefac, p.species.fslfn)
    new_phen, new_growth, grow = _growth_day(
        state, p, forcing_t, jnp.minimum(swfac, nstres), agefac=agefac, slfn=slfn
    )
    ph = state.phen
    first_efg = (
        grow
        & (ph.istage == ISTAGE_EFG)
        & (jnp.asarray(forcing_t.yrdoy) == ph.stgdoy[..., 3])  # STGDOY(4): ICSDUR = 1
        & (new_growth.grogrn != 0.0)
    )
    cap = grain_number_cap(ph.gpp, n.ndef3, n.npool, p.coef().grosub)
    gpp = jnp.where(first_efg, jnp.maximum(cap, 0.0), new_phen.gpp)
    new_phen = new_phen.replace(gpp=gpp)
    return eqx.tree_at(lambda x: (x.phen, x.growth), state, (new_phen, new_growth))
