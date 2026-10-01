"""The canopy record CERES-Maize publishes for the PET module (P6, ``iface.canopy.<slot>``).

Day entry 15a of the coupling contract (``crops.<slot>.canopy`` in
:data:`agrijax.iface.contract.DAY_TABLE`). The Shuttleworth-Wallace PET of RZWQM2 4.6 reads the
leaf area index, the total leaf area index and the canopy height that the plant manager ``MAPLNT``
left at the end of the previous day (the ``PHYSCL`` entry values of day ``t`` equal the ``MAPLNT``
exit values of day ``t - 1`` on all 3287 days of CA-TPA 2015-2023, dump tables of the instrumented
RZWQM2 4.6 build). Those are not the embedded crop's own outputs: the RZWQM2 driver of the embedded
DSSAT crop (``DSSATDRV``) derives them from the crop state after the crop's day, and ``MAPLNT``
zeroes them on the harvest day. This entry reproduces the driver's derivation from the CERES-Maize
state at the end of the crop's day; the harvest zeroing is the harvest reset's
(:func:`~.season.ceres_harvest`, which runs after this entry).

* **Leaf area index.** Before maturity it is the crop's green leaf area index (``XHLAI``). After
  maturity, when a harvest date later than maturity is planned, it declines linearly to 0 over
  ``min(dry_down_days, HDATE - MDATE)`` days after ``MDATE`` (a dry-down that leaves the leaf mass
  unchanged), and is never negative [RZWQM2 4.5 DSSATDRV.for:1650-1659]. Dates are differenced as
  ``YYYYDDD`` integers, as the reference does (a season does not span a new year).
* **Total leaf area index.** The reference passes the same number as the leaf area index
  [DSSATDRV.for:1660]; ``MAPLNT`` exit ``TLAI == LAI`` on all 3287 days. So ``tlai = lai``, not the
  green plus senesced area the record's field description allows for.
* **Height** [cm]. For maize the reference does not use CERES ``CANHT``; it computes the height
  from the stalk mass per plant with the regression of Ma et al. (2003) on D. C. Nielsen's corn
  data [DSSATDRV.for:1606-1616]: ``stalk = BIOMAS [kg ha-1] - NINT(grain [kg ha-1])``,
  ``S = stalk [g m-2] / PLTPOP``, ``h = HTMAX (1 - exp(-alpha S / (2 HTMAX)))`` with
  ``alpha = -2 HTMAX ln(1/2) / BIOHALF`` [DSSATDRV.for:888-890], i.e. the height is half of
  ``HTMAX`` at the stalk mass ``BIOHALF``. The published height is the running maximum of ``h``
  over the season [DSSATDRV.for:1631-1632], reset at the season's start [DSSATDRV.for:230]; here the
  running maximum is the previous day's published height (``canopy_out.height``), which the harvest
  reset sets to 0, so each season starts from 0. ``PLTPOP`` is the planting population of the
  season (the driver reads the management record, ``PLANTVAR%PLTPOP``), not the crop's
  population after a failure. On CA-TPA 2015-2021 ``100 CANHT`` of the embedded crop differs from
  the published height by up to 9.0 cm, so the CERES height is not a stand-in.

``HTMAX`` and ``BIOHALF`` come from the cultivar file of the embedded crop (CA-TPA
``MZCER040.CUL``, cultivar IB0012: 244.6 cm, 43.07 g plant-1; the reference's maize defaults are
the same numbers, DSSATDRV.for:876-877). The user resets of LAI and height (``IRESETLAI``,
``IRESETHT1``, off in CA-TPA) and a ``BIOHALF`` above 100 read as kg ha-1 [DSSATDRV.for:887-888]
are not reproduced.

RZWQM2 is read for its conventions only; no statement of its source is reproduced (file and line
citations only).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.grad import coef_div, round_st
from agrijax.core.process import process
from agrijax.core.units import KG_HA_PER_G_M2
from agrijax.iface.crop import CanopyRecord

from .canopy_params import RZWQM2_CANOPY, CanopyCoefficients, CeresCanopyParams
from .state import CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = [
    "RZWQM2_CANOPY",
    "CanopyCoefficients",
    "CeresCanopyParams",
    "ceres_canopy",
    "published_lai",
    "stalk_height",
]


def published_lai(xhlai: Array, mdate: Array, hdate: Array, yrdoy: Array, c: CanopyCoefficients) -> Array:
    """The leaf area index the driver publishes [m2 m-2]: ``XHLAI`` before maturity; after
    ``MDATE``, when ``HDATE > MDATE > 0``, ``XHLAI (1 - (YRDOY - MDATE) / min(dry_down_days,
    HDATE - MDATE))`` evaluated as ``XHLAI - XHLAI / span * days``; never below 0.

    ``mdate``, ``hdate`` and ``yrdoy`` are ``YYYYDDD`` integers, differenced as integers.

    Source: RZWQM2 4.5 DSSATDRV.for:1650-1659 (conventions; the 4.6 binary is built from this tree).
    """
    dt = xhlai.dtype
    mdate = jnp.asarray(mdate)
    hdate = jnp.asarray(hdate)
    yrdoy = jnp.asarray(yrdoy)
    decline = (yrdoy > mdate) & (hdate > mdate) & (mdate > 0)
    span = jnp.minimum(jnp.asarray(c.dry_down_days, dt), (hdate - mdate).astype(dt))
    span = jnp.where(decline, span, jnp.ones_like(span))  # both branches finite
    days = (yrdoy - mdate).astype(dt)
    lai = jnp.where(decline, xhlai - xhlai / span * days, xhlai)
    return jnp.maximum(lai, jnp.zeros_like(lai))


def stalk_height(biomas: Array, grnwt: Array, ears: Array, pltpop: Array, c: CanopyCoefficients) -> Array:
    """Height [cm] of the stalk-mass regression for maize, before the running maximum.

    ``biomas`` [g m-2], ``grnwt`` [g plant-1] (grain weight of an ear's plant), ``ears`` [ear m-2]
    and the planting population ``pltpop`` [plant m-2]. The grain mass in kg ha-1 is rounded to an
    integer as the reference does (Fortran ``NINT``, :func:`agrijax.core.grad.round_st`: identity
    derivative in the default gradient mode).

    Source: RZWQM2 4.5 DSSATDRV.for:1606-1611 (height), 888-890 (PLALFA); Ma et al. (2003).
    """
    dt = biomas.dtype
    htmax = jnp.asarray(c.htmax, dt)
    two_htmax = htmax + htmax  # 2 HTMAX, exactly
    alpha = coef_div(-two_htmax * jnp.log(jnp.asarray(c.height_fraction, dt)), jnp.asarray(c.biohalf, dt))
    stalk = biomas * KG_HA_PER_G_M2 - round_st(grnwt * ears * KG_HA_PER_G_M2)  # kg ha-1
    stem = stalk / KG_HA_PER_G_M2 / jnp.asarray(pltpop, dt)  # g plant-1
    return htmax * (1.0 - jnp.exp(coef_div(-alpha * stem, two_htmax)))


@process(
    reads=(
        "growth.lai",
        "growth.biomas",
        "growth.grnwt",
        "phen.ears",
        "phen.mdate",
        "season",
        "canopy_out.height",
    ),
    writes=("canopy_out",),
    source=(
        "RZWQM2 4.6 driver of the embedded DSSAT crop, canopy handed to the plant manager and to the "
        "next day's PET (behaviour of the 4.6 binary; line citations from the RZWQM2 4.5 source)"
    ),
    fortran_name="DSSATDRV",
    key="crop/ceres_maize.canopy@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="point",
    ref_build=(
        "RZWQM2 4.6 main_ryzen5_avx512 (instrumented build that dumps the daily entry and exit tables, "
        "CA-TPA 2015-2023)"
    ),
    sources=(
        (
            "PET reads the MAPLNT exit LAI, TLAI, HEIGHT of the previous day",
            "dump tables rzwqm46_catpa2015_2023 of the instrumented RZWQM2 4.6 build: PHYSCL entry of "
            "day t = MAPLNT exit of day t - 1 on all 3287 days",
        ),
        (
            "published LAI = XHLAI, with the post-maturity decline to 0; TLAI = LAI",
            "RZWQM2 4.5 source DSSATDRV.for:1650-1660 (read for conventions)",
        ),
        (
            "maize height from the stalk mass per plant, running maximum over the season",
            "DSSATDRV.for:230, 858-860, 876-877, 888-890, 1606-1611, 1631-1632; Ma et al. (2003)",
        ),
        (
            "the harvest day's record is 0 (MAPLNT), written by the harvest reset after this entry",
            "RZWQM2 4.5 source Rzman.for:5960-5978; crop/ceres_maize.harvest@rzwqm2-4.6:faithful",
        ),
    ),
    deviates=(
        (
            "REAL*4 intermediates of the reference (stalk mass, the declined LAI) are not reproduced",
            "the kernels run in float64 (float32 with AGRI_JAX_X64=0)",
            "tests/integration/test_ceres_canopy_catpa.py (within one REAL*4 rounding of the "
            "reference on every day of CA-TPA 2015-2021)",
        ),
        (
            "the running maximum of the height is the previous day's published height, not a "
            "separate driver variable reset at sowing",
            "the harvest reset zeroes the published record, so a season starts from 0 as the "
            "reference's reset at the season's first call does",
            "tests/integration/test_ceres_canopy_catpa.py",
        ),
    ),
)
def ceres_canopy(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """Write the canopy record (P6, ``canopy_out``) from the crop state at the end of its day.

    ``lai`` is :func:`published_lai` of the green leaf area index (``XHLAI``), ``tlai`` the same
    number, ``height`` [cm] the running maximum of :func:`stalk_height` (the previous published
    height and today's). The planned harvest date is the season's row of ``params.canopy.hdate``,
    the population the season's planting population. Forcing read: ``yrdoy``.

    Source: RZWQM2 4.5 DSSATDRV.for:1606-1660 (conventions); port P6 of ``agrijax.iface``.
    """
    cp = params.canopy
    if cp is None:  # static: the structure of the parameters
        raise ValueError(
            "ceres_canopy needs params.canopy (CeresCanopyParams: the planned harvest dates of the "
            "seasons, which set the post-maturity decline of the published LAI)"
        )
    if state.canopy_out is None:  # static: the structure of the state
        raise ValueError("ceres_canopy needs the canopy_out port (CanopyRecord.zeros(n_crop) or a binding)")
    c = cp.coef()
    g, ph = state.growth, state.phen
    pltpop = params.in_season(state.season).pltpop
    lai = published_lai(g.lai, ph.mdate, cp.hdate_of(state.season), forcing_t.yrdoy, c)
    h = stalk_height(g.biomas, g.grnwt, ph.ears, pltpop, c)
    prev = state.canopy_out.height
    height = jnp.maximum(prev, h.astype(prev.dtype))
    return state.replace(canopy_out=CanopyRecord(lai=lai, tlai=lai, height=height))
