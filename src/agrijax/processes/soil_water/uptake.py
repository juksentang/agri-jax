"""Crop water interface records and the DSSAT ``ROOTWU`` potential root water uptake (plan 19 A3, A4).

Interface records (plan 19 A2; ordinary ``State`` pytrees, bound to ``iface.*`` paths at assembly):

* :class:`CropWaterIn` - what a crop reads each day: the soil water of its layers ``sw``, the
  potential transpiration ``eop`` and the potential root water uptake ``trwup``. The crop computes
  its own water-stress factors from ``eop`` and ``trwup`` (DSSAT ``MZ_GROSUB``; decision Q2). Who
  writes ``eop`` and ``trwup`` is a binding choice: a replay process (from a reference run) or an
  uptake producer such as :func:`rootwu_supply`.
* :class:`RootRecord` - what a crop publishes each day for any uptake producer: root length
  density, rooting depth, and the crop numbers the estimate needs (``rwumx``, ``pormin``,
  ``xhlai``), as DSSAT's PLANT outputs ``RLV, RWUMX, PORMIN, XHLAI`` for SPAM [DSSAT-CSM
  ``CSM_Main/LAND.for``]. The estimate never closes over crop parameters (decision Q4).

The estimate :func:`rootwu_estimate` is a static function of explicit arguments. Its cross-day
counter of saturated days ``TSS`` (a ``SAVE``d array in the Fortran, distinct from CERES's own
``TSS`` for ``SATFAC``) is an explicit argument and result; :class:`RootwuState` holds it for the
process :func:`rootwu_supply`.

``ROOTWU`` (J. T. Ritchie), per layer with roots (``RLV > 1e-5``) and water above the lower limit:

.. math::

    \\mathrm{RWU}_L = \\min\\Big(\\frac{\\mathrm{SWCON1}\\,\\exp(\\min(\\mathrm{SWCON2}_L
    (\\mathrm{SW}_L - \\mathrm{LL}_L),\\, 40))}{\\mathrm{SWCON3} - \\ln \\mathrm{RLV}_L},\\;
    \\mathrm{RWUMX}\\cdot\\mathrm{SWEXF}_L\\Big)\\,\\mathrm{DLAYR}_L\\,\\mathrm{RLV}_L,
    \\qquad \\mathrm{TRWUP} = \\sum_L \\mathrm{RWU}_L

with ``SWCON1 = 1.32e-3``, ``SWCON3 = 7.01``, ``SWCON2 = 120 - 250 LL`` (45 when ``LL > 0.30``),
the denominator ``SWCON3 - ln SWCON3`` when ``RLV > exp(SWCON3)``, and the excess-water factor
``SWEXF = clip((SAT - SW) / PORMIN, 0, 1)`` once the layer has been near saturation
(``SAT - SW < PORMIN``) for more than two days (``TSS > 2``). SPAM calls ``ROOTWU`` only when the
canopy is present (``XHLAI > 0``); otherwise ``RWU = TRWUP = 0`` and ``TSS`` is not touched
[DSSAT-CSM ``SPAM/SPAM.for``, RATE].

Source: DSSAT-CSM v4.8.6.0 ``SPAM/ROOTWU.for`` and ``SPAM/SPAM.for`` (BSD-3, Copyright 1998-2026
DSSAT Foundation, University of Florida, International Fertilizer Development Center); Ritchie
J. T. (1998) Soil water balance and plant water stress, in Tsuji et al. (eds) *Understanding
Options for Agricultural Production*, Kluwer, 41-54.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field

__all__ = [
    "RWU_SWCON1",
    "RWU_SWCON3",
    "CropWaterIn",
    "RootRecord",
    "RootwuParams",
    "RootwuResult",
    "RootwuState",
    "SoilView",
    "rootwu_estimate",
    "rootwu_supply",
]

RWU_SWCON1 = 1.32e-3
"""``SWCON1`` of ``ROOTWU`` [cm3 water cm-1 root d-1]."""
RWU_SWCON3 = 7.01
"""``SWCON3`` of ``ROOTWU`` [-]."""

_RLV_MIN = 0.00001  # ROOTWU: layers with RLV <= 1e-5 take up nothing
_TSS_DAYS = 2.0  # ROOTWU: uptake is reduced after more than 2 days near saturation
_EXP_MAX = 40.0  # ROOTWU: MIN(SWCON2 * (SW - LL), 40.)
_PORMIN_FLOOD = 1e-6  # ROOTWU: PORMIN < 1e-6 (flooded rice) never counts saturated days
# guards of the quotients; their squares (the backward pass) stay normal in float32
_DEN_MIN = 1e-6

_C = ("n_crop",)
_CL = ("n_crop", "n_layer")
_L = ("n_layer",)


# ------------------------------------------------------------------------ interface records
class CropWaterIn(State):
    """The water a crop sees each day (bound to ``iface.crop_water.<slot>``).

    ``sw`` is on the crop's layers and shared by the crops of a sample (``[n_layer]``); ``eop`` and
    ``trwup`` are per crop. In DSSAT-CSM these are SPAM's ``SW``, ``EOP`` and ``TRWUP`` passed to
    PLANT [LAND.for]; in RZWQM2 with an embedded DSSAT crop (``ISTRESS = 0``) the node water
    mapped to the crop layers, ``EOP = 10 PET`` and ``ROOTWU``'s ``TRWUP``.
    """

    sw: Array = field(
        unit="cm3 cm-3",
        description="soil water content of the crop layers",
        fortran_name="SW",
        dims=_L,
        grid="dssat_layers",
    )
    eop: Array = field(unit="mm d-1", description="potential transpiration", fortran_name="EOP", dims=_C)
    trwup: Array = field(
        unit="cm d-1", description="potential root water uptake", fortran_name="TRWUP", dims=_C
    )

    @classmethod
    def zeros(cls, n_crop: int, n_layer: int, dtype: Any = None) -> CropWaterIn:
        """An all-zero record (``n_crop`` crops, ``n_layer`` layers)."""
        dt = dtype if dtype is not None else jnp.result_type(float)
        return cls(
            sw=jnp.zeros((n_layer,), dtype=dt),
            eop=jnp.zeros((n_crop,), dtype=dt),
            trwup=jnp.zeros((n_crop,), dtype=dt),
        )


class RootRecord(State):
    """What a crop publishes each day for the uptake producers (bound to ``iface.root.<slot>``).

    DSSAT's PLANT outputs ``RLV, RWUMX, PORMIN, XHLAI`` and SPAM reads them on its next call
    [LAND.for], so a producer running before the crop on day ``d`` sees the record of day
    ``d - 1``. ``rwumx`` and ``pormin`` are species parameters published as state, so their
    gradient flows through the record.
    """

    rlv: Array = field(
        unit="cm cm-3", description="root length density", fortran_name="RLV", dims=_CL, grid="dssat_layers"
    )
    rtdep: Array = field(unit="cm", description="rooting depth", fortran_name="RTDEP", dims=_C)
    rwumx: Array = field(
        unit="cm3 cm-1 d-1",
        description="maximum water uptake per unit root length",
        fortran_name="RWUMX",
        dims=_C,
    )
    pormin: Array = field(
        unit="cm3 cm-3",
        description="minimum air-filled porosity for root function",
        fortran_name="PORMIN",
        dims=_C,
    )
    xhlai: Array = field(
        unit="m2 m-2",
        description="healthy leaf area index (ROOTWU runs when > 0)",
        fortran_name="XHLAI",
        dims=_C,
    )

    @classmethod
    def zeros(cls, n_crop: int, n_layer: int, dtype: Any = None) -> RootRecord:
        """An all-zero record (no roots, no canopy)."""
        dt = dtype if dtype is not None else jnp.result_type(float)
        c = jnp.zeros((n_crop,), dtype=dt)
        return cls(rlv=jnp.zeros((n_crop, n_layer), dtype=dt), rtdep=c, rwumx=c, pormin=c, xhlai=c)


class SoilView(eqx.Module):
    """The soil a root water uptake estimate reads, on the crop layers (``[n_layer]`` each)."""

    dlayr: Array  # cm, layer thickness
    ll: Array  # cm3 cm-3, lower limit
    sat: Array  # cm3 cm-3, saturation
    sw: Array  # cm3 cm-3, soil water content


class RootwuResult(NamedTuple):
    """Result of :func:`rootwu_estimate`."""

    rwu: Array  # [n_crop, n_layer] cm d-1, potential uptake of each layer
    trwup: Array  # [n_crop] cm d-1, total potential uptake
    tss: Array  # [n_crop, n_layer] d, updated saturation-day counter


# ------------------------------------------------------------------------ the estimate
def rootwu_estimate(root: RootRecord, soil: SoilView, tss: ArrayLike) -> RootwuResult:
    """DSSAT ``ROOTWU`` (``DYNAMIC = RATE``) as called by SPAM: potential root water uptake.

    Pure and explicit: ``root`` is the crop's published record, ``soil`` the layer view,
    ``tss`` [n_crop, n_layer] the producer's saturation-day counter from its previous call. When
    ``root.xhlai <= 0`` SPAM does not call ``ROOTWU``: ``rwu = trwup = 0`` and ``tss`` is returned
    unchanged. The counter of a layer changes only on layers that take up water (roots and
    ``SW > LL``), as in the Fortran. Every logarithm, exponential and division has a clamped
    argument, so both branches of every ``where`` are finite.

    Source: DSSAT-CSM v4.8.6.0 SPAM/ROOTWU.for (RATE) and SPAM/SPAM.for (``XHLAI > 0`` gate), BSD-3.
    """
    rlv = jnp.asarray(root.rlv)
    ll = jnp.asarray(soil.ll)
    sw = jnp.asarray(soil.sw)
    sat = jnp.asarray(soil.sat)
    dlayr = jnp.asarray(soil.dlayr)
    tss = jnp.asarray(tss)
    rwumx = jnp.asarray(root.rwumx)[..., None]
    pormin = jnp.asarray(root.pormin)[..., None]

    swcon2 = jnp.where(ll > 0.30, 45.0, 120.0 - 250.0 * ll)
    active = (rlv > _RLV_MIN) & (sw > ll)
    big = rlv > jnp.exp(RWU_SWCON3)
    log_rlv = jnp.log(jnp.clip(rlv, _RLV_MIN, jnp.exp(RWU_SWCON3)))
    denom = jnp.where(big, RWU_SWCON3 - jnp.log(RWU_SWCON3), RWU_SWCON3 - log_rlv)
    # denom -> 0 only as RLV -> exp(SWCON3) = 1108 cm cm-3 (not a root density); keep it finite
    pot = RWU_SWCON1 * jnp.exp(jnp.minimum(swcon2 * (sw - ll), _EXP_MAX)) / jnp.maximum(denom, _DEN_MIN)

    air = sat - sw
    tss_act = jnp.where((air >= pormin) | (pormin < _PORMIN_FLOOD), 0.0, tss + 1.0)
    # with PORMIN < 1e-6 the counter stays 0, so the quotient is never selected: the guard is exact
    swexf = jnp.where(tss_act > _TSS_DAYS, jnp.maximum(air / jnp.maximum(pormin, _PORMIN_FLOOD), 0.0), 1.0)
    swexf = jnp.minimum(swexf, 1.0)
    per_root = jnp.minimum(jnp.minimum(pot, rwumx * swexf), rwumx)  # cm3 water cm-1 root d-1
    rwu = jnp.where(active, per_root, 0.0) * dlayr * rlv  # cm d-1

    called = (jnp.asarray(root.xhlai) > 0.0)[..., None]
    rwu = jnp.where(called, rwu, 0.0)
    tss_new = jnp.where(called & active, tss_act, tss)
    return RootwuResult(rwu=rwu, trwup=jnp.sum(rwu, axis=-1), tss=tss_new)


# ------------------------------------------------------------------------ the producer process
class RootwuParams(Params):
    """Soil layers of the uptake producer (the crop layers), ``[n_layer]``."""

    dlayr: Array = field(unit="cm", description="layer thickness", fortran_name="DLAYR", dims=_L)
    ll: Array = field(unit="cm3 cm-3", description="lower limit", fortran_name="LL", dims=_L)
    sat: Array = field(unit="cm3 cm-3", description="saturation", fortran_name="SAT", dims=_L)


class RootwuState(State):
    """State of the ``ROOTWU`` producer: its ``SAVE``d counter and today's layer uptake, plus the
    two records it exchanges (ports): the crop's root record (read) and the crop water record
    (``sw`` read, ``trwup`` written)."""

    tss: Array = field(
        unit="d",
        description="days each layer has been near saturation (ROOTWU)",
        fortran_name="TSS",
        dims=_CL,
    )
    rwu: Array = field(
        unit="cm d-1", description="potential root water uptake of each layer", fortran_name="RWU", dims=_CL
    )
    root: RootRecord = port(description="root record published by the crop (read)")
    water: CropWaterIn = port(description="crop water record: sw read, trwup written")

    @classmethod
    def initial(cls, n_crop: int, n_layer: int, dtype: Any = None) -> RootwuState:
        """``SEASINIT``: ``TSS = RWU = 0`` (ports empty; fill them or bind them)."""
        dt = dtype if dtype is not None else jnp.result_type(float)
        z = jnp.zeros((n_crop, n_layer), dtype=dt)
        return cls(tss=z, rwu=z)


@process(
    reads=("tss", "root", "water.sw"),
    writes=("tss", "rwu", "water.trwup"),
    source="DSSAT-CSM v4.8.6.0 SPAM/ROOTWU.for, SPAM/SPAM.for (BSD-3)",
    fortran_name="ROOTWU",
    key="water_supply/rootwu@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build="dscsm048 v4.8.6.0 (gfortran, A12 instrumented build)",
    sources=(
        ("layer potential uptake RWU, total TRWUP", "SPAM/ROOTWU.for, RATE (J. T. Ritchie)"),
        ("excess-water factor SWEXF and saturation days TSS", "SPAM/ROOTWU.for, RATE"),
        ("ROOTWU called only when XHLAI > 0", "SPAM/SPAM.for, RATE, potential root water uptake"),
    ),
    deviates=(
        (
            "DSSAT single precision (REAL*4) is not reproduced",
            "the kernel runs in float64 (float32 with AGRI_JAX_X64=0)",
            "tests/integration/test_rootwu_dssat.py tolerances",
        ),
    ),
)
def rootwu_supply(state: RootwuState, params: RootwuParams, forcing_t: Forcing) -> RootwuState:
    """One ``ROOTWU`` call: ``rwu, trwup, tss <- rootwu_estimate(root, soil, tss)``; ``trwup`` goes
    into the crop water record, ``rwu`` and the counter stay in the producer's state.

    Source: DSSAT-CSM v4.8.6.0 SPAM/ROOTWU.for (BSD-3).
    """
    water = state.water
    soil = SoilView(dlayr=params.dlayr, ll=params.ll, sat=params.sat, sw=water.sw)
    res = rootwu_estimate(state.root, soil, state.tss)
    return eqx.tree_at(
        lambda s: (s.tss, s.rwu, s.water.trwup),
        state,
        (res.tss, res.rwu, res.trwup.astype(water.trwup.dtype)),
    )
