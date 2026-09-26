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

Every number above is a field of :class:`RootwuCoefficients` (unit, meaning, ``ROOTWU.for`` line
and statement); :data:`ROOTWU_COEFFICIENTS` holds the DSSAT-CSM v4.8.6.0 values, and
``RootwuParams.coefficients`` takes a set with array leaves to calibrate or differentiate them.

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

from agrijax.core.coefficients import Coefficients, Provenance, coef, numerical_guard
from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field
from agrijax.iface.crop import CropWaterIn, RootRecord

__all__ = [
    "ROOTWU_COEFFICIENTS",
    "RWU_SWCON1",
    "RWU_SWCON3",
    "CropWaterIn",
    "RootRecord",
    "RootwuCoefficients",
    "RootwuParams",
    "RootwuResult",
    "RootwuState",
    "SoilView",
    "rootwu_estimate",
    "rootwu_supply",
]

#: the reference of every ROOTWU coefficient (the process registry's ``@dssat-4.8.6.0``)
_REF = "dssat-4.8.6.0"
_RITCHIE = "Ritchie (1998)"


def _rootwu(value: float, unit: str, description: str, line: int, statement: str, **kw: Any) -> Any:
    """A ``ROOTWU`` coefficient: DSSAT-CSM v4.8.6.0 ``SPAM/ROOTWU.for:<line>``, ``statement``."""
    prov = Provenance.at(
        _REF, f"SPAM/ROOTWU.for:{line}", routine="ROOTWU", statement=statement, paper=_RITCHIE
    )
    return coef(value, unit, description, prov, **kw)


class RootwuCoefficients(Coefficients):
    """The numbers DSSAT ``ROOTWU`` hard-codes (J. T. Ritchie's root water uptake).

    Defaults are the DSSAT-CSM v4.8.6.0 values (Python floats: a run with
    :data:`ROOTWU_COEFFICIENTS` traces exactly the literals of the equations). The uptake
    response (``swcon1``, ``swcon3``, the ``swcon2`` line and the excess-water delay
    ``tss_days``) is calibratable; the switches that only guard the reference's arithmetic
    (``exp_max``, ``pormin_flood``) are leaves kept out of the calibration vector.
    """

    swcon1: float = _rootwu(
        1.32e-3,
        "cm3 cm-1 d-1",
        "uptake per unit root length at SW = LL before the root-density term",
        43,
        "PARAMETER (SWCON1 = 1.32E-3)",
        fortran_name="SWCON1",
    )
    swcon3: float = _rootwu(
        7.01,
        "-",
        "constant of the root-density term SWCON3 - ln(RLV) of the uptake denominator",
        44,
        "PARAMETER (SWCON3 = 7.01)",
        fortran_name="SWCON3",
    )
    swcon2_base: float = _rootwu(
        120.0,
        "-",
        "intercept of SWCON2 = 120 - 250 LL (steepness of uptake in SW - LL)",
        70,
        "SWCON2(L) = 120. - 250. * LL(L)",
        fortran_name="SWCON2",
    )
    swcon2_slope: float = _rootwu(
        250.0,
        "-",
        "decrease of SWCON2 per unit lower limit (SWCON2 = 120 - 250 LL)",
        70,
        "SWCON2(L) = 120. - 250. * LL(L)",
        fortran_name="SWCON2",
    )
    swcon2_ll_max: float = _rootwu(
        0.30,
        "cm3 cm-3",
        "lower limit above which SWCON2 takes its fixed value",
        71,
        "IF (LL(L) .GT. 0.30) SWCON2(L) = 45.0",
    )
    swcon2_high_ll: float = _rootwu(
        45.0,
        "-",
        "SWCON2 of layers whose lower limit exceeds swcon2_ll_max",
        71,
        "IF (LL(L) .GT. 0.30) SWCON2(L) = 45.0",
        fortran_name="SWCON2",
    )
    rlv_min: float = _rootwu(
        0.00001,
        "cm cm-3",
        "root length density at or below which a layer takes up nothing",
        76,
        "IF (RLV(L) .LE. 0.00001 .OR. SW(L) .LE. LL(L)) THEN",
    )
    exp_max: float = _rootwu(
        40.0,
        "-",
        "cap of the exponent SWCON2 (SW - LL) of the uptake (overflow guard of the reference)",
        85,
        "RWU(L) = SWCON1*EXP(MIN((SWCON2(L)*(SW(L)-LL(L))),40.))/",
        calibrate=False,
    )
    pormin_flood: float = _rootwu(
        1e-6,
        "cm3 cm-3",
        "PORMIN below which saturated days are never counted (flooded rice, PORMIN = 0)",
        98,
        "IF ((SAT(L)-SW(L)) .GE. PORMIN .OR. PORMIN < 1.E-6) THEN",
        calibrate=False,
    )
    tss_days: float = _rootwu(
        2.0,
        "d",
        "days near saturation after which a layer's uptake is reduced by (SAT - SW) / PORMIN",
        107,
        "IF (TSS(L) .GT. 2.) THEN",
    )


#: the DSSAT-CSM v4.8.6.0 values of :class:`RootwuCoefficients`
ROOTWU_COEFFICIENTS = RootwuCoefficients()

RWU_SWCON1 = ROOTWU_COEFFICIENTS.swcon1
"""``SWCON1`` of ``ROOTWU`` [cm3 water cm-1 root d-1] (:attr:`RootwuCoefficients.swcon1`)."""
RWU_SWCON3 = ROOTWU_COEFFICIENTS.swcon3
"""``SWCON3`` of ``ROOTWU`` [-] (:attr:`RootwuCoefficients.swcon3`)."""

# guard of the uptake denominator; its square (the backward pass) stays normal in float32
_DEN_MIN = numerical_guard(
    "rootwu.den_min",
    1e-6,
    "floor of SWCON3 - ln(RLV), which tends to 0 only as RLV -> exp(SWCON3) (not a root density)",
)

_C = ("n_crop",)
_CL = ("n_crop", "n_layer")
_L = ("n_layer",)


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
def rootwu_estimate(
    root: RootRecord, soil: SoilView, tss: ArrayLike, c: RootwuCoefficients = ROOTWU_COEFFICIENTS
) -> RootwuResult:
    """DSSAT ``ROOTWU`` (``DYNAMIC = RATE``) as called by SPAM: potential root water uptake.

    Pure and explicit: ``root`` is the crop's published record, ``soil`` the layer view,
    ``tss`` [n_crop, n_layer] the producer's saturation-day counter from its previous call. When
    ``root.xhlai <= 0`` SPAM does not call ``ROOTWU``: ``rwu = trwup = 0`` and ``tss`` is returned
    unchanged. The counter of a layer changes only on layers that take up water (roots and
    ``SW > LL``), as in the Fortran. Every logarithm, exponential and division has a clamped
    argument, so both branches of every ``where`` are finite. ``c`` holds the numbers ``ROOTWU``
    hard-codes (:class:`RootwuCoefficients`, default the DSSAT-CSM v4.8.6.0 values).

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

    swcon2 = jnp.where(ll > c.swcon2_ll_max, c.swcon2_high_ll, c.swcon2_base - c.swcon2_slope * ll)
    active = (rlv > c.rlv_min) & (sw > ll)
    big = rlv > jnp.exp(c.swcon3)
    log_rlv = jnp.log(jnp.clip(rlv, c.rlv_min, jnp.exp(c.swcon3)))
    denom_big = c.swcon3 - jnp.log(c.swcon3)  # SWCON3 > 0: a declared coefficient, not state
    denom = jnp.where(big, denom_big, c.swcon3 - log_rlv)
    # denom -> 0 only as RLV -> exp(SWCON3) = 1108 cm cm-3 (not a root density); keep it finite
    pot = c.swcon1 * jnp.exp(jnp.minimum(swcon2 * (sw - ll), c.exp_max)) / jnp.maximum(denom, _DEN_MIN)

    air = sat - sw
    tss_act = jnp.where((air >= pormin) | (pormin < c.pormin_flood), 0.0, tss + 1.0)
    # with PORMIN < 1e-6 the counter stays 0, so the quotient is never selected: the guard is exact
    swexf = jnp.where(tss_act > c.tss_days, jnp.maximum(air / jnp.maximum(pormin, c.pormin_flood), 0.0), 1.0)
    swexf = jnp.minimum(swexf, 1.0)
    per_root = jnp.minimum(jnp.minimum(pot, rwumx * swexf), rwumx)  # cm3 water cm-1 root d-1
    rwu = jnp.where(active, per_root, 0.0) * dlayr * rlv  # cm d-1

    called = (jnp.asarray(root.xhlai) > 0.0)[..., None]
    rwu = jnp.where(called, rwu, 0.0)
    tss_new = jnp.where(called & active, tss_act, tss)
    return RootwuResult(rwu=rwu, trwup=jnp.sum(rwu, axis=-1), tss=tss_new)


# ------------------------------------------------------------------------ the producer process
class RootwuParams(Params):
    """Soil layers of the uptake producer (the crop layers), ``[n_layer]``, and the ``ROOTWU``
    coefficients (``None``: :data:`ROOTWU_COEFFICIENTS`; a :class:`RootwuCoefficients` with array
    leaves, ``RootwuCoefficients().as_arrays()``, to calibrate or differentiate them)."""

    dlayr: Array = field(unit="cm", description="layer thickness", fortran_name="DLAYR", dims=_L)
    ll: Array = field(unit="cm3 cm-3", description="lower limit", fortran_name="LL", dims=_L)
    sat: Array = field(unit="cm3 cm-3", description="saturation", fortran_name="SAT", dims=_L)
    coefficients: RootwuCoefficients | None = field(
        description="the coefficients ROOTWU hard-codes (None: the DSSAT-CSM v4.8.6.0 values)", default=None
    )

    def coef(self) -> RootwuCoefficients:
        """The ``ROOTWU`` coefficients in force: :attr:`coefficients`, or the DSSAT values."""
        return ROOTWU_COEFFICIENTS if self.coefficients is None else self.coefficients


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
    res = rootwu_estimate(state.root, soil, state.tss, params.coef())
    return eqx.tree_at(
        lambda s: (s.tss, s.rwu, s.water.trwup),
        state,
        (res.tss, res.rwu, res.trwup.astype(water.trwup.dtype)),
    )
