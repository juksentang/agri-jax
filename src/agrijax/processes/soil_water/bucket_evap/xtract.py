"""DSSAT-CSM v4.8.6.0 actual root water extraction: ``EP`` and ``XTRACT`` (``SPAM`` RATE, soil side).

After the potential transpiration ``EOP`` (``TRANS``), ``SPAM`` limits the day's transpiration to
the roots' potential uptake and extracts it from the layers (``SPAM/SPAM.for``):

1. ``EP = MIN(EOP, TRWUP * 10)`` when ``XHLAI > 1e-4`` and ``EOP > 1e-4``, else ``EP = 0``
   (lines 392-398);
2. the water available for extraction: ``SW_AVAIL = MAX(0, SW + SWDELTS + SWDELTU)`` of the
   start-of-day water content and the soil water rates (line 272), less the day's soil
   evaporation: ``MESEV = 'R'``: ``SW_AVAIL(1) = MAX(0, SW_AVAIL(1) - 0.1 ES / DLAYR(1))``;
   otherwise per layer ``MAX(0, SW_AVAIL(L) - 0.1 ES_LYR(L) / DLAYR(L))`` (lines 424-433);
3. ``XTRACT`` (``SPAM/SPSUBS.for:415-487``): ``SW_AVAIL = MAX(0, SW_AVAIL - LL)``; with ``EP > 0``
   the potential layer uptake ``RWU`` of ``ROOTWU`` is scaled by ``WUF = 0.1 EP / TRWUP`` (1 when
   ``0.1 EP > TRWUP``) in every layer whose start-of-day ``SW > LL``, capped at ``SW_AVAIL DLAYR``,
   and summed to ``TRWU``; layers with ``SW <= LL`` give nothing; ``EP = 10 TRWU``; ``SWDELTX =
   SWTEMP - SW`` (the change of the water content, applied by ``WATBAL`` INTEGR).

The layer extraction is published as the sink record's ``uptake`` (P4, cm d-1 per DSSAT layer,
positive removes water): the tipping bucket applies ``SWDELTX = -uptake / DLAYR``
(``processes/soil_water/bucket``). The actual transpiration ``EP`` goes to the evaporation record
(PD1 ``transpiration``, mm d-1) and ``TRWU`` [cm d-1] stays in the module's state.

Supported: the extraction of ``SPAM`` for crops without their own uptake (``UH2O = 0``, every
CERES and CROPGRO crop). Not supported (declared): the plant-routine uptake branch of ``XTRACT``
(``SUM(UH2O) > 1e-6``, lines 442-450), the zonal energy balance ``ETPHOT`` (``MEEVP = 'Z'`` or
``MEPHO = 'L'``, ``SPAM.for:404-415``).

**The ``SW <= LL`` test.** DSSAT holds ``SW`` and ``LL`` in REAL*4, and a layer that ``XTRACT``
has dried to its lower limit ends the day with ``SW`` equal to ``LL`` in REAL*4 (``WATBAL`` rounds
``SW`` to 1e-6), so ``SW .GT. LL`` is false and the layer gives no more water. The comparison is
made on the values the module receives; the DSSAT day assembly gives it the REAL*4 soil limits of
the reference (:func:`agrijax.models.day_dssat486.dssat_soil_values`, a labelled convention).

Source: DSSAT-CSM v4.8.6.0 ``SPAM/SPAM.for``, ``SPAM/SPSUBS.for`` (``XTRACT``, J. T. Ritchie),
BSD-3 (Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer
Development Center).
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
from agrijax.core.state import Params, State, field
from agrijax.core.units import CM_PER_MM, MM_PER_CM
from agrijax.iface.crop import CanopyRecord, CropWaterIn
from agrijax.iface.soil import SinkInputs
from agrijax.iface.surface import EvaporationRecord

from ..bucket.watbal import BucketFluxes

__all__ = [
    "XTRACT_COEFFICIENTS",
    "XtractCoefficients",
    "XtractParams",
    "XtractResult",
    "XtractState",
    "actual_transpiration",
    "extraction_supply",
    "spam_xtract",
    "xtract",
]

_REF = "dssat-4.8.6.0"
_L = ("n_layer",)
_GRID = "dssat_layers"


def _at(file_line: str, routine: str, statement: str, note: str = "") -> Provenance:
    return Provenance.at(_REF, file_line, routine=routine, statement=statement, note=note)


class XtractCoefficients(Coefficients):
    """The thresholds with which ``SPAM`` computes the actual transpiration ``EP``."""

    lai_min_ep: float = coef(
        1e-4,
        "m2 m-2",
        "healthy leaf area index above which SPAM limits EOP to the root supply (else EP = 0)",
        _at("SPAM/SPAM.for:392", "SPAM", "IF (XHLAI .GT. 1.E-4 .AND. EOP .GT. 1.E-4) THEN"),
        calibrate=False,
    )
    eop_min_ep: float = coef(
        1e-4,
        "mm d-1",
        "potential transpiration above which SPAM limits EOP to the root supply (else EP = 0)",
        _at("SPAM/SPAM.for:392", "SPAM", "IF (XHLAI .GT. 1.E-4 .AND. EOP .GT. 1.E-4) THEN"),
        calibrate=False,
    )


#: the DSSAT-CSM v4.8.6.0 values
XTRACT_COEFFICIENTS = XtractCoefficients()

# WUF = 0.1 EP / TRWUP is only formed when 0.1 EP <= TRWUP, i.e. TRWUP > 0 whenever EP > 0; the
# floor keeps the unselected branch finite on days with TRWUP = 0
_TRWUP_MIN = numerical_guard(
    "xtract.trwup_min", 1e-30, "floor of TRWUP in WUF = 0.1 EP / TRWUP (the branch is unused at TRWUP = 0)"
)


class XtractResult(NamedTuple):
    """``XTRACT``'s outputs: layer extraction ``uptake`` [cm d-1] (``-SWDELTX DLAYR``), ``trwu``
    [cm d-1] and the actual transpiration ``ep`` [mm d-1]."""

    uptake: Array
    trwu: Array
    ep: Array


def actual_transpiration(
    eop: ArrayLike, trwup: ArrayLike, xhlai: ArrayLike, c: XtractCoefficients = XTRACT_COEFFICIENTS
) -> Array:
    """``EP = MIN(EOP, 10 TRWUP)`` [mm d-1] when ``XHLAI > 1e-4`` and ``EOP > 1e-4``, else 0.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for lines 392-398 (BSD-3).
    """
    eop, trwup, xhlai = (jnp.asarray(x) for x in (eop, trwup, xhlai))
    on = (xhlai > c.lai_min_ep) & (eop > c.eop_min_ep)
    return jnp.where(on, jnp.minimum(eop, trwup * MM_PER_CM), 0.0)


def extraction_supply(
    sw: ArrayLike,
    swdelts: ArrayLike,
    swdeltu: ArrayLike,
    dlayr: ArrayLike,
    es_cm: ArrayLike,
    es_lyr_cm: ArrayLike,
    salus: ArrayLike,
) -> Array:
    """``SW_AVAIL`` before ``XTRACT`` [cm3 cm-3]: ``MAX(0, SW + SWDELTS + SWDELTU)`` less the day's
    soil evaporation (``MESEV = 'R'``: ``0.1 ES`` [cm] from layer 1; ``'S'`` (``salus``): ``0.1
    ES_LYR`` per layer), floored at 0. Padded layers (``DLAYR = 0``) have no supply.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for lines 270-273 and 424-433 (BSD-3).
    """
    sw, swdelts, swdeltu, dlayr, es_lyr = (jnp.asarray(x) for x in (sw, swdelts, swdeltu, dlayr, es_lyr_cm))
    es = jnp.asarray(es_cm)
    avail = jnp.maximum(0.0, sw + swdelts + swdeltu)
    safe = jnp.where(dlayr > 0.0, dlayr, 1.0)
    top = jnp.maximum(0.0, avail[..., 0] - es / safe[..., 0])
    ritchie = avail.at[..., 0].set(top)
    layered = jnp.maximum(0.0, avail - es_lyr / safe)
    out = jnp.where(jnp.asarray(salus, dtype=bool)[..., None], layered, ritchie)
    return jnp.where(dlayr > 0.0, out, 0.0)


def xtract(
    sw: ArrayLike,
    sw_avail: ArrayLike,
    ll: ArrayLike,
    dlayr: ArrayLike,
    trwup: ArrayLike,
    rwu: ArrayLike,
    ep: ArrayLike,
) -> XtractResult:
    """``XTRACT`` without plant-routine uptake (``UH2O = 0``): the layer extraction from the potential
    uptake ``rwu`` [cm d-1 per layer] of ``ROOTWU`` and the actual transpiration ``ep`` [mm d-1]:
    ``WUF = 0.1 EP / TRWUP`` (1 when ``0.1 EP > TRWUP``); a layer with start-of-day ``SW > LL`` gives
    ``MIN(RWU WUF, MAX(0, SW_AVAIL - LL) DLAYR)``, a drier one nothing; ``TRWU`` is the sum, and
    ``EP = 10 TRWU``. With ``EP <= 0`` nothing is extracted.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPSUBS.for XTRACT, lines 436-484 (BSD-3).
    """
    sw, sw_avail, ll, dlayr, rwu = (jnp.asarray(x) for x in (sw, sw_avail, ll, dlayr, rwu))
    trwup, ep = jnp.asarray(trwup), jnp.asarray(ep)
    avail = jnp.maximum(0.0, sw_avail - ll)
    tenth = CM_PER_MM * ep
    wuf = jnp.where(tenth <= trwup, tenth / jnp.maximum(trwup, _TRWUP_MIN), 1.0)
    r = rwu * wuf[..., None]
    safe = jnp.where(dlayr > 0.0, dlayr, 1.0)
    r = jnp.where(r / safe > avail, avail * dlayr, r)
    on = (ep > 0.0)[..., None] & (sw > ll) & (dlayr > 0.0)
    r = jnp.where(on, r, 0.0)
    trwu = jnp.sum(r, axis=-1)
    return XtractResult(uptake=r, trwu=trwu, ep=trwu * MM_PER_CM)


# ------------------------------------------------------------------------ the process
class XtractParams(Params):
    """The static soil layers ``XTRACT`` and the evaporation adjustment read (``SOILPROP``; used unless
    the forcing carries the day's soil, a ``SOILDYN`` replay), the soil evaporation method and the
    coefficients."""

    dlayr: Array = field(unit="cm", description="layer thickness", fortran_name="DLAYR", dims=_L, grid=_GRID)
    ll: Array = field(
        unit="cm3 cm-3",
        description="lower limit of plant-extractable water",
        fortran_name="LL",
        dims=_L,
        grid=_GRID,
    )
    salus_es: Array = field(
        dims=(),
        unit="-",
        description="SALUS layer-by-layer soil evaporation (MESEV = 'S')",
        fortran_name="MESEV",
    )
    coefficients: XtractCoefficients | None = field(
        description="the EP thresholds (None: the DSSAT-CSM v4.8.6.0 values)", default=None
    )

    def coef(self) -> XtractCoefficients:
        """The coefficients in force."""
        return XTRACT_COEFFICIENTS if self.coefficients is None else self.coefficients


class XtractState(State):
    """State of the extraction: ``TRWU`` of the day, and its ports::

        sw           Array              P7 soil_water.theta  read (start-of-day SW)
        flux         (swdelts, swdeltu) soil_water.flux        read (WATBAL RATE of the day)
        evaporation  EvaporationRecord  PD1 iface.evaporation  ES, ES_LYR read; EP written
        water        CropWaterIn        P1 iface.crop_water.<slot>  eop, trwup read
        rwu          Array              water_supply.<slot>.rwu     read (ROOTWU's layer RWU)
        canopy       CanopyRecord       P6 iface.canopy.<slot> read (XHLAI, yesterday's)
        sink         SinkInputs         P4 soil_water.sink_in  uptake written

    ``flux`` is the tipping bucket's flux record
    (:class:`~agrijax.processes.soil_water.bucket.BucketFluxes`; ``swdelts`` and ``swdeltu`` read).
    """

    trwu: Array = field(unit="cm d-1", dims=(), fortran_name="TRWU", description="actual root water uptake")
    sw: Array = port(
        unit="cm3 cm-3",
        dims=_L,
        grid=_GRID,
        fortran_name="SW",
        description="start-of-day layer water content (before WATBAL INTEGR)",
    )
    flux: BucketFluxes = port(description="the day's soil water rates (swdelts, swdeltu) of WATBAL RATE")
    evaporation: EvaporationRecord = port(description="the day's actual evaporation (PD1)")
    water: CropWaterIn = port(description="the crop water record (P1): eop and trwup read")
    rwu: Array = port(
        unit="cm d-1",
        dims=("n_crop", "n_layer"),
        grid=_GRID,
        fortran_name="RWU",
        description="potential root water uptake of each layer (ROOTWU)",
    )
    canopy: CanopyRecord = port(description="the crop's canopy (P6), yesterday's record: lai = XHLAI")
    sink: SinkInputs = port(description="the soil water sinks of the day (P4): uptake written")


def _day_soil(params: XtractParams, forcing_t: Any) -> tuple[Array, Array]:
    """``(DLAYR, LL)`` of the day: the ``SOILDYN`` replay of the forcing, else the static soil.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for:424-438 (SOILPROP DLAYR, LL at the SPAM call).
    """
    soil = getattr(forcing_t, "soil", None)
    if soil is None:
        return params.dlayr, params.ll
    return soil.dlayr, soil.ll


@process(
    reads=(
        "sw",
        "flux.swdelts",
        "flux.swdeltu",
        "evaporation.soil_evaporation",
        "evaporation.soil_evaporation_layers",
        "water.eop",
        "water.trwup",
        "rwu",
        "canopy.lai",
    ),
    writes=("trwu", "evaporation.transpiration", "sink.uptake"),
    source="DSSAT-CSM v4.8.6.0 SPAM/SPAM.for EP, SPAM/SPSUBS.for XTRACT (BSD-3)",
    fortran_name="XTRACT",
    key="soil_water/xtract@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid=_GRID,
    ref_build="dscsm048 v4.8.6.0 (gfortran 13, instrumented build: dumps at the SPAM entry and exit)",
    sources=(
        ("actual transpiration EP = MIN(EOP, 10 TRWUP)", "SPAM/SPAM.for:392-398"),
        ("water available for extraction after the soil evaporation", "SPAM/SPAM.for:270-273, 424-433"),
        ("layer extraction scaled by WUF, limited by SW_AVAIL - LL", "SPAM/SPSUBS.for:436-484 (XTRACT)"),
    ),
    deviates=(
        (
            "DSSAT single precision (REAL*4) is not reproduced; the SW <= LL test is made on the values "
            "given (the DSSAT day gives the REAL*4 soil limits, a labelled convention)",
            "the kernel runs in the precision of its inputs",
            "tests/integration/test_day_dssat486_free.py (XTRACT against the SPAM exit dump)",
        ),
        (
            "the plant-routine uptake branch (SUM(UH2O) > 1e-6) and the ETPHOT path are not implemented",
            "CERES-Maize passes no uptake (UH2O = 0); MEEVP = R, MEPHO = C in every supported run",
            "xtract.py module docstring",
        ),
        (
            "with several crops in the slot, EOP, TRWUP and XHLAI are summed and RWU is taken per crop",
            "one canopy per field (the PET split between crops is not defined); exact for one crop",
            "xtract.py spam_xtract docstring",
        ),
    ),
)
def spam_xtract(state: XtractState, params: XtractParams, forcing_t: Any) -> XtractState:
    """``EP`` and ``XTRACT``: today's layer extraction into P4 ``uptake`` [cm d-1], ``EP`` into the
    evaporation record [mm d-1] and ``TRWU`` into the state, from P1 ``eop`` and ``trwup``, ROOTWU's
    layer ``rwu``, yesterday's ``XHLAI`` (P6), the start-of-day ``SW``, the day's soil water rates
    and soil evaporation. Forcing read (optional): ``soil`` (the day's ``DLAYR`` and ``LL``).

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for lines 270-273, 392-398, 424-439; SPAM/SPSUBS.for XTRACT (BSD-3).
    """
    c = params.coef()
    dlayr, ll = _day_soil(params, forcing_t)
    w = state.water
    ev = state.evaporation
    eop = jnp.sum(w.eop, axis=-1)
    trwup = jnp.sum(w.trwup, axis=-1)
    xhlai = jnp.sum(state.canopy.lai, axis=-1)
    ep = actual_transpiration(eop, trwup, xhlai, c)
    avail = extraction_supply(
        state.sw,
        state.flux.swdelts,
        state.flux.swdeltu,
        dlayr,
        ev.soil_evaporation,
        ev.soil_evaporation_layers,
        params.salus_es,
    )
    res = xtract(state.sw, avail, ll, dlayr, trwup, jnp.sum(state.rwu, axis=-2), ep)
    dt = state.sink.uptake.dtype
    return eqx.tree_at(
        lambda s: (s.trwu, s.evaporation.transpiration, s.sink.uptake),
        state,
        (res.trwu.astype(dt), res.ep.astype(dt), res.uptake.astype(dt)),
    )
