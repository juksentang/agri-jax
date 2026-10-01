"""Crop-side port records: P1 crop water, P2 root record, P6 canopy record, P10 crop nitrogen; and
the post-M3 records P18 crop residue, P20 soil nitrogen on the crop layers, P21 crop nitrogen uptake,
P25 crop status (planned ports, stage ``post_m3`` in :data:`.contract.PORTS`).

``CropWaterIn`` and ``RootRecord`` are defined here and re-exported unchanged from
:mod:`agrijax.processes.water_supply.rootwu` (the ROOTWU producer, their first home), so
``from agrijax.iface.crop import CropWaterIn`` and that module name the same class. A crop
package imports its records from here, never from another slot's package.

Units follow the reference models where they differ from the internal convention, and say so on the
field: ``CropWaterIn.eop`` is DSSAT's ``EOP`` in ``mm d-1`` next to ``trwup`` in ``cm d-1``; the
crop's stress kernel converts. A consumer and a producer of the same record use the same class, so
the unit strings on both sides are the same by construction.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.state import State, field

__all__ = [
    "NPOOL_NO_CAP",
    "CanopyRecord",
    "CropNIn",
    "CropNUptake",
    "CropResidueOut",
    "CropSoilNIn",
    "CropStatus",
    "CropWaterIn",
    "RootRecord",
]

_C = ("n_crop",)
_CL = ("n_crop", "n_layer")
_L = ("n_layer",)


def _dtype(dtype: Any) -> Any:
    return dtype if dtype is not None else jnp.result_type(float)


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


class CanopyRecord(State):
    """The canopy a crop publishes each day for the PET module (P6, ``iface.canopy.<slot>``).

    Written by the crop's publish entry at the end of its day and read by the PET entry on the
    **next** day (a one-day lag: the reference computes PET before the crop runs). ``height`` is
    in ``cm``, the unit the Shuttleworth-Wallace kernel takes; a crop that keeps its height in
    ``m`` (CERES-Maize ``CANHT``) converts with :data:`agrijax.core.units.CM_PER_M` when it
    publishes. ``tlai`` is the green plus senesced leaf area index. The all-zero record is bare
    soil (the PET module computes soil evaporation only).
    """

    lai: Array = field(unit="m2 m-2", description="green leaf area index", fortran_name="LAI", dims=_C)
    tlai: Array = field(
        unit="m2 m-2", description="total (green + senesced) leaf area index", fortran_name="TLAI", dims=_C
    )
    height: Array = field(unit="cm", description="canopy height", fortran_name="HEIGHT", dims=_C)

    @classmethod
    def zeros(cls, n_crop: int, dtype: Any = None) -> CanopyRecord:
        """Bare soil: no canopy."""
        z = jnp.zeros((n_crop,), dtype=_dtype(dtype))
        return cls(lai=z, tlai=z, height=z)


#: ``NPOOL`` of the no-cap default [g plant-1]: a nitrogen pool so large that the grain-number cap
#: ``GPP = min(GPP NDEF3, NPOOL / (0.062 x 0.0095))`` (DSSAT-CSM v4.8.6.0 MZ_GROSUB.for:1557) never
#: binds (it allows 1.7e9 grains per plant); a sentinel of the record, not a model coefficient
NPOOL_NO_CAP = 1.0e6


class CropNIn(State):
    """The nitrogen factors a crop sees each day (P10, ``iface.crop_n.<slot>``).

    The outputs of CERES-Maize's ``MZ_NFACTO`` that change the crop's mass, and the pool of the
    grain-number cap, as DSSAT names them (0-1, 1 = no nitrogen limitation):

    * ``nstres`` - ``NSTRES``, on assimilation (``CARBO``);
    * ``agefac`` - ``AGEFAC``, on leaf expansion, the late stage-3 stem and the stage-4 ear growth
      (stages 2-4) and, through ``SLFN = (1 - FSLFN) + FSLFN AGEFAC``, on leaf senescence;
    * ``ndef3`` and ``npool`` - ``NDEF3`` and the plant's translocatable nitrogen ``NPOOL``
      [g plant-1], in the grain-number cap ``GPP = min(GPP NDEF3, NPOOL / (0.062 x 0.0095))`` on
      the first day of effective grain filling.

    In the RZWQM2 4.6 day the record is written by a replay of the reference run
    (``n_supply/forcing_replay@none:replay``) and read the same day by the ``nstress_replay``
    variant of the crop's growth; a nitrogen module replaces the producer later. The defaults are
    no nitrogen limitation (``1``, and ``npool =`` :data:`NPOOL_NO_CAP`): a field left out of the
    constructor (``CropNIn(nstres=x)``) gets it with the shape and dtype of ``nstres``, and with
    them the variant is the faithful nitrogen-off growth bit for bit (:meth:`initial`).
    """

    nstres: Array = field(
        unit="-",
        description="nitrogen stress factor on assimilation (1 = none)",
        fortran_name="NSTRES",
        dims=_C,
    )
    agefac: Array = field(
        unit="-",
        description="nitrogen stress factor on leaf expansion and leaf senescence (1 = none)",
        fortran_name="AGEFAC",
        dims=_C,
        default=None,
    )
    ndef3: Array = field(
        unit="-",
        description="nitrogen stress factor on the grain number (1 = none)",
        fortran_name="NDEF3",
        dims=_C,
        default=None,
    )
    npool: Array = field(
        unit="g plant-1",
        description="plant nitrogen available for translocation to the grain (caps the grain number)",
        fortran_name="NPOOL",
        dims=_C,
        default=None,
    )

    def __post_init__(self) -> None:
        # a factor left out is the no-stress default, with the shape and dtype of nstres
        base = jnp.asarray(self.nstres)
        if self.agefac is None:
            self.agefac = jnp.ones_like(base)
        if self.ndef3 is None:
            self.ndef3 = jnp.ones_like(base)
        if self.npool is None:
            self.npool = jnp.full_like(base, NPOOL_NO_CAP)

    @classmethod
    def initial(cls, n_crop: int, dtype: Any = None) -> CropNIn:
        """No nitrogen limitation (``nstres = agefac = ndef3 = 1``, ``npool =`` :data:`NPOOL_NO_CAP`)."""
        return cls(nstres=jnp.ones((n_crop,), dtype=_dtype(dtype)))

    @classmethod
    def like(
        cls, rec: CropNIn, nstres: Any, agefac: Any = None, ndef3: Any = None, npool: Any = None
    ) -> CropNIn:
        """A record with the shape and dtype of ``rec``: each given value broadcast to it (a series
        value shared by the crops of a sample), each ``None`` the no-stress default."""

        def put(x: Any, like: Array) -> Any:
            return None if x is None else jnp.asarray(x, dtype=like.dtype) * jnp.ones_like(like)

        return cls(
            nstres=jnp.asarray(nstres, dtype=rec.nstres.dtype) * jnp.ones_like(rec.nstres),
            agefac=put(agefac, rec.agefac),
            ndef3=put(ndef3, rec.ndef3),
            npool=put(npool, rec.npool),
        )


# ------------------------------------------------------------------------ post-M3 records (planned)
class CropResidueOut(State):
    """What a crop returns to the soil each day (P18, ``iface.crop_residue.<slot>``): senesced
    matter and, on a harvest day, the harvest residue, as dry matter and nitrogen; the surface part
    as one amount per crop, the roots per crop layer. Read by the organic-matter slot on the
    **next** day (DSSAT-CSM: ``SENESCE`` of day ``d`` enters SOM on day ``d + 1``). The all-zero
    record returns nothing."""

    surface_dm: Array = field(unit="kg ha-1 d-1", description="dry matter to the surface", dims=_C)
    surface_n: Array = field(unit="kg ha-1 d-1", description="nitrogen to the surface", dims=_C)
    root_dm: Array = field(
        unit="kg ha-1 d-1", description="root dry matter to each layer", dims=_CL, grid="dssat_layers"
    )
    root_n: Array = field(
        unit="kg ha-1 d-1", description="root nitrogen to each layer", dims=_CL, grid="dssat_layers"
    )
    kind: Array = field(
        unit="-", description="residue type code of the crop (as ResidueRecord.kind)", dims=_C
    )

    @classmethod
    def zeros(cls, n_crop: int, n_layer: int, dtype: Any = None) -> CropResidueOut:
        """No residue return."""
        c = jnp.zeros((n_crop,), dtype=_dtype(dtype))
        cl = jnp.zeros((n_crop, n_layer), dtype=_dtype(dtype))
        return cls(surface_dm=c, surface_n=c, root_dm=cl, root_n=cl, kind=c)


class CropSoilNIn(State):
    """The mineral nitrogen a crop sees on its layers (P20, ``iface.crop_soil_n.<slot>``): the soil
    nodes' nitrogen (P19) mapped onto the crop layers, conserving mass. Shared by the crops of a
    sample, like ``CropWaterIn.sw``."""

    no3: Array = field(
        unit="kg ha-1",
        description="nitrate nitrogen of the crop layers",
        fortran_name="NO3",
        dims=_L,
        grid="dssat_layers",
    )
    nh4: Array = field(
        unit="kg ha-1",
        description="ammonium nitrogen of the crop layers",
        fortran_name="NH4",
        dims=_L,
        grid="dssat_layers",
    )

    @classmethod
    def zeros(cls, n_layer: int, dtype: Any = None) -> CropSoilNIn:
        """No mineral nitrogen."""
        z = jnp.zeros((n_layer,), dtype=_dtype(dtype))
        return cls(no3=z, nh4=z)


class CropNUptake(State):
    """A crop's nitrogen uptake from each of its layers (P21, ``iface.n_uptake.<slot>``), written
    by the crop's nitrogen-on growth (DSSAT-CSM computes ``UNO3``/``UNH4`` in ``MZ_NUPTAK`` inside
    ``MZ_GROSUB``) and mapped onto the soil nodes by the crop interface (P22)."""

    no3: Array = field(
        unit="kg ha-1 d-1",
        description="nitrate uptake of each layer",
        fortran_name="UNO3",
        dims=_CL,
        grid="dssat_layers",
    )
    nh4: Array = field(
        unit="kg ha-1 d-1",
        description="ammonium uptake of each layer",
        fortran_name="UNH4",
        dims=_CL,
        grid="dssat_layers",
    )

    @classmethod
    def zeros(cls, n_crop: int, n_layer: int, dtype: Any = None) -> CropNUptake:
        """No uptake."""
        z = jnp.zeros((n_crop, n_layer), dtype=_dtype(dtype))
        return cls(no3=z, nh4=z)


class CropStatus(State):
    """Crop quantities read by the phosphorus module (P25, ``iface.crop_status.<slot>``): biomass by
    part, the potential biomass (biomass over the day's most limiting stress) and the fraction of
    the development cycle; written at the end of the crop's day. The all-zero record is no crop."""

    tops_dm: Array = field(unit="kg ha-1", description="above-ground dry matter", dims=_C)
    root_dm: Array = field(unit="kg ha-1", description="root dry matter", dims=_C)
    grain_dm: Array = field(unit="kg ha-1", description="grain (yield) dry matter", dims=_C)
    potential_dm: Array = field(unit="kg ha-1", description="biomass without the day's stress", dims=_C)
    dev_fraction: Array = field(unit="-", description="fraction of the development cycle completed", dims=_C)

    @classmethod
    def zeros(cls, n_crop: int, dtype: Any = None) -> CropStatus:
        """No crop."""
        z = jnp.zeros((n_crop,), dtype=_dtype(dtype))
        return cls(tops_dm=z, root_dm=z, grain_dm=z, potential_dm=z, dev_fraction=z)
