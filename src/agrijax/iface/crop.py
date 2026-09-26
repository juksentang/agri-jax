"""Crop-side port records: P1 crop water, P2 root record, P6 canopy record, P10 crop nitrogen.

``CropWaterIn`` and ``RootRecord`` are defined here and re-exported unchanged from
:mod:`agrijax.processes.soil_water.uptake` (where the M3 minimal set introduced them), so ``from
agrijax.iface.crop import CropWaterIn`` and the old import path name the same class. A crop
package imports its records from here, never from another slot's package.

Units follow the reference models where they differ from the internal convention, and say so
on the field (M3 coupling contract, section 2.3): ``CropWaterIn.eop`` is DSSAT's ``EOP`` in
``mm d-1`` next to ``trwup`` in ``cm d-1``; the crop's stress kernel converts. A consumer and a
producer of the same record use the same class, so the unit strings on both sides are the same
by construction.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.state import State, field

__all__ = ["CanopyRecord", "CropNIn", "CropWaterIn", "RootRecord"]

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


class CropNIn(State):
    """The nitrogen stress a crop sees each day (P10, ``iface.crop_n.<slot>``).

    ``nstres`` is DSSAT's ``NSTRES`` (0-1, 1 = no nitrogen limitation). In M3 it is written by a
    replay of the reference run (``n_supply/forcing_replay@none:replay``) and read the same day
    by the ``nstress_replay`` variant of the crop's growth; a nitrogen module replaces the
    producer later. :meth:`initial` (``nstres = 1``) makes the variant equal to the faithful
    nitrogen-off growth.
    """

    nstres: Array = field(
        unit="-",
        description="nitrogen stress factor on assimilation (1 = none)",
        fortran_name="NSTRES",
        dims=_C,
    )

    @classmethod
    def initial(cls, n_crop: int, dtype: Any = None) -> CropNIn:
        """No nitrogen stress (``nstres = 1``)."""
        return cls(nstres=jnp.ones((n_crop,), dtype=_dtype(dtype)))
