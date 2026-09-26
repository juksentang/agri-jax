"""Crop-side port records: P1 crop water, P2 root record, P6 canopy record, P10 crop nitrogen.

``CropWaterIn`` and ``RootRecord`` are defined in :mod:`agrijax.processes.soil_water.uptake`
(where the M3 minimal set introduced them) and re-exported here unchanged, so ``from
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
from agrijax.processes.soil_water.uptake import CropWaterIn, RootRecord

__all__ = ["CanopyRecord", "CropNIn", "CropWaterIn", "RootRecord"]

_C = ("n_crop",)


def _dtype(dtype: Any) -> Any:
    return dtype if dtype is not None else jnp.result_type(float)


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
