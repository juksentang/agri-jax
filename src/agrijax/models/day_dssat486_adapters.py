"""The processes of the DSSAT day's crop-interface adapters ``crops.<slot>.layers_in`` (row D10 of
:data:`agrijax.iface.contract.DSSAT_DAY_TABLE`) and ``crops.<slot>.canopy`` (row D16).

Kept apart from :mod:`agrijax.models.day_dssat486` (which builds the day) so that the
conformance kit's lint, which checks every function of the process module as a kernel, covers
exactly these processes.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import State
from agrijax.core.units import CM_PER_M
from agrijax.iface.crop import CanopyRecord, CropWaterIn
from agrijax.iface.surface import SnowOut

__all__ = [
    "CANOPY_KEY",
    "LAYERS_IN_KEY",
    "CanopyInState",
    "LayersInState",
    "canopy_from_ceres",
    "layers_in",
]

#: registry key of the ``crops.<slot>.layers_in`` adapter (not registered: an assembly adapter)
LAYERS_IN_KEY = "crop_iface/layers_in@dssat-4.8.6.0:faithful"


class LayersInState(State):
    """The records of ``crops.<slot>.layers_in``: the soil's layer water content (P7) and snow pack
    (read), the crop water record (P1, ``sw`` written) and the crop's snow record (P9, ``swe``
    written)."""

    theta: Array = port(
        unit="cm3 cm-3",
        dims=("n_node",),
        fortran_name="SW",
        description="the soil's end-of-day water content on its core grid (P7; the DSSAT layers)",
    )
    snow: Array = port(
        unit="mm",
        dims=(),
        fortran_name="SNOW",
        description="the soil's snow pack after the day's WATBAL RATE (soil_water.snow)",
    )
    water_out: CropWaterIn = port(description="crop water record: sw written")
    snow_out: SnowOut = port(description="the crop's snow record (P9): swe written")


def _layers_in(state: LayersInState, params: Any, forcing_t: Any) -> LayersInState:
    """``P1 sw = P7 theta`` and ``P9 swe = SNOW``: in DSSAT-CSM the crop layers are the soil
    layers, and PLANT receives the soil module's ``SW`` array as it is (no remapping, unlike
    RZWQM2's ``REALMATCH``) and the snow pack ``SNOW`` of the soil's WATBAL (the growing-point
    temperature of MZ_PHENOL reads it).

    Source: DSSAT-CSM v4.8.6.0 CSM_Main/LAND.for:386 (``CALL PLANT(..., SNOW, ..., SW, ...)``), BSD-3.
    """
    sw = state.theta.astype(state.water_out.sw.dtype)
    swe = state.snow.astype(state.snow_out.swe.dtype)
    return eqx.tree_at(lambda s: (s.water_out.sw, s.snow_out.swe), state, (sw, swe))


#: the adapter as an unregistered process (an assembly adapter, not a slot implementation; built with a
#: call rather than the decorator, like the kit fixtures, so the registry's decorated set is unchanged)
layers_in = process(
    _layers_in,
    name="layers_in",
    reads=("theta", "snow"),
    writes=("water_out.sw", "snow_out.swe"),
    source="DSSAT-CSM v4.8.6.0 CSM_Main/LAND.for (PLANT receives SOIL's SW and SNOW; BSD-3)",
    fortran_name="SW",
    key=LAYERS_IN_KEY,
    register=False,
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        ("the crop reads the soil's layer water content as it is", "CSM_Main/LAND.for:386 (PLANT, SW)"),
        ("the crop reads the snow pack of the soil's WATBAL", "CSM_Main/LAND.for:386 (PLANT, SNOW)"),
    ),
    deviates=(),
)


#: registry key of the ``crops.<slot>.canopy`` adapter (not registered: an assembly adapter, in force
#: until the crop's own DSSAT canopy producer ``crop/ceres_maize.canopy@dssat-4.8.6.0`` lands)
CANOPY_KEY = "crop_iface/canopy_from_ceres@dssat-4.8.6.0:faithful"


class CanopyInState(State):
    """The records of ``crops.<slot>.canopy``: the crop's end-of-day leaf area index and height (read)
    and the canopy record (P6, written)."""

    lai: Array = port(
        unit="m2 m-2",
        dims=("n_crop",),
        fortran_name="LAI",
        description="the crop's green leaf area index at the end of its day (CERES-Maize growth.lai)",
    )
    canht: Array = port(
        unit="m",
        dims=("n_crop",),
        fortran_name="CANHT",
        description="the crop's canopy height (growth.canht)",
    )
    canopy_out: CanopyRecord = port(description="canopy record (P6): lai, tlai, height written")


def _canopy_from_ceres(state: CanopyInState, params: Any, forcing_t: Any) -> CanopyInState:
    """P6 from the crop's end-of-day state as ``PLANT`` hands it to ``SPAM``: ``XLAI = XHLAI = LAI``
    (so ``lai = tlai = LAI``) and the canopy height ``CANHT`` in cm.

    Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for:1818-1819 (XLAI = LAI, XHLAI = LAI),
    CSM_Main/LAND.for:386 (PLANT outputs XHLAI, XLAI, CANHT to SPAM), BSD-3.
    """
    old = state.canopy_out
    dt = old.lai.dtype
    lai = state.lai.astype(dt)
    rec = CanopyRecord(lai=lai, tlai=lai, height=(state.canht * CM_PER_M).astype(dt))
    return eqx.tree_at(lambda s: s.canopy_out, state, rec)


#: the canopy adapter as an unregistered process (an assembly adapter, like :data:`layers_in`)
canopy_from_ceres = process(
    _canopy_from_ceres,
    name="canopy_from_ceres",
    reads=("lai", "canht"),
    writes=("canopy_out",),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_GROSUB.for (XLAI = XHLAI = LAI; BSD-3)",
    fortran_name="XHLAI",
    key=CANOPY_KEY,
    register=False,
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        ("XLAI = XHLAI = LAI at the end of MZ_GROSUB INTEGR", "Plant/CERES-Maize/MZ_GROSUB.for:1818-1819"),
        ("SPAM reads the canopy PLANT published on its previous INTEGR call", "CSM_Main/LAND.for:325-386"),
    ),
    deviates=(),
)
