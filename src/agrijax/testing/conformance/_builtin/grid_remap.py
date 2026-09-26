"""Kit fixture process: node soil water mapped onto the crop layers with :mod:`agrijax.core.grids`.

The M3 entry ``crops.<slot>.remap_in`` (``crop_iface/remap_in``, contract section 1.2 entry 8) has
no process yet; this unregistered fixture exercises the operator it will use, the intensive
(thickness-weighted) map of ``core.grids``, through the conformance kit: the day's node water
content comes from the forcing (a replay of the soil-water state), the layer water content goes
into the crop water record (port P1, field ``sw``). The module holds only the process, so the
kit's lint covers exactly it.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from agrijax.core.grids import SoilGrid, remap_intensive
from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field
from agrijax.iface.crop import CropWaterIn

__all__ = ["KEY", "RemapForcing", "RemapParams", "RemapState", "remap_in_fixture"]

KEY = "crop_iface/remap_in@none:kit_fixture"


class RemapState(State):
    """The node water of the day and the crop water record it is mapped into."""

    theta: Array = field(
        unit="cm3 cm-3", description="water content of the soil nodes", dims=("n_node",), grid="rzwqm2_nodes"
    )
    water_out: CropWaterIn = port(description="crop water record: sw written")


class RemapParams(Params):
    """The two static grids."""

    nodes: SoilGrid = field(description="the soil-water node grid", static=True)
    layers: SoilGrid = field(description="the crop layers (LYRSET)", static=True)


class RemapForcing(Forcing):
    """The day's node water content (a replay of the soil-water state)."""

    theta: Array = field(
        unit="cm3 cm-3", description="replay: node water content of the day", dims=("T", "n_node")
    )


def _remap_in(state: RemapState, params: RemapParams, forcing_t: RemapForcing) -> RemapState:
    """``sw = remap_intensive(theta)``: the thickness-weighted mean of the node water content over
    each crop layer (``LMATCH``), written into the crop water record.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/LMATCH.for (LMATCH), as agrijax.core.grids.remap.
    """
    theta = forcing_t.theta
    sw = remap_intensive(theta, params.nodes, params.layers).astype(state.water_out.sw.dtype)
    return eqx.tree_at(lambda s: (s.theta, s.water_out.sw), state, (theta, sw))


remap_in_fixture = process(
    _remap_in,
    reads=(),
    writes=("theta", "water_out.sw"),
    register=False,
    name="kit_remap_in_fixture",
    key=KEY,
    provenance="equations_only",
    grid="rzwqm2_lyrset",
    sources=(("thickness-weighted mean over each layer", "DSSAT-CSM LMATCH.for; agrijax.core.grids"),),
    deviates=(),
)
