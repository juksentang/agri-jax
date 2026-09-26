"""Soil-side port records: P3 node uptake, P4 sink inputs of the soil-water day.

The sink record the Richards kernel takes, :class:`SinkChannels` (with :class:`SinkChannel`
and :data:`SINK_CHANNELS`), is defined in :mod:`agrijax.processes.soil_water.sinks` and
re-exported here unchanged. It is **not** a port: a channel may carry a per-sub-step callable
(``SinkChannel.rate``), and there are no callables in state. The port is :class:`SinkInputs`,
the day's per-node amounts of every channel as plain arrays; :meth:`SinkInputs.to_sink_channels`
builds the kernel record with a static choice of active channels, so a per-sub-step sink stays a
static parameter or a variant's closure (M3 coupling contract, section 2.3 item 2).

Node uptake units (section 2.3 item 5): a node amount is **extensive**, the water depth removed
from the node's cell in a day [cm d-1] (``SinkChannel.daily``, divided by ``24 tl`` inside the
kernel). RZWQM2's ``qsr`` is an intensive rate per unit thickness; a producer that averages
layer uptake onto the nodes does so on the intensive quantity (thickness-weighted) and multiplies
by the node thickness to publish :class:`NodeUptake`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.state import State, field
from agrijax.processes.soil_water.sinks import SINK_CHANNELS, SinkChannel, SinkChannels

__all__ = ["SINK_CHANNELS", "NodeUptake", "SinkChannel", "SinkChannels", "SinkInputs"]

_N = ("n_node",)
_GRID = "rzwqm2_nodes"

#: channels that carry dissolved solute with their water (the :class:`SinkChannels` defaults)
SOLUTE_CARRYING: dict[str, bool] = {
    "uptake": False,
    "tile": True,
    "lateral": True,
    "subirrigation": True,
    "macropore_to_drain": True,
}


def _dtype(dtype: Any) -> Any:
    return dtype if dtype is not None else jnp.result_type(float)


class NodeUptake(State):
    """Root water uptake of a crop on the soil-water nodes (P3, ``iface.root_uptake.<slot>``).

    Written by the crop side at the end of the day and read by the soil side's uptake limit on
    the **next** day (the reference uses the previous day's node uptake). Extensive: water depth
    removed from each node cell per day, the meaning of ``SinkChannel.daily``.
    """

    uptake: Array = field(
        unit="cm d-1",
        description="root water uptake of each node cell (extensive)",
        fortran_name="QSR",
        dims=("n_crop", "n_node"),
        grid=_GRID,
    )

    @classmethod
    def zeros(cls, n_crop: int, n_node: int, dtype: Any = None) -> NodeUptake:
        """No uptake."""
        return cls(uptake=jnp.zeros((n_crop, n_node), dtype=_dtype(dtype)))


def _sink(description: str) -> Any:
    return field(unit="cm d-1", description=description, dims=_N, grid=_GRID)


class SinkInputs(State):
    """The day's sink amounts of the soil-water day on its nodes (P4, ``soil_water.sink_in``).

    One daily per-node amount [cm d-1] per channel of :data:`SINK_CHANNELS`, positive when it
    removes water; subirrigation adds water and is negative. Written the same day by the entries
    before the soil-water day (the uptake limit writes ``uptake``; drainage and lateral-flow
    modules the others). The all-zero record with only ``uptake`` active reproduces the single
    uptake sink of M1 bit for bit.
    """

    uptake: Array = _sink("root water uptake (all crops)")
    tile: Array = _sink("tile drainage")
    lateral: Array = _sink("lateral outflow")
    subirrigation: Array = _sink("subirrigation (negative: adds water)")
    macropore_to_drain: Array = _sink("macropore flow to drains")

    @classmethod
    def zeros(cls, n_node: int, dtype: Any = None) -> SinkInputs:
        """No sink in any channel."""
        z = jnp.zeros((n_node,), dtype=_dtype(dtype))
        return cls(uptake=z, tile=z, lateral=z, subirrigation=z, macropore_to_drain=z)

    def to_sink_channels(self, active: Sequence[str] = ("uptake",)) -> SinkChannels:
        """The kernel's :class:`SinkChannels` with the ``active`` channels (a static choice) given
        as daily amounts and every other channel statically absent (zero cost). With the default
        ``("uptake",)`` it is ``SinkChannels.from_uptake(self.uptake)``, the M1 record."""
        names = tuple(active)
        unknown = sorted(set(names) - set(SINK_CHANNELS))
        if unknown:
            raise ValueError(f"unknown sink channels {unknown} (known: {SINK_CHANNELS})")
        chans = {
            n: SinkChannel(daily=getattr(self, n), carries_solute=SOLUTE_CARRYING[n])
            for n in dict.fromkeys(names)
        }
        return SinkChannels(**chans)
