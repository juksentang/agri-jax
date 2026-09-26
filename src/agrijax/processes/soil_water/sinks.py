"""Typed sink channels of the Richards equation (plan 19 A6).

The sink term ``S`` of the Richards equation (``d theta / dt = -dq/dz - S``, positive for
extraction) is a record of named **channels** rather than one array, so that tile drainage,
lateral flow, subirrigation and macropore flow to drains can be added later (plan 19 L13, the
per-sub-step sink L-SINK) without a new Richards signature. Each channel gets its own daily
total in the fluxes (:class:`~agrijax.processes.soil_water.richards.SoilWaterFluxes`) and its own
term in the water balance and the ledger (:data:`SINK_LEDGER_OUTFLOWS`).

A channel (:class:`SinkChannel`) is given as

* ``daily``: a per-layer amount for the day [cm d-1] (``[n_node]``), spread uniformly over the
  24 h; or
* ``rate``: a per-sub-step callable ``rate(t0, dt, theta, h) -> [n_node]`` [cm h-1 per layer],
  evaluated at the start of every sub-step (``t0`` [h] the sub-step start, ``dt`` [h] its length,
  ``theta``/``h`` the node state at its start); or both (added);

and carries a static ``carries_solute`` flag (the channel moves dissolved solute with its water;
the solute modules will read it). A channel with neither is statically zero and costs nothing.

Sign: every channel is a sink, positive when it removes water from the soil. Subirrigation adds
water and is therefore given (and reported) as a negative sink.

``h_min`` cut: the channels are applied in the order :data:`SINK_CHANNELS`, each capped by the
water still available above ``theta(h_min)`` in the node after the channels before it (sources
first, then uptake, then the drains), and the cut part of each is reported. With every channel but
``uptake`` absent this is exactly the single-sink cut of M1, bit for bit.

In M3 only ``uptake`` is non-zero (RZWQM2 with a DSSAT crop and ``ISTRESS = 0``: the day's
per-layer uptake spread uniformly).

Source: plan 19 A6 (private design note); Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J.,
Ma, L. (eds.), 2000. Root Zone Water Quality Model, ch. 3 (sink terms of the Richards equation).
"""

from __future__ import annotations

from typing import Any

# the channel records are defined in agrijax.iface.soil (the P4 record SinkInputs builds them);
# re-exported here unchanged, so both import paths name the same classes
from agrijax.iface.soil import SINK_CHANNELS, SinkChannel, SinkChannels, SinkRate

__all__ = [
    "SINK_CHANNELS",
    "SINK_LEDGER_OUTFLOWS",
    "SinkChannel",
    "SinkChannels",
    "SinkRate",
    "as_sink_channels",
]

#: ledger outflow of each channel: ``{name: state path of its daily total}`` (for ``water_ledger``)
SINK_LEDGER_OUTFLOWS: dict[str, str] = {name: f"soil_water.flux.{name}" for name in SINK_CHANNELS}


def as_sink_channels(sink: Any) -> SinkChannels:
    """``sink`` as a :class:`SinkChannels`: an array is the per-layer daily uptake [cm d-1]."""
    if isinstance(sink, SinkChannels):
        return sink
    return SinkChannels.from_uptake(sink)
