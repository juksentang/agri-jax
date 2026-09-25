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

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.units import HOURS_PER_DAY

__all__ = [
    "SINK_CHANNELS",
    "SINK_LEDGER_OUTFLOWS",
    "SinkChannel",
    "SinkChannels",
    "SinkRate",
    "as_sink_channels",
]

#: channel names, in the order they are applied (and stacked on the channel axis)
SINK_CHANNELS: tuple[str, ...] = ("subirrigation", "uptake", "tile", "lateral", "macropore_to_drain")

#: ledger outflow of each channel: ``{name: state path of its daily total}`` (for ``water_ledger``)
SINK_LEDGER_OUTFLOWS: dict[str, str] = {name: f"soil_water.flux.{name}" for name in SINK_CHANNELS}

#: per-sub-step sink ``rate(t0, dt, theta, h) -> [n_node]`` [cm h-1 per layer], positive removes water
SinkRate = Callable[[Array, Array, Array, Array], Array]


class SinkChannel(eqx.Module):
    """One sink channel: a daily per-layer amount and/or a per-sub-step callable, and its solute flag."""

    daily: Array | None = None  # [n_node] cm d-1 per layer, spread uniformly over the day
    rate: SinkRate | None = eqx.field(static=True, default=None)
    carries_solute: bool = eqx.field(static=True, default=True)

    @property
    def active(self) -> bool:
        """``False`` when the channel is statically zero (no daily amount and no callable)."""
        return self.daily is not None or self.rate is not None

    def node_rate(self, tl: Array, t0: Array, dt: Array, theta: Array, h: Array) -> Array:
        """Sink rate of the sub-step on the nodes [h-1] (``daily / (24 tl)`` plus ``rate(...) / tl``).

        Source: plan 19 A6 (sink channels); Ahuja et al. (2000) ch. 3.
        """
        out = jnp.zeros_like(theta)
        if self.daily is not None:
            out = jnp.asarray(self.daily, theta.dtype) / (HOURS_PER_DAY * tl)
        if self.rate is not None:
            out = out + jnp.asarray(self.rate(t0, dt, theta, h), theta.dtype) / tl
        return out


def _absent(carries_solute: bool = True) -> SinkChannel:
    return SinkChannel(carries_solute=carries_solute)


class SinkChannels(eqx.Module):
    """The typed sink record of the soil-water step; in M3 only ``uptake`` is non-zero.

    Defaults: ``uptake`` does not carry solute (crop N uptake is its own process), the drains,
    lateral flow and subirrigation do.
    """

    uptake: SinkChannel = eqx.field(default_factory=lambda: _absent(False))
    tile: SinkChannel = eqx.field(default_factory=_absent)
    lateral: SinkChannel = eqx.field(default_factory=_absent)
    subirrigation: SinkChannel = eqx.field(default_factory=_absent)
    macropore_to_drain: SinkChannel = eqx.field(default_factory=_absent)

    @classmethod
    def from_uptake(cls, uptake: Any) -> SinkChannels:
        """The M3 record: the day's per-layer root water uptake [cm d-1], every other channel absent."""
        return cls(uptake=SinkChannel(daily=uptake, carries_solute=False))

    def channels(self) -> tuple[SinkChannel, ...]:
        """The channels in the order :data:`SINK_CHANNELS`."""
        return tuple(getattr(self, name) for name in SINK_CHANNELS)

    @property
    def only_daily_uptake(self) -> bool:
        """``True`` when ``uptake`` is a daily array and every other channel is absent (the M1/M3 case)."""
        others = [c.active for n, c in zip(SINK_CHANNELS, self.channels()) if n != "uptake"]
        return self.uptake.rate is None and not any(others)

    def daily_uptake_only(self) -> Array | None:
        """The daily uptake array when it is the only channel (the M1/M3 case), else ``None``."""
        return self.uptake.daily if self.only_daily_uptake else None

    def solute_channels(self) -> tuple[str, ...]:
        """Names of the active channels that carry solute."""
        return tuple(n for n, c in zip(SINK_CHANNELS, self.channels()) if c.active and c.carries_solute)


def as_sink_channels(sink: Any) -> SinkChannels:
    """``sink`` as a :class:`SinkChannels`: an array is the per-layer daily uptake [cm d-1]."""
    if isinstance(sink, SinkChannels):
        return sink
    return SinkChannels.from_uptake(sink)
