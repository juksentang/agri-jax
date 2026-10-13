"""Soil-side port records: P3 node uptake, P4 sink inputs of the soil-water day; and the post-M3
records P12 soil water fluxes, P13 water step trace, P15 soil ice, P19 soil mineral nitrogen, P22 node
nitrogen uptake, P23 drain control (planned ports, stage ``post_m3`` in :data:`.contract.PORTS`).

The sink record the Richards kernel takes, :class:`SinkChannels` (with :class:`SinkChannel`,
:data:`SinkRate` and :data:`SINK_CHANNELS`), is defined here, because :class:`SinkInputs` builds
it, and re-exported unchanged from :mod:`agrijax.processes.soil_water.sinks` (its first home;
both paths name the same classes). It is **not** a port: a channel may carry a per-sub-step callable
(``SinkChannel.rate``), and there are no callables in state. The port is :class:`SinkInputs`,
the day's per-node amounts of every channel as plain arrays; :meth:`SinkInputs.to_sink_channels`
builds the kernel record with a static choice of active channels, so a per-sub-step sink stays a
static parameter or a variant's closure.

Node uptake units: a node amount is **extensive**, the water depth removed from the node's cell in a
day [cm d-1] (``SinkChannel.daily``, divided by ``24 tl`` inside the kernel). RZWQM2's ``qsr`` is an
intensive rate per unit thickness; a producer that averages layer uptake onto the nodes does so on
the intensive quantity (thickness-weighted) and multiplies by the node thickness to publish
:class:`NodeUptake`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.dims import register_dim
from agrijax.core.state import State, field
from agrijax.core.units import HOURS_PER_DAY

__all__ = [
    "SINK_CHANNELS",
    "SOLUTE_CARRYING",
    "DrainControl",
    "NodeNUptake",
    "NodeUptake",
    "SinkChannel",
    "SinkChannels",
    "SinkInputs",
    "SinkRate",
    "SoilFluxRecord",
    "SoilIce",
    "SoilMineralN",
    "WaterStepTrace",
]

#: sub-steps of one soil-water day, padded to the day's static maximum (the step table of the
#: adaptive Richards day, or the fixed sub-steps); an empty step has ``dt = 0``
register_dim("n_step", "sub-steps of one soil-water day, padded to a static maximum (dt = 0: empty)")

_N = ("n_node",)
_GRID = "rzwqm2_nodes"

#: channel names, in the order they are applied (and stacked on the channel axis)
SINK_CHANNELS: tuple[str, ...] = ("subirrigation", "uptake", "tile", "lateral", "macropore_to_drain")

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

        Source: Ahuja et al. (2000) ch. 3 (the Richards-equation soil-water model); the split into sink
        channels is this package's own design.
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
    """The typed sink record of the soil-water step; in the RZWQM2 4.6 day only ``uptake`` is non-zero.

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
        """The uptake-only record: the day's per-layer root water uptake [cm d-1], nothing else."""
        return cls(uptake=SinkChannel(daily=uptake, carries_solute=False))

    def channels(self) -> tuple[SinkChannel, ...]:
        """The channels in the order :data:`SINK_CHANNELS`."""
        return tuple(getattr(self, name) for name in SINK_CHANNELS)

    @property
    def only_daily_uptake(self) -> bool:
        """``True`` when ``uptake`` is a daily array and all other channels are absent (uptake-only case)."""
        others = [c.active for n, c in zip(SINK_CHANNELS, self.channels()) if n != "uptake"]
        return self.uptake.rate is None and not any(others)

    def daily_uptake_only(self) -> Array | None:
        """The daily uptake array when it is the only channel (the uptake-only case), else ``None``."""
        return self.uptake.daily if self.only_daily_uptake else None

    def solute_channels(self) -> tuple[str, ...]:
        """Names of the active channels that carry solute."""
        return tuple(n for n, c in zip(SINK_CHANNELS, self.channels()) if c.active and c.carries_solute)


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
    modules the others). The all-zero record with only ``uptake`` active reproduces, bit for bit, the
    single uptake sink of the soil-water day validated with uptake as its only sink.
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
        ``("uptake",)`` it is ``SinkChannels.from_uptake(self.uptake)``, the record of that validated
        uptake-only day."""
        names = tuple(active)
        unknown = sorted(set(names) - set(SINK_CHANNELS))
        if unknown:
            raise ValueError(f"unknown sink channels {unknown} (known: {SINK_CHANNELS})")
        chans = {
            n: SinkChannel(daily=getattr(self, n), carries_solute=SOLUTE_CARRYING[n])
            for n in dict.fromkeys(names)
        }
        return SinkChannels(**chans)


# ------------------------------------------------------------------------ post-M3 records (planned)
def _nodes(unit: str, description: str, fortran_name: str = "") -> Any:
    return field(unit=unit, description=description, fortran_name=fortran_name, dims=_N, grid=_GRID)


class SoilFluxRecord(State):
    """The soil-water day's water movement on its nodes (P12, ``iface.soil_flux``), for the modules
    that move solutes or heat with the water (nitrogen, phosphorus, soil heat).

    Written by the soil-water day at its end and read the same day by the modules after it. Every
    amount is a daily total. ``q_down[i]`` is the net water that crossed the **bottom** face of node
    ``i`` downward by matrix flow (negative: upward); the face above node ``i > 0`` is the bottom
    face of node ``i - 1``, the soil surface carries ``q_surface`` (matrix infiltration minus
    evaporation), and ``event`` is the water the day's infiltration event placed in each node
    instantly (Green-Ampt: no face flux), so a node's water balance is
    ``tl (theta_end - theta_start) = q_in - q_down + event - sum(sinks)``, and ``q_down[-1]`` is the
    deep drainage. The sink amounts are the **actual** per-node amounts of
    each channel after the ``h_min`` cut (extensive, the meaning of ``SinkChannel.daily``); whether
    a channel carries solute is :data:`SOLUTE_CARRYING`. The all-zero record is no water movement.
    """

    q_down: Array = _nodes("cm d-1", "net downward water flux through the bottom face of each node", "TQFL")
    q_surface: Array = field(
        unit="cm d-1",
        description="net downward water flux at the soil surface (infiltration - evaporation)",
        dims=(),
    )
    theta_start: Array = _nodes("cm3 cm-3", "node water content at the start of the day")
    event: Array = _nodes("cm d-1", "water the day's infiltration event placed in each node cell")
    runoff: Array = field(unit="cm d-1", description="surface runoff of the day", fortran_name="RO", dims=())
    uptake: Array = _nodes("cm d-1", "actual root water uptake of each node cell (all crops)")
    tile: Array = _nodes("cm d-1", "actual tile drainage of each node cell", "UDRN")
    lateral: Array = _nodes("cm d-1", "actual lateral outflow of each node cell", "ULAT")
    subirrigation: Array = _nodes("cm d-1", "actual subirrigation of each node cell (negative: adds water)")
    macropore_to_drain: Array = _nodes("cm d-1", "actual macropore flow to the drains of each node cell")

    @classmethod
    def zeros(cls, n_node: int, dtype: Any = None) -> SoilFluxRecord:
        """No water movement."""
        z = jnp.zeros((n_node,), dtype=_dtype(dtype))
        s = jnp.zeros((), dtype=_dtype(dtype))
        return cls(
            q_down=z,
            q_surface=s,
            theta_start=z,
            event=z,
            runoff=s,
            uptake=z,
            tile=z,
            lateral=z,
            subirrigation=z,
            macropore_to_drain=z,
        )


_S = ("n_step",)
_SN = ("n_step", "n_node")


class WaterStepTrace(State):
    """The soil-water day's sub-steps (P13, ``iface.water_trace``), for the consumers that the
    reference advances inside the water time loop (the RZWQM2 soil heat ``HEATFX`` averages the
    water flux and sink over the water steps it accumulates; the reference's solute transport is
    assumed to follow the water steps too). Padded to a static number of steps; an empty step has
    ``dt = 0`` and zero fluxes. Rates are per hour over the step. The day's infiltration event is
    instantaneous and between steps: ``event`` is the water it placed in each node at ``t_event``."""

    t0: Array = field(unit="h", description="start of each sub-step within the day", dims=_S)
    dt: Array = field(unit="h", description="length of each sub-step (0: empty)", dims=_S)
    q_down: Array = field(
        unit="cm h-1",
        description="downward water flux through the bottom face of each node over the step",
        dims=_SN,
        grid=_GRID,
    )
    q_surface: Array = field(
        unit="cm h-1", description="net downward flux at the surface over the step", dims=_S
    )
    sink: Array = field(
        unit="cm h-1", description="sink of each node cell over the step (all channels)", dims=_SN, grid=_GRID
    )
    theta: Array = field(
        unit="cm3 cm-3", description="node water content at the end of each step", dims=_SN, grid=_GRID
    )
    t_event: Array = field(unit="h", description="time of the day's infiltration event (0 without)", dims=())
    event: Array = field(
        unit="cm", description="water the infiltration event placed in each node cell", dims=_N, grid=_GRID
    )

    @classmethod
    def zeros(cls, n_step: int, n_node: int, dtype: Any = None) -> WaterStepTrace:
        """An empty trace (every step empty)."""
        dt = _dtype(dtype)
        s = jnp.zeros((n_step,), dtype=dt)
        sn = jnp.zeros((n_step, n_node), dtype=dt)
        return cls(
            t0=s,
            dt=s,
            q_down=sn,
            q_surface=s,
            sink=sn,
            theta=sn,
            t_event=jnp.zeros((), dtype=dt),
            event=jnp.zeros((n_node,), dtype=dt),
        )


class SoilIce(State):
    """Ice content of the soil nodes (P15, ``iface.soil_ice``), written by the energy slot and read
    by the soil-water day on the **next** day (a declared lag: the reference with frozen soil
    co-integrates heat and water within the day). ``theta_ice = 0`` everywhere is unfrozen soil,
    and the soil-water day then runs exactly as without the port."""

    theta_ice: Array = _nodes("cm3 cm-3", "volumetric ice content of each node", "THETAI")

    @classmethod
    def zeros(cls, n_node: int, dtype: Any = None) -> SoilIce:
        """Unfrozen soil."""
        return cls(theta_ice=jnp.zeros((n_node,), dtype=_dtype(dtype)))


class SoilMineralN(State):
    """Mineral nitrogen of the soil nodes (P19, ``iface.soil_n``), extensive per node cell, written
    by the soil organic matter and nitrogen module after its day and mapped onto a crop's layers by
    the crop interface (P20)."""

    no3: Array = _nodes("kg ha-1", "nitrate nitrogen of each node cell", "NO3")
    nh4: Array = _nodes("kg ha-1", "ammonium nitrogen of each node cell", "NH4")

    @classmethod
    def zeros(cls, n_node: int, dtype: Any = None) -> SoilMineralN:
        """No mineral nitrogen."""
        z = jnp.zeros((n_node,), dtype=_dtype(dtype))
        return cls(no3=z, nh4=z)


class NodeNUptake(State):
    """A crop's nitrogen uptake on the soil nodes (P22, ``iface.root_n_uptake.<slot>``), the
    extensive per-node amount removed in a day. Written by the crop side at the end of its day and
    subtracted by the nitrogen module on the **next** day (the lag of both references)."""

    no3: Array = field(
        unit="kg ha-1 d-1",
        description="nitrate uptake of each node cell",
        fortran_name="UNO3",
        dims=("n_crop", "n_node"),
        grid=_GRID,
    )
    nh4: Array = field(
        unit="kg ha-1 d-1",
        description="ammonium uptake of each node cell",
        fortran_name="UNH4",
        dims=("n_crop", "n_node"),
        grid=_GRID,
    )

    @classmethod
    def zeros(cls, n_crop: int, n_node: int, dtype: Any = None) -> NodeNUptake:
        """No uptake."""
        z = jnp.zeros((n_crop, n_node), dtype=_dtype(dtype))
        return cls(no3=z, nh4=z)


class DrainControl(State):
    """The drain system seen by the soil-water day (P23, ``iface.drain``), written each day by the
    drainage slot's control entry from its parameters and the event table (a controlled-drainage
    headgate replaces the drain depth). The soil-water day reads it through the sub-step sink
    kernel of the drainage slot (the per-step tile flux of the reference); without the port the
    tile channel is statically absent."""

    depth: Array = field(
        unit="cm", description="drain (or headgate) depth below the surface", fortran_name="DRDEP", dims=()
    )
    spacing: Array = field(unit="cm", description="drain spacing", fortran_name="DRSPAC", dims=())
    radius: Array = field(unit="cm", description="drain radius", fortran_name="DRRAD", dims=())
    impermeable_depth: Array = field(
        unit="cm", description="depth of the impermeable layer below the surface", dims=()
    )
    k_lat: Array = _nodes("cm h-1", "lateral saturated hydraulic conductivity of each node", "CLAT")

    @classmethod
    def zeros(cls, n_node: int, dtype: Any = None) -> DrainControl:
        """No drains (depth 0: no head above the drain, no tile flow)."""
        s = jnp.zeros((), dtype=_dtype(dtype))
        return cls(
            depth=s, spacing=s, radius=s, impermeable_depth=s, k_lat=jnp.zeros((n_node,), dtype=_dtype(dtype))
        )
