"""Port records of the M3 coupling contract: the only types slot modules share.

A module of one slot never imports another slot's package; it imports ``agrijax.core`` and the
records here. Each record is an ordinary :class:`~agrijax.core.state.State` of arrays, placed at
its global path by :func:`agrijax.core.ports.bind`.

======  ==============================  ===========================  ==================
port    global path                     record                       module
======  ==============================  ===========================  ==================
P1      ``iface.crop_water.<slot>``     :class:`CropWaterIn`         :mod:`.crop`
P2      ``iface.root.<slot>``           :class:`RootRecord`          :mod:`.crop`
P3      ``iface.root_uptake.<slot>``    :class:`NodeUptake`          :mod:`.soil`
P4      ``soil_water.sink_in``          :class:`SinkInputs`          :mod:`.soil`
P5      ``iface.pet``                   :class:`PETFluxes`           :mod:`.surface`
P6      ``iface.canopy.<slot>``         :class:`CanopyRecord`        :mod:`.crop`
P7      ``soil_water.theta``            (the soil-water state)       --
P8      ``forcing.weather``             :class:`DailyWeather`        :mod:`.surface`
P9      ``iface.snow``                  :class:`SnowOut`             :mod:`.surface`
P10     ``iface.crop_n.<slot>``         :class:`CropNIn`             :mod:`.crop`
P11     ``ledger.water``                :class:`WaterLedger`         :mod:`agrijax.core.ledger`
======  ==============================  ===========================  ==================

Units, dims, time semantics and defaults of every field are in :data:`.contract.PORTS`.
Records defined before this package existed (``CropWaterIn``, ``RootRecord``, ``SinkChannels``,
``PETFluxes``, ``DailyWeather``, ``WaterLedger``) are re-exported, not copied: the old import
paths name the same classes.

The submodules are imported on first use, so importing ``agrijax.iface.crop`` from a crop
package does not import the PET or soil-water packages it does not need.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agrijax.core.ledger import WaterLedger

    from .contract import PORTS, AllowedLag, FieldSpec, PortSpec, allowed_lags, port_spec, record_problems
    from .crop import CanopyRecord, CropNIn, CropWaterIn, RootRecord
    from .soil import SINK_CHANNELS, NodeUptake, SinkChannel, SinkChannels, SinkInputs
    from .surface import DailyWeather, PETFluxes, SnowOut

__all__ = [
    "PORTS",
    "SINK_CHANNELS",
    "AllowedLag",
    "CanopyRecord",
    "CropNIn",
    "CropWaterIn",
    "DailyWeather",
    "FieldSpec",
    "NodeUptake",
    "PETFluxes",
    "PortSpec",
    "RootRecord",
    "SinkChannel",
    "SinkChannels",
    "SinkInputs",
    "SnowOut",
    "WaterLedger",
    "allowed_lags",
    "port_spec",
    "record_problems",
]

_WHERE: dict[str, str] = {
    **dict.fromkeys(("CanopyRecord", "CropNIn", "CropWaterIn", "RootRecord"), ".crop"),
    **dict.fromkeys(("SINK_CHANNELS", "NodeUptake", "SinkChannel", "SinkChannels", "SinkInputs"), ".soil"),
    **dict.fromkeys(("DailyWeather", "PETFluxes", "SnowOut"), ".surface"),
    **dict.fromkeys(
        ("PORTS", "AllowedLag", "FieldSpec", "PortSpec", "allowed_lags", "port_spec", "record_problems"),
        ".contract",
    ),
    "WaterLedger": "agrijax.core.ledger",
}


def __getattr__(name: str) -> Any:
    where = _WHERE.get(name)
    if where is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = (
        importlib.import_module(where, __name__) if where.startswith(".") else importlib.import_module(where)
    )
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
