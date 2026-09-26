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

Units, dims, time semantics and defaults of every field are in :data:`.contract.PORTS`; the
contract's day (entries, phases, producer keys) is :data:`.contract.DAY_TABLE`.
Records defined before this package existed (``CropWaterIn``, ``RootRecord``, ``SinkChannels``,
``PETFluxes``, ``DailyWeather``) are now defined here and re-exported unchanged from their old
modules (``processes/soil_water/uptake.py``, ``processes/soil_water/sinks.py``,
``processes/pet/daily.py``): both import paths name the same classes. ``WaterLedger`` stays in
:mod:`agrijax.core.ledger` and is re-exported.

``agrijax.iface`` imports ``agrijax.core`` and third-party packages only, never
``agrijax.processes`` (lint rule AJ008 per file; ``tests/unit/test_iface.py`` transitively), so
a slot package can import its records without pulling in another slot. P7 is a field of the
soil-water slot's own state: its spec names the holder class by dotted path
(:attr:`.contract.PortSpec.field_of`) instead of importing it. The submodules are imported on
first use.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agrijax.core.ledger import WaterLedger

    from .contract import (
        DAY_TABLE,
        PORTS,
        AllowedLag,
        DayEntry,
        FieldSpec,
        PortSpec,
        allowed_lags,
        day_entries,
        port_spec,
        record_problems,
    )
    from .crop import CanopyRecord, CropNIn, CropWaterIn, RootRecord
    from .soil import SINK_CHANNELS, NodeUptake, SinkChannel, SinkChannels, SinkInputs
    from .surface import DailyWeather, PETFluxes, SnowOut

__all__ = [
    "DAY_TABLE",
    "PORTS",
    "SINK_CHANNELS",
    "AllowedLag",
    "CanopyRecord",
    "CropNIn",
    "CropWaterIn",
    "DailyWeather",
    "DayEntry",
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
    "day_entries",
    "port_spec",
    "record_problems",
]

_WHERE: dict[str, str] = {
    **dict.fromkeys(("CanopyRecord", "CropNIn", "CropWaterIn", "RootRecord"), ".crop"),
    **dict.fromkeys(("SINK_CHANNELS", "NodeUptake", "SinkChannel", "SinkChannels", "SinkInputs"), ".soil"),
    **dict.fromkeys(("DailyWeather", "PETFluxes", "SnowOut"), ".surface"),
    **dict.fromkeys(
        (
            "DAY_TABLE",
            "PORTS",
            "AllowedLag",
            "DayEntry",
            "FieldSpec",
            "PortSpec",
            "allowed_lags",
            "day_entries",
            "port_spec",
            "record_problems",
        ),
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
