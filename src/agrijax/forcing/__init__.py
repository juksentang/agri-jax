"""Forcing preprocessing: what a reference model derives from its weather input before the day runs.

Not differentiable (NumPy; a differentiable port is deferred to a weather-forcing module).
:mod:`.radiation` rebuilds the daily radiation of RZWQM2 4.6 (RTH, the hourly disaggregation, RTS
and the SHAW cloud fraction); :mod:`.precipitation` sums the parsed breakpoint storms per day (the
amount a snow day hands to the pack; the ``.BRK`` reader :mod:`agrijax.io.rzwqm.storms` owns the
storm rules).
"""

from .precipitation import daily_storm_precipitation
from .radiation import (
    RZWQM_RADIATION,
    DailyRadiation,
    RadiationCoefficients,
    horizontal_radiation,
    hourly_horizontal_radiation,
    radiation_from_met,
    rzwqm_radiation,
    shaw_slope_partition,
)

__all__ = [
    "RZWQM_RADIATION",
    "DailyRadiation",
    "RadiationCoefficients",
    "daily_storm_precipitation",
    "horizontal_radiation",
    "hourly_horizontal_radiation",
    "radiation_from_met",
    "rzwqm_radiation",
    "shaw_slope_partition",
]
