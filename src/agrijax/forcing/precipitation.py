"""Daily precipitation of RZWQM2 4.6 from the breakpoint storms: the amount a snow day hands to PRMS.

RZWQM2 reads its rain from the ``.BRK`` breakpoint file one storm at a time (``STMINP``,
``RZWQM/Rzday.for`` 3758-3935): ``RFDNEW``, the storm's depth [cm] after the skip threshold and
the precipitation modifier. The reader owns those rules
(:func:`agrijax.io.rzwqm.storms.storm_depths` parses the file into the storm list). On a day that
enters the snow branch (a pack, or a mean air temperature at or below 0 degC) every storm of the
day is added in file order to ``RFDACC`` (``Rzday.for`` 1537-1541), which is what the PRMS
snowpack receives; the same sum is the day's ``DAYRAIN`` (``.ana`` column 3).

:func:`daily_storm_precipitation` sums the parsed storms per day in the reference's order. It is
forcing preprocessing (NumPy, not differentiable), like :mod:`.radiation`.

Not reproduced: the breakpoint-file checks of ``IMAGIC = 2`` (a data-verification run), and the
first-pass skip of the storms before the simulation start (the caller picks the dates).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = ["daily_storm_precipitation"]


def daily_storm_precipitation(storms: pd.DataFrame, dates: Any) -> np.ndarray:
    """The day's breakpoint precipitation [cm] on each of ``dates``: ``RFDNEW`` summed per day.

    ``storms`` is :func:`agrijax.io.rzwqm.storms.storm_depths` of the scenario's ``.BRK`` file
    (columns ``date`` and ``depth_cm``, in file order); storms on days outside ``dates`` are
    dropped.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(dates)))
    out = np.zeros(len(idx))
    if len(storms) == 0 or len(idx) == 0:
        return out
    ev_days = np.asarray(pd.to_datetime(storms["date"]).to_numpy(), dtype="datetime64[D]")
    depth = storms["depth_cm"].to_numpy(dtype=np.float64)
    pos = pd.Index(np.asarray(idx.to_numpy(), dtype="datetime64[D]")).get_indexer(ev_days)
    keep = pos >= 0
    np.add.at(out, pos[keep], depth[keep])  # in file order, from 0 (DAYRAIN = DAYRAIN + RFDNEW)
    return out
