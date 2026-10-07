"""The water-supply slot (``processes/water_supply``): the moved ROOTWU. Data-free.

The old path ``processes/soil_water/uptake.py`` is gone (no alias across slots), the producer is
registered from ``processes/water_supply/rootwu.py``.
"""

from __future__ import annotations

import importlib

import pytest

from agrijax.core.process import lookup
from agrijax.processes import water_supply as ws


# ------------------------------------------------------------------ the moved module
def test_old_path_is_gone_and_the_producer_lives_in_water_supply() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agrijax.processes.soil_water.uptake")
    new = importlib.import_module("agrijax.processes.water_supply.rootwu")
    assert len(new.__all__) == 13
    assert ws.rootwu_supply is new.rootwu_supply
    p = lookup("water_supply/rootwu@dssat-4.8.6.0:faithful")
    assert p is new.rootwu_supply and p.fn.__module__ == "agrijax.processes.water_supply.rootwu"
