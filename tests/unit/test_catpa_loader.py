"""CA-TPA loaders on synthetic snippets: LAYER.PLT reader and MANAGE.OUT events (no data).

The data-backed checks (reference runs, events against MANAGE.OUT, the 100k-run LHS arrays) are in
``tests/integration/test_catpa_loader_data.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agri_jax.io.catpa import (
    read_manage_out,
)
from agri_jax.io.rzwqm.layers import (
    layer_thickness_cm,
    profile_storage_cm,
    read_layer_output,
    simulation_start,
)

_LAYER_PLT = """RZWQM2-3D
*****************************
* Subplot Definitions     *
*****************************
    2    1
*****************************
*   COLUMN LABELS     *
*****************************
DAY
DEPTH (CM)
SOIL WATER CONTENT (VOL)
PRESSURE HEAD (CM)
****************** DATA STARTS HERE *****************
          1      1.00000       0.179042       -2933.61
          1      3.00000       0.185187       -2354.80
          1      6.00000       0.594117E-001  -2101.67
          2      1.00000       0.175250       -3378.21
          2      3.00000       0.181939       -2641.34
          2      6.00000       0.185910       -2296.27
"""

_MANAGE = """*RZWQM2 OUTPUTS Version 4.6

---- 14/ 4/2015 -----   104  ----

EVENT ==> SPECIFIC DATE TILLAGE EVENT
          WITH IMPLEMENT: CHISEL PLOW-STRAIGHT
---- 27/ 4/2015 -----   117  ----

EVENT ==> PESTICIDE APPLICATION-BROADCAST (LEAVE ON SURFACE)
          AMOUNT OF GLYPHOSATE ISOPROPYLAMINE SAL2.5000    [KG/HA]
---- 28/ 4/2015 -----   118  ----

EVENT ==> FERTILIZING BROADCAST (LEAVE ON SURFACE)
          AMOUNT OF NH4 [KG/HA]          0.0000
          AMOUNT OF NO3 [KG/HA]          180.00
EVENT ==> PLANTING OPERATION
            PLANTING DENSITY:      80000.0
----DSSAT Crop Harvest----MAIZE IB0012 PIO 3382
    ON  25/ 9/2015 -----   268  ----
"""


# ------------------------------------------------------------------ synthetic
def test_read_layer_output_synthetic(tmp_path: Path) -> None:
    p = tmp_path / "LAYER.PLT"
    p.write_text(_LAYER_PLT)
    ds = read_layer_output(p, start="2015-01-01")
    assert ds["soil_water_content"].dims == ("time", "depth")
    assert ds.sizes == {"time": 2, "depth": 3}
    assert ds["soil_water_content"].attrs["units"] == "VOL"
    assert ds["pressure_head"].attrs["units"] == "CM"
    np.testing.assert_array_equal(ds["depth"], [1.0, 3.0, 6.0])
    np.testing.assert_array_equal(ds["thickness"], [1.0, 2.0, 3.0])
    assert ds["soil_water_content"].values[0, 2] == pytest.approx(0.0594117)
    assert list(ds["time"].values.astype("datetime64[D]").astype(str)) == ["2015-01-01", "2015-01-02"]
    s = profile_storage_cm(ds).values
    assert s[0] == pytest.approx(0.179042 + 2 * 0.185187 + 3 * 0.0594117)


def test_read_layer_output_start_from_ipnames(tmp_path: Path) -> None:
    (tmp_path / "LAYER.PLT").write_text(_LAYER_PLT)
    (tmp_path / "IPNAMES.DAT").write_text("\n".join(["x"] * 8 + ["1  3  2016  31  12  2016", "0 0"]) + "\n")
    assert simulation_start(tmp_path / "IPNAMES.DAT") == np.datetime64("2016-03-01")
    ds = read_layer_output(tmp_path / "LAYER.PLT")
    assert str(ds["time"].values[1])[:10] == "2016-03-02"


def test_read_layer_output_rejects_ragged(tmp_path: Path) -> None:
    p = tmp_path / "LAYER.PLT"
    p.write_text(_LAYER_PLT.rsplit("\n", 2)[0] + "\n")  # drop the last node of day 2
    with pytest.raises(ValueError, match="node counts"):
        read_layer_output(p, start="2015-01-01")
    np.testing.assert_array_equal(layer_thickness_cm(np.array([2.0, 5.0])), [2.0, 3.0])


def test_read_manage_out_synthetic(tmp_path: Path) -> None:
    p = tmp_path / "MANAGE.OUT"
    p.write_text(_MANAGE)
    m = read_manage_out(p)
    got = [(str(d)[:10], e, v) for d, e, v in zip(m["date"], m["event"], m["value"], strict=True)]
    assert got[0][:2] == ("2015-04-14", "tillage")
    assert ("2015-04-27", "pesticide", 2.5) in got
    assert ("2015-04-28", "fertilizer_no3", 180.0) in got
    assert ("2015-04-28", "fertilizer_nh4", 0.0) in got
    assert ("2015-04-28", "planting", 80000.0) in got
    assert got[-1][:2] == ("2015-09-25", "harvest")
    assert m.loc[m["event"] == "tillage", "detail"].item() == "chisel plow-straight"
