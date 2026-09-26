"""H1-5: the Green-Ampt event against RZWQM2 4.6 ``EVNTRO``/``INFIL`` dumps on 19 scenarios (slow).

Every rain storm of the whole ``IPNAMES.DAT`` period of the 15 ``RZWQM_sw_batch`` scenarios and
the research group's 4 tile-drained scenarios is recomputed from its dumped entry state with
:func:`~agrijax.processes.soil_water.infiltration.green_ampt_event` and compared with RZWQM2's
infiltration, runoff, suction, front conductance, final rate, front, clock and water contents
(:mod:`green_ampt_scenarios`, which also documents the data and how it is produced).

Pinned per scenario (:data:`EXPECTED`): the event counts (all events; snowmelt, irrigation and
empty events; rain storms and those with runoff; the checked storms, their runoff storms and
those on tilled soil; the storms excluded because an unported Green-Ampt branch acts on them, and
their runoff storms), and, per unported branch, the storms it is switched on for and acts on
(:data:`BRANCHES`). A change in the reference or in the
branch detection shows up as a count change.

Checked storms (no unported branch acts): every difference below :data:`green_ampt_scenarios.TOL_CM`
= 1e-12 (float64 rounding level, as ``test_infiltration_dumps.py``), and the front position
identical. The excluded storms are not checked here; their differences are reported by the
script (``validation/h1_5_green_ampt/summary.json``).

The cases are generated on rorqual by the private collection script into
``<data>/dumps/h1_5_green_ampt/<site>/`` (a few hundred MB, never on the laptop); the test skips a
scenario whose cases are absent.
"""

from __future__ import annotations

import functools
from pathlib import Path

import green_ampt_scenarios as g
import jax
import numpy as np
import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="rounding-level float64 comparison"),
]

#: site -> counts, in the order of :data:`_COUNTS` (measured on rorqual 2026-09-25)
EXPECTED: dict[str, tuple[int, ...]] = {
    "CA-ER1": (1699, 470, 0, 0, 1229, 242, 1229, 242, 1227, 0, 0),
    "CA-MA1": (354, 78, 0, 0, 276, 53, 276, 53, 209, 0, 0),
    "CA-TPA": (1024, 195, 0, 0, 829, 2, 829, 2, 821, 0, 0),
    "US-LYS_NW": (3665, 179, 1639, 0, 1847, 17, 1847, 17, 1802, 0, 0),
    "US-LYS_SE": (3573, 179, 1547, 0, 1847, 172, 1847, 172, 1802, 0, 0),
    "US-LYS_SW": (3541, 179, 1515, 0, 1847, 48, 1847, 48, 1802, 0, 0),
    "US-Mj1": (2285, 760, 8, 0, 1517, 21, 1517, 21, 1469, 0, 0),
    "US-S2": (1589, 221, 986, 0, 382, 0, 382, 0, 0, 0, 0),
    "US-TW3": (1705, 0, 1235, 0, 470, 41, 470, 41, 0, 0, 0),
    "US-Tw2": (1064, 0, 126, 0, 938, 19, 938, 19, 884, 0, 0),
    "US-UA1_HartFarm": (3700, 0, 3160, 0, 540, 3, 540, 3, 509, 0, 0),
    "US-manilacotton": (1085, 56, 0, 0, 1029, 192, 1029, 192, 1006, 0, 0),
    "US_OPE": (1147, 89, 0, 0, 1058, 69, 1058, 69, 1042, 0, 0),
    "US_Rockfish": (989, 10, 0, 0, 979, 12, 979, 12, 0, 0, 0),
    "US_Rockford_Alfalfa": (1688, 95, 1011, 371, 211, 18, 211, 18, 0, 0, 0),
    "Ohio-td_1": (1870, 95, 9, 0, 1766, 155, 0, 0, 0, 1766, 155),
    "Ohio-td_2": (1870, 95, 9, 0, 1766, 155, 0, 0, 0, 1766, 155),
    "Lanna-td": (2135, 440, 0, 0, 1695, 86, 0, 0, 0, 1695, 86),
    "Melby-td": (2833, 350, 0, 0, 2483, 5, 0, 0, 0, 2483, 5),
}
#: site -> {unported branch: (storms it is switched on for, storms it acts on)}
BRANCHES: dict[str, dict[str, tuple[int, int]]] = {
    "CA-ER1": {},
    "CA-MA1": {},
    "CA-TPA": {},
    "US-LYS_NW": {"water_table": (1847, 0)},
    "US-LYS_SE": {"water_table": (1847, 0)},
    "US-LYS_SW": {"water_table": (1847, 0)},
    "US-Mj1": {},
    "US-S2": {},
    "US-TW3": {"water_table": (470, 0)},
    "US-Tw2": {"water_table": (938, 0)},
    "US-UA1_HartFarm": {"water_table": (540, 0)},
    "US-manilacotton": {},
    "US_OPE": {},
    "US_Rockfish": {},
    "US_Rockford_Alfalfa": {},
    "Ohio-td_1": {
        "crust": (1754, 1754),
        "macropore": (1766, 1766),
        "water_table": (1766, 1766),
        "bottom_flux": (1766, 0),
    },
    "Ohio-td_2": {
        "crust": (1754, 1754),
        "macropore": (1766, 1766),
        "water_table": (1766, 1766),
        "bottom_flux": (1766, 0),
    },
    "Lanna-td": {"macropore": (1695, 1695), "water_table": (1695, 1694), "bottom_flux": (1679, 0)},
    "Melby-td": {"macropore": (2483, 2483), "water_table": (2483, 2449), "bottom_flux": (2316, 0)},
}

#: one pass over a scenario's cases serves both tests
_site_rows = functools.cache(g.site_rows)

_COUNTS = (
    "events",
    "snowmelt",
    "irrigation",
    "empty",
    "rain",
    "rain_runoff",
    "checked",
    "checked_runoff",
    "checked_tilled",
    "excluded",
    "excluded_runoff",
)


@pytest.mark.allow_skip(
    reason="the H1-5 EVNTRO/INFIL cases exist only on rorqual (private collection script)"
)
@pytest.mark.parametrize("site", g.SITES)
def test_scenario_storms(data_dir: Path, site: str) -> None:
    """Every checked storm of one scenario equal to RZWQM2 to rounding; counts and branches pinned."""
    if not (data_dir / g.CASES / site / "EVNTRO").is_dir():
        pytest.skip(f"no H1-5 cases for {site} under {data_dir / g.CASES}")
    rows = _site_rows(data_dir, site)
    s = g.summary(rows)
    assert tuple(s[k] for k in _COUNTS) == EXPECTED[site]
    assert {k: (v["switched_on"], v["acts"]) for k, v in s["branches"].items()} == BRANCHES[site]
    assert s["rain_errors"] == 0, s["error_kinds"]
    checked = [r for r in rows if r.portable]
    for k in g.DIFFS:
        v = np.abs(np.array([getattr(r, k) for r in checked], float))
        if k == "d_vfin":  # NaN only where the profile saturated (no front step)
            v = np.where(np.isfinite(v), v, 0.0)
        else:  # a NaN anywhere else is a kernel failure, not a pass
            assert np.isfinite(v).all(), (k, [r.case for r, x in zip(checked, v) if not np.isfinite(x)][:5])
        if v.size == 0:
            continue
        i = int(np.argmax(v))
        bound = 0.0 if k == "d_front" else g.TOL_CM
        assert v[i] <= bound if k == "d_front" else v[i] < bound, (k, float(v[i]), checked[i].case)


#: unported branches that change the capacity itself (the others move water after or beside it)
_CAPACITY_BRANCHES = {"crust", "ice", "plastic"}


@pytest.mark.allow_skip(
    reason="the H1-5 EVNTRO/INFIL cases exist only on rorqual (private collection script)"
)
@pytest.mark.parametrize("site", g.TILE)
def test_capacity_on_tile_scenarios(data_dir: Path, site: str) -> None:
    """The capacity on the excluded tile-drain storms: suction, conductance and final rate at RZWQM2's front.

    Macropore and water-table flow change the event totals but not the capacity formula: the
    suction and front conductance per node (``INFIL`` entry) and the rate of the last front step
    at RZWQM2's own front slice agree to rounding. Storms on which a branch acts that changes the
    capacity itself (:data:`_CAPACITY_BRANCHES`) are left out.
    """
    if not (data_dir / g.CASES / site / "EVNTRO").is_dir():
        pytest.skip(f"no H1-5 cases for {site} under {data_dir / g.CASES}")
    rows = [
        r
        for r in _site_rows(data_dir, site)
        if r.kind == "rain" and not r.error and not _CAPACITY_BRANCHES & set(r.branches.split(";"))
    ]
    assert rows, site
    for k in ("d_swf", "d_cnn", "d_vfin"):
        v = np.abs(np.array([getattr(r, k) for r in rows], float))
        if k == "d_vfin":
            v = np.where(np.isfinite(v), v, 0.0)
        else:
            assert np.isfinite(v).all(), (k, site)
        i = int(np.argmax(v))
        assert v[i] < g.TOL_CM, (k, float(v[i]), rows[i].case)
