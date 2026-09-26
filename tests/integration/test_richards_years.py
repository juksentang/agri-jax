"""H1 item 4: M1 year by year, the Richards replay of every year of CA-TPA 2015-2023 and of the other local scenarios.

The replay is the M1 recipe of ``test_richards_catpa.py`` (RZWQM2's own infiltration, evaporation and
uptake prescribed, drainage and profile computed; :mod:`richards_years`), applied to every calendar
year of a reference run. ``restart``: each year starts from RZWQM2's ``LAYER.PLT`` profile of the
previous 31 December (the first year from ``rzinit.dat``), so every year is its own one-year M1 check;
2015 is exactly the M1 run (0.0257 / 0.0483 / 0.223 cm). ``free``: no reset over the nine years.

Measured (``python tests/integration/richards_years.py`` on rorqual; tables in
``<data>/validation/h1_4_richards_m1/final``):

* CA-TPA, 96 x 8: all nine years meet M1 (storage RMSE 0.0152-0.0480 cm, theta RMSE <= 0.0014);
  free-running too (max 0.0482 cm, 2018). The margin is thin: 2018 uses 96 % of the bound.
* CA-TPA, 24 x 3: **no safe margin**: 2016 (0.0819), 2017 (0.0588), 2018 (0.0579) and 2021 (0.0510)
  exceed 0.05 cm. 2016 is one unconverged day: the 2.56 cm storm of 2016-09-10 onto the profile dried
  by the 2016 drought leaves a -0.267 cm balance error with 3 Newton iterations (96 x 8: 2e-9 cm);
  in 2017, 2018 and 2021 the 96 x 8 residual already takes 0.034-0.048 cm and 24 x 3 adds 0.014-0.021 cm
  (RMSE against 96 x 8). 12 x 2 fails in every year but 2022.
* The CA-TPA residual is not tillage and not the reference build: RZWQM2's own post-tillage
  ``SOILHP`` (two-segment curve, ``PHYSCL`` dumps of the instrumented run) changes the per-year RMSE by
  0 to +0.0048 cm at 96 x 8 (never better), the ``MATILL`` profile mixing by <= 0.001 cm; the instrumented 4.5 build and
  the shipped 4.6 binary differ by <= 0.0018 cm RMSE per year (a measured reference-side spread).
* Other scenarios (``RZWQM_sw_batch``): 8 of the 14 with a ``rzwqm.dat`` have the replay's physics
  (free drainage ``IREBOT = 2``, no perched water table, no tile drains, no macropores); the three
  lysimeters US-LYS_NW/SE/SW (``IREBOT = 3``, water table) and US-TW3, US-Tw2, US-UA1_HartFarm
  (``IREBOT = 3``, water table, tile drains) do not, CFIA_Ottawa has no scenario. Of the 82
  site-years of the 8, M1 holds in 18 at 96 x 8 (15 at 24 x 3); the year's storage error budget (``budget_*`` columns)
  names the cause: replayed infiltration the Richards surface cannot take on the storm hours
  (``runoff``: CA-ER1, US-manilacotton, US_OPE; our Green-Ampt event day brings CA-ER1 2020 from
  0.747 to 0.120 cm), RZWQM2 extracting uptake from nodes held at Hmin (``uptake_cut``: US-S2
  2017-2022, US_Rockford_Alfalfa 2008-2012, 1e4 node-days at -15000 cm), evaporation the surface
  cannot supply (``evaporation_deficit``: US_Rockfish 1.3-6.1 cm a year), unconverged steps on
  saturated surfaces (``balance``), and a slow drainage lag in the wet years of US-Mj1 (2008, 2010,
  2011, 2018, 2023: 0.26-1.08 cm less drainage by year end, not tillage either: the post-tillage
  curve reproduces RZWQM2's printed heads there but leaves the RMSE unchanged). US-Mj1 meets M1
  in 14 of its 24 years.

Why the other scenarios fail (``richards_years.run_conventions``; tables ``final/conventions.csv``):
two RZWQM2 conventions that the replay (and ``richards.py``) do not have, and RZWQM2's time scheme.

* ``DRAIN`` (Rzday.for:3975, after every ``RICHRD`` step in ``REDIST``): no node holds more than
  ``PORI = AEF * theta_s``; the excess moves down at once and leaves the bottom as seepage. The reference
  never exceeds it (largest theta / PORI 1.0000 on US-Mj1, US-S2, US-manilacotton, US_OPE), the M1
  replay does on 1229 (US-Mj1), 1065 (US_OPE), 333, 195, 120, 14 and 9 node-days. This is the US-Mj1
  wet-year drainage lag and the replay runoff of CA-ER1 / US_OPE (with the cap the surface node is
  emptied to PORI each sub-step and takes the supply).
* flux-mode surface limit (``CHKBC`` / ``CNHEAD``): RZWQM2 keeps the evaporation a flux boundary while
  the ghost head found stays above Hmin. With the geometric-mean face K and ``eps > 2`` the Darcy flux
  to the ghost node peaks near ``eps / (eps - 2)`` times the node head and is smaller at Hmin, which is
  where ``richards.surface_fluxes`` caps it. This is the US_Rockfish evaporation deficit (1.3-6.1 cm a
  year at 96 x 8, <= 0.008 cm with the peak limit).
* time scheme: RZWQM2 steps <= 0.1 h with alpha 1 on the first step of a day and 1/2 after; our 96
  implicit sub-steps drain a saturated profile too slowly (US-Mj1 2008: 0.154 cm with both
  conventions at 96 x 8, 0.035 cm at 240 x 10 alpha 1/2).

With both conventions M1 holds in 39 of the 82 site-years at 96 x 8 and 54 at 240 x 10 with RZWQM2's
time scheme (US_Rockfish and US_OPE in every year). The coarse time step of CA-TPA's residual is part
of it too (2017: 0.0400 -> 0.0329 cm); what is left there (0.012-0.045 cm, top 60 cm wetter by year end)
is not explained. Still failing at 240 x 10: US-S2 2017-2022 and US_Rockford_Alfalfa 2008-2012
(``uptake_cut``: the reference theta(h) matches the rzwqm.dat curve to 1e-6 there, so no tillage; the cut
sits in the low-K horizons, US-S2 node 27.5 cm and US_Rockford_Alfalfa 5.5-13.5 cm, where on the cut
days RZWQM2's own profile implies 4.6 and 9.3 cm of inflow that our replay does not deliver: open, needs
``RICHRD`` dumps of those runs); 0.05-0.1 cm evaporation deficits on CA-ER1 (2012, 2016, 2019), CA-MA1
(2008, 2009, 2011), US-Mj1 (2001-2003, 2012) and US-manilacotton (2014, 2015), sites whose top-horizon
reference theta(h) departs from the rzwqm.dat curve by 5e-3 to 2e-2 (tillage-modified curves the replay
does not use; attribution not measured), and on US-S2 2023 and US_Rockford_Alfalfa 2007 (no tillage:
open); drainage on CA-ER1 2014 and 2017 and US-Mj1 2011 (0.055-0.097 cm; open).

The pinned values are the measured ones to 6 significant digits; the replay on the same inputs is
deterministic, so the tolerance is 1e-5 cm (the reference series are compared through their sums
first: a changed reference run fails there, not in the pins).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from richards_years import (
    BATCH,
    CATPA_BASE,
    CATPA_DUMPS,
    M1_STORAGE_RMSE,
    M1_THETA_RMSE,
    RunReplay,
    catpa_physcl,
    catpa_replay,
    comparability,
    internal_node_horizon,
    pori_of,
    run_conventions,
    run_scenario,
    soil_days_from_soilhp,
)

pytestmark = [
    pytest.mark.allow_skip(
        reason="needs the private CA-TPA 2015-2023 reference run and scenarios under the data dir"
    ),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="M1 is a float64 comparison"),
]

#: pinned replay values [cm]: the replay is deterministic on fixed inputs (6-digit pins)
PIN_ABS = 1e-5
#: reference-series sums, pinned to 6 significant digits (half a unit in the 6th digit: 5e-6 relative)
FINGERPRINT_REL = 5e-6
#: the instrumented 4.5 build against the shipped 4.6 binary, CA-TPA, per-year storage RMSE (measured 0.00176)
BUILD_SPREAD_CM = 0.002
#: largest change of a per-year RMSE by RZWQM2's post-tillage curve at 96 x 8 on CA-TPA (measured 0.0048)
TILLAGE_EFFECT_CM = 0.006

#: site-years of the other scenarios meeting M1 at 96 x 8 (of 82; 15 at 24 x 3)
N_PASS_96_OTHER = 18

COMPARABLE = {
    "CA-ER1": True,
    "CA-MA1": True,
    "CA-TPA": True,
    "US-LYS_NW": False,
    "US-LYS_SE": False,
    "US-LYS_SW": False,
    "US-Mj1": True,
    "US-S2": True,
    "US-TW3": False,
    "US-Tw2": False,
    "US-UA1_HartFarm": False,
    "US-manilacotton": True,
    "US_OPE": True,
    "US_Rockfish": True,
    "US_Rockford_Alfalfa": True,
}

#: CA-TPA 2015-2023, restart per year: year -> {(n_sub, n_iter): (storage RMSE [cm], theta RMSE)}
CATPA_RESTART = {
    2015: {
        (96, 8): (0.025716, 0.00106944),
        (24, 3): (0.0483071, 0.00122179),
        (12, 2): (0.223273, 0.00226542),
    },
    2016: {
        (96, 8): (0.0384752, 0.00109369),
        (24, 3): (0.0818849, 0.0020401),
        (12, 2): (0.477036, 0.00788944),
    },
    2017: {
        (96, 8): (0.040009, 0.000844203),
        (24, 3): (0.0588131, 0.00112036),
        (12, 2): (0.0886616, 0.00142332),
    },
    2018: {(96, 8): (0.047971, 0.00135554), (24, 3): (0.057925, 0.00149042), (12, 2): (0.0739608, 0.0017209)},
    2019: {
        (96, 8): (0.0343044, 0.00118508),
        (24, 3): (0.0457304, 0.00127716),
        (12, 2): (0.0758697, 0.00144176),
    },
    2020: {
        (96, 8): (0.0257737, 0.000774853),
        (24, 3): (0.0376986, 0.000957795),
        (12, 2): (1.33003, 0.0191779),
    },
    2021: {
        (96, 8): (0.0342963, 0.00101074),
        (24, 3): (0.0509889, 0.00118888),
        (12, 2): (0.0651566, 0.00152948),
    },
    2022: {
        (96, 8): (0.0152177, 0.000391139),
        (24, 3): (0.023476, 0.00055292),
        (12, 2): (0.0397283, 0.000788447),
    },
    2023: {
        (96, 8): (0.0329951, 0.00102479),
        (24, 3): (0.0474499, 0.00118196),
        (12, 2): (0.0684599, 0.00143612),
    },
}
#: CA-TPA free-running from rzinit.dat (no reset), 96 x 8: year -> storage RMSE [cm]
CATPA_FREE_96 = {
    2015: 0.025716,
    2016: 0.0389614,
    2017: 0.0415161,
    2018: 0.0482489,
    2019: 0.0347395,
    2020: 0.0263522,
    2021: 0.0353925,
    2022: 0.0164856,
    2023: 0.0333405,
}
#: other comparable scenarios, restart per year: (site, year) -> (96x8 RMSE, 24x3 RMSE [cm], largest 96x8 budget term)
SCENARIO_RESTART = {
    ("CA-ER1", 2011): (0.082475, 0.121892, "evaporation_deficit"),
    ("CA-ER1", 2012): (0.223329, 0.0871616, "runoff"),
    ("CA-ER1", 2013): (0.0455789, 0.078878, "runoff"),
    ("CA-ER1", 2014): (0.103713, 0.120145, "runoff"),
    ("CA-ER1", 2015): (0.0409806, 0.12176, "runoff"),
    ("CA-ER1", 2016): (0.0788884, 0.409629, "runoff"),
    ("CA-ER1", 2017): (0.204977, 0.0856386, "drainage"),
    ("CA-ER1", 2018): (0.0691449, 0.11605, "runoff"),
    ("CA-ER1", 2019): (0.182994, 0.128294, "drainage"),
    ("CA-ER1", 2020): (0.747233, 1.01664, "runoff"),
    ("CA-ER1", 2021): (0.113131, 0.0812508, "runoff"),
    ("CA-ER1", 2022): (0.085051, 0.0982581, "evaporation_deficit"),
    ("CA-ER1", 2023): (0.220103, 0.231043, "runoff"),
    ("CA-MA1", 2008): (0.076978, 0.0252752, "evaporation_deficit"),
    ("CA-MA1", 2009): (0.0760726, 0.346363, "runoff"),
    ("CA-MA1", 2010): (0.0484931, 0.0550487, "evaporation_deficit"),
    ("CA-MA1", 2011): (0.0841878, 0.0819703, "evaporation_deficit"),
    ("US-Mj1", 2000): (0.0491231, 0.0470944, "evaporation_deficit"),
    ("US-Mj1", 2001): (0.0491654, 0.0460917, "evaporation_deficit"),
    ("US-Mj1", 2002): (0.0971198, 0.0902315, "evaporation_deficit"),
    ("US-Mj1", 2003): (0.0543882, 0.0508427, "evaporation_deficit"),
    ("US-Mj1", 2004): (0.0312982, 0.0296429, "evaporation_deficit"),
    ("US-Mj1", 2005): (0.0170577, 0.0159656, "evaporation_deficit"),
    ("US-Mj1", 2006): (0.00990766, 0.00949195, "evaporation_deficit"),
    ("US-Mj1", 2007): (0.0132676, 0.01677, "drainage"),
    ("US-Mj1", 2008): (0.765282, 0.771932, "drainage"),
    ("US-Mj1", 2009): (0.00323586, 0.00303886, "drainage"),
    ("US-Mj1", 2010): (0.615182, 0.621333, "drainage"),
    ("US-Mj1", 2011): (1.12188, 1.13327, "drainage"),
    ("US-Mj1", 2012): (0.0950802, 0.0899868, "evaporation_deficit"),
    ("US-Mj1", 2013): (0.0379324, 0.0392353, "drainage"),
    ("US-Mj1", 2014): (0.0415846, 0.0404691, "drainage"),
    ("US-Mj1", 2015): (0.0231961, 0.02304, "drainage"),
    ("US-Mj1", 2016): (0.0201801, 0.0184946, "evaporation_deficit"),
    ("US-Mj1", 2017): (0.00826055, 0.00832673, "evaporation_deficit"),
    ("US-Mj1", 2018): (0.637482, 0.645441, "drainage"),
    ("US-Mj1", 2019): (0.07086, 0.0737249, "drainage"),
    ("US-Mj1", 2020): (0.0992914, 0.103406, "drainage"),
    ("US-Mj1", 2021): (0.00847259, 0.00871541, "evaporation_deficit"),
    ("US-Mj1", 2022): (0.0301927, 0.027497, "evaporation_deficit"),
    ("US-Mj1", 2023): (0.596808, 0.602068, "drainage"),
    ("US-S2", 2016): (0.0599246, 0.0570233, "evaporation_deficit"),
    ("US-S2", 2017): (0.357554, 0.358269, "uptake_cut"),
    ("US-S2", 2018): (0.326699, 0.328344, "uptake_cut"),
    ("US-S2", 2019): (0.302187, 0.295898, "uptake_cut"),
    ("US-S2", 2020): (0.349289, 0.338778, "uptake_cut"),
    ("US-S2", 2021): (0.556535, 0.553173, "uptake_cut"),
    ("US-S2", 2022): (0.0907577, 0.091433, "uptake_cut"),
    ("US-S2", 2023): (0.142237, 0.116913, "evaporation_deficit"),
    ("US-manilacotton", 2014): (0.0818791, 0.125942, "balance"),
    ("US-manilacotton", 2015): (0.155643, 0.160078, "drainage"),
    ("US-manilacotton", 2016): (0.0831708, 0.0786495, "runoff"),
    ("US-manilacotton", 2017): (0.056506, 0.0774821, "evaporation_deficit"),
    ("US-manilacotton", 2018): (0.122769, 0.168056, "balance"),
    ("US-manilacotton", 2019): (0.110678, 0.128774, "balance"),
    ("US-manilacotton", 2020): (0.0712406, 0.0944168, "evaporation_deficit"),
    ("US-manilacotton", 2021): (0.0588156, 0.0772785, "evaporation_deficit"),
    ("US-manilacotton", 2022): (0.0686956, 0.0825027, "evaporation_deficit"),
    ("US-manilacotton", 2023): (0.0807651, 0.103802, "evaporation_deficit"),
    ("US_OPE", 2015): (0.132562, 1.54548, "balance"),
    ("US_OPE", 2016): (0.0736886, 0.0806874, "evaporation_deficit"),
    ("US_OPE", 2017): (0.0677098, 0.137252, "balance"),
    ("US_OPE", 2018): (1.73174, 0.571211, "drainage"),
    ("US_OPE", 2019): (0.0878046, 0.134015, "evaporation_deficit"),
    ("US_OPE", 2020): (0.115197, 0.137172, "drainage"),
    ("US_OPE", 2021): (0.0696613, 0.0813228, "evaporation_deficit"),
    ("US_OPE", 2022): (0.0470568, 0.0833708, "drainage"),
    ("US_OPE", 2023): (0.102094, 0.123474, "drainage"),
    ("US_Rockfish", 2016): (0.708563, 0.461826, "drainage"),
    ("US_Rockfish", 2017): (0.309007, 0.464538, "evaporation_deficit"),
    ("US_Rockfish", 2018): (0.222069, 0.257243, "evaporation_deficit"),
    ("US_Rockfish", 2019): (0.267386, 1.82875, "evaporation_deficit"),
    ("US_Rockfish", 2020): (0.169937, 1.07331, "evaporation_deficit"),
    ("US_Rockfish", 2021): (0.299563, 3.05645, "evaporation_deficit"),
    ("US_Rockfish", 2022): (0.286727, 0.305488, "evaporation_deficit"),
    ("US_Rockfish", 2023): (0.384059, 0.409297, "evaporation_deficit"),
    ("US_Rockford_Alfalfa", 2007): (0.290247, 0.287138, "evaporation_deficit"),
    ("US_Rockford_Alfalfa", 2008): (0.207285, 0.258836, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2009): (0.721853, 0.716214, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2010): (0.905536, 0.877851, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2011): (0.738148, 0.749904, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2012): (1.10185, 1.05926, "uptake_cut"),
}
#: the replay with RZWQM2's two conventions emulated (richards_years.run_conventions: DRAIN cap after
#: every sub-step and the flux-mode surface limit), restart per year: (site, year) -> (storage RMSE [cm]
#: at 96 x 8, at 240 x 10 with RZWQM2's time scheme (alpha 1 then 1/2), largest budget term at 240 x 10)
CONVENTIONS = {
    ("CA-TPA", 2015): (0.0231183, 0.0173598, "drainage"),
    ("CA-TPA", 2016): (0.0363116, 0.0333504, "evaporation_deficit"),
    ("CA-TPA", 2017): (0.0390654, 0.0328763, "drainage"),
    ("CA-TPA", 2018): (0.0474006, 0.0445416, "drainage"),
    ("CA-TPA", 2019): (0.0337267, 0.0303979, "drainage"),
    ("CA-TPA", 2020): (0.0225418, 0.0191472, "drainage"),
    ("CA-TPA", 2021): (0.0335068, 0.028154, "drainage"),
    ("CA-TPA", 2022): (0.0151358, 0.0121127, "drainage"),
    ("CA-TPA", 2023): (0.031077, 0.0260471, "drainage"),
    ("CA-ER1", 2011): (0.0501574, 0.0482046, "evaporation_deficit"),
    ("CA-ER1", 2012): (0.0844011, 0.0852171, "evaporation_deficit"),
    ("CA-ER1", 2013): (0.0385724, 0.0361752, "evaporation_deficit"),
    ("CA-ER1", 2014): (0.0976506, 0.096706, "drainage"),
    ("CA-ER1", 2015): (0.0259563, 0.0241753, "evaporation_deficit"),
    ("CA-ER1", 2016): (0.0527704, 0.0516209, "evaporation_deficit"),
    ("CA-ER1", 2017): (0.0572215, 0.0552634, "drainage"),
    ("CA-ER1", 2018): (0.0498138, 0.0492343, "evaporation_deficit"),
    ("CA-ER1", 2019): (0.0623928, 0.0617823, "evaporation_deficit"),
    ("CA-ER1", 2020): (0.0410826, 0.0406135, "evaporation_deficit"),
    ("CA-ER1", 2021): (0.0381542, 0.037874, "evaporation_deficit"),
    ("CA-ER1", 2022): (0.049765, 0.0484324, "evaporation_deficit"),
    ("CA-ER1", 2023): (0.0365682, 0.0341106, "evaporation_deficit"),
    ("CA-MA1", 2008): (0.0765936, 0.080813, "evaporation_deficit"),
    ("CA-MA1", 2009): (0.0537699, 0.0529636, "evaporation_deficit"),
    ("CA-MA1", 2010): (0.0164045, 0.014162, "evaporation_deficit"),
    ("CA-MA1", 2011): (0.074684, 0.075964, "evaporation_deficit"),
    ("US-Mj1", 2000): (0.0491231, 0.0499116, "evaporation_deficit"),
    ("US-Mj1", 2001): (0.0491654, 0.0504069, "evaporation_deficit"),
    ("US-Mj1", 2002): (0.0971206, 0.0997655, "evaporation_deficit"),
    ("US-Mj1", 2003): (0.0543882, 0.0558332, "evaporation_deficit"),
    ("US-Mj1", 2004): (0.0312982, 0.0319094, "evaporation_deficit"),
    ("US-Mj1", 2005): (0.0170577, 0.0175037, "evaporation_deficit"),
    ("US-Mj1", 2006): (0.00994868, 0.0101224, "evaporation_deficit"),
    ("US-Mj1", 2007): (0.012177, 0.00955925, "drainage"),
    ("US-Mj1", 2008): (0.153693, 0.0351169, "drainage"),
    ("US-Mj1", 2009): (0.00323534, 0.00337359, "drainage"),
    ("US-Mj1", 2010): (0.126222, 0.0383373, "drainage"),
    ("US-Mj1", 2011): (0.192064, 0.059924, "drainage"),
    ("US-Mj1", 2012): (0.0950802, 0.0969883, "evaporation_deficit"),
    ("US-Mj1", 2013): (0.0193393, 0.0121285, "evaporation_deficit"),
    ("US-Mj1", 2014): (0.0417794, 0.0420214, "drainage"),
    ("US-Mj1", 2015): (0.0237722, 0.0261102, "drainage"),
    ("US-Mj1", 2016): (0.0201473, 0.0209113, "evaporation_deficit"),
    ("US-Mj1", 2017): (0.00826055, 0.00826243, "evaporation_deficit"),
    ("US-Mj1", 2018): (0.157064, 0.0493364, "drainage"),
    ("US-Mj1", 2019): (0.0353268, 0.0158597, "drainage"),
    ("US-Mj1", 2020): (0.0121401, 0.046107, "drainage"),
    ("US-Mj1", 2021): (0.00847259, 0.00841303, "evaporation_deficit"),
    ("US-Mj1", 2022): (0.0301927, 0.0312109, "evaporation_deficit"),
    ("US-Mj1", 2023): (0.144263, 0.0407531, "drainage"),
    ("US-S2", 2016): (0.0221381, 0.0225402, "evaporation_deficit"),
    ("US-S2", 2017): (0.347865, 0.347049, "uptake_cut"),
    ("US-S2", 2018): (0.319311, 0.319253, "uptake_cut"),
    ("US-S2", 2019): (0.30153, 0.301396, "uptake_cut"),
    ("US-S2", 2020): (0.347554, 0.348143, "uptake_cut"),
    ("US-S2", 2021): (0.555207, 0.557151, "uptake_cut"),
    ("US-S2", 2022): (0.0841053, 0.0838796, "uptake_cut"),
    ("US-S2", 2023): (0.0682364, 0.0716397, "evaporation_deficit"),
    ("US-manilacotton", 2014): (0.0621308, 0.0570095, "evaporation_deficit"),
    ("US-manilacotton", 2015): (0.0785695, 0.0695314, "evaporation_deficit"),
    ("US-manilacotton", 2016): (0.0522245, 0.0479575, "evaporation_deficit"),
    ("US-manilacotton", 2017): (0.0467757, 0.0416295, "evaporation_deficit"),
    ("US-manilacotton", 2018): (0.0578882, 0.0480024, "evaporation_deficit"),
    ("US-manilacotton", 2019): (0.0502039, 0.0400832, "evaporation_deficit"),
    ("US-manilacotton", 2020): (0.042138, 0.0351346, "drainage"),
    ("US-manilacotton", 2021): (0.0423742, 0.0362106, "evaporation_deficit"),
    ("US-manilacotton", 2022): (0.0402129, 0.0324422, "evaporation_deficit"),
    ("US-manilacotton", 2023): (0.0481495, 0.0409852, "evaporation_deficit"),
    ("US_OPE", 2015): (0.052973, 0.0413506, "evaporation_deficit"),
    ("US_OPE", 2016): (0.0386342, 0.0329828, "drainage"),
    ("US_OPE", 2017): (0.0265474, 0.018986, "evaporation_deficit"),
    ("US_OPE", 2018): (0.0555558, 0.0376144, "drainage"),
    ("US_OPE", 2019): (0.0410103, 0.0333047, "evaporation_deficit"),
    ("US_OPE", 2020): (0.0410561, 0.0297817, "drainage"),
    ("US_OPE", 2021): (0.036175, 0.0265402, "evaporation_deficit"),
    ("US_OPE", 2022): (0.0349079, 0.0276353, "drainage"),
    ("US_OPE", 2023): (0.0338058, 0.0226796, "drainage"),
    ("US_Rockfish", 2016): (0.0526706, 0.035292, "drainage"),
    ("US_Rockfish", 2017): (0.0441093, 0.0311382, "drainage"),
    ("US_Rockfish", 2018): (0.0609434, 0.0419378, "drainage"),
    ("US_Rockfish", 2019): (0.102842, 0.0296198, "uptake_cut"),
    ("US_Rockfish", 2020): (0.0644762, 0.0433703, "drainage"),
    ("US_Rockfish", 2021): (0.0522427, 0.0356588, "drainage"),
    ("US_Rockfish", 2022): (0.0463533, 0.0318408, "drainage"),
    ("US_Rockfish", 2023): (0.058183, 0.039079, "drainage"),
    ("US_Rockford_Alfalfa", 2007): (0.0906333, 0.0940814, "evaporation_deficit"),
    ("US_Rockford_Alfalfa", 2008): (0.438823, 0.43806, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2009): (0.716034, 0.719051, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2010): (0.960835, 0.96618, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2011): (0.883215, 0.884549, "uptake_cut"),
    ("US_Rockford_Alfalfa", 2012): (1.1004, 1.10203, "uptake_cut"),
}
#: site -> (largest reference theta / PORI, node-days of the M1 replay above PORI)
PORI_STATS = {
    "CA-TPA": (0.93352, 0),
    "CA-ER1": (0.967935, 9),
    "CA-MA1": (1.00047, 120),
    "US-Mj1": (1.0, 1229),
    "US-S2": (1.0, 14),
    "US-manilacotton": (1.0, 333),
    "US_OPE": (1.0, 1065),
    "US_Rockfish": (0.998403, 195),
    "US_Rockford_Alfalfa": (0.785269, 0),
}
#: site-years of the other scenarios meeting M1 with both conventions (of 82): 96 x 8 and 240 x 10 CN
N_PASS_CONVENTIONS = (39, 54)
#: the other scenarios' site-years still failing M1 with both conventions at 240 x 10 CN, by the
#: largest budget term: RZWQM2 feeding uptake in low-K horizons (open), evaporation on tilled surfaces
FAIL_UPTAKE_CUT = {("US-S2", y) for y in range(2017, 2023)} | {
    ("US_Rockford_Alfalfa", y) for y in range(2008, 2013)
}

#: sums of the reference series each replay reads (RunReplay.fingerprint): storage, infiltration,
#: evaporation, uptake, deep seepage [cm] and theta
FINGERPRINTS = {
    "CA-ER1": (217492.0, 1029.78, 304.423, 314.693, 403.056, 53671.8),
    "CA-MA1": (69815.9, 190.389, 93.5882, 65.4474, 37.3801, 17330.6),
    "CA-TPA": (123790.0, 834.94, 134.016, 226.923, 462.266, 30965.5),
    "US-Mj1": (413562.0, 1049.13, 709.145, 201.606, 135.283, 105790.0),
    "US-S2": (111194.0, 702.371, 196.476, 512.065, 8.18604, 29628.2),
    "US-manilacotton": (136819.0, 1195.65, 358.448, 347.68, 484.819, 35226.4),
    "US_OPE": (168030.0, 1048.03, 162.421, 328.377, 545.917, 40206.9),
    "US_Rockfish": (114200.0, 1117.94, 184.759, 227.65, 696.428, 24783.3),
    "US_Rockford_Alfalfa": (68033.3, 793.416, 193.875, 604.589, 2.04113e-05, 16551.7),
}


def _fingerprint_ok(rep: RunReplay, site: str) -> None:
    fp = rep.fingerprint()
    keys = ("storage", "infiltration", "evaporation", "uptake", "deep_seepage")
    got = (*(fp[f"sum_{k}_cm"] for k in keys), fp["sum_theta"])
    np.testing.assert_allclose(
        got,
        FINGERPRINTS[site],
        rtol=FINGERPRINT_REL,
        err_msg=f"{site}: the reference run changed; re-measure",
    )


def _rmse_by_year(rep: RunReplay, r: dict[str, np.ndarray], key: str = "storage") -> dict[int, float]:
    ref = rep.storage if key == "storage" else rep.theta
    return {
        int(y): float(np.sqrt(np.mean((r[key][rep.years == y] - ref[rep.years == y]) ** 2)))
        for y in np.unique(rep.years)
    }


@pytest.fixture(scope="module")
def catpa(data_dir: Path) -> RunReplay:
    for p in (data_dir / CATPA_BASE / "CA-TPA.ana", data_dir / CATPA_BASE / "LAYER.PLT"):
        if not p.is_file():
            pytest.skip(f"{p} not found")
    return catpa_replay(data_dir)


def test_comparability_of_the_local_scenarios(data_dir: Path) -> None:
    """Which scenarios have the replay's physics (free drainage, no water table / drains / macropores)."""
    got = {}
    for site in COMPARABLE:
        dat = data_dir / BATCH / site / "Scenario" / "rzwqm.dat"
        if not dat.is_file():
            pytest.skip(f"{dat} not found")
        got[site] = comparability(dat)["comparable"]
    assert got == COMPARABLE


def test_catpa_run_inputs(catpa: RunReplay, data_dir: Path) -> None:
    """The 2015-2023 run closes RZWQM2's own balance, its 2015 is the M1 run, and nothing but E/T/seepage leaves."""
    _fingerprint_ok(catpa, "CA-TPA")
    c = catpa.input_checks()
    assert c["grid_vs_layer_plt_maxabs_cm"] == 0.0
    assert abs(c["init_storage_vs_ana_cm"]) < 1e-9
    assert c["uptake_vs_col7_maxabs_cm"] < 2e-6
    assert c["supply_vs_col5_maxabs_cm"] < 1e-12
    assert c["balance_residual_maxabs_cm"] < 1e-4  # .ana print precision
    for k in (
        "tile_drainage_sum_cm",
        "lateral_water_flow_sum_cm",
        "water_added_due_to_using_measured_swc_sum_cm",
    ):
        assert c[k] == 0.0, k
    assert len(catpa.days) == 3287 and sorted(set(catpa.years)) == list(range(2015, 2024))
    ref = RunReplay("CA-TPA", data_dir / "catpa/ref_2015", data_dir / BATCH / "CA-TPA" / "Scenario")
    k = catpa.years == 2015
    np.testing.assert_allclose(catpa.storage[k], ref.storage, rtol=0, atol=1e-4)
    np.testing.assert_allclose(catpa.theta[k], ref.theta, rtol=0, atol=1e-6)
    np.testing.assert_allclose(catpa.supply[k], ref.supply, rtol=0, atol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize(("n_sub", "n_iter"), [(96, 8), (24, 3), (12, 2)])
def test_catpa_m1_every_year(catpa: RunReplay, n_sub: int, n_iter: int) -> None:
    """Every year of CA-TPA 2015-2023 (restart): pinned per-year RMSE; 96 x 8 meets M1, 24 x 3 does not everywhere."""
    r = catpa.run(mode="restart", n_sub=n_sub, n_iter=n_iter)
    storage, theta = _rmse_by_year(catpa, r), _rmse_by_year(catpa, r, "theta")
    for y, pins in CATPA_RESTART.items():
        s_pin, t_pin = pins[(n_sub, n_iter)]
        assert storage[y] == pytest.approx(s_pin, abs=PIN_ABS), y
        assert theta[y] == pytest.approx(t_pin, abs=PIN_ABS), y
    fails = {y for y, v in storage.items() if v >= M1_STORAGE_RMSE}
    if (n_sub, n_iter) == (96, 8):
        assert not fails
        assert max(theta.values()) < M1_THETA_RMSE
        assert max(storage, key=lambda y: storage[y]) == 2018
        # measured: 6.3e-6 cm on one day of 2020 (the year sum is 6.3e-6 cm too), <= 2.3e-9 cm a year otherwise
        assert np.max(np.abs(r["balance_error"])) < 1e-5
        assert r["n_clamp"].sum() == 0.0 and r["runoff"].sum() == 0.0
    elif (n_sub, n_iter) == (24, 3):
        assert fails == {2016, 2017, 2018, 2021}
        # 2016: one unconverged storm day carries the year
        k = catpa.years == 2016
        day = int(np.argmin(r["balance_error"][k]))
        assert str(catpa.days[k][day]) == "2016-09-10"
        assert r["balance_error"][k][day] < -0.26
    else:
        assert fails == set(CATPA_RESTART) - {2022}


@pytest.mark.slow
def test_catpa_free_running_96x8(catpa: RunReplay) -> None:
    """The nine years without any reset: every calendar year still meets M1 at 96 x 8."""
    r = catpa.run(mode="free", n_sub=96, n_iter=8)
    storage = _rmse_by_year(catpa, r)
    for y, pin in CATPA_FREE_96.items():
        assert storage[y] == pytest.approx(pin, abs=PIN_ABS), y
    assert max(storage.values()) < M1_STORAGE_RMSE


@pytest.mark.slow
def test_catpa_residual_is_not_tillage_nor_the_build(data_dir: Path, catpa: RunReplay) -> None:
    """RZWQM2's post-tillage curve does not remove the residual; the 4.5 build is within 0.002 cm of 4.6."""
    run45 = data_dir / "dumps/_runs/catpa2015_2023"
    if not (data_dir / CATPA_DUMPS / "physcl_entry.npz").is_file() or not (run45 / "LAYER.PLT").is_file():
        pytest.skip("instrumented CA-TPA 2015-2023 run (PHYSCL dumps) not found")
    rep = RunReplay("CA-TPA", run45, data_dir / BATCH / "CA-TPA" / "Scenario")
    spread = {
        int(y): float(np.sqrt(np.mean((rep.storage[rep.years == y] - catpa.storage[catpa.years == y]) ** 2)))
        for y in np.unique(rep.years)
    }
    assert max(spread.values()) < BUILD_SPREAD_CM
    ph = catpa_physcl(data_dir)
    assert (ph["date"] == rep.days).all()
    nhor, node_horizon = internal_node_horizon(ph, np.cumsum(np.asarray(rep.grid.tl)))
    assert nhor == 6  # the tillage split of horizon 1 (11 cm) added at start-up
    tilled = soil_days_from_soilhp(ph["SOILHP"], nhor, node_horizon, tilled=True)
    base = _rmse_by_year(rep, rep.run(mode="restart", n_sub=96, n_iter=8))
    with_curve = _rmse_by_year(rep, rep.run(mode="restart", n_sub=96, n_iter=8, soil_days=tilled))
    delta = {y: with_curve[y] - base[y] for y in base}
    assert max(abs(d) for d in delta.values()) < TILLAGE_EFFECT_CM
    assert sum(delta.values()) > 0.0  # on balance the post-tillage curve makes the replay worse, not better


@pytest.fixture(scope="module")
def scenario_runs(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, RunReplay]:
    """RZWQM2 runs of the 8 other comparable scenarios over their whole IPNAMES period (binary on rorqual)."""
    import agrijax.port.run_fortran as rf

    if not rf.RZWQM_BINARY.is_file():
        pytest.skip(f"RZWQM2 binary {rf.RZWQM_BINARY} not found")
    sites = sorted({s for s, _ in SCENARIO_RESTART})
    out = tmp_path_factory.mktemp("h1_4_runs")
    with ThreadPoolExecutor(min(len(sites), os.cpu_count() or 1)) as ex:
        recs = list(ex.map(lambda s: run_scenario(s, data_dir, out, out / "_stage"), sites))
    bad = [(r["site"], r["error"]) for r in recs if not r["ok"]]
    assert not bad, bad
    return {r["site"]: RunReplay(r["site"], Path(r["out"]), Path(r["source"])) for r in recs}


@pytest.mark.slow
def test_other_scenarios_every_year(scenario_runs: dict[str, RunReplay]) -> None:
    """Per-year RMSE (96 x 8, 24 x 3) of the other comparable scenarios and the largest term of each year's budget."""
    rows: dict[tuple[str, int], tuple[float, float, str]] = {}
    for site, rep in scenario_runs.items():
        _fingerprint_ok(rep, site)
        r96 = rep.run(mode="restart", n_sub=96, n_iter=8)
        r24 = rep.run(mode="restart", n_sub=24, n_iter=3)
        t96 = rep.year_table(r96)
        s24 = _rmse_by_year(rep, r24)
        for y, s96, dom in zip(t96["year"], t96["storage_rmse_cm"], t96["budget_dominant"], strict=True):
            rows[(site, int(y))] = (float(s96), s24[int(y)], str(dom))
    assert set(rows) == set(SCENARIO_RESTART)
    for key, (s96, s24, dominant) in SCENARIO_RESTART.items():
        got: Any = rows[key]
        assert got[0] == pytest.approx(s96, abs=PIN_ABS), key
        assert got[1] == pytest.approx(s24, abs=PIN_ABS), key
        assert got[2] == dominant, key
    # the record: M1 holds in 18 of 82 site-years at 96 x 8 outside CA-TPA
    assert sum(v[0] < M1_STORAGE_RMSE for v in rows.values()) == N_PASS_96_OTHER


@pytest.fixture(scope="module")
def all_runs(catpa: RunReplay, scenario_runs: dict[str, RunReplay]) -> dict[str, RunReplay]:
    return {"CA-TPA": catpa, **scenario_runs}


@pytest.mark.slow
def test_conventions_harness_is_the_m1_replay(catpa: RunReplay) -> None:
    """run_conventions with both conventions off is exactly RunReplay.run (measured: 0 cm on every site)."""
    a = catpa.run(mode="restart", n_sub=96, n_iter=8)
    b = run_conventions(catpa, drain=False, flux_mode=False)
    np.testing.assert_array_equal(a["storage"], b["storage"])
    np.testing.assert_array_equal(a["theta"], b["theta"])


@pytest.mark.slow
def test_reference_never_exceeds_field_saturation(all_runs: dict[str, RunReplay]) -> None:
    """RZWQM2 holds at most AEF * theta_s in a node (DRAIN); the M1 replay does not, the DRAIN cap does."""
    for site, (ratio, above) in PORI_STATS.items():
        rep = all_runs[site]
        pori = pori_of(rep)
        # CA-MA1 exceeds by 4.7e-4 (relative) on 2 node-days: its top-horizon curve changes with tillage
        # (reference theta(h) differs from the rzwqm.dat curve by up to 7.7e-3), so PORI there is not the
        # start-up AEF * theta_s used here
        assert float((rep.theta / pori).max()) == pytest.approx(ratio, abs=PIN_ABS), site
        m1 = rep.run(mode="restart", n_sub=96, n_iter=8)
        assert int((m1["theta"] - pori > 1e-6).sum()) == above, site


@pytest.mark.slow
@pytest.mark.parametrize("site", sorted(PORI_STATS))
def test_conventions_explain_the_failing_years(site: str, all_runs: dict[str, RunReplay]) -> None:
    """Per-year RMSE with the DRAIN cap and the flux-mode surface limit, 96 x 8 and 240 x 10 (RZWQM2 time scheme)."""
    rep = all_runs[site]
    pori = pori_of(rep)
    r96 = run_conventions(rep, n_sub=96, n_iter=8)
    rcn = run_conventions(rep, n_sub=240, n_iter=10, time_scheme="rzwqm")
    assert int((r96["theta"] - pori > 1e-9).sum()) == 0
    s96, scn = _rmse_by_year(rep, r96), _rmse_by_year(rep, rcn)
    dom = dict(zip(rep.year_table(rcn)["year"], rep.year_table(rcn)["budget_dominant"], strict=True))
    for (s, y), (p96, pcn, pdom) in CONVENTIONS.items():
        if s != site:
            continue
        assert s96[y] == pytest.approx(p96, abs=PIN_ABS), y
        assert scn[y] == pytest.approx(pcn, abs=PIN_ABS), y
        assert dom[y] == pdom, y
        if (s, y) in FAIL_UPTAKE_CUT:
            assert pcn >= M1_STORAGE_RMSE and pdom == "uptake_cut"


def test_conventions_record() -> None:
    """The counts the pins imply (no data needed): 18 -> 39 (96 x 8) -> 54 (240 x 10 CN) of 82 site-years."""
    other = {k: v for k, v in CONVENTIONS.items() if k[0] != "CA-TPA"}
    assert len(other) == 82
    assert sum(v[0] < M1_STORAGE_RMSE for v in other.values()) == N_PASS_CONVENTIONS[0]
    assert sum(v[1] < M1_STORAGE_RMSE for v in other.values()) == N_PASS_CONVENTIONS[1]
    assert all(
        v[0] < M1_STORAGE_RMSE and v[1] < M1_STORAGE_RMSE for k, v in CONVENTIONS.items() if k[0] == "CA-TPA"
    )
    assert {
        k for k, v in other.items() if v[1] >= M1_STORAGE_RMSE and v[2] == "uptake_cut"
    } == FAIL_UPTAKE_CUT
    assert sum(v[0] < M1_STORAGE_RMSE for v in SCENARIO_RESTART.values()) == N_PASS_96_OTHER
