"""The adaptive Richards mode on the M1 replays, year by year.

``AdaptiveStepping.exact()`` (``dt_max`` 0.1 h, RZWQM2's ``PDTMAX``) and ``"fast"`` (0.25 h)
on the M1 replay of every calendar year (restart mode) of CA-TPA 2015-2023 and of the 8 comparable
scenarios (82 site-years), the years of a site as the lanes of one ``vmap``
(:mod:`richards_adaptive_years`). Compared per site-year with RZWQM2 (M1: storage RMSE < 0.05 cm),
with the converged reference (``validation/w1_e0/ref``: CN with an alpha = 1 fallback, 960 steps a
day, Newton to 1e-10; a *clean* reference year had no unconverged step) and with the fixed 96 x 8
replay (``test_richards_years.SCENARIO_RESTART``).

Measured (rorqual, float64; ``<data>/validation/w1_c/years.csv``; with four changes of
:mod:`~agrijax.processes.soil_water.richards_adaptive`: the conservative dry bound, the clamp test on
the last update, the last-resort Newton cap and the surface-condition switching):

* CA-TPA: unchanged by the four changes, to the pinned digits.
  Every year meets M1 in both tiers (storage RMSE 0.0122-0.0450 cm; 2018 largest), no unconverged
  step, exhausted budget, clamp or held node.
  On the 8 clean reference years the largest daily storage difference to the converged reference
  is 4.7e-5-1.5e-4 cm exact and 2.3e-4-9.1e-4 cm fast (the margin is 1e-3 cm; the reference's own
  960 vs 1920 steps a day spread is <= 1.1e-5 cm there). 2020 has an unconverged reference step
  (960 vs 1920: 0.064 cm); exact and fast agree to 1.1e-4 cm on it. 2016-09-10 (the 2.56 cm storm
  onto the drought profile): exact 261 steps / 929 Newton evaluations, fast 128 / 495, no rejects,
  8.4e-5 cm (exact) and 3.3e-4 cm (fast) from the reference.
* Other scenarios: no unconverged step, exhausted budget or held node in any of the 82
  site-years, in either tier (before the four changes: 55 / 52
  unconverged steps in 19 / 20 site-years, one exhausted budget). M1 holds in 17 of 82 in both tiers
  against 18 at 96 x 8; the one difference, US-Mj1 2001 (96 x 8 0.0492, adaptive 0.0504 / 0.0503
  cm), fails with the converged reference too (0.0504 cm).
  12 (exact) / 13 (fast) site-years are further from RZWQM2 than 96 x 8 by more than 0.002 cm; the
  clean ones among them (CA-MA1 2008, US-Mj1 2002, US_Rockford_Alfalfa 2007) are within 5e-4 cm of
  the converged reference, so it is 96 x 8's own discretisation error that brings those closer. On
  the 50 clean reference years the numerical error is within 1e-3 cm in 49 (exact; CA-ER1 2011 4.0e-3
  cm) and 44 (fast; worst CA-ER1 2011 7.9e-3 cm).
* Step balance: in all 91 site-years of both tiers every accepted step closes its balance to <= 9.6e-12 cm
  (the Newton tolerance is 1e-10 cm) and every year's sum of |day balance| is <= 6.3e-11 cm.

The pins are the measured values to 6 significant digits; the replay is deterministic on fixed
inputs, so the tolerance is 1e-5 cm as in ``test_richards_years``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from richards_adaptive_years import SITES, TIERS, all_site_years, day_row
from richards_years import CATPA_BASE, M1_STORAGE_RMSE, M1_THETA_RMSE
from test_richards_years import SCENARIO_RESTART, _fingerprint_ok

pytestmark = [
    pytest.mark.allow_skip(
        reason="needs the CA-TPA 2015-2023 run, the scenario runs and the converged reference (data dir)"
    ),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="M1 is a float64 comparison"),
]

#: pinned replay values [cm] (6-digit pins of a deterministic replay)
PIN_ABS = 1e-5
#: numerical part of the M1 margin: adaptive vs the converged reference
NUM_MARGIN_CM = 1e-3
#: sub-step balance tolerance of the Newton solve (AdaptiveStepping.newton_tol_balance, float64)
TOL_BALANCE = 1e-10
#: a site-year further from RZWQM2 than 96 x 8 by more than this is examined
WORSE_THAN_96_CM = 0.002

#: 2016-09-10 at CA-TPA: (accepted steps, rejected tries, Newton evaluations)
DAY_2016_09_10 = {"exact": (261.0, 0.0, 929.0), "fast": (128.0, 0.0, 495.0)}
#: M1 passes outside CA-TPA: (adaptive, 96 x 8) of 82
N_PASS_OTHER = {"exact": (17, 18), "fast": (17, 18)}
#: clean reference site-years outside CA-TPA: (within NUM_MARGIN_CM of the reference, all)
N_CLEAN_WITHIN = {"exact": (49, 50), "fast": (44, 50)}
_WORSE_BOTH = {
    ("CA-ER1", 2016), ("CA-ER1", 2020), ("CA-ER1", 2023), ("CA-MA1", 2008), ("US-Mj1", 2002),
    ("US-S2", 2023), ("US-manilacotton", 2018), ("US_OPE", 2015), ("US_Rockford_Alfalfa", 2007),
    ("US_Rockford_Alfalfa", 2008), ("US_Rockford_Alfalfa", 2009), ("US_Rockford_Alfalfa", 2011),
}  # fmt: skip
#: site-years further from RZWQM2 than 96 x 8 by more than WORSE_THAN_96_CM
WORSE_96 = {"exact": _WORSE_BOTH, "fast": _WORSE_BOTH | {("US-S2", 2021)}}
#: largest |sum of a year's step balances| [cm] (bound: <= 1e-9 cm a year)
YEAR_BALANCE_CM = 1e-9

#: (site, year) -> tier -> (storage RMSE vs RZWQM2, max |storage - converged reference|, unconverged steps)
PINS: dict[tuple[str, int], dict[str, tuple[float, float, float]]] = {
    ("CA-TPA", 2015): {"exact": (0.0194332, 9.91759e-05, 0), "fast": (0.0195066, 0.000295624, 0)},
    ("CA-TPA", 2016): {"exact": (0.0353815, 0.000117776, 0), "fast": (0.0354944, 0.000637408, 0)},
    ("CA-TPA", 2017): {"exact": (0.0335203, 0.000150297, 0), "fast": (0.0336363, 0.000905041, 0)},
    ("CA-TPA", 2018): {"exact": (0.0449496, 8.51823e-05, 0), "fast": (0.0449941, 0.000455532, 0)},
    ("CA-TPA", 2019): {"exact": (0.0307449, 5.20365e-05, 0), "fast": (0.0308019, 0.00022574, 0)},
    ("CA-TPA", 2020): {"exact": (0.0216602, 0.0480603, 0), "fast": (0.0217124, 0.0481294, 0)},
    ("CA-TPA", 2021): {"exact": (0.028671, 0.000123935, 0), "fast": (0.0287826, 0.00082138, 0)},
    ("CA-TPA", 2022): {"exact": (0.0121697, 4.72059e-05, 0), "fast": (0.0122379, 0.000285099, 0)},
    ("CA-TPA", 2023): {"exact": (0.0277877, 7.08623e-05, 0), "fast": (0.0278892, 0.000402324, 0)},
    ("CA-ER1", 2011): {"exact": (0.0796241, 0.00401164, 0), "fast": (0.0807069, 0.00785874, 0)},
    ("CA-ER1", 2012): {"exact": (0.0569247, 0.0113586, 0), "fast": (0.057982, 0.00916963, 0)},
    ("CA-ER1", 2013): {"exact": (0.0432127, 0.00320728, 0), "fast": (0.0438667, 0.00239088, 0)},
    ("CA-ER1", 2014): {"exact": (0.102548, 0.000380775, 0), "fast": (0.102549, 0.000412382, 0)},
    ("CA-ER1", 2015): {"exact": (0.0234615, 0.0130741, 0), "fast": (0.0340516, 0.0358018, 0)},
    ("CA-ER1", 2016): {"exact": (0.206527, 0.00824614, 0), "fast": (0.188845, 0.0392925, 0)},
    ("CA-ER1", 2017): {"exact": (0.085913, 0.000762176, 0), "fast": (0.0859217, 0.00363201, 0)},
    ("CA-ER1", 2018): {"exact": (0.0550993, 0.00123634, 0), "fast": (0.0564244, 0.0146412, 0)},
    ("CA-ER1", 2019): {"exact": (0.11812, 0.00164607, 0), "fast": (0.119248, 0.0106978, 0)},
    ("CA-ER1", 2020): {"exact": (0.796023, 0.0148953, 0), "fast": (0.791297, 0.0295753, 0)},
    ("CA-ER1", 2021): {"exact": (0.0869207, 0.00675388, 0), "fast": (0.0842046, 0.0234591, 0)},
    ("CA-ER1", 2022): {"exact": (0.0847842, 0.000447334, 0), "fast": (0.0847225, 0.00127025, 0)},
    ("CA-ER1", 2023): {"exact": (0.232231, 0.00317979, 0), "fast": (0.232272, 0.00158996, 0)},
    ("CA-MA1", 2008): {"exact": (0.0813264, 7.4178e-05, 0), "fast": (0.0810932, 0.00044813, 0)},
    ("CA-MA1", 2009): {"exact": (0.0769781, 0.00166628, 0), "fast": (0.0769889, 0.00167883, 0)},
    ("CA-MA1", 2010): {"exact": (0.0461038, 2.10476e-05, 0), "fast": (0.046156, 0.000123073, 0)},
    ("CA-MA1", 2011): {"exact": (0.0853299, 0.00101776, 0), "fast": (0.0850774, 0.00427467, 0)},
    ("US-Mj1", 2000): {"exact": (0.0499147, 1.57739e-05, 0), "fast": (0.0498671, 8.81883e-05, 0)},
    ("US-Mj1", 2001): {"exact": (0.0504129, 2.055e-05, 0), "fast": (0.0503415, 0.000119264, 0)},
    ("US-Mj1", 2002): {"exact": (0.0997826, 4.18241e-05, 0), "fast": (0.0996385, 0.000246378, 0)},
    ("US-Mj1", 2003): {"exact": (0.0558411, 2.64064e-05, 0), "fast": (0.0557614, 0.000144794, 0)},
    ("US-Mj1", 2004): {"exact": (0.031911, 9.73389e-06, 0), "fast": (0.0318829, 5.36864e-05, 0)},
    ("US-Mj1", 2005): {"exact": (0.0175045, 6.33787e-06, 0), "fast": (0.0174853, 3.54206e-05, 0)},
    ("US-Mj1", 2006): {"exact": (0.0100815, 3.84483e-06, 0), "fast": (0.0100751, 2.02366e-05, 0)},
    ("US-Mj1", 2007): {"exact": (0.0120833, 5.92051e-06, 0), "fast": (0.0120948, 3.06232e-05, 0)},
    ("US-Mj1", 2008): {"exact": (0.762972, 9.10723e-06, 0), "fast": (0.762996, 4.71334e-05, 0)},
    ("US-Mj1", 2009): {"exact": (0.00337383, 2.77284e-06, 0), "fast": (0.00337127, 1.38914e-05, 0)},
    ("US-Mj1", 2010): {"exact": (0.613055, 1.18367e-05, 0), "fast": (0.613075, 5.65028e-05, 0)},
    ("US-Mj1", 2011): {"exact": (1.11791, 1.49751e-05, 0), "fast": (1.11793, 7.22519e-05, 0)},
    ("US-Mj1", 2012): {"exact": (0.0969984, 3.73409e-05, 0), "fast": (0.0969013, 0.000218689, 0)},
    ("US-Mj1", 2013): {"exact": (0.0375045, 1.67491e-06, 0), "fast": (0.0375059, 1.09241e-05, 0)},
    ("US-Mj1", 2014): {"exact": (0.0418363, 1.15362e-05, 0), "fast": (0.0418346, 6.95292e-05, 0)},
    ("US-Mj1", 2015): {"exact": (0.0233227, 4.36308e-06, 0), "fast": (0.0233203, 2.27703e-05, 0)},
    ("US-Mj1", 2016): {"exact": (0.0209462, 1.39904e-05, 0), "fast": (0.0209091, 7.69045e-05, 0)},
    ("US-Mj1", 2017): {"exact": (0.00826276, 2.38472e-06, 0), "fast": (0.00825625, 1.25161e-05, 0)},
    ("US-Mj1", 2018): {"exact": (0.634746, 1.20808e-05, 0), "fast": (0.634771, 5.94207e-05, 0)},
    ("US-Mj1", 2019): {"exact": (0.0698713, 5.71153e-06, 0), "fast": (0.0698852, 2.86132e-05, 0)},
    ("US-Mj1", 2020): {"exact": (0.0978929, 6.79529e-06, 0), "fast": (0.0979122, 3.57884e-05, 0)},
    ("US-Mj1", 2021): {"exact": (0.00841331, 2.45046e-06, 0), "fast": (0.00840534, 1.59254e-05, 0)},
    ("US-Mj1", 2022): {"exact": (0.0312145, 1.75325e-05, 0), "fast": (0.0311627, 0.000100877, 0)},
    ("US-Mj1", 2023): {"exact": (0.594991, 9.0003e-06, 0), "fast": (0.595009, 4.35617e-05, 0)},
    ("US-S2", 2016): {"exact": (0.0611549, 2.91331e-05, 0), "fast": (0.0610581, 0.000170445, 0)},
    ("US-S2", 2017): {"exact": (0.357533, 0.000180845, 0), "fast": (0.357749, 0.000527364, 0)},
    ("US-S2", 2018): {"exact": (0.327543, 0.000173868, 0), "fast": (0.327688, 0.00048005, 0)},
    ("US-S2", 2019): {"exact": (0.301944, 0.000107498, 0), "fast": (0.302144, 0.000306673, 0)},
    ("US-S2", 2020): {"exact": (0.349949, 0.000134946, 0), "fast": (0.350103, 0.000384826, 0)},
    ("US-S2", 2021): {"exact": (0.558506, 0.000296052, 0), "fast": (0.558788, 0.000855972, 0)},
    ("US-S2", 2022): {"exact": (0.0905235, 0.000196714, 0), "fast": (0.0904977, 0.000263729, 0)},
    ("US-S2", 2023): {"exact": (0.147362, 0.00012354, 0), "fast": (0.147036, 0.000639971, 0)},
    ("US-manilacotton", 2014): {"exact": (0.0831839, 0.0289201, 0), "fast": (0.0833541, 0.0289178, 0)},
    ("US-manilacotton", 2015): {"exact": (0.151322, 0.000729797, 0), "fast": (0.151349, 0.00496301, 0)},
    ("US-manilacotton", 2016): {"exact": (0.0771992, 0.0594737, 0), "fast": (0.0772953, 0.0571387, 0)},
    ("US-manilacotton", 2017): {"exact": (0.0514221, 0.000105089, 0), "fast": (0.051467, 0.000280743, 0)},
    ("US-manilacotton", 2018): {"exact": (0.14463, 0.0754762, 0), "fast": (0.145775, 0.0674563, 0)},
    ("US-manilacotton", 2019): {"exact": (0.102491, 0.0571198, 0), "fast": (0.102464, 0.0598995, 0)},
    ("US-manilacotton", 2020): {"exact": (0.0669971, 5.21227e-05, 0), "fast": (0.0670701, 0.000367051, 0)},
    ("US-manilacotton", 2021): {"exact": (0.0538584, 8.50019e-05, 0), "fast": (0.0539646, 0.000350187, 0)},
    ("US-manilacotton", 2022): {"exact": (0.0639523, 7.03147e-05, 0), "fast": (0.0640389, 0.000388716, 0)},
    ("US-manilacotton", 2023): {"exact": (0.073224, 0.0005533, 0), "fast": (0.0755995, 0.0326835, 0)},
    ("US_OPE", 2015): {"exact": (0.225981, 0.108663, 0), "fast": (0.226029, 0.1085, 0)},
    ("US_OPE", 2016): {"exact": (0.0691246, 3.76157e-05, 0), "fast": (0.0691803, 0.000225682, 0)},
    ("US_OPE", 2017): {"exact": (0.0586582, 0.0127609, 0), "fast": (0.0587161, 0.0132558, 0)},
    ("US_OPE", 2018): {"exact": (0.384485, 0.0209059, 0), "fast": (0.384549, 0.0211496, 0)},
    ("US_OPE", 2019): {"exact": (0.0827391, 0.000131575, 0), "fast": (0.0827903, 0.00031974, 0)},
    ("US_OPE", 2020): {"exact": (0.108194, 0.000127483, 0), "fast": (0.108233, 0.000431941, 0)},
    ("US_OPE", 2021): {"exact": (0.0632958, 5.92238e-05, 0), "fast": (0.0634103, 0.000354574, 0)},
    ("US_OPE", 2022): {"exact": (0.033998, 0.0194859, 0), "fast": (0.0340501, 0.0193352, 0)},
    ("US_OPE", 2023): {"exact": (0.0955533, 0.000595922, 0), "fast": (0.0955283, 0.00153528, 0)},
    ("US_Rockfish", 2016): {"exact": (0.455201, 0.00023672, 0), "fast": (0.455119, 0.000967923, 0)},
    ("US_Rockfish", 2017): {"exact": (0.302304, 0.00053956, 0), "fast": (0.302367, 0.000549027, 0)},
    ("US_Rockfish", 2018): {"exact": (0.208444, 0.000536462, 0), "fast": (0.208539, 0.000797, 0)},
    ("US_Rockfish", 2019): {"exact": (0.262079, 0.000173396, 0), "fast": (0.262036, 0.00040056, 0)},
    ("US_Rockfish", 2020): {"exact": (0.154642, 0.000206463, 0), "fast": (0.154751, 0.00111819, 0)},
    ("US_Rockfish", 2021): {"exact": (0.290169, 0.000507735, 0), "fast": (0.290205, 0.000680985, 0)},
    ("US_Rockfish", 2022): {"exact": (0.280346, 0.000343109, 0), "fast": (0.2804, 0.00152505, 0)},
    ("US_Rockfish", 2023): {"exact": (0.375923, 0.000322861, 0), "fast": (0.375953, 0.000890971, 0)},
    ("US_Rockford_Alfalfa", 2007): {"exact": (0.296376, 4.11044e-05, 0), "fast": (0.296054, 0.000498063, 0)},
    ("US_Rockford_Alfalfa", 2008): {"exact": (0.268163, 0.000986833, 0), "fast": (0.269752, 0.00524126, 0)},
    ("US_Rockford_Alfalfa", 2009): {"exact": (0.725439, 0.000758356, 0), "fast": (0.726356, 0.00231517, 0)},
    ("US_Rockford_Alfalfa", 2010): {"exact": (0.875082, 0.00590327, 0), "fast": (0.879275, 0.00954679, 0)},
    ("US_Rockford_Alfalfa", 2011): {"exact": (0.74084, 0.00136663, 0), "fast": (0.742103, 0.00433183, 0)},
    ("US_Rockford_Alfalfa", 2012): {"exact": (1.05519, 0.00132374, 0), "fast": (1.05707, 0.00445863, 0)},
}


@pytest.fixture(scope="module")
def results(data_dir: Path) -> dict[str, Any]:
    for p in (
        data_dir / CATPA_BASE / "CA-TPA.ana",
        data_dir / "validation/w1_e0/ref/CA-TPA_m1_fix_cnfb_960.npz",
    ):
        if not p.is_file():
            pytest.skip(f"{p} not found")
    res = all_site_years(data_dir, SITES)
    for site, r in res.items():
        _fingerprint_ok(r["rep"], site)
        assert r["ref"] is not None, site
    return res


def _tab(results: dict[str, Any], site: str, tier: str) -> Any:
    return results[site]["tables"][tier].set_index("year")


def _pinned(results: dict[str, Any], site: str) -> None:
    for tier in TIERS:
        t = _tab(results, site, tier)
        for (s, y), pins in PINS.items():
            if s != site:
                continue
            rmse, err, unc = pins[tier]
            assert t.loc[y, "storage_rmse_cm"] == pytest.approx(rmse, abs=PIN_ABS), (tier, y)
            assert t.loc[y, "err_ref_max_cm"] == pytest.approx(err, abs=PIN_ABS), (tier, y)
            assert t.loc[y, "n_unconverged"] == unc, (tier, y)


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_catpa_every_year_meets_m1_and_matches_the_converged_reference(
    results: dict[str, Any], tier: str
) -> None:
    """CA-TPA 2015-2023 (restart): every year meets M1, every step converges, and on the years with
    a clean reference the storage is within NUM_MARGIN_CM of it (the numerical part of the margin)."""
    _pinned(results, "CA-TPA")
    t = _tab(results, "CA-TPA", tier)
    assert list(t.index) == list(range(2015, 2024))
    assert (t.storage_rmse_cm < M1_STORAGE_RMSE).all()
    assert (t.theta_rmse < M1_THETA_RMSE).all()
    for c in ("n_unconverged", "budget_exhausted", "n_clamp", "n_active"):
        assert (t[c] == 0.0).all(), c
    clean = t[t.ref_clean]
    assert set(clean.index) == set(range(2015, 2024)) - {2020}  # 2020 has an unconverged reference step
    assert (clean.err_ref_max_cm <= NUM_MARGIN_CM).all()
    # the reference's own spread (960 vs 1920 steps a day) is far below the margin on the clean years
    assert (clean.ref_self_check_cm < 2e-5).all()
    # 2020: the two tiers agree to the margin where the reference cannot be the yardstick
    assert t.loc[2020, "d_exact_fast_max_cm"] <= NUM_MARGIN_CM
    print(t[["storage_rmse_cm", "err_ref_max_cm", "d_exact_fast_max_cm", "newton_per_day"]].to_string())


@pytest.mark.slow
def test_catpa_2016_09_10_converges(results: dict[str, Any]) -> None:
    """The 2.56 cm storm onto the drought profile (24 x 3: -0.267 cm balance error) converges."""
    r = results["CA-TPA"]
    for tier in TIERS:
        d = day_row(r["rep"], r["runs"][tier], r["ref"], "2016-09-10")
        print(tier, d)
        assert d["infiltration_cm"] == pytest.approx(2.563, abs=5e-4)
        assert d["n_unconverged"] == 0.0 and d["budget_exhausted"] == 0.0 and d["n_clamp"] == 0.0
        assert d["step_balance_max"] <= TOL_BALANCE
        assert abs(d["balance_error"]) <= d["n_steps"] * TOL_BALANCE
        assert abs(d["err_ref_cm"]) <= NUM_MARGIN_CM
        assert (d["n_steps"], d["n_rejects"], d["n_newton"]) == DAY_2016_09_10[tier]


@pytest.mark.slow
@pytest.mark.parametrize("tier", TIERS)
def test_other_scenarios_every_year_against_96x8(results: dict[str, Any], tier: str) -> None:
    """The 82 site-years of the 8 comparable scenarios, per site-year, against the fixed 96 x 8.

    96 x 8 (fully implicit, 8 iterations) is not the converged solution: where the adaptive run is
    further from RZWQM2 than 96 x 8 on a clean reference year, it is within NUM_MARGIN_CM of the
    converged reference, so the difference is 96 x 8's own discretisation error; the one year that
    96 x 8 passes and the adaptive run fails fails with the converged reference too.
    """
    rows = {}
    for site in SITES[1:]:
        _pinned(results, site)
        for y, row in _tab(results, site, tier).iterrows():
            rows[(site, int(y))] = row
    assert set(rows) == set(SCENARIO_RESTART)
    n_pass = sum(r.storage_rmse_cm < M1_STORAGE_RMSE for r in rows.values())
    n_pass_96 = sum(v[0] < M1_STORAGE_RMSE for v in SCENARIO_RESTART.values())
    assert (n_pass, n_pass_96) == N_PASS_OTHER[tier]
    for key, r in rows.items():
        r96 = SCENARIO_RESTART[key][0]
        if r96 < M1_STORAGE_RMSE <= r.storage_rmse_cm:
            assert r.ref_storage_rmse_cm >= M1_STORAGE_RMSE, key
        if r.storage_rmse_cm - r96 > WORSE_THAN_96_CM and r.ref_clean:
            assert r.err_ref_max_cm <= NUM_MARGIN_CM, key
    worse = {k for k, r in rows.items() if r.storage_rmse_cm - SCENARIO_RESTART[k][0] > WORSE_THAN_96_CM}
    assert worse == WORSE_96[tier]
    clean = [r for r in rows.values() if r.ref_clean]
    assert (sum(r.err_ref_max_cm <= NUM_MARGIN_CM for r in clean), len(clean)) == N_CLEAN_WITHIN[tier]
    # every step of every site-year converges, no budget runs out, no node is held at the guard
    for c in ("n_unconverged", "budget_exhausted", "n_active"):
        assert all(r[c] == 0.0 for r in rows.values()), c


@pytest.mark.slow
def test_every_accepted_step_closes_its_balance(results: dict[str, Any]) -> None:
    """In every site-year of both tiers every accepted step closes its water balance to the Newton
    tolerance, every day to its steps times it, and every year to YEAR_BALANCE_CM."""
    worst_step, worst_year = 0.0, 0.0
    for site in SITES:
        for tier in TIERS:
            t = _tab(results, site, tier)
            r = results[site]["runs"][tier]
            assert (r["n_unconverged"] == 0.0).all() and (r["n_active"] == 0.0).all(), (site, tier)
            assert (t.step_balance_max_cm <= TOL_BALANCE).all(), (site, tier)
            day = np.abs(r["balance_error"])
            assert np.all(day <= np.maximum(r["n_steps"], 1.0) * TOL_BALANCE), (site, tier)
            assert (t.year_balance_cm <= YEAR_BALANCE_CM).all(), (site, tier)
            worst_step = max(worst_step, float(t.step_balance_max_cm.max()))
            worst_year = max(worst_year, float(t.year_balance_cm.max()))
    print(f"largest step balance {worst_step:.3e} cm, largest year sum of |day balance| {worst_year:.3e} cm")
