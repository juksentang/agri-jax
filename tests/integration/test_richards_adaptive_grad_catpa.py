"""Gradients of the adaptive Richards mode on CA-TPA against central differences.

One restart year of the M1 replay (CA-TPA 2015, an ordinary year, and 2016, with the 2016-09-10
storm), ``AdaptiveStepping.exact()``, as a function of common factors ``s`` on the five
Brooks-Corey parameter classes (all horizons; :data:`richards_adaptive_years.SCALED`). Outputs: 31
December storage, the year's drainage, the M1 loss.

* frozen step tables: AD (the implicit-function VJP of every sub-step along the frozen step
  tables, the step decisions are constants) against central differences on the same frozen
  tables, at the two relative steps 1e-4 and 1e-6 (the two-step kink test);
* full adaptive run: central differences (the tables may move) at 1e-6 and 1e-4,
  whose step-table signature is compared with the nominal run's, and secants at 1 % and 5 %.

Measured (rorqual, float64; ``<data>/validation/w1_b/grad_CA-TPA_<year>_exact.csv``):

* the value of the frozen replay equals the adaptive run's to 2e-13 cm;
* frozen step tables: AD equals the central difference at 1e-6 to <= 2.4e-7 relative (storage
  and drainage <= 3e-8; the M1 loss, an RMSE near its minimum, 2.4e-7); at 1e-4 the difference is
  <= 5.5e-5 for storage and drainage and up to 1.8e-3 for the M1 loss, the curvature of the RMSE
  (its minimum lies within 1e-3 of s = 1), not a kink: the 1e-6 value agrees with AD;
* full adaptive run: no Newton count changes anywhere in the year at +-1e-6, and there the full adaptive
  difference equals the frozen one; at 1e-4 zero to five days change their Newton counts and the
  full difference stays within 5.2e-6 (relative) of the frozen one. At the calibration scale (1 %, 5 % secants), 2015 storage and
  drainage agree with AD to <= 0.06 % (1 %) and 0.6 % (5 %); in 2016 the 1 % secants differ from AD
  by up to 10 % (drainage against theta_r and theta_s), with 8-37 days changing their Newton counts
  in between. The staircase scan of 2016 over lambda
  separates the two: the jumps where a step table changes are <= 4.5e-6 cm (drainage), i.e.
  <= 2.3e-4 cm per unit s in a 1 % secant, while AD of the drainage varies from 20.2 to 22.0 over
  +-10 % with its fastest change between s = 0.998 and 1.0 (kinks at identical step tables, jump
  terms up to 8e-5 cm), so the secant differs from AD by the derivative's variation, not by steps.

With the conservative dry bound (rorqual, float64): the restart window 2016-08-25 to 09-30 from
RZWQM2's 2016-08-24 profile (a node drier than h_min) converges at every step (before it: one
unconverged first step). AD equals the frozen-table differences at 1e-6 to <= 1.4e-7 relative for
drainage and for every output against ksat; the storage and the M1 loss against lambda, hb, theta_r
and theta_s differ by 1.6-10 %, and there the differences at 1e-4 and 1e-6 disagree by 0.7-11 % too
(a kink within 1e-4 of s = 1). With the uptake removed every class and output agrees to <= 6.4e-7:
the kink is the uptake cap at theta(h_min), which those four classes set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import pytest
from richards_adaptive_years import grad_table, replay_for
from richards_years import CATPA_BASE

pytestmark = [
    pytest.mark.allow_skip(reason="needs the CA-TPA 2015-2023 reference run under the data dir"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="central differences need float64"),
]


@pytest.fixture(scope="module")
def catpa(data_dir: Path) -> Any:
    if not (data_dir / CATPA_BASE / "CA-TPA.ana").is_file():
        pytest.skip("CA-TPA 2015-2023 run not found")
    return replay_for("CA-TPA", data_dir)


YEARS = (2015, 2016)
#: bound on the relative AD - FD difference on the frozen step tables at the smaller step (measured 2.4e-7)
REL_2A = 1e-6
#: largest |frozen replay - adaptive run| [cm] (measured 2e-13)
REPLAY_ABS = 1e-11
#: the two-step kink test for storage and drainage: FD at 1e-4 vs 1e-6 (measured 5.5e-5)
REL_TWO_STEP = 1e-4

#: 1 % secants of the full adaptive run, storage and drainage (measured; the calibration-scale record)
SECANT_1PCT: dict[int, dict[tuple[str, str], float]] = {
    2015: {
        ("lambda_", "storage_end"): -20.055024,
        ("lambda_", "drainage"): 19.721231,
        ("hb", "storage_end"): 9.5306699,
        ("hb", "drainage"): -9.5654376,
        ("ksat", "storage_end"): -3.2178463,
        ("ksat", "drainage"): 3.0949539,
        ("theta_r", "storage_end"): 2.9283754,
        ("theta_r", "drainage"): -2.8301216,
        ("theta_s", "storage_end"): 36.770744,
        ("theta_s", "drainage"): -36.934641,
    },
    2016: {
        ("lambda_", "storage_end"): -20.212874,
        ("lambda_", "drainage"): 21.19402,
        ("hb", "storage_end"): 9.2395973,
        ("hb", "drainage"): -11.170795,
        ("ksat", "storage_end"): -3.0599625,
        ("ksat", "drainage"): 3.301062,
        ("theta_r", "storage_end"): 3.0367126,
        ("theta_r", "drainage"): -2.2652602,
        ("theta_s", "storage_end"): 35.756566,
        ("theta_s", "drainage"): -42.258306,
    },
}


@pytest.fixture(scope="module")
def tables(catpa: Any) -> dict[int, Any]:
    return {y: grad_table(catpa, y, "exact") for y in YEARS}


@pytest.mark.slow
@pytest.mark.parametrize("year", YEARS)
def test_gradient_equals_frozen_table_differences(tables: dict[int, Any], year: int) -> None:
    """Frozen step tables, with the two-step kink test: AD against central differences on the same tables."""
    t = tables[year]
    print(t[["class", "output", "ad", "fd_frozen_0.0001", "fd_frozen_1e-06", "rel_frozen_1e-06",
             "rel_frozen_two_step"]].to_string())  # fmt: skip
    assert (t.frozen_minus_search.abs() <= REPLAY_ABS).all()
    assert (t["rel_frozen_1e-06"] <= REL_2A).all()
    smooth = t[t.output != "m1_loss"]
    assert (smooth.rel_frozen_two_step <= REL_TWO_STEP).all()
    assert (t.nominal_unconverged == 0.0).all()


@pytest.mark.slow
@pytest.mark.parametrize("year", YEARS)
def test_gradient_equals_full_adaptive_differences(tables: dict[int, Any], year: int) -> None:
    """Full adaptive run: at +-1e-6 no step table moves and the full adaptive difference is the frozen one;
    the calibration-scale secants are recorded (2015 within 1 % of AD for storage and drainage)."""
    t = tables[year]
    print(t[["class", "output", "ad", "fd_free_1e-06", "same_tables_1e-06", "newton_changed_0.0001",
             "fd_free_0.01", "rel_free_0.01", "newton_changed_0.01", "fd_free_0.05", "rel_free_0.05"]].to_string())  # fmt: skip
    assert t["same_tables_1e-06"].all()
    assert (t["rel_free_1e-06"] <= REL_2A).all()
    for h in (1e-6, 1e-4, 1e-2):
        assert (t[f"unconverged_{h:g}"] == 0.0).all(), h
    smooth = t[t.output != "m1_loss"]
    if year == 2015:
        assert (smooth["rel_free_0.01"] <= 1e-3).all()
        assert (smooth["rel_free_0.05"] <= 1e-2).all()
    for (cls, out), pin in SECANT_1PCT[year].items():
        row = t[(t["class"] == cls) & (t.output == out)].iloc[0]
        assert row["fd_free_0.01"] == pytest.approx(pin, rel=1e-5), (cls, out)


#: a window restarted from RZWQM2's 2016-08-24 profile, which holds a node drier than h_min
WINDOW = ("2016-08-25", "2016-09-30")
#: the classes that set theta(h_min), the water the uptake cap keeps (not ksat)
CAP_CLASSES = {"lambda_", "hb", "theta_r", "theta_s"}


@pytest.mark.slow
def test_gradient_after_a_restart_drier_than_h_min(catpa: Any) -> None:
    """The gradient after a restart from a profile drier than h_min. Before the dry bound the first
    step clamped that node up to h_min and counted as unconverged; with the dry bound at the start
    head every step converges. AD then still differs from the frozen-table differences for the
    storage (and the M1 loss) against the four classes that set theta(h_min), and there the
    differences at 1e-4 and 1e-6 disagree as well (the two-step kink test): the uptake cap
    ``min(S, (theta - theta(h_min) - cutoff) / dt)`` binds on dry nodes, a kink of the model, not of
    the gradient. Without uptake the frozen-table comparison holds for every class and output."""
    t = grad_table(catpa, WINDOW, "exact")
    cols = ["class", "output", "ad", "fd_frozen_0.0001", "fd_frozen_1e-06", "rel_frozen_1e-06",
            "rel_frozen_two_step", "nominal_unconverged"]  # fmt: skip
    print(t[cols].to_string())
    assert (t.nominal_unconverged == 0.0).all()
    assert (t.frozen_minus_search.abs() <= REPLAY_ABS).all()
    off = t[t["rel_frozen_1e-06"] > REL_2A]
    assert set(off["class"]) <= CAP_CLASSES and set(off.output) <= {"storage_end", "m1_loss"}
    assert (off.rel_frozen_two_step > REL_TWO_STEP).all()  # every mismatch sits on a detected kink
    assert (t[t.output == "drainage"]["rel_frozen_1e-06"] <= REL_2A).all()
    t0 = grad_table(catpa, WINDOW, "exact", uptake_scale=0.0)
    print(t0[cols].to_string())
    assert (t0.nominal_unconverged == 0.0).all()
    assert (t0["rel_frozen_1e-06"] <= REL_2A).all()
