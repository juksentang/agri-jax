"""A one-row per-season parameter table reproduces the frozen CERES baseline.

The per-season management (``YRPLT``, ``PLTPOP``, ``SDEPTH``,
``ROWSPC``) moves into an ``n_season`` table (``CeresSeasons``) and a single season is the
one-row table; the acceptance condition is that the 58 M2 treatments and the frozen baseline stay
bit for bit. Each test here reruns dscsm048 for one treatment, runs the model with
``CeresSeasons.single(params)`` and the season index, and requires

* its daily ``PlantGro`` outputs to equal the validated runner's (no table) and every leaf of its
  daily state to equal the no-table run's, bit for bit (``ceres_baseline.compute_case`` raises
  otherwise), and
* every array of the frozen snapshot (parameters, forcing, every daily state leaf, outputs and,
  for the two gradient cases, the season gradients) to be reproduced within the baseline's
  atol 1e-12, rtol 1e-12 (``ceres_baseline.compare_case``); the new leaves (the season table,
  the season index, the snow port) are checked by ``ceres_baseline._legacy_layout``.

Skips (``allow_skip``) like ``test_ceres_baseline.py``; slow.
"""

from __future__ import annotations

import ceres_baseline as bl
import pytest
from test_ceres_baseline import _needs_baseline, snapshot, tables  # noqa: F401 (fixtures)


@_needs_baseline
@pytest.mark.parametrize(("exp", "trno"), bl.CASES, ids=[bl.case_id(e, t) for e, t in bl.CASES])
def test_one_row_season_table_matches_the_baseline(snapshot, tables, tmp_path, exp, trno):  # noqa: F811
    ref, _ = snapshot
    got = bl.compute_case(exp, trno, tmp_path / f"{exp}_{trno}", tables=tables, seasons=True)
    msg = bl.compare_case(bl.case_id(exp, trno), got, ref)
    assert msg is None, msg
