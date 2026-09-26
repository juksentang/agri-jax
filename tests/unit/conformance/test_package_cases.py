"""Every conformance case of this repository against every check (CI unit tier, no data).

The ``conformance_case`` fixture is parametrised by registry key by the kit's pytest plugin
(registered in ``tests/conftest.py``); ``--agrijax-key 'pet/*'`` narrows the run.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from agrijax.testing.conformance import CHECKS, ConformanceCase, check_name
from agrijax.testing.conformance.pytest_plugin import run_check


@pytest.mark.parametrize("check", CHECKS, ids=check_name)
def test_package_case(conformance_case: ConformanceCase, check: Callable[[ConformanceCase], None]) -> None:
    run_check(conformance_case, check)


# ------------------------------------------------------------------ the synthetic inputs are not vacuous
def _case(key: str) -> ConformanceCase:
    from agrijax.testing.conformance import builtin_cases

    return next(c for c in builtin_cases() if c.key == key)


def test_crop_cases_start_from_a_growing_crop() -> None:
    import numpy as np

    c = _case("crop/ceres_maize.growth@dssat-4.8.6.0:faithful")
    stages = {}
    for variant in c.variants:
        s, _, f = c.inputs(np.float32 if not _x64() else np.float64, variant)
        stages[variant] = int(np.asarray(s.phen.istage)[0])
        assert float(np.asarray(s.growth.lai)[0]) > 0.5, variant
        assert float(np.max(np.asarray(s.roots.rlv))) > 0.0, variant
        assert np.shape(f.tmax) == (c.n_days,)
    assert 1 <= stages["vegetative"] <= 3 < stages["grain_fill"] <= 5, stages


def test_rootwu_case_takes_up_water_and_reaches_the_excess_water_branch() -> None:
    import numpy as np

    from agrijax.processes.soil_water.uptake import SoilView, rootwu_estimate

    c = _case("water_supply/rootwu@dssat-4.8.6.0:faithful")
    dt = np.float64 if _x64() else np.float32
    for variant, excess in (("nominal", False), ("wet", True)):
        s, p, _ = c.inputs(dt, variant)
        soil = SoilView(dlayr=p.dlayr, ll=p.ll, sat=p.sat, sw=s.water.sw)
        res = rootwu_estimate(s.root, soil, s.tss, p.coef())
        assert float(np.min(np.asarray(res.trwup))) > 0.0, variant
        air = np.asarray(p.sat - s.water.sw)
        active = (np.asarray(res.tss) > p.coef().tss_days) & (air < np.asarray(s.root.pormin)[:, None])
        assert bool(np.any(active)) is excess, variant
    s, p, _ = c.inputs(dt, "no_canopy")
    res = rootwu_estimate(s.root, SoilView(dlayr=p.dlayr, ll=p.ll, sat=p.sat, sw=s.water.sw), s.tss, p.coef())
    assert float(np.max(np.asarray(res.trwup))) == 0.0


def _x64() -> bool:
    import jax

    return bool(jax.config.read("jax_enable_x64"))
