"""Soil water and evapotranspiration as calibration outputs of the free-run DSSAT day
(``calib/dssat_day.py``: ``OUT_NAMES`` ``et`` and ``sw<k>``), no data.

The synthetic run of ``test_calib_day_soil`` (the UFGA8201 soil profile, nine model layers, the crop
conformance season) is simulated twice: by :class:`DaySimulator` with entries reading ``et`` and
the soil water of every layer on every day, and by the day itself (``run_sites`` on the season
model, whose outputs are :func:`agrijax.models.day_dssat486.day_outputs`: ``soil_sw``, ``evap``,
``ep``). The entries equal the day state: ``sw<k>`` the end-of-day water of layer ``k``, ``et`` =
``EVAP + EP`` (``SPAM.for``: ``ET = EVAP + EP``). The crop entries of the same call equal those of a
simulator without water entries (a program carries the water outputs only when its entries read
them), and an entry naming a layer the run does not have is refused. The simulators run the
padded readout (entries on every simulated day); the season readout of water entries is checked in
``test_dssat_day_readout``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from agrijax.calib.dssat_day import (
    CROP_OUT_NAMES,
    E_DATE,
    E_DAY,
    E_FINAL,
    OUT_NAMES,
    SW_LAYERS,
    WATER_OUT_NAMES,
    DaySimulator,
    Entry,
)
from agrijax.io.dssat.native_soil import NL
from agrijax.sites.dssat_free_run import _season_model, stack_trees

from . import test_calib_day_soil as synth

#: the crop outputs and their indices before the water outputs were added (callers index them)
CROP_BEFORE = ("lai", "cwad", "gwad", "g_ad", "lsd", "lwad", "swad", "rwad", "pwad")


def test_out_names_keep_the_crop_indices_and_name_every_dssat_layer() -> None:
    assert OUT_NAMES[: len(CROP_BEFORE)] == CROP_BEFORE == CROP_OUT_NAMES
    assert SW_LAYERS == NL
    assert WATER_OUT_NAMES == ("et", *(f"sw{k}" for k in range(1, NL + 1)))
    assert OUT_NAMES[len(CROP_BEFORE) :] == WATER_OUT_NAMES
    assert len(set(OUT_NAMES)) == len(OUT_NAMES)


#: marks of the tests on the synthetic run (built in float64, as ``test_calib_day_soil``)
X64 = (
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the synthetic run is built in float64"),
    pytest.mark.allow_skip(reason="float64 synthetic run, skipped in the float32 tier"),
)


def _x64(f):
    for m in reversed(X64):
        f = m(f)
    return f


@pytest.fixture(scope="module")
def run():
    return synth._inputs(synth._profile())


@pytest.fixture(scope="module")
def season(run):
    """The day's own outputs (``day_outputs``) over the simulator's padded days, ``[T, ...]``."""
    from agrijax.core.runtime import run_sites

    n = DaySimulator([run], [[Entry(E_FINAL, 0)]], readout="padded").n_days[(run.nl, run.mesev)]
    p = run.params()
    o = run_sites(_season_model(run.mesev), *(stack_trees([t]) for t in (p, run.forcing(n), run.state(p))))
    return {k: np.asarray(v)[0] for k, v in o.items()}, n


def _water_entries(nl: int, n: int) -> list[Entry]:
    et = OUT_NAMES.index("et")
    ents = [Entry(E_DAY, et, t) for t in range(n)]
    ents += [Entry(E_DAY, OUT_NAMES.index(f"sw{k}"), t) for k in range(1, nl + 1) for t in range(n)]
    return ents


def _crop_entries(n: int) -> list[Entry]:
    lai, gwad = OUT_NAMES.index("lai"), OUT_NAMES.index("gwad")
    return [Entry(E_DATE, 1), Entry(E_FINAL, gwad), *(Entry(E_DAY, lai, t) for t in range(0, n, 5))]


@_x64
def test_water_entries_equal_the_day_state(run, season) -> None:
    o, n = season
    nl = run.nl
    ents = _crop_entries(n) + _water_entries(nl, n)
    sim = DaySimulator([run], [ents], readout="padded")
    assert sim.water[(nl, run.mesev)] == ("et", *(f"sw{k}" for k in range(1, nl + 1)))
    theta = np.asarray([[run.published()[c] for c in ("P1", "P2", "P5", "G2", "G3", "PHINT")]])
    y = sim(theta, np.zeros(1, np.int64))[0]
    k0 = len(_crop_entries(n))
    et = y[k0 : k0 + n]
    sw = y[k0 + n : k0 + n + nl * n].reshape(nl, n).T
    # the day state: end-of-day soil water of every layer, ET = EVAP + EP
    # two compiled programs (DaySimulator against run_sites): fusion or FMA contraction may differ
    # in the last bit on other CPUs; layer values differ by 1e-2 or more, so a wrong layer or day fails
    np.testing.assert_allclose(sw, o["soil_sw"][:n, :nl], rtol=1e-13, atol=0.0)
    # ET is summed in the jitted scan here and on the host there: the order may differ by an ulp
    np.testing.assert_allclose(et, o["evap"][:n] + o["ep"][:n], rtol=1e-13, atol=0.0)
    assert np.all(np.isfinite(et)) and np.ptp(et) > 0.0 and np.ptp(sw[:, 0]) > 0.0
    # the crop entries of the same call equal those of a simulator without water entries
    crop_only = DaySimulator([run], [_crop_entries(n)], readout="padded")
    assert crop_only.water[(nl, run.mesev)] == ()
    np.testing.assert_array_equal(crop_only(theta, np.zeros(1, np.int64))[0], y[:k0])


@_x64
def test_an_entry_beyond_the_run_layers_is_refused(run) -> None:
    deeper = OUT_NAMES.index(f"sw{run.nl + 1}")
    with pytest.raises(ValueError, match=rf"sw{run.nl + 1}, but the run has {run.nl} soil layers"):
        DaySimulator([run], [[Entry(E_DAY, deeper, 3)]], readout="padded")
    # an output index outside OUT_NAMES is refused, not clipped to the last row
    with pytest.raises(ValueError, match="OUT_NAMES has"):
        DaySimulator([run], [[Entry(E_FINAL, len(OUT_NAMES))]], readout="padded")
    # the deepest layer the run has is accepted, an E_FINAL entry too
    sim = DaySimulator([run], [[Entry(E_FINAL, OUT_NAMES.index(f"sw{run.nl}"))]], readout="padded")
    assert sim.water[(run.nl, run.mesev)] == (f"sw{run.nl}",)
