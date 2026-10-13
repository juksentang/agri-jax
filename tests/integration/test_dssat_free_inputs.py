"""The free-run inputs without any DSSAT run (``free_run_inputs(..., ref_out=None)``) against the ones
built with the reference run (``dscsm048``: ``DSSAT48.INP``, ``Weather.OUT``, ``Summary.OUT``).

On every run the native inputs support (50 of the 65 runs of the free-run acceptance; the default
tier checks a handful, the slow tier all):

* **CERES-Maize's parameters** rebuilt from the ``.CUL`` / ``.ECO`` / ``.SPE``, the FileX and the
  ``.SOL`` (:mod:`agrijax.io.dssat.native_crop`) equal the ones read from ``DSSAT48.INP``: every
  leaf of the day's params, exactly;
* **the crop weather** of the native weather chain printed as ``Weather.OUT`` prints it equals the
  one read from ``Weather.OUT``: every leaf of the forcing (season and real-weather extension),
  exactly; the morning state exactly;
* **the season**: the simulated days end where the reference run ends (the model's own maturity,
  which is DSSAT's on these runs); the emergence, silking and maturity dates equal ``dscsm048``'s and
  the yield is within the acceptance's 2 % of its ``HWAM``.
"""

from __future__ import annotations

import os
from pathlib import Path

import day_dssat486_free_harness as h
import jax
import numpy as np
import pytest
import test_dssat_native_inputs as ni

from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
from agrijax.sites.dssat_free_run import FreeRunInputs, NativeInputError, free_run_inputs

pytestmark = [
    pytest.mark.allow_skip(
        reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT), the DSSAT example data and the CA-TPA DSSAT case"
    ),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

#: real-weather days after the season compared (the facade's padding)
EXTEND = 60


def _same(a, b) -> list[str]:
    la, lb = ni._leaves(a), ni._leaves(b)
    assert la.keys() == lb.keys()
    return [k for k in la if la[k].dtype != lb[k].dtype or not np.array_equal(la[k], lb[k], equal_nan=True)]


@pytest.fixture(scope="module", params=["small", pytest.param("all", marks=pytest.mark.slow)])
def pairs(request: pytest.FixtureRequest, data_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file() or not ni.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    if not (data_dir / h.CATPA_CASE).is_dir():
        pytest.skip(f"{data_dir / h.CATPA_CASE} not found")
    keys = ni.SMALL if request.param == "small" else ni._keys(data_dir)
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tmp_path_factory.mktemp("free"))
    outs = h.run_references(keys, root / f"df_{request.param}", data_dir, jobs)
    out: dict[str, tuple[FreeRunInputs, FreeRunInputs]] = {}
    refused: dict[str, str] = {}
    for exp, t in keys:
        key = h.key_of(exp, t)
        pth = ni._paths(exp, data_dir)
        kw = {k: v for k, v in pth.items() if k != "filex"}
        try:
            ref = free_run_inputs(pth["filex"], t, outs[key], source="native", extend_days=EXTEND, **kw)
        except NativeInputError as e:
            refused[key] = str(e)
            with pytest.raises(NativeInputError):
                free_run_inputs(pth["filex"], t, None, source="native", extend_days=EXTEND, **kw)
            continue
        free = free_run_inputs(pth["filex"], t, None, source="native", extend_days=EXTEND, **kw)
        out[key] = (free, ref)
    return {"keys": keys, "pairs": out, "refused": refused}


def test_dssat_free_inputs_equal_the_reference_run_inputs(pairs) -> None:
    if len(pairs["keys"]) == 65:
        assert len(pairs["pairs"]) == ni.N_NATIVE and set(pairs["refused"]) == set(ni.REFUSED)
    for key, (free, ref) in pairs["pairs"].items():
        np.testing.assert_array_equal(free.days, ref.days, err_msg=f"{key}: season days")
        assert free.n_ext == ref.n_ext, (key, free.n_ext, ref.n_ext)
        assert free.cultivar == ref.cultivar and free.published() == ref.published(), key
        assert _same(free.params(), ref.params()) == [], (key, "params", _same(free.params(), ref.params()))
        assert _same(free.state(), ref.state()) == [], (key, "state")
        n = free.n_days + EXTEND + 10  # the season, the extension and some padding
        assert _same(free.forcing(n), ref.forcing(n)) == [], (
            key,
            "forcing",
            _same(free.forcing(n), ref.forcing(n)),
        )
        assert free.out is None and free.notes["dssat_free"]


def test_the_season_is_dssat_s(pairs) -> None:
    for key, (free, ref) in pairs["pairs"].items():
        s, r = free.summary, ref.summary
        for col in ("EDAT", "ADAT", "MDAT"):
            a, b = s[col], r[col]
            assert (a if np.isfinite(a) else -99) == (b if b > 0 else -99), (key, col, a, b)
        assert abs(s["HWAM"] - r["HWAM"]) <= h.YIELD_REL * r["HWAM"], (key, s["HWAM"], r["HWAM"])
    print(f"{len(pairs['pairs'])} runs: DSSAT-free inputs equal the reference-run inputs leaf by leaf")
