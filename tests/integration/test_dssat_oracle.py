"""DSSAT-CSM oracle: run the official Maize example UFGA8201 with the locally built
``dscsm048`` (v4.8.6.0, ``source/build486``) and check the parsed outputs.

The run is the session fixture ``ufga_run`` of ``tests/integration/conftest.py``, staged in a
private temporary directory (shared with ``test_io_dssat_data.py``, never with other processes). Paths: ``AGRI_JAX_DSSAT`` (engine root, default
``~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine``) and ``AGRI_JAX_DSCSM`` (binary, default
``<engine>/source/build486/bin/dscsm048``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from agri_jax.io.dssat import (
    read_out,
    read_plantgro,
    read_sol,
    read_summary,
    read_wth,
    run_dssat,
    write_sol,
    write_wth,
)

UFGA_FILEX = Path("UFGA8201.MZX")

pytestmark = pytest.mark.slow


def test_ufga8201_summary_hwam(ufga_run: Path) -> None:
    s = read_summary(ufga_run / "Summary.OUT")
    assert list(s["TRNO"]) == [1, 2, 3, 4, 5, 6]
    assert set(s["MODEL"]) == {"MZCER048"} and set(s["SOIL_ID"]) == {"IBMZ910014"}
    hwam = s.set_index("TRNO")["HWAM"]
    assert hwam.dtype.kind == "i"
    assert ((hwam > 500) & (hwam < 20000)).all(), hwam.to_dict()
    # irrigated + high N out-yields rainfed + low N
    assert hwam[4] > hwam[3] > hwam[1] and hwam[4] > 2 * hwam[2]

    obs = read_out(ufga_run / "UFGA8201.MZA").set_index("TRNO")["HWAM"]
    ratio = (hwam / obs).to_numpy()
    assert np.all((ratio > 0.6) & (ratio < 1.5)), dict(zip(obs.index, ratio, strict=True))
    assert float(np.mean(np.abs(ratio - 1.0))) < 0.2
    assert abs(hwam[4] / obs[4] - 1.0) < 0.05  # the fully-fertilised irrigated treatment


def test_ufga8201_plantgro_consistent(ufga_run: Path) -> None:
    s = read_summary(ufga_run / "Summary.OUT").set_index("TRNO")
    g = read_plantgro(ufga_run / "PlantGro.OUT")
    final = g.groupby("TRNO").tail(1).set_index("TRNO")
    np.testing.assert_array_equal(final["GWAD"].to_numpy(), s["HWAM"].to_numpy())
    np.testing.assert_array_equal(final["CWAD"].to_numpy(), s["CWAM"].to_numpy())


def test_writers_feed_the_binary(
    tmp_path_factory: pytest.TempPathFactory, stage_ufga8201: Callable[[Path], Path], ufga_run: Path
) -> None:
    """Re-write the weather and soil files with write_wth / write_sol and rerun: the
    simulated yields must not change."""
    run_dir = stage_ufga8201(tmp_path_factory.mktemp("ufga8201_rw"))
    wth = run_dir / "UFGA8201.WTH"
    write_wth(read_wth(wth), wth)
    sol_path = run_dir / "SOIL.SOL"
    prof = read_sol(sol_path)["IBMZ910014"]
    for f in run_dir.glob("*.SOL"):
        f.unlink()
    write_sol([prof], sol_path)
    r = run_dssat(run_dir, UFGA_FILEX.name)
    assert r.returncode == 0, r.stdout[-2000:]
    a = read_summary(ufga_run / "Summary.OUT")
    b = read_summary(run_dir / "Summary.OUT")
    for col in ("HWAM", "CWAM", "MDAT", "ADAT", "ETCM", "PRCM", "IRCM"):
        np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy(), err_msg=col)
