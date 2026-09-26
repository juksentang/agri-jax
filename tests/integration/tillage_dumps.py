"""Pre-tillage (start-up) hydraulic parameters of an instrumented RZWQM2 run, read from its dumps.

RZWQM2 saves the start-up ``SOILHP`` of every horizon in ``TRHYDP`` (``WC``, ``SPMOIS``, ``WCH``;
filled by the ``ISTAT = -1`` calls of ``TILADJ``) and the start-up ``C2``/``eps`` in ``C22``/``SN22``
(``POINTK``, first call per horizon). The instrumented ``WC`` and ``POINTK`` dumps carry these
saved arrays, so the pre-tillage curve of a run is known without re-deriving it from ``rzwqm.dat``.
Every ``WC`` dump of the run must hold the same ``TRHYDP`` (it is written only at start-up).

Helper module of the integration tier (no tests here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run_of_case(path: Path) -> str:
    """Run prefix of a dump case file name, ``catpa2015_d2015008_c...`` -> ``catpa2015``."""
    return path.name.split("_d", 1)[0]


def saved_trhydp(data_dir: Path, run: str) -> np.ndarray | None:
    """``TRHYDP(13, MAXHOR)`` of ``run`` from its ``WC`` dumps; ``None`` when there is none.

    Raises ``AssertionError`` when two dumps of the run disagree (the saved copy must be constant).
    """
    from agrijax.port.dumps import load_case

    files = sorted((data_dir / "dumps" / "WC").glob(f"{run}_d*.npz"))
    if not files:
        return None
    arrs = [np.asarray(load_case(f).entry["TRHYDP"], float) for f in files]
    for f, a in zip(files, arrs, strict=True):
        assert np.array_equal(a, arrs[0]), f"TRHYDP differs in {f.name}"
    return arrs[0]


def saved_c22_sn22(data_dir: Path, run: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(C22, SN22)`` [MAXHOR] of ``run`` from its ``POINTK`` dumps; ``None`` when there is none."""
    from agrijax.port.dumps import load_case

    files = sorted((data_dir / "dumps" / "POINTK").glob(f"{run}_d*.npz"))
    if not files:
        return None
    cases = [load_case(f).entry for f in files]
    c22 = np.asarray(cases[0]["C22"], float)
    sn22 = np.asarray(cases[0]["SN22"], float)
    for f, e in zip(files, cases, strict=True):
        assert np.array_equal(np.asarray(e["C22"], float), c22), f"C22 differs in {f.name}"
        assert np.array_equal(np.asarray(e["SN22"], float), sn22), f"SN22 differs in {f.name}"
    return c22, sn22
