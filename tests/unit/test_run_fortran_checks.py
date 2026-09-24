"""``check_rzwqm_outputs`` and the run-dir length guard, on synthetic logs and ``.ana`` files.

The real-binary counterparts are in ``tests/integration/test_rzwqm_all_scenarios.py``
(``test_truncated_run_is_detected`` forces a truncated CA-MA1 run; 15 scenarios pass the check).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from agri_jax.port.run_fortran import (
    MAX_RZWQM_RUN_DIR_LEN,
    FortranRunError,
    _make_run_dir,
    check_rzwqm_outputs,
)


def _ana(path: Path, start: dt.date, end: dt.date) -> Path:
    lines = ["  1) TIME (YEAR.DAY)   2) STORED SOIL WATER (CM)", f" {start.year}.000   30.0"]
    d = start
    while d <= end:
        lines.append(f" {d.year}.{d.timetuple().tm_yday:03d}   30.0")
        d += dt.timedelta(days=1)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_complete_run_passes(tmp_path: Path) -> None:
    s, e = dt.date(2015, 1, 1), dt.date(2016, 12, 31)  # a leap year inside
    log = tmp_path / "run.log"
    log.write_text("RZWQM2 finished\n")
    check_rzwqm_outputs(log, _ana(tmp_path / "x.ana", s, e), s, e)


@pytest.mark.parametrize("marker", ["Program will have to stop", " Could not find input file! WHCER040.EC"])
def test_stop_marker_raises(tmp_path: Path, marker: str) -> None:
    s, e = dt.date(2015, 1, 1), dt.date(2015, 12, 31)
    log = tmp_path / "run.log"
    log.write_text(f"reading ...\n{marker}\n")
    with pytest.raises(FortranRunError, match=r"run\.log reports"):
        check_rzwqm_outputs(log, _ana(tmp_path / "x.ana", s, e), s, e)


def test_truncated_ana_raises(tmp_path: Path) -> None:
    s, e = dt.date(2015, 1, 1), dt.date(2015, 12, 31)
    log = tmp_path / "run.log"
    log.write_text("ok\n")
    ana = _ana(tmp_path / "x.ana", s, dt.date(2015, 6, 30))
    with pytest.raises(FortranRunError, match="truncated run"):
        check_rzwqm_outputs(log, ana, s, e)
    empty = tmp_path / "e.ana"
    empty.write_text("")
    with pytest.raises(FortranRunError, match="missing/empty"):
        check_rzwqm_outputs(log, empty, s, e)


def test_run_dir_length_guard(tmp_path: Path) -> None:
    assert MAX_RZWQM_RUN_DIR_LEN == 64 - len("/DSSAT/") - len("WHCER040.ECO") == 45
    deep = tmp_path / ("d" * 60)
    with pytest.raises(FortranRunError, match="characters"):
        _make_run_dir("r", deep, max_len=MAX_RZWQM_RUN_DIR_LEN)
    assert not any(deep.iterdir())  # the rejected directory is removed
    ok = _make_run_dir("r", deep, max_len=None)
    assert ok.is_dir()
