"""``check_rzwqm_outputs``, ``check_dscsm_outputs`` and the run-dir length guard, on synthetic files.

Fault injection: each failure signature the Fortran can leave behind while exiting 0 is written into
a synthetic run directory and must raise; a clean run (with warnings and a crop failure) must pass.

The real-binary counterparts are in ``tests/integration/test_rzwqm_all_scenarios.py``
(``test_truncated_run_is_detected`` forces a truncated CA-MA1 run; 15 scenarios pass the check).
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from agrijax.port.run_fortran import (
    MAX_RZWQM_RUN_DIR_LEN,
    FortranRunError,
    _filex_treatments,
    _make_run_dir,
    _summary_trnos,
    check_dscsm_outputs,
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


@pytest.mark.parametrize(
    "marker",
    [
        ">>>> END OF FILE REACHED IN DAYMET.DAT <<<<",  # Rzmain.for weather reader, then STOP
        " >>> FATAL ERROR READING BRKPNT.DAT <<<",  # Rzday.for, then STOP
        " -- ERROR -- ERROR -- UNABLE TO OPEN FILE RZWQM.DAT --",  # Rzmain.for IPNAMES opener
        " <<< ERROR IN DATES >>> DATES NOT SEQ",  # Rzmain.for period check, then STOP
    ],
)
def test_rzwqm_fatal_markers_raise(tmp_path: Path, marker: str) -> None:
    s, e = dt.date(2015, 1, 1), dt.date(2015, 12, 31)
    log = tmp_path / "run.log"
    log.write_text(f"reading ...\n{marker}\n     PROGRAM TERMINATED\n")
    with pytest.raises(FortranRunError, match=r"run\.log reports"):
        check_rzwqm_outputs(log, _ana(tmp_path / "x.ana", s, e), s, e)


def test_rzwqm_normal_end_stop_text_passes(tmp_path: Path) -> None:
    """A finished RZWQM2 run ends in ``STOP 'check your expdata.dat file (daily data)'``."""
    s, e = dt.date(2015, 1, 1), dt.date(2015, 3, 31)
    log = tmp_path / "run.log"
    log.write_text(
        "  >>>> END OF BREAK-POINT DATA ENCOUNTERED\n ==> UPDATING NUTRIENT CHEMISTRY\n"
        "check your expdata.dat file (daily data)\n"
    )
    check_rzwqm_outputs(log, _ana(tmp_path / "x.ana", s, e), s, e)


# --------------------------------------------------------------------------- DSSAT-CSM

_FILEX = """*EXP.DETAILS: TEST0001MZ SYNTHETIC

*TREATMENTS                        -------------FACTOR LEVELS------------
@N R O C TNAME.................... CU FL SA IC MP MI MF MR MC MT ME MH SM
 1 1 0 0 LOW N                      1  1  0  1  1  1  1  0  0  0  0  0  1
! a comment line
 2 1 0 0 HIGH N                     1  1  0  1  1  1  2  0  0  0  0  0  1

 3 1 0 0 IRRIGATED                  1  1  0  1  1  2  1  0  0  0  0  0  1

*CULTIVARS
@C CR INGENO CNAME
 1 MZ IB0035 McCurdy 84aa
"""

_SUMMARY_HEAD = (
    "*SUMMARY : TEST0001MZ SYNTHETIC\n\n"
    "!IDENTIFIERS......................... EXPERIMENT AND TREATMENT..........\n"
    "@   RUNNO   TRNO R# O# P# CR MODEL... EXNAME.. TNAM..................... FNAM....   HWAM\n"
)


def _summary_row(run: int, trno: int) -> str:
    return f"{run:9d}{trno:7d}  1  0  1 MZ MZCER048 TEST0001 LOW N                     TEST0001   5000\n"


def _dscsm_run_dir(tmp_path: Path, trnos: tuple[int, ...] = (1, 2, 3)) -> Path:
    rd = tmp_path / "ds"
    rd.mkdir()
    (rd / "TEST0001.MZX").write_text(_FILEX)
    (rd / "run.log").write_text("RUN    TRT FLO MAT TOPWT HARWT\n  1 MZ   1  76 128  6556  2293\n")
    (rd / "WARNING.OUT").write_text(
        "*WARNING DETAIL FILE\n\n IPWTH   YEAR DOY = 1982  98\n Warning: SRAD < 1 MJ.m-2.d-1.\n"
        " MZ_GROSUB\n Crop experienced 10 days below 6.0C\n Growth program terminated.\n"
    )
    (rd / "Summary.OUT").write_text(
        _SUMMARY_HEAD + "".join(_summary_row(i + 1, t) for i, t in enumerate(trnos))
    )
    return rd


def test_filex_treatments_and_summary_trnos(tmp_path: Path) -> None:
    rd = _dscsm_run_dir(tmp_path)
    assert _filex_treatments(rd / "TEST0001.MZX") == [1, 2, 3]
    assert _summary_trnos(rd / "Summary.OUT") == [1, 2, 3]


def test_dscsm_complete_run_passes(tmp_path: Path) -> None:
    """Warnings and a crop failure ("Growth program terminated.") are model outcomes, not stops."""
    rd = _dscsm_run_dir(tmp_path)
    check_dscsm_outputs(rd, experiment_file="TEST0001.MZX")


def test_dscsm_missing_treatment_raises(tmp_path: Path) -> None:
    """A mid-season STOP in treatment 3 leaves its Summary.OUT row out (row written at season end)."""
    rd = _dscsm_run_dir(tmp_path, trnos=(1, 2))
    with pytest.raises(FortranRunError, match=r"missing \[3\]"):
        check_dscsm_outputs(rd, experiment_file="TEST0001.MZX")
    check_dscsm_outputs(rd, experiment_file="TEST0001.MZX", run_mode="B")  # no coverage check


def test_dscsm_empty_or_missing_summary_raises(tmp_path: Path) -> None:
    rd = _dscsm_run_dir(tmp_path, trnos=())
    with pytest.raises(FortranRunError, match="no run rows"):
        check_dscsm_outputs(rd)
    (rd / "Summary.OUT").unlink()
    with pytest.raises(FortranRunError, match=r"no Summary\.OUT"):
        check_dscsm_outputs(rd)


def test_dscsm_error_out_raises(tmp_path: Path) -> None:
    rd = _dscsm_run_dir(tmp_path)
    (rd / "ERROR.OUT").write_text("*RUN-TIME ERRORS OUTPUT FILE\n\n Problem with configuration file!\n")
    with pytest.raises(FortranRunError, match=r"ERROR\.OUT"):
        check_dscsm_outputs(rd, experiment_file="TEST0001.MZX")


def test_dscsm_gfortran_stop_line_raises(tmp_path: Path) -> None:
    """gfortran prints ``STOP  `` for ``STOP ' '`` (Cropsim/CSCAS readers, ``Utilities/CSUTS.for``)."""
    rd = _dscsm_run_dir(tmp_path)
    (rd / "run.log").write_text(" Problem with configuration file!\nSTOP  \n")
    with pytest.raises(FortranRunError, match="Fortran STOP"):
        check_dscsm_outputs(rd, experiment_file="TEST0001.MZX")


@pytest.mark.parametrize(
    ("where", "text"),
    [
        # Weather/IPWTH_alt.for WeatherError: season cut short, run continues, exit 0
        ("WARNING.OUT", " IPWTH   YEAR DOY = 1982 120\n Weather record not found\n Simulation will end.\n"),
        ("WARNING.OUT", " ENDRUN\n Simulation ended with error code  10\n"),  # CSM_Main/LAND.for
        ("WARNING.OUT", " DEMAND\n Run       3 will be terminated.\n"),  # Utilities/ERROR.for ErrorCode
        ("WARNING.OUT", " RADABS\n Error in RADHR or PARHR for hour 12.\n Program will stop.\n"),
        ("run.log", "*** UNKNOWN SOIL TYPE ***"),  # Soil/SoilUtilities/TextureClass.for, bare STOP
        ("run.log", " More than NL layers in soil profile :  21\n Please fix soil profile.\n"),
        ("run.log", "  Program will have to stop\n  Check WORK.OUT for details of run\n"),
    ],
)
def test_dscsm_stop_markers_raise(tmp_path: Path, where: str, text: str) -> None:
    rd = _dscsm_run_dir(tmp_path)
    with open(rd / where, "a") as fh:
        fh.write(text)
    with pytest.raises(FortranRunError, match=re.escape(where) + " reports"):
        check_dscsm_outputs(rd, experiment_file="TEST0001.MZX")
