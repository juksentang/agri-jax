"""Oracle runner: RZWQM2 on CA-TPA and DSSAT-CSM on UFGA8201 via agrijax.port.run_fortran."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from agrijax.port.run_fortran import (
    DSSAT_ENGINE,
    RZWQM_BINARY,
    elf_interpreter,
    parse_overview_yields,
    patch_ipnames,
    patch_rzx_database,
    run_dscsm,
    run_rzwqm,
)

_ROW = re.compile(r"^\s*\d{4}\.\d{3}\s")


def _data_rows(ana: Path) -> list[str]:
    return [ln for ln in ana.read_text(errors="replace").splitlines() if _ROW.match(ln)]


@pytest.fixture(scope="module")
def rz_binary() -> Path:
    if not RZWQM_BINARY.is_file():
        pytest.skip(f"RZWQM binary not found at {RZWQM_BINARY}")
    return RZWQM_BINARY


def test_patch_ipnames_paths_and_dates(catpa_scenario: Path) -> None:
    text = (catpa_scenario / "IPNAMES.DAT").read_text(errors="replace")
    out = patch_ipnames(text, Path("/x/run/rz_1"), start="2015-01-01", end="2015-12-31").splitlines()
    assert out[2] == "/x/run/rz_1/CA-TPA.MET"
    assert out[3] == "/x/run/rz_1/CA-TPA.BRK"
    assert out[6] == "/x/run/rz_1/CA-TPA.sno"
    assert out[7] == "/x/run/rz_1/CA-TPA.ana"
    assert out[8].split() == ["1", "1", "2015", "31", "12", "2015"]
    assert out[9:] == text.splitlines()[9:]
    with pytest.raises(ValueError, match="longer than"):
        patch_ipnames(text, Path("/" + "a" * 80))


def test_patch_rzx_database(catpa_scenario: Path) -> None:
    text = (catpa_scenario / "MZDSSAT.RZX").read_text(errors="replace")
    out = patch_rzx_database(text, Path("/x/run/rz_1")).splitlines()
    h = next(i for i, ln in enumerate(out) if ln.startswith("= DATABASE FILE LOCATIONS"))
    assert out[h + 2 : h + 4] == ["/x/run/rz_1/DSSAT/", "/x/run/rz_1/"]
    assert len(out) == len(text.splitlines())


def test_elf_interpreter(rz_binary: Path) -> None:
    interp = elf_interpreter(rz_binary)
    assert interp is not None and interp.endswith("ld-linux-x86-64.so.2")
    dscsm = DSSAT_ENGINE / "bin" / "dscsm048"
    if dscsm.is_file():
        assert elf_interpreter(dscsm) is None  # static


def test_rzwqm_one_year(catpa_scenario: Path, rz_binary: Path, tmp_path: Path) -> None:
    r = run_rzwqm(catpa_scenario, tmp_path / "out", start="2015-01-01", end="2015-12-31", timeout=120)
    assert r.ana_path.is_file() and r.run_log.is_file()
    assert r.overview_path is not None and r.overview_path.is_file()
    rows = _data_rows(r.ana_path)
    assert len(rows) == 365 + 1  # day 0 (initial state) + 365 days
    assert rows[0].split()[0] == "2015.000" and rows[-1].split()[0] == "2015.365"
    # 139 whitespace-separated fields per row: YYYY.DDD + 138 variables
    assert all(len(ln.split()) == 139 for ln in rows)
    assert len(parse_overview_yields(r.overview_path)) == 1
    assert r.elapsed_s < 60


def test_rzwqm_failure_raises_with_log(catpa_scenario: Path, rz_binary: Path, tmp_path: Path) -> None:
    from agrijax.port.run_fortran import FortranRunError

    bad = tmp_path / "bad.dat"
    bad.write_text("garbage\n")
    with pytest.raises(FortranRunError) as ei:
        run_rzwqm(
            catpa_scenario,
            tmp_path / "out",
            dat_override=bad,
            start="2015-01-01",
            end="2015-01-31",
            timeout=60,
        )
    run_dir = ei.value.run_dir
    assert run_dir is not None and run_dir.is_dir()
    shutil.rmtree(run_dir)


@pytest.mark.slow
def test_rzwqm_full_catpa_yields(catpa_scenario: Path, rz_binary: Path, tmp_path: Path) -> None:
    r = run_rzwqm(catpa_scenario, tmp_path / "out")
    assert r.overview_path is not None
    yields = parse_overview_yields(r.overview_path)
    assert yields[:4] == [9916.0, 9608.0, 5996.0, 10444.0]
    assert len(_data_rows(r.ana_path)) == 1 + sum(366 if y % 4 == 0 else 365 for y in range(2015, 2024))


def test_dscsm_ufga8201(tmp_path: Path) -> None:
    src = DSSAT_ENGINE / "example_data" / "Maize"
    if not (DSSAT_ENGINE / "bin" / "dscsm048").is_file() or not (src / "UFGA8201.MZX").is_file():
        pytest.skip("DSSAT engine / UFGA8201 example not found")
    exp = tmp_path / "exp"
    exp.mkdir()
    for f in src.glob("UFGA8201.MZ*"):
        (exp / f.name).write_bytes(f.read_bytes())
    r = run_dscsm(exp, tmp_path / "out", experiment_file="UFGA8201.MZX")
    assert r.summary_path is not None
    runs = [ln for ln in r.summary_path.read_text().splitlines() if "MZCER048" in ln]
    assert len(runs) >= 1


def _ufga8201_exp(tmp_path: Path) -> Path:
    src = DSSAT_ENGINE / "example_data" / "Maize"
    if not (DSSAT_ENGINE / "bin" / "dscsm048").is_file() or not (src / "UFGA8201.MZX").is_file():
        pytest.skip("DSSAT engine / UFGA8201 example not found")
    exp = tmp_path / "exp"
    exp.mkdir()
    for f in src.glob("UFGA8201.MZ*"):
        (exp / f.name).write_bytes(f.read_bytes())
    return exp


def test_dscsm_truncated_weather_is_detected(tmp_path: Path) -> None:
    """Weather ending on 1982-119 cuts every season short; dscsm048 still exits 0 with 6 rows.

    ``Weather/IPWTH_alt.for`` ``WeatherError`` writes "Simulation will end." to WARNING.OUT and the
    run continues (run mode A); :func:`check_dscsm_outputs` must reject it, ``check=False`` not.
    """
    from agrijax.port.run_fortran import FortranRunError

    exp = _ufga8201_exp(tmp_path)
    wth = DSSAT_ENGINE / "example_data" / "Weather" / "UFGA8201.WTH"
    if not wth.is_file():
        pytest.skip("UFGA8201.WTH not found")
    keep = [ln for ln in wth.read_text().splitlines() if not re.match(r"^82(1[2-9]\d|[23]\d\d)\s", ln)]
    (exp / "UFGA8201.WTH").write_text("\n".join(keep) + "\n")  # exp_dir files override the engine's
    with pytest.raises(FortranRunError, match="simulation will end") as ei:
        run_dscsm(exp, tmp_path / "out", experiment_file="UFGA8201.MZX")
    assert ei.value.run_dir is not None
    shutil.rmtree(ei.value.run_dir)
    r = run_dscsm(exp, tmp_path / "out2", experiment_file="UFGA8201.MZX", check=False)
    assert r.summary_path is not None


def test_dscsm_stale_outputs_not_staged(tmp_path: Path) -> None:
    """``*.OUT`` left in ``exp_dir`` by an earlier run must not reach the run dir."""
    exp = _ufga8201_exp(tmp_path)
    (exp / "ERROR.OUT").write_text("*RUN-TIME ERRORS OUTPUT FILE\n stale\n")
    (exp / "Summary.OUT").write_text("stale\n")
    r = run_dscsm(exp, tmp_path / "out", experiment_file="UFGA8201.MZX")
    assert r.summary_path is not None and "stale" not in r.summary_path.read_text()
    assert not (tmp_path / "out" / "ERROR.OUT").exists()
