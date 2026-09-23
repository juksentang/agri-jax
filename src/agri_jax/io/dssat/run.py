"""Stage and run the DSSAT-CSM binary (``dscsm048``) on an experiment file.

The staging pattern mirrors ``setup_run_dir`` of the AFSoil batch runner: every file the
binary needs (``*.CDE``, ``DSSATPRO.*``, ``DSCSM048.CTR``, ``MODEL.ERR``, StandardData,
Genotype and Pest files, the experiment, weather and soil files) is copied flat into one run
directory, ``DSSATPRO.v48`` is rewritten to point at that directory, and the binary is run
there as ``./dscsm048 <MODEL> A <FILEX>``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DssatRun", "run_dssat", "stage_run_dir"]

#: model code by experiment-file extension
MODEL_BY_EXT = {
    ".MZX": "MZCER048",
    ".SGX": "SGCER048",
    ".MLX": "MLCER048",
    ".WHX": "CSCER048",
    ".SBX": "CRGRO048",
}


@dataclass(frozen=True)
class DssatRun:
    """Result of one ``dscsm048`` invocation."""

    run_dir: Path
    returncode: int
    stdout: str
    stderr: str

    def out(self, name: str) -> Path:
        """Path of an output file (e.g. ``"Summary.OUT"``) in the run directory."""
        return self.run_dir / name


def _copy_flat(files: Iterable[Path], dst_dir: Path, *, overwrite: bool = False) -> None:
    for f in files:
        if not f.is_file():
            continue
        dst = dst_dir / f.name
        if overwrite or not dst.exists():
            shutil.copy2(f, dst)


def _soil_ids(filex_text: str) -> list[str]:
    """ID_SOIL values of the *FIELDS section (10-character ids)."""
    ids: list[str] = []
    in_fields = False
    col: int | None = None
    for line in filex_text.splitlines():
        if line.startswith("*"):
            in_fields = line.upper().startswith("*FIELDS")
            col = None
            continue
        if not in_fields:
            continue
        if line.startswith("@"):
            col = line.find("ID_SOIL") if "ID_SOIL" in line else None
            continue
        if col is not None and line.strip():
            sid = line[col : col + 10].strip()
            if sid and sid != "-99":
                ids.append(sid)
    return ids


def stage_run_dir(
    filex: Path,
    run_dir: Path,
    *,
    data_dir: Path,
    weather_dirs: Iterable[Path] = (),
    soil_dirs: Iterable[Path] = (),
    binary: Path | None = None,
    extra_files: Iterable[Path] = (),
) -> Path:
    """Assemble a complete, self-contained DSSAT run directory.

    Parameters
    ----------
    filex:
        Experiment file (``*.MZX`` ...). Its sibling files with the same stem (``.MZA``,
        ``.MZT`` observed data) are copied too.
    run_dir:
        Directory to create/populate.
    data_dir:
        DSSAT ``Data`` directory of the source tree (``*.CDE``, ``DSSATPRO.*``,
        ``StandardData/``, ``Genotype/``, ``Pest/``).
    weather_dirs:
        Directories searched for ``<WSTA><YY>01.WTH`` style files; every ``.WTH`` whose
        name starts with a station code of the experiment is copied.
    soil_dirs:
        Directories whose ``*.SOL`` files are searched for the experiment's ``ID_SOIL``;
        the matching files are copied (under their own names) and, when there is exactly
        one, also as ``SOIL.SOL``.
    binary:
        ``dscsm048`` executable to copy into the run dir (optional).
    """
    filex = Path(filex)
    run_dir = Path(run_dir)
    data_dir = Path(data_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # experiment + observed files
    _copy_flat(filex.parent.glob(filex.stem + ".*"), run_dir, overwrite=True)

    # engine files
    engine = [*data_dir.glob("*.CDE")] + [
        data_dir / n for n in ("DSCSM048.CTR", "MODEL.ERR", "DSSATPRO.v48", "DSSATPRO.L48", "DSSATPRO.L48.in")
    ]
    _copy_flat(engine, run_dir, overwrite=True)
    for sub in ("StandardData", "Genotype", "Pest"):
        d = data_dir / sub
        if d.is_dir():
            _copy_flat(sorted(d.iterdir()), run_dir, overwrite=True)

    text = filex.read_text(errors="replace")

    # weather: station codes from *FIELDS WSTA column (4 characters)
    stations: set[str] = set()
    in_fields = False
    wcol: int | None = None
    for line in text.splitlines():
        if line.startswith("*"):
            in_fields = line.upper().startswith("*FIELDS")
            wcol = None
            continue
        if in_fields and line.startswith("@"):
            wcol = line.find("WSTA") if "WSTA" in line else None
            continue
        if in_fields and wcol is not None and line.strip():
            code = line[wcol : wcol + 8].strip()[:4]
            if code and code != "-99":
                stations.add(code.upper())
    for d in weather_dirs:
        for f in sorted(Path(d).glob("*.WTH")):
            if f.name[:4].upper() in stations:
                _copy_flat([f], run_dir, overwrite=True)

    # soil
    wanted = _soil_ids(text)
    matched: list[Path] = []
    for d in soil_dirs:
        for f in sorted(Path(d).glob("*.SOL")):
            body = f.read_text(errors="replace")
            if any(f"*{sid}" in body for sid in wanted):
                matched.append(f)
    _copy_flat(matched, run_dir, overwrite=True)
    if len(matched) == 1 and matched[0].name.upper() != "SOIL.SOL":
        shutil.copy2(matched[0], run_dir / "SOIL.SOL")

    _copy_flat([Path(p) for p in extra_files], run_dir, overwrite=True)

    # DSSATPRO paths -> the run dir
    pro = run_dir / "DSSATPRO.v48"
    if pro.exists():
        t = pro.read_text()
        for pat in ("C: \\DSSAT48", "C:\\DSSAT48"):
            t = t.replace(pat, str(run_dir))
        pro.write_text(t.replace("\\", "/"))

    if binary is not None:
        dst = run_dir / "dscsm048"
        shutil.copy2(binary, dst)
        dst.chmod(dst.stat().st_mode | 0o111)
    return run_dir


def run_dssat(
    run_dir: Path,
    filex_name: str,
    *,
    model: str | None = None,
    binary: Path | None = None,
    mode: str = "A",
    timeout: float = 300.0,
) -> DssatRun:
    """Run ``dscsm048 <model> <mode> <filex_name>`` inside ``run_dir``.

    ``binary`` defaults to ``<run_dir>/dscsm048``; ``model`` defaults from the file
    extension (``.MZX`` -> ``MZCER048``).
    """
    run_dir = Path(run_dir)
    exe = Path(binary) if binary is not None else run_dir / "dscsm048"
    if model is None:
        model = MODEL_BY_EXT.get(Path(filex_name).suffix.upper(), "")
    args = [str(exe.resolve()), *([model] if model else []), mode, filex_name]
    proc = subprocess.run(
        args,
        cwd=run_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
        check=False,
    )
    return DssatRun(run_dir, proc.returncode, proc.stdout, proc.stderr)
