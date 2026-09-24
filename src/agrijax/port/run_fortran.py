"""Run the Fortran oracles (RZWQM2 and DSSAT-CSM) from Python.

This is step 3 ("Collect") of the porting procedure.
It implements the verified local recipes without any shell script:

RZWQM2 (:func:`run_rzwqm`)
    1. Stage the scenario directory plus ``RZWQM_Tool/DSSAT/*`` into a run directory
       ``<run_root>/rXXXXXXXX`` whose absolute path is at most :data:`MAX_RZWQM_RUN_DIR_LEN`
       (45) characters: the Cropsim-CERES ecotype path ``<rundir>/DSSAT/WHCER040.ECO`` must fit
       ``CHARACTER*64``. Staged path records are also kept below 80 characters (a conservative
       limit: RZWQM2 reads ``IPNAMES.DAT`` and the ``*.RZX`` database paths as ``A255``).
    2. Rewrite lines 1-8 of ``IPNAMES.DAT`` to absolute paths inside the run directory and,
       optionally, the simulation date line (line 9, ``DD MM YYYY DD MM YYYY``).
    3. Rewrite the two paths of the ``= DATABASE FILE LOCATIONS`` block of every ``*.RZX``
       to ``<rundir>/DSSAT/`` and ``<rundir>/``.
    4. Copy ``DSSAT/MODEL.ERR`` to ``<rundir>/main_ryzenMODEL.ERR`` (the binary looks it up
       by a name derived from its truncated ``argv[0]``).
    5. Execute the binary, through ``/lib64/ld-linux-x86-64.so.2`` when its ``PT_INTERP``
       (the cvmfs loader of the Alliance clusters) does not exist on this machine, with
       ``FORT_BUFFERED=TRUE``; stdout/stderr go to ``run.log``.
    6. :func:`check_rzwqm_outputs`: the binary exits 0 after a Fortran ``STOP``, so the run
       fails unless ``run.log`` has no fatal-error marker and the ``.ana`` covers the whole
       period (the row count is the complete guard; most STOPs print nothing distinctive).
    7. Copy the kept files (``*.ana``, ``OVERVIEW.OUT`` and ``run.log``) to ``out_dir``.

DSSAT-CSM (:func:`run_dscsm`)
    The run-dir staging pattern of ``AFSoil/.../0205_Run_DSSAT/01_run_all.py``
    (``setup_run_dir``): experiment files, engine ``*.CDE``/``DSCSM048.CTR``/``MODEL.ERR``/
    ``DSSATPRO.*``, ``StandardData`` and ``Genotype``, soil and weather files, and the
    ``DSSATPRO.v48`` paths rewritten from ``C:\\DSSAT48`` to the run directory. ``*.OUT`` files of
    ``exp_dir`` are not staged. :func:`check_dscsm_outputs` then rejects runs that exited 0 but
    stopped or ended a season early (``STOP`` / ``ERROR.OUT`` / WARNING.OUT termination messages /
    treatments missing from ``Summary.OUT``).

CLI::

    python -m agrijax.port.run_fortran rzwqm <scenario> <out> [--start 2015-01-01 --end 2015-12-31]
    python -m agrijax.port.run_fortran dscsm <exp_dir> <out> [--model MZCER048 --experiment UFGA8201.MZX]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DSCSM_STOP_MARKERS",
    "DSSAT_ENGINE",
    "MAX_RZWQM_RUN_DIR_LEN",
    "RUN_ROOT",
    "RZWQM_STOP_MARKERS",
    "RZWQM_TOOL",
    "DscsmResult",
    "FortranRunError",
    "RzwqmResult",
    "check_dscsm_outputs",
    "check_rzwqm_outputs",
    "elf_interpreter",
    "parse_overview_yields",
    "patch_ipnames",
    "patch_rzx_database",
    "run_dscsm",
    "run_rzwqm",
]

DATA_ROOT = Path(os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data"))).expanduser()
RUN_ROOT = Path(os.environ.get("AGRI_JAX_RUN_ROOT", str(DATA_ROOT / "run")))
RZWQM_TOOL = DATA_ROOT / "narval_mirror" / "RZWQM_Tool"
RZWQM_BINARY = RZWQM_TOOL / "main_ryzen5_avx512"
#: DSSAT-CSM engine root; ``AGRI_JAX_DSSAT`` (the name the tests use) or ``AGRI_JAX_DSSAT_ENGINE``.
DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT")
    or os.environ.get(
        "AGRI_JAX_DSSAT_ENGINE",
        str(Path.home() / "AFSoil" / "Formal_Analysis" / "02_DSSAT" / "dssat_engine"),
    )
).expanduser()
SYSTEM_LOADER = Path("/lib64/ld-linux-x86-64.so.2")

#: Fortran path records in IPNAMES.DAT / *.RZX are read into CHARACTER*80.
MAX_PATH_LEN = 79
#: Cropsim-CERES (wheat, canola; ``DSSAT40/CSCER/CSCER040.FOR``) builds the ecotype path in
#: ``CHARACTER*64 ECDIRFLE`` = ``<rundir>/DSSAT/`` + a 12-character name (``WHCER040.ECO``). A
#: longer run dir truncates the name and the model STOPs with exit status 0, so the RZWQM2 run
#: dir must be at most this long.
MAX_RZWQM_RUN_DIR_LEN = 64 - len("/DSSAT/") - len("WHCER040.ECO")
#: ``run.log`` text (lower case) that RZWQM2 / DSSAT40 prints only right before a Fortran ``STOP``
#: (or, for ``-- error -- error --``, before continuing with unread inputs); the binary still exits
#: with status 0. RZWQM2 4.6 sources: ``program will have to stop`` / ``could not find input file``
#: (Cropsim-CERES, ``DSSAT40/CSCER/CSUTS040.FOR`` FVCHECK and others); ``end of file reached in
#: daymet.dat`` (``Rzmain.for`` weather reader); ``fatal error reading brkpnt.dat``
#: (``Rzday.for``); ``-- error -- error --`` (``Rzmain.for`` IPNAMES opener, ``readrzx.for``);
#: ``<<< error in dates >>>`` (``Rzmain.for`` period check). Most RZWQM2 STOPs are bare or print
#: free text, and a normal run itself ends in ``STOP 'check your expdata.dat file ...'``, so the
#: ``.ana`` row count in :func:`check_rzwqm_outputs` is the complete guard; these markers only
#: give a clearer error.
RZWQM_STOP_MARKERS = (
    "program will have to stop",
    "could not find input file",
    "end of file reached in daymet.dat",
    "fatal error reading brkpnt.dat",
    "-- error -- error --",
    "<<< error in dates >>>",
)
#: Text (lower case) in dscsm048's ``run.log`` or ``WARNING.OUT`` that means a run stopped or a
#: season ended early; dscsm048 (gfortran) exits 0 after ``STOP ' '`` / bare ``STOP`` and after
#: a season cut short by :code:`WeatherError` / :code:`ErrorCode`. DSSAT-CSM v4.8 sources:
#: ``simulation will end`` (``Weather/IPWTH_alt.for`` WeatherError, ``InputModule/ipexp.for``);
#: ``will be terminated`` and ``simulations terminated`` (``Utilities/ERROR.for`` ErrorCode and
#: ERROR); ``simulation ended with error code`` (``CSM_Main/LAND.for``); ``program will stop`` /
#: ``model will stop`` (WARNING before ERROR or STOP, e.g. ``SPAM/ETPHOT.for``, ``Plant/plant.for``);
#: ``program will have to stop`` (``Utilities/CSUTS.for``, ``CSREADS.for``, Cropsim / CSCAS);
#: ``unknown soil type`` (``Soil/SoilUtilities/TextureClass.for``); ``more than nl layers``
#: (``Soil/SoilUtilities/LMATCH.for``). Crop failure (``Growth program terminated.``) is a model
#: outcome, not a stop, and is not listed.
DSCSM_STOP_MARKERS = (
    "simulation will end",
    "will be terminated",
    "simulations terminated",
    "simulation ended with error code",
    "program will stop",
    "model will stop",
    "program will have to stop",
    "unknown soil type",
    "more than nl layers",
)
#: gfortran prints ``STOP <code>`` on stderr for a ``STOP`` with a stop code (``STOP ' '``).
_GFORTRAN_STOP = re.compile(r"^\s*STOP\b", re.MULTILINE)
_ANA_ROW = re.compile(r"^\s*(\d{4})\.(\d{3})\s")
#: IPNAMES.DAT lines 1-8 (0-based index -> file role). Line 3 MET, 4 BRK, 7 SNO, 8 .ana.
_IPNAMES_ROLES = ("cntrl", "rzwqm", "met", "brk", "rzinit", "plgen", "sno", "ana")

DateLike = _dt.date | str


class FortranRunError(RuntimeError):
    """The oracle exited non-zero, timed out, or did not write its expected outputs.

    ``run_dir`` is the staging directory, which is left on disk for inspection.
    """

    def __init__(self, msg: str, run_dir: Path | None = None) -> None:
        super().__init__(msg)
        self.run_dir = run_dir


@dataclass(frozen=True)
class RzwqmResult:
    """Paths (inside ``out_dir``) of the kept RZWQM2 outputs."""

    ana_path: Path
    overview_path: Path | None
    run_log: Path
    elapsed_s: float
    run_dir: Path | None = None  # kept only when ``keep_run_dir=True``


@dataclass(frozen=True)
class DscsmResult:
    """Paths (inside ``out_dir``) of the kept DSSAT-CSM outputs."""

    out_dir: Path
    summary_path: Path | None
    run_log: Path
    elapsed_s: float
    files: tuple[Path, ...]
    run_dir: Path | None = None


# --------------------------------------------------------------------------- helpers


def _to_date(d: DateLike) -> _dt.date:
    if isinstance(d, _dt.date):
        return d
    return _dt.date.fromisoformat(str(d))


def elf_interpreter(path: str | os.PathLike[str]) -> str | None:
    """Return the ``PT_INTERP`` string of an ELF executable, or ``None`` (static / not ELF)."""
    with open(path, "rb") as fh:
        ident = fh.read(16)
        if len(ident) < 16 or ident[:4] != b"\x7fELF":
            return None
        is64 = ident[4] == 2
        end = "<" if ident[5] == 1 else ">"
        if is64:
            hdr = fh.read(48)
            e_phoff = struct.unpack(end + "Q", hdr[16:24])[0]
            e_phentsize, e_phnum = struct.unpack(end + "HH", hdr[38:42])
        else:
            hdr = fh.read(36)
            e_phoff = struct.unpack(end + "I", hdr[12:16])[0]
            e_phentsize, e_phnum = struct.unpack(end + "HH", hdr[26:30])
        for i in range(e_phnum):
            fh.seek(e_phoff + i * e_phentsize)
            ph = fh.read(e_phentsize)
            if struct.unpack(end + "I", ph[:4])[0] != 3:  # PT_INTERP
                continue
            if is64:
                p_offset, _, _, p_filesz = struct.unpack(end + "QQQQ", ph[8:40])
            else:
                p_offset, _, _, p_filesz = struct.unpack(end + "IIII", ph[4:20])
            fh.seek(p_offset)
            return fh.read(p_filesz).rstrip(b"\x00").decode()
    return None


def _exec_argv(binary: Path, name: str) -> list[str]:
    """argv for running ``./name`` (a copy of ``binary``) in the run directory."""
    interp = elf_interpreter(binary)
    if interp is not None and not Path(interp).exists():
        if not SYSTEM_LOADER.exists():
            raise FortranRunError(f"{binary} needs {interp}, and {SYSTEM_LOADER} is missing")
        return [str(SYSTEM_LOADER), f"./{name}"]
    return [f"./{name}"]


def _make_run_dir(prefix: str, run_root: Path | None, max_len: int | None = None) -> Path:
    """``tempfile.mkdtemp(prefix, dir=run_root)``; raise when the path exceeds ``max_len`` characters."""
    root = Path(run_root) if run_root is not None else RUN_ROOT
    root.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=prefix, dir=root)).resolve()
    if max_len is not None and len(str(d)) > max_len:
        shutil.rmtree(d, ignore_errors=True)
        raise FortranRunError(
            f"run dir {d} is {len(str(d))} characters; RZWQM2 needs <= {max_len} (the Cropsim-CERES "
            "ecotype path is CHARACTER*64). Use a shorter run_root or AGRI_JAX_RUN_ROOT."
        )
    return d


def _copy_tree_into(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        target = dst / p.name
        if p.is_dir():
            shutil.copytree(p, target, dirs_exist_ok=True)
        else:
            shutil.copy2(p, target)


def _tail(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return "<no log>"
    return "\n".join(lines[-n:])


def _run(
    argv: list[str], cwd: Path, log: Path, timeout: float, env: dict[str, str], stdin: str | None
) -> float:
    t0 = time.perf_counter()
    with open(log, "wb") as fh:
        try:
            proc = subprocess.run(
                argv,
                cwd=cwd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                input=None if stdin is None else stdin.encode(),
                stdin=subprocess.DEVNULL if stdin is None else None,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise FortranRunError(f"{argv} timed out after {timeout} s in {cwd}\n{_tail(log)}") from e
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise FortranRunError(f"{argv} exited {proc.returncode} in {cwd}; log tail:\n{_tail(log)}")
    return elapsed


def _keep(run_dir: Path, out_dir: Path, patterns: Sequence[str]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Path] = []
    names = sorted(p.name for p in run_dir.iterdir() if p.is_file())
    for pat in patterns:
        for name in names:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(name.upper(), pat.upper()):
                dst = out_dir / name
                if dst not in kept:
                    shutil.copy2(run_dir / name, dst)
                    kept.append(dst)
    return kept


# --------------------------------------------------------------------------- RZWQM2


def patch_ipnames(
    text: str,
    run_dir: Path,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
) -> str:
    """Rewrite IPNAMES.DAT: lines 1-8 -> ``run_dir/<basename>``; line 9 dates if given.

    Basenames are taken from the original (Windows) paths. Raises ``ValueError`` when a
    resulting path does not fit the 80-character Fortran record.
    """
    lines = text.splitlines()
    if len(lines) < 9:
        raise ValueError("IPNAMES.DAT has fewer than 9 lines")
    for i in range(len(_IPNAMES_ROLES)):
        base = re.split(r"[\\/]", lines[i].strip())[-1]
        new = f"{run_dir}/{base}"
        if len(new) > MAX_PATH_LEN:
            raise ValueError(f"IPNAMES line {i + 1} path longer than {MAX_PATH_LEN} chars: {new}")
        lines[i] = new
    if start is not None or end is not None:
        f = lines[8].split()
        if len(f) < 6:
            raise ValueError(f"unexpected IPNAMES date line: {lines[8]!r}")
        d0 = _to_date(start) if start is not None else _dt.date(int(f[2]), int(f[1]), int(f[0]))
        d1 = _to_date(end) if end is not None else _dt.date(int(f[5]), int(f[4]), int(f[3]))
        if d1 < d0:
            raise ValueError(f"end {d1} before start {d0}")
        lines[8] = f"{d0.day}  {d0.month}  {d0.year}  {d1.day}  {d1.month}  {d1.year}"
    return "\r\n".join(lines) + "\r\n" if "\r\n" in text else "\n".join(lines) + "\n"


def patch_rzx_database(text: str, run_dir: Path) -> str:
    """Replace the two paths after ``= DATABASE FILE LOCATIONS`` with ``<run_dir>/DSSAT/``, ``<run_dir>/``."""
    lines = text.splitlines()
    try:
        h = next(i for i, ln in enumerate(lines) if ln.strip().startswith("= DATABASE FILE LOCATIONS"))
    except StopIteration as e:
        raise ValueError("no '= DATABASE FILE LOCATIONS' block") from e
    new = [f"{run_dir}/DSSAT/", f"{run_dir}/"]
    for p in new:
        if len(p) > MAX_PATH_LEN:
            raise ValueError(f"RZX database path longer than {MAX_PATH_LEN} chars: {p}")
    j, k = h + 1, 0
    while j < len(lines) and k < 2:
        s = lines[j].strip()
        if s and not re.fullmatch(r"=+", s):
            lines[j] = new[k]
            k += 1
        j += 1
    if k < 2:
        raise ValueError("DATABASE FILE LOCATIONS block has fewer than two paths")
    return "\r\n".join(lines) + "\r\n" if "\r\n" in text else "\n".join(lines) + "\n"


def _ipnames_period(ip_text: str) -> tuple[_dt.date, _dt.date]:
    """Simulation period of line 9 of IPNAMES.DAT (``DD MM YYYY DD MM YYYY``)."""
    f = ip_text.splitlines()[8].split()
    return _dt.date(int(f[2]), int(f[1]), int(f[0])), _dt.date(int(f[5]), int(f[4]), int(f[3]))


def _ana_day_tokens(path: Path) -> list[tuple[int, int]]:
    """``(year, day)`` of every data row of a ``.ana`` file (``YYYY.DDD`` first token)."""
    out: list[tuple[int, int]] = []
    with open(path, errors="replace") as fh:
        for ln in fh:
            m = _ANA_ROW.match(ln)
            if m:
                out.append((int(m.group(1)), int(m.group(2))))
    return out


def check_rzwqm_outputs(log: Path, ana: Path, start: _dt.date, end: _dt.date) -> None:
    """Raise :class:`FortranRunError` unless an RZWQM2 run finished its whole period.

    RZWQM2 exits with status 0 after a Fortran ``STOP`` (``Program will have to stop``,
    ``Could not find input file!``), so the exit status alone does not tell a finished run from a
    truncated one. Checked: no :data:`RZWQM_STOP_MARKERS` line in ``run.log``; the ``.ana`` holds
    ``(end - start).days + 2`` rows (a ``YYYY.000`` initial state plus one row per day) and its
    last row is ``end``.
    """
    text = log.read_text(errors="replace") if log.is_file() else ""
    low = text.lower()
    hit = next((m for m in RZWQM_STOP_MARKERS if m in low), None)
    if hit is not None:
        raise FortranRunError(f"RZWQM exited 0 but run.log reports {hit!r}; log tail:\n{_tail(log)}")
    if not ana.is_file() or ana.stat().st_size == 0:
        raise FortranRunError(f"RZWQM finished but {ana} is missing/empty; log tail:\n{_tail(log)}")
    rows = _ana_day_tokens(ana)
    want = (end - start).days + 2
    last = (end.year, end.timetuple().tm_yday)
    if len(rows) != want or rows[-1] != last:
        got = f"{rows[-1][0]}.{rows[-1][1]:03d}" if rows else "none"
        raise FortranRunError(
            f"{ana.name}: {len(rows)} rows ending {got}, expected {want} rows ending "
            f"{last[0]}.{last[1]:03d} (truncated run); log tail:\n{_tail(log)}"
        )


def _as_paths(x: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None) -> list[Path]:
    if x is None:
        return []
    if isinstance(x, (str, os.PathLike)):
        return [Path(x)]
    return [Path(p) for p in x]


def run_rzwqm(
    scenario_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    binary: str | os.PathLike[str] | None = None,
    dssat_db: str | os.PathLike[str] | None = None,
    dat_override: str | os.PathLike[str] | None = None,
    cul_override: str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | None = None,
    start: DateLike | None = None,
    end: DateLike | None = None,
    keep_files: Sequence[str] = ("*.ana", "OVERVIEW.OUT"),
    timeout: float = 600,
    keep_run_dir: bool = False,
    run_root: str | os.PathLike[str] | None = None,
) -> RzwqmResult:
    """Run RZWQM2 on a scenario directory and copy its outputs to ``out_dir``.

    Parameters
    ----------
    scenario_dir : directory with ``rzwqm.dat``, ``IPNAMES.DAT``, ``*.RZX``, weather, ...
    out_dir : destination of the kept files (created). ``run.log`` is always kept.
    binary : RZWQM executable (default ``RZWQM_Tool/main_ryzen5_avx512``).
    dssat_db : DSSAT database dir copied to ``<rundir>/DSSAT`` (default ``RZWQM_Tool/DSSAT``).
    dat_override : a ``*.dat`` copied over ``rzwqm.dat``.
    cul_override : one or more cultivar files copied into the run dir (same name overwrites).
    start, end : ``datetime.date`` or ISO string; rewrite the IPNAMES simulation period.
    keep_files : glob patterns (case-insensitive) of run-dir files to copy to ``out_dir``.
    timeout : seconds before the run is killed.
    keep_run_dir : keep the staging directory (it is always kept on failure).
    run_root : parent of the staging directory (default ``~/agri_jax_data/run``).
    """
    scenario = Path(scenario_dir).resolve()
    out = Path(out_dir).resolve()
    exe = Path(binary) if binary is not None else RZWQM_BINARY
    db = Path(dssat_db) if dssat_db is not None else RZWQM_TOOL / "DSSAT"
    if not (scenario / "IPNAMES.DAT").is_file():
        raise FileNotFoundError(f"{scenario}/IPNAMES.DAT not found")
    if not exe.is_file():
        raise FileNotFoundError(f"RZWQM binary {exe} not found")
    if not db.is_dir():
        raise FileNotFoundError(f"DSSAT database {db} not found")

    run_dir = _make_run_dir(
        "r", Path(run_root) if run_root is not None else None, max_len=MAX_RZWQM_RUN_DIR_LEN
    )
    ok = False
    try:
        _copy_tree_into(scenario, run_dir)
        _copy_tree_into(db, run_dir / "DSSAT")
        if dat_override is not None:
            shutil.copy2(dat_override, run_dir / "rzwqm.dat")
        for cul in _as_paths(cul_override):
            shutil.copy2(cul, run_dir / cul.name)
        name = exe.name
        shutil.copy2(exe, run_dir / name)
        (run_dir / name).chmod(0o755)
        err = db / "MODEL.ERR"
        if err.is_file():
            shutil.copy2(err, run_dir / "main_ryzenMODEL.ERR")

        ip = run_dir / "IPNAMES.DAT"
        ip_text = patch_ipnames(ip.read_text(errors="replace"), run_dir, start=start, end=end)
        ip.write_text(ip_text, newline="")
        ana_name = Path(ip_text.splitlines()[7].strip()).name
        for rzx in run_dir.iterdir():
            if rzx.is_file() and rzx.suffix.upper() == ".RZX":
                t = rzx.read_text(errors="replace")
                if "= DATABASE FILE LOCATIONS" in t:
                    rzx.write_text(patch_rzx_database(t, run_dir), newline="")
        for stale in (ana_name, "OVERVIEW.OUT"):
            (run_dir / stale).unlink(missing_ok=True)

        env = dict(os.environ, FORT_BUFFERED="TRUE")
        log = run_dir / "run.log"
        elapsed = _run(_exec_argv(exe, name), run_dir, log, timeout, env, None)

        ana = run_dir / ana_name
        check_rzwqm_outputs(log, ana, *_ipnames_period(ip_text))
        kept = _keep(run_dir, out, [*keep_files, "run.log"])
        by_name = {p.name: p for p in kept}
        if ana_name not in by_name:
            shutil.copy2(ana, out / ana_name)
        ov = next((p for p in kept if p.name.upper() == "OVERVIEW.OUT"), None)
        ok = True
        return RzwqmResult(
            ana_path=out / ana_name,
            overview_path=ov,
            run_log=out / "run.log",
            elapsed_s=elapsed,
            run_dir=run_dir if keep_run_dir else None,
        )
    except FortranRunError as e:
        raise FortranRunError(f"{e}\n(run dir kept: {run_dir})", run_dir) from e
    finally:
        if ok and not keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


def parse_overview_yields(path: str | os.PathLike[str]) -> list[float]:
    """Seasonal yields (kg/ha) from ``Maize YIELD :  NNNN kg/ha`` lines of OVERVIEW.OUT."""
    pat = re.compile(r"YIELD\s*:\s*([-\d.]+)\s*kg/ha", re.IGNORECASE)
    return [float(m.group(1)) for m in pat.finditer(Path(path).read_text(errors="replace"))]


# --------------------------------------------------------------------------- DSSAT-CSM


def _rewrite_dssatpro(path: Path, run_dir: Path) -> None:
    text = path.read_text(errors="replace")
    if "DSSAT48" in text:
        text = text.replace("C: \\DSSAT48", str(run_dir)).replace("C:\\DSSAT48", str(run_dir))
        path.write_text(text.replace("\\", "/"))


def _filex_treatments(path: Path) -> list[int]:
    """``TRTNO`` of every data line of the ``*TREATMENTS`` section of a FileX.

    Mirrors DSSAT-CSM ``IPEXP`` (``InputModule/ipexp.for``): ``IGNORE`` skips blank, ``!`` and
    ``@`` lines and ends the section at a line starting ``*`` or ``$``; ``TRTALL`` is the number
    of lines read, and ``TRTNO`` is format ``I3`` (columns 1-3).
    """
    out: list[int] = []
    inside = False
    for ln in path.read_text(errors="replace").splitlines():
        if ln[:1] in ("*", "$"):
            inside = ln.upper().startswith("*TREATMENTS")
            continue
        if not inside or ln[:1] in ("!", "@") or not ln.strip():
            continue
        try:
            out.append(int(ln[:3]))
        except ValueError:
            continue
    return out


def _summary_trnos(path: Path) -> list[int]:
    """``TRNO`` of every data row of a ``Summary.OUT`` (the column after ``RUNNO``)."""
    out: list[int] = []
    col: int | None = None
    for ln in path.read_text(errors="replace").splitlines():
        if ln.startswith("@"):
            names = ln[1:].split()
            col = names.index("TRNO") if "TRNO" in names else None
            continue
        if col is None or ln[:1] in ("*", "!") or not ln.strip():
            continue
        tok = ln.split()
        if len(tok) > col and tok[0].isdigit() and tok[col].lstrip("-").isdigit():
            out.append(int(tok[col]))
    return out


def check_dscsm_outputs(
    run_dir: str | os.PathLike[str],
    log: str | os.PathLike[str] | None = None,
    *,
    experiment_file: str | None = None,
    run_mode: str = "A",
) -> None:
    """Raise :class:`FortranRunError` unless a dscsm048 run (exit status 0) finished every season.

    dscsm048 exits 0 after ``STOP ' '`` and bare ``STOP`` (e.g. ``InputModule/ipexp.for`` when
    both weather station and soil are ``-99``, ``Soil/SoilUtilities/LMATCH.for``,
    ``SPAM/ETPHOT.for``, the Cropsim/CSCAS readers) and when a season is cut short by missing or
    bad weather (``Weather/IPWTH_alt.for`` ``WeatherError`` in run modes other than F/Q/Y: the run
    continues, ``Summary.OUT`` gets a row with ``-99`` dates). ``ERROR`` (``Utilities/ERROR.for``)
    ends in ``STOP 99`` and is caught by the exit status. Checked here:

    * no ``ERROR.OUT`` (written by ``ERROR`` and by the ``CSUTS``/``CSREADS`` stops);
    * no ``STOP`` line (gfortran's report of ``STOP <code>``) in ``run.log``;
    * no :data:`DSCSM_STOP_MARKERS` text in ``run.log`` or ``WARNING.OUT``;
    * ``Summary.OUT`` exists with at least one run and, for ``run_mode="A"`` with the FileX in
      ``run_dir``, holds every ``TRTNO`` of its ``*TREATMENTS`` section (a mid-season STOP leaves
      that treatment and all later ones out, since the row is written at season end).
    """
    rd = Path(run_dir)
    lg = Path(log) if log is not None else rd / "run.log"
    log_text = lg.read_text(errors="replace") if lg.is_file() else ""

    def fail(why: str) -> FortranRunError:
        return FortranRunError(f"dscsm048 exited 0 but {why}; log tail:\n{_tail(lg)}")

    if (rd / "ERROR.OUT").is_file():
        err = (rd / "ERROR.OUT").read_text(errors="replace").strip().splitlines()
        raise fail("wrote ERROR.OUT:\n" + "\n".join(err[-12:]))
    m = _GFORTRAN_STOP.search(log_text)
    if m is not None:
        line = log_text[m.start() :].splitlines()[0].strip()
        raise fail(f"run.log reports a Fortran STOP ({line!r})")
    warn = rd / "WARNING.OUT"
    for name, text in (("run.log", log_text), ("WARNING.OUT", _read_or_empty(warn))):
        low = text.lower()
        hit = next((k for k in DSCSM_STOP_MARKERS if k in low), None)
        if hit is not None:
            n = low[: low.index(hit)].count("\n")
            ctx = "\n".join(text.splitlines()[max(0, n - 5) : n + 1])
            raise fail(f"{name} reports {hit!r}:\n{ctx}")
    summary = rd / "Summary.OUT"
    if not summary.is_file():
        raise fail(f"wrote no Summary.OUT in {rd}")
    got = _summary_trnos(summary)
    if not got:
        raise fail("Summary.OUT has no run rows")
    if run_mode.upper() == "A" and experiment_file is not None and (rd / experiment_file).is_file():
        want = _filex_treatments(rd / experiment_file)
        missing = sorted(set(want) - set(got))
        if missing or len(got) < len(want):
            raise fail(
                f"Summary.OUT has {len(got)} runs (treatments {sorted(set(got))}); "
                f"{experiment_file} lists {len(want)} treatments, missing {missing} (run stopped early)"
            )


def _read_or_empty(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def run_dscsm(
    exp_dir: str | os.PathLike[str],
    out_dir: str | os.PathLike[str] | None = None,
    *,
    model: str = "MZCER048",
    run_mode: str = "A",
    experiment_file: str | None = None,
    engine: str | os.PathLike[str] | None = None,
    weather_dir: str | os.PathLike[str] | None = None,
    soil_dir: str | os.PathLike[str] | None = None,
    extra_files: Sequence[str | os.PathLike[str]] = (),
    keep_files: Sequence[str] = ("*.OUT",),
    timeout: float = 180,
    keep_run_dir: bool = False,
    run_root: str | os.PathLike[str] | None = None,
    check: bool = True,
) -> DscsmResult:
    """Run DSSAT-CSM ``dscsm048 <model> <run_mode> <experiment_file>`` in a staged run dir.

    ``exp_dir`` holds the experiment (``*.MZX`` etc.); its files are all staged. Weather files
    ``<INSI>*.WTH`` matching the experiment's first four letters are taken from ``exp_dir`` or
    ``weather_dir`` (default ``<engine>/example_data/Weather``); all ``*.SOL`` from ``soil_dir``
    (default ``<engine>/example_data/Soil``). ``engine`` defaults to
    ``~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine``. Outputs matching ``keep_files`` go to
    ``out_dir`` (default ``<exp_dir>/dscsm_out``). ``*.OUT`` files in ``exp_dir`` (outputs of an
    earlier run) are not staged. With ``check=True`` (default) :func:`check_dscsm_outputs` must
    pass; ``check=False`` only requires ``Summary.OUT`` (for runs where a season is expected to be
    cut short, e.g. weather that ends before the last harvest).
    """
    exp = Path(exp_dir).resolve()
    eng = Path(engine) if engine is not None else DSSAT_ENGINE
    bindir = eng / "bin"
    exe = bindir / "dscsm048"
    if not exe.is_file():
        raise FileNotFoundError(f"dscsm048 not found at {exe}")
    if experiment_file is None:
        cands = sorted(p.name for p in exp.iterdir() if re.fullmatch(r".+\.[A-Z]{2}X", p.name.upper()))
        if len(cands) != 1:
            raise ValueError(f"pass experiment_file=; found {cands} in {exp}")
        experiment_file = cands[0]
    if run_mode.upper() == "A" and not (exp / experiment_file).is_file():
        raise FileNotFoundError(f"{exp / experiment_file} not found")
    out = Path(out_dir).resolve() if out_dir is not None else exp / "dscsm_out"
    wdir = Path(weather_dir) if weather_dir is not None else eng / "example_data" / "Weather"
    sdir = Path(soil_dir) if soil_dir is not None else eng / "example_data" / "Soil"

    run_dir = _make_run_dir("ds_", Path(run_root) if run_root is not None else None)
    ok = False
    try:
        for f in [*bindir.glob("*.CDE"), *(bindir / n for n in ("DSCSM048.CTR", "MODEL.ERR"))]:
            if f.is_file():
                shutil.copy2(f, run_dir / f.name)
        for n in ("DSSATPRO.v48", "DSSATPRO.L48", "DSSATPRO.L48.in"):
            if (bindir / n).is_file():
                shutil.copy2(bindir / n, run_dir / n)
                _rewrite_dssatpro(run_dir / n, run_dir)
        for sub in ("StandardData", "Genotype"):
            d = bindir / sub
            if d.is_dir():
                for f in d.iterdir():
                    if f.is_file():
                        shutil.copy2(f, run_dir / f.name)
        insi = experiment_file[:4].upper()
        if wdir.is_dir():
            for f in wdir.iterdir():
                if f.is_file() and f.name.upper().startswith(insi) and f.suffix.upper() == ".WTH":
                    shutil.copy2(f, run_dir / f.name)
        if sdir.is_dir():
            for f in sdir.glob("*.SOL"):
                shutil.copy2(f, run_dir / f.name)
        for f in exp.iterdir():
            if f.is_file() and f.suffix.upper() != ".OUT" and f.name != "run.log":
                shutil.copy2(f, run_dir / f.name)
        for f in map(Path, extra_files):
            shutil.copy2(f, run_dir / f.name)
        shutil.copy2(exe, run_dir / "dscsm048")
        (run_dir / "dscsm048").chmod(0o755)

        log = run_dir / "run.log"
        argv = [*_exec_argv(exe, "dscsm048"), model, run_mode, experiment_file]
        elapsed = _run(argv, run_dir, log, timeout, dict(os.environ), "\n")
        if check:
            check_dscsm_outputs(run_dir, log, experiment_file=experiment_file, run_mode=run_mode)
        elif not (run_dir / "Summary.OUT").is_file():
            raise FortranRunError(f"dscsm048 wrote no Summary.OUT in {run_dir}; log tail:\n{_tail(log)}")
        kept = _keep(run_dir, out, [*keep_files, "run.log"])
        ok = True
        return DscsmResult(
            out_dir=out,
            summary_path=next((p for p in kept if p.name.upper() == "SUMMARY.OUT"), None),
            run_log=out / "run.log",
            elapsed_s=elapsed,
            files=tuple(kept),
            run_dir=run_dir if keep_run_dir else None,
        )
    except FortranRunError as e:
        raise FortranRunError(f"{e}\n(run dir kept: {run_dir})", run_dir) from e
    finally:
        if ok and not keep_run_dir:
            shutil.rmtree(run_dir, ignore_errors=True)


# --------------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m agrijax.port.run_fortran", description=(__doc__ or "").split("\n")[0]
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    rz = sub.add_parser("rzwqm", help="run RZWQM2 on a scenario directory")
    rz.add_argument("scenario")
    rz.add_argument("out")
    rz.add_argument("--start", help="YYYY-MM-DD")
    rz.add_argument("--end", help="YYYY-MM-DD")
    rz.add_argument("--binary")
    rz.add_argument("--dssat-db")
    rz.add_argument("--dat", help="rzwqm.dat override")
    rz.add_argument("--cul", nargs="*", default=None, help="cultivar file override(s)")
    rz.add_argument("--keep", nargs="*", default=["*.ana", "OVERVIEW.OUT"], help="glob patterns to keep")
    rz.add_argument("--timeout", type=float, default=600)
    rz.add_argument("--keep-run-dir", action="store_true")
    ds = sub.add_parser("dscsm", help="run DSSAT-CSM dscsm048 on an experiment directory")
    ds.add_argument("exp_dir")
    ds.add_argument("out")
    ds.add_argument("--model", default="MZCER048")
    ds.add_argument("--mode", default="A")
    ds.add_argument("--experiment")
    ds.add_argument("--timeout", type=float, default=180)
    ds.add_argument("--keep-run-dir", action="store_true")
    ds.add_argument("--no-check", action="store_true", help="skip check_dscsm_outputs")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "rzwqm":
            r = run_rzwqm(
                a.scenario,
                a.out,
                binary=a.binary,
                dssat_db=a.dssat_db,
                dat_override=a.dat,
                cul_override=a.cul,
                start=a.start,
                end=a.end,
                keep_files=tuple(a.keep),
                timeout=a.timeout,
                keep_run_dir=a.keep_run_dir,
            )
            print(f"ok {r.elapsed_s:.1f} s  ana={r.ana_path}  overview={r.overview_path}")
            if r.overview_path is not None:
                print("yields kg/ha:", [int(y) for y in parse_overview_yields(r.overview_path)])
        else:
            d = run_dscsm(
                a.exp_dir,
                a.out,
                model=a.model,
                run_mode=a.mode,
                experiment_file=a.experiment,
                timeout=a.timeout,
                keep_run_dir=a.keep_run_dir,
                check=not a.no_check,
            )
            print(f"ok {d.elapsed_s:.1f} s  summary={d.summary_path}  ({len(d.files)} files)")
    except (FortranRunError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
