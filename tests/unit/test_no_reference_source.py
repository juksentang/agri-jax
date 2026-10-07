"""The public tree holds no RZWQM2-derived module, no reference-source fragment and no group scenario data.

RZWQM2 is closed source: it is used only as an external reference whose outputs are compared. This
pure text test scans every tracked file (``git ls-files``; a directory walk without git) for

* the paths and dotted names of the modules that were derived from the RZWQM2 source and are kept
  outside this repository;
* fragments of the reference source (Fortran statements, argument tables, data tables, log texts);
* the soil hydraulic parameter block of the group's CA-TPA scenario.

No data and no JAX are needed. The patterns are assembled from parts so that this file does not
match itself.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
#: directories never scanned (build artefacts, environments, the gitignored docs link)
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}
_SKIP_PREFIXES = ("docs/internal",)
#: binary or generated file types that are not scanned
_SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".npz", ".npy", ".zip", ".gz", ".whl", ".lock"}


def _j(*parts: str) -> str:
    return "".join(parts)


#: removed modules: path fragments and dotted names
REMOVED_MODULES: tuple[str, ...] = (
    _j("processes/", "snow"),
    _j("processes.", "snow"),
    _j("soil_water/", "infiltration"),
    _j("soil_water.", "infiltration"),
    _j("soil_water/", "conventions"),
    _j("soil_water.", "conventions"),
    _j("processes/soil_water/", "day.py"),
    _j("processes.soil_water.", "day"),
    _j("water_supply/", "uptake_limit"),
    _j("water_supply.", "uptake_limit"),
    _j("water_supply/", "publish"),
    _j("water_supply.", "publish"),
    _j("water_supply/", "season.py"),
    _j("water_supply.", "season"),
    _j("pet/", "shuttleworth_wallace.py"),
    _j("processes.pet.", "shuttleworth_wallace"),
    _j("ceres_maize/", "canopy.py"),
    _j("ceres_maize/", "canopy_params"),
    _j("processes.crop.ceres_maize.", "canopy"),
    _j("ceres_maize.", "canopy_params"),
    _j("forcing/", "radiation"),
    _j("forcing.", "radiation"),
    _j("forcing/", "precipitation"),
    _j("forcing.", "precipitation"),
    _j("day_", "rzwqm46"),
    _j("catpa_", "pet_demo"),
)

#: fragments of the reference source and of its run-time log (compared case-insensitively)
SOURCE_FRAGMENTS: tuple[str, ...] = (
    _j("IFPL(IR)", ".EQ."),
    _j("GOTO ", "150"),
    _j("GOTO ", "120"),
    _j("MAX(U, ", "UBREEZ)"),
    _j("1.0D", "-2"),
    _j("FC33 = ", "B/333"),
    _j("333.0", "D0"),
    _j("-15000.", "D0"),
    _j("CALL ", "CDATE"),
    _j("qsr * 24 ", "* TL"),
    _j("RICHRD", "_ARGS"),
    _j("POTEVPHR", "_ARGS"),
    _j("ECDIR", "FLE"),
    _j("WUF = min(1, ", "PET / TRWUP)"),
    _j("end of file reached ", "in daymet.dat"),
    _j("fatal error reading ", "brkpnt.dat"),
    _j("-- error -- ", "error --"),
    _j("<<< error ", "in dates >>>"),
    _j("could not find ", "input file"),
    _j("check your ", "expdata.dat"),
    _j("RZWQM_Linux", "_Ver45"),
    _j("SNAL", "BEDO"),
    _j("Accumulation", "Albedo"),
)

#: the PRMS albedo DATA tables (accumulation and melt stages), any separators between the values
_SEP = r"[^0-9]{1,12}"
ALBEDO_TABLES: tuple[re.Pattern[str], ...] = (
    re.compile(_SEP.join((r"0\.80?", r"0\.77", r"0\.75", r"0\.72", r"0\.70?", r"0\.69"))),
    re.compile(_SEP.join((r"0\.72", r"0\.65", r"0\.60?", r"0\.58", r"0\.56", r"0\.54"))),
)

#: the CA-TPA soil hydraulic parameter block (single values and the per-horizon sequences)
CATPA_VALUES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![0-9.])" + re.escape(_j("14.", "6545")) + r"(?![0-9])"),
    re.compile(r"(?<![0-9.])" + re.escape(_j("7440.", "01")) + r"(?![0-9])"),
    re.compile(r"(?<![0-9.])" + re.escape(_j("2.", "966")) + r"(?![0-9])"),
    re.compile(r"(?<![0-9.])" + re.escape(_j("0.25", "5198")) + r"(?![0-9])"),
    re.compile(r"(?<![0-9.])" + re.escape(_j("0.14", "1628")) + r"(?![0-9])"),
    re.compile(_SEP.join((r"0\.22", r"0\.26", r"0\.36", r"0\.17", r"0\.322"))),
    re.compile(_SEP.join((r"5\.41", r"3\.16", r"3\.31", r"3\.32", r"2\.59"))),
    re.compile(_SEP.join((r"0\.055", r"0\.032", r"0\.043", r"0\.048", r"0\.041"))),
)


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True, timeout=60
        ).stdout.decode()
        rels = [r for r in out.split("\0") if r]
    except (OSError, subprocess.SubprocessError):
        ignored = _gitignore_patterns()
        rels = [
            rel
            for p in REPO.rglob("*")
            if p.is_file()
            and not (set(p.relative_to(REPO).parts) & _SKIP_DIRS)
            and not _ignored(rel := p.relative_to(REPO).as_posix(), ignored)
        ]
    files = []
    for rel in rels:
        p = REPO / rel
        if rel.startswith(_SKIP_PREFIXES) or p.suffix.lower() in _SKIP_SUFFIXES:
            continue
        if p.resolve() == SELF or not p.is_file():
            continue
        files.append(p)
    return files


def _gitignore_patterns() -> list[str]:
    """The patterns of the top-level ``.gitignore`` (a tree copied without ``.git``)."""
    f = REPO / ".gitignore"
    lines = f.read_text().splitlines() if f.is_file() else []
    return [ln.strip().rstrip("/") for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "!"))]


def _ignored(rel: str, patterns: list[str]) -> bool:
    """Whether ``rel`` or one of its parent directories matches a ``.gitignore`` pattern, or lies in
    a dot directory (local run logs and scratch)."""
    parts = rel.split("/")
    if any(p.startswith(".") and p not in (".github", ".gitignore", ".gitattributes") for p in parts[:-1]):
        return True
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    return any(
        fnmatch.fnmatch(x, pat) or fnmatch.fnmatch(parts[i], pat)
        for pat in patterns
        for i, x in enumerate(prefixes)
    )


FILES = _tracked_files()


def _text(p: Path) -> str:
    return p.read_bytes().decode("utf-8", errors="replace")


def _hits(check: Callable[[str], str]) -> list[str]:
    out = []
    for p in FILES:
        text = _text(p)
        for i, line in enumerate(text.splitlines(), start=1):
            what = check(line)
            if what:
                out.append(f"{p.relative_to(REPO)}:{i}: {what}")
    return out


def test_the_scan_sees_the_tree() -> None:
    names = {str(p.relative_to(REPO)) for p in FILES}
    assert "README.md" in names and "src/agrijax/__init__.py" in names
    assert len(FILES) > 100


def test_no_removed_module_is_named() -> None:
    def check(line: str) -> str:
        return next((m for m in REMOVED_MODULES if m in line), "")

    hits = _hits(check)
    assert hits == [], "\n".join(hits)


def test_no_reference_source_fragment() -> None:
    frags = tuple(f.lower() for f in SOURCE_FRAGMENTS)

    def check(line: str) -> str:
        low = line.lower()
        hit = next((f for f in frags if f in low), "")
        if hit:
            return hit
        return next((r.pattern for r in ALBEDO_TABLES if r.search(line)), "")

    hits = _hits(check)
    assert hits == [], "\n".join(hits)


def test_no_catpa_hydraulic_parameter_block() -> None:
    def check(line: str) -> str:
        return next((r.pattern for r in CATPA_VALUES if r.search(line)), "")

    hits = _hits(check)
    assert hits == [], "\n".join(hits)


@pytest.mark.parametrize(
    ("line", "found"),
    [
        (_j("from agrijax.processes.", "snow import X"), True),
        (_j("hb = 14.", "6545"), True),
        (_j("[0.22, 0.26, 0.36, 0.17, 0.", "322]"), True),
        (_j("  U = ", "MAX(U, UBREEZ)"), True),
        ("hb = 15.0, lam = 0.25", False),
    ],
)
def test_the_patterns_find_what_they_should(line: str, found: bool) -> None:
    hit = (
        any(m in line for m in REMOVED_MODULES)
        or any(f.lower() in line.lower() for f in SOURCE_FRAGMENTS)
        or any(r.search(line) for r in (*ALBEDO_TABLES, *CATPA_VALUES))
    )
    assert hit is found
