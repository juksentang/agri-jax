"""The public tree names no internal work item.

Docstrings, comments, messages and scripts say what a thing is or what was measured, not which work
line, branch, private document or review it came from: a reader outside the project cannot resolve
those names. This test scans the text files of the repository (``src/``, ``tests/``, ``scripts/``,
``tutorial/``, ``examples/``, ``poc/``, ``docs/``, ``.github/`` and the top-level READMEs and
metadata) for the clear patterns of such references and fails with the ``file:line`` list:

* branch names (``wt/...``),
* numbers of private documents and plans (``docs/15``, ``doc 05``, ``plan 19``, ``§9.22``),
* work-item and stage ids (``D3-1``, ``D2-1a``, ``U1-B``, ``G0b``, ``gap G24``),
* review provenance (``independent review``, ``review round 2``, ``private archive``), the
  tracker, and the contract's private decision numbers (``M3 contract decision 11``).

It reads files only (no data, no JAX); the files are the tracked ones when ``git`` can list them,
else the checkout's own files without the untracked working directories. A reference that cannot be
reworded (for example a string pinned against stored data) goes in :data:`ALLOWED` with the reason.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

#: directories (and top-level files) that are scanned
SCAN_DIRS: tuple[str, ...] = ("src", "tests", "scripts", "tutorial", "examples", "poc", "docs", ".github")
SCAN_TOP: tuple[str, ...] = (
    "README.md",
    "README.zh.md",
    "CONTRIBUTING.md",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    ".gitignore",
)
TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".sh", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".html", ".svg", ".ipynb", ".cff"}
)
#: not scanned: caches and environments, the reference data of the tests, and the working directories that
#: are not part of the public tree (they matter for the fallback walk of a checkout without ``git``;
#: ``git ls-files`` never lists them)
SKIP_PARTS = frozenset({"__pycache__", ".git", ".venv", "venv"})
SKIP_PREFIXES = (
    "tests/fixtures/",
    "docs/internal/",
    "docs/validation/",
    "scripts/rorqual/",
    "scripts/narval/",
)
MAX_BYTES = 3_000_000

#: pattern name -> regular expression of the references that must not appear
PATTERNS: dict[str, re.Pattern[str]] = {
    "branch name": re.compile(r"\bwt/[a-z][a-z0-9_]*"),
    "private document number": re.compile(
        r"\bdocs?/[0-9]{2}(?![0-9])|\b(?:doc|docs|plan) [0-9]{2}(?![0-9])|§|\bsection 9\.[0-9]"
    ),
    "work-item or stage id": re.compile(r"\bD[2-5]-[0-9][a-z]?\b|\bU1-[AB]\b|\bG0b?\b|\bgaps? G[0-9]+\b"),
    "review provenance": re.compile(
        r"independent review|\breview round [0-9]|\busability review\b|\bprivate archive\b|\bprivate repo\b"
    ),
    "tracker or handover": re.compile(r"\b(?:the|our|project) tracker\b|TRACKER\.md|\bhandover\b"),
    "private contract decision": re.compile(r"\bM3 contract (?:decision|section)\b"),
}

#: ``(path relative to the repository, pattern name or "*") -> reason`` of the references that stay
ALLOWED: dict[tuple[str, str], str] = {
    ("tests/unit/test_no_internal_references.py", "*"): "this file spells the patterns it looks for",
    (
        "tests/integration/ceres_baseline.py",
        "private document number",
    ): "the HISTORY text is stored in the baseline snapshot's manifest and compared with it by "
    "test_ceres_baseline.test_manifest; it is reworded when the snapshot is regenerated",
}


def _tracked() -> list[str] | None:
    """The tracked files (repository-relative, ``/``-separated), or ``None`` without a usable ``git``."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
    return names or None


def _walked() -> list[str]:
    """The files under the scanned directories of a checkout (the fallback of :func:`_tracked`)."""
    names: list[str] = [n for n in SCAN_TOP if (ROOT / n).is_file()]
    for d in SCAN_DIRS:
        base = ROOT / d
        if base.is_dir():
            names += [p.relative_to(ROOT).as_posix() for p in sorted(base.rglob("*")) if p.is_file()]
    return names


def _selected(rel: str) -> bool:
    path = Path(rel)
    if rel.startswith(SKIP_PREFIXES) or SKIP_PARTS & set(path.parts[:-1]):
        return False
    if "/" not in rel:
        return rel in SCAN_TOP or rel.startswith("README")
    return path.parts[0] in SCAN_DIRS and path.suffix in TEXT_SUFFIXES


def scanned_files() -> list[Path]:
    """The text files the scan reads, sorted."""
    names = _tracked()
    if names is None:
        names = _walked()
    files = [ROOT / n for n in sorted(set(names)) if _selected(n)]
    return [p for p in files if p.is_file() and p.stat().st_size <= MAX_BYTES]


def scan_text(text: str) -> Iterator[tuple[int, str, str]]:
    """``(line number, pattern name, matched text)`` of every reference in ``text``."""
    for i, line in enumerate(text.splitlines(), start=1):
        for name, pat in PATTERNS.items():
            for m in pat.finditer(line):
                yield i, name, m.group(0)


def _allowed(rel: str, name: str) -> bool:
    return (rel, name) in ALLOWED or (rel, "*") in ALLOWED


def test_the_public_tree_names_no_internal_work_item() -> None:
    found: list[str] = []
    files = scanned_files()
    assert len(files) > 100, "the scan found almost no file: is the repository layout as expected?"
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for line, name, matched in scan_text(text):
            if not _allowed(rel, name):
                found.append(f"{rel}:{line}: {name}: {matched!r}")
    assert not found, (
        "internal references in the public tree (say what the thing is or what was measured instead):\n"
        + "\n".join(found)
    )


@pytest.mark.parametrize(
    ("sample", "name"),
    [
        ("# before wt/grad_gap the step was compared with itself", "branch name"),
        ("see docs/15 for the plan", "private document number"),
        ("(doc 05 section 4)", "private document number"),
        ("plan 19 A3/A4", "private document number"),
        ("closed in §9.22", "private document number"),
        ("D3-1: staged calibration", "work-item or stage id"),
        ("as the D2-1a acceptance runs it", "work-item or stage id"),
        ("the D4-1 rates", "work-item or stage id"),
        ("U1-B", "work-item or stage id"),
        ("G0b", "work-item or stage id"),
        ("until gap G24 is closed", "work-item or stage id"),
        ("(independent review of this branch)", "review provenance"),
        ("review round 2", "review provenance"),
        ("kept in the private archive", "review provenance"),
        ("the tracker lists it", "tracker or handover"),
        ("(M3 contract decision 11)", "private contract decision"),
    ],
)
def test_the_patterns_catch_what_they_are_for(sample: str, name: str) -> None:
    assert name in {n for _, n, _ in scan_text(sample)}, (sample, name)


@pytest.mark.parametrize(
    "sample",
    [
        "tracker.record(rows, full, z)  # a bench script's own object",
        "tracker = Tracker(r_n, T_START)",
        "see docs/calibration.md and docs/debugging.md",
        "Product wt (kg dm/ha;no loss)",
        "G2 and G3 lie along a curved valley; P1, P2, P5 and PHINT stay derivative-free",
        "the 58 M2 treatments, the M1 replay and the M3 day",
        "US-S2 and CA-TPA, the D1 row of the day table",
        "the AJ011 findings, level 2 of the trust test, section 4 of the paper",
        "12 doc strings and 20 docs of the plan: planting date 2015-05-01",
        "ste +0.117, exact -0.205 on grain weight (1982 -14 days)",
    ],
)
def test_the_patterns_let_ordinary_text_through(sample: str) -> None:
    assert list(scan_text(sample)) == [], sample


def test_every_allowlist_entry_still_applies() -> None:
    """A stale entry (the reference was reworded, or the file is gone) is removed, not kept."""
    for (rel, name), reason in ALLOWED.items():
        assert reason.strip(), (rel, name)
        path = ROOT / rel
        assert path.is_file(), f"{rel}: allowlisted file is missing"
        if name == "*":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        assert any(n == name for _, n, _ in scan_text(text)), f"{rel}: no {name!r} left, drop the entry"


def test_the_scan_leaves_out_what_is_not_public() -> None:
    rels = {p.relative_to(ROOT).as_posix() for p in scanned_files()}
    assert not [r for r in rels if r.startswith(("scripts/rorqual/", "docs/internal/", "tests/fixtures/"))]
    assert {"README.md", "src/agrijax/core/lint.py", "tests/conftest.py"} <= rels
    assert not _selected("docs/internal/36_note.md") and not _selected("scripts/rorqual/env.sh")
    assert _selected("scripts/bench/calib/d3_1_staged.py") and _selected(".gitignore")
    assert not _selected("tests/fixtures/dssat/MZCER048.CUL") and not _selected("uv.lock")
