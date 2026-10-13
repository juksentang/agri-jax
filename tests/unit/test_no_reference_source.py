"""The public tree holds only public material.

Two kinds of checks scan every tracked file (``git ls-files``; a directory walk without git):

* **structure** (no data, runs everywhere): no module path or dotted name of the modules kept
  outside this repository, no fragment of a reference model's source or run-time log, no removed
  constant name, no comment that cites a line of a scenario input file as the source of a value,
  and no prose that describes the management of a site scenario. The patterns are assembled
  from parts so that this file does not match itself.
* **values** (needs the user's local data): the values of the user's local site scenario (dates
  in every common notation, amounts, densities, cultivar codes and calibrated cultivar
  coefficients, grid and soil sequences, distinctive single values such as the site latitude) are derived at test time from the local data and compared, as HMAC-SHA256 digests of
  normalised tokens under a key kept outside the repository, with the same digests of the tokens
  found in the tree. Neither the values nor their digests are stored in this repository, and a
  failure reports a file, a line and a token kind only. Without the key or the data the check
  skips (it is marked ``allow_skip``).
"""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import re
import subprocess
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
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


# ============================================================================ structure
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

#: removed constants (names only)
REMOVED_NAMES: tuple[str, ...] = (_j("CATPA_MAIZE", "_SEASONS"),)

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

#: the PRMS albedo DATA tables of the reference source (accumulation and melt stages)
_SEP = r"[^0-9]{1,12}"
ALBEDO_TABLES: tuple[re.Pattern[str], ...] = (
    re.compile(_SEP.join((r"0\.80?", r"0\.77", r"0\.75", r"0\.72", r"0\.70?", r"0\.69"))),
    re.compile(_SEP.join((r"0\.72", r"0\.65", r"0\.60?", r"0\.58", r"0\.56", r"0\.54"))),
)

_SCENARIO_FILE = r"(?:rzwqm\.dat|MZDSSAT\.RZX|expdata\.dat)"
#: a comment that gives a value and cites a scenario input file (or a line of it) as its source
SOURCED_COMMENTS: tuple[re.Pattern[str], ...] = (
    re.compile(_SCENARIO_FILE + r"\s*(?:line\s*|:\s*)\d+", re.IGNORECASE),
    re.compile(r"=\s*[\[(]?\s*-?\d[\d.,\s\-eE]*[\])]?\s*#.*" + _SCENARIO_FILE, re.IGNORECASE),
)

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
#: prose that describes the management of a site scenario
DESCRIPTION: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:sown|planted|harvested|fertili[sz]ed|tilled|sprayed)\s+(?:on|in|around)\s+"
        r"(?:the\s+)?(?:early\s+|mid-?\s*|late\s+)?(?:\d{1,2}\s+)?" + _MONTH + r"(?![a-z])",
        re.IGNORECASE,
    ),
)
#: management words that describe the scenario when they stand on a line that names the site
_SITE = re.compile(r"CA-?TPA|\bcatpa\b|turkey point", re.IGNORECASE)
_SITE_MANAGEMENT = re.compile(_j(r"\b(?:chisel ", r"plow|glypho", r"sate)\b"), re.IGNORECASE)


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
        if rel.startswith(_SKIP_PREFIXES) or p.suffix.lower() in _SKIP_SUFFIXES or not p.is_file():
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


ALL_FILES = _tracked_files()
#: the structure checks skip this file (its patterns are assembled from parts); the value check does not
FILES = [p for p in ALL_FILES if p.resolve() != SELF]


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


def _structure_hit(line: str) -> str:
    hit = next((m for m in (*REMOVED_MODULES, *REMOVED_NAMES) if m in line), "")
    if hit:
        return hit
    low = line.lower()
    hit = next((f for f in SOURCE_FRAGMENTS if f.lower() in low), "")
    if hit:
        return hit
    hit = next((r.pattern for r in (*ALBEDO_TABLES, *SOURCED_COMMENTS, *DESCRIPTION) if r.search(line)), "")
    if hit:
        return hit
    if _SITE.search(line) and _SITE_MANAGEMENT.search(line):
        return "site management description"
    return ""


def test_the_scan_sees_the_tree() -> None:
    names = {str(p.relative_to(REPO)) for p in FILES}
    assert "README.md" in names and "src/agrijax/__init__.py" in names
    assert len(FILES) > 100 and len(ALL_FILES) == len(FILES) + 1


def test_no_removed_module_or_name() -> None:
    def check(line: str) -> str:
        return next((m for m in (*REMOVED_MODULES, *REMOVED_NAMES) if m in line), "")

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


def test_no_value_sourced_from_a_scenario_file() -> None:
    def check(line: str) -> str:
        return next((r.pattern for r in SOURCED_COMMENTS if r.search(line)), "")

    hits = _hits(check)
    assert hits == [], "\n".join(hits)


def test_no_scenario_description() -> None:
    def check(line: str) -> str:
        hit = next((r.pattern for r in DESCRIPTION if r.search(line)), "")
        if hit:
            return hit
        return "site management description" if _SITE.search(line) and _SITE_MANAGEMENT.search(line) else ""

    hits = _hits(check)
    assert hits == [], "\n".join(hits)


@pytest.mark.parametrize(
    ("line", "found"),
    [
        (_j("from agrijax.processes.", "snow import X"), True),
        (_j("  U = ", "MAX(U, UBREEZ)"), True),
        (_j("from agrijax.io.ameriflux import CATPA_MAIZE", "_SEASONS"), True),
        (_j("hb = 9.5  # horizon 1, rzwqm", ".dat"), True),
        (_j("see rzwqm", ".dat:1234 for the record"), True),
        ("the rzwqm.dat reader returns the horizon records", False),
        (_j("the crop is sown on 3 ", "May every year"), True),
        (_j("CA-TPA: a chisel ", "plow pass in spring"), True),
        ("tillage implement names: chisel plow-straight, field cultivator", False),
        ("hb = 15.0, lam = 0.25", False),
    ],
)
def test_the_structure_patterns_find_what_they_should(line: str, found: bool) -> None:
    assert bool(_structure_hit(line)) is found


# ============================================================================ values
#: the secret key of the value check (outside the repository, next to the local scenario data)
KEY_NAME = "public_guard.key"
#: years of the scenario management whose plantings and harvests are covered
MANAGEMENT_YEARS = (2002, 2021)
#: years of the seasons counted in prose ("<n> seasons"): the reference comparisons
SEASON_COUNT_WINDOWS = ((2015, 2021), (2002, 2021))
#: lengths of the windows of the node-grid sequences
GRID_WINDOWS = (5, 6)
#: lengths of the windows of the per-horizon sequences
HORIZON_WINDOWS = (4, 5)
#: significant digits from which a single soil value is distinctive
DISTINCT_DIGITS = 4
#: the cultivar coefficients of the CERES-Maize cultivar file, in file order
CULTIVAR_COEFFICIENTS = ("P1", "P2", "P5", "G2", "G3", "PHINT")
#: labelled calibrated coefficients of the scenario in one file from which the file is flagged
COEFFICIENT_HITS = 2
#: the public DSSAT-CSM cultivar file of the tree: a scenario coefficient equal to its value for
#: the same cultivar is public and not checked
PUBLIC_CUL = REPO / "tests" / "fixtures" / "dssat" / "MZCER048.CUL"
_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _g(x: float) -> str:
    """A number in a canonical spelling (15, 15.0 and 1.5e1 all give ``15``)."""
    return f"{float(x):.10g}"


def _iso(y: int, m: int, d: int) -> str | None:
    try:
        return str(np.datetime64(f"{y:04d}-{m:02d}-{d:02d}", "D"))
    except ValueError:
        return None


def _from_doy(y: int, doy: int) -> str | None:
    if not (1900 <= y <= 2100 and 1 <= doy <= 366):
        return None
    day = np.datetime64(f"{y:04d}-01-01", "D") + np.timedelta64(doy - 1, "D")
    return str(day) if int(str(day)[:4]) == y else None


_ORD = r"(?:st|nd|rd|th)?"


def _yy(y: str) -> int:
    return int(y) + (2000 if len(y) == 2 else 0)


def _mon(s: str) -> int:
    return _MONTHS.index(s[:3].lower()) + 1


_DATE_FORMS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], Iterable[str | None]]], ...] = (
    # YYYY-MM-DD, YYYY-M-D, YYYY/MM/DD, YYYY.MM.DD
    (
        re.compile(r"(?<![\d.])(\d{4})([-/.])(\d{1,2})\2(\d{1,2})(?![\d])"),
        lambda m: [_iso(int(m[1]), int(m[3]), int(m[4]))],
    ),
    # YYYYMMDD
    (re.compile(r"(?<![\d.])(\d{4})(\d{2})(\d{2})(?![\d.])"), lambda m: [_iso(*map(int, m.groups()))]),
    # MON DD YYYY, Month DDth, YYYY, Mon-DD-YYYY
    (
        re.compile(r"\b(" + _MONTH + r")[\s-]+(\d{1,2})" + _ORD + r",?[\s-]+(\d{4})\b", re.IGNORECASE),
        lambda m: [_iso(int(m[3]), _mon(m[1]), int(m[2]))],
    ),
    # DD MON YYYY, DD-Mon-YYYY, DD-MON-YY
    (
        re.compile(r"\b(\d{1,2})" + _ORD + r"[\s-]+(" + _MONTH + r"),?[\s-]+(\d{4}|\d{2})\b", re.IGNORECASE),
        lambda m: [_iso(_yy(m[3]), _mon(m[2]), int(m[1]))],
    ),
    # DD/MM/YYYY and MM/DD/YYYY (both read), DD-MM-YYYY, DD.MM.YYYY
    (
        re.compile(r"(?<![\d.])(\d{1,2})([-/.])\s*(\d{1,2})\2(\d{4})(?![\d])"),
        lambda m: [_iso(int(m[4]), int(m[3]), int(m[1])), _iso(int(m[4]), int(m[1]), int(m[3]))],
    ),
    # DD MM YYYY separated by blanks (the management records of rzwqm.dat), both orders read
    (
        re.compile(r"(?<![\d.])(\d{1,2})[ \t]+(\d{1,2})[ \t]+(\d{4})(?![\d.])"),
        lambda m: [_iso(int(m[3]), int(m[2]), int(m[1])), _iso(int(m[3]), int(m[1]), int(m[2]))],
    ),
    # (YYYY, M, D) and (YYYY, DOY) tuples
    (
        re.compile(r"\(\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)"),
        lambda m: [_iso(*map(int, m.groups()))],
    ),
    (re.compile(r"\(\s*(\d{4})\s*,\s*(\d{1,3})\s*\)"), lambda m: [_from_doy(int(m[1]), int(m[2]))]),
    # YYYY.DDD, YYYYDDD, YYDDD (DSSAT)
    (re.compile(r"(?<![\d.])(\d{4})\.(\d{3})(?!\d)"), lambda m: [_from_doy(int(m[1]), int(m[2]))]),
    (re.compile(r"(?<![\d.])(\d{4})(\d{3})(?![\d.])"), lambda m: [_from_doy(int(m[1]), int(m[2]))]),
    (re.compile(r"(?<![\d.])(\d{2})(\d{3})(?![\d.])"), lambda m: [_from_doy(2000 + int(m[1]), int(m[2]))]),
)
_AMOUNT = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*kg\s*(?:of\s+)?(?:n|no3|nh4|urea|nitrogen)?(?:-n)?\s*"
    r"(?:/\s*ha\b|ha-1|ha⁻¹|ha\^-1|per\s+ha)",
    re.IGNORECASE,
)
_AREA = r"(ha-1|ha⁻¹|ha\b|m-2|m⁻²|m2|m²|square met(?:re|er))"
_DENSITY = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:seeds|plants)\s*(?:/\s*|per\s+)?" + _AREA, re.IGNORECASE
)
#: the unit before the number (``PLANTS/m2 : 6.5`` as ``MANAGE.OUT`` prints it)
_DENSITY_UNIT_FIRST = re.compile(
    r"\b(?:seeds|plants)\s*(?:/\s*|per\s+)" + _AREA + r"\s*[:=]?\s*(\d+(?:\.\d+)?)(?![\d.])", re.IGNORECASE
)
_COUNT = re.compile(
    r"(?<![\d.])(\d+)\s+(?:(?:CA-?TPA|catpa|RZWQM2?|scenario|site|reference|maize|corn|growing)[\s()-]*){0,4}"
    r"(plantings|seasons)\b",
    re.IGNORECASE,
)
#: a labelled cultivar coefficient (``P1 : 210.00``, ``"G2": 690.0``, ``PHINT=41.5``)
_COEFFICIENT = re.compile(
    r"\b(" + "|".join(CULTIVAR_COEFFICIENTS) + r")\b[\"']?\s*[:=]\s*[\"']?(" + _NUM + r")(?![\w.])"
)
_CODE = re.compile(r"\b[A-Z]{2}\d{4}\b")
_STOCK_NAME = re.compile(r"\bPIO\s*\d{3,4}\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])" + _NUM + r"(?![\w.])")
#: what may stand between two values of one sequence (an array, a list, a slash-separated series)
_GAP = re.compile(r"[\s,;/\[\]()]*")
#: joins of adjacent string literals ("a" "b", "a" + "b", _j("a", "b")): the value check also reads
#: the text with them removed, so a value split into parts is found
_LITERAL_JOIN = re.compile(r"""["']\s*(?:,|\+)?\s*["']""")


def _sig_digits(tok: str) -> int:
    mant = tok.lower().split("e")[0].lstrip("-").replace(".", "").lstrip("0")
    return len(mant.rstrip("0")) if "." not in tok else len(mant)


def _stock_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.upper().replace("PIO", "PIO ")).strip()


def _line_tokens(line: str, site_context: bool) -> Iterator[tuple[str, str]]:
    """``(kind, canonical token)`` of every candidate value on one line."""
    for rx, conv in _DATE_FORMS:
        for m in rx.finditer(line):
            for d in conv(m):
                if d is not None:
                    yield "date", "D:" + d
    for m in _AMOUNT.finditer(line):
        yield "amount", "F:" + _g(float(m[1]))
    for v, unit in [(m[1], m[2]) for m in _DENSITY.finditer(line)] + [
        (m[2], m[1]) for m in _DENSITY_UNIT_FIRST.finditer(line)
    ]:
        yield "density", "P:" + _g(float(v) / 1.0e4 if unit.lower().startswith("ha") else float(v))
    for m in _COEFFICIENT.finditer(line):
        yield "cultivar coefficient", f"K:{m[1]}={_g(float(m[2]))}"
    for m in _COUNT.finditer(line):
        if m[2].lower() == "plantings" or _SITE.search(line):
            yield "count", "N:" + m[1]
    for m in _CODE.finditer(line):
        yield "cultivar", "C:" + m[0]
        if site_context:
            yield "cultivar", "S:" + m[0]
    if site_context:
        for m in _STOCK_NAME.finditer(line):
            yield "cultivar", "S:" + _stock_name(m[0])
    for m in _NUMBER.finditer(line):
        if _sig_digits(m[0]) >= DISTINCT_DIGITS:
            yield "soil value", "V:" + _g(float(m[0]))


def _sequences(text: str) -> Iterator[tuple[int, list[float]]]:
    """``(line, values)`` of every run of numbers separated only by spaces, commas, slashes or brackets."""
    run: list[float] = []
    start, last_end = 1, -1
    for m in _NUMBER.finditer(text):
        if run and _GAP.fullmatch(text, last_end, m.start()) is None:
            yield start, run
            run = []
        if not run:
            start = text.count("\n", 0, m.start()) + 1
        run.append(float(m[0]))
        last_end = m.end()
    if run:
        yield start, run


def _windows(values: list[float], k: int) -> Iterator[str]:
    for i in range(len(values) - k + 1):
        w = np.asarray(values[i : i + k], dtype=float)
        d = np.diff(w)
        if np.allclose(d, d[0], rtol=0.0, atol=1e-9):
            continue  # constant or evenly spaced: no information
        yield "A:" + ",".join(_g(x) for x in w)


def _file_tokens(text: str) -> Iterator[tuple[int, str, str]]:
    """``(line, kind, token)`` of a file's text."""
    site = bool(_SITE.search(text))
    for i, line in enumerate(text.splitlines(), start=1):
        for kind, tok in _line_tokens(line, site):
            yield i, kind, tok
    for line_no, values in _sequences(text):
        for k in sorted({*GRID_WINDOWS, *HORIZON_WINDOWS}):
            for tok in _windows(values, k):
                yield line_no, "sequence", tok


def _scenario_tokens(scenario: Path) -> set[str]:
    """The canonical tokens of the local scenario's values (derived from the local files)."""
    from agrijax.io.catpa import build_events
    from agrijax.io.rzwqm.dat import read_rzwqm_dat
    from agrijax.sites.catpa_dssat import genotype_sets

    dat = read_rzwqm_dat(scenario / "rzwqm.dat")
    lo, hi = MANAGEMENT_YEARS
    ev = build_events(dat, f"{lo}-01-01", f"{hi}-12-31")
    kind = ev["event"].astype(str)
    toks: set[str] = set()
    for d in ev.loc[kind.isin(["planting", "harvest"]), "date"]:
        toks.add("D:" + str(np.datetime64(d, "D")))
    for v in ev.loc[kind.str.startswith("fertilizer"), "value"]:
        toks.add("F:" + _g(v))
    for v in ev.loc[kind == "planting", "value"]:
        toks.add("P:" + _g(float(v) / 1.0e4))
    sow_years = np.asarray([int(str(np.datetime64(d, "Y"))) for d in ev.loc[kind == "planting", "date"]])
    for a, b in SEASON_COUNT_WINDOWS:
        toks.add("N:" + str(int(np.sum((sow_years >= a) & (sow_years <= b)))))
    sets = genotype_sets(dat)
    toks.add("C:" + sets["catpa"][0])
    toks.add("S:" + sets["stock"][0])
    toks.add("S:" + _stock_name(sets["stock"][1]))
    for seq in (dat.node_depths_cm, dat.node_spacing_cm[:-1]):
        for k in GRID_WINDOWS:
            toks.update(_windows([float(x) for x in seq], k))
    toks.update(_cultivar_tokens(scenario, sets["stock"][0]))
    lat = float(dat.physiography["latitude_rad"])
    toks.update("V:" + _g(round(x, n)) for x in (lat, np.degrees(lat)) for n in (4, 5, 6))
    for seq in (dat.horizon_depths_cm, *dat.hydraulics.values()):
        vals = [float(x) for x in np.ravel(seq)]
        for k in HORIZON_WINDOWS:
            toks.update(_windows(vals, k))
        for x in vals:
            if _sig_digits(_g(x)) >= DISTINCT_DIGITS:
                toks.add("V:" + _g(x))
    return toks


def _cultivar_tokens(scenario: Path, varno: str) -> set[str]:
    """The scenario's cultivar coefficients (its own ``MZCER040.CUL``): labelled single values where
    they differ from the public DSSAT-CSM row of the same cultivar, and windows of the row."""
    from agrijax.io.dssat.genotype import read_cul

    cul = scenario / "MZCER040.CUL"
    if not cul.is_file():
        return set()
    row = read_cul(cul).loc[varno]
    public = read_cul(PUBLIC_CUL)
    pub = public.loc[varno] if varno in public.index else None
    vals = [float(row[k]) for k in CULTIVAR_COEFFICIENTS]
    toks = {
        f"K:{k}={_g(v)}"
        for k, v in zip(CULTIVAR_COEFFICIENTS, vals, strict=True)
        if pub is None or not np.isclose(float(pub[k]), v, rtol=0.0, atol=1e-9)
    }
    for k in HORIZON_WINDOWS:
        toks.update(_windows(vals, k))
    return toks


def _digest(key: bytes, token: str) -> str:
    return hmac.new(key, token.encode("utf-8"), hashlib.sha256).hexdigest()


def _local_inputs() -> tuple[bytes, Path]:
    from agrijax.io.catpa import CATPA_DIR, SCENARIO_DIR

    key_path = CATPA_DIR / KEY_NAME
    if not key_path.is_file():
        pytest.skip(f"{key_path} not found (the value check runs where the key and the local data are)")
    if not (SCENARIO_DIR / "rzwqm.dat").is_file():
        pytest.skip(f"{SCENARIO_DIR / 'rzwqm.dat'} not found (local scenario data)")
    key = key_path.read_bytes().strip()
    if len(key) < 32:
        pytest.fail(f"{key_path}: the key is shorter than 32 bytes")
    return key, SCENARIO_DIR


@pytest.mark.allow_skip(reason="needs the key and the local scenario data, kept outside the repository")
def test_no_scenario_value_in_the_tree() -> None:
    key, scenario = _local_inputs()
    secret = {_digest(key, t) for t in _scenario_tokens(scenario)}
    assert len(secret) > 50  # dates, grid and soil sequences: a sanity bound on what is derived
    hits: set[str] = set()
    for p in ALL_FILES:
        text = _text(p)
        coefficient_hits: dict[str, str] = {}
        for variant in (text, _LITERAL_JOIN.sub("", text)):
            for line, kind, tok in _file_tokens(variant):
                h = _digest(key, tok)
                if h in secret:
                    msg = f"{p.relative_to(REPO)}:{line}: {kind} of the local scenario (digest {h[:12]})"
                    if tok.startswith("K:"):
                        coefficient_hits.setdefault(tok.split("=")[0], msg)
                    else:
                        hits.add(msg)
        # one labelled coefficient alone is common among cultivars (G3 = 8.5): a file is flagged
        # when it holds COEFFICIENT_HITS of the scenario's calibrated coefficients
        if len(coefficient_hits) >= COEFFICIENT_HITS:
            hits.update(coefficient_hits.values())
    assert not hits, "\n".join(sorted(hits))


def test_the_value_tokens_cover_every_notation() -> None:
    """The tokenizer finds a date in each notation and canonicalises it; synthetic values only."""
    forms = (
        "2013-05-14",
        "MAY 14 2013",
        "May 14, 2013",
        "14 May 2013",
        "14/5/2013",
        "14/ 5/2013",
        "date(2013, 5, 14)",
        "2013.134",
        "2013134",
        "13134",
        _j('"2013-0', '5-14"'),
    )
    for f in forms:
        text = _LITERAL_JOIN.sub("", f)
        toks = {t for _, k, t in _file_tokens(text) if k == "date"}
        assert "D:2013-05-14" in toks, f
    for f in ("17 Jan 2013", "17-Jan-2013", "17-JAN-13", "January 17th, 2013", "Jan-17-2013", "20130117",
              "2013/01/17", "2013-1-17", "17/01/2013", "17-01-2013", "  1  17   1  2013  76.0",
              "(2013, 17)"):  # fmt: skip
        toks = {t for _, k, t in _file_tokens(f) if k == "date"}
        assert "D:2013-01-17" in toks, f
    toks = {
        t for _, _, t in _file_tokens("one 95 kg N ha-1 dose, 6.5 plants/m2, 65000 seeds/ha, 11 plantings")
    }
    assert {"F:95", "P:6.5", "N:11"} <= toks
    for f in ("72000 seeds ha-1", "7.2 plants m-2", "PLANTS/m2 : 7.2", "7.2 plants per m2"):
        assert "P:7.2" in {t for _, k, t in _file_tokens(f) if k == "density"}, f
    assert "N:9" in {t for _, _, t in _file_tokens("the 9 CA-TPA seasons")}
    cul = {t for _, k, t in _file_tokens(' P1     : 210.00  P2     : 0.4500\n"G2": 690.0') if k != "sequence"}
    assert {"K:P1=210", "K:P2=0.45", "K:G2=690"} <= cul
    seq = {t for _, k, t in _file_tokens("z = [1, 3, 6, 9, 13,\n     16, 22]") if k == "sequence"}
    assert "A:1,3,6,9,13,16" in seq and "A:3,6,9,13,16,22" in seq and "A:1,3,6,9,13" in seq
    assert not {t for _, k, t in _file_tokens("[5, 10, 15, 20, 25, 30, 35]") if k == "sequence"}
    site = {t for _, _, t in _file_tokens("CA-TPA\nXY0001 PIO 1234")}
    assert {"S:XY0001", "S:PIO 1234", "C:XY0001"} <= site
    assert "S:XY0001" not in {t for _, _, t in _file_tokens("XY0001 PIO 1234")}
