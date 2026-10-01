"""DSSAT observed-data files: FILEA (``*.MZA``, season-end values) and FILET (``*.MZT``, time series).

**FILEA** holds one row per treatment: ``@TRNO  HWAM  CWAM  ADAT  MDAT ...``. It is read the way
DSSAT-CSM v4.8.6.0 CERES-Maize reads it: ``MZ_OPHARV`` calls ``READA_Y4K``
(``Plant/CERES-Maize/MZ_OPHARV.for:340-341``, ``Utilities/READS.for:1176-1323``), whose rule is
restated here:

* only the **first** ``@`` line is a header; its columns are those of ``PARSE_HEADERS``
  (:func:`~agrijax.io.dssat._fixed.dssat_header_spans`), at most ``MAXCOLN = 40`` of them, the
  names upper-cased (and at most 15 characters, ``CHARACTER*15``);
* header aliases: ``R5AT`` and ``PDFD`` are ``PDFT``, ``BWAH`` is ``BWAM``, ``HWAH`` is ``HWAM``,
  ``CWAH`` is ``CWAM``, ``HDAT`` is ``R8AT`` (:data:`FILEA_ALIASES`); when two columns carry the
  same (aliased) name, **the last one wins** (the engine's column loop overwrites);
* data lines are the lines ``IGNORE`` returns: not blank and not starting with ``!`` or ``@``;
  later ``@`` tables are therefore read **with the first table's columns**; the first line
  whose ``TRNO`` field (``C255(C1:C2+1)`` read ``(2X,I4)``) equals a treatment number is that
  treatment's row;
* the value of column ``J`` is the text between the header columns around it (``C2D(J-1)`` to
  ``C1D(J+1)``; one column further right after ``TRNO``; to ``C2D(J)+1`` for the last column),
  its first 12 characters (``A12``), kept as text: :attr:`FileA.text`. There is no numeric test
  and no ``TNAM`` special case.

:attr:`FileA.table` holds the numbers of that text as the engine's ``F12.0`` read of a date
(``READA_Dates``) takes them: blanks ignored, NaN when the text is not a number, ``-99`` NaN.

**Where ``READA_Y4K`` differs from this reader.** Four pathological FILEA layouts make the engine
do something its own rule above does not describe; none occurs in the DSSAT example tree, and the
reader keeps the rule instead of imitating them:

* *duplicate* ``TRNO`` *columns*: the engine takes a row when **any** ``TRNO`` column equals the
  treatment number; the reader lets the last ``TRNO`` column that reads as an integer decide;
* *a tab in the header line*: ``PARSE_HEADERS`` splits only at blanks, so two names joined by a tab
  are one column and the engine may find no ``TRNO`` column, hence no row for any treatment (all
  values missing); the reader, like the ``*.OUT`` readers, reads every tab as a blank;
* *more than 1000 data lines before a treatment's row*: the engine looks at 1000 data lines at most
  and then keeps the 1000th whatever its ``TRNO``; the reader searches the whole file;
* *more than* ``MAXCOLN = 40`` *header columns*: ``PARSE_HEADERS`` fills arrays of 40 entries, so
  the result is undefined; the reader keeps the first 40 columns and lists the others in
  :attr:`FileA.ignored_columns`.

Date columns (``ADAT``, ``MDAT``, ``EDAT``, ``R8AT`` ...) stay the numbers of the file;
:meth:`FileA.date` converts one with :func:`reada_date` (``READA_Dates`` and ``Y4K_DOY``).

**FILET** holds one or more ``@TRNO  DATE  <codes>`` tables, each a time series per treatment.
CERES-Maize itself never reads it (it is the file of DSSAT's graphics tools), so there is no
engine reading rule to follow; the tables are parsed by the positions of their header tokens
(:mod:`agrijax.io.dssat._fixed`, as the ``*.OUT`` readers), which also handles the left-shifted
and overrunning values of hand-edited files. ``DATE`` is ``YYDDD`` or ``YYYYDDD`` (``Y4K_DOY``,
:func:`y4k_date`); rows without a date are dropped, rows whose date is invalid (day 0, day 366 of a
non-leap year, outside the ``FirstWeatherDate`` window) are left out and listed in
:attr:`FileT.invalid_dates`. The result is tidy: one row per (treatment,
date, variable) with a finite value; a (treatment, date, variable) observed more than once keeps
every row (the converter to calibration targets averages them).

Dates follow ``Y4K_DOY`` (``Utilities/DATES.for:124-175``): with ``first_weather`` (DSSAT's
``FirstWeatherDate``, the first date of a weather file with four-digit years) a ``YYDDD`` code is the
first date on or after it with those year digits, and must lie within 99 years of it; without it,
``YY <= 35`` is 20YY, else 19YY; ``YYYYDDD`` codes are taken as written. ``FirstWeatherDate`` counts
only when it is positive (DSSAT's value for "no such weather file" is ``-99``,
``Utilities/ModuleDefs.for:93``): ``-99`` and ``0`` give the cross-over rule. A day of year beyond the
length of its year (366 in a non-leap year) is an error, not a roll-over into the next year. The
simulation start of :func:`reada_date` is a date or ``YYYYDDD``; a ``YYDDD`` start is converted like
any other ``YYDDD`` code, as the engine does with the ``SDATE`` of the simulation controls
(``InputModule/IPSIM.for:205``), never read as a year 0082.

``-99`` (and any value in (-99.5, -99]) is missing everywhere and never appears in the numbers.
"""

from __future__ import annotations

import calendar
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ._fixed import dssat_header_spans, frame_from_rows, header_tokens, read_lines, split_fixed, to_float
from .wth import Y4K_CROSSOVER

__all__ = [
    "FILEA_ALIASES",
    "FILEA_MAXCOLN",
    "FileA",
    "FileT",
    "ObservedData",
    "find_observed_files",
    "read_filea",
    "read_filet",
    "read_observed",
    "reada_date",
    "y4k_date",
]

#: header aliases of ``READA_Y4K`` (``READS.for:1234-1239``)
FILEA_ALIASES = {
    "R5AT": "PDFT",
    "PDFD": "PDFT",
    "BWAH": "BWAM",
    "HWAH": "HWAM",
    "CWAH": "CWAM",
    "HDAT": "R8AT",
}
#: most header columns ``READA_Y4K`` handles (``MAXCOLN``, ``READS.for:1199``)
FILEA_MAXCOLN = 40
#: the record length of ``READA_Y4K`` (``CHARACTER*255 C255``)
_RECORD = 255
#: characters of one value (``READ (...,'(A12)') DAT``)
_VALUE_WIDTH = 12
#: length of a header name (``CHARACTER*15 HEADER``)
_HEADER_WIDTH = 15
#: ``TRNO`` is read ``(2X,I4)``
_TRNO_SKIP, _TRNO_WIDTH = 2, 4
#: ``Y4K_DOY`` handles ``YYDDD`` codes up to this value (``DATES.for:128``)
_YYDDD_MAX = 99366
#: a ``Y4K_DOY`` date must lie within this many ``YYYYDDD`` units after ``FirstWeatherDate``
_Y4K_WINDOW = 99000
_CENTURY = 100000
_YEAR = 1000

_NAME = re.compile(r"^[A-Za-z#%][A-Za-z0-9#%_]*$")
# a Fortran ``F`` input item (blanks already removed): sign, digits with an optional point, exponent
_FREAL = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")
_INT = re.compile(r"^[+-]?\d+$")


def _title(lines: list[str]) -> str:
    for ln in lines:
        if ln.startswith("*"):
            return ln.split(":", 1)[1].strip() if ":" in ln else ln[1:].strip()
    return ""


def _is_data(ln: str) -> bool:
    """A FILET table row: not blank, not a comment, section, header or ``$`` line."""
    return bool(ln.strip()) and not ln.startswith(("!", "*", "@", "$")) and not ln.lstrip().startswith("!")


def _fread(text: str) -> float:
    """Fortran ``F`` read of a field (blanks ignored, ``BLANK='NULL'``); NaN when not a number or
    ``-99``, NaN for an all-blank field (which Fortran would read as 0)."""
    s = text.replace(" ", "")
    if not s or not _FREAL.match(s):
        return math.nan
    return to_float(s.replace("d", "e").replace("D", "e"))


def _date_of(yrdoy: int) -> date:
    year, doy = divmod(int(yrdoy), _YEAR)
    n = 366 if calendar.isleap(year) else 365
    if not 1 <= doy <= n:
        raise ValueError(f"date {yrdoy}: day of year {doy} outside 1..{n} of {year}")
    return date(year, 1, 1) + timedelta(days=doy - 1)


def _yrdoy(d: date) -> int:
    return d.year * _YEAR + d.timetuple().tm_yday


def _as_yrdoy(d: int | str | date) -> int:
    return _yrdoy(d) if isinstance(d, date) else int(float(d))


def y4k_date(code: int, first_weather: int | str | date | None = None) -> date:
    """``Y4K_DOY`` of a ``YYDDD`` / ``YYYYDDD`` code (``DATES.for:124-175``, see the module
    docstring); ``first_weather`` is ``FirstWeatherDate`` (``YYYYDDD`` or a date), used only when
    positive (``-99``, DSSAT's "none", and ``0`` give the cross-over rule)."""
    v = int(code)
    if v <= 0:
        raise ValueError(f"not a date: {code!r}")
    fw = None if first_weather is None else _as_yrdoy(first_weather)
    if fw is not None and fw > 0 and v <= _YYDDD_MAX:
        full = (fw // _CENTURY) * _CENTURY + v
        if full < fw:
            full = ((fw + _Y4K_WINDOW) // _CENTURY) * _CENTURY + v
        if full < fw or full > fw + _Y4K_WINDOW:
            raise ValueError(f"date {code} outside the 99 years after FirstWeatherDate {fw}")
        return _date_of(full)
    if v < _YYDDD_MAX:  # YYDDD with the cross-over year (``YRDOY .LE. 99365``)
        yy, doy = divmod(v, _YEAR)
        year = 2000 + yy if yy <= Y4K_CROSSOVER else 1900 + yy
        return _date_of(year * _YEAR + doy)
    return _date_of(v)


def _sim_yrdoy(sim_start: int | str | date, first_weather: int | str | date | None) -> int:
    """``YRSIM`` (``YYYYDDD``) of a simulation start: a date or ``YYYYDDD`` as given; a ``YYDDD`` start
    goes through :func:`y4k_date` like the ``SDATE`` the engine reads (``InputModule/IPSIM.for:205``)."""
    if isinstance(sim_start, date):
        return _yrdoy(sim_start)
    v = _as_yrdoy(sim_start)
    return _yrdoy(y4k_date(v, first_weather)) if 0 < v <= _YYDDD_MAX else v


def reada_date(
    value: float, sim_start: int | str | date, first_weather: int | str | date | None = None
) -> date | None:
    """Calendar date of an observed FILEA date as ``READA_Dates`` (``READS.for:952-986``): the value
    truncated to an integer (``0 < value < 1`` becomes 0); ``None`` when not positive; below 1000 a
    day of year in the year of ``sim_start`` (``YRSIM``: a date, ``YYYYDDD``, or ``YYDDD`` by
    :func:`y4k_date`) when later than its day of year, else in the next year; otherwise
    :func:`y4k_date`."""
    if not math.isfinite(value):
        return None
    idat = int(value)
    if idat <= 0:
        return None
    if idat < _YEAR:
        yrsim = _sim_yrdoy(sim_start, first_weather)
        yr, isim = divmod(yrsim, _YEAR)
        return _date_of((yr if idat > isim else yr + 1) * _YEAR + idat)
    return y4k_date(idat, first_weather)


@dataclass(frozen=True)
class FileA:
    """A parsed FILEA (see the module docstring): ``text`` is the engine's ``X`` text per treatment
    and code, ``table`` its numbers (NaN = missing); both indexed by ``TRNO``, one column per
    (aliased) header name."""

    path: Path
    title: str
    table: pd.DataFrame
    text: pd.DataFrame
    #: ``@`` lines in the file (only the first is a header; later tables use its columns)
    n_tables: int = 1
    #: ``{header name: alias}`` for the renamed columns (``HWAH`` -> ``HWAM`` ...)
    aliases: dict[str, str] = field(default_factory=dict)
    #: header columns beyond ``MAXCOLN`` (not read)
    ignored_columns: tuple[str, ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        """The variable codes of the file (header names that are codes; ``-99`` placeholders out)."""
        return tuple(c for c in self.table.columns if _NAME.match(str(c)))

    def value(self, trno: int, code: str) -> float:
        """Observed value of ``code`` for treatment ``trno`` (NaN when missing or absent)."""
        if trno not in self.table.index or code not in self.table.columns:
            return math.nan
        return float(self.table.at[trno, code])

    def date(
        self, trno: int, code: str, sim_start: int | str | date, first_weather: int | str | date | None = None
    ) -> date | None:
        """Calendar date of a date code (``ADAT``, ``MDAT`` ...) of treatment ``trno``
        (:func:`reada_date`); ``None`` when missing."""
        return reada_date(self.value(trno, code), sim_start, first_weather)


def _read_i4(text: str) -> int | None:
    """``(I4)`` of a field: blanks ignored, all-blank 0, ``None`` on a read error."""
    s = text.replace(" ", "")
    if not s:
        return 0
    return int(s) if _INT.match(s) else None


def read_filea(path: str | Path) -> FileA:
    """Read a FILEA (``*.MZA``) with the rule of ``READA_Y4K`` (see the module docstring)."""
    p = Path(path)
    lines = read_lines(p)
    at = [i for i, ln in enumerate(lines) if ln.startswith("@")]
    empty = pd.DataFrame(index=pd.Index([], name="TRNO", dtype=int))
    if not at:
        return FileA(p, _title(lines), empty, empty.copy(), 0)
    spans = dssat_header_spans(lines[at[0]][:_RECORD])
    ignored = tuple(n.upper() for n, _, _ in spans[FILEA_MAXCOLN:])
    spans = spans[:FILEA_MAXCOLN]
    names = [n[:_HEADER_WIDTH].upper() for n, _, _ in spans]
    fa = [FILEA_ALIASES.get(n, n) for n in names]
    aliases = {n: a for n, a in zip(names, fa, strict=True) if n != a}
    # 1-based inclusive columns of PARSE_HEADERS
    c1 = [a + 1 for _, a, _ in spans]
    c2 = [b for _, _, b in spans]
    acn = len(spans)
    trno_cols = [j for j in range(acn) if fa[j] == "TRNO"]

    def value_text(line: str, j: int) -> str:
        if j == 0:
            lo, hi = c1[0], (c1[1] if acn > 1 else c2[0])
        elif j == acn - 1:
            lo, hi = c2[j - 1], c2[j] + 1
        else:
            lo, hi = c2[j - 1], c1[j + 1]
        first = lo + 2 if (j > 0 and fa[j - 1] == "TRNO") else lo + 1
        return line[first - 1 : hi - 1][:_VALUE_WIDTH].rstrip()

    rows: dict[int, dict[str, str]] = {}
    for ln in lines[at[0] + 1 :]:
        if not ln.strip() or ln.startswith(("!", "@")):  # IGNORE
            continue
        rec = ln[:_RECORD].ljust(_RECORD)
        tr: int | None = None
        for j in trno_cols:
            field_txt = rec[c1[j] - 1 : c2[j] + 1]
            t = _read_i4(field_txt[_TRNO_SKIP : _TRNO_SKIP + _TRNO_WIDTH])
            if t is not None:
                tr = t
        if tr is None or tr <= 0 or tr in rows:
            continue
        vals: dict[str, str] = {}
        for j in range(acn):
            if fa[j] != "TRNO":
                vals[fa[j]] = value_text(rec, j)  # the last column of a name wins
        rows[tr] = vals
    cols = list(dict.fromkeys(n for n in fa if n != "TRNO"))
    text = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols).sort_index()
    text.index.name = "TRNO"
    text = text.fillna("")
    table = pd.DataFrame({c: [_fread(str(v)) for v in text[c]] for c in cols}, index=text.index, dtype=float)
    return FileA(p, _title(lines), table, text, len(at), aliases, ignored)


@dataclass(frozen=True)
class FileT:
    """A parsed FILET: ``table`` is tidy with columns ``TRNO`` (int), ``DATE`` (``datetime.date``),
    ``YRDOY`` (int ``YYYYDDD``), ``DATECODE`` (the file's ``DATE`` as written), ``variable`` (code)
    and ``value`` (float, finite). ``first_weather`` is the ``FirstWeatherDate`` the dates were
    converted with (``None``: the cross-over rule, also for ``-99`` and ``0``)."""

    path: Path
    title: str
    table: pd.DataFrame
    n_tables: int = 1
    first_weather: int | None = None
    #: rows left out because their ``DATE`` is not a valid date: ``(TRNO, DATE as written, reason)``
    invalid_dates: tuple[tuple[int, int, str], ...] = ()

    @property
    def codes(self) -> tuple[str, ...]:
        """The variable codes with at least one observation, in file order."""
        return tuple(dict.fromkeys(self.table["variable"]))

    def series(self, trno: int, code: str) -> pd.DataFrame:
        """The observations of ``code`` for treatment ``trno`` sorted by date (``DATE``, ``YRDOY``,
        ``DATECODE``, ``value``)."""
        t = self.table
        keep = (t["TRNO"].to_numpy() == trno) & (t["variable"].to_numpy() == code)
        sel = pd.DataFrame(t.loc[keep, ["DATE", "YRDOY", "DATECODE", "value"]])
        order = np.argsort(sel["YRDOY"].to_numpy(), kind="stable")
        return pd.DataFrame(sel.iloc[order]).reset_index(drop=True)


_TIDY_COLUMNS = ["TRNO", "DATE", "YRDOY", "DATECODE", "variable", "value"]


def read_filet(path: str | Path, *, first_weather: int | str | date | None = None) -> FileT:
    """Read a FILET (``*.MZT``): every ``@TRNO DATE ...`` table (see the module docstring);
    ``first_weather`` is DSSAT's ``FirstWeatherDate`` for the ``Y4K_DOY`` dates."""
    p = Path(path)
    lines = read_lines(p)
    rows: list[tuple[int, date, int, int, str, float]] = []
    n_tables = 0
    invalid: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.startswith("@"):
            i += 1
            continue
        toks = header_tokens(ln)
        names = [t[0] for t in toks]
        raw: list[list[str]] = []
        i += 1
        while i < len(lines) and not lines[i].startswith(("@", "*")):
            if _is_data(lines[i]):
                raw.append(split_fixed(toks, lines[i]))
            i += 1
        if "TRNO" not in names or "DATE" not in names or not raw:
            continue
        n_tables += 1
        df = frame_from_rows(names, raw)
        codes = [n for n in names if n not in ("TRNO", "DATE") and _NAME.match(n)]
        for _, r in df.iterrows():
            trno, dv = to_float(r["TRNO"]), to_float(r["DATE"])
            if not (math.isfinite(trno) and math.isfinite(dv)) or dv <= 0:
                continue
            try:
                d = y4k_date(int(dv), first_weather)
            except ValueError as e:  # e.g. day 0 or 366 of a non-leap year: reported, not rolled over
                invalid.append((int(trno), int(dv), str(e)))
                continue
            for c in codes:
                v = to_float(r[c])
                if math.isfinite(v):
                    rows.append((int(trno), d, _yrdoy(d), int(dv), c, v))
    table = pd.DataFrame(rows, columns=_TIDY_COLUMNS)
    for c in ("TRNO", "YRDOY", "DATECODE"):
        table[c] = table[c].astype(int)
    table["value"] = table["value"].astype(float)
    fw = None if first_weather is None else _as_yrdoy(first_weather)
    return FileT(p, _title(lines), table, n_tables, fw if fw is not None and fw > 0 else None, tuple(invalid))


def find_observed_files(experiment: str | Path) -> tuple[Path | None, Path | None]:
    """``(FILEA, FILET)`` next to an experiment file (``UFGA8201.MZX`` -> ``UFGA8201.MZA``,
    ``UFGA8201.MZT``; case-insensitive), ``None`` for a missing one. ``experiment`` may also be a
    FILEA or FILET path."""
    x = Path(experiment)
    ext = x.suffix.upper()
    if len(ext) != len(".MZX"):
        raise ValueError(f"{x}: not a DSSAT experiment file name (expected e.g. .MZX)")
    stem = x.stem.upper()
    found: dict[str, Path] = {}
    if x.parent.is_dir():
        for f in x.parent.iterdir():
            if f.is_file() and f.stem.upper() == stem and len(f.suffix) == len(ext):
                if f.suffix.upper() == ext[:-1] + "A":
                    found["A"] = f
                elif f.suffix.upper() == ext[:-1] + "T":
                    found["T"] = f
    return found.get("A"), found.get("T")


@dataclass(frozen=True)
class ObservedData:
    """The observations of one experiment: its FILEA and FILET (either may be absent)."""

    experiment: str
    filea: FileA | None = None
    filet: FileT | None = None
    #: free-text notes (tables DSSAT would not read, dropped columns)
    notes: tuple[str, ...] = field(default=())

    @property
    def trnos(self) -> tuple[int, ...]:
        """Every treatment with at least one observation."""
        s: set[int] = set()
        if self.filea is not None:
            s |= {int(t) for t in self.filea.table.index}
        if self.filet is not None:
            s |= {int(t) for t in np.unique(self.filet.table["TRNO"].to_numpy())}
        return tuple(sorted(s))

    def counts(self) -> dict[str, int]:
        """Numbers of finite observations: ``{"A:<code>": n, "T:<code>": n}``."""
        out: dict[str, int] = {}
        if self.filea is not None:
            for c in self.filea.codes:
                out[f"A:{c}"] = int(np.count_nonzero(self.filea.table[c].notna().to_numpy()))
        if self.filet is not None:
            vc = Counter(str(v) for v in self.filet.table["variable"].to_numpy())
            for c in self.filet.codes:
                out[f"T:{c}"] = vc[c]
        return out


def read_observed(experiment: str | Path, *, first_weather: int | str | date | None = None) -> ObservedData:
    """FILEA and FILET of an experiment file (``.../UFGA8201.MZX``), see :func:`find_observed_files`;
    ``first_weather`` (``FirstWeatherDate``) is used for the FILET dates."""
    fa, ft = find_observed_files(experiment)
    a = read_filea(fa) if fa is not None else None
    t = read_filet(ft, first_weather=first_weather) if ft is not None else None
    notes: list[str] = []
    if a is not None and a.n_tables > 1:
        notes.append(
            f"{fa}: {a.n_tables} '@' lines; READA_Y4K (READS.for:1176-1323) reads the rows of the later "
            "tables with the first table's columns, as done here"
        )
    if a is not None and a.aliases:
        notes.append(f"{fa}: READA_Y4K header aliases applied: {a.aliases}")
    if a is not None and a.ignored_columns:
        notes.append(f"{fa}: columns beyond READA_Y4K's MAXCOLN = 40 not read: {list(a.ignored_columns)}")
    if a is not None:
        junk = [c for c in a.table.columns if c not in a.codes]
        if junk:
            notes.append(f"{fa}: header names that are not codes (not used): {junk}")
    if t is not None and t.invalid_dates:
        notes.append(f"{ft}: rows with invalid dates left out: {list(t.invalid_dates)}")
    return ObservedData(Path(experiment).stem.upper(), a, t, tuple(notes))
