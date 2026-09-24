"""CERES-Maize genotype files: cultivar (``.CUL``), ecotype (``.ECO``), species (``.SPE``).

Both the DSSAT 4.0 / RZWQM-DSSAT layout (``MZCER040``) and the DSSAT 4.8 layout
(``MZCER048``) are handled.

Cultivar/ecotype rows: the id (``VAR#`` / ``ECO#``, 6 characters) and the name
(``VRNAME....`` / ``ECONAME...``, 16 characters) are fixed-width text under their ``@``
header tokens (a name that overruns its 16 columns keeps the overrunning word); every
remaining field is one blank-free token (``EXPNO`` ``.``, ``ECO#``, numbers), read in order
against the remaining header names -- the way the Fortran ``READ (A6,1X,A16,1X,A6,...)``
effectively works and robust to the small misalignments of hand-edited files (``P20``).

Writers (:func:`write_cul`, :func:`write_eco`, :func:`write_spe`) are layout preserving: the
readers keep the source lines and the column span of every value (``df.attrs["layout"]``,
:attr:`SpeciesFile.layout`), and a writer puts each value back into its own span -- the original
text when the value is unchanged (so an unmodified table is written back byte for byte), else
the shortest exact text with a decimal point that fits the span. DSSAT reads these files with
rigid Fortran formats (``(A6,1X,A16,7X,A6,6F6.0)`` for MZCER048.CUL, ``(A6,1X,A16,1X,9(1X,F5.1))``
for the ecotype, ``(7X,4(1X,F5.2))`` for ``PRFTC``...), where an ``F5.1`` field without a
decimal point is scaled (``12`` reads ``1.2``) and a value crossing its field is misread, so a
value that does not fit its span raises ``ValueError`` instead of being rounded or shifted.

``MZCER040.CUL`` written by RZWQM has no ``@`` line; it uses the DSSAT 4.0 format
``(A6,1X,A16,1X,A6,F6...)`` with ``VAR# VRNAME ECO# P1 P2 P5 G2 G3 PHINT`` and the two
RZWQM additions ``HTMAX`` (cm) and ``BIOHALF`` (``IPVARC.for``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ._fixed import convert, header_tokens, read_lines, short_float

__all__ = [
    "CUL040_COLUMNS",
    "SpeciesFile",
    "read_cul",
    "read_eco",
    "read_spe",
    "write_cul",
    "write_eco",
    "write_spe",
]

#: DSSAT 4.0 / RZWQM-DSSAT maize cultivar layout (no ``@`` header in the file)
CUL040_COLUMNS = ("VAR#", "VRNAME", "ECO#", "P1", "P2", "P5", "G2", "G3", "PHINT", "HTMAX", "BIOHALF")


def _num(tok: str) -> Any:
    v = convert(tok)
    return float(v) if isinstance(v, int) else v


def _raw_lines(path: str | Path) -> tuple[list[str], str]:
    """Lines exactly as in the file (tabs kept) and the line terminator used."""
    text = Path(path).read_bytes().decode("latin-1")
    nl = "\r\n" if "\r\n" in text else "\n"
    return text.split(nl), nl


_WORD = re.compile(r"\S+")


def _table(path: str | Path, default_header: str | None) -> pd.DataFrame:
    lines = read_lines(path)
    header = next((ln for ln in lines if ln.startswith("@")), None)
    if header is None:
        if default_header is None:
            raise ValueError(f"{path}: no '@' header line")
        header = default_header
    toks = header_tokens(header)
    names = [t[0] for t in toks]
    k_name = next((k for k, t in enumerate(names) if "NAME" in t), None)
    id_name = names[0]
    if k_name is not None:
        name_start, name_end = toks[k_name][1], toks[k_name][2]
        rest_names = names[k_name + 1 :]
    else:  # ecotype tables without a name column (BACER, WHCER ...): whitespace tokens
        name_start = name_end = 0
        rest_names = names[1:]

    records: list[dict[str, Any]] = []
    spans: list[tuple[int, dict[str, tuple[int, int]]]] = []
    seen_header = header is not None and header in lines
    after = lines.index(header) + 1 if seen_header else 0
    for li, ln in enumerate(lines[after:], start=after):
        body = ln.split("!", 1)[0].rstrip()
        if not body.strip() or ln.startswith(("*", "@", "$")) or ln.lstrip().startswith("!"):
            continue
        sp: dict[str, tuple[int, int]] = {}
        if k_name is not None:
            end = name_end
            while end < len(body) and body[end] != " ":  # name overrunning its columns
                end += 1
            id_txt = body[:name_start].strip() or body[:6].strip()
            rec: dict[str, Any] = {id_name: id_txt}
            rec[names[k_name]] = body[name_start:end].strip()
            sp[id_name] = (0, name_start)
            sp[names[k_name]] = (name_start, end)
            words = list(_WORD.finditer(body, end))
        else:
            words = list(_WORD.finditer(body))
            if not words:
                continue
            rec = {id_name: words[0].group(0)}
            sp[id_name] = (words[0].start(), words[0].end())
            words = words[1:]
        for j, n in enumerate(rest_names):
            if j < len(words):
                rec[n] = _num(words[j].group(0))
                sp[n] = (words[j].start(), words[j].end())
            else:
                rec[n] = math.nan
        records.append(rec)
        spans.append((li, sp))
    cols = [id_name, *([names[k_name]] if k_name is not None else []), *rest_names]
    df = pd.DataFrame.from_records(records, columns=cols)
    for c in rest_names:
        if all(isinstance(v, float) for v in df[c]):
            df[c] = df[c].astype(float)
            df.loc[(df[c] <= -99.0) & (df[c] > -99.5), c] = math.nan
    df = df.set_index(id_name)
    raw, nl = _raw_lines(path)
    df.attrs["layout"] = {
        "lines": raw,
        "newline": nl,
        "rows": spans,
        "header": header if seen_header else None,
    }
    return df


# --------------------------------------------------------------------------- writers


def _same(raw: str, v: Any) -> bool:
    """True when the source text ``raw`` already encodes value ``v``."""
    if isinstance(v, str):
        return raw.strip() == v.strip()
    try:
        x = float(v)
    except (TypeError, ValueError):
        return False
    try:
        r = float(raw)
    except ValueError:
        return False
    if math.isnan(x):
        return -99.5 < r <= -99.0
    return r == x or abs(r - x) <= 1e-12 * max(1.0, abs(x))


def _fit_number(x: float, width: int) -> str:
    """Shortest text of ``x`` with a decimal point (Fortran ``F`` input scales a value without
    one) in at most ``width`` characters that reproduces ``x``; ``ValueError`` when none fits."""
    if math.isnan(x):
        x = -99.0
    for d in range(0, 8):
        t = f"{x:.{d}f}" if d else f"{x:.0f}."
        if len(t) > width and t.lstrip("-").startswith("0."):
            t = t.replace("0.", ".", 1)
        if len(t) <= width and abs(float(t) - x) <= 1e-9 * max(1.0, abs(x)):
            return t
    raise ValueError(f"{x!r} does not fit a {width}-column field exactly (best: {short_float(x, width)!r})")


def _put(line: str, span: tuple[int, int], value: Any, *, text: bool) -> str:
    """``line`` with ``value`` written into ``span``: kept as is when the text there already
    encodes the value; text left-aligned; numbers right-aligned to the span end, allowed to grow
    left into the blanks before the span but one (never touching the previous value)."""
    a, b = span
    line = line.ljust(b)
    if _same(line[a:b], value):
        return line
    if text:
        t = str(value)
        if len(t) > b - a:
            raise ValueError(f"text {t!r} longer than its {b - a}-column field")
        return line[:a] + t.ljust(b - a) + line[b:]
    lo = a
    while lo >= 2 and line[lo - 1] == " " and line[lo - 2] == " ":
        lo -= 1
    t = _fit_number(float(value), b - lo)
    start = b - len(t)
    head = min(a, start)
    return line[:head] + " " * (start - head) + t + line[b:]


def _new_row(template: dict[str, tuple[int, int]], rec: dict[str, Any], text_cols: set[str]) -> str:
    width = max(b for _, b in template.values())
    line = " " * width
    for c, sp in template.items():
        v = rec.get(c, math.nan)
        line = _put(line[: sp[0]] + " " * (sp[1] - sp[0]) + line[sp[1] :], sp, v, text=c in text_cols)
    return line.rstrip()


def _write_table(df: pd.DataFrame, path: str | Path) -> Path:
    lay = df.attrs.get("layout")
    if not lay:
        raise ValueError("write_cul/write_eco need the layout of a table read by read_cul/read_eco")
    id_name = str(df.index.name)
    flat = df.reset_index()
    text_cols = {c for c in flat.columns if not pd.api.types.is_numeric_dtype(flat[c])}
    lines: list[str] = list(lay["lines"])
    by_id: dict[str, list[int]] = {}
    for k, ident in enumerate(flat[id_name]):
        by_id.setdefault(str(ident), []).append(k)
    used: set[int] = set()
    drop: set[int] = set()
    last = max((li for li, _ in lay["rows"]), default=len(lines) - 1)
    for li, sp in lay["rows"]:
        ident = lay["lines"][li][sp[id_name][0] : sp[id_name][1]].strip()
        ks = [k for k in by_id.get(ident, []) if k not in used]
        if not ks:
            drop.add(li)
            continue
        k = ks[0]
        used.add(k)
        rec = flat.iloc[k].to_dict()
        ln = lines[li]
        for c, span in sp.items():
            ln = _put(ln, span, rec[c], text=c in text_cols)
        lines[li] = ln
    template = lay["rows"][0][1] if lay["rows"] else None
    extra = [k for k in range(len(flat)) if k not in used]
    if extra and template is None:
        raise ValueError("cannot add rows to a table without a template row")
    new = [_new_row(template, flat.iloc[k].to_dict(), text_cols) for k in extra] if template else []
    out = [ln for i, ln in enumerate(lines[: last + 1]) if i not in drop] + new + lines[last + 1 :]
    p = Path(path)
    p.write_bytes(lay["newline"].join(out).encode("latin-1", errors="replace"))
    return p


def write_cul(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a cultivar table read by :func:`read_cul` (layout preserving, see module doc).

    Rows are matched by ``VAR#`` (in order for repeated ids); rows no longer in ``df`` are
    dropped and new rows are appended after the last cultivar in the layout of the first one.
    """
    return _write_table(df, path)


def write_eco(df: pd.DataFrame, path: str | Path) -> Path:
    """Write an ecotype table read by :func:`read_eco` (layout preserving, as :func:`write_cul`)."""
    return _write_table(df, path)


def read_cul(path: str | Path) -> pd.DataFrame:
    """Cultivar table indexed by ``VAR#`` (columns ``VRNAME``, [``EXPNO``], ``ECO#``,
    ``P1 P2 P5 G2 G3 PHINT`` [``HTMAX BIOHALF``]); -99 -> NaN."""
    default = "@VAR#  VRNAME.......... ECO#   " + " ".join(CUL040_COLUMNS[3:])
    df = _table(path, default)
    if not any(ln.startswith("@") for ln in read_lines(path)):
        # 040 rows without HTMAX/BIOHALF: drop the all-NaN extra columns
        df = df.drop(columns=[c for c in ("HTMAX", "BIOHALF") if bool(df[c].isna().to_numpy().all())])
    return df


def read_eco(path: str | Path) -> pd.DataFrame:
    """Ecotype table indexed by ``ECO#`` (``ECONAME``, ``TBASE TOPT ROPT P20 DJTI GDDE
    DSGFT RUE KCAN`` [+ ``TSEN CDAY`` in 048]); -99 -> NaN."""
    return _table(path, None)


@dataclass
class SpeciesFile:
    """Parsed ``.SPE`` file.

    ``params`` maps each named parameter (``PRFTC``, ``CO2X``, ``PARSR``, ``SRATPHOTO``...)
    to a float or a tuple of floats; ``sections`` maps section title -> parameter names in
    order; ``rows`` keeps unnamed numeric rows per section as ``(values, label)``.
    """

    title: str
    params: dict[str, float | tuple[float, ...]] = field(default_factory=dict)
    sections: dict[str, list[str]] = field(default_factory=dict)
    rows: dict[str, list[tuple[tuple[float, ...], str]]] = field(default_factory=dict)
    #: source lines and value spans used by :func:`write_spe` (``{"lines", "newline",
    #: "params": {name: [(line, start, end), ...]}, "rows": {section: [[(line, start, end)...]]}}``)
    layout: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __getitem__(self, key: str) -> float | tuple[float, ...]:
        return self.params[key]


_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _is_param_name(tok: str) -> bool:
    return bool(_IDENT.match(tok)) and sum(ch.isupper() for ch in tok) >= 2


def _float(tok: str) -> float | None:
    try:
        return float(tok)
    except ValueError:
        return None


def read_spe(path: str | Path) -> SpeciesFile:
    """Read a species file (see :class:`SpeciesFile`).

    Lines ``NAME v1 [v2 ...] [! comment]`` give named parameters. Lines starting with
    numbers (the phosphorus block of 048) are named when the trailing label lists as many
    identifiers as values (``0.80 1.00  SRATPHOTO, SRATPART``); otherwise they are kept in
    ``rows``. A name defined twice keeps its last value (and :func:`write_spe` writes that one).
    """
    lines = read_lines(path)
    raw, nl = _raw_lines(path)
    title = next((ln.strip() for ln in lines if ln.startswith("*")), "")
    spe = SpeciesFile(title)
    pspans: dict[str, list[tuple[int, int, int]]] = {}
    rspans: dict[str, list[list[tuple[int, int, int]]]] = {}
    section = ""
    for li, ln in enumerate(lines):
        if ln.startswith("*"):
            if ln.strip() == title and not spe.sections:
                continue
            section = ln[1:].strip()
            spe.sections.setdefault(section, [])
            spe.rows.setdefault(section, [])
            continue
        body = ln.split("!", 1)[0]
        if not body.strip() or ln.lstrip().startswith(("!", "@")):
            continue
        words = list(_WORD.finditer(body))
        toks = [w.group(0) for w in words]
        if _float(toks[0]) is None:
            name = toks[0]
            vals: list[float] = []
            sp: list[tuple[int, int, int]] = []
            for w in words[1:]:
                v = _float(w.group(0))
                if v is not None:
                    vals.append(v)
                    sp.append((li, w.start(), w.end()))
            spe.params[name] = vals[0] if len(vals) == 1 else tuple(vals)
            pspans[name] = sp
            spe.sections.setdefault(section, []).append(name)
            continue
        vals = []
        k = 0
        while k < len(toks) and (v := _float(toks[k])) is not None:
            vals.append(v)
            k += 1
        sp = [(li, w.start(), w.end()) for w in words[:k]]
        label = " ".join(toks[k:])
        cand = [t for t in re.split(r"[,\s]+", label) if t][: len(vals)]
        if len(cand) == len(vals) and all(_is_param_name(t) for t in cand):
            for n, v, s1 in zip(cand, vals, sp, strict=True):
                spe.params[n] = v
                pspans[n] = [s1]
                spe.sections.setdefault(section, []).append(n)
        else:
            spe.rows.setdefault(section, []).append((tuple(vals), label))
            rspans.setdefault(section, []).append(sp)
    spe.layout = {"lines": raw, "newline": nl, "params": pspans, "rows": rspans}
    return spe


def write_spe(spe: SpeciesFile, path: str | Path) -> Path:
    """Write a species file read by :func:`read_spe`, layout preserving (see module doc).

    Every value of :attr:`SpeciesFile.params` and :attr:`SpeciesFile.rows` goes back into its
    own span of the source line; the number of values of a parameter cannot change and new
    parameters cannot be added (DSSAT reads each line with a fixed format).
    """
    lay = spe.layout
    if not lay:
        raise ValueError("write_spe needs a SpeciesFile read by read_spe (its layout)")
    lines: list[str] = list(lay["lines"])

    def put(spans: list[tuple[int, int, int]], values: tuple[float, ...], what: str) -> None:
        if len(spans) != len(values):
            raise ValueError(f"{what}: {len(values)} values for {len(spans)} fields in the file")
        for (li, a, b), v in zip(spans, values, strict=True):
            lines[li] = _put(lines[li], (a, b), float(v), text=False)

    unknown = set(spe.params) - set(lay["params"])
    if unknown:
        raise ValueError(f"parameters not in the source file: {sorted(unknown)}")
    for name, v in spe.params.items():
        vals = v if isinstance(v, tuple) else (v,)
        put(lay["params"][name], tuple(float(x) for x in vals), name)
    for sec, rows in spe.rows.items():
        spans = lay["rows"].get(sec, [])
        if len(spans) != len(rows):
            raise ValueError(f"section {sec!r}: {len(rows)} rows for {len(spans)} in the file")
        for (vals, _label), sp in zip(rows, spans, strict=True):
            put(sp, vals, f"{sec} row")
    p = Path(path)
    p.write_bytes(lay["newline"].join(lines).encode("latin-1", errors="replace"))
    return p
