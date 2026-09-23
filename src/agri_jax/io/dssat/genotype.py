"""CERES-Maize genotype files: cultivar (``.CUL``), ecotype (``.ECO``), species (``.SPE``).

Both the DSSAT 4.0 / RZWQM-DSSAT layout (``MZCER040``) and the DSSAT 4.8 layout
(``MZCER048``) are handled.

Cultivar/ecotype rows: the id (``VAR#`` / ``ECO#``, 6 characters) and the name
(``VRNAME....`` / ``ECONAME...``, 16 characters) are fixed-width text under their ``@``
header tokens (a name that overruns its 16 columns keeps the overrunning word); every
remaining field is one blank-free token (``EXPNO`` ``.``, ``ECO#``, numbers), read in order
against the remaining header names -- the way the Fortran ``READ (A6,1X,A16,1X,A6,...)``
effectively works and robust to the small misalignments of hand-edited files (``P20``).

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

from ._fixed import convert, header_tokens, read_lines

__all__ = [
    "CUL040_COLUMNS",
    "SpeciesFile",
    "read_cul",
    "read_eco",
    "read_spe",
]

#: DSSAT 4.0 / RZWQM-DSSAT maize cultivar layout (no ``@`` header in the file)
CUL040_COLUMNS = ("VAR#", "VRNAME", "ECO#", "P1", "P2", "P5", "G2", "G3", "PHINT", "HTMAX", "BIOHALF")


def _num(tok: str) -> Any:
    v = convert(tok)
    return float(v) if isinstance(v, int) else v


def _table(path: str | Path, default_header: str | None) -> pd.DataFrame:
    lines = read_lines(path)
    header = next((ln for ln in lines if ln.startswith("@")), None)
    if header is None:
        if default_header is None:
            raise ValueError(f"{path}: no '@' header line")
        header = default_header
    toks = header_tokens(header)
    names = [t[0] for t in toks]
    try:
        k_name = next(k for k, t in enumerate(names) if "NAME" in t)
    except StopIteration as e:
        raise ValueError(f"{path}: header has no *NAME column: {header!r}") from e
    id_name = names[0]
    name_start, name_end = toks[k_name][1], toks[k_name][2]
    rest_names = names[k_name + 1 :]

    records: list[dict[str, Any]] = []
    seen_header = header is not None and header in lines
    after = lines.index(header) + 1 if seen_header else 0
    for ln in lines[after:]:
        body = ln.split("!", 1)[0].rstrip()
        if not body.strip() or ln.startswith(("*", "@", "$")) or ln.lstrip().startswith("!"):
            continue
        end = name_end
        while end < len(body) and body[end] != " ":  # name overrunning its columns
            end += 1
        rec: dict[str, Any] = {id_name: body[:name_start].strip() or body[:6].strip()}
        rec[names[k_name]] = body[name_start:end].strip()
        vals = body[end:].split()
        for j, n in enumerate(rest_names):
            rec[n] = _num(vals[j]) if j < len(vals) else math.nan
        records.append(rec)
    df = pd.DataFrame.from_records(records, columns=[id_name, names[k_name], *rest_names])
    for c in rest_names:
        if all(isinstance(v, float) for v in df[c]):
            df[c] = df[c].astype(float)
            df.loc[(df[c] <= -99.0) & (df[c] > -99.5), c] = math.nan
    return df.set_index(id_name)


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
    ``rows``.
    """
    lines = read_lines(path)
    title = next((ln.strip() for ln in lines if ln.startswith("*")), "")
    spe = SpeciesFile(title)
    section = ""
    for ln in lines:
        if ln.startswith("*"):
            if ln.strip() == title and not spe.sections:
                continue
            section = ln[1:].strip()
            spe.sections.setdefault(section, [])
            spe.rows.setdefault(section, [])
            continue
        body = ln.split("!", 1)[0].strip()
        if not body or ln.lstrip().startswith(("!", "@")):
            continue
        toks = body.split()
        if _float(toks[0]) is None:
            name = toks[0]
            vals = [v for v in (_float(t) for t in toks[1:]) if v is not None]
            spe.params[name] = vals[0] if len(vals) == 1 else tuple(vals)
            spe.sections.setdefault(section, []).append(name)
            continue
        vals: list[float] = []
        k = 0
        while k < len(toks) and (v := _float(toks[k])) is not None:
            vals.append(v)
            k += 1
        label = " ".join(toks[k:])
        cand = [t for t in re.split(r"[,\s]+", label) if t][: len(vals)]
        if len(cand) == len(vals) and all(_is_param_name(t) for t in cand):
            for n, v in zip(cand, vals, strict=True):
                spe.params[n] = v
                spe.sections.setdefault(section, []).append(n)
        else:
            spe.rows.setdefault(section, []).append((tuple(vals), label))
    return spe
