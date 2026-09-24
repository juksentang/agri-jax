"""DSSAT soil files (``*.SOL``): read and write.

``read_sol`` returns ``{profile_id: SoilProfile}``. Each profile holds the title fields
(source, texture, depth, description), the ``@SITE`` line, the surface line
(``SCOM SALB SLU1 SLDR SLRO SLNF SLPF SMHB SMPX SMKE``) and one layer DataFrame that merges
every layer table of the profile on ``SLB`` (first table: ``SLB SLMH SLLL SDUL SSAT SRGF
SSKS SBDM SLOC SLCL SLSI SLCF SLNI SLHW SLHB SCEC SADC``; second, optional: ``SLPX ...``).
Units follow the DSSAT SOIL.CDE (SLB cm, water contents cm3 cm-3, SSKS cm h-1, SBDM g
cm-3, SLOC/SLCL/SLSI %). ``-99`` becomes NaN in the layer table.

How DSSAT reads the file (``IPSOIL_Inp``): the title and ``@SITE`` lines have fixed formats; the
surface and layer tables are read by ``PARSE_HEADERS`` spans with a list-directed ``READ`` per
field (the column right after each header token belongs to no field); blank and ``!`` lines are
skipped (a table ends only at the next ``@`` or ``*`` line); the rows of a second layer table go
to layers 1, 2, ... by position, whatever their ``SLB``. By default :func:`read_sol` splits
fields by the header positions (:mod:`._fixed`), which also reads a value that touches its left
neighbour (``1.20 1.552`` under ``SBDM  SLOC``, where DSSAT reads ``SLOC`` = ``.552``), and
joins the layer tables on ``SLB``. ``read_sol(..., dssat_spans=True)`` returns exactly what
``dscsm048`` reads instead (spans, list-directed items, tables paired by row). Either way a later
layer table whose ``SLB`` column is not the first one's, row by row (it may stop early),
raises a ``UserWarning``, since the two rules then give different layers.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ._fixed import (
    convert,
    dssat_header_spans,
    fmt_num,
    frame_from_rows,
    header_tokens,
    list_directed_float,
    read_lines,
    short_float,
    split_fixed,
)

__all__ = ["SoilProfile", "read_sol", "write_sol"]

_JOINS = ("SCS FAMILY", "SCS Family")


@dataclass
class SoilProfile:
    """One ``*ID_SOIL`` block of a ``.SOL`` file."""

    id: str
    source: str = ""
    texture: str = ""
    depth: float = math.nan
    description: str = ""
    site: dict[str, Any] = field(default_factory=dict)
    surface: dict[str, Any] = field(default_factory=dict)
    layers: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: column groups of the layer tables in file order (each starts with ``SLB``)
    layer_groups: list[list[str]] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


def _parse_title(line: str) -> tuple[str, str, str, float, str]:
    """``*ID_SOIL  SOURCE      TXT    DEPTH DESCRIPTION`` (1X,A10,2X,A11,1X,A5,1X,F5.0,1X,A)."""
    pid = line[1:11].strip()
    source = line[13:24].strip()
    texture = line[25:30].strip()
    try:
        depth = float(line[31:36]) if line[31:36].strip() else math.nan
        desc = line[37:].strip()
    except ValueError:  # non-standard spacing: fall back to tokens
        rest = line[11:].split()
        source = rest[0] if rest else ""
        texture = rest[1] if len(rest) > 1 else ""
        depth = float(rest[2]) if len(rest) > 2 and rest[2].lstrip("-").isdigit() else math.nan
        desc = " ".join(rest[3:])
    return pid, source, texture, depth, desc


def _parse_site(line: str) -> dict[str, Any]:
    """``@SITE`` data line in DSSAT's fixed format ``(2(1X,A11),2(1X,F8.3),1X,A50)``."""
    padded = line.ljust(43)

    def num(s: str) -> float:
        try:
            v = float(s)
        except ValueError:
            return math.nan
        return math.nan if -99.5 < v <= -99.0 else v

    return {
        "SITE": padded[1:12].strip(),
        "COUNTRY": padded[13:24].strip(),
        "LAT": num(padded[25:33]),
        "LONG": num(padded[34:42]),
        "SCS FAMILY": padded[43:].strip(),
    }


def _record(toks: list[tuple[str, int, int]], line: str) -> dict[str, Any]:
    """One-line record; like DSSAT's list-directed read, only the first word of each field."""
    vals = split_fixed(toks, line)
    return {t[0]: convert(v.split()[0] if v.split() else "") for t, v in zip(toks, vals, strict=True)}


#: surface columns DSSAT reads as reals (``SCOM SMHB SMPX SMKE SGRP`` are text)
_REAL_SURFACE = frozenset({"SALB", "SLU1", "SLDR", "SLRO", "SLNF", "SLPF"})


def _dssat_fields(spans: list[tuple[str, int, int]], line: str, *, layer: bool) -> list[str]:
    """Raw first list-directed item of each ``PARSE_HEADERS`` span (``""`` when empty). A real
    column (every layer column but ``SLMH``; ``SALB``...``SLPF`` of the surface line) whose item
    is not a number gives ``""``: the read fails and DSSAT keeps ``-99``."""
    out: list[str] = []
    for name, a, b in spans:
        items = line[a:b].replace(",", " ").split()
        item = items[0] if items else ""
        real = name != "SLMH" if layer else name in _REAL_SURFACE
        if real and item and math.isnan(list_directed_float(item)):
            item = ""
        out.append(item)
    return out


def _pair_by_row(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Layer tables combined the way DSSAT does: row *i* of every table is layer *i*; a later
    table's ``SLB`` overwrites the depth (``ZLYR(L)`` is read again from each table)."""
    merged = tables[0].copy()
    for t in tables[1:]:
        t = t.reset_index(drop=True)
        if len(t) > len(merged):
            merged = merged.reindex(range(len(t)))
        for c in t.columns:
            if c == "SLB":
                merged.loc[: len(t) - 1, "SLB"] = t["SLB"].to_numpy()
            else:
                col = pd.Series(
                    math.nan if t[c].dtype.kind == "f" else "", index=merged.index, dtype=t[c].dtype
                )
                col.iloc[: len(t)] = t[c].to_numpy()
                merged[c] = col
    return merged


def read_sol(path: str | Path, *, dssat_spans: bool = False) -> dict[str, SoilProfile]:
    """Read every profile of a ``.SOL`` file (see module docstring).

    ``dssat_spans=True`` reads the surface and layer tables exactly as DSSAT 4.8 does
    (``PARSE_HEADERS`` spans, first list-directed item, layer tables paired by row), so the
    values are the ones ``dscsm048`` simulates with.
    """
    lines = read_lines(path)
    profiles: dict[str, SoilProfile] = {}
    prof: SoilProfile | None = None
    tables: list[pd.DataFrame] = []

    def finish() -> None:
        if prof is None:
            return
        if tables:
            first = tables[0]["SLB"].to_numpy(dtype=float)
            for t in tables[1:]:
                slb = t["SLB"].to_numpy(dtype=float)
                if len(slb) > len(first) or not (slb == first[: len(slb)]).all():
                    warnings.warn(
                        f"{path}: profile {prof.id}: a layer table lists SLB {slb.tolist()} but the "
                        f"first one {first.tolist()}; DSSAT pairs the tables by row, "
                        + ("as read here" if dssat_spans else "read_sol by SLB (dssat_spans=True: by row)"),
                        UserWarning,
                        stacklevel=3,
                    )
            if dssat_spans:
                merged = _pair_by_row(tables)
            else:
                merged = tables[0]
                for t in tables[1:]:
                    merged = merged.merge(t, on="SLB", how="left")
            prof.layers = merged.reset_index(drop=True)
        profiles[prof.id] = prof

    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("*"):
            if ln.upper().startswith("*SOILS"):
                i += 1
                continue
            finish()
            pid, source, texture, depth, desc = _parse_title(ln)
            prof = SoilProfile(pid, source, texture, depth, desc)
            tables = []
            i += 1
            continue
        if prof is not None and ln.startswith("@"):
            # DSSAT's PARSE_HEADERS ends the header at a '!' (a note, as in ``SADC   !  SSKS``)
            # and its last field there, so the data after that column are not read either
            bang = ln.find("!", 1)
            cut = bang if bang > 0 else None
            toks = header_tokens(ln[:cut], joins=_JOINS)
            names = [t[0] for t in toks]
            spans = dssat_header_spans(ln)
            use_spans = dssat_spans and names[:1] != ["SITE"]
            if use_spans:
                names = [sp[0] for sp in spans]
            rows: list[list[str]] = []
            data_lines: list[str] = []
            j = i + 1
            while j < n:  # like DSSAT's IGNORE3: blank and comment lines do not end a table
                d = lines[j]
                if d.startswith(("@", "*")):
                    break
                if d.strip() and not d.lstrip().startswith("!"):
                    rows.append(
                        _dssat_fields(spans, d, layer=names[:1] == ["SLB"])
                        if use_spans
                        else split_fixed(toks, d[:cut])
                    )
                    data_lines.append(d)
                j += 1
            if names and names[0] == "SITE" and rows:
                prof.site = _parse_site(data_lines[0])
            elif names and names[0] == "SCOM" and rows:
                if use_spans:
                    prof.surface = {k: convert(v) for k, v in zip(names, rows[0], strict=True)}
                else:
                    prof.surface = _record(toks, data_lines[0])
            elif names and names[0] == "SLB" and rows:
                df = frame_from_rows(names, rows, missing_to_nan=True)
                for c in df.columns:
                    if df[c].dtype.kind in "iu":
                        df[c] = df[c].astype(float)
                if tables:
                    df = pd.DataFrame(df[[c for c in df.columns if c == "SLB" or c not in tables[0].columns]])
                tables.append(df)
                prof.layer_groups.append(list(df.columns))
            i = j
            continue
        i += 1
    finish()
    return profiles


def _fmt_site(site: Mapping[str, Any]) -> str:
    def f(k: str) -> str:
        v = site.get(k)
        x = math.nan if v is None or v == "" else float(v)
        if math.isnan(x):
            return f"{'-99':>9}"
        s3 = f"{x:.3f}"
        return f"{s3 if float(s3) == x else short_float(x, 8):>9}"

    return (
        f" {site.get('SITE', '-99')!s:<11} {site.get('COUNTRY', '-99')!s:<11}"
        f"{f('LAT')}{f('LONG')} {site.get('SCS FAMILY', '-99')!s}"
    )


def _layer_value(col: str, v: Any) -> Any:
    if isinstance(v, str):
        return v
    if col == "SLB" and float(v).is_integer():
        return int(v)
    return float(v)


def _cell_text(v: Any) -> str:
    """Text of one value: the six-column form of :func:`fmt_num` when it is exact, else the
    shortest exact form (a wider column)."""
    t = fmt_num(v, 6).strip()
    if isinstance(v, float) and not math.isnan(v) and t != "-99":
        if abs(float(t) - v) > 1e-9 * max(1.0, abs(v)):
            t = fmt_num(v, 16).strip()
    return t


def _table_lines(cols: list[str], rows: list[list[Any]]) -> list[str]:
    """``@`` header and data lines of one table. Each column is six characters wide, wider
    when its name or a value needs it, so that every value keeps a blank before it: DSSAT's
    ``PARSE_HEADERS`` spans leave out the column right after each header token, so a value
    filling its whole field (``101.32`` in six columns) would lose its first digit."""
    texts = [[_cell_text(v) for v in r] for r in rows]
    widths = [max(6, len(c) + 1, 1 + max((len(t[k]) for t in texts), default=0)) for k, c in enumerate(cols)]
    head = "@" + "".join(f"{c:>{w}}" for c, w in zip(cols, widths, strict=True))[1:]
    body = ["".join(f"{t:>{w}}" for t, w in zip(tr, widths, strict=True)) for tr in texts]
    return [head, *body]


def write_sol(
    profiles: Mapping[str, SoilProfile] | list[SoilProfile],
    path: str | Path,
    *,
    title: str = "*SOILS: DSSAT Soil Input File",
) -> Path:
    """Write profiles to a ``.SOL`` file in the standard DSSAT layout (6-column fields).

    A column is widened when a value needs all six characters, so that DSSAT (which does not
    read the character right after a header token) reads every value as written: reading the
    file back with ``read_sol(..., dssat_spans=True)`` gives the values written.
    """
    items = list(profiles.values()) if isinstance(profiles, Mapping) else list(profiles)
    out = [title, ""]
    for p in items:
        depth = "-99" if math.isnan(p.depth) else f"{p.depth:.0f}"
        out.append(f"*{p.id:<10}  {p.source:<11} {p.texture:<5} {depth:>5} {p.description}".rstrip())
        out.append("@SITE        COUNTRY          LAT     LONG SCS FAMILY")
        out.append(_fmt_site(p.site))
        scols = list(p.surface) or [
            "SCOM", "SALB", "SLU1", "SLDR", "SLRO", "SLNF", "SLPF", "SMHB", "SMPX", "SMKE"
        ]  # fmt: skip
        out.extend(_table_lines(scols, [[p.surface.get(c) for c in scols]]))
        groups = p.layer_groups or [list(p.layers.columns)]
        for g in groups:
            rows = [[_layer_value(c, row[c]) for c in g] for _, row in p.layers[g].iterrows()]
            out.extend(_table_lines(g, rows))
        out.append("")
    pth = Path(path)
    pth.write_text("\n".join(out) + "\n", encoding="latin-1", errors="replace")
    return pth
