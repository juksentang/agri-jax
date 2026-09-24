"""DSSAT soil files (``*.SOL``): read and write.

``read_sol`` returns ``{profile_id: SoilProfile}``. Each profile holds the title fields
(source, texture, depth, description), the ``@SITE`` line, the surface line
(``SCOM SALB SLU1 SLDR SLRO SLNF SLPF SMHB SMPX SMKE``) and one layer DataFrame that merges
every layer table of the profile on ``SLB`` (first table: ``SLB SLMH SLLL SDUL SSAT SRGF
SSKS SBDM SLOC SLCL SLSI SLCF SLNI SLHW SLHB SCEC SADC``; second, optional: ``SLPX ...``).
Units follow the DSSAT SOIL.CDE (SLB cm, water contents cm3 cm-3, SSKS cm h-1, SBDM g
cm-3, SLOC/SLCL/SLSI %). ``-99`` becomes NaN in the layer table.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ._fixed import (
    convert,
    fmt_num,
    frame_from_rows,
    header_tokens,
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


def read_sol(path: str | Path) -> dict[str, SoilProfile]:
    """Read every profile of a ``.SOL`` file (see module docstring)."""
    lines = read_lines(path)
    profiles: dict[str, SoilProfile] = {}
    prof: SoilProfile | None = None
    tables: list[pd.DataFrame] = []

    def finish() -> None:
        if prof is None:
            return
        if tables:
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
            toks = header_tokens(ln, joins=_JOINS)
            names = [t[0] for t in toks]
            rows: list[list[str]] = []
            j = i + 1
            while j < n:
                d = lines[j]
                if d.startswith(("@", "*")) or not d.strip():
                    break
                if not d.lstrip().startswith("!"):
                    rows.append(split_fixed(toks, d))
                j += 1
            if names and names[0] == "SITE" and rows:
                prof.site = _parse_site(lines[i + 1])
            elif names and names[0] == "SCOM" and rows:
                prof.surface = _record(toks, lines[i + 1])
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


def write_sol(
    profiles: Mapping[str, SoilProfile] | list[SoilProfile],
    path: str | Path,
    *,
    title: str = "*SOILS: DSSAT Soil Input File",
) -> Path:
    """Write profiles to a ``.SOL`` file in the standard DSSAT layout (6-column fields)."""
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
        out.append("@" + "".join(f"{c:>6}" for c in scols)[1:])
        out.append("".join(fmt_num(p.surface.get(c), 6) for c in scols))
        groups = p.layer_groups or [list(p.layers.columns)]
        for g in groups:
            out.append("@" + "".join(f"{c:>6}" for c in g)[1:])
            for _, row in p.layers[g].iterrows():
                cells = []
                for c in g:
                    v: Any = row[c]
                    if isinstance(v, str):
                        cells.append(fmt_num(v, 6))
                    elif c == "SLB" and float(v).is_integer():
                        cells.append(fmt_num(int(v), 6))
                    else:
                        cells.append(fmt_num(float(v), 6))
                out.append("".join(cells))
        out.append("")
    pth = Path(path)
    pth.write_text("\n".join(out) + "\n", encoding="latin-1", errors="replace")
    return pth
