"""DSSAT experiment files (FileX: ``*.MZX``, ``*.SBX``, ...): section reader.

``read_filex`` returns ``{section: content}`` for every ``*SECTION`` of the file, with the
section name as written after the ``*`` (e.g. ``"TREATMENTS"``, ``"CULTIVARS"``,
``"FIELDS"``, ``"PLANTING DETAILS"``, ``"SIMULATION CONTROLS"``). Rows are parsed by the
positions of their ``@`` header line (``TNAME`` ``RAINFED LOW NITROGEN``, ``ID_SOIL``
``IBMZ910014`` and ``FLNAME`` ``Field section`` stay whole). Values are typed (int /
float / str; ``-99`` is kept as written, codes with leading zeros such as ``00000`` stay
str).

Content by section:

* ``TREATMENTS`` -- ``list[dict]``, one dict per treatment row (``N R O C TNAME CU FL SA
  IC MP MI MF MR MC MT ME MH SM``).
* ``SIMULATION CONTROLS`` -- ``{level: {group: dict}}`` where group is the second header
  token (``GENERAL``, ``OPTIONS``, ``METHODS``, ``MANAGEMENT``, ``OUTPUTS``, ``PLANTING``,
  ``IRRIGATION``, ``NITROGEN``, ``RESIDUES``, ``HARVEST``); the group code column (``GE``,
  ``OP``...) is dropped.
* every other section (``CULTIVARS``, ``FIELDS``, ``PLANTING DETAILS``, ``INITIAL
  CONDITIONS``, ``IRRIGATION AND WATER MANAGEMENT``, ``FERTILIZERS (INORGANIC)``, ...) --
  ``{level: dict}``. Header blocks with one row per level are merged into the level dict
  (the two ``*FIELDS`` lines, for instance); schedule / layer blocks, whose first column
  after the level is a date or ``ICBL`` (``IDATE``, ``FDATE``, ``ICBL``...), go to the level
  dict's ``"rows"`` list.

The level key is the first column (``C``, ``L``, ``P``, ``I``, ``F``, ``N``...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._fixed import convert, header_tokens, read_lines, split_fixed

__all__ = ["read_filex", "read_filex_blocks"]

#: first data column of blocks that list several rows per level (schedules, layers)
ROW_KEYS = frozenset({"ICBL", "IDATE", "FDATE", "RDATE", "CDATE", "TDATE", "HDATE", "ODATE", "WMDATE"})


Block = tuple[list[str], list[dict[str, Any]]]


def read_filex_blocks(path: str | Path) -> dict[str, list[Block]]:
    """Low level: ``{section: [(header_names, rows), ...]}`` in file order."""
    lines = read_lines(path)
    out: dict[str, list[Block]] = {}
    section: str | None = None
    toks: list[tuple[str, int, int]] | None = None
    for ln in lines:
        if ln.startswith("*"):
            name = ln[1:].strip()
            # "*TREATMENTS                        -------------FACTOR LEVELS------------"
            name = name.split("  ")[0].strip()
            if name.upper().startswith("EXP.DETAILS"):
                section, toks = None, None
                continue
            section = name
            out.setdefault(section, [])
            toks = None
            continue
        if section is None:
            continue
        if ln.startswith("@"):
            toks = header_tokens(ln)
            out[section].append(([t[0] for t in toks], []))
            continue
        if toks is None or not ln.strip() or ln.lstrip().startswith("!"):
            continue
        names, rows = out[section][-1]
        rows.append({n: convert(v) for n, v in zip(names, split_fixed(toks, ln), strict=True)})
    return out


def _levels(blocks: list[Block]) -> dict[int, dict[str, Any]]:
    res: dict[int, dict[str, Any]] = {}
    for names, rows in blocks:
        if not rows or not names:
            continue
        lev_key = names[0]
        is_rows = len(names) > 1 and names[1] in ROW_KEYS
        for r in rows:
            lev = r[lev_key]
            if not isinstance(lev, int):
                continue
            d = res.setdefault(lev, {})
            body = {k: v for k, v in r.items() if k != lev_key}
            if is_rows:
                d.setdefault("rows", []).append(body)
            else:
                d.update(body)
    return res


def _sim_controls(blocks: list[Block]) -> dict[int, dict[str, dict[str, Any]]]:
    res: dict[int, dict[str, dict[str, Any]]] = {}
    for names, rows in blocks:
        if len(names) < 2 or not rows:
            continue
        group = names[1]
        for r in rows:
            lev = r[names[0]]
            if not isinstance(lev, int):
                continue
            res.setdefault(lev, {})[group] = {k: v for k, v in r.items() if k not in names[:2]}
    return res


def read_filex(path: str | Path) -> dict[str, Any]:
    """Read a FileX into ``{section: content}`` (see module docstring)."""
    blocks = read_filex_blocks(path)
    out: dict[str, Any] = {}
    for sec, bl in blocks.items():
        key = sec.upper()
        if key.startswith("TREATMENTS"):
            out[sec] = [r for _, rows in bl for r in rows]
        elif key.startswith("SIMULATION CONTROLS"):
            out[sec] = _sim_controls(bl)
        elif key in {"GENERAL", "NOTES"}:
            out[sec] = {names[0]: rows for names, rows in bl}
        else:
            out[sec] = _levels(bl)
    return out
