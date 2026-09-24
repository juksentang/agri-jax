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

Where the header positions and DSSAT's own fixed formats (``IPEXP``, ``IPSIM`` ...) could read a
line differently, DSSAT's reading is followed:

* treatment ``N R O C`` are read with ``(I3,I1,1X,I1,1X,I1)`` (``IPEXP`` format 55), or
  ``(2I2,1X,I1,1X,I1)`` (format 56) for a sequence experiment (run mode ``Q``, the ``.SQX``
  files), where `` 110 1 0`` is treatment 1, rotation 10;
* DSSAT reads the simulation-control lines of a level in order, so when a group appears twice
  for one level the first line is the one used (a later duplicate is ignored).

Values are kept as written otherwise: DSSAT upper-cases the simulation-control flags
(``UPCASE``), reads a ``-99`` option code under ``(5X,A1)`` as ``9`` (meaning "not set"),
keeps the first 16 characters of ``CNAME``; none of that is applied here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._fixed import convert, header_tokens, read_lines, split_fixed

__all__ = ["read_filex", "read_filex_blocks"]

#: first data column of blocks that list several rows per level (schedules, layers)
ROW_KEYS = frozenset({"ICBL", "IDATE", "FDATE", "RDATE", "CDATE", "TDATE", "HDATE", "ODATE", "WMDATE"})


Block = tuple[list[str], list[dict[str, Any]]]


def _fortran_int(text: str) -> int | None:
    """Fortran ``I`` input of one field (blank -> 0), ``None`` when it is not an integer."""
    t = text.strip()
    if not t:
        return 0
    try:
        return int(t)
    except ValueError:
        return None


def _treatment_keys(line: str, *, sequence: bool) -> dict[str, int] | None:
    """``N R O C`` of a treatment line by DSSAT's fixed format (``IPEXP`` formats 55 / 56)."""
    cols = (
        {"N": (0, 2), "R": (2, 4), "O": (5, 6), "C": (7, 8)}
        if sequence
        else {"N": (0, 3), "R": (3, 4), "O": (5, 6), "C": (7, 8)}
    )
    out: dict[str, int] = {}
    for k, (a, b) in cols.items():
        v = _fortran_int(line[a:b])
        if v is None:
            return None
        out[k] = v
    return out


def read_filex_blocks(path: str | Path, *, sequence: bool | None = None) -> dict[str, list[Block]]:
    """Low level: ``{section: [(header_names, rows), ...]}`` in file order.

    ``sequence`` (default: the file is a ``.SQX``) selects DSSAT's run-mode ``Q`` treatment
    format (see module docstring).
    """
    if sequence is None:
        sequence = Path(path).suffix.upper() == ".SQX"
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
        row = {n: convert(v) for n, v in zip(names, split_fixed(toks, ln), strict=True)}
        if section.upper().startswith("TREATMENTS") and names[:4] == ["N", "R", "O", "C"]:
            keys = _treatment_keys(ln, sequence=sequence)
            if keys is not None:
                row.update(keys)
        rows.append(row)
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
            # DSSAT reads the lines of a level in order: the first line of a group is used
            res.setdefault(lev, {}).setdefault(group, {k: v for k, v in r.items() if k not in names[:2]})
    return res


def read_filex(path: str | Path, *, sequence: bool | None = None) -> dict[str, Any]:
    """Read a FileX into ``{section: content}`` (see module docstring; ``sequence`` as in
    :func:`read_filex_blocks`)."""
    blocks = read_filex_blocks(path, sequence=sequence)
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
