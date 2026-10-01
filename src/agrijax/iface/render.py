"""Markdown tables of the coupling contract, rendered from :mod:`agrijax.iface.contract`.

Documents quote these tables instead of copying them, so the text cannot drift from the code::

    python -m agrijax.iface.render                 # ports and day, both stages
    python -m agrijax.iface.render --stage m3      # the ``m3`` stage (RZWQM2 4.6 day) only
    python -m agrijax.iface.render --what ports --slot soy

The output is deterministic (the tables' order, no timestamps); ``tests/unit/test_iface_post_m3.py``
checks that every port and row appears.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence

from .contract import PORTS, STAGES, day_table, entry_ports_out

__all__ = ["main", "render", "render_day", "render_ports"]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_ports(stages: Iterable[str] = STAGES) -> str:
    """One row per port of ``stages``: id, stage, path, record, fields (unit, dims; pending ones in
    italics), producers, consumers, time and allowed lags, default and off behaviour."""
    st = tuple(stages)
    lines = [
        "| port | stage | path | record | fields | producers | consumers | time (lag readers) "
        "| off / default |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for pid, sp in PORTS.items():
        if sp.stage not in st:
            continue
        rec = sp.record.__name__ if sp.record is not None else f"({sp.field_of.rsplit('.', 1)[-1]})"
        fields = [f"`{n}` {f.unit} ({', '.join(f.dims)})" for n, f in sp.fields]
        fields += [f"*`{n}` {f.unit} ({', '.join(f.dims)}), pending*" for n, f, _ in sp.pending]
        lags = ", ".join(lag.reader + (f".{lag.field}" if lag.field else "") for lag in sp.lags)
        time = sp.time + (f" ({lags})" if lags else "")
        off = sp.off or sp.default
        named = {p.split(" ", 1)[0] for p in sp.producers}
        extra = [f"{e} ({', '.join(f) or 'all'})" for e, f in sp.writers if e not in named]
        producers = "; ".join((*sp.producers, *(f"+ {x}" for x in extra)))
        cells = (
            pid,
            sp.stage,
            f"`{sp.path}`",
            rec,
            "; ".join(fields),
            producers,
            "; ".join(sp.consumers),
            time,
            off,
        )
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def render_day(slot: str = "{slot}", stages: Iterable[str] = STAGES) -> str:
    """The merged day of ``stages`` in order: row, stage, phase, entry, default key, ports written,
    status, placement (``after``, ``supersedes``) and note."""
    lines = [
        "| row | stage | phase | entry | key | writes | status | placed | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in day_table(tuple(stages)):
        placed = ""
        if e.after:
            placed = f"after {e.after.format(slot=slot)}"
        if e.supersedes:
            placed += "; replaces " + ", ".join(x.format(slot=slot) for x in e.supersedes)
        cells = (
            e.row,
            e.stage,
            e.phase,
            f"`{e.name(slot)}`",
            f"`{e.key}`" if e.key else "",
            ", ".join(entry_ports_out(e)),
            e.status,
            placed,
            e.note,
        )
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def render(what: str = "all", slot: str = "{slot}", stages: Iterable[str] = STAGES) -> str:
    """``ports``, ``day`` or ``all`` (both, with headings)."""
    st = tuple(stages)
    parts = []
    if what in ("ports", "all"):
        parts.append(("## Ports\n\n" if what == "all" else "") + render_ports(st))
    if what in ("day", "all"):
        parts.append(("## Day\n\n" if what == "all" else "") + render_day(slot, st))
    if not parts:
        raise ValueError(f"what must be ports, day or all, not {what!r}")
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrijax.iface.render", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--what", choices=("ports", "day", "all"), default="all")
    ap.add_argument(
        "--stage",
        choices=("m3", "all"),
        default="all",
        help="m3: only the ports and rows of the m3 stage (the day in the RZWQM2 4.6 order)",
    )
    ap.add_argument("--slot", default="{slot}", help="crop slot to fill in (default: the {slot} template)")
    ns = ap.parse_args(argv)
    stages = ("m3",) if ns.stage == "m3" else STAGES
    sys.stdout.write(render(ns.what, ns.slot, stages))
    return 0


if __name__ == "__main__":
    sys.exit(main())
