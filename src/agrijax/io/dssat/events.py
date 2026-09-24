"""Management events of one DSSAT treatment: planting, inorganic fertilizer, residue, tillage.

:func:`read_treatment_management` resolves the ``*FERTILIZERS``, ``*RESIDUES``, ``*TILLAGE`` and
``*PLANTING DETAILS`` levels of a FileX treatment the way DSSAT-CSM v4.8 (``IPMAN``, ``IPTILL``,
``Fert_Place``, ``OM_Place``, ``Tillage``) reads and applies them, and returns one row per event
in the columns of :data:`agrijax.io.rzwqm.events.EVENT_COLUMNS`, so the same
:meth:`EventTable.from_frame <agrijax.core.events.EventTable.from_frame>` builds the table
(:func:`treatment_events`). Nothing here imports JAX.

Fertilizer (``FDATE FMCD FACD FDEP FAMN``)
    ``FAMN`` is total N [kg N/ha]; ``Fert_Place`` splits it into NO3-N, NH4-N and urea-N with the
    percentages of the fertilizer type ``FMCD`` in ``FERCH048.SDA`` (:func:`read_fertilizer_types`,
    read with DSSAT's format ``(6X,A35,3X,8F6.0,3X,A3,F6.0,6A6)``); a type without N content
    (``FertN`` not positive) applies no N. Each non-zero form is one ``fertilizer_<form>`` row with
    ``detail`` ``method_code=dssat:<n>;depth_cm=<FDEP>``, where ``n`` is the number of ``FACD``
    (``APnnn``; an invalid code is method 1, as ``IPMAN`` rewrites it). ``FAMN`` or ``FDEP``
    negative, or an invalid ``FMCD``, raise (``IPMAN`` stops). Controlled-release types
    (``NRL50 > 0``) and types with a urease or nitrification inhibitor raise
    ``NotImplementedError``: their N is released over time or their inhibitor window would be lost
    (plan item L1).
Residue (``RDATE RCOD RAMT RESN RINP RDEP``)
    ``value`` is ``RAMT`` [kg dry matter/ha], ``n_kg_ha = RAMT * RESN / 100``; ``RINP`` and
    ``RDEP`` follow ``IPMAN`` and ``OM_Place``: ``RINP < 0`` becomes 100 % with ``RDEP >= 15`` cm
    if ``RDEP > 0``, else 0 %; ``RINP > 0`` raises ``RDEP`` to at least 15 cm; no depth means no
    incorporation. Zero-amount rows are dropped; a missing ``RESN`` (``OM_Place`` then takes a
    ``RESCH048.SDA`` default) raises ``NotImplementedError``.
Tillage (``TDATE TIMPL TDEP``)
    ``value`` is the depth [cm] as ``Tillage`` uses it: ``TDEP``, or the implement's deepest layer
    in ``TILOP048.SDA`` (:func:`read_tillage_depths`) when ``TDEP`` exceeds it by more than 0.5 cm;
    an operation with depth 0 is not applied, a negative ``TDEP`` raises (``IPTILL`` stops).
    ``detail`` holds ``implement_code=dssat:<n>`` for ``TInnn``.
Timing (``SIMULATION CONTROLS``: ``FERTI``, ``RESID`` of ``MANAGEMENT``; ``TILL`` of ``OPTIONS``)
    ``R``: ``YYDDD`` dates (:func:`~agrijax.io.dssat.wth.parse_dssat_date`, ``first_weather`` as
    DSSAT's ``Y4K_DOY``); ``D``: days after planting (0 = the planting day); ``N``: nothing
    applied; ``A`` (automatic) raises ``NotImplementedError``. Tillage applies with ``TILL`` ``Y``
    or ``R``. A treatment without a simulation-control level uses ``R`` / ``Y``, as ``IPMAN`` does.
    ``Fert_Place`` and ``OM_Place`` leave their day loop at the first record dated after today, so
    a record listed after a later-dated one never fires; that rule is kept.

Single-season runs only: the year shifts of sequence (``Q``) and multi-year runs are not applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from ..rzwqm.events import EVENT_COLUMNS
from .filex import read_filex
from .wth import parse_dssat_date

if TYPE_CHECKING:
    from agrijax.core.events import EventTable

__all__ = [
    "FertilizerType",
    "read_fertilizer_types",
    "read_tillage_depths",
    "read_treatment_management",
    "treatment_events",
]

_FORMS = ("no3", "nh4", "urea")


class FertilizerType(tuple):
    """``(no3_pct, nh4_pct, urea_pct, has_n, slow_or_inhibited)`` of one ``FERCH048.SDA`` row."""

    __slots__ = ()

    def __new__(cls, no3: float, nh4: float, urea: float, has_n: bool, special: bool) -> FertilizerType:
        return super().__new__(cls, (no3, nh4, urea, has_n, special))

    @property
    def pct(self) -> tuple[float, float, float]:
        return self[0], self[1], self[2]

    @property
    def has_n(self) -> bool:
        return bool(self[3])

    @property
    def special(self) -> bool:
        return bool(self[4])


def _f6(s: str) -> float:
    """Fortran ``F6.0`` input of one field (blank -> 0)."""
    t = s.strip()
    return float(t) if t else 0.0


def _default_std_file(name: str) -> Path:
    from agrijax.port.run_fortran import dscsm_paths  # lazy: io does not depend on port otherwise

    _, data = dscsm_paths()
    for p in (data / "StandardData" / name, data / name):
        if p.is_file():
            return p
    raise FileNotFoundError(f"{name} not found under {data}")


def read_fertilizer_types(path: str | Path | None = None) -> dict[int, FertilizerType]:
    """``FERCH048.SDA`` -> ``{type number: FertilizerType}`` as ``FertTypeRead`` reads it.

    Data lines after the ``@CDE`` header up to the next ``*`` line; ``!``, ``@`` and blank lines
    skipped (``IGNORE``). ``has_n`` is ``FertN`` positive or ``var`` (``HASE_check``);
    ``special`` marks controlled release (``NRL50 > 0``) or an inhibitor (``UIEFF`` and
    ``UIDUR``, or ``NIEFF`` and ``NIDUR``, positive).
    """
    p = Path(path) if path is not None else _default_std_file("FERCH048.SDA")
    lines = p.read_bytes().decode("latin-1").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("@CDE")) + 1
    out: dict[int, FertilizerType] = {}
    for ln in lines[start:]:
        if ln.startswith(("*", "$")):
            break
        if not ln.strip() or ln.startswith(("!", "@")):
            continue
        ln = ln.ljust(122)
        ftype = int(ln[2:5])
        v = [
            _f6(ln[44 + 6 * k : 50 + 6 * k]) for k in range(8)
        ]  # NO3% NH4% Urea% UIEFF UIDUR NIEFF NIDUR NRL50
        fert_n = ln[104:110]
        if fert_n[3:6].lower() == "var":
            has_n = True
        else:
            try:
                has_n = _f6(fert_n) > 1e-6
            except ValueError:
                has_n = False
        special = v[7] > 1e-6 or (v[3] > 1e-6 and v[4] > 1e-6) or (v[5] > 1e-6 and v[6] > 1e-6)
        out[ftype] = FertilizerType(v[0], v[1], v[2], has_n, special)
    return out


def read_tillage_depths(path: str | Path | None = None) -> dict[str, float]:
    """``TILOP048.SDA`` -> ``{"TInnn": deepest layer [cm]}`` (the last ``SLB`` of the implement)."""
    p = Path(path) if path is not None else _default_std_file("TILOP048.SDA")
    lines = p.read_bytes().decode("latin-1").splitlines()
    out: dict[str, float] = {}
    cur: str | None = None
    n_data = 0
    for ln in lines:
        if ln.startswith("*"):
            cur, n_data = ln[1:6], 0
            continue
        if cur is None or not ln.strip() or ln.startswith(("!", "@")):
            continue
        n_data += 1
        if n_data == 1:  # CN2T RINP SSDT MIXT HPAN
            continue
        dep = _f6(ln[0:6])
        if dep < 0.01:
            cur = None
            continue
        out[cur] = dep
    return out


def _section(x: dict[str, Any], prefix: str) -> dict[int, Any]:
    for k, v in x.items():
        if k.upper().startswith(prefix):
            return v
    return {}


def _code_number(code: Any) -> int | None:
    s = str(code).strip()
    try:
        n = int(s[2:5]) if len(s) >= 5 else int(s)
    except ValueError:
        return None
    return n


def _yrdoy(v: Any) -> str:
    return f"{v:05d}" if isinstance(v, int) and v < 100000 else str(v).strip()


def read_treatment_management(
    filex: str | Path,
    trno: int,
    *,
    fertilizer_types: dict[int, FertilizerType] | None = None,
    tillage_depths: dict[str, float] | None = None,
    first_weather: date | str | int | None = None,
) -> pd.DataFrame:
    """Planting, fertilizer, residue and tillage events of treatment ``trno`` (module docstring).

    ``fertilizer_types`` / ``tillage_depths`` default to :func:`read_fertilizer_types` /
    :func:`read_tillage_depths` of the DSSAT data directory (:func:`agrijax.port.run_fortran.dscsm_paths`).
    """
    path = Path(filex)
    x = read_filex(path)
    trt = next((t for t in x["TREATMENTS"] if t["N"] == trno), None)
    if trt is None:
        raise ValueError(f"{path.name}: no treatment {trno}")
    sim = x.get("SIMULATION CONTROLS", {}).get(trt.get("SM", 0), {})
    man, opt = sim.get("MANAGEMENT", {}), sim.get("OPTIONS", {})
    ferti = str(man.get("FERTI", "R")).upper() if sim else "R"
    resid = str(man.get("RESID", "R")).upper() if sim else "R"
    till = str(opt.get("TILL", "Y")).upper() if sim else "Y"
    rows: list[dict[str, Any]] = []

    def add(d: date, event: str, value: float, unit: str, where: str, detail: str, order: int) -> None:
        rows.append(
            {
                "date": pd.Timestamp(d),
                "event": event,
                "value": float(value),
                "unit": unit,
                "source_line": f"{path.name}:{where}",
                "detail": detail,
                "_order": order,
            }
        )

    def when(v: Any) -> date:
        return parse_dssat_date(_yrdoy(v), first_weather=first_weather)

    planting: date | None = None
    mp = trt.get("MP", 0)
    if mp:
        pl = _section(x, "PLANTING")[mp]
        if str(man.get("PLANT", "R")).upper() == "R" or not sim:
            planting = when(pl["PDATE"])
            add(planting, "planting", float(pl["PPOP"]), "plants/m2", f"*PLANTING level {mp}", "", 3)

    def fires(recs: list[dict[str, Any]], key: str, mode: str, what: str) -> list[tuple[int, date]]:
        """``(row index, date)`` of the rows that fire (``R`` / ``D``, exit-at-later-record rule)."""
        if mode == "N":
            return []
        if mode not in ("R", "D"):
            raise NotImplementedError(
                f"{path.name} treatment {trno}: {what} option {mode!r} is not supported"
            )
        if mode == "D" and planting is None:
            raise ValueError(f"{path.name} treatment {trno}: {what} option D needs a reported planting date")
        out: list[tuple[int, date]] = []
        latest: date | None = None
        for i, r in enumerate(recs):
            if mode == "R":
                d = when(r[key])
            else:
                assert planting is not None
                d = planting + timedelta(days=int(r[key]))
            if latest is None or d >= latest:  # a later-dated earlier row ends the day loop first
                out.append((i, d))
            latest = d if latest is None else max(latest, d)
        return out

    mf = trt.get("MF", 0)
    if mf:
        types = fertilizer_types if fertilizer_types is not None else read_fertilizer_types()
        recs = _section(x, "FERTI")[mf]["rows"]
        for i, d in fires(recs, "FDATE", ferti, "fertilizer"):
            r = recs[i]
            where = f"*FERTILIZERS level {mf} row {i + 1}"
            famn, fdep = float(r["FAMN"]), float(r["FDEP"])
            if famn < 0 or fdep < 0:
                raise ValueError(f"{path.name} {where}: FAMN {famn} / FDEP {fdep} negative (IPMAN stops)")
            ftype = _code_number(r["FMCD"])
            if ftype is None or not 1 <= ftype < 999 or ftype not in types:
                raise ValueError(f"{path.name} {where}: invalid fertilizer type {r['FMCD']!r}")
            meth = _code_number(r["FACD"])
            if meth is None or not 1 <= meth <= 20:
                meth = 1  # IPMAN rewrites an invalid FACD to AP001
            ft = types[ftype]
            if ft.special and famn > 0:
                raise NotImplementedError(
                    f"{path.name} {where}: fertilizer {r['FMCD']} is controlled-release or inhibited"
                )
            if not ft.has_n or famn == 0:
                continue
            for form, pct in zip(_FORMS, ft.pct, strict=True):
                amt = famn * pct / 100.0
                if amt != 0.0:
                    detail = f"method_code=dssat:{meth};depth_cm={fdep:g};fmcd={r['FMCD']}"
                    add(d, f"fertilizer_{form}", amt, "kg/ha", where, detail, 2)

    mr = trt.get("MR", 0)
    if mr:
        recs = _section(x, "RESID")[mr]["rows"]
        for i, d in fires(recs, "RDATE", resid, "residue"):
            r = recs[i]
            where = f"*RESIDUES level {mr} row {i + 1}"
            ramt = float(r["RAMT"])
            if ramt < 0:
                raise ValueError(f"{path.name} {where}: RAMT {ramt} negative (IPMAN stops)")
            if ramt == 0:
                continue
            resn = max(float(r["RESN"]), 0.0)
            if resn < 1e-3:
                raise NotImplementedError(
                    f"{path.name} {where}: RESN not given (OM_Place takes a RESCH048.SDA default)"
                )
            rinp, rdep = float(r["RINP"]), float(r["RDEP"])
            if rinp < 0:  # IPMAN
                if rdep > 0:
                    rinp, rdep = 100.0, max(rdep, 15.0)
                else:
                    rinp = 0.0
            if rinp > 0 and rdep < 15.0:
                rdep = max(rdep, 15.0)
            if rdep < 0.001:  # OM_Place: no depth, no incorporation
                rinp = 0.0
            detail = (
                f"n_kg_ha={ramt * resn / 100.0!r};depth_cm={max(rdep, 0.0):g};incorp_pct={rinp:g};"
                f"rcod={r['RCOD']}"
            )
            add(d, "residue", ramt, "kg/ha", where, detail, 1)

    mt = trt.get("MT", 0)
    if mt and (till in ("Y", "R") or not sim):
        depths = tillage_depths if tillage_depths is not None else read_tillage_depths()
        recs = _section(x, "TILLA")[mt]["rows"]
        for i, r in enumerate(recs):
            where = f"*TILLAGE level {mt} row {i + 1}"
            tdep, timpl = float(r["TDEP"]), str(r["TIMPL"]).strip()
            n = _code_number(timpl)
            if n is None or not 0 < n < 999:
                raise ValueError(f"{path.name} {where}: invalid implement {timpl!r} (IPTILL stops)")
            if tdep < 0:
                raise ValueError(f"{path.name} {where}: TDEP {tdep} negative (IPTILL stops)")
            if timpl not in depths:
                raise ValueError(f"{path.name} {where}: implement {timpl} not in TILOP048.SDA")
            if tdep - depths[timpl] > 0.5:
                tdep = depths[timpl]
            if tdep > 0:
                add(when(r["TDATE"]), "tillage", tdep, "cm", where, f"implement_code=dssat:{n}", 0)

    df = pd.DataFrame(rows, columns=[*EVENT_COLUMNS, "_order"])
    df = df.sort_values(by=["date", "_order"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    return pd.DataFrame(df.loc[:, list(EVENT_COLUMNS)])


def treatment_events(
    filex: str | Path,
    trno: int,
    dates: pd.DatetimeIndex | Sequence[Any],
    *,
    outside: Literal["raise", "drop"] = "drop",
    **kw: Any,
) -> EventTable:
    """:class:`~agrijax.core.events.EventTable` of treatment ``trno`` on the forcing days ``dates``."""
    from agrijax.core.events import EventTable

    df = read_treatment_management(filex, trno, **kw)
    return EventTable.from_frame(df, dates, ignore=(), outside=outside, source=str(filex))
