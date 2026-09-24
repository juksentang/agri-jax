"""Management events of an RZWQM2 ``rzwqm.dat``: planting, harvest, fertilizer, pesticide, tillage.

:func:`read_management` resolves the management blocks the way ``Rzman.for`` (``MAQUE``,
``MAFERT``) schedules them and returns one row per event (columns :data:`EVENT_COLUMNS`);
:func:`frame_records` turns such a frame (or ``events.csv``) into
:class:`~agrijax.core.events.EventTable` records with the species, depth and codes kept.
Nothing here imports JAX.

* PLANT MANAGEMENT: one 3-record entry per planting (date, row spacing, depth layer, density);
  harvest option 3 = fixed date (the only option supported).
* FERTILIZER / PESTICIDE: records bound to a plant reference. Timing 1 = ``offset`` days before
  planting, 2 = ``offset`` days after planting, 5 = a fixed date. ``MAQUE`` checks a record only
  while its plant reference is the crop being managed (``IFPL(IR).EQ.IRPL``: the crop in the
  field, else this year's next planting), for fixed dates too; relative records count from that
  crop's planting day of the current year (:func:`_scheduler`). On any day only the first
  matching fertilizer record fires (``GOTO 150``); every matching pesticide record fires.
  Timings 3/4 (emergence / harvest based) and 6/7 (split applications) depend on the
  simulation and raise ``NotImplementedError``.
* A fertilizer record gives NO3-N, NH4-N and urea-N (records 2.5-2.7, kg N/ha); each non-zero
  form is one ``fertilizer_<form>`` row. Placement follows ``MAFERT``: methods 1 (broadcast,
  left on the surface) put every form in the top layer, depth 0; methods 3 / 4 (NH3 injector,
  without / with N-serve) put NH4-N at the injector depth (``BMPTIL``, item 11 of the BMP record,
  spread +-10 cm) and NO3-N / urea-N on the surface (an :class:`~agrijax.core.events.EventTable`
  holds one depth per day, so injected NH4-N together with NO3-N or urea-N raises there).
  Method 2 (broadcast and incorporated: RZWQM queues a field-cultivator pass), 5 (irrigation
  water) and 6 (BMP-computed amounts) depend on the simulation and raise ``NotImplementedError``.
  The ``detail`` of a row holds ``method=<name>`` and, for methods 3 / 4, ``depth_cm=<depth>``.
* TILLAGE: record ``ref when offset[dd mm yyyy] implement depth intensity operation [pmix]``; the
  timings and the plant-reference gate are those of fertilizer (3/4 raise), and on any day only
  the first matching record fires (``GOTO 120``). ``value`` is the implement depth in cm.
* MANURE: a non-zero count raises ``NotImplementedError`` (organic N would be lost silently).
  IRRIGATION: raises likewise unless ``"irrigation"`` is in ``skip`` (the irrigation schedule is
  not parsed yet).

Same-day rows are sorted in the order RZWQM reports them in ``MANAGE.OUT`` (tillage, pesticide,
fertilizer, planting, harvest).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dat import RzwqmDat, read_rzwqm_dat

__all__ = [
    "EVENT_COLUMNS",
    "FERT_DEPTH_CM",
    "FERT_METHOD",
    "HARVEST_TYPE",
    "PEST_METHOD",
    "TILL_IMPLEMENT",
    "TILL_OPERATION",
    "frame_records",
    "read_management",
]

EVENT_COLUMNS = ("date", "event", "value", "unit", "source_line", "detail")

FERT_METHOD = {
    1: "broadcast-surface",
    2: "broadcast-incorporated",
    3: "injected-NH3",
    4: "injected-NH3-nserve",
    5: "irrigation-water",
    6: "BMP",
}
"""``rzwqm.dat`` fertilizer method (record 2.4) -> name used in ``detail``."""
FERT_DEPTH_CM = {1: 0.0}
"""Application depth [cm] of the methods whose placement does not depend on another record."""
PEST_METHOD = {
    1: "broadcast-surface",
    2: "broadcast-incorporated",
    3: "foliar",
    4: "irrigation-water",
    5: "microencapsulated-surface",
    6: "microencapsulated-incorporated",
    7: "soil-surface",
    8: "fumigation",
}
TILL_IMPLEMENT = dict(
    enumerate(
        (
            "moldboard plow, chisel plow-straight, chisel plow-twisted, field cultivator, tandem disk, "
            "offset disk, one-way disk, paraplow, spike tooth harrow, spring tooth harrow, rotary hoe, "
            "bedder ridge, v-blade sweep, subsoiler, rototiller, roller package, "
            "row planter w/ smooth coulter, row planter w/ fluted coulter, row planter w/ sweeps, "
            "lister planter, drill, drill w/chain drag, row cultivator w/sweeps, "
            "row cultivator w/spider wheels, rod weeder, rolling cultivator, nh3 applicator, "
            "ridge-till cultivator, ridge-till planter"
        ).split(", "),
        start=1,
    )
)
"""``rzwqm.dat`` tillage implement (record 2.4) -> name."""
TILL_OPERATION = {1: "primary", 2: "secondary", 3: "tertiary"}
HARVEST_TYPE = {3: "seeds", 4: "above-ground biomass", 5: "roots", 6: "whole plant"}

_FERT_FORMS = ("no3", "nh4", "urea")


def _block(dat: RzwqmDat, key: str) -> Any:
    k = key.upper()
    for b in dat.blocks:
        if k in b.header.upper():
            return b
    raise KeyError(f"no data block after a comment containing {key!r}")


def _d(dd: str | int, mm: str | int, yyyy: str | int) -> np.datetime64:
    return np.datetime64(f"{int(yyyy):04d}-{int(mm):02d}-{int(dd):02d}", "D")


def _fixed_date(toks: list[str], i_off: int) -> np.datetime64:
    return _d(toks[i_off], toks[i_off + 1], toks[i_off + 2])


def _records(dat: RzwqmDat, key: str) -> list[tuple[int, list[str]]]:
    b = _block(dat, key)
    n = int(dat.tokens(b.start)[0])
    return [(ln, dat.tokens(ln)) for ln in range(b.start + 1, b.start + 1 + n)]


def _pesticide_names(dat: RzwqmDat) -> list[str]:
    b = _block(dat, "GENERAL PESTICIDE INFORMATION")
    n = int(dat.tokens(b.start)[0])
    # record 2 (name) is the line after the count; records 3-6 follow per pesticide (5 lines)
    return [dat.line(b.start + 1 + 5 * k).strip() for k in range(n)]


def _injector_depth_cm(dat: RzwqmDat) -> float:
    """``BMPTIL``: item 11 of the BMP record (depth of the anhydrous injector, cm)."""
    b = _block(dat, "B M P   M A N A G E M E N T")
    return float(dat.tokens(b.start)[10])


def _scheduler(dat: RzwqmDat, plantings: list[Any], simulation_start: np.datetime64 | None):
    """``firing(key, i_off, what, first_only)``: the days on which ``MAQUE`` applies each record.

    A day loop from ``simulation_start`` (default: 1 January of the first year the plantings
    and fixed dates span) with ``MAQUE``'s state:

    * ``IPL``, the crop in the field as ``MAQUE`` sees it: ``PLANT`` runs after ``MAQUE``, so a
      planting counts from the day after its planting date to the day after its harvest date
      (``IPL`` is cleared by the ``FIRST11`` pass of the next day's ``PLANT``); a planting dated
      before ``simulation_start`` never happens; planting is assumed on the planting date (the
      soil-water planting window is not modelled); else 0;
    * ``IRPL``, the crop being managed: with no crop in the field, the reference of this year's
      plantings up to the first one dated today or later (file order); with a crop in the field,
      ``IPL``; otherwise unchanged from the day before (``SAVE``; ``DATA IRPL /1/``);
    * ``JPLNT``, the day of year of the planting ``IRPL`` was taken from this year, else 999.

    A record is checked only when its plant reference equals ``IRPL``; timing 1 fires on
    ``JPLNT - offset``, timing 2 on ``JPLNT + offset`` (or that day minus the previous year's
    length), timing 5 on its date. Tillage and fertilizer fire only their first matching record
    of a day (``GOTO 120`` / ``GOTO 150``); every matching pesticide record fires.
    """
    pl = [
        (
            p.plant_ref,
            p.planting_date,
            p.harvest_date if p.harvest_date is not None else p.planting_date,
        )
        for p in plantings
    ]

    def doy(d: np.datetime64) -> int:
        return int((d - d.astype("datetime64[Y]").astype("datetime64[D]")).astype(int)) + 1

    def year(d: np.datetime64) -> int:
        return int(d.astype("datetime64[Y]").astype(int)) + 1970

    def firing(
        key: str, i_off: int, what: str, *, first_only: bool
    ) -> list[tuple[np.datetime64, int, list[str], int]]:
        """``(date, line, tokens, date shift)`` of the records that fire."""
        recs = []
        for ln, t in _records(dat, key):
            ref, when = int(t[0]), int(t[i_off - 1])
            if when == 5:
                recs.append((ln, t, ref, when, 0, _fixed_date(t, i_off)))
            elif when in (1, 2):
                recs.append((ln, t, ref, when, int(float(t[i_off])), None))
            elif any(r == ref for r, _, _ in pl):
                raise NotImplementedError(
                    f"rzwqm.dat line {ln}: {what} timing {when} (phenology-based or split) is not supported"
                )
        if not recs:
            return []
        years = [year(a) for _, a, _ in pl] + [year(h) for _, _, h in pl]
        years += [year(f) for *_, f in recs if f is not None]
        first = np.datetime64(f"{min(years):04d}-01-01", "D")
        if simulation_start is not None:
            first = simulation_start
        planted = [(r, a, h) for r, a, h in pl if a >= first]
        last = np.datetime64(f"{max(years) + 1:04d}-12-31", "D")
        by_year: dict[int, list[tuple[int, np.datetime64]]] = {}
        for r, a, _ in pl:
            by_year.setdefault(year(a), []).append((r, a))
        out: list[tuple[np.datetime64, int, list[str], int]] = []
        irpl = 1  # DATA IRPL /1/
        one = np.timedelta64(1, "D")
        d = first
        while d <= last:
            y, jday = year(d), doy(d)
            iyp = 366 if (y - 1) % 4 == 0 else 365  # MAQUE: MOD(IYYY-1,4)
            ipl = next((r for r, a, h in planted if a < d <= h + one), 0)
            jplnt = 999
            if ipl == 0:
                for r, a in by_year.get(y, ()):
                    jplnt, irpl = doy(a), r
                    if d <= a:
                        break
            else:
                for r, a in by_year.get(y, ()):
                    if r == ipl:
                        irpl, jplnt = ipl, doy(a)
            for ln, t, ref, when, off, fixed in recs:
                if ref != irpl:
                    continue
                if when == 5:
                    hit = d == fixed
                elif when == 1:
                    hit = jday == jplnt - off
                else:
                    hit = jday == jplnt + off or jday + iyp == jplnt + off
                if hit:
                    out.append((d, ln, t, 2 if when == 5 else 0))
                    if first_only:
                        break
            d = d + one
        return out

    return firing


def read_management(
    dat: RzwqmDat | str | Path,
    start: str | np.datetime64 | None = None,
    end: str | np.datetime64 | None = None,
    *,
    skip: Iterable[str] = (),
    simulation_start: str | np.datetime64 | None = None,
) -> pd.DataFrame:
    """Management event table of ``rzwqm.dat`` within ``[start, end]`` (module docstring).

    Columns :data:`EVENT_COLUMNS`: ``date`` (datetime64), ``event`` (``planting``, ``harvest``,
    ``fertilizer_no3`` / ``_nh4`` / ``_urea``, ``pesticide``, ``tillage``), ``value`` + ``unit``
    (density seeds/ha, harvest efficiency, kg N/ha, kg a.i./ha, tillage depth cm),
    ``source_line`` (``"rzwqm.dat:<1-based line>"`` of the record) and ``detail``
    (``key=value;...``). ``start`` / ``end`` default to no bound. ``skip`` may hold
    ``"irrigation"`` to accept a file whose irrigation schedule is not parsed.

    ``simulation_start`` is the first simulated day (``IPNAMES.DAT``,
    :func:`agrijax.io.rzwqm.layers.simulation_start`): which crop is in the field, and so which
    fertilizer / tillage records ``MAQUE`` checks, depends on it (a planting dated before it never
    happens, and ``IRPL`` starts at 1 on it). The default is 1 January of the first year the
    plantings and fixed dates span.
    """
    if not isinstance(dat, RzwqmDat):
        dat = read_rzwqm_dat(dat)
    skipped = {s.strip().lower() for s in skip}
    name = Path(dat.path).name if dat.path is not None else "rzwqm.dat"
    plants = dat.plants
    rows: list[dict[str, Any]] = []

    def add(date, event, value, unit, ln, detail="", order=0):
        rows.append(
            {
                "date": date,
                "event": event,
                "value": float(value),
                "unit": unit,
                "source_line": f"{name}:{ln}",
                "detail": detail,
                "_order": order,
            }
        )

    plantings = dat.plantings
    for p in plantings:
        crop = plants[p.plant_ref - 1].split(maxsplit=1)[-1] if p.plant_ref - 1 < len(plants) else ""
        add(
            p.planting_date,
            "planting",
            p.density_seeds_ha,
            "seeds/ha",
            p.line_no,
            f"plant_ref={p.plant_ref};crop={crop};row_spacing_cm={p.row_spacing_cm:g};"
            f"depth_layer={p.planting_depth_layer}",
            order=3,
        )
        if p.harvest_option == 3:
            if p.harvest_date is None:
                raise ValueError(f"line {p.line_no + 1}: harvest option 3 without a valid date")
            add(
                p.harvest_date,
                "harvest",
                p.harvest_efficiency,
                "fraction",
                p.line_no + 1,
                f"type={HARVEST_TYPE.get(p.harvest_type, p.harvest_type)};stubble_cm={p.stubble_height_cm:g}",
                order=4,
            )
        else:
            raise NotImplementedError(
                f"line {p.line_no + 1}: harvest option {p.harvest_option} (phenology-based)"
            )

    firing = _scheduler(
        dat, plantings, None if simulation_start is None else np.datetime64(simulation_start, "D")
    )

    # fertilizer: ref when offset[dd mm yyyy] method NO3 NH4 UREA bmp_chem bmp_app ...
    for dt, ln, t, s in firing("F E R T I L I Z E R", 2, "fertilizer", first_only=True):
        m = int(t[3 + s])
        if m not in (1, 3, 4):
            raise NotImplementedError(
                f"{name} line {ln}: fertilizer method {m} ({FERT_METHOD.get(m, '?')}) depends on the "
                "simulation (implied tillage, irrigation or BMP amounts) and is not supported"
            )
        meth = FERT_METHOD[m]
        amounts = dict(zip(_FERT_FORMS, (float(t[4 + s]), float(t[5 + s]), float(t[6 + s])), strict=True))
        nz = {k: v for k, v in amounts.items() if v != 0.0} or {"no3": 0.0}
        for k, v in nz.items():
            detail = f"method={meth}"
            if m in (3, 4):  # MAFERT: NH4-N at the injector, NO3-N and urea-N on the surface
                detail += f";depth_cm={_injector_depth_cm(dat) if k == 'nh4' else 0.0:g}"
            add(dt, f"fertilizer_{k}", v, "kg/ha", ln, detail, order=2)

    pest_names = _pesticide_names(dat)
    for dt, ln, t, s in firing("P E S T I C I D E   M A N A G E M E N T", 3, "pesticide", first_only=False):
        k = int(t[1])
        add(
            dt,
            "pesticide",
            float(t[5 + s]),
            "kg a.i./ha",
            ln,
            f"pesticide={pest_names[k - 1] if k - 1 < len(pest_names) else k};"
            f"method={PEST_METHOD.get(int(t[4 + s]), t[4 + s])}",
            order=1,
        )

    # tillage: ref when offset[dd mm yyyy] implement depth intensity operation [pmix]
    for dt, ln, t, s in firing("T I L L A G E", 2, "tillage", first_only=True):
        imp = int(t[3 + s])
        add(
            dt,
            "tillage",
            float(t[4 + s]),
            "cm",
            ln,
            f"implement={TILL_IMPLEMENT.get(imp, imp)};intensity={t[5 + s]};"
            f"operation={TILL_OPERATION.get(int(t[6 + s]), t[6 + s])}",
            order=0,
        )

    for key, what in (("M A N U R E", "manure"), ("I R R I G A T I O N", "irrigation")):
        b = _block(dat, key)
        if int(dat.tokens(b.start)[0]) != 0 and what not in skipped:
            raise NotImplementedError(f"{name} line {b.start}: {what} applications are not parsed yet")

    df = pd.DataFrame(rows, columns=[*EVENT_COLUMNS, "_order"])
    if start is not None:
        df = pd.DataFrame(df.loc[df["date"] >= np.datetime64(start, "D")])
    if end is not None:
        df = pd.DataFrame(df.loc[df["date"] <= np.datetime64(end, "D")])
    df = df.sort_values(by=["date", "_order"], kind="mergesort").drop(columns="_order").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return pd.DataFrame(df.loc[:, list(EVENT_COLUMNS)])


# ------------------------------------------------------------------ frame -> EventTable records


def _detail(detail: Any) -> dict[str, str]:
    s = "" if detail is None or (isinstance(detail, float) and np.isnan(detail)) else str(detail)
    return dict(p.split("=", 1) for p in s.split(";") if "=" in p)


def _named_code(names: Mapping[int, str], kv: dict[str, str], key: str, what: str, src: str) -> str | None:
    """Payload code from ``<key>_code=voc:code`` or from an RZWQM2 name ``<key>=<name>``."""
    if f"{key}_code" in kv:
        return kv[f"{key}_code"]
    if key not in kv:
        return None
    rev = {v.lower(): k for k, v in names.items()}
    n = rev.get(kv[key].strip().lower())
    if n is None:
        raise ValueError(f"{src}: {what} {kv[key]!r} is not an RZWQM2 name ({sorted(rev)})")
    return f"rzwqm2:{n}"


_IRRIG_UNIT_TO_CM = {"": 1.0, "cm": 1.0, "mm": 0.1}
_N_UNITS = {"kg/ha", "kg n/ha", "kg ha-1", "kg n ha-1"}


def frame_records(df: pd.DataFrame, *, source: str = "events table") -> list[tuple[Any, str, Any]]:
    """Rows of an event frame (:data:`EVENT_COLUMNS`) -> ``(date, kind, value)`` records.

    * irrigation: ``mm`` converted to cm;
    * ``fertilizer_<form>``: ``{form: value, method: code, depth_cm: depth}``; the method is
      ``method_code=<voc:code>`` or an RZWQM2 name ``method=<name>`` (:data:`FERT_METHOD`); the
      depth is ``depth_cm=`` or the method's depth (:data:`FERT_DEPTH_CM`); unit kg N/ha;
    * tillage: ``{depth_cm: value, implement: code, operation: 1..3}`` from
      ``implement[_code]=`` and ``operation=`` (:data:`TILL_IMPLEMENT`, :data:`TILL_OPERATION`);
    * residue: ``{amount_kg_ha: value}`` plus ``n_kg_ha=``, ``depth_cm=``, ``incorp_pct=``;
    * priming: ranks from ``rank_lo=<int>;rank_hi=<int>``;
    * any other row: ``(date, event, value)`` unchanged.
    """
    records: list[tuple[Any, str, Any]] = []
    for date, kind, value, unit, detail in zip(
        df["date"], df["event"], df["value"], df["unit"], df["detail"], strict=True
    ):
        k = str(kind).strip().lower()
        u = "" if unit is None or (isinstance(unit, float) and np.isnan(unit)) else str(unit).strip().lower()
        kv = _detail(detail)
        v: Any = value
        if k in {"irrigation", "irrig"}:
            if u not in _IRRIG_UNIT_TO_CM:
                raise ValueError(f"{source}: irrigation unit {unit!r} not in {list(_IRRIG_UNIT_TO_CM)}")
            v = float(value) * _IRRIG_UNIT_TO_CM[u]
        elif k.startswith("fertilizer_"):
            form = k.removeprefix("fertilizer_")
            if u not in _N_UNITS:
                raise ValueError(f"{source}: {k} unit {unit!r} is not kg N/ha")
            code = _named_code(FERT_METHOD, kv, "method", "fertilizer method", source)
            if "depth_cm" in kv:
                depth: float | None = float(kv["depth_cm"])
            elif code is not None and "method_code" not in kv:
                m = int(str(code).split(":")[1])
                if m not in FERT_DEPTH_CM:
                    raise ValueError(f"{source}: {k} on {date} with method {kv['method']!r} needs depth_cm=")
                depth = FERT_DEPTH_CM[m]
            else:
                depth = None
            v = {form: float(value), "method": code, "depth_cm": depth}
        elif k in {"tillage", "till"}:
            if u not in {"", "cm"}:
                raise ValueError(f"{source}: tillage unit {unit!r} is not cm")
            op = kv.get("operation")
            rev_op = {n: c for c, n in TILL_OPERATION.items()}
            v = {
                "depth_cm": float(value),
                "implement": _named_code(TILL_IMPLEMENT, kv, "implement", "tillage implement", source),
                "operation": None
                if op is None
                else rev_op.get(op.strip().lower(), int(op) if op.isdigit() else 0),
            }
        elif k == "residue":
            v = {"amount_kg_ha": float(value)}
            for key in ("n_kg_ha", "depth_cm", "incorp_pct"):
                if key in kv:
                    v[key] = float(kv[key])
        elif k == "priming":
            v = (int(kv["rank_lo"]), int(kv["rank_hi"]))
        records.append((date, str(kind), v))
    return records
