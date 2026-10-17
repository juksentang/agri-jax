"""DSSAT observations as calibration targets: FILEA / FILET codes -> :class:`~agrijax.calib.objective.Target`.

:func:`observation_targets` turns the observed data of one experiment
(:func:`agrijax.io.dssat.observed.read_observed`) into the targets and observed arrays of one
batch group (:class:`~agrijax.calib.objective.Group`), for the treatments of the group in batch
order and the simulated days of each treatment (``yrdoy[b, t]``, ``YYYYDDD``, as the stacked
forcing of :func:`agrijax.calib.ceres.stack_treatments` holds them).

Only the variables the CERES-Maize free-run model outputs are mapped, through the explicit table
:data:`DSSAT_TARGETS` (output names of :func:`agrijax.processes.crop.ceres_maize.plantgro_outputs`,
the same quantities and units as DSSAT's ``PlantGro.OUT``, validated against it on the DSSAT v4.8.6.0
maize example treatments in ``tests/integration/test_ceres_dssat.py``):

=====  ======  =========  ========  =====================================================
code   file    output     kind      meaning [unit]
=====  ======  =========  ========  =====================================================
HWAM   A       gwad       final     grain yield, dry [kg ha-1] (a ``HWAH`` column too: alias)
CWAM   A       cwad       final     tops weight [kg ha-1] (a ``CWAH`` column too: alias)
H#AM   A       g_ad       final     grain number at maturity [m-2]
L#SM   A       lsd        final     leaf number at maturity
ADAT   A       istage     date      anthesis (silking): first day of ISTAGE 4
MDAT   A       istage     date      physiological maturity: first day of ISTAGE 10
LAID   T       lai        series    leaf area index [m2 m-2]
CWAD   T       cwad       series    tops weight [kg ha-1]
GWAD   T       gwad       series    grain weight [kg ha-1]
LWAD   T       lwad       series    leaf weight [kg ha-1]
SWAD   T       swad       series    stem weight [kg ha-1]
RWAD   T       rwad       series    root weight [kg ha-1]
PWAD   T       pwad       series    ear weight [kg ha-1]
G#AD   T       g_ad       series    grain number [m-2]
L#SD   T       lsd        series    leaf number
=====  ======  =========  ========  =====================================================

**Soil water and evapotranspiration** (for calibrating soil parameters) are mapped only when the
model layers are given (``soil_layers``: the bottom depth of every model layer of each treatment,
``DS`` [cm]); the cultivar workflow does not give them, so its targets stay the crop ones above:

=====  ======  =========  ========  =====================================================
code   file    output     kind      meaning [unit]
=====  ======  =========  ========  =====================================================
SW#D   T       sw#        series    volumetric soil water of model layer ``#`` [cm3 cm-3],
                                    end of day (``SW<#>D`` of ``SoilWat.OUT``)
ETAA   T       et         series    actual evapotranspiration of the day ``EVAP + EP``
                                    [mm d-1] (``ETAA`` of a daily ``ET.OUT``)
=====  ======  =========  ========  =====================================================

``SW#D`` of a FILET names model layer ``#``, the layer of the same column of ``SoilWat.OUT`` that
DSSAT compares it with. This is the **model-layer index even for a split profile**: when DSSAT
splits a ``.SOL`` layer (``UFGA8201`` splits its 30-60 cm layer into model layers 4 and 5), ``SW5D``
is compared with model layer 5 (45-60 cm), not with ``.SOL`` layer 5 (60-90 cm). A FILET carries no
depth, so which reading the experimenters meant cannot be read from the file; this reading follows
the output column name and is an assumption to confirm per experiment (give such data by depth in
``soil_water`` instead when the ``.SOL`` layer is meant). Soil-water observations at a **depth**
(``soil_water``: a table with ``TRNO``, ``YRDOY``, ``DEPTH`` [cm] and ``SW``) are placed in the
model layer that holds the depth (top < depth <= bottom), both join the same per-layer targets
``SW#D`` (named by layer, as DSSAT names them), and :attr:`ObservationTargets.depths` gives the
depth range of each target's layer for every treatment. A soil-water value of 0 or less is no data
(a volumetric water content of 0 is no measurement: ``UFGA8201.MZT`` holds ``0.000`` in
``SW1D``-``SW5D`` of the treatments whose soil water was not measured), and so is a missing
(``-99``) one; a treatment whose every value of a code is 0 (soil water or ``ETAA``) is listed in
:attr:`ObservationTargets.notes` as having no data for it (with no target for a code left with only
such zeros), and a code without a value in any treatment (every value missing) gives no target. The
``YRDOY`` of ``soil_water`` must be a four-digit-year date ``YYYYDDD`` (year 1000 .. 9999). Notes
about the depth table are written only when some ``SW#D`` is requested (``codes`` is ``None`` or
holds one), and its missing or non-positive values are counted only for the layers requested.

FILEA codes are those of the reader, i.e. after ``READA_Y4K``'s header aliases
(:data:`agrijax.io.dssat.observed.FILEA_ALIASES`: ``HWAH`` -> ``HWAM``, ``CWAH`` -> ``CWAM``,
``BWAH`` -> ``BWAM``, ``HDAT`` -> ``R8AT`` ...; of two columns with one name the last wins), so
which of ``HWAM`` / ``HWAH`` is used follows the engine's own rule.

Every other code is **not mapped** and is listed, with the reason, in
:attr:`ObservationTargets.unsupported` (:data:`UNSUPPORTED_REASONS` for the known ones: soil
water, nitrogen, derived ratios, season maxima ...); nothing is guessed.

Conventions:

* a ``final`` target compares the value on the last simulated day (the simulation must end at or
  after maturity / harvest, e.g. the padded forcing of ``stack_treatments``); when some treatments
  of the group lack the observation, the target is written as a ``series`` target observed on the
  last day of the treatments that have it (the same loss);
* a ``date`` target compares day indices on the treatment's simulated days; it needs the
  observation in every treatment of the group (a date target has no mask), otherwise it is left
  out and the reason is in :attr:`ObservationTargets.notes` (put the observed treatments in their
  own group). Day-of-year dates of FILEA are counted from the simulation start (``sim_start``,
  default the first simulated day of the treatment), as DSSAT's ``READA_Dates``; ``YYDDD`` dates of
  both files follow ``Y4K_DOY`` with ``first_weather`` (DSSAT's ``FirstWeatherDate``) when given; a
  FILEA date whose integer part is not positive (``-99``, ``0``, ``0.5``) is missing;
* a ``series`` target observes the FILET dates that fall on simulated days; other dates are
  dropped with a note; repeated observations of one date are averaged;
* the default scale of a ``final`` / ``series`` target is the mean absolute observed value (the
  loss is then a squared normalised RMSE), of a ``date`` target one day; ``scales`` overrides it
  per code.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from agrijax.calib.dssat_day import SW_LAYERS
from agrijax.calib.objective import Group, Simulator, Target
from agrijax.io.dssat.observed import ObservedData, reada_date, y4k_date

__all__ = [
    "DSSAT_TARGETS",
    "UNSUPPORTED_REASONS",
    "WATER_TARGETS",
    "ObservationError",
    "ObservationTargets",
    "TargetMap",
    "observation_targets",
]


class ObservationError(ValueError):
    """An observed target cannot be used (it lies outside the simulated days, or its date is unreadable).

    Defined here (the lowest layer that raises it) and re-exported by :mod:`agrijax.calib.workflow` and
    :mod:`agrijax.calib`."""


#: first day of silking in the model output ``istage``: ``ISTAGE = 4`` (end of leaf growth,
#: ``processes/crop/ceres_maize/constants.py`` ``ISTAGE_END_LEAF_GROWTH``, MZ_PHENOL.for:783); the
#: same code as :data:`agrijax.calib.ceres.ISTAGE_SILKING_OUT` (a unit test checks both). Defined here
#: so that this module does not import the crop model.
ISTAGE_SILKING = 4
#: first day after physiological maturity: ``ISTAGE = 10`` (``ISTAGE_AFTER_MATURITY``,
#: MZ_PHENOL.for:899), as :data:`agrijax.calib.ceres.ISTAGE_MATURITY_OUT`
ISTAGE_AFTER_MATURITY = 10

#: default scale of a date target [d]
DATE_SCALE_DAYS = 1.0

#: a ``FirstWeatherDate``: ``YYYYDDD`` as a number (numpy ones and a whole float too), as text, or a
#: date (also a ``numpy.datetime64``); ``None`` (and NaN): none
_FirstWeather = int | float | np.integer | np.floating | str | date | np.datetime64 | None


@dataclass(frozen=True)
class TargetMap:
    """How one DSSAT observed code maps to a model output."""

    code: str
    file: Literal["A", "T"]
    output: str
    kind: Literal["final", "series", "date"]
    unit: str
    meaning: str
    code_stage: int | None = None


#: the explicit mapping of DSSAT observed codes to CERES-Maize outputs (module docstring table)
DSSAT_TARGETS: dict[str, TargetMap] = {
    m.code: m
    for m in (
        TargetMap("HWAM", "A", "gwad", "final", "kg ha-1", "grain yield (dry; HWAH alias)"),
        TargetMap("CWAM", "A", "cwad", "final", "kg ha-1", "tops weight (CWAH alias)"),
        TargetMap("H#AM", "A", "g_ad", "final", "m-2", "grain number at maturity"),
        TargetMap("L#SM", "A", "lsd", "final", "-", "leaf number at maturity"),
        TargetMap("ADAT", "A", "istage", "date", "d", "anthesis (silking) date", ISTAGE_SILKING),
        TargetMap("MDAT", "A", "istage", "date", "d", "physiological maturity date", ISTAGE_AFTER_MATURITY),
        TargetMap("LAID", "T", "lai", "series", "m2 m-2", "leaf area index"),
        TargetMap("CWAD", "T", "cwad", "series", "kg ha-1", "tops weight"),
        TargetMap("GWAD", "T", "gwad", "series", "kg ha-1", "grain weight"),
        TargetMap("LWAD", "T", "lwad", "series", "kg ha-1", "leaf weight"),
        TargetMap("SWAD", "T", "swad", "series", "kg ha-1", "stem weight"),
        TargetMap("RWAD", "T", "rwad", "series", "kg ha-1", "root weight"),
        TargetMap("PWAD", "T", "pwad", "series", "kg ha-1", "ear weight"),
        TargetMap("G#AD", "T", "g_ad", "series", "m-2", "grain number"),
        TargetMap("L#SD", "T", "lsd", "series", "-", "leaf number"),
    )
}

_SOIL_WATER = "soil water: a soil-calibration target (it needs the model layer depths), not a cultivar one"
_SOIL_WATER_TOTALS = "soil water: profile totals, not layer contents (no model output)"
_EVAPOTRANSPIRATION = (
    "evapotranspiration: a soil-calibration target (it needs the model layer depths), not a cultivar one"
)
#: the soil-water-balance targets (module docstring table), used only with ``soil_layers``: the soil
#: water of every model layer a DSSAT run can have (``SW_LAYERS``) and the daily evapotranspiration
WATER_TARGETS: dict[str, TargetMap] = {
    m.code: m
    for m in (
        *(
            TargetMap(f"SW{k}D", "T", f"sw{k}", "series", "cm3 cm-3", f"soil water of model layer {k}")
            for k in range(1, SW_LAYERS + 1)
        ),
        TargetMap("ETAA", "T", "et", "series", "mm d-1", "actual evapotranspiration of the day (EVAP + EP)"),
    )
}

_NITROGEN = "nitrogen: the CERES model runs with nitrogen off"
_PHOSPHORUS = "phosphorus/potassium: not modelled"
_DERIVED = "derived quantity (ratio or difference of outputs), not a model output"

#: why the known unmapped codes are not mapped (any other code: "no model output")
UNSUPPORTED_REASONS: dict[str, str] = {
    "LAIX": "season maximum: no Target kind for a maximum over the season",
    "CWAA": "value at anthesis: no Target kind for a value at a stage date",
    "CNAA": _NITROGEN,
    "HWUM": _DERIVED + " (grain unit weight = yield / grain number)",
    "H#UM": _DERIVED + " (grains per ear)",
    "BWAM": _DERIVED + " (by-product = tops - grain; BWAH alias)",
    "HIAM": _DERIVED + " (harvest index)",
    "HIAD": _DERIVED + " (harvest index)",
    "GWGD": _DERIVED + " (grain unit weight)",
    "SLAD": _DERIVED + " (specific leaf area)",
    "FRLF": _DERIVED + " (leaf fraction)",
    "FRST": _DERIVED + " (stem fraction)",
    "P#AD": "ear number: CERES reads it from the plant population, it is not simulated",
    "IDAT": "panicle-initiation date: the stage code of tassel initiation is not an observed-date target yet",
    "EDAT": "emergence date: not mapped (the DSSAT comparison test checks it, it is no calibration target)",
    "R8AT": "harvest date (HDAT alias): a management event, not a model output",
    "PDAT": "planting date: a management event, not a model output",
    "CHTA": "canopy height: not a CERES-Maize output",
    "CHTD": "canopy height: not a CERES-Maize output",
}


def _reason(code: str) -> str:
    if code in UNSUPPORTED_REASONS:
        return UNSUPPORTED_REASONS[code]
    if code in {"SWTD", "SWXD", "PESW"}:
        return _SOIL_WATER_TOTALS
    if _SW_CODE.match(code):
        return _SOIL_WATER
    if code == "ETAA":
        return _EVAPOTRANSPIRATION
    if code.startswith(("NI", "NH", "NO")) or "N%" in code or code.endswith(("NAD", "NAM")) or code == "NUPC":
        return _NITROGEN
    if "P%" in code or "K%" in code or code.endswith(("PAD", "KAD", "PAM", "KAM")):
        return _PHOSPHORUS
    return "no CERES free-run output for this code"


@dataclass(frozen=True)
class ObservationTargets:
    """Targets and observed arrays of one batch group, and what was left out."""

    targets: tuple[Target, ...]
    observed: dict[str, np.ndarray]
    trnos: tuple[int, ...]
    #: ``{code: reason}`` for every observed code that is not mapped
    unsupported: dict[str, str] = field(default_factory=dict)
    #: mapped codes left out or partly used, with the reason
    notes: tuple[str, ...] = ()
    #: finite observations used per target
    n_obs: dict[str, int] = field(default_factory=dict)
    #: ``{SW#D target: [(top, bottom) of model layer # per treatment, cm]}`` (with ``soil_layers``)
    depths: dict[str, tuple[tuple[float, float], ...]] = field(default_factory=dict)

    def group(self, name: str, simulate: Simulator) -> Group:
        """The :class:`Group` of these targets with the simulator of the same treatments."""
        return Group(name, simulate, self.targets, self.observed)


def _yrdoy_int(d: date) -> int:
    return d.year * 1000 + d.timetuple().tm_yday


def _plain_first_weather(f: _FirstWeather | np.ndarray) -> int | str | date | None:
    """A ``FirstWeatherDate`` as the readers take it: a numpy scalar (or 0-d array) becomes a Python
    ``int`` / ``date``, a whole float (a pandas column with gaps is float) the ``int`` of it, and
    NaN / ``NaT`` ``None`` (no date)."""
    v: Any = f
    if isinstance(v, np.ndarray):
        if v.ndim != 0:
            raise ValueError(f"first_weather: a date is a scalar, got an array of shape {v.shape}")
        v = v[()]
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return None
        if not float(v).is_integer():
            raise ValueError(f"first_weather: {v!r} is not a YYYYDDD date")
        return int(v)
    if isinstance(v, (np.datetime64, date)) and pd.isna(v):  # NaT
        return None
    if isinstance(v, np.datetime64):
        return cast("date", v.astype("datetime64[D]").astype(object))
    return int(v) if isinstance(v, np.integer) else v


def _is_one_first_weather(f: object) -> bool:
    """One ``FirstWeatherDate`` (for all treatments) rather than a sequence with one per treatment."""
    scalar = (int, float, np.integer, np.floating, str, date, np.datetime64)
    return f is None or isinstance(f, scalar) or (isinstance(f, np.ndarray) and f.ndim == 0)


def observation_targets(
    obs: ObservedData,
    trnos: Sequence[int],
    yrdoy: np.ndarray | Sequence[Sequence[int]],
    *,
    codes: Collection[str] | None = None,
    scales: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
    sim_start: Sequence[int] | None = None,
    first_weather: _FirstWeather | Sequence[_FirstWeather] | np.ndarray = None,
    available_outputs: Collection[str] | None = None,
    soil_layers: Sequence[Sequence[float]] | np.ndarray | None = None,
    soil_water: pd.DataFrame | None = None,
) -> ObservationTargets:
    """Targets of the treatments ``trnos`` (batch order) from their observations.

    ``yrdoy[b, t]`` are the simulated days (``YYYYDDD``) of treatment ``b``, all of one length
    ``T`` (the batch's day axis). ``codes`` restricts the mapped codes used (default every mapped
    code observed); ``scales`` / ``weights`` set a target's scale / weight per code;
    ``sim_start[b]`` is the simulation start of treatment ``b`` (``YYYYDDD``, default
    ``yrdoy[b, 0]``; ``YYDDD`` goes through ``Y4K_DOY``); ``first_weather`` is DSSAT's
    ``FirstWeatherDate`` (``YYYYDDD`` as a number, numpy ones and a whole float included, or a string,
    or a date: one value for all treatments, or a sequence or array with one per treatment; ``None``,
    NaN, ``-99`` and ``0``: the cross-over rule) for the ``Y4K_DOY`` dates of both files;
    ``available_outputs`` (the keys the group's simulator returns) leaves out,
    with a note, the targets whose output the simulator does not produce.

    ``soil_layers[b]`` (the bottom depth of every model layer of treatment ``b``, ``DS`` [cm],
    increasing) maps the soil-water-balance codes (:data:`WATER_TARGETS`: ``SW#D`` and ``ETAA`` of
    FILET) and the depth observations ``soil_water`` (columns ``TRNO``, ``YRDOY`` (``YYYYDDD``),
    ``DEPTH`` [cm], ``SW`` [cm3 cm-3]; module docstring); without it they are unsupported codes.
    """
    days = np.asarray(yrdoy, dtype=np.int64)
    trn = tuple(int(t) for t in trnos)
    if days.ndim != 2 or days.shape[0] != len(trn):
        raise ValueError(f"yrdoy must be [B, T] with B = {len(trn)} treatments, got shape {days.shape}")
    b_n, t_n = days.shape
    starts = [int(s) for s in (sim_start if sim_start is not None else days[:, 0])]
    fws: list[int | str | date | None]
    if _is_one_first_weather(first_weather):
        # one FirstWeatherDate for all treatments: a number (also a numpy one), a YYYYDDD text or a date
        fws = [_plain_first_weather(cast("_FirstWeather", first_weather))] * b_n
    else:
        fws = [_plain_first_weather(f) for f in cast("Iterable[_FirstWeather]", first_weather)]
        if len(fws) != b_n:
            raise ValueError(f"first_weather: {len(fws)} values for {b_n} treatments")
    bottoms = _soil_layers(soil_layers, b_n)
    if soil_water is not None and bottoms is None:
        raise ValueError("soil_water: observations by depth need soil_layers (the model layer depths)")
    scales = dict(scales or {})
    weights = dict(weights or {})
    targets: list[Target] = []
    observed: dict[str, np.ndarray] = {}
    notes: list[str] = []
    n_obs: dict[str, int] = {}
    unsupported: dict[str, str] = {}

    present: list[tuple[str, str]] = []  # (file, code) in file order
    if obs.filea is not None:
        present += [("A", c) for c in obs.filea.codes]
    if obs.filet is not None:
        present += [("T", c) for c in obs.filet.codes]
    for f, c in present:
        m = DSSAT_TARGETS.get(c) or (WATER_TARGETS.get(c) if bottoms is not None else None)
        if m is None or m.file != f:
            unsupported.setdefault(
                c, _reason(c) if m is None else f"{c} is mapped for FILE{m.file}, found in FILE{f}"
            )

    def use(m: TargetMap) -> bool:
        if codes is not None and m.code not in codes:
            return False
        if available_outputs is not None and m.output not in available_outputs:
            notes.append(f"{m.code}: output {m.output!r} not produced by the simulator; left out")
            return False
        return True

    def add(
        m: TargetMap,
        kind: Literal["final", "series", "date"],
        obs_arr: np.ndarray,
        mask: np.ndarray | None,
        n: int,
    ) -> None:
        vals = obs_arr[mask] if mask is not None else obs_arr
        if m.code in scales:
            scale = float(scales[m.code])
        elif kind == "date":
            scale = DATE_SCALE_DAYS
        else:
            scale = float(np.mean(np.abs(vals))) if np.any(vals != 0) else 1.0
        t = Target(
            m.code,
            m.output,
            kind,
            scale=scale,
            weight=float(weights.get(m.code, 1.0)),
            code=m.code_stage,
            mask=mask,
        )
        targets.append(t)
        observed[m.code] = obs_arr
        n_obs[m.code] = n

    # ---- FILEA: final values and dates
    a = obs.filea
    a_codes = set(a.codes) if a is not None else set()
    for m in DSSAT_TARGETS.values():
        if m.file != "A" or m.code not in a_codes or not use(m):
            continue
        assert a is not None
        vals = np.array([a.value(t, m.code) for t in trn], dtype=float)
        # a date counts when READA_Dates' truncated value is positive: 0 < value < 1 is missing
        have = np.isfinite(vals) & (np.trunc(vals) > 0 if m.kind == "date" else np.isfinite(vals))
        if not have.any():
            continue
        if m.kind == "final":
            if have.all():
                add(m, "final", vals, None, b_n)
            else:
                mask = np.zeros((t_n, b_n), dtype=bool)
                mask[-1, have] = True
                arr = np.zeros((t_n, b_n))
                arr[-1, have] = vals[have]
                add(m, "series", arr, mask, int(have.sum()))
                miss = [t for t, h in zip(trn, have, strict=True) if not h]
                notes.append(f"{m.code}: missing for TRNO {miss}; written as a last-day series target")
            continue
        # date target: index of the observed date on the treatment's simulated days
        idx = np.full(b_n, -1.0)
        for b in range(b_n):
            if not have[b]:
                continue
            d = reada_date(float(vals[b]), starts[b], fws[b])
            if d is None:  # defensive: ``have`` kept only finite values with a positive integer part
                raise ObservationError(
                    f"{obs.experiment}_t{trn[b]:02d}: observed {m.code} {vals[b]:g} is not a readable date"
                )
            hit = np.nonzero(days[b] == _yrdoy_int(d))[0]
            if hit.size:
                idx[b] = float(hit[0])
        if (idx < 0).any():
            miss = [t for t, i in zip(trn, idx, strict=True) if i < 0]
            notes.append(
                f"{m.code}: not observed (or outside the simulated days) for TRNO {miss}; "
                "a date target needs every treatment of the group, left out"
            )
            continue
        add(m, "date", idx, None, b_n)

    # ---- FILET: series
    tt = obs.filet
    t_codes = set(tt.codes) if tt is not None else set()
    for m in DSSAT_TARGETS.values():
        if m.file != "T" or m.code not in t_codes or not use(m):
            continue
        assert tt is not None
        arr = np.zeros((t_n, b_n))
        mask = np.zeros((t_n, b_n), dtype=bool)
        outside: list[str] = []
        repeated = 0
        for b, t in enumerate(trn):
            s = tt.series(t, m.code)
            if s.empty:
                continue
            if fws[b] is None or tt.first_weather is not None:
                yds = s["YRDOY"].to_numpy(dtype=np.int64)
            else:  # the file was read without FirstWeatherDate: redo Y4K_DOY with it
                yds = np.array([_yrdoy_int(y4k_date(int(c), fws[b])) for c in s["DATECODE"]], dtype=np.int64)
            vals = s["value"].to_numpy(dtype=float)
            for yd in np.unique(yds):
                hit = np.nonzero(days[b] == yd)[0]
                if not hit.size:
                    outside.append(f"{t}:{int(yd)}")
                    continue
                sel = vals[yds == yd]
                repeated += int(sel.size > 1)
                arr[hit[0], b] = float(np.mean(sel))
                mask[hit[0], b] = True
        if outside:
            notes.append(
                f"{m.code}: {len(outside)} observation dates outside the simulated days dropped "
                f"(TRNO:YRDOY {outside[:5]}...)"
            )
        if repeated:
            notes.append(f"{m.code}: {repeated} dates observed more than once; averaged")
        if mask.any():
            add(m, "series", arr, mask, int(mask.sum()))

    # ---- soil water by layer (FILET SW#D and observations by depth) and daily evapotranspiration
    depths: dict[str, tuple[tuple[float, float], ...]] = {}
    if bottoms is not None:
        cells = _water_cells(tt, soil_water, trn, days, bottoms, fws, notes, codes)
        for m in WATER_TARGETS.values():
            have = cells.get(m.code)
            if not have or not use(m):
                continue
            arr = np.zeros((t_n, b_n))
            mask = np.zeros((t_n, b_n), dtype=bool)
            repeated = 0
            for (t, b), vals in sorted(have.items()):
                repeated += int(len(vals) > 1)
                arr[t, b] = float(np.mean(vals))
                mask[t, b] = True
            if repeated:
                notes.append(f"{m.code}: {repeated} layer-days observed more than once; averaged")
            add(m, "series", arr, mask, int(mask.sum()))
            if m.output != "et":
                k = int(m.output[2:])
                depths[m.code] = tuple(
                    (float(bt[k - 2]) if k > 1 else 0.0, float(bt[k - 1]))
                    if k <= bt.size
                    else (math.nan, math.nan)
                    for bt in bottoms
                )
    return ObservationTargets(tuple(targets), observed, trn, unsupported, tuple(notes), n_obs, depths)


def _soil_layers(
    soil_layers: Sequence[Sequence[float]] | np.ndarray | None, b_n: int
) -> list[np.ndarray] | None:
    """The model layer bottoms of each treatment, checked: positive, increasing, one row per treatment."""
    if soil_layers is None:
        return None
    rows = [np.asarray(r, dtype=float).ravel() for r in soil_layers]
    if len(rows) != b_n:
        raise ValueError(f"soil_layers: {len(rows)} rows for {b_n} treatments")
    for b, r in enumerate(rows):
        if r.size == 0 or not np.all(np.isfinite(r)) or r[0] <= 0.0 or np.any(np.diff(r) <= 0.0):
            raise ValueError(
                f"soil_layers[{b}]: layer bottoms must be positive and increasing, got {r.tolist()}"
            )
        if r.size > SW_LAYERS:
            raise ValueError(f"soil_layers[{b}]: {r.size} layers, a DSSAT run has at most {SW_LAYERS}")
    return rows


_SW_CODE = re.compile(r"^SW([0-9]+)D$")
#: a four-digit-year date ``YYYYDDD``: ``YYYY * 1000 + DDD`` (``DDD`` = 1 .. 366, ``YYYY`` = 1000 .. 9999);
#: smallest such date (year 1000, day 1)
_YRDOY_MIN = 1000001
#: first number past the four-digit years (year 10000, day 0): ``YRDOY`` must be below it
_YRDOY_END = 10_000_000
#: the YRDOY encoding step of one year (YYYYDDD: year x 1000 + day; an encoding, not a unit change)
_YRDOY_YEAR_STEP = 1000
_DOY_MAX = 366


def _check_yrdoy(yd: np.ndarray) -> None:
    """Refuse ``soil_water`` dates that are not finite whole ``YYYYDDD`` numbers (a NaN, a fraction,
    a two-digit year ``YYDDD``, a year above 9999 or a day of year outside 1 .. 366)."""
    finite = np.isfinite(yd)
    whole = np.where(finite, yd, 0.0)
    doy = np.mod(whole, _YRDOY_YEAR_STEP)
    ok = (
        finite
        & (whole == np.trunc(whole))
        & (whole >= _YRDOY_MIN)
        & (whole < _YRDOY_END)
        & (doy >= 1)
        & (doy <= _DOY_MAX)
    )
    if not ok.all():
        bad = yd[~ok][:5].tolist()
        raise ValueError(f"soil_water: YRDOY must be a four-digit-year date YYYYDDD, got {bad}")


def _water_cells(
    tt: Any,
    soil_water: pd.DataFrame | None,
    trn: tuple[int, ...],
    days: np.ndarray,
    bottoms: list[np.ndarray],
    fws: Sequence[int | str | date | None],
    notes: list[str],
    codes: Collection[str] | None = None,
) -> dict[str, dict[tuple[int, int], list[float]]]:
    """``{code: {(day index, treatment position): [values]}}`` of the soil-water-balance targets:
    FILET ``SW#D`` and ``ETAA`` and the depth observations placed in their layers; no-data values
    (soil water <= 0, a treatment whose every value of a code is 0) and values off the simulated
    days or below the profile are left out, with notes; codes outside ``codes`` (when given) are
    neither collected nor noted."""
    cells: dict[str, dict[tuple[int, int], list[float]]] = {}
    outside: dict[str, list[str]] = {}
    zeros: dict[str, int] = {}
    all_zero: dict[str, list[int]] = {}

    def put(code: str, b: int, yd: int, v: float) -> None:
        if codes is not None and code not in codes:
            return
        hit = np.nonzero(days[b] == yd)[0]
        if not hit.size:
            outside.setdefault(code, []).append(f"{trn[b]}:{yd}")
            return
        cells.setdefault(code, {}).setdefault((int(hit[0]), b), []).append(v)

    if tt is not None:
        for c in tt.codes:
            sw = _SW_CODE.match(c)
            if (sw is None and c != "ETAA") or (codes is not None and c not in codes):
                continue
            k = int(sw.group(1)) if sw is not None else 0
            for b, t in enumerate(trn):
                s = tt.series(t, c)
                if s.empty:
                    continue
                if fws[b] is None or tt.first_weather is not None:
                    yds = s["YRDOY"].to_numpy(dtype=np.int64)
                else:  # the file was read without FirstWeatherDate: redo Y4K_DOY with it
                    yds = np.array(
                        [_yrdoy_int(y4k_date(int(d), fws[b])) for d in s["DATECODE"]], dtype=np.int64
                    )
                vals = s["value"].to_numpy(dtype=float)
                if np.all(vals == 0.0):
                    all_zero.setdefault(c, []).append(t)
                    continue
                if sw is not None and k > bottoms[b].size:
                    notes.append(
                        f"{c}: TRNO {t} has {bottoms[b].size} model layers, no layer {k}; "
                        f"{vals.size} observations left out"
                    )
                    continue
                for yd, v in zip(yds.tolist(), vals.tolist(), strict=True):
                    if sw is not None and not v > 0.0:
                        zeros[c] = zeros.get(c, 0) + 1
                        continue
                    put(c, b, int(yd), float(v))
    if soil_water is not None:
        need = {"TRNO", "YRDOY", "DEPTH", "SW"}
        if not need <= set(soil_water.columns):
            raise ValueError(f"soil_water: columns {sorted(need)} needed, got {list(soil_water.columns)}")
        _check_yrdoy(soil_water["YRDOY"].to_numpy(dtype=float))
        # notes about the depth table only when some layer target is requested
        want_sw = codes is None or any(_SW_CODE.match(c) for c in codes)
        pos = {t: b for b, t in enumerate(trn)}
        below: list[str] = []
        for t, yd, dep, v in zip(
            soil_water["TRNO"].to_numpy(dtype=np.int64).tolist(),
            soil_water["YRDOY"].to_numpy(dtype=np.int64).tolist(),
            soil_water["DEPTH"].to_numpy(dtype=float).tolist(),
            soil_water["SW"].to_numpy(dtype=float).tolist(),
            strict=True,
        ):
            b = pos.get(int(t))
            if b is None or not want_sw:
                continue
            bt = bottoms[b]
            if not 0.0 < dep <= bt[-1]:
                below.append(f"{t}:{yd}:{dep:g}")
                continue
            k = int(np.searchsorted(bt, dep, side="left")) + 1
            if codes is not None and f"SW{k}D" not in codes:
                continue
            if not v > 0.0:  # NaN (missing) too
                zeros["soil_water"] = zeros.get("soil_water", 0) + 1
                continue
            put(f"SW{k}D", b, int(yd), float(v))
        if below:
            notes.append(
                f"soil_water: {len(below)} observations outside the profile (depth <= 0 or below the "
                f"deepest layer) left out (TRNO:YRDOY:DEPTH {below[:5]})"
            )
    by_trnos: dict[tuple[int, ...], list[str]] = {}
    for c, ts in all_zero.items():
        by_trnos.setdefault(tuple(ts), []).append(c)
    for ts, cs in by_trnos.items():
        gone = [c for c in cs if c not in cells]
        notes.append(
            f"{', '.join(cs)}: every value of TRNO {list(ts)} is 0 (a placeholder, no measurement); "
            "treated as no data" + (f"; no target for {', '.join(gone)}" if gone else "")
        )
    for c, n in zeros.items():
        notes.append(f"{c}: {n} soil-water values missing or 0 or less (no measurement) treated as no data")
    for c, o in outside.items():
        notes.append(
            f"{c}: {len(o)} observation dates outside the simulated days dropped (TRNO:YRDOY {o[:5]}...)"
        )
    return cells
