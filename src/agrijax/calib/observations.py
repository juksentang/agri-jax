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
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from agrijax.calib.objective import Group, Simulator, Target
from agrijax.io.dssat.observed import ObservedData, reada_date, y4k_date

__all__ = [
    "DSSAT_TARGETS",
    "UNSUPPORTED_REASONS",
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

_SOIL_WATER = "soil water: the free-run CERES model does not output it (the day assembly replays the soil)"
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
    if (code.startswith("SW") and code.endswith("D")) or code in {"SWTD", "SWXD", "PESW"}:
        return _SOIL_WATER
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
        m = DSSAT_TARGETS.get(c)
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
    return ObservationTargets(tuple(targets), observed, trn, unsupported, tuple(notes), n_obs)
