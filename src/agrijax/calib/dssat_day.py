"""Batched evaluation of the free-run DSSAT-CSM v4.8.6.0 day for cultivar calibration.

:class:`DaySimulator` runs the day (:mod:`agrijax.models.day_dssat486`, the free-run configuration of
:mod:`agrijax.sites.dssat_free_run`) on **samples** ``(theta, treatment)``: ``theta`` holds the values
of the simulator's parameter space, by default the six ``MZCER048.CUL`` coefficients
(:data:`CUL_ORDER`, :func:`default_space`), the treatment is an index into the treatments the
simulator was built for. One call evaluates any number of samples of any treatments: the samples
of one batch group (treatments with one soil layer count and one ``MESEV``; their forcings padded to
one length) go through one compiled program (a ``vmap`` over the samples inside one ``scan`` over
the days), sharded over the host devices (``jax.devices()``; on a CPU node set
``XLA_FLAGS=--xla_force_host_platform_device_count=<cores>`` before JAX is imported to use every core).

A call returns the values of the treatment's **entries** only (:class:`Entry`: a stage date, a value
at the end of the crop or a value on a day; the crop outputs of ``PlantGro.OUT`` and the daily
evapotranspiration and layer soil water, :data:`OUT_NAMES`), gathered on the device, so a call moves
``[samples, entries]`` numbers back to the host. Programs are compiled once per (group, padded
sample count) and kept; compile time, call time, program calls and simulated seasons are counted
(:meth:`DaySimulator.stats`).

**Parameter spaces and site blocks.** ``space`` may be any :class:`~agrijax.calib.space.ParamSpace`
whose paths are cultivar coefficients (``"cultivar.<name>"``, e.g. :data:`agrijax.calib.ceres.CERES_SPECS`)
or soil values by ``.SOL`` layer (the quantities of :func:`agrijax.calib.soil.soil_space` built with
``prefix=None``), or a :class:`~agrijax.calib.space.ParamBlocks` of a shared space and one space per
site, with ``sites`` naming each treatment's site. ``theta`` has the layout of the blocks; inside the
program every sample reads its treatment's columns (the shared block and its site's block) through
an index table, so the treatments of several sites share one program per batch group. A soil value
is written with :func:`agrijax.calib.day_soil.write_soil` (every copy the day holds, the initial
water limited again; the treatment's :class:`~agrijax.calib.day_soil.SoilLayerMap` in ``soil``); a
treatment that has no coordinate of a quantity another treatment of its group calibrates keeps its
own value of it, and its morning state, unchanged. With the default space nothing of this enters
the program: it is the program the cultivar calibration ran before.

**Season readout** (``readout="season"``, the default). Each sample is read on three day indices:

* ``t_end``, the end of the crop, from the crop's ``MDATE`` (set at physiological maturity,
  MZ_PHENOL.for:897, the day ISTAGE 10 starts; at a cold or drought failure, MZ_GROSUB.for:1661 /
  :1706; at a germination / emergence failure of MZ_PHENOL.for; a failed crop in ISTAGE 6 can reach
  maturity later, which sets it again). Harvest at maturity (``HARVS = M``): the first day it is set,
  the day ``dscsm048`` harvests and ends the season (AUTHAR.for ``YREND = MDATE``, called after the
  plant in LAND.for's daily integration); any other harvest option: the day ``MDATE`` holds on
  ``t_last`` (the last setting up to the harvest day). Either way the quantity ``dscsm048`` writes to
  ``Summary.OUT`` as ``MDAT`` (MZ_OPHARV.for:468); never set by ``t_last``: none;
* ``t_last``, the run's own last real day (:meth:`~agrijax.sites.dssat_free_run.FreeRunInputs.last_real_day`):
  harvest at maturity (``HARVS = M``): the reference season plus its real weather extension, at
  most ``pad_days``; any other harvest option: the harvest day (the run's last season day);
* ``t_read = min(t_end, t_last)`` (``t_last`` without ``t_end``).

``E_FINAL`` reads the output on ``t_read``; a stage date reached after ``t_last`` (or never) is
right-censored at ``t_last + 1``, which depends on the treatment only, not on the batch group;
``E_STATUS`` says how the season ended (:data:`END_MATURED`; :data:`END_FAILED` when ``MDATE`` was
first set by a failure, not at ISTAGE 10;
:data:`END_HARVESTED`, :data:`END_CENSORED`). The days after ``t_last`` (the padding that brings
the runs of a group to one length, :data:`PAD_DAYS`) are never read. ``readout="padded"`` keeps the
earlier readout bit for bit (the value at the first ISTAGE 10 or on the group's last, possibly
padded, day; dates censored at the group's length), to check records made before the season readout.

**Gradient mode.** Every program is bound to one gradient mode (:data:`GRADIENT_MODE`,
:func:`agrijax.core.grad.bind_gradient_mode`); :meth:`DaySimulator.jax_loss` gives the pure JAX
per-treatment loss for the gradient-trust report and gradient steps, in the same mode, over every
batch group (one traced function; the losses come back in treatment order).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CROP_END",
    "CROP_OUT_NAMES",
    "CUL_ORDER",
    "END_CENSORED",
    "END_FAILED",
    "END_HARVESTED",
    "END_MATURED",
    "END_NAMES",
    "E_DATE",
    "E_DAY",
    "E_FINAL",
    "E_PAD",
    "E_STATUS",
    "GRADIENT_MODE",
    "OUT_NAMES",
    "PAD_DAYS",
    "READOUTS",
    "SW_LAYERS",
    "WATER_OUT_NAMES",
    "DaySimulator",
    "Entry",
    "default_space",
    "entry_values",
]

#: the calibrated coefficients in ``MZCER048.CUL`` column order: the ``theta`` columns of the default
#: space (:func:`default_space`)
CUL_ORDER: tuple[str, ...] = ("P1", "P2", "P5", "G2", "G3", "PHINT")
#: path prefix of a cultivar coefficient in a parameter space (:data:`agrijax.calib.ceres.CERES_SPECS`)
_CULTIVAR = "cultivar."
#: gradient mode of every program (``ste``: identity derivative through Fortran's truncations and roundings,
#: e.g. RLV and TURFAC to 1e-3 and SW to 1e-6, each a registered ``GradientConvention``; the CERES-Maize day
#: has no ``event_ste`` event: its stage changes are plain selections, so for every coefficient, phenology
#: included, the ste and exact derivatives differ only through those quantisers; forward values do not
#: depend on the mode)
GRADIENT_MODE = "ste"
#: days simulated past the longest season of a batch group: a run's real weather extension first
#: (:attr:`~agrijax.sites.dssat_free_run.FreeRunInputs.extension`, at most this many days are read),
#: then the last day repeated with rain and irrigation 0 only to bring the runs of the group to one
#: length; the season readout never reads a padded day
PAD_DAYS = 60
#: daily crop outputs (``plantgro_outputs`` names, the quantities and units of ``PlantGro.OUT``)
CROP_OUT_NAMES: tuple[str, ...] = ("lai", "cwad", "gwad", "g_ad", "lsd", "lwad", "swad", "rwad", "pwad")
#: model layers whose soil water an entry can name: DSSAT's largest layer count (``ModuleDefs.for``
#: ``NL = 20``, :data:`agrijax.io.dssat.native_soil.NL`; a unit test checks both)
SW_LAYERS = 20
#: daily soil-water-balance outputs of the day state (:func:`agrijax.models.day_dssat486.day_outputs`):
#: ``et`` = ``EVAP + EP`` [mm d-1], the day's actual evapotranspiration as ``SPAM.for`` INTEGR sums it
#: (``ET = EVAP + EP``, SPAM.for:464; ``ETAA`` of ``ET.OUT`` on daily output), and ``sw<k>`` = the
#: end-of-day volumetric water content of model layer ``k`` [cm3 cm-3] (``SW(k)``, ``SW<k>D`` of
#: ``SoilWat.OUT``), ``k = 1 ..`` :data:`SW_LAYERS`; an entry can only name a layer the run has
WATER_OUT_NAMES: tuple[str, ...] = ("et", *(f"sw{k}" for k in range(1, SW_LAYERS + 1)))
#: daily model outputs an entry can read (``Entry.out`` indexes this): the crop outputs, then the
#: soil-water-balance outputs. A program carries the crop outputs and only the water outputs its
#: entries read, so a calibration on crop observations runs the program it ran before.
OUT_NAMES: tuple[str, ...] = CROP_OUT_NAMES + WATER_OUT_NAMES
#: entry types (``E_STATUS``: the end status of the season, ``out`` and ``t`` ignored; season readout only)
E_PAD, E_DATE, E_FINAL, E_DAY, E_STATUS = 0, 1, 2, 3, 4
#: stage codes of the dates an entry can read (first day of the code in ``istage``): emergence,
#: silking (ISTAGE 4, MZ_PHENOL.for:783) and the day after physiological maturity (ISTAGE 10,
#: MZ_PHENOL.for:899), as :data:`agrijax.calib.ceres.ISTAGE_SILKING_OUT` / ``ISTAGE_MATURITY_OUT``
ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY = 1, 4, 10
#: the date code of the end of the crop (an ``E_DATE`` ``out``; not an ISTAGE value, those are 1-10):
#: the day ``MDATE`` is first set (harvest at maturity) or the day it holds on the run's last real day
#: (other harvest options; set at maturity or by a crop / seed failure, set again when a failed crop
#: matures; MZ_PHENOL.for:353 ``MDATE = -99`` before), the quantity ``dscsm048`` reports as ``MDAT``;
#: season readout only
CROP_END = 0
#: end status codes (``E_STATUS``): ISTAGE 10 reached by ``t_last`` (MZ_PHENOL.for:897-899,
#: ``CropStatus = 1``); ``MDATE`` set without ISTAGE 10 by ``t_last`` (cold or drought failure,
#: MZ_GROSUB.for:1661 / :1706, or a germination / emergence failure); not ended by the harvest day
#: (a harvest option other than at maturity); not ended by the last real day (harvest at maturity)
END_MATURED, END_FAILED, END_HARVESTED, END_CENSORED = 0, 1, 2, 3
#: the names of the end status codes (index = code)
END_NAMES: tuple[str, ...] = ("matured", "failed", "harvested", "censored")
#: readout modes of :class:`DaySimulator` (module docstring)
READOUTS: tuple[str, ...] = ("season", "padded")
#: largest sample count of one program call (memory), rounded down to the ladder of the padding
CHUNK_TARGET = 12288


@dataclass(frozen=True)
class Entry:
    """One simulated quantity of a treatment.

    ``kind``: ``E_DATE`` (``out`` = a stage code or :data:`CROP_END`; the value is the day index of
    its first day, ``t_last + 1`` if not reached by the run's last real day ``t_last``),
    ``E_FINAL`` (``out`` = an :data:`OUT_NAMES` index; the value on ``min(t_end, t_last)``: the end
    of the crop, or the harvest / last real day when the crop has not ended by then), ``E_DAY`` (the
    output on day index ``t``, at most ``t_last``) or ``E_STATUS`` (the end status code,
    :data:`END_NAMES`). The day indices: module docstring (season readout); under
    ``readout="padded"`` dates are censored at the group's length and ``E_FINAL`` reads the first
    ISTAGE 10 or the group's last day. A soil-water output ``sw<k>`` needs a run with at least ``k``
    model layers."""

    kind: int
    out: int
    t: int = 0


def _water_outputs(nl: int, entries: Sequence[tuple[int, Sequence[Entry]]]) -> tuple[str, ...]:
    """The soil-water-balance outputs (:data:`WATER_OUT_NAMES` order) the entries of one batch group
    read; an entry naming a soil layer the group's runs do not have (``nl`` layers) or an output
    index outside :data:`OUT_NAMES` is refused."""
    n_crop = len(CROP_OUT_NAMES)
    used: set[int] = set()
    for i, ents in entries:
        for e in ents:
            if e.kind not in (E_FINAL, E_DAY):
                continue
            if not 0 <= e.out < len(OUT_NAMES):
                raise ValueError(f"run {i}: an entry reads output {e.out}, OUT_NAMES has {len(OUT_NAMES)}")
            if e.out < n_crop:
                continue
            name = OUT_NAMES[e.out]
            if name.startswith("sw") and int(name[2:]) > nl:
                raise ValueError(f"run {i}: an entry reads {name}, but the run has {nl} soil layers")
            used.add(e.out)
    return tuple(OUT_NAMES[o] for o in sorted(used))


def _row(e: Entry, water: tuple[str, ...]) -> int:
    """The output row of entry ``e`` in its group's program (``E_FINAL`` / ``E_DAY``): a crop output
    keeps its :data:`OUT_NAMES` index, a water output follows the crop outputs in ``water`` order;
    any other entry keeps ``out`` (a stage code)."""
    if e.kind in (E_FINAL, E_DAY) and len(CROP_OUT_NAMES) <= e.out < len(OUT_NAMES):
        return len(CROP_OUT_NAMES) + water.index(OUT_NAMES[e.out])
    return e.out


def _first_idx(hit: Any) -> Any:
    import jax.numpy as jnp

    t = hit.shape[0]
    return jnp.where(jnp.any(hit, axis=0), jnp.argmax(hit, axis=0), t)


def _gather_one(y: Any, ist: Any, etype: Any, eout: Any, et: Any) -> Any:
    """One sample, ``readout="padded"``: entry values ``[E]`` from its daily outputs ``y [O, T]`` and
    stage codes ``ist [T]`` (the readout before the season readout, kept bit for bit)."""
    import jax.numpy as jnp

    t_n = ist.shape[0]
    codes = jnp.asarray([ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY])
    firsts = jnp.stack([_first_idx(ist == c) for c in (ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY)])
    date = jnp.sum(jnp.where(eout[:, None] == codes[None, :], firsts[None, :], 0), axis=1).astype(y.dtype)
    t_mat = jnp.minimum(firsts[2], t_n - 1)
    t_eff = jnp.where(etype == E_FINAL, t_mat, jnp.clip(et, 0, t_n - 1))
    o = jnp.clip(eout, 0, y.shape[0] - 1)
    val = y[o, t_eff]
    return jnp.where(etype == E_DATE, date, jnp.where(etype == E_PAD, 0.0, val))


def _gather_season(
    y: Any, ist: Any, mdate: Any, last: Any, open_: Any, etype: Any, eout: Any, et: Any
) -> Any:
    """One sample, season readout (module docstring): entry values ``[E]`` from its daily outputs
    ``y [O, T]``, stage codes ``ist [T]``, the crop's ``MDATE`` ``mdate [T]``, the run's last real day
    ``last`` and its status when the crop has not ended by then ``open_`` (``END_CENSORED``: harvest
    at maturity, the end of the crop is the first ``MDATE`` setting; ``END_HARVESTED``: another
    harvest option, the last setting up to ``last``). Only day indices are selected: the values (and
    derivatives) are those of ``y`` on a day ``<= last``."""
    import jax.numpy as jnp

    from agrijax.processes.crop.ceres_maize.constants import MDATE_NONE

    t_n = ist.shape[0]
    days = jnp.arange(t_n)
    prev = jnp.concatenate([jnp.full((1,), MDATE_NONE, mdate.dtype), mdate[:-1]])
    set_on = (mdate != prev) & (mdate != MDATE_NONE) & (days <= last)  # MDATE set that day
    f_set = _first_idx(set_on)  # the first setting: maturity, or a failure
    l_set = jnp.where(jnp.any(set_on), jnp.max(jnp.where(set_on, days, -1)), t_n)  # MDATE on `last`
    t_end = jnp.where(open_ == END_CENSORED, f_set, l_set)  # harvest at maturity: the first setting
    f_mat = _first_idx(ist == ISTAGE_MATURITY)
    cens = last + 1
    cap = lambda f: jnp.where(f <= last, f, cens)  # noqa: E731
    codes = jnp.asarray([CROP_END, ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY])
    firsts = jnp.stack(
        [
            cap(t_end),
            cap(_first_idx(ist == ISTAGE_EMERGENCE)),
            cap(_first_idx(ist == ISTAGE_SILKING)),
            cap(f_mat),
        ]
    )
    date = jnp.sum(jnp.where(eout[:, None] == codes[None, :], firsts[None, :], 0), axis=1).astype(y.dtype)
    ended = t_end <= last
    status = jnp.where(ended, jnp.where(f_mat == f_set, END_MATURED, END_FAILED), open_).astype(y.dtype)
    t_read = jnp.minimum(t_end, last)
    t_eff = jnp.where(etype == E_FINAL, t_read, jnp.clip(et, 0, last))
    o = jnp.clip(eout, 0, y.shape[0] - 1)
    val = y[o, t_eff]
    val = jnp.where(etype == E_STATUS, status, val)
    return jnp.where(etype == E_DATE, date, jnp.where(etype == E_PAD, 0.0, val))


def entry_values(
    outputs: dict[str, np.ndarray],
    entries: Sequence[Entry],
    *,
    last_day: int | None = None,
    harvest: str = "M",
) -> np.ndarray:
    """Host version of the season readout: ``outputs[name] [T]`` of one run (and ``istage``) -> ``[E]``.

    ``last_day``: the run's last real day ``t_last`` (default ``T - 1``: the last day of unpadded
    outputs); ``harvest``: the run's harvest option (the end status of a crop not ended by
    ``t_last``: ``END_CENSORED`` for ``"M"``, else ``END_HARVESTED``; the end of the crop: the first
    ``MDATE`` setting for ``"M"``, else the last one up to ``t_last``). The end of the crop comes from
    ``outputs["mdate"]`` (``MDATE``, -99 before; :func:`agrijax.sites.dssat_free_run.season_outputs`);
    without it, it falls back to the first ISTAGE 10, and a crop failure is then not seen (its
    ``E_FINAL`` reads ``t_last``). A soil-water-balance output (:data:`WATER_OUT_NAMES`) absent from
    ``outputs`` is derived from the day outputs (:func:`agrijax.models.day_dssat486.day_outputs`, the
    keys of :func:`~agrijax.sites.dssat_free_run.season_outputs`) as the program derives it:
    ``et = evap + ep``, ``sw<k> = soil_sw[:, k - 1]``."""
    from agrijax.processes.crop.ceres_maize.constants import MDATE_NONE

    ist = np.asarray(outputs["istage"])
    ist = ist.reshape(ist.shape[0], -1)[:, 0] if ist.ndim > 1 else ist
    t_n = ist.shape[0]
    last = t_n - 1 if last_day is None else int(last_day)
    if not 0 <= last < t_n:
        raise ValueError(f"last_day {last} outside the {t_n} days of the outputs")

    def first(hit: np.ndarray) -> int:
        idx = np.nonzero(hit)[0]
        return int(idx[0]) if idx.size else t_n

    f_mat = first(ist == ISTAGE_MATURITY)
    at_maturity = str(harvest).strip().upper()[:1] == "M"
    if "mdate" in outputs:
        md = np.ravel(np.asarray(outputs["mdate"]))[:t_n]
        prev = np.concatenate([[MDATE_NONE], md[:-1]])
        set_on = (md != prev) & (md != MDATE_NONE) & (np.arange(t_n) <= last)
        hits = np.nonzero(set_on)[0]
        f_set, l_set = (int(hits[0]), int(hits[-1])) if hits.size else (t_n, t_n)
        t_end = f_set if at_maturity else l_set
    else:
        f_set = t_end = f_mat
    cap = lambda f: f if f <= last else last + 1  # noqa: E731
    t_read = min(t_end, last)
    if t_end <= last:
        status = END_MATURED if f_mat == f_set else END_FAILED
    else:
        status = END_CENSORED if at_maturity else END_HARVESTED

    def col0(a: Any) -> np.ndarray:
        a = np.asarray(a)
        return a.reshape(a.shape[0], -1)[:, 0] if a.ndim > 1 else a

    def series(k: int) -> np.ndarray:
        name = OUT_NAMES[k]
        if name in outputs:
            return col0(outputs[name])
        if name == "et":
            return col0(outputs["evap"]) + col0(outputs["ep"])
        if name in WATER_OUT_NAMES:
            sw = np.asarray(outputs["soil_sw"])
            return sw.reshape(sw.shape[0], -1)[:, int(name[2:]) - 1]
        raise KeyError(name)

    out = np.zeros(len(entries))
    for j, e in enumerate(entries):
        if e.kind == E_DATE:
            out[j] = cap(t_end) if e.out == CROP_END else cap(first(ist == e.out))
        elif e.kind == E_FINAL:
            out[j] = float(series(e.out)[t_read])
        elif e.kind == E_DAY:
            if not 0 <= e.t <= last:
                raise ValueError(f"E_DAY on day {e.t}: outside the run's real days 0-{last}")
            out[j] = float(series(e.out)[e.t])
        elif e.kind == E_STATUS:
            out[j] = float(status)
    return out


def default_space() -> Any:
    """The default parameter space of :class:`DaySimulator`: the six ``MZCER048.CUL`` coefficients of
    :data:`CUL_ORDER` with their ``MINIMA`` / ``MAXIMA`` bounds (:func:`agrijax.calib.ceres.ceres_space`)."""
    from agrijax.calib.ceres import ceres_space

    return ceres_space(CUL_ORDER)


@dataclass(frozen=True, eq=False)
class _Plan:
    """How the samples of one batch group read ``theta`` (host-side tables, one row per treatment
    of the group).

    ``cul``: the cultivar fields written; ``cul_col[b, i]`` the ``theta`` column of field ``i`` for
    treatment ``b`` (-1: keep its own value); ``cul_static``: the columns when every treatment reads
    the same ones (the program then reads them as the simulator always has). ``soil``: the soil
    quantities written (empty: no soil is written); per quantity ``col`` (``[B, K]`` per ``.SOL``
    layer or ``[B]``, -1: the site value), ``site`` (the site values, same shapes), ``keep`` ``[B]``
    (the treatment has no coordinate of it); the layer gather ``index`` / ``coef`` ``[B, nl, k]`` and
    the initial water before the limits ``sw_input`` / ``sw_missing`` ``[B, nl]``."""

    cul: tuple[str, ...]
    cul_col: np.ndarray
    cul_static: tuple[int, ...] | None
    soil: tuple[str, ...] = ()
    col: dict[str, np.ndarray] | None = None
    site: dict[str, np.ndarray] | None = None
    keep: dict[str, np.ndarray] | None = None
    index: np.ndarray | None = None
    coef: np.ndarray | None = None
    sw_input: np.ndarray | None = None
    sw_missing: np.ndarray | None = None


def _coordinates(space: Any) -> list[tuple[str, str, int | None]]:
    """``(kind, name, index)`` of every spec of ``space``: ``("cultivar", field, None)`` or
    ``("soil", quantity, .SOL layer or None)``; any other path is refused."""
    from agrijax.calib.soil import LAYER_QUANTITIES, SCALAR_QUANTITIES

    out: list[tuple[str, str, int | None]] = []
    for sp in space.specs:
        if sp.path.startswith(_CULTIVAR) and sp.index is None and "." not in sp.path[len(_CULTIVAR) :]:
            out.append(("cultivar", sp.path[len(_CULTIVAR) :], None))
        elif sp.path in SCALAR_QUANTITIES and sp.index is None:
            out.append(("soil", sp.path, None))
        elif sp.path in LAYER_QUANTITIES and sp.index is not None:
            out.append(("soil", sp.path, int(sp.index)))
        else:
            raise ValueError(
                f"DaySimulator: cannot write {sp.name} (path {sp.path!r}, index {sp.index}); it writes "
                "cultivar coefficients ('cultivar.<name>') and soil values by .SOL layer "
                "(agrijax.calib.soil.soil_space with prefix=None)"
            )
    return out


def _identity_map(run: Any, params: Any) -> Any:
    """A :class:`~agrijax.calib.day_soil.SoilLayerMap` of ``run``'s own model layers (one file layer
    per model layer, the run's current values): the table row of a treatment that writes no soil
    in a group that does (every quantity kept, so its values are never used; they only keep the
    unused branch finite)."""
    from agrijax.calib.day_soil import SoilLayerMap

    nl = int(run.nl)
    soil = params["soil"].soil
    cs = params["crop"].soil
    a = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    sol = {
        "cn": a(soil.cn),
        "swcon": a(soil.swcon),
        "u": a(params["evap"].u),
        "slpf": a(cs.slpf).reshape(()),
        "ll": a(soil.ll),
        "dul": a(soil.dul),
        "sat": a(soil.sat),
        "srgf": a(cs.shf),
        "swcn": np.where(a(soil.swcn) > 0, a(soil.swcn), np.nan),
    }
    return SoilLayerMap(
        profile=f"{run.key} (own layers)",
        weight=np.eye(nl),
        index=np.arange(nl, dtype=np.int32)[:, None],
        coef=np.ones((nl, 1)),
        sol=sol,
        sw_input=np.asarray(run.sw0, dtype=np.float64),
        soildyn=bool(run.soildyn),
    )


def _plan(
    runs: Sequence[Any],
    idx: Sequence[int],
    coords: Sequence[list[tuple[str, str, int | None, int]]],
    maps: Sequence[Any],
    n_days: int,
) -> _Plan:
    """The :class:`_Plan` of the batch group of treatments ``idx`` (``coords[i]``: treatment ``i``'s
    coordinates with their ``theta`` columns; ``maps[i]`` its soil layer map or ``None``)."""
    from agrijax.calib.day_soil import _check, _check_static_forcing
    from agrijax.calib.soil import LAYER_QUANTITIES, SOIL_QUANTITIES

    cul: list[str] = []
    for i in idx:
        for kind, name, _, _ in coords[i]:
            if kind == "cultivar" and name not in cul:
                cul.append(name)
    cul_col = np.full((len(idx), len(cul)), -1, np.int64)
    for b, i in enumerate(idx):
        c = runs[i].params_crop.cultivar
        for kind, name, _, c_theta in coords[i]:
            if kind != "cultivar":
                continue
            if not hasattr(c, name):
                raise ValueError(f"DaySimulator: {runs[i].key}: the cultivar has no coefficient {name!r}")
            cul_col[b, cul.index(name)] = c_theta
    uniform = bool(cul_col.size == 0 or (np.all(cul_col >= 0) and np.all(cul_col == cul_col[:1])))
    static = tuple(int(c) for c in cul_col[0]) if uniform and cul_col.shape[0] else None
    used = {name for i in idx for kind, name, _, _ in coords[i] if kind == "soil"}
    soil = tuple(q for q in SOIL_QUANTITIES if q in used)
    if not soil:
        return _Plan(tuple(cul), cul_col, static)
    lms = []
    for i in idx:
        mine = [(name, h) for kind, name, h, _ in coords[i] if kind == "soil"]
        m = maps[i]
        if mine:
            if m is None:
                raise ValueError(
                    f"DaySimulator: {runs[i].key} has soil coordinates but no SoilLayerMap "
                    "(soil=[...], agrijax.calib.day_soil.soil_layer_map)"
                )
            if m.nl != int(runs[i].nl):
                raise ValueError(
                    f"DaySimulator: {runs[i].key}: map has {m.nl} model layers, the run {runs[i].nl}"
                )
            if bool(runs[i].soildyn) and not m.soildyn:
                raise ValueError(
                    f"DaySimulator: {runs[i].key}: the run's soil changes (SOILDYN), its map says static"
                )
            # shapes, SOILDYN and missing SSKS as with_soil checks them, on the site values
            _check({q: np.asarray(m.sol[q]) for q, _ in mine}, m)
            _check_static_forcing(runs[i].forcing(n_days), m)
            for q, h in mine:
                if h is not None and not 0 <= h < m.n_sol:
                    raise ValueError(
                        f"DaySimulator: {runs[i].key}: {q} of .SOL layer {h}, the profile has {m.n_sol}"
                    )
            lms.append(m)
        else:
            lms.append(_identity_map(runs[i], runs[i].params()))
    big = max(m.n_sol for m in lms)
    kk = max(m.index.shape[1] for m in lms)
    nl = lms[0].nl
    bn = len(idx)
    col: dict[str, np.ndarray] = {}
    site: dict[str, np.ndarray] = {}
    keep: dict[str, np.ndarray] = {}
    for q in soil:
        layered = q in LAYER_QUANTITIES
        col[q] = np.full((bn, big) if layered else (bn,), -1, np.int64)
        site[q] = np.zeros((bn, big) if layered else (bn,))
        keep[q] = np.ones(bn, bool)
        for b, i in enumerate(idx):
            v = np.nan_to_num(np.asarray(lms[b].sol[q], dtype=np.float64), nan=0.0)
            if layered:
                site[q][b, : v.size] = v.reshape(-1)
            else:
                site[q][b] = float(v)
            for kind, name, h, c in coords[i]:
                if kind == "soil" and name == q:
                    keep[q][b] = False
                    if layered:
                        col[q][b, h] = c
                    else:
                        col[q][b] = c
    index = np.zeros((bn, nl, kk), np.int32)
    coef = np.zeros((bn, nl, kk))
    sw_in = np.zeros((bn, nl))
    sw_miss = np.zeros((bn, nl), bool)
    for b, m in enumerate(lms):
        k = m.index.shape[1]
        index[b, :, :k] = m.index
        coef[b, :, :k] = m.coef
        miss = np.isnan(np.asarray(m.sw_input, dtype=float))
        sw_in[b] = np.where(miss, 0.0, m.sw_input)
        sw_miss[b] = miss
    return _Plan(tuple(cul), cul_col, static, soil, col, site, keep, index, coef, sw_in, sw_miss)


def _last_real_day(run: Any, pad_days: int) -> int:
    """``t_last`` of ``run`` (:meth:`FreeRunInputs.last_real_day`; a run object without it: harvest at
    maturity, no extension)."""
    if hasattr(run, "last_real_day"):
        return int(run.last_real_day(pad_days))
    return int(run.n_days) - 1


def _open_status(run: Any) -> int:
    """The end status of ``run`` when its crop has not ended by ``t_last`` (harvest at maturity:
    ``END_CENSORED``; another option: ``END_HARVESTED``); it also picks the ``MDATE`` setting read."""
    return END_CENSORED if str(getattr(run, "harvest", "M")).strip().upper()[:1] == "M" else END_HARVESTED


class DaySimulator:
    """The free-run day on samples ``(theta, treatment)`` (module docstring).

    ``runs`` are :class:`agrijax.sites.dssat_free_run.FreeRunInputs`; ``entries[b]`` the entries of
    run ``b``. Calling the simulator with ``theta [S, P]`` (``P`` = the space's size, 6 by default)
    and treatment indices ``tid [S]`` returns ``[S, E]`` (``E`` = the most entries of any treatment;
    unused entries 0).

    ``space``: a :class:`~agrijax.calib.space.ParamSpace` or :class:`~agrijax.calib.space.ParamBlocks`
    (default :func:`default_space`, the six :data:`CUL_ORDER` coefficients); ``sites``: the site of
    every run, needed when the blocks have site blocks (each a key of ``per_site``); ``soil``: the
    :class:`~agrijax.calib.day_soil.SoilLayerMap` of every run (``None`` for a run without soil
    coordinates), needed when the space writes soil values (module docstring).

    ``devices``: the devices the batch is sharded over (default: ``jax.devices()``, as before).
    ``dtype``: the floating dtype the inputs are stored in on the devices (default: as built,
    float64); ``np.float32`` stores them rounded once, and the calls must then run with
    ``jax_enable_x64`` off (:mod:`agrijax.facade_execution` does that).

    ``readout``: ``"season"`` (default; the season readout of the module docstring, with run ``b``'s
    last real day in :attr:`last_days`) or ``"padded"`` (the earlier readout, bit for bit; no
    ``E_STATUS`` or ``CROP_END`` entries). Under ``"season"`` an ``E_DAY`` entry after the run's last
    real day raises ``ValueError``, a soil-water entry included."""

    def __init__(
        self,
        runs: Sequence[Any],
        entries: Sequence[Sequence[Entry]],
        *,
        space: Any = None,
        sites: Sequence[str] | None = None,
        soil: Sequence[Any] | None = None,
        pad_days: int = PAD_DAYS,
        devices: Sequence[Any] | None = None,
        dtype: Any = None,
        readout: str = "season",
    ) -> None:
        if len(runs) != len(entries):
            raise ValueError(f"{len(runs)} runs, {len(entries)} entry lists")
        if readout not in READOUTS:
            raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
        self.readout = readout
        #: each run's last real day ``t_last`` (season readout; module docstring)
        self.last_days = [_last_real_day(r, int(pad_days)) for r in runs]
        for i, ents in enumerate(entries):
            for e in ents:
                if readout == "padded" and (e.kind == E_STATUS or (e.kind == E_DATE and e.out == CROP_END)):
                    raise ValueError(
                        f"run {i}: E_STATUS and the CROP_END date need the season readout (readout='season')"
                    )
                if readout == "season" and e.kind == E_DAY and not 0 <= e.t <= self.last_days[i]:
                    raise ValueError(
                        f"run {i}: E_DAY on day {e.t} is outside the run's real days 0-{self.last_days[i]}"
                    )
        self._open = [_open_status(r) for r in runs]
        self._build(
            runs,
            entries,
            space=space,
            sites=sites,
            soil=soil,
            pad_days=pad_days,
            devices=devices,
            dtype=dtype,
        )

    def _build(
        self,
        runs: Sequence[Any],
        entries: Sequence[Sequence[Entry]],
        *,
        space: Any,
        sites: Sequence[str] | None,
        soil: Sequence[Any] | None,
        pad_days: int,
        devices: Sequence[Any] | None,
        dtype: Any,
    ) -> None:
        import jax
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as PS

        from agrijax.calib.space import ParamBlocks
        from agrijax.sites.dssat_free_run import stack_trees

        self.runs = list(runs)
        #: the parameter blocks ``theta`` is laid out by (a plain space is the shared block)
        self.blocks = ParamBlocks.of(default_space() if space is None else space)
        #: the number of ``theta`` columns
        self.n_theta = self.blocks.n
        if self.blocks.per_site:
            if sites is None or len(sites) != len(self.runs):
                raise ValueError("DaySimulator: site blocks need sites=[the site of every run]")
            unknown = sorted({s for s in sites if s not in self.blocks.per_site})
            if unknown:
                raise ValueError(
                    f"DaySimulator: sites {unknown} have no block (blocks: {list(self.blocks.sites)})"
                )
            self.sites: list[str | None] = list(sites)
        else:
            self.sites = [None] * len(self.runs)
        if soil is not None and len(soil) != len(self.runs):
            raise ValueError(f"DaySimulator: {len(soil)} soil layer maps for {len(self.runs)} runs")
        maps = list(soil) if soil is not None else [None] * len(self.runs)
        coords = []
        for site in self.sites:
            cols = self.blocks.columns(site)
            cs = _coordinates(self.blocks.space_of(site))
            coords.append([(k, n, h, int(c)) for (k, n, h), c in zip(cs, cols, strict=True)])
        self.n_entries = max(1, max(len(e) for e in entries))
        self.devs = list(jax.devices() if devices is None else devices)
        self.ndev = len(self.devs)
        mesh = Mesh(np.asarray(self.devs), ("b",))
        self.rep = NamedSharding(mesh, PS())
        self.bsh = NamedSharding(mesh, PS("b"))
        groups: dict[tuple[int, str], list[int]] = {}
        for i, r in enumerate(self.runs):
            groups.setdefault((r.nl, r.mesev), []).append(i)
        self.groups = groups
        #: the soil-water-balance outputs each group's program carries (those its entries read)
        self.water: dict[tuple[int, str], tuple[str, ...]] = {
            g: _water_outputs(g[0], [(i, entries[i]) for i in idx]) for g, idx in groups.items()
        }
        self.where: dict[int, tuple[tuple[int, str], int]] = {}
        self.n_days: dict[tuple[int, str], int] = {}
        self.inputs: dict[tuple[int, str], Any] = {}
        self.tables: dict[tuple[int, str], Any] = {}
        self.plans: dict[tuple[int, str], _Plan] = {}
        for g, idx in groups.items():
            n = max(self.runs[i].n_days for i in idx) + int(pad_days)
            self.n_days[g] = n
            self.plans[g] = _plan(self.runs, idx, coords, maps, n)
            ps = [self.runs[i].params() for i in idx]
            args = (
                stack_trees(ps),
                stack_trees([self.runs[i].forcing(n) for i in idx]),
                stack_trees([self.runs[i].state(p) for i, p in zip(idx, ps, strict=True)]),
            )
            if dtype is not None:
                args = jax.tree.map(
                    lambda a: a.astype(dtype) if np.issubdtype(a.dtype, np.floating) else a, args
                )
            self.inputs[g] = jax.device_put(args, self.rep)
            tab: dict[str, np.ndarray] = {
                k: np.zeros((len(idx), self.n_entries), np.int32) for k in ("type", "out", "t")
            }
            if self.readout == "season":
                tab["last"] = np.asarray([self.last_days[i] for i in idx], np.int32)
                tab["open"] = np.asarray([self._open[i] for i in idx], np.int32)
            for b, i in enumerate(idx):
                self.where[i] = (g, b)
                for j, e in enumerate(entries[i]):
                    tab["type"][b, j], tab["out"][b, j], tab["t"][b, j] = e.kind, _row(e, self.water[g]), e.t
            self.tables[g] = jax.device_put(tab, self.rep)
        self.programs: dict[tuple[Any, ...], Any] = {}
        self.compile_s = 0.0
        self.call_s = 0.0
        self.n_compiled = 0
        self.program_calls = 0
        self.seasons = 0

    # ---------------------------------------------------------------- the traced simulation
    def _sim_fn(
        self, g: tuple[int, str], forcing_hook: Callable[..., Any] | None = None
    ) -> Callable[..., Any]:
        """``sim(inputs, tab, theta [S, P], tid [S]) -> [S, E]`` of group ``g`` (traceable).

        ``forcing_hook``: ``hook(forcing_t, t, tid, *extra) -> forcing_t``, applied to every day's
        forcing of the samples before the day runs; ``sim`` then takes the hook's ``extra``
        arguments after ``tid`` (the weather sensitivity, :mod:`agrijax.facade_weather`, adds its
        perturbation there). Without it the program is the one the calibration runs."""
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from agrijax.calib.day_soil import FORCING_SOIL_VALUES, write_soil
        from agrijax.core.state import get_path
        from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_outputs, day_processes
        from agrijax.processes.crop.ceres_maize import plantgro_outputs

        water = self.water[g]
        plan = self.plans[g]
        day_out = day_outputs(SLOT)
        names_out = (*CROP_OUT_NAMES, *water)
        season = self.readout == "season"

        def outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
            crop = get_path(state, f"crops.{SLOT}")
            pg = plantgro_outputs(crop, params["crop"], forcing_t["crop"])
            first = lambda x: jnp.ravel(x)[0]  # noqa: E731
            out = {**{n: first(pg[n]) for n in CROP_OUT_NAMES}, "istage": first(pg["istage"])}
            if season:  # the padded readout keeps its program as it was
                out["mdate"] = first(crop.phen.mdate)
            if water:
                # the day state as the assembly's own outputs read it (soil_sw, evap, ep)
                d = day_out(state, params, forcing_t)
                for w in water:
                    out[w] = first(d["evap"]) + first(d["ep"]) if w == "et" else d["soil_sw"][int(w[2:]) - 1]
            return out

        model = day_dssat486(SLOT).compile(day_processes(SLOT, mesev=g[1]), outputs=outputs, exact_lags=True)
        step = jax.vmap(model.compile(), in_axes=(0, 0, 0))
        n_days = self.n_days[g]
        names = plan.cul

        def take(theta: Any, cols: Any) -> Any:
            """``theta[s, cols[s, ...]]`` (columns -1 read column 0; callers mask them)."""
            c = jnp.maximum(cols, 0)
            flat = jnp.take_along_axis(theta, c.reshape(c.shape[0], -1), axis=1)
            return flat.reshape(c.shape)

        def set_cultivar(params: Any, theta: Any, tid: Any) -> Any:
            if not names:
                return params
            c = params["crop"].cultivar
            if plan.cul_static is not None:
                cols = plan.cul_static
                vals = tuple(
                    theta[:, cols[i]].astype(getattr(c, f).dtype).reshape(getattr(c, f).shape)
                    for i, f in enumerate(names)
                )
            else:
                ct = jnp.asarray(plan.cul_col, jnp.int32)[tid]
                got = take(theta, ct)
                vals = tuple(
                    jnp.where(
                        (ct[:, i] >= 0).reshape(getattr(c, f).shape[:1] + (1,) * (getattr(c, f).ndim - 1)),
                        got[:, i].astype(getattr(c, f).dtype).reshape(getattr(c, f).shape),
                        getattr(c, f),
                    )
                    for i, f in enumerate(names)
                )
            cul = eqx.tree_at(lambda x: tuple(getattr(x, f) for f in names), c, vals)
            crop = eqx.tree_at(lambda x: x.cultivar, params["crop"], cul)
            return {**params, "crop": crop}

        def set_soil(params: Any, state0: Any, theta: Any, tid: Any) -> tuple[Any, Any, Any, Any]:
            assert plan.col is not None and plan.site is not None and plan.keep is not None
            soil = {}
            for q in plan.soil:
                ct = jnp.asarray(plan.col[q], jnp.int32)[tid]
                base = jnp.asarray(plan.site[q])[tid]
                soil[q] = jnp.where(ct >= 0, take(theta, ct).astype(base.dtype), base)
            keep = {q: jnp.asarray(plan.keep[q])[tid] for q in plan.soil}

            def one(p: Any, s0: Any, sl: Any, kp: Any, index: Any, coef: Any, swi: Any, swm: Any) -> Any:
                def expand(x: Any) -> Any:
                    return jnp.sum(x[..., index] * coef.astype(x.dtype), axis=-1)

                return write_soil(p, s0, sl, expand=expand, sw_input=swi, sw_missing=swm, keep=kp)

            p, s, fsoil = jax.vmap(one)(
                params,
                state0,
                soil,
                keep,
                jnp.asarray(plan.index)[tid],
                jnp.asarray(plan.coef)[tid],
                jnp.asarray(plan.sw_input)[tid],
                jnp.asarray(plan.sw_missing)[tid],
            )
            return p, s, fsoil, keep

        def patch_forcing(f_t: Any, fsoil: Any, keep: Any) -> Any:
            """The day's forcing with the written soil quantities (a kept one stays the forcing's)."""
            bf = f_t["soil"]
            if bf.soil is None:
                return f_t
            fs = bf.soil
            repl = {}
            for k in FORCING_SOIL_VALUES:
                if k not in plan.soil:
                    continue
                cur = getattr(fs, k)
                kp = keep[k].reshape(keep[k].shape + (1,) * (cur.ndim - 1))
                repl[k] = jnp.where(kp, cur, jnp.broadcast_to(fsoil[k], cur.shape).astype(cur.dtype))
            return {**f_t, "soil": bf.replace(soil=fs.replace(**repl))}

        def sim(inputs: Any, tab: Any, theta: Any, tid: Any, *extra: Any) -> Any:
            params_tr, forcing_tr, state_tr = inputs
            params = set_cultivar(jax.tree.map(lambda x: x[tid], params_tr), theta, tid)
            state0 = jax.tree.map(lambda x: x[tid], state_tr)
            if plan.soil:
                params, state0, fsoil, keep = set_soil(params, state0, theta, tid)

                def day_forcing(t: Any) -> Any:
                    return patch_forcing(jax.tree.map(lambda x: x[tid, t], forcing_tr), fsoil, keep)
            else:

                def day_forcing(t: Any) -> Any:
                    return jax.tree.map(lambda x: x[tid, t], forcing_tr)

            if forcing_hook is None:

                def body(s: Any, t: Any) -> Any:
                    return step(s, params, day_forcing(t))

            else:
                hook = forcing_hook

                def body(s: Any, t: Any) -> Any:
                    return step(s, params, hook(day_forcing(t), t, tid, *extra))

            _, outs = jax.lax.scan(body, state0, jnp.arange(n_days))
            y = jnp.stack([outs[n] for n in names_out], axis=0)  # [O, T, S]
            if season:
                return jax.vmap(_gather_season, in_axes=(2, 1, 1, 0, 0, 0, 0, 0))(
                    y,
                    outs["istage"],
                    outs["mdate"],
                    tab["last"][tid],
                    tab["open"][tid],
                    tab["type"][tid],
                    tab["out"][tid],
                    tab["t"][tid],
                )
            return jax.vmap(_gather_one, in_axes=(2, 1, 0, 0, 0))(
                y, outs["istage"], tab["type"][tid], tab["out"][tid], tab["t"][tid]
            )

        return sim

    def _program(self, g: tuple[int, str], sp: int) -> Any:
        key = (g, sp)
        if key in self.programs:
            return self.programs[key]
        import jax
        import jax.numpy as jnp

        from agrijax.core.grad import bind_gradient_mode

        fn = bind_gradient_mode(self._sim_fn(g), GRADIENT_MODE)
        jf = jax.jit(fn, in_shardings=(self.rep, self.rep, self.bsh, self.bsh), out_shardings=self.bsh)
        args = (self.inputs[g], self.tables[g], jnp.zeros((sp, self.n_theta)), jnp.zeros(sp, jnp.int32))
        t0 = time.perf_counter()
        compiled = jf.lower(*args).compile()
        self.compile_s += time.perf_counter() - t0
        self.n_compiled += 1
        self.programs[key] = compiled
        return compiled

    def _pad(self, s: int) -> int:
        """Padded sample count: a multiple of the device count on the ladder ``ndev * 2^j``."""
        m = self.ndev
        while m < s:
            m *= 2
        return m

    @property
    def chunk(self) -> int:
        m = self.ndev
        while m * 2 <= max(CHUNK_TARGET, self.ndev):
            m *= 2
        return m

    def __call__(self, theta: np.ndarray, tid: np.ndarray) -> np.ndarray:
        import jax

        theta = np.asarray(theta, dtype=np.float64)
        tid = np.asarray(tid, dtype=np.int64)
        if theta.ndim != 2 or theta.shape[1] != self.n_theta:
            raise ValueError(
                f"theta must be [samples, {self.n_theta}] (the space's columns), got {theta.shape}"
            )
        out = np.zeros((theta.shape[0], self.n_entries))
        for g in self.groups:
            pos = np.nonzero(np.asarray([self.where[int(i)][0] == g for i in tid], dtype=bool))[0]
            for c0 in range(0, pos.size, self.chunk):
                p = pos[c0 : c0 + self.chunk]
                have = sorted(s_ for (gg, s_) in self.programs if gg == g and s_ >= p.size)
                sp = have[0] if have else self._pad(p.size)
                th = np.repeat(theta[p[:1]], sp, axis=0)
                th[: p.size] = theta[p]
                loc = np.full(sp, self.where[int(tid[p[0]])][1], np.int32)
                loc[: p.size] = [self.where[int(i)][1] for i in tid[p]]
                prog = self._program(g, sp)
                t0 = time.perf_counter()
                y = np.asarray(
                    prog(
                        self.inputs[g],
                        self.tables[g],
                        jax.device_put(th, self.bsh),
                        jax.device_put(loc, self.bsh),
                    )
                )
                out[p] = y[: p.size]
                self.call_s += time.perf_counter() - t0
                self.program_calls += 1
                self.seasons += int(p.size)
        return out

    def jax_loss(
        self, obs: np.ndarray, winv: np.ndarray, treatments: Sequence[int] | None = None
    ) -> Callable[[Any], Any]:
        """Pure JAX ``theta [P] -> [B]``: every treatment's loss ``sum_e ((y_e - obs_e) winv_e)^2`` at
        ``theta`` (``obs``, ``winv`` ``[B, E]``, treatment order), bound to :data:`GRADIENT_MODE`.
        Traced as a whole (not the compiled programs), every batch group in the one function, the
        losses returned in treatment order: for the trust report and gradient steps.

        ``treatments``: distinct run indices; the function then simulates those runs only and returns
        their losses in that order (``obs`` and ``winv`` still hold every run's rows)."""
        import jax.numpy as jnp

        from agrijax.core.grad import bind_gradient_mode

        n_runs = len(self.runs)
        if treatments is None:
            sel = list(range(n_runs))
        else:
            sel = [int(i) for i in treatments]
            if not sel or len(set(sel)) != len(sel) or not all(0 <= i < n_runs for i in sel):
                raise ValueError(f"treatments: distinct run indices in [0, {n_runs}), got {list(treatments)}")
        rank = {i: k for k, i in enumerate(sel)}
        parts = []
        for g, idx_all in self.groups.items():
            idx = [i for i in idx_all if i in rank]
            if not idx:
                continue
            order = np.asarray([self.where[i][1] for i in idx])
            parts.append(
                (
                    self._sim_fn(g),
                    len(idx),
                    self.inputs[g],
                    self.tables[g],
                    jnp.asarray(order, jnp.int32),
                    jnp.asarray(np.asarray(obs)[idx]),
                    jnp.asarray(np.asarray(winv)[idx]),
                    np.asarray([rank[i] for i in idx]),
                )
            )
        n_out = len(sel)
        n_theta = self.n_theta
        in_order = len(parts) == 1 and np.array_equal(parts[0][-1], np.arange(n_out))

        def f(theta: Any) -> Any:
            losses = []
            for sim, b_n, inputs, tab, order, o, w, _ in parts:
                th = jnp.broadcast_to(jnp.asarray(theta), (b_n, n_theta))
                y = sim(inputs, tab, th, order)
                losses.append(jnp.sum(((y - o) * w) ** 2, axis=1))
            if in_order:
                return losses[0]
            out = jnp.zeros(n_out, losses[0].dtype)
            for (*_, pos), l_g in zip(parts, losses, strict=True):
                out = out.at[pos].set(l_g)
            return out

        return bind_gradient_mode(f, GRADIENT_MODE)

    def stats(self) -> dict[str, Any]:
        return {
            "compile_s": self.compile_s,
            "call_s": self.call_s,
            "n_programs": self.n_compiled,
            "program_calls": self.program_calls,
            "seasons": self.seasons,
            "n_devices": self.ndev,
        }
