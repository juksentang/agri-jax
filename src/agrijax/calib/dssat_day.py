"""Batched evaluation of the free-run DSSAT-CSM v4.8.6.0 day for cultivar calibration.

:class:`DaySimulator` runs the day (:mod:`agrijax.models.day_dssat486`, the free-run configuration of
:mod:`agrijax.sites.dssat_free_run`) on **samples** ``(theta, treatment)``: ``theta`` holds the six
``MZCER048.CUL`` coefficients (:data:`CUL_ORDER`), the treatment is an index into the treatments the
simulator was built for. One call evaluates any number of samples of any treatments: the samples
of one batch group (treatments with one soil layer count and one ``MESEV``; their forcings padded to
one length) go through one compiled program (a ``vmap`` over the samples inside one ``scan`` over
the days), sharded over the host devices (``jax.devices()``; on a CPU node set
``XLA_FLAGS=--xla_force_host_platform_device_count=<cores>`` before JAX is imported to use every core).

A call returns the values of the treatment's **entries** only (:class:`Entry`: a stage date, a value
at the end of the crop or a value on a day), gathered on the device, so a call moves
``[samples, entries]`` numbers back to the host. Programs are compiled once per (group, padded
sample count) and kept; compile time, call time, program calls and simulated seasons are counted
(:meth:`DaySimulator.stats`).

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
per-treatment loss for the gradient-trust report and gradient steps, in the same mode.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CROP_END",
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
    "DaySimulator",
    "Entry",
    "entry_values",
]

#: the calibrated coefficients in ``MZCER048.CUL`` column order (``theta`` columns)
CUL_ORDER: tuple[str, ...] = ("P1", "P2", "P5", "G2", "G3", "PHINT")
_FIELDS = {"P1": "p1", "P2": "p2", "P5": "p5", "G2": "g2", "G3": "g3", "PHINT": "phint"}
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
#: daily model outputs an entry can read (``plantgro_outputs`` names; ``Entry.out`` indexes this)
OUT_NAMES: tuple[str, ...] = ("lai", "cwad", "gwad", "g_ad", "lsd", "lwad", "swad", "rwad", "pwad")
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
    ISTAGE 10 or the group's last day."""

    kind: int
    out: int
    t: int = 0


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
    ``E_FINAL`` reads ``t_last``)."""
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

    def series(k: int) -> np.ndarray:
        a = np.asarray(outputs[OUT_NAMES[k]])
        return a.reshape(a.shape[0], -1)[:, 0] if a.ndim > 1 else a

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
    """The free-run day on samples ``(theta [6], treatment)`` (module docstring).

    ``runs`` are :class:`agrijax.sites.dssat_free_run.FreeRunInputs`; ``entries[b]`` the entries of
    run ``b``. Calling the simulator with ``theta [S, 6]`` and treatment indices ``tid [S]`` returns
    ``[S, E]`` (``E`` = the most entries of any treatment; unused entries 0).

    ``devices``: the devices the batch is sharded over (default: ``jax.devices()``, as before).
    ``dtype``: the floating dtype the inputs are stored in on the devices (default: as built,
    float64); ``np.float32`` stores them rounded once, and the calls must then run with
    ``jax_enable_x64`` off (:mod:`agrijax.facade_execution` does that).

    ``readout``: ``"season"`` (default; the season readout of the module docstring, with run ``b``'s
    last real day in :attr:`last_days`) or ``"padded"`` (the earlier readout, bit for bit; no
    ``E_STATUS`` or ``CROP_END`` entries). Under ``"season"`` an ``E_DAY`` entry after the run's last
    real day raises ``ValueError``."""

    def __init__(
        self,
        runs: Sequence[Any],
        entries: Sequence[Sequence[Entry]],
        *,
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
        self._build(runs, entries, pad_days=pad_days, devices=devices, dtype=dtype)

    def _build(
        self,
        runs: Sequence[Any],
        entries: Sequence[Sequence[Entry]],
        *,
        pad_days: int,
        devices: Sequence[Any] | None,
        dtype: Any,
    ) -> None:
        import jax
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as PS

        from agrijax.sites.dssat_free_run import stack_trees

        self.runs = list(runs)
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
        self.where: dict[int, tuple[tuple[int, str], int]] = {}
        self.n_days: dict[tuple[int, str], int] = {}
        self.inputs: dict[tuple[int, str], Any] = {}
        self.tables: dict[tuple[int, str], Any] = {}
        for g, idx in groups.items():
            n = max(self.runs[i].n_days for i in idx) + int(pad_days)
            self.n_days[g] = n
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
                    tab["type"][b, j], tab["out"][b, j], tab["t"][b, j] = e.kind, e.out, e.t
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
        """``sim(inputs, tab, theta [S, 6], tid [S]) -> [S, E]`` of group ``g`` (traceable).

        ``forcing_hook``: ``hook(forcing_t, t, tid, *extra) -> forcing_t``, applied to every day's
        forcing of the samples before the day runs; ``sim`` then takes the hook's ``extra``
        arguments after ``tid`` (the weather sensitivity, :mod:`agrijax.facade_weather`, adds its
        perturbation there). Without it the program is the one the calibration runs."""
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from agrijax.core.state import get_path
        from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_processes
        from agrijax.processes.crop.ceres_maize import plantgro_outputs

        season = self.readout == "season"

        def outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
            crop = get_path(state, f"crops.{SLOT}")
            pg = plantgro_outputs(crop, params["crop"], forcing_t["crop"])
            first = lambda x: jnp.ravel(x)[0]  # noqa: E731
            out = {**{n: first(pg[n]) for n in OUT_NAMES}, "istage": first(pg["istage"])}
            if season:  # the padded readout keeps its program as it was
                out["mdate"] = first(crop.phen.mdate)
            return out

        model = day_dssat486(SLOT).compile(day_processes(SLOT, mesev=g[1]), outputs=outputs, exact_lags=True)
        step = jax.vmap(model.compile(), in_axes=(0, 0, 0))
        n_days = self.n_days[g]
        names = [_FIELDS[n] for n in CUL_ORDER]

        def set_cultivar(params: Any, theta: Any) -> Any:
            c = params["crop"].cultivar
            vals = tuple(
                theta[:, i].astype(getattr(c, f).dtype).reshape(getattr(c, f).shape)
                for i, f in enumerate(names)
            )
            cul = eqx.tree_at(lambda x: tuple(getattr(x, f) for f in names), c, vals)
            crop = eqx.tree_at(lambda x: x.cultivar, params["crop"], cul)
            return {**params, "crop": crop}

        def sim(inputs: Any, tab: Any, theta: Any, tid: Any, *extra: Any) -> Any:
            params_tr, forcing_tr, state_tr = inputs
            params = set_cultivar(jax.tree.map(lambda x: x[tid], params_tr), theta)
            state0 = jax.tree.map(lambda x: x[tid], state_tr)

            if forcing_hook is None:

                def body(s: Any, t: Any) -> Any:
                    return step(s, params, jax.tree.map(lambda x: x[tid, t], forcing_tr))

            else:
                hook = forcing_hook

                def body(s: Any, t: Any) -> Any:
                    f_t = hook(jax.tree.map(lambda x: x[tid, t], forcing_tr), t, tid, *extra)
                    return step(s, params, f_t)

            _, outs = jax.lax.scan(body, state0, jnp.arange(n_days))
            y = jnp.stack([outs[n] for n in OUT_NAMES], axis=0)  # [O, T, S]
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
        args = (self.inputs[g], self.tables[g], jnp.zeros((sp, len(CUL_ORDER))), jnp.zeros(sp, jnp.int32))
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

    def jax_loss(self, obs: np.ndarray, winv: np.ndarray) -> Callable[[Any], Any]:
        """Pure JAX ``theta [6] -> [B]``: every treatment's loss ``sum_e ((y_e - obs_e) winv_e)^2`` at
        ``theta`` (``obs``, ``winv`` ``[B, E]``; one group only), bound to :data:`GRADIENT_MODE`.
        Traced as a whole (not the compiled programs): for the trust report and gradient steps."""
        import jax.numpy as jnp

        from agrijax.core.grad import bind_gradient_mode

        if len(self.groups) != 1:
            raise ValueError("jax_loss needs the treatments in one batch group (one layer count and MESEV)")
        (g,) = self.groups
        sim = self._sim_fn(g)
        b_n = len(self.groups[g])
        inputs, tab = self.inputs[g], self.tables[g]
        order = np.asarray([self.where[i][1] for i in self.groups[g]])
        o = jnp.asarray(np.asarray(obs)[self.groups[g]])
        w = jnp.asarray(np.asarray(winv)[self.groups[g]])

        def f(theta: Any) -> Any:
            th = jnp.broadcast_to(jnp.asarray(theta), (b_n, len(CUL_ORDER)))
            y = sim(inputs, tab, th, jnp.asarray(order, jnp.int32))
            return jnp.sum(((y - o) * w) ** 2, axis=1)

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
