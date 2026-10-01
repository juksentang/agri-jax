"""Helpers of ``test_bucket_dssat.py``: the DSSAT-CSM v4.8.6.0 ``WATBAL`` dumps as model inputs.

The ``WATBAL`` collection (a local gfortran 13.3 build of the v4.8.6.0 source instrumented at
``WATBAL`` and its callees; every printed output of all 58 M2 treatments, the 14 CA-TPA runs and
the winter-wheat runs identical to build486's) wrote, per run, the daily tables
``<tables>/<key>/<table>.npz`` (:func:`agrijax.port.dumps.save_table`):

``wb_rate_in`` / ``wb_rate_out``, ``wb_integr_in`` / ``wb_integr_out``
    ``WATBAL`` entry / exit of the RATE and INTEGR calls (arguments, ``SAVE``d locals, the
    ``SOILPROP``, weather and ``MULCH`` components as ``SP_*``, ``RAIN``, ``TMAX``, ``M_*``);
``mulch_rate_*``, ``mulch_integr_*``, ``snow_*``, ``rnoff_*``, ``infil_*``, ``satflo_*``, ``up_flow_*``
    the callees' entry / exit. ``MULCHWATER``'s ``DYNAMIC`` is a local copied from ``CONTROL``
    after the entry dump, so ``mulch_rate_*`` (selected with ``DYNAMIC = RATE`` at entry) are the
    entry and exit of the INTEGR call and ``mulch_integr_*`` those of the OUTPUT call: their saved
    locals hold the RATE call's inputs and results (``MULCHEVAP`` of yesterday's SPAM,
    ``MULWATADD``, ``RESWATADD``) and the INTEGR call's (today's ``MULCHEVAP``; ``MULCHWAT`` after
    the integration) respectively; ``mulch_rate_in``'s ``MULCHWAT`` is the day's starting value.

:func:`load_run` reads one run; :func:`inputs` turns a list of runs into batched model inputs (every
run padded to the same layer count, ``dlayr = 0`` below its ``NLAYR``, and the same day count,
zero forcing after its last day): the bucket parameters and forcing, with the day's ``SOILPROP``
replayed, and the replay of what ``SPAM`` hands ``WATBAL`` INTEGR (``SWDELTX`` as the per-layer
uptake ``-SWDELTX DLAYR_YEST``, ``ES``, the mulch evaporation and, for ``MESEV = 'S'``, the SALUS
per-layer evaporation ``-SWDELTU DLAYR_YEST``). :func:`replay_model` is the day: a replay entry
(the stand-in for SPAM), ``bucket_rate`` and ``bucket_integrate`` in DSSAT's order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from agrijax.core.model import Model
from agrijax.core.process import process
from agrijax.core.state import Forcing, field
from agrijax.core.units import CM_PER_MM
from agrijax.iface.surface import PETFluxes
from agrijax.port import dumps
from agrijax.processes.soil_water.bucket import (
    BucketForcing,
    BucketParams,
    BucketSoil,
    BucketState,
    MulchForcing,
    bucket_integrate,
    bucket_rate,
    bucket_storage,
)

#: the ``WATBAL`` dump tables under the data directory
TABLES = Path("validation/aj_dsw/d1a/tables")
RATE_TABLES = ("wb_rate_in", "wb_rate_out", "wb_integr_in", "wb_integr_out", "mulch_rate_in", "mulch_integr_in",
               "mulch_integr_out")  # fmt: skip


def run_keys(tables: Path) -> list[str]:
    return sorted(p.name for p in tables.iterdir() if p.is_dir() and not p.name.startswith("_"))


def _char(a: Any) -> str:
    v = np.asarray(a).reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


@dataclass
class Run:
    """One run's dump tables (``values`` per table) and its meta."""

    key: str
    meta: dict[str, Any]
    t: dict[str, dict[str, np.ndarray]]
    date: np.ndarray
    nl: int
    dates: dict[str, np.ndarray]

    @property
    def n_days(self) -> int:
        return int(self.date.shape[0])

    def v(self, table: str, name: str) -> np.ndarray:
        return self.t[table][name]


def load_run(tables: Path, key: str) -> Run:
    d = tables / key
    t: dict[str, dict[str, np.ndarray]] = {}
    dates: dict[str, np.ndarray] = {}
    date = None
    for name in sorted(p.stem for p in d.glob("*.npz")):
        tab = dumps.load_table(d / f"{name}.npz")[0]
        t[name] = tab.values
        dates[name] = tab.date
        if name == "wb_rate_in":
            date = tab.date
        elif name in RATE_TABLES and date is not None and not np.array_equal(tab.date, date):
            raise AssertionError(f"{key}: {name} dates differ from wb_rate_in")
    assert date is not None, key
    for name in RATE_TABLES:
        if name in t and len(next(iter(t[name].values()))) != len(date):
            raise AssertionError(
                f"{key}: {name} has {len(next(iter(t[name].values())))} days, not {len(date)}"
            )
    nl = int(t["wb_rate_in"]["SP_NLAYR"][0])
    return Run(key=key, meta=json.loads((d / "meta.json").read_text()), t=t, date=date, nl=nl, dates=dates)


# ----------------------------------------------------------------------------- the replay
class ReplayForcing(Forcing):
    """The bucket's forcing and the replay of SPAM's outputs (time axis first)."""

    bucket: BucketForcing
    uptake: Array = field(unit="cm d-1", dims=("T", "n_layer"), description="-SWDELTX DLAYR_YEST")
    evap_layers: Array = field(
        unit="cm d-1", dims=("T", "n_layer"), description="-SWDELTU DLAYR_YEST (SALUS)"
    )
    es: Array = field(unit="cm d-1", dims=("T",), description="ES / 10")
    em: Array = field(unit="cm d-1", dims=("T",), description="mulch evaporation / 10")


@process(
    reads=(),
    writes=("sink_in.uptake", "evap_layers", "pet"),
    register=False,
    source="test replay of DSSAT-CSM v4.8.6.0 SPAM outputs",
)
def spam_replay(state: BucketState, params: Any, forcing_t: ReplayForcing) -> BucketState:
    """SPAM's outputs of the day from the reference dump.

    Source: DSSAT-CSM v4.8.6.0 SPAM/SPAM.for (ES, EM, SWDELTX handed to WATBAL INTEGR).
    """
    pet = state.pet.replace(soil_evaporation=forcing_t.es, residue_evaporation=forcing_t.em)
    return eqx.tree_at(
        lambda s: (s.sink_in.uptake, s.evap_layers, s.pet),
        state,
        (forcing_t.uptake, forcing_t.evap_layers, pet),
    )


def _bucket_view(proc: Any) -> Any:
    """Run a bucket process on the forcing's ``bucket`` subtree."""

    def fn(state: BucketState, params: BucketParams, forcing_t: ReplayForcing) -> BucketState:
        return proc(state, params, forcing_t.bucket)

    return process(
        fn, reads=proc.reads, writes=proc.writes, name=proc.name, register=False, source=proc.source
    )


def outputs(state: BucketState, params: BucketParams, forcing_t: ReplayForcing) -> dict[str, Array]:
    f = state.flux
    return {
        "sw": state.sw,
        "snow": state.snow,
        "mulch_wat": state.mulch_wat,
        "runoff": f.runoff,
        "infiltration": f.infiltration,
        "drain": f.drain,
        "drn": f.drn,
        "upflow": f.upflow,
        "swdelts": f.swdelts,
        "swdeltu": f.swdeltu,
        "watavl": f.watavl,
        "truncation": f.truncation,
        "residue_water": f.residue_water,
        "storage": bucket_storage(state, params, forcing_t.bucket),
    }


def replay_model() -> Model:
    """SPAM replay, then RATE, then INTEGR (the replay writes only what INTEGR reads)."""
    return Model(
        BucketState,
        [_bucket_view(bucket_rate), spam_replay, _bucket_view(bucket_integrate)],
        outputs=outputs,
        name="d1a_replay",
    )


# ----------------------------------------------------------------------------- inputs
def _series(a: Any, n_days: int, n_layer: int | None = None) -> np.ndarray:
    """``a`` ([d] or [d, >= n_layer]) as ``[n_days(, n_layer)]`` float64: layers beyond the run's
    zero, days after its last day repeating the last day (never compared)."""
    a = np.asarray(a, dtype=np.float64)
    if n_layer is not None:
        b = np.zeros((a.shape[0], n_layer))
        k = min(a.shape[1], n_layer)
        b[:, :k] = a[:, :k]
        a = b
    idx = np.minimum(np.arange(n_days), a.shape[0] - 1)
    return a[idx]


def _layers(r: Run, table: str, name: str, n_days: int, n_layer: int) -> np.ndarray:
    return _series(np.asarray(r.v(table, name), dtype=np.float64)[:, : r.nl], n_days, n_layer)


def _scalar(r: Run, table: str, name: str, n_days: int) -> np.ndarray:
    return _series(r.v(table, name), n_days)


def _zero_after(a: np.ndarray, n: int) -> np.ndarray:
    a = a.copy()
    a[n:] = 0.0
    return a


def inputs(runs: list[Run], n_layer: int | None = None, n_days: int | None = None):
    """``(params, forcing, state0)`` batched over ``runs`` (leading axis), float64, from the dumps.
    After a run's last day its water inputs are zero and its soil stays that of the last day."""
    nl = n_layer or max(r.nl for r in runs)
    nd = n_days or max(r.n_days for r in runs)
    P, F, S = [], [], []
    ri, ii = "wb_rate_in", "wb_integr_in"
    for r in runs:
        n = r.n_days
        lay = {
            k: _layers(r, ri, f"SP_{k.upper()}", nd, nl) for k in ("dlayr", "ds", "ll", "dul", "sat", "swcn")
        }
        sca = {k: _scalar(r, ri, f"SP_{k.upper()}", nd) for k in ("cn", "swcon")}
        soil_t = BucketSoil(**{k: jnp.asarray(v) for k, v in {**lay, **sca}.items()})
        dly = _layers(r, ii, "DLAYR_YEST", nd, nl)
        mulch = MulchForcing(
            mass=jnp.asarray(_scalar(r, ri, "M_MASS", nd)),
            cover=jnp.asarray(_scalar(r, ri, "M_COVER", nd)),
            new_mass=jnp.asarray(_zero_after(_scalar(r, ri, "M_NEW", nd), n)),
            watfac=jnp.asarray(_scalar(r, ri, "M_WATFAC", nd)),
        )
        bf = BucketForcing(
            rain=jnp.asarray(_zero_after(_scalar(r, ri, "RAIN", nd), n)),
            tmax=jnp.asarray(_scalar(r, ri, "TMAX", nd)),
            irrigation=jnp.asarray(_zero_after(_scalar(r, ri, "IRRAMT", nd), n)),
            mulch=mulch,
            soil=soil_t,
            dlayr_end=jnp.asarray(_layers(r, ii, "SP_DLAYR", nd, nl)),
        )
        salus = _char(r.v(ri, "MESEV")[0]) == "S"
        swdeltx = _zero_after(_layers(r, ii, "SWDELTX", nd, nl), n)
        swdeltu = _zero_after(_layers(r, ii, "SWDELTU", nd, nl), n)
        f = ReplayForcing(
            bucket=bf,
            uptake=jnp.asarray(-swdeltx * dly),
            evap_layers=jnp.asarray(-swdeltu * dly if salus else np.zeros_like(swdeltu)),
            es=jnp.asarray(_zero_after(_scalar(r, ii, "ES", nd), n) * CM_PER_MM),
            em=jnp.asarray(_zero_after(_scalar(r, "mulch_integr_in", "MULCHEVAP", nd), n) * CM_PER_MM),
        )
        soil0 = BucketSoil(**{k: jnp.asarray(v[0]) for k, v in {**lay, **sca}.items()})
        meinf = _char(r.v(ri, "MEINF")[0])
        p = BucketParams(
            soil=soil0,
            mulch_on=jnp.asarray(1.0 if meinf.strip() and meinf in "RSM" else 0.0),
            salus_es=jnp.asarray(1.0 if salus else 0.0),
            actwtd=jnp.asarray(float(r.v(ri, "ACTWTD")[0])),
            pm_fraction=jnp.asarray(float(r.v("rnoff_in", "PMFRACTION")[0]) if "rnoff_in" in r.t else 0.0),
        )
        s = BucketState.initial(
            _layers(r, ri, "SW", 1, nl)[0],
            snow=float(r.v(ri, "SNOW")[0]),
            mulch_wat=float(r.v("mulch_rate_in", "MULCHWAT")[0]),
            mulch_evap_prev=float(r.v("mulch_rate_in", "MULCHEVAP")[0]),
        ).replace(pet=PETFluxes.zeros())
        P.append(p)
        F.append(f)
        S.append(s)
    return tree_stack(P), tree_stack(F), tree_stack(S)


def tree_stack(trees: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


# ----------------------------------------------------------------------------- one-day inputs
def day_inputs(runs: list[Run], n_layer: int | None = None):
    """Every (run, day) as one sample: ``(params, forcing with a 1-day axis, state0)`` from the
    day's entry values, stacked over all days of all runs (a vmap batch), plus the (run, day) index."""
    nl = n_layer or max(r.nl for r in runs)
    P, F, S = inputs(runs, nl)
    idx = [(i, d) for i, r in enumerate(runs) for d in range(r.n_days)]
    ii = np.asarray([i for i, _ in idx])
    dd = np.asarray([d for _, d in idx])
    p = jax.tree_util.tree_map(lambda x: x[ii], P)
    f = jax.tree_util.tree_map(lambda x: x[ii, dd][:, None], F)
    sw = np.stack(
        [
            _layers(r, "wb_rate_in", "SW", r.n_days, nl)[d]
            for r, (_, d) in zip([runs[i] for i in ii], idx, strict=True)
        ]
    )
    s = jax.tree_util.tree_map(lambda x: x[ii], S)
    snow = np.asarray([float(runs[i].v("wb_rate_in", "SNOW")[d]) for i, d in idx])
    mw = np.asarray([float(runs[i].v("mulch_rate_in", "MULCHWAT")[d]) for i, d in idx])
    me = np.asarray([float(runs[i].v("mulch_rate_in", "MULCHEVAP")[d]) for i, d in idx])
    s = s.replace(sw=jnp.asarray(sw), theta=jnp.asarray(sw), snow=jnp.asarray(snow), mulch_wat=jnp.asarray(mw),
                  mulch_evap_prev=jnp.asarray(me))  # fmt: skip
    return p, f, s, idx


# ----------------------------------------------------------------------------- the reference
def reference(r: Run, n_layer: int) -> dict[str, np.ndarray]:
    """The reference's values of :func:`outputs` (``[n_days, ...]``), from the exits."""
    nd = r.n_days
    ro, io = "wb_rate_out", "wb_integr_out"
    out = {
        "sw": _layers(r, io, "SW", nd, n_layer),
        "snow": np.asarray(r.v(ro, "SNOW"), float),
        "mulch_wat": np.asarray(r.v("mulch_integr_out", "MULCHWAT"), float),
        "runoff": np.asarray(r.v(ro, "RUNOFF"), float),
        "infiltration": np.asarray(r.v(ro, "INFILT"), float),
        "drain": np.asarray(r.v(ro, "DRAIN"), float),
        "drn": _layers(r, ro, "DRN", nd, n_layer),
        "upflow": _layers(r, ro, "UPFLOW", nd, n_layer),
        "swdelts": _layers(r, ro, "SWDELTS", nd, n_layer),
        "swdeltu": _layers(r, ro, "SWDELTU", nd, n_layer),
        "watavl": np.asarray(r.v(ro, "WATAVL"), float),
    }
    return out
