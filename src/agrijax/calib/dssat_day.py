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
at simulated maturity or a value on a day), gathered on the device, so a call moves
``[samples, entries]`` numbers back to the host. Programs are compiled once per (group, padded
sample count) and kept; compile time, call time, program calls and simulated seasons are counted
(:meth:`DaySimulator.stats`).

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
    "CUL_ORDER",
    "E_DATE",
    "E_DAY",
    "E_FINAL",
    "E_PAD",
    "GRADIENT_MODE",
    "OUT_NAMES",
    "PAD_DAYS",
    "DaySimulator",
    "Entry",
    "entry_values",
]

#: the calibrated coefficients in ``MZCER048.CUL`` column order (``theta`` columns)
CUL_ORDER: tuple[str, ...] = ("P1", "P2", "P5", "G2", "G3", "PHINT")
_FIELDS = {"P1": "p1", "P2": "p2", "P5": "p5", "G2": "g2", "G3": "g3", "PHINT": "phint"}
#: gradient mode of every program (``ste``: identity derivative through Fortran's truncations and roundings,
#: e.g. RLV to 1e-3 and SW to 1e-6, and the straight-through ramp of ``event_ste`` events; the CERES-Maize
#: stage changes are plain selections, the same in every mode; forward values do not depend on the mode)
GRADIENT_MODE = "ste"
#: days of forcing added past the longest season of a batch group (the last day repeated, rain and
#: irrigation 0), so a candidate cultivar that matures later than the reference season still
#: reaches maturity; values on these days are not observations
PAD_DAYS = 60
#: daily model outputs an entry can read (``plantgro_outputs`` names; ``Entry.out`` indexes this)
OUT_NAMES: tuple[str, ...] = ("lai", "cwad", "gwad", "g_ad", "lsd", "lwad", "swad", "rwad", "pwad")
#: entry types
E_PAD, E_DATE, E_FINAL, E_DAY = 0, 1, 2, 3
#: stage codes of the dates an entry can read (first day of the code in ``istage``): emergence,
#: silking (ISTAGE 4, MZ_PHENOL.for:783) and the day after physiological maturity (ISTAGE 10,
#: MZ_PHENOL.for:899), as :data:`agrijax.calib.ceres.ISTAGE_SILKING_OUT` / ``ISTAGE_MATURITY_OUT``
ISTAGE_EMERGENCE, ISTAGE_SILKING, ISTAGE_MATURITY = 1, 4, 10
#: largest sample count of one program call (memory), rounded down to the ladder of the padding
CHUNK_TARGET = 12288


@dataclass(frozen=True)
class Entry:
    """One simulated quantity of a treatment.

    ``kind``: ``E_DATE`` (``out`` = a stage code; the value is the day index of its first day, the
    season length if never reached), ``E_FINAL`` (``out`` = an :data:`OUT_NAMES` index; the value on
    the simulated maturity day, the last day if maturity is never reached) or ``E_DAY`` (the output on
    day index ``t``)."""

    kind: int
    out: int
    t: int = 0


def _first_idx(hit: Any) -> Any:
    import jax.numpy as jnp

    t = hit.shape[0]
    return jnp.where(jnp.any(hit, axis=0), jnp.argmax(hit, axis=0), t)


def _gather_one(y: Any, ist: Any, etype: Any, eout: Any, et: Any) -> Any:
    """One sample: entry values ``[E]`` from its daily outputs ``y [O, T]`` and stage codes ``ist [T]``."""
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


def entry_values(outputs: dict[str, np.ndarray], entries: Sequence[Entry]) -> np.ndarray:
    """Host version of the gather: ``outputs[name] [T]`` of one run (and ``istage``) -> ``[E]``."""
    ist = np.asarray(outputs["istage"])
    t_n = ist.shape[0]

    def first(code: int) -> int:
        hit = np.nonzero(ist == code)[0]
        return int(hit[0]) if hit.size else t_n

    t_mat = min(first(ISTAGE_MATURITY), t_n - 1)
    out = np.zeros(len(entries))
    for j, e in enumerate(entries):
        if e.kind == E_DATE:
            out[j] = first(e.out)
        elif e.kind == E_FINAL:
            out[j] = float(np.asarray(outputs[OUT_NAMES[e.out]])[t_mat])
        elif e.kind == E_DAY:
            out[j] = float(np.asarray(outputs[OUT_NAMES[e.out]])[min(max(e.t, 0), t_n - 1)])
    return out


class DaySimulator:
    """The free-run day on samples ``(theta [6], treatment)`` (module docstring).

    ``runs`` are :class:`agrijax.sites.dssat_free_run.FreeRunInputs`; ``entries[b]`` the entries of
    run ``b``. Calling the simulator with ``theta [S, 6]`` and treatment indices ``tid [S]`` returns
    ``[S, E]`` (``E`` = the most entries of any treatment; unused entries 0).

    ``devices``: the devices the batch is sharded over (default: ``jax.devices()``, as before).
    ``dtype``: the floating dtype the inputs are stored in on the devices (default: as built,
    float64); ``np.float32`` stores them rounded once, and the calls must then run with
    ``jax_enable_x64`` off (:mod:`agrijax.facade_execution` does that)."""

    def __init__(
        self,
        runs: Sequence[Any],
        entries: Sequence[Sequence[Entry]],
        *,
        pad_days: int = PAD_DAYS,
        devices: Sequence[Any] | None = None,
        dtype: Any = None,
    ) -> None:
        import jax
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as PS

        from agrijax.sites.dssat_free_run import stack_trees

        if len(runs) != len(entries):
            raise ValueError(f"{len(runs)} runs, {len(entries)} entry lists")
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
            tab = {k: np.zeros((len(idx), self.n_entries), np.int32) for k in ("type", "out", "t")}
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
    def _sim_fn(self, g: tuple[int, str]) -> Callable[..., Any]:
        """``sim(inputs, tab, theta [S, 6], tid [S]) -> [S, E]`` of group ``g`` (traceable)."""
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from agrijax.core.state import get_path
        from agrijax.models.day_dssat486 import SLOT, day_dssat486, day_processes
        from agrijax.processes.crop.ceres_maize import plantgro_outputs

        def outputs(state: Any, params: Any, forcing_t: Any) -> dict[str, Any]:
            pg = plantgro_outputs(get_path(state, f"crops.{SLOT}"), params["crop"], forcing_t["crop"])
            first = lambda x: jnp.ravel(x)[0]  # noqa: E731
            return {**{n: first(pg[n]) for n in OUT_NAMES}, "istage": first(pg["istage"])}

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

        def sim(inputs: Any, tab: Any, theta: Any, tid: Any) -> Any:
            params_tr, forcing_tr, state_tr = inputs
            params = set_cultivar(jax.tree.map(lambda x: x[tid], params_tr), theta)
            state0 = jax.tree.map(lambda x: x[tid], state_tr)

            def body(s: Any, t: Any) -> Any:
                return step(s, params, jax.tree.map(lambda x: x[tid, t], forcing_tr))

            _, outs = jax.lax.scan(body, state0, jnp.arange(n_days))
            y = jnp.stack([outs[n] for n in OUT_NAMES], axis=0)  # [O, T, S]
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
