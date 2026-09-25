"""Water ledger: cumulative inputs and outputs as state, and a daily closure residual (plan 19 A10).

``ledger.water`` is an ordinary :class:`~agrijax.core.state.State` (:class:`WaterLedger`) that a
framework entry, built by :func:`water_ledger` and placed last in the day (the ``ledger`` phase),
updates every day:

* the cumulative total of every inflow and outflow **channel** (rain, irrigation, infiltration,
  runoff, soil evaporation, transpiration per crop slot, drainage, ...), each a scalar or one
  value per crop slot (:class:`LedgerChannel`);
* the storage at the end of the day;
* the daily residual ``residual = dW - (in - out)``, with ``dW`` the change of storage over the
  day and ``in``/``out`` the day's channel totals, and its running maximum.

Accumulators are float64 whenever JAX runs with x64 enabled, whatever the dtype of the model
state. In a float32 run (x64 disabled) every cumulative total carries a Neumaier compensation
term, so the long-run totals keep close to float64 accuracy (review R-14).

Under ``AGRI_JAX_CHECK=1`` the ledger entry asserts per-day closure, ``|residual| <= atol +
rtol * scale`` (``scale`` = storage plus the day's gross flows), with :func:`equinox.error_if`, so
the check also runs inside ``jit``, ``scan`` and ``vmap``. The check is decided at trace time.

The generic ``@process(balances=...)`` form, and other conserved quantities (N, C, energy), wait
for the first module that needs them (plan 19 L-ledger).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.process import Process, check_enabled, process
from agrijax.core.state import State, field, get_path, set_path

__all__ = [
    "Channel",
    "LedgerChannel",
    "WaterLedger",
    "ledger_dtype",
    "water_ledger",
]

#: a channel's daily amount: a dotted state path, or ``f(state, params, forcing_t) -> Array`` [cm d-1]
Channel = str | Callable[[Any, Any, Any], Any]


def ledger_dtype() -> Any:
    """``float64`` when JAX has x64 enabled, else ``float32`` (then with compensated sums)."""
    return jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32


def _neumaier(total: Array, comp: Array, x: Array) -> tuple[Array, Array]:
    """One step of Neumaier's compensated summation (both branches finite)."""
    t = total + x
    big = jnp.abs(total) >= jnp.abs(x)
    comp = comp + jnp.where(big, (total - t) + x, (x - t) + total)
    return t, comp


class LedgerChannel(State):
    """Cumulative amount of one channel and its Neumaier compensation term.

    A channel is a scalar or carries one value per crop slot (transpiration per slot).
    """

    total: Array = field(dims=("n_crop?",), unit="cm", description="cumulative amount")
    comp: Array = field(dims=("n_crop?",), unit="cm", description="compensation term (add to total)")

    @property
    def value(self) -> Array:
        """Compensated cumulative amount ``total + comp``."""
        return self.total + self.comp


class WaterLedger(State):
    """Cumulative water inputs and outputs [cm] and the daily closure residual [cm]."""

    inflow: dict[str, LedgerChannel]
    outflow: dict[str, LedgerChannel]
    storage0: Array = field(dims=(), unit="cm", description="storage at the start of the run")
    storage: Array = field(dims=(), unit="cm", description="storage at the end of the last closed day")
    residual: Array = field(dims=(), unit="cm", description="last day's dW - (in - out)")
    max_abs_residual: Array = field(dims=(), unit="cm", description="max |residual| over the closed days")

    @classmethod
    def init(
        cls,
        storage0: Any,
        *,
        inflows: Mapping[str, tuple[int, ...]] | Iterable[str],
        outflows: Mapping[str, tuple[int, ...]] | Iterable[str],
    ) -> WaterLedger:
        """A zero ledger with the given channels (``{name: shape}``, or names for scalars)."""
        dt = ledger_dtype()

        def zeros(ch: Mapping[str, tuple[int, ...]] | Iterable[str]) -> dict[str, LedgerChannel]:
            shapes = dict(ch) if isinstance(ch, Mapping) else {str(n): () for n in ch}
            for k, v in shapes.items():
                if len(tuple(v)) > 1:
                    raise ValueError(f"channel {k!r}: a scalar or one value per crop slot, got shape {v}")
            return {
                k: LedgerChannel(total=jnp.zeros(tuple(v), dt), comp=jnp.zeros(tuple(v), dt))
                for k, v in shapes.items()
            }

        s0 = jnp.asarray(storage0, dt)
        z = jnp.zeros((), dt)
        return cls(
            inflow=zeros(inflows),
            outflow=zeros(outflows),
            storage0=s0,
            storage=s0,
            residual=z,
            max_abs_residual=z,
        )

    def total_in(self) -> dict[str, Array]:
        """Compensated cumulative inflow per channel."""
        return {k: v.value for k, v in self.inflow.items()}

    def total_out(self) -> dict[str, Array]:
        """Compensated cumulative outflow per channel."""
        return {k: v.value for k, v in self.outflow.items()}

    def closure(self) -> Array:
        """Whole-run residual ``storage - storage0 - (sum in - sum out)`` [cm]."""
        tin = sum((jnp.sum(v) for v in self.total_in().values()), jnp.zeros((), self.storage.dtype))
        tout = sum((jnp.sum(v) for v in self.total_out().values()), jnp.zeros((), self.storage.dtype))
        return (self.storage - self.storage0) - (tin - tout)


def _channel_reads(ch: Mapping[str, Channel]) -> list[str]:
    return [v for v in ch.values() if isinstance(v, str)]


def _value(ch: Channel, state: Any, params: Any, forcing_t: Any) -> Any:
    return get_path(state, ch) if isinstance(ch, str) else ch(state, params, forcing_t)


def water_ledger(
    *,
    storage: Channel,
    inflows: Mapping[str, Channel],
    outflows: Mapping[str, Channel],
    reads: Iterable[str] = (),
    at: str = "ledger.water",
    atol: float | None = None,
    rtol: float | None = None,
    name: str = "ledger.close",
) -> Process:
    """The daily ledger entry: accumulate every channel, compute the residual, check closure.

    Parameters
    ----------
    storage:
        End-of-day storage [cm] (soil profile plus pond, ...): a state path or a callable.
    inflows, outflows:
        ``{channel name: daily amount}`` [cm d-1], each a state path or a callable
        ``(state, params, forcing_t) -> Array``. The names must match the ledger's channels.
    reads:
        State paths read by the callables (string channels are added automatically).
    at:
        Path of the :class:`WaterLedger` in the state.
    atol, rtol:
        Closure tolerance under ``AGRI_JAX_CHECK=1``: ``|residual| <= atol + rtol * scale``.
        Defaults: ``1e-10`` cm and ``1e-12`` in float64, ``1e-5`` cm and ``1e-6`` in float32.
        Assemblies whose numerics do not close to rounding (a fixed, small iteration count) set
        their own tolerance and report it.
    name:
        Entry name (default ``"ledger.close"``).
    """
    inflows = dict(inflows)
    outflows = dict(outflows)
    overlap = sorted(set(inflows) & set(outflows))
    if overlap:
        raise ValueError(f"channels both in and out: {overlap}")
    all_reads = (*_channel_reads(inflows), *_channel_reads(outflows), *reads, at)
    if isinstance(storage, str):
        all_reads = (storage, *all_reads)

    def close(state: Any, params: Any, forcing_t: Any) -> Any:
        """Accumulate the day's channels into the ledger and compute the closure residual.

        Source: plan 19 A10 (water balance dW = in - out; Neumaier 1974 compensated summation).
        """
        led: WaterLedger = get_path(state, at)
        dt = led.storage.dtype
        if set(led.inflow) != set(inflows) or set(led.outflow) != set(outflows):
            raise ValueError(
                f"ledger channels in={sorted(led.inflow)} out={sorted(led.outflow)} differ from the entry's "
                f"in={sorted(inflows)} out={sorted(outflows)}"
            )
        s_new = jnp.asarray(_value(storage, state, params, forcing_t), dt)
        d_in = {k: jnp.asarray(_value(v, state, params, forcing_t), dt) for k, v in inflows.items()}
        d_out = {k: jnp.asarray(_value(v, state, params, forcing_t), dt) for k, v in outflows.items()}
        zero = jnp.zeros((), dt)
        day_in = sum((jnp.sum(v) for v in d_in.values()), zero)
        day_out = sum((jnp.sum(v) for v in d_out.values()), zero)
        residual = (s_new - led.storage) - (day_in - day_out)
        if check_enabled():
            eps64 = dt == jnp.float64
            a = (1e-10 if eps64 else 1e-5) if atol is None else atol
            r = (1e-12 if eps64 else 1e-6) if rtol is None else rtol
            gross = sum((jnp.sum(jnp.abs(v)) for v in (*d_in.values(), *d_out.values())), zero)
            scale = jnp.abs(s_new) + gross
            residual = eqx.error_if(
                residual,
                ~(jnp.abs(residual) <= a + r * scale),
                f"water ledger at {at!r} does not close: |dW - (in - out)| > {a} + {r} * scale",
            )

        def add(ch: LedgerChannel, x: Array) -> LedgerChannel:
            total, comp = _neumaier(ch.total, ch.comp, x)
            return LedgerChannel(total=total, comp=comp)

        new = led.replace(
            inflow={k: add(led.inflow[k], x) for k, x in d_in.items()},  # static, over the channels
            outflow={k: add(led.outflow[k], x) for k, x in d_out.items()},
            storage=s_new,
            residual=residual,
            max_abs_residual=jnp.maximum(led.max_abs_residual, jnp.abs(residual)),
        )
        return set_path(state, at, new)

    return process(
        close,
        reads=tuple(dict.fromkeys(all_reads)),
        writes=(at,),
        name=name,
        register=False,
        source="plan 19 A10 water ledger",
    )
