"""Replay producer of the crop nitrogen port (P10): ``NSTRES`` from a forcing series.

The module state holds only its output port ``n_out`` (a :class:`~agrijax.iface.crop.CropNIn`),
bound at assembly to ``iface.crop_n.<slot>``; the forcing carries the daily ``NSTRES`` of the
reference run, the same for every crop of the sample. Run on its own the record lives in the
state (fill it with ``CropNIn.initial``). Reading the series from a reference run's output is
the caller's job (io), not this module's.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Forcing, State, field
from agrijax.iface.crop import CropNIn

__all__ = ["CropNReplayForcing", "CropNReplayState", "crop_n_replay"]


class CropNReplayForcing(Forcing):
    """The daily nitrogen stress to replay (time axis first)."""

    nstres: Array = field(
        unit="-",
        description="replay: nitrogen stress factor NSTRES of the reference run",
        fortran_name="NSTRES",
        dims="T",
    )


class CropNReplayState(State):
    """State of the replay producer: only its output port."""

    n_out: CropNIn = port(description="crop nitrogen record written each day (P10)")

    @classmethod
    def initial(cls, n_crop: int, dtype: Any = None) -> CropNReplayState:
        """The producer with its record filled (``NSTRES = 1``), for a run without binding."""
        return cls(n_out=CropNIn.initial(n_crop, dtype))


@process(
    reads=(),
    writes=("n_out",),
    source="replay of a reference run's crop nitrogen stress (M3 coupling contract, port P10)",
    fortran_name="",
    key="n_supply/forcing_replay@none:replay",
    provenance="equations_only",
    grid="point",
    sources=(
        (
            "crop nitrogen record NSTRES copied from the forcing",
            "M3 coupling contract, port P10 and decision 2",
        ),
    ),
    deviates=(),
)
def crop_n_replay(state: CropNReplayState, params: Any, forcing_t: CropNReplayForcing) -> CropNReplayState:
    """Write the ``n_out`` port from the forcing: ``nstres`` [n_crop] = today's ``NSTRES``.

    Source: M3 coupling contract, port P10 (replay of the reference run's daily NSTRES).
    """
    rec = state.n_out
    new = CropNIn(nstres=jnp.asarray(forcing_t.nstres, dtype=rec.nstres.dtype) * jnp.ones_like(rec.nstres))
    return eqx.tree_at(lambda s: s.n_out, state, new)
