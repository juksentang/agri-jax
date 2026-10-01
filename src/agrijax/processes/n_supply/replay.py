"""Replay producer of the crop nitrogen port (P10): the nitrogen factors from a forcing series.

The module state holds only its output port ``n_out`` (a :class:`~agrijax.iface.crop.CropNIn`),
bound at assembly to ``iface.crop_n.<slot>``; the forcing carries the daily ``NSTRES`` of the
reference run and, optionally, its ``AGEFAC``, ``NDEF3`` and ``NPOOL`` (a series left out is the
record's no-stress default), the same for every crop of the sample. Run on its own the record lives in the
state (fill it with ``CropNIn.initial``). Reading the series from a reference run's output is
the caller's job (io), not this module's.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
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
    agefac: Array | None = field(
        unit="-",
        description="replay: nitrogen stress factor AGEFAC of the reference run (None: 1)",
        fortran_name="AGEFAC",
        dims="T",
        default=None,
    )
    ndef3: Array | None = field(
        unit="-",
        description="replay: nitrogen stress factor NDEF3 of the reference run (None: 1)",
        fortran_name="NDEF3",
        dims="T",
        default=None,
    )
    npool: Array | None = field(
        unit="g plant-1",
        description="replay: translocatable plant nitrogen NPOOL of the reference run (None: no cap)",
        fortran_name="NPOOL",
        dims="T",
        default=None,
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
    source="replay of a reference run's crop nitrogen factors through port P10 (iface.crop_n)",
    fortran_name="",
    key="n_supply/forcing_replay@none:replay",
    provenance="equations_only",
    grid="point",
    sources=(
        (
            "crop nitrogen record NSTRES, AGEFAC, NDEF3, NPOOL copied from the forcing",
            "port P10 of agrijax.iface: the factors the RZWQM2 4.6 embedded CERES uses for the crop's "
            "mass (NFAC changes only the crop's nitrogen and is not replayed)",
        ),
    ),
    deviates=(),
)
def crop_n_replay(state: CropNReplayState, params: Any, forcing_t: CropNReplayForcing) -> CropNReplayState:
    """Write the ``n_out`` port from the forcing: ``nstres``, ``agefac``, ``ndef3``, ``npool``
    [n_crop] = today's ``NSTRES``, ``AGEFAC``, ``NDEF3``, ``NPOOL``; a series the forcing leaves
    out (``None``) writes the record's no-stress default (:meth:`CropNIn.like`). Forcing read:
    ``nstres``, ``agefac``, ``ndef3``, ``npool``.

    Source: replay of the reference run's daily nitrogen factors through port P10 of ``agrijax.iface``.
    """
    f = forcing_t
    new = CropNIn.like(state.n_out, f.nstres, f.agefac, f.ndef3, f.npool)
    return eqx.tree_at(lambda s: s.n_out, state, new)
