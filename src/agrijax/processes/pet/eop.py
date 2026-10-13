"""The crop's potential transpiration from the PET port: ``EOP = 10 PET`` (coupling contract, day entry 9).

RZWQM2 with an embedded DSSAT crop and ``ISTRESS = 0`` hands the crop the Shuttleworth-Wallace
potential transpiration of the day as DSSAT's ``EOP``: the same amount, converted from cm d-1
(``PET``) to mm d-1 (``EOP``). The process reads P5 (``iface.pet.transpiration``, written earlier
the same day by the PET entry) and writes P1 ``eop`` (``iface.crop_water.<slot>.eop``, read the
same day by the crop's stress and by ROOTWU). Every crop of the slot gets the field's value
(one canopy source per field; splitting the PET between several crops is not defined yet).

Module state (:class:`EOPState`): no fields of its own, two ports::

    pet         PETFluxes    P5  iface.pet                 read (transpiration)
    crop_water  CropWaterIn  P1  iface.crop_water.<slot>   written (eop only)
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import State
from agrijax.core.units import MM_PER_CM
from agrijax.iface.crop import CropWaterIn
from agrijax.iface.surface import PETFluxes

__all__ = ["EOPState", "eop_from_pet"]


class EOPState(State):
    """State of the ``EOP`` adapter: two ports and nothing of its own (see the module docstring)."""

    pet: PETFluxes = port(description="today's potential fluxes (P5), written earlier by the PET entry")
    crop_water: CropWaterIn = port(description="the crop's water record (P1); this module writes eop")

    @classmethod
    def module(cls, pet: PETFluxes, crop_water: CropWaterIn) -> EOPState:
        """The module state with its ports filled."""
        return cls(pet=pet, crop_water=crop_water)


@process(
    reads=("pet.transpiration",),
    writes=("crop_water.eop",),
    source="RZWQM2 4.6 DSSATDRV (ISTRESS = 0): the embedded crop's EOP is the S-W potential transpiration",
    fortran_name="EOP",
    key="crop_iface/eop_from_pet@rzwqm2-4.6:faithful",
    provenance="reference_only_conventions",
    grid="point",
    ref_build="RZWQM2 4.6 main_ryzen5_avx512",
    sources=(
        (
            "EOP [mm d-1] = PET [cm d-1] x 10 with ISTRESS = 0",
            "RZWQM2 4.5 DSSATDRV.for lines 630-634, 743-762 (ISTRESS = 0: EOP = 10 PET); the DSSATDRV "
            "exit EOP of the CA-TPA 2015-2023 dumps against the PHYSCL exit PET",
        ),
    ),
    deviates=(
        (
            "only the ISTRESS = 0 path (DSSAT water stress from ROOTWU) is implemented",
            "the configuration of the validated RZWQM2 runs (CA-TPA and the 9 daily S-W scenarios)",
            "eop.py module docstring",
        ),
        (
            "every crop of the slot gets the field's potential transpiration",
            "one canopy source per field; the split between crops is not defined",
            "eop.py module docstring",
        ),
    ),
)
def eop_from_pet(state: EOPState, params: Any, forcing_t: Any) -> EOPState:
    """The crop's potential transpiration ``EOP`` [mm d-1] from today's PET port [cm d-1].

    Reads ``pet.transpiration`` (P5), writes ``crop_water.eop`` (P1) for every crop of the slot,
    in the dtype of the record. Uses no parameters and no forcing.

    Source: RZWQM2 4.5 DSSATDRV.for lines 630-634 and 743-762 (ISTRESS = 0, EOP = PET in mm d-1);
    coupling contract (:mod:`agrijax.iface.contract`), day table entry 9.
    """
    old = state.crop_water.eop
    t = jnp.asarray(state.pet.transpiration) * MM_PER_CM
    eop = jnp.broadcast_to(t[..., None], old.shape).astype(old.dtype)
    return eqx.tree_at(lambda s: s.crop_water.eop, state, eop)
