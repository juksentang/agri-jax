"""The CERES-Maize day as an :class:`agrijax.core.Model`, its two port processes, and the daily
outputs of ``PlantGro.OUT``.

Order within a day follows ``MZ_CERES`` (``DYNAMIC = INTEGR``): phenology, the stress block and
growth of ``MZ_GROSUB``, then roots; then :func:`ceres_publish` writes the root record the uptake
producers read on the next call (DSSAT PLANT outputs ``RLV, RWUMX, PORMIN, XHLAI`` for SPAM).
The crop reads its soil water, ``EOP`` and ``TRWUP`` from its ``water_in`` port and computes its
water-stress factors itself (plan 19 A3). Run on its own (:func:`ceres_maize_model`), the port is
written first by :func:`ceres_water_replay` from the forcing, so the crop can be driven by any
soil-water model or, for validation, by the reference run itself; in a coupled assembly the same
processes are bound to ``iface.*`` paths and the port is written by the producers.

Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_CERES.for and MZ_OPGROW.for, CSM_Main/LAND.for
(BSD-3).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.model import Model
from agrijax.core.process import process
from agrijax.core.units import CM_PER_M, KG_HA_PER_G_M2
from agrijax.processes.soil_water.uptake import CropWaterIn, RootRecord

from .growth import ceres_growth, ceres_stress
from .phenology import ceres_phenology
from .roots import ceres_roots
from .state import CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = [
    "CROP_PROCESSES",
    "OUTPUT_UNITS",
    "ceres_maize_model",
    "ceres_publish",
    "ceres_water_replay",
    "plantgro_outputs",
    "yield_kg_ha",
]

# the unit conversions of MZ_OPGROW, from agrijax.core.units (same values, same operations)
_KG_HA = KG_HA_PER_G_M2  # g m-2 -> kg ha-1 (MZ_OPGROW: NINT(WTLF*10.), ...)
_CM_TO_M = CM_PER_M  # cm -> m (MZ_OPGROW: RDPD = RTDEP/100.)

OUTPUT_UNITS = {
    "istage": "-",
    "gstd": "-",
    "lsd": "-",
    "lai": "m2 m-2",
    "lwad": "kg ha-1",
    "swad": "kg ha-1",
    "gwad": "kg ha-1",
    "rwad": "kg ha-1",
    "cwad": "kg ha-1",
    "g_ad": "m-2",
    "pwad": "kg ha-1",
    "wspd": "-",
    "wsgd": "-",
    "ewsd": "-",
    "rdpd": "m",
    "dttd": "degC d",
    "rlv": "cm cm-3",
}


def plantgro_outputs(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> dict[str, Array]:
    """The daily ``PlantGro.OUT`` quantities (``MZ_OPGROW``) of every crop, unrounded.

    ``lai`` = LAID, ``lwad`` / ``swad`` / ``gwad`` / ``rwad`` / ``cwad`` = LWAD, SWAD, GWAD, RWAD,
    CWAD [kg ha-1], ``gstd`` = GSTD (``RSTAGE``), ``lsd`` = L#SD (``VSTAGE``), ``g_ad`` = G#AD,
    ``pwad`` = PWAD, ``wspd`` / ``wsgd`` / ``ewsd`` = 1 - SWFAC, 1 - TURFAC, SATFAC, ``rdpd`` =
    RDPD [m], ``dttd`` = DTTD and ``rlv`` = RL1D.. [cm cm-3].
    """
    g = state.growth
    ph = state.phen
    pop = g.pltpop
    lfwt = g.leaf.mass[..., 0]
    return {
        "istage": ph.istage,
        "gstd": g.rstage,
        "lsd": g.leafno,
        "lai": g.lai,
        "lwad": lfwt * pop * _KG_HA,
        "swad": g.stmwt * pop * _KG_HA,
        "gwad": g.grnwt * ph.ears * _KG_HA,
        "rwad": g.rtwt * pop * _KG_HA,
        "cwad": g.biomas * _KG_HA,
        "g_ad": ph.gpp * ph.ears,
        "pwad": g.earwt * ph.ears * _KG_HA,
        "wspd": 1.0 - state.stress.swfac,
        "wsgd": 1.0 - state.stress.turfac,
        "ewsd": state.stress.satfac,
        "rdpd": state.roots.rtdep / _CM_TO_M,
        "dttd": ph.dtt,
        "rlv": state.roots.rlv,
        "sumdtt": ph.sumdtt,
    }


@process(
    reads=(),
    writes=("water_in",),
    source="replay of a reference run's crop water drivers (plan 19 A2, A3)",
    fortran_name="",
    key="water_supply/forcing_replay@none:replay",
    provenance="equations_only",
    grid="dssat_layers",
    sources=(
        (
            "crop water record SW, EOP, TRWUP copied from the forcing",
            "plan 19 A3: DSSAT SPAM EOP / TRWUP dumps and SoilWat.OUT SW of the reference run",
        ),
    ),
    deviates=(),
)
def ceres_water_replay(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """Write the crop's ``water_in`` port from the forcing (the replay binding of plan 19 A2).

    ``sw`` [n_layer] is today's soil water after the soil update, ``eop`` and ``trwup`` today's
    potential transpiration and root water uptake, the same for every crop of the sample.

    Source: plan 19 A3 (replay of the reference run's SPAM EOP / TRWUP and soil water).
    """
    w = state.water_in
    dt = w.trwup.dtype
    rec = CropWaterIn(
        sw=jnp.asarray(forcing_t.sw, dtype=w.sw.dtype) * jnp.ones_like(w.sw),
        eop=jnp.asarray(forcing_t.eop, dtype=dt) * jnp.ones_like(w.eop),
        trwup=jnp.asarray(forcing_t.trwup, dtype=dt) * jnp.ones_like(w.trwup),
    )
    return eqx.tree_at(lambda x: x.water_in, state, rec)


@process(
    reads=("roots", "growth.lai"),
    writes=("root_out",),
    source="DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_CERES.for, MZ_GROSUB.for; CSM_Main/LAND.for (BSD-3)",
    fortran_name="MZ_CERES",
    key="crop/ceres_maize.publish@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="dssat_layers",
    ref_build="dscsm048 v4.8.6.0 (build486)",
    sources=(
        ("PLANT outputs RLV, RWUMX, PORMIN, XHLAI read by SPAM", "CSM_Main/LAND.for, PLANT / SPAM arguments"),
        ("XHLAI = LAI", "MZ_GROSUB.for, INTEGR totals ('Used in WATBAL')"),
        ("RWUMX, PORMIN from the species file *ROOT section", "MZ_GROSUB.for, SEASINIT reads"),
    ),
    deviates=(),
)
def ceres_publish(
    state: CeresMaizeState, params: CeresMaizeParams, forcing_t: CeresForcing
) -> CeresMaizeState:
    """Publish the root record of the day (plan 19 A4): today's ``RLV`` and ``RTDEP`` after root
    growth, ``XHLAI = LAI`` after growth, and the species ``RWUMX``, ``PORMIN`` as state.

    Source: DSSAT-CSM v4.8.6.0 MZ_CERES.for outputs and CSM_Main/LAND.for (BSD-3).
    """
    lai = state.growth.lai
    one = jnp.ones_like(lai)
    rec = RootRecord(
        rlv=state.roots.rlv,
        rtdep=state.roots.rtdep,
        rwumx=jnp.asarray(params.species.rwumx, dtype=lai.dtype) * one,
        pormin=jnp.asarray(params.species.pormin, dtype=lai.dtype) * one,
        xhlai=lai,
    )
    return eqx.tree_at(lambda x: x.root_out, state, rec)


#: the crop day, in the ``MZ_CERES`` order, without the replay producer (bind these in an assembly)
CROP_PROCESSES = (ceres_phenology, ceres_stress, ceres_growth, ceres_roots, ceres_publish)


def ceres_maize_model(*, outputs: Any = plantgro_outputs) -> Model:
    """CERES-Maize on its own as a :class:`~agrijax.core.Model`: :func:`ceres_water_replay`
    (the ``water_in`` port from the forcing), then phenology, stress, growth, roots and
    :func:`ceres_publish`."""
    return Model(
        CeresMaizeState,
        [ceres_water_replay, *CROP_PROCESSES],
        outputs=outputs,
        name="ceres_maize",
    )


def yield_kg_ha(state: CeresMaizeState) -> Array:
    """Grain yield ``YIELD = GRNWT x 10 x EARS`` [kg ha-1] (``MZ_GROSUB`` OUTPUT, Summary HWAM)."""
    return state.growth.grnwt * _KG_HA * state.phen.ears
