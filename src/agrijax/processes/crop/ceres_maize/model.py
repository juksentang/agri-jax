"""The CERES-Maize day as an :class:`agrijax.core.Model`, and the daily outputs of ``PlantGro.OUT``.

Order within a day follows ``MZ_CERES`` (``DYNAMIC = INTEGR``): phenology, the stress block and
growth of ``MZ_GROSUB``, then roots. Soil water and the water-stress factors are forcing, so the
crop can be driven by any soil-water model or, for validation, by the reference run itself.

Source: DSSAT-CSM v4.8.6.0 Plant/CERES-Maize/MZ_CERES.for and MZ_OPGROW.for (BSD-3).
"""

from __future__ import annotations

from typing import Any

from jaxtyping import Array

from agrijax.core.model import Model

from .growth import ceres_growth, ceres_stress
from .phenology import ceres_phenology
from .roots import ceres_roots
from .state import CeresForcing, CeresMaizeParams, CeresMaizeState

__all__ = ["OUTPUT_UNITS", "ceres_maize_model", "plantgro_outputs", "yield_kg_ha"]

_KG_HA = 10.0  # g m-2 -> kg ha-1 (MZ_OPGROW: NINT(WTLF*10.), ...)
_CM_TO_M = 100.0  # cm -> m (MZ_OPGROW: RDPD = RTDEP/100.)

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


def ceres_maize_model(*, outputs: Any = plantgro_outputs) -> Model:
    """CERES-Maize as a :class:`~agrijax.core.Model` (phenology, stress, growth, roots)."""
    return Model(
        CeresMaizeState,
        [ceres_phenology, ceres_stress, ceres_growth, ceres_roots],
        outputs=outputs,
        name="ceres_maize",
    )


def yield_kg_ha(state: CeresMaizeState) -> Array:
    """Grain yield ``YIELD = GRNWT x 10 x EARS`` [kg ha-1] (``MZ_GROSUB`` OUTPUT, Summary HWAM)."""
    return state.growth.grnwt * _KG_HA * state.phen.ears
