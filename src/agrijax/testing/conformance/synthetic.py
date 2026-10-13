"""Synthetic inputs of the conformance kit for examples and plugin authors (no data needed).

The maize season of the kit's CERES-Maize cases: the MZCER048 cultivar IB0035 and the DSSAT-CSM
v4.8.6.0 species values, a deterministic NumPy weather around a warm-season mean at latitude 29.6,
and a nine-layer sandy soil. These are test inputs, not a validated site.
"""

from __future__ import annotations

from typing import Any

from agrijax.processes.crop.ceres_maize import CeresMaizeParams, CeresReplayForcing

from ._builtin import crop as _crop
from ._builtin import dssat_evap as _evap

__all__ = ["DLAYR", "DUL", "KEP", "LL", "N_SEASON", "SAT", "START", "ceres_params", "ceres_weather"]

#: layer thickness [cm], lower limit, drained upper limit and saturation [cm3 cm-3] of the soil
DLAYR: tuple[float, ...] = _crop.DLAYR
LL: tuple[float, ...] = _crop.LL
DUL: tuple[float, ...] = _crop.DUL
SAT: tuple[float, ...] = _crop.SAT
#: first day of the weather (YYYYDDD) and the length of the season [d]
START: int = _crop.START
N_SEASON: int = _crop.N_SEASON
#: CERES-Maize canopy extinction coefficient KEP [-] (SPAM's KSEVAP = KTRANS) of the DSSAT maize
#: examples: KCAN 0.85 of MZCER048.SPE through MZ_PHENOL.for:338 (DSSAT-CSM v4.8.6.0)
KEP: float = _evap.KEP


def ceres_params(dtype: Any, yrplt: int) -> CeresMaizeParams:
    """CERES-Maize params of the synthetic season (cultivar IB0035), planting date ``yrplt``."""
    return _crop.params(dtype, yrplt)


def ceres_weather(seed: int, dtype: Any, n: int) -> CeresReplayForcing:
    """``n`` days of deterministic synthetic weather from :data:`START` (NumPy generator ``seed``)."""
    return _crop.weather(seed, dtype, n)
