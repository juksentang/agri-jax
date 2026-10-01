"""Parameters of the canopy record for the PET module (:mod:`.canopy`): the RZWQM2 driver's
canopy numbers (:class:`CanopyCoefficients`) and the planned harvest date of each season
(:class:`CeresCanopyParams`, ``CeresMaizeParams.canopy``).

Kept apart from :mod:`.canopy` so that :mod:`.state` can name the parameter class at import time
(the process module imports the state). RZWQM2 is cited by file and line only.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.dims import register_dim
from agrijax.core.state import Params, field

from .constants import MDATE_NONE

__all__ = [
    "DRY_DOWN_BASIS",
    "MA_2003",
    "REF_VERSION",
    "RZWQM2_CANOPY",
    "CanopyCoefficients",
    "CeresCanopyParams",
]

#: the season axis of the per-season rows (the same axis :mod:`.state` registers)
_S = ("n_season",)
register_dim("n_season", "seasons of a multi-season run (rows of a per-season parameter table)")

#: the reference of the canopy derivation (the registry key's ``@rzwqm2-4.6``)
REF_VERSION = "rzwqm2-4.6"
#: the published form of the height regression (named by the reference source)
MA_2003 = "Ma et al. (2003), Trans. ASAE 46(1) (maize height from stalk mass, D. C. Nielsen's data)"
#: the post-maturity decline has no published form known to us; the dump tables of the instrumented
#: RZWQM2 4.6 build verify it
DRY_DOWN_BASIS = (
    "RZWQM2 driver convention without a published form; verified on the 27 post-maturity days of "
    "CA-TPA 2020-2021 (dump tables of the instrumented RZWQM2 4.6 build)"
)


def _rz(file_line: str, *, paper: str = MA_2003, note: str = "") -> Provenance:
    """Provenance of an RZWQM2 4.6 driver convention: ``RZWQM/<file>:<line>`` of the reference
    source tree (the 4.5 tree the 4.6 binary is built from), routine ``DSSATDRV``, no statement.

    Source: project policy for RZWQM2 references (file, line and routine are cited, never the
    statement; CONTRIBUTING.md)."""
    return Provenance.at(REF_VERSION, f"RZWQM/{file_line}", routine="DSSATDRV", paper=paper, note=note)


class CanopyCoefficients(Coefficients):
    """The numbers of the RZWQM2 driver's canopy derivation (``CeresCanopyParams.coefficients``)."""

    htmax: float = coef(
        244.6,
        "cm",
        "maximum plant height of the stalk-mass regression (PLHGHT = cultivar HTMAX)",
        _rz(
            "DSSATDRV.for:876",
            note="maize default; the CA-TPA cultivar IB0012 of MZCER040.CUL gives the same HTMAX "
            "(read at DSSATDRV.for:858-860; the DSSATDRV entry of the instrumented build's dump tables "
            "holds PLHGHT = 244.6 as REAL*4)",
        ),
        bounds=(1.0, 1000.0),
        fortran_name="PLHGHT",
    )
    biohalf: float = coef(
        43.07,
        "g plant-1",
        "stalk mass per plant at which the height is half of htmax (cultivar BIOHALF)",
        _rz(
            "DSSATDRV.for:877",
            note="maize default; the CA-TPA cultivar IB0012 gives the same BIOHALF; PLALFA from it "
            "at DSSATDRV.for:890",
        ),
        bounds=(1.0e-3, 1.0e4),
        fortran_name="BIOHALF",
    )
    height_fraction: float = coef(
        0.5,
        "-",
        "fraction of htmax reached at the stalk mass biohalf (the definition of BIOHALF)",
        _rz("DSSATDRV.for:890", note="LOG(0.5) in PLALFA"),
        calibrate=False,
        bounds=(1.0e-6, 1.0 - 1.0e-6),
    )
    dry_down_days: float = coef(
        30.0,
        "d",
        "longest post-maturity decline of the published leaf area index to 0",
        _rz("DSSATDRV.for:1652", paper=DRY_DOWN_BASIS, note="min(30, HDATE - MDATE) days after MDATE"),
        bounds=(1.0, 366.0),
    )


RZWQM2_CANOPY = CanopyCoefficients()
"""The RZWQM2 4.6 values (what ``CeresCanopyParams.coefficients = None`` means)."""


class CeresCanopyParams(Params):
    """Parameters of the canopy entry: the planned harvest date of each season and the
    coefficients (``CeresMaizeParams.canopy``).

    ``hdate`` has one row per season (``[n_season]``, the rows of the crop's
    :class:`~.state.CeresSeasons`; one row for a single-season run) and is picked by the crop's
    ``season`` index. A season harvested at maturity, or without a planned date, gives ``-99``
    (``MDATE_NONE``): no post-maturity decline.
    """

    hdate: Array = field(
        dims=_S,
        unit="YYYYDDD",
        description="planned harvest date of each season (-99: none)",
        fortran_name="HDATE",
    )
    coefficients: CanopyCoefficients | None = field(
        description="the driver's canopy numbers (None: the RZWQM2 4.6 values)", default=None
    )

    def coef(self) -> CanopyCoefficients:
        """The coefficients in force: :attr:`coefficients`, or :data:`RZWQM2_CANOPY`."""
        return RZWQM2_CANOPY if self.coefficients is None else self.coefficients

    @classmethod
    def none(cls, n_season: int = 1) -> CeresCanopyParams:
        """No planned harvest date in any of ``n_season`` seasons (no post-maturity decline)."""
        return cls(hdate=jnp.full((n_season,), MDATE_NONE, dtype=jnp.int32))

    def hdate_of(self, season: Array | None) -> Array:
        """The planned harvest date of season ``season`` (row 0 without an index; past the last
        row, the last row's date).

        Source: this implementation: per-season parameter rows (a multi-season run carries one row per
        season, picked by the season index)."""
        t = jnp.asarray(self.hdate)
        if season is None:  # static: a single-season state carries no index
            return t[..., 0]
        k = jnp.clip(season, 0, t.shape[-1] - 1)
        return jnp.take(t, k, axis=-1)
