"""Coefficients and state of the smoothed CERES-Maize phenology (:mod:`.smoothed`).

The smoothed variant replaces every thermal-time stage threshold of ``MZ_PHENOL`` (and the
photoperiod-induction threshold ``SIND >= 1``) by a logistic of scale :attr:`SmoothingCoefficients.width`.
It is not the reference model: it is the forward-smoothing comparator of the gradient experiments,
kept beside the faithful process and recorded as a deviation
(``crop/ceres_maize.phenology@dssat-4.8.6.0:alt_smoothed``).

Kept apart from :mod:`.smoothed` so that :mod:`.state` can name these classes at import time (the
process module imports the state).

Source: this implementation; the logistic stage gate follows the forward-smoothing mode of torchcrop
(``sigmoid(k x)``, ``k = 50`` on the development stage, LINTUL-5) and the logistic chilling and forcing
requirements of van Bree et al. (2025, AAAI, arXiv:2501.16848, Eq. 4-6).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.coefficients import NO_REFERENCE, Coefficients, Provenance, coef
from agrijax.core.state import State, field

__all__ = [
    "N_PASSAGE",
    "PASSAGE_NAMES",
    "SmoothingCoefficients",
    "SoftPhenologyState",
]

#: the stage boundaries of the soft passage record, in order (index ``k`` of ``passage[:, k]``)
PASSAGE_NAMES = (
    "emergence",
    "end_juvenile",
    "tassel_initiation",
    "silking",
    "begin_effective_grain_fill",
    "end_effective_grain_fill",
    "maturity",
)
N_PASSAGE = len(PASSAGE_NAMES)

_TORCHCROP = "torchcrop smooth mode, sigmoid(k x) with k = 50 on the development stage (LINTUL-5)"
_VAN_BREE = "van Bree et al. (2025), AAAI, arXiv:2501.16848"


class SmoothingCoefficients(Coefficients):
    """The scale of the logistic stage gates of the smoothed phenology (``CeresMaizeParams.smoothing``).

    A threshold ``x >= x0`` on a thermal-time clock becomes ``1 / (1 + exp(-(x - x0) / width))``; the
    photoperiod-induction threshold ``SIND >= 1`` uses the scale ``width / sind_tt`` in SIND units.
    Neither is a crop coefficient: they set how far the smoothed model departs from the reference, so
    they are not calibrated.
    """

    width: float = coef(
        17.0,
        "degC d",
        "scale of the logistic that replaces each thermal-time stage threshold (default: torchcrop's "
        "k = 50 on a development scale of about 850 degC d per unit, emergence to silking)",
        Provenance(
            NO_REFERENCE,
            paper=_TORCHCROP,
            note="comparator choice; swept from 2 to 70 degC d (van Bree's slope 50 on the season "
            "length normalised sum is about 35 degC d for a 1700 degC d season)",
        ),
        calibrate=False,
        bounds=(1e-12, 1e3),
    )
    sind_tt: float = coef(
        70.0,
        "degC d",
        "thermal time per unit of summed photoperiod induction SIND (converts the width to SIND units)",
        Provenance(
            NO_REFERENCE,
            paper=_VAN_BREE,
            note="comparator choice: DJTI = 4 d of the maize species file times about 17.5 degC d "
            "per day during floral induction",
        ),
        calibrate=False,
        bounds=(1.0, 1e3),
    )


SMOOTHING_DEFAULT = SmoothingCoefficients()


class SoftPhenologyState(State):
    """The soft development clocks of the smoothed phenology (``CeresMaizeState.soft``).

    ``tt_germ`` is never reset within a season, ``tt_pre_ti`` / ``tt_ti`` replace the reset of
    ``SUMDTT`` at tassel initiation, ``passage`` holds the soft probability that each boundary of
    :data:`PASSAGE_NAMES` has been passed (the expected passage day is the sum of ``1 - passage``
    over the days), and ``dur4`` / ``sump`` are the soft stage-4 duration and assimilation that set
    the grain number.
    """

    tt_germ: Array = field(unit="degC d", description="thermal time since germination", dims=("n_crop",))
    sind: Array = field(
        unit="-",
        description="summed photoperiod induction, gated by the end-of-juvenile passage",
        dims=("n_crop",),
    )
    tt_pre_ti: Array = field(
        unit="degC d", description="thermal time from emergence to tassel initiation (soft)", dims=("n_crop",)
    )
    tt_ti: Array = field(
        unit="degC d", description="thermal time since tassel initiation (soft)", dims=("n_crop",)
    )
    passage: Array = field(
        unit="-",
        description="soft probability that each stage boundary (PASSAGE_NAMES) has been passed",
        dims=("n_crop", N_PASSAGE),
    )
    dur4: Array = field(
        unit="d", description="soft duration of stage 4 (silking to effective grain fill)", dims=("n_crop",)
    )
    sump: Array = field(
        unit="g plant-1", description="assimilation summed over the soft stage 4", dims=("n_crop",)
    )

    @classmethod
    def initial(cls, n_crop: int = 1, dtype: Any = None) -> SoftPhenologyState:
        """All clocks zero (the state before sowing)."""
        f = jnp.zeros((n_crop,), dtype=dtype if dtype is not None else jnp.result_type(float))
        return cls(
            tt_germ=f,
            sind=f,
            tt_pre_ti=f,
            tt_ti=f,
            passage=jnp.zeros((n_crop, N_PASSAGE), dtype=f.dtype),
            dur4=f,
            sump=f,
        )
