"""The process module of docs/tutorial_new_process.md (steps 2 to 5): a new leaf-area expansion formula.

This is the file you would move into ``src/agrijax/processes/<slot>/`` to merge a process, so it
holds only declarations and the process itself. The NumPy prototype, the synthetic season and the
conformance case live in ``new_process_tutorial.py``: the conformance lint checks every function of
this file as numerical kernel code, where a bare literal of a test input would be a finding.

The formula (illustrative coefficients, not a calibrated crop)::

    expansion, tt < tt_senesce:   dLAI = r_expand * dtt * LAI * (1 - LAI / lai_max) * swfac
    senescence, tt >= tt_senesce: dLAI = -k_senesce * LAI
    LAI(t) = max(LAI(t-1) + dLAI, 0)
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef, numerical_guard
from agrijax.core.process import process
from agrijax.core.state import Forcing, Params, State, field

__all__ = [
    "DemoParams",
    "LaiCoefficients",
    "LaiForcing",
    "LaiState",
    "ThermalTimeCoefficients",
    "degree_days",
    "lai_logistic",
]

#: registry key ``slot/impl@ref_version:variant``: ref_version ``none`` = no reference model
KEY = "crop/lai_logistic@none:demo"
_PAPER = "Agri-JAX tutorial (illustrative value, not from a publication)"
_TINY = numerical_guard(
    "lai_logistic.tiny", 1e-12, "floor of the LAI ceiling in a division (finite gradients)"
)


# ---- step 4: coefficients, each declared once with unit, meaning and provenance -------------------
class ThermalTimeCoefficients(Coefficients):
    """Coefficient of the degree-day sum that drives the expansion."""

    tbase: float = coef(
        8.0,
        "degC",
        "base temperature: no thermal time accumulates below it",
        Provenance("none", paper=_PAPER),
        bounds=(0.0, 15.0),
    )


class LaiCoefficients(Coefficients):
    """Coefficients of the logistic expansion and the senescence."""

    r_expand: float = coef(
        0.008,
        "degC-1 d-1",
        "relative LAI expansion rate per unit of thermal time",
        Provenance("none", paper=_PAPER),
        bounds=(0.0, 0.05),
    )
    lai_max: float = coef(
        5.0, "m2 m-2", "LAI at which the expansion stops (logistic ceiling)", Provenance("none", paper=_PAPER)
    )
    tt_senesce: float = coef(
        1200.0,
        "degC d",
        "thermal time after which the canopy senesces",
        Provenance("none", paper=_PAPER, note="a hard switch: its gradient is zero, see the tutorial"),
    )
    k_senesce: float = coef(
        0.03, "d-1", "fraction of the LAI lost per day in senescence", Provenance("none", paper=_PAPER)
    )


# ---- step 3: the state, parameters and forcing the processes read and write ----------------------
class LaiState(State):
    tt: Array = field(unit="degC d", description="thermal time since emergence", dims=("n_crop",))
    dtt: Array = field(unit="degC d", description="thermal time of today", dims=("n_crop",))
    lai: Array = field(unit="m2 m-2", description="green leaf area index", dims=("n_crop",))
    dlai: Array = field(unit="m2 m-2 d-1", description="change of the LAI today", dims=("n_crop",))


class DemoParams(Params):
    thermal: ThermalTimeCoefficients = field(description="degree-day coefficients")
    lai: LaiCoefficients = field(description="LAI expansion coefficients")


class LaiForcing(Forcing):
    tmean: Array = field(unit="degC", description="daily mean air temperature", dims="T")
    swfac: Array = field(unit="-", description="water stress factor (1 none, 0 full stress)", dims="T")


# ---- steps 2, 3 and 5: the processes --------------------------------------------------------------
@process(reads=("tt",), writes=("tt", "dtt"), register=False)
def degree_days(state: LaiState, params: DemoParams, forcing_t: LaiForcing) -> LaiState:
    """Growing degree days: ``dtt = max(tmean - tbase, 0)``, ``tt += dtt``.

    Source: the textbook degree-day sum (scaffolding of the tutorial, no reference model).
    """
    dtt = jnp.broadcast_to(jnp.maximum(forcing_t.tmean - params.thermal.tbase, 0.0), state.tt.shape)
    return state.replace(dtt=dtt, tt=state.tt + dtt)


# step 3: reads and writes; step 5: the registry key and metadata (key, provenance, sources, grid, deviates)
@process(
    reads=("tt", "dtt", "lai"),
    writes=("lai", "dlai"),
    key=KEY,
    provenance="equations_only",  # the nearest class: no reference code was used (see the tutorial)
    sources=[("logistic expansion in thermal time, exponential senescence", "tutorial demo formula")],
    grid="point",
    deviates=(),
)
def lai_logistic(state: LaiState, params: DemoParams, forcing_t: LaiForcing) -> LaiState:
    """Logistic LAI expansion in thermal time, scaled by water stress, then senescence.

    ``dLAI = r dtt LAI (1 - LAI / lai_max) swfac`` while ``tt < tt_senesce``, else ``-k LAI``;
    ``LAI = max(LAI + dLAI, 0)``. Reads ``tt``, ``dtt``, ``lai`` and the forcing ``swfac``.

    Source: tutorial demo formula (docs/tutorial_new_process.md), no reference model.
    """
    c = params.lai
    ceiling = jnp.maximum(c.lai_max, _TINY)  # both branches below stay finite
    grow = c.r_expand * state.dtt * state.lai * (1.0 - state.lai / ceiling) * forcing_t.swfac
    lose = -c.k_senesce * state.lai
    lai = jnp.maximum(state.lai + jnp.where(state.tt < c.tt_senesce, grow, lose), 0.0)
    return state.replace(lai=lai, dlai=lai - state.lai)
