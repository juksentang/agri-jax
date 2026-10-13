"""Soil water: implicit mixed-form Richards redistribution (modified Brooks-Corey curves) and the
DSSAT-CSM v4.8.6.0 tipping-bucket water balance (:mod:`.bucket`, :mod:`.bucket_evap`).

The Richards redistribution is two layers: the problem (:mod:`.problem`, the physics, one version) and
its time integrators (:mod:`.integrator`: protocol and registry; :mod:`.fixed_cn` ``FixedStepping``,
:mod:`.richards_adaptive` ``AdaptiveStepping``), chosen by ``RichardsParams.stepping``."""

from .fixed_cn import FixedCN, FixedStepping
from .integrator import INTEGRATORS, Capabilities, RichardsIntegrator, integrator_for, register_integrator
from .problem import PROBLEM_VERSION, RichardsProblem
from .richards import (
    RichardsConfig,
    RichardsForcing,
    RichardsGrid,
    RichardsParams,
    RichardsState,
    SoilWater,
    SoilWaterFluxes,
    richards_day,
    richards_redistribution,
    richards_step,
)
from .richards_adaptive import AdaptiveCN, AdaptiveStepping
from .sinks import SINK_CHANNELS, SINK_LEDGER_OUTFLOWS, SinkChannel, SinkChannels, as_sink_channels

__all__ = [
    "INTEGRATORS",
    "PROBLEM_VERSION",
    "SINK_CHANNELS",
    "SINK_LEDGER_OUTFLOWS",
    "AdaptiveCN",
    "AdaptiveStepping",
    "Capabilities",
    "FixedCN",
    "FixedStepping",
    "RichardsConfig",
    "RichardsForcing",
    "RichardsGrid",
    "RichardsIntegrator",
    "RichardsParams",
    "RichardsProblem",
    "RichardsState",
    "SinkChannel",
    "SinkChannels",
    "SoilWater",
    "SoilWaterFluxes",
    "as_sink_channels",
    "integrator_for",
    "register_integrator",
    "richards_day",
    "richards_redistribution",
    "richards_step",
]
