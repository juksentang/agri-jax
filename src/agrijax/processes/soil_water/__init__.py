"""Soil water: RZWQM-style implicit Richards (Brooks-Corey), Green-Ampt infiltration events, the day
and its RZWQM2 convention variants (DRAIN cap, flux-mode evaporation limit).

The Richards redistribution is two layers: the problem (:mod:`.problem`, the physics, one version) and
its time integrators (:mod:`.integrator`: protocol and registry; :mod:`.fixed_cn` ``FixedStepping``,
:mod:`.richards_adaptive` ``AdaptiveStepping``), chosen by ``RichardsParams.stepping``."""

from .day import (
    SOIL_WATER_LEDGER_INFLOWS,
    SOIL_WATER_LEDGER_OUTFLOWS,
    DayConfig,
    SoilWaterDayForcing,
    SoilWaterDayParams,
    infiltration_ga,
    soil_water_day,
    soil_water_day_drain_cap,
    soil_water_day_flux_evap,
    soil_water_day_kernel,
    soil_water_day_replay,
    soil_water_day_replay_conventions,
    soil_water_day_rzwqm2_conventions,
    soil_water_ledger,
    soil_water_ledger_init,
    with_conventions,
)
from .fixed_cn import FixedCN, FixedStepping
from .infiltration import GreenAmptConfig, GreenAmptParams, StormForcing, green_ampt_event
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
    "SOIL_WATER_LEDGER_INFLOWS",
    "SOIL_WATER_LEDGER_OUTFLOWS",
    "AdaptiveCN",
    "AdaptiveStepping",
    "Capabilities",
    "DayConfig",
    "FixedCN",
    "FixedStepping",
    "GreenAmptConfig",
    "GreenAmptParams",
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
    "SoilWaterDayForcing",
    "SoilWaterDayParams",
    "SoilWaterFluxes",
    "StormForcing",
    "as_sink_channels",
    "green_ampt_event",
    "infiltration_ga",
    "integrator_for",
    "register_integrator",
    "richards_day",
    "richards_redistribution",
    "richards_step",
    "soil_water_day",
    "soil_water_day_drain_cap",
    "soil_water_day_flux_evap",
    "soil_water_day_kernel",
    "soil_water_day_replay",
    "soil_water_day_replay_conventions",
    "soil_water_day_rzwqm2_conventions",
    "soil_water_ledger",
    "soil_water_ledger_init",
    "with_conventions",
]
