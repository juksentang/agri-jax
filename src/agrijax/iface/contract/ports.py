"""The port table of the coupling contract, :data:`PORTS` (P1-P11 of the RZWQM2 4.6 day and the
planned ports P12 and up).

Part of :mod:`agrijax.iface.contract` (its docstring explains the table); import the names from
there.
"""

from __future__ import annotations

from agrijax.core.ledger import WaterLedger

from ..crop import (
    CanopyRecord,
    CropNIn,
    CropNUptake,
    CropResidueOut,
    CropSoilNIn,
    CropStatus,
    CropWaterIn,
    RootRecord,
)
from ..soil import (
    DrainControl,
    NodeNUptake,
    NodeUptake,
    SinkInputs,
    SoilFluxRecord,
    SoilIce,
    SoilMineralN,
    WaterStepTrace,
)
from ..surface import (
    DailyWeather,
    HourlyWeather,
    PETFluxes,
    ResidueRecord,
    SedimentOut,
    SnowOut,
    SoilTemperature,
)
from .spec import AllowedLag, PortSpec, _f

#: the sink channels of the soil-water day, in :data:`agrijax.iface.soil.SINK_CHANNELS` order
_SINKS: tuple[str, ...] = ("uptake", "tile", "lateral", "subirrigation", "macropore_to_drain")

_RZ_ORDER = "RZWQM2 4.6 daily order (PHYSCL before the embedded crop), verified on the reference binary"

#: the port table of the coupling contract
PORTS: dict[str, PortSpec] = {
    p.id: p
    for p in (
        PortSpec(
            id="P1",
            path="iface.crop_water.{slot}",
            record=CropWaterIn,
            fields=(
                ("sw", _f("cm3 cm-3", "n_layer", grid="dssat_layers")),
                ("eop", _f("mm d-1", "n_crop")),
                ("trwup", _f("cm d-1", "n_crop")),
            ),
            producers=(
                "crops.{slot}.remap_in (sw)",
                "crops.{slot}.eop (eop)",
                "water_supply.{slot}.rootwu (trwup)",
                "water_supply.{slot}.season_end (trwup = 0 at the end of a harvest day, the same module)",
            ),
            # the fields each consumer reads (the crop reads its water_in port: phenology and
            # roots water_in.sw, stress the whole record; the uptake limit trwup, with a lag)
            consumers=(
                "crops.{slot}.phenology (sw)",
                "crops.{slot}.stress (sw, eop, trwup)",
                "crops.{slot}.roots (sw)",
                "soil_water.uptake_limit (trwup)",
            ),
            time="mixed",
            default=(
                "zeros: eop = 0 gives no water stress; trwup = 0 with eop > 0 full stress; "
                "sw = 0 no root growth"
            ),
            lags=(
                AllowedLag(
                    "soil_water.uptake_limit",
                    "trwup",
                    "RZWQM2 4.6 PHYSCL: the uptake limit reads the previous day's TRWUP (" + _RZ_ORDER + ")",
                ),
            ),
            writers=(
                ("crops.{slot}.remap_in", ("sw",)),
                ("crops.{slot}.eop", ("eop",)),
                ("water_supply.{slot}.rootwu", ("trwup",)),
                ("water_supply.{slot}.season_end", ("trwup",)),
            ),
        ),
        PortSpec(
            id="P2",
            path="iface.root.{slot}",
            record=RootRecord,
            fields=(
                ("rlv", _f("cm cm-3", "n_crop", "n_layer", grid="dssat_layers")),
                ("rtdep", _f("cm", "n_crop")),
                ("rwumx", _f("cm3 cm-1 d-1", "n_crop")),
                ("pormin", _f("cm3 cm-3", "n_crop")),
                ("xhlai", _f("m2 m-2", "n_crop")),
            ),
            producers=(
                "crops.{slot}.publish",
                "crops.{slot}.harvest (the next season's no-crop record at the end of a harvest day)",
            ),
            consumers=("water_supply.{slot}.rootwu",),
            time="lag1",
            default="zeros: xhlai = 0, ROOTWU not called, rwu = trwup = 0, TSS unchanged",
            lags=(
                AllowedLag(
                    "water_supply.{slot}.rootwu",
                    "",
                    "DSSAT-CSM LAND.for: SPAM reads the RLV the crop published on the previous call; "
                    "RZWQM2 4.6 ROOTWU reads the previous day's DSSATDRV exit RLV",
                ),
            ),
        ),
        PortSpec(
            id="P3",
            path="iface.root_uptake.{slot}",
            record=NodeUptake,
            fields=(("uptake", _f("cm d-1", "n_crop", "n_node", grid="rzwqm2_nodes")),),
            producers=("crops.{slot}.publish_uptake",),
            consumers=("soil_water.uptake_limit",),
            time="lag1",
            default="zeros: no sink",
            lags=(
                AllowedLag(
                    "soil_water.uptake_limit",
                    "",
                    "RZWQM2 4.6: the PHYSCL entry QSR is the previous day's DSSATDRV exit QSR ("
                    + _RZ_ORDER
                    + ")",
                ),
            ),
        ),
        PortSpec(
            id="P4",
            path="soil_water.sink_in",
            record=SinkInputs,
            fields=tuple(
                (n, _f("cm d-1", "n_node", grid="rzwqm2_nodes"))
                for n in ("uptake", "tile", "lateral", "subirrigation", "macropore_to_drain")
            ),
            producers=("soil_water.uptake_limit (uptake)",),
            consumers=("soil_water.day",),
            time="same_day",
            default=(
                "zeros with only uptake active: bit-identical to the soil-water day validated with "
                "uptake as its only sink"
            ),
            # post-M3: the daily variant of the tile drainage writes ``tile`` (the faithful variant
            # evaluates the tile sink inside the sub-steps from P23 instead)
            writers=(("soil_water.uptake_limit", ("uptake",)), ("drainage.tile", ("tile",))),
        ),
        PortSpec(
            id="P5",
            path="iface.pet",
            record=PETFluxes,
            fields=(
                ("transpiration", _f("cm d-1")),
                ("soil_evaporation", _f("cm d-1")),
                ("residue_evaporation", _f("cm d-1")),
                ("eo_priestley_taylor", _f("mm d-1")),
            ),
            producers=("pet.sw_daily",),
            # soil_water.day reads PES + PER as its evaporation demand (the reference's actual
            # evaporation equals it on the days that are not supply-limited)
            consumers=("snow.prms", "soil_water.uptake_limit", "soil_water.day", "crops.{slot}.eop"),
            time="same_day",
            default="zeros: no demand",
        ),
        PortSpec(
            id="P6",
            path="iface.canopy.{slot}",
            record=CanopyRecord,
            fields=(
                ("lai", _f("m2 m-2", "n_crop")),
                ("tlai", _f("m2 m-2", "n_crop")),
                ("height", _f("cm", "n_crop")),
            ),
            producers=(
                "crops.{slot}.canopy",
                "crops.{slot}.harvest (bare soil at the end of a harvest day)",
            ),
            consumers=("pet.sw_daily",),
            time="lag1",
            default="zeros: bare soil (soil evaporation only)",
            lags=(
                AllowedLag(
                    "pet.sw_daily", "", "RZWQM2 4.6: PET in PHYSCL runs before the crop (" + _RZ_ORDER + ")"
                ),
            ),
        ),
        PortSpec(
            id="P7",
            path="soil_water.theta",
            record=None,
            field_of="agrijax.processes.soil_water.richards.SoilWater",
            fields=(("theta", _f("cm3 cm-3", "n_node")),),
            producers=("soil_water.day",),
            consumers=("pet.sw_daily",),
            time="lag1",
            default="the initial soil water",
            lags=(
                AllowedLag(
                    "pet.sw_daily",
                    "",
                    "RZWQM2 4.6: PET is computed before the PHYSCL time loop (" + _RZ_ORDER + ")",
                ),
            ),
        ),
        PortSpec(
            id="P8",
            path="forcing.weather",
            record=DailyWeather,
            fields=(
                ("tmin", _f("degC", "T")),
                ("tmax", _f("degC", "T")),
                ("srad", _f("MJ m-2 d-1", "T")),
                ("rh", _f("percent", "T")),
                ("wind_run", _f("km d-1", "T")),
                ("doy", _f("d", "T")),
                # horizontal-surface radiation RTH of the Shuttleworth-Wallace PET (srad is RTS);
                # defaults to srad when a caller leaves it out
                ("srad_horizontal", _f("MJ m-2 d-1", "T")),
            ),
            producers=("io (reference weather files) and the radiation reconstruction",),
            consumers=("pet.sw_daily", "snow.prms", "crops.{slot}"),
            time="forcing",
            default="-",
            kind="forcing",
        ),
        PortSpec(
            id="P9",
            path="iface.snow",
            record=SnowOut,
            fields=(
                ("melt", _f("cm d-1")),
                ("melt_runoff", _f("cm d-1")),
                ("swe", _f("mm")),
                ("sublimation", _f("cm d-1")),
            ),
            producers=("snow.prms",),
            consumers=("soil_water.day", "crops.{slot}.phenology", "ledger.close"),
            time="same_day",
            default="zeros: no snow",
        ),
        PortSpec(
            id="P10",
            path="iface.crop_n.{slot}",
            record=CropNIn,
            fields=(
                ("nstres", _f("-", "n_crop")),
                ("agefac", _f("-", "n_crop")),
                ("ndef3", _f("-", "n_crop")),
                ("npool", _f("g plant-1", "n_crop")),
            ),
            # The producer entry is n_supply.{slot}.replay with the registry key
            # n_supply/forcing_replay@none:replay (DAY_TABLE). Its ref_version is none: it replays
            # data, so it is exempt from the faithful-sibling rule and its key names no reference.
            producers=(
                "n_supply.{slot}.replay (m3 stage: n_supply/forcing_replay@none:replay); "
                "a nitrogen module later",
            ),
            consumers=("crops.{slot}.growth (nstress_replay variant)",),
            time="same_day",
            default=(
                "nstres = agefac = ndef3 = 1, npool = NPOOL_NO_CAP: no nitrogen limitation "
                "(the faithful nitrogen-off growth)"
            ),
        ),
        PortSpec(
            id="P11",
            path="ledger.water",
            record=WaterLedger,
            fields=(("storage0", _f("cm")), ("storage", _f("cm")), ("residual", _f("cm"))),
            producers=("ledger.close",),
            consumers=("outputs and checks",),
            time="cumulative",
            default="WaterLedger.init(storage0, inflows=..., outflows=...)",
        ),
        # ============================================================ planned (post-M3) ports
        PortSpec(
            id="P12",
            path="iface.soil_flux",
            record=SoilFluxRecord,
            fields=(
                ("q_down", _f("cm d-1", "n_node", grid="rzwqm2_nodes")),
                ("q_surface", _f("cm d-1")),
                ("theta_start", _f("cm3 cm-3", "n_node", grid="rzwqm2_nodes")),
                ("event", _f("cm d-1", "n_node", grid="rzwqm2_nodes")),
                ("runoff", _f("cm d-1")),
                *((n, _f("cm d-1", "n_node", grid="rzwqm2_nodes")) for n in _SINKS),
            ),
            producers=("soil_water.day",),
            consumers=("soil_heat.heatfx", "soil_om.transport", "phosphorus.day", "drainage.tile"),
            time="mixed",
            default="SoilFluxRecord.zeros: no water movement",
            lags=(
                AllowedLag(
                    "drainage.tile",
                    "",
                    "daily tile variant: the drain flux of a day from the previous day's water table "
                    "(inferred; the reference evaluates it per water step from the step's start, "
                    "RZWQM2 Rzday.for:2044-2062)",
                ),
            ),
            pending=(
                (
                    "wt_depth",
                    _f("cm"),
                    "water-table depth: the soil-water day's water-table variant (ITBL = 1)",
                ),
                ("macropore", _f("cm d-1"), "macropore flow: no macropore model yet"),
            ),
            stage="post_m3",
            off="unbound: no consumer before the nitrogen module; module tests replay the reference's "
            "node fluxes (soil_water/flux_replay@none:replay)",
            writers=(("soil_water.day", ()),),
        ),
        PortSpec(
            id="P13",
            path="iface.water_trace",
            record=WaterStepTrace,
            fields=(
                ("t0", _f("h", "n_step")),
                ("dt", _f("h", "n_step")),
                ("q_down", _f("cm h-1", "n_step", "n_node", grid="rzwqm2_nodes")),
                ("q_surface", _f("cm h-1", "n_step")),
                ("sink", _f("cm h-1", "n_step", "n_node", grid="rzwqm2_nodes")),
                ("theta", _f("cm3 cm-3", "n_step", "n_node", grid="rzwqm2_nodes")),
                ("t_event", _f("h")),
                ("event", _f("cm", "n_node", grid="rzwqm2_nodes")),
            ),
            producers=("soil_water.day",),
            consumers=("soil_heat.heatfx", "soil_om.transport"),
            time="same_day",
            default="WaterStepTrace.zeros: every step empty",
            stage="post_m3",
            off="unbound: the consumers' daily variants read P12 (a declared deviation from the reference's "
            "per-water-step coupling)",
        ),
        PortSpec(
            id="P14",
            path="iface.soil_temp",
            record=SoilTemperature,
            fields=(("t", _f("degC", "n_node", grid="rzwqm2_nodes")), ("t_surface", _f("degC"))),
            producers=("soil_heat.heatfx",),
            consumers=("soil_om.day", "phosphorus.day"),
            time="same_day",
            default="SoilTemperature.constant",
            stage="post_m3",
            off="replay of the reference's node temperature (soil_heat/forcing_replay@none:replay); "
            "no consumer "
            "in an assembly without nutrients (the crop of the m3 day reads no soil temperature)",
        ),
        PortSpec(
            id="P15",
            path="iface.soil_ice",
            record=SoilIce,
            fields=(("theta_ice", _f("cm3 cm-3", "n_node", grid="rzwqm2_nodes")),),
            producers=("soil_heat.heatfx",),
            consumers=("soil_water.day",),
            time="lag1",
            default="SoilIce.zeros: unfrozen soil",
            lags=(
                AllowedLag(
                    "soil_water.day",
                    "",
                    "inferred: with frozen soil the reference updates ice after each water step (RZ-SHAW, "
                    "RZWQM2 Rzday.for:2262 GOSHAW, 2340 NWPRSTY); a daily soil-water day reads "
                    "yesterday's ice, "
                    "a declared deviation",
                ),
            ),
            stage="post_m3",
            off="SoilIce.zeros or unbound: the soil-water day's static no-ice path (bit-identical to the "
            "m3 day); the reference's HEATFX path (ISHAW = 0) has no ice",
        ),
        PortSpec(
            id="P16",
            path="forcing.hourly",
            record=HourlyWeather,
            fields=(("tair", _f("degC", "T", "hour")),),
            producers=("io (the reference's daily-to-hourly generator, forcing preprocessing)",),
            consumers=("soil_heat.heatfx",),
            time="forcing",
            default="-",
            kind="forcing",
            stage="post_m3",
            off="absent: no consumer before the soil-heat module",
        ),
        PortSpec(
            id="P17",
            path="iface.residue",
            record=ResidueRecord,
            fields=(
                ("mass", _f("kg ha-1")),
                ("age", _f("d")),
                ("wet", _f("-")),
                ("kind", _f("-")),
            ),
            producers=("soil_om.day",),
            consumers=("pet.sw_daily", "phosphorus.day"),
            time="mixed",
            default="ResidueRecord.none: no residue",
            lags=(
                AllowedLag(
                    "pet.sw_daily",
                    "",
                    "inferred: the reference decomposes residue in its chemistry after PHYSCL (PET) "
                    "(RZWQM2 Rzmain.for:1546 PHYSCL before the nutrient call); not yet confirmed on a "
                    "reference dump",
                ),
            ),
            pending=(
                (
                    "carbon",
                    _f("kg ha-1"),
                    "surface residue carbon: the phosphorus module's residue P diagnostic",
                ),
            ),
            stage="post_m3",
            off="unbound: the PET keeps its residue parameters (PETSiteParams.residue), constant",
        ),
        PortSpec(
            id="P18",
            path="iface.crop_residue.{slot}",
            record=CropResidueOut,
            fields=(
                ("surface_dm", _f("kg ha-1 d-1", "n_crop")),
                ("surface_n", _f("kg ha-1 d-1", "n_crop")),
                ("root_dm", _f("kg ha-1 d-1", "n_crop", "n_layer", grid="dssat_layers")),
                ("root_n", _f("kg ha-1 d-1", "n_crop", "n_layer", grid="dssat_layers")),
                ("kind", _f("-", "n_crop")),
            ),
            producers=("crops.{slot}.residue_out",),
            consumers=("soil_om.day", "phosphorus.day"),
            time="mixed",
            default="CropResidueOut.zeros: no residue return",
            lags=(
                AllowedLag(
                    "soil_om.day",
                    "",
                    "DSSAT-CSM: SENESCE of day d enters SOM on day d+1 (CSM_Main/LAND.for order); "
                    "RZWQM2: the "
                    "nutrient call runs before the crop (Rzmain.for:1887 MAPLNT after it)",
                ),
            ),
            pending=(
                (
                    "surface_p",
                    _f("kg ha-1 d-1", "n_crop"),
                    "phosphorus to the surface: with the phosphorus module",
                ),
                (
                    "root_p",
                    _f("kg ha-1 d-1", "n_crop", "n_layer", grid="dssat_layers"),
                    "root phosphorus: with the phosphorus module",
                ),
            ),
            stage="post_m3",
            off="CropResidueOut.zeros: no return (in the m3 day the residue is a constant PET parameter)",
        ),
        PortSpec(
            id="P19",
            path="iface.soil_n",
            record=SoilMineralN,
            fields=(
                ("no3", _f("kg ha-1", "n_node", grid="rzwqm2_nodes")),
                ("nh4", _f("kg ha-1", "n_node", grid="rzwqm2_nodes")),
            ),
            producers=("soil_om.day",),
            consumers=("crops.{slot}.n_remap_in", "phosphorus.day"),
            time="same_day",
            default="SoilMineralN.zeros",
            stage="post_m3",
            off="unbound: the crop runs nitrogen-off, or reads the replayed N factors (P10)",
        ),
        PortSpec(
            id="P20",
            path="iface.crop_soil_n.{slot}",
            record=CropSoilNIn,
            fields=(
                ("no3", _f("kg ha-1", "n_layer", grid="dssat_layers")),
                ("nh4", _f("kg ha-1", "n_layer", grid="dssat_layers")),
            ),
            producers=("crops.{slot}.n_remap_in",),
            consumers=("crops.{slot}.growth", "crops.{slot}.roots"),
            time="same_day",
            default="CropSoilNIn.zeros",
            stage="post_m3",
            off="unbound: the nitrogen-off crop reads no soil nitrogen (RNFAC = 1, NSTRES from P10)",
        ),
        PortSpec(
            id="P21",
            path="iface.n_uptake.{slot}",
            record=CropNUptake,
            fields=(
                ("no3", _f("kg ha-1 d-1", "n_crop", "n_layer", grid="dssat_layers")),
                ("nh4", _f("kg ha-1 d-1", "n_crop", "n_layer", grid="dssat_layers")),
            ),
            producers=("crops.{slot}.growth (nitrogen-on variant: MZ_NUPTAK inside MZ_GROSUB)",),
            consumers=("crops.{slot}.publish_n_uptake",),
            time="same_day",
            default="CropNUptake.zeros: no uptake",
            stage="post_m3",
            off="unbound: no uptake (nitrogen-off crop)",
        ),
        PortSpec(
            id="P22",
            path="iface.root_n_uptake.{slot}",
            record=NodeNUptake,
            fields=(
                ("no3", _f("kg ha-1 d-1", "n_crop", "n_node", grid="rzwqm2_nodes")),
                ("nh4", _f("kg ha-1 d-1", "n_crop", "n_node", grid="rzwqm2_nodes")),
            ),
            producers=("crops.{slot}.publish_n_uptake",),
            consumers=("soil_om.day",),
            time="lag1",
            default="NodeNUptake.zeros: no uptake",
            lags=(
                AllowedLag(
                    "soil_om.day",
                    "",
                    "DSSAT-CSM: UNO3/UNH4 computed in the crop's INTEGR of day d are subtracted in "
                    "SoilNi's RATE "
                    "of day d+1; RZWQM2: the nutrient call runs before the crop (Rzmain.for:1887)",
                ),
            ),
            stage="post_m3",
            off="NodeNUptake.zeros: no uptake",
        ),
        PortSpec(
            id="P23",
            path="iface.drain",
            record=DrainControl,
            fields=(
                ("depth", _f("cm")),
                ("spacing", _f("cm")),
                ("radius", _f("cm")),
                ("impermeable_depth", _f("cm")),
                ("k_lat", _f("cm h-1", "n_node", grid="rzwqm2_nodes")),
            ),
            producers=("drainage.control",),
            consumers=("soil_water.day", "drainage.tile"),
            time="same_day",
            default="DrainControl.zeros: no drains",
            stage="post_m3",
            off="unbound: the tile channel is statically absent (bit-identical to the m3 day)",
        ),
        PortSpec(
            id="P24",
            path="iface.sediment",
            record=SedimentOut,
            fields=(("sediment", _f("kg ha-1 d-1")),),
            producers=("erosion.gleams",),
            consumers=("phosphorus.day",),
            time="same_day",
            default="SedimentOut.zeros: no erosion",
            stage="post_m3",
            off="SedimentOut.zeros: the reference's own erosion-off behaviour (no GLEAMS input: particulate "
            "phosphorus in runoff 0)",
        ),
        PortSpec(
            id="P25",
            path="iface.crop_status.{slot}",
            record=CropStatus,
            fields=(
                ("tops_dm", _f("kg ha-1", "n_crop")),
                ("root_dm", _f("kg ha-1", "n_crop")),
                ("grain_dm", _f("kg ha-1", "n_crop")),
                ("potential_dm", _f("kg ha-1", "n_crop")),
                ("dev_fraction", _f("-", "n_crop")),
            ),
            producers=("crops.{slot}.status",),
            consumers=("phosphorus.day",),
            time="same_day",
            default="CropStatus.zeros: no crop",
            stage="post_m3",
            off="CropStatus.zeros: no crop phosphorus uptake",
        ),
    )
}
