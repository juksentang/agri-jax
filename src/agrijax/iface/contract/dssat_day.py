"""The DSSAT-CSM v4.8.6.0 reference day: :class:`RefDayEntry`, :class:`RefLag`,
:data:`DSSAT_DAY_TABLE`, :data:`DSSAT_DAY_LAGS`, the DSSAT day's ports :data:`DSSAT_PORTS` and its
declared extra writes (:data:`DSSAT_PHASED_WRITES`, :data:`DSSAT_SHARED_STATE`).

Part of :mod:`agrijax.iface.contract` (the functions over these tables are there); import the
names from there.
"""

from __future__ import annotations

from dataclasses import dataclass

from agrijax.core.day import PhasedWrite

from ..crop import CanopyRecord, CropWaterIn
from ..soil import SinkInputs
from ..surface import EvaporationRecord, PETFluxes, SnowOut, SoilAlbedo
from .rzwqm_day import ENTRY_STATUS
from .spec import AllowedLag, PortSpec, _f

# ------------------------------------------------------------------------ the DSSAT-CSM v4.8.6.0 day
# A second reference day, next to the RZWQM2 4.6 one (its soil water is DSSAT's tipping bucket, its
# crop CERES-Maize 4.8.6). It is a table of its own: the rows of DAY_TABLE and the M3 checks of
# agrijax.iface.contract are not touched. The phases follow CSM_Main/LAND.for: WEATHR and MGMTOPS,
# then every module's RATE call (SOIL, SPAM, PLANT), then every module's INTEGR call.

#: the reference of :data:`DSSAT_DAY_TABLE`
DSSAT_DAY_REF: str = "dssat-4.8.6.0"
#: the phases of the DSSAT-CSM day, in order (LAND.for: RATE block lines 288-352, INTEGR 355-410)
DSSAT_DAY_PHASES: tuple[str, ...] = ("weather", "management", "rate", "integr", "ledger")
#: status of a DSSAT day row: the :data:`ENTRY_STATUS` values plus ``replay`` (the entry writes its
#: output ports from a reference run until its owner's process replaces it)
DSSAT_ENTRY_STATUS: tuple[str, ...] = (*ENTRY_STATUS, "replay")


@dataclass(frozen=True)
class RefDayEntry:
    """One entry of a reference day other than the RZWQM2 4.6 one (the DSSAT-CSM day,
    :data:`DSSAT_DAY_TABLE`).

    ``key`` is the registry key of the process in force or planned (``""``: not fixed yet; the
    owner names it); ``owner`` is a short tag of the module that provides (or will provide) the
    real process, ``source`` the reference call site (``file:line``). A ``replay`` row is stood in
    for by a replay of a reference run until the owner's process replaces it.

    The owner tags are plain labels: ``io`` (the input readers), ``tipping_bucket`` (the tipping-bucket
    soil water), ``soil_evaporation`` (the soil-evaporation modules), ``root_water_uptake`` (ROOTWU),
    ``ceres_maize`` (the CERES-Maize crop) and ``day_adapters`` (the day's own adapters and the
    processes no other module owns; a replay row must name the module that will replace it, so it
    cannot be ``day_adapters``).
    """

    row: str
    phase: str
    entry: str
    key: str
    ports_out: tuple[str, ...]
    status: str
    owner: str
    source: str
    note: str = ""

    def name(self, slot: str) -> str:
        """The entry name for crop slot ``slot``."""
        return self.entry.format(slot=slot)


@dataclass(frozen=True)
class RefLag:
    """A one-day lag a reference day allows: entry ``reader`` reads ``path`` (``{slot}`` form, a
    global state path) before the entry that writes it; ``evidence`` is the reference's file:line."""

    reader: str
    path: str
    evidence: str


_WB = "Soil/SoilWater/WATBAL.for"
_SPAM = "SPAM/SPAM.for"
_LAND = "CSM_Main/LAND.for"
_SD = "Soil/SoilUtilities/SOILDYN.for"

#: the DSSAT-CSM v4.8.6.0 day for the maize treatments (MEEVP = R, MESEV = R or S, N off), in LAND.for
#: order. Every row runs its process (no replay); the replays of the first skeleton of this day are
#: still available as ``day_processes(replace=replay_processes())`` of
#: :mod:`agrijax.models.day_dssat486`.
DSSAT_DAY_TABLE: tuple[RefDayEntry, ...] = (
    RefDayEntry(
        "D1",
        "weather",
        "weather.daily",
        "",
        (),
        "none",
        "io",
        f"{_LAND}:295 (WEATHR)",
        "the day's weather record is forcing (io/dssat readers; TAVG is the hourly mean of HMET); no state",
    ),
    RefDayEntry(
        "D2",
        "management",
        "events.apply",
        "",
        (),
        "none",
        "io",
        f"{_LAND}:301 (MGMTOPS RATE)",
        "the irrigation depth IRRAMT is forcing of the soil rate; the planting date a crop parameter",
    ),
    RefDayEntry(
        "D3a",
        "rate",
        "soil_water.albedo",
        "soil_water/soil_albedo@dssat-4.8.6.0:faithful",
        ("PD2",),
        "registered",
        "day_adapters",
        f"{_LAND}:311 (SOIL RATE) -> Soil/SOIL.for:127 (SOILDYN) -> {_SD}:1053 (ALBEDO_avg)",
        "MSALB from the start-of-day SW(1), DUL(1), SALB and the mulch cover; the other daily SOILPROP "
        "changes of SOILDYN (SOM-driven DLAYR, DS, DUL, LL in 16 of 72 reference runs) are a labelled "
        "replay in the soil forcing (BucketForcing.soil), limited to those runs",
    ),
    RefDayEntry(
        "D3",
        "rate",
        "soil_water.rate",
        "soil_water/tipping_bucket.rate@dssat-4.8.6.0:faithful",
        ("PD4", "PD7"),
        "registered",
        "tipping_bucket",
        f"{_LAND}:311 (SOIL RATE) -> {_WB}:276-459",
        "snow, mulch interception, RNOFF, INFIL / SATFLO, UP_FLOW; state soil_water (flux incl. WINF, snow)",
    ),
    RefDayEntry(
        "D4",
        "rate",
        "pet.priestley_taylor",
        "pet/priestley_taylor@dssat-4.8.6.0:port_soil_albedo",
        ("P5",),
        "registered",
        "day_adapters",
        f"{_SPAM}:292-305 (ET_ALB = MSALB; PET -> PETPT)",
        "PETPT with the day's MSALB from PD2; writes P5 eo_priestley_taylor (potential EO, mm d-1)",
    ),
    RefDayEntry(
        "D5a",
        "rate",
        "spam.pse",
        "pet/spam_pse@dssat-4.8.6.0:faithful",
        ("P5",),
        "registered",
        "soil_evaporation",
        f"{_SPAM}:314 (PSE)",
        "writes P5 soil_evaporation: the POTENTIAL soil evaporation EOS / 10 (cm d-1)",
    ),
    RefDayEntry(
        "D5b",
        "rate",
        "spam.mulch_evap",
        "soil_water/mulch_evap@dssat-4.8.6.0:faithful",
        (),
        "registered",
        "soil_evaporation",
        f"{_SPAM}:338-346 (MULCH_EVAP)",
        "EM and EOS_SOIL from P5 EOS, the start-of-day mulch water and the residue record; state "
        "surface.soil_evap (the SPAM soil-evaporation store)",
    ),
    RefDayEntry(
        "D5",
        "rate",
        "spam.soil_evaporation",
        "soil_water/soilev@dssat-4.8.6.0:faithful",
        ("PD1", "PD6"),
        "registered",
        "soil_evaporation",
        f"{_SPAM}:349-373 (SOILEV or ESR_SoilEvap; EVAP = ES + EM + EF)",
        "MESEV = R: soil_water/soilev; MESEV = S (GHWA0401, SIAZ9501, SIAZ9601): "
        "soil_water/esr_soilevap@dssat-4.8.6.0:faithful (a static choice by registry key: "
        "day_processes(soil_evaporation=...)); writes PD1 "
        "(the ACTUAL ES, EM, EF, EVAP, ES_LYR) and the bucket's SALUS layer evaporation",
    ),
    RefDayEntry(
        "D6",
        "rate",
        "spam.transpiration",
        "pet/spam_trans@dssat-4.8.6.0:faithful",
        ("P5", "P1"),
        "registered",
        "soil_evaporation",
        f"{_SPAM}:378-387 (TRANS)",
        "reads PD1 evaporation (EVAP, same day); writes P5 transpiration (potential EOP / 10, cm d-1) and "
        "P1 eop (EOP, mm d-1: SPAM hands EOP to PLANT)",
    ),
    RefDayEntry(
        "D7",
        "rate",
        "water_supply.{slot}.rootwu",
        "water_supply/rootwu@dssat-4.8.6.0:faithful",
        ("P1", "PD3"),
        "registered",
        "root_water_uptake",
        f"{_SPAM}:280-287 (ROOTWU when XHLAI > 0)",
        "reads the start-of-day SW (P1 sw, lag 1) and yesterday's root record (P2, lag 1)",
    ),
    RefDayEntry(
        "D8",
        "rate",
        "spam.xtract",
        "soil_water/xtract@dssat-4.8.6.0:faithful",
        ("P4", "PD1"),
        "registered",
        "day_adapters",
        f"{_SPAM}:392-398, 422-439 (EP = MIN(EOP, 10 TRWUP); XTRACT)",
        "writes P4 uptake: the actual layer extraction [cm d-1] on the DSSAT layers (-SWDELTX DLAYR), "
        "and PD1 transpiration (the actual EP)",
    ),
    RefDayEntry(
        "D9",
        "integr",
        "soil_water.integrate",
        "soil_water/tipping_bucket.integrate@dssat-4.8.6.0:faithful",
        ("P7", "PD5"),
        "registered",
        "tipping_bucket",
        f"{_LAND}:359 (SOIL INTEGR) -> {_WB}:465-533",
        "SW from the rates, ES from layer 1 (PD1) and the uptake (P4), rounded to 1e-6; state "
        "soil_water (sw, theta)",
    ),
    RefDayEntry(
        "D10",
        "integr",
        "crops.{slot}.layers_in",
        "crop_iface/layers_in@dssat-4.8.6.0:faithful",
        ("P1", "P9"),
        "adapter",
        "day_adapters",
        f"{_LAND}:386 (PLANT receives SW and SNOW)",
        "the crop layers are the soil layers: P1 sw = P7 theta (identity); the crop's SNOW is the "
        "bucket's snow pack after WATBAL RATE: P9 swe = soil_water.snow",
    ),
    RefDayEntry(
        "D11",
        "integr",
        "crops.{slot}.phenology",
        "crop/ceres_maize.phenology@dssat-4.8.6.0:faithful",
        (),
        "registered",
        "ceres_maize",
        "Plant/CERES-Maize/MZ_CERES.for:635-654 (MZ_PHENOL, INTEGR)",
    ),
    RefDayEntry(
        "D12",
        "integr",
        "crops.{slot}.stress",
        "crop/ceres_maize.stress@dssat-4.8.6.0:faithful",
        (),
        "registered",
        "ceres_maize",
        "Plant/CERES-Maize/MZ_CERES.for:658-732 (MZ_GROSUB, water-stress block)",
    ),
    RefDayEntry(
        "D13",
        "integr",
        "crops.{slot}.growth",
        "crop/ceres_maize.growth@dssat-4.8.6.0:faithful",
        (),
        "registered",
        "ceres_maize",
        "Plant/CERES-Maize/MZ_CERES.for:658-732 (MZ_GROSUB, ISTAGE 1-6)",
        "nitrogen off (ISWNIT = N): the faithful growth, no P10",
    ),
    RefDayEntry(
        "D14",
        "integr",
        "crops.{slot}.roots",
        "crop/ceres_maize.roots@dssat-4.8.6.0:faithful",
        (),
        "registered",
        "ceres_maize",
        "Plant/CERES-Maize/MZ_CERES.for:737-745 (MZ_ROOTGR)",
    ),
    RefDayEntry(
        "D15",
        "integr",
        "crops.{slot}.publish",
        "crop/ceres_maize.publish@dssat-4.8.6.0:faithful",
        ("P2",),
        "registered",
        "ceres_maize",
        "Plant/CERES-Maize/MZ_GROSUB.for:1818-1819 (XLAI = XHLAI = LAI); LAND.for:386 PLANT outputs",
    ),
    RefDayEntry(
        "D16",
        "integr",
        "crops.{slot}.canopy",
        "crop_iface/canopy_from_ceres@dssat-4.8.6.0:faithful",
        ("P6",),
        "adapter",
        "day_adapters",
        "Plant/CERES-Maize/MZ_GROSUB.for:1818-1831 (XLAI, XHLAI, CANHT)",
        "the canopy SPAM reads the next day: lai = tlai = XHLAI = XLAI = LAI, height = 100 CANHT "
        "(PLANT's outputs)",
    ),
    RefDayEntry(
        "D17",
        "ledger",
        "ledger.close",
        "",
        ("P11",),
        "adapter",
        "day_adapters",
        f"{_WB} WBAL (daily balance)",
        "soil profile + snow + mulch water; rain, irrigation, residue water in; runoff, drainage, "
        "soil and mulch evaporation (PD1), uptake (P4) and the reference's truncations out",
    ),
)

_XHLAI = "PLANT's XHLAI / XLAI of the previous INTEGR call (LAND.for:325-333 before 386)"
_SW0 = "SW before the day's SOIL INTEGR (LAND.for:311-352 before 359)"
_MW0 = "MULCHWAT before the day's MULCHWATER INTEGR (WATBAL.for:362-365 RATE, 510-515 INTEGR)"

#: the lags the DSSAT-CSM day allows (``{slot}`` form); :func:`dssat_allowed_lags` makes the Lag table
DSSAT_DAY_LAGS: tuple[RefLag, ...] = (
    RefLag("pet.priestley_taylor", "iface.canopy.{slot}", f"{_SPAM}:300 PET reads XHLAI: {_XHLAI}"),
    RefLag("spam.pse", "iface.canopy.{slot}", f"{_SPAM}:314 PSE reads XLAI: {_XHLAI}"),
    RefLag(
        "spam.mulch_evap",
        "soil_water.mulch_wat",
        f"{_SPAM}:339 MULCH_EVAP reads MULCH%MULCHWAT: {_MW0}",
    ),
    RefLag("spam.soil_evaporation", "soil_water.theta", f"{_SPAM}:364 SOILEV reads SW: {_SW0}"),
    RefLag("spam.transpiration", "iface.canopy.{slot}", f"{_SPAM}:379 TRANS reads XHLAI: {_XHLAI}"),
    RefLag(
        "water_supply.{slot}.rootwu",
        "iface.root.{slot}",
        f"{_SPAM}:281 ROOTWU reads RLV, RWUMX, PORMIN, XHLAI: {_XHLAI}",
    ),
    RefLag(
        "water_supply.{slot}.rootwu", "iface.crop_water.{slot}.sw", f"{_SPAM}:281 ROOTWU reads SW: {_SW0}"
    ),
    RefLag("spam.xtract", "soil_water.theta", f"{_SPAM}:435 XTRACT reads SW: {_SW0}"),
    RefLag("spam.xtract", "iface.canopy.{slot}", f"{_SPAM}:392 EP reads XHLAI: {_XHLAI}"),
)


# ------------------------------------------------------------------------ the DSSAT day's ports
# The ports of the DSSAT-CSM day where they differ from PORTS (the ``m3`` = RZWQM2 day's table): the
# same record on the same path, with the DSSAT day's producers, consumers, grid and meaning; and two
# records of its own (PD1, PD2) and the fields of one module's state another module reads or writes
# (PD3-PD7, like P7). PORTS is not changed. Potential vs actual: P5 carries only the
# POTENTIAL rates (EO, EOS, EOP), PD1 only the ACTUAL ones (ES, EM, EF, EVAP, ES_LYR, EP).
_DPT = "same_day"
DSSAT_PORTS: dict[str, PortSpec] = {
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
                "crops.{slot}.layers_in (sw)",
                "spam.transpiration (eop)",
                "water_supply.{slot}.rootwu (trwup)",
            ),
            consumers=(
                "crops.{slot}.stress",
                "crops.{slot}.roots",
                "water_supply.{slot}.rootwu",
                "spam.xtract",
            ),
            time="mixed",
            default="zeros: eop = 0 gives no water stress; sw = 0 no root growth",
            lags=(AllowedLag("water_supply.{slot}.rootwu", "sw", f"{_SPAM}:281 ROOTWU reads SW: {_SW0}"),),
            writers=(
                ("crops.{slot}.layers_in", ("sw",)),
                ("spam.transpiration", ("eop",)),
                ("water_supply.{slot}.rootwu", ("trwup",)),
            ),
        ),
        PortSpec(
            id="P9",
            path="iface.snow",
            record=SnowOut,
            fields=(("swe", _f("mm")),),
            producers=("crops.{slot}.layers_in (swe)",),
            consumers=("crops.{slot}.phenology",),
            time=_DPT,
            default="zeros: no snow (swe = the initial snow pack in initial_state)",
            writers=(("crops.{slot}.layers_in", ("swe",)),),
        ),
        PortSpec(
            id="P4",
            path="soil_water.sink_in",
            record=SinkInputs,
            # the soil-water module's core grid: here the DSSAT layers themselves (the record class
            # declares the RZWQM2 nodes of the M3 day, so the grid is left to the binding)
            fields=(("uptake", _f("cm d-1", "n_node")),),
            producers=("spam.xtract (uptake)",),
            consumers=("soil_water.integrate", "ledger.close"),
            time=_DPT,
            default="zeros: no root extraction",
            writers=(("spam.xtract", ("uptake",)),),
        ),
        PortSpec(
            id="P5",
            path="iface.pet",
            record=PETFluxes,
            fields=(
                ("eo_priestley_taylor", _f("mm d-1")),
                ("soil_evaporation", _f("cm d-1")),
                ("transpiration", _f("cm d-1")),
            ),
            producers=(
                "pet.priestley_taylor (eo_priestley_taylor)",
                "spam.pse (soil_evaporation)",
                "spam.transpiration (transpiration)",
            ),
            consumers=("spam.pse", "spam.mulch_evap", "spam.transpiration"),
            time=_DPT,
            default="zeros: no demand",
            writers=(
                ("pet.priestley_taylor", ("eo_priestley_taylor",)),
                ("spam.pse", ("soil_evaporation",)),
                ("spam.transpiration", ("transpiration",)),
            ),
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
            producers=("crops.{slot}.canopy",),
            consumers=("pet.priestley_taylor", "spam.pse", "spam.transpiration", "spam.xtract"),
            time="lag1",
            default="zeros: bare soil",
            lags=tuple(
                AllowedLag(r, "", f"{_SPAM}: {r} reads XHLAI / XLAI: {_XHLAI}")
                for r in ("pet.priestley_taylor", "spam.pse", "spam.transpiration", "spam.xtract")
            ),
        ),
        PortSpec(
            id="P7",
            path="soil_water.theta",
            record=None,
            field_of="agrijax.processes.soil_water.bucket.BucketState",
            fields=(("theta", _f("cm3 cm-3", "n_node")),),
            producers=("soil_water.integrate",),
            # SOILDYN (same module), SOILEV and XTRACT read the start-of-day SW: yesterday's P7
            consumers=("crops.{slot}.layers_in", "soil_water.albedo", "spam.soil_evaporation", "spam.xtract"),
            time="mixed",
            default="the initial soil water",
            lags=(
                AllowedLag("spam.soil_evaporation", "", f"{_SPAM}:364 SOILEV reads SW: {_SW0}"),
                AllowedLag("spam.xtract", "", f"{_SPAM}:435 XTRACT reads SW: {_SW0}"),
            ),
        ),
        PortSpec(
            id="PD1",
            path="iface.evaporation",
            record=EvaporationRecord,
            fields=(
                ("soil_evaporation", _f("cm d-1")),
                ("residue_evaporation", _f("cm d-1")),
                ("flood_evaporation", _f("cm d-1")),
                ("evaporation", _f("mm d-1")),
                ("soil_evaporation_layers", _f("cm d-1", "n_layer", grid="dssat_layers")),
                ("transpiration", _f("mm d-1")),
            ),
            producers=(
                "spam.soil_evaporation (soil_evaporation, residue_evaporation, flood_evaporation, "
                "evaporation, soil_evaporation_layers)",
                "spam.xtract (transpiration)",
            ),
            consumers=("spam.transpiration", "spam.xtract", "soil_water.integrate", "ledger.close"),
            time=_DPT,
            default="zeros: no evaporation",
            writers=(
                (
                    "spam.soil_evaporation",
                    (
                        "soil_evaporation",
                        "residue_evaporation",
                        "flood_evaporation",
                        "evaporation",
                        "soil_evaporation_layers",
                    ),
                ),
                ("spam.xtract", ("transpiration",)),
            ),
        ),
        PortSpec(
            id="PD2",
            path="iface.soil_albedo",
            record=SoilAlbedo,
            fields=(("msalb", _f("-")), ("swalb", _f("-"))),
            producers=("soil_water.albedo",),
            consumers=("pet.priestley_taylor",),
            time=_DPT,
            default="SoilAlbedo.constant(SALB): the soil file's albedo (SOILDYN SEASINIT)",
            writers=(("soil_water.albedo", ("msalb", "swalb")),),
        ),
        PortSpec(
            id="PD3",
            path="water_supply.{slot}.rwu",
            record=None,
            field_of="agrijax.processes.water_supply.RootwuState",
            fields=(("rwu", _f("cm d-1", "n_crop", "n_layer")),),
            producers=("water_supply.{slot}.rootwu",),
            consumers=("spam.xtract",),
            time=_DPT,
            default="zeros: no potential uptake (ROOTWU not called while XHLAI = 0)",
        ),
        PortSpec(
            id="PD4",
            path="soil_water.flux",
            record=None,
            field_of="agrijax.processes.soil_water.bucket.BucketFluxes",
            fields=(
                ("swdelts", _f("cm3 cm-3", "n_layer", grid="dssat_layers")),
                ("swdeltu", _f("cm3 cm-3", "n_layer", grid="dssat_layers")),
                ("winf", _f("mm d-1")),
            ),
            producers=("soil_water.rate",),
            consumers=("spam.soil_evaporation", "spam.xtract", "ledger.close"),
            time=_DPT,
            default="zeros: no infiltration, drainage or upward flow",
        ),
        PortSpec(
            id="PD5",
            path="soil_water.mulch_wat",
            record=None,
            field_of="agrijax.processes.soil_water.bucket.BucketState",
            fields=(("mulch_wat", _f("mm")),),
            producers=("soil_water.integrate",),
            consumers=("spam.mulch_evap",),
            time="lag1",
            default="0: no water on the mulch",
            lags=(AllowedLag("spam.mulch_evap", "", f"{_SPAM}:339 MULCH_EVAP reads MULCH%MULCHWAT: {_MW0}"),),
        ),
        PortSpec(
            id="PD6",
            path="soil_water.evap_layers",
            record=None,
            field_of="agrijax.processes.soil_water.bucket.BucketState",
            fields=(("evap_layers", _f("cm d-1", "n_layer", grid="dssat_layers")),),
            producers=("spam.soil_evaporation",),
            consumers=("soil_water.integrate", "ledger.close"),
            time=_DPT,
            default="zeros: MESEV = R (the soil evaporation leaves layer 1 through PD1 soil_evaporation)",
        ),
        PortSpec(
            id="PD7",
            path="soil_water.snow",
            record=None,
            field_of="agrijax.processes.soil_water.bucket.BucketState",
            fields=(("snow", _f("mm")),),
            producers=("soil_water.rate",),
            consumers=("crops.{slot}.layers_in", "ledger.close"),
            time=_DPT,
            default="the initial snow pack (0: no snow)",
        ),
    )
}


#: the DSSAT-CSM day's declared extra writes (``{slot}`` form; see :data:`PHASED_WRITES`)
DSSAT_PHASED_WRITES: tuple[PhasedWrite, ...] = (
    PhasedWrite(
        "soil_water.integrate",
        "soil_water.flux.truncation",
        "daily",
        "WATBAL INTEGR books the rounding of SW to 1e-6 and the cut of tiny stocks into the truncation "
        "that RATE started, before the ledger reads the day's total (WATBAL.for:465-533)",
    ),
)

#: module state (not a port) of the DSSAT-CSM day with a declared extra write: ``{path: (producer,
#: readers of other modules)}`` (see :data:`SHARED_STATE`)
DSSAT_SHARED_STATE: dict[str, tuple[str, tuple[str, ...]]] = {
    "soil_water.flux.truncation": ("soil_water.rate", ("ledger.close",)),
}
