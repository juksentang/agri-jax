"""The module declaration of the DSSAT-CSM v4.8.6.0 tipping bucket (:data:`MODULE`).

Plain data (:class:`agrijax.iface.module.ModuleDeclaration`): the day entries, the scope, the
parameters with their DSSAT names, the ports, the own input fields written by other modules, the
forcing fields and the conservation responsibility. ``tests/unit/test_bucket.py``
(``test_module_declaration_matches_the_code``) checks it against the registry, the field metadata
and the ledger channels.

Source: DSSAT-CSM v4.8.6.0 ``Soil/SoilWater/WATBAL.for`` (BSD-3); ``CSM_Main/LAND.for`` (the day).
"""

from __future__ import annotations

from agrijax.iface.module import LedgerDuty, ModuleDeclaration, ParameterDecl

__all__ = ["MODULE"]

MODULE = ModuleDeclaration(
    slot="soil_water",
    name="tipping_bucket",
    reference="DSSAT-CSM v4.8.6.0 (dscsm048 build486)",
    entries=(
        ("soil_water.rate", "soil_water/tipping_bucket.rate@dssat-4.8.6.0:faithful"),
        ("soil_water.integrate", "soil_water/tipping_bucket.integrate@dssat-4.8.6.0:faithful"),
    ),
    order=(
        "soil_water.rate, then the evapotranspiration and root-uptake producers (DSSAT SPAM: they read "
        "SW + SWDELTS + SWDELTU from soil_water.sw and soil_water.flux), then soil_water.integrate "
        "(CSM_Main/LAND.for: SOIL RATE, SPAM RATE, PLANT RATE, SOIL INTEGR)"
    ),
    grid="dssat_layers (DLAYR; padded layers have dlayr = 0); ports P4 and P7 on the core grid "
    "(BucketParams.core, identity when None) through core/grids LMATCH remaps",
    supports=(
        "snow accumulation and melt (SNOWFALL)",
        "rain interception by surface residue and mulch water (MULCHWATER, MEINF in 'RSM')",
        "SCS curve-number runoff with Ritchie's abstraction index and mulch effect (RNOFF)",
        "plastic-mulch runoff share (PMFRACTION, a parameter)",
        "infiltration with SWCON drainage, SWCN caps and excess redistribution (INFIL)",
        "saturated drainage on days without infiltration (SATFLO)",
        "unsaturated upward flow (UP_FLOW; off with the SALUS evaporation MESEV = 'S')",
        "integration on yesterday's layer thickness, rounding to 1e-6 (WATBAL INTEGR)",
        "Ritchie (layer 1) and SALUS (per layer) soil evaporation removal",
        "daily soil properties replayed from SOILDYN (organic-matter changes of DLAYR, DS, LL, DUL)",
    ),
    not_supported=(
        "puddled or bunded (flooded) fields: FLOOD, PUDPERC (rice)",
        "tile drainage (TILEDRAIN, TDLNO > 0)",
        "managed water table and capillary fringe (WaterTable, ICWD; the depth enters INFIL as a parameter)",
        "tillage mixing of the water content (SoilMixing, SWDELTL)",
        "the zonal energy balance (MEEVP = 'Z', ETPHOT): WATBAL INTEGR's MEEVP = 'Z' branch is not ported",
        "the soil evaporation and root extraction themselves (SPAM: SOILEV, ESR_SoilEvap, XTRACT) - inputs",
        "the daily soil-property changes themselves (SOILDYN, soil organic matter) - replayed inputs",
    ),
    parameters=(
        ParameterDecl("soil.dlayr", "cm", "SLB (layer bottoms)", "layer thickness"),
        ParameterDecl("soil.ds", "cm", "SLB", "depth of the layer bottom"),
        ParameterDecl("soil.ll", "cm3 cm-3", "SLLL", "lower limit"),
        ParameterDecl("soil.dul", "cm3 cm-3", "SDUL", "drained upper limit"),
        ParameterDecl("soil.sat", "cm3 cm-3", "SSAT", "saturation"),
        ParameterDecl("soil.swcn", "cm h-1", "SSKS", "saturated hydraulic conductivity (<= 0: none)"),
        ParameterDecl("soil.cn", "-", "SLRO", "runoff curve number (as SOILDYN sets it)"),
        ParameterDecl("soil.swcon", "d-1", "SLDR", "drainage coefficient"),
        ParameterDecl("mulch_on", "-", "MEINF", "mulch effects (MEINF in 'RSM')"),
        ParameterDecl("salus_es", "-", "MESEV", "SALUS evaporation (MESEV = 'S')"),
        ParameterDecl("actwtd", "cm", "ICWD", "water table depth seen by INFIL (1000: none)"),
        ParameterDecl("pm_fraction", "-", "PMFRACTION", "plastic mulch cover"),
    ),
    ports_in=(
        (
            "P5",
            "iface.pet: soil_evaporation (ES, the actual amount, taken from the reference when the "
            "bucket_evap processes do not compute it) and residue_evaporation (EM)",
        ),
    ),
    ports_out=(("P7", "soil_water.theta: the water content on the core grid"),),
    inputs_own=(
        ("sink_in", "P4 soil_water.sink_in: uptake per core cell [cm d-1] written by the uptake producer"),
        ("evap_layers", "per-layer SALUS soil evaporation [cm d-1] written by the evaporation producer"),
    ),
    forcing=(
        ("rain", "mm d-1, WEATHER%RAIN"),
        ("tmax", "degC, WEATHER%TMAX"),
        ("irrigation", "mm d-1, IRRAMT"),
        ("mulch", "residue record (MULCHMASS, MULCHCOVER, NEWMULCH, MUL_WATFAC), optional"),
        ("soil, dlayr_end", "SOILDYN replay of SOILPROP at RATE and DLAYR at INTEGR, optional"),
    ),
    ledger=(
        LedgerDuty(
            quantity="water",
            storage="sum(SW DLAYR) + snow + mulch water [cm]",
            inflows=("rain", "irrigation", "residue_water"),
            outflows=(
                "runoff",
                "drainage",
                "soil_evaporation",
                "transpiration",
                "residue_evaporation",
                "truncation",
            ),
            relayed=("transpiration (the uptake producer decides it)", "soil_evaporation (SPAM decides it)"),
            note="truncation: the reference's rounding of SW to 1e-6 and its cuts of tiny stocks, booked "
            "so the ledger closes to floating-point rounding",
        ),
    ),
    accuracy="day by day against the instrumented dscsm048 v4.8.6.0 (tests/integration/test_bucket_dssat.py)",
    notes=(
        "parameters are not interchangeable with the Richards module's Brooks-Corey curve",
        "the process pair must bracket the evapotranspiration producers of the day",
    ),
)
