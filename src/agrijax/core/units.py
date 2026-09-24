"""Unit constants and conversions.

Internal units follow the RZWQM ``.ana`` convention so that comparison with the
Fortran oracle needs no conversion: length in cm, temperature in degC, radiation
in MJ m-2 d-1, mass in kg ha-1, time in days.
"""

from __future__ import annotations

# ---- length ---------------------------------------------------------------
CM_PER_MM: float = 0.1
MM_PER_CM: float = 10.0
CM_PER_M: float = 100.0
M_PER_CM: float = 0.01

# ---- time -----------------------------------------------------------------
SECONDS_PER_HOUR: float = 3600.0
SECONDS_PER_DAY: float = 86400.0
HOURS_PER_DAY: float = 24.0

# ---- energy / radiation ---------------------------------------------------
#: multiply an energy flux in MJ m-2 d-1 by this to get W m-2 (1e6 J / 86400 s).
W_PER_M2_PER_MJ_M2_DAY: float = 1.0e6 / SECONDS_PER_DAY
#: multiply a flux in W m-2 by this to get MJ m-2 d-1.
MJ_M2_DAY_PER_W_PER_M2: float = SECONDS_PER_DAY / 1.0e6
#: latent heat of vaporisation of water at ~20 degC [MJ kg-1] (FAO-56 constant).
LATENT_HEAT_MJ_PER_KG: float = 2.45
#: Stefan-Boltzmann constant [MJ m-2 K-4 d-1] (FAO-56 value).
STEFAN_BOLTZMANN_MJ_M2_K4_DAY: float = 4.903e-9
#: Stefan-Boltzmann constant [W m-2 K-4].
STEFAN_BOLTZMANN_W_M2_K4: float = 5.670374419e-8
#: PAR fraction of global shortwave radiation used by CERES and RZWQM.
PAR_FRACTION: float = 0.5

# ---- mass -----------------------------------------------------------------
G_M2_PER_KG_HA: float = 0.1
KG_HA_PER_G_M2: float = 10.0

# ---- temperature ----------------------------------------------------------
KELVIN_OFFSET: float = 273.15

# ---- water ----------------------------------------------------------------
#: density of liquid water [kg m-3].
WATER_DENSITY_KG_M3: float = 1000.0
#: 1 mm of water over 1 m2 is 1 kg; 1 cm over 1 ha is 100 t.
KG_PER_M2_PER_MM: float = 1.0


def mm_to_cm(x):
    """Convert a length or water depth from mm to cm."""
    return x * CM_PER_MM


def cm_to_mm(x):
    """Convert a length or water depth from cm to mm."""
    return x * MM_PER_CM


def mj_m2_day_to_w_m2(x):
    """Convert a daily energy flux from MJ m-2 d-1 to a mean power density in W m-2."""
    return x * W_PER_M2_PER_MJ_M2_DAY


def w_m2_to_mj_m2_day(x):
    """Convert a mean power density in W m-2 to a daily energy flux in MJ m-2 d-1."""
    return x * MJ_M2_DAY_PER_W_PER_M2


def celsius_to_kelvin(x):
    """Convert degC to K."""
    return x + KELVIN_OFFSET


def kelvin_to_celsius(x):
    """Convert K to degC."""
    return x - KELVIN_OFFSET


def kg_ha_to_g_m2(x):
    """Convert an areal mass from kg ha-1 to g m-2."""
    return x * G_M2_PER_KG_HA


def g_m2_to_kg_ha(x):
    """Convert an areal mass from g m-2 to kg ha-1."""
    return x * KG_HA_PER_G_M2


def mj_m2_day_to_mm_water(x, latent_heat=LATENT_HEAT_MJ_PER_KG):
    """Convert an energy flux in MJ m-2 d-1 into the equivalent evaporated water depth in mm d-1."""
    return x / latent_heat


def mm_water_to_mj_m2_day(x, latent_heat=LATENT_HEAT_MJ_PER_KG):
    """Convert an evaporated water depth in mm d-1 into an energy flux in MJ m-2 d-1."""
    return x * latent_heat
