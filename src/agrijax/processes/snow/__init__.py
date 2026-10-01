"""Snow slot: the pack on the soil surface and what it hands on (port P9, ``iface.snow``).

The slot is the snow entry (``snow.prms``) of the RZWQM2-4.6 day: :func:`snow_prms`, the PRMS
single-layer snowpack of RZWQM2 4.6 (``ISHAW = 0``), keyed ``snow/prms@rzwqm2-4.6:faithful``. Its own
state lives at ``surface.snow``; it reads the daily weather and the day's breakpoint precipitation
(:class:`SnowForcing`) and writes the melt that infiltrates, the melt runoff, the water equivalent
(mm, the crop's ``SNOW``) and the sublimation to ``iface.snow``. The assembled day
:func:`agrijax.models.day_rzwqm46.day_rzwqm46` still fills that port with a replay of a reference run;
:func:`snow_prms` is validated on its own against RZWQM2 4.6
(``tests/integration/test_snow_prms_reference.py``).
"""

from .coefficients import PRMS_SNOW, PrmsSnowCoefficients
from .prms import PACK_FIELDS, PrmsSnowParams, SnowForcing, SnowState, snow_prms
from .sno import SnoFile, parse_sno, read_sno

__all__ = [
    "PACK_FIELDS",
    "PRMS_SNOW",
    "PrmsSnowCoefficients",
    "PrmsSnowParams",
    "SnoFile",
    "SnowForcing",
    "SnowState",
    "parse_sno",
    "read_sno",
    "snow_prms",
]
