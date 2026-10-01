"""DSSAT-CSM v4.8.6.0 soil surface albedo ``MSALB`` (``SOILDYN`` RATE, ``ALBEDO_avg``).

``SOILDYN`` (``Soil/SoilUtilities/SOILDYN.for``) is called first in the soil module's RATE step,
before ``WATBAL`` (``Soil/SOIL.for:127``), and updates the soil albedo every day from the top
layer's water content and the mulch cover (``SOILDYN.for:1053`` calls ``ALBEDO_avg``):

* ``FF = (SW(1) - 0.03) / (DUL(1) - 0.03)``, limited to ``[0, 2]`` (lines 1528-1529);
* bare-soil albedo ``SWALB = SALB (1 - 0.45 FF)`` (line 1530);
* with the mulch switch on (``INFIL`` in ``R``, ``S``, ``M``): ``MSALB = MULCHCOVER MULCHALB + (1 -
  MULCHCOVER) SWALB``, else ``MSALB = SWALB`` (lines 1556-1565). ``MULCHALB = 0.45`` is set by the
  mulch module (``Soil/Mulch/MULCHLAYER.for:47``).

``PETPT`` reads ``MSALB`` the same day through ``SPAM`` (``ET_ALB = MSALB`` without flood,
``SPAM/SPAM.for:292-298``). The canopy-weighted ``CMSALB`` (line 1570) is not used by the
maize runs (no ``ETPHOT``) and is not computed.

Supported: the daily albedo of an upland soil with or without mulch. Not supported (declared):
the plastic-mulch albedo of ``SETPM`` (``SOILDYN.for:2330-2353``; ``PMFRACTION = 0`` in every
supported run), a flooded field (``ET_ALB = 0.05``, ``SPAM.for:293-295``).

Inputs: the start-of-day water content of the top layer (``SW(1)`` as ``SOILDYN`` sees it, before
``WATBAL`` RATE), ``DUL(1)`` of the day's soil (:class:`~agrijax.processes.soil_water.bucket.
BucketSoil`, the static soil or the ``SOILDYN`` replay of the forcing), ``SALB``, and the mulch
cover of the residue record (:class:`~agrijax.processes.soil_water.bucket.MulchForcing`). No store
changes (a property, not a flux).

Source: DSSAT-CSM v4.8.6.0 ``Soil/SoilUtilities/SOILDYN.for``, ``Soil/Mulch/MULCHLAYER.for`` (BSD-3,
Copyright 1998-2026 DSSAT Foundation, University of Florida, International Fertilizer Development
Center).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array

from agrijax.core.coefficients import Coefficients, Provenance, coef
from agrijax.core.ports import port
from agrijax.core.process import process
from agrijax.core.state import Params, State, field
from agrijax.iface.surface import SoilAlbedo

__all__ = [
    "ALBEDO_COEFFICIENTS",
    "AlbedoCoefficients",
    "SoilAlbedoParams",
    "SoilAlbedoState",
    "soil_albedo",
    "soil_albedo_rate",
]

_REF = "dssat-4.8.6.0"
_SD = "Soil/SoilUtilities/SOILDYN.for"


def _at(file_line: str, routine: str, statement: str, note: str = "") -> Provenance:
    return Provenance.at(_REF, file_line, routine=routine, statement=statement, note=note)


class AlbedoCoefficients(Coefficients):
    """The numbers of ``ALBEDO_avg`` and the mulch albedo of ``MULCHLAYER``."""

    sw_dry: float = coef(
        0.03,
        "cm3 cm-3",
        "top-layer water content at which the soil albedo is the file's SALB times one (FF = 0)",
        _at(f"{_SD}:1528", "ALBEDO_avg", "FF = (SW1 - 0.03) / (SOILPROP % DUL(1) - 0.03)"),
    )
    ff_max: float = coef(
        2.0,
        "-",
        "upper limit of the relative wetness FF of the top layer",
        _at(f"{_SD}:1529", "ALBEDO_avg", "FF = MAX(0.0, MIN(2.0, FF))"),
        calibrate=False,
    )
    wet_reduction: float = coef(
        0.45,
        "-",
        "relative decrease of the bare-soil albedo per unit FF",
        _at(f"{_SD}:1530", "ALBEDO_avg", "SWALB = SOILPROP % SALB * (1.0 - 0.45 * FF)"),
    )
    mulch_albedo: float = coef(
        0.45,
        "-",
        "albedo of the surface mulch (MULCH % MULCHALB)",
        _at("Soil/Mulch/MULCHLAYER.for:47", "MULCHLAYER", "MULCHALB = 0.45"),
    )


#: the DSSAT-CSM v4.8.6.0 values
ALBEDO_COEFFICIENTS = AlbedoCoefficients()


def soil_albedo(
    sw1: ArrayLike,
    dul1: ArrayLike,
    salb: ArrayLike,
    mulch_cover: ArrayLike,
    mulch_on: ArrayLike,
    c: AlbedoCoefficients = ALBEDO_COEFFICIENTS,
) -> tuple[Array, Array]:
    """``(MSALB, SWALB)`` of ``ALBEDO_avg``: the bare-soil albedo at the top layer's water content,
    then the mulch-weighted albedo (``mulch_on``: ``INFIL`` in ``R``, ``S``, ``M``). The denominator
    ``DUL(1) - 0.03`` is kept away from 0 (a soil with ``DUL(1) <= 0.03`` is not a DSSAT soil).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for ALBEDO_avg, lines 1528-1565 (BSD-3).
    """
    sw1, dul1, salb, cover = (jnp.asarray(x) for x in (sw1, dul1, salb, mulch_cover))
    den = dul1 - c.sw_dry
    safe = jnp.where(jnp.abs(den) > 0.0, den, 1.0)
    ff = jnp.where(jnp.abs(den) > 0.0, (sw1 - c.sw_dry) / safe, 0.0)
    ff = jnp.maximum(0.0, jnp.minimum(c.ff_max, ff))
    swalb = salb * (1.0 - c.wet_reduction * ff)
    mulched = cover * c.mulch_albedo + (1.0 - cover) * swalb
    msalb = jnp.where(jnp.asarray(mulch_on, dtype=bool), mulched, swalb)
    return msalb, swalb


class SoilAlbedoParams(Params):
    """The soil file's bare-soil albedo ``SALB``, the static top-layer ``DUL(1)`` (used unless the
    forcing replays the day's soil), the mulch switch and the coefficients."""

    salb: Array = field(
        dims=(), unit="-", description="bare soil albedo of the soil file (SALB)", fortran_name="SALB"
    )
    dul1: Array = field(
        dims=(), unit="cm3 cm-3", description="drained upper limit of the top layer", fortran_name="DUL"
    )
    mulch_on: Array = field(
        dims=(), unit="-", description="mulch albedo on (INFIL in R, S, M)", fortran_name="MEINF"
    )
    coefficients: AlbedoCoefficients | None = field(
        description="the ALBEDO_avg numbers (None: the DSSAT-CSM v4.8.6.0 values)", default=None
    )

    def coef(self) -> AlbedoCoefficients:
        """The coefficients in force."""
        return ALBEDO_COEFFICIENTS if self.coefficients is None else self.coefficients


class SoilAlbedoState(State):
    """Module state of the soil albedo: no fields of its own, two ports::

    sw      Array       P7 soil_water.theta  read (start-of-day layer water content; SW(1) used)
    albedo  SoilAlbedo  PD2 iface.soil_albedo  written (MSALB, SWALB)
    """

    sw: Array = port(
        unit="cm3 cm-3",
        dims=("n_layer",),
        grid="dssat_layers",
        fortran_name="SW",
        description="the soil's layer water content at the start of the day (SOILDYN runs before WATBAL)",
    )
    albedo: SoilAlbedo = port(description="the soil albedo record (PD2), written")


def _day_dul1(params: SoilAlbedoParams, forcing_t: Any) -> Array:
    """``DUL(1)`` of the day: the ``SOILDYN`` replay of the forcing when it carries the day's soil,
    else the static value.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for:1053 (SOILPROP % DUL(1) at the call).
    """
    soil = getattr(forcing_t, "soil", None)
    return params.dul1 if soil is None else soil.dul[..., 0]


def _day_cover(forcing_t: Any, like: Array) -> Array:
    """The mulch cover of the day's residue record (0 without one).

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for:1559 (MULCH % MULCHCOVER).
    """
    mulch = getattr(forcing_t, "mulch", None)
    return jnp.zeros_like(like) if mulch is None else jnp.asarray(mulch.cover)


@process(
    reads=("sw",),
    writes=("albedo",),
    source="DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for ALBEDO_avg (BSD-3)",
    fortran_name="ALBEDO_avg",
    key="soil_water/soil_albedo@dssat-4.8.6.0:faithful",
    provenance="translated_bsd3",
    grid="point",
    ref_build="dscsm048 v4.8.6.0 (gfortran 13, instrumented build: MSALB at the SPAM entry)",
    sources=(
        ("bare-soil albedo from the top layer's relative wetness", "SOILDYN.for:1528-1530"),
        ("mulch-weighted albedo (INFIL in R, S, M)", "SOILDYN.for:1556-1565; MULCHLAYER.for:47"),
        ("called every day in SOILDYN RATE, before WATBAL", "SOILDYN.for:1053; SOIL.for:127"),
    ),
    deviates=(
        (
            "DSSAT single precision (REAL*4) is not reproduced",
            "the kernel runs in the precision of its inputs",
            "tests/integration/test_day_dssat486_free.py (MSALB against the SPAM entry dump)",
        ),
        (
            "the plastic-mulch albedo (SETPM) and the flooded-field albedo are not implemented",
            "PMFRACTION = 0 and FLOOD = 0 in every supported run",
            "albedo.py module docstring",
        ),
        (
            "DUL(1) is the day's soil of the forcing (SOILDYN replay) or the static soil; the reference "
            "reads SOILPROP before its own daily soil-property update of the same call",
            "the soil properties change in 16 of 72 reference runs by at most 1.4e-3",
            "tests/integration/test_day_dssat486_free.py (measured MSALB error)",
        ),
    ),
)
def soil_albedo_rate(state: SoilAlbedoState, params: SoilAlbedoParams, forcing_t: Any) -> SoilAlbedoState:
    """``ALBEDO_avg``: the day's ``MSALB`` and ``SWALB`` into the albedo record (PD2) from the
    start-of-day ``SW(1)``, ``DUL(1)``, ``SALB`` and the residue record's mulch cover.

    Forcing fields read (optional): mulch.cover, soil.dul.

    Source: DSSAT-CSM v4.8.6.0 Soil/SoilUtilities/SOILDYN.for ALBEDO_avg, lines 1505-1577 (BSD-3).
    """
    sw1 = state.sw[..., 0]
    msalb, swalb = soil_albedo(
        sw1,
        _day_dul1(params, forcing_t),
        params.salb,
        _day_cover(forcing_t, sw1),
        params.mulch_on,
        params.coef(),
    )
    dt = state.albedo.msalb.dtype
    rec = SoilAlbedo(msalb=msalb.astype(dt), swalb=swalb.astype(dt))
    return eqx.tree_at(lambda s: s.albedo, state, rec)
