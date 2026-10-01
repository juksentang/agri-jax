"""Conformance cases of the RZWQM2 convention variants of the soil-water day: the DRAIN cap
(``:drain_cap``), the flux-mode evaporation limit (``:flux_evap``), both (``:rzwqm2_conventions``) and
both on the flux-replay day (``:replay_flux_conventions``).

The inputs, balances, losses and tolerances are those of the faithful day cases
(:mod:`._builtin.soil_water`: the CA-TPA grid, converged fixed numerics 48 x 10, aef 0.9); each
variant sets its convention switches itself from the same parameters. The DRAIN cap keeps the ledger:
the water it passes out of the bottom node is booked in ``drainage``.
"""

from __future__ import annotations

import dataclasses

from ..case import ConformanceCase, Tolerance
from . import soil_water as SW

#: variant key -> the faithful case whose inputs it shares
_VARIANTS = {
    "soil_water/day@rzwqm2-4.6:drain_cap": "soil_water/day@rzwqm2-4.6:faithful",
    "soil_water/day@rzwqm2-4.6:flux_evap": "soil_water/day@rzwqm2-4.6:faithful",
    "soil_water/day@rzwqm2-4.6:rzwqm2_conventions": "soil_water/day@rzwqm2-4.6:faithful",
    "soil_water/day@rzwqm2-4.6:replay_flux_conventions": "soil_water/day@rzwqm2-4.6:replay_flux",
}


#: the DRAIN cap on the event day: float32 vmap(jit) against jit rounds the day balance by up to
#: 1.14e-5 cm (measured), just above the faithful case's 1e-5 cm; float64 as the faithful case
#: (measured 1.4e-14 cm, h 9.7e-13 cm)
_DRAIN_TRANSFORMS = (SW._TRANSFORMS_TOL[0], Tolerance(1e-4, 1.2e-5))
_DRAIN_WHY = SW._TRANSFORMS_WHY + (
    "; with the DRAIN cap on the event day (float32, x64 off): vmap(jit) against jit balance_error "
    "1.14e-5 cm, h 3.4e-4 cm, drainage 1.3e-6 cm; eager against jit balance_error 7.1e-6 cm"
)
_DRAIN_KEYS = ("soil_water/day@rzwqm2-4.6:drain_cap", "soil_water/day@rzwqm2-4.6:rzwqm2_conventions")


def cases() -> list[ConformanceCase]:
    base = {c.key: c for c in SW.cases()}
    out = []
    for key, faithful in _VARIANTS.items():
        c = dataclasses.replace(base[faithful], key=key)
        if key in _DRAIN_KEYS:
            c = dataclasses.replace(c, transforms_tol=_DRAIN_TRANSFORMS, transforms_why=_DRAIN_WHY)
        out.append(c)
    return out
