r"""Two RZWQM2 soil-water conventions the Richards step itself does not have (``DRAIN`` and ``CHKBC``).

Both are switches of :class:`~agrijax.processes.soil_water.richards.RichardsConfig` (off by default)
and are selected by the convention variants of the soil-water day
(:mod:`~agrijax.processes.soil_water.day`: ``soil_water/day@rzwqm2-4.6:drain_cap``, ``:flux_evap``,
``:rzwqm2_conventions`` and ``:replay_flux_conventions``). With both off every mode computes what
it computed before they existed, bit for bit.

**DRAIN cap** (``RichardsConfig.drain_cap``; RZWQM2 ``DRAIN``, called by ``REDIST`` after every
``RICHRD`` step). A node never holds more water than the field-saturated porosity
``pori = aef theta_s`` (Ahuja et al. 2000, ch. 3: the soil fills to field saturation, not to
``theta_s``; ``aef`` the field-saturation fraction of the control record, the same one the
Green-Ampt event fills to). After each accepted Richards sub-step, starting at the first node whose
water content exceeds ``pori`` by more than :data:`DRAIN_EXCESS_TOL`, the excess of every node
passes at once to the node below; the excess of the bottom node leaves the profile as seepage,
counted in the drainage (and on its own in ``SoilWaterFluxes.drain_seepage``). The heads of every
node whose water content changed (the capped nodes and the nodes that received water) are
recomputed from the water content (RZWQM2 ``WCHEAD``); the other nodes keep head and water content
bit for bit, so a step without excess is the identity. The cascade conserves water node by node:
``sum(theta tl)`` before is ``sum(theta tl)`` after plus the seepage.

**Flux-mode evaporation limit** (``RichardsConfig.evaporation_limit = "flux_peak"``; RZWQM2
``CHKBC``/``CNHEAD``). RZWQM2 keeps the evaporation demand as a flux condition on the upper ghost
node as long as the iterated ghost head stays above ``Hmin``, and switches to the head condition
``Hmin`` only when it cannot. With the geometric-mean face conductivity
``K_g = (K(h_g) K(h_0))^m`` (``m`` = ``richards.face_k_mean_exponent`` = 1/2) the evaporation the
ghost face carries, ``E(h_g) = K_g ((h_0 - h_g)/dz - 1)``, is not monotone in the ghost head
``h_g``: on the Brooks-Corey segment ``K(h_g) ~ |h_g|^-eps``, so ``E ~ x^(-m eps) (x - a)`` with
``x = -h_g`` and ``a = dz - h_0``, which peaks at

.. math:: x^* = \frac{m\,\varepsilon}{m\,\varepsilon - 1}\,(dz - h_0) \qquad (m\,\varepsilon > 1)

and falls towards ``Hmin``. The largest evaporation the surface can deliver in flux mode is this
peak, not the flux with the ghost at ``Hmin`` that the default dry limit of
:func:`~agrijax.processes.soil_water.richards.surface_fluxes` uses. With this limit the dry
limit there is the Darcy flux at ``h_g = -x^*`` (:func:`flux_peak_head`, with the dry-end exponent
of the surface node's curve, ``Hmin`` where ``x^* > -Hmin`` or ``m eps <= 1``) or at ``Hmin``,
whichever carries more evaporation, so it is never below the default limit (and equal to it when
``m eps <= 1``). Not ported: once the demand exceeds
the peak, RZWQM2 holds the head condition ``Hmin`` for the rest of the day (``CHKBC`` keeps it until
``RICHRD`` lifts the ghost head at the next day's start); here the limit is the peak on every
sub-step (the rule that, at the US_Rockfish site, closes an evaporation deficit of 1.3-6.1 cm a-1
against the reference to 0.008 cm or less), so evaporation on such a day is at most over-, never
under-estimated against that rule, and the days where it matters are the days with an evaporation deficit
under this limit.

Source: Ahuja, L.R., Rojas, K.W., Hanson, J.D., Shaffer, M.J., Ma, L. (eds.), 2000. Root Zone
Water Quality Model, ch. 3 (field saturation, flux and head boundary conditions); RZWQM2 4.6
``DRAIN`` (``Rzday.for:3975``, called at ``Rzrich.for:1092``), ``WCHEAD``, ``CHKBC``
(``Rzrich.for:3-110``), ``PORI`` (``Rzmain.for:538``), read for conventions only.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
from jaxtyping import Array

from agrijax.core.depth_scan import depth_scan

from .coefficients import numerical_setting, rzwqm2
from .hydraulics import AnyHydraulicParams, SoilHydraulicParams, TilledSoilHydraulicParams, h_of_theta

__all__ = [
    "DRAIN_EXCESS_TOL",
    "DrainResult",
    "drain_cap",
    "dry_end_eps",
    "field_saturation",
    "flux_peak_head",
]

DRAIN_EXCESS_TOL: float = numerical_setting(
    "drain.excess_tol",
    1.0e-12,
    "cm3 cm-3",
    "DRAIN cap: the cascade starts at the first node whose water content exceeds the field-saturated "
    "porosity by more than this; below that node nothing moves",
    origin="rzwqm2-4.6",
    provenance=rzwqm2(
        "RZWQM/Rzday.for:4060", "DRAIN", note="the trigger of the node loop; called at Rzrich.for:1092"
    ),
)


def _current(soil: AnyHydraulicParams) -> SoilHydraulicParams:
    return soil.current if isinstance(soil, TilledSoilHydraulicParams) else soil


def field_saturation(soil: AnyHydraulicParams, aef: Any) -> Array:
    """Field-saturated porosity ``pori = aef theta_s`` per node [cm3 cm-3] (``soil`` on the node axis;
    the current curve's ``theta_s`` after tillage, as the Green-Ampt event).

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``PORI`` (``Rzmain.for:538``), read for conventions.
    """
    return _current(soil).theta_s * aef


class DrainResult(NamedTuple):
    """Outcome of :func:`drain_cap`."""

    theta: Array
    h: Array
    seepage: Array  # [cm] out of the bottom node
    moved: Array  # [cm] sum over the nodes of the excess each passed down (the seepage included)


def drain_cap(theta: Array, h: Array, soil: AnyHydraulicParams, tl: Array, pori: Array) -> DrainResult:
    """RZWQM2's ``DRAIN`` after a Richards sub-step: returns ``(theta, h, seepage, moved)`` [cm].

    From the first node with ``theta - pori > DRAIN_EXCESS_TOL`` down, each node keeps at most
    ``pori tl`` of its water plus the excess passed from the node above and passes the rest on;
    the bottom node's rest is the seepage. Changed nodes get ``h = h(theta)``; unchanged nodes keep
    ``theta`` and ``h`` bit for bit. ``soil`` on the node axis.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``DRAIN`` (``Rzday.for:3975``) and ``WCHEAD``, read for
    conventions.
    """
    zero = jnp.zeros((), theta.dtype)

    def step(carry: tuple[Array, Array], x: tuple[Array, Array, Array]) -> tuple[Any, Any]:
        excess_in, started = carry[0], carry[1]
        th, t, p = x
        started = started | (th - p > DRAIN_EXCESS_TOL)
        water = th * t + excess_in
        cap = p * t
        over = started & (water > cap)
        out = jnp.where(over, water - cap, zero)
        th_in = water / t  # tl > 0 (a grid of positive cells)
        th_new = jnp.where(over, p, jnp.where(excess_in > 0.0, th_in, th))
        return (out, started), (th_new, out)

    n = theta.shape[-1]
    init = (zero, jnp.zeros((), bool))
    (seep, _), (th, passed) = depth_scan(step, init, (theta, tl, pori), unroll=n)
    changed = th != theta
    h_new = jnp.where(changed, h_of_theta(th, soil), h)
    return DrainResult(th, h_new, seep, jnp.sum(passed))


def dry_end_eps(soil: AnyHydraulicParams) -> Array:
    """``eps`` of the conductivity curve at the dry end (the pre-tillage curve after tillage), per node.

    Source: RZWQM2 ``POINTK`` with ``SN22`` (``hydraulics.k_of_h_tilled``); Ahuja et al. (2000) ch. 3.
    """
    return soil.original.eps if isinstance(soil, TilledSoilHydraulicParams) else soil.eps


def flux_peak_head(ht0: Array, eps: Array, h_min: Array, dz_top: Array, m: float) -> Array:
    """Ghost head [cm] at which the flux-mode evaporation through the ghost face peaks.

    ``ht0`` the (time-weighted) head of node 0, ``eps`` the dry-end exponent of its conductivity
    curve, ``m`` the face mean exponent: ``-x^*`` with ``x^* = m eps / (m eps - 1) (dz - h_0)``
    (module docstring), or ``h_min`` when ``m eps <= 1`` (no interior peak) or ``x^* > -h_min``. The
    caller evaluates the Darcy flux here and at ``h_min`` and keeps the larger evaporation.

    Source: Ahuja et al. (2000) ch. 3; RZWQM2 ``CHKBC`` (``Rzrich.for:3-110``), read for conventions;
    the peak from the Brooks-Corey segment of ``K`` (``hydraulics.k_of_h``).
    """
    me = m * eps
    steep = me > 1.0
    ratio = me / jnp.where(steep, me - 1.0, 1.0)
    x_peak = ratio * (dz_top - ht0)
    return jnp.where(steep & (x_peak < -h_min), -x_peak, h_min)
