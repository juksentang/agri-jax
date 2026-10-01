"""Green-Ampt event on post-tillage soil (:class:`TilledSoilHydraulicParams`), no data.

After tillage RZWQM2 reads the initial suction of the wetting front from the two-segment
retention curve (``WCH`` with the start-up ``TRHYDP`` at the dry end) and everything else in the
event from the current ``SOILHP`` (``EVNTRO``). Checked against independent references:

* an untilled soil wrapped as ``TilledSoilHydraulicParams.untilled`` gives the event of the plain
  parameters bit for bit (the wrapper changes nothing by itself);
* the suction on a tilled soil equals ``1 + int_1^{s_i} K_c(-s)/K_s,c ds`` by quadrature, with
  ``s_i`` the two-segment head and ``K_c`` the current conductivity (dry end: the start-up
  retention branch; wet end: the current one);
* the tilled event conserves mass (rain = infiltration + runoff + seepage, storage change =
  infiltration) and fills to the current available porosity ``theta_s,c AEF``.

The event itself against RZWQM2 on tilled soil is ``tests/integration/test_green_ampt_scenarios.py``
(11 of the 15 ``RZWQM_sw_batch`` scenarios have storms on tilled soil).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from agrijax.processes.soil_water import infiltration as G
from agrijax.processes.soil_water.hydraulics import (
    SoilHydraulicParams,
    TilledSoilHydraulicParams,
    h_of_theta,
    k_of_h,
)

X64 = bool(jax.config.read("jax_enable_x64"))
AEF = 0.9
N = 10
TL = np.full(N, 2.0)
#: CA-TPA horizon 1 (rzwqm.dat) as the start-up curve; the "tilled" curve has a lower bubbling
#: pressure and a higher porosity (tillage loosens the soil), as the RZWQM2 tillage adjustment does
REC1_ORIG = np.array([14.6545, 0.22, 2.966, 5.41, 0.055, 0.453])
REC1_TILL = np.array([9.5, 0.22, 2.966, 7.8, 0.055, 0.49])


def _params(rec1: np.ndarray) -> SoilHydraulicParams:
    rec2 = np.array([0.0, 0.0, 0.0, rec1[0], 0.0, 0.0, 0.0])
    s = SoilHydraulicParams.from_rzwqm_records(rec1, rec2)
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(jnp.asarray(x), (N,)), s)


CUR = _params(REC1_TILL)
ORIG = _params(REC1_ORIG)
TILLED = TilledSoilHydraulicParams(current=CUR, original=ORIG)


def _run(theta: np.ndarray, soil, depth: float = 3.0, hours: float = 1.0) -> G.GAResult:
    th = jnp.asarray(theta)
    cfg = G.GreenAmptConfig.for_grid(TL)
    return G.green_ampt_event(
        th, h_of_theta(th, soil), soil, jnp.asarray(TL), jnp.asarray(AEF), jnp.asarray([hours]),
        jnp.asarray([depth]), cfg,
    )  # fmt: skip


def test_untilled_wrapper_is_bit_identical() -> None:
    theta = np.linspace(0.12, 0.30, N)
    a = _run(theta, CUR, depth=6.0, hours=0.5)
    b = _run(theta, TilledSoilHydraulicParams.untilled(CUR), depth=6.0, hours=0.5)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert float(a.runoff) > 0.0  # the storm runs off, so the capacity is exercised


@pytest.mark.parametrize("theta_i", [0.08, 0.12, 0.2, 0.3, 0.4])
def test_tilled_suction_equals_quadrature(theta_i: float) -> None:
    th = jnp.full(N, theta_i)
    sf = float(G.wetting_front_suction(th, TILLED, CUR.theta_s * AEF)[0])
    s_i = -float(h_of_theta(jnp.minimum(th, CUR.theta_s * AEF), TILLED)[0])
    one = jax.tree_util.tree_map(lambda x: x[:1], CUR)
    ks = REC1_TILL[3]
    # int_1^{s_i} K/Ks ds in log space (s = e^u): smooth on both power-law segments, exact to 1e-12
    kk = lambda u: float(k_of_h(jnp.asarray([-np.exp(u)]), one)[0]) / ks * np.exp(u)  # noqa: E731
    lo, hi, kink = 0.0, float(np.log(max(s_i, 1.0))), float(np.log(REC1_TILL[0]))
    pts = [kink] if lo < kink < hi else None
    ref = 1.0 + (quad(kk, lo, hi, points=pts, limit=200, epsabs=0.0, epsrel=1e-12)[0] if s_i > 1 else 1.0)
    assert sf == pytest.approx(ref, rel=1e-9 if X64 else 1e-5)


def test_tilled_suction_uses_the_start_up_curve_at_the_dry_end() -> None:
    dry, wet = jnp.full(N, 0.25), jnp.full(N, 0.40)  # below / above theta_orig(-10 hb_orig) = 0.295
    avail = CUR.theta_s * AEF
    h_t, h_c = h_of_theta(dry, TILLED), h_of_theta(dry, CUR)
    np.testing.assert_array_equal(np.asarray(h_t), np.asarray(h_of_theta(dry, ORIG)))  # start-up branch
    assert float(jnp.max(jnp.abs(h_t - h_c))) > 1.0
    s_t = G.wetting_front_suction(dry, TILLED, avail)
    s_c = G.wetting_front_suction(dry, CUR, avail)
    assert float(jnp.max(jnp.abs(s_t - s_c))) > 1e-6 * float(jnp.max(s_c))
    np.testing.assert_array_equal(
        np.asarray(G.wetting_front_suction(wet, TILLED, avail)),
        np.asarray(G.wetting_front_suction(wet, CUR, avail)),
    )


def test_tilled_event_conserves_mass_and_fills_to_current_porosity() -> None:
    theta = np.full(N, 0.10)
    r = _run(theta, TILLED, depth=8.0, hours=0.5)
    assert abs(float(r.error)) < (1e-12 if X64 else 1e-5)
    stored = float(jnp.sum((r.theta - jnp.asarray(theta)) * jnp.asarray(TL)))
    assert stored == pytest.approx(float(r.infiltration), abs=1e-12 if X64 else 1e-5)
    assert float(jnp.max(r.theta)) == pytest.approx(REC1_TILL[5] * AEF, rel=1e-12 if X64 else 1e-6)
    assert float(r.runoff) > 0.0
