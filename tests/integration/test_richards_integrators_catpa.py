"""The registered Richards integrators on CA-TPA 2015-2023 against the converged reference.

The data tier of the integrator conformance (:mod:`agrijax.testing.conformance.integrators`): every
integrator case of the kit, and the kit's reference scheme itself, runs the CA-TPA M1 replay as nine
restart year lanes in one ``vmap`` (:func:`richards_adaptive_years.year_lanes`); compared with
``validation/w1_e0/ref/CA-TPA_m1_fix_cnfb_960.npz`` (the converged reference: daily storage and water
contents, heads from ``h(theta)``) on the clean years of ``e0_reference_years.csv``, and the nine-year
cumulative water balance in two views: the method's conserved quantity (the sum of the daily
``balance_error`` of the state ``theta``) and the physical ``theta(h)`` of the heads (per lane: the
storage of ``theta(h)`` at the end minus at the start minus the booked fluxes; summed in absolute value
over the lanes). The declared tolerances are the measured values (``validation/w1_d``), rounded up in
the second digit. The years the converged reference itself did not converge on (CA-TPA 2020: 1 unconverged
step, 3 fallbacks) are reported (``storage_unclean_cm``, ``head_rel_unclean``; printed and written to
``W1D_METRICS``), not compared: there the reference is not converged, so it sets no tolerance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from richards_adaptive_years import e0_reference, replay_for, year_lanes

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the CA-TPA data and the converged-reference outputs"),
]

#: case name -> (max |storage - reference| over the clean years [cm], max head difference relative
#: to max(|h_ref|, hb), nine-year |balance| method view [cm], theta(h) view [cm]); measured values
DATA_TOL: dict[str, tuple[float, float, float, float]] = {
    # measured: 1.50e-4, 1.77e-3, 1.49e-11, 1.49e-11 (rounding-level balances: next power of ten)
    "faithful_exact": (1.6e-4, 1.8e-3, 1e-10, 1e-10),
    # 9.05e-4, 5.72e-3, 9.44e-12, 9.44e-12
    "faithful_fast": (9.1e-4, 5.8e-3, 1e-11, 1e-11),
    # 0.253, 2.92, 0.356, 0.356 (the unconverged residual of 3 iterations)
    "fixed_24x3": (0.26, 3.0, 0.36, 0.36),
    # 0.0440, 0.130, 6.25e-6, 6.25e-6
    "fixed_96x8": (0.044, 0.13, 6.3e-6, 6.3e-6),
    # the forward of fixed_96x8 (grad="implicit" changes only the reverse pass)
    "fixed_96x8_ift": (0.044, 0.13, 6.3e-6, 6.3e-6),
    # 1.90e-3, 0.0169, 0.359, 0.359
    "fixed_cn_96x8": (1.9e-3, 0.017, 0.36, 0.36),
    # the kit's reference is the converged reference: 3.4e-13, 4.3e-13, 8.1e-12, 8.1e-12
    "reference": (1e-12, 1e-12, 1e-11, 1e-11),
}


def _run(rep: Any, stepping: Any) -> dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp
    from jax import lax

    from agrijax.processes.soil_water.hydraulics import theta_of_h
    from agrijax.processes.soil_water.richards import RichardsParams, SoilWater, richards_day

    lanes = year_lanes(rep)
    params = RichardsParams(soil=rep.soil, grid=rep.grid, stepping=stepping)
    grid, soil = rep.grid, rep.soil
    nodes = soil.at_nodes()

    def lane(th0: Any, sup: Any, eva: Any, upt: Any) -> Any:
        w0 = SoilWater.from_theta(th0, soil)

        def body(w: Any, f: Any) -> Any:
            w2 = richards_day(w, params, *f)
            fl = w2.flux
            out = fl.evaporation + fl.drainage + fl.uptake + fl.runoff
            return w2, {
                "storage": w2.storage(grid),
                "storage_h": jnp.sum(theta_of_h(w2.h, nodes) * grid.tl),
                "theta": w2.theta,
                "h": w2.h,
                "balance_error": fl.balance_error,
                "booked": jnp.sum(f[0]) - out,
            }

        _, o = lax.scan(body, w0, (sup, eva, upt))
        o["storage_h0"] = jnp.sum(theta_of_h(w0.h, nodes) * grid.tl) + w0.pond
        return o

    args = tuple(jnp.asarray(x) for x in (lanes.th0, lanes.sup, lanes.eva, lanes.upt))
    out = jax.jit(jax.vmap(lane))(*args)
    return {k: np.asarray(v) for k, v in out.items()} | {"idx": lanes.idx}


def metrics(rep: Any, ref: Any, r: dict[str, np.ndarray]) -> dict[str, float]:
    from agrijax.processes.soil_water.hydraulics import h_of_theta

    m = r["idx"] >= 0
    order = np.argsort(r["idx"][m])
    storage, h = r["storage"][m][order], r["h"][m][order]
    clean = np.isin(rep.years, [y for y, c in ref.clean.items() if c])
    h_ref = np.asarray(h_of_theta(np.asarray(ref.theta), rep.soil.at_nodes()))
    hb = np.asarray(rep.soil.at_nodes().hb)
    head = np.abs(h - h_ref) / np.maximum(np.abs(h_ref), hb)
    bal = np.where(m, r["balance_error"], 0.0).sum(axis=1)
    last = m.sum(axis=1) - 1
    s_end = r["storage_h"][np.arange(len(last)), last]
    phys = s_end - r["storage_h0"] - np.where(m, r["booked"], 0.0).sum(axis=1)
    unclean = ~clean
    return {
        "storage_unclean_cm": float(np.max(np.abs(storage - ref.storage)[unclean], initial=0.0)),
        "head_rel_unclean": float(np.max(head[unclean], initial=0.0)),
        "storage_cm": float(np.max(np.abs(storage - ref.storage)[clean])),
        "head_rel": float(np.max(head[clean])),
        "balance_method_cm": float(np.sum(np.abs(bal))),
        "balance_theta_h_cm": float(np.sum(np.abs(phys))),
        "n_clean_days": int(clean.sum()),
    }


def _cases() -> list[tuple[str, Any]]:
    from agrijax.testing.conformance import integrators as K

    return [(c.name, c.stepping) for c in K.integrator_cases()] + [("reference", K.REFERENCE)]


@pytest.fixture(scope="module")
def catpa(data_dir: Path) -> tuple[Any, Any]:
    ref = e0_reference(data_dir, "CA-TPA")
    if ref is None:
        pytest.skip("no converged reference in the data tree")
    return replay_for("CA-TPA", data_dir), ref


@pytest.mark.parametrize(("name", "stepping"), _cases(), ids=[n for n, _ in _cases()])
def test_integrator_against_the_e0_reference_on_nine_years(
    catpa: tuple[Any, Any], name: str, stepping: Any
) -> None:
    rep, ref = catpa
    m = metrics(rep, ref, _run(rep, stepping))
    print(name, m)
    out = os.environ.get("W1D_METRICS")
    if out:
        with open(out, "a") as fh:
            fh.write(f"{name},{m['storage_cm']:.6g},{m['head_rel']:.6g},{m['balance_method_cm']:.6g},"
                     f"{m['balance_theta_h_cm']:.6g},{m['n_clean_days']},{m['storage_unclean_cm']:.6g},"
                     f"{m['head_rel_unclean']:.6g}\n")  # fmt: skip
    tol = DATA_TOL[name]
    got = (m["storage_cm"], m["head_rel"], m["balance_method_cm"], m["balance_theta_h_cm"])
    assert all(g <= t for g, t in zip(got, tol, strict=True)), (name, got, tol)
