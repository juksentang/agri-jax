"""Calibration with ``method="adam"`` before / after the split level-2 trust test.

``agrijax.calib.calibrate(..., method="adam")`` is the only calibration path that calls
``agrijax.calib.trust.trust_report`` (its gradient plan decides which coefficients Adam moves). "before"
runs it with the legacy level 2 (the straight-through derivative against the model's small-step
difference: the trust report gets the loss as a plain function, without counterparts); "after" with the
mode-bound loss (exact-mode and unrounded-model paths). Water-limited UFGA8201 treatments, 2 starts,
20 Adam steps; prints the plan, the calibrated coefficients and the loss of each.

    python scripts/diag/adam_trust_before_after.py
"""

from __future__ import annotations

import os

import jax

jax.config.update("jax_enable_x64", True)

import agrijax.calib.trust as T  # noqa: E402
from agrijax.calib import calibrate  # noqa: E402
from agrijax.calib.ceres import ceres_gradient_plan  # noqa: E402

ORIG = T.trust_report
MODE = {"which": "after"}


def wrapped(f, x, lo, hi, pn, on, cfg=T.DEFAULT_TRUST, **kw):
    rep = (
        ORIG(f, x, lo, hi, pn, on, cfg, **kw)
        if MODE["which"] == "after"
        else ORIG(lambda th: f(th), x, lo, hi, pn, on, cfg)
    )
    for p in ("G2", "G3"):
        for o, r in rep["params"][p]["outputs"].items():
            print(
                f"  {MODE['which']:6s} {p} {o:14s} L{r['level']} {r['class']:7s} ad={r['ad']:.6g} "
                f"ad_exact={r['ad_exact']:.6g} err_small={r['rel_err_small']:.2e} "
                f"err_small_unrounded={r.get('rel_err_small_unrounded', float('nan')):.2e} "
                f"err_large={r['rel_err_large']:.2e}"
            )
    print(f"  {MODE['which']:6s} plan:", ceres_gradient_plan(rep).method, flush=True)
    return rep


T.trust_report = wrapped

for trs in ([4], [6], [4, 6]):
    for which in ("before", "after"):
        MODE["which"] = which
        print(f"=== UFGA8201 treatments {trs} ({which})", flush=True)
        try:
            res = calibrate(
                "UFGA8201",
                treatments=trs,
                method="adam",
                starts=2,
                budget=20,
                data_dir=os.environ["AGRI_JAX_DATA"],
            )
            print(f"  {which:6s} calibrated: {res.free} loss: {res.loss}", flush=True)
        except Exception as e:  # a refused calibration is a result here
            print(f"  {which:6s} refused: {type(e).__name__}: {str(e)[:300]}", flush=True)
