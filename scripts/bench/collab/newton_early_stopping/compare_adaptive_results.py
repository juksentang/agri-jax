"""Recheck the saved experiment results; no model runs or timings here."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    out = Path(__file__).resolve().parent
    read = lambda name: json.loads((out / name).read_text(encoding="utf-8"))
    results = {variant: read(f"adaptive_{variant}_n12_tol1e-14_final.json")
               for variant in ("adaptive", "fixed", "unrolled")}
    adaptive = results["adaptive"]
    flatten = lambda row: np.concatenate([np.asarray(x).ravel() for x in row["gradient_leaves"]])
    ag = flatten(adaptive)
    checks = {}
    for variant in ("fixed", "unrolled"):
        ref = results[variant]
        rg = flatten(ref)
        np.testing.assert_allclose(ag, rg, rtol=1e-7, atol=1e-10)
        np.testing.assert_allclose(adaptive["loss"], ref["loss"], rtol=1e-12, atol=1e-11)
        np.testing.assert_allclose(adaptive["diagnostics"]["final_theta"],
            ref["diagnostics"]["final_theta"], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(adaptive["diagnostics"]["final_head"],
            ref["diagnostics"]["final_head"], rtol=1e-10, atol=1e-9)
        checks[variant] = dict(relative_gradient_l2=float(np.linalg.norm(ag-rg)/np.linalg.norm(rg)),
            maximum_absolute_gradient_difference=float(np.max(np.abs(ag-rg))),
            loss_absolute_difference=abs(adaptive["loss"]-ref["loss"]))
    rounds = []
    for tag in ("final", "repeat"):
        a = read(f"adaptive_adaptive_n12_tol1e-14_{tag}.json")
        f = read(f"adaptive_fixed_n12_tol1e-14_{tag}.json")
        np.testing.assert_allclose(flatten(a), ag, rtol=1e-7, atol=1e-10)
        assert a["accepted"] and f["accepted"]
        assert all(x["passed"] for x in a["finite_difference_checks"] + f["finite_difference_checks"])
        rounds.append(dict(round=tag, adaptive=a["timings"], fixed=f["timings"],
            speedups={metric: f["timings"][metric]["median_s"] / a["timings"][metric]["median_s"]
                      for metric in ("forward", "value_and_grad")}))
    audit = read("adaptive_adaptive_n12_tol1e-14_audit_final.json")
    assert len(audit["audit"]["forward"]) == len(audit["audit"]["backward"]) == 168
    assert all(x["accepted"] for x in audit["audit"]["backward"])
    assert max(x["residual"] for x in audit["audit"]["forward"]) <= 1e-14
    failures = read("check_adaptive_failure.json")
    assert all(x["passed"] for x in failures["checks"])
    assert adaptive["diagnostics"]["max_abs_daily_balance_cm"] < 1e-10
    scripts = ("experiment_adaptive_richards.py", "check_adaptive_failure.py", "compare_adaptive_results.py")
    result = dict(passed=True, gradient_components=len(ag), gradient_comparisons=checks,
        diagnostics=adaptive["diagnostics"], iteration_summary=audit["iteration_summary"],
        max_adjoint_backward_error=max(x["adjoint_backward_error"] for x in audit["audit"]["backward"]),
        max_finite_difference_relative_error=max(x["relative_error"] for x in adaptive["finite_difference_checks"]),
        timing_rounds=rounds, failure_checks=failures,
        provenance=dict(repository_commit=read("provenance.json")["source_repo_commit"],
            script_sha256={name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in scripts}))
    (out/"adaptive_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("timing_rounds", "diagnostics", "provenance")}, indent=2))


if __name__ == "__main__":
    main()
