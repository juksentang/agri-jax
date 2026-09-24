"""Regression baseline of the validated CERES-Maize implementation (commit ``b65f1a7``).

The CERES-Maize module (``agrijax.processes.crop.ceres_maize``) was validated at ``b65f1a7``
against DSSAT-CSM ``dscsm048`` v4.8.6 on 58 maize treatments (nitrogen off, crop driven by the
reference run's weather, soil water and water-stress factors). This module freezes what that
implementation computes, so a refactor can be checked to change nothing:

* for every one of the 58 treatments, driven exactly as ``test_ceres_dssat.py`` drives them
  (:func:`test_ceres_dssat.run_reference` + :func:`test_ceres_dssat.simulate`, the latter with
  ``daylength_from_output`` for GAGR0201): every leaf of the parameters, of the daily forcing, of
  the daily CERES state pytree (the state *after* each day) and of the daily ``PlantGro``
  outputs (:func:`plantgro_outputs`);
* for :data:`GRAD_CASES`: reverse-mode gradients (``jax.jacrev``) of the season yield
  (``gwad`` on the last day), the maximum LAI over the season (``max_t lai``) and the final
  above-ground biomass (``cwad`` on the last day) with respect to every field of the cultivar and
  species parameters.

Array keys in the snapshot (``.npz``, float leaves in float64, integer / boolean leaves in their
own dtype, every trajectory ``[T, ...]``)::

    <case>/params/<path>        e.g. UFGA8201_t1/params/cultivar.p1
    <case>/forcing/<field>      e.g. UFGA8201_t1/forcing/tmax
    <case>/state/<path>         e.g. UFGA8201_t1/state/growth.leaf.area
    <case>/out/<name>           e.g. UFGA8201_t1/out/cwad
    <case>/grad/<target>/<path> e.g. UFGA8201_t1/grad/yield/cultivar.g3   (GRAD_CASES only)
    __manifest__                JSON: git hash, case ids, field names, targets, versions

``<case>`` is ``<EXPERIMENT>_t<TRNO>``. A JSON sidecar ``<snapshot>.json`` repeats the manifest
and holds the sha256 of the ``.npz``.

Generate (local only: the dscsm048 binary and the DSSAT engine tree exist only locally)::

    uv run python tests/integration/ceres_baseline.py            # refuses unless HEAD == b65f1a7
                                                                  # and src/ is unmodified

DSSAT-CSM is distributed under the BSD 3-clause licence (Copyright 1998-2026 DSSAT Foundation,
University of Florida, International Fertilizer Development Center); the model here is an
independent implementation from its published equations.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "b65f1a7"
SNAPSHOT_NAME = f"baseline_{BASELINE_COMMIT}.npz"
MANIFEST_KEY = "__manifest__"

#: the 58 validated treatments (``m2_all_maize.csv`` in the validation archive): every treatment of
#: every maize example experiment except EBPL8501 (missing EBCH8401.WTH) and IUAF9902 / IUAF9903
#: (pest damage, not ported); the same set ``test_every_maize_example_experiment`` enumerates.
SKIP_EXPERIMENTS = frozenset({"EBPL8501", "IUAF9902", "IUAF9903"})
CASES: tuple[tuple[str, int], ...] = (
    *(("BRPI0202", t) for t in range(1, 9)),
    *(("FLSC8101", t) for t in range(1, 3)),
    *(("GAGR0201", t) for t in range(1, 7)),
    *(("GHWA0401", t) for t in range(1, 10)),
    *(("IBWA8301", t) for t in range(1, 7)),
    *(("IUAF9901", t) for t in range(1, 5)),
    *(("SIAZ9501", t) for t in range(1, 9)),
    *(("SIAZ9601", t) for t in range(1, 10)),
    *(("UFGA8201", t) for t in range(1, 7)),
)
#: Gainesville 1982 rainfed, and Ames 1999 (cold spring, waterlogging days: SATFAC > 0)
GRAD_CASES: tuple[tuple[str, int], ...] = (("UFGA8201", 1), ("IUAF9901", 1))
GRAD_TARGETS = ("yield", "max_lai", "final_cwad")
#: experiments whose daylength is replaced by an environment modification (read from Weather.OUT)
DAYLENGTH_FROM_OUTPUT = frozenset({"GAGR0201"})

ATOL = 1e-12
RTOL = 1e-12
_GRAD_DAY = 1 << 62  # sort key of gradient arrays in compare_case: after every day


def case_id(exp: str, trno: int) -> str:
    return f"{exp}_t{trno}"


def snapshot_path(data_dir: Path) -> Path:
    return Path(data_dir) / "validation" / "ceres" / SNAPSHOT_NAME


def enumerate_cases(maize_dir: Path) -> list[tuple[str, int]]:
    """Every treatment of every ``*.MZX`` under ``maize_dir`` (as the all-maize test enumerates)."""
    cases = []
    for x in sorted(Path(maize_dir).glob("*.MZX")):
        if x.stem in SKIP_EXPERIMENTS:
            continue
        sec = x.read_text(errors="replace").split("*TREATMENTS")[1].split("*")[0]
        cases += [(x.stem, int(ln[:3])) for ln in sec.splitlines() if ln[:3].strip().isdigit()]
    return cases


# ------------------------------------------------------------------------------ flattening
def _path_str(path: tuple[Any, ...]) -> str:
    parts = []
    for k in path:
        if isinstance(k, jax.tree_util.GetAttrKey):
            parts.append(k.name)
        elif isinstance(k, jax.tree_util.DictKey):
            parts.append(str(k.key))
        elif isinstance(k, jax.tree_util.SequenceKey):
            parts.append(str(k.idx))
        else:
            parts.append(str(k))
    return ".".join(parts)


def flatten(tree: Any) -> dict[str, np.ndarray]:
    """``{dotted path: array}`` for every array leaf of a pytree, in pytree order.

    Floating leaves are stored in float64, integer and boolean leaves in their own dtype."""
    out: dict[str, np.ndarray] = {}
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a = np.asarray(leaf)
        if np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float64)
        out[_path_str(path)] = a
    return out


# ------------------------------------------------------------------------------ the model runs
def _drivers() -> Any:
    """The helpers of ``test_ceres_dssat`` (imported lazily: it needs the tests directory)."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import test_ceres_dssat as t

    return t


def drivers_available() -> tuple[bool, str]:
    t = _drivers()
    if not t.DSCSM.is_file() or not t.MAIZE.is_dir():
        return False, f"dscsm048 / DSSAT example data not found under {t.DSSAT_ENGINE}"
    return True, ""


def _trajectory_model() -> Any:
    from agrijax.processes.crop.ceres_maize import ceres_maize_model, plantgro_outputs

    return ceres_maize_model(outputs=lambda s, p, f: {"state": s, "out": plantgro_outputs(s, p, f)})


_TRAJ: Callable[..., Any] | None = None
_GRAD: Callable[..., Any] | None = None


def _traj_runner() -> Callable[..., Any]:
    global _TRAJ
    if _TRAJ is None:
        from agrijax.core import run

        model = _trajectory_model()
        _TRAJ = jax.jit(lambda p, f, s: run(model, p, f, s))
    return _TRAJ


def _season_targets(p: Any, f: Any) -> jax.Array:
    from agrijax.core import run
    from agrijax.processes.crop.ceres_maize import CeresMaizeState, ceres_maize_model

    out = run(ceres_maize_model(), p, f, CeresMaizeState.initial(p, 1))
    return jnp.stack([out["gwad"][-1, 0], jnp.max(out["lai"][:, 0]), out["cwad"][-1, 0]])


def _grad_runner() -> Callable[..., Any]:
    global _GRAD
    if _GRAD is None:

        def targets(cs: tuple[Any, Any], p: Any, f: Any) -> jax.Array:
            return _season_targets(p.replace(cultivar=cs[0], species=cs[1]), f)

        _GRAD = jax.jit(jax.jacrev(targets))
    return _GRAD


def compute_case(
    exp: str, trno: int, workdir: Path, *, gradients: bool | None = None
) -> dict[str, np.ndarray]:
    """Run dscsm048 for one treatment, then the model; return the snapshot arrays of that case.

    The model runs on the CPU backend whatever the default device (a GPU's float64
    transcendentals need not match the CPU's to 1e-12), so the snapshot and every regeneration
    compare like with like."""
    with jax.default_device(jax.devices("cpu")[0]):
        return _compute_case(exp, trno, workdir, gradients=gradients)


def _compute_case(exp: str, trno: int, workdir: Path, *, gradients: bool | None) -> dict[str, np.ndarray]:
    from agrijax.processes.crop.ceres_maize import CeresMaizeState

    t = _drivers()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("the CERES baseline is float64: enable jax_enable_x64")
    out_dir = t.run_reference(exp, trno, Path(workdir))
    # exactly the validated path: parameters, forcing and the default outputs of simulate()
    p, f, res, _ = t.simulate(out_dir, trno, daylength_from_output=exp in DAYLENGTH_FROM_OUTPUT)
    traj = _traj_runner()(p, f, CeresMaizeState.initial(p, 1))
    cid = case_id(exp, trno)
    arrays: dict[str, np.ndarray] = {}
    for k, v in flatten(p).items():
        arrays[f"{cid}/params/{k}"] = v
    for k, v in flatten(f).items():
        arrays[f"{cid}/forcing/{k}"] = v
    for k, v in flatten(traj["state"]).items():
        arrays[f"{cid}/state/{k}"] = v
    out = flatten(traj["out"])
    # the trajectory runner must reproduce the validated runner's outputs bit for bit
    for k, v in res.items():
        if not np.array_equal(np.asarray(v), out[k]):
            raise AssertionError(f"{cid}: trajectory runner output {k!r} differs from simulate()")
    for k, v in out.items():
        arrays[f"{cid}/out/{k}"] = v
    if gradients if gradients is not None else (exp, trno) in GRAD_CASES:
        jac = _grad_runner()((p.cultivar, p.species), p, f)
        for part, tree in zip(("cultivar", "species"), jac):
            for k, v in flatten(tree).items():  # leaf shape [3, *param_shape]
                for i, target in enumerate(GRAD_TARGETS):
                    arrays[f"{cid}/grad/{target}/{part}.{k}"] = v[i]
    return arrays


def iter_cases(workroot: Path, cases: tuple[tuple[str, int], ...] = CASES) -> Iterator[dict[str, np.ndarray]]:
    for exp, trno in cases:
        d = Path(workroot) / f"{exp}_{trno}"
        yield compute_case(exp, trno, d)


# ------------------------------------------------------------------------------ comparison
def compare_case(
    cid: str,
    got: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> str | None:
    """``None`` when every snapshot array of case ``cid`` equals ``got``; else a message naming
    the earliest differing day, the first field that differs on it and every differing field.

    Floats: ``|got - ref| <= atol + rtol |ref|`` (NaN equal to NaN); integers / booleans: exact."""
    prefix = f"{cid}/"
    coef = f"{cid}/params/coefficients"  # calibratable coefficient leaves, added after the snapshot
    keys_ref = sorted(k for k in ref if k.startswith(prefix))
    keys_got = sorted(k for k in got if k.startswith(prefix) and not k.startswith(coef))
    if keys_ref != keys_got:
        missing = sorted(set(keys_ref) - set(keys_got))
        extra = sorted(set(keys_got) - set(keys_ref))
        return f"{cid}: field set changed; missing {missing[:10]}, new {extra[:10]}"
    yrdoy = ref.get(f"{cid}/forcing/yrdoy")
    first: tuple[int, str, str] | None = None  # (day, key, detail)
    bad: list[str] = []
    for k in keys_ref:
        a, b = np.asarray(got[k]), np.asarray(ref[k])
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            return f"{cid}: {k!r} changed shape / kind: {a.shape} {a.dtype} vs snapshot {b.shape} {b.dtype}"
        if np.issubdtype(b.dtype, np.floating):
            ok = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        else:
            ok = a == b
        if np.all(ok):
            continue
        bad.append(k)
        trajectory = "/state/" in k or "/out/" in k or "/forcing/" in k
        idx = np.argwhere(~ok)[0] if ok.ndim else np.zeros(0, int)
        # parameters sort before every day (a changed input explains the rest), gradients after
        day = int(idx[0]) if trajectory and ok.ndim else (_GRAD_DAY if "/grad/" in k else -1)
        detail = f"index {tuple(int(i) for i in idx)}: got {a[tuple(idx)]!r}, snapshot {b[tuple(idx)]!r}"
        if first is None or day < first[0]:
            first = (day, k, detail)
    if not bad:
        return None
    assert first is not None
    day, key, detail = first
    when = (
        "a parameter (not a daily field)"
        if day < 0
        else "a season gradient (not a daily field)"
        if day == _GRAD_DAY
        else f"day {day}" + (f" (YRDOY {int(yrdoy[day])})" if yrdoy is not None and day < len(yrdoy) else "")
    )
    return (
        f"{cid}: first difference in {key!r} on {when}, {detail}; "
        f"{len(bad)} differing fields: {bad[:20]}{' ...' if len(bad) > 20 else ''}"
    )


# ------------------------------------------------------------------------------ writing
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def build_manifest(arrays: dict[str, np.ndarray], cases: tuple[tuple[str, int], ...]) -> dict[str, Any]:
    cid0 = case_id(*cases[0])
    g0 = case_id(*GRAD_CASES[0])

    def names(kind: str, cid: str) -> list[str]:
        pre = f"{cid}/{kind}/"
        return [k[len(pre) :] for k in arrays if k.startswith(pre)]

    return {
        "git_hash": BASELINE_COMMIT,
        "git_head": _git("rev-parse", "HEAD"),
        "model": "agrijax.processes.crop.ceres_maize (nitrogen off), driven by dscsm048 v4.8.6 outputs",
        "cases": [case_id(e, t) for e, t in cases],
        "n_days": {case_id(e, t): int(arrays[f"{case_id(e, t)}/forcing/yrdoy"].shape[0]) for e, t in cases},
        "grad_cases": [case_id(e, t) for e, t in GRAD_CASES],
        "grad_targets": {
            "yield": "out gwad on the last day [kg ha-1]",
            "max_lai": "max over days of out lai [m2 m-2]",
            "final_cwad": "out cwad on the last day [kg ha-1]",
        },
        "fields": {
            "params": names("params", cid0),
            "forcing": names("forcing", cid0),
            "state": names("state", cid0),
            "out": names("out", cid0),
            "grad": sorted(names("grad", g0)),
        },
        "tolerance": {"atol": ATOL, "rtol": RTOL, "dtype": "float64"},
        "versions": {"jax": jax.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
        "platform": "cpu (jax.default_device)",
    }


def write_snapshot(path: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> str:
    """Write the ``.npz`` (with the manifest as a JSON string) and its sidecar; return the sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = dict(arrays)
    blob[MANIFEST_KEY] = np.asarray(json.dumps(manifest, sort_keys=True))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **blob)  # type: ignore[arg-type]
    tmp.replace(path)
    digest = sha256(path)
    side = dict(manifest, sha256=digest, file=path.name, n_arrays=len(arrays))
    path.with_suffix(".json").write_text(json.dumps(side, indent=1, sort_keys=True) + "\n")
    return digest


def load_snapshot(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != MANIFEST_KEY}
        manifest = json.loads(str(z[MANIFEST_KEY]))
    return arrays, manifest


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Write the CERES-Maize regression baseline.")
    ap.add_argument("--data-dir", default=os.environ.get("AGRI_JAX_DATA", str(Path.home() / "agri_jax_data")))
    ap.add_argument("--force", action="store_true", help="write even if HEAD / src differ from the baseline")
    args = ap.parse_args(argv)
    jax.config.update("jax_enable_x64", True)

    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain", "--", "src")
    if (not head.startswith(BASELINE_COMMIT) or dirty) and not args.force:
        print(f"refusing: HEAD {head[:7]} (want {BASELINE_COMMIT}), src changes: {dirty!r}", file=sys.stderr)
        return 2
    ok, why = drivers_available()
    if not ok:
        print(why, file=sys.stderr)
        return 2
    found = enumerate_cases(_drivers().MAIZE)
    if tuple(found) != CASES:
        print(f"treatment list changed: {found}", file=sys.stderr)
        return 2

    arrays: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix="ceres_bl_") as tmp:
        for exp, trno in CASES:
            arrays.update(compute_case(exp, trno, Path(tmp) / f"{exp}_{trno}"))
            print(
                f"{case_id(exp, trno)}: {sum(1 for k in arrays if k.startswith(case_id(exp, trno)))} arrays"
            )
    manifest = build_manifest(arrays, CASES)
    path = snapshot_path(Path(args.data_dir).expanduser())
    digest = write_snapshot(path, arrays, manifest)
    print(f"{path}\nsha256 {digest}\n{len(arrays)} arrays, {path.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
