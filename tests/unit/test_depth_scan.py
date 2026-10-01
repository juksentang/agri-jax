"""``agrijax.core.depth_scan`` (the sanctioned depth recurrence) and lint rule AJ006 (unrolled layer loops).

Independent reference: a plain Python loop over the depth entries (NumPy), for a nonlinear
recurrence whose step depends on the carried value, with and without a mask and in reverse.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import lint
from agrijax.core.depth_scan import depth_scan

X64 = bool(jax.config.read("jax_enable_x64"))
TOL = 1e-13 if X64 else 1e-5


def _step(c, x):
    """A carry-dependent recurrence: the rate at depth k depends on what was consumed above."""
    used, left = c
    cap, need = x
    rate = jnp.where(left > 1.0, cap, 0.5 * cap)
    take = jnp.where(need < left, need, left) * rate / (1.0 + rate)
    return (used + take, left - take), take


def _reference(cap, need, left0, mask=None, reverse=False):
    used, left = 0.0, left0
    out = np.zeros(len(cap))
    order = range(len(cap) - 1, -1, -1) if reverse else range(len(cap))
    for k in order:
        if mask is not None and not mask[k]:
            continue
        rate = cap[k] if left > 1.0 else 0.5 * cap[k]
        take = min(need[k], left) * rate / (1.0 + rate)
        used, left = used + take, left - take
        out[k] = take
    return used, left, out


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_depth_scan_equals_python_loop(reverse: bool, masked: bool) -> None:
    rng = np.random.default_rng(5)
    cap, need = rng.uniform(0.1, 3.0, 40), rng.uniform(0.0, 0.8, 40)
    mask = rng.random(40) > 0.3 if masked else None
    (used, left), ys = depth_scan(
        _step,
        (jnp.asarray(0.0), jnp.asarray(6.0)),
        (jnp.asarray(cap), jnp.asarray(need)),
        mask=None if mask is None else jnp.asarray(mask),
        reverse=reverse,
    )
    u_ref, l_ref, y_ref = _reference(cap, need, 6.0, mask, reverse)
    assert float(used) == pytest.approx(u_ref, rel=TOL, abs=TOL)
    assert float(left) == pytest.approx(l_ref, rel=TOL, abs=TOL)
    np.testing.assert_allclose(ys, y_ref, rtol=TOL, atol=TOL)
    if mask is not None:
        assert np.all(np.asarray(ys)[~mask] == 0.0)


def test_depth_scan_under_jit_vmap_and_grad() -> None:
    cap = jnp.linspace(0.2, 2.0, 16)
    need = jnp.full(16, 0.3)

    def total(left0, c):
        (used, _), _ = depth_scan(_step, (jnp.zeros(()), left0), (c, need))
        return used

    batch = jnp.asarray([2.0, 4.0, 8.0])
    v = jax.jit(jax.vmap(lambda l0: total(l0, cap)))(batch)
    for k, l0 in enumerate(batch):
        assert float(v[k]) == pytest.approx(
            _reference(np.asarray(cap), np.asarray(need), float(l0))[0], rel=TOL
        )
    g = jax.grad(total, argnums=1)(jnp.asarray(4.0), cap)
    assert np.all(np.isfinite(np.asarray(g)))
    # finite differences on one depth entry (the recurrence is smooth away from the left > 1 switch)
    if X64:
        e = 1e-6
        fd = (total(4.0, cap.at[3].add(e)) - total(4.0, cap.at[3].add(-e))) / (2 * e)
        assert float(g[3]) == pytest.approx(float(fd), rel=1e-6)


def test_depth_scan_validates_lengths() -> None:
    with pytest.raises(ValueError, match="inconsistent depth lengths"):
        depth_scan(_step, (0.0, 1.0), (jnp.ones(4), jnp.ones(5)))
    with pytest.raises(ValueError, match="mask shape"):
        depth_scan(_step, (0.0, 1.0), (jnp.ones(4), jnp.ones(4)), mask=jnp.ones(3, bool))
    with pytest.raises(ValueError, match="leading depth axis"):
        depth_scan(_step, (0.0, 1.0), (jnp.ones(4), jnp.asarray(1.0)))
    # no leaves: the length is given explicitly
    c, ys = depth_scan(lambda c, _: (c + 1.0, c), jnp.asarray(0.0), None, length=5)
    assert float(c) == 5.0 and ys.shape == (5,)


# ---------------------------------------------------------------------------
# AJ006: unrolled loops over shape-derived ranges in kernels
# ---------------------------------------------------------------------------

AJ006_KERNELS = """
import jax.numpy as jnp


def by_shape(theta, soil):
    out = []
    for i in range(theta.shape[0]):
        out.append(theta[i] * soil.ksat[i])
    return jnp.stack(out)


def by_len(h):
    return [h[i] + 1.0 for i in range(len(h))]


def by_grid(state, grid):
    n = grid.n_node
    acc = 0.0
    for k in range(n - 1):
        acc = acc + state.theta[k]
    return acc


def by_np_shape(x):
    m = np.shape(x)[-1]
    return sum(x[j] for j in jnp.arange(m))


def over_array(state, grid):
    return [t * 2.0 for t in state.theta]


def by_enumerate(state, grid):
    acc = 0.0
    for i, dz in enumerate(grid.tl):
        acc = acc + dz * i
    return acc


def by_len_of_local(theta):
    th = theta * 2.0
    for i in range(len(th)):
        th = th + i
    return th


def over_local_array(theta):
    w = jnp.cumsum(theta)
    return [v + 1.0 for v in w]
"""

AJ006_ALLOWED = """
import jax.numpy as jnp


def allowed(x, cfg, names):
    seeds = [x * k for k in range(3)]
    for _ in range(cfg.n_iter):
        x = x + 1.0
    rows = ["a", "b"]
    for i in range(len(rows)):
        x = x + i
    for name in names:
        x = x + 1.0
    return x, seeds
"""


def _kernel_file(tmp_path: Path, src: str) -> Path:
    f = tmp_path / "processes" / "soil_water" / "k.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(textwrap.dedent(src))
    return f


def test_aj006_flags_loops_over_shape_derived_ranges(tmp_path: Path) -> None:
    found = [x for x in lint.lint_file(_kernel_file(tmp_path, AJ006_KERNELS)) if x.rule == "AJ006"]
    lines = sorted(x.line for x in found)
    assert len(found) == 8, [x.format() for x in found]
    assert all(x.level == "error" for x in found)
    src_lines = AJ006_KERNELS.splitlines()
    for ln in lines:
        assert any(k in src_lines[ln - 1] for k in ("range(", " in state.theta", "enumerate(", " in w"))
    assert "AJ006" in lint.KERNEL_RULES and "AJ006" in lint.RULES


def test_aj006_allows_literal_and_configuration_counts(tmp_path: Path) -> None:
    found = lint.lint_file(_kernel_file(tmp_path, AJ006_ALLOWED))
    assert not [x for x in found if x.rule == "AJ006"], [x.format() for x in found]


def test_aj006_not_applied_outside_processes_unless_all(tmp_path: Path) -> None:
    g = tmp_path / "io" / "k.py"
    g.parent.mkdir()
    g.write_text(textwrap.dedent(AJ006_KERNELS))
    assert lint.lint_file(g) == []
    assert {x.rule for x in lint.lint_file(g, all_functions=True)} >= {"AJ006"}


def test_package_has_no_unrolled_layer_loops() -> None:
    src = Path(__file__).resolve().parents[2] / "src" / "agrijax"
    found = [x for x in lint.lint_paths([src]) if x.rule == "AJ006"]
    assert not found, [x.format() for x in found]
