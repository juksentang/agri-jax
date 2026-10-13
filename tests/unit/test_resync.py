"""Tests for the daily resynchronised comparison helpers (agrijax.port.resync); data-free."""

from __future__ import annotations

import io
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.port import dumps, resync


class Toy(eqx.Module):
    """A bucket (``w``) and a counter (``n``); the step drains 10 % of ``w`` and adds rain."""

    w: jax.Array
    n: jax.Array


def step(s: Toy, rain: jax.Array) -> tuple[Toy, dict[str, jax.Array]]:
    w = 0.9 * s.w + rain
    return Toy(w=w, n=s.n + 1), {"w_end": w, "w_morning": s.w}


DATES = np.array([2015001, 2015002, 2015003, 2015004, 2015005])
RAIN = jnp.array([1.0, 0.0, 2.0, 0.0, 0.5])


def test_resync_overwrites_selected_leaf_each_morning() -> None:
    s0 = Toy(w=jnp.array(10.0), n=jnp.array(0))
    ref_morning = np.array([10.0, 5.0, 4.0, 7.0, 3.0])
    ref = resync.DailyReference(DATES, {"W": ref_morning})
    vals, mask = resync.morning_values(ref, DATES, ["W"])
    final, out = jax.jit(
        lambda s: resync.resync_scan(step, s, RAIN, where=lambda t: t.w, values=vals, mask=mask)
    )(s0)
    np.testing.assert_allclose(out["w_morning"], ref_morning)
    np.testing.assert_allclose(out["w_end"], 0.9 * ref_morning + np.asarray(RAIN))
    assert int(final.n) == 5  # untouched leaves evolve freely


def test_mask_false_keeps_model_state_and_equals_free_run() -> None:
    s0 = Toy(w=jnp.array(10.0), n=jnp.array(0))
    _, free = jax.lax.scan(step, s0, RAIN)
    vals = (jnp.full((5,), 123.0),)
    _, out = resync.resync_scan(step, s0, RAIN, where=lambda t: t.w, values=vals, mask=jnp.zeros(5, bool))
    np.testing.assert_array_equal(out["w_end"], free["w_end"])
    # a gap in the reference (day 3 only): the model continues from its own state that day
    mask = jnp.array([False, False, True, False, False])
    vals = (jnp.array([0.0, 0.0, 1.0, 0.0, 0.0]),)
    _, out = resync.resync_scan(step, s0, RAIN, where=lambda t: t.w, values=vals, mask=mask)
    np.testing.assert_allclose(out["w_morning"][:2], free["w_morning"][:2])
    assert float(out["w_morning"][2]) == 1.0
    np.testing.assert_allclose(out["w_morning"][3], 0.9 * 1.0 + 2.0)


def test_end_of_day_table_uses_lag_one() -> None:
    ref = resync.DailyReference(DATES, {"W": np.array([1.0, 2.0, 3.0, 4.0, 5.0])})
    (v,), m = resync.morning_values(ref, DATES, ["W"], lag=1)
    np.testing.assert_array_equal(np.asarray(m), [False, True, True, True, True])
    np.testing.assert_array_equal(np.asarray(v), [0.0, 1.0, 2.0, 3.0, 4.0])
    # dates the reference lacks are masked out
    (v,), m = resync.morning_values(ref, [2015002, 2015009], ["W"])
    np.testing.assert_array_equal(np.asarray(m), [True, False])


def test_gradient_is_finite_and_cut_by_the_overwrite() -> None:
    vals, mask = (jnp.array([10.0, 5.0, 4.0, 7.0, 3.0]),), jnp.array([True, True, True, True, True])

    def loss(w0: jax.Array, k: jax.Array) -> jax.Array:
        s0 = Toy(w=w0, n=jnp.array(0))

        def st(s: Toy, r: jax.Array) -> tuple[Toy, jax.Array]:
            w = k * s.w + r
            return Toy(w=w, n=s.n + 1), w

        _, w = resync.resync_scan(st, s0, RAIN, where=lambda t: t.w, values=vals, mask=mask)
        return jnp.sum(w)

    g_w0, g_k = jax.grad(loss, argnums=(0, 1))(jnp.array(10.0), jnp.array(0.9))
    assert float(g_w0) == 0.0  # the first morning is overwritten: no dependence on w0
    assert np.isfinite(float(g_k)) and float(g_k) == pytest.approx(float(np.sum(vals[0])))


def test_several_leaves_and_shape_errors() -> None:
    s0 = Toy(w=jnp.zeros(3), n=jnp.array(0))

    def st(s: Toy, r: jax.Array) -> tuple[Toy, jax.Array]:
        return Toy(w=s.w + r, n=s.n + 1), s.n

    vals = (jnp.ones((5, 3)), jnp.arange(5) * 10)
    _, n_morning = resync.resync_scan(
        st, s0, RAIN, where=lambda t: (t.w, t.n), values=vals, mask=jnp.ones(5, bool)
    )
    np.testing.assert_array_equal(n_morning, [0, 10, 20, 30, 40])
    with pytest.raises(ValueError, match="selects 1 leaves"):
        resync.resync_scan(st, s0, RAIN, where=lambda t: t.w, values=vals, mask=jnp.ones(5, bool))
    with pytest.raises(ValueError, match="shape"):
        resync.resync_scan(
            st, s0, RAIN, where=lambda t: t.w, values=(jnp.ones((5, 4)),), mask=jnp.ones(5, bool)
        )
    with pytest.raises(ValueError, match="mask"):
        resync.resync_scan(
            st, s0, RAIN, where=lambda t: t.w, values=(jnp.ones((5, 3)),), mask=jnp.ones(4, bool)
        )


def test_reference_from_dump_table(tmp_path: Path) -> None:
    buf = io.BytesIO()
    for k, d in enumerate(DATES):
        dumps.write_record(buf, "PHYSCL", 0, k + 1, int(d), 1, {"THETA": np.full(2, 0.1 * k)}, seq=2 * k + 1)
        dumps.write_record(
            buf, "PHYSCL", 1, k + 1, int(d), 1, {"THETA": np.full(2, 0.1 * k + 0.05)}, seq=2 * k + 2
        )
    f = tmp_path / "ajdump_PHYSCL.bin"
    f.write_bytes(buf.getvalue())
    p = dumps.save_table(
        dumps.daily_table(f, ["THETA"], phase="entry"), tmp_path / "physcl_entry.npz", meta={"x": 1}
    )
    ref = resync.DailyReference.from_npz(p, {"theta": "THETA"})
    assert ref.meta["routine"] == "PHYSCL" and ref.meta["x"] == 1
    np.testing.assert_allclose(ref.values["theta"][:, 0], 0.1 * np.arange(5))
    got = ref.at([2015002, 2015010], "theta")
    np.testing.assert_allclose(got[0], [0.1, 0.1])
    assert np.isnan(got[1]).all()
    with pytest.raises(ValueError, match="increasing"):
        resync.DailyReference(np.array([2, 1]), {})
    with pytest.raises(ValueError, match="leading axis"):
        resync.DailyReference(DATES, {"a": np.zeros(4)})


def test_compare_daily_and_tolerance_file(tmp_path: Path) -> None:
    ref = np.array([[1.0, 2.0], [1.0, 2.0], [np.nan, np.nan], [0.0, 4.0]])
    sim = np.array([[1.0, 2.0], [1.001, 2.0], [5.0, 5.0], [0.0, 4.5]])
    d = np.array([1, 2, 3, 4])
    c = resync.compare_daily("theta", sim, ref, d, resync.Tolerance(rel=1e-2, abs=0.0))
    assert c.fail_dates == [4] and c.n_days == 3 and c.worst_date == 4 and not c.passed
    assert c.max_abs == pytest.approx(0.5) and c.max_rel == pytest.approx(0.125)
    assert "FAIL on 1 days" in str(c)
    assert resync.compare_daily("theta", sim, ref, d, resync.Tolerance(rel=0.2)).passed
    c = resync.compare_daily("x", np.array([np.nan]), np.array([1.0]), [7], resync.Tolerance(abs=1.0))
    assert c.fail_dates == [7]
    c = resync.compare_daily("m", sim, ref, d, resync.Tolerance(rel=1e-2), mask=np.array([1, 1, 1, 0], bool))
    assert c.passed
    with pytest.raises(ValueError, match="shapes"):
        resync.compare_daily("bad", sim[:2], ref, d, resync.Tolerance())
    with pytest.raises(ValueError, match="non-negative"):
        resync.Tolerance(rel=-1.0)
    tols = {"theta": resync.Tolerance(1e-6, 1e-9, "print precision"), "lai": resync.Tolerance(abs=1e-5)}
    p = resync.save_tolerances(tmp_path / "golden" / "tolerances.json", tols, meta={"case": "catpa2015"})
    assert resync.load_tolerances(p) == tols


def test_provenance_hashes_binary_and_case(tmp_path: Path) -> None:
    b = tmp_path / "main_ryzen5_avx512"
    b.write_bytes(b"binary")
    c1, c2 = tmp_path / "rzwqm.dat", tmp_path / "CA-TPA.BRK"
    c1.write_text("a")
    c2.write_text("b")
    p1 = resync.provenance("RZWQM2", "4.6", b, flags={"ISTRESS": 0}, case_files=[c1, c2])
    p2 = resync.provenance("RZWQM2", "4.6", b, flags={"ISTRESS": 0}, case_files=[c2, c1])
    assert p1 == p2 and p1["case_files"] == ["CA-TPA.BRK", "rzwqm.dat"]
    assert p1["binary_sha256"] == resync.file_sha256(b) and len(p1["case_hash"]) == 64
    c1.write_text("changed")
    assert resync.provenance("RZWQM2", "4.6", b, case_files=[c1, c2])["case_hash"] != p1["case_hash"]
    assert resync.provenance("DSSAT-CSM", "4.8.6.0", b)["case_hash"] is None
