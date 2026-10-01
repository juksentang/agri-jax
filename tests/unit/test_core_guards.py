"""Named numerical guards and demo constants of core/grad.py, core/ledger.py and the demo models."""

import jax.numpy as jnp

from agrijax.core import grad as G
from agrijax.core import ledger as L
from agrijax.core.coefficients import GUARDS
from agrijax.models import catpa_pet_demo as C
from agrijax.models import tobacco_demo as T


def test_guards_registered_with_their_values() -> None:
    expected = {
        "grad.segment_width_min": 1e-6,
        "grad.rate_floor": 1e-6,
        "ledger.atol_f64": 1e-10,
        "ledger.rtol_f64": 1e-12,
        "ledger.atol_f32": 1e-5,
        "ledger.rtol_f32": 1e-6,
    }
    for name, value in expected.items():
        assert GUARDS[name][0] == value and GUARDS[name][1].strip()
    assert G._SEGMENT_WIDTH_MIN == 1e-6 and G._RATE_FLOOR == 1e-6
    assert (L._ATOL_F64, L._RTOL_F64, L._ATOL_F32, L._RTOL_F32) == (1e-10, 1e-12, 1e-5, 1e-6)


def test_nint_half_away_from_zero() -> None:
    assert [float(G.round_st(v)) for v in (0.5, -0.5, 1.5, 2.5, -2.5, 0.49)] == [
        1.0,
        -1.0,
        2.0,
        3.0,
        -3.0,
        0.0,
    ]


def test_demo_defaults_are_declared() -> None:
    p = T.default_params()
    assert set(T._DEFAULT_VALUES) == {
        "phyllochron",
        "tbase",
        "leaf_mass_max",
        "growth_rate",
        "age_half",
        "sla",
        "density",
    }
    for k, v in T._DEFAULT_VALUES.items():
        assert float(getattr(p, k)) == float(jnp.asarray(v))  # the default dtype (float32 or float64)
    assert C._RESIDUE_TYPE_BY_CRES == {2.0: "corn", 2.5: "soybean", 4.0: "wheat"}
    assert C._MSALB_PLACEHOLDER == 0.13
