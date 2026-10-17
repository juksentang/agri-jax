"""Soil coordinates on an end of their domain left out on record (``calib/soil.py``
:func:`without_domain_bounds`), no data: on the DSSAT example profiles of UFGA8201 (IBMZ910014) and
IBWA8301 (IBMZ910013), ``SRGF`` 1.0 and 0.0 and ``SLPF`` 1.0 are taken out and recorded, a missing
value is kept (and refused by ``soil_space``), and the rest builds a space.
"""

from __future__ import annotations

import numpy as np
import pytest

from agrijax.calib.soil import SITE_BOUNDS, soil_space, without_domain_bounds

#: .SOL values of the DSSAT example profiles (DSSAT-CSM v4.8.6.0 example_data/Soil/SOIL.SOL)
UFGA = {
    "cn": 60.0, "swcon": 0.65, "u": 2.0, "slpf": 0.92,
    "ll": np.asarray([0.026, 0.025, 0.025, 0.025, 0.028, 0.028, 0.029, 0.070]),
    "dul": np.asarray([0.096, 0.086, 0.086, 0.086, 0.090, 0.090, 0.130, 0.258]),
    "sat": np.asarray([0.23] * 7 + [0.36]),
    "srgf": np.asarray([1.0, 1.0, 0.7, 0.3, 0.05, 0.03, 0.002, 0.0]),
    "swcn": np.full(8, np.nan),
}  # fmt: skip
IBWA = {
    "cn": 60.0, "swcon": 0.60, "u": 5.0, "slpf": 1.00,
    "ll": np.asarray([0.22, 0.23, 0.24, 0.25, 0.26, 0.25, 0.26]),
    "dul": np.asarray([0.35, 0.35, 0.35, 0.37, 0.38, 0.38, 0.40]),
    "sat": np.asarray([0.55, 0.55, 0.55, 0.48, 0.46, 0.46, 0.48]),
    "srgf": np.asarray([1.0, 1.0, 0.8, 0.4, 0.2, 0.05, 0.02]),
    "swcn": np.full(7, np.nan),
}  # fmt: skip


def test_srgf_and_slpf_on_a_domain_end_are_left_out_on_record() -> None:
    q, lay, ex = without_domain_bounds(UFGA, ["cn", "slpf", "ll", "srgf"], {"ll": [0, 1], "srgf": range(8)})
    assert q == ("cn", "slpf", "ll", "srgf")
    assert lay == {"ll": (0, 1), "srgf": (2, 3, 4, 5, 6)}
    assert [(e.name, e.value, e.bound) for e in ex] == [
        ("SRGF_H1", 1.0, 1.0),
        ("SRGF_H2", 1.0, 1.0),
        ("SRGF_H8", 0.0, 0.0),
    ]
    assert all("SOIL.CDE" in e.reason and "[0, 1]" in e.reason for e in ex)
    # the requested space without them builds (it is refused with them)
    sp = soil_space(UFGA, q, lay, SITE_BOUNDS)
    assert sp.names == ("CN", "SLPF", "LL_H1", "LL_H2", "SRGF_H3", "SRGF_H4", "SRGF_H5", "SRGF_H6", "SRGF_H7")
    with pytest.raises(ValueError):
        soil_space(UFGA, ["srgf"], {"srgf": [0]}, SITE_BOUNDS)
    # SLPF = 1.0 of IBMZ910013 (domain (0, 1]); a layered quantity left with no layer is dropped
    q2, lay2, ex2 = without_domain_bounds(IBWA, ["slpf", "u", "srgf"], {"srgf": [0, 1]})
    assert q2 == ("u",) and lay2 == {} and [e.name for e in ex2] == ["SLPF", "SRGF_H1", "SRGF_H2"]
    assert ex2[0].layer is None and "SLPF < 1e-4 as 1.0" in ex2[0].reason
    # a missing value (SSKS -99) is not a domain end: it stays, soil_space refuses it
    q3, lay3, ex3 = without_domain_bounds(UFGA, ["swcn"], [0])
    assert q3 == ("swcn",) and lay3 == {"swcn": (0,)} and ex3 == ()
    with pytest.raises(ValueError, match="unknown quantities"):
        without_domain_bounds(UFGA, ["bd"])
