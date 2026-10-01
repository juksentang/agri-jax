"""Unit tier (no data) for the RZWQM2-day io: ``.BRK`` write-back, per-day storm arrays, meteorology
modifiers, irrigation gate and the season table (:mod:`agrijax.io.rzwqm.storms`,
:mod:`agrijax.sites.catpa_m3`). The reference checks on CA-TPA are in
``tests/integration/test_io_m3_catpa.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agrijax.io.rzwqm.dat import Planting, RzwqmDat
from agrijax.io.rzwqm.met import format_brk, read_brk, write_brk
from agrijax.io.rzwqm.storms import (
    CM_PER_INCH,
    irrigation_cm,
    irrigation_operations,
    read_met_modifiers,
    storm_arrays,
)
from agrijax.sites.catpa_m3 import season_table, storm_forcing

_HEADER = [
    "===============================================================================",
    "=  synthetic breakpoint file",
    "===============================================================================",
]
# event records I9 I9 I9 I9 F10.3; pairs F9.3 I9 then F10.3 I10, five per record
_EVENTS = [
    "     2015        3        2        0     0.125",
    "    0.000        0     0.125       120",
    "     2015        4        2        0     0.005",  # below STMINP's 0.01 in: skipped
    "    0.000        0     0.005        60",
    "     2015        5        7        1     1.000",  # 7 breakpoints, crosses midnight
    "    0.000     1320     0.100      1350     0.200      1380     0.400      1410     0.600      1440",
    "    0.800     1500     1.000      1560",
]


def _brk_text(newline: str = "\r\n") -> str:
    return newline.join([*_HEADER, "1", *_EVENTS]) + newline


def test_brk_write_back_is_byte_identical(tmp_path: Path) -> None:
    for nl in ("\r\n", "\n"):
        src = tmp_path / "in.BRK"
        src.write_bytes(_brk_text(nl).encode("latin-1"))
        brk = read_brk(src)
        assert brk.header == tuple(_HEADER) and brk.newline == nl
        out = write_brk(brk, tmp_path / "out.BRK")
        assert out.read_bytes() == src.read_bytes()
    assert list(brk.events["n_breakpoints"]) == [2, 2, 7]
    assert format_brk(brk) == _brk_text("\n")


def test_brk_writer_rejects_fractional_minutes(tmp_path: Path) -> None:
    src = tmp_path / "in.BRK"
    src.write_text(
        "1\n     2015        3        2        0     0.125\n    0.000        0     0.125     120.5\n"
    )
    with pytest.raises(ValueError, match="not an integer"):
        format_brk(read_brk(src))


def test_storm_arrays_breakpoints_skip_and_midnight(tmp_path: Path) -> None:
    src = tmp_path / "in.BRK"
    src.write_bytes(_brk_text().encode("latin-1"))
    brk = read_brk(src)
    days = np.arange(np.datetime64("2015-01-01"), np.datetime64("2015-01-08"))
    s = storm_arrays(days, brk)
    # day 3: one interval of 2 h, 0.125 in
    assert s.ts0[2] == 0.0 and s.event[2] == 0
    np.testing.assert_allclose(s.duration[2, :1], [2.0], rtol=0, atol=0)
    np.testing.assert_allclose(s.depth[2, :1], [0.125 * CM_PER_INCH], rtol=0, atol=1e-15)
    # day 4: below the STMINP minimum
    assert s.event[3] == -1 and s.ts0[3] == 24.0 and s.depth[3].sum() == 0.0
    # day 5 from 22:00: four 30-min intervals before midnight, the rest on day 6 from 0:00
    assert s.ts0[4] == 22.0 and s.event[4] == 2 and s.event[5] == 2 and s.ts0[5] == 0.0
    np.testing.assert_allclose(s.duration[4, :4], [0.5] * 4, rtol=0, atol=1e-15)
    np.testing.assert_allclose(s.duration[5, :2], [1.0, 1.0], rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        s.depth[4, :4], np.array([0.1, 0.1, 0.2, 0.2]) * CM_PER_INCH, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(s.depth[5, :2], np.array([0.2, 0.2]) * CM_PER_INCH, rtol=0, atol=1e-14)
    np.testing.assert_allclose(s.total_cm.sum(), (0.125 + 1.0) * CM_PER_INCH, rtol=1e-15)
    assert s.n_bp == 4
    f = storm_forcing(s)
    assert f.depth.shape == (7, 4) and f.ts0.shape == (7,)


def test_storm_arrays_rain_modifier_and_one_storm_per_day(tmp_path: Path) -> None:
    src = tmp_path / "in.BRK"
    src.write_bytes(_brk_text().encode("latin-1"))
    brk = read_brk(src)
    days = np.arange(np.datetime64("2015-01-01"), np.datetime64("2015-01-08"))
    mod = np.full((8, 12), 100.0)
    mod[:2] = 0.0
    mod[6] = 50.0
    half = storm_arrays(days, brk, met_modifiers=mod)
    np.testing.assert_allclose(half.depth, 0.5 * storm_arrays(days, brk).depth, rtol=1e-15, atol=0)
    mod[6, 5] = 80.0
    with pytest.raises(NotImplementedError, match="month"):
        storm_arrays(days, brk, met_modifiers=mod)
    two = tmp_path / "two.BRK"
    two.write_text(
        "1\n     2015        3        2        0     0.125\n    0.000        0     0.125       120\n"
        "     2015        3        2        0     0.200\n    0.000      600     0.200       660\n"
    )
    with pytest.raises(ValueError, match="two storms"):
        storm_arrays(days, read_brk(two))


def test_read_met_modifiers(tmp_path: Path) -> None:
    rows = ["0  " * 12, "0  " * 12, *(["100  " * 12] * 5), "90  " * 12]
    p = tmp_path / "IPNAMES.DAT"
    p.write_text("\n".join([*(["C:\\x"] * 8), "1  1  2015  31  12  2023", *rows, "= comment"]) + "\n")
    m = read_met_modifiers(p)
    assert m.shape == (8, 12) and np.all(m[:2] == 0) and np.all(m[7] == 90.0) and np.all(m[6] == 100.0)


def _dat_with_irrigation(n_ops: int) -> RzwqmDat:
    lines = [
        "========================================================================",
        "==            I R R I G A T I O N   M A N A G E M E N T               ==",
        "========================================================================",
        f"{n_ops}  0  0  0.0  100.0  15.0  0.0  0  0  ",
        "========================================================================",
    ]
    return RzwqmDat([ln + "\n" for ln in lines])


def test_irrigation_gate() -> None:
    days = np.arange(np.datetime64("2015-01-01"), np.datetime64("2015-01-11"))
    dat = _dat_with_irrigation(0)
    assert irrigation_operations(dat) == 0
    np.testing.assert_array_equal(irrigation_cm(dat, days), np.zeros(10))
    with pytest.raises(NotImplementedError, match="irrigation"):
        irrigation_cm(_dat_with_irrigation(2), days)


def _planting(sow: str, harvest: str | None, option: int = 3) -> Planting:
    return Planting(
        plant_ref=1,
        planting_date=np.datetime64(sow),
        row_spacing_cm=76.0,
        planting_depth_layer=3,
        density_seeds_ha=80000.0,
        planting_method=0,
        harvest_option=option,
        harvest_growth_stage=0.0,
        harvest_growth_class=0,
        harvest_threshold=0.0,
        harvest_date=None if harvest is None else np.datetime64(harvest),
        stubble_height_cm=0.0,
        harvest_efficiency=0.0,
        harvest_type=0,
        soil_water_at_planting_cm=0.0,
        planting_window_days=0,
        line_no=1,
        density_address=(1, 6),
    )


def test_season_table_window_index_and_crop_days() -> None:
    pl = [
        _planting("2014-04-28", "2014-09-20"),
        _planting("2016-04-28", "2016-09-20"),
        _planting("2015-04-28", "2015-09-25"),
    ]
    s = season_table(pl, "2015-01-01", "2016-12-31")
    assert s.n_season == 2
    np.testing.assert_array_equal(s.yrplt, [2015118, 2016119])
    np.testing.assert_array_equal(s.harvest_yrdoy, [2015268, 2016264])
    np.testing.assert_array_equal(s.pltpop, [8.0, 8.0])
    np.testing.assert_array_equal(s.sdepth, [3.0, 3.0])
    np.testing.assert_array_equal(s.rowspc, [76.0, 76.0])
    days = np.array(
        ["2015-04-27", "2015-04-28", "2015-09-25", "2015-09-26", "2016-05-01"], dtype="datetime64[D]"
    )
    np.testing.assert_array_equal(s.season_index(days), [-1, 0, 0, 0, 1])
    np.testing.assert_array_equal(s.in_crop(days), [False, True, True, False, True])
    with pytest.raises(NotImplementedError, match="option"):
        season_table([_planting("2015-04-28", None, option=1)], "2015-01-01", "2015-12-31")
