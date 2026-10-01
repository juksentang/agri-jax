"""The native DSSAT inputs of the free-run day (``free_run_inputs(..., source="native")``) against the
instrumented-engine tables (``source="tables"``) and against ``dscsm048``.

* **Scope.** Of the 65 runs of the free-run acceptance (the 58 maize treatments of M2 and the 7
  nitrogen-off CA-TPA seasons), 50 run natively; the other 15 are refused with the unported DSSAT
  process named: GAGR0201 t1-t6 (automatic irrigation, ``IRRIG = A``, and the growth-chamber
  environment modifications, ``WTHMOD``) and GHWA0401 t1-t9 (CENTURY organic matter with an
  initial surface residue: the mulch decay and the ``SOILDYN`` soil changes).
* **Equality with the tables, field by field.** Static ``SOILPROP`` (``DLAYR``, ``DS``, ``LL``,
  ``DUL``, ``SAT``, ``SWCN``, ``CN``, ``SWCON``), ``SALB``, ``U``, the initial ``SW``, snow and mulch
  water, ``KSEVAP = KTRANS``, the daily ``RAIN``, ``TMAX``, ``IRRAMT``, the residue record, the
  SPAM weather record (``SRAD``, ``TMAX``, ``TMIN``, ``WINDSP``, ``CO2``, ``MULCH_AM``,
  ``MUL_EXTFAC``): **exact** (the same float64 numbers). The hourly mean ``TAVG``: within
  :data:`TAVG_ULPS` REAL*4 units in the last place, off on at most :data:`TAVG_ULP_DAYS_MAX` of
  the days (measured on the 50 runs: 70 of 7 301 days, 53 by one unit, 16 by two, one by three).
  The transcendentals are evaluated correctly rounded; the reference's glibc ``sinf`` / ``expf``
  are off by one unit on a few hours, and the 24-hour REAL*4 sum spreads that over up to three
  units of the mean. The static parts that need no unported process
  (the soil, the initial water, ``SALB``, ``U``, the residue parameters, ``KEP``) are also checked on
  the 15 refused treatments (default tier, no reference run needed).
* **The free run.** The day run from the native inputs reproduces the acceptance of the table path:
  the harvest-day grain weight within 2 % of ``HWAM`` on all 50, the emergence, silking and
  maturity days of the table path and of ``dscsm048``, and outputs within :data:`NATIVE_VS_TABLES_RTOL` of the table
  path's (the TAVG units only reach ``TRATIO``).

The default tier checks the static inputs on all 65 and the full comparison on a handful of runs;
the slow tier (``-m slow``) all 65.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import day_dssat486_free_harness as h
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.io.dssat import read_cul, read_eco, read_filex
from agrijax.io.dssat._f77 import decimal_of
from agrijax.io.dssat.native_management import (
    extinction_kep,
    irrigation_amounts,
    residue_parameters,
    switches,
    treatment_levels,
    unsupported_features,
)
from agrijax.io.dssat.native_soil import find_soil_profile, initial_soil_water, native_soil
from agrijax.io.dssat.native_weather import native_weather
from agrijax.port import dumps
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
from agrijax.sites.dssat_free_run import FreeRunInputs, NativeInputError, free_run_inputs

pytestmark = [
    pytest.mark.allow_skip(
        reason="needs the instrumented-engine tables (WATBAL, SPAM), the DSSAT example data and, for the "
        "free run, dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT) and the CA-TPA DSSAT case"
    ),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

MAIZE = DSSAT_ENGINE / "example_data" / "Maize"
WEATHER = DSSAT_ENGINE / "example_data" / "Weather"
SOIL = DSSAT_ENGINE / "example_data" / "Soil"
STD = dscsm_paths(DSSAT_ENGINE)[1]
#: the runs refused natively and the features named (a substring of each message)
REFUSED: dict[str, tuple[str, ...]] = {
    **{f"GAGR0201_t{k:02d}": ("IRRIG = A", "WTHMOD") for k in range(1, 7)},
    **{f"GHWA0401_t{k:02d}": ("MESOM = P", "initial surface residue") for k in range(1, 10)},
}
N_NATIVE = 50
#: TAVG: native within this many REAL*4 units of the table value (measured at most 3)
TAVG_ULPS = 4
#: at most this fraction of the days has TAVG off at all (measured 70 of the 7 301 table days of the
#: 50 runs, 0.96 %)
TAVG_ULP_DAYS_MAX = 0.02
#: native free run against the table path, relative to each output's season maximum
NATIVE_VS_TABLES_RTOL = 1e-6
SMALL = [
    ("UFGA8201", 3),
    ("UFGA8201", 4),
    ("IUAF9901", 1),
    ("SIAZ9501", 1),
    ("IBWA8301", 1),
    ("CTPA1501", 1),
    ("GAGR0201", 1),
    ("GHWA0401", 1),
]
#: series compared exactly, and the one with the TAVG tolerance
EXACT_SERIES = (
    "rain",
    "tmax",
    "irrigation",
    "m_mass",
    "m_cover",
    "m_new",
    "m_watfac",
    "mulch_am",
    "mulch_extfac",
    "wind_run",
    "co2",
    "srad",
    "tmax_w",
    "tmin_w",
    "dlayr_end",
)
SITE_FIELDS = (
    "dlayr",
    "ds",
    "ll",
    "dul",
    "sat",
    "swcn",
    "cn",
    "swcon",
    "salb",
    "u",
    "mesev",
    "meinf",
    "actwtd",
    "pmfraction",
)


def _paths(exp: str, data_dir: Path) -> dict[str, Any]:
    if exp == "CTPA1501":
        case = data_dir / h.CATPA_CASE
        return {
            "filex": case / h.CATPA_FILEX,
            "weather_dir": case,
            "soil_dirs": [case],
            "genotype_dir": case,
        }
    return {
        "filex": MAIZE / f"{exp}.MZX",
        "weather_dir": WEATHER,
        "soil_dirs": [MAIZE, SOIL],
        "genotype_dir": None,
    }


def _table(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    t, _ = dumps.load_table(path)
    return np.asarray(t.date, dtype=np.int64), {k: np.asarray(v) for k, v in t.values.items()}


def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in REAL*4 units of the last place of b."""
    b32 = np.asarray(b, dtype=np.float32)
    return np.abs(np.asarray(a, np.float64) - b32.astype(np.float64)) / np.spacing(np.abs(b32)).astype(
        np.float64
    )


# ------------------------------------------------------------------------ static inputs, all 65
def _keys(data_dir: Path) -> list[tuple[str, int]]:
    return h.a12_keys(data_dir) + h.catpa_keys()


def test_native_static_inputs_equal_the_tables_on_all_65(data_dir: Path) -> None:
    """What needs no reference run and no unported process, on every run (also the refused ones)."""
    for sub in (h.DSW, h.DET, h.A12, h.CATPA_CASE):
        if not (data_dir / sub).is_dir():
            pytest.skip(f"{data_dir / sub} not found")
    if not MAIZE.is_dir():
        pytest.skip(f"DSSAT example data not found under {DSSAT_ENGINE}")
    keys = _keys(data_dir)
    assert len(keys) == 65
    refused: dict[str, list[str]] = {}
    tavg_days = tavg_off = 0
    for exp, t in keys:
        key = h.key_of(exp, t)
        pth = _paths(exp, data_dir)
        x = read_filex(pth["filex"])
        tr = treatment_levels(x, t)
        sw = switches(x, t)
        bad = unsupported_features(x, t)
        if bad:
            refused[key] = bad
        d = data_dir / h.DSW / key
        rd, ri = _table(d / "wb_rate_in.npz")
        _, s0 = _table(d / "wb_seasinit_in.npz")
        sd, si = _table(data_dir / h.DET / f"{key}_spam_in.npz")
        nl = int(ri["SP_NLAYR"][0])
        src = s0 if "SP_DLAYR" in s0 else ri
        prof = find_soil_profile(str(x["FIELDS"][int(tr["FL"])]["ID_SOIL"]), pth["soil_dirs"])
        soil = native_soil(prof, sw.mesol)
        assert soil.nl == nl, key
        for k in ("dlayr", "ds", "ll", "dul", "sat", "swcn"):
            np.testing.assert_array_equal(
                getattr(soil, k), decimal_of(src[f"SP_{k.upper()}"][0, :nl]), err_msg=f"{key} {k}"
            )
        assert soil.cn == float(decimal_of(src["SP_CN"][0])), key
        assert soil.swcon == float(decimal_of(src["SP_SWCON"][0])), key
        assert soil.salb == float(decimal_of(si["SALB_S"][0])), key
        assert soil.u == float(decimal_of(si["U_S"][0])), key
        ic = x["INITIAL CONDITIONS"].get(int(tr["IC"]), {}) if int(tr["IC"]) else {}
        sw0 = initial_soil_water(soil, ic.get("rows"))
        np.testing.assert_array_equal(
            np.asarray(sw0, np.float32), ri["SW"][0, :nl], err_msg=f"{key} initial SW"
        )
        res = residue_parameters(str(ic.get("PCR", "") or ""), STD / "StandardData" / "RESCH048.SDA")
        assert (res.am, res.extfac, res.watfac) == (
            float(si["MULCH_AM"][0]),
            float(si["MUL_EXTFAC"][0]),
            float(ri["M_WATFAC"][0]),
        ), key
        geno = pth["genotype_dir"] or STD / "Genotype"
        eco = read_cul(Path(geno) / "MZCER048.CUL").loc[str(x["CULTIVARS"][int(tr["CU"])]["INGENO"]).strip()][
            "ECO#"
        ]
        kep = extinction_kep(float(read_eco(Path(geno) / "MZCER048.ECO").loc[str(eco).strip()]["KCAN"]))
        assert kep == float(si["KSEVAP"][0]) == float(si["KTRANS"][0]), key
        if bad:
            continue
        # the daily inputs of a supported run: the weather record and the irrigation
        fl = x["FIELDS"][int(tr["FL"])]
        wsta = str(fl["WSTA"]).strip()
        y = int(sd[0]) // 1000
        wname = f"{wsta[:4]}{y % 100:02d}01.WTH" if len(wsta) == 4 else f"{wsta}.WTH"
        w = native_weather(
            pth["weather_dir"] / wname,
            sd.tolist(),
            co2_option=sw.co2,
            co2_file=STD / "StandardData" / "CO2048.WDA",
        )
        for name, col in (
            ("srad", "SRAD_W"),
            ("tmax", "TMAX_W"),
            ("tmin", "TMIN_W"),
            ("windsp", "WINDSP_W"),
            ("co2", "CO2_W"),
        ):
            np.testing.assert_array_equal(getattr(w, name), si[col], err_msg=f"{key} {name}")
        u = _ulps(w.tavg, si["TAVG_W"])
        assert float(u.max()) <= TAVG_ULPS, (key, float(u.max()))
        tavg_days += int(u.size)
        tavg_off += int(np.sum(u > 0))
        at = np.searchsorted(sd, rd)
        np.testing.assert_array_equal(w.rain[at], ri["RAIN"], err_msg=f"{key} RAIN")
        np.testing.assert_array_equal(w.tmax[at], ri["TMAX"], err_msg=f"{key} TMAX")
        np.testing.assert_array_equal(
            irrigation_amounts(x, t, rd.tolist()), ri["IRRAMT"], err_msg=f"{key} IRRAMT"
        )
    assert {k: [f for f in REFUSED[k] if not any(f in b for b in refused[k])] for k in refused} == {
        k: [] for k in REFUSED
    }
    assert set(refused) == set(REFUSED), sorted(set(refused) ^ set(REFUSED))
    assert len(keys) - len(refused) == N_NATIVE
    print(f"TAVG off by 1-{TAVG_ULPS} REAL*4 units on {tavg_off} of {tavg_days} days")
    assert tavg_off <= TAVG_ULP_DAYS_MAX * tavg_days


# ------------------------------------------------------------------------ with the reference runs
def _leaves(tree: Any) -> dict[str, np.ndarray]:
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in flat}


def _compare_inputs(nat: FreeRunInputs, tab: FreeRunInputs) -> dict[str, float]:
    """Field-by-field equality (exact; TAVG within TAVG_ULPS); returns the TAVG unit count."""
    key = nat.key
    np.testing.assert_array_equal(nat.days, tab.days, err_msg=key)
    assert (nat.nl, nat.mesev, nat.soildyn) == (tab.nl, tab.mesev, tab.soildyn), key
    for f in SITE_FIELDS:
        a, b = getattr(nat.site, f), getattr(tab.site, f)
        if isinstance(a, str):
            assert a == b, (key, f)
        else:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=f"{key} site.{f}")
    np.testing.assert_array_equal(nat.sw0, tab.sw0, err_msg=f"{key} sw0")
    for f in ("snow0", "mulch0", "mulch_evap0", "ksevap"):
        assert getattr(nat, f) == getattr(tab, f), (key, f)
    for k in EXACT_SERIES:
        np.testing.assert_array_equal(nat.series[k], tab.series[k], err_msg=f"{key} series {k}")
    u = _ulps(nat.series["tavg"], tab.series["tavg"])
    assert float(u.max()) <= TAVG_ULPS, (key, "tavg", float(u.max()))
    # the trees the day runs on: params and state exact, forcing exact but the TAVG leaf
    for which in ("params", "state"):
        a = _leaves(nat.params() if which == "params" else nat.state())
        b = _leaves(tab.params() if which == "params" else tab.state())
        assert a.keys() == b.keys(), (key, which)
        bad = [k for k in a if a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k], equal_nan=True)]
        assert bad == [], (key, which, bad)
    n = nat.n_days + 30
    a, b = _leaves(nat.forcing(n)), _leaves(tab.forcing(n))
    assert a.keys() == b.keys(), key
    bad = [
        k
        for k in a
        if "tavg" not in k and (a[k].dtype != b[k].dtype or not np.array_equal(a[k], b[k], equal_nan=True))
    ]
    assert bad == [], (key, "forcing", bad)
    return {"tavg_days_off": float(np.sum(u > 0)), "days": float(u.size)}


def _run_free(runs: list[FreeRunInputs]) -> dict[str, dict[str, np.ndarray]]:
    """The free-run day on ``runs`` (batched per layer count and MESEV); outputs per run key."""
    from agrijax.core.runtime import run_sites
    from agrijax.models.day_dssat486 import (
        SLOT,
        SOIL_EVAPORATION_KEYS,
        day_dssat486,
        day_outputs,
        day_processes,
    )

    groups: dict[tuple[int, str], list[FreeRunInputs]] = {}
    for r in runs:
        groups.setdefault((r.nl, r.mesev), []).append(r)
    stack = lambda ts: jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(v) for v in xs]), *ts)  # noqa: E731
    res: dict[str, dict[str, np.ndarray]] = {}
    for (_, mesev), rs in groups.items():
        n = max(r.n_days for r in rs)
        ps = [r.params() for r in rs]
        model = day_dssat486(SLOT).compile(
            day_processes(SLOT, soil_evaporation=SOIL_EVAPORATION_KEYS[mesev]),
            outputs=day_outputs(SLOT),
            exact_lags=True,
        )
        out = run_sites(
            model,
            stack(ps),
            stack([r.forcing(n) for r in rs]),
            stack([r.state(p) for r, p in zip(rs, ps, strict=True)]),
        )
        for b, r in enumerate(rs):
            res[r.key] = {k: np.asarray(v)[b, : r.n_days] for k, v in out.items()}
    return res


def _first(istage: np.ndarray, code: int) -> int:
    hit = np.nonzero(istage == code)[0]
    return int(hit[0]) if hit.size else -1


@pytest.fixture(scope="module", params=["small", pytest.param("all", marks=pytest.mark.slow)])
def built(request: pytest.FixtureRequest, data_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    exe, _ = dscsm_paths(DSSAT_ENGINE)
    if not exe.is_file() or not MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    for sub in (h.DSW, h.DET, h.A12, h.CATPA_CASE):
        if not (data_dir / sub).is_dir():
            pytest.skip(f"{data_dir / sub} not found")
    keys = SMALL if request.param == "small" else _keys(data_dir)
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tmp_path_factory.mktemp("native"))
    outs = h.run_references(keys, root / f"ni_{request.param}", data_dir, jobs)
    pairs: dict[str, tuple[FreeRunInputs, FreeRunInputs]] = {}
    refused: dict[str, str] = {}
    for exp, t in keys:
        key = h.key_of(exp, t)
        pth = _paths(exp, data_dir)
        kw = {k: v for k, v in pth.items() if k != "filex"}
        tab = free_run_inputs(pth["filex"], t, outs[key], data_dir, source="tables", **kw)
        try:
            nat = free_run_inputs(pth["filex"], t, outs[key], source="native", **kw)
        except NativeInputError as e:
            refused[key] = str(e)
            continue
        pairs[key] = (nat, tab)
    return {"keys": keys, "pairs": pairs, "refused": refused}


def test_refused_treatments_name_the_missing_process(built) -> None:
    want = {k: v for k, v in REFUSED.items() if k in {h.key_of(e, t) for e, t in built["keys"]}}
    assert set(built["refused"]) == set(want), built["refused"]
    for k, feats in want.items():
        for f in feats:
            assert f in built["refused"][k], (k, f, built["refused"][k])
    if len(built["keys"]) == 65:
        assert len(built["pairs"]) == N_NATIVE


def test_native_inputs_equal_the_tables_field_by_field(built) -> None:
    off = days = 0.0
    for nat, tab in built["pairs"].values():
        c = _compare_inputs(nat, tab)
        off += c["tavg_days_off"]
        days += c["days"]
    print(f"TAVG off by 1-{TAVG_ULPS} REAL*4 units on {int(off)} of {int(days)} crop days")
    assert off <= TAVG_ULP_DAYS_MAX * days


def test_free_run_from_native_inputs_reproduces_the_acceptance(built) -> None:
    pairs = built["pairs"]
    nat = _run_free([p[0] for p in pairs.values()])
    tab = _run_free([p[1] for p in pairs.values()])
    worst = 0.0
    for key, (r, _) in pairs.items():
        o, q = nat[key], tab[key]
        hwam = r.summary["HWAM"]
        y = float(o["gwad"][-1, 0])
        assert abs(y - hwam) <= h.YIELD_REL * hwam, (key, y, hwam)
        # emergence, silking, maturity (the harness's ISTAGE codes): the table path's and dscsm048's days
        for code, col in ((1, "EDAT"), (4, "ADAT"), (10, "MDAT")):
            i = _first(o["istage"][:, 0], code)
            assert i == _first(q["istage"][:, 0], code), (key, code)
            ref = r.summary[col]
            assert (int(r.days[i]) if i >= 0 else -99) == (int(ref) if ref > 0 else -99), (key, col)
        for k in ("gwad", "cwad", "lai", "soil_sw", "es", "ep", "eo"):
            scale = max(float(np.max(np.abs(q[k]))), 1e-12)
            d = float(np.max(np.abs(o[k] - q[k]))) / scale
            worst = max(worst, d)
            assert d <= NATIVE_VS_TABLES_RTOL, (key, k, d)
    print(f"native vs tables: largest output difference {worst:.3g} of the season maximum")
