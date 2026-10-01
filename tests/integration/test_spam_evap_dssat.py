"""SPAM partition (pet slot) and soil / mulch evaporation (soil_water slot) against the instrumented
DSSAT-CSM v4.8.6.0 dscsm048, day by day, on the 58 M2 maize treatments and the 7 CA-TPA seasons.

Data and tolerances: see :mod:`spam_evap_dssat` (the helper module). Checked:

1. the collection: 65 runs (58 M2 treatments, 7 CA-TPA nitrogen-off seasons), every ``*.OUT`` of
   the instrumented build identical to build486's;
2. the configuration chain these runs take: ``EVAPO = R`` (``PETPT``),
   ``INFIL = S`` (mulch evaporation on), ``MESEV`` R on 39 runs (32 M2 + 7 CA-TPA, ``SOILEV`` every
   day) and S on 26 (``ESR_SoilEvap`` every day), no flood, no ``ETPHOT``, ``XLAI = XHLAI``,
   ``KSEVAP = KTRANS = KEP`` of CERES-Maize from the first day (the ``KSEVAP > 0`` form of ``PSE``);
3. each routine on its own dumped entry against its exit, every day: ``PETPT`` (EO), ``PSE``
   (EOS), ``MULCH_EVAP`` and SPAM's mulch step (EM, EOS_SOIL), ``SOILEV`` (ES, SUMES1, SUMES2, T;
   SEASINIT), ``ESR_SoilEvap`` (ES, and per layer ES_LYR, SWDELTU, UPFLOW, the profile type),
   ``TRATIO`` / ``TRANS`` (EOP), within the REAL*4 limits of :data:`spam_evap_dssat.OPS`;
4. the registered processes chained as the DSSAT day will run them (PT kernel -> ``pet/spam_pse``
   -> ``soil_water/mulch_evap`` -> ``soil_water/{soilev, esr_soilevap}`` -> ``pet/spam_trans``),
   driven only by SPAM's entry, the SOILEV store replayed: EO, EOS, EM, EOS_SOIL, ES, EVAP, EOP
   of every day within the summed limits of the chain;
5. ``SOILEV`` free-running: the store carried by the process from the reference's SEASINIT over
   the season (inputs replayed), batched over the runs; ES within the accumulated per-day limit;
6. the dumped daily values are the ones DSSAT prints (``ET.OUT``: EOAA, EOSA, ESAA, EMAA, EOPA at
   3 decimals).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spam_evap_dssat as h

from agrijax.io.dssat import read_out

pytestmark = [
    pytest.mark.allow_skip(reason="the SPAM dump tables are kept outside the repository"),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the dump comparison runs in float64"),
]


@pytest.fixture(scope="module")
def tables(data_dir: Path) -> Path:
    t = h.tables_dir(data_dir)
    if not (t / "collect_report.json").is_file():
        pytest.skip(f"no SPAM dump tables at {t}")
    return t


@pytest.fixture(scope="module")
def runs(tables: Path) -> list[h.Run]:
    return h.load_runs(tables)


def _assert_within(res: dict[str, np.ndarray], names: tuple[str, ...]) -> None:
    for k in names:
        a = res[k]
        lim = h.OPS[k.replace("MULCH_", "").replace("INIT_", "")]
        assert a.size > 0, k
        assert float(a.max()) <= lim, (k, float(a.max()), lim, int((a > lim).sum()))


# ------------------------------------------------------------------------------------ 1, 2
def test_collection_is_complete_and_leaves_the_outputs_unchanged(tables: Path) -> None:
    rep = json.loads((tables / "collect_report.json").read_text())
    keys = sorted(rep["treatments"])
    assert rep["n_treatments"] == 65 and rep["n_bad"] == 0
    assert sum(k.startswith("CTPA1501_") for k in keys) == 7 and len(keys) == 65
    for k, v in rep["treatments"].items():
        assert v["out_diffs"] == [] and "error" not in v, k
    assert h.run_keys(tables) == keys


def test_configuration_chain(runs: list[h.Run]) -> None:
    mesev = {"R": 0, "S": 0}
    for r in runs:
        si, so = r.spam_in, r.spam_out
        assert h._chr(si["MEEVP"][0]) == "R" and h._chr(si["MEINF"][0]) == "S", r.key
        assert h._chr(si["MEPHO"][0]) != "L" and h._chr(si["CROPC"][0]) == "MZ", r.key
        assert np.all(si["FLOOD_F"] == 0.0) and np.all(so["EF_L"] == 0.0), r.key
        assert np.array_equal(si["XLAI"], si["XHLAI"]), r.key
        assert np.all(si["KSEVAP"] == si["KTRANS"]) and np.unique(si["KSEVAP"]).size == 1, r.key
        assert float(si["KSEVAP"][0]) > 0.0, r.key
        m = r.mesev
        mesev[m] += 1
        n = len(r.dates)
        if m == "R":
            assert len(r.tables["soilev_in"]) == n and "esr_in" not in r.tables, r.key
        else:
            assert len(r.tables["esr_in"]) == n and "soilev_in" not in r.tables, r.key
        assert len(r.tables["mulch_in"]) == n, r.key  # EOS_SOIL > 1e-6 on every day
        if r.key.startswith("CTPA"):
            assert m == "R", r.key
    assert mesev == {"R": 39, "S": 26}
    assert sum(int((r.spam_out["EM_L"] > 0).sum()) for r in runs) > 1000  # the mulch path is active


# ------------------------------------------------------------------------------------ 3
def test_priestley_taylor_eo(runs: list[h.Run]) -> None:
    _assert_within(h.check_eo(runs), ("EO",))


def test_pse_eos(runs: list[h.Run]) -> None:
    _assert_within(h.check_eos(runs), ("EOS",))


def test_mulch_evap_and_spam_mulch_step(runs: list[h.Run]) -> None:
    res = h.check_mulch(runs)
    _assert_within(res, ("MULCH_EM", "MULCH_EOS", "EM", "EOS_SOIL"))
    assert int(res["_mulch_days"][0]) > 1000


def test_soilev_rate_and_seasinit(runs: list[h.Run]) -> None:
    res = h.check_soilev(runs)
    _assert_within(res, ("ES", "SUMES1", "SUMES2", "T", "INIT_SUMES1", "INIT_SUMES2", "INIT_T"))
    assert float(res["SW_AVAIL"].max()) <= 4.0  # three REAL*4 additions and the rounding of the result
    assert float(res["INIT_SWEF"].max()) <= 6.0  # 0.9 - 0.00038 (DLAYR - 30)**2: five operations
    assert res["INIT_SUMES1"].size == 39


def test_esr_soil_evaporation_per_layer(runs: list[h.Run]) -> None:
    res = h.check_esr(runs)
    _assert_within(res, ("ESR_ES", "ES_LYR", "SWDELTU", "UPFLOW"))
    assert float(res["PROFILE"].sum()) == 0.0  # the profile type of every day is the reference's


def test_tratio_trans_eop(runs: list[h.Run]) -> None:
    res = h.check_trans(runs)
    _assert_within(res, ("EOP",))
    assert int(res["_trat_ne_1_days"][0]) > 1000  # CO2 != 330 ppm: TRATIO is exercised


# ------------------------------------------------------------------------------------ 4
def _chain(group: list[h.Run], nl: int) -> dict[str, np.ndarray]:
    """The registered processes chained on SPAM's entry of every day of ``group`` (one batch)."""
    from agrijax.iface.crop import CanopyRecord
    from agrijax.iface.surface import PETFluxes
    from agrijax.processes.pet import priestley_taylor
    from agrijax.processes.pet.spam_dssat import (
        SpamParams,
        SpamPSEState,
        SpamTransState,
        SpamWeather,
        spam_potential_soil_evaporation,
        spam_potential_transpiration,
    )
    from agrijax.processes.soil_water.bucket_evap import (
        SoilEvapParams,
        SoilEvapState,
        soil_evaporation_esr,
        soil_evaporation_mulch,
        soil_evaporation_soilev,
    )

    def si(n: str) -> jnp.ndarray:
        return jnp.asarray(h.f64(h.cat(group, lambda r: r.spam_in[n])))

    def lay(n: str) -> jnp.ndarray:
        return jnp.asarray(h.f64(h.cat(group, lambda r: r.spam_in[n][:, :nl])))

    mesev = group[0].mesev
    xhlai = si("XHLAI")
    z = jnp.zeros_like(xhlai)
    eo = priestley_taylor(
        si("SRAD_W"),
        si("TMAX_W"),
        si("TMIN_W"),
        xhlai,
        jnp.asarray(h.f64(h.cat(group, lambda r: r.spam_out["ET_ALB_L"]))),
    )
    canopy = CanopyRecord(lai=xhlai[:, None], tlai=xhlai[:, None], height=z[:, None])
    pet = PETFluxes(z, z, z, z, z, eo)
    sp = SpamParams(ksevap=si("KSEVAP"), ktrans=si("KTRANS"), c4=True)
    pet = spam_potential_soil_evaporation(SpamPSEState(canopy=canopy, pet=pet), sp, None).pet
    if mesev == "R":
        store = {
            n: jnp.asarray(h.f64(h.cat(group, lambda r, n=n: r.tables["soilev_in"].values[n])))
            for n in ("SUMES1", "SUMES2", "T", "SWEF")
        }
    else:
        store = {n: z for n in ("SUMES1", "SUMES2", "T", "SWEF")}
    zl = jnp.zeros_like(lay("SW"))
    st = SoilEvapState(
        sw=lay("SW"),
        swdelts=lay("SWDELTS"),
        swdeltu=lay("SWDELTU"),
        winf=si("WINF"),
        dlayr=lay("DLAYR_S"),
        ds=lay("DS_S"),
        dul=lay("DUL_S"),
        ll=lay("LL_S"),
        mulch_mass=si("MULCHMASS"),
        mulch_cover=si("MULCHCOVER"),
        mulch_water=si("MULCHWAT"),
        mulch_am=si("MULCH_AM"),
        mulch_extfac=si("MUL_EXTFAC"),
        sumes1=store["SUMES1"],
        sumes2=store["SUMES2"],
        t=store["T"],
        swef=store["SWEF"],
        eos_soil=z,
        em=z,
        es=z,
        es_lyr=zl,
        upflow=zl,
        evap=z,
        pet=pet,
    )
    u = jnp.asarray(h.f64(h.cat(group, lambda r: r.spam_in["U_S"])))
    p = SoilEvapParams(u=u, pmfraction=z, mulch_active=True)
    st = soil_evaporation_mulch(st, p, None)
    st = (soil_evaporation_soilev if mesev == "R" else soil_evaporation_esr)(st, p, None)
    wx = SpamWeather(tavg=si("TAVG_W"), wind_run=si("WINDSP_W"), co2=si("CO2_W"))
    tr = spam_potential_transpiration(SpamTransState(canopy=canopy, pet=pet, evaporation=st.evap), sp, wx)
    return {
        "EO": np.asarray(eo),
        "EOS": np.asarray(pet.soil_evaporation) * 10.0,
        "EM": np.asarray(st.em),
        "EOS_SOIL": np.asarray(st.eos_soil),
        "ES": np.asarray(st.es),
        "EVAP": np.asarray(st.evap),
        "EOP": np.asarray(tr.pet.transpiration) * 10.0,
    }


def test_registered_processes_chained_on_spam_entry(runs: list[h.Run]) -> None:
    #: limits of the chained quantities: the OPS of every step on the path, in ulp of EO
    chain = {
        "EO": h.OPS["EO"],
        "EOS": h.OPS["EO"] + h.OPS["EOS"],
        "EM": h.OPS["EO"] + h.OPS["EOS"] + h.OPS["EM"],
        "EOS_SOIL": h.OPS["EO"] + h.OPS["EOS"] + h.OPS["EOS_SOIL"],
        "ES": h.OPS["EO"] + h.OPS["EOS"] + h.OPS["EOS_SOIL"] + max(h.OPS["ES"], h.OPS["ESR_ES"]),
        "EVAP": h.OPS["EO"] + h.OPS["EOS"] + h.OPS["EOS_SOIL"] + max(h.OPS["ES"], h.OPS["ESR_ES"]) + 1,
        "EOP": 2 * h.OPS["EO"]
        + h.OPS["EOS"]
        + h.OPS["EOS_SOIL"]
        + max(h.OPS["ES"], h.OPS["ESR_ES"])
        + h.OPS["EOP"],
    }
    ref_name = {
        "EO": "EO",
        "EOS": "EOS",
        "EM": "EM_L",
        "EOS_SOIL": "EOS_SOIL_L",
        "ES": "ES",
        "EVAP": "EVAP_L",
        "EOP": "EOP",
    }
    worst: dict[str, float] = {}
    n_days = 0
    groups: dict[tuple[str, int], list[h.Run]] = {}
    for r in runs:
        groups.setdefault((r.mesev, r.nlayr if r.mesev == "S" else 20), []).append(r)
    for (_, nl), g in sorted(groups.items()):
        out = _chain(g, nl)
        eo_ref = h.cat(g, lambda r: r.spam_out["EO"])
        # SOILEV rounds on the scale of its stage sums, U and WINF as well (spam_evap_dssat.check_soilev)
        soil = eo_ref
        if g[0].mesev == "R":
            soil = np.maximum.reduce(
                [
                    eo_ref,
                    *(
                        h.cat(g, lambda r, n=n: r.tables["soilev_in"].values[n])
                        for n in ("SUMES1", "SUMES2", "U", "WINF")
                    ),
                ]
            )
        for k, name in ref_name.items():
            ref = h.cat(g, lambda r, name=name: r.spam_out[name])
            e = h.ratio(out[k], ref, soil if k in ("ES", "EVAP", "EOP") else eo_ref)
            worst[k] = max(worst.get(k, 0.0), float(e.max()))
        n_days += len(eo_ref)
    assert n_days == sum(len(r.dates) for r in runs) == 11138
    for k, lim in chain.items():
        assert worst[k] <= lim, (k, worst[k], lim)


# ------------------------------------------------------------------------------------ 5
def test_soilev_free_running_store_batched_over_the_runs(runs: list[h.Run]) -> None:
    """``soil_water/soilev`` carries its store from the reference's SEASINIT; lax.scan over the
    days, vmap over the 39 MESEV = R runs (padded days: EOS_SOIL = 0, not called)."""
    import equinox as eqx

    from agrijax.iface.surface import PETFluxes
    from agrijax.processes.soil_water.bucket_evap import (
        SoilEvapParams,
        SoilEvapState,
        soil_evaporation_soilev,
    )

    rr = [r for r in runs if r.mesev == "R"]
    nd = max(len(r.dates) for r in rr)

    def pad(a: np.ndarray) -> np.ndarray:
        a = h.f64(a)
        w = [(0, nd - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
        return np.pad(a, w)

    def col(get) -> jnp.ndarray:
        return jnp.asarray(np.stack([pad(get(r)) for r in rr]))

    days = {
        "sw": col(lambda r: r.spam_in["SW"]),
        "swdelts": col(lambda r: r.spam_in["SWDELTS"]),
        "swdeltu": col(lambda r: r.spam_in["SWDELTU"]),
        "winf": col(lambda r: r.spam_in["WINF"]),
        "dlayr": col(lambda r: r.spam_in["DLAYR_S"]),
        "ll": col(lambda r: r.spam_in["LL_S"]),
        "eos_soil": col(lambda r: r.spam_out["EOS_SOIL_L"]),
        "em": col(lambda r: r.spam_out["EM_L"]),
    }
    init = {
        n: jnp.asarray(np.array([h.f64(r.tables["soilev_init"].values[n])[0] for r in rr]))
        for n in ("SUMES1", "SUMES2", "T", "SWEF")
    }
    u = jnp.asarray(np.array([h.f64(r.spam_in["U_S"])[0] for r in rr]))

    def season(d, s1, s2, t, swef, u):
        z = jnp.zeros(())
        zl = jnp.zeros(d["sw"].shape[-1])
        st0 = SoilEvapState(
            sw=zl,
            swdelts=zl,
            swdeltu=zl,
            winf=z,
            dlayr=zl,
            ds=zl,
            dul=zl,
            ll=zl,
            mulch_mass=z,
            mulch_cover=z,
            mulch_water=z,
            mulch_am=z,
            mulch_extfac=z,
            sumes1=s1,
            sumes2=s2,
            t=t,
            swef=swef,
            eos_soil=z,
            em=z,
            es=z,
            es_lyr=zl,
            upflow=zl,
            evap=z,
            pet=PETFluxes.zeros(),
        )
        p = SoilEvapParams(u=u, pmfraction=z, mulch_active=True)

        def step(st, x):
            st = eqx.tree_at(
                lambda s: (s.sw, s.swdelts, s.swdeltu, s.winf, s.dlayr, s.ll, s.eos_soil, s.em),
                st,
                tuple(x[k] for k in ("sw", "swdelts", "swdeltu", "winf", "dlayr", "ll", "eos_soil", "em")),
            )
            st = soil_evaporation_soilev(st, p, None)
            return st, (st.es, st.sumes1, st.sumes2)

        return jax.lax.scan(step, st0, d)[1]

    es, s1, s2 = jax.jit(jax.vmap(season))(days, init["SUMES1"], init["SUMES2"], init["T"], init["SWEF"], u)
    n = 0
    for i, r in enumerate(rr):
        m = len(r.dates)
        so = r.tables["soilev_out"].values
        sc = np.maximum.reduce(
            [
                h.f64(r.spam_out["EOS_SOIL_L"]),
                h.f64(so["SUMES1"]),
                h.f64(so["SUMES2"]),
                h.f64(r.spam_in["U_S"]),
                h.f64(r.spam_in["WINF"]),
            ]
        )
        # per-day REAL*4 limits accumulate (the stage sums pass an error on with gain <= 1)
        bound = h.OPS["SUMES2"] * np.cumsum(sc) * h.U32
        for got, ref in ((es, so["ES"]), (s1, so["SUMES1"]), (s2, so["SUMES2"])):
            d = np.abs(np.asarray(got[i, :m]) - h.f64(ref))
            assert np.all(d <= bound + h.OPS["ES"] * h.U32 * sc), (r.key, float((d - bound).max()))
        n += m
    assert n == sum(len(r.dates) for r in rr)


# ------------------------------------------------------------------------------------ 6
def test_dumped_values_are_the_printed_et_outputs(runs: list[h.Run], tables: Path) -> None:
    cols = {"EOAA": "EO", "EOSA": "EOS", "ESAA": "ES", "EMAA": "EM_L", "EOPA": "EOP"}
    for r in runs:
        et = read_out(tables / "_runs" / r.key / "ET.OUT")
        k = np.asarray(et["YEAR"], int) * 1000 + np.asarray(et["DOY"], int)
        j = np.searchsorted(r.dates, k)
        ok = (j < len(r.dates)) & (r.dates[np.minimum(j, len(r.dates) - 1)] == k)
        assert ok.sum() >= len(r.dates) - 1, r.key
        for c, name in cols.items():
            d = np.abs(h.f64(r.spam_out[name])[j[ok]] - np.asarray(et[c], float)[ok])
            assert d.max() <= 0.0005 + 1e-6, (r.key, c, float(d.max()))
