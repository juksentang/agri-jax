"""Contract smoke test (W0-C): one crop day and one winter day of CA-TPA through the M3 skeleton day.

Not an acceptance test and nothing is tuned. It checks that the coupling contract fits the
modules that exist today: the day of :mod:`agrijax.models.day_rzwqm46` binds the soil-water day,
ROOTWU, CERES-Maize (``nstress_replay`` growth) and the crop-interface adapters on the contract's
ports, and stands in for every missing producer (PET to the port, snow, the uptake limit WUF, the
canopy record, crop nitrogen) with a replay of the RZWQM2 4.6 reference run.

Data (private, ``allow_skip``):

* ``<data>/dumps/tables/rzwqm46_catpa2015_2023/*.npz``: the A12 daily entry/exit tables of the
  instrumented RZWQM2 4.6 run of CA-TPA 2015-2023 (PHYSCL, MAPLNT, DSSATDRV, ROOTWU, MZ_GROSUB);
* ``<data>/dumps/_runs/catpa2015_2023/CA-TPA.ana``: that run's daily output;
* ``<data>/narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario``: ``rzwqm.dat``, ``CA-TPA.BRK`` and the
  embedded crop's ``MZCER040.{CUL,ECO,SPE}``; ``$AGRI_JAX_DSSAT/source/Data/Genotype/MZCER048.ECO``
  for the two ecotype values the 4.0 file lacks (``TSEN``, ``CDAY``).

Days: 2015-195 (crop day: stage 3, LAI 4.5, a 2.2 cm storm, WUF < 1) and 2015-076 (winter day:
snowpack, 0.49 cm snowmelt, no rain, before the 2015-104 tillage that changes the soil). The
morning state is the reference's PHYSCL entry (soil water, yesterday's canopy, TRWUP and node
uptake); the crop state of the crop day is our CERES-Maize driven by the reference's own crop
water and nitrogen stress from sowing (2015-118) to the day before.

Checked on each day: ``Day.check`` passes (and which allowed lags it uses), every state leaf is
finite, the ledger closes (float64, ``AGRI_JAX_CHECK=1`` asserts it inside the run too), and the
replay binding of the crop water port P1 (whose producers exist) gives the crop bit for bit what
the coupled binding gives. Reference facts the contract relies on are checked on the whole run.
The measured day values are printed (``-s``) for the W0 report; none is asserted against the
reference.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agrijax.core import Model, bind, compose, run
from agrijax.core.grids import rzwqm_lyrset, rzwqm_nodes
from agrijax.core.ports import Binding
from agrijax.iface.crop import CanopyRecord, CropNIn, CropWaterIn, RootRecord
from agrijax.iface.soil import NodeUptake, SinkInputs
from agrijax.iface.surface import PETFluxes, SnowOut
from agrijax.io.dssat import read_cul, read_eco, read_spe
from agrijax.io.rzwqm import read_ana, read_brk, read_rzwqm_dat
from agrijax.models.day_rzwqm46 import (
    SLOT,
    CropIfaceParams,
    day_processes,
    day_rzwqm46,
    initial_state,
    publish_uptake_entry,
    replay_entry,
)
from agrijax.port import dumps
from agrijax.processes.crop.ceres_maize import (
    CROP_PROCESSES_NSTRESS_REPLAY,
    CeresForcing,
    CeresMaizeState,
    ceres_water_replay,
)
from agrijax.processes.crop.ceres_maize._util import daylength, twilight_daylength
from agrijax.processes.crop.ceres_maize.coefficients import BSGDD, CANHT_POT, DSSAT_COEFFICIENTS
from agrijax.processes.crop.ceres_maize.state import CeresCultivar, CeresMaizeParams, CeresSoil, CeresSpecies
from agrijax.processes.n_supply import CropNReplayForcing, CropNReplayState, crop_n_replay
from agrijax.processes.soil_water.day import DayConfig, SoilWaterDayParams
from agrijax.processes.soil_water.hydraulics import SoilHydraulicParams
from agrijax.processes.soil_water.infiltration import GreenAmptConfig, GreenAmptParams, StormForcing
from agrijax.processes.soil_water.richards import RichardsConfig, RichardsGrid, RichardsParams, SoilWater
from agrijax.processes.soil_water.uptake import RootwuParams, RootwuState

pytestmark = [
    pytest.mark.slow,
    pytest.mark.allow_skip(reason="needs the private RZWQM2 A12 tables of CA-TPA 2015-2023"),
    pytest.mark.skipif(not jax.config.read("jax_enable_x64"), reason="the smoke day runs in float64"),
]

TABLES = Path("dumps/tables/rzwqm46_catpa2015_2023")
ANA = Path("dumps/_runs/catpa2015_2023/CA-TPA.ana")
SCENARIO = Path("narval_mirror/RZWQM_sw_batch/CA-TPA/Scenario")
DSSAT_ENGINE = Path(
    os.environ.get("AGRI_JAX_DSSAT", "~/AFSoil/Formal_Analysis/02_DSSAT/dssat_engine")
).expanduser()
CROP_DAY, WINTER_DAY = 2015195, 2015076
FIRST_DAY = np.datetime64("2015-01-01")
#: the Richards / day numerics of the smoke day: the converged 96 x 8 of M1 with every event at t = 0
N_SUB, N_ITER = 96, 8
HOURS = 24.0  # RZWQM2 qsr is per hour: node amount [cm d-1] = qsr * 24 * tl
SW_KEYS = ("sw", "eop", "trwup")


def _yrdoy_next(d: int) -> int:
    t = datetime.date(d // 1000, 1, 1) + datetime.timedelta(days=d % 1000)
    return t.year * 1000 + t.timetuple().tm_yday


def _f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


class Reference:
    """The reference tables, the scenario inputs and the parameters built from them."""

    def __init__(self, data_dir: Path) -> None:
        tab, sc = data_dir / TABLES, data_dir / SCENARIO
        need = [tab / "physcl_entry.npz", data_dir / ANA, sc / "rzwqm.dat", sc / "MZCER040.CUL"]
        need.append(DSSAT_ENGINE / "source/Data/Genotype/MZCER048.ECO")
        for p in need:
            if not p.is_file():
                pytest.skip(f"{p} not found")
        self.t = {
            n: dumps.load_table(tab / f"{n}.npz")[0]
            for n in (
                "physcl_entry",
                "physcl_exit",
                "maplnt_exit",
                "dssatdrv_entry",
                "dssatdrv_exit",
                "rootwu_entry",
                "rootwu_exit",
                "mz_grosub_exit",
            )
        }
        pe = self.t["physcl_exit"].values
        self.nn = int(pe["NN"][0])
        self.nl = int(self.t["rootwu_entry"].values["NLAYR"][0])
        self.tl = _f64(pe["TL"][0, : self.nn])
        self.nodes = rzwqm_nodes(_f64(pe["TLT"][0, : self.nn]))
        self.layers = rzwqm_lyrset(self.nodes)
        # soil water: grid and Brooks-Corey parameters of rzwqm.dat (untilled; gap G15 after 2015-104)
        dat = read_rzwqm_dat(sc / "rzwqm.dat")
        tlt = dat.node_depths_cm
        grid = RichardsGrid.from_rzwqm(tlt, dat.node_spacing_cm)
        nh = np.searchsorted(dat.horizon_depths_cm, tlt, side="left")
        self.hyd = SoilHydraulicParams.from_rzwqm_dat(dat.hydraulics, node_horizon=nh)
        np.testing.assert_allclose(np.asarray(grid.tl), self.tl, atol=1e-12)
        self.soil = SoilWaterDayParams(
            richards=RichardsParams(
                soil=self.hyd, grid=grid, config=RichardsConfig(n_sub=N_SUB, n_iter=N_ITER)
            ),
            infiltration=GreenAmptParams(
                aef=jnp.asarray(float(self.t["physcl_entry"].values["AEF"][0])),
                config=GreenAmptConfig.for_grid(np.asarray(grid.tl)),
            ),
            config=DayConfig(n_pre=0, n_post=N_SUB),
        )
        # .ana daily output and the breakpoint storms
        ana = read_ana(data_dir / ANA)
        cols = {int(k): v for k, v in ana.attrs["columns"].items()}
        self.ana = {n: _f64(ana[cols[n]].values) for n in (2, 3, 5, 6, 7, 10, 12, 92, 105)}
        self.days = np.asarray(ana.time.values[1:], dtype="datetime64[D]")
        from test_infiltration_catpa import storm_forcing_from_breakpoints

        brk = read_brk(sc / "CA-TPA.BRK")
        self.storm = storm_forcing_from_breakpoints(self.days, brk.events, brk.breakpoints)
        self.crop = self._ceres_params(sc)
        rw = self.t["rootwu_entry"].values
        self.rootwu = RootwuParams(
            dlayr=jnp.asarray(_f64(rw["DLAYR"][0, : self.nl])),
            ll=jnp.asarray(_f64(rw["LL"][0, : self.nl])),
            sat=jnp.asarray(_f64(rw["SAT"][0, : self.nl])),
        )
        np.testing.assert_array_equal(self.layers.thickness, _f64(rw["DLAYR"][0, : self.nl]))
        self.iface = CropIfaceParams(nodes=self.nodes, layers=self.layers, ll=self.rootwu.ll)
        self.lat = float(self.t["dssatdrv_exit"].values["XLATR"][0])

    # ------------------------------------------------------------------ inputs
    def _ceres_params(self, sc: Path) -> CeresMaizeParams:
        """CA-TPA CERES-Maize parameters from the embedded crop's files and the dumped soil.

        The embedded crop is DSSAT 4.0 (``MZCER040``); its ecotype file has no ``TSEN``/``CDAY``
        (taken from the 4.8.6 ``MZCER048.ECO`` row of the same ecotype), and its species file
        differs from 4.8.6 in RGFIL and the CO2 table (the 4.0 values are used).
        """
        dd = self.t["dssatdrv_exit"].values  # the entry of the first crop day precedes the crop's reads
        vr = (
            bytes(dd["VRNAME"][0]).decode().strip()
            if dd["VRNAME"].dtype.kind == "S"
            else str(dd["VRNAME"][0])
        )
        cul_tab = read_cul(sc / "MZCER040.CUL")
        cul = cul_tab[cul_tab["VRNAME"].str.strip() == vr.strip()].iloc[0]
        for k in ("P1", "P2", "P5", "G2", "G3", "PHINT"):
            assert float(np.float32(cul[k])) == float(dd[f"PLANTVAR%{k}"][0]), k
        eco = read_eco(sc / "MZCER040.ECO").loc[cul["ECO#"]]
        eco48 = read_eco(DSSAT_ENGINE / "source/Data/Genotype/MZCER048.ECO").loc[cul["ECO#"]]
        spe = read_spe(sc / "MZCER040.SPE")

        def a(x: Any) -> jnp.ndarray:
            return jnp.asarray(np.asarray(x, dtype=float))

        cultivar = CeresCultivar(
            **{k.lower(): a(cul[k]) for k in ("P1", "P2", "P5", "G2", "G3", "PHINT")},
            **{k.lower(): a(eco[k]) for k in ("TBASE", "TOPT", "ROPT", "DJTI", "GDDE", "DSGFT", "RUE")},
            p2o=a(eco["P20"]),
            tsen=a(eco48["TSEN"]),
            cday=a(eco48["CDAY"]),
        )
        s = {k: spe[k] for k in spe.params}
        species = CeresSpecies(
            **{
                k.lower(): a(s[k])
                for k in (
                    "PRFTC",
                    "RGFIL",
                    "PARSR",
                    "CO2X",
                    "CO2Y",
                    "FSLFW",
                    "RSGR",
                    "RSGRT",
                    "CARBOT",
                    "DSGT",
                    "DGET",
                    "SWCG",
                    "STMWTE",
                    "RTWTE",
                    "LFWTE",
                    "SEEDRVE",
                    "LEAFNOE",
                    "PLAE",
                    "RLWR",
                    "RWUEP1",
                )
            },
            pormin=a(s["PORM"]),
            rwumx=a(s["RWMX"]),
            canht_pot=a(CANHT_POT),
            bsgdd=a(BSGDD),
        )
        nl = self.nl
        soil = CeresSoil(
            dlayr=a(dd["SOILPROP%DLAYR"][0, :nl]),
            ll=a(dd["SOILPROP%LL"][0, :nl]),
            dul=a(dd["SOILPROP%DUL"][0, :nl]),
            sat=a(dd["SOILPROP%SAT"][0, :nl]),
            shf=a(dd["SOILPROP%WR"][0, :nl]),
            slpf=a(dd["SOILPROP%SLPF"][0]),
        )
        yrplt = int(self.t["mz_grosub_exit"].values["YRPLT"][0])
        return CeresMaizeParams(
            cultivar=cultivar,
            species=species,
            soil=soil,
            pltpop=a(dd["PLANTVAR%PLTPOP"][0]),
            sdepth=a(dd["PLANTVAR%SDEPTH"][0]),
            rowspc=a(dd["PLANTVAR%ROWSPC"][0]),
            yrplt=jnp.asarray(yrplt, dtype=jnp.int32),
            coefficients=DSSAT_COEFFICIENTS.as_arrays(),
        )

    def row(self, name: str, day: int) -> int:
        j = int(self.t[name].index_of([day])[0])
        assert j >= 0, (name, day)
        return j

    def v(self, name: str, var: str, day: int) -> np.ndarray:
        return self.t[name].values[var][self.row(name, day)]

    def ana_i(self, day: int) -> int:
        d = FIRST_DAY + np.timedelta64(
            (
                datetime.date(day // 1000, 1, 1)
                + datetime.timedelta(days=day % 1000 - 1)
                - datetime.date(2015, 1, 1)
            ).days,
            "D",
        )
        return int(np.nonzero(self.days == d)[0][0])

    def crop_forcing(self, days: list[int], *, replay: bool) -> CeresForcing:
        """The crop's daily forcing: weather the embedded crop received (DSSATDRV) on crop
        days, PHYSCL's otherwise; with ``replay`` the crop water of the reference (ROOTWU entry SW,
        DSSATDRV exit EOP, ROOTWU exit TRWUP), else zeros (unused in the coupled day, gap G31)."""
        dde, pe = self.t["dssatdrv_exit"], self.t["physcl_exit"]
        n, nl = len(days), self.nl
        tmax, tmin, srad, co2 = (np.zeros(n) for _ in range(4))
        sw, eop, trwup = np.zeros((n, nl)), np.zeros(n), np.zeros(n)
        for k, d in enumerate(days):
            j = int(dde.index_of([d])[0])
            if j >= 0:
                v = dde.values
                tmax[k], tmin[k], srad[k], co2[k] = v["TMAXR"][j], v["TMINR"][j], v["SRADR"][j], v["CO2R"][j]
                if replay:
                    sw[k] = self.v("rootwu_entry", "SW", d)[:nl]
                    eop[k] = self.v("dssatdrv_exit", "EOP", d)
                    trwup[k] = self.v("rootwu_exit", "TRWUP", d)
            else:
                i = self.row("physcl_exit", d)
                tmax[k], tmin[k], srad[k] = pe.values["TMAX"][i], pe.values["TMIN"][i], pe.values["RTS"][i]
                co2[k] = float(dde.values["CO2R"][0])
        doy = jnp.asarray(np.asarray([d % 1000 for d in days], dtype=float))
        return CeresForcing(
            yrdoy=jnp.asarray(np.asarray(days, dtype=np.int32)),
            tmax=jnp.asarray(tmax),
            tmin=jnp.asarray(tmin),
            srad=jnp.asarray(srad),
            dayl=daylength(doy, self.lat),
            twilen=twilight_daylength(doy, self.lat),
            co2=jnp.asarray(co2),
            snow=jnp.zeros(n),
            sw=jnp.asarray(sw),
            eop=jnp.asarray(eop),
            trwup=jnp.asarray(trwup),
        )

    def nstres(self, days: list[int]) -> CropNReplayForcing:
        mg = self.t["mz_grosub_exit"]
        j = mg.index_of(days)
        v = np.where(j >= 0, _f64(mg.values["NSTRES"])[np.maximum(j, 0)], 1.0)
        return CropNReplayForcing(nstres=jnp.asarray(v))

    # ------------------------------------------------------------------ the crop before the day
    def spin_up(self, day: int) -> tuple[CeresMaizeState, RootRecord]:
        """Our CERES-Maize from sowing to ``day - 1`` driven by the reference's crop water and
        NSTRES (the replay bindings of P1 and P10): its state and root record on the morning of ``day``."""
        yrplt = int(self.crop.yrplt)
        days = [yrplt]
        while _yrdoy_next(days[-1]) < day:
            days.append(_yrdoy_next(days[-1]))
        assert np.all(self.t["rootwu_entry"].index_of(days) >= 0)
        ports = {
            "water_in": f"iface.crop_water.{SLOT}",
            "root_out": f"iface.root.{SLOT}",
            "n_in": f"iface.crop_n.{SLOT}",
        }
        own = f"crops.{SLOT}"
        procs = [
            bind(
                ceres_water_replay, own=own, ports=ports, forcing="crop", params="crop", name=f"{own}.water"
            ),
            bind(
                crop_n_replay,
                own=f"{own}_n_supply",
                ports={"n_out": ports["n_in"]},
                forcing="n",
                name="n.replay",
            ),
            *(
                bind(p, own=own, ports=ports, forcing="crop", params="crop", name=f"{own}.{k}")
                for k, p in enumerate(CROP_PROCESSES_NSTRESS_REPLAY)
            ),
        ]
        s0 = CeresMaizeState.initial(self.crop, 1).replace(n_in=CropNIn.initial(1))
        g0 = compose(
            {
                **Binding(own, tuple(ports.items())).entries(s0),
                f"{own}_n_supply": CropNReplayState(),
            }
        )
        forcing = {"crop": self.crop_forcing(days, replay=True), "n": self.nstres(days)}
        final, _ = jax.jit(lambda p, f, s: run(Model(None, procs, outputs=()), p, f, s, return_final=True))(
            {"crop": self.crop}, forcing, g0
        )
        return final["crops"][SLOT], final["iface"]["root"][SLOT]

    # ------------------------------------------------------------------ the day
    def day_inputs(self, day: int) -> tuple[dict, dict, dict]:
        """``(params, forcing [T = 1], morning state)`` of the skeleton day ``day``."""
        nn, nl, tl = self.nn, self.nl, self.tl
        pin, pout = self.t["physcl_entry"], self.t["physcl_exit"]
        i_in, i_out = self.row("physcl_entry", day), self.row("physcl_exit", day)
        ia = self.ana_i(day)
        a = self.ana
        snow_day = bool(a[92][ia] > 0.0 or a[92][ia + 1] > 0.0 or a[105][ia + 1] > 0.0)
        one = lambda x: jnp.asarray(np.asarray(x, dtype=float))[None]  # noqa: E731
        # --- replay records of today (the missing producers)
        vo = pout.values
        pet = PETFluxes(
            transpiration=one(vo["PET"][i_out]),
            soil_evaporation=one(vo["PES"][i_out]),
            residue_evaporation=one(vo["PER"][i_out]),
            reference_short=one(0.0),
            reference_tall=one(0.0),
            eo_priestley_taylor=one(0.0),
        )
        # snow: the reference tables carry no melt split, so the replay uses .ana: on a snow day
        # the infiltration (col 5) is the melt that enters the soil and the runoff (col 12) the
        # melt runoff; SWE col 92 [cm] -> mm; sublimation = SWE drop - melt (col 105)
        melt = a[5][ia + 1] if snow_day else 0.0
        snow = SnowOut(
            melt=one(melt),
            melt_runoff=one(a[12][ia + 1] if snow_day else 0.0),
            swe=one(a[92][ia + 1] * 10.0),
            sublimation=one(max(a[92][ia] - a[92][ia + 1] - a[105][ia + 1], 0.0)),
        )
        z = np.zeros(nn)
        q_wuf = _f64(vo["QSR"][i_out, :nn]) * HOURS * tl  # the uptake after WUF, as node amounts
        sink = SinkInputs(
            uptake=one(q_wuf), tile=one(z), lateral=one(z), subirrigation=one(z), macropore_to_drain=one(z)
        )
        mx = self.t["maplnt_exit"]
        j = self.row("maplnt_exit", day)
        canopy = CanopyRecord(
            lai=one([mx.values["LAI"][j]]),
            tlai=one([mx.values["TLAI"][j]]),
            height=one([mx.values["HEIGHT"][j]]),
        )
        k = ia
        storm = StormForcing(
            ts0=self.storm.ts0[k : k + 1],
            duration=self.storm.duration[k : k + 1] * (0.0 if snow_day else 1.0),
            depth=self.storm.depth[k : k + 1] * (0.0 if snow_day else 1.0),
        )
        forcing = {
            "soil": storm,
            "crop": self.crop_forcing([day], replay=False),
            "n": self.nstres([day]),
            "replay": {"pet": pet, "snow": snow, "sink_in": sink, "canopy": canopy},
        }
        # --- morning state: PHYSCL entry soil water, yesterday's lagged records
        vi = pin.values
        water = SoilWater.from_theta(jnp.asarray(_f64(vi["THETA"][i_in, :nn])), self.hyd)
        crop_day = int(self.t["rootwu_entry"].index_of([day])[0]) >= 0
        if crop_day:
            crop, root = self.spin_up(day)
            tss = _f64(self.v("rootwu_entry", "TSS", day)[:nl])[None]
        else:
            crop = CeresMaizeState.initial(self.crop, 1)
            root = RootRecord.zeros(1, nl)
            tss = np.zeros((1, nl))
        z1 = lambda x: jnp.asarray(np.asarray([x], dtype=float))  # noqa: E731
        records = {
            "pet": PETFluxes(**{n: jnp.zeros(()) for n in PETFluxes.__dataclass_fields__}),
            "snow": SnowOut.zeros(),
            "canopy": CanopyRecord(
                lai=z1(vi["LAI"][i_in]), tlai=z1(vi["TLAI"][i_in]), height=z1(vi["HEIGHT"][i_in])
            ),
            "crop_water": CropWaterIn(sw=jnp.zeros(nl), eop=jnp.zeros(1), trwup=z1(vi["TRWUP"][i_in])),
            "root": root,
            "root_uptake": NodeUptake(uptake=jnp.asarray(_f64(vi["QSR"][i_in, :nn]) * HOURS * tl)[None]),
            "crop_n": CropNIn.initial(1),
        }
        state = initial_state(
            water=water,
            sink_in=SinkInputs.zeros(nn),
            crop=crop,
            rootwu=RootwuState(tss=jnp.asarray(tss), rwu=jnp.zeros((1, nl))),
            records=records,
            storage0=jnp.sum(water.theta * jnp.asarray(tl)) + water.pond,
        )
        params = {"soil": self.soil, "crop": self.crop, "rootwu": self.rootwu, "crop_iface": self.iface}
        return params, forcing, state


@pytest.fixture(scope="module")
def ref(data_dir: Path) -> Reference:
    return Reference(data_dir)


def _run(model: Model, params: dict, forcing: dict, state: dict) -> dict:
    final, _ = jax.jit(lambda p, f, s: run(model, p, f, s, return_final=True))(params, forcing, state)
    return final


@pytest.fixture(scope="module", params=[CROP_DAY, WINTER_DAY], ids=["crop_day", "winter_day"])
def smoke(request: pytest.FixtureRequest, ref: Reference) -> dict[str, Any]:
    day = int(request.param)
    params, forcing, state = ref.day_inputs(day)
    d = day_rzwqm46(SLOT)
    model = d.compile(day_processes(SLOT))
    report = d.check(model)
    final = _run(model, params, forcing, state)
    return {"day": day, "dayd": d, "model": model, "report": report, "params": params, "forcing": forcing,
            "state": state, "final": final}  # fmt: skip


def _leaves(tree: Any) -> list[np.ndarray]:
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


def test_day_check_passes_with_the_contract_lags(smoke: dict) -> None:
    r = smoke["report"]
    assert set(r.used) == {
        ("pet.sw_daily", "iface.canopy.maize"),
        ("pet.sw_daily", "soil_water.theta"),
        ("soil_water.uptake_limit", "iface.root_uptake.maize"),
        ("soil_water.uptake_limit", "iface.crop_water.maize.trwup"),
    }
    assert r.unused_pairs == (("crops.maize.rootwu", "iface.root.maize"),)


def test_every_state_leaf_is_finite(smoke: dict) -> None:
    bad = [
        i for i, x in enumerate(_leaves(smoke["final"])) if x.dtype.kind == "f" and not np.all(np.isfinite(x))
    ]
    assert not bad


def test_ledger_closes(smoke: dict) -> None:
    led = smoke["final"]["ledger"]["water"]
    flux = smoke["final"]["soil_water"]["flux"]
    res = float(led.residual)
    scale = float(led.storage) + sum(
        abs(float(v.total)) for v in (*led.inflow.values(), *led.outflow.values())
    )
    assert abs(res) <= 1e-10 + 1e-12 * scale, res
    assert abs(float(led.closure())) <= 1e-10 + 1e-12 * scale
    assert abs(res - float(flux.balance_error)) < 1e-12
    print(
        json.dumps(
            {"day": smoke["day"], "ledger_residual_cm": res, "balance_error_cm": float(flux.balance_error)}
        )
    )


def _p1_replay(values: dict) -> dict:
    """Replay entries for the three P1 producers, fed from ``replay.p1`` of the forcing."""
    c = f"crops.{SLOT}"
    return {
        f"{c}.remap_in": replay_entry(f"{c}.remap_in", {f"iface.crop_water.{SLOT}.sw": "replay.p1.sw"}),
        f"{c}.eop": replay_entry(f"{c}.eop", {f"iface.crop_water.{SLOT}.eop": "replay.p1.eop"}),
        f"{c}.rootwu": replay_entry(
            f"{c}.rootwu",
            {
                f"iface.crop_water.{SLOT}.trwup": "replay.p1.trwup",
                f"{c}_rootwu.tss": "replay.p1.tss",
                f"{c}_rootwu.rwu": "replay.p1.rwu",
            },
        ),
    }


def test_replay_and_coupled_binding_of_p1_are_bit_identical(smoke: dict) -> None:
    """The crop, the root record, the node uptake and the ledger are the same bits whether the
    crop water record was produced in the day (coupled) or replayed from the coupled run's values."""
    fin = smoke["final"]
    cw = fin["iface"]["crop_water"][SLOT]
    rw = fin["crops"][f"{SLOT}_rootwu"]
    p1 = {
        "sw": cw.sw[None],
        "eop": cw.eop[None],
        "trwup": cw.trwup[None],
        "tss": rw.tss[None],
        "rwu": rw.rwu[None],
    }
    forcing = {**smoke["forcing"], "replay": {**smoke["forcing"]["replay"], "p1": p1}}
    model = smoke["dayd"].compile(day_processes(SLOT, replace=_p1_replay(p1)))
    rep = _run(model, smoke["params"], forcing, smoke["state"])
    for path in (f"crops.{SLOT}", f"iface.root.{SLOT}", f"iface.root_uptake.{SLOT}", f"iface.crop_water.{SLOT}",
                 f"crops.{SLOT}_rootwu", "ledger.water", "soil_water"):  # fmt: skip
        a = _leaves(_get(fin, path))
        b = _leaves(_get(rep, path))
        assert len(a) == len(b) and all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True)), path


def _get(tree: Any, path: str) -> Any:
    from agrijax.core.state import get_path

    return get_path(tree, path)


def test_day_values_against_the_reference_are_recorded(smoke: dict, ref: Reference) -> None:
    """Measured, not asserted against the reference (a smoke test): storage, fluxes, the crop
    water record and the node uptake of the day, next to the reference's values."""
    day, fin, st0 = smoke["day"], smoke["final"], smoke["state"]
    tl = ref.tl
    sw = fin["soil_water"]
    flux = sw["flux"]
    theta_ref = _f64(ref.v("physcl_exit", "THETA", day)[: ref.nn])
    out: dict[str, Any] = {
        "day": day,
        "storage_start_cm": float(np.sum(np.asarray(st0["soil_water"]["theta"]) * tl)),
        "storage_end_cm": float(np.sum(np.asarray(sw["theta"]) * tl)),
        "storage_end_ref_cm": float(np.sum(theta_ref * tl)),
        "theta_rmse_vs_ref": float(np.sqrt(np.mean((np.asarray(sw["theta"]) - theta_ref) ** 2))),
        **{
            f"flux_{k}_cm": float(getattr(flux, k))
            for k in (
                "infiltration",
                "evaporation",
                "evaporation_deficit",
                "drainage",
                "uptake",
                "uptake_cut",
                "runoff",
                "rain",
                "event_infiltration",
                "event_runoff",
                "n_clamp",
            )
        },
    }
    ia = ref.ana_i(day) + 1
    out |= {"ana_" + k: float(ref.ana[c][ia]) for k, c in (
        ("infiltration", 5), ("evaporation", 6), ("transpiration", 7), ("deep_seepage", 10), ("runoff", 12))}  # fmt: skip
    cw = fin["iface"]["crop_water"][SLOT]
    out["eop_mm"] = float(cw.eop[0])
    out["trwup_cm"] = float(cw.trwup[0])
    out["node_uptake_next_day_cm"] = float(np.sum(np.asarray(fin["iface"]["root_uptake"][SLOT].uptake)))
    if int(ref.t["rootwu_entry"].index_of([day])[0]) >= 0:
        eop_ref = float(ref.v("dssatdrv_exit", "EOP", day))
        # contract F10 (EOP = 10 PET): the adapter against the reference's own EOP (REAL*4)
        assert abs(out["eop_mm"] - eop_ref) <= 1e-6 * max(eop_ref, 1.0), (out["eop_mm"], eop_ref)
        out |= {
            "eop_ref_mm": eop_ref,
            "trwup_ref_cm": float(ref.v("rootwu_exit", "TRWUP", day)),
            "sw_ours": np.asarray(cw.sw).round(5).tolist(),
            "sw_ref_rootwu_entry": _f64(ref.v("rootwu_entry", "SW", day)[: ref.nl]).round(5).tolist(),
            "node_uptake_ref_next_day_cm": float(
                np.sum(_f64(ref.v("dssatdrv_exit", "QSR", day)[: ref.nn]) * HOURS * tl)
            ),
            "lai_ours": float(fin["crops"][SLOT].growth.lai[0]),
            "lai_ref": float(ref.v("dssatdrv_exit", "XLAI", day)),
            "canht_ours_cm": float(fin["crops"][SLOT].growth.canht[0]) * 100.0,
            "height_ref_cm": float(ref.v("dssatdrv_exit", "HEIGHT", day)),
            "rtdep_ours_cm": float(fin["crops"][SLOT].roots.rtdep[0]),
            "istage_ours": int(fin["crops"][SLOT].phen.istage[0]),
            "istage_ref": int(ref.v("dssatdrv_exit", "ISTAGE", day)),
            "nstres_replayed": float(fin["iface"]["crop_n"][SLOT].nstres[0]),
        }
    print("W0C " + json.dumps(out))


# ------------------------------------------------------------------ reference facts the contract relies on
def _prev_rows(t_prev: dumps.DailyTable, t_next: dumps.DailyTable) -> tuple[np.ndarray, np.ndarray]:
    """Rows ``(i, j)``: ``t_prev`` row ``i`` is the day before ``t_next`` row ``j``."""
    nxt = np.asarray([_yrdoy_next(int(d)) for d in t_prev.date])
    j = t_next.index_of(nxt)
    i = np.nonzero(j >= 0)[0]
    return i, j[i]


def test_reference_canopy_read_by_pet_is_yesterdays_maplnt_exit(ref: Reference) -> None:
    """P6 (lag 1): the canopy PHYSCL reads on day d is MAPLNT's exit canopy of d - 1 exactly; the
    embedded crop's own exit values (DSSATDRV) differ on harvest days and after maturity in
    2020-2021, so the P6 producer is the plant mapping after the crop, not the crop alone."""
    pin, mx, dd = ref.t["physcl_entry"], ref.t["maplnt_exit"], ref.t["dssatdrv_exit"]
    i, j = _prev_rows(mx, pin)
    assert len(i) > 3000
    for k in ("LAI", "TLAI", "HEIGHT"):
        np.testing.assert_array_equal(pin.values[k][j], mx.values[k][i], err_msg=k)
    m = mx.index_of(dd.date)
    diff = {
        k: int(np.sum(mx.values[k][m] != dd.values[kk]))
        for k, kk in (("LAI", "XLAI"), ("TLAI", "TLAI"), ("HEIGHT", "HEIGHT"))
    }
    print("W0C " + json.dumps({"maplnt_vs_dssatdrv_exit_days_differing": diff, "crop_days": len(dd.date)}))
    assert 0 < diff["LAI"] < 50


def test_reference_trwup_read_by_wuf_is_yesterdays_crop_exit(ref: Reference) -> None:
    """P1 ``trwup`` (lag 1 for the uptake limit): PHYSCL's entry TRWUP is DSSATDRV's exit TRWUP of
    d - 1 on every crop day. That is ROOTWU's exit TRWUP except on the harvest days (one per
    season), where the crop driver zeroes it: the harvest reset of the contract (section 7.3)."""
    pin, rx, dx = ref.t["physcl_entry"], ref.t["rootwu_exit"], ref.t["dssatdrv_exit"]
    np.testing.assert_array_equal(rx.date, dx.date)
    i, j = _prev_rows(dx, pin)
    assert len(i) > 1000
    np.testing.assert_array_equal(pin.values["TRWUP"][j], dx.values["TRWUP"][i])
    off = np.nonzero(rx.values["TRWUP"][i] != dx.values["TRWUP"][i])[0]
    harvest = sorted(int(d) for d in dx.date[i][off])
    print("W0C " + json.dumps({"trwup_reset_days": harvest}))
    assert np.all(dx.values["TRWUP"][i][off] == 0.0) and len(off) == 7


def test_reference_evaporation_is_pes_plus_per(ref: Reference) -> None:
    """The soil-water day's evaporation demand (the adapter's ``PES + PER``): the reference's
    actual evaporation never exceeds it and equals it (print precision) on the unlimited days."""
    pe = ref.t["physcl_exit"].values
    e = ref.ana[6][1:]
    demand = _f64(pe["PES"]) + _f64(pe["PER"])
    assert len(e) == len(demand)
    assert np.all(e <= demand + 1e-6)
    eq = np.abs(e - demand) <= 1e-6 + 1e-4
    print(
        "W0C "
        + json.dumps(
            {"days": len(e), "days_E_eq_PES_PER": int(eq.sum()), "max_deficit_cm": float(np.max(demand - e))}
        )
    )
    assert eq.sum() > 3000


def test_node_uptake_port_units_against_the_reference(ref: Reference) -> None:
    """P3 (``NodeUptake``, extensive cm d-1): :func:`publish_uptake_entry` on the reference's own
    layer uptake and water gives the reference's node ``qsr`` x 24 x node thickness (days without a
    layer at SW == LL exactly, where RZWQM2 keeps the old node value: gap G21)."""
    dd = ref.t["dssatdrv_exit"].values
    nl, nn = ref.nl, ref.nn
    sw, ll = dd["SW"][:, :nl], dd["SOILPROP%LL"][:, :nl]
    assert np.all(ll == ll[0])  # the crop layers' LL is fixed over the run (one params record)
    ok = ~np.any(sw == ll, axis=1)
    rwu, swd = _f64(dd["RWU"][ok, :nl]), _f64(sw[ok])
    state = {
        "crops": {f"{SLOT}_rootwu": {"rwu": jnp.asarray(rwu)}},
        "iface": {
            "crop_water": {SLOT: {"sw": jnp.asarray(swd)}},
            "root_uptake": {SLOT: {"uptake": jnp.zeros((len(rwu), nn))}},
        },
    }
    ll_ref = jnp.asarray(_f64(ll[0]))
    params = {"crop_iface": CropIfaceParams(nodes=ref.nodes, layers=ref.layers, ll=ll_ref)}
    got = np.asarray(publish_uptake_entry(SLOT)(state, params, None)["iface"]["root_uptake"][SLOT]["uptake"])
    want = _f64(dd["QSR"][ok, :nn]) * HOURS * ref.tl
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)
    # the intensive map keeps the thickness-weighted total: the node amounts add up to the active layers' uptake
    active = swd > _f64(ll[0])
    np.testing.assert_allclose(got.sum(axis=1), (rwu * active).sum(axis=1), rtol=1e-12, atol=1e-15)
    print("W0C " + json.dumps({"p3_days_compared": int(ok.sum()), "p3_days_sw_eq_ll": int((~ok).sum())}))
