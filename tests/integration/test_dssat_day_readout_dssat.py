"""The season readout of :class:`agrijax.calib.dssat_day.DaySimulator` on the free-run acceptance runs
(the 58 maize treatments of M2 and the nitrogen-off CA-TPA seasons; the default tier a handful, the slow
tier all the runs the native inputs support).

* **Unchanged where the crop matures on a real day.** The published cultivar and perturbed cultivars
  (later and earlier, a longer grain fill, another grain set, a slower leaf appearance) on the native
  inputs with :data:`EXTEND` days of real weather after the season: every sample whose crop reaches
  ISTAGE 10 on a real day has the same entries, bit for bit, under ``readout="season"`` and
  ``readout="padded"``, and its end of the crop (``CROP_END``) is that day.
* **The other samples against an independent host readout.** A sample that fails, is harvested before
  maturity or has not matured by its last real day: the end status and every date equal the host
  readout of the same season simulated alone (:meth:`FreeRunInputs.simulate` with its ``mdate``
  output and :func:`~agrijax.calib.dssat_day.entry_values`), the end values agree to rounding.
* **The end of the crop is dscsm048's.** On a weather-year x sowing-date scenario of IUAF9901 (sowing
  20 days later), harvested on the reported date and, in a second variant, at maturity (the FileX
  ``HARVS`` changed), two changed cultivars: one matures before the harvest, one is killed by cold
  before maturity (and matures the next day, which sets ``MDATE`` again: read under the reported
  harvest, never reached under harvest at maturity, where dscsm048 ends the season on the failure
  day). The batch table's ``end``, ``ADAT`` and ``MDAT`` equal ``dscsm048``'s ``Summary.OUT`` on the
  same scenario and cultivars, ``HWAM`` and ``CWAM`` agree within the acceptance's 2 % (or 1 kg ha-1).
* **A reported harvest reads the harvest day.** On the seasons harvested on a reported date
  (``HARVS = R``: the CA-TPA seasons, some harvested before maturity) the season readout's yield at
  the published cultivar is the harvest-day grain weight of the acceptance's own program
  (``gwad[n - 1]``, what the acceptance compares with ``dscsm048``'s ``HWAM``).
"""

from __future__ import annotations

import os
from pathlib import Path

import day_dssat486_free_harness as h
import jax
import numpy as np
import pytest
import test_dssat_native_inputs as ni

import agrijax as aj
from agrijax.calib.dssat_day import (
    CROP_END,
    CROP_OUT_NAMES,
    CUL_ORDER,
    E_DATE,
    E_DAY,
    E_FINAL,
    E_STATUS,
    END_MATURED,
    END_NAMES,
    OUT_NAMES,
    DaySimulator,
    Entry,
    entry_values,
)
from agrijax.io.dssat import read_summary
from agrijax.port.run_fortran import DSSAT_ENGINE, dscsm_paths
from agrijax.sites.dssat_free_run import (
    NativeInputError,
    cultivar_batch_filex,
    free_run_inputs,
    missing_tables,
)

pytestmark = [
    pytest.mark.allow_skip(
        reason="needs dscsm048 v4.8.6.0 (AGRI_JAX_DSSAT), the DSSAT example data, the free-run tables and the "
        "CA-TPA DSSAT case"
    ),
    pytest.mark.skipif(not jax.config.jax_enable_x64, reason="the reference comparison runs in float64"),
]

#: real-weather days after the season (the facade's and the calibration's native inputs)
EXTEND = 60
#: the cultivars: multipliers of the published P1, P2, P5, G2, G3, PHINT (the first is the published one)
SCALES = np.asarray(
    [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.3, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.7, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.4, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.6, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.2, 0.8, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.25],
        [1.5, 1.0, 1.6, 1.0, 1.0, 1.2],
        [1.8, 1.0, 1.9, 1.0, 1.0, 1.3],
    ]
)
#: the end values of the host readout (one season alone) against the batched program: rounding only
HOST_RTOL = 1e-9
#: the common entries (both readouts) and the season readout's own
ENTRIES = (
    [Entry(E_DATE, c) for c in (1, 4, 10)]
    + [Entry(E_FINAL, i) for i in range(len(CROP_OUT_NAMES))]
    + [Entry(E_DAY, OUT_NAMES.index("lai"), 30)]
)
SEASON_ONLY = [Entry(E_DATE, CROP_END), Entry(E_STATUS, 0)]
N_COMMON = len(ENTRIES)
I_DATES = [0, 1, 2, N_COMMON]


def _skip_without_data(data_dir: Path) -> None:
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file() or not ni.MAIZE.is_dir():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    for sub in (h.DSW, h.DET, h.A12, h.CATPA_CASE):
        if not (data_dir / sub).is_dir():
            pytest.skip(f"{data_dir / sub} not found")


@pytest.fixture(scope="module", params=["small", pytest.param("all", marks=pytest.mark.slow)])
def runs(request: pytest.FixtureRequest, data_dir: Path, tmp_path_factory: pytest.TempPathFactory):
    _skip_without_data(data_dir)
    keys = ni.SMALL if request.param == "small" else ni._keys(data_dir)
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    root = Path(os.environ.get("AGRI_JAX_RUN_ROOT") or tmp_path_factory.mktemp("rd"))
    outs = h.run_references(keys, root / f"rd_{request.param}", data_dir, jobs)
    native, tables = [], []
    for exp, t in keys:
        key = h.key_of(exp, t)
        pth = ni._paths(exp, data_dir)
        kw = {k: v for k, v in pth.items() if k != "filex"}
        try:
            native.append(
                free_run_inputs(pth["filex"], t, outs[key], source="native", extend_days=EXTEND, **kw)
            )
        except NativeInputError:
            pass
        if not missing_tables(exp, t, data_dir):
            tables.append((exp, t, free_run_inputs(pth["filex"], t, outs[key], data_dir, **kw)))
    return {"param": request.param, "native": native, "tables": tables, "outs": outs, "data": data_dir}


def _theta(xs) -> tuple[np.ndarray, np.ndarray]:
    pub = np.stack([np.asarray([x.published()[n] for n in CUL_ORDER]) for x in xs])
    theta = (pub[:, None, :] * SCALES[None]).reshape(-1, len(CUL_ORDER))
    return theta, np.repeat(np.arange(len(xs)), SCALES.shape[0])


def test_matured_samples_are_unchanged_and_the_others_match_the_host_readout(runs) -> None:
    xs = runs["native"]
    assert xs, "no run with native inputs"
    season = DaySimulator(xs, [ENTRIES + SEASON_ONLY] * len(xs))
    padded = DaySimulator(xs, [ENTRIES] * len(xs), readout="padded")
    theta, tid = _theta(xs)
    ys, yp = season(theta, tid), padded(theta, tid)
    status = ys[:, -1].astype(int)
    counts = {END_NAMES[k]: int(np.sum(status == k)) for k in range(len(END_NAMES))}
    print(f"{len(xs)} runs x {SCALES.shape[0]} cultivars: {counts}")
    mat = status == END_MATURED
    pub_m = [k * SCALES.shape[0] for k, x in enumerate(xs) if x.harvest == "M"]
    assert mat[pub_m].all(), "the published cultivar matures in every season harvested at maturity"
    np.testing.assert_array_equal(ys[mat, :N_COMMON], yp[mat])
    np.testing.assert_array_equal(ys[mat, N_COMMON], ys[mat, 2])  # CROP_END is the first ISTAGE 10
    if runs["param"] == "all":
        assert (~mat).any(), "the perturbed cultivars should leave some seasons not matured on real days"
    for s in np.nonzero(~mat)[0]:
        i = int(tid[s])
        x = xs[i]
        g = season.where[i][0]
        cul = dict(zip(CUL_ORDER, theta[s].tolist(), strict=True))
        o = x.simulate(cul, pad_days=season.n_days[g] - x.n_days)
        host = entry_values(o, ENTRIES + SEASON_ONLY, last_day=season.last_days[i], harvest=x.harvest)
        assert int(host[-1]) == status[s], (x.key, cul, host[-1], status[s])
        np.testing.assert_array_equal(ys[s, I_DATES], host[I_DATES], err_msg=x.key)
        np.testing.assert_allclose(ys[s, :-1], host[:-1], rtol=HOST_RTOL, atol=1e-9, err_msg=x.key)
        assert np.all(np.isfinite(ys[s]))


def test_a_reported_harvest_reads_the_harvest_day_of_the_acceptance_program(runs) -> None:
    rr = [(e, t, x) for e, t, x in runs["tables"] if x.harvest == "R"]
    if runs["param"] == "all":
        assert rr, "the CA-TPA seasons are harvested on a reported date"
    if not rr:
        pytest.skip("no season harvested on a reported date in this tier")
    xs = [x for _, _, x in rr]
    sim = DaySimulator(xs, [[Entry(E_FINAL, OUT_NAMES.index("gwad")), Entry(E_STATUS, 0)]] * len(xs))
    theta, _ = _theta(xs)
    y = sim(theta[:: SCALES.shape[0]], np.arange(len(xs)))
    acc = [h.build(e, t, runs["outs"][h.key_of(e, t)], runs["data"]) for e, t, _ in rr]
    worst = 0.0
    for grp in sorted({(r.nl, r.mesev) for r in acc}):
        rs = [r for r in acc if (r.nl, r.mesev) == grp]
        o = h.run_group(rs, "free")
        for k, r in enumerate(rs):
            i = next(j for j, a in enumerate(acc) if a.key == r.key)
            ref = float(np.asarray(o["gwad"])[k, r.n_days - 1, 0])
            assert sim.last_days[i] == r.n_days - 1 == xs[i].n_days - 1, r.key
            worst = max(worst, abs(float(y[i, 0]) - ref))
            assert float(y[i, 0]) == ref, (r.key, float(y[i, 0]), ref)
    print(
        f"{len(xs)} seasons harvested on a reported date: harvest-day yield equal, largest difference {worst}"
    )


#: the scenario and cultivars of the end-of-crop check: IUAF9901 treatment 1 sown 20 days later; the
#: published cultivar with these coefficients changed (a later cultivar killed by cold before maturity,
#: and one that matures before the reported harvest)
END_SCENARIO = ("IUAF9901", 1, 20)
END_CULTIVARS = (
    {"P1": 450.0, "P5": 900.0, "PHINT": 45.0},
    {"P1": 320.0, "P5": 900.0, "PHINT": 45.0},
)


def _harvest_at_maturity(text: str) -> str:
    """The FileX ``text`` with every simulation-control ``HARVS`` set to ``M`` (harvest at maturity)."""
    lines = text.splitlines(keepends=True)
    for i, ln in enumerate(lines[:-1]):
        if ln.startswith("@N MANAGEMENT") and "HARVS" in ln:
            col = ln.index("HARVS") + len("HARVS") - 1  # right-aligned under the header
            j = i + 1
            while j < len(lines) and lines[j][:1] == " " and len(lines[j]) > col:
                lines[j] = lines[j][:col] + "M" + lines[j][col + 1 :]
                j += 1
    return "".join(lines)


@pytest.mark.slow
@pytest.mark.parametrize("harvest", ["R", "M"])
def test_the_end_of_the_crop_is_dscsm048s_on_a_cold_failure_and_a_maturity(harvest: str) -> None:
    from agrijax.dssat import Scenarios
    from agrijax.io.dssat import read_filex
    from agrijax.sites.dssat_free_run import harvest_option

    name, trno, shift = END_SCENARIO
    if not dscsm_paths(DSSAT_ENGINE)[0].is_file() or not (ni.MAIZE / f"{name}.MZX").is_file():
        pytest.skip(f"dscsm048 / DSSAT example data not found under {DSSAT_ENGINE}")
    exp = aj.dssat.experiment(name, data_root=DSSAT_ENGINE)
    scen = exp.scenarios(trno, sowing_shift=[shift])
    if harvest == "M":
        fx = scen.filex[0]
        fx.write_text(_harvest_at_maturity(fx.read_text()))
        assert harvest_option(read_filex(fx), trno) == "M"
        run = exp._build(fx, trno)
        scen = Scenarios(scen.treatment, scen.table, [run], scen.published, [fx], _exp=exp)
    assert scen.runs[0].harvest == harvest
    pub = scen.published
    tag = f"readout_end_{harvest}"
    ids, cul, written = exp._cul_rows(trno, [{**pub, **c} for c in END_CULTIVARS], tag)
    samples = [{k: float(w[k]) for k in CUL_ORDER} for w in written]  # what DSSAT reads
    ours = scen.run({k: [s[k] for s in samples] for k in CUL_ORDER}).table
    assert list(ours["end"]) == ["failed", "matured"], list(ours["end"])
    out, _ = exp._run_ref(
        list(range(1, len(ids) + 1)),
        tag,
        filex=scen.filex[0],
        cul_file=cul,
        filex_edit=lambda t: cultivar_batch_filex(t, trno, ids),
    )
    ref = read_summary(out / "Summary.OUT").sort_values("TRNO").reset_index(drop=True)
    for j in range(len(samples)):
        a, d = ours.iloc[j], ref.iloc[j]
        for col in ("ADAT", "MDAT"):
            got = a[col] if np.isfinite(a[col]) else -99
            want = d[col] if d[col] > 0 else -99
            assert int(got) == int(want), (j, col, a[col], d[col])
        for col in ("HWAM", "CWAM"):
            assert abs(a[col] - d[col]) <= max(h.YIELD_REL * d[col], 1.0), (j, col, a[col], d[col])
    print(
        ours[["end", "ADAT", "MDAT", "HWAM", "CWAM"]].to_string(),
        ref[["ADAT", "MDAT", "HDAT", "HWAM", "CWAM"]].to_string(),
    )
