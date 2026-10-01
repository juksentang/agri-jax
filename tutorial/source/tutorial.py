# The Agri-JAX tutorial, one source for the English and the Chinese notebook.
# "# %% [markdown] <key>" is a text cell (tutorial/source/text.toml holds <key> in both languages);
# "# %% [messages]" is the code cell of the printed messages (text.toml's [print] table, both languages);
# "# %%" starts a code cell (the same code in both notebooks; "{LANG}" becomes the notebook's language);
# in a code cell "# @c:<key>" (a whole-line or a trailing comment) and a docstring written as @c:<key> are
# replaced by text.toml's [comment.<key>] of each language.
# Regenerate the notebooks with
#   python tutorial/build_notebooks.py

# %% [markdown] title

# %% [markdown] install

# %%
import sys

if "google.colab" in sys.modules:
    !pip install -q "agrijax[plot] @ git+https://github.com/juksentang/agri-jax"
LANG = "{LANG}"  # @c:lang

# %% [messages]

# %%
import os

import numpy as np

import agrijax as aj

SMALL = os.environ.get("AGRI_JAX_TUTORIAL_SMALL") == "1"  # @c:small
say("machine", machine=aj.machine())

# %% [markdown] experiment

# %%
exp = aj.dssat.experiment("UFGA8201")
print(exp)
exp.table()

# %% [markdown] run

# %%
season = exp.run(treatment=4)
s = season.summary
say("season", treatment=season.treatment, adat=int(s["ADAT"]), mdat=int(s["MDAT"]), hwam=s["HWAM"])
aj.plot.season(season, observed=exp.observed(4));

# %%
season.to_frame().tail()

# %% [markdown] compare

# %%
aj.dssat.install_reference()
ref = exp.reference(treatment=4)
season.compare_summary(ref)

# %% [markdown] season_daily

# %%
season.compare_daily(ref)

# %%
aj.plot.season(season, reference=ref, observed=exp.observed(4));

# %%
aj.plot.compare(season, ref);

# %% [markdown] scenarios

# %%
years = range(1981, 1984) if SMALL else range(1978, 1988)
shifts = [0] if SMALL else [-14, 0, 14]
scen = exp.scenarios(treatment=4, years=years, sowing_shift=shifts)
say("scenarios", n=len(scen.runs))
scen.table

# %% [markdown] samples

# %%
K = 8 if SMALL else 320
pub = scen.published


def samples(seed):
    """@c:samples_doc"""
    rng = np.random.default_rng(seed)
    c = {n: pub[n] * rng.uniform(0.9, 1.1, K) for n in ("P1", "P5", "G2", "G3", "PHINT")}
    for n in c:
        c[n][0] = pub[n]
    return c


first = scen.run(samples(0))
first

# %%
second = scen.run(samples(1))
second

# %%
aj.plot.batch(first, by="year");

# %% [markdown] check_scenarios

# %%
dss = scen.reference()
pub_runs = first.table[first.table["sample"] == 0]
t = pub_runs.merge(dss, on=["year", "sowing_shift"], suffixes=("", " DSSAT"))
rel = (t["HWAM"] - t["HWAM DSSAT"]).abs() / t["HWAM DSSAT"]
same_adat = int((t["ADAT"] == t["ADAT DSSAT"]).sum())
same_mdat = int((t["MDAT"] == t["MDAT DSSAT"]).sum())
say("check_yield", n=len(t), rel=rel.max())
say("check_dates", adat=same_adat, mdat=same_mdat, n=len(t))
say("dssat_per_scenario", sec=float(np.mean(scen.dssat_s)))

# %% [markdown] race

# %%
n_dssat = 5 if SMALL else 50
db = exp.dssat_batch(treatment=4, cultivar={n: v[:n_dssat] for n, v in samples(2).items()})
ours = scen.run(db.cultivar).table
ours = ours[(ours["year"] == 1982) & (ours["sowing_shift"] == 0)].reset_index(drop=True)
rel = (ours["HWAM"] - db.table["HWAM"]).abs() / db.table["HWAM"]
say("batch_yield", n=db.seasons, rel=rel.max())
say("batch_dates", same=int((ours["MDAT"] == db.table["MDAT"]).sum()), n=db.seasons)

# %%
n = second.timing["seasons"]
run_s = second.timing["run_s"]
per_dssat = db.elapsed_s / db.seasons
say("machine", machine=aj.machine())
say("dssat_speed", ms=1e3 * per_dssat, n=db.seasons, s=db.elapsed_s)
say("ours_speed", ms=1e3 * run_s / n, n=n, s=run_s)
say("compile", s=first.timing["compile_s"])
say("extrapolated", n=n, s=n * per_dssat)

# %% [markdown] calibrate

# %%
starts, budget = (2, 48) if SMALL else (8, 200)
res = aj.calibrate(exp, treatments=[4], holdout=[6], starts=starts, budget=budget, seed=0)
print(res)

# %%
aj.plot.calibration(res);

# %% [markdown] cul

# %%
if os.path.exists("MZCER048_calibrated.CUL"):
    os.remove("MZCER048_calibrated.CUL")
print(res.write_cul("MZCER048_calibrated.CUL"))
res.check_dssat()

# %% [markdown] next
