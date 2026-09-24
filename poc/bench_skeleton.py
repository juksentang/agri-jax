"""Throughput skeleton for Agri-JAX: NOT a real model, but the same computational shape as the PoC
(RZWQM Richards + S-W PET + CERES-Maize, CA-TPA 2015-2023 = 3287 days).
Per day: PET (~60 elementwise transcendental ops), crop step (~80 ops with jnp.where branches),
24 sub-steps x 3 Newton iterations of an implicit 37-node Richards step with a dense 37x37 solve.
Usage: python bench_skeleton.py --n 1000 10000 [--days 3287] [--sub 24] [--newton 3] [--grad 1] [--check 1]
Numerical guards (see poc/README.md, "NaN gradients"): the Newton update is clamped to [H_MIN, H_MAX], the top
boundary flux is signed positive downward with supply-limited evaporation, and gravity drains downward.
"""
import argparse, time, jax, jax.numpy as jnp
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, nargs="+", default=[1000]); ap.add_argument("--days", type=int, default=3287)
ap.add_argument("--sub", type=int, default=24); ap.add_argument("--newton", type=int, default=3)
ap.add_argument("--x64", type=int, default=1); ap.add_argument("--grad", type=int, default=0)
ap.add_argument("--solver", default="tridiag", choices=["tridiag", "dense"])
ap.add_argument("--remat", type=int, default=1, help="jax.checkpoint each day_step (needed for reverse-mode memory)")
ap.add_argument("--check", type=int, default=0, help="assert finite forward outputs / final state and (with --grad) finite gradients")
a = ap.parse_args()
jax.config.update("jax_enable_x64", bool(a.x64))
NN = 37; dz = jnp.full(NN, 150.0 / NN); DT = 1.0 / a.sub
H_MIN, H_MAX = -1.0e5, 10.0   # air-dry (pF 5) and shallow ponding, cm: bounds for the Newton iterate

def _x(h, p):  # safe argument for the power law: both jnp.where branches must stay finite under grad
    return jnp.where(h < -p["hb"], -h / p["hb"], 1.0)
def bc_theta(h, p):  # Brooks-Corey
    se = _x(h, p) ** (-p["lam"])
    return p["tr"] + (p["ts"] - p["tr"]) * se
def bc_k(h, p):
    se = _x(h, p) ** (-p["lam"])
    return p["ks"] * se ** (3.0 + 2.0 / p["lam"])
def bc_c(h, p):  # dtheta/dh
    c = (p["ts"] - p["tr"]) * p["lam"] / p["hb"] * _x(h, p) ** (-p["lam"] - 1)
    return jnp.where(h < -p["hb"], c, 1e-6)

def richards_substep(h, p, top_flux, sink):
    def newton(h_new, _):
        k = bc_k(h_new, p); kh = 0.5 * (k[1:] + k[:-1])
        q = -kh * ((h_new[1:] - h_new[:-1]) / dz[:-1] - 1.0)           # Darcy flux between nodes, + = downward
        q = jnp.concatenate([jnp.array([top_flux]), q, jnp.array([k[-1]])])  # top flux BC, free drainage bottom
        res = (bc_theta(h_new, p) - bc_theta(h, p)) / DT + (q[1:] - q[:-1]) / dz + sink
        c = bc_c(h_new, p)
        lo = -kh / (dz[:-1] * dz[1:]); up = lo
        di = c / DT + jnp.concatenate([jnp.array([0.0]), -lo]) + jnp.concatenate([-up, jnp.array([0.0])])
        if a.solver == "dense":
            J = jnp.diag(di) + jnp.diag(lo, -1) + jnp.diag(up, 1)
            dh = jnp.linalg.solve(J, res)
        else:  # tridiagonal (cuSPARSE gtsv on GPU, batched under vmap) -- what lineax/optimistix would do
            dl = jnp.concatenate([jnp.array([0.0]), lo]); du = jnp.concatenate([up, jnp.array([0.0])])
            dh = jax.lax.linalg.tridiagonal_solve(dl, di, du, res[:, None])[:, 0]
        return jnp.clip(h_new - dh, H_MIN, H_MAX), None   # bounded Newton step: c(h)->0 at both ends of the curve
    h_new, _ = jax.lax.scan(newton, h, None, length=a.newton)
    return h_new

def day_step(state, f):
    h, lai, stage, biom, root = state
    p = f["p"]
    # --- PET (Shuttleworth-Wallace-shaped arithmetic) ---
    es = 0.6108 * jnp.exp(17.27 * f["t"] / (f["t"] + 237.3)); ea = es * f["rh"]
    delta = 4098 * es / (f["t"] + 237.3) ** 2; rn = (1 - p["alb"]) * f["rad"] - 2.0
    raa = 4.7 / jnp.log(1 + f["u"]); rac = 30.0 / jnp.maximum(lai, 0.1); rsc = p["rs"] / jnp.maximum(lai, 0.1)
    rss = p["rss"] * jnp.exp(-2.0 * jnp.mean(bc_theta(h[:4], p)))
    pmc = (delta * rn + 1.2 * (es - ea) / raa) / (delta + 0.066 * (1 + rsc / rac))
    pms = (delta * rn * jnp.exp(-0.5 * lai) + 1.2 * (es - ea) / raa) / (delta + 0.066 * (1 + rss / raa))
    cc = 1 / (1 + (0.066 * rsc / rac) / (delta + 0.066)); cs = 1 / (1 + (0.066 * rss / raa) / (delta + 0.066))
    pet = jnp.maximum(cc * pmc + cs * pms, 0.0) / 245.0                    # -> cm/d
    ft = jnp.clip(1 - jnp.exp(-0.6 * lai), 0.0, 1.0); pt, pe = pet * ft, pet * (1 - ft)
    # --- crop (CERES-shaped) ---
    tt = jnp.clip(f["t"] - 8.0, 0.0, 26.0) * f["sow"]
    stage_n = stage + tt / p["p1"]
    theta = bc_theta(h, p); avail = jnp.clip((theta - p["wp"]) / (p["fc"] - p["wp"]), 0, 1)
    rdf = jnp.exp(-3.0 * jnp.cumsum(dz) / jnp.maximum(root, 1.0)) * (jnp.cumsum(dz) < root)
    swfac = jnp.clip(jnp.sum(rdf * avail) / jnp.maximum(jnp.sum(rdf), 1e-6) * 2, 0, 1)
    par = 0.5 * f["rad"] * (1 - jnp.exp(-0.65 * lai)); rue = 4.2 * jnp.where(stage_n < 1.0, 1.0, 0.9)
    dbiom = rue * par * jnp.minimum(swfac, 1.0) * f["sow"]
    dlai = jnp.where(stage_n < 1.0, 0.0012 * tt * swfac, jnp.where(stage_n < 2.5, 0.0, -0.02 * lai)) * f["sow"]
    lai_n = jnp.clip(lai + dlai, 0.0, 8.0); biom_n = biom + dbiom
    root_n = jnp.minimum(root + 2.5 * f["sow"] * swfac, 200.0)
    # --- soil water: 24 sub-steps ---
    sink = pt * rdf / jnp.maximum(jnp.sum(rdf * dz), 1e-6) * swfac
    top = f["rain"] - pe * jnp.clip((bc_theta(h[0], p) - p["tr"]) / (p["fc"] - p["tr"]), 0.0, 1.0)  # + = into the soil
    h_n = jax.lax.fori_loop(0, a.sub, lambda i, hh: richards_substep(hh, p, top, sink), h)
    aet = pt * swfac + pe; sw = jnp.sum(bc_theta(h_n, p) * dz)
    lai_n, stage_n, biom_n, root_n = [jnp.where(f["harv"] > 0, r, v) for r, v in ((0.0, lai_n), (0.0, stage_n), (0.0, biom_n), (5.0, root_n))]
    return (h_n, lai_n, stage_n, biom_n, root_n), jnp.stack([aet, lai_n, sw, biom_n]).astype(jnp.float32)

def run(params, forcing):
    p0 = {**params, "p": params}
    state0 = (jnp.full(NN, -200.0), 0.0, 0.0, 0.0, 5.0)
    fin = {k: forcing[k] for k in ("t", "rh", "rad", "u", "rain", "sow", "harv")}
    step = lambda s, f: day_step(s, {**f, "p": params})
    if a.remat: step = jax.checkpoint(step)   # store only per-day state; recompute sub-steps in the backward pass
    state, out = jax.lax.scan(step, state0, fin)
    return out, state[0]  # [days, 4], final pressure head [NN]

key = jax.random.PRNGKey(0); T = a.days; d = jnp.arange(T)
doy = d % 365
forcing = {"t": 10 + 15 * jnp.sin(2 * jnp.pi * (doy - 110) / 365) + 3 * jax.random.normal(key, (T,)),
           "rh": jnp.full(T, 0.7), "rad": 12 + 10 * jnp.sin(2 * jnp.pi * (doy - 100) / 365), "u": jnp.full(T, 2.0),
           "rain": jnp.where(jax.random.uniform(key, (T,)) < 0.3, 0.8, 0.0),
           "sow": ((doy > 130) & (doy < 270)).astype(float), "harv": ((doy == 270)).astype(float)}
def sample_params(n, k):
    u = jax.random.uniform(k, (n, 10))
    return {"lam": 0.14 + 0.5 * u[:, 0], "hb": 10 + 20 * u[:, 1], "ks": 1.5 + 2 * u[:, 2], "tr": 0.02 + 0.08 * u[:, 3],
            "ts": jnp.full(n, 0.453), "fc": 0.2 + 0.1 * u[:, 4], "wp": 0.09 + 0.06 * u[:, 5], "alb": 0.1 + 0.3 * u[:, 6],
            "rs": 100 + 300 * u[:, 7], "rss": 37 + 460 * u[:, 8], "p1": 200 + 100 * u[:, 9]}

run_batch = jax.jit(jax.vmap(run, in_axes=(0, None)))
print(f"jax {jax.__version__} devices={jax.devices()} x64={bool(a.x64)} days={T} sub={a.sub} newton={a.newton} solver={a.solver}", flush=True)
for n in a.n:
    P = sample_params(n, jax.random.PRNGKey(n))
    t0 = time.time(); out, hT = run_batch(P, forcing); out.block_until_ready(); tc = time.time() - t0
    t0 = time.time(); out, hT = run_batch(P, forcing); out.block_until_ready(); tr = time.time() - t0
    print(f"n={n:>7d}  compile+run={tc:7.2f}s  run={tr:7.2f}s  -> {n/tr:9.1f} sims/s  ({tr/n*1e3:.3f} ms/sim)  "
          f"nan={bool(jnp.isnan(out).any())}  mean_aet={float(out[...,0].mean()):.4f} mean_lai={float(out[...,1].mean()):.3f}  "
          f"h_final=[{float(hT.min()):.4g},{float(hT.max()):.4g}]", flush=True)
    if a.check:
        assert bool(jnp.isfinite(out).all()), f"non-finite forward output at n={n}"
        assert bool((hT >= H_MIN).all() & (hT <= H_MAX).all()), f"final h outside [{H_MIN},{H_MAX}] at n={n}"
if a.grad:
    def loss(P): return jnp.mean(run_batch(P, forcing)[0][..., 0])
    g = jax.jit(jax.grad(loss)); P = sample_params(a.n[0], jax.random.PRNGKey(1))
    t0 = time.time(); gv = g(P); jax.block_until_ready(gv); tc = time.time() - t0
    t0 = time.time(); gv = g(P); jax.block_until_ready(gv); tr = time.time() - t0
    n_bad = {k: int((~jnp.isfinite(v)).sum()) for k, v in gv.items() if not bool(jnp.isfinite(v).all())}
    gmax = max(float(jnp.abs(v).max()) for v in gv.values())
    print(f"grad n={a.n[0]}: compile+run={tc:.2f}s run={tr:.2f}s  finite={not n_bad}  |g|max={gmax:.3g}"
          + (f"  non-finite counts={n_bad}" if n_bad else ""), flush=True)
    if a.check:
        assert not n_bad, f"non-finite gradient: {n_bad}"
if a.check: print("check: OK (finite forward, h within bounds" + (", finite gradient)" if a.grad else ")"), flush=True)
