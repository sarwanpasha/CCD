"""
CCD simulation suite for the inference layer.  One experiment per invocation.

  python ccd_sim.py --exp {var,cover,boot,local,rank} --task K --ntask N --out DIR

Task K handles configs[K::N].  Replications inside a config are split over
SLURM_CPUS_PER_TASK worker processes, each running a vectorised block.
"""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import Pool
import ccd_fast as F

Z = 1.959963984540054          # qnorm(0.975)


# ---------------------------------------------------------------- populations
def ternary_at(rho, p0):
    """Probabilities (p-, p0, p+) on {-1,0,1} with exactly this rho*."""
    s, c = 1.0 - p0, np.exp(-rho) - p0
    if c <= 0 or s * s < c * c:
        raise ValueError("infeasible (rho=%.4g, p0=%.4g)" % (rho, p0))
    d = np.sqrt(s * s - c * c)
    return (s - d) / 2.0, p0, (s + d) / 2.0


def rho_exact_ternary(p):
    pm, p0, pp = p
    return -np.log(2.0 * np.sqrt(pp * pm) + p0)


def sigma2_exact_ternary(p):
    return np.expm1(2.0 * rho_exact_ternary(p))          # M(2 beta) == 1 here


def draw_ternary(p, n, reps, rng):
    """(reps, 3) multinomial counts -- the sufficient statistic."""
    return rng.multinomial(n, np.asarray(p, float), size=reps)


# ---------------------------------------------------------------- experiments
def cfg_var():
    out = []
    for rho in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        for p0 in (0.0, 0.3, 0.6):
            for n in (500, 2000, 10000):
                try:
                    ternary_at(rho, p0)
                except ValueError:
                    continue
                out.append(dict(kind="ternary", rho=rho, p0=p0, n=n, reps=40000))
    for rho in (0.005, 0.05, 0.2):
        for n in (500, 2000, 10000):
            out.append(dict(kind="gauss", rho=rho, n=n, reps=8000))
            out.append(dict(kind="graded", rho=rho, n=n, reps=8000))
    return out


def run_var(c, rng):
    n, reps = c["n"], c["reps"]
    if c["kind"] == "ternary":
        p = ternary_at(c["rho"], c["p0"])
        cts = draw_ternary(p, n, reps, rng)
        rho = F.rho_ternary(cts[:, 0], cts[:, 1], cts[:, 2])
        s2_pop = sigma2_exact_ternary(p)
        rho_pop = rho_exact_ternary(p)
    else:
        rho_pop, s2_pop, sampler = _cont_pop(c)
        blk = max(1, int(2.0e6 // n)); parts = []
        for s0 in range(0, reps, blk):
            parts.append(F.rho_general(sampler(min(blk, reps - s0), n, rng)))
        rho = np.concatenate(parts)
    fin = np.isfinite(rho)
    r = rho[fin]
    return dict(**c, rho_pop=float(rho_pop), sigma2_pop=float(s2_pop),
                n_inf=int((~fin).sum()),
                mean_rho=float(r.mean()), bias=float(r.mean() - rho_pop),
                var_scaled=float(n * r.var(ddof=1)),
                ratio=float(n * r.var(ddof=1) / s2_pop),
                mc_se_ratio=float(np.sqrt(2.0 / max(r.size - 1, 1))))


def _cont_pop(c):
    """Gaussian or 4-point graded witness calibrated to the target rho*.
    Returns (rho_pop, sigma2_pop, sampler(reps, n, rng))."""
    if c["kind"] == "gauss":
        mu = np.sqrt(2.0 * c["rho"])                      # V = 1 -> rho = mu^2/2
        return c["rho"], float(np.expm1(2.0 * c["rho"])), \
            (lambda r, n, rng: rng.normal(mu, 1.0, size=(r, n)))
    vals = np.array([-1.0, -0.5, 0.25, 1.0])
    base = np.array([0.15, 0.15, 0.30, 0.40])
    lo, hi = -3.0, 3.0
    for _ in range(100):
        m = 0.5 * (lo + hi)
        pr = base * np.exp(m * vals); pr /= pr.sum()
        if _pop_rho_sigma(vals, pr)[0] < c["rho"]: lo = m
        else: hi = m
    pr = base * np.exp(0.5 * (lo + hi) * vals); pr /= pr.sum()
    r, s2 = _pop_rho_sigma(vals, pr)
    return r, s2, (lambda rr, n, rng: rng.choice(vals, size=(rr, n), p=pr))


def _pop_rho_sigma(vals, pr):
    """Exact population rho* and sigma^2 via monotone bisection on E_Q[g]=0."""
    if float(pr @ vals) <= 0:
        return 0.0, 0.0
    def EQ(t):
        a = -t * vals; w = pr * np.exp(a - a.max()); return float(w @ vals) / float(w.sum())
    lo, hi = 0.0, 1.0
    while EQ(hi) > 0:
        lo, hi = hi, 2.0 * hi
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if EQ(m) > 0: lo = m
        else: hi = m
    t = 0.5 * (lo + hi)
    M = lambda u: float(pr @ np.exp(-u * vals))
    return float(-np.log(M(t))), float(M(2 * t) / M(t) ** 2 - 1.0)


def cfg_cover():
    return [dict(rho=rho, p0=p0, n=n, reps=20000, nboot=999)
            for rho in (0.002, 0.01, 0.05, 0.2) for p0 in (0.0, 0.5)
            for n in (250, 500, 1000, 2000, 5000, 10000)
            if _feasible(rho, p0)]


def _feasible(rho, p0):
    try:
        ternary_at(rho, p0); return True
    except ValueError:
        return False


def run_cover(c, rng):
    p = ternary_at(c["rho"], c["p0"]); rho0 = rho_exact_ternary(p)
    n, reps, nb = c["n"], c["reps"], c["nboot"]
    cts = draw_ternary(p, n, reps, rng)
    rho = F.rho_ternary(cts[:, 0], cts[:, 1], cts[:, 2])
    s2 = np.expm1(2.0 * np.minimum(rho, 50.0))            # exact ternary plug-in
    se = np.sqrt(np.maximum(s2, 0.0) / n)
    hit = {"normal": ((rho - Z * se <= rho0) & (rho0 <= rho + Z * se)),
           "vst": _vst_cov(rho, n, rho0)}
    lens = {"normal": 2 * Z * se}
    bp = np.zeros(reps, bool); bs = np.zeros(reps, bool)
    sub = min(reps, 4000)                                 # bootstrap on a subsample of reps
    for j in range(sub):
        rb = F.boot_ternary(cts[j, 0], cts[j, 1], cts[j, 2], nb, rng)
        rb = rb[np.isfinite(rb)]
        if rb.size < 50: continue
        lo, hi = np.quantile(rb, [0.025, 0.975]); bp[j] = lo <= rho0 <= hi
        u = _u(rho[j]); ub = _u(rb); t = (ub - u) * np.sqrt(n)
        q = np.quantile(t, [0.025, 0.975])
        bs[j] = _uinv(u - q[1] / np.sqrt(n)) <= rho0 <= _uinv(u - q[0] / np.sqrt(n))
    return dict(**c, rho_pop=float(rho0),
                cov_normal=float(hit["normal"].mean()), cov_vst=float(hit["vst"].mean()),
                cov_perc=float(bp[:sub].mean()), cov_stud=float(bs[:sub].mean()),
                len_normal=float(lens["normal"].mean()),
                n_inf=int((~np.isfinite(rho)).sum()), n_zero=int((rho == 0).sum()),
                reps_boot=int(sub), mc_se=float(np.sqrt(0.05 * 0.95 / reps)))


def _u(r):   return np.arctan(np.sqrt(np.expm1(2.0 * np.minimum(r, 50.0))))
def _uinv(u):
    u = np.clip(u, 0.0, np.pi / 2 - 1e-12)
    return -np.log(np.cos(u))


def _vst_cov(rho, n, rho0):
    u, u0 = _u(rho), _u(rho0)
    return (u - Z / np.sqrt(n) <= u0) & (u0 <= u + Z / np.sqrt(n))


def cfg_boot():
    return [dict(rho=rho, p0=0.3, n=n, nsamp=1500, nboot=2000, nmc=200000)
            for rho in (0.0, 0.0005, 0.002, 0.01, 0.05, 0.2)
            for n in (500, 2000, 10000) if _feasible(max(rho, 1e-9), 0.3)]


def run_boot(c, rng):
    rho = max(c["rho"], 0.0)
    p = ternary_at(rho, c["p0"]) if rho > 0 else _p_at_zero(c["p0"])
    rho0 = rho_exact_ternary(p); n = c["n"]
    mc = F.rho_ternary(*draw_ternary(p, n, c["nmc"], rng).T)
    mc = np.sort(mc[np.isfinite(mc)])
    ks = []
    cts = draw_ternary(p, n, c["nsamp"], rng)
    for j in range(c["nsamp"]):
        rb = F.boot_ternary(cts[j, 0], cts[j, 1], cts[j, 2], c["nboot"], rng)
        rb = rb[np.isfinite(rb)]
        if rb.size < 50: continue
        a = np.sqrt(n) * (rb - F.rho_ternary(cts[j, 0], cts[j, 1], cts[j, 2])[0])
        ks.append(_ks(np.sort(a), np.sqrt(n) * (mc - rho0)))
    ks = np.array(ks)
    return dict(**c, rho_pop=float(rho0), ks_mean=float(ks.mean()),
                ks_med=float(np.median(ks)), ks_q90=float(np.quantile(ks, 0.9)),
                nsamp_used=int(ks.size))


def _p_at_zero(p0):
    s = 1.0 - p0
    return (s / 2.0, p0, s / 2.0)


def _ks(a, b):
    """Two-sample KS between sorted arrays a and b."""
    allv = np.concatenate([a, b])
    fa = np.searchsorted(a, allv, "right") / a.size
    fb = np.searchsorted(b, allv, "right") / b.size
    return float(np.abs(fa - fb).max())


def cfg_local():
    return [dict(c=cc, n=n, p0=0.3, reps=200000)
            for cc in (0.5, 1.0, 2.0, 5.0) for n in (500, 2000, 10000, 50000)]


def run_local(c, rng):
    n, cc = c["n"], c["c"]
    p = ternary_at(cc / n, c["p0"]); rho0 = rho_exact_ternary(p)
    cts = draw_ternary(p, n, c["reps"], rng)
    r = F.rho_ternary(cts[:, 0], cts[:, 1], cts[:, 2])
    r = r[np.isfinite(r)]
    s = 2.0 * n * r                                        # predicted: (Z_+)^2, Z~N(sqrt(2c),1)
    zz = rng.normal(np.sqrt(2.0 * cc), 1.0, size=400000)
    pred = np.sort(np.maximum(zz, 0.0) ** 2)
    return dict(**c, rho_pop=float(rho0), ks=_ks(np.sort(s), pred),
                p0_mass=float((s == 0).mean()), p0_pred=float((pred == 0).mean()),
                mean=float(s.mean()), mean_pred=float(pred.mean()))


def cfg_rank():
    return [dict(k=k, tie=tie, n=n, reps=20000, nboot=499, nsub=300, rho=0.03)
            for k in (3, 6) for tie in (0, 1) for n in (1000, 5000)]


def _rank_draw(k, n, mu, L, reps, rng):
    """(reps, k, n) block of correlated Gaussian witnesses."""
    X = rng.standard_normal((reps, n, k)) @ L.T + mu
    return np.ascontiguousarray(X.transpose(0, 2, 1))


def run_rank(c, rng):
    """rho_min = min_j rho^(j) over k correlated Gaussian witnesses."""
    k, n, reps, nb, nsub = c["k"], c["n"], c["reps"], c["nboot"], c["nsub"]
    mu = np.full(k, np.sqrt(2.0 * c["rho"]))
    if not c["tie"]:
        mu = mu * np.linspace(1.0, 1.6, k)                 # unique argmin
    R = 0.4 * np.ones((k, k)) + 0.6 * np.eye(k)
    L = np.linalg.cholesky(R)
    rho0 = float(np.min(mu ** 2 / 2.0))
    zb = float(_qnorm(1.0 - 0.025 / k))

    hit_n = np.empty(reps, bool); hit_s = np.empty(reps, bool); est = np.empty(reps)
    blk = max(1, min(reps, int(2.0e7 // (k * n))))
    done = 0
    while done < reps:
        b = min(blk, reps - done)
        B = _rank_draw(k, n, mu, L, b, rng)
        r, be = F.rho_general(B.reshape(b * k, n), return_beta=True)
        s2 = F.sigma2_general(B.reshape(b * k, n), be)
        r = r.reshape(b, k); se = np.sqrt(np.maximum(s2, 0.0).reshape(b, k) / n)
        i = r.argmin(axis=1); ar = np.arange(b)
        rm, sm = r[ar, i], se[ar, i]
        est[done:done + b] = rm
        hit_n[done:done + b] = (rm - Z * sm <= rho0) & (rho0 <= rm + Z * sm)
        hit_s[done:done + b] = ((r - zb * se).min(1) <= rho0) & (rho0 <= (r + zb * se).min(1))
        done += b

    hit_b = np.empty(nsub, bool)
    for j in range(nsub):
        X = _rank_draw(k, n, mu, L, 1, rng)[0]             # (k, n)
        rj = float(F.rho_general(X).min())
        bb = np.empty(nb); step = max(1, int(2.0e7 // (k * n)))
        for s in range(0, nb, step):
            m = min(step, nb - s)
            idx = rng.integers(0, n, size=(m, n))
            bb[s:s + m] = F.rho_general(X[:, idx].transpose(1, 0, 2).reshape(m * k, n)).reshape(m, k).min(1)
        lo, hi = np.quantile(bb[np.isfinite(bb)], [0.025, 0.975])
        hit_b[j] = lo <= rho0 <= hi
    return dict(**c, rho_pop=rho0, cov_normal=float(hit_n.mean()),
                cov_boot=float(hit_b.mean()), cov_simul=float(hit_s.mean()),
                bias=float(est.mean() - rho0),
                mc_se=float(np.sqrt(0.05 * 0.95 / reps)),
                mc_se_boot=float(np.sqrt(0.05 * 0.95 / nsub)))


def _qnorm(q):
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(q))


EXP = {"vargr": (lambda: [c for c in cfg_var() if c["kind"] == "graded"], run_var), "var": (cfg_var, run_var), "cover": (cfg_cover, run_cover),
       "boot": (cfg_boot, run_boot), "local": (cfg_local, run_local),
       "rank": (cfg_rank, run_rank)}


def _work(a):
    cfg, fn, seed = a
    t0 = time.time()
    r = fn(cfg, np.random.default_rng(seed))
    r["secs"] = round(time.time() - t0, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXP))
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--ntask", type=int, default=1)
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=20261025)
    a = ap.parse_args()
    cfgf, runf = EXP[a.exp]
    cfgs = cfgf()[a.task::a.ntask]
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    os.makedirs(a.out, exist_ok=True)
    print("exp=%s task=%d/%d configs=%d cpus=%d" % (a.exp, a.task, a.ntask, len(cfgs), ncpu), flush=True)
    args = [(c, runf, a.seed + 1000 * a.task + i) for i, c in enumerate(cfgs)]
    with Pool(ncpu) as pool:
        res = []
        for r in pool.imap_unordered(_work, args):
            res.append(r); print("  done", {k: r[k] for k in list(r)[:4]}, r["secs"], "s", flush=True)
    f = os.path.join(a.out, "%s_%03d.json" % (a.exp, a.task))
    json.dump(res, open(f, "w"), indent=1)
    print("wrote", f, len(res), "rows", flush=True)


if __name__ == "__main__":
    main()
