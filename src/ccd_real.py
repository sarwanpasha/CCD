"""
CCD_real: inference for radii on the real evaluation records.

  --part bench : every pair x {raw, acc} in the v2 (MMLU-Pro, BBH, MuSR, GPQA) and
                 legacy (ARC, HellaSwag, WinoGrande, MMLU) caches.
  --part arena : the 45 LMArena top-10 pairs, linearised BT witness, with a JOINT
                 bootstrap that refits BT on every resample (two-step uncertainty)
                 and a frozen-witness bootstrap for comparison.
Witness convention: g > 0 favours the published winner; reversal is E_Q[g] <= 0.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccd_fast as F

ROOT = os.environ.get('CCD_ROOT', '.')
LOG20 = float(np.log(20.0))
Z2, Z1 = 1.959963984540054, 1.6448536269514722


def u_of(r):
    return float(np.arctan(np.sqrt(np.expm1(2.0 * min(r, 350.0)))))


def r_of(u):
    if u <= 0: return 0.0
    if u >= np.pi / 2: return float('inf')
    return float(-np.log(np.cos(u)))


def vst_ci(rho, sigma2, n, z):
    """delta-method interval on u = arctan sqrt(e^{2rho}-1); exact-pivot for ternary."""
    if not np.isfinite(rho):
        return (float('nan'), float('inf'))
    sc = np.sqrt(sigma2 / np.expm1(2 * rho)) if rho > 0 else 1.0
    u = u_of(rho); h = z * sc / np.sqrt(n)
    return (r_of(u - h), r_of(u + h))


# ------------------------------------------------------------------ benchmarks
def load_bench():
    fams = []
    Z = np.load(os.path.join(ROOT, 'cache_v2.npz'), allow_pickle=True)
    for f in ('mmlu_pro', 'bbh', 'musr', 'gpqa'):
        fams.append(dict(fam=f, src='v2', L=Z['L_' + f], A=Z['A_' + f],
                         models=[str(x) for x in Z['models_' + f]], cl=None))
    Z = np.load(os.path.join(ROOT, 'cache_lc.npz'))
    for f in ('arc', 'hellaswag', 'winogrande', 'mmlu'):
        cl = Z['cl_' + f]
        fams.append(dict(fam=f, src='legacy', L=Z['L_' + f], A=Z['C_' + f],
                         models=['m%d' % i for i in range(Z['L_' + f].shape[0])],
                         cl=cl if len(np.unique(cl)) > 1 else None))
    return fams


_FAMS = None


def _init_bench():
    global _FAMS
    _FAMS = load_bench()


def bench_task(a):
    fi, i, j, reg, B, seed = a
    d = _FAMS[fi]; rng = np.random.default_rng(seed)
    M = d['L'] if reg == 'raw' else -d['A']
    w_, l_ = (i, j) if M[i].mean() < M[j].mean() else (j, i)
    g = M[l_] - M[w_]                                   # positive mean for winner
    n = g.size
    out = dict(fam=d['fam'], src=d['src'], reg=reg, winner=d['models'][w_],
               loser=d['models'][l_], n=int(n), mean_g=float(g.mean()))
    if reg == 'acc':
        nm, n0, npv = int((g < 0).sum()), int((g == 0).sum()), int((g > 0).sum())
        rho = float(F.rho_ternary(nm, n0, npv)[0])
        out.update(n_minus=nm, n_zero=n0, n_plus=npv, boundary=bool(nm == 0))
        s2 = float(np.expm1(2 * rho)) if np.isfinite(rho) else float('nan')
        out['sigma2_plugin'] = s2
        if nm > 0 and np.isfinite(rho):
            beta = 0.5 * np.log(npv / nm)
            out['sigma2_plugin_weights'] = float(F.sigma2_general(g[None, :], np.array([beta]))[0])
        rb = F.boot_ternary(nm, n0, npv, B, rng) if np.isfinite(rho) else None
    else:
        rr, bb = F.rho_general(g[None, :], return_beta=True)
        rho = float(rr[0])
        s2 = float(F.sigma2_general(g[None, :], bb)[0]) if np.isfinite(rho) else float('nan')
        out['sigma2_plugin'] = s2; out['beta'] = float(bb[0])
        rb = None
        if np.isfinite(rho):
            rb = np.empty(B); blk = max(1, int(1.5e7 // n))
            for s0 in range(0, B, blk):
                m = min(blk, B - s0)
                rb[s0:s0 + m] = F.rho_general(g[rng.integers(0, n, size=(m, n))])
    out['rho'] = rho
    out['status'] = 'inf' if not np.isfinite(rho) else 'finite'
    if np.isfinite(rho):
        out['se'] = float(np.sqrt(max(s2, 0) / n))
        out['ess'] = float(n / (1 + s2))
        out['rel_se'] = out['se'] / rho if rho > 0 else float('inf')
        out['ci95_vst'] = vst_ci(rho, s2, n, Z2)
        out['lo95_1s_vst'] = vst_ci(rho, s2, n, Z1)[0]
        out['ci95_normal'] = (max(0.0, rho - Z2 * out['se']), rho + Z2 * out['se'])
        fin = rb[np.isfinite(rb)]
        out['boot_inf_frac'] = float(1 - fin.size / B)
        out['ci95_perc'] = tuple(float(x) for x in np.quantile(rb, [0.025, 0.975])) if fin.size else None
        out['lo95_1s_perc'] = float(np.quantile(rb, 0.05))
        out['boot_sd'] = float(fin.std(ddof=1)) if fin.size > 2 else float('nan')
        if d['cl'] is not None:                         # cluster (subject) bootstrap
            cl = d['cl']; ids = np.unique(cl); idx_by = [np.flatnonzero(cl == c) for c in ids]
            rc = np.empty(min(B, 2000))
            for b in range(rc.size):
                pick = rng.integers(0, ids.size, ids.size)
                gi = g[np.concatenate([idx_by[k] for k in pick])]
                rc[b] = (F.rho_ternary((gi < 0).sum(), (gi == 0).sum(), (gi > 0).sum())[0]
                         if reg == 'acc' else F.rho_general(gi[None, :])[0])
            fc = rc[np.isfinite(rc)]
            out['cluster_boot_sd'] = float(fc.std(ddof=1))
            out['ci95_cluster'] = tuple(float(x) for x in np.quantile(rc, [0.025, 0.975]))
            out['lo95_1s_cluster'] = float(np.quantile(rc, 0.05))
        nreq = LOG20 / rho if rho > 0 else float('inf')
        out['n_req_05'] = nreq
        out['cert_point'] = bool(nreq <= n)
        lo = out['lo95_1s_vst']
        out['cert_lo_vst'] = bool(lo > 0 and LOG20 / lo <= n)
        out['cert_lo_perc'] = bool(out['lo95_1s_perc'] > 0 and LOG20 / out['lo95_1s_perc'] <= n)
        out['n_req_est_10pct'] = float(2 * Z2 ** 2 / (0.1 ** 2 * rho)) if rho > 0 else float('inf')
    else:
        out.update(cert_point=True, cert_lo_vst=True, cert_lo_perc=True)
    return out


def run_bench(a):
    fams = load_bench()
    tasks = []
    for fi, d in enumerate(fams):
        m = d['L'].shape[0]
        for i, j in itertools.combinations(range(m), 2):
            for reg in ('raw', 'acc'):
                tasks.append((fi, i, j, reg, a.B, a.seed + len(tasks)))
    print('bench tasks', len(tasks), flush=True)
    ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    res = []
    with Pool(ncpu, initializer=_init_bench) as p:
        for k, r in enumerate(p.imap_unordered(bench_task, tasks, chunksize=2)):
            res.append(r)
            if k % 25 == 0: print('  %d/%d' % (k, len(tasks)), flush=True)
    f = os.path.join(a.out, 'real_bench.json')
    json.dump(res, open(f, 'w'), indent=1); print('wrote', f, flush=True)


# ------------------------------------------------------------------- arena
_AR = {}


def _init_arena(ia, ib, y, pairs, m):
    _AR.update(ia=ia, ib=ib, y=y, pairs=pairs, m=m)


def _arena_g(th, r, Hp, ia, ib, A, B):
    u = Hp[:, A] - Hp[:, B]
    return (th[A] - th[B]) + r * (u[ia] - u[ib])


def arena_rep(seed):
    from btlib import fit_bt, bt_pieces
    ia, ib, y, pairs, m = _AR['ia'], _AR['ib'], _AR['y'], _AR['pairs'], _AR['m']
    n = ia.size; rng = np.random.default_rng(seed)
    w = rng.multinomial(n, np.full(n, 1.0 / n)) / n
    th, _ = fit_bt(ia, ib, y, w=w, m=m)
    r, v, H = bt_pieces(ia, ib, y, th, w)
    Hp = np.linalg.pinv(H, rcond=1e-10)
    joint = []; frozen = []
    for (A, B, g0) in pairs:
        g = _arena_g(th, r, Hp, ia, ib, A, B)
        joint.append(F.rho_weighted(g, w)[0])
        frozen.append(F.rho_weighted(g0, w)[0])
    return joint, frozen


def run_arena(a):
    sys.path.insert(0, os.path.join(ROOT, '.'))
    os.environ.setdefault('MIN_BATTLES', '800')
    from arena_ccb import load
    from btlib import fit_bt, bt_pieces
    models, ia, ib, y = load()
    n = ia.size; m = len(models)
    th, _ = fit_bt(ia, ib, y, m=m)
    order = np.argsort(-th); top = [int(x) for x in order[:10]]
    r, v, H = bt_pieces(ia, ib, y, th)
    Hp = np.linalg.pinv(H, rcond=1e-10)
    pairs = []; rows = []
    for i, A in enumerate(top):
        for B in top[i + 1:]:
            g = _arena_g(th, r, Hp, ia, ib, A, B)
            rho, beta = F.rho_weighted(g, np.full(n, 1.0 / n))
            s2 = F.sigma2_weighted(g, np.full(n, 1.0 / n), beta) if np.isfinite(beta) else float('nan')
            pairs.append((A, B, g))
            rows.append(dict(A=models[A], B=models[B], delta=float(th[A] - th[B]),
                             rho=rho, beta=beta, sigma2_plugin=s2, n=int(n),
                             se_plugin=float(np.sqrt(s2 / n)),
                             ci95_vst_plugin=vst_ci(rho, s2, n, Z2),
                             n_req_05=LOG20 / rho if rho > 0 else float('inf')))
    print('arena n=%d models=%d pairs=%d' % (n, m, len(pairs)), flush=True)
    ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    J = np.empty((a.B, len(pairs))); Fz = np.empty((a.B, len(pairs)))
    t0 = time.time()
    with Pool(ncpu, initializer=_init_arena, initargs=(ia, ib, y, pairs, m)) as p:
        for k, (jj, ff) in enumerate(p.imap(arena_rep, [a.seed + s for s in range(a.B)], chunksize=1)):
            J[k] = jj; Fz[k] = ff
            if k % 50 == 0: print('  rep %d/%d  %.0fs' % (k, a.B, time.time() - t0), flush=True)
    for c, row in enumerate(rows):
        jf = J[:, c]; ff = Fz[:, c]
        row.update(boot_sd_joint=float(jf.std(ddof=1)), boot_sd_frozen=float(ff.std(ddof=1)),
                   ci95_joint=tuple(float(x) for x in np.quantile(jf, [0.025, 0.975])),
                   lo95_1s_joint=float(np.quantile(jf, 0.05)),
                   ci95_frozen=tuple(float(x) for x in np.quantile(ff, [0.025, 0.975])),
                   flip_frac_joint=float((jf == 0).mean()),
                   var_ratio_joint_over_plugin=float(jf.var(ddof=1) * n / row['sigma2_plugin']),
                   var_ratio_frozen_over_plugin=float(ff.var(ddof=1) * n / row['sigma2_plugin']))
        lo = row['lo95_1s_joint']
        row['cert_point'] = bool(row['n_req_05'] <= n)
        row['cert_lo_joint'] = bool(lo > 0 and LOG20 / lo <= n)
    f = os.path.join(a.out, 'real_arena.json')
    json.dump(dict(n=int(n), B=a.B, pairs=rows, J=J.tolist()), open(f, 'w'))
    print('wrote', f, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['bench', 'arena'], required=True)
    ap.add_argument('--B', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20261025)
    ap.add_argument('--out', default='out_real')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    (run_bench if a.part == 'bench' else run_arena)(a)
