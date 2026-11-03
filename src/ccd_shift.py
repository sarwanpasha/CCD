"""
Held-out subject-shift study on MMLU with point and lower-bound certification.

For each of the 15 model pairs and 400 random splits: certify the ordering seen on
one half of the 57 subjects (domain arm) or on a random half of the items with the
same size (i.i.d. arm), then check whether it reverses on the other half.
Certified by point estimate:   n_cert * rho_hat >= log(1/delta)
Certified by lower bound:      n_cert * rho_lo  >= log(1/delta), rho_lo the one-sided
                               95% variance-stabilized lower confidence bound.
"""
import argparse, itertools, json, os, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccd_fast as F

Z1 = 1.6448536269514722
LOGD = float(np.log(20.0))


def lower_bound(rho, s2, n):
    if not np.isfinite(rho) or rho <= 0:
        return 0.0 if rho <= 0 else float('inf')
    sc = np.sqrt(s2 / np.expm1(2 * rho))
    u = np.arctan(np.sqrt(np.expm1(2 * min(rho, 350.0)))) - Z1 * sc / np.sqrt(n)
    return 0.0 if u <= 0 else float(-np.log(np.cos(u)))


def one(gc, gt):
    """gc, gt: per-item loss differences L_i - L_j on certify / test halves."""
    if gc.mean() > 0:
        gc, gt = -gc, -gt                                 # orient: winner on certify half
    gm = -gc                                              # positive-mean form
    rho, beta = F.rho_general(gm[None, :], return_beta=True)
    rho = float(rho[0])
    s2 = float(F.sigma2_general(gm[None, :], beta)[0]) if np.isfinite(rho) else float('nan')
    n = gc.size
    lo = lower_bound(rho, s2, n)
    t = gc.mean() / gc.std()
    return dict(rho=rho, lo=lo, half_t2=0.5 * t * t, m=-gc.mean(), n=n,
                cert_pt=bool(np.isfinite(rho) and n * rho >= LOGD) or not np.isfinite(rho),
                cert_lo=bool(n * lo >= LOGD),
                rev=bool(gt.mean() >= 0))


def task(args):
    i, j, seed, L, cl, nsplit = args
    rng = np.random.default_rng(seed)
    g = L[i] - L[j]
    subj = np.unique(cl)
    out = []
    for _ in range(nsplit):
        pick = rng.permutation(subj)[: subj.size // 2]
        mask = np.isin(cl, pick)
        d = one(g[mask], g[~mask]); d.update(arm='domain', pair='%d:%d' % (i, j)); out.append(d)
        perm = rng.permutation(g.size); k = int(mask.sum())
        d = one(g[perm[:k]], g[perm[k:]]); d.update(arm='iid', pair='%d:%d' % (i, j)); out.append(d)
    return out


def auc(score, y):
    """AUC of a low radius as a predictor of reversal: P(score_rev < score_kept), ties 1/2."""
    from scipy.stats import rankdata
    score = np.asarray(score, float); y = np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if pos.size == 0 or neg.size == 0:
        return float('nan')
    rk = rankdata(np.concatenate([neg, pos]))
    return float((rk[:neg.size].sum() - neg.size * (neg.size + 1) / 2) / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('CCD_ROOT', '.'))
    ap.add_argument('--nsplit', type=int, default=400)
    ap.add_argument('--out', default='shift.json')
    a = ap.parse_args()
    Z = np.load(os.path.join(a.root, 'cache_lc.npz'))
    L, cl = Z['L_mmlu'], Z['cl_mmlu']
    jobs = [(i, j, 20261103 + 97 * k, L, cl, a.nsplit)
            for k, (i, j) in enumerate(itertools.combinations(range(L.shape[0]), 2))]
    ncpu = int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 1))
    with Pool(ncpu) as p:
        rows = [r for part in p.map(task, jobs) for r in part]
    summ = {}
    for arm in ('domain', 'iid'):
        R = [r for r in rows if r['arm'] == arm]
        y = [r['rev'] for r in R]
        def rate(sel):
            S = [r for r in R if sel(r)]
            return dict(count=len(S), reversed=int(sum(r['rev'] for r in S)),
                        rate=(sum(r['rev'] for r in S) / len(S)) if S else float('nan'))
        summ[arm] = dict(events=len(R),
                         cert_pt=rate(lambda r: r['cert_pt']), uncert_pt=rate(lambda r: not r['cert_pt']),
                         cert_lo=rate(lambda r: r['cert_lo']), uncert_lo=rate(lambda r: not r['cert_lo']),
                         auc_rho=auc([min(r['rho'], 1e9) for r in R], y),
                         auc_t2=auc([r['half_t2'] for r in R], y),
                         auc_m=auc([r['m'] for r in R], y))
        byp = {}
        for r in R:
            if r['cert_pt'] and r['rev']:
                byp[r['pair']] = byp.get(r['pair'], 0) + 1
        summ[arm]['violations_by_pair_pt'] = byp
    json.dump(dict(summary=summ, nsplit=a.nsplit), open(a.out, 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
