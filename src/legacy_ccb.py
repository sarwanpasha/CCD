"""Recompute the retired-archive per-pair table from the pinned loss matrices."""
import json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccb import ccb, kway_rho
from scipy.stats import spearmanr

ROOT = os.environ.get('CCD_ROOT', '.')
Z = np.load(os.path.join(ROOT, 'cache_lc.npz'))
OUT = os.path.join(ROOT, 'out', 'legacy_ccb.json')
SH = ['FA', 'L2', 'V15', 'MI', 'MB', 'V13']
TASKS = ['arc', 'winogrande', 'hellaswag', 'mmlu']
B = int(os.environ.get('BOOT', 300))
rng = np.random.default_rng(20260830)

def boot(g, B):
    out = []
    n = g.size
    for _ in range(B):
        o = ccb(g[rng.integers(0, n, n)])
        if np.isfinite(o.get('rho', np.nan)): out.append(o['rho'])
    if not out: return None
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

res = {}
for tk in TASKS:
    L = Z['L_' + tk]; C = Z['C_' + tk]
    n = L.shape[1]
    rows = []
    for i, j in itertools.combinations(range(6), 2):
        rec = {'pair': SH[i] + ':' + SH[j], 'n': int(n)}
        for reg, M in (('raw', L), ('acc', -C)):
            gij = M[i] - M[j]
            a, b = (i, j) if gij.mean() < 0 else (j, i)
            g = M[a] - M[b]
            o = ccb(g)
            if not np.isfinite(o.get('rho', np.nan)):
                rec[reg] = None; continue
            d = {k: float(o[k]) for k in ('rho', 'beta', 'half_t2', 't', 'skew', 'tau')}
            d['winner'] = SH[a]; d['loser'] = SH[b]
            d['m'] = float(-g.mean())
            d['ratio'] = d['rho'] / d['half_t2'] if d['half_t2'] > 0 else None
            d['n_req_05'] = float(np.log(20.0) / d['rho']) if d['rho'] > 0 else None
            d['certified'] = bool(d['n_req_05'] is not None and d['n_req_05'] <= n)
            d['sign_ok'] = bool(np.sign(d['rho'] - d['half_t2']) == -np.sign(d['skew']))
            if reg == 'raw': d['ci95'] = boot(g, B)
            rec[reg] = d
        rows.append(rec)
    summ = {}
    for reg in ('raw', 'acc'):
        S = [r[reg] for r in rows if r.get(reg)]
        summ[reg] = {'n': int(n), 'n_pairs': len(S),
                     'median_rho': float(np.median([s['rho'] for s in S])),
                     'min_rho': float(min(s['rho'] for s in S)),
                     'max_rho': float(max(s['rho'] for s in S)),
                     'certified': int(sum(s['certified'] for s in S)),
                     'sign_ok': int(sum(s['sign_ok'] for s in S)),
                     'neg_skew': int(sum(s['skew'] < 0 for s in S)),
                     'spearman': float(spearmanr([s['rho'] for s in S],
                                                 [s['half_t2'] for s in S]).statistic),
                     'median_tau': float(np.median([s['tau'] for s in S])),
                     'max_tau': float(max(s['tau'] for s in S))}
    res[tk] = {'rows': rows, 'summary': summ}
    print(tk, n, summ['raw'], flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('WROTE', OUT, flush=True)
