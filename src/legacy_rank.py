"""Rank-r certificates on the retired archive, recomputed from the pinned matrices."""
import json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccb import ccb, kway_rho

ROOT = os.environ.get('CCD_ROOT', '.')
Z = np.load(os.path.join(ROOT, 'cache_lc.npz'))
OUT = os.path.join(ROOT, 'out', 'legacy_rank.json')
SH = ['FA', 'L2', 'V15', 'MI', 'MB', 'V13']
TASKS = ['arc', 'winogrande', 'hellaswag', 'mmlu']
B = int(os.environ.get('BOOT', 300))
rng = np.random.default_rng(20260830)

def rho_of(g):
    o = ccb(g)
    return float(o['rho']) if np.isfinite(o.get('rho', np.nan)) else np.inf

out = {'projections': [], 'certificates': []}
for tk in TASKS:
    for reg, M in (('raw', Z['L_' + tk]), ('acc', -Z['C_' + tk])):
        n = M.shape[1]
        ml = M.mean(1)
        order = list(np.argsort(ml))              # best first
        pair_rho = {}
        for a in range(6):
            for b in range(6):
                if a == b: continue
                g = M[a] - M[b]
                if g.mean() < 0: pair_rho[(a, b)] = rho_of(g)
        for ti, A in enumerate(order):
            rivals = [j for j in order if ml[j] > ml[A]]
            for r in (2, 3):
                if len(rivals) < r: continue
                best = None
                for S in itertools.combinations(rivals, r):
                    G = np.column_stack([M[A] - M[j] for j in S])
                    res = kway_rho(G)
                    act = [int(j) for j, l in zip(S, res['lam']) if l > 1e-9]
                    rec = {'task': tk, 'reg': reg, 'target': SH[A], 'r': r,
                           'S': [SH[int(j)] for j in S], 'rho': float(res['rho']),
                           'a': len(act), 'active': [SH[j] for j in act],
                           'lower': float(max(pair_rho[(A, int(j))] for j in S)),
                           'sum': float(sum(pair_rho[(A, int(j))] for j in S))}
                    out['projections'].append(rec)
                    if best is None or rec['rho'] < best['rho']: best = dict(rec)
                pr = sorted(pair_rho[(A, int(j))] for j in rivals)
                best['order_stat'] = float(pr[r - 1])
                best['gap_abs'] = best['rho'] - best['order_stat']
                best['gap_pct'] = 100 * best['gap_abs'] / best['order_stat'] if best['order_stat'] > 0 else None
                best['cheapest_r'] = sorted([SH[int(j)] for j in rivals],
                                            key=lambda s: pair_rho[(A, SH.index(s))])[:r]
                best['argmin_is_cheapest'] = sorted(best['S']) == sorted(best['cheapest_r'])
                best['bracket_pos'] = ((best['rho'] - best['lower']) / (best['sum'] - best['lower'])
                                       if best['sum'] > best['lower'] else None)
                out['certificates'].append(best)
        print(tk, reg, 'done', flush=True)

# bootstrap the headline example: ARC, raw, Mistral-base target
M = Z['L_arc']; ml = M.mean(1); order = list(np.argsort(ml))
A = SH.index('MB')
rivals = [j for j in order if ml[j] > ml[A]]
ex = {'target': 'MB', 'task': 'arc', 'reg': 'raw', 'rivals': [SH[j] for j in rivals]}
for r in (1, 2, 3):
    if len(rivals) < r: continue
    subs = list(itertools.combinations(rivals, r))
    def cert(Mx):
        return min(kway_rho(np.column_stack([Mx[A] - Mx[j] for j in S]))['rho'] for S in subs)
    pt = cert(M)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, M.shape[1], M.shape[1])
        try: bs.append(cert(M[:, idx]))
        except Exception: pass
    bestS = min(subs, key=lambda S: kway_rho(np.column_stack([M[A] - M[j] for j in S]))['rho'])
    res = kway_rho(np.column_stack([M[A] - M[j] for j in bestS]))
    ex['r%d' % r] = {'rho': float(pt), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     'S': [SH[int(j)] for j in bestS],
                     'active': [SH[int(j)] for j, l in zip(bestS, res['lam']) if l > 1e-9]}
    print('example r=%d rho=%.6f ci=%s active=%s' % (r, pt, ex['r%d' % r]['ci'], ex['r%d' % r]['active']), flush=True)
out['example'] = ex
json.dump(out, open(OUT, 'w'), indent=1)
print('WROTE', OUT, len(out['projections']), 'projections', len(out['certificates']), 'certificates', flush=True)
