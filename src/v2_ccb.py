"""CCB analysis of the modern (Open LLM Leaderboard v2) loss matrices."""
import json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccb import ccb, kway_rho
from scipy.stats import spearmanr

ROOT = os.environ.get('CCD_ROOT', '.')
Z = np.load(os.path.join(ROOT, 'cache_v2.npz'), allow_pickle=True)
OUT = os.path.join(ROOT, 'out', 'v2_ccb.json')
B = int(os.environ.get('BOOT', 300))
NTGT = int(os.environ.get('NTGT', 3))
CAP = 1e3
rng = np.random.default_rng(20260830)

def radii(g):
    o = ccb(g)
    rho = o.get('rho', np.nan)
    if (g > 0).sum() == 0 and (g == 0).sum() == 0: return 'INF'
    if rho is None or np.isnan(rho): return None
    if not np.isfinite(rho) or rho > CAP: return 'INF'
    return {k: float(o[k]) for k in ('rho', 'beta', 'half_t2', 't', 'skew', 'tau', 'EQg')}

def boot_rho(g, B):
    n = g.size
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        o = ccb(g[idx])
        if np.isfinite(o.get('rho', np.nan)): out.append(o['rho'])
    if not out: return None
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

def main():
    res = {'families': {}}
    for fam in ('mmlu_pro', 'bbh', 'musr', 'gpqa'):
        if 'L_' + fam not in Z.files: continue
        L = Z['L_' + fam]; A = Z['A_' + fam]
        models = [str(x) for x in Z['models_' + fam]]
        m, n = L.shape
        print('==', fam, 'models', models, 'n', n, flush=True)
        fam_out = {'models': models, 'n': int(n), 'pairs': [], 'rank': []}
        mean_loss = L.mean(1); acc = np.nanmean(A, 1)
        fam_out['accuracy'] = {models[i]: float(acc[i]) for i in range(m)}
        fam_out['mean_loss'] = {models[i]: float(mean_loss[i]) for i in range(m)}
        for i, j in itertools.combinations(range(m), 2):
            rec = {'pair': models[i] + ':' + models[j]}
            for reg, M in (('raw', L), ('acc', -A)):
                gij = M[i] - M[j]
                a, b = (i, j) if gij.mean() < 0 else (j, i)
                g = M[a] - M[b]
                r = radii(g)
                if r is None:
                    rec[reg] = None; continue
                if r == 'INF':
                    rec[reg] = {'rho': None, 'infinite': True, 'winner': models[a],
                                'n_req_05': 1.0, 'certified': True}
                    continue
                r['infinite'] = False
                r['winner'] = models[a]
                r['n_req_05'] = float(np.log(20.0)/r['rho']) if r['rho'] > 0 else None
                r['certified'] = bool(r['n_req_05'] is not None and r['n_req_05'] <= n)
                r['sign_ok'] = bool(np.sign(r['rho'] - r['half_t2']) == -np.sign(r['skew']))
                rec[reg] = r
            fam_out['pairs'].append(rec)
        for reg in ('raw', 'acc'):
            got = [p[reg] for p in fam_out['pairs'] if p.get(reg)]
            S = [s for s in got if not s['infinite']]
            n_inf = sum(1 for s in got if s['infinite'])
            if not S: continue
            fam_out['summary_' + reg] = {
                'median_rho': float(np.median([s['rho'] for s in S])),
                'min_rho': float(min(s['rho'] for s in S)),
                'max_rho': float(max(s['rho'] for s in S)),
                'certified': int(sum(s['certified'] for s in S)), 'n_pairs': len(S),
                'n_inf': int(n_inf), 'n_total': int(m*(m-1)//2),
                'certified_total': int(sum(s['certified'] for s in S) + n_inf),
                'spearman_rho_halft2': float(spearmanr([s['rho'] for s in S], [s['half_t2'] for s in S]).statistic),
                'sign_ok': int(sum(s['sign_ok'] for s in S)),
                'neg_skew': int(sum(s['skew'] < 0 for s in S)),
                'median_tau': float(np.median([s['tau'] for s in S])),
                'max_tau': float(max(s['tau'] for s in S))}
            print(' ', reg, fam_out['summary_' + reg], flush=True)
        for reg, M in (('raw', L), ('acc', -A)):
            ml = M.mean(1)
            best_first = list(np.argsort(ml))
            for A_ in best_first[:NTGT]:
                rivals0 = [j for j in best_first if ml[j] > ml[A_]]
                pr = {}
                for j in rivals0:
                    o = ccb(M[A_] - M[j])
                    v = o.get('rho', np.nan)
                    pr[int(j)] = float(v) if (v is not None and np.isfinite(v) and v <= CAP) else np.inf
                rivals = [j for j in rivals0 if np.isfinite(pr[int(j)])]
                if len(rivals) < 3: continue
                cols = {int(j): M[A_] - M[j] for j in rivals}
                for r in (1, 2, 3):
                    best = None
                    for S in itertools.combinations(rivals, r):
                        G = np.column_stack([cols[int(j)] for j in S])
                        out = kway_rho(G)
                        if not np.isfinite(out['rho']) or out['rho'] > CAP: continue
                        if best is None or out['rho'] < best['rho']:
                            best = {'rho': float(out['rho']), 'S': [models[int(j)] for j in S],
                                    'lam': [float(x) for x in out['lam']],
                                    'active': [models[int(j)] for j, l in zip(S, out['lam']) if l > 1e-9]}
                    if best is None: continue
                    os_ = float(sorted(pr[int(j)] for j in rivals)[r-1])
                    best.update(regime=reg, target=models[int(A_)], r=r, order_stat=os_,
                                gap_pct=(100*(best['rho']-os_)/os_ if os_ > 0 else None),
                                a=len(best['active']))
                    fam_out['rank'].append(best)
                    print('  rank %s r=%d %-6s rho=%.6f os=%.6f gap=%s a=%d' %
                          (reg, r, best['target'], best['rho'], os_,
                           ('%.2f%%' % best['gap_pct']) if best['gap_pct'] is not None else 'na',
                           best['a']), flush=True)
        for p in fam_out['pairs']:
            q = p.get('raw')
            if q and not q['infinite']:
                i, j = [models.index(x) for x in p['pair'].split(':')]
                a, b = (i, j) if (L[i]-L[j]).mean() < 0 else (j, i)
                q['ci95'] = boot_rho(L[a]-L[b], B)
        res['families'][fam] = fam_out
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print('WROTE', OUT, flush=True)

if __name__ == '__main__':
    main()
