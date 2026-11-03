"""CCB certificates for the LMArena Bradley-Terry leaderboard."""
import json, os, sys, time, itertools
import numpy as np
import pyarrow.parquet as pq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ccb import ccb, kway_rho
from btlib import fit_bt, bt_pieces, psi_contrast, exact_radius

ROOT = os.environ.get('CCD_ROOT', '.')
DATA = os.path.join(ROOT, 'data_arena', 'battles_140k.parquet')
OUT  = os.path.join(ROOT, 'out', 'arena_ccb.json')
MIN_BATTLES = int(os.environ.get('MIN_BATTLES', 800))
TOPK        = int(os.environ.get('TOPK', 10))
NPROC       = int(os.environ.get('NPROC', 16))
WINMAP = {'model_a': 1.0, 'model_b': 0.0}

def load():
    t = pq.read_table(DATA, columns=['model_a', 'model_b', 'winner'])
    a = np.array(t.column('model_a').to_pylist())
    b = np.array(t.column('model_b').to_pylist())
    w = np.array(t.column('winner').to_pylist())
    y = np.array([WINMAP.get(x, 0.5) for x in w], float)
    keep = a != b
    a, b, y = a[keep], b[keep], y[keep]
    names, counts = np.unique(np.concatenate([a, b]), return_counts=True)
    good = set(names[counts >= MIN_BATTLES])
    keep = np.array([(x in good) and (z in good) for x, z in zip(a, b)])
    a, b, y = a[keep], b[keep], y[keep]
    models = sorted(good)
    idx = {m: i for i, m in enumerate(models)}
    ia = np.array([idx[x] for x in a], np.int64)
    ib = np.array([idx[x] for x in b], np.int64)
    return models, ia, ib, y

def pair_job(args):
    A, B, ia, ib, y, th = args
    t0 = time.time()
    r, v, H = bt_pieces(ia, ib, y, th)
    psi = psi_contrast(ia, ib, r, H, A, B)
    delta = float(th[A] - th[B])
    g = -(delta + psi)
    lin = ccb(g)
    ex = exact_radius(ia, ib, y, A, B, iters=6)
    return {'A': A, 'B': B, 'delta': delta,
            'rho_lin': lin['rho'], 'half_t2': lin['half_t2'], 'skew': lin['skew'],
            'beta': lin['beta'], 'tau': lin['tau'],
            'rho_exact': ex['rho_exact'], 'ratio': lin['rho']/ex['rho_exact'],
            'n_iters': len(ex['hist']), 'hist': ex['hist'], 'secs': time.time()-t0}

def main():
    models, ia, ib, y = load()
    n = ia.size
    print('models', len(models), 'battles', n, flush=True)
    th, _ = fit_bt(ia, ib, y)
    order = np.argsort(-th)
    rank = [{'model': models[i], 'theta': float(th[i]),
             'elo': float(1000 + 400*th[i]/np.log(10)),
             'n_battles': int((ia == i).sum() + (ib == i).sum())} for i in order]
    for k, rr in enumerate(rank[:25]):
        print('%2d %-42s theta=%+.4f elo=%.0f n=%d' % (k+1, rr['model'], rr['theta'], rr['elo'], rr['n_battles']), flush=True)
    top = list(order[:TOPK])
    jobs = [(int(A), int(B), ia, ib, y, th) for i, A in enumerate(top) for B in top[i+1:]]
    print('pairs', len(jobs), flush=True)
    with Pool(NPROC) as p:
        pairs = p.map(pair_job, jobs)
    for d in pairs:
        d['A_name'] = models[d['A']]; d['B_name'] = models[d['B']]
        print('%-30s vs %-30s d=%.4f rho_lin=%.6f rho_exact=%.6f ratio=%.4f' %
              (d['A_name'][:30], d['B_name'][:30], d['delta'], d['rho_lin'], d['rho_exact'], d['ratio']), flush=True)
    r_out = []
    rr, vv, HH = bt_pieces(ia, ib, y, th)
    for A in top[:5]:
        rivals = [j for j in top if th[j] < th[A]]
        if len(rivals) < 3: continue
        cols = {}
        for j in rivals:
            psi = psi_contrast(ia, ib, rr, HH, int(A), int(j))
            cols[int(j)] = -(float(th[A] - th[j]) + psi)
        pair_rho = {int(j): ccb(cols[int(j)])['rho'] for j in rivals}
        for r in (1, 2, 3):
            best = None
            for S in itertools.combinations(rivals, r):
                G = np.column_stack([cols[int(j)] for j in S])
                res = kway_rho(G)
                if best is None or res['rho'] < best['rho']:
                    best = {'rho': float(res['rho']), 'S': [models[int(j)] for j in S],
                            'lam': [float(x) for x in res['lam']],
                            'active': [models[int(j)] for j, l in zip(S, res['lam']) if l > 1e-9]}
            best.update(target=models[int(A)], r=r,
                        order_stat=float(sorted(pair_rho.values())[r-1]))
            best['gap_pct'] = 100*(best['rho'] - best['order_stat'])/best['order_stat']
            r_out.append(best)
            print('rank r=%d %-30s rho=%.6f order_stat=%.6f gap=%.2f%% active=%s' %
                  (r, best['target'][:30], best['rho'], best['order_stat'], best['gap_pct'], best['active']), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({'models': models, 'n_battles': int(n), 'min_battles': MIN_BATTLES,
               'ranking': rank, 'pairs': pairs, 'rank_certs': r_out}, open(OUT, 'w'), indent=1)
    print('WROTE', OUT, flush=True)

if __name__ == '__main__':
    main()
