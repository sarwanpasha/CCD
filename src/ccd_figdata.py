"""
Plot data, recomputed from the cached evaluation records.

Sign convention: g = loss(A) - loss(B) with A the nominal winner (mean g < 0);
the reversal event is mean(g) >= 0 and the witness tilt is exp(beta* g).
Writes a single JSON file.
"""
import argparse, itertools, json, os, sys
import numpy as np
from math import erfc, sqrt, log, pi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccd_infer as C

SH = ['FA', 'L2', 'V15', 'MI', 'MB', 'V13']        # row order of the retired-archive cache


def signed_g(L, i, j):
    """Orient so A is the winner; return (A, B, g) with g = L[A] - L[B], mean < 0."""
    a, b = (i, j) if L[i].mean() < L[j].mean() else (j, i)
    return a, b, L[a] - L[b]


def pair_stats(g):
    gm = -g                                          # positive-mean form used by the solver
    st, rho, beta = C.solve(gm)
    m, s = -g.mean(), g.std()
    t = m / s
    sk = float(((g - g.mean()) ** 3).mean() / s ** 3)
    half = 0.5 * t * t
    return dict(rho=rho, beta=beta, t=t, half_t2=half, skew=sk, m=m,
                ratio=rho / half, tau=abs(rho - half) / half, tau_pred=abs(sk) * abs(t) / 3)


def witness_curve(g, beta, grid):
    w = np.exp(beta * g - (beta * g).max()); w /= w.mean()
    ex = np.sort(np.maximum(w - 1.0, 0.0))[::-1]
    cum = np.cumsum(ex) / ex.sum()
    n = g.size
    return [float(cum[max(0, int(np.ceil(f * n)) - 1)]) if f > 0 else 0.0 for f in grid]


def phi_bar(x):
    return 0.5 * erfc(x / sqrt(2.0))


def calibration(g, rho, beta, grid, R_is, R_mc, rng):
    """Reversal probability P(mean_{n'} g >= 0) four ways."""
    n = g.size
    a = beta * g; w = np.exp(a - a.max()); q = w / w.sum()
    K2 = float(q @ g ** 2 - (q @ g) ** 2)            # K''(beta*) = tilted variance
    m, s = -g.mean(), g.std()
    out = []
    for k in grid:
        S = np.empty(R_is)
        for s0 in range(0, R_is, 2000):
            r = min(2000, R_is - s0)
            S[s0:s0 + r] = g[rng.choice(n, size=(r, k), p=q)].sum(1)
        Z = np.exp(-beta * S - k * rho) * (S >= 0)
        p_is = float(Z.mean()); rse = float(Z.std(ddof=1) / (Z.mean() * sqrt(R_is))) if Z.mean() > 0 else float('nan')
        br = float(np.exp(-k * rho) / (beta * sqrt(2 * pi * k * K2)))
        gauss = float(phi_bar(sqrt(k) * m / s))
        mc = None
        if p_is * R_mc >= 50:
            hits = 0
            for s0 in range(0, R_mc, 5000):
                r = min(5000, R_mc - s0)
                hits += int((g[rng.integers(0, n, size=(r, k))].sum(1) >= 0).sum())
            mc = hits / R_mc
        out.append(dict(n=k, is_=p_is, is_rse=rse, br=br, gauss=gauss, mc=mc, nrho=k * rho))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.environ.get('CCD_ROOT', '.'))
    ap.add_argument('--out', default='figdata.json')
    a = ap.parse_args()
    rng = np.random.default_rng(20261103)
    Z = np.load(os.path.join(a.root, 'cache_lc.npz'))
    res = {'pairs': {}}
    for fam in ('arc', 'winogrande', 'hellaswag', 'mmlu'):
        L = Z['L_' + fam]; rows = []
        for i, j in itertools.combinations(range(L.shape[0]), 2):
            A, B, g = signed_g(L, i, j)
            d = pair_stats(g); d.update(A=SH[A], B=SH[B], n=int(g.size)); rows.append(d)
        res['pairs'][fam] = rows
    L = Z['L_arc']
    grid = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    i, j = SH.index('MI'), SH.index('V15')
    A, B, g = signed_g(L, i, j); d = pair_stats(g)
    res['witness'] = dict(A=SH[A], B=SH[B], grid=grid, curve=witness_curve(g, d['beta'], grid),
                          top5=witness_curve(g, d['beta'], [0.05])[0])
    cal = {}
    for (x, y), grd in (((('MB', 'V15')), [5, 10, 20, 40, 80, 160, 320, 640, 1169]),
                        ((('MI', 'V15')), [10, 20, 40, 80, 160, 320])):
        A, B, g = signed_g(L, SH.index(x), SH.index(y)); d = pair_stats(g)
        cal[SH[A] + ':' + SH[B]] = dict(rho=d['rho'], beta=d['beta'],
                                        rows=calibration(g, d['rho'], d['beta'], grd, 30000, 200000, rng))
    res['calibration'] = cal
    p = os.path.join(a.root, 'out', 'arena_ccb.json')
    if os.path.exists(p):
        ar = json.load(open(p))
        res['arena_exact'] = [dict(A=q['A_name'], B=q['B_name'], rho_lin=q['rho_lin'],
                                   rho_exact=q['rho_exact']) for q in ar['pairs']]
        res['arena_n'] = ar['n_battles']
    json.dump(res, open(a.out, 'w'), indent=0)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
