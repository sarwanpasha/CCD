import os
"""Bradley-Terry leaderboard fitting and the influence-function contrast that
turns a rank comparison into a scalar mean comparison the CCB radius can certify.
"""
import numpy as np
from scipy.optimize import minimize

def sigmoid(z):
    out = np.empty_like(z)
    p = z >= 0
    out[p] = 1.0/(1.0+np.exp(-z[p]))
    e = np.exp(z[~p]); out[~p] = e/(1.0+e)
    return out

def fit_bt(ia, ib, y, w=None, m=None, l2=1e-6, tol=1e-12, maxit=200):
    n = ia.size
    m = int(max(ia.max(), ib.max()) + 1) if m is None else m
    w = np.full(n, 1.0/n) if w is None else np.asarray(w, float)

    def obj(th):
        z = th[ia] - th[ib]
        p = sigmoid(z)
        ll = w @ (y*np.log(np.maximum(p,1e-300)) + (1-y)*np.log(np.maximum(1-p,1e-300)))
        r = y - p
        gr = np.zeros(m)
        np.add.at(gr, ia, -w*r)
        np.add.at(gr, ib,  w*r)
        return -ll + 0.5*l2*(th@th), gr + l2*th

    th0 = np.zeros(m)
    res = minimize(obj, th0, jac=True, method='L-BFGS-B',
                   options={'maxiter': maxit, 'ftol': 1e-18, 'gtol': tol})
    th = res.x - res.x.mean()
    return th, res

def bt_pieces(ia, ib, y, th, w=None):
    n = ia.size
    w = np.full(n, 1.0/n) if w is None else np.asarray(w, float)
    z = th[ia] - th[ib]
    p = sigmoid(z)
    r = y - p
    v = p*(1-p)
    m = th.size
    H = np.zeros((m, m))
    np.add.at(H, (ia, ia),  w*v)
    np.add.at(H, (ib, ib),  w*v)
    np.add.at(H, (ia, ib), -w*v)
    np.add.at(H, (ib, ia), -w*v)
    return r, v, H

def psi_contrast(ia, ib, r, H, A, B):
    m = H.shape[0]
    c = np.zeros(m); c[A] = 1.0; c[B] = -1.0
    Hp = np.linalg.pinv(H, rcond=1e-10)
    u = Hp @ c
    return r * (u[ia] - u[ib])

def g_reversal(ia, ib, y, th, A, B, w=None):
    r, v, H = bt_pieces(ia, ib, y, th, w)
    psi = psi_contrast(ia, ib, r, H, A, B)
    delta = th[A] - th[B]
    return -(delta + psi), delta, psi

def _kl(w):
    n = w.size
    return float(w @ np.log(np.maximum(w*n, 1e-300)))

def exact_radius(ia, ib, y, A, B, iters=6, tol=1e-10, verbose=False):
    n = ia.size
    lw = np.full(n, -np.log(n))
    def norm(l):
        l = l - l.max()
        w = np.exp(l); s = w.sum()
        return w/s, l - np.log(s)
    w, lw = norm(lw)
    base = np.full(n, -np.log(n))
    hist = []
    for it in range(iters):
        th, _ = fit_bt(ia, ib, y, w=w)
        r, v, H = bt_pieces(ia, ib, y, th, w)
        psi = psi_contrast(ia, ib, r, H, A, B)
        def state(t):
            ww, ll = norm(base - t*psi)
            th2, _ = fit_bt(ia, ib, y, w=ww)
            return th2[A] - th2[B], ww, ll
        d0, _, _ = state(0.0)
        if d0 <= 0:
            break
        hi = 1.0; dh, _, _ = state(hi); k = 0
        while dh > 0 and k < 40:
            hi *= 2.0; dh, _, _ = state(hi); k += 1
        if dh > 0:
            hist.append({'iter': it, 'fail': 'no bracket', 'delta': float(dh)})
            break
        lo = 0.0
        for _ in range(50):
            mid = 0.5*(lo+hi)
            dm, wm, lm = state(mid)
            if dm > 0: lo = mid
            else: hi = mid
            if hi - lo < 1e-13*max(1.0, hi): break
        d_fin, w_new, lw_new = state(hi)
        kl_new = _kl(w_new)
        hist.append({'iter': it, 't': hi, 'kl': kl_new, 'delta': float(d_fin)})
        if verbose: print(hist[-1], flush=True)
        improved = (len(hist) < 2) or (hist[-2].get('kl', np.inf) - kl_new > tol*max(1e-12, kl_new))
        w, lw = w_new, lw_new
        if not improved:
            break
    return {'rho_exact': _kl(w), 'w': w, 'hist': hist}
