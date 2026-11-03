import os
"""
Vectorised solvers for the CCD inference layer.

Two fast paths:
  * ternary  g in {-a, 0, a}  -> closed form in the counts (n-, n0, n+);
    bootstrap is a multinomial resample of the counts, so B draws cost O(B).
  * general  -> damped Newton on K'(t) = 0 run simultaneously over a (B, n) block.

Closed form (ternary, a > 0):
    beta = (1/2a) log(n+/n-),  M = (2 sqrt(n+ n-) + n0)/n,  rho = -log M,
    sigma^2 = exp(2 rho) - 1          [exact: M(2 beta) == 1 for this support]
"""
import numpy as np

__all__ = ["rho_ternary", "boot_ternary", "rho_general", "sigma2_general"]

_INF = np.inf


def rho_ternary(nm, n0, npv):
    """rho_hat from counts. Arrays broadcast. +inf when nm == 0 and n0 == 0."""
    nm, n0, npv = np.atleast_1d(nm).astype(float), np.atleast_1d(n0).astype(float), np.atleast_1d(npv).astype(float)
    n = nm + n0 + npv
    M = (2.0 * np.sqrt(npv * nm) + n0) / n
    rho = np.where(npv <= nm, 0.0, -np.log(np.where(M > 0, M, 1.0)))
    rho = np.where((nm == 0) & (n0 == 0), _INF, rho)
    return rho


def boot_ternary(nm, n0, npv, nboot, rng):
    """Nonparametric bootstrap for a ternary witness: multinomial on the counts."""
    n = int(nm + n0 + npv)
    p = np.array([nm, n0, npv], dtype=float) / n
    c = rng.multinomial(n, p, size=nboot)
    return rho_ternary(c[:, 0], c[:, 1], c[:, 2])


def _newton(G, t0, iters=200, tol=1e-13):
    """Safeguarded Newton for E_{Q_t}[g] = 0, row-wise on G (B, n).

    E_{Q_t}[g] is strictly decreasing in t, positive at t = 0 (mean > 0) and
    negative for large t when some g < 0.  A bracket [lo, hi] is kept for every
    row; a Newton step is accepted only if it stays strictly inside the bracket,
    otherwise the row bisects.  This cannot stall when the tilted variance
    underflows on heavy-tailed witnesses.
    """
    B = G.shape[0]
    def eq(t):
        a = -t[:, None] * G
        a -= a.max(axis=1, keepdims=True)
        w = np.exp(a); s = w.sum(axis=1)
        m1 = (w * G).sum(axis=1) / s
        v = (w * G * G).sum(axis=1) / s - m1 * m1
        return m1, v
    lo = np.zeros(B)
    hi = np.ones(B)
    for _ in range(200):                                   # expand brackets
        m1, _v = eq(hi)
        bad = m1 > 0
        if not bad.any():
            break
        lo = np.where(bad, hi, lo); hi = np.where(bad, 2.0 * hi, hi)
    t = np.clip(np.asarray(t0, float), lo, hi)
    t = np.where((t <= lo) | (t >= hi), 0.5 * (lo + hi), t)
    for _ in range(iters):
        m1, v = eq(t)
        lo = np.where(m1 > 0, t, lo); hi = np.where(m1 <= 0, t, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            tn = t + m1 / v
        ok = np.isfinite(tn) & (tn > lo) & (tn < hi) & (v > 0)
        tnew = np.where(ok, tn, 0.5 * (lo + hi))
        if np.max(np.abs(tnew - t) / np.maximum(1.0, np.abs(t))) < tol:
            t = tnew; break
        t = tnew
    return t


def rho_general(G, return_beta=False):
    """
    rho_hat for each row of G (shape (B, n)). Rows with mean <= 0 give 0;
    rows with no non-positive entry give +inf.
    """
    G = np.atleast_2d(np.asarray(G, dtype=float))
    B, n = G.shape
    mu = G.mean(axis=1)
    V = G.var(axis=1)
    t = np.where(V > 0, np.maximum(mu, 0.0) / np.maximum(V, 1e-300), 0.0)
    t = _newton(G, t)
    a = -t[:, None] * G
    mx = a.max(axis=1, keepdims=True)
    logM = (mx[:, 0] + np.log(np.exp(a - mx).mean(axis=1)))
    rho = -logM
    rho = np.where(mu <= 0, 0.0, rho)
    rho = np.where((G > 0).all(axis=1), _INF, rho)
    rho = np.maximum(rho, 0.0)
    return (rho, t) if return_beta else rho


def sigma2_general(G, beta):
    """Plug-in sigma^2 = mean(w^2) - 1 with w the self-normalised tilt weights."""
    G = np.atleast_2d(np.asarray(G, dtype=float))
    a = -np.asarray(beta, dtype=float)[:, None] * G
    a -= a.max(axis=1, keepdims=True)
    w = np.exp(a)
    w /= w.mean(axis=1, keepdims=True)
    return (w * w).mean(axis=1) - 1.0


def rho_weighted(g, w, iters=200, tol=1e-13):
    """rho for the weighted empirical law sum_i w_i delta_{g_i}. Returns (rho, beta)."""
    g = np.asarray(g, float); w = np.asarray(w, float)
    keep = w > 0
    g, w = g[keep], w[keep] / w[keep].sum()
    mu = float(w @ g)
    if mu <= 0:
        return 0.0, 0.0
    if not np.any(g < 0):
        if np.any(g == 0):
            return float(-np.log(w[g == 0].sum())), np.inf
        return np.inf, np.inf
    def eq(t):
        a = -t * g; a -= a.max(); q = w * np.exp(a); q /= q.sum()
        m1 = float(q @ g); return m1, float(q @ (g * g)) - m1 * m1
    lo, hi = 0.0, 1.0
    while eq(hi)[0] > 0:
        lo, hi = hi, 2.0 * hi
    v0 = float(w @ (g - mu) ** 2)
    t = mu / v0 if v0 > 0 else 0.5 * (lo + hi)
    if not (lo < t < hi):
        t = 0.5 * (lo + hi)
    for _ in range(iters):
        m1, v = eq(t)
        if m1 > 0: lo = t
        else: hi = t
        tn = t + m1 / v if v > 0 else np.nan
        tnew = tn if (np.isfinite(tn) and lo < tn < hi) else 0.5 * (lo + hi)
        if abs(tnew - t) < tol * max(1.0, abs(t)):
            t = tnew; break
        t = tnew
    a = -t * g; mx = a.max()
    return float(-(mx + np.log(w @ np.exp(a - mx)))), float(t)


def sigma2_weighted(g, w, beta):
    g = np.asarray(g, float); w = np.asarray(w, float)
    a = -beta * g; a -= a.max()
    e = np.exp(a); m = float(w @ e)
    return float(w @ (e / m) ** 2) - 1.0
