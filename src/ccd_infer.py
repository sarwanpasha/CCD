import os
"""
CCD inference layer for the certified breakdown radius.

rho*(P) = inf{ KL(Q||P) : E_Q[g] <= 0 }  =  -min_{t>=0} K_P(t),
K_P(t) = log E_P[exp(-t g)].

Plug-in estimator rho_hat = rho*(P_hat_n) with:
  * envelope/Danskin influence function  psi = 1 - dQ*/dP
  * asymptotic variance sigma^2 = chi2(Q*||P) = M(2b*)/M(b*)^2 - 1
  * SE identity  SE = sqrt(1/ESS - 1/n)   (ESS = Kish ESS of the tilt weights)
  * exact ternary identity sigma^2 = exp(2 rho*) - 1 for g supported on {-a,0,a}
  * variance-stabilising transform u = arctan(sqrt(exp(2 rho)-1)), Var(u) -> 1/n
"""
import numpy as np

__all__ = ["solve", "tilt_weights", "infer", "INF", "ZERO", "FIN"]

INF, ZERO, FIN = "inf", "zero", "finite"
_TMAX = 1e4


def _logM(t, g):
    """log E_hat[exp(-t g)] via log-sum-exp (stable for large t)."""
    a = -t * g
    m = a.max()
    return m + np.log(np.mean(np.exp(a - m)))


def _EQ_g(t, g):
    """E_{Q_t}[g] where dQ_t ∝ exp(-t g) dP_hat.  Equals -K'(t)."""
    a = -t * g
    a -= a.max()
    w = np.exp(a)
    return float(np.dot(w, g) / w.sum())


def solve(g, tol=1e-12):
    """Return (status, rho_hat, beta_hat) for a sample of witness values g."""
    g = np.asarray(g, dtype=float)
    n = g.size
    if n == 0:
        raise ValueError("empty witness sample")
    if g.mean() <= 0:
        return ZERO, 0.0, 0.0
    if not np.any(g < 0):
        # no item can be up-weighted to drag the mean non-positive
        if np.any(g == 0):
            # infimum approached as t -> inf, value -log P_hat(g == 0)
            return FIN, float(-np.log(np.mean(g == 0))), np.inf
        return INF, np.inf, np.inf

    # K' is increasing from -mean(g) < 0 to |min g| > 0; bracket its root.
    lo, hi = 0.0, 1.0
    while _EQ_g(hi, g) > 0 and hi < _TMAX:   # E_Q[g] > 0  <=>  K'(t) < 0
        lo, hi = hi, hi * 2.0
    if hi >= _TMAX:
        return INF, np.inf, np.inf

    # bisection on E_{Q_t}[g] = 0 (monotone decreasing in t)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _EQ_g(mid, g) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    beta = 0.5 * (lo + hi)
    return FIN, float(-_logM(beta, g)), float(beta)


def tilt_weights(g, beta):
    """Self-normalised tilt weights w_i = e^{-beta g_i} / mean_j e^{-beta g_j}; mean(w) == 1."""
    a = -beta * np.asarray(g, dtype=float)
    a -= a.max()
    w = np.exp(a)
    return w / w.mean()


def _ternary_scale(g, atol=1e-9):
    """If g is supported on {-a, 0, a}, return a; else None."""
    v = np.unique(np.asarray(g, dtype=float))
    v = v[np.abs(v) > atol]
    if v.size == 0:
        return None
    a = np.abs(v).max()
    return a if np.all(np.abs(np.abs(v) - a) <= atol * max(1.0, a)) else None


def vst(rho):
    """Variance-stabilising transform u = arctan(sqrt(e^{2 rho} - 1)) in [0, pi/2)."""
    rho = np.asarray(rho, dtype=float)
    return np.arctan(np.sqrt(np.expm1(2.0 * np.minimum(rho, 700.0))))


def vst_inv(u):
    """Inverse: rho = log sec(u), with u >= pi/2 mapped to +inf."""
    u = np.asarray(u, dtype=float)
    out = np.where(u >= np.pi / 2, np.inf, -np.log(np.cos(np.clip(u, 0.0, np.pi / 2 - 1e-15))))
    return np.where(u <= 0, 0.0, out)


def infer(g, alpha=0.05, nboot=0, seed=0, boot_batch=None):
    """
    Point estimate, plug-in SE, and confidence intervals for rho*.

    Returns a dict with:
      status, rho, beta, sigma2 (plug-in), se, ess,
      sigma2_ternary (exact e^{2 rho}-1 when the witness is ternary, else None),
      ci_normal, ci_vst, and -- if nboot > 0 -- ci_perc, ci_stud, boot_inf_frac.
    """
    g = np.asarray(g, dtype=float)
    n = g.size
    rng = np.random.default_rng(seed)
    z = _z(alpha)

    status, rho, beta = solve(g)
    out = {"status": status, "n": int(n), "rho": rho, "beta": beta}

    if status != FIN or not np.isfinite(beta):
        out.update(sigma2=np.nan, se=np.nan, ess=np.nan, sigma2_ternary=None,
                   ci_normal=(np.nan, np.nan), ci_vst=(np.nan, np.nan))
        return out

    w = tilt_weights(g, beta)
    m2 = float(np.mean(w * w))
    sigma2 = m2 - 1.0
    se = float(np.sqrt(max(sigma2, 0.0) / n))
    ess = float(n / m2)                      # Kish ESS; sigma2 == n/ESS - 1
    a = _ternary_scale(g)
    out.update(sigma2=float(sigma2), se=se, ess=ess,
               sigma2_ternary=(float(np.expm1(2.0 * rho)) if a is not None else None))

    out["ci_normal"] = (max(0.0, rho - z * se), rho + z * se)
    # variance-stabilised interval: u_hat +- z/sqrt(n) * (sigma / sqrt(e^{2rho}-1))
    sc = np.sqrt(sigma2 / np.expm1(2.0 * rho)) if rho > 0 else 1.0
    u = float(vst(rho))
    out["ci_vst"] = (float(vst_inv(max(0.0, u - z * sc / np.sqrt(n)))),
                     float(vst_inv(u + z * sc / np.sqrt(n))))

    if nboot > 0:
        rb, nb_inf = _bootstrap(g, nboot, rng, boot_batch)
        fin = np.isfinite(rb)
        out["boot_inf_frac"] = float(1.0 - fin.mean())
        rbf = rb[fin]
        if rbf.size >= 20:
            out["ci_perc"] = (float(np.quantile(rbf, alpha / 2)),
                              float(np.quantile(rbf, 1 - alpha / 2)))
            # studentised (percentile-t) on the VST scale, which is the stable one
            ub = vst(rbf)
            t = (ub - u) * np.sqrt(n)
            out["ci_stud"] = (float(vst_inv(u - np.quantile(t, 1 - alpha / 2) / np.sqrt(n))),
                              float(vst_inv(u - np.quantile(t, alpha / 2) / np.sqrt(n))))
        else:
            out["ci_perc"] = out["ci_stud"] = (np.nan, np.nan)
    return out


def _bootstrap(g, nboot, rng, batch=None):
    n = g.size
    batch = batch or max(1, min(nboot, int(4e7 // max(n, 1)) or 1))
    rho = np.empty(nboot)
    done = 0
    while done < nboot:
        b = min(batch, nboot - done)
        idx = rng.integers(0, n, size=(b, n))
        for j in range(b):
            rho[done + j] = solve(g[idx[j]])[1]
        done += b
    return rho, int(np.sum(~np.isfinite(rho)))


def _z(alpha):
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2))
