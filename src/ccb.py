import os
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def K(b,g):
    return logsumexp(b*g)-np.log(g.size)

def Kd(b,g):
    w=np.exp(b*g-logsumexp(b*g)); return float(w@g)

def Kdd(b,g):
    w=np.exp(b*g-logsumexp(b*g)); m=w@g; return float(w@(g-m)**2)

def Kddd(b,g):
    w=np.exp(b*g-logsumexp(b*g)); m=w@g; return float(w@(g-m)**3)

def beta_star(g,tol=1e-14,it=200):
    # solve K'(b)=0 ; requires mean(g)<0<max(g)
    lo,hi=0.0,1.0
    while Kd(hi,g)<0:
        hi*=2.0
        if hi>1e6: break
    b=0.5*hi
    for _ in range(it):
        d=Kd(b,g)
        if abs(d)<tol: break
        h=Kdd(b,g)
        step=d/max(h,1e-300)
        nb=b-step
        if not (lo<nb<hi) or not np.isfinite(nb):
            nb=0.5*(lo+hi)
        if Kd(nb,g)<0: lo=nb
        else: hi=nb
        b=nb
    return b

def ccb(g):
    g=np.asarray(g,dtype=np.float64)
    n=g.size; m=float(g.mean()); s=float(g.std(ddof=0))
    out={'n':n,'mean':m,'sd':s}
    if m>=0 or g.max()<=0:
        out.update(rho=np.nan,beta=np.nan); return out
    b=beta_star(g)
    rho=-K(b,g)
    w=np.exp(b*g-logsumexp(b*g))
    t=abs(m)/s
    sk=float(((g-m)**3).mean()/s**3)
    out.update(beta=b,rho=float(rho),half_t2=0.5*t*t,t=t,skew=sk,
               Kdd=Kdd(b,g),Kddd=Kddd(b,g),
               tau=abs(rho-0.5*t*t)/(0.5*t*t),
               kl_check=float(w@np.log(np.maximum(w*n,1e-300))),
               EQg=float(w@g),w=w)
    return out

def vmin(g,b_star,npts=201):
    bs=np.linspace(0.0,b_star,npts)
    v=np.array([Kdd(b,g) for b in bs])
    i=int(v.argmin()); return float(v[i]),float(bs[i]),float(v[0]),float(v[-1])

def kl_of(w):
    n=w.size; return float(w@np.log(np.maximum(w*n,1e-300)))

def rand_tilt(g,rho,rng,mode='gauss'):
    """random Q with KL(Q||Phat)=rho, direction independent of g"""
    n=g.size
    if mode=='gauss': z=rng.standard_normal(n)
    elif mode=='unif': z=rng.random(n)-0.5
    else: z=rng.standard_normal(n)
    z=z-z.mean()
    lo,hi=0.0,1.0
    def klf(s):
        w=np.exp(s*z-logsumexp(s*z)); return kl_of(w),w
    while klf(hi)[0]<rho:
        hi*=2.0
        if hi>1e8: break
    for _ in range(200):
        mid=0.5*(lo+hi); k,_w=klf(mid)
        if k<rho: lo=mid
        else: hi=mid
    k,w=klf(0.5*(lo+hi))
    return w,k

def kway_rho(G,tol=1e-12):
    """G: (n,k) columns g_j. rho = sup_{lam>=0} -log mean exp(G lam)"""
    n,k=G.shape
    def neg(lam):
        lam=np.maximum(lam,0.0)
        z=G@lam
        val=logsumexp(z)-np.log(n)
        w=np.exp(z-logsumexp(z))
        return val, G.T@w
    f=lambda l:neg(l)[0]
    jac=lambda l:neg(l)[1]
    x0=np.zeros(k)
    res=minimize(f,x0,jac=jac,bounds=[(0,None)]*k,method='L-BFGS-B',
                 options={'ftol':1e-16,'gtol':1e-14,'maxiter':5000})
    lam=np.maximum(res.x,0.0)
    z=G@lam; w=np.exp(z-logsumexp(z))
    return {'rho':float(-res.fun),'lam':lam,'w':w,'EQg':(G.T@w),'ok':bool(res.success)}
