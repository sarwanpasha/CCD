import sys,os,json,time,itertools
sys.path.insert(0,'../src')
import numpy as np
from scipy.stats import spearmanr,kendalltau
from scipy.optimize import brentq
exec(open('run10.py').read().split('def fsbound')[0])
T0=time.time()
def lg(*a): print(*a,flush=True)
z=np.load('cache_lc.npz'); DAT={k:z[k] for k in z.files}
TASKS=['arc','winogrande','hellaswag','mmlu']
def rhow(g,w):
  g=np.asarray(g,float); w=np.asarray(w,float); w=w/w.sum()
  mu=float((w*g).sum())
  if mu>=0: return 0.0,0.0
  if g.max()<=0: return float('inf'),float('inf')
  sc=max(1.0,float(np.abs(g).mean()))
  def m12(b):
    zz=b*g; zz=zz-zz.max(); e=w*np.exp(zz); s=e.sum()
    m1=(g*e).sum()/s; return m1,(g*g*e).sum()/s-m1*m1
  hi=1.0
  while m12(hi)[0]<0:
    hi*=2.0
    if hi>1e13: return float('inf'),float('inf')
  lo=0.0; b=0.5*hi
  for _ in range(200):
    m1,m2=m12(b)
    if abs(m1)<1e-15*sc: break
    if m1>0: hi=b
    else: lo=b
    nb=b-m1/m2 if m2>1e-300 else 0.5*(lo+hi)
    b=nb if lo<nb<hi else 0.5*(lo+hi)
    if hi-lo<1e-15*max(1.0,hi): break
  zz=b*g; mz=zz.max(); e=w*np.exp(zz-mz)
  return float(-(mz+np.log(e.sum()))),float(b)
def chi2nn(g):
  g=np.asarray(g,float); mu=g.mean()
  if mu>=0: return 0.0,0.0
  def nu_of(lam):
    f=lambda nu: np.maximum(0.0,1.0+lam*g+nu).mean()-1.0
    a=-1.0-lam*float(g.max()); b=1.0+max(0.0,-lam*float(g.min()))
    return brentq(f,a,b,xtol=1e-15,rtol=8.9e-16,maxiter=300)
  def cons(lam):
    nu=nu_of(lam); w=np.maximum(0.0,1.0+lam*g+nu); return float((w*g).mean())
  hi=1.0
  while cons(hi)<0:
    hi*=2.0
    if hi>1e14: return float('inf'),float('inf')
  lam=brentq(cons,0.0,hi,xtol=1e-15,rtol=8.9e-16,maxiter=300)
  nu=nu_of(lam); w=np.maximum(0.0,1.0+lam*g+nu)
  return float(0.5*((w-1.0)**2).mean()),float(lam)
def alpha_star(g):
  s=np.sort(np.asarray(g,float))[::-1]; cs=np.cumsum(s); k=np.arange(1,len(s)+1)
  ok=(cs/k)>=0.0
  if not ok.any(): return 0.0,0
  kk=int(np.max(np.where(ok)[0])+1); return kk/float(len(s)),kk
def dropcount(g):
  s=np.sort(np.asarray(g,float)); n=len(s); tot=s.sum()
  cs=np.cumsum(s); rem=tot-cs; nn=n-np.arange(1,n+1)
  ok=(nn>0)&(rem/np.maximum(nn,1)>=0.0)
  if not ok.any(): return None,None
  kk=int(np.min(np.where(ok)[0])+1); return kk/float(n),kk
def ifse(g,b):
  g=np.asarray(g,float); zz=b*g; mz=zz.max(); e=np.exp(zz-mz); Z=e.mean()
  phi=1.0-e/(Z*np.exp(0.0))
  return float(phi.std(ddof=1)/np.sqrt(len(g)))
ROWS=[]
for reg in ['acc','raw']:
  for tk in TASKS:
    A=DAT[('C_' if reg=='acc' else 'L_')+tk]; n=A.shape[1]
    for ia,ib in itertools.combinations(range(6),2):
      nm=SHORT[MODELS[ia]]+':'+SHORT[MODELS[ib]]
      g=(A[ib]-A[ia]) if reg=='acc' else (A[ia]-A[ib])
      g=np.asarray(g,float)
      sg=-1.0 if g.mean()>0 else 1.0; h=sg*g
      rk,bk=rho(h); r2=float(h.mean()**2/(2.0*h.var()))
      cn,lam=chi2nn(h); al,ak=alpha_star(h); dp,dk=dropcount(h)
      se=ifse(h,bk) if np.isfinite(bk) else None
      rec={'reg':reg,'task':tk,'pair':nm,'n':n,'rho_kl':float(rk) if np.isfinite(rk) else None,
        'rho_chi2':r2,'rho_chi2_nn':cn,'chi2_lam':lam,'alpha_star':al,'alpha_k':ak,
        'drop_frac':dp,'drop_k':dk,'se_if':se,'skew':float(((h-h.mean())**3).mean()/h.std()**3),
        'ratio_kl_chi2':(float(rk)/r2 if (np.isfinite(rk) and r2>0) else None),
        'nn_over_signed':(cn/r2 if r2>0 else None)}
      ROWS.append(rec)
      lg('%-4s %-11s %-8s rkl=%.9f rchi=%.9f rchinn=%.9f ratio=%.6f alpha=%.6f(%d) drop=%.6f(%s) se=%.3e sk=%+.4f'%(
        reg,nm,tk,float(rk),r2,cn,(float(rk)/r2 if r2>0 else 0),al,ak,(dp if dp else -1),dk,(se if se else 0),rec['skew']))
json.dump({'rows':ROWS},open('out/run17a.json','w'))
lg('PARTA done %.0fs'%(time.time()-T0))
EPS=[0.001,0.005,0.01,0.02,0.05,0.10,0.25,0.50]
SUP=[]
for reg in ['raw','acc']:
  for tx in TASKS:
    AX=DAT[('C_' if reg=='acc' else 'L_')+tx]
    for ia,ib in itertools.combinations(range(6),2):
      nm=SHORT[MODELS[ia]]+':'+SHORT[MODELS[ib]]
      gx=(AX[ib]-AX[ia]) if reg=='acc' else (AX[ia]-AX[ib]); gx=np.asarray(gx,float)
      sg=-1.0 if gx.mean()>0 else 1.0; hx=sg*gx; nX=len(hx)
      r0,b0=rho(hx)
      for ty in TASKS:
        if ty==tx: continue
        AY=DAT[('C_' if reg=='acc' else 'L_')+ty]
        gy=(AY[ib]-AY[ia]) if reg=='acc' else (AY[ia]-AY[ib]); gy=sg*np.asarray(gy,float)
        nY=len(gy); gc=np.concatenate([hx,gy]); cur=[]; efl=None
        ry,by=rho(gy)
        for ep in EPS:
          w=np.concatenate([np.full(nX,(1.0-ep)/nX),np.full(nY,ep/nY)])
          rr,bb=rhow(gc,w); cur.append(float(rr))
          if rr<=0.0 and efl is None: efl=ep
        rec={'reg':reg,'task':tx,'novel':ty,'pair':nm,'nX':nX,'nY':nY,'rho0':float(r0),
          'mean_y':float(gy.mean()),'y_agrees':bool(gy.mean()<0),
          'rho_y':(float(ry) if np.isfinite(ry) else None),'eps':EPS,'curve':cur,'eps_flip':efl,
          'rel_at_05':(cur[EPS.index(0.05)]/float(r0) if r0>0 else None),
          'rel_at_01':(cur[EPS.index(0.01)]/float(r0) if r0>0 else None)}
        SUP.append(rec)
        lg('%-3s %-10s +%-10s %-11s r0=%.7f yagree=%d curve=%s flip=%s'%(reg,tx,ty,nm,float(r0),
          int(gy.mean()<0),' '.join('%.5f'%c for c in cur),efl))
json.dump({'eps':EPS,'rows':SUP},open('out/run17b.json','w'))
lg('PARTC done %.0fs'%(time.time()-T0))
try:
  R16={(r['task'],r['pair']):r for r in json.load(open('out/run16.json'))['rows']}
  rr=[]
  for r in ROWS:
    if r['reg']!='acc': continue
    q=R16.get((r['task'],r['pair']))
    if q and q['boot_sd']>0 and r['se_if']: rr.append(r['se_if']/q['boot_sd'])
  rr=np.array(rr)
  lg('SE_IF / BOOT_SD acc rows n=%d min=%.4f med=%.4f max=%.4f mean=%.4f'%(len(rr),rr.min(),np.median(rr),rr.max(),rr.mean()))
except Exception as e:
  lg('se-compare skipped',repr(e))
def agg(rows,key,f=None):
  v=[r[key] for r in rows if r.get(key) is not None]
  return (len(v),float(np.median(v)) if v else None)
for reg in ['raw','acc']:
  S=[r for r in SUP if r['reg']==reg]
  for ep in EPS:
    i=EPS.index(ep); c=[r['curve'][i] for r in S]; z=sum(1 for x in c if x<=0.0)
    rel=[r['curve'][i]/r['rho0'] for r in S if r['rho0']>0]
    lg('SUPAGG %s eps=%.3f n=%d flipped=%d medrel=%.4f'%(reg,ep,len(S),z,float(np.median(rel))))
  dis=[r for r in S if not r['y_agrees']]
  lg('SUPAGG %s pairs_with_disagreeing_novel=%d of %d'%(reg,len(dis),len(S)))
lg('ALL DONE %.0fs'%(time.time()-T0))
