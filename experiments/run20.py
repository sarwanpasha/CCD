import sys,json,time,itertools
sys.path.insert(0,'../src')
import numpy as np
from scipy.optimize import brentq
exec(open('run10.py').read().split('def fsbound')[0])
T0=time.time()
def lg(*a): print(*a,flush=True)
z=np.load('cache_lc.npz'); DAT={k:z[k] for k in z.files}
SH=['FA','L2','V15','MI','MB','V13']
PR=list(itertools.combinations(range(6),2))
TASKS=['arc','winogrande','hellaswag','mmlu']
def build(C):
  n=C.shape[1]; P=len(PR)
  WIN=np.zeros((n,P)); DEC=np.zeros((n,P))
  for p,(j,k) in enumerate(PR):
    WIN[:,p]=(C[j]>C[k]).astype(float); DEC[:,p]=(C[j]!=C[k]).astype(float)
  return WIN,DEC
def bt(Wn,Wj,th0=None):
  th=np.zeros(6) if th0 is None else np.array(th0,float)
  for it in range(400):
    g=np.zeros(6); H=np.zeros((6,6))
    for p,(j,k) in enumerate(PR):
      if Wn[p]<=0: continue
      s=1.0/(1.0+np.exp(-(th[j]-th[k]))); v=Wn[p]*s*(1.0-s); gv=Wj[p]-Wn[p]*s
      g[j]+=gv; g[k]-=gv; H[j,j]+=v; H[k,k]+=v; H[j,k]-=v; H[k,j]-=v
    if np.abs(g).max()<1e-14: break
    A=H+np.ones((6,6))*(H.trace()/6.0+1e-12)
    th=th+np.linalg.solve(A,g); th-=th.mean()
  return th,H
def hf(w,WIN,DEC,a,b,th0=None):
  th,H=bt(DEC.T@w,WIN.T@w,th0); return float(th[a]-th[b]),th,H
def psif(th,H,WIN,DEC,a,b):
  e=np.zeros(6); e[a]=1.0; e[b]=-1.0; c=np.linalg.pinv(H)@e
  ps=np.zeros(WIN.shape[0])
  for p,(j,k) in enumerate(PR):
    s=1.0/(1.0+np.exp(-(th[j]-th[k])))
    ps+=(WIN[:,p]-DEC[:,p]*s)*(c[j]-c[k])
  return ps
def btrho(WIN,DEC,a,b,outer=6):
  n=WIN.shape[0]; w0=np.full(n,1.0/n)
  h0,th,H=hf(w0,WIN,DEC,a,b)
  ps=psif(th,H,WIN,DEC,a,b); v0=float(ps.var())
  fo=0.5*h0*h0/v0 if v0>0 else None
  best=None
  for it in range(outer):
    sdv=float(ps.std())
    if sdv<=0: break
    d=-ps/sdv
    def hl(l):
      zz=l*d; zz=zz-zz.max(); q=np.exp(zz); q/=q.sum()
      return hf(q,WIN,DEC,a,b,th)[0]
    hi=1.0; ok=True
    while hl(hi)>0:
      hi*=2.0
      if hi>1e7: ok=False; break
    if not ok: break
    ls=brentq(hl,0.0,hi,xtol=1e-13,rtol=8.9e-16,maxiter=300)
    zz=ls*d; zz=zz-zz.max(); q=np.exp(zz); q/=q.sum()
    kl=float((q*np.log(np.maximum(q*n,1e-300))).sum())
    hh,th,H=hf(q,WIN,DEC,a,b,th); ps=psif(th,H,WIN,DEC,a,b)
    if best is None or kl<best[0]: best=(kl,ls,hh,it)
  return {'h0':h0,'firstorder':fo,'rho_bt':(best[0] if best else None),
    'lam':(best[1] if best else None),'h_at_sol':(best[2] if best else None),'iter':(best[3] if best else None)}
OUT=[]
for tk in TASKS:
  C=DAT['C_'+tk]; n=C.shape[1]; WIN,DEC=build(C)
  th0,H0=bt(DEC.T@np.full(n,1.0/n),WIN.T@np.full(n,1.0/n))
  lg('== %s n=%d BT theta=%s'%(tk,n,' '.join('%s=%+.5f'%(SH[i],th0[i]) for i in range(6))))
  for p,(j,k) in enumerate(PR):
    kp=float((C[j]>C[k]).sum()); km=float((C[k]>C[j]).sum())
    dirw=SH[j] if kp>km else SH[k]
    btw=SH[j] if th0[j]>th0[k] else SH[k]
    a,b=(j,k) if th0[j]>th0[k] else (k,j)
    u=(WIN[:,p]-0.5*DEC[:,p]); u=(-u if kp>km else u)
    r2m,b2m=rho(u)
    res=btrho(WIN,DEC,a,b)
    rec={'task':tk,'pair':SH[j]+':'+SH[k],'n':n,'kp':kp,'km':km,'dir_win':dirw,'bt_win':btw,
      'agree':dirw==btw,'rho_2model':float(r2m),'theta_gap':float(abs(th0[j]-th0[k]))}
    rec.update(res)
    OUT.append(rec)
    lg('%-11s %-8s dir=%-4s bt=%-4s%s gap=%.6f rho2=%.9f rhoBT=%s fo=%s ratio=%s h=%.2e'%(
      SH[j]+':'+SH[k],tk,dirw,btw,'' if dirw==btw else ' NONTRANS',rec['theta_gap'],float(r2m),
      ('%.9f'%res['rho_bt']) if res['rho_bt'] else 'NA',('%.9f'%res['firstorder']) if res['firstorder'] else 'NA',
      ('%.5f'%(res['rho_bt']/float(r2m))) if (res['rho_bt'] and r2m>0) else 'NA',
      res['h_at_sol'] if res['h_at_sol'] is not None else 0.0))
json.dump({'rows':OUT},open('out/run20.json','w'))
lg('WROTE out/run20.json %.0fs'%(time.time()-T0))
R16={(r['task'],r['pair']):r for r in json.load(open('out/run16.json'))['rows']}
d=[]
for r in OUT:
  q=R16.get((r['task'],r['pair']))
  if q: d.append(abs(r['rho_2model']-q['rho_cf']))
lg('CHECK 2-model BT radius vs accuracy closed form: n=%d max abs diff=%.3e'%(len(d),max(d)))
for tk in TASKS:
  S=[r for r in OUT if r['task']==tk]
  lg('%s nontrans=%d/15 medratio_BT_over_2model=%s'%(tk,sum(1 for r in S if not r['agree']),
    '%.5f'%np.median([r['rho_bt']/r['rho_2model'] for r in S if r['rho_bt'] and r['rho_2model']>0])))
S=[r for r in OUT if r['rho_bt'] and r['firstorder']]
lg('first-order vs exact BT: n=%d med ratio=%.5f min=%.5f max=%.5f'%(len(S),
  np.median([r['firstorder']/r['rho_bt'] for r in S]),min(r['firstorder']/r['rho_bt'] for r in S),
  max(r['firstorder']/r['rho_bt'] for r in S)))
lg('DONE %.0fs'%(time.time()-T0))
